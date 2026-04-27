from __future__ import annotations

import asyncio
import json
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from config import settings


# ── Omium init ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.omium_api_key:
        try:
            import omium
            omium.init(api_key=settings.omium_api_key, project="aria")
            omium.instrument_langgraph()
        except Exception:
            pass  # Omium is observability — never let it break the app
    yield


# ── Session store ──────────────────────────────────────────────────────────────
_sessions: dict[str, dict] = {}


def _new_session(thread_id: str) -> dict:
    session = {
        "queue": asyncio.Queue(),
        "review_future": None,
        "final_memo": None,
        "status": "running",
    }
    _sessions[thread_id] = session
    return session


# ── Models ─────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    target_company: str
    acquirer_company: str
    deal_size_usd: Optional[float] = None
    deal_brief: Optional[str] = None


class ReviewRequest(BaseModel):
    approved: bool
    notes: str = ""


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="ARIA API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "aria-api", "omium": bool(settings.omium_api_key)}


@app.post("/api/analyze")
async def start_analysis(req: AnalyzeRequest):
    thread_id = str(uuid.uuid4())
    session = _new_session(thread_id)
    asyncio.create_task(_run_pipeline(thread_id, req, session))
    return {"thread_id": thread_id}


@app.get("/api/stream/{thread_id}")
async def stream_events(thread_id: str):
    session = _sessions.get(thread_id)
    if not session:
        raise HTTPException(status_code=404, detail="Thread not found")

    queue: asyncio.Queue = session["queue"]

    async def generate():
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=25.0)
            except asyncio.TimeoutError:
                yield 'data: {"type":"heartbeat"}\n\n'
                continue

            yield f"data: {json.dumps(event, default=str)}\n\n"

            if event.get("type") in ("done", "error", "rejected"):
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/review/{thread_id}")
async def submit_review(thread_id: str, req: ReviewRequest):
    session = _sessions.get(thread_id)
    if not session:
        raise HTTPException(status_code=404, detail="Thread not found")

    future: asyncio.Future | None = session.get("review_future")
    if future is None or future.done():
        raise HTTPException(status_code=400, detail="No pending review")

    future.set_result({"approved": req.approved, "notes": req.notes})
    session["status"] = "running"
    return {"ok": True}


@app.get("/api/result/{thread_id}")
async def get_result(thread_id: str):
    session = _sessions.get(thread_id)
    if not session:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"status": session.get("status"), "final_memo": session.get("final_memo")}


# ── Pipeline runner ────────────────────────────────────────────────────────────

async def _run_pipeline(thread_id: str, req: AnalyzeRequest, session: dict):
    queue: asyncio.Queue = session["queue"]

    async def push(event: dict):
        await queue.put(event)

    # Correlate this run with Omium so CLI commands like
    # `omium show <thread_id>` and `omium logs <thread_id>` work.
    if settings.omium_api_key:
        try:
            import omium
            omium.set_execution_id(thread_id)
        except Exception:
            pass

    # ── Explicit Omium pipeline span ───────────────────────────────────────────
    # _patched_astream flushes before the root span is committed to _spans (SDK
    # ordering bug), so we send our own reliable top-level span here instead.
    _o_tracer = None
    _o_cm = None
    _o_span = None
    if settings.omium_api_key:
        try:
            from omium.integrations.tracer import OmiumTracer
            _o_tracer = OmiumTracer(execution_id=thread_id, project="aria")
            _o_cm = _o_tracer.span(
                "aria.pipeline",
                span_type="agent",
                target=req.target_company,
                acquirer=req.acquirer_company,
            )
            _o_span = _o_cm.__enter__()
        except Exception:
            pass

    try:
        from dotenv import load_dotenv
        load_dotenv()

        from main import build_initial_state
        from graph.workflow import build_graph
        from langgraph.types import Command

        deal_brief = req.deal_brief or (
            f"Analyze {req.target_company} as an acquisition target for {req.acquirer_company}"
            + (f" at a deal size of ${req.deal_size_usd:,.0f}" if req.deal_size_usd else "")
            + ". Provide a comprehensive investment-grade due diligence analysis."
        )

        state = build_initial_state(
            deal_brief, req.target_company, req.acquirer_company, req.deal_size_usd
        )
        graph = build_graph()
        config = {"configurable": {"thread_id": thread_id}}

        current_state: dict | None = state
        human_input_value: dict | None = None

        while True:
            if human_input_value is not None:
                events_iter = graph.astream(
                    Command(resume=human_input_value),
                    config=config,
                    stream_mode="updates",
                )
                human_input_value = None
            elif current_state is not None:
                events_iter = graph.astream(current_state, config=config, stream_mode="updates")
                current_state = None
            else:
                break

            interrupted = False

            async for event in events_iter:
                for node_name, node_output in event.items():
                    if node_name == "__interrupt__":
                        interrupted = True
                        interrupt_data = node_output
                        if isinstance(interrupt_data, (list, tuple)) and len(interrupt_data) > 0:
                            review_data = (
                                interrupt_data[0].value
                                if hasattr(interrupt_data[0], "value")
                                else interrupt_data[0]
                            )
                        else:
                            review_data = {}

                        loop = asyncio.get_running_loop()
                        future: asyncio.Future = loop.create_future()
                        session["review_future"] = future
                        session["status"] = "review_required"

                        await push({"type": "review_required", "data": review_data})

                        human_input_value = await future

                        if not human_input_value.get("approved", True):
                            session["status"] = "rejected"
                            await push({"type": "rejected", "message": "Rejected at human review gate"})
                            await push({"type": "done"})
                            return

                    else:
                        logs = node_output.get("agent_logs") or []
                        completed = node_output.get("completed_stages") or []
                        errors = node_output.get("errors") or []

                        await push({
                            "type": "node_complete",
                            "node": node_name,
                            "completed_stages": completed,
                            "logs": [
                                {
                                    "status": lg.get("status"),
                                    "message": lg.get("message"),
                                    "agent": lg.get("agent"),
                                }
                                for lg in logs
                            ],
                            "errors": errors,
                        })

                if interrupted:
                    break

            if not interrupted:
                break

        snapshot = graph.get_state(config)
        final_state = snapshot.values if snapshot else {}
        final_memo = final_state.get("final_memo")

        session["final_memo"] = final_memo
        session["status"] = "done"
        await push({"type": "done", "final_memo": final_memo})

    except Exception as exc:
        if _o_span is not None:
            try:
                _o_span.set_error(exc)
            except Exception:
                pass
        import traceback
        session["status"] = "error"
        await push({
            "type": "error",
            "message": str(exc),
            "detail": traceback.format_exc(),
        })

    finally:
        if _o_cm is not None:
            try:
                _o_cm.__exit__(None, None, None)  # commits span to _spans
            except Exception:
                pass
        if _o_tracer is not None:
            try:
                await _o_tracer.aflush()  # sends immediately, after span is committed
            except Exception:
                pass


# ── Static files (must be last) ────────────────────────────────────────────────

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
