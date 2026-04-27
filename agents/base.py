from __future__ import annotations
import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any

from langchain_openai import AzureChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from config import settings
from state import AgentLog

# ---------------------------------------------------------------------------
# Global call pacer — enforces a minimum gap between any two LLM calls so we
# never burst past rate limits regardless of how many agents run in parallel.
# ---------------------------------------------------------------------------
_PACER_LOCK: asyncio.Lock | None = None
_LAST_CALL_AT: float = 0.0
_MIN_CALL_GAP: float = 1.0   # seconds between calls

# GPT-4o-mini pricing (USD per token)
_COST_PER_INPUT_TOKEN  = 0.15 / 1_000_000
_COST_PER_OUTPUT_TOKEN = 0.60 / 1_000_000


def _pacer() -> asyncio.Lock:
    global _PACER_LOCK
    if _PACER_LOCK is None:
        _PACER_LOCK = asyncio.Lock()
    return _PACER_LOCK


async def _paced_invoke(bound_model: Any, messages: list) -> AIMessage:
    global _LAST_CALL_AT
    async with _pacer():
        gap = time.monotonic() - _LAST_CALL_AT
        if gap < _MIN_CALL_GAP:
            await asyncio.sleep(_MIN_CALL_GAP - gap)
        _LAST_CALL_AT = time.monotonic()
    return await bound_model.ainvoke(messages)


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    return (
        "429" in msg
        or "rate_limit" in msg
        or "rate limit" in msg
        or "too many requests" in msg
        or "ratelimit" in name
    )


def _extract_token_counts(response: AIMessage) -> tuple[int, int]:
    """Pull input/output token counts from Azure OpenAI response metadata."""
    meta = getattr(response, "response_metadata", {}) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    input_tokens  = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
    # Fallback: newer LangChain usage_metadata attribute
    if not input_tokens:
        um = getattr(response, "usage_metadata", None)
        if um:
            input_tokens  = getattr(um, "input_tokens", 0)
            output_tokens = getattr(um, "output_tokens", 0)
    return int(input_tokens), int(output_tokens)


async def _invoke_with_retry(
    bound_model: Any,
    messages: list,
    log_fn,
    omium_span=None,
    max_retries: int = 5,
) -> AIMessage:
    """Paced + retrying LLM call. Handles 429s with exponential back-off."""
    for attempt in range(max_retries):
        try:
            response = await _paced_invoke(bound_model, messages)

            # Record token counts + cost into the Omium span if one is active
            if omium_span is not None:
                try:
                    input_tok, output_tok = _extract_token_counts(response)
                    cost = (input_tok * _COST_PER_INPUT_TOKEN
                            + output_tok * _COST_PER_OUTPUT_TOKEN)
                    omium_span.set_token_counts(
                        input_tokens=input_tok,
                        output_tokens=output_tok,
                    )
                    omium_span.add_event("llm_call", {
                        "input_tokens": input_tok,
                        "output_tokens": output_tok,
                        "cost_usd": round(cost, 6),
                        "attempt": attempt + 1,
                    })
                except Exception:
                    pass

            return response

        except Exception as exc:
            if _is_rate_limit(exc) and attempt < max_retries - 1:
                wait = min(15 * 2 ** attempt, 120)
                log_fn("warning", f"Rate limit — waiting {wait}s (attempt {attempt + 1}/{max_retries})")
                if omium_span is not None:
                    try:
                        omium_span.add_event("rate_limit_retry", {"wait_s": wait, "attempt": attempt + 1})
                    except Exception:
                        pass
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError("Rate limit: max retries exceeded")


def _build_model(deployment_name: str, max_tokens: int = 8192) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=deployment_name,
        azure_endpoint=settings.azure_openai_endpoint,
        openai_api_version=settings.azure_openai_api_version,
        api_key=settings.azure_openai_api_key.get_secret_value(),
        max_tokens=max_tokens,
    )


def get_reasoning_model() -> AzureChatOpenAI:
    return _build_model(settings.reasoning_model)


def get_extraction_model() -> AzureChatOpenAI:
    return _build_model(settings.extraction_model)


async def run_agent(
    model: BaseChatModel,
    tools: list,
    system_prompt: str,
    user_message: str,
    agent_name: str,
    max_iterations: int = 8,
) -> tuple[str, list[dict], list[AgentLog]]:
    """
    Core agent loop with call-pacing, rate-limit retry, and Omium tracing.
    Returns (final_text, tool_calls_made, logs).
    """
    logs: list[AgentLog] = []
    tool_calls_made: list[dict] = []

    def _log(status: str, message: str) -> None:
        logs.append(AgentLog(
            agent=agent_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status,
            message=message,
        ))

    _log("started", "Beginning analysis")

    bound_model = model.bind_tools(tools) if tools else model
    tool_map = {t.name: t for t in tools}

    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    # Open an Omium agent span for the full duration of this agent's work.
    # All LLM calls, tool calls, and token counts nest inside this span.
    omium_span = None
    _omium_cm = None
    _omium_tracer = None
    try:
        if settings.omium_api_key:
            from omium.integrations.tracer import OmiumTracer
            from omium.integrations.core import get_current_config as _get_omium_cfg
            _cfg = _get_omium_cfg()
            _omium_tracer = OmiumTracer(
                execution_id=(_cfg.execution_id if _cfg else None),
                project="ARIA",
            )
            _omium_cm = _omium_tracer.span(agent_name, span_type="agent")
            omium_span = _omium_cm.__enter__()
    except Exception:
        pass

    try:
        for iteration in range(max_iterations):
            response: AIMessage = await _invoke_with_retry(
                bound_model, messages, _log, omium_span=omium_span
            )
            messages.append(response)

            if not getattr(response, "tool_calls", None):
                _log("completed", f"Finished in {iteration + 1} LLM call(s)")
                content = response.content
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    )
                if omium_span is not None:
                    try:
                        omium_span.set_output({"result_length": len(content), "iterations": iteration + 1})
                    except Exception:
                        pass
                return content, tool_calls_made, logs

            for tc in response.tool_calls:
                tool_name: str = tc["name"]
                tool_args: dict = tc["args"]
                tool_id: str = tc["id"]

                _log("tool_call", f"-> {tool_name}({list(tool_args.keys())})")

                if omium_span is not None:
                    try:
                        omium_span.add_event("tool_call", {"tool": tool_name, "args_keys": list(tool_args.keys())})
                    except Exception:
                        pass

                if tool_name not in tool_map:
                    _log("completed", f"Captured output via phantom tool '{tool_name}'")
                    return json.dumps(tool_args), tool_calls_made, logs

                try:
                    result = await tool_map[tool_name].ainvoke(tool_args)
                    tool_calls_made.append({"tool": tool_name, "args": tool_args})
                except Exception as exc:
                    result = f"Tool execution error: {exc}"
                    _log("warning", f"{tool_name} raised: {exc}")
                    if omium_span is not None:
                        try:
                            omium_span.add_event("tool_error", {"tool": tool_name, "error": str(exc)})
                        except Exception:
                            pass

                messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

        _log("warning", f"Reached max iterations ({max_iterations})")
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                content = msg.content
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    )
                return content, tool_calls_made, logs

        return "", tool_calls_made, logs

    except Exception as exc:
        if omium_span is not None:
            try:
                omium_span.set_error(exc)
            except Exception:
                pass
        raise

    finally:
        if _omium_cm is not None:
            try:
                _omium_cm.__exit__(None, None, None)
            except Exception:
                pass
        if _omium_tracer is not None:
            try:
                await _omium_tracer.aflush()
            except Exception:
                pass
