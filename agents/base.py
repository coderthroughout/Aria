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
# Azure OpenAI has high TPM limits so we use a small gap (1s).
# ---------------------------------------------------------------------------
_PACER_LOCK: asyncio.Lock | None = None
_LAST_CALL_AT: float = 0.0
_MIN_CALL_GAP: float = 1.0   # seconds between calls


def _pacer() -> asyncio.Lock:
    global _PACER_LOCK
    if _PACER_LOCK is None:
        _PACER_LOCK = asyncio.Lock()
    return _PACER_LOCK


async def _paced_invoke(bound_model: Any, messages: list) -> AIMessage:
    """Invoke the model, spacing calls at least _MIN_CALL_GAP seconds apart."""
    global _LAST_CALL_AT
    async with _pacer():
        gap = time.monotonic() - _LAST_CALL_AT
        if gap < _MIN_CALL_GAP:
            await asyncio.sleep(_MIN_CALL_GAP - gap)
        _LAST_CALL_AT = time.monotonic()
    # Lock released — actual API call happens outside the lock so other agents
    # can schedule their next call while this one is in-flight.
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


async def _invoke_with_retry(
    bound_model: Any,
    messages: list,
    log_fn,
    max_retries: int = 5,
) -> AIMessage:
    """Paced + retrying LLM call. Handles 429s with exponential back-off."""
    for attempt in range(max_retries):
        try:
            return await _paced_invoke(bound_model, messages)
        except Exception as exc:
            if _is_rate_limit(exc) and attempt < max_retries - 1:
                wait = min(15 * 2 ** attempt, 120)  # 15 → 30 → 60 → 120s
                log_fn("warning", f"Rate limit — waiting {wait}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError("Rate limit: max retries exceeded")  # unreachable


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
    Core agent loop with call-pacing and rate-limit retry.
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

    for iteration in range(max_iterations):
        response: AIMessage = await _invoke_with_retry(bound_model, messages, _log)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            _log("completed", f"Finished in {iteration + 1} LLM call(s)")
            content = response.content
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            return content, tool_calls_made, logs

        for tc in response.tool_calls:
            tool_name: str = tc["name"]
            tool_args: dict = tc["args"]
            tool_id: str = tc["id"]

            _log("tool_call", f"→ {tool_name}({list(tool_args.keys())})")

            if tool_name not in tool_map:
                # Model called a phantom tool — treat args as structured response
                _log("completed", f"Captured output via phantom tool '{tool_name}'")
                return json.dumps(tool_args), tool_calls_made, logs

            try:
                result = await tool_map[tool_name].ainvoke(tool_args)
                tool_calls_made.append({"tool": tool_name, "args": tool_args})
            except Exception as exc:
                result = f"Tool execution error: {exc}"
                _log("warning", f"{tool_name} raised: {exc}")

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
