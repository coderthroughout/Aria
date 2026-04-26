from __future__ import annotations
import json
import re

from state import DealState
from agents.base import get_extraction_model, run_agent
from tools.search_tools import search_web

_SYSTEM = """You are the Deal Director for ARIA, an autonomous M&A intelligence system.

Your job is to parse the deal brief and extract all structured parameters needed to
direct the full research team. You are precise, structured, and methodical.

Always return a single valid JSON object — no markdown fences, no prose outside the JSON.
"""


async def run_deal_director(state: DealState) -> dict:
    model = get_extraction_model()

    user_msg = f"""
Parse this M&A deal brief and return a structured JSON object.

DEAL BRIEF:
{state["deal_brief"]}

Return exactly this JSON structure (fill in all fields, use null where unknown):
{{
    "target_company": "Full legal company name",
    "target_ticker": "Exchange:TICKER or null if private",
    "acquirer_company": "Full legal company name",
    "acquirer_ticker": "Exchange:TICKER or null",
    "deal_size_usd": <number in USD or null>,
    "deal_type": "acquisition | merger | strategic_investment | majority_stake",
    "strategic_rationale": "1-2 sentence summary of why this deal makes sense",
    "research_priorities": ["list of 4-6 key areas this analysis must focus on"],
    "upfront_risk_flags": ["any obvious regulatory, competitive, or financial risks"],
    "comparable_deals": ["1-3 recent comparable M&A transactions in same space"],
    "analysis_scope": "Brief description of what the full analysis should cover"
}}
"""

    response, tool_calls, logs = await run_agent(
        model=model,
        tools=[search_web],
        system_prompt=_SYSTEM,
        user_message=user_msg,
        agent_name="DealDirector",
    )

    parsed = _parse_json(response)

    return {
        "current_stage": "data_gathering",
        "completed_stages": ["decompose"],
        "target_company": parsed.get("target_company") or state.get("target_company", ""),
        "acquirer_company": parsed.get("acquirer_company") or state.get("acquirer_company", ""),
        "deal_size_usd": parsed.get("deal_size_usd") or state.get("deal_size_usd"),
        "agent_logs": logs,
    }


def _parse_json(text: str) -> dict:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {}
