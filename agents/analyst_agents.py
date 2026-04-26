from __future__ import annotations
import json
import re
from state import DealState
from agents.base import get_reasoning_model, run_agent
from tools.search_tools import search_web, search_company_news


def _parse_json(text: str) -> dict:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {"raw_response": text}


def _summarize_state(state: DealState) -> str:
    """Compact summary of all gathered data for analyst agents to read."""
    sections = []
    if state.get("financial_data"):
        sections.append(f"FINANCIAL DATA:\n{json.dumps(state['financial_data'], indent=2)[:1500]}")
    if state.get("valuation_models"):
        sections.append(f"VALUATION MODELS:\n{json.dumps(state['valuation_models'], indent=2)[:1200]}")
    if state.get("market_data"):
        sections.append(f"MARKET DATA:\n{json.dumps(state['market_data'], indent=2)[:1000]}")
    if state.get("competitive_data"):
        sections.append(f"COMPETITIVE DATA:\n{json.dumps(state['competitive_data'], indent=2)[:1000]}")
    if state.get("tech_assessment"):
        sections.append(f"TECHNOLOGY:\n{json.dumps(state['tech_assessment'], indent=2)[:800]}")
    if state.get("management_assessment"):
        sections.append(f"MANAGEMENT:\n{json.dumps(state['management_assessment'], indent=2)[:800]}")
    return "\n\n".join(sections)


# ── Bull Analyst ───────────────────────────────────────────────────────────────

_BULL_SYSTEM = """You are the Bull Analyst for ARIA — an M&A research system.

Your mandate is to construct the strongest possible bullish investment thesis for this deal.
You must:
- Find the three most compelling growth catalysts
- Identify the most underappreciated competitive advantages
- Build the most optimistic but still credible valuation case
- Score your own conviction honestly from 1-10

You are arguing the bull case. You must be rigorous, not cheerleading.
Your thesis will be challenged by a Bear Analyst — prepare for rebuttals.

Return structured JSON only.
"""

_BEAR_SYSTEM = """You are the Bear Analyst for ARIA — an M&A research system.

Your mandate is to construct the strongest possible bearish / skeptical thesis about this deal.
You must:
- Identify the three biggest risks and headwinds
- Find the most overestimated assumptions in any optimistic case
- Build the most pessimistic but still credible valuation case
- Score your conviction honestly from 1-10

You are the skeptic. Your job is to protect capital, not be negative for its own sake.
Your thesis will be challenged by a Bull Analyst — prepare for rebuttals.

Return structured JSON only.
"""


async def run_bull_analyst(state: DealState) -> dict:
    model = get_reasoning_model()
    data_summary = _summarize_state(state)

    user_msg = f"""
Deal: {state['acquirer_company']} acquiring {state['target_company']}
Deal Size: ${state.get('deal_size_usd', 'TBD'):,} USD
Brief: {state['deal_brief']}

All gathered research data:
{data_summary}

Build the bull thesis and return:
{{
    "thesis_title": "Compelling one-line bull thesis",
    "conviction_score": <1-10>,
    "investment_highlights": [
        {{"point": "key highlight 1", "evidence": "supporting data/rationale", "strength": "high|medium"}},
        {{"point": "key highlight 2", "evidence": "...", "strength": "..."}},
        {{"point": "key highlight 3", "evidence": "...", "strength": "..."}}
    ],
    "growth_catalysts": [
        {{"catalyst": "catalyst 1", "timeline": "near|medium|long-term", "magnitude": "description"}},
        {{"catalyst": "catalyst 2", "timeline": "...", "magnitude": "..."}}
    ],
    "underappreciated_advantages": ["advantage 1", "advantage 2"],
    "bull_valuation": {{
        "methodology": "how you arrived at this",
        "target_ev_usd": <number>,
        "upside_from_deal_price_pct": <number or null>,
        "key_assumptions": ["assumption 1", "assumption 2", "assumption 3"]
    }},
    "bear_case_rebuttals": {{
        "anticipated_bear_arguments": ["expected criticism 1", "criticism 2"],
        "pre_rebuttals": ["counter-argument 1", "counter-argument 2"]
    }},
    "bull_summary": "2-3 sentence compelling bull narrative"
}}
"""

    response, tool_calls, logs = await run_agent(
        model=model,
        tools=[search_web],
        system_prompt=_BULL_SYSTEM,
        user_message=user_msg,
        agent_name="BullAnalyst",
    )

    return {
        "bull_thesis": _parse_json(response),
        "completed_stages": ["bull_analyst"],
        "agent_logs": logs,
    }


async def run_bear_analyst(state: DealState) -> dict:
    model = get_reasoning_model()
    data_summary = _summarize_state(state)

    user_msg = f"""
Deal: {state['acquirer_company']} acquiring {state['target_company']}
Deal Size: ${state.get('deal_size_usd', 'TBD'):,} USD
Brief: {state['deal_brief']}

All gathered research data:
{data_summary}

Build the bear thesis and return:
{{
    "thesis_title": "Compelling one-line bear thesis",
    "conviction_score": <1-10>,
    "key_risks": [
        {{"risk": "risk description 1", "probability": "high|medium|low", "impact": "high|medium|low", "evidence": "..."}},
        {{"risk": "risk description 2", "probability": "...", "impact": "...", "evidence": "..."}},
        {{"risk": "risk description 3", "probability": "...", "impact": "...", "evidence": "..."}}
    ],
    "overestimated_assumptions": [
        {{"assumption": "what bulls are overestimating", "reality": "why it's wrong", "evidence": "..."}}
    ],
    "bear_valuation": {{
        "methodology": "how you arrived at this",
        "target_ev_usd": <number>,
        "downside_from_deal_price_pct": <number or null>,
        "key_assumptions": ["assumption 1", "assumption 2", "assumption 3"]
    }},
    "bull_case_rebuttals": {{
        "anticipated_bull_arguments": ["expected bull point 1", "bull point 2"],
        "pre_rebuttals": ["counter-argument 1", "counter-argument 2"]
    }},
    "deal_breakers": ["any absolute deal-breaker conditions"],
    "bear_summary": "2-3 sentence compelling bear narrative"
}}
"""

    response, tool_calls, logs = await run_agent(
        model=model,
        tools=[search_web],
        system_prompt=_BEAR_SYSTEM,
        user_message=user_msg,
        agent_name="BearAnalyst",
    )

    return {
        "bear_thesis": _parse_json(response),
        "completed_stages": ["bear_analyst"],
        "agent_logs": logs,
    }


# ── Debate Coordinator ─────────────────────────────────────────────────────────

_DEBATE_SYSTEM = """You are the Debate Coordinator for ARIA.

You manage the structured adversarial debate between the Bull Analyst and Bear Analyst.
For each debate round:
1. Bull specifically rebuts the bear's strongest point with evidence
2. Bear specifically rebuts the bull's strongest point with evidence
3. You identify which arguments are strongest and which are weakest

You are neutral — your job is to surface the truth, not pick a side.
Return structured JSON only.
"""


async def run_debate(state: DealState) -> dict:
    model = get_reasoning_model()
    bull = state.get("bull_thesis", {})
    bear = state.get("bear_thesis", {})
    existing_rounds = state.get("debate_rounds") or []

    round_number = len(existing_rounds) + 1
    prev_rounds_section = (
        "PREVIOUS ROUNDS:\n" + json.dumps(existing_rounds, indent=2)[:1000]
        if existing_rounds else ""
    )

    user_msg = f"""
Conduct Debate Round {round_number} between Bull and Bear analysts.

BULL THESIS:
{json.dumps(bull, indent=2)[:1500]}

BEAR THESIS:
{json.dumps(bear, indent=2)[:1500]}

{prev_rounds_section}

Facilitate the debate and return:
{{
    "round": {round_number},
    "bull_rebuttal": {{
        "targeting_bear_point": "the specific bear argument being addressed",
        "rebuttal": "bull's counter-argument",
        "new_evidence_introduced": "any new data or logic introduced",
        "strength": "strong | moderate | weak"
    }},
    "bear_rebuttal": {{
        "targeting_bull_point": "the specific bull argument being addressed",
        "rebuttal": "bear's counter-argument",
        "new_evidence_introduced": "any new data or logic introduced",
        "strength": "strong | moderate | weak"
    }},
    "coordinator_assessment": {{
        "strongest_bull_point": "the most compelling bull argument overall",
        "strongest_bear_point": "the most compelling bear argument overall",
        "key_unresolved_questions": ["question that needs more data/analysis"],
        "debate_direction": "bull_winning | bear_winning | too_close_to_call",
        "updated_bull_conviction": <1-10>,
        "updated_bear_conviction": <1-10>
    }},
    "round_summary": "2-3 sentence summary of what this round established"
}}
"""

    response, tool_calls, logs = await run_agent(
        model=model,
        tools=[search_web],
        system_prompt=_DEBATE_SYSTEM,
        user_message=user_msg,
        agent_name=f"DebateCoordinator-R{round_number}",
    )

    new_round = _parse_json(response)
    all_rounds = existing_rounds + [new_round]

    return {
        "debate_rounds": all_rounds,
        "completed_stages": [f"debate_round_{round_number}"],
        "agent_logs": logs,
    }
