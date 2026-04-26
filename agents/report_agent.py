from __future__ import annotations
import json
import re
from datetime import datetime, timezone

from state import DealState
from agents.base import get_reasoning_model, run_agent


def _parse_json(text: str) -> dict:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {"raw_response": text}


_SYSTEM = """You are the Report Agent for ARIA — an M&A intelligence system.

Your mandate: Synthesize the full analysis into a structured, investment-grade memo.
This memo will be read by an investment committee making a multi-billion dollar decision.

Standards:
- Every major claim must be attributed to the agent that produced it
- Include both the bull and bear perspectives on key points
- The recommendation must be clear, direct, and defensible
- Write as a senior M&A banker or strategy consultant would write
- Be precise with numbers — no vague ranges without explanation

The memo structure is fixed — follow it exactly.
Return structured JSON only.
"""


async def run_report_agent(state: DealState) -> dict:
    model = get_reasoning_model()

    # Compress debate for the report
    debate_summary = ""
    if state.get("debate_rounds"):
        last_round = state["debate_rounds"][-1]
        assessment = last_round.get("coordinator_assessment", {})
        debate_summary = json.dumps({
            "bull_strongest_point": assessment.get("strongest_bull_point"),
            "bear_strongest_point": assessment.get("strongest_bear_point"),
            "debate_direction": assessment.get("debate_direction"),
            "final_bull_conviction": assessment.get("updated_bull_conviction"),
            "final_bear_conviction": assessment.get("updated_bear_conviction"),
            "unresolved_questions": assessment.get("key_unresolved_questions", []),
        }, indent=2)

    critique_flags = []
    if state.get("critique_results"):
        blocking = state["critique_results"].get("overall_quality", {}).get("blocking_issues", [])
        critique_flags = blocking

    user_msg = f"""
Generate the final investment memo for this deal.

DEAL:
- Target: {state['target_company']}
- Acquirer: {state['acquirer_company']}
- Deal Size: ${state.get('deal_size_usd', 'TBD'):,} USD
- Brief: {state['deal_brief']}
- Human Notes: {state.get('human_notes', 'None')}

FINANCIAL PROFILE:
{json.dumps((state.get('financial_data') or {}), indent=2)[:1500]}

VALUATION:
{json.dumps((state.get('valuation_models') or {}), indent=2)[:1200]}

MARKET:
{json.dumps((state.get('market_data') or {}), indent=2)[:900]}

COMPETITIVE:
{json.dumps((state.get('competitive_data') or {}), indent=2)[:900]}

TECHNOLOGY:
{json.dumps((state.get('tech_assessment') or {}), indent=2)[:700]}

MANAGEMENT:
{json.dumps((state.get('management_assessment') or {}), indent=2)[:700]}

BULL THESIS:
{json.dumps((state.get('bull_thesis') or {}), indent=2)[:900]}

BEAR THESIS:
{json.dumps((state.get('bear_thesis') or {}), indent=2)[:900]}

DEBATE OUTCOME:
{debate_summary or 'No debate conducted'}

RISK ASSESSMENT:
{json.dumps((state.get('risk_assessment') or {}), indent=2)[:1000]}

CRITIQUE FLAGS:
{json.dumps(critique_flags, indent=2)}

Generate the full investment memo:
{{
    "memo_metadata": {{
        "title": "Investment Committee Memorandum: [Target] / [Acquirer]",
        "date": "{datetime.now(timezone.utc).strftime('%B %d, %Y')}",
        "classification": "CONFIDENTIAL",
        "prepared_by": "ARIA Autonomous Research System"
    }},
    "executive_summary": {{
        "deal_description": "one paragraph deal description",
        "recommendation": "PROCEED | PROCEED_WITH_CONDITIONS | DO_NOT_PROCEED",
        "recommended_price_range": {{"low_usd": <number>, "high_usd": <number>}},
        "key_thesis": "2-3 sentence core investment thesis",
        "primary_upside": "top opportunity",
        "primary_risk": "top risk",
        "urgency": "time-sensitive considerations if any"
    }},
    "financial_analysis": {{
        "revenue_profile": "narrative",
        "profitability": "narrative",
        "balance_sheet": "narrative",
        "key_metrics_table": {{"metric1": "value", "metric2": "value"}},
        "financial_conclusion": "one paragraph"
    }},
    "strategic_analysis": {{
        "market_opportunity": "narrative",
        "competitive_position": "narrative",
        "technology_assets": "narrative",
        "management_quality": "narrative",
        "strategic_fit_with_acquirer": "narrative"
    }},
    "valuation": {{
        "methodology_summary": "narrative",
        "comps_range": {{"low": <number>, "high": <number>}},
        "precedent_transactions_range": {{"low": <number>, "high": <number>}},
        "fundamental_range": {{"low": <number>, "high": <number>}},
        "blended_fair_value_range": {{"low": <number>, "high": <number>}},
        "deal_premium_assessment": "narrative",
        "valuation_conclusion": "one paragraph"
    }},
    "bull_bear_debate": {{
        "bull_case": "2-3 paragraph bull narrative",
        "bear_case": "2-3 paragraph bear narrative",
        "debate_resolution": "which arguments prevailed and why",
        "final_conviction": "bull | bear | balanced"
    }},
    "risk_matrix": {{
        "top_risks": [
            {{"risk": "risk name", "probability": "H|M|L", "impact": "H|M|L", "mitigation": "..."}}
        ],
        "overall_risk_rating": "high | medium | low",
        "risk_conclusion": "one paragraph"
    }},
    "recommendation": {{
        "decision": "PROCEED | PROCEED_WITH_CONDITIONS | DO_NOT_PROCEED",
        "price_recommendation": "negotiation guidance",
        "key_conditions": ["any conditions that must be met"],
        "next_steps": ["immediate next step 1", "step 2", "step 3"],
        "dissenting_views": "any unresolved disagreements between agents"
    }},
    "appendix": {{
        "data_sources": ["list of all data sources used"],
        "agent_attribution": {{
            "financial_analysis": "FinancialAgent + ValuationAgent",
            "market_analysis": "MarketAgent",
            "competitive_analysis": "CompetitiveAgent",
            "technology_assessment": "TechAgent",
            "management_assessment": "ManagementAgent",
            "bull_thesis": "BullAnalyst",
            "bear_thesis": "BearAnalyst",
            "risk_assessment": "RiskManager",
            "quality_control": "CritiqueAgent",
            "synthesis": "ReportAgent"
        }},
        "caveats": [
            "This analysis was generated by an autonomous AI system.",
            "All figures should be independently verified before any investment decision.",
            "This memo does not constitute investment advice."
        ]
    }}
}}
"""

    response, tool_calls, logs = await run_agent(
        model=model,
        tools=[],
        system_prompt=_SYSTEM,
        user_message=user_msg,
        agent_name="ReportAgent",
    )

    return {
        "final_memo": _parse_json(response),
        "current_stage": "completed",
        "completed_stages": ["report"],
        "agent_logs": logs,
    }
