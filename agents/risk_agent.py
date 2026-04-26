from __future__ import annotations
import json
import re

from state import DealState
from agents.base import get_reasoning_model, run_agent
from tools.search_tools import search_web


def _parse_json(text: str) -> dict:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {"raw_response": text}


_SYSTEM = """You are the Risk Manager for ARIA — an M&A intelligence system.

Your mandate: Provide an independent risk assessment that operates outside the bull/bear
debate. You apply portfolio-level and deal-level risk constraints.

You are the last line of defence before the human review gate. Your assessment can:
- Recommend PROCEED (acceptable risk)
- Recommend PROCEED WITH CONDITIONS (acceptable if specific conditions are met)
- Issue a HARD STOP (unacceptable risk — deal should not proceed)

Be objective and thorough. A HARD STOP is serious — only issue it for genuine deal-breakers.
Return structured JSON only.
"""


async def run_risk_manager(state: DealState) -> dict:
    model = get_reasoning_model()

    context = {
        "deal": state["deal_brief"],
        "target": state["target_company"],
        "acquirer": state["acquirer_company"],
        "deal_size": state.get("deal_size_usd"),
        "bull_conviction": (state.get("bull_thesis") or {}).get("conviction_score"),
        "bear_conviction": (state.get("bear_thesis") or {}).get("conviction_score"),
        "key_risks_from_bear": (state.get("bear_thesis") or {}).get("key_risks", [])[:3],
        "deal_breakers_flagged": (state.get("bear_thesis") or {}).get("deal_breakers", []),
        "debate_direction": (
            ((state.get("debate_rounds") or [{}])[-1] or {})
            .get("coordinator_assessment", {})
            .get("debate_direction", "unknown")
        ),
    }

    user_msg = f"""
Conduct an independent risk assessment for this M&A deal.

Deal Context:
{json.dumps(context, indent=2)}

Financial Profile:
{json.dumps((state.get("financial_data") or {}), indent=2)[:800]}

Market and Competitive Context:
{json.dumps((state.get("market_data") or {}), indent=2)[:600]}

Management Assessment:
{json.dumps((state.get("management_assessment") or {}), indent=2)[:600]}

Conduct a thorough risk review and return:
{{
    "risk_categories": {{
        "financial_risk": {{
            "rating": "high | medium | low",
            "key_issues": ["issue 1", "issue 2"],
            "notes": "..."
        }},
        "strategic_risk": {{
            "rating": "high | medium | low",
            "key_issues": ["issue 1", "issue 2"],
            "notes": "..."
        }},
        "regulatory_antitrust_risk": {{
            "rating": "high | medium | low",
            "key_issues": ["issue 1"],
            "estimated_approval_probability_pct": <number>,
            "notes": "..."
        }},
        "integration_risk": {{
            "rating": "high | medium | low",
            "key_issues": ["issue 1", "issue 2"],
            "notes": "..."
        }},
        "key_person_risk": {{
            "rating": "high | medium | low",
            "key_issues": ["issue 1"],
            "notes": "..."
        }},
        "market_timing_risk": {{
            "rating": "high | medium | low",
            "key_issues": ["issue 1"],
            "notes": "..."
        }}
    }},
    "deal_level_risks": [
        {{
            "risk": "risk description",
            "probability": "high | medium | low",
            "impact": "high | medium | low",
            "mitigation": "suggested mitigation",
            "is_deal_breaker": true/false
        }}
    ],
    "hard_stops": ["any conditions that would make this a HARD STOP"],
    "conditions_for_proceed": ["conditions that must be met to proceed"],
    "risk_adjusted_recommendation": {{
        "decision": "PROCEED | PROCEED_WITH_CONDITIONS | HARD_STOP",
        "rationale": "2-3 sentence rationale",
        "confidence": "high | medium | low"
    }},
    "overall_risk_score": <1-10, where 10 is highest risk>,
    "risk_summary": "3-4 sentence risk narrative for the investment committee"
}}
"""

    response, tool_calls, logs = await run_agent(
        model=model,
        tools=[search_web],
        system_prompt=_SYSTEM,
        user_message=user_msg,
        agent_name="RiskManager",
    )

    return {
        "risk_assessment": _parse_json(response),
        "completed_stages": ["risk_manager"],
        "agent_logs": logs,
    }
