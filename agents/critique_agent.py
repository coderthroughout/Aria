from __future__ import annotations
import json
import re

from state import DealState
from agents.base import get_reasoning_model, run_agent
from tools.search_tools import search_web, verify_claim


def _parse_json(text: str) -> dict:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {"raw_response": text}


_SYSTEM = """You are the Critique Agent for ARIA — an M&A intelligence system.

You are the final quality-control layer before the investment memo is written.
You have NO knowledge of which agent produced which output. You read the full analysis
cold and your job is to find:

1. HALLUCINATED FACTS — claims that cannot be verified against public sources
2. INTERNAL CONTRADICTIONS — places where different parts of the analysis disagree
3. UNSUPPORTED ASSERTIONS — strong claims made without evidence
4. LOGICAL ERRORS — conclusions that don't follow from the evidence
5. MISSING CRITICAL ANALYSIS — important questions the analysis failed to address

You use web search to actively verify specific factual claims.
You are a constructive critic — find real problems, not nitpicks.

Be thorough. The investment committee is relying on you to catch errors.
Return structured JSON only.
"""


async def run_critique_agent(state: DealState) -> dict:
    model = get_reasoning_model()

    # Collect all claims to potentially verify
    all_data = {
        "financial": state.get("financial_data", {}),
        "valuation": state.get("valuation_models", {}),
        "market": state.get("market_data", {}),
        "competitive": state.get("competitive_data", {}),
        "technology": state.get("tech_assessment", {}),
        "management": state.get("management_assessment", {}),
        "bull_thesis": state.get("bull_thesis", {}),
        "bear_thesis": state.get("bear_thesis", {}),
        "risk_assessment": state.get("risk_assessment", {}),
    }

    user_msg = f"""
Critically review this full M&A analysis for {state['target_company']}.

The deal: {state['acquirer_company']} acquiring {state['target_company']}
Deal brief: {state['deal_brief']}

FULL ANALYSIS TO CRITIQUE:
{json.dumps(all_data, indent=2)[:6000]}

Your tasks:
1. Select 5 specific factual claims from the analysis and use verify_claim to check them
2. Check for internal contradictions across different sections
3. Identify any unsupported or suspicious assertions
4. Note any critical gaps in the analysis

Return your findings as:
{{
    "verified_claims": [
        {{
            "claim": "exact claim from the analysis",
            "source_section": "financial | market | competitive | bull | bear | risk",
            "verification_result": "confirmed | unverified | contradicted",
            "evidence": "what the verification found",
            "recommendation": "keep | revise | remove"
        }}
    ],
    "internal_contradictions": [
        {{
            "contradiction": "description of the conflict",
            "section_a": "first section making the claim",
            "claim_a": "what it says",
            "section_b": "second section making the conflicting claim",
            "claim_b": "what it says",
            "resolution": "which is likely correct and why"
        }}
    ],
    "unsupported_assertions": [
        {{
            "assertion": "the claim",
            "section": "where it appears",
            "issue": "why it's unsupported",
            "recommendation": "how to address it"
        }}
    ],
    "analysis_gaps": [
        {{
            "gap": "what's missing",
            "importance": "high | medium | low",
            "recommendation": "what should be addressed"
        }}
    ],
    "overall_quality": {{
        "score": <1-10>,
        "primary_strengths": ["strength 1", "strength 2"],
        "primary_weaknesses": ["weakness 1", "weakness 2"],
        "ready_for_report": true/false,
        "blocking_issues": ["any issues that must be fixed before report generation"]
    }},
    "critique_summary": "3-4 sentence overall critique"
}}
"""

    response, tool_calls, logs = await run_agent(
        model=model,
        tools=[search_web, verify_claim],
        system_prompt=_SYSTEM,
        user_message=user_msg,
        agent_name="CritiqueAgent",
    )

    return {
        "critique_results": _parse_json(response),
        "completed_stages": ["critique"],
        "agent_logs": logs,
    }
