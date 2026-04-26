from __future__ import annotations
import json
import re

from state import DealState
from agents.base import get_extraction_model, run_agent
from tools.financial_tools import get_company_financials, get_stock_price_history, get_peer_companies
from tools.sec_tools import search_sec_filings, get_company_facts
from tools.search_tools import search_web, search_company_news


# ── Shared JSON parser ─────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {"raw_response": text}


# ── Financial Agent ────────────────────────────────────────────────────────────

_FINANCIAL_SYSTEM = """You are the Financial Agent for ARIA, a senior financial analyst
specializing in M&A due diligence.

Your mandate: Gather and analyze all available financial data for the target company.
Focus on: revenue trends, profitability, cash generation, balance sheet strength,
and key valuation metrics. Use real data from financial tools and SEC filings.

Return your analysis as a structured JSON object. Be precise with numbers.
Always cite the source (yfinance, SEC EDGAR, etc.) for key figures.
"""


async def run_financial_agent(state: DealState) -> dict:
    model = get_extraction_model()
    target = state["target_company"]
    ticker = _extract_ticker(target)

    user_msg = f"""
Conduct a comprehensive financial analysis of {target} as an M&A target.
{"Ticker: " + ticker if ticker else "Note: This may be a private company — use web search for financials."}
Deal context: {state.get("deal_brief", "")}

Use the available tools to gather real data, then return a JSON object:
{{
    "company": "{target}",
    "data_sources": ["list of sources used"],
    "revenue_profile": {{
        "latest_annual_revenue": <number>,
        "revenue_3yr_cagr_pct": <number or null>,
        "revenue_breakdown": "description if available"
    }},
    "profitability": {{
        "gross_margin_pct": <number>,
        "ebitda_margin_pct": <number>,
        "net_margin_pct": <number>,
        "fcf_margin_pct": <number or null>
    }},
    "balance_sheet": {{
        "total_cash_usd": <number>,
        "total_debt_usd": <number>,
        "net_debt_usd": <number>,
        "debt_to_ebitda": <number or null>
    }},
    "scale": {{
        "market_cap_usd": <number or null>,
        "enterprise_value_usd": <number or null>,
        "employees": <number or null>
    }},
    "valuation_multiples": {{
        "ev_to_revenue": <number or null>,
        "ev_to_ebitda": <number or null>,
        "pe_ratio": <number or null>
    }},
    "key_financial_risks": ["list of 3-5 financial risk observations"],
    "financial_summary": "3-4 sentence narrative summary of the financial profile"
}}
"""

    response, tool_calls, logs = await run_agent(
        model=model,
        tools=[get_company_financials, get_stock_price_history, get_company_facts,
               search_sec_filings, search_web],
        system_prompt=_FINANCIAL_SYSTEM,
        user_message=user_msg,
        agent_name="FinancialAgent",
    )

    return {
        "financial_data": _parse_json(response),
        "completed_stages": ["financial_agent"],
        "agent_logs": logs,
    }


# ── Valuation Agent ────────────────────────────────────────────────────────────

_VALUATION_SYSTEM = """You are the Valuation Agent for ARIA, a senior M&A valuation specialist.

Your mandate: Build three independent valuation frameworks for the target company:
1. Comparable Company Analysis (trading multiples from peer set)
2. Precedent Transaction Analysis (M&A deal multiples)
3. DCF / Fundamental Value Assessment

Use the financial data already gathered plus your own research. Be quantitative and rigorous.
Return structured JSON only.
"""


async def run_valuation_agent(state: DealState) -> dict:
    model = get_extraction_model()
    target = state["target_company"]
    fin = state.get("financial_data", {})

    user_msg = f"""
Build a comprehensive valuation analysis for {target}.

Financial data already gathered:
{json.dumps(fin, indent=2)[:2000]}

Deal size context: ${state.get('deal_size_usd', 'unknown'):,} USD

Research comparable companies and recent transactions, then return:
{{
    "comparable_companies": {{
        "peer_set": ["company1 (ticker)", "company2 (ticker)", ...],
        "median_ev_to_revenue": <number>,
        "median_ev_to_ebitda": <number or null>,
        "implied_ev_low": <number>,
        "implied_ev_high": <number>,
        "notes": "methodology notes"
    }},
    "precedent_transactions": {{
        "transactions": [
            {{"target": "name", "acquirer": "name", "year": 2023, "ev_to_revenue": 5.2, "ev_to_ebitda": null}}
        ],
        "median_ev_to_revenue": <number>,
        "implied_ev": <number>,
        "notes": "methodology notes"
    }},
    "fundamental_value": {{
        "method": "DCF / normalized earnings / asset-based",
        "key_assumptions": "growth rate, discount rate, terminal value",
        "implied_ev_low": <number>,
        "implied_ev_high": <number>,
        "notes": "methodology notes"
    }},
    "valuation_summary": {{
        "blended_ev_low": <number>,
        "blended_ev_mid": <number>,
        "blended_ev_high": <number>,
        "proposed_deal_premium_pct": <number or null>,
        "fair_value_assessment": "undervalued | fairly valued | overvalued",
        "narrative": "2-3 sentence valuation conclusion"
    }}
}}
"""

    response, tool_calls, logs = await run_agent(
        model=model,
        tools=[get_peer_companies, search_web, get_company_financials],
        system_prompt=_VALUATION_SYSTEM,
        user_message=user_msg,
        agent_name="ValuationAgent",
    )

    return {
        "valuation_models": _parse_json(response),
        "completed_stages": ["valuation_agent"],
        "agent_logs": logs,
    }


# ── Market Agent ───────────────────────────────────────────────────────────────

_MARKET_SYSTEM = """You are the Market Intelligence Agent for ARIA.

Your mandate: Size and characterize the market in which the target company operates.
Cover: TAM/SAM/SOM, growth rates, key growth drivers, market structure, and
regulatory/macro environment. Use real web research for current figures.

Return structured JSON only.
"""


async def run_market_agent(state: DealState) -> dict:
    model = get_extraction_model()
    target = state["target_company"]

    user_msg = f"""
Conduct a market intelligence analysis for the space in which {target} operates.
Context: {state.get("deal_brief", "")}

Research the market thoroughly and return:
{{
    "market_definition": "precise definition of the primary market",
    "tam": {{
        "value_usd_billions": <number>,
        "year": <year of estimate>,
        "source": "source name",
        "methodology": "top-down | bottom-up | consensus"
    }},
    "sam": {{
        "value_usd_billions": <number>,
        "rationale": "why this subset of TAM"
    }},
    "market_growth": {{
        "historical_cagr_pct": <number>,
        "projected_cagr_pct": <number>,
        "projection_period": "e.g. 2024-2030",
        "growth_drivers": ["driver 1", "driver 2", "driver 3"]
    }},
    "market_structure": {{
        "type": "fragmented | concentrated | duopoly | monopoly | oligopoly",
        "hhi_estimate": "low | medium | high",
        "top_players_market_share": {{"player1": "pct", "player2": "pct"}}
    }},
    "target_market_position": {{
        "estimated_market_share_pct": <number or null>,
        "positioning": "leader | challenger | niche | fast-growing entrant"
    }},
    "regulatory_environment": {{
        "key_regulations": ["list of relevant regulations/bodies"],
        "regulatory_risk": "low | medium | high",
        "notes": "any pending regulatory changes"
    }},
    "market_summary": "3-4 sentence market narrative"
}}
"""

    response, tool_calls, logs = await run_agent(
        model=model,
        tools=[search_web, search_company_news],
        system_prompt=_MARKET_SYSTEM,
        user_message=user_msg,
        agent_name="MarketAgent",
    )

    return {
        "market_data": _parse_json(response),
        "completed_stages": ["market_agent"],
        "agent_logs": logs,
    }


# ── Competitive Agent ──────────────────────────────────────────────────────────

_COMPETITIVE_SYSTEM = """You are the Competitive Intelligence Agent for ARIA.

Your mandate: Map the competitive landscape for the target company and evaluate
the strength of its competitive moat. Use Porter's Five Forces and moat analysis frameworks.

Return structured JSON only.
"""


async def run_competitive_agent(state: DealState) -> dict:
    model = get_extraction_model()
    target = state["target_company"]

    user_msg = f"""
Conduct a competitive analysis for {target}.
Context: {state.get("deal_brief", "")}

Research competitors, moat factors, and competitive dynamics. Return:
{{
    "direct_competitors": [
        {{"name": "Competitor A", "ticker": "COMP", "market_cap_usd": null, "key_difference": "..."}}
    ],
    "indirect_competitors": ["name1", "name2"],
    "competitive_moat": {{
        "overall_moat_rating": "wide | narrow | none",
        "network_effects": {{"present": true/false, "description": "..."}},
        "switching_costs": {{"strength": "high|medium|low", "description": "..."}},
        "cost_advantages": {{"present": true/false, "description": "..."}},
        "intangible_assets": {{"description": "brands, patents, licenses, etc."}},
        "efficient_scale": {{"present": true/false, "description": "..."}}
    }},
    "porters_five_forces": {{
        "competitive_rivalry": "high | medium | low",
        "threat_of_new_entrants": "high | medium | low",
        "bargaining_power_buyers": "high | medium | low",
        "bargaining_power_suppliers": "high | medium | low",
        "threat_of_substitutes": "high | medium | low"
    }},
    "competitive_advantages": ["advantage 1", "advantage 2", "advantage 3"],
    "competitive_vulnerabilities": ["vulnerability 1", "vulnerability 2"],
    "acquirer_synergies": {{
        "revenue_synergies": ["potential synergy 1", "synergy 2"],
        "cost_synergies": ["potential cost saving 1", "saving 2"],
        "strategic_synergies": ["strategic benefit 1", "benefit 2"]
    }},
    "competitive_summary": "3-4 sentence competitive position narrative"
}}
"""

    response, tool_calls, logs = await run_agent(
        model=model,
        tools=[search_web, search_company_news],
        system_prompt=_COMPETITIVE_SYSTEM,
        user_message=user_msg,
        agent_name="CompetitiveAgent",
    )

    return {
        "competitive_data": _parse_json(response),
        "completed_stages": ["competitive_agent"],
        "agent_logs": logs,
    }


# ── Technology Agent ───────────────────────────────────────────────────────────

_TECH_SYSTEM = """You are the Technology Assessment Agent for ARIA.

Your mandate: Evaluate the target company's technology assets, IP position,
engineering capability, and technical risks. This covers: tech stack, patents,
R&D investment, open source activity, engineering team depth, and technical debt signals.

Return structured JSON only.
"""


async def run_tech_agent(state: DealState) -> dict:
    model = get_extraction_model()
    target = state["target_company"]

    user_msg = f"""
Conduct a technology assessment for {target}.
Context: {state.get("deal_brief", "")}

Research technology assets, IP, and engineering capability. Return:
{{
    "technology_overview": "brief description of core technology",
    "tech_stack": {{
        "primary_languages": ["lang1", "lang2"],
        "key_infrastructure": "cloud provider, databases, frameworks",
        "proprietary_vs_oss": "ratio/description"
    }},
    "ip_portfolio": {{
        "patents_held": <number or "unknown">,
        "key_patent_areas": ["area1", "area2"],
        "ip_strength": "strong | moderate | weak | unknown"
    }},
    "rd_investment": {{
        "rd_as_pct_revenue": <number or null>,
        "rd_trend": "increasing | stable | decreasing | unknown",
        "focus_areas": ["AI/ML", "infrastructure", etc.]
    }},
    "engineering_team": {{
        "estimated_size": <number or "unknown">,
        "key_person_risk": "high | medium | low",
        "talent_quality_signals": "description from GitHub, LinkedIn, publications"
    }},
    "technical_risks": [
        {{"risk": "description", "severity": "high|medium|low", "mitigation": "..."}}
    ],
    "technical_advantages": ["advantage 1", "advantage 2"],
    "technology_summary": "3-4 sentence technology narrative"
}}
"""

    response, tool_calls, logs = await run_agent(
        model=model,
        tools=[search_web, search_company_news, search_sec_filings],
        system_prompt=_TECH_SYSTEM,
        user_message=user_msg,
        agent_name="TechAgent",
    )

    return {
        "tech_assessment": _parse_json(response),
        "completed_stages": ["tech_agent"],
        "agent_logs": logs,
    }


# ── Management Agent ───────────────────────────────────────────────────────────

_MANAGEMENT_SYSTEM = """You are the Management Assessment Agent for ARIA.

Your mandate: Evaluate the quality, track record, and retention risk of the target
company's leadership team. Analyze: CEO, CFO, CTO backgrounds, insider ownership,
compensation structure, culture signals, and succession depth.

Return structured JSON only.
"""


async def run_management_agent(state: DealState) -> dict:
    model = get_extraction_model()
    target = state["target_company"]

    user_msg = f"""
Conduct a management team assessment for {target}.
Context: {state.get("deal_brief", "")}

Research leadership, ownership, and culture. Return:
{{
    "leadership_team": [
        {{
            "name": "Full Name",
            "role": "CEO/CFO/CTO/etc",
            "tenure_years": <number or null>,
            "background": "brief background",
            "key_strengths": "1-2 strengths",
            "retention_risk": "high | medium | low"
        }}
    ],
    "founder_led": true/false,
    "insider_ownership_pct": <number or null>,
    "compensation_structure": {{
        "equity_heavy": true/false,
        "pay_for_performance": true/false,
        "notes": "any notable compensation flags"
    }},
    "culture_signals": {{
        "glassdoor_rating": <number or null>,
        "employee_growth_trend": "growing | stable | shrinking | unknown",
        "culture_notes": "any notable public culture signals"
    }},
    "key_person_risks": ["specific key person dependencies"],
    "track_record": {{
        "value_creation_history": "description of past outcomes",
        "capital_allocation_quality": "strong | adequate | weak | unknown"
    }},
    "management_summary": "3-4 sentence management assessment narrative"
}}
"""

    response, tool_calls, logs = await run_agent(
        model=model,
        tools=[search_web, search_company_news, search_sec_filings],
        system_prompt=_MANAGEMENT_SYSTEM,
        user_message=user_msg,
        agent_name="ManagementAgent",
    )

    return {
        "management_assessment": _parse_json(response),
        "completed_stages": ["management_agent"],
        "agent_logs": logs,
    }


# ── Helper ─────────────────────────────────────────────────────────────────────

def _extract_ticker(company_name: str) -> str:
    """Naive heuristic: if a ticker-like string is in the name, extract it."""
    import re
    match = re.search(r"\(([A-Z]{1,5})\)", company_name)
    return match.group(1) if match else ""
