from __future__ import annotations
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import DealState
from agents.deal_director import run_deal_director
from agents.data_agents import (
    run_financial_agent,
    run_valuation_agent,
    run_market_agent,
    run_competitive_agent,
    run_tech_agent,
    run_management_agent,
)
from agents.analyst_agents import run_bull_analyst, run_bear_analyst, run_debate
from agents.risk_agent import run_risk_manager
from agents.critique_agent import run_critique_agent
from agents.report_agent import run_report_agent
from config import settings


# ── Omium checkpoint wrapper ───────────────────────────────────────────────────
# Wraps a node function so Omium records a checkpoint event after it succeeds.
# If Omium is not configured, the wrapper is transparent.

def _with_checkpoint(name: str, fn):
    """Return fn wrapped with an Omium checkpoint marker (no-op if no API key)."""
    if not settings.omium_api_key:
        return fn
    try:
        import omium

        async def _wrapped(state: DealState) -> dict:
            result = await fn(state)
            return result

        # Apply the checkpoint decorator to the inner wrapper
        _wrapped = omium.checkpoint(name)(_wrapped)
        _wrapped.__name__ = getattr(fn, "__name__", name)
        return _wrapped
    except Exception:
        return fn


# ── Human-in-the-loop gate ─────────────────────────────────────────────────────

async def human_review_gate(state: DealState) -> dict:
    from langgraph.types import interrupt

    bull = state.get("bull_thesis") or {}
    bear = state.get("bear_thesis") or {}
    risk = state.get("risk_assessment") or {}
    valuation = state.get("valuation_models") or {}

    review_summary = {
        "target": state["target_company"],
        "acquirer": state["acquirer_company"],
        "deal_size_usd": state.get("deal_size_usd"),
        "bull_conviction": bull.get("conviction_score"),
        "bear_conviction": bear.get("conviction_score"),
        "bull_thesis_title": bull.get("thesis_title"),
        "bear_thesis_title": bear.get("thesis_title"),
        "valuation_range": (valuation.get("valuation_summary") or {}).get("blended_ev_low"),
        "risk_recommendation": (risk.get("risk_adjusted_recommendation") or {}).get("decision"),
        "risk_score": risk.get("overall_risk_score"),
        "hard_stops": risk.get("hard_stops", []),
        "debate_direction": (
            ((state.get("debate_rounds") or [{}])[-1] or {})
            .get("coordinator_assessment", {})
            .get("debate_direction", "unknown")
        ),
    }

    human_input = interrupt(review_summary)

    approved = True
    notes = ""
    if isinstance(human_input, dict):
        approved = human_input.get("approved", True)
        notes = human_input.get("notes", "")
    elif isinstance(human_input, bool):
        approved = human_input

    return {
        "human_approved": approved,
        "human_notes": notes,
        "completed_stages": ["human_review"],
    }


# ── Routing functions ──────────────────────────────────────────────────────────

def route_after_human_review(state: DealState) -> Literal["critique_analysis", END]:
    if state.get("human_approved") is False:
        return END
    return "critique_analysis"


def route_after_debate(state: DealState) -> Literal["debate_round_2", "assess_risk"]:
    rounds = state.get("debate_rounds") or []
    if len(rounds) < 2:
        return "debate_round_2"
    return "assess_risk"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    g = StateGraph(DealState)

    # ── Stage 1: Director ─────────────────────────────────────────────────────
    g.add_node("decompose",            _with_checkpoint("stage_1_director",       run_deal_director))

    # ── Stage 2: Parallel Data Gathering ─────────────────────────────────────
    g.add_node("gather_financials",    _with_checkpoint("stage_2_financials",     run_financial_agent))
    g.add_node("run_valuation",        _with_checkpoint("stage_2_valuation",      run_valuation_agent))
    g.add_node("gather_market",        _with_checkpoint("stage_2_market",         run_market_agent))
    g.add_node("analyze_competition",  _with_checkpoint("stage_2_competition",    run_competitive_agent))
    g.add_node("assess_technology",    _with_checkpoint("stage_2_technology",     run_tech_agent))
    g.add_node("assess_management",    _with_checkpoint("stage_2_management",     run_management_agent))

    # ── Stage 3: Thesis Construction ──────────────────────────────────────────
    g.add_node("build_bull_thesis",    _with_checkpoint("stage_3_bull",           run_bull_analyst))
    g.add_node("build_bear_thesis",    _with_checkpoint("stage_3_bear",           run_bear_analyst))

    # ── Stage 4: Debate ───────────────────────────────────────────────────────
    g.add_node("debate_round_1",       _with_checkpoint("stage_4_debate_r1",      run_debate))
    g.add_node("debate_round_2",       _with_checkpoint("stage_4_debate_r2",      run_debate))

    # ── Stage 5: Risk Assessment ──────────────────────────────────────────────
    g.add_node("assess_risk",          _with_checkpoint("stage_5_risk",           run_risk_manager))

    # ── Stage 6: Human Review Gate ────────────────────────────────────────────
    g.add_node("human_review",         human_review_gate)

    # ── Stage 7: Critique ─────────────────────────────────────────────────────
    g.add_node("critique_analysis",    _with_checkpoint("stage_7_critique",       run_critique_agent))

    # ── Stage 8: Report ───────────────────────────────────────────────────────
    g.add_node("generate_report",      _with_checkpoint("stage_8_report",         run_report_agent))

    # ── Edges ─────────────────────────────────────────────────────────────────

    g.add_edge(START, "decompose")

    for node in [
        "gather_financials", "run_valuation", "gather_market",
        "analyze_competition", "assess_technology", "assess_management",
    ]:
        g.add_edge("decompose", node)

    for data_node in [
        "gather_financials", "run_valuation", "gather_market",
        "analyze_competition", "assess_technology", "assess_management",
    ]:
        g.add_edge(data_node, "build_bull_thesis")
        g.add_edge(data_node, "build_bear_thesis")

    g.add_edge("build_bull_thesis", "debate_round_1")
    g.add_edge("build_bear_thesis", "debate_round_1")
    g.add_edge("debate_round_1",    "debate_round_2")
    g.add_edge("debate_round_2",    "assess_risk")
    g.add_edge("assess_risk",       "human_review")

    g.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {"critique_analysis": "critique_analysis", END: END},
    )

    g.add_edge("critique_analysis", "generate_report")
    g.add_edge("generate_report",   END)

    checkpointer = checkpointer or MemorySaver()
    return g.compile(checkpointer=checkpointer)


def build_graph_postgres(connection_string: str):
    try:
        from psycopg_pool import ConnectionPool
        from langgraph.checkpoint.postgres import PostgresSaver

        pool = ConnectionPool(connection_string, max_size=10)
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        return build_graph(checkpointer=checkpointer)
    except ImportError:
        raise ImportError(
            "PostgreSQL checkpointing requires: pip install 'aria[postgres]'"
        )
