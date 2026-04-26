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


# ── Human-in-the-loop gate ─────────────────────────────────────────────────────

async def human_review_gate(state: DealState) -> dict:
    """
    Pauses the graph and waits for human approval before committing to full report
    generation. The graph is interrupted here — the human sees bull conviction,
    bear conviction, valuation range, and risk flags, then approves or rejects.
    """
    from langgraph.types import interrupt

    bull = state.get("bull_thesis") or {}
    bear = state.get("bear_thesis") or {}
    risk = state.get("risk_assessment") or {}
    valuation = state.get("valuation_models") or {}

    # Build a concise summary for the human reviewer
    review_summary = {
        "target": state["target_company"],
        "acquirer": state["acquirer_company"],
        "deal_size_usd": state.get("deal_size_usd"),
        "bull_conviction": bull.get("conviction_score"),
        "bear_conviction": bear.get("conviction_score"),
        "bull_thesis_title": bull.get("thesis_title"),
        "bear_thesis_title": bull.get("thesis_title"),
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

    # This call pauses graph execution until a human resumes it
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
    """Run two debate rounds then move to risk assessment."""
    rounds = state.get("debate_rounds") or []
    if len(rounds) < 2:
        return "debate_round_2"
    return "assess_risk"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """
    Build the full ARIA LangGraph workflow.

    Stages:
      1. decompose         — DealDirector parses the brief          [CHECKPOINT A]
      2-7. parallel data   — 6 specialist agents gather data        [CHECKPOINT B]
      8-9. parallel thesis — Bull + Bear build opposing theses      [CHECKPOINT C]
      10.  debate_round_1  — Structured adversarial debate round 1  [CHECKPOINT D]
      11.  debate_round_2  — Debate round 2                         [CHECKPOINT E]
      12.  assess_risk     — RiskManager independent assessment     [CHECKPOINT F]
      13.  human_review    — Human approval gate (interrupt)        [CHECKPOINT G]
      14.  critique        — CritiqueAgent validates all claims     [CHECKPOINT H]
      15.  generate_report — ReportAgent writes investment memo     [CHECKPOINT I]
    """
    g = StateGraph(DealState)

    # ── Stage 1: Decompose ────────────────────────────────────────────────────
    g.add_node("decompose", run_deal_director)

    # ── Stage 2: Parallel Data Gathering (6 agents) ───────────────────────────
    g.add_node("gather_financials", run_financial_agent)
    g.add_node("run_valuation", run_valuation_agent)
    g.add_node("gather_market", run_market_agent)
    g.add_node("analyze_competition", run_competitive_agent)
    g.add_node("assess_technology", run_tech_agent)
    g.add_node("assess_management", run_management_agent)

    # ── Stage 3: Parallel Thesis Construction ─────────────────────────────────
    g.add_node("build_bull_thesis", run_bull_analyst)
    g.add_node("build_bear_thesis", run_bear_analyst)

    # ── Stage 4-5: Debate (2 rounds) ──────────────────────────────────────────
    g.add_node("debate_round_1", run_debate)
    g.add_node("debate_round_2", run_debate)

    # ── Stage 6: Risk Assessment ──────────────────────────────────────────────
    g.add_node("assess_risk", run_risk_manager)

    # ── Stage 7: Human Review Gate ────────────────────────────────────────────
    g.add_node("human_review", human_review_gate)

    # ── Stage 8: Critique & Verification ─────────────────────────────────────
    g.add_node("critique_analysis", run_critique_agent)

    # ── Stage 9: Report Generation ────────────────────────────────────────────
    g.add_node("generate_report", run_report_agent)

    # ── Edges ─────────────────────────────────────────────────────────────────

    # Entry → Decompose
    g.add_edge(START, "decompose")

    # Decompose → Fan out to 6 parallel data agents
    for node in [
        "gather_financials", "run_valuation", "gather_market",
        "analyze_competition", "assess_technology", "assess_management",
    ]:
        g.add_edge("decompose", node)

    # All 6 data agents → Parallel thesis construction
    # LangGraph waits for all incoming edges before running a node,
    # so both thesis nodes only start once ALL 6 data agents have finished.
    for data_node in [
        "gather_financials", "run_valuation", "gather_market",
        "analyze_competition", "assess_technology", "assess_management",
    ]:
        g.add_edge(data_node, "build_bull_thesis")
        g.add_edge(data_node, "build_bear_thesis")

    # Both thesis nodes → Debate round 1
    g.add_edge("build_bull_thesis", "debate_round_1")
    g.add_edge("build_bear_thesis", "debate_round_1")

    # Debate round 1 → Debate round 2
    g.add_edge("debate_round_1", "debate_round_2")

    # Debate round 2 → Risk assessment
    g.add_edge("debate_round_2", "assess_risk")

    # Risk → Human review gate
    g.add_edge("assess_risk", "human_review")

    # Human review → Critique (approved) or END (rejected)
    g.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {"critique_analysis": "critique_analysis", END: END},
    )

    # Critique → Report
    g.add_edge("critique_analysis", "generate_report")

    # Report → END
    g.add_edge("generate_report", END)

    checkpointer = checkpointer or MemorySaver()
    return g.compile(checkpointer=checkpointer)


def build_graph_postgres(connection_string: str):
    """Build the graph with a PostgreSQL checkpointer for production use."""
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
