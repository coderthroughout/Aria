from __future__ import annotations
from typing import TypedDict, Optional, List, Annotated
from operator import add


class AgentLog(TypedDict):
    agent: str
    timestamp: str
    status: str      # "started" | "tool_call" | "completed" | "failed" | "warning"
    message: str


class DealState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────────
    deal_brief: str
    target_company: str
    acquirer_company: str
    deal_size_usd: Optional[float]

    # ── Orchestration ──────────────────────────────────────────────────────────
    current_stage: str
    completed_stages: Annotated[List[str], add]

    # ── Stage 2: Data Gathering (6 agents, parallel) ──────────────────────────
    financial_data: Optional[dict]
    valuation_models: Optional[dict]
    market_data: Optional[dict]
    competitive_data: Optional[dict]
    tech_assessment: Optional[dict]
    management_assessment: Optional[dict]

    # ── Stage 3-4: Thesis Construction & Debate ───────────────────────────────
    bull_thesis: Optional[dict]
    bear_thesis: Optional[dict]
    debate_rounds: Optional[List[dict]]

    # ── Stage 5: Risk Assessment ──────────────────────────────────────────────
    risk_assessment: Optional[dict]

    # ── Stage 6: Human Review Gate ────────────────────────────────────────────
    human_approved: Optional[bool]
    human_notes: Optional[str]

    # ── Stage 7: Critique & Verification ─────────────────────────────────────
    critique_results: Optional[dict]

    # ── Stage 8: Final Output ─────────────────────────────────────────────────
    final_memo: Optional[dict]

    # ── Metadata (all append-only) ────────────────────────────────────────────
    errors: Annotated[List[str], add]
    warnings: Annotated[List[str], add]
    agent_logs: Annotated[List[AgentLog], add]
