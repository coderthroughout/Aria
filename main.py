#!/usr/bin/env python3
"""
ARIA — Autonomous Research & Intelligence Analyst
M&A Due Diligence Multi-Agent System

Usage:
    python main.py
    python main.py --target "Stripe" --acquirer "Visa" --size 50000000000
    python main.py --resume <thread_id>
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 on Windows before Rich initialises
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

load_dotenv()

console = Console()


# ── State initializer ─────────────────────────────────────────────────────────

def build_initial_state(
    deal_brief: str,
    target_company: str,
    acquirer_company: str,
    deal_size_usd: float | None = None,
) -> dict:
    return {
        "deal_brief": deal_brief,
        "target_company": target_company,
        "acquirer_company": acquirer_company,
        "deal_size_usd": deal_size_usd,
        "current_stage": "init",
        "completed_stages": [],
        "financial_data": None,
        "valuation_models": None,
        "market_data": None,
        "competitive_data": None,
        "tech_assessment": None,
        "management_assessment": None,
        "bull_thesis": None,
        "bear_thesis": None,
        "debate_rounds": None,
        "risk_assessment": None,
        "human_approved": None,
        "human_notes": None,
        "critique_results": None,
        "final_memo": None,
        "errors": [],
        "warnings": [],
        "agent_logs": [],
    }


# ── Display helpers ────────────────────────────────────────────────────────────

def print_banner():
    console.print(Panel(
        "[bold cyan]ARIA[/bold cyan] — Autonomous Research & Intelligence Analyst\n"
        "[dim]M&A Due Diligence · 12-Agent System · LangGraph[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))


def print_stage(stage: str, detail: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    console.print(f"[dim]{ts}[/dim]  [bold green]▶[/bold green]  [white]{stage}[/white]  [dim]{detail}[/dim]")


def print_checkpoint(name: str):
    console.print(f"  [bold yellow]⬡ CHECKPOINT {name}[/bold yellow] saved")


def print_error(msg: str):
    console.print(f"  [bold red]✗[/bold red] {msg}")


def print_success(msg: str):
    console.print(f"  [bold green]✓[/bold green] {msg}")


def show_human_review_prompt(review_data: dict) -> tuple[bool, str]:
    """Display the human review gate and collect approval."""
    console.print()
    console.print(Panel(
        "[bold yellow]⏸  HUMAN REVIEW GATE[/bold yellow]\n"
        "Review the preliminary analysis before committing to full report generation.",
        border_style="yellow",
    ))

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="dim")
    table.add_column("Value", style="white")

    table.add_row("Target", review_data.get("target", ""))
    table.add_row("Acquirer", review_data.get("acquirer", ""))
    if review_data.get("deal_size_usd"):
        table.add_row("Deal Size", f"${review_data['deal_size_usd']:,.0f}")
    table.add_row("Bull Conviction", f"{review_data.get('bull_conviction', '?')}/10")
    table.add_row("Bear Conviction", f"{review_data.get('bear_conviction', '?')}/10")
    table.add_row("Debate Direction", str(review_data.get("debate_direction", "unknown")))
    table.add_row("Risk Recommendation", str(review_data.get("risk_recommendation", "unknown")))
    table.add_row("Risk Score", f"{review_data.get('risk_score', '?')}/10")

    if review_data.get("hard_stops"):
        table.add_row("[red]Hard Stops[/red]", str(review_data["hard_stops"]))

    console.print(table)
    console.print()

    while True:
        choice = console.input(
            "[bold]Proceed with full report generation? [/bold][dim](y/n/notes)[/dim] > "
        ).strip().lower()
        if choice in ("y", "yes", ""):
            return True, ""
        elif choice in ("n", "no"):
            return False, "Rejected at human review gate"
        else:
            notes = choice
            confirm = console.input(f"  Add note '{notes}' and proceed? [y/n] > ").strip().lower()
            if confirm in ("y", "yes"):
                return True, notes


def save_output(state: dict, thread_id: str, output_dir: str = "outputs"):
    """Save the final memo and full state to disk."""
    Path(output_dir).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = (state.get("target_company") or "unknown").replace(" ", "_")[:30]

    # Full state (for debugging/replay)
    state_path = Path(output_dir) / f"aria_{target}_{ts}_state.json"
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, default=str)

    # Final memo only
    if state.get("final_memo"):
        memo_path = Path(output_dir) / f"aria_{target}_{ts}_memo.json"
        with open(memo_path, "w") as f:
            json.dump(state["final_memo"], f, indent=2, default=str)
        console.print(f"\n[bold green]Memo saved:[/bold green] {memo_path}")

    console.print(f"[dim]Full state saved: {state_path}[/dim]")
    return str(state_path)


# ── Main execution loop ────────────────────────────────────────────────────────

async def run_aria(
    deal_brief: str,
    target_company: str,
    acquirer_company: str,
    deal_size_usd: float | None = None,
    thread_id: str | None = None,
    resume: bool = False,
    output_dir: str = "outputs",
):
    from config import settings
    from graph.workflow import build_graph, build_graph_postgres

    print_banner()

    # Build graph with appropriate checkpointer
    if settings.checkpoint_backend == "postgres" and settings.postgres_url:
        console.print("[dim]Using PostgreSQL checkpointer[/dim]")
        graph = build_graph_postgres(settings.postgres_url)
    else:
        console.print("[dim]Using in-memory checkpointer[/dim]")
        graph = build_graph()

    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    console.print(f"[dim]Thread ID: {thread_id}[/dim]\n")

    if not resume:
        state = build_initial_state(deal_brief, target_company, acquirer_company, deal_size_usd)
        console.print(Panel(
            f"[bold]{acquirer_company}[/bold] acquiring [bold]{target_company}[/bold]\n"
            f"[dim]{deal_brief}[/dim]",
            title="Deal Brief",
            border_style="blue",
        ))
        console.print()
    else:
        state = None  # Resume from checkpoint — state is in the checkpointer
        console.print(f"[yellow]Resuming thread {thread_id}[/yellow]\n")

    from langgraph.types import Command

    # Stream graph execution, handling interrupts (human review gate)
    current_state = state
    interrupted = False
    human_input_value = None

    while True:
        try:
            # Either start fresh or resume with human input after an interrupt
            if human_input_value is not None:
                events = graph.astream(
                    Command(resume=human_input_value),
                    config=config,
                    stream_mode="updates",
                )
            elif current_state is not None:
                events = graph.astream(current_state, config=config, stream_mode="updates")
                current_state = None
            else:
                break

            async for event in events:
                for node_name, node_output in event.items():
                    if node_name == "__interrupt__":
                        # Human review gate triggered
                        interrupted = True
                        print_checkpoint("G — human gate")
                        interrupt_data = node_output
                        if isinstance(interrupt_data, (list, tuple)) and len(interrupt_data) > 0:
                            review_data = interrupt_data[0].value if hasattr(interrupt_data[0], "value") else interrupt_data[0]
                        else:
                            review_data = {}

                        approved, notes = show_human_review_prompt(review_data)
                        human_input_value = {"approved": approved, "notes": notes}

                        if not approved:
                            print_error("Analysis rejected at human review gate. Stopping.")
                            # Still save what we have
                            snapshot = graph.get_state(config)
                            if snapshot and snapshot.values:
                                save_output(snapshot.values, thread_id, output_dir)
                            return
                        break
                    else:
                        # Normal node completion
                        stage_display = node_name.replace("_", " ").title()
                        completed = (node_output.get("completed_stages") or [])
                        logs = node_output.get("agent_logs") or []
                        errors = node_output.get("errors") or []

                        print_stage(stage_display, f"[{', '.join(completed)}]" if completed else "")

                        for log in logs:
                            if log.get("status") == "tool_call":
                                console.print(f"    [dim cyan]{log['message']}[/dim cyan]")
                        for err in errors:
                            print_error(err)

                if interrupted:
                    break

            if not interrupted:
                break  # Graph completed normally

            interrupted = False  # Reset for next loop iteration

        except Exception as exc:
            print_error(f"Graph execution error: {exc}")
            import traceback
            traceback.print_exc()
            break

    # Retrieve final state from checkpointer
    snapshot = graph.get_state(config)
    final_state = snapshot.values if snapshot else {}

    console.print()

    if final_state.get("final_memo"):
        rec = (final_state["final_memo"].get("recommendation") or {}).get("decision", "UNKNOWN")
        color = {"PROCEED": "green", "PROCEED_WITH_CONDITIONS": "yellow", "DO_NOT_PROCEED": "red"}.get(rec, "white")
        console.print(Panel(
            f"[bold {color}]{rec}[/bold {color}]\n\n"
            + (final_state["final_memo"].get("executive_summary") or {}).get("key_thesis", ""),
            title="[bold]Final Recommendation[/bold]",
            border_style=color,
        ))
        save_output(final_state, thread_id, output_dir)
    elif final_state:
        console.print("[yellow]Analysis incomplete — saving partial state.[/yellow]")
        save_output(final_state, thread_id, output_dir)
    else:
        console.print("[red]No state recovered.[/red]")

    # Print agent log summary
    all_logs = final_state.get("agent_logs", [])
    if all_logs:
        console.print(f"\n[dim]Total agent log entries: {len(all_logs)}[/dim]")
        agents_run = sorted({log["agent"] for log in all_logs})
        console.print(f"[dim]Agents executed: {', '.join(agents_run)}[/dim]")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ARIA — Autonomous M&A Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --target "Stripe" --acquirer "Visa" --size 50000000000
  python main.py --resume --thread abc-123-def
        """,
    )
    parser.add_argument("--target", help="Target company name")
    parser.add_argument("--acquirer", help="Acquiring company name")
    parser.add_argument("--size", type=float, help="Deal size in USD")
    parser.add_argument("--brief", help="Custom deal brief text")
    parser.add_argument("--thread", help="Thread ID (for resume)")
    parser.add_argument("--resume", action="store_true", help="Resume an existing thread")
    parser.add_argument("--output", default="outputs", help="Output directory")

    args = parser.parse_args()

    if args.resume and not args.thread:
        parser.error("--resume requires --thread <thread_id>")

    # Interactive prompts if not provided via CLI
    if not args.resume:
        target = args.target or console.input("[bold]Target company:[/bold] ").strip()
        acquirer = args.acquirer or console.input("[bold]Acquiring company:[/bold] ").strip()
        size_input = args.size
        if not size_input:
            raw = console.input("[bold]Deal size in USD (e.g. 50000000000, or press Enter to skip):[/bold] ").strip()
            size_input = float(raw.replace(",", "")) if raw else None

        deal_brief = args.brief or (
            f"Analyze {target} as an acquisition target for {acquirer}"
            + (f" at a deal size of ${size_input:,.0f}" if size_input else "")
            + f". Provide a comprehensive investment-grade due diligence analysis."
        )

        asyncio.run(run_aria(
            deal_brief=deal_brief,
            target_company=target,
            acquirer_company=acquirer,
            deal_size_usd=size_input,
            thread_id=args.thread,
            output_dir=args.output,
        ))
    else:
        # Resume mode — need some placeholder values (state comes from checkpoint)
        asyncio.run(run_aria(
            deal_brief="[Resuming from checkpoint]",
            target_company="[Resuming]",
            acquirer_company="[Resuming]",
            thread_id=args.thread,
            resume=True,
            output_dir=args.output,
        ))


if __name__ == "__main__":
    main()
