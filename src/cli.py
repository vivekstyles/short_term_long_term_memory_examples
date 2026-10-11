from __future__ import annotations
import sys
import os
import argparse
from typing import Optional

from src.core.llm_backend import get_llm, MockLLM, OllamaLLM
from src.visualizer import Visualizer, console
from src.unified.memgpt_agent import MemGPTAgent
from src.unified.consolidation import MemoryConsolidator
from src.long_term.episodic_memory import EpisodicMemory

import importlib

ex1 = importlib.import_module("examples.01_buffer_and_sliding_window")
ex2 = importlib.import_module("examples.02_summary_and_scratchpad")
ex3 = importlib.import_module("examples.03_semantic_memory")
ex4 = importlib.import_module("examples.04_episodic_memory")
ex5 = importlib.import_module("examples.05_procedural_memory")
ex6 = importlib.import_module("examples.06_unified_cognitive_agent")


def run_interactive_chat(force_mock: bool = False) -> None:
    """Live interactive chat session with real-time cognitive memory inspection."""
    Visualizer.print_header(
        "Interactive Agentic Memory Inspector",
        "Chat with the Agent & Watch its Memory Layers Mutate Live",
    )

    llm = get_llm(force_mock=force_mock)
    console.print(f"[bold cyan]Backend Active:[/bold cyan] {llm.__class__.__name__}")
    console.print("[dim]Commands available during chat:[/dim]")
    console.print("  [yellow]/memory[/yellow]      - Inspect Core Memory & Working Buffer")
    console.print("  [yellow]/consolidate[/yellow] - Trigger Memory Sleep Cycle (Consolidate into Long-Term)")
    console.print("  [yellow]/quit[/yellow]        - Return to main menu\n")

    agent = MemGPTAgent(llm=llm, db_path=":memory:")
    episodic_store = EpisodicMemory(db_path=":memory:", llm=llm)
    consolidator = MemoryConsolidator(
        semantic_memory=agent.archival_memory,
        episodic_memory=episodic_store,
        llm=llm,
    )

    while True:
        try:
            user_input = console.input("[bold green]You > [/bold green]").strip()
            if not user_input:
                continue

            if user_input.lower() in ("/quit", "exit", "quit"):
                console.print("[dim]Exiting interactive session.[/dim]\n")
                break

            if user_input.lower() == "/memory":
                Visualizer.print_memory_panel(
                    agent.core_memory.render(),
                    title="Active Core Memory Block",
                    border_style="cyan",
                )
                Visualizer.print_messages_table(
                    agent.buffer.get_messages(),
                    title="Short-Term Working Buffer",
                )
                continue

            if user_input.lower() == "/consolidate":
                console.print("[bold yellow]Executing Memory Consolidation Cycle...[/bold yellow]")
                report = consolidator.consolidate(buffer=agent.buffer, clear_buffer_after=True)
                Visualizer.print_generic_table(
                    title="Consolidation Report",
                    headers=["Metric / Category", "Details"],
                    rows=[
                        ["Extracted Facts", "\n".join(report.extracted_facts) if report.extracted_facts else "None"],
                        ["Recorded Episodes", "\n".join(report.recorded_episodes)],
                        ["Working Tokens Reclaimed", f"{report.tokens_freed} tokens"],
                    ],
                )
                continue

            # Standard chat turn
            reply, operations = agent.step(user_input)

            if operations:
                for op in operations:
                    console.print(f"  [cyan]Memory Tool Executed:[/cyan] [yellow]{op.get('function')}[/yellow] - [dim]{op.get('details')}[/dim]")

            console.print(f"[bold blue]Agent >[/bold blue] {reply}\n")

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session closed.[/dim]\n")
            break


def main_menu() -> None:
    parser = argparse.ArgumentParser(description="Agentic Memory Progressive Examples")
    parser.add_argument("--demo", type=str, choices=["1", "2", "3", "4", "5", "6", "all"], help="Run specific level")
    parser.add_argument("--interactive", action="store_true", help="Launch live interactive chat inspector")
    parser.add_argument("--mock", action="store_true", help="Force fast deterministic Mock LLM")
    args = parser.parse_args()

    demos = {
        "1": (ex1.run_demo, "Level 1: Buffer and Sliding Window (Short-Term)"),
        "2": (ex2.run_demo, "Level 2: Summary Buffer & Scratchpad (Short-Term)"),
        "3": (ex3.run_demo, "Level 3: Semantic / Declarative Memory (Long-Term)"),
        "4": (ex4.run_demo, "Level 4: Episodic Memory Stream & Reflection (Long-Term)"),
        "5": (ex5.run_demo, "Level 5: Procedural Memory & Dynamic Skills (Long-Term)"),
        "6": (ex6.run_demo, "Level 6: Unified Cognitive Agent & Consolidation"),
    }

    if args.mock:
        os.environ["USE_MOCK_LLM"] = "1"

    if args.interactive:
        run_interactive_chat(force_mock=args.mock)
        return

    if args.demo:
        if args.demo == "all":
            for k in sorted(demos.keys()):
                demos[k][0]()
        else:
            demos[args.demo][0]()
        return

    # Interactive Menu
    while True:
        Visualizer.print_header(
            "Agentic Memory Systems Curriculum",
            "Progressive Architecture from Working Context to Long-Term Cognition",
        )
        for k in sorted(demos.keys()):
            console.print(f"  [bold yellow][{k}][/bold yellow] {demos[k][1]}")
        console.print("  [bold yellow][7][/bold yellow] Interactive Live Chat (Live Memory Inspector)")
        console.print("  [bold yellow][8][/bold yellow] Run All Demonstrations Sequentially")
        console.print("  [bold yellow][0][/bold yellow] Exit\n")

        choice = console.input("[bold cyan]Select option [0-8]: [/bold cyan]").strip()
        if choice == "0":
            console.print("[dim]Goodbye![/dim]")
            break
        elif choice in demos:
            demos[choice][0]()
        elif choice == "7":
            run_interactive_chat(force_mock=args.mock)
        elif choice == "8":
            for k in sorted(demos.keys()):
                demos[k][0]()
        else:
            console.print("[red]Invalid selection. Please choose 0 through 8.[/red]\n")
