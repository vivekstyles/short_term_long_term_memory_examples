"""
Level 6: Advanced Unified Cognitive Architecture (MemGPT Pattern & Memory Consolidation).

Demonstrates:
1. MemGPT Hierarchical Memory System:
   - In-Context Core Memory (Persona + Human User Profile) with autonomous self-editing tools.
   - Out-of-Context Archival Memory (Persistent Vector Store).
   - Bounded Working Context (Sliding FIFO Window).
2. Autonomous Tool Execution:
   Agent uses tools like `core_memory_append` to update its own memory blocks live.
3. Offline Memory Consolidation ('Sleep Cycle'):
   Transferring short-term interaction dialogue into durable long-term storage
   and resetting the working buffer for the next session.
"""
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.llm_backend import get_llm
from src.long_term.episodic_memory import EpisodicMemory
from src.unified.memgpt_agent import MemGPTAgent
from src.unified.consolidation import MemoryConsolidator
from src.visualizer import Visualizer, console


def run_demo() -> None:
    Visualizer.print_header(
        "Level 6: Unified Cognitive Architecture",
        "MemGPT Hierarchical Memory & End-of-Session Memory Consolidation ('Sleep Cycle')",
    )

    llm = get_llm()
    console.print(f"[bold cyan]Using LLM Backend:[/bold cyan] {llm.__class__.__name__}\n")

    # =========================================================================
    # Step 1: Initializing Agent and Inspecting Initial Core Memory
    # =========================================================================
    Visualizer.print_step(1, "Inspecting Agent's Initial Core Memory (In-Context)")

    agent = MemGPTAgent(llm=llm, db_path=":memory:")
    Visualizer.print_memory_panel(
        agent.core_memory.render(),
        title="Initial Core Memory (Visible in System Prompt)",
        border_style="magenta",
    )

    # =========================================================================
    # Step 2: Multi-Turn Conversation with Autonomous Memory Self-Editing
    # =========================================================================
    Visualizer.print_step(2, "Conversational Interaction & Autonomous Memory Operations")

    interactions = [
        "Hello! My name is Jordan, and I develop flight telemetry software for autonomous drones.",
        "Remember that all drone logs must be encrypted with AES-256 before writing to local disk.",
        "Can you confirm what you remember about my identity and our encryption rule?",
    ]

    for turn_idx, user_input in enumerate(interactions, 1):
        console.print(f"[bold yellow]Turn {turn_idx} [USER]:[/bold yellow] \"{user_input}\"")
        reply, operations = agent.step(user_input)

        if operations:
            for op in operations:
                console.print(f"  [bold cyan]• Memory Tool Executed:[/bold cyan] [green]{op.get('function')}[/green]")
                console.print(f"    [dim]Details: {op.get('details')}[/dim]")

        console.print(f"[bold green]Turn {turn_idx} [AGENT]:[/bold green] \"{reply}\"\n")

    Visualizer.print_memory_panel(
        agent.core_memory.render(),
        title="Updated Core Memory (Autonomously Edited by Agent)",
        border_style="green",
    )

    # =========================================================================
    # Step 3: End-of-Session Memory Consolidation ('Sleep Cycle')
    # =========================================================================
    Visualizer.print_step(3, "Memory Consolidation Phase ('Sleep Cycle')")
    console.print(
        "[dim]Simulating session close / overnight sleep cycle: extracting durable facts and episodes, "
        "committing them to persistent SQLite storage, and resetting the volatile working buffer.[/dim]\n"
    )

    episodic_store = EpisodicMemory(db_path=":memory:", llm=llm)
    consolidator = MemoryConsolidator(
        semantic_memory=agent.archival_memory,
        episodic_memory=episodic_store,
        llm=llm,
    )

    initial_buffer_tokens = agent.buffer.get_total_tokens()
    console.print(f"Working buffer before consolidation: [cyan]{initial_buffer_tokens} tokens[/cyan].")

    report = consolidator.consolidate(buffer=agent.buffer, clear_buffer_after=True)

    Visualizer.print_generic_table(
        title="Consolidation Report",
        headers=["Category", "Details"],
        rows=[
            ["Extracted Semantic Facts", "\n".join(report.extracted_facts) if report.extracted_facts else "None"],
            ["Recorded Episodes", "\n".join(report.recorded_episodes)],
            ["Tokens Reclaimed", f"{report.tokens_freed} tokens"],
            ["Summary", report.summary],
        ],
    )

    console.print(f"Working buffer after consolidation: [green]{agent.buffer.get_total_tokens()} tokens[/green] (Buffer flushed for next session!).\n")

    # =========================================================================
    # Step 4: Proving Persistence Across Sessions
    # =========================================================================
    Visualizer.print_step(4, "Session 2 Bootstrapping (Recalling from Long-Term Memory)")
    console.print("[dim]Even with an empty short-term buffer, the agent recalls Jordan's telemetry and encryption rules:[/dim]\n")

    recalled_facts = agent.archival_memory.search("drone encryption telemetry rule", top_k=2)
    Visualizer.print_generic_table(
        title="Facts Recalled from Persistent Long-Term Memory",
        headers=["ID", "Cosine Sim", "Fact Text"],
        rows=[[f.id, f.similarity, f.fact] for f in recalled_facts],
    )

    console.print(
        "[bold cyan]Takeaway:[/bold cyan] Level 6 combines all memory levels: Core Memory (in-context), "
        "Working Buffer (sliding window), Archival/Semantic Memory (vector retrieval), and "
        "Memory Consolidation (sleep cycle) to create a truly persistent, autonomous agentic cognitive architecture!\n"
    )


if __name__ == "__main__":
    run_demo()
