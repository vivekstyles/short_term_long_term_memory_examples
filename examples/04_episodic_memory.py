"""
Level 4: Long-Term Memory — Episodic Memory Stream & Reflection.

Demonstrates:
1. Stanford Generative Agents Memory Stream (Park et al., 2023):
   Chronological event stream with temporal decay and importance weighting.
2. The Tri-Partite Retrieval Scoring Formula:
   Score = w_rec * Recency + w_imp * Importance + w_rel * Relevance
3. Cognitive Reflection:
   Synthesizing higher-level insights from clusters of recent episodic events
   and writing reflections back into the memory stream.
"""
import sys
import os
from datetime import datetime, timezone, timedelta

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.llm_backend import get_llm
from src.long_term.episodic_memory import EpisodicMemory
from src.visualizer import Visualizer, console


def run_demo() -> None:
    Visualizer.print_header(
        "Level 4: Episodic Memory Stream & Reflection",
        "Experience Logging, Cognitive Scoring & Higher-Order Reflection",
    )

    llm = get_llm()
    console.print(f"[bold cyan]Using LLM Backend:[/bold cyan] {llm.__class__.__name__}\n")

    # In-memory episodic store with 0.98 hourly decay
    memory = EpisodicMemory(db_path=":memory:", llm=llm, decay_factor=0.98)

    # =========================================================================
    # Step 1: Populating Episodic Memory with Historical Events
    # =========================================================================
    Visualizer.print_step(1, "Recording Historical Experience Stream (Simulated Timeline)")

    now = datetime.now(timezone.utc)
    historical_episodes = [
        ("User initialized project repository and configured Git hooks.", 3.0, now - timedelta(days=4)),
        ("Disk ran out of space during automated backup migration, causing system crash.", 9.5, now - timedelta(days=2)),
        ("Configured SQLite WAL mode and verified disk write concurrency.", 6.5, now - timedelta(days=1)),
        ("User joined team standup call and drank coffee.", 1.5, now - timedelta(hours=3)),
        ("Applied indexing optimization to customer_orders table.", 5.0, now - timedelta(hours=1)),
    ]

    for desc, imp, ts in historical_episodes:
        memory.record_episode(
            description=desc,
            importance_score=imp,
            created_at=ts,
            metadata={"simulated_time": ts.isoformat()},
        )
        time_ago_str = f"{(now - ts).total_seconds() / 3600.0:.1f}h ago"
        console.print(f"  [dim]Logged ({time_ago_str}, Importance {imp}/10): \"{desc[:65]}...\"[/dim]")

    console.print()

    # =========================================================================
    # Step 2: Demonstrating Tri-Partite Scoring
    # =========================================================================
    Visualizer.print_step(2, "Cognitive Retrieval: Recency vs. Importance vs. Relevance")
    console.print(
        "[bold cyan]Scoring Formula:[/bold cyan] "
        "[dim]Total = 1.0 * Recency + 1.0 * Importance + 1.0 * Relevance[/dim]\n"
    )

    query = "Why did the system crash earlier this week?"
    console.print(f"[bold yellow]Incoming Agent Query:[/bold yellow] [bold white]\"{query}\"[/bold white]")

    scored_episodes = memory.retrieve(
        query=query,
        top_k=3,
        w_rec=1.0,
        w_imp=1.0,
        w_rel=1.0,
        current_time=now,
    )

    table_rows = []
    for s in scored_episodes:
        table_rows.append([
            f"#{s.episode.id} ({s.episode.memory_type})",
            f"{s.recency_score:.3f}",
            f"{s.importance_score:.3f}",
            f"{s.relevance_score:.3f}",
            f"[bold green]{s.total_score:.3f}[/bold green]",
            s.episode.description,
        ])

    Visualizer.print_generic_table(
        title=f"Episodic Retrieval Ranking for: '{query}'",
        headers=["ID", "Recency", "Importance", "Relevance", "Total Score", "Episode Description"],
        rows=table_rows,
    )

    console.print(
        "[bold green]Observation:[/bold green] Even though the crash happened 48 hours ago (low recency), "
        "its extreme importance (0.95) and high semantic relevance elevated it to the #1 retrieved memory, "
        "completely outranking recent trivial events!\n"
    )

    # =========================================================================
    # Step 3: Reflection Loop (Synthesizing Abstract Insights)
    # =========================================================================
    Visualizer.print_step(3, "Cognitive Reflection (Synthesizing High-Order Insights)")
    console.print("[dim]The agent reviews recent episodes to synthesize generalized lessons learned...[/dim]\n")

    reflections = memory.reflect(current_time=now)
    for idx, ref in enumerate(reflections, 1):
        console.print(f"[bold magenta]Synthesized Reflection #{idx}:[/bold magenta] {ref}")

    console.print()
    console.print("[dim]Verifying that reflection was recorded into episodic memory stream with high importance:[/dim]")
    
    reflection_results = memory.retrieve("disk space and backup reliability", top_k=2, current_time=now)
    Visualizer.print_generic_table(
        title="Retrieval Matching Synthesized Reflections",
        headers=["Type", "Importance", "Total Score", "Description"],
        rows=[
            [r.episode.memory_type, r.episode.importance_score, r.total_score, r.episode.description]
            for r in reflection_results
        ],
    )

    console.print(
        "[bold cyan]Takeaway:[/bold cyan] Episodic memory ensures agents learn from past events. "
        "The Stanford scoring model balances what is recent, what is important, and what is relevant!\n"
    )


if __name__ == "__main__":
    run_demo()
