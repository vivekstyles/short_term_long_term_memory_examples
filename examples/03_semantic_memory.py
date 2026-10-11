"""
Level 3: Long-Term Memory — Semantic / Declarative Memory.

Demonstrates:
1. Persistent Fact Storage: Timeless declarative facts stored in SQLite.
2. Dense Vector Embeddings: Semantic conceptual search (not just exact keyword match).
3. Hybrid Retrieval: Combining structured metadata filtering (SQL WHERE clauses) with cosine similarity ranking.
"""
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.long_term.semantic_memory import SemanticMemory
from src.visualizer import Visualizer, console


def run_demo() -> None:
    Visualizer.print_header(
        "Level 3: Semantic / Declarative Memory",
        "Persistent Out-of-Context Fact Storage & Dense Vector Retrieval",
    )

    # Use in-memory SQLite for self-contained demonstration
    memory = SemanticMemory(db_path=":memory:")

    # =========================================================================
    # Step 1: Populating Semantic Memory with Facts
    # =========================================================================
    Visualizer.print_step(1, "Populating Declarative Knowledge Base")

    sample_facts = [
        ("user", "User is a senior software engineer named Alex who specializes in distributed backend systems.", "identity"),
        ("user", "User strongly prefers dark mode interfaces and high-contrast syntax highlighting.", "preference"),
        ("user", "User dislikes verbose boilerplate explanations and prefers direct code solutions.", "preference"),
        ("project", "The primary target runtime is Python 3.12+ running on low-resource edge gateways.", "constraint"),
        ("project", "Persistent storage must use SQLite with WAL (Write-Ahead Logging) mode enabled.", "architecture"),
        ("project", "Memory consumption must remain strictly under 256MB at all times.", "constraint"),
        ("world_knowledge", "SQLite WAL mode allows multiple readers and one writer concurrently.", "fact"),
    ]

    for entity, fact, category in sample_facts:
        fact_id = memory.add_fact(fact=fact, entity=entity, category=category)
        console.print(f"  [dim]Stored Fact #{fact_id} [{entity} / {category}]: \"{fact[:60]}...\"[/dim]")

    console.print()
    all_facts = memory.get_all_facts()
    Visualizer.print_generic_table(
        title="Stored Semantic Facts in SQLite Database",
        headers=["ID", "Entity", "Category", "Fact Statement"],
        rows=[[f.id, f.entity, f.category, f.fact] for f in all_facts],
    )

    # =========================================================================
    # Step 2: Semantic Queries (Conceptual Similarity, not just keyword matches)
    # =========================================================================
    Visualizer.print_step(2, "Semantic Retrieval (Dense Vector Cosine Similarity)")

    queries = [
        "What UI theme does Alex prefer?",
        "What are the hardware and RAM limits for our deployment?",
        "How should our database handle concurrency?",
    ]

    for q in queries:
        console.print(f"[bold yellow]Incoming Query:[/bold yellow] [bold white]\"{q}\"[/bold white]")
        results = memory.search(query=q, top_k=2)

        table_rows = []
        for r in results:
            table_rows.append([
                str(r.id),
                f"{r.similarity:.4f}",
                r.category,
                r.fact,
            ])

        Visualizer.print_generic_table(
            title=f"Top Matches for: '{q}'",
            headers=["ID", "Cosine Sim", "Category", "Fact Text"],
            rows=table_rows,
        )

    # =========================================================================
    # Step 3: Hybrid Filtering (Structured SQL Filter + Vector Similarity)
    # =========================================================================
    Visualizer.print_step(3, "Hybrid Search (SQL Filtering + Vector Ranking)")
    console.print("[dim]Filtering strictly by category='preference' while searching conceptually for 'editor appearance':[/dim]")

    filtered_results = memory.search(
        query="editor appearance and fonts",
        category="preference",
        top_k=2,
    )

    Visualizer.print_generic_table(
        title="Filtered Semantic Results (Category: preference)",
        headers=["ID", "Cosine Sim", "Category", "Fact Text"],
        rows=[[f.id, f.similarity, f.category, f.fact] for f in filtered_results],
    )

    console.print(
        "[bold cyan]Takeaway:[/bold cyan] Semantic memory stores permanent declarative knowledge out-of-context. "
        "Agents dynamically retrieve only the most relevant facts into their short-term prompt when needed!\n"
    )


if __name__ == "__main__":
    run_demo()
