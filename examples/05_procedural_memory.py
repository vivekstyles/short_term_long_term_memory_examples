"""
Level 5: Long-Term Memory — Procedural Memory & Dynamic Skills.

Demonstrates:
1. Reusable Skill Registry (Voyager Agent Pattern):
   Action recipes, tool-calling templates, and operational procedures stored persistently.
2. Few-Shot Trajectory Retrieval:
   Past successful execution traces retrieved dynamically to guide the LLM on complex tasks.
3. In-Context Dynamic Prompt Generation:
   Injecting only the relevant skills and exemplars into the agent's prompt.
"""
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.long_term.procedural_memory import ProceduralMemory
from src.visualizer import Visualizer, console


def run_demo() -> None:
    Visualizer.print_header(
        "Level 5: Procedural Memory & Dynamic Skills",
        "Action Recipes, Trajectory Exemplars & Few-Shot Procedural Guidance",
    )

    memory = ProceduralMemory(db_path=":memory:")

    # =========================================================================
    # Step 1: Registering Reusable Operational Skills
    # =========================================================================
    Visualizer.print_step(1, "Registering Reusable Skills in Procedural Memory")

    skills_data = [
        (
            "sqlite_wal_setup",
            "Configures SQLite database connection with WAL mode and busy timeout for high concurrency.",
            "setup sqlite database connection concurrency wal",
            """import sqlite3
def get_db(db_path: str):
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn""",
        ),
        (
            "safe_schema_migration",
            "Executes atomic schema migrations with pre-flight backup and rollback on failure.",
            "database schema migration alter table add column rollback",
            """def migrate(conn, sql_statements):
    with conn:
        for stmt in sql_statements:
            conn.execute(stmt)
    print("Migration committed safely.")""",
        ),
        (
            "edge_memory_profiler",
            "Monitors process memory footprint to ensure edge gateway limit (<256MB) is not breached.",
            "profile memory ram limit edge gateway benchmark",
            """import os, psutil
def check_memory():
    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / (1024 * 1024)
    assert rss_mb < 256.0, f"Memory exceeded limit: {rss_mb:.2f}MB"
    return rss_mb""",
        ),
    ]

    for name, desc, trig, recipe in skills_data:
        memory.register_skill(name=name, description=desc, trigger_pattern=trig, code_recipe=recipe)
        console.print(f"  [dim]Registered Skill: '{name}' ({desc})[/dim]")

    # Simulate skill usage metrics
    memory.record_skill_execution("sqlite_wal_setup", success=True)
    memory.record_skill_execution("sqlite_wal_setup", success=True)
    memory.record_skill_execution("edge_memory_profiler", success=True)

    console.print()

    # =========================================================================
    # Step 2: Recording Successful Execution Trajectories (Few-Shot Exemplars)
    # =========================================================================
    Visualizer.print_step(2, "Storing Past Successful Execution Trajectories")

    trajectories_data = [
        (
            "Recover from corrupted SQLite journal lock on edge gateway",
            [
                "Inspect active lock: check if -wal or -shm files are locked by another PID",
                "Execute PRAGMA integrity_check to verify B-Tree validity",
                "Force checkpoint: PRAGMA wal_checkpoint(TRUNCATE)",
                "Verify database returns SQLITE_OK",
            ],
        ),
        (
            "Deploy low-memory SQLite configuration on Raspberry Pi edge gateway",
            [
                "Set PRAGMA cache_size = -64000 (limit RAM cache to 64MB)",
                "Set PRAGMA temp_store = MEMORY",
                "Execute stress-test queries and measure memory footprint with edge_memory_profiler",
                "Confirm peak memory remained under 180MB",
            ],
        ),
    ]

    for goal, steps in trajectories_data:
        tid = memory.record_trajectory(task_goal=goal, steps=steps, success=True)
        console.print(f"  [dim]Recorded Trajectory #{tid} ({len(steps)} steps): \"{goal[:55]}...\"[/dim]")

    console.print()

    # =========================================================================
    # Step 3: Dynamic Procedural Prompt Synthesis for New Task
    # =========================================================================
    Visualizer.print_step(3, "Dynamic Procedural Retrieval for a New Incoming Task")

    new_task = "We need to set up a resilient SQLite database on an edge device without exceeding memory limits."
    console.print(f"[bold yellow]New Task Goal:[/bold yellow] [bold white]\"{new_task}\"[/bold white]\n")

    matched_skills = memory.retrieve_skills(new_task, top_k=2)
    Visualizer.print_generic_table(
        title="Retrieved Procedural Skills (Matched by Embedding Similarity)",
        headers=["Skill Name", "Cosine Sim", "Success Rate", "Invocations", "Description"],
        rows=[
            [s.name, s.similarity, f"{int(s.success_rate * 100)}%", s.invocation_count, s.description]
            for s in matched_skills
        ],
    )

    procedural_prompt = memory.format_procedural_prompt(new_task)
    Visualizer.print_memory_panel(
        procedural_prompt,
        title="Synthesized In-Context Procedural Guidance Block",
        border_style="cyan",
    )

    console.print(
        "[bold cyan]Takeaway:[/bold cyan] Procedural memory stores 'how-to' knowledge. Instead of cluttering the system "
        "prompt with every skill or few-shot example, the agent retrieves only the specific skills and trajectories "
        "relevant to the current task!\n"
    )


if __name__ == "__main__":
    run_demo()
