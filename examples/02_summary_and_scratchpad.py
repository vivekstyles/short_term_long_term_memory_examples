"""
Level 2: Short-Term Memory — Summary Buffer & Structured Scratchpad.

Demonstrates:
1. Conversation Summary Buffer:
   Instead of permanently discarding history (like a pure sliding window),
   older conversation turns are progressively summarized by the LLM while retaining
   the most recent dialogue verbatim.
2. Structured Working Memory / Scratchpad (Cognitive Blackboard):
   Separating task state (goals, subtasks, runtime variables) from conversational dialogue.
"""
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.message import Message, Role
from src.core.llm_backend import get_llm
from src.short_term.summary_scratchpad import ConversationSummaryBufferMemory, WorkingMemoryScratchpad
from src.visualizer import Visualizer, console


def run_demo() -> None:
    Visualizer.print_header(
        "Level 2: Summary Buffer & Working Scratchpad",
        "Context Compression & Structured Operational State Blackboard",
    )

    llm = get_llm()
    console.print(f"[bold cyan]Using LLM Backend:[/bold cyan] {llm.__class__.__name__}\n")

    # =========================================================================
    # Part 1: Conversation Summary Buffer Memory
    # =========================================================================
    Visualizer.print_step(1, "Conversation Summary Buffer (Progressive Compression)")
    
    # We set max_tokens low (120) and keep_recent_k=2 to illustrate the trigger
    summary_memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_tokens=110,
        keep_recent_k=2,
        system_prompt="You are a specialized AI systems engineer.",
    )

    conversation_script = [
        Message(role=Role.USER, content="Our team is migrating our PostgreSQL database to SQLite for edge deployments."),
        Message(role=Role.ASSISTANT, content="Understood. For edge environments, SQLite requires WAL mode and strict typing configuration."),
        Message(role=Role.USER, content="We also have a hard constraint: memory footprint cannot exceed 256MB on the target device."),
        Message(role=Role.ASSISTANT, content="Noted. We must set cache_size=2000 and avoid in-memory temporary tables."),
        Message(role=Role.USER, content="What was the memory limit constraint I mentioned earlier?"),
        Message(role=Role.ASSISTANT, content="Your hard constraint is that the memory footprint must not exceed 256MB."),
    ]

    for i, msg in enumerate(conversation_script, 1):
        summary_memory.add_message(msg)
        console.print(f"[dim]Turn {i}: Added {msg.role.value} message ({msg.token_count} tokens). Total: {summary_memory.get_total_tokens()} tokens.[/dim]")

    console.print()
    console.print(f"[bold green]Summarization Trigger Count:[/bold green] {summary_memory.summarization_events}")
    
    if summary_memory.summary:
        Visualizer.print_memory_panel(
            summary_memory.summary,
            title="Generated Rolling Summary (Older Turns Condensed)",
            border_style="cyan",
        )

    Visualizer.print_messages_table(
        summary_memory.get_messages(),
        title="Synthesized In-Context Payload (Summary + Recent Verbatim Turns)",
    )

    # =========================================================================
    # Part 2: Structured Working Memory / Scratchpad (Blackboard)
    # =========================================================================
    Visualizer.print_step(2, "Structured Working Memory / Scratchpad (Cognitive Blackboard)")
    console.print(
        "[dim]Agents must maintain intermediate state (variables, subtask progress) "
        "independent of the conversational chat stream.[/dim]\n"
    )

    scratchpad = WorkingMemoryScratchpad()
    scratchpad.set_goal("Migrate customer database to SQLite edge instance with WAL mode")
    
    t1 = scratchpad.add_subtask("Analyze source schema and column types")
    t2 = scratchpad.add_subtask("Enable PRAGMA journal_mode = WAL")
    t3 = scratchpad.add_subtask("Run memory benchmark test (<256MB limit)")

    # Simulate agent executing steps
    scratchpad.update_subtask_status(t1, "completed")
    scratchpad.update_subtask_status(t2, "in_progress")
    scratchpad.store_variable("source_table_count", 14)
    scratchpad.store_variable("target_db_path", "/var/edge/app_data.db")
    scratchpad.add_note("Verified WAL mode creates -wal and -shm files successfully.")

    rendered_blackboard = scratchpad.render_context()
    Visualizer.print_memory_panel(
        rendered_blackboard,
        title="In-Context Working Memory Blackboard (Injected into Agent Prompt)",
        border_style="magenta",
    )

    console.print(
        "[bold cyan]Takeaway:[/bold cyan] Summary Buffers compress historical dialogue to save context tokens, "
        "while Working Scratchpads prevent the agent from losing track of multi-step goals and discovered variables!\n"
    )


if __name__ == "__main__":
    run_demo()
