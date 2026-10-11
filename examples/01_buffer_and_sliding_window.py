"""
Level 1: Short-Term Memory — Buffer and Sliding Window.

Demonstrates:
1. Naive Full Buffer (grows unboundedly, threatening token limits).
2. Token-Bounded Sliding Window (FIFO eviction of oldest messages).
3. Anchor Preservation (System instructions & pinned messages are NEVER evicted).
"""
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.message import Message, Role
from src.short_term.buffer_window import ConversationBufferMemory, SlidingWindowMemory
from src.visualizer import Visualizer, console


def run_demo() -> None:
    Visualizer.print_header(
        "Level 1: Buffer and Sliding Window Memory",
        "Short-Term Working Memory Management & Token Budgeting",
    )

    # 1. Naive Buffer Demonstration
    Visualizer.print_step(1, "Naive Buffer (Unbounded Growth)")
    naive_buffer = ConversationBufferMemory()

    messages_to_add = [
        Message(role=Role.SYSTEM, content="You are a senior Python architecture assistant. Always follow PEP-8."),
        Message(role=Role.USER, content="Hello! We are planning to build an agentic memory system."),
        Message(role=Role.ASSISTANT, content="Excellent. In agentic systems, memory is split into short-term working context and long-term persistent storage."),
        Message(role=Role.USER, content="Can you explain what happens if we store all messages forever in the prompt?"),
        Message(role=Role.ASSISTANT, content="Every LLM has a finite context window. Without pruning, token count explodes, inference costs surge, and older context gets truncated or ignored."),
    ]

    for msg in messages_to_add:
        naive_buffer.add_message(msg)

    Visualizer.print_messages_table(
        naive_buffer.get_messages(),
        title="Naive Buffer: All Messages Retained",
    )

    # 2. Token-Bounded Sliding Window with Anchor Protection
    Visualizer.print_step(2, "Sliding Window with Token Budget & Anchor Preservation")
    TOKEN_BUDGET = 75  # Deliberately tight budget to trigger sliding
    sliding_window = SlidingWindowMemory(max_tokens=TOKEN_BUDGET, preserve_anchors=True)

    console.print(f"[bold cyan]Configured Token Budget:[/bold cyan] {TOKEN_BUDGET} tokens.\n")

    for i, msg in enumerate(messages_to_add, 1):
        console.print(f"[yellow]Adding Message {i} ({msg.role.value}):[/yellow] \"{msg.content[:50]}...\"")
        sliding_window.add_message(msg)
        current_tokens = sliding_window.get_total_tokens()
        console.print(f"  Current active tokens: [cyan]{current_tokens}[/cyan] / {TOKEN_BUDGET}")

    console.print()
    Visualizer.print_messages_table(
        sliding_window.get_messages(),
        title=f"Sliding Window Context (Budget: {TOKEN_BUDGET} tokens)",
        token_budget=TOKEN_BUDGET,
    )

    evicted = sliding_window.get_evicted_messages()
    if evicted:
        Visualizer.print_eviction_notice(
            evicted_count=len(evicted),
            freed_tokens=sum(m.token_count for m in evicted),
            reason=f"Context exceeded {TOKEN_BUDGET} token budget. Oldest non-anchor messages were evicted (FIFO).",
        )
        Visualizer.print_messages_table(
            evicted,
            title="Evicted Messages (Lost from Short-Term Context)",
        )

    # Key Takeaway check
    anchors_intact = any(m.role == Role.SYSTEM for m in sliding_window.get_messages())
    console.print(
        f"[bold green][OK] Anchor Check:[/bold green] System prompt remained intact: "
        f"[bold {'green' if anchors_intact else 'red'}]{anchors_intact}[/bold {'green' if anchors_intact else 'red'}]."
    )
    console.print(
        "[bold cyan]Takeaway:[/bold cyan] Sliding windows prevent Out-Of-Memory/Context Overflow errors, "
        "but naively evicted messages are permanently lost unless consolidated into long-term memory!\n"
    )


if __name__ == "__main__":
    run_demo()
