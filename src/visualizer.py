from __future__ import annotations
import sys
from typing import List, Optional, Any, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from src.core.message import Message, Role

# Ensure safe UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(safe_box=True)


class Visualizer:
    """Rich console visualizer for memory inspections, diffs, and cognitive traces."""

    @staticmethod
    def print_header(title: str, subtitle: Optional[str] = None) -> None:
        grid = Table.grid(expand=True)
        grid.add_column(justify="center")
        grid.add_row(Text(f"=== {title.upper()} ===", style="bold cyan"))
        if subtitle:
            grid.add_row(Text(subtitle, style="italic dim white"))
        console.print()
        console.print(Panel(grid, box=box.ROUNDED, border_style="cyan"))
        console.print()

    @staticmethod
    def print_step(step_num: int, title: str, details: Optional[str] = None) -> None:
        console.print(f"[bold yellow]Step {step_num}:[/bold yellow] [bold white]{title}[/bold white]")
        if details:
            console.print(f"  [dim]{details}[/dim]")

    @staticmethod
    def print_messages_table(
        messages: List[Message],
        title: str = "Active Messages in Context Window",
        token_budget: Optional[int] = None,
    ) -> None:
        table = Table(title=title, box=box.SIMPLE_HEAVY, header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("Role", style="bold", width=12)
        table.add_column("Content Snippet", style="white", ratio=1)
        table.add_column("Tokens", justify="right", style="cyan", width=8)

        total_tokens = sum(m.token_count for m in messages)

        role_colors = {
            Role.SYSTEM: "red",
            Role.USER: "green",
            Role.ASSISTANT: "blue",
            Role.TOOL: "yellow",
        }

        for i, msg in enumerate(messages, 1):
            color = role_colors.get(msg.role, "white")
            snippet = msg.content if len(msg.content) <= 80 else msg.content[:77] + "..."
            table.add_row(
                str(i),
                f"[{color}]{msg.role.value.upper()}[/{color}]",
                snippet,
                str(msg.token_count),
            )

        budget_str = f" / {token_budget}" if token_budget else ""
        caption = f"Total Tokens in Window: {total_tokens}{budget_str}"
        if token_budget and total_tokens > token_budget:
            caption += " [bold red](BUDGET EXCEEDED!)[/bold red]"
        else:
            caption += " [bold green](Within Budget)[/bold green]"

        table.caption = caption
        console.print(table)
        console.print()

    @staticmethod
    def print_eviction_notice(evicted_count: int, freed_tokens: int, reason: str) -> None:
        console.print(
            Panel(
                f"[bold red]Evicted {evicted_count} message(s)[/bold red] to reclaim "
                f"[bold cyan]{freed_tokens} tokens[/bold cyan].\n[dim]Reason: {reason}[/dim]",
                title="Context Window Sliding Action",
                border_style="red",
                box=box.ROUNDED,
            )
        )
        console.print()

    @staticmethod
    def print_memory_panel(content: str, title: str, border_style: str = "green") -> None:
        console.print(Panel(content, title=title, border_style=border_style, box=box.ROUNDED))
        console.print()

    @staticmethod
    def print_generic_table(title: str, headers: List[str], rows: List[List[Any]]) -> None:
        table = Table(title=title, box=box.ROUNDED, header_style="bold cyan")
        for h in headers:
            table.add_column(h)
        for r in rows:
            table.add_row(*[str(item) for item in r])
        console.print(table)
        console.print()
