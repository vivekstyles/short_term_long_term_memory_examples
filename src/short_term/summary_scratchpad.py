from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from src.core.message import Message, Role, count_tokens
from src.core.llm_backend import LLMBackend, get_llm
from src.short_term.buffer_window import BaseShortTermMemory


class ConversationSummaryBufferMemory(BaseShortTermMemory):
    """
    Level 2a: Conversation Summary Buffer Memory.
    Instead of permanently losing information through eviction (as in a naive sliding window),
    older message turns are iteratively condensed into an ongoing structured summary
    while keeping the most recent K message turns completely verbatim.
    """

    def __init__(
        self,
        llm: Optional[LLMBackend] = None,
        max_tokens: int = 400,
        keep_recent_k: int = 4,
        system_prompt: str = "You are a helpful and knowledgeable AI agent.",
    ) -> None:
        self.llm = llm or get_llm()
        self.max_tokens = max_tokens
        self.keep_recent_k = keep_recent_k
        self.system_prompt = system_prompt
        self.summary: str = ""
        self.raw_messages: List[Message] = []
        self.summarization_events: int = 0

    def add_message(self, message: Message) -> None:
        self.raw_messages.append(message)
        self._check_and_summarize()

    def get_messages(self) -> List[Message]:
        """
        Synthesizes the complete prompt payload:
        1. Base system prompt augmented with the progressive summary (if any).
        2. Recent raw messages preserved verbatim.
        """
        combined_system = self.system_prompt
        if self.summary:
            combined_system += (
                f"\n\n[COMPACT SUMMARY OF PREVIOUS CONVERSATION TURNS]:\n{self.summary}"
            )

        output: List[Message] = [Message(role=Role.SYSTEM, content=combined_system)]
        output.extend(self.raw_messages)
        return output

    def get_total_tokens(self) -> int:
        summary_tokens = count_tokens(self.summary) if self.summary else 0
        sys_tokens = count_tokens(self.system_prompt)
        raw_tokens = sum(m.token_count for m in self.raw_messages)
        return sys_tokens + summary_tokens + raw_tokens

    def clear(self) -> None:
        self.summary = ""
        self.raw_messages.clear()
        self.summarization_events = 0

    def _check_and_summarize(self) -> None:
        """Trigger summarization if total tokens exceed max_tokens and we have enough messages to split."""
        if self.get_total_tokens() <= self.max_tokens or len(self.raw_messages) <= self.keep_recent_k:
            return

        # Split messages: older ones to condense, newest K to retain verbatim
        split_idx = len(self.raw_messages) - self.keep_recent_k
        to_summarize = self.raw_messages[:split_idx]
        self.raw_messages = self.raw_messages[split_idx:]

        # Format transcript of older messages for LLM summarization
        transcript = "\n".join(f"{m.role.value}: {m.content}" for m in to_summarize)
        prompt = (
            f"Existing Summary:\n{self.summary or 'None'}\n\n"
            f"New Messages to condense:\n{transcript}\n\n"
            "Task: Update and synthesize the existing summary with the new messages into a concise, "
            "information-dense summary preserving key facts, user preferences, and task status. "
            "Summary:"
        )

        new_summary = self.llm.generate(prompt=prompt, system="You are an expert conversation summarizer.")
        self.summary = new_summary.strip()
        self.summarization_events += 1


class Subtask(BaseModel):
    id: int
    description: str
    status: str = "pending"  # "pending", "in_progress", "completed"


class WorkingMemoryScratchpad(BaseModel):
    """
    Level 2b: Structured Working Memory / Scratchpad (Cognitive Blackboard).
    Separates the conversation transcript from the agent's internal operational state.
    Maintains active goals, subtask progress, and intermediate variables discovered during tool use.
    """
    active_goal: str = ""
    subtasks: List[Subtask] = Field(default_factory=list)
    variables: Dict[str, Any] = Field(default_factory=dict)
    scratchpad_notes: List[str] = Field(default_factory=list)

    def set_goal(self, goal: str) -> None:
        self.active_goal = goal

    def add_subtask(self, description: str) -> int:
        new_id = len(self.subtasks) + 1
        self.subtasks.append(Subtask(id=new_id, description=description, status="pending"))
        return new_id

    def update_subtask_status(self, task_id: int, status: str) -> None:
        for t in self.subtasks:
            if t.id == task_id:
                t.status = status
                break

    def store_variable(self, key: str, value: Any) -> None:
        self.variables[key] = value

    def add_note(self, note: str) -> None:
        self.scratchpad_notes.append(note)

    def render_context(self) -> str:
        """Render scratchpad state as an in-context system block for the LLM."""
        lines = ["[AGENT WORKING SCRATCHPAD & OPERATIONAL STATE]"]
        if self.active_goal:
            lines.append(f"Primary Goal: {self.active_goal}")

        if self.subtasks:
            lines.append("Plan & Subtasks:")
            for t in self.subtasks:
                icon = "[OK]" if t.status == "completed" else "[>]" if t.status == "in_progress" else "[ ]"
                lines.append(f"  {icon} ({t.status.upper()}) #{t.id}: {t.description}")

        if self.variables:
            lines.append("Discovered Variables / State:")
            for k, v in self.variables.items():
                lines.append(f"  • {k}: {v}")

        if self.scratchpad_notes:
            lines.append("Internal Scratchpad Notes:")
            for note in self.scratchpad_notes[-5:]:
                lines.append(f"  - {note}")

        return "\n".join(lines)
