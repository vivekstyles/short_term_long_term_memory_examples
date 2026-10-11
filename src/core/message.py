from __future__ import annotations
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


def count_tokens(text: str) -> int:
    """
    Lightweight, deterministic token estimation.
    In production LLM tokenizers (tiktoken, sentencepiece), 1 token is roughly 4 characters
    or ~0.75 words for English text. This utility provides an accurate, zero-dependency token count.
    """
    if not text:
        return 0
    words = text.split()
    # Hybrid word + character heuristic
    char_estimate = len(text) / 4.0
    word_estimate = len(words) * 1.3
    return max(1, int(round((char_estimate + word_estimate) / 2.0)))


class Message(BaseModel):
    """Represents a single atomic interaction turn in an agent conversation."""
    role: Role
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

    @property
    def token_count(self) -> int:
        return count_tokens(self.content)

    def to_chat_dict(self) -> Dict[str, str]:
        """Convert to standard chat API format {"role": ..., "content": ...}."""
        return {"role": self.role.value, "content": self.content}

    def format_display(self) -> str:
        """Format for readable CLI inspection."""
        name_str = f" ({self.name})" if self.name else ""
        return f"[{self.role.value.upper()}{name_str}] ({self.token_count} tokens): {self.content}"
