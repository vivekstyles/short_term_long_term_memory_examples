from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
from src.core.message import Message, Role


class BaseShortTermMemory(ABC):
    """Abstract base class for in-context working short-term memory."""

    @abstractmethod
    def add_message(self, message: Message) -> None:
        """Record a new message turn."""
        pass

    @abstractmethod
    def get_messages(self) -> List[Message]:
        """Retrieve the active message list for context generation."""
        pass

    @abstractmethod
    def get_total_tokens(self) -> int:
        """Calculate total tokens currently consumed in this working memory."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Reset the working memory."""
        pass


class ConversationBufferMemory(BaseShortTermMemory):
    """
    Level 1a: Full message buffer memory.
    Stores all messages sequentially without modification or truncation.
    Prone to Context Window Overflow as conversations grow.
    """

    def __init__(self) -> None:
        self.messages: List[Message] = []

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def get_messages(self) -> List[Message]:
        return list(self.messages)

    def get_total_tokens(self) -> int:
        return sum(m.token_count for m in self.messages)

    def clear(self) -> None:
        self.messages.clear()


class SlidingWindowMemory(BaseShortTermMemory):
    """
    Level 1b: Sliding Window Memory.
    Enforces a strict bounded context window by:
    1. Maximum turns (k_turns)
    2. Maximum token budget (max_tokens)
    
    Crucial Feature: Anchor Preservation.
    System prompt messages and messages flagged with metadata `pinned=True`
    are anchored and NEVER pruned, preventing the agent from losing its fundamental identity/instructions.
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        k_turns: Optional[int] = None,
        preserve_anchors: bool = True,
    ) -> None:
        self.max_tokens = max_tokens
        self.k_turns = k_turns
        self.preserve_anchors = preserve_anchors
        self.messages: List[Message] = []
        self.evicted_history: List[Message] = []

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        self._prune()

    def get_messages(self) -> List[Message]:
        return list(self.messages)

    def get_evicted_messages(self) -> List[Message]:
        return list(self.evicted_history)

    def get_total_tokens(self) -> int:
        return sum(m.token_count for m in self.messages)

    def clear(self) -> None:
        self.messages.clear()
        self.evicted_history.clear()

    def _is_anchor(self, message: Message) -> bool:
        if not self.preserve_anchors:
            return False
        return message.role == Role.SYSTEM or message.metadata.get("pinned", False)

    def _prune(self) -> None:
        """Evict oldest non-anchor messages until constraints are satisfied."""
        # 1. Prune by turn count if specified
        if self.k_turns is not None:
            # Non-anchor messages
            non_anchors = [m for m in self.messages if not self._is_anchor(m)]
            while len(non_anchors) > self.k_turns:
                # Find index of the first non-anchor
                for idx, m in enumerate(self.messages):
                    if not self._is_anchor(m):
                        evicted = self.messages.pop(idx)
                        self.evicted_history.append(evicted)
                        non_anchors.pop(0)
                        break

        # 2. Prune by token budget if specified
        if self.max_tokens is not None:
            while self.get_total_tokens() > self.max_tokens:
                # Check if there is any non-anchor message to evict
                evicted_any = False
                for idx, m in enumerate(self.messages):
                    if not self._is_anchor(m):
                        evicted = self.messages.pop(idx)
                        self.evicted_history.append(evicted)
                        evicted_any = True
                        break

                # If only anchors remain and budget is exceeded, cannot prune further without breaking instructions
                if not evicted_any:
                    break
