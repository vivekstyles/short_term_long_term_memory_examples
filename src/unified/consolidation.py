from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.core.message import Message, Role
from src.core.llm_backend import LLMBackend, get_llm
from src.short_term.buffer_window import SlidingWindowMemory
from src.long_term.semantic_memory import SemanticMemory
from src.long_term.episodic_memory import EpisodicMemory


class ConsolidationReport(BaseModel):
    """Summary report detailing what was consolidated from short-term to long-term memory."""
    extracted_facts: List[str] = Field(default_factory=list)
    recorded_episodes: List[str] = Field(default_factory=list)
    tokens_freed: int = 0
    summary: str = ""


class MemoryConsolidator:
    """
    Cognitive Memory Consolidation Engine ('Sleep Phase').
    Simulates the biological/cognitive process of transferring volatile working experiences
    from the short-term hippocampal buffer into durable neocortical long-term storage
    (Semantic & Episodic stores), resolving contradictions and pruning transient noise.
    """

    def __init__(
        self,
        semantic_memory: SemanticMemory,
        episodic_memory: EpisodicMemory,
        llm: Optional[LLMBackend] = None,
    ) -> None:
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory
        self.llm = llm or get_llm()

    def consolidate(
        self,
        buffer: SlidingWindowMemory,
        clear_buffer_after: bool = True,
    ) -> ConsolidationReport:
        """
        Execute memory consolidation:
        1. Reads uncompressed short-term messages.
        2. Prompts LLM to extract durable facts and episodic events.
        3. Persists them into SQLite SemanticMemory & EpisodicMemory.
        4. Cleanses or compresses the short-term buffer.
        """
        raw_messages = [m for m in buffer.get_messages() if m.role in (Role.USER, Role.ASSISTANT)]
        if not raw_messages:
            return ConsolidationReport(summary="No messages to consolidate.")

        initial_tokens = buffer.get_total_tokens()
        transcript = "\n".join(f"{m.role.value}: {m.content}" for m in raw_messages)

        # 1. Extract Semantic Facts (User preferences, rules, identities)
        fact_prompt = (
            f"Review this conversation transcript:\n{transcript}\n\n"
            "Task: Extract any permanent user preferences, project constraints, or factual knowledge "
            "that should be retained for future sessions. Format each fact as a bullet point starting with '- '."
        )
        facts_text = self.llm.generate(prompt=fact_prompt, system="You are an expert knowledge extraction agent.")
        extracted_facts: List[str] = []
        for line in facts_text.split("\n"):
            cleaned = line.strip().lstrip("-* ").strip()
            if cleaned and not cleaned.lower().startswith("fact:") and len(cleaned) > 5:
                extracted_facts.append(cleaned)
                self.semantic_memory.add_fact(
                    fact=cleaned,
                    entity="user",
                    category="consolidated_preference",
                    confidence=0.95,
                    metadata={"source": "consolidation_sleep_cycle"},
                )

        # 2. Extract Episodic Event Summary (High-level narrative experience)
        episode_prompt = (
            f"Review this conversation transcript:\n{transcript}\n\n"
            "Task: Summarize the primary activity or event that occurred in this session in 1 single sentence. "
            "Describe it as an episode experienced by the agent."
        )
        episode_summary = self.llm.generate(prompt=episode_prompt, system="You are an episodic memory summarizer.")
        clean_episode = episode_summary.strip().replace("\n", " ")

        self.episodic_memory.record_episode(
            description=clean_episode,
            memory_type="action",
            importance_score=7.0,
            metadata={"source": "consolidation_sleep_cycle"},
        )
        recorded_episodes = [clean_episode]

        # 3. Prune/Clear working buffer
        if clear_buffer_after:
            buffer.clear()
            tokens_freed = initial_tokens
        else:
            tokens_freed = 0

        return ConsolidationReport(
            extracted_facts=extracted_facts,
            recorded_episodes=recorded_episodes,
            tokens_freed=tokens_freed,
            summary=f"Consolidated {len(extracted_facts)} facts and 1 episode. Reclaimed {tokens_freed} working tokens.",
        )
