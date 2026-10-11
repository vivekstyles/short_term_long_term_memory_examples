from __future__ import annotations
import sqlite3
import json
import math
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field

from src.core.embeddings import EmbeddingEngine, get_embedding_engine
from src.core.llm_backend import LLMBackend, get_llm


class Episode(BaseModel):
    id: Optional[int] = None
    description: str
    memory_type: str = "observation"  # "observation", "action", "reflection"
    importance_score: float = 5.0  # 1.0 to 10.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScoredEpisode(BaseModel):
    episode: Episode
    recency_score: float
    importance_score: float
    relevance_score: float
    total_score: float


class EpisodicMemory:
    """
    Level 4: Episodic Memory (Experience Stream).
    Based on the Stanford Generative Agents cognitive architecture (Park et al., 2023).
    
    Each episode represents a discrete past event, tool execution, or reflection.
    Retrieval scores memories via a linear combination of three cognitive dimensions:
        Score = w_rec * Recency + w_imp * Importance + w_rel * Relevance
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        embedding_engine: Optional[EmbeddingEngine] = None,
        llm: Optional[LLMBackend] = None,
        decay_factor: float = 0.99,  # Decay rate per hour
    ) -> None:
        self.db_path = db_path
        self.embedding_engine = embedding_engine or get_embedding_engine()
        self.llm = llm or get_llm()
        self.decay_factor = decay_factor
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_stream (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
                """
            )

    def record_episode(
        self,
        description: str,
        memory_type: str = "observation",
        importance_score: Optional[float] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record an event into the episodic memory stream."""
        if importance_score is None:
            importance_score = self._rate_importance(description)

        now = created_at or datetime.now(timezone.utc)
        emb = self.embedding_engine.embed_text(description)

        with self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO episodic_stream 
                (description, memory_type, importance_score, created_at, last_accessed_at, metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    description,
                    memory_type,
                    float(importance_score),
                    now.isoformat(),
                    now.isoformat(),
                    json.dumps(metadata or {}),
                    json.dumps(emb),
                ),
            )
            return cur.lastrowid  # type: ignore

    def _rate_importance(self, description: str) -> float:
        """Call LLM or heuristic to rate the importance of an episodic event on a 1-10 scale."""
        prompt = (
            f"On the scale of 1 to 10, where 1 is purely mundane (e.g., 'said hello') "
            f"and 10 is critically impactful (e.g., 'database destroyed', 'promoted to lead', 'discovered major bug'), "
            f"rate the likely importance of this memory:\n\nMemory: \"{description}\"\n\n"
            f"Respond ONLY with a single integer number between 1 and 10."
        )
        try:
            resp = self.llm.generate(prompt=prompt, system="You are an objective memory importance evaluator.")
            # Extract first digit found
            digits = "".join(c for c in resp if c.isdigit())
            if digits:
                score = float(digits[:2] if int(digits[:2]) <= 10 else digits[0])
                return min(max(score, 1.0), 10.0)
        except Exception:
            pass
        return 5.0

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        w_rec: float = 1.0,
        w_imp: float = 1.0,
        w_rel: float = 1.0,
        current_time: Optional[datetime] = None,
    ) -> List[ScoredEpisode]:
        """
        Stanford Generative Agents retrieval algorithm:
        Score = w_rec * Recency + w_imp * Importance + w_rel * Relevance
        """
        now = current_time or datetime.now(timezone.utc)
        query_vec = self.embedding_engine.embed_text(query)

        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT id, description, memory_type, importance_score, created_at, last_accessed_at, metadata, embedding FROM episodic_stream"
        ).fetchall()

        if not rows:
            return []

        scored_list: List[ScoredEpisode] = []

        for r in rows:
            # 1. Recency calculation (exponential decay based on hours elapsed)
            last_access = datetime.fromisoformat(r["last_accessed_at"])
            hours_elapsed = max(0.0, (now - last_access).total_seconds() / 3600.0)
            recency = math.pow(self.decay_factor, hours_elapsed)

            # 2. Importance calculation (normalized to [0, 1])
            importance = float(r["importance_score"]) / 10.0

            # 3. Relevance calculation (cosine similarity normalized to [0, 1])
            emb = json.loads(r["embedding"])
            relevance = max(0.0, min(1.0, (self.embedding_engine.cosine_similarity(query_vec, emb) + 1.0) / 2.0))

            total_score = (w_rec * recency) + (w_imp * importance) + (w_rel * relevance)

            ep = Episode(
                id=r["id"],
                description=r["description"],
                memory_type=r["memory_type"],
                importance_score=r["importance_score"],
                created_at=datetime.fromisoformat(r["created_at"]),
                last_accessed_at=last_access,
                metadata=json.loads(r["metadata"]),
            )

            scored_list.append(
                ScoredEpisode(
                    episode=ep,
                    recency_score=round(recency, 4),
                    importance_score=round(importance, 4),
                    relevance_score=round(relevance, 4),
                    total_score=round(total_score, 4),
                )
            )

        # Sort descending by total score
        scored_list.sort(key=lambda s: s.total_score, reverse=True)
        top_results = scored_list[:top_k]

        # Update last_accessed_at for retrieved episodes (re-activating them)
        with self.conn:
            for scored in top_results:
                if scored.episode.id:
                    self.conn.execute(
                        "UPDATE episodic_stream SET last_accessed_at = ? WHERE id = ?",
                        (now.isoformat(), scored.episode.id),
                    )

        return top_results

    def reflect(self, current_time: Optional[datetime] = None) -> List[str]:
        """
        Synthesizes high-level reflections from the most recent episodes.
        Reflections are higher-abstraction insights added back into the memory stream.
        """
        now = current_time or datetime.now(timezone.utc)
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT description FROM episodic_stream ORDER BY id DESC LIMIT 10"
        ).fetchall()

        if len(rows) < 3:
            return []

        recent_events = "\n".join(f"- {r['description']}" for r in rows)
        prompt = (
            f"Here is a chronological list of recent events experienced by the agent:\n{recent_events}\n\n"
            "Task: Formulate 1 to 2 high-level insights or reflections about the patterns, user needs, or learnings. "
            "Respond in 1-2 bullet points."
        )

        insights_text = self.llm.generate(prompt=prompt, system="You are a reflective cognitive agent synthesizer.")
        insights: List[str] = [
            line.strip().lstrip("-").strip()
            for line in insights_text.split("\n")
            if line.strip() and not line.strip().lower().startswith("insight:")
        ]

        if not insights and insights_text:
            insights = [insights_text.strip()]

        # Record reflections into stream with high default importance (8.0)
        for insight in insights:
            self.record_episode(
                description=insight,
                memory_type="reflection",
                importance_score=8.5,
                created_at=now,
                metadata={"source": "reflection_loop"},
            )

        return insights
