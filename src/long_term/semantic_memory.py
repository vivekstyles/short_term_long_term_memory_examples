from __future__ import annotations
import sqlite3
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from src.core.embeddings import EmbeddingEngine, get_embedding_engine


class Fact(BaseModel):
    """Represents a discrete semantic piece of knowledge or user profile attribute."""
    id: Optional[int] = None
    entity: str  # e.g., 'user', 'project', 'system'
    fact: str  # e.g., 'User prefers concise responses with code examples'
    category: str = "general"  # 'preference', 'constraint', 'identity', 'world_knowledge'
    confidence: float = 1.0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)
    similarity: Optional[float] = None


class SemanticMemory:
    """
    Level 3: Semantic / Declarative Memory.
    Stores timeless facts, profile data, and domain knowledge persistently in SQLite.
    Employs dense vector cosine similarity for conceptual retrieval alongside structured SQL filtering.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        embedding_engine: Optional[EmbeddingEngine] = None,
    ) -> None:
        self.db_path = db_path
        self.embedding_engine = embedding_engine or get_embedding_engine()
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
                """
            )

    def add_fact(
        self,
        fact: str,
        entity: str = "user",
        category: str = "general",
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Store a new declarative fact with its dense embedding."""
        emb = self.embedding_engine.embed_text(fact)
        created_at = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {})
        emb_json = json.dumps(emb)

        with self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO semantic_facts (entity, fact, category, confidence, created_at, metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (entity, fact, category, confidence, created_at, meta_json, emb_json),
            )
            return cur.lastrowid  # type: ignore

    def search(
        self,
        query: str,
        top_k: int = 3,
        entity: Optional[str] = None,
        category: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> List[Fact]:
        """
        Semantic retrieval:
        1. Encodes query into dense vector.
        2. Filters candidate rows by structured metadata (entity/category) via SQL.
        3. Ranks candidates by cosine similarity.
        """
        query_vec = self.embedding_engine.embed_text(query)

        sql = "SELECT id, entity, fact, category, confidence, created_at, metadata, embedding FROM semantic_facts WHERE 1=1"
        params: List[Any] = []
        if entity:
            sql += " AND entity = ?"
            params.append(entity)
        if category:
            sql += " AND category = ?"
            params.append(category)

        cur = self.conn.cursor()
        rows = cur.execute(sql, params).fetchall()

        scored_facts: List[Fact] = []
        for r in rows:
            emb = json.loads(r["embedding"])
            sim = self.embedding_engine.cosine_similarity(query_vec, emb)
            if sim >= min_similarity:
                fact_obj = Fact(
                    id=r["id"],
                    entity=r["entity"],
                    fact=r["fact"],
                    category=r["category"],
                    confidence=r["confidence"],
                    created_at=r["created_at"],
                    metadata=json.loads(r["metadata"]),
                    similarity=round(sim, 4),
                )
                scored_facts.append(fact_obj)

        scored_facts.sort(key=lambda f: f.similarity or 0.0, reverse=True)
        return scored_facts[:top_k]

    def get_all_facts(self, entity: Optional[str] = None) -> List[Fact]:
        """Fetch all facts, optionally filtered by entity."""
        sql = "SELECT id, entity, fact, category, confidence, created_at, metadata FROM semantic_facts"
        params = []
        if entity:
            sql += " WHERE entity = ?"
            params.append(entity)

        cur = self.conn.cursor()
        rows = cur.execute(sql, params).fetchall()
        return [
            Fact(
                id=r["id"],
                entity=r["entity"],
                fact=r["fact"],
                category=r["category"],
                confidence=r["confidence"],
                created_at=r["created_at"],
                metadata=json.loads(r["metadata"]),
            )
            for r in rows
        ]

    def delete_fact(self, fact_id: int) -> bool:
        with self.conn:
            cur = self.conn.execute("DELETE FROM semantic_facts WHERE id = ?", (fact_id,))
            return cur.rowcount > 0

    def clear(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM semantic_facts")
