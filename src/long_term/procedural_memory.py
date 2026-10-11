from __future__ import annotations
import sqlite3
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from src.core.embeddings import EmbeddingEngine, get_embedding_engine


class Skill(BaseModel):
    name: str
    description: str
    trigger_pattern: str
    code_recipe: str
    success_rate: float = 1.0
    invocation_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    similarity: Optional[float] = None


class Trajectory(BaseModel):
    id: Optional[int] = None
    task_goal: str
    steps: List[str]
    success: bool = True
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    similarity: Optional[float] = None


class ProceduralMemory:
    """
    Level 5: Procedural Memory (Skills, Recipes, and Few-Shot Trajectories).
    Encodes 'how-to' operational knowledge:
    1. Dynamic Skill Library: reusable task recipes and tool calling sequences (Voyager agent pattern).
    2. Trajectory Exemplars: successful execution traces dynamically recalled as few-shot in-context examples.
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
                CREATE TABLE IF NOT EXISTS procedural_skills (
                    name TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    trigger_pattern TEXT NOT NULL,
                    code_recipe TEXT NOT NULL,
                    success_count INTEGER NOT NULL,
                    invocation_count INTEGER NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_trajectories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_goal TEXT NOT NULL,
                    steps TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
                """
            )

    def register_skill(
        self,
        name: str,
        description: str,
        trigger_pattern: str,
        code_recipe: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register or update a procedural skill."""
        emb_text = f"{name} {description} {trigger_pattern}"
        emb = self.embedding_engine.embed_text(emb_text)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO procedural_skills 
                (name, description, trigger_pattern, code_recipe, success_count, invocation_count, metadata, embedding)
                VALUES (?, ?, ?, ?, 1, 1, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description=excluded.description,
                    trigger_pattern=excluded.trigger_pattern,
                    code_recipe=excluded.code_recipe,
                    embedding=excluded.embedding
                """,
                (name, description, trigger_pattern, code_recipe, json.dumps(metadata or {}), json.dumps(emb)),
            )

    def record_skill_execution(self, name: str, success: bool) -> None:
        """Update success tracking metrics for a procedural skill."""
        with self.conn:
            cur = self.conn.execute(
                "SELECT success_count, invocation_count FROM procedural_skills WHERE name = ?", (name,)
            ).fetchone()
            if cur:
                succ = cur["success_count"] + (1 if success else 0)
                invoc = cur["invocation_count"] + 1
                self.conn.execute(
                    "UPDATE procedural_skills SET success_count = ?, invocation_count = ? WHERE name = ?",
                    (succ, invoc, name),
                )

    def retrieve_skills(self, task_description: str, top_k: int = 2) -> List[Skill]:
        """Retrieve most relevant procedural skills for a given task description."""
        query_vec = self.embedding_engine.embed_text(task_description)
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT name, description, trigger_pattern, code_recipe, success_count, invocation_count, metadata, embedding FROM procedural_skills"
        ).fetchall()

        skills: List[Skill] = []
        for r in rows:
            emb = json.loads(r["embedding"])
            sim = self.embedding_engine.cosine_similarity(query_vec, emb)
            succ = r["success_count"]
            invoc = max(1, r["invocation_count"])
            rate = round(succ / invoc, 2)
            skills.append(
                Skill(
                    name=r["name"],
                    description=r["description"],
                    trigger_pattern=r["trigger_pattern"],
                    code_recipe=r["code_recipe"],
                    success_rate=rate,
                    invocation_count=invoc,
                    metadata=json.loads(r["metadata"]),
                    similarity=round(sim, 4),
                )
            )

        skills.sort(key=lambda s: s.similarity or 0.0, reverse=True)
        return skills[:top_k]

    def record_trajectory(self, task_goal: str, steps: List[str], success: bool = True) -> int:
        """Store a full execution trajectory trace as an exemplar for future few-shot retrieval."""
        emb = self.embedding_engine.embed_text(task_goal)
        now = datetime.now(timezone.utc).isoformat()
        with self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO execution_trajectories (task_goal, steps, success, created_at, embedding)
                VALUES (?, ?, ?, ?, ?)
                """,
                (task_goal, json.dumps(steps), 1 if success else 0, now, json.dumps(emb)),
            )
            return cur.lastrowid  # type: ignore

    def retrieve_trajectories(self, task_goal: str, top_k: int = 2) -> List[Trajectory]:
        """Find past successful trajectories matching a new incoming task."""
        query_vec = self.embedding_engine.embed_text(task_goal)
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT id, task_goal, steps, success, created_at, embedding FROM execution_trajectories WHERE success = 1"
        ).fetchall()

        trajectories: List[Trajectory] = []
        for r in rows:
            emb = json.loads(r["embedding"])
            sim = self.embedding_engine.cosine_similarity(query_vec, emb)
            trajectories.append(
                Trajectory(
                    id=r["id"],
                    task_goal=r["task_goal"],
                    steps=json.loads(r["steps"]),
                    success=bool(r["success"]),
                    created_at=r["created_at"],
                    similarity=round(sim, 4),
                )
            )

        trajectories.sort(key=lambda t: t.similarity or 0.0, reverse=True)
        return trajectories[:top_k]

    def format_procedural_prompt(self, task_goal: str) -> str:
        """Synthesize relevant skills and few-shot trajectories into a guidance block for the agent."""
        matched_skills = self.retrieve_skills(task_goal, top_k=2)
        matched_trajectories = self.retrieve_trajectories(task_goal, top_k=1)

        sections = ["[PROCEDURAL MEMORY & LEARNED WORKFLOWS]"]
        if matched_skills:
            sections.append("Relevant Reusable Skills:")
            for s in matched_skills:
                sections.append(f"• Skill '{s.name}' (Success rate: {int(s.success_rate * 100)}%):")
                sections.append(f"  Description: {s.description}")
                sections.append(f"  Workflow Recipe:\n{s.code_recipe.strip()}")

        if matched_trajectories:
            sections.append("\nPast Successful Execution Exemplar (Few-Shot Reference):")
            for t in matched_trajectories:
                sections.append(f"• Goal: {t.task_goal}")
                for idx, step in enumerate(t.steps, 1):
                    sections.append(f"    Step {idx}: {step}")

        return "\n".join(sections)
