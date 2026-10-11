from datetime import datetime, timezone, timedelta
from src.core.llm_backend import MockLLM
from src.long_term.semantic_memory import SemanticMemory
from src.long_term.episodic_memory import EpisodicMemory
from src.long_term.procedural_memory import ProceduralMemory


def test_semantic_memory_crud_and_search():
    mem = SemanticMemory(db_path=":memory:")
    f1 = mem.add_fact("User prefers dark mode syntax theme", entity="user", category="preference")
    f2 = mem.add_fact("Application requires Python 3.12", entity="project", category="constraint")

    assert f1 is not None and f2 is not None

    all_facts = mem.get_all_facts()
    assert len(all_facts) == 2

    # Semantic search
    results = mem.search("color scheme and dark theme", top_k=1)
    assert len(results) == 1
    assert "dark mode" in results[0].fact
    assert results[0].similarity is not None and results[0].similarity > 0

    # Structured SQL filtering
    project_facts = mem.search("configuration", category="constraint", top_k=5)
    assert len(project_facts) == 1
    assert project_facts[0].category == "constraint"

    # Deletion
    assert mem.delete_fact(f1) is True
    assert len(mem.get_all_facts()) == 1


def test_episodic_memory_scoring_and_reflection():
    llm = MockLLM()
    mem = EpisodicMemory(db_path=":memory:", llm=llm, decay_factor=0.9)

    now = datetime.now(timezone.utc)
    old_critical_id = mem.record_episode(
        description="Production database crashed due to disk overflow",
        importance_score=9.5,
        created_at=now - timedelta(days=2),
    )
    recent_mundane_id = mem.record_episode(
        description="User drank hot tea",
        importance_score=1.5,
        created_at=now - timedelta(minutes=10),
    )

    # Querying about database failure should rank the old critical episode highest
    results = mem.retrieve("database crash and failure", top_k=2, current_time=now)
    assert len(results) == 2
    assert results[0].episode.id == old_critical_id
    assert results[0].total_score > results[1].total_score

    # Reflection loop
    mem.record_episode("Verified disk metrics", importance_score=5.0, created_at=now)
    reflections = mem.reflect(current_time=now)
    assert len(reflections) > 0


def test_procedural_memory():
    mem = ProceduralMemory(db_path=":memory:")
    mem.register_skill(
        name="test_skill",
        description="A skill for testing database connections",
        trigger_pattern="database connect test verify",
        code_recipe="def connect(): pass",
    )

    mem.record_skill_execution("test_skill", success=True)
    skills = mem.retrieve_skills("how to connect to database", top_k=1)
    assert len(skills) == 1
    assert skills[0].name == "test_skill"
    assert skills[0].success_rate == 1.0

    # Trajectory
    tid = mem.record_trajectory("Fix lock", ["Step 1", "Step 2"], success=True)
    assert tid > 0
    trajs = mem.retrieve_trajectories("Fix lock issue", top_k=1)
    assert len(trajs) == 1
    assert len(trajs[0].steps) == 2
