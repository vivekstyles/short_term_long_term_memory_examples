from src.core.llm_backend import MockLLM
from src.long_term.semantic_memory import SemanticMemory
from src.long_term.episodic_memory import EpisodicMemory
from src.unified.memgpt_agent import MemGPTAgent, CoreMemory
from src.unified.consolidation import MemoryConsolidator


def test_core_memory_mutations():
    core = CoreMemory()
    assert "User" in core.human

    # Append
    assert core.append("human", "Likes Python 3") is True
    assert "Likes Python 3" in core.human

    # Replace
    assert core.replace("human", "User", "Dr. Jane") is True
    assert "Dr. Jane" in core.human
    assert "User" not in core.human

    # Render check
    rendered = core.render()
    assert "Dr. Jane" in rendered
    assert "Likes Python 3" in rendered


def test_memgpt_agent_step_and_tools():
    llm = MockLLM()
    agent = MemGPTAgent(llm=llm, db_path=":memory:")

    reply, operations = agent.step("Hello, my name is Alice and I prefer dark mode.")
    assert len(reply) > 0
    assert len(operations) > 0
    assert operations[0]["function"] == "core_memory_append"
    assert "Alice" in agent.core_memory.human


def test_memory_consolidation_sleep_cycle():
    llm = MockLLM()
    semantic = SemanticMemory(db_path=":memory:")
    episodic = EpisodicMemory(db_path=":memory:", llm=llm)

    agent = MemGPTAgent(llm=llm, db_path=":memory:")
    agent.step("We decided to use SQLite WAL mode for the edge deployment.")
    agent.step("Ensure memory usage stays strictly under 256MB.")

    assert agent.buffer.get_total_tokens() > 0

    consolidator = MemoryConsolidator(semantic_memory=semantic, episodic_memory=episodic, llm=llm)
    report = consolidator.consolidate(buffer=agent.buffer, clear_buffer_after=True)

    # Buffer should be wiped for next session
    assert agent.buffer.get_total_tokens() == 0
    assert report.tokens_freed > 0
    assert len(report.extracted_facts) > 0
    assert len(report.recorded_episodes) > 0

    # Durable facts should be searchable in long-term semantic memory
    results = semantic.search("edge deployment", top_k=2)
    assert len(results) > 0
