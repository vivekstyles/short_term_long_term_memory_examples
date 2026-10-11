from src.core.message import Message, Role
from src.core.llm_backend import MockLLM
from src.short_term.buffer_window import ConversationBufferMemory, SlidingWindowMemory
from src.short_term.summary_scratchpad import ConversationSummaryBufferMemory, WorkingMemoryScratchpad


def test_conversation_buffer_memory():
    mem = ConversationBufferMemory()
    assert mem.get_total_tokens() == 0
    assert len(mem.get_messages()) == 0

    mem.add_message(Message(role=Role.USER, content="Hello world"))
    assert len(mem.get_messages()) == 1
    assert mem.get_total_tokens() > 0

    mem.clear()
    assert len(mem.get_messages()) == 0
    assert mem.get_total_tokens() == 0


def test_sliding_window_turn_limit():
    # Keep max 2 non-anchor turns
    mem = SlidingWindowMemory(k_turns=2, preserve_anchors=True)
    mem.add_message(Message(role=Role.SYSTEM, content="System instructions anchor"))
    mem.add_message(Message(role=Role.USER, content="Turn 1"))
    mem.add_message(Message(role=Role.ASSISTANT, content="Turn 2"))
    mem.add_message(Message(role=Role.USER, content="Turn 3"))

    messages = mem.get_messages()
    # Should have 1 system anchor + last 2 turns (Turn 2 and Turn 3)
    assert len(messages) == 3
    assert messages[0].role == Role.SYSTEM
    assert messages[1].content == "Turn 2"
    assert messages[2].content == "Turn 3"
    assert len(mem.get_evicted_messages()) == 1
    assert mem.get_evicted_messages()[0].content == "Turn 1"


def test_sliding_window_token_budget_preserves_anchor():
    # Tight token budget
    mem = SlidingWindowMemory(max_tokens=30, preserve_anchors=True)
    mem.add_message(Message(role=Role.SYSTEM, content="System instructions"))
    mem.add_message(Message(role=Role.USER, content="A very long message that takes quite a few tokens to represent"))
    mem.add_message(Message(role=Role.ASSISTANT, content="Another message"))

    messages = mem.get_messages()
    # System message must never be evicted
    assert any(m.role == Role.SYSTEM for m in messages)
    assert mem.get_total_tokens() <= 30 or len(messages) == 1


def test_conversation_summary_buffer():
    llm = MockLLM()
    mem = ConversationSummaryBufferMemory(
        llm=llm,
        max_tokens=25,
        keep_recent_k=2,
        system_prompt="Base System Prompt",
    )

    mem.add_message(Message(role=Role.USER, content="Message 1 with extensive background context"))
    mem.add_message(Message(role=Role.ASSISTANT, content="Response 1 discussing architectural considerations"))
    mem.add_message(Message(role=Role.USER, content="Message 2 continuing the discussion on database indexing"))
    mem.add_message(Message(role=Role.ASSISTANT, content="Response 2 with recommendations"))

    # Summarization should have triggered
    assert mem.summarization_events >= 1
    assert mem.summary != ""
    output_messages = mem.get_messages()
    assert output_messages[0].role == Role.SYSTEM
    assert "[COMPACT SUMMARY OF PREVIOUS CONVERSATION TURNS]" in output_messages[0].content
    # Recent turns must be preserved verbatim
    assert output_messages[-1].content == "Response 2 with recommendations"


def test_working_memory_scratchpad():
    pad = WorkingMemoryScratchpad()
    pad.set_goal("Test Goal")
    t1 = pad.add_subtask("Step 1")
    t2 = pad.add_subtask("Step 2")

    pad.update_subtask_status(t1, "completed")
    pad.store_variable("temp_dir", "/tmp/agent")
    pad.add_note("Note 1")

    rendered = pad.render_context()
    assert "Test Goal" in rendered
    assert "[OK] (COMPLETED) #1: Step 1" in rendered
    assert "[ ] (PENDING) #2: Step 2" in rendered
    assert "temp_dir: /tmp/agent" in rendered
    assert "Note 1" in rendered
