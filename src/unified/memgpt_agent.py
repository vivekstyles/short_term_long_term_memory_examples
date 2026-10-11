from __future__ import annotations
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from src.core.message import Message, Role, count_tokens
from src.core.llm_backend import LLMBackend, get_llm
from src.short_term.buffer_window import SlidingWindowMemory
from src.long_term.semantic_memory import SemanticMemory


class CoreMemory(BaseModel):
    """
    In-Context Core Memory:
    Directly visible in the system prompt. Contains high-priority, persistent persona
    and human profile sections that can be autonomously edited by the agent.
    """
    human: str = "Name: User\nPreferences: Likes clear, modular Python code."
    persona: str = (
        "I am an autonomous, self-reflecting cognitive agent. "
        "I maintain and organize my own memory across sessions."
    )

    def render(self) -> str:
        return (
            "[CORE MEMORY - IN CONTEXT]\n"
            f"=== Human Profile ===\n{self.human}\n\n"
            f"=== Agent Persona ===\n{self.persona}\n"
            "[END CORE MEMORY]"
        )

    def append(self, section: str, content: str) -> bool:
        sec = section.lower().strip()
        if sec == "human":
            self.human += f"\n{content.strip()}"
            return True
        elif sec == "persona":
            self.persona += f"\n{content.strip()}"
            return True
        return False

    def replace(self, section: str, old_str: str, new_str: str) -> bool:
        sec = section.lower().strip()
        if sec == "human":
            if old_str in self.human:
                self.human = self.human.replace(old_str, new_str)
                return True
        elif sec == "persona":
            if old_str in self.persona:
                self.persona = self.persona.replace(old_str, new_str)
                return True
        return False


class MemGPTAgent:
    """
    Level 6: Advanced Unified Cognitive Architecture (MemGPT Pattern).
    Features a multi-tier memory system:
    1. Core Memory: In-context, editable via agent tool calls.
    2. Archival Memory: Out-of-context semantic database, queried via search tools.
    3. Working FIFO Window: Bounded short-term message buffer.
    """

    def __init__(
        self,
        llm: Optional[LLMBackend] = None,
        db_path: str = ":memory:",
        max_buffer_tokens: int = 800,
    ) -> None:
        self.llm = llm or get_llm()
        self.core_memory = CoreMemory()
        self.archival_memory = SemanticMemory(db_path=db_path)
        self.buffer = SlidingWindowMemory(max_tokens=max_buffer_tokens, preserve_anchors=True)
        self.memory_operations_log: List[Dict[str, Any]] = []

    def get_system_prompt(self) -> str:
        tools_guide = """
You have access to self-editing memory tools. To use a tool, output a line in this exact format:
FUNCTION_CALL: core_memory_append(section="human", content="...")
FUNCTION_CALL: core_memory_replace(section="human", old_str="...", new_str="...")
FUNCTION_CALL: archival_memory_insert(content="...", category="...")
FUNCTION_CALL: archival_memory_search(query="...")

Always update Core Memory whenever the user reveals new facts about themselves, preferences, or project rules.
"""
        return f"{self.core_memory.render()}\n\n{tools_guide}"

    def step(self, user_input: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Execute one conversational agent step:
        1. Adds user message to short-term buffer.
        2. Prepares active context (Core Memory + sliding window).
        3. Calls LLM.
        4. Detects and executes memory function calls.
        5. Returns final assistant reply and executed operations log.
        """
        user_msg = Message(role=Role.USER, content=user_input)
        self.buffer.add_message(user_msg)

        # Build full message payload
        messages = [
            Message(role=Role.SYSTEM, content=self.get_system_prompt()),
            *self.buffer.get_messages(),
        ]

        llm_response = self.llm.chat(messages)
        operations = self._parse_and_execute_tools(llm_response)

        # Remove function call syntax from user-facing response
        clean_response = re.sub(r"FUNCTION_CALL:[^\n]+", "", llm_response).strip()
        if not clean_response:
            clean_response = "I have updated my memory with that information."

        # Add assistant response to buffer
        asst_msg = Message(role=Role.ASSISTANT, content=clean_response)
        self.buffer.add_message(asst_msg)

        return clean_response, operations

    def _parse_and_execute_tools(self, response_text: str) -> List[Dict[str, Any]]:
        operations = []
        pattern = r"FUNCTION_CALL:\s*([a-zA-Z_]+)\((.*?)\)"
        matches = re.findall(pattern, response_text)

        for func_name, arg_str in matches:
            op_record: Dict[str, Any] = {"function": func_name, "args": arg_str, "status": "unknown"}
            try:
                # Parse key=value or JSON-like args
                if func_name == "core_memory_append":
                    sec_match = re.search(r'section=["\'](.*?)["\']', arg_str)
                    content_match = re.search(r'content=["\'](.*?)["\']', arg_str)
                    if sec_match and content_match:
                        sec = sec_match.group(1)
                        content = content_match.group(1)
                        success = self.core_memory.append(sec, content)
                        op_record["status"] = "success" if success else "failed_section"
                        op_record["details"] = f"Appended to {sec}: '{content}'"

                elif func_name == "core_memory_replace":
                    sec_match = re.search(r'section=["\'](.*?)["\']', arg_str)
                    old_match = re.search(r'old_str=["\'](.*?)["\']', arg_str)
                    new_match = re.search(r'new_str=["\'](.*?)["\']', arg_str)
                    if sec_match and old_match and new_match:
                        sec = sec_match.group(1)
                        old_s = old_match.group(1)
                        new_s = new_match.group(1)
                        success = self.core_memory.replace(sec, old_s, new_s)
                        op_record["status"] = "success" if success else "not_found"
                        op_record["details"] = f"Replaced in {sec}: '{old_s}' -> '{new_s}'"

                elif func_name == "archival_memory_insert":
                    content_match = re.search(r'content=["\'](.*?)["\']', arg_str)
                    cat_match = re.search(r'category=["\'](.*?)["\']', arg_str)
                    cat = cat_match.group(1) if cat_match else "general"
                    if content_match:
                        content = content_match.group(1)
                        fact_id = self.archival_memory.add_fact(fact=content, category=cat)
                        op_record["status"] = "success"
                        op_record["details"] = f"Archived fact #{fact_id} ({cat}): '{content}'"

                elif func_name == "archival_memory_search":
                    query_match = re.search(r'query=["\'](.*?)["\']', arg_str)
                    if query_match:
                        query = query_match.group(1)
                        results = self.archival_memory.search(query=query, top_k=2)
                        op_record["status"] = "success"
                        op_record["details"] = f"Found {len(results)} matches for '{query}'"
                        op_record["results"] = [r.fact for r in results]

                operations.append(op_record)
                self.memory_operations_log.append(op_record)

            except Exception as e:
                op_record["status"] = f"error: {e}"
                operations.append(op_record)

        return operations
