"""
tasks/language/agent.py
------------------------
Lightweight agent orchestration using LangChain 1.x tool-calling agents.

Uses the new create_agent API (LangChain >= 1.0) which replaces the
deprecated AgentExecutor + create_tool_calling_agent pattern.

Usage:
    from langchain_core.tools import tool
    from tasks.language.agent import AgentRunner

    @tool
    def lookup_score(patient_id: str) -> str:
        \"\"\"Look up the risk score for a patient by their ID.\"\"\"
        return f"Score for {patient_id}: 0.87"

    agent = AgentRunner(
        tools=[lookup_score],
        system_prompt="You are a medical risk assessment assistant.",
    )
    result = agent.run("What is the risk score for patient P-001?")
    print(result["output"])

    # Multi-turn
    history = []
    r1 = agent.run("What is the score for P-001?", chat_history=history)
    history = r1["chat_history"]
    r2 = agent.run("Is that score considered high?", chat_history=history)
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from tasks.language.factory import get_llm

# ---------------------------------------------------------------------------
# Convenience re-export so callers only need to import from this module
# ---------------------------------------------------------------------------
from langchain_core.tools import tool as make_tool  # noqa: F401  (re-exported)


class AgentRunner:
    """
    Thin wrapper around LangChain 1.x create_agent (tool-calling loop).

    Args:
        tools:          List of @tool-decorated callables or BaseTool instances.
        system_prompt:  System instruction prepended to every conversation.
        max_iterations: Max tool-call iterations before stopping (default 20).
        verbose:        Print agent steps to stdout.
    """

    DEFAULT_SYSTEM = (
        "You are a helpful, concise assistant. "
        "Use the available tools whenever they are relevant to answer the query."
    )

    def __init__(
        self,
        tools: list[BaseTool | Callable] | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 20,
        verbose: bool = False,
    ) -> None:
        self.tools = tools or []
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM
        self.max_iterations = max_iterations
        self.verbose = verbose

        self.llm = get_llm()
        self._agent = self._build_agent()

        tool_names = [t.name if hasattr(t, "name") else getattr(t, "__name__", str(t)) for t in self.tools]
        print(
            f"AgentRunner ready. LLM: {self.llm.__class__.__name__} | "
            f"Tools: {tool_names or ['none']} | MaxIter: {self.max_iterations}"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_agent(self):
        from langchain.agents import create_agent
        return create_agent(
            self.llm,
            self.tools or None,
            system_prompt=SystemMessage(content=self.system_prompt),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        chat_history: list[BaseMessage] | None = None,
    ) -> dict[str, Any]:
        """
        Run the agent on a single query.

        Returns:
            {
                "output":       str,               # final answer
                "chat_history": list[BaseMessage]  # updated history
            }
        """
        history: list[BaseMessage] = list(chat_history or [])

        messages = history + [HumanMessage(content=query)]
        result = self._agent.invoke(
            {"messages": messages},
            config={"recursion_limit": self.max_iterations * 2},
        )

        # LangChain 1.x returns {"messages": [...]} — last message is the answer
        output_msg = result["messages"][-1]
        raw_content = output_msg.content if hasattr(output_msg, "content") else str(output_msg)

        # Handle Gemini thinking models — content may be a list of blocks
        # (e.g. [{"type": "thinking", ...}, {"type": "text", "text": "..."}])
        if isinstance(raw_content, list):
            text_parts = []
            for block in raw_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            output = "\n".join(text_parts) if text_parts else str(raw_content)
        else:
            output = str(raw_content)

        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=output))

        return {"output": output, "chat_history": history}

    def run_batch(self, queries: list[str]) -> list[str]:
        """Run the agent independently on a list of queries (stateless per item)."""
        return [self.run(q)["output"] for q in queries]

    def add_tool(self, tool: BaseTool | Callable) -> None:
        """Register an additional tool and rebuild the agent."""
        self.tools.append(tool)
        self._agent = self._build_agent()
        print(f"Tool added. Active tools: {[t.name for t in self.tools]}")
