"""
tasks/language/agent.py
------------------------
Lightweight agent orchestration using LangChain tool-calling agents.

Sits on top of the existing factory.py LLM abstraction — no new dependencies.
All three LLM providers (Azure, OpenAI-compatible, Ollama) work out of the box.

Design notes:
- Single-agent by default. Multi-agent patterns can be composed by passing an
  AgentRunner's `run` method as a tool to another AgentRunner.
- Stateless per call. Pass `chat_history` for multi-turn conversations.
- Tools are plain @tool-decorated functions or LangChain BaseTool subclasses.

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
    A thin wrapper around LangChain's tool-calling agent pattern.

    Args:
        tools:          List of @tool-decorated callables or BaseTool instances.
                        Pass an empty list for a plain conversational agent.
        system_prompt:  System-level instruction injected at the top of every
                        conversation.  Defaults to a generic helpful assistant.
        max_iterations: Safety limit on agent reasoning loops (default 10).
        verbose:        Forward LangChain agent verbose output to stdout.
    """

    DEFAULT_SYSTEM = (
        "You are a helpful, concise assistant. "
        "Use the available tools whenever they are relevant to answer the query."
    )

    def __init__(
        self,
        tools: list[BaseTool | Callable] | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 10,
        verbose: bool = False,
    ) -> None:
        self.tools = tools or []
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM
        self.max_iterations = max_iterations
        self.verbose = verbose

        self.llm = get_llm()
        self._executor = self._build_executor()

        tool_names = [t.name if hasattr(t, "name") else getattr(t, "__name__", str(t)) for t in self.tools]
        print(
            f"🤖 AgentRunner ready. LLM: {self.llm.__class__.__name__} | "
            f"Tools: {tool_names or ['none']} | MaxIter: {self.max_iterations}"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_executor(self):
        """Build a LangChain AgentExecutor with tool-calling support."""
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        llm_with_tools = self.llm.bind_tools(self.tools) if self.tools else self.llm
        agent = create_tool_calling_agent(llm_with_tools, self.tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            handle_parsing_errors=True,
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

        Args:
            query:        The user's input string.
            chat_history: Optional list of prior BaseMessage objects for
                          multi-turn conversations.  Returned updated in the
                          response dict so callers can thread it back in.

        Returns:
            {
                "output":       str,              # final answer
                "chat_history": list[BaseMessage] # updated history
            }
        """
        history: list[BaseMessage] = list(chat_history or [])

        result = self._executor.invoke(
            {"input": query, "chat_history": history}
        )

        output: str = result.get("output", "")

        # Append this turn to the history
        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=output))

        return {"output": output, "chat_history": history}

    def run_batch(
        self,
        queries: list[str],
    ) -> list[str]:
        """
        Run the agent independently on a list of queries (stateless per item).

        Useful for scoring a batch of competition rows that each need tool calls.

        Args:
            queries: List of query strings.

        Returns:
            List of output strings in the same order.
        """
        return [self.run(q)["output"] for q in queries]

    def add_tool(self, tool: BaseTool | Callable) -> None:
        """
        Register an additional tool and rebuild the executor.

        Useful on competition day when a new tool (e.g. a database lookup)
        becomes available once the task spec is revealed.
        """
        self.tools.append(tool)
        self._executor = self._build_executor()
        print(f"🔧 Tool added. Active tools: {[t.name for t in self.tools]}")
