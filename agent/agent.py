"""LangChain-backed agent runner."""

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler

from agent.ollama import build_model
from agent.tools.sandbox import build_tools

logger = logging.getLogger(__name__)


class _StepLogger(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        n = len(messages[0]) if messages else 0
        logger.info("LLM thinking | context messages: %d", n)

    def on_tool_start(self, serialized, input_str, **kwargs):
        logger.info("Tool: %s | %s", serialized.get("name", "?"), str(input_str)[:120])

    def on_tool_end(self, output, **kwargs):
        logger.info("Tool done | %.120s", str(output))

    def on_agent_finish(self, finish, **kwargs):
        logger.info("Agent finished | %.120s", str(finish.return_values))


SYSTEM_PROMPT = """\
You are a developer-agent working inside a sandbox container.

Use the available tools to complete the user's task. Before making assumptions,
inspect /workspace with list_dir. Work in small concrete steps. For multi-step
tasks, maintain /workspace/plan.md so progress survives long runs.

Use run_command to verify changes. Treat failures as feedback, inspect relevant
files, and change something before repeating the same failed command. Only finish
when every deliverable is complete and verified.
"""


@dataclass
class State:
    task: str
    status: str = "running"
    iteration: int = 0
    max_iterations: int = 100
    error_threshold: int = 5
    stop_on_complete: bool = True
    run_until_limits: bool = False
    current_goal: str = ""
    error_count: int = 0
    final_output: str = ""
    artifacts: list[str] = field(default_factory=list)

    def compact(self) -> dict:
        return {
            "status": self.status,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "error_threshold": self.error_threshold,
            "stop_on_complete": self.stop_on_complete,
            "run_until_limits": self.run_until_limits,
            "current_goal": self.current_goal,
            "error_count": self.error_count,
            "final_output": self.final_output,
            "artifacts": self.artifacts[-40:],
        }


class Agent:
    def __init__(
        self,
        sandbox,
        logger,
        model_name: str,
        base_url: str,
        model_adapter: str = "wrapped",
    ) -> None:
        self.sandbox = sandbox
        self.logger = logger
        self.model = build_model(model_name, base_url, mode=model_adapter)
        self.tools = build_tools(sandbox, logger=logger)
        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
        )

    def run(
        self,
        task: str,
        max_iterations: int,
        error_threshold: int = 5,
        stop_on_complete: bool = True,
        run_until_limits: bool = False,
    ) -> State:
        state = State(
            task=task,
            max_iterations=max_iterations,
            error_threshold=error_threshold,
            stop_on_complete=stop_on_complete,
            run_until_limits=run_until_limits,
            current_goal="Complete the task using sandbox tools and verify the result.",
        )
        logger.info("Experiment starting | task: %.120s", task)
        try:
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": task}]},
                config={
                    "recursion_limit": max_iterations * 2 + 5,
                    "callbacks": [_StepLogger()],
                },
            )
            state.final_output = self._final_text(result)
            state.iteration = self._count_tool_steps(result)

            _ERROR_SIGNALS = ("model requested an unknown or malformed tool call",)
            if any(sig in state.final_output.lower() for sig in _ERROR_SIGNALS):
                state.status = "failed"
                state.error_count += 1
                logger.error(
                    "Run ended with model error after %d iterations", state.iteration
                )
            else:
                state.status = "complete"
                logger.info(
                    "Run complete | status=%s iterations=%d",
                    state.status,
                    state.iteration,
                )

            self.logger.write_event("final_state", state.iteration, state.compact())
            return state
        except Exception as exc:
            state.status = "failed"
            state.error_count = 1
            state.final_output = str(exc)
            self.logger.write_event(
                "final_state",
                state.iteration,
                {**state.compact(), "error": str(exc), "exception": type(exc).__name__},
            )
            return state

    def _count_tool_steps(self, result: dict[str, Any]) -> int:
        messages = result.get("messages", []) if isinstance(result, dict) else []
        return sum(1 for message in messages if getattr(message, "type", "") == "tool")

    def _final_text(self, result: dict[str, Any]) -> str:
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if not messages:
            return ""
        content = getattr(messages[-1], "content", "")
        if isinstance(content, str):
            return content
        return str(content)
