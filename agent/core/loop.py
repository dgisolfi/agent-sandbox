"""Agent Loop

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import time
from dataclasses import asdict
from datetime import datetime

from core.ollama import OllamaClient
from core.utils import ParseError, extract_json, require_fields
from core.prompts import evaluator_prompt, planner_prompt, repair_prompt
from core.utils import Evaluation, Plan, RunState, ToolResult, reduce_state



class AgentLoop:
    def __init__(self, llm: OllamaClient, tools: ToolRegistry, logger: RunLogger, max_parse_repairs: int = 1):
        self.llm = llm
        self.tools = tools
        self.logger = logger
        self.max_parse_repairs = max_parse_repairs

    def run(self, state: RunState) -> RunState:
        last_result = None
        last_eval = None

        while (
            state.status == "running"
            and state.iteration < state.max_iterations
            and state.error_count < state.error_threshold
        ):
         
            try:
                plan = self._plan(state, last_result, last_eval, stuck_hint=hint)
            except Exception as exc:
                plan = Plan(
                    thought="Planner failed to return a valid action.",
                    tool="planner_error",
                    args={},
                    expected="The planner error is evaluated and corrected on the next iteration.",
                )
                result = ToolResult(
                    ok=False,
                    tool=plan.tool,
                    args=plan.args,
                    error=str(exc),
                    metadata={"exception": type(exc).__name__},
                )
                evaluation = Evaluation(
                    status="continue",
                    success=False,
                    summary=f"Planner error: {exc}",
                    next_goal="Return valid JSON for one available tool call.",
                    plan_updates=[{"op": "add", "item": "Repair invalid planner output."}],
                )
                self.logger.write_event("plan", state.iteration + 1, plan)
                self.logger.write_event("execute", state.iteration + 1, result)
                self.logger.write_event("evaluate", state.iteration + 1, evaluation)

                state = reduce_state(state, plan, result, evaluation)
                last_result = result
                last_eval = evaluation
                continue


            result = self.tools.execute(plan.tool, plan.args)

            try:
                evaluation = self._evaluate(state, plan, result)
            except Exception as exc:
                evaluation = Evaluation(
                    status="continue",
                    success=False,
                    summary=f"Evaluator error: {exc}",
                    next_goal="Evaluate the last tool result with valid JSON.",
                    plan_updates=[{"op": "add", "item": "Repair invalid evaluator output."}],
                )
            self.logger.write_event("evaluate", state.iteration + 1, evaluation)

            state = reduce_state(state, plan, result, evaluation)
            last_result = result
            last_eval = evaluation

        self.logger.write_event("final_state", state.iteration, state.compact())
        return state

    def _plan(self, state, last_result, last_eval, stuck_hint=None) -> Plan:
        prompt = planner_prompt(
            state.task,
            state.compact(),
            _tool_result_for_prompt(last_result),
            asdict(last_eval) if last_eval else None,
            stuck_hint=stuck_hint,
        )
        data = self._json_from_model("planner", prompt, {"thought", "tool", "args", "expected"})
        return Plan(
            thought=str(data["thought"]),
            tool=str(data["tool"]),
            args=data["args"] if isinstance(data["args"], dict) else {},
            expected=str(data["expected"]),
        )

    def _evaluate(self, state, plan, result) -> Evaluation:
        prompt = evaluator_prompt(
            state.task,
            state.compact(),
            asdict(plan),
            _tool_result_for_prompt(result, full=True),
        )
        data = self._json_from_model(
            "evaluator",
            prompt,
            {"status", "success", "summary", "next_goal", "plan_updates"},
        )
        status = str(data["status"])
        if status not in {"continue", "complete", "blocked", "failed"}:
            status = "continue"
        return Evaluation(
            status=status,
            success=bool(data["success"]),
            summary=str(data["summary"]),
            next_goal=str(data.get("next_goal", "")),
            plan_updates=data.get("plan_updates") if isinstance(data.get("plan_updates"), list) else [],
        )
