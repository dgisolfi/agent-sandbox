"""Agent Loop

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import time
from dataclasses import asdict
from datetime import datetime

from agent.core.ollama import OllamaClient
from agent.core.prompts import evaluator_prompt, planner_prompt, repair_prompt
from agent.core.utils import (
    Evaluation,
    ParseError,
    Plan,
    RunLogger,
    RunState,
    ToolResult,
    extract_json,
    reduce_state,
    require_fields,
)
from agent.tools.registry import ToolRegistry

_STUCK_THRESHOLD = 3
_RETRY_WINDOW = 10


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _elapsed(since: float) -> str:
    return f"{time.monotonic() - since:.1f}s"


def _extract_source_path(command: str) -> str:
    for token in command.split():
        if token.startswith("/workspace/") and token.endswith(".py"):
            return token
    return "/workspace/plan.md"


def _enforce_change_before_retry(state: RunState, plan: Plan) -> Plan | None:
    if plan.tool != "run_command":
        return None
    cmd = plan.args.get("command", "")
    if not cmd:
        return None

    events = state.recent_events[-_RETRY_WINDOW:]
    last_failure_idx = None
    for i in range(len(events) - 1, -1, -1):
        e = events[i]
        if e.get("tool") == "run_command" and e.get("cmd") == cmd and not e.get("ok"):
            last_failure_idx = i
            break

    if last_failure_idx is None:
        return None

    for e in events[last_failure_idx + 1 :]:
        if e.get("tool") == "write_file" and e.get("ok"):
            return None

    path = _extract_source_path(cmd)
    return Plan(
        thought=f"Blocked repeat of failed command. Forcing read of {path} before retry.",
        tool="read_file",
        args={"path": path},
        expected="File content to inform the next fix attempt.",
    )


def _stuck_hint(state: RunState) -> str:
    events = state.recent_events[-_STUCK_THRESHOLD:]
    if len(events) < _STUCK_THRESHOLD:
        return None
    tools = [e.get("tool") for e in events]
    oks = [e.get("ok") for e in events]
    if len(set(tools)) == 1 and all(ok is False for ok in oks):
        return (
            f"You have called '{tools[0]}' {_STUCK_THRESHOLD} times in a row and it has failed "
            f"every time. Do NOT repeat this call. Instead, read the file that is causing the "
            f"error and write a fix before running again."
        )
    return None


class AgentLoop:
    def __init__(
        self,
        llm: OllamaClient,
        tools: ToolRegistry,
        logger: RunLogger,
        max_parse_repairs: int = 1,
    ):
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
            print(
                f"[{_ts()}] [Agent] Iteration {state.iteration + 1}/{state.max_iterations}"
            )

            hint = _stuck_hint(state)
            if hint:
                print(f"[{_ts()}] [Agent] Stuck detected — injecting recovery hint")

            t = time.monotonic()
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
                    plan_updates=[
                        {"op": "add", "item": "Repair invalid planner output."}
                    ],
                )
                self.logger.write_event("plan", state.iteration + 1, plan)
                self.logger.write_event("execute", state.iteration + 1, result)
                self.logger.write_event("evaluate", state.iteration + 1, evaluation)
                print(f"[{_ts()}] [Plan] planner_error ({_elapsed(t)}): {exc}")
                state = reduce_state(state, plan, result, evaluation)
                last_result = result
                last_eval = evaluation
                continue

            override = _enforce_change_before_retry(state, plan)
            if override is not None:
                print(
                    f"[{_ts()}] [Agent] Repeat-failure blocked — forcing read: {override.args['path']}"
                )
                plan = override

            self.logger.write_event("plan", state.iteration + 1, plan)
            print(
                f"[{_ts()}] [Plan] {plan.tool} ({_elapsed(t)}): {self._preview_args(plan.args)}"
            )

            t = time.monotonic()
            result = self.tools.execute(plan.tool, plan.args)
            self.logger.write_event("execute", state.iteration + 1, result)
            print(
                f"[{_ts()}] [Execute] ok={result.ok} exit={result.exit_code} timed_out={result.timed_out} ({_elapsed(t)})"
            )

            t = time.monotonic()
            try:
                evaluation = self._evaluate(state, plan, result)
            except Exception as exc:
                evaluation = Evaluation(
                    status="continue",
                    success=False,
                    summary=f"Evaluator error: {exc}",
                    next_goal="Evaluate the last tool result with valid JSON.",
                    plan_updates=[
                        {"op": "add", "item": "Repair invalid evaluator output."}
                    ],
                )
            self.logger.write_event("evaluate", state.iteration + 1, evaluation)
            print(
                f"[{_ts()}] [Evaluate] {evaluation.status} ({_elapsed(t)}): {evaluation.summary}"
            )

            state = reduce_state(state, plan, result, evaluation)
            last_result = result
            last_eval = evaluation

        if state.status == "running" and state.error_count >= state.error_threshold:
            state.status = "failed"
            state.recent_events.append(
                {"summary": "Error threshold reached", "status": "failed"}
            )
        elif state.status == "running" and state.iteration >= state.max_iterations:
            state.status = "complete" if state.run_until_limits else "failed"
            state.recent_events.append(
                {"summary": "Maximum iterations reached", "status": state.status}
            )

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
        data = self._json_from_model(
            "planner", prompt, {"thought", "tool", "args", "expected"}
        )
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
            plan_updates=(
                data.get("plan_updates")
                if isinstance(data.get("plan_updates"), list)
                else []
            ),
        )

    def _json_from_model(self, label: str, prompt: str, required: set) -> dict:
        raw = self.llm.generate(prompt)
        for attempt in range(self.max_parse_repairs + 1):
            try:
                data = extract_json(raw)
                require_fields(data, required, label)
                return data
            except ParseError as exc:
                if attempt >= self.max_parse_repairs:
                    raise
                raw = self.llm.generate(repair_prompt(label, raw, str(exc)))
        raise ParseError(f"Could not parse {label} response")

    @staticmethod
    def _preview_args(args: dict) -> str:
        if "command" in args:
            return str(args["command"])[:160]
        if "path" in args:
            return str(args["path"])
        return str(args)[:160]


def _tool_result_for_prompt(result, full=False):
    if result is None:
        return None
    data = asdict(result)
    if not full:
        for key in ("stdout", "stderr"):
            value = data.get(key)
            if isinstance(value, str) and len(value) > 3000:
                data[key] = value[-3000:]
                data[f"{key}_truncated"] = True
    return data
