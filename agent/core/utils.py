"""Utils

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


class ParseError(ValueError):
    pass


def extract_json(text: str) -> dict:
    text = text.strip()
    if not text:
        raise ParseError("Model returned an empty response")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            raise ParseError(f"Could not parse JSON object: {exc}") from exc

    raise ParseError("Could not find a JSON object in the model response")


def require_fields(data: dict, fields: set, label: str) -> None:
    missing = sorted(fields - set(data))
    if missing:
        raise ParseError(f"{label} is missing required fields: {', '.join(missing)}")


###################
# Pydantic States #
###################


@dataclass
class Plan:
    thought: str
    tool: str
    args: dict
    expected: str


@dataclass
class ToolResult:
    ok: bool
    tool: str
    args: dict
    stdout: str = ""
    stderr: str = ""
    exit_code: int = None
    timed_out: bool = False
    error: str = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Evaluation:
    status: str
    success: bool
    summary: str
    next_goal: str = ""
    plan_updates: list = field(default_factory=list)


@dataclass
class RunState:
    task: str
    status: str = "running"
    iteration: int = 0
    max_iterations: int = 100
    error_threshold: int = 5
    stop_on_complete: bool = True
    run_until_limits: bool = False
    plan_items: list = field(default_factory=list)
    completed_items: list = field(default_factory=list)
    current_goal: str = ""
    recent_events: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)
    error_count: int = 0

    def compact(self) -> dict:
        return {
            "status": self.status,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "error_threshold": self.error_threshold,
            "stop_on_complete": self.stop_on_complete,
            "run_until_limits": self.run_until_limits,
            "plan_items": self.plan_items[-20:],
            "completed_items": self.completed_items[-20:],
            "current_goal": self.current_goal,
            "recent_events": self.recent_events[-8:],
            "artifacts": self.artifacts[-40:],
            "error_count": self.error_count,
        }


def to_dict(value) -> dict:
    return asdict(value)


def reduce_state(
    state: RunState, plan: Plan, result: ToolResult, evaluation: Evaluation
) -> RunState:
    state.iteration += 1
    state.current_goal = evaluation.next_goal or state.current_goal
    if not result.ok:
        state.error_count += 1

    if state.error_count >= state.error_threshold:
        state.status = "failed"
    elif evaluation.status in {"blocked", "failed"}:
        state.status = evaluation.status
    elif evaluation.status == "complete" and state.stop_on_complete:
        state.status = "complete"
    else:
        state.status = "running"

    for update in evaluation.plan_updates:
        op = update.get("op")
        item = update.get("item")
        if not item:
            continue
        if op == "add" and item not in state.plan_items:
            state.plan_items.append(item)
        elif op == "complete":
            if item not in state.completed_items:
                state.completed_items.append(item)
            if item in state.plan_items:
                state.plan_items.remove(item)

    if plan.tool in {"write_file", "read_file"}:
        path = plan.args.get("path")
        if isinstance(path, str) and path not in state.artifacts:
            state.artifacts.append(path)

    event = {
        "iteration": state.iteration,
        "tool": plan.tool,
        "cmd": plan.args.get("command") or plan.args.get("path") or "",
        "ok": result.ok,
        "status": evaluation.status,
        "summary": evaluation.summary,
    }
    state.recent_events.append(event)
    state.recent_events = state.recent_events[-20:]
    return state


class RunLogger:
    def __init__(self, runs_dir: str = "/logs/runs", run_id: str = None):
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(runs_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.transcript_path = self.run_dir / "transcript.md"

    def write_event(self, phase: str, iteration: int, payload) -> None:
        data = asdict(payload) if hasattr(payload, "__dataclass_fields__") else payload
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iteration": iteration,
            "phase": phase,
            "payload": data,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._append_transcript(phase, iteration, data)

    def _append_transcript(self, phase: str, iteration: int, data) -> None:
        with self.transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n## Iteration {iteration}: {phase.title()}\n\n")
            handle.write("```json\n")
            handle.write(json.dumps(_compact(data), indent=2, ensure_ascii=False))
            handle.write("\n```\n")


def _compact(value):
    if isinstance(value, dict):
        compacted = {}
        for key, item in value.items():
            if (
                key in {"stdout", "stderr", "content"}
                and isinstance(item, str)
                and len(item) > 4000
            ):
                compacted[key] = item[-4000:]
                compacted[f"{key}_truncated"] = True
            else:
                compacted[key] = _compact(item)
        return compacted
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value
