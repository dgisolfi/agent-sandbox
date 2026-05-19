"""Utils

Parsing utilities and run logger. Stateless helpers with no dependency
on the MDP types or environment.

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import json
import re
from dataclasses import asdict
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
