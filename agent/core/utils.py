"""Utils

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import json
import re

from dataclasses import asdict, dataclass, field

class ParseError(ValueError):
    pass

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