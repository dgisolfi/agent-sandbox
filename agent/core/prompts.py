"""Prompts

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import json

TOOL_DOCS = """\
Available tools:
- run_command: args {command: string, cwd?: string, stdin?: string, timeout?: integer, env?: object}
- read_file: args {path: string}
- write_file: args {path: string, content: string, mode?: "overwrite"|"append", create_parents?: boolean}
- list_dir: args {path?: string, recursive?: boolean, max_entries?: integer}
"""


def _json(value) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def planner_prompt(
    task: str, state: dict, last_result: dict, last_eval: dict, stuck_hint: str = None
) -> str:
    stuck_block = f"\n⚠ {stuck_hint}\n" if stuck_hint else ""
    return f"""\
You are a developer-agent working inside a sandbox container.

Your job is to make exactly one concrete next move toward completing the task.
Use a Plan/Execute/Evaluate cycle: choose one tool call that will produce useful feedback.

{TOOL_DOCS}

BEFORE DOING ANYTHING ELSE: If this is your first action or you have not yet listed the
workspace, call `list_dir` on `/workspace` first. Use what you find to decide whether to
start from scratch or continue from where a prior run left off. You may create or update
a plan file (e.g. `/workspace/plan.md`) at any point to track progress across iterations.

Original task:
{task}

Current state:
{_json(state)}

Last tool result:
{_json(last_result or {})}

Last evaluation:
{_json(last_eval or {})}
{stuck_block}
Return only JSON with this exact shape:
{{
  "thought": "Brief reason for the next action.",
  "tool": "run_command",
  "args": {{"command": "pytest -q", "cwd": "/workspace", "timeout": 120}},
  "expected": "Tests pass or expose the next error."
}}
"""


def evaluator_prompt(task: str, state: dict, plan: dict, result: dict) -> str:
    return f"""\
You are evaluating one developer-agent step.

Decide whether the task is complete, should continue, is blocked, or has failed.
Nonzero exit codes, stderr, and exceptions are feedback for the next step unless truly unrecoverable.

IMPORTANT: Only return "complete" when every deliverable listed in the task's SUCCESS
CRITERIA has been produced and verified by actually running it. A single successful script
execution does not make the task complete unless it was the final deliverable and all
prior deliverables are also confirmed. If any required file, result, or document is
missing, return "continue" with a next_goal naming the specific missing item.

Allowed status values: continue, complete, blocked, failed.
Allowed plan update ops: add, complete.

Set "success": true when the tool call produced useful output that advanced the task,
even if more work remains. Set "success": false only when the tool itself failed
(non-zero exit with no useful output, sandbox error, timeout, or unrecoverable exception).
Expected failures during debugging — a failing test, a compile error being investigated —
should be "success": true because they produced actionable information.

Original task:
{task}

Current state before this result:
{_json(state)}

Plan:
{_json(plan)}

Tool result:
{_json(result)}

Return only JSON with this exact shape:
{{
  "status": "continue",
  "success": false,
  "summary": "What happened and why it matters.",
  "next_goal": "The next concrete goal.",
  "plan_updates": [
    {{"op": "add", "item": "Fix the failing test."}}
  ]
}}
"""


def repair_prompt(schema_name: str, invalid_text: str, error: str) -> str:
    return f"""\
Repair this invalid {schema_name} response into valid JSON only.

Error:
{error}

Invalid response:
{invalid_text}
"""
