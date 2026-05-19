"""Sandbox Tools

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import json
from typing import Any

import requests
from langchain_core.tools import tool


class SandboxClient:
    def __init__(self, base_url: str = "http://sandbox:8000", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def run_command(self, **kwargs) -> dict:
        return self._post("/v1/command", kwargs)

    def read_file(self, **kwargs) -> dict:
        return self._post("/v1/files/read", kwargs)

    def write_file(self, **kwargs) -> dict:
        return self._post("/v1/files/write", kwargs)

    def list_dir(self, **kwargs) -> dict:
        return self._post("/v1/files/list", kwargs)

    def _post(self, path: str, payload: dict) -> dict:
        try:
            response = requests.post(
                f"{self.base_url}{path}", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            envelope = response.json()
            if envelope.get("ok"):
                return {"ok": True, **(envelope.get("data") or {})}
            error = envelope.get("error") or {}
            return {
                "ok": False,
                "error": error.get("message", "Sandbox request failed"),
                "error_type": error.get("type", "sandbox_error"),
                "details": error.get("details", {}),
            }
        except requests.RequestException as exc:
            return {"ok": False, "error": str(exc), "error_type": "connection_error"}


def _compact(value: Any, TEXT_LIMIT=6000) -> Any:
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact(item) for item in value]
    if isinstance(value, str) and len(value) > TEXT_LIMIT:
        return value[-TEXT_LIMIT:] + "\n[truncated]"
    return value


def _as_json(raw: dict) -> str:
    return json.dumps(_compact(raw), ensure_ascii=False)


def build_tools(sandbox, logger=None):
    """Build LangChain tools backed by the sandbox service."""
    step = 0

    def _log(phase: str, iteration: int, name: str, payload: dict) -> None:
        if logger is not None:
            logger.write_event(phase, iteration, {"tool": name, **payload})

    @tool
    def run_command(
        command: str,
        cwd: str = "/workspace",
        stdin: str = "",
        timeout: int = 120,
        env: dict | None = None,
    ) -> str:
        """Run a shell command in the sandbox and return stdout, stderr, and exit metadata."""
        args = {
            "command": command,
            "cwd": cwd,
            "stdin": stdin,
            "timeout": timeout,
            "env": env or {},
        }
        nonlocal step
        step += 1
        iteration = step
        _log("tool_call", iteration, "run_command", {"args": args})
        result = sandbox.run_command(**args)
        _log("tool_result", iteration, "run_command", {"args": args, "result": result})
        return _as_json(result)

    @tool
    def read_file(path: str) -> str:
        """Read a file from the sandbox."""
        args = {"path": path}
        nonlocal step
        step += 1
        iteration = step
        _log("tool_call", iteration, "read_file", {"args": args})
        result = sandbox.read_file(**args)
        _log("tool_result", iteration, "read_file", {"args": args, "result": result})
        return _as_json(result)

    @tool
    def write_file(
        path: str,
        content: str,
        mode: str = "overwrite",
        create_parents: bool = True,
    ) -> str:
        """Write a file in the sandbox."""
        args = {
            "path": path,
            "content": content,
            "mode": mode,
            "create_parents": create_parents,
        }
        nonlocal step
        step += 1
        iteration = step
        _log("tool_call", iteration, "write_file", {"args": args})
        result = sandbox.write_file(**args)
        _log("tool_result", iteration, "write_file", {"args": args, "result": result})
        return _as_json(result)

    @tool
    def list_dir(
        path: str = "/workspace",
        recursive: bool = False,
        max_entries: int = 200,
    ) -> str:
        """List files and directories in the sandbox."""
        args = {
            "path": path,
            "recursive": recursive,
            "max_entries": max_entries,
        }
        nonlocal step
        step += 1
        iteration = step
        _log("tool_call", iteration, "list_dir", {"args": args})
        result = sandbox.list_dir(**args)
        _log("tool_result", iteration, "list_dir", {"args": args, "result": result})
        return _as_json(result)

    return [run_command, read_file, write_file, list_dir]
