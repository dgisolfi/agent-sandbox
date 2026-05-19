"""Tool Registry

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

from agent.core.utils import ToolResult
from agent.tools.sandbox import SandboxClient


class ToolValidationError(ValueError):
    pass


class ToolRegistry:
    def __init__(self, sandbox: SandboxClient):
        self.sandbox = sandbox
        self._tools = {
            "run_command": self.sandbox.run_command,
            "read_file": self.sandbox.read_file,
            "write_file": self.sandbox.write_file,
            "list_dir": self.sandbox.list_dir,
        }

    @property
    def names(self) -> set:
        return set(self._tools)

    def execute(self, name: str, args: dict) -> ToolResult:
        try:
            self.validate(name, args)
            raw = self._tools[name](**args)
            return self._result_from_raw(name, args, raw)
        except Exception as exc:
            return ToolResult(
                ok=False,
                tool=name,
                args=args,
                error=str(exc),
                metadata={"exception": type(exc).__name__},
            )

    def validate(self, name: str, args: dict) -> None:
        if name not in self._tools:
            raise ToolValidationError(f"Unknown tool: {name}")
        if not isinstance(args, dict):
            raise ToolValidationError("Tool args must be an object")
        if name == "run_command" and not isinstance(args.get("command"), str):
            raise ToolValidationError("run_command requires string arg: command")
        if name in {"read_file", "write_file"} and not isinstance(
            args.get("path"), str
        ):
            raise ToolValidationError(f"{name} requires string arg: path")
        if name == "write_file" and not isinstance(args.get("content"), str):
            raise ToolValidationError("write_file requires string arg: content")

    def _result_from_raw(self, name: str, args: dict, raw: dict) -> ToolResult:
        ok = bool(raw.get("ok"))
        if name == "run_command":
            exit_code = raw.get("exit_code")
            ok = ok and exit_code == 0 and not raw.get("timed_out", False)
            return ToolResult(
                ok=ok,
                tool=name,
                args=args,
                stdout=raw.get("stdout", ""),
                stderr=raw.get("stderr", ""),
                exit_code=exit_code,
                timed_out=bool(raw.get("timed_out", False)),
                error=raw.get("error"),
                metadata={
                    k: v
                    for k, v in raw.items()
                    if k not in {"stdout", "stderr", "ok", "error"}
                },
            )
        return ToolResult(
            ok=ok, tool=name, args=args, error=raw.get("error"), metadata=raw
        )
