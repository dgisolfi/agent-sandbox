"""Tools

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import asyncio
import os
import time
from pathlib import Path

from pydantic import BaseModel, Field


def _configured_roots() -> list:
    raw = os.environ.get("SANDBOX_ALLOWED_ROOTS", "/workspace")
    roots = []
    for item in raw.split(":"):
        item = item.strip()
        if item:
            roots.append(Path(item).resolve())
    return roots or [Path("/workspace").resolve()]


ALLOWED_ROOTS = _configured_roots()
DENIED_ROOTS = tuple(Path(p) for p in ("/proc", "/sys", "/dev"))


class SandboxFsError(Exception):
    def __init__(self, error_type: str, message: str, details: dict = None):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.details = details or {}

###################
# Pydantic models #
###################

class ErrorBody(BaseModel):
    type: str
    message: str
    details: dict = Field(default_factory=dict)


class Response(BaseModel):
    ok: bool
    data: dict | None = None
    error: ErrorBody | None = None


class CommandRequest(BaseModel):
    command: str
    cwd: str = "/workspace"
    stdin: str | None = None
    timeout: int = Field(default=30, ge=1, le=3600)
    env: dict = Field(default_factory=dict)


class FileReadRequest(BaseModel):
    path: str


class FileWriteRequest(BaseModel):
    path: str
    content: str
    mode: str = Field(default="overwrite", pattern="^(overwrite|append)$")
    create_parents: bool = True


class FileListRequest(BaseModel):
    path: str = "/workspace"
    recursive: bool = False
    max_entries: int = Field(default=200, ge=1, le=5000)


def ensure_allowed(raw_path: str, must_exist: bool = False) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path("/workspace") / path
    resolved = path.resolve(strict=False)

    for denied in DENIED_ROOTS:
        try:
            resolved.relative_to(denied)
            raise SandboxFsError("permission_denied", f"Path is denied: {resolved}")
        except ValueError:
            # Return explicit error maybve?
            pass

    if not any(_is_relative_to(resolved, root) for root in ALLOWED_ROOTS):
        raise SandboxFsError(
            "permission_denied",
            "Path is outside allowed roots.",
            {
                "path": str(resolved),
                "allowed_roots": [str(root) for root in ALLOWED_ROOTS],
            },
        )

    if must_exist and not resolved.exists():
        raise SandboxFsError(
            "not_found", f"Path does not exist: {resolved}", {"path": str(resolved)}
        )
    return resolved


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

#################################
# Tools for sandbox interaction #
#################################

def read_file(path: str) -> dict:
    resolved = ensure_allowed(path, must_exist=True)
    if not resolved.is_file():
        raise SandboxFsError(
            "invalid_path", "Path is not a file.", {"path": str(resolved)}
        )
    content = resolved.read_text(encoding="utf-8", errors="replace")
    stat = resolved.stat()
    return {"path": str(resolved), "content": content, "size_bytes": stat.st_size}


def write_file(
    path: str, content: str, mode: str = "overwrite", create_parents: bool = True
) -> dict:
    resolved = ensure_allowed(path)
    if create_parents:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    if mode == "append":
        with resolved.open("a", encoding="utf-8") as handle:
            handle.write(content)
    else:
        resolved.write_text(content, encoding="utf-8")
    stat = resolved.stat()
    return {"path": str(resolved), "size_bytes": stat.st_size, "mode": mode}


def list_dir(path: str, recursive: bool = False, max_entries: int = 200) -> dict:
    resolved = ensure_allowed(path, must_exist=True)
    if not resolved.is_dir():
        raise SandboxFsError(
            "invalid_path", "Path is not a directory.", {"path": str(resolved)}
        )

    iterator = resolved.rglob("*") if recursive else resolved.iterdir()
    entries = []
    truncated = False
    for child in sorted(iterator):
        if len(entries) >= max_entries:
            truncated = True
            break
        stat = child.stat()
        entries.append(
            {
                "path": str(child),
                "name": child.name,
                "type": "dir" if child.is_dir() else "file",
                "size_bytes": stat.st_size if child.is_file() else None,
            }
        )
    return {"path": str(resolved), "entries": entries, "truncated": truncated}


async def run_command(
    command: str,
    cwd: str = "/workspace",
    stdin: str = None,
    timeout: int = 30,
    env: dict = None,
) -> dict:
    cwd_path = ensure_allowed(cwd, must_exist=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    start = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        "bash",
        "-lc",
        command,
        stdin=(
            asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL
        ),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd_path),
        env=merged_env,
    )

    timed_out = False
    try:
        out, err = await asyncio.wait_for(
            process.communicate(stdin.encode("utf-8") if stdin is not None else None),
            timeout=timeout,
        )
        exit_code = process.returncode
    except asyncio.TimeoutError:
        timed_out = True
        process.kill()
        out, err = await process.communicate()
        exit_code = -1

    return {
        "command": command,
        "cwd": str(cwd_path),
        "stdout": out.decode("utf-8", errors="replace"),
        "stderr": err.decode("utf-8", errors="replace"),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_seconds": round(time.monotonic() - start, 3),
    }
