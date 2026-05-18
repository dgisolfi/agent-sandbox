"""Command

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import asyncio
import os

from filesystem import ensure_allowed


async def run_command(
    command: str,
    cwd: str = "/workspace",
    stdin: str = None,
    env: dict = None,
) -> dict:
    cwd_path = ensure_allowed(cwd, must_exist=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

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

    try:
        out, err = await asyncio.wait_for(
            process.communicate(stdin.encode("utf-8") if stdin is not None else None),
        )
        exit_code = process.returncode
    except asyncio.TimeoutError:
        process.kill()
        exit_code = -1

    return {
        "command": command,
        "cwd": str(cwd_path),
        "exit_code": exit_code,
    }


def ensure_allowed(raw_path: str, must_exist: bool = False):
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path("/workspace") / path
    resolved = path.resolve(strict=False)

    for denied in DENIED_ROOTS:
        try:
            resolved.relative_to(denied)
            raise Exception("permission_denied", f"Path is denied: {resolved}")
        except ValueError:
            pass

    return resolved

def read_file(path: str) -> dict:
    resolved = ensure_allowed(path, must_exist=True)

    content = resolved.read_text(encoding="utf-8", errors="replace")
    stat = resolved.stat()


    return {"path": str(resolved), "content": content}




def write_file(path: str, content: str, mode: str = "overwrite", create_parents: bool = True):
    resolved = ensure_allowed(path)
    if create_parents:
        resolved.parent.mkdir(parents=True, exist_ok=True)

    stat = resolved.stat()
    return {"path": str(resolved), "size_bytes": stat.st_size, "mode": mode}





def list_dir(path: str, recursive: bool = False, max_entries: int = 200) -> dict:
    resolved = ensure_allowed(path, must_exist=True)
  
    entries = []
    truncated = False
    for child in sorted(resolved["files"]):
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
