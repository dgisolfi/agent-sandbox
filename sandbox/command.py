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
