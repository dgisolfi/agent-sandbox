"""Sandbox API

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tools import (ALLOWED_ROOTS, CommandRequest, ErrorBody, FileListRequest,
                   FileReadRequest, FileWriteRequest, Response, SandboxFsError,
                   list_dir, read_file, run_command, write_file)

app = FastAPI(
    title="Code Sandbox",
    description="Versioned API for sandbox command execution and file IO.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def ok(data: dict) -> Response:
    return Response(ok=True, data=data, error=None)


def fail(error_type: str, message: str, details: dict = None) -> Response:
    return Response(
        ok=False,
        data=None,
        error=ErrorBody(type=error_type, message=message, details=details or {}),
    )


def handle_error(exc: Exception) -> Response:
    if isinstance(exc, SandboxFsError):
        return fail(exc.error_type, exc.message, exc.details)
    return fail("internal_error", str(exc))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "3.0.0",
        "allowed_roots": [str(root) for root in ALLOWED_ROOTS],
    }


@app.post("/v1/command", response_model=Response)
async def command(req: CommandRequest):
    try:
        return ok(
            await run_command(req.command, req.cwd, req.stdin, req.timeout, req.env)
        )
    except Exception as exc:
        return handle_error(exc)


@app.post("/v1/files/read", response_model=Response)
async def files_read(req: FileReadRequest):
    try:
        return ok(read_file(req.path))
    except Exception as exc:
        return handle_error(exc)


@app.post("/v1/files/write", response_model=Response)
async def files_write(req: FileWriteRequest):
    try:
        return ok(write_file(req.path, req.content, req.mode, req.create_parents))
    except Exception as exc:
        return handle_error(exc)


@app.post("/v1/files/list", response_model=Response)
async def files_list(req: FileListRequest):
    try:
        return ok(list_dir(req.path, req.recursive, req.max_entries))
    except Exception as exc:
        return handle_error(exc)
