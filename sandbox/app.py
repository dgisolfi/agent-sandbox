"""Sandbox API

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from process import run_command

app = FastAPI(
    title="Code Sandbox",
    description="Versioned API for sandbox command execution and file IO.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def ok(data: dict) -> Response:
    return Response(ok=True, data=data, error=None)


def fail(error_type: str, message: str, details: dict = None) -> Response:
    return Response(
        ok=False,
        data=None,
        error=ErrorBody(type=error_type, message=message, details=details or {}),
    )


def handle_error(exc: Exception) -> Response:
    return fail("internal_error", str(exc))


@app.post("/v1/command", response_model=Response)
async def command(req: CommandRequest):
    try:
        return ok(await run_command(req.command, req.cwd, req.stdin, req.env))
    except Exception as exc:
        return handle_error(exc)
