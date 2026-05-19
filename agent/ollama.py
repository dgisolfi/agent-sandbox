"""Ollama

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import json
import logging
import time
from typing import Any
from uuid import uuid4

import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama

from agent.utils import ParseError, extract_json

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code != 200:
                return False
            models = [m.get("name", "") for m in response.json().get("models", [])]
            if not any(
                m == self.model or m.startswith(self.model + ":") for m in models
            ):
                logger.warning(
                    "Model '%s' is not pulled. Run: ollama pull %s",
                    self.model,
                    self.model,
                )
                return False
            return True
        except requests.RequestException:
            return False


TOOL_WRAPPER_INSTRUCTIONS = """\
You are running inside a LangChain agent that can execute tools, but this model
does not provide native tool-call messages. Return exactly one JSON object.

To call a tool:
{
  "thought": "Brief reason for the tool call.",
  "tool": "run_command",
  "args": {"command": "pytest -q", "cwd": "/workspace", "timeout": 120}
}

To finish:
{
  "final": "Concise completion summary with verification performed."
}

Tool results will appear as TOOL: messages showing the sandbox response.
After reading a tool result, decide your next action and respond with one JSON object.

Do not include Markdown fences or any text outside the JSON object.
"""


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _format_messages(messages: list[BaseMessage]) -> str:
    lines = [TOOL_WRAPPER_INSTRUCTIONS, ""]
    for message in messages:
        role = getattr(message, "type", message.__class__.__name__)
        lines.append(f"{role.upper()}:")
        lines.append(_message_text(message))
        lines.append("")
    lines.append("AI:")
    return "\n".join(lines)


class OllamaToolAdapter(BaseChatModel):
    """Adapt plain Ollama text models into LangChain tool-call chat models."""

    model_name: str
    base_url: str
    temperature: float = 0.1
    timeout: int = 600
    max_parse_repairs: int = 1
    bound_tools: Any = ()

    @property
    def _llm_type(self) -> str:
        return "ollama-tool-calling-adapter"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
        }

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        return self.model_copy(update={"bound_tools": tuple(tools)})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        client = OllamaClient(self.base_url, self.model_name, timeout=self.timeout)
        prompt = self._prompt(messages)
        t = time.monotonic()
        raw = client.generate(prompt)
        logger.info("LLM responded in %.1fs | %d chars", time.monotonic() - t, len(raw))
        data = self._parse_or_repair(client, raw)
        message = self._message_from_data(data, raw)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _prompt(self, messages: list[BaseMessage]) -> str:
        tool_docs = []
        for item in self.bound_tools:
            args_schema = getattr(item, "args_schema", None)
            schema = {}
            if args_schema is not None:
                try:
                    schema = args_schema.model_json_schema()
                except Exception:
                    schema = {}
            tool_docs.append(
                {
                    "name": item.name,
                    "description": item.description,
                    "args_schema": schema,
                }
            )
        return (
            _format_messages(messages)
            + "\nAvailable tools:\n"
            + json.dumps(tool_docs, indent=2, ensure_ascii=False)
            + "\n"
        )

    def _parse_or_repair(self, client: OllamaClient, raw: str) -> dict:
        last_error = ""
        for attempt in range(self.max_parse_repairs + 1):
            try:
                data = extract_json(raw)
                if "final" not in data and "tool" not in data:
                    raise ParseError("Response must contain 'tool'+'args' or 'final'")
                return data
            except ParseError as exc:
                last_error = str(exc)
                if attempt >= self.max_parse_repairs:
                    logger.warning("Failed to parse tool wrapper response: %s", exc)
                    return {
                        "final": raw.strip() or f"Model response parse error: {exc}"
                    }
                raw = client.generate(
                    "Repair this response into exactly one valid JSON object. "
                    "It must use either {'tool': ..., 'args': ...} or {'final': ...}.\n"
                    f"Error: {last_error}\nResponse:\n{raw}"
                )
        return {"final": raw}

    def _message_from_data(self, data: dict, raw: str) -> AIMessage:
        if "final" in data:
            logger.info("Action: final | %.120s", data.get("final", ""))
            return AIMessage(content=str(data.get("final") or ""))

        tool_name = str(data.get("tool", "")).strip()
        args = data.get("args") if isinstance(data.get("args"), dict) else {}
        thought = str(data.get("thought") or "")
        known_tools = {tool.name for tool in self.bound_tools}
        if tool_name and (not known_tools or tool_name in known_tools):
            logger.info("Action: tool=%s | args=%s", tool_name, str(args)[:80])
            return AIMessage(
                content=thought,
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": args,
                        "id": f"call_{uuid4().hex}",
                        "type": "tool_call",
                    }
                ],
            )

        return AIMessage(
            content=(
                "The model requested an unknown or malformed tool call. "
                f"Raw response: {raw}"
            )
        )


def build_model(model_name: str, base_url: str, mode: str = "wrapped"):
    """Build a LangChain-compatible model.

    Use mode='native' to rely on ChatOllama's native tool calling. The default
    wrapper preserves compatibility with plain text models.
    """
    normalized = (mode or "wrapped").lower()
    if normalized == "native":
        return ChatOllama(model=model_name, base_url=base_url, temperature=0.1)
    if normalized not in {"wrapped", "auto"}:
        raise ValueError(f"Unknown model adapter mode: {mode}")
    return OllamaToolAdapter(model_name=model_name, base_url=base_url)
