"""Tool-call plumbing for `hf_chat_model.py` (Gate 1).

Deep Agents drives tool use through LangChain's `create_agent`, which calls
`request.model.bind_tools(...)` unconditionally whenever any tool is
configured (`langchain/agents/factory.py:993`). `BaseChatModel.bind_tools`
raises `NotImplementedError` in the base class, so a chat model wrapping a
local HF checkpoint **must** implement it or the graph dies on its first
model call. That is the trap this module exists to close.

The other half is output parsing. Qwen-family chat templates render a tool
schema block and train against Hermes-style
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`, so a local HF
model's tool calls arrive as text that has to be parsed back into LangChain
`ToolCall` dicts. `apply_chat_template(..., tools=[...])` renders the
schema side; `extract_tool_calls` handles the return side.

Verified against a real Qwen2.5-0.5B-Instruct tokenizer: the instruction
the template emits matches exactly what `extract_tool_calls` parses.

Salvaged from a scratch prototype (see PLAN.md §11). Deliberately carries
no model-loading, realignment, or transport logic — the released
RecursiveLink checkpoints and `inference_utils` primitives own all of that.

Usage sketch inside a `BaseChatModel` subclass:

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.bind(tools=format_tools(tools), **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        prompt = self.tokenizer.apply_chat_template(
            lc_messages_to_hf(messages),
            tools=kwargs.get("tools"),
            tokenize=False,
            add_generation_prompt=True,
        )
        ...                                    # HF generate(...)
        content, tool_calls = extract_tool_calls(text)
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content=content, tool_calls=tool_calls))])
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def lc_messages_to_hf(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """LangChain messages -> the role/content dicts `apply_chat_template` wants."""
    role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    return [{"role": role_map.get(m.type, m.type), "content": m.text} for m in messages]


def format_tools(
    tools: Sequence[BaseTool | Callable | dict] | None,
) -> list[dict[str, Any]] | None:
    """Tool specs -> OpenAI-function-schema dicts for `apply_chat_template(tools=...)`.

    An entry that can't be converted is dropped rather than raising: one bad
    tool spec shouldn't stop every other tool from reaching the model.
    """
    if not tools:
        return None
    formatted: list[dict[str, Any]] = []
    for tool in tools:
        try:
            formatted.append(convert_to_openai_tool(tool))
        except Exception:  # noqa: BLE001 - conversion failure modes vary by tool shape
            continue
    return formatted or None


def extract_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Split raw generated text into visible content and LangChain `ToolCall`s.

    A block that isn't valid JSON, or is valid JSON missing `name`, is
    dropped rather than raising. Malformed tool calls are a real failure
    mode under sampling, and one shouldn't take down an otherwise usable
    response — the surrounding text is still worth returning.

    Returns `(content, tool_calls)` where `content` has the `<tool_call>`
    blocks stripped out.
    """
    tool_calls: list[dict[str, Any]] = []
    for match in TOOL_CALL_RE.finditer(text):
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict) or "name" not in parsed:
            continue
        args = parsed.get("arguments", {})
        tool_calls.append(
            {
                "name": parsed["name"],
                "args": args if isinstance(args, dict) else {},
                "id": f"call_{uuid.uuid4().hex[:12]}",
            }
        )
    content = TOOL_CALL_RE.sub("", text).strip()
    return content, tool_calls
