"""LLM provider adapters."""
from __future__ import annotations
from typing import Protocol
from anthropic import Anthropic


class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    def chat(self, messages: list[dict], tools: list[dict] | None = None, system: str | None = None) -> dict: ...


class ClaudeProvider:
    """Claude API provider via Anthropic SDK."""

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-20250514") -> None:
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def chat(self, messages: list[dict], tools: list[dict] | None = None, system: str | None = None) -> dict:
        kwargs = {"model": self.model, "max_tokens": 4096, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if system:
            kwargs["system"] = system
        response = self.client.messages.create(**kwargs)

        # Extract text and tool use from response
        result = {"role": "assistant", "content": "", "tool_calls": []}
        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                result["tool_calls"].append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        result["stop_reason"] = response.stop_reason
        return result
