"""Interactive chat interface for the LLM assistant."""
from __future__ import annotations
from quantflow.assistant.provider import ClaudeProvider
from quantflow.assistant.tools import TOOL_DEFINITIONS, execute_tool


class ChatSession:
    """Manages a conversation with the LLM assistant."""

    def __init__(self, provider: ClaudeProvider | None = None) -> None:
        self.provider = provider
        self.messages: list[dict] = []
        self.system_prompt = (
            "You are QuantFlow Assistant, an AI-powered research assistant for quantitative trading. "
            "You help users analyze markets, run backtests, explain quant concepts, and debug strategies. "
            "You have access to tools that let you fetch data, run backtests, and list available strategies. "
            "Be concise and data-driven in your responses."
        )

    def send(self, user_message: str) -> str:
        """Send a message and get a response."""
        if self.provider is None:
            return "No LLM provider configured. Set ANTHROPIC_API_KEY environment variable."

        self.messages.append({"role": "user", "content": user_message})

        response = self.provider.chat(
            messages=self.messages,
            tools=TOOL_DEFINITIONS,
            system=self.system_prompt,
        )

        # Handle tool calls
        while response.get("tool_calls"):
            # Reconstruct full content blocks for the assistant message (required by Anthropic API)
            content_blocks = []
            if response.get("content"):
                content_blocks.append({"type": "text", "text": response["content"]})
            for tc in response["tool_calls"]:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                })
            self.messages.append({"role": "assistant", "content": content_blocks})

            # Add tool results as a user message
            tool_results = []
            for tool_call in response["tool_calls"]:
                result = execute_tool(tool_call["name"], tool_call["input"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": result,
                })
            self.messages.append({"role": "user", "content": tool_results})

            # Get next response
            response = self.provider.chat(
                messages=self.messages,
                tools=TOOL_DEFINITIONS,
                system=self.system_prompt,
            )

        # Add final assistant response
        assistant_text = response.get("content", "")
        self.messages.append({"role": "assistant", "content": assistant_text})
        return assistant_text


def run_interactive_chat():
    """Run an interactive chat session in the terminal."""
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Set ANTHROPIC_API_KEY environment variable to use the chat assistant.")
        print("Get a key at: https://console.anthropic.com/")
        return

    provider = ClaudeProvider(api_key=api_key)
    session = ChatSession(provider=provider)

    print("QuantFlow Assistant (type 'quit' to exit)")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        response = session.send(user_input)
        print(f"\nAssistant: {response}")
