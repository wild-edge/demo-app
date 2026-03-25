"""Remote LLM summariser using GPT-4o-mini via OpenRouter.

Handles articles where the sentiment classifier returns low confidence
(< 0.85), which typically indicates ambiguous or nuanced content.

Uses the OpenAI SDK pointed at OpenRouter. Cost is kept low by capping
input at 500 characters and output at 150 tokens.

WildEdge auto-instruments the OpenAI client via the `openai` integration.
"""

import os
from collections.abc import Iterator

from openai import OpenAI

SYSTEM_PROMPT = "You are a concise news summariser. Reply with exactly 2 sentences."


class RemoteLLM:
    """Summarises articles using GPT-4o-mini via the OpenRouter API."""

    def __init__(self, client: OpenAI) -> None:
        self.client = client

    def stream(self, text: str) -> Iterator[str]:
        """Yield summary tokens as they are generated."""
        if not os.environ.get("OPENROUTER_API_KEY", ""):
            yield "[Remote summarisation unavailable: OPENROUTER_API_KEY not set]"
            return

        response = self.client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Article: {text[:500]}"},
            ],
            max_tokens=150,
            temperature=0.3,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in response:
            if chunk.choices:
                yield chunk.choices[0].delta.content or ""
