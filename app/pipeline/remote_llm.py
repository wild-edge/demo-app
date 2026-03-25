"""Remote LLM summariser using GPT-4o-mini via OpenRouter.

Handles articles where the sentiment classifier returns low confidence
(< 0.85), which typically indicates ambiguous or nuanced content.

Uses the OpenAI SDK pointed at OpenRouter. Cost is kept low by capping
input at 500 characters and output at 150 tokens.

WildEdge auto-instruments the OpenAI client via the `openai` integration.
"""

import os

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
)

SYSTEM_PROMPT = "You are a concise news summariser. Reply with exactly 2 sentences."


class RemoteLLM:
    """Summarises articles using GPT-4o-mini via the OpenRouter API."""

    def __init__(self) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Export it before starting the app."
            )
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def summarise(self, text: str, context: str = "") -> str:
        """Summarise an article in 2 sentences using GPT-4o-mini via OpenRouter.

        Args:
            text: The article text to summarise.
            context: Optional retrieved context (not forwarded to save tokens).

        Returns:
            A 2-sentence summary string, or an unavailability message if the
            API key is missing at call time.
        """
        if not os.environ.get("OPENROUTER_API_KEY", ""):
            return "[Remote summarisation unavailable: OPENROUTER_API_KEY not set]"

        completion = self.client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Article: {text[:500]}"},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        return completion.choices[0].message.content or ""
