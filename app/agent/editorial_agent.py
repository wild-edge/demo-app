"""Editorial review agent.

Runs a short agentic loop that reviews the most recent processed articles
and produces an editorial summary. Demonstrates WildEdge tracing for
multi-step agent workflows with tool use.

Triggered manually via POST /agent/run. Uses GPT-4o-mini via OpenRouter
with a hard cap of 3 reasoning steps to limit token usage.
"""

import json
import os
import uuid

from openai import OpenAI
from wildedge import WildEdge

openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
)

SYSTEM_PROMPT = (
    "You are an editorial assistant. Review article stats, flag anything concerning, "
    "then write a concise 2-sentence editorial briefing."
)

TASK = (
    "Review today's articles. Check stats, flag anything that needs attention, "
    "write a 2-sentence editorial briefing."
)

MAX_STEPS = 3

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_article_stats",
            "description": (
                "Return counts of processed articles broken down by sentiment, "
                "routing target, and flagged status."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_article",
            "description": "Mark an article as flagged for editorial review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "article_id": {
                        "type": "string",
                        "description": "The id field of the article to flag.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Short reason for flagging.",
                    },
                },
                "required": ["article_id", "reason"],
            },
        },
    },
]


def get_article_stats(article_store: list[dict]) -> str:
    total = len(article_store)
    positive = sum(1 for a in article_store if a["sentiment"]["label"] == "POSITIVE")
    negative = total - positive
    local = sum(1 for a in article_store if a["routed_to"] == "local")
    remote = total - local
    flagged = sum(1 for a in article_store if a.get("flagged", False))
    return json.dumps(
        {
            "total": total,
            "positive": positive,
            "negative": negative,
            "local": local,
            "remote": remote,
            "flagged": flagged,
        }
    )


def flag_article(article_id: str, reason: str, article_store: list[dict]) -> str:
    for article in article_store:
        if article["id"] == article_id:
            article["flagged"] = True
            return json.dumps({"flagged": article_id, "reason": reason})
    return json.dumps({"error": f"Article {article_id!r} not found"})


def run_editorial_review(article_store: list[dict], we: WildEdge) -> dict:
    """Run one editorial review pass over recent articles.

    Args:
        article_store: The in-memory list of processed articles from app.main.
        we: The WildEdge client initialised at application startup.

    Returns:
        {"summary": str, "flagged_ids": list[str], "steps_taken": int}
    """
    run_id = str(uuid.uuid4())
    flagged_ids: list[str] = []
    steps_taken = 0

    with we.trace(agent_id="editorial-agent", run_id=run_id):
        with we.span(
            kind="retrieval",
            name="load_article_store",
            input_summary=f"loading {len(article_store)} articles",
        ) as span:
            article_ids = [a["id"] for a in article_store]
            span.output_summary = f"loaded {len(article_ids)} article ids"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": TASK},
        ]

        step_index = 1
        summary = ""

        while step_index <= MAX_STEPS:
            with we.span(
                kind="agent_step",
                name="reason",
                step_index=step_index,
                input_summary=TASK[:200],
            ) as span:
                # On the final step, forbid tool calls so the model must
                # produce a text answer rather than exhausting the limit.
                is_last_step = step_index >= MAX_STEPS
                response = openai_client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="none" if is_last_step else "auto",
                    max_tokens=200,
                )
                choice = response.choices[0]
                span.output_summary = choice.finish_reason

            messages.append(choice.message.model_dump(exclude_none=True))
            steps_taken = step_index

            if choice.finish_reason == "tool_calls":
                step_index += 1
                for tool_call in choice.message.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    with we.span(
                        kind="tool",
                        name=tool_name,
                        input_summary=json.dumps(arguments)[:200],
                    ) as tool_span:
                        if tool_name == "get_article_stats":
                            result = get_article_stats(article_store)
                        elif tool_name == "flag_article":
                            result = flag_article(
                                arguments.get("article_id", ""),
                                arguments.get("reason", ""),
                                article_store,
                            )
                            parsed = json.loads(result)
                            if "flagged" in parsed:
                                flagged_ids.append(parsed["flagged"])
                        else:
                            result = json.dumps({"error": f"Unknown tool: {tool_name}"})
                        tool_span.output_summary = result[:200]

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )
            else:
                summary = choice.message.content or ""
                break

        if not summary and steps_taken >= MAX_STEPS:
            summary = "[Editorial review reached step limit without a final answer.]"

    return {
        "summary": summary,
        "flagged_ids": flagged_ids,
        "steps_taken": steps_taken,
    }
