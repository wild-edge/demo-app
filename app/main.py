"""FastAPI application for the WildEdge demo.

Runs a three-stage article pipeline: sentiment classification (ONNX),
sentence embedding (ONNX), and summarisation (local GGUF or remote API).
Exposes an editorial review agent endpoint.

All pipeline components are loaded once at startup via the FastAPI lifespan
context. WildEdge is initialised before any model is loaded so that the
`onnx`, `gguf`, and `openai` integrations instrument the runtimes correctly.

Start with:
    ./demo.sh
or directly:
    uv run wildedge run --integrations onnx,gguf,openai -- uvicorn app.main:app --reload --port 8002
"""

import asyncio
import json
import os
import pathlib
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import wildedge
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from app.agent.editorial_agent import run_editorial_review
from app.pipeline.classifier import SentimentClassifier
from app.pipeline.embedder import SentenceEmbedder
from app.pipeline.local_llm import LocalLLM
from app.pipeline.remote_llm import RemoteLLM

STATIC_DIR = pathlib.Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Pipeline singletons, populated during lifespan startup
# ---------------------------------------------------------------------------

classifier: SentimentClassifier
embedder: SentenceEmbedder
local_llm: LocalLLM
remote_llm: RemoteLLM
we: wildedge.WildEdge
openai_client: OpenAI

# In-memory article store, newest first, capped at 50 entries.
article_store: list[dict] = []

# Confidence threshold for routing: articles above this go to the local model.
CONFIDENCE_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise WildEdge and load all pipeline models before serving requests."""
    global classifier, embedder, local_llm, remote_llm, we, openai_client

    # Initialise WildEdge first so integrations instrument runtimes at load time.
    we = wildedge.init(
        app_version="0.1.0",
        integrations=["onnx", "gguf", "openai"],
    )

    openai_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", "not-set"),
    )

    print("Loading pipeline models (first run downloads from HuggingFace Hub)...")
    classifier = SentimentClassifier()
    embedder = SentenceEmbedder()
    local_llm = LocalLLM()
    remote_llm = RemoteLLM(openai_client)
    print("Pipeline ready.")

    yield

    we.flush()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WildEdge Demo",
    description="Multi-model article pipeline: ONNX classifiers, local GGUF LLM, and remote API LLM.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ArticleRequest(BaseModel):
    text: str


class AgentRunResponse(BaseModel):
    summary: str
    flagged_ids: list[str]
    steps_taken: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/articles")
async def process_article(req: ArticleRequest) -> StreamingResponse:
    """Run an article through the full pipeline, streaming the result as NDJSON.

    Each line is a JSON object with an "event" field:
      {"event": "classification", "label": "POSITIVE", "confidence": 0.94}
      {"event": "routing", "routed_to": "local", "model_used": "llama-3.2-1b-q4"}
      {"event": "token", "text": "Apple's new chip..."}
      {"event": "done", "id": "abc123", "processed_at": "..."}

    Uses an async generator so that ContextVar-based tracing (we.trace / we.span)
    works correctly — context is stable across yields in an async generator,
    unlike a sync generator iterated via a threadpool.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Article text must not be empty.")

    async def generate():
        with we.trace(agent_id="article-pipeline", run_id=str(uuid.uuid4())):
            # Stage 1: classify sentiment (blocking — run in threadpool)
            sentiment = await asyncio.to_thread(classifier.predict, text)
            yield json.dumps({"event": "classification", **sentiment}) + "\n"

            # Stage 2: embed + fake retrieval (blocking — run in threadpool)
            await asyncio.to_thread(embedder.embed, text)
            with we.span(kind="retrieval", name="vector_search", input_summary=text[:100]) as span:
                await asyncio.sleep(0.08)
                context = f"[background context relevant to: {text[:60]}]"
                span.output_summary = context

            # Stage 3: route and stream tokens
            if sentiment["confidence"] >= CONFIDENCE_THRESHOLD:
                routed_to, model_used = "local", "llama-3.2-1b-q4"
                sync_stream = local_llm.stream
            else:
                routed_to, model_used = "remote", "gpt-4o-mini"
                sync_stream = remote_llm.stream

            yield (
                json.dumps({"event": "routing", "routed_to": routed_to, "model_used": model_used})
                + "\n"
            )

            summary_parts: list[str] = []
            for token in sync_stream(text):
                summary_parts.append(token)
                yield json.dumps({"event": "token", "text": token}) + "\n"

        summary = "".join(summary_parts).strip()
        article_id = uuid.uuid4().hex[:8]
        entry = {
            "id": article_id,
            "text_preview": text[:120] + ("..." if len(text) > 120 else ""),
            "sentiment": sentiment,
            "routed_to": routed_to,
            "model_used": model_used,
            "summary": summary,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "flagged": False,
        }
        article_store.insert(0, entry)
        if len(article_store) > 50:
            article_store.pop()

        yield (
            json.dumps({"event": "done", "id": article_id, "processed_at": entry["processed_at"]})
            + "\n"
        )

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/articles")
def list_articles() -> list[dict]:
    """Return the most recent processed articles, newest first."""
    return article_store


@app.post("/agent/run", response_model=AgentRunResponse)
def run_agent() -> AgentRunResponse:
    """Trigger the editorial review agent over the current article store."""
    result = run_editorial_review(article_store, we, openai_client)
    return AgentRunResponse(**result)


# ---------------------------------------------------------------------------
# Static files, serve index.html at /
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
