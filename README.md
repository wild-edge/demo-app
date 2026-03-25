# WildEdge Demo App

![Demo Screenshot](assets/Demo%20Screenshot.png)

A FastAPI app that processes news articles through a three-stage ML pipeline, with every inference tracked by the [WildEdge SDK](https://wildedge.dev). Learn more at [wildedge.dev](https://wildedge.dev).

The demo covers all three model origin styles that WildEdge is designed to monitor:

| Stage | Model | Format | Deployment |
|---|---|---|---|
| Sentiment classification | DistilBERT SST-2 | ONNX | On-device |
| Sentence embedding | all-MiniLM-L6-v2 | ONNX | On-device |
| Summarisation (clear articles) | Llama 3.2 1B Instruct | GGUF | On-device |
| Summarisation (ambiguous articles) | GPT-4o-mini | API | Remote (OpenRouter) |

An editorial review agent demonstrates WildEdge's agentic trace support.

## Pipeline

```
Article text
    |
    +-> DistilBERT (ONNX)    -- sentiment label + confidence score
    |
    +-> MiniLM (ONNX)        -- embedding -> context retrieval
    |
    +-> confidence >= 0.85?
            | yes                        | no
            v                            v
      Llama 3.2 1B (GGUF)        GPT-4o-mini (OpenRouter)
      local, no token cost       remote, ~$0.0001/article
            |                            |
            +------------+--------------+
                         v
                   2-sentence summary
```

Articles with high-confidence sentiment (>= 0.85) route to the local Llama model at zero cost. Articles below that threshold route to GPT-4o-mini.

All three model types send inference events to WildEdge, so you can compare latency, token metrics, and quality signals across ONNX, GGUF, and API models in one place.

## What to explore in WildEdge

- **Models**: three models listed side by side: two ONNX classifiers, one GGUF LLM, one API LLM
- **Traces**: one trace per article showing all pipeline steps with per-step timing
- **Traces (agent)**: multi-step agentic trace with tool calls and a retrieval span
- **Quality**: DistilBERT confidence scores feed into PSI drift tracking over time
- **Hardware**: on-device metrics for the ONNX and GGUF stages (accelerator, thermal state)

## Prerequisites

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) package manager
- ~2 GB free disk space for model downloads (cached after first run)
- Apple Silicon recommended for GPU-accelerated GGUF inference (works on CPU too, but slower)

## Quick start

```bash
# 1. Clone and enter the directory
git clone https://github.com/wild-edge/wildedge-demo-app
cd wildedge-demo-app

# 2. Set environment variables
export OPENROUTER_API_KEY="sk-or-..."     # required
export WILDEDGE_DSN="https://..."         # optional: enables cloud monitoring

# 3. Run
./demo.sh
```

Open http://localhost:8002 in your browser.

On first run, the app downloads three models from HuggingFace Hub (~900 MB total). Subsequent starts use the local cache and are fast.

## What to try

1. **Tech launch sample**: clearly positive sentiment, routes to local Llama
2. **Product recall sample**: clearly negative sentiment, routes to local Llama
3. **Earnings report sample**: ambiguous content, routes to GPT-4o-mini
4. **Run editorial review**: triggers the agent loop, which calls tools and flags articles

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | API key for [OpenRouter](https://openrouter.ai). Used for GPT-4o-mini calls. |
| `WILDEDGE_DSN` | No | WildEdge ingest endpoint. Events are buffered locally if not set. |

## Cost

Most articles route to the local Llama model (free). Only articles with classifier confidence below 0.85 hit OpenRouter. With the three provided samples, the earnings report triggers one remote call at approximately $0.0001.

## Architecture

```
wildedge-demo-app/
├── demo.sh                     # Start script: uv sync + wildedge run uvicorn
├── pyproject.toml
└── app/
    ├── main.py                 # FastAPI app, lifespan init, pipeline orchestration
    ├── pipeline/
    │   ├── classifier.py       # DistilBERT ONNX sentiment classifier
    │   ├── embedder.py         # MiniLM ONNX sentence embedder
    │   ├── local_llm.py        # Llama 3.2 1B GGUF summariser
    │   └── remote_llm.py       # GPT-4o-mini via OpenRouter
    ├── agent/
    │   └── editorial_agent.py  # Agent loop with tool use
    └── static/
        └── index.html          # Single-page UI
```
