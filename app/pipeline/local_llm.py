"""Local LLM summariser using Llama 3.2 1B Instruct (Q4_K_M GGUF).

Downloads `bartowski/Llama-3.2-1B-Instruct-GGUF` from HuggingFace Hub.
Runs entirely on-device via llama.cpp with no API calls or token cost.

Articles routed here have high-confidence sentiment (>= 0.85), meaning
they are clear-cut content that does not need a more capable model.

WildEdge auto-instruments llama-cpp-python via the `gguf` integration.
"""

from collections.abc import Iterator

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

REPO_ID = "bartowski/Llama-3.2-1B-Instruct-GGUF"
MODEL_FILE = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"


class LocalLLM:
    """Summarises articles using Llama 3.2 1B Instruct running via llama.cpp."""

    def __init__(self) -> None:
        model_path = hf_hub_download(REPO_ID, MODEL_FILE)
        self.llm = Llama(model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)

    def summarise(self, text: str, context: str = "") -> str:
        """Summarise an article in 2 sentences using the local model."""
        return "".join(self.stream(text))

    def stream(self, text: str) -> Iterator[str]:
        """Yield summary tokens as they are generated."""
        prompt = f"[INST] Summarise in 2 sentences.\nArticle: {text[:400]}[/INST]"
        for chunk in self.llm(prompt, max_tokens=120, temperature=0.3, stream=True):
            yield chunk["choices"][0]["text"]
