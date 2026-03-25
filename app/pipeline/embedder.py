"""Sentence embedder using all-MiniLM-L6-v2 (ONNX).

Downloads the ONNX variant of `sentence-transformers/all-MiniLM-L6-v2`
from HuggingFace Hub. Returns a 384-dimensional normalised embedding vector.

Used in the pipeline to embed the article before fake vector retrieval.
WildEdge auto-instruments ONNX Runtime sessions via the `onnx` integration.
"""

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceEmbedder:
    """Embeds text into a 384-dimensional normalised vector using MiniLM (ONNX)."""

    def __init__(self) -> None:
        model_path = hf_hub_download(REPO_ID, "onnx/model.onnx")
        self.session = ort.InferenceSession(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

    def embed(self, text: str) -> list[float]:
        """Embed text and return a normalised 384-dimensional vector.

        Uses mean pooling over token embeddings, then L2-normalises the result.
        """
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        feed = {k: v for k, v in inputs.items()}
        token_embeddings = self.session.run(None, feed)[0]
        attention_mask = inputs["attention_mask"]
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        mean_pooled = sum_embeddings / sum_mask
        norm = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        normalised = mean_pooled / np.clip(norm, a_min=1e-9, a_max=None)
        return normalised[0].tolist()
