"""Sentiment classifier using DistilBERT fine-tuned on SST-2 (ONNX).

Downloads `Xenova/distilbert-base-uncased-finetuned-sst-2-english` from
HuggingFace Hub on first use. Returns a label ("POSITIVE"/"NEGATIVE") and
a confidence score in [0, 1].

WildEdge auto-instruments ONNX Runtime sessions via the `onnx` integration.
"""

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

REPO_ID = "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
LABELS = ["NEGATIVE", "POSITIVE"]


class SentimentClassifier:
    """Runs DistilBERT SST-2 sentiment classification via ONNX Runtime."""

    def __init__(self) -> None:
        model_path = hf_hub_download(REPO_ID, "onnx/model.onnx")
        self.session = ort.InferenceSession(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

    def predict(self, text: str) -> dict:
        """Classify sentiment of text.

        Returns a dict with keys:
          - label: "POSITIVE" or "NEGATIVE"
          - confidence: float in [0, 1]
        """
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        feed = {k: v for k, v in inputs.items()}
        logits = self.session.run(None, feed)[0][0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        idx = int(np.argmax(probs))
        return {"label": LABELS[idx], "confidence": float(probs[idx])}
