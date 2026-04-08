"""Integration tests for pipeline components against real models.

Models are downloaded from HuggingFace Hub on first run and cached locally.
Run slow tests with: pytest -m slow
"""

import pytest

from app.pipeline.classifier import SentimentClassifier
from app.pipeline.embedder import SentenceEmbedder


@pytest.fixture(scope="module")
def classifier():
    return SentimentClassifier()


@pytest.fixture(scope="module")
def embedder():
    return SentenceEmbedder()


class TestSentimentClassifier:
    def test_positive(self, classifier):
        result = classifier.predict("Apple reports record quarterly earnings.")
        assert result["label"] in ("POSITIVE", "NEGATIVE")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_negative(self, classifier):
        result = classifier.predict("Company files for bankruptcy after massive fraud.")
        assert result["label"] == "NEGATIVE"

    def test_confidence_is_float(self, classifier):
        result = classifier.predict("Shares rose slightly today.")
        assert isinstance(result["confidence"], float)


class TestSentenceEmbedder:
    def test_output_shape(self, embedder):
        vec = embedder.embed("Short sentence.")
        assert len(vec) == 384

    def test_output_is_normalised(self, embedder):
        import math

        vec = embedder.embed("Another sentence for embedding.")
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-4

    def test_different_texts_differ(self, embedder):
        a = embedder.embed("Cat sat on a mat.")
        b = embedder.embed("Stock markets surged on strong jobs data.")
        assert a != b


@pytest.mark.slow
class TestLocalLLM:
    """Downloads a ~800 MB GGUF model. Skipped by default; run with -m slow."""

    @pytest.fixture(scope="class")
    def local_llm(self):
        from app.pipeline.local_llm import LocalLLM

        return LocalLLM()

    def test_stream_yields_text(self, local_llm):
        tokens = list(local_llm.stream("OpenAI releases a new model."))
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_summary_is_nonempty(self, local_llm):
        summary = local_llm.summarise("OpenAI releases a new model.")
        assert len(summary.strip()) > 0
