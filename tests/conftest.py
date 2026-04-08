"""Shared fixtures for the test suite."""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import app.main as main_module


@pytest.fixture()
def client():
    """TestClient with pipeline components mocked out.

    The real pipeline is exercised in test_pipeline.py.
    """
    mock_we = MagicMock()
    mock_classifier = MagicMock()
    mock_classifier.predict.return_value = {"label": "POSITIVE", "confidence": 0.95}
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.0] * 384
    mock_local_llm = MagicMock()
    mock_local_llm.stream.side_effect = lambda text: iter(["Article ", "summary."])
    mock_remote_llm = MagicMock()
    mock_openai = MagicMock()

    main_module.we = mock_we
    main_module.classifier = mock_classifier
    main_module.embedder = mock_embedder
    main_module.local_llm = mock_local_llm
    main_module.remote_llm = mock_remote_llm
    main_module.openai_client = mock_openai
    main_module.article_store.clear()

    original_lifespan = main_module.app.router.lifespan_context

    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    main_module.app.router.lifespan_context = noop_lifespan

    with TestClient(main_module.app) as c:
        yield c

    main_module.app.router.lifespan_context = original_lifespan
    main_module.article_store.clear()
