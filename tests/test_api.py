"""API integration tests using a mocked pipeline."""

import json


class TestHealth:
    def test_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestListArticles:
    def test_empty_on_startup(self, client):
        r = client.get("/articles")
        assert r.status_code == 200
        assert r.json() == []


class TestProcessArticle:
    def _events(self, response):
        return [json.loads(line) for line in response.text.strip().splitlines()]

    def test_returns_expected_events(self, client):
        r = client.post("/articles", json={"text": "Apple reports record quarterly earnings."})
        assert r.status_code == 200
        event_types = [e["event"] for e in self._events(r)]
        assert event_types == ["classification", "routing", "token", "token", "done"]

    def test_classification_event_shape(self, client):
        r = client.post("/articles", json={"text": "Stocks fall on weak jobs data."})
        classification = next(e for e in self._events(r) if e["event"] == "classification")
        assert classification["label"] in ("POSITIVE", "NEGATIVE")
        assert isinstance(classification["confidence"], float)

    def test_done_event_has_id(self, client):
        r = client.post("/articles", json={"text": "Tech rally continues into Q4."})
        done = next(e for e in self._events(r) if e["event"] == "done")
        assert "id" in done
        assert "processed_at" in done

    def test_article_appears_in_list(self, client):
        client.post("/articles", json={"text": "Fed holds rates steady."})
        articles = client.get("/articles").json()
        assert len(articles) == 1
        assert "sentiment" in articles[0]
        assert "summary" in articles[0]

    def test_empty_text_rejected(self, client):
        r = client.post("/articles", json={"text": "   "})
        assert r.status_code == 422

    def test_missing_text_rejected(self, client):
        r = client.post("/articles", json={})
        assert r.status_code == 422
