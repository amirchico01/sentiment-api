"""
Tests for the Sentiment Analysis API.
Run with: pytest test_main.py -v

I'm using pytest with fixtures and @patch to mock the model,
so the tests don't need the actual HuggingFace model to be
downloaded. This is the same approach we used in Week 7 with
the BookClient tests.
"""

import pytest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from model import SentimentResult
from store import ResultsStore


# Fixtures

@pytest.fixture
def results_store():
    """Fresh store for each test."""
    return ResultsStore()


@pytest.fixture
def mock_positive_result():
    return SentimentResult(label="POSITIVE", score=0.9987)


@pytest.fixture
def mock_negative_result():
    return SentimentResult(label="NEGATIVE", score=0.9432)


# ResultsStore tests

class TestResultsStore:

    def test_empty_store(self, results_store):
        summary = results_store.get_summary()
        assert summary["total_requests"] == 0
        assert summary["average_score"] is None
        assert summary["label_counts"] == {}

    def test_add_one_result(self, results_store, mock_positive_result):
        results_store.add("Great service!", mock_positive_result)
        summary = results_store.get_summary()
        assert summary["total_requests"] == 1
        assert summary["average_score"] == 0.9987

    def test_add_mixed_results(self, results_store, mock_positive_result, mock_negative_result):
        results_store.add("Love it!", mock_positive_result)
        results_store.add("Hate it!", mock_negative_result)
        summary = results_store.get_summary()
        assert summary["total_requests"] == 2
        assert summary["label_counts"]["POSITIVE"] == 1
        assert summary["label_counts"]["NEGATIVE"] == 1

    def test_percentages_add_up(self, results_store, mock_positive_result, mock_negative_result):
        results_store.add("Good", mock_positive_result)
        results_store.add("Bad", mock_negative_result)
        summary = results_store.get_summary()
        total_pct = sum(summary["label_percentages"].values())
        assert abs(total_pct - 100.0) < 0.1

    def test_total_property(self, results_store, mock_positive_result):
        assert results_store.total == 0
        results_store.add("Test", mock_positive_result)
        assert results_store.total == 1

    def test_clear(self, results_store, mock_positive_result):
        results_store.add("Test", mock_positive_result)
        results_store.clear()
        assert results_store.total == 0


# API tests (model is mocked so we don't need HuggingFace)

class TestAnalyzeEndpoint:

    @patch("main.model")
    def test_positive_text(self, mock_model):
        mock_model.analyze.return_value = SentimentResult(label="POSITIVE", score=0.9991)

        from main import app, store
        store.clear()
        client = TestClient(app)

        response = client.post("/analyze", json={"text": "Absolutely wonderful service!"})
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "POSITIVE"
        assert 0 <= data["score"] <= 1

    @patch("main.model")
    def test_negative_text(self, mock_model):
        mock_model.analyze.return_value = SentimentResult(label="NEGATIVE", score=0.9876)

        from main import app, store
        store.clear()
        client = TestClient(app)

        response = client.post("/analyze", json={"text": "Terrible, want a refund."})
        assert response.status_code == 200
        assert response.json()["label"] == "NEGATIVE"

    @patch("main.model")
    def test_empty_text_rejected(self, mock_model):
        from main import app
        client = TestClient(app)
        response = client.post("/analyze", json={"text": ""})
        assert response.status_code == 422

    @patch("main.model")
    def test_missing_text_field(self, mock_model):
        from main import app
        client = TestClient(app)
        response = client.post("/analyze", json={})
        assert response.status_code == 422

    @patch("main.model")
    def test_model_crash_returns_500(self, mock_model):
        """If the model throws an error the API should return 500."""
        mock_model.analyze.side_effect = RuntimeError("Model crashed")

        from main import app
        client = TestClient(app)
        response = client.post("/analyze", json={"text": "Test input"})
        assert response.status_code == 500


class TestResultsEndpoint:

    @patch("main.model")
    def test_empty_results(self, mock_model):
        from main import app, store
        store.clear()
        client = TestClient(app)

        response = client.get("/results")
        assert response.status_code == 200
        assert response.json()["total_requests"] == 0

    @patch("main.model")
    def test_results_go_up(self, mock_model):
        mock_model.analyze.return_value = SentimentResult(label="POSITIVE", score=0.95)

        from main import app, store
        store.clear()
        client = TestClient(app)

        client.post("/analyze", json={"text": "Great!"})
        client.post("/analyze", json={"text": "Amazing!"})

        response = client.get("/results")
        assert response.json()["total_requests"] == 2


class TestHealthCheck:

    def test_root(self):
        from main import app
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "running"
