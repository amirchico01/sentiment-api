"""
Sentiment Analysis REST API
Built with FastAPI. Takes in text and returns whether its
positive or negative using a transformer model.

The app uses the Strategy pattern — it depends on the SentimentModel
abstract class, not directly on the transformer. So if I wanted to
swap in a different model (like VADER), I'd just need to make a new
class that inherits from SentimentModel.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model import SentimentModel, SentimentResult
from transformer_model import TransformerSentimentModel
from store import ResultsStore


# App setup

app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "REST API that classifies the emotional tone of text "
        "using a pre-trained transformer model."
    ),
    version="1.0.0",
)

# load model once at startup, reuse for every request
model: SentimentModel = TransformerSentimentModel()
store: ResultsStore = ResultsStore()


# Request/Response schemas

class AnalyzeRequest(BaseModel):
    """What the client sends to POST /analyze."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The text to analyse for sentiment.",
        json_schema_extra={"example": "The service was absolutely wonderful!"},
    )


class AnalyzeResponse(BaseModel):
    """What POST /analyze sends back."""

    text: str = Field(description="The original input text.")
    label: str = Field(description="POSITIVE or NEGATIVE.")
    score: float = Field(description="Model confidence, 0 to 1.")


class AverageResponse(BaseModel):
    """What GET /results sends back."""

    total_requests: int = Field(description="How many analyses have been done.")
    average_score: float | None = Field(
        description="Mean confidence score, or null if nothing has been analysed yet."
    )
    label_counts: dict[str, int] = Field(
        description="How many times each label appeared."
    )
    label_percentages: dict[str, float] = Field(
        description="Percentage split between labels."
    )


# Endpoints

@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyse sentiment of submitted text",
    tags=["Sentiment"],
)
def analyze_sentiment(request: AnalyzeRequest):
    """
    Send text and get back a sentiment label with a confidence score.

    **Example:**
    ```json
    {"text": "The service was absolutely wonderful!"}
    ```
    """
    try:
        result: SentimentResult = model.analyze(request.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    store.add(request.text, result)

    return AnalyzeResponse(
        text=request.text,
        label=result.label,
        score=result.score,
    )


@app.get(
    "/results",
    response_model=AverageResponse,
    summary="Get average sentiment scores",
    tags=["Results"],
)
def get_results():
    """
    Returns stats across all the analyses done since the server started.
    Everything is in memory so it resets when the server restarts.
    """
    summary = store.get_summary()
    return AverageResponse(**summary)


@app.get("/", summary="Health check", tags=["System"])
def root():
    """Quick check that the API is running."""
    return {
        "status": "running",
        "api": "Sentiment Analysis API",
        "version": "1.0.0",
    }
