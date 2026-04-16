"""
Abstract base class for sentiment models.
Uses the Strategy pattern so we can swap different models
without changing the API code.
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


class SentimentResult(BaseModel):
    """Holds the output from a sentiment analysis."""

    label: str = Field(description="Sentiment label, e.g. POSITIVE or NEGATIVE")
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1",
    )


class SentimentModel(ABC):
    """
    Base class that any sentiment model must inherit from.
    Forces subclasses to implement analyze() so the API always
    knows what method to call regardless of which model is used.
    """

    @abstractmethod
    def analyze(self, text: str) -> SentimentResult:
        """Take in text and return a SentimentResult."""
        pass
