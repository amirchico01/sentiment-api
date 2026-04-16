"""
Concrete sentiment model using a HuggingFace transformer.
This is the actual model that gets used by the API — it implements
the SentimentModel interface defined in model.py.

I chose distilbert-base-uncased-finetuned-sst-2-english because its
a lighter version of BERT that still gets good accuracy for sentiment
classification, and its one of the recommended models from the
HuggingFace documentation.
"""

from transformers import pipeline as hf_pipeline
from model import SentimentModel, SentimentResult


class TransformerSentimentModel(SentimentModel):
    """
    Loads a pretrained transformer model and uses it to classify
    text as POSITIVE or NEGATIVE with a confidence score.

    The model gets loaded once when this object is created, so we
    dont have to reload it on every request.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self._model_name = model_name
        self._pipeline = hf_pipeline("sentiment-analysis", model=model_name)

    def analyze(self, text: str) -> SentimentResult:
        """Run the model on the input text and return the result."""
        try:
            prediction = self._pipeline(text)[0]
        except Exception as exc:
            raise RuntimeError(f"Model inference failed: {exc}") from exc

        return SentimentResult(
            label=prediction["label"],
            score=round(prediction["score"], 4),
        )
