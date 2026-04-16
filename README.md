# Sentiment Analysis REST API

REST API built with FastAPI that uses a HuggingFace transformer model to classify text as positive or negative. Built for organisations to analyse service user feedback.

## How it works

The app has three endpoints:
- `POST /analyze` takes in text and runs it through a sentiment model, returns whether its positive or negative with a confidence score
- `GET /results` gives you the average scores and label counts across all the requests made so far
- `GET /` is just a health check to see if the server is running

The main design decision was using the **Strategy pattern** for the model. There's an abstract `SentimentModel` class in `model.py` that defines the interface, and `TransformerSentimentModel` in `transformer_model.py` is the actual implementation that uses HuggingFace. The API only talks to the abstract class, so if I wanted to swap in a different model (like VADER) I'd just make a new class that inherits from `SentimentModel` and change one line in `main.py`.

Results are stored in a `ResultsStore` object that just keeps everything in a list in memory - nothing gets saved to a database, which is what the brief asked for.

**Model used:** `distilbert-base-uncased-finetuned-sst-2-english` - a smaller, faster version of BERT fine-tuned on SST-2 for binary sentiment classification.

## Project structure

```
sentiment-api/
-model.py               # Abstract SentimentModel class (ABC)
-transformer_model.py   # HuggingFace transformer implementation
-store.py               # In-memory results storage
-main.py                # FastAPI app and endpoints
-test_main.py           # Tests (pytest + mocks)
-requirements.txt       # Dependencies
-README.md
```

## Setup

### What you need
- Python 3.10+
- pip

### Steps

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/sentiment-api.git
   cd sentiment-api
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # mac/linux
   venv\Scripts\activate      # windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the server:
   ```bash
   uvicorn main:app --reload
   ```
   API runs at `http://127.0.0.1:8000`. First time will download the model (~260MB).

5. Check the auto-generated docs:
   - Swagger UI: `http://127.0.0.1:8000/docs`
   - ReDoc: `http://127.0.0.1:8000/redoc`

### Running tests
```bash
pytest test_main.py -v
```

## API Endpoints

### `POST /analyze`

Send text, get back a sentiment label and confidence score.

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "The service was absolutely wonderful!"}'
```

Or in Python:
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/analyze",
    json={"text": "The service was absolutely wonderful!"}
)
print(response.json())
```

Response:
```json
{
    "text": "The service was absolutely wonderful!",
    "label": "POSITIVE",
    "score": 0.9998
}
```

### `GET /results`

Get aggregate stats for all analyses since the server started.

```bash
curl http://127.0.0.1:8000/results
```

Response:
```json
{
    "total_requests": 42,
    "average_score": 0.9134,
    "label_counts": {"POSITIVE": 35, "NEGATIVE": 7},
    "label_percentages": {"POSITIVE": 83.33, "NEGATIVE": 16.67}
}
```

### `GET /`

Health check - just tells you the API is running.

```json
{"status": "running", "api": "Sentiment Analysis API", "version": "1.0.0"}
```

## Why I made the choices I did

- **Strategy pattern:** Lets me swap models without touching the API. If I wanted to try VADER or a fine-tuned model later, I just make a new class.
- **FastAPI over Flask:** Auto-generates docs, validates requests with Pydantic, and supports async - less boilerplate.
- **Transformer over VADER:** VADER is rule-based and struggles with sarcasm, negation, and longer sentences. The transformer understands context.
- **Separate ResultsStore class:** Keeps storage logic out of the API routes. Easier to test and avoids the kind of mutable state bugs I had in my previous assignment.
- **Mocked tests:** The tests use `@patch` to mock the model so they run fast without needing to download model weights.
