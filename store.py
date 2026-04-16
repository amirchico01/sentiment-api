"""
In-memory storage for sentiment results.
Results are not saved to a database — they only live while the server
is running, which is what the brief asked for.

I put this in its own class instead of just using a global list in main.py
because it makes testing easier and keeps the storage logic separate
from the API routing logic.
"""

from threading import Lock
from statistics import mean
from model import SentimentResult


class ResultsStore:
    """
    Stores analysis results and calculates aggregate stats.
    Uses a threading lock so it doesnt break if multiple
    requests come in at the same time.
    """

    def __init__(self):
        self._results: list[dict] = []
        self._lock = Lock()

    def add(self, text: str, result: SentimentResult) -> None:
        """Save a new analysis result."""
        with self._lock:
            self._results.append({
                "text": text,
                "label": result.label,
                "score": result.score,
            })

    def get_summary(self) -> dict:
        """Calculate averages and counts across all stored results."""
        with self._lock:
            total = len(self._results)

            if total == 0:
                return {
                    "total_requests": 0,
                    "average_score": None,
                    "label_counts": {},
                    "label_percentages": {},
                }

            avg_score = round(mean(r["score"] for r in self._results), 4)

            # count how many of each label we have
            label_counts: dict[str, int] = {}
            for r in self._results:
                label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

            label_percentages = {
                label: round((count / total) * 100, 2)
                for label, count in label_counts.items()
            }

            return {
                "total_requests": total,
                "average_score": avg_score,
                "label_counts": label_counts,
                "label_percentages": label_percentages,
            }

    @property
    def total(self) -> int:
        """How many results are currently stored."""
        with self._lock:
            return len(self._results)

    def clear(self) -> None:
        """Wipe all results. Mainly used in tests."""
        with self._lock:
            self._results.clear()
