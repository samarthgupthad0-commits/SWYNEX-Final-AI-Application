#!/usr/bin/env python3
"""Local employee support request triage demo."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


DEFAULT_DATASET = Path(__file__).parent / "data" / "sample_requests.csv"


class DatasetError(ValueError):
    """Raised when the training dataset is missing or malformed."""


def load_requests(path: str | Path) -> tuple[list[str], list[str]]:
    """Load and validate fictional request text and category labels from CSV."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as dataset_file:
            reader = csv.DictReader(dataset_file)
            required_columns = {"text", "category"}
            if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                raise DatasetError("CSV must have 'text' and 'category' columns.")

            texts: list[str] = []
            categories: list[str] = []
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise DatasetError(f"CSV row {row_number} has too many fields.")
                text = (row.get("text") or "").strip()
                category = (row.get("category") or "").strip()
                if not text or not category:
                    raise DatasetError(
                        f"CSV row {row_number} must have non-empty text and category."
                    )
                texts.append(text)
                categories.append(category)
    except DatasetError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetError(f"Could not read dataset '{source}': {exc}") from exc

    if not texts:
        raise DatasetError(f"Dataset '{source}' contains no request rows.")
    category_counts = Counter(categories)
    if len(category_counts) < 2:
        raise DatasetError("Dataset must contain at least two categories.")
    if min(category_counts.values()) < 2:
        raise DatasetError("Each category must contain at least two request rows.")
    return texts, categories


def create_model() -> Pipeline:
    """Create the TF-IDF and logistic regression text-classification pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    strip_accents="unicode",
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_model(dataset_path: str | Path = DEFAULT_DATASET) -> Pipeline:
    texts, categories = load_requests(dataset_path)
    return create_model().fit(texts, categories)


def classify(model: Pipeline, text: str) -> dict[str, str | float | bool]:
    """Return a suggested category; the result is not an automatic route."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Request text must not be empty.")

    probabilities = model.predict_proba([text.strip()])[0]
    best_index = int(probabilities.argmax())
    category = str(model.classes_[best_index])
    return {
        "suggested_category": category,
        "model_score": round(float(probabilities[best_index]), 4),
        "human_review_required": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Suggest an employee-support category for human review."
    )
    parser.add_argument("--text", required=True, help="Fictional request text to classify")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET,
        help="CSV training data (default: data/sample_requests.csv)",
    )
    args = parser.parse_args(argv)

    try:
        result = classify(train_model(args.data), args.text)
    except (DatasetError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2))
    print("This is a suggestion only. A human must review and decide the final routing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
