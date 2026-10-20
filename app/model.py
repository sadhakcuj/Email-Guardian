"""Model loading and inference."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import sklearn

log = logging.getLogger("spam.model")

Verdict = Literal["spam", "review", "ham"]

_URL = re.compile(r"(https?://|www\.)\S+", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
_MONEY = re.compile(r"[$£€₹]\s?\d|\b\d+\s?(?:usd|gbp|eur|inr|rs\.?)\b", re.I)
_URGENCY = re.compile(
    r"\b(urgent|act now|limited time|final notice|claim|winner|won|prize|free|verify your account|"
    r"password|suspended|click here)\b",
    re.I,
)


@dataclass(frozen=True)
class Prediction:
    score: float
    verdict: Verdict
    signals: list[str]


def explain_signals(text: str) -> list[str]:
    """Transparent, rule-based cues shown next to the model score.

    These do not drive the prediction; they help a human reviewer see why a message looks suspicious.
    """
    signals: list[str] = []
    if _URL.search(text):
        signals.append("contains_url")
    if _PHONE.search(text):
        signals.append("contains_phone_number")
    if _MONEY.search(text):
        signals.append("mentions_money")
    if _URGENCY.search(text):
        signals.append("urgency_or_prize_wording")
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 20 and sum(c.isupper() for c in letters) / len(letters) > 0.5:
        signals.append("mostly_uppercase")
    if text.count("!") >= 3:
        signals.append("many_exclamation_marks")
    return signals


class SpamModel:
    def __init__(self, artifact: dict, metrics: dict | None, review_floor: float) -> None:
        self._pipeline = artifact["pipeline"]
        self.name: str = artifact["model_name"]
        self.threshold: float = float(artifact["threshold"])
        self._frame_input: bool = artifact["text_input"] == "frame"
        self.trained_with_sklearn: str = artifact.get("sklearn_version", "unknown")
        self.trained_at: str = artifact.get("trained_at", "unknown")
        self.metrics = metrics or {}
        # The review band can never reach above the decision threshold.
        self.review_floor = min(review_floor, self.threshold)

    @classmethod
    def load(cls, model_path: Path, metrics_path: Path, review_floor: float) -> "SpamModel":
        artifact = joblib.load(model_path)
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        model = cls(artifact, metrics, review_floor)
        if model.trained_with_sklearn != sklearn.__version__:
            log.warning(
                "scikit-learn version mismatch: trained with %s, running %s. "
                "Retrain with `python -m training.train` if predictions look wrong.",
                model.trained_with_sklearn,
                sklearn.__version__,
            )
        return model

    def predict(self, texts: list[str]) -> list[Prediction]:
        data = pd.Series(texts, name="text")
        inputs = data.to_frame("text") if self._frame_input else data
        scores = self._pipeline.predict_proba(inputs)[:, 1]
        results = []
        for text, score in zip(texts, scores):
            s = float(score)
            if s >= self.threshold:
                verdict: Verdict = "spam"
            elif s >= self.review_floor:
                verdict = "review"
            else:
                verdict = "ham"
            results.append(Prediction(round(s, 4), verdict, explain_signals(text)))
        return results
