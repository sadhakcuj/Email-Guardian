"""Request and response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    """One message to classify. Provide ``text``, or ``subject`` and ``body``."""

    subject: str = Field(default="", max_length=1000)
    body: str = Field(default="", max_length=50_000)
    text: str = Field(default="", max_length=50_000)

    @field_validator("subject", "body", "text")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    def combined(self) -> str:
        parts = [self.subject, self.body, self.text]
        return "\n".join(p for p in parts if p)


class PredictResponse(BaseModel):
    score: float = Field(description="Spam probability in [0, 1].")
    verdict: Literal["spam", "review", "ham"]
    is_spam: bool
    threshold: float
    signals: list[str]
    model: str
    request_id: str


class BatchRequest(BaseModel):
    items: list[Message] = Field(min_length=1)


class BatchItem(BaseModel):
    score: float
    verdict: Literal["spam", "review", "ham"]
    is_spam: bool
    signals: list[str]


class BatchResponse(BaseModel):
    results: list[BatchItem]
    threshold: float
    model: str
    request_id: str


class FeedbackRequest(Message):
    correct_label: Literal["spam", "ham"]
    request_id: str | None = Field(default=None, max_length=64)


class FeedbackResponse(BaseModel):
    stored: bool
    total_feedback: int
