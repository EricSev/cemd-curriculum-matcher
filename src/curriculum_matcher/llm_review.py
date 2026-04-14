from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any


class MatchReviewer(Protocol):
    def review_match(self, context: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass
class NullMatchReviewer:
    def review_match(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "decision": "not_reviewed",
            "reasoning": "No LLM reviewer configured.",
            "confidence": 0.0,
        }
