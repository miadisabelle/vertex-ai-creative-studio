"""RISE Creative Orientation Evaluation helpers.

This is **not** yet wired into the text arena flow but is added now so future
iterations can leverage the scoring logic to automatically decide winning
models without human vote.
"""

from __future__ import annotations

from typing import Dict, Any
from enum import Enum


class CreativeIndicator(Enum):
    """Basic creative orientation buckets."""

    DESIRED_OUTCOME = "desired_outcome"
    STRUCTURAL_TENSION = "structural_tension"
    ADVANCING = "advancing"
    GAP_THINKING = "gap_thinking"
    PROBLEM_SOLVING = "problem_solving"
    OSCILLATING = "oscillating"


BIAS_TERMS = {
    CreativeIndicator.GAP_THINKING: ["bridge the gap", "close the gap", "fill the gap"],
    CreativeIndicator.PROBLEM_SOLVING: ["solve the problem", "fix the issue", "eliminate"],
    CreativeIndicator.OSCILLATING: ["back and forth", "try harder", "struggle"],
}

CREATIVE_TERMS = {
    CreativeIndicator.DESIRED_OUTCOME: ["desired outcome", "what you want", "bring into"],
    CreativeIndicator.STRUCTURAL_TENSION: ["structural tension", "natural movement"],
    CreativeIndicator.ADVANCING: ["natural progression", "building momentum"],
}


# ---------------------------------------------------------------------------
# Public helpers                                                              
# ---------------------------------------------------------------------------


def evaluate_response(response: str) -> Dict[str, Any]:
    """Return naive creative vs bias scores for *response*."""

    resp_lower = response.lower()
    scores: Dict[str, int] = {}

    def _count_terms(group):
        return sum(1 for term in group if term in resp_lower)

    bias_hits = sum(_count_terms(v) for v in BIAS_TERMS.values())
    creative_hits = sum(_count_terms(v) for v in CREATIVE_TERMS.values())

    total_words = max(1, len(resp_lower.split()))
    return {
        "creative_score": creative_hits / total_words,
        "bias_score": bias_hits / total_words,
    }


def compare_responses(resp1: str, resp2: str) -> str:
    """Return "resp1", "resp2", or "tie" depending on creative orientation score."""

    ev1 = evaluate_response(resp1)
    ev2 = evaluate_response(resp2)

    score1 = ev1["creative_score"] - ev1["bias_score"]
    score2 = ev2["creative_score"] - ev2["bias_score"]

    if abs(score1 - score2) < 1e-4:
        return "tie"
    return "resp1" if score1 > score2 else "resp2" 