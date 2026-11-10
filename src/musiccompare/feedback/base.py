"""The interface every feedback provider implements: the ML "socket".

A Protocol describes what an object must *have* (here: a ``source`` name and a
``generate`` method) without forcing it to inherit from anything. Any class
with that shape counts as a FeedbackProvider, and mypy checks it.

That's the Strategy pattern: the app asks "a provider" for feedback without
knowing which one. Today it's RuleBasedFeedback; later an ML model with the
same method can be swapped in and nothing else changes.

@runtime_checkable lets ``isinstance(x, FeedbackProvider)`` check the shape at
runtime too (it only checks the names exist, not their signatures).
"""

from typing import Protocol, runtime_checkable

from musiccompare.models import ComparisonResult, FeatureSet, FeedbackItem, PlayerContext


@runtime_checkable
class FeedbackProvider(Protocol):
    source: str  # stored on every item it writes, e.g. "rules-v1"

    def generate(
        self,
        ref: FeatureSet,
        user: FeatureSet,
        result: ComparisonResult,
        context: PlayerContext,
    ) -> list[FeedbackItem]:
        """Advice for this comparison, most worth acting on first."""
        ...
