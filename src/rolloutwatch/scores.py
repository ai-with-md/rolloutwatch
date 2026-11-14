"""Watch the average score, for graders that score on a scale instead of pass or fail.

Many LLM judges give a score from 1 to 5, or from 0 to 1. Turning scores into
pass or fail throws information away: a drop from 4.6 to 4.2 can hide inside
"4 or above passes". ScoreMonitor compares average scores directly, with the
same always-valid interval as the pass rate monitor, so checking every day
still keeps wrong calls at or below ``alpha``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

from .monitor import NOT_WORSE, WATCHING, WORSE, _check_level, always_valid_p, boundary


@dataclass
class _Sums:
    n: int = 0
    total: float = 0.0
    squares: float = 0.0

    def add(self, n: int, total: float, squares: float) -> None:
        self.n += n
        self.total += total
        self.squares += squares

    @property
    def mean(self) -> float:
        return self.total / self.n

    def variance_of_mean(self, prior_variance: float) -> float:
        """Variance of the average, with the sample variance pulled gently toward a cautious prior.

        Early on, a handful of identical scores would give a variance near zero
        and an overconfident interval. Two pseudo-observations at the variance
        of a uniform spread over the score range prevent that, and their weight
        fades as results come in.
        """
        n = self.n
        sample = max(0.0, (self.squares - n * self.mean ** 2) / (n - 1)) if n > 1 else prior_variance
        blended = ((n - 1) * sample + 2 * prior_variance) / (n + 1)
        return blended / n


@dataclass(frozen=True)
class ScoreCheck:
    """The score monitor's reading after one update."""

    check: int
    new_count: int
    new_mean: float
    old_count: Optional[int]
    old_mean: float
    difference: float
    low: float
    high: float
    p_value: float
    status: str

    def summary(self) -> str:
        old = (
            f"baseline {self.old_mean:.3g}"
            if self.old_count is None
            else f"old {self.old_mean:.3g} ({self.old_count} scores)"
        )
        return (
            f"Check {self.check}: new {self.new_mean:.3g} ({self.new_count} scores) vs {old}. "
            f"Difference {self.difference:+.3f} (always-valid interval {self.low:+.3f} to {self.high:+.3f}). "
            f"{self.status.upper()}."
        )

    def __str__(self) -> str:
        return self.summary()


@dataclass
class ScoreMonitor:
    """Watch the average score of a new prompt or model against the old one.

    Args:
        drop: the smallest drop in average score worth catching, in score
            units, for example 0.2 on a 1 to 5 scale.
        score_range: lowest and highest possible score, for example (1, 5).
            Scores outside it are rejected.
        alpha: chance of any wrong call over the whole rollout, however often
            you check.
        baseline: the old average score, when everyone is on the new version
            and the old average is known well from history.
        min_samples: scores needed in each group before any call is made.

    Pass each batch's scores to ``update``, or its count, average and
    standard deviation to ``update_summary`` if you only have aggregates.
    """

    drop: float = 0.2
    score_range: Tuple[float, float] = (0.0, 1.0)
    alpha: float = 0.05
    baseline: Optional[float] = None
    min_samples: int = 50
    history: List[ScoreCheck] = field(default_factory=list)

    def __post_init__(self) -> None:
        low, high = self.score_range
        if not high > low:
            raise ValueError("score_range must be (lowest, highest) with highest above lowest")
        if not 0 < self.drop < high - low:
            raise ValueError("drop must be positive and smaller than the score range")
        _check_level(self.alpha, "alpha")
        if self.baseline is not None and not low <= self.baseline <= high:
            raise ValueError("baseline must be inside the score range")
        if self.min_samples < 2:
            raise ValueError("min_samples must be at least 2")
        self._prior = (high - low) ** 2 / 12
        self._new, self._old = _Sums(), _Sums()
        self._min_p = 1.0

    @property
    def status(self) -> str:
        return self.history[-1].status if self.history else WATCHING

    def _sums(self, scores: Optional[Iterable[float]], name: str) -> Tuple[int, float, float]:
        values = [float(s) for s in scores]
        low, high = self.score_range
        bad = next((v for v in values if not low <= v <= high or math.isnan(v)), None)
        if bad is not None:
            raise ValueError(f"{name}: score {bad!r} is outside the score range {low:g} to {high:g}")
        return len(values), sum(values), sum(v * v for v in values)

    def update(self, new_scores: Iterable[float], old_scores: Optional[Iterable[float]] = None) -> ScoreCheck:
        """Add one batch of scores and return the current reading."""
        self._check_arms(old_scores is not None)
        new = self._sums(new_scores, "new")
        old = self._sums(old_scores, "old") if old_scores is not None else None
        return self._add(new, old)

    def update_summary(
        self,
        new_count: int,
        new_mean: float,
        new_sd: float,
        old_count: Optional[int] = None,
        old_mean: Optional[float] = None,
        old_sd: Optional[float] = None,
    ) -> ScoreCheck:
        """Add one batch from its count, average and standard deviation (with n - 1)."""
        has_old = old_count is not None or old_mean is not None or old_sd is not None
        self._check_arms(has_old)

        def sums(n, mean, sd, name):
            if n is None or mean is None or sd is None:
                raise ValueError(f"{name}: give the count, average and standard deviation")
            low, high = self.score_range
            if n < 0 or sd < 0 or (n and not low <= mean <= high):
                raise ValueError(f"{name}: count and standard deviation must not be negative, and the average must be in range")
            return n, n * mean, (n - 1) * sd * sd + n * mean * mean if n else 0.0

        new = sums(new_count, new_mean, new_sd, "new")
        old = sums(old_count, old_mean, old_sd, "old") if has_old else None
        return self._add(new, old)

    def _check_arms(self, has_old: bool) -> None:
        if self.baseline is None and not has_old:
            raise ValueError("without a baseline, pass the old version's scores too")
        if self.baseline is not None and has_old:
            raise ValueError("with a fixed baseline, pass only the new version's scores")

    def _add(self, new, old) -> ScoreCheck:
        self._new.add(*new)
        if old is not None:
            self._old.add(*old)
        if self._new.n == 0 or (self.baseline is None and self._old.n == 0):
            raise ValueError("no scores yet")
        variance = self._new.variance_of_mean(self._prior)
        if self.baseline is None:
            old_mean = self._old.mean
            variance += self._old.variance_of_mean(self._prior)
            enough = min(self._new.n, self._old.n) >= self.min_samples
        else:
            old_mean = self.baseline
            enough = self._new.n >= self.min_samples
        diff = self._new.mean - old_mean
        half = boundary(variance, self.drop, self.alpha)
        low, high = diff - half, diff + half
        self._min_p = min(self._min_p, always_valid_p(diff, variance, self.drop))
        if self.status != WATCHING:
            status = self.status
        elif not enough:
            status = WATCHING
        elif high < 0:
            status = WORSE
        elif low > -self.drop:
            status = NOT_WORSE
        else:
            status = WATCHING
        reading = ScoreCheck(
            check=len(self.history) + 1,
            new_count=self._new.n,
            new_mean=self._new.mean,
            old_count=None if self.baseline is not None else self._old.n,
            old_mean=old_mean,
            difference=diff,
            low=low,
            high=high,
            p_value=self._min_p,
            status=status,
        )
        self.history.append(reading)
        return reading

    def report(self) -> str:
        if not self.history:
            return "No checks yet."
        lines = [c.summary() for c in self.history]
        decided = next((c for c in self.history if c.status != WATCHING), None)
        if decided is None:
            lines.append(f"No decision yet: a drop of {self.drop:g} or more can neither be confirmed nor ruled out.")
        elif decided.status == WORSE:
            lines.append(f"The new version is reliably worse, first clear at check {decided.check}.")
        else:
            lines.append(f"The new version is not worse by {self.drop:g} or more, first clear at check {decided.check}.")
        return "\n".join(lines)
