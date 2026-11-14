"""Watch quality after a prompt or model change, checking as often as you like.

The usual way to check a change goes wrong in a quiet way. Run a significance
test every day, stop the first time it says "worse", and the false alarm rate
is no longer 5%: over 30 daily checks it is closer to 25%. The same happens in
the other direction with "looks fine".

This module uses an always-valid confidence sequence for the difference in
pass rates (the normal mixture boundary behind the mixture sequential
probability ratio test, Johari and others, 2017; see also Howard and others,
2021). The interval is valid at every check at once, so you can look after
every batch and stop whenever it gives an answer, and the 5% still holds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

WORSE = "worse"
NOT_WORSE = "not worse"
WATCHING = "keep watching"


def _check_level(value: float, name: str) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value!r}")


def _rate_variance(passed: int, total: int) -> float:
    """Variance of an observed pass rate, nudged away from 0 and 1 so early checks are not overconfident."""
    p = (passed + 0.5) / (total + 1)
    return p * (1 - p) / total


def boundary(variance: float, mixing_sd: float, alpha: float) -> float:
    """Half width of the always-valid interval for an estimate with this variance.

    ``mixing_sd`` tunes where the boundary is tightest: set it to the size of
    change you care about. Any positive value keeps the guarantee.
    """
    if variance <= 0:
        return 0.0
    tau2 = mixing_sd * mixing_sd
    ratio = (variance + tau2) / variance
    return math.sqrt(variance * (variance + tau2) / tau2 * (2 * math.log(1 / alpha) + math.log(ratio)))


def always_valid_p(estimate: float, variance: float, mixing_sd: float, null: float = 0.0) -> float:
    """1 / mixture likelihood ratio against ``null``. Take the running minimum over checks."""
    if variance <= 0:
        return 1.0
    tau2 = mixing_sd * mixing_sd
    diff = estimate - null
    log_lr = 0.5 * math.log(variance / (variance + tau2)) + tau2 * diff * diff / (2 * variance * (variance + tau2))
    return min(1.0, math.exp(-log_lr))


@dataclass(frozen=True)
class Check:
    """The monitor's reading after one update."""

    check: int
    new_passed: int
    new_total: int
    old_passed: Optional[int]
    old_total: Optional[int]
    new_rate: float
    old_rate: float
    difference: float
    low: float
    high: float
    p_value: float
    status: str

    def summary(self) -> str:
        if self.old_total is None:
            base = f"new {self.new_rate:.1%} ({self.new_passed}/{self.new_total}) vs baseline {self.old_rate:.1%}"
        else:
            base = (
                f"new {self.new_rate:.1%} ({self.new_passed}/{self.new_total}) vs old "
                f"{self.old_rate:.1%} ({self.old_passed}/{self.old_total})"
            )
        return (
            f"Check {self.check}: {base}. Difference {self.difference:+.1%} "
            f"(always-valid interval {self.low:+.1%} to {self.high:+.1%}). {self.status.upper()}."
        )

    def __str__(self) -> str:
        return self.summary()


@dataclass
class QualityMonitor:
    """Watch the pass rate of a new prompt or model against the old one.

    Two ways to use it:

    * **Side by side:** part of the traffic still goes to the old version.
      Call ``update(new_passed, new_total, old_passed, old_total)`` with each
      batch. This is the safer design, because it is not fooled by changes
      in traffic over time.
    * **Against a fixed baseline:** everyone gets the new version, and the
      old pass rate is known well from history. Create the monitor with
      ``baseline=0.82`` and call ``update(new_passed, new_total)``.

    Args:
        drop: the smallest drop worth catching, for example 0.03 for three
            points. The monitor says "not worse" once it can rule out a drop
            this large, and it tunes the boundary to catch drops of this size.
        alpha: chance of any wrong call over the whole rollout, however often
            you check.
        min_samples: graded requests needed in each group before any call is
            made. The intervals rest on a normal approximation, which needs a
            few dozen results to be reliable.

    Each update adds a batch to the running totals; pass only the new results
    since the last update.
    """

    drop: float = 0.03
    alpha: float = 0.05
    baseline: Optional[float] = None
    min_samples: int = 50
    history: List[Check] = field(default_factory=list)

    def __post_init__(self) -> None:
        _check_level(self.drop, "drop")
        _check_level(self.alpha, "alpha")
        if self.baseline is not None:
            _check_level(self.baseline, "baseline")
        if self.min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        self._new_passed = self._new_total = 0
        self._old_passed = self._old_total = 0
        self._min_p = 1.0

    @property
    def status(self) -> str:
        return self.history[-1].status if self.history else WATCHING

    def update(
        self,
        new_passed: int,
        new_total: int,
        old_passed: Optional[int] = None,
        old_total: Optional[int] = None,
    ) -> Check:
        """Add one batch of graded results and return the current reading."""
        side_by_side = self.baseline is None
        if side_by_side and (old_passed is None or old_total is None):
            raise ValueError("without a baseline, pass the old version's results too (old_passed, old_total)")
        if not side_by_side and (old_passed is not None or old_total is not None):
            raise ValueError("with a fixed baseline, pass only the new version's results")
        for passed, total, name in ((new_passed, new_total, "new"), (old_passed, old_total, "old")):
            if total is None:
                continue
            if total < 0 or passed < 0 or passed > total:
                raise ValueError(f"{name}: passed must be between 0 and total")
        self._new_passed += new_passed
        self._new_total += new_total
        if side_by_side:
            self._old_passed += old_passed
            self._old_total += old_total
        return self._read()

    def _read(self) -> Check:
        n_new, s_new = self._new_total, self._new_passed
        if n_new == 0 or (self.baseline is None and self._old_total == 0):
            raise ValueError("no results yet")
        new_rate = s_new / n_new
        variance = _rate_variance(s_new, n_new)
        if self.baseline is None:
            old_rate = self._old_passed / self._old_total
            variance += _rate_variance(self._old_passed, self._old_total)
            enough = min(n_new, self._old_total) >= self.min_samples
        else:
            old_rate = self.baseline
            enough = n_new >= self.min_samples
        diff = new_rate - old_rate
        half = boundary(variance, self.drop, self.alpha)
        low, high = diff - half, diff + half
        self._min_p = min(self._min_p, always_valid_p(diff, variance, self.drop))
        previous = self.status
        if previous != WATCHING:
            status = previous  # a decision, once made, stands
        elif not enough:
            status = WATCHING
        elif high < 0:
            status = WORSE
        elif low > -self.drop:
            status = NOT_WORSE
        else:
            status = WATCHING
        reading = Check(
            check=len(self.history) + 1,
            new_passed=s_new,
            new_total=n_new,
            old_passed=None if self.baseline is not None else self._old_passed,
            old_total=None if self.baseline is not None else self._old_total,
            new_rate=new_rate,
            old_rate=old_rate,
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
            lines.append(
                f"No decision yet: a drop of {self.drop:.1%} or more can neither be confirmed nor ruled out."
            )
        elif decided.status == WORSE:
            lines.append(f"The new version is reliably worse, first clear at check {decided.check}.")
        else:
            lines.append(
                f"The new version is not worse by {self.drop:.1%} or more, first clear at check {decided.check}."
            )
        return "\n".join(lines)
