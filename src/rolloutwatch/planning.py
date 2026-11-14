"""Plan a rollout before it starts, by simulation.

How many days until a drop of a given size is caught, how many until the new
version is cleared, and how badly naive daily checks mislead. Everything is
simulated with the same monitor you will run, so the answers match its
behaviour exactly rather than an approximation of it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist
from typing import List, Optional

from .monitor import NOT_WORSE, WATCHING, WORSE, QualityMonitor, _check_level

_NORMAL = NormalDist()


def _binomial(rng: random.Random, n: int, p: float) -> int:
    """Binomial draw; exact for small batches, normal approximation for large ones."""
    if n <= 0:
        return 0
    if n * p * (1 - p) < 30:
        return sum(1 for _ in range(n) if rng.random() < p)
    draw = round(rng.gauss(n * p, math.sqrt(n * p * (1 - p))))
    return max(0, min(n, draw))


def _quantile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


@dataclass(frozen=True)
class RolloutPlan:
    baseline: float
    true_change: float
    per_day: int
    new_share: float
    days: int
    decided_share: float
    right_share: float
    median_days: Optional[float]
    slow_days: Optional[float]
    expected_status: str

    def summary(self) -> str:
        if self.true_change <= -1e-12:
            change = f"a real drop of {-self.true_change:.1%}"
        elif self.true_change >= 1e-12:
            change = f"a real gain of {self.true_change:.1%}"
        else:
            change = "no real change"
        split = (
            "against a fixed baseline"
            if self.new_share >= 1
            else f"with {self.new_share:.0%} of traffic on the new version"
        )
        head = f"Baseline {self.baseline:.1%}, {change}, {self.per_day:,} graded requests a day {split}:"
        if self.median_days is None:
            return f"{head}\n  Usually no answer within {self.days} days ({self.decided_share:.0%} of rollouts decided)."
        return (
            f"{head}\n  Answer \"{self.expected_status}\" in about {self.median_days:.0f} days "
            f"(1 in 10 rollouts take {self.slow_days:.0f} or more). "
            f"Decided within {self.days} days: {self.decided_share:.0%}. Right call when decided: {self.right_share:.0%}."
        )

    def __str__(self) -> str:
        return self.summary()


def plan_rollout(
    baseline: float,
    true_change: float,
    per_day: int,
    new_share: float = 0.5,
    drop: float = 0.03,
    days: int = 60,
    alpha: float = 0.05,
    fixed_baseline: bool = False,
    simulations: int = 500,
    seed: int = 0,
) -> RolloutPlan:
    """How long until the monitor gives an answer, if the change really does ``true_change``?

    Args:
        baseline: the old version's pass rate.
        true_change: the change to simulate, for example -0.05 for a five
            point drop, or 0 for no change.
        per_day: graded requests per day, both versions together.
        new_share: share of those on the new version (side by side).
        drop: the monitor's smallest drop worth catching.
        fixed_baseline: compare against a known baseline instead; all graded
            requests then go to the new version.
    """
    _check_level(baseline, "baseline")
    if not 0 < baseline + true_change < 1:
        raise ValueError("baseline + true_change must stay between 0 and 1")
    if per_day < 1 or days < 1:
        raise ValueError("per_day and days must be at least 1")
    if not fixed_baseline:
        _check_level(new_share, "new_share")
    rng = random.Random(seed)
    new_rate = baseline + true_change
    n_new = per_day if fixed_baseline else max(1, round(per_day * new_share))
    n_old = 0 if fixed_baseline else max(1, per_day - n_new)
    right = WORSE if true_change <= -drop else NOT_WORSE if true_change >= 0 else None
    decided_days, right_calls, decided = [], 0, 0
    for _ in range(simulations):
        monitor = QualityMonitor(drop=drop, alpha=alpha, baseline=baseline if fixed_baseline else None)
        for day in range(1, days + 1):
            new_passed = _binomial(rng, n_new, new_rate)
            if fixed_baseline:
                reading = monitor.update(new_passed, n_new)
            else:
                reading = monitor.update(new_passed, n_new, _binomial(rng, n_old, baseline), n_old)
            if reading.status != WATCHING:
                decided += 1
                decided_days.append(day)
                right_calls += right is None or reading.status == right
                break
    expected = right or (WORSE if true_change < -drop / 2 else NOT_WORSE)
    return RolloutPlan(
        baseline=baseline,
        true_change=true_change,
        per_day=per_day,
        new_share=1.0 if fixed_baseline else new_share,
        days=days,
        decided_share=decided / simulations,
        right_share=right_calls / decided if decided else 0.0,
        median_days=_quantile(decided_days, 0.5) if decided * 2 >= simulations else None,
        slow_days=_quantile(decided_days, 0.9) if decided * 2 >= simulations else None,
        expected_status=expected,
    )


def peeking_curve(max_checks: int, per_check: int = 200, rate: float = 0.8, alpha: float = 0.05,
                  simulations: int = 2000, seed: int = 0) -> List[float]:
    """False alarm rate of an ordinary test after 1, 2, ... ``max_checks`` checks with no real change.

    Element k-1 is the share of rollouts that an ordinary two-sample test has
    called "different" at least once within k checks, stopping at the first
    alarm. One simulation gives the whole curve, so it only ever rises.
    """
    if max_checks < 1 or per_check < 1:
        raise ValueError("checks and per_check must be at least 1")
    _check_level(rate, "rate")
    _check_level(alpha, "alpha")
    rng = random.Random(seed)
    z = _NORMAL.inv_cdf(1 - alpha / 2)
    first_alarm = [0] * (max_checks + 1)
    for _ in range(simulations):
        s_a = s_b = n = 0
        for check in range(1, max_checks + 1):
            s_a += _binomial(rng, per_check, rate)
            s_b += _binomial(rng, per_check, rate)
            n += per_check
            pooled = (s_a + s_b) / (2 * n)
            se = math.sqrt(2 * pooled * (1 - pooled) / n) if 0 < pooled < 1 else 0
            if se and abs(s_b - s_a) / n / se > z:
                first_alarm[check] += 1
                break
    curve, total = [], 0
    for check in range(1, max_checks + 1):
        total += first_alarm[check]
        curve.append(total / simulations)
    return curve


def peeking_false_alarms(checks: int, per_check: int = 200, rate: float = 0.8, alpha: float = 0.05,
                         simulations: int = 2000, seed: int = 0) -> float:
    """Share of no-change rollouts an ordinary test calls "different" at least once, checking ``checks`` times.

    This is the problem the monitor solves: each single test has a 5% false
    alarm rate, but checking again and again and stopping at the first alarm
    multiplies it.
    """
    return peeking_curve(checks, per_check, rate, alpha, simulations, seed)[-1]
