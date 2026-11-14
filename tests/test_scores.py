"""The score monitor: unit checks, then its promises against simulated rollouts.

Scores are drawn from fixed 1 to 5 distributions (like a judge's ratings) and
from 0 to 1 judge scores, so the true averages are known exactly.
"""

import random

import pytest

import rolloutwatch as rw

# Probabilities of scores 1..5.
GOOD = [0.02, 0.05, 0.13, 0.40, 0.40]  # average 4.11
WORSE_BY_02 = [0.03, 0.07, 0.19, 0.41, 0.30]  # average 3.88, about 0.23 lower
SIMS = 300


def mean_of(probs):
    return sum((i + 1) * p for i, p in enumerate(probs))


def draw(rng, probs, n):
    return rng.choices(range(1, 6), weights=probs, k=n)


def run(rng, days, n, new_probs, old_probs=None, baseline=None, drop=0.2):
    m = rw.ScoreMonitor(drop=drop, score_range=(1, 5), baseline=baseline)
    for _ in range(days):
        if baseline is None:
            m.update(draw(rng, new_probs, n), draw(rng, old_probs, n))
        else:
            m.update(draw(rng, new_probs, n))
    return m


# Unit checks


def test_summary_and_raw_scores_agree():
    scores_new, scores_old = [4, 5, 3, 4, 4, 5, 2, 4], [5, 5, 4, 4, 5, 4, 3, 5]
    a = rw.ScoreMonitor(drop=0.2, score_range=(1, 5), min_samples=2).update(scores_new, scores_old)
    from statistics import mean, stdev

    b = rw.ScoreMonitor(drop=0.2, score_range=(1, 5), min_samples=2).update_summary(
        8, mean(scores_new), stdev(scores_new), 8, mean(scores_old), stdev(scores_old)
    )
    assert a.difference == pytest.approx(b.difference)
    assert a.low == pytest.approx(b.low)
    assert a.high == pytest.approx(b.high)


def test_clear_drop_is_worse_and_equal_scores_are_cleared():
    worse = rw.ScoreMonitor(drop=0.2, score_range=(1, 5))
    for _ in range(10):
        worse.update_summary(500, 3.7, 0.9, 500, 4.1, 0.9)
    assert worse.status == rw.WORSE
    same = rw.ScoreMonitor(drop=0.2, score_range=(1, 5))
    for _ in range(10):
        same.update_summary(500, 4.1, 0.9, 500, 4.1, 0.9)
    assert same.status == rw.NOT_WORSE
    assert "not worse by 0.2" in same.report()


def test_identical_early_scores_do_not_give_false_certainty():
    m = rw.ScoreMonitor(drop=0.2, score_range=(1, 5), min_samples=2)
    reading = m.update([5, 5, 5], [4, 4, 4])
    assert reading.high - reading.low > 1  # wide, despite zero sample variance


def test_fixed_baseline_scores():
    m = rw.ScoreMonitor(drop=0.05, score_range=(0, 1), baseline=0.8)
    reading = m.update_summary(2000, 0.7, 0.2)
    assert reading.status == rw.WORSE
    assert "baseline 0.8" in reading.summary()


@pytest.mark.parametrize(
    "call",
    [
        lambda: rw.ScoreMonitor(score_range=(5, 1)),
        lambda: rw.ScoreMonitor(drop=0, score_range=(1, 5)),
        lambda: rw.ScoreMonitor(drop=5, score_range=(1, 5)),
        lambda: rw.ScoreMonitor(score_range=(1, 5), baseline=7),
        lambda: rw.ScoreMonitor(score_range=(1, 5)).update([6], [4]),
        lambda: rw.ScoreMonitor(score_range=(1, 5)).update([4]),
        lambda: rw.ScoreMonitor(score_range=(1, 5), baseline=4).update([4], [4]),
        lambda: rw.ScoreMonitor(score_range=(1, 5)).update_summary(10, 4.0, -1, 10, 4.0, 1),
        lambda: rw.ScoreMonitor(score_range=(1, 5)).update_summary(10, 4.0, 1, 10, None, 1),
    ],
)
def test_bad_inputs_raise(call):
    with pytest.raises(ValueError):
        call()


# Simulation


def test_interval_holds_at_every_check_at_once():
    rng = random.Random(1)
    truth = mean_of(WORSE_BY_02) - mean_of(GOOD)
    held = sum(
        all(c.low <= truth <= c.high for c in run(rng, 40, 100, WORSE_BY_02, GOOD).history) for _ in range(SIMS)
    )
    assert held / SIMS >= 0.94


def test_daily_checks_keep_false_worse_calls_under_five_percent():
    rng = random.Random(2)
    false_alarms = sum(run(rng, 60, 150, GOOD, GOOD).status == rw.WORSE for _ in range(SIMS))
    assert false_alarms / SIMS <= 0.05


def test_real_drop_is_rarely_cleared():
    rng = random.Random(3)
    cleared = sum(run(rng, 60, 150, WORSE_BY_02, GOOD).status == rw.NOT_WORSE for _ in range(SIMS))
    assert cleared / SIMS <= 0.05


def test_real_drop_is_caught():
    rng = random.Random(4)
    caught = sum(run(rng, 60, 150, WORSE_BY_02, GOOD).status == rw.WORSE for _ in range(SIMS))
    assert caught / SIMS >= 0.9


def test_fixed_baseline_false_alarms_stay_low():
    rng = random.Random(5)
    false_alarms = sum(run(rng, 60, 150, GOOD, baseline=mean_of(GOOD)).status == rw.WORSE for _ in range(SIMS))
    assert false_alarms / SIMS <= 0.05


def test_zero_to_one_judge_scores():
    rng = random.Random(6)
    false_alarms = 0
    for _ in range(SIMS):
        m = rw.ScoreMonitor(drop=0.03, score_range=(0, 1))
        for _ in range(60):
            m.update([rng.betavariate(8, 2) for _ in range(100)], [rng.betavariate(8, 2) for _ in range(100)])
        false_alarms += m.status == rw.WORSE
    assert false_alarms / SIMS <= 0.05
