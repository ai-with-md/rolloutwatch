"""Check the monitor's promises against simulated rollouts.

Each test runs hundreds of rollouts with a known truth and checks the monitor
after every batch, the way people actually use it. Seeds are fixed so the
tests are repeatable.
"""

import random

import rolloutwatch as rw
from rolloutwatch.planning import _binomial

SIMS = 400


def run(rng, days, per_arm, old_rate, new_rate, drop=0.03, baseline=None, min_samples=50):
    m = rw.QualityMonitor(drop=drop, baseline=baseline, min_samples=min_samples)
    for _ in range(days):
        if baseline is None:
            m.update(_binomial(rng, per_arm, new_rate), per_arm, _binomial(rng, per_arm, old_rate), per_arm)
        else:
            m.update(_binomial(rng, per_arm, new_rate), per_arm)
    return m


def test_interval_holds_at_every_check_at_once():
    rng = random.Random(1)
    truth, held = -0.02, 0
    for _ in range(SIMS):
        m = run(rng, 40, 100, 0.80, 0.78)
        held += all(c.low <= truth <= c.high for c in m.history)
    assert held / SIMS >= 0.94


def test_checking_daily_keeps_false_worse_calls_under_five_percent():
    rng = random.Random(2)
    false_alarms = sum(run(rng, 60, 150, 0.8, 0.8).status == rw.WORSE for _ in range(SIMS))
    assert false_alarms / SIMS <= 0.05


def test_naive_daily_checks_do_not():
    assert rw.peeking_false_alarms(60, per_check=150, simulations=1000) > 0.2


def test_real_drop_is_rarely_cleared():
    rng = random.Random(3)
    wrongly_cleared = sum(run(rng, 60, 150, 0.8, 0.77).status == rw.NOT_WORSE for _ in range(SIMS))
    assert wrongly_cleared / SIMS <= 0.05


def test_fixed_baseline_false_alarms_stay_low():
    rng = random.Random(4)
    false_alarms = sum(run(rng, 60, 150, None, 0.8, baseline=0.8).status == rw.WORSE for _ in range(SIMS))
    assert false_alarms / SIMS <= 0.05


def test_small_daily_batches_stay_safe():
    rng = random.Random(5)
    false_alarms = sum(run(rng, 90, 15, 0.85, 0.85).status == rw.WORSE for _ in range(SIMS))
    assert false_alarms / SIMS <= 0.05


def test_real_drop_is_caught():
    rng = random.Random(6)
    caught = sum(run(rng, 60, 150, 0.8, 0.74).status == rw.WORSE for _ in range(SIMS))
    assert caught / SIMS >= 0.95
