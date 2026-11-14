import pytest

import rolloutwatch as rw


def test_boundary_and_p_value_agree():
    # At the edge of the interval the always-valid p-value equals alpha.
    for variance in (1e-5, 1e-4, 1e-3):
        radius = rw.boundary(variance, 0.03, 0.05)
        assert rw.always_valid_p(radius, variance, 0.03) == pytest.approx(0.05)


def test_interval_is_wider_than_a_single_test_but_not_much():
    variance = 0.01 ** 2
    radius = rw.boundary(variance, 0.03, 0.05)
    single = 1.96 * 0.01
    assert single < radius < 2 * single


def test_clear_drop_is_called_worse():
    m = rw.QualityMonitor(drop=0.03)
    for _ in range(10):
        reading = m.update(700, 1000, 820, 1000)
    assert reading.status == rw.WORSE
    assert reading.high < 0
    assert "reliably worse" in m.report()


def test_equal_versions_are_cleared():
    m = rw.QualityMonitor(drop=0.03)
    for _ in range(10):
        reading = m.update(800, 1000, 800, 1000)
    assert reading.status == rw.NOT_WORSE
    assert reading.low > -0.03


def test_no_call_before_min_samples():
    m = rw.QualityMonitor(drop=0.03, min_samples=50)
    reading = m.update(0, 40, 40, 40)  # dramatic, but only 40 results each
    assert reading.status == rw.WATCHING


def test_decision_stands_once_made():
    m = rw.QualityMonitor(drop=0.03)
    for _ in range(10):
        m.update(700, 1000, 820, 1000)
    first = next(c for c in m.history if c.status != rw.WATCHING).check
    m.update(1000, 1000, 0, 1000)  # wild later batch
    assert m.status == rw.WORSE
    assert "first clear at check %d" % first in m.report()


def test_fixed_baseline_mode():
    m = rw.QualityMonitor(drop=0.03, baseline=0.82)
    for _ in range(5):
        reading = m.update(700, 1000)
    assert reading.status == rw.WORSE
    assert reading.old_total is None
    assert "baseline 82.0%" in reading.summary()


def test_p_value_only_goes_down():
    m = rw.QualityMonitor(drop=0.03)
    values = [m.update(p, 200, 160, 200).p_value for p in (150, 170, 160, 155, 175)]
    assert all(b <= a for a, b in zip(values, values[1:]))


@pytest.mark.parametrize(
    "call",
    [
        lambda: rw.QualityMonitor(drop=0),
        lambda: rw.QualityMonitor(alpha=1.5),
        lambda: rw.QualityMonitor(baseline=1.2),
        lambda: rw.QualityMonitor().update(10, 20),
        lambda: rw.QualityMonitor(baseline=0.8).update(10, 20, 10, 20),
        lambda: rw.QualityMonitor().update(30, 20, 10, 20),
        lambda: rw.QualityMonitor().update(0, 0, 0, 0),
        lambda: rw.plan_rollout(0.8, 0.3, 100),
        lambda: rw.peeking_false_alarms(0),
    ],
)
def test_bad_inputs_raise(call):
    with pytest.raises(ValueError):
        call()


def test_plan_bigger_drops_are_caught_sooner():
    small = rw.plan_rollout(0.8, -0.04, 300, simulations=100)
    big = rw.plan_rollout(0.8, -0.08, 300, simulations=100)
    assert big.median_days < small.median_days
    assert "worse" in big.summary()


def test_plan_reports_when_traffic_is_too_low():
    plan = rw.plan_rollout(0.8, -0.01, 20, days=10, simulations=50)
    assert plan.median_days is None
    assert "no answer within 10 days" in plan.summary()


def test_peeking_curve_only_rises_and_matches_single_value():
    curve = rw.peeking_curve(20, simulations=300)
    assert all(b >= a for a, b in zip(curve, curve[1:]))
    assert curve[0] < 0.09
    assert rw.peeking_false_alarms(20, simulations=300) == curve[-1]
