"""Did my prompt or model change make things worse? Watch quality with always-valid sequential tests."""

from .monitor import NOT_WORSE, WATCHING, WORSE, Check, QualityMonitor, always_valid_p, boundary
from .planning import RolloutPlan, peeking_curve, peeking_false_alarms, plan_rollout

__all__ = [
    "QualityMonitor",
    "Check",
    "plan_rollout",
    "RolloutPlan",
    "peeking_false_alarms",
    "peeking_curve",
    "boundary",
    "always_valid_p",
    "WORSE",
    "NOT_WORSE",
    "WATCHING",
]

__version__ = "0.1.0"
