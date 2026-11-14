"""Command line: ``rolloutwatch check``, ``rolloutwatch plan`` and ``rolloutwatch peeking``."""

from __future__ import annotations

import argparse
import csv
import sys
from typing import List, Optional

from . import WATCHING, QualityMonitor, ScoreMonitor, load_requests, peeking_false_alarms, plan_rollout


def _int(row: dict, key: str, path: str, line: int) -> int:
    try:
        return int(float(row[key]))
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{path}, line {line}: cannot read {key} {row.get(key)!r} as a count") from None


def _file_kind(path: str) -> str:
    """'counts', 'score summaries' or 'requests'."""
    if not path.lower().endswith(".csv"):
        return "requests"
    with open(path, newline="", encoding="utf-8-sig") as handle:
        header = {c.strip().lower() for c in next(csv.reader(handle), [])}
    if {"new_passed", "new_total"} <= header:
        return "counts"
    if {"new_count", "new_mean", "new_sd"} <= header:
        return "score summaries"
    return "requests"


def _monitor(args: argparse.Namespace, scores: bool):
    if scores:
        low, high = args.score_range
        drop = args.drop if args.drop is not None else 0.05 * (high - low)
        args.drop = drop
        return ScoreMonitor(drop=drop, score_range=(low, high), alpha=args.alpha, baseline=args.baseline,
                            min_samples=args.min_samples)
    if args.drop is None:
        args.drop = 0.03
    return QualityMonitor(drop=args.drop, alpha=args.alpha, baseline=args.baseline, min_samples=args.min_samples)


def _check_requests(args: argparse.Namespace) -> str:
    log = load_requests(
        args.file,
        new=args.new,
        old=args.old,
        baseline=args.baseline is not None,
        version=args.version_column,
        time=args.time_column,
        outcome=args.outcome_column,
        score=args.score_column,
        pass_threshold=args.pass_threshold,
        mean_scores=args.mean_score,
        period=args.period,
    )
    if args.verbose:
        print(log.report(), file=sys.stderr)
    if args.save_counts:
        log.to_csv(args.save_counts)
    monitor = _monitor(args, args.mean_score)
    log.feed(monitor)
    return _format(args, monitor, [b.period for b in log.batches], args.period)


def _format(args: argparse.Namespace, monitor, labels: List[str], label_name: Optional[str]) -> str:
    scores = isinstance(monitor, ScoreMonitor)
    lines = []
    for name, reading in zip(labels, monitor.history):
        if scores:
            lines.append(
                f"  {name:>16}  new {reading.new_mean:6.3f}  vs {reading.old_mean:6.3f}  "
                f"diff {reading.difference:+6.3f}  ({reading.low:+.3f} to {reading.high:+.3f})  {reading.status}"
            )
        else:
            lines.append(
                f"  {name:>16}  new {reading.new_rate:6.1%}  vs {reading.old_rate:6.1%}  "
                f"diff {reading.difference:+6.1%}  ({reading.low:+.1%} to {reading.high:+.1%})  {reading.status}"
            )
    what = f"a drop in the average score of {args.drop:g}" if scores else f"a drop of {args.drop:.1%}"
    base = "" if args.baseline is None else format(args.baseline, "g" if scores else ".1%")
    header = (
        f"Watching for {what} or more "
        f"({'against a baseline of ' + base if args.baseline is not None else 'side by side'}), "
        f"chance of any wrong call {args.alpha:g}:"
    )
    last = monitor.report().splitlines()[-1]
    if monitor.status != WATCHING and label_name:
        first = next(c for c in monitor.history if c.status != WATCHING)
        last = last.replace(f"check {first.check}", f"{label_name} {labels[first.check - 1]}")
    return "\n".join([header] + lines + [last])


def _check_score_summaries(args: argparse.Namespace) -> str:
    with open(args.file, newline="", encoding="utf-8-sig") as handle:
        rows = [{k.strip().lower(): v for k, v in r.items() if k} for r in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"{args.file} has no rows")
    side_by_side = {"old_count", "old_mean", "old_sd"} <= set(rows[0])
    if side_by_side == (args.baseline is not None):
        raise ValueError(
            "pass --baseline only when the file has no old_count, old_mean and old_sd columns"
        )
    label = next((c for c in ("period", "date", "day", "batch") if c in rows[0]), None)
    monitor = _monitor(args, True)
    labels = []
    for line, row in enumerate(rows, start=2):
        try:
            values = [float(row[k]) for k in (("new_count", "new_mean", "new_sd", "old_count", "old_mean", "old_sd") if side_by_side else ("new_count", "new_mean", "new_sd"))]
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{args.file}, line {line}: cannot read the counts, averages and standard deviations") from None
        if side_by_side:
            reading = monitor.update_summary(int(values[0]), values[1], values[2], int(values[3]), values[4], values[5])
        else:
            reading = monitor.update_summary(int(values[0]), values[1], values[2])
        labels.append(row[label] if label else str(reading.check))
    return _format(args, monitor, labels, label)


def _cmd_check(args: argparse.Namespace) -> str:
    kind = _file_kind(args.file)
    if kind == "requests":
        return _check_requests(args)
    if kind == "score summaries":
        return _check_score_summaries(args)
    with open(args.file, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{args.file} has no rows")
    columns = {c.strip().lower() for c in rows[0] if c}
    side_by_side = {"old_passed", "old_total"} <= columns
    if not {"new_passed", "new_total"} <= columns:
        raise ValueError(f"{args.file} needs new_passed and new_total columns (and old_passed, old_total, or --baseline)")
    if side_by_side and args.baseline is not None:
        raise ValueError("the file has old_passed and old_total columns, so leave out --baseline")
    if not side_by_side and args.baseline is None:
        raise ValueError("the file has no old_passed and old_total columns: pass --baseline with the old pass rate")
    rows = [{k.strip().lower(): v for k, v in r.items() if k} for r in rows]
    label = next((c for c in ("period", "date", "day", "batch") if c in columns), None)
    monitor = _monitor(args, False)
    labels = []
    for line, row in enumerate(rows, start=2):
        new = (_int(row, "new_passed", args.file, line), _int(row, "new_total", args.file, line))
        old = (_int(row, "old_passed", args.file, line), _int(row, "old_total", args.file, line)) if side_by_side else (None, None)
        reading = monitor.update(new[0], new[1], *old)
        labels.append(row[label] if label else str(reading.check))
    return _format(args, monitor, labels, label)


def _cmd_plan(args: argparse.Namespace) -> str:
    if args.drop is None:
        args.drop = 0.03
    return plan_rollout(
        args.baseline, args.change, args.per_day, args.new_share, args.drop, args.days, args.alpha,
        fixed_baseline=args.fixed_baseline, simulations=args.simulations,
    ).summary()


def _cmd_peeking(args: argparse.Namespace) -> str:
    rate = peeking_false_alarms(args.checks, args.per_check, args.rate, args.alpha)
    return (
        f"Checking an ordinary test {args.checks} times and stopping at the first alarm: "
        f"{rate:.0%} of rollouts with no real change get a false alarm (each single test promises {args.alpha:.0%})."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rolloutwatch",
        description="Did my prompt or model change make things worse? Check as often as you like.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--drop", type=float,
            help="smallest drop worth catching (default 0.03 for pass rates, 5%% of the score range for scores)",
        )
        p.add_argument("--alpha", type=float, default=0.05, help="chance of any wrong call (default 0.05)")

    p = sub.add_parser("check", help="read counts or a request log and say worse, not worse, or keep watching")
    p.add_argument(
        "file",
        help="CSV of counts per day (new_passed, new_total, old_passed, old_total), "
        "or a CSV, JSON or JSON Lines log with one row per graded request",
    )
    p.add_argument("--new", help="request logs: version label of the new version")
    p.add_argument("--old", help="request logs: version label of the old version")
    p.add_argument("--period", choices=["day", "hour"], default="day", help="request logs: count per day or per hour")
    p.add_argument("--version-column", help="request logs: column with the version (found automatically if left out)")
    p.add_argument("--time-column", help="request logs: column with the time (found automatically if left out)")
    p.add_argument("--outcome-column", help="request logs: column with pass/fail (found automatically if left out)")
    p.add_argument("--score-column", help="request logs: numeric score column, used with --pass-threshold")
    p.add_argument("--pass-threshold", type=float, default=0.5, help="request logs: score needed to pass (default 0.5)")
    p.add_argument("--save-counts", help="request logs: also save the counts per period to this CSV")
    p.add_argument("--mean-score", action="store_true", help="compare average scores instead of pass rates")
    p.add_argument("--score-range", type=float, nargs=2, metavar=("LOW", "HIGH"), default=(0.0, 1.0),
                   help="lowest and highest possible score, with --mean-score (default 0 1)")
    p.add_argument("-v", "--verbose", action="store_true", help="say which columns were used")
    p.add_argument("--baseline", type=float, help="old pass rate, when there is no old version running side by side")
    p.add_argument("--min-samples", type=int, default=50, help="results needed per version before any call")
    common(p)
    p.set_defaults(run=_cmd_check)

    p = sub.add_parser("plan", help="how many days until an answer, by simulation")
    p.add_argument("--baseline", type=float, required=True, help="old pass rate")
    p.add_argument("--change", type=float, required=True, help="real change to simulate, e.g. -0.05, or 0")
    p.add_argument("--per-day", type=int, required=True, help="graded requests per day, both versions together")
    p.add_argument("--new-share", type=float, default=0.5, help="share of traffic on the new version (default 0.5)")
    p.add_argument("--fixed-baseline", action="store_true", help="all traffic on the new version, baseline known")
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--simulations", type=int, default=500)
    common(p)
    p.set_defaults(run=_cmd_plan)

    p = sub.add_parser("peeking", help="how often daily checks with an ordinary test raise false alarms")
    p.add_argument("--checks", type=int, required=True)
    p.add_argument("--per-check", type=int, default=200, help="graded results per version per check")
    p.add_argument("--rate", type=float, default=0.8)
    p.add_argument("--alpha", type=float, default=0.05)
    p.set_defaults(run=_cmd_peeking)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(args.run(args))
    except (ValueError, OSError) as exc:
        print(f"rolloutwatch: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
