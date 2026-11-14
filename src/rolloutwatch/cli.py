"""Command line: ``rolloutwatch check``, ``rolloutwatch plan`` and ``rolloutwatch peeking``."""

from __future__ import annotations

import argparse
import csv
import sys
from typing import List, Optional

from . import WATCHING, QualityMonitor, peeking_false_alarms, plan_rollout


def _int(row: dict, key: str, path: str, line: int) -> int:
    try:
        return int(float(row[key]))
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{path}, line {line}: cannot read {key} {row.get(key)!r} as a count") from None


def _cmd_check(args: argparse.Namespace) -> str:
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
    monitor = QualityMonitor(drop=args.drop, alpha=args.alpha, baseline=args.baseline, min_samples=args.min_samples)
    lines = []
    for line, row in enumerate(rows, start=2):
        new = (_int(row, "new_passed", args.file, line), _int(row, "new_total", args.file, line))
        old = (_int(row, "old_passed", args.file, line), _int(row, "old_total", args.file, line)) if side_by_side else (None, None)
        reading = monitor.update(new[0], new[1], *old)
        name = row[label] if label else str(reading.check)
        lines.append(
            f"  {name:>12}  new {reading.new_rate:6.1%}  vs {reading.old_rate:6.1%}  "
            f"diff {reading.difference:+6.1%}  ({reading.low:+.1%} to {reading.high:+.1%})  {reading.status}"
        )
    header = (
        f"Watching for a drop of {args.drop:.1%} or more "
        f"({'against a baseline of ' + format(args.baseline, '.1%') if args.baseline is not None else 'side by side'}), "
        f"chance of any wrong call {args.alpha:g}:"
    )
    last = monitor.report().splitlines()[-1]
    if monitor.status != WATCHING:
        first = next(c for c in monitor.history if c.status != WATCHING)
        if label:
            last = last.replace(f"check {first.check}", f"{label} {rows[first.check - 1][label]}")
    return "\n".join([header] + lines + [last])


def _cmd_plan(args: argparse.Namespace) -> str:
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
        p.add_argument("--drop", type=float, default=0.03, help="smallest drop worth catching (default 0.03)")
        p.add_argument("--alpha", type=float, default=0.05, help="chance of any wrong call (default 0.05)")

    p = sub.add_parser("check", help="read daily or batch counts and say worse, not worse, or keep watching")
    p.add_argument("file", help="CSV with new_passed, new_total and old_passed, old_total per row (one row per day or batch)")
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
