"""Turn a raw log of graded requests into daily (or hourly) counts for the monitor.

The log has one row per graded request: when it happened, which version
served it, and whether it passed. CSV, JSON and JSON Lines files work, and
nested JSON fields are named with dots (``metadata.version``).

Columns are found by common names; pass them yourself when the guess is
wrong. With a fixed baseline, the version column is optional: every row is
taken as the new version.
"""

from __future__ import annotations

import csv
import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

VERSION_NAMES = ["version", "variant", "arm", "group", "bucket", "promptversion", "modelversion", "treatment", "cohort"]
TIME_NAMES = ["timestamp", "time", "date", "datetime", "createdat", "starttime", "ts", "day"]
OUTCOME_NAMES = ["passed", "pass", "success", "succeeded", "correct", "ok", "issuccess", "outcome", "verdict", "judgeverdict", "result", "evalresult"]
SCORE_NAMES = ["score", "rating", "grade"]
STATUS_NAMES = ["status"]
ERROR_NAMES = ["error"]

_TRUE = {"1", "true", "t", "yes", "y", "pass", "passed", "success", "succeeded", "ok", "correct", "good", "thumbsup", "up"}
_FALSE = {"0", "false", "f", "no", "n", "fail", "failed", "failure", "error", "incorrect", "wrong", "bad", "thumbsdown", "down"}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _flatten(record: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in record.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, full))
        else:
            out[full] = value
    return out


def _read(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return [dict(r) for r in csv.DictReader(text.splitlines())]
    if suffix in {".jsonl", ".ndjson"}:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            data = next((data[k] for k in ("data", "requests", "results", "items", "rows") if isinstance(data.get(k), list)), data)
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a list of records, or an object with a 'data' list")
        records = data
    else:
        raise ValueError(f"{path}: use a .csv, .json or .jsonl file")
    if not all(isinstance(r, dict) for r in records):
        raise ValueError(f"{path}: every record must be an object")
    return [_flatten(r) for r in records]


def _find(columns: List[str], names: List[str]) -> Optional[str]:
    for wanted in names:
        for col in columns:
            if _norm(col) == wanted:
                return col
    for wanted in names:
        for col in columns:
            if "." in col and _norm(col.rsplit(".", 1)[-1]) == wanted:
                return col
    return None


def _parse_time(value: Any, where: str) -> datetime:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = value / 1000 if value > 1e11 else value  # epoch milliseconds or seconds
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    text = str(value).strip()
    if re.fullmatch(r"\d{10}(\d{3})?(\.\d+)?", text):
        return _parse_time(float(text), where)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text.replace(" ", "T", 1) if re.match(r"\d{4}-\d{2}-\d{2} \d", text) else text)
    except ValueError:
        raise ValueError(f"{where}: cannot read time {value!r}; use ISO dates like 2026-11-02 or 2026-11-02T14:05:00Z") from None


def _outcome(row: Dict[str, Any], outcome_col, score_col, threshold, status_col, error_col, where: str) -> bool:
    if outcome_col is not None:
        raw = row.get(outcome_col, "")
        if isinstance(raw, bool):
            return raw
        cell = _norm(raw)
        if cell in _TRUE:
            return True
        if cell in _FALSE:
            return False
        try:
            number = float(raw)
            if number in (0.0, 1.0):
                return number == 1.0
        except (TypeError, ValueError):
            pass
        raise ValueError(f"{where}: cannot read {row.get(outcome_col)!r} as pass or fail")
    if score_col is not None:
        try:
            return float(row.get(score_col)) >= threshold
        except (TypeError, ValueError):
            raise ValueError(f"{where}: cannot read score {row.get(score_col)!r} as a number") from None
    if status_col is not None:
        cell = _norm(row.get(status_col, ""))
        if cell in _TRUE or cell in {"completed", "done"}:
            return True
        if cell in _FALSE:
            return False
        raise ValueError(f"{where}: cannot read status {row.get(status_col)!r} as pass or fail")
    value = row.get(error_col)
    return value is None or _norm(value) in {"", "none", "null", "false"}


@dataclass(frozen=True)
class Batch:
    period: str
    new_passed: int
    new_total: int
    old_passed: Optional[int]
    old_total: Optional[int]


@dataclass
class RequestLog:
    """Requests counted per period, ready to feed to a QualityMonitor."""

    batches: List[Batch]
    rows: int
    skipped: Dict[str, int] = field(default_factory=dict)
    columns_used: Dict[str, str] = field(default_factory=dict)

    def report(self) -> str:
        used = ", ".join(f"{k} from '{v}'" for k, v in self.columns_used.items())
        lines = [f"Read {self.rows} requests into {len(self.batches)} periods ({used})."]
        for label, count in self.skipped.items():
            lines.append(f"Left out {count} requests from version '{label}'.")
        return "\n".join(lines)

    def feed(self, monitor) -> None:
        """Run every period through the monitor, in time order."""
        for b in self.batches:
            if b.old_total is None:
                monitor.update(b.new_passed, b.new_total)
            else:
                monitor.update(b.new_passed, b.new_total, b.old_passed, b.old_total)

    def to_csv(self, path) -> None:
        """Save the counts, in the format ``rolloutwatch check`` reads."""
        with open(path, "w", newline="") as handle:
            fields = ["period", "new_passed", "new_total"]
            if self.batches and self.batches[0].old_total is not None:
                fields += ["old_passed", "old_total"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for b in self.batches:
                row = {"period": b.period, "new_passed": b.new_passed, "new_total": b.new_total}
                if "old_total" in fields:
                    row.update(old_passed=b.old_passed, old_total=b.old_total)
                writer.writerow(row)


def load_requests(
    path,
    *,
    new: Optional[str] = None,
    old: Optional[str] = None,
    baseline: bool = False,
    version: Optional[str] = None,
    time: Optional[str] = None,
    outcome: Optional[str] = None,
    score: Optional[str] = None,
    pass_threshold: float = 0.5,
    period: str = "day",
) -> RequestLog:
    """Count graded requests per day or hour, for each version.

    Args:
        new, old: values in the version column for the new and the old
            version, for example ``new="prompt-v2", old="prompt-v1"``. If the
            column holds exactly the values ``new`` and ``old`` (or
            ``treatment`` and ``control``) they are found automatically.
        baseline: True when comparing against a fixed baseline. Only the new
            version's requests are counted; the version column is optional.
        version, time, outcome: column names, if the automatic guess is wrong.
        score, pass_threshold: use a numeric score column instead of pass or
            fail; a request passes when its score is at least the threshold.
        period: "day" or "hour".

    Requests are grouped by the date (or hour) in their time column, as
    written in the file, so make sure all times use the same time zone.
    Periods before both versions have results are merged into the first
    period that has both.
    """
    if period not in {"day", "hour"}:
        raise ValueError("period must be 'day' or 'hour'")
    path = Path(path)
    records = _read(path)
    if not records:
        raise ValueError(f"{path} has no rows")
    columns = list(OrderedDict.fromkeys(k for r in records for k in r))

    def pick(explicit, names, label, required=True):
        if explicit is not None:
            if explicit not in columns:
                raise ValueError(f"{path} has no {label} column named {explicit!r}")
            return explicit
        found = _find(columns, names)
        if found is None and required:
            raise ValueError(f"{path}: found no {label} column; name it with the {label} option. Columns are: {', '.join(columns[:15])}")
        return found

    time_col = pick(time, TIME_NAMES, "time")
    version_col = pick(version, VERSION_NAMES, "version", required=not baseline)
    outcome_col = pick(outcome, OUTCOME_NAMES, "outcome", required=False)
    score_col = pick(score, SCORE_NAMES, "score", required=False) if outcome_col is None else None
    status_col = _find(columns, STATUS_NAMES) if outcome_col is None and score_col is None else None
    error_col = _find(columns, ERROR_NAMES) if outcome_col is None and score_col is None and status_col is None else None
    if not any([outcome_col, score_col, status_col, error_col]):
        raise ValueError(f"{path}: found no pass/fail, score, status or error column")

    if version_col is not None and not baseline:
        values = list(OrderedDict.fromkeys(str(r.get(version_col)) for r in records))
        if new is None or old is None:
            pairs = [("new", "old"), ("treatment", "control"), ("b", "a")]
            lowered = {v.lower(): v for v in values}
            match = next(((lowered[n], lowered[o]) for n, o in pairs if n in lowered and o in lowered), None)
            if match is None:
                raise ValueError(
                    f"{path}: say which version is new and which is old (new=..., old=...). "
                    f"Versions in the file: {', '.join(values[:10])}"
                )
            new, old = match
        for label in (new, old):
            if label not in values:
                raise ValueError(f"{path}: no requests from version {label!r}. Versions in the file: {', '.join(values[:10])}")

    counts: Dict[str, List[int]] = {}
    skipped: Dict[str, int] = {}
    for i, row in enumerate(records, start=1):
        where = f"{path}, row {i}"
        arm = "new"
        if version_col is not None:
            label = str(row.get(version_col))
            if baseline:
                if new is not None and label != new:
                    skipped[label] = skipped.get(label, 0) + 1
                    continue
            elif label == new:
                arm = "new"
            elif label == old:
                arm = "old"
            else:
                skipped[label] = skipped.get(label, 0) + 1
                continue
        stamp = _parse_time(row.get(time_col), where)
        key = stamp.strftime("%Y-%m-%d") if period == "day" else stamp.strftime("%Y-%m-%d %H:00")
        passed = _outcome(row, outcome_col, score_col, pass_threshold, status_col, error_col, where)
        slot = counts.setdefault(key, [0, 0, 0, 0])
        offset = 0 if arm == "new" else 2
        slot[offset] += passed
        slot[offset + 1] += 1

    batches: List[Batch] = []
    carry = [0, 0, 0, 0]
    for key in sorted(counts):
        carry = [a + b for a, b in zip(carry, counts[key])]
        have_new = carry[1] > 0
        have_old = baseline or carry[3] > 0
        if not batches and not (have_new and have_old):
            continue  # merge into the first period that has both versions
        batches.append(
            Batch(key, carry[0], carry[1], None if baseline else carry[2], None if baseline else carry[3])
        )
        carry = [0, 0, 0, 0]
    if not batches:
        raise ValueError(f"{path}: no period has requests from both versions")

    used = {"time": time_col}
    if version_col:
        used["version"] = version_col
    for label, col in (("outcome", outcome_col), ("score", score_col), ("status", status_col), ("error", error_col)):
        if col:
            used[label] = col if label != "score" else f"{col} >= {pass_threshold:g}"
    return RequestLog(batches, len(records), skipped, used)
