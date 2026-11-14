import csv
import json

import pytest

import rolloutwatch as rw


def write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content if isinstance(content, str) else json.dumps(content))
    return path


def run(log, **kwargs):
    monitor = rw.QualityMonitor(**kwargs)
    log.feed(monitor)
    return monitor


def test_request_log_gives_exactly_the_same_readings_as_daily_counts():
    log = rw.load_requests("examples/requests.csv", new="prompt-v2", old="prompt-v1")
    from_log = run(log, drop=0.03)
    from_counts = rw.QualityMonitor(drop=0.03)
    for row in csv.DictReader(open("examples/side_by_side.csv")):
        from_counts.update(int(row["new_passed"]), int(row["new_total"]), int(row["old_passed"]), int(row["old_total"]))
    assert [b.period for b in log.batches] == [r["date"] for r in csv.DictReader(open("examples/side_by_side.csv"))]
    assert from_log.history == from_counts.history


def test_jsonl_with_epoch_millis_nested_version_and_status(tmp_path):
    rows = [
        {"ts": 1793577600000 + i * 600_000, "metadata": {"variant": "treatment" if i % 2 else "control"},
         "status": "success" if i % 3 else "error"}
        for i in range(24)
    ]
    path = write(tmp_path, "log.jsonl", "\n".join(json.dumps(r) for r in rows))
    log = rw.load_requests(path, period="hour")
    assert log.columns_used == {"time": "ts", "version": "metadata.variant", "status": "status"}
    assert sum(b.new_total + b.old_total for b in log.batches) == 24
    assert all(len(b.period) == 16 for b in log.batches)  # "YYYY-MM-DD HH:00"


def test_new_and_old_found_automatically(tmp_path):
    path = write(tmp_path, "log.csv", "date,group,passed\n2026-11-02,new,1\n2026-11-02,old,0\n")
    log = rw.load_requests(path)
    assert log.batches[0] == rw.Batch("2026-11-02", 1, 1, 0, 1)


def test_ambiguous_versions_are_explained(tmp_path):
    path = write(tmp_path, "log.csv", "date,version,passed\n2026-11-02,v1,1\n2026-11-02,v2,0\n")
    with pytest.raises(ValueError, match="say which version is new.*v1, v2"):
        rw.load_requests(path)


def test_other_versions_are_left_out_and_reported(tmp_path):
    path = write(tmp_path, "log.csv", "date,version,passed\n2026-11-02,v2,1\n2026-11-02,v1,1\n2026-11-02,v3,0\n2026-11-02,v3,1\n")
    log = rw.load_requests(path, new="v2", old="v1")
    assert log.skipped == {"v3": 2}
    assert "Left out 2 requests from version 'v3'" in log.report()


def test_baseline_mode_needs_no_version_column(tmp_path):
    path = write(tmp_path, "log.csv", "timestamp,passed\n2026-11-02T09:00:00Z,yes\n2026-11-02T10:00:00Z,no\n2026-11-03 08:00:00,true\n")
    log = rw.load_requests(path, baseline=True)
    assert log.batches == [rw.Batch("2026-11-02", 1, 2, None, None), rw.Batch("2026-11-03", 1, 1, None, None)]
    monitor = run(log, baseline=0.8)
    assert monitor.history[-1].new_total == 3


def test_score_column_with_threshold(tmp_path):
    path = write(tmp_path, "log.csv", "date,arm,score\n2026-11-02,new,0.9\n2026-11-02,new,0.4\n2026-11-02,old,0.7\n")
    log = rw.load_requests(path, new="new", old="old", pass_threshold=0.6)
    assert log.batches[0] == rw.Batch("2026-11-02", 1, 2, 1, 1)
    assert log.columns_used["score"] == "score >= 0.6"


def test_days_before_both_versions_have_results_are_merged(tmp_path):
    path = write(tmp_path, "log.csv", "date,version,passed\n2026-11-01,old,1\n2026-11-02,old,1\n2026-11-02,new,0\n2026-11-03,new,1\n")
    log = rw.load_requests(path)
    assert log.batches == [rw.Batch("2026-11-02", 0, 1, 2, 2), rw.Batch("2026-11-03", 1, 1, 0, 0)]


@pytest.mark.parametrize(
    "content, kwargs, message",
    [
        ("version,passed\nnew,1\n", {}, "no time column"),
        ("date,passed\n2026-11-02,1\n", {}, "no version column"),
        ("date,version\n2026-11-02,new\n", {}, "no pass/fail"),
        ("date,version,passed\nyesterday,new,1\nyesterday,old,1\n", {}, "cannot read time"),
        ("date,version,passed\n2026-11-02,new,maybe\n2026-11-02,old,1\n", {}, "cannot read 'maybe'"),
        ("date,version,passed\n2026-11-02,new,1\n2026-11-02,old,1\n", {"new": "v9", "old": "old"}, "no requests from version 'v9'"),
        ("date,version,passed\n2026-11-02,new,1\n", {"period": "week"}, "period must be"),
        ("date,version,passed\n2026-11-02,new,1\n2026-11-03,new,1\n", {"new": "new", "old": "old"}, "no requests from version 'old'"),
    ],
)
def test_clear_errors(tmp_path, content, kwargs, message):
    with pytest.raises(ValueError, match=message):
        rw.load_requests(write(tmp_path, "log.csv", content), **kwargs)
