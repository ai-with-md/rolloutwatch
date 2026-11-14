from rolloutwatch.cli import main


def test_check_side_by_side(capsys):
    assert main(["check", "examples/side_by_side.csv"]) == 0
    out = capsys.readouterr().out
    assert "reliably worse" in out and "2026-11-08" in out


def test_check_against_baseline(capsys):
    assert main(["check", "examples/after_launch.csv", "--baseline", "0.83"]) == 0
    assert "not worse" in capsys.readouterr().out


def test_check_needs_baseline_without_old_columns(capsys):
    assert main(["check", "examples/after_launch.csv"]) == 2
    assert "--baseline" in capsys.readouterr().err


def test_check_rejects_baseline_with_old_columns(capsys):
    assert main(["check", "examples/side_by_side.csv", "--baseline", "0.8"]) == 2
    assert "leave out --baseline" in capsys.readouterr().err


def test_check_bad_count(tmp_path, capsys):
    path = tmp_path / "bad.csv"
    path.write_text("new_passed,new_total\nmany,100\n")
    assert main(["check", str(path), "--baseline", "0.8"]) == 2
    assert "many" in capsys.readouterr().err


def test_plan_and_peeking(capsys):
    assert main(["plan", "--baseline", "0.8", "--change", "-0.06", "--per-day", "300", "--simulations", "100"]) == 0
    assert main(["peeking", "--checks", "10"]) == 0
    out = capsys.readouterr().out
    assert "worse" in out and "false alarm" in out


def test_cli_reads_request_logs_and_saves_counts(tmp_path, capsys):
    counts = tmp_path / "counts.csv"
    args = ["check", "examples/requests.csv", "--new", "prompt-v2", "--old", "prompt-v1", "--save-counts", str(counts)]
    assert main(args) == 0
    from_log = capsys.readouterr().out
    assert main(["check", str(counts)]) == 0
    from_counts = capsys.readouterr().out
    assert "first clear at day 2026-11-08" in from_log
    assert from_log.splitlines()[-1].split(" at ")[0] == from_counts.splitlines()[-1].split(" at ")[0]


def test_cli_verbose_report(capsys):
    assert main(["check", "examples/requests.csv", "--new", "prompt-v2", "--old", "prompt-v1", "-v"]) == 0
    assert "Read 5864 requests into 21 periods" in capsys.readouterr().err
