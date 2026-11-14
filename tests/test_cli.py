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
