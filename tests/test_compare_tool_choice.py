"""The regression guard has to compare, and has to know which way is better.

`run_tool_choice_eval.py` writes a baseline per run and prints nothing
comparable. Round SM_20260908_165950 ran it after rewriting 66% of the
operating contract, spent four iterations guessing the file's shape, and
landed quoting `correct_rate 0.95` against no prior at all — while a 0.90 from
09-04 sat on disk. A number with no baseline is not a guard.

The two things worth pinning: a rate where lower is better must not read as an
improvement when it rises, and a run that errored measured nothing whatever
its rates say.
"""
from __future__ import annotations

import json

import pytest

from eval import compare_tool_choice as C


def _run(**overall):
    base = {"correct_rate": 0.9, "http_tool_first_rate": 0.86,
            "control_correct_rate": 1.0, "shelled_to_web_rate": 0.0,
            "no_tool_call_rate": 0.0, "errors": 0, "n_queries": 20}
    base.update(overall)
    n = base.pop("n_queries")
    return {"n_queries": n, "summary": {"overall": base}}


def test_an_unchanged_run_reports_no_regression():
    regressions, _ = C.compare(_run(), _run())
    assert regressions == []


def test_a_drop_in_correct_rate_is_a_regression():
    regressions, _ = C.compare(_run(correct_rate=0.70), _run(correct_rate=0.90))
    assert any("correct_rate" in r for r in regressions)


def test_a_rise_in_correct_rate_is_not():
    regressions, _ = C.compare(_run(correct_rate=0.95), _run(correct_rate=0.90))
    assert regressions == []


def test_a_rise_in_a_failure_rate_is_a_regression():
    """`shelled_to_web_rate` going up is the defect the eval exists to catch."""
    regressions, _ = C.compare(_run(shelled_to_web_rate=0.30),
                               _run(shelled_to_web_rate=0.00))
    assert any("shelled_to_web_rate" in r for r in regressions)


def test_a_fall_in_a_failure_rate_is_an_improvement():
    regressions, _ = C.compare(_run(no_tool_call_rate=0.0), _run(no_tool_call_rate=0.4))
    assert regressions == []


def test_one_query_flipping_is_within_tolerance():
    """0.05 is a single query on a 20-query set — noise, not signal."""
    regressions, _ = C.compare(_run(correct_rate=0.85), _run(correct_rate=0.90))
    assert regressions == []


def test_two_queries_flipping_is_not():
    regressions, _ = C.compare(_run(correct_rate=0.80), _run(correct_rate=0.90))
    assert regressions != []


def test_an_errored_run_is_a_regression_whatever_its_rates():
    regressions, _ = C.compare(_run(errors=3, correct_rate=1.0), _run())
    assert any("errored" in r for r in regressions)


def test_a_flat_summary_shape_is_still_read():
    """Older runs wrote the metrics at the top of `summary`, not under `overall`."""
    flat = {"summary": {"correct_rate": 0.5}}
    assert C.overall(flat) == {"correct_rate": 0.5}


def test_a_missing_metric_is_reported_not_silently_passed():
    _, lines = C.compare({"summary": {"overall": {"correct_rate": "n/a"}}},
                         _run())
    assert any("not comparable" in ln for ln in lines)


# ── the CLI ─────────────────────────────────────────────────────────────────

@pytest.fixture
def baseline_dir(tmp_path, monkeypatch):
    d = tmp_path / "tool-choice"
    d.mkdir(parents=True)
    monkeypatch.setattr(C, "BASELINE_DIR", d)
    return d


def _write(d, name, run, mtime):
    p = d / name
    p.write_text(json.dumps(run), encoding="utf-8")
    import os
    os.utime(p, (mtime, mtime))
    return p


def test_cli_compares_the_two_newest_runs(baseline_dir, capsys):
    _write(baseline_dir, "old.json", _run(correct_rate=0.90), 1000)
    _write(baseline_dir, "new.json", _run(correct_rate=0.60), 2000)
    assert C.main([]) == 1
    assert "REGRESSED" in capsys.readouterr().out


def test_cli_skips_loudly_with_only_one_run(baseline_dir, capsys):
    """Exit 2, not 0: 'nothing to compare' must not read as 'nothing wrong'."""
    _write(baseline_dir, "only.json", _run(), 1000)
    assert C.main([]) == 2
    assert "not a pass" in capsys.readouterr().out


def test_cli_label_selects_the_run_under_test(baseline_dir, capsys):
    _write(baseline_dir, "prior-1.json", _run(correct_rate=0.90), 1000)
    _write(baseline_dir, "mine-2.json", _run(correct_rate=0.95), 2000)
    assert C.main(["--label", "mine"]) == 0
    out = capsys.readouterr().out
    assert "current : mine-2.json" in out and "baseline: prior-1.json" in out


def test_cli_says_so_when_the_label_has_no_prior(baseline_dir, capsys):
    _write(baseline_dir, "mine-1.json", _run(), 1000)
    _write(baseline_dir, "mine-2.json", _run(), 2000)
    assert C.main(["--label", "mine"]) == 2
    assert "no prior" in capsys.readouterr().out


def test_cli_flags_a_differing_query_count(baseline_dir, capsys):
    a = _run(); a["n_queries"] = 20
    b = _run(); b["n_queries"] = 8
    _write(baseline_dir, "old.json", b, 1000)
    _write(baseline_dir, "new.json", a, 2000)
    C.main([])
    assert "not the same test" in capsys.readouterr().out
