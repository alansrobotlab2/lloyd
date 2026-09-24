"""#761: scripts/maintenance/corpus_shape.py — cross-run shape of four corpora.

Every test builds its corpora under tmp_path; nothing reads the real vault or
the live `_pipeline/`.
"""
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "maintenance" / "corpus_shape.py"
NOW = "2026-09-20T03:00:00+00:00"
METRICS = ("n", "len_mean", "len_p95", "distinct_key_ratio", "duplicate_rate",
           "self_reference_rate")


def _load():
    spec = importlib.util.spec_from_file_location("corpus_shape", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cs = _load()


def _build(root: Path) -> tuple[Path, Path]:
    vault, pipeline = root / "vault", root / "pipeline"
    mem = vault / "memory"
    mem.mkdir(parents=True)
    for day, topic in (("2026-09-18", "backups"), ("2026-09-19", "voice"),
                       ("2026-09-20", "the guardian"), ("2026-08-01", "old")):
        (mem / f"{day}.md").write_text(
            f"---\ntype: note\n---\n## Morning on {topic}\n\nWe looked at {topic} for a while today.\n\n"
            f"## Evening on {topic}\n\nThe daily note for {topic} was written late.\n")
    lloyd = vault / "lloyd"
    lloyd.mkdir()
    (lloyd / "USER.md").write_text(
        "---\ntype: note\n---\n# User\n\n## operational\n"
        "- **Scope**: agent memory lives in the vault.\n"
        "- **Ceiling**: USER.md is capped at sixteen kilobytes,\n  enforced by nobody.\n")
    (lloyd / "MEMORY.md").write_text("# Memory\n\n- **Restarts** go through the round CLI.\n")
    for name in ("alpha", "beta"):
        d = vault / "skills" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\n---\n# {name}\n\nDo the {name} thing carefully.\n")
    traj = pipeline / "trajectories"
    traj.mkdir(parents=True)
    rows = [{"session_key": "s1", "tools": []}, {"session_key": "s2", "tools": []},
            {"session_key": "s2", "tools": []}]
    (traj / "2026-09-19.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (traj / "2026-08-01.jsonl").write_text(json.dumps({"session_key": "old"}) + "\n")
    return vault, pipeline


def _run(vault, pipeline, out, *extra):
    return cs.main(["--vault", str(vault), "--pipeline", str(pipeline),
                    "--out-dir", str(out), "--now", NOW, "--days", "7", *extra])


def test_prints_every_metric_for_each_of_the_four_corpora(tmp_path, capsys):
    vault, pipeline = _build(tmp_path)
    assert _run(vault, pipeline, tmp_path / "out") == 0
    out = capsys.readouterr().out
    for corpus in cs.CORPORA:
        line = next(l for l in out.splitlines() if l.strip().startswith(corpus + " ["))
        for metric in METRICS:
            assert f"{metric}=" in line, (corpus, metric, line)
    assert "dated 2026-09-14..2026-09-20" in out

    report = json.loads(next((tmp_path / "out").glob("corpus-shape-*.json")).read_text())
    c = report["corpora"]
    assert c["daily"]["n"] == 6                        # the August note is outside the window
    assert c["daily"]["self_reference_rate"] == 0.5    # each "Evening" names the daily note
    assert c["user_memory"]["n"] == 3
    assert c["user_memory"]["self_reference_rate"] == round(1 / 3, 4)
    assert c["skills"]["n"] == 2
    assert c["trajectories"]["n"] == 3
    assert c["trajectories"]["distinct_key_ratio"] == round(2 / 3, 4)
    assert c["trajectories"]["duplicate_rate"] == round(1 / 3, 4)


def test_a_run_never_rewrites_a_prior_file(tmp_path):
    vault, pipeline = _build(tmp_path)
    out = tmp_path / "out"
    _run(vault, pipeline, out)
    (first,) = list(out.glob("corpus-shape-*.json"))
    before = first.read_bytes()
    os.chmod(first, 0o444)          # an overwrite attempt would raise, not pass quietly
    _run(vault, pipeline, out)      # same pinned timestamp
    (second,) = set(out.glob("corpus-shape-*.json")) - {first}
    assert first.read_bytes() == before
    # The newer file is the one a third run diffs against.
    assert cs.previous_run(out)[0] == second


def test_diff_says_no_prior_run_then_one_verdict_line_per_corpus(tmp_path, capsys):
    vault, pipeline = _build(tmp_path)
    out = tmp_path / "out"
    _run(vault, pipeline, out)
    first = capsys.readouterr().out.splitlines()
    for corpus in cs.CORPORA:
        assert f"{corpus}: no prior run" in first

    # Grow one corpus past its threshold, then diff against the first file.
    extra = "".join(json.dumps({"session_key": f"n{i}"}) + "\n" for i in range(9))
    with open(pipeline / "trajectories" / "2026-09-19.jsonl", "a") as fh:
        fh.write(extra)
    assert _run(vault, pipeline, out) == 2
    second = capsys.readouterr().out.splitlines()
    assert "no prior run" not in "\n".join(second)
    for corpus in cs.CORPORA:
        verdicts = [l for l in second if l.startswith(corpus + ": ")
                    and ("within thresholds" in l or "MOVED" in l)]
        assert len(verdicts) == 1, (corpus, second)
    assert next(l for l in second if l.startswith("trajectories: MOVED")).count("n 3->12")
    assert any(l == "skills: within thresholds" for l in second)


def test_an_injected_repeated_sentence_is_named_with_its_file(tmp_path, capsys):
    clean_vault, pipeline = _build(tmp_path / "clean")
    dirty_root = tmp_path / "dirty"
    shutil.copytree(tmp_path / "clean", dirty_root)
    injected = "Remember to verify the watermark before trusting any count."
    target = dirty_root / "vault" / "memory" / "2026-09-19.md"
    target.write_text(target.read_text().replace(
        "for a while today.", f"for a while today. {injected}").replace(
        "was written late.", f"was written late. {injected}")
        + f"\n## Night\n\n{injected}\n")

    # Baseline both copies identically, then measure: only the dirty copy speaks.
    assert _run(clean_vault, pipeline, tmp_path / "out-clean") == 0
    clean_out = capsys.readouterr().out
    assert injected not in clean_out and "recurring" not in clean_out

    dirty_out_dir = tmp_path / "out-dirty"
    shutil.copytree(tmp_path / "out-clean", dirty_out_dir)
    code = _run(dirty_root / "vault", dirty_root / "pipeline", dirty_out_dir, "--quiet")
    dirty_out = capsys.readouterr().out
    assert code == 2
    line = next(l for l in dirty_out.splitlines() if injected in l)
    assert str(target) in line and "NEW recurring" in line

    # The untouched copy, run again in quiet mode against its own baseline, says nothing.
    assert _run(clean_vault, pipeline, tmp_path / "out-clean", "--quiet") == 0
    assert capsys.readouterr().out == ""


def test_the_first_run_is_a_baseline_even_with_a_recurrence(tmp_path, capsys):
    vault, pipeline = _build(tmp_path)
    note = vault / "memory" / "2026-09-20.md"
    rep = "The watchdog can no longer perform one of its own preconditions."
    note.write_text(note.read_text() + "".join(f"\n## Alert {i}\n\n{rep}\n" for i in range(3)))
    assert _run(vault, pipeline, tmp_path / "out", "--quiet") == 0
    assert capsys.readouterr().out == ""
    report = json.loads(next((tmp_path / "out").glob("*.json")).read_text())
    assert report["corpora"]["daily"]["recurring"][0]["sentence"] == rep


@pytest.mark.parametrize("old,new,hit", [(10, 16, True), (10, 14, False), (0, 0, False)])
def test_relative_threshold_on_n(old, new, hit):
    assert cs._moved("n", old, new)[0] is hit


def _added_files(patch: Path) -> dict:
    """{path: text} for each file a `git diff` patch creates."""
    files, cur = {}, None
    for line in patch.read_text().splitlines():
        if line.startswith("+++ b/"):
            cur = line[len("+++ b/"):]
            files[cur] = []
        elif cur and line.startswith("+"):
            files[cur].append(line[1:])
    return {k: "\n".join(v) + "\n" for k, v in files.items()}


def test_the_scheduled_task_runs_quiet_off_hours_and_names_its_skill():
    """The task lives in the vault, so it is held as a patch until this lands
    (the `vault-automod-skill-blast-radius.patch` precedent): applied earlier, it
    would schedule a script production does not have yet."""
    import yaml

    files = _added_files(ROOT / "scripts" / "maintenance" / "vault-corpus-shape-task.patch")
    task_path = next(p for p in files if p.startswith("autonomy/"))
    front = yaml.safe_load(files[task_path].split("---\n")[1])
    assert front["type"] == "autonomy" and front["status"] == "up_next"
    assert set(front["preferred_hours"]) <= {23, 0, 1, 2, 3, 4, 5}
    assert "scripts/maintenance/corpus_shape.py --quiet" in front["description"]
    assert "exit code 2" in front["description"]
    skill = files[f"skills/{front['skill_name']}/SKILL.md"]
    assert "corpus_shape.py --quiet" in skill and "exit code 2" in skill
