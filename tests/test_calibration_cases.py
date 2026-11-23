"""The known-bad calibration set, the label lock, and the two-rate trend.

`scripts/automod/calibration_cases.py`. Nothing here runs a grader: every
grading path takes a stub, and every git operation is on a repo the test made —
except the last node, which re-derives each shipped case's pinned sha from this
checkout's own history WITHOUT writing an object, and skips a case whose
commits this clone does not hold.
"""
from __future__ import annotations

import collections
import json
import subprocess
from pathlib import Path

import pytest

from scripts.automod import calibration_cases as CC
from scripts.automod import review_tools as RT

ROOT = Path(__file__).resolve().parents[1]
NEG = RT.FIXTURE_DIR / CC.NEGATIVE_SET


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return r.stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    """parent: a.py + doc.md. landed: a.py changed, doc.md changed, tests/test_a.py added."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@e.com")
    _git(r, "config", "user.name", "t")
    (r / "a.py").write_text("def f():\n    return 1\n")
    (r / "doc.md").write_text("f returns 1\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "base")
    parent = _git(r, "rev-parse", "HEAD")
    (r / "a.py").write_text("def f():\n    return 2\n")
    (r / "doc.md").write_text("f returns 2\n")
    (r / "tests").mkdir()
    (r / "tests" / "test_a.py").write_text("def test_f():\n    assert True\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "landed")
    return r, parent, _git(r, "rev-parse", "HEAD")


def _case(repo, parent, landed, kind, paths, **kw):
    return CC.constructed_case(repo=repo, name=kw.pop("name", f"x-{kind}"), item_id=1,
                               round_id="SM_x", parent=parent, source_commit=landed, kind=kind,
                               paths=paths, violates=[1], clause_text={1: "f returns 2 — tests"},
                               why="because.", **kw)


# ── building a defect ────────────────────────────────────────────────────

def test_a_restored_path_is_the_parents_and_an_added_one_is_gone(repo):
    r, parent, landed = repo
    sha = CC.build_commit(r, source=landed, parent=parent,
                          transform={"kind": "strip_tests", "paths": ["tests/test_a.py"]})
    assert _git(r, "diff", "--name-only", parent, sha).split() == ["a.py", "doc.md"]
    sha = CC.build_commit(r, source=landed, parent=parent,
                          transform={"kind": "revert_source", "paths": ["a.py"]})
    assert _git(r, "show", f"{sha}:a.py") == "def f():\n    return 1"
    assert _git(r, "show", f"{sha}:doc.md") == "f returns 2"
    assert _git(r, "rev-parse", f"{sha}^") == parent, "the defect is one commit on the parent"


def test_the_sha_is_a_function_of_its_inputs_and_building_touches_nothing(repo):
    r, parent, landed = repo
    t = {"kind": "drop_doc", "paths": ["doc.md"]}
    before = (_git(r, "rev-parse", "HEAD"), _git(r, "status", "--porcelain"),
              _git(r, "for-each-ref"))
    a = CC.build_commit(r, source=landed, parent=parent, transform=t)
    b = CC.build_commit(r, source=landed, parent=parent, transform=dict(t))
    assert a == b
    assert before == (_git(r, "rev-parse", "HEAD"), _git(r, "status", "--porcelain"),
                      _git(r, "for-each-ref")), "HEAD, the worktree or a ref moved"


def test_a_dry_build_names_the_same_sha_and_leaves_no_object(repo):
    r, parent, landed = repo
    t = {"kind": "revert_source", "paths": ["a.py"]}
    dry = CC.build_commit(r, source=landed, parent=parent, transform=t, write=False)
    assert not CC.has_object(r, dry), "write=False wrote the commit into the repo"
    assert CC.build_commit(r, source=landed, parent=parent, transform=t) == dry
    assert CC.has_object(r, dry)


def test_a_transform_that_removes_nothing_is_refused(repo):
    r, parent, landed = repo
    (r / "same.txt").write_text("x")
    with pytest.raises(ValueError, match="changes nothing"):
        CC.build_commit(r, source=parent, parent=parent,
                        transform={"kind": "drop_doc", "paths": ["doc.md"]})
    with pytest.raises(ValueError, match="kind must be"):
        CC.build_commit(r, source=landed, parent=parent, transform={"kind": "x", "paths": ["a.py"]})
    with pytest.raises(ValueError, match="at least one path"):
        CC.build_commit(r, source=landed, parent=parent, transform={"kind": "drop_doc", "paths": []})


def test_a_case_whose_rebuilt_commit_is_not_the_pinned_one_is_refused(repo):
    r, parent, landed = repo
    case = _case(r, parent, landed, "revert_source", ["a.py"])
    assert CC.materialize(r, case) == case["commit"]
    moved = {**case, "transform": {"kind": "revert_source", "paths": ["doc.md"]}}
    with pytest.raises(RuntimeError, match="not the one that was labelled"):
        CC.materialize(r, moved)
    with pytest.raises(RuntimeError, match="is not in"):
        CC.materialize(r, {"name": "gone", "commit": "0" * 40})


def test_a_proof_is_checked_when_the_case_is_built_and_again_on_verify(repo):
    r, parent, landed = repo
    case = _case(r, parent, landed, "drop_doc", ["doc.md"],
                 proof={"path": "doc.md", "text": "returns 2", "present": False})
    assert CC.verify_case(r, case) == []
    assert case["changed_paths"] == ["a.py", "tests/test_a.py"]
    with pytest.raises(ValueError, match="proof failed"):
        _case(r, parent, landed, "drop_doc", ["doc.md"],
              proof={"path": "doc.md", "text": "returns 2", "present": True})
    lying = {**case, "proof": {"path": "doc.md", "text": "returns 1", "present": False}}
    assert any("proof failed" in p for p in CC.verify_case(r, lying))
    left = {**case, "transform": {"kind": "strip_tests", "paths": ["doc.md"]}}
    assert any("strip_tests left test paths" in p or "rebuilt as" in p
               for p in CC.verify_case(r, left))


def test_a_refused_head_needs_a_reader_who_is_not_the_grader(repo):
    r, parent, landed = repo
    kw = dict(repo=r, name="h", item_id=1, round_id="SM_x", parent=parent, refused_head=landed,
              fixed_head=landed, violates=[], clause_text={}, refusal="clause 1 unmet: …",
              evidence="the fix diff adds the missing call", read_on="2026-11-22")
    case = CC.refused_head_case(read_by="a named reader", **kw)
    assert case["label_source"] == "refused_then_fixed" and case["expect"]["kind"] == "retry"
    assert "SELECTED" in case["note"]
    with pytest.raises(ValueError, match="not an independent reader"):
        CC.refused_head_case(read_by="grader", **kw)
    with pytest.raises(ValueError, match="needs the refusal"):
        CC.refused_head_case(read_by="a named reader", **{**kw, "evidence": " "})


# ── the lock ─────────────────────────────────────────────────────────────

def _root(tmp_path, repo):
    r, parent, landed = repo
    root = tmp_path / "cal"
    CC.write_case(_case(r, parent, landed, "revert_source", ["a.py"], name="bad"),
                  root / "negative")
    good = {"name": "good", "item_id": 1, "round_id": "SM_x", "parent": parent, "commit": landed,
            "changed_paths": ["a.py"], "strip_tests": False, "provisional": False, "note": "",
            "expect": {"kind": "pass", "unmet_clauses": [], "honesty_finding": False}}
    CC.write_case(good, root / "independent")
    CC.write_lock(root)
    return root


def _edit(root: Path, rel: str, **change) -> None:
    p = root / rel
    case = json.loads(p.read_text())
    for k, v in change.items():
        if k == "label":
            case["expect"]["kind"] = v
        else:
            case[k] = v
    p.write_text(json.dumps(case, indent=2) + "\n")


def test_the_shipped_labels_are_the_locked_ones():
    """THE FREEZE. A red here means a calibration label, or the diff under one, moved."""
    problems = CC.check_lock()
    assert not problems, "\n".join(problems) + "\n\n" + CC.RELABEL_RULE


def test_the_lock_covers_both_sets_and_only_them():
    locked = CC.read_lock(CC.lock_path())
    sets = collections.Counter(k.split("/")[0] for k in locked)
    assert set(sets) == set(CC.LOCKED_SETS)
    assert sets["independent"] == 12, "the independent set is not the twelve cases it was"
    assert {v[0] for k, v in locked.items() if k.startswith("independent/")} == {"pass"}
    assert {v[0] for k, v in locked.items() if k.startswith("negative/")} == {"retry"}


def test_a_changed_label_fails_the_check_and_cannot_be_relocked_bare(tmp_path, repo):
    root = _root(tmp_path, repo)
    assert CC.check_lock(root) == []
    _edit(root, "negative/bad.json", label="pass")
    assert CC.check_lock(root) == ["negative/bad: label retry -> pass"]
    with pytest.raises(ValueError, match="no `relabel` record") as exc:
        CC.write_lock(root)
    assert "never a valid reason" in str(exc.value)
    assert CC.read_lock(CC.lock_path(root))["negative/bad"][0] == "retry", "the lock moved anyway"


@pytest.mark.parametrize("by", ["grader", "Primary", "review rung", "calibrate", ""])
def test_the_graders_answer_is_never_the_reader_behind_a_relabel(tmp_path, repo, by):
    root = _root(tmp_path, repo)
    _edit(root, "negative/bad.json", label="pass",
          relabel={"from": "retry", "to": "pass", "by": by, "on": "2026-12-01",
                   "reason": "calibrate answered pass on three runs"})
    with pytest.raises(ValueError, match="must not be the grader"):
        CC.write_lock(root)


def test_a_named_dated_reread_relabels_and_a_mismatched_record_does_not(tmp_path, repo):
    root = _root(tmp_path, repo)
    rec = {"from": "pass", "to": "retry", "by": "Alan", "on": "2026-12-01",
           "reason": "re-read the diff: clause 2 names a file the commit never touches"}
    _edit(root, "independent/good.json", label="retry", relabel={**rec, "from": "retry"})
    with pytest.raises(ValueError, match="the change is pass -> retry"):
        CC.write_lock(root)
    _edit(root, "independent/good.json", relabel={**rec, "reason": "wrong"})
    with pytest.raises(ValueError, match="what the re-read found"):
        CC.write_lock(root)
    _edit(root, "independent/good.json", relabel=rec)
    CC.write_lock(root)
    assert CC.check_lock(root) == []
    assert CC.read_lock(CC.lock_path(root))["independent/good"][0] == "retry"


def test_a_new_diff_under_an_old_name_and_a_vanished_case_are_both_refused(tmp_path, repo):
    root = _root(tmp_path, repo)
    _edit(root, "independent/good.json", commit="1" * 40)
    assert "commit, parent or transform" in CC.check_lock(root)[0]
    with pytest.raises(ValueError, match="give it a new name"):
        CC.write_lock(root)
    root2 = _root(tmp_path / "b", repo)
    (root2 / "negative" / "bad.json").unlink()
    assert "fixture is gone" in CC.check_lock(root2)[0]
    with pytest.raises(ValueError, match="never by regenerating"):
        CC.write_lock(root2)


def test_a_new_case_is_unlocked_until_the_lock_is_regenerated(tmp_path, repo):
    root = _root(tmp_path, repo)
    r, parent, landed = repo
    CC.write_case(_case(r, parent, landed, "drop_doc", ["doc.md"], name="bad2"), root / "negative")
    assert CC.check_lock(root) == ["negative/bad2: not in labels.lock (a new case is locked when "
                                   "it is added)"]
    CC.write_lock(root)
    assert CC.check_lock(root) == []


# ── the shipped negative set ─────────────────────────────────────────────

def test_the_negative_set_is_known_bad_cases_whose_labels_no_grader_gave():
    cases = RT.load_fixtures(NEG)
    assert 12 <= len(cases) <= 24, len(cases)
    assert len({c["name"] for c in cases}) == len(cases)
    assert not {c["name"] for c in cases} & {c["name"] for c in RT.load_fixtures(
        RT.FIXTURE_DIR / "independent")}
    kinds = collections.Counter()
    for c in cases:
        assert c["expect"]["kind"] == "retry", c["name"]
        assert c["label_source"] in CC.LABEL_SOURCES, c["name"]
        assert c["strip_tests"] is False, "the transform is in the commit, not applied at grading"
        assert "Never re-mark" in c["note"], c["name"]
        for v in c["violates"]:
            assert v["text"].strip() and v["clause"] in c["expect"]["unmet_clauses"], c["name"]
        if c["label_source"] == "constructed":
            t = c["transform"]
            kinds[t["kind"]] += 1
            assert t["kind"] in CC.TRANSFORMS and t["paths"], c["name"]
            assert c["violates"], f"{c['name']}: a constructed defect names the clause it breaks"
            assert c["tests_rung"] in ("green", "red", "not run"), c["name"]
            assert not set(t["paths"]) & set(c["changed_paths"]), c["name"]
            assert "by construction" in c["note"], c["name"]
        else:
            assert "transform" not in c, c["name"]
            assert len(c["evidence"]) > 100 and c["refusal"].strip() and c["fixed_by"], c["name"]
            assert c["violates"] or c["expect"]["honesty_finding"], c["name"]
    assert set(kinds) == set(CC.TRANSFORMS), f"every transform is represented: {dict(kinds)}"
    assert max(kinds.values()) <= sum(kinds.values()) / 2, (
        f"one transform is more than half the constructed cases: {dict(kinds)}")
    rounds = collections.Counter(c["round_id"] for c in cases)
    assert max(rounds.values()) == 1, "one round, one defect: two cases off one diff are not two"


def test_every_shipped_case_this_clone_can_see_still_derives_its_pinned_commit():
    """No object is written: the sha is recomputed into a scratch object dir."""
    checked = 0
    for c in RT.load_fixtures(NEG):
        need = [c["parent"], c.get("source_commit") or c["commit"]]
        if not all(CC.has_object(ROOT, s) for s in need):
            continue          # a refused head lives only under the live repo's round refs
        assert CC.verify_case(ROOT, c) == [], c["name"]
        checked += 1
    if not checked:
        pytest.skip("this clone holds none of the cases' commits")


# ── several sets, two rates ──────────────────────────────────────────────

def _rows(*triples):
    """(set, name, label, got) -> rows and cases the way calibrate_sets returns them."""
    rows, cases = [], []
    for s, name, label, got, *src in triples:
        cases.append({"name": name, "set": s, "expect": {"kind": label},
                      "label_source": src[0] if src else None})
        rows.append({"name": name, "set": s, "kind": got, "ok": got == label, "seconds": 2.0})
    return rows, cases


def test_the_combined_line_carries_both_errors_each_with_its_interval():
    rows, cases = _rows(("negative", "a", "retry", "pass", "constructed"),
                        ("negative", "b", "retry", "retry", "constructed"),
                        ("negative", "c", "retry", "retry", "refused_then_fixed"),
                        ("independent", "a", "pass", "retry"),      # same name, other set
                        ("independent", "d", "pass", "pass"),
                        ("independent", "e", "pass", None))
    r = CC.rates(rows, cases)
    assert (r["false_met"], r["labelled_retry"]) == (1, 3)
    assert (r["false_refusal"], r["labelled_pass"], r["ungraded"]) == (1, 2, 1)
    assert r["false_met_ci"] == [0.0615, 0.7923] and r["false_refusal_ci"] == [0.0945, 0.9055]
    assert r["false_met_by_source"] == {
        "constructed": {"k": 1, "n": 2, "ci": [0.0945, 0.9055]},
        "refused_then_fixed": {"k": 0, "n": 1, "ci": [0.0, 0.7935]}}
    assert CC.rates_line(rows, cases) == (
        "false met 1/3 (95% Wilson 6%–79%); false refusal 1/2 (95% Wilson 9%–91%) "
        "[false met by label source: constructed 1/2, refused_then_fixed 0/1]; 1 ungraded")


def test_the_cases_only_the_review_stands_in_front_of_are_counted_apart():
    """A defect whose own tests go red cannot land on a false `met`; a green one can."""
    rows, cases = _rows(("negative", "a", "retry", "pass", "constructed"),
                        ("negative", "b", "retry", "retry", "constructed"),
                        ("negative", "c", "retry", "pass", "constructed"))
    cases[0]["tests_rung"], cases[1]["tests_rung"], cases[2]["tests_rung"] = "green", "green", "red"
    assert CC.rates(rows, cases)["false_met_review_only"] == {"k": 1, "n": 2,
                                                              "ci": [0.0945, 0.9055]}
    assert CC.rates_line(rows, cases).endswith("[tests rung green, review alone: 1/2]")
    shipped = [c for c in RT.load_fixtures(NEG) if c.get("tests_rung") == "green"]
    assert len(shipped) >= 4, "too few shipped cases are ones a false met would land"


def test_an_absent_class_reads_unmeasured_never_zero():
    rows, cases = _rows(("independent", "d", "pass", "pass"))
    line = CC.rates_line(rows, cases)
    assert line == "false met unmeasured (no such case); false refusal 0/1 (95% Wilson 0%–79%)"
    assert CC.rates(rows, cases)["false_met_ci"] is None


def test_several_sets_grade_in_one_run_and_a_missing_commit_is_ungraded(tmp_path, repo,
                                                                         monkeypatch):
    root = _root(tmp_path, repo)
    r, parent, landed = repo
    _edit(root, "independent/good.json", name="good")
    ghost = json.loads((root / "independent" / "good.json").read_text())
    CC.write_case({**ghost, "name": "ghost", "commit": "2" * 40}, root / "independent")
    seen = []

    def fake_grade_commit(*, commit, parent, label, **kw):
        seen.append((label, commit))
        assert CC.has_object(r, commit), "the constructed commit was not built before grading"
        return {"kind": "pass", "clauses": [], "findings": "", "session_id": "s"}

    monkeypatch.setattr(RT, "grade_commit", fake_grade_commit)
    rows, cases = CC.calibrate_sets(["negative", "independent"], repo=r, root=root)
    assert [(x["set"], x["name"], x["kind"]) for x in rows] == [
        ("negative", "bad", "pass"), ("independent", "ghost", None),
        ("independent", "good", "pass")]
    built = json.loads((root / "negative" / "bad.json").read_text())["commit"]
    assert ("cal_bad", built) in seen and len(seen) == 2
    assert "not graded" in rows[1]["why"]
    e = CC.rates(rows, cases)
    assert (e["false_met"], e["labelled_retry"], e["false_refusal"], e["labelled_pass"],
            e["ungraded"]) == (1, 1, 0, 1, 1)
    only, _ = CC.calibrate_sets(["negative", "independent"], repo=r, root=root, only="good")
    assert [x["name"] for x in only] == ["good"]


# ── the trend ────────────────────────────────────────────────────────────

def test_a_recorded_run_is_one_row_and_the_scorecard_reads_the_newest_full_one(tmp_path):
    path = tmp_path / "state" / CC.RECORD_NAME
    assert CC.scorecard_reading(path) == {"measured": False, "runs": 0}
    assert "unmeasured, not zero" in CC.render_scorecard_reading(CC.scorecard_reading(path))
    rows, cases = _rows(("negative", "a", "retry", "pass", "constructed"),
                        ("negative", "b", "retry", "retry", "constructed"),
                        ("independent", "d", "pass", "pass"))
    first = CC.record_run(rows, cases, ["negative", "independent"], repo=tmp_path, path=path,
                          now=1_795_000_000.0)
    assert first["created_at"] == "2026-11-18T11:06:40Z" and first["seconds"] == 6.0
    assert first["verdicts"] == {"negative/a": "pass", "negative/b": "retry",
                                 "independent/d": "pass"}
    rows[0]["kind"] = "retry"
    CC.record_run(rows, cases, ["negative", "independent"], repo=tmp_path, path=path,
                  now=1_795_000_000.0 + 86400)
    CC.record_run(rows[:1], cases, ["negative"], repo=tmp_path, path=path, partial=True,
                  now=1_795_000_000.0 + 2 * 86400)
    with path.open("a") as fh:
        fh.write("not json\n")
    assert len(path.read_text().splitlines()) == 4
    got = CC.scorecard_reading(path, now=1_795_000_000.0 + 3 * 86400)
    assert got["measured"] and got["runs"] == 2, "the --only run entered the trend"
    assert (got["false_met"], got["labelled_retry"], got["age_days"]) == (0, 2, 2.0)
    assert got["previous"]["false_met"] == 1
    assert CC.render_scorecard_reading(got) == (
        "false met 0/2 (0%–66%) | false refusal 0/1 (0%–79%); 2 d old, run 2")
    assert len(CC.read_records(path, include_partial=True)) == 3


def test_the_record_lives_in_the_automod_state_dir():
    from scripts.automod import state as S
    assert CC.record_path() == S.STATE_DIR / "review_calibration.jsonl"


# ── the CLI ──────────────────────────────────────────────────────────────

def test_calibrate_takes_several_sets_prints_one_combined_line_and_records(monkeypatch, capsys,
                                                                            tmp_path):
    rows, cases = _rows(("negative", "a", "retry", "retry", "constructed"),
                        ("independent", "d", "pass", "retry"))
    for x in rows:
        x.update(why="w", findings="f", provisional=False, cause_class="unlabelled")
    asked = {}

    def fake(sets, **kw):
        asked.update(sets=sets, **kw)
        return rows, cases

    monkeypatch.setattr(CC, "calibrate_sets", fake)
    monkeypatch.setattr(CC, "record_path", lambda: tmp_path / "rc.jsonl")
    rc = RT.main(["calibrate", "--set", "negative", "--set", "independent", "--record"])
    out = capsys.readouterr().out
    assert asked["sets"] == ["negative", "independent"] and asked["only"] is None
    assert rc == 1, "a disagreement still exits 1"
    assert "[PASS] negative/a" in out and "[FAIL] independent/d" in out
    assert ("false met 0/1 (95% Wilson 0%–79%); false refusal 1/1 (95% Wilson 21%–100%)"
            in out)
    assert out.strip().splitlines()[-1] == "1/2 agree"
    assert len(CC.read_records(tmp_path / "rc.jsonl")) == 1


def test_record_without_a_set_is_refused(capsys):
    with pytest.raises(SystemExit):
        RT.main(["calibrate", "--record"])
    assert "--record needs --set" in capsys.readouterr().err
