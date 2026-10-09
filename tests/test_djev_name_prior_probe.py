"""The djev name-prior probe (#1452), graded against a scripted engine.

No test here reaches a real djev: `_http` is replaced wholesale, so the engine
is whatever policy the test hands it — one that follows the option
DESCRIPTIONS (a name-robust engine), one that follows the option NAMES (the
paper's failure), one that cannot decide, one that is not there. The probe has
to tell those apart, and it has to refuse to turn the last one into a number.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import app.paths as paths
import eval.djev_name_prior_probe as probe
from eval.stats import wilson_ci

ROOT = Path(probe.__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# A scripted engine
# ---------------------------------------------------------------------------

class FakeEngine:
    """Answers `/health` and `/v1/systemone` from `policy(state, keys) -> position`.

    The position is into the question's options as SENT, so a policy that
    returns a fixed position follows the descriptions and one that looks the
    winning key up by name follows the names. `None` answers a dead-even tie.
    """

    def __init__(self, policy, *, health=200, post_status=200, die_after=None):
        self.policy = policy
        self.health = health
        self.post_status = post_status
        self.die_after = die_after
        self.bodies: list[dict] = []

    def __call__(self, method, url, body=None, *, timeout):
        if url.endswith("/health"):
            return self.health, b"{}"
        if self.die_after is not None and len(self.bodies) >= self.die_after:
            raise probe.Unreachable("connection reset")
        self.bodies.append(json.loads(json.dumps(body)))
        if self.post_status != 200:
            return self.post_status, b"boom"
        q = body["questions"][probe.QID]
        keys = list(q["criteria"])
        pos = self.policy(body["state"], keys)
        if pos is None:
            probs = {k: 1.0 / len(keys) for k in keys}
        else:
            rest = 0.1 / max(1, len(keys) - 1)
            probs = {k: (0.9 if i == pos else rest) for i, k in enumerate(keys)}
        answer = {"type": "choice", "choice": keys[pos if pos is not None else 0],
                  "confidence": max(probs.values()), "probabilities": probs}
        return 200, json.dumps({"answers": {probe.QID: answer},
                                "diagnostics": {"questions": {probe.QID: {
                                    "label_mass": 0.9, "argmax_is_label": True}}}}).encode()


def follows_descriptions(state, keys):
    return 0


def follows_names(state, keys):
    """Pick whichever option carries the name production's control would pick."""
    for shape, spec in probe.SHAPES.items():
        control = list(spec["criteria"])
        if len(control) == len(keys) and set(keys) <= set(control) | {
                f"opt_{i + 1}" for i in range(len(keys))}:
            if keys[0].startswith("opt_"):
                return 0
            return keys.index(control[0])
    return 0


@pytest.fixture
def fixture_corpus():
    return probe.load_corpus(probe.FIXTURE_CORPUS)


def _run_main(monkeypatch, engine, tmp_path, *extra):
    monkeypatch.setattr(probe, "_http", engine)
    out = tmp_path / "out"
    code = probe.main(["--corpus", str(probe.FIXTURE_CORPUS), "--out-dir", str(out),
                       "--date", "2026-09-24", *extra])
    return code, out


# ---------------------------------------------------------------------------
# Clause 1 — three paired schemas, descriptions byte-identical, keys the only difference
# ---------------------------------------------------------------------------

def test_fixture_corpus_covers_every_seam_shape_with_at_least_30_prompts(fixture_corpus):
    shapes = {r["shape"] for r in fixture_corpus}
    assert len(fixture_corpus) >= probe.MIN_PROMPTS == 30
    assert shapes == {"dedupe", "entity", "edges", "rank"}


def test_control_keys_are_the_production_keys():
    from app import djev
    from eval.djev import schemas
    assert probe.variant_keys("dedupe", "control") == list(
        schemas.DEDUPE.spec["same_finding"]["criteria"])
    assert probe.variant_keys("entity", "control") == list(
        schemas.ENTITY.spec["same_entity"]["criteria"])
    assert probe.variant_keys("edges", "control") == list(
        schemas.EDGES.spec["edge_type"]["criteria"])
    assert probe.variant_keys("rank", "control") == list(djev.RANK_LEVELS)


@pytest.mark.parametrize("shape", sorted(probe.SHAPES))
def test_variants_differ_only_in_their_option_keys(shape):
    qs = probe.paired_schemas(shape)
    control = qs["control"]
    descriptions = list(control["criteria"].values())
    n = len(descriptions)
    for variant, q in qs.items():
        # Everything but the keys is byte-identical, descriptions in order.
        assert json.dumps(list(q["criteria"].values())) == json.dumps(descriptions)
        assert {k: v for k, v in q.items() if k != "criteria"} == \
               {k: v for k, v in control.items() if k != "criteria"}
    assert list(qs["nonce"]["criteria"]) == [f"opt_{i + 1}" for i in range(n)]
    assert list(qs["inverted"]["criteria"]) == list(reversed(list(control["criteria"])))
    assert list(qs["inverted"]["criteria"]) != list(control["criteria"])
    assert qs["control_repeat"] == control


def test_every_prompt_is_sent_four_times_with_the_same_state(monkeypatch, tmp_path,
                                                             fixture_corpus):
    engine = FakeEngine(follows_descriptions)
    code, _ = _run_main(monkeypatch, engine, tmp_path)
    assert code == 0
    assert len(engine.bodies) == 4 * len(fixture_corpus)
    for i, prompt in enumerate(fixture_corpus):
        sent = engine.bodies[4 * i: 4 * i + 4]
        assert {b["state"] for b in sent} == {prompt["state"]}
        descs = {json.dumps(list(b["questions"][probe.QID]["criteria"].values()))
                 for b in sent}
        assert len(descs) == 1
        # control_repeat is the last request and byte-identical to control.
        assert json.dumps(sent[3], sort_keys=True) == json.dumps(sent[0], sort_keys=True)


# ---------------------------------------------------------------------------
# Clause 2 — a same-variant repeat arm beside the two treatments
# ---------------------------------------------------------------------------

def test_a_name_following_engine_flips_inverted_but_not_the_repeat(monkeypatch, tmp_path):
    code, out = _run_main(monkeypatch, FakeEngine(follows_names), tmp_path)
    assert code == 0
    report = json.loads((out / "name_prior_2026-09-24.json").read_text())
    comps = report["comparisons"]
    assert set(comps) == {"repeat", "nonce", "inverted"}
    assert comps["repeat"]["flips"] == 0
    assert comps["inverted"]["flips"] == comps["inverted"]["decided"] == 32
    assert comps["inverted"]["exceeds_repeat_floor"] is True
    assert comps["nonce"]["flips"] == 0
    assert comps["nonce"]["exceeds_repeat_floor"] is False


def test_a_noisy_repeat_arm_keeps_a_treatment_from_being_credited(monkeypatch, tmp_path):
    """A replay that flips as much as the treatment: no name effect claimed."""
    calls = {"n": 0}

    def noisy(state, keys):
        calls["n"] += 1
        return (calls["n"] // 4) % 2 if calls["n"] % 4 in (2, 0) else 0

    code, out = _run_main(monkeypatch, FakeEngine(noisy), tmp_path)
    assert code == 0
    comps = json.loads((out / "name_prior_2026-09-24.json").read_text())["comparisons"]
    assert comps["repeat"]["flips"] > 0
    assert comps["inverted"]["exceeds_repeat_floor"] is False


# ---------------------------------------------------------------------------
# Clause 3 — Wilson CI and a tie count, argmax only, no magnitudes
# ---------------------------------------------------------------------------

def test_flip_rate_carries_its_wilson_interval():
    rows = [{"argmax": {"control": 0, "inverted": 1 if i < 7 else 0}} for i in range(31)]
    rows.append({"argmax": {"control": 0, "inverted": None}})
    stats = probe.compare(rows, "inverted")
    assert stats == {"n": 32, "decided": 31, "ties": 1, "flips": 7,
                     "flip_rate": round(7 / 31, 4),
                     "ci95": [round(x, 4) for x in wilson_ci(7, 31)]}


def test_ties_are_counted_and_left_out_of_the_denominator(monkeypatch, tmp_path):
    def undecided_rank(state, keys):
        return None if len(keys) == 4 else 0

    code, out = _run_main(monkeypatch, FakeEngine(undecided_rank), tmp_path)
    assert code == 0
    report = json.loads((out / "name_prior_2026-09-24.json").read_text())
    rank = report["by_shape"]["rank"]["inverted"]
    assert rank["ties"] == rank["n"] == 8 and rank["decided"] == 0
    assert rank["flip_rate"] is None and rank["ci95"] is None
    assert report["comparisons"]["inverted"]["ties"] == 8
    assert report["indeterminate"]["control"] == {"tie": 8}


def test_the_argmax_is_read_by_description_position_not_by_key():
    keys = ["same", "different"]
    payload = {"answers": {probe.QID: {"probabilities": {"same": 0.2, "different": 0.8}}}}
    assert probe.argmax_position(payload, keys) == (1, "")
    assert probe.argmax_position({"answers": {}}, keys) == (None, "no_answer")
    wrong = {"answers": {probe.QID: {"probabilities": {"a": 0.2, "b": 0.8}}}}
    assert probe.argmax_position(wrong, keys) == (None, "label_set")
    even = {"answers": {probe.QID: {"probabilities": {"same": 0.5, "different": 0.5}}}}
    assert probe.argmax_position(even, keys) == (None, "tie")


def test_the_report_carries_no_score_magnitudes(monkeypatch, tmp_path):
    code, out = _run_main(monkeypatch, FakeEngine(follows_names), tmp_path)
    assert code == 0
    report = json.loads((out / "name_prior_2026-09-24.json").read_text())
    keys = set(probe._keys_anywhere(report))
    assert not keys & probe._MAGNITUDE_KEYS
    assert report["method"]["argmax_only"] is True
    report["rows"][0]["confidence"] = 0.9
    assert any("magnitudes" in e for e in probe.validate_report(report))


# ---------------------------------------------------------------------------
# Clause 4 — a dead engine is an explicit refusal, never 0 flips
# ---------------------------------------------------------------------------

def _dead(method, url, body=None, *, timeout):
    raise probe.Unreachable("[Errno 111] Connection refused")


def test_no_engine_prints_unreachable_writes_nothing_and_exits_nonzero(
        monkeypatch, tmp_path, capsys):
    code, out = _run_main(monkeypatch, _dead, tmp_path)
    captured = capsys.readouterr()
    assert code == 3
    assert "engine unreachable" in captured.err
    assert "flip" not in captured.out.lower()
    assert not out.exists() or not list(out.iterdir())


def test_unhealthy_engine_is_unreachable(monkeypatch, tmp_path, capsys):
    code, out = _run_main(monkeypatch, FakeEngine(follows_descriptions, health=503), tmp_path)
    assert code == 3
    assert "engine unreachable" in capsys.readouterr().err
    assert not out.exists()


def test_an_engine_that_dies_mid_run_leaves_no_report(monkeypatch, tmp_path, capsys):
    engine = FakeEngine(follows_descriptions, die_after=10)
    code, out = _run_main(monkeypatch, engine, tmp_path)
    assert code == 3
    assert "engine unreachable" in capsys.readouterr().err
    assert not out.exists()


def test_an_answer_out_of_shape_leaves_no_report(monkeypatch, tmp_path):
    code, out = _run_main(monkeypatch, FakeEngine(follows_descriptions, post_status=500),
                          tmp_path)
    assert code == 2
    assert not out.exists()


def test_dead_engine_exit_differs_from_the_determinism_probe():
    """The sibling probe exits 0 without an engine; this one must not."""
    src = (ROOT / "scripts" / "djev_determinism_probe.py").read_text()
    assert "exits 0, not 1, when the engine is not there" in src
    assert "exits 3" in probe.__doc__


# ---------------------------------------------------------------------------
# Clause 5 — non-interactive, date-stamped under the data root, a self-validating report
# ---------------------------------------------------------------------------

def test_report_is_date_stamped_and_validates_and_the_corpus_dir_stayed_behind(
        monkeypatch, tmp_path):
    """The report is date-stamped, and the two roots are the two #2455 separated.

    The line this node used to carry — `probe.REPORT_DIR == ROOT / "eval" / "djev"` —
    was the pin that MADE the old default load-bearing, so it moves here rather than
    being deleted: what must stay in the code tree is `CORPUS_DIR` (the pinned corpus and
    the fixture the weekly series is computed from), and what must never be in it again
    is `REPORT_ROOT`, which now resolves through `app.paths` exactly as
    `EVAL_BASELINES_DIR` does. Both halves are asserted because either one alone passes
    on a wrong split: pointing both at the data root would silently re-derive a
    measurement against a corpus that is no longer the tracked one, and pointing both at
    the tree re-creates the stray file that cost #2232 three rounds.
    """
    assert probe.CORPUS_DIR == ROOT / "eval" / "djev"
    assert probe.PINNED_CORPUS == ROOT / "eval" / "djev" / "name_prior_corpus.jsonl"
    assert probe.FIXTURE_CORPUS == ROOT / "eval" / "djev" / "name_prior_fixture.jsonl"
    assert probe.PINNED_CORPUS.is_file() and probe.FIXTURE_CORPUS.is_file(), (
        "a corpus the probe can no longer read is a changed measurement, not a moved path")
    assert probe.REPORT_ROOT == paths.EVAL_DJEV_REPORTS_DIR
    assert probe.REPORT_ROOT == paths.DATA_ROOT / "eval" / "djev"
    assert probe.REPORT_ROOT.is_relative_to(paths.DATA_ROOT)
    assert not probe.REPORT_ROOT.is_relative_to(ROOT), (
        f"an unflagged run would write into the code tree again: {probe.REPORT_ROOT}")
    assert probe.report_path(probe.REPORT_ROOT, "2026-09-24").name == (
        "name_prior_2026-09-24.json")

    code, out = _run_main(monkeypatch, FakeEngine(follows_descriptions), tmp_path)
    assert code == 0
    files = list(out.iterdir())
    assert [f.name for f in files] == ["name_prior_2026-09-24.json"]
    report = json.loads(files[0].read_text())
    assert probe.validate_report(report) == []
    assert report["corpus"]["n_prompts"] == 32
    assert report["corpus"]["kind"] == "custom"
    assert report["corpus"]["by_shape"] == {"dedupe": 8, "entity": 8, "edges": 8, "rank": 8}


#: The `--date` the unflagged run below stamps its report with: a date NO committed report
#: carries, so the node can never overwrite published evidence. The two tracked reports are
#: `name_prior_2026-09-24.json` and `name_prior_2026-10-04.json`, and this round's branch
#: carried a commit (`31a652f4`) in which the scripted engine's zeros landed INSIDE the
#: tracked 2026-09-24 report — produced while proving the guard fires, by planting
#: `REPORT_ROOT = HERE / "djev"` and running a node that passed that very date; the tracked
#: bytes are restored off `48d92a66` in this round, and the assertion that this date names
#: no report on disk is what stops the same accident needing a planted bug to happen.
UNFLAGGED_REPORT_DATE = "2026-08-20"


def test_an_unflagged_run_writes_its_report_outside_the_code_tree(monkeypatch, capsys):
    """One `main()` with no `--out-dir` in argv, measured against the tree it lives in.

    The claim #2455 exists to make false is a scheduled task-96 run leaving a report in
    `~/lloyd`. So this node runs the real entry point the way task 96 runs it — the
    engine scripted, `--corpus` and `--date` given so the answer and the filename are
    fixed, and `--out-dir` DELIBERATELY ABSENT — and then reads the two places the report
    could have gone.

    The witness is `git status --porcelain` read across the whole tree and filtered to the
    paths this change owns (`name_prior`, `eval/djev`), asserted EMPTY rather than merely
    unchanged — so a stray the run created for any reason reddens it. The filter is about
    whose dirt is blamed, not about weakening the claim: the suite runs `-n 8` in the tree
    under review, where other nodes write into that tree concurrently
    (`scripts/automod/gate.py` says so above its vault mirror), so demanding a wholly empty
    porcelain would blame a neighbouring node's file for this probe's behaviour. The filter
    is asserted empty BEFORE the run as well, so a tree already dirty on one of the probe's
    own paths cannot silently absorb the run's dirt into "unchanged".

    The `--date` this node stamps is `UNFLAGGED_REPORT_DATE`, not the date of a committed
    report, and the report it expects is deleted before the run: `written.is_file()` after
    an asserted absence is what proves a report was NEWLY written rather than merely
    present on a data root that survives a re-run.
    """
    tree_reports = ROOT / "eval" / "djev"
    assert f"name_prior_{UNFLAGGED_REPORT_DATE}.json" not in {
        p.name for p in tree_reports.glob("name_prior_*.json")}, (
        f"{UNFLAGGED_REPORT_DATE} now names a report on disk; pick a date no committed "
        "report carries — with a code-tree default this node would overwrite published "
        "evidence instead of merely failing")

    def _probe_lines(porcelain: str) -> set[str]:
        # Scoped to PATHS, not to the substring `name_prior`: this very file is
        # `tests/test_djev_name_prior_probe.py`, so a substring test flags the node's own
        # source as a stray the probe left. The first run of this version failed its own
        # precondition that way, on ` M tests/test_djev_name_prior_probe.py`. A report is a
        # path under `eval/djev/`, or a `name_prior_<date>.json` file wherever it sits.
        lines = set()
        for ln in porcelain.splitlines():
            path = ln[3:] if len(ln) > 3 else ln
            leaf = path.rsplit("/", 1)[-1]
            if path.startswith("eval/djev/") or (leaf.startswith("name_prior_")
                                                 and leaf.endswith(".json")):
                lines.add(ln)
        return lines

    before_status = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                   capture_output=True, text=True).stdout
    assert _probe_lines(before_status) == set(), (
        "this tree is already dirty on a name_prior/eval-djev path, so it cannot measure "
        f"what a run leaves behind: {sorted(_probe_lines(before_status))}")
    before_tree = sorted(p.name for p in tree_reports.iterdir())
    data_root = probe.REPORT_ROOT
    written = data_root / f"name_prior_{UNFLAGGED_REPORT_DATE}.json"
    written.unlink(missing_ok=True)  # only this node's own dated file, on a scratch root
    assert not written.exists()
    before_data_names = {p.name for p in data_root.iterdir()} if data_root.is_dir() else set()

    monkeypatch.setattr(probe, "_http", FakeEngine(follows_descriptions))
    # The corpus the sibling node runs the entry point on, so the ONE thing this node
    # varies from a run already known to exit 0 in every tree is the absent `--out-dir`.
    # The gate's tests rung reported `assert code == 0` failing here, and a nonzero exit
    # is a probe refusal whose reason it prints — so the reason travels in the message.
    code = probe.main(["--corpus", str(probe.PINNED_CORPUS),
                       "--date", UNFLAGGED_REPORT_DATE])

    printed = capsys.readouterr()
    assert code == 0, (
        f"the probe refused (exit {code}) before writing anything:\n"
        f"{printed.out}{printed.err}")
    # Absent a moment ago, here now: that pair is the newness proof, on a root that may
    # well have existed already. `report: <path>` is the probe's own last line, so the
    # destination it printed is checked too rather than inferred from the filename.
    assert written.is_file(), (
        f"the default root is {data_root}, which is not under the data root "
        f"({paths.DATA_ROOT}) — so --out-dir defaulted somewhere else again")
    assert f"report: {written}" in printed.out, (
        f"the probe exited 0 without printing the report it wrote:\n{printed.out}")
    report = json.loads(written.read_text())
    assert probe.validate_report(report) == []
    assert report["date"] == UNFLAGGED_REPORT_DATE
    assert data_root.is_dir(), "the default root was not created"

    assert sorted(p.name for p in tree_reports.iterdir()) == before_tree, (
        "a run wrote a file into the tracked eval/djev/ directory")
    after_status = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                  capture_output=True, text=True).stdout
    assert _probe_lines(after_status) == set(), (
        "an unflagged run left a path in the code tree: app/live_strays.py counts that "
        "as a stray and a person has to commit it, which is the #2232 wall\n  "
        + "\n  ".join(sorted(_probe_lines(after_status))))
    # Nothing but this node's own report may APPEAR in the data root: the diff against the
    # listing taken before the run, not a demand about what the directory holds overall.
    # Written as a subset rather than an equality because under `-n 8` another worker can
    # write into the shared redirected data root while this node runs, and the first version
    # demanded exactly one new entry, which the tests rung reported red on its own re-run
    # (`the run wrote [] ... not just its own report`). The unlink above is what carries
    # newness; this line only owns that the run wrote nothing else.
    appeared = {p.name for p in data_root.iterdir()} - before_data_names
    assert appeared <= {written.name}, (
        f"the run wrote {sorted(appeared)} into {data_root}, not just its own report")


def test_a_code_tree_default_leaves_a_stray_and_the_shipped_default_leaves_nothing(
        monkeypatch, tmp_path):
    """The counterfactual to the node above, laid in a throwaway checkout and read
    through `app.live_strays`, which is what counts a stray for the gate.

    The node above asserts an absence, and an absence passes for a change that never
    happened: `REPORT_ROOT` could point anywhere outside the tree and `git status
    --porcelain` would still be clean, so that node alone cannot tell "#2455 landed" from
    "#2455 was never attempted". This one measures both sides of the change in the same
    run — the old `HERE / "djev"` default and the shipped one — against a scratch git
    repository holding a byte-for-byte copy of the tracked `eval/djev/` files.

    It does not lay that default in the real checkout. The first attempt at this guard
    planted `REPORT_ROOT = HERE / "djev"` in the live tree and ran with a committed
    report's date, and the scripted engine's zeros were written INTO
    `eval/djev/name_prior_2026-09-24.json` — a published measurement, restored off
    `48d92a66` in this round. Here the damage can only ever land on a copy, and the last
    assertion checks the copy of that report still matches the tracked bytes.
    """
    from app import live_strays

    repo = tmp_path / "checkout"
    (repo / "eval" / "djev").mkdir(parents=True)
    for src in sorted((ROOT / "eval" / "djev").iterdir()):
        if src.is_file():
            shutil.copy2(src, repo / "eval" / "djev" / src.name)
    for cmd in (["init", "-q"], ["add", "-A"],
                ["-c", "user.email=tests@lloyd.local", "-c", "user.name=lloyd-tests",
                 "commit", "-qm", "fixture: a copy of the tracked eval/djev/"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True, capture_output=True)
    assert live_strays.untracked(repo) == set(), "the scratch checkout starts dirty"

    old_default = repo / "eval" / "djev"          # what `HERE / "djev"` used to resolve to
    shipped = paths.DATA_ROOT / "eval" / "djev"
    corpus = str(old_default / "name_prior_corpus.jsonl")
    monkeypatch.setattr(probe, "_http", FakeEngine(follows_descriptions))

    monkeypatch.setattr(probe, "REPORT_ROOT", old_default)
    assert probe.main(["--corpus", corpus, "--date", UNFLAGGED_REPORT_DATE]) == 0
    stray = f"eval/djev/name_prior_{UNFLAGGED_REPORT_DATE}.json"
    assert live_strays.untracked(repo) == {stray}, (
        "a run through the old code-tree default did not produce the stray "
        f"{stray!r} that app/live_strays.py counts: the premise of #2455 is being "
        "contradicted, and the nodes pinning the new default prove nothing about it")

    (old_default / f"name_prior_{UNFLAGGED_REPORT_DATE}.json").unlink()
    monkeypatch.setattr(probe, "REPORT_ROOT", shipped)
    assert probe.main(["--corpus", corpus, "--date", UNFLAGGED_REPORT_DATE]) == 0
    assert live_strays.untracked(repo) == set(), (
        f"the shipped default wrote into a checkout again: {stray}")
    assert (old_default / "name_prior_2026-09-24.json").read_bytes() == (
        (ROOT / "eval" / "djev" / "name_prior_2026-09-24.json").read_bytes()), (
        "the run changed a committed report's bytes, which no probe run is allowed to do")


def test_default_report_path_is_git_tracked_not_ignored():
    res = subprocess.run(["git", "check-ignore", "-q", "eval/djev/name_prior_2026-09-24.json"],
                         cwd=ROOT, capture_output=True)
    assert res.returncode == 1  # 1 = not ignored


def test_validate_report_rejects_a_broken_report():
    assert probe.validate_report({}) != []
    assert probe.validate_report({"probe": "x"}) != []


def test_every_committed_report_validates():
    for path in sorted((ROOT / "eval" / "djev").glob("name_prior_*.json")):
        report = json.loads(path.read_text())
        assert probe.validate_report(report) == [], path.name
        assert report["corpus"]["n_prompts"] >= probe.MIN_PROMPTS


def test_runs_non_interactively_through_the_venv_python(tmp_path):
    """A real subprocess, stdin closed, pointed at a port nothing listens on."""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    res = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "djev_name_prior_probe.py"),
         "--url", "http://127.0.0.1:9", "--out-dir", str(tmp_path / "o"),
         "--corpus", str(probe.FIXTURE_CORPUS)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, env=env)
    assert res.returncode == 3, res.stderr
    assert "engine unreachable" in res.stderr
    assert not (tmp_path / "o").exists()


def test_a_corpus_under_30_prompts_is_refused(tmp_path, monkeypatch, capsys):
    small = tmp_path / "small.jsonl"
    rows = probe.FIXTURE_CORPUS.read_text().splitlines()[:29]
    small.write_text("\n".join(rows) + "\n")
    monkeypatch.setattr(probe, "_http", FakeEngine(follows_descriptions))
    assert probe.main(["--corpus", str(small), "--out-dir", str(tmp_path / "o")]) == 2
    assert "needs >= 30" in capsys.readouterr().err


def test_build_corpus_rebuilds_seam_states_from_shadow_meta(tmp_path):
    shadow = tmp_path / "shadow.jsonl"
    shadow.write_text("\n".join(json.dumps(r) for r in [
        {"seam": "dedupe", "ts": 1, "meta": {"name": "new finding",
                                             "candidates": [{"id": 7}, {"id": 8}]}},
        {"seam": "entity", "ts": 2, "meta": {"a": "Foo", "b": "Foo System"}},
        {"seam": "entity", "ts": 3, "meta": {"a": "NoDef", "b": "Foo"}},
        {"seam": "rerank", "ts": 4, "meta": {"query": "how?"}, "actual": ["a.md", "b.md"]},
        "a bare string row",
    ]) + "\n{truncated\n")
    readers = {
        "backlog_head": lambda i: {"id": int(i), "title": f"item {i}", "text": "body"},
        "definition": lambda n: "" if n == "NoDef" else f"def of {n}",
        "vault_text": lambda rel: f"text of {rel}",
        "edges": lambda limit: [{"a_title": "S", "b_title": "T", "quote": "S uses T"}],
    }
    rows = probe.build_corpus(5, shadow_path=shadow, readers=readers)
    by = {s: [r for r in rows if r["shape"] == s] for s in probe.SHAPES}
    assert len(by["dedupe"]) == 2 and "#7 item 7" in by["dedupe"][0]["state"]
    assert len(by["entity"]) == 1   # a pair with no definition is not judged
    assert len(by["rank"]) == 2 and by["rank"][0]["state"].startswith("Query: how?")
    assert len(by["edges"]) == 1
    assert len({r["id"] for r in rows}) == len(rows)
