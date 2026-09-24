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
import subprocess
import sys
from pathlib import Path

import pytest

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
# Clause 5 — non-interactive, date-stamped under eval/djev/, a self-validating report
# ---------------------------------------------------------------------------

def test_report_is_date_stamped_under_eval_djev_and_validates(monkeypatch, tmp_path):
    assert probe.REPORT_DIR == ROOT / "eval" / "djev"
    code, out = _run_main(monkeypatch, FakeEngine(follows_descriptions), tmp_path)
    assert code == 0
    files = list(out.iterdir())
    assert [f.name for f in files] == ["name_prior_2026-09-24.json"]
    report = json.loads(files[0].read_text())
    assert probe.validate_report(report) == []
    assert report["corpus"]["n_prompts"] == 32
    assert report["corpus"]["kind"] == "custom"
    assert report["corpus"]["by_shape"] == {"dedupe": 8, "entity": 8, "edges": 8, "rank": 8}


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
