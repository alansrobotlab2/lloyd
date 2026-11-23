"""The `judgment_replay` rung: a round that rewrites a judge is measured.

No engine, no backend, no wall clock: every place the rung reaches outside
itself is a method of `judgment_replay.Deps`, and `_Deps` below replaces all of
them. What is pinned is the decision — what fires the rung, what refuses, and
above all what is NOT a verdict (no baseline, a held lock, a run cut short).
"""

from __future__ import annotations

import contextlib
import json
import types
from pathlib import Path

import pytest
import yaml

from scripts.automod import gate as G
from scripts.automod import judgment_replay as JR
from scripts.automod import spec

ROOT = Path(__file__).resolve().parent.parent

TRIAGE_PATH = "workers/sources/autotriage.py"
REVIEW_PATH = "scripts/automod/review.py"

TRIAGE_SRC = '''
import re
HEAD = """You are triaging item {item_id}."""
TAIL = """Finish with VERDICT."""
PROMPT = HEAD + TAIL
GROUP_PROMPT = """Judge the cluster."""
SWEEP_PROMPT = """Rank them."""
NO_UMBRELLA_RULE = """No umbrellas."""
LIMIT = 3

def _origin(c):
    return "origin"

def render_prompt(candidate):
    """The single-item prompt."""
    return PROMPT.format(item_id=candidate.id) + _origin(candidate)

def execute(job):
    return LIMIT
'''


# ── mode ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,want", [
    (None, "off"), (False, "off"), ("off", "off"), ("OFF ", "off"),
    ("advisory", "advisory"), ("enforce", "enforce"), (True, "enforce"),
    # A typo neither switches the guard off nor starts refusing rounds.
    ("enforced", "advisory"), ("on", "advisory"), (1, "advisory"),
])
def test_mode_reads_what_yaml_makes_of_the_key(value, want):
    assert JR.mode(value) == want


def test_the_switch_ships_off_and_is_quoted():
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    assert cfg["automod"]["gate"]["judgment_replay"] == "off"
    assert JR.mode(cfg["automod"]["gate"]["judgment_replay"]) == "off"
    assert 'judgment_replay: "off"' in text, "a bare off is YAML false; the value is quoted"


# ── what fires it: the prompt's text, not the file ───────────────────────

def test_a_comment_a_docstring_and_unrelated_code_do_not_move_the_fingerprint():
    roots = ("render_prompt",)
    base = JR.fingerprint_source(TRIAGE_SRC, roots)
    edited = (TRIAGE_SRC
              .replace('"""The single-item prompt."""', '"""Reworded docstring."""')
              .replace("LIMIT = 3", "LIMIT = 4  # tuned")
              .replace("def execute(job):", "# a comment\ndef execute(job, extra=None):")
              .replace('GROUP_PROMPT = """Judge the cluster."""', 'GROUP_PROMPT = """Judge it."""'))
    assert edited != TRIAGE_SRC
    assert JR.fingerprint_source(edited, roots) == base


@pytest.mark.parametrize("old,new", [
    ("You are triaging item", "You are judging item"),          # a constant the root reaches
    ('return "origin"', 'return "where it came from"'),         # a helper the root calls
    ("PROMPT = HEAD + TAIL", "PROMPT = HEAD + TAIL + HEAD"),    # how the pieces are joined
])
def test_text_the_prompt_is_rendered_from_moves_it(old, new):
    roots = ("render_prompt",)
    assert old in TRIAGE_SRC
    assert (JR.fingerprint_source(TRIAGE_SRC.replace(old, new), roots)
            != JR.fingerprint_source(TRIAGE_SRC, roots))


def test_a_rebinding_at_the_foot_of_the_module_is_part_of_the_prompt():
    """`PROMPT = _IC.splice_triage_prompt(PROMPT)` is the last line of
    autotriage.py: every binding of a reached name counts, not the first."""
    roots = ("render_prompt",)
    spliced = TRIAGE_SRC + "\nPROMPT = PROMPT + ' CLASS: ...'\n"
    assert JR.fingerprint_source(spliced, roots) != JR.fingerprint_source(TRIAGE_SRC, roots)


def test_absent_and_unparseable_sources_have_fingerprints_of_their_own():
    roots = ("render_prompt",)
    assert JR.fingerprint_source(None, roots) == "absent"
    broken = JR.fingerprint_source("def (:", roots)
    assert broken.startswith("unparsed:") and broken != JR.fingerprint_source("def ):", roots)


def _reader(files: dict):
    return lambda path: files.get(path)


def test_changed_surfaces_names_the_judge_whose_text_moved_and_only_that_one():
    base = {TRIAGE_PATH: TRIAGE_SRC}
    single = {TRIAGE_PATH: TRIAGE_SRC.replace("You are triaging", "You are judging")}
    batch = {TRIAGE_PATH: TRIAGE_SRC.replace("Judge the cluster.", "Judge it.")}
    code = {TRIAGE_PATH: TRIAGE_SRC.replace("return LIMIT", "return LIMIT + 1")}
    assert JR.changed_surfaces(_reader(base), _reader(single), [TRIAGE_PATH]) == ["triage"]
    assert JR.changed_surfaces(_reader(base), _reader(batch), [TRIAGE_PATH]) == ["triage_batch"]
    assert JR.changed_surfaces(_reader(base), _reader(code), [TRIAGE_PATH]) == []


def test_a_surface_whose_file_is_not_in_the_diff_is_never_read():
    def read(path):
        raise AssertionError(f"read {path} for a diff that does not touch it")
    assert JR.changed_surfaces(read, read, ["app/x.py", "tests/test_x.py"]) == []


def test_every_root_names_a_symbol_the_real_tree_binds():
    """A renamed prompt constant must fail here, not silently stop firing the rung."""
    for surface, entry in JR.SURFACES.items():
        for path, roots in entry["paths"].items():
            names, _ = JR.closure((ROOT / path).read_text(encoding="utf-8"), roots)
            assert set(roots) <= set(names), (surface, path, set(roots) - set(names))
    assert {s for s, e in JR.SURFACES.items() if e["instrument"]} == {"triage", "grader"}


def test_the_real_triage_surface_reaches_the_prompt_text_and_stops_there():
    names, _ = JR.closure((ROOT / TRIAGE_PATH).read_text(encoding="utf-8"), ("render_prompt",))
    assert {"PROMPT", "PROMPT_HEAD", "PROMPT_TAIL", "BASH_CEILING_BULLET"} <= set(names)
    assert not {"execute", "parse_verdict", "GROUP_PROMPT", "SWEEP_PROMPT"} & set(names)
    assert len(names) < 25, f"the closure is meant to be the prompt, not the module: {names}"


# ── triage: paired per case ──────────────────────────────────────────────

def _tv(agree: dict) -> dict:
    return {k: {"got": "confirmed" if v else "not_confirmed", "agree": v}
            for k, v in agree.items()}


def _cases(n: int = 42) -> list[str]:
    return [f"good_confirm-{i}" for i in range(n)]


def _flip(base: dict, wrong: int = 0, right: int = 0) -> dict:
    head, keys = dict(base), sorted(base)
    for k in [k for k in keys if base[k]][:wrong]:
        head[k] = False
    for k in [k for k in keys if not base[k]][:right]:
        head[k] = True
    return head


@pytest.mark.parametrize("wrong,right,p", [(3, 0, 0.125), (0, 0, 1.0), (4, 1, 0.1875),
                                           (5, 1, 0.1094), (5, 0, 0.0312), (0, 3, 1.0)])
def test_sign_test_is_the_one_sided_binomial_tail(wrong, right, p):
    assert round(JR.sign_test_p(wrong, right), 4) == p


@pytest.mark.parametrize("wrong,right,status", [
    (0, 0, "pass"),
    (2, 0, "pass"),       # under the net threshold
    (3, 0, "refuse"),     # the smallest result that can refuse
    (4, 1, "pass"),       # net 3, but 4 of 5 is inside the sign-test bound
    (5, 1, "refuse"),
    (1, 3, "pass"),       # it got better
])
def test_triage_is_refused_on_net_flips_to_wrong_outside_the_sign_bound(wrong, right, status):
    base = {k: i >= 4 for i, k in enumerate(_cases())}        # 4 disagree at base
    out = JR.compare_triage(_tv(base), _tv(_flip(base, wrong, right)), cases=42)
    assert (len(out["flips_to_wrong"]), len(out["flips_to_right"])) == (wrong, right)
    assert out["status"] == status
    assert out["thresholds"]["provisional"] is True


def test_a_head_run_missing_too_many_cases_is_incomplete_not_a_verdict():
    base = {k: True for k in _cases()}
    head = {k: False for k in _cases()[:30]}                   # 30 flips, and 12 unanswered
    assert JR.compare_triage(_tv(base), _tv(head), cases=42)["status"] == "incomplete"
    few = {k: True for k in _cases()[:40]}                     # 2 unanswered is tolerated
    assert JR.compare_triage(_tv(base), _tv(few), cases=42)["status"] == "pass"


def test_an_errored_case_is_no_answer_rather_than_a_disagreement():
    rows = [{"key": "a", "got": "confirmed", "agree": True, "error": ""},
            {"key": "b", "got": "no_verdict", "agree": False, "error": "TimeoutError: x"},
            {"key": "c", "got": "no_verdict", "agree": False, "error": ""},
            {"key": "a", "got": "not_confirmed", "agree": False, "error": ""}]   # latest wins
    assert JR.triage_verdicts(rows) == {"a": {"got": "not_confirmed", "agree": False},
                                        "c": {"got": "no_verdict", "agree": False}}


# ── grader: a new false met ──────────────────────────────────────────────

NEG = [f"negative/{n}" for n in JR.GRADER_NEGATIVE]
IND = [f"independent/{n}" for n in JR.GRADER_INDEPENDENT]


def _gv(neg="retry", ind="pass", **over) -> dict:
    return {**{k: neg for k in NEG}, **{k: ind for k in IND}, **over}


def test_the_subset_is_the_green_known_bad_cases_first_then_good_landings():
    """A `pass` on a constructed defect whose own tests stay green is the error
    nothing else in the ladder would stop; those five are graded first so a run
    cut short still has them."""
    fixtures = ROOT / "eval" / "review_calibration"
    subset = JR.grader_subset()
    assert [kind for _s, _n, kind in subset] == ["retry"] * 5 + ["pass"] * 6
    for set_name, name, kind in subset:
        case = json.loads((fixtures / set_name / f"{name}.json").read_text(encoding="utf-8"))
        assert case["expect"]["kind"] == kind, name
        if kind == "retry":
            assert case.get("tests_rung") == "green" and case["label_source"] == "constructed"
        else:
            assert not case.get("provisional"), name
    green = {p.stem for p in (fixtures / "negative").glob("*.json")
             if json.loads(p.read_text(encoding="utf-8")).get("tests_rung") == "green"}
    assert set(JR.GRADER_NEGATIVE) == green, "every green constructed defect is in the subset"


def test_a_known_bad_case_the_base_refused_and_the_head_passes_refuses():
    out = JR.compare_grader(_gv(), _gv(**{NEG[2]: "pass"}))
    assert out["status"] == "refuse" and out["new_false_met"] == [NEG[2]]


def test_a_false_met_the_base_already_made_is_not_new():
    out = JR.compare_grader(_gv(**{NEG[0]: "pass"}), _gv(**{NEG[0]: "pass"}))
    assert out["status"] == "pass" and out["standing_false_met"] == [NEG[0]]


def test_a_new_false_refusal_is_reported_and_never_refuses():
    out = JR.compare_grader(_gv(), _gv(**{IND[1]: "retry"}))
    assert out["status"] == "pass" and out["new_false_refusal"] == [IND[1]]


def test_an_ungraded_known_bad_case_is_incomplete_and_an_ungraded_good_one_is_not():
    head = _gv()
    del head[NEG[4]]
    assert JR.compare_grader(_gv(), head)["status"] == "incomplete"
    head = _gv()
    del head[IND[5]]
    out = JR.compare_grader(_gv(), head)
    assert out["status"] == "pass" and out["head_ungraded"] == [IND[5]]


def test_no_base_grade_of_any_known_bad_case_is_unmeasured_not_a_pass():
    out = JR.compare_grader({k: "pass" for k in IND}, _gv(neg="pass"))
    assert out["status"] == "unmeasured" and out["new_false_met"] == []


def test_the_grader_baseline_is_keyed_by_what_the_prompt_was_not_by_commit():
    runs = [{"surface": "grader", "fingerprint": "FP", "ts": 5.0, "verdicts": {NEG[0]: "retry"}},
            {"surface": "grader", "fingerprint": "OTHER", "ts": 9.0, "verdicts": {NEG[0]: "pass"}},
            {"surface": "triage", "fingerprint": "FP", "ts": 9.0, "verdicts": {NEG[0]: "pass"}}]
    calibration = [{"commit": "aaa", "ts": 7.0, "verdicts": {NEG[1]: "retry", NEG[0]: "pass"}},
                   {"commit": "bbb", "ts": 8.0, "verdicts": {NEG[1]: "pass"}},
                   {"commit": "aaa", "ts": 6.0, "verdicts": {NEG[2]: None, NEG[3]: "retry"}}]
    asked: list[str] = []

    def fingerprint_at(commit):
        asked.append(commit)
        return {"aaa": "FP", "bbb": "OTHER"}[commit]
    out = JR.grader_baseline("FP", runs=runs, calibration=calibration,
                             fingerprint_at=fingerprint_at)
    # newest row per case wins; a case with no kind never overwrites one.
    assert out == {NEG[0]: "pass", NEG[1]: "retry", NEG[3]: "retry"}
    assert sorted(asked) == ["aaa", "bbb"], "one git read per commit"


def test_the_subset_runner_keeps_what_it_graded_when_one_case_breaks(tmp_path, monkeypatch):
    from scripts.automod import calibration_cases as CC
    monkeypatch.setattr(CC, "materialize", lambda repo, case: case["commit"])
    labels: list[str] = []

    def grade(**kw):
        labels.append(kw["label"])
        if "4274" in kw["label"]:
            raise RuntimeError("backend went away")
        if "3981" in kw["label"]:
            return {"error": "no structured review", "session_id": "s"}
        return {"kind": "retry" if "stripped" in kw["label"] else "pass", "session_id": "s"}
    out = tmp_path / "rows.jsonl"
    assert JR.run_grader_subset(out, label="SM_X", grade=grade, repo=ROOT) == 0
    rows = {r["key"]: r for r in map(json.loads, out.read_text().splitlines())}
    assert len(rows) == 11 and len(set(labels)) == 11, "a label of its own per case and round"
    assert all(label.startswith("jr_SM_X_") for label in labels)
    assert rows["negative/4274-stripped-tests"]["kind"] is None
    assert "backend went away" in rows["negative/4274-stripped-tests"]["error"]
    assert rows["independent/3981-gh-token-fallback"]["kind"] is None
    assert rows["negative/4180-stripped-tests"]["kind"] == "retry"


# ── the rung ─────────────────────────────────────────────────────────────

REVIEW_SRC = 'REVIEW_SCHEMA = {"type": "object"}\n\ndef build_prompt(**kw):\n    return "Grade the diff."\n'
CASES_TEXT = json.dumps({"cases": [{"key": k} for k in _cases()]})


class _Deps(JR.Deps):
    """Every outside reach replaced. `head_rows` is what the 'engine' answers."""

    def __init__(self, *, mode="enforce", base=None, head=None, changed=(), runs=(),
                 calibration=(), head_rows=None, ran=None, up=True, lock_held=False,
                 calibration_fp=None):
        self.gate = types.SimpleNamespace(
            round_id="SM_TEST", report=types.SimpleNamespace(changed_paths=list(changed)))
        self._mode, self._base, self._head = mode, base or {}, head or {}
        self._runs, self._calibration = list(runs), list(calibration)
        self._head_rows, self._ran = head_rows or [], ran or {"returncode": 0, "timed_out": False}
        self._up, self._lock_held, self._calibration_fp = up, lock_held, calibration_fp or {}
        self.engine_runs: list[str] = []
        self.recorded: list[dict] = []
        self.events: list[dict] = []
        self.locked = 0
        self._scratch = None

    def mode(self):
        return self._mode

    def read_base(self, path):
        return self._base.get(path)

    def read_head(self, path):
        return self._head.get(path)

    def head_commit(self):
        return "headsha"

    def runs(self):
        return list(self._runs)

    def calibration(self):
        return list(self._calibration)

    def grader_fingerprint_at(self, commit):
        return self._calibration_fp.get(commit, "")

    def record(self, **row):
        self.recorded.append(row)
        return row

    def append_event(self, event):
        self.events.append(event)

    @contextlib.contextmanager
    def locks(self):
        if self._lock_held:
            raise JR.Blocked("regression.lock still held after 1800s (a bench)")
        self.locked += 1
        yield

    def scratch(self):
        return self._scratch

    def engine(self):
        return self._up, "http://engine.invalid", "probe"

    def _write(self, out):
        Path(out).write_text("".join(json.dumps(r) + "\n" for r in self._head_rows))
        return dict(self._ran)

    def run_triage(self, base_url, out):
        self.engine_runs.append("triage")
        return self._write(out)

    def run_grader(self, out):
        self.engine_runs.append("grader")
        return self._write(out)


def _triage_files(text="You are triaging"):
    return {TRIAGE_PATH: TRIAGE_SRC.replace("You are triaging", text),
            "scripts/automod/item_class.py": "def splice_triage_prompt(p):\n    return p\n",
            JR.TRIAGE_CASES: CASES_TEXT}


def _triage_rows(agree: dict) -> list[dict]:
    return [{"key": k, "got": "confirmed" if v else "not_confirmed", "agree": v, "error": ""}
            for k, v in agree.items()]


def _triage_deps(tmp_path, *, head_agree=None, baseline=True, **kw):
    base, head = _triage_files(), _triage_files("You are judging")
    all_agree = {k: True for k in _cases()}
    runs = list(kw.pop("runs", []))
    if baseline:
        runs.append({"event": JR.EVENT, "surface": "triage", "created_at": "2026-11-22T00:00:00Z",
                     "fingerprint": JR.surface_fingerprint("triage", _reader(base)),
                     "cases_sha": JR.text_sha(CASES_TEXT), "verdicts": _tv(all_agree)})
    d = _Deps(base=base, head=head, changed=[TRIAGE_PATH], runs=runs,
              head_rows=_triage_rows(all_agree if head_agree is None else head_agree), **kw)
    d._scratch = tmp_path
    return d


def _grader_deps(tmp_path, *, head_kinds=None, baseline=True, **kw):
    base = {REVIEW_PATH: REVIEW_SRC}
    head = {REVIEW_PATH: REVIEW_SRC.replace("Grade the diff.", "Grade the diff strictly.")}
    calibration = [{"commit": "c0", "ts": 1.0, "verdicts": _gv()}] if baseline else []
    rows = [{"key": k, "kind": v, "error": ""} for k, v in (head_kinds or _gv()).items()]
    d = _Deps(base=base, head=head, changed=[REVIEW_PATH], calibration=calibration,
              calibration_fp={"c0": JR.surface_fingerprint("grader", _reader(base))},
              head_rows=rows, **kw)
    d._scratch = tmp_path
    return d


def test_off_skips_before_reading_anything():
    d = _Deps(mode="off", changed=[TRIAGE_PATH])
    d.read_base = d.read_head = lambda p: pytest.fail("off reads nothing")
    ok, detail, data = JR.rung(d.gate, d)
    assert ok and data == {"skipped": True, "reason": "off"} and not d.events


def test_a_diff_that_changes_no_prompt_text_skips_and_writes_no_ledger_row(tmp_path):
    files = _triage_files()
    edited = {**files, TRIAGE_PATH: files[TRIAGE_PATH].replace("return LIMIT", "return 7")}
    d = _Deps(base=files, head=edited, changed=[TRIAGE_PATH, "tests/test_x.py"])
    ok, detail, data = JR.rung(d.gate, d)
    assert ok and data == {"skipped": True, "reason": "not touched"}
    assert not d.events and not d.engine_runs
    assert "judgment_replay" in G.Gate.QUIET_SKIP_RUNGS, "a skip writes no gate row either"


def test_a_prompt_with_no_replay_set_is_recorded_uncovered_and_let_through():
    base = {"workers/sources/owed_check.py": 'PROMPT = "a"\nOWED_SCHEMA = {}\n\ndef build_prompt(o):\n    return PROMPT\n'}
    head = {k: v.replace('"a"', '"b"') for k, v in base.items()}
    d = _Deps(base=base, head=head, changed=list(base))
    ok, detail, data = JR.rung(d.gate, d)
    assert ok and data["status"] == "uncovered" and "skipped" not in data
    assert data["surfaces"]["owed"]["status"] == "uncovered"
    assert d.events[0]["statuses"] == {"owed": "uncovered"} and not d.engine_runs


def test_a_triage_prompt_that_still_confirms_what_landed_passes_and_records_its_run(tmp_path):
    d = _triage_deps(tmp_path)
    ok, detail, data = JR.rung(d.gate, d)
    assert ok and data["status"] == "pass" and "42/42 agree" in detail
    assert d.engine_runs == ["triage"] and d.locked == 1
    # Written under the HEAD's fingerprint: the next round's base finds it.
    (row,) = d.recorded
    assert row["fingerprint"] == data["surfaces"]["triage"]["fingerprint_head"]
    assert row["surface"] == "triage" and row["source"] == "gate" and len(row["verdicts"]) == 42
    assert d.events[0]["event"] == "judgment_replay_gate" and d.events[0]["status"] == "pass"


def test_a_triage_prompt_that_stops_confirming_landed_work_is_refused(tmp_path):
    agree = {k: i >= 3 for i, k in enumerate(_cases())}
    d = _triage_deps(tmp_path, head_agree=agree)
    ok, detail, data = JR.rung(d.gate, d)
    assert not ok and data["status"] == "refuse" and "external_blocker" not in data
    assert "3 case(s) flipped" in detail and "provisional" in detail
    assert data["surfaces"]["triage"]["flips_to_wrong"] == _cases()[:3]


def test_advisory_records_what_enforce_would_have_refused_and_passes(tmp_path):
    agree = {k: i >= 3 for i, k in enumerate(_cases())}
    d = _triage_deps(tmp_path, head_agree=agree, mode="advisory")
    ok, detail, data = JR.rung(d.gate, d)
    assert ok and detail.startswith("ADVISORY (refuse")
    assert data["status"] == "refuse" and data["mode"] == "advisory"
    assert d.events[0]["status"] == "refuse" and d.events[0]["mode"] == "advisory"


def test_no_baseline_is_unmeasured_external_under_enforce_and_never_a_pass(tmp_path):
    d = _triage_deps(tmp_path, baseline=False)
    ok, detail, data = JR.rung(d.gate, d)
    assert not ok and data["status"] == "unmeasured" and data["external_blocker"] is True
    assert "judgment_replay record triage" in detail
    assert not d.engine_runs, "nothing to compare against: the engine is not spent"
    adv = _triage_deps(tmp_path, baseline=False, mode="advisory")
    ok, detail, data = JR.rung(adv.gate, adv)
    assert ok and data["status"] == "unmeasured" and "unmeasured" in detail


def test_a_baseline_for_other_cases_is_no_baseline(tmp_path):
    d = _triage_deps(tmp_path)
    d._runs[0]["cases_sha"] = "a-different-case-file"
    ok, _detail, data = JR.rung(d.gate, d)
    assert not ok and data["status"] == "unmeasured"


@pytest.mark.parametrize("kw,needle", [
    ({"lock_held": True}, "regression.lock still held"),
    ({"up": False}, "engine is not answering"),
    ({"ran": {"returncode": None, "timed_out": True, "tail": ""}}, "timed out"),
    ({"ran": {"returncode": 2, "timed_out": False, "tail": "No module named x"}}, "exited 2"),
])
def test_what_is_outside_the_diff_is_an_external_blocker_not_a_pass_or_a_refusal(tmp_path, kw, needle):
    d = _triage_deps(tmp_path, **kw)
    ok, detail, data = JR.rung(d.gate, d)
    assert not ok and needle in detail
    assert data["status"] == "external" and data["external_blocker"] is True
    assert data["retry_after_s"] == JR.RETRY_AFTER_S and data["external_failures"] == []
    assert not d.recorded, "a run that did not finish is not a baseline for anyone"
    adv = _triage_deps(tmp_path, mode="advisory", **kw)
    ok, _detail, data = JR.rung(adv.gate, adv)
    assert ok and data["status"] == "external" and "external_blocker" not in data


def test_an_engine_that_answered_too_few_cases_is_external_even_when_every_answer_flipped(tmp_path):
    d = _triage_deps(tmp_path, head_agree={k: False for k in _cases()[:20]})
    ok, detail, data = JR.rung(d.gate, d)
    assert not ok and data["status"] == "external" and "20/42" in detail and not d.recorded


def test_a_head_prompt_already_measured_is_answered_without_the_engine(tmp_path):
    """A re-gate after a review refusal, and the promoter's rebase chase, ask
    the same question of the same prompt: an hour of primary is spent once."""
    first = _triage_deps(tmp_path)
    JR.rung(first.gate, first)
    stored = {"event": JR.EVENT, "created_at": "now", **first.recorded[0]}
    again = _triage_deps(tmp_path, runs=[stored])
    ok, _detail, data = JR.rung(again.gate, again)
    assert ok and data["surfaces"]["triage"]["head_run"] == "recorded"
    assert not again.engine_runs and not again.recorded and again.locked == 0


def test_a_grader_that_still_refuses_the_known_bad_cases_passes(tmp_path):
    d = _grader_deps(tmp_path)
    ok, detail, data = JR.rung(d.gate, d)
    assert ok and data["status"] == "pass" and d.engine_runs == ["grader"] and d.locked == 1
    assert d.recorded[0]["surface"] == "grader" and len(d.recorded[0]["verdicts"]) == 11


def test_a_grader_rewrite_that_passes_a_known_bad_change_is_refused(tmp_path):
    d = _grader_deps(tmp_path, head_kinds=_gv(**{NEG[0]: "pass"}))
    ok, detail, data = JR.rung(d.gate, d)
    assert not ok and data["status"] == "refuse" and "external_blocker" not in data
    assert "NEW false `met`" in detail and NEG[0] in detail


def test_a_calibration_run_cut_short_after_the_known_bad_cases_still_decides(tmp_path):
    kinds = {k: "retry" for k in NEG} | {IND[0]: "pass"}
    d = _grader_deps(tmp_path, head_kinds=kinds, ran={"returncode": None, "timed_out": True})
    ok, _detail, data = JR.rung(d.gate, d)
    assert ok and data["status"] == "pass" and data["surfaces"]["grader"]["timed_out"] is True
    assert len(data["surfaces"]["grader"]["head_ungraded"]) == 5


def test_a_calibration_run_cut_short_before_them_is_external(tmp_path):
    kinds = {k: "retry" for k in NEG[:3]}
    d = _grader_deps(tmp_path, head_kinds=kinds, ran={"returncode": None, "timed_out": True})
    ok, detail, data = JR.rung(d.gate, d)
    assert not ok and data["status"] == "external" and data["external_blocker"] is True
    assert "did not answer" in detail and not d.recorded


def test_a_grader_with_no_recorded_baseline_is_unmeasured(tmp_path):
    d = _grader_deps(tmp_path, baseline=False)
    ok, detail, data = JR.rung(d.gate, d)
    assert not ok and data["status"] == "unmeasured" and "record grader" in detail
    assert not d.engine_runs


def test_a_refusal_on_one_surface_outranks_a_blocker_on_another(tmp_path):
    base = {**_triage_files(), REVIEW_PATH: REVIEW_SRC}
    head = {**_triage_files("You are judging"),
            REVIEW_PATH: REVIEW_SRC.replace("Grade the diff.", "Grade.")}
    agree = {k: i >= 3 for i, k in enumerate(_cases())}
    d = _Deps(base=base, head=head, changed=[TRIAGE_PATH, REVIEW_PATH],
              runs=[{"surface": "triage", "created_at": "x", "cases_sha": JR.text_sha(CASES_TEXT),
                     "fingerprint": JR.surface_fingerprint("triage", _reader(base)),
                     "verdicts": _tv({k: True for k in _cases()})}],
              head_rows=_triage_rows(agree))
    d._scratch = tmp_path
    ok, _detail, data = JR.rung(d.gate, d)
    assert not ok and data["status"] == "refuse" and "external_blocker" not in data
    assert {s: r["status"] for s, r in data["surfaces"].items()} == {
        "triage": "refuse", "grader": "unmeasured"}


def test_a_broken_instrument_is_external_not_a_crash(tmp_path):
    d = _triage_deps(tmp_path)
    d.run_triage = lambda *a: (_ for _ in ()).throw(OSError("no such interpreter"))
    ok, detail, data = JR.rung(d.gate, d)
    assert not ok and data["status"] == "external" and "OSError" in detail


# ── the store ────────────────────────────────────────────────────────────

def test_the_store_round_trips_and_skips_what_is_not_its_own(tmp_path):
    path = tmp_path / JR.STORE_NAME
    JR.record_run(surface="triage", fingerprint="fp", verdicts={"a": {"agree": True}},
                  cases_sha="s", source="record", now=100.0, path=path)
    with path.open("a") as fh:
        fh.write("not json\n" + json.dumps({"event": "something_else", "surface": "triage"}) + "\n")
    (row,) = JR.read_runs(path)
    assert row["fingerprint"] == "fp" and row["created_at"] == "1970-01-01T00:01:40Z"
    assert JR.read_runs(tmp_path / "absent.jsonl") == []


# ── the gate and the fence ───────────────────────────────────────────────

def test_the_gate_carries_the_rung_and_hands_itself_to_it(monkeypatch):
    seen = []
    monkeypatch.setattr(JR, "rung", lambda gate: seen.append(gate) or (True, "stub", {}))
    g = G.Gate.__new__(G.Gate)
    assert g.rung_judgment_replay() == (True, "stub", {}) and seen == [g]


def test_the_real_deps_run_the_rounds_tree_under_a_scratch_state_dir(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(JR, "run_child", lambda cmd, **kw: calls.append((cmd, kw)) or {})
    gate = types.SimpleNamespace(python=Path("/venv/python"), worktree=tmp_path / "wt",
                                 round_id="SM_R", _child_env=lambda: {"LLOYD_AUTOMOD_STATE": "s"})
    d = JR.Deps(gate)
    d.run_triage("http://engine.invalid", tmp_path / "t.jsonl")
    d.run_grader(tmp_path / "g.jsonl")
    for cmd, kw in calls:
        assert kw["cwd"] == tmp_path / "wt", "the ROUND's prompt, so the round's tree"
        assert kw["env"] == {"LLOYD_AUTOMOD_STATE": "s"} and 0 < kw["timeout"] <= 3600
    assert calls[0][0][1:3] == ["eval/triage_replay/run.py", "run"]
    assert calls[1][0][1:4] == ["-m", "scripts.automod.judgment_replay", "run-grader"]
    assert JR.LOCKS == ("regression.lock", "primary.lock")


@pytest.mark.parametrize("path", ["workers/sources/autotriage.py", "workers/sources/autocode.py",
                                  "workers/sources/owed_check.py", "scripts/automod/review.py",
                                  "scripts/automod/owed.py", "scripts/automod/item_class.py",
                                  "scripts/automod/judgment_replay.py"])
def test_every_file_that_holds_a_judge_takes_the_drill(path):
    assert spec.classify(path) == "protected" and spec.requires_drill([path])
    ok, _reason, buckets = spec.check_scope([path])
    assert ok and buckets["protected"] == [path], "protected is not denied: the round lands"


def test_the_rest_of_the_worker_sources_are_not_swept_in():
    assert spec.JUDGE_SOURCE_GLOBS == ("workers/sources/autotriage.py",
                                       "workers/sources/autocode.py",
                                       "workers/sources/owed_check.py")
    assert spec.classify("workers/sources/arch_review.py") == "allowed"
    assert spec.classify("workers/queue.py") == "allowed"
    for surface in JR.SURFACES.values():
        for path in surface["paths"]:
            assert spec.classify(path) == "protected", path


def test_a_round_may_not_switch_off_the_guard_nor_flip_its_own_selection():
    assert spec.JUDGE_CONFIG_DENIED_KEYS == ("automod.gate.judgment_replay", "automod.selection",
                                             "workers.sources.fitness")
    assert spec._config_denied_prefix("workers.sources.fitness.file_items") == "workers.sources.fitness"
    assert spec._config_denied_prefix("workers.sources.fitness.enabled") == "workers.sources.fitness"
    assert spec._config_denied_prefix("workers.sources.arch-review.spawn_cap") is None
    gate = 'automod:\n  gate:\n    judgment_replay: "{}"\n    test_workers: {}\n'
    ok, why, _ = spec.config_value_change(gate.format("enforce", 8), gate.format("off", 8))
    assert not ok and "automod.gate.judgment_replay" in why
    ok, why, changed = spec.config_value_change(gate.format("enforce", 8), gate.format("enforce", 4))
    assert ok and changed == ["automod.gate.test_workers"], "the rest of the gate block is a tunable"
    sel = "automod:\n  selection:\n    prefer_improvement: {}\n"
    ok, why, _ = spec.config_value_change(sel.format("false"), sel.format("true"))
    assert not ok and "automod.selection" in why
    assert "gate" not in spec.CONFIG_DENIED_BLOCKS and "selection" not in spec.CONFIG_DENIED_BLOCKS
