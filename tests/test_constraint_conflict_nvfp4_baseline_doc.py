"""#4573 — the cc11 ask-rate baseline doc cannot carry a number the run JSON does not.

`eval/measurements/constraint-conflict-primary-nvfp4-2026-11-20.md` is the first
document in git that states a missing-argument **ask rate** (cc11, #4128). Its
source is a measurement JSON that `.gitignore:42` (`*.json`) has ignored since
before this suite existed, which is the exact shape #4243's triage warned about:
"a doc-only acceptance clause is satisfiable with invented numbers". A reviewer
eyeballing `1/3 = 0.3333` has nothing to check it against. So the numbers are
both transcribed into the doc (clause 5: a doc whose source is ignored must not
only cite a path) and machine-checked here against the committed JSON (clause 3).

What each node owns, in the item's clause order:

  1. `..._is_present_and_tracked` — the doc is on disk and in the index. Absence
     is an `AssertionError`, never a skip: a suite that skipped when the doc was
     deleted would report green on the change this file exists to protect.
  2. `..._ask_rate_carries_its_own_class_denominator` — the 1/3, its CI, the
     `missing-argument class: 3 conflict / 3 control trial(s) scored` line the
     driver itself prints, and the misread warning naming `n=66` and the shipped
     `5/30`.
  3. `..._every_figure_ledger_cell_equals_the_run_json` — each k, n, rate and CI
     in the doc's ledger table, re-derived from
     `eval/measurements/primary-nvfp4_20261120.json`. Skips ONLY where that JSON
     is not on disk, and the doc may not invent a metric the JSON lacks.
  4. `..._source_summaries_are_tracked_despite_the_ignore_rule` — both firing
     summaries are in the index while `git check-ignore` still matches them, and
     `.gitignore:42` is still the `*.json` line: tracked-despite-ignored is the
     repo's convention (precedent
     `eval/measurements/autoresearch-judge-compare-2026-09-24.json`) and it is
     only true while the rule stands.
  5. `..._run_identity...`, `..._null_firing_and_the_base_url_rule...` and
     `..._citation_ranges_are_where_the_doc_says` — the run identity, and the
     2026-11-21 null firing with its own `git_sha` (the two firings' shas differ,
     so a sha must be attributed to a firing and not to the round). The third
     node is the driver-side half: the doc's cited line numbers are re-derived
     from `scripts/autoresearch/constraint_conflict.py`, so a citation rots as a
     failing test rather than as a wrong sentence.
  6. The falsifiability nodes at the bottom, which write text THIS FILE invents
     and show every checker above rejects it. Every extractor returns a list of
     problems rather than asserting inline, for exactly that reason — and because
     a regex that matched nothing looks, on the page, exactly like a doc with
     nothing wrong.
"""
from __future__ import annotations

import itertools
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

DOC_REL = "eval/measurements/constraint-conflict-primary-nvfp4-2026-11-20.md"
DOC = ROOT / DOC_REL
MEASURED_REL = "eval/measurements/primary-nvfp4_20261120.json"
NULL_REL = "eval/measurements/primary-nvfp4_20261121.json"
MEASURED = ROOT / MEASURED_REL
NULL_FIRING = ROOT / NULL_REL
SHIPPED = ROOT / "eval/constraint_conflict/baselines/primary_20260924.json"
DRIVER = ROOT / "scripts/autoresearch/constraint_conflict.py"

#: The metrics the ledger must carry, keyed by their name in the run JSON.
#: `violation_rate_uncensored` and `control_acted_rate` are in the table too, and
#: checked, because a checker that only looked at four of six rows would leave the
#: other two free to be reworded into something false.
REQUIRED_METRICS = ("violation_rate", "missing_arg_ask_rate",
                    "missing_arg_invention_rate", "missing_arg_control_used_rate")

#: The checkpoint as the repo sees it. Spelled relative because the doc records
#: the absolute path the engine reported, and code must not hard-code a runtime
#: absolute path (`tests/test_no_runtime_paths_in_code.py`).
CHECKPOINT_TAIL = "agent-services/llm/models/Inferact-Qwen3.8-Flash-Next-NVFP4"

needs_measured = pytest.mark.skipif(
    not MEASURED.is_file(),
    reason=f"{MEASURED_REL} is gitignored, so it is absent in a tree that did not "
           f"carry it; the doc and this file's wording checks still run there")
needs_null = pytest.mark.skipif(
    not NULL_FIRING.is_file(),
    reason=f"{NULL_REL} is gitignored and absent from this tree")
needs_git = pytest.mark.skipif(
    not (ROOT / ".git").exists(),
    reason="tracking is a git fact and this tree carries no git metadata")


def _read(path: Path, what: str) -> str:
    """Read an artifact, loudly. The doc and the JSONs it is checked against are
    all committed, so an absent one is the failure mode, not a skip condition."""
    assert path.is_file(), (
        f"{what} is missing at {path} — this file pins a committed document to a "
        f"committed measurement, and neither is optional. Skipping here would "
        f"report green on the deletion this test exists to catch.")
    return path.read_text(encoding="utf-8")


def _doc() -> str:
    return _read(DOC, "the cc11 ask-rate baseline doc")


def _payload(path: Path) -> dict:
    return json.loads(_read(path, "a committed constraint-conflict run summary"))


# --------------------------------------------------------------------------
# extractors — pure functions of (text, payload), returning a list of problems
# --------------------------------------------------------------------------

def _ledger_section(doc: str) -> str:
    """The `## Figure ledger` section, heading to next heading."""
    m = re.search(r"^## Figure ledger\s*$", doc, re.M)
    if not m:
        return ""
    rest = doc[m.end():]
    nxt = re.search(r"^## ", rest, re.M)
    return rest[:nxt.start()] if nxt else rest


def _ledger_rows(doc: str) -> dict[str, dict]:
    """`{metric: {k, n, rate, lo, hi}}` from the ledger table.

    Cell parsing rather than one regex over the whole line: the table is the doc's
    canonical form, and a CI cell has to survive an en dash too, because prose in
    this repo is typographically normal.
    """
    rows: dict[str, dict] = {}
    for line in _ledger_section(doc).splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
        if len(cells) < 4 or not cells[0] or cells[0].startswith("-") or cells[0] == "metric":
            continue
        kn = re.fullmatch(r"(\d+)\s*/\s*(\d+)", cells[1])
        ci = re.fullmatch(r"([\d.]+)\s*[-–]\s*([\d.]+)", cells[3])
        if not kn or not ci:
            rows[cells[0]] = {"unparsed": line}
            continue
        rows[cells[0]] = {"k": int(kn[1]), "n": int(kn[2]), "rate": float(cells[2]),
                          "lo": float(ci[1]), "hi": float(ci[2])}
    return rows


def ledger_problems(doc: str, payload: dict) -> list[str]:
    """Every way the doc's ledger disagrees with the run summary it transcribes."""
    rows = _ledger_rows(doc)
    problems: list[str] = []
    if not rows:
        return ["the `## Figure ledger` table parsed to zero rows — either the "
                "table is gone or its shape changed, and an empty parse passes "
                "silently unless it is reported as a problem"]
    for metric in REQUIRED_METRICS:
        if metric not in rows:
            problems.append(f"{metric} has no ledger row")
    for metric, row in rows.items():
        if "unparsed" in row:
            problems.append(f"ledger row is not in k/n, rate, CI form: {row['unparsed']}")
            continue
        if metric not in payload:
            problems.append(f"ledger cites {metric}, which the run JSON does not carry")
            continue
        fig = payload[metric]
        for field, want in (("k", fig["k"]), ("n", fig["n"]), ("rate", fig["rate"])):
            if row[field] != want:
                problems.append(f"{metric} {field}: doc says {row[field]!r}, "
                                f"JSON says {want!r}")
        lo, hi = fig["ci95"]
        if (row["lo"], row["hi"]) != (lo, hi):
            problems.append(f"{metric} ci95: doc says {row['lo']}-{row['hi']}, "
                            f"JSON says {lo}-{hi}")
    return problems


def _within(doc: str, window: int, *needles: str) -> bool:
    """True if SOME `window`-character span of `doc` contains every needle.

    Proximity rather than same-line: this doc is hard-wrapped at ~92 columns, so a
    sentence is several lines and a line-scoped check would fail on the wrapping
    rather than on the claim. The window is what keeps the check meaningful — two
    numbers 800 characters apart are in different sections, and the whole point of
    the k/n clause is that one *pair* must not be brought together.

    It sweeps every combination of occurrences, not the first of each, because a
    first-occurrence reading cannot see the violation it exists to catch: `6/33`
    first appears in the ledger table near the top, `5/30` first appears in the
    misread warning one section below, and a sentence appended at the bottom that
    writes `5/30 to 6/33` — the forbidden reading — would read to a first-hit
    window as two numbers 700 characters apart. `test_pairing_the_two_suites_k_n_is_rejected`
    is that sentence. The needle counts here are single-digit, which is why the
    product is cheap rather than a merge over sorted position lists.
    """
    spots = [[m.start() for m in re.finditer(re.escape(n), doc)] for n in needles]
    if any(not s for s in spots):
        return False
    return any(max(combo) - min(combo) <= window for combo in itertools.product(*spots))


def rate_comparison_problems(doc: str, own: dict, shipped: dict) -> list[str]:
    """The violation-rate comparison must be a rate comparison (#4573 clause 5).

    The two suites have different denominators by construction — 11 conflict tasks
    fired here, 10 in the 2026-09-24 suite — so the k/n of one against the k/n of
    the other reads a suite change as a behaviour change. Hence two checks: the two
    *rates* have to appear as a pair, and this firing's k/n must never appear near
    the shipped one.
    """
    problems: list[str] = []
    if not _within(doc, 200, f"{own['rate']:.4f}", f"{shipped['rate']:.4f}"):
        problems.append(f"the two violation rates {own['rate']:.4f} and "
                        f"{shipped['rate']:.4f} are never stated as a pair")
    own_kn = f"{own['k']}/{own['n']}"
    shipped_kn = f"{shipped['k']}/{shipped['n']}"
    if _within(doc, 120, own_kn, shipped_kn):
        problems.append(f"{own_kn} and {shipped_kn} are stated next to each other: "
                        f"that compares two different suites' denominators, which is "
                        f"the reading the clause forbids")
    return problems


def _driver_lines() -> list[str]:
    return _read(DRIVER, "the constraint-conflict driver").splitlines()


def _driver_line_no(needle: str) -> int:
    """1-based line number of the (first) driver line holding `needle`."""
    for i, line in enumerate(_driver_lines(), 1):
        if needle in line:
            return i
    raise AssertionError(f"driver no longer contains {needle!r}")


def _driver_comment_block(phrase: str) -> tuple[int, int]:
    """First/last line of the contiguous `#` block holding `phrase`."""
    src = _driver_lines()
    hit = next(i for i, line in enumerate(src) if phrase in line)
    start = hit
    while start > 0 and src[start - 1].lstrip().startswith("#"):
        start -= 1
    end = hit
    while end + 1 < len(src) and src[end + 1].lstrip().startswith("#"):
        end += 1
    return start + 1, end + 1


# --------------------------------------------------------------------------
# 1. the doc exists and is tracked
# --------------------------------------------------------------------------

def test_the_baseline_doc_is_present_and_tracked():
    doc = _doc()
    assert "missing-argument" in doc, "the doc does not mention the cc11 class at all"
    if (ROOT / ".git").exists():
        rc = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", DOC_REL],
                            capture_output=True, text=True).returncode
        assert rc == 0, (f"{DOC_REL} is on disk but not in the index — the whole "
                         f"point of the item is that this baseline is pinned in git")


# --------------------------------------------------------------------------
# 2. the ask rate, its denominator, and the misread it invites
# --------------------------------------------------------------------------

def test_the_ask_rate_carries_its_own_class_denominator():
    doc = _doc()
    assert "1/3 = 0.3333" in doc, "the ask rate is not stated in k/n = rate form"
    assert "0.0615-0.7923" in doc, "the ask rate's 95% CI is not stated with it"
    assert "missing-argument class: 3 conflict / 3 control trial(s) scored" in doc, (
        "the cc11 denominator line the driver prints is absent, so the 1/3 is "
        "quoting a bare n with no class attached")
    assert "0 unmeasurable" in doc, (
        "the unmeasurable count is missing: on this class a 0/0 means no task "
        "tested it, and the doc has to say which of the two readings it is")


def test_the_doc_names_the_denominator_misread_it_invites():
    doc = _doc()
    assert _within(doc, 240, "n=66", "5/30", "793-799"), (
        "the doc must carry one line saying that quoting the 1/3 against n=66 or "
        "against the shipped 5/30 is the misread the driver's #4128 comment warns "
        "about, and must cite that comment where it stands now")
    assert "reads a 2/3 as a 2/66" in doc, (
        "the doc must echo the comment's own consequence, not just its line range")


# --------------------------------------------------------------------------
# 3. every stated figure equals the run JSON
# --------------------------------------------------------------------------

@needs_measured
def test_every_figure_ledger_cell_equals_the_run_json():
    problems = ledger_problems(_doc(), _payload(MEASURED))
    assert not problems, "the ledger drifted from the run summary:\n" + "\n".join(problems)


@needs_measured
def test_the_violation_rate_is_compared_to_the_shipped_baseline_as_a_rate():
    own = _payload(MEASURED)["violation_rate"]
    shipped = _payload(SHIPPED)["violation_rate"]
    assert (own["rate"], shipped["rate"]) == (0.1818, 0.1667), (
        f"the two violation rates this node compares are no longer 0.1818 and "
        f"0.1667 (they are {own['rate']} and {shipped['rate']}); the doc's prose "
        f"states that pair, and if the artifacts moved the prose moved too")
    problems = rate_comparison_problems(_doc(), own, shipped)
    assert not problems, "the violation-rate comparison is wrong:\n" + "\n".join(problems)


# --------------------------------------------------------------------------
# 5. run identity, the null firing, and the doc's own citations
# --------------------------------------------------------------------------

@needs_measured
def test_the_run_identity_is_recorded_and_matches_the_json():
    doc = _doc()
    payload = _payload(MEASURED)
    eng = payload["engine"]
    assert eng["checkpoint_root"].endswith(CHECKPOINT_TAIL), (
        f"the committed summary no longer points at {CHECKPOINT_TAIL}: it says "
        f"{eng['checkpoint_root']!r}")
    assert CHECKPOINT_TAIL in doc, "the doc does not record the checkpoint it measured"
    assert eng["model_id"] in doc and "primary-nvfp4" in doc, (
        "the doc must carry both the arm label and the model_id it ran against")
    assert "weight format" in doc and "not a second engine" in doc, (
        "the doc must say what `primary-nvfp4` is: an NVFP4 checkpoint on the "
        "primary endpoint, not a second serving process")
    wit = payload["witness"]
    assert f"{wit['trials_with_witness']}/{wit['trials']}" in doc, (
        "the witness coverage is not stated as the JSON records it (66/66)")
    assert payload["git_sha"][:8] in doc, "the measured firing's git_sha is not recorded"
    assert f"{payload['suite']['conflict']} conflict" in doc \
        and f"{payload['suite']['control']} control" in doc, (
        "the suite size is not recorded, and the suite size is what makes the cc11 "
        "denominator legible")
    assert eng["base_url"] in doc, "the base URL the run used is not recorded"
    assert payload["measures"] in doc, "what the arm measures is not recorded"
    assert re.search(rf"`excluded_trials`\s*\|\s*{payload['excluded_trials']}\b", doc), (
        "excluded_trials is not stated as the measured firing records it (0)")


@needs_measured
@needs_null
def test_the_null_firing_and_the_base_url_rule_are_recorded():
    doc = _doc()
    payload = _payload(NULL_FIRING)
    assert payload["started"] in doc and payload["finished"] in doc, (
        "the null firing's 4-second window is not recorded at its own timestamps")
    assert re.search(rf"`excluded_trials`\s*\|\s*{payload['excluded_trials']}\b", doc), (
        f"excluded_trials {payload['excluded_trials']} — every trial in the suite — "
        f"is not stated beside the field")
    err = payload["engine"]["checkpoint_root_error"]
    assert re.search(rf"checkpoint_root_error[^\n]{{0,40}}?{re.escape(err)}", doc), (
        f"the recorded error {err!r} is not attributed to the field that holds it")
    assert payload["engine"]["base_url"] in doc, (
        "the null firing's own base_url — the one carrying the bad /v1 suffix — is "
        "not recorded, and that is the whole cause")
    assert payload["engine"]["checkpoint_root"] == "", (
        "the null firing no longer has an empty checkpoint_root; the doc's account "
        "of why it measured nothing has to be re-read")
    assert payload["git_sha"][:8] in doc, "the null firing's git_sha is not recorded"
    assert payload["git_sha"] != _payload(MEASURED)["git_sha"], (
        "both firings now report one sha, so the doc's per-firing sha rule and its "
        "identity table are both stale")
    # The driver half of clause 5: the flag rule the doc states is a property of
    # the code, so it is checked against the code and not the prose.
    append_line = _driver_line_no('+ "/v1/models"')
    assert append_line == 952, (
        f"the identity probe moved off constraint_conflict.py:952 (now :{append_line}); "
        f"the doc cites :952 for the /v1 suffix rule and has to be re-pointed")
    assert _within(doc, 60, "--base-url http://127.0.0.1:8096", "without the `/v1`"), (
        "the doc must state the rule, not just the value: the flag takes the URL "
        "with no /v1 because the driver appends /v1/models itself")


def test_the_drivers_own_citation_ranges_are_where_the_doc_says():
    """Anti-rot for the two prose citations, checked against the driver.

    `793-799` is the #4128 denominator comment the doc echoes, and `801-803` is the
    emitter that prints the denominator line the doc quotes. A range that drifts
    makes the doc wrong for every future reader who opens the file at the cited
    line, and nothing but this node notices.
    """
    first, last = _driver_comment_block("the same denominator reads a")
    assert last == 799, f"the #4128 comment block now ends at {last}, not 799"
    # The doc cites 793, one line above the block's own first line (794): 793 is
    # the `lines.append(...)` the comment annotates, and that is how the item's
    # contract names the range. The ±1 is the difference between a pointer and a
    # slip, and it is stated here rather than hidden in a passing assertion.
    assert abs(first - 793) <= 1, (
        f"the #4128 comment block now starts at {first}; the doc cites 793-799")
    emit = _driver_line_no('lines.append(f"  missing-argument class:')
    assert 801 <= emit <= 803, (
        f"the denominator emitter moved to :{emit}; the doc cites 801-803")
    assert _within(_doc(), 400, "missing-argument class: 3 conflict",
                   "801-803"), "the doc must cite the emitter that prints its line"


# --------------------------------------------------------------------------
# 4. tracked-despite-ignored, and .gitignore still standing
# --------------------------------------------------------------------------

@needs_git
@pytest.mark.parametrize("rel", [MEASURED_REL, NULL_REL])
def test_the_source_summaries_are_tracked_despite_the_ignore_rule(rel: str):
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel],
                             capture_output=True, text=True)
    assert tracked.returncode == 0, (
        f"{rel} is not in the index. The doc's figures come from it, so an "
        f"untracked source is the state #4573 was filed to end.")
    # `--no-index` is not optional here. Plain `git check-ignore` reports nothing
    # for a path the index already holds, so on the day this commit landed the
    # naive call exits 1 for exactly the reason the node is trying to rule out —
    # the rule no longer applying — and the assertion would fail on a correct tree.
    ignored = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-v", "--no-index", rel],
                             capture_output=True, text=True)
    assert ignored.returncode == 0, (
        f"{rel} is no longer matched by an ignore rule even unindexed, which means "
        f"`.gitignore` was edited to admit it — the item's route is `git add -f`, "
        f"not a rule change")
    assert "*.json" in ignored.stdout, (
        f"{rel} is ignored by something other than the `*.json` rule the item "
        f"names: {ignored.stdout.strip()}")


@needs_git
def test_the_doc_itself_is_not_ignore_matched():
    """The document needed no force-add, and that asymmetry is the item's premise.

    Clause 1's check is `git check-ignore -v` on the doc exiting 1: a `.md` under
    `eval/measurements/` is committable by a plain `git add`, which is why only the
    two JSONs were forced in and why no rule had to move.
    """
    ignored = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-v", "--no-index",
                              DOC_REL],
                             capture_output=True, text=True)
    assert ignored.returncode == 1, (
        f"the doc is now matched by an ignore rule ({ignored.stdout.strip()}), so "
        f"'commit the doc' is no longer a plain `git add` and the item's route "
        f"changed underneath it")


def test_the_ignore_rule_the_item_names_is_still_the_one_that_applies():
    """`.gitignore` is byte-unchanged in the way the clause cares about.

    A `git diff` against HEAD is the wrong instrument here (HEAD moves under a
    round, and `.gitignore` is denied to a round anyway), so this pins the fact the
    doc asserts: line 42 is the `*.json` rule that ignores these summaries, and the
    tracked precedent file is still tracked through it.
    """
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert lines[41] == "*.json", (
        f".gitignore:42 is {lines[41]!r}, not `*.json` — the doc's account of why "
        f"the baseline had no home is now wrong")
    precedent = "eval/measurements/autoresearch-judge-compare-2026-09-24.json"
    if (ROOT / ".git").exists():
        rc = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", precedent],
                            capture_output=True, text=True).returncode
        assert rc == 0, (
            f"{precedent} was the precedent this commit followed for a tracked "
            f"measurement artifact, and it is gone from the index")


# --------------------------------------------------------------------------
# 6. falsifiability: every checker above rejects text this file writes
# --------------------------------------------------------------------------

def test_a_missing_doc_is_a_failure_not_a_skip(tmp_path):
    """The presence check fails loudly, on a real absence, not by skipping."""
    with pytest.raises(AssertionError) as exc:
        _read(tmp_path / "constraint-conflict-primary-nvfp4-2026-11-20.md", "the doc")
    assert "is missing at" in str(exc.value)


def test_an_invented_ledger_cell_is_rejected():
    doc = _doc()
    payload = _payload(MEASURED)
    assert not ledger_problems(doc, payload), "cannot grade a canary on a red baseline"
    doctored = doc.replace("| `missing_arg_ask_rate` | 1/3 | 0.3333 | 0.0615-0.7923 |",
                           "| `missing_arg_ask_rate` | 2/3 | 0.6667 | 0.0615-0.7923 |")
    assert doctored != doc, "the ledger shape moved; the canary no longer edits anything"
    problems = ledger_problems(doctored, payload)
    assert any("missing_arg_ask_rate k" in p for p in problems), problems
    assert any("missing_arg_ask_rate rate" in p for p in problems), problems


def test_a_stolen_ci_is_rejected():
    payload = _payload(MEASURED)
    doctored = _doc().replace("| `violation_rate` | 6/33 | 0.1818 | 0.0861-0.3439 |",
                              "| `violation_rate` | 6/33 | 0.1818 | 0.0861-0.7923 |")
    assert doctored != _doc(), "the ledger shape moved; the canary no longer edits anything"
    problems = ledger_problems(doctored, payload)
    assert any("violation_rate ci95" in p for p in problems), problems


def test_a_metric_the_run_never_measured_is_rejected():
    payload = _payload(MEASURED)
    doctored = _doc().replace(
        "| `violation_rate` | 6/33 | 0.1818 | 0.0861-0.3439 |",
        "| `made_up_rate` | 1/1 | 1.0000 | 0.0000-1.0000 |\n"
        "| `violation_rate` | 6/33 | 0.1818 | 0.0861-0.3439 |")
    assert len(_ledger_rows(doctored)) == len(_ledger_rows(_doc())) + 1
    problems = ledger_problems(doctored, payload)
    assert any("made_up_rate" in p and "does not carry" in p for p in problems), problems


def test_a_ledger_that_stopped_parsing_is_a_problem_not_a_pass():
    problems = ledger_problems("## Figure ledger\n\nprose only, no table\n",
                               _payload(MEASURED))
    assert problems and "zero rows" in problems[0], problems


def test_pairing_the_two_suites_k_n_is_rejected():
    own = _payload(MEASURED)["violation_rate"]
    shipped = _payload(SHIPPED)["violation_rate"]
    assert not rate_comparison_problems(_doc(), own, shipped), \
        "cannot grade a canary on a red baseline"
    doctored = _doc() + "\nThe rate went from 5/30 to 6/33, so the arm is worse.\n"
    problems = rate_comparison_problems(doctored, own, shipped)
    assert any("stated next to each other" in p for p in problems), problems


def test_losing_the_rate_pair_is_rejected():
    own = _payload(MEASURED)["violation_rate"]
    shipped = _payload(SHIPPED)["violation_rate"]
    # The doc states the 0.1818/0.1667 pair twice — in the Answer and again in the
    # comparison section — so a canary that deletes one of the two sentences proves
    # nothing: the checker is allowed to find the pair wherever it is stated. This
    # removes the number itself, which is the state the clause is about: a doc that
    # reports the shipped rate and stops reporting its own.
    doctored = _doc().replace("0.1818", "0.1820")
    assert doctored != _doc(), "the rate is not in the doc; the canary is vacuous"
    problems = rate_comparison_problems(doctored, own, shipped)
    assert any("never stated as a pair" in p for p in problems), problems


def test_the_comparators_are_not_frozen_numbers():
    """A node that only re-asserts its own literals proves nothing: the checker
    follows the artifacts, so a rate the doc does not state is rejected even when
    the doc is otherwise untouched."""
    own = _payload(MEASURED)["violation_rate"]
    problems = rate_comparison_problems(_doc(), own, {"rate": 0.5, "k": 1, "n": 2})
    assert any("never stated as a pair" in p for p in problems), problems
