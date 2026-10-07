"""The mechanical gold-shape pass over a frozen LloydMemEval set (#2353).

Three things this suite pins, in the order the item states them: the dev leg's
three counts over 264 with the rules that produced them printed beside the
figures; the holdout leg counted as three numbers and never opened, per the
manifest's reserve_rule; and the report being a pure function of the set's bytes.

The exact figures pinned below (dev 70 / 0 / 26, holdout 19 / 0 / 4) are not
deposited numbers: each test recomputes at least one of them from the frozen YAML
with its own reading of the printed definitions, so a count that drifted apart
from the set fails here rather than agreeing with itself. They are also NOT the
figures #2170's filing quoted (61 / 2 / 24 and 16 / 0 / 5) — that pass left "what
is a gold value" and "what is a common English word" undefined, which is the gap
clause 1 exists to close.
"""
from __future__ import annotations

import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import label_audit as LA  # noqa: E402

SET = LA.SET_ROOT / "v2"
DEV_N, HOLDOUT_N = 264, 66


def _run(*argv: str) -> str:
    """The report stdout for one invocation, via main() so the flag wiring is what
    is exercised and not the library call the flags eventually reach."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert LA.main(list(argv)) == 0, f"exit non-zero for {argv}"
    return buf.getvalue()


def _counts(out: str, leg: str) -> dict[str, int]:
    """Parse the `  <defect>   n / N` lines back out of one leg's section."""
    parts = out.split(f"## leg: {leg} —", 1)
    assert len(parts) > 1, f"no section for leg {leg!r} in:\n{out}"
    body = parts[1].split("## leg:", 1)[0]
    rows = re.findall(r"^\s+(\S+)\s+(\d+) / (\d+)$", body, re.M)
    got = {d: int(n) for d, n, _n in rows}
    assert set(got) == set(LA.DEFECTS), f"expected the three defect rows, got {sorted(got)}"
    denoms = {int(n) for _, _, n in rows}
    assert len(denoms) == 1, f"one denominator expected, got {denoms}"
    return {**got, "n": denoms.pop()}


def _independent_counts(leg: str) -> dict[str, int]:
    """Recount the three defects straight from the YAML, reading the definitions
    off the script's own printed rule text rather than off its internals: a gold
    value is every alias of every group, the word count is len(value.split()), a
    common word is a listed word, a content word is not one."""
    qs = list((yaml.safe_load((SET / leg / "questions.yaml").read_text())
               or {}).get("questions") or [])
    # Deliberately not importing the module's punctuation: the recount has to be
    # independent enough to notice if the printed rule and the applied one differ.
    punct = " \t\n.,;:!?\"'()[]{}<>—–|/"
    words = set(LA.COMMON_WORDS)

    def toks(text):
        return {w.strip(punct).lower() for w in str(text).split()}

    over = sole = clash = 0
    for q in qs:
        acc = q.get("accept") or {}
        golds = [v for g in acc.get("all_of") or [] for v in (g if isinstance(g, list) else [g])]
        if any(len(str(v).split()) > LA.WORD_CAP for v in golds):
            over += 1
        distinct = {str(v).strip().lower() for v in golds}
        if len(distinct) == 1:
            only = next(iter(distinct))
            if only in words and len(only.split()) == 1:
                sole += 1
        gw = {t for v in golds for t in toks(v) if t and t not in words}
        if any({t for t in toks(a) if t} & gw for a in acc.get("none_of") or []):
            clash += 1
    return {LA.DEFECTS[0]: over, LA.DEFECTS[1]: sole, LA.DEFECTS[2]: clash, "n": len(qs)}


# ── clause 1: the dev leg, three counts over 264, with their rules ───────────

def test_the_dev_leg_prints_three_gold_shape_counts_over_264(tmp_path, monkeypatch):
    """Every gold-shape class, one count each, over the dev leg's own 264 — and
    the figures are the set's: the same classes recomputed independently from the
    frozen YAML in `_independent_counts` must agree item-for-item."""
    out = _run("--set", "v2")
    got = _counts(out, "dev")
    assert got["n"] == DEV_N == 264, got
    assert got[LA.DEFECTS[0]] == 70, got
    assert got[LA.DEFECTS[1]] == 0, got
    assert got[LA.DEFECTS[2]] == 26, got
    assert got == _independent_counts("dev"), (
        f"the printed counts and an independent count over the same frozen YAML "
        f"disagree: script {got} vs recount {_independent_counts('dev')}"
    )


def test_the_report_prints_the_word_list_and_the_word_count_rule_it_applied():
    """Clause 1's second half: the counts arrive with the definitions that made
    them, because two passes over one frozen YAML produced 61/2/24 and 73/0/35
    before anybody wrote down what a gold value is. A figure whose rule is not
    printed is not reproducible, whatever it says."""
    out = _run("--set", "v2")
    assert "definitions" in out
    assert "word count" in out and "len(gold_value.split())" in out, out
    assert f"exceeds {LA.WORD_CAP}" in out, out
    assert f"word list (n={len(LA.COMMON_WORDS)})" in out, out
    # The list is printed IN FULL, not named: a reader must be able to check a
    # single entry's effect without opening the source.
    listed = out.split("word list", 1)[1]
    # Entries checked at the head, middle and tail of the list, so a list that
    # stopped early or got truncated in the middle still fails here.
    for w in ("a,", "each,", "your"):
        assert (w in listed or listed.rstrip().endswith(w.split(",")[0])), f"{w!r} not listed"
    assert "every alias string in every accept.all_of group" in out, out


def test_the_word_cap_and_common_word_rules_flag_the_shapes_they_name(tmp_path):
    """The rules on constructed items, so a passing count over the real set cannot
    be a count of something else: a 4-word gold flags, a 3-word gold does not, a
    lone listed word as the only gold flags, and a none_of that shares the gold's
    one content word flags."""
    def q(golds, anti=()):
        return {"id": "x", "accept": {"all_of": [list(g) for g in golds],
                                      "none_of": list(anti)}}

    assert LA.classify(q([["ports 8182 and 9000"]]))["gold_over_word_cap"] is True
    assert LA.classify(q([["on port 8182"]]))["gold_over_word_cap"] is False  # exactly 3
    assert LA.classify(q([["any"], ["any"]]))["sole_gold_is_common_word"] is True
    # Two aliases for one answer is not two gold values worth of variety: still
    # ONE distinct gold string, and still a common word.
    assert LA.classify(q([["all", "all"]]))["sole_gold_is_common_word"] is True
    # ...and an item with a second, specific gold is no longer resting on the word.
    assert LA.classify(q([["all", "all"], ["OAuth 2.0"]]))["sole_gold_is_common_word"] is False
    assert LA.classify(q([["OAuth 2.0"]], anti=["the oauth flow"]))["none_of_shares_content_word"] \
        is True
    # Two golds whose answer substance genuinely differs from the anti value's,
    # sharing only LISTED words: pr-044 is about the answer's own substance
    # appearing inside the anti value, and "the"/"on" carry no substance, so
    # flagging these would bury the 26 real ones under stopword noise.
    assert LA.classify(q([["the relay"]], anti=["the"]))["none_of_shares_content_word"] \
        is False
    assert LA.classify(q([["on the 12th"]], anti=["on the"])
                       )["none_of_shares_content_word"] is False


# ── clause 2: the holdout leg is counted, never opened ───────────────────────

def test_the_holdout_leg_prints_three_counts_over_66_and_nothing_else(tmp_path):
    """66 is the holdout denominator and the three figures are 19 / 0 / 4 — and
    no holdout item id, gold string or prompt reaches stdout, with or without
    --list, which is the manifest's reserve_rule for anything reading that leg:
    "it reports aggregates, never per-question rows or ids"."""
    holdout = list((yaml.safe_load((SET / "holdout" / "questions.yaml").read_text())
                    or {}).get("questions") or [])
    assert len(holdout) == HOLDOUT_N == 66

    for argv in (["--set", "v2", "--holdout"], ["--set", "v2", "--holdout", "--list"]):
        out = _run(*argv)
        got = _counts(out, "holdout")
        assert got["n"] == HOLDOUT_N, got
        assert got[LA.DEFECTS[0]] == 19, got
        assert got[LA.DEFECTS[1]] == 0, got
        assert got[LA.DEFECTS[2]] == 4, got
        assert got == _independent_counts("holdout"), got

        # Structural: the holdout section holds three count rows and the
        # reserve-rule note and NOTHING else. This is the load-bearing check —
        # a leg with no room for prose cannot leak a gold, and it does not depend
        # on which strings happen to be distinctive.
        # Drop the rest of the section's own header line before parsing rows.
        body = out.split("## leg: holdout", 1)[1].split("\n", 1)[1]
        rows = [ln for ln in body.splitlines() if ln.strip()]
        counts_only = [ln for ln in rows if re.match(r"^\s+\S+ +\d+ / 66$", ln)]
        assert len(counts_only) == 3 == len(LA.DEFECTS), \
            f"expected exactly the three count rows, got:\n{rows}"
        notes = ("counts only", "counted only", "ids withheld:")
        for ln in rows:
            assert ln in counts_only or ln.lstrip().startswith(notes), \
                f"the holdout section carries something besides its counts: {ln!r}"

        # And per item, as a second net over the whole report: no id, no prompt.
        for q in holdout:
            assert q["id"] not in out, f"{q['id']} named in a holdout report"
            assert str(q["prompt"]) not in out, f"holdout prompt leaked for {q['id']}"
            # A multi-word gold is checkable as a whole string; a bare token is
            # not — `0` and `yes` occur inside the script's own counts and printed
            # word list, and reporting those as leaks would only train the next
            # reader to delete the check.
            for g in [v for grp in q["accept"]["all_of"] for v in grp]:
                if len(str(g).split()) > 1:
                    assert str(g) not in out, f"holdout gold {g!r} leaked ({q['id']})"


def test_without_the_holdout_flag_the_leg_is_reported_as_not_opened():
    """The default run says so out loud rather than reporting nothing, so a
    reader of the audit cannot mistake "absent" for "clean"."""
    out = _run("--set", "v2")
    assert "holdout — not opened" in out
    assert "66" not in out.split("## leg: holdout")[1]


# ── clause 3: a pure function of the frozen set ──────────────────────────────

def test_two_runs_on_the_frozen_v2_set_are_byte_identical():
    assert _run("--set", "v2", "--holdout") == _run("--set", "v2", "--holdout")
    # And the two ways of naming the set give the same bytes: an absolute path in
    # the header would make the same set differ between a checkout and a worktree.
    assert _run("--set", "v2", "--holdout") == \
        _run("--set", str(SET.relative_to(ROOT)), "--holdout")


def test_the_pass_writes_nothing_under_the_set(tmp_path):
    """Read-only means read-only: every file under the set keeps its exact size
    and mtime through a full two-leg run, and no new file appears anywhere —
    the questions are frozen by content hash, and a pass that touched them would
    change what `verify` reports as the set."""
    def fingerprint(root: Path):
        return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in sorted(root.rglob("*")) if p.is_file()}

    before = fingerprint(SET)
    assert before, "nothing to fingerprint: the set root has no files"
    out = _run("--set", "v2", "--holdout", "--list")
    assert fingerprint(SET) == before, "a file under the set changed size or mtime"
    assert "set_sha=16ec1ae14c046c61" in out, \
        "the report must name the frozen set it measured, in the engine's own form"


def test_the_manifest_record_and_the_questions_are_untouched_by_the_pass():
    """#2353 clause 5's mechanism, checked where a mechanical pass could have
    quietly improved the audited record: the manifest's label_audit still reports
    the 32 items v1's audit opened, and the set's own questions.yaml bytes are
    what the manifest hashes say they are."""
    man = json.loads((SET / "manifest.json").read_text())
    assert (man["label_audit"]["audited"], man["label_audit"]["clean"],
            man["label_audit"]["brittle"], man["label_audit"]["defective"]) == (32, 24, 8, 0)
    assert man["label_audit"]["derived_from"] == "v1"
    assert man["label_status"] == "pilot"
    assert man["set_sha"].startswith("16ec1ae14c046c61")
