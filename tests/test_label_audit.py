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

A fourth section, at the bottom, covers v3 — the frozen derivation of v2 in which
the 26 dev items this pass flags on v2 had their `accept.none_of` re-stated as the
generic alternative (#2508). Those nodes read the two sets against each other, so
v2's pinned readings are both the control and the premise: a repair that moved
gold, shortened a long value, touched an item outside the 26, or reached the
reserved holdout leg fails by comparison rather than by a claim.
"""
from __future__ import annotations

import io
import json
import random
import re
import subprocess
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
    assert f"set_sha={V2_SHA_PREFIX}" in out, \
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
    assert man["set_sha"].startswith(V2_SHA_PREFIX)


# ── #2486: the judged census, and the table that has to agree with it ──────────
#
# Everything above grades the MECHANICAL pass: gold shape, over 264 items, with no
# claim about whether a gold value is the right value. #2486 adds the judgements —
# every dev item read once against the v1 rubric and recorded in
# `eval/memory_eval/v2/label_verdicts.yaml` — and with them the failure mode #2356
# was filed for: a published table whose numbers stop being the numbers its own
# data says. These five nodes pin the census file, the table, the bounds, the
# frozen set, and the seeded blind re-read that gives the census an error rate.

VERDICTS = SET / "label_verdicts.yaml"
AUDIT = SET / "AUDIT.md"
V1_AUDIT = LA.SET_ROOT / "v1" / "AUDIT.md"
CATS = ["knowledge_update", "multi_session", "preference", "single_session", "temporal"]


def _dev_ids() -> list[str]:
    """The dev set's ids in file order, read the way the item states the check: the
    `^- id:` lines of dev/questions.yaml."""
    txt = (SET / "dev" / "questions.yaml").read_text()
    return [ln[len("- id:"):].strip() for ln in txt.splitlines() if ln.startswith("- id:")]


def _dev_categories() -> dict[str, str]:
    """id -> category as dev/questions.yaml itself records it. The census file carries
    a `category` per row for legibility only; the frozen set is what says which leg an
    item belongs to, and a row that moves one would move a published cell with it."""
    qs = yaml.safe_load((SET / "dev" / "questions.yaml").read_text())["questions"]
    return {q["id"]: q["category"] for q in qs}


def _verdicts() -> list[dict]:
    return yaml.safe_load(VERDICTS.read_text())


def _census_section() -> str:
    """AUDIT.md's census section, heading to next heading. Scoped because the same
    file publishes a second table whose `all dev` row holds the SHAPE counts over
    the same 264 items (the mechanical pass's 70 / 0 / 26), and reading the whole
    file would compare my column against that one."""
    txt = AUDIT.read_text()
    start = txt.index("## Judged verdicts")
    nxt = txt.find("\n## ", start + 1)
    return txt[start:] if nxt < 0 else txt[start:nxt]


def _table() -> dict[str, list[int]]:
    """The published per-category rows: category -> [n, clean, brittle, defective]."""
    rows = {}
    for ln in _census_section().splitlines():
        cells = [c.strip().strip("*").strip() for c in ln.split("|")]
        if len(cells) < 6 or cells[1] not in CATS + ["all dev"]:
            continue
        nums = cells[2:6]
        if not all(re.fullmatch(r"\d+", n) for n in nums):
            continue
        rows[cells[1]] = [int(n) for n in nums]
    return rows


def test_label_verdicts_carries_exactly_one_row_per_dev_item():
    """Clause 1: 264 rows, ids exactly the dev set's, each once, each judged.

    The set-root placement is part of the clause and is asserted, not assumed:
    `_leg_files` (run_memory_eval.py:343-345) rglobs *inside* each leg and
    `write_manifest` hashes that listing, so the same file one directory down
    would move `set_sha` and `load_set` would then refuse the set outright.
    """
    assert VERDICTS.exists(), f"{VERDICTS.name} must sit at the set root"
    assert VERDICTS.parent == SET, "the census file moved into a leg and would move set_sha"
    # Nothing looser than that: `SET` is the set root, so a parent match already
    # excludes every leg. A `"dev" not in VERDICTS.parts` check would test the
    # absolute path too, and a checkout under any directory named `dev` would fail
    # it for a reason that has nothing to do with the census.

    rows = _verdicts()
    ids = _dev_ids()
    assert len(ids) == DEV_N, f"the dev leg itself is no longer {DEV_N} items"
    got = [r["id"] for r in rows]
    assert len(rows) == DEV_N, f"{len(rows)} verdict rows for {DEV_N} dev items"
    assert len(set(got)) == DEV_N, f"duplicate rows: {sorted({i for i in got if got.count(i) > 1})}"
    assert set(got) == set(ids), (
        f"census ids differ from dev/questions.yaml: missing "
        f"{sorted(set(ids) - set(got))[:5]}, extra {sorted(set(got) - set(ids))[:5]}")
    for r in rows:
        assert r["verdict"] in ("clean", "brittle", "defective"), f"{r['id']}: {r['verdict']}"
        reason = " ".join(str(r["reason"]).split())
        assert reason, f"{r['id']} carries an empty reason"
        assert len(reason) <= 260, f"{r['id']}'s reason is not a one-liner ({len(reason)} chars)"
        assert r["category"] in CATS, f"{r['id']}: unknown category {r['category']}"
    by_cat = {c: [r["verdict"] for r in rows if r["category"] == c] for c in CATS}
    assert sum(len(v) for v in by_cat.values()) == DEV_N, "a row's category is outside the five"
    # The per-category counts the table publishes are only the leg's own counts if
    # each row sits in the leg the frozen set put that id in. Without this, a block
    # of rows re-labelled to another category keeps every total green and still
    # mis-describes the legs — the table-vs-data node below reads this same field.
    devcat = _dev_categories()
    assert set(devcat) == set(ids), "dev/questions.yaml itself stopped carrying a category per id"
    moved = sorted(r["id"] for r in rows if devcat.get(r["id"]) != r["category"])
    assert not moved, (
        f"census rows name a category the frozen set disagrees with: "
        f"{moved[:5]} of {len(moved)} moved rows")


def test_the_published_table_equals_the_counts_recomputed_from_the_verdicts():
    """Clause 2, the #2356 mode: the table in AUDIT.md is the verdicts file's own
    counts, recomputed here and compared cell by cell, per category and in total."""
    rows = _verdicts()
    table = _table()
    assert set(table) == set(CATS) | {"all dev"}, f"table rows found: {sorted(table)}"
    for c in CATS:
        vs = [r["verdict"] for r in rows if r["category"] == c]
        want = [len(vs), vs.count("clean"), vs.count("brittle"), vs.count("defective")]
        assert want[0] == sum(want[1:]), f"{c}: the row does not add up"
        assert table[c] == want, f"{c}: AUDIT.md says {table[c]}, label_verdicts says {want}"
    allv = [r["verdict"] for r in rows]
    total = [len(allv), allv.count("clean"), allv.count("brittle"), allv.count("defective")]
    assert table["all dev"] == total, f"all-dev row {table['all dev']} vs recomputed {total}"
    assert sum(table[c][0] for c in CATS) == total[0] == DEV_N


def test_the_none_of_the_264_heading_is_gone_and_the_bounds_are_the_shipped_function():
    """Clause 3: the section that said no verdicts existed is replaced, and the
    interval beside the census is what eval/stats.py's own wilson_ci returns for
    the defective count the table publishes — not a figure typed in beside it."""
    import stats

    txt = AUDIT.read_text()
    assert "## Judged verdicts: none of the 264" not in txt
    assert "Judged verdicts: all 264 dev items" in txt, "the census section is not where the old one was"
    assert "have not been made. Not for one item." not in txt

    defective = _table()["all dev"][3]
    lo, hi = stats.wilson_ci(defective, DEV_N)
    assert f"{defective}/{DEV_N}" in txt, "the census does not state its own denominator"
    assert f"[{lo * 100:.1f}%, {hi * 100:.1f}%]" in txt, \
        f"wilson_ci({defective}, {DEV_N}) = [{lo*100:.1f}%, {hi*100:.1f}%] is not the printed interval"
    assert f"stats.wilson_ci({defective}, {DEV_N})" in txt, "the bound's own call is not named"
    assert "eval/label_audit.py --set v2" not in txt.split("## Mechanical pass")[0].split(
        "## Judged verdicts")[1], "the census must not claim the shape pass produced it"
    # Scoped to the census section, not the file: the mechanical pass below also
    # names this path and the date 2026-10-07, so a whole-file `in` would let either
    # string satisfy the census from the wrong section's prose.
    census = _census_section()
    assert "eval/memory_eval/v2/label_verdicts.yaml" in census, "the table must name its data file"
    assert "2026-10-09" in census, "no date on the reading"


def test_the_frozen_set_is_untouched_by_the_census():
    """Clause 4: the census is a document beside a frozen set, not an edit to it.

    `verify` is the engine's own check of that claim — it re-hashes every file
    `_leg_files` finds inside both legs and compares the digest to the manifest —
    so a gold or session file edited after the freeze fails here on the runner's
    terms rather than on a diff. The manifest's audited record is asserted beside
    it: `audited: 32` with `derived_from: v1` is still what the gate reads, because
    moving v2 off pilot is a ruling that follows the census (#2486 out of scope,
    AUDIT.md).
    """
    out = subprocess.run([sys.executable, str(ROOT / "eval" / "run_memory_eval.py"), "verify",
                          "--set", str(SET)], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-800:]
    assert f"set_sha={V2_SHA_PREFIX}" in out.stdout, out.stdout.splitlines()[:1]
    assert "label_status=pilot" in out.stdout
    assert "audited 32 of 330" in out.stdout, out.stdout.splitlines()[1:2]
    man = json.loads((SET / "manifest.json").read_text())
    assert man["label_audit"]["audited"] == 32 and man["label_audit"]["derived_from"] == "v1"
    assert (man["label_audit"]["clean"], man["label_audit"]["brittle"],
            man["label_audit"]["defective"]) == (24, 8, 0)


def test_the_second_reader_pass_is_a_rerunnable_draw_with_its_disagreements_recorded():
    """Clause 5: the blind pass is reproducible from the seed the file states, its
    sample is a real dev-id set of the stated size, and the disagreement count it
    prints is the number of ids it lists.

    This is the census's only control, so the check is against the draw itself:
    `random.Random(seed).sample` is re-run here over the dev ids as sorted by the
    file's own recorded command, and the recorded list must be that draw.
    """
    txt = AUDIT.read_text()
    seed = int(re.search(r"Seed \*\*(\d+)\*\*", txt).group(1))
    size = int(re.search(r"sample \*\*(\d+)\*\* dev ids", txt).group(1))
    assert size >= 30, f"a {size}-item second read is below the floor the clause sets"
    block = re.search(r"in the order the draw produced them:\n\n```\n(.*?)```", txt, re.S).group(1)
    recorded = block.split()
    assert len(recorded) == size == len(set(recorded)), \
        f"recorded sample has {len(recorded)} ids for a stated {size}"
    ids = set(_dev_ids())
    assert set(recorded) <= ids, f"non-dev ids in the sample: {sorted(set(recorded) - ids)[:5]}"
    # Compared as a list, not a set: the file publishes the ids "in the order the
    # draw produced them", so the order is part of the claim and is what a later
    # reader re-runs the seed to confirm.
    assert recorded == random.Random(seed).sample(sorted(ids), size), \
        "the recorded sample is not the draw the stated seed produces, in that order"

    m = re.search(r"It disagreed on \*\*(\d+) of (\d+) = ([\d.]+)%\*\*", txt)
    assert m, "the disagreement count is not stated in the clause's own form"
    n_dis, drawn, rate = int(m.group(1)), int(m.group(2)), float(m.group(3))
    assert drawn == size == len(recorded), f"the pass says {drawn} items, the draw is {size}"
    listed = re.findall(
        r"^\| `(lme-[a-z]{2}-\d{3})` \| (\w+) \| (\w+) \| \*\*(\w+)\*\* \|", txt, re.M)
    assert len(listed) == n_dis, \
        f"the text says {n_dis} disagreements, the table lists {len(listed)}"
    assert {r[0] for r in listed} <= set(recorded), "a disagreement outside the drawn sample"
    assert abs(n_dis / size * 100 - rate) < 0.06, f"{n_dis}/{size} is not the printed {rate}%"
    for iid, first, blind, final in listed:
        v = [r["verdict"] for r in _verdicts() if r["id"] == iid][0]
        assert v == final, f"{iid}: AUDIT re-adjudicated to {final}, the verdict file holds {v}"
        assert first != blind, f"{iid} is listed as a disagreement but the two verdicts match"


_FENCE_RX = re.compile(r"^```[a-z]*\n(.*?)\n```", re.S | re.M)
_REJUDICATION_ROW_RX = re.compile(r"^\| `lme-[a-z]{2}-\d{3}` \|")
_REJUDICATION_RX = re.compile(
    r"^\| `(lme-[a-z]{2}-\d{3})` \| (\w+) \| (\w+) \| \*\*(\w+)\*\* \| (.*?)\s*\|\s*$")
_THREE = ("clean", "brittle", "defective")


def _census_command_code() -> str:
    """The code inside the reproduction block AUDIT.md's census section publishes.

    Selected by what it reads, not by position: the census section holds two fenced
    blocks, and the other one is the clause-5 sample draw.
    """
    blocks = [b for b in _FENCE_RX.findall(_census_section())
              if b.startswith(".venvs/lloyd/bin/python") and "label_verdicts" in b]
    assert len(blocks) == 1, \
        f"the census section must publish exactly one reproduction command, found {len(blocks)}"
    # The invocation line is asserted too, not just the code: it is the one part of
    # the reproduction a node cannot execute (a round worktree has no `.venvs/`), so
    # if it drifted from the repo's canonical test interpreter nothing else would
    # notice. Existence is deliberately NOT asserted — the gate runs tests from a
    # worktree where that path is absent by design.
    assert blocks[0].splitlines()[0] == '.venvs/lloyd/bin/python -c "', (
        f"the census block is not invoked through the repo's test venv: "
        f"{blocks[0].splitlines()[0]!r}")
    _, sep, rest = blocks[0].partition('-c "')
    assert sep, f"the published command is not a `-c \"…\"` one-liner: {blocks[0][:80]}"
    assert rest.rstrip().endswith('"'), "the published command's quoting is malformed"
    return rest.rstrip()[:-1]


def _printed_line(stdout: str, prefix: str) -> str:
    for ln in stdout.strip().splitlines():
        if ln.startswith(prefix):
            return ln
    raise AssertionError(f"the published command prints no {prefix!r} line:\n{stdout}")


def _four(line: str) -> list[int]:
    """The four count tokens of one printed line. By token shape, not position: the
    `all dev` label is two words and the incl-brittle line's is one, so a fixed
    offset reads the word `dev` as a count."""
    return [int(t) for t in line.split() if t.isdigit()][:4]


def test_the_census_command_reproduces_every_published_figure():
    """Clause 3's other half: the section ships the command that produced its own
    counts, and running it has to print the figures the section then states.

    #2356 is a two-direction failure. A table that drifts from the verdicts file is
    caught by the recomputation node above; this node catches the opposite, that the
    command a reader is told to run no longer produces the figures published beside
    it. Its arithmetic was the unpinned half: dropping the block's per-category
    lines, or nudging one printed count, left all thirteen pre-existing nodes green
    (measured 2026-10-09) — deleting the block outright was caught only incidentally,
    by an assertion about the section naming its data file, whose wording a real
    edit has no obligation to keep.

    The block's own interpreter line is the shell's, and a round worktree has no
    `.venvs/`; what is published and what is run here is the code, executed through
    the interpreter under test at the repo root, exactly as the block says to run it.
    """
    import stats

    proc = subprocess.run([sys.executable, "-c", _census_command_code()],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-500:]
    table = _table()
    for c in CATS:
        got = _four(_printed_line(proc.stdout, c))
        assert got == table[c], f"{c}: the published command prints {got}, the table says {table[c]}"

    all_line = _printed_line(proc.stdout, "all dev")
    assert _four(all_line) == table["all dev"], \
        f"all-dev: command {_four(all_line)} vs table {table['all dev']}"
    sec = _census_section()
    defective = table["all dev"][3]
    assert f"{defective}/{DEV_N}" in sec, "the section does not state the count the command printed"
    lo, hi = stats.wilson_ci(defective, DEV_N)
    assert f"[{lo * 100:.1f}%, {hi * 100:.1f}%]" in sec, (
        f"wilson_ci({defective}, {DEV_N}) is not the interval beside the census")

    plus_line = _printed_line(proc.stdout, "defective+brittle")
    plus = _four(plus_line)[0]
    assert plus == table["all dev"][2] + defective, (
        f"the command's incl-brittle figure {plus} is not brittle+defective "
        f"({table['all dev'][2]}+{defective})")
    assert f"{plus}/{DEV_N}" in sec, "the incl-brittle count the command prints is not published"
    lo2, hi2 = stats.wilson_ci(plus, DEV_N)
    assert f"[{lo2 * 100:.1f}%, {hi2 * 100:.1f}%]" in sec, (
        f"wilson_ci({plus}, {DEV_N}) is not the printed incl-brittle interval")


def test_the_table_percent_cells_are_its_own_count_cells_divided():
    """The last two columns of the census table are arithmetic on the first four, so
    they fail the same #2356 way a count does — and they were the only cells in the
    published table nothing read: the recomputation node compares columns 2-5."""
    table = _table()
    cells: dict[str, list[str]] = {}
    for ln in _census_section().splitlines():
        parts = [c.strip().strip("*").strip() for c in ln.split("|")]
        if len(parts) >= 8 and parts[1] in CATS + ["all dev"]:
            cells[parts[1]] = parts[6:8]
    assert set(cells) == set(table), f"percent cells parsed for {sorted(cells)}"
    for c, (def_pct, plus_pct) in cells.items():
        n, clean, brittle, defective = table[c]
        assert def_pct == f"{defective / n * 100:.1f}%", f"{c}: {def_pct}% vs {defective}/{n}"
        assert plus_pct == f"{(brittle + defective) / n * 100:.1f}%", (
            f"{c}: +brittle {plus_pct}% vs {(brittle + defective)}/{n}")


def test_every_re_adjudication_row_states_a_why_and_names_only_rubric_verdicts():
    """Clause 5's other half: each disagreement is re-adjudicated in prose, not just
    listed, and the three verdict columns are verdicts the rubric defines.

    The sample was already pinned to the seed's draw, the disagreement count to the
    number of rows, and each settled verdict to the verdicts file. What nothing read
    was the `why` cell: emptying one out of the published table left this whole file
    green until this node existed (measured, 2026-10-09 — the sibling node counts
    rows by a pattern that does not reach the fifth column, so a row stripped of its
    reason is a row still counted and no longer explained). Rows are therefore
    selected by their leading id and then required to match all five columns whole,
    so a truncated cell fails instead of dropping out of the parse and being
    silently uncounted — the shape of a guard that reads its own missing input.

    The `first` column is the ONLY record of what the census said before the blind
    re-read — the verdicts file holds the settled verdict — so its legal values are
    asserted here too.
    """
    lines = [ln for ln in AUDIT.read_text().splitlines() if _REJUDICATION_ROW_RX.match(ln)]
    assert lines, "AUDIT.md publishes no re-adjudication rows"
    for ln in lines:
        m = _REJUDICATION_RX.match(ln)
        assert m, f"a re-adjudication row is missing one of the five published columns: {ln[:100]!r}"
        iid, first, blind, final, why = m.groups()
        assert first in _THREE, f"{iid}: 'first' holds {first!r}, not a rubric verdict"
        assert blind in _THREE, f"{iid}: the blind verdict {blind!r} is not a rubric verdict"
        assert final in _THREE, f"{iid}: settled verdict {final!r} is not a rubric verdict"
        reason = " ".join(why.split())
        assert len(reason) >= 40, f"{iid}'s re-adjudication states no reason ({why!r})"
        assert not re.match(r"^(?:see above|ditto|n/?a)\b", reason, re.I), \
            f"{iid}'s re-adjudication defers its reason to another row"


def test_the_census_states_v1s_sample_size_as_v1s_own_table_does():
    """The census spends a paragraph on how its defect rate compares with v1's
    reading, so the sample size it cites for v1 is part of the published claim — and
    it was wrong in three places: the section said "25 sampled items" and "v1's
    25-item sample", the verdicts file said "v1's 25 verdicts", while the same
    section elsewhere says "its 35 items". The 25 is v1's sampling SEED,
    `random.Random(25)`, read as a count.

    Not asserted as a literal: the figure is derived from v1's own total row, which
    has to be internally consistent (one denominator, three numerators that sum to
    it) before it is required of both v2 files.
    """
    v1_lines = V1_AUDIT.read_text().splitlines()
    total = [ln for ln in v1_lines if ln.startswith("| **total**")]
    assert len(total) == 1, f"v1's audit publishes {len(total)} total rows"
    pairs = [(int(a), int(b)) for a, b in re.findall(r"\*\*(\d+) / (\d+)\*\*", total[0])]
    assert len(pairs) == 3, f"v1's total row is not three verdict cells: {total[0]!r}"
    denoms = {b for _, b in pairs}
    assert len(denoms) == 1, f"v1's own denominators disagree: {sorted(denoms)}"
    n = denoms.pop()
    assert sum(a for a, _ in pairs) == n, (
        f"v1's verdict counts sum to {sum(a for a, _ in pairs)}, not its denominator {n}")

    census = _census_section()
    assert f"v1's audit covered {n} sampled items" in census, (
        f"the census does not cite v1's own {n}-item sample: "
        f"{re.findall(r'v1[^.\\n]{0,45}items', census)[:2]}")
    assert f"v1's {n}-item sample" in census, f"the comparison paragraph names no {n}-item sample"
    assert "25 sampled items" not in census and "25-item sample" not in census, \
        "v1's sampling seed is not its sample size"
    header = VERDICTS.read_text().split("\n# One row per dev item")[0]
    assert f"v1's {n}" in header, "the verdicts file cites a v1 sample size v1's table contradicts"
    assert "v1's 25" not in header, "the verdicts file repeats v1's seed as its count"


def test_the_census_quotes_v1s_numerators_and_intervals_from_v1s_own_table():
    """Clause 3's other half, named by SM_20261010_183551's first review: the census
    compares its own defect rate with v1's twice over, and until this node only
    v1's DENOMINATOR was derived. The numerators (3 defective, 8 brittle, and the 11
    the audit's own brittle-plus-defective convention makes of them), the two
    percentages (8.6% and 31.4%) and all four interval bounds were literals, so
    editing v1's table would have left the comparison green while it drifted.

    Nothing is hardcoded here either: the column order comes from v1's own header
    row, the counts from its `**x / n**` total row, and both intervals from
    eval/stats.py's shipped `wilson_ci` — the same call the census's own bounds node
    is pinned to, so a v1 interval typed in beside the table fails for the same
    reason a v2 one would.
    """
    import stats

    v1_lines = V1_AUDIT.read_text().splitlines()
    header = [ln for ln in v1_lines
              if [c.strip() for c in ln.strip().strip("|").split("|")]
              == ["category", "clean", "brittle", "defective"]]
    assert len(header) == 1, f"v1's table does not publish one clean header row: {header!r}"
    total = [ln for ln in v1_lines if ln.startswith("| **total**")]
    assert len(total) == 1, f"v1's audit publishes {len(total)} total rows"
    pairs = [(int(a), int(b)) for a, b in re.findall(r"\*\*(\d+) / (\d+)\*\*", total[0])]
    assert len(pairs) == 3, f"v1's total row is not three verdict cells: {total[0]!r}"
    denoms = {b for _, b in pairs}
    assert len(denoms) == 1, f"v1's own denominators disagree: {sorted(denoms)}"
    n = denoms.pop()
    # Order is v1's header row's, asserted above, not this file's guess.
    clean, brittle, defective = (a for a, _ in pairs)
    assert clean + brittle + defective == n, (
        f"v1's verdict cells sum to {clean + brittle + defective}, not its denominator {n}")

    census = _census_section()
    both = defective + brittle          # v1's own "labels that under-report" figure
    for want in (f"{defective} of {n}",          # "v1's own reading of its 35 items was 3 of 35"
                 f"{brittle} brittle",           # "its 3 defective plus its 8 brittle"
                 f"{both}/{n}",                  # "was 11/35 = 31.4%"
                 f"{defective / n * 100:.1f}%",  # 8.6%
                 f"{both / n * 100:.1f}%"):      # 31.4%
        assert want in census, f"the census does not quote v1's {want!r} as v1's table has it"
    lo, hi = stats.wilson_ci(defective, n)
    assert f"[{lo * 100:.1f}%, {hi * 100:.1f}%]" in census, (
        f"wilson_ci({defective}, {n}) = [{lo * 100:.1f}%, {hi * 100:.1f}%] is not the v1 "
        f"defect interval the census prints")
    lo2, hi2 = stats.wilson_ci(both, n)
    assert f"[{lo2 * 100:.1f}%, {hi2 * 100:.1f}%]" in census, (
        f"wilson_ci({both}, {n}) = [{lo2 * 100:.1f}%, {hi2 * 100:.1f}%] is not the v1 "
        f"defective-plus-brittle interval the census prints")
    assert f"{clean} / {n}" in census, \
        f"v1's clean count ({clean} of {n}) is quoted in no form the table can be read from"


# ───────────────────── v3: the anti-overlap repair (#2508) ────────────────────
# v3 is v2 with one thing changed: the 26 dev items whose `accept.none_of` value
# shared a content word with their own `accept.all_of` gold — the `pr-044` class
# this file's mechanical pass counts as `none_of_shares_content_word`, which read
# 26 / 264 on v2's dev leg. The nodes below are the five clauses of #2508, and
# they read the two frozen sets against each other rather than against a number
# deposited in prose: v2 is the control, so a repair that quietly re-golded an
# item, shortened a long gold, or moved v2's digest fails here.
V3 = LA.SET_ROOT / "v3"
#: v2's frozen digest, spelled in full because `verify` prints only its first 16
#: characters, and a 16-character digest of CONTENT is not a git object: the gate's
#: sha rail asks `git cat-file -t` of every 7-40-character hex token a review note
#: carries and voids the review over a set_sha prefix it took for a commit
#: (`SM_20261010_204036`, first gate). Pinning all 64 and slicing at the use site
#: states the stronger fact and leaves no bare 16-hex token for that rail to trip
#: on; v2's digest is unchanged by this round.
V2_SET_SHA = "16ec1ae14c046c614428c14d87918734ce3e0cc8530b7d13f8f885120651a079"
V2_SHA_PREFIX = V2_SET_SHA[:16]
# The repaired count is a LITERAL at each site that asserts it (26), never a
# constant declared in this file: an assertion that compares against `REPAIRS_N`
# here moves whenever someone edits `REPAIRS_N`, so it pins this file's prose and
# not the shipped pass. What pins the number is the cross-check beside each
# literal — the count `label_audit.py` prints for v2, and the id set
# `LA.classify` flags over v2's frozen YAML — so a repair that fixed a different
# number of items fails against the tool's own reading (#2508's review raised
# this twice, as an advisory, on the constant-mirroring assert).


def _leg_questions(root: Path, leg: str) -> dict[str, dict]:
    return {q["id"]: q for q in yaml.safe_load(
        (root / leg / "questions.yaml").read_text())["questions"]}


def _repairs() -> list[dict]:
    """The repair record at v3's set root: id, the v2 value, the v3 value(s)."""
    return yaml.safe_load((V3 / "anti_repairs.yaml").read_text())


def _verify(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "eval" / "run_memory_eval.py"),
                           "verify", "--set", str(root)],
                          capture_output=True, text=True, cwd=ROOT)


def test_v3_is_a_frozen_derivation_of_v2_with_its_own_digest():
    """Clause 1: v3 carries both legs and its own manifest, and verifies.

    The boundary is the freeze itself: `write_manifest` computed v3's `set_sha`
    from the bytes on disk and `load_set` re-hashes them and refuses a mismatch,
    so `verify` exiting 0 is the engine's own statement that the manifest belongs
    to these bytes — a manifest pasted from v2 over v3's legs (or legs edited
    after the fact) fails on the runner's terms, not on a diff. The digest must
    DIFFER from v2's: identical digests would mean the 26 anti values were never
    written, which is the failure this item exists to prevent.
    """
    out = _verify(V3)
    assert out.returncode == 0, out.stderr[-800:]
    head = out.stdout.splitlines()[0]
    assert "v=v3" in head, head
    assert f"set_sha={V2_SHA_PREFIX}" not in head, \
        f"v3 verifies at v2's digest, so the repair is not in the hashed bytes: {head}"
    m = re.search(r"set_sha=([0-9a-f]{16})", head)
    assert m and len(m.group(1)) == 16, head
    assert "label_status=pilot" in head, head
    assert f"dev={DEV_N} holdout={HOLDOUT_N}" in head, head

    man = json.loads((V3 / "manifest.json").read_text())
    assert man["version"] == "v3"
    # derived_from is NOT a manifest top-level key: write_manifest takes
    # version/generator_model/notes/label_status/label_audit and nothing else, and
    # both label keys sit outside the digest `body` — which is why retiring
    # derived_from moves no set_sha (#2486's owed entry, #2508 clause 1).
    assert "derived_from" not in man, sorted(man)
    assert man["label_audit"]["derived_from"] == "v2", man["label_audit"]
    assert man["label_status"] == "pilot"
    assert len(_leg_questions(V3, "dev")) == DEV_N and len(_leg_questions(V3, "holdout")) == HOLDOUT_N
    for leg in ("dev", "holdout"):
        head_line = (V3 / leg / "questions.yaml").read_text().split("\n")[0]
        assert head_line == "version: v3", \
            f"{leg}'s version line is {head_line!r}: load_set refuses a copy that " \
            f"disagrees with its directory, so a byte-copy of v2 is not a v3"


def test_v3_dev_leg_reports_zero_anti_values_overlapping_their_own_gold():
    """Clause 2: the mechanical pass reads `none_of_shares_content_word 0 / 264`
    on v3's dev leg, and v2's leg still reads 26.

    Recomputed independently of the report, because a printed 0 is exactly what a
    pass that silently skipped the rule would also print: the same tokens are
    re-derived here from v3's frozen YAML with the definitions the report prints.
    The holdout leg is explicitly out of scope (#2508 repairs dev only) and its
    count is pinned so that a "fix" reaching into the reserved split shows up.
    """
    out = _run("--set", "v3", "--holdout")
    dev = _counts(out, "dev")
    assert dev["n"] == DEV_N, dev
    assert dev["none_of_shares_content_word"] == 0, \
        f"v3's dev leg still flags {dev['none_of_shares_content_word']} anti/gold overlaps"
    hold = _counts(out, "holdout")
    assert hold["none_of_shares_content_word"] == 4, \
        f"the holdout leg is out of this round's scope and must stay v2's 4, got {hold}"

    flagged = [i for i, q in _leg_questions(V3, "dev").items()
               if LA.classify(q)["none_of_shares_content_word"]]
    assert flagged == [], flagged
    v2_dev = _counts(_run("--set", "v2", "--holdout"), "dev")
    assert v2_dev["none_of_shares_content_word"] == 26, \
        f"v2's control reading is {v2_dev['none_of_shares_content_word']}, not the 26 " \
        f"this item's premise names — the repair is measured against that reading"


def test_v3_repaired_exactly_the_flagged_items_and_only_their_anti_values():
    """Clause 3: the 26 items differ ONLY in `accept.none_of`; the other 238 are
    byte-equal in parsed form; no item was added or dropped.

    That is the clause's two halves at once. "Restates anti values" — every gold
    group identical, and the long-gold population unchanged, so nobody bought the
    zero by shortening gold (v2's 70 long-gold items are v3's 70, item part 6).
    "Never rewrites gold" — no dev item outside the 26 differs at all, which also
    forbids repairing an item the pass cannot name: the three remaining anti-shape
    defects (`lme-pr-002`, `lme-pr-017`, `lme-pr-043`) collide semantically, not by
    token, so `--list` never named them and this round leaves them untouched.
    """
    v2_dev, v3_dev = _leg_questions(SET, "dev"), _leg_questions(V3, "dev")
    assert v2_dev.keys() == v3_dev.keys(), \
        f"items added/dropped: {sorted(set(v2_dev) ^ set(v3_dev))[:5]}"
    repaired = {r["id"] for r in _repairs()}
    assert len(repaired) == len(_repairs()) == 26, \
        f"the repair record holds {len(_repairs())} rows (deduped to {len(repaired)}), and " \
        f"v2's own pass flags 26 dev items — the record has to name exactly those"
    flagged_v2 = {i for i, q in v2_dev.items() if LA.classify(q)["none_of_shares_content_word"]}
    assert repaired == flagged_v2, \
        f"the repair record and what v2's pass flag disagree: {sorted(repaired ^ flagged_v2)}"

    for i, q2 in v2_dev.items():
        q3 = v3_dev[i]
        if i not in repaired:
            assert q3 == q2, f"an item outside the 26 repaired ones differs: {i}"
            continue
        assert q3["accept"]["all_of"] == q2["accept"]["all_of"], \
            f"{i}: gold changed — the repair restates anti values, it never rewrites gold"
        assert q3["accept"]["none_of"] != q2["accept"]["none_of"], f"{i}: anti unchanged"
        assert {k: v for k, v in q3.items() if k != "accept"} == \
               {k: v for k, v in q2.items() if k != "accept"}, f"{i}: something besides accept moved"
    long2 = {i for i, q in v2_dev.items() if LA.classify(q)["gold_over_word_cap"]}
    long3 = {i for i, q in v3_dev.items() if LA.classify(q)["gold_over_word_cap"]}
    assert long2 == long3 and len(long2) == 70, \
        f"the long-gold population moved ({len(long2)} -> {len(long3)}): gold was shortened"


def test_v2_is_byte_identical_and_still_verifies_at_its_frozen_sha():
    """Clause 4: the round that created v3 left v2 exactly as it found it.

    Two independent witnesses, because the failure this guards is subtle: v2 and
    v3 are *both* hashed sets, so an edit that landed in v2 by mistake still
    verifies (its manifest is recomputed by nobody) — only the tree comparison
    catches it. `git diff --stat` over the v2 directory is the tree's own answer;
    `verify` is the digest's. Both must be quiet, and v2's own AUDIT.md keeps the
    `26 / 264` line it was written with, which is true of v2 and is why v3 exists.
    """
    d = subprocess.run(["git", "-C", str(ROOT), "diff", "--stat", "HEAD", "--",
                        "eval/memory_eval/v2"], capture_output=True, text=True)
    assert d.returncode == 0, d.stderr[-400:]
    assert d.stdout.strip() == "", f"v2 changed on disk:\n{d.stdout}"
    out = _verify(SET)
    assert out.returncode == 0, out.stderr[-800:]
    assert f"set_sha={V2_SHA_PREFIX}" in out.stdout, out.stdout.splitlines()[0]
    assert "audited 32 of 330" in out.stdout, "v2's own audit row moved too"
    assert _counts(_run("--set", "v2"), "dev")["none_of_shares_content_word"] \
        == 26, "v2's AUDIT-era 26 / 264 reading is no longer what its bytes produce"


def test_v3_label_audit_is_a_recomputation_and_not_v2s_copied_row():
    """Clause 5: v3's audit row is computed over v3's repaired gold, and
    `label_quality` says `gold_eligible: true` from THOSE counts while the status
    stays `pilot`.

    The counts are re-tallied here from `label_verdicts.yaml`, and the ceiling
    decision is recomputed with the same `wilson_ci` the runner uses — an
    `eligible: true` copied in from prose, or a row whose upper bound no longer
    clears 0.10, fails. v2's row (32 / 24 / 8 / 0, upper 10.7%) is asserted
    beside it as the contrast: it said `false` because 32 items cannot bound a
    rate at 10% however clean they are, so the difference below is coverage over
    the dev census, not a friendlier opinion. `label_status` staying `pilot` is
    the owed-check ruling #2508 does not make for it (#2486 owed entry 5).
    """
    v3man = json.loads((V3 / "manifest.json").read_text())["label_audit"]
    v2man = json.loads((SET / "manifest.json").read_text())["label_audit"]
    assert (v2man["audited"], v2man["clean"], v2man["brittle"], v2man["defective"]) \
        == (32, 24, 8, 0), v2man
    got = (v3man["audited"], v3man["clean"], v3man["brittle"], v3man["defective"])
    assert got != (32, 24, 8, 0), "v3 carries v2's row verbatim"
    rows = yaml.safe_load((V3 / "label_verdicts.yaml").read_text())
    assert len(rows) == DEV_N, f"the census copy has {len(rows)} rows, not {DEV_N}"
    tally = {v: sum(1 for r in rows if r["verdict"] == v)
             for v in ("clean", "brittle", "defective")}
    assert got == (len(rows), tally["clean"], tally["brittle"], tally["defective"]), \
        f"manifest row {got} is not what the census says {tally}"
    assert got[0] == got[1] + got[2] + got[3] == 264, got

    import stats as eval_stats                      # eval/stats.py
    lo, hi = eval_stats.wilson_ci(tally["defective"], len(rows))
    assert hi < 0.10, f"defect upper {hi:.4f} does not clear the 0.10 ceiling"
    sys.path.insert(0, str(ROOT / "eval"))
    import run_memory_eval as RME
    lq = RME.label_quality(RME.load_set(V3, view="all"))
    assert lq["gold_eligible"] is True, lq
    assert lq["label_status"] == "pilot", \
        "publishing v3 at gold is owed-check's ruling, not this round's"
    assert lq["audited"] == 264 and lq["n_items"] == DEV_N + HOLDOUT_N, lq
    assert (lq["clean"], lq["brittle"], lq["defective"]) == \
        (tally["clean"], tally["brittle"], tally["defective"]), lq
    assert abs(lq["defect_ci95"][1] - hi) < 1e-4, f"{lq['defect_ci95']} vs recomputed {hi:.4f}"

    method = v3man["method"].lower()
    assert "re-deriv" in method and "v3" in method, v3man["method"]
    assert "not v1's spot audit" in method, \
        "the method must say what it is NOT: v2's row was v1's spot audit, copied"
    assert "holdout" in method and "not read" in method.replace("not opened", "not read"), \
        "a 264-of-330 census must say the reserved leg was not read"
    assert v3man["source"].startswith("eval/memory_eval/v3/"), v3man["source"]


def test_v3_records_sit_at_the_set_root_where_the_digest_cannot_see_them():
    """Clause 5's other half, across the hashing boundary.

    `write_manifest` hashes what `_leg_files` rglobs *inside* a leg, so the same
    census placed under `dev/` would move `set_sha` and `load_set` would then
    refuse v3 with `manifest edited after freezing` — the failure v2 recorded for
    its own verdict file (`v2/AUDIT.md:16-23`). So this asserts both edges: the
    records exist at the root, and neither leg gained or lost a hashed file
    relative to v2, which is what makes v3's digest difference attributable to
    the 26 anti values and nothing else.
    """
    for name in ("label_verdicts.yaml", "anti_repairs.yaml", "AUDIT.md"):
        assert (V3 / name).is_file(), f"{name} missing from the v3 set root"
        assert not (V3 / "dev" / name).exists() and not (V3 / "holdout" / name).exists(), \
            f"{name} under a leg would move set_sha"
    sys.path.insert(0, str(ROOT / "eval"))
    import run_memory_eval as RME
    for leg in ("dev", "holdout"):
        f2 = sorted(str(p.relative_to(SET / leg)) for p in RME._leg_files(SET, leg))
        f3 = sorted(str(p.relative_to(V3 / leg)) for p in RME._leg_files(V3, leg))
        assert f2 == f3, f"{leg}: the hashed file set changed {set(f2) ^ set(f3)}"


def test_v3_audit_and_repair_record_quote_the_values_the_frozen_set_carries():
    """The document half of the repair, pinned against the data it documents.

    `anti_repairs.yaml` and `AUDIT.md` are the only reading a later reviewer has
    of *why* each value was chosen, and both are prose a set edit can orphan: this
    checks every recorded row against the item it names — the v2 value really was
    v2's, the v3 value really is v3's, and the v2 value is gone — plus the two
    claims the whole round rests on, that the collision named is a real token
    share on v2 and that no repaired value shares one on v3.
    """
    v2_dev, v3_dev = _leg_questions(SET, "dev"), _leg_questions(V3, "dev")
    for r in _repairs():
        i = r["id"]
        assert i in v2_dev and i in v3_dev, f"{i} is not a dev id"
        assert r["v2_anti"] in v2_dev[i]["accept"]["none_of"], \
            f"{i}: {r['v2_anti']!r} was not v2's anti value"
        assert r["v2_anti"] not in v3_dev[i]["accept"]["none_of"], \
            f"{i}: the flagged value survived into v3"
        for new in r["v3_anti"]:
            assert new in v3_dev[i]["accept"]["none_of"], f"{i}: {new!r} is not in v3's set"
        gold = {t for a in v2_dev[i]["accept"]["all_of"] for t in LA._tokens(a)} - set(LA.COMMON_WORDS)
        shared = {t for a in v2_dev[i]["accept"]["none_of"] for t in LA._tokens(a)} & gold
        assert shared, f"{i}: v2's flagged anti shares no content word, so the record lies"
        now = {t for a in v3_dev[i]["accept"]["none_of"] for t in LA._tokens(a)} - set(LA.COMMON_WORDS)
        assert not (now & gold), f"{i}: the v3 value still shares a content word"

    txt = (V3 / "AUDIT.md").read_text()
    assert "## The 26 repairs" in txt
    for r in _repairs():
        assert f"| {r['id']} |" in txt, f"{r['id']} is in anti_repairs.yaml but not the AUDIT table"
    for want in ("0 / 264", "26 / 264", "4 / 66", "70 / 264",
                 V2_SET_SHA, "190", "63", "11", "gold_eligible: true"):
        assert want in txt, f"the v3 AUDIT does not state {want!r}"
