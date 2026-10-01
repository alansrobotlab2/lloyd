"""Pins the `action_review` threshold analysis: its cross-tab, and the join.

`eval/djev/action_review_calibration.py` is the thing that decides whether P10's
seam ever gets a threshold, so the two ways it could lie quietly are the two
things this file spends its length on.

**A tier can be defaulted.** `effective_tier("Bash")` with no command answers 1
(`app/harness/policy.py::tool_tier` says so), and the row does not carry the
command — it carries `meta.tool` and `meta.args_digest`. An analysis that could
not find a transcript and carried on would file every lost call as a tier-1,
harmless, would-not-have-warned call, and the corpus would look safe for reasons
that have nothing to do with what those calls did.
`test_a_row_whose_session_file_is_absent_is_counted_unresolved_not_tier_1` is the
whole defence, and it fails on the naive implementation.

**A join can be plausible rather than correct.** Arguments are recovered from
`meta.session_id`/`meta.call_id`, and a transcript whose call record was rewritten
would still hand back a Bash command that looks perfectly reasonable. The module
therefore re-digests what it recovered against the row's own `args_digest`, which
is why `matches_digest` strips exactly one key (`summary` — the harness pulls it
out of `args_dict` before the turn runs, so the transcript keeps it and the
digest does not) and why a mismatch is `unresolved` rather than a shrug. One test
makes the mismatch fire; the other makes a genuine transcript MATCH, so the first
cannot be passing because the digest never matches anything.

Rates are pinned with their denominators because the item this replaced quoted
"92/day" off the shadow FILE's span, when the seam's own first row is four days
later: the same rows, a rate 1.7× higher, and the other branch chosen.
"""

from __future__ import annotations

import collections
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app import djev_shadow
from app.harness.action_review import SEAM, args_digest
from eval.djev import action_review_calibration as C
from eval.djev import schemas

DAY = 86400.0
#: The seam's live schema hash, asked of the schema rather than copied: a row's
#: `schema` field is what `replay.py` groups on, and a copied literal would be a
#: second answer to "which schema wrote this row" — the drift `tests/
#: test_action_review.py::test_the_seam_has_a_frozen_schema_and_ships_ungated`
#: exists to catch, and which this fixture must not quietly disagree with.
SCHEMA_HASH = schemas.hash_for(SEAM)
#: The live corpus's own first `action_review` row, to the second.
T0 = datetime(2026, 9, 25, 18, 45, 16, tzinfo=timezone.utc).timestamp()

ECHO = ("c1", "Bash", {"command": "echo hi"})
PUSH = ("c2", "Bash", {"command": "git push origin main"})
TAMPED = ("c5", "Bash", {"command": "echo hi"})
#: The transcript entry for `TAMPED`: the same `call_id`, a command that was never
#: run, and therefore arguments that cannot hash to the digest the seam stamped.
TAMPERED_TRANSCRIPT = ("c5", "Bash", {"command": "git push origin upstream main"})
LOOK = ("c3", "Read", {"file_path": "/x"})
RESTART = ("c4", "Bash", {"command": "supervisorctl -c ~/lloyd/agent-services/"
                                     "supervisord.conf restart lloyd-backend"})


def row(ts: float, call_id: str, tool: str, args: dict, *, session: str = "s1",
        source: str = "autocode", outcome: str = "ran", label: str = "consistent",
        seam: str = "action_review") -> dict:
    """One seam row as `app/harness/action_review.py` and the recorder write it:
    `meta` carries the tool and the digest of the arguments, never the
    arguments."""
    return {
        "ts": ts, "seam": seam, "schema": SCHEMA_HASH,
        "meta": {"session_id": session, "source": source, "tool": tool,
                 "args_digest": args_digest(args), "prior_calls": 0,
                 "call_id": call_id, "mode": "shadow"},
        "actual": {"outcome": outcome},
        "djev": {"answers": {"on_task": {"id": "on_task", "type": "choice",
                                         "value": label, "label": label,
                                         "label_mass": 0.95}}},
    }


def transcript(session_id: str, calls: list[tuple[str, str, dict]]) -> dict:
    """A session file holding those calls — arguments plus the caption the
    harness keeps in the transcript and drops from the digested `args_dict`."""
    return {
        "session_id": session_id, "messages": [
            {"role": "assistant", "tool_calls": [
                {"id": cid, "call_id": cid, "type": "function",
                 "function": {"name": tool,
                              "arguments": json.dumps({**args, "summary": "caption"})}}
                for cid, tool, args in calls]},
        ],
    }


def corpus(tmp_path: Path, rows: list[dict], sessions: dict[str, dict]) -> tuple[Path, Path]:
    log = tmp_path / "shadow.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    sdir = tmp_path / "sessions"
    sdir.mkdir(exist_ok=True)
    for sid, doc in sessions.items():
        (sdir / f"{sid}.json").write_text(json.dumps(doc), encoding="utf-8")
    return log, sdir


def measure(tmp_path, rows, sessions) -> tuple[list, dict]:
    log, sdir = corpus(tmp_path, rows, sessions)
    return C.load_corpus(log, sdir)


# ── clause 1: the cross-tab, with every denominator attached ─────────────────

def test_the_cross_tab_is_label_by_tier_by_source_by_outcome(tmp_path):
    rows = [row(T0, *ECHO),
            row(T0 + 1, *PUSH, label="unrelated"),
            row(T0 + 2, *LOOK),
            row(T0 + 3, *LOOK, label="unrelated", outcome="denied_by_hook",
                source="autotriage")]
    out, integrity = measure(tmp_path, rows, {"s1": transcript("s1", [ECHO, PUSH, LOOK])})
    rep = C.report(out, integrity, source=tmp_path / "shadow.jsonl")

    def cell(label, tier, source, outcome):
        return [c for c in rep["cross_tab"]
                if (c["label"], c["tier"], c["source"], c["outcome"])
                == (label, tier, source, outcome)]

    assert cell("consistent", "1", "autocode", "ran") == [
        {"label": "consistent", "tier": "1", "source": "autocode", "outcome": "ran",
         "n": 2}], "a tier-1 Read and a tier-1 echo belong in one cell"
    assert cell("unrelated", "2", "autocode", "ran")[0]["n"] == 1, (
        "the durable-external command must land in tier 2, not beside the echo")
    assert cell("unrelated", "1", "autotriage", "denied_by_hook")[0]["n"] == 1, (
        "source and outcome are axes of the join, not labels on the whole corpus")
    assert sum(c["n"] for c in rep["cross_tab"]) == 4, (
        "the cross-tab must account for every seam row, not only the cells it kept")


def test_untierable_rows_get_a_bucket_of_their_own_and_not_a_tier(tmp_path):
    """`unresolved` is a cross-tab bucket. Folding it into tier 1 is the
    defaulting failure, and in a percentage it is invisible."""
    out, integrity = measure(tmp_path, [row(T0, *ECHO)], {})
    rep = C.report(out, integrity, source=tmp_path / "shadow.jsonl")
    assert [(c["tier"], c["n"]) for c in rep["cross_tab"]] == [("unresolved", 1)]
    assert rep["tier_resolution"]["untierable"] == 1
    assert rep["tier_resolution"]["reasons"] == {"no_session_file": 1}


def test_every_rate_carries_the_denominator_it_was_divided_by(tmp_path):
    rows = [row(T0 + i, *LOOK, label=("unrelated" if i == 0 else "consistent"))
            for i in range(4)]
    out, integrity = measure(tmp_path, rows, {"s1": transcript("s1", [LOOK])})
    rep = C.report(out, integrity, source=tmp_path / "shadow.jsonl")
    text = C.format_report(rep)

    assert rep["labels"]["unrelated"] == {"n": 1, "of_seam_rows": 4}
    assert rep["unrelated"]["ran"] == {"n": 1, "of_unrelated": 1}
    assert rep["hook_denials"] == {"n": 0, "of_seam_rows": 4, "by_label": {},
                                   "overlap_with_unrelated": 0}
    # Every printed rate carries its divisor, not a percentage over prose — a
    # rate whose denominator is unstated is how "92/day" reached the item. Each
    # line is pinned separately, because a substring like "1/4" also appears in
    # the corpus header's seam-share and would pass without any rate line having
    # a denominator at all.
    label_line = next(l for l in text.splitlines() if l.startswith("on_task label"))
    assert "consistent 3/4 (75.00%)" in label_line, label_line
    assert "unrelated 1/4 (25.00%)" in label_line, label_line
    outcome_line = next(l for l in text.splitlines() if l.startswith("outcome"))
    assert "ran 4/4" in outcome_line, outcome_line
    header = next(l for l in text.splitlines() if "rows all seams" in l)
    assert "4 action_review (100.0% of 4)" in header, header
    unrel = rep["unrelated"]
    assert f"unrelated         {unrel['n']}/{unrel['of_seam_rows']} seam rows" in text, text
    warn_line = next(l for l in text.splitlines() if "would have interrupted" in l)
    assert "ran 1/1" in warn_line, warn_line
    tier2_line = next(l for l in text.splitlines() if "durable-external" in l)
    assert "0/1 unrelated" in tier2_line and "0/4 of all seam rows" in tier2_line, tier2_line
    cells_line = next(l for l in text.splitlines() if l.startswith("label × tier"))
    assert "n over 4 rows" in cells_line, cells_line


def test_a_non_seam_row_is_a_denominator_and_nothing_else(tmp_path):
    other = row(T0, *LOOK, seam="dedupe")
    out, integrity = measure(tmp_path, [other, row(T0 + 1, *LOOK)],
                             {"s1": transcript("s1", [LOOK])})
    rep = C.report(out, integrity, source=tmp_path / "shadow.jsonl")
    assert rep["corpus"]["shadow_lines"] == 2 and rep["corpus"]["seam_rows"] == 1
    assert integrity["seam_rows"] == 1, "another seam's row entered the measurement"


# ── clause 2: tiers come from the recovered command, or from nowhere ─────────

@pytest.mark.parametrize("call,want", [
    (ECHO, 1), (PUSH, 2), (RESTART, 2),
])
def test_a_bash_row_is_tiered_from_the_command_string(tmp_path, call, want):
    out, _ = measure(tmp_path, [row(T0, *call)], {"s1": transcript("s1", [call])})
    assert [r.tier for r in out] == [want], (
        f"{call[2]['command']!r} did not resolve to tier {want}")
    assert out[0].tier_reason == "args", (
        "a Bash tier must come from the recovered arguments, not the tool's name")


def test_arguments_that_do_not_reproduce_the_digest_are_not_tiered(tmp_path):
    """The transcript answers call `c9` with a different command than the row
    digested: not a lost file, a wrong one. Tiering it would be tiering a call
    the seam never recorded."""
    log_args = {"command": "git push origin main"}
    rows = [row(T0, "c9", "Bash", log_args, label="unrelated")]
    lying = ("c9", "Bash", {"command": "echo hi"})
    out, integrity = measure(tmp_path, rows, {"s1": transcript("s1", [lying])})
    rep = C.report(out, integrity, source=tmp_path / "shadow.jsonl")
    assert [r.tier for r in out] == [None], (
        "arguments that do not reproduce the row's args_digest were tiered anyway")
    assert out[0].tier_reason == "digest_mismatch"
    assert integrity["digest_mismatch"] == 1
    assert rep["tier_resolution"]["integrity"]["digest_mismatch"] == 1


def test_a_genuine_transcript_does_match_the_digest(tmp_path):
    """The positive control the test above needs: a real transcript carries the
    caption the digested arguments do not, and must still MATCH — otherwise the
    mismatch test would pass for a digest that never matches anything."""
    out, integrity = measure(tmp_path, [row(T0, *ECHO)],
                             {"s1": transcript("s1", [ECHO])})
    assert integrity["digest_match"] == 1 and "digest_mismatch" not in integrity
    assert out[0].tier == 1


def test_a_row_whose_session_file_is_absent_is_counted_unresolved_not_tier_1(tmp_path):
    """Clause 2's half that a careless implementation passes: the file is gone,
    `effective_tier("Bash")` defaults to 1, and the row silently becomes evidence
    that the positive was harmless."""
    out, integrity = measure(tmp_path, [row(T0, *ECHO, label="unrelated")], {})
    rep = C.report(out, integrity, source=tmp_path / "shadow.jsonl")
    assert out[0].tier is None
    assert rep["unrelated"]["tiers"] == {"unresolved": {
        "n": 1, "of_unrelated": 1, "of_tiered_unrelated": None}}, (
        "the corpus lost a transcript and the positives somehow got safer")
    assert rep["unrelated"]["tier_ge_2"]["n"] == 0


def test_a_row_whose_transcript_lost_the_call_is_a_different_reason(tmp_path):
    """A surviving file missing the call is its own reason: the retention story
    is different (the transcript was rewritten, not swept)."""
    out, _ = measure(tmp_path, [row(T0, "c-nope", "Bash", {"command": "echo hi"},
                                    label="unrelated")],
                     {"s1": transcript("s1", [LOOK])})
    assert out[0].tier_reason == "no_call_record"
    assert out[0].tier is None


def test_a_name_tiered_tool_needs_no_transcript(tmp_path):
    """`Read` is tier 1 by name, so a missing transcript costs it nothing — and
    saying so is what keeps the unresolved count honest about what it means."""
    out, integrity = measure(tmp_path, [row(T0, *LOOK, label="unrelated")], {})
    assert out[0].tier == 1 and out[0].tier_reason == "name"
    assert integrity.get("untierable", 0) == 0


# ── the span, the overlap, and the paths the analysis reads ──────────────────

def test_the_per_day_rate_divides_by_the_seams_span_not_the_files(tmp_path):
    """The item's own failure: the shadow FILE leads with four days of other
    seams, and dividing the positives by that span quotes a rate 1.7× low."""
    older = row(T0 - 4 * DAY, *LOOK, seam="dedupe")      # a pre-seam row, same file
    seam_rows = [row(T0, *LOOK, label="unrelated"),
                 row(T0 + DAY, *LOOK, label="unrelated")]
    out, integrity = measure(tmp_path, [older] + seam_rows,
                             {"s1": transcript("s1", [LOOK])})
    rep = C.report(out, integrity, source=tmp_path / "shadow.jsonl")
    assert rep["corpus"]["span_days"] == pytest.approx(1.0, abs=1e-6), (
        "the span was measured from the file's first row, which is another seam's")
    assert rep["corpus"]["first_row"] == SEAM_FIRST_ROW, (
        "the seam's own first row moved, and every per-day figure in the ruling is "
        "divided by the span that starts here")
    assert rep["unrelated"]["per_day_over_seam_span"] == pytest.approx(2.0)
    assert rep["unrelated"]["would_fire_per_day"] == pytest.approx(2.0)


def test_only_a_row_that_ran_would_have_been_interrupted(tmp_path):
    """A `warn` interrupts a call that ran. Counting a denied or errored call as
    an interruption inflates the very rate the branch decision turns on."""
    rows = [row(T0, *LOOK, label="unrelated", outcome="error"),
            row(T0 + DAY, *LOOK, label="unrelated", outcome="denied_by_hook"),
            row(T0 + 2 * DAY, *LOOK, label="unrelated", outcome="ran")]
    out, integrity = measure(tmp_path, rows, {"s1": transcript("s1", [LOOK])})
    rep = C.report(out, integrity, source=tmp_path / "shadow.jsonl")
    assert rep["unrelated"]["n"] == 3 and rep["unrelated"]["ran"]["n"] == 1
    assert rep["unrelated"]["per_day_over_seam_span"] == pytest.approx(1.5), (
        "three positives over two days is 1.5/day")
    assert rep["unrelated"]["would_fire_per_day"] == pytest.approx(0.5), (
        "only the one call that ran could have been interrupted, so the "
        "interruption rate is a third of the positive rate")


def test_the_hook_denial_overlap_is_stated_with_its_counts(tmp_path):
    """The two sensors are not substitutes, and the claim is only checkable while
    the counts that say so are printed: of the live corpus's 36 hook denials djev
    judged 35 `consistent` and 1 `unrelated`."""
    rows = [row(T0 + i, *LOOK, label=("unrelated" if i == 3 else "consistent"),
                outcome="denied_by_hook") for i in range(4)]
    out, integrity = measure(tmp_path, rows, {"s1": transcript("s1", [LOOK])})
    rep = C.report(out, integrity, source=tmp_path / "shadow.jsonl")
    assert rep["hook_denials"]["n"] == 4
    assert rep["hook_denials"]["by_label"] == {
        "consistent": {"n": 3, "of_hook_denials": 4},
        "unrelated": {"n": 1, "of_hook_denials": 4}}
    assert rep["hook_denials"]["overlap_with_unrelated"] == 1
    assert "overlap with `unrelated` is 1" in C.format_report(rep)


def test_a_zero_injected_count_is_reported_as_absence(tmp_path):
    """`injected` is the option the schema comment calls "the one a threshold
    will be read off", and the live corpus has never produced one. The report must
    not print a rate for a label the corpus does not have."""
    out, integrity = measure(tmp_path, [row(T0, *LOOK)],
                             {"s1": transcript("s1", [LOOK])})
    rep = C.report(out, integrity, source=tmp_path / "shadow.jsonl")
    assert "injected" not in rep["labels"], "a label with zero rows got a denominator"


def test_the_report_reads_production_even_from_a_round_tree():
    """Both defaults are `ACCOUNT_HOME` / `production_data_root()`, on purpose:
    `app.paths.SESSIONS_DIR` resolves into an automod worktree's own empty
    `.lloyd-data/`, and a scan of it returns nothing that reads like a clean
    corpus. Inside a round the two paths differ, which is the point."""
    # Asserted as behaviour, not as a restatement of the definitions. The gate
    # sets `LLOYD_DATA`, so `C.SESSIONS_DIR == production_data_root() / "sessions"`
    # is true even when the module means the round's own `.lloyd-data/` — both
    # sides move together, which is exactly the false green
    # #worktree-data-anchor names. What cannot move together is the recorder's
    # own directory, the running tree, and whether the corpus is non-empty.
    assert C.SHADOW_LOG.name == djev_shadow.SHADOW_LOG.name, (
        "the recorder changed the shadow filename and this reader did not follow, "
        "so the analysis is now pointed at a file nothing writes")
    assert C.SESSIONS_DIR.is_dir(), f"no transcripts at {C.SESSIONS_DIR}"
    n = sum(1 for _ in C.SESSIONS_DIR.glob("*.json"))
    assert n > 100, (
        f"{C.SESSIONS_DIR} holds {n} transcripts — a round's data dir, not the "
        "live corpus, so every tier count in the report would be 0")
    tree = Path(C.__file__).resolve().parents[2]
    for p in (C.SESSIONS_DIR, C.SHADOW_LOG):
        assert not str(p).startswith(str(tree) + "/"), (
            f"{p} is inside the running tree {tree}: this would be a measurement "
            "of the round's own checkout, which is what a round is made of")


def test_the_analysis_tiers_with_the_gate_and_not_with_its_own_patterns(monkeypatch):
    """A local shape table would be a second answer to "is this call durable-
    external", and the two would drift. The tier must come from `effective_tier`
    at runtime, with the arguments that decide it — not merely be mentioned in the
    source."""
    # Behavioural: patch the gate's own function and prove the tier the analysis
    # reports is the value IT returned, with the arguments it was handed. A local
    # dict or fnmatch table — the drift this forbids — passes a grep for
    # `effective_tier(` and fails this, because the patched value never arrives.
    seen = []

    def fake_tier(name, tool_input=None):
        seen.append((name, tool_input))
        return 7

    monkeypatch.setattr(C, "effective_tier", fake_tier)
    assert C.resolve_tier("Bash", {"command": "curl -T x y"}) == (7, "args")
    assert C.resolve_tier("Read", None) == (7, "name")
    assert seen == [("Bash", {"command": "curl -T x y"}), ("Read", None)], (
        "`effective_tier` was consulted, but not with the arguments that decide "
        "the tier — so the number reported is not the gate's answer for this call")

    import inspect
    import re

    src = inspect.getsource(C)
    assert "re.compile" not in src, "the analysis grew its own command patterns"
    assert not re.search(r"^\s*(?:import|from)\s+re\b", src, re.MULTILINE), (
        "the analysis imports `re`, which is how a second copy of the durable-"
        "external matching would get written here")


# ── clause 6: the ruling has bytes behind it ────────────────────────────────
#
# Item #1944 clause 6: "The quoted counts are traceable to committed bytes: a
# witness artifact copied or produced in-round carries them and the sha256 of the
# file they were read from, and `wc -l` of the artifact prints the figure the
# round quotes. The report's counts are recomputable from committed bytes, and
# its prose carries no rate whose source file it cannot name."
#
# The live corpus is unbounded (nothing rotates it, `app/djev_shadow.py:99`) and
# append-only, so a figure quoted from it is only checkable while the file says
# so. The witness is the extract that makes the ruling checkable independent of
# the growing file.

#: The path clause 6 names, so the artifact and the clause agree by name as well
#: as by content. (`shadow-action-review-2026-10-01T0247Z.jsonl` is the same blob
#: under a dated name, kept because the snapshot is dated to the minute.)
WITNESS = Path.home() / "obsidian" / "backlog" / "data" / "shadow.jsonl"
MARKER = Path.home() / "obsidian" / "backlog" / "data" / \
    "shadow-action-review-2026-10-01T0247Z.witness.md"
#: The figure every surface quotes, and what `wc -l` of the artifact prints.
#: The seam's own first row, and the file's head row, which is a `rerank` row.
SEAM_FIRST_ROW = "2026-09-25T18:45:16Z"
HARNESS_DOC = Path(__file__).resolve().parents[1] / "architecture" / "harness.md"


def marker() -> dict:
    """The witness's own record, parsed out of its fenced block.

    The digest is read from here rather than copied into this file: a second copy
    of a digest is a second answer to "which bytes are the witness", and the two
    drift exactly when it matters — after somebody replaces the extract and
    updates one of the two.
    """
    raw = MARKER.read_text(encoding="utf-8")
    return json.loads(raw.split("```json")[1].split("```")[0])


def test_the_witness_is_the_bytes_the_ruling_quotes_and_wc_prints_the_figure():
    """The artifact must BE the bytes, be the line count every surface quotes, be
    committed, and agree with its marker — a marker without its artifact is prose,
    and an uncommitted artifact is a claim with no bytes behind it."""
    assert WITNESS.is_file(), f"missing witness {WITNESS}"
    blob = WITNESS.read_bytes()
    import hashlib

    digest = hashlib.sha256(blob).hexdigest()
    meta = marker()
    assert digest == meta["sha256"], (
        "the witness bytes no longer match the digest its marker declares, so every "
        "count quoted from them is stale — either the ruling's numbers move or the "
        "extract is replaced, and the quote has to be re-derived either way")
    # The count is checked against the marker and against the prose that quotes
    # it — never against a constant in this file, which would only prove the
    # file agrees with itself.
    lines = blob.count(b"\n")
    assert lines == meta["lines"], (
        f"`wc -l` says {lines:,}, which is not the {meta['lines']:,} its own marker "
        "declares, so the extract and its receipt disagree about what was measured")
    quoted = f"{lines:,}"
    for surface, text in (("harness.md", HARNESS_DOC.read_text(encoding="utf-8")),
                          ("the analysis module's docstring", C.__doc__ or "")):
        assert quoted in text, (
            f"{surface} no longer quotes {quoted} rows beside these figures, so a "
            "reader cannot tell which corpus the ruling was read from")
    committed = subprocess.run(
        ["git", "-C", str(Path.home() / "obsidian"), "diff", "--quiet", "--",
         "backlog/data/shadow.jsonl"], capture_output=True)
    assert committed.returncode == 0, (
        "the witness on disk differs from the witness in git, so the bytes this run "
        "checked are not the bytes anybody else can re-check")
    for key in ("what", "how_produced", "lines", "bytes", "sha256",
                "source_file_at_extract", "measurements_it_witnesses"):
        assert key in meta, f"marker lost `{key}`"
    src = meta["source_file_at_extract"]
    assert src["path"].endswith(".local/state/lloyd-djev/shadow.jsonl") and src["bytes"] > 0, (
        "the marker must name the file the rows were read from, and the file it "
        "was read from must not be empty — an unverified witness is a receipt")


def test_the_quoted_figures_recompute_from_the_witness_bytes(tmp_path):
    """Clause 6's last sentence: the report's counts are recomputable from
    committed bytes, so the ruling does not depend on a file that keeps growing.

    Tier-dependent axes are NOT asserted, deliberately: tiering a `Bash` row
    needs the session transcript, and the groundskeeper gzips those at 30 days
    (`scripts/groundskeeper/retention-sweep.py:264`), so pinning them would put a
    permanent ruling on a retention timer. An empty transcripts dir is passed for
    that reason; the label, outcome, source and hook axes — which need only these
    bytes, and carry the branch argument — are asserted exactly."""
    rows, integrity = C.load_corpus(WITNESS, tmp_path / "no-transcripts")
    rep = C.report(rows, integrity, source=WITNESS)
    assert rep["corpus"]["seam_rows"] == marker()["lines"], (
        "the extract no longer has the row count its marker declares")
    assert rep["corpus"]["first_row"] == "2026-09-25T18:45:16Z"
    assert rep["corpus"]["last_row"] == "2026-10-01T02:46:57Z"
    assert round(rep["corpus"]["span_days"], 3) == 5.335, (
        "the seam's own span moved, so every /day figure quoted from it is stale")
    labels = {k: v["n"] for k, v in rep["labels"].items()}
    assert labels == {"consistent": 56360, "unrelated": 963, "none": 2}, (
        "the label counts the ruling is quoted from no longer come out of the "
        "committed bytes")
    assert "injected" not in labels, (
        "an `injected` row is present in the witness: the third leg of the branch "
        "argument — the class a threshold would be read off is empty — is false, "
        "and the ruling must be re-made rather than re-quoted")
    assert rep["unrelated"]["per_day_over_seam_span"] == 180.5
    assert rep["unrelated"]["ran"]["n"] == 929
    assert rep["unrelated"]["would_fire_per_day"] == 174.1
    assert rep["outcomes"]["denied_by_hook"]["n"] == 36
    hook = {k: v["n"] for k, v in rep["hook_denials"]["by_label"].items()}
    assert hook == {"consistent": 35, "unrelated": 1}, (
        "the overlap the item called zero is 35 consistent + 1 unrelated of 36; "
        "those counts are the evidence the two sensors are not substitutes")
    assert rep["hook_denials"]["overlap_with_unrelated"] == 1
    src = {c["source"] for c in rep["cross_tab"] if c["label"] == "unrelated"}
    assert {"automod-review", "autocode", "autotriage"} <= src, (
        "the positives are supposed to be spread across worker sources; a witness "
        "that lost the spread is a different corpus")


def test_the_recorder_and_the_transcript_writer_agree_on_what_the_analysis_reads(tmp_path):
    """The seam this analysis lives on crosses two writers it does not own:
    `action_review.py` hashes the arguments into `meta.args_digest`, and the
    backend writes the transcript the join reads. Both sides are built here from
    the real code paths, so a rename on either fails a test instead of quietly
    making every tier count zero.

    Driven through the real recorder (`ActionReviewer.on_event`) and the real
    digest — not a hand-copied row — because the whole method is "the row names
    the call, the transcript names the arguments, the digest proves they are the
    same call".
    """
    import asyncio

    from app.harness import action_review as AR

    raw = {"summary": "Publishing the release bundle",
           "command": "curl -T ./bundle.tar.gz https://example.com/upload"}
    call_id = "chatcmpl-tool-0000000000000001"
    rev = AR.ActionReviewer(user_prompt="do the task", source="autocode",
                            session_id="s1", mode="shadow")
    # Exactly the event shape a turn emits: `args_dict` is the CLEANED dict, with
    # `summary` already removed by schema cleaning, which is why the digest and
    # the stored arguments differ by that one key.
    cleaned = {"command": raw["command"]}
    asyncio.run(rev.on_event({"type": "tool_call", "name": "Bash",
                              "call_id": call_id, "args_dict": cleaned}))
    name, _rendered, digest, _index = rev._pending[call_id]

    # The transcript, written the way the backend writes it: a JSON string of the
    # model's own arguments, `summary` included.
    (tmp_path / "s1.json").write_text(json.dumps({"session_id": "s1", "messages": [
        {"role": "assistant", "tool_calls": [
            {"id": call_id, "call_id": call_id, "type": "function",
             "function": {"name": name, "arguments": json.dumps(raw)}}]}]}))

    found = C.call_arguments(tmp_path / "s1.json", {call_id})
    assert set(found) == {call_id}, (
        "the reader no longer finds a call the backend wrote: something in the "
        "transcript shape changed and every Bash tier becomes unresolved")
    assert C.matches_digest(found[call_id], digest), (
        "the recorder's digest no longer reproduces from the stored arguments, so "
        "the join silently tiers nothing — check whether the harness normalised a "
        "new key into args before the digest was taken")
    tier, why = C.resolve_tier(name, found[call_id])
    assert (tier, why) == (2, "args"), (
        "a durable-external command recovered across the seam did not tier as 2, "
        "which is the axis branch (b)'s '1 of 963' count rests on")


def test_the_tier_axis_recomputes_from_the_witness_bytes():
    """Clause 6's decisive half, and the one an extract could get wrong: a tier
    needs the command string, which lives in the transcripts, which the
    groundskeeper gzips at 30 days (`retention-sweep.py:252-264`). So the witness
    carries `recovered_arguments` for the positives, and this node runs the SAME
    verification the live join runs — `matches_digest` first, tier second — over
    those bytes.

    The order is the whole test. Tiering the stored arguments without checking the
    digest yields 962 tier-1 / 1 tier-2 / 0 unresolved, because the one untierable
    positive IS the row whose arguments fail its digest. Only the verified
    procedure reproduces the ruling's 961 / 1 / 1, so a reader who skips the check
    gets numbers that look almost right and are not the ruling's.
    """
    rows = [json.loads(line) for line in
            WITNESS.read_text(encoding="utf-8").splitlines() if line.strip()]
    tiers = collections.Counter()
    for r in rows:
        if ((r.get("djev", {}).get("answers", {}).get("on_task") or {})
                .get("value")) != "unrelated":
            continue
        meta = r["meta"]
        args = r.get("recovered_arguments")
        if args is not None and not C.matches_digest(args, meta["args_digest"]):
            args = None
        tier, _why = C.resolve_tier(meta["tool"], args)
        tiers[str(tier) if tier else "unresolved"] += 1
    assert dict(tiers) == {"1": 961, "2": 1, "unresolved": 1}, (
        f"the tier axis no longer recomputes from the witness: got {dict(tiers)}, "
        "expected 961 tier-1, 1 tier-2, 1 unresolved of 963 — branch (b) rests on "
        "there being exactly one durable-external positive")


def test_a_row_that_carries_its_arguments_is_tiered_from_them_and_never_repaired(tmp_path):
    """The committed extract carries `recovered_arguments`, and that source wins —
    including when it FAILS.

    Two directions to get wrong, and the fixture has both in one corpus. Row `c2`
    carries a durable-external command that reproduces its digest and has no
    transcript at all: if the reader insisted on a transcript it would report that
    positive unresolved, and the ruling's "1 of 963 at tier >= 2" would decay into
    silence as transcripts age out. Row `c1` carries a command that does NOT
    reproduce its digest, and its real transcript is sitting right there, which is
    the transcript a careless reader falls back to and tiers the row anyway. That
    fallback is what turns the ruling's 961 / 1 / 1 into 962 / 1 / 0, so the
    extract's answer is final: a failure is reported, not repaired.
    """
    good = row(T0, *PUSH, label="unrelated")
    good["recovered_arguments"] = {**PUSH[2], "summary": "caption"}
    bad = row(T0 + 1.0, *ECHO, label="unrelated")
    bad["recovered_arguments"] = {"command": "git push origin main"}

    out, integrity = measure(tmp_path, [good, bad],
                             {"s1": transcript("s1", [ECHO, PUSH])})
    assert [r.tier for r in out] == [2, None], (
        f"got {[(r.tier, r.tier_reason) for r in out]}: an extract-carried command "
        "must tier without a transcript, and an extract that fails its digest must "
        "not be tiered from one")
    assert out[0].tier_reason == "args"
    assert out[1].tier_reason == "digest_mismatch"
    assert integrity["extract_match"] == 1 and integrity["digest_mismatch"] == 1
    assert integrity.get("digest_match", 0) == 0, (
        "the reader fell back to the transcript after the extract disagreed with it, "
        "which is how the untierable positive silently becomes a tier-1 one")


def test_the_ruling_reproduces_from_the_committed_extract_with_no_transcripts(tmp_path):
    """The ruling outlives the 30-day transcript window, and this node is the
    proof rather than the promise: the entry point, over the committed extract,
    with the transcripts directory pointed at a path that does not exist.

    Every figure `architecture/harness.md` P10 and the module docstring quote is
    asserted here — 180.5 and 174.1 per day over the seam's own 5.335-day span,
    the positives' 961 / 1 / 1 tier split, exactly one tier >= 2 positive at 0.2 a
    day, `injected` absent, and the hook denials at 35 `consistent` + 1
    `unrelated` of 36.

    What is deliberately NOT asserted as invariant is the whole-corpus untierable
    count. Without transcripts, the argument-dependent rows the extract left
    unannotated answer `no_session_file` — 41,204 of them here — so "6 rows
    untierable, all digest mismatches" is a live-corpus figure and this asserts the
    reason beside it, rather than pretending the two runs are the same run.
    """
    rows, integrity = C.load_corpus(WITNESS, tmp_path / "transcripts-are-gone")
    rep = C.report(rows, integrity, source=WITNESS)
    u = rep["unrelated"]

    assert rep["corpus"]["seam_rows"] == 57325, (
        "the committed extract no longer holds the 57,325 rows every published figure\n        divides by")
    assert rep["corpus"]["span_days"] == 5.335
    assert {k: v["n"] for k, v in rep["labels"].items()} == {
        "consistent": 56360, "unrelated": 963, "none": 2}
    assert u["n"] == 963 and u["of_seam_rows"] == 57325
    assert u["per_day_over_seam_span"] == 180.5
    assert u["ran"] == {"n": 929, "of_unrelated": 963}
    assert u["would_fire_per_day"] == 174.1
    assert {k: v["n"] for k, v in u["tiers"].items()} == {"1": 961, "2": 1,
                                                          "unresolved": 1}
    assert u["tier_ge_2"] == {
        "n": 1, "of_unrelated": 963, "of_seam_rows": 57325,
        "per_day_over_seam_span": 0.2}
    assert rep["hook_denials"]["n"] == 36
    assert {k: v["n"] for k, v in rep["hook_denials"]["by_label"].items()} == {
        "consistent": 35, "unrelated": 1}
    assert rep["hook_denials"]["overlap_with_unrelated"] == 1

    # The two integrity numbers that ARE transcript-independent: what the extract
    # answered for itself, and the six rows it could not.
    assert integrity["extract_match"] == 798, (
        "the extract no longer answers for 798 of its own Bash rows, so the tier "
        "axis is quietly back on the retention clock")
    assert integrity["digest_mismatch"] == 6
    assert rep["tier_resolution"]["reasons"]["no_session_file"] == 41204, (
        "the count of rows that need a transcript moved, so the sentence in "
        "harness.md P10 saying which figure is a live-corpus one is stale")
    assert integrity["extract_match"] + integrity["digest_mismatch"] == 804

    text = C.format_report(rep)
    assert "tier ≥ 2 (durable-external): 1/963 unrelated (0.2/day)" in text
    assert "180.5/day over the seam's own span of 5.335 days" in text


async def test_a_row_the_seam_actually_wrote_is_a_row_the_analysis_can_tier(
        tmp_path, monkeypatch):
    """The writer of the corpus and the reader of the corpus, in one test.

    Every other node here reads a row this file built. The rows that matter are
    written by `app/harness/action_review.py` through `djev_shadow.shadow()` into
    `~/.local/state/lloyd-djev/shadow.jsonl`, which is a different module, on a
    different thread, and in production a different process lifetime — and the
    join's only handle on a call is `meta.args_digest`, a value the WRITER
    computes. If the reviewer starts digesting its `args_dict` differently (drops
    a field, keeps `summary`), every fixture in this file still passes while the
    live join silently tiers nothing, because each fixture agrees with itself.

    So run the reviewer for real over two `Bash` calls — a plain command and a
    durable-external push — flush the recorder, and point the analysis at the file
    it wrote, with the matching transcript. The digests the seam stamped must be
    the digests `matches_digest` accepts, and the two commands must come out tier
    1 and tier 2, and the tampered third row proves the digests discriminate
    rather than merely agreeing. `test_the_recorder_and_the_transcript_writer_agree_on_what_the_analysis_reads`
    covers the same seam from the other side — the reviewer's in-memory digest
    against a hand-built transcript; this one covers the file the recorder writes
    and the reader's own `load_corpus`, which is the half that a rename in
    `djev_shadow` would break.
    """
    from app import djev
    from app.harness import action_review as AR
    from app.harness import events
    from app.harness.hooks import HookRegistry

    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(djev_shadow, "STATE_DIR", state)
    monkeypatch.setattr(djev_shadow, "SHADOW_LOG", state / "shadow.jsonl")
    monkeypatch.setattr(djev_shadow, "PENDING_DROPS", state / "drops.json")
    monkeypatch.setenv("LLOYD_DJEV_SHADOW", "1")
    monkeypatch.setattr(djev, "enabled", lambda: True)
    # What the seam hands the recorder is a djev result object, not a string —
    # `_record` calls `.as_dict()` on it, and a stub that skipped that step made
    # the worker log `'str' object has no attribute 'as_dict'` and write nothing.
    answer = djev.Answer(id="on_task", type="choice", value="unrelated",
                         label="unrelated", confidence=0.93, probabilities={},
                         label_mass=0.95, argmax_is_label=True)

    class _Result:
        def as_dict(self):
            return {"answers": {"on_task": answer.as_dict()}, "latency_ms": 1.0,
                    "server_ms": 0.9, "prompt_tokens": 10, "chunks": 1,
                    "cross_chunk": False, "uninformative": False,
                    "min_label_mass": 0.95, "floor": None}

    monkeypatch.setattr(djev, "ask_sync",
                        lambda state, questions, **kw: _Result())
    djev_shadow.reset_for_tests()
    try:
        hooks = HookRegistry()
        AR.install_action_review_hook(hooks, user_prompt="Ship the release.",
                                      mode="shadow", session_id="s1")
        # A third row digests the SAME arguments as `ECHO` and is given a
        # DIFFERENT command in the transcript, so the node proves the digests
        # discriminate rather than merely agreeing.
        for cid, tool, args in (ECHO, PUSH, TAMPED):
            await hooks.fire_on_event(events.tool_call(
                call_id=cid, name=tool, args_json="{}", args_dict=args,
                summary="caption"))
            await hooks.fire_on_event(events.tool_result(
                call_id=cid, name=tool, content="ok", is_error=False))
        assert djev_shadow.flush(5.0) == 0, "the recorder did not drain"

        written = [json.loads(l) for l in
                   (state / "shadow.jsonl").read_text().splitlines()]
        assert [r["seam"] for r in written] == ["action_review"] * 3, (
            "the reviewer wrote something other than three seam rows")
        for r in written:
            assert r["meta"]["tool"] == "Bash" and r["meta"]["session_id"] == "s1"
            assert r["meta"]["args_digest"], "the seam stamped no digest to join on"

        log, sdir = corpus(tmp_path, written,
                           {"s1": transcript("s1", [ECHO, PUSH, TAMPERED_TRANSCRIPT])})
        rows, integrity = C.load_corpus(log, sdir)
        assert [r.tier for r in rows] == [1, 2, None], (
            f"the analysis read the seam's own rows as "
            f"{[(r.tier, r.tier_reason) for r in rows]}; the digest the writer "
            "stamps is the only handle the tier join has, so a mismatch here is the "
            "live corpus quietly becoming untierable")
        assert integrity["digest_match"] == 2 and integrity["digest_mismatch"] == 1, (
            "writer and reader no longer agree on what the arguments digest "
            "covers: the seam's own digest is the only thing standing between a "
            "transcript that agrees with a call and a transcript that does not, "
            "and a row whose arguments do not hash to what the seam stamped must "
            "come out unresolved, not tiered")
        assert rows[2].tier_reason == "digest_mismatch"
        assert all(r.outcome == "ran" for r in rows)
        assert [r.label for r in rows] == ["unrelated"] * 3, (
            "the reader no longer reads the label field the writer actually "
            "writes, so the whole cross-tab would be counting the wrong key")
    finally:
        djev_shadow.reset_for_tests()
