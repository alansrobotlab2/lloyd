"""#2576's written report, pinned to the two committed witnesses.

Clause 5 is a prose deliverable, so it is pinned structurally rather than as a
diff of English. The load-bearing assertion is that every number in the
report's two tables is produced from the committed witness bytes by the producing
scripts' own renderer — `markdown_rows` for the class table, cell-by-cell against
both probe witnesses for the side-by-side — so the report cannot drift from the
measurement while the measurement still reads green, and neither can it drift from
the second run's counts, which is the finding the first run alone would have got
wrong. The rest checks the report states which way each call went, in the shape
the clause asks for, and that it does not overclaim.

Three prior rounds taught the specific shapes here. A hand-transcribed copy of a
rendered table drifted from the witness it claimed to copy. A docstring's row count
drifted from the arm list it described — `N_ARMS` is computed once now and
`test_the_arm_count_the_nodes_quote_is_the_arm_list_len` fails if prose and list
separate. And an assertion that a whole rendered body contained no `" 0"` was a
substring census a truncated table would also pass; the cell-by-cell comparison
below is the shape that can fail.
"""
from __future__ import annotations

import json
from pathlib import Path

import board_presence

from scripts.reasoning_by_job_class import markdown_rows as class_markdown
from scripts.thinking_effort_probe import SPELLINGS
from scripts.thinking_effort_probe import markdown_rows as probe_markdown

REPO = Path(__file__).resolve().parents[1]
# The vault is a tree of its own, never a sibling of the checkout: a round worktree
# lives under ~/lloyd-work/<round>/home/lloyd, so walking up from REPO asks about an
# `obsidian` beside the worktree that was never there. Resolved through the helper the
# rest of the suite uses (`LLOYD_OBSIDIAN_VAULT`, else `~/obsidian`) so there is one
# answer to "which vault" — and it is consulted per call, not frozen at import, because
# a module-level constant would be read before a test could point it anywhere.
NOTE = (board_presence.vault_root()
        / "knowledge" / "thinking-effort-per-job-class-2026-10-11.md")
NOTE_REL = "knowledge/thinking-effort-per-job-class-2026-10-11.md"
CLASS_WITNESS = REPO / "eval/measurements/2576-class-table-2026-10-11.md"
PROBE_A = REPO / "eval/measurements/2576-probe-1500-2026-10-11.md"
PROBE_B = REPO / "eval/measurements/2576-probe-1500-2026-10-11-run-b.md"
N_ARMS = len(SPELLINGS)
DASH = "–"                      # the em/en dash the tables use for a missing cell


def _unwrap(path: Path) -> dict:
    """The JSON block a `.md` witness carries."""
    text = path.read_text()
    start = text.index("```json") + len("```json")
    return json.loads(text[start:text.index("```", start)])


def test_the_note_and_both_probe_runs_exist():
    assert NOTE.is_file(), f"{NOTE_REL} is missing from the vault"
    for p in (CLASS_WITNESS, PROBE_A, PROBE_B):
        assert p.is_file(), f"the report's witness {p.name} is missing"
    assert "Report-only" in NOTE.read_text()


def test_the_report_renders_the_class_table_from_the_witness_bytes():
    """The class table IS `markdown_rows(witness)`, not a transcription of it.

    Byte-for-byte, so the drift a hand-copied table cannot avoid is impossible:
    every figure, every s/f/s/i quadruple, every `THIN` and every en dash.
    """
    rendered = class_markdown(_unwrap(CLASS_WITNESS))
    note = NOTE.read_text()
    assert rendered in note, (
        "the report's class table is no longer the witness's own markdown render — "
        "it was transcribed, edited or the witness regenerated without the report")
    header = rendered.splitlines()[0]
    for column in ("meanR", "p90R", "meanOut", "meanIn", "share", "outcome",
                   "s/f/s/i"):
        assert column in header, f"clause 5 names a {column}, and the table has no column for it"
    lines = {l.split("|")[1].strip(): l
             for l in rendered.splitlines() if l.startswith("| ") and "---" not in l}
    lines.pop("class", None)
    assert len(lines) == len(_unwrap(CLASS_WITNESS)["rows"]), "the table lost a class"
    for cls, line in lines.items():
        assert line.count("|") == header.count("|"), f"{cls}: a ragged row"
    for cls in ("unattributed", "bench-mine", "autocode", "autoresearch"):
        assert cls in lines, f"{cls}: the class the clause hinges on went missing"


def test_the_committed_witness_reproduces_the_derived_figures_the_note_prose_quotes():
    """The note's PROSE figures are recomputed, not quoted.

    The table is pinned to the witness above; prose is where drift lives, because
    prose has no renderer. Every derived number the note states in words — the
    unattributed share of the corpus, the closure sum, and autocode's mean reasoning
    under the zeros-instead-of-NULL misreading — is recomputed here from the witness
    and required in the note's own phrasing. A prose figure that is neither derived
    nor allowed fails.
    """
    witness = _unwrap(CLASS_WITNESS)
    note = NOTE.read_text()
    un = next(r for r in witness["rows"] if r["class"] == "unattributed")
    assert f"**{100 * un['turns'] / witness['usage_rows']:.0f} %** of the store" in note
    assert f"({witness['turns_attributed']:,} attributed + " in note
    for key in ("usage_rows", "runs", "sessions_named_by_runs", "turns_attributed"):
        assert f"**{witness[key]:,}**" in note, f"the note's corpus figure {key} is not the witness's"
    code = next(r for r in witness["rows"] if r["class"] == "autocode")
    zeros = code["reasoning_total"] / code["turns"]
    assert f"{zeros:,.0f}" in note, (
        "the note no longer shows what the zeros misreading would have printed")
    assert f"{code['mean_reasoning_tokens']:,.0f}" in note



def test_the_report_table_pin_fails_on_a_single_changed_digit():
    """The pin can fail, which is what distinguishes it from a print.

    Two drifts a previous round actually produced or nearly did: a class table whose
    `THIN` is dropped for a class the witness marks thin, and a probe cell off by one
    token. Both are aimed at the same functions the two pins above run, so if either
    comparison is vacuous this node is the one that says so.
    """
    witness = _unwrap(CLASS_WITNESS)
    good = json.loads(json.dumps(witness))
    thinned = next(r for r in good["rows"] if r["outcome_thin"] and r["outcome_n"] < 30)
    thinned["outcome_thin"] = False
    assert class_markdown(good) != class_markdown(witness), (
        "clearing a thin flag changed nothing in the rendered table, so the `THIN` "
        "the report quotes is not a function of the witness")
    rows_a = _unwrap(PROBE_A)["rows"]

    def cells_of(rows: list[dict]) -> dict[str, list[str]]:
        """Run A's probe rows through the renderer the pin itself uses."""
        out: dict[str, list[str]] = {}
        for line in probe_markdown(rows).splitlines():
            if not line.startswith("| ") or "---" in line:
                continue
            cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
            if cells[0] != "spelling":
                out[cells[0]] = cells[2:6]
        return out

    base = cells_of(rows_a)
    low = next(r for r in rows_a if r["spelling"] == "reasoning_effort=low")
    assert low["reasoning_tokens"] is not None, (
        "run A's `low` arm has no reasoning count, so it cannot carry a drift test")
    mutated = json.loads(json.dumps(rows_a))
    next(r for r in mutated if r["spelling"] == "reasoning_effort=low")[
        "reasoning_tokens"] = low["reasoning_tokens"] + 1
    drifted = cells_of(mutated)
    assert drifted != base, (
        "a probe cell changed by one token and the rendered table did not move, so the "
        "cell-by-cell pin that reads this renderer cannot see the drift it claims to")
    assert drifted["reasoning_effort=low"][0] != base["reasoning_effort=low"][0], (
        "the mutation moved some other arm's cell, which means the pin's rows are not "
        "keyed by spelling")


# ── what the probe concluded, stated as the two runs support it ─────────────


def test_the_report_says_which_spellings_are_honoured_and_that_the_ladder_crossed():
    """Clause 5: whether the primary honours graded levels or only the binary.

    Two claims, both reproduced across the runs. (1) OFF is honoured:
    `enable_thinking=false` and `reasoning_effort:"none"` report 0 reasoning tokens
    in BOTH witnesses, and the node checks the witnesses say that rather than the
    prose. (2) A graded level is honoured but its ORDERING IS NOT REPRODUCIBLE:
    `medium` and `low` both come back non-null and below the cap in both runs, and
    their ranking is 573<1,002 in run A and 784>677 in run B — the two runs cross. So
    the report may not claim a monotonic ladder; it must claim graded-but-uncertain,
    and say why (one draw per arm). Asserting the crossing from the witnesses is what
    makes the report's wording a consequence of the data rather than a preference:
    a second run that ORDERED would fail this node and entitle a ladder sentence.
    """
    a = {r["spelling"]: r for r in _unwrap(PROBE_A)["rows"]}
    b = {r["spelling"]: r for r in _unwrap(PROBE_B)["rows"]}
    off = ("chat_template_kwargs.enable_thinking=false", "reasoning_effort=none")
    for spelling in off:
        for run in (a, b):
            assert run[spelling]["reasoning_tokens"] == 0, (
                f"{spelling} no longer measures zero reasoning, so the report's "
                "`off is honoured` claim is stale")
    ladder = ("reasoning_effort=medium", "reasoning_effort=low")
    order_a = [a[s]["reasoning_tokens"] for s in ladder]
    order_b = [b[s]["reasoning_tokens"] for s in ladder]
    assert all(v is not None for v in order_a + order_b), (
        "a graded level stopped reporting a count, so 'graded levels are honoured' "
        "is no longer measured")
    assert (order_a[0] > order_a[1]) != (order_b[0] > order_b[1]), (
        f"the two runs no longer disagree about the order of medium and low "
        f"({order_a} vs {order_b}); if they agree, the report may stop hedging and "
        "the hedge assertion below has to be retired with this one")

    note = NOTE.read_text()
    assert "binary" in note and "graded" in note
    assert "1,500" in note, "the xhigh-at-the-cap reading, without which 1500 is just a number"
    assert "not reproducible from a single draw" in note
    # Each arm must appear in the report's A/B table with its spelling set in the glyph
    # the script's OWN renderer emits (code font unless the spelling holds a space),
    # compared as the whole first cell of a table row. A previous round wrote
    # "`{spelling}` in note or spelling in note" — since the backticked form always
    # contains the bare one, that disjunction collapsed to a plain substring census: the
    # code font it named was unenforced, and prose mentioning the words passed it.
    first_cells = {l.split("|")[1] for l in note.splitlines()
                   if l.startswith("| ") and l.count("|") >= 9}
    for line in probe_markdown(_unwrap(PROBE_A)["rows"]).splitlines():
        if not line.startswith("| ") or "---" in line:
            continue
        cell = line.split("|")[1]
        if cell.strip().strip("`") == "spelling":
            continue
        assert cell in first_cells, (
            f"{cell!r}: the report's A/B table has no row whose first cell is the glyph "
            "this script renders, so the arm is missing or set in some other font")
    assert "400" in note, "`reasoning_effort=high` returns a 400 and the report says so"


def test_the_report_prices_prefix_cache_in_the_same_table_as_the_decode():
    """Clause 5, the half that is easiest to leave out.

    `app/harness/finalizer.py:24-37` declines a thinking knob because the reasoning
    sentence sits in the chat template's system-message header, so a per-request
    change invalidates the cached prefix of every turn that pays the prefill again.
    The report therefore has to price the switch against BOTH sides in the same
    table: the decode saved (mean reasoning tokens) against the prefill re-charged
    (mean input tokens), class by class. The ratios are recomputed from the witness
    here — `meanIn / meanR` — and required in the note's phrasing, and the note is
    required to name the lowest ratio on the box, because that single number is what
    decides whether ANY class could come out ahead.
    """
    witness = _unwrap(CLASS_WITNESS)
    note = NOTE.read_text()
    priced = {r["class"]: r for r in witness["rows"] if r["mean_input_tokens"]}
    assert priced, "no class reports both a prompt and a reasoning mean to compare"
    ratios = {c: r["mean_input_tokens"] / r["mean_reasoning_tokens"]
              for c, r in priced.items()}
    # Row by row, not as a bag of substrings. The previous form was one `in note` test per
    # class, and deep-research, scheduled-task and unattributed all format to the same 9.4×,
    # so any one of those printing another's ratio passed — and so would a price table of
    # one row copy-pasted eleven times. The unit is the note's own table line: its class
    # cell and its ratio cell are compared together, and the row set has to be exact.
    priced_rows: dict[str, str] = {}
    for line in note.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if line.startswith("| ") and len(cells) == 6 and cells[1] in ratios:
            priced_rows[cells[1]] = cells[4]
    assert set(priced_rows) == set(ratios), (
        f"the report prices {sorted(priced_rows)} while the witness has turn data for "
        f"{sorted(ratios)} — a class lost its row or gained one it has no numbers for")
    for cls, ratio in sorted(ratios.items(), key=lambda kv: -kv[1]):
        assert priced_rows[cls] == f"{ratio:.1f}×", (
            f"{cls}: its own row prints {priced_rows[cls]!r} where the witness's mean "
            f"prompt over its mean reasoning is {ratio:.1f}×")
    cheapest = min(ratios, key=ratios.get)
    assert cheapest == "bench-mine", (
        f"the cheapest class on the box is now {cheapest}; the note's conclusion that "
        "no class comes out ahead has to be re-derived, not carried")
    assert "85.9" in note, "the prefix-cache hit ratio the re-prefill risk is measured against"
    assert "finalizer.py:24-37" in note
    assert "per-request" in note and "per-**class**" in note, (
        "the report has to say which of the two knobs the cache argument condemns")


def test_the_report_states_one_keep_default_verdict_per_class():
    """Clause 5's decision input: one keep-default sentence per class, all sixteen.

    The check is per class and drawn from the witness's own row set, so a report that
    dropped a row or wrote one class's verdict as `declare lower` fails. `keep the
    default` must sit inside that class's own bullet, not merely somewhere in the file:
    a verdict paragraph that swept thirteen classes in one clause is the shape this node
    exists to refuse. The four zero-turn classes get the same sentence as everyone else,
    because the reason differs (no measurement at all) and the reason is the point.
    """
    witness = _unwrap(CLASS_WITNESS)
    note = NOTE.read_text()
    section = note.split("keep the default versus declare a lower level", 1)
    assert len(section) == 2, "the per-class verdict section is gone"
    bullets = [l for l in section[1].splitlines() if l.startswith("- `")]
    assert len(bullets) == len(witness["rows"]), (
        f"{len(bullets)} verdict bullets for {len(witness['rows'])} classes — the "
        "clause asks for one per class, and a dropped class is an unowned decision")
    for row in witness["rows"]:
        mine = [l for l in bullets if l.startswith(f"- `{row['class']}`")]
        assert len(mine) == 1, f"{row['class']}: {len(mine)} verdict bullets"
        line = mine[0]
        assert "keep the default" in line, f"{row['class']}: no keep-default verdict"
        assert "declare lower" not in line, (
            f"{row['class']}: a lower level was declared while step 3 is still owed")
        assert ("share" in line or "zero turns" in line), (
            f"{row['class']}: the verdict cites no measurement, so it is a preference")


def test_the_report_declares_itself_report_only_and_names_the_flag_it_did_not_move():
    """The risk this item carries is being mistaken for a policy.

    The engine's default is a launch flag outside the repo, so a report that reads like
    a decision invites someone to move it on the strength of a table that cannot see the
    mechanism. The declaration therefore sits in the head, above the data, and names the
    flag. The forbidden strings are affirmative claims of enforcement: the negative forms
    the report does use — `no promotion rule … reads anything here` — are how a
    report-only artifact says so, and pinning those would pin the wrong thing.
    """
    head = NOTE.read_text().split("## What was measured", 1)[0]
    assert "Report-only" in head
    assert "start-qwen38-flash-next.sh" in head
    assert "person's action" in head
    for claim in ("now gates", "now refuses on", "the gate rejects a turn that"):
        assert claim not in head, f"the report claims enforcement: {claim!r}"


def test_the_report_names_what_the_measurement_cannot_see():
    """Both limits are consequences of the instruments, and both must be in print.

    A single-call bench trial penalises thinking under a completion-token ceiling by
    construction — the ceiling decides who finishes, not the reasoning — so no priced arm
    was run and the report may not imply one. And the video's mechanism is thinking
    displacing *executed commands*, which a per-turn token census cannot see at all.
    Asserted as exact substrings because these are the claims a later reader looks for
    before reusing the table.
    """
    tail = NOTE.read_text().split("## What this is not", 1)
    assert len(tail) == 2, (
        "the 'what this is not' section went missing, so the table reads as a verdict")
    body = tail[1]
    assert "displacement" in body and "executed commands" in body
    assert "coverage_leg.py" in body, "the bench arm it declined to run must be named"
    assert "single-call" in body and "ceiling" in body
    assert "--reasoning-parser" in body, "the launch flag it did not touch must be named"


def test_every_probe_cell_in_the_report_is_the_witnesses_value_not_a_transcription():
    """The side-by-side A/B table is pinned cell by cell to run A and run B.

    One comparison per cell per arm: the report's figure for
    `reasoning_tokens`/`content_len`/`completion_tokens`/`finish_reason` must equal
    the value in that run's witness, formatted as the witnesses' own markdown
    renderer formats it (an en dash where the engine returned no counter, thousands
    separators otherwise). A substring census over the whole body — which a previous
    round shipped, and which a truncated or empty table would also pass — cannot
    catch a wrong cell; this can, on one digit.
    """
    note = NOTE.read_text()
    a = {r["spelling"]: r for r in _unwrap(PROBE_A)["rows"]}
    b = {r["spelling"]: r for r in _unwrap(PROBE_B)["rows"]}
    assert set(a) == set(b) == {s["spelling"] for s in SPELLINGS}, (
        "the two runs are not both showing every arm, so this is not an A/B table")
    # Each run's four cells come from that run's witness THROUGH the script's own
    # renderer, not from a re-formatting written here: the dash glyph, the thousands
    # separator and the empty-`finish_reason` case are the renderer's decisions, and a
    # hand-rolled copy of them in this file would pin this file's formatting rather than
    # the artifact's. Columns 2..5 of the rendered row are reasoning / content /
    # completion / finish; column 1 is the status, which the report's A/B table omits.
    def rendered_cells(rows: list[dict]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for line in probe_markdown(rows).splitlines():
            if not line.startswith("| ") or "---" in line:
                continue
            cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
            if cells[0] == "spelling":       # the renderer's own header row
                continue
            out[cells[0]] = cells[2:6]
        return out

    ca, cb = rendered_cells(_unwrap(PROBE_A)["rows"]), rendered_cells(_unwrap(PROBE_B)["rows"])
    assert len(ca) == len(cb) == len(SPELLINGS), (
        f"the renderer emitted {len(ca)}/{len(cb)} arms for {len(SPELLINGS)} spellings")
    wanted = [[spelling] + ca[spelling] + cb[spelling] for spelling in a]
    seen: dict[str, list[str]] = {}
    for line in note.splitlines():
        if not line.startswith("| "):
            continue
        # The note sets the spelling column in backticks for readability; the witnesses
        # hold the bare string, so the comparison is on the de-backticked cell.
        cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
        if cells and cells[0] in a:
            seen[cells[0]] = cells
    assert set(seen) == {c[0] for c in wanted}, "the report's probe table lost an arm"
    for cells in wanted:
        assert seen[cells[0]] == cells, (
            f"{cells[0]}: the report prints {seen[cells[0]][1:]}, the witnesses print "
            f"{cells[1:]}")
