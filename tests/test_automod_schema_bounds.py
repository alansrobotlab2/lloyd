"""Grammar bounds on the schemas the automod graders send to the finalizer.

`app.harness.finalizer.run_finalizer` splits a completion it cannot parse into
two diagnoses, and which one it gives is decided entirely by the schema it was
handed (`app/harness/finalizer.py`, `_schema_is_bounded`): when every string the
schema admits carries a positive `maxLength` it reports `generation diverged at N
tokens`, and when any string is open it reports `raise harness.finalizer.max_tokens`
— advice to edit config.yaml. #1706 established that the second message is a lie
when the grammar could have capped the runaway itself
(`app/harness/tests/test_finalizer.py::test_a_cut_under_a_schema_that_caps_every_field_is_a_divergence`),
and #2240 applies it to `REVIEW_SCHEMA`, the review grader's own answer shape.

The defect this file pins is the one the item's own fix would have missed:
`_schema_is_bounded` requires EVERY property bounded, and capping the three
clause-row fields the item named (`note`, `evidence_path`, `test_node_id`) still
measures False — eight string leaves were open, so the control node below keeps
the under-scoped fix from coming back.
"""
from __future__ import annotations

import copy
import inspect
import json
import re
from pathlib import Path

from app.harness.finalizer import _schema_is_bounded
from scripts.automod import review as RV

# The eight string leaves `REVIEW_SCHEMA` admitted without a bound, by their path
# in the schema. `premise`, `verdict`, `how_verified` and `severity` are also
# strings and need no cap: an `enum` is bounded on its own, and
# `_node_is_bounded` says so before it ever looks for a `maxLength`.
CAPED_LEAVES = {
    "clauses[].evidence_path": "EVIDENCE_PATH_MAX",
    "clauses[].test_node_id": "TEST_NODE_ID_MAX",
    "clauses[].note": "CLAUSE_NOTE_MAX",
    "test_honesty[].file": "HONESTY_FILE_MAX",
    "test_honesty[].problem": "HONESTY_PROBLEM_MAX",
    "seams_unverified[].seam": "SEAM_MAX",
    "amendments_note": "AMENDMENTS_NOTE_MAX",
    "summary": "SUMMARY_MAX",
}

# The three the item named. Capping these and nothing else is the under-scoped
# fix the control node refuses.
CLAUSE_ROW_FIELDS = ("clauses[].evidence_path", "clauses[].test_node_id", "clauses[].note")


def string_leaves(node: dict, prefix: str = "") -> dict:
    """Every `{"type": "string"}` node in a schema, keyed by its schema path.

    Array items are spelled `field[]`, so a path here reads the way the clauses
    and the item write them (`clauses[].note`).
    """
    out: dict[str, dict] = {}
    if not isinstance(node, dict):
        return out
    if node.get("type") == "string":
        out[prefix or "<root>"] = node
    for key, child in (node.get("properties") or {}).items():
        out.update(string_leaves(child, f"{prefix}.{key}" if prefix else key))
    items = node.get("items")
    if isinstance(items, dict):
        out.update(string_leaves(items, f"{prefix}[]"))
    return out


def _strip_caps(schema: dict) -> dict:
    """A copy with every `maxLength` removed: the pre-#2240 grammar."""
    out = copy.deepcopy(schema)
    stack = [out]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            cur.pop("maxLength", None)
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return out


def _cap_field(schema: dict, path: str, value: int) -> None:
    """Set `maxLength` on the leaf at a `string_leaves` path."""
    node = schema
    for part in [p for p in re.split(r"\.|\[\]", path) if p]:
        if "items" in node:
            node = node["items"]
        node = node["properties"][part]
    node["maxLength"] = value


# ── the grammar is bounded, and only the whole set makes it so ──────────────

def test_every_string_leaf_of_the_review_schema_carries_a_positive_max_length():
    """The acceptance check: eight leaves, eight caps, and `_schema_is_bounded`
    True. Before #2240 the only `maxLength` in this file's `git grep` was the
    comment that banned having one."""
    leaves = string_leaves(RV.REVIEW_SCHEMA)
    open_strings = {path: node for path, node in leaves.items() if not node.get("enum")}

    assert set(open_strings) == set(CAPED_LEAVES), (
        f"the schema's unenumerated string leaves moved: {sorted(open_strings)}")
    for path, node in open_strings.items():
        cap = node.get("maxLength")
        assert isinstance(cap, int) and not isinstance(cap, bool) and cap > 0, (
            f"{path} is a string with no positive maxLength — the finalizer would "
            "read a truncation under this schema as the budget")
        assert getattr(RV, CAPED_LEAVES[path]) == cap, (
            f"{CAPED_LEAVES[path]} and the grammar disagree for {path}")
    assert _schema_is_bounded(RV.REVIEW_SCHEMA) is True


def test_capping_only_the_three_clause_row_fields_leaves_the_schema_unbounded():
    """The item's own fix, sized as written, still measures False.

    `_schema_is_bounded` asks every property, so five open leaves left behind
    (`test_honesty[].file`, `test_honesty[].problem`, `seams_unverified[].seam`,
    `amendments_note`, `summary`) keep the whole schema unbounded and the reader
    still gets the config-knob advice. The first assert is the pre-change shape,
    which is what makes this node able to fail."""
    stripped = _strip_caps(RV.REVIEW_SCHEMA)
    assert _schema_is_bounded(stripped) is False, (
        "an uncapped REVIEW_SCHEMA reads as bounded — the split is not being "
        "measured the way the finalizer measures it")

    partial = copy.deepcopy(stripped)
    for path in CLAUSE_ROW_FIELDS:
        _cap_field(partial, path, getattr(RV, CAPED_LEAVES[path]))
    assert _schema_is_bounded(partial) is False, (
        "capping only the clause row now bounds the schema, so the five leaves "
        "this node exists to protect have gone uncapped again")

    whole = copy.deepcopy(stripped)
    for path in CAPED_LEAVES:
        _cap_field(whole, path, getattr(RV, CAPED_LEAVES[path]))
    assert _schema_is_bounded(whole) is True


# ── the caps are the ceiling the code already enforces, not a new one ───────

#: Each capped field whose post-parse slice exists, with the fragment of
#: `parse_review`'s own source that carries it. Built from the constants, so the
#: cap and the slice can only agree by one of them being edited: change either
#: and the fragment string stops matching and this file says so.
SLICE_FRAGMENTS = {
    "note": lambda: f'"note": note[:{RV.CLAUSE_NOTE_MAX}],',
    "file": lambda: f'str(raw.get("file") or "")[:{RV.HONESTY_FILE_MAX}]',
    "problem": lambda: f'str(raw["problem"]).split())[:{RV.HONESTY_PROBLEM_MAX}]',
    "seam": lambda: f'str(raw_seam.get("seam") or "").split())[:{RV.SEAM_MAX}]',
    "summary": lambda: f'str(obj.get("summary") or "").split())[:{RV.SUMMARY_MAX}]',
    "amendments_note": lambda: (
        f'str(obj.get("amendments_note") or "").split())[:{RV.AMENDMENTS_NOTE_MAX}]'),
}

#: The five of those six whose value is collapsed BEFORE the slice runs, each with
#: the fragment that carries its `" ".join(...split())`. `file` is absent because it
#: is the only cap that counts the same characters `maxLength` counts.
COLLAPSED_SLICE = {
    "note": lambda: 'note = " ".join(str(raw.get("note") or "").split())',
    "problem": lambda: 'str(raw["problem"]).split())',
    "seam": lambda: 'str(raw_seam.get("seam") or "").split())',
    "summary": lambda: 'str(obj.get("summary") or "").split())',
    "amendments_note": lambda: 'str(obj.get("amendments_note") or "").split())',
}


def test_every_cap_sits_at_the_slice_parsereview_already_applies():
    """Six of the eight caps are not a decision: they are the ceiling
    `parse_review` has always applied to that same value after parsing.

    The six numbers agree; six counts of them do not. `file` is the only field
    sliced as it arrives. The other five — `note`, `problem`, `seam`, `summary`,
    `amendments_note` — are run through `" ".join(str(...).split())` FIRST, which
    collapses runs of whitespace, so their slice counts the collapsed string while
    `maxLength` counts the raw text the decoder wrote. For those five the cap
    therefore bounds MORE raw characters than the slice keeps: a note the grammar
    admits at exactly 600 raw characters, three of them doubled spaces, collapses to
    597 and `[:600]` keeps every one of them, so nothing the decoder was allowed to
    write is lost. The other direction cannot happen by construction — the grammar
    counts the raw text, so it never admits a value the slice would have cut.
    `COLLAPSED_SLICE` pins which five those fields are, so this sentence and the
    code cannot drift apart silently.

    (The legacy bare-string seam is trimmed by a second, field-name-less slice at
    the same length; only the dict shape's slice is pinned here.)"""
    src = inspect.getsource(RV.parse_review)
    for field, fragment in SLICE_FRAGMENTS.items():
        text = fragment()
        assert text in src, (
            f"parse_review no longer trims `{field}` the way the grammar caps it "
            f"({text!r} is not in its source) — one of the two is off by now")
    for field, fragment in COLLAPSED_SLICE.items():
        assert fragment() in src, (
            f"`{field}` no longer collapses whitespace before its slice, so the "
            f"raw-versus-collapsed paragraph above is wrong about it — fix the "
            f"prose or the code, not this assert ({fragment()!r} is missing)")
    assert set(COLLAPSED_SLICE) == set(SLICE_FRAGMENTS) - {"file"}, (
        "the collapse map and the slice map no longer differ by `file` alone")
    assert 'str(raw.get("file") or "")[:' in src, (
        "`file` gained a collapse, so it is no longer the one character-exact cap")


def test_the_two_path_fields_have_no_post_parse_slice_to_read_off():
    """`evidence_path` and `test_node_id` are trimmed by nothing: `parse_review`
    strips them and passes them on, so the grammar cap is the only ceiling either
    field has ever had. That is what makes their numbers a decision (see the
    sizing node), and a later reader must not assume a slice is waiting there.
    Graded on the two lines that decide what is stored — the assignment and the
    clause row — not on the presence of any `[:120]` elsewhere, several of which
    are display slices inside a refusal message and trim nothing.
    """
    src = inspect.getsource(RV.parse_review)
    assert 'raw_path = str(raw.get("evidence_path") or "").strip()' in src, (
        "evidence_path's assignment changed shape; re-read it and check whether "
        "the grammar cap is still the only ceiling on it")
    assert 'node = str(raw.get("test_node_id") or "").strip()' in src, (
        "test_node_id's assignment changed shape; same re-read")
    assert '"evidence_path": path,' in src, "the clause row now trims evidence_path"
    assert '"test_node_id": node,' in src, "the clause row now trims test_node_id"


#: Longest value each of the two undecided fields has on record, measured 2026-10-05
#: over the `review`/`vault_review`/`vault_land` rows of a 22,916-line
#: `~/.local/state/lloyd-automod/promotions.jsonl` — append-only state in no tree, so
#: these are the figures, quoted with their measurement time. `EVIDENCE_PATH_MEASURED_MAX`
#: is also below the longest path this repo can name (100 characters,
#: `git ls-files | awk '{print length($0)}' | sort -n | tail -1`), which is the half a
#: reader CAN re-check from a checkout.
EVIDENCE_PATH_MEASURED_MAX = 95
TEST_NODE_ID_MEASURED_MAX = 420


def test_the_two_decided_caps_are_pinned_at_their_chosen_headroom():
    """These two caps are the only ones that are a decision rather than a copy of a
    slice, so the node pins the decision itself — the chosen number, and how much
    room it leaves over the longest value on record — rather than a bare inequality
    that a quiet retune could satisfy.

    Headroom today is 205 characters over `evidence_path`'s 95 and 180 over
    `test_node_id`'s 420. A cap below the longest value a field holds would trade a
    loud truncation for a silent one, which is what the lower bounds are for; they
    do NOT promise the headroom survives, because a longer value than either figure
    is a fact about future grader output and no node here can read it."""
    assert (RV.EVIDENCE_PATH_MAX, RV.TEST_NODE_ID_MAX) == (300, 600), (
        "the decided caps moved; re-measure the maxima before changing them")
    assert RV.EVIDENCE_PATH_MAX > max(EVIDENCE_PATH_MEASURED_MAX, 100), (
        "the cap is at or below the longest value on record, or below the longest "
        "path this repo can name")
    assert RV.TEST_NODE_ID_MAX > TEST_NODE_ID_MEASURED_MAX, (
        "the cap is below the longest node id on record")
    assert RV.CLAUSE_NOTE_MAX == 600, (
        "clause 2 of #2240 pins this one specifically: it is the ceiling "
        "`note[:600]` already applied, and `test_every_cap_sits_at_the_slice_"
        "parsereview_already_applies` is what keeps the two together")


def test_a_clause_note_at_the_cap_survives_parse_review_whole(tmp_path):
    """The cap must bind the guided decoder and nobody else: a note of exactly 600
    characters comes out of `parse_review` at 600, and one character over comes out
    at 600 — cut by the slice that has cut it all along, not by anything #2240
    added. This is the parser's boundary; the clause row that goes on from there is
    `parse_review`'s own business, and the wire boundary has its own node below."""
    def _one(note: str, wt: Path) -> dict:
        obj = {"premise": "sound",
               "clauses": [{"clause": 1, "verdict": "partial", "evidence_path": "",
                            "evidence_line": 0, "test_node_id": "",
                            "how_verified": "read", "note": note}],
               "test_honesty": [], "seams_unverified": [], "summary": ""}
        return RV.parse_review(obj, worktree=wt, changed_tests=[], n_clauses=1)

    exact = _one("x" * RV.CLAUSE_NOTE_MAX, tmp_path)
    assert len(exact["clauses"][0]["note"]) == RV.CLAUSE_NOTE_MAX, (
        "a note the grammar admits is being shortened before the clamp sees it")
    over = _one("x" * (RV.CLAUSE_NOTE_MAX + 1), tmp_path)
    assert len(over["clauses"][0]["note"]) == RV.CLAUSE_NOTE_MAX


# ── the comment that used to ban this ───────────────────────────────────────

def _comment_above(marker: str) -> str:
    """The contiguous `#` block sitting directly above `marker` in review.py."""
    lines = Path(inspect.getsourcefile(RV)).read_text().splitlines()
    at = next((i for i, line in enumerate(lines) if line.startswith(marker)), None)
    if at is None:
        return ""
    out: list[str] = []
    i = at - 1
    while i >= 0 and lines[i].lstrip().startswith("#"):
        out.append(lines[i])
        i -= 1
    return "\n".join(reversed(out))


def test_the_comment_above_the_schema_names_the_cap_and_the_runaway():
    """The comment above `REVIEW_SCHEMA` used to read "No maxLength: the decoder
    would stop mid-sentence at it". #1706 judged that exact argument and rejected
    it for `RESULT_SCHEMA`, and its own node requires the replacement to say both
    halves "or the next reader reverts it"
    (`tests/test_deep_research_source.py::test_the_schema_comment_names_the_bound_and_what_it_makes_impossible`).
    Same rule here."""
    comment = _comment_above("REVIEW_SCHEMA: dict")
    assert comment, "no comment above REVIEW_SCHEMA to grade"
    assert "maxLength" in comment, "the comment no longer names the cap"
    assert "8192" in comment, (
        "the comment no longer names the runaway the cap prevents, so it reads "
        "as decoration and the next pass strips the caps out")


# ── the figures that comment publishes have bytes behind them ───────────────

#: Same 11 rows as the vault's
#: `~/obsidian/backlog/data/2026-10-05.2240-truncation-witness.jsonl`, sha256
#: a41c646825779eca893b8a9f05b199715efe4567aa558a6d287133cb7a4ebdb5. The vault copy is
#: what #2240 clause 5 asked for; this one exists because the gate runs with HOME at the
#: round home, where `~/obsidian` is not there — a node that opened only the vault file
#: would skip, and a skipping node pins nothing.
WITNESS = Path(__file__).parent / "fixtures" / \
    "promotions_review_truncation_rows_2026-10-05-item2240.jsonl"


def test_the_committed_witness_holds_the_eleven_rows_the_comment_counts():
    """The comment above `REVIEW_SCHEMA` publishes counts out of an append-only ledger
    that lives in no tree, so the rows are committed and this node re-derives every
    published figure from them: eleven truncations in total, nine of them the vault
    leg and two the code leg, each of the nine landing `kind: skipped` with
    `clauses: []` — the vault change validated and graded nothing — and each of the
    eleven carrying the advice #2240 exists to withdraw.

    The item list is parsed out of the comment and compared to the rows, so the
    comment cannot keep a number the bytes do not hold."""
    assert WITNESS.is_file(), (
        f"{WITNESS.name} is the committed witness; without it the comment's figures "
        "are a claim about a file nobody can read")
    rows = [json.loads(line) for line in WITNESS.read_text().splitlines() if line.strip()]
    assert len(rows) == 11, f"the comment counts eleven rows, the file holds {len(rows)}"
    assert sum(1 for r in rows if r.get("event") == "vault_review") == 9
    assert sum(1 for r in rows if r.get("event") == "review") == 2
    for row in rows:
        flat = json.dumps(row)
        assert "output truncated at 8192 tokens" in flat, row.get("created_at")
        assert "raise harness.finalizer.max_tokens" in flat, (
            "a row in here is not the defect this file is about: "
            + str(row.get("created_at")))
    graded = [r for r in rows if r.get("event") == "vault_review"]
    assert all(r.get("kind") == "skipped" for r in graded)
    assert all(r.get("clauses") == [] for r in graded), (
        "a vault review that graded clauses is in the truncation witness")
    assert len({r.get("item_id") for r in graded}) == 9, "one row per item expected"

    listed = re.search(r"\(items ([\d, ]+)\)", _comment_above("REVIEW_SCHEMA: dict"))
    assert listed, "the comment no longer lists the items behind the count"
    assert sorted(int(n) for n in listed.group(1).split(",")) == \
        sorted(int(r["item_id"]) for r in graded), (
        "the comment's item list and the committed rows disagree")


def test_the_review_payload_carries_the_capped_schema(monkeypatch):
    """What this pins, and what it deliberately does not.

    Pinned: the dict `_grade_once` puts into the `final_schema` field of the POST to
    `{backend}/api/message/stream` IS `REVIEW_SCHEMA`, object-identical, and that
    object measures bounded. Before this node the only schema-identity assertion in
    the suite was the confirm leg's, so a capped module and an uncapped request could
    have coexisted.

    NOT pinned, and said plainly because the node's name used to imply it:
    `_post_stream` is monkeypatched, so no bytes leave the process and whether the
    engine's guided decoder actually honours each `maxLength` is not testable from
    here. That half rests on `app/harness/finalizer.py`'s boundedness measurement of
    2026-09-28 and on #2240's owed 7-day re-count of the truncation rows, which is
    the only observation that can show a 4-5 clause grading now fits."""
    captured: dict = {}

    def fake_post(url, payload, timeout):
        captured["url"] = url
        captured["payload"] = payload
        yield ("done", {"response": "{}", "stop_reason": "stop"})

    monkeypatch.setattr(RV, "_post_stream", fake_post)
    report: dict = {"text": "", "stop_reason": "", "structured": None,
                    "structured_error": ""}
    RV._grade_once(report, backend="http://backend.invalid", payload_prompt="grade this",
                   session_id="s bounds", timeout=5.0, model="m", max_turns=1)

    assert captured.get("url", "").endswith("/api/message/stream"), captured.get("url")
    sent = captured["payload"]["final_schema"]
    assert sent is RV.REVIEW_SCHEMA, "the wire carries a different schema than the capped one"
    assert _schema_is_bounded(sent) is True, (
        "the schema reaching the engine still has an open string in it")
