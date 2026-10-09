"""Grammar bounds on the schemas the automod graders send to the finalizer.

`app.harness.finalizer.run_finalizer` splits a completion it cannot parse into
two diagnoses, and which one it gives is decided entirely by the schema it was
handed (`app/harness/finalizer.py`, `_schema_is_bounded`): when every string the
schema admits carries a positive `maxLength` it reports `generation diverged at N
tokens … not a budget`, and when any string is open it reports `output truncated …
raise harness.finalizer.max_tokens` — advice to edit config.yaml. The two messages are
`DIVERGENCE_MARKER` and its opposite (`app/harness/finalizer.py:314-325`), and quoting
one where you mean the other is the same false lead #1706 removes, which is why the
nodes below name the branch they expect. #1706 established that the second message is a lie
when the grammar could have capped the runaway itself
(`app/harness/tests/test_finalizer.py::test_a_cut_under_a_schema_that_caps_every_field_is_a_divergence`),
and #2240 applies it to `REVIEW_SCHEMA`, the review grader's own answer shape.

The defect this file pins is the one the item's own fix would have missed:
`_schema_is_bounded` requires EVERY property bounded, and capping the three
clause-row fields the item named (`note`, `evidence_path`, `test_node_id`) still
measures False: `REVIEW_SCHEMA` has thirteen string leaves and #2240 capped all
thirteen, so the three named fields were a quarter of the grammar. The control node
below keeps that under-scoped fix from coming back, and `EIGHT` here is what the same
shape looks like on the eight schemas #2444 closes.
"""
from __future__ import annotations

import copy
import importlib
import inspect
import json
import pathlib
import re
from pathlib import Path

import pytest

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


def _leaf_node(schema: dict, path: str) -> dict:
    """The node a `string_leaves` path names, walked one token per step.

    `[]` is a step, not decoration on the name before it. For
    `acceptance_clauses[]` the cap sits on the array's `items` node —
    `{"type": "string", "maxLength": 900}` — while the array node itself holds
    only `type`, `items` and a description. Splitting the path on `. | []` and
    dropping the empties made that path a single name, so both readers of it
    landed on the array node: setting a cap there wrote a key the grammar never
    reads, and popping one removed nothing and left the copy byte-identical,
    which is how a node in this file came to assert something it could not fail
    (#2444 review). The walk is shared with `_cap_field` so the two cannot drift
    apart again, and the caller's `KeyError` on a missing `properties` is the
    path being wrong, which is what a fixture mistake should look like.
    """
    node = schema
    for token in re.findall(r"[^\.\[\]]+|\[\]", path):
        node = node["items"] if token == "[]" else node["properties"][token]
    return node


def _cap_field(schema: dict, path: str, value: int) -> None:
    """Set `maxLength` on the leaf at a `string_leaves` path."""
    _leaf_node(schema, path)["maxLength"] = value


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


# ── the eight schemas capped by #2444 ───────────────────────────────────────
# The nodes above bound REVIEW_SCHEMA (#2240); these bound the other eight in the
# same shape — cap read against the reader's slice, never restated.

EIGHT = {
    "IMPLEMENT_OUTCOME": "scripts.automod.backlog:IMPLEMENT_OUTCOME_SCHEMA",
    "TRIAGE_VERDICT": "scripts.automod.backlog:TRIAGE_VERDICT_SCHEMA",
    "GROUP_TRIAGE": "scripts.automod.backlog:GROUP_TRIAGE_SCHEMA",
    "SWEEP": "scripts.automod.backlog:SWEEP_SCHEMA",
    "CONFIRM": "scripts.automod.review:CONFIRM_SCHEMA",
    "ARCH_REVIEW": "workers.sources.arch_review:ARCH_REVIEW_SCHEMA",
    "OWED": "workers.sources.owed_check:OWED_SCHEMA",
    "DIGEST_RESULT": "workers.sources.youtube_digest:RESULT_SCHEMA",
}

# The three shapes `_schema_is_bounded` must keep answering False for: an open
# grammar, which is what the advice branch at `finalizer.py:314` exists for.
CONTROLS = (
    {},
    {"type": "object"},
    {"type": "object", "description": "x"},
)


def _schema(spec: str) -> dict:
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


def _slice_max(module_qual: str, needle: str) -> int:
    """The largest `[:N]` on any line of that module containing `needle` — read from
    the module, so growing a reader's clamp past a cap fails here instead of quietly
    handing the cut to the grammar. A needle matching nothing fails, never skips.
    """
    text = pathlib.Path(importlib.import_module(module_qual).__file__).read_text()
    hits = [int(n) for line in text.splitlines() if needle in line
            for n in re.findall(r"\[\s*:\s*(\d+)\s*\]", line)]
    assert hits, f"nothing in {module_qual} matches {needle!r} any more"
    return max(hits)


def _open_leaves(schema: dict) -> set[str]:
    """String leaves neither capped nor enum-bound: the grammar's holes. An `enum`
    is already finite, so counting one as open would make the check unfalsifiable.
    """
    return {p for p, node in string_leaves(schema).items()
            if "enum" not in node
            and not (isinstance(node.get("maxLength"), int) and node["maxLength"] > 0)}


def _capped_leaves(schema: dict) -> set[str]:
    return {p for p, node in string_leaves(schema).items()
            if isinstance(node.get("maxLength"), int)}


def _cap(schema: dict, path: str) -> int:
    node = string_leaves(schema)[path]
    value = node.get("maxLength")
    assert isinstance(value, int) and value > 0, f"{path} carries {value!r}"
    return value


# (schema, leaf, reader module, needle naming that field's slice line): the cap must
# be STRICTLY above the `[:N]` that reader applies.
EIGHT_SLICED = (
    ("IMPLEMENT_OUTCOME", "clause_outcomes[].evidence",
     "scripts.automod.backlog", 'str(raw.get("evidence") or "").split())'),
    ("IMPLEMENT_OUTCOME", "summary",
     "scripts.automod.backlog", 'str(structured.get("summary") or "").split())'),
    ("IMPLEMENT_OUTCOME", "human_paths[].path",
     "scripts.automod.backlog", 'str(raw.get("path") or "").split())'),
    ("IMPLEMENT_OUTCOME", "human_paths[].reason",
     "scripts.automod.backlog", 'str(raw.get("reason") or "").split())'),
    ("TRIAGE_VERDICT", "check",
     "workers.sources.autotriage", 'str(obj.get("check") or "").split())'),
    ("TRIAGE_VERDICT", "evidence",
     "workers.sources.autotriage", 'str(obj.get("evidence") or "").strip()'),
    ("TRIAGE_VERDICT", "acceptance",
     "workers.sources.autotriage", '_acceptance_text(str(obj.get("acceptance") or ""))'),
    ("GROUP_TRIAGE", "items[].evidence",
     "workers.sources.autotriage", 'str(raw.get("evidence") or "").split())'),
    ("SWEEP", "items[].evidence",
     "workers.sources.autotriage", 'str(raw.get("evidence") or "").split())'),
    ("CONFIRM", "reason",
     "scripts.automod.review", 'str(ans.get("reason") or "no reason given")'),
    ("ARCH_REVIEW", "summary",
     "workers.sources.arch_review", 'raw["summary"]'),
    ("OWED", "entries[].artifact",
     "workers.sources.owed_check", 'str(raw.get("artifact") or "").split())'),
    ("OWED", "entries[].recheck_after",
     "workers.sources.owed_check", 'str(raw.get("recheck_after") or "")'),
    ("OWED", "entries[].outside",
     "workers.sources.owed_check", 'str(raw.get("outside") or "").split())'),
    ("OWED", "entries[].follow_up.name",
     "workers.sources.owed_check", 'str(follow.get("name") or "").split())'),
    ("OWED", "entries[].follow_up.body",
     "workers.sources.owed_check", 'str(follow.get("body") or "")'),
    ("OWED", "entries[].amend_clause.text",
     "workers.sources.owed_check", 'str(amend.get("text") or "").split())'),
    ("OWED", "summary",
     "workers.sources.owed_check", 'str(structured.get("summary") or "").split())'),
)

# (schema key, leaf path, reader module, constant) for the fields whose reader
# clamps with a module constant rather than a literal `[:N]`.
EIGHT_CONSTANTS = (
    ("OWED", "entries[].evidence", "workers.sources.owed_check", "_FIELD_CEILING"),
    ("OWED", "entries[].ruling", "workers.sources.owed_check", "_FIELD_CEILING"),
    # Clause lists go through `clean_clauses`, whose slice is the named constant.
    ("TRIAGE_VERDICT", "acceptance_clauses[]",
     "scripts.automod.backlog", "CLAUSE_MAX_CHARS"),
    ("TRIAGE_VERDICT", "human_clauses[]",
     "scripts.automod.backlog", "CLAUSE_MAX_CHARS"),
    ("GROUP_TRIAGE", "umbrella.acceptance_clauses[]",
     "scripts.automod.backlog", "CLAUSE_MAX_CHARS"),
)

# String leaves with no Python slice to read a cap off, pinned at the declared value
# so a re-size is a decision with a diff rather than a drift.
EIGHT_WITHOUT_A_SLICE = (
    ("GROUP_TRIAGE", "umbrella.check"),
    ("GROUP_TRIAGE", "umbrella.evidence"),
    ("GROUP_TRIAGE", "umbrella.acceptance"),
    ("DIGEST_RESULT", "note"),
    ("DIGEST_RESULT", "idea"),
    ("DIGEST_RESULT", "duplicate_of"),
    ("DIGEST_RESULT", "filed"),
    ("DIGEST_RESULT", "papers"),
)


def _check_eight_caps_above_reader_slices(table, resolver=None) -> list[str]:
    """Every entry's grammar cap must be strictly above its reader's slice."""
    resolve = resolver or _schema
    bad = []
    for key, path, module_qual, needle in table:
        cap = _cap(resolve(EIGHT[key]), path)
        sliced = _slice_max(module_qual, needle)
        if not cap > sliced:
            bad.append(f"{key}.{path}: cap {cap} does not sit above the "
                       f"{module_qual} slice [{sliced}]")
    return bad


def _check_caps_above_constant_slices(table) -> list[str]:
    bad = []
    for key, path, module_qual, const in table:
        cap = _cap(_schema(EIGHT[key]), path)
        ceiling = getattr(importlib.import_module(module_qual), const)
        if not cap > ceiling:
            bad.append(f"{key}.{path}: cap {cap} does not sit above "
                       f"{module_qual}.{const} = {ceiling}")
    return bad


def test_all_eight_verdict_schemas_are_bounded_and_their_controls_are_not():
    """Clause 1: True for each of the eight, False for the three controls. The
    controls are the shapes that may legitimately need more room, so they keep the
    budget advice at `finalizer.py:314`; a control reading bounded would rewrite which
    failures get told "raise the budget", and that split is the whole of #1706.
    """
    from app.harness.finalizer import _schema_is_bounded as bounded
    for key, spec in EIGHT.items():
        schema = _schema(spec)
        open_holes = sorted(_open_leaves(schema))
        assert not open_holes, (
            f"{key} leaves the grammar open at {open_holes}: an enum is finite by its "
            "choices, but a free-text leaf with no cap is exactly what the else branch "
            "at finalizer.py:314 reports as a budget problem")
        assert bounded(schema), (
            f"{key} reads as unbounded, so a degenerate {key} completion is told to "
            "raise harness.finalizer.max_tokens — the false lead #1706 exists to stop")
    for control in CONTROLS:
        assert not bounded(control), (
            f"the control {control} now reads as bounded: an open grammar would stop "
            "being reportable as needing more room, and #1706's advice branch dies")


def test_opening_any_single_leaf_of_the_eight_makes_it_read_unbounded():
    """The can-fail half of the node above, one leaf at a time, across all 31.

    Stripping every cap is not the test: the eight contain `enum` leaves, and an enum
    is finite on its own, so a fully-stripped copy stays bounded by its vocabularies
    and would pass for the wrong reason. Opening ONE free-text leaf at a time is the
    case where `_schema_is_bounded` has to notice.

    No leaf is held out, including the three array-item caps. An earlier version of
    this node skipped them on the claim that the predicate never sees an array's
    `items`; that was measured false — `_node_is_bounded` recurses through `items`
    (finalizer.py:123-127) and removing one of those caps flips the schema to False,
    which the node below pins in the open. Holding them out hid a walker bug that
    made the hold-out itself unfalsifiable (#2444 review).
    """
    from app.harness.finalizer import _schema_is_bounded as bounded
    assert _ARRAY_ITEM_LEAVES <= {(k, p) for k, sp in EIGHT.items()
                                  for p in _capped_leaves(_schema(sp))}
    checked = 0
    for key, spec in EIGHT.items():
        schema = _schema(spec)
        assert bounded(schema), key
        for path in sorted(_capped_leaves(schema)):
            opened = copy.deepcopy(schema)
            _open_one_leaf(opened, path)
            assert opened != schema, (
                f"opening {key}.{path} changed nothing, so the leaf walk is wrong and "
                "this node would pass without testing anything")
            assert not bounded(opened), (
                f"{key}.{path} has its cap removed and the schema still reads as "
                "bounded: the predicate is not looking at this leaf")
            checked += 1
    assert checked == 31, (
        f"{checked} leaves checked, not 31 — the eight schemas have changed shape, so "
        "every field table in this file needs re-counting against them")


# The three caps that sit on the `items` node of an array of strings, so a reader
# looking at the property sees `{"type": "array", "items": {...900}}` and no figure.
_ARRAY_ITEM_LEAVES = {
    ("TRIAGE_VERDICT", "acceptance_clauses[]"),
    ("TRIAGE_VERDICT", "human_clauses[]"),
    ("GROUP_TRIAGE", "umbrella.acceptance_clauses[]"),
}


def test_an_array_of_strings_is_bounded_only_by_the_cap_on_its_items():
    """The three `..._clauses[]` caps are load-bearing, and they live one level down.

    `_node_is_bounded` reaches an array's strings by recursing into `items`
    (finalizer.py:123-127), so a list of uncapped strings is the open grammar
    `TRIAGE_VERDICT_SCHEMA`'s `acceptance_clauses` was when #1706 was filed — the
    node above grades those three leaves exactly like the other 27, and the copy it
    hands the predicate has to differ from the original for that to mean anything.
    Pinned separately because the shape is the easy one to get wrong: the array node
    carries no `maxLength`, so a walker that stops one level too high removes
    nothing, and that is the bug this node's predecessor was written to excuse.
    """
    from app.harness.finalizer import _schema_is_bounded as bounded
    assert _ARRAY_ITEM_LEAVES, "the three array-item leaves are the point of this node"
    for key, path in sorted(_ARRAY_ITEM_LEAVES):
        schema = _schema(EIGHT[key])
        array_node = _leaf_node(copy.deepcopy(schema), path[:len(path) - 2] or path)
        assert "maxLength" not in array_node, (
            f"{key}.{path}: the ARRAY node now carries a cap, so the cap this file "
            "sizes has moved and the tables describing it are stale")
        opened = copy.deepcopy(schema)
        _open_one_leaf(opened, path)
        assert opened != schema, f"{key}.{path}: opening removed nothing"
        assert not bounded(opened), (
            f"{key}.{path} lost its item cap and the schema still reads as bounded: "
            "the item cap is not what bounds these arrays, so this file's tables "
            "are sizing a value the decoder never sees")


def test_every_capped_leaf_of_the_eight_is_graded_by_one_of_the_three_tables():
    """Every capped leaf is graded by one of the three tables. The tables are lists a
    person edits, so this guards a new capped leaf no relation node looks at — a cap
    that could drift below its slice with nothing red anywhere.
    """
    graded = {(key, path) for key, path, _m, _n in EIGHT_SLICED}
    graded |= {(key, path) for key, path, _m, _c in EIGHT_CONSTANTS}
    graded |= set(EIGHT_WITHOUT_A_SLICE)
    actual = {(key, path) for key, spec in EIGHT.items()
              for path in _capped_leaves(_schema(spec))}
    assert graded == actual, (
        f"graded but no longer a leaf: {sorted(graded - actual)}; a capped leaf "
        f"nobody grades: {sorted(actual - graded)}")


def _open_one_leaf(schema: dict, path: str) -> None:
    """Remove `maxLength` from the leaf at a `string_leaves` path.

    The assert is the honesty rail, and it exists because of a concrete failure:
    the walker used to stop at the array node for a path ending in `[]`, where
    there is no `maxLength` to remove, so every "opened" copy of
    `acceptance_clauses[]`, `human_clauses[]` and `umbrella.acceptance_clauses[]`
    came back identical to the original and the node that graded them could not
    fail whatever the schemas did (#2444 review). Removing nothing is now an
    error at the helper, not a green test.
    """
    node = _leaf_node(schema, path)
    assert "maxLength" in node, (
        f"opening {path} reached a node with no cap to remove ({sorted(node)}), "
        "which means the walk is not landing on the leaf the cap lives on")
    node.pop("maxLength")


def test_every_cap_of_the_eight_schemas_sits_above_the_slice_its_parser_applies():
    """Clause 2, per field: cap > the `[:N]` the reader applies, read off the reader.
    A cap at or below the slice trades a loud truncation for a silent one. Same
    relation `test_every_cap_sits_at_the_slice_parsereview_already_applies` pins for
    `REVIEW_SCHEMA`, over every sliced leaf of the eight.
    """
    bad = _check_eight_caps_above_reader_slices(EIGHT_SLICED)
    bad += _check_caps_above_constant_slices(EIGHT_CONSTANTS)
    assert not bad, "cap at or below its reader's slice:\n  " + "\n  ".join(bad)


def test_a_cap_below_a_parses_slice_is_detected():
    """The can-fail half: lower one cap below its slice, the relation reports it."""
    schema = copy.deepcopy(_schema(EIGHT["IMPLEMENT_OUTCOME"]))
    _cap_field(schema, "clause_outcomes[].evidence", 10)   # the slice is [:300]
    table = [("IMPLEMENT_OUTCOME", "clause_outcomes[].evidence",
              "scripts.automod.backlog", 'str(raw.get("evidence") or "").split())')]
    saved = _schema.__globals__["_schema"]

    def patched(spec):
        return schema if spec == EIGHT["IMPLEMENT_OUTCOME"] else saved(spec)
    try:
        _schema.__globals__["_schema"] = patched
        bad = _check_eight_caps_above_reader_slices(table)
    finally:
        _schema.__globals__["_schema"] = saved
    assert len(bad) == 1 and "clause_outcomes[].evidence" in bad[0], bad
    assert "does not sit above" in bad[0], bad


def test_the_seven_leaves_with_no_reader_slice_are_pinned_at_their_declared_caps():
    """The seven leaves whose reader applies no `[:N]`, so no cap can be read off a
    slice and nothing here pretends they were measured that way: the group run's
    umbrella strings are stored uncut, and `_shape` filters against enums instead of
    slicing. Their caps are chosen, so each is pinned at the declared value — a
    re-size has to change the constant and this file's reason with it.
    """
    declared = {
        ("GROUP_TRIAGE", "umbrella.check"): 600,
        ("GROUP_TRIAGE", "umbrella.evidence"): 3000,
        ("GROUP_TRIAGE", "umbrella.acceptance"): 4500,
        ("DIGEST_RESULT", "note"): 400,
        ("DIGEST_RESULT", "idea"): 400,
        ("DIGEST_RESULT", "duplicate_of"): 20,
        ("DIGEST_RESULT", "filed"): 20,
        ("DIGEST_RESULT", "papers"): 600,
    }
    assert set(declared) == set(EIGHT_WITHOUT_A_SLICE)
    for (key, path), value in sorted(declared.items()):
        assert _cap(_schema(EIGHT[key]), path) == value, (
            f"{key}.{path} moved off {value} without this node's reason moving with it")


# ── the comment above each of the eight (#2444 clause 5) ────────────────────

#: (schema key, `def` marker, module). The `#` block above each marker is what a
#: reader of that schema actually reads.
EIGHT_MARKERS = (
    ("IMPLEMENT_OUTCOME", "IMPLEMENT_OUTCOME_SCHEMA: dict", "scripts.automod.backlog"),
    ("TRIAGE_VERDICT", "TRIAGE_VERDICT_SCHEMA: dict", "scripts.automod.backlog"),
    ("GROUP_TRIAGE", "GROUP_TRIAGE_SCHEMA: dict", "scripts.automod.backlog"),
    ("SWEEP", "SWEEP_SCHEMA: dict", "scripts.automod.backlog"),
    ("CONFIRM", "CONFIRM_SCHEMA: dict", "scripts.automod.review"),
    ("ARCH_REVIEW", "ARCH_REVIEW_SCHEMA: dict", "workers.sources.arch_review"),
    ("OWED", "OWED_SCHEMA = {", "workers.sources.owed_check"),
    ("DIGEST_RESULT", "RESULT_SCHEMA: dict", "workers.sources.youtube_digest"),
)


def _comment_above_in(module_qual: str, marker: str) -> str:
    """The contiguous `#` block directly above `marker` in that module.

    Same shape as `_comment_above` above, which reads only `review.py`: the eight
    live in five modules, and a comment that names a cap is only worth what a reader
    finds where the schema is.
    """
    path = pathlib.Path(importlib.import_module(module_qual).__file__)
    lines = path.read_text().splitlines()
    at = next((i for i, line in enumerate(lines) if line.startswith(marker)), None)
    if at is None:
        return ""
    out: list[str] = []
    i = at - 1
    while i >= 0 and lines[i].lstrip().startswith("#"):
        out.append(lines[i])
        i -= 1
    return "\n".join(reversed(out))


def test_the_comment_above_each_of_the_eight_names_a_cap_and_the_runaway():
    """Clause 5, for all eight: the comment above each schema states a numeric cap and
    the runaway it stops, as `test_the_comment_above_the_schema_names_the_cap_and_the_
    runaway` requires of `REVIEW_SCHEMA`.

    The two comments this round replaced said the opposite — `backlog.py`'s header
    that "Length clamps stay in Python" and `parse_outcome`'s docstring that the
    triage schema "carries no `maxLength`" — and both were the argument #1706 already
    answered. A comment that only decorates is what gets the caps stripped again by
    the next pass, so the figure has to be one THIS schema carries.

    The figure is compared against the schema, not against prose: the first version of
    this node accepted any parenthesised two-to-four-digit figure, which the review
    caught as an assertion that could not fail on the half it was written for — the
    runaway alone, `(8192)`, satisfies that pattern, and `IMPLEMENT_OUTCOME` passed on
    a block quoting nothing else. So the caps come from
    `_capped_leaves` on the live schema and at least one of them has to appear as a
    number in the block, with 8192 still required as the separate half that says what
    a stripped cap costs.
    """
    for key, marker, module_qual in EIGHT_MARKERS:
        comment = _comment_above_in(module_qual, marker)
        assert comment, f"no comment above {key} in {module_qual} to grade"
        schema = _schema(EIGHT[key])
        caps = sorted({_cap(schema, path) for path in _capped_leaves(schema)})
        assert caps, f"{key} carries no caps to name, so this node has nothing to grade"
        named = [c for c in caps if re.search(rf"\b{c}\b", comment)]
        assert named, (
            f"{key}'s comment names none of the {len(caps)} caps its schema actually "
            f"carries ({caps}) — quoting the 8192 budget is quoting the runaway, not a "
            "cap, and a reader at the schema still cannot see what the fields allow")
        assert "8192" in comment, (
            f"{key}'s comment does not name the runaway the caps stop "
            "(harness.finalizer.max_tokens is 8192), which is the half that tells a "
            "reader what removing a cap costs")
        # Wording, not spelling: every one of these blocks says "cap"/"caps" rather
        # than repeating the key name, and the teeth of this node are the comparison
        # above — a figure that matches a cap the schema really carries.
        assert "maxLength" in comment or "cap" in comment.lower(), (
            f"{key}'s comment names a figure without saying what carries it")


def test_the_two_clamps_stay_in_python_comments_are_gone():
    """Clause 5's other half: the two comments #2444 was filed to replace no longer
    forbid the caps. Both survived every earlier pass because they were prose — `git
    grep maxLength` found the schemas and missed the sentences, so each round
    re-derived the same conclusion. This node stops the sentence coming back.
    """
    src = pathlib.Path(importlib.import_module("scripts.automod.backlog").__file__) \
        .read_text()
    assert "Length clamps stay in Python" not in src, (
        "the header ban is back: it is the sentence that kept the eight schemas open")
    assert "carries no `maxLength`" not in src, (
        "parse_outcome's docstring re-forbade the caps it now sits under")
    docstring = src[src.index("def parse_outcome("):src.index("def settle_item_verdict(")]
    assert "8192" in docstring and "harness.finalizer.max_tokens" in docstring, (
        "the reader explains no purpose for the caps, so the comment is decoration")


# ── the caller → finalizer seam for the eight (#2444 review, seams finding) ──
#
# `test_the_review_payload_carries_the_capped_schema` is the only node in the suite that
# crosses a caller's boundary into a request body, and it crosses review.py's. The eight
# all enter through `run_finalizer(final_schema=…)` in a worker module instead, so a
# capped module and an uncapped hand-off could have coexisted — which is what the
# previous round's `seams_unverified` named.

EIGHT_CALL_SITES = (
    ("workers.sources.autotriage", "B.TRIAGE_VERDICT_SCHEMA", "TRIAGE_VERDICT"),
    ("workers.sources.autotriage", "B.GROUP_TRIAGE_SCHEMA", "GROUP_TRIAGE"),
    ("workers.sources.autotriage", "B.SWEEP_SCHEMA", "SWEEP"),
    ("workers.sources.autocode", "B.IMPLEMENT_OUTCOME_SCHEMA", "IMPLEMENT_OUTCOME"),
    ("scripts.automod.review", "CONFIRM_SCHEMA", "CONFIRM"),
    ("workers.sources.arch_review", "ARCH_REVIEW_SCHEMA", "ARCH_REVIEW"),
    ("workers.sources.owed_check", "OWED_SCHEMA", "OWED"),
    ("workers.sources.youtube_digest", "RESULT_SCHEMA", "DIGEST_RESULT"),
)


@pytest.mark.parametrize("module_qual,expr,key", EIGHT_CALL_SITES)
def test_the_schema_a_worker_hands_the_finalizer_is_the_capped_object(
        module_qual: str, expr: str, key: str):
    """What this pins: the expression a worker's `final_schema=` carries names THIS
    capped object, object-identical, and that object measures bounded.

    What it does not pin, said plainly because the name says "hands", not "sends": no
    bytes are captured here. Whether the engine's guided decoder honours each
    `maxLength` on the wire is the same half `test_the_review_payload_carries_the_capped_
    schema` declines, and rests on `app/harness/finalizer.py`'s boundedness measurement
    and #2240's owed re-count, not on this node.
    """
    module = importlib.import_module(module_qual)
    source = pathlib.Path(module.__file__).read_text()
    site = next((line for line in source.splitlines()
                 if "final_schema=" in line and expr in line), None)
    assert site is not None, (
        f"{module_qual} no longer passes {expr} at a final_schema= call, so the cap "
        f"pinned on {EIGHT[key]} is not the schema that worker sends")
    # `B.` is how autotriage and autocode spell the backlog module, and in both it is a
    # function-local import (`from scripts.automod import backlog as B`), so resolving
    # the alias through `vars(module)` raises NameError — the alias exists at the call
    # site's scope, not the module's. The attribute after the dot is what names the
    # object either way.
    attribute = expr.rpartition(".")[2]
    holder = (importlib.import_module("scripts.automod.backlog")
              if expr.startswith("B.") else module)
    handed = getattr(holder, attribute)
    assert handed is _schema(EIGHT[key]), (
        f"{module_qual} hands a copy or a rebuilt dict to the finalizer, not the object "
        f"whose caps the suite measures: {site.strip()}")
    assert _schema_is_bounded(handed) is True, (
        f"{key} reaches the finalizer with an open string in it")
