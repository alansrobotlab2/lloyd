"""The `arch-review:` comment in `config.yaml` must describe the rails the source
runs today, not the ones it ran when the block was written.

The block came in with f80c9d00 (2026-09-11) and `git log -S"skills/ and autonomy/"
-- config.yaml` returned that one commit and nothing after it: the sentence sitting
over the knobs a person actually edits (`enabled`, `daily_max`,
`max_delta_lines`) was never touched, while the code it tunes moved twice.

- **#709** turned the vault sweep inside out. It used to be an allow-list, so
  anything outside the named prefixes was unprotected *by omission*; it is now a
  deny-list — the sweep subtracts `VAULT_UNSWEPT_PREFIXES` (today
  `("backlog/",)`) from the whole vault, so `knowledge/`, `memory/`, `plans/` and
  the daily notes are all swept. The comment named two directories — `skills/` and
  `autonomy/` — which reads as a claim that the rest of the vault survives a review
  turn. It does not.
- **The same change's other half went the opposite way.** A vault `autonomy/` path
  is the autonomy scheduler's own bookkeeping, so `revert_strays` reports it with
  action `kept (scheduler-owned)` — driven by `SCHEDULER_OWNED_VAULT_PREFIXES` —
  instead of restoring it, because checking out a task file un-runs a finished run
  and re-arms the task (the file's own example is task #82, which dispatched twice
  on the night of 2026-09-08). The comment filed that path under "reverts", which
  is the opposite verdict to the one the code reaches.
- **b377bea5 (2026-10-25, #3338)** added `keep_rejected` and the shortened rest
  (`REJECT_REST_SECONDS`, `MAX_REJECT_REQUEUES`, written by `_rest_fields`), so a
  bound refuses the *commit* and keeps the *correction*. "throws away a diff" is
  the pre-#3338 behaviour, and the reason it was removed is the 2026-09-25 Ingest
  corrections, which were refused for one hunk and survived nothing.

`architecture/arch-review.md` carries the current version of all three clauses and
is guarded by `tests/test_architecture_coverage_doc_claims.py`. `config.yaml` is the
surface a person reads before changing a knob and nothing guarded it — the #1759
asymmetry again, one surface corrected and the other left to rot.

The ban reads the **raw bytes** of `config.yaml`. `yaml.safe_load` drops every
comment, so a test written against the parsed mapping would pass whatever this
comment said; `test_the_parsed_mapping_cannot_see_the_block_at_all` is the standing
control for that, and `test_config_yaml_still_parses` is the one for the other
direction — a comment edit that mangles the file.
`test_the_owners_the_block_names_say_what_the_block_says` closes the seam the
rewrite opens: the comment now points
at code owners instead of listing directories, so the test checks the owners still
say what the comment claims, which is the failure mode of naming an owner and then
letting it move.

Style follows `tests/test_stale_mtls_comment_claims.py`: name the corpus, assert it
is present and non-empty before anything is searched, and put a positive control
beside every zero-hit ban so an empty result can only mean clean prose.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config.yaml"
MODULE = REPO / "workers" / "sources" / "arch_review.py"

#: The two clauses the sentence carried that the code no longer honours. Banned
#: across the whole raw file, not just this block: the fix is not a rewrite of one
#: paragraph, it is that config.yaml must not contain either claim anywhere.
STALE_PHRASES = (
    "skills/ and autonomy/",
    "throws away a diff",
)

#: What the ban fires on, kept here so the positive control and the ban are the
#: same code path — a control that searches differently than the ban proves nothing.
def _stale_hits(text: str) -> list[str]:
    return [p for p in STALE_PHRASES if p in text]


def _raw() -> str:
    """The raw text of config.yaml.

    Bytes-then-decode rather than `read_text()` so the file's existence and
    readability are the same failure the ban would report, and the size assertion
    below is about the same bytes the ban searched.
    """
    assert CONFIG.is_file(), f"{CONFIG} is not a file — this test would vacuously pass"
    text = CONFIG.read_bytes().decode("utf-8")
    assert len(text) > 100_000, (
        f"config.yaml read back as {len(text)} chars; the file is 163,101 bytes with "
        "#4059's rewrite in it, so this is a truncated read or the wrong file, and a "
        "zero-hit ban would mean nothing")
    assert "arch-review:" in text, (
        "the corpus does not contain the arch-review block at all: reading the wrong "
        "file, not a clean config")
    return text


def _block() -> str:
    """The comment block immediately above the `arch-review:` mapping key.

    The ban is over the whole file, but the positive clauses have to be about
    THIS comment — a rewrite that moved the prose into an unrelated comment, or
    into the architecture doc, would satisfy a `raw_text contains` assertion while
    leaving the block over the knobs still wrong. So: walk backwards from the key
    over contiguous comment lines and return them with the `# ` stripped, joined
    into one string so sentence-level assertions see the prose as a reader does.
    """
    lines = _raw().splitlines()
    try:
        key = next(i for i, l in enumerate(lines) if l == "    arch-review:")
    except StopIteration:
        raise AssertionError("no `    arch-review:` key line in config.yaml") from None
    start = key
    while start > 0 and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    body = [l.lstrip()[1:].lstrip() for l in lines[start:key]]
    assert len(body) >= 20, (
        f"the comment above `arch-review:` is {len(body)} lines; #4059 left 28, so the "
        "block has been cut down or detached from its key, and a reader of the knobs "
        "would not see the rails")
    joined = " ".join(body)
    assert "architecture review unit" in joined, (
        "the block above the key is not the arch-review prose this test is about — "
        "the positive assertions below would be grading some other comment")
    return joined


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _sentence_naming(token: str) -> str:
    """One sentence of the block that names `token`, or a failure.

    Per sentence, not per block: the clauses of #4059 are each satisfied by ONE
    sentence saying one thing. A block that mentions the sweep in one sentence and
    `autonomy/` in another has stated neither direction.
    """
    hits = [s for s in _sentences(_block()) if token in s]
    assert hits, f"no single sentence of the arch-review block names {token!r}"
    assert len(hits) == 1, (
        f"{len(hits)} sentences name {token!r}; the block re-states it, so which one "
        "carries the claim is ambiguous")
    return hits[0]


# --- clause 1: the two stale clauses are gone from the whole file -------------

def test_the_two_stale_rail_claims_are_gone_from_config():
    """`skills/ and autonomy/` and `throws away a diff` appear nowhere in the raw
    file. The first is the pre-#709 allow-list; the second is pre-#3338. Both are
    banned file-wide rather than pinned out of one block, because the rot this
    found was a comment that drifted away from the code it tunes and nothing
    noticed: a re-wording that keeps the claim in another sentence is the same bug.
    """
    assert _stale_hits(_raw()) == [], (
        "config.yaml still claims the rail reverts the vault's skills/ and autonomy/ "
        "list, or that a bound throws a diff away — see the module docstring for which "
        "code moved and when")


def test_the_ban_is_not_vacuous():
    """The positive control #1823 asks for: the same `_stale_hits` call that
    returns `[]` above must fire on the sentence it is meant to catch, verbatim.
    Without this, a renamed constant or a typo in `STALE_PHRASES` turns clause 1
    into a permanent pass, which is exactly how the comment came to be trusted.
    """
    old_sentence = (
        "The source reverts every other write in this repo and in the vault's "
        "skills/ and autonomy/, throws away a diff that breaks its bounds, and "
        "commits the doc to main itself — the model never runs git.")
    assert _stale_hits(old_sentence) == list(STALE_PHRASES), (
        "the banned phrases no longer match the sentence they were written to ban, "
        "so clause 1 has stopped meaning anything")


# --- clause 2: owners, not a directory list ----------------------------------

def test_the_block_names_both_code_owners():
    """The rewrite points at `VAULT_UNSWEPT_PREFIXES` and `keep_rejected` instead
    of re-listing directories. The standing rule behind #4059 is "name the owner
    instead of the list", because a list rots the moment the constant moves and a
    reader cannot tell which is stale.
    """
    block = _block()
    for owner in ("VAULT_UNSWEPT_PREFIXES", "keep_rejected"):
        assert owner in block, (
            f"the arch-review block no longer names {owner}; without the owner the "
            "comment is a list again, and lists are what rotted")


def test_the_owners_the_block_names_say_what_the_block_says():
    """The seam the rewrite opens: a comment that points at a constant is only
    true while the constant agrees. Read the module's own AST and check the four
    values the block quotes — the exempt prefixes, the scheduler-owned prefix, the
    rest interval and the re-queue cap — so a later change to any of them is a red
    node naming the comment, not another six weeks of a confident wrong sentence.
    """
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    consts: dict[str, object] = {}
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            consts[node.targets[0].id] = node.value
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)

    def literal(key: str):
        assert key in consts, f"{key} is not a module-level constant in arch_review.py"
        return ast.literal_eval(consts[key])

    assert literal("VAULT_UNSWEPT_PREFIXES") == ("backlog/",), (
        "the block says backlog/ is the only exempt prefix; that is now false, so the "
        "coverage sentence and the test have to be re-read together")
    assert literal("SCHEDULER_OWNED_VAULT_PREFIXES") == ("autonomy/",), (
        "the block says a vault autonomy/ path is the exception that is kept; the "
        "prefix set moved")
    def seconds(key: str) -> int:
        """An int literal or a `N * 86400` product — `literal_eval` folds Add/Sub of
        numbers but not Mult, and both constants here are written the second way."""
        v = consts.get(key)
        ok = ((isinstance(v, ast.Constant) and isinstance(v.value, int))
              or (isinstance(v, ast.BinOp) and isinstance(v.op, ast.Mult)
                  and isinstance(v.left, ast.Constant)
                  and isinstance(v.right, ast.Constant)))
        assert ok, f"{key} is neither an int literal nor a product of two literals"
        return (v.value if isinstance(v, ast.Constant)
                else v.left.value * v.right.value)

    assert seconds("REJECT_REST_SECONDS") == 4 * 86400, (
        "the block says the refused unit is re-queued in 4 days")
    assert seconds("MAX_REJECT_REQUEUES") == 2, (
        "the block says the shortened rest is capped at 2 refusals")
    assert "keep_rejected" in names and "revert_strays" in names, (
        "one of the two owners the block names is gone from the module")


# --- clause 3: the two directions the old sentence got backwards -------------

def test_the_block_says_a_vault_stray_outside_the_exempt_prefixes_is_reverted():
    """One sentence, containing the owner, the revert, and at least one vault area
    the OLD comment implied was untouched. The direction matters: the misleading
    half was that a stray into `knowledge/` or a daily note was thought to survive
    the turn, so the corrected sentence has to say so where the reader will not
    have to go find the constant to learn it.
    """
    s = _sentence_naming("VAULT_UNSWEPT_PREFIXES")
    assert re.search(r"\brevert", s, re.I), (
        "the sentence that names the coverage constant never says such a write is "
        "reverted, which is the direction the old comment got wrong")
    assert "vault" in s, "the sentence does not say the vault is the swept area"
    swept = [a for a in ("knowledge/", "memory/", "plans/", "daily note") if a in s]
    assert swept, (
        "the sentence names the constant but no vault area it now covers; the whole "
        "point of clause 3 is that a reader of THIS comment learns that a stray into "
        "`knowledge/`, `memory/`, `plans/` or a daily note goes back to HEAD")
    assert "backlog" in s, "the exempt prefix the constant names is not stated"


def test_the_block_says_a_vault_autonomy_stray_is_reported_and_left_alone():
    """The opposite direction, which the old comment had flatly wrong: `autonomy/`
    was filed under "reverts". Demanding the negation AND the reporting verb AND
    the owner is deliberate — "not reverted" alone could be a sentence about the
    scheduler's tree surviving for some other reason, and the reason (a second
    writer owns the path, so the run record carries the stray) is what stops a
    later reader from "fixing" it back.
    """
    s = _sentence_naming("autonomy/")
    assert re.search(r"\b(?:not|never)\s+reverted\b", s, re.I), (
        "the block files a vault autonomy/ stray under reverts again")
    assert re.search(r"\breport", s, re.I), (
        "the block does not say the path is reported to the run record instead")
    assert "SCHEDULER_OWNED_VAULT_PREFIXES" in s or "kept (scheduler-owned)" in s, (
        "the block gives the verdict without the owner or the action string, so a "
        "reader cannot check it or find the branch")


def test_the_block_says_a_bound_refusal_keeps_the_correction():
    """#3338's half of clause 2: the bound refuses the commit and keeps the edit.
    Asserted per sentence and against the re-queue, because "kept" without the
    cadence is still the old half-truth — the correction's value was that the unit
    comes back in days rather than a month.
    """
    s = _sentence_naming("keep_rejected")
    assert re.search(r"commit", s, re.I) and re.search(r"correction", s, re.I), (
        "the sentence does not distinguish the commit the bound refuses from the "
        "correction it keeps")
    assert "REJECT_REST_SECONDS" in s, "the re-queue the keep enables is not stated"


# --- clause 4: the instrument itself ----------------------------------------

def test_the_parsed_mapping_cannot_see_the_block_at_all():
    """Why the ban reads raw bytes. Load the same bytes through `yaml.safe_load`
    and show that neither the stale claim nor the new owners reach the parsed
    mapping: the mapping's `arch-review` value is seven tuning keys, and a test
    written against it would pass on a comment saying anything at all. This is the
    control that keeps "just read the config dict" from being re-proposed.
    """
    parsed = yaml.safe_load(_raw())
    block_keys = parsed["workers"]["sources"]["arch-review"]
    assert not any(isinstance(k, str) and k.startswith("#") for k in block_keys), (
        "comment lines are being parsed as keys, which means the file is malformed "
        "somewhere above and the block is not a comment block")
    seen = yaml.safe_dump(block_keys)
    for token in ("VAULT_UNSWEPT_PREFIXES", "keep_rejected", "skills/"):
        assert token not in seen, (
            f"{token} reached the parsed mapping, so this file's premise — that a "
            "comment-only pin cannot be asserted against the parsed config — is false")


def test_config_yaml_still_parses_and_gained_no_key():
    """The other direction of the same instrument: a comment edit is only safe if
    the file still loads and the `arch-review` mapping still holds exactly the
    knobs it held before. `#4059` is comment-only, and this node is what makes
    that claim checkable rather than asserted in a commit message.
    """
    parsed = yaml.safe_load(_raw())
    keys = set(parsed["workers"]["sources"]["arch-review"])
    # 17 keys, measured off this file at #4059.
    assert {"batch", "daily_max", "enabled", "groups", "inner_voice",
            "interval_seconds", "max_attempts", "max_delta_lines",
            "max_duration_seconds", "max_inflight", "max_open_items",
            "max_shrink_pct", "max_turns", "priority", "retry_spacing_seconds",
            "review_interval_days", "spawn_cap"} == keys, (
        f"the knob set moved ({len(keys)} keys); the block over it is prose for the "
        "person editing these, so it and this node both need re-reading against "
        "whatever changed")
