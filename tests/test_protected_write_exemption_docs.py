"""#3563 — the write deny-set has one exemption, and the tree says so.

Three comments and two architecture rows asserted that a sanctioned in-process
writer depended on a `ContextVar` lift over the write deny-set. The lift had no
caller and never had one — all eight commits that ever touched it were read and
contain none — and the writer named in the prose does not exist either: the
paths the nightly reflection and the loaded-memory writers land are outside the
deny-set, and the one deny-set file that does get written by a sanctioned route
(`~/obsidian/lloyd/SOUL.md`) goes through `automod_vault_land`, which validates
the path against the prompt-surface contract and commits it with a `git`
subprocess without ever calling `write_deny_reason`.

So the mechanism is deleted, and this file is the rail that keeps the *story*
deleted with it. The prose is what makes a deleted guard dangerous again: a
future reader who finds "lifted only by the allow-listed contextvar" in a doc
row will not re-check whether the contextvar exists, which is precisely how an
escape hatch with no owner, no reason anyone reads back and no ledger row sat
beside a journalled refusal for weeks. The architecture doc-claims checker
cannot grade this — `tests/test_architecture_coverage_doc_claims.py` states that
a claim is graded only where the cited module has a dispatch chain, and
`allow_protected_writes` appears there as a *symbol* citation, not a command —
so the doc text is asserted here instead of riding ungraded.

#4438 is the hole this file's first arm did not cover. Its `FORBIDDEN` sweep
matches five exact *symbol names*, and a promise needs no symbol:
`check_bash_write_denied`'s docstring said it refused "on exactly the predicate
the file lane refuses on, grant contextvar included" — the mechanism, in the
present tense, in the lane that answers "may I write here?" for every Bash
command — and the name sweep passed it a thousand lines below the header that
same commit rewrote to say no such exemption exists. `LIFT_WORDS` and the
prose arm below are that missing arm; the docstring nodes at the end pin what
the rewritten sentence has to keep saying.
"""

from __future__ import annotations

import inspect
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

#: Where the deleted mechanism was named. A sweep of the whole tree would be
#: noise (`tests/` keeps the name deliberately: it is the spoof key in
#: `test_no_tool_argument_lifts_the_denial`, a model can send any key it likes,
#: and the point of that node is that reading none of them is the property), and
#: these three are everywhere the mechanism was claimed to *work*.
SWEEP_ROOTS = ("app", "agent_mcp", "architecture")

#: Every name the lift went by. Not one of them may survive outside `tests/`,
#: because a name in a comment is a promise, and a promise about an exemption
#: nobody can audit is the bug — #3563 is about the promise as much as the code.
FORBIDDEN = (
    "allow_protected_writes",
    "grant_protected_writes",
    "release_protected_writes",
    "protected_writes_granted",
    "_write_grant",
)

_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".sh", ".conf", ".toml"}


def _sweep_files(base: pathlib.Path) -> list[pathlib.Path]:
    """Every swept file under `base`, the same roots and suffixes throughout.

    Parameterised by tree so the seeded nodes below can point this exact
    collection at a scratch directory: an arm proven on an in-memory string
    list is a different arm from the one that reads the shipped files.
    """
    out: list[pathlib.Path] = []
    for root in SWEEP_ROOTS:
        d = base / root
        assert d.is_dir(), f"{base / root} is gone, so the sweep below proves nothing"
        out.extend(p for p in sorted(d.rglob("*"))
                   if p.is_file() and p.suffix in _SUFFIXES
                   and "__pycache__" not in p.parts)
    return out


def _swept() -> list[pathlib.Path]:
    out = _sweep_files(REPO)
    assert len(out) > 200, (
        f"the sweep found {len(out)} files under {SWEEP_ROOTS}; a pattern "
        "searched over nothing is a clean result about nothing")
    return out


@pytest.mark.parametrize("name", FORBIDDEN)
def test_no_non_test_file_still_names_the_deleted_lift(name):
    """The acceptance check, as a test rather than as a grep someone runs once.

    The name is what matters, not the code: with the definitions gone the
    behaviour cannot regress by itself, but a comment that still promises a lift
    is a reader's invitation to add one back — and it is the exact prose that
    hid the fact that no writer needed it.
    """
    hits = []
    for path in _swept():
        text = path.read_text(encoding="utf-8", errors="replace")
        if name in text:
            line = next(i for i, ln in enumerate(text.splitlines(), 1) if name in ln)
            hits.append(f"{path.relative_to(REPO)}:{line}")
    assert not hits, (
        f"`{name}` was deleted with the lift, so nothing outside tests/ may "
        f"still assert a writer depends on it: {hits}")


# ---------------------------------------------------------------------------
# Shipped before #4438 and kept verbatim: the live exception and the two
# architecture rows that were rewritten when the lift went.
# ---------------------------------------------------------------------------

def test_the_docs_name_the_exempt_route_and_its_mechanism():
    """"Exempt by design" is not an explanation, so the rows have to say which
    lane is exempt and by what mechanism: a `git` commit behind a validator.

    Both rows are pinned for the route *and* the mechanism word, because naming
    `automod_vault_land` alone would let a reader conclude the tool holds a
    permission inside the guard. It does not: it never calls the predicate, so
    deleting the lift could not break it.
    """
    rows = {
        "authority-surfaces.md": (REPO / "architecture" / "authority-surfaces.md").read_text(),
        "tools.md": (REPO / "architecture" / "tools.md").read_text(),
    }
    for doc, text in rows.items():
        assert "automod_vault_land" in text, f"{doc} does not name the exempt route"
        i = text.index("automod_vault_land")
        window = text[max(0, i - 700): i + 700]
        assert re.search(r"\bgit\b", window), f"{doc}: no commit mechanism near the route"
        assert re.search(r"validat", window), (
            f"{doc}: the exemption is not tied to validation, which is the only "
            "thing that distinguishes it from a bypass")


def test_authority_surfaces_no_longer_calls_the_exemption_a_design_choice():
    """Row 3's old words were "`automod_vault_land` is the one exempt lane, by
    design". "By design" reads as a decision that was made and can be revisited;
    the truth is that the exemption is a *mechanism* that does not pass through
    the guard at all — and the design sentence was standing where the actual
    design (no in-process lift, ever) belonged."""
    text = (REPO / "architecture" / "authority-surfaces.md").read_text()
    row = next(ln for ln in text.splitlines()
               if ln.startswith("| 3 |"))
    assert "by design" not in row, row
    assert "automod_vault_land" in row and "CONTRACT_PATHS" in row, row
    assert "Nothing inside this process lifts the write deny set" in row, (
        "the row must state the positive property, not merely drop the old claim")


def test_the_grants_row_denies_that_a_minted_grant_reaches_the_deny_set():
    """Row 8 closed with "the protected-write escape in layer 3 is the same idea
    scoped to one call" — the sentence that told a reader a human-minted grant
    and a deny-set exemption were one mechanism with two scopes. They are not,
    and `grant_create` has never had a scope that lifts a deny-set path: the row
    now says so, in those terms, next to the row that explains the route."""
    text = (REPO / "architecture" / "authority-surfaces.md").read_text()
    row = next(ln for ln in text.splitlines() if ln.startswith("| 8 |"))
    assert "no" in row.lower() and "deny-set" in row, row
    assert "automod_vault_land" in row, (
        "the row must point at the route rather than only deny the mechanism")


# ---------------------------------------------------------------------------
# The prose arm (#4438): the same promise, made without a symbol
# ---------------------------------------------------------------------------

#: The words the lift was promised by. `FORBIDDEN` catches the five symbols; a
#: sentence can carry the identical claim with none of them, which is how
#: #4438's stale docstring survived the commit that deleted the mechanism and
#: rewrote the header two lines above it.
LIFT_WORDS = (
    "contextvar", "context var", "allow_", "lift", "lifts", "lifted", "lifting",
    "exempt", "exemption", "wrapper", "override", "bypass", "escape", "release",
)

#: Words that make a *grant* the condition of a refusal, which is the claim
#: #4438's sentence was making ("refuses on ... grant contextvar included").
CONDITION_WORDS = ("unless", "held", "minted", "consult", "condition")

#: `grant` names a live mechanism too — #534's `grant_create`, whose scopes
#: reach none of the deny-sets — so a mention of one beside the other is legal
#: prose. What makes it a promise is the claim that the deny-set is *lifted* or
#: *conditioned*: `architecture/index.md`'s guard-order row lists `grants` as
#: one layer after "the write deny-set", and that row is true; the
#: refusal-logging histories in `app/harness/denial_journal.py` and
#: `architecture/harness.md` list "the protected-path deny-sets" among a dozen
#: guards, and those are true too. A bare mention is not the story this rail
#: exists to catch; a lift verb, or a refusal made conditional on a grant, is.
GRANT_WORD = "grant"

#: What a sentence has to name to be about THIS deny-set. "deny-set" alone is
#: not a handle, for the two listing cases named just above.
DENY_SET_HANDLES = (
    "write_deny_reason", "write_deny", "write deny", "protected_write",
    "protected-write", "protected write", "bash_write_denied", "write-side lift",
)

#: The words #3563 left at every site it rewrote to deny the lift ("NO
#: IN-PROCESS EXEMPTION EXISTS", "no second argument, no contextvar", "No grant
#: lifts it", "Nothing lifts it", "#3563 deleted the contextvar lift").
_NEGATION = re.compile(
    r"\b(no|not|never|nothing|neither|nor|without|none|zero|deleted?|gone|absent)\b")

#: A sentence ends at `.!?` before whitespace, or at a table cell's `|`.
_SENT_BREAK = re.compile(r"(?<=[.!?|])(?=\s)|\|")


def _sentence_units(text: str) -> list[tuple[int, str]]:
    """Every sentence of `text`, as `(first line it appears on, sentence)`.

    Two rules, and both decide whether this rail can fail at all:

    * A *block* — a maximal run of consecutive non-blank lines — is joined
      before it is split. A docstring wraps one sentence over five lines, and a
      unit that stopped at every newline would put the deny-set's name in one
      unit and the promise in another, so no sentence would ever hold both.
    * The joined block is then split into sentences rather than kept whole,
      because the negation has to belong to the sentence that promises. The
      stale sentence #4438 is about ran, in the same line, into "A reader of a
      deny-set path is not a target of this check": under any whole-block or
      whole-line rule that unrelated "not" cures the promise, and the rail
      passes the exact sentence the item was filed for. `tmp_path` node
      `test_a_negation_in_the_next_sentence_does_not_cure_the_promise` pins it.
    """
    out: list[tuple[int, str]] = []
    block: list[tuple[int, str]] = []

    def flush(block: list[tuple[int, str]]) -> None:
        if not block:
            return
        joined = " ".join(ln.strip() for _, ln in block)
        starts: list[int] = []
        pos = 0
        for _, ln in block:
            starts.append(pos)
            pos += len(ln.strip()) + 1

        def line_at(idx: int) -> int:
            for i, start in enumerate(starts):
                if start > idx:
                    return block[i - 1][0]
            return block[-1][0]

        idx = 0
        for m in _SENT_BREAK.finditer(joined):
            chunk = joined[idx:m.start()]
            if chunk.strip():
                out.append((line_at(idx), chunk))
            idx = m.end()
        if joined[idx:].strip():
            out.append((line_at(idx), joined[idx:]))

    for i, ln in enumerate(text.splitlines(), 1):
        if ln.strip():
            block.append((i, ln))
        else:
            flush(block)
            block = []
    flush(block)
    return out


def _offence(low: str) -> list[str]:
    """The words making this lower-cased sentence a lift promise, `[]` if not.

    Returning them rather than a bool is what puts the offending word in the
    failure message: an arm that reports only a truncated sentence cannot say
    which half of the promise it read, and the promise #4438 left behind wrapped
    past 200 characters before its `contextvar`.
    """
    words = [word for word in LIFT_WORDS if word in low]
    grant = GRANT_WORD in low
    if not words and not grant:
        return []
    if not words:
        # A bare grant is a promise only where the sentence conditions the
        # refusal on it — see GRANT_WORD's note on the true listings it must
        # not report.
        conditions = [word for word in CONDITION_WORDS if word in low]
        return [GRANT_WORD, *conditions] if conditions else []
    if grant:
        words.append(GRANT_WORD)
    return words


def _lift_promises(base: pathlib.Path) -> list[str]:
    """Every sentence under `base` that promises a lift and denies nothing.

    One report line per offence — path, line, the promise words, the sentence —
    the shape the `FORBIDDEN` arm above reports in.
    """
    hits: list[str] = []
    for path in _sweep_files(base):
        text = path.read_text(encoding="utf-8", errors="replace")
        for line, sentence in _sentence_units(text):
            low = sentence.lower()
            if not any(handle in low for handle in DENY_SET_HANDLES):
                continue
            offence = _offence(low)
            if not offence:
                continue
            if _NEGATION.search(low):
                continue
            hits.append(f"{path.relative_to(base)}:{line} "
                        f"({', '.join(offence)}): "
                        f"{' '.join(sentence.split())[:200]}")
    return hits


def test_no_shipped_prose_promises_an_in_process_lift():
    """Nothing in `app/`, `agent_mcp/` or `architecture/` may say the write
    deny-set can be lifted in process, without denying it in the same breath.

    This is #4438's arm: the name sweep cannot see the claim made in prose, and
    every surviving mention of the lift in the shipped tree is a negation, so
    the arm is green here and red on the pre-fix commit.
    """
    files = _sweep_files(REPO)
    assert len(files) > 200, (
        f"the prose arm scanned {len(files)} files under {SWEEP_ROOTS}; a clean "
        "result about almost nothing is not a pass")
    hits = _lift_promises(REPO)
    assert not hits, (
        "the write deny-set has no in-process exemption (#3563), so a sentence "
        "that names a lift or a grant over it must say so in its own sentence: "
        + "; ".join(hits))


def _seed(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
    """A three-root tree holding `body` in one swept `app/` file."""
    for root in SWEEP_ROOTS:
        (tmp_path / root).mkdir()
    pkg = tmp_path / "app" / "harness"
    pkg.mkdir(parents=True)
    (pkg / "seeded.py").write_text(
        '"""The seeded sentence, in the shape it shipped in.\n\n'
        f"{body}\n\"\"\"\n", encoding="utf-8")
    return tmp_path


#: The sentence #4438 was filed for, verbatim from `check_bash_write_denied`'s
#: docstring before the fix, and the only form of it that this rail must catch:
#: the promise, wrapped over five lines, with the deny-set named on the first
#: and the lift on the fourth.
STALE_SENTENCE = (
    "    The Bash half of `write_deny_reason`, for the lane that reaches a shell\n"
    "    instead of a path argument: it names the shape and the resolved target and\n"
    "    then refuses on exactly the predicate the file lane refuses on, grant\n"
    "    contextvar included. A reader of a deny-set path is not a target of this\n"
    "    check at all -- see the block comment above."
)


def test_the_prose_arm_bites_on_the_shipped_stale_sentence(tmp_path):
    """The arm #4438 adds has to have caught the sentence #4438 was filed for."""
    hits = _lift_promises(_seed(tmp_path, STALE_SENTENCE))
    assert len(hits) == 1, f"expected exactly one offence, got {hits}"
    assert hits[0].startswith("app/harness/seeded.py:"), hits[0]
    assert "contextvar" in hits[0], hits[0]


def test_the_prose_arm_bites_on_a_refusal_made_conditional_on_a_grant(tmp_path):
    """Same claim, no `contextvar` in it: "refuses unless a grant is live" is
    the identical invitation to go hunting for the grant."""
    body = ("The Bash lane refuses a write deny-set target unless the caller's\n"
            "grant is live, which is how the nightly writer gets through.")
    hits = _lift_promises(_seed(tmp_path, body))
    assert len(hits) == 1, f"expected exactly one offence, got {hits}"
    assert "grant" in hits[0], hits[0]


def test_the_prose_arm_passes_the_same_sentence_with_its_negation(tmp_path):
    """The arm bans a *promise*, not a vocabulary: deny the lift in the
    promising sentence and the identical sentence goes green. Without this
    control node the arm could be satisfied only by deleting every mention."""
    body = ("The write deny-set is lifted by no in-process contextvar and no\n"
            "allow_* wrapper; #3563 deleted the lift.")
    assert _lift_promises(_seed(tmp_path, body)) == [], "a denial was reported"


def test_a_negation_in_the_next_sentence_does_not_cure_the_promise(tmp_path):
    """The scope choice, pinned: a denial belonging to another sentence is not
    a denial of *this* one, which is why the window is the sentence and not the
    block or the line — the stale sentence's own line carried an unrelated
    "not", and a ±lines arm reported it clean."""
    body = ("The write deny-set is lifted by the grant a sanctioned writer holds.\n"
            "No such mechanism exists, and #3563 deleted it.")
    hits = _lift_promises(_seed(tmp_path, body))
    assert len(hits) == 1, f"the neighbouring denial cured it: {hits}"


# ---------------------------------------------------------------------------
# The lane's own docstring (#4438): what the rewritten sentence has to say
# ---------------------------------------------------------------------------


def test_the_bash_lane_docstring_names_one_predicate_and_no_lift():
    """Clauses 1 and 2, pinned on the live object rather than on a grep.

    The deleted clause was the only description of the Bash lane's refusal a
    reader got without opening `write_deny_reason` two hundred lines up, so
    what it now says is the contract: one predicate, shared, one input.
    """
    from app.harness import protected_paths as PP

    doc = " ".join((PP.check_bash_write_denied.__doc__ or "").split())
    assert doc, "check_bash_write_denied has no docstring to read"
    assert "contextvar" not in doc.lower(), (
        "the lift #3563 deleted is back in the sentence a triager reads: " + doc)
    assert "grant" not in doc.lower(), (
        "a grant is not an input to this refusal, so naming one here promises "
        "a route: " + doc)
    # Clause 2: the true content the deleted clause was carrying.
    assert "write_deny_reason" in doc, doc
    assert "one predicate the bash lane and the file lane both refuse on" in doc.lower(), (
        "the sentence must name the single predicate the two lanes share: " + doc)
    assert "nothing lifts it" in doc.lower(), (
        "the lane's own denial of a lift belongs in its own docstring: " + doc)


def test_the_shared_predicate_takes_the_target_alone():
    """The property the deleted clause denied in words, as an executable fact:
    `write_deny_reason` answers from set membership and nothing else, so there
    is no second input for a grant, a contextvar or a wrapper to reach."""
    from app.harness import protected_paths as PP

    params = list(inspect.signature(PP.write_deny_reason).parameters)
    assert params == ["target"], (
        f"write_deny_reason grew a second input ({params}); #3563's answer to "
        "'may I write here?' is set membership alone")


def test_the_correct_negations_survive():
    """#4438's fix is local: the sites that already deny the lift must keep
    denying it. Rewriting them is how a correction becomes a second stale
    claim, and the sweep above only catches a denial that turned into a
    promise if the denial is still there to turn."""
    from app.harness import protected_paths as PP

    assert "No second argument, no contextvar, no `allow_*` wrapper" in (
        PP.write_deny_reason.__doc__ or ""), "write_deny_reason's denial moved or went"
    source = (REPO / "app" / "harness" / "protected_paths.py").read_text()
    assert "NO IN-PROCESS EXEMPTION EXISTS (#3563)" in source, (
        "the header that names the deletion went, so nothing explains the gap")
    assert "No grant lifts it, and since #3563 there is no write-side lift" in source, (
        "the read side's denial of a write-side lift went with it")
    assert '"no second argument, no contextvar, no `allow_*` wrapper"' in source, (
        "bash_vault_write_denial quotes write_deny_reason's denial; the quote "
        "is what keeps the two lanes' wording from drifting apart")
