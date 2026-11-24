"""#4121: a writer growing `USER.md` is TOLD before it is refused, and refused nowhere new.

The item is a staging gap, measured twice with the same number. `USER.md` sits at
16,247 of its 16,384-byte ceiling — 137 bytes, and #3629 measured 59 bytes five weeks
earlier — while `scripts/memory/memory_ledger.py status` prints, for that file and not
for `MEMORY.md`, `tight limit 13107 not enforced on writes; enforced write bound is its
16384 B ceiling … -> CURATE`. The word pair that differs from the `MEMORY.md` line one
line below (`tight limit 20480 enforced on growth`) is the whole finding: the index has
two staged bounds in front of its ceiling and the profile has none, so the first thing
its writer ever hears is `USER.md is 16,385 bytes, over its 16,384-byte ceiling (1 B
over)` — the Edit path told the dream-consolidation pass exactly that while filing it.

The triage settled the design call on the WARNING rung rather than a reserve, and the
reason is arithmetic: a reserve of N bytes over a file with 137 bytes of headroom
freezes USER.md growth the day it ships. `INDEX_MEMORY_FILE`'s own comment calls that
"a ceiling below the file it bounds is a freeze, not a tripwire", and
`prompt_surface`'s `USER_MD_CEILING_BYTES` comment records #1425's ruling via #2161 that
the number moves only "with a trim in the same commit, never by lowering it alone". The
item's scope guard points the same way: no trim is the fix here.

So what these tests pin is a notice that refuses nothing:

* it fires at or above `HEADROOM_WARNING_TIGHTNESS` of the file's own
  `memory_ceiling()`, derived from that function and not from a byte count written down
  beside it (clause 1) — including when the ceiling itself is moved under it;
* every write it warns about still lands, including a shrink of an over-ceiling file,
  and the 16,384-byte refusal is undisturbed (clause 2);
* it does not widen the index scope it deliberately is not gated on:
  `index_bounds_apply("USER.md")` stays False, the three `index_*` arms keep answering
  None for `USER.md`, and `memory_write_error`'s body still calls `index_bounds_apply`
  exactly twice (clause 3 — the count #3661's regression test pins, restated here so a
  change to this file cannot quietly make it three);
* for `USER.md` both the notice and the over-ceiling refusal send the detail to the
  vault-root `memory/mental-models.md`, and neither tells the writer to move profile
  content into a `topics/<slug>` index file — while `MEMORY.md` keeps the topic-file
  route the index genuinely has (clause 4).

The fifth clause — that every handler surfaces the text instead of discarding it — is
pinned in `tests/test_builtin_fs_protected_write.py`, the file whose fixtures already
drive the real Write, Edit, `memory_add`, `memory_replace` and `vault_write` handlers.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import memory_ceiling as MC  # noqa: E402
from app import prompt_surface as PS  # noqa: E402

_SRC = (ROOT / "app" / "memory_ceiling.py").read_text(encoding="utf-8")


def _func(name: str) -> ast.AST:
    tree = ast.parse(_SRC)
    return next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == name)


@pytest.fixture
def memories_root(tmp_path, monkeypatch):
    """A scratch memories root the real path-resolution in `memory_ceiling` points at.

    `MEMORIES_DIR` is resolved from the real `$HOME` at import, so without this every
    scratch path answers "not a memory file" and a node passes for the wrong reason.
    """
    root = tmp_path / "lloyd"
    root.mkdir()
    monkeypatch.setattr(MC, "MEMORIES_DIR", root)
    return root


def _text_of_nbytes(n: int, *, seed: str = "- [user] a ruling that stays loaded\n") -> str:
    """A memory file of EXACTLY `n` bytes: the seed, filler lines, then a tail.

    Byte-exact and not character-exact, because every bound in this module is
    `len(text.encode("utf-8"))` and #1010's whole lesson is that the two differ.
    """
    filler = "- filler entry that pads the file out to size\n"
    assert len(seed.encode()) + len(filler.encode()) <= n, "fixture smaller than its seed"
    text = seed
    while len(text.encode()) + len(filler.encode()) <= n:
        text += filler
    return text + "x" * (n - len(text.encode()))


def _user_md(memories_root, size: int) -> Path:
    p = memories_root / "USER.md"
    p.write_text(_text_of_nbytes(size), encoding="utf-8")
    return p


# ── clause 1: the rung exists, and it is derived, not written down ─────────────

def test_the_notice_fires_at_the_tightness_of_the_ceiling_and_is_quiet_below_it():
    """`headroom_warning` answers non-None at 90 % of `USER.md`'s ceiling, None below.

    The boundary is read off `memory_ceiling()` and `HEADROOM_WARNING_TIGHTNESS` here
    rather than as a literal, and that is the clause: the rung has to move with the
    ceiling, and a test that hard-coded 14,745 would pass against a byte constant the
    item forbids.
    """
    ceiling = MC.memory_ceiling("USER.md")
    rung = int(ceiling * MC.HEADROOM_WARNING_TIGHTNESS)

    assert MC.headroom_warning("USER.md", _text_of_nbytes(rung - 1)) is None, \
        "the notice fired below its rung"
    msg = MC.headroom_warning("USER.md", _text_of_nbytes(rung))
    assert msg is not None, "the notice did not fire AT the rung"
    assert msg.startswith("USER.md would be "), msg
    assert f"{rung:,}-byte headroom rung" in msg, msg
    assert f"{ceiling:,}-byte ceiling" in msg, msg
    assert "NOT refused" in msg, msg


def test_the_rung_is_a_ratio_over_the_ceiling_and_moves_when_the_ceiling_moves():
    """No byte constant of its own: shrink the ceiling under it and the rung follows.

    `MEMORY_TIGHTNESS` (0.80) is the index's REFUSAL tightness and is deliberately not
    reused here — 0.90 is a notice's, on the same decision that made the notice a LEVEL:
    it prices a prospective write, where the dose reading prices a day, and the trigger
    that would reopen that call is the command named in the `#:` block above
    `HEADROOM_WARNING_TIGHTNESS`, run by whoever asks, not a row a closed item is still
    owed. What may not exist is a number in BYTES: this module has exactly the ceilings
    `prompt_surface` owns and no others, which is the second assertion below.
    """
    assert isinstance(MC.HEADROOM_WARNING_TIGHTNESS, float)
    assert 0.0 < MC.HEADROOM_WARNING_TIGHTNESS < 1.0
    assert MC.HEADROOM_WARNING_TIGHTNESS != MC.MEMORY_TIGHTNESS

    moved = 1_000
    rung = int(moved * MC.HEADROOM_WARNING_TIGHTNESS)
    assert rung == 900, "arithmetic drifted from an int() over 90 % of 1,000 B"
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(PS.MEMORY_CEILINGS, "USER.md", moved)
        assert MC.headroom_warning("USER.md", _text_of_nbytes(rung - 1)) is None
        assert MC.headroom_warning("USER.md", _text_of_nbytes(rung)) is not None

    literals = [v.value for v in ast.walk(_func("headroom_warning"))
                if isinstance(v, ast.Constant) and isinstance(v.value, (int, float))]
    assert not [n for n in literals if n >= 100], \
        f"`headroom_warning` wrote down a byte-sized number: {literals}"


def test_a_name_that_is_never_loaded_has_no_rung_and_no_notice():
    """`SOUL.md` and a topic file are not bounded, so they are never warned about.

    `tight_limit` answers None for anything outside `MEMORY_CEILINGS`, and a notice on
    an unbounded file would be a claim about a ceiling that does not exist.
    """
    big = _text_of_nbytes(60_000)
    assert MC.headroom_warning("SOUL.md", big) is None
    assert MC.headroom_warning("some-slug", big) is None


# ── clause 2: advisory, never a refusal ────────────────────────────────────────

def test_every_write_the_notice_warns_about_still_lands(memories_root):
    """Between the rung and the ceiling, `memory_write_error` is still None (#4121).

    Priced at the live shape: `USER.md` at its measured 16,247 B growing by one short
    bullet. The warning is non-None on exactly that prospective text — that IS the
    item — and the refusal is None, so the growth that #3629 watched happen twice in five
    weeks still happens. A shrink is still allowed even over the ceiling, which is how a
    trim through the same lane stays possible after a refusal.
    """
    live_size = 16_247
    p = _user_md(memories_root, live_size)
    grown = p.read_text(encoding="utf-8") + "\n- x"

    assert MC.memory_write_error(p, grown) is None, \
        "the advisory rung leaked a refusal into the write path"
    assert MC.memory_write_warning(p, grown) is not None

    over = p.read_text(encoding="utf-8") + "z" * 200          # 16,447 B, over
    assert MC.memory_write_error(p, over) is not None, \
        "the 16,384-byte refusal stopped firing"
    assert MC.memory_write_warning(p, over) is not None

    shrunk = over[:live_size]                                  # a repair write
    assert MC.memory_write_error(p, shrunk) is None, \
        "the shrink escape moved, so a trim is no longer possible through this lane"


def test_the_path_form_abstains_on_everything_that_is_not_a_loaded_file(memories_root):
    """Topic files, `SOUL.md` and paths outside the memories root get no notice at all.

    Same gate as the refusal — `_loaded_memory_name` — so the warning cannot fire on a
    file that has no prompt cost, and cannot fire on the overflow target itself
    (`memory/mental-models.md` is uncapped by design and lives above this root).
    """
    topic = memories_root / "memory" / "some-slug.md"
    topic.parent.mkdir(parents=True, exist_ok=True)
    big = _text_of_nbytes(60_000)
    topic.write_text(big, encoding="utf-8")
    assert MC.memory_write_warning(topic, big) is None

    outside = topic.parent.parent.parent / "mental-models.md"
    outside.write_text(big, encoding="utf-8")
    assert MC.memory_write_warning(outside, big) is None

    soul = memories_root / "SOUL.md"
    soul.write_text(big, encoding="utf-8")
    assert MC.memory_write_warning(soul, big) is None


# ── clause 3: the index scope is not widened to let the notice through ─────────

def test_the_notice_reaches_where_the_index_bounds_decline_to_go():
    """`USER.md` gets the notice precisely because the index arms answer None for it.

    The three scoped arms and the notice are given the identical oversized text. The
    arms stay silent — that is `index_bounds_apply("USER.md") is False`, the same
    assertion #3629 and #3661 pin in their own files — and the notice answers. If this
    one day fails because the arms started speaking for `USER.md`, the freeze
    `INDEX_MEMORY_FILE`'s comment warns about has arrived wearing a different coat.
    """
    assert MC.index_bounds_apply("USER.md") is False
    big = _text_of_nbytes(16_300)
    assert MC.index_tight_limit_error("USER.md", big) is None
    assert MC.index_line_length_error("USER.md", big) is None
    assert MC.index_room_reserve_error(MC.MEMORIES_DIR / "USER.md", "USER.md", big) is None
    assert MC.headroom_warning("USER.md", big) is not None

    # And the index keeps its own bounds, at their own numbers, untouched by this.
    assert MC.index_bounds_apply("MEMORY.md") is True
    assert MC.index_tight_limit_error(
        "MEMORY.md", _text_of_nbytes(20_481, seed="- [project] x\n")) is not None


def test_memory_write_error_still_asks_the_index_scope_exactly_twice():
    """The count #3661 pins, restated: the notice is a separate function, not a third ask.

    The widening was tried and refused once already — a scope that answers True for
    `USER.md` would have made the room reserve and the line cap refuse USER.md writes,
    which is the freeze. The shape that keeps both guarantees is a second predicate
    beside `memory_write_error`, and this is the node that stops someone collapsing it.
    """
    calls = [n for n in ast.walk(_func("memory_write_error"))
             if isinstance(n, ast.Call)
             and (getattr(n.func, "id", "") or getattr(n.func, "attr", ""))
             == "index_bounds_apply"]
    assert len(calls) == 2, \
        f"`memory_write_error` asks `index_bounds_apply` {len(calls)} times, expected 2"

    # The notice is not wired INTO the refusal: nothing in that body names it.
    named = {getattr(n.func, "id", "") or getattr(n.func, "attr", "")
             for n in ast.walk(_func("memory_write_error")) if isinstance(n, ast.Call)}
    assert "headroom_warning" not in named and "memory_write_warning" not in named, \
        "the advisory rung was moved into the refusal path"


# ── clause 4: USER.md's detail goes to mental-models, not to an index topic ────

def test_the_notice_and_the_refusal_both_send_user_md_detail_to_mental_models(
        memories_root):
    """`memory/mental-models.md`, and no `topics/<slug>` instruction, in either message.

    `lloyd/USER.md:13` already says this file's reinforcements go to the uncapped
    vault-root `memory/mental-models.md` and NOT to `lloyd/memory/` topic files; until
    now the one message `USER.md`'s writer ever received said the opposite, because
    `consolidation_hint` was appended to every loaded file's refusal and named the
    index's route. A writer that obeyed it deleted a ruling from every prompt: topic
    files are never rendered (`memory_ceiling`'s own comment says so), and the 2026-10-28
    ruling is that only SOUL/MEMORY/USER.md reach a prompt. So the text has to carry the
    caveat too — a relocation buys bytes by taking that detail out of every prompt, which
    is why it says move the DETAIL and keep the ruling.
    """
    p = _user_md(memories_root, 16_247)
    grown = p.read_text(encoding="utf-8") + "\n- x"
    notice = MC.memory_write_warning(p, grown)
    assert "memory/mental-models.md" in notice, notice
    assert "topics/<slug>" not in notice, notice

    over = p.read_text(encoding="utf-8") + "z" * 200
    refusal = MC.memory_write_error(p, over)
    assert refusal is not None and "16,384-byte ceiling" in refusal, refusal
    assert "memory/mental-models.md" in refusal, refusal
    assert "topics/<slug>" not in refusal, refusal
    assert "keep the ruling" in refusal, refusal


def test_memory_md_keeps_the_topic_file_route_the_index_actually_has():
    """The split is per file, not a rename: the INDEX still goes to `topics/<slug>`.

    An index line is a pointer with a hook and its detail genuinely has somewhere to go
    inside `lloyd/memory/`. Making the two files share one route either way is the
    regression this node watches: `MEMORY.md` must keep the topic-file instruction, on
    both its messages, and must not be pointed at the profile's file.

    The sizes here record an ordering worth having written down. `MEMORY.md`'s 80 %
    REFUSAL limit is 20,480 B and its 90 % NOTICE rung is 23,040 B, so on the index the
    bound sits BEHIND the notice: an ordinary growth write there is refused before it is
    ever noticed, and the notice only rides on the writes the bound lets through — a
    shrink, or a repair of a file that grew before either bound existed. On `USER.md`,
    which has no bound behind its ceiling, the notice is the only staging there is. That
    is the difference in kind between the two files, and it is why 24,000 B is the right
    fixture here and 16,251 B is the right one above.
    """
    assert MC.tight_limit("MEMORY.md") == 20_480, "the index's bound moved"
    big = _text_of_nbytes(24_000, seed="- [project] x\n")
    notice = MC.headroom_warning("MEMORY.md", big)
    assert notice is not None and "topics/<slug>" in notice, notice
    assert "mental-models" not in notice, notice
    # 20,481 B is over the index's refusal limit and under this file's notice rung: the
    # tight limit answers, the notice does not. Were the rung at or below 0.80, every
    # write the tight limit already refuses would carry a warning too, and a warning
    # attached to a refusal is noise rather than staging.
    assert MC.headroom_warning(
        "MEMORY.md", _text_of_nbytes(20_481, seed="- [project] x\n")) is None, \
        "the notice rung sits at or below the index's own refusal limit"

    refusal = PS.size_error("MEMORY.md", _text_of_nbytes(26_000, seed="- [project] x\n"))
    assert refusal is not None and "topics/<slug>" in refusal, refusal
    assert "mental-models" not in refusal, refusal


def test_the_prose_route_and_the_guard_scope_agree_on_which_file_is_the_index():
    """`prompt_surface` and `memory_ceiling` may not disagree about the scope's name.

    `overflow_route` chooses a destination on `INDEX_MEMORY_NAME`; the refusal arms
    choose theirs on `INDEX_MEMORY_FILE`. The two modules cannot share one constant in
    either direction — `memory_ceiling` imports `prompt_surface`, so the reverse import
    is a cycle — which leaves two spellings of one fact and no check until this one.
    """
    assert PS.INDEX_MEMORY_NAME == MC.INDEX_MEMORY_FILE == "MEMORY.md"


# ── #4303: the constant documents its own ruling, and cites nobody who is shut ───

_TEST_SRC = Path(__file__).resolve().read_text(encoding="utf-8")

#: The node whose docstring used to defer the tightness to the dose item's row store.
_TIGHTNESS_NODE = ("test_the_rung_is_a_ratio_over_the_ceiling_and_moves_when_"
                   "the_ceiling_moves")

#: The closed dose item, in two pieces on purpose: the guard below forbids its digit
#: sequence anywhere in this file, and a guard that named the number to forbid it would
#: find only itself. Concatenation is evaluated at import, so the prose never carries it.
_SHUT_DOSE_ITEM = "412" + "2"


def _tightness_comment() -> str:
    """The `#:` block over `HEADROOM_WARNING_TIGHTNESS`, as the prose a reader receives.

    Contiguous comment lines above the constant, prefixes stripped and hard wraps folded,
    so a claim is matched as a sentence rather than at whatever column it happens to break.
    Read off the source instead of compared to one fixed blob: a rewording that keeps the
    rulings still passes, and one that drops a ruling has to drop a substring a node below
    asks for.
    """
    lines = _SRC.splitlines()
    at = next(i for i, line in enumerate(lines)
              if line.startswith("HEADROOM_WARNING_TIGHTNESS ="))
    block: list[str] = []
    i = at
    while i > 0 and lines[i - 1].startswith("#:"):
        block.append(lines[i - 1])
        i -= 1
    assert block, "the constant lost its `#:` comment block entirely"
    return " ".join(line.removeprefix("#:").strip()
                    for line in reversed(block))


def test_the_tightness_comment_carries_the_level_ruling_and_no_dead_pointer():
    """#4303 clause 1: the block says what it settled, and sends no reader to a shut item.

    It used to defer the level-versus-dose question to the item that measured the dose.
    That item is closed and its row store has never had a row written to it, so the
    sentence had no owner: no future run would ever answer the question it raised. The
    ruling therefore has to live in the block itself — where the next reader of the
    constant already is — and the pointer may not come back in either direction: the item
    it was deferred to and the item filed beside it are both shut.
    """
    block = _tightness_comment()

    assert _SHUT_DOSE_ITEM not in _SRC, \
        "`app/memory_ceiling.py` cites the closed dose-rate item again"
    assert "4121" not in block, \
        "the deferral was re-pointed to another closed item, which is the same defect"

    assert "by decision" in block, \
        "the level is asserted rather than ruled: the reasoning is what outlives a number"
    assert "prices a prospective write" in block, "the notice's own unit is gone"
    assert "prices a day" in block, \
        "the dose's unit is gone, and with it the only reason the two differ"

    # The disproven report-only alarm is the evidence that makes this a ruling and not a
    # preference, and it is easy to misstate: the two clearances come from two reports five
    # weeks apart, and between them the alarm printed and reached nobody — the file did not
    # "grow anyway". A block that credits one item with both figures, or invents growth,
    # cites an event that did not happen; the dates are what make them two readings.
    assert "-> CURATE" in block, "the disproven report-only alarm is left unnamed"
    assert "59 B" in block and "137 B" in block, \
        "the two measured clearances that evidence the failure are gone"
    assert "2026-11-06" in block and "2026-11-14" in block, \
        "the clearances lost the dates that hold them five weeks apart"


def test_the_reopen_trigger_is_a_measurement_anyone_can_take_today():
    """#4303 clause 2: the trigger is a command that answers now, not data to be waited for.

    The old sentence's defect was not that it deferred — deferring a call to an instrument
    is fine — but that it deferred to a series nobody was writing, so the condition for
    reopening could never be observed. What replaces it names the threshold, the window
    that threshold is meaningless without, the script that prints both, and the fact that
    the script's own row store has no writer, so no reader goes looking for a JSONL that
    will not appear. It also names what a fired trigger earns — the direction the constant
    moves, and how far — which is what makes it a decision rule rather than a curiosity.
    """
    block = _tightness_comment()

    assert "headroom_days" in block and "window_days" in block, \
        "the threshold lost either its quantity or the window it is divided by"
    assert "under 1" in block, "the cut the trigger fires at is not stated"
    assert "loaded_memory_dose_rate.py --window-days 30" in block, \
        "the trigger names no command a reader can run"
    assert "loaded-memory-dose-rate.jsonl" in block, \
        "the row store the trigger is NOT read from goes unnamed"
    assert "no writer" in block, \
        "the block still reads as though those rows are accumulating somewhere"
    assert "one edit to this line" in block, \
        "the cost of re-settling — what makes the ruling cheap to revisit — is gone"
    assert "0.95" in block, \
        ("the trigger names a threshold but no action, so a reader who measures it still "
         "has to guess which way the constant moves and how far")


def test_repeated_firings_inside_one_pass_are_recorded_as_intended():
    """#4303 clause 3: several notices in one write pass is the design, and it says so.

    Priced per prospective text and refusing nothing, the notice answers every write in a
    consolidation pass rather than once per pass. A reader who finds nine of them in one
    session will otherwise reach for a suppression this module does not have, so the block
    records the behaviour and the count that evidenced it — with the command that counts
    it, because the figure the item was filed with did not reproduce.

    What this node deliberately does NOT do is re-derive the nine. Those are occurrences of
    the notice's signature in one session transcript under session scratch, which retention
    sweeps; the owed half of the item is what copies those bytes somewhere durable. The node
    pins that the block hands over a counting command with its number and carries no figure
    that no command produces — the filed item's eleven is the one that does not.
    """
    block = _tightness_comment()

    assert "not a defect" in block, "a repeat firing is left to read as a bug"
    assert "nine" in block.lower(), "the reproduced count is gone from the block"
    assert 'grep -o "headroom rung"' in block, \
        "the count lost the command that reproduces it"
    assert "wc -l" in block, "the counting command is truncated to a fragment"
    # The counted phrase exists in the emitted message and nowhere in this file as one
    # literal, because the message is built from two adjacent f-strings. A reader who greps
    # the module for it gets nothing and may conclude the witness is invented, so the block
    # says where the phrase actually lives.
    assert "f-string" in block, \
        "the block leaves a reader to grep this file for a phrase it never stores whole"
    assert "eleven" not in block.lower() and " 11 " not in block, \
        "the count that no command reproduces is back in the block"


def test_the_pinning_docstring_carries_the_same_ruling():
    """#4303 clause 4: the file that guards the number no longer routes a reader onward.

    The module comment and the docstring of the node that pins the tightness carried the
    same deferral, and fixing only the module would leave the dead pointer in the one place
    a reader lands on right after `git grep HEADROOM_WARNING_TIGHTNESS`. A test docstring is
    instruction, not decoration, so it carries the same ruling rather than a reference.
    """
    assert _SHUT_DOSE_ITEM not in _TEST_SRC, \
        "this file still sends a reader to the closed dose-rate item"

    node = next(n for n in ast.walk(ast.parse(_TEST_SRC))
                if isinstance(n, ast.FunctionDef) and n.name == _TIGHTNESS_NODE)
    doc = ast.get_docstring(node) or ""
    assert doc, f"{_TIGHTNESS_NODE} lost its docstring"
    assert "prices a prospective write" in doc and "prices a day" in doc, \
        "the docstring defers again instead of stating the ruling"
    assert "LEVEL" in doc, "the decision the docstring exists to carry is unnamed"


def test_the_comment_round_moved_nothing_it_was_not_asked_to_move():
    """#4303 clause 5: prose changed; the rung, the cited lines and the provenance did not.

    Three things outside the two blocks have to survive a comment edit, and two of them show
    up only as an absence: the constant is still 0.90 (this round is a notice about a rung,
    not a re-tuning of it); the audit's line citations into this file still land on the
    definitions they name, which is the standing reason the block sits at the END of the
    module; and the dose script and its own suite still cite the item that filed the
    instrument. Deleting those last two along with the deferrals is the over-correction — a
    grep that has removed every mention looks exactly like a finished job.
    """
    assert MC.HEADROOM_WARNING_TIGHTNESS == 0.90, "a comment round changed the rung"

    lines = _SRC.splitlines()
    assert lines[113].strip() == "MEMORY_TIGHTNESS = 0.80", \
        f"the audit's `:114` no longer reads what it cites: {lines[113]!r}"
    assert lines[161].startswith("def tight_limit("), \
        f"the audit's `:162` no longer reads what it cites: {lines[161]!r}"
    assert lines[261].startswith("def memory_write_error("), \
        f"the audit's `:262` no longer reads what it cites: {lines[261]!r}"

    for rel in ("scripts/maintenance/loaded_memory_dose_rate.py",
                "tests/test_loaded_memory_dose_rate.py"):
        # Composed, not spelled: a literal here would put the shut item's number back into
        # this file and trip the guard two nodes above.
        assert _SHUT_DOSE_ITEM in (ROOT / rel).read_text(encoding="utf-8"), \
            f"{rel} lost the provenance cite for the item that filed it"
