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
    reused here — 0.90 is a notice's, and #4122 may re-settle it once its headroom-days
    row has data. What may not exist is a number in BYTES: this module has exactly the
    ceilings `prompt_surface` owns and no others, which is the second assertion below.
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
