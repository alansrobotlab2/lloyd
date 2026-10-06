"""ALERT.md must retract on the surface it alarmed (#1967).

`Notifier.alert` writes `~/.local/state/lloyd-guardian/ALERT.md` on **every**
run mode — `_alert_file` is called at `notify.py:191`, above the
`if not self.external` return at `notify.py:192-193` — but `resolve()` opened
with `if not self.external: return True` at `notify.py:397-398` and otherwise
only sealed the **daily note**. The consequence is on disk at triage:
the 2026-09-30 runtime-data alarm is still the live content of that file
(`grep -c 'cleared:'` → 0, body still ordering a reader to remove
`/home/alansrobotlab/lloyd/.t`) while `~/lloyd/.t` has been absent for over a
day, `datawatch.py status` reports `"stray_in_tree": []`, and
`~/obsidian/memory/2026-09-30.md` holds eight `cleared:` lines proving the
hourly all-clear ran and never touched this surface. ALERT.md is the artefact
an agent reads *first*, so a run that inherits it without running `stat` on it
starts from a cause fixed ~18 h earlier — the 2026-10-01 signals headline
asserted the writer "stands unfixed" off exactly these bytes.

The two asymmetries this file pins, and the one the fix must not introduce:

* placement — the retraction has to sit above the `external` gate, exactly as
  the write does, or every drill and `--no-external-alerts` run keeps writing
  an alarm it can never close (`test_the_retraction_runs_below_the_external_gate`);
* scope — ALERT.md is a single-slot, last-writer-wins file (`write_text`,
  `notify.py:305`) whose H1 names its title, so a retraction must be
  title-scoped and must never create a file that isn't there
  (`test_retracting_one_incident_leaves_another_incident_s_bytes_intact`,
  `test_resolve_creates_no_alert_file_when_none_exists`).

The process boundary under test is the file itself: `resolve()` runs inside
`lloyd-guardian.service`, and whatever it writes is read later, in a different
process, by a person or by a signal pass. Every node here therefore judges the
bytes re-read from disk, not anything in memory. (`tests/test_guardian_alert_
stamp.py` owns the zone-marking rule for the timestamps this file's retraction
writes; its `test_the_cleared_stamp_*` nodes pin clause 5.)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
for _p in (str(ROOT), str(GUARDIAN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import notify  # noqa: E402

TITLE = "Runtime data is being written into the code tree"
OTHER = "Tmp headroom is running out"
LEVEL = "error"
#: Shaped like the live 2026-09-30 alarm: imperative instructions at column
#: zero, which is exactly what must not survive as a live directive.
BODY = ("These exist inside the tree again:\n"
        "  /home/alansrobotlab/lloyd/.t\n"
        "\n"
        "Find the writer, move the data across, and remove the in-tree copy.")
#: The retraction body, taken from the producer rather than invented. #2110 split the
#: guardian's one all-clear into two sentences — "moved N inert files on this check"
#: when the guardian relocated residue, and this one, the "absent with no move
#: recorded by the guardian" form, when it measured an empty tree and did nothing —
#: and this node exercises the second, which is the case that has no prior cause on
#: record. `agent-services/guardian/guardian.py` builds both in
#: `_runtime_data_incident`; `tests/test_guardian_inert_stray.py` pins each wording
#: against the producer. What this node owns is the retraction MACHINERY — section
#: closed, ALERT.md entry removed, ledger line — which is wording-independent, so a
#: sentence change here must not silently leave a stale one behind: an earlier revision
#: of this file carried a third variant ("on the latest check"), which no producer
#: emitted and which no node would have noticed drifting further.
NOTE = ("no runtime stores inside the code tree of /repo — absent with no move "
        "recorded by the guardian, so the instructions above are stale and nothing "
        "here accounts for a path that was named and is now gone")


def _notifier(tmp_path: Path, *, external: bool = True) -> notify.Notifier:
    """A Notifier whose whole world is `tmp_path`: ledger, ALERT.md and vault
    all inside it, pointed at a dead backend so no backlog POST can reach the
    live board. Room channels are muted by `tests/conftest.py` for every node,
    so `external=True` fans out into files only — the same shape
    `tests/test_guardian_predicates.py::_stray_incident` runs the whole
    incident cycle in."""
    vault = tmp_path / "obsidian"
    (vault / "memory").mkdir(parents=True, exist_ok=True)
    return notify.Notifier(ledger=tmp_path / "ledger.jsonl",
                           state_dir=tmp_path,
                           vault_root=str(vault),
                           backend_url="http://127.0.0.1:1",
                           external=external)


def _raise(n: notify.Notifier, title: str = TITLE) -> bytes:
    """Fan out one alarm through the shipped `alert()` and hand back the ALERT.md
    bytes it wrote. `alert()`, not `_alert_file()` directly, because the seam
    under test is the whole write path a production tick takes."""
    results = n.alert(LEVEL, title, BODY, coalesce=True)
    assert results["alert_file"] is True, results
    return (n.state_dir / "ALERT.md").read_bytes()


def _lines(tmp_path: Path) -> list[str]:
    return (tmp_path / "ALERT.md").read_text(encoding="utf-8").splitlines()


def _cleared_lines(tmp_path: Path) -> list[str]:
    return [ln for ln in _lines(tmp_path)
            if ln.startswith(f"{notify.ALERT_CLEARED_PREFIX} ")]


def test_retraction_seals_the_alert_it_was_written_for(tmp_path):
    """#1967 clause 1: the file read alone reports the incident closed.

    The acceptance sentence this pins is "reading ALERT.md alone after an
    incident clears says the incident is closed rather than ordering the
    reader to act". Three mechanically checkable halves: a `cleared:` line
    naming the retraction note exists; the imperative text of the alarm no
    longer sits at column zero as a live directive, and is instead present as
    a quotation of the record; and the original `written:` value still names
    when the alarm fired — retraction adds a closing stamp, it does not falsify
    the opening one. All of it judged from the bytes on disk.
    """
    n = _notifier(tmp_path)
    _raise(n)

    written_before = [ln for ln in _lines(tmp_path) if ln.startswith("written: ")]
    assert n.resolve(TITLE, NOTE) is True

    text = (tmp_path / "ALERT.md").read_text(encoding="utf-8")
    cleared = _cleared_lines(tmp_path)
    assert len(cleared) == 1, f"expected one `cleared:` line, got {cleared!r}"
    assert NOTE in cleared[0], cleared[0]
    assert [ln for ln in _lines(tmp_path) if ln.startswith("written: ")] == (
        written_before), "the retraction rewrote the alarm's own `written:` stamp"
    assert text.splitlines()[0] == f"# {TITLE}", "the H1 title moved"
    assert not any(ln.startswith("These exist inside the tree again:")
                   for ln in _lines(tmp_path)), (
        "the alarm's instruction is still presented at column zero as a live "
        "directive")
    assert "\nFind the writer" not in text, (
        "the imperative sentence survived unquoted as a live directive")
    assert "> Find the writer, move the data across, and remove the in-tree copy." in text, (
        "the record of what was ordered should still be readable, quoted")
    assert text.index(notify.ALERT_CLEARED_BANNER) < text.index("\n> Find the writer"), (
        "the banner saying the incident is closed must precede the quoted "
        "instructions, not trail them")


def test_retracting_one_incident_leaves_another_incident_s_bytes_intact(tmp_path):
    """#1967 clause 2: ALERT.md is one slot, so scope the seal by title.

    `resolve()` has several callers with different titles
    (`guardian.py:855` poolwatch, `:917` tmpwatch, `:1017` runtime data), and
    the file is last-writer-wins — so the only safe rule is that a retraction
    touches the file when its H1 names this exact title and not one byte when
    it names another. The fixture raises B; the call retracts A. Byte equality
    of the whole file is the assertion, because a live B alarm that quietly
    lost its directive to A's `cleared:` line would still look fine to A's
    tick while hiding a live incident from every reader of this file.
    """
    n = _notifier(tmp_path)
    before = _raise(n, OTHER)

    assert n.resolve(TITLE, NOTE) is True

    assert (tmp_path / "ALERT.md").read_bytes() == before, (
        "retracting a title the file does not hold modified the live incident "
        "it does hold")
    assert not _cleared_lines(tmp_path), (
        "a retraction for another title stamped this file `cleared:`")


def test_the_retraction_runs_below_the_external_gate(tmp_path):
    """#1967 clause 3: `external=False` writes the alarm, so it must close it.

    `_alert_file` is called at `notify.py:191`, before the
    `if not self.external: return results` at `notify.py:192-193`; `resolve()`
    answered early at `notify.py:397-398`. That asymmetry means every drill and
    `--no-external-alerts` run can write ALERT.md and never retract it — the
    same failure the 09-22 class rule names: a guard on one of two write
    surfaces is not a guard. This node runs the whole cycle with
    `external=False` and asserts both halves of the correct asymmetry: the
    alarm file is sealed, and the vault note behind the gate is still not
    touched — closing the drill's alarm must not smuggle the daily-note
    surface back through the gate.
    """
    n = _notifier(tmp_path, external=False)
    _raise(n)

    assert n.resolve(TITLE, NOTE) is True

    assert len(_cleared_lines(tmp_path)) == 1, (
        "external=False sealed nothing; the drill wrote an alarm it cannot "
        "close and its inherited alert now outranks the board forever")
    assert not any((tmp_path / "obsidian" / "memory").iterdir()), (
        "the external=False path wrote the vault note — the gate must stay "
        "closed on the daily-note surface even as ALERT.md gains a retraction")


def test_resolve_creates_no_alert_file_when_none_exists(tmp_path):
    """#1967 clause 4a: an all-clear must not conjure the file it retracts.

    The hourly all-clear runs whether or not an alarm was ever written
    (`guardian.py:1010-1020`, every `STRAY_CHECK_SECONDS`); if `resolve()`
    created ALERT.md to write a `cleared:` line into it, the guardian would
    manufacture an artefact that reads, to the next agent who opens the state
    dir, as an incident that happened and closed. No file, no write, silent
    success.
    """
    n = _notifier(tmp_path)
    assert not (tmp_path / "ALERT.md").exists(), "fixture: nothing has alerted"

    assert n.resolve(TITLE, NOTE) is True

    assert not (tmp_path / "ALERT.md").exists(), (
        "resolve() created an ALERT.md for an incident that never alerted")


def test_the_second_all_clear_leaves_the_sealed_file_byte_identical(tmp_path):
    """#1967 clause 4b: retraction is idempotent, as the daily note's is.

    The stray check fires hourly and the condition stays clear, so `resolve()`
    runs against an already-sealed file every hour after the first — the same
    idempotent-silence contract
    `test_the_set_emptying_writes_one_cleared_line_and_stays_silent_after`
    holds for the daily note, and for the same reason: the heartbeat reads
    `resolve()`'s True, and a second `cleared:` line every hour would rebuild
    the exact noise #1536 removed. Byte equality plus exactly one `cleared:`
    line is the assertion.
    """
    n = _notifier(tmp_path)
    _raise(n)
    assert n.resolve(TITLE, NOTE) is True
    sealed = (tmp_path / "ALERT.md").read_bytes()

    assert n.resolve(TITLE, NOTE) is True

    assert (tmp_path / "ALERT.md").read_bytes() == sealed, (
        "the second all-clear modified an already-retracted alert file")
    assert len(_cleared_lines(tmp_path)) == 1, "the second all-clear added a stamp"


# ---------------------------------------------------------------- the firing family
#
# Everything above exercises `resolve` directly, which is how #1536 shipped: the only
# family that called it was tree-strays, and the only test was of the notifier. The two
# families that ACTUALLY fire — `Service down, but no promotion to revert` (18 sections
# across the dated notes) and `supervisord was unreachable` (7) — never called it, and
# never could have: their sections were written without `coalesce`, so no open marker was
# ever in them for `resolve` to seal. That is #2221 clause 1 and clause 2, and a
# notifier-level test cannot express it, because the defect is a call-site option. So
# these nodes drive `tick()` with I/O stubbed and read the note.
SERVICE_DOWN = "Service down, but no promotion to revert"
PROGRAM = "lloyd-mc:lloyd-backend"
FATAL = (f"{PROGRAM}: FATAL: can't find command "
         "'/home/alansrobotlab/lloyd/.venvs/lloyd/bin/python'")
HEAD = "0" * 40


def _statvfs(used_inodes: int, *, inodes: int = 1_048_576,
             used_blocks: int = 1_101_005, blocks: int = 33_292_288):
    """A `statvfs` answer with this box's /tmp geometry: 1,048,576 inodes over 126 GiB.

    The inode count is what fills here while the bytes stay nearly empty (4.2 of
    126 GiB used), which is the reading `df -h` hides. Same fake shape as
    `tests/test_guardian_tmpwatch.py::_statvfs`; the readings callers pass are ones
    `df -i /tmp` has actually reported on this machine.
    """
    def fake(_path):
        return types.SimpleNamespace(f_files=inodes, f_ffree=inodes - used_inodes,
                                     f_blocks=blocks, f_bfree=blocks - used_blocks,
                                     f_frsize=4096)
    return fake


def _ticking(tmp_path, monkeypatch, *, liveness):
    """A Guardian whose `tick()` can be driven, with its note in `tmp_path`.

    Built the way `test_guardian_predicates` builds one (same args shape, same stub set),
    plus two things that file does not need: `alert` forwards to the REAL notifier instead
    of recording, because the claim is about the bytes in the daily note, and `vault_root`
    is moved off the live vault, because a test that journalled a fake FATAL into
    `~/obsidian/memory/` would be writing the exact kind of alarm this item is about.

    `liveness` is a function of the tick number so a test can say "down, then healthy"
    without reaching into the guardian's private streak counters.

    `alert` forwards to the notifier instead of going through `Guardian.alert`, which is
    the harness's one deliberate shortcut and it needs stating: that wrapper suppresses a
    repeat of one title inside `policy.ALERT_REPEAT_SECONDS`, so a real guardian's SECOND
    call for one incident arrives only after that window has passed. What the clauses are
    about is what the notifier does with two calls for one incident, so the forward is the
    two calls the notifier receives across a window — and it is the only way to observe the
    second one in a test that does not sleep for the window.
    """
    import types

    import guardian as G
    import rollback as RB

    memory = tmp_path / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    args = types.SimpleNamespace(
        repo=str(tmp_path), state=str(tmp_path / "state"),
        guardian_state=str(tmp_path / "gstate"), supervisor_sock="/nonexistent",
        backend_url="http://127.0.0.1:1/health", mcp_url="http://127.0.0.1:2/health",
        programs=PROGRAM, interval=5.0,
    )
    g = G.Guardian(args)
    monkeypatch.setattr(g.state, "current", lambda: None)      # nothing under observation
    monkeypatch.setattr(g.state, "lkg", lambda: {"commit": HEAD})
    monkeypatch.setattr(g.state, "rollback_target", lambda: (HEAD, "test"))
    monkeypatch.setattr(g.state, "is_broken", lambda: False)
    monkeypatch.setattr(g.state, "pause_remaining", lambda cap: 0.0)
    monkeypatch.setattr(RB, "head_commit", lambda repo: HEAD)
    monkeypatch.setattr(g, "collect", lambda: {"now": 1_700_000_000.0, "supervisord": "ok",
                                               "procs": {}, "probes": {}})
    monkeypatch.setattr(g, "heartbeat", lambda *a, **k: None)
    monkeypatch.setattr(g, "do_rollback", lambda *a: True)
    monkeypatch.setattr(g.notifier, "vault_root", tmp_path)

    # The /tmp the tick measures is the one this test hands it, not the host's.
    # `check_tmp` (guardian.py:1316-1332) calls `notifier.alert(..., coalesce=True)` on
    # every reading over `tmpwatch.WARN_FRACTION` (0.80), and a coalesced section is a
    # SECOND `DAILY_STILL_OPEN` in the note. That is what put the three #2290 nodes red
    # on 2026-10-06 while the box sat at 81% (`df -i /tmp`: 848,936 of 1,048,576) — the
    # nodes read the note, not the reading, so the host's weather decided this file's
    # colour. Every other host read inside a tick is already the test's (`collect`,
    # `evaluate_liveness`, the subprocess runner); this is that same stub set covering
    # the one probe that was missed. `check_tmp` itself still runs, for real, on the
    # reading below.
    monkeypatch.setattr(g.tmp, "statvfs", _statvfs(10_000))
    calls: list[tuple[tuple, dict]] = []

    def _alert(*a, **k):
        calls.append((a, k))
        g.notifier.alert(*a, **k)          # the real one: the note is the artefact

    monkeypatch.setattr(g, "alert", _alert)
    # One read per tick is what the decision path does (`evaluate_liveness` is called
    # once, at the top of the branch), so the counter rides the read rather than a
    # method the guardian does not have: the test's "tick N" is literally the Nth
    # liveness read, which is the thing clause 4 counts.
    ticks = {"n": 0}

    def _liveness(snap):
        answer = liveness(ticks["n"])
        ticks["n"] += 1
        return answer

    monkeypatch.setattr(g, "evaluate_liveness", _liveness)
    return g, calls


def _note(tmp_path: Path) -> str:
    mem = sorted((tmp_path / "memory").glob("*.md"))
    assert len(mem) == 1, f"expected one daily note, got {[str(m) for m in mem]}"
    return mem[0].read_text(encoding="utf-8")


DOWN = (True, FATAL)
HEALTHY = (False, f"{PROGRAM} RUNNING, {2 * PROGRAM} RUNNING")


def test_the_service_down_section_is_coalesced_so_one_incident_is_one_section(tmp_path,
                                                                              monkeypatch):
    """#2221 clause 1: repeat checks of ONE down incident refresh one section.

    Two ticks, both seeing the same FATAL program, with nothing under observation so the
    guardian cannot roll back and only alerts. The claim is about the note: one heading
    for this title and one open marker, not two. It matters because the marker is what
    `resolve` seals — write the alarm twice without it and the incident has two
    unsealable sections, which is precisely the state 18 dated-note blocks are in today.
    """
    g, calls = _ticking(tmp_path, monkeypatch, liveness=lambda n: DOWN)

    assert g.tick() == "down_unobserved"
    assert g.tick() == "down_unobserved"

    assert calls[0][1].get("coalesce") is True, (
        "clause 1: the alert must be written coalesced, or the section carries no open "
        f"marker for a later recovery to seal: kwargs={calls[0][1]}")
    text = _note(tmp_path)
    assert text.count(f"## Self-mod guardian: {SERVICE_DOWN}") == 1, (
    "the second check of the same incident appended a second section:\n" + text)
    assert text.count(notify.DAILY_STILL_OPEN) == 1, text


def test_the_first_healthy_tick_seals_the_section_naming_the_program(tmp_path, monkeypatch):
    """#2221 clause 2: the first healthy liveness read retracts, in the same note.

    The sealed line must NAME the program that came back. The generic `cleared: (the
    condition is no longer observed)` that `resolve` uses when given no text is the
    sentence that makes a reader check whether the alert was real at all — and this
    family's whole history in the notes (`memory/2026-09-29.md`'s `lloyd-mc:lloyd-backend:
    FATAL`, no retraction, service RUNNING for days) is what it costs.
    """
    g, _calls = _ticking(tmp_path, monkeypatch,
                         liveness=lambda n: DOWN if n == 0 else HEALTHY)

    assert g.tick() == "down_unobserved"
    assert g.tick() == "armed"

    text = _note(tmp_path)
    assert notify.DAILY_STILL_OPEN not in text, text
    cleared = [ln for ln in text.splitlines() if ln.startswith(notify.DAILY_CLEARED_PREFIX)]
    assert len(cleared) == 1, f"expected exactly one retraction line: {cleared}"
    assert PROGRAM in cleared[0], cleared[0]


def test_the_second_healthy_tick_writes_nothing_at_all(tmp_path, monkeypatch):
    """#2221 clause 2, second half: recovery is one event, not a per-tick sentence.

    Byte identity is the assertion, not another `cleared:` count, because #2130 found an
    identical claim that passed on a call whose effect was a no-op — the file bytes are
    the effect, so the file bytes are what is compared.
    """
    g, _calls = _ticking(tmp_path, monkeypatch,
                         liveness=lambda n: DOWN if n == 0 else HEALTHY)
    g.tick()          # down
    assert notify.DAILY_STILL_OPEN in _note(tmp_path), (
        "precondition: a section that never opened would leave the byte identity below "
        "true of a note that was never written to at all — the vacuous green #2130 names. "
        "This node has to see the section open before it can say a later tick left it "
        "sealed and alone.")
    g.tick()          # healthy: seals
    after_first = _note(tmp_path)
    assert notify.DAILY_CLEARED_PREFIX in after_first, (
        f"precondition: nothing had been sealed, so a write on tick three had nothing to "
        f"duplicate:\n{after_first}")

    assert g.tick() == "armed"
    assert _note(tmp_path) == after_first, "the third tick wrote to the note again"


# ------------------------------------------------------- the historical half (clause 5)
#
# The go-forward fix cannot reach a block older than `notify.DAILY_SCAN_DAYS` = 3, so the
# 15 dated-note blocks that a live `supervisorctl status` refutes today need a separate
# pass. That pass is `scripts/maintenance/guardian_note_sweep.py`, and the reason it writes
# `dated:` where the guardian writes `cleared:` is the reason these nodes exist at all:
# #2221's own example — the 09-29 note telling a human that `lloyd-mc:lloyd-backend` is
# FATAL while that program has been RUNNING for days — must not be answered with a fake
# observation of a recovery that nobody watched.
import importlib.util as _ilu
import types

import pytest

_SWEEP_PATH = ROOT / "scripts" / "maintenance" / "guardian_note_sweep.py"
_spec = _ilu.spec_from_file_location("guardian_note_sweep", _SWEEP_PATH)
SWEEP = _ilu.module_from_spec(_spec)
sys.modules["guardian_note_sweep"] = SWEEP
_spec.loader.exec_module(SWEEP)

DOWN_BODY = ("lloyd-mc:lloyd-backend: FATAL: can't find command "
             "'/home/alansrobotlab/lloyd/.venvs/lloyd/bin/python'\n\n"
             "HEAD is 8328347d and no self-modification is being observed, so this is "
             "infrastructure rather than a bad change. Not rewriting history — this needs "
             "a human.\n")
HIST_NOTE = ("---\nsegment: memory\n---\n"
        "## Self-mod guardian: Service down, but no promotion to revert\n\n"
        + DOWN_BODY)
RUNNING = {"lloyd-mc:lloyd-backend", "lloyd-mc:lloyd-mcp"}


def test_the_sweep_dates_a_refuted_block_without_claiming_a_recovery():
    """#2221 clause 5's instrument: a refuted block gains `dated:`, never `cleared:`.

    Three assertions, one per lie the sweep could tell. It must find the block. It must
    write `dated:` and NOT `cleared:` — the second would assert an observed recovery, and
    the whole reason the go-forward fix cannot reach this note is that nobody is around to
    observe a recovery of a 2026-09-29 alarm. And it must leave the body's own words alone:
    the outage happened, the record stands, and only the standing instruction is answered.
    """
    assert SWEEP.count_refutations(HIST_NOTE, RUNNING) == 1, "the sweep must find the refuted block"

    new, n, notes = SWEEP.date_note(HIST_NOTE, RUNNING, "2026-10-05")
    assert n == 1 and len(notes) == 1, (n, notes)
    assert notify.DAILY_DATED_PREFIX in new, new[-400:]
    assert notify.DAILY_CLEARED_PREFIX not in new, (
        "a dating that says `cleared:` reports a recovery somebody watched, which is the "
        "false all-clear this item is fixing, in the opposite direction")
    assert DOWN_BODY in new, "the record of the outage is not rewritten"
    assert "RUNNING" in new and "`lloyd-mc:lloyd-backend`" in new, (
        "the dating must name the program and the read that refutes the state, or a "
        "reader has nothing to check it against")
    assert new.count("## ") == HIST_NOTE.count("## "), "one block in, one block out"


def test_a_block_the_sweep_dates_is_never_dated_again():
    """Idempotence, because this runs by hand over published notes.

    A sweep that appends on every run turns one correction into N, which is the failure
    mode of every remediation pass over a file nobody re-reads. `count_refutations` going
    to zero after the pass is also the item's proving check, so this node is asserting the
    same number the acceptance clause does.
    """
    once, n1, _ = SWEEP.date_note(HIST_NOTE, RUNNING, "2026-10-05")
    assert n1 == 1
    assert SWEEP.count_refutations(once, RUNNING) == 0, "second pass must find nothing"

    twice, n2, _ = SWEEP.date_note(once, RUNNING, "2026-10-06")
    assert n2 == 0 and twice == once, "the second pass rewrote a dated block"


def test_a_block_naming_a_program_that_is_not_running_is_left_alone():
    """The safety property, and the one that must not be traded for a cleaner count.

    On a box where the service really is down, the sweep has to say nothing. Asserting the
    refusal with the same helper the CLI uses is what stops a future edit from "fixing" a
    non-match by dating the block anyway to make the sweep report zero.
    """
    assert SWEEP.count_refutations(HIST_NOTE, {"lloyd-mc:lloyd-mcp"}) == 0, (
        "a program the supervisor does not report RUNNING has not been refuted")
    assert SWEEP.count_refutations(HIST_NOTE, set()) == 0
    text, n, _ = SWEEP.date_note(HIST_NOTE, {"lloyd-mc:lloyd-mcp"}, "2026-10-05")
    assert n == 0 and text == HIST_NOTE


def test_a_block_that_is_not_an_alert_is_never_touched(tmp_path):
    """Only a guardian alert BLOCK is a candidate — which includes a rollback notice.

    The distinction the sweep has to get right is a BLOCK TYPE, not a family: a `## Sessions`
    section that quotes a log line is not an alarm and never a candidate, while a
    `Self-mod guardian: Rolled back …` block IS one, because the 15-block refutation set
    #2221 measured includes the two rollback notices on 2026-09-06 — they assert
    `lloyd-mc:lloyd-backend: FATAL` to a reader today. Dating a historical block answers the
    stale instruction; whether the rollback family should gain a go-forward retraction is the
    ruling this item defers, and the sweep does not make it.

    The session half is why the title test cannot be a substring match over the whole text:
    a captured summary that quotes an alert title would otherwise become a candidate.
    """
    rollback = ("## Self-mod guardian: Rolled back 26574f87 → cb381484\n\n"
                "Rolled back 26574f87 → cb381484\n\nTrigger: crash\n"
                "lloyd-mc:lloyd-backend: FATAL: Exited too quickly\n")
    session = ("## Sessions\n\n### Session 11:24\n\n"
               "The agent read `lloyd-mc:lloyd-backend: FATAL` in a log.\n"
               "It also saw the heading `Self-mod guardian: Service down, but no promotion "
               "to revert` quoted in a note.\n")
    everything = rollback + session + HIST_NOTE[len("---\nsegment: memory\n---\n"):]

    assert SWEEP.count_refutations(everything, RUNNING) == 2, (
        "the alert block and the rollback notice are the candidates — the session section "
        "quotes a log line and an alert heading and must not become one")
    new, n, notes = SWEEP.date_note(everything, RUNNING, "2026-10-05")
    assert n == 2, notes
    assert any("Service down" in t for t in notes) and any("Rolled back" in t for t in notes), notes
    assert session in new, "the session section was rewritten"


def test_the_supervisor_table_is_read_for_program_names_not_for_its_exit_code():
    """`supervisorctl status` exits 3 when anything is stopped, which is the normal answer.

    Treating a nonzero rc as a failed read would make the sweep blind on exactly the boxes
    where an alert family is firing, and blind means silent: no table parsed, no
    refutations, "0 hits" reported. So the read is judged on the table it produced, and an
    empty one is refused rather than reported clean.

    The read is exercised through `read_running`'s `runner` argument rather than a real
    `supervisorctl`, and that is deliberate: the gate runs the suite where no supervisor is
    answering, so a node that shelled out could only assert this on some environments —
    which is how #2221's own two nodes came to be red at `cb235020` while passing on the
    box. The property is about parsing and refusal, and a stand-in answers those the same
    on every box; the live read stays the sweep's own job, where a failure stops it loudly.
    """
    table = ("lloyd-mc:lloyd-backend           RUNNING   pid 3841854, uptime 4:46:13\n"
             "agent-llm-secondary               STOPPED   Not started\n"
             "lloyd-mc:lloyd-mcp                RUNNING   pid 3841394, uptime 4:46:29\n")
    assert SWEEP.running_programs(table) == {"lloyd-mc:lloyd-backend", "lloyd-mc:lloyd-mcp"}

    def _res(stdout, rc):
        return types.SimpleNamespace(stdout=stdout, stderr="", returncode=rc)

    # rc 3 means "something is stopped", the ordinary case: the table still counts.
    got = SWEEP.read_running(Path("pytest.ini"),
                             runner=lambda *a, **k: _res(table, 3))
    assert got == {"lloyd-mc:lloyd-backend", "lloyd-mc:lloyd-mcp"}, (
        "a nonzero rc must not discard a table it can read")

    with pytest.raises(RuntimeError):
        SWEEP.read_running(Path("pytest.ini"), runner=lambda *a, **k: _res("", 4))
    with pytest.raises(RuntimeError):
        SWEEP.read_running(Path("pytest.ini"),
                           runner=lambda *a, **k: (_ for _ in ()).throw(OSError("no unit")))


# The 2026-09-29 note is the one the item names, so its pre-sweep bytes are committed: the
# fixture is the "before" the acceptance check needs, and it survives the sweep of the note
# itself, which is what makes the pair below a measurement rather than a wish.
PRE_SWEEP = Path(__file__).resolve().parent / "fixtures" / "guardian_note_2026-09-29_pre-sweep.md"
LIVE_NOTE = Path.home() / "obsidian" / "memory" / "2026-09-29.md"


def test_the_note_the_item_names_had_one_refutable_block_and_the_sweep_clears_it():
    """#2221 clause 5's proving check, both directions, on one file.

    The fixture side is the "before" and it cannot rot: the bytes are the note as committed
    at the time of triage, and it asserts the FATAL block the item quotes is there, is
    unsealed, and is exactly ONE refutable block under a supervisor that reports the program
    RUNNING. The live side is the "after" — the note in the vault must carry a dating and
    must present zero refutable blocks — and it is guarded on the file being readable,
    because the gate points `HOME` at the round home where `~/obsidian` does not exist. A
    node that skipped there would report the clause green without ever reading the note, so
    which branch ran is printed, and the fixture half asserts unconditionally.
    """
    pre = PRE_SWEEP.read_text(encoding="utf-8")
    assert "lloyd-mc:lloyd-backend: FATAL" in pre, PRE_SWEEP
    # Asked of the BLOCK, not the file: this note also holds a stray-family block that was
    # legitimately retracted, and `not in pre` over the whole text would be a claim about
    # the note rather than about the alarm the clause is about.
    service_down = [body for title, _s, body in
                    ((t, st, pre[st:en]) for t, st, en in SWEEP._guardian_sections(pre))
                    if "Service down" in title]
    assert len(service_down) == 1, service_down
    assert not SWEEP.already_sealed(service_down[0]), (
        "the fixture is supposed to be the unsealed 'before'")
    assert SWEEP.count_refutations(pre, {"lloyd-mc:lloyd-backend"}) == 1, (
        "the item's own example must be one refutable block in the bytes this node reads")

    swept, n, notes = SWEEP.date_note(pre, {"lloyd-mc:lloyd-backend"}, "2026-10-05")
    assert n == 1 and "Service down" in notes[0], notes
    assert SWEEP.count_refutations(swept, {"lloyd-mc:lloyd-backend"}) == 0

    if LIVE_NOTE.is_file():
        live = LIVE_NOTE.read_text(encoding="utf-8")
        assert notify.DAILY_DATED_PREFIX in live, (
            f"{LIVE_NOTE} still carries no dating: the sweep has not been applied to it")
        # The supervisor read is guarded, and the reason is this round's own history: the
        # gate runs the suite with HOME pointed at the round home, which holds a PARTIAL
        # vault — so `LIVE_NOTE.is_file()` is true there while `supervisorctl` may not
        # answer at all. Unguarded, that made this node red at `cb235020` for a reason that
        # has nothing to do with the clause: a node whose green depends on whether the box
        # answering is the real box is measuring the environment. The dating above is
        # asserted either way, and the sweep's own apply run is what measured 0 refutable
        # blocks against a table with 11 RUNNING programs.
        try:
            running = SWEEP.read_running(ROOT / "agent-services" / "supervisor"
                                         / "supervisord.conf")
        except RuntimeError as exc:
            print(f"#2221 clause 5: no supervisor read here ({exc}); the dating above "
                  "still stands and the live refutation count is what the apply measured")
        else:
            assert SWEEP.count_refutations(live, running) == 0, \
                f"{LIVE_NOTE} still has a refutable block"
            print(f"#2221 clause 5: {LIVE_NOTE} measured against {len(running)} RUNNING "
                  "programs, 0 refutable blocks")
    else:
        print(f"#2221 clause 5: {LIVE_NOTE} is not readable from this home; the fixture "
              "halves above were asserted and the live note is what the sweep applies to")


@pytest.mark.live_vault
def test_the_refused_round_s_witness_is_committed_and_says_zero_real_failures():
    """Clause 6 (the auto-added witness clause): the gate report this item was resumed
    from is in git, and a reader holding only the item can re-derive from it that the
    refusal was the skip ceiling and not the diff.

    Read-only on the vault, so `live_vault`-marked and skipped by the gate's rungs —
    a re-write of the committed file by any later job reddens it on the Pre-Flight rung
    that does run it. What it asserts is exactly the numbers #2221 quotes: `tests` NOT
    ok at 43 skipped over limit 40, with `base_probe` saying all 10 failures were
    already failing at the base, and `red_tree_item` 2218 owning them. The last
    assertion is the one that keeps this witness from becoming the next collision: the
    clause's literal name is `backlog/data/gate.json`, and that path is
    round SM_20260930_063800's witness for #1878/#1883, pinned at (34, 1143) by
    tests/test_automod_spec.py — writing there would destroy that record.
    """
    import json

    witness = (Path.home() / "obsidian" / "backlog" / "data"
               / "2026-10-05.2221-gate-witness.json")
    assert witness.name != "gate.json"
    w = json.loads(witness.read_text(encoding="utf-8"))

    assert w["witness_of"].endswith("rounds/SM_20261005_091450/gate.json")
    assert w["round_id"] == "SM_20261005_091450"
    assert w["head"] == "172c8c7dda44810e1219e72856c41c03b46891ae"
    assert w["ok"] is False

    by_name = {r["name"]: r for r in w["rungs"]}
    assert w["rung_count"] == len(w["rungs"]) == 5, (
        f"the extract holds {w['rung_count']} of {len(w['rungs'])} rungs — a partial "
        "report is the -latest-pointer defect, not a witness"
    )
    assert [by_name[n]["ok"] for n in ("preflight", "vet", "static", "frontend")] == \
        [True] * 4
    assert by_name["tests"]["ok"] is False
    assert "43 tests skipped (limit 40)" in by_name["tests"]["detail"]

    d = w["tests_rung_data"]
    assert (d["passed"], d["failed"], d["errors"], d["tests_skipped"]) == (
        16120, 10, 0, 43), "the figures #2221 quotes are not the committed ones"
    assert d["red_tree_item"] == 2218
    assert "all 10 already failing" in d["base_probe"], (
        "nothing in the committed bytes says the 10 failures pre-existed the diff"
    )


def test_the_tmp_reading_a_driven_tick_sees_is_the_one_the_builder_hands_it(tmp_path,
                                                                           monkeypatch):
    """#2290: this file's note holds only the incident the test drives, in any weather.

    The two clause nodes above count `DAILY_STILL_OPEN` across the whole note. On
    2026-10-06 they counted 2, because `check_tmp` measured the host's own `/tmp` —
    848,936 of 1,048,576 inodes, 81%, over `tmpwatch.WARN_FRACTION` — and journalled a
    second coalesced section beside the service-down one. Two halves pinned here,
    because the first alone would keep passing if the alerting mechanism itself had
    been what changed:

    * `_ticking` hands its guardian a reading below the line, so after the two down
      ticks the clause node drives the note carries ONE open marker and no `/tmp`
      title;
    * hand the same builder's guardian today's over-the-line reading and the SAME two
      ticks produce TWO markers with the `/tmp` title in the note — the exact red,
      reproduced from the reading rather than from the weather.
    """
    import tmpwatch as TW

    (tmp_path / "quiet").mkdir()
    (tmp_path / "full").mkdir()

    quiet, _ = _ticking(tmp_path / "quiet", monkeypatch, liveness=lambda n: DOWN)
    assert quiet.tmp.statvfs is not TW.os.statvfs, (
        "`_ticking` handed its guardian the host's own probe again: that reading is "
        "#2290's entire cause, and it decides this file's colour by weather")
    assert TW.measure(quiet.tmp.path, quiet.tmp.statvfs).fraction < TW.WARN_FRACTION, (
        "the reading the builder hands must be below the line the alert fires over")

    quiet.tick()
    quiet.tick()
    text = _note(tmp_path / "quiet")
    assert TW.ALERT_TITLE not in text, text
    assert text.count(notify.DAILY_STILL_OPEN) == 1, text

    full, _ = _ticking(tmp_path / "full", monkeypatch, liveness=lambda n: DOWN)
    monkeypatch.setattr(full.tmp, "statvfs", _statvfs(848_936))   # 81%, 2026-10-06
    full.tick()
    full.tick()
    noisy = _note(tmp_path / "full")
    assert TW.ALERT_TITLE in noisy, noisy
    assert noisy.count(notify.DAILY_STILL_OPEN) == 2, (
        "over the line the note must hold the incident's own marker AND the tmp "
        f"section — that pair is the count the two clause nodes were red at: {noisy}")
