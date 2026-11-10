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

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


def _notifier(tmp_path: Path, *, external: bool = True,
              backend_url: str = "http://127.0.0.1:1") -> notify.Notifier:
    """A Notifier whose whole world is `tmp_path`: ledger, ALERT.md and vault
    all inside it, pointed at a dead backend so no backlog POST can reach the
    live board. Room channels are muted by `tests/conftest.py` for every node,
    so `external=True` fans out into files only — the same shape
    `tests/test_guardian_predicates.py::_stray_incident` runs the whole
    incident cycle in.

    `backend_url` is a parameter for the #3757 nodes, which need the board to
    ANSWER: their claim is about how many requests it receives, and a discarded
    connection cannot count them.
    """
    vault = tmp_path / "obsidian"
    (vault / "memory").mkdir(parents=True, exist_ok=True)
    return notify.Notifier(ledger=tmp_path / "ledger.jsonl",
                           state_dir=tmp_path,
                           vault_root=str(vault),
                           backend_url=backend_url,
                           external=external)


def _raise(n: notify.Notifier, title: str = TITLE, body: str = BODY) -> bytes:
    """Fan out one alarm through the shipped `alert()` and hand back the ALERT.md
    bytes it wrote. `alert()`, not `_alert_file()` directly, because the seam
    under test is the whole write path a production tick takes."""
    results = n.alert(LEVEL, title, body, coalesce=True)
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


# ── #3019: a seal that does not need the process that raised the alarm ──────
# Everything above this line seals an incident from inside the process that alerted
# for it, which is the only shape `resolve` has ever had a caller for. These two
# nodes are the other shape: the file is still on disk, the process that wrote it is
# not, and what is left is a Guardian that has never measured the cause.
TMP_BODY = ("/tmp: 845,813 of 1,048,576 inodes used (81%); 6.8 of 126 GiB (5%).\n"
            "\n"
            "Look before deleting: `df -i /tmp`, then `du --inodes -x -d1 /tmp | "
            "sort -n | tail`.")


def test_a_process_that_never_raised_the_alarm_seals_the_incident_it_inherited(tmp_path,
                                                                             monkeypatch):
    """#3019 clause 1: ALERT.md outlives its writer, so the seal may not need a latch.

    Every guard retracts on state inside its own process — `tmpwatch.TmpWatch.open_level`,
    `_sup_unreachable_open`, `_down_programs` — and a restart zeroes all three. The new
    process's first tick reads `/tmp`, finds the cause gone, and correctly reports
    `none`, because nothing in it knows an alarm was ever raised; the previous
    process's instructions therefore stay a live directive forever. That is this box at
    filing: `/tmp is running out of inodes` written 2026-10-15T08:33:02.547375-07:00
    with `grep -c 'cleared:'` → 0 while `df -i /tmp` reads 45,203 of 1,048,576 (4.3%)
    against `tmpwatch.CLEAR_FRACTION = 0.70`, in a guardian that started Sat Oct 17
    03:13:19 2026 and so could not have raised it.

    The incident is raised through the shipped `alert()`, then reconciled by a Guardian
    whose `open_level` is still None — a NEW process, not another tick of the one that
    alerted. The bytes must show the shipped seal: one `cleared:` header key carrying
    the retraction sentence, `ALERT_CLEARED_BANNER` above the alarm text now quoted
    for the record, and the original `level:`/`written:` keys copied byte-for-byte so
    a reader can still age the alarm itself.
    """
    import tmpwatch as TW

    g, _ = _ticking(tmp_path, monkeypatch, liveness=lambda n: HEALTHY)
    path = tmp_path / "gstate" / notify.ALERT_FILE_NAME

    raised = g.notifier.alert("error", TW.ALERT_TITLE, TMP_BODY, coalesce=True)
    assert raised["alert_file"] is True, raised
    before = notify.Notifier._alert_file_split(path.read_text(encoding="utf-8"))
    assert before is not None and before[0] == f"# {TW.ALERT_TITLE}", before
    assert g.tmp.open_level is None, (
        "the reconciling guardian must hold no latch. A process that DID raise the "
        "alarm is the case the tick nodes above already own; inheriting one is the "
        "case that has never had a test, and is the case that is broken")

    assert g.reconcile_stranded_alert() == "sealed"

    text = path.read_text(encoding="utf-8")
    after = notify.Notifier._alert_file_split(text)
    assert after is not None and after[0] == f"# {TW.ALERT_TITLE}", "the H1 title moved"
    cleared = [k for k in after[1] if k.startswith(f"{notify.ALERT_CLEARED_PREFIX} ")]
    assert len(cleared) == 1, f"expected one closing key, got {after[1]!r}"
    assert after[1][:len(before[1])] == before[1], (
        "the original header keys must survive byte-for-byte — a reader ages the "
        f"`written:` stamp to tell a stale alarm from a fresh one\n"
        f"  before: {before[1]!r}\n  after:  {after[1]!r}")
    assert TW.clear_body(TW.measure(g.tmp.path, g.tmp.statvfs)) in cleared[0], (
        "the closing line must state the reading that closed it, in the same words "
        f"the in-process all-clear uses: {cleared[0]}")
    assert notify.ALERT_CLEARED_BANNER in text, "the record/banner split is the seal"
    assert text.index(notify.ALERT_CLEARED_BANNER) < text.index("> /tmp: 845,813"), (
        "the banner closes the incident BEFORE the text that ordered the reader to "
        "act, or the file still reads as a directive with a footnote")
    assert not any(ln.startswith("Look before deleting:") for ln in text.splitlines()), (
        "the alarm's own imperative survives, so it must survive quoted — at column "
        "zero it is still an instruction")


def test_an_incident_this_process_cannot_clear_is_left_byte_identical(tmp_path,
                                                                      monkeypatch):
    """#3019 clause 3: reconciliation never retracts a live incident, and never guesses.

    Two ways an inherited ALERT.md is NOT this process's to close, and the cost of
    getting either wrong is an all-clear nobody observed — strictly worse than the
    stranded alarm, because the stranded one at least tells the truth:

    * the title is one of the three it owns, and that guard's clear condition does
      not hold. `/tmp` at 845,813 of 1,048,576 inodes (81%, the reading this box
      actually reported when the alarm fired) is over `CLEAR_FRACTION`, so the
      incident is live and the file is untouched;
    * the title is not one of them at all. Runtime data is the other resident of the
      single slot and its recovery is a measured read, not a latch, so it is
      deliberately absent from the reconciliation map — and a /tmp that IS clear must
      not become a licence to seal whatever the file happens to name.

    Byte-identical is the claim, `written:` stamp and alarm body included, and the
    daily note gains no `cleared:` line either: a retraction that reaches one surface
    and not the other is the #1967 asymmetry this file exists to keep shut.
    """
    import tmpwatch as TW

    g, _ = _ticking(tmp_path, monkeypatch, liveness=lambda n: HEALTHY)
    path = tmp_path / "gstate" / notify.ALERT_FILE_NAME
    assert g.notifier.alert("error", TW.ALERT_TITLE, TMP_BODY,
                            coalesce=True)["alert_file"] is True
    monkeypatch.setattr(g.tmp, "statvfs", _statvfs(845_813))    # 81%: still over the line
    live_bytes = path.read_bytes()
    note_before = _note(tmp_path)

    assert g.reconcile_stranded_alert() == "still-open"
    assert path.read_bytes() == live_bytes, "a live incident was edited by a reconciliation"
    assert _note(tmp_path) == note_before, "and its daily note gained a retraction"
    assert notify.DAILY_CLEARED_PREFIX not in _note(tmp_path)

    # A different incident entirely, in the same slot, with /tmp clear again.
    assert g.notifier.alert("error", TITLE, BODY, coalesce=True)["alert_file"] is True
    monkeypatch.setattr(g.tmp, "statvfs", _statvfs(45_203))     # 4.3%: this one IS clear
    foreign_bytes = path.read_bytes()

    assert g.reconcile_stranded_alert() == "nothing-to-reconcile"
    assert path.read_bytes() == foreign_bytes, (
        "a clear condition holding for one guard must not seal a different guard's "
        "incident — ALERT.md is one last-writer-wins slot and this is the guard "
        "that keeps that from mattering")



# ── the tripwires: an alarm a later tick can close (#3106) ──────────────────
#
# The two `critical` titles this file had no node for were the only two alarms in
# the guardian with no resolver of any kind. Each was spelled once, inside its own
# `self.alert(...)` call, and nowhere else in the tracked tree, while every
# `notifier.resolve` site named some other title — poolwatch's, tmpwatch's,
# `RUNTIME_DATA_ALERT_TITLE`, `SUPERVISOR_ALERT_TITLE`, `SERVICE_DOWN_ALERT_TITLE`,
# and the map-constrained variable in `reconcile_stranded_alert`. So the two
# families whose body orders a human to "restore into a side directory … swap it in
# with the stack stopped" were exactly the two that could never be retracted, while
# the one whose remedy is moving a stray file was retracted constantly.
# `vaultwatch.clear()` / `datawatch.clear()` — what that human then runs — unlinks
# the marker, re-baselines and persists without naming the notifier: clearing the
# incident and closing the alarm were two operations only `resolve` performed.
#
# Every node here drives a REAL trip over a temporary tree — a top-level folder of
# 25 files, over `vaultwatch.TOPDIR_MIN_FILES`, emptied — then the real `clear()`,
# and requires the alarm to close: on the tick, and in a second process.

def _gstate(tmp_path):
    """Where a Guardian from `_tripwire_guardian` keeps its state — and so its
    ALERT.md: `args.guardian_state`, not the vault box `_notifier` builds."""
    return tmp_path / "gstate"

FAMILIES = ("vault", "data")


def _tripwire_guardian(tmp_path, monkeypatch, family):
    """A Guardian whose watched tree is this test's, alarm surfaces all real.

    Only the family under test has anything to lose: the other root is an empty
    directory for `vaultwatch` (no top-level folder of its own to empty) and one
    without `.lloyd-data-root` for `datawatch`, which refuses to arm a root it has
    never seen. That keeps ALERT.md's single slot un-contested — exactly one
    tripwire can fire per node — instead of the harness deciding which of two
    critical alerts won the file.

    Calling it twice on one `tmp_path` is how a restart is tested here: the tree is
    created only when it is not there already, so the second Guardian sees the tree
    the first one left, with the same `gdir`, nothing of its own remembered, and the
    baseline reloaded from the state file the first process persisted.

    `_stop_sync`, `_pause_workers` and `_vault_evidence` are stubbed because they
    stop supervisord programs, POST at the live backend and shell out to `ps`.
    `check_data`'s other two reads are stubbed because they judge the code tree and
    the snapshot directory, neither of which these nodes are about — and an
    un-stubbed "Data snapshots are not arriving" would take ALERT.md's slot away
    from the title under test. The watch, its latch, the marker, `clear()`, the
    notifier, ALERT.md and the daily note are the shipped code, and `Guardian.alert`
    runs with its real repeat window set to zero so a second call for one incident
    needs no sleep.
    """
    import types

    import guardian as G
    import policy as GP
    import datawatch as DW

    watched = tmp_path / f"watched-{family}"
    if not watched.exists():
        (watched / "notes").mkdir(parents=True)
        for i in range(25):                 # over vaultwatch.TOPDIR_MIN_FILES
            (watched / "notes" / f"n{i}.md").write_text("x\n", encoding="utf-8")
        if family == "data":
            (watched / DW.ROOT_MARKER).write_text("", encoding="utf-8")
    inert = tmp_path / f"inert-{family}"
    inert.mkdir(exist_ok=True)
    vault_root, data_root = ((str(inert), str(watched)) if family == "data"
                             else (str(watched), str(inert)))

    monkeypatch.setattr(GP, "VAULT_ROOT", vault_root)
    monkeypatch.setattr(GP, "DATA_ROOT", data_root)
    monkeypatch.setattr(GP, "REPO", str(tmp_path / "repo"))
    monkeypatch.setattr(GP, "ALERT_REPEAT_SECONDS", 0.0)
    (tmp_path / "memory").mkdir(exist_ok=True)

    args = types.SimpleNamespace(
        repo=str(tmp_path), state=str(tmp_path / "state"),
        guardian_state=str(tmp_path / "gstate"), supervisor_sock="/nonexistent",
        backend_url="http://127.0.0.1:1/health", mcp_url="http://127.0.0.1:2/health",
        programs=PROGRAM, interval=5.0,
    )
    g = G.Guardian(args)
    monkeypatch.setattr(g.notifier, "vault_root", tmp_path)
    monkeypatch.setattr(g, "_stop_sync", lambda: "stopped")
    monkeypatch.setattr(g, "_pause_workers", lambda actor: "paused")
    monkeypatch.setattr(g, "_vault_evidence", lambda stamp: None)
    monkeypatch.setattr(g, "heartbeat", lambda *a, **k: None)
    monkeypatch.setattr(g, "ensure_chronic", lambda: None)
    monkeypatch.setattr(g, "_runtime_data_incident", lambda *a, **k: [])
    monkeypatch.setattr(DW, "snapshot_report", lambda *a, **k: (True, ""))
    watch = g.vault if family == "vault" else g.data
    check = g.check_vault if family == "vault" else g.check_data
    title = (G.VAULT_TRIP_ALERT_TITLE if family == "vault"
             else G.DATA_TRIP_ALERT_TITLE)
    return g, watch, check, title


def _trip_and_note(tmp_path, monkeypatch, family):
    """Trip one wire for real: arm, delete a folder, alert. Both artefacts back."""
    import shutil

    g, watch, check, title = _tripwire_guardian(tmp_path, monkeypatch, family)
    check()                                 # arms the watch; nothing has been lost yet
    assert g.notifier.alert_file_incident() is None, "arming the watch raised an alarm"
    shutil.rmtree(f"{watch.root}/notes")
    check()                                 # the mass deletion
    assert g.notifier.alert_file_incident() == (title, False), (
        f"a wipe of {watch.root} did not put {title!r} in ALERT.md as a live directive")
    return g, watch, check, title, _note(tmp_path)


def _guardian_sections(text: str) -> dict[str, str]:
    """Each `## Self-mod guardian: <title>` heading mapped to its body."""
    out: dict[str, str] = {}
    for chunk in text.split(f"\n{notify.DAILY_SECTION_PREFIX}")[1:]:
        head, _, body = chunk.partition("\n")
        out[head.strip()] = body
    return out


def _open_titles(text: str) -> list[str]:
    """Which incidents the note still reports open, by its own markers.

    Read from the markers rather than from a count of headings, because the marker is
    what `notify._daily_open_at` asks and what `resolve` erases: a node that counted
    headings would pass on a section `resolve` had never been able to see.
    """
    return [t for t, body in _guardian_sections(text).items()
            if body.rstrip().endswith(notify.DAILY_STILL_OPEN)]


@pytest.mark.parametrize("family", FAMILIES)
def test_a_tripwire_alert_is_written_where_a_retraction_can_reach_it(
        tmp_path, monkeypatch, family):
    """Clause 3, first half: the section a trip writes must be sealable at all.

    This is the block the item's own triage found standing in
    `~/obsidian/memory/2026-10-20.md` long after the trip was adjudicated fictitious
    261 seconds after it fired. An uncoalesced section carries no
    `DAILY_STILL_OPEN`; `_daily_open_at` calls a section open only while its body
    ends with that marker; `resolve` seals only what it finds open. So the
    restore-and-swap paragraph was not merely un-retracted — it was invisible to the
    call that would have retracted it, which returned True over it (#1590).
    """
    g, watch, check, title, note = _trip_and_note(tmp_path, monkeypatch, family)
    sections = _guardian_sections(note)
    assert list(sections) == [title], f"the note's sections are {list(sections)}"
    assert _open_titles(note) == [title], (
        "a trip's section must end with the still-open marker, or `resolve` has "
        f"nothing to seal:\n{note}")
    assert "swap it in" in sections[title], (
        "the body under test has to be the restore directive itself")
    assert _lines(_gstate(tmp_path))[0] == f"# {title}", _lines(tmp_path)


@pytest.mark.parametrize("family", FAMILIES)
def test_a_tripwire_alarm_closes_on_the_tick_its_tree_measures_whole(
        tmp_path, monkeypatch, family):
    """Clause 2: no restart is wanted and none happens.

    The order is the real one: trip, a human restores the tree and runs the watch's
    own `clear()`, the guardian's next check of that root measures it whole. The seal
    has to be on ALERT.md by the end of that tick, because the stranded alarm this
    item was filed from had already outlived three guardian restarts, each of which
    called `reconcile_stranded_alert` and logged nothing.
    """
    g, watch, check, title, note = _trip_and_note(tmp_path, monkeypatch, family)
    assert watch.clear() is True, "`clear()` reports no marker to remove"
    assert watch.tripped() is None

    check()                                                 # same process, next check

    assert g.notifier.alert_file_incident() == (title, True), (
        "ALERT.md is still a live directive after the root measured whole")
    assert len(_cleared_lines(_gstate(tmp_path))) == 1, _cleared_lines(_gstate(tmp_path))
    sealed = (_gstate(tmp_path) / "ALERT.md").read_text(encoding="utf-8")
    assert notify.ALERT_CLEARED_BANNER in sealed, (
        "the retracted file must still carry the alarm's text, quoted")
    assert "swap it in" in sealed, "quoted, not deleted — that is the whole point"
    sealed_note = _note(tmp_path)
    assert _open_titles(sealed_note) == [], "the daily-note section is still open"
    dated = [ln for ln in sealed_note.splitlines()
             if ln.startswith(f"{notify.DAILY_CLEARED_PREFIX} ")]
    assert len(dated) == 1, dated

    # A third and fourth check write nothing: the alarm is closed, and re-stamping a
    # closed incident would move the one time a later reader ages.
    alert_bytes, sealed_bytes = (_gstate(tmp_path) / "ALERT.md").read_bytes(), sealed_note.encode()
    check()
    check()
    assert (_gstate(tmp_path) / "ALERT.md").read_bytes() == alert_bytes, (
        "a closed tripwire alarm was re-sealed")
    assert _note(tmp_path).encode() == sealed_bytes, (
        "a second retraction of a closed incident rewrote the daily note")


@pytest.mark.parametrize("family", FAMILIES)
def test_a_restarted_guardian_retracts_a_tripwire_alarm_it_can_measure(
        tmp_path, monkeypatch, family):
    """Clause 1: `reconcile_stranded_alert` owns both tripwire titles.

    One process trips the wire and is dropped; the tree is restored and cleared; a
    SECOND Guardian over the same state directory and the same root decides from the
    file's own H1 plus its own two reads. This is the half that already existed for
    the three latch-backed titles (#3019); for these two it was a silent
    `nothing-to-reconcile`, because nothing in the tree had ever handed either title
    to `resolve` and the map cannot tell "no guard owns this" from "the guard that
    owns it never named it".
    """
    g, watch, check, title, note = _trip_and_note(tmp_path, monkeypatch, family)
    watch.clear()

    g2, watch2, check2, title2 = _tripwire_guardian(tmp_path, monkeypatch, family)
    assert title2 == title and watch2.tripped() is None
    assert g2.notifier.alert_file_incident() == (title, False)

    assert g2.reconcile_stranded_alert() == "sealed", (
        "the marker is gone and the watch's own baseline reads the root whole, so the "
        "restore directive in ALERT.md is a live order for an incident that is over")
    assert g2.notifier.alert_file_incident() == (title, True)
    assert len(_cleared_lines(_gstate(tmp_path))) == 1, _cleared_lines(_gstate(tmp_path))
    assert g2.reconcile_stranded_alert() == "nothing-to-reconcile", (
        "a sealed incident must not be reconciled again on a later start")


@pytest.mark.parametrize("family", FAMILIES)
def test_a_tripwire_alarm_stays_open_while_its_marker_is_still_set(
        tmp_path, monkeypatch, capsys, family):
    """Clause 1's other half, and clause 4's "no seal from an unmet condition".

    The marker IS the incident: `clear()` is what the human runs once the tree is
    back, so a guardian that sealed ALERT.md while it stood would be announcing an
    all-clear over a wipe that is still latched and still refusing sync and
    snapshots. Byte-for-byte identical, because the sealed file carries the stamp a
    later reader ages and a second write would move it.
    """
    g, watch, check, title, note = _trip_and_note(tmp_path, monkeypatch, family)
    before = (_gstate(tmp_path) / "ALERT.md").read_bytes()

    g2, watch2, check2, _ = _tripwire_guardian(tmp_path, monkeypatch, family)
    assert watch2.tripped() is not None, "the second process cannot see the latch"
    assert g2.reconcile_stranded_alert() == "still-open"
    assert (_gstate(tmp_path) / "ALERT.md").read_bytes() == before, (
        "a seal was written over a live tripwire")
    assert "does not hold" in capsys.readouterr().out

    # The same verdict on the check tick with the marker set and the tree otherwise
    # whole — a human who restored the files but has not cleared the latch yet, which
    # can last hours while the tick runs every 5 s. So the verdict is journalled on
    # the transition into it, not per tick.
    check2()
    assert (_gstate(tmp_path) / "ALERT.md").read_bytes() == before
    capsys.readouterr()
    check2()
    check2()
    assert "does not hold" not in capsys.readouterr().out, (
        "the same not-clear verdict was journalled on every tick of one incident")


# ── the rename hazard: an all-clear must reach a section opened under a heading that no
#    longer exists (#3124) ────────────────────────────────────────────────────────────────
#
# #3124 split the runtime-data alert's heading in two: a check that finds a file inside the
# code tree keeps "Runtime data is being written into the code tree", and a check that finds
# only 0-byte stand-ins under the data root gets a heading that says that instead. `resolve`
# matches the daily-note heading LITERALLY, so the split orphans whatever a pre-rename
# guardian left open: the all-clear carries the new words, walks past the old heading, and the
# section keeps its `_(still open on the next check)_` marker — the standing claim a
# retraction exists to remove, which is #1536's failure mode rather than a cosmetic one.
# `RUNTIME_DATA_ALERT_TITLES` is the guardian's answer (agent-services/guardian/guardian.py);
# these two nodes are that walk over a note it was not written for, and over a note where the
# walk must change nothing.
#
# Both titles are spelled out as literals here, not imported. They are the contract's own
# words — clause 1 and clause 2 name them — and the point of the first node is that the bytes
# a retired guardian wrote are the bytes the new one has to reach.

TREE_TITLE = "Runtime data is being written into the code tree"
SHADOW_TITLE = "0-byte databases are shadowing Lloyd's stores under the data root"

#: Shaped like the live 2026-10-21 incident that made the old heading wrong: every path it
#: names is under `/home/alansrobotlab/lloyd-data`, and it still orders the reader to go find
#: a writer in the tree.
SHADOWISH_BODY = ("These exist under the data root:\n"
                  "  /home/alansrobotlab/lloyd-data/kg.sqlite (0 bytes)\n"
                  "\n"
                  "Find the writer, move the data across, and remove the in-tree copy.")


def _vault_note(n: notify.Notifier) -> Path:
    mem = sorted((Path(n.vault_root) / "memory").glob("*.md"))
    assert len(mem) == 1, f"expected one daily note, got {[str(m) for m in mem]}"
    return mem[0]


def _open_incident(tmp_path: Path, title: str, body: str) -> notify.Notifier:
    """Raise `title` twice, so the marker the walk has to remove is notify's own.

    The second alert is the hourly re-check that found the condition again: the coalescing
    path is what appends `DAILY_STILL_OPEN` (notify.py:524-545). Handing the note a
    hand-written marker would test the fixture rather than the machinery.
    """
    n = _notifier(tmp_path)
    n.alert("error", title, body, coalesce=True)
    n.alert("error", title, body, coalesce=True)
    return n


def test_an_all_clear_walks_the_heading_a_rename_left_behind(tmp_path):
    """#3124 clause 3, the half a retitle cannot get for free: the retired heading seals.

    Two readings, and the first one is what stops this node passing on a resolve that simply
    never reached anything. (a) An all-clear carrying ONLY the new shadow heading leaves a
    section opened under the retired heading standing open, marker and all — that orphan is
    the hazard, so a one-title `resolve` is a failure here rather than a silent no-op.
    (b) The walk the guardian now does on a check that measures neither condition — one
    `resolve` per title, in both directions — seals it: the section gains a `cleared:` line,
    the marker goes, and one incident is still one section.

    The marker's removal is the assertion that matters, not the `cleared:` line's arrival: a
    reader scanning the note for open incidents scans for the marker, and a section carrying
    both is the note still claiming the incident is open.
    """
    # (a) the hazard, measured on the same fixture the walk is run on.
    solo = _open_incident(tmp_path / "solo", TREE_TITLE, SHADOWISH_BODY)
    assert notify.DAILY_STILL_OPEN in _vault_note(solo).read_text(encoding="utf-8"), (
        "the fixture is not an open incident, so nothing below would prove a seal")
    # The return value is the trap in its purest form: this call reports SUCCESS — the True is
    # a claim about the surfaces it could reach, not about the incident, and `resolve`'s own
    # docstring counts "held another title's alarm" as a satisfied surface. So an all-clear
    # carrying only the new heading looks closed from the caller's side, the ledger, and the
    # log, while the retired section keeps its marker for as long as the note is read.
    assert solo.resolve(SHADOW_TITLE, NOTE) is True
    left = _vault_note(solo).read_text(encoding="utf-8")
    assert "cleared:" not in left, (
        "a shadow-headed clear reached a section opened under the tree heading, which is not "
        "how a literal heading match works — the fixture, not the fix, is wrong")
    assert notify.DAILY_STILL_OPEN in left, left

    # (b) the walk, over the same note.
    n = _open_incident(tmp_path, TREE_TITLE, SHADOWISH_BODY)
    for title in (SHADOW_TITLE, TREE_TITLE):
        assert n.resolve(title, NOTE) is True, title
    sealed = _vault_note(n).read_text(encoding="utf-8")
    assert NOTE in sealed, sealed
    assert notify.DAILY_STILL_OPEN not in sealed, (
        "the section still tells a reader scanning for the marker that the incident is open")
    assert sealed.count(f"{notify.DAILY_SECTION_PREFIX}{TREE_TITLE}") == 1, (
        "the walk split one incident into two sections")


def test_the_walk_writes_nothing_to_a_note_that_never_carried_those_headings(tmp_path):
    """#3124 clause 3's other half: the extra resolves are idempotent, not noise.

    The all-clear runs on every stray-check interval, so from now on every quiet hour the
    guardian calls `resolve` once per title against a note holding nothing for any of them.
    Each must leave the file byte-identical and still report True: `Guardian.alert` treats a
    falsy resolve as a broken notifier and says so on every incident cycle, so a walk that
    returned False for a heading with no open section would make every clean hour look
    broken — and a walk that rewrote the note would put a `cleared:` line under an unrelated
    incident.
    """
    n = _notifier(tmp_path)
    n.alert("error", OTHER, BODY, coalesce=True)
    note = _vault_note(n)
    before = note.read_bytes()

    for title in (SHADOW_TITLE, TREE_TITLE):
        assert n.resolve(title, NOTE) is True, title

    assert note.read_bytes() == before, (
        "a resolve for a heading the note never carried wrote the note")
    assert "cleared:" not in before.decode(encoding="utf-8")


def test_a_check_that_measures_neither_condition_still_walks_every_heading(tmp_path,
                                                                           monkeypatch):
    """#3124 clause 3's first half, driven by the shipped `Guardian` over the shipped notifier.

    `_tripwire_guardian` stubs `_runtime_data_incident` out because the tripwire nodes are
    about a wipe, so this node puts the real method back and points both legs of the check at
    an empty world: no file inside `policy.REPO`, nothing under the data root. That is the
    hourly state on a healthy box, and it is where the walk's cost is paid, so this is the
    node that answers the question the rename raises — what does the guardian now hand the
    notifier on a check that found nothing?

    One `resolve` per heading it could have opened, in the guardian's own order, each
    reported True. The artefacts are asserted too: a walk is only free while it writes
    nothing, and writing nothing on a box that never raised — no daily note, no ALERT.md —
    is what makes an extra call per heading safe to repeat every check interval for as long
    as the box is healthy.
    """
    import time as _time

    import datawatch as DW
    import guardian as G

    g, _watch, _check, _title = _tripwire_guardian(tmp_path, monkeypatch, "data")
    monkeypatch.setattr(g, "_runtime_data_incident",
                        lambda *a, **k: G.Guardian._runtime_data_incident(g, *a, **k))
    monkeypatch.setattr(DW, "DATA_ROOT", str(tmp_path / "empty-data"))
    (tmp_path / "empty-data").mkdir()
    monkeypatch.setattr(DW, "stray_in_tree", lambda _tree: [])

    seen: list[tuple[str, bool]] = []
    real_resolve = g.notifier.resolve
    # `measured_root` is in the shim's signature because #3792 clause 1 put it in the
    # shipped call: the clearance must carry the root it measured for the daily-note
    # writer to be able to say so. The node's own claim — one `resolve` per heading, each
    # True, nothing written — is asserted over `seen` and is unchanged by the widening.
    monkeypatch.setattr(
        g.notifier, "resolve",
        lambda title, text, measured_root="": (
            (lambda ok: (seen.append((title, ok)), ok)[1])(
                real_resolve(title, text, measured_root=measured_root))))

    # Nothing is asserted about the return: on this path it is `retracted`, whether any
    # `resolve` succeeded, and the walk below is what this node is about.
    g._runtime_data_incident(_time.time())
    assert seen == [(t, True) for t in G.RUNTIME_DATA_ALERT_TITLES], seen
    # The strongest form of "writes nothing": this box never raised, so there was no section
    # to seal, and the walk must not conjure a note to put a `cleared:` line in.
    assert not list((tmp_path / "memory").glob("*.md")), (
        "the walk wrote a daily note on a check that measured nothing")
    assert g.notifier.alert_file_incident() is None, (
        "the walk opened an ALERT.md incident for a condition it did not measure")


# ── #3150: a retraction reaches every open copy, and a seal is permanent ─────
#
# `resolve`'s loop asks `_daily_open_at` for the next open section until it answers
# None. Pre-#3150 that probe returned at the NEWEST section matching the title,
# marker or no marker, so a note holding several copies of one incident — which is
# what the broken openness test produced whenever another writer appended — sealed
# only the newest copy and left the older ones standing. `memory/2026-10-21.md` is
# the artefact: its `cleared:` line sits under one copy while three
# `_(still open on the next check)_` markers survive above it, hours after the
# incident closed. The #1536 docstring's own reason for this method — "no surviving
# record of a finished incident is an instruction to do work" — was failing silently.

#: What the other writer appended between two copies on the real note. Kept out of
#: the sealed text on purpose: `resolve` rewrites `sealed[start:end]`, so this is the
#: line a seal could eat while still producing the right marker count.
FOLDED_ALARM_LINE = ("- 04:07 PDT [tmpwatch] /tmp is running out of inodes: 81% "
                     "[daily-note fold: this alarm writes at most one line a day "
                     "here]")


def _sections_of(text: str, title: str) -> list[str]:
    """Each `## Self-mod guardian: {title}` window in `text`, by the shipped parser.

    `Notifier._daily_sections`, not a re-typed split, so the windows asserted below
    are the ones the notifier itself reads.
    """
    return [text[a:b] for (t, a, b) in notify.Notifier._daily_sections(text) if t == title]


def _copies_of_one_open_incident(tmp_path: Path, copies: int = 3) -> tuple:
    """One open section, repeated `copies` times with the fold's line between them.

    The byte shape `memory/2026-10-21.md` holds. It is NOT written by calling
    `_vault_note` `copies` times: the fixed writer refreshes in place, which is the
    whole point of #3150, so the duplicates have to come from the note. They are
    therefore the shipped writer's OWN bytes — one real `_vault_note` call, then that
    block repeated with a foreign line between each pair — not a hand-typed format
    this file could get wrong and move on from.
    """
    n = _notifier(tmp_path)
    assert n._vault_note(TITLE, BODY, coalesce=True) is True
    note = _vault_note(n)
    written = note.read_text(encoding="utf-8")
    head, _, section = written.partition(notify.DAILY_SECTION_PREFIX)
    block = notify.DAILY_SECTION_PREFIX + section      # ends at the marker's newline
    gap = f"\n{FOLDED_ALARM_LINE}\n\n"
    note.write_text(head + gap.join([block] * copies), encoding="utf-8")
    return n, note


def test_a_retraction_seals_every_open_copy_of_one_title(tmp_path):
    """#3150 clause 3: one all-clear seals the whole title, not just its newest copy.

    Three open sections for one title and one `resolve`: after it, no open marker
    survives anywhere in the file, and each of the three sections carries its own
    `cleared:` line inside its own window — the reader who lands on the oldest copy
    still finds the contradiction at the foot of the section they are reading, which
    is the contract #1536 opened with.

    The fold's line is asserted as well: it sits inside the first two windows, and a
    seal whose `end` ran to the window's end would delete it while still reporting
    the file as closed.
    """
    n, note = _copies_of_one_open_incident(tmp_path)
    before = note.read_text(encoding="utf-8")
    assert before.count(notify.DAILY_STILL_OPEN) == 3, (
        f"fixture: three standing markers are the state being sealed, got:\n{before}")
    assert len(_sections_of(before, TITLE)) == 3, before

    assert n.resolve(TITLE, NOTE) is True

    text = note.read_text(encoding="utf-8")
    assert notify.DAILY_STILL_OPEN not in text, (
        "clause 3: a `cleared:` line stands in this note, so every standing marker "
        f"below it is an instruction nobody should follow:\n{text}")
    secs = _sections_of(text, TITLE)
    assert len(secs) == 3, f"the seal collapsed or duplicated sections:\n{text}"
    for i, sec in enumerate(secs):
        assert f"\n{notify.DAILY_CLEARED_PREFIX} " in sec, (
            f"section {i} was sealed without a retraction inside its own window:\n{sec}")
    assert text.count(FOLDED_ALARM_LINE) == 2, (
        f"the seal rewrote the other writer's line:\n{text}")
    assert secs[0].count(FOLDED_ALARM_LINE) == 1 and secs[2].count(FOLDED_ALARM_LINE) == 0


def test_a_sealed_section_is_never_refreshed_by_the_next_alert(tmp_path):
    """#3150 clause 4: a `cleared:` section is history; a new finding opens a new one.

    The other half of the seal. Coalescing may not turn a title into a mute: after a
    clear, the next alert for the SAME title has to be a new heading, or a second
    incident hides inside the first one's sealed section — worse than the repetition
    #1536 removed. The probe is pinned on both sides of that: the sealed note answers
    None (a `cleared:` line is not an open marker), and the new alert lands as
    section two while section one keeps its body, its retraction and its count.
    """
    n = _notifier(tmp_path)
    note = n._daily_note()
    assert n._vault_note(TITLE, BODY, coalesce=True) is True
    assert n._daily_open_at(note.read_text(encoding="utf-8"), TITLE) is not None, (
        "precondition: nothing was open, so a seal that wrote nothing would leave "
        "the byte-identity below true of a note that was never written to")

    assert n.resolve(TITLE, NOTE) is True
    sealed = note.read_text(encoding="utf-8")
    assert notify.DAILY_STILL_OPEN not in sealed, sealed
    assert n._daily_open_at(sealed, TITLE) is None, (
        "clause 4: the probe still calls a sealed section open, so the next alert "
        f"would rewrite history instead of reporting a new incident:\n{sealed}")

    assert n._vault_note(TITLE, "a second, separate finding", coalesce=True) is True
    text = note.read_text(encoding="utf-8")
    secs = _sections_of(text, TITLE)
    assert len(secs) == 2, (
        f"clause 4: a fresh incident after a clear wrote {len(secs)} sections "
        f"instead of one:\n{text}")
    assert f"\n{notify.DAILY_CLEARED_PREFIX} " in secs[0], secs[0]
    assert BODY in secs[0] and "a second, separate finding" not in secs[0], secs[0]
    assert "a second, separate finding" in secs[1], secs[1]
    assert secs[1].count(notify.DAILY_STILL_OPEN) == 1, secs[1]
    assert text.count(f"\n{notify.DAILY_CLEARED_PREFIX} ") == 1, (
        f"the new alert retracted the incident it was reporting:\n{text}")


# ── #3757: the board half of a retraction, and the two ways it must not fire ────

LIST_ROUTE = "/api/backlog/tasks"
UPDATE_ROUTE = "/api/backlog/task-update"
BOARD_ROW = {"id": 3755, "name": "[guardian] Service down, but no promotion to revert",
             "status": "draft"}
SERVICE_DOWN = "Service down, but no promotion to revert"
#: The alarm and the retraction note the liveness branch actually produces:
#: `guardian.py:93` is this title's only definition, `guardian.py:2773` builds the note,
#: and the 24-timeout FATAL line is what the 2026-11-04 alert carried into ALERT.md.
SERVICE_DOWN_BODY = "lloyd-mc:lloyd-backend: FATAL: can't find command (repeated 24 times)"
SERVICE_DOWN_NOTE = "liveness read came back healthy: lloyd-mc:lloyd-backend."


class _WatchingBoard:
    """A loopback board that counts every request, so "no board read" is checkable.

    Clause 4 and clause 5 are both about requests that must NOT happen, and a dead
    backend (`_notifier`'s default `127.0.0.1:1`) answers a refused connection for
    both the code under test and a call that was never made — which pins nothing. So
    these two nodes are the only ones in this file that stand up a server: the count
    it takes is the measurement, and it answers exactly as `app/routers/backlog.py`
    does (a bare array on the list at `:686`,
    `{"success": true, "description_ignored": …}` on the update at `:811`) so the code
    under test is not steered by a shape the real board would not send.
    """

    def __init__(self, rows=(), list_status=200):
        self.rows = list(rows)
        self.list_status = list_status
        self.gets: list[str] = []
        self.posts: list[tuple[str, dict]] = []
        board = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, obj):
                payload = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                board.gets.append(self.path)
                if board.list_status != 200:
                    self._send(board.list_status, {"detail": "board unavailable"})
                    return
                # A BARE array, because that is the route's shape
                # (`app/routers/backlog.py:686`, `JSONResponse(tasks)`). A wrapper
                # object is precisely what `_board_rows` refuses as
                # `"body is dict, not a list"`, so a stand-in that answered the wrong
                # shape would pin a read production never performs.
                self._send(200, board.rows)

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                board.posts.append((self.path, json.loads(self.rfile.read(length) or b"{}")))
                self._send(200, {"success": True, "description_ignored": False})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"

    @property
    def closes(self) -> list[dict]:
        return [body for path, body in self.posts if path == UPDATE_ROUTE]

    def close(self):
        self._server.shutdown()
        self._server.server_close()


def test_an_all_clear_with_nothing_to_seal_never_asks_the_board(tmp_path):
    """#3757 clause 4: the ordinary healthy all-clear costs no board read.

    `resolve` runs on every clean tick — the stray check is `STRAY_CHECK_SECONDS =
    3600.0` (`agent-services/guardian/policy.py:426`, gated at `guardian.py:2381`, and a
    clear leg like `guardian.py:2197` is what calls it) — as well as on the liveness
    recovery branch, and the row lookup it now performs for a sealed incident is the
    same `?q=` scan #3551 measured at 18.6 s cold over 3,693 rows — under the 5 s
    ceiling then in force, and the ceiling is `BACKLOG_READ_TIMEOUT_SECONDS = 30.0`
    beside it now, so a read that slow now completes rather than timing out. Scheduled
    onto every clear tick, that read is the dedup guard's own worst case turned into a
    timer, so the read is owed only by a call that actually sealed something.

    The two shapes of "nothing to seal" are both here because they are different code
    paths: an all-clear for a title that never alerted, and the SECOND all-clear of an
    incident the first one already sealed — which is the hourly case, since the
    condition stays clear and `resolve` keeps running against sealed bytes.
    """
    never_alerted = _WatchingBoard(rows=[BOARD_ROW])
    n = _notifier(tmp_path, backend_url=never_alerted.url)

    assert n.resolve(SERVICE_DOWN, SERVICE_DOWN_NOTE) is True
    assert never_alerted.gets == [], (
        "a clear for a title that never alerted read the board, paying the 30 s scan "
        "for a row it was not going to close")
    assert never_alerted.posts == []

    reopened = tmp_path / "second"
    reopened.mkdir()
    board = _WatchingBoard(rows=[dict(BOARD_ROW)])
    m = _notifier(reopened, backend_url=board.url)
    m.alert("error", SERVICE_DOWN, SERVICE_DOWN_BODY, coalesce=True)
    assert m.resolve(SERVICE_DOWN, SERVICE_DOWN_NOTE) is True
    first_closes = list(board.closes)
    assert len(first_closes) == 1, (
        "fixture: the seal that DID happen is expected to close the open row")

    assert m.resolve(SERVICE_DOWN, SERVICE_DOWN_NOTE) is True
    assert board.closes == first_closes, (
        "the second all-clear of an already-sealed incident re-read the board and "
        "posted again: "
        f"{board.posts[len(first_closes):]}")
    lookups = [p for p in board.gets if p.startswith(LIST_ROUTE)]
    assert len(lookups) == 1, (
        f"the row lookup ran {len(lookups)} times for one seal: {board.gets}")


def test_a_drill_seals_alert_md_and_never_posts_a_close(tmp_path):
    """#3757 clause 5: the close sits BELOW `resolve`'s `external` gate.

    `resolve` returns at its `if not self.external` when `external` is false, the same
    gate `alert` applies before filing — so a drill that cannot
    file must not be able to close either: a drill's all-clear aimed at the live board
    would move a row the drill never created. #1967's node above
    (`test_the_retraction_runs_below_the_external_gate`) pins the ALERT.md half of that
    asymmetry, which is why the seal is asserted here too and not assumed: the
    retraction reaching the board must not cost the drill its ability to close its own
    alarm file.
    """
    board = _WatchingBoard(rows=[BOARD_ROW])
    n = _notifier(tmp_path, external=False, backend_url=board.url)
    _raise(n, SERVICE_DOWN, SERVICE_DOWN_BODY)

    assert n.resolve(SERVICE_DOWN, SERVICE_DOWN_NOTE) is True

    assert len(_cleared_lines(tmp_path)) == 1, (
        "the board-silence below the gate cost the drill its ALERT.md seal")
    assert board.posts == [], (
        f"a drill posted to the live board: {board.posts}")
    assert board.gets == [], (
        "the board read sits above the external gate, so a drill pays the row "
        "lookup it cannot act on")


# ── #3792: the SEAL side — a foreign `cleared:` line says so in the line itself ──
#
# #3628 gave the ALERT SIDE a label (`cross-root seal:`) and said outright, at
# `notify.py:79-80`, that "whether a cross-root process may seal at all is a ruling
# #3628 does not take". The seal side stayed unlabelled: `Notifier.resolve` compared no
# roots, so the line a human reads FIRST — the `cleared:` line that closed the section —
# carried no marker whenever the finding had genuinely gone and no later check existed
# to re-alert. `memory/2026-10-29.md` carries five of those, each certifying a
# `lloyd-work/SM_*/home/lloyd-data` tree that no longer exists, over a rule whose purpose
# is protecting `/home/alansrobotlab/lloyd-data`.
#
#: The live runtime-data root, i.e. the root the reader of the daily note means.
DATA_LIVE = "/home/alansrobotlab/lloyd-data"
#: The code tree the live guardian walks, and the path every shipped clearance prints
#: beside its data root (`guardian.py`: "inside the code tree of {policy.REPO}").
TREE_LIVE = "/home/alansrobotlab/lloyd"
#: One automod round's data root: `worktree.py` symlinks `~/obsidian` into the round
#: home, so a round's guardian writes THIS live vault while `LLOYD_DATA` names this tree.
ROUND_DATA = "/home/alansrobotlab/lloyd-work/SM_FAKE0000_000000/home/lloyd-data"
#: The same round's code tree. Printed by a round's clearance exactly as the live one
#: prints `TREE_LIVE`, and the path that must NOT reach the marker (#3628's own review
#: advisory: the label listed `policy.REPO`, "a path the same check walks and prints in
#: every clearance").
ROUND_TREE = "/home/alansrobotlab/lloyd-work/SM_FAKE0000_000000/home/lloyd"
#: The shipped shadow alert's heading, spelled as `RUNTIME_DATA_SHADOW_ALERT_TITLE`
#: spells it at `guardian.py:193`.
DATA_SHADOW_TITLE = (
    "Content-empty databases under the data root are shadowing Lloyd's populated stores")
#: Shaped like `_shadow_alert_body`'s output: the root it measured in its first
#: sentence, then the named pair under that root. Every absolute path it prints is
#: either `DATA_LIVE` or under it, which is what makes an all-clear naming `DATA_LIVE`
#: an honest one and an all-clear naming `ROUND_DATA` a foreign one.
DATA_SHADOW_BODY = (
    "A content-empty database inside /home/alansrobotlab/lloyd-data has the same name "
    "as a populated database elsewhere under that same root (1 such file, checked "
    "2026-11-05T09:00:00Z). This check moves and deletes NOTHING.\n\n"
    "- empty `/home/alansrobotlab/lloyd-data/kg.sqlite` (81,920 bytes, 0 rows in 4 "
    "tables) — trust `/home/alansrobotlab/lloyd-data/kg.sqlite` (245,760 bytes)")
#: The clearance the shipped check writes when both scans came back clean
#: (`guardian.py:2364-2371`): the tree it walked, the data root it walked, and the
#: "absent with no move recorded" wording.
def _clearance(tree: str, data_root: str) -> str:
    return (f"no runtime stores inside the code tree of {tree}, and no database with "
            f"zero data rows shadowing a populated store under {data_root} — absent "
            "with no move recorded by the guardian, so the instructions above are "
            "stale and nothing here accounts for a path that was named and is now gone")


def _cleared_line(text: str, title: str) -> str:
    """The one `cleared:` line inside `title`'s section, by the shipped parser.

    Asserted singular on purpose: a node that took `lines[0]` would pass on a note whose
    seal duplicated the section, which is the failure #3150 exists to prevent.
    """
    hits = [ln for sec in _sections_of(text, title)
            for ln in sec.splitlines()
            if ln.startswith(f"{notify.DAILY_CLEARED_PREFIX} ")]
    assert len(hits) == 1, f"expected exactly one cleared line:\n{text}"
    return hits[0]


def _marker_segment(line: str) -> str:
    """The marker segment of one `cleared:` line, from its prefix to the line's end.

    Scoped because the clearance's own prose — shipped bytes, unchanged by this round —
    names its code tree ahead of the marker: asking whether a path is in the LINE could
    never fail, and clause 3 is a claim about what the MARKER adds.
    """
    cut = line.find(notify.DAILY_CROSS_ROOT_CLEAR_PREFIX)
    assert cut >= 0, f"no #3792 marker in that line:\n{line}"
    return line[cut:]


def _marker_paths(line: str) -> set[str]:
    """Every absolute path inside the marker segment of one `cleared:` line."""
    return set(notify._absolute_roots(_marker_segment(line)))


def test_a_foreign_data_root_seal_marks_its_own_cleared_line_and_still_closes(
        tmp_path):
    """#3792 clause 1: the seal that measured another root says so IN ITS OWN LINE.

    The open section's body measured `DATA_LIVE`; the clearance that closes it measured
    one automod round's `home/lloyd-data`. The `cleared:` line therefore carries the
    cross-root marker naming BOTH roots — the one this seal measured and the one the
    section it closed measured — while the section still closes: the open marker is gone,
    `resolve` still answers True, and the next finding of the same incident opens a fresh
    section, which is #3628's re-alert behaviour and must stay exactly as shipped.

    This is the byte that `memory/2026-10-29.md` lacked: five `cleared:` lines each
    certifying a deleted `lloyd-work/SM_*` tree, none of them saying so.
    """
    n = _notifier(tmp_path)
    assert n.alert("error", DATA_SHADOW_TITLE, DATA_SHADOW_BODY, coalesce=True)
    note = _vault_note(n)
    before = note.read_text(encoding="utf-8")
    assert notify.DAILY_STILL_OPEN in before, (
        f"precondition: nothing open, so a seal that wrote nothing would leave every "
        f"assertion below true of a note that was never sealed:\n{before}")

    assert n.resolve(DATA_SHADOW_TITLE, _clearance(ROUND_TREE, ROUND_DATA),
                     measured_root=ROUND_DATA) is True, (
        "clause 1: a marked seal must still CLOSE the section — `resolve` returning "
        "False would keep the incident live and re-file it")

    text = note.read_text(encoding="utf-8")
    assert notify.DAILY_STILL_OPEN not in text, (
        f"clause 1: the section did not close:\n{text}")
    line = _cleared_line(text, DATA_SHADOW_TITLE)
    assert notify.DAILY_CROSS_ROOT_CLEAR_PREFIX in line, (
        f"clause 1: a foreign seal published an unmarked all-clear:\n{line}")
    assert ROUND_DATA in line, f"clause 1: the root this seal measured:\n{line}"
    assert DATA_LIVE in line, (
        f"clause 1: the root the section it closed measured:\n{line}")
    # #3628's alert-side label is a different line in a different section; the seal-side
    # marker must not be mistaken for it, or the node below re-labels nothing new.
    assert notify.DAILY_CROSS_ROOT_PREFIX not in text, (
        f"the seal-side marker borrowed #3628's own prefix:\n{text}")
    # The finding survives its own retraction (#1536): the reader still sees what was
    # open, and can now see which root the clearance spoke for.
    assert "/home/alansrobotlab/lloyd-data/kg.sqlite" in text, text
    # …and the incident is genuinely closed, so the next finding is a NEW section rather
    # than a refresh of history — #3628's re-alert, untouched by this change.
    assert n._vault_note(DATA_SHADOW_TITLE, "a second, separate finding",
                         coalesce=True) is True
    text = note.read_text(encoding="utf-8")
    assert len(_sections_of(text, DATA_SHADOW_TITLE)) == 2, (
        f"the marked seal left the incident open to the next check:\n{text}")


def test_an_all_clear_that_measured_the_root_the_section_measured_is_unchanged(
        tmp_path):
    """#3792 clause 2: the honest hourly all-clear keeps today's bytes, marker-free.

    The same section, and a clearance naming the root its body already names (plus the
    code tree every shipped clearance names). The `cleared:` line is then byte-for-byte
    `cleared: ` plus the producer's own sentence with nothing appended, and the word
    `cross-root` appears nowhere in the note — the live guardian's ordinary all-clear may
    not start carrying a marker, which would make the marker mean nothing.
    """
    n = _notifier(tmp_path)
    assert n.alert("error", DATA_SHADOW_TITLE, DATA_SHADOW_BODY, coalesce=True)
    note = _vault_note(n)

    assert n.resolve(DATA_SHADOW_TITLE, _clearance(TREE_LIVE, DATA_LIVE),
                     measured_root=DATA_LIVE) is True

    text = note.read_text(encoding="utf-8")
    assert notify.DAILY_STILL_OPEN not in text, text
    assert _cleared_line(text, DATA_SHADOW_TITLE) == (
        f"{notify.DAILY_CLEARED_PREFIX} {_clearance(TREE_LIVE, DATA_LIVE)}"), (
        "clause 2: the same-root all-clear is no longer the sentence the producer "
        "wrote")
    assert "cross-root" not in text, (
        f"clause 2: an honest seal grew a marker:\n{text}")


def test_the_seal_side_marker_names_no_path_the_two_sides_do_not_already_print(
        tmp_path):
    """#3792 clause 3: two roots in the marker, and nothing that is not one of them.

    The clearance a round's check writes prints THREE absolute paths: its code tree, its
    data root, and the files its body names. The marker's job is to say which root the
    seal measured and which the section measured — so it names `ROUND_DATA` and
    `DATA_LIVE` and NO THIRD ABSOLUTE PATH: `ROUND_TREE`, which the clearance prints as
    plainly as its data root, stays out. That is the half #3628's own review advisories
    never pinned (`tests/test_guardian_predicates.py:2277`,
    `tests/test_guardian_data_root_shadow.py:633` assert both roots are IN the label and
    never what must be out of it), and the reason the marker reads the measured root from
    the caller rather than sweeping every path token out of the clearance.
    """
    n = _notifier(tmp_path)
    assert n.alert("error", DATA_SHADOW_TITLE, DATA_SHADOW_BODY, coalesce=True)
    note = _vault_note(n)

    assert n.resolve(DATA_SHADOW_TITLE, _clearance(ROUND_TREE, ROUND_DATA),
                     measured_root=ROUND_DATA) is True

    line = _cleared_line(note.read_text(encoding="utf-8"), DATA_SHADOW_TITLE)
    # The set IS the whole clause: `_absolute_roots` returns every absolute token in the
    # marker segment, so equality with exactly these two roots says the clearance's own
    # code tree — printed in that same sentence, one step ahead of the marker — did not
    # join it. Written as a set because `ROUND_TREE` is a raw SUBSTRING of `ROUND_DATA`
    # (the `/lloyd` vs `/lloyd-data` trap `_root_covered` dodges with its trailing slash),
    # which makes `ROUND_TREE not in line` an assertion with no false answer: if this
    # method ever fell back to sweeping the clearance's path tokens instead of the root it
    # was handed, this node is what goes red.
    assert _marker_paths(line) == {ROUND_DATA, DATA_LIVE}, (
        f"clause 3: the marker names only the seal's root and the section's root, each "
        f"printed by one of the two sides already:\n{line}")


def test_a_clearance_naming_only_one_root_marks_exactly_that_root_and_the_section_s(
        tmp_path):
    """#3792 clause 3, the plain case: one foreign root in, two roots out, no third.

    A clearance with no `measured_root` handed over — the shape `_seal_measured_incident`
    and every boot-reconcile call already take — and a sentence naming exactly one
    absolute path, which is not the one the section measured. The marker then names that
    one root plus the section's, and no third path can come from anywhere, because the
    only pool it draws from is the paths these two lines already print.
    """
    n = _notifier(tmp_path)
    assert n.alert("error", DATA_SHADOW_TITLE, DATA_SHADOW_BODY, coalesce=True)
    note = _vault_note(n)

    assert n.resolve(DATA_SHADOW_TITLE,
                     f"nothing content-empty remains under {ROUND_DATA} — the "
                     "instructions above are stale") is True

    line = _cleared_line(note.read_text(encoding="utf-8"), DATA_SHADOW_TITLE)
    assert _marker_paths(line) == {ROUND_DATA, DATA_LIVE}, (
        f"clause 3: one foreign root in, and the marker holds it plus the section's and "
        f"nothing else:\n{line}")


def test_the_runtime_data_check_hands_its_clearance_the_root_it_measured(tmp_path,
                                                                         monkeypatch):
    """#3792 across the seam: a round's hourly check marks its own all-clear, end to end.

    The four nodes above drive `Notifier.resolve` directly. This one runs the producer:
    `Guardian._runtime_data_incident` with `datawatch.DATA_ROOT` pointed at an automod
    round's `home/lloyd-data` — which is exactly what a round's process measures, because
    `gate.py`'s `_child_env` sets `LLOYD_DATA` into the round home while
    `worktree.py:65` symlinks the LIVE `~/obsidian` beside it. The open section in that
    live vault was written against `/home/alansrobotlab/lloyd-data`, the round's check
    finds nothing, and the line its clearance writes must therefore say which root it
    measured. `graph_explain` cannot see this boundary: it is a call from one module into
    another whose value is a process-local constant, and the artefact is a file a third
    process reads hours later.
    """
    import types

    import guardian as G
    import notify

    round_home = tmp_path / "lloyd-work" / "SM_FAKE0000_000000" / "home"
    round_data = round_home / "lloyd-data"
    round_data.mkdir(parents=True)
    vault = tmp_path / "obsidian"
    (vault / "memory").mkdir(parents=True)

    monkeypatch.setattr(G.policy, "REPO", str(round_home / "lloyd"))
    monkeypatch.setattr(G.datawatch, "DATA_ROOT", str(round_data))
    monkeypatch.setattr(G.datawatch, "stray_in_tree", lambda _tree: [])
    monkeypatch.setattr(G.policy, "GUARDIAN_STATE", str(tmp_path / "guardian-state"))
    clock = {"t": 1_762_300_000.0}
    monkeypatch.setattr(G.time, "time", lambda: clock["t"])

    n = notify.Notifier(ledger=tmp_path / "ledger.jsonl", state_dir=tmp_path,
                        vault_root=str(vault), backend_url="http://127.0.0.1:1")
    n.announce = lambda *a, **k: {}
    # The section the LIVE guardian opened an hour ago, in the live vault, measured
    # against the live data root — the bytes `_shadow_alert_body` writes.
    assert n._vault_note(G.RUNTIME_DATA_SHADOW_ALERT_TITLE, DATA_SHADOW_BODY,
                         coalesce=True, measured_root=DATA_LIVE) is True
    note = _vault_note(n)
    assert notify.DAILY_STILL_OPEN in note.read_text(encoding="utf-8")

    g = G.Guardian.__new__(G.Guardian)
    g.notifier = n
    g.data = types.SimpleNamespace(armed=True)
    g._alert_seen = {}
    g.last_alert = ""

    g._runtime_data_incident(clock["t"] + G.policy.STRAY_CHECK_SECONDS)

    text = note.read_text(encoding="utf-8")
    assert notify.DAILY_STILL_OPEN not in text, (
        f"the round's check did not close the section:\n{text}")
    line = _cleared_line(text, G.RUNTIME_DATA_SHADOW_ALERT_TITLE)
    assert notify.DAILY_CROSS_ROOT_CLEAR_PREFIX in line, (
        f"the process the witness names published an unmarked all-clear:\n{line}")
    assert _marker_paths(line) == {str(round_data), DATA_LIVE}, (
        f"the marker names the root this check measured and the root the section "
        f"measured, whatever the two happen to be:\n{line}")


# ── #3882: the vault-snapshot alarm's own seal ────────────────────────────────
#
# `Guardian._vault_snapshot_stream` measured its subject every 15 minutes and re-paged
# it, and had no way to stop saying it: the publish carried no `coalesce`, and no fresh
# reading ever handed the title to `resolve`. The page that shows is
# `~/obsidian/memory/2026-11-04.md` — three sections of `VAULT_SNAPSHOT_ALERT_TITLE` at
# filing (`:26`, `:39`, `:52`), none carrying a `cleared:` line, each still ordering a
# reader to run `restore-vault.sh`, over a backup stream that delivered 276 commits in
# the three days they stood.
#
# The nodes below drive the REAL method over a `Notifier` rooted in a tmp vault, so what
# is asserted is the marker the writer emits and the seal the resolver writes, not a
# stand-in's record of intending to. Only the reading of the backup repository is
# injected, which is the seam this leg's own sibling file already uses:
# `tests/test_guardian_vaultwatch.py:608` replaces `vaultwatch.snapshot_age_state` with an
# `unreadable` stand-in, and where that file needs a STALE reading instead it plants a real
# repository with a back-dated HEAD (`_owner_backup_repo`), which a fixture can do but a
# gate cannot steer to a chosen age on the machine it runs on. A FRESH reading has no such
# plant — the repository would have to have committed in the last three minutes — so the
# reading is handed in, and every byte the clauses are about still comes from the real
# writer and the real resolver.

import datetime                    # noqa: E402  (the dated daily note is named by today)
import guardian as gmod            # noqa: E402  (Guardian, VAULT_SNAPSHOT_ALERT_TITLE)

SNAPSHOT = gmod.VAULT_SNAPSHOT_ALERT_TITLE

#: Shaped like the producer's dict key for key (`vaultwatch.snapshot_age_state`'s return),
#: because the leg reads `fresh`, `owned`, `max_age` and `text` out of it and a fixture
#: with a typo in a key would pass a leg that crashes on the real reading.
SNAPSHOT_REPO = "/home/alansrobotlab/.local/state/lloyd-vault-backup/vault.git"
SNAPSHOT_SHA = "14f47bfd" + "0" * 32

#: The body the leg publishes, reduced to the two lines that matter here: the measurement,
#: and the directive a reader obeys. That directive is why an unsealed block is a defect
#: and not a formatting slip.
SNAPSHOT_BODY = (
    "vault.git commit 00000000 is 4 h 12 m old, past the 3 h limit.\n\n"
    "Restore into a side directory with "
    "`~/lloyd/scripts/backup/restore-vault.sh <rev> <dest>` — verify the copy.")


def _snapshot_state(*, fresh: bool = True, owned: bool = True) -> dict:
    """One reading of the backup repository, in the producer's own shape."""
    age = 180.0 if fresh else 15_120.0
    return {"repo": SNAPSHOT_REPO, "sha": SNAPSHOT_SHA, "committed_at": 1_762_500_000,
            "age": age, "max_age": 3 * 3600.0, "fresh": fresh, "owned": owned,
            "beat_age": age,
            "text": (f"{SNAPSHOT_REPO} commit {SNAPSHOT_SHA[:8]} is 3 m old" if fresh
                     else f"{SNAPSHOT_REPO} commit {SNAPSHOT_SHA[:8]} is 4 h 12 m old, "
                          "past the 3 h limit")}


def _snapshot_guardian(tmp_path, monkeypatch, state):
    """A `Guardian` on a tmp vault whose snapshot reading is exactly `state`.

    `state` may be an exception, which is the shape of a read that did not happen:
    `snapshot_age_state` shells out to `git`, and the leg's own contract is that a backup
    reading may not raise into the tick that stops sync within one tick of a wipe.
    """
    import types
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    # A `Path`, not a str: the alert body this leg publishes builds its published-command
    # line through `policy.published_watchdog_script`, which does
    # `Path(policy.REPO).expanduser()`, and a str REPO raises `AttributeError` inside the
    # publish — the failure lands in the daily note of exactly the family this node is
    # about, so the node would report a missing marker where the real fault is a fixture.
    monkeypatch.setattr(gmod.policy, "REPO", tmp_path / "repo")
    monkeypatch.setattr(gmod.policy, "VAULT_ROOT", str(tmp_path / "vault"))
    monkeypatch.setattr(gmod.policy, "DATA_ROOT", str(tmp_path / "data"))
    # A `Path` for the same reason as `REPO`: `published_watchdog_script` calls
    # `GUARDIAN_STATE.expanduser()`, and the alert body this leg publishes runs through it
    # (#3795). A str here fails inside the publish rather than in the fixture.
    monkeypatch.setattr(gmod.policy, "GUARDIAN_STATE", tmp_path / "guardian-state")

    def _state(*_a, **_k):
        if isinstance(state, Exception):
            raise state
        return state

    monkeypatch.setattr(gmod.vaultwatch, "snapshot_age_state", _state)
    args = types.SimpleNamespace(
        repo=str(tmp_path / "repo"), state=str(tmp_path / "state.json"),
        guardian_state=str(tmp_path / "guardian-state"), supervisor_sock="/nonexistent",
        backend_url="http://127.0.0.1:1/health", mcp_url="http://127.0.0.1:2/health",
        programs="lloyd-mc:lloyd-backend", interval=5.0)
    (home / "obsidian" / "memory").mkdir(parents=True, exist_ok=True)
    guard = gmod.Guardian(args)
    guard.notifier.vault_root = home / "obsidian"
    guard.notifier._last_key = None        # one firing per node, not the 15-min throttle
    guard.alert_file_incident = lambda: (home / "ALERT.md", False)
    guard._vault_snapshots_checked_at = -1e9   # the leg's own throttle, opened once
    return guard


def _snapshot_note(guard) -> str:
    return (guard.notifier.vault_root / "memory"
            / f"{datetime.date.today()}.md").read_text(encoding="utf-8")


def _tick(guard, now: float) -> None:
    """One tick of the leg, with the notifier's own repeat throttle opened for it.

    Three gates stand between a tick and the daily note, and only the first is on the
    clock this node controls: the leg's own `SNAPSHOT_CHECK_SECONDS` gate reads the `now`
    passed in, while `Guardian.alert`'s `_alert_seen` window and `Notifier`'s `_last_key`
    both read `time.time()`. Two ticks a simulated 15 minutes apart are 40 milliseconds
    apart in real time, so the second fan-out would be logged as "(suppressed repeat)"
    and the fold asserted below would never be reached. Clearing the other two is what
    the tripwire nodes beside this one do for the same reason.
    """
    guard._alert_seen.clear()
    guard.notifier._last_key = None
    guard._vault_snapshot_stream(now)


def test_a_stale_snapshot_publish_writes_a_section_the_resolver_can_seal(tmp_path,
                                                                        monkeypatch):
    """Clause 1: a fan-out of this alarm appends ONE section that carries the open marker.

    `memory/2026-11-04.md` is the falsifier this node inverts: three sections, three
    bodies, zero markers, because the uncoalesced append branch of `_vault_note` never
    writes `notify.DAILY_STILL_OPEN` and that marker is the only thing
    `Notifier._daily_open_at` calls a section open. So the assertion that matters is not
    "the note got the alarm" — it always did — but "the note got one section, and it is
    open in the sense the resolver understands".
    """
    guard = _snapshot_guardian(tmp_path, monkeypatch, _snapshot_state(fresh=False))
    _tick(guard, 10_000.0)
    _tick(guard, 10_900.0)                     # the next tick, same stalled condition

    note = _snapshot_note(guard)
    assert note.count(f"## Self-mod guardian: {SNAPSHOT}") == 1, (
        "two ticks of one stalled stream wrote two bodies into the page a person reads; "
        f"that is 2026-11-04 again:\n{note}")
    assert note.count(notify.DAILY_STILL_OPEN) == 1, (
        "the section carries no standing open marker, which is the state that makes "
        f"`resolve` unable to seal it:\n{note}")
    assert guard.notifier._daily_open_at(note, SNAPSHOT) is not None, (
        "`Notifier._daily_open_at` does not call this section open, so clause 2's seal "
        "would have nothing to seal")
    assert note.count("Restore into a side directory") == 1, (
        "the second firing pasted the body again rather than refreshing the open section "
        "in place — `memory/2026-11-04.md` is three copies of this one directive:\n{note}")


def test_a_fresh_owned_snapshot_pass_seals_that_section_on_the_same_pass(tmp_path,
                                                                        monkeypatch):
    """Clause 2: a fresh, owned reading seals the open section on the pass that reads it.

    The seal is the whole point of #3882 and it is the half #3132 declined to build ("the
    next fresh reading is simply quiet"). Quiet is what left a restore directive standing
    on a daily note for three days over a stream that delivered 276 commits in that time.
    The reading here is the leg's own — one injected `snapshot_age_state`, no notifier
    stubbed — so the seal's numbers are the ones the fresh reading printed.
    """
    guard = _snapshot_guardian(tmp_path, monkeypatch, _snapshot_state(fresh=False))
    _tick(guard, 10_000.0)
    monkeypatch.setattr(gmod.vaultwatch, "snapshot_age_state",
                        lambda *_a, **_k: _snapshot_state(fresh=True))

    _tick(guard, 10_900.0)

    note = _snapshot_note(guard)
    assert notify.DAILY_CLEARED_PREFIX in note, (
        "the stream came back and the daily note still says snapshots are not arriving:\n"
        f"{note}")
    assert guard.notifier._daily_open_at(note, SNAPSHOT) is None, (
        "the open marker survived the seal, so the next publish would fold into a "
        "cleared incident")
    assert SNAPSHOT_SHA[:8] in note and "3 m" in note, (
        "the seal does not name the commit and age the fresh reading measured, so a "
        f"reader cannot check it: {note}")
    assert "restore" in note.lower() and "restores nothing" in note, (
        "the seal must repeat the half-truth the alarm could not carry — a snapshot "
        f"repo restores no running tree:\n{note}")


def test_a_snapshot_seal_is_never_written_from_a_reading_that_is_not_the_vault_s(
        tmp_path, monkeypatch):
    """Clause 3: no seal from a repository this check may not speak for, and none from a
    read that did not happen.

    At the base commit this node passed vacuously: the leg had no retracting branch at
    all, so nothing could write a seal from anywhere. It is the branch added by clause 2
    that makes a refusal something this check can be wrong about, which is why it stays.

    Two refusals, one assertion each on the note's bytes. An unowned repository is the
    #3149 case — `memory/2026-10-21.md` carries three sections that raised this title
    about somebody's fixture repo — and an unreadable one is the case the leg already
    refuses to page about. Freshness is irrelevant to both: the reading must first be
    *of the vault* before it can retire an alarm about the vault, and on this box the
    raising side is silent while the retracting side would still be writing.
    """
    for label, state in (("unowned", _snapshot_state(fresh=True, owned=False)),
                         ("unreadable", OSError("git: blocked by the sandbox"))):
        guard = _snapshot_guardian(tmp_path / label, monkeypatch,
                                   _snapshot_state(fresh=False))
        _tick(guard, 10_000.0)
        before = _snapshot_note(guard)
        assert notify.DAILY_STILL_OPEN in before, (
            "the publish left no open marker for a seal to consume, so the byte-identity "
            f"below would be claimed over a note that never held an open section:\n{before}")
        assert notify.DAILY_CLEARED_PREFIX not in before, (
            "the publish itself sealed its own section, so the refusal below would be "
            "asserted over an empty note")

        def _reading(*_a, _s=state, **_k):
            if isinstance(_s, Exception):
                raise _s
            return _s

        monkeypatch.setattr(gmod.vaultwatch, "snapshot_age_state", _reading)

        _tick(guard, 10_900.0)                  # must not raise into the tick

        after = _snapshot_note(guard)
        assert after == before, (
            f"the {label} reading sealed a snapshot section: a retraction nobody "
            f"measured ({label})")
        assert after.count(notify.DAILY_CLEARED_PREFIX) == 0, (
            f"a `cleared:` line reached the note through the {label} branch")


def test_an_uncoalesced_snapshot_section_is_the_one_resolve_could_never_seal(tmp_path,
                                                                            monkeypatch):
    """The deeper of #3882's two causes, pinned so a future round cannot re-ship it.

    A `resolve` call alone would not have fixed `memory/2026-11-04.md`, and a round that
    reads the item's title and adds only the call would land that. `resolve` seals only
    what `_daily_open_at` calls open, and an uncoalesced section carries no marker — so
    the pre-fix leg's answer to a healthy stream was a return value of True and a note
    that did not change. This node is the pre-fix state, written against the current
    notifier, and it is why `coalesce=True` is load-bearing rather than tidy.
    """
    guard = _snapshot_guardian(tmp_path, monkeypatch, _snapshot_state(fresh=False))
    guard.notifier.alert("error", SNAPSHOT, SNAPSHOT_BODY)   # the OLD call: no coalesce
    before = _snapshot_note(guard)
    assert before.count(notify.DAILY_STILL_OPEN) == 0, (
        "this node's premise is the uncoalesced append; if the writer changed, the "
        "falsifier below is proving nothing")

    assert guard.notifier.resolve(SNAPSHOT, "the stream is fresh again") is True
    assert _snapshot_note(guard) == before, (
        "`resolve` reported a seal and wrote nothing: exactly the dead end an "
        "uncoalesced section is")


# ── #3933: the store a seal names must outlive the tree that wrote it ────────
#
# `vaultwatch.snapshot_repo()` returned the import-time `Path.home() / rel` unresolved, so
# a guardian tick run by an automod round — `HOME=<round>/home`, whose `.local` is a
# symlink into the real home (`scripts/automod/worktree.py`: `round_home`, and
# `HOME_LINK_SKIP`, which spares only `lloyd` and `lloyd-data`) — read the backup
# repository THROUGH that link and published the round's own doomed path. The page is
# `~/obsidian/memory/2026-11-07.md`, whose one `cleared:` line still instructs a reader to
# `ls -d /home/alansrobotlab/lloyd-work/SM_20261107_145946/home/.local/state/`
# `lloyd-vault-backup/vault.git`, a root deleted with the round, while the store it meant
# answers at `/home/alansrobotlab/.local/state/lloyd-vault-backup/vault.git` and holds the
# commit the line cites (`a7c268ca`, committed `2026-11-07 08:00:37 -0800`). The seal was
# honest at the instant it was written and became unreadable in the sweep — the worse
# failure of the two, because the surviving page is now an instruction that cannot be run.
#
# #3795 landed that rule for the published COMMAND line (`policy.published_watchdog_script`)
# and left the published STORE, which `guardian.py` feeds into the alert body's "`<repo>`
# is the vault's only copy outside the vault" sentence and `snapshot_age_state` feeds into
# the seal's own sentence through `state["text"]`. Two published surfaces, one fix site.
#
# These nodes run the REAL leg over a REAL planted repository with NOTHING injected into
# the reading — the #3882 nodes above inject a stand-in because they were about the two
# edges of the seal, and this item is about the STRING the reading carries. The path token
# under test therefore comes from the shipping accessor, and the boundary is crossed for
# real: a round's `$HOME`, then the sweep that deletes it.

import os as _os                    # noqa: E402
import shutil as _shutil            # noqa: E402
import subprocess as _subprocess    # noqa: E402
import time as _time                # noqa: E402

#: Past `vaultwatch.SNAPSHOT_MAX_AGE_SECONDS` (3 h) by enough that a slow gate cannot
#: read it as fresh, and the age `memory/2026-11-07.md:19`'s own alarm printed.
STALE_AGE = 4 * 3600.0 + 12 * 60.0

#: The round's component. Absent from a page the fix wrote, and the marker of a page it
#: did not: nothing else in these fixtures is spelled this way.
ROUND = "SM_PROBE"


def _git_c(*args: str, at_age: float | None = None) -> str:
    """`git` with its identity on the command line.

    `-c user.*` rather than the ambient config because these nodes MOVE `$HOME` — the
    whole point of them does — and a plant that depends on whichever home the gate handed
    the run fails in the fixture instead of reporting the defect.
    """
    env = dict(_os.environ)
    if at_age is not None:
        stamp = int(_time.time() - at_age)
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = f"@{stamp} +0000"
    r = _subprocess.run(
        ["git", "-c", "user.name=guardian-node",
         "-c", "user.email=guardian-node@example.invalid", *args],
        capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"{args}: {r.stdout}{r.stderr}"
    return r.stdout.strip()


def _plant_backup_repo(owner_dir: Path, *, age_seconds: float) -> Path:
    """A bare backup repository in the owner position, its HEAD `age_seconds` old.

    `backup-vault.sh`'s shape — a bare repo driven by `--git-dir` against a separate
    `--work-tree` — because the stamp the check reads is git's `%ct`, and the planted age
    is read back out of git below: a node that planted nothing and passed anyway is the
    failure this helper exists to refuse.
    """
    owner_dir.mkdir(parents=True, exist_ok=True)
    repo, work = owner_dir / "vault.git", owner_dir / "vault-worktree"
    work.mkdir(parents=True)
    (work / "note.md").write_text("a note\n")
    _git_c("init", "--quiet", "--bare", str(repo))
    _git_c("--git-dir", str(repo), "--work-tree", str(work), "add", "--all", ".")
    _git_c("--git-dir", str(repo), "--work-tree", str(work), "commit", "--quiet",
           "--no-verify", "-m", "snapshot", at_age=age_seconds)
    stored = int(_git_c("--git-dir", str(repo), "log", "-1", "--format=%ct"))
    assert abs(_time.time() - stored - age_seconds) < 120.0, (
        f"git stored a HEAD {stored}, not one {age_seconds} s old: the age this whole "
        "family weighs is the commit's own, so a plant that did not land makes every "
        "assertion below about a stream that is actually fine")
    return repo


def _deliver_again(repo: Path) -> str:
    """One more snapshot reaches the repository: the outage is over, HEAD is now's.

    The fresh edge cannot be planted by a date the way a stale one can — "committed within
    the last three hours" is whatever the run's clock says when it runs — so this is what
    `backup-vault.sh` does between a stall and the tick that retracts it.
    """
    work = repo.parent / "vault-worktree"
    (work / "note.md").write_text("a note\nand a delivery\n")
    _git_c("--git-dir", str(repo), "--work-tree", str(work), "add", "--all", ".")
    _git_c("--git-dir", str(repo), "--work-tree", str(work), "commit", "--quiet",
           "--no-verify", "-m", "snapshot after the stall")
    return _git_c("--git-dir", str(repo), "log", "-1", "--format=%H")


def _guardian_in_home(home: Path, tmp_path: Path, monkeypatch):
    """A real `Guardian` standing in `home`, reading a real repository: nothing injected.

    `_snapshot_guardian` above replaces `snapshot_age_state` with a stand-in. Here the
    accessor, the `git` call, the age judgement, the alert body and the seal sentence all
    have to be the shipped ones, because the thing under test is a path token that only
    they produce. So the fixture arranges a filesystem and `$HOME` — the two facts that
    decide that token — and nothing else.

    The daily note lives UNDER `home`, which is how a round's tick reached the live page
    at all: `obsidian` is in the round home's link farm, and `HOME_LINK_SKIP` spares only
    `lloyd` and `lloyd-data`.
    """
    monkeypatch.setenv("HOME", str(home))
    # A `Path` for `REPO` and `GUARDIAN_STATE`, as above: `policy.published_watchdog_script`
    # calls `.expanduser()` on both while building the alert body this leg publishes.
    monkeypatch.setattr(gmod.policy, "REPO", tmp_path / "repo")
    monkeypatch.setattr(gmod.policy, "VAULT_ROOT", str(tmp_path / "vault"))
    monkeypatch.setattr(gmod.policy, "DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(gmod.policy, "GUARDIAN_STATE", tmp_path / "guardian-state")
    args = types.SimpleNamespace(
        repo=str(tmp_path / "repo"), state=str(tmp_path / "state.json"),
        guardian_state=str(tmp_path / "guardian-state"), supervisor_sock="/nonexistent",
        backend_url="http://127.0.0.1:1/health", mcp_url="http://127.0.0.1:2/health",
        programs="lloyd-mc:lloyd-backend", interval=5.0)
    (home / "obsidian" / "memory").mkdir(parents=True, exist_ok=True)
    guard = gmod.Guardian(args)
    guard.notifier.vault_root = home / "obsidian"
    guard.notifier._last_key = None
    guard.alert_file_incident = lambda: (home / "ALERT.md", False)
    guard._vault_snapshots_checked_at = -1e9
    return guard


def _round_home(tmp_path: Path, real_home: Path) -> Path:
    """`<tmp>/SM_PROBE/home`, the layout a round's tick is handed, over `real_home`.

    `.local` and `obsidian` are links into the real home and nothing else: a store that
    reads through a link but is printed under the link's own name, and a vault note both
    processes reach. Everything under `real_home` survives what `real_home`'s neighbour
    does not.
    """
    (real_home / ".local").mkdir(parents=True, exist_ok=True)
    (real_home / "obsidian" / "memory").mkdir(parents=True, exist_ok=True)
    home = tmp_path / ROUND / "home"
    home.mkdir(parents=True, exist_ok=True)
    (home / ".local").symlink_to(real_home / ".local", target_is_directory=True)
    (home / "obsidian").symlink_to(real_home / "obsidian", target_is_directory=True)
    return home


def _note_of(real_home: Path) -> str:
    return (real_home / "obsidian" / "memory"
            / f"{datetime.date.today()}.md").read_text(encoding="utf-8")


def _store_tokens(text: str) -> list[str]:
    """The absolute backup-store path tokens one passage names — what a reader goes to `stat`.

    Asserted non-empty on purpose: the clauses below are claims about a published path, and
    a node that silently had no path to check is #3792's amended owed row — a green that
    only an empty corpus could produce.
    """
    tokens = [t for t in notify._absolute_roots(text) if "lloyd-vault-backup" in t]
    assert tokens, f"that passage names no backup store, so nothing was measured:\n{text}"
    return tokens


def _witness_of_a_round_seal(tmp_path, monkeypatch) -> tuple[str, str, Path]:
    """The 2026-11-07 shape, end to end: the live watchdog opens the section, a round's
    tick seals it. Returns `(note, cleared line, round home)`.

    Two processes because the page was written by two. The section names the store as the
    live watchdog knows it — which is what makes a seal naming a DIFFERENT route a foreign
    clearance rather than a refresh, and what makes clause 4's marker falsifiable at all.
    """
    real_home = tmp_path / "real"
    repo = _plant_backup_repo(real_home / gmod.vaultwatch.SNAPSHOT_OWNER_DIR_REL,
                              age_seconds=STALE_AGE)
    live = _guardian_in_home(real_home, tmp_path, monkeypatch)
    _tick(live, 10_000.0)                        # the stall, published by the live tick
    assert notify.DAILY_STILL_OPEN in _note_of(real_home), (
        "the live tick left no open section, so a seal below would be writing into a page "
        "that never carried the alarm it is closing")

    sha = _deliver_again(repo)                   # the stream comes back
    round_home = _round_home(tmp_path, real_home)
    ticking = _guardian_in_home(round_home, tmp_path, monkeypatch)
    _tick(ticking, 10_900.0)                     # and a round's tick retracts it

    note = _note_of(real_home)
    line = _cleared_line(note, SNAPSHOT)
    assert sha[:8] in line, (
        "the round's tick wrote a seal that does not carry the commit it measured, so "
        f"these clauses would be graded over a line no one wrote:\n{line}")
    return note, line, round_home


def test_an_honest_seal_still_lands_and_its_store_path_can_be_walked(tmp_path, monkeypatch):
    """Clause 2, and the load-bearing green case: the fix retires a spelling, not the seal.

    A readable fixture repository in the owner position, an ordinary (unredirected) home, a
    stale tick that opens the section and a fresh tick that closes it: the open marker is
    gone, the `cleared:` line names the commit and age the fresh reading measured, and the
    backup-store token in it `stat()`s. Nothing was refused, and no marker was added —
    `memory/2026-11-07.md:17`'s own body is honest prose, and an honest seal is supposed to
    stay byte-identical marker-less prose (#3792 clause 2).
    """
    home = tmp_path / "home"
    repo = _plant_backup_repo(home / gmod.vaultwatch.SNAPSHOT_OWNER_DIR_REL,
                              age_seconds=STALE_AGE)
    guard = _guardian_in_home(home, tmp_path, monkeypatch)

    _tick(guard, 10_000.0)
    opened = _snapshot_note(guard)
    assert notify.DAILY_STILL_OPEN in opened, (
        "the stall published no open section, so the seal below would seal nothing")
    assert set(_store_tokens(opened)) == {str(repo)}, (
        "the alarm body named some store other than the one this tick measured")

    sha = _deliver_again(repo)
    _tick(guard, 10_900.0)

    note = _snapshot_note(guard)
    assert notify.DAILY_CLEARED_PREFIX in note, (
        "the stream came back and the page still says snapshots are not arriving — the "
        f"fix refused an honest seal:\n{note}")
    line = _cleared_line(note, SNAPSHOT)
    assert guard.notifier._daily_open_at(note, SNAPSHOT) is None, (
        "the open marker survived the seal")
    assert sha[:8] in line and "4 h 12 m" not in line, (
        "the seal is not carrying the FRESH reading — the commit the second tick measured "
        f"and its age, not the stalled one's: {line}")
    for token in _store_tokens(line):
        Path(token).stat()                       # the route a reader can walk today
    assert notify.DAILY_CROSS_ROOT_CLEAR_PREFIX not in line, (
        f"an honest all-clear gained a #3792 marker, which is what makes the marker mean "
        f"nothing:\n{line}")


def test_a_seal_written_under_a_round_home_names_a_store_the_sweep_leaves_behind(
        tmp_path, monkeypatch):
    """Clause 3: the repository reachable only through a round-style HOME symlink is still
    named by its resolved route, so every store token the seal wrote `stat()`s AFTER the
    round home is deleted.

    This is the property the 11-07 line failed. The seal it wrote was readable for exactly
    as long as the round that wrote it, which is no time at all to a reader who arrives the
    next morning — and `already_sealed` (`scripts/maintenance/guardian_note_sweep.py`) bars
    the sweep from editing an already-`cleared:` block, so a page can carry the dead
    instruction indefinitely whatever later rounds do.
    """
    note, line, round_home = _witness_of_a_round_seal(tmp_path, monkeypatch)

    _shutil.rmtree(tmp_path / ROUND)             # the sweep takes the round's tree

    assert not (tmp_path / ROUND).exists(), "the sweep did not take the round's tree"
    assert not round_home.exists()
    assert notify.DAILY_STILL_OPEN not in note, (
        f"the round's tick closed the section and left its open marker standing:\n{note}")
    for token in _store_tokens(line):
        Path(token).stat()                       # still walks, with the round gone


def test_a_round_rooted_seal_of_a_store_the_page_already_names_carries_no_marker(
        tmp_path, monkeypatch):
    """Clause 4: that same seal carries no `cross-root clearance:` marker.

    The marker means "this all-clear measured a root the section did not", and the round's
    tick DID measure the store the live watchdog's section named — through a link, which is
    the same repository (`snapshot_repo_is_owned` resolves both sides for exactly this
    reason). The unresolved spelling made the two strings disagree, so `memory/2026-11-07.md`
    carries a marker asserting that the process which wrote the seal stood somewhere the
    vault backup was not. With the resolved store the seal root is covered by the section's
    roots, `_cross_root_clear_marker` returns "", and an honest seal is plain prose again.
    """
    note, line, _round_home = _witness_of_a_round_seal(tmp_path, monkeypatch)

    assert ROUND not in line, (
        f"the seal published the round's own tree as the store's address:\n{line}")
    assert notify.DAILY_CROSS_ROOT_CLEAR_PREFIX not in line, (
        "a seal of the store the section already names was labelled a clearance of a "
        f"different root, which is the claim the marker exists to make and this is not:\n"
        f"{line}")


def test_a_stale_reading_taken_under_a_round_home_publishes_the_durable_store(
        tmp_path, monkeypatch):
    """Clause 5: the alert BODY's "`<repo>` is the vault's only copy outside the vault"
    sentence names the resolved store, not one under the round home.

    The scoping claim that this surface was already fixed is false: #3795 landed the
    published-COMMAND line, and `memory/2026-11-07.md`'s body names the real path only
    because that body came from the live watchdog. A round's tick that reads a stall raises
    this sentence too, and a reader told to check "the vault's only copy outside the vault"
    at a path the sweep deleted is being sent to check a copy that does not exist.
    """
    real_home = tmp_path / "real"
    repo = _plant_backup_repo(real_home / gmod.vaultwatch.SNAPSHOT_OWNER_DIR_REL,
                              age_seconds=STALE_AGE)
    round_home = _round_home(tmp_path, real_home)
    ticking = _guardian_in_home(round_home, tmp_path, monkeypatch)

    _tick(ticking, 10_000.0)                     # the stall, seen through the link

    note = _note_of(real_home)
    sections = _sections_of(note, SNAPSHOT)
    assert len(sections) == 1, note
    assert f"{repo} is the vault's only copy outside the vault" in note, (
        "the sentence that sends a reader to the vault's only off-vault copy named a "
        f"repository other than the one that holds it:\n{note}")
    assert ROUND not in note, note

    _shutil.rmtree(tmp_path / ROUND)             # the sweep, then a reader obeys the page

    for token in _store_tokens(sections[0]):
        Path(token).stat()
