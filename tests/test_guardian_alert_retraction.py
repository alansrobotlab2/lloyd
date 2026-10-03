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
