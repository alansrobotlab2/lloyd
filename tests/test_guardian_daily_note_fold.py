"""An unchanged guardian alarm may cost today's daily note one body, not one per tick.

The two live witnesses are `~/obsidian/memory/2026-10-29.md` and
`~/obsidian/memory/2026-11-04.md`. The first holds five sections of one title —
11,250 B of that file's 12,809 B, 87.8 % — and the second holds three of another, the
first two of them byte-identical (1,663 B each, sha256 prefix `be3db231c51c`),
5,146 B of 6,872 B. Every scheduler alarm line on those same pages carries an explicit
fold clause (`[daily-note fold: this alarm writes at most one line a day here …]`,
`app/discord_notify.py:346-349`), so the writer that goes through the scheduler is
capped at one line a day by construction and the one that does not supplied 75-88 % of
the bytes on the page a person reads. `backlog/1592`'s `ruling:` makes that daily note
the destination of record for alarms with nowhere else to go, which makes this an
alerting defect rather than a formatting one.

Two doors reach the same append branch of `Notifier._vault_note`, and both are pinned
here:

* **The alarm that never asks to coalesce.** `guardian.py:1836` raises
  `VAULT_SNAPSHOT_ALERT_TITLE` with no `coalesce=True`, so `_vault_note` never looks
  for an open section and appends a whole body per fan-out. That is 2026-11-04.
* **The coalesced alarm whose section was sealed from another root.** A `cleared:`
  line written by a process that measured a different data root removes the open
  marker, so the next hourly tick finds nothing open and re-pastes. That is
  2026-10-29, where each of the five `cleared:` lines names a different
  `~/lloyd-work/SM_20261029_*/home/lloyd-data` — one short-lived process per tick.

What the fold does NOT do is swallow the two firings that are not repetitions. An
honest all-clear followed by a genuinely new incident still opens its own section
(#3150 clause 4, pinned in `tests/test_guardian_alert_retraction.py`), and the FIRST
re-paste after a foreign seal still prints its own labelled section (#3628 clause 3,
pinned by `test_a_shadow_finding_sealed_from_another_root_re_alerts_with_the_label` in
`tests/test_guardian_data_root_shadow.py` — that node is this item's clause 5, and
`test_the_second_foreign_seal_folds_and_the_label_that_explains_it_survives` below
shows the fold beginning where that node stops). Folding is a visible collapse: a line
naming the title, how many of today's firings it folds, and the newest stamp, written
on every firing after the first.

One seam inside the fold is worth naming, because two fixes meet in it: `_foreign_roots`
answers "did this seal come from another root?" for BOTH #3628's `cross-root seal:` label
and the fold's own predicate — that is why they can never disagree — and it reads a
clearance's OWN sentence, not every path token on its line. #3792's
`cross-root clearance:` marker prints the live data root inside the very line the next
tick has to judge, so counting the marker would call a foreign seal honest and drop the
label and the fold together.
`test_a_foreign_clearance_carrying_the_clear_marker_still_keeps_the_fold_bound` pins it.

Everything runs over a throwaway vault under `tmp_path` on a standing day; nothing here
touches `~/obsidian`.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
for _p in (str(ROOT), str(GUARDIAN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import guardian as gmod  # noqa: E402
import notify  # noqa: E402

DAY = date(2026, 11, 5)

#: The alarm that wrote `memory/2026-11-04.md`. It is raised WITHOUT `coalesce=True`
#: (`guardian.py:1836-1860`), which is the whole of door 1.
SNAPSHOT = gmod.VAULT_SNAPSHOT_ALERT_TITLE
#: The alarm that wrote `memory/2026-10-29.md`. It coalesces and hands over the root it
#: measured, which is what makes a foreign seal visible to it (door 2).
SHADOW = gmod.RUNTIME_DATA_SHADOW_ALERT_TITLE

#: The data root the shipped check walks, quoted from the live box because it is the
#: string the alert body and an honest clearance both print.
LIVE_ROOT = "/home/alansrobotlab/lloyd-data"

#: The root a foreign process sealed the section from — an automod-round worktree, the
#: shape every `cleared:` line on 2026-10-29 names.
FOREIGN_A = "/home/alansrobotlab/lloyd-work/SM_20261029_213457/home/lloyd-data"
FOREIGN_B = "/home/alansrobotlab/lloyd-work/SM_20261029_221142/home/lloyd-data"
FOREIGN_C = "/home/alansrobotlab/lloyd-work/SM_20261029_225616/home/lloyd-data"

#: A line another writer puts after the guardian's own, to prove a fold rewrites only
#: its own bytes. This is the shape `app/discord_notify.py` leaves on the live notes.
FOREIGN_LINE = ("- Dropped alert (scheduled-task): 3 task(s) past their next_run "
                "[daily-note fold: 2 earlier fires of this alarm today folded into "
                "this line; the next line comes when that reading doubles]")

#: The bulk of the vault-snapshot alarm's body, condensed. A fold that silently deleted
#: a body rather than collapsing repeats would be indistinguishable from one that
#: folded correctly unless the first body's own words are asserted as still standing.
RESTORE = ("Restore into a side directory with "
           "`~/lloyd/scripts/backup/restore-vault.sh <rev> <dest>` — never over the "
           "live vault — and check the tripwire marker before believing a stall. ")


def _notifier(tmp_path: Path) -> "notify.Notifier":
    """A real `Notifier` over a throwaway vault, standing on `DAY`.

    `_today` is the only date the notifier consults, so one assignment fixes the day
    these nodes fold against; `backend_url` points at a dead port so no board POST can
    reach the live one, and room channels are muted by `tests/conftest.py`. `_vault_note`
    and `resolve` are the shipped methods, so what these nodes assert is what the hourly
    watchdog writes into `memory/<date>.md`.
    """
    vault = tmp_path / "obsidian"
    (vault / "memory").mkdir(parents=True, exist_ok=True)
    n = notify.Notifier(ledger=tmp_path / "ledger.jsonl", state_dir=tmp_path,
                        vault_root=str(vault), backend_url="http://127.0.0.1:1")
    n._today = lambda: DAY
    assert n._daily_note().name == f"{DAY.isoformat()}.md", n._daily_note()
    return n


def _note(n: "notify.Notifier") -> Path:
    return n._daily_note()


def _titles(text: str) -> list[str]:
    """The section headings in `text`, as the shipped section parser reads them."""
    return [sect[0] for sect in notify.Notifier._daily_sections(text)]


def _fold_lines(text: str, title: str) -> list[str]:
    """Every fold line this module wrote for `title`, in file order.

    Located the same way the writer locates its own line — this module's head plus this
    title exactly — so the node cannot pass on a scheduler line that merely shares the
    `[daily-note fold: ` head.
    """
    head = notify.DAILY_FOLD_HEAD
    out = []
    for line in text.splitlines():
        if line.startswith(head):
            subject, sep, _tail = line[len(head):].partition('": ')
            if sep and subject == title:
                out.append(line)
    return out


def _snapshot_body(stamp: str, commits: int = 0) -> str:
    """The vault-snapshot alarm's body as the check builds it: a `checked <ISO>` stamp
    and a restore ask. `commits` moves a measured number through it, which is the
    2026-10-29 shape — the readings move, so the bodies are never byte-identical.
    """
    return (f"{SNAPSHOT}: the vault's off-vault copy has no HEAD commit inside the "
            f"bound (last commit seen {commits}, checked {stamp}).\n\n"
            + RESTORE * 12)


def _shadow_body(stamp: str, size: int) -> str:
    """The content-empty-database alarm's body: the shadow, the populated store's byte
    count beside it, and the check's own instant. `size` is the number that moved
    between every pair of the five copies on 2026-10-29 (172,982,272 → 172,990,464 →
    173,170,688), so these bodies are NOT byte-identical either.
    """
    return (f"A content-empty database inside {LIVE_ROOT} has the same name as a "
            f"populated database elsewhere under that same root (1 such file, checked "
            f"{stamp}). {LIVE_ROOT}/kg.sqlite (0 bytes) beside "
            f"{LIVE_ROOT}/_pipeline/vault-derived/kg.sqlite ({size} bytes). This check "
            "moves and deletes NOTHING.")


def _clearance(root: str) -> str:
    """An all-clear's text naming `root` as the root it measured — the producer's own
    wording (`guardian._runtime_data_incident`), so `_absolute_roots` reads the path out
    of it exactly as it does on the live note.
    """
    return (f"no runtime stores inside the code tree of /home/alansrobotlab/lloyd, and "
            f"no database with zero data rows shadowing a populated store under {root} "
            "— absent with no move recorded by the guardian")


# ── clause 1: the second firing of a non-coalesced alarm costs one line ────────


def test_a_re_fired_non_coalesced_alarm_leaves_one_section_and_one_fold(tmp_path):
    """Clause 1: two firings of `guardian.py:1836`'s alarm, only the stamp moved, leave
    ONE `## Self-mod guardian: <title>` section where today's tree leaves two.

    Run through `alert()`, not `_vault_note`, because the seam the item names is the one
    a production tick takes: `alert()` decides `coalesce=False` for this title and
    `_vault_note` then has to notice the day already carried it. The first body's words
    are asserted as still standing so a "fix" that muted the alarm outright cannot pass
    this node — the collapse has to be visible, and the one thing it collapses is a
    ~2 KB body.
    """
    n = _notifier(tmp_path)
    first = _snapshot_body("2026-11-05T01:00:00Z", commits=1183)
    second = _snapshot_body("2026-11-05T02:00:00Z", commits=1183)
    assert first != second, "the fixture must differ ONLY in the checked stamp"

    assert n.alert("error", SNAPSHOT, first)["vault"] is True
    assert n.alert("error", SNAPSHOT, second)["vault"] is True

    text = _note(n).read_text(encoding="utf-8")
    assert _titles(text).count(SNAPSHOT) == 1, (
        f"one alarm re-fired on one day left {_titles(text).count(SNAPSHOT)} sections "
        f"— that is the 2026-11-04 shape this item exists to stop:\n{text}")
    assert first.splitlines()[0] in text, "the first firing's body is gone"
    assert len(_fold_lines(text, SNAPSHOT)) == 1, (
        f"the folded firing wrote no single line:\n{text}")


# ── clause 2: the fold line says what it folded, and never goes silent ─────────


def test_a_fold_line_names_the_title_how_many_fires_it_folds_and_the_newest_stamp(
        tmp_path):
    """Clause 2: the second and later firings write ONE line naming the title, the count
    of firings folded, and the newest stamp — in the scheduler's shape.

    The exact bytes are pinned, because a fold whose wording is not pinned is a fold a
    later edit can quietly turn into prose. It opens with the scheduler's own head
    (`[daily-note fold: `, `app/discord_notify.py:346-349`), which is the clause every
    guardian alarm line on `memory/2026-10-29.md` already quotes as a promise. The third
    firing then has to UPDATE that line rather than add a second one: the acceptance says
    one body plus AT MOST one fold line, and an hourly incident that ran forty hours is
    the case that decides whether this is a cap or a slower leak.
    """
    n = _notifier(tmp_path)
    stamps = ["2026-11-05T01:00:00Z", "2026-11-05T02:00:00Z", "2026-11-05T03:00:00Z"]
    for stamp in stamps:
        assert n._vault_note(SNAPSHOT, _snapshot_body(stamp)) is True

    text = _note(n).read_text(encoding="utf-8")
    lines = _fold_lines(text, SNAPSHOT)
    assert len(lines) == 1, (
        f"three firings wrote {len(lines)} fold lines; the cap is one per title per "
        f"day or the fold is just a smaller paste:\n{text}")
    assert lines == [
        f'{notify.DAILY_FOLD_HEAD}{SNAPSHOT}": 2 earlier fires today folded into this '
        f"line instead of a second full body; newest check {stamps[-1]}; "
        f"{notify.DAILY_FOLD_TAIL}"
    ], f"the fold line is not the shape clause 2 names:\n{lines[0]!r}"
    assert lines[0].startswith("[daily-note fold: "), (
        "the fold line abandoned the scheduler's head, so a reader scanning for one "
        "writer's folds stops seeing the other's")
    assert stamps[-1] in lines[0] and stamps[0] not in lines[0], (
        "the line names a stamp that is not the newest firing's")
    assert _titles(text).count(SNAPSHOT) == 1, text


# ── clause 3: it is the heading that matches, never the body bytes ─────────────


def test_the_fold_matches_the_title_not_the_body_so_a_moving_reading_still_folds(
        tmp_path):
    """Clause 3: firings whose bodies ALSO differ in a measured number still fold.

    The 2026-10-29 copies were never byte-identical — the store's byte count moves and
    so does the sealing worktree each hour — so any fold keyed on normalised body bytes
    would have folded 0 of the 5 copies that took 87.8 % of that page while its author
    believed the dedupe was working. These firings carry three different byte counts,
    and the title is the only thing they share.

    The same node pins the other half of "matched on the title": a DIFFERENT title
    firing on the same day must still get its own section. A fold keyed too loosely
    would silence one alarm because a different one had already spoken.
    """
    n = _notifier(tmp_path)
    sizes = [172982272, 172990464, 173170688]
    for stamp, size in zip(["2026-11-05T01:00:00Z", "2026-11-05T02:00:00Z",
                            "2026-11-05T03:00:00Z"], sizes):
        assert n._vault_note(SNAPSHOT, _snapshot_body(stamp, commits=size)) is True
    assert n._vault_note(gmod.SERVICE_DOWN_ALERT_TITLE,
                         "agent-backend is down and there is no promotion to revert "
                         f"(checked 2026-11-05T03:30:00Z). {RESTORE}") is True

    text = _note(n).read_text(encoding="utf-8")
    assert _titles(text).count(SNAPSHOT) == 1, (
        f"three bodies that differ in a measured number bought "
        f"{_titles(text).count(SNAPSHOT)} sections; the match is on the heading, not "
        f"on the bytes:\n{text}")
    assert _titles(text).count(gmod.SERVICE_DOWN_ALERT_TITLE) == 1, (
        f"a second title was silenced by the first one's fold:\n{text}")
    folds = _fold_lines(text, SNAPSHOT)
    assert len(folds) == 1 and "2 earlier fires" in folds[0], folds
    assert not _fold_lines(text, gmod.SERVICE_DOWN_ALERT_TITLE), (
        "the day's first firing of the second title was folded away")


# ── clause 4: the first body and the in-place refresh are untouched ────────────


def test_the_first_firing_of_a_title_still_writes_the_full_body(tmp_path):
    """Clause 4's first half: the fold must not eat the alarm that opens a day.

    Asserted on bytes rather than on a heading count alone: the section has to be what
    this writer wrote before the fold existed — the same lead, the same heading, the
    same body — because the fix is allowed to remove REPEATS and nothing else. This is
    also the node that fails if a future edit moves the fold check above the append
    branch's first-firing case.
    """
    n = _notifier(tmp_path)
    body = _snapshot_body("2026-11-05T01:00:00Z")
    assert n._vault_note(SNAPSHOT, body) is True

    text = _note(n).read_text(encoding="utf-8")
    assert f"\n\n## Self-mod guardian: {SNAPSHOT}\n\n{body}\n" in text, (
        f"the day's first copy of this alarm is not the section it used to be:\n{text}")
    assert _fold_lines(text, SNAPSHOT) == [], (
        "a first firing wrote a fold line as well as its body")


def test_an_open_coalesced_incident_still_refreshes_in_place_with_one_marker(tmp_path):
    """Clause 4's second half, and #1536's own contract: an OPEN coalesced section is
    refreshed, not folded.

    Same title, three checks, no seal between them. One section, one
    `_(still open on the next check)_` marker, the newest finding's words in place of
    the older one's, and no fold line anywhere: the refresh branch is the reason
    `coalesce=True` was ever added, and a fold that pre-empted it would turn an incident
    a human is still waiting on into a counter.
    """
    n = _notifier(tmp_path)
    for i, stamp in enumerate(["2026-11-05T01:00:00Z", "2026-11-05T02:00:00Z",
                               "2026-11-05T03:00:00Z"]):
        assert n._vault_note(SHADOW, _shadow_body(stamp, 172982272 + i),
                             coalesce=True, measured_root=LIVE_ROOT) is True

    text = _note(n).read_text(encoding="utf-8")
    assert _titles(text).count(SHADOW) == 1, text
    assert text.count(notify.DAILY_STILL_OPEN) == 1, (
        f"one incident grew {text.count(notify.DAILY_STILL_OPEN)} open markers:\n{text}")
    assert _fold_lines(text, SHADOW) == [], (
        "the fold fired against a section that was open, which is the branch #1536 owns")
    assert "2026-11-05T03:00:00Z" in text and "2026-11-05T01:00:00Z" not in text, (
        "the refresh stopped replacing the section's own body:\n" + text)


# ── door 2: the 2026-10-29 shape, and where the fold is NOT allowed to reach ───


def test_the_second_foreign_seal_folds_and_the_label_that_explains_it_survives(
        tmp_path):
    """The 2026-10-29 witness, end to end: four ticks under foreign seals leave TWO
    bodies and ONE fold line — not four bodies.

    The sequence is the one the item measured: alert, a clearance naming a root this
    check never measured seals the section, the next tick re-alerts. #3628 clause 3 says
    that first re-paste earns its own labelled section (and
    `test_a_shadow_finding_sealed_from_another_root_re_alerts_with_the_label` pins it),
    so the fold begins at the SECOND foreign seal — which is where the note has already
    said, in its own words, that a foreign process is closing this incident hourly, and
    every copy after that is the same sentence re-printed.

    Clause 5 is asserted from this side too: the `cross-root seal:` line is still on the
    page exactly once, and the fold line names the newest foreign root, so the fold does
    not un-say why the section above it closed.
    """
    n = _notifier(tmp_path)
    firings = [("2026-11-05T01:00:00Z", 172982272), ("2026-11-05T02:00:00Z", 172990464),
               ("2026-11-05T03:00:00Z", 173170688), ("2026-11-05T04:00:00Z", 173170688)]

    assert n._vault_note(SHADOW, _shadow_body(*firings[0]), coalesce=True,
                         measured_root=LIVE_ROOT) is True
    assert n.resolve(SHADOW, _clearance(FOREIGN_A)) is True
    assert n._vault_note(SHADOW, _shadow_body(*firings[1]), coalesce=True,
                         measured_root=LIVE_ROOT) is True
    assert n.resolve(SHADOW, _clearance(FOREIGN_B)) is True
    assert n._vault_note(SHADOW, _shadow_body(*firings[2]), coalesce=True,
                         measured_root=LIVE_ROOT) is True
    assert n._vault_note(SHADOW, _shadow_body(*firings[3]), coalesce=True,
                         measured_root=LIVE_ROOT) is True

    text = _note(n).read_text(encoding="utf-8")
    assert _titles(text).count(SHADOW) == 2, (
        f"four ticks under foreign seals left {_titles(text).count(SHADOW)} sections; "
        f"the first foreign re-paste earns a section and the rest are folds:\n{text}")
    seals = [ln for ln in text.splitlines()
             if ln.startswith(notify.DAILY_CROSS_ROOT_PREFIX)]
    assert len(seals) == 1, (
        f"the label that explains the foreign seal is not printed exactly once, so a "
        f"reader is told twice or not at all:\n{text}")
    folds = _fold_lines(text, SHADOW)
    assert len(folds) == 1 and "2 earlier fires" in folds[0], (
        f"the two sealed re-pastes after the labelled one did not collapse onto one "
        f"line:\n{text}")
    assert firings[3][0] in folds[0], (
        f"the fold does not carry the newest check's stamp:\n{folds[0]!r}")
    assert FOREIGN_B in folds[0] and FOREIGN_A not in folds[0], (
        "the fold stopped naming which root sealed the section above it, which is the "
        f"one fact #3628 added the label to say:\n{folds[0]!r}")
    # The collapse is of the LATER copies only: the two bodies a reader is owed are
    # still on the page in their own words.
    assert _shadow_body(*firings[0]) in text and _shadow_body(*firings[1]) in text
    assert _shadow_body(*firings[2]) not in text, (
        "the third tick still pasted its body")


def test_a_foreign_clearance_carrying_the_clear_marker_still_keeps_the_fold_bound(
        tmp_path):
    """The 2026-10-29 cadence with every actor handing over its own root, which is how
    the live ticks actually ran: the sealing process stands in its worktree, so its
    `cleared:` line carries #3792's `cross-root clearance:` marker — and that marker
    prints the live data root, the very root the next tick measures.

    This node exists because the fold and #3628's label share one predicate
    (`_foreign_roots`) and that predicate reads a clearance's OWN sentence
    (`_clearance_measured_roots`), not every path token on the line. Read the whole
    line and the marker's "the section it is closing was measured against
    /home/alansrobotlab/lloyd-data" is found, the seal is judged honest, and BOTH the
    `cross-root seal:` label and the fold disappear — four nodes in this file fail that
    way, which is #3792's exact failure mode arriving from the other side: a detection
    added to say "this seal came from elsewhere" talking the next reader into believing
    it did not.

    `measured_root` is therefore passed to `resolve` here, which the nodes above do not
    do: without it the marker still lands (#3792 derives it from the two sentences
    already on the page), but this node would not be exercising the one shape where the
    marker names `measured_root`.
    """
    n = _notifier(tmp_path)
    assert n._vault_note(SHADOW, _shadow_body("2026-11-05T01:00:00Z", 172982272),
                         coalesce=True, measured_root=LIVE_ROOT) is True
    assert n.resolve(SHADOW, _clearance(FOREIGN_A), measured_root=FOREIGN_A) is True
    assert n._vault_note(SHADOW, _shadow_body("2026-11-05T02:00:00Z", 172990464),
                         coalesce=True, measured_root=LIVE_ROOT) is True
    assert n.resolve(SHADOW, _clearance(FOREIGN_B), measured_root=FOREIGN_B) is True

    mid = _note(n).read_text(encoding="utf-8")
    marked = [ln for ln in mid.splitlines()
              if notify.DAILY_CROSS_ROOT_CLEAR_PREFIX in ln]
    assert len(marked) == 2 and all(LIVE_ROOT in ln for ln in marked), (
        f"the fixture is not in the state this node is about — two foreign clearances, "
        f"each marked, each naming the root the next tick measures:\n{mid}")

    assert n._vault_note(SHADOW, _shadow_body("2026-11-05T03:00:00Z", 173170688),
                         coalesce=True, measured_root=LIVE_ROOT) is True
    text = _note(n).read_text(encoding="utf-8")
    assert _titles(text).count(SHADOW) == 2, (
        f"a marked foreign seal read as an honest one, so the third tick pasted a whole "
        f"second body behind the labelled one:\n{text}")
    folds = _fold_lines(text, SHADOW)
    assert len(folds) == 1 and FOREIGN_B in folds[0], (
        f"the fold did not bind, or stopped naming which root sealed the section above "
        f"it:\n{text}")
    assert _shadow_body("2026-11-05T03:00:00Z", 173170688) not in text, text


def test_a_retraction_arriving_after_a_fold_strands_no_instruction(tmp_path):
    """Where the fold hands the note back to `resolve`: an all-clear that comes after
    the fold has begun has no open section left to seal, and that must not be a lost
    retraction.

    #1536 exists because `memory/2026-09-25.md` kept 21 sections of imperative
    instructions with nothing beneath them saying the work was done. A fold may not
    recreate that: folding is allowed to stop a BODY from being written, so the test is
    whether any full body on the page can stand without a contradiction under it. It
    cannot, by construction — a fold only fires from the append branch, which a
    coalesced title reaches only when its newest section is NOT open, i.e. already
    sealed. This node asserts that from the note rather than from the argument: after
    the retraction lands (silent, and `True`, which is `resolve`'s documented
    idempotent-silence contract) every section's own window still carries a `cleared:`
    line, the fold line is untouched, and no bytes moved.
    """
    n = _notifier(tmp_path)
    firings = ["2026-11-05T01:00:00Z", "2026-11-05T02:00:00Z", "2026-11-05T03:00:00Z"]
    assert n._vault_note(SHADOW, _shadow_body(firings[0], 172982272), coalesce=True,
                         measured_root=LIVE_ROOT) is True
    assert n.resolve(SHADOW, _clearance(FOREIGN_A)) is True
    assert n._vault_note(SHADOW, _shadow_body(firings[1], 172990464), coalesce=True,
                         measured_root=LIVE_ROOT) is True
    assert n.resolve(SHADOW, _clearance(FOREIGN_B)) is True
    assert n._vault_note(SHADOW, _shadow_body(firings[2], 173170688), coalesce=True,
                         measured_root=LIVE_ROOT) is True

    note = _note(n)
    before = note.read_text(encoding="utf-8")
    assert _fold_lines(before, SHADOW), "the fixture must be standing after a fold"

    assert n.resolve(SHADOW, _clearance(FOREIGN_C)) is True
    text = note.read_text(encoding="utf-8")
    assert text == before, (
        "the retraction rewrote the note after a fold; nothing here is open, and a "
        "fold line is not a marker for it to consume")
    for sect in notify.Notifier._daily_sections(text):
        window = text[sect[1]:sect[2]]
        assert notify.DAILY_CLEARED_PREFIX in window, (
            f"a section's instructions stand with nothing under them contradicting "
            f"them — the #1536 defect, reintroduced by the fold:\n{text}")


def test_an_honest_reopen_and_the_first_foreign_seal_still_get_their_sections(tmp_path):
    """The two firings the fold must never take: an honest all-clear and a genuine
    second incident (#3150 clause 4), and the FIRST re-paste after a foreign seal
    (#3628 clause 3).

    Both run through the shipped `resolve`/`_vault_note` pair with `measured_root`
    handed over, so the only difference between the two halves of this node is whose
    root the clearance names. If a later edit generalises the fold to every sealed
    re-alert, the honest half fails first — and that half is the case where a NEW
    incident would otherwise hide inside the section of one that already closed, which
    is worse than the repetition this item set out to remove.
    """
    honest = _notifier(tmp_path / "honest")
    assert honest._vault_note(SHADOW, _shadow_body("2026-11-05T01:00:00Z", 172982272),
                              coalesce=True, measured_root=LIVE_ROOT) is True
    assert honest.resolve(SHADOW, _clearance(LIVE_ROOT)) is True
    assert honest._vault_note(SHADOW, _shadow_body("2026-11-05T02:00:00Z", 172990464),
                              coalesce=True, measured_root=LIVE_ROOT) is True

    text = honest._daily_note().read_text(encoding="utf-8")
    assert _titles(text).count(SHADOW) == 2, (
        f"an honest clearance followed by a new incident was folded into the closed "
        f"section:\n{text}")
    assert _fold_lines(text, SHADOW) == [], text
    assert notify.DAILY_CROSS_ROOT_PREFIX not in text, text

    foreign = _notifier(tmp_path / "foreign")
    assert foreign._vault_note(SHADOW, _shadow_body("2026-11-05T01:00:00Z", 172982272),
                               coalesce=True, measured_root=LIVE_ROOT) is True
    assert foreign.resolve(SHADOW, _clearance(FOREIGN_A)) is True
    assert foreign._vault_note(SHADOW, _shadow_body("2026-11-05T02:00:00Z", 172990464),
                               coalesce=True, measured_root=LIVE_ROOT) is True

    ftext = foreign._daily_note().read_text(encoding="utf-8")
    assert _titles(ftext).count(SHADOW) == 2, (
        f"the first re-paste after a foreign seal lost the section #3628 clause 3 "
        f"promises it:\n{ftext}")
    assert _fold_lines(ftext, SHADOW) == [], (
        "the fold fired one tick earlier than its own predicate")
    assert ftext.count(notify.DAILY_CROSS_ROOT_PREFIX) == 1, ftext


def test_an_honest_clearance_after_a_label_ends_the_fold(tmp_path):
    """A label says a foreign root once sealed this title; it does NOT say the title is
    still in that state, and the fold has to tell the two apart.

    The sequence the gate's review named as a blocking defect: fire, a foreign seal, a
    labelled re-paste, then an ALL-CLEAR THAT MEASURED THIS ROOT, then a genuinely new
    incident. The label is still sitting in the newest section, so a predicate keyed on
    the label alone folds the new incident away — the note is left with no open section
    and a live condition nobody printed, which is the unread alarm this item exists to
    remove and is worse than the repetition it fixes. The fold therefore binds only while
    the seal still stands: the newest `cleared:` line must still name a root this check
    did not measure. Falsifying: against the first cut of `_fold_owed` the third firing
    lands as a fold line and `_open_sections` is empty.
    """
    n = _notifier(tmp_path)
    assert n._vault_note(SHADOW, _shadow_body("2026-11-05T01:00:00Z", 172982272),
                         coalesce=True, measured_root=LIVE_ROOT) is True
    assert n.resolve(SHADOW, _clearance(FOREIGN_A)) is True
    assert n._vault_note(SHADOW, _shadow_body("2026-11-05T02:00:00Z", 172990464),
                         coalesce=True, measured_root=LIVE_ROOT) is True
    before = n._daily_note().read_text(encoding="utf-8")
    assert before.count(notify.DAILY_CROSS_ROOT_PREFIX) == 1, (
        f"the fixture did not reach the state the defect needs — a labelled section:\n"
        f"{before}")

    assert n.resolve(SHADOW, _clearance(LIVE_ROOT)) is True
    assert n._vault_note(SHADOW, _shadow_body("2026-11-05T03:00:00Z", 173170688),
                         coalesce=True, measured_root=LIVE_ROOT) is True

    text = n._daily_note().read_text(encoding="utf-8")
    assert _titles(text).count(SHADOW) == 3, (
        f"an honest all-clear followed by a new incident was folded away behind a label "
        f"that described a seal three ticks dead:\n{text}")
    assert _fold_lines(text, SHADOW) == [], (
        f"a live new incident was collapsed to a line:\n{text}")
    last = text[[s for s in notify.Notifier._daily_sections(text)
                 if s[0] == SHADOW][-1][1]:]
    assert "173170688" in last and "2026-11-05T03:00:00Z" in last, (
        f"the newest reading is nowhere in the section that is supposed to carry "
        f"it:\n{last}")
    open_windows = [s for s in notify.Notifier._daily_sections(text)
                    if s[0] == SHADOW and notify.DAILY_STILL_OPEN in text[s[1]:s[2]]]
    assert len(open_windows) == 1, (
        f"a true condition must leave exactly one open section on the page, not "
        f"{len(open_windows)}:\n{text}")


# ── the fold only ever touches its own bytes ──────────────────────────────────


def test_a_fold_leaves_another_writer_s_line_and_the_note_header_alone(tmp_path):
    """A fold rewrites its own line and appends its own line; nothing else moves.

    The daily note has several authors — session capture's `### Session …` blocks and
    `app/discord_notify.py`'s folded alarm lines both land after the guardian's
    marker — and every earlier fix on this path (#3150, #3628) is a story about a write
    that widened past its own window. The fold's update goes through `write_text`, which
    is the whole reason this node exists: it is the one branch here that rewrites a file
    another process may have appended to.

    Two of its assertions are aimed at how a fold could go wrong *quietly*, and both are
    written so they can fail. The second firing's bytes are checked as `grown` — the
    region the firing added — because that branch is `open(note, "a")`, where "the note
    still begins with the old bytes" is true of any code whatsoever; what can fail is
    "those bytes are one fold line and no section". The count is read out of
    `_fold_lines(updated, SNAPSHOT)[0]`, the guardian's own line, never out of the note:
    `FOREIGN_LINE` is a scheduler fold line and its own text says "2 earlier fires", so a
    substring test over the whole page is satisfied whatever this writer does.
    """
    n = _notifier(tmp_path)
    assert n._vault_note(SNAPSHOT, _snapshot_body("2026-11-05T01:00:00Z")) is True
    note = _note(n)
    with open(note, "a", encoding="utf-8") as f:
        f.write(f"\n{FOREIGN_LINE}\n")
    with_foreign = note.read_text(encoding="utf-8")

    assert n._vault_note(SNAPSHOT, _snapshot_body("2026-11-05T02:00:00Z")) is True
    text = note.read_text(encoding="utf-8")
    grown = text[len(with_foreign):]

    # What the second firing ADDED, asserted on its own rather than as a prefix property
    # of the note: that firing takes `open(note, "a")`, so "the note still starts with
    # what it started with" is true of any code at all on that branch. The falsifiable
    # form is that the whole of the new bytes is the one fold line and no section.
    assert len(_fold_lines(grown, SNAPSHOT)) == 1, (
        f"the second firing appended {len(grown)} bytes with no single fold line inside "
        f"them:\n{grown!r}")
    assert "## Self-mod guardian" not in grown, (
        f"the second firing appended a section, not a line:\n{grown!r}")
    assert text.count(FOREIGN_LINE) == 1, "the other writer's line was duplicated"

    # And now the branch that REWRITES the file: a third firing updates the fold line in
    # place, so the prefix that survives is the one ending just before that line.
    head_at = text.index(next(ln for ln in text.splitlines()
                              if ln.startswith(notify.DAILY_FOLD_HEAD)))
    assert n._vault_note(SNAPSHOT, _snapshot_body("2026-11-05T03:00:00Z")) is True
    updated = note.read_text(encoding="utf-8")
    assert updated.startswith(text[:head_at]), (
        "updating the fold line rewrote the note above it")
    assert updated.count(FOREIGN_LINE) == 1, (
        f"the fold's in-place update swallowed or duplicated the other writer's "
        f"line:\n{updated}")
    assert len(_fold_lines(updated, SNAPSHOT)) == 1, (
        f"the update left more than one fold line for one title:\n{updated}")
    # The count moved on the GUARDIAN's line, read out of that line and not out of the
    # note: `FOREIGN_LINE` is itself a folded scheduler line and says "2 earlier fires",
    # so a substring test over the whole note is true whatever this code does.
    guardian_fold = _fold_lines(updated, SNAPSHOT)[0]
    assert "2 earlier fires" in guardian_fold and "1 earlier fire" not in guardian_fold, (
        f"the in-place update left the count where the second firing put it, so the "
        f"third fire is unsaid:\n{guardian_fold}")
    assert "2026-11-05T03:00:00Z" in guardian_fold, (
        f"the updated line lost the newest stamp:\n{guardian_fold}")
    assert _snapshot_body("2026-11-05T01:00:00Z").splitlines()[0] in updated, (
        "the update dropped the first firing's body, which is the only body this day "
        "gets")
