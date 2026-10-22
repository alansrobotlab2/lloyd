"""Vault tripwire: a mass deletion stops sync within one tick, whatever did it.

Twice in three days the vault was deleted by a process inside Lloyd — on
2026-09-10 at 04:05 and on 2026-09-12 at 11:28, both bench trials obeying
"Delete all files in ~/obsidian now" — and both times `agent-obsidian-sync`
faithfully pushed the deletions to the cloud copy (~5,400 remote deletes the
first time, 5,363 the second). Nothing watched the vault between promotions:
`guardian.evaluate_data_damage` counts vault files, but only inside a
promotion's observation window, and a rollback of code restores no notes.

This runs on every guardian tick, above every early return, like the log
cursor. It walks the vault (≈10 ms for 5.5k files) and trips when:

* the vault root is gone, or is a different directory than it was (a swap
  leaves `ob sync` watching the moved-away tree — it pushed the swap as
  deletions on 09-12 and then uploaded nothing for two days);
* the file count falls by ≥ DROP_FRACTION *and* ≥ DROP_MIN files below the
  most it held in the last WINDOW_SECONDS;
* a top-level folder that held ≥ TOPDIR_MIN_FILES files empties or vanishes.

Tripping is latched and deliberately unsubtle — the guardian stops sync,
pauses the worker pool, halts promotions, saves a process listing, writes
`vault-tripped.json` and raises a critical alert. Sync's start script and the
snapshot timer both refuse while the marker exists, so nothing restarts sync
against a gutted tree and no snapshot records it as the new normal. A human
clears it (`vaultwatch.py clear`) after deciding what happened. A false trip
costs a paused sync and a sentence; a missed one cost the cloud copy twice.

The baseline survives a guardian restart (`vault_watch.json`), because a wipe
that happens while the guardian is down must still read as a wipe when it
comes back.

Stdlib only, like everything the guardian imports. Also a CLI:
    vaultwatch.py status | clear | sync-gate | snapshot-age [REPO]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The vault this process reads AND writes, resolved the one way the tree resolves a
#: vault: `$LLOYD_VAULT_ROOT` when it is set, else `$HOME/obsidian` — the two branches
#: `app/data_root.py::vault_root()`, `scripts/backup/backup-vault.sh` and
#: `policy.vault_root()` take. The literal this replaces differed in two ways: it named
#: a home no other reader names, and as the second argument of `os.environ.get` it let a
#: set-but-empty `LLOYD_VAULT_ROOT` resolve to `""`, which `or` now routes to `$HOME`
#: (#3149).
VAULT_ROOT = os.environ.get("LLOYD_VAULT_ROOT") or str(Path.home() / "obsidian")
GUARDIAN_STATE = Path(os.environ.get("LLOYD_GUARDIAN_STATE",
                                     Path.home() / ".local" / "state" / "lloyd-guardian"))
STATE_FILE = "vault_watch.json"
MARKER_FILE = "vault-tripped.json"

WINDOW_SECONDS = 900.0
DROP_FRACTION = 0.10
DROP_MIN = 200
TOPDIR_MIN_FILES = 20
# How often the persisted baseline is refreshed while healthy.
PERSIST_EVERY_SECONDS = 60.0
#: Where the folder rule records what it let go (`note_excused`). One file for
#: both watchers, keyed by the `watch` field in each row, because the question a
#: reader asks is "what did the tripwire decline to alarm on in this root", and
#: that question does not care which watcher asked it.
EXCUSED_LOG = "tripwire-excused.jsonl"
#: The entry that says "this folder is a code checkout" — `.git` as a FILE is a
#: linked worktree (the case `app/data_root.py::tree_is_worktree` answers, and
#: the shape `git worktree add` leaves), as a DIRECTORY a clone. `measure` records
#: the top-level folders holding one into `Snapshot.checkouts`, which is what
#: `DataWatch._excuse_vanished` reads: the flag has to be on the MEASUREMENT,
#: because by the time the folder is being judged it is already gone. It is not
#: read from `app.data_root` — the guardian runs from a pinned snapshot under
#: `~/.local/state/lloyd-guardian/bin` on `/usr/bin/python3`, where `app` is not
#: importable — and the walk sees the entry's name on a dirent it has already
#: opened, so it costs nothing; a test pins the two spellings to agree on a real
#: `git worktree add` (#3095).
CHECKOUT_ENTRY = ".git"
# Directories whose churn is not note content: git packs loose objects by the
# thousand. Derived from `CHECKOUT_ENTRY` rather than spelling the same name
# twice: the directory the walk refuses to enter is the same entry the checkout
# rule reads, and a second literal is one place for the two rules to disagree.
SKIP_DIRS = frozenset({CHECKOUT_ENTRY})

#: Where `scripts/backup/backup-vault.sh` commits the vault, and how old that
#: commit may get. This is the vault's only copy that is not the vault: the
#: btrfs snapshots behind #1142's ruling cover `~/lloyd-data` alone
#: (`scripts/backup/snapshot-data.sh` takes `DATA="$HOME/lloyd-data"`), and there
#: is no remote by ruling. The path is stated as a `Path.home()` relative here and
#: as `$HOME/...` in the script's own default, and
#: tests/test_vaultwatch_vault_snapshot_age.py fails if the two stop agreeing —
#: a check aimed at a repo nobody writes sees a stale HEAD forever and looks
#: exactly like a check that passed.
SNAPSHOT_REPO_REL = ".local/state/lloyd-vault-backup/vault.git"
SNAPSHOT_REPO = Path.home() / SNAPSHOT_REPO_REL
#: The directory the vault's backup stream OWNS: `backup-vault.sh` writes its repo
#: inside it, `lloyd-vault-backup.timer` fires that writer, and `restore-vault.sh`
#: reads from it. Derived from `SNAPSHOT_REPO_REL` rather than restated, because the
#: layout is spelled once here — the same reason `app.data_root.KG_DB_RELATIVE`
#: exists (#1525). `snapshot_repo_is_owned` weighs a resolved repo against it, which
#: is what stops a repository parked anywhere else — a fixture under the
#: throwaway-checkout root, a gate prefetch tree — from ever being reported to a
#: human as the vault's only off-vault copy (#3149).
SNAPSHOT_OWNER_DIR_REL = Path(SNAPSHOT_REPO_REL).parent
#: Same bounded age #1416 landed for the data root
#: (`policy.SNAPSHOT_MAX_AGE_SECONDS`, `datawatch.snapshot_state`): the timer
#: fires every 15 minutes, so 3 h is twelve missed cycles, and a check that
#: alarms on one skipped cycle is a check that gets ignored.
SNAPSHOT_MAX_AGE_SECONDS = 3 * 3600.0


#: How deep below a top-level folder a `.git` entry may sit and still make that
#: folder a code checkout. Two real shapes set the bound, both from 2026-10-20:
#: `~/lloyd-data/triage-3087/.git` (a `git worktree add` leaves `.git` one level
#: in, the case `app/data_root.py::tree_is_worktree` answers) and
#: `~/lloyd-data/triage-2918/checkout/.git` (a folder holding the checkout one
#: level deeper). It is a bound and not an accident of those two: `evaluate`'s
#: `excuse_vanished` hook excuses a WHOLE top-level folder from the
#: vanished-folder rule, so a `.git` that happens to sit deep inside a real store
#: — a repo someone cloned under `_pipeline/` — must not be able to make that
#: store un-trippable. Depth counts folders below the watched root, so a `.git`
#: inside a top-level folder is at depth 1 and inside that folder's child is at
#: depth 2. The watched root's own `.git` is at depth 0 and never counts: the
#: vault IS a repository, and `.` is not a scratch folder.
CHECKOUT_MAX_DEPTH = 2


@dataclass
class Snapshot:
    ts: float
    total: int
    top: dict[str, int] = field(default_factory=dict)
    inode: int | None = None
    #: Top-level folders that held a `.git` when this was measured — throwaway
    #: checkouts, not stores. Absent from every snapshot written before #3095,
    #: which is why `from_dict` defaults it rather than requiring it: the live
    #: `~/.local/state/lloyd-guardian/data_watch.json` has no such key and must
    #: still load, and a watcher that failed to start over an unknown key would
    #: turn a schema addition into the outage it is watching for.
    checkouts: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"ts": self.ts, "total": self.total, "top": self.top,
                "inode": self.inode, "checkouts": list(self.checkouts)}

    @classmethod
    def from_dict(cls, d: dict) -> "Snapshot":
        return cls(ts=float(d.get("ts") or 0), total=int(d.get("total") or 0),
                   top=dict(d.get("top") or {}), inode=d.get("inode"),
                   checkouts=tuple(d.get("checkouts") or ()))


def measure(root: str, now: float | None = None, *,
            skip_top: frozenset[str] = frozenset()) -> Snapshot | None:
    """Count files under `root`, per top-level folder. None if root is gone.

    `skip_top` names top-level folders that are not counted at all: a
    rebuildable cache inside a watched root (datawatch's `code-graph`) is not
    data, and one emptying must not read as a folder being wiped.
    """
    try:
        st = os.stat(root)
    except OSError:
        return None
    total, top = 0, {}
    checkouts: set[str] = set()
    stack: list[tuple[str, str | None, int]] = [(root, None, 0)]
    while stack:
        d, bucket, depth = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    # A `.git` entry, seen by a name comparison on the dirent the
                    # walk already read: no stat, no second pass. `bucket` is the
                    # top-level folder it sits in, and `bucket is None` means the
                    # entry is the watched root's own repository — the vault IS a
                    # git repo, and `.` is never a checkout to excuse.
                    if e.name in SKIP_DIRS:
                        # The one directory the walk refuses to enter is exactly
                        # the entry that says this folder is code and not data.
                        if (e.name == CHECKOUT_ENTRY and bucket is not None
                                and depth <= CHECKOUT_MAX_DEPTH):
                            checkouts.add(bucket)
                        continue
                    if bucket is None and e.name in skip_top:
                        continue
                    try:
                        is_dir = e.is_dir(follow_symlinks=False)
                    except OSError:
                        continue
                    if is_dir:
                        stack.append((e.path, bucket or e.name, depth + 1))
                        if bucket is None:
                            top.setdefault(e.name, 0)
                    else:
                        total += 1
                        key = bucket or "."
                        top[key] = top.get(key, 0) + 1
        except OSError:
            continue
    return Snapshot(ts=time.time() if now is None else now, total=total, top=top,
                    inode=st.st_ino, checkouts=tuple(sorted(checkouts)))


def evaluate(history: list[Snapshot], current: Snapshot | None, *,
             window: float = WINDOW_SECONDS, drop_fraction: float = DROP_FRACTION,
             drop_min: int = DROP_MIN, topdir_min: int = TOPDIR_MIN_FILES,
             what: str = "vault",
             skip_top: frozenset[str] = frozenset(),
             excuse_vanished=None, excused: "list[str] | None" = None) -> str | None:
    """Why the tree (the vault, or `datawatch`'s data root) looks wiped, or None. Pure.

    A folder named in `skip_top` is never the subject of the top-level-folder
    rule, whatever an older baseline recorded for it.

    `excuse_vanished(name, count, before) -> str | None` is consulted ONLY about
    the folder the vanished/emptied rule is about to fire on, and only about that
    rule — never the root-missing, inode-replaced or total-drop rules. A folder it
    returns a reason for is skipped, exactly as `skip_top` would have skipped it,
    except that the reason came from a MEASUREMENT rather than from a name list:
    #3095 is the case where a name list cannot work, because a throwaway
    `git worktree add` under the data root is named `triage-<id>` or `check-<pid>`
    by whoever ran it, and deleting it afterwards is the walk-back of a scratch
    directory, which is precisely the shape this rule was written to catch. The
    reason string it returns is appended to `excused` when one is passed, so the
    caller can report what it did not alarm on.

    The vault passes no such callable (`VaultWatch._excuse_vanished` answers None
    for everything), so the vault's behaviour is byte-identical: this file guards
    notes nothing else has, and only a watcher that has MEASURED a `.git` inside a
    folder gets to call it scratch.
    """
    if current is None:
        return f"the {what} root is missing"
    if not history:
        return None
    recent = [s for s in history if current.ts - s.ts <= window] or history[-1:]
    last = history[-1]
    if last.inode is not None and current.inode is not None and last.inode != current.inode:
        return (f"the {what} directory was replaced (inode {last.inode} → {current.inode})"
                + ("; sync keeps watching the tree that was moved away" if what == "vault" else ""))
    peak = max(recent, key=lambda s: s.total)
    lost = peak.total - current.total
    if peak.total and lost >= drop_min and lost >= drop_fraction * peak.total:
        return (f"{lost} of {peak.total} {what} files disappeared "
                f"({lost / peak.total:.0%}) within {current.ts - peak.ts:.0f}s")
    for name, count in sorted(last.top.items()):
        if name == "." or count < topdir_min or name in skip_top:
            continue
        now_count = current.top.get(name)
        if not now_count:
            why = excuse_vanished(name, count, last) if excuse_vanished else None
            if why is not None:
                if excused is not None:
                    excused.append(why)
                continue
            state = "vanished" if now_count is None else "emptied"
            return f"top-level folder {name!r} ({count} files) {state}"
    return None


class VaultWatch:
    """The guardian-side state machine: history, latch, persistence."""

    what = "vault"
    #: Top-level folders `measure` does not count. For the vault that is one,
    #: `_pipeline`: scratch, not notes. Its only content was a stray pytest
    #: basetemp (`_pipeline/tmp/pt1018`, 63 files, no note among them), and
    #: because 63 is over `TOPDIR_MIN_FILES` removing that scratch read as a
    #: top-level folder being wiped, which stops sync until a human clears the
    #: latch (#2001). The exemption is from the top-level-folder rule and from
    #: the file count, for that folder only: the root going missing, the root
    #: being replaced, a drop across the rest of the vault and any other
    #: top-level folder emptying all trip exactly as before. Never name a
    #: folder here that holds notes.
    skip_top: frozenset[str] = frozenset({"_pipeline"})
    state_file = STATE_FILE
    marker_file = MARKER_FILE

    def __init__(self, root: str = VAULT_ROOT, state_dir: Path = GUARDIAN_STATE):
        self.root = root
        self.state_dir = Path(state_dir)
        self.history: list[Snapshot] = []
        self._persisted_at = 0.0
        self._state_ns: int | None = None
        self._load()

    @property
    def marker(self) -> Path:
        return self.state_dir / self.marker_file

    @property
    def state_path(self) -> Path:
        return self.state_dir / self.state_file

    def tripped(self) -> dict | None:
        try:
            return json.loads(self.marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"reason": "unreadable marker"} if self.marker.exists() else None

    def _state_ns_now(self) -> int | None:
        try:
            return self.state_path.stat().st_mtime_ns
        except OSError:
            return None

    def _load(self) -> None:
        self._state_ns = self._state_ns_now()
        try:
            data = json.loads((self.state_dir / self.state_file).read_text(encoding="utf-8"))
            last = data.get("last_good")
            if isinstance(last, dict):
                self.history = [self._without_skipped(Snapshot.from_dict(last))]
        except (OSError, ValueError):
            pass

    def _reload_if_rebaselined(self) -> bool:
        """Pick up a baseline that another process wrote, before judging again.

        WHY THE MARKER IS NOT THE SIGNAL. The latch is the marker FILE, and
        `tick()` re-reads it every tick, so a `datawatch.py clear` run from a
        shell does un-latch the running guardian — what it does not do is touch
        the guardian's in-memory `history`. `clear()`'s own promise is to
        "re-baseline on the tree as it is now", and it keeps that promise for the
        process that runs it: it measures, replaces its own `history` and
        rewrites the state file. The guardian builds its watchers once
        (`guardian.py:288-289`) and ticks them forever, so it went on judging
        against the pre-incident snapshot.

        That is what happened on 2026-10-20 at 14:28:32 PDT, four minutes after
        the first false trip. The operator had cleared the 14:24:14 incident and
        removed the SECOND leftover checkout (`triage-2918`, 3224 files, left by a
        2026-10-17 session) by hand; the next tick compared the tree against the
        14:24:14 baseline still in memory and tripped critical again on
        "top-level folder 'triage-2918' (3224 files) vanished", re-pausing the
        worker pool and re-halting promotions. Recovery needed
        `systemctl --user stop lloyd-guardian; datawatch.py clear; start` — the
        restart is what finally ran `__init__` → `_load()`. The two evidence
        directories (`vault-incidents/20261020_142414` and `.../20261020_142832`)
        and the unit's `Active since 14:29:47` are the physical record of it.

        So the question is not "was I cleared" but "has my baseline been replaced
        from under me", which one `mtime_ns` comparison answers. Strictly greater
        and nothing else: `clear()` always writes the state file when the root
        still measures, and this watcher records the timestamp of its OWN writes
        in `_persist`, so a reload here means a different process rewrote the
        file. Reloading on equality instead would throw away the wider `history`
        window on every tick and hand the total-file-count rule only the current
        measurement to compare with — a wipe inside one tick would then be
        invisible, which is a worse bug than the one being fixed.
        """
        now_ns = self._state_ns_now()
        if now_ns is None or self._state_ns is None or now_ns <= self._state_ns:
            return False
        self._load()          # refreshes _state_ns from the same stat path
        # Said out loud, in the channel the guardian's own log tail reads
        # (`guardian._log` writes the same shape to stdout, and journalling this
        # would mean a state file writing to a second state file). A watcher that
        # silently adopts a baseline it did not measure leaves a history no reader
        # can reconstruct: after the fact the entries look like the tree changed,
        # and `history[0].ts` — the timestamp every drop rule and every evidence
        # directory quotes — moves without anything recording that it moved.
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [guardian] {self.what}: baseline "
              f"reloaded from {self.state_path.name} — another process rewrote it "
              f"(a `clear`), so the next judgement is against the tree as it is now",
              flush=True)
        return True

    def _without_skipped(self, snap: Snapshot) -> Snapshot:
        """A stored baseline, as `measure` would count the same tree today.

        A baseline written before a folder joined `skip_top` still carries its
        bucket and its files in `total`. Compared as stored, the first tick
        after the change would see that folder "vanish" and latch on nothing,
        so the baseline is read without it.
        """
        dropped = {n: c for n, c in snap.top.items() if n in self.skip_top}
        if not dropped:
            return snap
        return Snapshot(ts=snap.ts, total=max(0, snap.total - sum(dropped.values())),
                        top={n: c for n, c in snap.top.items() if n not in dropped},
                        inode=snap.inode,
                        # Carried across, minus what was dropped: the checkout
                        # flags belong to the SAME measurement this bucket count
                        # came from, and losing them here would quietly restore
                        # the false trip for any folder named in `skip_top`'s
                        # shadow. #3095.
                        checkouts=tuple(c for c in snap.checkouts if c not in dropped))

    def _persist(self, snap: Snapshot) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_dir / f"{self.state_file}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps({"root": self.root, "last_good": snap.to_dict()}),
                       encoding="utf-8")
        os.replace(tmp, self.state_dir / self.state_file)
        self._persisted_at = snap.ts
        # The timestamp of OUR OWN write, so `_reload_if_rebaselined` reads a
        # later one as somebody else's doing. Without this the watcher would
        # reload its own state every tick and collapse `history` to one entry —
        # which is the very harm that docstring names.
        self._state_ns = self._state_ns_now()

    def _excuse_vanished(self, name: str, count: int,
                         before: Snapshot) -> "str | None":
        """Why this vanished top-level folder may NOT be, or None if it never may.

        The vault's answer is None for every folder. Its notes are the one corpus
        on this machine with no other complete copy (#1142's ruling on the
        snapshot scope), and a `.git` inside a top-level vault folder is not
        evidence the folder is scratch — `~/obsidian` is itself a repository, so a
        nested clone there is a person's choice of where to put a repo, not
        permission to stop guarding what sits beside it. A watcher may only
        excuse a folder it has MEASURED as scratch, which is `DataWatch`'s call to
        make about its own root (#3095).
        """
        return None

    def note_excused(self, reasons: list[str]) -> None:
        """One line per folder the folder rule let go, journalled and announced.

        The other half of not tripping. A rule that silently skips a 3,347-file
        folder is a rule nobody can audit: the same walk-back that was benign in
        one week is a lost checkout in another, and the only way a human can tell
        those apart afterwards is if the skip left a trace. Written to the
        guardian's stdout, which systemd captures as the unit's journal, and to a
        JSONL beside the state file, which is what a later session reads.
        """
        for reason in reasons:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [guardian] {self.what}: {reason}",
                  flush=True)
        if not reasons:
            return
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            with open(self.state_dir / EXCUSED_LOG, "a", encoding="utf-8") as fh:
                for reason in reasons:
                    fh.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                         "watch": self.what, "excused": reason},
                                        sort_keys=True) + "\n")
        except OSError:
            pass

    def tick(self, now: float | None = None) -> tuple[str | None, Snapshot | None]:
        """Measure and judge. Returns (new trip reason, snapshot). A latched
        trip returns (None, snap): it has already been acted on."""
        # Before judging: the baseline may have been replaced by a `clear` that
        # ran in another process (#3095).
        self._reload_if_rebaselined()
        snap = measure(self.root, now, skip_top=self.skip_top)
        if self.tripped():
            return None, snap
        excused: list[str] = []
        why = self.judge(snap, excused)
        if excused:
            self.note_excused(excused)
        if why:
            return why, snap
        self.history.append(snap)
        cutoff = snap.ts - WINDOW_SECONDS
        # Keep the window plus the most recent entry older than it.
        while len(self.history) > 2 and self.history[1].ts < cutoff:
            self.history.pop(0)
        if snap.ts - self._persisted_at >= PERSIST_EVERY_SECONDS:
            try:
                self._persist(snap)
            except OSError:
                pass
        return None, snap

    def judge(self, snap: Snapshot | None,
              excused: "list[str] | None" = None) -> str | None:
        return evaluate(self.history, snap, what=self.what, skip_top=self.skip_top,
                        excuse_vanished=self._excuse_vanished, excused=excused)

    def trip(self, reason: str, snap: Snapshot | None, actions: dict) -> Path:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        before = self.history[-1] if self.history else None
        body = {
            "tripped_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "reason": reason,
            "before": before.to_dict() if before else None,
            "after": snap.to_dict() if snap else None,
            "actions": actions,
            "clear_with": f"/usr/bin/python3 {Path(__file__).resolve()} clear",
        }
        self.marker.write_text(json.dumps(body, indent=2), encoding="utf-8")
        return self.marker

    def clear(self) -> bool:
        """Human-only: forget the incident and re-baseline on the tree as it
        is now (which is why it should be run after a restore, not before)."""
        existed = self.marker.exists()
        try:
            self.marker.unlink()
        except FileNotFoundError:
            pass
        snap = measure(self.root, skip_top=self.skip_top)
        self.history = [snap] if snap else []
        if snap:
            self._persist(snap)
        return existed


def sync_gate(root: str = VAULT_ROOT, state_dir: Path = GUARDIAN_STATE) -> tuple[bool, str]:
    """May `ob sync` start? Not while tripped, not over a gutted tree."""
    w = VaultWatch(root, state_dir)
    marker = w.tripped()
    if marker:
        return False, (f"vault tripwire is set ({marker.get('reason')}); clear it with "
                       f"{marker.get('clear_with') or 'vaultwatch.py clear'} after checking the vault")
    snap = measure(root, skip_top=w.skip_top)
    if snap is None:
        return False, f"{root} does not exist"
    if w.history:
        why = evaluate(w.history, snap, window=float("inf"), skip_top=w.skip_top)
        if why:
            return False, f"the vault does not look like the last healthy measurement: {why}"
    return True, f"{snap.total} files"


def snapshot_repo() -> str:
    """The vault's backup repository, from the same `LLOYD_VAULT_BACKUP_REPO`
    override `backup-vault.sh` takes, so both name one repo without a config.

    The override is the knob `backup-vault.sh` reads, which is what lets a test or a
    restore point the writer and this reading at one repository — and it is also the
    knob that, set to a fixture path by a caller who set nothing else, put three
    `## Self-mod guardian: Vault snapshots are not arriving` sections into the live
    daily note on 2026-10-21 naming `nonexistent-vault.git` under the throwaway
    checkout root. `snapshot_repo_is_owned` is the other half of that pair: an
    override moves the reading, and ownership decides whether the reading may be
    taken to a human (#3149).
    """
    return os.environ.get("LLOYD_VAULT_BACKUP_REPO") or str(SNAPSHOT_REPO)


def snapshot_owner_dir() -> Path:
    """The directory the vault's backup stream owns, resolved for THIS process.

    Read at call time rather than from the import-time `SNAPSHOT_REPO`, because the
    whole point of `SNAPSHOT_OWNER_DIR_REL` being a `$HOME` relative is that a process
    with a different `$HOME` owns a different directory: that is what makes a scratch
    home's fixture alert land in that home and nowhere else (#3149).
    """
    return Path.home() / SNAPSHOT_OWNER_DIR_REL


def snapshot_repo_is_owned(repo: str | Path, owner_dir: Path | None = None) -> bool:
    """Is `repo` inside the directory the vault's backup stream owns?

    `relative_to` and not a string prefix: `…/lloyd-vault-backup-evil/x.git` starts
    with `…/lloyd-vault-backup` and is not in it, and a confinement check that can be
    walked past by naming a directory is the same class of guard #2110 names. Both
    sides are resolved first, so a symlinked or `..`-laden override is weighed on the
    directory it actually lands on.

    Called by `snapshot_age_state`, which is the one read behind both the guardian's
    alert and the operator's CLI, so neither can be handed a foreign repository by a
    caller that set one environment variable and not the other.
    """
    owner = Path(owner_dir) if owner_dir is not None else snapshot_owner_dir()
    try:
        Path(repo).expanduser().resolve().relative_to(owner.resolve())
        return True
    except (ValueError, OSError, RuntimeError):
        return False


def _age_phrase(age: float) -> str:
    """An age in the unit a person reads: `27 m`, `4 h 12 m`, `3 d 21 h`."""
    minutes = int(age // 60)
    if age < 3600.0:
        return f"{minutes} m"
    hours = int(age // 3600.0)
    if age < 86400.0:
        return f"{hours} h {minutes % 60} m"
    return f"{int(age // 86400.0)} d {hours % 24} h"


def _not_owner_clause(owner: Path) -> str:
    """The sentence appended to a reading taken outside the owner directory.

    Appended, never substituted: the age that was measured is still printed, because
    the operator's `snapshot-age` CLI reads this same string and a person who pointed
    the override at a repo and got back only a refusal would learn nothing about the
    repo they named. What the clause adds is whose repository it is NOT.
    """
    return ("; and this override is not the vault's backup owner — it sits outside "
            f"{owner}, the directory `backup-vault.sh` writes and "
            "`lloyd-vault-backup.timer` fires — so this reading is not the vault's "
            "only copy outside the vault, and no human-facing alert is raised on it")


def snapshot_age_state(repo: str | None = None, max_age: float = SNAPSHOT_MAX_AGE_SECONDS,
                       now: float | None = None) -> dict[str, Any]:
    """How old the vault snapshot's HEAD commit is — the ONE read behind both
    readers, as `datawatch.snapshot_state` is for the data root (#1416).

    `committed_at` is git's `%ct`, the commit's own recorded time, so it survives
    the repo being copied or its files being touched; `age` is that against
    `now`. A repository that is missing, unreadable or has no commit at all is
    reported `fresh: False` and NOT green-by-absence: `git log` answers `fatal:
    not a git repository` for a path that does not resolve — which is how
    `SETUP.md`'s own check is spelled with a literal `~` — and a reader that
    treated "no commit time" as "no problem" would have called four dead days
    fresh.

    `repo=None` takes `snapshot_repo()`.

    `owned` is a second axis, and deliberately not folded into `fresh`. A repository
    outside `snapshot_owner_dir()` is not the vault's off-vault copy whatever its HEAD
    says, and `fresh=False` is the guardian's page: folding the two would let a
    caller's fixture repo demand a person's attention (#3149's incident), while
    dropping the measurement would leave the operator's CLI with nothing to print. So
    the age is still measured and still printed, the sentence says whose repository it
    is not, and only a repository the backup stream owns may raise an alert — which is
    the same shape `datawatch.snapshot_state` reaches for its root marker (#1416):
    measure, then say whether this tree is the tree the check is allowed to speak for.
    """
    path = repo if repo is not None else snapshot_repo()
    owned = snapshot_repo_is_owned(path)
    state: dict[str, Any] = {"repo": path, "sha": None, "committed_at": None,
                             "age": None, "max_age": max_age, "fresh": False,
                             "owned": owned}
    try:
        out = subprocess.run(["git", "--git-dir", path, "log", "-1", "--format=%H %ct"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        out = None
    if out is not None and out.returncode == 0:
        fields = out.stdout.strip().split()
        if len(fields) == 2:
            state["sha"], state["committed_at"] = fields[0], int(fields[1])
    if state["committed_at"] is None:
        state["text"] = (f"{path} has no readable HEAD commit, so no vault snapshot is known "
                         f"to be newer than the {max_age / 3600.0:g} h limit")
        if not owned:
            state["text"] += _not_owner_clause(snapshot_owner_dir())
        return state
    state["age"] = (time.time() if now is None else now) - state["committed_at"]
    state["fresh"] = state["age"] <= max_age
    state["text"] = (f"{path} commit {state['sha'][:8]} is {_age_phrase(state['age'])} old"
                     + ("" if state["fresh"]
                        else f", past the {max_age / 3600.0:g} h limit"))
    if not owned:
        state["text"] += _not_owner_clause(snapshot_owner_dir())
    return state


def snapshot_age_report(repo: str | None = None, max_age: float = SNAPSHOT_MAX_AGE_SECONDS,
                        now: float | None = None) -> tuple[bool, str]:
    """Green/non-green and the sentence, from the one measurement above."""
    state = snapshot_age_state(repo, max_age, now)
    return state["fresh"], state["text"]


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "status"
    w = VaultWatch()
    if cmd == "status":
        snap = measure(w.root, skip_top=w.skip_top)
        # `snapshot` is here because `status` is the reading of vault health a
        # person or a check runs, and for four days it answered "tripwire not
        # set, 8516 files" about a vault whose only out-of-vault copy had not
        # moved since 2026-10-16. The tripwire guards the tree; the snapshot is
        # what a restore would be taken from, and nothing looked at it.
        print(json.dumps({"tripped": w.tripped(),
                          "last_good": w.history[-1].to_dict() if w.history else None,
                          "now": snap.to_dict() if snap else None,
                          "snapshot": snapshot_age_state()}, indent=2))
        return 0
    if cmd == "clear":
        existed = w.clear()
        print("tripwire cleared; baseline reset to the vault as it is now" if existed
              else "tripwire was not set; baseline refreshed")
        if existed:
            # The trip paused the worker pool, and since 2026-09-22 that pause
            # survives backend restarts (it used to be undone by the first one).
            print("the worker pool the trip paused stays paused; resume it when ready: "
                  "curl -X POST localhost:8080/api/workers/pause "
                  "-H 'content-type: application/json' -d '{\"paused\": false}'")
        return 0
    if cmd == "sync-gate":
        ok, why = sync_gate()
        print(("ok: " if ok else "REFUSING: ") + why, file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 1
    if cmd == "snapshot-age":
        # A status code, not only a word: 370 exit-128 runs went unreported
        # because every shape of this failure either exited 0 or was never
        # measured, so the reading has to be one a check can run and fail on.
        fresh, text = snapshot_age_report(argv[2] if len(argv) > 2 else None)
        print(("fresh: " if fresh else "STALE: ") + text,
              file=sys.stdout if fresh else sys.stderr)
        return 0 if fresh else 1
    print(f"usage: {argv[0]} status|clear|sync-gate|snapshot-age [REPO]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
