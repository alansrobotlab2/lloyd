"""A disposable copy of Lloyd's state, for bench tasks whose success is a write (#3403).

Harbor's primitive, recreated locally: an agent runs against seeded state, and what it
did is read out of that state afterwards by a script the agent never saw. The unit under
test is Lloyd's harness plus its own MCP tools, so the state that has to be copyable is
the state those tools write — the vault, the KG store, `workers.db`.

Why this file exists at all is `bench_runner_sdk.STATEFUL_TOOLS`: `memory_add`,
`fact_add`, `backlog_write_task` and their neighbours are pushed into `disallowed_tools`
for every trial, on the grounds that a trial must not edit cross-session memory or move
a card. That is the right rule against the live house, and it is why no bench task could
ever ask "did the run actually do the thing" — a skill's normal successful outcome IS a
durable write, so no skill's own job was gradeable. The ban is a rule about names; a
copy is isolation. This module builds the copy; `bench_side_effects.py` reads what
landed in it.

Three rules make the copy safe, and each one exists because of a specific way the
half-built version fails:

1. **The destination is refused if it resolves inside a live tree.** A fixture that
   lands in `~/obsidian` or `~/lloyd-data` is not a fixture, it is a scratch write into
   the house wearing a name that says otherwise.
2. **`DATA_ROOT_MARKER` travels with the copy**, because `app.data_root.resolve_data_root`
   refuses a data root without it, and a silently-missing marker is how a trial ends up
   reading the production store while believing it was handed one.
3. **The vault copy gets its own `git init` and a baseline commit**, so "what did this
   run write" is one `git diff --name-only` rather than a checksum manifest of 25k files.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.data_root import (
    DATA_ROOT_MARKER, KG_DB_RELATIVE, PRODUCTION_DATA_ROOT, PRODUCTION_VAULT,
    VAULT_ROOT_ENV, VAULT_ROOT_FALLBACK_ENV)

logger = logging.getLogger(__name__)

#: Where a fixture lives under a data root. Under `DATA_ROOT`, not beside it, for the
#: reason the item states: the copy is a second copy of Alan's actual memory, and it has
#: to sit where the rest of the runtime data's handling (permissions, the retention
#: sweep, the backup exclusion) already applies.
FIXTURE_ROOT_RELATIVE = Path("eval") / "fixtures"

#: How long a directory in the fixtures area may outlive its own build: 1800 s, three times
#: the runner's operative per-trial turn cap — `DEFAULT_PER_TASK_TIMEOUT = 600`
#: (`bench_runner_sdk.py:153`), handed to the SDK arm as `per_task_timeout` at
#: `run_round.py:390`. Restated as a number rather than imported; the node that pins this
#: constant also re-reads that cap out of `bench_runner_sdk.py`, so the two cannot quietly
#: come apart the way the last restatement did.
#:
#: The arithmetic, which is what the previous 300 got wrong. A trial's hold on its own copy
#: is not only its turn: `asyncio.wait_for` bounds the agent loop at `per_task_timeout`
#: (`bench_runner_sdk.py:1141`), and the deletion sits in a `finally` at `:1199` — AFTER
#: `score_side_effects` runs at `:1184`, whose window is `VERIFIER_TIMEOUT_SECONDS = 120`
#: PER CHECK (`bench_side_effects.py:48`, applied to each check at `:119`). So one live
#: hold is 600 (turn cap) + 120 x the checks the task declares (verifier) + margin, and 1800
#: is 600 + 120 + 1080: the cap, one check's window, and the 1080 s of margin that keeps a
#: copy whose own trial is still in its `finally` out of the sweep's reach for up to ten
#: checks' worth of verifying. 300 was HALF the turn budget alone, so the sweep a sibling's
#: build runs could delete the copy a live trial was still writing into.
#:
#: The ceiling is not a storage policy, it is a fact about the object: a fixture is built by
#: one trial and deleted by the same trial, so a copy in this area older than a whole trial
#: plus its verifier cannot be a live one. What it ages out is the orphan the always-delete
#: rule cannot reach — `cleanup()` is called at the END of the trial
#: (`bench_runner_sdk.py:1199`), after the side-effect verifier, so a raise between the
#: trial's `finally` and that line, or a process killed mid-trial, leaves a second copy of
#: Alan's actual memory with nothing that ages it. Whether the ceiling should be longer —
#: kept for a failed run's forensics rather than deleted with the trial — is the ruling #3690
#: leaves to owed-check; the mechanism lives here so that ruling changes a number and not a
#: design. It has not been made: this copy is still deleted with its trial, always.
ORPHAN_MAX_AGE_S = 1800

#: The stores a task may declare, as `writes:` entries, that live in the data root
#: rather than the vault. Anything else in `writes:` is a vault-relative path.
DATA_STORE_SCOPES = {"kg": KG_DB_RELATIVE, "workers.db": "workers.db"}

#: Vault paths never copied: `.git` would carry the live vault's history (and its
#: `origin`) into a copy whose whole purpose is an enumerable diff, and `.obsidian`/
#: `.trash` are the user's editor state, not state a task operates on.
VAULT_IGNORE = {".git", ".obsidian", ".trash", ".DS_Store"}


class FixtureRefused(RuntimeError):
    """The fixture could not be built safely. Always means: nothing was copied."""


def fixture_parent(data_root: Path | str) -> Path:
    return Path(data_root) / FIXTURE_ROOT_RELATIVE


def new_trial_id(task_id: str) -> str:
    """A fixture directory name: the task it belongs to, plus a token nobody collides on.

    `uuid4` and not a counter because two runners can be building fixtures for the same
    task at the same moment — `--trials 5` is a loop, and a round's matrix runs tasks
    concurrently — and two fixtures under one name are one fixture that sees the other's
    writes.
    """
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)[:48]
    return f"{safe}_{uuid.uuid4().hex[:12]}"


def _refuse_if_inside(candidate: Path, live: Path, what: str) -> None:
    """Refuse when `candidate` is `live` or underneath it.

    `Path.relative_to` rather than a string prefix match: `/home/alan/lloyd-data-2` is a
    prefix of nothing but reads as "inside `/home/alan/lloyd-data`" to `startswith`, and
    the check that misjudges a sibling directory is the check that greenlights a write
    into the house.
    """
    c, l = Path(candidate), Path(live)
    for a, b in ((c, l), (c.resolve(), l.resolve())):
        if a == b:
            raise FixtureRefused(
                f"{what} would be the live tree itself ({b}) — a fixture must be a copy, "
                f"not the thing it is graded on")
        try:
            a.relative_to(b)
        except ValueError:
            continue
        raise FixtureRefused(
            f"{what} resolves inside the live tree ({b}) — refusing to copy into it")


def _delete_refusal(target: Path, fixtures_area: Path | None = None) -> str | None:
    """Why deleting `target` must be refused, or `None` when the rails let it through.

    THIS is the guard, and it is the only one: `BenchFixture.cleanup()` asks it about the
    one directory it owns, and `sweep_orphan_fixtures()` asks it about every directory it is
    offered. A second copy of the rule is a second way for the two to disagree about what is
    deletable, which is the failure #3690's "keep the mechanism to ONE" is about.

    Three rails, each earned:

    (1) The parent must be named `fixtures`. A deletion whose target sits somewhere else is
        a caller that passed the wrong parent, and `shutil.rmtree` does not undo.
        `fixtures_area` is `cleanup()`'s escape hatch for the deeper layout rule (1) of
        `build_fixture` allows — a destination nested under the data root's fixtures area —
        so the sweep passes it when it knows the area and refuses by name alone when not.
    (2) and (3) Neither production tree may be inside the target, by the same strong test
        `build_fixture` refuses a destination with: `relative_to` on both the literal and the
        resolved path, so a symlinked entry that points at `~/obsidian` is refused by the
        rail rather than deleted through it.
    """
    t = Path(target)
    if t.parent.name != FIXTURE_ROOT_RELATIVE.name and \
            (fixtures_area is None or t.parent.parent != fixtures_area):
        return (f"refusing to delete {t}: it is not a fixture directory under "
                f"{FIXTURE_ROOT_RELATIVE}")
    for live, what in ((PRODUCTION_VAULT, "PRODUCTION_VAULT"),
                       (PRODUCTION_DATA_ROOT, "PRODUCTION_DATA_ROOT")):
        try:
            # The strong test IS `build_fixture`'s own call, so a destination that cannot be
            # BUILT inside a live tree cannot be DELETED out of one either and neither rail
            # can be narrowed on its own. Only the sentence is this one's: a caller asking to
            # delete is not being refused a copy.
            _refuse_if_inside(t, live, "the live tree")
        except FixtureRefused:
            return (f"refusing to delete {t}: it resolves inside {what} ({live}) — the same "
                    "test `build_fixture` refuses a destination with")
    return None


def sweep_orphan_fixtures(fixtures_parent: Path | str, max_age_s: float,
                          *, now: float | None = None,
                          fixtures_area: Path | None = None) -> dict[str, list[str]]:
    """Delete fixture copies older than `max_age_s` seconds; report every decision.

    Returns `{"deleted": [...], "kept": [...], "refused": [...]}` of path strings, and never
    raises for a path it declines: a housekeeping pass that can veto the build it is part of
    is a housekeeping pass that gets removed from the build. A refused path is reported
    instead, and `build_fixture` logs it, so an entry that will not age out stays visible
    rather than becoming a silent leak.

    `now` is injectable so a test can age an entry by the clock rather than by sleeping, and
    `fixtures_area` forwards rail (1)'s escape hatch — `build_fixture` passes the area it has
    already validated its destination against.

    Only a directory can be an orphan: a file in this area is not a fixture copy, and a
    SYMLINK is refused rather than followed — `shutil.rmtree` will not delete a link to a
    directory, so treating one as deleted would be a report of a removal that did not happen.
    """
    parent = Path(fixtures_parent)
    report: dict[str, list[str]] = {"deleted": [], "kept": [], "refused": []}
    if not parent.is_dir():
        return report
    cutoff = (time.time() if now is None else float(now)) - float(max_age_s)
    for entry in sorted(parent.iterdir()):
        if entry.is_symlink():
            report["refused"].append(f"{entry}: refusing to delete a symlink")
            continue
        if not entry.is_dir():
            continue
        reason = _delete_refusal(entry, fixtures_area)
        if reason:
            report["refused"].append(f"{entry}: {reason}")
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError as exc:              # raced away, or unreadable: neither is a delete
            report["refused"].append(f"{entry}: stat failed ({exc})")
            continue
        if mtime > cutoff:
            report["kept"].append(str(entry))
            continue
        shutil.rmtree(entry, ignore_errors=True)
        if entry.exists():
            report["refused"].append(f"{entry}: still present after rmtree")
            continue
        report["deleted"].append(str(entry))
    if report["deleted"]:
        logger.warning("bench_fixture: swept %d orphaned fixture copy/copied: %s",
                       len(report["deleted"]), ", ".join(report["deleted"]))
    if report["refused"]:
        logger.warning("bench_fixture: refused %d fixture entr(ies): %s",
                       len(report["refused"]), "; ".join(report["refused"][:5]))
    return report


def declared_scopes(task: dict[str, Any]) -> list[str]:
    """The task's `writes:` list, normalised, or [] when it declares none."""
    raw = task.get("writes")
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    return [str(s).strip().strip("/") for s in raw if str(s).strip().strip("/")]


def build_fixture(*, trial_id: str, writes: list[str], vault: Path | str,
                  data_root: Path | str, fixtures_parent: Path | str | None = None,
                  git_identity: str = "bench",
                  orphan_max_age_s: float | None = None) -> "BenchFixture":
    """Copy the declared slice into `<fixtures_parent>/<trial_id>/` and return it.

    `vault` and `data_root` are the SOURCE — passed in rather than looked up, so a test
    can build from a small fake tree and the caller cannot be surprised by which house it
    copied. `fixtures_parent` defaults to `data_root/eval/fixtures`, which is the
    production layout; a test passes its own tmp dir.

    Raises `FixtureRefused` before the first byte is copied if the destination resolves
    inside the source vault, the source data root, or either production tree — the
    refusal is checked against four paths, not two, because `data_root` here is a
    parameter and a caller that passed a copy's path as its source still must not be
    allowed to write into `~/lloyd-data`.
    """
    scopes = declared_scopes({"writes": writes})
    if not scopes:
        raise FixtureRefused(
            "no `writes:` scope declared — a fixture with no declared scope is how a "
            "trial ends up sandboxed from nothing in particular")

    src_vault, src_data = Path(vault), Path(data_root)
    dest_parent = Path(fixtures_parent) if fixtures_parent else fixture_parent(src_data)
    dest = dest_parent / trial_id

    # Three rules, and the reason each exists.
    #
    # (1) The destination must sit in the data root's fixtures area. That is where
    #     clause 3 puts a fixture, and it is what makes the layout auditable: one place
    #     under the trial's own overridden data root holds every fixture, and cleanup
    #     knows the shape it may delete. `fixture_parent(src_data)` is that area, so the
    #     default path is legal by construction and an explicit `fixtures_parent`
    #     anywhere else — inside the data root's other directories, or outside it
    #     altogether — is refused rather than quietly building in the house.
    # (2) The two PRODUCTION roots are refused by the strong test, a destination
    #     ANYWHERE inside them: with `LLOYD_DATA` unset the source data root IS the live
    #     one and rule (1) would happily bless `~/lloyd-data/eval/fixtures/<id>`.
    # (3) No destination may be or contain a root it copies from or the live tree, and
    #     nothing may be copied inside the source vault, whose tree is the thing under
    #     test rather than a place to stage a copy.
    fixtures_area = fixture_parent(src_data)
    if not dest.is_relative_to(fixtures_area):
        raise FixtureRefused(
            f"fixture destination {dest} is not inside the data root's fixtures area "
            f"({fixtures_area}) — fixtures are built in one auditable place under the "
            "trial's own overridden data root, and nowhere else")
    for live, what in ((PRODUCTION_VAULT, "PRODUCTION_VAULT"),
                       (PRODUCTION_DATA_ROOT, "PRODUCTION_DATA_ROOT")):
        _refuse_if_inside(dest, live, f"fixture destination {dest} (inside {what})")
    _refuse_if_inside(dest, src_vault, "fixture destination inside the source vault")
    for live, what in ((src_vault, "the source vault"),
                       (src_data, "the source data root"),
                       (PRODUCTION_VAULT, "PRODUCTION_VAULT"),
                       (PRODUCTION_DATA_ROOT, "PRODUCTION_DATA_ROOT")):
        if dest == live or live.is_relative_to(dest):
            raise FixtureRefused(
                f"fixture destination {dest} is or contains {what} ({live}) — the copy "
                "would be built over what it copies, and cleanup would delete it")

    if dest.exists():
        raise FixtureRefused(f"fixture destination already exists: {dest}")

    # Bound the orphans before making room for this one. A build is the one moment the leak
    # is observable from the inside: the area is about to be used, so anything in it older
    # than a whole trial and its verifier belongs to a build that never reached its
    # `cleanup()` — a trial that raised between its own `finally` and the call that follows
    # it (`bench_runner_sdk.py:1199`), or a process killed mid-run. This is also why the
    # ceiling is a multiple of the turn cap rather than the cap itself: the sweep runs from a
    # sibling trial's build in this same gathered process, so a copy inside its own trial's
    # total hold must be KEPT here. Concurrency is not on today's default path — every
    # in-tree caller passes `max_parallel=1` — but `run_bench_sdk`'s own signature default is
    # 2 (`bench_runner_sdk.py:1220`) and every process shares this one area, so the day it
    # does, an under-sized ceiling is a live sibling deleted mid-trial.
    #
    # Here and not in the data-root retention sweep deliberately: the sweep resolves
    # `$LLOYD_DATA` (`scripts/groundskeeper/retention-sweep.py:395`), which rule (2) above
    # guarantees can never hold a fixture, so a store added there would enumerate an empty
    # path forever. `orphan_max_age_s` overrides the ceiling for a caller that knows better
    # — the runner's own per-trial timeout — and `None` means the documented default.
    sweep_orphan_fixtures(dest_parent,
                          ORPHAN_MAX_AGE_S if orphan_max_age_s is None else orphan_max_age_s,
                          fixtures_area=fixtures_area)

    dest.mkdir(parents=True)
    fixture_data = dest / "lloyd-data"
    fixture_vault = dest / "obsidian"
    fixture_vault.mkdir(parents=True)
    fixture_data.mkdir(parents=True)
    (fixture_data / DATA_ROOT_MARKER).write_text(
        f"bench fixture {trial_id}\n", encoding="utf-8")

    copied: list[str] = []
    for scope in scopes:
        if scope in DATA_STORE_SCOPES:
            rel = DATA_STORE_SCOPES[scope]
            src = src_data / rel
            dst = fixture_data / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(src, dst)
            else:
                # A missing store is not a refusal: an empty fixture KG is a valid
                # environment for "add a fact", and creating the file is what
                # `app.kg_store` would do itself. Copying nothing but recording the
                # intended path is the honest version of that.
                dst.touch()
            copied.append(rel)
            continue
        # `scope` is the declaration ("vault/memory"); inside a vault root the path is
        # the same declaration with its prefix taken back off. Joining the whole
        # declaration onto the source vault asks for `<vault>/vault/memory` and every
        # vault-scoped build refuses with "scope does not exist", which is the shape
        # that reads like a bad task rather than a broken resolver.
        # A vault scope is a path INSIDE the vault root, so joining the declaration
        # straight onto the source asks for `<vault>/vault` and every vault-scoped
        # build refuses with "scope does not exist" — which reads like a bad task and
        # is a broken resolver. Bare `vault` is the whole slice, `vault/backlog` and
        # `backlog` are the same directory declared two ways.
        if scope == "vault":
            src, dst = src_vault, fixture_vault
        else:
            rel = scope[len("vault/"):] if scope.startswith("vault/") else scope
            src, dst = src_vault / rel, fixture_vault / rel
        if not src.exists():
            shutil.rmtree(dest, ignore_errors=True)
            raise FixtureRefused(
                f"declared vault scope does not exist in the source vault: {scope} "
                f"(looked at {src})")
        dst.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True, ignore=_vault_ignore,
                            dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        copied.append(scope)

    fx = BenchFixture(id=trial_id, root=dest, vault=fixture_vault,
                      data_root=fixture_data, scopes=scopes, copied=copied,
                      source_vault=src_vault, source_data_root=src_data)
    fx.init_git(git_identity=git_identity)
    return fx


def _vault_ignore(directory: str, names: list[str]) -> set[str]:
    return {n for n in names if n in VAULT_IGNORE}


@dataclass
class BenchFixture:
    """One built fixture: the paths, the env that points a process at them, and cleanup.

    `.cleanup()` is not optional-by-convention: this copy is the user's actual memory in
    a second location, so it lives exactly as long as the trial that needed it.
    """

    id: str
    root: Path
    vault: Path
    data_root: Path
    scopes: list[str]
    copied: list[str] = field(default_factory=list)
    source_vault: Path | None = None
    source_data_root: Path | None = None
    git_initialized: bool = False

    @property
    def kg_db(self) -> Path:
        return self.data_root / KG_DB_RELATIVE

    @property
    def workers_db(self) -> Path:
        return self.data_root / "workers.db"

    def env(self) -> dict[str, str]:
        """The environment a trial process needs to resolve every root into this copy.

        Both vault names are set, deliberately, even though the resolver honours both:
        `agent-services/guardian/*.py` and one or two scripts read one name only, and an
        env that sets both cannot leave a reader on the live vault because it picked the
        other spelling. That belt-and-braces is the whole lesson of this item.
        """
        return {
            "LLOYD_DATA": str(self.data_root),
            VAULT_ROOT_ENV: str(self.vault),
            VAULT_ROOT_FALLBACK_ENV: str(self.vault),
            "LLOYD_KG_DB": str(self.kg_db),
        }

    def init_git(self, git_identity: str = "bench") -> None:
        """`git init` + baseline commit, so the run's writes are one `diff --name-only`.

        Configured with a fixture-local identity, and with `commit.gpgsign=false`: the
        account's global git config signs commits, and a fixture whose baseline commit
        invokes a gpg agent is a fixture build that fails for a reason that has nothing
        to do with the task.
        """
        r = subprocess.run(["git", "-C", str(self.vault), "init", "-q", "-b", "main"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise FixtureRefused(f"fixture `git init` failed: {r.stderr.strip()[:200]}")
        for key, value in (("user.name", git_identity),
                           ("user.email", f"{git_identity}@bench.lloyd.local"),
                           ("commit.gpgsign", "false")):
            subprocess.run(["git", "-C", str(self.vault), "config", key, value],
                           capture_output=True, text=True)
        subprocess.run(["git", "-C", str(self.vault), "add", "-A"],
                       capture_output=True, text=True)
        c = subprocess.run(["git", "-C", str(self.vault), "commit", "-q", "--allow-empty",
                            "-m", "bench fixture baseline"],
                           capture_output=True, text=True)
        if c.returncode != 0:
            raise FixtureRefused(f"fixture baseline commit failed: {c.stderr.strip()[:200]}")
        self.git_initialized = True

    def writes(self) -> list[str]:
        """Every path this copy holds that differs from the baseline, vault-relative.

        `git add -N` (intent-to-add) first, then `diff --name-only`: without it a file
        the run CREATED is untracked and `git diff` says nothing, so a task whose whole
        success condition is "wrote a new note" would grade as a no-op. `-N` records the
        path without staging content, which keeps the working tree exactly as the run
        left it — a verifier that reads the file afterwards reads what the agent wrote,
        not a staged copy of it.
        """
        if not self.git_initialized:
            raise RuntimeError("fixture has no git baseline; `init_git()` was skipped")
        subprocess.run(["git", "-C", str(self.vault), "add", "-A", "-N"],
                       capture_output=True, text=True)
        r = subprocess.run(["git", "-C", str(self.vault), "diff", "--name-only"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"fixture diff failed: {r.stderr.strip()[:200]}")
        return sorted(p for p in r.stdout.splitlines() if p.strip())

    def cleanup(self) -> None:
        """Delete the copy. Refuses to delete anything that is not under its parent.

        The guard is not paranoia about a bug in this file: `shutil.rmtree` on a path
        that resolved to the vault because a caller passed the wrong parent is
        unrecoverable, and the cost of checking is one comparison. The comparison lives in
        `_delete_refusal`, which `sweep_orphan_fixtures` asks the same question of — one
        guard shared by the two deleters, because two rails that can disagree about what is
        deletable is the failure mode neither of them is allowed to have.
        """
        reason = _delete_refusal(self.root, fixture_parent(self.data_root))
        if reason:
            raise RuntimeError(reason)
        shutil.rmtree(self.root, ignore_errors=True)

    def as_row(self) -> dict[str, Any]:
        """Fixture facts worth recording on the trial row."""
        return {
            "bench_fixture": str(self.root),
            "bench_fixture_scopes": list(self.scopes),
            "bench_fixture_vault": str(self.vault),
            "bench_fixture_data_root": str(self.data_root),
            "bench_fixture_git": self.git_initialized,
        }
