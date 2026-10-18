#!/home/alansrobotlab/lloyd/.venvs/lloyd/bin/python
"""The find-all arms run's owner: an engine lock held with patience, loud refusals, a published table.

Why this file exists. ``eval/run_find_all_arms.py`` has been fixed-but-unused since
``244ae542`` shipped on 2026-10-15 (#2741): the only artifact on the box is still
``eval/measurements/find-all-2026-09-24/`` (mtime Sep 29) whose 20 rows all lack
``turn_budget``, so #647's phrasing clause — the all-vs-one recall ordering over answered
trials on #014 and #016 at n>=3 — has never been measurable. It stayed unused because no
automod round can hold the engine lock and run a harness about itself, and the round route
is the only owner it has ever had. This launcher is the other kind of owner: the one a
skill and an autonomy task can fire.

The lock, precisely. This waits on ``~/.local/state/lloyd-automod/primary.lock`` because
every eval driver's docstring says to and it is the only guard against two eval runs on
one engine. It is NOT what an automod round holds — the round lock is ``STATE_DIR /
"lock"`` (``scripts/automod/state.py:68``), a different file — so the standing theory that
"a ``flock -n`` one-shot exits silently while a round holds it" was never the reason
nothing ran (measured 2026-10-18: ``git grep -n "primary.lock" -- app/ scripts/
workers/`` returns one hit, a docstring at ``scripts/skill_pseudo_queries.py:23``, and
``lsof`` on the lock was empty at 11:07Z with 17 rounds already booked). The wait is
therefore sized for the contention that does exist — another eval holding the primary for
its own duration — rather than the 12-hour one that does not, and it is a bounded wait
rather than a ``-n`` one-shot because ~41-51 automod rounds a day keep an engine busy and
an attempt that gives up on first contact is an attempt that never happens.

Three rules this launcher will not bend:

1. **It never exits 0 having run nothing.** Every stop prints WHICH precondition stopped
   it: the lock still held when the wait ended, the driver's own tool-sandbox refusal, a
   gold mismatch, or a driver that exited without writing ``trials.jsonl``. That naming is
   the whole design: for a week the absence of an artifact was blamed on the lock, and a
   refusal that says only "failed" is how a wrong cause survives.
2. **It re-derives the gold before spending engine time.** Each of these four answer keys
   is a statement about the live tree — #014 is dead wikilinks under ``projects/``, #016 is
   dead ``~/lloyd/`` paths in active skills, #017 is autonomy tasks naming a missing skill,
   #015 is fact copies the store still holds — and the 2026-09-24 report's own closing
   line is "re-label on rot". A drifted gold scores every arm wrong identically, which
   reads as a null result rather than a broken answer key, and 18 harness sessions is an
   expensive way to find out. Running outside pytest is deliberate: ``tests/conftest.py``'s
   ``_production_tree`` guard refuses the production tree, which is the same reason
   ``tests/test_bench_audit_tasks.py:149-190`` exists.
3. **It publishes the table and stops there.** The report prints each cell's counts and its
   at-cap accounting beside the P/R/F1 means, plus the judge-independence disagree-rate
   block. It does NOT print the all-versus-one recall ordering: that ruling is a reading of the
   table by whoever owns #647, and a script that printed the verdict would be read as
   having made it.

Nothing here scores anything. The counts come from the driver's own ``summarise`` (invoked
as ``--summarise``, which re-reads ``trials.jsonl`` at zero engine cost); P/R/F1 are means
of the per-trial fields the driver already wrote. There is no second implementation of
precision in this file, which is the failure #2918/#2919 recorded for an arm that
re-derived an answer key from a fixture instead of loading the answer.

Usage
-----
    .venvs/lloyd/bin/python scripts/find_all_arms_run.py                # the run
    .venvs/lloyd/bin/python scripts/find_all_arms_run.py --dry-run      # print the command
    .venvs/lloyd/bin/python scripts/find_all_arms_run.py --summarise-only eval/measurements/find-all-2026-09-24

Fired DETACHED. The run is 18 harness sessions over ~1.5-2.5 h of primary-engine time and
the scheduled-task pool caps one session at 3570 s (``max_duration=3600`` minus
``_POOL_TIMEOUT_MARGIN=30``: ``app/autonomy.py:4572-4581``,
``workers/sources/scheduled_task.py:415``), so the autonomy task that owns this starts it
in its own process group and reports the log path rather than hosting it. ``scripts/`` has
no ``__init__.py``, so a caller imports this by file path the way the tests do.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.atomic_io import _flock_exclusive  # noqa: E402  (bounded-wait flock, already in the tree)
from app.paths import VAULT_FACTS_ROOT  # noqa: E402  (the env-overridable derived-fact root)

#: The engine lock. The same root as `scripts/automod/state.py` STATE_DIR, spelled here so
#: importing this launcher does not import the automod state module, which mkdirs on load.
PRIMARY_LOCK = Path.home() / ".local" / "state" / "lloyd-automod" / "primary.lock"

DRIVER = "eval/run_find_all_arms.py"
BENCH_DIR = Path.home() / "obsidian" / "lloyd" / "bench"
VAULT = Path.home() / "obsidian"
#: The markdown tree the derived fact index is built from; #015's copies live here. NOT
#: ``~/obsidian/facts``: that path exists and is EMPTY (verified 2026-10-18 — it is a real
#: directory holding 0 ``*.md``, while the index is built from the derived root below, which
#: holds 41,759). A walk of an empty tree derives 0, and 0 against a bench declaring 3 reads
#: as a drifted answer key rather than as the launcher having looked in the wrong place.
#: ``app.paths.VAULT_FACTS_ROOT`` is the same path the store itself indexes (``app/paths.py:278``),
#: honouring ``LLOYD_FACTS_ROOT``, so this launcher and the judge never disagree about the tree.
VAULT_FACTS = VAULT_FACTS_ROOT

#: The pinned run #647's phrasing clause needs. `--timeout 900` is NOT the driver's
#: default (600, `eval/run_find_all_arms.py:304`, left deliberately by #2160 because an
#: empty-but-finished trial is a real result and a wall is not a verdict), and
#: `--max-turns 25` is not the runner's default either (the bench's 12 is what left three
#: of the eight 2026-09-24 `all` trials empty, confounding the phrasing gap on one task).
PINNED_TRIALS = 3
PINNED_MAX_TURNS = 25
PINNED_TIMEOUT = 900
DEFAULT_WAIT_SECONDS = 3600

_WIKILINK = re.compile(r"\[\[([^\]|#]+)")
_FENCE = re.compile(r"```.*?```|`[^`\n]*`", re.S)
#: #016's key, verbatim from its bench note's `dedupe_key_regex`: the path after `lloyd/`.
_SKILL_PATH = re.compile(r"(?:~|/home/alansrobotlab)/lloyd/([A-Za-z0-9_./-]*[A-Za-z0-9_-])")
#: A section that presents a path as a wrong-path EXAMPLE rather than as something to run, which
#: #016's prompt excludes by name. Matched against the enclosing HEADING, not the line, because
#: the live case (`file-path-resolution` line 25, under "### Pattern 1: Guessing Wrong File
#: Paths") carries no such word on its own line.
_WRONG_PATH_SECTION = re.compile(
    r"wrong|mistak|failure|anti-?pattern|do not|don.?t|never|bad path|example of", re.I)
_TASK_SKILL = re.compile(r"^skill_name:\s*(\S+)\s*$", re.M)
_NUMBERED_TASK = re.compile(r"^\d+-.*\.md$")
#: The number inside a bench note's own id: `bench_014_audit_dead_wikilinks` -> `#014`.
_BENCH_ID_NUM = re.compile(r"^bench_(\d{3})[._-]")


class Refusal(Exception):
    """A precondition that stopped the run. The message names which one, so the next
    reader does not have to guess the way this run's absence was guessed at."""


def _bench_key(task: dict) -> str:
    """The short key a bench note is known by here: `#014`, from whatever it calls itself.

    The four live notes key themselves ``id: bench_014_audit_dead_wikilinks`` and carry **no**
    ``task_id`` key at all (verified 2026-10-18: ``grep -c '^task_id:'`` over the four notes →
    ``0`` for each; ``grep -h '^id:'`` → the four ``bench_0NN_...`` ids). The driver agrees:
    ``eval/run_find_all_arms.py:127`` writes ``"task_id": task["id"]`` and keys its summary
    cells on that. So a launcher that read only a ``task_id`` field would fall back to the
    file stem, match no entry in ``DERIVATIONS``, and REFUSE the real fire on its own bench —
    the same "refused and nothing ran" shape this item exists to end, from a new cause.

    Order of precedence: an explicit ``task_id`` (a fixture's ``'#014'``), then a ``bench_NNN``
    id folded to ``#NNN``, then the file stem, which stays unmatched and is refused by name.
    """
    explicit = str(task.get("task_id") or "").strip()
    if explicit:
        return explicit
    note_id = str(task.get("id") or "").strip()
    m = _BENCH_ID_NUM.match(note_id)
    if m:
        return f"#{m.group(1)}"
    return note_id or Path(task.get("_path", "")).stem


# ── the gold re-derivation (clause 3) ─────────────────────────────────────────

@dataclass
class GoldRow:
    """One find_all task's answer key: what the tree says now, beside what the bench declares."""

    task_id: str
    derived: int | None
    declared: int
    note: str = ""
    #: Anything the derivation left out on purpose, printed with the count it produced. An
    #: exclusion that is not printed is a second, invisible answer key.
    extra: str = ""

    @property
    def ok(self) -> bool:
        return self.derived is not None and self.derived == self.declared

    def render(self) -> str:
        d = "unavailable" if self.derived is None else str(self.derived)
        flag = "ok" if self.ok else "MISMATCH"
        return (f"gold {self.task_id}: derived={d} declared={self.declared} {flag}"
                + (f" ({self.note})" if self.note else "")
                + (f" [{self.extra}]" if self.extra else ""))


def _prose(text: str) -> str:
    """`text` with fenced blocks and inline code removed — what "ignore anything inside
    code" means in #014's own prompt, so a wikilink in a code sample is not a citation.
    """
    return _FENCE.sub(" ", text)


def _derive_dead_wikilinks(vault_root: Path) -> int:
    """#014's answer key: wikilinks under `projects/` whose target is nowhere in the vault.

    The rule is the task prompt's own, re-implemented because the launcher runs outside
    pytest: a target resolves when it names an existing note by vault-relative path, or —
    for a bare name with no slash — by file name ANYWHERE in the vault; `[[#Heading]]` and
    anything inside code do not count; one missing target is one answer however many notes
    link it. Measured against the live vault on 2026-10-18 this returns 15, which is what
    the bench declares.
    """
    vault_root = Path(vault_root)
    notes = [f for f in sorted(vault_root.rglob("*.md")) if ".git" not in f.parts]
    by_name: dict[str, list[Path]] = {}
    for f in notes:
        by_name.setdefault(f.name.lower(), []).append(f)
    missing: set[str] = set()
    for f in notes:
        if "projects" not in f.relative_to(vault_root).parts:
            continue
        for target in _WIKILINK.findall(_prose(f.read_text(encoding="utf-8",
                                                            errors="replace"))):
            t = target.strip().rsplit("|", 1)[0].strip()
            if not t or t.startswith("#"):
                continue
            key = t.rstrip("/").lower()
            if key in missing:
                continue
            cand = vault_root / (t if t.endswith(".md") else t + ".md")
            if cand.is_file():
                continue
            if "/" in t:
                missing.add(key)
            elif not by_name.get((t if t.endswith(".md") else t + ".md").lower()):
                missing.add(key)
    return len(missing)


def _heading_above(lines: list[str], idx: int) -> str:
    """The nearest heading at or above `idx`, with its `#`s stripped; '' above the first one."""
    for j in range(idx, -1, -1):
        s = lines[j].lstrip()
        if s.startswith("#"):
            return s.lstrip("# ").strip()
    return ""


def _dead_skill_path_citations(vault_root: Path, *, home_root: Path | None):
    """#016's answer key: `~/lloyd/...` paths in active skills that name a missing file.

    Returns `(missing_keys, skipped)`, where `skipped` is
    `(skill, line, key, heading)` for every dead-looking citation the rule excludes — printed
    by `gold_checks` so an exclusion is never invisible (see below).

    The task's own rule: every `SKILL.md` under `skills/` except anything in `.archived`; the
    path must name a file, which is what "has an extension in its last component" tests;
    templated paths do not survive the extraction at all (`<`, `{` and `*` are not in the
    character class, so `~/lloyd/docs/<date>.md` truncates to a directory and drops out, which
    is the intended skip); and the key is the part after `lloyd/`, so the `~` and
    `/home/alansrobotlab` spellings of one path are one finding.

    The one exclusion that decides whether this derivation agrees with the label at all: the
    prompt says to skip **"paths the skill itself presents as an example of a wrong path"**,
    and the note's body names the live case — `file-path-resolution` line 25,
    `/home/alansrobotlab/lloyd/lloyd/inner-voice/system-prompt.md`, listed under
    "### Pattern 1: Guessing Wrong File Paths" and recorded under "Not gold, deliberately",
    with a hand-checked `judge.grade_answer_set` negative case beside it. A bare
    path-exists walk counts it and derives **5 against a declared 4**, which would refuse the
    run on a tree whose label is right — the inverse of what this check is for.

    Mechanically, "presents as an example of a wrong path" is the section it sits in, so the
    rule is the enclosing heading: a citation under a heading naming wrongness, a mistake, a
    failure pattern, or an example is not a defect the skill depends on. Chosen over a
    line-local marker because line 25 carries no such word on its own line, and over a
    doubled-path shape rule (`lloyd/lloyd/…`) because that would silently drop a genuine
    nested-path defect. A key is kept if it is cited under ANY non-example heading anywhere, so
    the exclusion cannot swallow a path that is also a real defect elsewhere; and every
    exclusion is printed with its skill, line and heading, so an over-exclusion is legible in
    the run log rather than being another number nobody can argue with.
    """
    #: `home_root=None` is the real fire's spelling (`main` passes it when `--home-root` is
    #: absent), and `Path(None)` raises TypeError — which would kill the run with a traceback
    #: instead of the named refusal every stop here is required to print. What `~` means in a
    #: skill's `~/lloyd/...` path is the home the launcher runs in.
    vault_root = Path(vault_root)
    home_root = Path(home_root) if home_root is not None else Path.home()
    skills = [f for f in sorted((vault_root / "skills").rglob("SKILL.md"))
              if ".archived" not in f.parts]
    missing: set[str] = set()
    skipped: list[tuple[str, int, str, str]] = []
    for f in skills:
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        for idx, line in enumerate(lines):
            for key in _SKILL_PATH.findall(line):
                last = key.rstrip("/").rsplit("/", 1)[-1]
                if not last or "." not in last:
                    continue                   # names a directory, not a file
                if (home_root / "lloyd" / key).exists():
                    continue
                head = _heading_above(lines, idx)
                if _WRONG_PATH_SECTION.search(head) and key not in missing:
                    skipped.append((f.parent.name, idx + 1, key, head))
                    continue
                missing.add(key)
    #: A key first seen under an example heading and later cited as a real instruction is a
    #: finding, so a second pass drops the now-kept keys out of the exclusion report.
    skipped = [s for s in skipped if s[2] not in missing]
    return missing, skipped


def _derive_dead_skill_paths(vault_root: Path, *, home_root: Path) -> int:
    """#016's gold count; the rule and its one exclusion are in `_dead_skill_path_citations`."""
    return len(_dead_skill_path_citations(vault_root, home_root=home_root)[0])


def _derive_unresolved_task_skills(vault_root: Path) -> int:
    """#017's answer key (the zero-item control): numbered tasks naming a skill that is not there.

    `~/obsidian/autonomy/NN-*.md` only, `skill_name:` read from front matter, and the skill
    exists when `skills/<slug>/SKILL.md` does — the same resolution
    `app/autonomy.py:1077-1089` warns about (a slug whose file sits at the skills root
    loads as nothing). Live on 2026-10-18 this returns 0, which is what the note declares;
    a nonzero here is real rot, not a broken derivation, and it is why the control's
    derivation is checked rather than assumed.
    """
    vault_root = Path(vault_root)
    adir = vault_root / "autonomy"
    if not adir.is_dir():
        return 0
    unresolved: set[str] = set()
    for f in sorted(adir.glob("*.md")):
        if not _NUMBERED_TASK.match(f.name):
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        head = text.split("---")[:3]
        for slug in _TASK_SKILL.findall(head[1] if len(head) > 1 else ""):
            slug = slug.strip("'\"")
            if slug in ("", "null"):
                continue
            if not (vault_root / "skills" / slug / "SKILL.md").is_file():
                unresolved.add(f.name)
    return len(unresolved)


def _derive_fact_copies(check: dict, copy_lookup) -> int:
    """#015's answer key: how many declared gold rows the store still holds, live, for the entity named.

    One judgement, one source: the store's own `(entity, text_hash)` lookup
    (`_FactsIdx.find_duplicate`), the key `scripts/autoresearch/judge.py`'s dedupe uses, so
    a re-derived count cannot disagree with the bench about "the same claim". An expired or
    invalidated copy is not a hit — re-stating a superseded claim is a new claim, and a
    gold row nobody holds leaves the agent nothing to find.

    This covers the half pytest could check. The OTHER half of what that task claims — the
    same text held under a DIFFERENT entity, which is the actual defect — is read out of
    `kg_health.json`'s `cross_entity_exact_duplicates` by hand and is deliberately NOT
    re-derived here: nothing in the tree exposes it as a query, and a launcher that
    printed such a number would be inventing it.
    """
    #: The gold is a list of hash STRINGS (bench_015_audit_cross_entity_fact_copies.md:16-19),
    #: not of dicts: an earlier draft of this function expected entity/text keys, found
    #: none, and would have derived 0 forever against a bench declaring 3.
    #: The bench notes spell the claim `fact`; a test fixture that spells it `text` is the
    #: same shape. Reading only one key silently derives 0 for a gold set that is not empty.
    return sum(1 for h in (check.get("gold_items") or []) if copy_lookup(h))


#: How each find_all task's answer set is derived, keyed by task id. A find_all task that
#: is not in here REFUSES the run rather than being skipped: silently re-deriving three of
#: four answer keys is exactly the half-measure that lets a drifted gold through, and the
#: refusal names the task to add.
DERIVATIONS = {
    "#014": "dead_wikilinks",
    "#015": "fact_copies",
    "#016": "dead_skill_paths",
    "#017": "unresolved_task_skills",
}


def _fact_copy_lookup(facts_root: Path):
    """A lookup for #015's gold, on the store's OWN hash and parser over its OWN tree.

    #015's ``gold_items`` are 16-hex ``text_hash`` values, and its answer is "this hash is held
    by two or more distinct entities". The store exposes no public per-hash read:
    ``find_duplicate`` needs the entity AND the plaintext, ``search`` matches a substring, and
    ``exact_duplicate_stats`` answers only store-wide totals. So this walks the markdown tree the
    index is BUILT from with the index's own public primitives — ``app.kg_store.parse_fact_file``
    and ``app.kg_store.text_hash``, the latter documented at ``app/kg_store.py:181`` as what a
    writer must compute to ask the store whether it already holds a fact. The entity is the fact
    file's own front matter, which is how ``_rows_for_file`` reads it too.

    One limit, stated not assumed: the index's alias canonicalization (``_canonical_key``) is
    private, so a claim held twice under two surface names of ONE entity counts as one here and
    the derived number comes in LOW. Low refuses the run, which is the safe direction.
    """
    from app.kg_store import parse_fact_file, text_hash

    root = Path(facts_root)
    cache: dict[str, dict[str, set]] = {}

    def _entities_by_hash() -> dict[str, set]:
        if "map" not in cache:
            if not root.is_dir():
                raise Refusal(
                    f"the gold cannot be derived: no fact tree at {root}, so bench_015's "
                    f"cross-entity copy count has nothing to walk and would come out 0. Name "
                    f"one with --facts-root. No trial was spawned.")
            by_hash: dict[str, set] = {}
            n_files = 0
            for path in sorted(root.rglob("*.md")):
                n_files += 1
                fm, claims = parse_fact_file(path)
                entity = fm.get("entity")
                if not entity:
                    continue
                for entry in claims:
                    #: `_rows_for_file` reads the claim from `fact:`; derived copies also carry
                    #: `text:`. Reading one key silently under-counts the tree.
                    claim = entry.get("fact") or entry.get("text")
                    if claim:
                        by_hash.setdefault(text_hash(str(claim)), set()).add(str(entity))
            #: An EMPTY tree is the second way to derive 0, and it is not a gold mismatch.
            #: `~/obsidian/facts` is exactly this: it exists, and it holds zero `*.md`
            #: (measured 2026-10-18), so pointing here and reporting "the answer key drifted"
            #: would send the next reader to re-label a gold set that is fine. Say which
            #: directory was walked and what was found in it.
            if n_files == 0:
                raise Refusal(
                    f"the gold cannot be derived: the fact tree at {root} holds no markdown "
                    f"fact files at all, so bench_015's copy count would come out 0 for an "
                    f"environmental reason and read as a drifted answer key. The live index is "
                    f"built from app.paths.VAULT_FACTS_ROOT (LLOYD_FACTS_ROOT overrides it). "
                    f"Name the right tree with --facts-root. No trial was spawned.")
            cache["map"] = by_hash
        return cache["map"]

    def lookup(hash_value: str) -> bool:
        return len(_entities_by_hash().get(str(hash_value), ())) >= 2

    return lookup


def gold_checks(bench_dir: Path = BENCH_DIR, *, vault_root: Path = VAULT,
                home_root: Path | None = None, store_lookup=None,
                facts_root: Path = VAULT_FACTS,
                only: set[str] | None = None) -> list[GoldRow]:
    """Re-derive every find_all answer key on the box and line it up against the bench.

    `store_lookup(hash_value) -> bool` is injectable so #015's derivation can be tested
    without walking the real 41,759 fact files; the default is `_fact_copy_lookup(facts_root)`.
    Called before anything is spawned, because a drifted gold costs 18 harness sessions to
    discover any other way, and `tests/conftest.py`'s `_production_tree` guard (:170-171) is
    why this runs as a script and not a pytest node - the same reason
    `tests/test_bench_audit_tasks.py:149-190` does.
    """
    from scripts.autoresearch.common import load_bench_tasks

    if store_lookup is None:
        store_lookup = _fact_copy_lookup(facts_root)

    rows: list[GoldRow] = []
    keyed = [(task, _bench_key(task)) for task in load_bench_tasks(Path(bench_dir))]
    for task, tid in sorted(keyed, key=lambda pair: pair[1]):
        #: The real bench notes key their checks `objective_checks`; `checks` is the
        #: shorter spelling other bench notes use. Reading only one silently reads none.
        declared_checks = (task.get("objective_checks") or task.get("checks") or [])
        checks = [c for c in declared_checks if c.get("type") == "find_all"]
        if not checks:
            continue
        if only is not None and tid not in only:
            continue
        check = checks[0]
        declared = len(check.get("gold_items") or [])
        kind = DERIVATIONS.get(tid)
        if kind is None:
            raise Refusal(
                f"gold mismatch: {tid} declares a find_all answer set with no derivation in "
                f"scripts/find_all_arms_run.py DERIVATIONS (which knows "
                f"{', '.join(sorted(DERIVATIONS))}). Add one rather than run 18 sessions "
                f"against an answer key nobody re-derived.")
        extra = ""
        if kind == "dead_wikilinks":
            derived: int | None = _derive_dead_wikilinks(Path(vault_root))
        elif kind == "dead_skill_paths":
            missing, skipped = _dead_skill_path_citations(Path(vault_root),
                                                          home_root=home_root)
            derived = len(missing)
            if skipped:
                #: Printed, because "the derivation dropped one" has to be checkable from the
                #: log line rather than from reading the source.
                extra = "excluded as a wrong-path example: " + "; ".join(
                    f"{skill}:{line} {key!r} under heading {head!r}"
                    for skill, line, key, head in skipped)
        elif kind == "unresolved_task_skills":
            derived = _derive_unresolved_task_skills(Path(vault_root))
        else:
            derived = _derive_fact_copies(check, store_lookup)
        #: The note's own id travels with the row, so a log line saying `gold #014` can be
        #: traced to the file that declares it without a second lookup.
        rows.append(GoldRow(tid, derived, declared, note=str(task.get("id") or ""),
                            extra=extra))
    return rows


# ── the lock (clause 1) ───────────────────────────────────────────────────────

def acquire_lock(lock_path: Path, wait_seconds: int):
    """Hold the engine lock with patience, or refuse naming the lock and the wait.

    The shared helper does the waiting (`app.atomic_io._flock_exclusive`, 20 ms polls on a
    non-blocking fd) — no sleep/poll loop is open-coded here, because a second wait loop is a
    second set of semantics about who yields to whom. Its failure mode is what this function
    exists to translate: on expiry it RAISES `TimeoutError("could not lock <path> within
    Ns")` (`app/atomic_io.py:234`) and it never returns a negative fd, so letting that
    exception escape would end the run with a traceback — a non-zero exit, but not the named
    refusal clause 1 asks for, and a reader who has to guess the cause is exactly what this
    launcher was written against.
    """
    p = Path(lock_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        return _flock_exclusive(p, float(wait_seconds))
    except TimeoutError as exc:
        raise Refusal(
            f"lock still held when the wait ended: {p} was not acquired within "
            f"{wait_seconds} s ({exc}). Another eval is holding the primary engine (an automod "
            f"round does NOT take this lock — its lock is STATE_DIR/'lock'). Refusing "
            f"rather than running unlocked and sharing the engine mid-measurement.") from exc


# ── the run ───────────────────────────────────────────────────────────────────

def resolve_out(out: str | None, *, root: Path = _REPO_ROOT) -> Path:
    """The out dir, defaulting to `eval/measurements/find-all-<UTC date>`."""
    if out:
        return Path(out)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return Path(root) / "eval" / "measurements" / f"find-all-{date}"


def resolve_argv(args, *, python: str | None = None, root: Path = _REPO_ROOT) -> list[str]:
    """The command, spelled once so what --dry-run prints IS what the run spawns.

    `args.driver` is the test seam (a stand-in that writes rows or writes nothing without
    spending eighteen harness sessions); with it unset it is the real driver, which is
    what --dry-run prints.
    """
    return [python or sys.executable,
            str(getattr(args, "driver", None) or (Path(root) / DRIVER)),
            "--trials", str(args.trials),
            "--max-turns", str(args.max_turns),
            "--timeout", str(args.timeout),
            "--out", str(resolve_out(args.out, root=root))]


# ── the published table (clauses 4 and 5) ─────────────────────────────────────

def _num(v) -> str:
    """A driver-computed number as a table cell; None (no scored trial) as a dash."""
    return "-" if v is None else f"{float(v):.2f}"


def report_path(out_dir: Path) -> Path:
    """``eval/measurements/find-all-<UTC date>.md``, beside the run dir it reports on.

    The shape the 2026-09-24 artifact has: ``find-all-2026-09-24/`` and
    ``find-all-2026-09-24.md`` side by side in ``eval/measurements/``. Clause 4 names the
    file, so it lands there and not inside the run dir — the run dir holds the raw rows.
    """
    return out_dir.with_suffix(".md")


def summarise_via_driver(out_dir: Path, driver: Path) -> dict:
    """The driver's OWN summarise, over the run's rows, at zero engine cost.

    `--summarise` already re-reads `trials.jsonl` and prints the dict but writes no file —
    the missing half was the `.md`, which is what this file adds.
    """
    proc = subprocess.run([sys.executable, str(driver), "--summarise", "--out", str(out_dir)],
                          cwd=str(_REPO_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise Refusal(
            f"the summarise step could not read {out_dir}: the driver exited "
            f"{proc.returncode}: {(proc.stderr or '')[-400:]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise Refusal(f"the summarise step returned no JSON for {out_dir}: {exc}; stdout "
                      f"began {(proc.stdout or '')[:200]!r}")


def write_report(out_dir: Path, summary: dict) -> Path:
    """Render `eval/measurements/find-all-<date>.md`: the cells table and the judge block.

    Deliberately silent on two things. It compares no arms (the #647 ruling is read off
    this table by a human; printing the verdict would read as making it), and it computes
    no score (the counts are the driver's `summarise`; P/R/F1 are means of fields the
    driver wrote).
    """
    cells = summary.get("cells") or {}
    if not cells:
        raise Refusal(
            f"no trials to summarise in {out_dir}: the table would be empty, so no report "
            f"is written — the run produced no rows to read.")
    rows: list[dict] = []
    tl = out_dir / "trials.jsonl"
    if tl.exists():
        for line in tl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [f"# find-all arms run — {out_dir.name}", ""]
    lines += [f"Written {stamp} by `scripts/find_all_arms_run.py` from "
              f"`{out_dir.name}/trials.jsonl`. Counts are the driver's own "
              "(`eval/run_find_all_arms.py --summarise`, zero engine time in this step); "
              "P/R/F1 are means over the trials that scored them. No arm is compared to "
              "another here — the ordering #647 asks about is a reading of these numbers, "
              "not a line this file computes."]
    lines += ["", "## Measurement", "",
              "| arm / task | n | n_scored | n_truncated | n_timeout | at-cap | "
              "turn budget | P | R | F1 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    #: `summarise` keys a cell `f"{arm}/{task_id}"` and already holds its n, n_scored,
    #: n_truncated, n_timeout and the score means over the ANSWERED trials only. Every
    #: number in this table is copied from it; the only thing read off the raw rows is the
    #: turn budget each cell was measured against, which summarise does not carry.
    for key in sorted(cells):
        #: `summarise` builds the key as `f"{arm}/{task['task_id']}"` — one slash, so one
        #: split. (An earlier line here asked for three names out of a 2-tuple and raised
        #: ValueError on the first real table.)
        arm, check = key.rsplit("/", 1)
        c = cells[key]
        cell_rows = [r for r in rows if r.get("arm") == arm
                     and str(r.get("task_id", "?")) == check]
        budgets = sorted({r["turn_budget"] for r in cell_rows
                          if isinstance(r.get("turn_budget"), int)})
        n = c.get("n", 0)
        lines.append(f"| {arm} / {check} | {n} | {c.get('n_scored', 0)} | "
                     f"{c.get('n_truncated', 0)} | {c.get('n_timeout', 0)} | "
                     f"{c.get('n_truncated', 0)}/{n} | "
                     f"{'/'.join(str(b) for b in budgets) if budgets else '-'} | "
                     f"{_num(c.get('precision'))} | {_num(c.get('recall'))} | "
                     f"{_num(c.get('f1'))} |")
    lines += ["",
              "`at-cap` is the share of the cell whose trials stopped AT the turn cap "
              "(`turn_truncated`, which fires at `turns >= turn_budget`). Read every recall "
              "above on top of it: on 2026-09-24 three of eight `all` trials used their "
              "whole budget and answered nothing, which confounded the phrasing gap on one "
              "of the two tasks. A cell under 3 answered trials is too thin for the clause "
              "either way.", ""]

    ji = summary.get("judge_independence") or {}
    n = ji.get("n", 0)
    rate = ji.get("disagree_rate")
    pearson = ji.get("pearson_rubric_vs_PxR")
    lines += ["## Judge independence", "",
              f"rubric vs objective (P×R) over the {n} rankable `all`/`one` trials:", ""]
    lines.append(f"- Pearson r (rubric vs P×R): "
                 f"{'-' if pearson is None else round(pearson, 3)}")
    lines.append(f"- disagreement at the 0.5 line: {ji.get('disagree_at_0.5', 0)} / {n}"
                 + (f" ({rate:.0%})" if isinstance(rate, (int, float)) else ""))
    lines += ["", "The judge is not independent of the objective score on this corpus, so a "
              "rubric gain on its own is not a win. Recorded as a measurement; re-weighting "
              "the rubric is a ruling this file does not make."]

    report = report_path(out_dir)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ── entry ─────────────────────────────────────────────────────────────────────

def _refuse(reason: str) -> int:
    """The one refusal shape: REFUSED plus the named precondition, on both streams."""
    msg = f"FIND-ALL ARMS RUN REFUSED: {reason}"
    print(msg)
    print(msg, file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Owner of the find-all arms run: waits for the primary engine lock, "
                    "re-derives the gold before spending engine time, and publishes the "
                    "table. Refuses loudly, naming the precondition, whenever nothing ran.")
    ap.add_argument("--trials", type=int, default=PINNED_TRIALS)
    ap.add_argument("--max-turns", dest="max_turns", type=int, default=PINNED_MAX_TURNS)
    ap.add_argument("--timeout", type=int, default=PINNED_TIMEOUT,
                    help="per-trial seconds; NOT the driver's 600 default — a 'find all' "
                         "sweeps the whole set and #016 needs the longer wall")
    ap.add_argument("--out", default=None,
                    help="out dir; default eval/measurements/find-all-<UTC date>")
    ap.add_argument("--wait-seconds", dest="wait_seconds", type=int,
                    default=DEFAULT_WAIT_SECONDS,
                    help=f"how long to wait for {PRIMARY_LOCK} before refusing "
                         f"(default {DEFAULT_WAIT_SECONDS} s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved command and exit 0 without touching the lock")
    ap.add_argument("--summarise-only", dest="summarise_only", metavar="OUT_DIR",
                    default=None, help="render the report from an existing run's rows")
    ap.add_argument("--bench-dir", dest="bench_dir", default=str(BENCH_DIR))
    ap.add_argument("--vault-root", dest="vault_root", default=str(VAULT))
    ap.add_argument("--home-root", dest="home_root", default=None,
                    help="what `~` expands to when resolving a skill's `~/lloyd/...` path")
    ap.add_argument("--facts-root", dest="facts_root", default=str(VAULT_FACTS),
                    metavar="PATH",
                    help="the markdown fact tree #015's copy count is walked (default: the "
                         "vault facts dir). The seam a test uses. Every find_all task on the "
                         "bench is re-derived either way: there is no flag that skips one, "
                         "because a skipped count is a green light nobody checked.")
    ap.add_argument("--lock", default=str(PRIMARY_LOCK))
    ap.add_argument("--driver", default=str(_REPO_ROOT / DRIVER),
                    help=argparse.SUPPRESS)      # test seam: stand in for the real driver
    args = ap.parse_args(argv)
    driver = Path(args.driver)

    if args.summarise_only:
        out = Path(args.summarise_only)
        try:
            report = write_report(out, summarise_via_driver(out, driver))
        except Refusal as exc:
            return _refuse(str(exc))
        print(f"report written: {report}")
        return 0

    cmd = resolve_argv(args)
    if args.dry_run:
        print(shlex.join(cmd))
        return 0

    try:
        rows = gold_checks(Path(args.bench_dir), vault_root=Path(args.vault_root),
                           home_root=Path(args.home_root) if args.home_root else None,
                           facts_root=Path(args.facts_root))
        #: All four print, all four must agree. A flag that skipped one would leave a green
        #: exit next to an answer key nobody re-derived, which is the failure this gate is for.
        for r in rows:
            print(r.render())
        bad = [r for r in rows if not r.ok]
        if bad:
            raise Refusal(
                "gold mismatch — the answer key does not match the tree, so every arm "
                "would be scored against a key nobody re-derived: "
                + "; ".join(r.render() for r in bad))
        fd = acquire_lock(Path(args.lock), args.wait_seconds)
    except Refusal as exc:
        return _refuse(str(exc))

    try:
        proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True)
    finally:
        os.close(fd)
    out_text = (proc.stdout or "") + (proc.stderr or "")

    out = Path(cmd[cmd.index("--out") + 1])
    if not (out / "trials.jsonl").exists():
        tail = (out_text.strip().splitlines() or ["(no output)"])[-1]
        #: `require_tool_sandbox` (`scripts/autoresearch/bench_runner_sdk.py:349-367`) asks the
        #: aggregator's `/state` for `tool_sandbox` and raises `ToolSandboxUnavailable` with the
        #: words "does not report an enforced read-only tool sandbox" — with a SPACE, no
        #: underscore. Matching only the key spelling `tool_sandbox` would miss it and print the
        #: generic "driver exited rc=1" reason, which is how a sandbox stop got misread as a
        #: lock stop for a week. All three vocabularies are checked, including the key itself in
        #: case a future message quotes the field name.
        sandbox_markers = ("tool_sandbox", "ToolSandboxUnavailable", "tool sandbox",
                           "does not sandbox")
        what = ("the driver's tool-sandbox precondition refused: the aggregator's /state does "
                "not report an enforced read-only bwrap sandbox (runtime state, not a property "
                "of this checkout — `require_tool_sandbox`, "
                "scripts/autoresearch/bench_runner_sdk.py:349), so a probe-free agent could "
                "edit the vault mid-run. This is NOT the primary.lock: the lock was acquired "
                "and released above."
                if any(m in out_text for m in sandbox_markers) else
                f"the driver exited rc={proc.returncode} and wrote no trials.jsonl under "
                f"{out}")
        return _refuse(f"{what}. Its own last line: {tail[:300]}")

    print(f"trials landed: {out / 'trials.jsonl'}")
    try:
        report = write_report(out, summarise_via_driver(out, driver))
    except Refusal as exc:
        return _refuse(str(exc))
    print(f"report written: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
