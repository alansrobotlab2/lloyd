"""Landing a change to the Obsidian vault through the self-modification loop.

The vault (`~/obsidian`) is not code and cannot be handled like it. It is a
separate git repo and a *live* tree that Lloyd, the nightly jobs and Alan all
write into at once — 65 files were dirty the day this was written — and
nothing reads it from a worktree: `prompt_builder` reads SOUL.md and the
skills straight from the live tree, `autonomy` reads its tasks the same way.
So there is no candidate to gate in isolation; an edit is live the moment it
is saved.

What the loop can still guarantee is the shape of its promise — nothing
lands unverified, and what lands is one commit that can be reverted alone.
The vault route is therefore validate → commit only these paths → revert on
failure:

  1. **Scope.** `.obsidian/**` (the app's own config), `.git/**` and
     `.trash/**` are denied; everything else is writable. Paths that feed a
     prompt or the scheduler — `skills/**`, `lloyd/**`, `autonomy/**` — are
     *validated*: the loaders below actually run against them.
  2. **Front matter.** Every changed `.md` that opens with `---` must parse
     to a mapping. A task file whose YAML broke is a task the scheduler
     silently drops; a skill whose front matter broke is a skill nobody can
     find.
  3. **Loaders**, for validated paths only and only for what changed: the
     system prompt must still build (SOUL.md, memories, the skills index),
     each touched skill directory must be undamaged by `agent_mcp.skills`'s own
     verdict — a skill the loader abstains on because its front matter
     quarantined it is retired, not broken, and a path under a dot-directory is
     not a skill slug at all (#777) — and each
     touched task must still parse through `autonomy`. Scoped to the diff on
     purpose — a pre-existing broken file elsewhere in the vault must not
     block every round, the same delta principle as pyflakes and tsc in the
     code gate.
     A rewritten `skills/<slug>/SKILL.md` is also scored for activation
     against the labelled corpus (#711) — recorded on every landing, and a
     refusal on the skills that corpus covers since #2148 turned
     `SKILL_ACTIVATION_ENFORCE` on.
  4. **A failure reverts the round's paths** — tracked ones back to HEAD, new
     ones deleted. The vault is live, so "nothing lands" has to mean "nothing
     stays".
  5. **Success commits exactly these paths**, on `main` (the branch guard
     mirrors `scripts/util/vault-commit.sh`, which every nightly writer
     uses), and records a `vault_land` ledger event with the sha.
     `revert(sha)` is the rollback, and it is a plain `git revert`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.automod import spec, state as S, vault_guards as VG

LLOYD_HOME = Path(__file__).resolve().parent.parent.parent
VAULT = Path(os.environ.get("LLOYD_VAULT") or (Path.home() / "obsidian"))
PYTHON = Path(sys.executable)

DENIED_GLOBS: tuple[str, ...] = (".obsidian/**", ".git/**", ".trash/**")
VALIDATED_GLOBS: tuple[str, ...] = ("skills/**", "lloyd/**", "autonomy/**")


class VaultRoundError(RuntimeError):
    pass


def classify(path: str) -> str:
    norm = spec.normalize(path)
    if norm is None or spec._match(norm, DENIED_GLOBS):
        return "denied"
    if spec._match(norm, VALIDATED_GLOBS):
        return "validated"
    return "allowed"


def check_scope(paths: list[str]) -> tuple[bool, str, dict[str, list[str]]]:
    buckets: dict[str, list[str]] = {"denied": [], "validated": [], "allowed": []}
    for p in paths:
        buckets[classify(p)].append(p)
    if buckets["denied"]:
        return False, f"denied vault paths: {sorted(buckets['denied'])}", buckets
    return True, "in scope", buckets


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(VAULT), *args],
                          capture_output=True, text=True, check=False)


def frontmatter_error(path: Path) -> str | None:
    """None if the file has no front matter or it parses to a mapping."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"unreadable: {exc}"
    if not text.startswith("---"):
        return None
    lines = text.splitlines()
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return "front matter never closes"
    try:
        fm = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError as exc:
        return f"front matter is not valid YAML: {str(exc).splitlines()[0]}"
    if not isinstance(fm, dict):
        return "front matter is not a mapping"
    return None


def knowledge_type_error(path: Path) -> str | None:
    """A `knowledge/` file whose front-matter `type` may not land, else None.

    The other writer-side half of #872. `vault_write` normalises a retired
    spelling and refuses an invented one, but a note written with the generic
    file tools reaches the tree through HERE, not through that tool — and #780
    was exactly a knowledge file whose invented `type` was noticed only when a
    promotion gate went red seven hours later. Front matter that merely *parses*
    (``frontmatter_error`` above) was never a check on the vocabulary.

    Unlike `vault_write` this refuses a retired alias rather than rewriting it:
    the lander commits files byte-for-byte, and quietly editing an author's front
    matter to make a change land is worse than one line naming the value to
    write. An absent or empty `type` is left alone — that is #478's sweep, not a
    vocabulary error.
    """
    try:
        from scripts.vault import okf_taxonomy
    except Exception as exc:  # noqa: BLE001
        # A check that cannot read its vocabulary must not report "clean" — the
        # reason vault_write fails closed too (lloyd/MEMORY.md: four instances).
        return f"cannot check the `type` vocabulary: okf_taxonomy is unimportable ({exc})"
    try:
        rejected = okf_taxonomy.rejected_document_type(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return f"unreadable: {exc}"
    if rejected is None:
        return None
    return f"{okf_taxonomy.KnowledgeTypeError(rejected)}"


_LOADER_SCRIPT = r"""
import json, sys
from pathlib import Path
paths = json.loads(sys.argv[1]); vault = Path(sys.argv[2]); errs = []
if any(p.startswith(("lloyd/", "skills/")) for p in paths):
    from app.prompt_builder import build_system_prompt
    prompt = build_system_prompt()
    if not isinstance(prompt, str) or len(prompt) < 500:
        errs.append("system prompt failed to build or came back empty")
# A path is a skill's own file only if its FIRST segment under `skills/` is the
# slug. `skills/.archived/ingest/SKILL.md` has two, and `p.split("/")[1]` there
# yields `.archived` — a slug for a tree `agent_mcp/skills.py:90` told discovery
# to ignore (the dot-prefix is exactly what makes the archive an archive). The
# loader was then asked to load that invented slug, abstained, and the rename
# into the archive could never land, taking the rest of the batch down with it
# (#777). Taking segment [1] and dropping dot-segments says the same thing the
# discovery walk already says.
skills = sorted({p.split("/")[1] for p in paths
                 if p.startswith("skills/") and p.count("/") >= 2
                 and not p.split("/")[1].startswith(".")})
# The one shape a source path may name without existing: a rename INTO an
# excluded tree. `skills/.archived/foo/SKILL.md` exists and
# `skills/foo/SKILL.md` is gone, so the skill was retired, and the loader must
# not be asked whether the empty source directory loads — that is the same
# verdict-as-damage again, and it is why the retirement had to be done by hand
# (vault `60776c12`). Stated as its own verdict rather than as an absence, so
# `skills/bar/` with no SKILL.md and no destination stays the error it is.
moved = {p.split("/")[-2] for p in paths
         if p.startswith("skills/") and p.endswith("/SKILL.md") and p.count("/") >= 3
         and any(seg.startswith(".") for seg in p.split("/")[1:-2])
         and (vault / p).is_file()}
if skills:
    # NOT `_load_skill(...) is None`: that sentinel also means "deliberately
    # quarantined by front-matter `status`", so reading it as damage made
    # `automod_vault_land` unable to retire a skill by either of the two ways the
    # skills lifecycle retires one — moving it, or setting `status: archived`
    # (#777, second half, proved in item #432's implement round). `skill_load_defect`
    # answers only "is this skill damaged"; a quarantine is not damage.
    from agent_mcp.skills import skill_load_defect
    for name in skills:
        d = vault / "skills" / name
        if not d.is_dir() or name in moved:
            continue
        defect = skill_load_defect(d)
        if defect:
            errs.append(f"skills/{name}: {defect}")
tasks = [p for p in paths if p.startswith("autonomy/") and p.endswith(".md")]
if tasks:
    from app.autonomy import _parse_task_file
    for p in tasks:
        f = vault / p
        if f.exists() and _parse_task_file(f) is None:
            errs.append(f"{p}: autonomy cannot parse it")
print(json.dumps(errs))
"""


def loader_errors(paths: list[str]) -> list[str]:
    """Run the real loaders in a fresh interpreter, scoped to `paths`.

    The child's `PYTHONPATH` is `LLOYD_HOME` and nothing else, because the whole
    claim of this check is that it ran the loaders *of the tree it was pointed
    at*. Passing no `env` inherits the caller's, and `app` is a regular package
    (it has an `__init__.py`): PEP 420 keeps scanning after a namespace portion
    and returns the first regular package it finds, so a caller whose
    `PYTHONPATH` names some other checkout beats the `LLOYD_HOME` copy sitting at
    `sys.path[0]` (cwd) and the verdict comes back for the wrong tree. The gate's
    `tests` rung does exactly that — `_child_env` sets `PYTHONPATH` to the round's
    worktree (`scripts/automod/gate.py`) — so every gate run judged a tree other
    than the one it was handed, and the two tests that aim this subprocess at a
    stub checkout by patching `LLOYD_HOME` alone (#1099's auto-restore refusal,
    the fresh-interpreter loader rung inside a rollback) read "the contract
    builds" and let a broken restore through (#1562). Pinned by
    `tests/test_automod_vault_round.py::test_the_loader_subprocess_judges_the_checkout_named_by_loyd_home`.
    """
    r = subprocess.run([str(PYTHON), "-c", _LOADER_SCRIPT, json.dumps(paths), str(VAULT)],
                       cwd=str(LLOYD_HOME), capture_output=True, text=True, timeout=180,
                       env={**os.environ, "PYTHONPATH": str(LLOYD_HOME)})
    if r.returncode != 0:
        return [f"loader crashed: {(r.stdout + r.stderr).strip()[-600:]}"]
    try:
        return list(json.loads(r.stdout.strip().splitlines()[-1]))
    except (ValueError, IndexError):
        return [f"loader produced no verdict: {(r.stdout + r.stderr).strip()[-300:]}"]


#: Every loaded prompt file this route can commit. `VALIDATED_GLOBS` already
#: classifies `lloyd/USER.md` as validated, so the round could write it; until #1010
#: this tuple did not name it, which meant a diff whose only change was
#: `lloyd/USER.md` — the largest of the three, the one #507 found at 95,302 B —
#: reached no prompt-surface check at all. Naming a file here is what makes the
#: scoping guard below run the check on it; the check itself is `prompt_surface`'s.
CONTRACT_PATHS = ("lloyd/SOUL.md", "lloyd/MEMORY.md", "lloyd/USER.md")


def contract_errors(paths: list[str]) -> list[str]:
    """Prompt-surface invariants, when the change touches the identity files.

    The loaders below already answer "does the prompt still build". They do
    not answer "is it still the shape #377 left it in", and that is the
    question a writer has to answer, because the reader cannot: the gate's
    `tests` rung judges a candidate commit and `SOUL.md` is not in it.

    Scoped to a diff that names one of the contract files, like every other
    check here — a pre-existing condition elsewhere in the vault must not
    block an unrelated round. The scope is by *name*, and all three files are
    then read: a round that rewrites MEMORY.md is also the round that can push
    the total past a ceiling by growing the file it did not touch, and a check
    that only looked at the touched file would be measuring a document nobody
    is about to load.
    """
    if not any(p in CONTRACT_PATHS for p in paths):
        return []
    try:
        from app import prompt_surface
    except ImportError as exc:  # pragma: no cover - repo is always importable
        return [f"prompt_surface unavailable, cannot check the contract: {exc}"]
    errs = prompt_surface.check_paths(
        VAULT / "lloyd" / "SOUL.md",
        VAULT / "lloyd" / "MEMORY.md",
        VAULT / "lloyd" / "USER.md",
    )
    return [f"prompt surface: {e}" for e in errs]


def reflection_archive_errors(paths: list[str]) -> list[str]:
    """#436: a touched skill must still archive a reflection report before it
    overwrites it.

    `skills/**` is the only tree this reads, and only the ones the round names —
    the same scoping `contract_errors` argues for. The loaders answer "does the
    skill still load"; a skill that instructs an in-place overwrite of
    `_pipeline/reflection/<name>-latest.md` with no prior dated copy loads
    perfectly and destroys a report nobody can recover, because `_pipeline/` is
    gitignored. That question has to be answered by the writer, which is this
    function: `tests/test_skill_reflection_archive.py` marks its live-vault
    assertions `live_vault` precisely because the gate's hard `tests` rung is the
    wrong place for an invariant about a tree no round under test controls, and
    an unmarked version of them would fail the next author for the previous
    writer's wording. Here the invariant runs on the path that actually lands.
    """
    skills = sorted({
        p for p in paths
        if p.startswith("skills/") and p.endswith("/SKILL.md") and (VAULT / p).exists()
    })
    if not skills:
        return []
    try:
        from scripts import reflection_archive
    except ImportError as exc:  # pragma: no cover - repo is always importable
        return [f"reflection_archive unavailable, cannot check report retention: {exc}"]
    errs: list[str] = []
    for p in skills:
        body = (VAULT / p).read_text(encoding="utf-8", errors="replace")
        for e in reflection_archive.skill_rule_violations(p.split("/")[1], body):
            errs.append(f"{p}: reflection report retention: {e}")
    return errs


def skill_timezone_errors(paths: list[str]) -> list[str]:
    """#1189: a touched skill template must not hand-type PST/PDT beside a
    displayed time.

    Same shape and same scoping as `reflection_archive_errors` directly above,
    for the same reason: skill prose is state no round under test controls, so
    the invariant that must *stop* a bad template belongs at the writer, and
    the live-vault scan in `tests/test_skill_timezone_literals.py` is the
    reporting copy. The rule itself is `scripts/skill_timezone.py` — one
    definition, shared by this call site and that test. The class this refuses
    (a season's zone abbreviation typed into a template that the clock would
    print differently for half the year) recurred five times (#601, #1079,
    #1080, #1081, #1112) precisely because nothing on the landing path could
    fail when a template re-typed one.
    """
    skills = sorted({
        p for p in paths
        if p.startswith("skills/") and p.endswith("/SKILL.md") and (VAULT / p).exists()
    })
    if not skills:
        return []
    try:
        from scripts import skill_timezone
    except ImportError as exc:  # pragma: no cover - repo is always importable
        return [f"skill_timezone unavailable, cannot check clock literals: {exc}"]
    errs: list[str] = []
    for p in skills:
        body = (VAULT / p).read_text(encoding="utf-8", errors="replace")
        for e in skill_timezone.template_clock_violations(p.split("/")[1], body):
            errs.append(f"{p}: skill clock literal: {e}")
    return errs


# Enforcing since #2148 (ruled 2026-10-04). What #711's human clause asked for
# before a flip — real consolidation runs showing what this would have refused —
# is the ledger's own `skill_gate` block: 137 rows from 2026-09-25T01:22:44Z to
# 2026-10-03T23:26:57Z, 0 of them would-refusals, committed as witness bytes at
# vault bb3ed6f0. Nine days of landings, nothing it would have blocked.
# Enforcement still reaches only what the corpus can measure — the five slugs
# carrying a `recall_floor` in `eval/skill_activation_cases.yaml`. An uncovered
# skill, an unchanged body, an improved body and a gate that cannot run each
# still land. A refusal names every offending skill and reverts the whole batch:
# #2148 rules that house behaviour, not a stall.
SKILL_ACTIVATION_ENFORCE = True


def skill_activation_findings(paths: list[str]) -> list[dict]:
    """#711: one row per touched `skills/<slug>/SKILL.md` — does the rewrite
    trigger falsely more often, or push recall under the skill's floor?

    The third per-skill check on the landing path after the two above. Its rule
    (`scripts/skill_activation.py`) compares the text on disk against the vault's
    HEAD through the production matcher over the labelled corpus, and a skill
    without an entry in `eval/skill_activation_cases.yaml` has nothing to
    compare, so it is never blockable however the flag is set — enforcement is
    bounded by the corpus, not by a list here. `validate()` refuses on a
    would-refuse row since #2148; `land()` records every row on the passing row
    and on the refusal row alike, so the rail is legible from the ledger either
    way. Never raises.
    """
    slugs = sorted({p.split("/")[1] for p in paths
                    if p.startswith("skills/") and p.endswith("/SKILL.md")
                    and p.count("/") == 2})
    if not slugs:
        return []
    try:
        from scripts import skill_activation
    except ImportError as exc:  # pragma: no cover - repo is always importable
        return [{"skill": s, "has_eval": False, "would_refuse": False,
                 "reason": f"skill_activation unavailable: {exc}"} for s in slugs]
    rows = []
    for slug in slugs:
        rel = f"skills/{slug}/SKILL.md"
        f = VAULT / rel
        candidate = f.read_text(encoding="utf-8", errors="replace") if f.exists() else None
        head = _git("show", f"HEAD:{rel}")
        current = head.stdout if head.returncode == 0 else None
        rows.append(skill_activation.gate(slug, candidate, current))
    return rows


# #1985 shipped this log-only; #2158 ruled it enforcing on 2026-10-04, on what
# the log caught: two real over-cap landings while the row only recorded — a
# 103-line body on 2026-10-03T08:48:22Z (commit 98823179, `would_refuse: true`,
# landed anyway; hand-fixed by 9b7b8986) and #1534 at 116 lines whose only red
# node was a `live_vault` check the gate deselects. The scope is
# `skill_lint.SPILL_SAMPLE` only: library-wide enforcement stays a person's
# call, and `architecture/skills.md` records both halves of that ruling.
SKILL_BODY_ENFORCE = True


def skill_body_findings(paths: list[str]) -> list[dict]:
    """#1985: one row per touched spill-sampled `skills/<slug>/SKILL.md` whose
    body is past `skill_lint.MAX_BODY_LINES`.

    The ceiling's only failing check was a `live_vault` node the gate
    deselects, so a landing could push a spilled skill back over it and nothing
    on the landing path could say so. Scoped to `skill_lint.SPILL_SAMPLE` on
    purpose: 106 of 197 live skills were over the cap on 2026-10-01, so a
    library-wide rule would refuse most landings; a touched skill outside the
    sample yields no row. Body = the text after front matter, the same rule as
    `skill_lint.skill_size`.

    The ceiling measures the candidate's resulting state, never its delta: a
    landing that SHRINKS a sampled skill — 120 lines cut to 101 — is refused,
    because after the cut the file still sits past `MAX_BODY_LINES`, while one
    that lands at <= 100 lines is allowed however much it grew. Nothing here
    compares against HEAD.

    Each row carries `largest_block` (heading + line count, from
    `skill_lint.skill_size`, whose `_largest_block` names it the first spill
    candidate) and says so in `reason`, so a refusal tells the writer which
    section to move into a sibling file. `land()` records every row — on the
    passing landing and on the refusal alike; `validate()` refuses on a
    would-refuse row since #2158 flipped `SKILL_BODY_ENFORCE` on. Never raises.
    """
    slugs = sorted({p.split("/")[1] for p in paths
                    if p.startswith("skills/") and p.endswith("/SKILL.md")
                    and p.count("/") == 2})
    if not slugs:
        return []
    try:
        from scripts import skill_lint
    except Exception:  # noqa: BLE001 — an advisory check never blocks a landing
        return []
    rows = []
    for slug in slugs:
        if slug not in skill_lint.SPILL_SAMPLE:
            continue
        f = VAULT / "skills" / slug / "SKILL.md"
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
            size = skill_lint.skill_size(slug, f, content, skill_lint.parse_frontmatter(content)[1])
        except Exception:  # noqa: BLE001 — missing (a deletion) or unreadable: nothing to measure
            continue
        if size["over_cap"]:
            block = size["largest_block"]
            rows.append({"skill": slug, "body_lines": size["body_lines"],
                         "max_body_lines": skill_lint.MAX_BODY_LINES, "would_refuse": True,
                         "largest_block": block,
                         "reason": f"body is {size['body_lines']} lines, past the "
                                   f"{skill_lint.MAX_BODY_LINES}-line ceiling for a "
                                   f"spill-sampled skill; clear it by spilling the "
                                   f"largest block ({block['lines']} lines: "
                                   f"{block['heading']!r}) into a sibling file and "
                                   f"naming it in the body's index"})
    return rows


def _front_matter_map(text: str) -> dict | None:
    """The front matter of `text` as a mapping, or None (never raises).

    #2190: the dispatch-field check below needs the parsed VALUES, where
    `frontmatter_error` above answers only "does this parse". Recovery is the
    scheduler's own (`parse_frontmatter_text`, which is what
    `app.autonomy._parse_task_file` calls), so a file whose YAML is corrupt but
    field-recoverable is judged on the fields the scheduler would actually read
    instead of being waved through as "unparseable, not my business".
    """
    if not text.startswith("---"):
        return None
    lines = text.splitlines()
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return None
    try:
        from agent_mcp._shared import AUTONOMY_TASK_FIELDS, parse_frontmatter_text
        fm = parse_frontmatter_text("\n".join(lines[1:end]),
                                    fallback_fields=AUTONOMY_TASK_FIELDS,
                                    log_label="vault_round:#2190")
    except Exception:  # noqa: BLE001 - a reader that cannot read is not a refusal
        return None
    return fm if isinstance(fm, dict) else None


def _same_dispatch_value(old, new) -> bool:
    """Whether two front-matter values are the same value, YAML spellings aside.

    PyYAML resolves an unquoted ISO timestamp to `datetime`, so quoting a
    `scheduled_at` that HEAD holds unquoted is a reformat, not a schedule change.
    Comparing serialized YAML instead of values is exactly what the owed-after-
    landing measurement on #2190 watches for: it would refuse the nightly writer's
    own rewrites.
    """
    import datetime  # local: a top-of-file import would shift a cited line

    if old == new:
        return True
    # `datetime.date` on purpose: `datetime.datetime` is its subclass, and a
    # `scheduled_at` of `2026-10-06` resolves to the date, not the datetime.
    for a, b in ((old, new), (new, old)):
        if isinstance(a, (datetime.date, datetime.time)) and isinstance(b, str):
            if a.isoformat() == b.strip():
                return True
    return False


def _shown(value) -> str:
    """A value as it appears in a refusal. Total: a `datetime` front matter value
    (`scheduled_at` unquoted) must not raise on the way to being refused."""
    return "absent" if value is None else (
        value if isinstance(value, str)
        else json.dumps(value, sort_keys=True, default=str))


def schedule_state_errors(paths: list[str]) -> list[str]:
    """Refuse a round that moves an autonomy task's dispatch state (#2190).

    The #724 grant rail (`app/harness/policy.py`) refuses an unattended
    `autonomy_write_task` call that changes a field the scheduler reads to decide
    whether to run a task. A round could always write the same field into
    `autonomy/*.md` and land it through HERE: `validate()` re-parsed a touched task
    file and asked nothing of its values. The route is proved live —
    `autonomy/92-vllm-prefix-miss-daily.md` landed at 2026-10-01T18:51:05Z, eighteen
    seconds after the rail denied that same round's tool call
    (`~/lloyd-data/safety/denials.jsonl`; the witness copy is
    `backlog/data/denials.jsonl`), and that commit's own message records the
    refusal it routed around. The vault route stays the way to write a task file;
    only the dispatch-affecting fields are held to the tool's rule.

    Four things this deliberately is not:

    * **Not presence-based.** Every task file already carries `status:`, so
      refusing a touched file that merely has one would refuse every autonomy
      round. The baseline is `HEAD:<path>` — the same read
      `skill_activation_findings` does — and only a changed VALUE refuses, which is
      what lets a round rewrite a description or append an activity note.
    * **Not a byte comparison.** Values are parsed (`_front_matter_map`), so
      re-quoting or reordering front matter is not a dispatch change.
    * **Not the tool gate's over-refusal.** `policy.schedule_fields_changed` reads
      the call and never the disk, so a resend of `up_next` to an already-armed
      task is denied there; this route can read HEAD, so landing the file with that
      value unchanged moves nothing and passes. Copying the over-refusal would
      deny the nightly writers that rewrite task files every cycle.
    * **Not a gate on deletions.** A round that removes a task file is not refused
      here, because the lifecycle retires a task by moving it out of the live set
      and #777 is the record of what a lander made unable to retire anything does:
      it stops retiring. That half of the door is named open on item #2190 rather
      than closed by a refusal that breaks a documented route.
    """
    # Imported here, never copied: the #724 field set has one definition, in
    # `app/harness/policy.py`, so a field added there is refused by this route with
    # no edit to this file, and tests/test_automod_vault_round.py parametrises over
    # the frozenset itself to pin that from the consumer's side. Function-local for
    # the same reason `app.autonomy` is imported locally elsewhere in this module:
    # a loaded memory note cites `vault_round.py:237` by line number, and
    # test_prompt_surface_budget.py fails the round whose diff moves it.
    from app.harness.policy import DISPATCHING_STATUS, SCHEDULE_STATE_FIELDS

    errors: list[str] = []
    for p in sorted(set(paths)):
        if not p.startswith("autonomy/") or not p.endswith(".md"):
            continue  # `status` is a backlog item's field too; the scheduler reads only these
        f = VAULT / p
        if not f.exists():
            continue  # a deletion: see the docstring
        try:
            new_fm = _front_matter_map(f.read_text(encoding="utf-8")) or {}
        except OSError:
            continue  # `frontmatter_error` reports an unreadable file; do not double-book
        shown = _git("show", f"HEAD:{p}")
        if shown.returncode != 0:
            # A create. Mirror policy's create branch: of the dispatch fields, only
            # this status dispatches, so a draft with its skill and window set is
            # the nightly chain's hand-down shape and must stay accepted.
            if str(new_fm.get("status") or "").strip() == DISPATCHING_STATUS:
                errors.append(
                    f"{p}: creating it `status: {DISPATCHING_STATUS}` arms a task, which "
                    f"the #724 grant rail refuses unattended on autonomy_write_task — "
                    f"create it `status: draft` and let a human, or a granted call, arm it")
            continue
        old_fm = _front_matter_map(shown.stdout) or {}
        for field in sorted(SCHEDULE_STATE_FIELDS):
            old, new = old_fm.get(field), new_fm.get(field)
            if _same_dispatch_value(old, new):
                continue
            errors.append(
                f"{p}: `{field}` moved {_shown(old)} -> {_shown(new)}: a dispatch-affecting "
                f"field, which the #724 grant rail refuses unattended on "
                f"autonomy_write_task, and this route holds to the same rule — land the "
                f"rest, and let a human or a granted call move `{field}`")
    return errors


def validate(paths: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """(errors, buckets). Empty errors means the change may land."""
    ok, why, buckets = check_scope(paths)
    if not ok:
        return [why], buckets
    errors: list[str] = []
    for p in paths:
        f = VAULT / p
        if f.exists() and f.suffix == ".md":
            err = frontmatter_error(f)
            if err:
                errors.append(f"{p}: {err}")
            elif p.startswith("knowledge/"):
                # Parsing front matter was never a check on its vocabulary (#872).
                terr = knowledge_type_error(f)
                if terr:
                    errors.append(f"{p}: {terr}")
    if not errors:
        # A task file whose front matter parses is not a task file whose schedule is
        # unchanged: the #724 rail's second door closes here (#2190).
        errors.extend(schedule_state_errors(paths))
    if not errors:
        errors.extend(contract_errors(paths) + reflection_archive_errors(paths)
                      + skill_timezone_errors(paths))
    if not errors and buckets["validated"]:
        errors.extend(loader_errors(buckets["validated"]))
    if not errors and SKILL_ACTIVATION_ENFORCE:
        errors.extend(f"skills/{r['skill']}/SKILL.md: skill activation: {r['reason']}"
                      for r in skill_activation_findings(paths) if r.get("would_refuse"))
    if not errors and SKILL_BODY_ENFORCE:
        errors.extend(f"skills/{r['skill']}/SKILL.md: skill body-line ceiling: {r['reason']}"
                      for r in skill_body_findings(paths))
    return errors, buckets


def revert_paths(paths: list[str]) -> list[str]:
    """Put the round's paths back: tracked ones to HEAD, new ones removed."""
    undone: list[str] = []
    for p in paths:
        tracked = _git("cat-file", "-e", f"HEAD:{p}").returncode == 0
        if tracked:
            _git("checkout", "HEAD", "--", p)
            undone.append(p)
        elif (VAULT / p).exists():
            (VAULT / p).unlink()
            undone.append(p)
    return undone


def _ensure_main() -> None:
    branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch == "main":
        return
    # Same policy as vault-commit.sh (#341): nothing commits to a stranded
    # experiment branch. Plain checkout, not -f: if main cannot be reached
    # without discarding work, that is a human's problem, not this round's.
    r = _git("checkout", "main")
    if r.returncode != 0:
        raise VaultRoundError(f"vault HEAD is on {branch!r} and main cannot be checked out: "
                              f"{r.stderr.strip()[:200]}")


# The review grader for vault rounds, or None. A vault round has no worktree
# and no gate ladder — the edit is live the moment it is saved — so the
# second reader runs here, between validation and `git add`. Set by the
# aggregator (`agent_mcp/automod.py`) to `review.grade_vault`; None in tests
# and from the CLI, where a missing grader records `review: skipped` rather
# than reaching for the network.
GRADER = None
VAULT_REVIEW_MAX = 2


def _ledger_clause(c: dict) -> dict:
    """One graded clause, as the promotions ledger records it.

    Three keys travel, each for a different reason. `clause` and `verdict` are the
    grading. `subject: landing` is how `land()` finds the clauses it overwrites with the
    verdict it derives from the sha. `accepted` is the record of a rail that was WAIVED —
    on a deletion clause, the evidence of absence the grader was allowed to cite — and it
    has to reach the row because on this surface the `vault_review` / `vault_land` rows
    are the only account a landing has: #2038 is readable at all today only because its
    refusal named the rail, and a `met` earned from a file's absence is otherwise
    indistinguishable in the ledger from one earned from a file. A waiver that leaves no
    trace is a rail that quietly stopped existing, which is what #2040 exists to prevent.
    """
    row = {"clause": int(c["clause"]), "verdict": str(c["verdict"])}
    if c.get("subject"):
        row["subject"] = str(c["subject"])
    if c.get("accepted"):
        row["accepted"] = c["accepted"]
    return row

def _vault_review(norm: list[str], item_id: int,
                  attempt: int = 1) -> tuple[str, str, list[dict]]:
    """`(kind, findings, clauses)` from the grader over the staged diff. Never
    raises; an unusable grader is `("skipped", why, [])` and the landing
    proceeds — a vault edit is already validated through the real loaders,
    and a grader outage must not hold every skill edit hostage.

    `attempt` is which grading of this round this is, and it goes to the grader
    (#1868): the seam-severity decision inside `grade_vault` is a function of it,
    so a round graded on its second attempt has to be decided as attempt 2.
    `land()` passes the same count it would write to the ledger for a refusal,
    which is the only reason that number existed before this parameter — the
    grader itself was told nothing and fell back to 1.

    On a `skipped` the `findings` slot is the REASON, and every cause has its
    own wording: no grader wired, the grader raising, the grader not answering,
    an unusable object, the surface not being `vault`, the item having no
    clauses. `land()` writes it to the ledger. Before #955 the exception label
    was a bare `RuntimeError: engine gone`, which read as engine noise rather
    than the reviewer being absent, and the abstentions all collapsed to one
    word.

    `clauses` is the grader's per-clause verdicts (`review.grade_vault`). A
    grader answering the older two-element shape reads as none graded; a row
    marked `subject: landing` says the clause is about the commit this landing
    is about to produce, which is `land()`'s to grade."""
    if GRADER is None:
        # "not consulted", spelled so it cannot be read as a grader that failed.
        # The module CLI and `scripts/autoresearch/promote.py` are the callers
        # this is: neither wires a grader, so every one of their landings is
        # reviewed by nobody (#955's merged finding: one bare word covered both).
        return ("skipped",
                "no grader configured: this caller never wires one "
                "(module CLI or autoresearch promote — the reviewer was not consulted)", [])
    try:
        # `-M`: a rename already staged with `git mv` is shown as one, so the
        # reader grades the move the committer commits (#1360). The new-file
        # fallback covers only what git cannot see — an UNTRACKED path, the
        # plain-`mv` destination — or a staged destination would appear twice.
        diff = _git("diff", "-M", "HEAD", "--", *norm).stdout
        for p in norm:
            if (_git("cat-file", "-e", f"HEAD:{p}").returncode != 0 and (VAULT / p).exists()
                    and not _in_index(p)):
                diff += f"\n+++ new file {p}\n" + (VAULT / p).read_text(encoding="utf-8", errors="replace")
        res = tuple(GRADER(item_id=item_id, paths=norm, diff=diff, attempt=attempt))
        graded = res[2] if len(res) > 2 else []
        clauses = [_ledger_clause(c) for c in (graded or []) if isinstance(c, dict)
                   and str(c.get("clause", "")).isdigit() and c.get("verdict")]
        return str(res[0]), str(res[1]), clauses
    except Exception as exc:  # noqa: BLE001 — the grader never fails a landing on its own
        return ("skipped", f"grader raised {type(exc).__name__}: {str(exc)[:200]}", [])


def _in_index(path: str) -> bool:
    return _git("ls-files", "--error-unmatch", "--", path).returncode == 0


def _stageable(norm: list[str]) -> list[str]:
    """The named paths `git add` can still match: on disk or in the index.

    A source path whose rename was already staged with `git mv` is in neither,
    and naming it made `git add -A` die with `pathspec did not match` — AFTER
    the review had passed, so the land never committed (#1360, #409). Its
    deletion is already in the index, so dropping it from the pathspec loses
    nothing. The plain-`mv` destination is untracked but on disk, so it stays.
    A path in neither that was never staged away is caught after the commit,
    by the rename-aware landing check, which grades it `unmet` by name.
    """
    return [p for p in norm if (VAULT / p).exists() or _in_index(p)]


def _committed_paths(sha: str) -> set[str]:
    """Every path a commit touched, both sides of a rename or copy.

    `git show --name-only` prints only a rename's destination under git's
    default rename detection, so a clean archive move graded its own source
    path `unmet` (#1360). `--name-status` carries the `R<score>` pair.
    """
    out: set[str] = set()
    for line in _git("show", "--name-status", "--format=", sha).stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            out.update(p.strip() for p in parts[1:] if p.strip())
    return out


def _landing_clause_indices(item_id: int) -> list[int]:
    """Which of this item's acceptance clauses have the landing as their subject.

    The grader marks the same clauses on its own rows (`subject: landing`); this
    is the belt to that brace, and it is the only one that exists on the skipped
    path where no rows come back to be marked. A contract that cannot be read
    grades nothing, so the whole call is swallowed: a missing backlog file is
    never a reason to stop a landing. The rule itself is one function,
    `review.landing_clause_indices`, shared by both callers.
    """
    try:
        from scripts.automod import review as RV
        return RV.landing_clause_indices(RV.item_contract(int(item_id))["clauses"])
    except Exception:  # noqa: BLE001 — no contract, no landing clauses
        return []


def _vault_review_attempts(item_id: int) -> int:
    """Blocking vault reviews for this item since its implement turn started."""
    events = S.read_events(limit=500)
    started = 0.0
    for e in events:
        if (e.get("event") == "backlog_implement" and e.get("phase") == "started"
                and e.get("item_id") == item_id):
            started = float(e.get("ts") or 0)
    return sum(1 for e in events if e.get("event") == "vault_review" and e.get("blocking")
               and e.get("item_id") == item_id and float(e.get("ts") or 0) >= started)


def _guards_row(guards: dict) -> dict:
    """What goes on the ledger about one code-agreement probe.

    Carried on the refusal row, the pass row and the not-judged row alike, because
    the three are one measurement read three ways, and a row that names only the
    refusal cannot answer the question the owed-check job asks once this ships
    ("did the first real mixed-surface land get a validated pass or a refusal?").
    `state` and `refuse` are on every row: a pass that says nothing about the probe
    is indistinguishable from a probe that never ran, which is the #1691 failure
    shape. The detail keys — `nodes`, `reason`, the two run summaries — are written
    only when there is something to say, so an absent key means "nothing reported"
    and never "reported as nothing", the convention `review_findings` below uses
    for #1868.

    The cost keys are what #2042 added, and they are the row's answer to its own
    `skipped`: `seconds` is the whole probe and `lock_wait_s` how much of it was
    queueing behind the gate's `tests` rung, while each run carries the seconds it
    actually ran and how many vault-reading files it ran over. Before this the
    projection dropped both, and a `timed out after 300s` row could not be told
    apart from a hang: three lands (#2040, #2027, #2038) recorded `ran=0` with an
    empty reason and no witness of the run at all. `excerpt` is the pytest tail the
    pipe held when the child was killed.

    The three parallelism keys are #2046. Since #2044 the probe decides its own
    worker count and re-asks a parallel failure serially before it may refuse
    anything, and `agreement` reports all of it (`vault_guards.py:717`, `:831`,
    `:844`) — but the projection dropped every one of those keys, so ledger row
    29142 (`ts` 2026-10-02T06:33:44Z, commit 88bab697) carries
    `candidate.failed=1, refuse=false` and states that the one failure was dismissed
    as load ONLY in its `reason` prose: a reader could not tell that row's parallel
    probe from a serial one, nor a flaker that survived the re-ask from a real
    refusal. `workers` is the count the proposed-vault run got (1 when the probe
    settled on serial, since `agreement` names the count before it decides),
    `parallel_retry` is the serial re-ask's own cost — `failed` is a node COUNT like
    the two run blocks beside it, and `workers` is 1 because the re-ask is serial by
    construction at `vault_guards.py:829`, which is the whole point of it — and
    `parallel_only_failures` names the nodes the re-ask cleared, capped at the same
    10 as `nodes` above with the true total beside it when the cap bites. A
    `parallel_retry` with `ran: 0` is the re-ask that answered nothing
    (`vault_guards.py:836-839`), which is why `parallel_only_failures` can be absent
    beside a present re-ask.
    """
    out: dict = {"state": guards.get("state", "skipped"),
                 "refuse": bool(guards.get("refuse"))}
    if guards.get("seconds") is not None:
        out["seconds"] = guards["seconds"]
    if guards.get("lock_wait_s"):
        out["lock_wait_s"] = guards["lock_wait_s"]
    if guards.get("workers") is not None:
        # A serial probe states `workers: 1` rather than dropping the key: absence
        # here means the probe never reached its worker decision (`vault_guards.py`
        # returns before :717 on a bad HEAD or an empty selection), which is a
        # different non-answer from "decided, and decided serial".
        out["workers"] = guards["workers"]
    cand = guards.get("candidate") or {}
    if cand:
        out["candidate"] = {"ran": cand.get("ran", 0), "failed": len(cand.get("failed") or [])}
        for k in ("seconds", "files"):
            if cand.get(k) is not None:
                out["candidate"][k] = cand[k]
    base = guards.get("baseline") or {}
    if base:
        out["baseline"] = {"ran": base.get("ran", 0), "failed": len(base.get("failed") or [])}
        for k in ("seconds", "files"):
            if base.get(k) is not None:
                out["baseline"][k] = base[k]
    ack_row = guards.get("ack")
    if ack_row and (ack_row.get("requested") or ack_row.get("accepted")
                    or ack_row.get("unmatched")):
        # The attempt, what it bought, and what it could not buy — all three, because an
        # excuse that is only visible when it succeeds is indistinguishable from a probe
        # that quietly agreed. `unmatched` entries are the ones naming a path this land does
        # not declare: recorded, void, and readable on the row that was refused or landed.
        out["ack"] = {"requested": list(ack_row.get("requested") or []),
                      "accepted": list(ack_row.get("accepted") or []),
                      "unmatched": list(ack_row.get("unmatched") or [])}
    if guards.get("excused"):
        # Every node the ack spoke for, by id. This is the clause: an excused failure is
        # never silent, so a later reader of the ledger can see exactly which code-side
        # guard was told to move with the bytes, and go re-measure it.
        out["excused"] = list(guards["excused"])
    retry = guards.get("parallel_retry")
    if retry:
        # Written only when a re-ask ran, so an absent key means the probe never
        # re-asked and never "re-asked and reported nothing" — the same convention
        # as the two run blocks above. `note`, `seconds` and `files` stay off the
        # row on purpose: the note's content is already in `reason`, and the row is
        # the shape a clause put there, not a dump of the report.
        out["parallel_retry"] = {"ran": retry.get("ran", 0),
                                 "failed": len(retry.get("failed") or []),
                                 "workers": retry.get("workers") or 1}
    dismissed = guards.get("parallel_only_failures")
    if dismissed:
        out["parallel_only_failures"] = list(dismissed)[:10]
        if len(dismissed) > 10:
            out["parallel_only_failures_count"] = len(dismissed)
    if guards.get("nodes"):
        out["nodes"] = list(guards["nodes"])[:10]
    if guards.get("reason"):
        out["reason"] = str(guards["reason"])[:400]
    if guards.get("excerpt"):
        out["excerpt"] = str(guards["excerpt"])[-200:]
    return out


#: The head substring of every "staged nothing" refusal, and the reason the phrases
#: appended below are additions and never a replacement: `scripts/autoresearch/
#: promote.py` classifies a repeated rollback as `no_change: true` by matching this
#: text in the exception (`promote.py`'s `if "nothing to commit" in str(exc)`), so a
#: refusal that stopped carrying it would answer a re-requested undo as a failed one.
NOTHING_TO_COMMIT = "nothing to commit on those paths"


def _held_by_head(paths: list[str]) -> dict[str, str]:
    """`path -> short sha`, for the named paths HEAD's history already holds, where the
    sha is the commit that last committed that path: `git log -1 --format=%h -- <path>`.

    A land that stages nothing is not always a land that changed nothing. A concurrent
    job's pre-flight snapshot commits whatever is dirty at its instant — `vault-commit.sh
    ` with no pathspec, its documented whole-tree snapshot mode — and takes another
    job's in-flight file with it, under the OTHER job's `--author` and `Job:` trailer.
    The author's own land then answers "nothing to commit on those paths", which is true
    of the index and silent about who took the file. This is the lookup that makes that
    answer point at a sha a reader can go and read: since #1070/#1867 a swept commit
    discloses the paths it took in its own body, so the sha is the pointer to the
    disclosure.

    A path git has never committed yields NO entry — `git log` on it is empty — and
    naming a sha for it would send the next reader to a commit that does not exist
    (#2175 clause 4). A git error or a timeout is absence too, never a guess.
    """
    held: dict[str, str] = {}
    for rel in paths:
        try:
            r = _git("log", "-1", "--format=%h", "--", rel)
        except Exception:
            continue
        out = (r.stdout or "").strip()
        if r.returncode == 0 and out:
            held[rel] = out.splitlines()[0].strip()
    return held


def _no_change_refusal(norm: list[str], held: dict[str, str]) -> str:
    """The refusal text for "staged nothing", extended with the commits that already
    hold these paths. Says what the shas ARE and not which of the two readings applies:
    either this job wrote the content and somebody else's snapshot committed it, or
    there was no change to make, and the tree looks identical either way.
    """
    lacking = sorted(set(norm) - set(held))
    if not held:
        return (f"{NOTHING_TO_COMMIT}: the working tree matches HEAD for every named "
                f"path, and none of them has a commit in HEAD's history — {norm[:5]}. "
                "A path git has never committed is not someone else's taking; this "
                "land had nothing to write.")
    named = ", ".join(f"{rel} -> {held[rel]}" if rel in held else f"{rel} -> (none)"
                      for rel in sorted(norm))
    return (f"{NOTHING_TO_COMMIT}: the working tree matches HEAD for every named path, "
            f"and HEAD already holds them from these commits — {named[:600]}. If this "
            "job authored that content, a commit not its own took it: read that sha's "
            "body for its `unattributed dirty state` / LLOYD_JOB_WRITES disclosure "
            "(#2175)."
            + (f" With no commit named for it, and none invented: {lacking[:5]}."
               if lacking else ""))


def land(paths: list[str], message: str, *, item_id: int | None = None,
         session_id: str | None = None, ack: list[str] | None = None) -> dict:
    """Validate these paths, commit exactly them on the vault's main, ledger it.

    `session_id` is the calling turn's session and it goes on the `vault_land`
    row. The tool handler writes it (`agent_mcp/automod.py`) from the session the
    harness stamped into the request's `_meta` — never from the caller's
    arguments — because `item_id` is optional at that boundary, and a landing
    whose row carries neither an item nor a session belongs to no one: the
    implement reconciler then reads a vault round that really landed as one that
    did not (`scripts/automod/backlog.py:round_landing_rows`).
    """
    norm = []
    for p in paths:
        n = spec.normalize(p)
        if n is None:
            raise VaultRoundError(f"not a safe vault-relative path: {p!r}")
        norm.append(n)
    if not norm:
        raise VaultRoundError("no paths given")
    if not (message or "").strip():
        raise VaultRoundError("a commit message is required")

    errors, buckets = validate(norm)
    if errors:
        # The rail that refused a landing has to be named on the row that records
        # the refusal (#2148). Until now the activation gate ran only *after*
        # validation passed, so a refusal of a rewritten skill named itself in
        # prose `errors` alone while the passing row carried the whole
        # `skill_gate` block — the ledger could not answer "which skill, on what",
        # which is precisely the question a refusal exists to answer. Computed
        # BEFORE `revert_paths`, and that order is the change: once the tree is
        # back at HEAD the same call reports "no regression", so a refusal row
        # written afterwards would state the absence of the thing it is refusing.
        # Recomputed here rather than threaded out of `validate()`, which keeps
        # the `(errors, buckets)` shape its other callers — this module's CLI and
        # the five test nodes in three other files — depend on. The body-line
        # ceiling is recomputed here for the same reason since #2158 flipped it
        # enforcing: after `revert_paths` the sampled skill is back at its
        # under-cap HEAD text (or deleted), so a row written afterwards would
        # carry no `skill_body` finding on the very refusal that measured one.
        refused_skill_gate = skill_activation_findings(norm)
        refused_skill_body = skill_body_findings(norm)
        undone = revert_paths(norm) if not buckets["denied"] else []
        S.append_event({"event": "vault_land", "ok": False, "item_id": item_id,
                        "paths": norm, "errors": errors[:10], "reverted": undone,
                        # The same rows, in the same shape, as the passing row
                        # below. No key at all when no touched path is a
                        # `skills/<slug>/SKILL.md`, so an absent key still means
                        # "no skill was in this batch" and never "nothing to see".
                        **({"skill_gate": refused_skill_gate} if refused_skill_gate else {}),
                        **({"skill_body": refused_skill_body} if refused_skill_body else {}),
                        **({"session_id": session_id} if session_id else {})})
        raise VaultRoundError("validation failed; the change was reverted: "
                              + "; ".join(errors[:5]))

    # Recorded on every landing, and now on a refusal too. These are the rows the
    # flip was ruled on — 137 of them, 0 would-refusals, over the nine days the
    # rail logged (#2148) — and the first thing to read when a real refusal comes.
    skill_gate = skill_activation_findings(norm)
    skill_body = skill_body_findings(norm)

    review = "skipped"
    # Always says WHICH abstention it was, and is None when nothing was abstained:
    # an item-bound land that the reviewer passed must not arrive explained by a
    # reviewer who was never consulted. #955's whole point is that a skip reason
    # has to be true, and a wrong one is worse than an absent one.
    review_reason: str | None = None
    # A land with no item never reaches the second reader at all — there is no
    # contract to grade — which is a different fact from "a grader was asked and
    # could not answer". `scripts/autoresearch/promote.py` and this module's CLI
    # land here, and until #955 both shared the single word `skipped` with a
    # grader outage.
    if item_id is None:
        review_reason = ("no item bound: the second reader has no contract to grade, "
                         "so it was not consulted (module CLI, autoresearch promote)")
    clauses: list[dict] = []
    landing: set[int] = set()
    # What the reviewer said about a round it did NOT refuse. A seam the policy
    # makes advisory lands here, and this is the only surface a passing round's
    # finding ever reaches: a refusal writes `findings` on its own row and then
    # reverts, so without this field the advisory sentence is discarded at the
    # point it first matters (#1868 clause 4 — a seam that no longer blocks has to
    # survive the pass as a post-landing check, or making it advisory is the same
    # as deleting the rail). `review_findings`, not `findings`: on a `skipped` the
    # grader's findings slot holds the skip reason, which `review_reason` already
    # carries, and one field meaning two things is what that field was fixed for.
    # Empty findings write no key, so an absent key means "nothing reported" and
    # never "reported as nothing".
    review_findings: str = ""
    if item_id is not None:
        # Which grading of this round this is, computed BEFORE the grader runs.
        # `_vault_review_attempts` counts this item's prior blocking reviews, and
        # the grader has to decide the seam on the same number the refusal row
        # would carry — a round on its second attempt decided as attempt 1 is how
        # #1621 was refused twice under a policy that permits neither (#1868).
        attempts = _vault_review_attempts(int(item_id)) + 1
        kind, findings, clauses = _vault_review(norm, int(item_id), attempts)
        review = kind
        # The grader marks the clauses it refused to grade because they are about
        # this commit; the contract read is the belt to that brace, for the
        # skipped path where no rows come back to be marked.
        landing = {int(c["clause"]) for c in clauses if c.get("subject") == "landing"}
        landing |= set(_landing_clause_indices(int(item_id)))
        if kind == "skipped":
            review_reason = findings
        elif kind == "pass":
            # A refusal never gets here (it raises, and its own row carries the
            # text), and a `skipped` puts the skip REASON in that slot, which
            # `review_reason` already carries — so this is only ever a pass's
            # advisories.
            review_findings = findings
        if kind in ("retry", "unsound"):
            final = kind == "unsound" or attempts >= VAULT_REVIEW_MAX
            # First refusal: the edits stay in place so the model can fix them
            # and land again. Second, or an unsound premise: revert, with the
            # text in the event so the re-offer carries what was attempted —
            # and so the nightly vault-commit sweep cannot land it under
            # someone else's commit.
            undone = revert_paths(norm) if final else []
            S.append_event({"event": "vault_review", "item_id": item_id, "paths": norm,
                            "kind": kind, "blocking": True, "attempt": attempts,
                            "findings": findings[:2000], "reverted": undone,
                            # #1987: the refused clause in structured form, the
                            # same rows the non-blocking append below writes.
                            # Without it `met` was the only verdict any vault
                            # row could carry, and a refusal named its clause
                            # in prose alone. No `round_id`: a vault landing
                            # has no round.
                            "clauses": clauses,
                            "review_premise_unsound": kind == "unsound",
                            "review_retry": kind == "retry"})
            raise VaultRoundError(
                ("review: premise unsound — " if kind == "unsound" else
                 f"review sent it back ({attempts}/{VAULT_REVIEW_MAX}): ") + findings[:800]
                + ("; the edits were reverted" if undone else "; the edits are still in place — fix and land again"))
        # Every non-blocking outcome is recorded, an abstention included: before
        # #955 the reason for a skip was discarded here and the only trace was
        # the bare word `skipped` on the landing, which could not distinguish
        # "not consulted" from "the grader 503'd".
        S.append_event({"event": "vault_review", "item_id": item_id, "paths": norm,
                        "kind": kind, "blocking": False, "findings": findings[:600],
                        "review_reason": findings[:600] if kind == "skipped" else "",
                        "clauses": clauses})

    # #2036: does the code tree's own vault-reading selection still agree with the
    # vault as proposed? Validation above proves the prose PARSES, and that is its
    # ceiling — a sentence that parses perfectly can state a count the code does
    # not have, and then the live tree's guards start demanding code that has not
    # landed: #1975's prose land moved the retention skill to "thirteen" stores
    # while `scripts/groundskeeper/retention-sweep.py` still printed twelve, and
    # `tests/test_retention_sweep.py` has been red on every later round's base
    # probe since. Neither rung could see it, because each is blind to the half it
    # does not own — the code gate's diff never contains the vault, and
    # `review.grade_vault` abstains on a `mixed` surface at review.py:1433 for the
    # #551 reason. So this check is consistency-scoped and never surface-scoped: a
    # `mixed` item's vault land and a plain `vault` land run the identical probe,
    # and the review-rung skip exempts the land from nothing.
    #
    # It sits after validation and after the reviewer — the cheap rails first, the
    # ~70 s subprocess last — and before the commit, which is the only point that
    # can still leave the vault where it was. A probe that cannot judge does NOT
    # refuse: it records why and proceeds, the same shape as the reviewer's
    # abstention above, because holding the vault route hostage to a subprocess it
    # does not own is worse than the hole being closed. Silence is what is not
    # allowed: the report, with its denominator, goes on the row whatever it says.
    # `ack` is the one excuse this route has, and it is passed THROUGH rather than
    # interpreted here: `accepted_ack` decides which entries speak for this land, and
    # `excused_ids` decides which nodes they speak for. What is deliberately NOT passed is
    # `live_root` — the probe keeps judging HEAD's code, which is the hole #2036 closed, and a
    # caller that could point it at a candidate tree could make any land agree with itself.
    # An unacknowledged land makes byte-identically the call it made before this parameter
    # existed — `tests/test_automod_vault_round.py::
    # test_the_code_agreement_call_carries_no_surface_and_no_item` pins that keyword set as
    # exactly {"paths"}, and it should keep failing for anything that leaks into the probe.
    guards = VG.agreement(paths=list(norm),
                          **({"ack": [str(a) for a in ack]} if ack else {}))
    if guards.get("refuse"):
        undone = revert_paths(list(norm))
        S.append_event({"event": "vault_land", "ok": False, "item_id": item_id,
                        "session_id": session_id or "", "paths": norm,
                        "validated": list(norm),
                        "errors": [VG.refusal_text(guards)],
                        "reverted": undone,
                        "review": "skipped",
                        "review_reason": "code agreement refused the land",
                        "guards": _guards_row(guards)})
        raise VaultRoundError("code agreement failed; the change was reverted: "
                              + VG.refusal_text(guards)[:1200])

    _ensure_main()
    # Never `git add -A --` with an empty pathspec: that stages the whole vault.
    stage = _stageable(norm)
    if stage:
        add = _git("add", "-A", "--", *stage)
        if add.returncode != 0:
            raise VaultRoundError(f"git add failed: {add.stderr.strip()[:300]}")
    if _git("diff", "--cached", "--quiet").returncode == 0:
        # Every other refusal on this path — validation, code agreement — writes an
        # `ok: False` row before it raises, and this one wrote nothing at all: the
        # three refusals task-83 hit on 2026-10-04 survive only in its run record, and
        # of the twelve `vault_land` rows dated that day in promotions.jsonl every one
        # is `ok: true`. A land that committed nothing is now findable in the ledger
        # like any other, naming the paths, the refusal and the sha(s) it resolved to
        # (#2175 clause 3).
        held = _held_by_head(norm)
        text = _no_change_refusal(norm, held)
        S.append_event({"event": "vault_land", "ok": False, "item_id": item_id,
                        "paths": norm, "errors": [text],
                        # `commit` stays absent: this land made no commit. What HEAD
                        # holds is in `held_by`, keyed path -> sha, so a reader cannot
                        # mistake someone else's sha for this land's output.
                        "held_by": held,
                        "review": "skipped",
                        "review_reason": "nothing staged: HEAD already holds the "
                                         "named paths, or they were never changed",
                        **({"session_id": session_id} if session_id else {})})
        raise VaultRoundError(text)
    commit = _git("commit", "-q", "-m", message.strip())
    if commit.returncode != 0:
        _git("reset", "-q", "--", *norm)
        raise VaultRoundError(f"git commit failed: {(commit.stdout + commit.stderr).strip()[:300]}")
    sha = _git("rev-parse", "HEAD").stdout.strip()
    # A clause about the landing is graded here, from the sha, and nowhere else:
    # `land()` reviews before it commits, so no diff can satisfy such a clause at
    # grading time and #425/#502 each died on attempt 2 for exactly that. The
    # verdict is DERIVED, not asserted: the commit's own file list is compared
    # against the paths the clause names, so a landing that silently dropped one
    # of them — a path identical to HEAD, a path someone else had staged away —
    # comes back `unmet`. "Always met" would be the same unevidenceable claim the
    # reviewer was refused for making.
    committed = _committed_paths(sha)
    missing = sorted(set(norm) - committed)
    landing_verdict = "met" if not missing else "unmet"
    landing_note = (f"named paths not in this commit: {', '.join(missing[:5])}"
                    if missing else "")
    landing_rows = [{"clause": i, "verdict": landing_verdict, "commit": sha,
                     **({"note": landing_note} if missing else {})}
                    for i in sorted(landing)]
    # `review_clauses` is what `backlog.vault_review_outcome` reads: on a
    # landing that passed review it is the grader's verdict on the whole
    # contract, and the one verdict left when the turn dies at its budget. The
    # landing rows replace the reviewer's placeholder verdicts, so a contract
    # whose last clause is the landing can close on a passing review.
    review_clauses = clauses if review == "pass" else []
    if review_clauses and landing_rows:
        graded_indices = {int(r["clause"]) for r in landing_rows}
        review_clauses = ([r for r in review_clauses
                           if int(r.get("clause") or 0) not in graded_indices] + landing_rows)
        review_clauses.sort(key=lambda r: int(r.get("clause") or 0))
    S.append_event({"event": "vault_land", "ok": True, "item_id": item_id, "commit": sha,
                    "paths": norm, "validated": buckets["validated"],
                    "guards": _guards_row(guards),
                    "review": review, "review_reason": review_reason,
                    "review_clauses": review_clauses, "landing_clauses": landing_rows,
                    # Only when the reviewer said something about a round it
                    # passed: an advisory seam must outlive the pass, and this is
                    # the row a post-landing reader looks at.
                    **({"review_findings": review_findings} if review_findings else {}),
                    # The attribution `round_landing_rows` falls back to when the
                    # caller passed no item_id. A CLI/autoresearch land has no
                    # turn and so no session; it writes no key, which is honest.
                    **({"session_id": session_id} if session_id else {}),
                    **({"skill_gate": skill_gate} if skill_gate else {}),
                    **({"skill_body": skill_body} if skill_body else {}),
                    "message": message.strip()[:200]})
    return {"ok": True, "commit": sha, "paths": norm, "validated": buckets["validated"],
            "guards": _guards_row(guards),
            "review": review, "review_reason": review_reason,
            "landing_clauses": landing_rows,
            **({"review_findings": review_findings} if review_findings else {}),
            **({"skill_gate": skill_gate} if skill_gate else {}),
            **({"skill_body": skill_body} if skill_body else {})}


def revert_many(shas: list[str], reason: str = "rollback") -> dict:
    """Revert several vault commits, newest first, stopping at the first failure.

    Newest first because reverting an older commit before a newer one that
    touches the same file conflicts by construction. Partial success is
    reported rather than raised: a rollback that undid two of three commits
    has still changed the tree, and a caller told only "it failed" would have
    no idea which state it is now in.
    """
    done: list[str] = []
    for sha in reversed([s for s in shas if s]):
        try:
            revert(sha, reason=reason)
            done.append(sha)
        except Exception as exc:
            return {"ok": False, "reverted": done, "failed": sha, "error": str(exc)}
    return {"ok": True, "reverted": done}


def revert(sha: str, reason: str = "manual") -> dict:
    sha = (sha or "").strip()
    if not sha or _git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
        raise VaultRoundError(f"no such commit in the vault: {sha!r}")
    _ensure_main()
    r = _git("revert", "--no-edit", sha)
    if r.returncode != 0:
        _git("revert", "--abort")
        raise VaultRoundError(f"revert conflicted and was aborted: {r.stderr.strip()[:300]}")
    new = _git("rev-parse", "HEAD").stdout.strip()
    S.append_event({"event": "vault_revert", "reverted": sha, "commit": new, "reason": reason[:300]})
    return {"ok": True, "reverted": sha, "commit": new}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="vault_round")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("validate"); a.add_argument("paths", nargs="+")
    b = sub.add_parser("land"); b.add_argument("-m", "--message", required=True)
    b.add_argument("--item", type=int)
    # Repeatable, so `--ack` is a list the way the MCP tool's `ack` is a list, and it
    # stays `None` when absent — argparse's `default=[]` would make every ordinary land
    # look like a land that acknowledged something.
    b.add_argument("--ack", action="append", metavar="PATH",
                   help="Excuse PATH's witness-count change (repeatable). PATH must be one of "
                        "the paths being landed — an entry naming anything else is recorded and "
                        "void. The code agreement probe excuses only the test nodes that read "
                        "the named file, and records the whole exchange on the vault_land row.")
    b.add_argument("paths", nargs="+")
    c = sub.add_parser("revert"); c.add_argument("sha"); c.add_argument("--reason", default="manual")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "validate":
            errors, buckets = validate(args.paths)
            print(json.dumps({"ok": not errors, "errors": errors, "buckets": buckets}, indent=2))
            return 0 if not errors else 1
        if args.cmd == "land":
            # Same conditional as the MCP handler (`agent_mcp/automod.py`) and for the
            # same reason: an absent `--ack` must reach `land` as no keyword at all, so
            # the lander's probe call is byte-identically the one it made before the
            # parameter existed. Pinned by
            # `tests/test_automod_vault_round.py::test_the_cli_land_forwards_its_ack_flag_and_invents_none_without_it`.
            ack = [str(a) for a in (args.ack or []) if str(a).strip()]
            print(json.dumps(land(args.paths, args.message, item_id=args.item,
                                  **({"ack": ack} if ack else {})), indent=2))
            return 0
        print(json.dumps(revert(args.sha, args.reason), indent=2))
        return 0
    except VaultRoundError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
