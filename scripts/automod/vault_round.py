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

from scripts.automod import spec, state as S

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
    from prompt_builder import build_system_prompt
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
    from autonomy import _parse_task_file
    for p in tasks:
        f = vault / p
        if f.exists() and _parse_task_file(f) is None:
            errs.append(f"{p}: autonomy cannot parse it")
print(json.dumps(errs))
"""


def loader_errors(paths: list[str]) -> list[str]:
    """Run the real loaders in a fresh interpreter, scoped to `paths`."""
    r = subprocess.run([str(PYTHON), "-c", _LOADER_SCRIPT, json.dumps(paths), str(VAULT)],
                       cwd=str(LLOYD_HOME), capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        return [f"loader crashed: {(r.stdout + r.stderr).strip()[-600:]}"]
    try:
        return list(json.loads(r.stdout.strip().splitlines()[-1]))
    except (ValueError, IndexError):
        return [f"loader produced no verdict: {(r.stdout + r.stderr).strip()[-300:]}"]


CONTRACT_PATHS = ("lloyd/SOUL.md", "lloyd/MEMORY.md")


def contract_errors(paths: list[str]) -> list[str]:
    """Prompt-surface invariants, when the change touches the identity files.

    The loaders below already answer "does the prompt still build". They do
    not answer "is it still the shape #377 left it in", and that is the
    question a writer has to answer, because the reader cannot: the gate's
    `tests` rung judges a candidate commit and `SOUL.md` is not in it.

    Scoped to a diff that names one of the contract files, like every other
    check here — a pre-existing condition elsewhere in the vault must not
    block an unrelated round.
    """
    if not any(p in CONTRACT_PATHS for p in paths):
        return []
    try:
        import prompt_surface
    except ImportError as exc:  # pragma: no cover - repo is always importable
        return [f"prompt_surface unavailable, cannot check the contract: {exc}"]
    errs = prompt_surface.check_paths(VAULT / "lloyd" / "SOUL.md",
                                      VAULT / "lloyd" / "MEMORY.md")
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
        errors.extend(contract_errors(paths) + reflection_archive_errors(paths)
                      + skill_timezone_errors(paths))
    if not errors and buckets["validated"]:
        errors.extend(loader_errors(buckets["validated"]))
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


def _vault_review(norm: list[str], item_id: int) -> tuple[str, str, list[dict]]:
    """`(kind, findings, clauses)` from the grader over the staged diff. Never
    raises; an unusable grader is `("skipped", why, [])` and the landing
    proceeds — a vault edit is already validated through the real loaders,
    and a grader outage must not hold every skill edit hostage.

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
        diff = _git("diff", "HEAD", "--", *norm).stdout
        for p in norm:
            if _git("cat-file", "-e", f"HEAD:{p}").returncode != 0 and (VAULT / p).exists():
                diff += f"\n+++ new file {p}\n" + (VAULT / p).read_text(encoding="utf-8", errors="replace")
        res = tuple(GRADER(item_id=item_id, paths=norm, diff=diff))
        graded = res[2] if len(res) > 2 else []
        clauses = [{"clause": int(c["clause"]), "verdict": str(c["verdict"]),
                    **({"subject": str(c["subject"])} if c.get("subject") else {})}
                   for c in (graded or []) if isinstance(c, dict)
                   and str(c.get("clause", "")).isdigit() and c.get("verdict")]
        return str(res[0]), str(res[1]), clauses
    except Exception as exc:  # noqa: BLE001 — the grader never fails a landing on its own
        return ("skipped", f"grader raised {type(exc).__name__}: {str(exc)[:200]}", [])


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


def land(paths: list[str], message: str, *, item_id: int | None = None,
         session_id: str | None = None) -> dict:
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
        undone = revert_paths(norm) if not buckets["denied"] else []
        S.append_event({"event": "vault_land", "ok": False, "item_id": item_id,
                        "paths": norm, "errors": errors[:10], "reverted": undone,
                        **({"session_id": session_id} if session_id else {})})
        raise VaultRoundError("validation failed; the change was reverted: "
                              + "; ".join(errors[:5]))

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
    if item_id is not None:
        kind, findings, clauses = _vault_review(norm, int(item_id))
        review = kind
        # The grader marks the clauses it refused to grade because they are about
        # this commit; the contract read is the belt to that brace, for the
        # skipped path where no rows come back to be marked.
        landing = {int(c["clause"]) for c in clauses if c.get("subject") == "landing"}
        landing |= set(_landing_clause_indices(int(item_id)))
        if kind == "skipped":
            review_reason = findings
        if kind in ("retry", "unsound"):
            attempts = _vault_review_attempts(int(item_id)) + 1
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

    _ensure_main()
    add = _git("add", "-A", "--", *norm)
    if add.returncode != 0:
        raise VaultRoundError(f"git add failed: {add.stderr.strip()[:300]}")
    if _git("diff", "--cached", "--quiet").returncode == 0:
        raise VaultRoundError("nothing to commit on those paths")
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
    committed = {line.strip() for line in
                 _git("show", "--name-only", "--format=", sha).stdout.splitlines()}
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
                    "review": review, "review_reason": review_reason,
                    "review_clauses": review_clauses, "landing_clauses": landing_rows,
                    # The attribution `round_landing_rows` falls back to when the
                    # caller passed no item_id. A CLI/autoresearch land has no
                    # turn and so no session; it writes no key, which is honest.
                    **({"session_id": session_id} if session_id else {}),
                    "message": message.strip()[:200]})
    return {"ok": True, "commit": sha, "paths": norm, "validated": buckets["validated"],
            "review": review, "review_reason": review_reason,
            "landing_clauses": landing_rows}


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
    b.add_argument("--item", type=int); b.add_argument("paths", nargs="+")
    c = sub.add_parser("revert"); c.add_argument("sha"); c.add_argument("--reason", default="manual")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "validate":
            errors, buckets = validate(args.paths)
            print(json.dumps({"ok": not errors, "errors": errors, "buckets": buckets}, indent=2))
            return 0 if not errors else 1
        if args.cmd == "land":
            print(json.dumps(land(args.paths, args.message, item_id=args.item), indent=2))
            return 0
        print(json.dumps(revert(args.sha, args.reason), indent=2))
        return 0
    except VaultRoundError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
