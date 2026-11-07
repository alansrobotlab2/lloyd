"""Which durable files still name what this round retires (#3538).

The gate reads the code tree or the behaviour the code produces. `code_rules`
takes one module's AST at a time with no cross-file flow, and `pyright` adds the
*in-tree* inbound callers of changed module-level symbols — #528's family — and
stops at the code boundary by construction (`cwd=str(tree)` in
:func:`scripts.automod.typecheck._run_pyright`). The `tests` rung deliberately
excludes anything that reads the vault (`TESTS_MARK_EXPR` in
`scripts/automod/gate.py`). So the durable consumers of a retired name live where
no rung looks: the SKILL.md files and `autonomy/*.md` task files that store
commands verbatim, the backlog items, the knowledge notes, the loaded-memory
files. A rename whose consumers are prose passes the whole ladder green and
surfaces days later in a nightly job. Three times this box has paid for exactly
that, each discovered after landing:

  * Two data roots moved under `~/lloyd-data/` and every stored command naming
    the old root began returning a clean-looking non-answer; the same move left
    `USER.md` and `mental-models.md` naming the kg store under the CODE tree
    instead of the data root, where its live home is
    `~/lloyd-data/_pipeline/vault-derived/kg.sqlite` — found only afterwards by
    #882's resolver.
  * `CLAUDE.md`'s venv interpreter path: the code comment still says "247 files"
    where #1928 measured 697 files naming the spelling — hundreds of durable
    consumers, and no rung that counts them.
  * #407: `graph_affected` *under*-reports in-tree callers in
    `agent_mcp/vault.py`, so a complete caller list is not available even in
    code. This rung complements `pyright`'s caller list; it does not complete it.

This is the inverse query the loop does not have. #94's
`scripts/maintenance/referential_integrity.py` answers "does the cite in this
memory file still resolve?" over three loaded-memory files, paths only. This
answers "given a name this diff retired, which durable files name it?" over the
whole vault — and it reuses that resolver's exemption conventions rather than
writing a second extractor, per the item's own "do not build a third extractor if
one can be extended".

Three rules, each bought with a measured false positive:

  * **The candidate list is emitted before anything is matched.** Retired-name
    naming is the claim most likely to be wrong (a rename is a delete+add in a
    diff, so pairing is heuristic), so
    :attr:`ProseConsumerResult.candidates` carries every candidate verbatim —
    including the ones deliberately NOT searched for, each with its reason. An
    empty or wrong candidate list has to be legible in the round event, not
    inferred from an empty hit list.
  * **A hit requires the retired token including its root.** Grepping the retired
    store name without its root (`_pipeline/vault-derived/kg.sqlite`) returned
    196 vault files on 2026-10-30, and they were almost all the LIVE root
    carrying the retired string as a suffix: `skills/kg-mention-classifier/SKILL.md`
    reads `~/lloyd-data/_pipeline/vault-derived/kg.sqlite` (cited by path and not by
    line number: SKILL.md files are edited by their own skills, and the claim that
    matters is which file, so a pinned line would only be true until the next edit —
    `test_automod_prose_consumers_doc_claims.py` re-checks the content). So a retired
    path is never shortened to a sub-path, and the match may not begin mid-path
    either — a spelling matches only at the start of a path token.
  * **The instruction corpus is the actionable half.** `autonomy/*.md` task files
    and `SKILL.md` files are instructions the engine executes; `backlog/`,
    `knowledge/`, `memory/` and `lloyd/` are "the record of what was true when
    they were written" (`scripts/maintenance/rewrite_vault_data_paths.py:14-17`
    rules on exactly this split). Measured on 2026-10-30 the record corpus is
    ~91% of the files naming the venv spelling (1401 of 1529), so a matcher over
    both would be mostly report noise by construction. The two are reported as
    two numbers and only the instruction half is listed.

Precision is the known killer for this class and it is already measured here:
#882's triage got ~11% precision out of a prose-cite extractor and concluded "the
extractor, not the resolver, is the binding risk". Hence the absence-prose
exemption imported from that resolver — a line saying the target is retired,
gone, superseded, or carrying `<!-- ri:known-absent -->` is naming the dead thing
ON PURPOSE, which is a mention and not a consumer — and hence this module blocks
nothing: the item labels hits over the last 20 landed rounds before anyone is
allowed to act on one.

Never raises, never refuses. A matcher that cannot run returns
`status="unevaluated"` with the reason, because "did not run" and "found nothing"
are different answers and only one of them is clean — the rule the `pyright` rung
runs on.

The two ends of the acceptance check are both real files on this box. The one that
MUST be reported is `autonomy/70-skill-lint.md`, whose Command section stores
`/home/alansrobotlab/lloyd/.venvs/lloyd/bin/python` verbatim — a round that retired
that interpreter would break the task, and today no rung notices. The one that must
NOT is the kg-mention-classifier file above. Both are asserted against the live
vault, not against a fixture, by the seed tests.

Two entry points: :func:`scan` is pure (two trees and a diff body — what the
tests call) and :func:`scan_round` does the git plumbing around it for a real
round. Hand-run it over any round:

    python -m scripts.automod.prose_consumers \\
        --worktree ~/lloyd-work/SM_…/home/lloyd --base HEAD~1 [--vault ~/obsidian] [--json]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from app.data_root import vault_root as _live_vault_root

EVALUATED = "evaluated"
UNEVALUATED = "unevaluated"

#: The two kinds resolved here. A retired dict/state key, config key or tool name
#: is the third kind the item also asks for; it is NOT here, because a bare key
#: is the short-token case that measured worst in #882's ablation and it needs a
#: pairing rule of its own. Recorded as owed on #3538.
KIND_SYMBOL = "symbol"
KIND_PATH = "path"

#: The two corpus halves. `RECORD_DIRS` is #3538's own list, and the same split
#: `scripts/maintenance/rewrite_vault_data_paths.py` applies when it decides what
#: a root move may rewrite.
RECORD_DIRS = ("backlog", "knowledge", "memory", "lloyd")
#: The instruction half, in the scope vocabulary of the tool that rewrites these
#: very files. The second entry is deliberately NARROWER than that tool's
#: `skills/**/*.md`: a skill's sidecar prose (`skills/entity-resolution-sweep/
#: tiers-and-guardrails.md`) is read by whoever opens it and is not what the loader
#: injects, so a hit there is counted in `out_of_corpus_hits` — kept, not listed.
INSTRUCTION_GLOBS = ("autonomy/*.md", "skills/**/SKILL.md")
INSTRUCTION_DIRS = ("autonomy", "skills")
#: An autonomy file is a task file when the autonomy loader can name it: the
#: writer names them `<task-id>-<slug>.md`. `autonomy/referential-integrity-
#: latest.md` and its siblings are generated reports — no id, nothing runs them —
#: so they are out of corpus; counting a report's own quote of a dead path as a
#: consumer is the #882 error, re-imported.
_AUTONOMY_TASK_FILE = re.compile(r"^\d+-[^/]+\.md$")

#: Skipped everywhere: archived skills, editor state, the vault's own `.git`.
#: A path is excluded if ANY segment matches, not just the first — the vault holds
#: archives at every depth (`skills/.archived/<x>/SKILL.md`,
#: `backlog/_archived/<x>.md`), and both spellings are in use: the 52 live
#: instruction hits this rung reported for the interpreter path on 2026-11-03
#: included archived skills, which nothing loads and so nothing can break.
_EXCLUDE_DIRS = ("_archived", ".archived", ".obsidian", ".git", "node_modules",
                 "__pycache__")

#: What the GREP skips, as opposed to what `classify` refuses to call a consumer.
#: The archives are deliberately NOT here: an archived skill still names the
#: retired path, and `out_of_corpus_hits` is the number that says "the rung saw this
#: and decided it is not actionable". Excluding it in the grep would make a file
#: vanish from every count, which is the thing this rung must never do — report a
#: clean corpus because it filtered the evidence away.
_SKIP_WALK_DIRS = (".git", ".obsidian", "node_modules", "__pycache__")

#: A symbol candidate shorter than this is emitted but not searched. D13 retired
#: the `RunOptions.env` field, and `env` is three characters: a free-text grep for
#: a short identifier across thousands of notes IS #882's 11% precision, which the
#: item's own Risks section names as the killer. The candidate is still printed, so
#: the miss is a published decision rather than a silent hole.
MIN_SYMBOL_CHARS = 8

#: Storage ceilings. `gate._clip_detail` clips the detail again; these keep
#: `gate.json` and the ledger row proportionate to a single rename.
MAX_CANDIDATES_LISTED = 30
MAX_HITS_LISTED = 25
MAX_RECORD_FILES_LISTED = 10
MAX_GREP_LINES = 4000
GREP_TIMEOUT_SECONDS = 60.0

# ── the matcher ─────────────────────────────────────────────────────────────

#: A path spelling matches only at the start of a path token: not after another
#: path character, not after a `~`. That is what stops a retired path being found
#: in the tail of a longer live path — the 196-file error above.
_LEFT = r"(?<![A-Za-z0-9._~/-])"
#: …and not in the middle of a name or of a deeper path: a retired `bin/python`
#: does not name `bin/python3`, and a retired `…/kg.sqlite` does not name
#: `…/kg.sqlite/backup`. A sentence ending in a path still counts, so a full stop
#: is NOT in this class. A retired ROOT (trailing `/`) is exempt from the right
#: edge entirely — everything beneath a retired root is a consumer of it.
_RIGHT = r"(?![A-Za-z0-9_/-])"


def _pattern(kind: str, spelling: str) -> re.Pattern[str]:
    """The ONE boundary pattern for one spelling of one kind."""
    if kind == KIND_SYMBOL:
        return re.compile(rf"\b{re.escape(spelling)}\b")
    tail = "" if spelling.endswith("/") else _RIGHT
    return re.compile(f"{_LEFT}{re.escape(spelling)}{tail}")


# ── candidates: what the round retired ──────────────────────────────────────

#: A removed literal is a path only if every character of it could be one.
#: `{`/`*`/`%`/`<` are templates, globs and format specs — `referential_integrity
#: .py`'s `_TEMPLATE` refuses the same shapes for the same reason.
_PATH_CHARS = re.compile(r"^[~.]?[A-Za-z0-9_$+,\-]*(?:/[A-Za-z0-9_.$+,\-]+)*/?$")
_NOT_A_PATH = re.compile(r"^https?://|^www\.|@|\.\.\.|\^/\*$|^/$|^\./$|^\$\{|%s|%d")
#: A path component has to be recognisable as one: a dot in it, or a trailing
#: slash (a root). `bin/python` alone is a directory pair and nothing says which
#: root, so it is not a candidate.
_RECOGNISABLE = re.compile(r"\.|/$")

#: Quoted literals on a removed diff line. Deliberately simple: the line is
#: already known to be a deletion, and a multi-line string is never a path.
_QUOTED = re.compile(r"'([^'\n]{2,200})'|\"([^\"\n]{2,200})\"")


def path_candidate(lit: str, *, checkout: Path, home: Path) -> list[str] | None:
    """Every spelling durable prose could use for one removed literal, or None.

    Only spellings of the SAME file: the literal as written, plus its absolute
    and `~/` forms once its root is resolved against the two roots this loop has
    — the checkout and the home it sits in. A literal is never SHORTENED (see the
    docstring), and one whose root cannot be resolved is refused rather than
    turned into a rootless grep.
    """
    lit = (lit or "").strip()
    if len(lit) < 4 or any(ch.isspace() for ch in lit):
        return None
    if _NOT_A_PATH.search(lit) or not _PATH_CHARS.match(lit):
        return None
    if not _RECOGNISABLE.search(lit) and not lit.startswith(("~", "/")):
        return None
    if lit.startswith("~/"):
        absolute = home / lit[2:]
    elif lit.startswith("/"):
        absolute = Path(lit)
    elif lit.startswith("./"):
        absolute = checkout / lit[2:]
    elif "/" in lit:
        absolute = checkout / lit
    else:
        return None
    out = [lit, str(absolute)]
    try:
        # The `~` spelling only exists for a path that is under home. A candidate
        # outside it (`/etc/…`, a checkout elsewhere) must never yield `~/…`,
        # because `~/foo` would then be searched as a ROOT form of something the
        # vault names for a completely unrelated reason.
        out.append("~/" + str(absolute.relative_to(home)))
    except ValueError:
        pass
    # A ROOT survives Path normalisation only if it is put back: `Path("~/x/y/")`
    # renders without its slash, and that slash is the whole difference between
    # "retire this one file" and "retire everything under here" — see _RIGHT.
    if lit.endswith("/"):
        out = [s if s.endswith("/") else s + "/" for s in out]
    deduped: list[str] = []
    for s in out:
        if s and s not in deduped:
            deduped.append(s)
    return deduped


def removed_literals(diff_text: str) -> list[str]:
    """Quoted literals on lines the diff deleted, in first-seen order."""
    out: list[str] = []
    for line in diff_text.splitlines():
        if not line.startswith("-") or line.startswith("---"):
            continue
        for m in _QUOTED.finditer(line):
            lit = (m.group(1) or m.group(2) or "").strip()
            if lit and lit not in out:
                out.append(lit)
    return out


@dataclass
class Candidate:
    """One name the round retired, and what this module decided to do about it."""

    name: str
    kind: str
    origin: str
    forms: list[str] = field(default_factory=list)
    why_not_searched: str = ""

    def to_dict(self) -> dict:
        d: dict = {"name": self.name, "kind": self.kind, "origin": self.origin}
        if self.forms:
            d["forms"] = list(self.forms)
        if self.why_not_searched:
            d["not_searched"] = self.why_not_searched
        return d


def symbol_candidates(retired: dict[str, set[str]]) -> list[Candidate]:
    """Candidates for the module-level names a round deleted, by module.

    A `Class.method` retire is cited in prose as the bare method as often as not,
    so the leaf is a candidate alongside the dotted key. Each keeps its own
    origin, and a name that collides with one already listed is dropped rather
    than duplicated.
    """
    out: list[Candidate] = []
    for module in sorted(retired):
        for name in sorted(retired[module]):
            out.append(Candidate(name, KIND_SYMBOL, module))
            if "." in name:
                out.append(Candidate(name.split(".")[-1], KIND_SYMBOL, module))
    seen: set[str] = set()
    uniq: list[Candidate] = []
    for c in out:
        if c.name in seen:
            continue
        seen.add(c.name)
        if len(c.name) < MIN_SYMBOL_CHARS:
            c.why_not_searched = (
                f"shorter than {MIN_SYMBOL_CHARS} characters: a free-text grep for "
                "a short identifier is #882's 11%-precision case")
        else:
            c.forms = [c.name]
        uniq.append(c)
    return uniq


def path_candidates(literals: list[str], *, checkout: Path,
                    home: Path) -> tuple[list[Candidate], list[str]]:
    """Candidates for the paths/roots the diff deleted, plus what was refused."""
    out: list[Candidate] = []
    refused: list[str] = []
    seen: set[str] = set()
    for lit in literals:
        if lit in seen:
            continue
        seen.add(lit)
        forms = path_candidate(lit, checkout=checkout, home=home)
        if not forms:
            refused.append(lit)
            continue
        out.append(Candidate(lit, KIND_PATH, "diff", forms=forms))
    return out, refused


# ── the corpus ──────────────────────────────────────────────────────────────

def corpus_scopes(vault: Path) -> tuple[str, ...]:
    """Which corpus directories this vault actually holds, sorted.

    Empty means there is no corpus to search — the rung must say so rather than
    report the zero consumers its empty grep would otherwise produce.
    """
    return tuple(sorted({d for d in set(INSTRUCTION_DIRS) | set(RECORD_DIRS)
                         if (vault / d).is_dir()}))


def classify(rel: str) -> str:
    """`instruction` | `record` | `""` (out of corpus) for a vault-relative path.

    `skills/**/SKILL.md` and `autonomy/<id>-<slug>.md` are the files something
    executes; `backlog`, `knowledge`, `memory` and `lloyd` are the record of what
    was true when written. Everything else — `projects/`, `plans/`, an `autonomy/`
    report — is out of corpus and counted as such, not folded into either number.
    """
    parts = [p for p in (rel or "").replace("\\", "/").split("/") if p]
    if not parts or any(p in _EXCLUDE_DIRS for p in parts):
        return ""
    top = parts[0]
    if top == "skills":
        return "instruction" if parts[-1] == "SKILL.md" else ""
    if top == "autonomy":
        # Depth 2: the vault-relative path of a file sitting IN `autonomy/`. A
        # nested one is not a task file whatever it is called.
        return ("instruction" if len(parts) == 2
                and _AUTONOMY_TASK_FILE.match(parts[-1]) else "")
    return "record" if top in RECORD_DIRS else ""


def grep_corpus(vault: Path, spellings: list[str], *,
                timeout: float = GREP_TIMEOUT_SECONDS,
                ) -> tuple[list[tuple[str, int, str]], str]:
    """`( [(vault-relative file, line, text)], error )` for fixed-string hits.

    `grep -rnF` rather than a Python walk: thousands of markdown files under one
    root, and a fixed string is a superset of every boundary-restricted pattern
    below — so the exact match happens once, in Python, and the accelerator can
    never invent a hit. An error is returned, not raised: no grep is
    `unevaluated`, never a clean report.
    """
    if not spellings:
        return [], ""
    cmd = ["grep", "-rnI", "-F", "--include=*.md"]
    for d in _SKIP_WALK_DIRS:
        cmd.append(f"--exclude-dir={d}")
    for s in spellings:
        cmd += ["-e", s]
    cmd.append(str(vault))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           check=False)
    except Exception as exc:                                    # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"
    if r.returncode not in (0, 1):
        return [], f"grep exited {r.returncode}: {(r.stderr or '').strip()[:200]}"
    pre = str(vault).rstrip("/") + "/"
    out: list[tuple[str, int, str]] = []
    for line in r.stdout.splitlines():
        head, sep, rest = line.partition(":")
        if not sep:
            continue
        lineno, sep2, text = rest.partition(":")
        if not sep2 or not lineno.isdigit():
            continue
        rel = head[len(pre):] if head.startswith(pre) else head
        out.append((rel, int(lineno), text))
        if len(out) >= MAX_GREP_LINES:
            return out, ""
    return out, ""


# ── the result ──────────────────────────────────────────────────────────────

@dataclass
class ProseConsumerResult:
    """What the matcher concluded — including whether it got to conclude."""

    status: str = EVALUATED
    reason: str = ""
    candidates: list[dict] = field(default_factory=list)
    hits: list[dict] = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    top_record: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"status": self.status, "reason": self.reason,
                "candidates": self.candidates, "counts": self.counts,
                "top_record_files": self.top_record, "hits": self.hits}


def scan(head_tree: Path, base_tree: Path | None, changed: list[str],
         diff_text: str, *, checkout: Path, vault: Path, home: Path,
         timeout: float = GREP_TIMEOUT_SECONDS) -> ProseConsumerResult:
    """Retire, spell, grep, classify — over two trees and a diff body.

    The pure half: no git, so a test hands it fixture directories. Never raises.
    """
    from scripts.automod import typecheck as TC     # same extractor, gate-cheap
    try:
        from scripts.maintenance import referential_integrity as RI
        exempt_lines, exempt_marker = RI.ABSENCE_PROSE, RI.EXEMPT_MARKER
    except Exception as exc:                        # noqa: BLE001 — no exemptions
        return ProseConsumerResult(
            status=UNEVALUATED,
            reason=f"the exemption conventions could not be loaded "
                   f"({type(exc).__name__}: {exc})")
    changed_py = [p for p in changed if p.endswith(".py") and not TC.in_fork(p)]
    notes: list[str] = []
    try:
        retired = TC.retired_module_symbols(head_tree, base_tree, changed_py)
    except Exception as exc:                        # noqa: BLE001 — half a verdict
        retired = {}
        notes.append(f"symbol extraction failed ({type(exc).__name__}: {exc})")
    candidates = symbol_candidates(retired)
    path_cands, refused = path_candidates(removed_literals(diff_text),
                                          checkout=checkout, home=home)
    candidates += path_cands
    res = _match(vault, candidates, exempt_lines=exempt_lines,
                 exempt_marker=exempt_marker, timeout=timeout)
    res.counts["literals_refused"] = len(refused)
    if notes:
        res.status = UNEVALUATED
        res.reason = "; ".join(notes + ([res.reason] if res.reason else []))[:300]
    return res


def _match(vault: Path, candidates: list[Candidate], *,
           exempt_lines: re.Pattern[str] | None, exempt_marker: str,
           timeout: float) -> ProseConsumerResult:
    """Grep the durable corpus for the candidates and classify every hit."""
    searched = [c for c in candidates if c.forms]
    counts: dict = {
        "candidates": len(candidates),
        "searched": len(searched),
        "not_searched": len(candidates) - len(searched),
        "instruction_hits": 0, "instruction_exempt": 0,
        "record_hits": 0, "out_of_corpus_hits": 0,
        "instruction_files": 0, "record_files": 0,
        "grep_lines": 0, "corpus_truncated": False,
    }
    res = ProseConsumerResult(candidates=[c.to_dict() for c in candidates],
                              counts=counts)
    if not searched:
        res.reason = ("no searchable retired name (nothing retired, or every "
                      "candidate too short to grep for)")
        return res
    spellings = sorted({f for c in searched for f in c.forms})
    lines, err = grep_corpus(vault, spellings, timeout=timeout)
    if err:
        res.status, res.reason = UNEVALUATED, err
        return res
    counts["grep_lines"] = len(lines)
    counts["corpus_truncated"] = len(lines) >= MAX_GREP_LINES
    patterns = [(c, _pattern(c.kind, f), f) for c in searched for f in c.forms]
    listed: list[dict] = []
    instr_files: set[str] = set()
    record_files: dict[str, int] = {}
    for rel, lineno, text in lines:
        found = None
        for cand, pat, spelling in patterns:
            if spelling in text:
                m = pat.search(text)
                if m:
                    found = (cand, m)
                    break
        if found is None:
            continue                        # a mid-path occurrence: not a consumer
        bucket = classify(rel)
        if bucket == "":
            counts["out_of_corpus_hits"] += 1
            continue
        cand, m = found
        if exempt_marker in text or (exempt_lines is not None
                                     and exempt_lines.search(text)):
            # A line that says the thing is retired is naming it on purpose.
            if bucket == "instruction":
                counts["instruction_exempt"] += 1
            continue
        if bucket == "record":
            counts["record_hits"] += 1
            record_files[rel] = record_files.get(rel, 0) + 1
            continue
        counts["instruction_hits"] += 1
        instr_files.add(rel)
        listed.append({"name": cand.name, "kind": cand.kind, "file": rel,
                       "line": lineno, "matched": m.group(0),
                       "text": text.strip()[:240]})
    # Sorted BEFORE the cut, not after. `grep -r` walks in directory order, so
    # truncating first would keep an arbitrary subset that changes when a skill is
    # added — and two runs of the same round have to name the same files, or the
    # ledger entries a reviewer compares are not comparable. #1928 measured 697
    # files naming one interpreter path, which is a list worth sorting.
    hits = sorted(listed, key=lambda h: (h["file"], h["line"]))[:MAX_HITS_LISTED]
    counts["hits_listed"] = len(hits)
    counts["instruction_files"] = len(instr_files)
    counts["record_files"] = len(record_files)
    res.hits = hits
    res.top_record = [f"{f} ({n})" for f, n in
                      sorted(record_files.items(), key=lambda kv: (-kv[1], kv[0]))
                      [:MAX_RECORD_FILES_LISTED]]
    return res


# ── the round-facing half: git plumbing, and nothing else ───────────────────

@contextmanager
def scratch_base_tree(worktree: Path, base: str):
    """Yield `(a detached checkout of `base`, why-not)`, and always clean up.

    The `pyright` rung plays the same trick: a type/AST delta needs the base
    image as a real tree, and a scratch of individual blobs would leave every
    import unresolved. The path is created here and deleted here — never a
    computed delete of something this module did not make.
    """
    parent = Path(tempfile.mkdtemp(prefix=f"prose-{base[:8]}-"))
    target = parent / "base"
    tree: Path | None = None
    why = ""
    try:
        shutil.rmtree(parent, ignore_errors=True)      # `worktree add` wants absent
        r = subprocess.run(["git", "-C", str(worktree), "worktree", "add",
                            "--detach", "-q", str(target), base],
                           capture_output=True, text=True, timeout=180, check=False)
        if r.returncode == 0:
            tree = target
        else:
            why = f"base checkout failed: {(r.stderr or r.stdout).strip()[:160]}"
    except Exception as exc:                            # noqa: BLE001 — no base
        why = f"base checkout: {type(exc).__name__}: {exc}"
    try:
        yield tree, why
    finally:
        if tree is not None:
            subprocess.run(["git", "-C", str(worktree), "worktree", "remove",
                            "--force", str(target)], capture_output=True,
                           timeout=120, check=False)
        shutil.rmtree(parent, ignore_errors=True)


def scan_round(*, worktree: Path, base: str, changed: list[str], checkout: Path,
               vault: Path | None = None, home: Path | None = None,
               timeout: float = GREP_TIMEOUT_SECONDS) -> ProseConsumerResult:
    """The whole thing for one round: base image, diff, then :func:`scan`."""
    vault = Path(vault or _live_vault_root())
    home = Path(home or Path.home())
    notes: list[str] = []
    with scratch_base_tree(worktree, base) as (base_tree, why):
        if why:
            notes.append(why)
        try:
            r = subprocess.run(["git", "-C", str(worktree), "diff", "--no-color",
                                "--unified=0", base, "--"] + list(changed),
                               capture_output=True, text=True, timeout=120,
                               check=False)
            diff_text = r.stdout if r.returncode == 0 else ""
            if r.returncode != 0:
                notes.append(f"git diff exit {r.returncode}")
        except Exception as exc:                        # noqa: BLE001 — no diff
            diff_text = ""
            notes.append(f"git diff: {type(exc).__name__}: {exc}")
        res = scan(worktree, base_tree, changed, diff_text, checkout=checkout,
                   vault=vault, home=home, timeout=timeout)
    if notes:
        res.status = UNEVALUATED
        res.reason = "; ".join(notes + ([res.reason] if res.reason else []))[:300]
    return res


def detail(res: ProseConsumerResult) -> str:
    """The rung's one-line record: the candidates first, then the two counts.

    The candidate list leads because it is the claim a reader has to be able to
    check before the counts mean anything, and because a rung that reports only
    "0 hits" cannot be told apart from a rung that found nothing to look for.
    """
    listed = res.candidates[:MAX_CANDIDATES_LISTED]
    names = ", ".join(f"{c['kind']}:{c['name']}" for c in listed)
    more = f" (+{len(res.candidates) - len(listed)} more)" if (
        len(res.candidates) > len(listed)) else ""
    head = (f"observe-only: this round retires {len(res.candidates)} name(s) "
            f"[{names}]{more}") if res.candidates else (
        "observe-only: this round retires no name, so no durable consumer can "
        "exist")
    c = res.counts
    if res.status == UNEVALUATED:
        return f"{head} — UNEVALUATED: {res.reason[:160]}"
    if not c.get("searched"):
        return f"{head} — nothing searched ({res.reason[:120]})"
    tail = (f"instruction {c['instruction_hits']} actionable hit(s) in "
            f"{c['instruction_files']} file(s), {c['instruction_exempt']} "
            f"exempt-by-absence-prose; record {c['record_hits']} line(s) in "
            f"{c['record_files']} file(s) (counted, not actionable)")
    if c.get("corpus_truncated"):
        tail += f" [corpus capped at {MAX_GREP_LINES} grep lines]"
    top = "; ".join(f"{h['file']}:{h['line']} [{h['matched']}]" for h in res.hits[:3])
    return f"{head} — {tail}" + (f" — top: {top}" if top else "")


def main(argv: list[str] | None = None) -> int:
    """Hand-run the matcher over any worktree/base pair. Reports only, exit 0."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--base", default="HEAD")
    ap.add_argument("--checkout", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--vault", default="")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    from scripts.automod import worktree as W
    wt = Path(a.worktree)
    res = scan_round(worktree=wt, base=a.base, changed=W.changed_paths(wt, a.base),
                     checkout=Path(a.checkout),
                     vault=Path(a.vault) if a.vault else None)
    print(json.dumps(res.to_dict(), indent=2) if a.json else detail(res))
    for h in res.hits:
        print(f"  {h['file']}:{h['line']} {h['kind']} {h['name']}: {h['text'][:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
