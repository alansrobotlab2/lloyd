#!/usr/bin/env python3
"""Measure extraction error rate with two independent passes reconciled against the source.

Gallon's Level-2 review ("I Believe Everything My Agents Tell Me", AI Engineer,
2026-10-13): extract the same document twice under identical instructions,
neither pass seeing the other's work, then a third read of the SOURCE settles
every difference — never by majority, never by confidence. Independent-pass
disagreement estimates an error rate with **no gold labels**, which is the one
number Lloyd cannot produce today: `scripts/memory/next-gen-memory/fact_extractor.py`
records `created_at, source_doc, source_hash, provenance` per fact (attribution)
and nothing ever asks whether a written fact agrees with the document it came
from. Every existing check asks a different question — `workers/evidence.py`
(#525) verifies that an artifact EXISTS, `eval/cross_engine_disagreement.py`
(#2639) compares two engines over grader VERDICTS that have gold, the
`faithfulness-verification` skill checks that cited fact UIDs EXIST in a file.

**This module measures and publishes. It wires nothing.** No gate, no nightly
change, no auto-merge, and above all no repair: `scripts/memory/repair_fact_duplicates.py`
and `_merge_facts` collapse near-duplicates, and a disagreement is NOT a
duplicate — the 08-22 entity-graph incident is the standing warning. A run
writes only inside its `--out` directory.

What a pair has to be before it may be reported
-----------------------------------------------
Two passes of the same weights are not independent the way Gallon's two agents
are (the same objection `scripts/automod/review.py` states about a second reader
with "the same weights as the first pass"), so disagreement here is a LOWER
BOUND on the error rate. That makes byte-identity of the two passes the whole
instrument: any difference in prompt bytes, chunk windows or sampling knobs
turns "the model disagreed with itself" into "the two calls were asked
different questions", which is a number about nothing. So the prompts are built
ONCE per document from the production prompt template and the production
chunker, sent verbatim twice, and `pair_guard` refuses to report a pair whose
prompt digest or chunk offsets differ. The one knob this module holds fixed by
hand is `existing_facts=""` (the production default, which the extractor renders
as "None"): the incremental run passes the entity's live fact list, which is
vault state, and vault state can move between the two calls.

Cost
----
`_call_llm` in the extractor returns the message text and drops the response's
`usage` field, so the usage store has no extraction rows to read a cost from
(it is session-keyed harness telemetry). This module therefore takes `usage` off
every response it sends and times every call: the artifact's per-document cost
is measured here or not at all. EVERY call is metered exactly once — pass A, pass
B, and the reconciler's reads too, each kept in its own arm — because Gallon's
7.5M tokens is the cost of a reconciliation that includes the third agent, so an
artifact that billed only its two passes would be compared against a smaller
thing than the method it cites. That figure is also why the method is affordable
as a sampled audit and never as a doubling of the nightly corpus.

Usage
-----
    eval/extraction_disagreement.py --out <dir> [--sample-size 40] [--seed ...]
                                     [--port 8096] [--settle-port 8096] [--dry-run]

A `--dry-run` draws the sample, prints it and writes the artifact without
sending a single request — the offline shape the tests run over.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: The module under audit, and the config that names the corpus. The directory
#: is hyphenated, so it is not importable by name; `importlib` loads it by path
#: the way `tests/test_fact_extractor.py` and `eval/cross_engine_disagreement.py`
#: load the things they measure.
EXTRACTOR_PATH = ROOT / "scripts" / "memory" / "next-gen-memory" / "fact_extractor.py"
PIPELINE_CONFIG_PATH = ROOT / "scripts" / "memory" / "next-gen-memory" / "pipeline_config.yaml"
NIGHTLY_PATH = ROOT / "scripts" / "memory" / "next-gen-memory" / "nightly_extraction.py"

#: The nightly job module, imported once (`_nightly_rules`).
_NIGHTLY = None

#: The extraction corpus is an allow-list, not the vault. `skills/`, `backlog/`,
#: `architecture/`, `autonomy/`, `agents/`, `facts/`, `sessions/` and `youtube/`
#: are absent from it deliberately (pipeline_config.yaml:18-22), so sampling one
#: would audit a document the production write path never reads and the rate
#: would not transfer. Read from the YAML, never restated as a literal: a list
#: copied here is a list that goes stale the day the allow-list moves, silently.

#: A document shorter than this has nothing to disagree about; longer than the
#: one-pass chunk budget and a single pass cannot even see the whole of it, so
#: the pair would be settled against text neither pass was shown.
MIN_DOC_CHARS = 200

#: The four settlements of clause 4. `unsettled` is not in this tuple: it is
#: what a difference that cannot be placed becomes, and it is never counted as
#: agreement.
LABELS = ("pass_a_supported", "pass_b_supported", "both_supported", "neither_supported")
#: The labels `defects` adds, and the ONLY thing that ratio sums: the three labels
#: in which at least one pass wrote something the document does not back.
#: `both_supported` is excluded because it is the settlement saying neither pass
#: invented anything. This tuple is the single source for both the counter in
#: `audit_document` and the `defect_yield_definition` the artifact publishes, so
#: the definition a later reader weighs the number against cannot drift from the
#: arithmetic that produced it — which is what it would have done when settling
#: rewordings two-sided (#2918) put `both_supported` answers into a run that had
#: never counted one.
DEFECT_LABELS = ("pass_a_supported", "pass_b_supported", "neither_supported")

#: The two kinds of pair `difference_pairs` produces, and the axis the settlement
#: counts are split by: a `difference` is a key one pass holds alone (one reading),
#: a `rewording` is one key both passes hold in different words (two readings). The
#: split exists because the labels the two kinds can honestly reach differ —
#: `both_supported` is only ever a rewording's answer — and a run that reported them
#: as one four-way count could not tell a judge that started working from a pairing
#: that finally asked the question.
PAIR_KINDS = ("difference", "rewording")

UNSETTLED = "unsettled"

SETTLE_PROMPT = """Two independent extraction passes read this document and disagreed about one fact. \
Settle it from the DOCUMENT. Not from which pass produced a reading, not from \
how confident either sounds, not by majority: from what the text says.

Document:
{document}

Reading A (only pass A extracted this): {fact_a}
Reading B (only pass B extracted this): {fact_b}

Reply with JSON only, nothing else:
{{"label": "pass_a_supported" | "pass_b_supported" | "both_supported" | "neither_supported", \
"span": "the shortest verbatim stretch of the document that settles it, or null if the \
document settles it by NOT containing either reading"}}

`both_supported` means the document supports A and B as two different real facts \
(the passes split them, neither invented one). `neither_supported` means the \
document supports neither, so at least one pass wrote something it did not say.
"""


VAULT_ROOT = Path.home() / "obsidian"


def _load_module(path: Path, name: str):
    """Load a module by path, registered under `name` first — the order that lets
    a dataclass defined in it resolve its own annotations (a `__module__` that is
    not in `sys.modules` makes `dataclasses` reach for a missing entry)."""
    if not Path(path).exists():
        raise SystemExit(f"module not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_extractor():
    if not EXTRACTOR_PATH.exists():
        raise SystemExit(f"extractor module not found: {EXTRACTOR_PATH}")
    return _load_module(EXTRACTOR_PATH, "fe_under_audit")


def _read_config(config_path: Path = PIPELINE_CONFIG_PATH) -> dict:
    """The pipeline's own YAML, or a stop. `yaml` is already how the nightly job
    reads this file, so a config this audit cannot parse is a config it must not
    sample against, and a config that yields no corpus is the one failure that
    would otherwise quietly audit the whole vault."""
    import yaml  # in requirements.txt and already a dependency of the job

    path = Path(config_path)
    if not path.exists():
        print(f"  ✗ pipeline config not found: {path}", file=sys.stderr)
        raise SystemExit(2)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _self_written_memory_notes(path: Path = NIGHTLY_PATH) -> frozenset:
    """The job's own set of pipeline-generated `memory/` notes, read out of its
    source text rather than by importing the job.

    Importing `nightly_extraction.py` to reach one frozenset would drag in its whole
    world — `relations_index`, `content_hasher`, `app.kg_store`, a thread pool — and
    an audit that cannot run because a scheduler-side module is unimportable reports
    nothing at all. So the assignment is parsed out of the source: still the job's
    list, never a copy of mine, and if it moves, changes shape or stops being a
    literal, this refuses instead of guessing. #487 is what a stale list produced —
    two generated reports stayed in that set nine days past their generators, so the
    extractor kept skipping files it had stopped writing — and a literal copy in this
    file would rebuild exactly that drift inside the audit, silently.
    """
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        raise SystemExit(f"cannot read the corpus rules from {path}: {exc}")
    for node in tree.body:
        targets = ([node.target] if isinstance(node, ast.AnnAssign)
                   else getattr(node, "targets", []))
        if not any(isinstance(x, ast.Name) and x.id == "_SELF_WRITTEN_MEMORY_NOTES"
                   for x in targets):
            continue
        target = node.value
        if (isinstance(target, ast.Call) and isinstance(target.func, ast.Name)
                and target.func.id in ("frozenset", "set", "tuple", "list")
                and len(target.args) == 1):
            target = target.args[0]      # the SET is the literal; the call is not
        try:
            value = ast.literal_eval(target)
        except ValueError:
            value = None
        if not isinstance(value, (frozenset, set, list, tuple)):
            raise SystemExit(
                f"{path}: _SELF_WRITTEN_MEMORY_NOTES is no longer a literal, so the "
                "audit cannot know which memory/ notes the pipeline regenerates — "
                "guessing that set is how #487 happened")
        return frozenset(value)
    raise SystemExit(f"{path}: no _SELF_WRITTEN_MEMORY_NOTES assignment found")


def _today_str() -> str:
    """The job's settling date string (`nightly_extraction.py:98`), local time: a
    daily note is appended to all day and settles tomorrow, which is a local calendar
    fact and not a UTC one."""
    return datetime.now().strftime("%Y-%m-%d")


def _fresh_cutoff() -> float:
    """The job's 24-hour mtime boundary (`nightly_extraction.py:597`), computed per
    walk rather than at import so a long audit run cannot drift across it."""
    return time.time() - 86400



def allow_listed_dirs(config_path: Path = PIPELINE_CONFIG_PATH,
                      vault_root: Path = VAULT_ROOT) -> list[str]:
    """The corpus's top-level directories, exactly as the nightly job derives them
    from `sources.paths` (`nightly_extraction.py:576`) — one vault directory per
    entry, each required to sit inside the vault.

    The directory NAMES are derived from those entries, not written down here,
    because the one thing a copied list cannot do is fail when the allow-list moves.
    Which dirs are in it matters less than who is not: `skills/`, `backlog/`,
    `architecture/`, `autonomy/`, `agents/`, `facts/`, `sessions/`, `templates/`,
    `images/` and `youtube/` are absent deliberately (pipeline_config.yaml:18-22),
    so sampling one would audit a document the write path never reads and the rate
    would say nothing about the write path. An entry outside the vault is a typo
    rather than a corpus: the job drops it with a warning (`:586`) and so does this.
    """
    cfg = _read_config(config_path)
    raw = cfg.get("sources", {}).get("paths") or []
    if isinstance(raw, str):
        raw = [raw]
    root = Path(str(vault_root)).expanduser().resolve()
    dirs: list[str] = []
    skipped: list[str] = []
    for entry in raw:
        candidate = Path(str(entry)).expanduser()
        try:
            rel = candidate.resolve().relative_to(root)
        except ValueError:
            skipped.append(str(entry))
            continue
        if not rel.parts:
            continue
        if rel.parts[0] not in dirs:
            dirs.append(rel.parts[0])
    if skipped:
        print(f"  ⚠ {config_path}: sources.paths entry outside the vault, ignored: "
              + ", ".join(skipped), file=sys.stderr)
    if not dirs:
        print(f"  ✗ {config_path} lists no usable sources.paths; refusing to sample "
              "nothing and call the result a measurement.", file=sys.stderr)
        raise SystemExit(2)
    return dirs


def _corpus_filters(config_path: Path = PIPELINE_CONFIG_PATH, *, fresh_only: bool = False,
                    rules_today: str | None = None) -> dict:
    """Every filter the job applies between an allow-listed directory and an
    eligible file, gathered from where they actually live.

    `exclude_patterns` is applied the job's own way — `pat.strip("*/") in path_str`
    over the ABSOLUTE path, `nightly_extraction.py:615` — and the two hardcoded
    drop-lists (`:610`, `:621`) and the self-written-notes set are the job's, not
    restatements of them. `cutoff_ok` (`:597`) is deliberately NOT here: it is the
    incremental-mode "changed in the last 24h" filter, not a property of the corpus,
    and a sampled audit of the corpus is a full-mode question. It is wired here as
    `fresh_only` rather than dropped, because the window is load-bearing in a way
    that would starve the draw: measured on this box 2026-10-13, `knowledge/` holds
    3053 `.md` files and 15 were touched in the last 24 hours, `projects/` 1 of 196,
    `people/` 0 of 31, `personal/` 0 of 31, `work/` 0 of 57, `memory/` 7 of 709 — 23
    eligible documents across six directories, so a 40-document stratified sample is
    not drawable under it. The artifact records which mode a run used.
    """
    cfg = _read_config(config_path)
    return {
        "excludes": [str(x) for x in (cfg.get("sources", {})
                                      .get("exclude_patterns") or [])],
        "plumbing": ("node_modules", "/.venv", "/.cache", "/.git/", "__pycache__"),
        "never": ("/facts/", "/_pipeline/", "/skills/", "/agents/", "/autonomy/",
                  "/memory/vault-maintenance/"),
        "self_written": sorted(_self_written_memory_notes()),
        "today": str(rules_today or _today_str()),
        "max_bytes": int(cfg.get("extraction", {}).get("max_file_size") or 5_242_880),
        "fresh_only": bool(fresh_only),
    }


def _eligible(md_file: Path, root: Path, rules: dict) -> bool:
    """The job's per-file walk (`nightly_extraction.py:606-634`) for one file."""
    try:
        if not md_file.is_file():
            return False
        path_str = str(md_file)
        if any(part in path_str for part in rules["plumbing"]):
            return False
        if any(pat.strip("*/") and pat.strip("*/") in path_str
               for pat in rules["excludes"]):
            return False
        if any(skip in path_str for skip in rules["never"]):
            return False
        if md_file.name in rules["self_written"]:
            return False
        # A daily note is appended to all day; it settles tomorrow.
        if md_file.parent.name == "memory" and md_file.stem >= rules["today"]:
            return False
        if rules["fresh_only"] and md_file.stat().st_mtime < _fresh_cutoff():
            return False
        if md_file.stat().st_size >= rules["max_bytes"]:
            return False
        md_file.relative_to(root)
    except (OSError, ValueError):
        return False
    return True


def _candidates(vault_root: Path, dirs: list[str], rules: dict) -> dict[str, list[Path]]:
    """Every file in the corpus under each allow-listed directory, in a
    deterministic sort so the only randomness in the draw is the seed."""
    root = Path(vault_root)
    out: dict[str, list[Path]] = {}
    for d in dirs:
        base = root / d
        out[d] = [f for f in sorted(base.rglob("*.md")) if _eligible(f, root, rules)] \
            if base.is_dir() else []
    return out


def sample_documents(vault_root: Path, n: int, seed,
                     config_path: Path = PIPELINE_CONFIG_PATH,
                     dirs: list[str] | None = None, fresh_only: bool = False,
                     today: str | None = None) -> dict:
    """A seeded draw over the allow-listed extraction dirs that covers each one.

    Coverage is allocated first and filled second: one slot per directory in the
    configured order, the remainder shared round-robin, and a directory holding
    fewer files than its share yields its whole contents while the shortfall is
    re-covered from the directories that have more. So `n == 40` over six dirs is 6
    plus 34 spread across them, and `n == 6` is exactly one each. A plain
    unstratified draw of 40 from the corpus would land almost nowhere near all six,
    and the rate from it would be a rate for whichever directory is largest.

    The return carries the seed, the drawn paths, the per-directory share, candidate
    and drawn counts, and the `sources.paths` the directories came from, so the
    artifact shows its own shape and re-running the SAME sample is one command with
    the same `--seed`.
    """
    vault_root = Path(vault_root)
    dirs = dirs if dirs is not None else allow_listed_dirs(config_path, vault_root)
    rules = _corpus_filters(config_path, fresh_only=fresh_only,
                          rules_today=today)
    by_dir = _candidates(Path(vault_root), dirs, rules)
    missing = [d for d in dirs if not by_dir[d]]
    if missing:
        raise SystemExit(f"allow-listed extraction dir has no eligible file to sample: "
                         f"{', '.join(missing)} (under {vault_root})")

    quotas = {d: 1 for d in dirs}
    remaining = max(0, int(n) - len(dirs))
    i = 0
    while remaining and any(len(by_dir[d]) > quotas[d] for d in dirs):
        d = dirs[i % len(dirs)]
        if len(by_dir[d]) > quotas[d]:
            quotas[d] += 1
            remaining -= 1
        i += 1

    rng = random.Random(seed)
    picked: list[tuple[str, Path]] = []
    taken: dict[str, int] = {}
    for d in dirs:
        shuffled = sorted(by_dir[d])
        rng.shuffle(shuffled)
        take = shuffled[: quotas[d]]
        picked.extend((d, f) for f in take)
        taken[d] = len(take)
    picked.sort(key=lambda t: str(t[1]))
    return {
        "seed": seed, "seed_type": type(seed).__name__, "requested": int(n), "dirs": list(dirs),
        "sources_paths": [str(Path(str(vault_root)) / d) for d in dirs],
        "exclude_patterns": rules["excludes"], "never_in_corpus": list(rules["never"]),
        # The eligibility echo: the rules this draw was actually taken under, so a
        # reader can tell "7 documents from each of six dirs" from "1, 1 and none".
        "max_file_size": rules["max_bytes"], "settled_daily_notes_only_before": rules["today"],
        "corpus_rules": {k: (sorted(v) if isinstance(v, (list, set, frozenset, tuple)) else v)
                         for k, v in rules.items()},
        "fresh_only": bool(rules["fresh_only"]),
        "dir_share": quotas, "dir_candidates": {d: len(by_dir[d]) for d in dirs},
        # Vault-RELATIVE, the form every other path in this system takes (and the
        # form qmd stores its own rows in): an absolute draw cannot be re-run
        # against a different vault root, and `str(p.split("/")[0])`-style checks on
        # an absolute path silently return the empty string instead of a directory.
        "dir_drawn": taken,
        # Vault-RELATIVE, the form every other path in this system takes and the
        # form qmd stores its own rows in: an absolute draw cannot be re-run
        # against a different vault root, and a check like
        # `p.split("/")[0] == "knowledge"` on an absolute path silently returns
        # the directory two up instead of the corpus directory.
        "paths": [str(p.relative_to(Path(vault_root))) for _d, p in picked],
        "docs": [{"dir": d, "path": str(p.relative_to(Path(vault_root)))}
                 for d, p in picked],
        # For the run loop only, and deliberately NOT echoed into the artifact's
        # sample block: `paths` + `vault_root` is the re-runnable record, and a
        # second absolute copy of the same draw in the artifact invites a reader to
        # pin the one that stops being true when the vault moves.
        "paths_abs": [str(p) for _d, p in picked],
        "vault_root": str(vault_root.resolve()),
    }


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_FILLER_RE = re.compile(
    r"^(the |a |an |it is |this is |there (?:is|are) |has |have |had |currently |"
    r"as of [^,]+, |note:? |lloyd (?:is|uses|has) )")


def canonical_text(text: str) -> str:
    """Wording-insensitive form of a fact, for the rewording bucket only.

    Deliberately coarser than an exact hash and deliberately NOT used as the
    primary match: `fact_identity(entity, category, fact_text)` is the store's
    triple but its own docstring says it is not canonicalised through the alias
    table and its text component is an exact `text_hash` — keying a diff on that
    makes every paraphrase a difference and reports the model's synonym habit as
    a defect. Here a wording-only match is a `rewording`, counted apart from a
    difference; the difference bucket is what settlement reads.
    """
    t = (text or "").strip().lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    prev = None
    while prev != t:
        prev = t
        t = _FILLER_RE.sub("", t).strip()
    return t


def fact_key(entity: str, category: str, text: str, resolve=None) -> tuple:
    """`(canonical entity, category, canonical text)` — clause 3's key.

    Entity names go through `app/entity_naming`, which is the alias resolution
    the store itself uses, so "Lloyd" and "lloyd (this instance)" do not open two
    buckets and a reworded fact does not collapse into a different fact.
    """
    resolve = resolve or _resolve_entity
    return (resolve(entity) or "", (category or "general").strip().lower(), canonical_text(text))


_RESOLVE_FAIL = object()


def _resolve_entity(name: str) -> str | None:
    """`app.entity_naming.resolve_entity`, once loaded, and never registered.

    Read-only by construction: `resolve_entity(name, enforce=False)` can attach a
    name to its declared canonical, which is a write to the alias table, and an
    audit that measures the corpus is not the thing that edits it. The lookup
    below is the same map, read.
    """
    if not name:
        return None
    global _ALIAS_MAP
    if _ALIAS_MAP is _RESOLVE_FAIL:
        return name.strip()
    if _ALIAS_MAP is None:
        try:
            from app.entity_naming import _load_alias_map
            _ALIAS_MAP = _load_alias_map()
        except Exception:  # noqa: BLE001 — a missing alias map must not fake a match
            _ALIAS_MAP = _RESOLVE_FAIL
            return name.strip()
    return _ALIAS_MAP.get(str(name).strip().lower(), str(name).strip())


_ALIAS_MAP: dict | None | object = None


def reload_alias_map() -> None:
    """Drop the cached alias map (a test that writes a fixture map calls this)."""
    global _ALIAS_MAP
    _ALIAS_MAP = None


def facts_dir() -> Path:
    """The derived fact tree the extraction writes into, read through `app.paths`
    like the writers do (`fact_extractor.py:56, 245`).

    A defect citation has to name a file the writer actually uses, so the root is
    the pipeline's own rather than a path guessed from this module's location. With
    `app.paths` unimportable the fallback is the production shape, because a
    citation that is merely unresolvable is better than a citation with no root at
    all — and `fact_file_path` still says which root it used by returning it.
    """
    try:
        from app.paths import VAULT_FACTS_ROOT
        return Path(VAULT_FACTS_ROOT)
    except Exception:  # noqa: BLE001 — the citation degrades to the default shape
        return Path.home() / "lloyd-data" / "_pipeline" / "vault-derived" / "facts"


def _path_safe(name: str) -> str:
    """A filesystem name out of a model's phrase, in the extractor's shape.

    `_sanitize_entity` (`fact_extractor.py:533`) delegates to `_gate_entity`
    (`fact_extractor.py:494`), and its shaping is three operations at
    `fact_extractor.py:507-511`: keep the last path segment, delete the characters
    `<>:"|?*`, collapse runs of whitespace. Case is PRESERVED there, so it is
    preserved here — the live store's entity directories are lower-case because the
    prompt's canonical entity list is, not because the gate folded them, and a
    citation that lower-cased on its own would name a directory the writer never
    made for an entity spelled with a capital.

    The same three operations, in the same direction: DELETED rather than
    substituted, so a citation and the note the writer files agree character for
    character wherever the gate is defined. A trailing dot is dropped too, which is
    what makes `..` collapse to nothing instead of to a hop out of the facts tree.

    The gate is NOT called, and not for the reason a comment used to give (it no
    longer registers anything — `fact_extractor.py:499-501` records that the junk
    predicate was moved AHEAD of the alias table precisely so a refused name never
    enters it). The reason is that it is a gate and not a formatter: it judges the
    name against the entity registry and returns `""` for one it calls junk or
    candidate (`fact_extractor.py:543-545`), so a citation routed through it would
    silently lose its path whenever the registry disagreed with the model — and it
    wants the source document and declared type, which an audit over a corpus has no
    business guessing. Shaping is all a locator needs.
    """
    s = str(name or "").strip().split("/")[-1].split("\\")[-1]
    s = re.sub(r"[<>:\"|?*]", "", s)
    s = re.sub(r"\s+", " ", s).strip().rstrip(".")
    return s or "unknown"


def fact_file_path(entity, category, facts_root=None) -> str:
    """The note this fact is filed under: `self.facts_dir / entity /
    f"{entity}-{category}.md"`, the writer's own pattern
    (`fact_extractor.py:479`).

    The point is that citing a settled difference stops being a grep. A record that
    named only the fact's text could be chased to its note only by searching the
    fact tree for that text — and a text the two passes worded differently has two
    plausible greps, which is how a defect report turns into a hunt.

    Two caveats a reader needs, both stated rather than discovered: the ENTITY is
    shaped here rather than judged (see `_path_safe`), so an entity that aliases to
    another is filed under the canonical name this module cannot resolve without the
    alias store; and the CATEGORY segment is lower-cased with non-word characters
    replaced, which is this module's shaping, whereas the writer maps the category
    onto a fixed 287-spelling-vocabulary-first `normalize_category`
    (`fact_extractor.py:127`, used at `:476` and `:689`). For a category already in
    that vocabulary the two agree — both lower-case it and neither leaves a slash in
    it — and for one outside it they do not, because that one lands on whatever
    token-overlap winner `normalize_category` picks, or on `general`. Which is why
    the record carries the raw `category` field beside the path: the path is a
    locator to open, not a claim that a row exists at it.
    """
    root = Path(facts_root) if facts_root else facts_dir()
    safe = _path_safe(entity)
    cat = re.sub(r"[^\w\-]", "_", str(category or "").strip().lower()) or "general"
    return str(root / safe / f"{safe}-{cat}.md")


def diff_passes(facts_a: list[dict], facts_b: list[dict], resolve=None) -> dict:
    """Match pass A against pass B on the clause-3 key, three buckets deep.

    The key is `(canonical entity, category, canonical text)`, so a key that both
    sides hold is an agreement, and the only thing left to ask of it is whether
    the sentence is the same sentence: raw text equal → `identical`, raw text
    differing while the canonical text matched → `rewording`. A rewording is the
    two passes agreeing in a different accent, and counting it as a difference is
    exactly the inflation the item's acceptance note warns about — "zero defects
    across 40 documents means the normalisation is too coarse" cuts both ways.
    A key held by one side only is a real disagreement, split by which side.
    """
    def bucket(facts):
        out: dict[tuple, list[dict]] = {}
        for f in facts:
            k = fact_key(f.get("entity") or "", f.get("category") or "",
                         f.get("fact") or "", resolve)
            out.setdefault(k, []).append(f)
        return out

    ba, bb = bucket(facts_a), bucket(facts_b)
    identical = 0
    rewordings: list[dict] = []
    only_a: list[dict] = []
    only_b: list[dict] = []
    for k in sorted(set(ba) | set(bb), key=str):
        la, lb = list(ba.get(k, [])), list(bb.get(k, []))
        while la and lb:
            fa, fb = la.pop(0), lb.pop(0)
            if (fa.get("fact") or "").strip() == (fb.get("fact") or "").strip():
                identical += 1
            else:
                rewordings.append({"a": fa["fact"], "b": fb["fact"],
                                   "entity": k[0], "category": k[1],
                                   "canonical_text": k[2]})
        only_a.extend(la)
        only_b.extend(lb)
    return {"identical": identical, "rewordings": rewordings,
            "only_a": only_a, "only_b": only_b,
            "differences": len(only_a) + len(only_b)}


# ── the two passes ──────────────────────────────────────────────────────────

class PairNotComparable(Exception):
    """The two calls were not the same question. Reported as an error, never as
    a disagreement: a rate from a non-comparable pair is a number about nothing."""


def build_prompts(fe, content: str) -> list[dict]:
    """Build the prompts ONCE: production chunker, production template.

    `_chunk_content` is reached as an unbound method because it reads no
    instance state — only the module's `CHUNK_SIZE` / `CHUNK_OVERLAP` /
    `MAX_CHUNKS` — so the audit gets the real windows without constructing an
    extractor, whose `__init__` opens the facts tree.
    """
    existing = "None"   # `extract_from_document`'s own default for existing_facts
    prompts = []
    for offset, chunk in fe.FactExtractor._chunk_content(None, content):
        known = fe._known_entities(chunk, 60)
        known_block = "\n".join(f"- {k}" for k in known) if known else "(none recognised)"
        prompts.append({
            "offset": offset,
            "chunk_chars": len(chunk),
            "text": fe.EXTRACTION_PROMPT.format(
                content=chunk, existing_facts=existing, known_entities=known_block,
                entity_types=" | ".join(fe.SCHEMA_TYPES)),
        })
    return prompts


def prompt_digest(prompts: list[dict]) -> str:
    return hashlib.sha256("\x1f".join(p["text"] for p in prompts).encode("utf-8")).hexdigest()


#: The temperature the production extraction samples at
#: (`fact_extractor.py:444`). Both production arms of this audit send it, and a
#: pass that sampled at any other value is not measuring production — which is
#: why the greedy control arm (#2918) is a separate, opt-in pair of passes
#: (`--temperature-0`) rather than a changed constant here.
DEFAULT_TEMPERATURE = 0.3

#: The control arm's temperature: greedy, so two passes of identical prompt bytes
#: differ only where the model cannot place its answer, not where it sampled.
GREEDY_TEMPERATURE = 0.0


def build_payload(prompt: str, temperature: float | None = None) -> dict:
    """The extractor's request payload, field for field, with one knob.

    Same system line, same `max_tokens` 6000, same `chat_template_kwargs`, same
    `priority` 2 as `_call_llm`'s literal (`fact_extractor.py:438-450`), which is
    what makes a pass here a pass of production. `temperature` defaults to
    `DEFAULT_TEMPERATURE`, production's own; naming it exists only so the greedy
    control arm can differ from production in that one field and nothing else.
    """
    return {
        "model": "primary",
        "messages": [
            {"role": "system", "content": "You are a fact extraction engine."},
            {"role": "user", "content": prompt},
        ],
        "temperature": DEFAULT_TEMPERATURE if temperature is None else temperature,
        "max_tokens": 6000,
        "chat_template_kwargs": {"enable_thinking": False},
        "priority": 2,
    }


def send(port: int, prompt: str, timeout: float = 240.0,
         temperature: float | None = None) -> tuple[str, dict, float]:
    """One chat call, and the only reason this module talks HTTP itself.

    The payload is `build_payload`'s — the extractor's own, and identical for both
    production arms, which is what the wire digest in `pair_guard` certifies. It
    differs in exactly one way: it keeps the `usage` field the extractor
    drops, which is where this audit's cost line comes from.
    """
    payload = build_payload(prompt, temperature=temperature)
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SystemExit(f"model endpoint on port {port} unreachable: {exc}") from exc
    seconds = time.monotonic() - started
    text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    usage = {k: int(data.get("usage", {}).get(k) or 0)
             for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    usage["seconds"] = round(seconds, 3)
    # The digest of the bytes actually posted. `pair_guard` compares the two
    # passes on THIS as well as on the prompts they were handed: a divergence
    # introduced between building a prompt and posting it is the one a
    # prompt-side check cannot see.
    usage["sent_sha256"] = hashlib.sha256(body).hexdigest()
    return text, usage, seconds


#: The token counters of a response's `usage` block, and the ONLY keys summed into
#: a meter. A named constant instead of `for k in usage` is a bug with a number
#: attached, not tidiness: the accumulator gained a `seconds` key after its first
#: call, so iterating it added `int(u["seconds"])` into the same key the next line
#: was already adding `float(u["seconds"])` to. Three 4.2-second calls therefore
#: metered 20.6 s against a true 12.6 s — which is what `send`'s own third element
#: says, and the artifact would have billed itself to Gallon's 7.5M-token figure
#: using a number inflated by its own double-add.
TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def new_cost() -> dict:
    """One meter: the calls, the tokens and the wall clock of one arm of a run."""
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "seconds": 0.0}


def book_call(cost: dict, usage: dict | None, seconds: float) -> dict:
    """Record exactly one model call against one meter.

    Every model call this module makes is booked here and nowhere else, which makes
    "each call counted once" a property of the code rather than of care: the passes
    book their prompts, the reconciler books its reads, and no call site is left
    with a second arithmetic to get wrong. `usage` may be None — a call whose
    transport raised has no token count to read, but it spent wall clock and it is
    still a call the run made.
    """
    u = usage or {}
    for k in TOKEN_KEYS:
        cost[k] += int(u.get(k) or 0)
    cost["seconds"] = round(cost["seconds"] + float(seconds or 0.0), 3)
    cost["calls"] += 1
    return cost


def merge_costs(*blocks: dict) -> dict:
    """The sum of DISJOINT meters: the run's total is derived from its arms, so a
    call counted twice would have to sit in two arms — and one meter per arm is the
    shape that makes that impossible rather than merely unlikely."""
    out = new_cost()
    for b in blocks:
        b = b or {}
        for k in TOKEN_KEYS:
            out[k] += int(b.get(k) or 0)
        out["seconds"] = round(out["seconds"] + float(b.get("seconds") or 0.0), 3)
        out["calls"] += int(b.get("calls") or 0)
    return out


def _as_answer(raw):
    """A settlement answer, however the judge reports it, as `(raw, usage, seconds)`.

    A judge may be a transport — the real one is, since it calls `send` — or a bare
    answer (every offline judge in the tests is one). A bare answer carries no
    tokens, so its meter reads zero tokens and the wall clock measured here: an
    under-read of tokens, stated rather than hidden, because the cost line is
    compared against a token figure taken from a paper.
    """
    if isinstance(raw, tuple):
        usage = raw[1] if len(raw) > 1 and isinstance(raw[1], dict) else {}
        seconds = float(raw[2]) if len(raw) > 2 and raw[2] is not None else None
        return raw[0], usage, seconds
    return raw, None, None


def run_pass(fe, prompts: list[dict], port: int, pass_id: str, sender=None,
             cost: dict | None = None, temperature: float | None = None) -> dict:
    """Send the pre-built prompts and parse the answers. Given nothing from the
    other pass — no shared facts, no prior list, no feedback. That is what makes
    the two of them a pair at all.

    `cost` is this arm's meter. One prompt sends one call, which books one entry in
    it. A call whose transport raised books no call (nothing came back to meter) and
    lands in `errors`, so `calls` below — prompts attempted — can exceed the metered
    calls by exactly the number of errors.

    `temperature` is left at production's own value unless an arm asks for another
    (the greedy control), and it is passed only when named: a two-argument transport —
    every offline fake in the suite — keeps working, and a production-mirror arm keeps
    sending `build_payload`'s unchanged bytes."""
    sender = sender or (lambda p, pt, temperature=None: send(pt, p, temperature=temperature))
    cost = new_cost() if cost is None else cost
    facts: list[dict] = []
    errors: list[str] = []
    sent_hashes: list[str] = []
    for p in prompts:
        try:
            if temperature is None:
                text, u, s = sender(p["text"], port)
            else:
                text, u, s = sender(p["text"], port, temperature=temperature)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 — a lost call is an error row, not a crash
            errors.append(f"offset {p['offset']}: {exc}")
            continue
        sent_hashes.append(str(u.get("sent_sha256") or ""))
        book_call(cost, u, s)
        parsed = fe.FactExtractor._parse_response(None, text)
        for f in parsed.get("facts") or []:
            if isinstance(f, dict) and (f.get("fact") or "").strip():
                facts.append({"entity": f.get("entity") or "", "category": f.get("category") or "",
                              "fact": (f.get("fact") or "").strip()})
    return {"pass_id": pass_id, "facts": facts, "usage": cost, "errors": errors,
            "calls": len(prompts),
            # What this pass was built from, and the digest of everything it put
            # on the wire — the two things `pair_guard` compares.
            "prompt_sha256": prompt_digest(prompts),
            "offsets": [[p["offset"], p["chunk_chars"]] for p in prompts],
            # Digest of the request bytes this pass posted, kept as an audit trail
            # for the cost line — not a pair check (see `pair_guard`).
            "sent_sha256": hashlib.sha256("|".join(sent_hashes).encode("utf-8")).hexdigest()}


def pair_guard(prompts: list[dict], a: dict, b: dict) -> None:
    """Refuse to report a pair that was not the same question.

    Same prompts, same offsets, same bytes on the wire. The digest and the offset
    list are both checked because they fail differently: a prompt-template edit
    under a running audit moves the digest while the windows look untouched, and a
    chunking change moves both.

    Independence is discharged by those checks rather than by a flag: `prompts` is
    built once and handed to both passes, so if the bytes and windows are equal then
    pass B's input contained no fact pass A produced — `run_pass` has no parameter
    through which another pass's facts could arrive. An earlier draft of this
    function carried `if res.get("injected_from_other_pass")` as well; it sat below
    the `raise` so it could never run, and nothing ever set that key, so it was a
    check the docstring promised and the code did not perform.
    """
    digest = prompt_digest(prompts)
    offsets = [(p["offset"], p["chunk_chars"]) for p in prompts]
    bad: list[str] = []
    for side, res in (("A", a), ("B", b)):
        if res.get("prompt_sha256") != digest:
            bad.append(f"pass {res.get('pass_id') or side} prompt digest "
                       f"{res.get('prompt_sha256')!r} != the digest the pair was built "
                       f"from {digest!r}")
        if [tuple(o) for o in res.get("offsets", [])] != offsets:
            bad.append(f"pass {res.get('pass_id') or side} chunk offsets "
                       f"{res.get('offsets')!r} != {offsets!r}")
    # What each pass actually put on the wire. The real `send` digests the request
    # body it posts, so a sender that edits a prompt between the build and the POST
    # cannot hide behind matching prompt digests above. Absent on both sides is not
    # a mismatch: a fake transport that reports no digest must not be refused for
    # not reporting one.
    if a.get("sent_sha256") and b.get("sent_sha256") and a["sent_sha256"] != b["sent_sha256"]:
        bad.append("the two passes put different bytes on the wire (sent digest "
                   f"{a['sent_sha256']!r} vs {b['sent_sha256']!r})")
    if bad:
        raise PairNotComparable("; ".join(bad) + " — the pair is not one question put "
                                "twice, and a disagreement rate taken from it would "
                                "be a measurement of nothing")


# ── settlement against the source ──────────────────────────────────────────

def _collapse(source: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed text, plus the raw offset each collapsed character
    came from, so a hit found in the collapsed form can be reported as a real
    offset into the real document rather than as a guess."""
    out: list[str] = []
    starts: list[int] = []
    for i, ch in enumerate(source):
        if ch.isspace():
            if out and out[-1] != " ":
                out.append(" ")
                starts.append(i)
        else:
            out.append(ch)
            starts.append(i)
    return "".join(out), starts


def _locate(source: str, span: str) -> dict | None:
    """Where the settling stretch sits: exact search first, whitespace-insensitive
    second, raw offsets either way.

    A span this cannot place is not a span — it is the judge quoting something the
    document does not say, which is a defect in the *settlement* and belongs in
    `unsettled`, never in the agreement count. Both forms report offsets into the
    document as stored, because a citation whose offsets are approximate is worse
    than no citation: the whole point of the span is that a later reader can open
    the file at that place and check the verdict without re-running anything.
    """
    if not span or not span.strip():
        return None
    needle = span.strip()
    i = source.find(needle)
    if i >= 0:
        return {"start": i, "end": i + len(needle), "text": needle, "form": "exact"}
    flat, starts = _collapse(source)
    flat_needle = _WS_RE.sub(" ", needle)
    j = flat.find(flat_needle)
    if j < 0:
        return None
    last = starts[j + len(flat_needle) - 1]
    return {"start": starts[j], "end": last + 1, "text": source[starts[j]:last + 1],
            "form": "collapsed"}


def settle(diff: dict, source: str, judge=None, port: int = 8096,
           meter: dict | None = None) -> dict:
    """Place one difference in exactly one of the four labels, by reading the source.

    Never majority, never confidence, never "both said it so it must be true":
    the judge is shown the document and the two readings and answers for the
    label plus the verbatim stretch that settles it. A label outside the four, a
    single-sided answer with no locatable span, or a judge that could not be
    reached makes the difference `unsettled` — recorded in its own bucket with
    the reason, and never folded into agreement. An empty answer is not
    `both_supported` either way: `neither_supported` with a null span means the
    document settled it by absence, which is a real result.

    `meter` is the run's settlement cost block, and every judge call is booked
    into it — including one that raised or answered gibberish, both of which spent
    the wall clock. Gallon's 7.5M-token figure is the cost of a reconciliation that
    INCLUDES the reconciler, so a cost line built from the two passes alone is
    compared against the wrong thing: an audit whose reconciler is unmetered
    under-states the method it is citing.
    """
    # The default judge returns `send`'s whole triple rather than `[0]`: the text
    # alone carries no token count, so a settlement arm built on the transport and
    # then asked for its cost could only report wall clock. `_as_answer` unpacks the
    # triple, and a bare-answer judge (every offline one) still works.
    judge = judge or (lambda doc, fa, fb, pt: send(pt, SETTLE_PROMPT.format(
        document=doc[:20000], fact_a=fa or "(none)", fact_b=fb or "(none)")))
    fact_a = (diff.get("a") or {}).get("fact") or ""
    fact_b = (diff.get("b") or {}).get("fact") or ""
    started = time.perf_counter()
    try:
        raw = judge(source[:20000], fact_a, fact_b, port)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        if meter is not None:
            book_call(meter, None, time.perf_counter() - started)
        return {"label": UNSETTLED, "reason": f"judge unreachable: {exc}", "span": None}
    raw, usage, reported = _as_answer(raw)
    if meter is not None:
        book_call(meter, usage, reported if reported is not None
                  else time.perf_counter() - started)
    if isinstance(raw, dict):
        label, span = raw.get("label"), raw.get("span")
    else:
        try:
            parsed = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip()))
        except Exception:  # noqa: BLE001 — an unparseable verdict settles nothing
            return {"label": UNSETTLED, "reason": f"unparseable settlement: {(raw or '')[:120]!r}",
                    "span": None}
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        label, span = parsed.get("label"), parsed.get("span")
    if label not in LABELS:
        return {"label": UNSETTLED, "reason": f"label outside the four: {label!r}", "span": None}
    located = _locate(source, span if isinstance(span, str) else "")
    if label != "neither_supported" and located is None:
        return {"label": UNSETTLED,
                "reason": f"{label} named no stretch that is in the document: {span!r}",
                "span": None}
    if located is None and label in ("pass_a_supported", "pass_b_supported",
                                      "both_supported"):
        # A label that names a supporting stretch must point at one. `both_supported`
        # with nothing to point at is not "the document backs both" — it is the judge
        # declining to say which stretch backs either, and counting it as agreement
        # is the one outcome that would report two real facts where nothing was
        # checked. Only `neither_supported` may answer spanless: the document
        # settling that contains NEITHER reading is itself the evidence.
        return {"label": UNSETTLED, "span": None,
                "reason": f"{label} named no stretch in the document"}
    return {"label": label, "span": located}


def difference_pairs(diff: dict) -> list[dict]:
    """The pairs to put to the source, each labelled with what it can honestly show.

    An only-A fact and an only-B fact are settled separately — pairing them by
    position would invent a comparison the passes never made — and each is ONE
    reading: the judge is given the claim and an explicit `(none)` for the other
    pass, so `both_supported` cannot be its answer, and `side` records which pass is
    claiming it alone.

    A rewording is TWO readings of one claim — the same canonical text in two
    accents — and it is the only class this instrument produces that can honestly be
    answered `both_supported`. It used to be dropped on the floor here: this function
    read only `only_a`/`only_b`, so the 2026-10-17 run put 1,007 settlements to the
    judge and none of its 11 rewordings, and every pair it did put came back
    single-sided by construction. That is why the label sat at 0, and why the fix is
    the pairing rather than the prompt: "does the document back BOTH of these?" is
    the question that separates a paraphrase of one real fact from two passes that
    each invented a different one.
    """
    pairs = [{"a": f, "b": None, "side": "a", "kind": "difference", "two_sided": False}
             for f in diff["only_a"]]
    pairs += [{"a": None, "b": f, "side": "b", "kind": "difference", "two_sided": False}
              for f in diff["only_b"]]
    pairs += [{"a": {"fact": r["a"], "entity": r["entity"], "category": r["category"]},
               "b": {"fact": r["b"], "entity": r["entity"], "category": r["category"]},
               "side": "either", "kind": "rewording", "two_sided": True,
               "entity": r["entity"], "category": r["category"],
               "canonical_text": r["canonical_text"]}
              for r in diff.get("rewordings", [])]
    return pairs


def settlement_record(pair: dict, verdict: dict, facts_root=None) -> dict:
    """One settled difference, as a row a reader can go and check.

    This is what the artifact never had: `settled` was a `dict[str, int]`, so a run
    could publish "329 defects" and say nothing about which ones, and the one piece
    of information that makes a defect actionable — which note of the derived fact
    tree holds it — appeared nowhere in the module. A record therefore carries the
    key the difference was matched on (entity, category, canonical text), both
    readings as the passes wrote them, the span that settled it, and the fact file
    the extractor writes by. `label` and `kind` are the two axes the counters split
    on, and every counter in the artifact is a count of these rows.
    """
    a, b = pair.get("a") or {}, pair.get("b") or {}
    entity = pair.get("entity") or a.get("entity") or b.get("entity") or ""
    category = pair.get("category") or a.get("category") or b.get("category") or ""
    canonical = pair.get("canonical_text") or canonical_text(
        a.get("fact") or b.get("fact") or "")
    span = verdict.get("span")
    return {"kind": pair.get("kind", "difference"),
            "label": verdict["label"],
            "side": pair.get("side"),
            "two_sided": bool(pair.get("two_sided", False)),
            "entity": _resolve_entity(entity) if entity else "",
            "category": str(category or "").strip().lower(),
            "canonical_text": canonical,
            "fact_a": a.get("fact"),
            "fact_b": b.get("fact"),
            "span": span,
            "span_form": (span or {}).get("form"),
            "fact_file": fact_file_path(entity, category, facts_root),
            "reason": verdict.get("reason")}


# ── the audit ───────────────────────────────────────────────────────────────

def audit_document(fe, path: Path, vault_root: Path, port: int, settle_port: int | None = None,
                   sender=None, judge=None, facts_dir=None, temperature_0: bool = False) -> dict:
    """Audit one document: two (optionally four) passes, then settle what differs.

    `temperature_0` adds a THIRD and FOURTH pass of the same prompt bytes, decoded
    greedily, and reports their disagreement beside the production pair's. It is
    opt-in because the two production passes have to keep sending the extractor's own
    payload byte for byte — the wire digest `pair_guard` compares is the witness that
    they do — and a control arm that came along by default would double the run's
    cost on every audit that only wanted the headline rate.

    `facts_dir` is the derived fact tree settlement records cite. It defaults to the
    pipeline's own (`facts_dir()`), and a test passes its own so the citation is
    asserted as a path rather than as a string copied out of this file.
    """
    rel = str(path.relative_to(vault_root))
    source = path.read_text(encoding="utf-8", errors="replace")
    prompts = build_prompts(fe, source)
    digest = prompt_digest(prompts)
    offsets = [[p["offset"], p["chunk_chars"]] for p in prompts]
    # One meter per arm. A call belongs to exactly one arm, so the run's cost is
    # their disjoint sum: pass A's prompts, pass B's prompts, every read the
    # reconciler made to settle a difference, and (opt-in) the greedy control pair.
    cost_a, cost_b, cost_settle, cost_greedy = new_cost(), new_cost(), new_cost(), new_cost()
    a = run_pass(fe, prompts, port, "A", sender, cost=cost_a)
    b = run_pass(fe, prompts, port, "B", sender, cost=cost_b)
    pair_guard(prompts, a, b)          # raises rather than reporting a bad pair
    diff = diff_passes(a["facts"], b["facts"])
    settled: dict[str, int] = {label: 0 for label in LABELS}
    settled_by_kind: dict[str, dict[str, int]] = {
        kind: {label: 0 for label in LABELS} for kind in PAIR_KINDS}
    settlements: list[dict] = []
    unsettled: list[dict] = []
    spans: list[dict] = []
    for pair in difference_pairs(diff):
        verdict = settle(pair, source, judge, settle_port or port, meter=cost_settle)
        if verdict["label"] == UNSETTLED:
            unsettled.append({"a": (pair["a"] or {}).get("fact"),
                              "b": (pair["b"] or {}).get("fact"),
                              "kind": pair.get("kind", "difference"),
                              "reason": verdict["reason"]})
            continue
        record = settlement_record(pair, verdict, facts_dir)
        settlements.append(record)
        settled[verdict["label"]] += 1
        settled_by_kind[record["kind"]][verdict["label"]] += 1
        if verdict.get("span"):
            spans.append(verdict["span"])
    # The denominator is every fact BOTH passes held: identical matches, rewordings
    # (same canonical text, different words) and the one-sided keys, which are the
    # disagreements inside it. Rewordings are counted, because a rewording IS a fact
    # both passes produced.
    compared = diff["identical"] + len(diff["rewordings"]) + diff["differences"]
    n_settled = sum(settled.values())
    # `DEFECT_LABELS` is the whole of the defect count, and the same tuple builds
    # the definition the artifact publishes, so the two cannot disagree. What the
    # labels mean: a single-sided settlement is one pass holding a reading the
    # document supports and the other not, and `neither_supported` is the document
    # backing neither — a defect in at least one pass either way. `both_supported`
    # is excluded because it is the settlement finding nothing invented: the passes
    # split ONE supported fact into two wordings, which is an instrument artefact of
    # keying on canonical text, not an error either pass made.
    defects = sum(settled[label] for label in DEFECT_LABELS)

    t0_row: dict = {"differences_t0": None, "identical_t0": None, "rewordings_t0": None,
                    "compared_t0": None, "disagreement_rate_t0": None}
    if temperature_0:
        # Two greedy passes of the SAME prompts, then the same diff. Greedy decoding
        # of identical bytes is what the production pair's rate would be with the
        # sampling removed, so the gap between it and `disagreement_rate` is the part
        # of the headline number that is the decoder and not the corpus. The guard
        # runs here too: the control is only a control if its two passes were asked
        # the same question, and a `PairNotComparable` from a control arm must not
        # take the production measurement down with it.
        try:
            c = run_pass(fe, prompts, port, "greedy-A", sender, cost=cost_greedy,
                         temperature=GREEDY_TEMPERATURE)
            d = run_pass(fe, prompts, port, "greedy-B", sender, cost=cost_greedy,
                         temperature=GREEDY_TEMPERATURE)
            pair_guard(prompts, c, d)
            t0 = diff_passes(c["facts"], d["facts"])
            t0_compared = t0["identical"] + len(t0["rewordings"]) + t0["differences"]
            t0_row = {
                "differences_t0": t0["differences"],
                "identical_t0": t0["identical"],
                "rewordings_t0": len(t0["rewordings"]),
                "compared_t0": t0_compared,
                "disagreement_rate_t0": (round(t0["differences"] / t0_compared, 4)
                                        if t0_compared else None),
            }
        except PairNotComparable as exc:
            t0_row = {**t0_row, "t0_error": str(exc)}
    # Derived from the disjoint arms rather than accumulated alongside them: the
    # total of a run is a function of meters that each counted their own calls, so
    # there is no second place a call could be added. It is built AFTER every arm has
    # run — the control's two greedy passes included — because a total derived before
    # the last arm filled its meter would under-bill the run by exactly that arm, and
    # the artifact bills itself against Gallon's 7.5M with this number.
    cost_total = merge_costs(cost_a, cost_b, cost_settle, cost_greedy)
    return {
        "path": rel,
        # The root this document was read from, as given to the run: a reader
        # re-checking a span needs the file, and `path` is vault-relative by
        # contract, so the root has to travel with it. NOT an absolute path of the
        # document — that would pin one machine's tree into a re-runnable record.
        "vault_root": str(Path(vault_root).resolve()),
        "category": Path(rel).parts[0],
        "chars_total": len(source),
        "chars_covered": max([o + c for o, c in offsets] or [0]),
        "prompt_sha256": digest,
        "chunk_offsets": offsets,
        "facts_a": len(a["facts"]), "facts_b": len(b["facts"]),
        "identical": diff["identical"],
        "rewordings": len(diff["rewordings"]),
        "differences": diff["differences"],
        "compared": compared,
        "disagreement_rate": round(diff["differences"] / compared, 4) if compared else None,
        # ── the greedy control arm (`--temperature-0`), None when it did not run ──
        # Absent is reported as absent, never as 0.0: a control that did not run and
        # rendered as a perfect score would be the most convincing wrong number in
        # the file.
        **t0_row,
        "settled": settled,
        "settled_total": n_settled,
        "settled_by_kind": settled_by_kind,
        # One row per settlement, and the counters above are counts of these rows:
        # `defects` is recomputable from here without re-reading a single document.
        "settlements": settlements,
        "unsettled": unsettled,
        "defects": defects,
        "defect_yield": round(defects / n_settled, 4) if n_settled else None,
        "supporting_spans": spans,
        # The arms are disjoint, so the document's cost is their sum and a call
        # cannot be in two of them: the reconciler's reads are in `settlement` and
        # nowhere else, the greedy control's in `t0`. `total_tokens`/`seconds` are the
        # figure the artifact bills itself against, and they now include the
        # reconciler — which is what makes the comparison to Gallon's 7.5M an
        # apples-to-apples one, since his figure is the cost of the whole
        # reconciliation and not of the two passes.
        "cost": {**cost_total, "a": cost_a, "b": cost_b, "settlement": cost_settle,
                 "t0": cost_greedy},
        "pass_errors": a["errors"] + b["errors"],
    }


def summarize(sample: dict, docs: list[dict], errors: list[dict], started: float) -> dict:
    differences = sum(d["differences"] for d in docs)
    settled_total = sum(d["settled_total"] for d in docs)
    settled = {label: sum(d["settled"][label] for d in docs) for label in LABELS}
    unsettled = sum(len(d["unsettled"]) for d in docs)
    defects = sum(d["defects"] for d in docs)
    return {
        "schema": "extraction-disagreement/1",
        # The drawn paths travel with the seed, which is what makes a re-run of
        # THIS sample one command instead of a reconstruction.
        # The seed AND the drawn paths, so a re-run of THIS sample is one command
        # rather than a reconstruction: the draw is a function of
        # (config, vault, n, seed), and pinning the seed alone would redraw a
        # different corpus the day `sources.paths` moved.
        "sample": {k: sample[k] for k in ("seed", "seed_type", "requested", "dirs",
                                          "paths", "dir_share", "dir_candidates",
                                          "dir_drawn", "sources_paths",
                                          "exclude_patterns", "never_in_corpus",
                                          "corpus_rules", "fresh_only",
                                          "settled_daily_notes_only_before",
                                          "max_file_size")},
        # Which vault the draw came from, so the relative paths in it resolve and the
        # whole draw can be re-run against a different root on purpose.
        "vault_root": sample["vault_root"],
        # What was ASKED for, beside what was read: a draw that came up short (a
        # directory with too few eligible files) has to be visible next to the rate,
        # or a 9-document audit reads as a 40-document one.
        "requested_documents": sample["requested"],
        "documents_measured": len(docs),
        "documents_errored": len(errors),
        "coverage_by_dir": {d: sum(1 for x in docs if x["category"] == d) for d in sample["dirs"]},
        "totals": {
            "facts_a": sum(d["facts_a"] for d in docs),
            "facts_b": sum(d["facts_b"] for d in docs),
            "identical": sum(d["identical"] for d in docs),
            "rewordings": sum(d["rewordings"] for d in docs),
            "differences": differences,
            "settled": settled,
            "settled_total": settled_total,
            "settled_by_kind": {kind: {label: sum(d["settled_by_kind"][kind][label] for d in docs)
                                       for label in LABELS} for kind in PAIR_KINDS},
            "settlements": sum(len(d["settlements"]) for d in docs),
            "unsettled": unsettled,
            "defects": defects,
        },
        "disagreement_rate_overall": (round(differences / sum(d["compared"] for d in docs), 4)
                                      if sum(d["compared"] for d in docs) else None),
        # The greedy control, aggregated the same way. `None` when the arm did not run
        # (`temperature_0_arm` below says which of the two kinds of None this is), and
        # a real number when it did: `disagreement_rate_overall` minus this is the part
        # of the headline rate that decoding at 0.3 contributed rather than the corpus.
        "disagreement_rate_overall_t0": (
            round(sum(d["differences_t0"] or 0 for d in docs)
                  / sum(d["compared_t0"] or 0 for d in docs), 4)
            if any(d["compared_t0"] for d in docs) else None),
        "temperature_0_arm": any(d["compared_t0"] is not None for d in docs),
        "temperature_arms": {"production": DEFAULT_TEMPERATURE,
                             "control": GREEDY_TEMPERATURE,
                             "control_requested": any(d["compared_t0"] is not None
                                                      for d in docs)},
        "defect_yield_overall": round(defects / settled_total, 4) if settled_total else None,
        # The numerator is `DEFECT_LABELS`, rendered — not restated — so the string a
        # later reader weighs the number against and the sum that produced it are the
        # same list by construction. This matters most where the labels moved: settling
        # rewordings two-sided (#2918) is the first time a run could produce a
        # `both_supported` settlement at all, and a definition still describing a world
        # in which that label never happened would be a published false statement.
        "defect_yield_definition": (
            "(" + " + ".join(DEFECT_LABELS) + ") / settled, where `settled` counts every "
            "label settled from either kind of pair (a one-reading `difference` or a "
            "two-reading `rewording`): a single-sided settlement is a reading the document "
            "supports that the other pass did not produce, `neither_supported` is a reading "
            "neither is supported by. `both_supported` — the document backing BOTH wordings "
            "of one claim, which only a two-reading rewording pair can answer — is not "
            "counted as a defect, because it is the settlement finding nothing invented. "
            "`unsettled` is never folded into either side of the ratio."),
        "disagreement_rate_is_a_lower_bound": (
            "Both passes are the same weights, so this understates the true rate; the "
            "cross-family arm (pass B on djev :8010) is the check for a near-zero result."),
        "cost": {
            "per_document_mean_tokens": (round(statistics.mean(d["cost"]["total_tokens"] for d in docs))
                                         if docs else 0),
            "per_document_mean_seconds": (round(statistics.mean(d["cost"]["seconds"] for d in docs), 3)
                                          if docs else 0.0),
            "total_tokens": sum(d["cost"]["total_tokens"] for d in docs),
            "total_seconds": round(sum(d["cost"]["seconds"] for d in docs), 3),
            # The calls, and what the reconciler alone cost — the two numbers that
            # let a later reader tell whether widening this would stay affordable.
            "total_calls": sum(d["cost"]["calls"] for d in docs),
            "settlement_tokens": sum(d["cost"]["settlement"]["total_tokens"] for d in docs),
            "settlement_seconds": round(sum(d["cost"]["settlement"]["seconds"] for d in docs), 3),
            # The control arm's own price, kept separate because it is the line that
            # moves when `--temperature-0` is passed and the line a widening decision
            # has to weigh before opting in to it.
            "temperature_0_calls": sum(d["cost"]["t0"]["calls"] for d in docs),
            "temperature_0_tokens": sum(d["cost"]["t0"]["total_tokens"] for d in docs),
            "wall_clock_seconds": round(time.monotonic() - started, 3),
            "source": ("taken off each model response's `usage` by this module and timed here: "
                       "the extractor's `_call_llm` drops the field, and the usage store is "
                       "session-keyed harness telemetry with no extraction rows. Every call the "
                       "run made is metered exactly once, the settlement reads included, so this "
                       "is comparable to a figure for a whole reconciliation."),
            "gallon_reference": "7.5M tokens for a year of scanned statements (2 agents + reconciler)",
        },
        "documents": docs,
        "errors": errors,
    }


def render_markdown(summary: dict) -> str:
    s = summary["sample"]
    lines = [
        "# Extraction disagreement audit",
        "",
        f"- seed `{s['seed']}`, requested {s['requested']}, drawn {summary['documents_measured']} "
        f"(+{summary['documents_errored']} errored) from {', '.join(s['dirs'])}",
        f"- disagreement rate: **{summary['disagreement_rate_overall']}** "
        f"(lower bound — same weights both passes)",
        f"- defect yield: **{summary['defect_yield_overall']}** — {summary['defect_yield_definition']}",
        f"- settled {summary['totals']['settled_total']}: "
        + ", ".join(f"{k} {v}" for k, v in summary["totals"]["settled"].items()),
        f"- unsettled: {summary['totals']['unsettled']} (never counted as agreement)",
        "- settled by kind: " + "; ".join(
            f"{kind}: " + "/".join(str(summary['totals']['settled_by_kind'][kind][lb])
                                   for lb in LABELS)
            for kind in PAIR_KINDS)
        + " (a/b/both/neither — only a rewording pair can answer both_supported)",
        f"- rewordings: {summary['totals']['rewordings']} "
        f"(canonical text matched, wording differed — settled two-sided since #2918)",
        f"- cost per document: mean {summary['cost']['per_document_mean_tokens']} tokens / "
        f"{summary['cost']['per_document_mean_seconds']} s, over "
        f"{summary['cost']['total_calls']} calls; run total "
        f"{summary['cost']['total_tokens']} tokens / {summary['cost']['total_seconds']} s, of "
        f"which the reconciler alone is {summary['cost']['settlement_tokens']} tokens / "
        f"{summary['cost']['settlement_seconds']} s; "
        f"{summary['cost']['source']}",
        "",
        # `rate@0.3` is the production pair and `rate@0.0` the greedy control, named by
        # the temperature rather than by a word, so the two columns cannot be read as
        # two samples of the same arm. An arm that did not run is an em dash, never a
        # 0.0 that would read as the document agreeing with itself perfectly.
        "| document | dir | facts A/B | identical | reword | diffs | rate@0.3 | rate@0.0 "
        "| settled (a/b/both/neither) | unsettled | defects | tokens | s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in summary["documents"]:
        st = d["settled"]
        rate_t0 = d.get("disagreement_rate_t0")
        lines.append(
            f"| `{d['path'][:60]}` | {d['category']} | {d['facts_a']}/{d['facts_b']} "
            f"| {d['identical']} | {d['rewordings']} | {d['differences']} "
            f"| {d['disagreement_rate']} | {'—' if rate_t0 is None else rate_t0} "
            f"| {d['settled_total']} ({st['pass_a_supported']}/{st['pass_b_supported']}/"
            f"{st['both_supported']}/{st['neither_supported']}) | {len(d['unsettled'])} "
            f"| {d['defects']} | {d['cost']['total_tokens']} | {d['cost']['seconds']} |")
    for e in summary["errors"]:
        lines.append("| **" + e["path"] + "** | " + e.get("category", "")
                     + " | errored: " + e["error"] + " |" + " |" * 10)
    return "\n".join(lines) + "\n"


def write_artifact(out_dir: Path, ts: str, summary: dict) -> Path:
    run_dir = out_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "extraction-disagreement.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "extraction-disagreement.md").write_text(render_markdown(summary), encoding="utf-8")
    return run_dir


def _endpoint_up(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3.0) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="run dir is written under here; nothing else is")
    ap.add_argument("--vault-root", default=str(Path.home() / "obsidian"))
    ap.add_argument("--sample-size", type=int, default=40)
    # A re-run has to name the config as well as the seed and the vault: the draw is
    # a function of all three, so pinning the seed alone redraws a different corpus
    # the day `sources.paths` moves.
    ap.add_argument("--config", default=str(PIPELINE_CONFIG_PATH),
                    help="the pipeline_config.yaml whose sources.paths is the corpus")
    ap.add_argument("--fresh-only", action="store_true",
                    help="apply the job's incremental 24h mtime window to the corpus")
    ap.add_argument("--seed", default="extraction-disagreement-2026-10-13")
    ap.add_argument("--port", type=int, default=8096, help="extraction model port (primary)")
    ap.add_argument("--settle-port", type=int, default=None,
                    help="port for the reconciling read; defaults to --port. The cross-family "
                         "arm runs pass B / settlement on djev :8010 — :8096 + :8010 is the only "
                         "pair this box can run (eval/cross_engine_disagreement.py:9-19).")
    ap.add_argument("--dry-run", action="store_true",
                    help="draw the sample and write the artifact without sending a request")
    ap.add_argument("--temperature-0", action="store_true",
                    help=f"add a THIRD AND FOURTH pass of the same prompts at "
                         f"{GREEDY_TEMPERATURE} and report each document's disagreement rate "
                         f"at both temperatures, so the headline rate at "
                         f"{DEFAULT_TEMPERATURE} can be separated from decoding variance. "
                         f"Opt-in because it doubles the extraction cost of the run; the "
                         f"two production passes keep sending production's payload either way.")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vault = Path(args.vault_root).expanduser()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    started = time.monotonic()
    sample = sample_documents(vault, args.sample_size, args.seed,
                              config_path=Path(args.config).expanduser(),
                              fresh_only=bool(args.fresh_only))
    if not sample["paths_abs"]:
        print(f"nothing to sample under the allow-list in {vault}; refusing to write an empty audit",
              file=sys.stderr)
        return 2
    fe = _load_extractor()
    if not args.dry_run and not _endpoint_up(args.port):
        # Refuse before writing: a run dir from an unreachable engine would read
        # as a zero-disagreement audit if anything downstream ever globbed it.
        print(f"primary engine on port {args.port} is not answering /health; refusing to run "
              f"(pass B on a dead port would report disagreement 0.0)", file=sys.stderr)
        return 2

    docs: list[dict] = []
    errors: list[dict] = []
    for p in sample["paths_abs"]:
        path = Path(p)
        if args.dry_run:
            docs.append({"path": str(path.relative_to(vault)),
                         "vault_root": str(vault.resolve()),
                         "category": path.relative_to(vault).parts[0], "chars_total": 0,
                         "chars_covered": 0, "prompt_sha256": None, "chunk_offsets": [],
                         "facts_a": 0, "facts_b": 0, "identical": 0, "rewordings": 0,
                         "differences": 0, "compared": 0, "disagreement_rate": None,
                         "differences_t0": None, "identical_t0": None,
                         "rewordings_t0": None, "compared_t0": None,
                         "disagreement_rate_t0": None,
                         "settled": {label: 0 for label in LABELS}, "settled_total": 0,
                         "settled_by_kind": {kind: {label: 0 for label in LABELS}
                                             for kind in PAIR_KINDS},
                         "settlements": [],
                         "unsettled": [], "defects": 0, "defect_yield": None,
                         "supporting_spans": [],
                         # Shaped like a measured row, zero-filled: a dry run sent
                         # no call, so every arm's meter reads zero rather than the
                         # row lacking the keys a reader (or `summarize`) indexes.
                         "cost": {**new_cost(), "a": new_cost(), "b": new_cost(),
                                  "settlement": new_cost(), "t0": new_cost()},
                         "pass_errors": [], "dry_run": True})
            continue
        try:
            docs.append(audit_document(fe, path, vault, args.port, args.settle_port,
                                       temperature_0=bool(args.temperature_0)))
        except PairNotComparable as exc:
            errors.append({"path": str(path.relative_to(vault)), "error": f"pair not comparable: {exc}",
                           "category": path.relative_to(vault).parts[0]})
        except Exception as exc:  # noqa: BLE001 — one document must not lose the run
            errors.append({"path": str(path.relative_to(vault)), "error": str(exc),
                           "category": path.relative_to(vault).parts[0]})
    summary = summarize(sample, docs, errors, started)
    summary["ts"] = ts
    summary["port"] = args.port
    summary["settle_port"] = args.settle_port or args.port
    summary["dry_run"] = bool(args.dry_run)
    run_dir = write_artifact(out, ts, summary)
    print(json.dumps({"run_dir": str(run_dir),
                      "documents": summary["documents_measured"],
                      "errored": summary["documents_errored"],
                      "disagreement_rate_overall": summary["disagreement_rate_overall"],
                      "defect_yield_overall": summary["defect_yield_overall"],
                      "settled": summary["totals"]["settled"],
                      "unsettled": summary["totals"]["unsettled"],
                      "rewordings": summary["totals"]["rewordings"],
                      "total_tokens": summary["cost"]["total_tokens"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
