#!/usr/bin/env python3
"""
Lloyd MCP Server: Skills — search and read skill definitions.

Tools: skills_search, skills_read

Skills live in directories under ~/obsidian/skills and ~/lloyd/skills.
Each skill is a folder containing a SKILL.md with YAML frontmatter
(name, description, category, tags) followed by the skill body.
"""

import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

import yaml
from mcp.types import Tool

# Skill-side query stopwords moved to agent_mcp._shared (#340 P3 cleanup).
# Local alias preserves the existing internal name `_QUERY_STOPWORDS` so
# the rest of skills.py doesn't change.
from agent_mcp._shared import _SKILLS_QUERY_STOPWORDS as _QUERY_STOPWORDS, text_result

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

#: The roots as the code ships them. `SKILLS_DIRS` below is this list unioned with
#: whatever `config.yaml skills.directories` names, so that the config key steers
#: every surface instead of only the two that happened to read it (#1294).
_BUILTIN_SKILL_DIRS = [
    Path.home() / "obsidian" / "skills",
    Path(__file__).parent.parent / "skills",
]


def _config_skill_dirs() -> list[Path]:
    """`config.yaml skills.directories`, `~`-expanded, best effort.

    `GET /api/skills` and the Mission Control tab used to read this key while
    retrieval and the prompt index read the constant above — two answers to "where
    do skills live", which is one reason the two human-facing numbers could read
    194 and 189 about the same vault. Folding it in here leaves one list. It is
    defensive because this module is also imported by standalone scripts that have
    no backend config in their environment.
    """
    try:
        from app.config import CONFIG
        raw = (CONFIG.get("skills") or {}).get("directories") or []
    except Exception:
        return []
    out: list[Path] = []
    for item in raw:
        text = str(item).strip()
        if text:
            out.append(Path(text.replace("~", str(Path.home()))))
    return out


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        if str(path) not in seen:
            seen.add(str(path))
            out.append(path)
    return out


#: One definition of where live skills may sit, highest priority first: the roots
#: the code ships, then any the config adds. Every surface walks it through
#: `iter_active_skills` (#1294), and a test that needs a controlled tree replaces
#: THIS list — the one knob all five surfaces share. Two names in two roots is
#: decided by order, which is why this one list, not two, is the answer.
SKILLS_DIRS = _dedupe_paths([*_BUILTIN_SKILL_DIRS, *_config_skill_dirs()])

# Frontmatter `status:` values that quarantine a skill — it stays on disk
# (and in git history) but is excluded from retrieval entirely. This is the
# lever for pulling a misbehaving skill out of circulation without deleting
# it (e.g. auto-generated bash-runbook skills that the model echoes verbatim
# instead of acting on). The norm is `status: active`; anything in this set
# is skipped. Comparison is case-insensitive on the stripped value.
_QUARANTINE_STATUSES = {"inactive", "archived", "disabled", "retired", "quarantined"}

# ── Helpers ───────────────────────────────────────────────────────────────────

# (source, front-matter text) pairs already warned about, so a skill walked on
# every search logs its broken block once per edit, not once per call.
_WARNED_UNPARSEABLE: set[tuple[str, int]] = set()


def _parse_frontmatter(content: str, source: str = "") -> tuple[dict, str]:
    """Return (frontmatter_dict, body_text). Body is everything after the closing ---.

    An unparseable block still yields `{}` — the skill loads, with no
    description and no tags — but it is no longer silent (#561): a skill in
    that state has quietly left retrieval while looking installed, so the
    parse error is logged once, naming `source` (the skill) when given.
    """
    if not content.startswith("---"):
        return {}, content
    end = content.find("\n---", 3)
    if end == -1:
        return {}, content
    fm_text = content[3:end]
    body = content[end + 4:].strip()
    try:
        fm = yaml.safe_load(fm_text) or {}
    except Exception as exc:
        key = (source, hash(fm_text))
        if key not in _WARNED_UNPARSEABLE:
            _WARNED_UNPARSEABLE.add(key)
            logger.warning(
                "skills: front matter of %s does not parse (%s); it loads with "
                "no description or tags and drops out of retrieval",
                source or "a skill", str(exc).splitlines()[0] if str(exc) else type(exc).__name__)
        fm = {}
    return fm, body


def _load_skill(skill_dir: Path) -> Optional[dict]:
    """Load and parse a single skill directory. Returns None if no SKILL.md."""
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        return None
    try:
        content = skill_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    fm, body = _parse_frontmatter(content, source=f"skill {skill_dir.name!r} ({skill_file})")
    status = str(fm.get("status", "") or "").strip().lower()
    if status in _QUARANTINE_STATUSES:
        # Quarantined — present on disk but pulled from retrieval.
        return None
    return {
        "name": skill_dir.name,
        "description": fm.get("description", ""),
        "category": fm.get("category", ""),
        "tags": fm.get("tags") or [],
        "body": body,
        "raw": content,
        "path": skill_file,
    }


# ── The one definition of "a live skill" (#1294) ──────────────────────────────

class ActiveSkill(NamedTuple):
    """One live skill, as `iter_active_skills` found it.

    A NamedTuple carrying the parsed front matter, because the five surfaces that
    need this set need different slices of it: the prompt index wants `name`,
    `GET /api/skills` wants `description`/`category` out of `frontmatter`, the
    Mission Control tab only counts, and `skill_lint` wants the front matter and
    the directory it sits in. `directory` rather than a path is deliberate — the
    lint also resolves `scripts/` cited from inside the skill.

    `frontmatter` is `{}` when the block will not parse. That is the whole point of
    carrying it here: the route used to raise on an unexpected shape
    (`metadata:\n  openclaw: null`) and its bare `except Exception: continue`
    deleted seven live skills from the Skills page. An unparseable block is a
    finding for `skill_lint`, not a reason to hide that the skill exists.
    """

    name: str
    directory: Path
    frontmatter: dict

    @property
    def skill_file(self) -> Path:
        return self.directory / "SKILL.md"


def is_quarantined_skill_file(skill_file: Path) -> bool:
    """Whether the skill at `skill_file` has pulled itself out of circulation.

    One spelling of the rule (#1294). `prompt_builder` used to re-scan the
    front matter line-by-line for a `status:` key and compare it against a copy of
    the set — same answer today, two answers whenever the front-matter formats
    differ, and the two implementations were held in step only by an equality
    assertion in one test.

    A missing or unreadable file answers `False`: that is damage, not retirement,
    and `skill_load_defect` is the function that says which.
    """
    try:
        content = skill_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    status = str(_parse_frontmatter(content)[0].get("status", "") or "").strip().lower()
    return status in _QUARANTINE_STATUSES


def skill_roots(overlay: Optional[Path] = None,
                base: Optional[list[Path]] = None) -> list[Path]:
    """The directories to walk for live skills, highest priority first.

    `overlay` — autoresearch's variant directory — goes first as
    `<overlay>/skills`, which is how a promoted prompt variant can replace a skill
    for one build without touching the vault. `base` replaces `SKILLS_DIRS` for the
    one caller that keeps its own canonical pair (`prompt_builder`); it exists so
    that caller can hand the walker its roots without re-implementing this
    composition a second time.
    """
    dirs: list[Path] = []
    if overlay:
        dirs.append(Path(overlay) / "skills")
    dirs.extend(SKILLS_DIRS if base is None else base)
    return dirs


def iter_active_skills(overlay: Optional[Path] = None,
                       roots: Optional[list[Path]] = None) -> "Iterator[ActiveSkill]":
    """Yield every live skill exactly once — the definition every surface shares.

    Three rules, decided here and nowhere else (#1294; five code paths each had
    their own answer, and only two of them honoured the second):

      1. a dot-prefixed directory is the archive, not a skill — `skills/.archived/`
         holds retired skills in per-skill subdirectories;
      2. a front-matter `status:` in `_QUARANTINE_STATUSES` is retired in place:
         on disk, in git, absent from every surface;
      3. a name present in two roots belongs to the first root — the copy
         `_skills_read` would actually serve.

    A directory is yielded as soon as it has a `SKILL.md`, even if its front
    matter is unreadable: presence is what the index, the Skills page and the tab
    count are reporting, and a defect in the file is `skill_lint`'s to find.

    `roots` overrides the walked list for a caller that already has one — the
    prompt builder passes its own canonical pair. A caller should not pass a list
    that disagrees with `skill_roots()` on the live tree; that is the drift this
    function exists to end.
    """
    seen: set[str] = set()
    for skills_dir in skill_roots(overlay, roots):
        if not skills_dir.is_dir():
            continue
        for entry in sorted(skills_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith(".") or entry.name in seen:
                continue
            skill_file = entry / "SKILL.md"
            if not skill_file.is_file():
                continue
            try:
                content = skill_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                content = ""
            fm, _body = _parse_frontmatter(
                content, source=f"skill {entry.name!r} ({skill_file})")
            status = str(fm.get("status", "") or "").strip().lower()
            if status in _QUARANTINE_STATUSES:
                continue
            seen.add(entry.name)
            yield ActiveSkill(name=entry.name, directory=entry, frontmatter=fm)


def skill_load_defect(skill_dir: Path) -> Optional[str]:
    """Why `skill_dir` is a *damaged* skill, or None when it may sit on disk.

    `_load_skill` answers one question — "put this in retrieval?" — with one
    sentinel, `None`, for three different facts: there is no `SKILL.md`, the
    file cannot be read, and the skill is deliberately quarantined by its
    front-matter `status`. For discovery all three mean the same thing. For a
    validator that asks "does this skill still load?" they do not, and reading
    the third as damage is how `automod_vault_land` ended up unable to retire a
    skill at all (#777): neither moving it into `skills/.archived/` nor
    flipping its `status` could clear the land check, so every "archive this
    skill" item had to be done by hand outside the only sanctioned route.

    The quarantine is therefore stated, not inferred from the sentinel. A `None`
    that is neither missing, unreadable, nor quarantined is still reported — the
    day `_load_skill` grows a fourth reason to abstain, this names it instead of
    quietly passing, which is the difference between loosening a check and
    emptying one.
    """
    if _load_skill(skill_dir) is not None:
        return None
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        return "no SKILL.md"
    try:
        content = skill_file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"SKILL.md unreadable: {exc}"
    status = str(_parse_frontmatter(content)[0].get("status", "") or "").strip().lower()
    if status in _QUARANTINE_STATUSES:
        return None
    return "does not load"


def _iter_skills():
    """Yield loaded skill dicts from all skill directories.

    The walk is `iter_active_skills()`' (#1294); this turns its records back into
    the dicts the scorer and `skills_read` want. `_load_skill` re-reads the file
    because `skills_search` returns the body and `skills_read` the raw text, which
    the walker deliberately does not carry, and it abstains on a record whose body
    turned out to be unreadable.
    """
    for active in iter_active_skills():
        skill = _load_skill(active.directory)
        if skill:
            yield skill


# Suffixes we collapse for matching. "systems" → "system", "services" → "service",
# "checks" → "check", "tasks" → "task". Conservative: only strip if the result is
# still a real-looking word (≥ 3 chars) and the original isn't already an
# exception (`news`, `bus`, `gas`, `os`, `is`, `us`, `ss`-endings).
_STEM_SKIP = {"news", "bus", "gas", "lens", "kids", "his", "its", "yes", "this", "thus", "plus", "loss", "boss"}


def _stem(token: str) -> str:
    """Cheap morphology collapse for skill-matching only.

    Folds simple English plurals so a query token ("systems") matches a skill
    token ("system"). Not a real stemmer — this is a single-suffix rule tuned
    to fix the prefetch failure mode where "full systems check" picked
    `claude-sdk-check` over `system-health-check` because "systems" ≠ "system".

    Rules (applied in order, first match wins):
      - len ≤ 3: return as-is (too short to safely strip)
      - in _STEM_SKIP: return as-is
      - ends in 'ies' (len > 4): 'ies' → 'y'  (queries → query)
      - ends in 'sses': strip 'es'  (passes → pass)
      - ends in 'ss', 'us', 'is', 'os': return as-is
      - ends in 's': strip 's'  (systems → system)
      - else: return as-is
    """
    if len(token) <= 3 or token in _STEM_SKIP:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith(("ss", "us", "is", "os")):
        return token
    if token.endswith("s"):
        return token[:-1]
    return token


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens, plural-collapsed for stable matching.

    Both query side and skill side go through stemming so "systems" in a
    query matches "system" in a skill name (and vice versa).
    """
    raw = re.findall(r"\b\w+\b", text.lower())
    return {_stem(t) for t in raw}


def _query_tokens(text: str) -> set[str]:
    """Tokenize a *query* string: lowercase words, stopwords dropped, len ≥ 2.

    Use this for the query side of skill matching. Both sides are stemmed
    via `_tokenize`; the asymmetry vs. skill-side is the stopword filter —
    a query like "lets dig into 311" should not fire skills just because
    "lets", "dig", and "into" appear in arbitrary skill bodies.
    """
    return {t for t in _tokenize(text) if t not in _QUERY_STOPWORDS and len(t) > 1}


def _excerpt(body: str, query_tokens: set[str], max_len: int = 200) -> str:
    """Find the first paragraph containing a query token and return a trimmed excerpt."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
    for para in paragraphs:
        if _tokenize(para) & query_tokens:
            return para[:max_len] + ("…" if len(para) > max_len else "")
    # Fallback: first paragraph
    if paragraphs:
        p = paragraphs[0]
        return p[:max_len] + ("…" if len(p) > max_len else "")
    return ""


# Body-hit cap: with 223 skills, each ~5-15KB of body text, a 5-token query
# will coincidentally match body words in half the skills. Cap the body
# contribution so a skill can't qualify on body-noise alone.
_BODY_HITS_CAP = 4


def _skill_token_sets(skill: dict) -> tuple[set[str], set[str], set[str], set[str]]:
    """Return (name, description, tag, body) token sets, memoized on the dict.

    Tokenizing + stemming a 5-15KB skill body costs ~0.3ms; across ~280
    skills that was ~83ms of GIL-held CPU on *every* prefetch turn, and it
    starved the other prefetch workers (facts/backlog) of the interpreter.
    The prefetch skill cache keeps the same dicts alive across turns, so
    memoizing here turns scoring into ~280 set intersections (~1ms).

    The memo lives under the private `_tok` key. Callers that serialize a
    skill dict must pick fields explicitly (they all do) — sets aren't
    JSON-encodable.
    """
    tok = skill.get("_tok")
    if tok is None:
        tags = skill.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        tok = (
            _tokenize((skill.get("name") or "").replace("-", " ")),
            _tokenize(skill.get("description") or ""),
            _tokenize(" ".join(str(t) for t in tags)),
            _tokenize(skill.get("body") or ""),
        )
        skill["_tok"] = tok
    return tok


# ── Pseudo-queries per skill (#1490, Skill2Query) ─────────────────────────────
#
# `scripts/skill_pseudo_queries.py` asks the primary, offline, for a handful of
# requests a user would type when a skill's procedure is the job, generated from
# the skill's own name/description/tags/body and nothing else. With
# `skills.pseudo_queries.enabled` the scorer counts query tokens found in those
# requests — and in none of the skill's name/description/tags, so nothing is
# counted twice — as a metadata hit at `weight`. Off by default, and off is
# byte-for-byte the old scorer: `eval/measurements/skill-pseudo-queries-2026-09-25.md`.

PSEUDO_QUERY_MODES = ("union", "max")


def skill_text_hash(skill: dict) -> str:
    """What a skill's pseudo-queries were generated from; a change invalidates them."""
    tags = skill.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    blob = "\x1f".join([skill.get("name") or "", str(skill.get("description") or ""),
                        " ".join(map(str, tags)), skill.get("body") or ""])
    return hashlib.sha1(blob.encode("utf-8", "replace")).hexdigest()


def pseudo_query_settings() -> dict | None:
    """`skills.pseudo_queries` as `{weight, mode, path}`, or None when off.

    Every failure reads as off, which is today's scorer.
    """
    try:
        from app.config import CONFIG
        cfg = (CONFIG.get("skills") or {}).get("pseudo_queries") or {}
    except Exception:  # noqa: BLE001
        return None
    if cfg.get("enabled") is not True:
        return None
    try:
        weight = float(cfg.get("weight", 3.0))
    except (TypeError, ValueError):
        weight = 3.0
    mode = cfg.get("mode", "union")
    if mode not in PSEUDO_QUERY_MODES:
        mode = "union"
    path = cfg.get("path") or ""
    if not path:
        try:
            from app.paths import SKILL_PSEUDO_QUERIES_PATH
            path = SKILL_PSEUDO_QUERIES_PATH
        except Exception:  # noqa: BLE001
            return None
    return {"weight": weight, "mode": mode, "path": Path(str(path)).expanduser()}


_pq_file_cache: dict = {"key": None, "skills": {}}
_pq_lock = threading.Lock()


def _load_pseudo_queries(path: Path) -> tuple[tuple, dict]:
    """`(cache key, {name: {"hash", "queries"}})`, re-read when the file's mtime moves.

    A missing or unreadable file is `{}` — no skill gains anything, which is
    the scorer with the flag off.
    """
    try:
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return (str(path), None, None), {}
    with _pq_lock:
        if _pq_file_cache["key"] == key:
            return key, _pq_file_cache["skills"]
        try:
            skills = json.loads(path.read_text()).get("skills") or {}
        except (OSError, ValueError, AttributeError):
            skills = {}
        _pq_file_cache["key"] = key
        _pq_file_cache["skills"] = skills
        return key, skills


def pseudo_query_index() -> dict | None:
    """The settings plus the loaded table — one config read and one stat.

    `_search_skills` takes this once per turn and hands it to every
    `_score_skill` call; the default path of `_score_skill` takes it per call.
    """
    settings = pseudo_query_settings()
    if settings is None:
        return None
    key, table = _load_pseudo_queries(settings["path"])
    return {**settings, "key": key, "table": table}


def _pseudo_query_token_sets(skill: dict, index: dict) -> tuple[frozenset, ...]:
    """Per pseudo-query token sets for this skill, minus its own metadata tokens.

    Memoized on the dict under `_pq`, keyed by the cache file's identity, so a
    regenerated file is picked up and an unchanged one costs a tuple compare.
    Queries generated from a different version of the skill (hash mismatch) are
    ignored rather than trusted.
    """
    key = index["key"]
    memo = skill.get("_pq")
    if memo is not None and memo[0] == key:
        return memo[1]
    entry = index["table"].get(skill.get("name") or "") or {}
    sets: tuple[frozenset, ...] = ()
    if entry.get("queries") and entry.get("hash") == skill_text_hash(skill):
        name_t, desc_t, tag_t, _ = _skill_token_sets(skill)
        meta = name_t | desc_t | tag_t
        sets = tuple(s for s in (frozenset(_query_tokens(str(q)) - meta)
                                 for q in entry["queries"] if isinstance(q, str)) if s)
    skill["_pq"] = (key, sets)
    return sets


def _pseudo_query_hits(skill: dict, query_tokens: set[str], index: dict) -> int:
    sets = _pseudo_query_token_sets(skill, index)
    if not sets:
        return 0
    if index["mode"] == "max":
        return max(len(query_tokens & s) for s in sets)
    return len(query_tokens & frozenset().union(*sets))


_UNSET = object()


def _score_skill(skill: dict, query_tokens: set[str],
                 require_metadata_hit: bool = True,
                 pseudo_queries=_UNSET) -> float:
    """Score a skill against query tokens. Higher = more relevant.

    If `require_metadata_hit` is True (default), a skill with zero
    name/desc/tag overlap scores 0.0 regardless of body overlap. This
    prevents the "12× powerpoint on graph-classifier work" failure mode
    where generic English stopwords in queries bag-of-words-match arbitrary
    skill bodies.

    Body hits are capped (see `_BODY_HITS_CAP`) and weighted to be a
    tiebreaker, not a qualifier.
    """
    name_tokens, desc_tokens, tag_tokens, body_tokens = _skill_token_sets(skill)

    name_hits = len(query_tokens & name_tokens)
    desc_hits = len(query_tokens & desc_tokens)
    tag_hits = len(query_tokens & tag_tokens)
    body_hits = min(len(query_tokens & body_tokens), _BODY_HITS_CAP)

    pq = pseudo_query_index() if pseudo_queries is _UNSET else pseudo_queries
    pq_hits = _pseudo_query_hits(skill, query_tokens, pq) if pq else 0

    if require_metadata_hit and (name_hits + desc_hits + tag_hits + pq_hits) == 0:
        return 0.0

    score = name_hits * 3.0 + desc_hits * 2.0 + tag_hits * 1.5 + body_hits * 0.3
    if pq_hits:
        score += pq_hits * pq["weight"]
    return score


# ── Tool handlers ─────────────────────────────────────────────────────────────

def _skills_search(params: dict) -> str:
    query = params.get("query", "").strip()
    if not query:
        return json.dumps({"error": "query is required", "results": []})
    max_results = int(params.get("max_results", 10))
    query_tokens = _query_tokens(query)

    scored = []
    for skill in _iter_skills():
        score = _score_skill(skill, query_tokens, require_metadata_hit=True)
        if score > 0:
            scored.append((score, skill))

    scored.sort(key=lambda x: -x[0])

    results = []
    for score, skill in scored[:max_results]:
        results.append({
            "name": skill["name"],
            "description": skill["description"],
            "category": skill["category"],
            "tags": skill["tags"],
            "excerpt": _excerpt(skill["body"], query_tokens),
            "score": round(score, 2),
        })

    return json.dumps({"query": query, "results": results, "total": len(scored)})


def _skills_read(params: dict) -> str:
    name = params.get("name", "").strip()
    if not name:
        return json.dumps({"error": "name is required"})
    for skills_dir in SKILLS_DIRS:
        skill_dir = skills_dir / name
        skill = _load_skill(skill_dir)
        if skill:
            return json.dumps({"name": skill["name"], "content": skill["raw"]})
    return json.dumps({"error": f"Skill not found: {name}"})


# ── MCP registration ──────────────────────────────────────────────────────────

async def list_tools():
    return [
        Tool(
            name="skills_search",
            description=(
                "Search available skills by keyword. Searches skill names, descriptions, "
                "tags, and body content. Returns ranked results with name, description, "
                "and a body excerpt. Use this to discover which skill to apply to a task."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords to search for"},
                    "max_results": {"type": "integer", "description": "Max results to return (default 10)"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="skills_read",
            description=(
                "Use to load one skill's full instructions once you know its name; to find one use skills_search first. "
                "Read the full SKILL.md content for a named skill. Use after skills_search "
                "to get the complete instructions for a specific skill."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Skill directory name (e.g. 'research-agent')"},
                },
                "required": ["name"],
            },
        ),
    ]


async def call_tool(name: str, arguments: dict):
    handlers = {
        "skills_search": _skills_search,
        "skills_read": _skills_read,
    }
    handler = handlers.get(name)
    if handler:
        return text_result(handler(arguments))
    return text_result(json.dumps({"error": f"Unknown tool: {name}"}))
