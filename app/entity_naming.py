"""Shared entity-name normalization for Lloyd's memory graph.

This is the single source of truth for resolving entity names before any
read or write under the facts tree. Every code path that touches the
facts tree (routers, extractors, classifiers, profile generators) must
normalize through this module.

Key facts about the data layout:

- Aliases and the entity registry live in `app.kg_store` (SQLite). Until the
  2026-09 migration they were `entity-aliases.json`, a flat map that six
  programs rewrote whole with no lock; `register_canonical` read it, added a
  key and wrote the entire file back, which is how concurrent writers lost
  each other's entries.
- One dir per canonical entity under `<facts_root>/<Name>/`. The store's
  entity table mirrors that set.

The facts root lives at `app.paths.VAULT_FACTS_ROOT` (currently
`~/lloyd/_pipeline/vault-derived/facts/`).
"""

from __future__ import annotations

import re

from app.kg_store import StoreUnavailable, alias_kind as _alias_kind, store as _store

# ── Junk-entity guard ────────────────────────────────────────────────────────
# The LLM extractor occasionally emits a *filename* or a code/description
# fragment as the "entity" when a document discusses source code or plumbing
# (e.g. `server.py`, `judge.py (line 181)`, `SKILL.md description field`,
# `{name}.md`). Those leak into the facts tree as bogus entity dirs. This
# predicate flags them so writers can skip and a purge can find them.
#
# Design bias is PRECISION over recall: a real entity is either a plain name
# (person / project / tool / concept) or a single-token knowledge-doc slug that
# ends in `.md`. We only flag high-confidence junk, so a Capitalized dotted
# tech name (`Node.js`, `Config.yaml`) is intentionally NOT flagged rather than
# risk dropping a legitimate entity.

# Source-code / config / data extensions that never name a real entity.
_CODE_EXTS = (
    "py", "pyc", "pyi", "js", "jsx", "ts", "tsx", "mjs", "cjs", "sh", "bash",
    "zsh", "yml", "yaml", "toml", "ini", "cfg", "conf", "json", "jsonl",
    "ndjson", "bak", "log", "lock", "env", "sql", "css", "scss", "less",
    "html", "htm", "xml", "csv", "tsv", "kit", "launch", "ipynb", "rs", "go",
    "java", "cpp", "hpp",
)
_EXT_ALT = "|".join(_CODE_EXTS)
# A code-ext filename that is the WHOLE name, with an all-lowercase, filename-
# shaped basename (`server.py`, `test-vllm.yml`, `azure-gpt-4o-mini.json`).
# Requiring lowercase avoids flagging Capitalized tech names like `Node.js`.
_CODE_FILE_RE = re.compile(rf"^[a-z0-9._/\\-]+\.(?:{_EXT_ALT})$")
# A multi-word name that ends in a code ext → glued filename
# (`main agent's tools.json`, `SDK subprocess_cli.py`, `gr00t eval_so100.py`).
_ENDS_EXT_RE = re.compile(rf"\.(?:{_EXT_ALT})$", re.IGNORECASE)
# A code-ext token embedded mid-string, followed by more text → fragment
# (`judge.py aggregation`, `messages.py573`, `judge.py (line 181)`,
# `Idler (autonomy_db.py)`). No end-of-string alternative, so a clean
# Capitalized `Node.js` is left alone.
_CODE_FRAG_RE = re.compile(rf"\.(?:{_EXT_ALT})[\s_)\d]", re.IGNORECASE)
# A function/method call: an identifier char immediately before `(`
# (`query()`, `handleToolExecutionStart()`, `models.load_lora_adapter()`).
# Requiring NO space before `(` spares abbreviations like `Chain-of-Thought (CoT)`.
# The call content must look code-like (empty, or containing a lowercase letter,
# `=`, `,`, or a quote) so math notation `SE(3)`, `TD(λ)`, `FP8(E4M3)` is spared.
_CODE_CALL_RE = re.compile(r"\w\(([^)]*)\)")
_CODE_CALL_CONTENT_RE = re.compile(r"[a-z=,'\"]")
# `.md` as a real extension (not `.mdx`/`.mdc`).
_MD_RE = re.compile(r"\.md(?![a-z])", re.IGNORECASE)


def looks_like_junk_entity(name: str) -> bool:
    """True if `name` is a leaked filename / code fragment / template, not a
    real entity.

    High precision by design (see module note). Legit entities pass: plain
    names, and single-token doc slugs ending in `.md` with no interior space.
    """
    if not name:
        return True
    s = name.strip()
    if not s or s == ".md":
        return True
    # Template / placeholder markers.
    if "{" in s or "}" in s or "<" in s or "YYYY-MM-DD" in s or "YYYY-MM" in s:
        return True
    # Function/method-call fragment: `query()`, `models.load_lora_adapter()`.
    m = _CODE_CALL_RE.search(s)
    if m and (m.group(1) == "" or _CODE_CALL_CONTENT_RE.search(m.group(1))):
        return True
    # `.md` handling: a real doc-slug entity is a single token ending in `.md`.
    low = s.lower()
    if _MD_RE.search(s):
        if " " in s:            # multiword + .md → glued fragment / malformed
            return True
        if not low.endswith(".md"):   # `.md` mid-string → fragment
            return True
        return False            # clean single-token doc slug → keep
    # Whole name is a lowercase code/config filename.
    if _CODE_FILE_RE.match(s):
        return True
    # Multi-word name glued to a trailing code-ext filename.
    if " " in s and _ENDS_EXT_RE.search(s):
        return True
    # Code-ext token embedded in a longer string → fragment.
    if _CODE_FRAG_RE.search(s):
        return True
    return False


def is_valid_entity_name(name: str) -> bool:
    """Convenience inverse of :func:`looks_like_junk_entity`."""
    return not looks_like_junk_entity(name)

def _load_alias_map() -> dict[str, str]:
    """Case-insensitive surface→canonical map, entities included.

    Memoised in the store on `PRAGMA data_version`, so it refreshes when any
    process commits and costs one pragma otherwise.
    """
    try:
        return _store().aliases.all_lower()
    except StoreUnavailable:
        return {}


def normalize(name: str) -> str:
    """Resolve an entity name to its canonical form via the alias map.

    Returns the input unchanged if no alias matches. Safe to call on empty
    strings. This is the function every reader/writer should use before
    touching the facts tree.
    """
    if not name:
        return name
    try:
        return _store().resolve(name) or name
    except StoreUnavailable:
        return name


def register_canonical(name: str) -> str:
    """Ensure `name` is known to the store as an entity of its own.

    If `name` already resolves (case-insensitive) to a canonical, returns
    that canonical. Otherwise registers it and returns it.

    This is the right thing to call at entity-dir creation time in writers:
    it guarantees that next time someone looks the name up (even in a
    different case) they'll hit the same canonical. Idempotent, and now a
    single INSERT rather than a whole-file rewrite.
    """
    if not name:
        return name
    try:
        st = _store()
    except StoreUnavailable:
        return name
    existing = st.resolve(name)
    if existing is not None:
        return existing
    st.entities.register(name)
    return name


def normalize_and_register(name: str) -> str:
    """Combined helper: resolve if known, else register as a new entity.

    This is the most useful call for writers: pass whatever surface form
    you have, get back the canonical, and guarantee the store knows about it.
    """
    if not name:
        return name
    resolved = normalize(name)
    if resolved == name:
        # Either nothing matched or it self-resolved. Make sure we're
        # registered for next time.
        return register_canonical(name)
    return resolved


def set_alias(surface: str, canonical: str, *, kind: str | None = None,
              origin: str = "manual", report_path: str | None = None) -> None:
    """Route `surface` to `canonical` in the store."""
    if not surface or not canonical:
        return
    try:
        _store().aliases.set(surface, canonical, kind=kind or _alias_kind(surface, canonical),
                             origin=origin, report_path=report_path)
    except StoreUnavailable:
        pass


# ── Extraction-time entity linking ───────────────────────────────────────────
# The fact extractor never saw a known entity name: every caller passed an
# empty "known facts" context, so the model coined `Intel Pipeline System` while
# `Intel Pipeline` already existed. Measured 2026-09-03: in 303 of 442
# near-duplicate clusters the later variant was created a day or more after an
# existing one (median gap 6 days) — a lookup would have hit 69% of the time.
# This gives the extractor the known names that actually appear in a chunk, at
# a median cost of ~114 tokens.

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_GENERIC_SINGLE = frozenset({
    "test", "agent", "agents", "memory", "system", "state", "config", "update",
    "status", "event", "error", "general", "model", "tools", "tool", "skill",
    "skills", "plan", "plans", "notes", "note", "data", "pipeline", "server",
    "service", "client", "task", "tasks", "build", "setup", "review", "research",
    "project", "debug", "audit", "queue", "cache", "user", "users", "session",
    "knowledge", "quality", "policy", "developer", "render", "intelligence",
    "retrieval", "worker", "workers", "graph", "vault", "fact", "facts",
})


def _known_index() -> tuple[dict, int]:
    """{(token, ...): canonical} over every canonical the store knows.

    Cached in the store on `PRAGMA data_version` — the JSON version rebuilt
    this from a 939 KB file whenever its mtime moved.
    """
    def build():
        st = _store()
        index: dict[tuple, str] = {}
        max_n = 1
        canonicals = set(st.entities.all()) | set(st.aliases.all().values())
        for canon in sorted(canonicals):
            if not canon or looks_like_junk_entity(canon):
                continue
            toks = tuple(t.lower() for t in _TOKEN_RE.findall(canon))
            if not toks or len(toks) > 8:
                continue
            # Proper-noun shape only: a canonical with no capital letter and no digit
            # is a slug or a common word (`segment`, `active`, `stack-updates` are
            # all registered "entities" — frontmatter leaked into the alias map).
            # Hints must be high-precision; a missed hint costs nothing, a wrong one
            # steers the extractor into filing facts under a bogus entity.
            if not any(ch.isupper() or ch.isdigit() for ch in canon):
                continue
            if len(toks) == 1 and (len(toks[0]) < 3 or toks[0] in _GENERIC_SINGLE):
                continue
            # keep the first canonical for a token shape; case/punct siblings are aliases anyway
            index.setdefault(toks, canon)
            max_n = max(max_n, len(toks))
        return index, max_n
    try:
        return _store().cached("known_entity_index", build)
    except StoreUnavailable:
        return {}, 1


def known_entities_in_text(text: str, limit: int = 60) -> list[str]:
    """Canonical entity names that appear verbatim (word-bounded) in `text`.

    Multi-token names match case-insensitively. Single-token names must appear
    with the canonical's own capitalisation (or be an all-caps acronym), so a
    lowercase common word in prose does not summon an entity that merely shares
    its spelling. Ordered by first occurrence, longest match first at a
    position, capped at `limit`.
    """
    if not text:
        return []
    index, max_n = _known_index()
    if not index:
        return []
    words = [(m.group(0), m.start()) for m in _TOKEN_RE.finditer(text)]
    lower = [w.lower() for w, _ in words]
    found: dict[str, int] = {}
    i = 0
    while i < len(lower):
        hit = None
        for n in range(min(max_n, len(lower) - i), 0, -1):
            key = tuple(lower[i:i + n])
            canon = index.get(key)
            if canon is None:
                continue
            if n == 1:
                surface = words[i][0]
                if surface != canon and not (surface.isupper() and len(surface) >= 3):
                    continue
            hit = (canon, n)
            break
        if hit:
            canon, n = hit
            found.setdefault(canon, words[i][1])
            i += n
        else:
            i += 1
    ordered = sorted(found.items(), key=lambda kv: kv[1])
    return [c for c, _ in ordered[:limit]]
