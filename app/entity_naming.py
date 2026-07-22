"""Shared entity-name normalization for Lloyd's memory graph.

This is the single source of truth for resolving entity names before any
read or write under the facts tree. Every code path that touches the
facts tree (routers, extractors, classifiers, profile generators) must
normalize through this module.

Key facts about the data layout:

- `entity-aliases.json` is a flat `{surface_form: canonical_name}` map.
  Keys are case-sensitive surface forms; multiple casings/punctuations
  can map to the same canonical.
- `entity-registry.json` is metadata only — it does NOT carry any
  canonical_mapping field, despite what older docs claimed.
- One dir per canonical entity under `<facts_root>/<Name>/`.

The facts root lives at `app.paths.VAULT_FACTS_ROOT` (currently
`~/lloyd/_pipeline/vault-derived/facts/`).
"""

from __future__ import annotations

import json
import re

from app.paths import VAULT_FACTS_ALIASES as _ALIASES_PATH

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

_ALIAS_CACHE: dict = {"mtime": 0.0, "map": {}}


def _load_alias_map() -> dict[str, str]:
    """Return case-insensitive surface→canonical map. Lazily refreshed on mtime.

    On lowercase-key collision (multiple surface forms differing only in case
    or punctuation that would map to different canonicals — see the historic
    `Task #21` / `Task #215` bug), prefer the self-identity entry
    (surface == canon). This avoids a dubious cross-map clobbering the
    authoritative self-identity.
    """
    try:
        mtime = _ALIASES_PATH.stat().st_mtime
    except FileNotFoundError:
        return {}
    if mtime == _ALIAS_CACHE["mtime"] and _ALIAS_CACHE["map"]:
        return _ALIAS_CACHE["map"]
    try:
        raw = json.loads(_ALIASES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _ALIAS_CACHE["map"]
    lowered: dict[str, str] = {}
    for surface, canon in raw.items():
        key = surface.lower()
        if key in lowered:
            if surface == canon:
                lowered[key] = canon
            # else keep existing — don't let a cross-map overwrite self-identity
            continue
        lowered[key] = canon
    _ALIAS_CACHE["mtime"] = mtime
    _ALIAS_CACHE["map"] = lowered
    return lowered


def normalize(name: str) -> str:
    """Resolve an entity name to its canonical form via the alias map.

    Returns the input unchanged if no alias matches. Safe to call on empty
    strings. This is the function every reader/writer should use before
    touching the facts tree.
    """
    if not name:
        return name
    return _load_alias_map().get(name.lower(), name)


def register_canonical(name: str) -> str:
    """Ensure `name` is present in `entity-aliases.json` as a self-identity entry.

    If `name` already resolves (case-insensitive) to a canonical, returns
    that canonical. Otherwise writes `{name: name}` into `entity-aliases.json`
    and returns `name`.

    This is the right thing to call at entity-dir creation time in writers:
    it guarantees that next time someone looks the name up (even in a
    different case) they'll hit the same canonical. Idempotent and safe to
    call frequently; only writes when something actually changes.
    """
    if not name:
        return name
    alias_map = _load_alias_map()
    existing = alias_map.get(name.lower())
    if existing is not None:
        return existing

    # Read raw file (we need to preserve the case-sensitive structure on disk).
    try:
        raw = json.loads(_ALIASES_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raw = {}
    except Exception:
        # Corrupt file — do not clobber. Just return the name.
        return name

    if name in raw:
        return raw[name]
    raw[name] = name
    try:
        _ALIASES_PATH.write_text(
            json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8"
        )
    except Exception:
        return name
    # Invalidate cache so next call picks up the new entry.
    _ALIAS_CACHE["mtime"] = 0.0
    _ALIAS_CACHE["map"] = {}
    return name


def normalize_and_register(name: str) -> str:
    """Combined helper: resolve if known, else register as self-identity.

    This is the most useful call for writers: pass whatever surface form
    you have, get back the canonical, and guarantee the alias map knows
    about it.
    """
    if not name:
        return name
    resolved = normalize(name)
    if resolved == name:
        # Either nothing matched or it self-resolved. Make sure we're
        # registered for next time.
        return register_canonical(name)
    return resolved
