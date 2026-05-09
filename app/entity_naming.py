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

from app.paths import VAULT_FACTS_ALIASES as _ALIASES_PATH

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
