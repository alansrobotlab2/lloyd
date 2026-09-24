"""The KV gate's prose enumerations match the `LONG_LIVED` flags (#1341).

`workers/pool.py::long_lived_sources` reads a module attribute; three places
restate the resulting list by hand — the `kv_gate` comment in `config.yaml`,
`architecture/workers.md` § The KV budget gate, and `architecture/vllm.md`
§6.3. `arch-review` declared `LONG_LIVED = True` when it shipped and two of
the three kept naming the original three sources, so the sentence a person
reads to learn why a source is being held under-reported exactly the source
running unattended at the time. Nothing derives the prose from the flag, so
this test does the next best thing: it fails the moment they disagree, in
either direction.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from workers import pool

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "workers" / "sources"

# Each site is (file, a regex that captures the sentence naming the sources).
# The capture is bounded at the em dash that closes the enumeration, so a
# name mentioned elsewhere on the page cannot satisfy or trip the check.
SITES = {
    "config.yaml": re.compile(
        r"A source that declares LONG_LIVED —\s*#\s*(?P<names>.*?):", re.S),
    "architecture/workers.md": re.compile(
        r"A source that declares `LONG_LIVED = True` —\s*(?P<names>.*?)—", re.S),
    "architecture/vllm.md": re.compile(
        r"a source declaring\s*`LONG_LIVED = True` —\s*(?P<names>.*?)—", re.S),
}


def _registry() -> dict[str, object]:
    """Every source module by its NAME, imported the way the pool sees it."""
    registry = {}
    for path in sorted(SOURCES.glob("*.py")):
        if path.name.startswith("_"):
            continue
        mod = importlib.import_module(f"workers.sources.{path.stem}")
        name = getattr(mod, "NAME", None)
        assert name, f"{path.name} defines no NAME"
        registry[name] = mod
    return registry


def _named(site: str) -> set[str]:
    text = (ROOT / site).read_text()
    match = SITES[site].search(text)
    assert match, f"{site}: the KV gate sentence is gone or reworded; update SITES"
    # Names arrive as `autocode`, `arch-review`, "deep-research:" etc.; a
    # source name is lowercase words joined by hyphens.
    return set(re.findall(r"[a-z][a-z0-9-]*[a-z0-9]", match.group("names")))


def test_the_flag_gates_more_than_one_source():
    registry = _registry()
    assert len(pool.long_lived_sources(registry)) >= 2, (
        "the pool's own reading of the flag; if this shrinks to one the "
        "enumeration checks below are meaningless")


@pytest.mark.parametrize("site", sorted(SITES))
def test_every_long_lived_source_is_named_and_nothing_else_is(site):
    gated = set(pool.long_lived_sources(_registry()))
    named = _named(site)
    assert gated <= named, (
        f"{site} omits gated sources {sorted(gated - named)}: a source flipped "
        "LONG_LIVED without the sentence being updated")
    assert named <= gated, (
        f"{site} names {sorted(named - gated)}, which are not LONG_LIVED; "
        "the sentence claims a hold the pool does not apply")
