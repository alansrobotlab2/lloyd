"""config.yaml must not repeat a mapping key (#1463).

YAML is last-wins: a second copy of a key silently shadows the first, so an
edit to the copy a reader finds first does nothing. `workers.sources.
youtube-digest.inner_voice` sat there twice until #1463; both said `false`,
so nothing was visibly broken — which is exactly why it needs a test.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class _DupKeyLoader(yaml.SafeLoader):
    """SafeLoader that records every repeated key instead of overwriting."""

    duplicates: list[str]


def _construct_mapping(loader: _DupKeyLoader, node: yaml.MappingNode, deep: bool = False):
    seen: dict[object, int] = {}
    for key_node, _value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            hash(key)
        except TypeError:
            continue
        if key in seen:
            loader.duplicates.append(
                f"{key!r} at line {key_node.start_mark.line + 1} "
                f"(first at line {seen[key]})"
            )
        else:
            seen[key] = key_node.start_mark.line + 1
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_DupKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _duplicates(text: str) -> list[str]:
    loader = _DupKeyLoader(text)
    loader.duplicates = []
    try:
        loader.get_single_data()
    finally:
        loader.dispose()
    return loader.duplicates


def test_detector_catches_a_repeated_key():
    # Fails-first: the detector must see the shape #1463 was.
    text = "a:\n  inner_voice: false\n  x: 1\n  inner_voice: false\n"
    dups = _duplicates(text)
    assert len(dups) == 1 and "inner_voice" in dups[0]


def test_config_yaml_has_no_duplicate_keys():
    text = (ROOT / "config.yaml").read_text()
    assert _duplicates(text) == []


def test_data_trip_hold_ships_only_the_deleter_tasks_key():
    """The #4383 pin added ONE key under `workers:`, and only one inside its block.

    `workers/pool.py` defaults the data-trip gate to ON — `data_trip_hold_status`
    reads `cfg.get("enabled", True)` — so the operator's kill switch is
    deliberately absent from tracked config, the same shape `primary_hold` has
    (also absent, also defaulted on). Shipping `enabled: true` would put a
    one-word edit that disarms a data-loss gate inside the blast radius of any
    future config commit, which is the class of mistake #4359 exists for.

    The value is pinned beside it because the key REPLACES
    `workers.pool.DEFAULT_DATA_TRIP_DELETER_TASKS = (\"79\",)` rather than
    extending it: the shipped pair must be `['79', '98']`, 79 the retention sweep
    that deleted into the 2026-11-19 trip and 98 the `facts-snapshot-*` prune
    ruled a second member of the class.
    """
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    block = (cfg.get("workers") or {}).get("data_trip_hold")
    assert isinstance(block, dict), (
        "workers.data_trip_hold is missing, so the class in force is the code "
        "default and the #4359 ruling that named task 98 is unenforced")
    assert sorted(block) == ["deleter_tasks"], (
        f"workers.data_trip_hold ships {sorted(block)}; only deleter_tasks belongs "
        f"there — an `enabled` key would be a kill switch no test re-arms")
    assert block["deleter_tasks"] == ["79", "98"], (
        f"the shipped class is {block['deleter_tasks']!r}; both ids are strings and "
        f"79 stays, because the key replaces the default outright")
