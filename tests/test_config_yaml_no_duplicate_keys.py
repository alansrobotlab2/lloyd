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
