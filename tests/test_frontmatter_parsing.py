"""Shared resilient frontmatter parser (agent_mcp._shared.parse_frontmatter_text).

Pins the three-layer recovery contract: plain YAML → orphaned-tags repair →
regex field extraction. A record may come back degraded (_yaml_broken) but
never None — the silent-task-drop failure mode (2026-05-28, 34/40 tasks)
must stay dead in every reader of agent-written frontmatter.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_mcp._shared import parse_frontmatter_text


def test_clean_yaml_parses_normally():
    fm = parse_frontmatter_text("id: 38\nname: nightly\ntags: [a, b]\n")
    assert fm == {"id": 38, "name": "nightly", "tags": ["a", "b"]}
    assert "_yaml_broken" not in fm


def test_orphaned_tags_corruption_recovers():
    # The exact corruption shape from project_autonomy_silent_task_drop:
    # inline tags followed by orphaned block-list items.
    fm_text = (
        "id: 38\n"
        "name: nightly reflection\n"
        "tags: [38-foo, autonomy, pipeline]\n"
        "- nightly\n"
        "- reflection\n"
        "status: active\n"
    )
    fm = parse_frontmatter_text(fm_text, log_label="test")
    assert fm["id"] == 38
    assert fm["status"] == "active"
    assert set(fm["tags"]) == {"38-foo", "autonomy", "pipeline", "nightly", "reflection"}
    assert "_yaml_broken" not in fm


def test_unparseable_yaml_falls_back_to_regex():
    # Unquoted colon in a value — classic agent-written YAML breakage that
    # the tags repair can't fix.
    fm_text = (
        "id: 12\n"
        "name: fix: the thing: again\n"
        "status: active\n"
        "priority: high\n"
    )
    fm = parse_frontmatter_text(
        fm_text, fallback_fields=("id", "name", "status", "priority"), log_label="test"
    )
    assert fm["_yaml_broken"] is True
    assert fm["id"] == 12  # scalar re-parse recovers the int
    assert fm["status"] == "active"
    assert fm["priority"] == "high"
    assert "fix" in fm["name"]


def test_non_mapping_frontmatter_degrades_not_none():
    fm = parse_frontmatter_text("- just\n- a\n- list\n", fallback_fields=("id",))
    assert isinstance(fm, dict)
    assert fm["_yaml_broken"] is True


def test_never_returns_none_on_garbage():
    fm = parse_frontmatter_text("{{{{:::not yaml at all\x00", fallback_fields=("id",))
    assert isinstance(fm, dict)
