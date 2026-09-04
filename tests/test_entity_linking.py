"""app.entity_naming.known_entities_in_text — extraction-time entity hints."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import entity_naming as en  # noqa: E402


@pytest.fixture
def aliases(tmp_path, monkeypatch):
    p = tmp_path / "entity-aliases.json"
    names = ["Intel Pipeline", "Intel", "vLLM", "segment", "active", "stack-updates",
             "Alfie", "GR00T N1", "Node.js", "GPU", "qmd-sdk", "Claude Agent SDK"]
    p.write_text(json.dumps({n: n for n in names} | {"intel pipeline": "Intel Pipeline"}))
    monkeypatch.setattr(en, "_ALIASES_PATH", p)
    en._KNOWN_INDEX.update(mtime=0.0, index={}, max_n=1)
    return p


def test_multi_token_names_match_case_insensitively(aliases):
    hits = en.known_entities_in_text("the INTEL pipeline scans arxiv and the claude agent sdk streams")
    assert hits == ["Intel Pipeline", "Claude Agent SDK"]


def test_single_token_names_need_proper_casing_or_acronym(aliases):
    assert en.known_entities_in_text("intel corp shipped a gpu") == []          # lowercase surface ≠ Intel/GPU
    assert en.known_entities_in_text("Intel shipped a new GPU today") == ["Intel", "GPU"]
    assert en.known_entities_in_text("Alfie picked up the block") == ["Alfie"]


def test_lowercase_slug_and_frontmatter_words_are_never_hints(aliases):
    text = "---\nsegment: knowledge\nstatus: active\ntags: [stack-updates]\n---\nqmd-sdk wraps QMD"
    assert en.known_entities_in_text(text) == []


def test_longest_match_wins_and_order_is_first_occurrence(aliases):
    hits = en.known_entities_in_text("GR00T N1 runs on vLLM; Intel Pipeline uses Node.js")
    assert hits == ["GR00T N1", "vLLM", "Intel Pipeline", "Node.js"]


def test_limit_and_empty(aliases):
    assert en.known_entities_in_text("") == []
    assert en.known_entities_in_text("Intel vLLM Alfie", limit=2) == ["Intel", "vLLM"]


def test_index_refreshes_when_alias_file_changes(aliases):
    assert en.known_entities_in_text("Blackwell rocks") == []
    data = json.loads(aliases.read_text()); data["Blackwell"] = "Blackwell"
    aliases.write_text(json.dumps(data))
    import os, time
    os.utime(aliases, (time.time() + 5, time.time() + 5))
    assert en.known_entities_in_text("Blackwell rocks") == ["Blackwell"]
