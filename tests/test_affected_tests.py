"""Affected-test selection in shadow (`scripts/automod/affected_tests.py`).

The one property that matters most: the wrapper returns the tests rung's own
result untouched, whatever the shadow does. Everything else is the selection's
arithmetic on a small fake tree.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.automod import affected_tests as AT

ROOT = Path(__file__).resolve().parent.parent


def _tree(tmp_path: Path) -> Path:
    files = {
        "app/__init__.py": "",
        "app/core.py": "X = 1\n",
        "app/mid.py": "from app import core\n",
        "app/leaf.py": "from .mid import *\n",
        "scripts/__init__.py": "",
        "scripts/tool.py": "import json\n",
        "tests/test_core.py": "from app.core import X\n",
        "tests/test_mid.py": "import app.mid\n",
        "tests/test_leaf.py": "from app import leaf\n",
        "tests/test_rail.py": "DOC = 'architecture/thing.md'\nMOD = 'scripts.tool'\n",
        "tests/test_other.py": "import os\n",
        "tests/_helper.py": "",
    }
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


def test_a_changed_module_selects_its_importers_transitively_and_directly(tmp_path):
    root = _tree(tmp_path)
    sel = AT.select(root, ["app/core.py"])
    assert sel["full_run"] is False and sel["test_files"] == 5
    assert sel["selected"] == ["tests/test_core.py", "tests/test_leaf.py", "tests/test_mid.py"]
    assert sel["direct"] == ["tests/test_core.py"], "only the file that imports it itself"


def test_relative_imports_resolve(tmp_path):
    root = _tree(tmp_path)
    assert "tests/test_leaf.py" in AT.select(root, ["app/mid.py"])["selected"]


def test_a_path_or_dotted_name_mentioned_as_text_selects_the_rail(tmp_path):
    root = _tree(tmp_path)
    assert AT.select(root, ["architecture/thing.md"])["selected"] == ["tests/test_rail.py"]
    assert AT.select(root, ["scripts/tool.py"])["direct"] == ["tests/test_rail.py"]


def test_a_changed_test_file_selects_itself(tmp_path):
    root = _tree(tmp_path)
    assert AT.select(root, ["tests/test_other.py"])["selected"] == ["tests/test_other.py"]


@pytest.mark.parametrize("path", ["config.yaml", "tests/conftest.py", "tests/_helper.py",
                                  "pytest.ini", "web/src/api.ts", "eval/fitness/manifest.yaml",
                                  "requirements-dev.txt", "agent-services/guardian/guardian.py"])
def test_what_the_import_graph_cannot_see_is_a_full_run(tmp_path, path):
    sel = AT.select(_tree(tmp_path), ["app/core.py", path])
    assert sel["full_run"] is True and path in sel["reason"] and sel["selected"] == []


def test_only_the_blamed_nodes_can_be_missed():
    data = {"failed_node_ids": ["tests/a.py::t1", "tests/b.py::t2", "tests/c.py::t3"],
            "pre_existing_failures": ["tests/b.py::t2"], "flaky_node_ids": ["tests/c.py::t3"]}
    assert AT.blamed_nodes(data) == ["tests/a.py::t1"]


def test_shadow_names_a_blamed_node_outside_each_selection(tmp_path):
    root = _tree(tmp_path)
    data = {"failed_node_ids": ["tests/test_mid.py::t", "tests/test_other.py::t"]}
    row = AT.shadow(root, ["app/core.py"], data)
    assert row["blamed"] == 2 and row["selected_files"] == 3 and row["direct_files"] == 1
    assert row["missed"] == ["tests/test_other.py::t"]
    assert row["missed_direct"] == ["tests/test_mid.py::t", "tests/test_other.py::t"]
    full = AT.shadow(root, ["config.yaml"], data)
    assert full["full_run"] is True and full["missed"] == [] == full["missed_direct"]


class _Gate:
    def __init__(self, worktree, result, raise_in_rung=False):
        self.worktree, self.round_id = worktree, "SM_x"
        self.report = SimpleNamespace(changed_paths=["app/core.py"])
        self._result, self._raise = result, raise_in_rung

    def rung_tests(self, only=None):
        if self._raise:
            raise RuntimeError("the rung's own error")
        return self._result


def _attached(monkeypatch, events):
    class G(_Gate):
        pass
    monkeypatch.setattr(AT, "configured_mode", lambda: "shadow")
    from scripts.automod import state as S
    monkeypatch.setattr(S, "append_event", events.append)
    AT.attach(G)
    AT.attach(G)  # idempotent
    return G


def test_the_wrapper_returns_the_rungs_result_and_records_one_row(tmp_path, monkeypatch):
    events: list[dict] = []
    G = _attached(monkeypatch, events)
    data = {"failed_node_ids": ["tests/test_other.py::t"]}
    result = (False, "red", data)
    assert G(_tree(tmp_path), result).rung_tests() is result
    assert len(events) == 1 and events[0]["event"] == AT.EVENT and events[0]["rung_ok"] is False
    assert events[0]["missed"] == ["tests/test_other.py::t"]
    assert data["selection_shadow"]["missed"] == ["tests/test_other.py::t"]


def test_a_shadow_that_breaks_costs_nothing(tmp_path, monkeypatch, capsys):
    events: list[dict] = []
    G = _attached(monkeypatch, events)
    monkeypatch.setattr(AT, "shadow", lambda *a, **k: 1 / 0)
    result = (True, "green", {})
    assert G(_tree(tmp_path), result).rung_tests() is result
    assert events == [] and "test selection shadow" in capsys.readouterr().out


def test_the_rungs_own_error_still_propagates(tmp_path, monkeypatch):
    G = _attached(monkeypatch, [])
    with pytest.raises(RuntimeError, match="the rung's own error"):
        G(_tree(tmp_path), None, raise_in_rung=True).rung_tests()


def test_a_selected_run_off_mode_or_a_skip_records_nothing(tmp_path, monkeypatch):
    events: list[dict] = []
    G = _attached(monkeypatch, events)
    G(_tree(tmp_path), (True, "", {})).rung_tests(only=["tests/test_core.py"])
    G(_tree(tmp_path), (True, "", {"skipped": True})).rung_tests()
    monkeypatch.setattr(AT, "configured_mode", lambda: "off")
    G(_tree(tmp_path), (True, "", {})).rung_tests()
    assert events == []


def test_the_real_gate_is_wrapped_and_the_switch_ships_shadow_and_is_fenced():
    from scripts.automod import gate, spec
    assert getattr(gate.Gate.rung_tests, "_selection_shadow", False) is True
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert AT.mode(cfg["automod"]["gate"]["test_selection"]) == "shadow"
    assert "automod.gate.test_selection" in spec.CONFIG_DENIED_KEYS
    assert AT.mode("off") == "off" and AT.mode(False) == "off" and AT.mode("selct") == "shadow"


def test_report_counts_rungs_shares_and_every_miss():
    rows = [{"event": AT.EVENT, "round_id": "a", "full_run": True, "missed": [], "missed_direct": []},
            {"event": AT.EVENT, "round_id": "b", "full_run": False, "test_files": 10,
             "selected_files": 7, "direct_files": 2, "blamed": 1,
             "missed": [], "missed_direct": ["tests/x.py::t"]},
            {"event": "gate"}]
    r = AT.report(rows)
    assert r["rungs"] == 2 and r["selecting"] == 1 and r["selecting_with_blamed_red"] == 1
    assert r["transitive"] == {"mean_selected_share": 0.7, "misses": []}
    assert r["direct"] == {"mean_selected_share": 0.2, "misses": [("b", "tests/x.py::t")]}


def test_this_module_is_not_collected_as_a_test_file():
    """`pytest.ini` collects under `scripts/`; a `test_*.py` name there would be."""
    assert not AT.is_test_file("scripts/automod/affected_tests.py")
    assert AT.is_test_file("tests/test_affected_tests.py")


def test_the_fence_names_exactly_the_one_key():
    from scripts.automod import spec
    assert spec.GATE_SELECTION_DENIED_KEYS == ("automod.gate.test_selection",)
