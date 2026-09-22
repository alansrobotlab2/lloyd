"""The recall's first stage: the keyword leg ORs its terms, and qmd's paths arrive decoded.

`agent_mcp/vault.py` (RECALL_LEX_MODE) carries the measurement.
"""
from __future__ import annotations

import json

import pytest

from agent_mcp import vault
from scripts.automod import evalpin


@pytest.fixture
def wire(monkeypatch):
    sent: list[dict] = []
    monkeypatch.setattr(vault, "_qmd_post", lambda payload: sent.append(dict(payload)) or [])
    monkeypatch.setattr(vault, "_djev_shadow_rerank", lambda *a, **k: None)
    return sent


def _recall():
    vault._vault_recall({"query": "which autonomy tasks maintain the knowledge graph?", "limit": 20,
                         "grep_code": False, "include_facts": False, "expand_graph": False})


def _doc_leg(sent):
    return [b for b in sent if len(b["searches"]) == 2 and b.get("fusion") == "global"][0]


def test_the_doc_leg_ors_its_keyword_terms_on_the_cross_encoder_path(wire):
    _recall()
    assert _doc_leg(wire)["lexMode"] == "or"


def test_the_doc_leg_ors_its_keyword_terms_when_djev_ranks(wire, monkeypatch):
    from app import djev
    monkeypatch.setattr(vault, "RECALL_RERANKER", "djev")
    monkeypatch.setattr(djev, "enabled", lambda: True)
    monkeypatch.setattr(djev, "rank", lambda *a, **k: None)   # falls back; both legs are checked
    rows = [{"file": f"qmd://knowledge/d{i}.md", "title": f"d{i}", "snippet": "s", "score": 1 / (i + 1)} for i in range(5)]
    monkeypatch.setattr(vault, "_qmd_post", lambda payload: wire.append(dict(payload)) or list(rows))
    _recall()
    legs = [b for b in wire if len(b["searches"]) == 2 and b.get("fusion") == "global"]
    assert [b["lexMode"] for b in legs] == ["or", "or"]


def test_and_is_the_kill_switch_and_sends_no_key(wire, monkeypatch):
    monkeypatch.setattr(vault, "RECALL_LEX_MODE", "and")
    _recall()
    assert "lexMode" not in _doc_leg(wire)


def test_other_callers_keep_and(wire):
    vault._qmd_daemon_search("anything at all", 5, list(vault.VAULT_SEGMENTS))
    vault._qmd_daemon_search("anything at all", 5, ["backlog"])
    assert all("lexMode" not in b for b in wire)


def test_the_regression_pin_warms_the_or_request():
    assert evalpin.production_payload("a question")["lexMode"] == "or"


@pytest.mark.parametrize("raw, decoded", [
    ("qmd://people/Ali%20Behrouz.md", "qmd://people/Ali Behrouz.md"),
    ("qmd://knowledge/llm-serving/model-sleep-wake-latency-96b%2B.md", "qmd://knowledge/llm-serving/model-sleep-wake-latency-96b+.md"),
    ("qmd://work/a%23b.md", "qmd://work/a#b.md"),
    ("qmd://knowledge/plain.md", "qmd://knowledge/plain.md"),
    ("", ""),
])
def test_qmd_paths_arrive_decoded(raw, decoded):
    assert vault.qmd_file(raw) == decoded


def test_post_decodes_every_result(monkeypatch):
    class _Resp:
        def __init__(self, body): self.body = body
        def read(self): return self.body
        def __enter__(self): return self
        def __exit__(self, *a): return False
    body = json.dumps({"results": [{"file": "qmd://people/Ali%20Behrouz.md", "title": "Ali", "snippet": "", "score": 1}]}).encode()
    monkeypatch.setattr(vault.urllib.request, "urlopen", lambda req, timeout=None: _Resp(body))
    rows = vault._qmd_post({"searches": [{"type": "lex", "query": "ali"}], "rerank": False})
    assert rows[0]["file"] == "qmd://people/Ali Behrouz.md"


def test_the_grep_leg_admits_the_same_files_whatever_order_rg_prints_them(monkeypatch):
    """rg prints files as its threads finish. Before the sort, the same query on
    the same tree admitted a different 8 files from run to run (7 of 81 gold
    queries, 2026-09-21), which moved djev's input and the regression check
    read the difference as a change."""
    import random
    import subprocess
    import agent_mcp.vault as V

    files = [f"{V.LLOYD_CODE_PREFIX}app/f{i:02d}.py" for i in range(20)]
    orders = []

    def fake_run(argv, **kw):
        shuffled = files[:]
        random.Random(len(orders)).shuffle(shuffled)
        orders.append(shuffled)
        return subprocess.CompletedProcess(argv, 0, stdout="\n".join(shuffled) + "\n", stderr="")

    monkeypatch.setattr(V.subprocess, "run", fake_run)
    first = [d["file"] for d in V._grep_lloyd_code("where is vault_recall defined", limit=8)]
    second = [d["file"] for d in V._grep_lloyd_code("where is vault_recall defined", limit=8)]
    assert orders[0] != orders[1], "the fake must print two different orders"
    assert first == second == [f"app/f{i:02d}.py" for i in range(8)]
