"""Code graph module — resolution, traversal, staleness, refresh, hygiene.

Everything runs against a synthetic graph.json in tmp_path rather than the
real 10 MB build: the shape is what is being pinned, and a test that needs a
15-second graphify run is a test nobody runs.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from agent_mcp import annotations as A, code_graph as CG, main as M


# ── fixtures ────────────────────────────────────────────────────────────────

def _node(nid, label, src, loc="L1", **kw):
    d = {"id": nid, "label": label, "source_file": src, "source_location": loc,
         "file_type": "code", "_origin": "ast"}
    d.update(kw)
    return d


def _link(u, v, relation, src, loc):
    return {"source": u, "target": v, "relation": relation, "_origin": "ast",
            "confidence": "EXTRACTED", "source_file": src, "source_location": loc}


def _graph_doc(commit="c0ffee", edges_key="links"):
    """a() -> b(); Gate.run() called by uses_gate(); duplicate main()."""
    nodes = [
        _node("mod_a", "a.py", "pkg/a.py"),
        _node("mod_b", "b.py", "pkg/b.py"),
        _node("a_fn", "a()", "pkg/a.py", "L10", _callable=True),
        _node("b_fn", "b()", "pkg/b.py", "L20", _callable=True),
        _node("gate", "Gate", "pkg/b.py", "L40", _callable=True, _callable_class=True),
        _node("gate_run", ".run()", "pkg/b.py", "L45", _callable=True),
        _node("uses_gate", "uses_gate()", "pkg/a.py", "L60", _callable=True),
        _node("main_a", "main()", "pkg/a.py", "L90", _callable=True),
        _node("main_b", "main()", "pkg/b.py", "L91", _callable=True),
        _node("why_a", "why a exists", "pkg/a.py", "L10", file_type="rationale"),
    ]
    links = [
        _link("mod_a", "a_fn", "contains", "pkg/a.py", "L10"),
        _link("mod_a", "uses_gate", "contains", "pkg/a.py", "L60"),
        _link("mod_a", "main_a", "contains", "pkg/a.py", "L90"),
        _link("mod_b", "b_fn", "contains", "pkg/b.py", "L20"),
        _link("mod_b", "gate", "contains", "pkg/b.py", "L40"),
        _link("mod_b", "main_b", "contains", "pkg/b.py", "L91"),
        _link("gate", "gate_run", "method", "pkg/b.py", "L45"),
        _link("a_fn", "b_fn", "calls", "pkg/a.py", "L12"),
        _link("uses_gate", "gate_run", "calls", "pkg/a.py", "L62"),
        _link("main_a", "a_fn", "calls", "pkg/a.py", "L92"),
        _link("why_a", "a_fn", "rationale_for", "pkg/a.py", "L10"),
    ]
    return {"directed": False, "multigraph": False, "graph": {},
            "nodes": nodes, edges_key: links, "built_at_commit": commit}


@pytest.fixture(autouse=True)
def _clean_module_state():
    CG._CACHE.clear()
    CG._LOCKS.clear()
    CG._LAST_REFRESH.clear()
    CG._WORKTREES["at"] = 0.0
    CG._WORKTREES["paths"] = []
    CG._VERSION_CACHE.clear()
    yield
    CG._CACHE.clear()


def _write_graph(root: Path, doc: dict) -> Path:
    out = root / CG.GRAPH_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    gp = out / CG.GRAPH_FILENAME
    gp.write_text(json.dumps(doc))
    return gp


def _git(root: Path, *args: str):
    return subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True, check=False)


@pytest.fixture
def repo(tmp_path):
    """A real git repo with a graph built at HEAD."""
    root = tmp_path / "tree"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("def a():\n    pass\n")
    (root / "pkg" / "b.py").write_text("def b():\n    pass\n")
    _git(root.parent, "init", "-q", str(root))
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    _write_graph(root, _graph_doc(commit=head))
    return root


@pytest.fixture
def plain(tmp_path):
    """A non-git directory with a graph."""
    root = tmp_path / "plain"
    root.mkdir()
    _write_graph(root, _graph_doc())
    return root


# ── registration and contract ───────────────────────────────────────────────

def test_module_is_registered_and_meets_the_contract():
    assert CG in M.MODULES
    M._check_module(CG)


async def test_six_graph_tools_are_advertised():
    names = [t.name for t in await CG.list_tools()]
    assert names == ["graph_explain", "graph_affected", "graph_path",
                     "graph_hubs", "graph_status", "graph_refresh"]


async def test_query_tools_are_read_only_and_refresh_is_only_idempotent():
    for n in ("graph_explain", "graph_affected", "graph_path", "graph_hubs",
              "graph_status"):
        assert n in A.READ_ONLY, f"{n} must be usable in plan mode"
    assert "graph_refresh" not in A.READ_ONLY
    assert "graph_refresh" in A.IDEMPOTENT
    assert not (A.DESTRUCTIVE & {"graph_refresh"})


async def test_no_tool_declares_its_own_caption_field():
    """The harness injects `summary`; a second caption field splits the answer."""
    for t in await CG.list_tools():
        props = set((t.input_schema or {}).get("properties", {}))
        assert "summary" not in props, t.name
        assert "description" not in props, t.name


async def test_unknown_tool_name_is_an_error_not_a_crash():
    res = await CG.call_tool("graph_nope", {})
    assert res.is_error
    assert "Unknown tool" in res.content[0].text


# ── loading ─────────────────────────────────────────────────────────────────

async def test_loader_accepts_the_raw_edges_shape(tmp_path):
    """A half-finished `graphify update --no-cluster` writes `edges`."""
    root = tmp_path / "raw"
    root.mkdir()
    _write_graph(root, _graph_doc(edges_key="edges"))
    entry = await CG.load_graph(root)
    assert entry is not None
    assert entry.n_edges == 11


async def test_missing_graph_loads_as_none(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    assert await CG.load_graph(root) is None


async def test_cache_is_keyed_on_mtime_and_size(plain):
    first = await CG.load_graph(plain)
    again = await CG.load_graph(plain)
    assert again is first, "unchanged graph.json must not be re-parsed"

    doc = _graph_doc()
    doc["nodes"].append(_node("extra", "extra()", "pkg/c.py", "L1", _callable=True))
    gp = _write_graph(plain, doc)
    os.utime(gp, (time.time() + 5, time.time() + 5))
    third = await CG.load_graph(plain)
    assert third is not first
    assert third.n_nodes == first.n_nodes + 1


async def test_cache_is_bounded_by_max_cached_roots(tmp_path, monkeypatch):
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "code_graph", {"max_cached_roots": 2})
    roots = []
    for i in range(3):
        r = tmp_path / f"r{i}"
        r.mkdir()
        _write_graph(r, _graph_doc())
        roots.append(r)
        await CG.load_graph(r)
    assert len(CG._CACHE) == 2
    assert str(roots[0]) not in CG._CACHE


# ── root resolution ─────────────────────────────────────────────────────────

def test_root_defaults_to_lloyd_home():
    from app.paths import LLOYD_HOME
    assert CG.resolve_root("") == Path(os.path.realpath(LLOYD_HOME))
    assert CG.resolve_root(None) == Path(os.path.realpath(LLOYD_HOME))


def test_root_rejects_a_relative_path():
    with pytest.raises(CG.RootError):
        CG.resolve_root("app/harness")


def test_root_rejects_a_nonexistent_absolute_path(tmp_path):
    with pytest.raises(CG.RootError):
        CG.resolve_root(str(tmp_path / "nope"))


def test_root_accepts_a_round_id(monkeypatch, tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    import scripts.automod.worktree as W
    monkeypatch.setattr(W, "worktree_path", lambda rid: wt)
    assert CG.resolve_root("SM_20260908_101010") == Path(os.path.realpath(wt))


def test_round_id_without_a_worktree_is_an_error(monkeypatch, tmp_path):
    import scripts.automod.worktree as W
    monkeypatch.setattr(W, "worktree_path", lambda rid: tmp_path / "gone")
    with pytest.raises(CG.RootError):
        CG.resolve_root("SM_20260908_101010")


# ── symbol resolution ───────────────────────────────────────────────────────

async def test_bare_name_resolves_through_the_parenthesised_label(plain):
    entry = await CG.load_graph(plain)
    assert CG.resolve_symbol(entry, "a").node == "a_fn"
    assert CG.resolve_symbol(entry, "a()").node == "a_fn"


async def test_class_dot_method_resolves_through_containment(plain):
    entry = await CG.load_graph(plain)
    assert CG.resolve_symbol(entry, "Gate.run").node == "gate_run"


async def test_file_argument_disambiguates(plain):
    entry = await CG.load_graph(plain)
    res = CG.resolve_symbol(entry, "main")
    assert res.node is None and sorted(res.candidates) == ["main_a", "main_b"]
    assert CG.resolve_symbol(entry, "main", "pkg/b.py").node == "main_b"


async def test_a_path_resolves_to_the_file_node(plain):
    entry = await CG.load_graph(plain)
    assert CG.resolve_symbol(entry, "pkg/a.py").node == "mod_a"


async def test_rationale_nodes_are_never_a_resolution(plain):
    entry = await CG.load_graph(plain)
    res = CG.resolve_symbol(entry, "why a exists")
    assert res.node is None and res.candidates == []


async def test_no_match_returns_nearest(plain):
    entry = await CG.load_graph(plain)
    res = CG.resolve_symbol(entry, "uses_gat")
    assert res.node is None
    assert any("uses_gate" in n for n in res.nearest)


# ── explain ─────────────────────────────────────────────────────────────────

async def test_explain_uses_the_edge_call_site_not_the_node(plain):
    out = (await CG.call_tool("graph_explain",
                              {"symbol": "b", "root": str(plain)})).content[0].text
    assert "<-- a() [calls] pkg/a.py:L12" in out


async def test_explain_hides_rationale_and_containment_from_the_lists(plain):
    out = (await CG.call_tool("graph_explain",
                              {"symbol": "a", "root": str(plain)})).content[0].text
    assert "rationale_for" not in out
    assert "[contains]" not in out
    assert "contained by: a.py" in out


async def test_explain_reports_members(plain):
    out = (await CG.call_tool("graph_explain",
                              {"symbol": "Gate", "root": str(plain)})).content[0].text
    assert "members (1): .run()" in out


async def test_ambiguous_symbol_is_a_listing_not_an_error(plain):
    res = await CG.call_tool("graph_explain", {"symbol": "main", "root": str(plain)})
    assert res.is_error is False
    text = res.content[0].text
    assert "matches 2 nodes" in text and "id: main_a" in text and "id: main_b" in text


async def test_unknown_symbol_errors_with_nearest(plain):
    res = await CG.call_tool("graph_explain", {"symbol": "uses_gat", "root": str(plain)})
    assert res.is_error
    payload = json.loads(res.content[0].text)
    assert "no node matching" in payload["error"]
    assert payload["nearest"]


async def test_output_is_bounded_by_limit(plain):
    out = (await CG.call_tool("graph_explain",
                              {"symbol": "a", "root": str(plain), "limit": 3})
           ).content[0].text
    assert "more line(s) omitted" in out


# ── affected ────────────────────────────────────────────────────────────────

async def test_affected_groups_by_depth(plain):
    out = (await CG.call_tool("graph_affected",
                              {"symbol": "b", "root": str(plain), "depth": 2})
           ).content[0].text
    assert "depth 1 (1):" in out and "a()" in out
    assert "depth 2 (1):" in out and "main()" in out


async def test_affected_seeds_with_members_so_a_class_reaches_method_callers(plain):
    out = (await CG.call_tool("graph_affected",
                              {"symbol": "Gate", "root": str(plain)})).content[0].text
    assert "member(s) of the symbol" in out
    assert "uses_gate()" in out


async def test_affected_summarises_files(plain):
    out = (await CG.call_tool("graph_affected",
                              {"symbol": "b", "root": str(plain), "depth": 2})
           ).content[0].text
    assert "files (1): pkg/a.py x2" in out


async def test_affected_depth_is_clamped(plain):
    out = (await CG.call_tool("graph_affected",
                              {"symbol": "b", "root": str(plain), "depth": 99})
           ).content[0].text
    assert "depth 4," in out


# ── path ────────────────────────────────────────────────────────────────────

async def test_path_excludes_containment(plain):
    """Without this every symbol is two hops from every other via its file."""
    out = (await CG.call_tool("graph_path",
                              {"a": "a", "b": "Gate.run", "root": str(plain)})
           ).content[0].text
    assert "no path" in out


async def test_path_reports_hops_with_relations(plain):
    out = (await CG.call_tool("graph_path",
                              {"a": "main", "a_file": "pkg/a.py", "b": "b",
                               "root": str(plain)})).content[0].text
    assert "2 hop(s):" in out and "[calls]" in out


async def test_path_supports_undirected(plain):
    res = await CG.call_tool("graph_path",
                             {"a": "b", "b": "main", "b_file": "pkg/a.py",
                              "root": str(plain), "undirected": True})
    assert res.is_error is False
    assert "2 hop(s):" in res.content[0].text


async def test_no_path_is_not_an_error(plain):
    res = await CG.call_tool("graph_path",
                             {"a": "a", "b": "Gate.run", "root": str(plain)})
    assert res.is_error is False


# ── hubs ────────────────────────────────────────────────────────────────────

async def test_hubs_excludes_file_and_rationale_nodes(plain):
    body = (await CG.call_tool("graph_hubs", {"root": str(plain), "top": 20})
            ).content[0].text.split("connected code nodes")[1]
    rows = [ln.strip() for ln in body.splitlines() if ln.strip()]
    assert not [r for r in rows if r.startswith("a.py ") or r.startswith("b.py ")], rows
    assert "why a exists" not in body
    assert any(r.startswith("a() ") for r in rows), rows


async def test_hubs_prefix_scopes_the_ranking(plain):
    out = (await CG.call_tool("graph_hubs",
                              {"root": str(plain), "prefix": "pkg/b.py"})
           ).content[0].text
    assert "uses_gate()" not in out
    assert "b()" in out


# ── staleness ───────────────────────────────────────────────────────────────

async def test_fresh_repo_is_not_stale(repo):
    entry = await CG.load_graph(repo)
    st = await CG.staleness(repo, entry)
    assert st.stale is False, st.reasons


async def test_commit_mismatch_is_stale(repo):
    _write_graph(repo, _graph_doc(commit="deadbeef" * 5))
    CG._CACHE.clear()
    entry = await CG.load_graph(repo)
    st = await CG.staleness(repo, entry)
    assert st.stale and "HEAD moved" in st.reasons[0]
    assert st.dirty_only is False


async def test_newer_uncommitted_python_is_stale_but_a_txt_is_not(repo):
    entry = await CG.load_graph(repo)
    (repo / "pkg" / "c.txt").write_text("scratch")
    st = await CG.staleness(repo, entry)
    assert st.stale is False, st.reasons

    (repo / "pkg" / "c.py").write_text("def c():\n    pass\n")
    st = await CG.staleness(repo, entry)
    assert st.stale and st.dirty_only
    assert "uncommitted edits" in st.reasons[0]


async def test_non_git_root_is_never_stale_once_built(plain):
    entry = await CG.load_graph(plain)
    st = await CG.staleness(plain, entry)
    assert st.is_git is False and st.stale is False


async def test_non_git_root_never_auto_refreshes(plain, monkeypatch):
    calls = []
    async def _never(*a, **kw):
        calls.append(a)
        return {"error": "should not run"}
    monkeypatch.setattr(CG, "refresh", _never)
    entry, st, notes = await CG.ensure_graph(plain)
    assert entry is not None and calls == []


async def test_header_says_a_non_git_root_has_unknown_staleness(plain):
    out = (await CG.call_tool("graph_status", {"root": str(plain)})).content[0].text
    assert json.loads(out)["is_git_repo"] is False


# ── refresh ─────────────────────────────────────────────────────────────────

class _FakeProc:
    def __init__(self, rc=0, out=b"", err=b"", hang=False):
        self.returncode = rc
        self._out, self._err, self._hang = out, err, hang
        self.pid = os.getpid()

    async def communicate(self):
        if self._hang:
            await asyncio.sleep(3600)
        return self._out, self._err

    async def wait(self):
        return self.returncode


def _record_exec(monkeypatch, doc_after=None, root=None, rc=0, hang=False):
    seen = []

    real_exec = asyncio.create_subprocess_exec

    async def fake(*argv, cwd=None, stdout=None, stderr=None, start_new_session=False):
        seen.append({"argv": list(argv), "cwd": cwd,
                     "start_new_session": start_new_session})
        if argv[0] == "git":
            return await real_exec(
                *argv, cwd=cwd, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE)
        if doc_after is not None and root is not None and argv[1] == "update":
            _write_graph(root, doc_after)
        return _FakeProc(rc=rc, hang=hang)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake)
    return seen


async def test_refresh_runs_update_then_cluster_only_in_its_own_session(repo, monkeypatch):
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    seen = _record_exec(monkeypatch, doc_after=_graph_doc(commit=head), root=repo)
    result = await CG.refresh(repo, force=True)
    assert "error" not in result, result
    steps = [s["argv"] for s in seen if s["argv"][0] != "git"]
    assert steps[0][1:] == ["update", str(repo), "--no-cluster"]
    assert steps[1][1:] == ["cluster-only", str(repo), "--no-label", "--no-viz"]
    assert all(s["start_new_session"] for s in seen if s["argv"][0] != "git")


async def test_refresh_timeout_kills_the_process_group(repo, monkeypatch):
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "code_graph", {"refresh_timeout_s": 0.05})
    _record_exec(monkeypatch, hang=True)
    killed = []
    import agent_mcp.builtin_bash as BB
    monkeypatch.setattr(BB, "_kill_proc_tree", lambda p: killed.append(p))
    result = await CG.refresh(repo, force=True)
    assert "timed out" in result["error"]
    assert killed


async def test_missing_binary_is_reported_not_raised(repo, monkeypatch):
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "code_graph", {"graphify_bin": "/nonexistent/graphify"})
    result = await CG.refresh(repo, force=True)
    assert "not found" in result["error"]


async def test_missing_binary_still_answers_from_an_existing_graph(repo, monkeypatch):
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "code_graph", {"graphify_bin": "/nonexistent/graphify"})
    _write_graph(repo, _graph_doc(commit="deadbeef" * 5))
    CG._CACHE.clear()
    res = await CG.call_tool("graph_explain", {"symbol": "a", "root": str(repo)})
    assert res.is_error is False
    assert "cannot refresh" in res.content[0].text
    assert "<-- main() [calls]" in res.content[0].text


async def test_concurrent_callers_build_once(repo, monkeypatch):
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    seen = _record_exec(monkeypatch, doc_after=_graph_doc(commit=head), root=repo)
    _write_graph(repo, _graph_doc(commit="deadbeef" * 5))
    CG._CACHE.clear()
    await asyncio.gather(*[CG.refresh(repo) for _ in range(3)])
    updates = [s for s in seen if s["argv"][0] != "git" and s["argv"][1] == "update"]
    assert len(updates) == 1, f"built {len(updates)} times"


async def test_refresh_false_answers_stale_and_says_so(repo):
    _write_graph(repo, _graph_doc(commit="deadbeef" * 5))
    CG._CACHE.clear()
    out = (await CG.call_tool("graph_explain",
                              {"symbol": "a", "root": str(repo), "refresh": False})
           ).content[0].text
    assert "STALE" in out and "refresh=false" in out


async def test_dirty_rule_is_debounced(repo, monkeypatch):
    await CG.load_graph(repo)
    (repo / "pkg" / "c.py").write_text("def c():\n    pass\n")
    CG._LAST_REFRESH[str(repo)] = time.time()
    calls = []
    async def _never(*a, **kw):
        calls.append(a)
        return {}
    monkeypatch.setattr(CG, "refresh", _never)
    _entry, _st, notes = await CG.ensure_graph(repo)
    assert calls == []
    assert any("debounced" in n for n in notes), notes


async def test_status_never_builds(repo, monkeypatch):
    _write_graph(repo, _graph_doc(commit="deadbeef" * 5))
    CG._CACHE.clear()
    calls = []
    async def _never(*a, **kw):
        calls.append(a)
        return {}
    monkeypatch.setattr(CG, "refresh", _never)
    out = json.loads((await CG.call_tool("graph_status", {"root": str(repo)})
                      ).content[0].text)
    assert calls == []
    assert out["stale"] is True and out["exists"] is True


async def test_status_on_a_root_with_no_graph(tmp_path):
    root = tmp_path / "bare"
    root.mkdir()
    out = json.loads((await CG.call_tool("graph_status", {"root": str(root)})
                      ).content[0].text)
    assert out["exists"] is False and out["nodes"] == 0


# ── implementation constraints ──────────────────────────────────────────────

def test_module_never_blocks_the_loop_with_subprocess_run():
    src = Path(CG.__file__).read_text()
    assert "subprocess.run(" not in src
    assert "subprocess.check_output(" not in src


def test_graphify_out_is_ignored_by_gitignore_not_info_exclude():
    """A build must not dirty the tree — the automod gate refuses a dirty one."""
    from app.paths import LLOYD_HOME
    r = subprocess.run(
        ["git", "-C", str(LLOYD_HOME), "check-ignore", "-v",
         "graphify-out/GRAPH_REPORT.md"],
        capture_output=True, text=True, check=False)
    assert r.returncode == 0, "graphify-out/ is not ignored"
    source = r.stdout.split(":")[0]
    assert source.endswith(".gitignore"), \
        f"ignored by {source!r}, which a fresh clone or worktree would not have"
