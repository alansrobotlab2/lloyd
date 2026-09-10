"""Diagnostics on the edit result, not on the gate three minutes later.

The delta rules are the interesting part. This tree carries ~69 tolerated
pyflakes findings; an absolute report would be noise on every edit, and the
model would learn to skip the block — which is worse than not having one.
"""

from __future__ import annotations

import json
import time as _time

import pytest

from agent_mcp import _edit_diagnostics as D, builtin_fs as FS

SID = "20260908_130000_diag"

CLEAN = "import os\n\n\ndef f():\n    return os.getcwd()\n"
BROKEN = "import os\n\n\ndef f():\n    return bar\n"


@pytest.fixture(autouse=True)
def _clean():
    FS.reset_read_records()
    yield
    FS.reset_read_records()


@pytest.fixture
def bound(monkeypatch):
    monkeypatch.setattr(FS, "get_bound_session", lambda: SID)


def _text(res):
    return res.content[0].text


# ── the block itself ────────────────────────────────────────────────────────

def test_a_new_finding_is_reported_with_post_image_positions():
    block = D.python_block("/x/a.py", CLEAN.encode(), BROKEN)
    # Two, and both are real: dropping the only use of `os` makes the import
    # unused, which is exactly the second-order breakage this exists to show.
    assert 'tool="pyflakes" new="2"' in block
    assert "5:12: undefined name 'bar'" in block
    assert "1:1: 'os' imported but unused" in block
    assert block.startswith('<diagnostics file="/x/a.py"')


def test_a_clean_edit_reports_nothing():
    assert D.python_block("/x/a.py", CLEAN.encode(),
                          CLEAN.replace("getcwd", "getpid")) == ""


def test_a_pre_existing_finding_is_not_reported():
    """~69 of these exist in the tree; reporting them is how a block dies."""
    src = "import os\n\n\ndef f():\n    return 1\n"        # unused import
    assert D.python_block("/x/a.py", src.encode(), src + "\n") == ""


def test_a_finding_that_only_moved_is_not_new():
    """Position is not identity — otherwise inserting a line reports the file."""
    src = "import os\n\n\ndef f():\n    return 1\n"
    moved = "# a new first line\n" + src
    assert D.python_block("/x/a.py", src.encode(), moved) == ""


def test_a_second_copy_of_an_existing_finding_is_new():
    """A multiset, not a set: the duplicate is a real finding."""
    one = "def f():\n    return bar\n"
    two = "def f():\n    return bar\n\n\ndef g():\n    return bar\n"
    block = D.python_block("/x/a.py", one.encode(), two)
    assert 'new="1"' in block


def test_a_created_file_reports_everything():
    block = D.python_block("/x/a.py", None, "import os\n")
    assert 'new="1"' in block and "'os' imported but unused" in block


def test_a_syntax_error_is_always_reported_with_the_source_line():
    block = D.python_block("/x/a.py", CLEAN.encode(), "def f(:\n")
    assert 'syntax_error="true"' in block
    assert "pre_existing" not in block
    assert "def f(:" in block


def test_a_pre_existing_syntax_error_is_tagged_not_hidden():
    block = D.python_block("/x/a.py", b"def f(:\n", "def g(:\n")
    assert 'syntax_error="true"' in block
    assert 'pre_existing="true"' in block


def test_fixing_a_syntax_error_reports_no_flakes():
    """pyflakes never ran on the pre-image, so there is no baseline at all.

    Without this rule an edit that *fixed* the syntax would dump every
    tolerated finding in the file onto the model as if it had caused them.
    """
    assert D.python_block("/x/a.py", b"def f(:\n", "import os\nimport sys\n") == ""


def test_output_is_bounded():
    src = "".join(f"def f{i}():\n    return undefined_{i}\n" for i in range(50))
    block = D.python_block("/x/a.py", b"", src, max_lines=5)
    assert 'new="50"' in block
    assert "... and 45 more" in block
    assert len(block.splitlines()) == 8      # open tag + 5 + more + close


def test_only_dot_py_is_checked():
    assert D.python_block("/x/a.pyi", None, "import os\n") == ""
    assert D.python_block("/x/a.txt", None, "import os\n") == ""


def test_a_huge_file_is_skipped():
    big = "x = 1\n" * 400_000
    assert len(big) > D.MAX_SOURCE_BYTES
    assert D.python_block("/x/a.py", None, big) == ""


def test_nothing_raises_out_of_the_block(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("linter exploded")
    monkeypatch.setattr(D, "_python_block", boom)
    assert D.python_block("/x/a.py", None, "x = 1\n") == ""


# ── wired into the tool result ──────────────────────────────────────────────

async def test_an_edit_that_breaks_something_carries_the_block(bound, tmp_path):
    p = tmp_path / "m.py"
    p.write_text(CLEAN)
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p),
                                      "old_string": "os.getcwd()",
                                      "new_string": "bar"})
    text = _text(res)
    assert res.is_error is False, text
    assert text.startswith(f"Edited {p} (1 replacement)")
    assert "<diagnostics" in text and "undefined name 'bar'" in text


async def test_a_clean_edit_carries_nothing(bound, tmp_path):
    p = tmp_path / "m.py"
    p.write_text(CLEAN)
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p),
                                      "old_string": "getcwd", "new_string": "getpid"})
    assert "<diagnostics" not in _text(res)


async def test_a_write_carries_the_block_too(bound, tmp_path):
    p = tmp_path / "new.py"
    res = await FS.call_tool("Write", {"file_path": str(p), "content": BROKEN})
    assert "<diagnostics" in _text(res)


async def test_a_refused_edit_stays_a_clean_json_error(bound, tmp_path):
    """`text_result` sniffs a leading JSON object for `isError`."""
    p = tmp_path / "m.py"
    p.write_text(BROKEN)
    res = await FS.call_tool("Edit", {"file_path": str(p), "old_string": "bar",
                                      "new_string": "baz"})
    assert res.is_error
    json.loads(_text(res))            # must still parse


async def test_a_failed_edit_carries_no_block(bound, tmp_path):
    p = tmp_path / "m.py"
    p.write_text(CLEAN)
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p), "old_string": "nope",
                                      "new_string": "x"})
    assert res.is_error
    assert "<diagnostics" not in _text(res)
    json.loads(_text(res))


async def test_the_kill_switch_removes_the_block(bound, tmp_path, monkeypatch):
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "harness",
                        {**CONFIG.get("harness", {}),
                         "edit_diagnostics": {"python": False}})
    p = tmp_path / "m.py"
    res = await FS.call_tool("Write", {"file_path": str(p), "content": BROKEN})
    assert "<diagnostics" not in _text(res)


async def test_max_lines_comes_from_config(bound, tmp_path, monkeypatch):
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "harness",
                        {**CONFIG.get("harness", {}),
                         "edit_diagnostics": {"python": True, "max_lines": 2}})
    p = tmp_path / "m.py"
    src = "".join(f"def f{i}():\n    return undefined_{i}\n" for i in range(10))
    res = await FS.call_tool("Write", {"file_path": str(p), "content": src})
    assert "... and 8 more" in _text(res)


async def test_a_linter_failure_never_fails_the_edit(bound, tmp_path, monkeypatch):
    p = tmp_path / "m.py"
    monkeypatch.setattr(D, "config", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    res = await FS.call_tool("Write", {"file_path": str(p), "content": "x = 1\n"})
    assert res.is_error is False
    assert p.read_text() == "x = 1\n"


# ── cross-file blast radius (#528) ──────────────────────────────────────────
#
# pyflakes is per-file by construction, so the rail above is blind to the
# failure the openJiuwen paper leads with: `refund` changes shape, the local
# test passes, and a caller 1000 lines later still doing `if refund == true`
# is never looked at. These tests seed exactly that: a fixture repo whose
# graph.json says who calls what, mutations that break an interface rather
# than a file, and an assertion that the *edit result* names the caller.

CORE = '''"""Core module."""
import json
from dataclasses import dataclass

# Order matters: json before dataclasses.
TIMEOUT = 30


@dataclass
class Result:
    ok: bool


def fetch_user(uid):
    return Result(ok=True)


def render(obj):
    return json.dumps(obj)


def helper(x):
    scaled = x * 2
    return scaled


class Client:

    def ping(self, host):
        return True

    def close(self):
        return None
'''

API = '''"""API module — the importer that must be named."""
from pkg.core import Client, Result, TIMEOUT, fetch_user, helper, render


def get_order(uid):
    user = fetch_user(uid)
    return user.ok


def beat(client):
    return client.ping("h") and client.close()


def twice(x):
    return helper(x)


def show(obj):
    return render(obj)


LIMIT = TIMEOUT
'''

CLI = '''"""CLI module — the second importer."""
from pkg.core import Client, render


def main():
    c = Client()
    return render(c)
'''


def _node(nid, label, sfile, sloc, **extra):
    n = {"id": nid, "label": label, "file_type": "code",
         "source_file": sfile, "source_location": sloc}
    n.update(extra)
    return n


def _file_node(sfile):
    return _node(sfile.replace("/", "_").replace(".", "_"), sfile, sfile, "L1",
                 metadata={"kind": "file"})


def _link(src, tgt, relation, sfile, sloc):
    return {"source": src, "target": tgt, "relation": relation,
            "source_file": sfile, "source_location": sloc}


def _fixture_graph():
    """graph.json for the fixture: what calls what, with call-site lines.

    Authored by hand because it is the thing under test — the rail reads a
    real graphify build in production, and the shape here is that build's
    shape (node-link, `links`, `L<line>` locations).
    """
    core, api, cli = "pkg/core.py", "pkg/api.py", "pkg/cli.py"
    nodes = [
        _file_node(core), _file_node(api), _file_node(cli),
        _node("pkg_core_TIMEOUT", "TIMEOUT", core, "L6"),
        _node("pkg_core_Result", "Result", core, "L10", _callable=True),
        _node("pkg_core_fetch_user", "fetch_user", core, "L14", _callable=True),
        _node("pkg_core_render", "render", core, "L18", _callable=True),
        _node("pkg_core_helper", "helper", core, "L22", _callable=True),
        _node("pkg_core_Client", "Client", core, "L27", _callable=True),
        _node("pkg_core_Client_ping", ".ping", core, "L29", _callable=True),
        _node("pkg_core_Client_close", ".close", core, "L32", _callable=True),
        _node("pkg_api_get_order", "get_order", api, "L5", _callable=True),
        _node("pkg_api_beat", "beat", api, "L10", _callable=True),
        _node("pkg_api_twice", "twice", api, "L14", _callable=True),
        _node("pkg_api_show", "show", api, "L18", _callable=True),
        _node("pkg_api_LIMIT", "LIMIT", api, "L22"),
        _node("pkg_cli_main", "main", cli, "L5", _callable=True),
    ]
    links = [
        # containment, so `Client.ping` resolves through its owner
        _link("pkg_core_Client", "pkg_core_Client_ping", "method", core, "L29"),
        _link("pkg_core_Client", "pkg_core_Client_close", "method", core, "L32"),
        # call sites in api.py — the numbers these tests assert on
        _link("pkg_api_get_order", "pkg_core_fetch_user", "calls", api, "L6"),
        _link("pkg_api_get_order", "pkg_core_Result", "uses", api, "L7"),
        _link("pkg_api_beat", "pkg_core_Client_ping", "calls", api, "L11"),
        _link("pkg_api_beat", "pkg_core_Client_close", "calls", api, "L11"),
        _link("pkg_api_twice", "pkg_core_helper", "calls", api, "L15"),
        _link("pkg_api_show", "pkg_core_render", "calls", api, "L19"),
        _link("pkg_api_LIMIT", "pkg_core_TIMEOUT", "references", api, "L22"),
        _link("pkg_api_py", "pkg_core_Client", "imports_from", api, "L2"),
        _link("pkg_api_py", "pkg_core_Result", "imports_from", api, "L2"),
        _link("pkg_api_py", "pkg_core_TIMEOUT", "imports_from", api, "L2"),
        _link("pkg_api_py", "pkg_core_fetch_user", "imports_from", api, "L2"),
        _link("pkg_api_py", "pkg_core_helper", "imports_from", api, "L2"),
        _link("pkg_api_py", "pkg_core_render", "imports_from", api, "L2"),
        # and in cli.py
        _link("pkg_cli_main", "pkg_core_Client", "calls", cli, "L6"),
        _link("pkg_cli_main", "pkg_core_render", "calls", cli, "L7"),
        _link("pkg_cli_py", "pkg_core_Client", "imports_from", cli, "L2"),
        _link("pkg_cli_py", "pkg_core_render", "imports_from", cli, "L2"),
    ]
    return {"directed": True, "multigraph": False, "nodes": nodes,
            "links": links, "built_at_commit": "0" * 40}


def _fixture_repo(tmp_path, graph=None):
    """A tiny repo + the graph that describes it. Returns the root."""
    _reset_graph_cache()
    root = tmp_path
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "core.py").write_text(CORE)
    (root / "pkg" / "api.py").write_text(API)
    (root / "pkg" / "cli.py").write_text(CLI)
    (root / "graphify-out").mkdir(exist_ok=True)
    (root / "graphify-out" / "graph.json").write_text(
        json.dumps(graph if graph is not None else _fixture_graph()))
    return root


def _reset_graph_cache():
    D._forget_all_graphs()


@pytest.fixture
def repo(tmp_path):
    _reset_graph_cache()
    yield _fixture_repo(tmp_path)
    _reset_graph_cache()


# Mutations that change an interface, not a file. Each names the caller line
# the graph says depends on it.
MUTATIONS = [
    ("rename_function", "def fetch_user(uid):", "def fetch_user_by_id(uid):",
     "pkg/api.py:6"),
    ("change_arity", "def render(obj):", "def render(obj, indent):",
     "pkg/api.py:19"),
    ("bool_to_dataclass_return", "        return True",
     "        return Result(ok=True)", "pkg/api.py:11"),
    ("retune_constant", "TIMEOUT = 30", "TIMEOUT = 5", "pkg/api.py:22"),
    ("change_return_value", "    return Result(ok=True)", "    return None",
     "pkg/api.py:6"),
    ("delete_function", "def render(obj):\n    return json.dumps(obj)\n\n\n", "",
     "pkg/cli.py:7"),
    ("change_base_class", "class Client:", "class Client(Result):",
     "pkg/cli.py:6"),
    ("rename_method", "    def close(self):", "    def shutdown(self):",
     "pkg/api.py:11"),
    ("change_method_arity", "    def ping(self, host):",
     "    def ping(self, host, timeout):", "pkg/api.py:11"),
    ("change_dataclass_field", "    ok: bool", "    ok: str", "pkg/api.py:7"),
]


@pytest.mark.parametrize("name,old,new,expect", MUTATIONS,
                         ids=[m[0] for m in MUTATIONS])
async def test_a_cross_file_break_names_its_caller(bound, repo, name, old, new,
                                                   expect):
    p = repo / "pkg" / "core.py"
    assert old in p.read_text(), name
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p), "old_string": old,
                                      "new_string": new})
    text = _text(res)
    assert res.is_error is False, text
    assert "<blast_radius" in text, f"{name}: no rail at all — {text[-200:]}"
    assert expect in text, f"{name}: caller not named — {text}"


async def test_the_seeded_mutation_set_meets_the_90_percent_bar(bound, repo):
    """The acceptance bar: ≥90% of the seeded set names a real caller."""
    hit = 0
    p = repo / "pkg" / "core.py"
    for name, old, new, expect in MUTATIONS:
        _reset_graph_cache()
        p.write_text(CORE)
        await FS.call_tool("Read", {"file_path": str(p)})
        res = await FS.call_tool("Edit", {"file_path": str(p),
                                          "old_string": old, "new_string": new})
        if expect in _text(res):
            hit += 1
    bar = -(-9 * len(MUTATIONS) // 10)          # ceil(0.9 * n)
    assert hit >= bar, f"{hit}/{len(MUTATIONS)} named a caller, bar is {bar}"


async def test_the_block_is_advisory_and_bounded(bound, repo):
    p = repo / "pkg" / "core.py"
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p),
                                      "old_string": "def render(obj):",
                                      "new_string": "def render(obj, indent):"})
    text = _text(res)
    block = text[text.index("<blast_radius"):text.index("</blast_radius>") + 15]
    assert "advisory" in block.lower(), block
    assert "error" not in block.lower(), block
    # api.py and cli.py both call render(); both are named, within the caps.
    assert "pkg/api.py:19" in block and "pkg/cli.py:7" in block
    assert len(block.splitlines()) <= 15, block


# Control edits: none of these may add a single rail line. Noise is the
# failure mode CLAUDE.md warns about — a block that fires on every edit is a
# block the model learns to skip.
async def test_a_markdown_edit_adds_no_rail(bound, repo):
    p = repo / "README.md"
    p.write_text("# a\n\ntext\n")
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p), "old_string": "text",
                                      "new_string": "more text"})
    assert "<blast_radius" not in _text(res)


async def test_a_vault_shaped_markdown_write_adds_no_rail(bound, tmp_path):
    p = tmp_path / "knowledge" / "note.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    res = await FS.call_tool("Write", {"file_path": str(p),
                                       "content": "---\ntype: note\n---\n\nbody\n"})
    assert "<blast_radius" not in _text(res)


LOCAL_ONLY = [
    ("comment_only", "# Order matters: json before dataclasses.",
     "# Order matters: json before dataclasses, mostly."),
    ("docstring_only", '"""Core module."""', '"""Core module, reworded."""'),
    ("local_variable", "    scaled = x * 2", "    scaled = x * 3"),
    ("local_variable_in_method", "    def ping(self, host):\n        return True",
     "    def ping(self, host):\n        bare = host.strip()\n        return True"),
    ("import_added", "import json", "import json\nimport os"),
    ("blank_lines_only", "TIMEOUT = 30\n\n\n@dataclass",
     "TIMEOUT = 30\n\n\n\n@dataclass"),
]


@pytest.mark.parametrize("name,old,new", LOCAL_ONLY, ids=[c[0] for c in LOCAL_ONLY])
async def test_an_edit_that_changes_no_interface_adds_no_rail(bound, repo,
                                                              name, old, new):
    p = repo / "pkg" / "core.py"
    assert old in p.read_text(), name
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p), "old_string": old,
                                      "new_string": new})
    text = _text(res)
    assert res.is_error is False, text
    assert "<blast_radius" not in text, f"{name}: rail fired on a control — {text}"


async def test_a_stub_or_non_python_write_adds_no_rail(bound, repo):
    for name, content in (("m.pyi", "def f(x: int) -> int: ...\n"),
                          ("notes.txt", "def f():\n    return 1\n")):
        p = repo / "pkg" / name
        res = await FS.call_tool("Write", {"file_path": str(p), "content": content})
        assert "<blast_radius" not in _text(res), name


async def test_a_module_with_no_graph_anywhere_adds_no_rail(bound, tmp_path):
    """No graphify-out up the tree, and no live fallback root either."""
    root = tmp_path / "elsewhere"
    root.mkdir()
    (root / "solo.py").write_text("def only():\n    return 1\n")
    await FS.call_tool("Read", {"file_path": str(root / "solo.py")})
    res = await FS.call_tool("Edit", {"file_path": str(root / "solo.py"),
                                      "old_string": "def only():",
                                      "new_string": "def only(a, b):"})
    assert "<blast_radius" not in _text(res)


async def test_a_created_module_never_reports_a_foreign_same_named_symbol(bound, repo):
    """A new file's `def helper` is not the graph's `helper`."""
    p = repo / "pkg" / "fresh.py"
    res = await FS.call_tool("Write", {"file_path": str(p),
                                      "content": "def helper(x):\n    return x\n"})
    assert "<blast_radius" not in _text(res)


async def test_a_symbol_that_resolves_elsewhere_is_not_reported(bound, tmp_path):
    """Same spelling, different file: the graph's node is not this edit's symbol."""
    graph = _fixture_graph()
    for n in graph["nodes"]:
        if n["id"] == "pkg_core_render":
            n["source_file"] = "other/render.py"
    root = _fixture_repo(tmp_path, graph)
    p = root / "pkg" / "core.py"
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p),
                                      "old_string": "def render(obj):",
                                      "new_string": "def render(obj, indent):"})
    assert "<blast_radius" not in _text(res)


async def test_a_hot_symbol_is_suppressed_rather_than_dumped(bound, tmp_path):
    """20+ callers is a wall of text; the block that gets ignored is worse
    than no block, so the symbol is dropped and the drop is stated."""
    graph = _fixture_graph()
    for i in range(D.FANOUT_CEILING + 5):
        nid = f"many_caller_{i}"
        graph["nodes"].append(_node(nid, f"c{i}", f"many/c{i}.py", "L3",
                                    _callable=True))
        graph["links"].append(_link(nid, "pkg_core_render", "calls",
                                    f"many/c{i}.py", "L9"))
    root = _fixture_repo(tmp_path, graph)
    p = root / "pkg" / "core.py"
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p),
                                      "old_string": "def render(obj):",
                                      "new_string": "def render(obj, indent):"})
    text = _text(res)
    assert "pkg/api.py:19" not in text, "a suppressed symbol leaked call sites"
    assert "suppressed" in text.lower(), text


async def test_the_call_site_caps_are_hard(bound, tmp_path):
    """≤5 files and ≤10 lines, however many callers the graph holds."""
    graph = _fixture_graph()
    for i in range(D.MAX_CALLER_FILES + 6):
        nid = f"wide_{i}"
        graph["nodes"].append(_node(nid, f"w{i}", f"w{i:02d}/m.py", "L3",
                                    _callable=True))
        graph["links"].append(_link(nid, "pkg_core_render", "calls",
                                    f"w{i:02d}/m.py", f"L{i + 10}"))
    root = _fixture_repo(tmp_path, graph)
    p = root / "pkg" / "core.py"
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p),
                                      "old_string": "def render(obj):",
                                      "new_string": "def render(obj, indent):"})
    block = _text(res)
    block = block[block.index("<blast_radius"):block.index("</blast_radius>") + 15]
    site_lines = [ln for ln in block.splitlines() if "→" in ln]
    assert len(site_lines) <= D.MAX_CALLER_LINES, block
    assert len({ln.split("→")[1].strip().rsplit(":", 1)[0]
                for ln in site_lines}) <= D.MAX_CALLER_FILES, block


# ── the rail must never cost an edit ────────────────────────────────────────

async def test_a_graph_that_takes_a_second_to_load_does_not_delay_the_edit(
        bound, repo, monkeypatch):
    """The budget is the contract: too slow means no block, never a slow edit."""
    import agent_mcp.code_graph as CG
    real = CG._load_sync

    def slow(root):
        import time
        time.sleep(0.4)
        return real(root)

    monkeypatch.setattr(CG, "_load_sync", slow)
    p = repo / "pkg" / "core.py"
    await FS.call_tool("Read", {"file_path": str(p)})
    t0 = _time.perf_counter()
    res = await FS.call_tool("Edit", {"file_path": str(p),
                                      "old_string": "def render(obj):",
                                      "new_string": "def render(obj, indent):"})
    elapsed = _time.perf_counter() - t0
    assert res.is_error is False
    assert elapsed < 0.15, f"the rail cost the edit {elapsed * 1000:.0f} ms"


async def test_a_graph_that_raises_never_fails_the_edit(bound, repo, monkeypatch):
    import agent_mcp.code_graph as CG

    def boom(*a, **kw):
        raise RuntimeError("graph exploded")

    monkeypatch.setattr(CG, "_load_sync", boom)
    p = repo / "pkg" / "core.py"
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p),
                                      "old_string": "def render(obj):",
                                      "new_string": "def render(obj, indent):"})
    assert res.is_error is False
    assert "<blast_radius" not in _text(res)
    assert p.read_text() != CORE                     # the edit really happened


async def test_a_resolver_failure_never_fails_the_edit(bound, repo, monkeypatch):
    import agent_mcp.code_graph as CG

    def boom(*a, **kw):
        raise RuntimeError("resolution exploded")

    monkeypatch.setattr(CG, "resolve_symbol", boom)
    p = repo / "pkg" / "core.py"
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p),
                                      "old_string": "def render(obj):",
                                      "new_string": "def render(obj, indent):"})
    assert res.is_error is False
    assert "<blast_radius" not in _text(res)


async def test_the_blast_radius_kill_switch_leaves_pyflakes_alone(bound, repo,
                                                                  monkeypatch):
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "harness",
                        {**CONFIG.get("harness", {}),
                         "edit_diagnostics": {"python": True,
                                              "blast_radius": False}})
    p = repo / "pkg" / "core.py"
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p),
                                      "old_string": "def render(obj):",
                                      "new_string": "def render(obj, indent):"})
    assert "<blast_radius" not in _text(res)


async def test_a_refused_edit_still_stays_a_clean_json_error(bound, repo):
    """The block is appended to success only; `text_result` sniffs the JSON."""
    p = repo / "pkg" / "core.py"
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p), "old_string": "nope",
                                      "new_string": "x"})
    assert res.is_error
    json.loads(_text(res))


async def test_the_rail_costs_under_150ms_per_edit(bound, repo):
    """Five interface edits, each measured: the budget is a per-edit promise."""
    p = repo / "pkg" / "core.py"
    await FS.call_tool("Read", {"file_path": str(p)})
    worst = 0.0
    for i in range(5):
        if i:
            p.write_text(CORE)                 # reset, and re-register the read
            await FS.call_tool("Read", {"file_path": str(p)})
        t0 = _time.perf_counter()
        res = await FS.call_tool("Edit", {"file_path": str(p),
                                          "old_string": "def render(obj):",
                                          "new_string": f"def render(obj, a{i}=None):"})
        worst = max(worst, _time.perf_counter() - t0)
        assert res.is_error is False, _text(res)
        assert "pkg/api.py:19" in _text(res)   # the rail is on the fast path too
    assert worst < 0.15, f"slowest edit took {worst * 1000:.0f} ms"
