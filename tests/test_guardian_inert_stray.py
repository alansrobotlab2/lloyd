"""An empty, idle second copy of a runtime store is moved out of the code tree, not
alerted on (`datawatch.inert_residue` / `quarantine_inert`).

On 2026-10-02 a 0-byte `workers.db` — created by `sqlite3` opening a guessed path —
alerted hourly for seven hours and was parked on a human, because a file git ignores
gives an automod round no diff to land. Everything needed to rule on it was in one
`lstat`. Pinned here: the move happens only when every measured condition holds, each
condition that fails leaves the file where it is and alerting, and nothing is deleted
— the file is in the data root's quarantine afterwards, with a line saying what it was.

Each node builds its own tree and data root under `tmp_path`.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
for _p in (str(ROOT), str(GUARDIAN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import datawatch  # noqa: E402

NOW = 1_800_000_000.0
OLD = NOW - 2 * datawatch.INERT_MIN_AGE_SECONDS


def _with_layout_module(tree: Path) -> Path:
    """Put the repo-owned data-layout module where a real checkout has it.

    Since #2110 the quarantine destination is read out of `app/data_root.py`'s layout
    constant instead of being spelled inside the guardian, so a fixture tree that wants
    its residue moved has to carry that file — the same copy a real tree resolves, taken
    from the tree under test so the constant under test is the one that answers.
    """
    (tree / "app").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "app" / "data_root.py", tree / "app" / "data_root.py")
    return tree


@pytest.fixture
def roots(tmp_path):
    """A tree holding an old, empty `workers.db` and a data root holding the real one."""
    tree, data = tmp_path / "lloyd", tmp_path / "lloyd-data"
    tree.mkdir()
    _with_layout_module(tree)
    data.mkdir()
    (data / "workers.db").write_bytes(b"SQLite format 3\x00" + b"x" * 100)
    stray = tree / "workers.db"
    stray.write_bytes(b"")
    os.utime(stray, (OLD, OLD))
    return tree, data, stray


def _inert(tree, data, names=("workers.db",)):
    return datawatch.inert_residue(str(tree), list(names), str(data), NOW)


def test_the_incident_file_is_inert_and_is_moved_not_deleted(roots):
    tree, data, stray = roots
    assert _inert(tree, data) == ["workers.db"]

    moved = datawatch.quarantine_inert(str(tree), ["workers.db"], str(data), NOW)

    assert [name for name, _ in moved] == ["workers.db"]
    dest = Path(moved[0][1])
    assert not stray.exists()
    assert dest.is_file() and dest.stat().st_size == 0
    # Spelled here and nowhere under `agent-services/guardian/` (#2110): a test that
    # read the code's own constant could not catch that constant being moved, which is
    # the divergence this item is about.
    assert dest.parent == data / "quarantine" / "tree-strays"
    assert dest.stat().st_mtime == OLD, "the copy keeps the time the stray was written"
    rows = [json.loads(l) for l in (dest.parent / "log.jsonl").read_text().splitlines()]
    assert rows == [{**rows[0], "source": str(stray), "destination": str(dest), "size": 0}]
    assert (data / "workers.db").stat().st_size > 0, "the real store is untouched"


def test_one_byte_of_data_is_not_residue(roots):
    tree, data, stray = roots
    stray.write_bytes(b"x")
    os.utime(stray, (OLD, OLD))
    assert _inert(tree, data) == []
    assert datawatch.quarantine_inert(str(tree), ["workers.db"], str(data), NOW) == []
    assert stray.read_bytes() == b"x"


def test_a_file_just_created_is_left_for_the_next_check(roots):
    tree, data, stray = roots
    os.utime(stray, (NOW - 5, NOW - 5))
    assert _inert(tree, data) == []


def test_a_sqlite_sidecar_means_a_writer_has_it_open(roots):
    tree, data, stray = roots
    (tree / "workers.db-wal").write_bytes(b"")
    assert _inert(tree, data) == []


def test_a_store_with_no_copy_in_the_data_root_might_be_the_only_one(roots):
    tree, data, stray = roots
    (data / "workers.db").unlink()
    assert _inert(tree, data) == []


def test_a_name_nobody_listed_is_a_persons_call(roots):
    tree, data, _ = roots
    other = tree / "scratch.out"
    other.write_bytes(b"")
    os.utime(other, (OLD, OLD))
    (data / "scratch.out").write_bytes(b"x")
    assert _inert(tree, data, ["scratch.out"]) == []


def test_a_directory_and_a_link_are_never_residue(roots):
    tree, data, stray = roots
    stray.unlink()
    (tree / "sessions").mkdir()
    (data / "sessions").mkdir()
    os.symlink(data / "workers.db", tree / "workers.db")
    assert _inert(tree, data, ["sessions", "workers.db"]) == []


# ── through the guardian's hourly check ───────────────────────────────

def _guardian(tmp_path, monkeypatch, tree, data, strays):
    import guardian as gmod
    import notify

    vault = tmp_path / "obsidian"
    (vault / "memory").mkdir(parents=True)
    monkeypatch.setattr(gmod.policy, "REPO", str(tree))
    monkeypatch.setattr(gmod.policy, "DATA_ROOT", str(data))
    monkeypatch.setattr(gmod.datawatch, "stray_in_tree", lambda _tree: list(strays))
    g = gmod.Guardian.__new__(gmod.Guardian)
    g.notifier = notify.Notifier(ledger=tmp_path / "l.jsonl", state_dir=tmp_path,
                                 vault_root=str(vault),
                                 backend_url="http://127.0.0.1:1")
    g.data = types.SimpleNamespace(armed=True)
    g._alert_seen = {}
    g.last_alert = ""
    news: list[tuple[str, str]] = []
    g.notifier.announce = lambda title, body="", level="info": news.append((title, body)) or {}
    alerts: list[tuple[str, str]] = []
    real_alert = g.notifier.alert

    def _alert(level, title, body, **kw):
        alerts.append((title, body))
        return real_alert(level, title, body, **kw)

    g.notifier.alert = _alert
    return gmod, g, news, alerts


def test_the_hourly_check_moves_residue_and_raises_no_incident(roots, tmp_path, monkeypatch):
    tree, data, stray = roots
    gmod, g, news, alerts = _guardian(tmp_path, monkeypatch, tree, data, ["workers.db"])
    g._runtime_data_incident(NOW)

    assert not stray.exists()
    assert alerts == [], "an empty second copy is news, not an incident"
    assert len(news) == 1 and str(stray) in news[0][1]
    assert not (tmp_path / "l.jsonl").exists() or "workers.db" not in (
        tmp_path / "l.jsonl").read_text()


def test_what_is_not_residue_still_alerts_and_alone(roots, tmp_path, monkeypatch):
    tree, data, stray = roots
    (tree / ".t").mkdir()
    gmod, g, news, alerts = _guardian(tmp_path, monkeypatch, tree, data,
                                      ["workers.db", ".t"])
    g._runtime_data_incident(NOW)

    assert not stray.exists()
    assert len(alerts) == 1 and alerts[0][0] == gmod.RUNTIME_DATA_ALERT_TITLE
    assert os.path.join(str(tree), ".t") in alerts[0][1]
    assert os.path.join(str(tree), "workers.db") not in alerts[0][1]


def test_a_disarmed_guardian_moves_nothing(roots, tmp_path, monkeypatch):
    tree, data, stray = roots
    gmod, g, news, alerts = _guardian(tmp_path, monkeypatch, tree, data, ["workers.db"])
    g.data = types.SimpleNamespace(armed=False)
    g._runtime_data_incident(NOW)
    assert stray.exists() and news == [] and alerts == []


# ── what the retraction may claim, and where residue is allowed to go (#2110) ──

def _cleared(tmp_path, monkeypatch, tree, data, strays, armed=True):
    """Run one hourly stray check and return everything it resolved.

    `resolve` is captured rather than allowed to write, because the assertion is about
    the sentence the guardian hands the human surface — which section is open on this
    synthetic vault is `notify`'s own business, already pinned by
    `tests/test_guardian_alert_retraction.py`.
    """
    gmod, g, news, alerts = _guardian(tmp_path, monkeypatch, tree, data, strays)
    g.data = types.SimpleNamespace(armed=armed)
    cleared: list[tuple[str, str]] = []

    def _resolve(title: str, body: str = "") -> bool:
        cleared.append((title, body))
        return True          # the real method's answer: a section was closed

    g.notifier.resolve = _resolve
    g._runtime_data_incident(NOW)
    return gmod, cleared, news, alerts


def test_an_empty_tree_with_no_move_says_no_move_was_recorded(roots, tmp_path, monkeypatch):
    """Clause 1: a clear may claim only what the check measured.

    The shape that stranded `~/lloyd/workers.db` on 2026-10-03: the tree measured empty,
    the guardian moved nothing, and the line written beside the four unretracted
    `remove the in-tree copy` instructions in `memory/2026-10-02.md` was a plain
    all-clear — `nothing further to move` — which is what a tree the guardian itself
    emptied would also say. A reader cannot tell the two apart, so a vanished stray has
    no cause on record either way.
    """
    import guardian as gmod

    tree, data, stray = roots
    stray.unlink()
    _, cleared, news, alerts = _cleared(tmp_path, monkeypatch, tree, data, [])

    assert [t for t, _ in cleared] == [gmod.RUNTIME_DATA_ALERT_TITLE]
    body = cleared[0][1]
    assert "no move recorded by the guardian" in body, body
    assert "nothing further to move" not in body, (
        f"the all-clear this clause replaces is still in the line: {body}")
    assert str(tree) in body, "the line still names the root it measured (#2056)"
    assert news == [] and alerts == [], "an empty tree is neither news nor an incident"


def test_the_move_that_emptied_the_tree_names_the_move(roots, tmp_path, monkeypatch):
    """Clause 2: the same call site, the opposite fact, on the same tick.

    `moved` is bound only inside the alert branch and a move that empties the tree
    falls through to the retraction on that SAME tick, so the two sentences hang off
    one `resolve` call — the reason the old line could only ever say one thing. Here
    the guardian really did move the last stray, so the line must name it and must not
    claim nothing was recorded.
    """
    tree, data, stray = roots
    import guardian as gmod
    _, cleared, news, alerts = _cleared(tmp_path, monkeypatch, tree, data, ["workers.db"])

    assert [t for t, _ in cleared] == [gmod.RUNTIME_DATA_ALERT_TITLE]
    body = cleared[0][1]
    assert "no move recorded" not in body, body
    assert "moved 1 inert file" in body and "(workers.db)" in body, body
    assert str(tree) in body, "the line still names the root it measured (#2056)"
    assert not stray.exists()
    assert len(news) == 1, "the move is still announced as news in its own right"
    assert alerts == []


def test_the_quarantine_destination_is_one_constant_from_the_data_layout(tmp_path):
    """Clause 3: the destination is exported once and read, never re-spelled.

    Two names for the same destination is what made a stray unfindable: shipped code
    moved residue to `quarantine/tree-strays` while a route text told a person to look
    in `_quarantine/in-tree-strays`, and nothing in `app/paths.py` — the module every
    in-venv reader resolves paths from — said which was the real one. The stand-in
    below answers with a path no restatement could produce, so the guardian's call can
    only return it by reading the loaded module.
    """
    from app import paths as app_paths
    import policy

    repo = tmp_path / "repo"
    (repo / "app").mkdir(parents=True)
    (repo / "app" / "data_root.py").write_text(
        'from pathlib import Path\n\n'
        'QUARANTINE_DIR_RELATIVE = Path("quarantine") / "tree-strays-4242"\n',
        encoding="utf-8")

    got = policy.quarantine_dir(repo=str(repo), data_root=str(tmp_path / "data"))

    assert got == str(tmp_path / "data" / "quarantine" / "tree-strays-4242"), got
    assert policy.quarantine_dir(repo=str(tmp_path / "no-such-repo"),
                                  data_root=str(tmp_path / "data")) is None, (
        "a tree whose layout module cannot be loaded must answer 'no destination', "
        "not a guessed one")
    # The real tree, read the way both readers read it, lands on the exported constant.
    live = policy.quarantine_dir(repo=str(ROOT), data_root=str(app_paths.DATA_ROOT))
    assert live == str(app_paths.QUARANTINE_DIR), (
        f"the guardian resolves {live} and `app.paths` exports "
        f"{app_paths.QUARANTINE_DIR}: one destination, two spellings")


def test_no_quarantine_path_is_spelled_inside_the_guardian():
    """Clause 3, second half: the ban the constant exists to enforce.

    Checked over the guardian's own files, including this round's edits, because a
    comment naming a path is how the next divergence gets written by hand: prose in
    that directory is as much a route as `os.path.join` is. `quarantine_inert` as a
    function name is the function, not a destination, so only a quoted path segment
    counts.
    """
    offenders = []
    for path in sorted(GUARDIAN_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "tree-strays" in text or "in-tree-strays" in text:
            offenders.append(f"{path.name}: names a quarantine directory")
        if "QUARANTINE_SUBDIR" in text:
            offenders.append(f"{path.name}: still spells QUARANTINE_SUBDIR")
    assert offenders == [], offenders


def test_a_tree_with_no_layout_constant_moves_nothing_and_says_why(roots):
    """Clause 3's failure shape: no destination is a refusal, not a guess.

    The pinned snapshot the guardian runs from can fail to load the repo's layout
    module (`policy._data_root_module` returns None for it), and moving an unseen file
    to a directory this process invented is precisely the un-witnessed relocation this
    item is about. `guardian.py` catches the exception, logs it, and lets the file
    alert — the loud half of a refusal the old inline spelling could never express.
    """
    tree, data, stray = roots
    shutil.rmtree(tree / "app")

    with pytest.raises(RuntimeError, match="QUARANTINE_DIR_RELATIVE"):
        datawatch.quarantine_inert(str(tree), ["workers.db"], str(data), NOW)

    assert stray.exists(), "refusing to move means the file is still where it was"


def test_a_tree_whose_layout_module_lacks_the_constant_also_refuses(roots):
    """Clause 3's other None branch: the module LOADS and does not name the constant.

    `policy.quarantine_dir` returns None on two different failures and the node above only
    exercises one of them. This is the shape a real deployment produces — a pinned stage or
    a one-commit-behind checkout whose `app/data_root.py` parses fine and predates
    `QUARANTINE_DIR_RELATIVE` — and it is the more dangerous of the two, because nothing
    about reading the file fails: the constant is simply not there, and code that treated
    "loaded" as "answered" would move residue to a directory it had guessed.
    """
    import policy

    tree, data, stray = roots
    (tree / "app" / "data_root.py").write_text(
        'from pathlib import Path\n\nKG_DB_RELATIVE = Path("_pipeline") / "kg.sqlite"\n',
        encoding="utf-8")

    assert policy.quarantine_dir(repo=str(tree), data_root=str(data)) is None, (
        "a layout module that loads and omits the constant must answer 'no destination', "
        "the same as a module that cannot be read at all")
    with pytest.raises(RuntimeError, match="QUARANTINE_DIR_RELATIVE"):
        datawatch.quarantine_inert(str(tree), ["workers.db"], str(data), NOW)
    assert stray.exists(), "refusing to move means the file is still where it was"


# ── one reach rule, two readers (#2172) ────────────────────────────────
#
# `agent_mcp/_bash_tree_strays.py` labels every row of its journal with the instrument
# that could act on the path, and the only thing that keeps that label honest is that it
# is the alert's own rule. These two helpers build a checkout that carries the guardian's
# modules, so the file under test is a file in a tree rather than a module import, and the
# node below turns exactly one name in one constant in it.


def _staged_checkout(tmp_path, gitignore="*.db\n/cache/\n"):
    """A checkout shaped the way the reach rule reads one, with the guardian inside it.

    `cache/` is the open-set case: a name in nobody's `RUNTIME_NAMES`, hidden by the
    checkout's own ignore rule, reachable for exactly one reason — git tracks nothing
    under it. The guardian's modules are copied in as `guardian-stage.sh:42` does
    (`cp "$SRC"/*.py "$STAGE"/`) and staged into git's index, because this node's variable
    is a name inside one of those files and both readers have to be reading that file.
    """
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / ".gitignore").write_text(gitignore, encoding="utf-8")
    (tree / "README.md").write_text("# lloyd\n", encoding="utf-8")

    def run(*args):
        return subprocess.run(["git", "-C", str(tree), *args], check=True,
                              capture_output=True)

    run("init", "-q")
    dest = tree / "agent-services" / "guardian"
    dest.mkdir(parents=True)
    for module in sorted(GUARDIAN_DIR.glob("*.py")):
        shutil.copy2(module, dest / module.name)
    run("add", "-f", ".gitignore", "README.md", "agent-services")
    return tree


def _reach_rule(tree: Path):
    """The tree's own copy of the reach rule, loaded the way the bracket loads it.

    The alert cannot be read off the `datawatch` this file imported at the top: that one is
    the live checkout's file, while the label is computed from THIS tree's file, so
    comparing them would be comparing two files and calling one name the variable. One
    file, one load per call, and both answers come out of it.
    """
    import importlib.util

    src = tree / "agent-services" / "guardian" / "datawatch.py"
    sys.path.insert(0, str(src.parent))
    try:
        spec = importlib.util.spec_from_file_location("reach_rule_under_test", src)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(src.parent))
    return module


def test_the_alert_and_the_journalled_label_move_on_one_name(tmp_path):
    """Clause 3 (#2172): `stray_in_tree` builds its candidates through the exported
    predicate, so turning ONE name moves the alert and the journalled label together.

    The name turned is in `KNOWN_GOOD_TOPLEVEL`: `cache` is added to it in this tree's own
    copy of `datawatch.py`. Before, the check reports `cache` and the row for
    `cache/r1873/x.json` reads `guardian-strays`; after, the check reports nothing and the
    same row reads `bracket-only`. Nothing else is touched — same tree, same index, same
    file on disk.

    That is the property two hand-maintained reach lists could never have kept, and the
    reason it holds is structural: the label is produced by
    `_bash_tree_strays._actionable_by`, which loads the reach rule from the tree it is
    labelling a row about and calls `reachable_by_stray_check`, the same function
    `stray_in_tree` filters its candidates with. A drift between the alert and the label
    is a change to one function that has to be made twice, in two files, in both
    directions.
    """
    from agent_mcp import _bash_tree_strays

    tree = _staged_checkout(tmp_path)
    (tree / "cache" / "r1873").mkdir(parents=True)
    (tree / "cache" / "r1873" / "x.json").write_text("{}", encoding="utf-8")

    assert _reach_rule(tree).stray_in_tree(str(tree)) == ["cache"]
    assert _bash_tree_strays._actionable_by(tree, ["cache/r1873/x.json"]) == "guardian-strays"

    src = tree / "agent-services" / "guardian" / "datawatch.py"
    before = src.read_text(encoding="utf-8")
    moved = before.replace('    "node_modules",\n', '    "node_modules",\n    "cache",\n', 1)
    assert moved != before, (
        "the fixture can no longer turn the one name this node turns: the node would "
        "pass without anything moving")
    src.write_text(moved, encoding="utf-8")

    assert _reach_rule(tree).stray_in_tree(str(tree)) == [], (
        "`cache` is explained by KNOWN_GOOD_TOPLEVEL now, so the alert must stop naming it")
    assert _bash_tree_strays._actionable_by(tree, ["cache/r1873/x.json"]) == "bracket-only", (
        "the alert stopped reaching this path while the label kept claiming it could — "
        "the exact drift this item exists to make impossible")
