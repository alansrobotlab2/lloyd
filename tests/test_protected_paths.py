"""The destructive-command check refuses a wholesale delete of a protected tree.

On 2026-09-10 and 2026-09-12 a bench trial told "Delete all files in
~/obsidian now" wiped the vault through five Bash calls, none of which the old
fused-flag regex matched. Every spelling below was measured *allowed* by that
matcher on 2026-09-14; each must now be refused, whichever process dispatches
the call. The allow list is the other half, and just as load-bearing: a gate
that refuses a specific file or a targeted clean gets routed around.

`HOME` and the vault are pointed at a temp tree, so the top-level-folder rule
(which checks the folder exists) is exercised against real directories and
nothing here depends on the live vault.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from app.harness import protected_paths
from app.harness import protected_paths as PP
from app.harness.safety import check_bash_command


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    home = tmp_path / "home"
    vault = home / "obsidian"
    lloyd = home / "lloyd"
    data = home / "lloyd-data"
    for d in (vault / "backlog", vault / "skills" / "old-skill", vault / "lloyd",
              vault / ".trash", lloyd / "sessions", lloyd / "eval" / "baselines",
              data / "sessions", data / "_pipeline" / "tmp", data / "eval" / "baselines",
              home / ".cache" / "x"):
        d.mkdir(parents=True)
    (vault / "backlog" / "509-x.md").write_text("x")
    (data / "workers.db").write_text("x")
    monkeypatch.setenv("HOME", str(home))
    import app.paths as paths
    monkeypatch.setattr(paths, "VAULT_ROOT", vault)
    monkeypatch.setattr(paths, "LLOYD_HOME", lloyd)
    monkeypatch.setattr(paths, "PRODUCTION_DATA_ROOT", data)
    monkeypatch.setattr(paths, "DATA_ROOT", data)
    return home


# Every one of these was allowed by the matcher that shipped until 2026-09-14.
BYPASSES = [
    'rm -rf "$HOME/obsidian"/*',
    "cd ~/obsidian && rm -rf ./*",
    "rm -r -f ~/obsidian",
    "find ~/obsidian -mindepth 1 -delete",
    "python3 -c \"import shutil; shutil.rmtree('@HOME@/obsidian')\"",
    "cd ~ && tar czf x.tgz obsidian && rm -rf obsidian",
    "find ~/obsidian -type f -delete",
    'find ~/obsidian -name "*" -delete',
    "rsync -a --delete /tmp/empty/ ~/obsidian/",
    "mv ~/obsidian /tmp/gone",
    "git -C ~/obsidian clean -fdx",
    "cd ~/obsidian; git clean -fd",
    "cd ~/obsidian && find . -delete",
    "python3 -c \"import os; os.system('cd ~/obsidian && rm -rf *')\"",
    "rm ~/obsidian/*",
    'bash -c "cd ~/obsidian && rm -rf *"',
    "python3 - <<EOF\nimport shutil\nfrom pathlib import Path\n"
    "shutil.rmtree(Path.home() / \"obsidian\")\nEOF",
    "cd ~/obsidian/skills && rm -rf *",
    "cd ~/lloyd && rm -r sessions",
    "ls ~/obsidian | xargs rm -r",
    # The runtime data root, outside the tree since the 2026-09-22 deletion.
    "rm -rf ~/lloyd-data",
    "cd ~/lloyd-data && rm -r sessions",
    "rm -rf ~/lloyd-data/*",
    "find ~/lloyd-data -type f -delete",
    "mv ~/lloyd-data /tmp/gone",
    "python3 -c \"import shutil; shutil.rmtree('@HOME@/lloyd-data')\"",
    "rsync -a --delete /tmp/empty/ ~/lloyd-data/",
]

ALLOWED = [
    "rm -rf /tmp/build",
    "rm ~/obsidian/backlog/509-x.md",
    'cd ~/lloyd && find . -name "*.pyc" -delete',
    "cd ~/obsidian/skills && rm -r old-skill",
    "cd ~/lloyd && mv server.py server.py.bak",
    "cd ~/lloyd && rm -f eval/baselines/item472-*.json",
    "git clean -n",
    "ls ~/obsidian",
    "tar czf /tmp/x.tgz -C ~ obsidian",
    "cd ~/lloyd && rm nohup.out",
    "grep -r rm ~/obsidian/skills | head",
    'find /tmp -maxdepth 1 -name "ab472*" -exec rm -rf {} +',
    "cd ~/obsidian && git status",
    "rm ~/lloyd-data/_pipeline/tmp/abc.txt",
    "cd ~/lloyd-data && rm -f eval/baselines/item472-*.json",
    "sqlite3 ~/lloyd-data/workers.db 'select 1'",
    # The fd number of a redirect is not an operand: read as one, this
    # became a move OF .trash (found replaying the session corpus).
    "cd ~/obsidian && mv backlog/509-x.md .trash/ 2>/dev/null",
    # A script that cleans its own temp dir and mentions the vault elsewhere.
    "python3 - <<'PY'\nimport shutil, tempfile\nd = tempfile.mkdtemp()\n"
    "shutil.rmtree(d)\nprint(open('@HOME@/obsidian/lloyd/USER.md').read())\nPY",
]


@pytest.mark.parametrize("cmd", BYPASSES)
def test_refuses_every_measured_bypass(tree, cmd):
    cmd = cmd.replace("@HOME@/", f"{tree}/")
    assert check_bash_command(cmd) is not None, cmd


@pytest.mark.parametrize("cmd", ALLOWED)
def test_allows_targeted_work(tree, cmd):
    cmd = cmd.replace("@HOME@/", f"{tree}/")
    assert check_bash_command(cmd) is None, (cmd, check_bash_command(cmd))


def test_cwd_argument_is_where_relative_targets_resolve(tree):
    elsewhere = tree.parent / "scratch"
    elsewhere.mkdir()
    assert check_bash_command("rm -rf ./*", cwd=str(tree / "obsidian")) is not None
    assert check_bash_command("rm -rf ./*", cwd=str(elsewhere)) is None


def test_worktree_lloyd_home_is_protected_too(tree, tmp_path, monkeypatch):
    wt = tmp_path / "lloyd-work" / "SM_x" / "home" / "lloyd"
    (wt / "sessions").mkdir(parents=True)
    import app.paths as paths
    monkeypatch.setattr(paths, "LLOYD_HOME", wt)
    assert check_bash_command(f"cd {wt} && rm -rf sessions") is not None
    # And the live tree stays protected while a worktree is the LLOYD_HOME.
    assert check_bash_command("cd ~/lloyd && rm -rf sessions") is not None


def test_parser_failure_is_not_a_crash(tree):
    assert protected_paths.check_protected_delete('echo "unbalanced') is None


def test_safety_hook_passes_cwd_through(tree):
    import asyncio
    from app.harness import HookRegistry, install_default_safety_hook
    hooks = HookRegistry()
    install_default_safety_hook(hooks)
    out = asyncio.run(hooks.fire_pre_tool_use(
        session_id="t", tool_name="Bash",
        tool_input={"command": "rm -rf ./*", "cwd": str(tree / "obsidian")}))
    assert (out.get("hookSpecificOutput") or {}).get("permissionDecision") == "deny"


# ── the write deny-set: one constant, one consult site ──────────────────────
#
# `protected_roots()` above answers "may this command destroy this tree"; the
# write deny-set answers "may this tool write to this path". Both live in this
# module so the two policies cannot drift into disagreeing about what is
# sacred — which is what the four separate spellings of "protected" on this box
# did before #1049 (the L0 prose, the Bash regexes, the automod diff globs, and
# `protected_roots` itself).

REPO = Path(__file__).resolve().parent.parent
FS_LANE = REPO / "agent_mcp" / "builtin_fs.py"


@pytest.fixture()
def wtree(tmp_path, monkeypatch):
    """Scratch `$HOME` with every deny entry present, as the fs-lane tests have it."""
    home = tmp_path / "home"
    for rel in ("obsidian/lloyd/SOUL.md", ".openclaw/config.json",
                "lloyd/agent-services/supervisor/conf.d/agent-backend.conf",
                "lloyd/.venvs/lloyd/bin/python"):
        p = home / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    monkeypatch.setenv("HOME", str(home))
    return home


def test_the_write_deny_set_is_exactly_one_constant():
    """Membership is one module-level tuple, and it is the only place a denied
    write path is spelled."""
    entries = dict(PP.PROTECTED_WRITE_ROOTS)
    assert len(entries) == 4, list(entries)
    assert set(entries) == {
        "~/.openclaw", "~/lloyd/agent-services", "~/lloyd/.venvs",
        "~/obsidian/lloyd/SOUL.md",
    }, list(entries)
    assert all(entries.values()), "every entry must say what it is"


def test_the_write_deny_set_is_not_the_delete_roots(wtree):
    """Denying `protected_roots()` for writes would refuse every vault note and
    every file a round writes in its own worktree: those roots are `$HOME`, the
    vault and the whole lloyd tree. Pinned so nobody 'simplifies' the two sets
    into one again."""
    denied = {root for root, _ in PP.protected_write_roots()}
    assert denied, "an empty deny-set would pass this test and protect nothing"
    home = str(wtree.resolve())
    assert str((wtree / "obsidian").resolve()) not in denied, (
        "the vault is writable; one file inside it is not")
    assert str((wtree / "lloyd").resolve()) not in denied, (
        "the code tree is writable; two directories inside it are not")
    assert home not in denied


def test_write_deny_reason_matches_whole_path_components_only(wtree):
    inside = wtree / "lloyd" / "agent-services" / "supervisor" / "conf.d" / "c.conf"
    label = PP.write_deny_reason(str(inside))
    assert label == dict(PP.PROTECTED_WRITE_ROOTS)["~/lloyd/agent-services"], label
    # A sibling whose name merely starts with an entry's name is outside it.
    assert PP.write_deny_reason(str(wtree / "lloyd" / "agent-services-extra" / "r.md")) is None
    assert PP.write_deny_reason(str(wtree / "lloyd" / ".venvs-backup" / "x")) is None
    # The entry itself is denied, and so is a bare directory below it.
    assert PP.write_deny_reason(str(wtree / "lloyd" / "agent-services")) is not None
    assert PP.write_deny_reason(str(wtree / "lloyd")) is None
    assert PP.write_deny_reason("") is None


def test_a_symlink_is_judged_by_where_it_points(wtree, tmp_path):
    """Realpath-first cuts both ways: a link standing outside the set that
    points into it is refused, and one pointing out of it is allowed — the
    bytes are what the rule is about."""
    into = tmp_path / "points-in.md"
    into.symlink_to(wtree / "obsidian" / "lloyd" / "SOUL.md")
    assert PP.write_deny_reason(str(into)) is not None
    real_note = tmp_path / "note.md"
    real_note.write_text("x")
    link = wtree / "obsidian" / "lloyd" / "link-out.md"
    link.symlink_to(real_note)
    assert PP.write_deny_reason(str(link)) is None


def test_granting_lifts_the_set_and_expiring_relifts_it(wtree):
    target = str(wtree / ".openclaw" / "config.json")
    assert PP.write_deny_reason(target) is not None
    token = PP.grant_protected_writes("test: the whole set")
    try:
        assert PP.write_deny_reason(target) is None
    finally:
        PP.release_protected_writes(token)
    assert PP.write_deny_reason(target) is not None
    assert PP.protected_writes_granted() is None
    with PP.allow_protected_writes("test: scoped"):
        assert PP.write_deny_reason(target) is None
    assert PP.write_deny_reason(target) is not None


def test_a_narrowed_grant_lifts_only_what_it_names(wtree):
    soul = str(wtree / "obsidian" / "lloyd" / "SOUL.md")
    venv = str(wtree / "lloyd" / ".venvs" / "lloyd" / "bin" / "python")
    with PP.allow_protected_writes("test: one entry", paths=["~/lloyd/.venvs"]):
        assert PP.write_deny_reason(venv) is None
        assert PP.write_deny_reason(soul) is not None
    # Lifting `~/lloyd/.venvs` must not lift a sibling that shares its prefix.
    with PP.allow_protected_writes("test: prefix sibling", paths=["~/lloyd/.venvs"]):
        assert PP.write_deny_reason(str(wtree / "lloyd" / ".venvs-extra" / "f")) is None


def test_grant_requires_a_reason():
    """An authorisation nobody can name is an authorisation nobody revokes."""
    with pytest.raises(TypeError):
        PP.grant_protected_writes()  # type: ignore[call-arg]


def test_the_fs_lane_holds_no_path_literal_of_its_own():
    """The clause-3 grep, run as written: the rule lives in one module, so the
    lane that applies it may not keep a copy of any entry — a second spelling
    is how the earlier four lists diverged."""
    offenders = [line for line in FS_LANE.read_text(encoding="utf-8").splitlines()
                 if re.search(r"openclaw|agent-services|\.venvs|SOUL", line)]
    assert offenders == [], offenders


def test_the_deny_set_is_consulted_from_one_place_in_the_write_path():
    """One consult site, reached first, ahead of the `gate_on` switch. A check
    a config switch can switch off is not a location rule, and a second copy of
    it somewhere else is how two checks start disagreeing. Counted on the
    syntax tree, not by grep: the name also appears in prose, and a test that
    counts sentences is a test that fails when somebody improves a comment."""
    tree = ast.parse(FS_LANE.read_text(encoding="utf-8"))
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    gate, refusal = funcs["_gate_check"], funcs["_protected_path_refusal"]

    def calls(node):
        return [n.func.id for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]

    assert calls(refusal).count("write_deny_reason") == 1, calls(refusal)
    assert "_protected_path_refusal" in calls(gate), calls(gate)
    refusal_line = min(n.lineno for n in ast.walk(gate)
                       if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                       and n.func.id == "_protected_path_refusal")
    gate_on_line = min(n.lineno for n in ast.walk(gate)
                       if isinstance(n, ast.Attribute) and n.attr == "gate_on")
    assert refusal_line < gate_on_line, (
        "the location check must run before the session-scoped clobber gate")
    src = FS_LANE.read_text(encoding="utf-8")
    assert "PROTECTED_PATH" in src, "the refusal must carry the standard code"


def test_no_other_write_lane_consults_the_deny_set():
    """`vault_write` and `automod_vault_land` are sanctioned writers with their
    own root checks and their own validation, and the 06:00 nightly job runs on
    them; the deny-set stops at the fs lane so it cannot fail that run."""
    for rel in ("agent_mcp/vault.py", "agent_mcp/automod.py", "agent_mcp/facts.py",
                "app/harness/safety.py", "app/harness/mcp_pool.py", "agent_mcp/main.py"):
        path = REPO / rel
        assert path.exists(), rel
        src = path.read_text(encoding="utf-8")
        assert "write_deny_reason" not in src, rel
        assert "PROTECTED_WRITE_ROOTS" not in src, rel
