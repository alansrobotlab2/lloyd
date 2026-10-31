"""#3403 clause 1: the vault resolves from an override, and the flag says so.

`app.paths.VAULT_ROOT` used to be `Path.home() / "obsidian"` with no knob, which is why
no bench task in the suite can ask "did the run do the thing": the durable-write tools
are config-disabled for every trial because a trial that could call `memory_add` or
`backlog_write_task` edits the house. Every node below runs a FRESH interpreter, because
`VAULT_ROOT` and `IS_PRODUCTION_VAULT` are constants resolved once at import — and that
import-time resolution is exactly what the runner asserts on before dispatch, so a test
that monkeypatched the constant would be testing a value no trial ever reads.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_PROBE = """
import json, sys
sys.path.insert(0, {repo!r})
import app.paths as p
import app.data_root as d
print(json.dumps({{
    "paths_vault": str(p.VAULT_ROOT),
    "resolver_vault": str(d.vault_root()),
    "production_vault": str(d.PRODUCTION_VAULT),
    "is_production_vault": p.IS_PRODUCTION_VAULT,
    "is_production_data": p.IS_PRODUCTION_DATA,
}}))
"""


def probe(env_extra: dict[str, str]) -> dict:
    """One fresh interpreter's reading of the vault root and both flags."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("LLOYD_VAULT_ROOT", "LLOYD_VAULT")}
    env.update(env_extra)
    r = subprocess.run([sys.executable, "-c", _PROBE.format(repo=str(REPO))],
                       capture_output=True, text=True, env=env, timeout=90,
                       cwd=str(REPO))
    assert r.returncode == 0, f"probe failed: {r.stderr[-800:]}"
    return json.loads(r.stdout)


def _one_vault(resolved_root: str, production: str) -> bool:
    """Whether the resolver's answer and `PRODUCTION_VAULT` name the SAME vault.

    Realpath, not string comparison, because the two legitimately differ in spelling
    under a gate round: `worktree.py::ensure_round_home` points `$HOME` at
    `~/lloyd-work/<round>/home` and leaves `obsidian` a SYMLINK into the live vault
    (`HOME_LINK_SKIP` covers `lloyd` and `lloyd-data`, not `obsidian`), so with no
    override `vault_root()` answers `<round>/home/obsidian` while `production_vault()`
    answers the real `/home/<account>/obsidian` — one vault, two paths, exactly the
    case `is_production_vault` resolves rather than compares (#1786). A node that
    compared the strings failed on a gate for something true on every machine, and
    passed locally only because there the two spellings coincide.
    """
    return Path(resolved_root).resolve() == Path(production).resolve()


def test_no_override_falls_back_to_the_accounts_own_vault():
    out = probe({})
    assert _one_vault(out["paths_vault"], out["production_vault"]), (
        f"with neither override set, `VAULT_ROOT` resolved to {out['paths_vault']} "
        f"instead of the account vault {out['production_vault']}")
    assert out["is_production_vault"] is True


def test_lloyd_vault_root_redirects_the_root_and_clears_the_flag(tmp_path):
    fixture = tmp_path / "obsidian"
    fixture.mkdir()
    out = probe({"LLOYD_VAULT_ROOT": str(fixture)})
    assert Path(out["paths_vault"]) == fixture
    assert out["is_production_vault"] is False, (
        "a redirected vault still reported as production would refuse every scoped "
        "trial the runner is about to build for it")


def test_lloyd_vault_redirects_too_because_two_readers_already_read_it(tmp_path):
    """The second name is not an alias for ceremony.

    `scripts/automod/vault_round.py` and `agent_mcp/vault.py` read `LLOYD_VAULT` while
    `app.paths` read neither name: a caller that set the one `app.paths` ignored got a
    fixture from half the tree and the live vault from the other half, which is the
    failure mode the item calls the dangerous one.
    """
    fixture = tmp_path / "obsidian"
    fixture.mkdir()
    out = probe({"LLOYD_VAULT": str(fixture)})
    assert Path(out["paths_vault"]) == fixture
    assert out["is_production_vault"] is False


def test_the_older_name_wins_when_both_are_set(tmp_path):
    """Precedence is pinned, because two roots that disagree are one bug per reader.

    `LLOYD_VAULT_ROOT` is what the guardian and the backup script read; the precedence
    that already existed in `data_root.vault_root()` is kept rather than reversed, so
    nobody's watchdog starts watching a different tree than the writer writes.
    """
    old_name, new_name = tmp_path / "a", tmp_path / "b"
    old_name.mkdir()
    new_name.mkdir()
    out = probe({"LLOYD_VAULT_ROOT": str(old_name), "LLOYD_VAULT": str(new_name)})
    assert Path(out["paths_vault"]) == old_name


@pytest.mark.parametrize("name", ["LLOYD_VAULT_ROOT", "LLOYD_VAULT"])
def test_an_empty_override_falls_back_rather_than_becoming_the_cwd(name):
    """`LLOYD_VAULT=""` must not resolve to `Path("")`, which is `.`.

    A relative vault would make the trial's working directory the vault: writes land
    inside whatever tree the runner happened to `chdir` into, and `IS_PRODUCTION_VAULT`
    reads False because `.` is not `~/obsidian` — a sandbox flag that is green for the
    wrong reason is the one outcome worse than no flag at all.
    """
    out = probe({name: ""})
    assert _one_vault(out["paths_vault"], out["production_vault"]), (
        f"{name}='' resolved to {out['paths_vault']}, which is not the account vault "
        f"{out['production_vault']} — an empty override became a path instead of no "
        f"override")
    assert out["is_production_vault"] is True


def test_the_flag_is_a_predicate_on_the_resolved_root_not_on_the_env_var():
    """Pointing the override AT the live vault must still read as production.

    A flag implemented as "`LLOYD_VAULT_ROOT` is set" would answer False for that call and
    the runner would hand its tools the real vault on the strength of a variable name.
    """
    live = probe({})["production_vault"]
    out = probe({"LLOYD_VAULT": live})
    assert Path(out["paths_vault"]) == Path(live)
    assert out["is_production_vault"] is True


def test_a_home_whose_vault_is_a_symlink_into_the_house_is_not_isolated(tmp_path):
    """The flag is a realpath test, because a gate round's vault IS a symlink.

    `scripts/automod/worktree.py::ensure_round_home` points `$HOME` at
    `~/lloyd-work/<round>/home` and leaves `obsidian` a symlink to the live vault —
    `HOME_LINK_SKIP` spares `lloyd` and `lloyd-data`, not `obsidian`. A flag comparing
    path STRINGS (`VAULT_ROOT == PRODUCTION_VAULT`) answers False for that round: the
    constant reads "sandboxed" while `VAULT_ROOT` resolves straight into Alan's real
    vault, and both of its readers — the aggregator's pre-admission check and the
    runner's in-process probe — would then admit a durable-write trial whose every write
    follows the link. So this probe sets `HOME` to a directory holding exactly that
    symlink and no override at all, which is the round's own environment.
    """
    live = probe({})["production_vault"]
    round_home = tmp_path / "home"
    round_home.mkdir()
    (round_home / "obsidian").symlink_to(live)
    # No `LLOYD_DATA`: a fresh directory is not a data root without
    # `DATA_ROOT_MARKER`, and `resolve_data_root` would refuse the import for a reason
    # this node is not about. `app.data_root` derives the production root from the
    # passwd entry, so changing `HOME` moves the vault default and nothing else.
    out = probe({"HOME": str(round_home)})
    assert Path(out["paths_vault"]) == round_home / "obsidian", (
        "with no override the root still resolved off the passwd entry, so this probe "
        "never tested the symlink at all")
    assert out["is_production_vault"] is True, (
        f"`$HOME/obsidian` is a symlink to {live} and `IS_PRODUCTION_VAULT` read "
        f"{out['is_production_vault']}: a lexical comparison, which is the answer that "
        "admits a scoped trial inside a gate round")


def test_is_production_vault_is_exported_beside_its_data_twin():
    out = probe({})
    assert isinstance(out["is_production_vault"], bool)
    assert isinstance(out["is_production_data"], bool)
    assert set(out) >= {"paths_vault", "resolver_vault", "is_production_vault"}
