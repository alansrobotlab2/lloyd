"""#3688 clause 4: the admission CLI runs, and prints oracle GREEN and empty-run RED.

Two crashes lived here, both pinned from the OUTSIDE by a subprocess, because both were
properties of *how the file is started* and not of the functions inside it:

1. `bench_side_effects.py:32` imported `scripts.autoresearch.bench_fixture` at module top,
   so `python scripts/autoresearch/bench_side_effects.py --dir …` died with
   `ModuleNotFoundError: No module named 'scripts'`.
2. The `--data-root` default called `resolve_data_root()` positionally; the function is
   keyword-only with five required arguments (`app/data_root.py:173`), so the spelling that
   got past (1) died with `TypeError: resolve_data_root() missing 5 required keyword-only
   arguments`. The script-facing entry point is `resolve_data_root_for_tree`.

Calling the functions from a test would pass with neither bug present, which is how "3 of 3
admitted" got reported once without the live corpus ever being admitted.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH_DIR = Path.home() / "obsidian" / "lloyd" / "bench"
SCRIPT = REPO / "scripts" / "autoresearch" / "bench_side_effects.py"
PY = Path(sys.executable)
if not PY.exists():
    PY = REPO / ".venvs" / "lloyd" / "bin" / "python"


def _cli(*argv: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([str(PY), *argv], cwd=str(cwd or REPO), capture_output=True,
                          text=True, timeout=900)


def test_the_admission_cli_runs_from_its_script_path_without_the_import_crash(tmp_path):
    """Bug 1: the `python scripts/autoresearch/bench_side_effects.py` spelling."""
    proc = _cli(str(SCRIPT), "--dir", str(tmp_path / "nothing-here"))
    combined = proc.stdout + proc.stderr
    assert "No module named 'scripts'" not in combined, combined[-1200:]
    assert proc.returncode == 0, combined[-1200:]


def test_the_data_root_default_resolves_without_the_keyword_only_typeerror(tmp_path):
    """Bug 2: `--data-root` has a default, and reaching it used to raise TypeError."""
    proc = _cli(str(SCRIPT), "--dir", str(tmp_path / "nothing-here"))
    combined = proc.stdout + proc.stderr
    assert "missing 5 required keyword-only arguments" not in combined, combined[-1200:]
    assert proc.returncode == 0, combined[-1200:]


def test_the_cli_names_the_writes_card_it_refuses(tmp_path):
    """The point of the CLI: a card without a scope cannot be admitted, and says so."""
    cards = tmp_path / "bench"
    cards.mkdir()
    (cards / "bench_x_no_scope.md").write_text(
        "---\nid: bench_x_no_scope\n"
        "oracle: scripts/autoresearch/bench_oracles/oracle_durable_write_vault_note.sh\n"
        "side_effect_checks:\n"
        "- scripts/autoresearch/bench_oracles/check_durable_write_vault_note.sh\n---\nbody\n",
        encoding="utf-8")
    proc = _cli(str(SCRIPT), "--dir", str(cards), "--json")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report[0]["admitted"] is False
    # `build_fixture` raises `FixtureRefused` for an undeclared store, and the CLI turns it
    # into a named refusal instead of a traceback: the crash and the refusal both exit 1, and
    # only one of them tells the author which key to add.
    assert "fixture refused" in report[0]["reason"], report[0]
    assert "writes:" in report[0]["reason"], report[0]


def test_a_card_with_a_missing_verifier_is_refused(tmp_path):
    cards = tmp_path / "bench"
    cards.mkdir()
    (cards / "bench_y.md").write_text(
        "---\nid: bench_y\nwrites: [vault]\n"
        "oracle: scripts/autoresearch/bench_oracles/oracle_durable_write_vault_note.sh\n"
        "side_effect_checks:\n- eval/bench/gone.sh\n---\nbody\n",
        encoding="utf-8")
    proc = _cli(str(SCRIPT), "--dir", str(cards), "--json")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report[0]["admitted"] is False
    # A verifier that cannot be found cannot go green on the oracle arm, so the card is
    # refused with the oracle leg RED rather than admitted on an empty verdict.
    assert report[0]["oracle_passed"] is False, report[0]
    assert report[0]["reason"].startswith("oracle RED"), report[0]


def test_a_card_with_a_missing_oracle_is_refused(tmp_path):
    """Clause 4's `naming an oracle script file that exists`, from the admission side."""
    cards = tmp_path / "bench"
    cards.mkdir()
    (cards / "bench_z.md").write_text(
        "---\nid: bench_z\nwrites: [vault]\n"
        "oracle: scripts/autoresearch/bench_oracles/no_such_oracle.sh\n"
        "side_effect_checks:\n"
        "- scripts/autoresearch/bench_oracles/check_durable_write_vault_note.sh\n---\nbody\n",
        encoding="utf-8")
    proc = _cli(str(SCRIPT), "--dir", str(cards), "--json")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report[0]["admitted"] is False
    assert "oracle script not found" in report[0]["reason"], report[0]


def test_the_empty_run_leg_is_red_when_nothing_wrote(tmp_path):
    """The half that makes a verifier mean something: a fixture nobody wrote in must refuse."""
    oracles = REPO / "scripts" / "autoresearch" / "bench_oracles"
    assert _cli(str(oracles / "check_durable_write_vault_note.sh"),
                cwd=tmp_path).returncode == 1
    assert _cli(str(oracles / "check_durable_write_kg_row.sh"), cwd=tmp_path).returncode == 1
    assert _cli(str(oracles / "check_durable_write_backlog_card.sh"),
                cwd=tmp_path).returncode == 1


def test_the_oracle_leg_is_green_when_the_write_is_performed(tmp_path):
    """The other half: something must be able to satisfy the verifier, or the reward is 0 forever."""
    vault = tmp_path / "obsidian"
    (vault / "lloyd").mkdir(parents=True)
    # A real vault always has a loaded index, and the baseline commit below needs at least one
    # tracked file to exist: `git commit` on a tree with only empty directories exits 1 with
    # "no changes added to commit", which is why this seed is a file and not a `mkdir`.
    (vault / "lloyd" / "MEMORY.md").write_text("# seed\n", encoding="utf-8")
    data = tmp_path / "lloyd-data"
    (data / "_pipeline" / "vault-derived").mkdir(parents=True)
    (data / "backlog").mkdir(parents=True)
    oracles = REPO / "scripts" / "autoresearch" / "bench_oracles"
    env = {"BENCH_FIXTURE_VAULT": str(vault), "BENCH_FIXTURE_DATA_ROOT": str(data)}
    # `build_fixture` gives every copy its own `git init` plus a baseline commit of what it
    # copied (`bench_fixture.py:454-458`), and `check_durable_write_backlog_card.sh` reads that
    # baseline to tell a trial's card from the 3,788 items the copy already had — so a
    # hand-made tree has to carry one too, or this node would be grading a tree that cannot
    # exist in production. Committed BEFORE the plant, so the planted card is the diff.
    for step in (["git", "init", "-q"],
                 ["git", "-c", "user.email=bench@lloyd", "-c", "user.name=bench",
                  "add", "-A"],
                 ["git", "-c", "user.email=bench@lloyd", "-c", "user.name=bench",
                  "commit", "-qm", "baseline"]):
        subprocess.run(step, cwd=str(vault), capture_output=True, text=True,
                       timeout=120, check=True)
    for name in ("oracle_durable_write_vault_note.sh", "oracle_durable_write_kg_row.sh",
                 "oracle_durable_write_backlog_card.sh"):
        r = subprocess.run([str(oracles / name)], capture_output=True, text=True,
                           env={**_base_env(), **env}, timeout=120)
        assert r.returncode == 0, name + r.stderr
    for name in ("check_durable_write_vault_note.sh", "check_durable_write_kg_row.sh",
                 "check_durable_write_backlog_card.sh"):
        r = subprocess.run([str(oracles / name)], capture_output=True, text=True,
                           env={**_base_env(), **env}, timeout=120)
        assert r.returncode == 0, name + " said: " + r.stderr


def _base_env() -> dict[str, str]:
    import os
    keep = {k: v for k, v in os.environ.items()
            if k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER")}
    return keep


#: The token each card hands the model and each card's verifier greps for. Coined nonces,
#: not descriptions: the empty-run leg copies the WHOLE vault, so these cards' own text is
#: inside the fixture it grades, and a descriptive sentinel that a nightly topic note could
#: quote would satisfy a verifier for the life of the vault.
LIVE_SENTINELS = {
    "bench_029_durable_write_vault_note": "ZEDQUILL-3688-VANE",
    "bench_030_durable_write_kg_row": "QIVRE-3688-ROW",
    "bench_031_durable_write_backlog_card": "QIVRE-3688-CARD",
}


def test_the_live_corpus_admits_all_three_cards_from_the_repo_root_only():
    """Clause 4's acceptance check: `python -m scripts.autoresearch.bench_side_effects
    --dir ~/obsidian/lloyd/bench` prints oracle GREEN and empty-run RED for all 3 cards.

    No `--vault` and no `--data-root`, which is the spelling the item names and the one a
    person types. It is also the only test here that resolves the scripts the way a card
    does: the cards name `scripts/autoresearch/bench_oracles/...`, a path that exists in the
    repo and not in the vault, so a `--dir`-relative resolution would report all three
    missing and the CLI would exit 1 printing nothing about the two bugs it was written for.

    The `empty run red` half is the load-bearing one, and this node is a trip wire rather
    than a summary. That leg builds a real fixture over the live vault, which now holds
    these three cards and their sentinels under `lloyd/bench/`; a verifier whose grep reached
    the corpus would read GREEN off the card's own body and this node would fail. That is not
    hypothetical -- the first `check_durable_write_vault_note.sh` grepped `lloyd/`, reproduced
    green against a copy nothing had written with the card's body as its evidence, and the fix
    was to aim it at `lloyd/memory/`, the directory the lane actually writes.
    """
    proc = _cli("-m", "scripts.autoresearch.bench_side_effects", "--dir", str(BENCH_DIR))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = {ln.split(":", 1)[0]: ln for ln in proc.stdout.splitlines() if ": " in ln}
    for card, token in LIVE_SENTINELS.items():
        assert card in lines, f"{card} missing from the CLI output: {proc.stdout}"
        assert "ADMITTED" in lines[card], lines[card]
        assert "oracle green" in lines[card] and "empty run red" in lines[card], lines[card]
        # Not vacuous: the prompt is what hands the model the string the verifier greps for.
        card_text = (BENCH_DIR / f"{card}.md").read_text(encoding="utf-8")
        assert token in card_text, f"{card} does not state {token}"
