"""#1204: the three front-matter readers on the dashboard path parse with libyaml.

`GET /api/dashboard` re-YAML-parsed the whole board on every cold cycle — 1,141
front matters inside the request — and it did it on PyYAML's pure-Python
scanner while libyaml sat installed on the box. The measured gap on the real
board files was 1.56 s (SafeLoader) against 0.10 s (CSafeLoader), 15.1x, with
0 mismatches across 1,140 front matters.

Why the equivalence is asserted rather than assumed: the dict these readers
return is what the board steward's prompt and the unattended loop's dispatch
decisions are built from. A loader swap that silently changed one value would
be a worse trade than staying slow, so every file on the board is parsed under
both loaders here and the dicts are compared.

Two things this pins that inspecting a module attribute would not:

* the parse must be dispatched through the module's `_YamlLoader` **attribute
  at call time**. A `yaml.safe_load` left in the body would still leave the
  right `_YamlLoader` sitting in the module for a `getattr` check to find,
  while doing none of the work — so the proof here calls each reader and
  observes which loader produced the dict.
* `scripts/automod/scorecard.py:140` runs `import yaml` *inside*
  `_frontmatter`, so an edit that only added a module-level import would have
  changed nothing. Each reader is exercised on real board bytes.
"""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from app.routers import dashboard as dash
from board_presence import VAULT_ROOT_ENV, board_files_or_stop
from scripts.automod import backlog as B
from scripts.automod import scorecard as SC

BOARD_DIR = Path.home() / "obsidian" / "backlog"

# The closing delimiter, anchored, same rule as `dashboard._frontmatter`: an
# unanchored `---` split also fires on a `---` in prose and truncates the block
# at the wrong place.
_FM_END_RE = re.compile(r"^---[ \t]*$", re.M)

# Files walked by the per-reader equivalence pass. Every file on the board gets
# the direct both-loaders comparison below; this sample additionally runs each
# *reader* under both loaders, which is the 3x-multiplied one.
SAMPLE = 120


def _board_files() -> list[Path]:
    """Every item file on the live board, in a stable order."""
    return sorted(BOARD_DIR.glob("*.md")) if BOARD_DIR.is_dir() else []


def _readers():
    """(label, callable taking a Path, owning module) for the three named sites.

    `_split_frontmatter` takes the file's text, not a path, so its wrapper does
    the read — the caller that matters (`load_item`) reads the file too.
    """
    return [
        ("dashboard._frontmatter (app/routers/dashboard.py:118)", dash._frontmatter, dash),
        ("scorecard._frontmatter (scripts/automod/scorecard.py:150)", SC._frontmatter, SC),
        ("backlog._split_frontmatter (scripts/automod/backlog.py:886)",
         lambda p: B._split_frontmatter(p.read_text(encoding="utf-8"))[0], B),
    ]


def _recording_loader(base):
    """A `base` subclass that counts how many times yaml.load() built one.

    Counting construction is the observation: `yaml.load(text, Loader=X)`
    instantiates X once per parse, so a nonzero count means the parse went
    through X and not through some other loader the call site reached for.
    """
    class Recorder(base):
        seen = 0

        def __init__(self, *args, **kwargs):
            type(self).seen += 1
            super().__init__(*args, **kwargs)

    Recorder.__name__ = f"Recording{base.__name__}"
    return Recorder


def _fm_text(text: str) -> str:
    """The front-matter block of a markdown file, or "" when there is none."""
    if not text.startswith("---"):
        return ""
    opening = text.find("\n")
    end = _FM_END_RE.search(text, 3)
    if opening < 0 or end is None or opening > end.start():
        return ""
    return text[opening + 1:end.start()]


def _one_real_item(tmp_path: Path) -> Path:
    """A copy of a real board item, in a path the test owns.

    A copy, not the live file, because the readers are read-only here but the
    board is somebody else's working tree.

    There is deliberately no written-down fallback. A four-field fixture that I
    authored would let clause 1's proof report "the reader parsed through the
    C loader" while parsing bytes nobody on the board wrote — the exact
    shape-mismatch failure the rest of this file exists to close. So when the
    board yields no file with front matter, `board_files_or_stop` decides:
    **fail** if the vault root is there (an emptied or moved board is the state
    this test is supposed to notice), **skip** naming the path if the vault is
    absent entirely, in which case clause 1 is stated as unpinned rather than
    proven on an invented file.
    """
    board_files_or_stop(what="per-reader loader proof")
    for path in _board_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if _fm_text(text).strip():
            out = tmp_path / "999999-loader-proof.md"
            out.write_text(text, encoding="utf-8")
            return out
    pytest.fail(
        f"{BOARD_DIR} has {len(_board_files())} files and not one of them has a "
        f"parseable front-matter block, so there is no real byte to prove the "
        f"loader on. Failing rather than writing my own fixture: a file I "
        f"invented proves the reader can parse a file I invented."
    )


# ── clause 1: which loader produced the dict ─────────────────────────────

def test_each_named_reader_parses_through_its_module_loader_attribute(tmp_path, monkeypatch):
    """Each reader is called on real board bytes and observed.

    The recorder counts loader constructions, so a reader that still called
    `yaml.safe_load` would return a correct dict with `seen == 0` and fail
    here — which is the whole point: the dict is not the evidence, the
    loader that made it is.
    """
    sample = _one_real_item(tmp_path)
    for label, call, module in _readers():
        assert hasattr(module, "_YamlLoader"), (
            f"{label}: no module-level _YamlLoader to dispatch through, so the "
            f"call site cannot be proven — the shape to copy is "
            f"app/kg_store.py:48")
        recorder = _recording_loader(module._YamlLoader)
        monkeypatch.setattr(module, "_YamlLoader", recorder)
        before = recorder.seen
        produced = call(sample)
        assert isinstance(produced, dict), f"{label}: reader returned {type(produced)}"
        assert produced, f"{label}: reader returned no front matter for {sample}"
        assert recorder.seen > before, (
            f"{label}: the parse did NOT go through the module's _YamlLoader "
            f"({module._YamlLoader.__name__}) — recorder built "
            f"{recorder.seen - before} loaders. A `yaml.safe_load` still in the "
            f"body parses on PyYAML's pure-Python scanner, which is the 15.1x "
            f"cost #1204 is about.")


@pytest.mark.skipif(getattr(yaml, "_yaml", None) is None,
                    reason="libyaml is not importable on this box, so CSafeLoader "
                           "legitimately is not the loader — the fallback test below "
                           "covers that case")
def test_all_three_modules_select_the_c_loader_when_libyaml_is_present():
    """The fallback branch must not be silently swallowing the C loader here.

    Without this, a typo that always took the `except ImportError` arm would
    leave every other test in this file green (they patch whatever loader the
    module holds) while the box kept parsing on the slow one.
    """
    for label, _call, module in _readers():
        assert module._YamlLoader is yaml.CSafeLoader, (
            f"{label}: _YamlLoader is {module._YamlLoader.__name__}, not "
            f"CSafeLoader, although libyaml is importable")


def test_the_pure_python_fallback_is_selected_when_libyaml_is_absent():
    """The `except ImportError` arm is real, in a box that has no libyaml.

    Run in a subprocess: deleting `yaml.CSafeLoader` makes
    `from yaml import CSafeLoader` raise ImportError against the already-loaded
    module, which is exactly the state of a PyYAML built without the C
    extension. In-process monkeypatching would leave the live app parsing on
    the pure-Python loader for the rest of the session.
    """
    code = (
        "import yaml\n"
        "del yaml.CSafeLoader\n"
        "from app.routers import dashboard as dash\n"
        "from scripts.automod import backlog as B, scorecard as SC\n"
        "print(dash._YamlLoader.__name__, B._YamlLoader.__name__,"
        " SC._YamlLoader.__name__)\n"
    )
    # Pin the automod state dir to the REAL one. The automod gate's tests rung
    # runs pytest with `LLOYD_AUTOMOD_STATE` pointed at the round's scratch dir
    # (`scripts/automod/gate.py:384`), which has no `promotions.jsonl` in it. This
    # subprocess inherits that variable; `scripts.automod.state` resolves
    # `LEDGER_PATH` from it at import, and `backlog` and `scorecard` hand it out as
    # the default (`B.LEDGER_DEFAULT()`, `SC.compute(ledger=None)`). Left inherited,
    # those two readers would resolve the ledger to a file that does not exist,
    # `read_events` would return `[]`, `total` would be 0, and the "a real file was
    # parsed, not a written fixture" assert below would fail for a reason that has
    # nothing to do with the loader. `dashboard._automod` reads `S.LEDGER_PATH`
    # fresh at each call (`app/routers/dashboard.py:721`), so pinning is exactly the
    # path the endpoint takes: the subprocess then parses the same ledger the
    # dashboard does.
    env = {**os.environ, "LLOYD_AUTOMOD_STATE": str(
        Path.home() / ".local" / "state" / "lloyd-automod")}
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(Path.cwd()),
                          capture_output=True, text=True, timeout=180, env=env)
    assert proc.returncode == 0, f"fallback probe failed: {proc.stderr[-800:]}"
    got = proc.stdout.strip().split()[-3:]
    assert got == ["SafeLoader", "SafeLoader", "SafeLoader"], (
        f"with libyaml removed the three readers must fall back to the "
        f"pure-Python SafeLoader; got {got}. A reader whose import arm raises "
        f"something other than ImportError (AttributeError, a missing module) "
        f"takes the board down with the dashboard.")


# ── clause 2: the swap is provably behaviour-preserving ──────────────────

def test_both_loaders_agree_on_every_front_matter_on_the_board():
    """Parse every item file twice — CSafeLoader and SafeLoader — and compare.

    The file count is asserted, not just printed: a glob that matched nothing
    would make the loop body run zero times and the test report a clean board.
    Baseline at triage (2026-09-16): 1,142 files, 1,140 front matters, 0
    mismatches.

    Board presence is `board_files_or_stop`, not a `pytest.skip` when the glob
    comes up empty — the flag's point. The skip was wrong because the board is
    *this item's own tree*: with the vault root present and the board missing or
    emptied, something has happened to the corpus the clause certifies, and
    `skipped` is where that goes hidden. The helper fails on that state and
    skips only when the vault root itself is absent (a machine with no vault).
    `test_a_moved_or_emptied_board_fails_instead_of_skipping` pins all three
    outcomes on synthetic directories.
    """
    files = board_files_or_stop(what="loader-equivalence sweep")
    print(f"board dir: {BOARD_DIR}  files walked: {len(files)}")

    mismatched: list[str] = []
    parsed = 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:  # unreadable is a finding, not a skip
            mismatched.append(f"{path.name}: unreadable ({exc})")
            continue
        block = _fm_text(text)
        if not block.strip():
            continue
        parsed += 1
        try:
            under_c = yaml.load(block, Loader=yaml.CSafeLoader)
        except Exception as exc:
            mismatched.append(f"{path.name}: CSafeLoader raised {exc!r}")
            continue
        try:
            under_py = yaml.load(block, Loader=yaml.SafeLoader)
        except Exception as exc:
            mismatched.append(f"{path.name}: SafeLoader raised {exc!r} "
                              f"where CSafeLoader returned {type(under_c)}")
            continue
        if under_c != under_py:
            keys = [k for k in set(under_c or {}) | set(under_py or {})
                    if (under_c or {}).get(k) != (under_py or {}).get(k)]
            mismatched.append(f"{path.name}: differs on {keys}")

    print(f"front matters compared under both loaders: {parsed}")
    assert parsed > 0, f"no file under {BOARD_DIR} carried a parseable front matter"
    assert not mismatched, (
        f"{len(mismatched)} of {parsed} front matters parse differently under "
        f"the two loaders — the swap would change what the board steward's "
        f"prompt and the dispatch loop read. First: {mismatched[0]}"
    )


@pytest.mark.skipif(getattr(yaml, "_yaml", None) is None,
                    reason="no libyaml on this box: both arms would be SafeLoader "
                           "and the comparison could not fail")
def test_each_reader_returns_the_same_dict_under_either_loader(tmp_path):
    """Per-reader equivalence, including each reader's own slicing.

    The three readers slice the block differently — `dashboard` reads in
    4 KiB chunks to a 64 KiB bound, `scorecard` slices `text[3:3 + m.start()]`,
    `backlog` splits on `---\\n` — so an equivalence claim about the loader
    alone does not cover them. A sample of the board (in name order, so the
    set is stable) is run through each reader twice.
    """
    files = _board_files()
    assert files, f"{BOARD_DIR} matched nothing; the sample below is empty"
    sample = files[:SAMPLE]
    assert len(sample) > 0
    for label, call, module in _readers():
        differs = []
        for path in sample:
            c_loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
            monkey_c, monkey_py = module._YamlLoader, yaml.SafeLoader
            try:
                module._YamlLoader = c_loader
                under_c = call(path)
                module._YamlLoader = monkey_py
                under_py = call(path)
            finally:
                module._YamlLoader = monkey_c
            if under_c != under_py:
                differs.append(path.name)
        assert not differs, (
            f"{label}: {len(differs)} of {len(sample)} sampled items parse "
            f"differently under the two loaders, first {differs[:5]}"
        )


def test_reload_does_not_lose_the_loader_binding():
    """`importlib.reload` of each reader module still leaves `_YamlLoader` bound.

    What this does **not** prove, stated here so nobody cites it for that: it
    is not a fresh-interpreter import. `importlib.reload` re-executes the module
    body in the *existing* namespace, so a name that survived the first import
    would still be there even if the `try: from yaml import CSafeLoader` block
    were deleted from the source — the old object is simply rebound to the same
    slot. A previous cut of this test monkeypatched a fake `yaml` into
    `sys.modules` before the reload and read the result as "a fresh import still
    binds the loader"; that read was wrong for the same reason.

    The load-bearing proof of a real fresh import is
    `test_the_pure_python_fallback_is_selected_when_libyaml_is_absent`, which
    runs `subprocess.run([sys.executable, "-c", ...])` in a genuinely new
    interpreter and asserts which class each module bound. This test is the
    cheaper reload-durability check, and that is the whole of its claim.
    """
    for module in (dash, SC, B):
        reloaded = importlib.reload(module)
        assert hasattr(reloaded, "_YamlLoader"), (
            f"{module.__name__} lost its `_YamlLoader` binding across "
            f"importlib.reload. Reload re-executes the module body, so this "
            f"fires only when the import block itself now raises — which on a "
            f"box with libyaml present means the block is broken, not that a "
            f"fallback was taken.")
    dash._cache.clear()


# ── the presence helper itself ────────────────────────────────────────────
#
# Clause 2's denominator guard and its skip policy both live in
# `tests/board_presence.py`. A guard nobody has tested is a guard that reports
# what you expected, so each of the helper's three outcomes is driven here
# against synthetic directories — including the one that must NOT be a skip.

def test_a_moved_or_emptied_board_fails_instead_of_skipping(tmp_path, monkeypatch):
    """The flagged state, pinned instead of left to the next reader.

    Two shapes where a skip would hide the thing this item is about: the board
    directory is gone while the vault root is there, and the board exists but
    holds no item file. Both are a broken *corpus*, which is what a denominator
    guard exists to report. The third shape — no vault directory at all — is a
    property of the machine, and only that one skips.
    """
    # `pytest.raises(pytest.fail.Exception)`, not `AssertionError`: the helper
    # stops the run with `pytest.fail`, which raises `Failed` — a sibling of
    # `Skipped`, not of `AssertionError`. Asserting the wrong exception type here
    # would fail this test while the helper was behaving correctly, which is the
    # same class of mistake the helper exists to prevent.
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv(VAULT_ROOT_ENV, str(vault))
    absent_board = vault / "backlog"  # deliberately never created

    with pytest.raises(pytest.fail.Exception) as moved:
        board_files_or_stop(what="moved-board proof", board=absent_board)
    assert "is the board this item measures" in str(moved.value), str(moved.value)
    assert "moved-board proof" in str(moved.value), str(moved.value)

    emptied = vault / "backlog"
    emptied.mkdir()
    with pytest.raises(pytest.fail.Exception) as empty:
        board_files_or_stop(what="emptied-board proof", board=emptied)
    assert "exactly the state the emptied-board proof exists to notice" \
        in str(empty.value), str(empty.value)

    monkeypatch.setenv(VAULT_ROOT_ENV, str(tmp_path / "no-such-vault"))
    with pytest.raises(pytest.skip.Exception) as skipped:
        board_files_or_stop(what="no-vault proof", board=absent_board)
    # The phrase the helper must carry: a skip is not a pass of the clause. It
    # is the sentence that makes `skipped` honest in the report.
    assert "NOT pinned by this run" in str(skipped.value), str(skipped.value)


def test_the_helper_counts_the_real_board_and_the_numeric_subset(tmp_path, monkeypatch):
    """`numeric_names` is the difference between 1,143 and 1,142, so it is tested.

    The cold-cycle test counts only `NN-*.md`; the equivalence sweep counts every
    `.md`, because a README with front matter is a front matter the readers will
    parse. Both are ~1,14x, so a swapped flag would not show up as a wrong order
    of magnitude — it would show up as this item's own denominator drifting by one
    and nobody knowing which one was the contract.
    """
    vault = tmp_path / "vault"
    board = vault / "backlog"
    board.mkdir(parents=True)
    (board / "11.md").write_text("---\nstatus: done\n---\n\nbody\n", encoding="utf-8")
    (board / "999.md").write_text("---\nstatus: done\n---\n\nbody\n", encoding="utf-8")
    (board / "1204-dashboard.md").write_text("---\nstatus: done\n---\n\nbody\n", encoding="utf-8")
    (board / "README.md").write_text("---\ntype: note\n---\n\nnotes\n", encoding="utf-8")
    monkeypatch.setenv(VAULT_ROOT_ENV, str(vault))

    all_files = board_files_or_stop(what="counting proof", board=board)
    numbered = board_files_or_stop(what="counting proof", board=board,
                                  numeric_names=True)
    assert [p.name for p in all_files] == [
        "11.md", "1204-dashboard.md", "999.md", "README.md"]
    assert [p.name for p in numbered] == ["11.md", "1204-dashboard.md", "999.md"], (
        "numeric_names is the first-token-is-a-digit rule: `11.md` is the 11th "
        "item and `999.md` the 999th (the board passes 999, and the old "
        "three-digit regex would have dropped it). README.md is not an item.")
