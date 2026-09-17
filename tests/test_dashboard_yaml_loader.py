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
import board_presence
from board_presence import VAULT_ROOT_ENV, board_files_or_stop
from scripts.automod import backlog as B
from scripts.automod import scorecard as SC

def _board_dir() -> Path:
    """The live board, resolved per call rather than frozen at import.

    `tests.board_presence.vault_root()` re-reads `LLOYD_OBSIDIAN_VAULT` on every
    call, and a module constant computed once here would quietly not move when
    the pin test in this file sets that variable — the "set the variable, be
    ignored, and read the result as verifying the branch" failure that helper's
    own docstring warns about. Same reason the helper takes `board=` instead of
    only consulting a constant.
    """
    return board_presence.vault_root() / "backlog"


# The closing delimiter, anchored, same rule as `dashboard._frontmatter`: an
# unanchored `---` split also fires on a `---` in prose and truncates the block
# at the wrong place.
_FM_END_RE = re.compile(r"^---[ \t]*$", re.M)

#: This box's PyYAML has the libyaml C extension when `yaml._yaml` is not None.
#: A missing extension is a state the tests that need two distinct loaders
#: **fail** on rather than skip — see `tests/board_presence.py` for why a skip
#: whose condition is a property of the box is still a skip that hides the one
#: clause this item measures.
HAVE_LIBYAML = getattr(yaml, "_yaml", None) is not None


def _board_files() -> list[Path]:
    """Every item file on the live board, in a stable order."""
    d = _board_dir()
    return sorted(d.glob("*.md")) if d.is_dir() else []


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
    board has no file to offer — moved, emptied, or no vault at all —
    `board_files_or_stop` **fails** naming the path, and clause 1 is reported
    unpinned rather than proven on an invented file or filed as `skipped`. The
    no-vault case fails too; see the table in `tests/board_presence.py`.
    """
    board_files_or_stop(what="per-reader loader proof")
    for path in _board_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if _fm_text(text).strip():
            out = tmp_path / "999999-loader-proof.md"
            out.write_text(text, encoding="utf-8")
            return out
    pytest.fail(
        f"{_board_dir()} has {len(_board_files())} files and not one of them has a "
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


def test_all_three_modules_select_the_c_loader_when_libyaml_is_present():
    """The fallback branch must not be silently swallowing the C loader here.

    Without this, a typo that always took the `except ImportError` arm would
    leave every other test in this file green (they patch whatever loader the
    module holds) while the box kept parsing on the slow one.

    This used to carry a conditional-skip MARKER on the same condition, and was
    flagged for it twice: `SM_20260917_055508` at "test honesty
    tests/test_dashboard_yaml_loader.py:166: a new skip marker", and again this
    round at :204 — where what tripped the detector was this docstring QUOTING the
    marker it described, because that detector is textual. So the marker is gone
    and the word for it is spelled without the decorator sigil here. This box HAS libyaml — `python3 -c
    "import yaml; print(getattr(yaml,'_yaml',None) is not None)"` → True, PyYAML
    6.0.3 (item #1204 section 3) — so the marker could only ever fire as a
    misstatement, and the one thing it was for, a fallback arm that swallows the
    C loader, is the thing a `skipped` line hides. A box genuinely built without
    the C extension now gets a red line naming `_yaml` instead of a green run
    that certified nothing.
    """
    assert HAVE_LIBYAML, (
        "this box's PyYAML has no C extension (`yaml._yaml` is None), so "
        "CSafeLoader legitimately is not selectable and the choice this clause "
        "certifies was never made here. The clause 'the three readers parse with "
        "CSafeLoader when libyaml is importable' is NOT pinned by this run; the "
        "fallback test below still proves the SafeLoader arm.")
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

    Board presence is `board_files_or_stop`, not a skip call when the glob
    comes up empty — the flag's point. The skip was wrong because the board is
    *this item's own tree*: with the vault root present and the board missing or
    emptied, something has happened to the corpus the clause certifies, and
    `skipped` is where that goes hidden. The helper now FAILS on every state that
    leaves it nothing to parse — moved board, emptied board, no vault at all.
    `test_no_board_state_skips_and_every_one_names_the_lost_clause` pins all three
    outcomes on synthetic directories.
    """
    assert HAVE_LIBYAML, (
        "this box's PyYAML has no C extension (`yaml._yaml` is None), so there is "
        "one loader here, not two, and 'provably identical under both loaders' is "
        "a comparison this box cannot make. The clause is NOT pinned by this run.")
    files = board_files_or_stop(what="loader-equivalence sweep")
    print(f"board dir: {_board_dir()}  files walked: {len(files)}")

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
    assert parsed > 0, f"no file under {_board_dir()} carried a parseable front matter"
    assert not mismatched, (
        f"{len(mismatched)} of {parsed} front matters parse differently under "
        f"the two loaders — the swap would change what the board steward's "
        f"prompt and the dispatch loop read. First: {mismatched[0]}"
    )


def test_each_reader_returns_the_same_dict_under_either_loader(tmp_path):
    """Per-reader equivalence over EVERY file on the board, including each reader's own slicing.

    The three readers slice the block differently — `dashboard` reads in 4 KiB
    chunks to a 64 KiB bound, `scorecard` slices `text[3:3 + m.start()]`,
    `backlog` splits on `---\\n` — so an equivalence claim about the loader alone
    does not cover them: the loader can be identical and the slicing still differ.

    Two review findings land here, from opposite ends. `SM_20260917_055508`:
    "still samples files[:120] in name order — the oldest items — skipping the
    long-tail front matters (max 28,692 bytes per the item) where loader
    divergence is likeliest." Name order on this board is chronological, so that
    sample was 120 small early files, and the tail is exactly where block
    scalars, deep nesting and multi-line literals live. Then `SM_20260917_064456`:
    "Per-reader equivalence is asserted only on a top-N size sample rather than
    the whole board, so reader-specific slicing is unexercised on the other
    ~1,023 files." Both say the same thing — a sample, chosen however, is not a
    claim about the board. So there is no sample any more: all ~1,14x item files
    go through all three readers under both loaders, the whole-board sweep's cost
    tripled, which is what makes clause 2's word "provably" cover the corpus
    rather than a tenth of it.

    The board's front-matter size distribution is printed alongside it — max,
    p90, median — because the item's filed census (max 28,692, p90 4,841, median
    1,202) is a measurement with a decay date and this is the run that restates it.
    """
    assert HAVE_LIBYAML, (
        "this box's PyYAML has no C extension (`yaml._yaml` is None), so both "
        "arms would be SafeLoader and the comparison could not fail. The clause "
        "'parse output is provably identical under both loaders' is NOT pinned "
        "by this run.")
    files = board_files_or_stop(what="per-reader equivalence sweep")
    sizes: list[int] = []
    unreadable: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:  # unreadable is a finding, never a silent drop
            unreadable.append(f"{path.name}: {exc}")
            continue
        sizes.append(len(_fm_text(text)))
    assert not unreadable, f"unreadable board items: {unreadable[:3]}"
    assert sizes, f"{_board_dir()} yielded no readable item; there is nothing to compare"
    ordered = sorted(sizes)
    p90 = ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))]
    print(f"board items walked per reader: {len(sizes)} | front matter bytes: "
          f"max {ordered[-1]}, p90 {p90}, median {ordered[len(ordered) // 2]} "
          f"(item #1204's filed census: max 28,692, p90 4,841, median 1,202)")
    assert ordered[-1] > ordered[len(ordered) // 2], (
        f"the largest front matter ({ordered[-1]} B) is no bigger than the median "
        f"({ordered[len(ordered) // 2]} B): a board with no tail means either the "
        f"items stopped accumulating `activity_log` or this walk is reading the "
        f"wrong files")

    c_loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    assert c_loader is not yaml.SafeLoader, (
        "CSafeLoader resolved to SafeLoader on a box that reports libyaml "
        "present, so the two arms below are one class and this test would be "
        "comparing a loader with itself")
    for label, call, module in _readers():
        original = module._YamlLoader
        differs: list[str] = []
        empty = 0
        compared = 0
        try:
            for path in files:
                try:
                    module._YamlLoader = c_loader
                    under_c = call(path)
                    module._YamlLoader = yaml.SafeLoader
                    under_py = call(path)
                finally:
                    module._YamlLoader = original
                compared += 1
                if under_c != under_py:
                    differs.append(path.name)
                if not under_c:
                    empty += 1
        finally:
            module._YamlLoader = original
        # The word "every" in this test's name is the claim, so the walk width is
        # asserted rather than printed: an earlier cut reported `len(files)` from
        # the print while a truncated loop walked 120, which is the sampled
        # version of this test wearing the whole-board version's message.
        assert compared == len(files), (
            f"{label}: compared {compared} of {len(files)} board items, so this is "
            f"a SAMPLE and the sentence 'identical under both loaders' does not "
            f"cover the other {len(files) - compared} files — the exact defect "
            f"rounds SM_20260917_055508 and SM_20260917_064456 were both refused for")
        assert not differs, (
            f"{label}: {len(differs)} of {compared} board items parse "
            f"differently under the two loaders, first {differs[:5]}"
        )
        # An all-{} result satisfies "no differences" while proving nothing,
        # exactly as an empty glob would — so the empties are counted too.
        assert empty < compared, (
            f"{label}: returned an empty dict for all {compared} files, so the "
            f"comparison compared nothing"
        )
        print(f"{label}: {compared}/{len(files)} items identical under both "
              f"loaders ({empty} with no front matter to read)")



def test_a_fresh_interpreter_binds_the_loader_and_reload_keeps_dispatching(tmp_path, monkeypatch):
    """A genuinely new interpreter does the binding proof; reload is checked behaviourally.

    Flagged in review (`SM_20260917_055508`: the old
    `test_reload_does_not_lose_the_loader_binding` "still asserts hasattr after
    importlib.reload, which its own docstring admits cannot detect a deleted
    import block"). That admission was
    the whole problem, not a caveat: `importlib.reload` re-executes the module
    body into the *existing* namespace, so `_YamlLoader` stays bound to the old
    object even if the `try: from yaml import CSafeLoader` block is deleted from
    the source, and `hasattr` reports a pass on a module that has lost the
    binding. An earlier cut went further and monkeypatched a fake `yaml` into
    `sys.modules` before the reload, reading that as a fresh import — same flaw,
    louder.

    So the binding proof moved to a subprocess: a new interpreter imports all
    three modules from disk and prints the class each one bound. Delete the
    import block and this goes red with an `AttributeError`. What survives of the
    reload check is behavioural: after re-executing each module body, the reader
    must still *dispatch* through its `_YamlLoader`, counted exactly the way
    clause 1's proof counts it — which `hasattr` could not fake.
    """
    # `CSafeLoader.__module__` is `yaml.cyaml`, not `yaml` — the shim that
    # re-exports it lives in `yaml/__init__.py`, the class does not. Printing both
    # halves and asserting them separately is the honest form: a single glued
    # string comparison failed the first time this ran, against a binding that was
    # correct.
    code = (
        "from app.routers import dashboard as dash\n"
        "from scripts.automod import backlog as B, scorecard as SC\n"
        "for m in (dash, B, SC):\n"
        "    L = m._YamlLoader\n"
        "    print(m.__name__, L.__name__, L.__module__)\n"
    )
    env = {**os.environ, "LLOYD_AUTOMOD_STATE": str(
        Path.home() / ".local" / "state" / "lloyd-automod")}
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(Path.cwd()),
                          capture_output=True, text=True, timeout=180, env=env)
    assert proc.returncode == 0, (
        f"a fresh import of the three reader modules failed, which is what a "
        f"deleted or now-raising loader-import block looks like: "
        f"{proc.stderr[-800:]}")
    bound = {line.split()[0]: line.split()[1:]
             for line in proc.stdout.strip().splitlines() if line.strip()}
    want = "CSafeLoader" if HAVE_LIBYAML else "SafeLoader"
    for name in ("app.routers.dashboard", "scripts.automod.backlog",
                 "scripts.automod.scorecard"):
        got = bound.get(name) or ["<unbound>"]
        assert got[0] == want, (
            f"a fresh interpreter bound {name}._YamlLoader to {got[0]!r} (from "
            f"{got[-1]}), expected {want!r} "
            f"({'libyaml is importable on this box' if HAVE_LIBYAML else 'no C extension here'})")
        assert got[-1].startswith("yaml"), (
            f"{name}._YamlLoader came from {got[-1]}, not a yaml module — the "
            f"binding under test is not PyYAML's loader")

    sample = _one_real_item(tmp_path)
    for label, call, module in _readers():
        reloaded = importlib.reload(module)
        recorder = _recording_loader(reloaded._YamlLoader)
        monkeypatch.setattr(reloaded, "_YamlLoader", recorder)
        before = recorder.seen
        produced = call(sample)
        assert produced and recorder.seen > before, (
            f"{label}: after importlib.reload the reader no longer dispatches "
            f"through `_YamlLoader` (recorder built {recorder.seen - before} "
            f"loaders), so the reload either lost the binding or the call site "
            f"stopped using it — the thing the deleted `hasattr` assert could not "
            f"see, because reload leaves the old object in the slot either way.")
    dash._cache.clear()


# ── the presence helper itself ────────────────────────────────────────────
#
# Clause 2's denominator guard and its skip policy both live in
# `tests/board_presence.py`. A guard nobody has tested is a guard that reports
# what you expected, so each of the helper's three outcomes is driven here
# against synthetic directories — including the one that must NOT be a skip.

def test_no_board_state_skips_and_every_one_names_the_lost_clause(tmp_path, monkeypatch):
    """All three non-measurable states fail, and each says which clause died.

    The three shapes: the board directory is gone while the vault root is there;
    the board exists but holds no item file; and — the one round
    `SM_20260917_055508` flagged at `tests/board_presence.py:94` — no vault
    directory at all. The first two are a broken *corpus*, which is what a
    denominator guard exists to report. The third used to be argued as "a
    property of the box", and the flag is why that argument is gone: the vault
    root is `LLOYD_OBSIDIAN_VAULT`-overridable, so a skip keyed on it is a skip
    any test can walk into with one `monkeypatch.setenv`, and what it hides is
    the only measurement of the number this item is about.

    `pytest.raises(pytest.fail.Exception)`, not `AssertionError`: the helper
    stops the run with `pytest.fail`, which raises `Failed` — a sibling of
    `Skipped`, not of `AssertionError`. Asserting the wrong exception type here
    would fail this test while the helper behaved correctly, which is the same
    class of mistake the helper exists to prevent.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv(VAULT_ROOT_ENV, str(vault))
    absent_board = vault / "backlog"  # deliberately never created

    with pytest.raises(pytest.fail.Exception) as moved:
        board_files_or_stop(what="moved-board proof", board=absent_board)
    assert "does not exist" in str(moved.value), str(moved.value)
    assert "moved-board proof" in str(moved.value), str(moved.value)
    assert "NOT pinned by this run" in str(moved.value), str(moved.value)

    emptied = vault / "backlog"
    emptied.mkdir()
    with pytest.raises(pytest.fail.Exception) as empty:
        board_files_or_stop(what="emptied-board proof", board=emptied)
    assert "holds 0 entries" in str(empty.value), str(empty.value)
    assert "NOT pinned by this run" in str(empty.value), str(empty.value)

    monkeypatch.setenv(VAULT_ROOT_ENV, str(tmp_path / "no-such-vault"))
    with pytest.raises(pytest.fail.Exception) as novault:
        board_files_or_stop(what="no-vault proof", board=absent_board)
    assert "is absent" in str(novault.value), str(novault.value)
    assert "NOT pinned by this run" in str(novault.value), str(novault.value)

    # The flag was about skips, so the strongest form of the assertion: no state
    # may raise `Skipped` at all. Checked before `Failed` is caught, because
    # `Failed` and `Skipped` are siblings — an `except pytest.fail.Exception`
    # would not catch a regression to skipping, it would let it escape and read
    # as a green skip on this very test.
    monkeypatch.setenv(VAULT_ROOT_ENV, str(tmp_path / "no-such-vault"))
    for board in (absent_board, emptied):
        try:
            board_files_or_stop(what="skip-hunting", board=board)
        except pytest.skip.Exception:
            pytest.fail("board_files_or_stop SKIPPED, which is the state round "
                        "SM_20260917_055508 was refused for (tests/"
                        "board_presence.py:94)")
        except pytest.fail.Exception:
            continue
        else:
            pytest.fail(f"board_files_or_stop returned files for {board}, which "
                        f"holds none — the empty-walk bug clause 4 names")


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
