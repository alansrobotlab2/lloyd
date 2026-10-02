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

from app import frontmatter as FM
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
    # Each label is `module.fn (repo-relative path)` with NO line number: all
    # three carried one at #2069's triage and all three had drifted off the def
    # they named (`backlog.py:886` against a def at 1490, `scorecard.py:150`
    # against 230, `dashboard.py:118` against 112), because a label is a string
    # and nothing re-reads it. `test_every_reader_label_names_a_symbol_and_a_path`
    # is what keeps this form.
    return [
        ("dashboard._frontmatter (app/routers/dashboard.py)", dash._frontmatter, dash),
        ("scorecard._frontmatter (scripts/automod/scorecard.py)", SC._frontmatter, SC),
        ("backlog._split_frontmatter (scripts/automod/backlog.py)",
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


# ── #2069: the loader-swap ERROR path — one malformed item, never the walk ──
#
# `scripts/automod/backlog.py::_split_frontmatter` documented its own error path
# as "pinned" by a test that no revision of this repository ever contained
# (`git log --all -S` on the name cited returns only the commit that wrote the
# sentence). What was unpinned was worse than the cross-loader equivalence the
# sentence described: no fixture anywhere drove a malformed front matter through
# that reader under *either* loader, so `except yaml.YAMLError: fm = {}` had
# never run. These two tests are the witness the sentence claimed already existed.

#: Front matter whose `title` opens a double quote and never closes it: the shape
#: `_unparsed_guard`'s docstring records as the one that actually corrupted an
#: item (#1020) — the parse comes back empty, and a writer that re-dumped it
#: would have written back only the keys that writer sets itself, with the
#: surviving front-matter text glued onto the top of the body.
_BROKEN_FM_BLOCK = (
    "type: note\n"
    "status: up_next\n"
    'title: "an unterminated quote that runs off the end of the block\n'
    "board: lloyd\n"
)

#: What the body must come back as: verbatim, `# ` heading still in place, which
#: is the line `load_item` reads an item's name from.
_BROKEN_FM_BODY = ("\n# Item with an unterminated quote\n\n"
                   "Body prose that must survive the unread item.\n")

_BROKEN_FM_TEXT = f"---\n{_BROKEN_FM_BLOCK}---\n{_BROKEN_FM_BODY}"

#: The same body behind front matter that *does* parse, so the empty dict the
#: broken file gets is a degradation and not this reader answering `{}` to
#: everything.
_GOOD_FM_TEXT = ("---\ntype: note\nstatus: up_next\n"
                 "title: A well-formed item\nboard: lloyd\n---\n" + _BROKEN_FM_BODY)

#: `_YamlLoader`'s two possible bindings, as `(label, class)`.
def _loader_arms() -> list[tuple[str, type]]:
    """Both arms, and the test that needs two of them says so when there is one.

    No conditional skip marker anywhere in here, in the policy of `HAVE_LIBYAML`
    above: `test_all_three_modules_select_the_c_loader_when_libyaml_is_present`
    already fails red on a box with no C extension, and a marker keyed on the box
    is where an unmade comparison goes to hide.
    """
    return [("CSafeLoader", getattr(yaml, "CSafeLoader", yaml.SafeLoader)),
            ("SafeLoader", yaml.SafeLoader)]


def _two_loaders_present_or_fail(arms) -> None:
    """Refuse to compare a loader with itself.

    A comparison whose two arms are one class cannot fail, which is the shape the
    review refused on #1975. On a box with libyaml the two are distinct — the
    assert below is the positive control on that, not a formality: `CSafeLoader`
    resolving to `SafeLoader` while `yaml._yaml` reports present is exactly the
    drift `test_all_three_modules_select_the_c_loader_when_libyaml_is_present`
    exists to catch.
    """
    assert HAVE_LIBYAML, (
        "this box's PyYAML has no C extension (`yaml._yaml` is None), so both arms "
        "here are SafeLoader and 'identical under the C loader and the pure-Python "
        "one' is a comparison this run cannot make. NOT pinned by this run.")
    names = {label: loader for label, loader in arms}
    assert names["CSafeLoader"] is not names["SafeLoader"], (
        "CSafeLoader resolved to SafeLoader on a box that reports libyaml present, "
        "so the two arms below are one class")


def test_unterminated_front_matter_degrades_the_same_under_either_loader(monkeypatch):
    """`_split_frontmatter` swallows the scanner error from BOTH loaders, alike.

    Three things, in the order the clause names them:

    1. The bytes are malformed **to the parse the handler catches**. Handed the
       block directly, each loader raises a *subclass* of `yaml.YAMLError`
       (`yaml.scanner.ScannerError` from both arms, the C one carrying a
       `yaml._yaml.Mark`) — that subclass relation is the whole claim that
       "`yaml.YAMLError` still catches both", and it is now pinned instead of
       asserted by a citation to a test that did not exist. Without this step the
       test would be proving a degradation that never happens.
    2. Handed the same text through the reader, `_split_frontmatter` returns an
       **empty front-matter dict and the body untouched**, under the C loader and
       under the pure-Python one, and the two results are equal to each other. No
       exception reaches the caller — the `except yaml.YAMLError: fm = {}` arm is
       what the docstring's stakes depend on, and delete it and step 2 fails with
       the scanner error named in this test's output.
    3. The same reader on well-formed bytes returns its keys, so the empty dict in
       (2) is a degradation rather than a reader that returns `{` for every file.
    """
    arms = _loader_arms()
    _two_loaders_present_or_fail(arms)

    split = FM.split_frontmatter(_BROKEN_FM_TEXT)
    assert split is not None and split[0] == _BROKEN_FM_BLOCK, (
        "the fixture's fences are not the fences the reader splits on, so the "
        "parse below would be handed the wrong bytes")

    raised: dict[str, type] = {}
    for label, loader in arms:
        with pytest.raises(yaml.YAMLError) as caught:
            yaml.load(_BROKEN_FM_BLOCK, Loader=loader)
        exc = caught.value
        assert type(exc) is not yaml.YAMLError, (
            f"{label}: raised `yaml.YAMLError` itself rather than a subclass, so "
            f"'the C scanner raises subclasses of it too' is not what this box "
            f"does — the handler catches it, but not as the docstring says")
        raised[label] = type(exc)
    # `ScannerError` from BOTH arms: the pure-Python scanner's class, which the
    # libyaml extension constructs too (with its own `yaml._yaml.Mark`). One
    # `except yaml.YAMLError` therefore covers the box with libyaml and the box
    # without it — the claim this item's whole change rests on, measured here
    # rather than cited.
    got = {label: cls.__name__ for label, cls in raised.items()}
    assert got == {"CSafeLoader": "ScannerError", "SafeLoader": "ScannerError"}, (
        f"expected a ScannerError subclass out of each loader, got {got}")

    results: dict[str, tuple[dict, str]] = {}
    for label, loader in arms:
        monkeypatch.setattr(B, "_YamlLoader", loader)
        try:
            results[label] = B._split_frontmatter(_BROKEN_FM_TEXT)
        except Exception as exc:  # noqa: BLE001 — reaching here IS the failure
            pytest.fail(
                f"{label}: `_split_frontmatter` let the parse error escape "
                f"({type(exc).__module__}.{type(exc).__name__}: {exc}), so one "
                f"malformed item now raises through every caller of it — the "
                f"25 functions in scripts/automod/backlog.py whose board walks "
                f"this reader is called from. The `except yaml.YAMLError` arm is "
                f"the only thing between a broken item and a dead walk.")
        good_fm, good_body = B._split_frontmatter(_GOOD_FM_TEXT)
        assert good_fm.get("status") == "up_next", (
            f"{label}: the well-formed control parsed to {good_fm!r}, so an empty "
            f"dict from the broken file above proves nothing — a reader that "
            f"returns {{}} to every file would pass step 2 by accident")
        assert good_body == _BROKEN_FM_BODY, f"{label}: control body came back changed"

    under_c, under_py = results["CSafeLoader"], results["SafeLoader"]
    assert under_c[0] == {}, (
        f"CSafeLoader: malformed front matter produced {under_c[0]!r} rather than "
        f"an empty dict — a half-parsed item is what a writer re-dumps and loses")
    assert under_c[1] == _BROKEN_FM_BODY, (
        f"CSafeLoader: the body came back changed — {under_c[1][:80]!r} — which is "
        f"the #1020 corruption: the unread front matter landing on the body and "
        f"destroying the `# ` heading every later read extracts the name from")
    assert under_c == under_py, (
        f"the two loaders do not degrade the same item the same way: "
        f"CSafeLoader {under_c[0]!r}, SafeLoader {under_py[0]!r}. The claim that "
        f"`yaml.YAMLError` catches both is what makes the loader swap safe, and "
        f"it has to hold on the error path, not only on files that parse.")


def test_one_malformed_item_costs_one_item_not_the_whole_board_walk(tmp_path, monkeypatch):
    """The consequence the docstring's stakes assert: the walk survives the bad file.

    Three item files, one of them the unterminated-quote front matter, walked by
    `all_items` — the function the board steward and the dispatch loop read —
    under each loader in turn. All three items come back: the two well-formed
    ones with the status their own front matter states, the broken one on the
    reader's defaults with its `# ` heading still readable out of the body. That
    is "one malformed item degrades to one unread item": a status the loop does
    not dispatch on, not an exception through the walk.

    The walk is a cached one (`_load_item_cached`), so the parse counter
    `_items_parses` — the instrument `tests/test_backlog_item_cache.py` reads — is
    checked beside it: a second arm that served the cache would compare nothing,
    so each arm must prove it parsed all three files again.

    `_unparsed_guard` is asserted at the end because it is why degrading to `{}`
    is safe rather than lossy: the guard is what stops a writer re-dumping an
    unread item and destroying it (#1020).
    """
    arms = _loader_arms()
    _two_loaders_present_or_fail(arms)

    board = tmp_path / "board"
    board.mkdir()
    good_open = board / "2069001-open.md"
    good_open.write_text(_GOOD_FM_TEXT, encoding="utf-8")
    broken = board / "2069002-broken.md"
    broken.write_text(_BROKEN_FM_TEXT, encoding="utf-8")
    good_done = board / "2069003-done.md"
    good_done.write_text(_GOOD_FM_TEXT.replace("status: up_next", "status: done"),
                         encoding="utf-8")
    paths = [good_open, broken, good_done]

    for label, loader in arms:
        monkeypatch.setattr(B, "_YamlLoader", loader)
        B._items_cache.clear()
        before = sum(B._items_parses.get(str(p), 0) for p in paths)
        items = B.all_items(boards=None, backlog_dir=board)
        parsed = sum(B._items_parses.get(str(p), 0) for p in paths) - before
        assert parsed == len(paths), (
            f"{label}: the walk parsed {parsed} of {len(paths)} files, the rest "
            f"served from `_items_cache` — so this arm did not read the broken "
            f"file with this loader at all and proved nothing")
        assert len(items) == len(paths), (
            f"{label}: the walk returned {len(items)} of {len(paths)} items, so "
            f"the malformed file took something other than itself")
        by_id = {item.id: item for item in items}
        assert by_id[2069001].status == "up_next", (
            f"{label}: the well-formed open item came back {by_id[2069001].status!r}")
        assert by_id[2069003].status == "done", (
            f"{label}: the well-formed done item came back {by_id[2069003].status!r}")
        unread = by_id[2069002]
        assert unread.status == "draft", (
            f"{label}: the malformed item read as {unread.status!r}, not the "
            f"reader's `draft` default — an unread item that reports a status it "
            f"did not state is a dispatch decision out of a parse failure")
        assert unread.name == "Item with an unterminated quote", (
            f"{label}: name extracted as {unread.name!r}, so the body heading the "
            f"unread front matter used to be glued onto is gone (#1020)")
        assert "Body prose that must survive the unread item." in unread.body, (
            f"{label}: the malformed item's body did not survive the walk")

    # The writer half of the same shape: an unread item must not be re-dumped.
    assert B._unparsed_guard(broken, _BROKEN_FM_TEXT, {}, "test-guard") is True, (
        "a fenced file that parsed to no keys is not refused, so the empty dict "
        "this degradation produces would be written back as an item stripped of "
        "everything the writer did not itself set")
    assert B._unparsed_guard(good_open, _GOOD_FM_TEXT, {"status": "up_next"},
                             "test-guard") is False, (
        "a well-formed item is refused by the guard, which would make the board "
        "unwritable rather than degrading one item")


def test_every_reader_label_names_a_symbol_and_a_path():
    """The labels in `_readers()` resolve; the line numbers they used to carry did not.

    All three carried a `file.py:NNN` and #2069's triage found all three off the
    def they named — `backlog._split_frontmatter` labelled `backlog.py:886` while
    the def sat at 1490. A label is printed into every failure message in this
    file, so a drifted one points a reader at unrelated code while looking like a
    citation. The form now is symbol plus path, which cannot drift, and this is
    the guard that keeps it: no `:digits`, and the named path must exist and
    really define the symbol the label claims.
    """
    root = Path(__file__).resolve().parents[1]
    for label, _call, module in _readers():
        assert ":" not in label, f"{label!r}: a label that carries a line number drifts"
        symbol, _, path = label.partition(" (")
        path = path.rstrip(")")
        assert path, f"{label!r}: no path in the label"
        source = root / path
        assert source.is_file(), f"{label!r}: names {path}, which is not a file"
        fn = symbol.split(".")[-1]
        assert f"def {fn}(" in source.read_text(encoding="utf-8"), (
            f"{label!r}: {path} does not define `{fn}`")
        assert fn in dir(module), (
            f"{label!r}: {module.__name__} has no attribute `{fn}`")
