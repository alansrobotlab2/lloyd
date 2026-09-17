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
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from app.routers import dashboard as dash
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
    board is somebody else's working tree. Falls back to a written fixture when
    the board is absent, so the proof runs either way — a skipped proof is how
    #1199's matcher found an unenforced rule.
    """
    for path in _board_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if _fm_text(text).strip():
            out = tmp_path / "999999-loader-proof.md"
            out.write_text(text, encoding="utf-8")
            return out
    out = tmp_path / "999999-loader-proof.md"
    out.write_text("---\nname: loader proof\nstatus: up_next\npriority: high\n"
                   "tags: [spawned-by-triage, blocker]\n---\n\n# loader proof\n\nbody\n",
                   encoding="utf-8")
    return out


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
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(Path.cwd()),
                          capture_output=True, text=True, timeout=180)
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
    """
    files = _board_files()
    print(f"board dir: {BOARD_DIR}  files walked: {len(files)}")
    assert BOARD_DIR.is_dir(), (
        f"board directory {BOARD_DIR} is absent — this test cannot certify "
        f"loader equivalence over an empty corpus")
    assert files, (
        f"{BOARD_DIR} exists but the *.md glob matched nothing, so the "
        f"comparison below would pass on zero work. A broken glob is not a "
        f"clean board.")

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


def test_module_import_freshly_still_binds_the_loader(tmp_path):
    """Reloading must not lose the loader binding (a module-level name the
    readers close over at call time, not at import of the caller)."""
    for module in (dash, SC, B):
        reloaded = importlib.reload(module)
        assert hasattr(reloaded, "_YamlLoader"), f"{module.__name__} lost _YamlLoader"
    dash._cache.clear()
