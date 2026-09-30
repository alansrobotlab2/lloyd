"""One builder renders a fresh daily note, and every writer reaches it (#1887).

Three writers create `~/obsidian/memory/<date>.md`:

| writer | route | before this round |
|---|---|---|
| `app/post_capture._append_daily_note` | session capture | rendered the block with `yaml.safe_dump` |
| `app/autonomy.append_daily_alert_line` | scheduler/worker alerts | a SECOND `yaml.safe_dump` copy |
| `Notifier._vault_note` | the fleet watchdog | no block at all |

Two copies of a literal plus one writer with no copy is how a day whose first event
is a watchdog alert arrives non-conformant: `scripts/vault/segment_scan.py` reads
front matter with `validate_okf.STRICT_FM_RE`, which is anchored at offset 0, and
`missing_keys` scores a file it cannot parse as missing BOTH required keys.

Byte equality is the property that matters, not parsed equality: the day a writer
emits a leading blank line is the day `segment_scan` counts that note as missing
both keys — which is the live failure this round fixes.
"""

from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
BUILDER = GUARDIAN_DIR / "daily_note.py"
for _p in (str(ROOT), str(GUARDIAN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import notify  # noqa: E402  the watchdog's alert notifier
from app import daily_note as app_daily_note  # noqa: E402
from app import post_capture  # noqa: E402
from scripts.vault.validate_okf import STRICT_FM_RE  # noqa: E402

#: One instant, used by every writer in this file. The `timestamp:` key is part of
#: the block, so two writers can only be byte-identical on the same reading.
STAMP = datetime(2026, 9, 30, 8, 32, 0)
DAY = date(2026, 9, 30)
TITLE = "Runtime data is being written into the code tree"


def _builder():
    """Load the watchdog's builder the way the staged watchdog sees it: by path.

    `agent-services` is not a package, so this is the route `app/daily_note.py` itself
    and `scripts/automod/promote.py:48` use.
    """
    spec = importlib.util.spec_from_file_location("_dn_under_test", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _block(text: str) -> str:
    m = STRICT_FM_RE.match(text)
    assert m, f"no `---` block at offset 0; file starts {text[:40]!r}"
    return m.group(1)


def _capture_block(tmp_path: Path, monkeypatch) -> str:
    """What session capture writes for a day nothing else has touched.

    Redirected by a temp `~`, which is how `tests/test_post_capture_daily_note_okf.py`
    does it: `_append_daily_note` resolves `Path.home()/"obsidian"/"memory"` itself
    (`app/post_capture.py:470`) and takes no vault argument. The day is forced with the
    same `now` the other two writers are frozen at, since LA-local midnight is ~16 h
    from UTC and a fixture keyed on the real date is red for four hours a day (#642).
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("USERPROFILE", raising=False)
    note = tmp_path / "obsidian" / "memory" / f"{DAY}.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    post_capture._append_daily_note("sess-under-test", "the first summary", STAMP)
    return _block(note.read_text())


def _alert_block(tmp_path: Path, monkeypatch) -> str:
    """What the scheduler/worker alert writer writes for a fresh day.

    Its directory comes from `LLOYD_DAILY_NOTE_DIR`, read at call time
    (`app/autonomy.py:3097`), which exists for exactly this. The clock is frozen
    because this writer reads it itself — `datetime.datetime.now(ZoneInfo(TZ))`
    inside the function — and takes no `now` parameter, unlike `_append_daily_note`,
    which is handed one. Freezing the module attribute beats adding a production
    parameter so a test can compare `timestamp:`.
    """
    from app import autonomy

    vault = tmp_path / "memory"
    vault.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LLOYD_DAILY_NOTE_DIR", str(vault))

    class _Naive(datetime):
        @classmethod
        def now(cls, tz=None):
            return STAMP

    class _Clock:
        datetime = _Naive
        date = date

    monkeypatch.setattr(autonomy, "datetime", _Clock)
    assert autonomy.append_daily_alert_line("fleet watchdog ERROR") is True
    return _block((vault / f"{DAY}.md").read_text())


def _guardian_block(tmp_path: Path) -> str:
    """What the fleet watchdog writes for a fresh day — the route that had no block.

    Same construction as `tests/test_guardian_daily_note_okf.py::_notifier`: the
    backend URL points at a closed port and no channel script is passed, so the note
    is the only thing this can affect. `_today` decides the filename and `_stamp` the
    `timestamp:` key — byte equality against the other two writers needs both frozen,
    and `_stamp` exists (#1887) precisely so a test is not standing on a given second.
    """
    vault = tmp_path / "obsidian"
    (vault / "memory").mkdir(parents=True)
    n = notify.Notifier(ledger=tmp_path / "l.jsonl", state_dir=tmp_path,
                        vault_root=str(vault),
                        backend_url="http://127.0.0.1:1")   # nothing listens
    n._today = lambda: DAY
    n._stamp = lambda: STAMP
    assert n._daily_note().name == f"{DAY.isoformat()}.md", n._daily_note()
    assert n._vault_note(TITLE, "move `eval/baselines` out of the tree") is True
    return _block((vault / "memory" / f"{DAY}.md").read_text())


# ── clause 3: one builder, byte-identical across the writers ────────────────


def test_the_two_backend_writers_emit_the_same_block(tmp_path, monkeypatch):
    """`_append_daily_note` and `append_daily_alert_line` no longer disagree.

    They did before this round: both rendered `segment/tags/type/timestamp` with their
    own `yaml.safe_dump`, in two modules, and nothing tested that they produced the
    same bytes. Parsed-equal was the accident that hid it; byte-equal is the property
    `STRICT_FM_RE` and `segment_scan` actually check.
    """
    capture = _capture_block(tmp_path / "cap", monkeypatch)
    alert = _alert_block(tmp_path / "al", monkeypatch)
    assert alert == capture, (
        "the two backend writers disagree:\n"
        f"  alert   ={alert!r}\n  capture ={capture!r}")


def test_the_guardian_writes_the_same_block_as_both_backend_writers(tmp_path, monkeypatch):
    """The clause #1887 is about: the watchdog's block IS the capture writer's block.

    Byte equality, not merely "both carry `segment` and `tags`": a header that differs
    in key order or quoting is a second format that the next repair migration has to
    know about.
    """
    capture = _capture_block(tmp_path / "cap", monkeypatch)
    guardian = _guardian_block(tmp_path / "gd")
    assert guardian == capture, (
        "the watchdog's fresh note is not capture's fresh note:\n"
        f"  guardian={guardian!r}\n  capture  ={capture!r}")
    assert yaml.safe_load(guardian)["segment"] == "memory"
    assert yaml.safe_load(guardian)["tags"] == ["memory", "daily-notes"]


def test_the_backend_reaches_the_watchdogs_builder():
    """`app/daily_note.py` loads the watchdog's file; it does not restate it.

    Compared by source file, source text and rendered bytes — not by object identity.
    `agent-services` is not a package, so the bridge loads the builder under its own
    module name and the two `fresh_header`s are distinct function objects by
    construction (`_builder()` above loads a third instance). Identity would assert
    something about `sys.modules`, not the thing that matters: that the rendering lives
    in exactly one file, and that file is the watchdog's.
    """
    builder = _builder()
    for name in ("fresh_front_matter", "fresh_header"):
        reached = getattr(app_daily_note, name)
        assert Path(inspect.getsourcefile(reached)) == BUILDER, (
            f"app.daily_note.{name} is defined somewhere other than the watchdog's "
            f"builder: {inspect.getsourcefile(reached)}")
        assert inspect.getsource(reached) == inspect.getsource(getattr(builder, name))
    assert app_daily_note.fresh_header(STAMP, DAY.isoformat()) \
        == builder.fresh_header(STAMP, DAY.isoformat())
    # And the bridge renders nothing itself: no literal, no emitter, no second copy.
    body = (ROOT / "app" / "daily_note.py").read_text().split('"""', 2)[2]
    assert "daily-notes" not in body, "app/daily_note.py spells the header inline"
    assert "safe_dump" not in body, "app/daily_note.py renders the header itself"


# ── the properties the replaced yaml.safe_dump gave for free ────────────────


def test_the_block_is_what_yaml_safe_dump_would_emit():
    """`post_capture` dumped rather than spelled, "so the frontmatter is
    strict-parseable by construction", with the kwargs `okf_migrate` uses. The
    builder is hand-rolled because the watchdog may not import `yaml`, so the
    emitter's two real properties have to be pinned here or the rewrite silently
    drops one: block style for the list, and quoting on the `timestamp` scalar so it
    reloads as the string every existing note holds rather than a `datetime`.
    """
    fresh_front_matter = app_daily_note.fresh_front_matter
    dumped = yaml.safe_dump(
        {"segment": "memory", "tags": ["memory", "daily-notes"],
         "type": "note", "timestamp": STAMP.strftime("%Y-%m-%dT%H:%M:%S")},
        sort_keys=False, allow_unicode=True, default_flow_style=False).rstrip()
    assert fresh_front_matter(STAMP) == dumped, (
        "the builder's block is not the bytes `yaml.safe_dump` emits with the kwargs "
        f"okf_migrate uses.\n  yaml:    {dumped!r}\n  builder: {fresh_front_matter(STAMP)!r}")
    assert "- memory" in dumped and "{" not in dumped, (
        "the tags list must stay in block style: the flow form is what the note's "
        f"predecessors did not have:\n{dumped!r}")
    parsed = yaml.safe_load(dumped)
    assert isinstance(parsed["timestamp"], str), (
        f"timestamp reloaded as {type(parsed['timestamp'])}; the hand-rolled block "
        "must quote it or a repair migration reads a datetime where every existing "
        "daily note has a string")
    # The fence too, at the header level. STRICT_FM_RE is anchored at offset 0, so a
    # leading blank line is the difference between a conformant note and one
    # segment_scan scores as missing BOTH keys — today's failure, exactly.
    header = app_daily_note.fresh_header(STAMP, DAY.isoformat())
    assert header.startswith(f"---\n{dumped}\n---\n"), repr(header[:40])
    assert not header.startswith(("\n", " ", "#")), repr(header[:12])


def test_the_shared_builder_needs_nothing_but_the_standard_library():
    """The watchdog's stdlib-only rule, applied to the file that now carries the
    header. `notify.py`'s module docstring states it — the watchdog "must still be
    able to alert when the venv, the backend and the repo are all down" — and a
    `daily_note.py` importing `yaml` would make the watchdog's own import of it the
    thing that fails during an incident.

    Imports come out of the file's syntax tree, and the allowed set is enumerated
    rather than looked up in `stdlib_module_names`: that module is importable in the
    test venv but is itself third-party on the box, and a rail that cannot run is
    worse than a short one.
    """
    imported: set[str] = set()
    for node in __import__("ast").walk(
            __import__("ast").parse(BUILDER.read_text())):
        if isinstance(node, __import__("ast").Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif getattr(node, "module", None) and getattr(node, "level", 1) == 0:
            imported.add(node.module.split(".")[0])
    assert "yaml" not in imported, f"the builder imports yaml: {sorted(imported)}"
    allowed = {"__future__", "datetime", "os", "re", "pathlib", "typing"}
    extra = sorted(imported - allowed)
    assert not extra, (
        f"the watchdog's shared module grew an import outside {sorted(allowed)}: "
        f"{extra}. Anything beyond the standard library makes the watchdog unable to "
        "write a conformant daily note during the outage it is alerting about")


# ── anti-duplication: the literal lives in one module ───────────────────────


def test_no_writer_holds_a_copy_of_the_header_literal():
    """The point of #1887, stated as a rail: exactly one module emits the string.

    `yaml.safe_dump({"tags": ["memory", "daily-notes"]})` in two modules plus a third
    writer with no block at all is what let a watchdog-first day arrive non-conformant
    while a capture-first day looked fine, so the regression guard is that the literal
    exists in one production module and each writer reaches it.

    `--untracked` is load-bearing: a round's worktree holds its own new files, and
    plain `git grep` consults only the index — where a file added minutes ago does not
    exist yet — so without it this node reports "no module emits it" and reads like the
    builder was deleted.
    """
    for name, rel in {
            "app/post_capture.py": "app/post_capture.py",
            "app/autonomy.py": "app/autonomy.py",
            "the watchdog's notify.py": "agent-services/guardian/notify.py"}.items():
        src = (ROOT / rel).read_text()
        assert "daily-notes" not in src, (
            f"{name} spells the daily-note tags inline again — the two-places defect "
            f"#1887 removed, back one edit later")
        assert "daily_note" in src or "fresh_header" in src, (
            f"{name} no longer reaches the shared builder, so nothing pins its header "
            f"to the other writers'")

    holders = subprocess.run(
        ["git", "grep", "--untracked", "-l", "daily-notes", "--", "*.py"],
        cwd=ROOT, capture_output=True, text=True, timeout=120).stdout.split()
    producers = sorted(h for h in holders if not h.startswith("tests/"))
    assert producers == ["agent-services/guardian/daily_note.py"], (
        f"more than one module emits the daily-note header literal: {producers}. The "
        "positive control for a 0-hit grep is the assertion above, which runs first: "
        "if the grep is empty the writers would already have failed it")
