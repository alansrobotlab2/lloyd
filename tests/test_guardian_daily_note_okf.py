"""The guardian's daily-note route must produce an OKF-conformant note.

Two writers touch `memory/<date>.md`. The backend's
(`app/post_capture._append_daily_note`, mirrored by
`app/autonomy.append_daily_alert_line`) creates a missing day with a
`---`-delimited front-matter block. The guardian's
`agent-services/guardian/notify.py::Notifier._vault_note` used to read
`body = note.read_text(...) if note.exists() else ""` and then
`open(note, "a")`, so on a day the guardian reached before any session capture the
`open()` CREATED the file from a blank lead and a `## Self-mod guardian:` heading —
no front matter at all.

That is not a cosmetic problem: `scripts/vault/segment_scan.py` counts a file it
cannot parse as missing BOTH required keys ("nothing that cannot be read can prove
it carries a list"), so the bare note turned
`tests/test_okf_segment_producers.py::test_scan_exits_0_on_the_live_vault` red —
which is how #83's Pre-Flight found it on 2026-09-30, where `memory/2026-09-30.md`
appeared under both `missing segment:` and `missing tags:`. Which writer wins is
decided by which fires first after midnight, so some days are clean and some are
not.

These nodes pin the fix: a guardian-created note is conformant ON ARRIVAL, the
section lands after the front-matter block rather than as the file's first line,
the real scanner reports zero on a vault holding only that note, and appending to a
note that already exists is byte-for-byte what it was before.

The append path is what every daily note in the vault has already had done to it,
so it is pinned by prefix equality rather than by a prose claim: the bytes that were
in the file before the call must still be there, unchanged and unprepended.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(GUARDIAN_DIR) not in sys.path:
    sys.path.insert(0, str(GUARDIAN_DIR))

import notify  # noqa: E402
from scripts.vault import segment_scan  # noqa: E402
from scripts.vault.validate_okf import STRICT_FM_RE  # noqa: E402

DAY = date(2026, 9, 30)
NOTE_REL = f"memory/{DAY.isoformat()}.md"
TITLE = "Runtime data is being written into the code tree"
BODY = "move `eval/baselines` out of the tree"

# The exact bytes the shipped writer produced on 2026-09-30, quoted from
# `~/obsidian/memory/2026-09-30.md` at triage: a blank lead, then the section.
# Reconstructed here so the scanner half of this file has a POSITIVE CONTROL — a
# note that really does score missing is the only thing that makes its reporting
# zero mean anything.
BARE_NOTE_PRE_FIX = f"\n\n## Self-mod guardian: {TITLE}\n\n{BODY}\n"


def _notifier(tmp_path: Path) -> "notify.Notifier":
    """A real `Notifier` over a throwaway vault, standing on `DAY`.

    `_today` is the only date the notifier consults (see
    `tests/test_guardian_predicates.py::_day_notifier`), and `_vault_note` and
    `resolve` are the shipped methods, so what these nodes assert is what the hourly
    watchdog writes to the files.
    """
    vault = tmp_path / "obsidian"
    (vault / "memory").mkdir(parents=True, exist_ok=True)
    n = notify.Notifier(ledger=tmp_path / "l.jsonl", state_dir=tmp_path,
                        vault_root=str(vault), backend_url="http://127.0.0.1:1")
    n._today = lambda: DAY
    assert n._daily_note().name == f"{DAY.isoformat()}.md", n._daily_note()
    return n


def _fm_of(text: str) -> dict:
    m = STRICT_FM_RE.match(text)
    assert m, f"no front-matter block at offset 0; file starts {text[:60]!r}"
    fm = yaml.safe_load(m.group(1))
    assert isinstance(fm, dict), f"front matter is not a mapping: {fm!r}"
    return fm


def _scan_of(vault: Path):
    """`scan` over `vault` with the shipped dir/exclude lists, plus that list.

    The directory list comes back so a caller can assert `memory/` was in scope:
    an empty scan result and a scanner that never looked are otherwise identical.
    """
    dirs, excludes = segment_scan.scan_dirs()
    return segment_scan.scan(vault, dirs, excludes), dirs


# ── clause 1: the note the guardian creates carries the header ──────────────


def test_a_guardian_first_day_creates_the_note_with_conformant_front_matter(tmp_path):
    """The alert fires before any session capture: the file does not exist yet.

    The created note must satisfy the scanner's two required keys on arrival — a
    `segment:` scalar and a `tags:` LIST — because the scanner's own rule is that a
    file it cannot parse proves nothing about either key, and a day created this way
    failed `test_scan_exits_0_on_the_live_vault` on 2026-09-30.
    """
    n = _notifier(tmp_path)
    note = n._daily_note()
    assert not note.exists(), "fixture: this is the guardian-first case"

    assert n._vault_note(TITLE, BODY, coalesce=False) is True, "_vault_note reported failure"
    text = note.read_text()

    fm = _fm_of(text)
    assert fm.get("segment") == "memory", fm
    assert isinstance(fm.get("tags"), list), f"tags must be a LIST, got {fm.get('tags')!r}"
    assert {"memory", "daily-notes"} <= set(fm["tags"]), fm["tags"]
    assert text.splitlines()[0] == "---", "front matter must start at offset 0"


def test_the_guardian_section_lands_after_the_header_never_as_the_first_line(tmp_path):
    """The section is still the alert a reader recognises, just no longer the head.

    Asserted on the section's position relative to the closing `---` fence rather
    than on a count of headings: the defect was the section being the file's first
    non-blank line, and a count would also pass on a note whose header arrived after
    it.
    """
    n = _notifier(tmp_path)
    n._vault_note(TITLE, BODY, coalesce=False)
    text = n._daily_note().read_text()

    section_at = text.index(f"{notify.DAILY_SECTION_PREFIX}{TITLE}")
    fence_at = text.index("---", 3)                       # the closing fence
    assert section_at > fence_at, "the alert section precedes the front matter"
    assert text[:fence_at].count("---") == 1, "more than one fence before the section"
    assert BODY in text[section_at:], "the alert body is not in its section"


# ── clause 2: the real scanner scores the guardian's note at zero ───────────


def test_a_bare_section_only_note_scores_missing_on_the_real_scanner(tmp_path):
    """POSITIVE CONTROL for the node below, and the bug stated as a test.

    `main` returning 0 over a root would mean nothing if the scanner were not
    looking at the file at all, so the pre-fix bytes — the blank lead plus the
    section, exactly what `~/obsidian/memory/2026-09-30.md` held — are scored first
    and must come back missing BOTH keys. This node is what fails when the scanner
    stops seeing daily notes, and the node below passes.
    """
    vault = tmp_path / "obsidian"
    (vault / "memory").mkdir(parents=True)
    note = vault / "memory" / f"{DAY.isoformat()}.md"
    note.write_text(BARE_NOTE_PRE_FIX)

    assert segment_scan.missing_keys(note) == (True, True), (
        "the scanner did not score the bare note as missing both keys")
    result, dirs = _scan_of(vault)
    assert "memory" in dirs and NOTE_REL in result.get("memory", {}).get("segment", []), (
        f"`memory/` was in scope {dirs} but the bare note was not named: {result}")
    assert segment_scan.main(["--root", str(vault)]) == 1, "the pre-fix note must exit 1"


def test_a_vault_holding_only_the_guardians_note_scores_zero(tmp_path):
    """The fix, measured by the shipped scanner over the root it ships with.

    Same root shape, same filename, same section as the control above: the only
    difference is who created the file. Zero offenders and `main` == 0 here is the
    condition `test_scan_exits_0_on_the_live_vault` needs on a guardian-first day.
    """
    n = _notifier(tmp_path)
    n._vault_note(TITLE, BODY, coalesce=False)

    assert segment_scan.missing_keys(n._daily_note()) == (False, False)
    result, dirs = _scan_of(n.vault_root)
    assert "memory" in result, (
        f"the scanner scanned no `memory/` directory ({dirs} from its own config), so "
        f"the emptiness below would be a measurement of nothing")
    assert result["memory"] == {"segment": [], "tags": []}, result["memory"]
    assert segment_scan.main(["--root", str(n.vault_root)]) == 0


# ── clause 4: appending to a note that already exists is unchanged ──────────


def test_appending_to_an_existing_note_leaves_its_header_and_prefix_untouched(tmp_path):
    """Two alerts on a day someone else started: one header, both sections.

    The pre-existing front matter is what the daily notes in the vault already
    carry, so the regression this guards is the loud one — a second `---` block, or a
    header prepended above the reader's own. Pinned by prefix equality on the bytes
    that were in the file before the call, not by a re-derived rendering of what the
    header should look like.
    """
    n = _notifier(tmp_path)
    note = n._daily_note()
    original = (
        "---\nsegment: memory\ntags:\n- memory\n- daily-notes\ntype: note\n"
        "timestamp: '2026-09-30T00:07:00'\n---\n\n"
        "# 2026-09-30 Daily Notes\n\n## Sessions\n\n- 00:07 — captured a session\n"
    )
    note.write_text(original)

    assert n._vault_note(TITLE, BODY, coalesce=False) is True
    after = note.read_text()
    assert after.startswith(original), "the pre-existing bytes were rewritten or prepended"
    assert after.count("---\nsegment: memory") == 1, "a second front-matter block appeared"
    assert after.count(f"{notify.DAILY_SECTION_PREFIX}{TITLE}") == 1, (
        "the alert section was not appended exactly once")
    assert after.index(f"{notify.DAILY_SECTION_PREFIX}{TITLE}") > after.index(
        "## Sessions"), "the alert was inserted before the existing body"


def test_a_second_alert_appends_and_a_coalescing_alert_replaces_in_place(tmp_path):
    """Both append-mode routes keep the header they did not write.

    The non-coalescing path is #1887's alert; the coalescing one is the hourly
    re-write that `resolve` and the cross-day retraction depend on. Neither may
    disturb the block, and the counts below are the denominator for that claim: four
    sections written, three headings' worth of text left behind, exactly one fence
    pair.
    """
    n = _notifier(tmp_path)
    note = n._daily_note()
    note.write_text(
        "---\nsegment: memory\ntags:\n- memory\n- daily-notes\n---\n\n"
        "# 2026-09-30 Daily Notes\n")

    n._vault_note(TITLE, BODY, coalesce=False)
    n._vault_note("A second, distinct alarm", "second body", coalesce=False)
    n._vault_note("Coalesced alarm", "first saying", coalesce=True)
    n._vault_note("Coalesced alarm", "second saying", coalesce=True)

    text = note.read_text()
    fm = _fm_of(text)
    assert fm.get("segment") == "memory" and isinstance(fm.get("tags"), list), fm
    assert text.count("---\nsegment: memory") == 1, "the header moved or duplicated"
    assert text.count(notify.DAILY_SECTION_PREFIX) == 3, (
        f"expected three distinct sections (two appended, one coalesced from two "
        f"writes), got {text.count(notify.DAILY_SECTION_PREFIX)}: "
        f"{[l for l in text.splitlines() if l.startswith('#')]}")
    assert "second saying" in text and "first saying" not in text, (
        "the coalescing route no longer replaces its own section")


# ── the route this item does NOT change ────────────────────────────────────


def test_resolve_still_refuses_to_create_a_note_it_did_not_open(tmp_path):
    """`resolve` guards itself with `not note.is_file(): continue`.

    It never created a bare note, and it must not start creating conformant ones
    either: an all-clear for an incident nobody logged is not worth inventing a
    day's file. This node exists so the fix is not over-applied by a later edit that
    "shares the helper everywhere".
    """
    n = _notifier(tmp_path)
    assert not n._daily_note().exists()

    # True, not False: `resolve`'s own docstring says an all-clear that "none of them
    # held an open section ... is the ordinary hourly all-clear and stays True so the
    # idempotent-silence contract holds". What this node pins is that the silent path
    # creates NOTHING.
    assert n.resolve("An alarm never written", "all clear") is True
    assert not n._daily_note().exists(), (
        "`resolve` created the day's note. A conformant empty note is still a note "
        "nobody asked for, and the hourly all-clear would leave one behind every day")


# ── the process boundary: repo → pinned snapshot → systemd unit ─────────────


def test_the_snapshot_the_unit_actually_runs_will_carry_the_builder(tmp_path):
    """`agent-services/guardian/daily_note.py` reaches the machine, or the fix is inert.

    `lloyd-guardian.service:26,29` runs `%h/.local/state/lloyd-guardian/bin/guardian.py`
    — a pinned copy promoted by `guardian-stage.sh`, which copies `guardian/*.py` after
    compiling it and running `selftest.py --profile staging`. The guardian never
    executes the repo file, so a builder that exists only in the checkout changes
    nothing on this box: the process boundary is the unit's start, not the commit.

    Two halves, both checkable from the repo: the copy glob must actually select the
    new file (a module staged by an enumerated list would need editing here, and the
    glob is `*.py`, so it does not), and the whole module set must compile and import
    under `/usr/bin/python3` — the interpreter the unit names, which has NO venv and
    therefore no `yaml`. That is the same interpreter `guardian-stage.sh:34` runs the
    promoting selftest in, so a `daily_note.py` that needed `yaml` would be declined by
    staging itself and the box would keep a stale-but-working watchdog.
    """
    script = (ROOT / "agent-services" / "bin" / "guardian-stage.sh").read_text()
    # Two glob copies, and both must stay globs: SRC -> STAGE (where it is compiled
    # and selftested) at :42, then STAGE -> DST (the pinned snapshot) at :79. If
    # either becomes a list of filenames, `daily_note.py` must be added to it or the
    # watchdog keeps running the headerless writer forever and nothing says so.
    globs = [lit for lit in ('cp "$SRC"/*.py', 'cp "$STAGE"/*.py')
             if lit in script]
    assert len(globs) == 2, (
        "guardian-stage.sh is expected to copy `*.py` in both hops — source to stage "
        f"(line 42) and stage to snapshot (line 79); matched only {globs}. An "
        "enumerated copy list would silently leave the watchdog running the "
        "headerless writer, which no test in this repo could see")
    staged = subprocess.run(
        [sys.executable, "-c",
         "import glob,os,sys\n"
         "print('\\n'.join(sorted(os.path.basename(p) for p in "
         "glob.glob(os.path.join(sys.argv[1],'*.py')))))",
         str(GUARDIAN_DIR)],
        capture_output=True, text=True, timeout=60).stdout.split()
    assert "daily_note.py" in staged, (
        f"the file the snapshot's glob selects does not include the builder: {staged}")
    # The general property behind that: `notify.py` now resolves modules from its own
    # directory, so EVERY sibling it imports must be a file in that directory, or the
    # snapshot the unit runs dies at import and the box silently loses its watchdog.
    # (This is the failure the suite caught on the gate's first pass here, from a
    # fixture that copied `notify.py` alone: ModuleNotFoundError: daily_note.)
    spec = __import__("ast").parse((GUARDIAN_DIR / "notify.py").read_text())
    wanted: set[str] = set()
    for node in __import__("ast").walk(spec):
        if isinstance(node, __import__("ast").Import):
            wanted |= {a.name.split(".")[0] for a in node.names}
        elif getattr(node, "module", None) and getattr(node, "level", 1) == 0:
            wanted.add(node.module.split(".")[0])
    local = {n[:-3] for n in staged}
    unresolved = sorted(m for m in wanted
                        if m not in local and m not in sys.stdlib_module_names)
    assert not unresolved, (
        f"notify.py imports {unresolved}, which is neither a module in "
        f"{GUARDIAN_DIR.name}/ (so the snapshot's `*.py` copy carries it) nor "
        "standard library — the pinned copy the unit runs would fail to import at "
        "boot and the box would silently lose its watchdog")
    assert "daily_note" in wanted, (
        "notify.py no longer imports the shared builder at module level, so this "
        "node's subject moved: find where fresh_header is reached from now")

    src = tmp_path / "guardian"
    src.mkdir()
    for name in staged:
        (src / name).write_text((GUARDIAN_DIR / name).read_text())
    compile_dir = tmp_path / "bin"
    compile_dir.mkdir()
    out = subprocess.run(
        ["/usr/bin/python3", "-c",
         "import sys, py_compile, pathlib\n"
         "sys.dont_write_bytecode = True\n"
         "src = pathlib.Path(sys.argv[1])\n"
         "for p in sorted(src.glob('*.py')):\n"
         "    py_compile.compile(str(p), cfile=str(pathlib.Path(sys.argv[2]) / (p.stem + '.pyc')), doraise=True)\n"
         "sys.path.insert(0, str(src))\n"
         "import notify\n"
         "assert notify.daily_note is not None\n"
         "print('compiled and imported', len(list(src.glob('*.py'))), 'modules')",
         str(src), str(compile_dir)],
        capture_output=True, text=True, timeout=120,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
    assert out.returncode == 0, (
        f"/usr/bin/python3 could not compile and import the snapshot's module set — "
        f"guardian-stage.sh would decline it and the box would keep the previous "
        f"watchdog: {out.stderr[-400:]}")
    n_compiled = int(out.stdout.split()[-2])
    assert n_compiled == len(staged), f"{out.stdout!r} vs {len(staged)} source modules"
