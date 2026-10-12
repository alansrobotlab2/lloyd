"""Item #474 — the legacy ``entity-aliases.json`` export is a snapshot, not a table.

``_pipeline/vault-derived/facts/entity-aliases.json`` was written once, by the
2026-09 store migration, and never again. It sat *inside* the fact tree it
claimed to describe, so anything that measured it got a number frozen at the
migration date that was ~5x the live ``aliases`` table — and read the gap as a
defect in the store (backlog #342 "corrupt JSON tail",
``autonomy-runs/24/run_24_20260823_055828.md`` "frozen-corrupt 22:04:59 copy",
and a loaded-memory line that cited its entry count as a standing defect long
after both it and the store row count it was compared against had gone stale).

The fate chosen in #474: the export stays as a *snapshot* and stops living in
the live tree. ``Store.export_json()`` is called only by the one-shot migration
and by the pre-rebuild freeze, and both write timestamped directories under
``_pipeline/backups/``. Nothing else writes it, and no doc or task may call the
JSON file the alias source.

Each test below pins one half of that decision. The failure mode being guarded
is silent — an export that goes stale cannot raise, it just gets believed — so
the pins are on the writer set, the destination, and the claims, not on the
bytes of the file.
"""

import ast
import inspect
import json
import re
import sqlite3
from pathlib import Path

from tests._live_data import require_live_data
from app.paths import production_data_root  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
AUTONOMY_DIR = Path.home() / "obsidian" / "autonomy"

# Files allowed to reach Store.export_json(): the definition itself plus the two
# one-shot tools. A fourth entry means something new started writing a file that
# reads as live and never refreshes — which is the whole of #474. Add an entry
# only with a destination under _pipeline/backups/<ts>/ and a reason.
EXPORT_JSON_CALLERS = {
    "scripts/memory/kg_migrate_to_sqlite.py",   # migration round-trip check
    "scripts/memory/kg_rebuild.py",             # `json-before` freeze snapshot
}
# The one module allowed to define the exporter. A second definition means a
# second writer that this pin cannot see.
EXPORTER_HOME = "app/kg_store.py"

# What the nightly scheduled pipeline actually runs. This is the writer that
# wiped the graph on 2026-08-22, so it is the one place an export call must
# never appear.
SCHEDULED_PIPELINE_DIR = "scripts/memory/next-gen-memory"

# The claim that must not exist in code: naming the JSON authoritative. Judged
# over a comment window rather than one line — and an adjective is no excuse.
# "The legacy file is the source of truth" is exactly the shape that fooled
# readers for nine days, so a window may keep a claim only by *both* negating it
# and naming where the truth actually lives.
SOURCE_CLAIM = re.compile(
    r"source of truth|authoritative|canonical alias (?:file|map|table)"
    r"|(?:live|current|real) alias(?:es)? (?:file|map|table|export|source)", re.I)
RETIRES_CLAIM = re.compile(r"\bnot\b|never|no longer|retired|frozen|deprecated|stray", re.I)
LIVES_HERE = re.compile(r"app\.kg_store|kg\.sqlite|`aliases`|aliases? table", re.I)

_SKIP_DIRS = {".venvs", ".git", "node_modules", "__pycache__", ".pytest_cache",
              "_pipeline", "logs", "sessions", "event_logs", "graphify-out",
              "autonomy-runs", "obsidian", ".mypy_cache"}


def _py_files(needle: str | None = None):
    """Every tracked-ish .py in the checkout, heavy dirs pruned.

    `needle` is a cheap pre-filter: the pins ask about one symbol, so files that
    cannot mention it are not worth parsing.
    """
    for path in REPO.rglob("*.py"):
        if _SKIP_DIRS.intersection(path.parts):
            continue
        if needle is not None and needle not in path.read_text(encoding="utf-8", errors="replace"):
            continue
        yield path


def _callers_of(method: str, include_tests: bool = False) -> set[str]:
    """Repo-relative paths of non-test files containing a ``.<method>(`` call.

    Test files call the exporter freely — against temp stores, which is the whole
    point — so the pin is over the code that ships. Pass `include_tests` to see
    the full set anyway.
    """
    found: set[str] = set()
    for path in _py_files(needle=f".{method}("):
        if not include_tests and path.relative_to(REPO).parts[0] == "tests":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # a script the suite loads by exec, or py2-era debris
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == method):
                found.add(path.relative_to(REPO).as_posix())
                break
    return found


# ── clause 2: who may write the file ────────────────────────────────────────

def test_export_json_has_exactly_the_migration_and_rebuild_callers():
    """The writer set is closed. A new caller is a new stale file in waiting, and
    clause 2 asks for this set to be pinned rather than asserted in prose."""
    assert _callers_of("export_json") == EXPORT_JSON_CALLERS


def test_the_exporter_is_defined_in_exactly_one_module():
    """A second `def export_json` would be a writer the caller pin cannot see."""
    defs = {path.relative_to(REPO).as_posix() for path in _py_files(needle="def export_json")
            if any(isinstance(n, ast.FunctionDef) and n.name == "export_json"
                   for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))))}
    assert defs == {EXPORTER_HOME}


def test_the_scheduled_extraction_pipeline_never_calls_export_json():
    """Task #24 runs scripts/memory/next-gen-memory/. If the exporter ever moves
    in there, the legacy file becomes live-shaped again on a 6x-daily cadence —
    with that directory being the writer that destroyed the graph on 08-22."""
    scheduled = {f for f in _callers_of("export_json")
                 if f.startswith(SCHEDULED_PIPELINE_DIR + "/")}
    assert scheduled == set(), f"scheduled pipeline scripts export the legacy JSON: {sorted(scheduled)}"


def test_export_json_takes_no_default_destination():
    """`dest_dir` must stay a required argument. A default is how a live path
    reappears inside the library instead of at the call site that can be read."""
    from app.kg_store import KGStore
    param = inspect.signature(KGStore.export_json).parameters["dest_dir"]
    assert param.default is inspect.Parameter.empty


ACTIVITY_HEADING = re.compile(r"^##\s+Activity", re.I)
DATED_BULLET = re.compile(r"^-\s+\d{4}-\d{2}-\d{2}T")


def _task_instructions(text: str) -> str:
    """The part of an autonomy task file that *registers work*.

    Everything after `## Activity Log` is history the scheduler never executes,
    and dated bullets under it are run reports — #24 and #74 both legitimately
    record "kg_rebuild.py freeze paused this task". Matching those would make
    the pin false-positive on prose, and a pin that fires on an honest log line
    gets deleted the first time it annoys someone.
    """
    kept = []
    for line in text.splitlines():
        if ACTIVITY_HEADING.match(line):
            break
        if DATED_BULLET.match(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def test_no_recurring_autonomy_task_is_told_to_run_an_exporter():
    """Clause 2's other half: neither caller is wired to a recurring job. The
    migration and the rebuild are run by hand; a task that automates either one
    puts the frozen-file failure back on a schedule.

    Matches the script names a task body would actually contain (`python3
    scripts/memory/kg_rebuild.py freeze`), not the Python method name — no task
    file has ever spelled `export_json`. Reads the live task dir, which is what
    the scheduler reads (precedent: test_automod_hardening, test_backlog_tags_shape).

    #2561 is the hit this fires on that is not an instruction at all. #79's weekly
    retention sweep took on a `facts-quarantine-*` store and its numbered list named
    `scripts/memory/kg_rebuild.py` as the writer that moves a superseded facts tree
    into it — true, useful, and nothing to do with export. But the needle is a bare
    substring, `kg_rebuild.py` sits in EXPORT_JSON_CALLERS above, and a task's
    front matter IS the prompt `_build_task_prompt` hands the worker, so a sentence
    of provenance reddened main. The rule the fix follows: a recurring task that has
    to refer to one of these two scripts names WHAT IT DOES ("the KG rebuild that
    swaps the facts tree"), never the path, because the only reading of one of these
    paths inside a recurring task's instructions is "run this". Answer a hit here by
    rephrasing the task, not by widening the scan back over the Activity Log or
    dropping a needle — the log is excluded because #24's and #74's honest
    freeze-pause lines are in it, and the needles are the point.
    """
    assert AUTONOMY_DIR.is_dir(), f"{AUTONOMY_DIR} missing — the pin would pass vacuously"
    needles = ["export_json", "kg_rebuild.py", "kg_migrate_to_sqlite.py",
               "kg-migrate-to-sqlite"]
    offenders = []
    for path in sorted(AUTONOMY_DIR.glob("*.md")):
        text = _task_instructions(path.read_text(encoding="utf-8", errors="replace"))
        hits = [n for n in needles if n in text]
        if hits:
            offenders.append(f"{path.name}: {hits}")
    assert not offenders, f"recurring tasks told to run an exporter: {offenders}"


# ── clause 1: the file reflects a store, and only where it was told ─────────

def test_export_reflects_the_store_not_a_snapshot(tmp_path):
    """The one honest property of an export: its non-self mappings equal the
    store's `aliases` rows at the moment of writing — not a 2026-09-03 count."""
    from app import kg_store

    db_path = tmp_path / "kg.sqlite"
    kg_store.configure(db_path)
    try:
        st = kg_store.store()
        st.entities.register("Widget")
        st.entities.register("Gadget")
        for surface in ("widget", "the widget", "gad"):
            st.aliases.set(surface, "Widget" if "wid" in surface else "Gadget",
                           kind="semantic", origin="test")
        expected = sqlite3.connect(db_path).execute("select count(*) from aliases").fetchone()[0]
        assert expected == 3

        dest = tmp_path / "backups" / "pre-clean-20260912T000000Z"
        out = st.export_json(dest)
        written = json.loads(out["aliases"].read_text(encoding="utf-8"))
        non_self = {k: v for k, v in written.items() if k != v}
        assert len(non_self) == expected == st.aliases.count()
        assert non_self == {"widget": "Widget", "the widget": "Widget", "gad": "Gadget"}
        # self-identities are the exporter's addition for legacy readers, so the
        # file is larger than the table by exactly the entity count.
        assert len(written) == expected + 2
    finally:
        kg_store.reset()


def test_export_writes_only_inside_the_directory_it_is_given(tmp_path):
    """Nothing escapes the destination — the destination is the caller's promise
    that it is a snapshot directory, so an escape is a write into the live tree."""
    from app import kg_store

    kg_store.configure(tmp_path / "kg.sqlite")
    try:
        st = kg_store.store()
        st.entities.register("Widget")
        dest = tmp_path / "snap"
        out = st.export_json(dest)
        for path in out.values():
            assert Path(path).resolve().is_relative_to(dest.resolve()), path
        stray = sorted(p.relative_to(tmp_path).as_posix()
                       for p in tmp_path.rglob("entity-aliases.json")
                       if not p.resolve().is_relative_to(dest.resolve()))
        assert stray == []
    finally:
        kg_store.reset()


# ── clause 3: no code calls the JSON the alias source ───────────────────────

def test_no_code_calls_the_legacy_alias_json_the_source_of_truth():
    """The doc claim is the defect: `next-gen-memory/fact_extractor.py` told the
    nightly pipeline the JSON was authoritative, so a human reading the writer
    had no way to tell a snapshot from the table."""
    offenders = []
    for path in _py_files(needle="entity-aliases.json"):
        if path.relative_to(REPO).parts[0] == "tests":
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if "entity-aliases.json" not in line:
                continue
            window = "\n".join(lines[max(0, i - 2):i + 3])
            if not SOURCE_CLAIM.search(window):
                continue
            if RETIRES_CLAIM.search(window) and LIVES_HERE.search(window):
                continue
            offenders.append(f"{path.relative_to(REPO).as_posix()}:{i + 1}: {line.strip()}")
    assert not offenders, "code still asserts the legacy JSON is authoritative:\n" + "\n".join(offenders)


def test_fact_extractor_docstring_points_at_the_store():
    """The specific line the item missed. It must name the store, and must not
    survive as a comment that reads as current."""
    lines = (REPO / SCHEDULED_PIPELINE_DIR / "fact_extractor.py").read_text(encoding="utf-8").splitlines()
    at = [i for i, ln in enumerate(lines) if "entity-aliases.json" in ln]
    assert at, "the comment was deleted rather than corrected; the file's absence needs explaining"
    assert not any("source of truth" in lines[i].lower() for i in at), [lines[i] for i in at]
    # The mention has to point somewhere: the store named in the same comment block.
    for i in at:
        window = "\n".join(lines[max(0, i - 3):i + 3])
        assert "app.kg_store" in window or "aliases table" in window, lines[i]


def test_paths_comment_states_the_fate_chosen():
    """app/paths.py owns the constant, so it is where "what happened to this
    path" has to be answerable without opening the backlog."""
    lines = (REPO / "app" / "paths.py").read_text(encoding="utf-8").splitlines()
    at = next(i for i, ln in enumerate(lines) if ln.startswith("VAULT_FACTS_ALIASES"))
    block = "\n".join(lines[max(0, at - 14):at])
    # Checked for meaning, not wording: a reworded comment that still says the
    # same four things must not re-fail.
    assert LIVES_HERE.search(block), "the comment does not say what the live alias table is"
    assert re.search(r"not an export target|no longer (?:an |written here)|stopped (?:being|writing)",
                     block, re.I), "the comment does not say the export stopped at this path"
    assert re.search(r"_pipeline/backups|backups/|snapshot dir", block, re.I), \
        "the comment does not say where a snapshot goes instead"
    assert re.search(r"#474|item 474", block), "the comment does not name the item that chose the fate"


# ── clause 4: loaded memory, and the disk state it asserts ──────────────────
#
# USER.md is loaded into every system prompt, so its claim about this file is
# read as fact by every run. The half that can be pinned is the *coupling*: the
# sentence must be dated history, and the state it describes must be true. The
# first version of this round shipped the sentence while the 939 KB copy was
# still on disk — the review rung caught what reading my own diff could not,
# because a memory file and a data file never appear in the same `git status`.
#
# These read the live checkout and the live vault, not the worktree: a claim
# about production state is only falsifiable against production (precedent:
# test_automod_hardening reads the live skills tree).

LOADED_MEMORY = Path.home() / "obsidian" / "lloyd" / "USER.md"
LIVE_FACT_TREE = production_data_root() / "_pipeline" / "vault-derived" / "facts"
LIVE_BACKUPS = production_data_root() / "_pipeline" / "backups"
# A line counts as history if it is marked closed/retired or carries a date in
# either form this vault uses: `2026-09-03`, or the `08-22` bullet prefix the
# incident notes use. Loose by design — it asks "is this line standing on a
# date", not "is the date well-formed".
HISTORY_MARK = re.compile(r"CLOSED|history|retired|\d{4}-\d{2}-\d{2}|\b\d{1,2}-\d{1,2}\b")


def test_loaded_memory_cites_the_export_only_as_dated_history():
    """The 19,914 / 16,041 figures sat in loaded memory as a standing defect for
    nine days, and the SQLite row count on the same line had already drifted."""
    assert LOADED_MEMORY.is_file(), f"{LOADED_MEMORY} missing — the pin would pass vacuously"
    lines = LOADED_MEMORY.read_text(encoding="utf-8").splitlines()
    assert not [ln for ln in lines if "19,914" in ln], \
        "loaded memory still carries the frozen export's entry count"
    undated = [ln[:110] for ln in lines
               if "entity-aliases.json" in ln and not HISTORY_MARK.search(ln)]
    assert not undated, f"undated live-sounding claims in loaded memory: {undated}"


def test_the_state_loaded_memory_describes_is_true_on_disk():
    """USER.md now says the export no longer lives in the fact tree. That
    sentence is only honest while the file is actually absent — and this is the
    only thing in the repo that checks the two against each other."""
    assert LOADED_MEMORY.is_file(), f"{LOADED_MEMORY} missing — the pin would pass vacuously"
    text = LOADED_MEMORY.read_text(encoding="utf-8")
    claims_gone = re.search(r"no longer lives under|not in the live tree", text)
    live_copy = LIVE_FACT_TREE / "entity-aliases.json"
    if claims_gone:
        assert not live_copy.exists(), (
            f"loaded memory says {live_copy} is gone but it is on disk "
            f"({live_copy.stat().st_size} bytes, mtime {live_copy.stat().st_mtime_ns}); "
            "either move the file back into the tree or stop claiming it is gone")
    # "Stop emitting into the live tree" is a move, not a deletion: the migration
    # source of record still has to exist somewhere, or the next rebuild has no
    # pre-migration state to diff against.
    # The backup root itself is gitignored and went with the tree on 2026-09-22, so
    # "no snapshot here" no longer distinguishes a wrong fate from a deleted one.
    # Where the root exists the check still bites: a present root holding no
    # entity-aliases.json is still the migration losing its source of record.
    require_live_data(LIVE_BACKUPS, "the fact-tree backup root")
    snapshots = list(LIVE_BACKUPS.rglob("entity-aliases.json"))
    assert snapshots, (
        f"{live_copy} is gone from the fact tree but no snapshot exists under "
        f"{LIVE_BACKUPS} — the fate was 'move to backups', not 'delete'")


def test_constant_still_names_the_legacy_path_for_the_migration():
    """The migration's `--aliases` default and the characterization pin both
    resolve this name; retiring the constant is not the same as retiring the
    live-tree file, and quietly changing its value would be the worse bug."""
    from app.paths import VAULT_FACTS_ALIASES, VAULT_FACTS_ROOT
    assert VAULT_FACTS_ALIASES == VAULT_FACTS_ROOT / "entity-aliases.json"


# ── #2561 clause 3: the green came from the vault, not from a marker ────────

#: The four nodes the gate listed as failing at 0ae3064f, each with the file that
#: defines it. The first is this file's own; the three retention nodes are the
#: vault-count guards #2559 moved to eighteen, whose second name this list spells
#: as it reads on disk today (`…says_nineteen_stores…`, #2615's hold journal being
#: the store that moved it past #2559's `…says_eighteen_stores…`) rather than as
#: #2561's original filing spelled it (`…says_seventeen_stores…`), which no longer
#: exists. What the list holds is the four NODES, not one round's spelling.
_GREEN_AT_HEAD: dict[str, list[str]] = {
    "tests/test_legacy_alias_export_fate.py": [
        "test_no_recurring_autonomy_task_is_told_to_run_an_exporter",
    ],
    "tests/test_retention_sweep.py": [
        "test_the_skill_says_nineteen_stores_and_its_table_has_a_row_per_report_line",
        "test_the_table_guard_refuses_a_row_short_and_a_row_long",
        "test_the_task_description_names_every_store_the_sweep_prints",
    ],
}


def test_the_four_head_green_nodes_carry_no_skip_or_xfail_marker():
    """#2561 clause 3: the four named nodes exist, are collected, and carry no marker.

    The clause is not "they pass" — the run says that — it is *how* main got green: no
    test skipped, xfailed, deleted or re-marked `live_vault`. Read statically out of the
    two source files, so no environment can satisfy it. Three claims, each one a
    different way of faking green:

      * each named function still EXISTS in its file — a deleted test is the cheapest
        green on the box, and the file would still exit 0;
      * none carries a `skip`/`skipif`/`xfail` decorator, and neither file sets a
        module-level `pytestmark` naming one, which is the same thing one level up;
      * none skips UNCONDITIONALLY: a `skip(...)` call that is not nested inside an `if`
        is a marker written as a statement.

    What it deliberately does not forbid is a CONDITIONAL `pytest.skip` on an unreachable
    vault, which three of these four nodes have always had (`test_the_skill_says…`,
    `test_the_table_guard…`, `test_the_task_description…` each read
    `~/obsidian/skills/retention-sweep/SKILL.md` or `~/obsidian/autonomy/79-…md` and
    skip when the vault is away). That guard predates #2561 and is not "to get there";
    the run that settles whether it was taken is the acceptance run itself, which showed
    all four PASSED, not skipped — and clause 3's ban is on the marker, not on the vault
    being mounted.
    """
    forbidden = {"skip", "skipif", "xfail", "live_vault", "fault_injection"}
    found: list[str] = []
    problems: list[str] = []
    for rel, names in _GREEN_AT_HEAD.items():
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"), filename=rel)
        by_name = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for node in tree.body:  # module-level pytestmark = [pytest.mark.xfail, …]
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", "") == "pytestmark" for t in node.targets):
                words = {a.attr for a in ast.walk(node) if isinstance(a, ast.Attribute)}
                if words & forbidden:
                    problems.append(f"{rel}: module pytestmark names "
                                    f"{sorted(words & forbidden)}")
        for name in names:
            fn = by_name.get(name)
            if fn is None:
                problems.append(f"{rel}::{name} is GONE from the file")
                continue
            found.append(name)
            for dec in fn.decorator_list:
                words = {a.attr for a in ast.walk(dec) if isinstance(a, ast.Attribute)}
                if words & forbidden:
                    problems.append(f"{rel}::{name} carries a "
                                    f"{sorted(words & forbidden)} decorator")
            # Every skip-shaped call must sit under an `if`, or it always runs.
            guarded: set[int] = set()
            for outer in ast.walk(fn):
                if isinstance(outer, ast.If):
                    for inner in ast.walk(outer):
                        guarded.add(id(inner))
            for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
                words = {a.attr for a in ast.walk(call.func) if isinstance(a, ast.Attribute)}
                words |= {getattr(call.func, "id", "")}
                if words & {"skip", "skipif", "xfail"} and id(call) not in guarded:
                    problems.append(f"{rel}::{name} skips unconditionally")
    assert not problems, "clause 3 forbids these: " + "; ".join(problems)
    assert len(found) == 4, f"expected the four named nodes, found {len(found)}: {found}"
