"""What the research skills claim, checked against the machine.

Three of the four signal paths the generator skill named did not exist. Not
subtly: `~/obsidian/pending-research/` is not a directory, gap-fill has never
written a note, and the daily notes are at `~/obsidian/memory/`, not
`~/obsidian/lloyd/memory/`. The model that ran it worked that out at runtime,
said so in its report, and nobody read the report — so the skill kept sending
every run to look in three empty places for months.

A skill is a prompt with no compiler. This file is the compiler for the parts
that are checkable: does the path exist, does the tool exist, does the number
in the doc match the code.
"""

from __future__ import annotations

import ast
import asyncio
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
VAULT = Path.home() / "obsidian"

GENERATOR = VAULT / "skills" / "research-queue-generator" / "SKILL.md"
DEEP_DIVE = VAULT / "skills" / "deep-dive-research" / "SKILL.md"
DOC = ROOT / "architecture" / "research-pipeline.md"

#: A backticked path in a skill, `~`-anchored. Globs are resolved as "at least
#: one match", because these name a dated directory or file per run.
_PATH_RE = re.compile(r"`(~/[A-Za-z0-9_./*<>-]+)`")


def _skill_paths(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    out = []
    for raw in _PATH_RE.findall(text):
        # `<last 3 days>` is prose inside a path and `YYYY-MM-DD` is a date
        # template; both mean "one of these per run", so both become a glob.
        cleaned = re.sub(r"<[^>]*>", "*", raw)
        cleaned = cleaned.replace("YYYY-MM-DD", "*")
        out.append(cleaned)
    return out


#: Roots whose contents a job writes and rewrites. A doc naming something under one
#: of these is naming an output, not a file in the checkout, so its absence is a
#: statement about the machine. All four are gitignored; `_pipeline/` and `sessions/`
#: were emptied by the 2026-09-22 tree deletion and are refilled by the jobs that
#: produce them, so a hard failure here is a red node at base in every round for a
#: path no commit broke.
RUNTIME_ROOTS = ("_pipeline/", "sessions/", "logs/", "data/")


def _is_runtime_absence(spec: str) -> bool:
    """True when `spec` names a job's output under a root that holds none yet.

    Deliberately narrow: it does not ask whether the path exists (the caller already
    knows it does not), only whether the thing named is regenerable output. A typo in
    a `scripts/` or `app/` path is still drift and still fails.
    """
    rel = spec.replace("~/lloyd/", "", 1).replace(f"{Path.home()}/lloyd/", "", 1)
    return rel.startswith(RUNTIME_ROOTS)


def _resolves(spec: str) -> bool:
    p = Path(spec.replace("~", str(Path.home()), 1))
    if "*" not in spec:
        return p.exists()
    # Walk down to the first globbed segment and match from there.
    parts = p.parts
    for i, part in enumerate(parts):
        if "*" in part:
            base = Path(*parts[:i])
            pattern = str(Path(*parts[i:]))
            return base.exists() and any(base.glob(pattern))
    return p.exists()


@pytest.mark.parametrize("skill", [GENERATOR, DEEP_DIVE], ids=["generator", "deep-dive"])
def test_the_skill_exists(skill):
    assert skill.exists(), f"{skill} is missing"


@pytest.mark.parametrize("skill", [GENERATOR, DEEP_DIVE], ids=["generator", "deep-dive"])
def test_every_path_a_skill_names_resolves(skill):
    """The regression: three of four signal paths pointed at nothing."""
    missing = [spec for spec in _skill_paths(skill) if not _resolves(spec)]
    drift = [spec for spec in missing if not _is_runtime_absence(spec)]
    assert not drift, (
        f"{skill.name} names paths that do not exist: {drift}. A skill is a "
        f"prompt with no compiler; this is the compiler.")
    if missing:
        pytest.skip(
            f"{skill.name} names only regenerable output that this machine holds "
            f"none of yet: {missing}. The 2026-09-22 deletion emptied these roots "
            "and the producing jobs refill them; nothing here is a wrong claim.")


def test_the_generator_names_the_four_live_signals():
    text = GENERATOR.read_text(encoding="utf-8")
    assert "knowledge-health-" in text
    assert "session-distill" in text and "## Gaps" in text
    assert "backlog_tasks" in text
    assert "obsidian/memory/" in text
    assert "gap-fill" not in text, "gap-fill has never produced a note"


def test_neither_skill_reads_the_retired_checklist():
    """It is renamed and imported; a skill still reading it would find an
    archive header and 3,690 lines of history."""
    for skill in (GENERATOR, DEEP_DIVE):
        text = skill.read_text(encoding="utf-8")
        assert "research-queue.md" not in text.replace("research-queue-archive.md", ""), \
            f"{skill.name} still reads the retired checklist"


def test_the_retired_checklist_is_archived_and_unreferenced_by_code():
    assert not (VAULT / "lloyd" / "research-queue.md").exists()
    archive = VAULT / "lloyd" / "research-queue-archive.md"
    assert archive.exists() and "Archived" in archive.read_text(encoding="utf-8")[:400]

    # Code, not prose: several docstrings still recount what reading that file
    # cost, and that history is worth keeping. What must not survive is a
    # module that still opens it.
    for py in (ROOT / "workers").rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef))
            and node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        live = [n for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and id(n) not in docstrings and "research-queue.md" in n.value]
        assert not live, f"{py} still reads the retired checklist"


def test_every_tool_the_skills_call_is_advertised():
    """A skill naming a tool that does not exist is a run that fails at the
    moment it matters."""
    import agent_mcp.main as M

    real = {t.name for t in asyncio.run(M.list_tools())}
    named = set()
    for skill in (GENERATOR, DEEP_DIVE):
        text = skill.read_text(encoding="utf-8")
        named |= set(re.findall(r"\b(research_[a-z_]+|backlog_tasks|vault_recall|"
                                r"vault_write|fact_add)\b", text))
    assert named, "the skills name no tools at all — did they get rewritten?"
    assert named <= real, f"skills name tools that do not exist: {sorted(named - real)}"


def test_the_deep_dive_skill_carries_the_contract_the_worker_parses():
    text = DEEP_DIVE.read_text(encoding="utf-8")
    assert "RESULT:" in text
    for verb in ("written", "nothing_found", "duplicate"):
        assert verb in text, verb
    assert "source: deep-research" in text, "the note's frontmatter names the source"
    assert "topic_id:" in text
    assert "date +%F" not in text, (
        "the source supplies the date and the path; that bash call is why past "
        "runs produced notes misdated by days")


def test_the_deep_dive_skill_does_not_pick_its_own_topic():
    text = DEEP_DIVE.read_text(encoding="utf-8")
    assert "The topic is given to you" in text
    assert "fallback" not in text.lower()


def test_the_generator_acts_on_similar_rather_than_reading_it():
    """Without this rule the model treats `similar` as informational and
    proposes the reword anyway — which is how the old queue accumulated the
    same 82 topics 390 times."""
    text = GENERATOR.read_text(encoding="utf-8")
    assert "similar" in text
    assert "Act on `similar`" in text


def test_the_generator_checks_the_code_before_calling_something_broken():
    """On 2026-09-07 it proposed the TTS EQ and speed control as open defects.
    Both were implemented and configured."""
    text = GENERATOR.read_text(encoding="utf-8")
    assert "Grep" in text and "before" in text
    assert (ROOT / "agent-services" / "tts_shaping.py").exists(), \
        "the example the skill cites must stay true"


# ---------------------------------------------------------------------------
# Task files and config
# ---------------------------------------------------------------------------


def test_the_generator_task_has_room_to_finish():
    """900s auto-disabled it on 2026-09-04 after three consecutive timeouts."""
    import autonomy

    task = autonomy._parse_task_file(autonomy._find_task_file(65))
    assert task and task["skill_name"] == "research-queue-generator"
    assert int(task["timeout_seconds"]) >= 1500

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    pool_cap = cfg["workers"]["sources"]["scheduled-task"]["max_duration_seconds"]
    assert int(task["timeout_seconds"]) < pool_cap - 30, (
        "the task's own timer must fire before the pool's, or the pool cancels "
        "it and no run record is written at all")


def test_the_generator_description_is_what_the_model_actually_reads():
    """`_build_task_prompt` renders the skill and the `description` field, and
    never the markdown body — so a stale description is a stale prompt."""
    import autonomy

    task = autonomy._parse_task_file(autonomy._find_task_file(65))
    desc = task["description"]
    assert "research_propose" in desc or "registry" in desc
    assert "research-queue.md" not in desc


def test_the_deep_dive_task_is_retired_not_still_scheduled():
    """It runs as the `deep-research` worker source now. Two schedulers for one
    skill would research two topics a day and record one."""
    import autonomy

    assert autonomy._find_task_file(52) is None, "#52 is still dispatchable"
    archived = VAULT / "autonomy" / "_archived" / "52-deep-dive-research.md"
    assert archived.exists()
    task = autonomy._parse_task_file(archived)
    assert task["status"] == "paused"
    assert "archived_reason" in task


def test_the_retired_source_is_gone_from_config_and_the_registry():
    import workers.sources as sources

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    names = cfg["workers"]["sources"]
    assert "domain-research" not in names
    assert "deep-research" in names
    assert "domain-research" not in sources.SOURCE_REGISTRY
    assert "deep-research" in sources.SOURCE_REGISTRY


def test_the_promotion_default_for_the_old_staging_dir_survives():
    """142 `domain-research` notes are still on disk under pending-research/,
    and the Review tab promotes them by directory name."""
    import app.routers.workers as W

    staged = ROOT / "_pipeline" / "vault-derived" / "pending-research" / "domain-research"
    if staged.exists():
        assert "domain-research" in W._DEFAULT_DEST, (
            "removing this makes the leftover notes unpromotable")


# ---------------------------------------------------------------------------
# The architecture doc
# ---------------------------------------------------------------------------


def test_the_doc_exists_and_keeps_its_numbers():
    assert DOC.exists()
    text = DOC.read_text(encoding="utf-8")
    for claim in ("314", "2,839", "research.db"):
        assert claim in text, claim


def test_the_doc_names_the_states_the_store_has():
    from app import research_store

    text = DOC.read_text(encoding="utf-8")
    for status in research_store.STATUSES:
        assert status in text, f"{status} is undocumented"


def test_the_written_row_names_finish_as_the_verifier():
    """§2's `written` row claimed the disk check lives in the worker and *not*
    in `finish` (#1276). It was true when written; #1276 moved the check into
    the store, so the sentence that says it is absent is now the defect — the
    next reader would trust it and go looking for the hole in the wrong place.

    Pinned in both directions: the row must name `finish` as the verifier, and
    no sentence may still assert `finish` does not check disk.
    """
    from app import research_store

    text = DOC.read_text(encoding="utf-8")
    row = next(line for line in text.splitlines()
               if line.startswith("| `written` |"))
    assert "`finish`" in row, "the row must name the method that verifies"
    assert "_require_real_note" in row, "and the helper that does it"
    assert str(research_store.MIN_NOTE_BYTES) in row, (
        "the byte floor the row quotes must be the one the code enforces")
    for line in text.splitlines():
        if re.search(r"finish`? itself does not", line) or \
           re.search(r"not in `finish`", line):
            raise AssertionError(
                f"a sentence still says finish does no disk check: {line.strip()[:100]}")


# ---------------------------------------------------------------------------
# Worker-source docstrings (#705)
# ---------------------------------------------------------------------------

WORKER_SOURCES = ROOT / "workers" / "sources"

#: The staging root as a checkout-relative path, spelled the way a docstring
#: spells it. Derived from `app.paths` rather than written as a literal so the
#: doc can never drift from the constant the writer uses.
STAGING_CHECKOUT_REL = "lloyd/_pipeline/vault-derived/pending-research"

#: A `~`-anchored path in a worker-source docstring, backticked or not.
#: `_PATH_RE` above cannot serve here for two reasons: it requires a backtick
#: on each side, and these modules name paths in RST double-backticks and in
#: bare prose alike — `_common.py` carried `~/obsidian/pending-research/` in
#: bare prose, which is the exact string that poisoned #522's acceptance
#: check; and its class has no braces, so
#: `…/pending-research/{source}/{yyyy-mm-dd}/` would be cut at the first brace
#: and the parent directory would resolve for the wrong reason.
_DOC_PATH_RE = re.compile(r"~/[A-Za-z0-9_./*<>{}-]+")

#: A staging leaf named without its root: `pending-research/gaps/`. Each of the
#: three staging sources names one, and #705 found all three wrong —
#: `distill/`, `gaps/` and `bench/` against the real `session-distill/`,
#: `gap-fill/` and `bench-mine/`. A guard that compiles only the root passes on
#: every one of those, which is why the leaf is extracted separately.
_STAGING_LEAF_RE = re.compile(r"pending-research/(?P<leaf>\{[^}]+\}|[A-Za-z0-9_.-]+)")

#: Prose punctuation glued to the end of a path is not part of the path.
_PATH_TRAILING = ".,;:`'\" "


def _doc_claims(py: Path) -> dict:
    """What one module's OWN docstring asserts about the filesystem.

    Only the module docstring: that is the surface #705 is about — the text a
    triage or selfmod run reads to decide where to go looking. `NAME` comes
    from the source text because it is the leaf `write_staging_note` fixes.
    """
    src = py.read_text(encoding="utf-8")
    doc = ast.get_docstring(ast.parse(src)) or ""
    name = re.search(r'^NAME\s*=\s*["\']([^"\']+)["\']', src, re.M)
    return {
        "name": name.group(1) if name else None,
        "paths": [p.rstrip(_PATH_TRAILING) for p in _DOC_PATH_RE.findall(doc)],
        "leaves": _STAGING_LEAF_RE.findall(doc),
    }


def _doc_resolves(spec: str) -> bool:
    """Resolve a docstring path; `{yyyy-mm-dd}` and `<…>` mean "at least one".

    The same rule `_skill_paths` applies to a skill's `<last 3 days>`, and it
    has to be "at least one" rather than "exists": these are dated directories,
    written one per run, so no literal path is ever the whole claim.
    """
    cleaned = re.sub(r"\{[^}]*\}", "*", spec)
    cleaned = re.sub(r"<[^>]*>", "*", cleaned).rstrip("/")
    return _resolves(cleaned)


def test_every_path_a_worker_source_docstring_names_resolves():
    """The regression: `~/obsidian/pending-research/` is not a directory.

    Every knowledge-acquisition source documented its output landing there, so
    a run that went to look got `No such file or directory` and wrote it down
    as proof nothing had been staged. #522's triage turned that into an
    acceptance clause — `≥6 staged bench task files exist under
    ~/obsidian/pending-research/bench/{yyyy-mm-dd}/` — unsatisfiable by any
    diff, because no source in the tree writes there, and three run records
    under `autonomy-runs/65/` (`run_65_20260901_174920`, `run_65_20260906_181631`,
    `run_65_20260907_221114`) each name the dead path as a finding nobody read.
    The root the code uses is `app.paths.VAULT_PENDING_RESEARCH_DIR`.

    Four kinds of failure, each with a denominator beside it so a guard that
    has stopped matching cannot report clean on an empty list:

      * the literal string `obsidian/pending-research` in any worker source,
        over the whole module and not just the docstring, which is what the
        item's own grep looks for and what nothing below can see;
      * every `~/…` path must exist, and sit under the root the code uses — a
        wrong ROOT;
      * a bare `pending-research/<leaf>/` must be that module's own `NAME` — a
        wrong LEAF, which a root-only check waves through;
      * both extractors must actually extract, one from prose that carries no
        backticks and one from a leaf that carries no root.
    """
    # First the string itself, over the WHOLE module source rather than the
    # docstring, because that is the check #705's acceptance is written as
    # (`grep -rn "obsidian/pending-research" workers/sources/*.py` is empty).
    # Everything below is anchored on `~/`, so a mention that drops the tilde
    # — `under obsidian/pending-research/` — would sail past every path
    # assertion in this file while leaving the grep red.
    scanned = sorted(WORKER_SOURCES.glob("*.py"))
    assert scanned, f"no worker sources found under {WORKER_SOURCES}"
    dead_root = sorted(p.name for p in scanned
                       if "obsidian/pending-research" in p.read_text(encoding="utf-8"))
    assert not dead_root, (
        f"these worker sources still name the vault copy of `pending-research/`, "
        f"which is not a directory and never was ({dead_root} of "
        f"{len(scanned)} files scanned). Any run that navigates by it gets "
        f"`No such file or directory` and reads that as an empty staging area.")

    claims = {p.name: _doc_claims(p) for p in scanned}
    speaking = {m: c for m, c in claims.items() if c["paths"] or c["leaves"]}
    assert len(speaking) >= 3, (
        f"the guard extracted path claims from {len(speaking)} of "
        f"{len(claims)} worker-source docstrings ({sorted(speaking)}); it must "
        f"find at least 3, or an extractor regression reads as a clean tree")

    missing = sorted({(m, spec) for m, c in speaking.items() for spec in c["paths"]
                      if not _doc_resolves(spec)})
    drift = [(m, spec) for m, spec in missing if not _is_runtime_absence(spec)]
    assert not drift, (
        f"worker-source docstrings name paths that do not exist: {drift}. A "
        f"module docstring is the map a triage run navigates by, and one wrong "
        f"path cost #522 a whole unsatisfiable acceptance clause.")
    if missing:
        pytest.skip(
            f"the only unresolved docstring paths name regenerable output this "
            f"machine holds none of yet: {missing}")

    from app.paths import LLOYD_HOME, VAULT_PENDING_RESEARCH_DIR
    import app.routers.workers as W

    assert W.PENDING_ROOT == VAULT_PENDING_RESEARCH_DIR, (
        "the Review tab lists a different root than app.paths names, so the "
        "root below is not the surface a human promotes from")
    root_rel = VAULT_PENDING_RESEARCH_DIR.relative_to(LLOYD_HOME)
    assert f"lloyd/{root_rel.as_posix()}" == STAGING_CHECKOUT_REL, (
        f"app.paths moved the staging root to {root_rel}; the docstrings and "
        f"this guard's spelling have to move with it")

    off_root = []
    for m, c in speaking.items():
        for spec in c["paths"]:
            if "pending-research" not in spec:
                continue
            parts = [p for p in spec[len("~/"):].split("/") if p]
            lead = tuple(parts[1:1 + len(root_rel.parts)])
            if parts[:1] != ["lloyd"] or lead != root_rel.parts:
                off_root.append((m, spec, f"~/{STAGING_CHECKOUT_REL}"))
    assert not off_root, (
        f"these docstrings name a pending-research root that is not "
        f"app.paths.VAULT_PENDING_RESEARCH_DIR (checkout-relative "
        f"{root_rel}): {off_root}")

    # Two non-vacuity pins for the extractors themselves, because each half of
    # #705 is a half a narrower guard cannot see: the poisoned line in
    # `_common.py` was bare prose (a backtick-requiring regex like `_PATH_RE`
    # extracts nothing from it), and `distill/`, `gaps/`, `bench/` were leaves,
    # which a root-only check waves through. A guard that quietly stops
    # matching one of these reports green on an empty list.
    assert f"~/{STAGING_CHECKOUT_REL}/{{source}}/{{yyyy-mm-dd}}/" in _doc_claims(
        WORKER_SOURCES / "_common.py")["paths"], (
        "the extractor only matches backticked paths, and the line that "
        "poisoned #522's acceptance check was bare prose — which is why this "
        "guard does not reuse `_PATH_RE`")
    staging = {m: c["name"] for m, c in speaking.items() if c["name"] and c["leaves"]}
    assert len(staging) >= 3, (
        f"only {len(staging)} worker-source docstrings named a "
        f"pending-research leaf ({sorted(staging)}); the three staging sources "
        f"all do, so the leaf extractor has stopped matching and a wrong leaf "
        f"would sail through while the root check stayed green")

    wrong_leaf = []
    for m, c in speaking.items():
        for leaf in c["leaves"]:
            if leaf.startswith("{"):
                continue  # a template, not a claim about one directory
            if leaf != c["name"]:
                wrong_leaf.append((m, leaf, c["name"]))
    assert not wrong_leaf, (
        f"these docstrings stage under a leaf that is not the module's own "
        f"NAME, while `write_staging_note` fixes the directory to NAME: "
        f"{wrong_leaf}. `distill/`, `gaps/` and `bench/` were all of these.")

    step3 = [ln for ln in ast.get_docstring(
        ast.parse((WORKER_SOURCES / "_common.py").read_text(encoding="utf-8"))
    ).splitlines() if "lands under" in ln]
    assert len(step3) == 1, f"step 3 of the shared pattern is not stated once: {step3}"
    assert f" ~/{STAGING_CHECKOUT_REL}/{{source}}/{{yyyy-mm-dd}}/ " in f" {step3[0]} ", (
        f"step 3 must name the checkout's own staging root with both date and "
        f"source templates, i.e. `~/{STAGING_CHECKOUT_REL}/"
        f"{{source}}/{{yyyy-mm-dd}}/`, got {step3[0]!r}")


def test_the_staging_leaf_is_the_directory_a_note_actually_lands_in(tmp_path, monkeypatch):
    """The leaf assertion above compares a docstring to a constant. This one
    compares the same promise to the only writer that makes the directory.

    Pointing `STAGING_ROOT` at a tmp dir and calling `write_staging_note`
    proves the layout the three docstrings now claim — `<root>/<NAME>/<date>/`
    — is what production code produces, and so is what
    `GET /api/workers/pending` reads back as the source name
    (`src.relative_to(PENDING_ROOT).parts[0]`). Without it `leaf == NAME`
    would only be two files agreeing with each other.
    """
    import workers.sources._common as C
    import workers.sources.bench_mine as bench_mine
    import workers.sources.gap_fill as gap_fill
    import workers.sources.session_distill as session_distill

    monkeypatch.setattr(C, "STAGING_ROOT", tmp_path)
    for mod in (bench_mine, gap_fill, session_distill):
        note = C.write_staging_note(source=mod.NAME, slug="probe", body="body")
        rel = note.relative_to(tmp_path)
        assert len(rel.parts) == 3, f"{mod.__name__}: expected root/NAME/date/note.md, got {rel}"
        assert rel.parts[0] == mod.NAME, f"{mod.__name__}: leaf is {rel.parts[0]}, not NAME"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", rel.parts[1]), (
            f"{mod.__name__}: the date directory is {rel.parts[1]!r}, not the "
            f"{{yyyy-mm-dd}} the docstrings promise")
        assert rel.parts[2] == f"{note.stem.rsplit('-', 1)[0]}-probe.md"
        assert note.read_text(encoding="utf-8").startswith("---\n"), (
            "a staged note without frontmatter is unpromotable: the Review tab "
            "reads review_status and source out of it")


def test_gap_fill_documents_the_staging_step_not_a_fact_write_it_never_makes():
    """#705's second half, and it sits two lines under the wrong path.

    The docstring said the handler "and (at high confidence) updates the fact".
    `gap_fill.execute` researches, calls `write_staging_note`, and returns: no
    path through that module writes a fact, and the whole point of the staging
    step is that a human promotes the note. A reader who believed the sentence
    would look for a confidence threshold that does not exist, and a reader of
    the facts tree would look for a `resolved_at` this source never sets.
    """
    src = (WORKER_SOURCES / "gap_fill.py").read_text(encoding="utf-8")
    doc = (ast.get_docstring(ast.parse(src)) or "").lower()

    for claim in ("updates the fact", "update the fact", "writes the fact",
                  "wrote the fact"):
        assert claim not in doc, f"gap_fill's docstring still promises a fact write: {claim!r}"
    assert "promot" in doc and "human" in doc, (
        "the docstring must say where the resolution note actually goes: "
        "staged under the source's own leaf for a human to promote")

    # The prose is only half of it: the claim fails again the moment the module
    # grows a fact writer, so pin the behaviour side too.
    for writer in ("fact_add", "fact_invalidate", "kg_store", "remember("):
        assert writer not in src, (
            f"gap_fill now uses {writer}; then the docstring is allowed to "
            f"mention a fact write again, and this test is the one to change")
