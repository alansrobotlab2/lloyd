"""#3907 clause 1 and clause 3: the entity-vector artifact has one status command,
and the tree says what is true about it.

The defect this pins is the absence of a reading. `vectors.npy` sat at 11,990 rows /
42.9 days against 16,822 live entities while `config.yaml` advertised the semantic leg
as though it were *inert until the index exists* — a sentence that stopped being true
the day the index was written — and nothing in the tree could say the age or the
coverage out loud. `app/entity_linker.py:84-100` read the mtime for one purpose only:
invalidating its own cache. So the failure mode of an unbounded-age artifact was not
an error, a log or a degraded metric; it was a healthy-looking retrieval whose seeds
came from an index 4,851 live entities were never in.

Clause 1 is the command. Clause 3 is the prose: `config.yaml` states the disposition
actually taken, and no architecture doc may claim a schedule the tree does not have.
#4021 did not relax that clause, it moved which state the prose has to describe: autonomy
job **#111** is now the runner, so the same check refuses a doc that names a cadence
without naming the job, and refuses the sentence it used to *require* ("the job that would
run it is NOT armed") just as hard. `test_no_architecture_doc_names_a_rebuild_schedule_
this_tree_does_not_have` is therefore gone and
`test_the_armed_runner_is_the_one_the_prose_names_and_no_doc_still_says_unarmed` is what
it reached for, run against seeded copies so the deny-list has to prove it bites.

What #4021 added on top of the prose is a job, and a job has two surfaces. The repo one is
`test_the_rail_that_let_this_job_arm_refuses_the_same_commit_moving_a_schedule_key`, which
exercises `app.harness.policy.first_arm_allowed` with this task's own create/arm maps —
that is the machine that makes "the arm commit moved `status` alone" a necessity rather
than a habit. The vault one is the two `@live_vault` nodes at the end of this file, which
read `~/obsidian/autonomy/111-entity-vectors-rebuild.md` and the skill that names it. They
are marked rather than unmarked because `pytest.ini:15-18` says what the mark is for: the
gate runs `-m "not live_vault"`, since a round cannot control what the vault looks like on
the day it runs. The consequence is stated where it belongs, in those nodes' docstrings:
the gate does not execute them, so they are run by hand, and the repo nodes grade only the
numbers the repo itself publishes.

Nothing here touches the live artifact or the live store: every case builds its own
`names.json` + `vectors.npy` under `tmp_path` and points `app.kg_store.store` at a
fixture store. The dated readings — #3907's 2026-11-07T22:19Z (11,990 rows, 42.91 days)
and #4021's 2026-11-09T21:59Z (11,990 rows, 44.90 days, 70.4%, exit 1) — are quoted in
`config.yaml` and in `architecture/retrieval.md`, not asserted against the disk here,
because a test that reads the production artifact goes red or green on a rebuild. What
is asserted is that the prose quotes a re-run reading and not a stale one.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import entity_linker as EL                        # noqa: E402
from app import kg_store as ks                             # noqa: E402
from app.harness import policy as POL                      # noqa: E402
from scripts.memory import build_entity_vectors as BUILD   # noqa: E402

# The marker `pytest.ini` registers for a node that reads the live ~/obsidian vault, which
# no round under test controls and which the gate therefore deselects. Same alias
# tests/test_autonomy_jobs_doc_claims.py:104 uses, so the two files mark alike.
live_vault = pytest.mark.live_vault


def _front_matter(text: str) -> dict:
    # The YAML front matter of a vault note, with no vault-specific logic in it. The
    # vault's own loaders are the authority on its bytes (`app/autonomy.py` reads task
    # files, `scripts/autonomy/validate_tasks.py` validates them), and a test that
    # re-implements a parse risks disagreeing with the engine about what a file says.
    # That risk is taken for the handful of scalar keys these two nodes grade, on the
    # grounds that `yaml.safe_load` is the same call the validator makes.
    parts = text.split("---", 2)
    assert parts[0].strip() == "" and len(parts) >= 3, (
        "the note must open with a YAML front-matter block")
    return yaml.safe_load(parts[1]) or {}


DIM = 4


def _index(d: Path, names: list[str], *, age_days: float = 0.0,
           build_json: dict | None = None) -> Path:
    """An artifact with `names` rows, `age_days` old, provenance only if given.

    `age_days` is set on `vectors.npy` because that is the file both the loader's
    cache rule and `artifact_status` measure; `names.json`'s mtime is what the
    builder happens to write it with, and is not the bound's input.
    """
    d.mkdir(parents=True, exist_ok=True)
    (d / "names.json").write_text(json.dumps(names))
    (d / "vectors.npy").write_bytes(
        np.zeros((len(names), DIM), dtype=np.float32).tobytes())
    ts = time.time() - age_days * 86400.0
    (d / "vectors.npy").touch()
    import os
    os.utime(d / "vectors.npy", (ts, ts))
    if build_json is not None:
        (d / "build.json").write_text(json.dumps(build_json))
    return d


class _StubStore:
    """The `entities` table the coverage figure is measured against."""

    def __init__(self, names):
        self._names = list(names)

    def _query(self, sql, params=()):
        assert "entities" in sql, f"coverage must read the entity table, got {sql!r}"
        return [{"name": n} for n in self._names]


def _live_names(monkeypatch, names):
    """Point the coverage read at `names`, not at the live KG.

    A helper rather than a fixture because the count IS the fixture's subject: each
    node states how many live entities its coverage arithmetic is against, in the
    same place it states the expected percentage. A fixture with an implicit name list
    would let the two drift apart, and a wrong denominator is the defect this whole
    item is about.
    """
    monkeypatch.setattr(ks, "store", lambda: _StubStore(names))
    return list(names)


def test_status_reports_rows_age_live_coverage_and_exits_one_past_the_bound(
        tmp_path, monkeypatch, capsys):
    """Clause 1. The four figures the item could not obtain, and a nonzero exit.

    The fixture is the item's own shape scaled down: an index of 4 rows over a store
    holding 6 entities, two of which the index has never seen — the same arithmetic
    that on 2026-11-07 produced 11,990 rows, 16,822 live entities and 71.2%.
    """
    _live_names(monkeypatch, ["a", "b", "c", "d", "e", "f"])
    d = _index(tmp_path / "ev", ["a", "b", "c", "d"], age_days=EL.MAX_INDEX_AGE_DAYS + 1)
    rc = BUILD.status(d)
    out = capsys.readouterr().out

    assert rc == BUILD.EXIT_STALE == 1, (
        f"a 42-day-style artifact must EXIT nonzero or a scheduled gate cannot see it: "
        f"rc={rc}\n{out}")
    assert "rows: 4" in out, out
    assert f"age {EL.MAX_INDEX_AGE_DAYS + 1.0:.2f} days" in out, out
    assert "live entities: 6" in out, out
    # 4 index names, all 6 live entities named, so 4/6 = 66.7%. Stated as a fraction
    # in the same line so a reader cannot read the percentage without its denominator.
    assert "coverage: 66.7% (4 of 6 live entities have a vector" in out, out
    # The gap is stated as its own clause, not folded into the percentage: 4/6 is the
    # figure, "2 cannot be seeded semantically" is what the figure costs the leg.
    assert "2 cannot be seeded semantically" in out, out


def test_a_fresh_artifact_exits_zero_and_the_bound_is_one_named_constant_both_read(
        tmp_path, monkeypatch, capsys):
    """The bound is a constant, not a tuned number inlined at the call site.

    Clause 5 of the item forbids a percentile or tuned constant in a RESOLUTION path;
    this is not one — it decides an exit code and a log line, and no seed or score.
    What makes that safe is that there is exactly one value: the command, the
    consumer's `stale` flag and the printed bound all read
    `app.entity_linker.MAX_INDEX_AGE_DAYS`.
    """
    _live_names(monkeypatch, ["a"])
    fresh = _index(tmp_path / "fresh", ["a"], age_days=1.0)
    assert BUILD.status(fresh) == BUILD.EXIT_FRESH == 0
    line = capsys.readouterr().out
    assert f"(bound {EL.MAX_INDEX_AGE_DAYS} days)" in line, line

    # Move the one constant and both readers move with it. This is the check that the
    # bound is not duplicated somewhere the change cannot reach.
    monkeypatch.setattr(EL, "MAX_INDEX_AGE_DAYS", 0.5)
    assert BUILD.status(fresh) == BUILD.EXIT_STALE == 1
    st = EL.artifact_status(fresh)
    assert st["stale"] is True and st["max_age_days"] == 0.5, st


def test_an_absent_build_json_is_reported_as_absent_provenance_not_assumed(
        tmp_path, monkeypatch, capsys):
    """Clause 1's provenance half, and the one case this artifact is actually in.

    The builder has written `build.json` since its only commit `dbfde750`, and the
    production artifact does not have one — so which store those 11,990 rows came from
    is unrecoverable. The honest reading is to say so; the tempting one is to infer a
    build-time denominator out of the live count, which is the exact class of error that
    produced the figure #3907 retired from the complement probe — a number attributed to
    a population its writer never measured. `--status` therefore prints the `built
    against` figure ONLY when build.json supplies it.
    """
    _live_names(monkeypatch, ["a", "b", "z"])   # z is live but unindexed
    d = _index(tmp_path / "no-prov", ["a", "b"], age_days=EL.MAX_INDEX_AGE_DAYS + 2)
    BUILD.status(d)
    out = capsys.readouterr().out
    assert "provenance: build.json ABSENT" in out, out
    assert "unrecoverable" in out, out
    assert "built against" not in out, (
        f"with no build.json there is no build-time denominator, so the command must "
        f"refuse to print one rather than derive it from names_not_live:\n{out}")


def test_a_build_json_present_is_quoted_verbatim_including_its_own_row_count(
        tmp_path, monkeypatch, capsys):
    """The other half: when provenance exists, its figures are copied, not recomputed.

    `n: 11990` is what the builder recorded. `build.json` can also disagree with the
    matrix, and this node pins that both are shown — a provenance record silently
    overwritten by the live row count would hide exactly the drift #3907 is about.
    """
    _live_names(monkeypatch, ["a", "b", "c"])
    d = _index(tmp_path / "prov", ["a", "b", "c"], age_days=0.0,
               build_json={"built_at": "2026-09-26T00:22:22Z", "n": 11990,
                           "model": "Qwen3-Embedding-0.6B", "store": "/x/kg.sqlite"})
    assert BUILD.status(d) == BUILD.EXIT_FRESH == 0
    out = capsys.readouterr().out
    assert "built_at=2026-09-26T00:22:22Z" in out, out
    assert "n=11990" in out, out
    assert "model=Qwen3-Embedding-0.6B" in out, out
    assert "built against: 11990 entities" in out, out
    assert "rows: 3" in out, out      # the matrix's own count stays separate


def test_a_missing_artifact_says_absent_and_exits_two(tmp_path, monkeypatch, capsys):
    """No artifact is a different answer from a stale one, and it exits differently.

    This is the ONLY state in which `config.yaml`'s retired sentence was true: with no
    `vectors.npy`, `semantic_candidates` really does return [] and recall is lexical.
    Keeping it a distinct exit code is what lets a reader tell "inert" from "rotten",
    which the old comment collapsed.
    """
    rc = BUILD.status(tmp_path / "nothing" / "entity-vectors")
    out = capsys.readouterr().out
    assert rc == BUILD.EXIT_ABSENT == 2, f"rc={rc}\n{out}"
    assert "ABSENT" in out and "inert" in out, out


def test_an_unreadable_store_reports_coverage_as_unknown_and_still_judges_the_age(
        tmp_path, monkeypatch, capsys):
    """Coverage needs the store; age does not, and the two must not be entangled.

    `_live_entity_names` returning None used to be indistinguishable from "0 covered",
    which would have made the new figure lie in the direction of health. Here the
    store raises: coverage cannot be stated and prints `n/a`, while rows and age still
    print and the STATUS line still says STALE — those two facts need no store. What the
    unreadable half does change is the exit code, so a caller cannot read "it printed a
    verdict" as "it saw the population".
    """
    def boom():
        raise ks.StoreUnavailable("simulated: no database at the resolved path")
    monkeypatch.setattr(ks, "store", boom)
    d = _index(tmp_path / "nostore", ["a"], age_days=EL.MAX_INDEX_AGE_DAYS + 3)
    rc = BUILD.status(d)
    out = capsys.readouterr().out
    assert "STALE" in out.splitlines()[0], out
    assert "live entities: UNKNOWN" in out, out
    assert "coverage: n/a" in out, out
    assert "coverage: 0" not in out, (
        f"an unreadable store must never render as zero coverage:\n{out}")
    # EXIT_STORE, not EXIT_STALE: the age verdict still PRINTED (the line above says
    # STALE, and rows/age need no store), but a caller that acts on the artifact has to
    # know it could not see the population it is serving. An exit code is a contract,
    # so the code names the blocked half rather than the judged half.
    assert rc == BUILD.EXIT_STORE == 3, f"rc={rc}\n{out}"
    assert "rows and age above still stand" in out, out


# ── clause 3: the prose ─────────────────────────────────────────────────────

CONFIG = (ROOT / "config.yaml").read_text(encoding="utf-8")
RETRIEVAL_DOC = (ROOT / "architecture" / "retrieval.md").read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """`text` collapsed to one line, with each line's leading comment marker removed.

    Both prose surfaces wrap, so a pattern that has to span a line break cannot match the
    raw source: `config.yaml`'s "age 44.90 / #     days" collapses to "age 44.90 # days" if
    the `#` is left standing, and the reading then reads as absent — a red that is really a
    wrapper bug. Only a line's LEADING marker goes, so a mid-line "#111" survives.
    """
    return re.sub(
        r"\s+", " ",
        "\n".join(ln.lstrip().lstrip("#").lstrip() for ln in text.splitlines()))


def _seeding_block() -> str:
    start = CONFIG.index("# Query → entity seeding beyond the lexical extractor")
    end = CONFIG.index("retrieval:\n  entity_seeding:")
    return CONFIG[start:end]


def test_config_no_longer_calls_the_semantic_half_inert_and_states_the_disposition():
    """Clause 3. The comment is the interface a future operator reads before deciding.

    It said, verbatim, "The semantic half is inert until the entity-vector index
    exists: scripts/memory/build_entity_vectors.py writes it to …". The index has
    existed since 2026-09-26, so on the day #3907 was measured the block read as
    though nothing was being served while every `vault_recall` was.
    """
    b = _seeding_block()
    assert "inert until the entity-vector index exists" not in b, b
    assert "NOT INERT" in b, "the correction has to be as plain as the claim it replaces"
    for claim in ("HAND-BUILT", "no autonomy", "MAX_INDEX_AGE_DAYS",
                  "build_entity_vectors.py --status", "build.json", "ABSENT"):
        assert claim in b, f"the disposition must say {claim!r}:\n{b}"
    # The exit codes are in the comment because a job gate reads codes, not prose.
    assert "1 STALE" in b and "2 no artifact" in b, b
    assert "enabled: true" in CONFIG[CONFIG.index("entity_seeding:"):], (
        "the leg stays on: retiring it is the OTHER disposition, and this round's "
        "clause is that the state be truthful, not that it be off")


# ── #4021: the arm, and what the prose owes it ──────────────────────────────

#: The job #4021 armed to run the builder, and the only id either prose surface may name
#: as its runner. `~/obsidian/autonomy/111-entity-vectors-rebuild.md` is the task file;
#: `architecture/autonomy-jobs.md` §Bound entropy is where the fleet registers it.
RUNNER_ID = "#111"

#: The day the arm landed. A reading quoted inside the arming change cannot predate it, so
#: the staleness rule below is a predicate on the date rather than a compare against one
#: literal instant — review attempt 1 of this round caught the weaker form, in which a
#: figure carried from #3907 but stamped with any other minute would have passed. #3907's
#: own reading is 2026-11-07T22:19Z, two days before this.
ARMED_ON = dt.date(2026, 11, 9)

#: The sentences that describe an unarmed rebuild. Each was TRUE until #4021 armed #111;
#: each is FALSE now, so a surface that still carries one is describing a state this tree
#: does not have — the same defect class #3907 filed, read from the other side.
UNARMED_CLAIMS: tuple[str, ...] = ("not armed", "never been scheduled", "never armed")

#: A dated reading, as both surfaces must write one: an ISO instant to the minute. The
#: verbs are part of the pattern on purpose — #3907's carried line ("A dated reading,
#: 2026-11-07T22:19Z:") is a quoted historical figure, and a check that matched it would
#: punish the file for keeping the provenance the item asked it to keep.
_DATED_READING = re.compile(r"Read (?:at|again at) (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z)")


def _reading_date(text: str) -> dt.date:
    """The date of `text`'s dated reading, failing loudly if it has none."""
    m = _DATED_READING.search(_flat(text))
    assert m, (
        "the surface must quote a dated reading (`Read at <ISO8601>Z`) so a later reader "
        f"can tell a measurement from a promise:\n{text}")
    return dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%MZ").date()


def _assert_narration_is_armed(text: str, *, where: str) -> None:
    """Every predicate clause 5 puts on an arming narrative, in one callable place.

    A function and not assertions inline in the node because the node runs it a second and
    third time on SEEDED copies. This file's own idiom for a check that must prove itself —
    `test_the_own_series_ruling_lands_in_the_definition_and_not_in_the_tier_table` seeds a
    tier movement and asserts the seeded text still differs — applied to a deny-list, where
    "the assertion is in the file" says nothing on its own: a deny-list that never fires is
    indistinguishable from a doc that never regressed. Each seeded copy therefore names the
    predicate it expects to be tripped, so a control that tripped a DIFFERENT one is a red
    rather than a pass.

    Raises `AssertionError` naming the first predicate that failed.
    """
    flat = _flat(text)
    low = flat.lower()
    for claim in UNARMED_CLAIMS:
        if claim in low:
            raise AssertionError(
                f"{where} still claims {claim!r}, which the arm made false")
    if RUNNER_ID not in text:
        raise AssertionError(
            f"{where} names a cadence without naming {RUNNER_ID}: \"there is a weekly job\" "
            "is a promise, an id is something a reader can go and read")
    if "autonomy job" not in low:
        raise AssertionError(f"{where} must say what kind of thing runs the builder")
    if "weekly" not in low:
        raise AssertionError(f"{where} must state the cadence the job runs on")
    if "timeout_seconds: 7200" not in flat:
        raise AssertionError(
            f"{where} must state `timeout_seconds: 7200` — the 2 h envelope around a "
            "~63 min CPU run — or the next editor inherits a cadence with no idea of its "
            "cost")
    if _reading_date(flat) < ARMED_ON:
        raise AssertionError(
            f"{where}'s quoted reading predates the {ARMED_ON} arm, so the figure is "
            "carried rather than re-run — a carried number is a claim nobody re-ran")


def test_the_armed_runner_is_the_one_the_prose_names_and_no_doc_still_says_unarmed():
    """#4021 clause 5, repo half: the arming reaches the prose and the falsified claims die.

    #3907 wrote the node this replaces as "the doc must state that the job is NOT armed",
    because in that state it was the truth and a reader needed it said. Arming #111 turns
    that assertion into a pin on a dead state: it went red on the first honest revision,
    which is the work this item is, not a blocker — the previous round's gate `tests` rung
    reported exactly that one failure as new. What the node was reaching for is kept, and is
    now checkable positively instead of by the absence of a runner:

      * a passage that names a cadence names the JOB ID, and says it is an autonomy job;
      * that id has a functions-table row in `architecture/autonomy-jobs.md` in the same
        tree, so retrieval.md and the fleet doc cannot drift apart — the pairing
        `validate_tasks.py`'s `DOC_MEMBERSHIP` warning fires on when they do;
      * neither surface still says the rebuild has no runner. `git grep -n 'NOT armed'
        -- config.yaml architecture/` is the item's own acceptance grep; this is the form
        of it that runs in the gate;
      * each surface quotes a re-run reading stamped on or after the arm day. Never the
        figure itself: a node asserting today's 44.90 days would go red on the first real
        #111 run, which is the entire point of the job.

    `config.yaml` is checked inside the seeding block rather than whole-file: :2615 says
    "UPDATE is never armed" about conflict resolution, which is true and none of this
    item's business. The item's grep is over the file; this is the same deny-list scoped to
    the surface it was about.

    The vault half of the clause — that `#111` resolves to a real, armed task file — is not
    this node's, and this file cannot run it under the gate: `~/obsidian` is not a tree a
    round controls, which is what `pytest.ini`'s `live_vault` marker exists to record. It is
    pinned by `test_the_task_file_the_prose_names_declares_the_envelope_the_prose_quotes`,
    which that marker deselects in the gate and which must be run by hand.
    """
    _assert_narration_is_armed(_seeding_block(), where="config.yaml's seeding block")

    low = RETRIEVAL_DOC.lower()
    i = low.index("build_entity_vectors.py")
    passage = RETRIEVAL_DOC[max(0, i - 1200): i + 1600]
    assert "build_entity_vectors.py" in passage
    _assert_narration_is_armed(passage, where="architecture/retrieval.md's seeding passage")

    jobs = (ROOT / "architecture" / "autonomy-jobs.md").read_text(encoding="utf-8")
    assert re.search(rf"^\| {re.escape(RUNNER_ID)} \|", jobs, re.M), (
        f"{RUNNER_ID} is named as the runner by the prose but has no functions-table row in "
        "architecture/autonomy-jobs.md — the doc-to-doc drift DOC_MEMBERSHIP warns on")

    # The four predicates above have to be shown to bite. Each copy seeds ONE regression.
    controls = {
        "the sentence the arm deleted, seeded back": (
            "#4021 armed autonomy job",
            "the job that would run it is NOT armed, and it has never been scheduled to",
            "not armed"),
        "the runner id moved to a job that does not exist": (
            "armed autonomy job **#111**", "armed autonomy job **#999**", "without naming"),
        "the envelope dropped from the narrative": (
            "`timeout_seconds: 7200`", "`timeout_seconds` unstated", "timeout_seconds"),
        "the reading carried back to #3907's instant": (
            "2026-11-09T21:59Z", "2026-11-07T22:19Z", "predates"),
    }
    for label, (old, new, expect) in controls.items():
        seeded = RETRIEVAL_DOC.replace(old, new, 1)
        assert seeded != RETRIEVAL_DOC, (
            f"the control {label!r} did not take, so it proved nothing")
        with pytest.raises(AssertionError) as excinfo:
            _assert_narration_is_armed(seeded, where="architecture/retrieval.md's seeding passage")
        assert expect in str(excinfo.value).lower(), (
            f"the control {label!r} tripped {str(excinfo.value)!r} instead of the predicate "
            f"it was seeded to trip ({expect!r}), which means the predicate is not the one "
            "this node claims")


def test_the_rail_that_let_this_job_arm_refuses_the_same_commit_moving_a_schedule_key(
        tmp_path, monkeypatch):
    """#4021 clause 2, repo half: the one-line arm is the only shape that can land.

    Clause 2 is a claim about a commit in a tree the gate does not run against — the arm is
    vault sha `18da0197c`, one changed line, `status: draft` -> `status: up_next` — and no
    repo node can re-run a vault `git log` under the gate's `-m "not live_vault"`. What a
    repo node CAN pin is the reason that one-line shape is a necessity and not a habit:
    `app.harness.policy.first_arm_allowed`, which every file lane asks (`agent_mcp/vault.py`
    for `automod_vault_land`, `agent_mcp/builtin_fs.py`, `scripts/automod/vault_round.py`),
    refuses a first arm whose dispatch-affecting move list is anything other than exactly
    `["status"]`. The difference between an arm that lands and an arm that is refused is
    precisely the key set, so this pair IS the clause: this job's create-then-arm maps pass
    the rail, and every variant that co-moves a field in `SCHEDULE_STATE_FIELDS` with
    `status` is refused as a cadence move rather than split into the half the rail would
    allow. The last block states where that guarantee stops, because clause 2 names two
    keys (`timeout_seconds`, `agent_id`) the rail does not count at all.

    Fixture maps rather than the live vault file, for the reason the docstring of
    `first_arm_allowed` gives for reading both sides' bytes: the rule is about the SHAPE a
    write has, and a shape is testable without the vault. `LLOYD_DATA` points at an empty
    tmp dir because the rail's last condition is "no `<runs-root>/<id>/` yet": the real
    root has no `111/` today, but it will have one the first Wednesday the fleet dispatches
    #111, and a node that read the real root would go red on the job succeeding.
    """
    monkeypatch.setenv("LLOYD_DATA", str(tmp_path))

    # The create commit's dispatch keys, verbatim from clause 1: weekly, 2 h, this skill,
    # this agent. The arm moves `status` and none of the rest.
    created = {"id": 111, "skill_name": "entity-vectors-rebuild", "frequency": "weekly",
               "timeout_seconds": 7200, "agent_id": "worker", "max_retries": 3,
               "status": "draft"}
    armed = {**created, "status": "up_next"}
    task = "autonomy/111-entity-vectors-rebuild.md"

    assert POL.first_arm_allowed(task, created, armed) is True, (
        "the shape this item's arm commit used is the shape the rail exists to allow; if "
        "this is False the arm is not policy-compliant and clause 2 cannot be met")

    # Every field the rail counts. `SCHEDULE_STATE_FIELDS` is the whole of it, so the loop
    # reads the set rather than a hand-typed sample: a field added there later is refused
    # here on the day it is added, which is the day the guarantee is worth something.
    guarded = sorted(POL.SCHEDULE_STATE_FIELDS - {"status"})
    assert guarded, "the rail guards no field beside status, so nothing below can prove"
    for field in guarded:
        co_moved = {**armed, field: "changed-for-the-test"}
        assert POL.first_arm_allowed(task, created, co_moved) is False, (
            f"an arm that also moves {field} must be refused as a cadence move, not split "
            "into the half the rail would allow — that refusal is what forces clause 2's "
            "one-line diff")

    # And the honest boundary of that guarantee. Two of the keys clause 2 names as things
    # the arm must not co-move — `timeout_seconds` and `agent_id` — are NOT in the rail's
    # field set, so for THOSE keys the one-line arm is a discipline the create-first shape
    # imposes (an envelope written in the create cannot be re-written by an arm that wants
    # to stay a first arm), not something `first_arm_allowed` refuses. Stated as an
    # assertion because it is the fact a future reader needs to grade the clause by: the
    # rail is why `frequency` and `skill_name` could not have co-moved, and nothing else.
    assert not {"timeout_seconds", "agent_id"} & set(POL.SCHEDULE_STATE_FIELDS), (
        "the rail grew a guard the clause did not expect; re-read what clause 2's "
        "one-line-diff check is actually enforcing before this red is made to pass")

    # First arm means first. Once a run record exists the same two maps are refused, which
    # is the rail saying "this is a restart, and restarts are not this lane's to grant".
    (tmp_path / "autonomy-runs" / "111").mkdir(parents=True)
    assert POL.first_arm_allowed(task, created, armed) is False, (
        "the run-record condition did not fire, so the rail would re-arm a job that is "
        "already running weekly")


@live_vault
def test_the_task_file_the_prose_names_declares_the_envelope_the_prose_quotes():
    """#4021 clauses 1 and 2, vault half: the file the prose points at says what it runs.

    `@live_vault` is not a dodge, it is the harness's own rule (`pytest.ini:14-15`): the
    gate runs `-m "not live_vault"` because no round under test controls `~/obsidian`, so
    this node does NOT execute in the gate. Run it by hand from a worktree:

        ~/lloyd/.venvs/lloyd/bin/python -m pytest \\
            tests/test_entity_vectors_status.py::test_the_task_file_the_prose_names_declares_the_envelope_the_prose_quotes -q

    It is the only node anywhere that pins the six front-matter keys clause 1 asks for,
    which is worth stating plainly rather than leaving to inference: clause 1 is a claim
    about a vault file, and a repo test that pretended to grade it would be reading a tree
    the round does not own. What the repo side pins instead is that its own narrative and
    the fleet doc carry the same id, cadence and envelope
    (`test_the_armed_runner_is_the_one_the_prose_names_and_no_doc_still_says_unarmed`), so a
    task file whose numbers disagree with the prose the arm wrote has one of the two red.

    The clause-2 half is the pair of commits, read as data rather than as a diff: a create
    whose dispatch keys are all present at `status: draft`, and an armed state that differs
    from it in `status` alone. `first_arm_allowed` is what refused any other shape at the
    time, and the node above proves that rail bites; this proves the thing that landed
    through it is the file the prose describes.
    """
    task = Path.home() / "obsidian" / "autonomy" / "111-entity-vectors-rebuild.md"
    if not task.is_file():
        pytest.skip(f"{task} is not on this box, so clause 1 has nothing to grade")
    fm = _front_matter(task.read_text(encoding="utf-8"))
    for key, want in (("id", 111),
                      ("skill_name", "entity-vectors-rebuild"),
                      ("frequency", "weekly"),
                      ("timeout_seconds", 7200),
                      ("agent_id", "worker"),
                      ("status", "up_next")):
        assert fm.get(key) == want, (
            f"clause 1 puts {key}: {want!r} in the task file; it reads {fm.get(key)!r}")

    desc = re.sub(r"\s+", " ", str(fm.get("description", "")))
    assert "build_entity_vectors.py --status" in desc, (
        "the description must carry the before/after command, or a dispatched run has "
        f"nothing to report from: {desc!r}")
    assert "build_entity_vectors.py" in desc and "--status" in desc
    assert "~/lloyd" in desc and "ABSENT" in desc, (
        f"Rule 1 is the clause the fleet most needs stated — `app.paths` anchors "
        f"VAULT_DERIVED_ROOT to the tree it is imported from, so `--status` run from an "
        f"automod worktree prints ABSENT and a run built there measures nothing: {desc!r}")

    # Clause 2 as data: the arm moved the status key and nothing dispatch-affecting.
    assert fm.get("parked") in (None, False), (
        f"a parked declaration and an armed status contradict each other: {fm.get('parked')!r}")


@live_vault
def test_the_skill_names_the_job_that_runs_it_and_no_other_job_at_all():
    """#4021 clause 5, vault half: `entity-vectors-rebuild` routes a reader to #111 only.

    Also `@live_vault`, so the gate never runs it and it must be run by hand from a
    worktree; the reason it is a separate node from the one above is that it grades a
    different file with a different failure mode — a mis-numbered routing, not a missing
    key. The mis-numbering this pins is the kind a `#`-prefixed grep misses: the skill
    claimed #110 in its `description`, its heading and its §What-closes-this, AND carried
    `autonomy-110` as a front-matter TAG with no `#` in front of it, so `grep -rn '#110'`
    went to zero while the front matter still routed a reader to
    `110-guardian-note-sweep-daily.md` — a job whose whole body is guardian `dated:`
    retractions and which shares nothing with this one. Hence the tag form as well as the
    hash form, and the tag read back against the file it resolves to.
    """
    skill = Path.home() / "obsidian" / "skills" / "entity-vectors-rebuild" / "SKILL.md"
    if not skill.is_file():
        pytest.skip(f"{skill} is not on this box, so the skill cannot be read")
    text = skill.read_text(encoding="utf-8")
    stray = sorted({m.group(0) for m in re.finditer(r"#1\d\d\b|autonomy-1\d\d\b", text)}
                   - {RUNNER_ID, "autonomy-111"})
    assert not stray, (
        f"the skill still names {stray} as its job; only {RUNNER_ID} runs this builder")
    front = _front_matter(text)
    assert "autonomy-111" in (front.get("tags") or []), (
        f"the front-matter tag must resolve to 111, not {front.get('tags')!r}")
    assert RUNNER_ID in re.sub(r"\s+", " ", str(front.get("description", ""))), (
        "the description must say which job runs this")
    heading = [ln for ln in text.splitlines() if ln.startswith("# ")]
    assert heading and RUNNER_ID in heading[0], f"the H1 must name the job: {heading}"
    assert (Path.home() / "obsidian" / "autonomy" / "111-entity-vectors-rebuild.md").is_file(), (
        "the tag `autonomy-111` resolves to no task file, which is the orphan this round "
        "was filed to close")
