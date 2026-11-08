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
actually taken, and no architecture doc claims a rebuild schedule this checkout does
not contain.

Nothing here touches the live artifact or the live store: every case builds its own
`names.json` + `vectors.npy` under `tmp_path` and points `app.kg_store.store` at a
fixture store. The real 2026-11-07 reading — 11,990 rows, 42.9 days, 71.2%, exit 1 —
is quoted in `config.yaml` and in the item, not asserted against the disk here,
because a test that reads the production artifact goes red or green on a rebuild,
which is exactly the thing that has no schedule.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import entity_linker as EL                        # noqa: E402
from app import kg_store as ks                             # noqa: E402
from scripts.memory import build_entity_vectors as BUILD   # noqa: E402


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


def test_no_architecture_doc_names_a_rebuild_schedule_this_tree_does_not_have():
    """The second half of clause 3: a doc may not promise a job that isn't there.

    `architecture/retrieval.md` names the BUILDER, which is honest — the file exists
    and is the thing that wrote the artifact. What it may not do, and does not, is
    name a queue kind, an autonomy task or a unit that would run it on a schedule,
    because `git grep` over the tree finds none and a reader who trusted the doc would
    wait for a rebuild that never comes. This is the check that keeps it that way: the
    scheduler-shaped words near the entity-vector passage must be accompanied by the
    fact that the job is not armed.
    """
    low = RETRIEVAL_DOC.lower()
    i = low.index("build_entity_vectors.py")
    passage = RETRIEVAL_DOC[max(0, i - 1200): i + 1600]
    assert "build_entity_vectors.py" in passage
    scheduled = [w for w in ("autonomy task", "queue job", "systemd", "weekly job",
                             "rebuild job", "nightly") if w in passage.lower()]
    if scheduled:
        assert "not armed" in passage.lower() or "hand-built" in passage.lower(), (
            f"the doc says {scheduled} but does not state that the job is unarmed, "
            f"which is the promise #3907 exists to stop:\n{passage}")
    assert "never been scheduled" in RETRIEVAL_DOC or "never armed" in RETRIEVAL_DOC \
        or "not armed" in RETRIEVAL_DOC, (
        "architecture/retrieval.md must state the actual state of the schedule")
