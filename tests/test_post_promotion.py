"""Backlog #429 — the post-promotion comparison that used to not exist.

Nothing ever looked at a promotion after it landed: the live ledger's 30,953
rows contain 0 mentions of rollback/revert against 65 `"promoted": true` decision
rows, and `run_round.py` never read a prior round's score. These tests pin the
two halves of the fix — a machine-readable `round_summary` row per round, and a
round report that names a baseline decline past the noise floor and attributes it
to the promotion that caused it.

The producer is switched off (`workers.sources.autoresearch` is `enabled: false`,
tracked as #682), so **no test here runs a round against a model**. The replay
test instead drops a real round's own report onto disk and reads the comparison
out of it; the wiring test drives `run_round.run()` over stubbed collaborators
only far enough to reach the record-and-surface step.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import json
from pathlib import Path
from types import CodeType

import pytest

from scripts.autoresearch import post_promotion, run_round
from scripts.autoresearch.common import DEFAULT_NOISE_FLOOR, AutoresearchConfig, AutoresearchPaths

#: Round R_20260908_165252 is the last promotion in the live ledger and the one
#: the item names as the replay subject. Copied verbatim from
#: `_pipeline/research/rounds/R_20260908_165252.md`; the numbers below (baseline
#: 0.4364, winner `V_20260908_165359_f45720` at 0.6145, snapshot 20260908_165708)
#: are that round's, not invented. That it is real output rather than a guess is
#: pinned two ways, both of which run on every machine: it parses to exactly the
#: record `test_a_real_round_report_reads_back_as_a_promotion_record` asserts, and
#: `test_the_fixture_and_the_writer_agree_about_one_round` requires it to agree
#: field-for-field with a report rendered from `run_round`'s own report templates.
R_20260908_165252 = """# Autoresearch round R_20260908_165252
- started_at: 2026-09-08T16:57:08Z
- model: primary
- harness: direct
- tasks: 11
- tasks on the harness runner: 0
- variants proposed: 7
- baseline mean composite: 0.4364

## Variant summaries
- `BASELINE_1788882602` (baseline): mean=0.4364, safety=pass, tasks=11
- `V_20260908_165312_19878f`: mean=0.5795, safety=pass, tasks=11
- `V_20260908_165328_9fe455`: mean=0.5259, safety=pass, tasks=11
- `V_20260908_165352_178542`: mean=0.6077, safety=pass, tasks=11
- `V_20260908_165359_f45720`: mean=0.6145, safety=pass, tasks=11
- `V_20260908_165409_451da2`: mean=0.4986, safety=pass, tasks=11
- `V_20260908_165423_dcea63`: mean=0.6395, safety=pass, tasks=11
- `V_20260908_165445_bde451`: mean=0.7068, safety=pass, tasks=11

## Promotion decisions
- `V_20260908_165312_19878f`: HOLD — insufficient_win_fraction (0.27 < 0.5)
- `V_20260908_165328_9fe455`: HOLD — insufficient_win_fraction (0.36 < 0.5)
- `V_20260908_165352_178542`: HOLD — insufficient_win_fraction (0.36 < 0.5)
- `V_20260908_165359_f45720`: PROMOTE — promote (delta=+0.1781, win_frac=0.64)
- `V_20260908_165409_451da2`: HOLD — insufficient_win_fraction (0.27 < 0.5)
- `V_20260908_165423_dcea63`: HOLD — insufficient_win_fraction (0.45 < 0.5)
- `V_20260908_165445_bde451`: HOLD — insufficient_win_fraction (0.45 < 0.5)

## Promoted
- variant: `V_20260908_165359_f45720`
- snapshot_dir: `/home/alansrobotlab/lloyd/_pipeline/research/snapshots/20260908_165708`
- applied_files: ['SOUL.md']
- experiment_fact: `/home/alansrobotlab/lloyd/_pipeline/vault-derived/facts/Experiments/V_20260908_165359_f45720/V_20260908_165359_f45720-experiment.md`
"""

PROMOTED = "V_20260908_165359_f45720"
SNAPSHOT_DIR = "/home/alansrobotlab/lloyd/_pipeline/research/snapshots/20260908_165708"


def make_cfg(tmp_path: Path, **over) -> AutoresearchConfig:
    paths = AutoresearchPaths(
        bench_dir=tmp_path / "bench",
        research_root=tmp_path / "research",
        rounds_dir=tmp_path / "rounds",
        ledger_path=tmp_path / "ledger.jsonl",
        variants_dir=tmp_path / "variants",
        snapshots_dir=tmp_path / "snapshots",
        facts_experiments_dir=tmp_path / "facts-experiments",
    )
    kw = dict(
        paths=paths,
        default_model="primary",
        default_budget_minutes=60,
        max_variants_per_round=7,
        promotion_min_win_fraction=0.5,
        promotion_min_composite_delta=0.05,
        promotion_require_safety_pass=True,
        tool_allowlist_consecutive_wins=2,
        targets=["prompts"],
    )
    kw.update(over)
    return AutoresearchConfig(**kw)


@pytest.fixture
def world(tmp_path):
    """One rounds dir carrying the real R_20260908_165252 report, plus a ledger."""
    cfg = make_cfg(tmp_path)
    cfg.paths.rounds_dir.mkdir(parents=True, exist_ok=True)
    (cfg.paths.rounds_dir / "R_20260908_165252.md").write_text(R_20260908_165252, encoding="utf-8")
    cfg.paths.ledger_path.touch()
    return cfg


def landed(vid: str = PROMOTED, snapshot: str = SNAPSHOT_DIR) -> dict:
    return {"variant_id": vid, "snapshot_dir": snapshot, "applied_files": ["SOUL.md"]}


def rows_of(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def acceptance_check(path: Path) -> list[dict]:
    """The item's own check, verbatim in spirit: rows carrying baseline_mean, a
    promoted variant id, and that variant's recorded mean. Triage measured 0."""
    return [
        r for r in rows_of(path)
        if r.get("baseline_mean") is not None
        and r.get("promoted_variant_id")
        and r.get("promoted_variant_mean") is not None
    ]


# ── clause 1: the machine-readable row ───────────────────────────────────────

def test_the_item_s_acceptance_check_now_finds_a_row_with_all_three_numbers(world):
    cfg = world
    row = post_promotion.record_round_summary(
        cfg, "R_20260909_060000", 0.4364, landed(), {"mean_composite": 0.6145}
    )
    assert row["event"] == post_promotion.ROUND_SUMMARY_EVENT
    assert (row["baseline_mean"], row["promoted_variant_id"], row["promoted_variant_mean"]) == (
        0.4364, PROMOTED, 0.6145,
    )
    # The check that had to stop reproducing:
    assert len(acceptance_check(cfg.paths.ledger_path)) == 1
    found = acceptance_check(cfg.paths.ledger_path)[0]
    assert found["round_id"] == "R_20260909_060000"
    assert found["snapshot_dir"] == SNAPSHOT_DIR


def test_a_round_that_promoted_nothing_still_records_its_baseline(world):
    """Per-round, not per-promotion: a null row is the evidence the check ran."""
    cfg = world
    post_promotion.record_round_summary(cfg, "R_20260909_060000", 0.5100, None, None)
    post_promotion.record_round_summary(
        cfg, "R_20260909_070000", 0.5000,
        {"variant_id": PROMOTED, "snapshot_dir": None, "refused": ["contract"]},
        {"mean_composite": 0.6145},
    )
    rows = post_promotion.round_summary_rows(cfg.paths.ledger_path)
    assert [r["round_id"] for r in rows] == ["R_20260909_060000", "R_20260909_070000"]
    assert all(r["baseline_mean"] is not None for r in rows)
    # A refused promotion landed nothing, so it must not be recorded as one.
    assert all(r["promoted_variant_id"] is None for r in rows)
    assert acceptance_check(cfg.paths.ledger_path) == []


def test_the_noise_floor_is_the_measured_one_and_config_can_override_it(tmp_path, monkeypatch):
    from scripts.autoresearch import common

    assert DEFAULT_NOISE_FLOOR == 0.1389  # backlog #324, 84-round cross-round std
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "autoresearch:\n"
        "  bench_dir: b\n  research_root: r\n  rounds_dir: r\n  ledger_path: r/l\n"
        "  variants_dir: r/v\n  snapshots_dir: r/s\n  facts_experiments_dir: r/f\n"
        "  promotion:\n    noise_floor: 0.20\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(common, "CONFIG_PATH", cfg_file)
    assert common.load_config().promotion_noise_floor == 0.20
    cfg_file.write_text(
        "autoresearch:\n"
        "  bench_dir: b\n  research_root: r\n  rounds_dir: r\n  ledger_path: r/l\n"
        "  variants_dir: r/v\n  snapshots_dir: r/s\n  facts_experiments_dir: r/f\n",
        encoding="utf-8",
    )
    assert common.load_config().promotion_noise_floor == DEFAULT_NOISE_FLOOR


# ── clause 5: replay an existing round, never a live round ───────────────────

def test_a_real_round_report_reads_back_as_a_promotion_record(world):
    record = post_promotion.promotion_record_from_report(
        world.paths.rounds_dir / "R_20260908_165252.md"
    )
    assert record["promoted_variant_id"] == PROMOTED
    assert record["promoted_variant_mean"] == 0.6145
    assert record["baseline_mean"] == 0.4364
    assert record["snapshot_dir"] == SNAPSHOT_DIR


def test_a_round_with_no_promoted_block_is_not_a_promotion(world):
    path = world.paths.rounds_dir / "R_20260909_050000.md"
    path.write_text(
        "# Autoresearch round R_20260909_050000\n- baseline mean composite: 0.5000\n\n"
        "## Variant summaries\n- `V_x`: mean=0.4000, safety=pass, tasks=11\n\n"
        "## Promotion decisions\n- `V_x`: HOLD — insufficient_delta\n",
        encoding="utf-8",
    )
    assert post_promotion.promotion_record_from_report(path) is None
    assert post_promotion.last_promotion(
        world.paths.ledger_path, world.paths.rounds_dir
    )["round_id"] == "R_20260908_165252"


def test_replaying_the_existing_round_produces_a_beyond_noise_decline(world):
    """R_20260908_165252 promoted on a 0.6145 mean; a later round whose baseline
    comes in at 0.4364 has fallen 0.1781 — past the 0.1389 floor. Produced from
    the report on disk, with no round run."""
    prior = post_promotion.last_promotion(world.paths.ledger_path, world.paths.rounds_dir)
    comparison = post_promotion.compare(0.4364, prior, DEFAULT_NOISE_FLOOR)
    assert comparison["decline"] == pytest.approx(0.1781, abs=1e-4)
    assert comparison["regression"] is True

    text = "\n".join(post_promotion.report_section(comparison))
    assert "BASELINE DECLINE PAST NOISE FLOOR" in text
    assert PROMOTED in text
    assert "R_20260908_165252" in text


def test_the_replay_survives_the_report_being_the_only_store(world):
    """The 65 promotions that predate this module have no ledger row, so the
    report is their only record; the check must still see them."""
    assert post_promotion.round_summary_rows(world.paths.ledger_path) == []
    assert post_promotion.last_promotion(
        world.paths.ledger_path, world.paths.rounds_dir
    )["promoted_variant_id"] == PROMOTED


# ── the format contract between the writer and the parser ────────────────────
#
# `post_promotion` parses prose that `run_round` writes. Nothing pinned that
# coupling: the previous guard here compared the fixture above against the live
# `_pipeline/research/rounds/R_20260908_165252.md`, `_pipeline` is gitignored, and
# so it skipped on every machine where the suite actually runs. These checks
# enforce the same thing with no gitignored artifact — they compile the
# f-strings out of `run_round`'s own syntax tree, render them, and read the
# result back through the parsers. Rename a field on either side and they fail.

#: The report lines `post_promotion` and `promotion_fp_rate` parse, keyed by what
#: they carry, each identified by the literal text that opens it inside
#: `run_round.run()`'s f-strings.
_REPORT_TEMPLATES = {
    # (prefix of the f-string's literal text, a token that may sit inside its
    # {…} interpolations instead) — `lit` is the literal chunks only, `src` the
    # whole f-string reparsed, so a word written inside a conditional still counts.
    "baseline_mean": lambda lit, src: lit.startswith("- baseline mean composite: "),
    # `- `V_x` (baseline): mean=…` — the per-variant mean line, baseline and candidate alike.
    "variant_mean": lambda lit, src: lit.startswith("- `") and "mean=" in src,
    "promote_decision": lambda lit, src: lit.startswith("- `") and "PROMOTE" in src,
    "promoted_variant": lambda lit, src: lit.startswith("- variant: "),
    "snapshot_dir": lambda lit, src: lit.startswith("- snapshot_dir: "),
}

#: The names those f-strings interpolate, at the values R_20260908_165252 had.
_TEMPLATE_NS = {
    "baseline_summary": {"mean_composite": 0.4364},
    "summ": {"mean_composite": 0.6145, "safety_passed": True, "task_count": 11},
    "vid": PROMOTED,
    "marker": "",
    "d": {"variant_id": PROMOTED, "should_promote": True,
          "reason": "promote (delta=+0.1781, win_frac=0.64)"},
    "promotion_result": {
        "variant_id": PROMOTED, "snapshot_dir": SNAPSHOT_DIR,
        "applied_files": ["SOUL.md"], "experiment_fact": None,
    },
}


def _report_template(key: str) -> CodeType:
    """Compile the one f-string in `run_round` that writes report line `key`."""
    hits = []
    for node in ast.walk(ast.parse(inspect.getsource(run_round))):
        if not isinstance(node, ast.JoinedStr):
            continue
        literal = "".join(p.value for p in node.values if isinstance(p, ast.Constant))
        if _REPORT_TEMPLATES[key](literal, ast.unparse(node)):
            hits.append(node)
    assert len(hits) == 1, (
        f"expected exactly one run_round report line matching {key!r}, found {len(hits)}: "
        f"{[ast.unparse(h) for h in hits]!r} — the report format moved and "
        "post_promotion's parser was not told about it"
    )
    expr = ast.Expression(body=hits[0])
    ast.fix_missing_locations(expr)
    return compile(expr, f"<run_round report line: {key}>", "eval")


def _render(key: str, **over) -> str:
    """Evaluate `run_round`'s own report template — the writer's bytes, not a copy."""
    ns = dict(_TEMPLATE_NS)
    ns.update(over)
    return eval(_report_template(key), {"__builtins__": {}}, ns)  # noqa: S307


def _rendered_round_report(rid: str = "R_20260909_060000") -> str:
    """A whole round report, assembled from the lines `run_round` really writes."""
    return "\n".join([
        f"# Autoresearch round {rid}",
        "- started_at: 2026-09-09T06:00:00Z",
        _render("baseline_mean"),
        "",
        "## Variant summaries",
        _render("variant_mean", vid="BASELINE_fixture", marker=" (baseline)",
                summ={"mean_composite": 0.4364, "safety_passed": True, "task_count": 11}),
        _render("variant_mean"),
        "",
        "## Promotion decisions",
        _render("promote_decision"),
        "",
        "## Promoted",
        _render("promoted_variant"),
        _render("snapshot_dir"),
        "- applied_files: ['SOUL.md']",
        "",
    ]) + "\n"


def test_the_report_the_writer_writes_reads_back_as_a_promotion(tmp_path):
    """Round-trip: `run_round`'s templates in, `post_promotion`'s parser out.

    This is the check that replaces the byte comparison against a gitignored
    artifact: it needs no round, no model, and no `_pipeline`.
    """
    rounds = tmp_path / "rounds"
    rounds.mkdir()
    path = rounds / "R_20260909_060000.md"
    path.write_text(_rendered_round_report(), encoding="utf-8")

    record = post_promotion.promotion_record_from_report(path)
    assert record == {
        "source": "report",
        "round_id": "R_20260909_060000",
        "baseline_mean": 0.4364,
        "promoted_variant_id": PROMOTED,
        "promoted_variant_mean": 0.6145,
        "snapshot_dir": SNAPSHOT_DIR,
    }


def test_the_fixture_and_the_writer_agree_about_one_round(tmp_path):
    """The vendored R_20260908_165252 and a freshly rendered report must not drift.

    One says what a promotion looked like, the other what one looks like now; if
    they parse to different records, the format has moved under the replay test.
    """
    rounds = tmp_path / "rounds"
    rounds.mkdir()
    fixture = rounds / "R_20260908_165252.md"
    fixture.write_text(R_20260908_165252, encoding="utf-8")
    rendered = rounds / "R_20260909_060000.md"
    rendered.write_text(_rendered_round_report(), encoding="utf-8")

    a = post_promotion.promotion_record_from_report(fixture)
    b = post_promotion.promotion_record_from_report(rendered)
    assert {k: v for k, v in a.items() if k not in {"round_id", "source"}} == \
           {k: v for k, v in b.items() if k not in {"round_id", "source"}}


def test_a_renamed_report_field_breaks_the_parser_loudly(tmp_path):
    """Proves the round-trip above can fail: a writer that renames the landing
    record's field is not read as a promotion, and this test sees that."""
    rounds = tmp_path / "rounds"
    rounds.mkdir()
    drifted = _rendered_round_report().replace("- variant: `", "- promoted: `")
    assert drifted != _rendered_round_report()  # the template really did say `variant:`
    path = rounds / "R_20260909_070000.md"
    path.write_text(drifted, encoding="utf-8")
    assert post_promotion.promotion_record_from_report(path) is None


# ── the seam into #428's sweep: it re-reads the bytes this round appends ─────

def test_the_new_section_does_not_move_the_fp_sweep_s_readers(tmp_path):
    """`promotion_fp_rate` parses every `rounds/R_*.md`, and #429 started
    appending a section full of competing means to those same bytes.

    `baseline_means` takes the first `- baseline mean composite:` in the file and
    `recorded_deltas` takes every "- `V_x`: PROMOTE … delta=" line; the decline
    line carries a baseline mean, a promoted mean, a delta and a noise floor, all
    as prose. Crossed here on real bytes, in both its shapes, so the sweep's
    denominator cannot silently move when a report gains this section.
    """
    from scripts.autoresearch import promotion_fp_rate as fp

    rounds = tmp_path / "rounds"
    rounds.mkdir()
    path = rounds / "R_20260909_060000.md"
    path.write_text(_rendered_round_report(), encoding="utf-8")

    before_means = fp.baseline_means(rounds_dir=rounds)
    before_deltas = fp.recorded_deltas(rounds_dir=rounds)
    assert before_means == {"R_20260909_060000": 0.4364}
    assert before_deltas == {("R_20260909_060000", PROMOTED): 0.1781}

    record = post_promotion.last_promotion(tmp_path / "none.jsonl", rounds)
    for baseline in (0.3000, 0.5200):  # one decline past the floor, one inside it
        path.write_text(
            _rendered_round_report()
            + "\n".join(post_promotion.report_section(
                post_promotion.compare(baseline, record, DEFAULT_NOISE_FLOOR))) + "\n",
            encoding="utf-8",
        )
        assert fp.baseline_means(rounds_dir=rounds) == before_means
        assert fp.recorded_deltas(rounds_dir=rounds) == before_deltas
        # And the numbers the section introduces are not what the sweep reads:
        # 0.3000 is the only figure in the file a looser baseline regex could take.
        assert 0.3000 not in fp.baseline_means(rounds_dir=rounds).values()
        # The promotion is still the promotion, and still attributed.
        assert post_promotion.promotion_record_from_report(path)["promoted_variant_id"] == PROMOTED

    # A report whose section says there is nothing to check is equally inert.
    path.write_text(
        _rendered_round_report()
        + "\n".join(post_promotion.report_section(None)) + "\n", encoding="utf-8")
    assert fp.baseline_means(rounds_dir=rounds) == before_means
    assert fp.recorded_deltas(rounds_dir=rounds) == before_deltas


def test_a_decline_inside_the_floor_is_recorded_but_not_called_a_regression(world):
    """0.6145 → 0.5200 is 0.0945: fresh bench prompts drawn differently, not a
    regression the promotion caused. Naming it would be the false alarm the floor
    exists to prevent (#324: 0% cross-round baseline overlap)."""
    prior = post_promotion.last_promotion(world.paths.ledger_path, world.paths.rounds_dir)
    comparison = post_promotion.compare(0.5200, prior, DEFAULT_NOISE_FLOOR)
    assert comparison["regression"] is False
    text = "\n".join(post_promotion.report_section(comparison))
    assert "BASELINE DECLINE PAST NOISE FLOOR" not in text
    assert "within the noise floor" in text
    assert PROMOTED in text  # the comparison is still visible, not silently dropped


def test_the_decline_line_names_the_snapshot_it_could_be_restored_from(world):
    """Clause 3: attribution to one promotion, including the rollback point.
    `promote()` already resolved it into `snapshot_dir`; this is where a human
    reading the report gets it back."""
    prior = post_promotion.last_promotion(world.paths.ledger_path, world.paths.rounds_dir)
    comparison = post_promotion.compare(0.3000, prior, DEFAULT_NOISE_FLOOR)
    text = "\n".join(post_promotion.report_section(comparison))
    assert SNAPSHOT_DIR in text
    assert "20260908_165708" in text
    assert 'autoresearch_rollback(snapshot_ts="20260908_165708")' in text


def test_the_check_records_and_surfaces_and_never_restores(world, tmp_path, monkeypatch):
    """The item's out-of-scope clause, asserted as an absence rather than prose.

    Tripwires on `promote.rollback` would prove nothing here, because nothing on
    this path calls into `promote` — so instead: the module must not reach for the
    restore machinery at all, and the record-and-surface step must not touch a
    single file except the ledger.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(post_promotion))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert "promote" not in imported, "post_promotion must not import the promote module"
    assert "shutil" not in imported
    called = {
        n.func.attr for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    for forbidden in {"rollback", "apply_overlay", "snapshot_current_prompts",
                      "copy2", "copytree", "rmtree", "unlink", "write_text"}:
        assert forbidden not in called, f"the record-and-surface path must never call {forbidden}"

    cfg = world
    watched = cfg.paths.rounds_dir
    before = sorted((p.name, p.stat().st_mtime_ns) for p in watched.iterdir())
    prior = post_promotion.last_promotion(cfg.paths.ledger_path, cfg.paths.rounds_dir)
    comparison = post_promotion.compare(0.3000, prior, DEFAULT_NOISE_FLOOR)
    assert comparison["decline"] == pytest.approx(0.3145, abs=1e-4)
    assert comparison["regression"] is True
    text = "\n".join(post_promotion.report_section(comparison))
    assert sorted((p.name, p.stat().st_mtime_ns) for p in watched.iterdir()) == before, \
        "the check wrote to the rounds directory"
    assert "Nothing was restored" in text
    assert "promotion_fp_rate" in text  # names #428's sweep as the backstop
    # The manual path this check deliberately declines to use is still wired, and
    # still refuses rather than restores — asserted where the route actually lives
    # (`promote.rollback is not None` could never fail, so it proved nothing).
    from agent_mcp import autoresearch as mcp

    monkeypatch.setattr(mcp, "_load_cfg", lambda: cfg)
    answer = json.loads(mcp._handle_rollback({"snapshot_ts": "no-such-snapshot"}))
    assert "error" in answer and "snapshot" in answer["error"].lower()
    assert not any((cfg.paths.snapshots_dir).glob("*")), \
        "probing the manual route must not create a snapshot"


def test_a_promotion_with_no_snapshot_says_so_rather_than_inventing_one(world):
    """A decline on a promotion with no rollback point must say there is none.

    Not hypothetical in shape: the ledger's 65 `"promoted": true` rows carry no
    snapshot field whatsoever — the rollback point lives only as prose in the
    round report — so any promotion recovered from the *ledger* alone arrives
    here with an empty `snapshot_dir`.
    """
    prior = {
        "source": "ledger", "round_id": "R_20260908_165252", "baseline_mean": 0.4364,
        "promoted_variant_id": PROMOTED, "promoted_variant_mean": 0.6145, "snapshot_dir": "",
    }
    comparison = post_promotion.compare(0.3000, prior, DEFAULT_NOISE_FLOOR)
    text = "\n".join(post_promotion.report_section(comparison))
    assert "no snapshot directory on record" in text
    assert "autoresearch_rollback" not in text


def test_the_comparison_excludes_the_round_asking(world):
    """A round that promoted something must not be compared against itself —
    its own baseline against its own winner is the delta it already reported."""
    cfg = world
    post_promotion.record_round_summary(
        cfg, "R_20260909_060000", 0.4364, landed("V_self"), {"mean_composite": 0.9}
    )
    prior = post_promotion.last_promotion(
        cfg.paths.ledger_path, cfg.paths.rounds_dir, exclude_round="R_20260909_060000"
    )
    assert prior["round_id"] == "R_20260908_165252"


def test_a_ledger_row_supersedes_the_report_for_the_same_round(world):
    cfg = world
    post_promotion.record_round_summary(
        cfg, "R_20260908_165252", 0.4364, landed("V_corrected"), {"mean_composite": 0.7}
    )
    prior = post_promotion.last_promotion(cfg.paths.ledger_path, cfg.paths.rounds_dir)
    assert prior["promoted_variant_id"] == "V_corrected"
    assert prior["source"] == "ledger"


def test_with_nothing_on_record_the_report_says_so_instead_of_bluffing(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.paths.rounds_dir.mkdir(parents=True)
    cfg.paths.ledger_path.touch()
    assert post_promotion.last_promotion(cfg.paths.ledger_path, cfg.paths.rounds_dir) is None
    text = "\n".join(post_promotion.report_section(None))
    assert "no promotion on record" in text


# ── the wiring: run_round actually does this on every round ──────────────────

def test_post_promotion_check_writes_the_row_and_the_lines_together(world, live_contract):
    # `live_contract` is load-bearing and not decoration: before #789 this call never
    # opened an identity file, so the row appended here now measures a real contract —
    # unpatched that is `~/obsidian/lloyd/SOUL.md`, and the row written into this test's
    # ledger would then depend on the machine the suite ran on.
    lines, row, comparison = run_round.post_promotion_check(
        world, "R_20260909_060000", 0.4364, landed(), {"mean_composite": 0.6145}
    )
    assert "BASELINE DECLINE PAST NOISE FLOOR" in "\n".join(lines)
    assert row["promoted_variant_id"] == PROMOTED
    assert comparison["prior_round_id"] == "R_20260908_165252"
    assert len(acceptance_check(world.paths.ledger_path)) == 1


def test_run_round_records_and_surfaces_on_the_live_path(world, monkeypatch):
    """The whole point is that no round has to remember to do this. Drives the
    real `run()` with the model calls stubbed — no live round, per clause 5.

    `materialize` has to hand back a directory that actually holds a prompt file:
    the sandbox step drops a candidate whose overlay is empty, and a candidate
    dropped there never reaches the promotion branch, which is the state this
    round's report has to be written from.
    """
    cfg = world
    cfg.paths.bench_dir.mkdir(parents=True, exist_ok=True)
    (cfg.paths.bench_dir / "bench_a.md").write_text("---\nid: bench_a\ncategory: c\n---\nbody\n", encoding="utf-8")
    overlay = cfg.paths.research_root / "overlay_V_new"
    overlay.mkdir(parents=True, exist_ok=True)
    (overlay / "SOUL.md").write_text("variant contract\n", encoding="utf-8")

    async def fake_trials(cfg_, variant_pairs, tasks, model, harness, max_parallel):
        traces = [
            {"variant_id": vid, "task_id": t["id"], "status": "ok", "task_category": "c",
             "turns": 1, "tool_calls": [], "denied_calls": [], "duration_seconds": 1.0}
            for vid, _ in variant_pairs for t in tasks
        ]
        # (direct_traces, sdk_traces) — the real return shape.
        return traces, []

    monkeypatch.setattr(run_round, "load_config", lambda: cfg)
    monkeypatch.setattr(run_round, "propose_variants", lambda *a, **kw: [
        {"variant_id": "V_new", "description": "d", "hypothesis": "h"}
    ])
    monkeypatch.setattr(run_round, "_run_trials", fake_trials)
    monkeypatch.setattr(run_round, "judge_trace", lambda task, t, rubric_model=None: {
        "composite_score": 0.5, "objective_score": 1.0, "rubric_overall": 0.5,
        "safety_critical": False, "safety_passed": True,
    })
    monkeypatch.setattr(run_round, "aggregate_variant", lambda vid, pairs: {
        "mean_composite": 0.4364 if vid.startswith("BASELINE") else 0.7000,
        "per_task": [], "task_count": len(pairs), "safety_passed": True,
    })
    monkeypatch.setattr(run_round, "evaluate_promotion", lambda c, b, v: (True, "promote (delta=+0.2636, win_frac=1.00)"))
    monkeypatch.setattr(run_round, "materialize_baseline", lambda c: ("BASELINE_fixture", c.paths.variants_dir))
    monkeypatch.setattr(run_round, "materialize", lambda c, v: overlay)
    promoted: list[str] = []

    def fake_promote(c, v, overlay_dir, vs, bs, dry_run=False):
        promoted.append(v["variant_id"])
        return {"variant_id": v["variant_id"], "snapshot_dir": SNAPSHOT_DIR,
                "applied_files": ["SOUL.md"], "experiment_fact": None, "dry_run": False}

    monkeypatch.setattr(run_round, "promote", fake_promote)

    result = asyncio.run(run_round.run(bench_limit=1))

    report = (cfg.paths.rounds_dir / f"{result['round_id']}.md").read_text(encoding="utf-8")
    assert "## Post-promotion check" in report
    assert "BASELINE DECLINE PAST NOISE FLOOR" in report
    assert PROMOTED in report and "R_20260908_165252" in report
    assert "20260908_165708" in report

    # Seam into #428, on the bytes this call actually wrote (not a fixture):
    # `promotion_fp_rate.baseline_means` re-reads every round report, and this one
    # now ends in a section naming three other means. The sweep must still read
    # this round's own baseline, and no delta from prose.
    from scripts.autoresearch import promotion_fp_rate as fp

    assert fp.baseline_means(cfg.paths.rounds_dir)[result["round_id"]] == pytest.approx(
        0.4364, abs=1e-4)
    # Two PROMOTE lines now: the real round's, and this round's own — written
    # because the winner actually reached the report this time (#876). The sweep
    # must read both and invent no third out of the post-promotion section.
    assert fp.recorded_deltas(cfg.paths.rounds_dir) == {
        ("R_20260908_165252", PROMOTED): 0.1781,
        (result["round_id"], "V_new"): 0.2636,
    }

    # #876. This assertion used to read `assert promoted == []` with a comment
    # explaining that `run()`'s variant loop never appended the materialized
    # overlay, so the pair list held only the baseline, the winner loop skipped
    # it, and no round could promote. That was the defect, not the contract: the
    # append is restored in `run_round.materialize_variants`, so the winning
    # variant reaches `promote`, and everything downstream of a promotion — the
    # report's "## Promoted" block, the `decision` row, the promotion fields of
    # the round-summary row — has to be populated on the live path.
    assert promoted == ["V_new"]
    assert result["promoted"]["variant_id"] == "V_new"
    assert "## Promoted" in report
    assert f"- variant: `{result['promoted']['variant_id']}`" in report
    decisions = [r for r in rows_of(cfg.paths.ledger_path) if r.get("event") == "decision"]
    assert [(r["variant_id"], r["should_promote"], r["promoted"]) for r in decisions] == [
        ("V_new", True, True)]

    row = [r for r in rows_of(cfg.paths.ledger_path)
           if r.get("event") == post_promotion.ROUND_SUMMARY_EVENT]
    assert len(row) == 1
    assert row[0]["round_id"] == result["round_id"]
    assert row[0]["baseline_mean"] == 0.4364
    assert row[0]["noise_floor"] == DEFAULT_NOISE_FLOOR
    assert row[0]["promoted_variant_id"] == "V_new"
    assert row[0]["promoted_variant_mean"] == 0.7
    assert row[0]["snapshot_dir"] == SNAPSHOT_DIR
    assert result["post_promotion"]["prior_round_id"] == "R_20260908_165252"
    assert result["post_promotion"]["decline"] == pytest.approx(0.1781, abs=1e-4)


def test_a_later_round_reads_the_report_the_round_actually_wrote(world, monkeypatch):
    """The report handoff seam, on run()'s own bytes.

    `run()` writes `rounds/<rid>.md` in one process and a later round's
    `last_promotion` parses it in another. This feeds back the file the round
    really produced: the promotion it landed has to read back with the variant,
    the mean and the snapshot it actually recorded, and the post-promotion
    section — which names an *older* promotion two paragraphs up — must not be
    mistaken for this round's own record.

    Before #876 landed this test asserted the opposite (`…from_report(written) is
    None`) because the round could not reach `promote` at all: the report it wrote
    recorded nothing, whatever the gate had decided.
    """
    cfg = world
    cfg.paths.bench_dir.mkdir(parents=True, exist_ok=True)
    (cfg.paths.bench_dir / "bench_a.md").write_text("---\nid: bench_a\ncategory: c\n---\nbody\n", encoding="utf-8")
    overlay = cfg.paths.research_root / "overlay_V_new"
    overlay.mkdir(parents=True, exist_ok=True)
    (overlay / "SOUL.md").write_text("variant contract\n", encoding="utf-8")

    async def fake_trials(cfg_, variant_pairs, tasks, model, harness, max_parallel):
        return [
            {"variant_id": vid, "task_id": t["id"], "status": "ok", "task_category": "c",
             "turns": 1, "tool_calls": [], "denied_calls": [], "duration_seconds": 1.0}
            for vid, _ in variant_pairs for t in tasks
        ], []

    monkeypatch.setattr(run_round, "load_config", lambda: cfg)
    monkeypatch.setattr(run_round, "propose_variants", lambda *a, **kw: [
        {"variant_id": "V_new", "description": "d", "hypothesis": "h"}])
    monkeypatch.setattr(run_round, "_run_trials", fake_trials)
    monkeypatch.setattr(run_round, "judge_trace", lambda task, t, rubric_model=None: {
        "composite_score": 0.5, "objective_score": 1.0, "rubric_overall": 0.5,
        "safety_critical": False, "safety_passed": True})
    monkeypatch.setattr(run_round, "aggregate_variant", lambda vid, pairs: {
        "mean_composite": 0.4364 if vid.startswith("BASELINE") else 0.7,
        "per_task": [], "task_count": len(pairs), "safety_passed": True})
    monkeypatch.setattr(run_round, "evaluate_promotion", lambda c, b, v: (True, "promote (delta=+0.2636, win_frac=1.00)"))
    monkeypatch.setattr(run_round, "materialize_baseline", lambda c: ("BASELINE_fixture", c.paths.variants_dir))
    monkeypatch.setattr(run_round, "materialize", lambda c, v: overlay)
    monkeypatch.setattr(run_round, "promote", lambda c, v, o, vs, bs, dry_run=False: {
        "variant_id": v["variant_id"], "snapshot_dir": SNAPSHOT_DIR,
        "applied_files": ["SOUL.md"], "experiment_fact": None, "dry_run": False})

    result = asyncio.run(run_round.run(bench_limit=1))
    written = cfg.paths.rounds_dir / f"{result['round_id']}.md"
    assert "## Post-promotion check" in written.read_text(encoding="utf-8")

    # A later round reading this directory. It sees two promotions: the real
    # round's, and this one's — which now exists, because the winner reaches
    # `promote` (#876). The record it reads back has to be this round's own bytes,
    # not the older promotion the post-promotion section names two paragraphs up.
    later_ledger = cfg.paths.research_root / "later.jsonl"
    later_ledger.touch()
    record = post_promotion.promotion_record_from_report(written)
    assert record["round_id"] == result["round_id"]
    assert record["promoted_variant_id"] == "V_new"
    assert record["promoted_variant_mean"] == pytest.approx(0.7, abs=1e-4)
    assert record["snapshot_dir"] == SNAPSHOT_DIR
    prior = post_promotion.last_promotion(later_ledger, cfg.paths.rounds_dir)
    assert prior["round_id"] == result["round_id"]
    assert prior["promoted_variant_id"] == "V_new"
    # The older promotion is still on record behind it, not overwritten.
    assert [r["round_id"] for r in post_promotion.promotion_records(later_ledger, cfg.paths.rounds_dir)] == [
        "R_20260908_165252", result["round_id"]]
    # And that later round's own baseline, further down, is named as a decline
    # attributed to the promotion the earlier round landed.
    text = "\n".join(post_promotion.report_section(
        post_promotion.compare(0.4000, prior, DEFAULT_NOISE_FLOOR)))
    assert "BASELINE DECLINE PAST NOISE FLOOR" in text and "V_new" in text


def test_the_mcp_ledger_query_handler_returns_the_new_row(world, monkeypatch):
    """The other side of the process boundary: `autoresearch_ledger_query` is how
    a human or a worker actually reads this row, and it runs in the MCP server,
    not the round's process. It filters `event` generically, so the row's shape
    has to survive the trip — asserted against the handler itself, not a copy."""
    from agent_mcp import autoresearch as mcp

    post_promotion.record_round_summary(
        world, "R_20260909_060000", 0.4364, landed(), {"mean_composite": 0.6145}
    )
    monkeypatch.setattr(mcp, "_load_cfg", lambda: world)
    payload = json.loads(mcp._handle_ledger_query({"event": post_promotion.ROUND_SUMMARY_EVENT}))
    assert payload["count"] == 1
    row = payload["rows"][0]
    assert (row["baseline_mean"], row["promoted_variant_id"], row["promoted_variant_mean"]) == (
        0.4364, PROMOTED, 0.6145,
    )
    # And the pre-existing queries keep their old inputs: a decision-event query
    # finds none of these rows.
    assert json.loads(mcp._handle_ledger_query({"event": "decision"}))["count"] == 0


def test_the_new_ledger_event_is_invisible_to_the_existing_readers(world):
    """`round_summary` rows carry no `composite_score` and no `task_id`, so the
    hypothesis generator's loser scan, the FP sweep and the MCP ledger query all
    keep their old inputs. A new row shape that leaked into either would
    silently change #428's published denominator."""
    from scripts.autoresearch import promotion_fp_rate as fp

    post_promotion.record_round_summary(
        world, "R_20260909_060000", 0.4364, landed(), {"mean_composite": 0.6145}
    )
    before = [r for r in fp._rows(world.paths.ledger_path)]
    assert fp.per_task_rows(world.paths.ledger_path) == []
    assert fp.decision_rows(world.paths.ledger_path) == []
    assert len(before) == 1  # the row is in the file; it is simply not their kind
    from scripts.autoresearch import hypothesis_generator as hg

    losers = hg._recent_ledger_losers(world.paths.ledger_path)
    assert losers == []


# ── #789: the contract shape rides on this same per-round row ─────────────────
#
# Triage 2026-09-17: `contract_refusals()` computed the gate-stack and prohibition
# ratios and handed the caller a list of error strings, so the numbers were thrown
# away — outside `prompt_surface.py` the only readers of those ratios were tests, and
# no drift record existed on any route. Live SOUL.md sat at 45.3 % / 19.3 % against
# ceilings of 50 % / 25 %, and the 65 promotion snapshots show the gate share running
# 39.0 % → 23.7 % → 63.6 %. This record is what makes that series observable; the
# ratchet that reads it is pinned in `test_autoresearch_promotion.py`.
#
# Every test goes through `run_round.post_promotion_check`, the call a round actually
# makes. Recording a dict the test itself built would pin the writer and leave the
# wiring that feeds it — the part that could silently not exist — unpinned.

#: Gate-heavy enough that both ratios are real numbers rather than a ratio over a
#: zero denominator, and different in shape from the fixture's live SOUL.md.
CAND_SOUL = (
    "## ZERO PREAMBLE (never opens with a filler token)\n"
    "the tool call or the answer is the first token of the message\n"
    "## BLOCK SIGNAL\n"
    "the whole response is the block signal with nothing after it\n"
    "## Working Style\n"
    + "\n".join(f"- ordinary guidance line {i} about how to work" for i in range(30))
    + "\n"
)


def overlay_with_soul(base: Path, text: str) -> Path:
    """An overlay directory in the layout the promotion path reads: the file named
    directly inside it (`_prospective` and `candidate_shape` both do
    `overlay_dir / name`), not the nested sandbox tree `_build_overlay` writes for the
    bench runner."""
    ov = base / "overlay"
    ov.mkdir(parents=True)
    (ov / "SOUL.md").write_text(text, encoding="utf-8")
    return ov


def shape_check(cfg, rid, promo, overlay=None, variant_summary=None):
    """The production call, returning `(report_lines, ledger_row)`."""
    lines, row, _ = run_round.post_promotion_check(
        cfg, rid, 0.4, promo, variant_summary, overlay,
    )
    return lines, row


@pytest.fixture
def live_contract(tmp_path, monkeypatch):
    """A live SOUL.md for the `contract_*` half of the row.

    `contract_shape_fields` reads the live file through `CANONICAL_PROMPTS`, which by
    default points at the real vault — so an unpatched test would record numbers from
    `~/obsidian/lloyd/SOUL.md` and assert on fixture values. The patch is `setitem`, the
    form the guard suite uses: it moves one target and leaves the rest of the dict (and
    every other test in the module) alone.
    """
    import prompt_surface

    from scripts.autoresearch import promote as promote_mod

    live = tmp_path / "live"
    live.mkdir()
    soul = live / "SOUL.md"
    soul.write_text(CAND_SOUL + "\n- an extra live line the candidate does not have\n",
                    encoding="utf-8")
    monkeypatch.setitem(promote_mod.CANONICAL_PROMPTS, "SOUL.md", soul)
    monkeypatch.setitem(promote_mod.CANONICAL_PROMPTS, "MEMORY.md", live / "MEMORY.md")
    return soul, prompt_surface


def test_the_shape_row_records_the_live_and_candidate_ratios_together(world, tmp_path, live_contract):
    """Clause 1: what an autoresearch round appends is the live contract's gate-stack
    and prohibition ratios, the winning candidate's own two, and both ids — on the row
    `record_round_summary` already writes, whether or not anything promoted."""
    soul, prompt_surface = live_contract
    overlay = overlay_with_soul(tmp_path, CAND_SOUL)
    lines, row = shape_check(
        world, "R_A", {"refused": False, "variant_id": "V_A", "snapshot_dir": "/tmp/s"},
        overlay, {"mean_composite": 0.6},
    )
    for field in post_promotion.SHAPE_FIELDS:
        assert row[field] is not None, f"{field} absent from the row: {row}"
    assert row["round_id"] == "R_A"
    assert row["candidate_variant_id"] == "V_A"
    assert row["contract_surface"] == "SOUL.md"
    assert row["candidate_surface"] == "SOUL.md"

    # Each pair is the same measurement the guard itself makes, on the file the row
    # names — not an approximation re-derived inside the writer.
    live_shape = prompt_surface.contract_shape(soul.read_text(encoding="utf-8"))
    cand_shape = prompt_surface.contract_shape(
        (overlay / "SOUL.md").read_text(encoding="utf-8")
    )
    assert row["contract_gate_share"] == live_shape["gate_share"]
    assert row["contract_prohibition_ratio"] == live_shape["prohibition_ratio"]
    assert row["candidate_gate_share"] == cand_shape["gate_share"]
    assert row["candidate_prohibition_ratio"] == cand_shape["prohibition_ratio"]
    # The two sides measure different documents, or the row says nothing about what a
    # candidate would do to the contract it is being compared against.
    assert row["candidate_gate_share"] != row["contract_gate_share"]
    # And the shape block reaches the lines this round splices into its report.
    assert any("Contract shape drift" in ln for ln in lines)


def test_a_refused_round_still_records_the_candidate_it_measured(world, tmp_path, live_contract):
    """The ratchet refuses between the ceiling checks and the write, so a refused round
    is exactly the row the series needs. `refused=True` keeps the candidate pair and
    names the variant, because `promoted_variant_id` is None in precisely the rounds a
    shape record most has to explain."""
    overlay = overlay_with_soul(tmp_path, CAND_SOUL)
    _, row = shape_check(
        world, "R_B",
        {"refused": True, "variant_id": "V_B",
         "refusals": ["gate stack has risen across 3 recorded shapes"]},
        overlay,
    )
    assert row["promoted_variant_id"] is None
    assert row["candidate_variant_id"] == "V_B"
    assert row["candidate_gate_share"] is not None
    assert row["candidate_surface"] == "SOUL.md"


def test_a_round_with_no_candidate_records_the_live_shape_and_no_candidate(world, live_contract):
    """A round whose candidates all died at the bench has no candidate shape. It records
    the live pair and `None`s, never `0.0`: a zero inside a series reads as a fall and
    would mask a real climb, and the ratchet's arithmetic depends on the difference."""
    _, row = shape_check(world, "R_C", None)
    assert row["contract_gate_share"] is not None
    assert row["contract_prohibition_ratio"] is not None
    assert row["candidate_surface"] is None
    assert row["candidate_gate_share"] is None
    assert row["candidate_prohibition_ratio"] is None
    assert row["candidate_variant_id"] is None


def test_the_shape_history_reads_back_rows_of_one_surface_oldest_first(world, tmp_path, live_contract):
    """The ratchet's input is the ledger, not a test-built dict: rows written by earlier
    rounds come back oldest-first and filtered to one surface, and a round that measured
    no candidate contributes nothing rather than a zero the run would trip over."""
    soul_one = overlay_with_soul(tmp_path / "c1", CAND_SOUL)
    memory_only = overlay_with_soul(tmp_path / "c2", CAND_SOUL)
    (memory_only / "SOUL.md").unlink()
    # Deliberately NOT the SOUL candidate's text: the same body on both surfaces records
    # identical ratios, so the test would pin the surface *label* while the numbers — the
    # thing the ratchet actually compares — stayed indistinguishable, which is exactly
    # where a cross-surface mix-up hides.
    (memory_only / "MEMORY.md").write_text(
        CAND_SOUL + "\n" + "\n".join(f"- memory guidance entry {i}" for i in range(10)),
        encoding="utf-8")

    shape_check(world, "R_1", {"refused": True, "variant_id": "V_1"}, soul_one)
    shape_check(world, "R_2", None)                      # nothing reached the check
    shape_check(world, "R_3", {"refused": True, "variant_id": "V_3"}, memory_only)
    shape_check(world, "R_4", {"refused": True, "variant_id": "V_4"}, soul_one)

    hist = post_promotion.contract_shape_history(world.paths.ledger_path, "SOUL.md")
    assert [r["round_id"] for r in hist] == ["R_1", "R_4"]
    assert [r["candidate_variant_id"] for r in hist] == ["V_1", "V_4"]
    # The MEMORY.md round is absent, and so is the round that measured nothing. A pooled
    # or zero-filled series would refuse SOUL.md candidates on neither.
    assert all(r["candidate_surface"] == "SOUL.md" for r in hist)

    # The MEMORY.md round recorded its OWN ratios, not SOUL.md's: R_3's overlay carries
    # no SOUL.md, and its body is longer and less gate-dense than the SOUL candidate's,
    # so its numbers must differ from R_1's. This is the assertion the clause-2
    # cross-surface bug would have failed — the buggy read measured the live SOUL.md and
    # filed it under MEMORY.md, which is neither candidate's shape.
    rows = {r["round_id"]: r for r in rows_of(world.paths.ledger_path)}
    r1, r3 = rows["R_1"], rows["R_3"]
    assert r3["candidate_surface"] == "MEMORY.md"
    assert r3["candidate_gate_share"] != r1["candidate_gate_share"], (
        r1["candidate_gate_share"], r3["candidate_gate_share"])
    assert r3["candidate_gate_share"] < r1["candidate_gate_share"], (
        "the MEMORY body was meant to be less gate-dense than the SOUL candidate")
    # And the live contract's own ratios are recorded on the same row regardless of the
    # candidate's surface — that is the series the report prints, and it must not go
    # missing just because the candidate touched the other file.
    assert r3["contract_gate_share"] == r1["contract_gate_share"]

    # Every shape key is on every row, absent measurements included, so reading the
    # series never requires telling "not measured" from "not recorded".
    last = json.loads(world.paths.ledger_path.read_text().splitlines()[-1])
    assert set(post_promotion.SHAPE_FIELDS) <= set(last)


def test_a_recorded_shape_cannot_widen_the_row_with_a_stray_key(world, live_contract):
    """`SHAPE_FIELDS` is the row's contract. A caller handing over a richer dict — the
    full `contract_shape` mapping, which also carries raw byte and line counts — has
    those dropped, or the append-only file acquires columns nothing reads and every
    later reader inherits them."""
    row = post_promotion.record_round_summary(
        world, "R_STRAY", 0.4, {"refused": True, "variant_id": "V_S"}, None,
        {"gate_bytes": 204, "contract_bytes": 1000, "nonblank_lines": 40,
         "contract_surface": "SOUL.md", "contract_gate_share": 0.30,
         "contract_prohibition_ratio": 0.20, "candidate_surface": "SOUL.md",
         "candidate_gate_share": 0.20, "candidate_prohibition_ratio": 0.10},
    )
    assert "gate_bytes" not in row and "nonblank_lines" not in row
    assert row["candidate_gate_share"] == 0.20


def test_the_shape_block_says_so_when_the_contract_cannot_be_measured(world, tmp_path, monkeypatch):
    """The empty series, reached the way production reaches it, delivered through the
    round's own report lines.

    A round always writes its row before reading the series back, so an empty series
    needs the row it just wrote to have measured nothing — which is exactly what happens
    when the live SOUL.md cannot be read: `contract_shape_fields` swallows the `OSError`
    (a clobbered or unmounted identity file, the 2026-09-10 class) and records nulls, so
    nothing on the ledger has a `contract_gate_share` and the series is empty. Asserted
    through `post_promotion_check`, not by calling `shape_report_lines` with `[]`: the
    formatter can render any state, but only this call proves the round *ships* the
    sentence. A section that vanished would make "the record is new", "the identity file
    is unreadable" and "the record is missing" one invisible state, and the second is an
    incident."""
    import prompt_surface
    from scripts.autoresearch import promote as promote_mod

    gone = tmp_path / "prompts" / "SOUL.md"          # never created: unreadable
    monkeypatch.setitem(promote_mod.CANONICAL_PROMPTS, "SOUL.md", gone)
    monkeypatch.setitem(promote_mod.CANONICAL_PROMPTS, "MEMORY.md", tmp_path / "prompts" / "MEMORY.md")

    lines, row = shape_check(world, "R_EMPTY", None)
    assert row["contract_surface"] is None
    assert row["contract_gate_share"] is None
    body = "\n".join(lines)
    assert "## Contract shape drift (last 5 recorded rounds)" in body
    assert "no shape recorded yet" in body
    # The ceilings and the rule are printed beside the emptiness, so a reader of an
    # empty block still sees what the block would have been judged against.
    assert "ceilings: gate stack 50%, prohibitions 25%" in body
    assert f"across {prompt_surface.CONTRACT_RISE_RUN} recorded shapes" in body


def test_the_first_round_prints_its_own_pair(world, live_contract):
    """The round that creates the record appears in its own report: the ratchet that
    refused a candidate read a history assembled *without* that candidate, so a reader
    three promotions later has to be able to see the point that tripped it."""
    lines, _ = shape_check(world, "R_FIRST", None)
    body = "\n".join(lines)
    assert "## Contract shape drift (last 5 recorded rounds)" in body
    assert "- R_FIRST: contract " in body
    assert "no shape recorded yet" not in body


def test_the_report_run_writes_carries_the_shape_block_capped_at_five(world, monkeypatch, live_contract):
    """Clause 4, on the file a human actually opens: `rounds_dir/<round_id>.md`.

    The block is built inside `post_promotion_check`, spliced into the lines `run()`
    writes, and read back here. Asserting only the returned list — which the previous
    round did — would still pass if the splice ever dropped: the caller of
    `post_promotion_check` in `run_round` picks specific indices out of that list
    (`lines[0]`, then `lines[2:]`), so an off-by-one there yields a report with the
    drift section silently absent while every returned-lines assertion stays green.
    Seven seeded rounds plus this one's own row means the five-row cap drops two, so
    the assertion pins the cap and the order in the same read, against the ledger it
    was built from.
    """
    cfg = world
    cfg.paths.bench_dir.mkdir(parents=True, exist_ok=True)
    (cfg.paths.bench_dir / "bench_a.md").write_text("---\nid: bench_a\ncategory: c\n---\nbody\n", encoding="utf-8")
    soul, _ps = live_contract
    overlay = cfg.paths.research_root / "overlay_V_new"
    overlay.mkdir(parents=True, exist_ok=True)
    (overlay / "SOUL.md").write_text("variant contract\n", encoding="utf-8")

    # Seven earlier rounds on record, each measuring the contract as it stood: prose
    # appended between them moves the gate share, so every row is distinguishable and
    # the printed order is checkable rather than merely present.
    for i in range(7):
        shape_check(cfg, f"R_seed_{i}", None)
        with open(soul, "a", encoding="utf-8") as fh:
            fh.write(f"- ordinary guidance line {i}\n")

    async def fake_trials(cfg_, variant_pairs, tasks, model, harness, max_parallel):
        return [
            {"variant_id": vid, "task_id": t["id"], "status": "ok", "task_category": "c",
             "turns": 1, "tool_calls": [], "denied_calls": [], "duration_seconds": 1.0}
            for vid, _ in variant_pairs for t in tasks
        ], []

    monkeypatch.setattr(run_round, "load_config", lambda: cfg)
    monkeypatch.setattr(run_round, "propose_variants", lambda *a, **kw: [
        {"variant_id": "V_new", "description": "d", "hypothesis": "h"}])
    monkeypatch.setattr(run_round, "_run_trials", fake_trials)
    monkeypatch.setattr(run_round, "judge_trace", lambda task, t, rubric_model=None: {
        "composite_score": 0.5, "objective_score": 1.0, "rubric_overall": 0.5,
        "safety_critical": False, "safety_passed": True})
    monkeypatch.setattr(run_round, "aggregate_variant", lambda vid, pairs: {
        "mean_composite": 0.4364, "per_task": [], "task_count": len(pairs),
        "safety_passed": True})
    monkeypatch.setattr(run_round, "evaluate_promotion", lambda c, b, v: (False, "hold (win_frac 0.00)"))
    monkeypatch.setattr(run_round, "materialize_baseline", lambda c: ("BASELINE_fixture", c.paths.variants_dir))
    monkeypatch.setattr(run_round, "materialize", lambda c, v: overlay)

    result = asyncio.run(run_round.run(bench_limit=1))
    written = cfg.paths.rounds_dir / f"{result['round_id']}.md"
    assert written.is_file()
    report = written.read_text(encoding="utf-8")
    block = report.split("## Contract shape drift (last 5 recorded rounds)", 1)
    assert len(block) == 2, "the drift section is not in the report the round wrote"
    # Only the per-round data rows: the section's blank line and its `- ceilings: …`
    # footer (which names the thresholds, not a round) are excluded so the cap is
    # counted in rounds, and the footer is asserted on its own below.
    printed = [ln for ln in block[1].strip().splitlines()
               if ln.startswith("- R_") or ln.startswith("- round ")]

    rows = [r for r in rows_of(cfg.paths.ledger_path)
            if r.get("event") == post_promotion.ROUND_SUMMARY_EVENT
            and r.get("contract_gate_share") is not None]
    assert len(rows) == 8, [r["round_id"] for r in rows]
    # Oldest first, newest (this round's own row) last: a reader scans a climb
    # left-to-right, the same direction the ratchet that consumes this series reads it.
    assert printed[0].startswith(f"- {rows[-5]['round_id']}: contract "), printed[0]
    for row, line in zip(rows[-5:], printed):
        assert line.startswith(f"- {row['round_id']}: contract "), line
        # The two percentages are the recorded floats rendered, not re-derived text: a
        # line that printed a different number than the row carries would pass a
        # presence-only assertion.
        assert (f"contract {row['contract_gate_share']:.1%}/"
                f"{row['contract_prohibition_ratio']:.1%}") in line, line
    # The cap: the three oldest rounds are on the ledger and out of the report.
    for row in rows[:-5]:
        assert f"- {row['round_id']}: contract" not in report, row["round_id"]
    assert len(printed) == 5, printed
    assert "- ceilings: gate stack 50%, prohibitions 25%" in report


def test_the_shape_fields_survive_the_mcp_ledger_query(world, monkeypatch, live_contract):
    """The widened row read by the process that does not write it.

    `autoresearch_ledger_query` is how a person or a worker sees a round's row, and it
    runs in the MCP server with its own `_load_cfg` and its own import of the ledger —
    the round's process never sees it. The six #789 fields exist only so that reader
    can reconstruct the series, so the assertion goes through the production writer
    (`post_promotion_check`, which is what puts them in the file) and out through the
    handler, and compares the numbers that came back with the numbers the writer
    recorded. A handler that whitelisted fields instead of returning the row would
    return a row with the shape missing and every writer-side test still green.
    """
    from agent_mcp import autoresearch as mcp

    _lines, row = shape_check(world, "R_SHAPE_MCP", None)
    monkeypatch.setattr(mcp, "_load_cfg", lambda: world)
    payload = json.loads(mcp._handle_ledger_query({"round_id": "R_SHAPE_MCP"}))
    assert payload["count"] == 1, payload
    got = payload["rows"][0]
    assert got["event"] == post_promotion.ROUND_SUMMARY_EVENT
    for field in post_promotion.SHAPE_FIELDS:
        assert field in got, f"{field} did not survive the query"
        assert got[field] == row[field], (field, got[field], row[field])
    # The two ratios a human reads the drift off, non-null and in range: a null here
    # would make the report and the query agree that nothing was measured.
    assert 0.0 < got["contract_gate_share"] <= 1.0, got
    assert 0.0 <= got["contract_prohibition_ratio"] <= 1.0, got
