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

def test_post_promotion_check_writes_the_row_and_the_lines_together(world):
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
    # The only delta on record is the real round's PROMOTE line; this report has no
    # PROMOTE line (see below) and must not have invented one out of the section.
    assert fp.recorded_deltas(cfg.paths.rounds_dir) == {
        ("R_20260908_165252", PROMOTED): 0.1781}

    # `promote` is NOT reached, and not for want of a stub: run_round's variant
    # loop never appends the materialized overlay to `variant_pairs`
    # (scripts/autoresearch/run_round.py:183-192), so the pair list holds only the
    # baseline, the winner loop skips it, and no round can promote. Filed as a
    # blocker on this item; the landed-promotion case is pinned here instead, one
    # frame closer, at the single function `run()` calls.
    assert promoted == []
    assert result["promoted"] is None
    row = [r for r in rows_of(cfg.paths.ledger_path)
           if r.get("event") == post_promotion.ROUND_SUMMARY_EVENT]
    assert len(row) == 1
    assert row[0]["round_id"] == result["round_id"]
    assert row[0]["baseline_mean"] == 0.4364
    assert row[0]["noise_floor"] == DEFAULT_NOISE_FLOOR
    assert row[0]["promoted_variant_id"] is None
    assert result["post_promotion"]["prior_round_id"] == "R_20260908_165252"
    assert result["post_promotion"]["decline"] == pytest.approx(0.1781, abs=1e-4)


def test_a_later_round_reads_the_report_the_round_actually_wrote(world, monkeypatch):
    """The report handoff seam, on run()'s own bytes.

    `run()` writes `rounds/<rid>.md` in one process and a later round's
    `last_promotion` parses it in another. This feeds back the file the round
    really produced: it must not be mistaken for a promotion, the section this
    round appended must not confuse the parser, and the promotion that *is* on
    record in the same directory must still be the one a later round acts on.
    (A promotion-bearing report is exercised on run()'s own historical output in
    `test_a_real_round_report_reads_back_as_a_promotion_record`, and that fixture
    is held to the writer's current format by
    `test_the_fixture_and_the_writer_agree_about_one_round`.)
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

    # A later round reading this directory: this round promoted nothing, so the
    # promotion on record is still the one it was comparing against.
    later_ledger = cfg.paths.research_root / "later.jsonl"
    later_ledger.touch()
    assert post_promotion.promotion_record_from_report(written) is None
    prior = post_promotion.last_promotion(later_ledger, cfg.paths.rounds_dir)
    assert prior["round_id"] == "R_20260908_165252"
    assert prior["promoted_variant_id"] == PROMOTED
    # And that later round's own baseline, further down, is named as a decline
    # attributed to the promotion the earlier round surfaced.
    text = "\n".join(post_promotion.report_section(
        post_promotion.compare(0.4000, prior, DEFAULT_NOISE_FLOOR)))
    assert "BASELINE DECLINE PAST NOISE FLOOR" in text and PROMOTED in text


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
