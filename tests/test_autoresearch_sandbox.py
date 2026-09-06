"""Variant sandbox materialization + shared config, bench loading, ledger, run spec.

Why this file exists
--------------------
Everything in `scripts/autoresearch/` that decides what gets evaluated, and
where, was untested. Two things here are load-bearing for the self-modification
loop:

  * `materialize()`'s path allowlist is the only thing keeping a variant from
    writing an arbitrary file into the vault. A variant whose `overlay_files`
    names something outside the supported set must be dropped, not written.
  * `load_bench_tasks()` silently *skips* any task whose frontmatter fails to
    parse (common.py:289-293) — a malformed bench file shrinks the denominator
    instead of failing loudly, and `win_frac` granularity is computed off that
    count.

Nothing here touches the vault or vLLM.
"""
from __future__ import annotations

import json

import pytest
import yaml

from scripts.autoresearch import variant_sandbox as vs
from scripts.autoresearch import common
from scripts.autoresearch.common import AutoresearchConfig, AutoresearchPaths


def make_cfg(tmp_path, **over) -> AutoresearchConfig:
    paths = AutoresearchPaths(
        bench_dir=tmp_path / "bench",
        research_root=tmp_path / "research",
        rounds_dir=tmp_path / "rounds",
        ledger_path=tmp_path / "ledger.jsonl",
        variants_dir=tmp_path / "variants",
        snapshots_dir=tmp_path / "snapshots",
        facts_experiments_dir=tmp_path / "facts",
    )
    kw = dict(
        paths=paths, default_model="primary", default_budget_minutes=120,
        max_variants_per_round=7, promotion_min_win_fraction=0.5,
        promotion_min_composite_delta=0.05, promotion_require_safety_pass=True,
        tool_allowlist_consecutive_wins=2, targets=["prompts"],
    )
    kw.update(over)
    return AutoresearchConfig(**kw)


@pytest.fixture
def cfg(tmp_path):
    c = make_cfg(tmp_path)
    c.paths.ensure()
    return c


# ── the allowlist ────────────────────────────────────────────────────────────

def test_supported_paths_are_exactly_the_canonical_prompt_names():
    assert vs._supported_relative_paths() == {"SOUL.md", "MEMORY.md", "USER.md"}


def test_allowlist_has_one_source_of_truth():
    """The gate and the canonical list cannot drift: same function, same keys."""
    assert vs._supported_relative_paths() == set(common._canonical_prompt_paths())


# ── materialize ──────────────────────────────────────────────────────────────

def test_materialize_writes_supported_overlay_files(cfg, tmp_path):
    variant = {
        "variant_id": "V_1",
        "description": "sharpen the gate",
        "hypothesis": "fewer hedge words",
        "overlay_files": {"SOUL.md": "NEW SOUL", "MEMORY.md": "NEW MEM"},
    }
    out = vs.materialize(cfg, variant)
    assert (out / "SOUL.md").read_text() == "NEW SOUL"
    assert (out / "MEMORY.md").read_text() == "NEW MEM"


def test_materialize_refuses_an_unsupported_path(cfg):
    """The whole point: a variant cannot smuggle a path outside the allowlist
    into the vault."""
    variant = {
        "variant_id": "V_evil",
        "overlay_files": {
            "SOUL.md": "fine",
            "config.yaml": "autonomy:\n  enabled: true\n",
            "../lloyd/server.py": "import os",
        },
    }
    out = vs.materialize(cfg, variant)
    assert (out / "SOUL.md").exists()
    assert not (out / "config.yaml").exists()
    assert not (cfg.paths.variants_dir.parent / "server.py").exists()
    assert {p.name for p in out.rglob("*") if p.is_file()} == {"SOUL.md", "variant.json"}


def test_materialize_records_the_attempted_paths_in_meta(cfg):
    """variant.json logs what was *asked for*, including what was refused, so a
    smuggling attempt stays visible in the round record."""
    variant = {"variant_id": "V_2", "overlay_files": {"SOUL.md": "x", "../../etc/passwd": "y"}}
    out = vs.materialize(cfg, variant)
    meta = json.loads((out / "variant.json").read_text())
    assert meta["overlay_files"] == ["../../etc/passwd", "SOUL.md"]
    # Nothing escaped the variant dir: the only files anywhere under the
    # variants root are this variant's own two.
    assert {p.name for p in cfg.paths.variants_dir.rglob("*") if p.is_file()} == {
        "SOUL.md", "variant.json"}


def test_materialize_keeps_parent_lineage_and_metadata(cfg):
    variant = {
        "variant_id": "V_child", "target_surface": "prompts",
        "description": "d", "hypothesis": "h",
        "parent_variant_id": "V_parent",
        "overlay_files": {"MEMORY.md": "m"},
    }
    meta = json.loads((vs.materialize(cfg, variant) / "variant.json").read_text())
    assert meta["parent_variant_id"] == "V_parent"
    assert meta["target_surface"] == "prompts"
    assert meta["description"] == "d" and meta["hypothesis"] == "h"
    assert meta["created_at"].endswith("Z")


def test_materialize_with_no_overlay_files_still_produces_meta(cfg):
    out = vs.materialize(cfg, {"variant_id": "V_empty"})
    assert json.loads((out / "variant.json").read_text())["overlay_files"] == []


def test_materialize_is_idempotent_per_variant_id(cfg):
    """Re-materializing the same id overwrites in place rather than piling up
    directories — the leak that accumulated 2,185 dirs lives in round bookkeeping,
    not here."""
    v = {"variant_id": "V_same", "overlay_files": {"SOUL.md": "one"}}
    vs.materialize(cfg, v)
    v["overlay_files"] = {"SOUL.md": "two"}
    out = vs.materialize(cfg, v)
    assert (out / "SOUL.md").read_text() == "two"
    assert len(list(cfg.paths.variants_dir.iterdir())) == 1


def test_materialize_reuses_the_existing_dir_for_a_repeated_id(cfg):
    """Characterized: the id is a timestamp+uuid4 hex, so a real collision needs
    the same second *and* the same 6 hex chars. mkdir(exist_ok=True) means a
    collision silently reuses the earlier variant's directory."""
    first = vs.materialize(cfg, {"variant_id": "V_dup", "overlay_files": {"SOUL.md": "a"}})
    second = vs.materialize(cfg, {"variant_id": "V_dup", "overlay_files": {"MEMORY.md": "b"}})
    assert first == second


# ── materialize_baseline ─────────────────────────────────────────────────────

def test_baseline_overlay_is_empty_so_everything_falls_through(cfg):
    """`build_system_prompt` grades an empty dir as the canonical prompt. This is
    the comparison every variant is scored against."""
    bid, overlay = vs.materialize_baseline(cfg)
    assert bid.startswith("BASELINE_")
    assert [p.name for p in overlay.iterdir()] == ["variant.json"]
    meta = json.loads((overlay / "variant.json").read_text())
    assert meta["target_surface"] == "baseline" and meta["overlay_files"] == []


def test_two_baseline_materializations_in_one_second_collide(cfg):
    """REAL DEFECT (found writing this file): the id comes from
    `variants_dir.stat().st_ctime`, so two calls with an unchanged directory
    ctime produce the *same* id, and `mkdir(exist_ok=True)` makes the second one
    silently reuse and overwrite the first one's `variant.json`.

    It only works in practice because each round is minutes apart and creating a
    subdir bumps the parent's ctime — an accident of timing, not a design. Two
    rounds launched in the same second would share a baseline directory. Fix:
    derive the id from a run timestamp like `variant_id()` does.
    """
    first_id, first = vs.materialize_baseline(cfg)
    second_id, second = vs.materialize_baseline(cfg)
    assert first_id == second_id, (
        "collision no longer reproduces — either fixed (then delete this test and "
        "add the distinctness assertion) or the ctime dependency changed"
    )
    assert first == second


@pytest.mark.xfail(
    reason="See test_two_baseline_materializations_in_one_second_collide — replace "
           "with a plain assertion once the id comes from the run time.",
    strict=False,
)
def test_baseline_ids_are_distinct_per_call(cfg):
    assert vs.materialize_baseline(cfg)[0] != vs.materialize_baseline(cfg)[0]


def test_baseline_materialization_needs_the_variants_dir_to_exist(tmp_path):
    """`stat()` on a missing dir raises. `AutoresearchPaths.ensure()` is what
    makes this work, and it is called by the round runner — asserting the
    dependency so reordering those calls breaks a test."""
    c = make_cfg(tmp_path)                     # no ensure()
    with pytest.raises(FileNotFoundError):
        vs.materialize_baseline(c)


# ── paths.ensure ─────────────────────────────────────────────────────────────

def test_ensure_creates_every_writable_dir_and_the_ledger(cfg, tmp_path):
    c = make_cfg(tmp_path / "fresh")
    c.paths.ensure()
    for p in (c.paths.research_root, c.paths.rounds_dir, c.paths.variants_dir,
              c.paths.snapshots_dir, c.paths.facts_experiments_dir):
        assert p.is_dir()
    assert c.paths.ledger_path.exists()


def test_ensure_survives_a_second_call(cfg):
    cfg.paths.ensure()


# ── load_config ──────────────────────────────────────────────────────────────

def test_load_config_refuses_a_missing_autoresearch_block(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("autonomy:\n  enabled: true\n", encoding="utf-8")
    monkeypatch.setattr(common, "CONFIG_PATH", cfg_file)
    with pytest.raises(RuntimeError, match="missing 'autoresearch' block"):
        common.load_config()


def test_load_config_refuses_an_empty_autoresearch_block(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("autoresearch: {}\n", encoding="utf-8")
    monkeypatch.setattr(common, "CONFIG_PATH", cfg_file)
    with pytest.raises(RuntimeError, match="missing 'autoresearch' block"):
        common.load_config()


def test_load_config_requires_every_path_key(tmp_path, monkeypatch):
    """A half-populated block must fail loudly at load, not midway through a
    round with prompts already swapped."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump({"autoresearch": {"bench_dir": "/tmp/b"}}),
                        encoding="utf-8")
    monkeypatch.setattr(common, "CONFIG_PATH", cfg_file)
    with pytest.raises(KeyError):
        common.load_config()


def test_load_config_expands_user_paths(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump({"autoresearch": {
        "bench_dir": "~/bench-here", "research_root": "~/r", "rounds_dir": "~/r/rounds",
        "ledger_path": "~/r/l.jsonl", "variants_dir": "~/r/v", "snapshots_dir": "~/r/s",
        "facts_experiments_dir": "~/r/f",
    }}), encoding="utf-8")
    monkeypatch.setattr(common, "CONFIG_PATH", cfg_file)
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = common.load_config()
    assert cfg.paths.bench_dir == tmp_path / "bench-here"
    assert not str(cfg.paths.bench_dir).startswith("~")


def test_load_config_promotion_defaults_are_the_fallback_values(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump({"autoresearch": {
        "bench_dir": "b", "research_root": "r", "rounds_dir": "r", "ledger_path": "r/l",
        "variants_dir": "r/v", "snapshots_dir": "r/s", "facts_experiments_dir": "r/f",
    }}), encoding="utf-8")
    monkeypatch.setattr(common, "CONFIG_PATH", cfg_file)
    cfg = common.load_config()
    assert cfg.promotion_min_win_fraction == 0.60
    assert cfg.promotion_min_composite_delta == 0.05
    assert cfg.promotion_require_safety_pass is True


def test_the_live_config_is_loadable_and_its_gate_is_the_documented_one():
    """Reads the real config.yaml. If someone loosens the gate, this fails and
    the change has to be a deliberate edit with a test in front of it."""
    cfg = common.load_config()
    assert cfg.promotion_require_safety_pass is True
    assert cfg.promotion_min_composite_delta == 0.05
    assert cfg.promotion_min_win_fraction == 0.50
    assert cfg.default_model == "primary"
    assert "prompts" in cfg.targets


# ── load_bench_tasks ─────────────────────────────────────────────────────────

def bench_file(path, **frontmatter):
    path.write_text(
        "---\n" + yaml.safe_dump(frontmatter, sort_keys=False) + "---\nbody prose\n",
        encoding="utf-8",
    )
    return path


def test_missing_bench_dir_yields_no_tasks(tmp_path):
    assert common.load_bench_tasks(tmp_path / "absent") == []


def test_bench_tasks_load_with_ids_and_body(tmp_path):
    bench_file(tmp_path / "bench_001.md", id="bench_001", category="synthetic")
    tasks = common.load_bench_tasks(tmp_path)
    assert len(tasks) == 1
    assert tasks[0]["id"] == "bench_001"
    assert tasks[0]["category"] == "synthetic"
    assert tasks[0]["_body"] == "body prose"
    assert tasks[0]["_path"].endswith("bench_001.md")


def test_missing_id_falls_back_to_the_filename_stem(tmp_path):
    bench_file(tmp_path / "bench_042.md", category="synthetic")
    assert common.load_bench_tasks(tmp_path)[0]["id"] == "bench_042"


def test_tasks_load_in_filename_order(tmp_path):
    for n in ("bench_003", "bench_001", "bench_002"):
        bench_file(tmp_path / f"{n}.md", id=n)
    assert [t["id"] for t in common.load_bench_tasks(tmp_path)] == [
        "bench_001", "bench_002", "bench_003"]


def test_non_markdown_files_are_ignored(tmp_path):
    bench_file(tmp_path / "bench_001.md", id="a")
    (tmp_path / "notes.txt").write_text("---\nid: nope\n---\n", encoding="utf-8")
    assert [t["id"] for t in common.load_bench_tasks(tmp_path)] == ["a"]


def test_file_without_frontmatter_is_skipped(tmp_path):
    (tmp_path / "bench_002.md").write_text("# just a heading\n", encoding="utf-8")
    (tmp_path / "bench_001.md").write_text("---\nid: ok\n---\nx\n", encoding="utf-8")
    assert [t["id"] for t in common.load_bench_tasks(tmp_path)] == ["ok"]


def test_unterminated_frontmatter_is_skipped(tmp_path):
    (tmp_path / "bench_003.md").write_text("---\nid: broken\n", encoding="utf-8")
    assert common.load_bench_tasks(tmp_path) == []


def test_malformed_frontmatter_is_skipped_rather_than_raising(tmp_path):
    """Characterized, and a hole: a task that fails to parse silently leaves the
    bench. `win_frac` granularity is computed off the surviving count, so a
    corrupt file quietly changes the promotion threshold's meaning. The
    companion test in test_bench_invariants.py guards the *live* bench against
    exactly this."""
    (tmp_path / "bench_004.md").write_text("---\nid: [unclosed\n---\n", encoding="utf-8")
    bench_file(tmp_path / "bench_005.md", id="fine")
    assert [t["id"] for t in common.load_bench_tasks(tmp_path)] == ["fine"]


def test_the_live_bench_dir_loads_every_file(tmp_path):
    """Reads the real bench. If a task file ever fails to parse, the loaded count
    drops below the file count and this fails loudly instead of shrinking the
    denominator of a promotion decision."""
    cfg = common.load_config()
    files = sorted(cfg.paths.bench_dir.glob("*.md"))
    tasks = common.load_bench_tasks(cfg.paths.bench_dir)
    assert len(tasks) == len(files) > 0, (
        f"{len(files)} bench files but {len(tasks)} loaded — one is malformed "
        "and was silently skipped"
    )


# ── ledger ───────────────────────────────────────────────────────────────────

def test_ledger_append_is_one_json_line_per_entry(cfg):
    common.ledger_append(cfg.paths.ledger_path, {"event": "spec", "round_id": "R_1"})
    common.ledger_append(cfg.paths.ledger_path, {"event": "decision", "promoted": True})
    lines = cfg.paths.ledger_path.read_text().strip().splitlines()
    assert [json.loads(l)["event"] for l in lines] == ["spec", "decision"]


def test_ledger_append_creates_missing_parent_dirs(tmp_path):
    p = tmp_path / "deep" / "nested" / "l.jsonl"
    common.ledger_append(p, {"a": 1})
    assert p.exists()


def test_ledger_append_never_raises(cfg, tmp_path):
    """Documented best-effort: a ledger that won't write must not abort a round
    after prompts have already been swapped."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    common.ledger_append(blocker / "l.jsonl", {"a": 1})       # no exception


def test_ledger_append_writes_non_ascii_verbatim(cfg):
    common.ledger_append(cfg.paths.ledger_path, {"fact": "Δ +0.2 → 均值"})
    assert "Δ" in cfg.paths.ledger_path.read_text(encoding="utf-8")


# ── find_last_promoted_variant ───────────────────────────────────────────────

def test_no_ledger_means_no_parent(cfg):
    cfg.paths.ledger_path.unlink()
    assert common.find_last_promoted_variant(cfg.paths.ledger_path) is None


def test_unpromoted_decisions_are_not_a_parent(cfg):
    common.ledger_append(cfg.paths.ledger_path,
                         {"event": "decision", "promoted": False, "variant_id": "V_a"})
    assert common.find_last_promoted_variant(cfg.paths.ledger_path) is None


def test_most_recent_promotion_wins(cfg):
    for vid in ("V_old", "V_new"):
        common.ledger_append(cfg.paths.ledger_path,
                             {"event": "decision", "promoted": True, "variant_id": vid})
    assert common.find_last_promoted_variant(cfg.paths.ledger_path)["variant_id"] == "V_new"


def test_promotion_is_enriched_from_the_variant_metadata(cfg):
    vid = "V_rich"
    (cfg.paths.variants_dir / vid).mkdir(parents=True)
    (cfg.paths.variants_dir / vid / "variant.json").write_text(json.dumps({
        "variant_id": vid, "description": "desc here", "hypothesis": "hypo here",
    }), encoding="utf-8")
    common.ledger_append(cfg.paths.ledger_path,
                         {"event": "decision", "promoted": True, "variant_id": vid})
    found = common.find_last_promoted_variant(cfg.paths.ledger_path, cfg.paths.variants_dir)
    assert found["description"] == "desc here" and found["hypothesis"] == "hypo here"


def test_spec_events_and_garbage_lines_are_tolerated(cfg):
    """The ledger holds 24k lines with no `event` key at all; the scan must walk
    past them rather than die."""
    cfg.paths.ledger_path.write_text(
        "not json at all\n"
        + json.dumps({"variant_id": "V_x"}) + "\n"
        + json.dumps({"event": "spec", "promoted": True}) + "\n"
        + json.dumps({"event": "decision", "promoted": True, "variant_id": "V_ok"}) + "\n",
        encoding="utf-8",
    )
    assert common.find_last_promoted_variant(cfg.paths.ledger_path)["variant_id"] == "V_ok"


def test_missing_variant_meta_degrades_to_empty_strings(cfg):
    common.ledger_append(cfg.paths.ledger_path,
                         {"event": "decision", "promoted": True, "variant_id": "V_nometa"})
    found = common.find_last_promoted_variant(cfg.paths.ledger_path, cfg.paths.variants_dir)
    assert found == {"variant_id": "V_nometa", "description": "", "hypothesis": ""}


# ── run spec ─────────────────────────────────────────────────────────────────

def valid_spec():
    return {
        "objective": "improve prompts",
        "evaluation": {"timeout_secs": 300, "command": "judge.py"},
        "budget": {"max_rounds": 0, "max_variants_per_round": 7},
        "mutation_scope": {"writable_paths": ["/tmp/SOUL.md"]},
        "stop_conditions": [],
        "sampling": {"algorithm": "baseline"},
    }


def test_valid_spec_passes():
    assert common.validate_run_spec(valid_spec()) is None


@pytest.mark.parametrize("missing", ["objective", "evaluation", "budget", "mutation_scope"])
def test_each_required_top_level_key_is_required(missing):
    spec = valid_spec()
    spec.pop(missing)
    assert common.validate_run_spec(spec) == f"missing required key: {missing}"


def test_top_level_scalar_types_are_not_validated(cfg):
    """Characterized gap: `RUN_SPEC_SCHEMA` only type-checks keys whose schema
    value is a dict, so top-level scalars are presence-checked only —
    `objective: 42` and `stop_conditions: "yes"` both validate clean.

    Harmless today (the spec is generated, not hand-written), but it means the
    schema cannot catch a hand-edited spec. Tightening it is a behavior change.
    """
    spec = valid_spec()
    spec["objective"] = 42
    spec["stop_conditions"] = "not a list"
    assert common.validate_run_spec(spec) is None


def test_mutation_scope_writable_paths_must_be_a_list():
    spec = valid_spec()
    spec["mutation_scope"]["writable_paths"] = "/tmp/SOUL.md"
    assert "writable_paths" in common.validate_run_spec(spec)


def test_wrongly_typed_nested_object_is_rejected():
    spec = valid_spec()
    spec["budget"] = "unbounded"
    assert "must be an object" in common.validate_run_spec(spec)


def test_run_spec_from_cfg_names_only_the_canonical_prompts_as_writable(cfg):
    """The run spec is the round's own record of what it may overwrite. If it
    ever lists more than the prompt files, the round is claiming a wider blast
    radius than the allowlist grants."""
    spec = common._run_spec_from_cfg(cfg, "primary", None)
    writable = spec["mutation_scope"]["writable_paths"]
    assert len(writable) == 3
    assert all(w.endswith(("SOUL.md", "MEMORY.md", "USER.md")) for w in writable)
    assert common.validate_run_spec(spec) is None


def test_write_run_spec_round_trips(cfg):
    path = common.write_run_spec("R_test", cfg, valid_spec())
    assert path.name == "run_spec.yaml"
    assert common.validate_run_spec(yaml.safe_load(path.read_text())) is None


# ── identifiers ──────────────────────────────────────────────────────────────

def test_variant_ids_are_unique_and_ordered(cfg):
    ids = [common.variant_id() for _ in range(20)]
    assert len(set(ids)) == 20
    # V_<YYYYMMDD>_<HHMMSS>_<6 hex>
    assert all(i.startswith("V_") and len(i.split("_")) == 4 for i in ids)


def test_variant_id_prefix_is_configurable():
    assert common.variant_id("BASE").startswith("BASE_")


def test_round_id_shape():
    rid = common.round_id()
    assert rid.startswith("R_") and len(rid.split("_")) == 3 and rid[2:10].isdigit()
