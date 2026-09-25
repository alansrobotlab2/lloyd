"""#1425: the USER.md-trim A/B driver (`eval/run_memory_trim_ab.py`).

Everything here is offline: synthetic ledger/archive text, tmp overlays, the
real `prompt_builder.build_system_prompt` for arm equality, and stubbed
`run_bench` / grader for the run loop. No engine, no aggregator, no vault.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import run_memory_trim_ab as ab  # noqa: E402

AUDIT = """# audit

## Per-entry ledger

| `## ` section | decision | entry (pre-trim, first 62 chars) | what the trim did | destination |
|---|---|---|---|---|
| infra | move | **Cut rule** — never quote a stale count, re-measure it … | cut from the loaded surface; verbatim text preserved in the archive | archive |
| infra | condense | **Kept rule** — tightened into one line with its command… | absorbed into `## infra` → “**Kept rule**…” (300 B → 120 B, word overlap 0.45) | (stays in USER.md, tightened) |
| infra | move | **Loaded rule** — this one is already in MEMORY.md verbatim… | cut from the loaded surface; verbatim text preserved in the archive | archive |

## After
"""

ARCHIVE = """---
type: note
---

# archive preamble
- not a USER.md bullet

# User (Alan) — Memory & Context

## infra
- **Cut rule** — never quote a stale count, re-measure it in the run that uses it.
- **Kept rule** — tightened into one line with its command and a long tail.
- **Loaded rule** — this one is already in MEMORY.md verbatim and loads today.
"""

USER = """---
type: note
---

# User (Alan) — Memory & Context

## infra
- **Kept rule** tightened.

## prefs
- terse.
"""

MEMORY = "# Memory\n\n- **Loaded rule** — this one is already in MEMORY.md verbatim and loads today.\n"
SOUL = "# Soul\n\nBe direct.\n"


def _raw(**kw):
    base = {"id": "p1", "section": "infra",
            "entry": "**Cut rule** — never quote a stale count",
            "prompt": "q?", "criterion": "PASS if it re-measures."}
    base.update(kw)
    return base


def _loaded():
    return {"USER.md": USER, "MEMORY.md": MEMORY, "SOUL.md": SOUL}


def _resolve(**kw):
    return ab.resolve_probe(_raw(**kw), ab.load_ledger(AUDIT), ab.load_archive_bullets(ARCHIVE),
                            _loaded())


# -- clause 2: probes bind to a move row, verbatim in the archive, not loaded --


def test_ledger_parses_decisions_and_overlap_only_on_condense():
    rows = ab.load_ledger(AUDIT)
    assert [r.decision for r in rows] == ["move", "condense", "move"]
    assert rows[1].overlap == pytest.approx(0.45)
    assert rows[0].overlap is None and not rows[0].prefix.endswith("…")


def test_archive_bullets_start_at_the_copied_user_md():
    bullets = ab.load_archive_bullets(ARCHIVE)
    assert all(sec == "infra" for sec, _ in bullets)
    assert not any("not a USER.md bullet" in ln for _, ln in bullets)


def test_move_probe_resolves_to_its_verbatim_archive_line():
    p = _resolve()
    assert p.line == ("- **Cut rule** — never quote a stale count, re-measure it in the run "
                      "that uses it.")
    assert p.population == "move"


def test_probe_citing_a_condense_row_is_rejected():
    with pytest.raises(ab.ProbeRejected, match="condense row is not a cut line"):
        _resolve(entry="**Kept rule** — tightened")


def test_condense_population_is_refused_rather_than_guessed():
    with pytest.raises(ab.ProbeRejected, match="condense arms are not implemented"):
        _resolve(entry="**Kept rule** — tightened", population="condense")


def test_probe_whose_line_is_already_loaded_is_rejected():
    with pytest.raises(ab.ProbeRejected, match="already loaded .*MEMORY.md"):
        _resolve(entry="**Loaded rule** — this one")


def test_probe_with_no_ledger_row_is_rejected():
    with pytest.raises(ab.ProbeRejected, match="0 ledger rows"):
        _resolve(entry="**Nothing like this**")


def test_probe_whose_line_is_not_in_the_archive_is_rejected():
    archive = ARCHIVE.replace("- **Cut rule**", "- **Renamed rule**")
    with pytest.raises(ab.ProbeRejected, match="0 archive lines"):
        ab.resolve_probe(_raw(), ab.load_ledger(AUDIT), ab.load_archive_bullets(archive),
                         _loaded())


def test_shipped_probe_file_loads_against_the_live_ledger():
    """The committed probes resolve on the real audit + archive (skips off-box)."""
    if not ab.AUDIT_PATH.exists() or not ab.ARCHIVE_PATH.exists():
        pytest.skip("vault ledger not present on this machine")
    probes = ab.load_probes()
    assert len(probes) >= 20
    assert {p.population for p in probes} == {"move"}


# -- clause 1: two arms, byte-identical but for the restored block ------------


def _surface(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "USER.md").write_text(USER)
    (src / "MEMORY.md").write_text(MEMORY)
    (src / "SOUL.md").write_text(SOUL)
    return src


def _build(overlay_dir):
    from prompt_builder import build_system_prompt
    return build_system_prompt(include_skills_index=False, overlay_dir=overlay_dir)


def test_restore_line_appends_at_the_end_of_its_section():
    out = ab.restore_line(USER, "infra", "- NEW")
    assert "- **Kept rule** tightened.\n- NEW\n\n## prefs" in out


def test_restore_line_adds_a_missing_section():
    assert ab.restore_line(USER, "gone", "- NEW").endswith("## gone\n- NEW\n")


def test_arms_built_through_prompt_builder_differ_only_by_the_line(tmp_path):
    p = _resolve()
    canonical = ab.freeze_surface(_surface(tmp_path), tmp_path / "arms" / "canonical")
    restored = {p.id: ab.build_restored_arm(canonical, p, tmp_path / "arms" / p.id)}
    deltas = ab.verify_arms(canonical, restored, [p], build=_build)
    assert deltas[p.id] == len(p.line) + 1
    a, b = _build(canonical), _build(restored[p.id])
    assert b.replace(p.line + "\n", "", 1) == a


def test_arm_equality_fails_when_the_remainder_moves(tmp_path):
    """The MEMORY.md-growth confound: any other byte of difference is refused."""
    p = _resolve()
    canonical = ab.freeze_surface(_surface(tmp_path), tmp_path / "arms" / "canonical")
    arm = ab.build_restored_arm(canonical, p, tmp_path / "arms" / p.id)
    (arm / "MEMORY.md").write_text(MEMORY + "- grew overnight\n")
    with pytest.raises(ab.ArmMismatch):
        ab.verify_arms(canonical, {p.id: arm}, [p], build=_build)


def test_arm_equality_fails_when_the_line_is_missing():
    with pytest.raises(ab.ArmMismatch):
        ab.check_arm_equality("same", "same", "- the line")


# -- clause 3: both outputs, one fixed grader, no rate over incomplete pairs ----


def _trace(arm, text="answer", status="success"):
    return {"variant_id": arm, "status": status, "final_text": text, "tool_calls": [],
            "session_id": f"s-{arm}", "turns": 1, "duration_seconds": 1.0}


def test_grade_answer_records_the_fixed_grader_and_skips_failed_trials():
    p = _resolve()
    seen = []

    def call(prompt):
        seen.append(prompt)
        return '{"verdict": "pass", "reason": "ok"}'

    g = ab.grade_answer(p, _trace("canonical"), call=call)
    assert g["verdict"] == "PASS" and g["grader"] == ab.GRADER
    assert "canonical" not in seen[0]  # blind to the arm
    assert ab.grade_answer(p, _trace("x", status="timeout"), call=call) is None
    assert ab.grade_answer(p, _trace("x", text="  "), call=call) is None
    assert ab.grade_answer(p, _trace("x"), call=lambda _: "not json") is None


def test_run_stores_both_arms_outputs_and_one_verdict_each(tmp_path):
    p = _resolve()
    calls = []

    async def fake_bench(cfg, variants, tasks, model, **kw):
        calls.append([v for v, _ in variants])
        return [_trace(v, text=f"{v} says") for v, _ in variants]

    def fake_grade(probe, trace):
        v = "PASS" if trace["variant_id"] == "restored" else "FAIL"
        return {"verdict": v, "reason": "", "grader": dict(ab.GRADER)}

    recs = asyncio.run(ab.run([p], tmp_path, {p.id: tmp_path}, model="primary",
                              out_dir=tmp_path, run_bench=fake_bench, grade=fake_grade))
    assert calls == [list(ab.ARMS)]
    r = recs[0]
    assert set(r["outputs"]) == set(ab.ARMS) and all(r["outputs"].values())
    assert {a: r["grades"][a]["verdict"] for a in ab.ARMS} == {
        "canonical": "FAIL", "canonical_rep": "FAIL", "restored": "PASS"}
    assert r["pair"] == "lost"
    assert (tmp_path / "trials.jsonl").exists()


def test_run_retries_an_errored_trial_once(tmp_path):
    p = _resolve()
    n = {"restored": 0}

    async def flaky_bench(cfg, variants, tasks, model, **kw):
        out = []
        for v, _ in variants:
            if v == "restored":
                n["restored"] += 1
                out.append(_trace(v, status="error" if n["restored"] == 1 else "success"))
            else:
                out.append(_trace(v))
        return out

    ok = {"verdict": "PASS", "reason": "", "grader": dict(ab.GRADER)}
    recs = asyncio.run(ab.run([p], tmp_path, {p.id: tmp_path}, model="primary",
                              out_dir=tmp_path, run_bench=flaky_bench,
                              grade=lambda probe, t: ok))
    assert n["restored"] == 2 and recs[0]["outputs"]["restored"]


def test_run_retries_a_turn_cut_off_by_the_iteration_cap(tmp_path):
    """A preamble left by max_turns grades FAIL for budget, not knowledge."""
    p = _resolve()
    seen = {"canonical": 0}

    async def capped_bench(cfg, variants, tasks, model, **kw):
        out = []
        for v, _ in variants:
            t = _trace(v, text=f"{v} says")
            if v == "canonical":
                seen["canonical"] += 1
                if seen["canonical"] == 1:
                    t.update(stop_reason="max_turns", final_text="I'll check first.")
            out.append(t)
        return out

    ok = {"verdict": "PASS", "reason": "", "grader": dict(ab.GRADER)}
    recs = asyncio.run(ab.run([p], tmp_path, {p.id: tmp_path}, model="primary",
                              out_dir=tmp_path, run_bench=capped_bench,
                              grade=lambda probe, t: ok, probe_parallel=2))
    assert seen["canonical"] == 2
    assert recs[0]["outputs"]["canonical"] == "canonical says"


def _rec(pid, pop="move", c="PASS", rep="PASS", r="PASS", overlap=None):
    g = lambda v: {"verdict": v, "reason": "", "grader": dict(ab.GRADER)}  # noqa: E731
    return {"probe_id": pid, "population": pop, "overlap": overlap,
            "outputs": {a: "text" for a in ab.ARMS},
            "grades": {"canonical": g(c), "canonical_rep": g(rep), "restored": g(r)}}


def test_summarize_raises_on_a_missing_output():
    rec = _rec("p1")
    rec["outputs"]["restored"] = None
    with pytest.raises(ab.IncompletePairs, match="p1"):
        ab.summarize([_rec("p0"), rec])


def test_summarize_raises_on_a_missing_verdict():
    rec = _rec("p1")
    rec["grades"]["canonical"] = None
    with pytest.raises(ab.IncompletePairs):
        ab.summarize([rec])


# -- clause 4: populations apart, no pooled rate, an explicit overlap statement --


def test_move_and_condense_are_reported_apart_with_no_pooled_rate():
    recs = [_rec("m1", c="FAIL", r="PASS"), _rec("m2"), _rec("m3", rep="FAIL"),
            _rec("c1", pop="condense", c="FAIL", r="PASS", overlap=0.31),
            _rec("c2", pop="condense", overlap=0.78)]
    s = ab.summarize(recs)
    assert set(s) == {"move", "condense", "overlap_orders_divergence"}
    move = s["move"]
    assert move["effect"]["n"] == 3 and move["effect"]["divergent"] == 1
    assert move["effect"]["lost"] == 1 and move["effect"]["gained"] == 0
    assert move["noise_floor"]["divergent"] == 2  # m1 (canonical FAIL vs rep PASS) and m3
    assert s["condense"]["n"] == 2
    assert set(s["condense"]["by_overlap_decile"]) == {"0.3-0.4", "0.7-0.8"}
    assert s["overlap_orders_divergence"].startswith("yes")
    assert "pooled" not in str(s)


def test_overlap_statement_is_explicit_when_undeterminable_or_not_monotone():
    assert ab.summarize([_rec("m1")])["overlap_orders_divergence"].startswith("undetermined")
    recs = [_rec("c1", pop="condense", overlap=0.31),
            _rec("c2", pop="condense", c="FAIL", overlap=0.78)]
    assert ab.summarize(recs)["overlap_orders_divergence"].startswith("no")


def test_sign_test_is_exact_and_two_sided():
    assert ab._sign_test_p(0, 0) is None
    assert ab._sign_test_p(3, 3) == pytest.approx(0.25)
    assert ab._sign_test_p(0, 5) == pytest.approx(0.0625)
