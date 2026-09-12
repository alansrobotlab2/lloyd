"""#879 — bound and de-noise the #67 semantic resolution run.

Four contracts in `semantic-entity-resolution.py` that nothing held before:

- **what enters the pool**: `.md`-suffixed and date-shaped names are note
  filenames, not entities. 8.61 % and 3.43 % of the live candidate pool had one
  (`semantic-entity-candidates-2026-09-08.jsonl`, 566,170 pairs), and
  `out.sort` put them at the TOP, so they dominated every `--limit` slice and
  40 of the proposals the sweep was handed.
- **what the run may select**: without a score floor, `--limit 2000` was
  charging LLM budget against a 566,170-pair pool of which only 6,863 score
  ≥ 4.0. The floor makes the work bounded, and the run must say what it did.
- **whether the output survives**: the dated proposals file was opened `"w"`
  and `semantic-proposals-latest.jsonl` — the only path the sweep reads — was
  re-pointed at it every run, so a proposal the sweep had not reached simply
  disappeared with the next run.
- **whether the docs match the box**: the skill described a pool, a limit and a
  timeout that the live task does not use.

The seam that matters is between two processes: #67 (weekly, proposes) and the
sweep (every 15 min, disposes), which meet only through
`semantic-proposals-latest.jsonl` and the sweep's own ledger. Those tests below
therefore write through the real emitter and read back through the sweep's real
loader, not through a mock of either.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEMORY = ROOT / "scripts" / "memory"
sys.path.insert(0, str(ROOT))
VAULT_SKILL = Path.home() / "obsidian" / "skills" / "semantic-entity-resolution" / "SKILL.md"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, str(MEMORY / filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ser = _load("ser_under_test", "semantic-entity-resolution.py")
sweep = _load("sweep_under_test", "entity-resolution-sweep.py")

SCRIPT = MEMORY / "semantic-entity-resolution.py"

# Synthetic entity names chosen so each pair really does pass candidate
# generation's own signals (shared 5-char normalized prefix), which is what
# makes "the pair is absent" a statement about the filter and not about
# bucketing.
MD_PAIR = ("knowledge-library", "knowledge-library.md")
DATE_PAIR = ("2026-09-08-daily-note", "2026-09-08-daily-notes")
MD_ON_OTHER_SIDE = ("qwen3-tts-voice-cloning", "qwen3-tts-voice-cloning.md")
KEEP_PAIR = ("intel-pipeline-config", "intel-pipeline-config notes")

ENTITIES = list(dict.fromkeys([*MD_PAIR, *DATE_PAIR, *MD_ON_OTHER_SIDE, *KEEP_PAIR]))


def _candidates(entities=None, neighbors=None):
    return ser.generate_candidates(
        entities or ENTITIES, {}, neighbors or {}
    )


def _pairs(candidates):
    return {(c["a"], c["b"]) for c in candidates}


# ---------------------------------------------------------------------------
# clause 1 + 2: candidate generation drops artifact name shapes
# ---------------------------------------------------------------------------


def test_artifact_name_shape_recognises_both_classes():
    assert ser.is_artifact_name("knowledge-library.md")
    assert ser.is_artifact_name("2026-09-08-daily-note")
    assert ser.is_artifact_name("2026-09-08")


def test_artifact_name_shape_does_not_eat_real_entities():
    # A bare leading year is not a date, and a hyphenated 'md' is not an extension.
    assert not ser.is_artifact_name("2026 Roadmap")
    assert not ser.is_artifact_name("readme-md")
    assert not ser.is_artifact_name("Markdown Syntax")
    assert not ser.is_artifact_name("Qwen3-TTS")


def test_generation_yields_no_pair_that_is_the_other_plus_md():
    """clause 1 — one entity name equal to the other plus `.md`."""
    got = _pairs(_candidates())
    assert got, "test is vacuous unless generation produced candidates"
    for a, b in got:
        assert not (a + ".md" == b or b + ".md" == a), (a, b)
    assert MD_PAIR not in got
    assert MD_ON_OTHER_SIDE not in got


def test_generation_yields_no_date_shaped_pair():
    """clause 2 — no side matches ^\\d{4}-\\d{2}-\\d{2}."""
    pat = re.compile(r"^\d{4}-\d{2}-\d{2}")
    got = _candidates()
    assert got, "test is vacuous unless generation produced candidates"
    for c in got:
        assert not pat.match(c["a"]) and not pat.match(c["b"]), c
    assert DATE_PAIR not in _pairs(got)


def test_filter_does_not_drop_real_duplicates():
    """Generation keeps a genuine variant, and only about generation.

    This pair scores below the default floor, so surviving the FILTER does not
    mean it gets judged — selection is a separate rule, pinned by
    test_no_candidate_below_the_floor_is_selected and by the main() tests below.
    """
    assert KEEP_PAIR in _pairs(_candidates())
    selected, _above = ser.select_candidates(_candidates())
    assert KEEP_PAIR not in {(c["a"], c["b"]) for c in selected}


# ---------------------------------------------------------------------------
# clause 3: the written candidate file is clean under the cited check
# ---------------------------------------------------------------------------


def _run_cited_check(path: Path) -> tuple[int, int, int]:
    """The exact counting code the item's acceptance check uses."""
    pat = re.compile(r"^\d{4}-\d{2}-\d{2}")
    tot = md = dt = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        tot += 1
        if c["a"].endswith(".md") or c["b"].endswith(".md"):
            md += 1
        if pat.match(c["a"]) or pat.match(c["b"]):
            dt += 1
    return tot, md, dt


def test_freshly_written_candidate_file_is_clean(tmp_path):
    """clause 3 — 0 `.md`-suffixed and 0 date-shaped pairs, by the cited check."""
    candidates = _candidates()
    assert candidates
    f = tmp_path / "semantic-entity-candidates-2099-01-01.jsonl"
    with f.open("w") as fh:
        for c in candidates:
            fh.write(json.dumps(c) + "\n")
    tot, md, dt = _run_cited_check(f)
    assert tot > 0
    assert (md, dt) == (0, 0)


# ---------------------------------------------------------------------------
# clause 4: emitted proposals carry no artifact rows
# ---------------------------------------------------------------------------


def _proposal(canonical, variant, confidence=0.9, action="merge"):
    return {
        "canonical": canonical, "variant": variant, "action": action,
        "verdict": "same", "confidence": confidence, "reason": "test",
        "guard_reason": None, "cached": False,
        "proposed_at": "2026-09-12T00:00:00+00:00",
    }


def test_proposal_filter_drops_md_rows_from_both_sides():
    kept = ser.filter_artifact_proposals([
        _proposal("knowledge-library", "knowledge-library.md"),
        _proposal("knowledge-library.md", "knowledge-library"),
        _proposal("2026-09-08-daily-note", "2026-09-08-daily-notes"),
        _proposal("intel-pipeline", "intel-pipeline config", action="alias_only"),
    ])
    assert [p["canonical"] for p in kept] == ["intel-pipeline"]


def test_written_proposal_file_holds_no_md_row(tmp_path):
    """clause 4 — the file a run writes, not just the in-memory list."""
    run_log = tmp_path / "semantic-proposals-2099-01-01.jsonl"
    cumulative = tmp_path / "semantic-proposals-cumulative.jsonl"
    latest = tmp_path / "semantic-proposals-latest.jsonl"
    emitted = ser.filter_artifact_proposals([
        _proposal("knowledge-library", "knowledge-library.md"),
        _proposal("intel-pipeline", "intel-pipeline config"),
    ])
    ser.emit_proposals(emitted, run_log=run_log, cumulative=cumulative, latest=latest)
    rows = [json.loads(l) for l in run_log.read_text().splitlines() if l.strip()]
    assert rows
    for r in rows:
        assert not r["canonical"].endswith(".md") and not r["variant"].endswith(".md"), r


# ---------------------------------------------------------------------------
# clause 5 + 6: the score floor is a knob, and the default is where the
# --limit slice already happened to land
# ---------------------------------------------------------------------------


def _cand(a, b, score):
    return {"a": a, "b": b, "score": score, "jaccard": 1.0,
            "shares_stem": True, "shared_neighbors": 0}


def test_min_score_flag_exists_and_defaults_to_four():
    args = ser.build_arg_parser().parse_args([])
    assert args.min_score == 4.0
    assert ser.build_arg_parser().parse_args(["--min-score", "0"]).min_score == 0.0


def test_no_candidate_below_the_floor_is_selected():
    """clause 5 — default floor."""
    pool = [_cand("a1", "b1", 9.0), _cand("a2", "b2", 4.0),
            _cand("a3", "b3", 3.9), _cand("a4", "b4", 0.1)]
    selected, above_floor = ser.select_candidates(pool)
    assert [c["a"] for c in selected] == ["a1", "a2"]
    assert above_floor == 2


def test_min_score_zero_reopens_the_tail():
    """clause 6 — the floor is a knob, not a hard-coded cut."""
    pool = [_cand("a1", "b1", 9.0), _cand("a3", "b3", 3.9), _cand("a4", "b4", 0.1)]
    selected, above_floor = ser.select_candidates(pool, min_score=0.0)
    assert [c["a"] for c in selected] == ["a1", "a3", "a4"]
    assert above_floor == 3


def test_floor_is_applied_before_the_limit_slice():
    """`--limit` must cut the eligible head, never reach below the floor."""
    pool = [_cand("a1", "b1", 9.0), _cand("a2", "b2", 5.0), _cand("a3", "b3", 1.0)]
    selected, above_floor = ser.select_candidates(pool, min_score=4.0, limit=10)
    assert [c["a"] for c in selected] == ["a1", "a2"]
    assert above_floor == 2


def test_selection_drops_artifact_pairs_from_a_stale_pool():
    """`--from-candidates` skips generation, so selection is the second lock.

    Without it, re-judging a pool written before #879 would put the LLM budget
    straight back onto `.md` twins.
    """
    pool = [_cand("knowledge-library", "knowledge-library.md", 9.0),
            _cand("2026-09-08-note", "2026-09-08-notes", 8.0),
            _cand("intel-pipeline", "intel-pipeline config", 7.0)]
    selected, above_floor = ser.select_candidates(pool, min_score=0.0)
    assert [c["a"] for c in selected] == ["intel-pipeline"]
    assert above_floor == 1


def test_cli_advertises_min_score():
    """The seam: the autonomy task invokes this by argv, not by calling main()."""
    out = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-400:]
    assert "--min-score" in out.stdout


# ---------------------------------------------------------------------------
# clause 7: one line, three numbers
# ---------------------------------------------------------------------------


def test_summary_line_reports_newly_judged_cached_and_remaining():
    line = ser.run_summary(newly_judged=1961, from_cache=39,
                           above_floor_total=6863, selected=2000, min_score=4.0)
    assert "newly_judged=1961" in line
    assert "from_cache=39" in line
    assert "above_floor_remaining=4863" in line
    assert line.count("\n") == 0


def test_summary_line_reports_a_drained_head_as_zero():
    line = ser.run_summary(newly_judged=81, from_cache=0,
                           above_floor_total=81, selected=81, min_score=4.0)
    assert "above_floor_remaining=0" in line


# ---------------------------------------------------------------------------
# clause 8 + 9: durability across runs, read back through the sweep's loader
# ---------------------------------------------------------------------------


def _paths(tmp_path):
    return (tmp_path / "semantic-proposals-2099-01-01.jsonl",
            tmp_path / "semantic-proposals-cumulative.jsonl",
            tmp_path / "semantic-proposals-latest.jsonl")


def test_run_n_proposal_still_readable_after_run_n_plus_one(tmp_path):
    """clause 8 — crosses the process seam: emitted by #67's code, read by the
    sweep's real `load_semantic_proposals`, which only ever reads the -latest
    path."""
    run1, run2 = tmp_path / "p-1.jsonl", tmp_path / "p-2.jsonl"
    cumulative = tmp_path / "semantic-proposals-cumulative.jsonl"
    latest = tmp_path / "semantic-proposals-latest.jsonl"
    ser.emit_proposals([_proposal("intel-pipeline", "intel-pipeline config")],
                       run_log=run1, cumulative=cumulative, latest=latest)
    ser.emit_proposals([_proposal("qwen3-tts", "qwen 3 tts")],
                       run_log=run2, cumulative=cumulative, latest=latest)

    seen = sweep.load_semantic_proposals(tmp_path)
    pairs = {(r["canonical"], r["variant"]) for r in seen}
    assert ("intel-pipeline", "intel-pipeline config") in pairs, \
        "run N's proposal vanished from semantic-proposals-latest.jsonl"
    assert ("qwen3-tts", "qwen 3 tts") in pairs


def test_pair_appears_once_after_two_runs(tmp_path):
    """clause 9 — restating a proposal refreshes it, never duplicates it."""
    run1, run2 = tmp_path / "p-1.jsonl", tmp_path / "p-2.jsonl"
    cumulative = tmp_path / "semantic-proposals-cumulative.jsonl"
    latest = tmp_path / "semantic-proposals-latest.jsonl"
    ser.emit_proposals([_proposal("intel-pipeline", "intel-pipeline config", 0.8)],
                       run_log=run1, cumulative=cumulative, latest=latest)
    ser.emit_proposals([_proposal("intel-pipeline", "intel-pipeline config", 0.93)],
                       run_log=run2, cumulative=cumulative, latest=latest)
    seen = sweep.load_semantic_proposals(tmp_path)
    hits = [r for r in seen if (r["canonical"], r["variant"]) == ("intel-pipeline", "intel-pipeline config")]
    assert len(hits) == 1, hits
    assert float(hits[0]["confidence"]) == 0.93
    assert hits[0]["first_seen_at"] and hits[0]["last_seen_at"]


def test_cumulative_file_is_what_latest_points_at(tmp_path):
    _, cumulative, latest = _paths(tmp_path)
    ser.emit_proposals([_proposal("intel-pipeline", "intel-pipeline config")],
                       run_log=tmp_path / "p.jsonl", cumulative=cumulative, latest=latest)
    assert latest.is_symlink()
    assert Path(latest.readlink()).name == cumulative.name


# ---------------------------------------------------------------------------
# clause 10: the sweep names what it has never evaluated
# ---------------------------------------------------------------------------


def _write_latest(tmp_path, proposals):
    (tmp_path / "semantic-proposals-latest.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in proposals))


def test_sweep_plan_output_names_never_evaluated_proposals(tmp_path, capsys):
    """clause 10 — the count comes from a ledger the sweep itself persists."""
    _write_latest(tmp_path, [
        _proposal("intel-pipeline", "intel-pipeline config"),
        _proposal("qwen3-tts", "qwen 3 tts"),
    ])
    proposals, never = sweep.surface_semantic_proposals(tmp_path)
    out = capsys.readouterr().out
    assert len(proposals) == 2
    assert never == 2
    assert "never evaluated" in out
    assert "2 never evaluated" in out
    ledger = tmp_path / "semantic-proposals-seen.jsonl"
    assert ledger.exists(), "the count must come from state the sweep persists"


def test_sweep_reports_zero_never_on_the_next_plan(tmp_path, capsys):
    _write_latest(tmp_path, [_proposal("intel-pipeline", "intel-pipeline config")])
    sweep.surface_semantic_proposals(tmp_path)
    capsys.readouterr()
    proposals, never = sweep.surface_semantic_proposals(tmp_path)
    out = capsys.readouterr().out
    assert proposals and never == 0
    assert "0 never evaluated" in out


def test_sweep_names_only_the_newly_arrived_proposals(tmp_path, capsys):
    _write_latest(tmp_path, [_proposal("intel-pipeline", "intel-pipeline config")])
    sweep.surface_semantic_proposals(tmp_path)
    capsys.readouterr()
    _write_latest(tmp_path, [
        _proposal("intel-pipeline", "intel-pipeline config"),
        _proposal("qwen3-tts", "qwen 3 tts"),
    ])
    _, never = sweep.surface_semantic_proposals(tmp_path)
    capsys.readouterr()
    assert never == 1


def test_two_emitted_runs_reach_the_sweep_plan_line(tmp_path, capsys):
    """End to end across the seam: #67's emitter twice, then the sweep's plan line.

    This is the whole of #744 in one assertion — two runs happen, the sweep still
    sees both proposals, and it can say which of them it has not evaluated.
    """
    cumulative = tmp_path / "semantic-proposals-cumulative.jsonl"
    latest = tmp_path / "semantic-proposals-latest.jsonl"
    ser.emit_proposals([_proposal("intel-pipeline", "intel-pipeline config")],
                       run_log=tmp_path / "p-1.jsonl", cumulative=cumulative, latest=latest)
    proposals, never = sweep.surface_semantic_proposals(tmp_path)
    assert [p["canonical"] for p in proposals] == ["intel-pipeline"]
    assert never == 1
    capsys.readouterr()

    ser.emit_proposals([_proposal("intel-pipeline", "intel-pipeline config"),
                        _proposal("qwen3-tts", "qwen 3 tts")],
                       run_log=tmp_path / "p-2.jsonl", cumulative=cumulative, latest=latest)
    proposals, never = sweep.surface_semantic_proposals(tmp_path)
    out = capsys.readouterr().out
    assert len(proposals) == 2, "run N's proposal must survive run N+1"
    assert never == 1, "only the newly arrived pair is unevaluated"
    assert "2 pairs awaiting review, 1 never evaluated" in out


def test_sweep_tolerates_a_missing_ledger_and_missing_proposals(tmp_path, capsys):
    proposals, never = sweep.surface_semantic_proposals(tmp_path)
    capsys.readouterr()
    assert (proposals, never) == ([], 0)


# ---------------------------------------------------------------------------
# the whole run, driven through main()
#
# Everything above tests functions; the acceptance clauses are about FILES a run
# writes and the LINE a run prints, and the only code that produces those is
# main(). These three drive main() with every production path and every external
# call redirected into tmp_path — argv in, candidate file, proposals and the
# summary line out.
# ---------------------------------------------------------------------------

# Synthetic pool, sized so each pair's fate is decided by the two rules under
# test rather than by bucketing: all three groups share a 5-char normalized
# prefix, and the neighbor sets give the first group exactly the 3 shared
# neighbors that lift it over the floor (0.75 jaccard + 1.0 stem + 3.0 = 4.75).
# score = jaccard*3 + (1.0 if shares_stem) + min(shared_neighbors,10)*0.3, and
# `pipeline`/`agent` are name stopwords, so the first pair tokenizes identically
# (jaccard 1.0) and shares 4 neighbors → 3.0 + 1.0 + 1.2 = 5.2. The qwen pair
# keeps its 4th token ("cloning"), so jaccard 0.75 → 3.25: a real pair the floor
# leaves out until --min-score says otherwise.
RUN_ENTITIES = [
    "intel-pipeline-config",          # real entity
    "intel-agent-pipeline-config",    # its real variant — clears the floor (5.2)
    "intel-pipeline-config.md",       # the twin — also 5.2, must never be judged
    "qwen3-tts-voice",                # real entity
    "qwen3-tts-voice cloning",        # real variant, 3.25: under the floor
]

RUN_NEIGHBORS = {
    "intel-pipeline-config": {"n1", "n2", "n3", "n4", "n5"},
    "intel-agent-pipeline-config": {"n1", "n2", "n3", "n4", "n6"},
    "intel-pipeline-config.md": {"n1", "n2", "n3", "n4", "n7"},
}

INTEL_PAIR = {"intel-pipeline-config", "intel-agent-pipeline-config"}
QWEN_PAIR = {"qwen3-tts-voice", "qwen3-tts-voice cloning"}


def _drive_main(tmp_path, monkeypatch, argv, confidence=0.93):
    """Run the real main() against tmp_path. Returns the pairs it judged."""
    monkeypatch.setattr(ser, "PIPELINE_ROOT", tmp_path)
    monkeypatch.setattr(ser, "CANDIDATE_LOG", tmp_path / "semantic-entity-candidates-2099-01-01.jsonl")
    monkeypatch.setattr(ser, "JUDGMENT_LOG", tmp_path / "judgments.jsonl")
    monkeypatch.setattr(ser, "PROPOSAL_LOG", tmp_path / "semantic-proposals-2099-01-01.jsonl")
    monkeypatch.setattr(ser, "PROPOSAL_CUMULATIVE", tmp_path / "semantic-proposals-cumulative.jsonl")
    monkeypatch.setattr(ser, "PROPOSAL_LATEST", tmp_path / "semantic-proposals-latest.jsonl")
    # Store, facts and verdicts are the outside world; the run under test must
    # not read or write any of them.
    monkeypatch.setattr(ser, "list_entities", lambda: list(RUN_ENTITIES))
    monkeypatch.setattr(ser, "load_aliases", lambda: {})
    monkeypatch.setattr(ser, "load_graph", lambda: {"edges": []})
    monkeypatch.setattr(ser, "build_neighbors", lambda graph: dict(RUN_NEIGHBORS))
    # `already_aliased` is nested inside generate_candidates and reaches for the
    # KG store through this global; with no store it falls back to name equality,
    # which keeps the run off the live database entirely.
    monkeypatch.setattr(ser, "get_store", lambda: None, raising=False)
    monkeypatch.setattr(ser, "count_facts", lambda entity: 0)
    monkeypatch.setattr(ser, "load_verdict_cache", lambda *a, **k: {})
    monkeypatch.setattr(ser, "append_verdict", lambda rec, *a, **k: None)
    monkeypatch.setattr(ser, "_definition", lambda entity: "a definition")
    judged = []

    def fake_judge(pair, endpoint, model, timeout):
        judged.append({pair["a"], pair["b"]})
        return {"verdict": "same", "confidence": confidence, "reason": "test"}

    monkeypatch.setattr(ser, "judge_pair", fake_judge)
    monkeypatch.setattr(sys, "argv", ["semantic-entity-resolution.py", *argv])
    assert ser.main() == 0
    return judged


def test_run_writes_a_clean_pool_judges_only_above_the_floor_and_reports_it(
        tmp_path, monkeypatch, capsys):
    """clauses 3, 4, 5 and 7, through the code a run actually executes."""
    judged = _drive_main(tmp_path, monkeypatch, [])
    out = capsys.readouterr().out

    # clause 3 — the file main wrote, measured by the cited check.
    pool = tmp_path / "semantic-entity-candidates-2099-01-01.jsonl"
    tot, md, dt = _run_cited_check(pool)
    assert tot == 2, [json.loads(l) for l in pool.read_text().splitlines()]
    assert (md, dt) == (0, 0)

    # clause 5 — the 4.75 artifact twin is absent AND the 1.75 real pair is
    # unjudged: the floor, not the filter, is what left the pair out.
    assert judged == [INTEL_PAIR]
    assert QWEN_PAIR not in judged

    # clause 7 — one line, three numbers, printed by main.
    summary = [l for l in out.splitlines() if l.startswith("[summary]")]
    assert len(summary) == 1, out
    assert "newly_judged=1" in summary[0]
    assert "from_cache=0" in summary[0]
    assert "above_floor_remaining=0" in summary[0]

    # clause 4 — what the sweep will be handed.
    seen = sweep.load_semantic_proposals(tmp_path)
    assert [p["canonical"] for p in seen] == ["intel-pipeline-config"]
    for p in seen:
        assert not p["canonical"].endswith(".md") and not p["variant"].endswith(".md")


def test_min_score_zero_from_the_command_line_judges_the_low_scoring_pair(
        tmp_path, monkeypatch, capsys):
    """clause 6 at the seam that matters: argv, not a call into a helper."""
    judged = _drive_main(tmp_path, monkeypatch, ["--min-score", "0"])
    out = capsys.readouterr().out
    assert QWEN_PAIR in judged, "the floor is not a knob if 0 does not open the tail"
    assert len(judged) == 2, judged
    assert "newly_judged=2" in out


def test_limit_cuts_the_eligible_head_and_reports_what_is_left(
        tmp_path, monkeypatch, capsys):
    """`--limit` after the floor, so the remaining number means backlog."""
    judged = _drive_main(tmp_path, monkeypatch, ["--min-score", "0", "--limit", "1"])
    out = capsys.readouterr().out
    assert judged == [INTEL_PAIR], "the slice must take the highest-scoring eligible pair"
    assert "above_floor_remaining=1" in out, "the 1.75 pair is eligible and left over"


# ---------------------------------------------------------------------------
# clause 11 + 12: the docs match the box
# ---------------------------------------------------------------------------


def test_skill_states_the_measured_pool_limit_and_timeout():
    """clause 11.

    No skip marker on the two doc tests: they are the only thing standing between
    a reader and clauses 11/12, so a missing vault has to be a failure. A skip
    would let the contract pass by not being checked.
    """
    assert VAULT_SKILL.exists(), (
        f"{VAULT_SKILL} is absent, and clause 11 of #879 is a claim about that "
        "file — refusing to treat an unchecked claim as a pass")
    text = VAULT_SKILL.read_text(encoding="utf-8")
    assert "566,170" in text
    assert "--limit 2000" in text
    assert "timeout_seconds: 3000" in text
    for stale in ("65K", "65k", "--limit 4000", "3600s"):
        assert stale not in text, f"skill still carries {stale!r}"


def test_skill_says_propose_only_and_drops_the_shrinking_pool_claim():
    """clause 12 — the skill. See the note on the test above about the skip."""
    assert VAULT_SKILL.exists(), "vault skill absent; clause 12 cannot be checked"
    text = VAULT_SKILL.read_text(encoding="utf-8")
    assert "propose-only" in text
    assert "cannot shrink" in text
    for stale in ("shrinks", "drains across weeks"):
        assert stale not in text, f"skill still claims the pool {stale!r}"


def test_verdict_cache_comment_no_longer_claims_a_shrinking_pool():
    """clause 12 — the script's own comment above VERDICT_CACHE."""
    src = SCRIPT.read_text(encoding="utf-8")
    m = re.search(r"(?:^#.*\n)+\s*VERDICT_CACHE", src, re.MULTILINE)
    assert m, "the comment block above VERDICT_CACHE went missing"
    block = m.group(0)
    assert "65k" not in block.lower()
    assert "shrink" not in block.lower()
