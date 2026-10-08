"""#1487 — the ADD / UPDATE / NOOP write gate in front of `fact_add`.

A paraphrase of a fact the entity already holds in that category walks through
#499's verbatim guard. `agent_mcp/fact_write_gate.py` asks djev one question
about the lexically closest active fact and, under `mode`, either writes
nothing (NOOP), supersedes that one fact (UPDATE: new appended, old stamped
`expired_at`), or appends as today (ADD). Every djev failure is ADD.

djev is stubbed at `app.djev.ask_sync` with a real server-shaped payload, so
`fact_write_gate.ask` and the `Answers` normalisation run for real.

Run: .venvs/lloyd/bin/python -m pytest tests/test_fact_write_gate.py
"""
import datetime
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_mcp import _shared, fact_write_gate as gate, facts, retrieval  # noqa: E402
from app import djev, kg_store, paths  # noqa: E402
from app.data_root import production_data_root  # noqa: E402
from tests._live_data import require_live_data  # noqa: E402
from tests.board_presence import vault_root  # noqa: E402

ENTITY, CAT = "Zedlink", "state"
OLD = "stream_chat races SSE line reads against cancel_event to allow Stop during prefill"
PARAPHRASE = "stream_chat races SSE line reads against a cancel event to allow stopping during prefill"
RICHER = ("stream_chat races SSE line reads against cancel_event to allow Stop during "
          "prefill in client.py")
OTHER = "Zedlink runs on the tailnet behind the office router"


@pytest.fixture
def tree(tmp_path, monkeypatch):
    root = tmp_path / "facts"
    root.mkdir()
    kg_store.configure(tmp_path / "kg.sqlite")
    monkeypatch.setattr(_shared, "FACTS_ROOT", root)
    monkeypatch.setattr(_shared, "ALIASES_PATH", root / "entity-aliases.json")
    monkeypatch.setattr(_shared, "_entity_dirs_cache", None)
    monkeypatch.setattr(facts, "FACTS_ROOT", root)
    monkeypatch.setattr(retrieval, "FACTS_ROOT", root)
    monkeypatch.setattr(paths, "FACT_WRITE_GATE_LOG", tmp_path / "fact-write-gate.jsonl")
    yield root
    kg_store.reset()


class Djev:
    """Stands in for the djev server: answers one `choice` per question with
    the probabilities it is given, and records what it was asked."""

    def __init__(self, probs=None, fail=False, mass=0.95):
        self.probs, self.fail, self.mass, self.calls = probs, fail, mass, []

    def __call__(self, state, questions, **kw):
        self.calls.append({"state": state, "questions": questions, **kw})
        if self.fail:
            return None
        payload = {
            "answers": {q: {"type": "choice", "choice": max(self.probs, key=self.probs.get),
                            "confidence": max(self.probs.values()),
                            "probabilities": dict(self.probs)} for q in questions},
            "diagnostics": {"questions": {q: {"label_mass": self.mass, "argmax_is_label": True}
                                          for q in questions}},
        }
        return djev._build(payload, 12.0, None, kw.get("seam", ""))


def _arm(monkeypatch, mode, fake):
    monkeypatch.setenv(gate.MODE_ENV, mode)
    monkeypatch.setattr(djev, "ask_sync", fake)
    return fake


def _add(fact, **extra):
    return facts._fact_add({"entity": ENTITY, "category": CAT, "fact": fact, **extra})


def _file(root):
    return root / ENTITY / f"{ENTITY}-{CAT}.md"


def _file_facts(root):
    fm = yaml.safe_load(_file(root).read_text(encoding="utf-8").split("---")[1])
    return fm["facts"]


def _rows():
    return kg_store.store().facts_idx.for_entity(ENTITY, include_expired=True)


RESTATED = {"different": 0.02, "restated": 0.95, "superseded": 0.03}
SUPERSEDED = {"different": 0.10, "restated": 0.20, "superseded": 0.70}
DIFFERENT = {"different": 0.97, "restated": 0.02, "superseded": 0.01}


# ── clause 1: one question, and the result names the verdict ────────────────

def test_one_question_about_the_closest_active_fact_and_the_verdict_is_named(tree, monkeypatch):
    assert _add(OLD)["verdict"] == "add"           # nothing to compare against: no call
    assert _add(OTHER)["verdict"] == "add"
    fake = _arm(monkeypatch, "on", Djev(DIFFERENT))
    r = _add(PARAPHRASE)
    assert r["success"] and r["verdict"] == "add", r
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert list(call["questions"]) == ["c0"]
    # The candidate is the lexically closest active fact in this category.
    assert OLD in call["state"] and OTHER not in call["state"]
    assert PARAPHRASE in call["state"]
    assert list(call["questions"]["c0"]["criteria"]) == ["different", "restated", "superseded"]

    _arm(monkeypatch, "on", Djev(RESTATED))
    assert _add(PARAPHRASE + " again")["verdict"] == "noop"


def test_a_fact_in_another_category_is_never_a_candidate(tree, monkeypatch):
    facts._fact_add({"entity": ENTITY, "category": "event", "fact": OLD})
    fake = _arm(monkeypatch, "on", Djev(RESTATED))
    r = _add(PARAPHRASE)
    assert r["verdict"] == "add" and fake.calls == []


# ── clause 2: noop writes nothing and still succeeds ────────────────────────

def test_noop_writes_nothing_and_returns_success(tree, monkeypatch):
    _add(OLD)
    before_bytes = _file(tree).read_bytes()
    before_rows = len(_rows())
    _arm(monkeypatch, "on", Djev(RESTATED))
    r = _add(PARAPHRASE)
    assert r["success"] is True and r["skipped"] is True and r["verdict"] == "noop", r
    assert r["restates"]["fact"] == OLD
    assert _file(tree).read_bytes() == before_bytes
    assert len(_rows()) == before_rows
    logged = [json.loads(l) for l in paths.FACT_WRITE_GATE_LOG.read_text().splitlines()]
    assert logged[-1]["applied"] == "noop" and logged[-1]["fact"] == PARAPHRASE


# ── clause 3: update supersedes exactly the one named fact ──────────────────

def test_update_appends_and_expires_exactly_the_named_fact(tree, monkeypatch):
    _add(OLD)
    _add(OTHER)
    _add("stream_chat omits tools from the request when the tools list is empty")
    _arm(monkeypatch, "on", Djev(SUPERSEDED))
    r = _add(RICHER)
    assert r["success"] and r["verdict"] == "update" and not r["skipped"], r
    assert r["superseded"]["fact"] == OLD
    by_text = {f["fact"]: f for f in _file_facts(tree)}
    assert by_text[OLD]["expired_at"], "the superseded fact is expired, not deleted"
    assert by_text[RICHER]["expired_at"] is None
    for text, f in by_text.items():
        if text != OLD:
            assert f["expired_at"] is None and f.get("invalid_at") is None, text
    rows = {row["fact"]: row for row in _rows()}
    assert set(rows) == set(by_text)                 # nothing removed from the index
    assert rows[OLD]["expired_at"] and not rows[OTHER]["expired_at"]


def test_update_needs_the_new_fact_to_contain_the_old_one(tree, monkeypatch):
    """djev's superseded vote alone never expires a fact: a paraphrase that
    drops a word of the old one (here its number) is ADD, both kept."""
    _add("Claude Code version v2.1.143 was released on 2026-05-16")
    _arm(monkeypatch, "on", Djev(SUPERSEDED))
    r = _add("Claude Code version v2.1.143 was released on 2026-07-06 to everyone")
    assert r["verdict"] == "add", r
    assert all(f["expired_at"] is None for f in _file_facts(tree))


def test_noop_mode_never_expires_and_shadow_mode_never_skips(tree, monkeypatch):
    _add(OLD)
    _arm(monkeypatch, "noop", Djev(SUPERSEDED))
    assert _add(RICHER)["verdict"] == "add"
    assert all(f["expired_at"] is None for f in _file_facts(tree))
    _arm(monkeypatch, "shadow", Djev(RESTATED))
    assert _add(PARAPHRASE)["verdict"] == "add"
    assert PARAPHRASE in [f["fact"] for f in _file_facts(tree)]
    logged = [json.loads(l) for l in paths.FACT_WRITE_GATE_LOG.read_text().splitlines()]
    assert [(r["mode"], r["verdict"], r["applied"]) for r in logged] == [
        ("noop", "update", "add"), ("shadow", "noop", "add")]


# ── the switch ──────────────────────────────────────────────────────────────

def test_off_is_the_default_and_asks_nothing(tree, monkeypatch):
    # The CODE default: no env, no config key. The live config ships `noop`,
    # which is the next test's business, so hide it here.
    import app.config
    monkeypatch.delenv(gate.MODE_ENV, raising=False)
    monkeypatch.setattr(app.config, "CONFIG", {})
    fake = Djev(RESTATED)
    monkeypatch.setattr(djev, "ask_sync", fake)
    _add(OLD)
    assert _add(PARAPHRASE)["verdict"] == "add"
    assert fake.calls == []


def test_config_ships_noop_and_an_unknown_mode_reads_off(monkeypatch):
    # `noop` is the measured-safe mode (76/77 NOOPs right on 400 held-out
    # writes); `on` also expires facts and had 3 samples — not shipped (#1487).
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert cfg["knowledge_graph"]["write_gate"]["mode"] == "noop"
    monkeypatch.setenv(gate.MODE_ENV, "yes-please")
    assert gate.mode() == "off"


def test_a_caller_can_opt_out_but_not_in(tree, monkeypatch):
    _add(OLD)
    fake = _arm(monkeypatch, "on", Djev(RESTATED))
    assert _add(PARAPHRASE, write_gate="off")["verdict"] == "add"
    assert fake.calls == []
    monkeypatch.setenv(gate.MODE_ENV, "off")
    assert _add(PARAPHRASE + " again", write_gate="on")["verdict"] == "add"
    assert fake.calls == []


# ── the pure decision ───────────────────────────────────────────────────────

def _ans(probs, mass=0.9):
    return {"c0": {"probabilities": probs, "label_mass": mass}}


def test_verdict_from_thresholds_and_trust_floor():
    cands = [{"fact": OLD, "id": "stat-001"}]
    assert gate.verdict_from(_ans(RESTATED), cands, PARAPHRASE)[0] == "noop"
    assert gate.verdict_from(_ans(SUPERSEDED), cands, RICHER)[0] == "update"
    assert gate.verdict_from(_ans(SUPERSEDED), cands, PARAPHRASE)[0] == "add"   # not contained
    assert gate.verdict_from(_ans(DIFFERENT), cands, RICHER)[0] == "add"
    # A low-mass answer is not trusted, whatever it says.
    assert gate.verdict_from(_ans(RESTATED, mass=0.1), cands, PARAPHRASE)[0] == "add"
    assert gate.verdict_from(_ans(RESTATED), cands, PARAPHRASE,
                             noop_threshold=None, update_threshold=None)[1] == "uncalibrated"


def test_contains_counts_numbers_and_negation():
    assert gate.contains("X ships in v2 and v3", "X ships in v2")
    assert not gate.contains("X ships in v2", "X ships in v2")            # says nothing more
    assert not gate.contains("released 2026-07-06 widely", "released 2026-05-16")
    # A negation is a word like any other: dropping it is dropping information.
    assert not gate.contains("Lloyd does use SDK hooks for control", "Lloyd does not use SDK hooks")


# ── the nightly extractor, which bypasses `_fact_add`, carries the same gate ─

def _load_extractor():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fact_extractor_gate", ROOT / "scripts/memory/next-gen-memory/fact_extractor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_extractor_holds_back_a_restatement_and_supersedes_in_place(tmp_path, monkeypatch):
    fx = _load_extractor()
    root = tmp_path / "facts"
    root.mkdir()
    kg_store.configure(tmp_path / "kg.sqlite")
    try:
        monkeypatch.setattr(fx, "FACTS_DIR", root)
        monkeypatch.setattr(paths, "FACT_WRITE_GATE_LOG", tmp_path / "gate.jsonl")
        e = fx.FactExtractor()
        e.facts_dir = root
        f = lambda t: {"fact": t, "confidence": 0.9, "category": CAT}  # noqa: E731
        e.write_fact_file(ENTITY, CAT, {"facts": [f(OLD), f(OTHER)]})
        path = root / ENTITY / f"{ENTITY}-{CAT}.md"

        _arm(monkeypatch, "on", Djev(RESTATED))
        e.write_fact_file(ENTITY, CAT, {"facts": [f(PARAPHRASE)]})
        texts = [x["fact"] for x in yaml.safe_load(path.read_text().split("---")[1])["facts"]]
        assert PARAPHRASE not in texts and texts == [OLD, OTHER]

        _arm(monkeypatch, "on", Djev(SUPERSEDED))
        e.write_fact_file(ENTITY, CAT, {"facts": [f(RICHER)]})
        by = {x["fact"]: x for x in yaml.safe_load(path.read_text().split("---")[1])["facts"]}
        assert by[OLD]["expired_at"] and by[RICHER]["expired_at"] is None
        assert by[OTHER]["expired_at"] is None

        monkeypatch.setenv(gate.MODE_ENV, "off")
        e.write_fact_file(ENTITY, CAT, {"facts": [f(PARAPHRASE)]})
        assert PARAPHRASE in [x["fact"] for x in yaml.safe_load(path.read_text().split("---")[1])["facts"]]
    finally:
        kg_store.reset()


# ── #2167: the shipped prose reserves UPDATE; the sample it cites does not ───
#
# Two sentences used to leave the question open for a count to answer.
# `config.yaml`'s write_gate comment said `on` "stays off until `shadow`-style
# data shows supersedes are safe at n > 3", and this module's docstring asked
# whether a djev label may expire a fact at all as a question still on the
# table. Both invited a round to arm a fact-expiring capability on a log count,
# so both are replaced by a reservation with an owner (Alan) and a prerequisite
# (a measured LloydMemEval `knowledge_update` gain). The numbers the new prose
# quotes are checked against the log they cite and against the copy of that log
# committed to the vault at backlog/data/fact-write-gate.jsonl, so a pinned
# figure that outgrows its evidence is a red node, not a stale sentence.

#: config.yaml's replaced comment lines, verbatim, and the sentence they carry.
#: The fold below must FIND this sentence in the held bytes: an assertion that
#: the shipped file lacks it is worth nothing until the scan has hit it once.
PRE_FIX_CONFIG_BLOCK = (
    "  # writes) and never expires a fact. `on` stays off until `shadow`-style "
    "data\n  # shows supersedes are safe at n > 3. Cost: ~340 ms p50 djev per "
    "asked write,")
PRE_FIX_CONFIG_SENTENCE = ("`on` stays off until `shadow`-style data shows supersedes "
                           "are safe at n > 3")
#: The same for the docstring: the pre-fix wording, and the question it posed.
PRE_FIX_DOCSTRING_BLOCK = (
    "* ``noop`` — apply NOOP (nothing written), record UPDATE as ADD. The scope\n"
    "  call the item left to a person is whether a djev label may expire a fact at\n"
    "  all (#499 recorded `fact_entity_recall` 0.35 → 0.30 from retiring the wrong\n"
    "  copy), so NOOP — which destroys nothing — is its own step.")
PRE_FIX_DOCSTRING_SENTENCE = ("The scope call the item left to a person is whether a djev "
                              "label may expire a fact at all")

#: The log the prose cites, spelled as the clause spells it; where that log
#: lives on a running box; and the copy of its would-be-supersede rows
#: committed to the vault, which is what the quoted figure is re-derived from
#: (#2167 clause 6).
CITED_LOG_PATH = "_pipeline/vault-derived/fact-write-gate.jsonl"
LIVE_LOG = production_data_root() / "_pipeline" / "vault-derived" / "fact-write-gate.jsonl"
WITNESS = Path("backlog") / "data" / "fact-write-gate.jsonl"
#: What counts as a verdict's outcome being recorded: absent here, which is
#: precisely why no logged row can say whether expiring its fact was safe.
GROUND_TRUTH_KEYS = {"ground_truth", "label", "labels", "outcome", "expected",
                     "correct", "verdict_correct", "human_label", "reviewed"}


def _fold(lines) -> str:
    """Comment markers off, one space between the wrapped lines, one string.

    Both shipped sites wrap mid-sentence, so a line-by-line scan sees
    ``…stays off until `shadow`-style data`` on one line and ``shows supersedes
    are safe at n > 3.`` on the next and matches neither. Folding is what makes
    the absence assertions below assertions about sentences.
    """
    out = []
    for line in lines:
        text = line.strip()
        out.append(text[1:].strip() if text.startswith("#") else text)
    return " ".join(part for part in out if part)


def _write_gate_comment() -> str:
    """The comment block sitting directly above the `write_gate:` key, folded."""
    lines = (ROOT / "config.yaml").read_text(encoding="utf-8").splitlines()
    at = next(i for i, line in enumerate(lines) if line == "  write_gate:")
    start = at
    while start and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    assert start < at, "no comment block above `write_gate:` to grade"
    return _fold(lines[start:at])


def _config():
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def _pinned_citation(block: str) -> tuple[int, datetime.date]:
    """The (count, as-of date) the comment block names for its cited log."""
    found = re.search(r"(\d[\d,]*)\s+`\"verdict\": \"update\"` rows in `"
                      + re.escape(CITED_LOG_PATH)
                      + r"` as of (\d{4}-\d{2}-\d{2})", block)
    assert found, f"the block cites no pinned count, path and as-of date:\n{block}"
    return int(found.group(1).replace(",", "")), datetime.date.fromisoformat(found.group(2))


def _witness_rows() -> list[dict]:
    """The committed witness rows, skipping by name if this box has no vault."""
    path = vault_root() / WITNESS
    require_live_data(path, "the committed write-gate witness", kind="file")
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ── clauses 1 and 2: the count is gone, the reservation is stated ───────────

def test_no_line_of_config_leaves_update_arming_to_a_count():
    cfg = (ROOT / "config.yaml").read_text(encoding="utf-8")
    assert [ln for ln in cfg.splitlines() if "n > 3" in ln] == []
    assert PRE_FIX_CONFIG_SENTENCE not in _write_gate_comment()
    # The control, so the two lines above cannot be a scan that matched nothing:
    # the same fold over the bytes that used to be shipped finds the sentence.
    pre_fix = _fold(PRE_FIX_CONFIG_BLOCK.splitlines())
    assert PRE_FIX_CONFIG_SENTENCE in pre_fix
    assert "n > 3" in pre_fix


def test_the_write_gate_comment_reserves_on_for_alan_and_for_a_measured_gain():
    block = _write_gate_comment()
    for words in ("UPDATE is never armed under #1487",
                  "`on` is RESERVED",
                  "explicit decision from Alan",
                  "LloydMemEval `knowledge_update` gain"):
        assert words in block, words
    # And who is actually standing between `mode: "noop"` and an expiring
    # write: the comment, because nothing else does (the next node checks).
    assert "no code guard" in block


# ── clause 3: the quoted count cannot rot ───────────────────────────────────

def test_the_quoted_supersede_count_is_at_most_what_the_live_log_holds():
    pinned, as_of = _pinned_citation(_write_gate_comment())
    require_live_data(LIVE_LOG, "the fact write gate's decision log", kind="file")
    logged = sum(1 for line in LIVE_LOG.read_text(encoding="utf-8").splitlines()
                 if '"verdict": "update"' in line)
    assert pinned <= logged, (
        f"config.yaml quotes {pinned} would-be supersedes but "
        f"{LIVE_LOG} holds {logged}: the prose number has outgrown its evidence")
    assert as_of <= datetime.date.today(), as_of


def test_the_quoted_count_is_re_derivable_from_the_committed_witness():
    pinned, as_of = _pinned_citation(_write_gate_comment())
    rows = _witness_rows()
    assert len(rows) == pinned, (
        f"`wc -l < {WITNESS}` is {len(rows)}, config.yaml quotes {pinned}")
    assert all(r["verdict"] == "update" for r in rows)
    assert all(r["ts"][:10] <= as_of.isoformat() for r in rows)
    # The three claims the new sentence makes about that sample, checked in it:
    # every row is a `noop` row, `noop` applied each one as ADD, and no row
    # records an outcome — which is the whole reason the count is not
    # authorizing. `applied` is the field an expiry would have landed in.
    assert all(r["mode"] == "noop" for r in rows)
    assert all(r["applied"] == "add" for r in rows)
    keys = set().union(*(set(r) for r in rows))
    assert not (keys & GROUND_TRUTH_KEYS), sorted(keys & GROUND_TRUTH_KEYS)


def test_the_witness_guard_skips_by_name_rather_than_failing_on_a_missing_vault(tmp_path,
                                                                                 monkeypatch):
    # `tests/_live_data.py`: an absent live root is a property of the machine,
    # so the skip names it; a present-but-wrong thing still fails. Proved by
    # moving the vault elsewhere, not by asserting an absence.
    monkeypatch.setenv("LLOYD_OBSIDIAN_VAULT", str(tmp_path))
    with pytest.raises(pytest.skip.Exception, match="committed write-gate witness"):
        _witness_rows()


# ── clause 4: the docstring reserves it too, instead of asking ──────────────

def test_the_module_docstring_records_the_scope_call_as_reserved_not_open():
    doc = _fold((gate.__doc__ or "").splitlines())
    assert PRE_FIX_DOCSTRING_SENTENCE not in doc
    assert PRE_FIX_DOCSTRING_SENTENCE in _fold(PRE_FIX_DOCSTRING_BLOCK.splitlines())
    for words in ("RESERVED, not open",
                  "explicit decision from Alan",
                  "LloydMemEval `knowledge_update` gain",
                  "no ground-truth field"):
        assert words in doc, words


# ── clause 5: prose only — no mode, no threshold, no branch moved ───────────

def test_the_change_moved_no_mode_no_threshold_and_no_branch():
    assert _config()["knowledge_graph"]["write_gate"]["mode"] == "noop"
    assert gate.MODES == ("off", "shadow", "noop", "on")
    assert (gate.SHORTLIST_FLOOR, gate.SHORTLIST_K) == (0.30, 1)
    assert (gate.NOOP_THRESHOLD, gate.UPDATE_THRESHOLD, gate.LABEL_MASS_FLOOR) == \
        (0.8, 0.3, 0.3)
    assert (gate.FACT_CHARS, gate.TIMEOUT_S) == (400, 3.0)
    gate_write = inspect.getsource(gate.gate_write)
    assert 'if m == "shadow" or (m == "noop" and took == "update"):' in gate_write
    assert 'target["expired_at"]' in gate_write
    assert 'raw if raw in MODES else "off"' in inspect.getsource(gate.mode)


def test_no_code_refuses_on_so_the_reservation_really_is_prose(monkeypatch):
    # The comment's sharpest claim is behavioural: mode() would arm `on` today
    # if the env said so. If a later round adds a guard, this node is the one
    # that says the prose may now say something weaker.
    monkeypatch.setenv(gate.MODE_ENV, "on")
    assert gate.mode() == "on"


# ── #2441: the n = 60 hand-labelled UPDATE sample, and the note that carries it ─
#
# #2344 left one thing owed to a round that no round could do with a log count:
# sixty hand-labelled E/S/P/D UPDATE verdicts, scored into a false-supersede CI,
# beside the 2026-09-25 baselines. The result is prose
# (`eval/measurements/fact-write-gate-2026-10-08.md`), and prose about a ship bar
# is exactly the kind that outlives its evidence, so every figure the note prints
# is checked here against bytes this repo now carries: the 60 sampled rows, their
# labels, the seed and window of the draw, and the 160 calibration `new_hash`
# values the overlap is computed against. Two of these nodes cross a real process
# boundary rather than re-reading a constant: one runs the shipped scorer
# (`eval/run_fact_write_gate_eval.py score`) as a subprocess over the committed
# bytes and demands that the note quote its output, and one re-derives the note's
# overlap count through `app.kg_store.text_hash`, the store's own key function.
#
# What stays out of reach of a gate, and is recorded as owed on #2441 rather than
# pretended away: whether a Lloyd-agent labeler is the labeler the bar demands
# (the note names its own), and measurement (b) of two — a paired LloydMemEval
# `knowledge_update` gain — which no number in this file is evidence for.

#: The note, its committed sample directory, and the note it supplements.
NOTE_2441 = ROOT / "eval" / "measurements" / "fact-write-gate-2026-10-08.md"
SAMPLE_2441 = ROOT / "eval" / "measurements" / "fact-write-gate-2026-10-08"
BASELINE_NOTE = ROOT / "eval" / "measurements" / "fact-write-gate-2026-09-25.md"
#: The one line per row the scorer consumes: `<row> <E|S|P|D>`.
LABELS_2441 = SAMPLE_2441 / "labels.txt"
#: Seed drawn from the item id, so a re-draw is never mistaken for a tuning.
SEED_2441 = 2441


def _note2441() -> str:
    assert NOTE_2441.is_file(), f"{NOTE_2441} is the measurement note #2441 exists to write"
    return NOTE_2441.read_text(encoding="utf-8")


def _prose(text: str) -> str:
    """One whitespace, so a claim wrapped across two source lines is still one
    sentence to match. Every phrase asserted below is matched through this."""
    return " ".join(text.split())


def _sample2441() -> list[dict]:
    path = SAMPLE_2441 / "decisions.jsonl"
    assert path.is_file(), f"the sampled rows behind the note ({path}) are not in the tree"
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _labels2441() -> dict[int, str]:
    """The same shape `run_fact_write_gate_eval._labels` reads. The authority for
    what these labels score is the subprocess node, which runs that parser."""
    out = {}
    for line in LABELS_2441.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] in "ESPD":
            out[int(parts[0])] = parts[1]
    return out


def _sample_meta() -> dict:
    """The one line recording how the committed draw was taken. `.jsonl`, because
    `.gitignore:42` ignores `*.json` repo-wide except for named negations, and a
    witness written as `.json` would never have been committed at all."""
    return json.loads((SAMPLE_2441 / "sample.jsonl").read_text(encoding="utf-8"))


def _calibration() -> tuple[dict, list[str]]:
    """`(<header>, the new_hash values>`), one hash per line so the 160 is also a
    line count of the committed bytes."""
    recs = [json.loads(line) for line in
            (SAMPLE_2441 / "calibration-new-hashes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return recs[0]["_header"], [r["new_hash"] for r in recs[1:]]


def _stated_n(text: str) -> int:
    """The largest `n = <int>` the note states — the sample size it claims."""
    found = [int(m) for m in re.findall(r"\bn = (\d+)", text)]
    assert found, "the note states no `n = <count>`"
    return max(found)


#: `k/n [= p] … [lo, hi]` on one line: the note's printed proportion and the
#: interval printed beside it. Lazy gap so each `k/n` binds to its OWN bracket,
#: which matters on the table row carrying both this note's and the baseline's.
_KN_INTERVAL = re.compile(r"(\d+)/(\d+)(?:\s*=\s*[\d.]+)?[^\[\n]{0,40}?"
                          r"\[(\d\.\d{1,3}),\s*(\d\.\d{1,3})\]")
#: … and the prose form that cites only the bound the ship decision weighs.
_UPPER_BOUND = re.compile(r"(\d+)/(\d+)[^\[\n]{0,60}?Wilson upper bound (\d\.\d{1,3})")
#: Half a millipoint: the note prints three decimals, so anything looser than the
#: rounding itself is a different interval.
_CI_TOL = 5e-4


def _interval_problems(text: str) -> list[str]:
    """Every printed interval in `text` that `eval.stats.wilson_ci` did not print."""
    from eval.stats import wilson_ci
    bad = []
    for line in text.splitlines():
        for k, n, lo, hi in _KN_INTERVAL.findall(line):
            k, n = int(k), int(n)
            if n == 0 or k > n:
                bad.append(f"{line.strip()}: {k}/{n} is not a proportion")
                continue
            wlo, whi = wilson_ci(k, n)
            if abs(float(lo) - wlo) > _CI_TOL or abs(float(hi) - whi) > _CI_TOL:
                bad.append(f"{line.strip()}: printed [{lo}, {hi}] but "
                           f"wilson_ci({k}, {n}) = ({wlo:.3f}, {whi:.3f})")
        for k, n, hi in _UPPER_BOUND.findall(line):
            k, n = int(k), int(n)
            if n == 0 or k > n:
                bad.append(f"{line.strip()}: {k}/{n} is not a proportion")
                continue
            wlo, whi = wilson_ci(k, n)
            if abs(float(hi) - whi) > _CI_TOL:
                bad.append(f"{line.strip()}: upper bound {hi} but "
                           f"wilson_ci({k}, {n})[1] = {whi:.3f}")
    return bad


#: "this note/CI/sample … clears/arms/authorises … the bar/gate/mode", in that
#: order, in a sentence that does not negate it. The note has to say a great deal
#: about arming in order to say it does not authorize arming, so the shape that
#: is actually dangerous is an affirmative clause with the note as its subject.
_ARMING_CLAIM = re.compile(
    r"(?:this|these)\s+(?:note|ci|sample|measurement|numbers?)|the\s+ci",
    re.I)

#: … and the dangerous predicate, as whole words. "arming" is an OBJECT word
#: ("clears the arming bar"), never a verb here, or every honest sentence that
#: names the bar reads as a claim to have crossed it.
_ARMING_TAIL = re.compile(
    r"[^.!?\n]{0,50}?\b(?:clears?|arms|armed|authoriz\w*|justif\w*|satisfies?|meets?)\b"
    r"[^.!?\n]{0,50}?(?:arming|bar|gate|mode|threshold)", re.I)
_NEGATION = re.compile(r"\b(?:not|never|nor|nothing|without)\b", re.I)


def _affirmative_arming_claims(text: str) -> list[str]:
    """Every sentence that credits the note/CI/sample with clearing the bar.

    The note must name the bar to explain why it does not clear it, so the shape
    worth refusing is an AFFIRMATIVE clause with the note as its subject and an
    un-negated clearing verb in front of an arming object.
    """
    out = []
    # `.**` ends a bolded lead-in, so the split swallows trailing stars too.
    for chunk in re.split(r"(?<=[.!])\*{0,2}\s+|\n", text):
        if _NEGATION.search(chunk) or not _ARMING_CLAIM.search(chunk):
            continue
        tail = chunk[_ARMING_CLAIM.search(chunk).end():]
        if _ARMING_TAIL.search(tail):
            out.append(" ".join(chunk.split()))
    return out


# ── clause 1: the note states n, the labeler, the seed, the window, the path ──

def test_the_2441_note_states_n_the_labeler_the_seed_the_window_and_the_labels():
    note = _prose(_note2441())
    # Dated later than the note it supplements, by its own filename and in body.
    assert NOTE_2441.name > "fact-write-gate-2026-09-25.md", NOTE_2441.name
    assert datetime.date.fromisoformat(
        re.search(r"fact-write-gate-(\d{4}-\d{2}-\d{2})", NOTE_2441.name).group(1)) \
        > datetime.date(2026, 9, 25)
    assert _stated_n(note) >= 60, _stated_n(note)
    # The control for that parse: the same reader reading a smaller n says so.
    assert _stated_n("scored at **n = 12** beside n = 3") == 12
    # Seed, and the seed the committed draw actually carries, are the same number.
    assert f"seed {SEED_2441}" in note, note[:400]
    assert _sample_meta()["seed"] == SEED_2441
    # The window sampled, stated as a range and matching the committed rows.
    assert "2026-09-26 to 2026-10-08" in note
    assert _sample_meta()["frame_window"] == ["2026-09-26", "2026-10-08"]
    # The labeler, by name, and named as not djev.
    assert "Lloyd labelled this sample" in note
    assert "SM_20261008_220449" in note
    assert "djev as labeler because djev is the system under test" in note
    # The label file it was scored from, and the committed copy of those bytes.
    assert "~/lloyd-data/eval/2441/labels.txt" in note
    assert "eval/measurements/fact-write-gate-2026-10-08/labels.txt" in note
    for name, sha in _sample_meta()["sha256"].items():
        assert sha in note, (name, sha)


def test_the_committed_sample_is_the_size_the_note_claims_and_every_row_is_labelled():
    # The note's `n` is a claim about bytes in this repo, not about prose: a row
    # dropped from the extract, or left unlabelled, moves n and reddens this.
    rows, labels = _sample2441(), _labels2441()
    assert len(rows) == _sample_meta()["n"] == _stated_n(_prose(_note2441())) >= 60
    assert sorted(labels) == sorted(r["row"] for r in rows) == list(range(len(rows)))
    assert set(labels.values()) <= set("ESPD")
    # Every row is the shape the label is asked about: a new fact and an existing
    # one it would have superseded, from the log's own `target`/`candidates`.
    assert all(r["verdict"] == "update" and r["fact"] and r["target"]["fact"] for r in rows)
    assert all(r["fact"] != r["target"]["fact"] for r in rows)
    assert len({r["new_hash"] for r in rows}) == len(rows), "the sample holds a repeat row"
    # And the committed bytes are the bytes the note says it scored.
    import hashlib
    for name, sha in _sample_meta()["sha256"].items():
        assert hashlib.sha256((SAMPLE_2441 / name).read_bytes()).hexdigest().startswith(sha)


# ── clause 2: every printed interval is the one eval.stats.wilson_ci prints ───

def test_every_wilson_interval_the_note_prints_is_the_one_eval_stats_prints():
    note = _note2441()
    # Both named measurements are there, as k/n plus an interval, before the scan.
    # In the fenced block the labels read as the scorer prints them; in the table
    # the same two measurements carry a pipe-escaped label and a bolder value, so
    # the separator between label and number is matched loosely on purpose.
    scored = re.findall(r"(UPDATE safe \(E\|P\)|false supersede \(S\|D\))"
                        r"[^\d\n]{1,10}(\d+/\d+) = (\d\.\d{3}) \[(\d\.\d{3}), (\d\.\d{3})\]",
                        note)
    assert {m[0] for m in scored} == {"UPDATE safe (E|P)", "false supersede (S|D)"}, scored
    assert not _interval_problems(note), _interval_problems(note)
    # The scan is an instrument, not a shrug: it fires on a bound that moved.
    assert _interval_problems("UPDATE safe (E|P):      60/60 = 1.000 [0.940, 0.999]")
    assert _interval_problems("false supersede (S|D):  0/60 = 0.000 [0.000, 0.160]")
    assert _interval_problems("the old figure was 0/12, Wilson upper bound 0.281")
    assert not _interval_problems("UPDATE safe (E|P):      60/60 = 1.000 [0.940, 1.000]")


def test_the_note_numbers_are_what_the_shipped_scorer_prints_over_the_committed_bytes():
    """The seam: `eval/run_fact_write_gate_eval.py score` runs as a subprocess over
    the committed extract, and the note has to quote its output line for line. The
    CIs in the note are therefore the scorer's, not the note author's arithmetic."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "run_fact_write_gate_eval.py"), "score",
         "--decisions", str(SAMPLE_2441 / "decisions.jsonl"),
         "--labels", str(LABELS_2441)],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-800:]
    printed = {}
    for label in ("UPDATE safe (E|P):", "false supersede (S|D):"):
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(label)), None)
        assert line, f"{label!r} missing from:\n{proc.stdout}"
        quoted = next((ln for ln in _note2441().splitlines() if ln.startswith(label)), None)
        assert quoted, f"the note quotes no {label!r} line"
        printed[label] = " ".join(line.split())
        assert " ".join(quoted.split()) == printed[label], (quoted, line)
    # And the scorer's numerator and denominator are the labels' own tally, so the
    # line the note quotes is a reading of these bytes, not a pasted number.
    labels = _labels2441()
    n = len(labels)
    safe = sum(1 for lab in labels.values() if lab in "EP")
    false_sup = sum(1 for lab in labels.values() if lab in "SD")
    assert f" {safe}/{n} = " in printed["UPDATE safe (E|P):"], (safe, n, printed)
    assert f" {false_sup}/{n} = " in printed["false supersede (S|D):"], (false_sup, n, printed)
    assert {k: sum(1 for lab in labels.values() if lab == k) for k in "ESPD"} \
        == {"E": 5, "S": 0, "P": 55, "D": 0}, labels
    assert "**55 P, 5 E, 0 S, 0 D.**" in _prose(_note2441())


# ── clause 3: labels from log rows (no replay), and the calibration overlap ───

def test_the_note_records_that_the_labels_came_from_log_rows_so_no_replay_ran():
    prose = _prose(_note2441())
    for phrase in ("the gate log's own `fact`", "`target.fact` / `candidates[0].fact`",
                   "no replay and no `.backup` store copy were needed",
                   "`cmd_score` opens only `--decisions` and `--labels`",
                   "which only `replay` reaches"):
        assert phrase in prose, phrase
    # The claim that djev was asked nothing is a claim about the rows, checked:
    # the frame is every row the gate asked, so an un-asked row could not carry
    # a verdict at all — and the note says so rather than leaving it implied.
    assert all(r["asked"] for r in _sample2441())
    assert "djev was not asked a single question for this note" in prose


def test_the_stated_calibration_overlap_is_re_derivable_from_the_committed_bytes():
    """The note's overlap figure recomputed through `app.kg_store.text_hash` — the
    store's own key — over the committed rows against the committed calibration
    hashes. A sample drawn off the set the cutoffs were tuned on is not held out,
    so the figure the note quotes has to survive being re-derived."""
    cal, hashes = _calibration()
    assert len(hashes) == len(set(hashes)) == cal["n_distinct_new_hash"] == 160, cal
    assert cal["source"] == "~/lloyd-data/eval/1487/pairs.json" and cal["field"] == "new_hash"
    rows = _sample2441()
    known = set(hashes)
    overlap = {r["row"] for r in rows if kg_store.text_hash(r["fact"]) in known}
    assert overlap == {r["row"] for r in rows if r["in_calibration"]}
    assert overlap == set(_sample_meta()["calibration_overlap_rows"])
    found = re.search(r"\*\*(\d+) of the (\d+)\*\* sampled new facts", _note2441())
    assert found, "the note states no `**N of the M** sampled new facts` overlap"
    assert (int(found.group(1)), int(found.group(2))) == (len(overlap), len(rows)), found.groups()
    assert cal["hash_function"].startswith("app.kg_store.text_hash"), cal["hash_function"]


# ── clause 4: both 09-25 baselines, quoted as they read there today ───────────

def test_the_note_sits_beside_both_09_25_baselines_quoted_as_they_read_there():
    prose, baseline = _prose(_note2441()), _prose(BASELINE_NOTE.read_text(encoding="utf-8"))
    # Quoted here…
    assert "0/12, Wilson upper bound 0.243" in prose
    assert "3/3 [0.438, 1.0]" in prose
    # …and quoted the way the 09-25 note actually writes them, so the "as they read
    # today" half of the clause is checked against the source rather than memory.
    assert "0/12, Wilson upper bound 0.243" in baseline
    assert "([0.438, 1.0])" in baseline
    # The two are different measurements, and the note says so instead of merging
    # them into one "n=12 UPDATE-safe" figure that was never taken.
    assert "they are not the same\n" in _note2441() or "they are not the same measurement" in prose
    for words in ("neither is replaced by the row above it", "this note supplements both"):
        assert words in prose, words
    assert "0.243 at n=12 becomes 0.060 at n = 60" in prose


# ── clause 5: the close names measurement (b) and authorizes nothing ─────────

def test_the_close_completes_measurement_a_only_and_names_the_untaken_measurement_b():
    prose = _prose(_note2441())
    assert "## What this does and does not authorize" in prose
    assert "**two measurements plus a" in prose and "decision**" in prose
    assert "completes measurement (a) only" in prose
    for words in ("eval/fact_write_gate_snapshot.py", "run_memory_eval.py --fact-snapshot",
                  "`knowledge_update`", "Alan's explicit ship decision", "**not taken**"):
        assert words in prose, words
    # The half the CI cannot do: a clean interval at n = 60 still says nothing
    # about (b), and the note is required to say that out loud.
    assert "no paired run is on record" in prose
    # And the fence is still standing where the note says it is: no config value
    # moved, and the round-facing refusal is a real one.
    assert _config()["knowledge_graph"]["write_gate"]["mode"] == "noop"
    assert "test_a_round_cannot_arm_the_fact_write_gate" in prose


def test_the_note_arming_scan_fires_only_on_an_affirmative_claim():
    note = _note2441()
    assert not _affirmative_arming_claims(note), _affirmative_arming_claims(note)
    # The control, or the scan above is a scan that matches nothing.
    assert _affirmative_arming_claims("This CI now clears the arming bar for the gate.")
    assert _affirmative_arming_claims("The CI satisfies the bar.")
    # Negated, the same shape is exactly what the note is allowed to say.
    assert not _affirmative_arming_claims("This note does not authorize flipping the mode.")
