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
# What stays out of reach of a gate: measurement (b) of two — a paired LloydMemEval
# `knowledge_update` gain — which no number in this file is evidence for, and
# Alan's ship decision. The labeler question this block used to name as owed was
# RULED on 2026-10-11 (#2441's `owed_settled` entry 1: an agent reader's labels
# count as measurement (a)'s evidence provisionally and are not hand labels), #2556
# recorded the ruling in the note, and the #2556 nodes at the foot of this file pin
# it. What is still owed there is Alan's spot-check at ship time, which no round may
# take or simulate.

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


# ── #2556: the 2026-10-11 labeler ruling, recorded in the note it rules on ────
#
# #2441 left two things owed to a reader that is not a round, and owed-check ruled
# both on 2026-10-11 (vault `backlog/2441-*.md`, `owed_settled` entries 1 and 3,
# `follow_up: 2556`): a Lloyd-agent reader's labels count as measurement (a)'s
# evidence PROVISIONALLY and are not hand labels — bound (i) of that ruling is that
# the note may never be quoted as Alan's labels or as a human-labelled CI — and no
# recurring re-score is warranted. The note recorded neither. Worse, its title and
# the close's **(a)** bullet both called these 60 labels hand-labelled, which is the
# quotation bound (i) forbids: a reader who quoted the note without reaching "Who
# labelled" would have carried a human-labelled CI into the ship decision. So the
# nodes below pin the ruling to the sentence that carries it, and pin the ruling's
# own bound to the committed labels — the E-row indices Alan is to spot-check come
# out of `labels.txt`, not out of the prose that names them.
#
# What these nodes deliberately do NOT do: take the ship decision, or pretend the
# spot-check ran. Every clause here is a claim about what the note says, plus one
# claim about the labels file, and the arming scan above still has to stay green
# over the new sentences.

#: The paragraph #2556 removed, byte for byte as #2441 shipped it at 50bba19b. The
#: control that makes "the owed sentence is gone" below a finding rather than a
#: scan over text that never carried it.
PRE_RULING_WHO_LABELLED = """That does not settle the harder half of the clause, which #2441 leaves owed:
whether a Lloyd-agent reader is the reader the bar demands, or whether the sample
has to be Alan's. If the ruling comes back "Alan's", the rows are committed, the
label format is one line per row, and the label file is the thing to over-label —
the sample does not have to be drawn again."""

#: … and the close's **(a)** bullet as it shipped, which said bar (a) was "taken".
PRE_RULING_BULLET_A = """- **(a)** a false-supersede CI over >= 60 hand-labelled UPDATE verdicts — **taken
  by this note**: 0/60 = 0.000 [0.000, 0.060]."""

_WHO_LABELLED = re.compile(r"^## Who labelled\n(.*?)(?=^## )", re.M | re.S)
_BULLET_A = re.compile(r"^- \*\*\(a\)\*\*(.*?)(?=^- \*\*\(b\)\*\*)", re.M | re.S)


def _who_labelled(text: str) -> str:
    """The body of the '## Who labelled' section — the one section #2556 rewrites,
    so clauses 1, 2 and 4 are graded against the section that carries the ruling
    and not against a matching sentence parked somewhere else in the note."""
    found = _WHO_LABELLED.search(text)
    assert found, "the note has no '## Who labelled' section to record a ruling in"
    return _prose(found.group(1))


def _bullet_a(text: str) -> str:
    """The close's **(a)** bullet: the words a reader quotes when they quote the bar
    as satisfied. That the extraction finds it at all is shown by the control in the
    clause 3 node, which reads this same extractor over the shipped bullet."""
    found = _BULLET_A.search(text)
    assert found, "the note's close has no '- **(a)**' bullet to qualify"
    return _prose(found.group(1))


# ── clause 1: the labeler is ruled, and what was ruled is provisional ────────

def test_the_note_records_the_ruling_that_agent_labels_are_provisional_evidence():
    """Clause 1: 'Who labelled' no longer leaves the labeler owed, and says instead
    that an agent reader's labels count as measurement (a)'s evidence, provisionally."""
    note = _note2441()
    who = _who_labelled(note)
    assert "which #2441 leaves owed" not in _prose(note)
    assert "which #2441 leaves owed" in _prose(PRE_RULING_WHO_LABELLED)
    assert "Ruled on 2026-10-11" in who
    assert "COUNT as measurement (a)'s evidence, provisionally" in who
    assert "follow-up #2556" in who


# ── clause 2: not hand labels, and not quotable as one ───────────────────────

def test_the_note_withholds_the_hand_labelled_word_from_the_agent_labels():
    """Clause 2: bound (i) of the ruling — these labels are NOT hand labels, the
    bar's word is not satisfied by them, and the note may not be quoted as Alan's
    labels or as a human-labelled CI."""
    who = _who_labelled(_note2441())
    assert "They are not hand labels" in who
    assert 'The bar\'s word is "hand-labelled" and it is not satisfied by them' in who
    assert "may never be quoted as Alan's labels or as a human-labelled CI" in who
    # Control: the shipped section carried no such bound, so the three assertions
    # above are about the ruling and not about prose that was always there.
    assert "They are not hand labels" not in _prose(PRE_RULING_WHO_LABELLED)


# ── clause 3: title and **(a)** bullet stop asserting a hand-labelled sample ──

def test_the_title_and_the_measurement_a_bullet_no_longer_claim_hand_labels():
    """Clause 3: the two places a reader meets this note before 'Who labelled' — the
    title and the close's **(a)** bullet — cannot be read as claiming hand labels,
    while `_stated_n` still reads n = 60 off the note."""
    note = _note2441()
    title = note.splitlines()[0]
    assert not re.search(r"hand[- ]label", title, re.I), title
    assert "n = 60" in title, title
    a = _bullet_a(note)
    assert "agent-labelled, not hand-labelled" in a, a
    assert "provisional evidence toward (a)" in a, a
    assert "not (a) satisfied" in a, a
    assert "0/60 = 0.000 [0.000, 0.060]" in a, a
    assert "taken by this note" not in a
    # Controls: the ban bites on the shipped bullet, and the extractor found the
    # bullet it is quoting rather than matching an empty span.
    assert re.search(r"hand[- ]label", _prose(PRE_RULING_BULLET_A), re.I)
    assert "taken by this note" in _prose(PRE_RULING_BULLET_A)
    assert _stated_n(_prose(note)) == 60 == _sample_meta()["n"]


# ── clause 4: the ship-time guard, its indices read off the label file ────────

def test_the_note_records_the_ship_time_spot_check_and_the_no_redraw_bound():
    """Clause 4: the bounded guard that lets the provisional count stand — which
    rows Alan spot-checks at ship time, that he over-labels any line he rejects, and
    that the draw is never repeated. The five indices are checked against
    `labels.txt`, so the note cannot name a row that is not an E row."""
    who = _who_labelled(_note2441())
    assert "Alan spot-checks the committed labels" in who
    assert "over-labels any line he rejects" in who
    assert "The draw is never repeated" in who
    assert "one line per row" in who
    found = re.search(r"the 5 E rows \(0-indexed ([\d, ]+)\) plus (\d+) P rows", who)
    assert found, who
    assert [int(x) for x in found.group(1).split(",")] == [16, 37, 47, 50, 57]
    assert int(found.group(2)) == 10
    # …and those five ARE the E-labelled rows of the committed sample: the
    # spot-check list is a reading of labels.txt, not a list the prose invented.
    assert sorted(r for r, lab in _labels2441().items() if lab == "E") == \
        [16, 37, 47, 50, 57]


# ── clause 5: one re-score bundled with (b), and no recurring job ────────────

def test_the_note_rules_out_a_recurring_re_score_and_bundles_the_one_it_allows():
    """Clause 5: #2441's entry-3 ruling — no recurring re-score and no autonomy
    task, exactly one re-score bundled with the measurement-(b) run, an early one
    only if a labelled reader records an S or D row. That trigger is shown never to
    have fired over the committed sample, which is why the bound is a trigger and
    not a schedule."""
    prose = _prose(_note2441())
    assert "No recurring re-score is warranted" in prose
    assert "no autonomy task for one" in prose
    assert "exactly one re-score, bundled with the measurement-(b) run" in prose
    assert "an early one only if a labelled reader ever records an S or D row" in prose
    committed = _labels2441()
    assert sum(1 for lab in committed.values() if lab in "SD") == 0
    # Control for that zero: the same reader over one planted S row says one.
    assert sum(1 for lab in {**committed, 58: "S"}.values() if lab in "SD") == 1


# ───────────────────────────────────────────────────────── #2768 measurement (b) ──
# The paired `knowledge_update` gain bar (b) is stated in. The instrument existed
# (`eval/fact_write_gate_snapshot.py`, `run --fact-snapshot`) and had never been
# run, and two of its pieces had to be built before the bar could be measured:
# a targeted replay list (pinned in `tests/test_fact_write_gate_snapshot.py`) and
# a cross-artifact join, because the runner's only paired interval is arm-vs-arm
# inside one process, while a gate-mode pair is one arm across two processes.
MEAS_B = ROOT / "eval" / "measurements" / "fact-write-gate-b-2026-10-15.md"
DIR_B = ROOT / "eval" / "measurements" / "fact-write-gate-b-2026-10-15"
JOIN_B = DIR_B / "knowledge-update-join.jsonl"


def _leg(ids: list[str], correct: list[int], arm: str = "prefetch_rel") -> dict:
    return {"rows": [{"id": i, "arm": arm, "category": "knowledge_update",
                      "correct": c} for i, c in zip(ids, correct)]}


def test_the_join_pairs_two_artifacts_on_the_question_id():
    """The delta bar (b) is stated in cannot be computed by the runner as shipped.

    `run`'s `dev_comparisons` pair arms within one artifact and
    `eval/stats.paired_bootstrap_ci` takes two vectors over shared records;
    two `--fact-snapshot` legs are two artifacts, so the join lives here and names
    its own seed and resample count, which is what the note has to be able to say.
    """
    from eval import run_memory_eval as rme
    ids = [f"q{i}" for i in range(4)]
    res = rme.join_arms({"noop": _leg(ids, [1, 1, 0, 0]),
                         "on": _leg(ids, [1, 1, 1, 0])},
                        arm="prefetch_rel", metric="correct")
    assert res["n"] == 4
    assert res["diff"] == pytest.approx(0.25)
    assert res["flips_a_to_b"] == 1 and res["flips_b_to_a"] == 0
    assert res["ci"][0] <= 0.0 <= res["ci"][1]
    from eval import stats
    assert res["bootstrap"] == {"seed": stats.SEED, "n_resamples": stats.N_RESAMPLES,
                                "source": "eval.stats defaults (SEED/N_RESAMPLES)"}
    assert [r["id"] for r in res["per_question"]] == ids


def test_the_join_carries_a_string_shaped_fact_snapshot_through():
    """`run` writes `fact_snapshot` as the root PATH, not a dict.

    Reading it as one raises `AttributeError` on the first real artifact and never
    on a fixture that omits the field, so the join would have died on the first
    pair anyone ever tried to join.
    """
    from eval import run_memory_eval as rme
    leg = _leg(["q1", "q2"], [1, 0])
    leg["fact_snapshot"] = "/tmp/snapshots/noop"
    leg["facts_rank"] = {"configured": "relevance", "mismatch": []}
    res = rme.join_arms({"noop": leg, "on": _leg(["q1", "q2"], [1, 1])},
                        arm="prefetch_rel", metric="correct")
    assert res["provenance"]["noop"]["fact_snapshot"] == "/tmp/snapshots/noop"


def test_the_join_refuses_a_question_that_appears_twice_in_one_leg():
    """Pooling two arms of one question into a leg halves the variance for free.

    The interval that comes out is narrower than the 95% it claims and looks
    entirely reasonable, which is the worst way for a measurement to be wrong.
    """
    from eval import run_memory_eval as rme
    doubled = _leg(["q1", "q2"], [1, 0])
    doubled["rows"].append({"id": "q1", "arm": "prefetch_rel",
                            "category": "knowledge_update", "correct": 1})
    with pytest.raises(SystemExit) as exc:
        rme.join_arms({"noop": doubled, "on": _leg(["q1", "q2"], [1, 1])},
                      arm="prefetch_rel", metric="correct")
    assert "narrower" in str(exc.value)


def test_the_join_refuses_legs_that_scored_different_questions():
    from eval import run_memory_eval as rme
    with pytest.raises(SystemExit) as exc:
        rme.join_arms({"noop": _leg(["q1", "q2"], [1, 0]),
                       "on": _leg(["q1", "q3"], [1, 1])},
                      arm="prefetch_rel", metric="correct")
    assert "different question sets" in str(exc.value)


def test_dump_blocks_is_refused_for_a_leg_that_rendered_no_facts_block(capsys):
    """A dump written by a run with no prefetch arm would be an empty witness."""
    from eval import run_memory_eval as rme
    with pytest.raises(SystemExit) as exc:
        rme.run(["--arms", "closed_book,history", "--dump-blocks", "/tmp/x.jsonl"])
    assert exc.value.code == 2
    assert "needs a prefetch arm" in capsys.readouterr().err


def test_categories_is_validated_before_the_set_is_loaded(capsys):
    """A typo'd category names the real ones instead of scoring zero questions."""
    from eval import run_memory_eval as rme
    with pytest.raises(SystemExit) as exc:
        rme.run(["--arms", "prefetch_rel", "--categories", "not_a_category"])
    assert exc.value.code == 2
    assert "knowledge_update" in capsys.readouterr().err


def test_join_writes_the_committed_jsonl_form_and_nothing_the_header_denies(tmp_path):
    """`.jsonl` out: one header row with the result, then exactly n question rows.

    Two reasons this is a real boundary and not a formatting preference. The
    committed evidence of a measurement has to be a tracked file, and
    `.gitignore:42` keeps every `*.json` untracked — so the note cites a `.jsonl`,
    and the file it cites has to exist in that form. And the row count has to equal
    the header's `n`: an artifact whose rows disagree with its own `n` is read as a
    wider measurement than the one that was made.
    """
    from eval import run_memory_eval as rme

    def artifact(path: Path, correct: list[int]) -> Path:
        doc = {"eval": "lloydmemeval.run.v1", "arms": ["prefetch_rel"],
               "fact_snapshot": str(path.parent / "snap"),
               "rows": [{"id": f"q{i}", "arm": "prefetch_rel",
                         "category": "knowledge_update", "correct": c}
                        for i, c in enumerate(correct)]}
        path.write_text(json.dumps(doc), encoding="utf-8")
        return path

    a = artifact(tmp_path / "noop.json", [1, 1, 0, 0])
    b = artifact(tmp_path / "on.json", [1, 1, 1, 0])
    out = tmp_path / "join" / "knowledge-update-join.jsonl"
    rme.join_cli(["--leg", f"noop={a}", "--leg", f"on={b}",
                  "--arm", "prefetch_rel", "--category", "knowledge_update",
                  "--out", str(out)])
    lines = out.read_text(encoding="utf-8").splitlines()
    head = json.loads(lines[0])
    assert head["_header"] == "run_memory_eval join" and head["n"] == 4
    assert head["diff"] == pytest.approx(0.25) and "per_question" not in head
    rows = [json.loads(l) for l in lines[1:]]
    assert len(rows) == head["n"] == 4
    assert {r["category"] for r in rows} == {"knowledge_update"}


def test_join_refuses_a_leg_that_scored_no_rows_of_the_named_category(tmp_path):
    """The other way a named bar can silently be a zero-question measurement."""
    from eval import run_memory_eval as rme

    def artifact(path: Path, cat: str) -> Path:
        doc = {"eval": "lloydmemeval.run.v1", "arms": ["prefetch_rel"],
               "rows": [{"id": "q1", "arm": "prefetch_rel", "category": cat,
                         "correct": 1}]}
        path.write_text(json.dumps(doc), encoding="utf-8")
        return path

    a = artifact(tmp_path / "noop.json", "temporal")
    b = artifact(tmp_path / "on.json", "knowledge_update")
    with pytest.raises(SystemExit) as exc:
        rme.join_arms({"noop": rme.read_artifact(a), "on": rme.read_artifact(b)},
                      arm="prefetch_rel", metric="correct",
                      category="knowledge_update")
    assert "no" in str(exc.value).lower()


def _b_note() -> str:
    assert MEAS_B.is_file(), f"{MEAS_B} is the record measurement (b) was taken"
    return MEAS_B.read_text(encoding="utf-8")


def test_the_b_note_records_the_paired_delta_with_its_interval_and_both_commands():
    """Clause 1: the number, the interval, and the two command lines that made it."""
    note = _b_note()
    # The headline interval itself, not any bracketed number: the note carries
    # several Wilson intervals, and a check that matched one of those would pass on
    # a note that never stated the paired result.
    assert "+0.000, 95% CI [-0.113, +0.094]" in note.replace("**", ""), (
        "the paired delta and its 95% interval must be stated together")
    assert "[+0.000, +0.094]" in note, "the correct_strict / evidence rows are part of the record"
    assert re.search(r"n=53", note), "n is the 53 scored knowledge_update questions"
    assert "seed" in note and "n_resamples" in note, (
        "the interval is a bootstrap, so its seed and resample count are part of it")
    assert JOIN_B.is_file(), (
        f"{JOIN_B} is the join's own output; the interval in the note is readable "
        f"there and cannot be a transcription of a terminal")
    for snapshot in ("pair-2026-10-15/noop", "pair-2026-10-15/on"):
        assert snapshot in note, f"the {snapshot} leg's command line is missing"
    assert note.count("--arms prefetch_rel") >= 2, (
        "both arms' command lines must name the scored arm: --fact-snapshot alone "
        "does not say which <facts> ordering was scored")


def test_every_artifact_the_b_note_cites_is_a_tracked_file_and_not_ignored():
    """Clause 1's last half: the evidence a note cites has to survive `.gitignore`.

    `.gitignore:42` is a blanket `*.json`, so every `run`/`join` artifact — including
    the two artifacts this measurement was scored from — is invisible to git by
    default. A note that cites one of those would be citing a file that does not
    exist for anyone who checks it out, which is how a measurement becomes
    unauditable after the machine that made it is gone. `git check-ignore` is the
    only faithful reader of that rule, so the test asks git rather than re-implementing
    the pattern.
    """
    import subprocess

    note = _b_note()
    cited = sorted({name for name in re.findall(r"`([\w.+-]+\.(?:jsonl|md))`", note)
                    if (DIR_B / name).is_file()})
    assert len(cited) >= 4, (
        f"the note cites {cited}; it has to name the block dumps, the join and the "
        f"build manifest beside it")
    for name in cited:
        path = DIR_B / name
        probe = subprocess.run(
            ["git", "-C", str(ROOT), "check-ignore", "-q", str(path.relative_to(ROOT))],
            capture_output=True)
        assert probe.returncode != 0, (
            f"{name} is swallowed by .gitignore, so the evidence {name} is cited for "
            f"is untracked and unreadable to anyone checking this repo out")
        listed = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
             str(path.relative_to(ROOT))], capture_output=True)
        assert listed.returncode == 0, f"{name} is not tracked"
    assert not list(DIR_B.glob("*.json")), (
        "a .json artifact in the measurement dir is gitignored by .gitignore's "
        "blanket `*.json`; the committed form is the .jsonl `--out` writes")


def _brace_names(spec: str) -> list[str]:
    """`witness-entity-facts-{noop,on}.jsonl` -> the two files it stands for."""
    m = re.search(r"\{([^}]*)\}", spec)
    if not m:
        return [spec]
    return [spec[:m.start()] + alt + spec[m.end():]
            for alt in m.group(1).split(",")]


def test_every_node_and_artifact_the_clause_map_names_resolves():
    """The map is a claim about this repo, so it is checked against this repo.

    The note's opening table is the entry point for anyone auditing one clause of
    #2768: it says which section holds the evidence, which file is the artifact, and
    which test node pins it. Each of those three can rot on its own — a renamed test
    node, a deleted artifact, a section heading reworded — and a reader following a
    stale pointer concludes the clause is unpinned, which is the failure this item has
    already paid for twice. So every backticked span in the map is resolved here:
    `path::node` must be a `def node(` in a file that exists, and every artifact name
    (brace form expanded) must be a tracked file.
    """
    import subprocess

    note = _b_note()
    start = note.find("## Clause map")
    assert start >= 0, "the note's clause map heading is what an auditor starts at"
    section = note[start:note.index("## What bar (b) asked for", start)]
    spans = re.findall(r"`([^`]+)`", section)
    nodes: list[tuple[str, str]] = []
    artifacts: list[str] = []
    last_file = None
    for span in spans:
        m = re.fullmatch(r"(tests/[\w/]+\.py)?(::)([a-z0-9_]+)", span)
        if m:
            file_part, _, node = m.group(1), m.group(2), m.group(3)
            if file_part:
                last_file = file_part
            assert last_file is not None, (
                f"{span} names a node before any test file appears in the map")
            nodes.append((last_file, node))
        elif re.fullmatch(r"[\w./+,-]+\.(jsonl|md|txt)", span):
            artifacts.append(span)
    assert len(nodes) >= 11, f"the map pins only {len(nodes)} node(s)"
    assert len(artifacts) >= 6, f"the map names only {artifacts}"
    sources: dict[str, str] = {}
    for rel, node in nodes:
        path = ROOT / rel
        assert path.is_file(), f"the map points at {rel}, which is not in the repo"
        sources.setdefault(rel, path.read_text(encoding="utf-8"))
        assert f"def {node}(" in sources[rel], (
            f"{rel} has no `def {node}(` — the map cites a test node that does not "
            f"exist, so the clause it pins reads as unpinned")
    for spec in artifacts:
        for name in _brace_names(spec):
            # The map sits in the measurement dir, so `../x/labels.txt` is a sibling
            # measurement's file, not one level above the dir.
            path = (DIR_B / name).resolve()
            assert path.is_file(), f"the map names {name}, which is not on disk"
            listed = subprocess.run(
                ["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
                 str(path.relative_to(ROOT))], capture_output=True)
            assert listed.returncode == 0, f"{name} is cited by the map but untracked"
    for heading in ("The two legs", "The paired delta", "The witness",
                    "The bundled re-score", "Verdict on bar (b)"):
        assert heading in note, f"the map points at a section named {heading!r}"


def test_the_b_note_legs_both_scored_the_ordering_production_ships():
    """Clause 2 (#2778's rail, applied to the pair): no mismatch on the scored arm.

    `config.yaml` ships `prefetch.facts.rank: relevance`; `--arms` defaults to
    `closed_book,history,prefetch`, and the `prefetch` arm renders the confidence
    order unconditionally. A gain measured on that arm is not a gain for the
    ordering anyone runs, so the note's own command lines and both dumps have to
    say `prefetch_rel` — and the dumps' headers carry the runner's disclosure.
    """
    note = _b_note()
    assert "prefetch.facts.rank" in note
    for path in (DIR_B / "knowledge-update-blocks-noop.jsonl",
                 DIR_B / "knowledge-update-blocks-on.jsonl"):
        head = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert head["facts_rank"]["mismatch"] == [], (
            f"{path} scored an arm whose <facts> ordering is not the shipped one")


LEG_HEADERS = DIR_B / "leg-headers.jsonl"


def _header_rows(path: Path) -> dict[str, dict]:
    """Each leg's committed header row, keyed by leg name."""
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "_header" in row:
            continue
        out[row["leg"]] = row
    return out


def test_each_leg_artifacts_disclosure_is_committed_and_agrees_with_the_shipped_one():
    """Clause 2: each leg's ARTIFACT header, in a file a reader can actually open.

    The disclosure #2778 added lives in the `run` artifact, and every `run` artifact
    is `.json` — swallowed by `.gitignore:42`'s blanket `*.json`. So the claim "each
    leg's artifact carries the shipped `facts_rank` header with an empty mismatch"
    was, until `run_memory_eval.py headers`, checkable only through the block dumps:
    a different file, written by a different flag, that happens to reuse the same
    in-process disclosure. This pins the artifact's own header instead, and against
    the shipped function rather than a substring — a committed block that merely
    happened to contain the word `relevance` would pass a text search, so the row is
    required to equal `facts_rank_disclosure(arms, configured=…)` for the ordering
    `config.yaml` carries, `mismatch: []` included, and to carry the digest of the
    artifact it was read from so it cannot be quoted off some other file.
    """
    from eval import run_memory_eval as rme

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    configured = cfg["prefetch"]["facts"]["rank"]
    assert configured == "relevance", (
        "the shipped ordering is what this clause is about; if config moved, the "
        "expected block has to be re-derived rather than this assertion kept")
    assert LEG_HEADERS.is_file(), (
        f"{LEG_HEADERS} is the tracked projection of each leg's run-artifact header; "
        f"without it the disclosure clause 2 names sits in a gitignored .json")
    rows = _header_rows(LEG_HEADERS)
    assert {"noop", "on"} <= set(rows), f"{sorted(rows)} — both legs' rows are needed"
    for leg in ("noop", "on"):
        row = rows[leg]
        assert row["facts_rank"]["mismatch"] == [], (
            f"the {leg} leg scored an arm whose <facts> ordering is not the shipped "
            f"{configured!r}: {row['facts_rank']['mismatch']}")
        assert row["facts_rank"] == rme.facts_rank_disclosure(
            row["arms"], configured=configured), (
            f"the {leg} row's facts_rank block is not what "
            f"`facts_rank_disclosure({row['arms']}, configured={configured!r})` "
            f"returns: {row['facts_rank']}")
        assert row["fact_snapshot"].endswith(f"/{leg}"), (
            f"the {leg} row's artifact names {row['fact_snapshot']!r}, not the "
            f"{leg} snapshot tree, so the leg cannot be told apart from its pair")
        assert re.fullmatch(r"[0-9a-f]{64}", row["artifact_sha256"]), (
            f"the {leg} row carries no digest of the artifact it was projected from")
    assert f"`{LEG_HEADERS.name}`" in _b_note(), (
        f"the note has to cite {LEG_HEADERS.name} — the tracked-artifact check only "
        f"looks at the files the note names")


def _artifact(arm: str, *, leg: str = "on", disclose: bool = True) -> dict:
    """A `run` artifact shaped like the real one, small enough to hold in a test."""
    doc: dict = {
        "created": "2026-10-15T18:39:29", "set": {"version": "v3", "view": "dev"},
        "item": 2768, "arms": [arm], "min_category_n": 20, "wall_s": 210.0,
        "fact_snapshot": f"/scratch/pair-2026-10-15/{leg}",
        "prefetch_corpus": f"/scratch/pair-2026-10-15/{leg}/facts",
        "answerer": {"model": "Qwen3.8-Flash-Next-nvfp4", "temperature": 0},
        "judge": {"model": "djev"},
        "rows": [{"id": "lme-kn-013", "arm": arm, "category": "knowledge_update",
                  "correct": 1}],
    }
    if disclose:
        doc["facts_rank"] = {
            "config_key": "prefetch.facts.rank", "configured": "relevance",
            "arm_ordering": {arm: ("confidence" if arm == "prefetch" else "relevance")},
            "mismatch": ([{"arm": "prefetch", "rendered": "confidence",
                           "configured": "relevance"}] if arm == "prefetch" else [])}
    return doc


def test_the_headers_command_exits_nonzero_on_a_leg_that_is_not_the_shipped_ordering(
        tmp_path):
    """The rail clause 2 rests on is the command's exit status, not one assert.

    A measurement note is written by whoever ran the command, so "this committed
    header is empty of mismatch" has to be a property of the tool that writes it:
    `headers` returns 1 when a scored arm's ordering disagrees with the ordering the
    artifact itself says is configured. It still writes the row, because a mismatch
    is evidence too and refusing to write it is how a pair gets re-run until the
    field comes out empty; `--allow-mismatch` is how one is committed on purpose,
    and the waiver never rewrites the mismatch it waived.
    """
    from eval import run_memory_eval as rme

    good = tmp_path / "lloydmemeval-good.json"
    good.write_text(json.dumps(_artifact("prefetch_rel")), encoding="utf-8")
    bad = tmp_path / "lloydmemeval-bad.json"
    bad.write_text(json.dumps(_artifact("prefetch", leg="noop")), encoding="utf-8")
    out = tmp_path / "leg-headers.jsonl"

    assert rme.main(["headers", "--leg", f"on={good}", "--out", str(out)]) == 0
    assert _header_rows(out)["on"]["artifact_sha256"], (
        "the row names the digest of the file it was read from")

    assert rme.main(["headers", "--leg", f"noop={bad}", "--out", str(out)]) == 1, (
        "a leg whose scored arm renders the confidence order while the artifact "
        "names relevance as configured must fail the command, not merely a test")
    assert rme.main(["headers", "--leg", f"noop={bad}", "--out", str(out),
                     "--allow-mismatch"]) == 0
    assert _header_rows(out)["noop"]["facts_rank"]["mismatch"], (
        "--allow-mismatch commits the row WITH its mismatch visible; a cleared "
        "mismatch under the waiver would be a falsified artifact")


def test_the_headers_command_refuses_an_artifact_that_discloses_nothing(tmp_path):
    """A row without a `facts_rank` block reads as `mismatch: []` — so it is refused.

    Artifacts written before #2778 carry no disclosure at all. Projecting one would
    put a row in the measurement dir whose absence a later reader scans past, and the
    block exists precisely so that an empty list means CHECKED-and-agrees rather than
    not-checked.
    """
    from eval import run_memory_eval as rme

    plain = tmp_path / "lloydmemeval-pre2778.json"
    plain.write_text(json.dumps(_artifact("prefetch_rel", disclose=False)),
                     encoding="utf-8")
    out = tmp_path / "leg-headers.jsonl"
    with pytest.raises(SystemExit) as caught:
        rme.main(["headers", "--leg", f"on={plain}", "--out", str(out)])
    assert caught.value.code == 2
    assert not out.exists(), "a refused projection must not leave a half-written row"


def test_the_b_note_names_the_trees_and_keeps_every_path_out_of_the_live_root():
    """Clause 4: the seed tree and snapshot root are named, and no path is the live one.

    The live root is named by its canonical location rather than through
    `production_data_root()`: a checkout of this repo in a worktree gets its own
    `.lloyd-data`, so inside the round `production_data_root()` IS the tree the pair
    was built in, and reading it here would make the assertion pass on `main` and
    fail in exactly the tree that has to be checked.
    """
    note = _b_note()
    assert "seed" in note and "snapshot root" in note
    live = str(Path.home() / "lloyd-data")
    checked = 0
    for line in note.splitlines():
        if not line.lstrip().startswith("$"):
            continue
        checked += 1
        assert live not in line, f"an arm's command line reaches the live root: {line}"
        assert "~/lloyd-data" not in line, (
            f"an arm names the live root by its tilde form: {line}")
    assert checked >= 2, "the note must carry both arms' command lines"


def test_the_b_note_ends_on_the_bar_and_the_gate_is_still_noop():
    """Clauses 5 and 4's config half: a verdict, Alan's decision, and no arming."""
    import yaml
    note = _b_note()
    # The FINAL section, not the last N characters: "ends with" is about the note's
    # closing section, and a fixed window would drift as the prose around the verdict
    # is edited — silently moving the check off the verdict it exists to pin.
    tail = note.rsplit("\n## ", 1)[-1]
    assert tail.startswith("Verdict"), (
        "the note's last section must be the verdict, not prose that follows it")
    assert re.search(r"\bUNMET\b", tail), (
        "the note must end with a met-or-unmet verdict on bar (b)")
    assert "Alan" in tail, "the arming decision is Alan's, and the note has to say so"
    assert "not-triggered" in note or "not triggered" in note
    assert "5 E" in note and "55 P" in note, (
        "the re-score is not-triggered because the 2026-10-08 labels are 5 E / 55 P "
        "with 0 S/D rows, and the note cites that rather than asserting it")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["knowledge_graph"]["write_gate"]["mode"] == "noop", (
        "measurement (b) does not arm the gate: arming is Alan's ship action")
