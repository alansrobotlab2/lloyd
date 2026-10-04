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
