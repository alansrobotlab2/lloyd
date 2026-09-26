"""LloydMemEval's runner (#1480): the frozen set's load contract, the "use"
judgement kept apart from retrieval, the artifact, the holdout reserve and the
self-judging refusal. Hermetic: no engine, no djev, no facts store."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import run_memory_eval as M  # noqa: E402

GENERATOR = "Qwen3.8-Flash-Next-nvfp4"


def _q(i: int, cat: str, *, gold="8182", anti=None, leg="dev") -> dict:
    ep = f"{leg}-{cat[:2]}-{i:03d}"
    return {
        "id": ep, "category": cat, "prompt": f"Give me the curl command for the relay ({i}).",
        "asked_on": "2026-08-20", "probe": "action",
        "accept": {"all_of": [[gold]], "none_of": list(anti or [])},
        "evidence": [f"{ep}-s1"],
        "source": {"session_file": f"sessions/{ep}.json", "facts": [f"Relay/Relay-state.md#s{i:03d}"]},
        "grounding": [{"entity": f"Relay{i}", "fact": f"Relay{i} listens on port {gold}."}],
    }


def _episode(qid: str, gold: str) -> dict:
    return {"episode": qid, "sessions": [{"sid": f"{qid}-s1", "date": "2026-08-10", "turns": [
        {"role": "user", "text": f"the relay moved to port {gold} today"},
        {"role": "assistant", "text": "noted"}]}]}


def make_set(tmp: Path, counts: dict[str, int], holdout: int = 2,
             generator: str = GENERATOR) -> Path:
    root = tmp / "v-test"
    for leg, spec in (("dev", counts), ("holdout", {"single_session": holdout})):
        (root / leg / "sessions").mkdir(parents=True)
        qs = []
        for cat, n in spec.items():
            for i in range(n):
                q = _q(i, cat, anti=["8181"], leg=leg)
                qs.append(q)
                (root / leg / q["source"]["session_file"]).write_text(json.dumps(_episode(q["id"], "8182")))
        (root / leg / "questions.yaml").write_text(yaml.safe_dump({"questions": qs}, sort_keys=False))
    M.write_manifest(root, version="vtest", generator_model=generator)
    return root


def _refreeze(root: Path) -> None:
    M.write_manifest(root, version="vtest", generator_model=GENERATOR)


def _edit_question(root: Path, fn) -> None:
    p = root / "dev" / "questions.yaml"
    doc = yaml.safe_load(p.read_text())
    fn(doc["questions"][0])
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    _refreeze(root)


# ── clause 1: the load contract ──────────────────────────────────────────────

def test_a_valid_frozen_set_loads_with_every_field(tmp_path):
    ms = M.load_set(make_set(tmp_path, {"single_session": 3, "temporal": 2}))
    assert len(ms.dev) == 5 and ms.holdout is None
    q = ms.dev[0]
    assert q.id and q.category in M.CATEGORIES and q.prompt and q.gold and q.source["session_file"]
    assert ms.generator_model == GENERATOR and ms.version == "vtest" and len(ms.set_sha) == 64


@pytest.mark.parametrize("field", list(M.REQUIRED_FIELDS))
def test_a_missing_field_is_a_load_error_not_a_skip(tmp_path, field):
    root = make_set(tmp_path, {"single_session": 3})
    _edit_question(root, lambda q: q.pop(field))
    with pytest.raises(M.SetLoadError, match=field):
        M.load_set(root)


def test_an_unresolvable_source_is_a_load_error(tmp_path):
    root = make_set(tmp_path, {"single_session": 3})
    _edit_question(root, lambda q: q["source"].__setitem__("session_file", "sessions/nope.json"))
    with pytest.raises(M.SetLoadError, match="does not resolve"):
        M.load_set(root)
    root2 = make_set(tmp_path / "b", {"single_session": 3})
    _edit_question(root2, lambda q: q["source"].__setitem__("session_file", "../../etc/passwd"))
    with pytest.raises(M.SetLoadError):
        M.load_set(root2)


def test_a_bad_category_or_an_evidence_sid_not_in_the_source_is_refused(tmp_path):
    root = make_set(tmp_path, {"single_session": 3})
    _edit_question(root, lambda q: q.__setitem__("category", "vibes"))
    with pytest.raises(M.SetLoadError, match="category"):
        M.load_set(root)
    root2 = make_set(tmp_path / "b", {"single_session": 3})
    _edit_question(root2, lambda q: q.__setitem__("evidence", ["ghost-s9"]))
    with pytest.raises(M.SetLoadError, match="evidence"):
        M.load_set(root2)


def test_a_file_edited_after_freezing_is_refused(tmp_path):
    root = make_set(tmp_path, {"single_session": 3})
    p = next((root / "dev" / "sessions").iterdir())
    p.write_text(p.read_text().replace("noted", "noted!"))
    with pytest.raises(M.SetLoadError, match="edited after"):
        M.load_set(root)


# ── clause 2: "use" is scored apart from retrieval ──────────────────────────

def _one(tmp_path) -> M.Question:
    return M.load_set(make_set(tmp_path, {"knowledge_update": 1}, holdout=0)).dev[0]


ACTS = "Here you go:\n\n    curl http://localhost:8182/health"
RESTATES_BUT_STALE = ("The relay's current port is 8182 (it was 8181 before).\n\n"
                      "    curl http://localhost:8181/health")


def test_an_answer_that_restates_the_value_but_acts_on_the_stale_one_is_not_correct(tmp_path):
    q = _one(tmp_path)
    good = M.judge_rules(q, ACTS)
    bad = M.judge_rules(q, RESTATES_BUT_STALE)
    assert good["verdict"] == "correct" and good["mentioned"]
    # retrieval-in-the-answer holds for both: the gold value IS mentioned
    assert bad["mentioned"] and bad["verdict"] == "mixed"
    # rules alone never call it correct ...
    assert M.settle(bad, None)["final"] != "correct"
    # ... and the judge's reading of what it ACTED on decides
    stale = M.settle(bad, {"label": "superseded"})
    assert stale["final"] == "stale" and stale["strict"] == "stale"
    assert M.settle(good, {"label": "superseded"})["final"] == "correct"  # rules-settled stays


def test_score_rows_keeps_mentioned_and_correct_as_separate_numbers(tmp_path):
    q = _one(tmp_path)
    fake = lambda *a, **k: None  # djev unreachable  # noqa: E731
    jobs = [{"id": q.id, "arm": "history", "answer": ACTS, "evidence_in_context": True},
            {"id": q.id, "arm": "prefetch", "answer": RESTATES_BUT_STALE, "evidence_in_context": True}]
    rows = M.score_rows(jobs, {q.id: q}, use_djev=True, djev_ask=fake)
    assert [r["mentioned"] for r in rows] == [True, True]
    assert [r["correct"] for r in rows] == [True, False]
    assert [r["evidence_in_context"] for r in rows] == [True, True]


# ── clause 3: the artifact ──────────────────────────────────────────────────

def _fake_complete(answer_for):
    async def complete(client, base_url, model, messages, seed, max_tokens):
        return {"answer": answer_for(messages), "finish": "stop", "completion_tokens": 5,
                "prompt_tokens": 50, "latency_s": 0.01}
    return complete


def _run(tmp_path, root, *extra, answer_for=lambda m: ACTS if "Conversation on" in m[0]["content"] else "no idea",
         djev_ask=None):
    out = tmp_path / "runs"
    return M.run(["--set", str(root), "--arms", "closed_book,history", "--judge", "rules",
                  "--label", "t", "--out-dir", str(out), *extra],
                 complete=_fake_complete(answer_for), primary=("http://x", "fake"),
                 djev_ask=djev_ask)


def test_one_artifact_with_set_sha_per_category_scores_ci_and_mde(tmp_path):
    root = make_set(tmp_path, {"single_session": 22, "temporal": 3})
    rep = _run(tmp_path, root)
    art = json.loads(Path(rep["_path"]).read_text())
    assert art["set"]["version"] == "vtest" and art["set"]["set_sha"] == M.load_set(root).set_sha
    ss = art["dev"]["history"]["single_session"]["correct"]
    assert ss["n"] == 22 and ss["rate"] == 1.0 and len(ss["ci"]) == 2 and ss["mde_80"] > 0
    # below the declared minimum: insufficient, not a number
    t = art["dev"]["history"]["temporal"]["correct"]
    assert t["insufficient"] is True and t["rate"] is None
    comp = next(c for c in art["dev_comparisons"] if c["metric"] == "correct")
    ss_c = comp["by_category"]["single_session"]
    # the paired interval comes from eval/stats.paired_bootstrap_ci
    import stats
    a = [0.0] * 22
    b = [1.0] * 22
    ref = stats.paired_bootstrap_ci(a, b)
    assert ss_c["ci"] == [round(ref["lo"], 4), round(ref["hi"], 4)] and ss_c["mde_80"] >= 0
    assert comp["by_category"]["temporal"]["insufficient"] is True
    assert len(list((tmp_path / "runs").glob("*.json"))) == 1


# ── clause 4: the holdout slice ─────────────────────────────────────────────

def test_the_tuning_view_excludes_the_holdout_and_never_opens_it(tmp_path):
    root = make_set(tmp_path, {"single_session": 3}, holdout=4)
    full = M.load_set(root, view="all")
    assert len(full.holdout) == 4 and full.reserved_ids == {q.id for q in full.holdout}
    tuning = M.load_set(root)
    assert tuning.holdout is None
    assert not ({q.id for q in tuning.dev} & tuning.reserved_ids)
    # provable: the tuning view loads with the holdout leg gone entirely
    shutil.rmtree(root / "holdout")
    assert {q.id for q in M.load_set(root).dev} == {q.id for q in tuning.dev}
    with pytest.raises(M.SetLoadError):
        M.load_set(root, view="all")


def test_a_reserved_id_in_the_dev_leg_is_refused(tmp_path):
    root = make_set(tmp_path, {"single_session": 3}, holdout=2)
    man = json.loads((root / "manifest.json").read_text())
    dev_id = yaml.safe_load((root / "dev" / "questions.yaml").read_text())["questions"][0]["id"]
    man["reserved_ids"] = sorted(man["reserved_ids"] + [dev_id])
    body = {k: man[k] for k in ("version", "generator_model", "reserved_ids", "files")}
    man["set_sha"] = M._digest(body)
    (root / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(M.SetLoadError, match="reserved"):
        M.load_set(root)


def test_a_holdout_run_is_its_own_arm_and_writes_no_holdout_rows(tmp_path):
    root = make_set(tmp_path, {"single_session": 21}, holdout=3)
    rep = _run(tmp_path, root, "--holdout")
    reserved = M.load_set(root, view="all").reserved_ids
    assert "holdout" in rep and rep["holdout"]["history"]["all"]["correct"]["n"] == 3
    assert not ({r["id"] for r in rep["rows"]} & reserved)
    assert reserved.isdisjoint(json.dumps(rep["dev"]).split('"'))
    plain = _run(tmp_path, root)
    assert "holdout" not in plain


# ── clause 5: self-judging is refused ───────────────────────────────────────

def test_the_generator_may_not_judge(tmp_path):
    with pytest.raises(M.SelfJudgeRefused):
        M.check_judge(GENERATOR, "Qwen3.8-Flash-Next")
    with pytest.raises(M.SelfJudgeRefused):
        M.check_judge("qwen3.8-flash-next", "Qwen3.8-Flash-Next-nvfp4")
    M.check_judge(GENERATOR, M.DJEV_JUDGE_MODEL)
    M.check_judge(GENERATOR, "rules")


def test_run_refuses_before_answering_when_the_judge_generated_the_set(tmp_path, monkeypatch):
    root = make_set(tmp_path, {"single_session": 3}, generator=M.DJEV_JUDGE_MODEL)
    called = []
    with pytest.raises(M.SelfJudgeRefused):
        M.run(["--set", str(root), "--arms", "closed_book", "--judge", "djev",
               "--out-dir", str(tmp_path / "o")],
              complete=_fake_complete(lambda m: called.append(1) or "x"), primary=("u", "m"))
    assert not called


def test_a_set_that_does_not_record_its_generator_will_not_load(tmp_path):
    root = make_set(tmp_path, {"single_session": 3}, generator="")
    with pytest.raises(M.SetLoadError, match="generating model"):
        M.load_set(root)


# ── the prefetch arm's splice keeps everything but the facts block ──────────

def test_splice_facts_replaces_only_the_facts_section():
    q = "what port"
    rendered = ("<context>\n<skill name=\"x\">body</skill>\n<facts>\n- [A] old\n</facts>\n"
                "<vault-context>\n- v\n</vault-context>\n</context>\n\n" + q)
    out = M.splice_facts(rendered, q, ["- [A] new"])
    assert "- [A] new" in out and "old" not in out and "<skill" in out and "- v" in out
    assert out.index("<facts>") < out.index("<vault-context>") and out.endswith(q)
    assert M.splice_facts(q, q, []) == q
    assert M.splice_facts(q, q, ["- [A] f"]).startswith("<context>\n<facts>")


# ── #1485: the vault_recall arms ────────────────────────────────────────────

def test_exported_sessions_look_like_chat_exports_and_are_found_by_id(tmp_path):
    root = make_set(tmp_path, {"multi_session": 2})
    ms = M.load_set(root)
    mapping = M.export_sessions(ms.dev, tmp_path / "exp")
    assert len(mapping) == 2 and not any("holdout" in s for s in mapping)
    rel = mapping[ms.dev[0].evidence[0]]
    text = (tmp_path / "exp" / rel).read_text()
    stem = Path(rel).stem
    assert rel.startswith("2026-08-10/20260810_") and len(stem.split("_")) == 3
    assert text.startswith(f"# {stem}\n# 2026-08-10T") and "\nuser: the relay moved" in text
    assert "\nlloyd: noted" in text
    q = ms.dev[0]
    assert M.evidence_hits(q, [f"sessions/{rel}", "knowledge/x.md"]) == (True, True)
    assert M.evidence_hits(q, ["knowledge/x.md"]) == (False, False)


def test_recall_arms_carry_the_documents_and_are_compared(tmp_path):
    root = make_set(tmp_path, {"multi_session": 22})
    seen = []

    def fake_recall(q):
        seen.append(q.id)
        docs_on = [{"path": "sessions/x.md", "snippet": "the relay moved to port 8182"}]
        return {"recall": M.render_recall(q.prompt, []),
                "recall_episodic": M.render_recall(q.prompt, docs_on),
                "recall_docs": {"recall": [], "recall_episodic": ["sessions/x.md"]},
                "recall_ms": {"recall": 1.0, "recall_episodic": 2.0}}
    out = tmp_path / "runs"
    rep = M.run(["--set", str(root), "--arms", "recall,recall_episodic", "--judge", "rules",
                 "--label", "r", "--out-dir", str(out)],
                complete=_fake_complete(lambda m: ACTS if "<vault_recall>" in m[-1]["content"] else "no idea"),
                primary=("http://x", "fake"), recall_fn=fake_recall)
    assert len(seen) == 22
    art = json.loads(Path(rep["_path"]).read_text())
    assert art["dev"]["recall_episodic"]["multi_session"]["correct"]["rate"] == 1.0
    assert art["dev"]["recall"]["multi_session"]["correct"]["rate"] == 0.0
    assert any(c["metric"] == "correct" and "recall_episodic" in json.dumps(c)
               for c in art["dev_comparisons"])
    assert art["recall"][seen[0]]["docs"]["recall_episodic"] == ["sessions/x.md"]


def test_render_recall_is_the_bare_question_without_documents():
    assert M.render_recall("q?", []) == "q?"
    out = M.render_recall("q?", [{"path": "sessions/a.md", "snippet": "x\n  y"}])
    assert out == "<vault_recall>\n- sessions/a.md: x y\n</vault_recall>\n\nq?"
