"""LloydMemEval's runner (#1480): the frozen set's load contract, the "use"
judgement kept apart from retrieval, the artifact, the holdout reserve and the
self-judging refusal. Hermetic: no engine, no djev, no facts store."""
from __future__ import annotations

import json
import os
import re
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


# ── #1516: the sleep_notes arm answers through the next-session channel ─────
#
# The item's measurement half is a transport comparison: the same facts
# `prefetch` renders inside the per-turn `<context>` block, delivered instead
# through the file-backed channel a nightly pass writes to. So the retrieval leg
# is planted here (`blocks_fn`) and the engine is faked — what is under test is
# that the arm really goes through `app.next_session_notes` and that the
# existing arm-vs-arm path emits its paired-bootstrap row per category.
# Whether the channel is worth keeping is a CI over a real GPU run, which is a
# human call (`--arms prefetch,sleep_notes`); nothing below presumes an answer.

def _facts_blocks(lines):
    def blocks(q):
        return {"prefetch": M.splice_facts(q.prompt, q.prompt, lines),
                "prefetch_rel": M.splice_facts(q.prompt, q.prompt, lines),
                "facts_conf": list(lines), "facts_rel": list(lines), "prefetch_ms": 1.0}
    return blocks


def test_the_sleep_notes_arm_answers_through_the_channel(tmp_path, monkeypatch):
    from app import next_session_notes as nsn

    store = tmp_path / "run-store.json"
    monkeypatch.setenv(nsn.STORE_PATH_ENV, str(store))
    root = make_set(tmp_path, {"multi_session": 22, "knowledge_update": 22})
    lines = ["- [A] Relay listens on port 8182 (Relay/Relay-state.md#s000)"]
    turns: list[str] = []

    def answer_for(messages):
        turn = messages[-1]["content"]
        turns.append(turn)
        return ACTS if "<next-session-notes>" in turn else "no idea"

    rep = M.run(["--set", str(root), "--arms", f"prefetch,{M.SLEEP_NOTES_ARM}",
                 "--judge", "rules", "--label", "s", "--out-dir", str(tmp_path / "runs")],
                complete=_fake_complete(answer_for), primary=("http://x", "fake"),
                blocks_fn=_facts_blocks(lines))
    art = json.loads(Path(rep["_path"]).read_text())

    assert M.SLEEP_NOTES_ARM == "sleep_notes" and M.SLEEP_NOTES_ARM in M.ARMS, (
        "the runner refuses `--arms sleep_notes`, so the comparison cannot be run")
    # Both transports reached the model, and the notes arm reached it through the
    # store: the note is the channel's own envelope, carrying the same fact line
    # the prefetch arm renders inside `<facts>`.
    assert any("<next-session-notes>" in t and "8182" in t for t in turns), turns[:1]
    assert any("<facts>" in t for t in turns), turns[:1]
    assert art["dev"]["sleep_notes"]["multi_session"]["correct"]["rate"] == 1.0
    assert art["dev"]["prefetch"]["multi_session"]["correct"]["rate"] == 0.0
    # The producer named a store, so the run used it, and it drained what it
    # wrote: a benchmark that left a note standing would hand it to a real turn.
    assert art["sleep_notes_store"] == str(store)
    assert json.loads(store.read_text())["notes"] == []


def test_the_channel_row_is_paired_against_prefetch_per_category(tmp_path, monkeypatch):
    """The ship/no-ship number the item asks for, in the shape that decides it:
    one paired-bootstrap row for `sleep_notes` vs `prefetch` in every category,
    `multi_session` and `knowledge_update` among them — the two the item names.
    """
    from app import next_session_notes as nsn

    monkeypatch.setenv(nsn.STORE_PATH_ENV, str(tmp_path / "store.json"))
    root = make_set(tmp_path, {"multi_session": 22, "knowledge_update": 22})
    rep = M.run(["--set", str(root), "--arms", f"prefetch,{M.SLEEP_NOTES_ARM}",
                 "--judge", "rules", "--label", "s", "--out-dir", str(tmp_path / "runs")],
                complete=_fake_complete(lambda m: "no idea"),
                primary=("http://x", "fake"),
                blocks_fn=_facts_blocks(["- [A] Relay listens on port 8182"]))
    art = json.loads(Path(rep["_path"]).read_text())
    rows = [c for c in art["dev_comparisons"]
            if c["a"] == M.SLEEP_NOTES_ARM and c["b"] == "prefetch"]

    assert {c["metric"] for c in rows} == {"correct_strict", "correct", "evidence_in_context"}
    correct = next(c for c in rows if c["metric"] == "correct")["by_category"]
    for cat in ("multi_session", "knowledge_update"):
        cell = correct[cat]
        assert cell["n"] == 22, (cat, cell)
        assert "ci" in cell and len(cell["ci"]) == 2, (cat, cell)
        assert cell["a"] == 0.0 and cell["b"] == 0.0, (cat, cell)


def test_a_run_names_the_channel_store_it_used_inside_its_own_out_dir(tmp_path, monkeypatch):
    """A run that does not name a store still must not write the live channel:
    its notes go to a file in its own `--out-dir`, and the artifact says which.
    """
    from app import next_session_notes as nsn

    monkeypatch.delenv(nsn.STORE_PATH_ENV, raising=False)
    root = make_set(tmp_path, {"multi_session": 1})
    out = tmp_path / "runs"
    rep = M.run(["--set", str(root), "--arms", M.SLEEP_NOTES_ARM, "--judge", "rules",
                 "--label", "s", "--out-dir", str(out)],
                complete=_fake_complete(lambda m: "no idea"), primary=("http://x", "fake"),
                blocks_fn=_facts_blocks(["- [A] Relay listens on port 8182"]))
    art = json.loads(Path(rep["_path"]).read_text())

    used = Path(art["sleep_notes_store"])
    assert used.parent == out, art["sleep_notes_store"]
    # Not `nsn.store_path()`: the run set the override, so that call reports the
    # run's own file back and the check could never fail. The comparison is with
    # the name the channel carries when nothing overrides it — the file a real
    # morning turn reads.
    assert used != nsn.DATA_ROOT / nsn.FILE_NAME, "the run pointed the arm at the live channel"

# ── #1631: the committed write-up of the channel (eval/measurements/sleep-notes-*.md) ──
#
# #1516 left a deployment decision owed — keep the next-session channel or delete
# it — and the decision was to be made on a paired-bootstrap number that did not
# exist. These nodes read the COMMITTED report rather than a synthetic run, so a
# write-up that dropped a category, printed a CI it never computed, or quietly
# pointed the arm at the live channel fails here. Nothing is taken on trust: `n`
# is checked against the frozen dev slice, Δ against the two rates printed beside
# it, the interval against the Δ it sits over, and the clears-0 verdict against
# that interval.

MEASUREMENTS = ROOT / "eval" / "measurements"
#: Each `correct_strict` comparison in the report is a `### ` section at exactly
#: this heading, so a reader gets the pair it asked for and not whichever
#: `sleep_notes` table happens to come first.
CI_SECTION = {(other, metric): f"### {M.SLEEP_NOTES_ARM} vs {other} — {metric} (dev slice)"
              for other in ("prefetch", "prefetch_rel")
              for metric in ("correct_strict", "evidence_in_context")}
#: Fixed column order of every comparison table in the report.
CI_COLUMNS = ("category", "n", "a", "b", "diff", "ci", "clears")
#: Δ, both arm rates and both CI bounds are printed to three decimals, so two
#: independently rounded rates can differ from a Δ computed on the unrounded
#: means by a thousandth. A transcription slip is bigger than that.
ROUND = 0.0015


def _report() -> tuple[Path, str]:
    """The newest committed sleep-notes write-up, globbed rather than named
    because the file carries the date its run answered on. No file is the
    failure it looks like: the measurement #1516 owed has not been written up."""
    files = sorted(MEASUREMENTS.glob("sleep-notes-*.md"))
    assert files, (f"nothing matches {MEASUREMENTS}/sleep-notes-*.md: the "
                   f"paired-bootstrap run that decides the #1516 channel has not "
                   f"been written up")
    path = files[-1]
    return path, path.read_text(encoding="utf-8")


def _section(md: str, other_arm: str, metric: str) -> str:
    """One comparison's slice of the report: its heading to the next heading."""
    heading = CI_SECTION[(other_arm, metric)]
    if heading not in md:
        raise AssertionError(f"the report has no section headed {heading!r}")
    return md.split(heading, 1)[1].split("\n### ")[0]


def _check_report_section(md: str, other_arm: str, metric: str,
                          cats=("multi_session", "knowledge_update")) -> dict[str, dict]:
    """Both categories the item names, in one report section, each surviving
    `_check_ci_row`. Returns the parsed table so a caller can say more about it."""
    body = _section(md, other_arm, metric)
    table, verdicts, counts = _ci_table(body), _verdict_lines(body), _dev_counts()
    for cat in cats:
        assert cat in table, (CI_SECTION[(other_arm, metric)], cat, sorted(table))
        assert cat in verdicts, (CI_SECTION[(other_arm, metric)], cat, sorted(verdicts))
        _check_ci_row(cat, table[cat], verdicts[cat], counts)
    return table


def _ci_table(body: str) -> dict[str, dict]:
    """{category: row} of one comparison table (see `CI_COLUMNS`)."""
    rows: dict[str, dict] = {}
    for line in body.split("\n"):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != len(CI_COLUMNS) or cells[0] in ("category", "---", ""):
            continue
        ci = re.fullmatch(r"\[(-?\d+\.\d+), (-?\d+\.\d+)\]", cells[5])
        assert ci, f"unparseable 95% CI {cells[5]!r} in the {cells[0]!r} row"
        rows[cells[0]] = {"n": int(cells[1]), "a": float(cells[2]), "b": float(cells[3]),
                          "diff": float(cells[4]), "ci": (float(ci.group(1)), float(ci.group(2))),
                          "clears": cells[6]}
    return rows


def _verdict_lines(body: str) -> dict[str, tuple]:
    """The one-line clears-0 statement per category under one comparison."""
    out: dict[str, tuple] = {}
    for line in body.split("\n"):
        m = re.fullmatch(r"- (\w+) \(n=(\d+)\): CI \[(-?\d+\.\d+), (-?\d+\.\d+)\] "
                         r"(clears 0|does not clear 0)", line.strip())
        if m:
            out[m.group(1)] = (int(m.group(2)), (float(m.group(3)), float(m.group(4))), m.group(5))
    return out


def _dev_counts() -> dict[str, int]:
    """Per-category n of the frozen dev slice, off the set the runner loads."""
    counts: dict[str, int] = {}
    for q in M.load_set(M.SET_ROOT / M.DEFAULT_VERSION, view="tuning").dev:
        counts[q.category] = counts.get(q.category, 0) + 1
    return counts


def _check_ci_row(cat: str, row: dict, verdict: tuple, counts: dict[str, int]) -> None:
    """What a printed row has to survive to be a measurement at all: enough
    questions to be a number rather than `insufficient`, an n that IS the frozen
    slice's, a Δ that is the difference of the two rates it prints, an interval
    that covers that Δ, and a verdict the interval actually supports."""
    assert row["n"] >= M.MIN_CATEGORY_N, (cat, row)
    assert row["n"] == counts[cat], (cat, row["n"], counts[cat])
    assert abs(row["diff"] - (row["b"] - row["a"])) <= ROUND, (cat, row)
    assert row["ci"][0] - ROUND <= row["diff"] <= row["ci"][1] + ROUND, (cat, row)
    assert verdict[0] == row["n"] and verdict[1] == row["ci"], (cat, row, verdict)
    assert verdict[2] == ("clears 0" if (row["ci"][0] > 0 or row["ci"][1] < 0)
                          else "does not clear 0"), (cat, row, verdict)


def test_the_committed_report_carries_the_ship_no_ship_ci_for_both_named_categories():
    """Clause 1. The dev-slice paired-bootstrap CI of the channel against the
    arm its own docstring names, in `multi_session` and `knowledge_update`, each
    with its n and a one-line statement of whether the interval clears 0. Both
    legs the report prints are read — `correct_strict` and the
    `evidence_in_context` retrieval leg — so a number that appears only in prose
    cannot pass for a measured one."""
    path, md = _report()
    for metric in ("correct_strict", "evidence_in_context"):
        table = _check_report_section(md, "prefetch", metric)
        assert table["multi_session"]["n"] == _dev_counts()["multi_session"], (path.name, metric)


def test_the_committed_report_names_the_command_and_a_store_that_is_not_the_live_channel():
    """Clause 2. The run has to be re-runnable as written — under the SHARED
    primary lock, with its own label — and the file its notes went through has
    to be inside that run's out dir. A report whose arm wrote
    `~/lloyd-data/next-session-notes.json` handed a synthetic "what to know
    today" note to the next real chat turn."""
    path, md = _report()
    block = re.search(r"```bash\n(.*?)```", md, re.S)
    assert block, f"{path.name} records no reproduction command"
    cmd = block.group(1)
    assert "flock -s ~/.local/state/lloyd-automod/primary.lock" in cmd, cmd
    assert "eval/run_memory_eval.py run" in cmd, cmd
    assert re.search(r"--label\s+\S*sleep-notes-\d{4}-\d{2}-\d{2}", cmd), cmd
    arms = set(re.search(r"--arms\s+(\S+)", cmd).group(1).split(","))
    assert arms == {"prefetch", "prefetch_rel", M.SLEEP_NOTES_ARM}, cmd

    from app import next_session_notes as nsn
    out = Path(os.path.expanduser(re.search(r"--out-dir\s+(\S+)", cmd).group(1)))
    stated = re.search(r"^sleep_notes_store:\s*(\S+)$", md, re.M)
    assert stated, f"{path.name} never says which file the arm's notes went through"
    used = Path(stated.group(1))
    assert used.parent == out, (str(used), str(out))
    assert used.name.startswith("sleep-notes-store-"), used
    assert used != nsn.DATA_ROOT / nsn.FILE_NAME, "the run pointed the arm at the live channel"


def test_the_channel_is_priced_against_the_ranking_production_ships(tmp_path, monkeypatch):
    """Clause 3. A CI against `prefetch` alone cannot justify wiring a producer,
    because live turns order `<facts>` by relevance: `prefetch` is an arm no
    running system renders. So the channel owes a second row against
    `prefetch_rel`, which needs the pair list to name it, a three-arm run to
    emit it, and the committed report to print it."""
    assert (M.SLEEP_NOTES_ARM, "prefetch_rel") in M.DEV_COMPARISON_PAIRS

    from app import next_session_notes as nsn
    monkeypatch.setenv(nsn.STORE_PATH_ENV, str(tmp_path / "store.json"))
    root = make_set(tmp_path, {"multi_session": 22, "knowledge_update": 22})
    rep = M.run(["--set", str(root), "--arms", f"prefetch,prefetch_rel,{M.SLEEP_NOTES_ARM}",
                 "--judge", "rules", "--label", "s", "--out-dir", str(tmp_path / "runs")],
                complete=_fake_complete(lambda m: "no idea"),
                primary=("http://x", "fake"),
                blocks_fn=_facts_blocks(["- [A] Relay listens on port 8182"]))
    art = json.loads(Path(rep["_path"]).read_text())
    pairs = {(c["a"], c["b"]) for c in art["dev_comparisons"]}
    assert {(M.SLEEP_NOTES_ARM, "prefetch"), (M.SLEEP_NOTES_ARM, "prefetch_rel")} <= pairs, pairs
    strict = [c for c in art["dev_comparisons"]
              if (c["a"], c["b"], c["metric"]) == (M.SLEEP_NOTES_ARM, "prefetch_rel", "correct_strict")]
    assert len(strict) == 1 and strict[0]["by_category"]["knowledge_update"]["n"] == 22, strict

    path, md = _report()
    for metric in ("correct_strict", "evidence_in_context"):
        _check_report_section(md, "prefetch_rel", metric)
    answered = re.search(r"^\| arms answered \| (.+) \|$", md, re.M)
    assert answered, "the report never records which arms the run answered"
    assert {"prefetch", "prefetch_rel", M.SLEEP_NOTES_ARM} <= set(answered.group(1).split(", ")), (
        "the report's row against the shipped ranking is not from a run that "
        f"answered that arm: {answered.group(1)!r}")


# ── #1556: prefetch_rawspan — the facts prefetch_rel renders, as their own source ──
#
# The arm holds retrieval fixed and swaps only the rendering, so every test below
# drives ONE selection and asks what the model was shown. The gold value in the set
# `make_set` builds is 8182, which is why the planted source document carries it and
# the distilled bullet does not: that is what makes the pair's sign observable.

#: Three blocks, one of which a fact about the CTO can be found in. The middle one
#: is the only place 8182 appears, and the first is the head of the file — so a
#: windowing rule that just took the head would render a block with no gold value
#: in it and every pair test below would go red.
RAWSPAN_SOURCE = """# Northwind notes

The depot at Bilbao ships pallets on Thursdays from the warehouse.

The CTO of Northwind Traders is Dana Whitfield, and the relay she moved listens on port 8182.

Lunch is at noon.
"""

BILBAO_SOURCE = """# Bilbao depot

The Bilbao depot re-orders grommets every week under a standing order.
"""

#: What `app.prefetch.prefetch_context` is stood in for: shaped like the real
#: envelope so `splice_facts` has a `<facts>` block to replace and the arms can be
#: compared on bytes rather than on a fixture that flatters one of them.
RAWSPAN_ENVELOPE = ("<skills>\n</skills>\n<facts>\nplaceholder\n</facts>\n"
                    "<recent_turns>\n</recent_turns>")


def _record(entity, fact, source_doc, conf=0.9):
    """One selected fact exactly as `app.prefetch._search_fact_records` hands it
    over: the fact's stored fields plus the bullet the distilled arm renders."""
    return {"entity": entity, "fact": fact, "confidence": conf, "source_doc": source_doc,
            "line": f"- [{entity}] {fact} (confidence: {conf})"}


def _rawspan_records(fact="The current CTO of Northwind Traders is Dana Whitfield"):
    return [_record("Northwind Traders", fact, "notes/northwind.md"),
            _record("Bilbao Depot", "The Bilbao depot re-orders grommets every week",
                    "notes/bilbao.md", conf=0.7)]


def _plant_sources(tmp_path, monkeypatch):
    """Plant the source documents under a vault root the runner will look in, so
    span resolution is measured against a corpus this test owns."""
    (tmp_path / "notes").mkdir(parents=True, exist_ok=True)
    (tmp_path / "notes" / "northwind.md").write_text(RAWSPAN_SOURCE, encoding="utf-8")
    (tmp_path / "notes" / "bilbao.md").write_text(BILBAO_SOURCE, encoding="utf-8")
    monkeypatch.setenv("LLOYD_VAULT_ROOT", str(tmp_path))
    return tmp_path


def _isolate_store(tmp_path, monkeypatch):
    """Keep the runner off the live fact store: it sets those two variables itself,
    so set them first and `monkeypatch` puts them back after."""
    monkeypatch.setenv("LLOYD_FACTS_ROOT", str(tmp_path / "facts"))
    monkeypatch.setenv("LLOYD_KG_DB", str(tmp_path / "kg.sqlite"))


def _a_question(tmp_path) -> M.Question:
    """One real `Question` off a planted set, for the tests that drive a block
    builder rather than a whole command."""
    return M.load_set(make_set(tmp_path, {"knowledge_update": 1}, holdout=0)).dev[0]


def test_rawspan_arm_passes_arms_validation_on_both_commands(tmp_path, monkeypatch):
    """Clause 1 (#1556): `prefetch_rawspan` is in `ARMS` and is valid to BOTH
    `--arms` validators, so the pair runs on the no-model `prefetch-retrieval`
    command — which returns its scores with no completion injected, no engine
    named and no answer generated anywhere."""
    assert "prefetch_rawspan" in M.ARMS
    assert M.parse_arms("prefetch_rel,prefetch_rawspan") == ["prefetch_rel", "prefetch_rawspan"]
    with pytest.raises(SystemExit):
        M.parse_arms("prefetch_rawspan,nope")

    root = make_set(tmp_path, {"knowledge_update": 2})
    _plant_sources(tmp_path, monkeypatch)
    _isolate_store(tmp_path, monkeypatch)
    argv = ["--set", str(root), "--corpus", str(tmp_path / "corpus"),
            "--arms", "prefetch_rel,prefetch_rawspan", "--out-dir", str(tmp_path / "runs")]
    out = M.prefetch_retrieval(argv, select=lambda query, rank: _rawspan_records())
    assert out["arms"] == ["prefetch_rel", "prefetch_rawspan"]
    assert out["n"] == 2 and out["by_category"]["all"]["n"] == 2
    assert out["by_category"]["all"]["rawspan"] is not None

    # An arm valid to one command and refused by the other would leave the pair
    # runnable in the expensive half only — backwards for an arm whose retrieval
    # reading is meant to cost no model call.
    with pytest.raises(SystemExit):
        M.prefetch_retrieval(argv[:-2] + ["--arms", "prefetch_rawspan,bogus",
                                          "--out-dir", str(tmp_path / "runs")],
                             select=lambda query, rank: _rawspan_records())
    with pytest.raises(SystemExit):
        M.run(["--set", str(root), "--arms", "prefetch_rawspan,bogus", "--out-dir",
               str(tmp_path / "runs")])
    rep = M.run(["--set", str(root), "--arms", "prefetch_rel,prefetch_rawspan", "--judge",
                 "rules", "--label", "s", "--out-dir", str(tmp_path / "runs")],
                complete=_fake_complete(lambda m: "the relay listens on port 8182"),
                primary=("http://x", "fake"),
                blocks_fn=lambda q: {arm: M.splice_facts(q.prompt, q.prompt,
                                                         ["- [Northwind Traders] relay port 8182"])
                                     for arm in ("prefetch_rel", "prefetch_rawspan")})
    assert set(rep["dev"]) == {"prefetch_rel", "prefetch_rawspan"}


def test_rawspan_renders_the_same_selection_as_prefetch_rel(tmp_path, monkeypatch):
    """Clause 2 (#1556): the two arms take their facts from ONE shared selection
    call that hands back the fact RECORDS, so the entities, their order and the
    count are equal by construction and not by two lookups happening to agree."""
    import app.prefetch as pf
    _plant_sources(tmp_path, monkeypatch)
    calls: list = []

    def fake_records(query, rank=None):
        calls.append(rank)
        recs = _rawspan_records()
        return list(reversed(recs)) if rank == "confidence" else recs

    monkeypatch.setattr(pf, "_search_fact_records", fake_records)
    monkeypatch.setattr(pf, "prefetch_context", lambda text: RAWSPAN_ENVELOPE)
    blocks = M.prefetch_blocks(_a_question(tmp_path))

    # Two calls, one per ordering, and no third re-deriving a selection for the
    # rawspan arm: it renders the relevance records themselves.
    assert calls == ["confidence", "relevance"]
    ents = lambda lines: [ln.split("]")[0].removeprefix("- [") for ln in lines]
    assert ents(blocks["facts_rawspan"]) == ents(blocks["facts_rel"])
    assert len(blocks["facts_rawspan"]) == len(blocks["facts_rel"]) == 2
    assert M.PREFETCH_RAWSPAN_ARM in blocks
    assert blocks[M.PREFETCH_RAWSPAN_ARM] != blocks["prefetch_rel"], (
        "the two arms rendered identically, so nothing is being compared")

    # And the shared selection is the production one: the bullets `_search_facts`
    # renders are the records' own `line` field in the records' own order, so the
    # arm's records and the shipped arm's lines cannot drift apart. Selection caps
    # and the empty-fact rule are the shipped arm's too, not a second copy.
    monkeypatch.undo()  # the fake selection above has to stop standing in for the real one
    monkeypatch.setattr(pf, "_extract_entities_from_query", lambda q: [("Northwind Traders", 6.5)])
    monkeypatch.setattr(pf, "_get_facts_sync", lambda entity, *a, **k: {
        "facts": [_record(entity, f"Northwind Traders note {i}", "notes/northwind.md",
                          conf=0.5 + i / 10) for i in range(5)] + [{"fact": "   "}]})
    for mode in ("confidence", "relevance"):
        recs = pf._search_fact_records("Northwind Traders update", rank=mode)
        assert [r["line"] for r in recs] == pf._search_facts("Northwind Traders update", rank=mode)
        assert len(recs) == pf.FACT_MAX_PER_ENTITY, "selection caps are not the shipped arm's"
        assert all({"entity", "fact", "confidence", "source_doc", "line"} <= set(r) for r in recs)
        assert all(r["source_doc"] == "notes/northwind.md" for r in recs), (
            "the record lost the field the rawspan arm renders from")


def test_rawspan_renders_the_source_text_and_one_char_budget_bounds_both_arms(tmp_path,
                                                                             monkeypatch):
    """Clause 3 (#1556): the line carries text drawn from the fact's `source_doc`
    and not the distilled bullet, and the one named char budget caps the facts
    block of BOTH arms."""
    import app.prefetch as pf
    roots = _plant_sources(tmp_path, monkeypatch)
    lines, counts = M.render_rawspan_lines(_rawspan_records(), roots=[roots])
    assert counts["n_rendered"] == 2 and counts["n_unresolved_source"] == 0
    northwind = lines[0]
    # Words that exist only in the document, in the block the fact's own terms
    # point at — not the file's head, and not the distilled bullet's wording.
    assert "relay she moved listens on port 8182" in northwind
    assert "ships pallets" not in northwind and "Lunch is at noon" not in northwind
    assert "The current CTO of Northwind Traders is Dana Whitfield" not in northwind
    assert "(confidence:" not in northwind
    assert "notes/northwind.md" in northwind, "the line did not name the document it drew from"

    # One budget, both arms: shrink the named constant and both renderings obey
    # it, each reporting how much of its own block the cap dropped.
    monkeypatch.setattr(M, "FACTS_RENDER_CHAR_BUDGET", 120)
    monkeypatch.setattr(pf, "prefetch_context", lambda text: RAWSPAN_ENVELOPE)
    monkeypatch.setattr(pf, "_search_fact_records", lambda query, rank=None: _rawspan_records())
    blocks = M.prefetch_blocks(_a_question(tmp_path))
    assert len("\n".join(blocks["facts_rel"])) <= 120
    assert len("\n".join(blocks["facts_rawspan"])) <= 120
    assert blocks["facts_rel_counts"]["n_budget_cut"] >= 1
    assert blocks["rawspan_counts"]["n_budget_cut"] >= 1
    assert blocks["prefetch_rel"].count("<facts>") == 1
    assert blocks[M.PREFETCH_RAWSPAN_ARM].count("<facts>") == 1


def test_rawspan_counts_unresolved_sources_and_unwindowed_facts(tmp_path, monkeypatch):
    """Clause 4 (#1556): a fact that cannot be rendered is counted and not dropped,
    and the counts reach the run artifact beside the arm's scores — including the
    run in which nothing renders at all."""
    roots = _plant_sources(tmp_path, monkeypatch)
    _isolate_store(tmp_path, monkeypatch)
    recs = _rawspan_records() + [
        _record("Nowhere Co", "The managing director of Nowhere Co is Wei Chen",
                "notes/never-written.md"),                # source_doc names a file that isn't there
        _record("Quokka Trust", "The quorum of the Quokka Trust is 1874 members",
                "notes/northwind.md"),                    # resolves, but no term of the fact is in it
    ]
    lines, counts = M.render_rawspan_lines(recs, roots=[roots])
    assert counts == {"n_selected": 4, "n_unresolved_source": 1, "n_no_span": 1,
                      "n_budget_cut": 0, "n_rendered": 2}
    assert len(lines) == 2

    # Nothing renders at all: the block is empty and the counts are still there,
    # all five of them. "The arm rendered nothing" has to read as a result about
    # the corpus, and an absent block would read as a run that never happened.
    _, empty = M.render_rawspan_lines([recs[2]], roots=[roots])
    assert empty["n_rendered"] == 0 and empty["n_unresolved_source"] == 1
    assert set(empty) == set(M.RAWSPAN_COUNT_KEYS)

    # The no-model command carries the totals beside the rates, question by
    # question and for the leg: n_selected 4 × 2 dev questions, half of them
    # unreachable, none of it silent.
    root = make_set(tmp_path, {"knowledge_update": 2})
    out = M.prefetch_retrieval(["--set", str(root), "--corpus", str(tmp_path / "corpus"),
                               "--arms", "prefetch_rel,prefetch_rawspan", "--out-dir", str(tmp_path / "runs")],
                               select=lambda query, rank: recs)
    assert out["rawspan_counts"] == {"n_selected": 8, "n_unresolved_source": 2, "n_no_span": 2,
                                     "n_budget_cut": 0, "n_rendered": 4}
    assert out["by_category"]["all"]["rawspan_counts"]["n_unresolved_source"] == 2
    # The rate says what the block held; the counts beside it say how much of the
    # selection that block was. Half of this one was unreachable and the arm still
    # surfaced the value — which is exactly the reading a rate alone hides.
    assert out["by_category"]["all"]["rawspan"] == 1.0

    # On disk, not only on stdout: the file the next reader quotes has to carry
    # the counts beside the rates it is quoting.
    art = json.loads(Path(out["_path"]).read_text())
    assert art["rawspan_counts"] == out["rawspan_counts"]
    assert art["by_category"]["all"]["rawspan_counts"] == \
        out["by_category"]["all"]["rawspan_counts"]
    assert art["comparison"]["by_category"]["all"]["n"] == 2

    # And on a fully-resolving run the same keys are present at zero — a clean
    # denominator has to be readable as clean, not as a block that went missing.
    clean = M.prefetch_retrieval(["--set", str(root), "--corpus", str(tmp_path / "corpus"),
                                  "--arms", "prefetch_rel,prefetch_rawspan",
                                  "--out-dir", str(tmp_path / "runs"), "--label", "clean"],
                                 select=lambda query, rank: _rawspan_records())
    clean_art = json.loads(Path(clean["_path"]).read_text())
    assert clean_art["rawspan_counts"] == {"n_selected": 4, "n_unresolved_source": 0,
                                           "n_no_span": 0, "n_budget_cut": 0, "n_rendered": 4}
    assert clean_art["char_budget"] == M.FACTS_RENDER_CHAR_BUDGET

    # And the modelled run's artifact reports them beside the scores.
    per_q = {"n_selected": 4, "n_unresolved_source": 1, "n_no_span": 1,
             "n_budget_cut": 0, "n_rendered": 2}
    rep = M.run(["--set", str(root), "--arms", "prefetch_rel,prefetch_rawspan", "--judge",
                 "rules", "--label", "rawspan-counts", "--out-dir", str(tmp_path / "runs")],
                complete=_fake_complete(lambda m: "the relay listens on port 8182"),
                primary=("http://x", "fake"),
                blocks_fn=lambda q: {"prefetch_rel": "<ctx/>", "prefetch_rawspan": "<ctx/>",
                                     "rawspan_counts": dict(per_q)})
    reported = json.loads(Path(rep["_path"]).read_text())["prefetch_rawspan"]
    assert reported["counts"] == {"n_selected": 8, "n_unresolved_source": 2, "n_no_span": 2,
                                  "n_budget_cut": 0, "n_rendered": 4}
    assert set(reported["counts"]) == set(M.RAWSPAN_COUNT_KEYS)
    assert reported["char_budget"] == M.FACTS_RENDER_CHAR_BUDGET
    assert reported["window_chars"] == M.rawspan_window_chars()


def test_rawspan_pair_is_compared_with_a_ci_and_no_expected_direction(tmp_path, monkeypatch):
    """Clause 5 (#1556): the artifact carries the paired `prefetch_rel` vs
    `prefetch_rawspan` comparison with its bootstrap interval on the no-model path,
    and it reports whichever way the delta lands."""
    roots = _plant_sources(tmp_path, monkeypatch)
    _isolate_store(tmp_path, monkeypatch)
    root = make_set(tmp_path, {"knowledge_update": 2})
    argv = ["--set", str(root), "--corpus", str(tmp_path / "corpus"),
            "--arms", "prefetch_rel,prefetch_rawspan", "--out-dir", str(tmp_path / "runs")]

    # The source text holds the gold value and the distilled bullet does not.
    src_first = M.prefetch_retrieval(argv, select=lambda query, rank: _rawspan_records())
    comp = src_first["comparison"]
    assert (comp["a"], comp["b"]) == ("prefetch_rel", "prefetch_rawspan")
    assert comp["metric"] == "gold_in_block"
    first = comp["by_category"]["all"]
    assert first["n"] == 2 and first["a"] == 0.0 and first["b"] == 1.0
    assert first["diff"] == 1.0 and first["ci"][0] <= first["diff"] <= first["ci"][1]

    # The other way round: the distilled bullet holds the value and the windowed
    # source no longer does. Same artifact, same keys, sign flipped. A regression
    # that suppressed, clamped or special-cased a negative delta fails here.
    (roots / "notes" / "northwind.md").write_text(
        RAWSPAN_SOURCE.replace("the relay she moved listens on port 8182",
                               "the relay she moved listens on a port nobody remembers"),
        encoding="utf-8")
    rev = M.prefetch_retrieval(
        argv, select=lambda query, rank: _rawspan_records(
            "The CTO of Northwind Traders is Dana Whitfield and the relay listens on port 8182"))
    second = rev["comparison"]["by_category"]["all"]
    assert set(rev["comparison"]) == set(comp) and set(second) == set(first)
    assert second["a"] == 1.0 and second["b"] == 0.0 and second["diff"] == -1.0
    assert second["ci"][0] <= second["diff"] <= second["ci"][1]

    # The modelled run compares the same pair, and the pair is named in one place.
    rep = M.run(["--set", str(root), "--arms", "prefetch_rel,prefetch_rawspan", "--judge",
                 "rules", "--label", "rawspan-pair", "--out-dir", str(tmp_path / "runs")],
                complete=_fake_complete(lambda m: "the relay listens on port 8182"),
                primary=("http://x", "fake"),
                blocks_fn=lambda q: {arm: M.splice_facts(q.prompt, q.prompt,
                                                         ["- [Northwind Traders] relay port 8182"])
                                     for arm in ("prefetch_rel", "prefetch_rawspan")})
    art = json.loads(Path(rep["_path"]).read_text())
    assert list(M.RENDER_PAIR) == ["prefetch_rel", "prefetch_rawspan"]
    assert {(c["a"], c["b"]) for c in art["dev_comparisons"]} == {M.RENDER_PAIR}
    pair = next(c for c in art["dev_comparisons"] if c["metric"] == "correct_strict")
    assert pair["by_category"]["knowledge_update"]["n"] == 2
