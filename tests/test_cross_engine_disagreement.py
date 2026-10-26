"""`eval/cross_engine_disagreement.py` — the #2639 localization instrument.

The item is a measurement, not a feature: does a second engine from a different family
localize verdict error, or do the two of them agree on being wrong? What can be pinned
offline is the instrument's honesty, which is the part that decides whether the
number it prints means anything:

  * a row records gold plus BOTH argmaxes, and an engine that returned no usable argmax
    ABSTAINS — it is never counted toward agreement, because an engine that did not
    answer cannot agree with anything (`argmax_verdict`: no valid draw, or a tie);
  * every published rate carries a Wilson CI, `both_wrong_and_agreeing` is its own
    column rather than a clause inside "agreement", and every rate sits beside its
    abstention share and over a denominator that excludes abstentions — the nightly
    labeler offers 44 of 94 entity labels, so a rate computed over all rows is not a
    conservative reading, it is a different quantity;
  * the gold is reused, never re-authored: the retrieval arm asks nothing outside the
    rows the shipped djev artifact reports as offered, and the durable-write arm joins
    the corpus ids to the 50 `primary` rows #2254 already ran, aborting if the prompt
    bytes no longer match the recorded sha256;
  * every row is asked 5 times per engine and reported as a per-case pass rate, and a
    thin cell prints `UNMEASURABLE` instead of a rate;
  * `POSITIVE` requires all three of the item's thresholds on a measurable surface,
    and the run writes its two files under `--out` and nowhere else.

Nothing here calls a live engine. Each node injects a transport — a callable the
module's only socket call goes through — so the routing is asserted, including that
the retired llama.cpp secondary at :8091 receives no request. #580's original judging
ran there; #2254's owed-check re-ran the same prompt on `primary` at :8096, and a run
that quietly resumed asking :8091 would record abstentions and read as an engine that
disagrees with everything.
"""

from __future__ import annotations

import io
import json
import re
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import cross_engine_disagreement as X  # noqa: E402

#: The pair as the run resolves it. The real resolver reads `config.yaml` and each
#: endpoint's `/v1/models`; a fixture with the same shape keeps the routing assertions
#: about URLs rather than about a served name that only exists while the GPU is up.
ENGINES = {
    "primary": {"url": "http://127.0.0.1:8096/v1/chat/completions", "model": "primary"},
    "djev": {"url": "http://127.0.0.1:8010/v1/chat/completions",
             "model": "DiffusionGemma-26B-A4B-NVFP4"},
}
PRIMARY = ENGINES["primary"]["url"]
DJEV = ENGINES["djev"]["url"]

#: A runtime store, not a checkout file: the shipped djev labeling artifact and the
#: #2254 owed-check rows. Present on this box; a bench tree without them skips rather
#: than inventing a stand-in, because a fabricated artifact would test the reader and
#: not the reader's claim about the real one.
HAVE_STORE = X.DEFAULT_RETRIEVAL_ARTIFACT.exists() and X.DEFAULT_ARMS_FINAL.exists()
needs_store = pytest.mark.skipif(
    not HAVE_STORE,
    reason=f"needs {X.DEFAULT_RETRIEVAL_ARTIFACT} and {X.DEFAULT_ARMS_FINAL}")


class Recorder:
    """A transport that answers by prompt, in draw order, and records every URL.

    `plan` maps a prompt to the replies its draws get IN ORDER (a short sequence
    repeats its last entry), so a node states a row's whole draw history and the
    pass-rate and tie claims below are checked against exactly that history rather
    than against a cycling guess. `engine_plan` overrides per URL, which is how a
    node writes "primary says yes, djev says no" — the only pair shape that
    disagreement can come from.
    """

    def __init__(self, plan: dict[str, list[str]], default: str = "{}",
                 engine_plan: dict[str, dict[str, list[str]]] | None = None):
        self.plan = plan
        self.engine_plan = engine_plan or {}
        self.default = default
        self.calls: list[tuple[str, str]] = []
        self.seen: dict[str, int] = {}

    def __call__(self, url: str, model: str, prompt: str) -> str:
        self.calls.append((url, prompt))
        seq = self.engine_plan.get(url, {}).get(prompt) or self.plan.get(prompt)
        if not seq:
            return self.default
        i = self.seen.get(prompt, 0)
        self.seen[prompt] = i + 1
        return seq[i] if i < len(seq) else seq[-1]

    def urls(self) -> set[str]:
        return {u for u, _ in self.calls}

    def n_calls(self) -> int:
        return len(self.calls)


def durable_rows(n: int = 40) -> list[dict]:
    """Durable-write-surface rows with prompts the transport can key on exactly.

    The surface is the module's real one — `parse_durable` runs, gold is the corpus's
    own bad/good mapped to flag/accept — and only the prompt text is synthetic, which
    is what lets a node say "row 3 gets four `flag`s and one `accept`" and have the
    per-case pass rate mean it.
    """
    rows = []
    for i in range(n):
        bad = i % 2 == 0
        rows.append({
            "surface": X.SURFACE_DURABLE_WRITE,
            "id": f"row-{i}", "query_id": f"row-{i}", "leg": "note",
            "gold": "bad" if bad else "good",
            "gold_verdict": "flag" if bad else "accept",
            "labels": ("flag", "accept"),
            "prompt": f"PROMPT-{i}",
            "primary_recorded": None, "defect_classes": [], "offered": True,
        })
    return rows


def flag(text: str = "flag") -> str:
    return json.dumps({"verdict": text, "defect_class": "none", "reason": "r"})


def menu(entities: list[int], docs: list[int] | None = None) -> str:
    """A retrieval-arm reply in the shipped labeler's own shape."""
    return json.dumps({"entities": entities, "docs": docs if docs is not None else []})


# ── clause 1: one record per row, and no answer is never an agreement ──────

def test_one_record_per_row_holds_gold_and_both_argmaxes():
    """Rows survive the run with gold and both engines' argmaxes, classified."""
    rows = durable_rows(6)
    plan = {f"PROMPT-{i}": [flag("flag" if i % 2 == 0 else "accept")] for i in range(6)}
    rec = X.run_rows(rows, Recorder(plan), X.MIN_REPEATS, ENGINES)

    assert len(rec) == 6, "the run dropped or duplicated a row"
    for r in rec:
        assert r["gold_verdict"] in ("flag", "accept")
        assert set(r["engines"]) == {"primary", "djev"}
        for eng in r["engines"].values():
            assert set(("label", "pass_rate", "draws", "answered", "abstain")) <= set(eng)
        assert r["pair_class"] in (X.AGREE, X.DISAGREE, X.ABSTAIN)
    assert all(r["pair_class"] == X.AGREE for r in rec), (
        "both engines answered the gold label on every row and the classification "
        "still did not read agree — agreement is the denominator of every figure "
        "the item asks for, so a misclassification here misstates all of them")


def test_no_usable_argmax_is_abstain_and_never_counts_toward_agreement():
    """An engine that never produced a verdict cannot be part of an agreement rate.

    Both routes to that state are pinned: every reply unparsable, and a TIE. The tie
    matters most on this pair — `architecture/tools.md:475` says djev's scores are not
    calibrated and only the argmax label is admissible, so a 5-draw tie is not a weak
    yes, it is no answer at all.
    """
    rows = durable_rows(4)
    plan = {
        "PROMPT-0": ["the model declined to answer in the requested shape"],
        "PROMPT-1": ["not json at all"],
        # 2 flag, 2 accept, one junk: five draws, no plurality winner.
        "PROMPT-2": [flag("flag"), flag("flag"), flag("accept"),
                   flag("accept"), "could not parse"],
        "PROMPT-3": [flag("accept")],
    }
    # primary draws 5 times: prompts 0/1 never parse; prompt 2 alternates -> a tie.
    rec = X.run_rows(rows, Recorder(plan), X.MIN_REPEATS, ENGINES)
    assert rec[0]["engines"]["primary"]["abstain"] is True
    assert rec[0]["pair_class"] == X.ABSTAIN
    assert rec[2]["engines"]["primary"]["abstain"] is True, (
        "a 2-and-2 tie was reported as a verdict; with no argmax it is an abstention")
    assert rec[2]["pair_class"] == X.ABSTAIN

    cell = X.localize(rec, X.SURFACE_DURABLE_WRITE)
    assert cell["non_abstaining"] == len(rec) - sum(
        1 for r in rec if r["pair_class"] == X.ABSTAIN)
    assert cell["agreement_rate"]["den"] == cell["non_abstaining"], (
        "the agreement denominator included rows where an engine never answered — "
        "that is abstention masquerading as agreement, the first risk the item names")


def test_a_tie_and_a_clean_4_of_5_are_different_states():
    """4-of-5 is a verdict with a pass rate; 2-of-5-then-2-of-5 is not a verdict."""
    four = X.argmax_verdict(["flag"] * 4 + ["accept"], ("flag", "accept"))
    tie = X.argmax_verdict(["flag"] * 2 + ["accept"] * 2 + [None], ("flag", "accept"))
    assert four == {"label": "flag", "pass_rate": 0.8, "draws": 5,
                    "answered": 5, "abstain": False}
    assert tie["abstain"] is True and tie["label"] is None
    empty = X.argmax_verdict([None, None, None], ("flag", "accept"))
    assert empty["abstain"] is True and empty["answered"] == 0


# ── clause 2: the table's shape, CIs and abstention beside every rate ──────

def test_every_published_rate_carries_a_wilson_ci_and_its_own_denominator():
    rows = durable_rows(40)
    plan = {f"PROMPT-{i}": [flag("flag" if i % 4 else "accept")] for i in range(40)}
    cell = X.localize(X.run_rows(rows, Recorder(plan), X.MIN_REPEATS, ENGINES),
                      X.SURFACE_DURABLE_WRITE)

    for key in ("agreement_rate", "error_rate_inside_agreement",
                "disagreement_volume_share", "error_localization_recall"):
        r = cell[key]
        assert {"num", "den", "value", "ci95"} <= set(r), f"{key} lost its shape"
        if r["den"] > 0:
            lo, hi = r["ci95"]
            assert lo <= r["value"] <= hi, f"{key}: point estimate outside its own CI"
            assert lo > 0 or hi < 1, (
                f"{key} reported a zero-width interval at an endpoint — that is the "
                "Wald failure `eval/stats.py` exists to avoid")
    bwa = cell["both_wrong_and_agreeing"]
    assert "count" in bwa and "share_of_all_errors" in bwa and "share_of_rows" in bwa, (
        "both-wrong-and-agreeing is not its own column: it is the finding the item "
        "expects most (four families, one wrong total to the cent) and folding it "
        "into 'agreement' is how a shared prior reads as a second reader")
    assert cell["abstention_share"]["den"] == cell["rows"] == (
        cell["non_abstaining"] + cell["abstained"])


def test_no_rate_is_computed_over_rows_where_an_engine_abstained():
    """One engine answers on 20 rows and abstains on 20; no rate may see all 40."""
    rows = durable_rows(40)
    plan = {f"PROMPT-{i}": [flag("flag" if i % 2 == 0 else "accept")] for i in range(20)}
    plan.update({f"PROMPT-{i}": [""] for i in range(20, 40)})
    cell = X.localize(X.run_rows(rows, Recorder(plan), X.MIN_REPEATS, ENGINES),
                      X.SURFACE_DURABLE_WRITE)

    assert cell["abstained"] == 20
    assert cell["agreement_rate"]["den"] == 20
    assert cell["disagreement_volume_share"]["den"] == 20
    assert cell["abstention_share"]["value"] == 0.5, (
        "half the rows abstained and the printed abstention share does not say so")


def test_markdown_prints_the_abstention_share_and_the_bwa_column():
    rows = durable_rows(35)
    plan = {f"PROMPT-{i}": [flag("flag" if i % 3 else "accept")] for i in range(35)}
    cells = [X.localize(X.run_rows(rows, Recorder(plan), X.MIN_REPEATS, ENGINES),
                        X.SURFACE_DURABLE_WRITE)]
    meta = {"ran_at": "2026-10-13T00:00:00+00:00", "repeats": X.MIN_REPEATS,
            "engines": ENGINES, "verdict": "NEGATIVE"}
    text = X.markdown(cells, meta)
    header = next(l for l in text.splitlines() if l.startswith("| surface"))
    for col in ("non-abst", "abstained", "agreement", "error-in-agreement",
                "disagree volume", "localization recall", "BOTH wrong+agree"):
        assert col in header, f"the table lost the {col!r} column"
    assert "[" in text, "no Wilson interval in the rendered table"


# ── clause 3: gold is reused, and :8091 gets nothing ─────────────────────

@needs_store
def test_the_retrieval_arm_asks_only_the_rows_the_shipped_artifact_offered():
    """Rows are exactly the offered gold labels: 44 entity + 63 doc on the 2026-09-29
    artifact. A label the candidate menu never carried is not a disagreement with
    anything — on 2026-09-29 that distinction was the whole gap between the raw 0.3404
    entity figure and the offered-only 0.7273."""
    art = X.load_retrieval_artifact(X.DEFAULT_RETRIEVAL_ARTIFACT)
    rows = X.retrieval_rows(art)
    a = art["agreement"]
    assert sum(1 for r in rows if r["leg"] == "entity") == a["entity_labels_offered"]
    assert sum(1 for r in rows if r["leg"] == "doc") == a["doc_labels_offered"]
    assert all(r["gold_verdict"] == "relevant" for r in rows)
    assert all(r["djev_recorded"] in ("relevant", "not_relevant") for r in rows), (
        "the arm lost the shipped reading it is meant to be compared against")


@needs_store
def test_the_durable_arm_joins_the_2254_rows_and_reuses_their_gold():
    """The durable arm's gold is the corpus's own label, its `primary` half is #2254's
    record, and the question both engines get is one typed question — so nothing has to
    be re-asked of an endpoint, and of :8091 nothing is asked at all.

    The arm deliberately does NOT rebuild #2254's rubric prompt. That prompt comes out
    of `arms.build_prompt` carrying a seeded example draw, so a copy of it here would be
    a second prompt to keep in step with the first, and a rubric verdict recorded against
    it would be published as though it were a re-ask of the question asked here. What
    crosses the join instead is that run's own `prompt_sha256` and `url`, published as
    provenance so a reader can tell the recorded verdict from the new draws.
    """
    corpus = X.load_corpus(X.DEFAULT_CORPUS)
    recorded = X.load_arms_final(X.DEFAULT_ARMS_FINAL)
    rows = X.durable_rows(corpus, recorded)
    by_id = {c["id"]: c for c in corpus}
    assert len(rows) == len(corpus) == 50
    assert [r["id"] for r in rows] == [c["id"] for c in corpus], (
        "the arm reordered or dropped corpus rows, so the published ids no longer "
        "identify the rows the gold came from")
    assert all(r["gold"] == by_id[r["id"]]["label"] for r in rows), (
        "gold was re-authored when the corpus already carries it")

    prior = {r["sample_id"]: r for r in recorded
             if r.get("model") == "primary" and r.get("arm") == "none"}
    assert len(prior) == 50
    for r in rows:
        assert r["recorded_primary_verdict"] == prior[r["id"]]["verdict"], (
            f"{r['id']}: #2254's recorded verdict was not carried across")
        assert r["recorded_prompt_sha256"] == prior[r["id"]]["prompt_sha256"], (
            f"{r['id']}: the published sha is not #2254's, so nothing lets a reader "
            "check which question the recorded half answered")
        assert ":8091" not in str(r["recorded_url"]), (
            f"{r['id']}'s recorded half came from the retired endpoint, which "
            "#2254's own runner refuses to re-ask at (run_arms.py:31)")

    # One typed question: the row's whole text in it, and no rubric copy in it.
    bad = by_id["bad-001"]
    body = bad.get("pre_repair_text") or bad["accepted_text"]
    assert body in rows[0]["prompt"], (
        "a bad row was not asked with its own defect in view")
    # A line that IS in #2254's template (measured: `arms.build_prompt` for bad-001
    # opens "You are auditing one durable knowledge note", 14,632 chars with a seeded
    # example draw). Its absence is what says this arm asked its own question.
    assert "You are auditing one durable knowledge note" not in rows[0]["prompt"], (
        "the arm rebuilt #2254's rubric prompt, so it would report #2254's rubric "
        "verdict as a re-ask of the question this arm asked")
    assert not any(k in r for r in rows for k in ("leg", "offered", "query_id")), (
        "a durable row published a retrieval-only field, so the artifact describes a "
        "menu question that was never asked of it")

    tr = Recorder({}, default=flag("accept"))
    X.run_rows(rows, tr, X.MIN_REPEATS, ENGINES)
    assert tr.urls() == {PRIMARY, DJEV}, (
        f"the durable arm sent a request outside the live pair: {tr.urls()}")


def test_a_partial_join_stops_the_durable_arm_rather_than_renaming_its_denominator():
    """50 corpus rows measured against 49 recorded verdicts is a different measurement,
    not a smaller one, so the arm refuses rather than lose a row quietly."""
    corpus = X.load_corpus(X.DEFAULT_CORPUS)
    dropped = [r for r in X.load_arms_final(X.DEFAULT_ARMS_FINAL)
               if not (r.get("model") == "primary" and r.get("arm") == "none"
                       and r.get("sample_id") == "bad-001")]
    with pytest.raises(RuntimeError, match="cannot join 1 of 50"):
        X.durable_rows(corpus, dropped)
    with pytest.raises(RuntimeError, match="cannot join 50 of 50"):
        X.durable_rows(corpus, [r for r in X.load_arms_final(X.DEFAULT_ARMS_FINAL)
                                if r.get("model") != "primary"])


    with pytest.raises(ValueError, match="retired endpoint"):
        X.ask("http://127.0.0.1:8091/v1/chat/completions", "x", "p")


def _http_error(body: str, code: int = 400) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "Bad Request", {},  # type: ignore[arg-type]
                                  io.BytesIO(body.encode()))


def test_an_endpoint_that_refuses_a_parameter_gets_one_retry_without_it():
    """djev is a diffusion build: it answers any `temperature` with a 400 naming seven
    sampling parameters, and #2254's primary pass needed `chat_template_kwargs` dropped
    the same way. Without the retry every djev row dies on the first POST and the whole
    surface comes back as abstention — which would read as an engine that never agrees,
    the exact opposite of what is happening."""
    refusals = ["The temperature, min_p, seed, min_tokens, logit_bias, bad_words, and "
                "allowed_token_ids sampling parameters are not yet supported with "
                "diffusion models."]
    sent: list[dict] = []

    def post(url: str, payload: dict) -> dict:
        sent.append(dict(payload))
        if refusals:
            raise _http_error(refusals.pop())
        return {"choices": [{"message": {"content": flag("flag")}}]}

    assert json.loads(X.ask(DJEV, "m", "p", post=post))["verdict"] == "flag"
    assert "temperature" not in sent[1] and "min_p" not in sent[1]
    assert "temperature" in sent[0], "the first attempt should still ask for it"
    assert any(d["url"] == DJEV and d["dropped"] == "temperature"
               for d in X.PARAM_DROPS), "a dropped parameter must be recorded"


def test_a_400_that_names_no_sent_parameter_propagates():
    """The retry must not become a way to retry anything: an unknown model or a
    malformed body is a request bug, and silently retrying it turns one error into a
    weirder one."""
    def post(url: str, payload: dict) -> dict:
        raise _http_error('{"error": {"message": "model not found"}}')

    with pytest.raises(urllib.error.HTTPError):
        X.ask(DJEV, "no-such-model", "p", post=post)


def test_the_http_seam_refuses_the_retired_port_before_sending():
    sent: list[str] = []

    def post(url: str, payload: dict) -> dict:
        sent.append(url)
        return {"choices": [{"message": {"content": ""}}]}

    with pytest.raises(ValueError, match="retired endpoint"):
        X.ask("http://127.0.0.1:8091/v1/chat/completions", "m", "p", post=post)
    assert sent == [], "the request was built and sent before the check could stop it"


# ── clause 4: repeats, per-case pass rate, and the UNMEASURABLE floor ─────

def test_every_row_is_asked_repeats_times_per_engine():
    rows = durable_rows(7)
    plan = {f"PROMPT-{i}": [flag("flag" if i % 2 == 0 else "accept")] for i in range(7)}
    tr = Recorder(plan)
    rec = X.run_rows(rows, tr, X.MIN_REPEATS, ENGINES)
    assert tr.n_calls() == 7 * 2 * X.MIN_REPEATS, (
        "an engine was sampled fewer than 5 times on some row; one draw is an "
        "anecdote and the item names 5 explicitly")
    assert X.MIN_REPEATS >= 5
    for r in rec:
        for eng in r["engines"].values():
            assert eng["draws"] == X.MIN_REPEATS


def test_the_table_reports_a_per_case_pass_rate_not_a_single_draw():
    rows = durable_rows(30)
    plan = {f"PROMPT-{i}": [flag("flag")] * 4 + [flag("accept")]
            for i in range(30)}
    cell = X.localize(X.run_rows(rows, Recorder(plan), X.MIN_REPEATS, ENGINES),
                      X.SURFACE_DURABLE_WRITE)
    summary = cell["per_case_pass_rate"]["primary"]
    assert summary["rows_with_an_answer"] == 30
    assert summary["mean"] == 0.8, (
        "4-of-5 draws agreed and the reported per-case pass rate is not 0.8, so the "
        "table is summarising something other than the draw history")


def test_a_thin_cell_prints_unmeasurable_instead_of_a_rate():
    """29 non-abstaining pairs is below the floor: the cell says so in the exact
    string, because a printed 0.7 on n=29 reads identically to a 0.7 on n=207."""
    rows = durable_rows(29)
    plan = {f"PROMPT-{i}": [flag("flag" if i % 2 == 0 else "accept")] for i in range(29)}
    cell = X.localize(X.run_rows(rows, Recorder(plan), X.MIN_REPEATS, ENGINES),
                      X.SURFACE_DURABLE_WRITE)
    assert cell["non_abstaining"] == 29 and not cell["measurable"]
    assert cell["unmeasurable"] == (
        f"UNMEASURABLE surface={X.SURFACE_DURABLE_WRITE} non_abstaining=29")
    text = X.markdown([cell], {"ran_at": "x", "repeats": 5, "engines": ENGINES,
                               "verdict": "NEGATIVE"})
    assert "UNMEASURABLE surface=" in text and "non_abstaining=29" in text
    assert cell["positive_gate"]["positive"] is False, (
        "a thin cell passed the POSITIVE gate; a small n must never look like a "
        "result, and the stop clause closes a surface that cannot produce 30")


# ── clause 5: the POSITIVE test, and the write scope ─────────────────────

def _cell(recall: float | None, volume: float, bwa: float | None,
          n: int = 60, errors: int = 20) -> dict:
    """A cell at given readings, built through the same shape `localize` returns."""
    disagree = round(volume * n)
    bwa_n = errors - disagree if bwa is None else int((bwa or 0) * errors)
    c = {"measurable": n >= X.MIN_NON_ABSTAINING, "non_abstaining": n,
         "errors_total": errors,
         "error_localization_recall": {"value": recall},
         "disagreement_volume_share": {"value": volume},
         "both_wrong_and_agreeing": {"share_of_all_errors": {
             "value": (None if errors == 0 else round(bwa_n / errors, 4))}}}
    return c


def test_positive_requires_all_three_thresholds_on_one_surface():
    good = _cell(0.90, 0.20, 0.05)
    assert X.positive_for_surface(good)["positive"] is True

    low_recall = _cell(0.69, 0.20, 0.05)
    wide = _cell(0.90, 0.36, 0.05)
    shared = _cell(0.90, 0.20, 0.10)
    for cell, why in ((low_recall, "recall"), (wide, "volume"), (shared, "bwa")):
        gate = X.positive_for_surface(cell)
        assert gate["positive"] is False, f"{why} at the threshold read as POSITIVE"
        assert gate["reasons"], "a refused gate must say which condition it refused"
    assert X.overall_verdict([low_recall, wide, shared]) == "NEGATIVE"
    assert X.overall_verdict([low_recall, good]) == "POSITIVE", (
        "the item's test is 'on SOME surface', so one passing surface must decide it")


def test_a_surface_with_no_errors_is_not_positive():
    """Zero errors means localization is undefined, not perfect. A pair that made no
    error on 40 rows has not shown that disagreement finds errors."""
    cell = _cell(None, 0.0, None, n=40, errors=0)
    gate = X.positive_for_surface(cell)
    assert gate["positive"] is False
    assert any("no errors" in r for r in gate["reasons"])


#: A transport that always answers `flag`, whatever it is shown. Driving the REAL
#: durable arm (50 corpus rows, real prompts, real gold) with it produces the shape
#: this item exists to look for: perfect agreement, and every agreement on a `good`
#: row wrong. Both engines agree, both are wrong, 30 times.
ALWAYS_FLAG = Recorder({}, default=flag("flag"))


def _fake_resolve(got: list[str]):
    """A `/v1/models` fetcher that records which engines the run went looking for."""
    def fetch(url: str) -> dict:
        got.append(url)
        tag = "8096" if "8096" in url else ("8010" if "8010" in url else "other")
        return {"data": [{"id": f"served-{tag}"}]}
    return fetch


def test_the_pair_is_resolved_by_name_and_secondary_is_never_looked_up():
    """The run resolves `primary` and `djev` through the shipped resolver, and asks no
    question of :8091 — not at POST time and not at resolution time either."""
    got: list[str] = []
    engines = X.resolve_engines(fetcher=_fake_resolve(got))
    assert tuple(engines) == X.ENGINE_NAMES == ("primary", "djev")
    assert not any("8091" in u for u in got), (
        "the retired secondary was resolved, which means it was considered as a "
        "reader: the box cannot run it and the pair must not silently become it")
    assert engines["primary"]["url"].endswith("/v1/chat/completions")
    assert engines["djev"]["model"] == "served-8010", (
        "the served name came from somewhere other than the endpoint that will answer")


@needs_store
def test_the_run_writes_one_dated_json_and_one_table_under_out(tmp_path):
    """`--out` gets exactly two dated files, over the real gold, with no socket.

    The always-`flag` transport makes the arithmetic checkable by reading: the pair
    agrees on all 50 rows, is wrong on the 30 `good` ones, and so `both_wrong_and_agreeing`
    is 30 of 30 errors while the disagreement set is empty — the shared-convention
    outcome, which must come out NEGATIVE and must not be able to hide inside a
    100% agreement rate.
    """
    out = tmp_path / "cross-engine"
    assert X.main(["--out", str(out), "--arms", X.SURFACE_DURABLE_WRITE],
                  transport=ALWAYS_FLAG, engines=ENGINES) == 0
    files = sorted(p.name for p in out.iterdir())
    assert len(files) == 2, f"expected one json + one table, got {files}"
    assert any(f.endswith(".json") for f in files) and any(f.endswith(".md") for f in files)
    art = json.loads(next(out.glob("*.json")).read_text())
    assert art["meta"]["kind"] == "cross_engine_disagreement"
    assert art["meta"]["verdict"] == "NEGATIVE"
    assert art["meta"]["repeats"] == X.MIN_REPEATS
    assert art["meta"]["wires_nothing"] is True, (
        "the artifact must state that no gate, rung or dispatch path reads it; the "
        "item measures and stops, and a reopening ruling is not this run's to make")
    surface = art["surfaces"][0]
    assert surface["surface"] == X.SURFACE_DURABLE_WRITE
    assert surface["non_abstaining"] == 50 and surface["abstained"] == 0
    assert surface["agreement_rate"]["value"] == 1.0
    assert surface["both_wrong_and_agreeing"]["count"] == 30, (
        "both engines said `flag` on all 30 good rows and 20 bad ones; the wrong "
        "agreements are the 30 good rows and the count does not say so")
    assert surface["error_localization_recall"]["value"] == 0.0, (
        "an empty disagreement set was given nonzero localization recall")
    table = next(out.glob("*.md")).read_text()
    assert "BOTH wrong+agree" in table and "| 30 (" in table, (
        "the rendered table folded the 30 wrong agreements away instead of printing "
        "them in their own column")


@needs_store
def test_the_published_denominators_recompute_from_the_artifact_alone(tmp_path):
    """The acceptance check: a reader holding the artifact and no engine reproduces
    every cell. Anything else would make `surfaces` the run's private opinion about
    rows it also published."""
    out = tmp_path / "out"
    X.main(["--out", str(out), "--arms", X.SURFACE_DURABLE_WRITE],
           transport=ALWAYS_FLAG, engines=ENGINES)
    path = next(out.glob("*.json"))
    art = json.loads(path.read_text())
    again = X.recompute(path)
    for stored in art["surfaces"]:
        re_cell = again[stored["surface"]]
        for key in ("agreement_rate", "error_rate_inside_agreement",
                    "disagreement_volume_share", "error_localization_recall"):
            assert stored[key]["den"] == re_cell[key]["den"], (
                f"{key}'s denominator is not reproducible from the published rows")
            assert stored[key]["value"] == re_cell[key]["value"]
        assert stored["both_wrong_and_agreeing"]["count"] == \
            re_cell["both_wrong_and_agreeing"]["count"]


def test_repeats_below_five_is_refused_at_the_door():
    """The floor is a CLI refusal, not a comment: a 3-draw run would print the same
    table and read as a measurement."""
    with pytest.raises(SystemExit):
        X.main(["--out", "/tmp/never-written", "--repeats", "3",
                "--arms", X.SURFACE_DURABLE_WRITE], transport=Recorder({}, default=""))
    assert not Path("/tmp/never-written").exists()


def test_the_change_that_added_this_measurement_touched_only_eval_and_tests():
    """Clause 5's path scope, pinned on the commit that ADDED these two files.

    Reading `HEAD` would be fragile — the day an unrelated commit lands in `app/` the
    node would start failing for a reason that has nothing to do with this measurement.
    The commit that added these paths is immutable, so asking it for its own file list
    is a claim that cannot rot: this item's footprint in the repo is these two files and
    nothing else, and if a future round widens it this node says so.
    """
    import subprocess

    sha = subprocess.run(
        ["git", "log", "--format=%H", "--diff-filter=A", "-1", "--",
         "eval/cross_engine_disagreement.py", "tests/test_cross_engine_disagreement.py"],
        cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()[0]
    files = subprocess.run(["git", "show", "--name-only", "--format=", sha],
                           cwd=ROOT, capture_output=True,
                           text=True).stdout.split()
    assert files, f"no file list for {sha[:8]}"
    stray = [f for f in files if not (f.startswith("eval/") or f.startswith("tests/"))]
    assert stray == [], (
        f"{sha[:8]} added this measurement alongside {stray}; the item ships a "
        "measurement and wires nothing, so no third path belongs in that commit")


def test_the_runner_writes_nothing_outside_its_out_directory(tmp_path):
    """The measurement reads two live corpora and writes only what it is told to.

    Its inputs are the gold: `eval/durable_write_judge/corpus.jsonl` is the labelled
    20-bad/30-good set another item still has to get ratified, and the label artifact is
    the shipped djev run whose offered-label set defines this arm's denominator. A run
    that truncated either while producing a table would destroy the thing it measured and
    still print a number, so this is checked over a real end-to-end run rather than by
    inspection: every byte of every input, and the set of paths that appeared anywhere
    under the repo, is compared before and after.
    """
    import hashlib
    import os
    import subprocess

    inputs = [X.DEFAULT_CORPUS, X.DEFAULT_RETRIEVAL_ARTIFACT, X.DEFAULT_ARMS_FINAL]
    before = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs}
    # A set DIFFERENCE, not emptiness: this file is itself an uncommitted edit while
    # the round is in flight, so a guard demanding a clean tree would fail for a reason
    # unrelated to the run. What may not happen is the run ADDING an entry.
    def dirty() -> set[str]:
        return set(subprocess.run(["git", "status", "--porcelain", "--", "eval",
                                   "tests"], cwd=ROOT, capture_output=True,
                                  text=True).stdout.splitlines())

    before_dirty = dirty()

    out = tmp_path / "out"
    X.main(["--out", str(out), "--arms", X.SURFACE_DURABLE_WRITE],
           transport=ALWAYS_FLAG, engines=ENGINES)

    after = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs}
    assert before == after, "the run modified a corpus it only reads"
    assert dirty() == before_dirty, (
        f"the run touched eval/ or tests/: {sorted(dirty() - before_dirty)}; its only "
        "output surface is the --out directory it was given")
    written = sorted(os.path.relpath(os.path.join(d, fn), out)
                     for d, _, fs in os.walk(out) for fn in fs)
    assert len(written) == 2 and all(f.endswith((".json", ".md")) for f in written), (
        f"expected exactly the dated json + table under --out, got {written}")


def test_the_module_wires_nothing_into_the_live_paths():
    """No gate, rung, worker or config reads this module. The claim is the item's stop
    clause, and a stray import would be exactly the wiring it forbids."""
    import subprocess

    hits = subprocess.run(
        ["git", "grep", "-l", "cross_engine_disagreement", "--", "app", "scripts",
         "workers", "config.yaml", "autonomy", "skills"],
        cwd=ROOT, capture_output=True, text=True).stdout.strip()
    assert hits == "", f"the measurement leaked into the live paths: {hits}"


# ── arm 1: the third surface (#2824) ────────────────────────────────────────

from eval.cross_engine_disagreement_arm1 import rubric as _RB  # noqa: E402

_REPO = Path(X.__file__).resolve().parents[1]
ARM1_GOLD = _REPO / "eval" / "cross_engine_disagreement_arm1" / "gold.jsonl"
ARM1_SEED = 20261016


def _arm1_gold_rows(n: int, *, gold: str = "consistent") -> list[dict]:
    """Arm-1 shaped rows with the real rubric hash and an artifact-naming warrant, so
    the loader accepts them and only their count varies between tests."""
    line = ("AR-CONSISTENT-PATH: the reviewed call's arguments (Bash) name "
            "'tests/test_widget.py', read from "
            "/home/alansrobotlab/lloyd-data/sessions/sess-x.json")
    return [{"inputs_digest": f"{i:016x}", "seam": "action_review",
             "schema_hash": "6caedf2ab308eb80", "stratum": "action_review_high",
             "question": "Which describes it?",
             "options": {"consistent": "on task", "injected": "off task",
                         "unrelated": "unrelated"},
             "state": f"CANVAS {i}", "question_id": "on_task",
             "gold": gold, "warrant": line,
             "rubric_sha256": _RB.rubric_sha256(),
             "labelled_at": "2026-10-16T00:00:00+00:00",
             "labelled_by": ("lloyd (rule pass over the artifacts named in `warrant`; "
                             "not a human label)")} for i in range(n)]


def test_arm1_is_a_third_surface_registered_and_selectable(tmp_path):
    """Clause 3: PARSERS holds three surfaces, `--arms` selects the new one alone,
    and the rules it is scored under are the ones the file already had."""
    assert set(X.PARSERS) == {X.SURFACE_RETRIEVAL, X.SURFACE_DURABLE_WRITE,
                              X.SURFACE_ARM1}
    assert callable(X.PARSERS[X.SURFACE_ARM1])
    gold = tmp_path / "gold.jsonl"
    gold.write_text("".join(json.dumps(r) + "\n" for r in _arm1_gold_rows(31)),
                    encoding="utf-8")
    rows = X.build_rows(tmp_path / "no-retrieval.json", tmp_path / "no-corpus.jsonl",
                        tmp_path / "no-arms.jsonl", (X.SURFACE_ARM1,), None,
                        arm1_gold_path=gold)
    assert len(rows) == 31
    assert {r["surface"] for r in rows} == {X.SURFACE_ARM1}
    # the rules are the file's existing constants, unchanged by this round
    assert X.MIN_REPEATS == 5 and X.MIN_NON_ABSTAINING == 30
    assert X.POSITIVE_MIN_RECALL == 0.70 and X.POSITIVE_MAX_VOLUME_SHARE == 0.35
    assert X.POSITIVE_MAX_BWA_SHARE == 0.10
    # The surface's default gold path is the tracked file, so `--arms shadow-gold-sample`
    # with no `--gold` scores the sample this round ships rather than nothing.
    assert X.DEFAULT_ARM1_GOLD == ARM1_GOLD and X.DEFAULT_ARM1_GOLD.exists()


def test_arm1_gold_loader_refuses_an_unvalidated_file(tmp_path):
    """The runner's own read of arm 1's gold validates it: the denominators come from
    here, so a hand-edited row must not reach a table."""
    with pytest.raises((RuntimeError, FileNotFoundError)):
        X.load_arm1_gold(tmp_path / "absent.jsonl")
    bad = tmp_path / "bad.jsonl"
    row = _arm1_gold_rows(1)[0]
    row["warrant"] = "it looks right to me"
    bad.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(Exception) as e:
        X.load_arm1_gold(bad)
    assert "warrant" in str(e.value).lower()


def test_arm1_row_prompt_carries_the_canvas_and_its_own_options():
    """An arm-1 row must put the seam's own canvas text and the seam's own option
    words to the endpoint — not a prompt this file invented for it."""
    r = _arm1_gold_rows(1, gold="unrelated")[0]
    row = X.arm1_rows([r])[0]
    assert row["prompt"].startswith("CANVAS STATE:")
    assert "CANVAS 0" in row["prompt"]
    assert "Answer with exactly one of: consistent, injected, unrelated" in row["prompt"]
    assert row["gold_verdict"] == "unrelated"
    assert row["labels"] == ("consistent", "injected", "unrelated")
    assert row["warrant"] == r["warrant"]
    assert row["inputs_digest"] == r["inputs_digest"]
    assert X.PARSERS[X.SURFACE_ARM1](row, "unrelated") == "unrelated"
    assert X.PARSERS[X.SURFACE_ARM1](row, "consistent") == "consistent"
    assert X.PARSERS[X.SURFACE_ARM1](row, "the answer is obvious") is None


def _arm1_asked(n_rows: int, transport, repeats: int = X.MIN_REPEATS):
    rows = X.arm1_rows(_arm1_gold_rows(n_rows))
    return X.run_rows(rows, transport, repeats, dict(ENGINES))


def test_arm1_cell_under_thirty_non_abstaining_prints_unmeasurable(capsys):
    """Clause 4: a thin arm-1 cell prints
    `UNMEASURABLE surface=shadow-gold-sample non_abstaining=<n>` in place of a rate,
    and is never reported POSITIVE."""
    def abstainer(url, model, prompt):
        return "neither of these"
    cell = X.localize(_arm1_asked(29, abstainer), X.SURFACE_ARM1)
    assert cell["non_abstaining"] == 0
    assert cell["measurable"] is False
    assert cell["unmeasurable"] == ("UNMEASURABLE surface=shadow-gold-sample "
                                    "non_abstaining=0")  # the printed sentence itself
    assert cell["positive_gate"]["positive"] is False
    assert "non_abstaining=0 < 30" in cell["positive_gate"]["reasons"]
    md = X.markdown([cell], {"ran_at": "x", "repeats": X.MIN_REPEATS,
                             "engines": ENGINES, "verdict": "INDETERMINATE"})
    assert "UNMEASURABLE surface=shadow-gold-sample non_abstaining=0" in md
    assert not any(l.startswith("|") and "0." in l and X.SURFACE_ARM1 in l
                   for l in md.splitlines()), "a thin cell must not print a rate"


def test_arm1_thirtyth_pair_makes_the_cell_measurable_not_the_thirty_first():
    """The floor is 30, and it is the same floor the other two surfaces use."""
    def answerer(url, model, prompt):
        return "consistent"
    thick = X.localize(_arm1_asked(30, answerer), X.SURFACE_ARM1)
    assert thick["non_abstaining"] == 30 and thick["measurable"] is True
    # a measurable cell carries no UNMEASURABLE sentence at all
    assert "unmeasurable" not in thick


def test_arm1_positive_requires_the_same_three_numbers_and_cannot_be_a_stacked_yes():
    """Clause 3's other half: the POSITIVE thresholds are unchanged for arm 1, and on
    this surface at least one of them always refuses — perfect agreement scores
    agreement 1.0, recall 0 (no errors to catch) and so stays not-POSITIVE."""
    def answerer(url, model, prompt):
        return "consistent"
    cell = X.localize(_arm1_asked(40, answerer), X.SURFACE_ARM1)
    assert cell["measurable"] and cell["errors_total"] == 0
    assert cell["positive_gate"]["positive"] is False
    assert ("no errors observed, so localization is undefined"
            in cell["positive_gate"]["reasons"])


def test_arm1_never_ships_a_positive_even_when_every_engine_is_wrong_together():
    def both_wrong_agree(url, model, prompt):
        return "injected"
    cell = X.localize(_arm1_asked(40, both_wrong_agree), X.SURFACE_ARM1)
    assert cell["both_wrong_and_agreeing"]["count"] == 40
    assert cell["positive_gate"]["positive"] is False


def test_arm1_gold_is_shipped_120_seeded_rows_with_warrants():
    """The contract's own acceptance check, run against the tracked file: exactly 120
    rows, all validating, all warrant-bearing, and the sampler's recorded ids
    reproducing them."""
    rows = _RB.load_gold(ARM1_GOLD)
    assert len(rows) == 120
    assert all(r["warrant"].strip() and r["rubric_sha256"].strip() for r in rows)
    side = json.loads((ARM1_GOLD.parent / "sample.json").read_text())
    assert side["inputs_digests"] == [r["inputs_digest"] for r in rows]
    assert len(set(side["inputs_digests"])) == 120
    assert side["seed"] == ARM1_SEED


def test_the_change_that_added_arm_1_touched_only_eval_and_tests():
    """Arm 1's commit carried no CODE outside `eval/` and `tests/`: a third surface must
    not need an `app/` change to exist. Two defects, both found by the node going red at
    `caafecd6` while the rule it states was satisfied.

    **Which commit it asks.** It used to read "the commit that most recently touched
    `eval/cross_engine_disagreement.py`", which is not arm 1's commit — that file is the
    shared runner, so the answer moves every time anyone edits it, and the day #2824
    landed, the answer became #2824's commit for the wrong reason. Its own docstring said
    THIS commit, and the node above
    (`test_the_change_that_added_this_measurement_touched_only_eval_and_tests`) already
    states the rule this one violated: only the commit that ADDED a path is immutable, so
    only that one makes a claim that cannot rot. Arm 1 added its own surface under
    `eval/cross_engine_disagreement_arm1/`, so that is what is resolved now, and the
    membership check below proves the resolution landed on the commit that adds arm 1's
    surface AND arm 1's suite, each path by exact name. `--diff-filter=A` over two paths
    returns the newest
    commit adding EITHER one, so membership of both is what excludes a later commit that
    re-adds only one of them, as well as the previous arm's commit and any later edit to
    the shared runner.

    **What it forbids.** It used to reject every path outside `eval/` and `tests/`, which
    is a stricter rule than the one filed and was unsatisfiable on the facts: shipping an
    `eval/<arm>/` directory compels a row in the arms inventory, because
    `tests/test_architecture_coverage_doc_claims.py::test_measurement_doc_names_exactly_the_eval_arms_that_exist`
    fails the doc the moment an arm directory is unlisted. No arm-1 commit could exist
    that kept both nodes green — one demanded a documentation edit outside `eval/` and
    `tests/`, the other refused it. So the name now means what the clause means: no CODE
    outside `eval/` and `tests/`. Documentation is admitted on one narrow, checked ground
    — the arms row the other guard compels — and a byte of Python, YAML, service or
    frontend code anywhere else still reddens this node.
    """
    import subprocess

    def _git(*args: str) -> str:
        run = subprocess.run(["git", *args], cwd=str(_REPO),
                             capture_output=True, text=True)
        assert run.returncode == 0, f"git {' '.join(args)} failed: {run.stderr[:300]}"
        return run.stdout

    #: Arm 1's own added paths — the surface its commit created, not the shared runner it
    #: wired into. `--diff-filter=A` keeps the answer fixed for the life of the repo, but
    #: over several paths it returns the newest commit adding EITHER one, so a later
    #: commit that deleted and re-added one path would re-anchor the node. The membership
    #: check below closes that: the resolved commit has to add BOTH, each by exact name.
    ARM1_ADDED_PATHS = ("eval/cross_engine_disagreement_arm1/rubric.py",
                        "tests/test_cross_engine_arm1_gold.py")
    found = _git("log", "--format=%H", "--diff-filter=A", "-1", "--",
                 *ARM1_ADDED_PATHS).strip()
    assert found, f"no commit adds {ARM1_ADDED_PATHS[0]} — arm 1 has no history"
    sha = found.splitlines()[0]
    files = [f for f in _git("show", "--pretty=format:", "--name-only", sha).splitlines()
             if f.strip()]
    assert files, f"no file list for {sha[:8]}"
    for added in ARM1_ADDED_PATHS:
        assert added in files, (
            f"{sha[:8]} does not add {added}, so it is not the commit that added both of "
            "the paths this node resolves on: it is the previous arm's commit, a later "
            "commit that re-added only one of them, or an edit to the shared runner")

    stray = [f for f in files
             if not (f.startswith("eval/") or f.startswith("tests/")
                     or f.endswith(".md"))]
    assert stray == [], (
        f"{sha[:8]} added arm 1 alongside code outside eval/ and tests/: {stray}; the "
        "third surface is a corpus and a sampler, so it wires nothing and no app/, "
        "agent_mcp/, script, service or web file belongs in that commit")

    # The one permission the widened rule grants, granted exactly: the docs it admits are
    # the arms inventory, and its hunk has to be the row naming this arm — which is the
    # row `test_measurement_doc_names_exactly_the_eval_arms_that_exist` compels. Any
    # other documentation in that commit is not compelled by anything and so is stray.
    docs = [f for f in files
            if f.endswith(".md")
            and not (f.startswith("eval/") or f.startswith("tests/"))]
    assert docs == ["architecture/measurement.md"], (
        f"arm 1's commit carries documentation outside eval/ and tests/ beyond the arms "
        f"inventory the doc-claim guard compels: {docs}")
    hunk = _git("show", "--pretty=format:", sha, "--", "architecture/measurement.md")
    added = [ln for ln in hunk.splitlines() if ln.startswith("+")]
    assert any("cross_engine_disagreement_arm1" in ln for ln in added), (
        "arm 1 touched architecture/measurement.md without adding its arms row, so the "
        "doc edit is not the one the doc-claim guard required")


# ── #3345: Cohen's kappa against gold, degeneracy, and what a κ may license ──
#
# The published table reported uncorrected agreement, so the question "is djev close
# enough to the primary judge to take this criterion off it" had to be derived by hand
# from the artifact's rows — and the one thing that stops a wrong answer, an arm that
# emits a single label on every row, was invisible. Two numbers from the shipped
# artifacts, quoted below and re-derived by a node in this file:
#
#   durable-write-judge, 2026-10-16-103133, scored on `gold_verdict` —
#       primary κ 0.1096 (po 0.480, pe 0.416, n 50, labels {'flag': 46, 'accept': 4})
#       djev    κ 0.0000 (po 0.400, pe 0.400, n 50, labels {'flag': 50}) ← constant arm
#   shadow-gold-sample, 2026-10-17-063430, scored on `gold`/`gold_verdict` (one
#   vocabulary on this surface) — primary κ 0.9125, djev κ 0.7881, both over the shared
#   116 non-abstaining pairs.
#
# Those rows also carry the misread this rail exists to kill: score the DURABLE surface
# on its `gold` column (`bad`/`good`, the corpus's own label) and κ is 0.0000 for BOTH
# engines with po 0.0 — not one row matches, because the vocabularies do not overlap.
#
# The fixtures below are synthetic and go through the real pipeline: `durable_rows`'
# prompts, `Recorder`'s per-engine plan, `run_rows`, `parse_durable`, `localize`,
# `markdown` and `write_artifact`. No node here opens a socket, and no node needs the
# ~1,200-call live re-run to pin a figure.

def _durable_plan_by_engine(rows_n: int, primary_of, djev_of) -> dict:
    """Per-engine replies for `durable_rows(rows_n)`, keyed the way `Recorder` keys them.

    `Recorder.plan` is shared by both transports, so a node that wants "primary says one
    thing, djev says another" — the only pair shape disagreement can come from — has to
    use `engine_plan`, which is keyed by URL. Both callables take the row index and return
    the label that engine's five draws should all produce, so the argmax is a certainty
    rather than a plurality.
    """
    out: dict[str, dict[str, list[str]]] = {}
    for eng, fn in (("primary", primary_of), ("djev", djev_of)):
        out[ENGINES[eng]["url"]] = {f"PROMPT-{i}": [flag(fn(i))] for i in range(rows_n)}
    return out


def _marginal_run(primary_of, djev_of,
                  rows_n: int = 40) -> tuple[list[dict], dict]:
    """The asked rows AND their cell, for the labels a node specifies.

    Both are returned because `write_artifact` stores the rows beside the cell and the
    shipped property is that the cell is recomputable from them; a node holding only the
    cell could not test that. The shared `plan` is empty on purpose — every reply comes
    from `engine_plan`, so the two engines cannot silently be shown the same text.
    """
    rows = durable_rows(rows_n)
    rec = X.run_rows(rows, Recorder({}, engine_plan=_durable_plan_by_engine(
        rows_n, primary_of, djev_of)), X.MIN_REPEATS, ENGINES)
    return rec, X.localize(rec, X.SURFACE_DURABLE_WRITE)


def _marginal_cell(primary_of, djev_of, rows_n: int = 40) -> dict:
    return _marginal_run(primary_of, djev_of, rows_n)[1]


def test_a_known_marginals_run_publishes_a_hand_computed_kappa_in_cell_and_table():
    """Clause 1: κ over a fixture whose marginals are arithmetic, not folklore.

    40 rows, gold flag on the 20 even indices and accept on the 20 odd ones. Primary
    answers `flag` 24 times and `accept` 16 times, and is right on 32 rows: 18 of the 20
    gold-flag rows and 14 of the 20 gold-accept rows. So

        po = 32/40                 = 0.800
        pe = (24/40)(20/40) + (16/40)(20/40) = 0.300 + 0.200 = 0.500
        κ  = (0.800 - 0.500) / (1 - 0.500)   = 0.600

    djev answers the gold label on every row, so po = 1.000 with the same 0.500 marginals
    and κ = (1.000 - 0.500) / 0.500 = 1.000. Both over all 40 pairs, which is the point of
    the shared denominator: a κ computed over a different row set is not this κ.
    """
    cell = _marginal_cell(
        lambda i: "accept" if i in (36, 38) else ("flag" if i % 2 == 0
                                                  else ("flag" if i >= 29 else "accept")),
        lambda i: "flag" if i % 2 == 0 else "accept")
    assert cell["non_abstaining"] == 40 and cell["measurable"]

    pri, dje = cell["engine_kappa"]["primary"], cell["engine_kappa"]["djev"]
    assert (pri["n"], pri["po"], pri["pe"], pri["kappa"]) == (40, 0.8, 0.5, 0.6), pri
    assert (dje["n"], dje["po"], dje["pe"], dje["kappa"]) == (40, 1.0, 0.5, 1.0), dje
    assert pri["labels"] == {"accept": 16, "flag": 24}, pri["labels"]
    assert pri["column"] == dje["column"] == X.SCORED_GOLD_COLUMN == "gold_verdict"
    assert pri["n_classes"] == 2 and pri["degenerate"] is False

    text = X.markdown([cell], {"ran_at": "x", "repeats": X.MIN_REPEATS,
                              "engines": ENGINES, "verdict": "NEGATIVE"})
    line = next(l for l in text.splitlines() if l.startswith(f"- **{X.SURFACE_DURABLE_WRITE}**"))
    assert "κ vs gold_verdict 0.600 (po 0.800, pe 0.500, n=40, 2 label(s) emitted)" in line, line
    assert "κ vs gold_verdict 1.000 (po 1.000, pe 0.500, n=40, 2 label(s) emitted)" in line, line


def test_the_kappa_and_its_scored_column_survive_into_the_artifact_and_recompute(tmp_path):
    """Clause 1's other half: the JSON record, not only the rendered table.

    `recompute()` re-runs `localize` over the artifact's own rows and the shipped
    property is that the two agree, so a κ that lived only in the `.md` would be caught
    here — as would a κ that could not be reproduced from stored per-row verdicts at all.
    """
    rec, cell = _marginal_run(lambda i: "flag" if i % 2 == 0 else "accept",
                              lambda i: "flag" if i % 2 == 0 else "accept")
    meta = {"ran_at": "2026-10-26T00:00:00+00:00", "repeats": X.MIN_REPEATS,
            "engines": ENGINES, "verdict": "NEGATIVE",
            "scored_gold_column": X.scored_gold_meta([cell])}
    paths = X.write_artifact(tmp_path, meta, [cell], rec)
    art = json.loads(paths["json"].read_text())

    stored = art["surfaces"][0]["engine_kappa"]
    # Both engines answer the gold label here, so po is 1.0 against pe 0.5 and κ is 1.0 —
    # the arithmetic of a perfect arm, stored as data rather than as a rendered string.
    assert stored["primary"]["kappa"] == 1.0 and stored["primary"]["n"] == 40, stored
    assert stored["primary"]["column"] == "gold_verdict"
    assert art["meta"]["scored_gold_column"][X.SURFACE_DURABLE_WRITE] == {
        "column": "gold_verdict", "labels": ["accept", "flag"],
        "kappa_basis": X.KAPPA_BASIS}
    assert X.recompute(paths["json"])[X.SURFACE_DURABLE_WRITE]["engine_kappa"] == stored, (
        "the published κ and the κ recomputed from the artifact's rows disagree, which "
        "makes the number in the table unfalsifiable by anyone holding the artifact")
    assert "κ vs gold_verdict" in paths["markdown"].read_text()


def test_a_constant_label_arm_is_flagged_degenerate_and_its_kappa_is_not_a_number():
    """Clause 2: the arm that prints 0.000 abstention and a perfect pass rate.

    The shipped shape: on the 2026-10-16 durable artifact djev answered `flag` on 50 of
    50 rows, which scores po 0.400 against pe 0.400 and κ exactly 0.0000 — while the same
    .md line prints `abstains 0.000 [0.000,0.071] (0/50), per-case pass rate mean 1.0`
    beside it. A constant arm's po IS its pe, by algebra and not by coincidence: its po is
    the prevalence of its one class in gold, which is exactly what pe computes. So 0.000 is
    not a measurement of agreement with gold, it is a measurement of the arm's own
    constancy, and printing it is the failure this clause refuses.

    Here: 50 rows, gold flag on the 25 even indices, djev answers `flag` on every row,
    primary answers the gold label. κ for djev is 0.0 with po == pe == 0.5.
    """
    cell = _marginal_cell(lambda i: "flag" if i % 2 == 0 else "accept",
                          lambda i: "flag", rows_n=50)
    dje = cell["engine_kappa"]["djev"]
    assert dje["n"] == 50 and dje["n_classes"] == 1 and dje["labels"] == {"flag": 50}
    assert dje["degenerate"] is True and dje["kappa"] == 0.0
    assert dje["po"] == dje["pe"], (dje["po"], dje["pe"],
                                    "a constant arm's po is its pe by construction; if "
                                    "that stops holding, the degeneracy argument here is "
                                    "no longer the reason the number is uninformative")
    assert dje["uninformative"].startswith(X.DEGENERATE_LABEL), dje["uninformative"]
    assert cell["engine_kappa"]["primary"]["degenerate"] is False, (
        "the flag fired on an arm that emitted both labels, so it is not a degeneracy "
        "detector but a blanket refusal")

    text = X.markdown([cell], {"ran_at": "x", "repeats": X.MIN_REPEATS,
                              "engines": ENGINES, "verdict": "NEGATIVE"})
    assert f"κ vs {X.SCORED_GOLD_COLUMN} UNINFORMATIVE — {X.DEGENERATE_LABEL}" in text
    assert "κ vs gold_verdict 0.000" not in text, (
        "the degenerate arm's κ is back to printing as a figure a routing call could "
        "quote, which is the exact failure #3345 exists to close")
    assert "po 0.500, pe 0.500, n=50" in text, (
        "the refusal lost its margins: po on top of pe is the only evidence in the line "
        "that the refusal is right rather than a missing number")


def test_a_thin_cell_refuses_kappa_with_the_same_sentence_it_refuses_the_rates_in():
    """Clause 3: `retrieval-labels-offered` is this shape — 107 rows, primary answered 3
    and djev 26, and the table already prints UNMEASURABLE for its rates.

    A κ computed over that thin n would be the one number in the row that ignored the
    floor the whole table is built on, and 30 is the floor the module already ships
    (`MIN_NON_ABSTAINING`). So the κ cell repeats the rates' own sentence rather than
    inventing a second kind of refusal, and publishes its rating count while withholding
    the statistic — `n` is what lets a reader see the cell is thin.
    """
    rows = durable_rows(29)
    plan = {f"PROMPT-{i}": [flag("flag" if i % 2 == 0 else "accept")] for i in range(29)}
    cell = X.localize(X.run_rows(rows, Recorder(plan), X.MIN_REPEATS, ENGINES),
                      X.SURFACE_DURABLE_WRITE)
    assert cell["non_abstaining"] == 29 and not cell["measurable"]

    for eng in ("primary", "djev"):
        k = cell["engine_kappa"][eng]
        assert k["measurable"] is False and k["kappa"] is None, k
        assert k["n"] == 29, (
            "the ratings existed and the cell refused to average them, so the count is "
            "what a reader needs; a κ printed here would be a figure under the floor")
        assert k["uninformative"] == cell["unmeasurable"], (
            "kappa invented its own refusal wording, so the table now states two "
            "different reasons for the same missing number")
        assert "po" not in k and "pe" not in k, k

    text = X.markdown([cell], {"ran_at": "x", "repeats": X.MIN_REPEATS,
                              "engines": ENGINES, "verdict": "NEGATIVE"})
    assert text.count(f"κ vs gold_verdict {cell['unmeasurable']}") == 2, (
        "both engines' κ cells must carry the UNMEASURABLE sentence; one of them is "
        f"printing something else\n{text}")
    assert "κ vs gold_verdict 0." not in text and "κ vs gold_verdict 1." not in text


def test_every_surface_names_the_gold_column_it_scored_and_arm1_carries_its_caveat():
    """Clause 4: the column, on every cell and in `meta`, plus the licence refusal.

    The durable rows carry TWO gold columns of different vocabularies (`gold` = bad/good
    alongside the `gold_verdict` = flag/accept the run scores), and scoring the shipped
    2026-10-16 rows on the wrong one gives κ 0.0000 for both engines with po 0.0. Arm 1's
    gold is worse in a different way: Lloyd wrote it, and its own README says it "does not
    license" reopening #708 or flipping `automod.review.confirm` — so a κ of 1.000 there
    measures two families agreeing with a key Lloyd wrote, and must not read as a routing
    licence. `wires_nothing` already declines to wire the POSITIVE gate; this is the same
    refusal applied to the newer number.
    """
    durable = _marginal_cell(lambda i: "flag" if i % 2 == 0 else "accept",
                             lambda i: "flag" if i % 2 == 0 else "accept")
    arm1 = X.localize(_arm1_pair_rows(40), X.SURFACE_ARM1)
    assert arm1["measurable"] and arm1["non_abstaining"] == 40
    # Two CLOSE kappas, not one: primary 1.000 and djev (0.950 − 0.475) / 0.525 = 0.9048.
    # The gap the routing question would be asked about is 0.095, inside #3344's proposed
    # 0.1 — on gold Lloyd wrote itself, which is the case the caveat exists for.
    assert arm1["engine_kappa"]["djev"]["kappa"] == pytest.approx(0.9048, abs=1e-4), (
        arm1["engine_kappa"]["djev"])

    meta_named = X.scored_gold_meta([durable, arm1])
    assert meta_named[X.SURFACE_DURABLE_WRITE]["column"] == "gold_verdict"
    assert meta_named[X.SURFACE_DURABLE_WRITE]["labels"] == ["accept", "flag"]
    # Two labels in the scored column, and the third one the fixture's menu holds is
    # present only as a mistake: djev's `unrelated` on rows 1 and 3 moves its marginals.
    assert meta_named[X.SURFACE_ARM1]["labels"] == ["consistent", "injected"]
    assert all(v["kappa_basis"] == X.KAPPA_BASIS for v in meta_named.values())

    caveat = arm1["engine_kappa"]["primary"]["gold_caveat"]
    assert caveat and "Lloyd-authored" in caveat and "does not license" in caveat
    assert "rubric.py" in caveat and "#708" in caveat, caveat
    assert arm1["engine_kappa"]["djev"]["gold_caveat"] == caveat, (
        "the caveat covers one engine's κ and not the other's, so half the comparison "
        "still reads as a licence")
    assert durable["engine_kappa"]["primary"]["gold_caveat"] is None, (
        "the durable surface reuses a label another process wrote, so it carries no "
        "authorship caveat; printing one there would dilute the one that matters")

    text = X.markdown([arm1], {"ran_at": "x", "repeats": X.MIN_REPEATS,
                              "engines": ENGINES, "verdict": "NEGATIVE"})
    assert arm1["engine_kappa"]["primary"]["kappa"] == 1.0, (
        "arm 1's fixture κ stopped being perfect, so the caveat below is no longer being "
        "shown attached to a CLOSE kappa, which is the only case that could mislead")
    lines = text.splitlines()
    caveat_line = next(l for l in lines if "κ caveat:" in l)
    assert caveat in caveat_line, caveat_line
    # Order matters: the κ, the caveat and the POSITIVE verdict are three lines of one
    # block, and a reader who stops at the first two must still meet the caveat before the
    # sentence that starts "POSITIVE here" — the line a routing argument gets quoted from.
    assert lines.index(caveat_line) < next(i for i, l in enumerate(lines)
                                           if "POSITIVE here" in l)
    assert lines.index(caveat_line) > next(i for i, l in enumerate(lines)
                                           if l.startswith(f"- **{X.SURFACE_ARM1}**")), (
        "the caveat now precedes the κ it qualifies, which reads as a note about "
        "something else")


def _arm1_pair_rows(n: int) -> list[dict]:
    """Arm-1 rows through the real loader, parser and classifier, with gold varied.

    Built from `_arm1_gold_rows` — the file's arm-1-shaped generator, real rubric hash and
    artifact-naming warrant — because the loader validates the warrant and the parser
    validates the label menu; hand-assembled rows would grade a vocabulary arm 1 does not
    ship. Gold is varied here and nowhere else on this surface: `_arm1_gold_rows` keys
    EVERY row to one label, and under a constant gold the chance term pe collapses, so an
    arm's κ stops measuring agreement and starts measuring the fixture. Gold is
    `consistent` on the even indices and `injected` on the odd ones; primary answers the
    gold label on all n, and djev is wrong on rows 1 and 3, answering `unrelated`:

        primary  po = 40/40 = 1.000, pe = (20/40)(20/40) + (20/40)(20/40) = 0.500
                 κ   = (1.000 - 0.500) / 0.500 = 1.000
        djev     po = 38/40 = 0.950, pe = (20/40)(20/40) + (18/40)(20/40) = 0.475
                 κ   = (0.950 - 0.475) / 0.525 = 0.9048
    """
    gold_rows = _arm1_gold_rows(n)
    for i, g in enumerate(gold_rows):
        g["gold"] = "consistent" if i % 2 == 0 else "injected"
    rows = X.arm1_rows(gold_rows)
    plan: dict[str, dict[str, list[str]]] = {}
    for i, row in enumerate(rows):
        gold = row["gold_verdict"]
        plan.setdefault(ENGINES["primary"]["url"], {})[row["prompt"]] = [gold]
        plan.setdefault(ENGINES["djev"]["url"], {})[row["prompt"]] = [
            "unrelated" if i in (1, 3) else gold]
    return X.run_rows(rows, Recorder({}, engine_plan=plan), X.MIN_REPEATS, ENGINES)


def test_a_published_run_names_the_scored_column_and_prints_every_kappa(tmp_path, capsys):
    """Clauses 1 and 4 across the one boundary this module owns: argv in, artifact out.

    The nodes above call `localize` / `markdown` directly, which is the right level for the
    arithmetic and the wrong level for "does the published file carry it": a κ computed and
    then dropped on the way to disk would satisfy every one of them. So this node runs
    `main` exactly as the nightly does, over the SHIPPED durable corpus and recorded
    verdicts, with the file's own always-`flag` transport — the same constant-arm shape
    #3345 was filed over, now supplied to both engines, so both must be refused.

    No arithmetic is asserted here: the corpus's gold marginals belong to the shipped
    #2254 samples and this node must not restate numbers it did not compute. The hand-
    computed figures live in the known-marginals node above; what is pinned below is that
    the column is named, the refusals are published, and the basis line states the
    denominator.
    """
    out = tmp_path / "cross-engine"
    assert X.main(["--out", str(out), "--arms", X.SURFACE_DURABLE_WRITE],
                  transport=ALWAYS_FLAG, engines=ENGINES) == 0
    art_path = next(out.glob("cross-engine-disagreement-*.json"))
    art = json.loads(art_path.read_text())
    md = art_path.with_suffix(".md").read_text()

    named = art["meta"]["scored_gold_column"][X.SURFACE_DURABLE_WRITE]
    assert named["column"] == X.SCORED_GOLD_COLUMN == "gold_verdict", named
    assert named["labels"] == ["accept", "flag"], (
        f"the published vocabulary of the scored column changed: {named['labels']}")

    stored = art["surfaces"][0]["engine_kappa"]
    assert set(stored) == set(ENGINES), sorted(stored)
    for eng, k in stored.items():
        assert k["n"] == 50 and k["n_classes"] == 1 and k["degenerate"] is True, (eng, k)
        assert k["uninformative"].startswith(X.DEGENERATE_LABEL), (eng, k)
    assert art["surfaces"][0]["non_abstaining"] == 50 and art["surfaces"][0]["measurable"], (
        "the cell stopped being measurable, so the refusals below would be the thin-n "
        "kind and this node would be pinning the wrong one")

    κ = next(l for l in md.splitlines() if l.startswith(f"- **{X.SURFACE_DURABLE_WRITE}**"))
    assert κ.count(f"κ vs gold_verdict UNINFORMATIVE — {X.DEGENERATE_LABEL}") == 2, κ
    assert "κ vs gold_verdict 0." not in κ and "κ vs gold_verdict 1." not in κ, κ
    assert (f"κ basis: {X.KAPPA_BASIS} = 50 of 50 rows; scored gold column "
            "gold_verdict = accept/flag") in md, (
        "the table stopped stating what a κ was computed over and against which column, "
        "leaving the figure a bare number with no denominator beside it")

    # The runner's stdout is what a human reads first and what a re-run gets compared
    # against, so the scored column travels there too — never as a number this time.
    printed = capsys.readouterr().out
    seg = re.search(r"kappa_vs_gold_verdict=\{([^}]*)\}", printed)
    assert seg, f"stdout no longer names the scored column it scored: {printed}"
    assert seg.group(1).count("UNINFORMATIVE-degenerate") == 2, seg.group(1)
    assert not re.search(r"\d\.\d", seg.group(1)), (
        f"stdout is back to publishing a number for the constant-label arms: "
        f"{seg.group(1)} — the table refuses it and the terminal is the line people copy")
