"""#2254 — the four-arm attribution probe is byte-identical, audited and honest.

`eval/judge_self_preference/` re-scores the shipped 50-note corpus under four provenance
claims that differ in one line, to measure whether an LLM judge prefers text it believes
came from its own model family. Kumar's demo (AI Engineer, 2026-10-05) shows the effect;
what is unknown is its size on a 50-sample pointwise pass/fail judge, which is what the
run will report. Nothing here calls a model, so these tests are offline and deterministic
— the ~200-call pass and its interpretation are the post-landing step (the item's own
stop rule: if the attribution effect does not exceed the `repeat` floor, that negative
result is the finding).

Two properties carry the whole instrument and get proven rather than asserted in prose:

* **Byte-identity.** If anything but the attribution line varied between arms, a flip
  would be evidence about the arm and about whatever else moved. The same shape
  `tests/test_durable_write_judge.py:158` enforces for the two #580 judges, extended
  from three parts to four, and checked over all 50 samples rather than one.
* **A denominator that cannot move.** A row that is not a verdict (`unparsed`, `error`)
  must stop the run naming the sample. Half the rates on the report page are ratios over
  the sample set, so one silently-absent row changes every one of them at once. That is
  the zero-denominator class rule #580 already applies to its own run, applied here
  before the percentages exist rather than after.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.judge_self_preference import arms as A          # noqa: E402
from eval.judge_self_preference import report as R        # noqa: E402
from eval.judge_self_preference import run_arms as RUN    # noqa: E402
from eval.durable_write_judge.judge import SAMPLE_OPEN    # noqa: E402

CORPUS = ROOT / "eval" / "durable_write_judge" / "corpus.jsonl"
PROBE_DIR = ROOT / "eval" / "judge_self_preference"


@pytest.fixture(scope="module")
def samples() -> list[dict]:
    return A.load_corpus(CORPUS)


@pytest.fixture(scope="module")
def prompts(samples) -> list[A.ArmPrompt]:
    return A.generate(samples)


@pytest.fixture(scope="module")
def by_sample(prompts) -> dict[str, dict[str, A.ArmPrompt]]:
    out: dict[str, dict[str, A.ArmPrompt]] = {}
    for p in prompts:
        out.setdefault(p.sample_id, {})[p.arm] = p
    return out


# ── clause 1 — four arms over the real corpus, one line apart, reproducible ─────

def test_all_fifty_corpus_samples_get_all_four_arms(samples, prompts, by_sample):
    """200 rows, because an arm missing from some samples would silently shrink the
    flip denominator for that arm alone — different samples in, different rate out."""
    assert len(samples) == 50, (
        f"the probe is specified over the shipped 50-note corpus, found {len(samples)} "
        f"rows in {CORPUS} — re-scope the item rather than reporting a rate over a "
        "different corpus than #580 published its 45.0 % on")
    assert [s["label"] for s in samples].count("bad") == 20
    assert len(prompts) == 50 * len(A.ARMS)
    assert set(by_sample) == {s["id"] for s in samples}
    for sid, arms in by_sample.items():
        assert set(arms) == set(A.ARMS), f"{sid} is missing an arm: {sorted(arms)}"


def test_the_arms_of_one_sample_differ_only_in_the_attribution_line(by_sample, samples):
    """Byte-identity across three parts, proved per sample over all 50.

    `before` and `after` must be the same bytes in all four arms and the attribution
    line must be the only difference between them, which is a stronger claim than
    diffing two whole prompts and noticing one changed line: it also rules out an arm
    that moved the note, the rubric preamble, the five labelled examples or the JSON
    output contract. The note itself is asserted to sit in `after`, so the attribution
    cannot be smuggled into the text being judged.
    """
    for sid, arms in by_sample.items():
        base = arms["none"]
        assert arms["repeat"].text == base.text, (
            f"{sid}: `repeat` is defined as the baseline prompt sent again; any byte "
            "difference makes its flip rate a treatment effect, not a noise floor")
        for arm in ("self", "human"):
            assert arms[arm].before == base.before, f"{sid}/{arm}: preamble or examples moved"
            assert arms[arm].after == base.after, (
                f"{sid}/{arm}: the note or the output contract moved — the only "
                "permitted difference is the tail of the attribution line")
            assert arms[arm].line.startswith(A.ATTRIBUTION_PREFIX)
            assert arms[arm].line != base.line, f"{sid}/{arm}: attribution did not differ"
            rebuilt = arms[arm].before + arms[arm].line + arms[arm].after
            assert rebuilt == arms[arm].text, f"{sid}/{arm}: parts do not reassemble"
        # The attribution is metadata about the note, not another sentence of it: the
        # line sits at the tail of `before`, and the note being judged is entirely in
        # `after` — an arm that put its claim inside the note would be editing evidence.
        assert base.after.startswith(SAMPLE_OPEN), (
            f"{sid}: everything after the attribution line is #580's own note-and-output-"
            "contract half, so the claim cannot be sitting inside the note being judged")


def test_the_three_attribution_texts_are_distinct_and_both_treatments_are_named():
    assert A.BASELINE_ARM == "none"
    assert list(A.ATTRIBUTIONS) == ["none", "self", "human"]
    assert len(set(A.ATTRIBUTIONS.values())) == 3, (
        f"two arms share wording, so they are one arm: {A.ATTRIBUTIONS}")
    assert A.ARMS == ("repeat", "none", "self", "human")
    assert "qwen" in A.ATTRIBUTIONS["self"].lower(), (
        "the self arm must name the model family — that is the axis Kumar's rule is on")
    assert "human" in A.ATTRIBUTIONS["human"].lower()


def test_regenerating_the_arms_reproduces_them_byte_for_byte(samples, prompts):
    """A rerun must emit the same 200 hashes, or the run cannot be reproduced from
    its own artifact — and the seeded example draw is what makes that true."""
    again = A.generate(samples)
    assert [p.sha256 for p in again] == [p.sha256 for p in prompts], (
        "the example draw or the preamble moved between calls; #580's SEED=580 rule is "
        "what keeps a rerun comparable to the run it is rerunning")
    rows = A.prompt_rows(prompts)
    assert all(r["prompt_chars"] == len(r["text"]) for r in rows)
    assert all(len(r["prompt_sha256"]) == 64 for r in rows)


def test_the_generator_main_writes_the_same_bytes_twice(tmp_path):
    out1, out2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    assert A.main(["--corpus", str(CORPUS), "--out", str(out1)]) == 0
    assert A.main(["--corpus", str(CORPUS), "--out", str(out2)]) == 0
    assert out1.read_bytes() == out2.read_bytes(), (
        "regeneration is not byte-reproducible even through the committed entry point")
    rows = [json.loads(ln) for ln in out1.read_text().splitlines()]
    assert len(rows) == 200


# ── clause 2 — audited phrasings, and a rail that actually bites ────────────────

def test_both_attribution_strings_and_the_forbidden_token_list_are_committed():
    assert A.FORBIDDEN_HINT_TOKENS == ("fine", "ok", "correct", "approved",
                                       "checked", "good"), (
        f"the audited token list is {A.FORBIDDEN_HINT_TOKENS}; the clause names the six "
        "words, so adding or dropping one is a scope change, not a fix")
    assert A.audit_attribution_text() == [], (
        f"committed phrasings carry a verdict hint: {A.audit_attribution_text()}")


def test_the_audit_rejects_a_phrasing_that_hints_at_the_expected_verdict(monkeypatch):
    """The rail has to bite, or it is a comment with a name.

    'a human editor checked this and it is fine' is the exact confound the item's Risks
    section names: it reads as a hint about the verdict, not as provenance. A constant
    list that no code reads would pass clause 2 while permitting precisely that.
    """
    monkeypatch.setitem(A.ATTRIBUTIONS, "human",
                        "written by a human editor who checked this and it is fine")
    offenders = A.audit_attribution_text()
    assert offenders, "the audit let a hinting phrasing through"
    assert any(o.startswith("attribution[human]") for o in offenders), offenders
    assert "ok" not in [t for t in A.FORBIDDEN_HINT_TOKENS if t == ""]


def test_the_runner_refuses_an_unaudited_phrasing_before_any_call(monkeypatch):
    calls: list[str] = []
    monkeypatch.setitem(A.ATTRIBUTIONS, "self",
                        "generated by the same Qwen family and approved by its author")
    rc = RUN.main(["--corpus", str(CORPUS), "--raw", "/tmp/2254-never.raw",
                   "--report", "/tmp/2254-never.md"],
                  call=lambda *a: calls.append(a) or ('{"verdict":"flag"}', "stop"))
    assert rc == 1
    assert calls == [], "the runner spent engine calls on phrasings it had refused"
    assert not Path("/tmp/2254-never.md").exists()


# ── clause 3 — an unjudged row stops the run and no report is written ───────────

def _reply(verdict: str = "flag") -> tuple[str, str]:
    return json.dumps({"verdict": verdict, "defect_class": None,
                       "reason": "fixture reply"}), "stop"


def test_a_run_with_one_unparsed_row_exits_nonzero_naming_it_and_writes_no_report(
        tmp_path, capsys):
    """One arm of one sample answers with prose instead of JSON.

    The raw rows must still land — the evidence of a failed run is what you fix — and
    the report must not: `silent-pass` over 199 judged rows printed under a heading
    that says 200 is a false number, not a slightly-off one.
    """
    raw, rep = tmp_path / "raw.jsonl", tmp_path / "report.md"
    poisoned: list[str] = []

    def call(prompt_text, url, model, temperature=0):
        if "source of this note: written by a human editor" in prompt_text:
            if len(poisoned) < 1:
                poisoned.append(prompt_text[:80])
                return ("I think this note looks reasonable overall.", "stop")
        return _reply("accept" if "good" not in prompt_text else "flag")

    rc = RUN.main(["--corpus", str(CORPUS), "--raw", str(raw), "--report", str(rep)],
                  call=call)
    captured = capsys.readouterr()      # one call: reading twice drains the first half
    err = captured.out + captured.err
    assert rc == 1, "an unjudged row must fail the run"
    rows = [json.loads(ln) for ln in raw.read_text().splitlines()]
    assert len(rows) == 200, "the raw evidence must survive the refusal"
    bad = [r for r in rows if r["verdict"] not in R.JUDGED]
    assert len(bad) == 1 and bad[0]["verdict"] == "unparsed"
    assert bad[0]["sample_id"] in err, (
        f"the refusal must name the offending sample; said: {err[:400]}")
    assert not rep.exists(), "a report was written over an unjudged row"


def test_the_renderer_also_refuses_a_row_that_is_not_a_verdict(samples):
    """Same rule at the second boundary, so a hand-fed rows file cannot skip it."""
    rows = [{"sample_id": s["id"], "arm": a,
             "verdict": "flag" if s["label"] == "bad" else "accept"}
            for s in samples for a in A.ARMS]
    rows[3]["verdict"] = "unparsed"
    with pytest.raises(ValueError) as exc:
        R.compute(samples, rows)
    assert rows[3]["sample_id"] in str(exc.value)
    assert "unparsed" in str(exc.value)


# ── clause 4 — the report prints the promised numbers, against the right baseline ─

@pytest.fixture(scope="module")
def fixture_rows(samples) -> list[dict]:
    """A synthetic verdict set with every number in it known before the code runs.

    `none` flags all 20 bad and accepts all 30 good: silent-pass 0/30 = 0.0 %.
    `repeat` is identical, so its flip rate is 0.0 % — the floor.
    `self` accepts the first bad note carrying `invented_url` and the first carrying
    `omitted_numerical_data`, and flags one `good` note that `none` accepted: accepted
    becomes 30 - 1 + 2 = 31 and silent-pass 2/31 = 6.5 %, so Δ vs `none` is +6.5 pts on
    a run whose own baseline is 0.0 %.
    `human` is identical to `none`: flip rate 0.0 %, and its five class counts are all 0,
    which is the shape clause 4 asks the report to print rather than omit.
    """
    bad = [s for s in samples if s["label"] == "bad"]
    good = [s for s in samples if s["label"] == "good"]
    url_bad = next(s for s in bad if "invented_url" in s["defect_classes"])
    num_bad = next(s for s in bad if "omitted_numerical_data" in s["defect_classes"])
    flipped_good = good[0]
    accepted_by = {"none": {s["id"] for s in good},
                   "repeat": {s["id"] for s in good},
                   "human": {s["id"] for s in good},
                   "self": ({s["id"] for s in good} - {flipped_good["id"]}
                            | {url_bad["id"], num_bad["id"]})}
    return [{"sample_id": s["id"], "arm": arm,
             "verdict": "accept" if s["id"] in accepted_by[arm] else "flag",
             "defect_class": None, "reason": "fixture"}
            for s in samples for arm in A.ARMS]


def test_the_report_prints_every_rate_the_probe_promises(samples, fixture_rows):
    text = R.render(samples, fixture_rows, url="http://127.0.0.1:8080/v1/chat/completions",
                    model="Qwen3.8-Flash-Next")
    machine = {ln.split(" ", 1)[0]: ln for ln in text.splitlines() if "=" in ln}

    assert machine["flip_vs_none_pct"].endswith("repeat=0.0 self=6.0 human=0.0"), (
        f"repeat must be 0.0 (identical prompt) and self 2 flips of 50 = 6.0 %: "
        f"{machine['flip_vs_none_pct']}")
    assert machine["pass_rate_on_bad_pct"].endswith("none=0.0 self=10.0 human=0.0"), (
        machine["pass_rate_on_bad_pct"])
    assert "none=0.0" in machine["silent_pass_pct"] and "self=6.5" in machine["silent_pass_pct"], (
        f"2 silent passes of 31 accepted is 6.5 %: {machine['silent_pass_pct']}")
    assert machine["delta_silent_pass_vs_none_pts"] == \
        "delta_silent_pass_vs_none_pts repeat=+0.0 self=+6.5 human=+0.0", (
        "the comparator is this run's own `none` arm at 0.0 %, so `repeat` and `human` "
        "delta nothing and only `self` moves: "
        f"{machine['delta_silent_pass_vs_none_pts']}")
    assert "engine=http://127.0.0.1:8080/v1/chat/completions" in text
    assert "model=Qwen3.8-Flash-Next" in text
    assert "samples=50 n_bad=20 n_good=30 rows=200 temperature=0" in text


def test_the_report_prints_per_class_flip_counts_including_the_classes_at_zero(
        samples, fixture_rows):
    text = R.render(samples, fixture_rows, url="u", model="m")
    lines = [ln for ln in text.splitlines() if ln.startswith("class_flips arm=")]
    assert len(lines) == 3, f"one machine line per non-baseline arm: {lines}"
    for ln in lines:
        assert "false_cutoff_claim=" in ln and "missing_section=" in ln \
            and "wrong_metadata=" in ln and "invented_url=" in ln \
            and "omitted_numerical_data=" in ln, (
            f"all five rubric classes must be named even at zero: {ln}")
    self_line = next(ln for ln in lines if "arm=self" in ln)
    assert "invented_url=1" in self_line and "omitted_numerical_data=1" in self_line
    assert "false_cutoff_claim=0" in self_line and "missing_section=0" in self_line
    human_line = next(ln for ln in lines if "arm=human" in ln)
    assert human_line.split(" ", 1)[1].count("=0") == 5, (
        f"a no-flip arm must print five zeros: {human_line}")
    assert "| false_cutoff_claim | 2 | 0 | 1 | 0 |" in text or \
        "| false_cutoff_claim | 2 |" in text, (
        "the human-readable class table must carry the class with its corpus count")


def test_the_published_32_4_is_labelled_as_a_different_model_not_the_comparator(
        samples, fixture_rows):
    text = R.render(samples, fixture_rows, url="u", model="Qwen3.8-Flash-Next")
    assert "32.4" in text and "Qwen3.6-35B-A3B" in text, (
        "the published figure must appear with the model that produced it")
    assert "comparable_to_this_run=no" in text, (
        "the report must say outright that the published number is not this run's "
        "comparator: the secondary engine that produced it has been off since "
        "2026-09-20 (config.yaml:37), so a delta against it moves the judge model too")
    assert "eval/durable_write_judge/report.md:32" in text
    assert "optimistic" in text, (
        "#580's caveat is on disk (report.md:113) and applies to every silent-pass "
        "figure printed here")


def test_the_runner_end_to_end_writes_a_report_when_every_row_is_a_verdict(tmp_path):
    """The runner→renderer boundary, with no engine between them: a stub stands in for
    the model, and what is asserted is that the committed path produces a report whose
    machine lines name the engine and model it was told to use."""
    raw, rep = tmp_path / "raw.jsonl", tmp_path / "report.md"
    rc = RUN.main(["--corpus", str(CORPUS), "--raw", str(raw), "--report", str(rep),
                   "--url", "http://127.0.0.1:8080/v1/chat/completions",
                   "--model", "Qwen3.8-Flash-Next"],
                  call=lambda *a: _reply("flag"))
    assert rc == 0
    assert rep.exists()
    text = rep.read_text()
    assert "engine=http://127.0.0.1:8080/v1/chat/completions" in text
    assert "model=Qwen3.8-Flash-Next" in text
    rows = [json.loads(ln) for ln in raw.read_text().splitlines()]
    assert len(rows) == 200 and {r["url"] for r in rows} == \
        {"http://127.0.0.1:8080/v1/chat/completions"}


# ── clause 4, second half — which engine a real run would reach ───────────────────

def test_the_unregistered_job_resolves_to_the_engine_that_answers():
    """Measured, not assumed: the job map defaults an unknown job to the DEAD engine.

    `judge_self_preference` is in no `decisions.yaml` and cannot be added to
    `JOBS_ON_PRIMARY` by hand (`app/secondary_models.py:64-70` — that set is written by
    `eval/secondary_routing_eval.py --pin` and checked by
    `tests/test_secondary_routing_eval.py`), so `_engine_for()` answers `secondary`
    (`app/secondary_models.py:82`) — the :8091 that has answered nothing since
    2026-09-20. What saves the run is `resolve_model_alias` rewriting `secondary` to
    `primary` while `secondary_enabled: false` (`app/config.py:192-199`, the #1445
    rule). Pin both halves: the answer is the live engine, AND the reason is an
    override rather than a default naming primary. A future `keep_secondary`
    registration, or the secondary re-enabled beside `djev`, flips the second half and
    this node fails — which is the moment a ~200-call run would silently start
    measuring a different judge than its own `none` arm, or stop measuring at all.
    """
    from app import secondary_models as SM

    choice = RUN.resolve_endpoint(None, None)
    assert choice.asked_engine == "secondary", (
        "the default changed under this probe: if an unknown job no longer defaults to "
        f"the secondary, {RUN.JOB!r} is being routed somewhere nobody named")
    assert SM._engine_for(RUN.JOB) == "secondary"
    assert choice.served_engine == "primary" and choice.rewritten, (
        f"the #1445 override stopped firing; the probe would run on {choice.describe()}")
    assert "8091" not in choice.url, (
        "the resolver handed back the dead secondary port; every one of the ~200 calls "
        "would return nothing and the run would have no verdicts to refuse over")
    assert choice.url == SM._endpoint(RUN.JOB)[0]


def test_the_report_header_records_the_engine_and_its_provenance(samples, fixture_rows):
    """Engine and model in the header, and asked-vs-served beside them.

    The header is the first line a reader of a report sees and the report is the only
    durable record of which model produced these verdicts; `asked_engine`/`served_engine`
    are kept as two fields because collapsing them into one word `engine` is what hid
    #1445. Rendered without them (a hand-built report) the header still names url and
    model and says nothing about routing — no invented provenance."""
    text = R.render(samples, fixture_rows, url="http://127.0.0.1:8096/v1/chat/completions",
                    model="Qwen3.8-Flash-Next", engine_asked="secondary",
                    engine_rewritten=True)
    head = text.split("## ")[0]
    assert "http://127.0.0.1:8096/v1/chat/completions" in head and "Qwen3.8-Flash-Next" in head
    assert "asked_engine=secondary" in head and "served_engine=Qwen3.8-Flash-Next" in head
    bare = R.render(samples, fixture_rows, url="u", model="M")
    assert "asked_engine" not in bare, "a report must not claim routing it was not given"


# ── clause 5 — offline pin ──────────────────────────────────────────────────────

#: The shape that would actually break offline-ness: an import, in code, somewhere else.
_IMPORT_NEEDLES = ("from eval.judge_self_preference", "import eval.judge_self_preference",
                   "import judge_self_preference", "from judge_self_preference")


def test_nothing_outside_eval_and_tests_imports_the_probe():
    """The same offline pin #580 clause 6 holds over `durable_write_judge`, re-run for
    this package — but pinned on the *import*, not on the name being mentioned.

    The distinction is not pedantry, it is this round's own history: the first version
    greped the bare string and stayed green only until `architecture/measurement.md`
    gained the inventory row that `tests/test_architecture_coverage_doc_claims.py`
    REQUIRES of every `eval/*` arm. A doc listing an arm is the arm being discoverable;
    a live path importing it is the arm being online. Pin the second, or the two tests
    cannot both pass and one of them gets weakened by the next person who adds a row.
    """
    for needle in _IMPORT_NEEDLES:
        out = subprocess.run(
            ["git", "grep", "-l", "-F", needle, "--", "*.py",
             ":!eval/judge_self_preference", ":!tests/test_judge_self_preference.py"],
            cwd=ROOT, capture_output=True, text=True).stdout
        assert out.strip() == "", f"{needle!r} imported outside eval/ and tests/:\n{out}"

    mentions = subprocess.run(
        ["git", "grep", "-l", "-F", "judge_self_preference", "--",
         ":!eval/judge_self_preference", ":!tests/test_judge_self_preference.py"],
        cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert set(mentions) <= {"architecture/measurement.md"}, (
        "the name may appear in the arm inventory and nowhere else; anything more is "
        f"either a second doc to keep honest or a caller: {mentions}")
    assert "architecture/measurement.md" in mentions, (
        "positive control: the inventory row is what makes this grep non-empty at all — "
        "if it vanished, the assertion above would be satisfied by there being nothing "
        "to find")
    assert PROBE_DIR.is_dir()


def test_the_probe_ships_the_three_files_the_contract_names():
    """Generator, runner, renderer — and no second corpus. The probe re-scores #580's
    own 50 notes, which is the only reason its rates are comparable to the published
    45.0 %/32.4 % at all; a corpus of its own would be a second labelled set to defend."""
    for name in ("arms.py", "run_arms.py", "report.py"):
        assert (PROBE_DIR / name).is_file(), name
    assert not list(PROBE_DIR.glob("corpus*")), (
        "the probe shipped its own corpus instead of reusing durable_write_judge's")
