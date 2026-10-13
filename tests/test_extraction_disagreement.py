"""Tests for `eval/extraction_disagreement.py` — backlog item #2656.

The module is a measurement instrument, so the properties pinned here are the
ones that make its number mean something:

1. the sample is drawn only from the allow-listed extraction dirs, covers each
   one, and is redrawn identically from its recorded seed;
2. pass B is sent byte-identical prompt bytes and the same chunk windows as
   pass A, gets none of pass A's output, and a pair whose bytes or offsets
   differ is refused rather than reported;
3. the diff key is (canonical entity, category, canonical text) with entities
   through `app/entity_naming`'s alias map, and a canonical-text match with
   different wording is a rewording, not a difference;
4. every difference lands in exactly one of four settlement labels with a
   supporting span from the document, and anything unplaceable is `unsettled`
   rather than folded into agreement;
5. the artifact carries per-document disagreement rate, the four-way counts,
   defect yield and per-document cost measured by this module, and a run writes
   only inside its `--out` directory.

Everything runs offline: an injected fake of the extractor module and an injected
`sender`/`judge` stand in for the engine, so no test opens a socket. The real
extractor is loaded in exactly one test, to pin that the prompts are built from
its own template and chunker.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOD_PATH = ROOT / "eval" / "extraction_disagreement.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("extraction_disagreement", MOD_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


X = _load_module()

#: The six directories the production extraction is allowed to read
#: (`scripts/memory/next-gen-memory/pipeline_config.yaml:25-32`). Named here as
#: the *expected* value so a silent widening of the allow-list fails this file
#: rather than quietly widening the audit too.
SIX = ["knowledge", "projects", "people", "personal", "work", "memory"]
NOT_ALLOWED = ["skills", "backlog", "architecture", "autonomy", "agents", "facts",
               "sessions", "youtube"]


def make_vault(root: Path, dirs: list[str], per_dir: int = 8, body: int = 900) -> Path:
    """A vault whose allow-listed dirs each hold `per_dir` documents, plus one
    file in every directory the allow-list leaves out — because the only way to
    prove the sampler excludes them is to offer them to it."""
    vault = root / "obsidian"
    for d in dirs:
        for i in range(per_dir):
            # `memory/` holds daily notes, and the job skips one while its name is
            # today's date or later (`nightly_extraction.py:623`) — a stem like
            # "doc-0" sorts AFTER a date string, so undated fixtures would make the
            # whole directory ineligible and the draw would refuse to run.
            name = f"2026-01-{i:02d}.md" if d == "memory" else f"doc-{i}.md"
            p = vault / d / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"# {d} note {i}\n\n" + (f"{d} fact sentence {i}. " * body),
                         encoding="utf-8")
    for d in NOT_ALLOWED:
        p = vault / d / "trap.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# trap\n\n" + "never extracted by the nightly job. " * 200,
                     encoding="utf-8")
    return vault


@pytest.fixture
def vault(tmp_path):
    return make_vault(tmp_path, SIX)


def config_with(dirs: list[str], tmp_path: Path) -> Path:
    # The real key shape: `sources.paths` holds vault directories and the audit
    # derives the names from them (`nightly_extraction.py:576`). A fixture writing
    # some other key would be testing a config format nothing parses.
    cfg = tmp_path / "pipeline_config.yaml"
    vault = tmp_path / "obsidian"
    cfg.write_text("extraction:\n  max_file_size: 5242880\n"
                   "sources:\n  paths:\n"
                   + "".join(f'    - "{vault / d}"\n' for d in dirs)
                   + "  exclude_patterns:\n    - '.git/'\n    - '__pycache__/'\n",
                   encoding="utf-8")
    return cfg


# ── clause 1: the sample ────────────────────────────────────────────────────

def test_the_sample_is_drawn_only_from_the_allow_listed_dirs_and_covers_each_one(vault, tmp_path):
    """Six dirs are allow-listed, so a 42-document draw is 7 each; the eight
    directories the allow-list omits are present in the fixture and may not be
    drawn at all, because auditing a document the write path never reads makes
    the rate non-transferable to it."""
    sample = X.sample_documents(vault, 42, "s1", dirs=SIX)
    assert len(sample["paths"]) == 42
    assert set(p.split("/")[0] for p in sample["paths"]) == set(SIX)
    assert all(not p.split("/")[0] in NOT_ALLOWED for p in sample["paths"])
    assert set(sample["dir_drawn"].values()) == {7}, sample["dir_drawn"]
    # The excluded directories are not walked at all, so they do not appear as
    # candidate pools: `"skills": 0` would mean the sampler looked into them and
    # declined, which is a different and weaker guarantee than never going there.
    assert "skills" not in sample["dir_candidates"], sample["dir_candidates"]
    assert set(sample["dir_candidates"]) == set(SIX)


def test_a_sample_smaller_than_the_directory_count_still_places_one_per_directory(vault, tmp_path):
    """`--sample-size 6` over six dirs is one each, and the per-dir floor is what
    stops a single huge directory (the vault's `memory/` holds most of its files)
    from being the whole audit."""
    sample = X.sample_documents(vault, 6, "s2", dirs=SIX)
    assert sorted(p.split("/")[0] for p in sample["paths"]) == sorted(SIX)


def test_the_recorded_seed_and_paths_redraw_the_identical_sample(vault):
    """Same seed → same paths. Different seed → a different draw, which is what
    makes "re-runnable" mean reproducibility and not just determinism-by-accident
    on a one-file pool."""
    a = X.sample_documents(vault, 12, "seed-A", dirs=SIX)
    b = X.sample_documents(vault, 12, "seed-A", dirs=SIX)
    c = X.sample_documents(vault, 12, "seed-B", dirs=SIX)
    assert a["paths"] == b["paths"] and a["seed"] == b["seed"] == "seed-A"
    assert c["paths"] != a["paths"]


def test_the_allow_list_is_read_from_the_pipeline_config_not_from_a_literal(vault, tmp_path):
    """The dir list comes from the same YAML the nightly job reads: a copy of it
    in this module would go stale silently the day the corpus moved, and the
    audit would keep measuring the old corpus while claiming to measure the new."""
    cfg = config_with(SIX, tmp_path)
    assert X.allow_listed_dirs(cfg, vault) == SIX
    narrow = config_with(["knowledge"], tmp_path)
    assert X.allow_listed_dirs(narrow, vault) == ["knowledge"]
    empty = tmp_path / "empty.yaml"
    empty.write_text("extraction:\n  max_file_size: 1\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        X.allow_listed_dirs(empty, vault)


# ── clause 2: the pair is the same question ─────────────────────────────────

class FakeFE:
    """The pieces of `fact_extractor.FactExtractor` the module actually calls.
    The real class's `__init__` opens the facts tree, so the module reaches the
    two pure methods unbound — which is also why a fake is enough to test all of
    the pairing logic offline."""

    CHUNK_BUDGET_CHARS = 3000
    SCHEMA_TYPES = ["person", "project", "system"]
    EXTRACTION_PROMPT = ("Extract facts.\nKnown entities:\n{known_entities}\n"
                         "Existing facts:\n{existing_facts}\nTypes: {entity_types}\n"
                         "---\n{content}\n")

    class FactExtractor:
        @staticmethod
        def _chunk_content(_self, content):
            step = 1000
            return [(i, content[i:i + step]) for i in range(0, len(content), step)]

        @staticmethod
        def _parse_response(_self, text):
            try:
                data = json.loads(text)
            except Exception:  # noqa: BLE001
                return {"facts": []}
            return {"facts": data.get("facts") or []}

    @staticmethod
    def _known_entities(text, limit):
        return ["Lloyd"]


def _fake_fe():
    return FakeFE()


def _sender(facts_by_pass):
    """A model that answers with the facts it was told to, per pass.

    Which pass a call belongs to is decided by the PROMPT, not by a call counter: a
    document's chunks are one call each, so a counter would hand pass A the second
    element from its second chunk onwards and every pair in these tests would be a
    mixture. The first time a prompt is seen is pass A; the second time the same
    prompt comes round is pass B — which is exactly what a paired audit is.
    """
    calls = {"n": 0, "bodies": [], "_seen": []}

    def send(prompt, port):
        calls["n"] += 1
        if prompt in calls["_seen"]:
            idx = 1
        else:
            calls["_seen"].append(prompt)
            idx = 0
        facts = facts_by_pass[min(idx, len(facts_by_pass) - 1)]
        body = json.dumps({"facts": facts}, sort_keys=True)
        calls["bodies"].append((prompt, port, body))
        usage = {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70,
                 "seconds": 0.5,
                 # What a real transport digests is the bytes it posts; the fake
                 # digests the prompt it was handed, so a sender that edits the
                 # prompt before "posting" it is caught by `pair_guard`.
                 "sent_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        return body, usage, 0.5
    return send, calls


FACT_A = [{"entity": "Lloyd", "category": "state", "fact": "The queue runs nightly."}]
FACT_B = [{"entity": "Lloyd", "category": "state", "fact": "The queue runs nightly at two."}]


def test_pass_b_gets_the_same_prompt_bytes_and_offsets_and_none_of_pass_a_output():
    """Two calls, identical prompt text and identical offsets, and the second is
    not given the first's facts — the definition of an independent pair."""
    fe = _fake_fe()
    # Windows must DIFFER: identical chunk texts make two chunks share one prompt,
    # and the fake transport keys a call's pass on its prompt.
    content = "".join(str(n % 7) for n in range(2200))
    prompts = X.build_prompts(fe, content)
    send, calls = _sender([FACT_A, FACT_B])
    a = X.run_pass(fe, prompts, 8096, "A", send)
    b = X.run_pass(fe, prompts, 8096, "B", send)
    prompts_sent = [c[0] for c in calls["bodies"]]
    assert prompts_sent == [p["text"] for p in prompts] * 2
    assert a["offsets"] == b["offsets"] == [[p["offset"], p["chunk_chars"]] for p in prompts]
    assert a["prompt_sha256"] == b["prompt_sha256"] == X.prompt_digest(prompts)
    X.pair_guard(prompts, a, b)          # the pair is reportable
    # Every fact pass A holds came from pass A's answer and none of it is a fact
    # pass B said: pass B's output reaching pass A's set would be the leakage that
    # makes a pair one pass twice.
    assert {f["fact"] for f in a["facts"]} == {f["fact"] for f in FACT_A}, a["facts"]
    assert not ({f["fact"] for f in a["facts"]} & {f["fact"] for f in FACT_B})


def test_a_pair_whose_prompt_bytes_or_chunk_windows_differ_is_refused_not_reported():
    """The instrument's own failure mode: two passes of the same weights are only
    a lower bound on the error rate, so a pair that was asked different questions
    is not even that — it is noise with a decimal point. Both ways it happens are
    pinned: a prompt that moved (digest) and a document that was chunked
    differently (offsets)."""
    fe = _fake_fe()
    prompts = X.build_prompts(fe, "y" * 2200)
    send, _ = _sender([FACT_A, FACT_B])
    a = X.run_pass(fe, prompts, 8096, "A", send)
    b = X.run_pass(fe, prompts, 8096, "B", send)

    moved = dict(b, prompt_sha256=hashlib.sha256(b"other prompt").hexdigest())
    with pytest.raises(X.PairNotComparable) as exc:
        X.pair_guard(prompts, a, moved)
    assert "prompt digest" in str(exc.value)

    reshaped = dict(b, offsets=[[p[0], p[1] + 1] for p in b["offsets"]])
    with pytest.raises(X.PairNotComparable) as exc:
        X.pair_guard(prompts, a, reshaped)
    assert "chunk offsets" in str(exc.value)

    # The bytes on the wire: a sender that silently edits the prompt between the
    # build and the POST is caught by the digest of what each pass actually sent.
    def tampering(prompt, port):
        return send(prompt + "\nAnd add one more fact please.", port)
    tampered = X.run_pass(fe, prompts, 8096, "B", tampering)
    with pytest.raises(X.PairNotComparable) as exc:
        X.pair_guard(prompts, a, tampered)
    assert "bytes on the wire" in str(exc.value)


def test_prompts_are_built_from_the_production_template_and_chunker():
    """The audit measures the production write path, so the prompt it sends is
    the extractor's own template rendered the way `extract_from_document` renders
    it, and its windows are the extractor's own `_chunk_content` output — same
    template, same `existing_facts` default of "None", same entity-type line, same
    chunk boundaries. A prompt edited here would be measuring a different engine.

    This is the one test that loads the real extractor module (it is 1200 lines
    and imports `app.kg_store`, hence the skip if the checkout cannot)."""
    try:
        fe = X._load_extractor()
    except SystemExit:  # pragma: no cover
        pytest.skip("fact_extractor.py not reachable from this checkout")
    content = "Lloyd runs a nightly extraction job over the vault. " * 300
    prompts = X.build_prompts(fe, content)
    assert prompts, "the real chunker produced nothing"
    assert prompts[0]["text"].startswith(fe.EXTRACTION_PROMPT.split("{content}")[0][:40])
    assert "\nNone\n" in prompts[0]["text"], "existing_facts is not the production default"
    assert " | ".join(fe.SCHEMA_TYPES) in prompts[0]["text"]
    assert [p["offset"] for p in prompts] == [o for o, _c in fe.FactExtractor._chunk_content(None, content)]
    assert prompts[-1]["offset"] + prompts[-1]["chunk_chars"] <= len(content)
    # `FACT_PROMPT` is the one-shot sibling constant, not the incremental path —
    # the audit must not be accused of sending it.
    for one in prompts:
        for placeholder in ("{content}", "{existing_facts}", "{known_entities}",
                            "{entity_types}"):
            # A surviving placeholder would mean the chunk, the entity prior or the
            # category vocabulary was never substituted into the production
            # template — the prompt would then be asking about a literal brace.
            assert placeholder not in one["text"], f"unfilled {placeholder}"


# ── clause 3: the diff key ──────────────────────────────────────────────────

@pytest.fixture
def alias_map(monkeypatch):
    """A fixture alias table, so the test proves the map is CONSULTED rather than
    that the machine's real `memory/aliases.yaml` happens to contain a line."""
    monkeypatch.setattr(X, "_ALIAS_MAP", {"lloyd (this instance)": "Lloyd", "the assistant": "Lloyd"})
    return X


def test_a_reworded_fact_is_a_rewording_and_not_a_difference(alias_map):
    """`fact_identity(entity, category, fact_text)` is the store's triple but its
    text component is an exact `text_hash`, so keying a diff on it would make
    every paraphrase a difference and report the model's synonym habit as a
    defect. Here the canonical text matches, so the passes agree."""
    a = [{"entity": "Lloyd", "category": "state", "fact": "The queue runs nightly at 02:00."}]
    b = [{"entity": "lloyd (this instance)", "category": "State", "fact": "the queue runs nightly at 02:00"}]
    diff = X.diff_passes(a, b)
    assert diff["differences"] == 0 and diff["identical"] == 0
    assert len(diff["rewordings"]) == 1
    assert diff["rewordings"][0]["entity"] == "Lloyd", "the alias map was not consulted"
    assert diff["only_a"] == [] and diff["only_b"] == []


def test_a_genuinely_different_fact_stays_a_difference_even_when_it_is_a_rewording_of_nothing(alias_map):
    """The other direction of the same key: a fact about the same entity and
    category with different canonical text is a difference. If canonicalisation
    were loose enough to swallow this, the audit would report a clean corpus by
    bucketing real differences as duplicates."""
    a = [{"entity": "Lloyd", "category": "state", "fact": "The queue runs nightly."}]
    b = [{"entity": "Lloyd", "category": "state", "fact": "The queue was disabled in September."}]
    diff = X.diff_passes(a, b)
    assert diff["differences"] == 2 and diff["rewordings"] == []
    assert [f["fact"] for f in diff["only_a"]] == [a[0]["fact"]]
    assert [f["fact"] for f in diff["only_b"]] == [b[0]["fact"]]


def test_identical_wording_is_neither_a_rewording_nor_a_difference(alias_map):
    diff = X.diff_passes(FACT_A, FACT_A)
    assert diff == {"identical": 1, "rewordings": [], "only_a": [], "only_b": [],
                    "differences": 0}


def test_the_key_canonicalises_text_but_the_raw_fact_is_kept_for_the_report(alias_map):
    """The key decides matching; the report must still quote what each pass
    wrote, or settlement would be judging a string neither pass produced."""
    k = X.fact_key("  Lloyd (this instance) ", " STATE ", "It is, running: nightly!")
    assert k[0] == "Lloyd" and k[1] == "state"
    # Asserted against the form itself, not against `canonical_text(...)` again: the
    # line used to read `k[2] == X.canonical_text("it is running nightly")`, which
    # compares a function's output with its own output and so is true for any
    # function at all — it could not fail short of an exception. The literal is the
    # claim: which words the canonical form throws away is what decides whether a
    # rewording collapses, and a change there moves the disagreement rate.
    assert k[2] == "running nightly", f"canonical form moved: {k[2]!r}"
    # And the property the bucket rests on, stated as a pair of inputs rather than
    # as one: two wordings of one claim share a key, one word of difference does not.
    assert X.fact_key("Lloyd", "state", "Lloyd runs the audit nightly")[2] == \
        X.fact_key("Lloyd", "state", "  Lloyd, runs the audit  nightly! ")[2]
    assert X.fact_key("Lloyd", "state", "Lloyd runs the audit nightly")[2] != \
        X.fact_key("Lloyd", "state", "Lloyd runs the audit weekly")[2]
    diff = X.diff_passes(FACT_A, FACT_B)
    assert diff["differences"] == 2 and diff["rewordings"] == []


# ── clause 4: settling against the source ───────────────────────────────────

SOURCE = ("The extraction job runs nightly over the vault. It reads only the "
          "allow-listed directories, and the run reports its own coverage.")
SPANS = {"pass_a_supported": "runs nightly over the vault",
         "pass_b_supported": "reports its own coverage",
         "both_supported": "reads only the allow-listed directories",
         "neither_supported": None}


_ANY = object()


def _judge(label, span=_ANY):
    """A settlement answer shaped like the judge's, over a span the caller names —
    defaulting to the good stretch registered for that label."""
    def judge(document, fact_a, fact_b, port):
        return json.dumps({"label": label,
                           "span": SPANS.get(label) if span is _ANY else span})
    return judge


def _diff_pair():
    return {"a": FACT_A[0], "b": None, "side": "a"}


@pytest.mark.parametrize("label", X.LABELS)
def test_a_difference_settles_into_exactly_one_of_the_four_labels_with_a_span(label):
    """One input difference in, exactly one label out. The span is located in the
    document, and the only label that may answer without one is
    `neither_supported` — the document settling it by NOT containing either
    reading is a real result, not a missing answer."""
    verdict = X.settle(_diff_pair(), SOURCE, _judge(label), 8096)
    assert verdict["label"] == label
    if label == "neither_supported":
        assert verdict["span"] is None
    else:
        assert SOURCE[verdict["span"]["start"]:verdict["span"]["end"]] == SPANS[label]


@pytest.mark.parametrize("label", ["pass_a_supported", "pass_b_supported", "both_supported"])
def test_a_label_naming_a_supporting_stretch_may_not_answer_without_one(label):
    """The clause's `any difference the settlement cannot place is emitted as
    unsettled`: a label whose evidence is absent is placed nowhere, and the finding
    this pins is `both_supported` in particular — it is agreement-shaped, so folding
    it in silently would raise the agreement count using an answer that checked
    nothing. `neither_supported` is the exception, and the test above covers it: the
    document containing NEITHER reading is itself the evidence."""
    verdict = X.settle(_diff_pair(), SOURCE, _judge(label, None), 8096)
    assert verdict["label"] == X.UNSETTLED, (label, verdict)
    assert "named no stretch" in verdict["reason"], verdict


def test_a_label_outside_the_four_and_a_single_sided_answer_with_no_span_are_unsettled():
    """The failure this clause exists for: a settlement that cannot be placed must
    become `unsettled`, never agreement. A judge that answers `both_supported`
    with nothing to point at has not found two facts, it has given up."""
    for label, span, why in (("maybe_supported", SPANS["pass_a_supported"], "outside the four"),
                             ("pass_a_supported", None, "named no stretch"),
                             ("pass_a_supported", "the document plainly implies this overall", "named no stretch")):
        verdict = X.settle(_diff_pair(), SOURCE, _judge(label, span), 8096)
        assert verdict["label"] == X.UNSETTLED, (label, verdict)
        assert why in verdict["reason"], (label, verdict["reason"])


def test_an_unparseable_or_unreachable_judge_leaves_the_difference_unsettled():
    def broken(document, fa, fb, port):
        return "the document seems to support both readings, probably"
    verdict = X.settle(_diff_pair(), SOURCE, broken, 8096)
    assert verdict["label"] == X.UNSETTLED and "unparseable" in verdict["reason"]

    def down(document, fa, fb, port):
        raise RuntimeError("connection refused")
    verdict = X.settle(_diff_pair(), SOURCE, down, 8096)
    assert verdict["label"] == X.UNSETTLED and "unreachable" in verdict["reason"]


def test_a_settled_span_survives_whitespace_differences_but_not_invention():
    """The judge quotes across a line break: whitespace-insensitive matching is
    legitimate and reports the offsets into the collapsed text it actually used.
    A quote that is in neither form is invention, and clause 4 turns it into an
    unsettled row."""
    doc = "The job runs nightly.\n\tIt reports coverage."
    hit = X._locate(doc, "runs nightly.\n\tIt reports")
    assert hit and hit["form"] == "exact"
    hit = X._locate(doc, "runs nightly. It reports")
    assert hit and hit["form"] == "collapsed"
    assert X._locate(doc, "runs weekly") is None


# ── clause 5: the artifact and its blast radius ─────────────────────────────

def _facts(n, tag):
    return [{"entity": "Lloyd", "category": "state", "fact": f"{tag} fact {i}."} for i in range(n)]


def _run(tmp_path, vault, sender_factory, settle=None, argv_extra=()):
    out = tmp_path / "out"
    mod = _load_module()          # a fresh import per run: the alias cache is module state
    mod._ALIAS_MAP = {}

    def fake_load():
        return FakeFE()
    mod._load_extractor = fake_load
    send, calls = sender_factory
    mod.send = lambda port, prompt, **kw: send(prompt, port)
    if settle:
        # The settlement arm is the `settle` function itself, not the transport:
        # replacing `send` would also replace the EXTRACTION transport, both passes
        # would then be handed a settlement-shaped answer, and the run would report
        # zero facts compared — a test of the fixture, not of the bookkeeping.
        #
        # The signature is `settle`'s CURRENT one, `meter` included, and that is a
        # test-honesty matter rather than tidiness. This fixture used to be
        # `(diff, source, judge=None, port=None)` while the function it replaces had
        # grown a fifth parameter, so every call raised TypeError, `audit_document`
        # caught it as a per-document error, and the artifact came back with zero
        # documents measured. An assertion that runs after the document rows are
        # read then blew up on an IndexError and the suite noticed; one that runs
        # before them would have reported a green pass over a run that had failed
        # every document. Both halves are fixed: this signature, and the assert order
        # in the tests that read the rows.
        def forced(diff, source, judge=None, port=None, meter=None):
            located = X._locate(source, SPANS.get(settle)) if SPANS.get(settle) else None
            return {"label": settle, "span": located}
        mod.settle = forced
    # `--config` is not optional here: left out, `main` reads the LIVE
    # pipeline_config.yaml and resolves its `~/obsidian` paths against this tmp
    # vault, every entry falls outside it, and the draw refuses. The dirs are read
    # off the fixture's own vault so the config always matches what was built.
    # SIX, not "every directory in the fixture vault": the fixture also plants a
    # trap file in each excluded directory, and naming those as the corpus would
    # make the draw refuse on their exclusion filters instead of sampling.
    cfg = config_with(SIX, tmp_path)
    argv = ["--out", str(out), "--vault-root", str(vault), "--config", str(cfg),
            "--sample-size", "6", "--seed", "run-seed"] + list(argv_extra)
    rc = mod.main(argv)
    return rc, out, calls, mod


@pytest.fixture
def both_passes():
    """Pass A answers with one fact, pass B with a different one: the smallest
    corpus that produces a real disagreement to settle."""
    def factory():
        return _sender([_facts(2, "a"), _facts(1, "b")])
    return factory


def test_the_run_writes_one_dated_json_and_one_table_under_out(tmp_path, vault, both_passes):
    """The artifact's shape, over a faked transport with no socket: a run dir
    dated to the second, holding `extraction-disagreement.json` and
    `.md`."""
    rc, out, calls, mod = _run(tmp_path, vault, both_passes())
    assert rc == 0
    runs = [p for p in out.iterdir()]
    assert len(runs) == 1 and runs[0].is_dir()
    names = sorted(p.name for p in runs[0].iterdir())
    assert names == ["extraction-disagreement.json", "extraction-disagreement.md"]
    summary = json.loads((runs[0] / "extraction-disagreement.json").read_text())
    assert summary["schema"] == "extraction-disagreement/1"
    assert summary["documents_measured"] == 6 and summary["sample"]["seed"] == "run-seed"
    assert summary["sample"]["paths"], "the drawn paths must travel with the seed"
    assert set(summary["coverage_by_dir"].values()) == {1}


def test_the_artifact_carries_per_document_rates_settled_counts_yield_and_its_own_cost(
        tmp_path, vault, both_passes):
    """Per-document disagreement rate, the four-way settled counts, defect yield,
    and a cost line measured by this module: the extractor's `_call_llm` drops the
    response's `usage` field, and the usage store is session-keyed harness
    telemetry with no extraction rows, so tokens off each response plus wall clock
    is the only cost available — and it is what a widening decision needs."""
    rc, out, calls, mod = _run(tmp_path, vault, both_passes(), settle="neither_supported")
    summary = json.loads(next((out / p.name / "extraction-disagreement.json")
                              for p in out.iterdir())
                         .read_text(encoding="utf-8"))
    doc = summary["documents"][0]
    n = doc["differences"]
    assert n > 0 and doc["compared"] == n
    assert doc["disagreement_rate"] == 1.0
    # Every difference of the document lands in exactly the one label the
    # settlement was forced to, so the counts are the difference count itself.
    assert doc["settled"] == {"pass_a_supported": 0, "pass_b_supported": 0,
                              "both_supported": 0, "neither_supported": n}, doc["settled"]
    assert doc["defects"] == n and doc["defect_yield"] == 1.0
    assert doc["cost"]["total_tokens"] > 0 and doc["cost"]["seconds"] > 0
    assert summary["totals"]["settled"]["neither_supported"] == summary["totals"]["differences"]
    assert summary["defect_yield_overall"] == 1.0
    assert "7.5M" in summary["cost"]["gallon_reference"]
    assert "usage" in summary["cost"]["source"]
    assert summary["cost"]["per_document_mean_tokens"] > 0
    assert summary["disagreement_rate_is_a_lower_bound"]


def test_an_unsettled_difference_is_counted_and_never_folded_into_agreement(
        tmp_path, vault, both_passes):
    """The rate and the yield keep their own denominators: differences the judge
    could not place are reported as `unsettled`, and defect yield divides by what
    was actually settled (3 of 3 here), not by the difference count."""
    rc, out, calls, mod = _run(tmp_path, vault, both_passes())   # judge quotes nothing
    summary = json.loads(next((out / p.name / "extraction-disagreement.json")
                              for p in out.iterdir())
                         .read_text(encoding="utf-8"))
    assert summary["totals"]["unsettled"] == summary["totals"]["differences"]
    assert sum(summary["totals"]["settled"].values()) == 0
    assert summary["defect_yield_overall"] is None, "nothing settled: yield must not be 0.0"
    # Settling failed for every difference, and the disagreement rate is a property
    # of the two passes — it does not move because the judge was unreachable.
    recomputed = summary["totals"]["differences"] / sum(d["compared"] for d in summary["documents"])
    assert summary["disagreement_rate_overall"] == round(recomputed, 4)
    doc = summary["documents"][0]
    assert len(doc["unsettled"]) == doc["differences"] and doc["unsettled"][0]["reason"]


def test_three_calls_of_4_2_seconds_meter_12_6_seconds_and_not_20_6():
    """The arithmetic defect, pinned as numbers.

    The accumulator used to be walked with `for k in usage`, and by the third call
    `usage` had gained a `seconds` key from the line below it — so each call added
    `int(4.2) = 4` to `seconds` AND the real `4.2`, metering **20.6 s for a true
    12.6 s** and billing the artifact against Gallon's figure using a number its own
    double-add had inflated. `TOKEN_KEYS` is the fix: the keys summed are named, so
    a key that arrives later cannot silently join the sum.
    """
    cost = X.new_cost()
    for _ in range(3):
        X.book_call(cost, {"prompt_tokens": 50, "completion_tokens": 20,
                           "total_tokens": 70, "seconds": 4.2}, 4.2)
    assert cost["seconds"] == 12.6, "the wall clock is double-added again"
    assert cost["calls"] == 3
    assert cost["total_tokens"] == 210
    # A meter is only meaningful if the run's total is the sum of its arms; the sum
    # helper reads the same named keys, so both sides of the ledger move together.
    assert X.merge_costs(cost, X.new_cost()) == cost


def test_the_per_document_cost_meters_the_reconciler_and_counts_each_call_once(tmp_path):
    """Clause 5 end to end on a transport-shaped judge: one chunk, so two
    extraction calls, and one difference, so one reconciler read — three calls at
    4.2 s each, and the document's cost line must read 12.6 s and 210 tokens.

    This is the number the clause names, and the defect it is the witness for:
    `settle` returned its `(answer, usage, seconds)` triple and `audit_document`
    discarded it, so the reconciler's calls were in the artifact's denominator
    nowhere — a cost line that billed two arms of a three-arm method while comparing
    itself to a figure for the whole reconciliation. `pair_guard`'s own lesson is the
    reason the arms are separate meters rather than one accumulator threaded through:
    an assertion written below a `raise` checks nothing, and a total derived from
    disjoint arms cannot double-count without putting a call in two arms.
    """
    vault = tmp_path / "obsidian"
    doc = vault / "knowledge" / "three-calls.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("# Three calls\n\nLloyd runs the audit over six directories.\n",
                   encoding="utf-8")
    CALL = {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70, "seconds": 4.2}

    seen: list[str] = []

    def sender(prompt, port):
        # First sight of a prompt is pass A (one fact); the repeat is pass B
        # (nothing) — so exactly one difference goes to the reconciler.
        first = prompt not in seen
        seen.append(prompt)
        facts = [{"entity": "Lloyd", "category": "state",
                  "fact": "Lloyd runs the audit over six directories."}] if first else []
        return (json.dumps({"facts": facts}), dict(CALL, sent_sha256=hashlib.sha256(prompt.encode()).hexdigest()),
                4.2)

    def judge(document, fact_a, fact_b, port):
        # A transport, like the real one: the whole `send` triple, tokens included.
        return json.dumps({"label": "neither_supported", "span": None}), dict(CALL), 4.2

    row = X.audit_document(_fake_fe(), doc, vault, port=8096, judge=judge, sender=sender)
    assert row["differences"] == 1 and row["settled"]["neither_supported"] == 1
    cost = row["cost"]
    assert cost["a"]["calls"] == 1 and cost["b"]["calls"] == 1
    assert cost["settlement"]["calls"] == 1, "the reconciler's read is unmetered again"
    assert cost["calls"] == 3
    assert cost["seconds"] == 12.6, f"three 4.2 s calls must meter 12.6 s, got {cost['seconds']}"
    assert cost["total_tokens"] == 210
    assert cost["settlement"]["total_tokens"] == 70
    assert X.merge_costs(cost["a"], cost["b"], cost["settlement"]) == {
        k: cost[k] for k in ("calls", "prompt_tokens", "completion_tokens",
                             "total_tokens", "seconds")}


def test_the_run_total_includes_every_settlement_call(tmp_path, vault, both_passes):
    """The same bookkeeping one level up, over a whole six-document run, asserted
    against the fake transport's OWN call counter rather than a literal — the
    documents are 900 words long, so their chunk counts vary with the fixture and a
    hard-coded total would be a second thing to keep in sync.

    Every stubbed call answers 70 tokens in 0.5 s, so the meter's totals are exact
    multiples of the call count and any call the artifact failed to bill shows up as
    a shortfall. The settlement here deliberately FAILS to place its spans (the
    transport answers with extraction JSON), which is the better witness: a call is
    metered when it is made, before its answer is judged, so a run whose judge could
    not be reached still reports what it cost to find that out — under the
    discarded-return defect the artifact reported a third of the calls it made.
    """
    rc, out, calls, mod = _run(tmp_path, vault, both_passes())
    assert rc == 0
    summary = json.loads(next((out / p.name / "extraction-disagreement.json")
                              for p in out.iterdir()).read_text(encoding="utf-8"))
    doc = summary["documents"][0]
    assert doc["cost"]["a"]["calls"] == doc["cost"]["b"]["calls"], "the passes differ in calls"
    assert doc["cost"]["settlement"]["calls"] == doc["differences"], (
        "one read per difference is what the reconciler costs")
    assert doc["cost"]["calls"] == (doc["cost"]["a"]["calls"] + doc["cost"]["b"]["calls"]
                                   + doc["cost"]["settlement"]["calls"])
    # The strongest form: the artifact's own count of model calls equals the count
    # the transport received. Nothing else in the file can say that.
    assert summary["cost"]["total_calls"] == calls["n"], (
        f"billed {summary['cost']['total_calls']} of the {calls['n']} calls sent")
    assert summary["cost"]["total_tokens"] == calls["n"] * 70
    assert summary["cost"]["total_seconds"] == round(calls["n"] * 0.5, 3)
    assert summary["cost"]["settlement_tokens"] == summary["totals"]["differences"] * 70
    assert summary["cost"]["settlement_seconds"] == round(
        summary["totals"]["differences"] * 0.5, 3)
    # And the direction the discarded return moved the number: the run's cost is
    # strictly larger than its two extraction passes, which is the whole content of
    # "the per-document cost counts every model call, settlement included".
    two_passes_only = sum(d["cost"]["a"]["calls"] + d["cost"]["b"]["calls"]
                          for d in summary["documents"])
    assert summary["cost"]["total_calls"] > two_passes_only
    assert "exactly once" in summary["cost"]["source"]


def test_the_run_bills_every_call_it_made_once_and_the_three_arms_are_disjoint(tmp_path):
    """The blast radius of the fix: a whole document, cost read three ways.

    Pass A's calls, pass B's calls and the reconciler's calls are disjoint sets, so
    their sum is the document's cost and the sum can be checked against the number of
    calls the fake transport actually received. The fixture's chunker yields four
    windows, so a measured document is 4 + 4 + N calls for N differences, and `N` is
    read back from the row rather than hard-coded: what is asserted is the RELATION
    between the arms and the transport, which is the thing that broke.
    """
    # 3,605 characters, so the fixture's 1,000-character chunker yields exactly four
    # windows; both claim sentences sit in the first window so settlement can place a
    # span when it reads the document back.
    src = ("---\ntype: note\n---\n\n# Doc\n\n"
           "Lloyd wrote alpha. Lloyd wrote beta. Lloyd wrote gamma. shared reading here. "
           + "word " * 700 + "\n")
    mod = _load_module()
    mod._ALIAS_MAP = {}
    prompts = mod.build_prompts(FakeFE(), src)
    assert len(prompts) == 4, f"the fixture's chunking moved: {len(prompts)} windows"

    def _json(texts):
        return json.dumps({"facts": [{"entity": "Lloyd", "category": "state", "fact": t}
                                     for t in texts]})

    # Pass A answers every chunk with one reading B also gives and one it does not;
    # pass B with the shared reading alone. `diff_passes` compares fact OCCURRENCES,
    # so four chunks each holding one unmatched pass-A reading is four differences, and
    # the reconciler is owed exactly four reads. Every count below is therefore fixed:
    # 4 + 4 + 4 = 12 calls.
    a_answers = [_json(["shared reading here", "Lloyd wrote alpha."]) for _ in prompts]
    b_answers = [_json(["shared reading here"]) for _ in prompts]
    queue = a_answers + b_answers
    seen = {"n": 0}

    def meter(prompt_tokens=1000, completion_tokens=500, seconds=1.0):
        return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens, "seconds": seconds}

    def transport(prompt, port):
        text = queue[seen["n"]]
        seen["n"] += 1
        u = meter()
        u["sent_sha256"] = "x"
        return text, u, 1.0

    def judge(doc, fa, fb, port):
        # The transport's own shape — a triple, not a bare dict — because that is
        # what the default settlement judge now returns and what `_as_answer`
        # unpacks. A bare dict would be metered at zero tokens by design (a stub
        # has no count to report), and this test's whole claim is the token total.
        seen["n"] += 1
        return ({"label": "pass_a_supported", "span": "Lloyd wrote alpha."}, meter(), 1.0)

    d = tmp_path / "vault" / "knowledge"
    d.mkdir(parents=True)
    (d / "doc.md").write_text(src, encoding="utf-8")
    row = mod.audit_document(FakeFE(), d / "doc.md", tmp_path / "vault", port=9,
                             judge=judge, sender=transport)
    assert row["pass_errors"] == [] and "error" not in row, row
    n_a, n_b, n_s = (row["cost"]["a"]["calls"], row["cost"]["b"]["calls"],
                     row["cost"]["settlement"]["calls"])
    assert (n_a, n_b) == (4, 4), "both passes send one call per chunk"
    assert row["differences"] == 4, (
        f"four chunks × one unmatched pass-A reading is four differences, got "
        f"{row['differences']}")
    assert n_s == row["differences"], "one reconciler read per difference"
    assert row["settled"]["pass_a_supported"] == 4, "every read landed a label"
    assert row["cost"]["calls"] == n_a + n_b + n_s == seen["n"] == 12, (
        f"transport carried {seen['n']} calls; artifact billed {row['cost']['calls']}")
    # 1500 tokens and 1.0 s a call, every arm included: 18,000 tokens and 12.0 s for
    # the document. Billing only the two passes — the defect this fixes — reported
    # 12,000 tokens and 8.0 s, two thirds of the truth, against a reference figure
    # (Gallon's 7.5M) that itself includes the reconciler.
    assert row["cost"]["total_tokens"] == 12 * 1500 == 18000
    assert row["cost"]["seconds"] == 12.0
    assert row["cost"]["settlement"]["total_tokens"] == 4 * 1500 == 6000
    assert row["cost"]["settlement"]["seconds"] == 4.0


def test_the_run_writes_only_inside_its_out_dir(tmp_path, vault, both_passes):
    """Wires nothing, the item's own last step. The tree is snapshotted by path,
    size and BYTES over the three roots this run could reach — the fixture vault,
    the checkout's `eval/`, and `tmp_path/state`, the state root a content-hash or
    fact-tree write would be handed — and a run that touched any of it shows up
    here. The roots are what `_fingerprint` walks, named so that a reader can tell
    what this test does NOT cover: the live vault under `~/obsidian` and the live
    content-hash store are not in the snapshot, because this run is pointed at a
    tmp vault and a tmp state root and any write that escaped those would be a bug
    in a path the audit never takes."""
    before = _fingerprint(vault, tmp_path)
    rc, out, calls, mod = _run(tmp_path, vault, both_passes())
    # One assert per fact, in the order that fails loudest: `rc` first, because a run
    # that returned 2 or 3 has produced no artifact at all and every read below it
    # would then be a read of nothing. This line used to read
    # `assert rc == 0 and calls["n"] == 12 * 6 // 6 * 2 * 6 or calls["n"] > 0`, whose
    # `or` bound looser than the `and`, so ANY non-zero call count satisfied the whole
    # statement and the call count itself was never asserted.
    assert rc == 0
    # The count is asserted against the run's own meter rather than a literal: the
    # fixture's documents are 900 words long, so how many chunk-calls a document is
    # worth is a property of the fixture and belongs to the cost tests, which read it
    # back per arm. What belongs here is that the transport was reached at all, and
    # that the number of calls it received is the number the artifact billed — the
    # equality a discarded settlement return silently broke.
    assert calls["n"] > 0, "the fake transport was never called: nothing was measured"
    summary = json.loads(next((out / p.name / "extraction-disagreement.json")
                              for p in out.iterdir()).read_text(encoding="utf-8"))
    assert summary["cost"]["total_calls"] == calls["n"], (
        f"the artifact billed {summary['cost']['total_calls']} calls; the transport "
        f"carried {calls['n']}")
    allowed = _known(out, before)
    changed = {k: v for k, v in _fingerprint(vault, tmp_path).items() if allowed.get(k) != v}
    assert changed == {}, f"the run wrote outside --out: {sorted(changed)[:5]}"
    assert all(str(out) in str(p) for p in changed), changed


def _fingerprint(vault: Path, tmp_path: Path) -> dict:
    """Every file the run could have touched, as `(size, sha256 of its bytes)`.

    Bytes, not mtimes: rewriting a file to the same length inside the same clock tick
    is precisely what an mtime snapshot misses, and the claim under test is that the
    run touched nothing outside `--out`. Three roots — the fixture vault the audit
    reads, the checkout's `eval/` where the module itself lives, and
    `tmp_path/state`, the state root a content-hash or fact-tree write would be
    handed. `tmp_path` as a whole is deliberately NOT walked: `_run` writes the
    fixture's `pipeline_config.yaml` into it AFTER the before-snapshot, and a witness
    that indicts the fixture for existing is a witness nobody trusts.
    """
    fp = {}
    for base in (vault, ROOT / "eval", tmp_path / "state"):
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_file():
                try:
                    fp[str(p)] = (p.stat().st_size,
                                  hashlib.sha256(p.read_bytes()).hexdigest())
                except OSError:
                    pass
    return fp


def _known(out: Path, before: dict) -> dict:
    """The paths a run is allowed to have touched, WITH their before-values.

    Returning a set of paths was the defect: a file the run MODIFIED IN PLACE was
    treated as known, and the comparison `before.get(k) != v` could then never fire —
    so a write into the vault or into `eval/` that did not add a path was invisible to
    the one test whose job is to catch it. Carrying the value through makes a modified
    file compare as unchanged only when the bytes and stamps really are the same, which
    is the whole claim. Paths under `--out` still carry their (empty) before-value, so
    a run dir being created is allowed and a second run dir appearing inside another
    run's directory is not: `before` holds no value for it, and neither does this.
    """
    known = dict(before)
    for p in out.rglob("*"):
        known.setdefault(str(p), None)
    return known


def test_the_run_refuses_to_write_anything_when_the_engine_is_down(tmp_path, vault, both_passes):
    """Exit 2 BEFORE creating a run dir: `extraction-disagreement.json` from a run
    whose second pass never ran would read as a clean audit to anything that ever
    globbed the directory."""
    out = tmp_path / "out"
    mod = _load_module()
    mod._load_extractor = lambda: FakeFE()
    mod._endpoint_up = lambda port: False
    cfg = config_with(SIX, tmp_path)
    rc = mod.main(["--out", str(out), "--vault-root", str(vault), "--config", str(cfg),
                   "--sample-size", "6"])
    assert rc == 2
    assert list(out.rglob("*")) == []


def test_dry_run_writes_the_sample_but_sends_no_request(tmp_path, vault, both_passes):
    """The offline shape the artifact can be inspected in: the sample is drawn
    and recorded, no socket is opened, and the document rows are marked so a
    reader cannot mistake an unmeasured run for a clean one."""
    rc, out, calls, mod = _run(tmp_path, vault, both_passes(), argv_extra=("--dry-run",))
    assert rc == 0
    summary = json.loads(next((out / p.name / "extraction-disagreement.json")
                              for p in out.iterdir())
                         .read_text(encoding="utf-8"))
    assert summary["dry_run"] is True
    assert summary["documents_measured"] == 6
    assert all(d["disagreement_rate"] is None for d in summary["documents"])


def test_the_change_that_added_this_measurement_touched_only_eval_and_tests():
    """Wires nothing. Anything else in the diff is a second change smuggled into
    a measurement round — and the specific smuggling this one is watching for is a
    call site in the nightly path, which is the item's own last step."""
    # The range is this measurement's OWN commits: everything from the commit that
    # first added the module onward. `merge-base HEAD origin/main` would also work in
    # a checkout sitting on main, but in a self-mod round the branch base is already
    # ahead of origin/main by other rounds' commits, and their files would be
    # attributed to this one — a scope test that fails on an untouched repo proves
    # nothing about scope.
    try:
        added = subprocess.run(
            ["git", "-C", str(ROOT), "log", "--diff-filter=A", "-1", "--format=%H", "--",
             "eval/extraction_disagreement.py"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        if not added:
            pytest.skip("the module is not committed in this checkout")
        base = subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{added}^"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        pytest.skip("no git history for this module in this checkout")
    changed = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", f"{base}..HEAD"],
                             capture_output=True, text=True, timeout=60).stdout.splitlines()
    # The control must be empty for the RIGHT reason. `git diff HEAD` compares the
    # working tree against HEAD and is therefore blind to a staged-but-uncommitted
    # change, which is the exact state this suite runs in: `automod_gate` stages and
    # commits with `git commit -- F --` against a shared index, and any other
    # unattended round's work sitting in that index shows up in HEAD's diff as a file
    # THIS commit touched — a scope test that indicts an untouched module on another
    # round's files, which is the false block #2658's review recorded. What must be
    # empty is a path that is STAGED but not in this commit's range: a file staged and
    # never committed is neither in the range nor in the tree, so reading the range
    # alone would not see it.
    staged = subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--name-only", "HEAD"],
                            capture_output=True, text=True, timeout=60).stdout.splitlines()
    assert staged == [], f"staged-but-uncommitted tracked change: {staged}"
    assert changed, "this commit is not in the range the test can see"
    for path in changed:
        assert path.startswith(("eval/", "tests/")), path
    assert "scripts/memory/next-gen-memory/nightly_extraction.py" not in changed
