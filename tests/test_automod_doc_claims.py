"""The architecture doc's load-bearing numbers must match the code.

`architecture/automod.md` states specific thresholds, path rules and
config placements as fact. A doc that quietly drifts from the implementation is
worse than no doc: it is the thing someone reads at 3am while deciding whether
the watchdog can be trusted.

Only claims where being wrong would mislead an operator are pinned here — not
prose. The one class of prose that IS load-bearing: a sentence that disarms a
metric the code has armed. #505 found `architecture/automod.md` stating the
document metrics were "reported and never fire" for a week after
`08e998a` armed all seven, 31 lines below the sentence that said they were
armed, and an arch-review pass graded the file `current` on top of it. A
reader who believes that sentence skips the gate on a document-ranking change,
or discounts a real 3σ rollback signal as noise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent-services" / "guardian"))

import policy  # noqa: E402

DOC = ROOT / "architecture" / "automod.md"
REGRESSION_SRC = ROOT / "workers" / "sources" / "automod_regression.py"

# The doc's canonical statement of the armed set, written so a test can read
# the number and the names out of it: "**All seven are armed:** `a`, `b`, ...".
ARMED_STATEMENT_RE = re.compile(
    r"\*\*All ([a-z]+) are armed:\*\*\s+((?:`[a-z0-9_]+`,?\s*)+)")

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

# Any number word standing next to "armed", in either English order:
# "the armed three", "three metrics are armed", "all seven are armed".
_NUM = "|".join(NUMBER_WORDS)
ARMED_COUNT_RES = (
    re.compile(rf"\barmed\s+(?:metrics?\s+)?({_NUM})\b"),
    re.compile(rf"\b({_NUM})\s+(?:metrics?\s+)?(?:are|were|is|was)\s+armed\b"),
)

# The comment header above the armed tuples. It shipped duplicated.
ARMED_SET_HEADER = ("# What the armed set can and cannot see, MEASURED "
                    "rather than assumed.")

# Sentences that tell the reader the document metrics cannot fire. Each one
# was true while the corpus moved between arms and is false under the pin.
DISARMING_PHRASES = ("reported and never fire",
                     "doc-side four are now reported",
                     "Re-arming a doc-side metric")

# The same claim in words the three above do not cover, checked per sentence
# against the names in ARMED_METRICS rather than against a fixed string: a
# reworded disarm ("`mrr_doc` is not armed and can never fire") reintroduces
# the #505 defect while every literal blocklist stays green.
DISARM_CLAIM_WORDS = ("never fire", "never fires", "cannot fire", "not armed",
                      "no longer armed", "is unarmed", "are unarmed", "disarm",
                      "reported only", "only reported")


def _disarm_claims(text: str, metrics) -> list[str]:
    """Sentences naming one armed metric AND that it cannot fire."""
    hits = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        low = sentence.lower()
        named = [m for m in metrics
                 if re.search(rf"\b{re.escape(m)}\b", sentence)]
        if not named:
            continue
        for word in DISARM_CLAIM_WORDS:
            if word in low:
                hits.append(f"{named[0]}: {sentence}")
                break
    return hits


def _flat(path: Path) -> str:
    """File text with every run of whitespace collapsed to one space.

    The doc wraps at ~80 columns, so "…reported and never\nfire." is one
    sentence split across two lines. Matching raw text would let exactly the
    violation this guards reproduce itself with a newline in the middle.
    """
    return " ".join(path.read_text(encoding="utf-8").split())


def test_the_doc_exists():
    assert DOC.exists()


# ── §7.2 probe budgets ──────────────────────────────────────────────────────

def test_probe_budgets_match_the_doc():
    """"refused: 3 ticks ... timeout: 24 ticks (2 minutes)" """
    assert policy.PROBE_FAIL_STREAK == 3
    assert policy.PROBE_TIMEOUT_STREAK == 24
    assert policy.PROBE_TIMEOUT_STREAK * policy.TICK_SECONDS == 120


def test_probe_timeout_matches_the_doc():
    """"the probe timeout went 2s -> 10s" """
    assert policy.PROBE_TIMEOUT_SECONDS == 10.0


def test_the_rpc_timeout_exceeds_stopwaitsecs():
    """§7.4: a blocking stop legitimately takes stopwaitsecs (15s)."""
    assert policy.SUPERVISOR_RPC_TIMEOUT > 15.0


# ── §7 the unit ─────────────────────────────────────────────────────────────

def test_start_limit_interval_is_in_the_unit_section():
    """§7: "StartLimitIntervalSec belongs in [Unit]".

    In [Service] systemd ignores it and applies the default 5-starts-in-10s
    limit, letting the watchdog rate-limit itself into a failed state.
    """
    unit = (ROOT / "agent-services" / "systemd" / "lloyd-guardian.service").read_text()
    section, placed = None, {}
    for line in unit.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
        elif stripped.startswith("StartLimitIntervalSec"):
            placed[section] = stripped
    assert list(placed) == ["[Unit]"], placed


@pytest.mark.parametrize("conf", ["lloyd-backend", "lloyd-mcp"])
def test_supervisor_confs_stop_process_groups(conf):
    """§7.4: without these a Bash tool's child outlives the stop and can write
    into the tree mid-reset."""
    text = (ROOT / "agent-services" / "supervisor" / "conf.d" / f"{conf}.conf").read_text()
    assert "stopasgroup=true" in text
    assert "killasgroup=true" in text


# ── §10 what Lloyd may change ───────────────────────────────────────────────

@pytest.mark.parametrize("path,expected", [
    ("config.yaml", "denied"),
    ("data/tool_overrides.yaml", "denied"),
    (".env", "denied"),
    ("pytest.ini", "denied"),
    (".gitignore", "denied"),
    ("web/src/App.tsx", "allowed"),
    ("web/package.json", "denied"),
    ("web/vite.config.ts", "denied"),
    ("agent-services/guardian/guardian.py", "protected"),
    ("scripts/automod/gate.py", "protected"),
    ("app/routers/health.py", "protected"),
    ("requirements.lock", "allowed"),
    ("app/harness/loop.py", "allowed"),
    # #1073: the two paths the settled solver route needs, both `unlisted`
    # before this round, which is why no round could have landed the route.
    ("requirements-dev.txt", "allowed"),
    ("SETUP.md", "allowed"),
])
def test_path_policy_matches_the_doc(path, expected):
    from scripts.automod import spec
    assert spec.classify(path) == expected


def test_the_dev_requirements_file_is_documented_as_never_an_install_target():
    """§10 has to say WHY a third requirements file is allowed, because the
    sentence it replaces ("`requirements*` is allowed only because the `venv`
    rung exists") is false for it — and that sentence is the whole hazard
    #1073 is about: a reader who believes it would put a solver in
    `requirements.txt`, where the candidate venv installs it but the live venv
    may already have it, or not put it anywhere the candidate can see.

    So the doc names the two files the rung installs from and states, beside
    them, that `requirements-dev.txt` is not one of them. Both halves are
    checked against the code that decides it, not just against the prose.

    What this test does NOT do, and must not be read as doing: it is §10 of
    *this* file, not `SETUP.md`. #1073's clauses 1 and 2 put the route and the
    refreeze exclusion in `SETUP.md`, and this round cannot write that file —
    admitting the path and writing it are two rounds, because rung 0 reads
    `ALLOWED_GLOBS` from the live tree (`round.py:445` spawns the gate with
    `cwd=LIVE_ROOT`). Those two clauses are #1378's, and when it lands it should
    assert the same two facts against `SETUP.md`'s dependency and refreeze
    sections here, not only against §10.
    """
    from scripts.automod import spec
    doc = (ROOT / "architecture" / "automod.md").read_text()
    section = doc.split("## 10. What Lloyd may change", 1)[1].split("\n## 11", 1)[0]
    assert "requirements-dev.txt" in section
    assert "requirements.lock" in section and "requirements.txt" in section
    # The claim is only true because of these two facts, so pin them here.
    assert spec.classify("requirements-dev.txt") == "allowed"
    assert not spec.touches_requirements(["requirements-dev.txt"])
    assert spec.touches_requirements(["requirements.txt"])
    assert spec.touches_requirements(["requirements.lock"])


def test_the_install_targets_stay_free_of_the_solver():
    """#1073 clause 1's checkable half, pinned on the files instead of on prose.

    `grep -n z3 requirements.txt requirements.lock` has to stay empty for as long
    as the settled route holds. Those two names are the only install lists the
    gate's candidate venv reads, so a solver reaching either one does two things
    at once: it makes an optional dev package a container-rebuild dependency, and
    it is the only way the live venv and a candidate can end up disagreeing about
    whether `import z3` succeeds. The route sends it to `requirements-dev.txt`,
    which neither mechanism can name — pinned just above.
    """
    for name in ("requirements.txt", "requirements.lock"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "z3" not in text.lower(), f"{name} gained the solver; the route changed"


def test_the_doc_names_every_site_carrying_the_refreeze_command():
    """#1073 clause 2's hazard, recorded where a refreeze will be read.

    `requirements.lock` is a `pip freeze` snapshot, so a package installed out of
    band — which is how the solver reaches the live venv — is captured by the next
    refreeze and becomes a container-rebuild dependency. The fix is a dev-package
    exclusion, and the trap is that the regeneration command is written in three
    places, so amending one of them leaves two readers still following the
    unfiltered form. §10 has to name all three.

    Line 310 of `SETUP.md` is the one #1073's clause names; the lock's own header
    and `requirements.txt`'s pointer are the other two, and the assertion below
    re-reads each of them so the section cannot stay green on a remembered
    address.
    """
    section = (DOC.read_text(encoding="utf-8")
               .split("## 10. What Lloyd may change", 1)[1].split("\n## 11", 1)[0])
    assert "SETUP.md:310" in section
    assert "requirements.lock" in section and "header" in section
    assert "requirements.txt" in section and "lines 4-5" in section
    setup = (ROOT / "SETUP.md").read_text(encoding="utf-8").splitlines()
    assert ".venvs/lloyd/bin/python -m pip freeze > requirements.lock" in setup[309]
    lock_head = (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines()[2]
    assert "pip freeze > requirements.lock" in lock_head
    txt = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "then refreeze" in txt[3] and "requirements.lock header" in txt[4]


def test_item_559_carries_the_solver_route_itself():
    """#1073 clause 5: #559's acceptance names the file and the command, not #815.

    The pointer this clause exists to break was real: #559's acceptance deferred
    the route to #815, #815 was folded into #1072, and #1072 is `done` and never
    covered this finding — so the route was deferred to an item that could not
    answer. #559's own body now carries the file (`requirements-dev.txt`), the pin
    (`z3-solver==4.15.4.0`) and the install command, and says in words that the
    route is not #815.

    Read from the live board through `board_presence`, unmarked like the other
    board-reading claims in this tree (`test_bench_split`, `test_bench_invariants`):
    a `live_vault`-marked node is deselected by the gate's own `-m "not
    live_vault"` and certifies nothing — failure mode (1) in the header of
    `tests/test_archived_skill_artifacts.py`. The coupling that buys is real:
    rewriting #559's acceptance reddens the next round that runs this.
    """
    import board_presence

    files = [p for p in board_presence.board_files_or_stop(what="#559 route claim",
                                                           numeric_names=True)
             if p.name.startswith("559-")]
    assert len(files) == 1, f"expected exactly one #559 item file, got {[p.name for p in files]}"
    body = files[0].read_text(encoding="utf-8")
    assert "requirements-dev.txt" in body
    assert "pip install -r requirements-dev.txt" in body
    assert "z3-solver==4.15.4.0" in body
    assert "see #815" not in body, "the acceptance deferred the route to #815 again"


# ── §4 the gate ─────────────────────────────────────────────────────────────

def test_the_gate_uses_reflink_always_not_auto():
    """§10: "--reflink=always, not auto — auto degrades to a real 6GB copy
    silently"."""
    gate = (ROOT / "scripts" / "automod" / "gate.py").read_text()
    assert "--reflink=always" in gate


def test_the_collected_floor_matches_the_doc():
    from scripts.automod import gate
    assert gate.PYTEST_MIN_COLLECTED == 1000


# ── §8.1 regression detector ────────────────────────────────────────────────

def test_latency_is_never_armed():
    """§8.1: latency has no sigma and cannot make a comparison regress.

    The two assertions are byte-identical to the version that pinned the doc
    sentence "Only `latency_ms_avg` moved ... and it is never compared". #1129
    deleted that sentence, because the second half became false — both contexts
    now have a ceiling in `LATENCY_BUDGET_MS` and a run past it is reported. What
    this test still owns is the half that survives it: latency is out of the armed
    set. The reason is measured, not asserted — the daemon's query-embedding
    cache moves it 20-34x on arm order alone (3,672-4,027 ms fresh against
    119-183 ms for the identical cached repeat at pool 240, priced 2026-09-18), so a PAIRED
    delta grades nothing but which arm ran first. What a ceiling may and may not do
    is `test_the_doc_states_the_latency_budget_and_its_field`.
    """
    from workers.sources import automod_regression as R
    assert "latency_ms_avg" not in R.ARMED_METRICS
    assert "latency_ms_avg" in R.REPORT_ONLY


def test_the_doc_states_the_latency_budget_and_its_field():
    """§8.1: the doc states the ceiling, the field, and not the retired band.

    The field name is asserted against `R.OVER_BUDGET_FIELD` rather than a literal,
    because a doc naming a key the code stopped writing is the exact defect this
    file exists to catch — the sentence it replaced was itself only ever true of
    the code as it stood when it was written.
    """
    from workers.sources import automod_regression as R
    assert set(R.LATENCY_BUDGET_MS) == {R.CONTEXT_NIGHTLY, R.CONTEXT_PAIRED_CHECK}
    flat = _flat(DOC)
    assert "it is never compared" not in flat, (
        "the doc still tells the reader latency goes unread, which #1129 ended")
    for phrase in ("LATENCY_BUDGET_MS", R.OVER_BUDGET_FIELD):
        assert phrase in flat, f"the doc does not state the budget ({phrase})"
    # What the doc must now SAY, rather than a bare number it must not contain.
    # `"1.7s" not in flat` was written here first and is gone: `git log
    # -S'1.7s' -- architecture/automod.md` is empty, so this 2,100-line file never
    # carried the asserted band — the module did, and
    # `test_the_budget_comment_prices_itself_on_fresh_queries` is where removing it
    # is pinned. Left here it could only ever fail on an unrelated future
    # paragraph that happened to measure 1.7 seconds of something.
    assert "3,672-4,027 ms" in flat and "119-183 ms" in flat, (
        "the doc no longer states the fresh-vs-cached spread that replaced the "
        "asserted 1.7 s band, so a reader cannot see why latency is graded "
        "absolutely and never paired")


def test_all_seven_are_armed_because_the_corpus_is_now_pinned():
    """§8.1: the armed set was wrong twice, in opposite directions.

    Disarming the document metrics was right while the corpus moved between
    arms. Once BOTH halves are pinned, a frozen qmd snapshot and one shared
    LLOYD_CODE_ROOT, the arms agree to 0.0000 on all seven and four repeat
    runs move 0.0000. So all seven are armed again.
    """
    from workers.sources import automod_regression as R
    assert set(R.ARMED_METRICS) == {
        "entity_hit_rate", "entity_recall_avg", "fact_entity_recall_avg",
        "ndcg10", "mrr_doc", "doc_hit_rate", "doc_recall_avg"}
    assert set(R.REPORT_ONLY) == {"latency_ms_avg", "n_queries"}


def test_the_pin_is_a_precondition_not_an_optimisation():
    """A comparison falling back to the live daemon would be the broken one
    wearing the fixed one's name."""
    src = (ROOT / "workers" / "sources" / "automod_regression.py").read_text()
    assert "PinError" in src and "pinned corpus unavailable" in src


def test_the_doc_states_the_armed_set_the_code_has():
    """§8.1: the doc must state the armed set the code actually holds.

    The armed set went seven → three → seven and the doc kept the middle
    value: `architecture/automod.md` said "the armed three" in two places
    while `ARMED_METRICS` had carried seven since `08e998a`. Pinning the
    number and every name is the only way a count in prose stays a fact.
    """
    from workers.sources import automod_regression as R
    flat = _flat(DOC)
    matches = list(ARMED_STATEMENT_RE.finditer(flat))
    assert matches, ('the doc must state the armed set as '
                     '"**All N are armed:**" followed by the metric names')
    for match in matches:  # every statement, not just the first
        stated_count = NUMBER_WORDS[match.group(1).lower()]
        assert stated_count == len(R.ARMED_METRICS), (
            f"the doc says {match.group(1)} armed, the code has "
            f"{len(R.ARMED_METRICS)}")
        named = re.findall(r"`([a-z0-9_]+)`", match.group(2))
        assert len(named) == len(R.ARMED_METRICS), (
            f"the doc names {len(named)} metrics, the code arms "
            f"{len(R.ARMED_METRICS)}")
        assert set(named) == set(R.ARMED_METRICS)


def test_the_doc_states_no_armed_count_other_than_the_real_one():
    """§8.1: the count itself, in any phrasing, not just the canonical sentence.

    `08e998a` armed seven metrics and left "the armed three" standing twice in
    this doc, so a guard that reads only one sentence shape leaves the other
    shape open. Any number word beside "armed" has to be the number the code
    actually has, which also means a reworded disarm that keeps a wrong count
    is caught here even when it dodges the phrase list below.
    """
    from workers.sources import automod_regression as R
    flat = _flat(DOC)
    wrong = []
    for pattern in ARMED_COUNT_RES:
        for word in pattern.findall(flat):
            if NUMBER_WORDS[word.lower()] != len(R.ARMED_METRICS):
                wrong.append(word)
    assert not wrong, (
        f"architecture/automod.md states an armed count of "
        f"{sorted(set(wrong))}, the code has {len(R.ARMED_METRICS)}")


def test_the_armed_set_comment_header_is_not_duplicated():
    """§8.1: one comment header, not two, above the armed tuples.

    `9a6861c` landed the line twice in a row. Harmless to read, but it is the
    only marker separating the armed-set justification from the tuples below
    it, and a duplicated marker is what a stale copy of the block looks like.
    """
    src = REGRESSION_SRC.read_text(encoding="utf-8")
    assert src.count(ARMED_SET_HEADER) == 1, (
        f"{ARMED_SET_HEADER!r} occurs {src.count(ARMED_SET_HEADER)} times in "
        "workers/sources/automod_regression.py")


def test_the_doc_never_disarms_a_metric_the_code_has_armed():
    """§8.1: "reported and never fire" was false for a week and survived review.

    `evaluate` loops `ARMED_METRICS` and appends a rollback reason for any
    drop past `SIGMA_MULTIPLIER × σ`, and `execute` refuses to run without the
    pin (`PinError` → "pinned corpus unavailable"). So a document metric is
    the line that stops a promotion. A doc saying otherwise is not stale
    prose, it is an instruction to skip the gate.
    """
    from workers.sources import automod_regression as R
    assert "mrr_doc" in R.ARMED_METRICS, (
        "this test guards prose that is only false while the document "
        "metrics are armed; if they are genuinely disarmed, rewrite it")
    flat = _flat(DOC)
    for phrase in DISARMING_PHRASES:
        assert " ".join(phrase.split()) not in flat, (
            f"architecture/automod.md tells the reader {phrase!r} while the "
            "code has that metric armed")
    claims = _disarm_claims(flat, R.ARMED_METRICS)
    assert not claims, "architecture/automod.md disarms an armed metric: " + claims[0]


def test_the_regression_module_comment_never_disarms_an_armed_metric():
    """§8.1: the same false sentence sat above `FACT_LAYER_METRICS` itself.

    `9a6861c` wrote "Re-arming a doc-side metric means first making the doc
    corpus part of the pairing"; `08e998a` shipped that pairing eleven hours
    later and left the sentence standing. The comment beside the tuple is the
    place a reader looks when they doubt the armed set.
    """
    from workers.sources import automod_regression as R
    assert "mrr_doc" in R.ARMED_METRICS
    flat = _flat(REGRESSION_SRC)
    for phrase in DISARMING_PHRASES:
        assert " ".join(phrase.split()) not in flat, (
            f"workers/sources/automod_regression.py says {phrase!r} while "
            "the code has that metric armed")
    claims = _disarm_claims(flat, R.ARMED_METRICS)
    assert not claims, (
        "workers/sources/automod_regression.py disarms an armed metric: "
        + claims[0])


def test_the_doc_still_states_the_limits_that_survive_the_pin():
    """§8.1/§13: pinning the corpus fixed one limit and left two standing.

    `latency_ms_avg` still cannot be compared paired — the cold-to-warm
    embedding-cache spread (3.5-3.9 s here: 3,672-4,027 ms fresh against 119-183 ms
    for the cached repeat at pool 240) is wider than any step worth catching — and
    expiring 70% of the active edge set still moves nothing, so edge quality has no
    armed metric. Rewriting the disarmed-metric prose must not quietly drop either
    one.

    The latency sentence this test used to pin was `"it is never compared"`.
    #1129 replaced it: the limit that survives is that no PAIRED delta is
    graded, while an absolute per-context ceiling is, so this now pins the
    ceiling sentence instead of the exemption — and pins `REPORT_ONLY` itself
    unchanged, since a budget that quietly armed latency would have made this
    whole paragraph a lie.
    """
    from workers.sources import automod_regression as R
    assert "latency_ms_avg" not in R.ARMED_METRICS
    assert "latency_ms_avg" in R.REPORT_ONLY
    assert set(R.REPORT_ONLY) == {"latency_ms_avg", "n_queries"}
    flat = _flat(DOC)
    assert R.OVER_BUDGET_FIELD in flat, "the latency budget verdict left the doc"
    for value in R.LATENCY_BUDGET_MS.values():
        assert f"{int(value):,} ms" in flat, (
            f"the doc does not state the {int(value)} ms ceiling the code has")
    assert "Edge quality has no armed metric" in flat, "the edge limit left the doc"
    assert "Graph EDGE quality is not checked by anything" in flat


def test_the_grep_corpus_is_pinnable():
    """§8.1: this retriever greps the repository it ships in, so the code
    under test is also part of the corpus it is scored against."""
    assert "LLOYD_CODE_ROOT" in (ROOT / "agent_mcp" / "vault.py").read_text()


def test_both_arms_score_the_same_questions():
    """The baseline arm runs the OLD run_eval.py out of a worktree, carrying
    the OLD query set. Editing the eval would otherwise ask the arms different
    questions and score the difference as a code regression."""
    src = (ROOT / "workers" / "sources" / "automod_regression.py").read_text()
    assert "LIVE_QUERIES" in src and '"--queries"' in src


def test_the_noise_file_is_not_in_the_eval_run_record_directory():
    """§13: eval/baselines holds run records, and test_eval_scorer globs it."""
    from workers.sources import automod_regression as R
    assert "eval/baselines" not in str(R.NOISE_PATH)


# ── §11 state ───────────────────────────────────────────────────────────────

def test_state_lives_outside_the_repo():
    """§11: so `git reset --hard` and `git clean -fdx` cannot reach it."""
    from scripts.automod import state as S
    assert ROOT not in S.STATE_DIR.parents and S.STATE_DIR != ROOT


def test_the_ledger_raises_where_autoresearchs_swallows(tmp_path):
    """§11: the documented divergence."""
    from scripts.automod import state as S

    blocker = tmp_path / "f"
    blocker.write_text("not a dir", encoding="utf-8")
    with pytest.raises(OSError):
        S.append_event({"event": "x"}, path=blocker / "nested" / "l.jsonl")


# ── §7.2 the third probe class ──────────────────────────────────────────────

def test_the_http_error_budget_matches_the_doc():
    """§7.2: "The backend's own 503 ... gets its own much wider budget: 36 ticks"."""
    assert policy.PROBE_HTTP_ERROR_STREAK == 36
    assert (policy.PROBE_FAIL_STREAK < policy.PROBE_TIMEOUT_STREAK
            < policy.PROBE_HTTP_ERROR_STREAK)


# ── §8 the chronic set expires ──────────────────────────────────────────────

def test_the_chronic_set_expires_daily():
    """§8: "The chronic set expires after 24 hours, in the cache and in the
    process"."""
    assert policy.CHRONIC_REFRESH_SECONDS == 24 * 3600


# ── §8.1 which metrics can actually see the graph ───────────────────────────

def test_the_armed_metrics_are_named_for_what_they_actually_read():
    """§8.1: measured, not assumed.

    Deleting 70% of fact_idx rows moves entity_hit_rate and entity_recall_avg
    well past tolerance. Expiring 70% of ACTIVE EDGES moves nothing at all —
    so these read the fact layer, not the edge set, and calling them "graph
    sensitive" would be the same overclaim this detector exists to avoid.
    """
    from workers.sources import automod_regression as R
    assert not hasattr(R, "GRAPH_SENSITIVE_METRICS"), \
        "the old name overclaims: edge expiry is invisible to every armed metric"
    assert set(R.FACT_LAYER_METRICS) == {
        "entity_hit_rate", "entity_recall_avg", "fact_entity_recall_avg"}
    for doc_side in ("mrr_doc", "ndcg10", "doc_hit_rate"):
        assert doc_side not in R.FACT_LAYER_METRICS


def test_the_eval_refuses_an_empty_corpus_by_default():
    """§8.1: "refuses an empty one unless --allow-empty-corpus is passed"."""
    src = (ROOT / "eval" / "run_eval.py").read_text()
    assert "--allow-empty-corpus" in src
    assert "corpus_ok" in src


def test_the_quality_check_compares_against_the_parent_not_the_lkg():
    """§8.1: after settling the LKG pointer IS the promoted commit."""
    src = (ROOT / "workers" / "sources" / "automod_regression.py").read_text()
    assert 'subject.get("parent")' in src


# ── §6 landing ──────────────────────────────────────────────────────────────

def test_the_landing_is_detached():
    """§2/§6: the promoter restarts the process it is usually called from."""
    assert "spawn_detached" in (ROOT / "agent_mcp" / "automod.py").read_text()
    assert "start_new_session=True" in (ROOT / "scripts" / "automod" / "state.py").read_text()


def test_only_one_promotion_may_be_under_observation():
    """§6 step 0: a second landing overwrote current.json, so the first never
    settled and the new rollback target had never survived a window."""
    src = (ROOT / "scripts" / "automod" / "promote.py").read_text()
    assert "still under observation" in src


def test_the_promoter_refuses_a_commit_the_gate_did_not_judge():
    """§6 step 0."""
    src = (ROOT / "scripts" / "automod" / "promote.py").read_text()
    assert "gate_head" in src and "re-gate before landing" in src


# ── §7.4 route selection ────────────────────────────────────────────────────

def test_a_rollback_can_revert_in_place():
    """§7.4: "Reset when HEAD is still the promotion; revert in place when it
    is not" — nightly jobs commit straight to live main."""
    import rollback as rb
    assert hasattr(rb, "revert_commit")
    assert "surgical" in (ROOT / "agent-services" / "guardian" / "guardian.py").read_text()


# ── §11 state ───────────────────────────────────────────────────────────────

def test_the_new_state_files_exist_where_the_doc_says():
    from scripts.automod import state as S
    assert S.LAST_SETTLED_PATH.name == "last_settled.json"
    assert S.ROLLBACK_REQUEST_PATH.name == "rollback_request.json"
    assert S.EVAL_LAST_PATH.name == "eval_last.json"


def test_the_aggregator_verdict_streak_matches_the_doc():
    """§7.2: the aggregator's verdict is confirmed across ticks."""
    assert policy.MCP_FATAL_STREAK >= 2


def test_a_round_requires_an_observer():
    """§2: "A round runs under Inner Voice, or not at all"."""
    src = (ROOT / "agent_mcp" / "automod.py").read_text()
    assert "_inner_voice_gate" in src
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert cfg["automod"]["require_inner_voice"] is True


# ─── the qmd fork: the one tree outside the gate (backlog #854) ───────────
#
# `~/lloyd/qmd` is a separate clone the outer repo ignores, so no rung builds or
# tests it and no rollback reverts it — while `agent-qmd-daemon` serves its
# `dist/` and so answers the vector leg of retrieval. The only defence a round
# has is the two documents it actually reads stating the rule, so these pin the
# sentences, not just the word "qmd": a mention in passing satisfies `grep -ci
# qmd` and tells a round nothing.

AUTOMOD_SECTION_START = "## Automod (self-modification)"


def _automod_section() -> str:
    """CLAUDE.md's Automod section only: everything from its heading up to the
    next top-level heading. A rule stated anywhere else in a file of CLAUDE.md's
    size is not in the section a round is pointed at."""
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    start = text.index(AUTOMOD_SECTION_START)
    rest = text[start + len(AUTOMOD_SECTION_START):]
    end = rest.index("\n## ")
    return rest[:end]


def _assert_fork_rule(text: str, where: str) -> None:
    """The four things the rule has to say, each checked on the words a round
    would act on. Prose is whitespace-normalised first: these files are
    hand-wrapped at ~80 columns, and a check that broke on a newline in the
    middle of a phrase would fail for a reason no reader could see."""
    low = " ".join(text.split()).lower()
    assert "qmd" in low, f"{where}: never mentions qmd at all"
    # 1. What the tree is: a separate clone the outer repo ignores, so it is
    # neither a submodule nor present in a round's worktree.
    for phrase in ("gitignore", "separate", "clone", "not a submodule"):
        assert phrase in low, f"{where}: does not say the fork is a separate gitignored clone, not a submodule"
    # 2. That no rung builds or tests it, which is the whole hazard.
    assert "never build" in low, f"{where}: does not say the gate never builds the fork"
    # 3. The prohibition, aimed at a round, naming the path.
    assert "must not edit" in low, f"{where}: does not forbid a round editing qmd/src/**"
    assert "qmd/src" in low, f"{where}: never names the path it forbids editing"
    # 4. Where a fork change goes instead, in the order the steps run.
    steps = low[low.index("human-landed"):low.index("human-landed") + 300]
    for step in ("build", "suite", "branch `lloyd`", "push to `origin`"):
        assert step in steps, f"{where}: human-landing steps omit {step!r}"
    assert (steps.index("build") < steps.index("suite") < steps.index("branch `lloyd`")
            < steps.index("push to `origin`")), f"{where}: the fork's landing steps are out of order"


def test_claude_md_states_the_fork_rule_in_the_automod_section():
    """Clause 1 of #854. The section is the one a round is sent to by the
    implement prompt's `grep '^## Automod' CLAUDE.md`; a sentence about the
    fork anywhere else in the file is not where it would look."""
    section = _automod_section()
    assert section, "CLAUDE.md has no Automod section body"
    _assert_fork_rule(section, "CLAUDE.md Automod section")


def test_the_automod_skill_states_the_fork_rule_where_a_round_reads_it():
    """Clause 2 of #854. `grep -ci qmd` on this file returned 0 when the item
    was triaged; the count is the cheap half — the same sentence has to be in
    the Boundaries section, which is the section that tells a round what it may
    not touch.

    No skip when the file is missing: this rule's whole subject is a tree
    outside the gated repo, so a guard that could go quietly unverified would
    repeat the defect it is guarding. The path is not worktree-relative —
    `app.paths.VAULT_ROOT` is `Path.home() / "obsidian"` (`app/paths.py:11`), so
    every worktree reads the same live vault, and a missing file is a real
    regression rather than an artefact of where the test was run.
    `tests/test_skill_tool_names.py`'s `SKILLS_DIRS` reads that same tree
    unguarded, so this test may too."""
    from app.paths import VAULT_ROOT

    skill = VAULT_ROOT / "skills" / "automod-change-own-code" / "SKILL.md"
    assert skill.is_file(), f"clause 2's subject is absent: {skill} does not exist"
    text = skill.read_text(encoding="utf-8")
    boundaries = text[text.index("## Boundaries"):]
    _assert_fork_rule(boundaries[:4000], "SKILL.md Boundaries")
    # A round that cannot edit the fork still has to close the item, so the
    # skill has to say what to do instead of silently reporting a change.
    assert "human_paths" in boundaries[:4000], "skill does not name the route out"


# --- the axis the post-landing check cannot see (#829) ----------------------

def _prose(src: str) -> str:
    """Lowercase with comment sigils and line wraps collapsed away.

    Both prose surfaces here are hand-wrapped, so a phrase pin written against the
    raw text would break on a reflow that changed nothing about the claim.
    `test_the_worker_states_what_the_pinned_corpus_cannot_see` collapses the same
    way for the same reason; a wrapped sentence and an absent sentence are
    different findings and the test must not confuse them.
    """
    return re.sub(r"\s+", " ", src.replace("#", " ").replace("*", " ")).lower()


def test_the_post_landing_check_no_longer_calls_itself_behavioural():
    """Clause 1 of #829. The one differential post-landing check opened with
    "Nightly behavioural-regression check" and logged a rollback reason reading
    "behavioural regression", while `ARMED_METRICS` is the seven retrieval metrics
    and `eval/run_eval.py` issues no model request at all. So the name promised an
    axis no armed metric can move: dropping `Grep` from the baseline tool set
    leaves every one of the seven exactly where it was.

    Pinned on the whole module text rather than the docstring's first line,
    because the word sat in four places at once — the docstring, the
    `logger.error`, the rollback `reason` and the argparse description — and a pin
    on one of them leaves the other three claiming coverage the file does not
    have. The `config.yaml` comment naming the same job is a human path (the loop
    may not write that file) and is not asserted here.
    """
    src = REGRESSION_SRC.read_text(encoding="utf-8")
    assert "behavioural" not in src.lower(), "the worker still describes itself as behavioural"
    assert "behavioral" not in src.lower(), "the US spelling of the same overclaim"
    # What replaced the word, at each of the four sites.
    assert src.splitlines()[0].startswith(
        '"""Nightly retrieval-quality regression check'), src.splitlines()[0]
    assert 'logger.error("retrieval-quality regression after' in src
    assert 'reason = (f"retrieval-quality regression after' in src
    assert 'description="Paired retrieval-quality regression check"' in src


def test_the_worker_states_the_agent_loop_limit_beside_the_edge_set_limit():
    """Clause 2 of #829: the loop-side blindness has to be in the file a reader of
    the green result opens, next to the coverage statement that file already
    carries for graph edges.

    It must say two things, and the second is what stops the rewrite
    overcorrecting. A scored loop-side check DOES exist — the gate's
    `prompt_surface` rung — but only pre-landing, and only when the diff names one
    of five paths. Those names are compared against the gate's own tuples, so the
    prose cannot drift from the trigger the way a hand-copied list drifts from the
    code it describes.
    """
    from scripts.automod.gate import Gate

    src = REGRESSION_SRC.read_text(encoding="utf-8")
    low = _prose(src)
    assert "tool call" in low and "turn count" in low and "model decision" in low, \
        "the worker never says what it cannot observe on the loop side"
    assert "prompt_surface" in src, "the one loop-side check is not named"
    surface = Gate.PROMPT_SURFACE_PATHS + Gate.PROMPT_SURFACE_VAULT
    assert len(surface) == 5, f"the five-path claim no longer matches the gate: {surface}"
    for name in surface:
        assert name in src, f"{name} is a prompt-surface path but is not named in the worker"

    # Beside the edge-set limit: after the edge-blind measurement, inside the same
    # coverage block (which ends at FACT_LAYER_METRICS).
    armed = src.index("ARMED_METRICS = (")
    edge = src.index("unchanged  <-- blind")
    loop = src.index("THE SAME STATEMENT, ONE AXIS OVER")
    end = src.index("FACT_LAYER_METRICS = (")
    assert armed < edge < loop < end, (
        f"the loop-side limit is not beside the edge-set limit "
        f"(armed={armed}, edge={edge}, loop={loop}, end={end})")
    # And it must not claim that nothing observes the loop.
    assert "nothing observes the loop" in low, \
        "the correction that a pre-landing loop-side check exists is missing"
    assert "pre-landing" in low, "the loop-side check is not marked as pre-landing only"


def test_architecture_states_the_loop_axis_and_stops_naming_the_check_behavioural():
    """Clause 3 of #829, in the two places a human reads instead of the source.

    §8's detector table row and the §8.1 heading both called this check
    behavioural, which is what made the gap invisible from the doc: the section
    was honest about retrieval *scope* ("did this commit make retrieval worse",
    not "is retrieval good") while its title promised the loop. §13 is where a
    reader goes to ask what the loop cannot see, so the new bullet goes there,
    beside the existing edge-quality one — and it has to name the pre-landing
    `prompt_surface` rung, because "nothing observes the agent loop" would be a
    false sentence about a gate that scores tool choice today.
    """
    text = DOC.read_text(encoding="utf-8")
    rows = [ln for ln in text.splitlines() if ln.startswith("| retrieval-quality regression")]
    assert len(rows) == 1, f"the detector table row is gone or duplicated: {rows}"
    row = rows[0].lower()
    assert "behavioural" not in row and "behavioral" not in row
    assert "loop" in row, "the table row does not point at the axis it cannot see"
    assert "### 8.1 Retrieval-quality regression: measured, not assumed" in text
    assert "### 8.1 Behavioural" not in text and "### 8.1 Behavioral" not in text

    sec13 = text.split("## 13. Known limits")[1]
    # Asserted against the NEW BULLET, not all of §13. 'tool call' already
    # appeared in §13's edge-quality bullet before this change ("a change that
    # adds a tool call or a retrieval behaviour is invisible to every detector
    # here"), so a phrase pin on the section could be satisfied by the pre-existing
    # text and report the gap closed while the new bullet was absent. The exact
    # conjunction — all three things in one clause — belongs to the new bullet
    # alone, so that is what gets pinned, and it is pinned inside the bullet.
    marker = "- **Agent-LOOP behaviour is not covered after landing either**"
    assert marker in sec13, "no §13 bullet names the agent-loop axis"
    bullet = _prose(sec13.split(marker, 1)[1].split("\n- ", 1)[0])
    assert "a tool call, a turn count or a model decision" in bullet, \
        "the §13 loop bullet never says what no armed metric can observe"
    assert "pre" in bullet and "landing" in bullet, \
        "the §13 loop bullet does not mark the surviving check as pre-landing only"
    assert "prompt_surface" in sec13, "§13 does not name the loop-side check that does exist"
    for name in ("prompt_builder.py", "prefetch.py", "SOUL.md", "MEMORY.md", "USER.md"):
        assert name in sec13, f"{name} is a prompt-surface path §13 does not name"
    # The last-known-good `eval` slot is the reader's endpoint: §13 has to say
    # what it covers and that a carried-over number is not this commit's.
    assert "eval` slot" in sec13, "§13 does not address the last-known-good eval slot"
    assert "eval_for_recorded_commit" in sec13, \
        "§13 does not name the field that says whether the eval is this commit's"


# ---------------------------------------------------------------------------
# The rung table vs the ladder the gate actually runs (#679)
# ---------------------------------------------------------------------------

#: The module's `NUMBER_WORDS` stops at ten, and widening it would widen the
#: armed-metric regexes that share it. The ladder needs eleven.
RUNG_NUMBERS = {**NUMBER_WORDS, "eleven": 11}


def _real_ladder(monkeypatch) -> list[str]:
    """The rung names `Gate.run` executes, in order, by stubbing `_rung`."""
    from scripts.automod import gate as G
    names: list[str] = []
    g = G.Gate("SM_DOC_CLAIMS", ROOT, "HEAD")

    def _record(name, fn):          # every rung "passes": only the names matter
        names.append(name)
        return True

    monkeypatch.setattr(g, "_rung", _record)
    g.run()
    return names


def test_the_doc_rung_table_lists_the_ladder_that_actually_runs(monkeypatch):
    """The doc's rung table is a claim about the code, so it gets pinned to it.

    This is the shape §4 has always had a silent gap in: adding `vet` to the
    ladder is a change to eleven names, and only one of the doc's two places
    that enumerate them (`architecture/testing.md` counted `nine rungs` until
    this round) said anything at all. A rung missing from this table is a rung
    an operator reading §4 at 3am does not know exists — which is exactly how
    `prompt_surface` sat unlisted here for a week while the ladder ran it.
    """
    real = _real_ladder(monkeypatch)
    text = DOC.read_text(encoding="utf-8")

    start = text.index("| Rung | Typical | Catches |")
    rows = []
    for line in text[start:].splitlines()[2:]:
        if not line.startswith("|"):
            break
        rows.append(line.split("|")[1].strip())
    assert rows, "no rung table found in §4"
    assert rows == real, (
        f"doc lists {rows} but the gate runs {real}; "
        "a rung must be added to the table in the same commit that adds it "
        "to the ladder")


def test_the_doc_rung_count_matches_the_ladder_that_runs(monkeypatch):
    """§4 names the rung count in its opening sentence, so the count is checked.

    A hand-typed integer in prose is a second definition of the ladder, and the
    second definition is the one that goes stale: `architecture/testing.md` said
    "nine rungs" while the gate ran ten, and it took this round adding an
    eleventh to notice. The number the doc states is now compared with the
    length of the ladder read out of `Gate.run`, so the sentence cannot survive
    a rung being added or removed without being edited in the same commit.
    """
    real = len(_real_ladder(monkeypatch))
    stated = re.search(r"\b([A-Za-z]+) rungs, cheapest first",
                       DOC.read_text(encoding="utf-8"))
    assert stated, "§4 no longer states the rung count next to the ladder"
    word = stated.group(1).lower()
    assert word in RUNG_NUMBERS, f"unparseable rung count {word!r}"
    assert RUNG_NUMBERS[word] == real, (
        f"§4 states {word!r} rungs but Gate.run executes {real}")
