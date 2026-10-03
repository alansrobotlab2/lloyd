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

# How many paths the `prompt_surface` rung triggers on, as a prose count. Both
# shapes the corpus uses: "one of six path names" (the worker's limit block),
# "one of six paths" (§13) and "for the six path names" (§8.1). A guard that read
# only one shape is how the armed-set count rotted once (§8.1 said "the armed
# three" beside "All seven are armed" for a week), so the count is read however it
# is phrased and then checked against `Gate.PROMPT_SURFACE_PATHS +
# PROMPT_SURFACE_VAULT` — never against a number carried here, which is the second
# copy that goes stale.
TRIGGER_COUNT_RE = re.compile(r"\b(?:one of|for the) ([a-z]+) paths?(?: names)?\b")

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
    # #1883: the tracked Qwen3-TTS patch. The row is the operator-facing claim
    # — a round may rewrite the frame cap in this file, and only this file under
    # `agent-services/services/`, because the clone it patches is untracked.
    ("agent-services/services/tts/qwen3-tts-local.patch", "allowed"),
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

    This is §10 of *this* file only. The same two facts are asserted against
    `SETUP.md` Part 4 by the #1378 tests below
    (`test_setup_names_the_dev_file_its_install_command_and_the_candidate_targets`).
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


def _dev_packages() -> list[str]:
    """Package names `requirements-dev.txt` lists, read the way the refreeze reads them."""
    import re
    text = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    return [m.group(1) for m in re.finditer(r"(?m)^([A-Za-z0-9._-]+)", text)]


def _setup_lloyd_venv_section() -> str:
    text = (ROOT / "SETUP.md").read_text(encoding="utf-8")
    return text.split("### `lloyd` — backend", 1)[1].split("\n### ", 1)[0]


def test_the_dev_requirements_file_pins_the_solver_and_says_it_is_never_installed():
    """#1378 clause 1: the file exists, carries the pin, and its header says it is
    never installed by the candidate venv and never frozen into the lock."""
    text = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "z3-solver==4.15.4.0" in text.splitlines()
    header = " ".join(l.lstrip("# ") for l in text.splitlines() if l.startswith("#"))
    assert "NEVER installed by the automod gate's candidate venv" in header
    assert "NEVER frozen" in header and "requirements.lock" in header
    assert _dev_packages() == ["z3-solver"]


def test_setup_names_the_dev_file_its_install_command_and_the_candidate_targets():
    """#1378 clauses 2 and 4: SETUP.md Part 4's lloyd section names the file, the
    exact install command, and that the candidate venv installs only from the lock
    else requirements.txt — and that last half is re-derived from the code that
    decides it, so a change to the rung's targets reddens here."""
    from scripts.automod import spec
    section = _setup_lloyd_venv_section()
    assert "requirements-dev.txt" in section
    assert ".venvs/lloyd/bin/python -m pip install -r requirements-dev.txt" in section
    flat = " ".join(section.split())
    assert "installs only from `requirements.lock`, else `requirements.txt`" in flat
    assert spec.touches_requirements(["requirements.lock"])
    assert spec.touches_requirements(["requirements.txt"])
    assert not spec.touches_requirements(["requirements-dev.txt"])


def test_both_refreeze_sites_exclude_every_dev_package():
    """#1378 clause 3 (#1073 clause 2): the lock is a `pip freeze` snapshot, so a
    dev package installed out of band would be swept into the rebuild set by the
    next refreeze. Both surviving copies of the command — SETUP.md Part 4 and the
    lock's header — exclude what requirements-dev.txt lists, and name that; the
    unfiltered form is gone from all three files. `requirements.txt` only points at
    the lock header, which is still what its lines 4-5 say."""
    import re
    import subprocess
    for name in ("SETUP.md", "requirements.lock", "requirements.txt"):
        assert "pip freeze > requirements.lock" not in (ROOT / name).read_text(
            encoding="utf-8"), f"{name} still carries the unfiltered refreeze"
    setup_cmds = [l for l in _setup_lloyd_venv_section().splitlines()
                  if "pip freeze" in l and "> requirements.lock" in l]
    lock_cmds = [l.lstrip("# ") for l in (ROOT / "requirements.lock").read_text(
        encoding="utf-8").splitlines()[:8] if "pip freeze" in l]
    assert len(setup_cmds) == 1 and len(lock_cmds) == 1
    for cmd in (setup_cmds[0], lock_cmds[0]):
        assert "requirements-dev.txt" in cmd and "--exclude" in cmd
        # Run the command's own exclusion clause and check it names every package.
        clause = re.search(r"\$\((sed .*)\) > requirements\.lock", cmd).group(1)
        out = subprocess.run(["bash", "-c", clause], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.split()
        assert [out[i + 1] for i in range(0, len(out), 2)] == _dev_packages()
        assert set(out[0::2]) == {"--exclude"}
    assert "excluding every package" in " ".join(_setup_lloyd_venv_section().split())
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


def test_the_observer_requirement_is_off_by_default(monkeypatch):
    """§2's "a round runs under Inner Voice, or not at all" was retired on
    2026-09-24 (IV plan R5): since 09-12 it refused only chat-driven rounds,
    and every turn is recorded and runs the turn guards now. The gate still
    exists behind `automod.require_inner_voice: true`."""
    src = (ROOT / "agent_mcp" / "automod.py").read_text()
    assert "_inner_voice_gate" in src
    import agent_mcp.automod as M
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "automod", {k: v for k, v in
                                            (CONFIG.get("automod") or {}).items()
                                            if k != "require_inner_voice"})
    assert M._require_inner_voice() is False


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
    of the paths in the gate's own tuples. Both the names and the count the prose
    states are read off those tuples, so the prose cannot drift from the trigger
    the way a hand-copied list drifts from the code it describes — and so that
    adding a trigger path moves the expectation with the code instead of turning a
    correct fix into a red test. #1758 is that case: `app/prompt_surface.py`, the
    module that defines the contract's ceilings, became the sixth, and the
    `len(surface) == 5` that used to sit here refused it from the wrong side.
    """
    from scripts.automod.gate import Gate

    src = REGRESSION_SRC.read_text(encoding="utf-8")
    low = _prose(src)
    assert "tool call" in low and "turn count" in low and "model decision" in low, \
        "the worker never says what it cannot observe on the loop side"
    assert "prompt_surface" in src, "the one loop-side check is not named"
    surface = Gate.PROMPT_SURFACE_PATHS + Gate.PROMPT_SURFACE_VAULT
    for name in surface:
        assert name in src, f"{name} is a prompt-surface path but is not named in the worker"
    # The count the prose states, checked against the gate's tuples rather than
    # against a number this test also carries. A count in prose is a claim about
    # the code, and §8.1's armed-set count is already pinned this way for exactly
    # the reason it went wrong once: the code moved, the sentence stayed.
    stated = TRIGGER_COUNT_RE.search(low)
    assert stated, (
        "the worker never states how many paths the prompt_surface rung triggers "
        "on, so the count could not be checked against the gate")
    assert NUMBER_WORDS.get(stated.group(1)) == len(surface), (
        f"the worker says '{stated.group(1)}' trigger paths; "
        f"Gate.PROMPT_SURFACE_PATHS + PROMPT_SURFACE_VAULT has {len(surface)}: "
        f"{surface}")

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
    # Same rule as the worker's limit block: the names and the count come from
    # the gate's tuples, so a trigger path added to the gate cannot leave the doc
    # and this test standing on a five-name list that is no longer the trigger.
    from scripts.automod.gate import Gate

    surface = Gate.PROMPT_SURFACE_PATHS + Gate.PROMPT_SURFACE_VAULT
    for name in surface:
        assert name in sec13, f"{name} is a prompt-surface path §13 does not name"
    stated = TRIGGER_COUNT_RE.findall(_flat(DOC))
    assert stated, "the doc never states how many paths the prompt_surface rung triggers on"
    for word in stated:
        assert NUMBER_WORDS.get(word) == len(surface), (
            f"architecture/automod.md states {word} trigger paths; the gate has "
            f"{len(surface)}: {surface}")
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


WORKERS_JOBS = ROOT / "architecture" / "workers-jobs.md"


def test_workers_jobs_counts_and_rosters_every_registered_source():
    """#1015: the roster said twelve while thirteen were registered, and the
    missing one (`board-steward`) was the source running observed by nobody's
    decision. The count is read off `register(` lines, not restated."""
    text = WORKERS_JOBS.read_text()
    init = (ROOT / "workers" / "sources" / "__init__.py").read_text()
    registered = len(re.findall(r"^register\(", init, re.M))
    words = {v: k for k, v in {**NUMBER_WORDS, "eleven": 11, "twelve": 12,
                                "thirteen": 13, "fourteen": 14,
                                "fifteen": 15}.items()}
    m = re.search(r"^(\w+) sources are registered\.", text, re.M)
    assert m, "the roster's count sentence is gone"
    assert m.group(1).lower() == words[registered], (m.group(1), registered)
    names = re.findall(r'^NAME = "([a-z-]+)"', "\n".join(
        p.read_text() for p in (ROOT / "workers" / "sources").glob("*.py")), re.M)
    for name in names:
        assert f"| `{name}` |" in text, f"no roster row for {name}"


def test_workers_jobs_documents_the_board_steward():
    text = WORKERS_JOBS.read_text()
    m = re.search(r"^### `board-steward`.*?(?=^### |^## )", text, re.M | re.S)
    assert m, "no ### board-steward entry"
    body = m.group(0)
    assert "apply: false" in body
    assert "primary" in body and "secondary" in body
    assert "never set `done`" in body


# ── who may open the knowledge-graph store (#1525) ─────────────────────────
# Three sentences claimed `app.kg_store` was the store's only opener. All three
# were false: the guardian's data-damage tripwire has opened it read-only for a row
# count since it shipped. A rule nobody can act on is worse than no rule — a reader
# who believed them treated the guardian's handle as the violation and "fixed" the
# watchdog. Each site now names the second opener, and the wording that lied is
# gone from the corpus.

KG_OPENER_CLAIM_SITES = [
    ("app/paths.py, the store comment",
     ROOT / "app" / "paths.py",
     r"(?sm)^# The knowledge-graph store:.*?^VAULT_KG_DB_DEFAULT[^\n]*"),
    ("architecture/autonomy-jobs.md, the One store paragraph",
     ROOT / "architecture" / "autonomy-jobs.md",
     r"(?sm)^\*\*One store.*?(?=\n\n---\n)"),
    ("CLAUDE.md, the Knowledge graph section",
     ROOT / "CLAUDE.md",
     r"(?sm)^## Knowledge graph\n\n.*?(?=^- An unreadable store)"),
]

#: The three spellings that claimed a sole opener, verbatim from the filing. Each
#: is asserted absent rather than matched-and-counted, so a rewording that keeps
#: the meaning ("only that module ever opens it") still has to be caught by hand —
#: which is why the per-site assertions above require the guardian's name in the
#: same sentence rather than merely the absence of a phrase.
SOLE_OPENER_PHRASES = ("Nothing opens it except", "nothing else opens",
                       "Nothing opens the store except")


@pytest.mark.parametrize("label, path, pattern", KG_OPENER_CLAIM_SITES)
def test_each_sole_opener_sentence_now_names_the_read_only_guardian(label, path,
                                                                    pattern):
    """Both halves, inside one extracted region: the site still says `app.kg_store`
    owns the writing, and it names the guardian's read-only read as the second
    opener. Extracting the region is the point: `CLAUDE.md` mentions the guardian
    21 times and `app/paths.py` not at all outside its own comment, so a
    file-wide search would pass on the wrong sentence."""
    text = path.read_text()
    m = re.search(pattern, text)
    assert m, f"{label}: the region moved or lost its heading marker"
    body = m.group(0)
    assert "writer" in body.lower(), f"{label} no longer says who owns the writing"
    assert "guardian" in body.lower(), f"{label} names no second opener"
    assert "count_kg_rows" in body, (
        f"{label} does not name the read that is the second opener")
    assert "read-only" in body, f"{label} does not say the second read is read-only"
    for phrase in SOLE_OPENER_PHRASES:
        assert phrase not in body, f"{label} still says {phrase!r}"


def test_no_doc_in_the_corpus_still_claims_a_sole_opener():
    """The item's own check, over prose AND docstrings.

    Wider than the three filed files because the claim lived in both media:
    correcting those sentences left two more copies inside
    `eval/counterfactual.py:532` and `eval/run_fact_write_gate_eval.py:56`, each
    quoting the CLAUDE.md line as though it were still the rule. No assertion
    covered those quotations, so nothing noticed the file they cite changing
    underneath them. A fourth copy elsewhere would be believed by the next reader
    exactly as these five were.
    """
    here = Path(__file__).resolve()          # this file states the phrases to ban
    targets = (list((ROOT / "architecture").rglob("*.md"))
               + list((ROOT / "app").rglob("*.py"))
               + list((ROOT / "eval").rglob("*.py"))
               + [ROOT / "CLAUDE.md"])
    hits = []
    for path in targets:
        if path.resolve() == here:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits += [(str(path.relative_to(ROOT)), phrase)
                 for phrase in SOLE_OPENER_PHRASES if phrase in text]
    assert hits == [], f"sole-opener claims still standing: {hits}"


# ── #1790: what the prompt_surface rung's two prose carriers and the doc say ──
#
# The clause exists because the comment outlived the layout it described. As
# written by `d5edc6cb` (2026-09-12) the eval really did keep its baseline inside
# the checkout, and `SM_20260908_165950` lost one that way, so the launch was
# pinned to the live tree and the comment said why. `6426668b` moved runtime data
# out of the tree ten days later and the comment stayed, and the sentence then
# did what a stale comment always does: it justified a kwarg that had become the
# defect, and `architecture/measurement.md` went and cited it as the reason. So
# this is a prose pin with teeth — it names the phrase the comment used and the
# fact the comment must now carry, and it reads the doc section rather than the
# file, because a correct sentence in the wrong section is what #1790 was.

#: The claim both carriers had, in their own words. Asserted absent from both.
STALE_BASELINE_CLAIM = "next to the script"

#: The post-`6426668b` fact each carrier has to state instead. The data-root half
#: and the import half, because the comment conflated them: one is why the
#: baseline survives a teardown, the other is why the cwd decides what is scored.
REQUIRED_FACTS = ("6426668b", "EVAL_BASELINES_DIR")


def _rung_source() -> str:
    import inspect

    from scripts.automod import gate as G
    return inspect.getsource(G.Gate.rung_prompt_surface)


def _measurement_gate_section() -> str:
    """§Which measurement stands between a round and landing, and nothing else.
    Extracted, because the doc mentions `prompt_surface` in its out-of-scope list
    too, and a node that searched the file would pass on that line."""
    text = (ROOT / "architecture/measurement.md").read_text(encoding="utf-8")
    m = re.search(r"^## Which measurement stands between a round and landing$"
                  r".*?(?=^## )", text, re.M | re.S)
    assert m, ("architecture/measurement.md lost the section #1790 has to state "
               "the rung's job in")
    return m.group(0)


def test_the_prompt_surface_carriers_say_why_the_worktree_is_safe_now():
    """Clause 4: the comment, the test and the doc each carry the current reason,
    and none of them still says the baseline lives beside the script.

    Three surfaces, one fact. `inspect.getsource` is what makes the comment check
    real — a comment inside the method is part of the method's source, so this
    grades the words a future reader finds beside the kwarg and not a copy of them
    kept somewhere greener. The test file is read rather than imported so the
    assertion covers its docstring whether or not the node that owns it is
    collected.
    """
    rung = _rung_source()
    assert STALE_BASELINE_CLAIM not in rung, (
        "rung_prompt_surface's comment is back to claiming the eval keeps its "
        "baseline beside the script, the justification `6426668b` falsified")
    assert "sys.path" in rung and "6426668b" in rung, (
        "the comment no longer says what the cwd actually decides (which tree's "
        f"`app/` the child imports) or when that stopped mattering: {rung[:200]!r}")
    assert "live_data" in rung, (
        "the comment no longer names the flag that keeps the baseline outside "
        "both trees, which is the reason the worktree launch is safe")

    test_src = (ROOT / "tests/test_gate_prompt_surface_rung.py").read_text(
        encoding="utf-8")
    assert STALE_BASELINE_CLAIM not in test_src, (
        "the inversion's own file still recites the dead justification")
    assert "6426668b" in test_src and "EVAL_BASELINES_DIR" in test_src, (
        "the test's docstring states neither the date the layout ended nor where "
        "the baseline really goes")

    section = _measurement_gate_section()
    assert "worktree" in section and "candidate" in section, (
        "the gate section no longer says the rung scores the candidate's tree")
    assert STALE_BASELINE_CLAIM not in section, (
        "the doc is reciting the comment again, which is the whole failure mode "
        "this clause exists to stop")
    assert "live tree, not the worktree" not in section
    for fact in REQUIRED_FACTS:
        assert fact in section, f"the section no longer states {fact!r}"


def test_the_measurement_doc_still_names_the_live_data_root_as_named():
    """The doc's second half: it must say the data root is NAMED, because "the
    child resolves production's root anyway" was the sentence that let an unset
    variable read as a location.

    `unset` is the word that made the old prose wrong — not false, but wrong in
    the way that costs a round its record — so it is asserted absent from the
    section, and the positive form is asserted present.
    """
    section = _measurement_gate_section()
    assert "leaves `LLOYD_DATA` unset" not in section, (
        "the doc is back to describing live_data as leaving the variable unset, "
        "which is true of the code #1790 replaced and false of the code it left")
    assert "names the production data root" in section, (
        "the doc no longer says the live data root is exported by name, the one "
        "thing that keeps a worktree launch's baseline findable")
    assert ".lloyd-data" in section, (
        "the doc no longer names where an inherited root would put the record — "
        "inside the worktree, which is the whole reason it must not be inherited")


# ─── the Venv rule: the interpreter a round's worktree can actually execute ──
#
# CLAUDE.md's Project Overview is the first thing a session reads, and until
# #1928 its Venv entry prescribed `.venvs/lloyd/bin/python` as the interpreter
# for "every lloyd script". That path is cwd-relative and `.venvs/` is
# gitignored (`.gitignore:3`), so it exists only in the live checkout: inside a
# round worktree a command written that way dies with `No such file or
# directory` (exit 127), which #692 measured against 3 passes for the same
# pytest invocation run through the absolute interpreter `automod_start`
# returns. The mechanism shipped with #1611 — `scripts/automod/round.py`
# `live_venv_python()`, carried in the start response and written into the
# round's `run_spec.yaml` — and the skill has told a round to use it since vault
# `54663efc`. What stayed wrong was the doc a session reads *before* the skill,
# which stated the opposite as a blanket rule. These nodes pin the doc to the
# mechanism, in the one region a session reads first.

#: The absolute form has to be this tree's, not merely a path beginning with
#: `/`: the file states machine paths in this form elsewhere, and a round's
#: worktree sits under `/home/alansrobotlab/lloyd-work/`, which a looser prefix
#: would also accept.
LIVE_ROOT_PREFIX = "/home/alansrobotlab/lloyd/"

#: The relative interpreter only — the leading `/` of the absolute path is held
#: back by the lookbehind. A plain substring count is 4 both before and after
#: the fix, because the absolute path ends in exactly this text, which is why
#: #1928's original check (2) could never go to 3 and is not the pin here.
RELATIVE_VENV_RE = re.compile(r"(?<!/)\.venvs/lloyd/bin/python")

PROJECT_OVERVIEW_RE = re.compile(r"(?sm)^## Project Overview\n\n.*?(?=^## )")

#: A region is cut into statements at sentence ends and at markdown bullet
#: markers, because the Venv rule is a bullet list. Splitting on a bullet is
#: what makes "named in the same breath" checkable: the statement that names
#: `venv_python` has to carry the rest of the rule itself, not lean on an
#: adjacent bullet. And the region's lines are joined before splitting, so a
#: future rewrap of the prose at 80 columns cannot move a fact out of reach.
STATEMENT_BREAK_RE = re.compile(r"(?<=[.!?])\s+|\s+-\s+")


def _venv_rule_region() -> str:
    """CLAUDE.md's `**Venv**:` bullet and any line indented under it, whitespace
    normalised.

    The section is fixed to Project Overview because that is what the clause
    pins: a `venv_python` stated three headings away, in the Automod section,
    would satisfy a whole-file grep and still leave the first rule a session
    reads prescribing an interpreter its worktree cannot execute.
    """
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    m = PROJECT_OVERVIEW_RE.search(text)
    assert m, ("CLAUDE.md has no `## Project Overview` section — that is where "
               "the Venv rule a session reads first lives")
    lines = m.group(0).splitlines()
    starts = [i for i, ln in enumerate(lines) if "**Venv**:" in ln]
    assert len(starts) == 1, (
        f"Project Overview has {len(starts)} '**Venv**:' entries, so which one "
        "a session reads first is not decidable")
    region = [lines[starts[0]]]
    for ln in lines[starts[0] + 1:]:
        if not ln.strip() or not ln[:1].isspace():
            break
        region.append(ln)
    return " ".join(" ".join(region).split())


def test_claude_md_venv_rule_names_an_absolute_interpreter():
    """Clause 1: the interpreter named in the first rule a session reads has to
    resolve from a round worktree too, where there is no `.venvs/` to resolve a
    relative path against."""
    region = _venv_rule_region()
    m = re.search(r"\*\*Venv\*\*:\s*`([^`]+)`", region)
    assert m, f"the '**Venv**:' entry names no backticked path: {region!r}"
    path = m.group(1)
    assert path.startswith(LIVE_ROOT_PREFIX), (
        f"'**Venv**:' names {path!r}, which only resolves with the live "
        f"checkout as cwd — a round worktree needs {LIVE_ROOT_PREFIX}…")
    assert path.endswith("/.venvs/lloyd/bin/python"), (
        f"{path!r} is absolute but is not the lloyd venv's interpreter")
    assert "every lloyd script" in region.lower(), (
        "the entry dropped the standing instruction that makes the absolute "
        "form matter, reducing the rule to a path with no use stated")


def test_claude_md_routes_a_round_worktree_command_through_the_run_spec_venv():
    """Clause 2: `venv_python` and `run_spec.yaml` named in one statement about
    the commands a round runs from its worktree, with the reason attached, so
    this file and `skills/automod-change-own-code/SKILL.md` ("Run every verify
    or acceptance command with the absolute `venv_python` the round returned")
    stop disagreeing.
    """
    region = _venv_rule_region()
    assert "venv_python" in region.lower(), (
        "the Venv rule still gives one interpreter for every cwd; a round "
        "worktree has no `.venvs/`, so the relative form dies there with exit "
        f"127: {region!r}")
    statement = next((s for s in STATEMENT_BREAK_RE.split(region)
                      if "venv_python" in s.lower()), "")
    assert statement, "`venv_python` is named in a fragment with no sentence end"
    low = statement.lower()
    assert "run_spec.yaml" in low, (
        "the statement names `venv_python` but not the round's `run_spec.yaml`, "
        "which is where a session that never saw the `automod_start` response "
        "reads the path from")
    assert "verify" in low or "acceptance" in low, (
        "the statement does not say which commands it routes — the "
        "verify/acceptance run is the one whose 127 reads as a failed check "
        "rather than a missing interpreter")
    assert "worktree" in low, (
        "the statement does not say the hazard is a round worktree, so it "
        "reads as a general style preference")
    assert "no `.venvs/`" in low or "gitignore" in low, (
        "the statement gives no reason, which is the half that survives a "
        "rewording: a worktree has no `.venvs/` because .gitignore ignores it, "
        "while the live checkout does")


def test_only_the_restart_commands_keep_the_relative_venv_path():
    """Clause 3: the three live-checkout restart commands at the Restart section
    are the only surviving relative uses, measured with the lookbehind form —
    `grep -c '\\.venvs/lloyd/bin/python'` is 4 before and after this change,
    because the absolute path ends in that substring, so the plain count is not
    a measurement of the relative rule at all.
    """
    lines = (ROOT / "CLAUDE.md").read_text(encoding="utf-8").splitlines()
    relative = [(i + 1, ln) for i, ln in enumerate(lines)
                if RELATIVE_VENV_RE.search(ln)]
    assert len(relative) == 3, (
        f"expected 3 lines carrying the relative interpreter, found "
        f"{[n for n, _ in relative]}: a fourth means a rule still points every "
        "script at a path that cannot resolve from a round worktree")
    restarts = [(i + 1, ln) for i, ln in enumerate(lines)
                if "scripts.automod.round restart" in ln]
    assert len(restarts) == 3, (
        f"CLAUDE.md states {len(restarts)} `scripts.automod.round restart` "
        "commands, not the three the clause counts")
    assert [n for n, _ in relative] == [n for n, _ in restarts], (
        "the surviving relative paths are not the restart commands — those run "
        "from the live checkout, where the relative form works; anywhere else "
        "it must be absolute")


# ── #1982: SETUP.md's procedure for the untracked Qwen3-TTS clone ──
#
# `scripts/automod/spec.py` says applying `qwen3-tts-local.patch` to the live
# clone "stays a human action (SETUP.md)". These pin that SETUP.md really
# carries that action, for the clone as it is found (dirty), and that the
# re-sync rule is spelled one way with a reason that measures true.

_TTS_RESYNC = "git -C qwen3-tts diff -- api config.yaml > qwen3-tts-local.patch"
_TTS_PATCHED_FILES = ("api/backends/optimized_backend.py",
                      "api/routers/openai_compatible.py", "config.yaml")


def _setup_text() -> str:
    return (ROOT / "SETUP.md").read_text(encoding="utf-8")


def _tts_upgrade_section() -> str:
    text = _setup_text()
    m = re.search(r"^### Upgrade the vendored clone in place\n(.*?)(?=^#{2,3} |\Z)",
                  text, re.M | re.S)
    assert m, "SETUP.md lost its 'Upgrade the vendored clone in place' section"
    return m.group(1)


def test_setup_gives_the_in_place_upgrade_sequence_for_a_dirty_clone():
    section = _tts_upgrade_section()
    steps = [
        "git -C qwen3-tts diff -- api config.yaml > /tmp/qwen3-tts-local-before.diff",
        "comm -23 <(changed /tmp/qwen3-tts-local-before.diff) <(changed qwen3-tts-local.patch)",
        "git -C qwen3-tts checkout -- " + " ".join(_TTS_PATCHED_FILES),
        "git -C qwen3-tts apply -p1 ../qwen3-tts-local.patch",
        "supervisorctl -c ~/lloyd/agent-services/supervisor/supervisord.conf restart agent-tts",
    ]
    at = [section.find(step) for step in steps]
    assert all(i >= 0 for i in at), [s for s, i in zip(steps, at) if i < 0]
    assert at == sorted(at), "the steps are out of order: save, compare, checkout, apply, restart"
    flat = " ".join(section.split())
    assert "works only on a freshly cloned pristine tree" in flat
    assert "dirty live clone" in flat
    assert "`index` / `diff --git` lines" in flat


def test_the_files_the_upgrade_checks_out_are_the_files_the_patch_touches():
    """The checkout list is a copy of a fact in the patch; a fourth patched file left off
    it would make step 3's `git apply` fail on a clone that still carries the old hunk."""
    patch = (ROOT / "agent-services" / "services" / "tts"
             / "qwen3-tts-local.patch").read_text(encoding="utf-8")
    touched = sorted(set(re.findall(r"^\+\+\+ \w/(\S+)", patch, re.M)))
    assert touched == sorted(_TTS_PATCHED_FILES)
    assert all(f.startswith("api/") or f == "config.yaml" for f in touched), (
        "the re-sync pathspec `-- api config.yaml` no longer covers the patch")


def test_every_patch_regeneration_in_setup_is_path_scoped():
    text = _setup_text()
    assert "git -C qwen3-tts diff >" not in text
    flat = " ".join(text.split())
    assert flat.count(_TTS_RESYNC) == 2, (
        "the re-sync command appears in Part 8 and again in Troubleshooting; both must "
        "be the path-scoped spelling")
    every = re.findall(r"git -C qwen3-tts diff[^\n|]*> *qwen3-tts-local\.patch", text)
    assert every and all(cmd == _TTS_RESYNC for cmd in every), every


def test_the_resync_prose_names_the_real_hazard_and_not_the_one_that_measures_false():
    text = _setup_text()
    flat = " ".join(text.split())
    assert ("regenerating from a clone that has not yet had the current patch applied "
            "overwrites the only gate-visible copy") in flat
    assert "tests/test_qwen3_tts_frame_cap.py" in flat
    # Measured 2026-10-01: `__pycache__/` is gitignored inside the clone and no `.pyc` is
    # tracked, so an unscoped diff is byte-identical to the scoped one. A sentence saying
    # it emits binary diffs or fails `--check` would be a reason that is not true.
    assert "Binary files" not in text
    assert ".pyc" not in text[text.index("## Part 8"):text.index("## Part 9")]


def test_the_upgrade_section_names_the_patch_that_must_not_be_applied():
    flat = " ".join(_tts_upgrade_section().split())
    assert "Do not apply `~/obsidian/backlog/data/qwen3-tts-local-1878-framecap.patch`" in flat
    for part in ("`streaming_opts` as a required argument", "`NameError`",
                 "`except` swallows it", "comes up uncompiled"):
        assert part in flat, part


# ── #2009: the two places that cite the Venv rule cite it by anchor ──────────

#: The docstrings that justify `live_venv_python` by pointing at the Venv rule.
VENV_RULE_CITERS = ("scripts/automod/round.py", "tests/test_automod_worktree_verify.py")
CLAUDE_LINE_CITATION_RE = re.compile(r"CLAUDE\.md:(\d+)")
VAULT_FILE_COUNT_RE = re.compile(r"\d+\s+files\s+under\s+`?~/obsidian")


def _venv_citer_docstring(rel: str) -> str:
    import ast
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    if rel.startswith("tests/"):
        return ast.get_docstring(tree) or ""
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "live_venv_python")
    return ast.get_docstring(fn) or ""


@pytest.mark.parametrize("rel", VENV_RULE_CITERS)
def test_the_venv_rule_is_cited_by_anchor_with_no_line_and_no_corpus_count(rel):
    """Both docstrings read "`CLAUDE.md:12`, 247 files under `~/obsidian`": line 12
    held an unrelated sentence, and the count measured 732 on 2026-10-01. The
    mechanism rests on `.venvs/` being gitignored, on neither figure."""
    text = (ROOT / rel).read_text(encoding="utf-8")
    assert "CLAUDE.md:" + "12" not in text and "247 " + "files" not in text
    assert not CLAUDE_LINE_CITATION_RE.search(text), "a line number in CLAUDE.md rots"
    assert not VAULT_FILE_COUNT_RE.search(text), "a vault corpus count moves daily"
    doc = " ".join(_venv_citer_docstring(rel).split())
    assert "**Venv**:" in doc and "Project Overview" in doc
    # The anchor the docstring names resolves, through the same helper the Venv
    # section above uses: one `**Venv**:` bullet inside `## Project Overview`.
    assert "**Venv**:" in _venv_rule_region()


def _stale_claude_line_citations(text: str, claude_lines: list[str]) -> list[int]:
    """Every `CLAUDE.md:<n>` in `text` whose line <n> does not hold the Venv rule."""
    return [int(n) for n in CLAUDE_LINE_CITATION_RE.findall(text)
            if not (0 < int(n) <= len(claude_lines) and "**Venv**:" in claude_lines[int(n) - 1])]


def test_a_line_citation_of_claude_md_must_land_on_the_venv_rule():
    """The forward guard: should either file cite a CLAUDE.md line again, that
    line has to be the Venv rule. Shown to bite on the text this replaced."""
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8").splitlines()
    for rel in VENV_RULE_CITERS:
        assert _stale_claude_line_citations((ROOT / rel).read_text(encoding="utf-8"), claude) == [], rel
    was = "(`.venvs/lloyd/bin/python`, `CLAUDE.md:" + "12`, 247 files under `~/obsidian`)"
    assert _stale_claude_line_citations(was, claude) == [12], "line 12 is not the Venv rule"
    venv_line = next(i for i, ln in enumerate(claude, 1) if "**Venv**:" in ln)
    assert _stale_claude_line_citations(f"CLAUDE.md:{venv_line}", claude) == []
    assert VAULT_FILE_COUNT_RE.search(was)


# ── #2140: SETUP.md's ### bun section describes the prefix that is really there ──
#
# "Installs to `~/.bun`. Only `@tobilu/qmd` lives here." stood in Part 2 for a
# fortnight after the 2026-09-19 uninstall removed that package, and it
# contradicted Part 6 of the same file, which forbids the published install and
# gives the removal command. Measured 2026-10-03: `~/.bun/bin` holds `bun` and
# the `bunx` symlink to it and nothing else; `~/.bun/install/global/node_modules/@tobilu/`
# is an EMPTY directory — the husk of the uninstall, which a glob over `@tobilu/`
# reads as an install — and the global `package.json` names no dependency at all.
# The bytes behind that last claim are committed at
# `~/obsidian/backlog/data/package.json` (vault `3488566`, `cmp`-clean), because
# a claim about a file outside this repo has no history and no reviewer otherwise.

BUN_INSTALL_CMD = "curl -fsSL https://bun.sh/install | bash"
NPM_PREFIX_HEADING = "### npm global prefix"
BUN_PATH_EXPORT = 'export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$HOME/.bun/bin:$PATH"'
#: The acceptance reads the three keep-anchors out of ONE window — `sed -n
#: '168,190p'`, 23 lines, as filed. Stated as a span from the `### bun` heading
#: it is the same measurement with no absolute line number left to rot: 19 on
#: 2026-10-03 before the rewrite, 21 after it.
BUN_KEEP_ANCHOR_SPAN = 23

#: A predicate that puts a qmd build inside the bun prefix. `~/.bun` need not be
#: spelled for the sentence to mislead — "lives here", inside the bun section, is
#: the same claim, and it is the spelling this rot actually used.
QMD_IN_BUN_CLAIM_RE = re.compile(
    r"\bqmd\b.{0,60}?\b(lives?|installed|install|present|ships|lived)\b", re.I | re.S)
#: A sentence carrying one of these is reporting the prefix's real state, not
#: asserting an install: "No qmd build lives here" is the truth this section now
#: tells, and a detector that flags it would pin prose nobody can write.
QMD_CLAIM_NEGATION_RE = re.compile(r"\b(no|not|never|without|uninstalled|no longer)\b", re.I)


def _setup_bun_region() -> tuple[str, str, str]:
    """(the `### bun` heading, its body, the heading that follows the section)."""
    text = _setup_text()
    m = re.search(r"(?ms)^(### bun[^\n]*)\n(.*?)(?=^#{2,3} )", text)
    assert m, "SETUP.md has no `### bun` section in Part 2's package roots"
    nxt = re.match(r"#{2,3} [^\n]*", text[m.end():])
    assert nxt, "nothing follows the `### bun` section"
    return m.group(1), m.group(2), nxt.group(0).rstrip()


def _sentences(text: str) -> list[str]:
    """Sentences, with the doc's ~80-column wraps collapsed away first: a
    wrapped sentence and an absent sentence are different findings."""
    return [s for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if s]


def _qmd_in_bun_prefix_claims(text: str) -> list[str]:
    """Sentences asserting a qmd build is installed in, or lives in, the bun prefix."""
    return [s for s in _sentences(text)
            if "qmd" in s.lower()
            and QMD_IN_BUN_CLAIM_RE.search(s)
            and not QMD_CLAIM_NEGATION_RE.search(s)]


def test_setup_bun_section_claims_no_qmd_build_lives_in_the_bun_prefix():
    """#2140 clause 1: neither the heading nor any sentence puts a qmd build in
    `~/.bun`. The heading counts — `### bun (qmd only)` made the false claim in
    the one position a skimmer is guaranteed to read."""
    heading, body, _ = _setup_bun_region()
    assert "qmd" not in heading.lower(), f"the heading locates qmd again: {heading!r}"
    assert _qmd_in_bun_prefix_claims(body) == [], (
        "the bun section is claiming a qmd build is installed there again")
    # The detector bites: the sentence this clause exists to keep out, verbatim
    # out of SETUP.md at eb6645ed, is caught rather than merely absent. A pin
    # that only greps for that one string would survive a reworded reinstatement.
    assert _qmd_in_bun_prefix_claims(
        "Installs to `~/.bun`. Only `@tobilu/qmd` lives here.") == [
        "Only `@tobilu/qmd` lives here."]


def test_setup_bun_section_states_the_measured_state_of_the_bun_prefix():
    """#2140 clause 2: the four measured facts are IN the section — where the
    prefix is, what its bin holds, that nothing qmd is in it, and which qmd the
    PATH actually resolves to."""
    _, body, _ = _setup_bun_region()
    flat = " ".join(body.split())
    assert "Installs to `~/.bun`" in flat
    assert re.search(r"`~/\.bun/bin` holds only `bun` and[^.]*`bunx`", flat), flat
    assert re.search(r"[Nn]o qmd build (?:lives|is installed)", flat), (
        "the section no longer says outright that nothing qmd is in the prefix")
    assert "`~/.local/bin/qmd` → `~/lloyd/qmd/bin/qmd`" in flat, (
        "the fork is the qmd on PATH, so the section has to say how it is reached")


def test_setup_bun_section_agrees_with_part_6_rather_than_contradicting_it():
    """#2140 clause 3, pinned at both ends. Part 6 forbids the published install
    and carries the removal command; a bun section that quietly re-asserted the
    install would leave the file arguing with itself, and a section pointing at
    a Part 6 that had since dropped the prohibition would point at nothing."""
    text = _setup_text()
    _, body, _ = _setup_bun_region()
    flat = " ".join(body.split())
    assert "must not be installed" in flat, (
        "the bun section no longer states that the published build stays uninstalled")
    assert "Part 6" in flat, "and it no longer points at the prose that says why"
    anchor = "## Part 6 — qmd (vault search)"
    assert anchor in text, "the section this one points at is gone: the pointer dangles"
    part6 = text[text.index(anchor):]
    assert "must not be installed" in " ".join(part6.split()), (
        "Part 6 no longer forbids the published install")
    assert "bun remove -g @tobilu/qmd" in part6, "Part 6 lost the removal command"


def test_setup_bun_section_keeps_the_install_block_and_the_npm_prefix_section():
    """#2140 clause 4: the rewrite took one claim out and nothing else. The curl
    command is still the section's own, `### npm global prefix` is still the
    heading that follows, and the `~/.bun/bin` entry is still in the PATH export
    — all three within the single window the acceptance reads them from."""
    text = _setup_text()
    heading, body, following = _setup_bun_region()
    assert BUN_INSTALL_CMD in body, "the bun install command left the section"
    assert following == NPM_PREFIX_HEADING, f"{following!r} is in its place instead"
    lines = text.splitlines()
    head_n = next((i + 1 for i, ln in enumerate(lines) if ln == heading), 0)
    path_n = next((i + 1 for i, ln in enumerate(lines) if ln.strip() == BUN_PATH_EXPORT), 0)
    assert head_n and path_n, f"`### bun` at {head_n}, the PATH export at {path_n}"
    assert BUN_INSTALL_CMD in "\n".join(lines[head_n - 1:path_n]), (
        "an anchor fell outside the span between the section and the PATH export")
    assert path_n - head_n <= BUN_KEEP_ANCHOR_SPAN, (
        f"the keep-anchors now span {path_n - head_n} lines from `### bun`; the "
        f"acceptance reads all three out of one {BUN_KEEP_ANCHOR_SPAN}-line window")


def test_the_committed_witness_backs_the_bun_prefix_claim():
    """#2140 clause 5: the claim is about a file outside this repo, so its bytes
    are committed and re-read here. `wc -l < backlog/data/package.json` → 4 is
    the re-derivation, and the file names no dependency at all — which is what
    lets the section say no qmd build is installed under the bun prefix. Read
    through `board_presence`, unmarked like the other board-reading claims in
    this tree: a `live_vault`-marked node is deselected by the gate's own
    `-m "not live_vault"` and would certify nothing."""
    import board_presence
    import json

    witness = board_presence.BOARD_DIR / "data" / "package.json"
    assert witness.is_file(), (
        f"{witness} is the only history behind SETUP.md's claim that the published "
        "build is not installed under ~/.bun; a claim about an unrepo file with no "
        "committed bytes is a sentence nobody can re-check")
    doc = json.loads(witness.read_text(encoding="utf-8"))
    assert "dependencies" not in doc, f"something is globally installed again: {doc}"
    assert doc["trustedDependencies"] == ["node-llama-cpp"], doc
    _, body, _ = _setup_bun_region()
    assert re.search(r"[Nn]o qmd build (?:lives|is installed)", " ".join(body.split())), (
        "the witness is on disk but the section stopped claiming what it proves")
