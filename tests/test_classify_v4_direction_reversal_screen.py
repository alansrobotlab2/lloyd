"""#1256 — a `confirmed` direction check whose own reason argues the reverse.

Task #74's run at 2026-09-19T08Z wrote 4,020 records; 2,052 of them carry a
direction check with `verdict: "confirmed"`, and 8 of those have a `reason`
that argues the opposite direction. The check for
`Granite Models -[uses]-> OpenRAG` says: "Granite Models are used within the
OpenRAG stack, implying OpenRAG uses them, not vice versa." That sentence
proves `OpenRAG uses Granite Models` — the reverse of the edge the record
went on to write at confidence 0.95, far above the applier's floor of 0.6
(`apply-classifications-v4.py` DEFAULT_MIN_CONF).

The cause was in `classify_edge_v4`
(scripts/memory/classify-relationships-v4.py): the direction-check block
read the reason text only for the `reversed` and `unclear` verdicts, so a
`confirmed` verdict was accepted on its label alone.

The screen adds no model and no new softening path — a cue-bearing
`confirmed` reason takes the same `downgraded_reversed` route an explicit
`reversed` verdict already takes (type `mentions`, confidence ≤ 0.5), which
is the one route the applier never upgrades (`new_type == "mentions"` is
skipped). The cue is matched against `direction_check.reason` only: over the
2,052 confirmed records of that run, `rather than` and `instead` occur in
primary reasons of correctly-directed edges, so screening those texts would
downgrade good edges.

Every test here stubs the module-level `_call_llm_v4`, so no request leaves
the process.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "memory" / "classify-relationships-v4.py"

# The four reversal cues clause 1 names. Each is a phrase that states the
# relation runs B → A while the verdict says A → B.
REVERSAL_CUES = ("not vice versa", "in reverse", "the opposite",
                 "not the other way")

ENDPOINT = "http://stub.invalid/v1/chat/completions"
MODEL = "stub"


@pytest.fixture
def mod(tmp_path, monkeypatch):
    """Import the hyphenated v4 classifier by path, with a throwaway kg store.

    The store is configured only because importing the module resolves store
    and alias paths; the alias table stays empty, so nothing here depends on
    the live graph.
    """
    monkeypatch.setenv("LLOYD_KG_DB", str(tmp_path / "kg.sqlite"))
    from app import kg_store
    kg_store.configure(tmp_path / "kg.sqlite")
    name = "classify_v4_reversal_screen_under_test"
    spec = importlib.util.spec_from_file_location(name, str(SCRIPT))
    m = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, m)
    spec.loader.exec_module(m)
    try:
        yield m
    finally:
        kg_store.reset()


def _classify(monkeypatch, mod, *, source, target, context, relation,
              confidence, reason, direction_check):
    """Run `classify_edge_v4` against two scripted LLM responses.

    The first response is the classifier's, the second the direction check's.
    A third call would mean the pipeline asked for a verdict that was already
    on the table, so the stub raises instead of calling out.
    """
    scripted = [{"type": relation, "confidence": confidence, "reason": reason},
                dict(direction_check)]

    def fake(endpoint, model, system, prompt, timeout):
        if not scripted:
            raise AssertionError("_call_llm_v4 called past the two scripted responses")
        assert endpoint == ENDPOINT and model == MODEL
        return scripted.pop(0)

    monkeypatch.setattr(mod, "_call_llm_v4", fake)
    return mod.classify_edge_v4(source, target, context, ENDPOINT, MODEL, 5)


# ---------------------------------------------------------------------------
# Clause 1 — a cue-bearing `confirmed` verdict is not accepted as-is
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cue", REVERSAL_CUES)
def test_confirmed_verdict_with_reversal_cue_downgrades_to_mentions(
        monkeypatch, mod, cue):
    """`confirmed` + a reason saying the relation runs the other way."""
    ctx = "The Acme Gateway forwards requests to the Acme Token Vault for auth."
    out = _classify(
        monkeypatch, mod,
        source="Acme Gateway", target="Acme Token Vault", context=ctx,
        relation="uses", confidence=0.95,
        reason="The gateway `forwards requests` to the vault for auth.",
        direction_check={"verdict": "confirmed",
                         "reason": f"The vault is the actor here ({cue})."},
    )
    assert out["direction_check"]["verdict"] == "confirmed"
    assert out["verdict_adjustment"] == "downgraded_reversed"
    assert out["type"] == "mentions"
    assert out["confidence"] <= 0.5
    assert "[dir-check]" in out["reason"]


def test_confirmed_verdict_without_any_cue_is_accepted(monkeypatch, mod):
    """Control: the screen fires on the cue text, not on `confirmed` itself."""
    ctx = "The Acme Gateway forwards requests to the Acme Token Vault for auth."
    out = _classify(
        monkeypatch, mod,
        source="Acme Gateway", target="Acme Token Vault", context=ctx,
        relation="uses", confidence=0.95,
        reason="The gateway `forwards requests` to the vault for auth.",
        direction_check={"verdict": "confirmed",
                         "reason": "The gateway actively invokes the vault."},
    )
    assert out["verdict_adjustment"] == "none"
    assert out["type"] == "uses"
    assert out["confidence"] == 0.95


def test_explicit_reversed_verdict_still_downgrades(monkeypatch, mod):
    """The pre-existing `reversed` path is untouched by the new trigger."""
    ctx = "The Acme Gateway forwards requests to the Acme Token Vault for auth."
    out = _classify(
        monkeypatch, mod,
        source="Acme Gateway", target="Acme Token Vault", context=ctx,
        relation="uses", confidence=0.95,
        reason="The gateway `forwards requests` to the vault for auth.",
        direction_check={"verdict": "reversed",
                         "reason": "The vault is the actor, not the gateway."},
    )
    assert out["verdict_adjustment"] == "downgraded_reversed"
    assert out["type"] == "mentions"
    assert out["confidence"] <= 0.5


# ---------------------------------------------------------------------------
# Clause 2 — the 8 rationales captured from the 2026-09-19T08Z run
# ---------------------------------------------------------------------------

# source, target, classifier relation, classifier confidence, classifier
# reason, direction_check reason — each copied verbatim from
# `_pipeline/memory-graph/classified-v4-batch.jsonl` for that run. The
# context is rebuilt to contain the captured `reason_quote`, because only
# its hash is stored: it exists so the hallucination gate passes and the
# confidence that reaches the assertions is the one the run had.
CAPTURED = [
    ("Granite Models", "OpenRAG", "uses", 0.95,
     "Granite Models are `used for generation and document parsing in the "
     "OpenRAG stack`.",
     "Granite Models are used within the OpenRAG stack, implying OpenRAG "
     "uses them, not vice versa.",
     "used for generation and document parsing in the OpenRAG stack"),
    ("Axios", "OpenClaw", "depends_on", 0.95,
     "Axios is cited as an `unpinned dependency` in third-party integrations "
     "causing supply chain exposure for OpenClaw.",
     "OpenClaw uses Axios as a dependency, so OpenClaw depends on Axios, not "
     "vice versa.",
     "unpinned dependency"),
    ("Cron Jobs", "AgentCraft", "uses", 0.95,
     "Cron Jobs are `leveraged in AgentCraft` for scheduled autonomous "
     "workflows.",
     "AgentCraft leverages Cron Jobs, meaning AgentCraft uses Cron Jobs, not "
     "vice versa.",
     "leveraged in AgentCraft"),
    ("Cross-App Access (XAA)", "AI Agents", "uses", 0.95,
     "XAA `enables seamless, consent-free connections for AI agents` by "
     "leveraging Identity Provider.",
     "XAA enables connections for AI agents, implying AI agents use XAA, not "
     "vice versa.",
     "enables seamless, consent-free connections for AI agents"),
    ("The Wiki", "Markdown Files", "part_of", 0.95,
     "The Wiki `consists of LLM-generated markdown files`, indicating the "
     "files are components inside the system.",
     "The Wiki consists of markdown files, so the Wiki is composed of them, "
     "not vice versa.",
     "consists of LLM-generated markdown files"),
    ("Tu Vu", "WIKISKILL Paper", "created_by", 0.55,
     "Tu Vu is an author of the WikiSkill paper, meaning the paper was "
     "created by Tu Vu.",
     "Tu Vu is the author, so the paper was created by Tu Vu, not vice versa.",
     None),
    ("Graphify", "AI Agents", "uses", 0.9,
     "Graphify is a tool for AI agents that `addresses context economy`",
     "Graphify is a tool used by AI agents, so agents use Graphify, not vice "
     "versa.",
     "addresses context economy"),
    ("QMD Daemon", "Backlog Item", "discusses", 0.95,
     "The `backlog item` regarding GPU OOM monitoring for the QMD Daemon was "
     "determined to be stale.",
     "The backlog item discusses the QMD Daemon, so the topic is the daemon, "
     "not vice versa.",
     "backlog item"),
]

def _captured_params():
    return [pytest.param(*row, id=f"{row[0]}->{row[1]}") for row in CAPTURED]


@pytest.mark.parametrize(
    "source,target,relation,confidence,reason,dc_reason,quote",
    _captured_params())
def test_captured_reversal_rationale_downgrades_without_an_llm_call(
        monkeypatch, mod, source, target, relation, confidence, reason,
        dc_reason, quote):
    """Each of the 8 records the screen would have caught, replayed offline.

    Without the screen the record keeps the relation the classifier proposed
    and the confidence it carried — which is why the assertions below are
    `mentions` and ≤ 0.5 rather than a comparison against the old output.
    """
    ctx = f"Context line containing the quoted phrase: {quote}." if quote else (
        "Tu Vu is listed among the authors of the WikiSkill paper.")
    out = _classify(
        monkeypatch, mod,
        source=source, target=target, context=ctx,
        relation=relation, confidence=confidence, reason=reason,
        direction_check={"verdict": "confirmed", "reason": dc_reason},
    )
    assert "not vice versa" in dc_reason  # every captured row used this cue
    assert out["direction_check"] == {"verdict": "confirmed",
                                      "reason": dc_reason[:200]}
    assert out["verdict_adjustment"] == "downgraded_reversed"
    assert out["type"] == "mentions"
    assert out["confidence"] <= 0.5


# ---------------------------------------------------------------------------
# Clause 3 — the screen reads only the direction check's own reason
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cue_word", ["rather than", "instead"])
def test_primary_reason_cue_word_does_not_downgrade(
        monkeypatch, mod, cue_word):
    """A cue word in the *primary* reason, with a clean direction check, is
    not evidence of a reversal: in the 2026-09-19 run these words appear in
    the primary reasons of correctly-directed edges."""
    ctx = "The Acme Gateway forwards requests to the Acme Token Vault for auth."
    reason = (f"The gateway actively invokes the vault {cue_word} treating it "
              f"as a peer; it `forwards requests` for auth.")
    out = _classify(
        monkeypatch, mod,
        source="Acme Gateway", target="Acme Token Vault", context=ctx,
        relation="uses", confidence=0.95, reason=reason,
        direction_check={"verdict": "confirmed",
                         "reason": "The gateway actively invokes the vault."},
    )
    assert cue_word in out["reason"]
    assert out["verdict_adjustment"] == "none"
    assert out["type"] == "uses"
    assert out["confidence"] == 0.95


def test_a_cue_in_the_primary_reason_alone_never_reaches_the_screen(
        monkeypatch, mod):
    """Same edge, this time with the reversal cue in the primary reason and a
    clean direction check: the screen must not fire on it."""
    ctx = "The Acme Gateway forwards requests to the Acme Token Vault for auth."
    out = _classify(
        monkeypatch, mod,
        source="Acme Gateway", target="Acme Token Vault", context=ctx,
        relation="uses", confidence=0.95,
        reason="The vault is the actor, not vice versa, though it merely "
               "`forwards requests` for auth.",
        direction_check={"verdict": "confirmed",
                         "reason": "The gateway actively invokes the vault."},
    )
    assert "not vice versa" in out["reason"]
    assert out["verdict_adjustment"] == "none"
    assert out["type"] == "uses"
    assert out["confidence"] == 0.95


# ---------------------------------------------------------------------------
# Seam — classifier output → classified-v4 JSONL → the applier
# ---------------------------------------------------------------------------
#
# `classify_edge_v4` and `apply-classifications-v4.py` are separate processes:
# the classifier is a function in this one, the applier is a script invoked
# later, and the only thing between them is the record written to
# `_pipeline/memory-graph/classified-v4*.jsonl`. The defence clause 1 rests on
# is on the far side of that boundary — `apply-classifications-v4.py` skips a
# record whose `new_type` is `mentions` (`still_mentions`, before the
# confidence floor and before any edge lookup), so an inverted verb softened
# to `mentions` cannot retype a live edge. Testing only the returned dict
# would leave that claim untested, so this pins the whole path: the captured
# Granite Models → OpenRAG rationale through the real classifier, persisted
# exactly as the batch writer persists it, then applied by a real
# `--apply` run of the applier against a throwaway store seeded with the
# `mentions` edge that was live when the run classified it.
#
# The control pair proves the applier was actually run and actually able to
# upgrade: the same store holds a second `mentions` edge whose record cleared
# the direction check cleanly, and that one does get retyped to `uses`.

APPLIER = ROOT / "scripts" / "memory" / "apply-classifications-v4.py"

# Second pair, used only as the control: a clean `confirmed` verdict, so the
# applier is free to upgrade it.
CONTROL_SRC, CONTROL_TGT = "Acme Gateway", "Acme Token Vault"
CONTROL_CTX = ("The Acme Gateway forwards requests to the Acme Token Vault "
               "for auth.")


def _record_from(out, source, target):
    """The JSONL row the batch writer persists for one classified edge.

    Mirrors how scripts/memory/classify-relationships-v4.py emits it: the
    classifier's `type` becomes `new_type`, and the live edge it was
    classified from is recorded as `original_type`/`original_provenance`
    `mentions`/`EXTRACTED`, which is the provenance the applier accepts.
    """
    return {
        "source": source,
        "target": target,
        "resolved_src": out["resolved_src"],
        "resolved_tgt": out["resolved_tgt"],
        "original_type": "mentions",
        "original_provenance": "EXTRACTED",
        "new_type": out["type"],
        "confidence": out["confidence"],
        "reason": out["reason"],
        "reason_quote": out["reason_quote"],
        "quote_verified": out["quote_verified"],
        "direction_check": out["direction_check"],
        "verdict_adjustment": out["verdict_adjustment"],
        "model": MODEL,
        "prompt_version": "v4",
    }


def test_screened_record_cannot_upgrade_a_live_edge_end_to_end(
        monkeypatch, mod, tmp_path):
    """Granite Models → OpenRAG, replayed through classifier and applier."""
    import json
    import subprocess

    src, tgt, quote = ("Granite Models", "OpenRAG",
                       "used for generation and document parsing in the "
                       "OpenRAG stack")
    ctx = f"Context line containing the quoted phrase: {quote}."
    screened = _classify(
        monkeypatch, mod,
        source=src, target=tgt, context=ctx, relation="uses", confidence=0.95,
        reason="Granite Models are `used for generation and document parsing "
               "in the OpenRAG stack`.",
        direction_check={
            "verdict": "confirmed",
            "reason": "Granite Models are used within the OpenRAG stack, "
                      "implying OpenRAG uses them, not vice versa."},
    )
    control = _classify(
        monkeypatch, mod,
        source=CONTROL_SRC, target=CONTROL_TGT, context=CONTROL_CTX,
        relation="uses", confidence=0.95,
        reason="The gateway `forwards requests` to the vault for auth.",
        direction_check={"verdict": "confirmed",
                         "reason": "The gateway actively invokes the vault."},
    )
    assert screened["verdict_adjustment"] == "downgraded_reversed"
    assert control["verdict_adjustment"] == "none"

    # Seed the store with the two live `mentions` edges the run read.
    db = tmp_path / "apply-kg.sqlite"
    from app.kg_store import KGStore
    st = KGStore(db)
    for s, t in ((src, tgt), (CONTROL_SRC, CONTROL_TGT)):
        st.edges.add({"source": s, "target": t, "type": "mentions",
                      "provenance": "EXTRACTED", "confidence": 0.3},
                     origin="extractor")
    st.close()

    classified = tmp_path / "classified"
    classified.mkdir()
    with (classified / "classified-v4-batch.jsonl").open("w") as fh:
        for out, s, t in ((screened, src, tgt),
                          (control, CONTROL_SRC, CONTROL_TGT)):
            fh.write(json.dumps(_record_from(out, s, t)) + "\n")

    proc = subprocess.run(
        [sys.executable, str(APPLIER),
         "--classified-dir", str(classified),
         "--pattern", "classified-v4*.jsonl",
         "--db", str(db), "--apply"],
        capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    after = KGStore(db)
    live = {(e["source"], e["target"]): e["type"] for e in after.edges.active()}
    # The inverted verb never reached the graph…
    assert live.get((src, tgt)) == "mentions", (
        "a cue-bearing `confirmed` verdict retyped the live edge — the "
        "screen did not route it to `mentions`, or the applier wrote it anyway")
    # …while the clean verdict did, so the assertion above is not passing on
    # an applier that upgraded nothing at all.
    assert live.get((CONTROL_SRC, CONTROL_TGT)) == "uses", (
        "the control edge was not upgraded, so the `mentions` assertion "
        "proves nothing about the screen: the applier run did not work")
    after.close()
