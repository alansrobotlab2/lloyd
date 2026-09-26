#!/usr/bin/env python3
"""Counterfactual-perturbation rater for the retrieval eval (#541).

One perturbed variant per nightly query: flip exactly one named constraint, run
the variant through the identical retrieval path, and score two independent
directions.

  moved   the changed constraint moved what the retriever pulled
  pinned  the constraints that did not change pulled the same rows as before

Both directions are the point. A moved-only rater rewards an extractor that
churns everything on every edit — and it cannot see an *incorrectly unchanged*
result, which is the specific signature of a retriever keying on surface
bag-of-words instead of the named thing. Since entity_hit_rate sits near 0.5
while doc_hit_rate sits near 0.95, that is the failure this measures.

Two design constraints, both from the item's own risks:

The records are **committed, never re-derived at eval time**. The sibling pairs
come from the graph alias table (``sibling_source: kg_alias_table``), but they
were drawn once, by hand, from pairs already known to be normalization
artifacts, and then frozen into ``counterfactual_perturbations.yaml``. If the
live table picked the pairs every night the measurement would change underneath
the trend line — the defect the nightly compare step already guards against for
the query set. ``verify_siblings`` is the audit against the live table, run by
hand, not by the nightly path.

The metrics are **entity/fact-level only**. The doc corpus is the same vault in
both arms, so any doc-level comparison reads as spurious success (risk 3). The
retrieved projection is built from fact/graph entity attributions and fact text;
``documents`` is never consulted.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RECORD_PATH = HERE / "counterfactual_perturbations.yaml"

#: The fixed taxonomy. One axis per query, decided by the plan below.
AXES = ("entity", "date", "qualifier", "artifact")

ENTITY_AXIS = "entity"

#: axis_changed -> what the perturbation is allowed to name as a new constraint.
AXES_REQUIRE_MOVE_TARGET = {ENTITY_AXIS}

# ── the axis plan ────────────────────────────────────────────────────────────
# Hand-authored and reviewed, one entry per query id. Sibling values are
# canonicals read out of the graph alias table (surface -> canonical pairs that
# already coexist there, i.e. known normalization artifacts), never invented:
# an unfair sibling manufactures false failures, which is risk 1 in #541.
#
#   expected_to_move   entity names the variant must pull that the original did
#                      not (entity axis only — the other axes name no new row)
#   expected_pinned    constraints the query still names, which must pull the
#                      same rows in both arms. Anything not named here is not
#                      scored, so a query with one constraint contributes to
#                      moved_rate only — a vacuous pin would inflate pinned_rate
#
#   "still names" is a testable phrase, and #763 is what made it one: a pin must
#   be a case-insensitive substring of `apply_perturbation(query, entry)`, the
#   variant's OWN text. A pin the swap deletes is not an unchanged constraint,
#   it is the constraint that changed, and scoring it reports a defect on every
#   nightly that behaves correctly (`autonomy-pipeline` pinned "autonomy" while
#   its swap rewrote "the autonomy pipeline" to "the Data Pipeline", and
#   `nightly-20260917` / `nightly-20260921` each booked it as
#   `pinned_axis_churned / autonomy: present=True->False`). The audit that keeps
#   that from recurring is `pins_absent_from_their_own_swap`, run by `--check`
#   and by the suite; the five entries whose pin is knowingly not in their own
#   variant text are enumerated in `PINS_ABSENT_BY_DESIGN` with the reason.
# PLAN[id] = (axis, old_value, new_value, expected_to_move, expected_pinned)
PLAN: dict[str, tuple[str, str, str, list[str], list[str]]] = {
    # entity swaps — the axis whose failure mode is the direct evidence for or
    # against #537's typed-identity-key premise.
    # #763 option 1: the pin is the surviving half of the swapped term — the
    # artifact TYPE, whose number is what changed. Measured, not guessed:
    # "Backlog Item" was the obvious string and it MANUFACTURES a churn — the
    # original arm attributes `backlogtask363`, which normalized does not
    # contain "backlogitem", so `score_pair` would have read
    # `Backlog Item: present=False->True` on a retriever that behaved (the
    # clause-2 defect, imported rather than removed). "Backlog" is the nearest
    # string that both survives into "tell me about Backlog Item #313" and is
    # attributed by BOTH arms — `backlogtask363` original, `backlogitem313`
    # variant, same attributed text — scored live 2026-09-22 through
    # `run_eval._vault_recall`, pinned=True, moved=True.
    "backlog-363": ("entity", "backlog item 363", "Backlog Item #313",
                    ["Backlog Item #313"], ["Backlog"]),
    # Same shape: "Entity Resolution" is the shared stem of the sweep and its
    # sibling, it survives into "what does the Semantic Entity Resolution skill
    # do?", and the row `entityresolution` is attributed by BOTH arms of
    # nightly-20260919/20/21 — a pin with row evidence on each side, not a
    # substring accident.
    "entity-resolution-sweep": ("entity", "entity resolution sweep",
                               "Semantic Entity Resolution",
                               ["Semantic Entity Resolution"], ["Entity Resolution"]),
    # The residual pair: these two stay unscored and visible in the printed
    # fraction, and #763 leaves them to a person. Each candidate pin was tested
    # and each failed. The two arms' attributed rows are DISJOINT for `inner-voice`
    # in every recent nightly (`inner-voice`/`inner-voice-event-log` against
    # `voice-mode`/`agent-voice-mode`/`port-8096`), so any pin taken from the
    # original arm is absent from the variant arm by construction — which is a
    # manufactured churn, risk 1, not a measurement. For `vault-recall` the only
    # string that scores a pass is "Vault Recall", and it passes by matching a
    # generic `vault` row through `_match`'s bidirectional-substring tolerance
    # rather than the recall row it means. The remaining route is widening each
    # query in `eval/vault_recall_queries.yaml` to name a second constraint, and
    # #541 reserved that ground-truth file against perturbation work.
    "inner-voice": ("entity", "inner voice", "Voice Mode", ["Voice Mode"], []),
    "vault-recall": ("entity", "vault_recall", "Vault Index", ["Vault Index"], []),
    # "QMD" is the shared stem of `QMD` and `QMD Search`, so it survives the
    # swap into "what is QMD Search?", and the row `qmd` is attributed by both
    # arms of nightly-20260919/20/21 — the pin asks that naming one QMD
    # subsystem not drop the others (#763 option 1).
    "qmd": ("entity", "QMD", "QMD Search", ["QMD Search"], ["QMD"]),
    "kg-maintenance-tasks": ("entity", "knowledge graph", "Entity Graph",
                            ["Entity Graph"], ["autonomy"]),
    "lloyd-vllm-rel": ("entity", "vLLM", "TensorRT-LLM",
                       ["TensorRT-LLM"], ["lloyd"]),
    # NOT an alias-table normalization pair — the live table says so, which is why
    # this comment was rewritten on 2026-09-22 (it used to claim both surfaces
    # routed to `Autonomy Data Pipeline`). Through `aliases.all_lower()` today:
    # 'data pipeline' is the surface of its OWN canonical `Data Pipeline`, and
    # 'autonomy pipeline' is a surface of TWO canonicals — `Autonomy Data
    # Pipeline` (row created 2026-09-10) and `Autonomy Pipeline` (created
    # 2026-09-04) — and `all_lower()` keeps the earliest-created row, so it
    # resolves to `Autonomy Pipeline`. So the swap crosses two canonicals, which
    # is what `--verify` certifies (a swapped-in value that resolves, to a
    # canonical different from the old one's), and the seed set staying put is
    # the identity-keying result, not a broken perturbation.
    #
    # #763 clause 2: the pinned leg is dropped, because it was the rater's false
    # alarm. "autonomy" came from `expect_entities`, never from anything the
    # swap leaves alone — the perturbation rewrites "the autonomy pipeline" into
    # "the Data Pipeline", so the variant text contains no "autonomy" to pin, and
    # `nightly-20260917` and `nightly-20260921` each booked the row as
    # `pinned_axis_churned / autonomy: present=True->False`: a defect reported on
    # the one axis that legitimately moved. The moved leg is untouched and still
    # scores this query against `Data Pipeline`, so the #537 evidence the entry
    # exists for is unaffected; only the unsatisfiable pin goes, and the entry
    # moves from `pinned_axis_churned` to `nothing_pinned` in the label column.
    "autonomy-pipeline": ("entity", "autonomy pipeline", "Data Pipeline",
                          ["Data Pipeline"], []),
    # artifact-type changes: "the daily note" -> "the knowledge note" family
    "harness-tools": ("artifact", "tool calls", "tool results", [], ["agent harness"]),
    "nightly-reflection": ("artifact", "skills", "autonomy tasks", [],
                           ["nightly reflection"]),
    "relationships-location": ("artifact", "index", "backup", [], ["relationships"]),
    "robotics-projects": ("artifact", "projects", "research notes", [], ["robotics"]),
    # qualifier negation / replacement
    "qwen38-local-serving": ("qualifier", "24GB", "48GB", [], ["Qwen3.8"]),
    "tgs-rag-state": ("qualifier", "current", "original", [], ["TGS-RAG"]),
    "graph-quality": ("qualifier", "noise", "precision", [], ["graph"]),
    "classifier-v4": ("qualifier", "upgrade", "keep", [], ["mention classifier"]),
    "godnode-threshold": ("qualifier", "why does it exist", "what does it cost",
                          [], ["FACT_GODNODE_THRESHOLD"]),
    "memory-persistence": ("qualifier", "across sessions", "within one session",
                           [], ["Lloyd"]),
    "backlog-overview": ("qualifier", "interesting", "stale", [], ["Backlog"]),
    # date swap
    "this-week-autonomy": ("date", "this week", "last quarter", [], ["autonomy"]),

    # ── added with the #1319 corpus growth (20 -> 87 queries) ───────────────
    # Authored from the query text alone, never from a recall run: each entry names
    # the swap that tests it.
    # an entity that is a real, DISTINCT canonical in the live alias table, so
    # `--verify` can audit the chosen sibling (#537's risk-1 rule applies).
    "tts-voice-cloning": ("entity", "TTS", "Piper TTS", ["Piper TTS"], []),
    "wake-word-models": ("entity", "wake phrases", "OpenWakeWord", ["OpenWakeWord"], []),
    "thunderbird-mcp-toolset": ("entity", "Thunderbird MCP",
                                "Thunderbird Service", ["Thunderbird Service"], []),
    "browser-tool-validation": ("entity", "browser tool",
                                "Browser Extraction", ["Browser Extraction"], []),
    "youtube-transcript-workflow": ("entity", "YouTube transcript",
                                    "Transcript Extraction", ["Transcript Extraction"], []),
    "mcp-transport-error-recovery": ("entity", "MCP server",
                                     "GitHub MCP Server", ["GitHub MCP Server"], []),
    "kg-rebuild-abandoned": ("entity", "knowledge graph", "Entity Graph", ["Entity Graph"], []),
    "kg-dedup-key": ("entity", "knowledge graph",
                     "Lloyd Memory Graph", ["Lloyd Memory Graph"], []),
    "guardian-to-backlog": ("entity", "guardian",
                            "Self-mod guardian", ["Self-mod guardian"], []),
    "dream-to-skill-edit": ("entity", "dream consolidation",
                            "Nightly Skills Management", ["Nightly Skills Management"], []),
    "automod-to-entity-guard": ("entity", "self-modification round",
                                "Lloyd automod", ["Lloyd automod"], []),
    "memory-capture-to-kg": ("entity", "knowledge graph", "Memory Graph", ["Memory Graph"], []),
    "research-queue-to-vault-note": ("entity", "research queue",
                                     "Groundskeeper Research", ["Groundskeeper Research"], []),
    # Sibling is the OTHER instrument that has to make the same stale-vs-real call.
    # It must not be the query's own gold entity (clause 2 re-points that label to
    # `Nightly Retrieval Eval`): a twin that names the gold answer outright measures
    # the perturbation as unseen even when the retriever is right.
    # Canonical in the live alias table, so `--verify` can actually confirm it.
    "counterfactual-to-trend-audit": ("entity", "retrieval eval", "Knowledge Health Report",
                                      ["Knowledge Health Report"], []),
    "autonomy-task-to-skill": ("entity", "autonomy task",
                               "Autonomy Scheduler", ["Autonomy Scheduler"], []),
    "inner-voice-to-surface": ("entity", "inner voice",
                               "Inner Voice Observer", ["Inner Voice Observer"], []),
    "entity-guard-to-alias-table": ("entity", "knowledge graph",
                                    "Memory Graph", ["Memory Graph"], []),
    "email-pipeline-to-daily-note": ("entity", "morning briefing",
                                     "morning-brief", ["morning-brief"], []),
    "guardian-alert-to-retraction": ("entity", "guardian",
                                     "Groundskeeper", ["Groundskeeper"], []),
    "voice-session-to-room": ("entity", "voice", "Voice Mode", ["Voice Mode"], []),
    "memory-md-clobber-to-guard": ("entity", "loaded memory file",
                                   "MEMORY.md", ["MEMORY.md"], []),
    "gpu-daemon-ipc-timeout": ("entity", "GPU daemon",
                               "LiveKit Agents", ["LiveKit Agents"], []),
    "stop-auto-merging-entities": ("entity", "automatically merging",
                                   "Semantic Entity Resolution", ["Semantic Entity Resolution"], []),
    "write-that-ate-the-memory-file": ("entity", "memory file", "MEMORY.md", ["MEMORY.md"], []),
    "nightly-cannot-tell": ("entity", "nightly retrieval run",
                            "Groundskeeper Survey", ["Groundskeeper Survey"], []),
    "facts-that-contradict": ("entity", "memory", "Fact Store", ["Fact Store"], []),
    "skill-that-never-improves": ("entity", "reflection loop",
                                  "Nightly Skill Consolidation", ["Nightly Skill Consolidation"], []),
    "numbers-differ-after-rebuild": ("entity", "graph",
                                     "Lloyd Memory Graph", ["Lloyd Memory Graph"], []),
    "memory-line-contradicts-vault": ("entity", "memory", "MEMORY.md", ["MEMORY.md"], []),
    "ambient-prefetch-ttl-reclaim": ("entity", "background context",
                                     "Ambient Turns", ["Ambient Turns"], []),
    "retrieval-seed-anchoring-contract": ("entity", "seed extractor",
                                          "skills_search", ["skills_search"], []),
    # the swap is one modifier that flips what the answer should say, and names no
    # sibling, so `--verify` owes nothing for them.
    "grafana-monitoring-stack": ("qualifier", "small", "large", [], ["Grafana"]),
    "gpu-model-naming-collision": ("qualifier", "different", "identical", [], ["GPU"]),
    "three-d-printing-calibration": ("qualifier", "flow rate", "bed level", [], ["printer"]),
    "config-yaml-readonly": ("qualifier", "never", "sometimes", [], ["config.yaml"]),
    "model-alias-resolution": ("qualifier", "actually", "originally", [], ["routing name"]),
    "groundskeeper-queue-corruption": ("qualifier", "corrupted", "emptied", [], ["queue"]),
    "watchdog-metacharacter": ("qualifier", "every process on the machine",
                               "every process in the container", [], ["pkill"]),
    "yaml-scalar-block-indent": ("qualifier", "first line", "last line", [], ["YAML Parser"]),
    "json-filter-epoch-vs-iso": ("qualifier", "zero rows", "every row", [], ["JSONL"]),
    "git-signals-in-async-code": ("qualifier", "can't", "always", [], ["asyncio"]),
    "docker-volumes-uv-cache": ("qualifier", "re-downloading", "re-building", [], ["wheels"]),
    "regex-lookbehind-recall": ("qualifier", "work", "fail", [], ["ripgrep"]),
    "job-that-changes-its-own-code": ("qualifier", "rewriting", "deleting", [], ["code"]),
    "check-that-cannot-see-its-input": ("qualifier", "cannot read",
                                        "always reads", [], ["guard"]),
    "cheaper-model-every-turn": ("qualifier", "cheap", "biggest", [], ["expensive one"]),
    "gpu-ram-thin-should-not-reboot": ("qualifier", "thin", "abundant", [], ["model"]),
    "alarm-comes-back-after-fixed": ("qualifier", "fixed", "deleted", [], ["alert"]),
    "wrong-thing-broke-simultaneously": ("qualifier", "same minute",
                                         "same hour", [], ["cause"]),
    "djev-decision-engine-integration": ("qualifier", "second GPU",
                                         "third GPU", [], ["yes-or-no"]),
    "primary-ram-floor-changed": ("qualifier", "allowed to start above",
                                  "forbidden to start above", [], ["abort line"]),
    "prompt-surface-open-set": ("qualifier", "hand-written list",
                                "generated rule", [], ["prompt files"]),
    "graph-rerank-ab-cache": ("qualifier", "A/B tested", "never measured", [], ["production"]),
    "eval-north-star-candidate": ("qualifier", "not simply recall",
                                  "exactly recall", [], ["headline metric"]),
    "self-referential-check-catalogue": ("qualifier", "cannot read",
                                         "always reads", [], ["note"]),
    "eval-artifact-absolute-path": ("qualifier", "worktree", "live checkout", [], ["artifact"]),
    # the swap exchanges the artifact the answer points at, and names no sibling.
    "skill-mining-to-promotion": ("artifact", "trajectory skill mining",
                                  "nightly build", [], ["promotion step"]),
    "bash-arrays-vs-strings": ("artifact", "filenames", "package versions", [], ["bash"]),
    "browser-tool-falls-back": ("artifact", "HTTP request",
                                "WebSocket session", [], ["browser tool"]),
    "isaac-gr00t-n17": ("artifact", "humanoid foundation model",
                        "manipulation policy", [], ["NVIDIA"]),
    "eval-corpus-naming-conventions": ("artifact", "baseline artifacts",
                                       "session transcripts", [], ["naming rules"]),

    # ── #1354: questions answered by the lloyd checkout's architecture docs ──
    # No expect_entities and no sibling: one modifier or artifact swap each, the
    # pin being the system the question is about.
    "harness-system-prompt-frozen": ("qualifier", "never rebuild", "always rebuild",
                                     [], ["agent loop"]),
    "automod-gate-rungs": ("qualifier", "before it can land", "after it has landed",
                           [], ["automod"]),
    "recall-doc-pool-ordering": ("artifact", "candidate documents",
                                 "candidate entities", [], ["vault recall"]),
    "qmd-embed-model-switch": ("artifact", "embedding model", "reranking model",
                               [], ["qmd"]),
    "djev-rank-not-gate": ("qualifier", "fixed cutoff", "learned cutoff", [], ["djev"]),
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _match(value: str, candidates: set[str]) -> bool:
    """Presence test against a normalized retrieved-entity set.

    Substring in either direction: an expected 'entity graph' is satisfied by a
    retrieved row attributed to 'Entity Graph (KG)', which is what the store
    actually emits for some canonicals.
    """
    needle = _norm(value)
    if not needle:
        return False
    return any(needle in c or c in needle for c in candidates if c)


# ── generation (pure: no store, no network, no LLM) ──────────────────────────

def apply_perturbation(query: str, record: dict) -> str:
    """Swap the one named constraint. Case-insensitive, first occurrence only.

    Replacing only the first occurrence is what keeps this a one-axis change: a
    query that named the constraint twice would otherwise become two edits.
    """
    old, new = record["old_value"], record["new_value"]
    idx = query.lower().find(old.lower())
    if idx < 0:
        raise ValueError(f"{record.get('id', '?')}: {old!r} not in the query")
    return query[:idx] + new + query[idx + len(old):]


def build_perturbations(specs: list[dict]) -> list[dict]:
    """The committed records, regenerated deterministically from the plan.

    Raises rather than silently skipping: if the query set gains an id with no
    plan entry, or a plan entry's old_value stops matching its query, the rater
    would quietly start scoring fewer queries than the trend line claims.
    """
    out: list[dict] = []
    for spec in specs:
        qid = spec["id"]
        if qid not in PLAN:
            raise ValueError(f"{qid}: no perturbation plan entry")
        axis, old, new, to_move, pinned = PLAN[qid]
        if axis not in AXES:
            raise ValueError(f"{qid}: axis {axis!r} outside the taxonomy")
        query = spec["query"]
        if old.lower() not in query.lower():
            raise ValueError(f"{qid}: old_value {old!r} not in its query")
        rec = {
            "id": qid,
            "axis_changed": axis,
            "old_value": old,
            "new_value": new,
            "perturbed_query": apply_perturbation(query, {"old_value": old,
                                                          "new_value": new,
                                                          "id": qid}),
            "expected_to_move": list(to_move),
            "expected_pinned": list(pinned),
        }
        if axis in AXES_REQUIRE_MOVE_TARGET:
            if not to_move:
                raise ValueError(f"{qid}: {axis} axis names no move target")
            rec["sibling_source"] = "kg_alias_table"
        out.append(rec)
    planned = set(PLAN) - {s["id"] for s in specs}
    if planned:
        raise ValueError(f"plan entries with no query: {sorted(planned)}")
    return out


# Every entry whose declared pin is knowingly NOT a substring of its own
# perturbed query, with why it is still the right expectation. These are
# non-entity axes whose pin was taken from the query's `expect_entities` rather
# than from its prose — e.g. `qwen38-local-serving` asks "how do we serve a
# hybrid linear-attention model on a single 24GB GPU?" and pins "Qwen3.8", a
# model name the query only implies. That is a legitimate thing to ask retrieval
# to hold across a qualifier swap, so the pins stay; what must not happen again
# is the ENTITY axis doing it, because there the pin then names the very
# constraint the swap moved (#763 clause 2, `autonomy-pipeline` / "autonomy").
# `pins_absent_from_their_own_swap` returns exactly this set, and the suite
# asserts it — a sixth offender fails the test rather than joining the
# baseline's `counterfactual_failures` as a manufactured defect.
PINS_ABSENT_BY_DESIGN = (
    "qwen38-local-serving",       # pins "Qwen3.8"; the swap moves the VRAM qualifier
    "yaml-scalar-block-indent",   # pins "YAML Parser"; the swap moves "last"/"first"
    "check-that-cannot-see-its-input",   # pins "guard"; named in no query text
    "self-referential-check-catalogue",  # pins "note"; named in no query text
    "skill-mining-to-promotion",  # pins "promotion step"; prose says "which loop decides"
)


def pins_absent_from_their_own_swap(specs: list[dict],
                                    plan: dict[str, tuple] | None = None,
                                    ) -> dict[str, list[str]]:
    """{query id: the declared pins that its OWN variant text does not contain}.

    The mechanical reading of "constraints the query still names": for every
    entry with a non-empty `expected_pinned`, build the variant exactly as the
    rater will (`apply_perturbation`, case-insensitive first occurrence) and ask
    whether each pin survives into it. A pin that does not is an expectation the
    perturbed query cannot satisfy, so scoring it reports a defect on a retriever
    that behaved — which is what `autonomy-pipeline` has been doing on every
    nightly since it was written.

    Pure: reads no store, runs no retrieval. Entries with no pin are not
    reported — they are the unscored half, visible in the denominator, not a
    defect.
    """
    plan = PLAN if plan is None else plan
    out: dict[str, list[str]] = {}
    for spec in specs:
        qid = spec["id"]
        entry = plan.get(qid)
        if not entry:
            continue
        _axis, old, new, _to_move, pins = entry
        if not pins:
            continue
        try:
            perturbed = apply_perturbation(spec["query"],
                                           {"old_value": old, "new_value": new,
                                            "id": qid})
        except ValueError:
            # build_perturbations raises on this with a better message; the
            # audit has no opinion on a plan that cannot be applied.
            continue
        missing = [p for p in pins if p.lower() not in perturbed.lower()]
        if missing:
            out[qid] = missing
    return out


def emit_records(records: list[dict], path: Path = RECORD_PATH) -> None:
    blob = {
        "_comment": ("Deterministic counterfactual perturbations for the "
                     "retrieval eval (#541). Regenerate with "
                     "`python eval/counterfactual.py --write`; never re-derived "
                     "from live graph data at eval time. The pinned corpus and "
                     "vault_recall_queries.yaml ground truth are not read by "
                     "the generator."),
        "perturbations": sorted(records, key=lambda r: r["id"]),
    }
    path.write_text(yaml.safe_dump(blob, sort_keys=False, allow_unicode=True,
                                   default_flow_style=False))


def load_records(path: Path = RECORD_PATH) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — the perturbation records are committed and "
            "the rater cannot score without them")
    blob = yaml.safe_load(path.read_text()) or {}
    return {r["id"]: r for r in blob.get("perturbations") or []}


def verify_siblings(records: list[dict], resolve) -> list[dict]:
    """Audit the sibling chosen for every entity-axis swap (risk 1).

    ``resolve`` maps a name to its canonical, or None. Two things make a swap
    unfair, and only the swapped-in value is held to the table:

      new_value resolves to nothing  — an invented entity cannot be expected to
        move a seed set, so a failure there is the perturbation's fault, not the
        retriever's;
      old and new resolve to the SAME canonical — two surfaces of one entity.
        Asking retrieval to distinguish them asks it to be wrong.

    old_value resolving to nothing is fine and expected: it is the surface the
    query happens to use ('vLLM' is an entity name that was never registered as
    an alias surface). Run by hand after editing the plan, not from the nightly
    path — the alias table is live data and the records are frozen on purpose.
    """
    bad: list[dict] = []
    for rec in records:
        if rec.get("axis_changed") != ENTITY_AXIS:
            continue
        new_c = resolve(rec["new_value"])
        if not new_c:
            bad.append(rec)
            continue
        old_c = resolve(rec["old_value"])
        if old_c and _norm(old_c) == _norm(new_c):
            bad.append(rec)
    return bad


def _kg_store_module():
    """`app.kg_store`, imported lazily.

    This file is run BOTH as a script (`python eval/counterfactual.py --verify`,
    where `sys.path[0]` is `eval/` and `app` is unimportable) and imported as
    `eval.counterfactual` by `run_eval`, so the repo root is not on `sys.path`
    at import time. It is the same dance the `kg_db_path` helper used to do to
    reach `app.paths.VAULT_KG_DB`; that helper is gone now (removed with #1046)
    because the audit asks the store for its path instead of computing one.
    """
    if str(HERE.parent) not in sys.path:
        sys.path.insert(0, str(HERE.parent))
    from app import kg_store

    return kg_store


def _resolver_from_map(surfaces: dict[str, str]) -> callable:
    """Bind one surface->canonical map into the resolver shape `verify_siblings` takes."""
    def resolve(name: str):
        return surfaces.get((name or "").strip().lower())

    return resolve


def alias_resolver() -> callable:
    """surface -> canonical, from the store's own map (`aliases.all_lower()`).

    `all_lower()` is surface_lc -> canonical **plus every `entities.name` mapped
    to itself**, which closes the gap that made the first version of this
    resolver go around the table: the `aliases` table stores variant surfaces, so
    a canonical no alias row names is present nowhere in it, and a resolver built
    on that table alone calls such a name unknown. That is not hypothetical —
    `QMD Reranker`, `QMD Storage`, `qmd watcher`, `QMD memory search` and
    `QMD Search Pipeline Optimization` are each a registered entity holding zero
    alias rows (re-checked through `store()` on 2026-09-22), which is how #762
    measured a table-only resolver manufacturing an
    `entity_swap_seed_set_unchanged` label for `qmd` out of a seed set that had
    in fact moved. This function's first docstring gave a different example —
    that the table reports 'Knowledge Graph' as unknown — and that one is no
    longer true, since `knowledge graph` is a surface row today; the no-alias-row
    case above is what remains, and it is enough.

    Reading the store's dict is also what makes the old defect unrepresentable.
    `select distinct canonical` returns a one-column ROW, and `.lower()` on that
    tuple raised AttributeError: from this file's first commit (`af1e8c1`,
    2026-09-09) until `98fa216b` (2026-09-20) hand-indexed the row, `--verify` —
    the ONLY audit of an entity swap's chosen sibling — could not be invoked at
    all, and every `verify_siblings` claim in this file's tests had rested on a
    hand-written resolver that cannot fail that way.

    Going through `app.kg_store` rather than opening the file is the stated
    boundary — "`app.kg_store` is the only writer of the store" — and it is what
    makes an absent derived store raise `StoreUnavailable` instead of reading as
    an empty alias table (#1236). The one opener this does not cover is the
    guardian's read-only row count (#1525); an audit script is not that.
    """
    return _resolver_from_map(_kg_store_module().store().aliases.all_lower())


# ── scoring ──────────────────────────────────────────────────────────────────

def _retrieved(result: dict) -> tuple[set[str], dict[str, set[str]]]:
    """Entity-level projection of one recall result: the attributed entity set,
    and the fact text attributed to each entity.

    Never reads ``documents`` — see the module docstring on risk 3.
    """
    entities: set[str] = set()
    by_entity: dict[str, set[str]] = {}
    for key in ("facts", "graph_expanded_facts"):
        for fact in result.get(key) or []:
            ent = _norm(fact.get("entity", ""))
            if not ent:
                continue
            entities.add(ent)
            by_entity.setdefault(ent, set()).add(_norm(fact.get("text", "")))
    for nb in result.get("graph_neighbors_used") or []:
        ent = _norm(nb.get("entity", ""))
        if ent:
            entities.add(ent)
    return entities, by_entity


def _matched_by(value: str, by_entity: dict[str, set[str]]) -> set[str]:
    """Fact text attributed to every retrieved entity matching `value`."""
    needle = _norm(value)
    out: set[str] = set()
    if not needle:
        return out
    for ent, texts in by_entity.items():
        if needle in ent or ent in needle:
            out |= texts
    return out


def score_pair(record: dict, orig_result: dict, orig_seeds: list[str],
               var_result: dict, var_seeds: list[str]) -> dict:
    """moved / pinned for one query and its perturbed twin."""
    axis = record.get("axis_changed")
    retrieved_o, facts_o = _retrieved(orig_result)
    retrieved_v, facts_v = _retrieved(var_result)

    to_move = [t for t in (record.get("expected_to_move") or [])]
    if to_move:
        # Scored against the rows the variant attributed and the original did
        # not — never against the whole variant output. Tolerance to naming is
        # wanted (the store's canonical for 'Semantic Entity Resolution' is
        # 'semantic-entity-resolution-via-graph-embeddings', and refusing to
        # call that a move manufactures the false failure risk 1 warns about),
        # and restricting to `added` is what keeps that tolerance safe: a row
        # already present in the original arm cannot be credited as movement,
        # so 'QMD' never satisfies a swap to 'QMD Index'.
        added = retrieved_v - retrieved_o
        moved = any(_match(t, added) for t in to_move)
    else:
        moved = retrieved_v != retrieved_o

    pins = record.get("expected_pinned") or []
    if pins:
        pinned = True
        broken: list[str] = []
        for pin in pins:
            was = _match(pin, retrieved_o)
            now = _match(pin, retrieved_v)
            if was != now:
                pinned = False
                broken.append(f"{pin}: present={was}->{now}")
                continue
            if was and now and _matched_by(pin, facts_o) != _matched_by(pin, facts_v):
                pinned = False
                broken.append(f"{pin}: facts churned")
    else:
        pinned = None
        broken = []

    seeds_o = {_norm(s) for s in (orig_seeds or []) if _norm(s)}
    seeds_v = {_norm(s) for s in (var_seeds or []) if _norm(s)}

    return {
        "axis_changed": axis,
        "new_value": record.get("new_value"),
        "perturbed_query": record.get("perturbed_query"),
        "expected_to_move": to_move,
        "expected_pinned": pins,
        "counterfactual_moved": bool(moved),
        "counterfactual_pinned": pinned,
        "pinned_unscored": not pins,
        "pinned_failures": broken,
        # Seeds are extracted from the query text, so a swap the extractor never
        # noticed shows up here first — the label #537 needs.
        "seed_moved": seeds_v != seeds_o,
        "retrieved": sorted(retrieved_o),
        "retrieved_variant": sorted(retrieved_v),
        "retrieved_unchanged": retrieved_v == retrieved_o,
    }


def label_failures(records: list[dict]) -> list[dict]:
    """Cause labels over the per-query counterfactual blocks.

    The label set is the defect taxonomy #541 asks for; `label` is None where
    the query behaved. Ordered by diagnostic value, not by frequency: an entity
    swap the seed extractor did not notice is the cleanest existing evidence
    for or against #537's identity-key premise, so it wins over a generic
    'did not move'.
    """
    out: list[dict] = []
    for rec in records:
        block = rec.get("counterfactual") or {}
        label = None
        if block:
            if (block.get("axis_changed") == ENTITY_AXIS
                    and not block.get("seed_moved")):
                label = "entity_swap_seed_set_unchanged"
            elif block.get("counterfactual_pinned") is False:
                label = "pinned_axis_churned"
            elif not block.get("counterfactual_moved"):
                label = "axis_not_moved"
            elif block.get("counterfactual_pinned") is None:
                label = "nothing_pinned"
        out.append({"id": rec.get("id"), "label": label,
                    "axis_changed": block.get("axis_changed"),
                    "new_value": block.get("new_value"),
                    "pinned_failures": block.get("pinned_failures") or []})
    return out


def identity_keying_evidence(labelled: list[dict]) -> list[str]:
    """The query ids where an entity-name swap left the seed set unchanged."""
    return [row["id"] for row in labelled
            if row.get("label") == "entity_swap_seed_set_unchanged"]


def main(argv: list[str]) -> int:
    specs = yaml.safe_load((HERE / "vault_recall_queries.yaml").read_text())["queries"]
    if "--write" in argv:
        records = build_perturbations(specs)  # validates before writing
        emit_records(records)
        print(f"wrote {RECORD_PATH} ({len(records)} records)")
        return 0
    if "--verify" in argv:
        kg_store = _kg_store_module()
        try:
            # The store the scored run reads: `app.kg_store.store()` is the
            # process default over `app.paths.VAULT_KG_DB`, the same
            # ``VAULT_KG_DB`` run_eval records in corpus_provenance, so an audit
            # never runs against a different graph than the numbers it checks.
            kg = kg_store.store()
        except kg_store.StoreUnavailable as exc:
            print(f"no alias table to audit the siblings against: {exc}")
            return 2
        recs = list(load_records().values())
        entity_axis = [r for r in recs if r.get("axis_changed") == ENTITY_AXIS]
        bad = verify_siblings(recs, _resolver_from_map(kg.aliases.all_lower()))
        for rec in bad:
            print(f"  UNVERIFIED {rec['id']}: {rec['old_value']!r} -> "
                  f"{rec['new_value']!r} — the swapped-in value is not a "
                  "distinct canonical in the alias table")
        print(f"{len(bad)} of {len(entity_axis)} entity-axis pairs unverified "
              f"against {kg.path}")
        return 1 if bad else 0
    if "--check" in argv:
        built = {r["id"]: r for r in build_perturbations(specs)}
        committed = load_records()
        drift = [i for i, r in built.items() if committed.get(i) != r]
        print(f"{'DRIFT ' + str(drift) if drift else 'records match the generator'}")
        # The #763 clause-2 audit, on the same entry point a person runs before
        # committing a plan edit: an entry that pins a term its own swap deletes
        # can only ever be reported as a defect by a nightly that behaved, so it
        # belongs next to the drift check rather than only in the suite.
        offenders = pins_absent_from_their_own_swap(specs)
        unexpected = {q: p for q, p in offenders.items()
                      if q not in PINS_ABSENT_BY_DESIGN}
        for qid, missing in sorted(offenders.items()):
            flag = "by design" if qid in PINS_ABSENT_BY_DESIGN else "UNEXPECTED"
            print(f"  {flag:<11} {qid}: pins {missing} absent from its own "
                  f"perturbed query")
        print(f"pin audit: {len(offenders)} entries pin a term their own swap "
              f"deletes, {len(unexpected)} of them outside "
              f"PINS_ABSENT_BY_DESIGN")
        return 1 if (drift or unexpected) else 0
    print("usage: counterfactual.py --write | --check | --verify")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
