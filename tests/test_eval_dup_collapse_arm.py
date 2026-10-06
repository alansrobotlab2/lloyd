#!/usr/bin/env python3
"""The `--collapse-dups` ablation arm (#2271, clause 5).

Three things have to be true at once, and each has its own test below:

  * **one slot per near-duplicate cluster**, at a slot budget the arm may not
    raise;
  * **gold identity survives** — where the gold note is the OLDER twin, the
    collapsed arm still scores a doc hit on it, because an arm that collapsed gold
    into its newer twin would score a loss the collapse itself caused and would be
    measuring the scorer, not the retrieval (this item's named risk);
  * **slots freed is reported**, so the ablation says what it bought.

Fixture texts are vault-shaped notes; the seam is `ev._slot_text`, which is the ONE
reader the scoring path uses for slot content — stubbing it costs a real nightly
about 90 s of retrieval it does not need, and does not fake the part under test
(the clustering, the survivor choice, and the re-score).

Run from a worktree, never from a running checkout.
"""
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load():
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("run_eval", REPO_ROOT / "eval" / "run_eval.py")
    m = ilu.module_from_spec(spec)
    sys.modules["run_eval"] = m
    spec.loader.exec_module(m)
    return m


ev = _load()

_SHARED = (
    "The retrieval eval scores whether the gold document is in the returned top "
    "K and how high it sits, and when those two disagree the cause has never been "
    "named by any arm of this harness. Duplicate slots are one candidate mecha-"
    "nism for that divergence, because a twin occupies a slot that could have held "
    "something else, and the metric measures the share of slots that cost is."
)
_OTHER = (
    "Wake-word audio depends on the pulse media binding far more than on the "
    "acoustic models, and an outage blamed on a model is usually a device that "
    "moved between capture cards. The check is the binding, read directly, before "
    "anyone is allowed to say anything at all about the models themselves."
)
_THIRD = (
    "Nightly writers append knowledge notes without a principled forgetting rule, "
    "so the corpus grows sideways rather than upward and the same finding arrives "
    "again under a slightly different title a few weeks later. A number for that "
    "growth is what a retention proposal would need before anyone deletes a file."
)


def _note(title: str, body: str, stamp: str = "2026-05-01T00:00:00") -> str:
    return (f"---\ntags:\n- retrieval-eval\ntimestamp: '{stamp}'\n"
            f"title: {title}\n---\n\n{body}\n")


@pytest.fixture
def slots(monkeypatch):
    """Route every slot read in the scoring path through a caller-supplied map.

    Also asserts the seam is LIVE: a refactor that gives `_slot_metas` its own
    reader would leave this stub unreached, and every test below would read the
    empty map and pass vacuously.
    """
    holder: dict[str, str] = {}
    real = ev._slot_text
    seen = []

    def fake(path):
        assert holder, "stub reached with no fixture content: the seam moved"
        seen.append(path)
        return holder.get(path)

    monkeypatch.setattr(ev, "_slot_text", fake)
    yield holder, seen
    monkeypatch.setattr(ev, "_slot_text", real)


def _spec(documents, gold):
    return {"id": "collapse-arm", "query": "q", "expect_entities": [],
            "expect_docs": gold, "retrieval_result": {"documents": documents}}


def _docs(*paths):
    return [{"path": p, "citation": p, "snippet": "s"} for p in paths]


# ── the arm, end to end ──────────────────────────────────────────────────────


def _arm(query_spec: dict, docs: list, **kw):
    """Run the paired arm exactly as `run_eval` does: the baseline `base` is the
    same `_score` pass the nightly records, and the arm re-scores the collapsed
    document list against it."""
    result = {"documents": docs, "entities": [], "facts": [],
              "graph_facts": [], "vault_facts": []}
    base = ev._score(query_spec, result, **kw)
    return ev._collapse_rescore(query_spec, result, None, base)


def test_collapse_arm_scores_a_doc_hit_when_gold_is_the_older_twin(slots):
    """The clause's own scenario, and the named risk it exists to rule out.

    The returned top-K is [gold-older-twin, its newer twin, one distinct note].
    A newest-first collapse would keep the NEWER twin, delete the gold document,
    and score `doc_hit: False` — an artefact of the collapse reading as a retrieval
    regression. Gold identity wins over recency, so the arm still hits.
    """
    holder, _ = slots
    holder.update({
        "knowledge/gold-note.md": _note("gold-note", _SHARED, "2026-01-01T00:00:00"),
        "knowledge/gold-note-twin.md": _note("gold-note-twin", _SHARED,
                                             "2026-06-01T00:00:00"),
        "knowledge/wake.md": _note("wake-word", _OTHER, "2026-02-01T00:00:00"),
    })
    base = ev._score(_spec(_docs("knowledge/gold-note.md",
                                 "knowledge/gold-note-twin.md",
                                 "knowledge/wake.md"),
                           # the FULL slot path as the gold label: `_slot_gold_matcher`
                           # tests slot identity, and a substring gold label like
                           # `gold-note` is deliberately not an identity test (it would
                           # "match" every note in the family), which is the very
                           # confusion this arm must not fall into.
                           ["knowledge/gold-note.md"]),
                     {"documents": _docs("knowledge/gold-note.md",
                                         "knowledge/gold-note-twin.md",
                                         "knowledge/wake.md")},
                     dup_metrics=ev._dup_metrics_for_paths(
                         ["knowledge/gold-note.md", "knowledge/gold-note-twin.md",
                          "knowledge/wake.md"]),
                     collapse_dups=True)
    arm = base["dup_collapse"]
    assert arm["ran"] is True
    assert arm["doc_hit"] is True, "gold identity must survive the collapse"
    # The gold SLOT survives and is kept at the earliest gold position in its
    # cluster; its index can still move up when a twin above it is freed, which is
    # compaction of a slot that no longer exists, not a rank the collapse invented.
    # What the clause forbids — a collapse-induced miss — is `doc_hit` above.
    assert arm["first_doc_rank"] is not None
    assert arm["first_doc_rank"] <= base["first_doc_rank"], (
        "the gold slot never loses ground it had already earned")
    assert arm["gold_slots_in"] == 1 and arm["gold_slots_kept"] == 1
    assert arm["gold_survived"] is True
    assert arm["slots_freed"] == 1, "one twin removed from three slots"
    assert arm["n_slots_in"] == 3 and arm["n_slots_out"] == 2
    assert arm["dup_slot_share_after"] == 0.0, "the arm's own post-condition"


def test_collapse_arm_keeps_exactly_one_slot_per_cluster_and_never_raises_the_budget(
        slots):
    """`n_slots_out <= n_slots_in`, and no cluster contributes two survivors."""
    holder, _ = slots
    holder.update({
        "knowledge/a.md": _note("a", _SHARED, "2026-01-01T00:00:00"),
        "knowledge/a2.md": _note("a2", _SHARED, "2026-06-01T00:00:00"),
        "knowledge/a3.md": _note("a3", _SHARED, "2026-03-01T00:00:00"),
        "knowledge/wake.md": _note("wake-word", _OTHER, "2026-02-01T00:00:00"),
    })
    paths = ["knowledge/a.md", "knowledge/a2.md", "knowledge/a3.md",
             "knowledge/wake.md"]
    out = _arm(_spec(_docs(*paths), ["wake-word"]), _docs(*paths))
    assert out["n_slots_in"] == 4 and out["n_slots_out"] == 2
    assert out["n_slots_out"] <= out["n_slots_in"], "the budget may not rise"
    assert out["slots_freed"] == 2
    assert out["dup_slot_share_after"] == 0.0


def test_collapse_arm_uses_slot_identity_not_the_scorers_substring_rule(slots):
    """A gold LABEL is a substring, a gold SLOT is an identity, and the arm uses identity.

    Gold `2026-05-1` is satisfied by BOTH daily notes: `_doc_pair_satisfied` matches
    it against either path, so the scorer records this query as hitting its gold
    document. The arm must not read that as "both slots ARE the gold document".
    Run against the same cluster through the shipped `collapse_slots`, the two rules
    free the SAME number of slots — one per cluster is unconditional — and differ
    only in which slot survives: the exact matcher keeps the NEWER twin, while the
    substring rule finds both slots "gold", applies gold-beats-recency, and pins the
    EARLIER one. So `slots_freed` cannot detect the confusion at all, which is
    exactly why this test pins the survivor and not the count: an arm wired to the
    scorer's rule would report a clean-looking 1 slot freed on every query while
    quietly running the oldest-not-newest collapse the ablation is not.
    """
    holder, _ = slots
    holder.update({
        "memory/2026-05-11.md": _note("2026-05-11", _SHARED, "2026-05-11T00:00:00"),
        "memory/2026-05-12.md": _note("2026-05-12", _SHARED, "2026-05-12T00:00:00"),
    })
    paths = ["memory/2026-05-11.md", "memory/2026-05-12.md"]
    out = _arm(_spec(_docs(*paths), ["2026-05-1"]), _docs(*paths),
               dup_metrics=ev._dup_metrics_for_paths(paths))
    assert out["baseline"]["doc_hit"] is True, (
        "the scorer's substring rule IS satisfied by both slots here; the label is "
        "not a nonexistent one — that is what makes this the collision case")
    assert out["doc_hit"] is True, "the surviving slot still satisfies the label"
    assert out["gold_slots_in"] == 0, (
        "slot identity sees no gold slot in a set the scorer just called a hit")
    assert out["slots_freed"] == 1 and out["n_slots_out"] == 1

    metas, texts = ev._slot_metas(paths)
    sims = ev._dup.pairwise_sims(texts)
    slot_paths = [m["path"] for m in metas]
    exact = ev._dup.collapse_slots(slot_paths, metas, sims,
                                   ev._slot_gold_matcher(["2026-05-1"]))
    substring = ev._dup.collapse_slots(slot_paths, metas, sims,
                                       lambda p: "2026-05-1" in p)
    assert exact.paths == ["memory/2026-05-12.md"], (
        f"the shipped exact matcher keeps the newer twin: {exact.paths}")
    assert substring.paths == ["memory/2026-05-11.md"], (
        "a label that only SATISFIES both paths, mistaken for identity, makes "
        "gold-beats-recency pin the EARLIER slot — the ablation would report a "
        f"newest-first collapse and run an oldest-first one: {substring.paths}")
    assert len(exact.paths) == len(substring.paths), (
        "the freed counts are identical under both rules, which is why the survivor "
        "above is the assertion that matters")


def test_collapse_arm_off_by_default_and_reported_when_on(slots):
    """The nightly default records no arm; asking for it records it."""
    holder, _ = slots
    holder.update({"knowledge/a.md": _note("a", _SHARED),
                   "knowledge/a2.md": _note("a2", _SHARED)})
    paths = ["knowledge/a.md", "knowledge/a2.md"]
    spec, res = _spec(_docs(*paths), ["a"]), {"documents": _docs(*paths)}
    dm = ev._dup_metrics_for_paths(paths)
    assert ev._score(spec, res, dup_metrics=dm)["dup_collapse"] is None
    assert ev._score(spec, res, dup_metrics=dm,
                     collapse_dups=True)["dup_collapse"]["ran"] is True


def test_collapse_arm_reports_a_no_verdict_rather_than_a_fabricated_one(tmp_path,
                                                                       monkeypatch):
    """Nothing readable means nothing measured: `ran: False` with a reason.

    A zero-slot arm that reported `doc_hit: False, slots_freed: 0` would read as a
    query where collapsing changed nothing, which is the opposite of true.
    """
    monkeypatch.setattr(ev, "_slot_text", lambda p: None)
    paths = ["knowledge/gold.md", "knowledge/twin.md"]
    out = _arm(_spec(_docs(*paths), ["gold"]), _docs(*paths))
    assert out["ran"] is False and out["slots_freed"] == 0
    assert "readable" in out["reason"]
    # the record says WHY, and no scores are attached to it, so a later reader can
    # tell "collapsing changed nothing" from "nothing was measurable" — a 2-slot
    # return with an unreadable corpus is the second case, not the first
    assert out["n_slots_in"] == out["n_slots_out"] == 2
    assert "doc_hit" not in out, "a no-verdict record publishes no scores"
    assert out["gold_survived"] is True


def test_collapse_error_keeps_the_baseline_and_is_named(slots, monkeypatch):
    """A raise inside the predicate degrades to the un-collapsed set, named.

    The arm is optional on every query; it must never cost the nightly its scores.
    """
    holder, _ = slots
    holder.update({"knowledge/a.md": _note("a", _SHARED),
                   "knowledge/a2.md": _note("a2", _SHARED)})
    paths = ["knowledge/a.md", "knowledge/a2.md"]
    monkeypatch.setattr(ev._dup, "collapse_slots",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = _arm(_spec(_docs(*paths), ["a"]), _docs(*paths),
               dup_metrics=ev._dup_metrics_for_paths(paths))
    assert out["dup_collapse_error"] == "RuntimeError: boom"
    assert out["ran"] is True
    assert out["slots_freed"] == 0 and out["n_slots_out"] == out["n_slots_in"] == 2
    assert out["doc_hit"] is True, "the baseline's own verdict stands untouched"
    # The degraded arm re-scored the FULL set, and its own post-condition says so:
    # both twins are still in the slots, so the share it reports is the 2 / 2 it
    # started with. A reader who saw `dup_slot_share_after: 0.0` here would record
    # "collapsed clean" for a query where the collapse never ran — which is the only
    # way this arm can lie about a failure, so it is the thing pinned.
    assert out["dup_slot_share_after"] == pytest.approx(1.0), (
        f"a failed arm must not report a collapsed set: {out['dup_slot_share_after']}")
    assert out["baseline"]["dup_slot_share"] == pytest.approx(1.0), (
        "the after-share and the baseline share are the same measurement here, "
        "because the fallback kept every slot")
