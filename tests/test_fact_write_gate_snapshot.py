"""#2344 — the noop/on fact-tree pair the write gate's arming bar is built from.

`config.yaml` reserves `knowledge_graph.write_gate.mode: "on"` behind "an explicit
decision from Alan plus a measured LloydMemEval `knowledge_update` gain, never a
count of logged rows", and the gain has been ungettable because every prefetch arm
read ONE tree — the live one, at `noop`. These tests pin the instrument that
changes that, and they pin it on the mechanism, not on a stub of it:

  * clause 3 — the builder replays one fixed sequence through the production
    `agent_mcp.fact_write_gate.gate_write` into two isolated trees, and the trees
    differ by exactly the field arming buys: the prior fact the decision named is
    `expired_at`-stamped in the `on` tree and still active in the `noop` tree.
  * clause 4 — no artifact of the build, and no path its readers are pointed at,
    resolves under `production_data_root()`.

What stays under the real code: `gate_write`'s decision, its in-place
`expired_at` stamp, `app.fact_ids` numbering, the front-matter writer, the gate's
own jsonl decision log, and `kg_store`'s index. The one thing a test controls is
the classifier's answer, through the same stand-in
`tests/test_fact_write_gate.py` installs — `djev.ask_sync`, with the same
superseded probabilities — and never through a `mode_=` argument. `mode_` is the
path an accidental arming could slip in by, so `LLOYD_FACT_WRITE_GATE` is the only
way a mode reaches these calls, and one test below proves that by watching every
`mode()` call the replay makes.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import fact_write_gate_snapshot as snap  # noqa: E402
from app import djev, kg_store, paths  # noqa: E402
from app.data_root import production_data_root  # noqa: E402
from agent_mcp import _shared, fact_write_gate as gate  # noqa: E402
from tests.test_fact_write_gate import Djev, SUPERSEDED  # noqa: E402

NOW = "2026-10-07T00:00:00+00:00"
# The id `app.fact_ids` mints for the first fact of the `state` file. It is NOT
# `fact-001`: `fact_id_stem(category)` derives the stem from the category, so
# this tree's ids read `stat-001`, `hist-001` — which is also why the report is
# keyed by file as well as id.
STATE_KEY = "Relay/Relay-state.md::stat-001"


@pytest.fixture
def supersede(monkeypatch):
    """Answer `superseded` at the engine seam, once per candidate, and count asks.

    Not the mechanism under test: what the gate asks djev and what an applied
    UPDATE stamps are already pinned in `tests/test_fact_write_gate.py`, which is
    where this stand-in and these probabilities come from. What is pinned here is
    WHERE the stamp lands — so only the model answer is answered for, the way a
    real run would not be.
    """
    fake = Djev(SUPERSEDED)
    monkeypatch.setattr(djev, "ask_sync", fake)
    return fake


def _facts_of(tree: Path) -> dict:
    """`file::id` → (text, expired_at) out of a built tree, through the real reader.

    Keyed `file::id` because `app.fact_ids` numbers within one file: `fact-001` is a
    handle inside a category file and an alias across every entity's first fact."""
    out = {}
    facts_root = tree / snap.SNAPSHOT_FACTS_SUBDIR
    for ffile in sorted(facts_root.glob("*/*.md")):
        fm = _shared._parse_fact_frontmatter(ffile.read_text(encoding="utf-8"))
        rel = str(ffile.relative_to(facts_root))
        for e in fm.get("facts") or []:
            out[f"{rel}::{e['id']}"] = (e["fact"], e.get("expired_at"))
    return out


def _log_rows(tree: Path) -> list[dict]:
    """The gate's own decision rows for one leg, read off the leg's own file."""
    return [json.loads(line) for line in
            (tree / snap.SNAPSHOT_LOG_NAME).read_text(encoding="utf-8").splitlines()]


# ── clause 3: the pair, and the one field that differs ───────────────────────

def test_the_pair_replays_one_sequence_and_differs_only_on_the_named_prior(tmp_path,
                                                                          supersede):
    """One fixed write sequence, two trees, and the differential is exactly
    `expired_at` on the fact the decision named — clause 3.

    Both legs must have run the SAME writes for the pair to be paired, so the fact
    keys and texts are asserted equal across them before anything is allowed to
    differ: a tree that differs because it wrote less is not a gate-mode pair.
    """
    manifest = snap.build(tmp_path / "pair", now_iso=NOW)

    noop, on = manifest["writes"]["noop"], manifest["writes"]["on"]
    assert [w["category"] for w in noop] == [w["category"] for w in on] == \
        [c for _, c, _ in snap.WRITE_SEQUENCE], "both legs replay the one sequence"
    assert [w["fact"] for w in noop] == [w["fact"] for w in on]

    noop_facts = _facts_of(tmp_path / "pair" / "noop")
    on_facts = _facts_of(tmp_path / "pair" / "on")
    assert set(noop_facts) == set(on_facts), "same facts in both trees"
    assert all(noop_facts[k][0] == on_facts[k][0] for k in noop_facts), \
        "same text for each fact"
    differs = [k for k in noop_facts if noop_facts[k][1] != on_facts[k][1]]
    assert differs == [STATE_KEY], f"expected one differing fact, got {differs}"
    assert on_facts[STATE_KEY][1] == NOW, "the on-tree prior carries the stamp"
    assert noop_facts[STATE_KEY][1] is None, "the noop-tree prior is still active"
    assert manifest["differential"] == {"expired_only_in_on": [STATE_KEY],
                                        "expired_only_in_noop": []}
    assert manifest["shows_expiry_differential"] is True

    # The fact the two trees disagree about is the fact the DECISION named, in the
    # leg that was allowed to act on it — not merely some fact that differs.
    decided = [w for w in on if w["verdict"] == "update"]
    assert len(decided) == 1, [w["verdict"] for w in on]
    assert STATE_KEY.endswith(f"::{decided[0]['target_fact_id']}")
    assert on_facts[STATE_KEY][0] == decided[0]["target_fact"], \
        "the stamped fact is the fact the verdict named, text for text"


def test_the_noop_leg_decides_and_logs_but_never_stamps_which_is_what_ships(tmp_path,
                                                                           supersede):
    """`noop` is the mode `config.yaml` ships, and it is not silence: it decides, it
    logs, and it applies nothing. Both legs therefore have the SAME decision on the
    record and differ only in the `applied` field and the tree — which is the whole
    point of pairing them. If this goes green the other way, the shipped mode
    changed under the fence."""
    manifest = snap.build(tmp_path / "pair", now_iso=NOW)

    assert [w["took"] for w in manifest["writes"]["noop"]] == ["add"] * 3, \
        "noop never withholds and never expires: every write lands as an add"
    assert manifest["writes"]["noop"][1]["verdict"] == "update", \
        "noop still reaches the same verdict, or the pair proves nothing"
    assert len(supersede.calls) == 2, \
        "one ask per leg — only the write with a candidate above the shortlist " \
        "floor asks, and the noop leg asks too"

    noop_rows, on_rows = (_log_rows(tmp_path / "pair" / "noop"),
                          _log_rows(tmp_path / "pair" / "on"))
    assert [(r["mode"], r["verdict"], r["applied"]) for r in noop_rows] == \
        [("noop", "update", "add")]
    assert [(r["mode"], r["verdict"], r["applied"]) for r in on_rows] == \
        [("on", "update", "update")]


def test_a_pair_with_no_update_warns_instead_of_publishing_an_empty_run(tmp_path,
                                                                       monkeypatch):
    """The failure this instrument would otherwise ship silently: djev unreachable,
    every decision falls back to ADD, both trees come out identical, and the run is
    a measurement of nothing. The builder says so in the manifest rather than
    emitting a pair that scores 0.0/0.0 — and every failure being ADD is the gate's
    own rule (`agent_mcp/fact_write_gate.py`), so the pair inherits it."""
    monkeypatch.setattr(djev, "ask_sync", Djev(fail=True))
    manifest = snap.build(tmp_path / "pair", now_iso=NOW)
    assert manifest["shows_expiry_differential"] is False
    assert manifest["warning"] and "djev did not answer" in manifest["warning"]
    assert json.loads((tmp_path / "pair" / "manifest.json")
                      .read_text(encoding="utf-8"))["warning"] == manifest["warning"]


def test_the_mode_reaches_the_gate_only_through_the_environment(tmp_path, supersede,
                                                               monkeypatch):
    """The builder never passes `mode_=` to `gate_write`, so the mode is read from
    `LLOYD_FACT_WRITE_GATE` by the production `mode()` — the same variable a live
    process reads. A builder that passed `mode_="on"` would be proving a wiring that
    exists nowhere in production, and would be the shape an accidental arming takes:
    the same call, with the armed mode, and nothing in the config to show for it."""
    seen: list[str | None] = []
    real_mode = gate.mode

    def spy():
        seen.append(os.environ.get(gate.MODE_ENV))
        return real_mode()

    monkeypatch.setattr(gate, "mode", spy)
    snap.build(tmp_path / "pair", now_iso=NOW)
    assert seen == ["noop"] * 3 + ["on"] * 3, \
        "mode() asked once per write, env-first, the noop leg entirely first"


# ── clause 4: nothing of this resolves inside the live data root ─────────────

def test_no_artifact_of_a_build_resolves_under_the_production_data_root(tmp_path,
                                                                       supersede):
    """Trees, indexes, decision logs and the manifest — every path a build emits —
    stay off the live root, clause 4. The builder re-points process-wide reader
    attributes, so a tree that resolved under `~/lloyd-data` would be a replay
    writing into the store that live turns read."""
    prod = Path(production_data_root()).resolve()
    manifest = snap.build(tmp_path / "pair", now_iso=NOW)

    emitted = [tmp_path / "pair", tmp_path / "pair" / "manifest.json"]
    for mode in snap.MODE_PAIR:
        tree = tmp_path / "pair" / mode
        emitted += [tree, tree / snap.SNAPSHOT_FACTS_SUBDIR,
                    tree / snap.SNAPSHOT_KG_DB_NAME, tree / snap.SNAPSHOT_LOG_NAME]
        assert (tree / snap.SNAPSHOT_KG_DB_NAME).exists(), "the leg built an index"
        assert (tree / snap.SNAPSHOT_LOG_NAME).exists(), "the leg built its log"
    for path in emitted:
        resolved = path.resolve()
        assert resolved != prod and prod not in resolved.parents, str(resolved)
    assert manifest["snapshots"] == {m: str((tmp_path / "pair" / m).resolve())
                                     for m in snap.MODE_PAIR}
    assert manifest["gate_logs"] == {m: str(tmp_path / "pair" / m /
                                            snap.SNAPSHOT_LOG_NAME)
                                     for m in snap.MODE_PAIR}


def test_a_root_under_the_live_data_root_is_refused_before_anything_is_written():
    """Not a warning: a `--out` that lands inside the live root is refused, and
    refused before a byte is written, because the argument is a bare directory path
    and a typo in it is exactly as cheap to catch here as it is expensive to undo."""
    prod = Path(production_data_root())
    with pytest.raises(snap.SnapshotRefused, match="production_data_root"):
        snap.build(prod / "_pipeline" / "gate-snapshots" / "oops")
    with pytest.raises(snap.SnapshotRefused, match="production_data_root"):
        snap.replay_tree(prod / "_pipeline" / "kg", "noop")
    assert not (prod / "_pipeline" / "gate-snapshots").exists(), \
        "the refusal wrote the directory anyway"


def test_the_live_readers_are_pointed_at_the_tree_only_for_the_replay(tmp_path,
                                                                     supersede):
    """The other half of clause 4: the build borrows the fact root, the alias file,
    the KG store and the decision-log path, and it gives all four back. A leaked
    pointer would leave the next reader in the same process — the suite, or anything
    that imports this builder — reading a throwaway tree and reporting it as the
    store. No `LLOYD_KG_DB` in the environment is the case that matters, because
    then the value being restored is the live store's own default."""
    # What the process looked like before, mode variable included: the live
    # backend's own environment carries `LLOYD_FACT_WRITE_GATE=off`, so "absent" is
    # not the resting state this has to return to — the PRIOR value is.
    before = (_shared.FACTS_ROOT, _shared.ALIASES_PATH, paths.FACT_WRITE_GATE_LOG,
              kg_store._default_path, os.environ.get(gate.MODE_ENV))
    snap.build(tmp_path / "pair", now_iso=NOW)
    assert (_shared.FACTS_ROOT, _shared.ALIASES_PATH, paths.FACT_WRITE_GATE_LOG,
            kg_store._default_path, os.environ.get(gate.MODE_ENV)) == before


def test_isolated_tree_refuses_a_live_root_even_when_asked_nicely():
    """The context manager is the piece a future caller will reuse, so the rail sits
    on the context manager and not only on `build`."""
    prod = Path(production_data_root())
    with pytest.raises(snap.SnapshotRefused):
        with snap.isolated_tree(prod / "_pipeline" / "vault-derived", "noop"):
            pass
