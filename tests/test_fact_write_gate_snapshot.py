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
  * #2775 — a seeded leg must not cost one inode per corpus file: the seed's file
    inodes are shared, a replayed write never reaches through a shared inode, a
    seed that cannot be linked is copied rather than crashing the run, a `--out`
    filesystem that cannot hold the run is refused before anything is created,
    and no documented example points `--out` at tmpfs `/tmp`.

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
import errno
import json
import os
import re
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


# ── #2775: seeding a leg must not cost a corpus inode per corpus file ────────
#
# The failure this section pins is measured, not hypothetical: the #2768 round
# filled `/tmp` (tmpfs, 1,048,576 inodes) to 81 % with ten full `copytree`
# copies of a 62,806-file facts corpus — 816,670 inodes in `/tmp/fwg-b` — one
# inode per corpus file per arm, with nothing anywhere on the path checking the
# filesystem it was filling. The five rails #2775 asks for, in the order they are
# pinned below: share the seed's inodes, never write through a shared one, survive
# a seed that cannot be linked, price the run before paying for it, and do not
# point the documented example at a filesystem with a million inodes on it. Each
# of the first four was mutation-checked — removing the mechanism from the module
# kills its node — and the fifth is a text assertion on the two surfaces the
# clause names.

#: The two seeded files a leg's replay rewrites: `Relay` is the entity
#: `WRITE_SEQUENCE` writes, so its `state` and `history` files are the ones that
#: must not still be the seed's inode by the end of a build.
SEED_REPLAYED = ("Relay/Relay-state.md", "Relay/Relay-history.md")

#: The file `isolated_tree` points `_shared.ALIASES_PATH` at, so it is the one
#: seeded file a leg is allowed to write even though nothing in this builder
#: writes it today. Copied per leg rather than linked for that reason.
SEED_ALIAS_FILE = "entity-aliases.json"

#: The documented `--out`, spelled once here so a docstring or help edit that
#: moves it off on-disk scratch has to fail something.
DOCUMENTED_OUT = ".lloyd-data/eval/fact-write-gate/pair"


def _seed_facts(root: Path) -> Path:
    """A facts root in the `<corpus>/facts` layout, written by the real writer.

    Twelve fact files in six entity dirs plus the root's `entity-aliases.json`:
    enough files that "one inode per corpus file per leg" and "one seed's
    inodes" are plainly different numbers, small enough to read.
    `Relay/Relay-state.md` is seeded with the prior `WRITE_SEQUENCE` rewrites, so
    a seeded pair has a file to diverge on INSIDE the seeded tree — the only
    place a shared inode is dangerous. The `Bearing*` entities are filler no
    replay touches.
    """
    facts = root / snap.SNAPSHOT_FACTS_SUBDIR
    files: dict[str, list[str]] = {
        "Relay/state": [snap.WRITE_SEQUENCE[0][2]],
        "Relay/history": ["Relay moved to the tailnet in 2026-08."],
    }
    for i in range(5):
        files[f"Bearing{i}/state"] = [f"Bearing{i} is torqued to 12 N-m."]
        files[f"Bearing{i}/history"] = [f"Bearing{i} was replaced in 2026-04."]
    for rel, texts in files.items():
        entity, category = rel.split("/")
        entries = [{"fact": t, "confidence": 0.9, "category": category, "id": None,
                    "created_at": NOW, "valid_at": None, "invalid_at": None,
                    "expired_at": None, "provenance": "STATED", "source_doc": None}
                   for t in texts]
        snap.assign_ids(entries, category)
        snap._write_list(snap.fact_file(facts, entity, category), entity, category,
                         entries, NOW)
    (facts / SEED_ALIAS_FILE).write_text('{"aliases": {}}', encoding="utf-8")
    return facts


def _seeded_files(seed: Path) -> list[Path]:
    """Every file under a seed facts root, sorted, so a fixture can assert its own size."""
    return [p for p in sorted(seed.rglob("*")) if p.is_file()]


def _inode_set(*roots: Path) -> set[int]:
    """The distinct file inodes the given trees address. This, not `len(walk())`,
    is what a filesystem charges for: two links to one file cost one inode."""
    out: set[int] = set()
    for root in roots:
        for p in sorted(root.rglob("*")):
            if p.is_file() or p.is_symlink():
                out.add(p.stat().st_ino)
    return out


def test_seeding_two_legs_shares_the_seed_s_inodes_instead_of_paying_a_corpus_per_leg(tmp_path,
                                                                                    supersede):
    """Clause 1: two legs seeded from one tree address one seed's file inodes, not
    two seeds' — pinned by the same-relative-path file in each leg reporting an
    identical `st_ino`, and by the seed file itself reporting three links.

    The 12 untouched seeded files are the ones that must be shared. The two
    `Relay` files a replay rewrites and the aliases file a leg may write are the
    two exceptions, and they are counted as exceptions rather than ignored: a
    build that copied everything would cost 26 file inodes here against the 16
    this asserts.
    """
    seed = _seed_facts(tmp_path / "corpus")
    seed_files = _seeded_files(seed)
    assert len(seed_files) == 13, "12 fact files plus the root's aliases file"
    untouched = [p for p in seed_files
                 if p.relative_to(seed).as_posix() not in SEED_REPLAYED
                 and p.name != SEED_ALIAS_FILE]
    assert len(untouched) == 10

    out = tmp_path / "pair"
    manifest = snap.build(out, now_iso=NOW, seed=seed)
    legs = [out / m / snap.SNAPSHOT_FACTS_SUBDIR for m in snap.MODE_PAIR]
    assert manifest["seed"] == str(seed.resolve())

    shared = seed / "Bearing0/Bearing0-state.md"
    for leg in legs:
        leg_file = leg / "Bearing0/Bearing0-state.md"
        assert leg_file.stat().st_ino == shared.stat().st_ino, \
            "the same file in the seed and in a leg is a different inode"
        assert leg_file.read_bytes() == shared.read_bytes()
    assert shared.stat().st_nlink == 3, "seed + both legs are one inode, three links"

    # The filesystem's own bill: one seed's worth of inodes for the shared part,
    # plus a fresh inode for each of the 2 replayed files in each of the 2 legs,
    # plus a copied aliases file per leg. Two `copytree`s would bill 26.
    assert len(_inode_set(*legs)) == len(untouched) + 2 * len(SEED_REPLAYED) + 2
    assert len(_inode_set(*legs)) < 2 * len(seed_files)


def test_a_replayed_seeded_pair_leaves_the_seed_tree_byte_identical(tmp_path, supersede):
    """Clause 2: a leg's replay can never alter the seed, or the other leg, through
    a shared inode.

    `_write_list` used to `write_text` in place with no unlink. Against a hardlinked
    seed that writes INTO the seed's inode: the frozen corpus stops being the thing
    both legs were measured against, and the `noop` leg's reconstruction of a file
    leaks through the same inode into the `on` leg — destroying the exact
    differential the pair exists to measure. So the seed is hashed before and after,
    and each leg's rewritten file has to be a DIFFERENT inode from the seed's and
    from the other leg's while the bytes on the two legs diverge.
    """
    seed = _seed_facts(tmp_path / "corpus")
    before = {p.relative_to(seed).as_posix(): (p.read_bytes(), p.stat().st_ino)
              for p in _seeded_files(seed)}

    out = tmp_path / "pair"
    snap.build(out, now_iso=NOW, seed=seed)

    after = {p.relative_to(seed).as_posix(): (p.read_bytes(), p.stat().st_ino)
             for p in _seeded_files(seed)}
    assert after == before, "a leg's replay wrote through a shared inode into the seed"
    seeded = _facts_of(tmp_path / "corpus")
    assert len(seeded) == 12, "the seed still holds exactly the facts it was seeded with"
    assert seeded[STATE_KEY] == (snap.WRITE_SEQUENCE[0][2], None), \
        "the seed's own prior picked up a leg's expired_at"

    rel = Path("Relay/Relay-state.md")
    seed_ino = (seed / rel).stat().st_ino
    leg_paths = [out / m / snap.SNAPSHOT_FACTS_SUBDIR / rel for m in snap.MODE_PAIR]
    leg_inos = [p.stat().st_ino for p in leg_paths]
    assert all(i != seed_ino for i in leg_inos), \
        "a rewritten leg file is still the seed's inode"
    assert leg_inos[0] != leg_inos[1], "the noop leg and the on leg share one inode"
    assert leg_paths[0].read_bytes() != leg_paths[1].read_bytes(), \
        "the two legs wrote the same bytes through one link"

    for leg in [out / m / snap.SNAPSHOT_FACTS_SUBDIR for m in snap.MODE_PAIR]:
        assert (leg / SEED_ALIAS_FILE).stat().st_nlink == 1, \
            "the aliases file `isolated_tree` points a leg at is a link into the seed"


def test_a_seed_the_filesystem_cannot_link_falls_back_to_copying_instead_of_crashing(tmp_path,
                                                                                    supersede,
                                                                                    monkeypatch):
    """Clause 3: seed and `--out` on different filesystems is the ordinary case
    (a disk corpus, a scratch `--out`), and `os.link` answers it with `EXDEV`. A
    builder that lets that escape crashes mid-seed; this one copies.

    `os.link` is the only thing patched, so everything below it — the walk, the
    copy fallback, the replay, the differential — is the real path.
    """
    seed = _seed_facts(tmp_path / "corpus")

    def exdev(*args, **kwargs):
        raise OSError(errno.EXDEV, "Invalid cross-device link", args[1] if len(args) > 1 else "")

    monkeypatch.setattr(os, "link", exdev)
    out = tmp_path / "pair"
    manifest = snap.build(out, now_iso=NOW, seed=seed)
    assert manifest["seed"] == str(seed.resolve())

    for mode in snap.MODE_PAIR:
        leg = out / mode / snap.SNAPSHOT_FACTS_SUBDIR
        filler = leg / "Bearing0/Bearing0-state.md"
        assert filler.is_file(), "the EXDEV fallback lost the seeded corpus"
        assert filler.read_bytes() == (seed / "Bearing0/Bearing0-state.md").read_bytes()
        assert filler.stat().st_nlink == 1, "a leg built by fallback still links the seed"
    assert manifest["shows_expiry_differential"] is True, \
        "the pair still measures something when it had to be copied"


def test_a_filesystem_short_on_inodes_is_refused_before_the_out_dir_is_created(tmp_path,
                                                                              supersede,
                                                                              monkeypatch):
    """Clause 4: the builder prices the run against the `--out` filesystem's free
    inodes and refuses with both numbers, leaving the out dir uncreated.

    Nothing on the eval path ever read `f_ffree`: `git grep -n "statvfs\\|f_ffree"
    -- eval/` at #2775's triage returned one hit, and it was prose inside a stored
    eval answer. The run that filled `/tmp` to the guardian's own 80 % alarm line
    was one pair away from exhausting it, and the only thing between that and a
    host where every `mktemp` fails (the 2026-09-29 class) was luck.

    The build that fits is asserted FIRST, so the refusal below is the estimate
    being tight and not the fixture being unable to write anything at all. Where
    the host filesystem keeps an inode table that is checked through the real
    `statvfs`; where it does not — btrfs under `/home`, which is both this round's
    worktree and the gate's own pytest basetemp — `f_ffree` is 0 for that reason
    alone, and the room has to be synthesised or the node is asserting on the
    filesystem type of the machine that happens to run it. It did exactly that, and
    the tests rung caught it, on the first gate of this round.
    """
    seed = _seed_facts(tmp_path / "corpus")
    out = tmp_path / "pair"
    needed = snap.planned_inode_need(legs=len(snap.MODE_PAIR), seed=seed)
    assert needed > 0
    total, free = snap.inode_capacity(out)
    if total == 0:
        # `statvfs_result` is positional: (bsize, frsize, blocks, bfree, bavail,
        # FILES, FFREE, favail, fsid, flag, namemax). Room is f_ffree, the 7th field.
        monkeypatch.setattr(os, "statvfs", lambda path: os.statvfs_result(
            (4096, 4096, 0, 0, 0, needed + 1000, needed + 1000, 7, 0, 0, 255)))
    else:
        assert free >= needed
    assert snap.build(out, now_iso=NOW, seed=seed)["out"] == str(out.resolve())

    tight = tmp_path / "tight"
    # A table that exists (f_files = needed + 1, so the check runs) with 7 inodes
    # free in it — seven against the `needed` the run is about to ask for.
    monkeypatch.setattr(os, "statvfs", lambda path: os.statvfs_result(
        (4096, 4096, 0, 0, 0, needed + 1, 7, 7, 0, 0, 255)))
    with pytest.raises(snap.SnapshotRefused) as exc:
        snap.build(tight, now_iso=NOW, seed=seed)
    msg = str(exc.value)
    assert not tight.exists(), "the refusal created the out directory anyway"
    assert "7 free inodes" in msg, msg
    assert f"needs {needed}" in msg, msg
    # Unseeded legs are priced too: three synthetic facts are a handful of inodes,
    # but "this filesystem cannot hold what I am about to write" is not a property
    # of the seed.
    with pytest.raises(snap.SnapshotRefused):
        snap.build(tight / "plain", now_iso=NOW)


def test_a_filesystem_that_keeps_no_inode_table_is_not_read_as_being_full(tmp_path, supersede,
                                                                         monkeypatch):
    """`f_files == 0` means the filesystem has no inode table, which is not the same
    fact as having no free inodes — and the first real run of the guard below
    conflated them on this host's own btrfs and refused the build it was pricing.

    Everything under `/home` is btrfs, INCLUDING the data root and every round's
    worktree, and `statvfs` there reports `(f_files, f_ffree) == (0, 0)` because its
    inodes are 64-bit numbers rather than slots in a table. Refusing on that reads
    as a safety rail and is a rail that blocks every seeded eval on the machine.
    """
    seed = _seed_facts(tmp_path / "corpus")
    monkeypatch.setattr(os, "statvfs", lambda path: os.statvfs_result(
        (4096, 4096, 0, 0, 0, 0, 0, 0, 0, 0, 255)))
    out = tmp_path / "pair"
    manifest = snap.build(out, now_iso=NOW, seed=seed)
    assert manifest["inode_budget"]["checked"] is False, manifest["inode_budget"]
    assert manifest["shows_expiry_differential"] is True


def test_no_documented_out_example_puts_a_snapshot_on_tmpfs_tmp():
    """Clause 5: the invocation examples are the interface, and on both revisions
    they put `--out` at `/tmp/fwg-…` — onto a tmpfs that holds 1,048,576 inodes in
    total and was at 81 % of them when #2775 was filed.

    The documented place is the tree's own gitignored data dir, `.lloyd-data`,
    which is on disk, is under no VCS, and is not `production_data_root()` (which
    `refuse_production_root` rules out, so the live data root is NOT the answer to
    "where do eval artifacts go").
    """
    doc = snap.__doc__ or ""
    assert "/tmp/" not in doc, "the module docstring still points an example at tmpfs"
    assert DOCUMENTED_OUT in doc, "no example documents an on-disk out dir"

    # `format_help()` hard-wraps at terminal width AND breaks a long token at a
    # hyphen, so the example path can straddle two lines. It is compared with
    # whitespace removed rather than wherever the wrap happened to fall; the `/tmp`
    # ban is checked on the same text so a help edit cannot smuggle one past a line
    # break either.
    help_text = re.sub(r"\s+", "", snap.cli_parser().format_help())
    assert "/tmp" not in help_text, "the --out help still names tmpfs"
    assert DOCUMENTED_OUT in help_text, "the --out help names no on-disk location"


# ─────────────────────────────────────────────────────────────── #2768 ──
# Measurement (b) — the paired `knowledge_update` gain bar (b) is stated in.
#
# Two things about the instrument had to change before that bar could be measured
# at all, and both are pinned here. The first is the null: `WRITE_SEQUENCE` is
# three facts about one synthetic `Relay` entity, which no scored question
# mentions, so the pair the builder used to produce retired a fact no `<facts>`
# block ever rendered and the delta was 0.000 by construction (#2768 Findings
# bullet 2). The fix is `--replay-from`: writes named against the seeded corpus,
# executed on BOTH legs through the same `gate_write`, so the legs differ only by
# what the gate's verdict buys. The second is the witness: the pair is only worth
# scoring if a scored question's rendered block actually differs, and that claim
# is now committed as JSONL beside the note rather than asserted in prose.
REPLAY_DOC = (Path(__file__).resolve().parent.parent
              / "eval" / "measurements" / "fact-write-gate-b-2026-10-15")
BLOCKS_NOOP = REPLAY_DOC / "knowledge-update-blocks-noop.jsonl"
BLOCKS_ON = REPLAY_DOC / "knowledge-update-blocks-on.jsonl"
WITNESS = "lme-kn-013"


def test_a_replay_list_is_read_in_order_and_skips_blanks_and_leading_comments(tmp_path):
    src = tmp_path / "replay.jsonl"
    src.write_text(
        "# a comment line, which is not a write\n"
        "\n"
        '{"entity": "B", "category": "state", "fact": "second"}\n'
        '   \n'
        '{"entity": "A", "category": "state", "fact": "first"}\n',
        encoding="utf-8")
    assert snap.load_replay_list(src) == [
        ("B", "state", "second"), ("A", "state", "first")]


def test_a_replay_list_keeps_an_inline_hash_because_an_entity_is_named_one(tmp_path):
    """`#` mid-line is content: the corpus has an entity named `GitHub issue #38606`.

    A rule that stripped from any `#` would truncate that entity to `GitHub issue `
    and then refuse the file it no longer asked for — a builder that silently
    dropped a write is the thing that makes a pair quietly stop differing.
    """
    src = tmp_path / "replay.jsonl"
    src.write_text('{"entity": "GitHub issue #38606", "category": "state",'
                   ' "fact": "the entity name contains a hash"}\n', encoding="utf-8")
    assert snap.load_replay_list(src) == [
        ("GitHub issue #38606", "state", "the entity name contains a hash")]


def test_a_replay_list_row_missing_a_key_names_the_line_that_is_wrong(tmp_path):
    src = tmp_path / "replay.jsonl"
    src.write_text('{"entity": "A", "category": "state", "fact": "ok"}\n'
                   '{"entity": "A", "category": "state"}\n', encoding="utf-8")
    with pytest.raises(snap.SnapshotRefused) as exc:
        snap.load_replay_list(src)
    assert ":2" in str(exc.value) and "fact" in str(exc.value)


def _seed_entity(root: Path, entity: str, category: str, texts: list[str]) -> Path:
    """One entity's category file in the `<corpus>/facts` layout, by the real writer."""
    entries = [{"fact": t, "confidence": 0.9, "category": category, "id": None,
                "created_at": NOW, "valid_at": None, "invalid_at": None,
                "expired_at": None, "provenance": "STATED", "source_doc": None}
               for t in texts]
    snap.assign_ids(entries, category)
    path = snap.fact_file(root, entity, category)
    snap._write_list(path, entity, category, entries, NOW)
    return root


WITNESS_PRIOR = ("Hornet.dev presented on the effectiveness of BM25 for agentic "
                "search.")
WITNESS_WRITE = ("Hornet.dev presented on the effectiveness of BM25 for agentic "
                 "search, per the 2026-10-14 sync that replaces the earlier "
                 "reading on record")


def test_a_targeted_replay_reaches_both_legs_and_only_the_on_leg_applies_it(
        tmp_path, supersede):
    """One replayed write, both legs see it, only `on` stamps the prior.

    The `noop` leg is not a leg that skipped the write: it decided the same
    verdict and was forbidden to apply it, which is what production `noop` means.
    That is the whole content of the measurement — the pair differs by the
    retirement and by nothing else — so both legs' fact sets are asserted to carry
    the same texts before anything is allowed to differ, which a differential
    alone would not catch: it would also be non-empty if the legs had run
    different write sequences.
    """
    seed = _seed_entity(tmp_path / "seed" / "facts", "Hornet.dev", "event",
                        [WITNESS_PRIOR])
    out = tmp_path / "pair"
    manifest = snap.build(out, seed=seed, now_iso=NOW,
                          replay=[("Hornet.dev", "event", WITNESS_WRITE)])
    assert manifest["writes_seen"]["noop"] == manifest["writes_seen"]["on"]
    assert manifest["writes_applied"]["noop"] == 0, (
        "the noop leg applied an UPDATE: it is no longer the arm production runs")
    for mode in ("noop", "on"):
        replayed = [w for w in manifest["writes"][mode] if w["origin"] == "replay"]
        assert len(replayed) == 1, "the replayed write reached only one leg"
        assert replayed[0]["verdict"] == "update", replayed[0]
        assert replayed[0]["took"] == ("add" if mode == "noop" else "update")
    hornet = {k for k in manifest["differential"]["expired_only_in_on"]
              if k.startswith("Hornet.dev/")}
    assert hornet == {"Hornet.dev/Hornet.dev-event.md::evnt-001"}
    assert manifest["differential"]["expired_only_in_noop"] == []
    noop_facts, on_facts = _facts_of(out / "noop"), _facts_of(out / "on")
    assert {txt for txt, _ in noop_facts.values()} \
        == {txt for txt, _ in on_facts.values()}, (
        "the legs hold different fact texts, so the pair is not paired")
    assert noop_facts["Hornet.dev/Hornet.dev-event.md::evnt-001"][1] is None
    assert on_facts["Hornet.dev/Hornet.dev-event.md::evnt-001"][1] == NOW


def test_the_b_pair_is_built_from_a_replay_list_the_builder_reads(tmp_path):
    """The CLI's `--replay-from` is wired: the manifest names the file and its rows.

    Without this the measurement note's pair could have been built by anything —
    the manifest is what a reader re-checks the input against.
    """
    seed = _seed_entity(tmp_path / "seed" / "facts", "Relay", "state",
                        ["Relay's cap is 40 units."])
    src = tmp_path / "replay.jsonl"
    src.write_text('{"entity": "Relay", "category": "state",'
                   ' "fact": "Relay\'s cap is 40 units, per the 2026-10-14 sync."}\n',
                   encoding="utf-8")
    out = tmp_path / "pair"
    assert snap.main(["--out", str(out), "--seed-from", str(seed),
                      "--replay-from", str(src)]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["replay_from"] == str(src.resolve())
    assert manifest["replay"] == [
        {"entity": "Relay", "category": "state",
         "fact": "Relay's cap is 40 units, per the 2026-10-14 sync."}]


def test_a_replay_list_under_the_live_root_is_refused_like_every_other_input(tmp_path):
    src = production_data_root() / "replay-list.jsonl"
    assert src.is_relative_to(production_data_root())
    with pytest.raises(snap.SnapshotRefused) as exc:
        snap.load_replay_list(src)
    assert str(production_data_root()) in str(exc.value)


def _dump_rows(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "_header" in row:
            continue
        out[row["id"]] = row["facts_rel"]
    return out


def test_the_committed_block_dumps_differ_for_the_witness_question():
    """The measurement's whole claim, checkable without re-running the model.

    #2768 was filed because the previous pair's delta could not be read either
    way: no artifact showed what each leg rendered, so a 0.000 was equally a
    finding about the gate and a finding about the fixture. These two files are
    the per-question `<facts>` block lines of the two legs, committed as JSONL
    (a `.json` would be swallowed by `.gitignore`), and they differ on the
    witness: the `noop` leg renders the prior the `on` leg has retired.
    """
    assert BLOCKS_NOOP.is_file() and BLOCKS_ON.is_file(), (
        f"{BLOCKS_NOOP} and {BLOCKS_ON} are the committed renderings of each leg's "
        f"`<facts>` block; without them the paired delta has no witness")
    noop, on = _dump_rows(BLOCKS_NOOP), _dump_rows(BLOCKS_ON)
    assert WITNESS in noop and WITNESS in on, (
        f"{WITNESS} is the witness `knowledge_update` question (entity Hornet.dev)")
    assert noop[WITNESS] != on[WITNESS], (
        "the two legs render the same block for the witness, so the delta measures "
        "the instrument and not the gate")
    hornet = [ln for ln in noop[WITNESS] if "Hornet.dev" in ln]
    assert hornet and hornet != [ln for ln in on[WITNESS] if "Hornet.dev" in ln], (
        "the differing lines are not the witness entity's, so the block that "
        "changed belongs to some other entity")


def test_the_committed_block_dumps_name_the_arm_and_the_tree_each_came_from():
    """Each dump's header says which leg it is, from which store, in which order.

    The #2778 rail is the reason the arm matters: production ships
    `prefetch.facts.rank: relevance`, and a witness read off the confidence arm
    would not be a witness about the ordering anyone runs.
    """
    for path in (BLOCKS_NOOP, BLOCKS_ON):
        head = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert head["arms"] == ["prefetch_rel"], head["arms"]
        assert head["fact_snapshot"].endswith(("/noop", "/on")), head["fact_snapshot"]
        assert head["facts_rank"]["configured"] == "relevance", head["facts_rank"]
        assert head["facts_rank"]["arm_ordering"] == {"prefetch_rel": "relevance"}


def test_the_committed_witness_rows_show_which_prior_the_gate_retired():
    """The mechanism behind the block diff, per leg, without opening a snapshot tree.

    `Hornet.dev/Hornet.dev-state.md::stat-002` is the prior whose line the witness
    question renders: active in the `noop` leg, `expired_at`-stamped in the `on`
    leg. The line in the clause that names this witness originally pointed at
    `Hornet.dev-event.md::evnt-001`, and the measured ledger says that write was
    NOT an UPDATE (`target_fact: null`, `took: "add"` in both legs), so the id is
    pinned to what the ledger actually shows rather than to the triage note.
    """
    def facts(mode: str) -> dict[str, dict]:
        path = REPLAY_DOC / f"witness-entity-facts-{mode}.jsonl"
        return {f"{json.loads(l)['file']}::{json.loads(l)['id']}": json.loads(l)
                for l in path.read_text(encoding="utf-8").splitlines()}
    noop, on = facts("noop"), facts("on")
    prior = "Hornet.dev/Hornet.dev-state.md::stat-002"
    assert noop[prior]["expired_at"] is None, (
        "the noop leg must keep the prior active: that is what production `noop` is")
    assert on[prior]["expired_at"], "the on leg must stamp the prior expired"
    assert noop[prior]["fact"] == on[prior]["fact"], (
        "the legs hold different texts for the prior, so the pair is not paired")
    assert any("evnt-001" in k for k in noop), "the event file's rows are committed too"
