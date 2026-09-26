"""The nightly template<->live qmd config drift check (#1298).

qmd is configured by two hand-maintained copies of the same collection list:
the tracked template `agent-services/conf/qmd-index.yml` and the file the daemon
actually reads, `~/.config/qmd/index.yml`. SETUP.md:619 installs one onto the
other and SETUP.md:631-635 re-syncs it back; no code enforces either direction.
That is how the 2026-09-19 live edit — retargeting `sessions` from the empty
`~/obsidian/sessions` onto the real export directory, and dropping `facts` —
went unrecorded for a day while the tracked copy, the thing an operator or a new
host reads, still described 2026-09-07. Re-syncing from the stale template would
have retargeted `sessions` at an empty directory and silently dropped ~650
indexed documents from both the FTS and the vector legs.

The instrument that catches that is a *drift* check, not the path-existence
check the item originally proposed: both "dead" paths exist (each is an empty
directory), so existence was never the fault and a check on it passes today.

Every case here runs over fixture copies in `tmp_path`. Nothing reads
`~/.config/qmd/index.yml`, the repo's own template, or the live index — the
check is asserted against files this test owns, so it is green on a machine with
no qmd at all and cannot go red because somebody hand-edited their config.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.maintenance import qmd_index_maintenance as m  # noqa: E402

# A template/live pair in the shape #1298 was filed about: `facts` kept only by
# the template, `sessions` pointed at two different roots.
TEMPLATE = """\
collections:
  facts:
    path: /home/me/obsidian/facts
    pattern: "**/*.md"
  memory:
    path: /home/me/obsidian/memory
    pattern: "**/*.md"
  sessions:
    path: /home/me/obsidian/sessions
    pattern: "**/*.md"
"""

LIVE = """\
collections:
  memory:
    path: /home/me/obsidian/memory
    pattern: "**/*.md"
  autonomy-runs:
    path: /home/me/lloyd/autonomy-runs
    pattern: "*/run_*.md"
  sessions:
    path: /home/me/lloyd/_pipeline/vault-derived/sessions
    pattern: "**/*.md"
"""


def _pair(tmp_path: Path, template: str = TEMPLATE, live: str | None = LIVE) -> tuple[Path, Path]:
    t = tmp_path / "qmd-index.yml"
    t.write_text(template)
    l = tmp_path / "index.yml"
    if live is not None:
        l.write_text(live)
    return t, l


def _by_name(drift: list[dict]) -> dict[str, dict]:
    return {d["collection"]: d for d in drift}


# --- clause 3: name each collection whose path or presence differs ----------

def test_a_differing_path_is_named_with_both_sides(tmp_path):
    t, l = _pair(tmp_path)
    out = m.config_drift(t, l)
    d = _by_name(out["drift"])["sessions"]
    assert d["kind"] == "path_differs"
    assert d["template_path"] == "/home/me/obsidian/sessions"
    assert d["live_path"] == "/home/me/lloyd/_pipeline/vault-derived/sessions"


def test_a_template_only_collection_is_named(tmp_path):
    t, l = _pair(tmp_path)
    out = m.config_drift(t, l)
    d = _by_name(out["drift"])["facts"]
    assert d["kind"] == "template_only"
    assert d["template_path"] == "/home/me/obsidian/facts"
    assert d["live_path"] is None


def test_a_live_only_collection_is_named(tmp_path):
    """The `autonomy-runs` case: indexed, and invisible to a reader of the template."""
    t, l = _pair(tmp_path)
    out = m.config_drift(t, l)
    d = _by_name(out["drift"])["autonomy-runs"]
    assert d["kind"] == "live_only"
    assert d["live_path"] == "/home/me/lloyd/autonomy-runs"


def test_the_drift_set_is_exactly_the_differing_collections(tmp_path):
    """`memory` agrees in both files and must not be reported.

    A check that listed every collection would be indistinguishable from one
    that compared nothing, which is the failure mode this whole item is about.
    """
    t, l = _pair(tmp_path)
    out = m.config_drift(t, l)
    assert set(_by_name(out["drift"])) == {"facts", "sessions", "autonomy-runs"}
    assert out["drift_count"] == 3
    assert out["in_sync"] is False


def test_reordering_and_comments_are_not_drift(tmp_path):
    """A re-sync copies whole files, so order and comments reconcile themselves.

    Reporting them would make the check wrong on the run right after every
    legitimate re-sync, and a check that cries wolf gets ignored — which is how
    this one would end up back where #1298 started.
    """
    same = yaml.safe_load(TEMPLATE)["collections"]
    text = ("# someone's rationale comment, four lines\n"
            "# kept only on one side\n"
            "collections:\n"
            + "".join(
                f"  {name}:\n    path: {spec['path']}\n    pattern: \"{spec['pattern']}\"\n"
                for name, spec in reversed(list(same.items()))
            ))
    t, l = _pair(tmp_path, template=TEMPLATE, live=text)
    out = m.config_drift(t, l)
    assert out["drift"] == []
    assert out["in_sync"] is True


# --- clause 3: the re-sync command and its direction -------------------------

def test_drift_names_the_setup_documented_resync_in_the_live_to_template_direction(tmp_path):
    """The reported command must be SETUP.md's, not a paraphrase of it.

    Direction matters: the daemon's file is what retrieval is actually serving,
    so reconciling means copying live *onto* the template. The opposite
    direction is what would have dropped ~650 session documents.
    """
    t, l = _pair(tmp_path)
    out = m.config_drift(t, l)
    assert out["resync_command"] == (
        "cp ~/.config/qmd/index.yml agent-services/conf/qmd-index.yml"
    )
    setup = (ROOT / "SETUP.md").read_text()
    assert out["resync_command"] in setup, "SETUP.md no longer documents this command"
    assert "live" in out["resync_direction"] and "template" in out["resync_direction"]


def test_the_report_names_the_file_the_daemon_actually_reads(tmp_path):
    """`~/.config/qmd/index.yml`, pinned rather than paraphrased.

    The whole item is two files that look like one contract; a report that says
    "the config" without naming which one it compared cannot answer the question
    a reader came with. The template half of the same sentence belongs in the
    file's own header comment, which is a person's edit (#1301) because
    agent-services/conf/** is outside this loop's writable set.
    """
    assert m.LIVE_CONFIG == Path.home() / ".config/qmd/index.yml"
    t, l = _pair(tmp_path)
    out = m.config_drift(t, l)
    assert out["live"] == str(l) and out["template"] == str(t)


def test_an_in_sync_pair_reports_no_drift_but_still_carries_the_command(tmp_path):
    t, l = _pair(tmp_path, template=LIVE, live=LIVE)
    out = m.config_drift(t, l)
    assert out["drift_count"] == 0
    assert out["in_sync"] is True
    assert out["resync_command"] == m.RESYNC_COMMAND


# --- clause 3: no live config => no drift recorded, and no failure ----------

def test_missing_live_config_records_no_drift_and_never_raises(tmp_path):
    t, l = _pair(tmp_path, live=None)
    assert not l.exists()
    out = m.config_drift(t, l)
    assert out["drift"] == []
    assert out["drift_count"] == 0
    assert "no qmd config" in out["note"]
    assert out["resync_command"] == m.RESYNC_COMMAND


def test_missing_template_records_no_drift_and_never_raises(tmp_path):
    t, l = _pair(tmp_path)
    t.unlink()
    out = m.config_drift(t, l)
    assert out["drift"] == []
    assert out["drift_count"] == 0
    assert "no qmd config" in out["note"]


def test_an_unparsable_config_is_a_note_not_an_exception(tmp_path):
    """The nightly job prunes embeddings; a broken config must not kill it."""
    t, l = _pair(tmp_path, template="collections: [this is not a mapping\n  broken")
    out = m.config_drift(t, l)
    assert out["drift"] == []
    assert out["drift_count"] == 0
    assert "note" in out


def test_a_collections_block_that_is_not_a_mapping_is_a_note(tmp_path):
    """Parses fine, still unreadable — and must not read as "no such file"."""
    for body in ("collections: qmd\n", "collections:\n  - memory\n"):
        t, l = _pair(tmp_path, template=body)
        out = m.config_drift(t, l)
        assert "collections" in out["note"], body
        assert out["drift"] == []


def test_a_collection_whose_body_is_not_a_mapping_does_not_raise(tmp_path):
    """`vault: qmd` written where `vault:\\n    path: ...` belongs.

    The first version read the block with `dict(spec or {})`, and `dict("qmd")`
    is a ValueError: one mistyped line took the nightly job down, which is the
    file the check exists to notice stopping the check. The other collections are
    still compared and the unreadable one is named, never silently agreed.
    """
    t, l = _pair(
        tmp_path,
        template="collections:\n"
                 "  vault: qmd\n"
                 "  memory:\n    path: /home/me/obsidian/memory\n",
        live="collections:\n"
             "  memory:\n    path: /home/me/obsidian/elsewhere\n",
    )
    out = m.config_drift(t, l)
    assert out["comparable"] is True, out
    assert out["malformed"] == {"template": ["vault"], "live": []}
    kinds = _by_name(out["drift"])
    # `vault` is in the template and not in the live config at all, so its
    # presence is already an answer that does not need the body: template_only.
    assert kinds["vault"]["kind"] == "template_only"
    assert kinds["memory"]["kind"] == "path_differs"


def test_a_collection_declared_on_both_sides_with_an_unreadable_body(tmp_path):
    """Both files name the collection; one body will not parse. Not a comparison."""
    t, l = _pair(
        tmp_path,
        template="collections:\n  vault: qmd\n",
        live="collections:\n  vault:\n    path: /home/me/vault\n",
    )
    out = m.config_drift(t, l)
    assert out["in_sync"] is False
    entry = _by_name(out["drift"])["vault"]
    assert entry["kind"] == "uncomparable_body"
    assert entry["live_path"] == "/home/me/vault"
    assert out["malformed"] == {"template": ["vault"], "live": []}


def test_two_unreadable_bodies_are_never_scored_as_in_sync(tmp_path):
    """The same malformed line on both sides is "not compared", not "agree"."""
    body = "collections:\n  vault: qmd\n  memory: qmd\n"
    t, l = _pair(tmp_path, template=body, live=body)
    out = m.config_drift(t, l)
    assert out["in_sync"] is False
    assert out["drift_count"] == 2
    assert {d["kind"] for d in out["drift"]} == {"uncomparable_body"}


# --- the seam: the job the daemon's operator actually reads ------------------
# `config_drift` returning a dict proves nothing on its own. What the nightly
# task (#81) produces is a JSON file under _pipeline/reflection, and the failure
# this guards is the check existing but its verdict never reaching that file.

def _run_main(monkeypatch, tmp_path, template: Path, live: Path) -> dict:
    """Run main() end to end over fixture configs, with the index side stubbed.

    `inspect_index`/`pending_embeddings` are forced to "nothing to do" so no
    test touches ~/.cache/qmd or the daemon, and REPORT_DIR points at tmp_path.

    That last promise is why `daemon_healthy` is stubbed here. "Nothing to do" is
    the branch #958 added the health probe to, so an unstubbed probe would have
    every case in this file curling localhost:8181 — green only on a box with the
    daemon up, and 30 s of retries on one without it. What the probe is *for* is
    pinned in tests/test_qmd_maintenance_health.py; here it is an unrelated
    collaborator, forced healthy so the `rc == 0` below keeps testing the thing it
    says it tests: drift never changes the exit code.
    """
    monkeypatch.setattr(m, "TEMPLATE_CONFIG", template)
    monkeypatch.setattr(m, "LIVE_CONFIG", live)
    monkeypatch.setattr(m, "REPORT_DIR", tmp_path / "reflection")
    monkeypatch.setattr(m, "inspect_index", lambda: {"orphan_ratio": 0.0, "vectors_orphaned": 0})
    monkeypatch.setattr(m, "pending_embeddings", lambda: 0)
    monkeypatch.setattr(m, "daemon_healthy", lambda retries=10: True)
    monkeypatch.setattr(sys, "argv", ["qmd_index_maintenance.py"])
    rc = m.main()
    assert rc == 0, "drift must not change the job's exit code"
    reports = sorted((tmp_path / "reflection").glob("qmd-index-maintenance-*.json"))
    assert len(reports) == 1, f"expected exactly one dated report, got {reports}"
    return json.loads(reports[0].read_text())


def test_the_nightly_report_carries_the_drift_verdict_and_the_command(monkeypatch, tmp_path):
    t, l = _pair(tmp_path)
    report = _run_main(monkeypatch, tmp_path, t, l)
    cd = report["config_drift"]
    assert set(_by_name(cd["drift"])) == {"facts", "sessions", "autonomy-runs"}
    assert cd["drift_count"] == 3
    assert cd["resync_command"] == m.RESYNC_COMMAND


def test_the_report_lands_even_on_a_run_with_nothing_to_do(monkeypatch, tmp_path):
    """Written only on a mutating run before #1298 — 14 files for 45 nights.

    A nightly check whose file appears a handful of days a month is a check
    nobody can read on the day they need it.
    """
    t, l = _pair(tmp_path)
    report = _run_main(monkeypatch, tmp_path, t, l)
    assert report["need_prune"] is False and report["need_embed"] is False
    assert report["config_drift"]["drift_count"] == 3


def test_no_live_config_still_writes_a_report_and_exits_zero(monkeypatch, tmp_path):
    t, l = _pair(tmp_path, live=None)
    report = _run_main(monkeypatch, tmp_path, t, l)
    assert report["config_drift"]["drift_count"] == 0
    assert "no qmd config" in report["config_drift"]["note"]


def test_dry_run_prints_the_drift_and_writes_no_file(monkeypatch, tmp_path, capsys):
    t, l = _pair(tmp_path)
    monkeypatch.setattr(m, "TEMPLATE_CONFIG", t)
    monkeypatch.setattr(m, "LIVE_CONFIG", l)
    monkeypatch.setattr(m, "REPORT_DIR", tmp_path / "reflection")
    monkeypatch.setattr(m, "inspect_index", lambda: {"orphan_ratio": 0.0, "vectors_orphaned": 0})
    monkeypatch.setattr(m, "pending_embeddings", lambda: 0)
    # #958 put a health probe on the branch --dry-run shares with the no-op case, so
    # the rc below now follows that probe. Stubbed healthy, because this case is
    # about drift being printed and no file being written; the shared-branch choice
    # and the dry run's own health line are pinned in test_qmd_maintenance_health.py.
    monkeypatch.setattr(m, "daemon_healthy", lambda retries=10: True)
    monkeypatch.setattr(sys, "argv", ["qmd_index_maintenance.py", "--dry-run"])
    assert m.main() == 0
    assert not (tmp_path / "reflection").exists(), "--dry-run promises to change nothing"
    printed = capsys.readouterr().out
    assert "sessions" in printed
    assert "cp ~/.config/qmd/index.yml agent-services/conf/qmd-index.yml" in printed


# --- #1367: the unattended backfill refuses a corpus-wide re-embed -----------
# Pending is counted *per configured model*: `getHashesNeedingEmbedding`
# (qmd/src/store.ts:2561-2580) LEFT-JOINs on `model` + `embed_fingerprint`, not
# on the content hash. Pointing `models: embed` in ~/.config/qmd/index.yml at a
# different same-dimension model therefore makes every hash in the index read as
# unembedded — measured read-only on the live index at triage: 10,798 pending
# against 16,127 `documents` rows, i.e. 0.67 of the index and every active
# distinct hash in it. The job used to turn that number straight into
# `qmd embed --db ~/.cache/qmd/index.sqlite` with a 5400 s timeout, once a day,
# unattended, beside the serving daemon (stopping the daemon is forbidden by
# test_qmd_single_build.py). An interrupted run of that leaves two models'
# vectors in `vectors_vec`, which is keyed `hash_seq` with no model column and
# only catches a *dimension* change, so cosine search cannot tell them apart.
# Ordinary pending is 0, 3 or 2 in the three most recent reports — three orders
# of magnitude below the corpus-wide figure — which is why a fraction of the
# index size is the right shape for a cap and why the cap must not be tuned down
# toward the ordinary case.

#: Pending the moment `models: embed` names a different model: every active
#: distinct hash in the live index (triage 2026-09-22; re-measure before citing).
CORPUS_WIDE_PENDING = 10_798
#: `documents` rows behind those hashes — the denominator `inspect_index()`
#: already reports, so the guard's ratio is auditable from the same report.
LIVE_DOCUMENT_ROWS = 16_127
#: The embed model the live config names today.
EMBED_MODEL = "hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf"
#: A live config in the shape ~/.config/qmd/index.yml has it, `models:` included:
#: editing that one line is the trigger the guard exists to survive.
LIVE_WITH_MODELS = LIVE + f"models:\n  embed: {EMBED_MODEL}\n"


class _StubSh:
    """Records every subprocess command `main()` would have run.

    The claim under test is "no `qmd embed` subprocess was invoked", so the
    assertion is on the command line, not on a flag in the report. Returning
    rc 0 keeps the daemon-health probe and the restart path inert-but-plausible.
    """

    def __init__(self):
        self.cmds: list[list[str]] = []

    def __call__(self, cmd, timeout, env=None):
        self.cmds.append([str(c) for c in cmd])
        return 0, "stubbed"

    def qmd(self, verb: str) -> list[list[str]]:
        return [c for c in self.cmds if str(c[0]).endswith("node") and verb in c]


def _guard_run(monkeypatch, tmp_path, *, pending: int, documents: int,
               live: str | None = LIVE_WITH_MODELS):
    """Run main() over fixture configs with the index and every subprocess stubbed.

    Returns (rc, report, calls). Nothing here touches ~/.cache/qmd, the daemon or
    the real config: `inspect_index`/`pending_embeddings` are forced to the
    numbers under test and `_sh` is replaced by a recorder.
    """
    t, l = _pair(tmp_path, live=live)
    monkeypatch.setattr(m, "TEMPLATE_CONFIG", t)
    monkeypatch.setattr(m, "LIVE_CONFIG", l)
    monkeypatch.setattr(m, "REPORT_DIR", tmp_path / "reflection")
    monkeypatch.setattr(m, "inspect_index", lambda: {
        "index_bytes": 901_943_360, "chunks": 44_430, "documents": documents,
        "collections": 9, "collection_errors": 0, "files_skipped_missing": 0,
        "vectors_total": 38_665, "vectors_orphaned": 0,
        "vectors_live": 38_665, "orphan_ratio": 0.0,
    })
    monkeypatch.setattr(m, "pending_embeddings", lambda: pending)
    calls = _StubSh()
    monkeypatch.setattr(m, "_sh", calls)
    # #1545 put a bounded re-read of `pending_embeddings` after the mutating
    # section, and that loop sleeps between tries. The wait is not what any case
    # here is about — same reasoning, and the same patch, as
    # test_qmd_maintenance_health.py:277 — so the real sleep is replaced by a
    # no-op rather than slowing every mutating run in this file.
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    monkeypatch.setattr(sys, "argv", ["qmd_index_maintenance.py"])
    rc = m.main()
    reports = sorted((tmp_path / "reflection").glob("qmd-index-maintenance-*.json"))
    assert len(reports) == 1, f"expected exactly one dated report, got {reports}"
    return rc, json.loads(reports[0].read_text()), calls


def test_the_cap_is_one_named_constant_and_it_is_a_quarter():
    """One knob, worth a quarter of the index — not a scattering of literals.

    The threshold is what a later operator has to reason about, so it is a
    single module constant and it is pinned here rather than being inferable
    from a test that trips it.
    """
    assert m.EMBED_PENDING_MAX_RATIO == 0.25


def test_a_corpus_wide_pending_count_invokes_no_embed_subprocess(monkeypatch, tmp_path, capsys):
    """Clause 1: 10,798 pending against a 16,127-row index is every hash in it.

    The job runs nightly at hour 5 with nobody reading the number first, so the
    only thing standing between that config edit and an hour-long rewrite of the
    live index beside the serving daemon is this comparison.
    """
    rc, report, calls = _guard_run(monkeypatch, tmp_path,
                                   pending=CORPUS_WIDE_PENDING,
                                   documents=LIVE_DOCUMENT_ROWS)
    assert rc == 0, "a refused guard is not a failed job"
    assert calls.qmd("embed") == [], f"the guard let a corpus-wide embed through: {calls.cmds}"
    assert report["need_embed"] is False
    assert report["pending_embeddings"] == CORPUS_WIDE_PENDING
    # Pending counts distinct hashes and `documents` counts rows, so the printed
    # verdict has to carry the base or the ratio cannot be checked afterwards.
    printed = capsys.readouterr().out
    assert "documents" in printed and "25%" in printed, printed


def test_the_refused_run_reports_why_and_still_lands_its_report(monkeypatch, tmp_path):
    """Clause 2: the refusal is the useful output, so it has to be readable.

    The report names the pending count, the denominator the ratio used —
    `documents` rows, 16,127, not the 10,798 distinct hashes, which are the two
    numbers most likely to be confused by whoever reads this at 5 a.m. — and the
    configured embed model, which is the line to go check in the config.
    """
    rc, report, _ = _guard_run(monkeypatch, tmp_path,
                               pending=CORPUS_WIDE_PENDING,
                               documents=LIVE_DOCUMENT_ROWS)
    assert rc == 0
    mc = report["model_change_suspected"]
    assert mc["pending_embeddings"] == CORPUS_WIDE_PENDING
    assert "documents" in mc["denominator"]
    assert mc["denominator_value"] == LIVE_DOCUMENT_ROWS
    assert mc["configured_embed_model"] == EMBED_MODEL
    assert mc["threshold_fraction"] == m.EMBED_PENDING_MAX_RATIO
    assert mc["max_pending"] == int(m.EMBED_PENDING_MAX_RATIO * LIVE_DOCUMENT_ROWS)
    actions = " | ".join(report["actions"])
    assert "model_change_suspected" in actions, report["actions"]
    assert not any("rc=" in a for a in report["actions"]), report["actions"]


def test_the_guard_trips_without_a_readable_model_name(monkeypatch, tmp_path):
    """The refuse decision cannot depend on reading the config.

    The model name is what the report points a human at, not what the guard
    compares; a missing `models:` block (or an unreadable config) must still
    stop the rewrite, with the name reported as unknown rather than the check
    skipped.
    """
    rc, report, calls = _guard_run(monkeypatch, tmp_path,
                                   pending=CORPUS_WIDE_PENDING,
                                   documents=LIVE_DOCUMENT_ROWS, live=LIVE)
    assert rc == 0
    assert calls.qmd("embed") == []
    assert report["model_change_suspected"]["configured_embed_model"] is None


def test_the_cap_bites_only_past_a_quarter_of_the_index(monkeypatch, tmp_path):
    """The boundary, both ways, so the guard cannot fail to fail.

    At the limit the backfill still runs; one hash past it is refused. A cap
    whose edge nobody pins is a cap that quietly became 0 or infinity.
    """
    limit = int(m.EMBED_PENDING_MAX_RATIO * LIVE_DOCUMENT_ROWS)
    _, at_limit, calls = _guard_run(monkeypatch, tmp_path,
                                    pending=limit, documents=LIVE_DOCUMENT_ROWS)
    assert at_limit["need_embed"] is True
    assert "model_change_suspected" not in at_limit
    assert len(calls.qmd("embed")) == 1
    _, over, calls2 = _guard_run(monkeypatch, tmp_path,
                                 pending=limit + 1, documents=LIVE_DOCUMENT_ROWS)
    assert over["need_embed"] is False
    assert "model_change_suspected" in over
    assert calls2.qmd("embed") == []


def test_an_ordinary_backfill_still_runs_exactly_one_embed(monkeypatch, tmp_path):
    """Clause 3: this guard exists to leave the 3-document case working.

    Pending is almost never zero while the loop is writing — the nightly run's
    whole job is clearing one to four documents — so a cap that also blocked
    those would take the index down to a permanently stale vector leg.
    """
    rc, report, calls = _guard_run(monkeypatch, tmp_path, pending=3, documents=16_000)
    assert rc == 0
    assert report["need_embed"] is True
    assert "model_change_suspected" not in report
    embeds = calls.qmd("embed")
    assert len(embeds) == 1, f"expected exactly one qmd embed, got {calls.cmds}"
    assert "embed" in embeds[0]


# --- #844: vec0 capacity and the whole-footprint size ------------------------
# The orphan ratio cannot see dead vec0 slots: sqlite-vec allocates fixed-size
# chunks and neither `qmd cleanup` nor VACUUM reclaims or reuses a dead slot. On
# 2026-09-18 an index at 7.2% orphans was 12.8% occupied with 766 MiB dead. The
# fixture below is a real SQLite file with sqlite-vec's shadow-table layout
# (plain tables, readable without the extension), never the live index.

def _vec0_index(path: Path, *, chunks: int, chunk_slots: int, live: int,
                vec_bytes: int = 16, orphaned: int = 0) -> Path:
    import sqlite3
    con = sqlite3.connect(path)
    con.executescript("""
        create table content(hash text primary key);
        create table content_vectors(hash text, seq int);
        create table documents(id integer primary key, hash text);
        create table vectors_vec_chunks(chunk_id integer primary key, size integer,
                                        validity blob, rowids blob);
        create table vectors_vec_rowids(rowid integer primary key, id text,
                                        chunk_id integer, chunk_offset integer);
        create table vectors_vec_vector_chunks00(rowid integer primary key, vectors blob);
    """)
    for c in range(chunks):
        con.execute("insert into vectors_vec_chunks values (?,?,?,?)",
                    (c + 1, chunk_slots, b"\0" * (chunk_slots // 8), b""))
        con.execute("insert into vectors_vec_vector_chunks00 values (?, zeroblob(?))",
                    (c + 1, chunk_slots * vec_bytes))
    for i in range(live):
        con.execute("insert into vectors_vec_rowids values (?,?,?,?)",
                    (i + 1, f"h{i}_0", i // chunk_slots + 1, i % chunk_slots))
        con.execute("insert into content values (?)", (f"h{i}",))
        con.execute("insert into content_vectors values (?, 0)", (f"h{i}",))
    for i in range(orphaned):
        con.execute("insert into content_vectors values (?, 0)", (f"gone{i}",))
    con.commit()
    con.close()
    return path


def test_inspect_index_reads_vec0_capacity_from_the_shadow_tables(monkeypatch, tmp_path, capsys):
    idx = _vec0_index(tmp_path / "index.sqlite", chunks=4, chunk_slots=64, live=32)
    monkeypatch.setattr(m, "INDEX", idx)
    v = m.inspect_index()["vec0"]
    assert v["chunks"] == 4 and v["allocated_slots"] == 256 and v["live_rows"] == 32
    assert v["occupancy"] == 0.125
    # 4 chunks x 64 slots x 16 B = 16 KiB allocated, 7/8 of it dead.
    assert v["dead_mib"] == round(4 * 64 * 16 * 7 / 8 / 2**20, 1)
    assert v["allocated_mib"] == round(4 * 64 * 16 / 2**20, 1)
    m._emit({"before": m.inspect_index(), "actions": [], "need_capacity": False}, False)
    assert "vec0 occupancy    12.5 %" in capsys.readouterr().out


def test_an_index_without_vec0_tables_reports_an_error_and_does_not_raise(monkeypatch, tmp_path):
    import sqlite3
    idx = tmp_path / "index.sqlite"
    con = sqlite3.connect(idx)
    con.executescript("create table content(hash text); create table content_vectors(hash text);"
                      "create table documents(id int);")
    con.close()
    monkeypatch.setattr(m, "INDEX", idx)
    out = m.inspect_index()
    assert "error" in out["vec0"] and "error" not in out
    assert m.capacity_verdict(out["vec0"]) is False


#: The 2026-09-18 index, as the triage measured it: 293 chunks of 1,024 slots at
#: 3,072 B, 38,454 live rows, 2,777 of 38,446 vectors orphaned.
SEPT_18_SHAPE = {
    "index_bytes": 1_188_237_312, "documents": 16_000,
    "vectors_total": 38_446, "vectors_orphaned": 2_777,
    "vectors_live": 35_669, "orphan_ratio": 0.0722,
    "vec0": {"chunks": 293, "allocated_slots": 300_032, "live_rows": 38_454,
             "occupancy": 0.1282, "allocated_mib": 879.0, "dead_mib": 766.3},
}


def test_the_capacity_verdict_fires_on_the_sept_18_shape_while_prune_does_not(
        monkeypatch, tmp_path):
    """Reachability, in the style of test_orphan_prune_is_reachable_on_ratio_alone:
    at the index shape that motivated #844 the orphan triggers say nothing to do,
    and the capacity verdict has to say otherwise on its own."""
    t, l = _pair(tmp_path)
    monkeypatch.setattr(m, "TEMPLATE_CONFIG", t)
    monkeypatch.setattr(m, "LIVE_CONFIG", l)
    monkeypatch.setattr(m, "REPORT_DIR", tmp_path / "reflection")
    monkeypatch.setattr(m, "inspect_index", lambda: dict(SEPT_18_SHAPE))
    monkeypatch.setattr(m, "pending_embeddings", lambda: 0)
    monkeypatch.setattr(m, "daemon_healthy", lambda retries=10: True)
    monkeypatch.setattr(sys, "argv", ["qmd_index_maintenance.py"])
    assert m.main() == 0
    report = json.loads(next((tmp_path / "reflection").glob("*.json")).read_text())
    assert report["need_prune"] is False
    assert report["need_capacity"] is True
    # Reported, never acted on: the no-op branch ran and says why rebuild is not its job.
    assert report["actions"] == ["none — nothing to do"]
    assert "ruled out" in report["vec0_rebuild"] and "side copy" in report["vec0_rebuild"]


def test_the_capacity_verdict_stays_quiet_after_the_sept_21_rebuild():
    # 56 chunks x 1,024 slots, 32,147 live: the side-copy rebuild's result.
    assert m.capacity_verdict({"occupancy": 0.561, "dead_mib": 98.4}) is False
    # A small index that is mostly empty is not worth a rebuild either.
    assert m.capacity_verdict({"occupancy": 0.05, "dead_mib": 10.0}) is False
    assert m.capacity_verdict({"occupancy": 0.20, "dead_mib": 300.0}) is True


def test_the_footprint_counts_the_wal_and_shm(monkeypatch, tmp_path):
    idx = _vec0_index(tmp_path / "index.sqlite", chunks=1, chunk_slots=8, live=2)
    main_size = idx.stat().st_size
    (tmp_path / "index.sqlite-wal").write_bytes(b"w" * 5000)
    (tmp_path / "index.sqlite-shm").write_bytes(b"s" * 300)
    monkeypatch.setattr(m, "INDEX", idx)
    out = m.inspect_index()
    assert out["index_bytes"] == main_size + 5300 > main_size
    assert out["footprint"] == {"total": main_size + 5300, "main": main_size,
                                "wal": 5000, "shm": 300}


# --- #1545: an embed that did no work must not be reported as one ------------
# The task #81 run of 2026-09-26 05:00Z wrote
# `_pipeline/reflection/qmd-index-maintenance-2026-09-26.json` with `actions:
# ["embed rc=0 in 0s"]`, `embed_ok: true`, `elapsed_s: 0.3`, `pending_embeddings:
# 4` — and an `after` block byte-identical to `before` (`vectors_total 33010`,
# `index_bytes 1105788304`), while the index itself had moved on (33,081 vectors
# by 12:18Z the same day, read-only). Two faults, both in the report and neither
# in the index:
#
#   * `qmd embed` exits **0 having done no work** whenever another live process
#     holds `~/.cache/qmd/.qmd-embed.lock` — which the watcher does most of the
#     day, since this job's own header says pending is never zero while the loop
#     is writing (`qmd/src/cli/qmd.ts:2173-2178` prints
#     `EMBED_LOCK_BUSY_MESSAGE` from `qmd/src/cli/embed-lock.ts:99-100` and
#     returns), and again when nothing is pending (`qmd.ts:2187-2191`). A skip is
#     this branch's *normal* outcome. The embed branch threw its stdout away —
#     unlike the cleanup branch beside it, which keeps its last line — so every
#     skip printed as `rc=0` and became `embed_ok: true`. The timing says no
#     embed happened: one standalone `qmd status` alone costs 0.209 s against the
#     whole run's 0.3 s, which did not leave the subprocess enough time to boot
#     node, load `Qwen3-Embedding-0.6B-Q8_0.gguf`, and embed 57 chunks.
#   * The report could not show pending moving at all. `pending_embeddings()` ran
#     exactly once, before any action, and `inspect_index()` returns no pending
#     key, so `after` could not contain one. "Identical before/after" and "there
#     was nothing to do" were the same sentence to whoever read it.
#
# Every case here runs with `_sh`, `inspect_index`, `pending_embeddings` and
# `time.sleep` stubbed: no node process, no live index, no daemon, no wait.

#: What `qmd embed` prints when the watcher holds the embed lock, verbatim from
#: `EMBED_LOCK_BUSY_MESSAGE` (`qmd/src/cli/embed-lock.ts:99-100`), rc 0.
EMBED_LOCK_BUSY = "Another embed process is already running. Skipping.\n"
#: The other silent rc-0 no-work path (`qmd/src/cli/qmd.ts:2187-2191`).
EMBED_ALREADY_DONE = "✓ All content hashes already have embeddings.\n"
#: And the third, when the pending hashes turn out to have no text
#: (`qmd.ts:2240`): rc 0, nothing written.
EMBED_NO_DOCS = "✓ No non-empty documents to embed.\n"
#: What an embed that actually embedded looks like (`qmd.ts:2244`).
EMBED_LANDED = "✓ Done! Embedded 57 chunks from 4 documents in 0:12\n"

#: The 2026-09-26 05:00Z `before` block, field for field, as the dated report
#: recorded it. `documents` is what keeps the #1367 guard's ratio at 0.0003, and
#: `orphan_ratio` is far under both prune triggers, so every case below reaches
#: the mutating section on the embed leg alone.
BEFORE_0926 = {
    "index_bytes": 1_105_788_304,
    "footprint": {"total": 1_105_788_304, "main": 550_739_968,
                  "wal": 553_966_992, "shm": 1_081_344},
    "vectors_total": 33_010, "vectors_orphaned": 52, "documents": 11_942,
    "vec0": {"chunks": 69, "allocated_slots": 70_656, "live_rows": 33_010,
             "occupancy": 0.4672, "allocated_mib": 276.0, "dead_mib": 147.1},
    "vectors_live": 32_958, "orphan_ratio": 0.0016,
}
#: What the same index looked like with the 57 chunks in it: +57 live vectors,
#: the WAL checkpointed down to the main file, and the orphans still there. Only
#: the vector count and the file size need to move for `after` to differ.
AFTER_0926_LANDED = {
    **BEFORE_0926,
    "index_bytes": 1_105_112_704,
    "footprint": {"total": 1_105_112_704, "main": 1_105_112_704,
                  "wal": 0, "shm": 0},
    "vectors_total": 33_067, "vectors_live": 33_015,
    "vec0": {**BEFORE_0926["vec0"], "live_rows": 33_067},
}

#: `pending embeds   4  →  0` — the pre-run and post-run figures on one line.
PENDING_LINE = re.compile(r"pending embeds\s+(\d+)\s+→\s+(\d+)")


class _EmbedSh:
    """Answers the subprocesses a mutating run issues, with `qmd embed` configured.

    The claim under test is what the report says about the *embed subprocess's
    own output*, so the stub is what puts that output there: `embed` answers
    `embed_rc`/`embed_out`, everything else (the health probe's curl) answers rc 0
    so the daemon-health path stays inert-but-plausible.
    """

    def __init__(self, embed_out: str, embed_rc: int = 0):
        self.embed_out, self.embed_rc = embed_out, embed_rc
        self.cmds: list[list[str]] = []

    def __call__(self, cmd, timeout, env=None):
        c = [str(x) for x in cmd]
        self.cmds.append(c)
        if c[0].endswith("node") and "embed" in c:
            return self.embed_rc, self.embed_out
        return 0, "stubbed"

    def qmd(self, verb: str) -> list[list[str]]:
        return [c for c in self.cmds if c[0].endswith("node") and verb in c]


def _embed_run(monkeypatch, tmp_path, *, embed_out: str, embed_rc: int = 0,
               pending: list[int], snapshots: list[dict]):
    """Run `main()` through the mutating section with nothing real behind it.

    `pending` is the sequence `pending_embeddings()` hands back — its first value
    is the pre-run count, the rest are the post-action re-reads — and `snapshots`
    is the sequence `inspect_index()` hands back (`before`, then `after`). Both
    raise rather than repeat if the run asks more times than the case supplies,
    because "how many times did it re-read" *is* one of the claims.

    Returns `(rc, report, calls, slept)`: the report re-read off the dated file
    the run wrote (not from memory, so a key that never reaches JSON fails), and
    the seconds `time.sleep` was asked to wait, which pins the re-read's waits
    without paying them.
    """
    t, l = _pair(tmp_path, live=LIVE_WITH_MODELS)
    monkeypatch.setattr(m, "TEMPLATE_CONFIG", t)
    monkeypatch.setattr(m, "LIVE_CONFIG", l)
    monkeypatch.setattr(m, "REPORT_DIR", tmp_path / "reflection")

    shapes, pends, slept = list(snapshots), list(pending), []

    def next_snapshot():
        assert shapes, "inspect_index() was called more times than this case supplies"
        return dict(shapes.pop(0))

    def next_pending():
        assert pends, (
            "pending_embeddings() was called more times than this case supplies "
            "tries for — the re-read count is itself one of the claims")
        return pends.pop(0)

    monkeypatch.setattr(m, "inspect_index", next_snapshot)
    monkeypatch.setattr(m, "pending_embeddings", next_pending)
    calls = _EmbedSh(embed_out, embed_rc)
    monkeypatch.setattr(m, "_sh", calls)
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(sys, "argv", ["qmd_index_maintenance.py"])
    rc = m.main()
    reports = sorted((tmp_path / "reflection").glob("qmd-index-maintenance-*.json"))
    assert len(reports) == 1, f"expected exactly one dated report, got {reports}"
    return rc, json.loads(reports[0].read_text()), calls, slept


# --- clause 1: the embed subprocess's own words reach the dated JSON ---------

def test_the_embed_action_keeps_the_subprocess_s_own_last_line(monkeypatch, tmp_path):
    """The sentence that explains the rc 0 has to be on disk, not discarded.

    The cleanup branch beside it already keeps its last output line
    (`cleanup rc=… :: …`); the embed branch did not, which is why a run whose
    embed said "Another embed process is already running. Skipping." left a
    reader with only `embed rc=0 in 0s` to go on.
    """
    rc, report, calls, _ = _embed_run(
        monkeypatch, tmp_path, embed_out=EMBED_LOCK_BUSY,
        pending=[4, 4, 4, 4], snapshots=[BEFORE_0926, dict(BEFORE_0926)])
    assert len(calls.qmd("embed")) == 1
    embed_actions = [a for a in report["actions"] if a.startswith("embed rc=")]
    assert len(embed_actions) == 1, report["actions"]
    assert "Another embed process is already running. Skipping." in embed_actions[0], (
        embed_actions[0])


# --- clause 2: rc 0 alone is not a successful embed --------------------------

@pytest.mark.parametrize("embed_out,ok", [
    (EMBED_LOCK_BUSY, False),      # the lock skip: the nightly normal case
    (EMBED_ALREADY_DONE, False),   # nothing pending after all
    (EMBED_NO_DOCS, False),        # pending hashes with no text to embed
    (EMBED_LANDED, True),          # work done: must still read as a pass
])
def test_embed_ok_reports_whether_the_subprocess_did_work_not_just_its_exit_code(
        monkeypatch, tmp_path, embed_out, ok):
    """`embed_ok = rc == 0` was a false claim on every skipped embed.

    Three of `qmd embed`'s exits are rc 0 with nothing written, so the exit code
    cannot answer "did the vectors arrive" — and the landed case has to stay
    True, or the fix is just an alarm.
    """
    _, report, _, _ = _embed_run(
        monkeypatch, tmp_path, embed_out=embed_out,
        pending=[4, 0, 0, 0], snapshots=[BEFORE_0926, dict(AFTER_0926_LANDED)])
    assert report["embed_ok"] is ok, (embed_out.strip(), report["embed_ok"])


def test_a_non_zero_embed_still_records_embed_ok_false(monkeypatch, tmp_path):
    """The old path's one honest case must not regress: rc != 0 is False."""
    _, report, _, _ = _embed_run(
        monkeypatch, tmp_path, embed_out="some node crash\n", embed_rc=1,
        pending=[4, 4, 4, 4], snapshots=[BEFORE_0926, dict(BEFORE_0926)])
    assert report["embed_ok"] is False


# --- clause 3: the after block measures pending after the actions ------------

def test_the_after_block_carries_a_pending_count_measured_after_the_actions(
        monkeypatch, tmp_path, capsys):
    """Before this, `after` had no pending key at all: the report could not show
    the number the run was launched to move, because the only read sat before any
    action. Here the first post-action read still sees 4 (the watcher is writing
    under the same lock) and the second sees 0.
    """
    _, report, _, slept = _embed_run(
        monkeypatch, tmp_path, embed_out=EMBED_LANDED,
        pending=[4, 4, 0], snapshots=[BEFORE_0926, dict(AFTER_0926_LANDED)])
    assert report["pending_embeddings"] == 4, "the pre-run figure stays where it was"
    assert report["after"]["pending_embeddings"] == 0
    assert slept == [m.AFTER_PENDING_SLEEP_S], (
        f"the still-nonzero first re-read should wait once, got {slept}")
    hit = PENDING_LINE.search(capsys.readouterr().out)
    assert hit, "the emitted run must print both pending figures on one line"
    assert hit.groups() == ("4", "0"), hit.groups()


def test_the_post_action_pending_re_read_is_bounded_and_reports_what_it_saw(
        monkeypatch, tmp_path):
    """A retry with no ceiling is a nightly job that hangs on a wedged daemon.

    The embed was skipped here, so pending never moves: the run spends
    `AFTER_PENDING_RETRIES` post-action reads, waits between them rather than in
    a busy loop, and reports the residual 3 rather than the pre-run 4 or a zero it
    never measured.
    """
    _, report, _, slept = _embed_run(
        monkeypatch, tmp_path, embed_out=EMBED_LOCK_BUSY,
        pending=[4, 3, 3, 3], snapshots=[BEFORE_0926, dict(BEFORE_0926)])
    assert report["after"]["pending_embeddings"] == 3
    assert len(slept) == m.AFTER_PENDING_RETRIES - 1, slept
    assert all(s == m.AFTER_PENDING_SLEEP_S for s in slept), slept


# --- clause 4: an identical pair never goes unexplained ----------------------

def test_an_embed_run_whose_after_equals_before_says_the_embed_did_not_land(
        monkeypatch, tmp_path, capsys):
    """The 2026-09-26 shape exactly: an embed asked for, an `after` block
    indistinguishable from `before`, and a report that read as "nothing
    changed" — the one sentence that must not be sayable.
    """
    _, report, _, _ = _embed_run(
        monkeypatch, tmp_path, embed_out=EMBED_LOCK_BUSY,
        pending=[4, 4, 4, 4], snapshots=[BEFORE_0926, dict(BEFORE_0926)])
    assert report["after"]["pending_embeddings"] == 4
    verdict = report["embed_did_not_land"]
    assert verdict["after_equals_before"] is True
    assert verdict["embed_ok"] is False
    assert verdict["residual_pending_embeddings"] == 4
    printed = capsys.readouterr().out
    line = [l for l in printed.splitlines() if "did not land" in l]
    assert line, printed
    assert "4" in line[0], "the verdict has to carry the pending still outstanding"


def test_an_embed_that_landed_carries_no_not_landed_verdict(monkeypatch, tmp_path,
                                                            capsys):
    """The verdict must be able to stay silent, or it is noise and gets ignored.

    Same run shape with the embed actually embedding: `after` differs from
    `before`, `embed_ok` is True, pending went 4 → 0, and nothing on disk or on
    stdout claims the embed failed.
    """
    _, report, _, _ = _embed_run(
        monkeypatch, tmp_path, embed_out=EMBED_LANDED,
        pending=[4, 0], snapshots=[BEFORE_0926, dict(AFTER_0926_LANDED)])
    assert report["embed_ok"] is True
    assert report["after"]["pending_embeddings"] == 0
    assert report["after"] != report["before"]
    assert "embed_did_not_land" not in report, report
    assert "did not land" not in capsys.readouterr().out
