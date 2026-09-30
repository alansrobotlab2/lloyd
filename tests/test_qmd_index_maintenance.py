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
import os
import re
import sys
import time
from datetime import datetime
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
    # #1598 put an acting delete into main()'s no-work path, so a run over an
    # unpatched INDEX would unlink real files out of ~/.cache/qmd — the directory
    # this job owns on a production box. Pointed at a fixture that does not exist,
    # which is the measured-empty case: no strays, nothing to delete.
    monkeypatch.setattr(m, "INDEX", tmp_path / "qmd" / "index.sqlite")
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
    # #1598: main() now deletes in the index directory, so even a dry run points at
    # a fixture path here — the case asserts no file is written, and it must not be
    # the case that decides whether the real ~/.cache/qmd survives the suite.
    monkeypatch.setattr(m, "INDEX", tmp_path / "qmd" / "index.sqlite")
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
    # #1598: main() now deletes files from the index directory on an acting run, so
    # the path it reads has to be a fixture's. "Nothing here touches ~/.cache/qmd"
    # in this docstring is now enforced by this line and not by the stubs alone.
    monkeypatch.setattr(m, "INDEX", tmp_path / "qmd" / "index.sqlite")
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
    and the capacity verdict has to say otherwise on its own.

    Two assertions here stated the PRE-#1897 behaviour and are updated, not deleted:
    `m.main() == 0` and `actions == ["none — nothing to do"]`. Both held only
    because a fired verdict rode a green run and the run called itself idle; #1897
    makes it exit non-zero and stops calling a run that owes a rebuild idle. The
    reachability claim this node was written for is unchanged, and the exit code it
    now asserts is the point of the change — see
    test_a_fired_capacity_verdict_exits_non_zero for the 2026-09-30 shape.
    """
    t, l = _pair(tmp_path)
    monkeypatch.setattr(m, "TEMPLATE_CONFIG", t)
    monkeypatch.setattr(m, "LIVE_CONFIG", l)
    monkeypatch.setattr(m, "REPORT_DIR", tmp_path / "reflection")
    # #1598: an empty fixture pile, so the `actions` assertion below is about the
    # job's own verdict and not about files it measured next to the live index.
    monkeypatch.setattr(m, "INDEX", tmp_path / "qmd" / "index.sqlite")
    monkeypatch.setattr(m, "inspect_index", lambda: dict(SEPT_18_SHAPE))
    monkeypatch.setattr(m, "pending_embeddings", lambda: 0)
    monkeypatch.setattr(m, "daemon_healthy", lambda retries=10: True)
    monkeypatch.setattr(sys, "argv", ["qmd_index_maintenance.py"])
    # #1897: the fired verdict is what makes the run red. Before it this line read
    # `== 0`, which is the whole defect — the shape that motivated #844 was green
    # every night it was live.
    assert m.main() == 1
    report = json.loads(next((tmp_path / "reflection").glob("*.json")).read_text())
    assert report["need_prune"] is False
    assert report["need_capacity"] is True
    # Never acted on, and no longer called idle either: the mutating section did not
    # run, so `actions` is empty rather than the "nothing to do" line — work IS owed,
    # it is just a person's to do. SEPT_18_SHAPE carries no `footprint` key, which is
    # also the case this change has to survive: the owed entry records None, and the
    # summary says `footprint not measured` instead of printing a made-up 0 B.
    assert report["actions"] == []
    assert report["capacity_owed_to_a_person"]["footprint_bytes"] is None
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
    # #1598: this helper runs main() all the way through the mutating section, and
    # that path now unlinks files in the index directory. "Nothing real behind it"
    # in this docstring holds only because of this line.
    monkeypatch.setattr(m, "INDEX", tmp_path / "qmd" / "index.sqlite")

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


# --- #1598: the stray pile beside the live index ------------------------------
#
# `du -sh ~/.cache/qmd/*` measured on 2026-09-27: four non-live databases — the
# 09-07 `index.backup-*.sqlite` (998 MB), two embedding-model-switch copies from
# 09-19 and 09-21 (1.25 GB and 1.27 GB), and `perfbench.sqlite` (1.24 GB) — whose
# main files sum to 4,863,918,080 B against a live `index.sqlite` of 582,688,768 B.
# Their `-wal`/`-shm` sidecars add 65,536 B more, so the directory's measured stray
# total is 4,863,983,616 B. The nightly job that owns that directory printed
# `need_prune False` and exit 0 with no key for any of them: the pile was invisible
# to the run's verdict, so each run re-derived nobody's number. #844 had closed on
# 2026-09-24 carrying #855's "decide their retention in the same change" unexecuted
# — a closed item is not an owner.
#
# Every case here builds its own index directory under `tmp_path` and points `INDEX`
# at it. Nothing in this section may run the delete path against the real
# ~/.cache/qmd: four real backups are in there, and two of them are a person's call.
#
# And no concrete `index.sqlite.bak-<date>` name of a file that really exists is
# spelled out anywhere in this file either. The code-reference grep the rule re-runs
# walks `tests/**.py` too, so a comment naming one would make the run report *this
# comment* as its reader — which the first live dry run of this change did, until the
# names were reworded out.

def _qmd_dir(tmp_path: Path, spec: list[tuple[str, int, float]]) -> Path:
    """Build a fixture index directory: (name, bytes, days ago as mtime)."""
    d = tmp_path / "qmd"
    d.mkdir(parents=True, exist_ok=True)
    # One clock reading for the whole fixture: two files given the same `days_ago`
    # must come out with *equal* mtimes, the way a `cp -a` leaves a database and its
    # WAL. Reading the clock per file would make each sidecar a few microseconds
    # newer than the main file beside it, and the hold-under-test would fire on the
    # clock rather than on the fact.
    now = time.time()
    for name, size, days_ago in spec:
        p = d / name
        p.write_bytes(b"\0" * size)
        ts = now - days_ago * 86400
        os.utime(p, (ts, ts))
    return d


def _listing(d: Path) -> dict[str, int]:
    """Every regular file in `d`, as name -> bytes."""
    return {p.name: p.stat().st_size for p in sorted(d.iterdir()) if p.is_file()}


#: The live trio, present in every fixture below: strays are what is beside them.
LIVE_TRIO = [("index.sqlite", 4000, 0.0), ("index.sqlite-wal", 3000, 0.0),
             ("index.sqlite-shm", 200, 0.0)]


def _retention_run(monkeypatch, tmp_path, index_dir: Path, repo_root: Path, *,
                   dry_run: bool, capsys=None) -> dict:
    """Run `main()` over a fixture index directory and a fixture code tree.

    The index and the daemon are stubbed to "nothing to do" the way the helpers
    above stub them, so the only thing this run can *do* is the retention pass, and
    the only files it can do it to are the ones the case built. `repo_root` is what
    the code-reference re-run walks in place of the real tree.

    An acting run's report is read back off the dated file, the way the helpers
    above do. A dry run's comes off `--json` stdout instead, because `--dry-run`
    writes no file at all — and "the file is not the proof" is the point: the dry
    run still has to say what it would have deleted, and it does so on the only
    surface it has.
    """
    t, l = _pair(tmp_path)
    monkeypatch.setattr(m, "TEMPLATE_CONFIG", t)
    monkeypatch.setattr(m, "LIVE_CONFIG", l)
    monkeypatch.setattr(m, "REPORT_DIR", tmp_path / "reflection")
    monkeypatch.setattr(m, "INDEX", index_dir / "index.sqlite")
    monkeypatch.setattr(m, "REPO_ROOT", repo_root)
    monkeypatch.setattr(m, "inspect_index", lambda: {
        "orphan_ratio": 0.0, "vectors_orphaned": 0})
    monkeypatch.setattr(m, "pending_embeddings", lambda: 0)
    monkeypatch.setattr(m, "daemon_healthy", lambda retries=10: True)
    argv = ["qmd_index_maintenance.py"]
    if dry_run:
        assert capsys is not None, "a dry run has no dated file, so it needs --json"
        argv += ["--dry-run", "--json"]
    monkeypatch.setattr(sys, "argv", argv)
    rc = m.main()
    assert rc == 0, "measuring or retaining a backup never changes the exit code"
    if dry_run:
        return json.loads(capsys.readouterr().out)
    reports = sorted((tmp_path / "reflection").glob("qmd-index-maintenance-*.json"))
    assert len(reports) == 1, f"expected exactly one dated report, got {reports}"
    return json.loads(reports[0].read_text())


# --- clause 1: the dated report measures the pile -----------------------------

def test_the_report_measures_every_non_live_file_with_bytes_and_mtime(monkeypatch,
                                                                      tmp_path):
    """Four 2026-09-27 strays, one named in #855 and still there 20 days later.

    The claim is per-file — name, bytes, mtime — and the total, because "the pile
    grew" and "the pile is the same pile" are different statements a run cannot
    make from `index_bytes`, which is main+wal+shm of the live index alone.
    """
    d = _qmd_dir(tmp_path, LIVE_TRIO + [
        ("index.backup-20260907_100044.sqlite", 998, 20.0),
        ("perfbench.sqlite", 1024, 8.0),
    ])
    (d / "models").mkdir()
    (d / "models" / "big.onnx").write_bytes(b"x" * 5000)
    # An acting run, because clause 1 is about the dated artifact: `--dry-run`
    # writes no file at all, so the key has to be proven to reach the JSON on disk.
    report = _retention_run(monkeypatch, tmp_path, d, tmp_path / "repo", dry_run=False)
    assert [f["name"] for f in report["stray"]] == [
        "index.backup-20260907_100044.sqlite", "perfbench.sqlite"], (
        "the live trio is not a stray, and neither is anything inside a directory")
    assert {f["bytes"] for f in report["stray"]} == {998, 1024}
    assert report["stray_bytes"] == 998 + 1024
    by_name = {f["name"]: f for f in report["stray"]}
    for name, days in (("index.backup-20260907_100044.sqlite", 20.0),
                       ("perfbench.sqlite", 8.0)):
        age_days = (datetime.now() - datetime.fromisoformat(by_name[name]["mtime"])).days
        assert age_days == pytest.approx(days, abs=1), (name, by_name[name]["mtime"])


def test_a_directory_holding_only_the_live_trio_records_zero_stray_bytes(
        monkeypatch, tmp_path):
    """An empty pile is a measurement of 0, not an omitted key.

    A key that appears only when something is wrong is the absent-input default the
    catalogued class is about: a reader who cannot tell "measured nothing" from
    "did not look" inherits no number from it either.
    """
    d = _qmd_dir(tmp_path, LIVE_TRIO)
    report = _retention_run(monkeypatch, tmp_path, d, tmp_path / "repo", dry_run=False)
    assert report["stray"] == []
    assert "stray_bytes" in report, "0 must be recorded, not left out"
    assert report["stray_bytes"] == 0


# --- clause 2: the newest backup is kept, an older one is a candidate ---------

def test_an_acting_run_keeps_the_newest_backup_by_mtime_and_deletes_the_older_ones(
        monkeypatch, tmp_path):
    """One kept copy, and the mtime decides which — the name's date does not.

    `bak-a` is the newest file on disk and `bak-z` an older one, so a rule that
    sorted by name would keep the wrong database. Each candidate goes with its own
    `-wal`/`-shm` sidecars: a main file deleted and its WAL left behind is a 0-byte
    orphan nobody can read.
    """
    d = _qmd_dir(tmp_path, LIVE_TRIO + [
        ("index.sqlite.bak-a", 3000, 1.0),
        ("index.sqlite.bak-a-wal", 30, 1.0), ("index.sqlite.bak-a-shm", 32, 1.0),
        ("index.sqlite.bak-z", 2000, 5.0),
        ("index.sqlite.bak-z-wal", 20, 5.0), ("index.sqlite.bak-z-shm", 32, 5.0),
        ("index.sqlite.bak-m", 1000, 9.0), ("index.sqlite.bak-m-wal", 10, 9.0),
    ])
    report = _retention_run(monkeypatch, tmp_path, d, tmp_path / "repo", dry_run=False)
    after = _listing(d)
    assert report["stray_retention"]["kept"] == ["index.sqlite.bak-a"]
    assert after["index.sqlite.bak-a"] == 3000
    assert "index.sqlite.bak-a-wal" in after and "index.sqlite.bak-a-shm" in after
    assert sorted(report["stray_retention"]["deleted"]) == [
        "index.sqlite.bak-m", "index.sqlite.bak-z"]
    for gone in ("index.sqlite.bak-z", "index.sqlite.bak-z-wal", "index.sqlite.bak-z-shm",
                 "index.sqlite.bak-m", "index.sqlite.bak-m-wal"):
        assert gone not in after, gone
    assert report["stray_retention"]["deleted_bytes"] == 2000 + 20 + 32 + 1000 + 10
    assert after["index.sqlite"] == 4000 and after["index.sqlite-wal"] == 3000, (
        "the live trio is never this job's to touch")
    assert len([n for n in after if n.startswith("index.sqlite.bak")
                and not n.endswith(("-wal", "-shm"))]) == 1, (
        "an acting run leaves at most one .bak main file on disk")


def test_a_lone_backup_is_not_a_delete_candidate_because_no_newer_one_exists(
        monkeypatch, tmp_path):
    """The rule is "keep the newest", not "keep one and free the rest".

    With a single backup there is nothing newer to fall back on, so deleting it
    would leave the live index with no copy at all.
    """
    d = _qmd_dir(tmp_path, LIVE_TRIO + [
        ("index.sqlite.bak-only", 1500, 30.0), ("index.sqlite.bak-only-wal", 15, 30.0)])
    report = _retention_run(monkeypatch, tmp_path, d, tmp_path / "repo", dry_run=False)
    assert report["stray_retention"]["kept"] == ["index.sqlite.bak-only"]
    assert report["stray_retention"]["planned"] == []
    assert report["stray_retention"]["deleted"] == []
    assert "index.sqlite.bak-only" in _listing(d)


def test_a_backup_s_sidecar_is_never_its_own_member_of_the_series(tmp_path):
    """`…-wal` matches the `index.sqlite.bak*` prefix as a string and is not a copy.

    Treating it as a member would let a 0-byte WAL be "the newest backup" and put
    the database it belongs to on the delete list.
    """
    d = _qmd_dir(tmp_path, [("index.sqlite", 4000, 0.0),
                            ("index.sqlite.bak-a", 3000, 2.0),
                            ("index.sqlite.bak-a-wal", 30, 1.0)])
    assert m.bak_series(d / "index.sqlite") == [d / "index.sqlite.bak-a"]


def test_a_run_that_found_no_live_index_deletes_nothing(monkeypatch, tmp_path):
    """A missing `index.sqlite` means a swap or a restore is in progress.

    That is the worst possible moment to free a backup, so the pass declines and
    says so, while still measuring the pile it can see.
    """
    d = _qmd_dir(tmp_path, [
        ("index.sqlite.bak-a", 3000, 1.0), ("index.sqlite.bak-z", 2000, 5.0),
        ("index.sqlite.bak-m", 1000, 9.0)])
    report = _retention_run(monkeypatch, tmp_path, d, tmp_path / "repo", dry_run=False)
    assert report["stray_bytes"] == 3000 + 2000 + 1000, "measuring is still safe"
    assert report["stray_retention"]["planned"] == []
    assert report["stray_retention"]["deleted"] == []
    assert "index.sqlite.bak-z" in _listing(d) and "index.sqlite.bak-m" in _listing(d)
    assert "index.sqlite" in report["stray_retention"]["skipped"]


# --- clause 3: a candidate is held while something still reads it -------------

def test_a_candidate_still_named_by_a_code_reference_is_held_and_left_on_disk(
        monkeypatch, tmp_path):
    """The grep is re-run inside the same change, on the file it is about to delete.

    #855 asked for `perfbench.sqlite` to go "unless a named benchmark still reads
    it", and the reason the 2026-09-27 zero-hit grep is trustworthy is that its
    control hit seven real code files. A hold that cites the file it found is the
    same discipline applied at the moment of the delete.
    """
    repo = tmp_path / "repo"
    (repo / "app").mkdir(parents=True)
    (repo / "app" / "bench.py").write_text(
        "DB = Path.home() / '.cache/qmd/index.sqlite.bak-z'\n")
    d = _qmd_dir(tmp_path, LIVE_TRIO + [
        ("index.sqlite.bak-a", 3000, 1.0),
        ("index.sqlite.bak-z", 2000, 5.0), ("index.sqlite.bak-z-wal", 20, 5.0),
        ("index.sqlite.bak-m", 1000, 9.0)])
    report = _retention_run(monkeypatch, tmp_path, d, repo, dry_run=False)
    held = {h["name"]: h for h in report["stray_retention"]["held"]}
    assert list(held) == ["index.sqlite.bak-z"], report["stray_retention"]
    assert "index.sqlite.bak-z" in _listing(d), "a held candidate stays on disk"
    assert held["index.sqlite.bak-z"]["code_references"] == ["app/bench.py"]
    assert any("code reference" in b for b in held["index.sqlite.bak-z"]["because"]), held
    assert "index.sqlite.bak-m" not in _listing(d), (
        "the hold must not be a blanket refusal to ever delete anything")


def test_a_candidate_whose_sidecar_is_newer_than_its_own_main_file_is_held(
        monkeypatch, tmp_path):
    """A `-wal`/`-shm` newer than the database means something opened it read-write.

    Measured on the 2026-09-27 pile: `index.sqlite.bak-gemma-20260921-wal` is
    2026-09-24 13:36 and both `-shm` files 2026-09-24 14:07, against main files from
    09-19 and 09-21 — and no filename grep can see whoever did it, because a grep
    cannot match a path built at runtime. Report and hold; do not delete.
    """
    d = _qmd_dir(tmp_path, LIVE_TRIO + [
        ("index.sqlite.bak-a", 3000, 1.0),
        ("index.sqlite.bak-z", 2000, 5.0), ("index.sqlite.bak-z-wal", 20, 2.0),
        ("index.sqlite.bak-m", 1000, 9.0), ("index.sqlite.bak-m-shm", 32, 9.0)])
    report = _retention_run(monkeypatch, tmp_path, d, tmp_path / "repo", dry_run=False)
    held = {h["name"]: h for h in report["stray_retention"]["held"]}
    assert list(held) == ["index.sqlite.bak-z"], report["stray_retention"]
    assert "index.sqlite.bak-z" in _listing(d)
    assert any("-wal" in b for b in held["index.sqlite.bak-z"]["because"]), held
    assert "index.sqlite.bak-m" not in _listing(d), (
        "a sidecar the same age as its main file is a clean close, not a hold")


@pytest.mark.parametrize("relpath", [
    ".venvs/lloyd/lib/read.py", "llama.cpp/tools/thing.py", "qmd/src/store.ts",
    "node_modules/pkg/db.ts", ".git/hooks/pre-push.py", "docs/notes.md",
    "scripts/notes.txt",
])
def test_a_name_mentioned_only_outside_lloyd_s_own_code_is_not_a_reader(
        tmp_path, relpath):
    """The hold reads only `*.py`, `*.ts`, `*.sh`, `*.yml` under our own tree.

    The four 2026-09-27 strays are named in `architecture/qmd.md` and
    `qmd/WORKLOG.md` and opened by no program: prose is not a reader, and neither
    is a vendored or installed tree. Every one of these paths mentions the name.
    """
    repo = tmp_path / "repo"
    p = repo / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("DB = 'index.sqlite.bak-z'\n")
    assert m.code_reference_hits(["index.sqlite.bak-z"], repo) == {
        "index.sqlite.bak-z": []}


def test_the_code_reference_walk_finds_a_real_reader_and_a_real_absence():
    """A 0-hit grep needs its positive control, on the tree it will really run on.

    `index.sqlite` is named by code in this repo, so an empty answer for it would
    mean the walk is broken rather than that a name is unread — which is exactly how
    the 2026-09-27 zero became trustworthy: the same invocation's control hit seven
    files.
    """
    assert m.code_reference_hits(["index.sqlite"], m.REPO_ROOT)["index.sqlite"], (
        "the walk must find the name the live tree really does reference")
    # Assembled at runtime, and not written as one literal anywhere in this file:
    # this test file is itself part of the tree the walk reads, so a literal here
    # would be a code reference to the very name it is asserting is unreferenced.
    # That is not a hypothetical — the first run of this test failed exactly that
    # way, on its own sentinel.
    absent = "index.sqlite.bak-" + "19700101T000000Z-never-written"
    assert m.code_reference_hits([absent], m.REPO_ROOT)[absent] == []


def test_a_stray_outside_the_backup_series_is_held_for_a_person_and_never_deleted(
        monkeypatch, tmp_path):
    """"Not the live index" is not a licence to delete: the directory is shared.

    `eval/contextual_titles.py:41,71` opens `~/.cache/qmd/sub06.sqlite` and
    `eval/embed_side_index.py:52` builds `subctx.sqlite` there — live side-indexes
    of this very repo — and #844:141-142 already ruled that deleting someone's
    backup database is not a code round's call. So `perfbench.sqlite` (whose
    deletion also retires #408's subject) and `index.backup-*.sqlite` stay: measured,
    reported, and left for a person.
    """
    d = _qmd_dir(tmp_path, LIVE_TRIO + [
        ("index.backup-20260907_100044.sqlite", 998, 20.0),
        ("perfbench.sqlite", 1024, 8.0),
        ("sub06.sqlite", 512, 0.0),
        ("index.sqlite.bak-a", 3000, 1.0),
        ("index.sqlite.bak-z", 2000, 5.0)])
    report = _retention_run(monkeypatch, tmp_path, d, tmp_path / "repo", dry_run=False)
    after = _listing(d)
    assert {h["name"] for h in report["stray_retention"]["held_for_person"]} == {
        "index.backup-20260907_100044.sqlite", "perfbench.sqlite", "sub06.sqlite"}
    for keep in ("index.backup-20260907_100044.sqlite", "perfbench.sqlite",
                 "sub06.sqlite"):
        assert keep in after, keep
    assert "index.sqlite.bak-z" not in after, (
        "the .bak rule still works beside the files it may not touch")


# --- clause 4: --dry-run deletes nothing an acting run would ------------------

def test_a_dry_run_leaves_the_directory_byte_identical_to_what_an_acting_run_would_delete(
        monkeypatch, tmp_path, capsys):
    """Same fixture, both ways: the dry run must have something to delete and not
    delete it, or "deletes nothing" is only proven on a case that had nothing.

    The listing and the byte total are compared file-by-file — `--dry-run` already
    declines to write its report, and a flag that promises to change nothing has to
    be held to the directory it is pointed at, not just to the index. The dry run's
    own verdict comes off `--json` stdout, which is the only surface it leaves.
    """
    spec = LIVE_TRIO + [
        ("index.sqlite.bak-a", 3000, 1.0), ("index.sqlite.bak-a-wal", 30, 1.0),
        ("index.sqlite.bak-z", 2000, 5.0), ("index.sqlite.bak-z-wal", 20, 5.0),
        ("perfbench.sqlite", 1024, 8.0)]
    d = _qmd_dir(tmp_path, spec)
    before, total_before = _listing(d), sum(_listing(d).values())
    report = _retention_run(monkeypatch, tmp_path, d, tmp_path / "repo",
                            dry_run=True, capsys=capsys)
    assert _listing(d) == before, "--dry-run changed the index directory"
    assert sum(_listing(d).values()) == total_before
    assert report["stray_retention"]["planned"] == ["index.sqlite.bak-z"], (
        "the dry run has to name what an acting run would delete")
    assert report["stray_retention"]["deleted"] == []
    assert report["stray_retention"]["dry_run"] is True
    assert report["stray_bytes"] == 3000 + 30 + 2000 + 20 + 1024
    acting = _retention_run(monkeypatch, tmp_path, d, tmp_path / "repo", dry_run=False)
    assert acting["stray_retention"]["deleted"] == ["index.sqlite.bak-z"]
    assert "index.sqlite.bak-z" not in _listing(d)


# ── #1897: a fired capacity verdict escalates instead of riding a green run ──
#
# The trigger (#844) and its threshold pair are pinned above and are not re-specified
# here. What was missing is everything downstream of the verdict: `report["need_capacity"]`
# was set under a comment saying "Reported, never acted on", and both exit lines read
# only `daemon_healthy`, so the night occupancy crosses 0.25 the nightly run (#81)
# stays GREEN and the whole trace is one JSON field plus the summary's
# `capacity verdict True` suffix. These nodes pin the escalation — exit code, owed
# entry, summary line, and that escalating mutates nothing.

#: The live 2026-09-30 nightly artifact, transcribed where it matters
#: (`~/lloyd-data/_pipeline/reflection/qmd-index-maintenance-2026-09-30.json`):
#: 35,981 live rows in 106,496 allocated slots = occupancy 0.3379, 275.4 MiB dead,
#: footprint main 726,282,240 + wal 729,985,752 + shm 1,441,792 = 1,457,709,784 B,
#: and `need_capacity: false`. The dead-MiB floor (256) was cleared between 09-28 and
#: 09-29; only the ratio holds the verdict False, and across six nightly readings
#: occupancy fell 0.514 → 0.3379, so at the measured 0.0343/day it crosses 0.25 in
#: about 2.6 days — which is why the escalation is worth landing before it does.
#: `documents: 12_410` is the embed-guard denominator and is in the artifact, so it
#: is in the fixture too: at `pending_embeddings` 0 the guard cannot trip either way,
#: and a shape that omits the one field a later node would need to trip it is a shape
#: that quietly decides what the next node is allowed to ask.
SEPT_30_SHAPE = {
    "index_bytes": 1_457_709_784,
    "footprint": {"total": 1_457_709_784, "main": 726_282_240,
                  "wal": 729_985_752, "shm": 1_441_792},
    "vectors_total": 35_981, "vectors_orphaned": 78, "vectors_live": 35_903,
    "orphan_ratio": 0.0022, "documents": 12_410,
    "vec0": {"chunks": 104, "allocated_slots": 106_496, "live_rows": 35_981,
             "occupancy": 0.3379, "allocated_mib": 416.0, "dead_mib": 275.4},
}

#: The same snapshot with the ratio pushed over the trigger and nothing else moved,
#: so every assertion below is about the verdict and not about the shape around it.
FIRED_SHAPE = {**SEPT_30_SHAPE,
               "vec0": {**SEPT_30_SHAPE["vec0"], "occupancy": 0.2379}}


def _run_over_shape(monkeypatch, tmp_path, shape: dict, capsys,
                    tag: str = "run", probe_via_sh: bool = False):
    """`main()` end to end over a supplied `inspect_index` snapshot.

    Everything the index side would touch is stubbed the way `_run_main` stubs it —
    a nonexistent fixture INDEX so the acting stray delete has nothing to delete,
    zero pending embeds, a healthy daemon — and unlike that helper this one does NOT
    assert the exit code: the exit code is the thing under test. Returns the run's
    rc, the report it wrote, and what it printed.

    `tag` puts each call in its own `REPORT_DIR`. `_write_report` names the file by
    `started.date()`, so two runs in one `tmp_path` land on the same dated path and
    the second replaces the first — a node that runs the quiet shape to prove it
    stays green and then reads that report has read a different run's bytes.
    """
    t, l = _pair(tmp_path)
    monkeypatch.setattr(m, "TEMPLATE_CONFIG", t)
    monkeypatch.setattr(m, "LIVE_CONFIG", l)
    monkeypatch.setattr(m, "REPORT_DIR", tmp_path / f"reflection-{tag}")
    monkeypatch.setattr(m, "INDEX", tmp_path / "qmd" / "index.sqlite")
    monkeypatch.setattr(m, "inspect_index", lambda: dict(shape))
    monkeypatch.setattr(m, "pending_embeddings", lambda: 0)
    if not probe_via_sh:
        monkeypatch.setattr(m, "daemon_healthy", lambda retries=10: True)
    monkeypatch.setattr(sys, "argv", ["qmd_index_maintenance.py"])
    rc = m.main()
    out = capsys.readouterr().out
    reports = sorted((tmp_path / f"reflection-{tag}").glob("qmd-index-maintenance-*.json"))
    assert len(reports) == 1, f"expected exactly one dated report, got {reports}"
    return rc, json.loads(reports[0].read_text()), out


def test_a_fired_capacity_verdict_exits_non_zero(monkeypatch, tmp_path, capsys):
    """Clause 1, the half that was missing: the #958 pattern in this same file.

    The verdict fires on the 09-30 shape with occupancy pushed one trigger-width
    under 0.25 (0.2379, dead 275.4 MiB ≥ 256), on a daemon reported HEALTHY — and
    the run must not exit 0. This is the exact state 2026-10-03 is projected to be
    in, where every nightly run since then stayed green.
    """
    rc, report, _ = _run_over_shape(monkeypatch, tmp_path, FIRED_SHAPE, capsys)
    assert report["need_capacity"] is True and report["daemon_healthy"] is True
    assert rc != 0, f"a fired verdict exited {rc}: the fleet would read this green"


def test_a_quiet_capacity_verdict_on_a_healthy_daemon_still_exits_zero(monkeypatch, tmp_path, capsys):
    """Clause 1's other half: the escalation is a tripwire, not a red-by-default.

    Measured on the live 2026-09-30 snapshot itself — occupancy 0.3379, 275.4 MiB
    dead, so the floor is already cleared and only the ratio keeps the verdict
    False. A run today must stay green, or this change floods the fleet with
    failures for a condition that has not fired.
    """
    rc, report, out = _run_over_shape(monkeypatch, tmp_path, SEPT_30_SHAPE, capsys)
    assert report["need_capacity"] is False
    assert rc == 0, out
    assert "capacity OWED" not in out


def test_the_exit_rule_reads_both_conditions_and_nothing_else():
    """`_exit_code` on its own, so the pair of conditions is pinned without a run.

    A fired verdict is red whatever the daemon says; an unfired one is red only
    when retrieval is down, which is #958's rule and still is.
    """
    assert m._exit_code({"need_capacity": True, "daemon_healthy": True}) == 1
    assert m._exit_code({"need_capacity": True, "daemon_healthy": False}) == 1
    assert m._exit_code({"need_capacity": False, "daemon_healthy": True}) == 0
    assert m._exit_code({"need_capacity": False, "daemon_healthy": False}) == 1
    # A run that never measured capacity (no vec0 tables) is not a fired run.
    assert m._exit_code({"daemon_healthy": True}) == 0


def test_a_fired_verdict_reports_what_is_owed_and_on_what_numbers(monkeypatch, tmp_path, capsys):
    """Clause 2: the owed-to-a-person entry, carrying the measured numbers.

    A bare boolean cannot be actioned. The entry has to hold the ruling
    (`VEC0_REBUILD_RULED_OUT`, unchanged), the instruction naming the side copy and
    the hand swap, and the three figures a person needs to size the job —
    occupancy, dead MiB and the footprint total the rebuild would reclaim.
    """
    rc, report, _ = _run_over_shape(monkeypatch, tmp_path, FIRED_SHAPE, capsys,
                                    tag="fired")
    owed = report["capacity_owed_to_a_person"]
    assert owed["occupancy"] == 0.2379 and owed["dead_mib"] == 275.4
    assert owed["footprint_bytes"] == 1_457_709_784
    assert owed["because"] == m.VEC0_REBUILD_RULED_OUT
    assert owed["owed_to"] == "a person, by hand — this job will not do it"
    assert "side copy" in owed["what_to_do"] and "BY HAND" in owed["what_to_do"]
    assert "swap" in owed["what_to_do"] and "Never in place" in owed["what_to_do"]
    # The unfired case records no owed entry at all: nothing is owed.
    # Its own REPORT_DIR (`tag`): `_write_report` names the file by date, so a second
    # run in the same dir would have replaced the report above rather than adding one.
    _, quiet, _ = _run_over_shape(monkeypatch, tmp_path, SEPT_30_SHAPE, capsys,
                                  tag="quiet")
    assert "capacity_owed_to_a_person" not in quiet


def test_the_summary_prints_the_owed_entry_and_the_by_hand_instruction(monkeypatch, tmp_path, capsys):
    """Clause 3: the human summary, not only the `capacity verdict True` suffix.

    Asserted on the printed lines and not on the report, because the summary is
    what a person reads when the nightly run goes red.
    """
    rc, _, out = _run_over_shape(monkeypatch, tmp_path, FIRED_SHAPE, capsys)
    owed = [ln for ln in out.splitlines() if "capacity OWED" in ln]
    assert len(owed) == 1, out
    assert "a person, by hand" in owed[0]
    assert "23.8 % occupancy" in owed[0], owed[0]
    assert "275 MiB dead" in owed[0], owed[0]
    assert "1,457,709,784 B" in owed[0], owed[0]
    instruction = [ln for ln in out.splitlines() if "BY HAND" in ln]
    assert len(instruction) == 1 and "side copy" in instruction[0], out
    assert "Never in place" in instruction[0], out
    # The measurement line is still there beside it — the owed line replaces nothing.
    assert "capacity verdict True" in out, out


def test_a_fired_verdict_escalates_without_touching_the_index(monkeypatch, tmp_path, capsys):
    """Clause 4: the escalation is a report and an exit code, never an action.

    Nothing may be added to `actions`, and no mutating subprocess may run: not
    `cleanup`, not `embed`, and certainly not an in-place vec0 drop, which is the
    outcome VEC0_REBUILD_RULED_OUT exists to prevent. `actions` is empty rather than
    "none — nothing to do", because work IS owed — just not this job's.

    `daemon_healthy` is deliberately NOT stubbed here (the helper's `probe_via_sh`),
    so `_sh` is exercised on the path it actually takes: the fired run does spawn one
    subprocess, the read-only health probe curl at :733, and asserting `calls == []`
    would have asserted that away rather than asserting the mutating commands are
    absent. So the assertion is on the commands, and the probe's presence is asserted
    too — a run that skipped the probe would be a different run.
    """
    calls: list[list[str]] = []
    monkeypatch.setattr(m, "_sh", lambda cmd, timeout, env=None: calls.append(cmd) or (0, ""))
    rc, report, _ = _run_over_shape(monkeypatch, tmp_path, FIRED_SHAPE, capsys,
                                    tag="mutating", probe_via_sh=True)
    assert rc != 0 and report["need_capacity"] is True
    assert report["actions"] == [], report["actions"]
    mutating = [c for c in calls
                if any(w in c for w in ("cleanup", "embed", "vacuum", "reindex"))]
    assert mutating == [], f"the fired path ran a mutating command: {mutating}"
    assert [c for c in calls if c and c[0] == "curl"] == \
        [["curl", "-s", "-m", "3", "-o", "/dev/null", m.DAEMON_PROBE_URL]], calls
    assert report["daemon_healthy"] is True
    assert report["need_prune"] is False and report["need_embed"] is False
    assert "after" not in report, "the mutating section ran on a fired verdict"
    # And the numbers it escalated on are the ones it measured, unchanged.
    assert report["before"]["vec0"] == FIRED_SHAPE["vec0"]


#: Where #1897's witness lives once committed. `~/lloyd-data/_pipeline/reflection/`
#: is gitignored AND under the retention sweep, so the six nightly readings the
#: projection is drawn from had no history: the item quoted numbers nobody could
#: re-measure a month from now.
WITNESS_REL = Path("backlog") / "data" / "qmd-index-maintenance-2026-09-30.json"


def test_the_committed_witness_agrees_with_the_fixture_this_change_is_pinned_to(
        monkeypatch):
    """The projection in #1897 must be re-derivable from bytes that have history.

    Reads the artifact through `board_presence.vault_root()` (env
    `LLOYD_OBSIDIAN_VAULT`, read per call, so a caller can point it at a fixture
    vault). The skip is for the one case with nothing to check: a checkout with no
    vault directory at all, which `tests/board_presence.py` rules must never be
    permanently red. A vault that exists and holds no witness is a FAILURE and not a
    skip — the copy IS the deliverable, so a node that skipped when it was absent
    would pass on exactly the regression it is here for (the copy never landed, or a
    later sweep took it) and would only ever catch drift between two files that are
    both present. What it then pins is the pair that matters: the committed witness
    reproduces every figure the item quotes, AND the fixture the other nodes run on
    is the same shape, so a transcription cannot drift from the bytes it came from.
    """
    from board_presence import VAULT_ROOT_ENV, vault_root

    root = vault_root()
    if not root.is_dir():
        pytest.skip(f"no vault directory at {root} (set {VAULT_ROOT_ENV} to point at "
                    f"one): there is no board here for the witness to live on")
    witness = root / WITNESS_REL
    assert witness.is_file(), (
        f"#1897's witness is not committed at {witness}. Without it the six nightly "
        f"readings behind the 0.25 projection live only in gitignored "
        f"~/lloyd-data/_pipeline/reflection/, which the retention sweep bounds — the "
        f"numbers this change is pinned to would have no history to check against.")
    art = json.loads(witness.read_text())
    v, f = art["before"]["vec0"], art["before"]["footprint"]
    assert (v["occupancy"], v["dead_mib"]) == (0.3379, 275.4)
    assert (v["live_rows"], v["allocated_slots"]) == (35_981, 106_496)
    assert f["total"] == 1_457_709_784 == f["main"] + f["wal"] + f["shm"]
    assert art["need_capacity"] is False, "the witness is the fired-but-green night"

    assert SEPT_30_SHAPE["vec0"] == v, "fixture drifted from the committed witness"
    assert SEPT_30_SHAPE["footprint"] == f
    # Re-derived, not quoted: at the measured 0.0343/day (the true last-3-day rate —
    # (0.4407 - 0.3379) / 3; the item's 0.013/day divided a 2-day span by 3), 0.3379
    # reaches the 0.25 trigger in under three days, which is why the escalation is
    # worth landing before it does rather than after.
    assert (0.3379 - 0.25) / 0.0343 < 3.0
