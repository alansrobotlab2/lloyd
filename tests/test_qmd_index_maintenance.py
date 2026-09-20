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
import sys
from pathlib import Path

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
    """
    monkeypatch.setattr(m, "TEMPLATE_CONFIG", template)
    monkeypatch.setattr(m, "LIVE_CONFIG", live)
    monkeypatch.setattr(m, "REPORT_DIR", tmp_path / "reflection")
    monkeypatch.setattr(m, "inspect_index", lambda: {"orphan_ratio": 0.0, "vectors_orphaned": 0})
    monkeypatch.setattr(m, "pending_embeddings", lambda: 0)
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
    monkeypatch.setattr(sys, "argv", ["qmd_index_maintenance.py", "--dry-run"])
    assert m.main() == 0
    assert not (tmp_path / "reflection").exists(), "--dry-run promises to change nothing"
    printed = capsys.readouterr().out
    assert "sessions" in printed
    assert "cp ~/.config/qmd/index.yml agent-services/conf/qmd-index.yml" in printed
