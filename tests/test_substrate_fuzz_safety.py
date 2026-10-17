#!/usr/bin/env python3
"""#2620 clause 5: blast radius. These tasks write files, spawn children and eat
memory while the box is serving Alan, so the guard runs BEFORE dispatch — a refusal
after the tool ran is a deletion that already happened.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _substrate_fuzz import REAL  # noqa: E402

from eval.substrate_fuzz import runner as R  # noqa: E402


#: One template per protected tree, in the four spellings the acceptance names plus
#: one absolute spelling of the vault, because `~/obsidian` and
#: `/home/<user>/obsidian` are the same tree to the guard and two different strings
#: to a guard that forgot `expanduser` — which is the defect this node found on the
#: previous round's head. Each carries a `{probe}` slot the test fills with a fresh
#: token, for a reason that is itself a finding: the first version of this node used
#: a fixed name, and the run whose guard was still broken created
#: `~/obsidian/memory/planted.md` for real, after which every honest re-run of the
#: node was red on the leftover rather than on the regression. A probe that leaves
#: debris behind turns a red instrument into a permanently red one, and the tree it
#: dirties is the one this node exists to protect.
_PROBE_TEMPLATES = [
    "~/obsidian/knowledge/probe-{probe}.md",
    "~/lloyd/app/probe-{probe}.py",
    "~/.openclaw/probe-{probe}.json",
    "~/.venvs/lloyd/probe-{probe}.py",
    str(Path.home() / "obsidian" / "memory" / "probe-{probe}.md"),
]


@pytest.mark.parametrize("template", _PROBE_TEMPLATES)
def test_a_protected_target_is_refused_before_dispatch(template):
    """No tool call happens, the file does not appear, and the task is SKIPPED —
    counted in neither k nor N and named, because a domain whose tasks were all
    skipped measured nothing rather than measuring clean.

    The probe name is fresh per run, and the node refuses to start on a target that
    already exists: an absent-before/absent-after pair is the only evidence that the
    guard fired before dispatch, and a path that was already standing proves nothing
    either way. If the guard ever regresses the plant this node caused is removed in
    the same breath as the failure, so the instrument never accumulates litter in the
    trees it polices."""
    target = template.format(probe=uuid.uuid4().hex[:10])
    resolved = Path(target).expanduser()
    assert not resolved.exists(), (
        f"{resolved} already exists before the run: this node plants a path it expects "
        "nobody owns, so a pre-existing file makes the after-check unprovable")
    try:
        rec = R.run_task(_task(pointing_at=target))
        assert rec["ok"] is None, rec
        assert rec["skipped"] == "protected-target", rec
        assert not resolved.exists(), (
            f"{target} was written: the guard ran after dispatch, not before")
    finally:
        if resolved.exists():
            resolved.unlink()


def test_the_guard_does_not_refuse_every_path():
    """Positive control for this file: a plain path under the run root is not
    refused, so a refusal above cannot be the guard blocking everything.

    The write now lands INSIDE the run root and the postcondition checks it, so the control is
    a GREEN record rather than merely a non-skipped one. That change is not a tightening for its
    own sake: the row used to write to a tempfile outside the run root, where its own
    postcondition could never see it, and that was harmless only while a payload miss always
    printed a layer. #2929 made an unattributable miss RAISE instead of filing itself under
    `model` — the layer arm A cannot earn — so this control started dying on `AttributionError`
    for a reason that has nothing to do with the guard. A guard that lets an ordinary write
    through, and leaves the row green, is the assertion this node means to make."""
    rec = R.run_task(_task(pointing_at="{root}/plain.bin",
                           post={"kind": "file_exists", "path": "plain.bin"}))
    assert rec["skipped"] is None, (
        f"the guard refused an ordinary path under the run root ({rec['evidence']}), so the "
        "refusals above say nothing about protected trees specifically")
    assert rec["ok"] is True, f"{rec['rule']!r} / {rec['evidence']!r}"


def test_the_run_root_is_never_under_a_protected_tree():
    """The root is the one path the runner picks itself, so it is checked too."""
    rec = R.run_task(REAL[0])
    root = rec["run_root"]
    assert not any(seg in root for seg in ("obsidian", "/lloyd/app", ".openclaw",
                                           ".venvs")), root


def test_the_artifact_constant_lives_inside_data_root():
    """Clause 5 asks for the declaration in app/paths.py, so the store is findable
    in the one file a reader looks in — and it must hang off DATA_ROOT, not the code
    tree, or a `rm -r` aimed at the checkout reaches a live store."""
    from app import paths as P
    d = P.EVAL_SUBSTRATE_FUZZ_DIR
    assert str(d).startswith(str(P.DATA_ROOT)), d
    assert P.DATA_ROOT != ROOT, "DATA_ROOT resolved to the code tree"


def test_the_writer_prunes_its_own_dated_rows(tmp_path):
    """The directory is bounded without a retention-sweep row: #2615 is settling a
    half-landing of that table, and a store that waits for it is unbounded until
    somebody writes the row."""
    old = tmp_path / f"arm-a-{(datetime.now(timezone.utc) - timedelta(days=R.ARTIFACT_RETENTION_DAYS + 5)):%Y-%m-%d}.json"
    old.write_text("{}")
    fresh = tmp_path / f"arm-a-{datetime.now(timezone.utc):%Y-%m-%d}.json"
    fresh.write_text("{}")
    removed = R.prune_artifacts(tmp_path)
    assert not old.exists(), "a row past its retention day survived the writer"
    assert fresh.exists() and removed == 1, removed


def test_a_run_writes_one_dated_artifact_and_nothing_else(tmp_path):
    """The artifact is the only output worth keeping; the run root is removed with
    the spill the control injection produces."""
    out = R.run_corpus(REAL[:2])
    p = R.write_artifact(out, out_dir=tmp_path)
    assert p.parent == tmp_path and p.name.startswith("arm-a-"), p
    assert json.loads(p.read_text())["arm"] == "A"
    assert len(list(tmp_path.glob("arm-a-*.json"))) == 1


def test_the_surface_arm_a_drives_names_no_sending_tool(tmp_path):
    """Arm A dispatches through the real seam with a real session id, so a task naming
    `email_send` would mail somebody from a test run, and `session_inject_context` would put
    text in front of Alan. What closes that is not the path guard but the corpus loader's closed
    tool surface (`KNOWN_TOOLS`), and this node pins both halves: the surface itself holds none
    of the tools that can reach outward, and a task naming one is refused at load rather than
    skipped at dispatch — a refusal after the fact would be a send that already happened, the
    same reason the path guard runs before dispatch.

    The names below are the outward half of this agent's own tool pool, not a list inferred from
    the corpus; if a sender is ever added to `KNOWN_TOOLS` this node is the thing that notices.
    """
    from eval.substrate_fuzz.corpus import KNOWN_TOOLS, CorpusError, load_corpus
    outward = ("email_send", "email_reply", "email_forward", "email_save_draft",
               "calendar_create", "calendar_update_event", "calendar_delete_event",
               "tasks_create", "tasks_update", "contacts_create", "contacts_update",
               "contacts_delete", "session_inject_context", "memory_add", "fact_add",
               "automod_land", "desktop_act", "email_update")
    assert not (set(outward) & set(KNOWN_TOOLS)), sorted(set(outward) & set(KNOWN_TOOLS))
    for tool in ("email_send", "session_inject_context"):
        row = {"id": f"probe-{tool}", "domain": "filesystem", "title": "surface probe",
               "reason": "a test needs a step naming a tool outside the surface",
               "steps": [{"tool": tool, "args": {"to": "someone@example.com"}}],
               "postcondition": {"kind": "file_exists", "path": "plain.bin"}}
        p = tmp_path / f"corpus-{tool}.json"
        p.write_text(json.dumps({"version": 1, "item": 2620, "tasks": [row]}))
        with pytest.raises(CorpusError, match="outside the arm-A surface"):
            load_corpus(p, check_cross_row=False)


def _task(pointing_at: str, post: dict | None = None):
    """A task whose single Write names `pointing_at` — built through the real loader
    so a refusal cannot be an artefact of a malformed synthetic row.

    `post` defaults to a digest no write can satisfy, which is what the refusal nodes want:
    they never reach it, and if the guard ever stops firing the row goes red instead of green."""
    from eval.substrate_fuzz.corpus import load_corpus
    import tempfile
    d = Path(tempfile.mkdtemp())
    row = {"id": "guard-probe", "domain": "filesystem", "title": "guard probe",
           "reason": "a test needs a step naming one path",
           "steps": [{"tool": "Write",
                      "args": {"file_path": pointing_at, "content": "planted"}}],
           "postcondition": post or {"kind": "file_sha256", "path": "plain.bin",
                                     "expected": "0" * 64}}
    p = d / "corpus.json"
    p.write_text(json.dumps({"version": 1, "item": 2620, "tasks": [row]}))
    return load_corpus(p, check_cross_row=False)[0]
