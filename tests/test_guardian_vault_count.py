"""Both vault counters skip `.git`, and they are one definition.

`git gc` packs loose objects by the hundred. A counter that walks `.git/**`
reads a repack as note loss, and it did so twice: 2026-09-09 22:01 (#537
reverted) and 2026-09-17 08:59 (#1206 reverted, "vault files dropped 6.7%
(6075 → 5667)" while the note count sat at 5622 the whole time). The
promoter's baseline (`scripts/automod/promote.py`) and the guardian's live
count (`agent-services/guardian/guardian.py`) had private copies of the same
`rglob`; now both go through `vaultwatch.measure`, whose `SKIP_DIRS` is the
tripwire's rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
sys.path.insert(0, str(GUARDIAN_DIR))

import detect  # noqa: E402
import guardian as G  # noqa: E402
import policy  # noqa: E402

from scripts.automod import promote as P  # noqa: E402

NOTES = 30
LOOSE_OBJECTS = 400


def _vault(tmp_path: Path) -> Path:
    root = tmp_path / "obsidian"
    for i in range(NOTES):
        d = root / ("knowledge" if i % 2 else "backlog")
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{i}.md").write_text("note\n")
    objects = root / ".git" / "objects"
    for i in range(LOOSE_OBJECTS):
        d = objects / f"{i % 256:02x}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{i:038x}").write_bytes(b"blob")
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    return root


def _repack(root: Path) -> None:
    """What `git gc` does to the file count: loose objects become one pack."""
    for p in list((root / ".git" / "objects").rglob("*")):
        if p.is_file():
            p.unlink()
    pack = root / ".git" / "objects" / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "pack-deadbeef.pack").write_bytes(b"pack")


def test_the_guardian_counts_notes_not_git_objects(tmp_path):
    root = _vault(tmp_path)
    assert G.count_vault_files(str(root)) == NOTES
    _repack(root)
    assert G.count_vault_files(str(root)) == NOTES


def test_the_promoter_count_is_the_guardians(tmp_path):
    root = _vault(tmp_path)
    assert P.count_vault_files(root) == G.count_vault_files(str(root)) == NOTES
    _repack(root)
    assert P.count_vault_files(root) == G.count_vault_files(str(root)) == NOTES


def test_a_missing_vault_is_none_not_zero(tmp_path):
    assert G.count_vault_files(str(tmp_path / "gone")) is None
    assert P.count_vault_files(tmp_path / "gone") is None


def test_a_repack_cannot_trip_the_detector_but_note_loss_still_does(tmp_path):
    """The incident, end to end: baseline before the repack, live count after."""
    root = _vault(tmp_path)
    before = P.count_vault_files(root)
    _repack(root)
    hit, _ = detect.data_damage(before, G.count_vault_files(str(root)),
                                policy.DATA_DROP_FRACTION)
    assert hit is False
    for p in list((root / "knowledge").glob("*.md"))[: NOTES // 5]:
        p.unlink()
    hit, why = detect.data_damage(before, G.count_vault_files(str(root)),
                                  policy.DATA_DROP_FRACTION)
    assert hit is True, why
