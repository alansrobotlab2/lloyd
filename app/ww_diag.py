"""Where the wake-miss diagnostic corpus lives, and nothing else.

`WakeMissCapture` (`agent-services/livekit_worker.py`) writes this corpus and
four scripts read it: `scripts/ww_diag_summary.py`, `scripts/voice/replay.py`,
`scripts/voice/wake_eval.py`, `scripts/voice/hotword_eval.py`. Until #1444 every
one of them spelled the location as the account home's hidden `.lloyd` directory
plus `ww_diag`, which put the only copy of the wake-word tuning corpus outside
every guarantee the data root carries:

* `scripts/backup/snapshot-data.sh` snapshots `${LLOYD_DATA:-…/lloyd-data}`
  alone, and the box's other snapshotter (snapper, one config, `SUBVOLUME="/"`)
  does not reach the `@home` subvolume the dot-directory sits on — so the corpus
  had no snapshot at all;
* `app.harness.protected_paths.protected_roots()` guards the data root, the
  vault and the trees, so `rm -r` over the corpus's top-level folder was allowed
  while the same call on `<data root>/voice_profiles` is refused;
* an automod round's home *symlinks* that dot-directory
  (`scripts/automod/worktree.py:HOME_LINK_SKIP` skips `lloyd` and `lloyd-data`
  only), so candidate code run by a gate appended to the **live** corpus instead
  of a per-round one. `lloyd-data` *is* skipped and `round_data_root` is what the
  gate exports as `LLOYD_DATA`, so moving the corpus under the root is what
  buys the isolation, not a per-round rewrite of the path.

One module therefore owns the layout, so the writer and the four readers cannot
drift into naming different places. `diag_dir()` is the only expression of it and
the four names below are its files; the corpus is a *top-level folder of the data
root*, which is the shape `protected_roots()` refuses to delete wholesale.

The resolution is the one resolver — `app.data_root`, the stdlib-only half, so a
reader run under plain `python3` (cron, a unit's `sh -c` child) gets the same
answer as the worker running in the venv. That is the #1415 rule: the copy of the
rules, not the rules, is what diverges.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: The checkout this module is in. The corpus belongs to the data root of *this*
#: tree, which is what makes a round's copy its own (`LLOYD_DATA`, else
#: `<tree>/.lloyd-data`) and production's the live root.
_TREE = Path(__file__).resolve().parents[1]

# The same two spellings `scripts/groundskeeper/retention-sweep.py` uses: the
# package form whenever `app` is importable — which needs no path surgery, so
# importing this module does not rearrange `sys.path` for the caller — and the
# loose file form for a caller that reached it by putting a directory on the
# path with no view of the package. The fallback appends rather than inserts:
# `<tree>/app` holds `paths.py`, `config.py` and friends, and pushing them to
# the front of every later import in a voice script would shadow packages of
# those names for work this corpus has nothing to do with.
try:
    from app.data_root import resolve_data_root_for_tree
except ImportError:  # `app` unimportable or shadowed: the same file, loose
    for _p in (_TREE, _TREE / "app"):
        if str(_p) not in sys.path:
            sys.path.append(str(_p))
    del _p  # a module-level Path left by that loop is a name a test can delete
    from data_root import resolve_data_root_for_tree

#: The corpus's name under the data root. A top-level folder, deliberately: that
#: is the level the delete guard refuses.
CORPUS_DIR = "ww_diag"


def data_root() -> Path:
    """This tree's runtime data root — the three rules, no second copy."""
    return resolve_data_root_for_tree(_TREE)


def diag_dir() -> Path:
    """The corpus directory. Resolved per call, so a relocated `LLOYD_DATA`
    (a test, a gate round, a canary) moves it without a re-import."""
    return data_root() / CORPUS_DIR


def utterances_dir() -> Path:
    """Every utterance the segmenter emitted, capped by
    `WakeMissCapture.MAX_UTTERANCE_FILES`."""
    return diag_dir() / "utterances"


def misses_dir() -> Path:
    """Explicit miss reports from `/api/voice/ww_miss`: `<ts>_<label>` plus a
    `.wav` (the room's rolling raw-audio ring) and a `.json`."""
    return diag_dir() / "misses"


def scores_path() -> Path:
    """The append-only per-utterance record: the ground-truth log every replay
    and threshold sweep is computed from."""
    return diag_dir() / "scores.jsonl"


def labels_path() -> Path:
    """Ground-truth labels attached by `/api/voice/ww_label`."""
    return diag_dir() / "labels.jsonl"
