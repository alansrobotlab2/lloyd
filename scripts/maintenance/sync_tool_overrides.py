#!/usr/bin/env python3
"""Re-sync `data/tool_overrides.yaml`'s tool_search block from config.yaml.

WHY THIS EXISTS
---------------
`data/tool_overrides.yaml` is gitignored runtime state, merged OVER config.yaml
at load (`app/config.py::_merge_tool_overrides`). The Tools page rewrites the
whole `harness.tool_search` block on every toggle, so the override shadows
`baseline_tools` **wholesale** — which means a change to that list in the
tracked file leaves the two disagreeing, and the state a fresh clone would boot
into is not the state being served.

`tests/test_tool_overrides.py::test_config_yaml_agrees_with_the_live_override`
catches that, and it can only catch it **in live**: a autoimplement worktree has no
override file, so the test returns early there. So the failure mode is a
tracked-config change that passes the gate in every worktree and then fails the
`tests` rung on the live tree afterwards — which is how three rounds aborted in
fifteen hours on 2026-09-07 for an unrelated version of the same shape.

Rather than "remember to hand-edit a gitignored file after merging", this makes
it one idempotent command. Run it after any commit that changes
`harness.tool_search` in config.yaml.

    python -m scripts.maintenance.sync_tool_overrides [--check] [ROOT]

`--check` reports drift and exits 1 without writing, for a pre-merge probe.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

BLOCK = "tool_search"


def drift(root: pathlib.Path) -> tuple[dict, dict, dict]:
    """(served, tracked, drift) for the tool_search block under `root`."""
    cfg = yaml.safe_load((root / "config.yaml").read_text()) or {}
    tracked = ((cfg.get("harness") or {}).get(BLOCK) or {})
    ovr_path = root / "data" / "tool_overrides.yaml"
    if not ovr_path.exists():
        return {}, tracked, {}
    ovr = yaml.safe_load(ovr_path.read_text()) or {}
    served = ((ovr.get("harness") or {}).get(BLOCK) or {})
    # Only keys the override actually shadows, which is what the test compares.
    return served, tracked, {
        k: (tracked.get(k), v) for k, v in served.items()
        if k in tracked and tracked[k] != v
    }


def describe(served: dict, tracked: dict, d: dict) -> list[str]:
    lines = []
    for key, (want, have) in d.items():
        lines.append(f"  {key}: served={have!r} tracked={want!r}"
                     if not isinstance(want, list) else f"  {key}:")
        if isinstance(want, list) and isinstance(have, list):
            added = [t for t in want if t not in have]
            removed = [t for t in have if t not in want]
            if added:
                lines.append(f"    + {', '.join(added)}")
            if removed:
                lines.append(f"    - {', '.join(removed)}")
            if not added and not removed:
                lines.append("    (same members, different order)")
    return lines


def sync(root: pathlib.Path, *, check: bool = False) -> int:
    ovr_path = root / "data" / "tool_overrides.yaml"
    served, tracked, d = drift(root)
    if not ovr_path.exists():
        print(f"no override at {ovr_path} — config.yaml is the served state")
        return 0
    if not d:
        print("already in sync")
        return 0

    print(f"drift between config.yaml and {ovr_path}:")
    for line in describe(served, tracked, d):
        print(line)
    if check:
        print("\n--check: not writing. Re-run without it to sync.")
        return 1

    ovr = yaml.safe_load(ovr_path.read_text()) or {}
    for key in d:
        served[key] = tracked[key]
    ovr.setdefault("harness", {})[BLOCK] = served
    # Written whole, the same way `save_tool_overrides` writes it. Every other
    # top-level key (mcp_servers, workers) is carried through untouched — this
    # file also holds the UI's tool disables and the worker switch.
    ovr_path.write_text(yaml.dump(ovr, sort_keys=False, default_flow_style=False))
    print(f"\nwrote {ovr_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", default=str(pathlib.Path.home() / "lloyd"))
    ap.add_argument("--check", action="store_true",
                    help="report drift and exit 1 without writing")
    args = ap.parse_args()
    return sync(pathlib.Path(args.root), check=args.check)


if __name__ == "__main__":
    sys.exit(main())
