#!/usr/bin/env python3
"""Report (and where possible repair) drift between config.yaml and the override.

WHY THIS EXISTS
---------------
`data/tool_overrides.yaml` (under the data root, `app.paths.TOOL_OVERRIDES_PATH`) is runtime state, merged OVER config.yaml
at load (`app/config.py::_merge_tool_overrides`). The Tools page rewrites the
whole `harness.tool_search` block on every toggle, so the override shadows
`baseline_tools` **wholesale**, and it is the authority on each server's
`enabled` / `disabled_tools` and on `workers.enabled`. So the tracked file can
lie about the served surface in either direction: a config.yaml edit the
override does not restate never reaches the served state, and a tool the
override disables but config.yaml never lists makes a rebuild boot something
this machine switched off (`architecture/tools.md` §8).

WHAT IT REPORTS
---------------
`parity()` returns one entry per disagreement, and `--check` prints them all and
exits 1 without writing:

    harness.tool_search.<key>            shadowed with a different value
    mcp_servers.<name>                   the override names a server config.yaml has no block for
    mcp_servers.<name>.enabled           the two files disagree on whether the server runs
    mcp_servers.<name>.disabled_tools    the two disable-sets differ (both directions)
    workers.enabled                      the two disagree on whether the pool runs

Until #3128/#3383 this compared `harness.tool_search` only, and printed
`already in sync` and exited 0 on a box whose override disabled fifteen tools
config.yaml does not list. A `--check` that returns 0 on a divergent tree is a
red instrument: wired into a gate it would have assured exactly the divergence
it exists to catch.

WHAT IT CAN FIX
---------------
Only `harness.tool_search`, by rewriting the override to config.yaml's values —
the one direction that reconciles the files without changing behaviour. For the
other classes the override wins by design, so the file to edit is the tracked
one; this command prints them and exits non-zero rather than rewriting your tool
disables for you.

WHY IT IS NOT A TEST
--------------------
`tests/test_tool_overrides.py::test_config_yaml_agrees_with_the_live_override`
pins this comparison, and it does so on fixture pairs — it CANNOT ask the
question about the box it runs on. The suite repoints `LLOYD_DATA` at a scratch
data root and `tests/conftest.py` refuses to run against the production tree, so
no test can read the live override. That used to be written the other way round:
the comment here claimed the test "can only catch it **in live**", but the test
resolved `ROOT / "data/tool_overrides.yaml"`, a path that has existed in no
checkout since runtime data left the code tree in `6426668b` (2026-09-22), so its
`if not live.exists(): return` fired everywhere and the guard was silent
everywhere. Asking about this box is this command, outside pytest:

    cd ~/lloyd && LLOYD_DATA=/home/alansrobotlab/lloyd-data \\
        .venvs/lloyd/bin/python -m scripts.maintenance.sync_tool_overrides --check

Run it after any commit that changes `harness.tool_search` in config.yaml, and
before merging one that changes a server's `disabled_tools`.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from app import paths  # noqa: E402

BLOCK = "tool_search"
SERVERS = "mcp_servers"
#: The tracked-side value for a server the override names and config.yaml does
#: not define. A string, not None, so the printed line reads as an absence
#: rather than as a value that happens to be null.
ABSENT = "<no block in config.yaml>"


def overrides_path(root: pathlib.Path) -> pathlib.Path:
    """The override file served for the checkout at `root`.

    It lives in that checkout's data root, not the tree: the running tree's is
    `paths.TOOL_OVERRIDES_PATH` (so `LLOYD_DATA` is honoured), the live checkout's
    is under `production_data_root()`, and any other checkout keeps its own under
    `data_root_for_tree(root)`.
    """
    rel = paths.TOOL_OVERRIDES_PATH.relative_to(paths.DATA_ROOT)
    resolved = pathlib.Path(root).resolve()
    if resolved == paths.LLOYD_HOME:
        return paths.TOOL_OVERRIDES_PATH
    if resolved == paths.LIVE_CHECKOUT.resolve():
        return paths.production_data_root() / rel
    return paths.data_root_for_tree(resolved) / rel


def _load_yaml(path: pathlib.Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def parity(config: dict, overrides: dict) -> dict:
    """Every disagreement between the tracked `config` and the served `overrides`.

    Returns `{dotted_key: (tracked, served)}`; an empty dict means the tracked
    file describes what the override serves. The two dicts are the parsed
    documents, so a caller can run this over fixture pairs or over two files on
    disk — the comparison is the same one either way.

    A key config.yaml never mentions is NOT a disagreement: an override may add
    a default the tracked file leaves unstated (`app/config.py` warns the same
    way, and `test_a_key_absent_from_config_is_not_a_disagreement` pins it). A
    key the override does not carry is not drift either — the merge keeps
    config.yaml's value for it.
    """
    report: dict = {}

    tracked_ts = ((config.get("harness") or {}).get(BLOCK) or {})
    for key, served in ((overrides.get("harness") or {}).get(BLOCK) or {}).items():
        if key in tracked_ts and tracked_ts[key] != served:
            report[f"harness.{BLOCK}.{key}"] = (tracked_ts[key], served)

    tracked_servers = config.get(SERVERS) or {}
    for name, ovr in (overrides.get(SERVERS) or {}).items():
        ovr = ovr or {}
        tracked = tracked_servers.get(name)
        if not isinstance(tracked, dict):
            # The merge drops the whole block — "overrides can't introduce
            # servers" (`app/config.py`) — so nothing is served from it and the
            # tracked file is not wrong about the surface. It is still the one
            # class of divergence the merge hid completely, and the usual cause
            # is a renamed or deleted server, which is worth a line.
            report[f"{SERVERS}.{name}"] = (ABSENT, ovr)
            continue
        if ("enabled" in ovr and "enabled" in tracked
                and bool(tracked["enabled"]) != bool(ovr["enabled"])):
            report[f"{SERVERS}.{name}.enabled"] = (
                bool(tracked["enabled"]), bool(ovr["enabled"]))
        if "disabled_tools" in ovr:
            tracked_set = set(tracked.get("disabled_tools") or [])
            served_set = set(ovr.get("disabled_tools") or [])
            if tracked_set != served_set:
                report[f"{SERVERS}.{name}.disabled_tools"] = (
                    sorted(tracked_set), sorted(served_set))

    tracked_workers = config.get("workers") or {}
    ovr_workers = overrides.get("workers") or {}
    if ("enabled" in ovr_workers and "enabled" in tracked_workers
            and bool(tracked_workers["enabled"]) != bool(ovr_workers["enabled"])):
        report["workers.enabled"] = (
            bool(tracked_workers["enabled"]), bool(ovr_workers["enabled"]))

    return report


def drift(root: pathlib.Path) -> tuple[dict, dict, dict]:
    """(served, tracked, drift) for the checkout at `root`.

    The first two are the `harness.tool_search` blocks — the half `sync` can
    rewrite. The third is the whole `parity()` report over the two files: that
    block plus every server's `enabled` / `disabled_tools` and `workers.enabled`.
    A missing override file is agreement, because then config.yaml *is* the
    served state.
    """
    cfg = _load_yaml(root / "config.yaml")
    tracked = ((cfg.get("harness") or {}).get(BLOCK) or {})
    ovr_path = overrides_path(root)
    if not ovr_path.exists():
        return {}, tracked, {}
    ovr = _load_yaml(ovr_path)
    served = ((ovr.get("harness") or {}).get(BLOCK) or {})
    return served, tracked, parity(cfg, ovr)


def describe(d: dict) -> list[str]:
    """Printable lines for a `parity()` report.

    A list-valued entry (a disable-set or `baseline_tools`) prints its members
    one per line: `+` is tracked-but-not-served, `-` is served-but-not-tracked.
    A tool named after `-` is one the box has switched off that config.yaml does
    not know about — the rebuild hazard.
    """
    lines = []
    for key, (want, have) in d.items():
        if isinstance(want, list) and isinstance(have, list):
            only_tracked = [t for t in want if t not in have]
            only_served = [h for h in have if h not in want]
            lines.append(f"  {key}:")
            for name in only_tracked:
                lines.append(f"    + {name}")
            for name in only_served:
                lines.append(f"    - {name}")
            if not only_tracked and not only_served:
                lines.append("    (same members, different order)")
        else:
            lines.append(f"  {key}: tracked={want!r} served={have!r}")
    return lines


def sync(root: pathlib.Path, *, check: bool = False) -> int:
    """Reconcile what can be reconciled; report the rest. Returns the exit code."""
    ovr_path = overrides_path(root)
    if not ovr_path.exists():
        print(f"no override at {ovr_path} — config.yaml is the served state")
        return 0

    cfg = _load_yaml(root / "config.yaml")
    ovr = _load_yaml(ovr_path)
    d = parity(cfg, ovr)
    if not d:
        print("already in sync")
        return 0

    print(f"drift between config.yaml and {ovr_path}:")
    print("  (+ = config.yaml lists it and the override does not; "
          "- = the override serves it and config.yaml does not)")
    for line in describe(d):
        print(line)

    fixable = [k for k in d if k.startswith(f"harness.{BLOCK}.")]
    if check:
        print("\n--check: not writing.")
        return 1
    if not fixable:
        print("\nnot fixable by rewriting the override: the override is the "
              "authority for every class above, so config.yaml is the file to "
              "edit. Leaving it untouched.")
        return 1

    tracked_ts = (cfg.get("harness") or {}).get(BLOCK) or {}
    served = dict(((ovr.get("harness") or {}).get(BLOCK) or {}))
    for key in fixable:
        name = key.rsplit(".", 1)[1]
        served[name] = tracked_ts[name]
    ovr.setdefault("harness", {})[BLOCK] = served
    # Written whole, the same way `save_tool_overrides` writes it. Every other
    # top-level key (mcp_servers, workers) is carried through untouched — this
    # file also holds the UI's tool disables and the worker switch.
    ovr_path.write_text(yaml.dump(ovr, sort_keys=False, default_flow_style=False))
    print(f"\nwrote {ovr_path}")

    left = parity(cfg, _load_yaml(ovr_path))
    if left:
        print("still drifting, and not this command's to fix "
              "(config.yaml is the file to edit):")
        for line in describe(left):
            print(line)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", default=str(paths.LIVE_CHECKOUT))
    ap.add_argument("--check", action="store_true",
                    help="report drift and exit 1 without writing")
    args = ap.parse_args()
    return sync(pathlib.Path(args.root), check=args.check)


if __name__ == "__main__":
    sys.exit(main())
