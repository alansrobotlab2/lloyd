"""Which prompt component differs between two recorded model requests (#581).

    python scripts/meta_review/prompt_diff.py <request_id_a> <request_id_b> [--json]
    python scripts/meta_review/prompt_diff.py --session <session_id> [--last N]

Reads the content-addressed manifest lines `app/component_manifest.py` writes and
prints every position whose digest differs, in position order: the named
system-prompt components first, then `prefetch`, the `tools` array, each tool
definition, and each message index. That ordering is the answer to "was it just
my message, or did the model get a different tool?" — a prefix-cache miss can be
blamed on a position only if the position is named.

`--session` diffs consecutive recorded requests of one session, which is how a
mid-turn prefix cliff is found: take the session whose `prefix_misses` series
jumps at iteration N, and the request pair straddling N names the component that
moved. `app/prefix_miss.py` is the other instrument for that claim and says
nothing about which component — this is the half that does.

Nothing here sends anything or writes anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.component_manifest import (  # noqa: E402
    diff_manifests, format_diff, read_manifest, store_root,
)


def _iter_lines(store: Path):
    manifests = store / "manifests"
    if not manifests.is_dir():
        return
    for path in sorted(manifests.glob("*.ndjson"), reverse=True):
        try:
            with open(path, encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        yield json.loads(raw)
                    except ValueError:
                        continue
        except OSError:
            continue


def session_pairs(session_id: str, *, store: Path, last: int = 0):
    """Consecutive request pairs of one session, oldest first."""
    rows = [ln for ln in _iter_lines(store) if ln.get("session_id") == session_id]
    rows.sort(key=lambda ln: (ln.get("iteration") is None,
                              ln.get("iteration") or 0, ln.get("ts") or 0))
    if last:
        rows = rows[-(last + 1):]
    return list(zip(rows, rows[1:]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("request_id_a", nargs="?", default="")
    parser.add_argument("request_id_b", nargs="?", default="")
    parser.add_argument("--session", default="", help="diff consecutive requests of one session")
    parser.add_argument("--last", type=int, default=0, help="with --session: only the last N pairs")
    parser.add_argument("--store", default="", help="manifest store root (default: the configured one)")
    parser.add_argument("--json", action="store_true", help="emit the structured diff")
    args = parser.parse_args(argv)

    store = Path(args.store) if args.store else store_root()

    if args.session:
        pairs = session_pairs(args.session, store=store, last=args.last)
        if not pairs:
            print(f"no manifest pair for session {args.session!r} under {store}",
                  file=sys.stderr)
            return 1
        for a, b in pairs:
            out = (json.dumps(diff_manifests(a, b), indent=2) if args.json
                   else format_diff(a, b))
            print(out)
        return 0

    if not args.request_id_a or not args.request_id_b:
        parser.error("pass two request ids, or --session")
    a = read_manifest(args.request_id_a, store=store)
    b = read_manifest(args.request_id_b, store=store)
    missing = [rid for rid, line in ((args.request_id_a, a), (args.request_id_b, b))
               if line is None]
    if missing:
        # A missing record is reported as missing, never as an empty diff: an
        # empty diff would read as "nothing changed between these two requests",
        # which is a verdict about a pair that was never compared.
        print(f"no manifest line for {', '.join(missing)} under {store}",
              file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(diff_manifests(a, b), indent=2))
    else:
        print(format_diff(a, b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
