"""Regenerate `app/harness/tool_roster.py` from the pool's own `tools/list`.

The roster is the closed set a declared worker capability name is checked
against (#2269). It can only be as fresh as the last time someone asked the
running MCP servers what they serve, which is exactly what this does — it
opens the pool, reads `tools/list`, and rewrites the one file. No turn is
dispatched and nothing is written outside that file.

    cd ~/lloyd && .venvs/lloyd/bin/python -m scripts.refresh_tool_roster

`--check` prints the diff and exits 1 instead of writing, for a caller that
wants to know the snapshot is stale without being handed a modified tree.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "harness" / "tool_roster.py"


def render_roster(names) -> str:
    """The module text for one set of normalised tool names."""
    ordered = sorted({str(n) for n in names if str(n)})
    lines = [
        '"""The tools the MCP pool serves, as a committed snapshot (#2269).',
        "",
        "A worker source's declared capability set is checked against this at",
        "import, so a typo in a declared name is an exception at boot rather",
        "than a tool that quietly stops being reachable — the failure mode an",
        "allow-list has that a deny-list does not. Nothing else in the tree",
        "enumerates the served set without a live pool: discovery happens",
        "inside a turn, after the routers have built their options.",
        "",
        "GENERATED — do not hand-edit:",
        "    cd ~/lloyd && .venvs/lloyd/bin/python -m scripts.refresh_tool_roster",
        "",
        "Two known stalenesses, both deliberate. A tool ADDED upstream is",
        "absent until someone regenerates, so `registered_tool_names()` unions",
        "this with the live discovery universe and a fresh name still passes.",
        "A tool REMOVED upstream stays listed here, which only ever accepts a",
        "declared name that then fails at dispatch — visible, not silent.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "REGISTERED_TOOLS: frozenset[str] = frozenset({",
    ]
    lines += [f'    "{n}",' for n in ordered]
    lines += [
        "})",
        "",
        "#: How many names the snapshot carries, for a drift message that does",
        "#: not have to import the set to state a count.",
        f"REGISTERED_TOOL_COUNT = {len(ordered)}",
        "",
    ]
    return "\n".join(lines)


async def _discover() -> list[str]:
    from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_SERVERS, MCPPool
    from app.harness.policy import normalize_tool_name

    pool = MCPPool(dict(DEFAULT_LLOYD_MCP_SERVERS))
    await pool.open()
    try:
        return sorted({normalize_tool_name(t["name"])
                       for _srv, tools in pool.discovered for t in tools})
    finally:
        await pool.aclose()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report drift and exit 1 without writing")
    args = ap.parse_args(argv)

    names = asyncio.run(_discover())
    if not names:
        print("discovery returned no tools; roster left untouched",
              file=sys.stderr)
        return 1
    text = render_roster(names)
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current == text:
            print(f"roster current ({len(names)} tools)")
            return 0
        print(f"roster is stale: live pool serves {len(names)} tools",
              file=sys.stderr)
        return 1
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} with {len(names)} tools")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
