"""#4348: the browser smoke script covers exactly the browser tools the aggregator advertises.

`scripts/smoke_browser_mcp.py` drives a live lloyd-mcp over HTTP, so pytest cannot run
it; what pytest CAN check is the script's source against `agent_mcp.browser.list_tools()`.
#4236 landed `browser_upload_files` and the script never named it while its docstring
still said "all 14", so the advertised surface under-reported for a week with nothing
red. "Covered" is a `browser_*` name the script passes to `call_tool(...)` or to
`report(...)` — `browser_select` and `browser_drag` are reported as SKIPs, which is
coverage of the name, not of the behaviour, and the docstring says so.
"""

import asyncio
import re
from pathlib import Path

from agent_mcp import browser as bm

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "smoke_browser_mcp.py"

_COVERED = re.compile(r"""\b(?:call_tool|report)\(\s*["'](browser_[a-z_]+)""")
_ADVERTISED = re.compile(r"\ball (\d+) browser MCP tools\b")


def registered_browser_tools() -> set[str]:
    return {t.name for t in asyncio.run(bm.list_tools()) if t.name.startswith("browser_")}


def covered_browser_tools(src: str) -> set[str]:
    return set(_COVERED.findall(src))


def test_the_smoke_script_covers_every_registered_browser_tool():
    src = SCRIPT.read_text(encoding="utf-8")
    registered = registered_browser_tools()
    assert len(registered) >= 14, f"only {len(registered)} browser tools registered: {sorted(registered)}"
    covered = covered_browser_tools(src)
    assert covered == registered, (
        f"smoke script misses {sorted(registered - covered)} and names unregistered "
        f"{sorted(covered - registered)}")


def test_the_upload_step_is_driven_not_skipped():
    src = SCRIPT.read_text(encoding="utf-8")
    assert re.search(r"""call_tool\(\s*["']browser_upload_files["']""", src), (
        "browser_upload_files is reported but never called over the MCP transport")
    assert re.search(r"""report\(\s*["']browser_upload_files["']\s*,\s*all_ok""", src), (
        "the upload step does not report under the name browser_upload_files")
    assert "upload_roots" in src and "input[type=file]" in src


def test_the_docstring_count_is_the_registered_count():
    src = SCRIPT.read_text(encoding="utf-8")
    m = _ADVERTISED.search(src)
    assert m, "the script's docstring no longer states how many browser tools it covers"
    assert int(m.group(1)) == len(registered_browser_tools()), (
        f"docstring advertises {m.group(1)} browser tools; "
        f"{len(registered_browser_tools())} are registered")
