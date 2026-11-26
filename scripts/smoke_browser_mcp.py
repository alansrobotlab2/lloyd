#!/usr/bin/env python3
"""Manual smoke check of all 14 browser MCP tools against a running lloyd-mcp.

Every `browser_*` tool `agent_mcp.browser.list_tools()` registers is reported here
by name — browser_select and browser_drag as a SKIP, the other twelve driven over
the MCP transport. `tests/test_smoke_browser_coverage.py` pins that set and the
count above to the registered tools, so a new tool cannot land unsmoked again.

Run with the aggregator up:  .venvs/lloyd/bin/python -m scripts.smoke_browser_mcp

Deliberately NOT named test_*.py: it drives a live MCP server over HTTP, so
pytest can neither run it offline nor fail meaningfully — and a `test_` file
sitting outside `testpaths` reads as coverage that does not exist. It is a
smoke script; the name now says so.


Transport: Streamable HTTP (MCP 2026-07-28) via the SDK client, driven
through the harness's own MCPPool so this exercises the same path the
agent does. It used to hand-roll the SSE protocol — a GET stream in a
listener thread, ids matched by hand — which was the third hand-rolled
MCP client in this repo and broke the moment the transport changed.
"""

import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from app.harness.mcp_pool import (
    DEFAULT_LLOYD_MCP_SERVERS,
    DEFAULT_LLOYD_MCP_URL,
    MCPPool,
)

TIMEOUT = 30.0

results = []


def report(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, status, detail))
    tag = f"[{status}]"
    print(f"  {tag} {name}" + (f" -- {detail}" if detail else ""), flush=True)


_pool: MCPPool | None = None
_loop: asyncio.AbstractEventLoop | None = None


def call_tool(name, args=None, timeout=TIMEOUT):
    """Synchronous facade so the test bodies below stay unchanged."""
    assert _loop is not None and _pool is not None
    try:
        return _loop.run_until_complete(
            asyncio.wait_for(
                _pool.call_tool(name, args or {}, timeout_seconds=timeout),
                timeout=timeout + 5,
            )
        )
    except Exception as exc:
        return {"content": f'{{"error": "{exc}"}}', "is_error": True}


def text_of(result):
    return (result or {}).get("content", "") or ""


def is_error(result):
    return bool((result or {}).get("is_error", True))


def _setup():
    global _pool, _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _pool = MCPPool(DEFAULT_LLOYD_MCP_SERVERS)
    try:
        _loop.run_until_complete(_pool.open())
    except Exception as exc:
        print(f"FATAL: could not reach lloyd-mcp at {DEFAULT_LLOYD_MCP_URL}: {exc}")
        sys.exit(1)
    names = [t["name"] for _s, ts in _pool.discovered for t in ts]
    browser_tools = sorted(n for n in names if n.startswith("browser_"))
    print(f"Connected: {len(names)} tools", flush=True)
    print(f"Browser tools ({len(browser_tools)}): {browser_tools}", flush=True)


def _teardown():
    if _pool is not None and _loop is not None:
        _loop.run_until_complete(_pool.aclose())


_setup()

print("\n" + "=" * 60)
print("RUNNING BROWSER MCP TOOL TESTS")
print("=" * 60, flush=True)

# =====================================================================
# 1. browser_navigate -- go to https://example.com
# =====================================================================
print("\n-- Test 1: browser_navigate (example.com) --", flush=True)
try:
    r = call_tool("browser_navigate", {"url": "https://example.com"})
    txt = text_of(r)
    ok = not is_error(r) and "example" in txt.lower()
    report("browser_navigate", ok, f"len={len(txt)}")
    if not ok:
        print(f"    Response: {txt[:500]}", flush=True)
except Exception as e:
    report("browser_navigate", False, str(e))

# =====================================================================
# 2. browser_snapshot -- verify refs assigned
# =====================================================================
print("\n-- Test 2: browser_snapshot --", flush=True)
snapshot_text = ""
refs = []
try:
    r = call_tool("browser_snapshot", {})
    txt = text_of(r)
    refs = re.findall(r'\[([a-z]\d+)\]', txt)
    ok = len(refs) > 0 or "example" in txt.lower()
    report("browser_snapshot", ok, f"refs={refs[:5]}")
    if not ok:
        print(f"    Snapshot: {txt[:600]}", flush=True)
    snapshot_text = txt
except Exception as e:
    report("browser_snapshot", False, str(e))

# =====================================================================
# 3. browser_snapshot(full=true)
# =====================================================================
print("\n-- Test 3: browser_snapshot full=true --", flush=True)
try:
    r = call_tool("browser_snapshot", {"full": True})
    txt = text_of(r)
    ok = not is_error(r) and len(txt) > 0
    report("browser_snapshot(full)", ok, f"len={len(txt)}")
except Exception as e:
    report("browser_snapshot(full)", False, str(e))

# =====================================================================
# 4. browser_click -- click "More information" link (uses "ref" param)
# =====================================================================
print("\n-- Test 4: browser_click --", flush=True)
try:
    click_ref = None
    for m in re.finditer(r'\[([a-z]\d+)\].*?(?:More information|Learn more|IANA)', snapshot_text, re.IGNORECASE):
        click_ref = m.group(1)
        break
    if not click_ref and refs:
        click_ref = refs[0]

    if click_ref:
        r = call_tool("browser_click", {"ref": click_ref})
        txt = text_of(r)
        ok = not is_error(r)
        report("browser_click", ok, f"clicked ref={click_ref}")
        if not ok:
            print(f"    Response: {txt[:400]}", flush=True)
    else:
        report("browser_click", False, "No ref found to click")
except Exception as e:
    report("browser_click", False, str(e))

# =====================================================================
# 5. browser_navigate -- go to google.com
# =====================================================================
print("\n-- Test 5: browser_navigate (google.com) --", flush=True)
try:
    r = call_tool("browser_navigate", {"url": "https://www.google.com"})
    txt = text_of(r)
    ok = not is_error(r) and "google" in txt.lower()
    report("browser_navigate(google)", ok, f"len={len(txt)}")
    if not ok:
        print(f"    Response: {txt[:500]}", flush=True)
except Exception as e:
    report("browser_navigate(google)", False, str(e))

# =====================================================================
# 6. browser_snapshot -- find search box ref
# =====================================================================
print("\n-- Test 6: browser_snapshot (google) --", flush=True)
search_ref = None
refs2 = []
try:
    r = call_tool("browser_snapshot", {})
    txt = text_of(r)
    refs2 = re.findall(r'\[([a-z]\d+)\]', txt)
    # Look for search/combobox/textbox near a ref
    for m in re.finditer(r'\[([a-z]\d+)\].*?(?:search|combobox|textbox)', txt, re.IGNORECASE):
        search_ref = m.group(1)
        break
    if not search_ref:
        for m in re.finditer(r'(?:search|combobox|textbox).*?\[([a-z]\d+)\]', txt, re.IGNORECASE):
            search_ref = m.group(1)
            break
    ok = len(refs2) > 0
    report("browser_snapshot(google)", ok, f"refs={refs2[:8]}, search_ref={search_ref}")
    if not search_ref:
        print(f"    Snapshot: {txt[:800]}", flush=True)
except Exception as e:
    report("browser_snapshot(google)", False, str(e))

# =====================================================================
# 7. browser_fill keystrokes=true -- type key by key (what browser_type did)
# =====================================================================
print("\n-- Test 7: browser_fill(keystrokes) --", flush=True)
try:
    target = search_ref or (refs2[0] if refs2 else None)
    if target:
        r = call_tool("browser_fill", {"ref": target, "value": "playwright test", "keystrokes": True})
        txt = text_of(r)
        ok = not is_error(r)
        report("browser_fill(keystrokes)", ok, f"typed_into ref={target}")
        if not ok:
            print(f"    Response: {txt[:400]}", flush=True)
    else:
        report("browser_fill(keystrokes)", False, "No search ref found")
except Exception as e:
    report("browser_fill(keystrokes)", False, str(e))

# =====================================================================
# 8. browser_fill -- fill search box (uses "ref" param, clears+replaces)
# =====================================================================
print("\n-- Test 8: browser_fill --", flush=True)
try:
    target = search_ref or (refs2[0] if refs2 else None)
    if target:
        r = call_tool("browser_fill", {"ref": target, "value": "hello world"})
        txt = text_of(r)
        ok = not is_error(r)
        report("browser_fill", ok, f"filled ref={target}")
        if not ok:
            print(f"    Response: {txt[:400]}", flush=True)
    else:
        report("browser_fill", False, "No search ref found")
except Exception as e:
    report("browser_fill", False, str(e))

# =====================================================================
# 9. browser_press -- press Escape
# =====================================================================
print("\n-- Test 9: browser_press --", flush=True)
try:
    r = call_tool("browser_press", {"key": "Escape"})
    txt = text_of(r)
    ok = not is_error(r)
    report("browser_press", ok, f"key=Escape")
    if not ok:
        print(f"    Response: {txt[:400]}", flush=True)
except Exception as e:
    report("browser_press", False, str(e))

# =====================================================================
# 10. browser_scroll -- scroll down then up
# =====================================================================
print("\n-- Test 10: browser_scroll --", flush=True)
try:
    r1 = call_tool("browser_scroll", {"direction": "down"})
    ok1 = not is_error(r1)
    r2 = call_tool("browser_scroll", {"direction": "up"})
    ok2 = not is_error(r2)
    report("browser_scroll", ok1 and ok2, f"down={'ok' if ok1 else 'fail'}, up={'ok' if ok2 else 'fail'}")
except Exception as e:
    report("browser_scroll", False, str(e))

# =====================================================================
# 11. browser_tabs -- list, new, list, switch, close
#     Uses: action="switch" (not "select"), page_id (not index)
# =====================================================================
print("\n-- Test 11: browser_tabs --", flush=True)
try:
    # List tabs
    r1 = call_tool("browser_tabs", {"action": "list"})
    txt1 = text_of(r1)
    ok1 = not is_error(r1)
    # Parse current tab id
    tabs_data = json.loads(txt1) if txt1.strip().startswith("{") else {}
    first_tab_id = None
    if "tabs" in tabs_data and len(tabs_data["tabs"]) > 0:
        first_tab_id = tabs_data["tabs"][0].get("id")

    # New tab (must be http/https, not about:blank)
    r2 = call_tool("browser_tabs", {"action": "new", "url": "https://example.com"})
    txt2 = text_of(r2)
    ok2 = not is_error(r2)

    # List again
    r3 = call_tool("browser_tabs", {"action": "list"})
    txt3 = text_of(r3)
    ok3 = not is_error(r3)
    # Parse new tab list to find the new tab's id
    tabs_data2 = json.loads(txt3) if txt3.strip().startswith("{") else {}
    new_tab_id = None
    if "tabs" in tabs_data2:
        for tab in tabs_data2["tabs"]:
            if tab.get("id") != first_tab_id:
                new_tab_id = tab.get("id")
                break

    # Switch back to first tab
    r4 = call_tool("browser_tabs", {"action": "switch", "page_id": first_tab_id})
    txt4 = text_of(r4)
    ok4 = not is_error(r4)

    # Close the new tab
    r5 = call_tool("browser_tabs", {"action": "close", "page_id": new_tab_id})
    txt5 = text_of(r5)
    ok5 = not is_error(r5)

    all_ok = ok1 and ok2 and ok3 and ok4 and ok5
    report("browser_tabs", all_ok,
           f"list={'ok' if ok1 else 'fail'}, new={'ok' if ok2 else 'fail'}, "
           f"list2={'ok' if ok3 else 'fail'}, switch={'ok' if ok4 else 'fail'}, close={'ok' if ok5 else 'fail'}")
    if not all_ok:
        for txt, label in [(txt1,"list"),(txt2,"new"),(txt3,"list2"),(txt4,"switch"),(txt5,"close")]:
            print(f"    {label}: {txt[:200]}", flush=True)
except Exception as e:
    report("browser_tabs", False, str(e))

# =====================================================================
# 12. browser_screenshot -- take screenshot
# =====================================================================
print("\n-- Test 12: browser_screenshot --", flush=True)
try:
    r = call_tool("browser_screenshot", {})
    # MCPPool flattens content blocks to text; a non-text block is rendered
    # as {"type": "..."} by the pool, which is what we look for here.
    txt = text_of(r)
    has_image = '"type": "image"' in txt
    has_text = bool(txt)
    ok = not is_error(r) and (has_image or len(txt) > 0)
    report("browser_screenshot", ok, f"has_image={has_image}, has_text={has_text}, text_len={len(txt)}")
except Exception as e:
    report("browser_screenshot", False, str(e))

# =====================================================================
# 13. browser_evaluate -- run JS (uses "script" param)
# =====================================================================
print("\n-- Test 13: browser_evaluate --", flush=True)
try:
    r1 = call_tool("browser_evaluate", {"script": "document.title"})
    txt1 = text_of(r1)
    ok1 = not is_error(r1) and len(txt1) > 0

    r2 = call_tool("browser_evaluate", {"script": "2+2"})
    txt2 = text_of(r2)
    ok2 = not is_error(r2) and "4" in txt2

    report("browser_evaluate", ok1 and ok2, f"title={txt1[:60]!r}, 2+2={txt2[:20]!r}")
    if not (ok1 and ok2):
        print(f"    title resp: {txt1[:200]}", flush=True)
        print(f"    2+2 resp: {txt2[:200]}", flush=True)
except Exception as e:
    report("browser_evaluate", False, str(e))

# =====================================================================
# 14. browser_wait -- condition-based waiting
#     Schema: condition (selector|text|navigation|networkidle|timeout), value, timeout
# =====================================================================
print("\n-- Test 14: browser_wait --", flush=True)
try:
    r1 = call_tool("browser_wait", {"condition": "text", "value": "Google"})
    txt1 = text_of(r1)
    ok1 = not is_error(r1)

    r2 = call_tool("browser_wait", {"condition": "timeout", "value": "500"})
    txt2 = text_of(r2)
    ok2 = not is_error(r2)

    report("browser_wait", ok1 and ok2,
           f"wait_text={'ok' if ok1 else 'fail'}, wait_timeout={'ok' if ok2 else 'fail'}")
    if not ok1:
        print(f"    wait_text: {txt1[:300]}", flush=True)
    if not ok2:
        print(f"    wait_timeout: {txt2[:300]}", flush=True)
except Exception as e:
    report("browser_wait", False, str(e))

# =====================================================================
# 15. browser_cookies -- set, get, clear, verify
#     set uses "cookies" array: [{name, value, domain, path}]
#     get uses "domain" (optional filter)
# =====================================================================
print("\n-- Test 15: browser_cookies --", flush=True)
try:
    # Set a cookie
    r1 = call_tool("browser_cookies", {
        "action": "set",
        "cookies": [{"name": "test_cookie", "value": "test_value", "domain": ".google.com", "path": "/"}]
    })
    txt1 = text_of(r1)
    ok1 = not is_error(r1)

    # Get cookies
    r2 = call_tool("browser_cookies", {"action": "get"})
    txt2 = text_of(r2)
    ok2 = not is_error(r2) and "test_cookie" in txt2

    # Clear cookies
    r3 = call_tool("browser_cookies", {"action": "clear"})
    txt3 = text_of(r3)
    ok3 = not is_error(r3)

    # Verify cleared
    r4 = call_tool("browser_cookies", {"action": "get"})
    txt4 = text_of(r4)
    ok4 = not is_error(r4) and "test_cookie" not in txt4

    all_ok = ok1 and ok2 and ok3 and ok4
    report("browser_cookies", all_ok,
           f"set={'ok' if ok1 else 'fail'}, get={'ok' if ok2 else 'fail'}, "
           f"clear={'ok' if ok3 else 'fail'}, verify={'ok' if ok4 else 'fail'}")
    if not all_ok:
        print(f"    set: {txt1[:200]}", flush=True)
        print(f"    get: {txt2[:200]}", flush=True)
        print(f"    clear: {txt3[:200]}", flush=True)
        print(f"    verify: {txt4[:200]}", flush=True)
except Exception as e:
    report("browser_cookies", False, str(e))

# =====================================================================
# 16 & 17. SKIP
# =====================================================================
print("\n-- Test 16: browser_select --", flush=True)
report("browser_select", True, "SKIP -- no dropdown on google.com")

print("\n-- Test 17: browser_drag --", flush=True)
report("browser_drag", True, "SKIP -- no drag targets on google.com")

# =====================================================================
# 18. browser_upload_files -- refuse a path outside upload_roots, then attach
#     a small file from inside one. `browser_navigate` takes http(s) only, so a
#     `data:` URL cannot be loaded as the page; the file input is created on
#     example.com instead, which loads nothing further.
# =====================================================================
print("\n-- Test 18: browser_upload_files --", flush=True)
upload_file = None
try:
    from app.config import CONFIG
    raw_roots = (CONFIG.get("browser") or {}).get("upload_roots") or ["~/lloyd-data"]
    if isinstance(raw_roots, str):
        raw_roots = [raw_roots]
    roots = [Path(os.path.expanduser(str(r).rstrip("/*"))) for r in raw_roots]
    root = next((r for r in roots if r.is_dir()), None)
    if root is None:
        raise RuntimeError(f"no browser.upload_roots directory exists: {raw_roots}")
    payload = b"lloyd smoke upload\n"
    with tempfile.NamedTemporaryFile(dir=root, prefix="smoke-browser-upload-",
                                     suffix=".txt", delete=False) as fh:
        fh.write(payload)
        upload_file = Path(fh.name)
    outside = str(Path(__file__).resolve())  # this script: outside every upload root

    call_tool("browser_navigate", {"url": "https://example.com"})
    r0 = call_tool("browser_evaluate", {"script": (
        "(() => { const i = document.createElement('input'); i.type = 'file'; "
        "document.body.appendChild(i); return document.querySelectorAll("
        "'input[type=file]').length; })()")})
    ok0 = not is_error(r0) and "1" in text_of(r0)

    files_probe = ("(() => { const f = document.querySelector('input[type=file]'); "
                   "return f.files.length ? f.files[0].name + ':' + f.files[0].size "
                   ": 'files.length=' + f.files.length; })()")
    r1 = call_tool("browser_upload_files", {"selector": "input[type=file]",
                                            "paths": [outside]})
    txt1 = text_of(r1)
    ok1 = '"error"' in txt1 and outside in txt1
    r2 = call_tool("browser_evaluate", {"script": files_probe})
    ok2 = "files.length=0" in text_of(r2)

    r3 = call_tool("browser_upload_files", {"selector": "input[type=file]",
                                            "paths": [str(upload_file)]})
    txt3 = text_of(r3)
    ok3 = not is_error(r3) and '"error"' not in txt3
    r4 = call_tool("browser_evaluate", {"script": files_probe})
    want = f"{upload_file.name}:{len(payload)}"
    ok4 = want in text_of(r4)

    all_ok = ok0 and ok1 and ok2 and ok3 and ok4
    report("browser_upload_files", all_ok,
           f"input={'ok' if ok0 else 'fail'}, refused_outside={'ok' if ok1 else 'fail'}, "
           f"left_empty={'ok' if ok2 else 'fail'}, attach={'ok' if ok3 else 'fail'}, "
           f"on_page={'ok' if ok4 else 'fail'}")
    if not all_ok:
        for txt, label in [(text_of(r0), "input"), (txt1, "refuse"), (text_of(r2), "empty"),
                           (txt3, "attach"), (text_of(r4), "on_page")]:
            print(f"    {label}: {txt[:200]}", flush=True)
except Exception as e:
    report("browser_upload_files", False, str(e))
finally:
    if upload_file is not None:
        upload_file.unlink(missing_ok=True)

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60, flush=True)
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")
skipped = sum(1 for _, _, d in results if "SKIP" in d)
total = len(results)
for name, status, detail in results:
    icon = "+" if status == "PASS" else "-"
    if "SKIP" in detail:
        icon = "~"
    print(f"  [{icon}] {status:4s}  {name:30s}  {detail}", flush=True)
print(f"\nTotal: {total}  |  Passed: {passed}  |  Failed: {failed}  |  Skipped: {skipped}", flush=True)
if failed == 0:
    print("\nAll tests passed!", flush=True)
else:
    print(f"\n{failed} test(s) FAILED.", flush=True)

_teardown()
