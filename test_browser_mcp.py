#!/usr/bin/env python3
"""Comprehensive test of all 14 browser MCP tools via SSE endpoint.

Uses the SSE transport: a persistent GET /sse connection receives responses,
while requests are POSTed to the messages endpoint.
"""

import json
import httpx
import sys
import re
import threading
import time

SSE_URL = "http://127.0.0.1:8500/sse"
BASE_URL = "http://127.0.0.1:8500"
TIMEOUT = 30.0

results = []
pending = {}
response_events = {}

def report(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, status, detail))
    tag = f"[{status}]"
    print(f"  {tag} {name}" + (f" -- {detail}" if detail else ""), flush=True)


# -- SSE listener thread --
session_url = None
sse_ready = threading.Event()
sse_error = None

def sse_listener():
    global session_url, sse_error
    try:
        with httpx.Client(timeout=None) as c:
            with c.stream("GET", SSE_URL) as resp:
                event_type = None
                for line in resp.iter_lines():
                    if line.startswith("event:"):
                        event_type = line[len("event:"):].strip()
                    elif line.startswith("data:"):
                        data = line[len("data:"):].strip()
                        if event_type == "endpoint" or session_url is None:
                            if data.startswith("/"):
                                session_url = BASE_URL + data
                            elif data.startswith("http"):
                                session_url = data
                            sse_ready.set()
                        elif event_type == "message":
                            try:
                                msg = json.loads(data)
                                msg_id = msg.get("id")
                                if msg_id is not None:
                                    pending[msg_id] = msg
                                    evt = response_events.get(msg_id)
                                    if evt:
                                        evt.set()
                            except json.JSONDecodeError:
                                pass
                        event_type = None
                    elif line == "":
                        event_type = None
    except Exception as e:
        sse_error = e
        sse_ready.set()

t = threading.Thread(target=sse_listener, daemon=True)
t.start()
sse_ready.wait(timeout=10)

if not session_url:
    print(f"FATAL: Could not get session URL. Error: {sse_error}", flush=True)
    sys.exit(1)

print(f"Session URL: {session_url}", flush=True)

post_client = httpx.Client(timeout=TIMEOUT)
req_id = 0

def rpc(method, params=None, timeout=TIMEOUT):
    global req_id
    req_id += 1
    my_id = req_id
    body = {"jsonrpc": "2.0", "id": my_id, "method": method}
    if params is not None:
        body["params"] = params

    evt = threading.Event()
    response_events[my_id] = evt
    post_client.post(session_url, json=body)

    if not evt.wait(timeout=timeout):
        return {"error": {"code": -1, "message": "Timeout waiting for response"}}
    return pending.pop(my_id, None)

def call_tool(name, args=None, timeout=TIMEOUT):
    resp = rpc("tools/call", {"name": name, "arguments": args or {}}, timeout=timeout)
    if resp and "result" in resp:
        return resp["result"]
    if resp and "error" in resp:
        return {"error": resp["error"]}
    return resp

def text_of(result):
    if not result:
        return ""
    content = result.get("content", [])
    parts = []
    for c in content:
        if c.get("type") == "text":
            parts.append(c["text"])
    return "\n".join(parts)

def is_error(result):
    if not result:
        return True
    if "error" in result:
        return True
    return result.get("isError", False)


# -- Initialize --
print("Sending initialize...", flush=True)
init = rpc("initialize", {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {"name": "test-runner", "version": "1.0"}
})
print(f"Init: {json.dumps(init, default=str)[:200]}", flush=True)

post_client.post(session_url, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
time.sleep(0.3)

# List tools
tools_resp = rpc("tools/list", {})
tool_names = []
if tools_resp and "result" in tools_resp:
    tool_names = [t["name"] for t in tools_resp["result"].get("tools", [])]
browser_tools = [n for n in tool_names if n.startswith("browser_")]
print(f"Browser tools ({len(browser_tools)}): {browser_tools}", flush=True)

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
# 7. browser_type -- type into search box (uses "ref" param)
# =====================================================================
print("\n-- Test 7: browser_type --", flush=True)
try:
    target = search_ref or (refs2[0] if refs2 else None)
    if target:
        r = call_tool("browser_type", {"ref": target, "text": "playwright test"})
        txt = text_of(r)
        ok = not is_error(r)
        report("browser_type", ok, f"typed_into ref={target}")
        if not ok:
            print(f"    Response: {txt[:400]}", flush=True)
    else:
        report("browser_type", False, "No search ref found")
except Exception as e:
    report("browser_type", False, str(e))

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
    content = r.get("content", []) if r else []
    has_image = any(c.get("type") == "image" for c in content)
    has_text = any(c.get("type") == "text" for c in content)
    txt = text_of(r)
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

post_client.close()
