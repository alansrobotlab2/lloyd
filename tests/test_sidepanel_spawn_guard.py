"""The side panel's non-checkable-site blocklist is held by the service worker,
not by the button (#1765).

`shouldSpawnSession` (chrome-extension/src/background/url.ts) refuses Google
hosts and non-video YouTube pages. Until 2026-09-28 its only reader was
`pushFocus`, whose `canCheck` flag reached nothing but the button's `disabled`
attribute; `handleManualCheck`, which mints the session, checked the URL scheme
alone. A stale panel or a message not sent by the button could check Gmail.

No TS runner covers chrome-extension/, so the wiring is pinned at source level
(the style of tests/test_browser_panel.py's guard-wiring test), and the one
predicate both readers share is run for real under node, which strips types
natively.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SW = ROOT / "chrome-extension/src/background/service-worker.ts"
URL_TS = ROOT / "chrome-extension/src/background/url.ts"


def _fn_body(src: str, name: str) -> str:
    return src.split(f"async function {name}(", 1)[1].split("\nasync function ", 1)[0] \
        .split("\nfunction ", 1)[0]


def test_the_check_refuses_before_taking_the_tab_lock_and_still_pushes_focus():
    body = _fn_body(SW.read_text(encoding="utf-8"), "handleManualCheck")
    guard = body.index("canCheckUrl(tab.url)")
    assert guard < body.index("withTabLock"), "refuse before minting anything"
    assert guard < body.index("spawnSession("), "refuse before the kickoff"
    refused = body[guard:body.index("withTabLock")]
    assert "await pushFocus(windowId, tabId)" in refused and "return" in refused, (
        "a refused check must re-push focus so the panel settles its pending state")


def test_the_button_and_the_check_read_one_predicate():
    src = SW.read_text(encoding="utf-8")
    assert "canCheck = canCheckUrl(tab.url)" in _fn_body(src, "pushFocus")
    assert "canCheckUrl(" in _fn_body(src, "handleManualCheck")
    # Nothing in the service worker decides checkability on its own.
    assert "shouldSpawnSession" not in src


URLS = {
    "https://google.com/": False,
    "https://mail.google.com/mail/u/0/#inbox": False,
    "https://www.youtube.com/feed/subscriptions": False,
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ": True,
    "https://youtu.be/dQw4w9WgXcQ": True,
    "https://www.youtube.com/shorts/abcdefghijk": True,
    "https://www.youtube.com/shorts/abc": False,
    "https://example.com/article": True,
    "chrome://extensions": False,
    "": False,
}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_can_check_url_refuses_exactly_what_should_spawn_session_refuses(tmp_path):
    script = (f"import {{ canCheckUrl, shouldSpawnSession }} from {json.dumps(URL_TS.as_uri())};\n"
              f"const urls = {json.dumps(list(URLS))};\n"
              "console.log(JSON.stringify(urls.map(u => [u, canCheckUrl(u),"
              " /^https?:/i.test(u) && shouldSpawnSession(u)])));\n")
    probe = tmp_path / "probe.mjs"
    probe.write_text(script, encoding="utf-8")
    out = subprocess.run(["node", str(probe)], capture_output=True, text=True, timeout=60)
    if "Unknown file extension" in out.stderr:
        pytest.skip("this node cannot strip TypeScript types")
    assert out.returncode == 0, out.stderr
    rows = json.loads(out.stdout)
    assert len(rows) == len(URLS), "positive control: every URL was judged"
    for url, can, spawn in rows:
        assert can == URLS[url], url
        assert can == spawn, f"{url}: the button and the check disagree"
