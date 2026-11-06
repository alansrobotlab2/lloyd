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


# ── #3741: the tab-switch listener must DEFER to pushFocus, not re-describe it ──
#
# `chrome.tabs.onActivated` is the one listener that used to build its own
# `focus-session` literal instead of calling `pushFocus`, and the literal named
# `type/windowId/tabId/sessionKey/url/title` with no `canCheck` — so with the panel open,
# activating an already-loaded checkable tab made the panel read `Boolean(undefined)` as
# false (`web/src/sidepanel/SidePanelApp.tsx:69`) and the button at :164 went dead while
# the copy at :177 claimed "No checkable page here". `pushFocus` already hydrates, reads
# the mapping AND computes the flag, so the second description of the same message was the
# whole defect: any field it forgot was a field the tab-switch silently dropped.
#
# `_fn_body` cannot reach this one: it splits on `async function {name}(` and the listener
# is an anonymous arrow handed to `addListener`, so no name ever reaches it — the slice is
# textual, from the registration to the next registration, the style
# `tests/test_automod_spec.py` already uses for the shipped-bundle pin.

def _listener_slice(text: str) -> str:
    at = text.index("tabs.onActivated.addListener")
    nxt = text.find("addListener(", at + len("tabs.onActivated.addListener"))
    assert nxt != -1, "no listener after onActivated, so the slice has no end"
    return text[at:nxt]


def test_the_tab_switch_listener_defers_to_push_focus():
    """#3741 clause 1: the listener body is exactly `await pushFocus(windowId, tabId)`.

    The three helpers it used to call by hand are what made the omission possible, so the
    pin is their absence, not just the call's presence: a listener that hydrates and
    reads the mapping before delegating is one `await` away from re-introducing a field
    that only `pushFocus` knows about. And `focus-session` appears at exactly three places
    in the file — the `FocusMessage` field, `pushFocus`'s comment, and `pushFocus`'s one
    construction — which is the same fact stated as a count so a fourth hand-built literal
    anywhere in the file is caught whether or not it lands in this listener.
    """
    src = SW.read_text(encoding="utf-8")
    slice_ = _listener_slice(src)
    body = slice_[slice_.index("{", slice_.index("=>")) + 1:slice_.rindex("}")].strip()
    assert body == "await pushFocus(windowId, tabId)", slice_
    for helper in ("hydrate", "getMapping", "pushToPanel"):
        assert helper not in slice_, f"{helper} is back inside the listener: {slice_}"
    assert src.count('focus-session') == 3, [
        line for line in src.splitlines() if "focus-session" in line]


def test_the_tab_switch_never_fabricates_can_check_a_second_time():
    """#3741 clause 2: the flag is computed in exactly one place, and the slice reaches it.

    The slice must name `pushFocus(` and contain neither `canCheck` nor `focus-session`:
    a listener that passes the flag through by hand would fix the symptom and keep the
    duplicate description that caused it. `canCheck = canCheckUrl(` appearing exactly once
    is the other half — and it is asserted *inside* `pushFocus`'s body, because a
    second computation in some future helper would also satisfy a bare count.
    """
    src = SW.read_text(encoding="utf-8")
    slice_ = _listener_slice(src)
    assert "pushFocus(" in slice_, slice_
    for forbidden in ("canCheck", "focus-session"):
        assert forbidden not in slice_, f"{forbidden} fabricated in the listener: {slice_}"
    assert src.count("canCheck = canCheckUrl(") == 1, src
    assert "canCheck = canCheckUrl(" in _fn_body(src, "pushFocus"), (
        "the one computation is not inside pushFocus, so the delegation does not carry it")


# ── #3854 / #3645: the mint waits on the kickoff, and both stay under one lock ──────────
#
# `spawnSession` used to write `setMapping` + `setCanonical` and only THEN await the kickoff
# POST. A POST refused out of the self-mod drain window enqueues no turn, so the tab was left
# mapped to a session with no turn and a canonical that matched, and `handleManualCheck`'s first
# guard re-focused that empty session on every later press while the tab stayed on that URL.
#
# The behaviour is driven over the shipped bytes by `tests/test_sidepanel_focus_headless.py`
# (press the button, answer `/message/stream` with a 503, read what the panel was handed and
# what the store holds). What these two nodes add is the SHAPE that behaviour depends on, at
# source level, because the drive cannot express it: that the refused path has no committed
# state to strand — as opposed to committing and clearing, where the clearing helper matters,
# since `clearTab` drops `lastCommitted` and `clearMapping` does NOT
# (`chrome-extension/src/background/tab-session-map.ts`, where keeping it is the documented
# design for a navigation) — and that the mint still runs inside the one lock
# `handleManualCheck` takes rather than a second one of its own.

def test_a_refused_kickoff_leaves_no_committed_state_to_clear():
    """#3645 clauses 1 and 3: the commit follows the proof, so nothing needs unwinding.

    Both `setMapping` and `setCanonical` index AFTER the awaited `fireKickoff`, which is the
    reorder the item names as its preferred variant — and `index` returns the FIRST occurrence,
    so the two comparisons are also the proof that neither appears before it. The mint still
    precedes the kickoff (there is a session to post against), and the failure path clears
    nothing because it wrote nothing: `clearMapping(` is absent from `spawnSession` and its one
    remaining call site in the file is `handleNavigation`'s, where keeping `lastCommitted` is
    the point. Had the clearing variant been taken instead, the helper in the rejection path
    would have to be `clearTab` — the one that drops the mapping AND `lastCommitted` — and this
    node reddens either way for a `clearMapping` on the failure path, because a stranded
    canonical reproduces the silent no-op one press later.

    The tail `pushFocus` is pinned as unreachable-by-`return`: the region after the
    `kickoffAnswered` guard contains the push and no return, which is what leaves the panel on
    not-checked with the button live instead of holding a spinner for a session no one started.
    """
    src = SW.read_text(encoding="utf-8")
    body = _fn_body(src, "spawnSession")

    kickoff = body.index("await fireKickoff(")
    assert body.index("createBrowserSession(") < kickoff, "the mint still precedes the kickoff"
    assert body.index("await setMapping(") > kickoff, (
        "the tab is mapped before the kickoff POST is proven again — a refused one then leaves "
        "a mapped session with no turn")
    assert body.index("await setCanonical(") > kickoff, (
        "the canonical is committed before the proof, which is the half that closes the URL to "
        "retries via handleManualCheck's first guard")

    assert "clearMapping(" not in body, (
        "clearMapping drops the mapping and KEEPS lastCommitted by design, so it can never be "
        f"the refused-kickoff cleanup: {body}")
    assert src.count("clearMapping(") == 1, [
        line for line in src.splitlines() if "clearMapping(" in line]

    tail = body[body.index("if (kickoffAnswered"):]
    assert "await pushFocus(" in tail, "the panel is no longer told anything on either path"
    assert "return" not in tail, "the failure path returns before the focus push it must make"


def test_the_mint_still_runs_inside_the_lock_the_check_already_holds():
    """#3645 clause 4: one lock, taken by the caller, covers both the commit and the kickoff.

    `spawnSession` names no `withTabLock` — the file keeps exactly two call sites, the
    navigation handler and the manual check — and the commit and the kickoff are both statements
    of `spawnSession`, which is itself called from inside `handleManualCheck`'s locked region.
    That region is sliced out rather than assumed: the callback ends at the first `\n  })` after
    the lock, and the `spawnSession` call has to be inside it. This is the shape that makes two
    presses one session — a second press queues, then reads a mapping whose canonical matches and
    re-focuses — and it is what `test_two_presses_while_the_first_kickoff_is_in_flight_mint_one_session`
    measures over the shipped bytes; neither alone is enough, because a lock grabbed inside
    `spawnSession` would pass the drive only until some other caller appeared.
    """
    src = SW.read_text(encoding="utf-8")
    spawn = _fn_body(src, "spawnSession")

    assert "withTabLock" not in spawn, "spawnSession introduced a lock of its own"
    assert "await setMapping(" in spawn and "await fireKickoff(" in spawn, (
        "the commit or the kickoff moved out of spawnSession, so the containment below is about "
        "a call that no longer covers them")

    check = _fn_body(src, "handleManualCheck")
    lock = check.index("await withTabLock(")
    end = check.index("\n  })", lock)
    region = check[lock:end]
    assert "spawnSession(" in region, (
        "spawnSession is called outside the lock, so a second press during an in-flight kickoff "
        "mints a second session")
    assert src.count("withTabLock(") == 2, [
        line for line in src.splitlines() if "withTabLock(" in line]
