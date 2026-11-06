"""#3772: the shipped extension bundle is DRIVEN, not read.

Everything else in the suite that speaks about `chrome-extension/dist/service-worker.js`
reads it as text: `tests/test_chrome_extension_manifest_tracked.py` harvests literals from
`chrome-extension/src/` and requires each one in the bundle, and
`tests/test_automod_spec.py` slices the bundle and follows the awaited callee into the
compiled function. Those pins can only ever say a string is present. #3568 was exactly the
defect a string cannot settle: the `chrome.tabs.onActivated` listener built its own
`focus-session` literal, so activating an already-loaded checkable tab with the panel open
delivered a message whose `canCheck` was absent, and the panel's `Boolean(msg.canCheck)`
(`web/src/sidepanel/SidePanelApp.tsx:69`) disabled the button. The bundle contained the
string `canCheck` the whole time — `grep -c canCheck` reads `1` both before and after the
#3741 fix — because the one hit was `pushFocus`'s own send site, which the tab-switch chunk
never reached.

So this file runs the bytes. A node child installs a stub `chrome` global covering the ten
`chrome.X.Y` members the bundle calls, `import()`s the TRACKED bundle as an ES module, and
drives the #3568 sequence in order: register a port named `lloyd-sidepanel`, hand it to the
captured `runtime.onConnect` listener, send `{type: "panel-ready", windowId, tabId}`, and
only THEN fire the captured `tabs.onActivated` listener — panel connected first is the repro
condition. What comes back is what `port.postMessage` was handed, as data.

Five things the FOCUS drive measures, one per test below (the three PRESS tests appended at the
foot of the file press the button instead of activating a tab, and are introduced there):

  * the import. Seven listeners register at import time. A stub missing either
    `webNavigation` member throws a `TypeError` in the middle of `import()`, which is why the
    stub is enumerated from the bundle's call sites rather than from the source.
  * the flag on a checkable tab: `type: "focus-session"`, `canCheck: true`.
  * the flag on Gmail: the same shape with `canCheck: false`, from the SAME process, which is
    what makes it a computation rather than a constant.
  * the field set: with `chrome.storage.session.get` seeded with `{"tab:7": {…}}` the push
    carries that `sessionKey`, `url` and `title`; with the store empty the same tab arrives
    with `sessionKey: null` and `canCheck` still `true`. `url` and `title` are `undefined`
    there, and `JSON.stringify` drops those keys entirely, so the drive reports each message
    as its own key list plus a value map with `undefined` replaced by `"__undefined__"` — the
    seven names `pushFocus` writes are then visible on every leg, seeded or not.
  * the control: the identical drive against a temporary copy whose one `canCheck:` send site
    has been replaced delivers no `canCheck` key at all, on either leg. Without that leg the
    two flag assertions are green by construction, because nothing else in the file proves the
    harness can see the field it asserts.

The control removes the send site by SHAPE (`canCheck:` plus an identifier plus `})`), never
by identifier: `architecture/browser-side-panel.md` records that a rebuild renames
minifier-local identifiers across the whole bundle, and today's site reads `canCheck:s}` only
until the next `vite build`. The real bundle is never written to — the control's mutant is a
copy under `tmp_path`, which is also why the drive reports the sha256 of the bytes it actually
imported, so "asserted against `dist/`, not the TypeScript source" is a reading rather than an
intention.

The tabs, the window id, the port name and the seed all travel to the child as one `FIXTURE`
JSON value built from the constants below, so the strings asserted here are the strings the
drive served — no literal is written twice.

No network: the bundle's `fetch` is stubbed and the drive asserts zero calls. No TS runner is
involved — `web/vitest.config.ts` `include` reaches only `web/src/**`, so `chrome-extension/`
still has no JS-side runner and `tests/test_sidepanel_spawn_guard.py`'s header stays true; this
is that file's pytest + node-subprocess route, which is the route a round can land.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "chrome-extension" / "dist" / "service-worker.js"
TS_SOURCE = REPO / "chrome-extension" / "src" / "background" / "service-worker.ts"

WINDOW_ID = 3
CHECKABLE_TAB = 7
GMAIL_TAB = 8
UNMAPPED_TAB = 9
CHECKABLE_URL = "https://example.com/article"
GMAIL_URL = "https://mail.google.com/mail/u/0/#inbox"
UNMAPPED_URL = "https://example.com/orphan"
CHECKABLE_TITLE = "An Article"
GMAIL_TITLE = "Gmail Inbox"
UNMAPPED_TITLE = "An Unmapped Tab"
SEED_KEY = "ext_seed"
PANEL_PORT = "lloyd-sidepanel"
DECOY_PORT = "some-other-panel"

#: The tab the PRESS drive below checks, and what `chrome.tabs.get` answers for it. Not one of
#: tabs 7-9: those are the tabs the focus legs activate, and a leg that mints has to start with
#: no mapping and no canonical of its own, so the store assertions have one clean subject.
PRESS_TAB = 11
PRESS_URL = "https://example.com/press"
PRESS_TITLE = "A Pressed Page"

#: What `chrome.tabs.get` answers for each tab the drive activates. Tab 7 is checkable and
#: mapped, tab 8 is Gmail, tab 9 is checkable but never mapped — the mapped/unmapped pair.
#: Tab 11 is the press drive's subject: checkable, and never seeded with anything.
TABS = {
    CHECKABLE_TAB: {"id": CHECKABLE_TAB, "windowId": WINDOW_ID,
                    "url": CHECKABLE_URL, "title": CHECKABLE_TITLE},
    GMAIL_TAB: {"id": GMAIL_TAB, "windowId": WINDOW_ID,
                "url": GMAIL_URL, "title": GMAIL_TITLE},
    UNMAPPED_TAB: {"id": UNMAPPED_TAB, "windowId": WINDOW_ID,
                   "url": UNMAPPED_URL, "title": UNMAPPED_TITLE},
    PRESS_TAB: {"id": PRESS_TAB, "windowId": WINDOW_ID,
                "url": PRESS_URL, "title": PRESS_TITLE},
}

#: The session key the press drive's `POST /api/sessions/create` stub mints. Deliberately not
#: `SEED_KEY`: "the push carries the key THIS press minted" must not be satisfiable by a mapping
#: the harness hydrated out of the seed.
MINTED_KEY = "sk_driven"

#: What the press drive's `POST /api/message/stream` answers with. 503 is the self-mod drain
#: window `chrome-extension/src/background/lloyd-client.ts` names, and `fireKickoff` THROWS on
#: it (`if (!res.ok) throw …`) rather than resolving quietly — that throw is the event #3645 is
#: about.
STREAM_ANSWERED = 200
STREAM_REFUSED = 503

#: The names `pushFocus` writes into the object literal it hands `postMessage`, in the order it
#: writes them. Measured on EVERY leg of the drive — including the legs where `url` and `title`
#: hold `undefined`, which the drive's key list still shows and a raw `JSON.stringify` would
#: have dropped.
FOCUS_FIELD_SET = ["type", "windowId", "tabId", "sessionKey", "url", "title", "canCheck"]

#: The listeners the bundle registers while it is being imported, sorted. `webNavigation` is in
#: here because `chrome.webNavigation.onCommitted.addListener(k, {url: [...]})` runs at import
#: time: a stub without it fails the `import()` itself.
IMPORT_TIME_LISTENERS = [
    "runtime.onConnect",
    "runtime.onInstalled",
    "tabs.onActivated",
    "tabs.onRemoved",
    "tabs.onUpdated",
    "webNavigation.onCommitted",
    "webNavigation.onHistoryStateUpdated",
]

#: The single site that puts `canCheck` into a message: the property name, an identifier for the
#: computed value, and the closing `})` of the object literal. Deliberately shape-based — the
#: identifier is minifier-local and does not survive a rebuild.
CAN_CHECK_SEND_SITE = re.compile(r"canCheck:[a-zA-Z$_]+\}\)")

#: The property the positive control substitutes for the flag, so the mutant's message is the
#: same push with one field swapped rather than a syntax error.
STAND_IN = "x1"

#: What the drive records for a key whose value is `undefined`, which JSON has no spelling for.
UNDEFINED = "__undefined__"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

_DRIVE = r'''
// #3772: import the shipped service worker under a stub `chrome` global and drive the #3568
// panel-focus sequence. Reads BUNDLE (a path) and FIXTURE (window id, tab table, tab ids, port
// names, and the object chrome.storage.session.get returns) from the environment; prints one
// `REPORT <json>` line and nothing else.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const BUNDLE = process.env.BUNDLE;
const FX = JSON.parse(process.env.FIXTURE);

const listeners = {};
const reg = (key) => (fn) => { (listeners[key] ||= []).push(fn); };
const panelBehavior = [];
const fetchCalls = [];

globalThis.chrome = {
  storage: {
    session: {
      get: async () => ({ ...FX.seed }),
      set: async () => {},
      remove: async () => {},
    },
  },
  tabs: {
    get: async (tabId) => {
      const t = FX.tabs[tabId];
      if (!t) throw new Error(`no such tab ${tabId}`);
      return { ...t };
    },
    onActivated: { addListener: reg("tabs.onActivated") },
    onUpdated: { addListener: reg("tabs.onUpdated") },
    onRemoved: { addListener: reg("tabs.onRemoved") },
  },
  runtime: {
    onConnect: { addListener: reg("runtime.onConnect") },
    onInstalled: { addListener: reg("runtime.onInstalled") },
  },
  webNavigation: {
    onCommitted: { addListener: reg("webNavigation.onCommitted") },
    onHistoryStateUpdated: { addListener: reg("webNavigation.onHistoryStateUpdated") },
  },
  sidePanel: { setPanelBehavior: async (o) => { panelBehavior.push(o); } },
};

globalThis.fetch = async (url) => {
  fetchCalls.push(String(url));
  return {
    ok: true,
    status: 200,
    json: async () => ({ session_key: "sk_offline" }),
    text: async () => "",
    body: { cancel: async () => {} },
  };
};

const report = { IMPORT_SHA256: createHash("sha256").update(readFileSync(BUNDLE)).digest("hex") };
await import(pathToFileURL(BUNDLE).href);

// `JSON.stringify` drops a key whose value is `undefined`, and `pushFocus` writes
// `url: n?.url, title: n?.title` — undefined for a tab with no stored mapping. Report the key
// list beside a value map that spells those out, or an unmapped leg looks like a shorter
// message rather than the same message with two unset fields.
const encode = (m) => ({
  keys: Object.keys(m),
  values: Object.fromEntries(
    Object.entries(m).map(([k, v]) => [k, v === undefined ? "__undefined__" : v])),
});

const makePort = (name) => {
  const sent = [];
  const port = {
    name,
    _msg: null,
    _disc: null,
    postMessage: (m) => { sent.push(m); },
    onMessage: { addListener: (fn) => { port._msg = fn; } },
    onDisconnect: { addListener: (fn) => { port._disc = fn; } },
    drain: () => sent.splice(0, sent.length).map(encode),
  };
  return port;
};

const panel = makePort(FX.panelPort);
const decoy = makePort(FX.decoyPort);
const fire = async (key, arg) => { await listeners[key][0](arg); };

report.LISTENERS_REGISTERED_AT_IMPORT = Object.keys(listeners).sort();
report.onActivated_count = (listeners["tabs.onActivated"] || []).length;
report.onConnect_count = (listeners["runtime.onConnect"] || []).length;

await fire("runtime.onInstalled", undefined);
report.PANEL_BEHAVIOR = panelBehavior;

// An activation BEFORE any panel exists: the bundle holds no port to push to.
await fire("tabs.onActivated", { tabId: FX.checkableTab, windowId: FX.windowId });
report.BEFORE_CONNECT = panel.drain();

// A port whose name is not the side panel's is refused outright: the connect listener
// registers NO message handler for it, so it can never enter the window map or be pushed to.
// Asking it for a `panel-ready` is what distinguishes that from a listener that accepted it and
// then ignored the message — the two look identical from the buffer alone.
await fire("runtime.onConnect", decoy);
report.DECOY_MESSAGE_LISTENER = Boolean(decoy._msg);
if (decoy._msg) await decoy._msg({ type: "panel-ready", windowId: FX.windowId, tabId: FX.checkableTab });
await fire("tabs.onActivated", { tabId: FX.checkableTab, windowId: FX.windowId });
report.DECOY_PORT = decoy.drain();

// The #3568 order: the panel connects and says it is ready, and only then does the tab switch
// happen. This is the condition the click-through used to establish by hand.
await fire("runtime.onConnect", panel);
await panel._msg({ type: "panel-ready", windowId: FX.windowId, tabId: FX.checkableTab });
report.AFTER_PANEL_READY = panel.drain();

await fire("tabs.onActivated", { tabId: FX.checkableTab, windowId: FX.windowId });
report.AFTER_ACTIVATE_CHECKABLE = panel.drain();

await fire("tabs.onActivated", { tabId: FX.gmailTab, windowId: FX.windowId });
report.AFTER_ACTIVATE_GMAIL = panel.drain();

await fire("tabs.onActivated", { tabId: FX.unmappedTab, windowId: FX.windowId });
report.AFTER_ACTIVATE_UNMAPPED = panel.drain();

report.FETCH_CALLS = fetchCalls;
console.log(`REPORT ${JSON.stringify(report)}`);
'''


def _drive(tmp_path: Path, name: str, bundle: Path, seed: dict) -> dict:
    """Run the drive over `bundle` in its own node process and return its report.

    Each leg gets its own directory and its own node child, so the two drives a test compares
    never share a module cache, a hydrated tab map, or a port.
    """
    work = tmp_path / name
    work.mkdir(parents=True, exist_ok=True)
    script = work / "drive.mjs"
    script.write_text(_DRIVE, encoding="utf-8")
    fixture = {
        "windowId": WINDOW_ID,
        "tabs": {str(tab_id): tab for tab_id, tab in TABS.items()},
        "checkableTab": CHECKABLE_TAB,
        "gmailTab": GMAIL_TAB,
        "unmappedTab": UNMAPPED_TAB,
        "panelPort": PANEL_PORT,
        "decoyPort": DECOY_PORT,
        "seed": seed,
    }
    proc = subprocess.run(
        [shutil.which("node"), str(script)],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "BUNDLE": str(bundle), "FIXTURE": json.dumps(fixture)},
    )
    assert proc.returncode == 0, f"the drive exited {proc.returncode}: {proc.stderr}"
    lines = [line for line in proc.stdout.splitlines() if line.startswith("REPORT ")]
    assert len(lines) == 1, f"expected one report line, got: {proc.stdout!r}"
    return json.loads(lines[0][len("REPORT "):])


def _seed() -> dict:
    """A stored tab→session mapping for the checkable tab, in the shape `restore()` reads.

    The bundle keys its session map `tab:<tabId>` and hydrates from
    `chrome.storage.session.get(null)`, so this is the whole seed — no `request-session`
    message, and therefore no API call, is needed to get a mapped tab.
    """
    return {f"tab:{CHECKABLE_TAB}": {"sessionKey": SEED_KEY,
                                     "url": CHECKABLE_URL,
                                     "title": CHECKABLE_TITLE}}


def _one_leg(report: dict, label: str) -> dict:
    """The single message `label`'s activation delivered, as {keys, values}."""
    leg = report[label]
    assert len(leg) == 1, f"{label} delivered {len(leg)} messages, expected exactly one: {leg}"
    return leg[0]


def test_the_shipped_bundle_imports_under_a_stub_of_every_member_it_touches(tmp_path):
    """Clause 1: the import runs, exactly one tab-switch and one panel listener are registered,
    and the panel listener serves only the side panel's own port name.

    The seven names are the import-time surface: both `webNavigation` registrations are in it
    because they run while the module body executes, so a stub written from the TypeScript
    source rather than from the bundle's own call sites dies in the `import()` before any
    assertion can run. One `tabs.onActivated` listener and one `runtime.onConnect` listener is
    the count that matters for #3568: a second tab-switch listener would mean a second builder
    of the same message, which is the shape the #3741 fix removed. The name gate is checked by
    handing the connect listener a decoy port first and asking whether any message handler got
    attached to it — the real listener returns before `t.onMessage.addListener` for a name that
    is not `lloyd-sidepanel`, so the decoy is never registered, never enters the window map, and
    never receives a push. Handler-absence is the assertion rather than an empty buffer, because
    a listener that accepted the port and then ignored its messages would leave that buffer just
    as empty; on a bundle with the name check stripped the flag reads true (verified out of band
    against such a copy), and this node goes red.
    """
    report = _drive(tmp_path, "import", BUNDLE, _seed())

    assert report["LISTENERS_REGISTERED_AT_IMPORT"] == IMPORT_TIME_LISTENERS
    assert report["onActivated_count"] == 1, report["LISTENERS_REGISTERED_AT_IMPORT"]
    assert report["onConnect_count"] == 1, report["LISTENERS_REGISTERED_AT_IMPORT"]
    # The sidePanel member is reached only from the onInstalled listener, so firing that
    # listener is the only way to show this end of the stub is the end the bundle calls.
    assert report["PANEL_BEHAVIOR"] == [{"openPanelOnActionClick": True}]
    assert report["DECOY_MESSAGE_LISTENER"] is False, (
        f"the connect listener attached a message handler to a port named {DECOY_PORT}, so the "
        "panel-name gate is gone")
    assert report["DECOY_PORT"] == [], (
        f"a port named {DECOY_PORT} was pushed to, so delivery does not gate on the panel name")
    assert report["FETCH_CALLS"] == [], "the drive reached the network"


def test_the_panel_connects_first_and_the_activation_that_follows_pushes_can_check_true(tmp_path):
    """Clause 2: with the port attached, activating a checkable tab delivers `canCheck: true`.

    The order is the item, not an implementation detail. Firing the activation before a panel
    exists delivers nothing at all — `pushFocus` has no port for that window — so the empty leg
    below is what makes "the panel was connected BEFORE the activation" a checked fact rather
    than a comment. `AFTER_PANEL_READY` is the push `panel-ready` itself causes; the leg this
    clause is about is the activation that follows it.
    """
    report = _drive(tmp_path, "connected", BUNDLE, _seed())

    assert report["BEFORE_CONNECT"] == [], (
        "an activation with no panel attached delivered a message, so the connected-first "
        "ordering below would not have been the thing under test")

    _one_leg(report, "AFTER_PANEL_READY")

    delivered = _one_leg(report, "AFTER_ACTIVATE_CHECKABLE")
    assert delivered["values"]["type"] == "focus-session", delivered
    assert delivered["values"]["canCheck"] is True, delivered
    assert delivered["values"]["tabId"] == CHECKABLE_TAB, delivered
    assert delivered["values"]["windowId"] == WINDOW_ID, delivered


def test_activating_gmail_in_the_same_drive_pushes_the_same_shape_with_the_flag_false(tmp_path):
    """Clause 3: Gmail arrives as `focus-session` with `canCheck: false`.

    Both legs come from one node process, one import, one port. A stub that fabricated the
    flag, or a bundle that pushed a constant, could produce `true` on the checkable tab and
    `false` on Gmail only by reading each tab's own URL through `chrome.tabs.get` — which is
    the claim. The field set is asserted too: the Gmail push carries the same seven names, so
    the flag is one computed field of one message shape and not a different, shorter message.
    """
    report = _drive(tmp_path, "gmail", BUNDLE, _seed())

    checkable = _one_leg(report, "AFTER_ACTIVATE_CHECKABLE")
    gmail = _one_leg(report, "AFTER_ACTIVATE_GMAIL")

    assert gmail["values"]["type"] == "focus-session", gmail
    assert gmail["values"]["canCheck"] is False, gmail
    assert gmail["values"]["tabId"] == GMAIL_TAB, gmail
    assert gmail["keys"] == FOCUS_FIELD_SET, gmail
    assert checkable["values"]["canCheck"] is True, (
        "the same drive did not push the flag true for the checkable tab, so the Gmail leg's "
        "false proves nothing about where the flag comes from")


def test_the_stored_mapping_supplies_the_fields_and_the_flag_survives_without_it(tmp_path):
    """Clause 4: seeded fields ride along, and an unmapped tab still gets a computed flag.

    Two drives over the SAME tab 7, differing only in the seed handed to
    `chrome.storage.session.get`. Seeded, the push carries the stored `sessionKey`, `url` and
    `title` beside `canCheck` — the field set `pushFocus` builds when a mapping exists. With
    the store empty, tab 7 arrives with `sessionKey: null`, `url` and `title` `undefined`, and
    `canCheck` still `true`. That pair is the point: the mapping and the flag are independent
    fields of one push, so a harness that only ever seeded could not tell a per-activation
    computation from a stub that never maps anything. Tab 9, unmapped inside the seeded drive,
    is the same fact one tab later — checkable URL, no stored session, flag `true`.
    """
    seeded = _drive(tmp_path, "seeded", BUNDLE, _seed())
    unseeded = _drive(tmp_path, "unseeded", BUNDLE, {})

    with_mapping = _one_leg(seeded, "AFTER_ACTIVATE_CHECKABLE")
    assert with_mapping["keys"] == FOCUS_FIELD_SET, with_mapping
    assert with_mapping["values"]["sessionKey"] == SEED_KEY, with_mapping
    assert with_mapping["values"]["url"] == CHECKABLE_URL, with_mapping
    assert with_mapping["values"]["title"] == CHECKABLE_TITLE, with_mapping
    assert with_mapping["values"]["canCheck"] is True, with_mapping

    without = _one_leg(unseeded, "AFTER_ACTIVATE_CHECKABLE")
    assert without["keys"] == FOCUS_FIELD_SET, without
    assert without["values"]["sessionKey"] is None, without
    assert without["values"]["url"] == UNDEFINED, without
    assert without["values"]["title"] == UNDEFINED, without
    assert without["values"]["canCheck"] is True, (
        "the flag was not computed for a tab with no stored mapping, so the seeded leg's true "
        "cannot be attributed to canCheckUrl either")

    unmapped = _one_leg(seeded, "AFTER_ACTIVATE_UNMAPPED")
    assert unmapped["values"]["tabId"] == UNMAPPED_TAB, unmapped
    assert unmapped["values"]["sessionKey"] is None, unmapped
    assert unmapped["values"]["canCheck"] is True, unmapped


def test_removing_the_one_can_check_send_site_takes_the_field_off_every_push(tmp_path):
    """Clause 5: the control — the identical drive over a bundle missing the flag sends no flag.

    The bundle puts `canCheck` into a message at exactly one site, so one regex finds it and one
    substitution removes it, leaving the rest of the push byte-identical. Driving that copy under
    the same seed must deliver a `focus-session` message that is the same object with the flag
    gone: not an error, not a skipped assertion, an absent key. Without this leg there is no
    evidence the harness can observe the field it asserts — #3568 is precisely the case where the
    string was in the file and the value never arrived, and every text pin stayed green through
    it.

    The substitute is `x1: 1`, and the surviving shape is asserted as a set difference rather
    than as that name, so the claim is about `canCheck` and not about the stand-in.
    """
    raw = BUNDLE.read_text(encoding="utf-8")
    sites = CAN_CHECK_SEND_SITE.findall(raw)
    assert len(sites) == 1, (
        f"the control removes one send site and the bundle has {len(sites)}: {sites}")

    mutant = tmp_path / "bundle-without-cancheck.js"
    mutant.write_text(CAN_CHECK_SEND_SITE.sub(f"{STAND_IN}:1}})", raw), encoding="utf-8")

    real = _drive(tmp_path, "control-real", BUNDLE, _seed())
    control = _drive(tmp_path, "control-mutant", mutant, _seed())

    assert _one_leg(real, "AFTER_ACTIVATE_CHECKABLE")["values"]["canCheck"] is True
    for label in ("AFTER_ACTIVATE_CHECKABLE", "AFTER_ACTIVATE_GMAIL"):
        delivered = _one_leg(control, label)
        assert "canCheck" not in delivered["keys"], f"{label} still carries the flag"
        assert delivered["values"]["type"] == "focus-session", (
            f"{label}: the mutant stopped pushing focus-session, so the missing key is a broken "
            "drive rather than a missing field")
        assert set(delivered["keys"]) - {STAND_IN} == set(FOCUS_FIELD_SET) - {"canCheck"}, (
            f"{label}: the mutant changed more than the flag: {delivered['keys']}")


def test_every_assertion_above_runs_against_the_tracked_build_output(tmp_path):
    """The harness imports `dist/service-worker.js`, so a stale bundle reddens the suite.

    The whole value of this file over the text pins is that Chrome and the test load the same
    bytes. Two things make that checkable rather than aspirational. `git ls-files
    --error-unmatch` proves the path is tracked build output rather than a local artifact, so
    the assertions are grading something a reader has. And the drive hashes the file it imported
    before importing it, which the test compares against this checkout's copy: a harness quietly
    pointed at the TypeScript source, or at a rebuilt copy in some temp directory, changes that
    hash and goes red here even if every behavioural leg still passes.
    """
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--error-unmatch",
         str(BUNDLE.relative_to(REPO))],
        capture_output=True, text=True, timeout=60)
    assert tracked.returncode == 0, (
        f"{BUNDLE.relative_to(REPO)} is not tracked, so this file would be asserting over an "
        f"untracked artifact: {tracked.stderr}")

    report = _drive(tmp_path, "identity", BUNDLE, _seed())
    assert report["IMPORT_SHA256"] == hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    assert report["IMPORT_SHA256"] != hashlib.sha256(TS_SOURCE.read_bytes()).hexdigest()


# ── #3854 / #3645: press the button against a REFUSED kickoff ───────────────────────────
#
# Everything above activates tabs, which never mints: the focus legs read a store they were
# handed. The three legs below press the button — they hand the captured `runtime.onConnect`
# message listener a `{type: "request-session"}` and answer `POST /api/message/stream` with a
# 503, the drain-window refusal `fireKickoff` throws on. What that used to leave behind is the
# defect: `setMapping` and `setCanonical` ran BEFORE the awaited POST, so the tab ended up
# mapped to a session with no turn, `handleManualCheck`'s first guard matched the canonical,
# and every later press on that URL re-focused the empty session instead of retrying — while
# the tab sat on that one page (`architecture/browser-side-panel.md` §What one check does).
#
# This drive therefore cannot share `_DRIVE` above. That harness answers
# `chrome.storage.session.get` from a fixed seed and its `set` is a no-op, which is correct for
# a leg that only reads: a leg that COMMITS needs a store that keeps what it is given, or
# "no mapping and no canonical survived" is unfalsifiable — an empty fake answers that for a
# bundle that mapped everything. So `store` here is a real fake, `STORE` is reported back, and
# `console.error` is captured because the bundle reports a refused kickoff nowhere else.

_PRESS_DRIVE = r'''
// #3854: press "Check it out, Lloyd" against a bundle whose kickoff POST is refused or answered.
// Reads BUNDLE and FIXTURE (window id, tab table, the tab to press, the port name, the seed
// store, what /message/stream answers, the key /sessions/create mints, and whether the two
// presses are fired one after the other or both before awaiting either) from the environment;
// prints one `REPORT <json>` line and nothing else.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const BUNDLE = process.env.BUNDLE;
const FX = JSON.parse(process.env.FIXTURE);

const listeners = {};
const reg = (key) => (fn) => { (listeners[key] ||= []).push(fn); };
const fetchCalls = [];
const consoleErrors = [];

// A store that KEEPS writes, unlike the focus drive's fixed answer: the claim under test is
// what is NOT in here after a refused kickoff.
const store = { ...FX.seed };

globalThis.chrome = {
  storage: {
    session: {
      get: async () => ({ ...store }),
      set: async (obj) => { Object.assign(store, obj); },
      remove: async (keys) => {
        for (const k of (Array.isArray(keys) ? keys : [keys])) delete store[k];
      },
    },
  },
  tabs: {
    get: async (tabId) => {
      const t = FX.tabs[tabId];
      if (!t) throw new Error(`no such tab ${tabId}`);
      return { ...t };
    },
    onActivated: { addListener: reg("tabs.onActivated") },
    onUpdated: { addListener: reg("tabs.onUpdated") },
    onRemoved: { addListener: reg("tabs.onRemoved") },
  },
  runtime: {
    onConnect: { addListener: reg("runtime.onConnect") },
    onInstalled: { addListener: reg("runtime.onInstalled") },
  },
  webNavigation: {
    onCommitted: { addListener: reg("webNavigation.onCommitted") },
    onHistoryStateUpdated: { addListener: reg("webNavigation.onHistoryStateUpdated") },
  },
  sidePanel: { setPanelBehavior: async () => {} },
};

// `/message/stream` answers FX.streamStatus and everything else answers 200 with FX.mintedKey,
// so the ONLY failing call in a refused drive is the kickoff — `createBrowserSession` succeeds
// and the minted session exists, which is exactly the state the bug mapped the tab onto.
globalThis.fetch = async (url) => {
  const u = String(url);
  fetchCalls.push(u);
  const refused = u.includes("/message/stream") && FX.streamStatus !== 200;
  return {
    ok: !refused,
    status: refused ? FX.streamStatus : 200,
    json: async () => ({ session_key: FX.mintedKey }),
    text: async () => (refused ? "self-mod drain" : ""),
    body: { cancel: async () => {} },
  };
};

// A refused kickoff reaches the user only as a service-worker console line, so capturing it is
// what turns "the rejection branch ran" from an inference into a reported field.
console.error = (...a) => { consoleErrors.push(a.map((x) => String(x)).join(" ")); };

const report = { IMPORT_SHA256: createHash("sha256").update(readFileSync(BUNDLE)).digest("hex") };
await import(pathToFileURL(BUNDLE).href);

const encode = (m) => ({
  keys: Object.keys(m),
  values: Object.fromEntries(
    Object.entries(m).map(([k, v]) => [k, v === undefined ? "__undefined__" : v])),
});

const makePort = (name) => {
  const sent = [];
  const port = {
    name,
    _msg: null,
    _disc: null,
    postMessage: (m) => { sent.push(m); },
    onMessage: { addListener: (fn) => { port._msg = fn; } },
    onDisconnect: { addListener: (fn) => { port._disc = fn; } },
    drain: () => sent.splice(0, sent.length).map(encode),
  };
  return port;
};

const panel = makePort(FX.panelPort);
await (listeners["runtime.onConnect"][0])(panel);
await panel._msg({ type: "panel-ready", windowId: FX.windowId, tabId: FX.pressTab });
report.PANEL_READY_PUSH = panel.drain();

const press = () => panel._msg({ type: "request-session", windowId: FX.windowId,
                                 tabId: FX.pressTab });

if (FX.inFlight) {
  // Both presses are dispatched before either is awaited, which is the only way to get the
  // second one into the queue while the first kickoff POST is still outstanding.
  const first = press();
  const second = press();
  await first;
  await second;
  report.INFLIGHT_PAIR = panel.drain();
} else {
  await press();
  report.PRESS_ONE = panel.drain();
  await press();
  report.PRESS_TWO = panel.drain();
}

report.STORE = { ...store };
report.FETCH_CALLS = fetchCalls;
report.CONSOLE_ERRORS = consoleErrors;
console.log(`REPORT ${JSON.stringify(report)}`);
'''


def _press_drive(tmp_path: Path, name: str, *, stream_status: int,
                 in_flight: bool = False, bundle: Path = BUNDLE) -> dict:
    """Run the PRESS drive over `bundle` in its own node child and return its report.

    One node process per leg, as in `_drive`: the tab map, the canonical map and the module's
    single-flight `hydrate()` are process state, so a refused leg and the answered leg it is
    contrasted with must not share one.
    """
    work = tmp_path / name
    work.mkdir(parents=True, exist_ok=True)
    script = work / "press.mjs"
    script.write_text(_PRESS_DRIVE, encoding="utf-8")
    fixture = {
        "windowId": WINDOW_ID,
        "tabs": {str(tab_id): tab for tab_id, tab in TABS.items()},
        "pressTab": PRESS_TAB,
        "panelPort": PANEL_PORT,
        "seed": {},
        "streamStatus": stream_status,
        "mintedKey": MINTED_KEY,
        "inFlight": in_flight,
    }
    proc = subprocess.run(
        [shutil.which("node"), str(script)],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "BUNDLE": str(bundle), "FIXTURE": json.dumps(fixture)},
    )
    assert proc.returncode == 0, f"the press drive exited {proc.returncode}: {proc.stderr}"
    lines = [line for line in proc.stdout.splitlines() if line.startswith("REPORT ")]
    assert len(lines) == 1, f"expected one report line, got: {proc.stdout!r}"
    return json.loads(lines[0][len("REPORT "):])


def _focus_pushes(report: dict, label: str) -> list[dict]:
    """Every `focus-session` message the press drive reported under `label`.

    A press answers the panel more than once — `spawnSession`'s own tail push, then
    `handleManualCheck`'s push after the lock — and the claim is about all of them: a single
    `sessionKey: null`-shaped push in the middle of an otherwise mapped press is the same user
    symptom (the panel flickers back to not-checked), so this returns the list rather than the
    last element.
    """
    leg = report[label]
    assert leg, f"{label} delivered no message to the panel at all, so the press never ran"
    pushes = [m for m in leg if m["values"].get("type") == "focus-session"]
    assert pushes, f"{label} delivered no focus-session: {leg}"
    return pushes


def _n_calls(report: dict, needle: str) -> int:
    """How many of the drive's recorded `fetch` calls were to a URL containing `needle`."""
    return sum(1 for url in report["FETCH_CALLS"] if needle in url)


def test_a_refused_kickoff_leaves_the_tab_unmapped_and_the_next_press_retries(tmp_path):
    """#3645 clauses 1 and 3: a refused kickoff POST commits nothing, so the next press mints.

    `POST /api/message/stream` answers 503 for both presses. Four things are read out of the
    bytes Chrome loads: every `focus-session` the panel received carries `sessionKey: null`
    (nothing was mapped, and the tail `pushFocus` still ran, so the panel is on not-checked with
    `canCheck` true and the button live to retry); the storage fake holds neither `tab:11` nor
    `lastCommitted:11`, which is the pair `clearTab` would have had to drop and the reorder
    never writes — the canonical half is the one that strands a URL, because a lingering
    `lastCommitted` is what `handleNavigation` compares a reload against; the kickoff POST was
    attempted TWICE, one attempt per press, so the second press was a retry and not the silent
    re-focus of an empty session that used to close the URL; and `fireKickoff failed` reached
    the service-worker console, so the rejection branch is the branch that ran.

    Red before the fix on three of those four: the two pushes of press one carried the minted
    key, the store held both keys, and press two fired no POST at all.
    """
    report = _press_drive(tmp_path, "refused", stream_status=STREAM_REFUSED)
    assert report["IMPORT_SHA256"] == hashlib.sha256(BUNDLE.read_bytes()).hexdigest(), (
        "the press drive did not import the tracked bundle")

    for label in ("PRESS_ONE", "PRESS_TWO"):
        for msg in _focus_pushes(report, label):
            assert msg["values"]["sessionKey"] is None, (
                f"{label}: the panel was shown {msg['values']['sessionKey']} for a kickoff the "
                "backend refused, so the tab is mapped to a session with no turn")
            assert msg["values"]["canCheck"] is True, (
                f"{label}: the refused press also took the button away, so the panel cannot "
                "retry this page")

    assert f"tab:{PRESS_TAB}" not in report["STORE"], report["STORE"]
    assert f"lastCommitted:{PRESS_TAB}" not in report["STORE"], (
        f"a canonical survived the refusal, so a reload of the same page is still 'already "
        f"checked': {report['STORE']}")

    assert _n_calls(report, "/sessions/create") == 2, report["FETCH_CALLS"]
    assert _n_calls(report, "/message/stream") == 2, (
        f"the second press did not attempt a kickoff, so it re-focused instead of retrying: "
        f"{report['FETCH_CALLS']}")
    assert any("fireKickoff failed" in line for line in report["CONSOLE_ERRORS"]), (
        report["CONSOLE_ERRORS"])


def test_an_answered_kickoff_commits_the_mapping_before_the_panel_is_told(tmp_path):
    """#3645 clause 2 and 4: the success path is unchanged — one press, one session, no re-press.

    The same drive with `/message/stream` answering 200. Press one's pushes all carry the key
    that press minted, which is the ordering the awaited POST exists for: `spawnSession`'s tail
    `pushFocus` is the first thing the panel is told and it already names the session, so
    ChatPanel's first `GET /status` lands inside the turn rather than racing it. The store holds
    both halves of the mapping — `tab:11` with that key and `lastCommitted:11` with the page's
    canonical — and press two re-focuses the same session without minting: one
    `/sessions/create` in the whole drive. `console.error` stayed silent, so the refused leg's
    console line is attributable to the status code and not to the harness.
    """
    report = _press_drive(tmp_path, "answered", stream_status=STREAM_ANSWERED)

    for label in ("PRESS_ONE", "PRESS_TWO"):
        for msg in _focus_pushes(report, label):
            assert msg["values"]["sessionKey"] == MINTED_KEY, (
                f"{label}: the panel was not shown the session this press minted: "
                f"{msg['values']}")

    assert report["STORE"].get(f"tab:{PRESS_TAB}", {}).get("sessionKey") == MINTED_KEY, (
        report["STORE"])
    assert report["STORE"].get(f"lastCommitted:{PRESS_TAB}") == PRESS_URL, report["STORE"]
    assert _n_calls(report, "/sessions/create") == 1, report["FETCH_CALLS"]
    assert _n_calls(report, "/message/stream") == 1, report["FETCH_CALLS"]
    assert report["CONSOLE_ERRORS"] == [], report["CONSOLE_ERRORS"]


def test_two_presses_while_the_first_kickoff_is_in_flight_mint_one_session(tmp_path):
    """#3645 clause 4: `spawnSession` needs no lock of its own — `withTabLock` still serialises it.

    Both `request-session` messages are handed to the panel's message listener before either is
    awaited, so the second is queued while the first kickoff POST is outstanding. One
    `/sessions/create` and one `/message/stream` for the pair is the claim: the second press
    enters `handleManualCheck`, waits on the lock the first already holds, and then reads a
    mapping whose canonical matches, so it re-focuses. That only holds because the mint commits
    inside the same locked region the check runs in — a `spawnSession` that grabbed its own lock
    (or committed outside the caller's) would let press two mint a duplicate. Every push of the
    pair carries the minted key, so the queued press still lands the panel on the session.
    """
    report = _press_drive(tmp_path, "in-flight", stream_status=STREAM_ANSWERED, in_flight=True)

    assert _n_calls(report, "/sessions/create") == 1, report["FETCH_CALLS"]
    assert _n_calls(report, "/message/stream") == 1, report["FETCH_CALLS"]
    pushes = _focus_pushes(report, "INFLIGHT_PAIR")
    assert len(pushes) >= 2, (
        f"a queued press that delivered nothing is indistinguishable from one that was dropped: "
        f"{pushes}")
    for msg in pushes:
        assert msg["values"]["sessionKey"] == MINTED_KEY, msg
