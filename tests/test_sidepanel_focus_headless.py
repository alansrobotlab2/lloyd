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

Five things this measures, one per test below:

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

#: What `chrome.tabs.get` answers for each tab the drive activates. Tab 7 is checkable and
#: mapped, tab 8 is Gmail, tab 9 is checkable but never mapped — the mapped/unmapped pair.
TABS = {
    CHECKABLE_TAB: {"id": CHECKABLE_TAB, "windowId": WINDOW_ID,
                    "url": CHECKABLE_URL, "title": CHECKABLE_TITLE},
    GMAIL_TAB: {"id": GMAIL_TAB, "windowId": WINDOW_ID,
                "url": GMAIL_URL, "title": GMAIL_TITLE},
    UNMAPPED_TAB: {"id": UNMAPPED_TAB, "windowId": WINDOW_ID,
                   "url": UNMAPPED_URL, "title": UNMAPPED_TITLE},
}

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
