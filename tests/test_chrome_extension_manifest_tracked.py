"""The Chrome side-panel extension's manifest must be in the tree (#1701), and
its tracked build output must match the source it was built from (#3571).

A repo-wide `*.json` rule in `.gitignore` hid `chrome-extension/manifest.json`
while `chrome-extension/dist/**` (service worker, side panel, assets, icons) was
tracked, so a fresh clone shipped a `dist/` Chrome cannot load, and nothing
could regenerate it: `web/vite.chrome.config.ts` copies the manifest, it does
not generate it. `git add -A` skips an ignored file silently, which is how it
stayed out for two weeks.

Both copies are pinned: the hand-written source, and the one the build writes
into the tracked `dist/` — keeping `dist/` tracked without its manifest would
leave the committed extension one file short again.

The second half of this file (#3571) is about the other half of the same
directory. Chrome is pointed at `chrome-extension/dist/`, which is a
hand-committed build artifact: nothing type-checks the extension source
(`web/tsconfig.json` `"include": ["src"]` excludes `chrome-extension/src/**`),
nothing builds it in the gate (`_vite_build` in `scripts/automod/gate.py` passes
no `-c`, so `web/vite.chrome.config.ts` is never the config used), and
`spec.classify` returns `unlisted` for every `chrome-extension/**` path, so a
round cannot rebuild it either. Before these assertions the only thing standing
between a landed `web/src/sidepanel/` change and a stale bundle was somebody
remembering to rebuild — and `spec.classify("web/src/sidepanel/SidePanelApp.tsx")`
is `allowed`, so a round can land the source change that strands the bundle
without any rung noticing.

So the literals below are **harvested from the current source, not spelled
into this file**: a hardcoded list would still pass after somebody renamed a
prompt in the source and forgot to rebuild, which is the exact accident this is
meant to catch. Each pin extracts string literals out of the source as it
stands right now and requires each one in the tracked bundle, so the suite goes
red the moment source and bundle disagree. The rebuild that clears a red run is
`cd web && VITE_API_BASE='http://127.0.0.1:8080/api' npx vite build -c
vite.chrome.config.ts`, and clause 4's API-base assertion is the part of this
that a literal grep alone cannot do — see its docstring.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

MANIFESTS = (
    "chrome-extension/manifest.json",
    "chrome-extension/dist/manifest.json",
)

# ── #3571: the tracked bundle must match the source it was built from ─────────

DIST = REPO / "chrome-extension" / "dist"
PANEL_HTML = DIST / "sidepanel.html"
SW_BUNDLE = DIST / "service-worker.js"
SW_SRC = REPO / "chrome-extension" / "src" / "background" / "service-worker.ts"
PANEL_SRC = REPO / "web" / "src" / "sidepanel" / "SidePanelApp.tsx"
API_SRC = REPO / "web" / "src" / "api.ts"
README = REPO / "chrome-extension" / "README.md"

#: (what the construct is, the source it lives in, a pattern whose capture
#: groups are the literals themselves). Every pattern here matches the source as
#: it stands at #3571; `_harvest` fails loudly, not vacuously, if one stops
#: matching, so a rename in the source cannot silently delete a pin.
WORKER_PINS: tuple[tuple[str, Path, str], ...] = (
    ("the kickoff prompt the check button sends", SW_SRC,
     r"`(pull the [^\n]*?)\\n\$\{"),
    # `request-session` is the one that names the manual check (`#1957`-era
    # protocol); the harvest also takes its siblings — `focus-session`,
    # `panel-ready` and `lloyd-sidepanel` are all the file declares today — so
    # renaming any message-type without rebuilding goes red too.
    ("message-type strings", SW_SRC, r'"([a-z][a-z0-9]*-[a-z0-9-]+)"'),
    ("the session-key prefix", SW_SRC, r"`(\w+_)\$\{"),
    # The field name, not the interface line: `canCheck?: boolean` in the source
    # is a type and is erased by the build, while the object shorthand `canCheck,`
    # is what survives — the bundle carries it exactly once, as `canCheck:s`.
    ("the canCheck field", SW_SRC, r"\b(\w+)\?:\s*boolean"),
)

PANEL_PINS: tuple[tuple[str, Path, str], ...] = (
    ("the check button's label", PANEL_SRC, r'const \w+ = "([^"]+)"'),
    ("the empty-state strings", PANEL_SRC, r'\?\s*"([^"]+)"\s*:\s*"([^"]+)"'),
)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


@pytest.mark.parametrize("rel_path", MANIFESTS)
def test_manifest_is_tracked(rel_path: str):
    res = _git("ls-files", "--error-unmatch", rel_path)
    assert res.returncode == 0, (
        f"{rel_path} is not tracked — a fresh clone gets an unloadable extension: "
        f"{res.stderr.strip()}")


@pytest.mark.parametrize("rel_path", MANIFESTS)
def test_manifest_is_not_ignored(rel_path: str):
    # `-q` exits 0 when the path IS ignored, 1 when it is not (a negation that
    # re-includes it counts as not ignored), 128 on error.
    res = _git("check-ignore", "-q", "--no-index", rel_path)
    assert res.returncode == 1, (
        f"{rel_path} is ignored (exit {res.returncode}) — `git add -A` would skip "
        f"the next change to it silently: {_git('check-ignore', '-v', '--no-index', rel_path).stdout.strip()}")


def test_the_ignore_check_can_fail():
    """Positive control: the blanket `*.json` rule still hides an ordinary JSON
    file beside the manifest, so the exit-1 above means a negation matched,
    not that `check-ignore` stopped answering."""
    res = _git("check-ignore", "-q", "--no-index", "chrome-extension/not-a-manifest.json")
    assert res.returncode == 0, "the *.json rule no longer hides stray JSON — the control is inert"


# ── #3571: the helpers the freshness pins share ───────────────────────────────

def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _text(path: Path) -> str:
    assert path.is_file(), (
        f"{_rel(path)} is missing from the tree — every bundle pin in this file "
        f"reads it; a tracked file that has vanished is not a stale bundle, it is "
        f"a deleted one")
    return path.read_text(encoding="utf-8", errors="replace")


def _harvest(path: Path, pattern: str, what: str) -> list[str]:
    """Every literal the pattern's capture groups yield from the source as it
    stands now, deduplicated in order of appearance.

    Asserting the harvest is non-empty is what keeps each pin from becoming a
    loop over nothing: a renamed or deleted construct then fails here naming the
    pattern, instead of quietly removing an assertion from the suite.
    """
    found = list(dict.fromkeys(
        g for m in re.finditer(pattern, _text(path)) for g in m.groups()))
    assert found, (
        f"{what}: {pattern!r} matched nothing in {_rel(path)} — the construct this "
        f"pin harvests its literal from has moved or been renamed. Update the "
        f"pattern in tests/{Path(__file__).name} to match the new source, then "
        f"rebuild the bundle; do not delete the pin.")
    return found


def _assert_in_bundle(literals: list[str], bundle: Path, source: Path) -> None:
    """Require each harvested literal verbatim in the bundle, naming the ones
    that dropped out.

    A missing literal means the bundle was not rebuilt after the source changed
    (or the source was rebuilt with a different bundler contract). The message
    names the literal, not just the count, because the fix is to look at that
    string in the source and decide whether the bundle or the source is wrong.
    """
    body = _text(bundle)
    missing = [lit for lit in literals if lit not in body]
    assert not missing, (
        f"{_rel(bundle)} is stale against {_rel(source)}: "
        f"{', '.join(repr(m) for m in missing)} "
        f"{'is' if len(missing) == 1 else 'are'} in the source and not in the "
        f"tracked bundle. Nothing else detects this — the extension source is "
        f"outside web/tsconfig.json's `include` and web/vite.chrome.config.ts is "
        f"never built by a gate — so rebuild by hand: cd web && "
        f"VITE_API_BASE='http://127.0.0.1:8080/api' npx vite build "
        f"-c vite.chrome.config.ts")


def _html_assets() -> list[str]:
    """The asset paths `dist/sidepanel.html` references, repo-`dist`-relative.

    One direction only on purpose: the html names what Chrome will load, and
    `dist/assets/` legitimately holds a `highlighted-body-OFNGDK62-*.js` chunk
    too, which the panel imports dynamically and the html never names.
    """
    return re.findall(r'(?:src|href)="/((?:assets|icons)/[^"]+)"', _text(PANEL_HTML))


# ── #3571: the pins ───────────────────────────────────────────────────────────

def test_worker_bundle_carries_every_literal_the_current_source_defines():
    """#3571 clause 1: the tracked `dist/service-worker.js` carries the literals
    that exist only in `chrome-extension/src/background/service-worker.ts`.

    Harvested, so the assertions are `canCheck` (the field name, whose surviving
    use is the object shorthand at `:97`), `request-session` and its three
    message-type siblings, the session-key prefix `ext_tab_` (`:243`) and both
    kickoff prompts (`:57-58`) as the source spells them today — eight strings,
    all present in the bundle on the tree this test landed on. The failure
    message names the literal that dropped out.
    """
    for what, src, pattern in WORKER_PINS:
        literals = _harvest(src, pattern, what)
        _assert_in_bundle(literals, SW_BUNDLE, src)


def test_every_asset_the_built_sidepanel_html_names_exists_on_disk():
    """#3571 clause 2: every asset `dist/sidepanel.html` names is a file under
    `chrome-extension/dist/`.

    Vite content-hashes the panel's chunks (one `assets/sidepanel-*.js` and one
    `assets/sidepanel-*.css`, both named by the build that produced them) and
    rewrites `sidepanel.html` to match, so a rebuild whose renamed html was not
    committed — or whose new
    chunk was not — leaves Chrome loading a path that is not in the directory.
    The two `manifest.json`/`icons` copies above pin what is tracked; this pins
    that the built html and the built chunks agree.
    """
    named = _html_assets()
    assert named, (
        f"{_rel(PANEL_HTML)} names no /assets/ or /icons/ path at all — the "
        f"vite-built html always names at least its script and its stylesheet, "
        f"so this means the pattern stopped matching, not that the build got "
        f"smaller: \n{_text(PANEL_HTML)}")
    missing = [a for a in named if not (DIST / a).is_file()]
    assert not missing, (
        f"{_rel(PANEL_HTML)} names {', '.join(missing)} "
        f"{'but it is' if len(missing) == 1 else 'but none of them are'} under "
        f"{_rel(DIST)}/ — the html and the hashed chunks were not committed "
        f"together. Named: {named}")


def test_panel_bundle_named_by_the_html_carries_the_panel_strings():
    """#3571 clause 3: the panel half of the bundle is pinned too, not just the
    worker.

    The bundle under test is the one `dist/sidepanel.html` names rather than a
    name spelled into this file, because the name is a content hash. From
    `web/src/sidepanel/SidePanelApp.tsx` this harvests the check button's label
    (`const CHECK_LABEL = "Check it out, Lloyd"`, `:28`) and both branches of
    every ternary in the file — `Ready when you are` / `No checkable page here`
    at `:177` among them, five strings on the tree as it stands. A round is
    allowed to edit that source file; this is the check that tells it whether
    the bundle it left behind still says what the panel says.
    """
    js_assets = [a for a in _html_assets() if a.endswith(".js")]
    assert js_assets, f"{_rel(PANEL_HTML)} names no .js asset to read"
    bundle = DIST / js_assets[0]
    for what, src, pattern in PANEL_PINS:
        literals = _harvest(src, pattern, what)
        _assert_in_bundle(literals, bundle, src)


def test_both_tracked_bundles_embed_the_api_base_the_readme_prescribes():
    """#3571 clause 4: both tracked bundles embed the API base the build recipe
    prescribes, so a rebuild with `VITE_API_BASE` unset is caught.

    `chrome-extension/README.md:34` is the recipe, and `web/src/api.ts:1005` is
    `(import.meta.env.VITE_API_BASE as string | undefined) || '/api'` — so the
    easy-to-paste `--watch` command without the env prefix produces a bundle
    whose base is the relative `/api`, which is wrong for an extension page
    (there is no same-origin backend at chrome-extension://) and which every
    literal assertion above still passes, because the strings are unaffected.
    The base is therefore harvested from the README rather than hardcoded, and
    asserted in the worker bundle and in the panel chunk the html names.

    The fallback itself is asserted too: if `api.ts` stops defaulting to `/api`,
    the reasoning in this docstring no longer describes the build, and the next
    reader of a red run would be sent looking for the wrong mistake.
    """
    bases = _harvest(README, r"VITE_API_BASE='([^']+)'", "the build recipe's API base")
    js_assets = [a for a in _html_assets() if a.endswith(".js")]
    assert js_assets, f"{_rel(PANEL_HTML)} names no .js asset to read"
    for bundle in (SW_BUNDLE, DIST / js_assets[0]):
        _assert_in_bundle(bases, bundle, README)
    api_src = _text(API_SRC)
    assert "'/api'" in api_src, (
        f"{_rel(API_SRC)} no longer falls back to the relative '/api', so an "
        f"unset VITE_API_BASE no longer produces the silent mistake this node "
        f"exists to catch — re-read the recipe in {_rel(README)} before "
        f"trusting a red run here to mean a stale bundle")


def test_the_bundle_assertions_can_fail():
    """Positive control for every pin above.

    Each of them is a substring test against a minified bundle, which is only
    meaningful if the bundle is not simply the source with a `.js` suffix. The
    source's `//` comments are stripped by the build, so the first real comment
    line must be absent from the worker bundle. If this ever fires, the bundler
    started emitting source prose — and the literal pins above stopped being
    evidence about freshness.
    """
    src = _text(SW_SRC)
    comments = [ln.strip()[2:].strip() for ln in src.splitlines()
                if ln.strip().startswith("//") and len(ln.strip()) > 24]
    assert comments, f"{_rel(SW_SRC)} has no comment line over 24 characters to test with"
    probe = comments[0]
    assert probe not in _text(SW_BUNDLE), (
        f"the comment {probe!r} from {_rel(SW_SRC)} appears verbatim in "
        f"{_rel(SW_BUNDLE)} — the bundle is carrying source prose, so a substring "
        f"match in this file no longer distinguishes built output from source")


def test_the_focus_message_is_built_once_in_the_source_and_once_in_the_bundle():
    """#3741: one construction site, and the shipped bytes agree with it.

    The literal harvest above can only ever ask whether a string is SOMEWHERE in
    `dist/service-worker.js`, and that is precisely why it stayed green through
    #3568: the string it wanted, `canCheck`, was in the bundle the whole time — inside
    `pushFocus` — while the code Chrome runs on a tab switch never reached it. The count
    is the stronger reading. Before this fix both sides built the focus message TWICE
    (pushFocus, and the `chrome.tabs.onActivated` listener's hand-built literal that
    omitted `canCheck`), and they agreed at 2 and 2, which is the trap: an equality on a
    wrong number. So this pins the number itself to one — the message is described in
    exactly one place in the worker, and the artefact Chrome loads was rebuilt from that
    same source. A source edit that re-adds a second literal fails here even if nobody
    rebuilds, and a rebuild of an edited source fails here even if the second literal was
    the one kept.

    The rest of what a rebuild must not break is this file's other three bundle nodes,
    each of which this round had to keep green and did:
    `test_worker_bundle_carries_every_literal_the_current_source_defines` (every
    harvested literal still present), `test_both_tracked_bundles_embed_the_api_base_the_readme_prescribes` (both bundles still spell the pinned host) and
    `test_every_asset_the_built_sidepanel_html_names_exists_on_disk` (every asset the
    built panel names is on disk). A rebuild that dropped a literal, the host or an
    asset is named by them; a rebuild that left the message shape behind is named here.
    """
    # A construction ends in a comma; the `FocusMessage` field spells the same literal
    # in type position (`type: "focus-session";`) and is not one.
    src_built = re.findall(r'type:\s*"focus-session",', _text(SW_SRC))
    bundle_built = re.findall(r'type:"focus-session"', _text(SW_BUNDLE))
    assert len(src_built) == 1, (
        f"{_rel(SW_SRC)} constructs a focus-session message {len(src_built)} times; one "
        "is pushFocus's, and a second is a hand-built literal that will drift from it — "
        "that duplicate is the defect #3568 shipped and #3741 removed")
    assert len(bundle_built) == len(src_built), (
        f"the tracked bundle builds it {len(bundle_built)} times against the source's "
        f"{len(src_built)}: dist/service-worker.js is not a build of the source in this "
        "tree, and Chrome loads the bundle, not the TypeScript")


def test_the_three_rebuild_consistency_legs_still_hold_after_the_copy_back():
    """#3741 clause 4: run the three legs, rather than name them.

    Clause 4 is about what the rebuild must NOT have broken, and the checks for that are
    three nodes in this file that predate this round: every literal harvested from the
    worker source is still in `dist/service-worker.js`, both tracked bundles still embed
    the pinned API base, and every asset the built `dist/sidepanel.html` names still exists
    on disk. This round rewrote one tracked build output and left the other eight alone, so
    those three are clause 4's substance — but a node that only cites their names proves
    nothing about the bytes, and their own greenness was never this node's evidence. So this
    calls them: the assertions run, and a rebuild that dropped a literal, the host or an
    asset reddens here as well as in its own node. Their ids are named as data so the
    failure says which leg moved.
    """
    for node in (test_worker_bundle_carries_every_literal_the_current_source_defines,
                 test_both_tracked_bundles_embed_the_api_base_the_readme_prescribes,
                 test_every_asset_the_built_sidepanel_html_names_exists_on_disk):
        try:
            node()
        except AssertionError as exc:
            raise AssertionError(f"{node.__name__}: {exc}") from None
