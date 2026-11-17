"""#4236 — `browser_upload_files`: putting a LOCAL FILE into a page's file input.

Before this, the browser surface could click, fill, select, drag and evaluate,
and had no verb that attached a file, so every real web upload stopped at the
file picker. The nodes below pin the four things that were actually missing:

  * the tool is served, and classified as an actuator rather than an observer;
  * real BYTES land — asserted as the page reporting `name:size` for a 2 MiB
    file, because a test that only asserted the call returned `ok` would pass on
    a stub that never touched the input;
  * the case the tool exists for: an `input[type=file]` with `display:none`,
    which the accessibility tree never shows, so it has NO ref and only a
    selector can reach it. YouTube Studio's upload dialog is that input. A test
    built on a visible input would pass on a design that still could not do the
    job, which is why the hidden case is pinned twice — once that the selector
    works, once that no ref in the snapshot points at that input at all;
  * the policy. Attaching a file is a file-read-and-exfil primitive wearing a
    form-filling hat, so the refusals are the substance: each refused path is
    named, and after every refusal the page's input still holds nothing, which
    is what checking the whole batch up front buys.

The Chromium-backed nodes use a real browser on purpose (`requires_chromium`).
A fake page could prove that `set_input_files` was called; it cannot prove that
bytes crossed into the document, that a hidden input is attachable, or that a
refused path left the input empty.
"""
import asyncio
import importlib.util
import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_mcp import annotations as A  # noqa: E402
from agent_mcp import browser as bm  # noqa: E402
from app.config import CONFIG  # noqa: E402
from app.harness.tool_roster import REGISTERED_TOOLS  # noqa: E402

CHROMIUM = "/usr/bin/chromium"
requires_chromium = pytest.mark.skipif(
    not (Path(CHROMIUM).is_file() and importlib.util.find_spec("playwright")),
    reason="needs playwright and /usr/bin/chromium",
)

PROBE_BYTES = 2 * 1024 * 1024  # exactly 2 MiB: the echo asserts 2097152
# Captured at import, before any fixture pins HOME: the vault the tool must
# refuse is the live one, and `~` under a pinned HOME is this test's tree.
REAL_HOME = Path.home()
# A real http(s) origin rather than `set_content`'s about:blank, because half of
# what an attach discloses is the site that receives the file and the audit line
# names it. Nothing listens on this port and the route below answers every
# request, so no socket is opened and no DNS is asked — loopback is also the one
# private address the browser guard permits, so `_enforce_landing` lets the page
# through instead of faulting on an unresolvable hostname.
PAGE_ORIGIN = "http://127.0.0.1:45173"
PAGE_URL = f"{PAGE_ORIGIN}/visible"   # the fixture opens the default document

FILES_LEN = "document.getElementById('f').files.length"


def _doc(input_attrs: str = "", extra_body: str = "") -> str:
    """A page whose script echoes the first attached file as `name:size`.

    A visible caption field rides along so the snapshot always has at least one
    ref: the hidden-input node then proves the input is NOT among them, rather
    than proving the snapshot came back empty.
    """
    return (
        "<!doctype html><html><body style='margin:0'>"
        f"<input id=f type=file multiple {input_attrs}>"
        "<p id=out>none</p>"
        "<label>Caption <input id=cap type=text "
        "style='display:block;width:220px;height:28px'></label>"
        + extra_body +
        "<script>const f=document.getElementById('f');"
        "f.addEventListener('change',()=>{"
        "const el=document.getElementById('out');"
        "el.textContent=f.files.length?"
        "f.files[0].name+':'+f.files[0].size:'none';});"
        "</script></body></html>"
    )


VISIBLE_DOC = _doc()
HIDDEN_DOC = _doc("style='display:none'")
# The shape that made the item: a visible button whose handler opens the picker
# on a hidden input. The accessibility tree shows the button and not the input,
# and the click returns while the native dialog is still up.
DIALOG_DOC = (
    "<!doctype html><html><body style='margin:0'>"
    "<button id=pick style='display:block;padding:12px'>Choose a file</button>"
    "<input id=f type=file style='display:none'>"
    "<p id=out>none</p>"
    "<script>document.getElementById('pick').addEventListener("
    "'click',()=>document.getElementById('f').click());"
    "const f=document.getElementById('f');"
    "f.addEventListener('change',()=>{document.getElementById('out').textContent="
    "f.files.length?f.files[0].name+':'+f.files[0].size:'none';});"
    "</script></body></html>"
)
IFRAME_DOC = _doc(
    "",
    "<iframe width=400 height=120 srcdoc=\"<input id=g type=file "
    "style='display:none'><script>const g=document.getElementById('g');"
    "g.addEventListener('change',()=>{parent.document.getElementById('out')"
    ".textContent='frame:'+g.files[0].name+':'+g.files[0].size;});"
    "</script>\"></iframe>",
)


TWO_DOC = _doc(extra_body="<input id=f2 type=file multiple>")
DOC_BY_NAME = {
    "visible": VISIBLE_DOC,
    "hidden": HIDDEN_DOC,
    "dialog": DIALOG_DOC,
    "iframe": IFRAME_DOC,
    "two": TWO_DOC,
}


@pytest.fixture
async def upload_root(tmp_path, monkeypatch):
    """An approved directory, and a CONFIG that approves only it.

    The roots are replaced rather than extended: a test that ran against the
    shipped defaults would pass while proving nothing about default-deny.
    """
    root = tmp_path / "uploads"
    root.mkdir()
    # Pin HOME so the default roots the policy falls back to live under THIS
    # test. Under a TMPDIR inside ~/lloyd-data — what the gate's round data root
    # does — pytest's temp tree is genuinely inside the defaults, and a path
    # meant to be "outside browser.upload_roots" stops being outside it.
    monkeypatch.setenv("HOME", str(tmp_path))
    browser_cfg = dict(CONFIG.get("browser") or {})
    browser_cfg["upload_roots"] = [str(root)]
    monkeypatch.setitem(CONFIG, "browser", browser_cfg)
    return root


@pytest.fixture
async def live_page(upload_root, monkeypatch):
    """A real headless Chromium on the probe page, wired into `bm`'s globals."""
    from playwright.async_api import async_playwright

    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        executable_path=CHROMIUM, headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage"])
    context = await browser.new_context(viewport={"width": 800, "height": 600})
    page = await context.new_page()

    async def _answer(route):
        # Served as a real document at a real origin. `set_content` is not the
        # same thing: it does not run the page's inline script, and every claim
        # here about what the PAGE reports depends on that script existing.
        name = route.request.url.rsplit("/", 1)[-1].split("?")[0] or "visible"
        await route.fulfill(status=200, body=DOC_BY_NAME[name],
                            content_type="text/html; charset=utf-8")

    await page.route("**/*", _answer)
    await page.goto(PAGE_URL, wait_until="domcontentloaded")

    async def _page():
        return bm._active_page

    monkeypatch.setattr(bm, "_context", context)
    monkeypatch.setattr(bm, "_active_page", page)
    monkeypatch.setattr(bm, "_get_page", _page)
    monkeypatch.setattr(bm, "_push_browser_state", lambda tool_name: asyncio.sleep(0))
    try:
        yield page
    finally:
        if bm._ui_trailing_push is not None:
            bm._ui_trailing_push.cancel()
        await context.close()
        await browser.close()
        await pw.stop()


async def _show(page, name: str) -> None:
    """Navigate the open page to one of the probe documents, by name.

    A navigation, never `set_content`: the point of these pages is the script
    that reports what landed in the input, and `set_content` does not run it.
    """
    await page.goto(f"{PAGE_ORIGIN}/{name}", wait_until="domcontentloaded")
    bm._ref_map = {}  # a ref belongs to the document it was taken from


def _probe(root: Path, name: str = "probe.bin", size: int = PROBE_BYTES) -> Path:
    path = root / name
    path.write_bytes(b"\0" * size)
    return path


async def _out(page) -> str:
    await page.wait_for_function(
        "document.getElementById('out').textContent !== 'none'", timeout=3000)
    return await page.locator("#out").text_content()


async def _ref_map_for(page) -> dict:
    """Fresh refs for the open page, as `browser_snapshot` hands them out."""
    await bm._browser_snapshot()
    return dict(bm._ref_map)


async def _refs_naming_a_file_input(page) -> list[str]:
    """Every snapshot ref that resolves to an `<input>` of type file.

    Asked of the refs themselves rather than of the snapshot text, so the answer
    is the same question `_browser_upload_files(ref=…)` will be asked.
    """
    found: list[str] = []
    for ref_id in list(bm._ref_map):
        try:
            loc = await bm._locate(page, ref_id)
            type_ = await loc.get_attribute("type")
        except Exception:  # a ref whose element vanished is not a file input
            continue
        if type_ == "file":
            found.append(ref_id)
    return found


def _call_out(result) -> dict:
    # Every node here calls `bm.call_tool` — the same entry point the MCP
    # aggregator calls — so the payload arrives as a tool result, not a string.
    return json.loads(result.content[0].text)


# ── Clause 1: served, mutating, and rostered ─────────────────────────────────


async def test_upload_files_is_served_with_its_schema_and_classified_mutating():
    tools = {t.name: t for t in await bm.list_tools()}
    assert "browser_upload_files" in tools, sorted(tools)
    schema = tools["browser_upload_files"].input_schema
    assert set(schema["properties"]) == {"ref", "paths", "selector", "frame_index"}
    assert schema["required"] == ["paths"]

    # Mutating on every surface that asks: plan mode derives its block list from
    # `readOnlyHint`, and the effect ledger from `side_effecting`.
    annotations = A.annotations_for("browser_upload_files")
    assert annotations.read_only_hint is False
    assert annotations.open_world_hint is True
    assert A.side_effecting("browser_upload_files") is True
    # `REPEAT_EXPECTED` is the set that takes a name OUT of the ledger, and the
    # other browser verbs live there. An attach must not join them: a replayed
    # attach is a second file in a form, which is the effect #544 exists to stop.
    assert "browser_upload_files" not in A.REPEAT_EXPECTED
    assert "browser_upload_files" not in A.READ_ONLY
    assert "browser_upload_files" not in A.IDEMPOTENT

    # The closed set a worker's declared capability is checked against (#2269).
    assert "browser_upload_files" in REGISTERED_TOOLS


async def test_the_dispatch_table_routes_the_name_to_the_handler():
    """`tools/list` and `call_tool` are two tables; a name in one is nothing.

    Refused here for the missing argument, which proves the handler ran rather
    than that the name fell through to "Unknown tool".
    """
    out = _call_out(await bm.call_tool("browser_upload_files", {}))
    assert "paths is required" in out["error"]
    assert "Unknown tool" not in out["error"]


def test_the_shipped_defaults_and_the_vault_entry_are_the_paths_they_claim():
    """The roots a reader of config.yaml sees, and the vault the refusal cites.

    Read off CONFIG rather than off `DEFAULT_UPLOAD_ROOTS`, so the shipped file
    and the in-code default cannot drift apart unnoticed.
    """
    shipped = ((CONFIG.get("browser") or {}).get("upload_roots") or [])
    assert [str(r) for r in shipped] == ["~/lloyd-data", "~/Downloads"]
    assert tuple(str(r) for r in shipped) == bm.DEFAULT_UPLOAD_ROOTS
    assert Path(str(bm.VAULT_ROOT)) == Path.home() / "obsidian"
    assert bm.DEFAULT_UPLOAD_MAX_BYTES > 3_300_000_000, (
        "the 3.3 GB recording that motivated the tool must be attachable")


# ── The policy, on paths alone ────────────────────────────────────────────────


def test_the_policy_says_which_paths_it_would_approve(tmp_path, monkeypatch):
    root = tmp_path / "uploads"
    root.mkdir()
    ok = _probe(root, "small.bin", 12)
    browser_cfg = dict(CONFIG.get("browser") or {})
    browser_cfg["upload_roots"] = [str(root)]
    monkeypatch.setitem(CONFIG, "browser", browser_cfg)
    roots, ceiling = bm._upload_config()

    approved, reason = bm._upload_path_policy(str(ok), roots, ceiling)
    assert approved == ok.resolve() and reason == ""

    # A single string is as good as a one-element list, and `~` is expanded.
    approved, reason = bm._upload_path_policy(str(ok), roots, ceiling)
    assert approved is not None and reason == ""

    # A relative path is refused rather than resolved against whoever's cwd this
    # process happens to have.
    approved, reason = bm._upload_path_policy("small.bin", roots, ceiling)
    assert approved is None and "small.bin" in reason and "absolute" in reason


def test_an_empty_upload_roots_list_falls_back_to_the_defaults(monkeypatch):
    """Clearing the list must not be the widest the policy has ever been."""
    browser_cfg = dict(CONFIG.get("browser") or {})
    browser_cfg["upload_roots"] = []
    monkeypatch.setitem(CONFIG, "browser", browser_cfg)
    roots, _ceiling = bm._upload_config()
    assert roots == [Path.home() / "lloyd-data", Path.home() / "Downloads"]

    browser_cfg["upload_roots"] = ["~/Downloads/**"]
    monkeypatch.setitem(CONFIG, "browser", browser_cfg)
    globbed, _ = bm._upload_config()
    assert globbed == [Path.home() / "Downloads"], "the ** spelling is a directory"


@pytest.mark.parametrize("bad", [
    ["relative/uploads"],            # cannot be compared to anything
    ["~/lloyd-data", 7],             # not a path
    [""],                            # empty entry
    "not-a-list-at-all",             # wrong type entirely
], ids=["relative", "non-string", "empty-entry", "wrong-type"])
def test_an_unevaluable_policy_refuses_everything(monkeypatch, bad):
    """Fail closed: a policy that cannot be read has no idea where bytes go."""
    browser_cfg = dict(CONFIG.get("browser") or {})
    browser_cfg["upload_roots"] = bad
    monkeypatch.setitem(CONFIG, "browser", browser_cfg)
    with pytest.raises(ValueError):
        bm._upload_config()


def test_a_file_over_the_byte_ceiling_is_refused_by_name(tmp_path, monkeypatch):
    root = tmp_path / "uploads"
    root.mkdir()
    big = _probe(root, "big.bin", 4096)
    browser_cfg = dict(CONFIG.get("browser") or {})
    browser_cfg["upload_roots"] = [str(root)]
    browser_cfg["upload_max_bytes"] = 4095
    monkeypatch.setitem(CONFIG, "browser", browser_cfg)
    roots, ceiling = bm._upload_config()
    approved, reason = bm._upload_path_policy(str(big), roots, ceiling)
    assert approved is None
    assert "big.bin" in reason and "4095" in reason


def test_the_dot_rule_is_judged_below_the_root_not_on_its_spelling(tmp_path,
                                                                   monkeypatch):
    """A root under a dot-directory is still a root; a dot BELOW it is refused.

    The gate runs every test under `~/lloyd-work/.t/<hex>`, and gate attempt 2 of
    round SM_20261117_174945 went red on 16 nodes because the rule was applied to
    the whole absolute path: every approved `tmp_path` file had `.t` above it.
    What the rule exists to stop is a `.ssh` reached by walking down from a broad
    root, and that half must keep refusing whatever the root is called.
    """
    root = tmp_path / ".dotted-parent" / "uploads"
    root.mkdir(parents=True)
    ok = _probe(root, "fine.bin", 8)
    (root / ".secrets").mkdir()
    hidden_dir_file = _probe(root / ".secrets", "key.bin", 8)
    hidden_file = _probe(root, ".env", 8)
    outside_dotted = _probe(tmp_path / ".dotted-parent", "beside.bin", 8)
    browser_cfg = dict(CONFIG.get("browser") or {})
    browser_cfg["upload_roots"] = [str(root)]
    monkeypatch.setitem(CONFIG, "browser", browser_cfg)
    roots, ceiling = bm._upload_config()

    approved, reason = bm._upload_path_policy(str(ok), roots, ceiling)
    assert approved == ok.resolve() and reason == "", reason
    for dotted in (hidden_dir_file, hidden_file):
        approved, reason = bm._upload_path_policy(str(dotted), roots, ceiling)
        assert approved is None and "dotfiles and dot-directories" in reason, reason
        assert str(dotted) in reason
    # Outside every root: the roots refusal answers, and it says the dot too
    # rather than advising the operator to approve a dot-directory.
    approved, reason = bm._upload_path_policy(str(outside_dotted), roots, ceiling)
    assert approved is None and "outside browser.upload_roots" in reason
    assert "dot-directory" in reason and "add that directory" not in reason
    # A missing file outside the roots is refused as OUTSIDE, never as missing:
    # the refusal must not say which files exist off the approved roots.
    approved, reason = bm._upload_path_policy(
        str(tmp_path / "nowhere" / "ghost.bin"), roots, ceiling)
    assert approved is None and "outside browser.upload_roots" in reason
    assert "does not exist" not in reason


# ── Clause 2: real bytes, on a real page ─────────────────────────────────────


@requires_chromium
async def test_attaching_a_2_mib_file_by_ref_puts_its_bytes_on_the_page(live_page,
                                                                        upload_root):
    page = live_page
    refs = await _ref_map_for(page)
    file_refs = await _refs_naming_a_file_input(page)
    assert file_refs, (
        f"a VISIBLE file input should have a ref; snapshot refs: "
        f"{sorted(refs)}")

    probe = _probe(upload_root)
    out = _call_out(await bm.call_tool("browser_upload_files", {
        "ref": file_refs[0], "paths": [str(probe)]}))
    assert out["ok"] is True, out
    assert out["attached"] == 1
    assert out["target_origin"] == PAGE_ORIGIN

    # The page's own verdict, and the exact number: 2 MiB of bytes arrived, not
    # just a call that said it worked.
    assert await _out(page) == "probe.bin:2097152"
    assert await page.evaluate(FILES_LEN) == 1


@requires_chromium
async def test_two_paths_attach_in_one_call_and_the_count_is_echoed(live_page,
                                                                    upload_root):
    page = live_page
    file_ref = (await _refs_naming_a_file_input(page))[0]
    first = _probe(upload_root, "one.bin", 1024)
    second = _probe(upload_root, "two.bin", 2048)

    out = _call_out(await bm.call_tool("browser_upload_files", {
        "ref": file_ref, "paths": [str(first), str(second)]}))
    assert out["ok"] is True and out["attached"] == 2, out
    assert await page.evaluate(FILES_LEN) == 2
    assert await page.evaluate(
        "Array.from(document.getElementById('f').files)"
        ".map(x => x.name + ':' + x.size).join(',')") == "one.bin:1024,two.bin:2048"


@requires_chromium
async def test_a_single_string_for_paths_is_accepted(live_page, upload_root):
    page = live_page
    file_ref = (await _refs_naming_a_file_input(page))[0]
    out = _call_out(await bm.call_tool("browser_upload_files", {
        "ref": file_ref, "paths": str(_probe(upload_root, "single.bin", 8))}))
    assert out["ok"] is True and out["attached"] == 1, out
    assert await _out(page) == "single.bin:8"


# ── Clause 3: the hidden input, which is the case the tool exists for ────────


@requires_chromium
async def test_a_display_none_input_has_no_ref_and_only_a_selector_reaches_it(
        live_page, upload_root, monkeypatch):
    page = live_page
    monkeypatch.setattr(bm, "_active_page", page)
    await _show(page, "hidden")
    assert await page.locator("input#f").is_hidden()

    await _ref_map_for(page)
    # The snapshot has refs — the caption field is visible — and none of them is
    # the file input. That is the whole reason `selector` exists.
    assert bm._ref_map, "the page should still offer refs for its visible parts"
    assert await _refs_naming_a_file_input(page) == []

    probe = _probe(upload_root)
    refused = _call_out(await bm.call_tool("browser_upload_files", {
        "ref": "e1", "paths": [str(probe)]}))
    assert "error" in refused, refused
    assert await page.evaluate(FILES_LEN) == 0, "a ref-only call attached anyway"

    out = _call_out(await bm.call_tool("browser_upload_files", {
        "selector": "input[type=file]", "paths": [str(probe)]}))
    assert out["ok"] is True, out
    assert out["via"] == {"selector": "input[type=file]"}
    assert await _out(page) == "probe.bin:2097152"


@requires_chromium
async def test_a_button_that_opens_the_system_dialog_still_takes_the_attach(live_page,
                                                                           upload_root):
    """Clause 3's other shape: the dialog is open, and the file still arrives.

    This is the exact page that made the item — a visible button whose click
    handler opens the picker on a `display:none` input, which is how YouTube
    Studio's Upload button works. Three things have to hold together for it to
    be drivable by an agent: `browser_click` has to get the dialog open (it
    clicks the visible button, not the hidden input), the input has to have no
    ref (so a ref-only tool could not finish the job), and the attach has to
    reach the input while the dialog is still up — Playwright answers the open
    chooser as part of the same call, which is what makes the upload
    scriptable at all. The dialog being open is witnessed, not assumed: the
    FileChooser object names the input it belongs to.
    """
    page = live_page
    await _show(page, "dialog")
    await _ref_map_for(page)
    assert await _refs_naming_a_file_input(page) == [], (
        "the picker's input is display:none and must have no ref")
    button = [ref_id for ref_id, info in bm._ref_map.items() if info["role"] == "button"]
    assert button, f"the visible Upload button should have a ref: {sorted(bm._ref_map)}"

    async with page.expect_file_chooser(timeout=8000) as pending:
        clicked = _call_out(await bm.call_tool("browser_click", {"ref": button[0]}))
        assert clicked.get("ok") is True, clicked
        chooser = await pending.value
    assert await chooser.element.get_attribute("id") == "f"

    out = _call_out(await bm.call_tool("browser_upload_files", {
        "selector": "input[type=file]", "paths": [str(_probe(upload_root))]}))
    assert out["ok"] is True, out
    assert await page.evaluate(FILES_LEN) == 1, "the dialog swallowed the attach"
    assert await page.evaluate(
        "document.getElementById('f').files[0].name + ':' + "
        "document.getElementById('f').files[0].size") == "probe.bin:2097152"


@requires_chromium
async def test_a_selector_matching_several_inputs_is_refused_as_an_ambiguous_target(
        live_page, upload_root):
    page = live_page
    await _show(page, "two")
    out = _call_out(await bm.call_tool("browser_upload_files", {
        "selector": "input[type=file]", "paths": [str(_probe(upload_root))] }))
    assert "error" in out and "2 elements" in out["error"], out
    assert await page.evaluate(FILES_LEN) == 0


@requires_chromium
async def test_a_selector_matching_nothing_says_so(live_page, upload_root):
    """A selector that matches nothing is an error, not a silent no-op."""
    page = live_page
    out = _call_out(await bm.call_tool("browser_upload_files", {
        "selector": "input[type=file][name=gone]",
        "paths": [str(_probe(upload_root))] }))
    assert "no element matches" in out["error"], out
    assert await page.evaluate(FILES_LEN) == 0


@requires_chromium
async def test_a_file_input_inside_an_iframe_is_reachable_by_frame_index(live_page,
                                                                         upload_root):
    """Item clause 8: deferred from the gate, pinned here so it cannot rot."""
    page = live_page
    await _show(page, "iframe")
    assert len(page.frames) == 2, "the srcdoc frame is frame_index 1"
    out = _call_out(await bm.call_tool("browser_upload_files", {
        "selector": "input[type=file]", "frame_index": 1,
        "paths": [str(_probe(upload_root, "framed.bin", 16))] }))
    assert out["ok"] is True, out
    assert await _out(page) == "frame:framed.bin:16"
    bad = _call_out(await bm.call_tool("browser_upload_files", {
        "selector": "input[type=file]", "frame_index": 9,
        "paths": [str(_probe(upload_root, "framed.bin", 16))] }))
    assert "frame_index 9 is out of range" in bad["error"], bad


# ── Clause 4: every refusal names its path and leaves the page untouched ─────


@requires_chromium
async def test_every_refused_path_is_named_and_leaves_the_input_empty(live_page,
                                                                      upload_root,
                                                                      tmp_path):
    page = live_page
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "outside.bin").write_bytes(b"x")
    link = upload_root / "link-to-passwd"
    link.symlink_to("/etc/passwd")
    home = REAL_HOME

    cases = [
        (str(outside / "outside.bin"), "outside browser.upload_roots"),
        (str(home / "obsidian" / "memory" / "MEMORY.md"), "vault"),
        (str(home / ".ssh" / "id_rsa"), "dotfile"),
        (str(link), "/etc/passwd"),
        (str(upload_root / "missing.bin"), "does not exist"),
        (str(upload_root), "is a directory"),
    ]
    for path, needle in cases:
        out = _call_out(await bm.call_tool("browser_upload_files", {
            "selector": "input[type=file]", "paths": [path]}))
        error = out.get("error", "")
        assert "error" in out, f"{path} was attached: {out}"
        assert path in error, f"{path}: refusal does not name it — {error!r}"
        assert needle in error, f"{path}: refusal says {error!r}, not {needle!r}"
        assert await page.evaluate(FILES_LEN) == 0, f"{path} half-attached"
        assert await page.locator("#out").text_content() == "none"


@requires_chromium
async def test_one_refused_path_in_a_batch_attaches_none_of_them(live_page,
                                                                 upload_root,
                                                                 tmp_path):
    """The batch is checked before the page is touched, so a refusal cannot
    leave the first two files sitting in the input."""
    page = live_page
    ok1 = _probe(upload_root, "one.bin", 512)
    ok2 = _probe(upload_root, "two.bin", 512)
    out = _call_out(await bm.call_tool("browser_upload_files", {
        "selector": "input[type=file]",
        "paths": [str(ok1), str(ok2), str(upload_root / "missing.bin")]}))
    assert "missing.bin" in out["error"], out
    assert await page.evaluate(FILES_LEN) == 0
    assert out["not_attached"] == [str(ok1), str(ok2),
                                   str(upload_root / "missing.bin")]


@requires_chromium
async def test_an_unevaluable_policy_attaches_nothing(live_page, upload_root,
                                                      monkeypatch):
    page = live_page
    browser_cfg = dict(CONFIG.get("browser") or {})
    browser_cfg["upload_roots"] = ["uploads"]  # relative: nothing to compare to
    monkeypatch.setitem(CONFIG, "browser", browser_cfg)
    out = _call_out(await bm.call_tool("browser_upload_files", {
        "selector": "input[type=file]", "paths": [str(_probe(upload_root))] }))
    assert "upload policy could not be evaluated" in out["error"], out
    assert await page.evaluate(FILES_LEN) == 0


@requires_chromium
async def test_an_empty_upload_roots_list_still_refuses_this_test_directory(
        live_page, upload_root, monkeypatch):
    """The fallback is the defaults, and an artifact under /tmp is not in them."""
    page = live_page
    browser_cfg = dict(CONFIG.get("browser") or {})
    browser_cfg["upload_roots"] = []
    monkeypatch.setitem(CONFIG, "browser", browser_cfg)
    out = _call_out(await bm.call_tool("browser_upload_files", {
        "selector": "input[type=file]", "paths": [str(_probe(upload_root))] }))
    assert "outside browser.upload_roots" in out.get("error", ""), out
    assert str(Path.home() / "lloyd-data") in out["error"], out
    assert await page.evaluate(FILES_LEN) == 0


# ── The audit line ────────────────────────────────────────────────────────────


@requires_chromium
async def test_the_attach_is_logged_with_its_resolved_path_and_origin(live_page,
                                                                     upload_root,
                                                                     caplog):
    """Which local file went to which origin is the question this answers."""
    caplog.set_level(logging.INFO, logger="lloyd-browser")
    probe = _probe(upload_root, "audit.bin", 64)
    out = _call_out(await bm.call_tool("browser_upload_files", {
        "selector": "input[type=file]", "paths": [str(probe)]}))
    assert out["ok"] is True, out
    line = [r.getMessage() for r in caplog.records
            if "browser_upload_files" in r.getMessage()]
    assert line, caplog.text
    assert str(probe.resolve()) in line[0]
    assert PAGE_ORIGIN in line[0]


@requires_chromium
async def test_a_refused_path_is_logged_as_a_warning(live_page, upload_root,
                                                     caplog):
    caplog.set_level(logging.INFO, logger="lloyd-browser")
    out = _call_out(await bm.call_tool("browser_upload_files", {
        "selector": "input[type=file]",
        "paths": [str(upload_root / "missing.bin")]}))
    assert "error" in out
    warnings = [r for r in caplog.records
                if "refused" in r.getMessage() and r.levelno >= logging.WARNING]
    assert warnings, caplog.text
    assert str(upload_root / "missing.bin") in warnings[0].getMessage()
