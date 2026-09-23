---
title: Desktop computer use
status: current
updated: 2026-09-23
---

# Desktop computer use

Lloyd can look at Alan's desktop and, when Alan lends it, drive it: capture a
window, read its interactable elements, click, type, press keys, scroll, drag,
set values. The design follows Nous Research's Hermes Agent `computer_use`
tool (MIT, `github.com/NousResearch/hermes-agent`, `tools/computer_use/`). The
backend, the lease and the dispatch rules are Lloyd's own, because this
desktop is Hyprland on Wayland and the seat is shared with a person.

Code:

| Piece | Where |
|---|---|
| Tools `desktop_capture`, `desktop_act` | `agent_mcp/desktop/__init__.py` |
| Hyprland backend (windows, grim, wtype, sendshortcut, helper client) | `agent_mcp/desktop/hypr.py` |
| Hard refusals (key combos, typed text, denied windows) | `agent_mcp/desktop/guards.py` |
| AT-SPI + uinput helper, system python | `agent-services/bin/lloyd-desktop-helper.py` |
| Lease | `app/desktop_lease.py`, file `~/lloyd-data/desktop/lease.json` |
| Desktop tab routes | `app/routers/desktop.py`, `web/src/components/pages/DesktopPage.tsx` |
| Images through the harness | `app/harness/tool_images.py` |
| Config | `desktop:` and `harness.images:` in `config.yaml` |
| Tests | `tests/test_desktop_module.py`, `tests/test_tool_images.py` |

## 1. What Hermes does, and what transferred

Hermes exposes one `computer_use` tool with an `action` discriminator. It speaks
MCP over stdio to `cua-driver` (trycua/cua, Rust, MIT). The loop is:

1. `capture(mode="som")` returns a screenshot plus a numbered list of
   interactable elements.
2. `click(element=N)` acts through accessibility where it can, and by pixel
   where it cannot.
3. Every result carries a `verdict` saying whether the effect was confirmed.
   The model re-captures to check anything unverified.

Around that loop it adds hard-blocked key combos and typed patterns,
per-action approval, screenshot dedup and eviction, an auxiliary vision model
for text-only models, and a private "Bot Screen" desktop with a human/agent
lease.

What transferred as-is:

- the capture → element index → act → verify loop;
- the element list format and the verdict and decision fields;
- the blocked combos and typed patterns;
- the unchanged-screen dedup rule (omit twice, then re-send);
- the outbound image cap and the eviction placeholder;
- the aux-vision prompt;
- the "stale element index is an error, never a guess" rule.

What did not transfer:

- **cua-driver itself.** Its supported Linux path is X11/XWayland (XTEST +
  AT-SPI), and its Linux validation document lists Hyprland as not validated.
  Every client on this desktop is native Wayland. The backend here uses the
  compositor's own tools instead.
- **Per-action approval.** A chat turn has no approval channel. It is replaced
  by a lease that the human grants for a period and can revoke by moving the
  mouse.
- **One tool.** A single tool cannot be both read-only (capture) and
  side-effecting (act), and Lloyd's annotations decide plan mode, parallel
  batches and the effect ledger. So there are two tools.

## 2. The backend (Hyprland, measured 2026-09-23)

| Need | Mechanism | Notes |
|---|---|---|
| Windows | `hyprctl clients/activewindow/monitors -j` | Matching goes address → exact class → exact title → substring. No match is an error that lists the windows; it never falls back to the focused one. |
| A window's pixels | `grim -T <stableId>` | ext-image-copy-capture reads the toplevel itself. It works on a window two workspaces away and on an occluded one. |
| The screen | `grim -g` over the monitor | Image only, no elements. |
| Downscale | `grim -s` | The longest edge is at most `max_dimension` (1456). JPEG q85 is ~150 KB, against ~1 MB as PNG. |
| Elements | AT-SPI in the helper | Interactable roles that are showing, with window-relative bounds. On Wayland, AT-SPI "screen" extents are window-relative, and an unlaid-out widget reports INT_MIN; both give `bounds: null`. |
| Background click | AT-SPI `do_action` | The pointer does not move. This is the first rung of the ladder. |
| Pixel click, wheel, drag | `hyprctl dispatch movecursor` then a uinput device | The helper owns a virtual relative mouse with three buttons, two wheels and modifier keys. A nudge (+1, -1) makes the compositor deliver motion at the warped position. |
| Keys to the focused window | `wtype` | Uses the virtual-keyboard protocol. |
| Hotkey to an unfocused window | `hyprctl dispatch sendshortcut MODS,KEY,address:` | Hyprland's one true background input path. |
| Text into a field | AT-SPI `set_text_contents` via `set_value` | Background. The result is verified by reading it back. |

The helper runs on `/usr/bin/python3` because that is the only interpreter
with `gi.repository.Atspi`. It uses the stdlib plus gi, so nothing needs
installing. It is a separate process for Hermes's reason: a hung toolkit
stalls an AT-SPI call, and a helper that times out is killed and restarted
while the aggregator keeps running. The uinput device lives and dies with the
helper, so a killed aggregator cannot leave a button held down.

**Coordinates.** Every coordinate the model sees or sends is in the
screenshot pixels of the capture it came from. `Capture.to_screen` maps a
coordinate back through that capture's origin and scale.
`desktop.coordinate_space: norm1000` switches both directions to a 0–1000 grid
per axis instead, which is how Qwen3-VL-family models ground. Which of the two
to use is decided by `eval/desktop_grounding/`.

### Accessibility coverage

GTK apps (Remmina, portal dialogs) expose a full tree. **Chromium, Electron
(VS Code) and Firefox/Thunderbird register with AT-SPI only if they start
with accessibility on.** Flipping `org.a11y.Status.IsEnabled` or
`ScreenReaderEnabled` at runtime did not make the running instances register
(measured). A fresh Chromium started with `ACCESSIBILITY_ENABLED=1` did.

For those apps to have element lists, two things are needed:

- Hyprland must export `ACCESSIBILITY_ENABLED=1`, with `env =
  ACCESSIBILITY_ENABLED,1` in the Hyprland config;
- the apps must then be restarted.

Until then those windows are pixel-only. `desktop_capture` says so ("the app
exposes no AT-SPI tree"), and the model acts by coordinate from the image.

## 3. The lease and the tripwire

The seat is shared. `desktop_capture` needs no lease. Every `desktop_act`
action except `wait` needs one.

- `app/desktop_lease.py` keeps `{holder, epoch, expires_at, last_cursor,
  last_window}`. A missing file, a corrupt file or an expired grant all mean
  the human holds the lease. That is the opposite of Hermes's private-desktop
  default, and it is on purpose.
- Only the Desktop tab grants a lease (`POST /api/desktop/lease`, 10/30/60
  minutes, capped at 240). A grant or a revoke toasts through `hyprctl notify`.
- Before each action the tripwire compares the pointer and the focused window
  with where Lloyd left them. A move of more than `tripwire_px` (24), or a focus
  change Lloyd did not make, revokes the lease and the action answers
  `human_has_control`.
- `epoch` is read before an action and again after it. A result produced
  across a takeover is void.
- Nothing Lloyd can call grants a lease. `safety.check_bash_command` refuses:
  - a request to the route (curl, wget, xh);
  - a write to the file (`>`, tee, cp, mv…);
  - a Python call into `desktop_lease.grant`.

  `agent_mcp/main.call_tool` refuses any other non-read-only tool whose
  arguments name the route or the file. That covers `http_request` to
  loopback, and `Write`/`Edit`.

## 4. Who may call it

These rules are refused in `agent_mcp/main.call_tool`, the one path every tool
call takes, so it does not matter which hooks a caller installed:

- `desktop_*` is refused for a background session: worker, autonomy, or a
  `task:*` child of one, via `service_control.is_background_session`.
- It is also refused for a bench or eval session and for a sessionless call.
  The screen is Alan's private desktop.
- `_tool_sandbox.refusal` refuses `desktop_*`, capture included, for read-only
  sessions. An in-process capture is outside any bwrap.
- `discord_bot.NON_OWNER_DISALLOWED` lists both tools.
- Annotations:
  - `desktop_capture` is `READ_ONLY`;
  - `desktop_act` is `REPEAT_EXPECTED`, because a second identical click is a
    second click and the effect ledger must not replay it;
  - `desktop_` is open-world.

Refusals applied inside the module:

- **Key combos.** Lock, log out, power, the system menu and VT switches are
  blocked. Combos are canonicalised on `+` and `-` with aliases folded, and
  matched as subsets. The table is joined at runtime by every Hyprland bind
  whose description says lock, power, system or log out.
- **Typed text.** Hermes's shell patterns apply to every target. When the
  focused window is a terminal, the text also goes through
  `check_bash_command`, so typing into `foot` is exactly as guarded as Bash.
- **Password fields.** Click, type and `set_value` on role `password text` are
  refused.
- **Denied windows.** Windows on `desktop.deny_windows` can be neither captured
  nor acted on. That list covers password managers, polkit, pinentry and the
  lock screen.
- **Hidden windows.** Typing into a window other than the focused one is
  refused as `input_target_mismatch`. Pixel input into a window on a hidden
  workspace is refused as `not_visible`. `focus_window` is the visible,
  explicit rung.

## 5. Images through the harness

This is what made any of the above visible to a model, and it also fixed
`browser_screenshot`.

- `mcp_pool` used to flatten `ImageContent` to `{"type": "image"}`. It now
  carries images out as `images: [{data, mime_type}]`, and the key is absent
  when there are none.
- The loop persists each image beside the text spills
  (`<sid>.tool-results/<call_id>.img<N>.<ext>`). The event, the history, the
  session row and the SSE frame carry an `ImageRef` (path, sha256, dims,
  scale), never base64.
- The route is chosen per model from config:
  - **native** only when `models.<alias>.supports_vision` is literally `true`.
    Refs ride on `_image_refs` and become `image_url` parts in
    `wire_messages`, at send time only.
  - **aux** when `harness.images.aux_model` names a seeing alias. The image is
    described with Hermes's prompt.
  - **drop** otherwise, with a note to the model.
- A 400 that names image input raises `MultimodalRejectedError`. The loop
  strips every image and retries once.
- Budget:
  - At most 20 images or 24 MiB per request. Past either, the oldest 8
    image-bearing messages are evicted to `[screenshot removed to save
    context]`. This is the one rewrite of earlier history, and it happens once
    per crossing.
  - Relief rung 0 keeps the newest 3 images.
  - Microcompaction and truncation drop refs.
  - Each image is estimated at 1,500 tokens. A 1456×819 frame is ~1,200 on
    this ViT (patch 16, merge 2).
- The chat renders refs from `GET /api/sessions/{sid}/tool-results/{name}`.

## 6. Turning it on

1. `desktop.enabled: true`.
2. The primary sees images only after Phase 2:
   - `agent-llm-primary.conf` sets `LANGUAGE_MODEL_ONLY="0"` and
     `MM_IMAGES_PER_PROMPT="20"`;
   - one `round restart --only agent-llm-primary`;
   - then `models.primary.supports_vision: true`.

   The tower costs ~0.84 GiB, which is ~29k KV tokens.
   `expect_kv_pool_tokens_min` must allow for that. Until then captures reach
   the model as the element list only, which already works for GTK apps.
3. Accessibility for Chromium, Electron and Firefox apps: see §2.

## 7. Not built yet

- The bot screen: a private Xvnc desktop with cua-driver adopted as a second
  backend, for unattended work. It needs `tigervnc` and `xfce4` from pacman,
  which requires sudo, so it is blocked on Alan installing them.
- Capturing without a toplevel id (every current client has one).
- An agent-cursor overlay on the real seat.
