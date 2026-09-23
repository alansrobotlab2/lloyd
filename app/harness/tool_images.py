"""Images returned by tools: persisted, referenced, and shown to a model that can see.

Until this module every ``ImageContent`` block an MCP tool returned was
flattened to the literal string ``{"type": "image"}`` in ``mcp_pool``, so a
screenshot never reached the model and ``browser_screenshot`` shipped its PNG
as a base64 blob inside JSON text instead — into the prompt, the session file
and every transcript producer.

The rule this module keeps: **base64 exists in exactly two places**, the spill
file on disk and the outbound request body. Everything in between — the
``tool_result`` event, the loop's ``chat_messages``, the session JSON row, the
SSE frame — carries an ``ImageRef``:

    {path, sha256, mime, width, height, bytes}

plus whatever the tool reported about scaling (``orig_width``/``orig_height``/
``scale``) when it said so in its text.

Route, per model, from config (never a guess):

* ``native`` — ``models.<alias>.supports_vision`` is literally ``true``. The
  loop keeps refs on the history message under ``_image_refs`` and
  ``wire_messages`` materialises them as OpenAI ``image_url`` parts at send
  time. vLLM 0.28 keeps list-form ``role:"tool"`` content when a non-text part
  is present, and Qwen3.8-Flash-Next's own template renders image parts inside
  tool messages, so this is the same wire shape Hermes Agent uses.
* ``aux`` — ``harness.images.aux_model`` names an alias that can see: the image
  is described by that model (Hermes's prompt, verbatim) and only the text
  reaches the turn's model.
* ``drop`` — no route. The result says where the file is and that this model
  cannot see it.

The invariant, pinned by ``tests/test_tool_images.py``: **no image part is ever
sent toward an alias whose ``supports_vision`` is not literally ``true``.** An
engine started without multimodal input answers such a request with a 400 and
the turn dies; ``MultimodalRejectedError`` is the backstop for a config that
lies.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import re
import struct
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

from app.paths import SESSIONS_DIR

logger = logging.getLogger("lloyd-harness-images")

Route = Literal["native", "aux", "drop"]

EVICTED_PLACEHOLDER = "[screenshot removed to save context]"
UNCHANGED_NOTE = (
    "(screen unchanged since the previous capture — image omitted to save "
    "context; the previous screenshot still shows the current state. Element "
    "indices in this result are fresh and remain the preferred way to act.)"
)

# Hermes Agent's describe prompt (tools/computer_use/tool.py, MIT), verbatim.
AUX_PROMPT = (
    "Describe what is visible in this desktop application screenshot in "
    "concise but specific terms. Mention the app name and window title if "
    "visible, the overall layout, any labelled buttons, menus or text fields, "
    "and any prominent text content the user would need to know about. Do not "
    "invent details that are not actually visible.\n\nAX/SOM index for "
    "cross-reference:\n"
)

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "route": "auto",            # native | aux | auto | drop
    "aux_model": "",
    "max_outbound": 20,
    "max_bytes": 24 * 1024 * 1024,
    "evict_batch": 8,
    "keep_on_compaction": 3,
    "token_estimate": 1500,
    "dedup_max_consecutive": 2,
    "aux_max_tokens": 700,
    "aux_timeout_s": 120.0,
}

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]")
_MIME_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg",
             "image/webp": "webp", "image/gif": "gif"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _config() -> dict[str, Any]:
    try:
        from app.config import CONFIG
        return CONFIG if isinstance(CONFIG, dict) else {}
    except Exception:   # pragma: no cover - harness used without app config
        return {}


def images_cfg() -> dict[str, Any]:
    """`harness.images` over the defaults. Read at call time, never cached."""
    raw = ((_config().get("harness") or {}).get("images") or {})
    out = dict(DEFAULTS)
    if isinstance(raw, dict):
        out.update({k: v for k, v in raw.items() if v is not None})
    return out


def _model_entry(model: str) -> dict[str, Any] | None:
    models = _config().get("models") or {}
    if not isinstance(models, dict) or not model:
        return None
    if isinstance(models.get(model), dict):
        return models[model]
    for entry in models.values():
        if isinstance(entry, dict) and entry.get("alias") == model:
            return entry
    return None


def model_supports_vision(model: str) -> bool:
    """True only when the slot's config says ``supports_vision: true``."""
    entry = _model_entry(model)
    return bool(entry) and entry.get("supports_vision") is True


def image_token_estimate(model: str = "") -> int:
    entry = _model_entry(model) or {}
    val = entry.get("image_token_estimate")
    if isinstance(val, (int, float)) and val > 0:
        return int(val)
    return int(images_cfg().get("token_estimate") or DEFAULTS["token_estimate"])


def resolve_image_route(model: str, cfg: dict[str, Any] | None = None) -> Route:
    """Which way an image travels for a turn running on ``model``."""
    cfg = cfg or images_cfg()
    if not cfg.get("enabled", True):
        return "drop"
    route = str(cfg.get("route") or "auto").strip().lower()
    aux = str(cfg.get("aux_model") or "").strip()
    aux_ok = bool(aux) and model_supports_vision(aux)
    if route == "drop":
        return "drop"
    if route in ("native", "auto") and model_supports_vision(model):
        return "native"
    if route == "native":
        logger.warning(
            "images: route=native but models.%s.supports_vision is not true — "
            "refusing to send image parts to it", model)
    if route in ("aux", "auto", "native") and aux_ok:
        return "aux"
    if route == "aux" and aux and not aux_ok:
        logger.warning("images: aux_model %r is not supports_vision: true", aux)
    return "drop"


# ---------------------------------------------------------------------------
# Persisting
# ---------------------------------------------------------------------------

def _dims(raw: bytes) -> tuple[int, int]:
    """Width and height from a PNG, JPEG or GIF header. (0, 0) if unknown.

    Parsed by hand because the backend venv has no Pillow, and the header is
    all a ref needs.
    """
    try:
        if raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[12:16] == b"IHDR":
            w, h = struct.unpack(">II", raw[16:24])
            return int(w), int(h)
        if raw[:6] in (b"GIF87a", b"GIF89a"):
            w, h = struct.unpack("<HH", raw[6:10])
            return int(w), int(h)
        if raw[:2] == b"\xff\xd8":
            i = 2
            while i + 9 < len(raw):
                if raw[i] != 0xFF:
                    i += 1
                    continue
                marker = raw[i + 1]
                if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                    i += 2
                    continue
                seg = struct.unpack(">H", raw[i + 2:i + 4])[0]
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    h, w = struct.unpack(">HH", raw[i + 5:i + 9])
                    return int(w), int(h)
                i += 2 + seg
    except Exception:
        pass
    return 0, 0


def _spill_dir(session_id: str) -> Path:
    # Same directory the text spills use (tool_result_spill._spill_dir).
    return SESSIONS_DIR / f"{session_id}.tool-results"


_SCALE_KEYS = ("orig_width", "orig_height", "scale")


_CAPTURE_HEAD = re.compile(r"image=(\d+)x(\d+) screen=(\d+)x(\d+) scale=([\d.]+)")


def _scale_hints(text: str) -> dict[str, Any]:
    """Scaling facts a tool put in its own text, if it did.

    Either a JSON object with ``orig_width``/``orig_height``/``scale``, or the
    desktop capture's header line (``image=WxH screen=WxH scale=F``).
    """
    m = _CAPTURE_HEAD.search(text[:600])
    if m:
        return {"orig_width": int(m.group(3)), "orig_height": int(m.group(4)),
                "scale": float(m.group(5))}
    try:
        obj = json.loads(text)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out = {}
    for k in _SCALE_KEYS:
        v = obj.get(k)
        if isinstance(v, (int, float)):
            out[k] = v
    return out


def persist_tool_images(
    images: list[dict[str, Any]],
    *,
    session_id: str,
    call_id: str,
    text: str = "",
) -> list[dict[str, Any]]:
    """Write each image under ``<sid>.tool-results/`` and return its ref."""
    refs: list[dict[str, Any]] = []
    if not images:
        return refs
    safe_call = _SAFE_NAME.sub("_", call_id or "call")[:120] or "call"
    safe_sid = _SAFE_NAME.sub("_", session_id or "nosession")[:160] or "nosession"
    d = _spill_dir(safe_sid)
    hints = _scale_hints(text) if text else {}
    for n, img in enumerate(images):
        data = img.get("data") or ""
        mime = str(img.get("mime_type") or "image/png").lower()
        try:
            raw = base64.b64decode(data, validate=False)
        except (binascii.Error, ValueError):
            logger.warning("images: undecodable base64 from %s", call_id)
            continue
        if not raw:
            continue
        ext = _MIME_EXT.get(mime, "bin")
        path = d / f"{safe_call}.img{n}.{ext}"
        try:
            d.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        except OSError as exc:
            logger.warning("images: could not persist %s: %s", path, exc)
            continue
        w, h = _dims(raw)
        ref: dict[str, Any] = {
            "path": str(path),
            "name": path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "mime": mime,
            "width": w,
            "height": h,
            "bytes": len(raw),
        }
        ref.update(hints)
        refs.append(ref)
    return refs


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

class _Dedup:
    """Per (session, tool): last image sha and how many omissions in a row.

    Module-level so a session's second turn remembers the first turn's last
    frame. Bounded LRU; a restart forgets, which costs one re-sent image.
    """

    def __init__(self, cap: int = 512) -> None:
        self._d: OrderedDict[tuple[str, str], tuple[str, int, str]] = OrderedDict()
        self._lock = threading.Lock()
        self._cap = cap

    def check(self, session_id: str, tool: str, ref: dict[str, Any],
              max_consecutive: int) -> dict[str, Any] | None:
        """The previous ref's name if this frame repeats it and may be omitted."""
        key = (session_id or "", tool or "")
        sha = ref.get("sha256", "")
        with self._lock:
            prev = self._d.get(key)
            if prev and prev[0] == sha and prev[1] < max_consecutive:
                self._d[key] = (sha, prev[1] + 1, prev[2])
                self._d.move_to_end(key)
                return {"deduped_from": prev[2]}
            self._d[key] = (sha, 0, ref.get("name", ""))
            self._d.move_to_end(key)
            while len(self._d) > self._cap:
                self._d.popitem(last=False)
        return None

    def reset(self, session_id: str | None = None) -> None:
        with self._lock:
            if session_id is None:
                self._d.clear()
                return
            for k in [k for k in self._d if k[0] == session_id]:
                del self._d[k]


DEDUP = _Dedup()


# ---------------------------------------------------------------------------
# Wire
# ---------------------------------------------------------------------------

_bytes_cache: OrderedDict[tuple[str, str], str] = OrderedDict()
_bytes_lock = threading.Lock()


def data_url(ref: dict[str, Any]) -> str | None:
    """``data:<mime>;base64,…`` from the file on disk, cached.

    Read back rather than re-encoded so the bytes are identical on every
    iteration — vLLM keys its multimodal prefix cache on content.
    """
    key = (str(ref.get("path") or ""), str(ref.get("sha256") or ""))
    with _bytes_lock:
        hit = _bytes_cache.get(key)
        if hit is not None:
            _bytes_cache.move_to_end(key)
            return hit
    try:
        raw = Path(key[0]).read_bytes()
    except OSError:
        return None
    url = f"data:{ref.get('mime') or 'image/png'};base64," + base64.b64encode(raw).decode()
    with _bytes_lock:
        _bytes_cache[key] = url
        while len(_bytes_cache) > 48:
            _bytes_cache.popitem(last=False)
    return url


def wire_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The list to put on the wire: refs become ``image_url`` parts.

    A message without ``_image_refs`` is passed through as the same object,
    so every request that carries no image is byte-identical to before. The
    private key is never sent.
    """
    if not any(isinstance(m, dict) and "_image_refs" in m for m in messages):
        return messages
    out: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict) or "_image_refs" not in m:
            out.append(m)
            continue
        refs = m.get("_image_refs") or []
        msg = {k: v for k, v in m.items() if k != "_image_refs"}
        parts: list[dict[str, Any]] = []
        for ref in refs:
            url = data_url(ref)
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})
        if parts:
            text = msg.get("content")
            if not isinstance(text, str):
                text = "" if text is None else json.dumps(text)
            msg["content"] = [{"type": "text", "text": text}] + parts
        out.append(msg)
    return out


def payload_has_images(messages: list[dict[str, Any]]) -> bool:
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list) and any(
                isinstance(p, dict) and p.get("type") == "image_url" for p in c):
            return True
    return False


def looks_like_multimodal_rejection(body_text: str) -> bool:
    t = (body_text or "").lower()
    return any(k in t for k in (
        "image", "multimodal", "multi-modal", "vision", "limit_mm_per_prompt",
        "mm_per_prompt", "language-model-only", "language_model_only"))


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------

def _ref_count_and_bytes(messages: list[dict[str, Any]]) -> tuple[int, int]:
    n = b = 0
    for m in messages:
        for r in (m.get("_image_refs") or []) if isinstance(m, dict) else []:
            n += 1
            b += int(r.get("bytes") or 0)
    return n, b


def _evict(m: dict[str, Any]) -> None:
    m.pop("_image_refs", None)
    c = m.get("content")
    if isinstance(c, str) and EVICTED_PLACEHOLDER not in c:
        m["content"] = f"{c}\n{EVICTED_PLACEHOLDER}"


def enforce_outbound_cap(messages: list[dict[str, Any]], *,
                         protect_from: int | None = None,
                         cfg: dict[str, Any] | None = None) -> int:
    """Retire the oldest image-bearing messages once the cap is crossed.

    Nothing is rewritten below the cap — that is what keeps the prefix cache
    intact across ordinary iterations. Past it, the oldest ``evict_batch``
    image-bearing messages lose their refs in one step (a step function, so
    the prefix is rewritten once per batch, not once per image). Index 0 and
    everything at or after ``protect_from`` are never touched.
    """
    cfg = cfg or images_cfg()
    max_n = int(cfg.get("max_outbound") or 20)
    max_b = int(cfg.get("max_bytes") or DEFAULTS["max_bytes"])
    batch = max(1, int(cfg.get("evict_batch") or 8))
    n, b = _ref_count_and_bytes(messages)
    if n <= max_n and b <= max_b:
        return 0
    limit = len(messages) if protect_from is None else max(1, protect_from)
    evicted = 0
    for i in range(1, limit):
        m = messages[i]
        if isinstance(m, dict) and m.get("_image_refs"):
            _evict(m)
            evicted += 1
            n, b = _ref_count_and_bytes(messages)
            if evicted >= batch and n <= max_n and b <= max_b:
                break
    if evicted:
        logger.info("images: evicted %d image-bearing message(s) past the cap", evicted)
    return evicted


def keep_newest(messages: list[dict[str, Any]], keep: int) -> int:
    """Drop refs from all but the newest ``keep`` image-bearing messages."""
    idx = [i for i, m in enumerate(messages)
           if isinstance(m, dict) and m.get("_image_refs")]
    drop = idx[:-keep] if keep > 0 else idx
    for i in drop:
        _evict(messages[i])
    return len(drop)


def strip_all_image_refs(messages: list[dict[str, Any]]) -> int:
    n = 0
    for m in messages:
        if isinstance(m, dict) and "_image_refs" in m:
            m.pop("_image_refs", None)
            n += 1
    return n


# ---------------------------------------------------------------------------
# History rebuild
# ---------------------------------------------------------------------------

def refs_for_history(row: dict[str, Any], route: Route) -> list[dict[str, Any]]:
    """Refs from a persisted tool row that should ride into this turn."""
    if route != "native":
        return []
    out = []
    for r in row.get("images") or []:
        if not isinstance(r, dict) or r.get("described") or r.get("deduped_from"):
            continue
        if r.get("evicted") or not r.get("path") or not Path(r["path"]).exists():
            continue
        out.append({k: r[k] for k in ("path", "sha256", "mime", "width",
                                      "height", "bytes") if k in r})
    return out


def row_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What a session JSON row keeps for an image: the ref, never the bytes."""
    keep = ("path", "name", "sha256", "mime", "width", "height", "bytes",
            "orig_width", "orig_height", "scale", "described", "deduped_from",
            "route")
    return [{k: r[k] for k in keep if k in r} for r in refs if isinstance(r, dict)]


# ---------------------------------------------------------------------------
# Aux describer
# ---------------------------------------------------------------------------

def _scale_note(ref: dict[str, Any]) -> str:
    ow, oh = ref.get("orig_width"), ref.get("orig_height")
    w, h = ref.get("width"), ref.get("height")
    if ow and oh and w and h and (ow != w or oh != h):
        f = float(ow) / float(w)
        return (f"Screenshot downscaled from {ow}x{oh} to {w}x{h} for vision; "
                f"multiply any coordinates you report by {f:.2f} to map back to "
                f"the real screen.")
    return ""


async def describe_image(ref: dict[str, Any], element_summary: str, *,
                         model_alias: str, session_id: str = "") -> str:
    """One non-streaming call to a vision alias. ``""`` on any failure."""
    import httpx

    entry = _model_entry(model_alias) or {}
    base_url = str(entry.get("base_url") or "").rstrip("/")
    url_img = data_url(ref)
    if not base_url or not url_img:
        return ""
    cfg = images_cfg()
    note = _scale_note(ref)
    prompt = AUX_PROMPT + (element_summary or "(none)")[:6000] + (
        f"\n\nNote: {note}" if note else "")
    payload = {
        "model": entry.get("alias") or model_alias,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": url_img}},
        ]}],
        "max_tokens": int(cfg.get("aux_max_tokens") or 700),
        "temperature": 0.1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        from app.component_manifest import record_request
        record_request(base_url=base_url, model=payload["model"], payload=payload,
                       session_id=session_id,
                       send_site="app/harness/tool_images.py::describe_image")
    except Exception:
        pass
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(
                float(cfg.get("aux_timeout_s") or 120.0))) as cli:
            resp = await cli.post(f"{base_url}/v1/chat/completions", json=payload,
                                  headers={"Authorization": "Bearer no-key-required"})
        if resp.status_code >= 400:
            logger.warning("images: aux describe HTTP %s: %s",
                           resp.status_code, resp.text[:200])
            return ""
        body = resp.json()
        return str(body["choices"][0]["message"].get("content") or "").strip()
    except Exception as exc:
        logger.warning("images: aux describe failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# The one entry point the loop calls
# ---------------------------------------------------------------------------

async def shape_tool_images(
    *,
    images: list[dict[str, Any]],
    content: str,
    session_id: str,
    call_id: str,
    tool_name: str,
    model: str,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Persist, dedup and route one result's images.

    Returns ``(content, event_refs, history_refs)``: the text the model is
    shown, the refs the event/row carry (for the UI and the record), and the
    refs to hang on the history message (non-empty only on the native route).
    Never raises — a failure here costs the image, not the tool result.
    """
    try:
        cfg = images_cfg()
        if not cfg.get("enabled", True):
            return content, [], []
        refs = persist_tool_images(images, session_id=session_id,
                                   call_id=call_id, text=content)
        if not refs:
            return content, [], []
        route = resolve_image_route(model, cfg)
        maxc = int(cfg.get("dedup_max_consecutive") or 0)
        send: list[dict[str, Any]] = []
        for ref in refs:
            ref["route"] = route
            dup = DEDUP.check(session_id, tool_name, ref, maxc) if maxc > 0 else None
            if dup:
                ref.update(dup)
                content = f"{content}\n{UNCHANGED_NOTE}"
                continue
            send.append(ref)
        if route == "native":
            return content, refs, [dict(r) for r in send]
        if route == "aux" and send:
            alias = str(cfg.get("aux_model") or "")
            desc = await describe_image(send[0], content, model_alias=alias,
                                        session_id=session_id)
            for r in send:
                r["described"] = True
            if desc:
                content = f"{content}\n\n[screenshot described by {alias}]\n{desc}"
            else:
                content = (f"{content}\n(vision unavailable: {alias} could not "
                           f"describe the screenshot; drive by the element list)")
            return content, refs, []
        if send:
            content = (f"{content}\n[image saved to {send[0]['path']}; the model "
                       f"'{model}' has no vision route configured, so it was not "
                       f"shown. Drive by the text/element list.]")
        return content, refs, []
    except Exception:   # pragma: no cover - defensive
        logger.exception("images: shaping failed for %s", tool_name)
        return content, [], []
