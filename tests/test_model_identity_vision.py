"""A slot that claims vision must be served by an engine that takes images.

`models.<alias>.supports_vision: true` is true only while the primary's conf
keeps `LANGUAGE_MODEL_ONLY=0`; the launcher's default is text-only, so an A/B
arm, a rollback or a reverted conf leaves the flag behind an engine that
refuses every screenshot, while `status` and `kv_status` both read ok (#1420).
The engine publishes no modality list, so the identity sweep asks it with one
image, and these tests answer through a scripted loopback engine rather than
a stubbed probe — the request shape and the refusal wording are what is under
test.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import model_identity as mi

# The engine's own words for a text-only boot (tests/test_multimodal_recovery.py).
ENGINE_400 = ("This model does not support image input. Set "
              "LANGUAGE_MODEL_ONLY=0 to serve vision requests.")


class _Engine:
    """Answers /v1/models like vLLM and /v1/chat/completions from a script."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.chat_bodies: list[dict] = []
        engine = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/v1/models":
                    self._send(200, {"data": [{"id": "primary",
                                               "root": "/m/Qwen3.8-Flash-Next-nvfp4"}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                engine.chat_bodies.append(json.loads(self.rfile.read(n) or b"{}"))
                code = engine.answers.pop(0) if engine.answers else 200
                if code == 200:
                    self._send(200, {"choices": [{"message": {"content": "."}}]})
                else:
                    self._send(code, {"object": "error", "message": ENGINE_400,
                                      "type": "BadRequestError", "code": 400})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def image_requests(self) -> int:
        return sum(1 for b in self.chat_bodies
                   for m in b.get("messages") or []
                   if isinstance(m.get("content"), list)
                   and any(p.get("type") == "image_url" for p in m["content"]))


@pytest.fixture
def engine():
    made = []

    def make(*answers):
        e = _Engine(answers)
        made.append(e)
        return e
    yield make
    for e in made:
        e.close()


@pytest.fixture(autouse=True)
def _clear_cache():
    mi.LAST_RESULT.clear()
    yield
    mi.LAST_RESULT.clear()


def _slots(monkeypatch, **slots):
    monkeypatch.setattr("app.config.MODEL_CONFIGS", slots, raising=False)


def _primary(url, **extra):
    return {"alias": "primary", "base_url": url, "expect_model": "Qwen3.8-Flash-Next", **extra}


@pytest.mark.asyncio
async def test_a_slot_not_claiming_vision_is_never_sent_an_image(engine, monkeypatch):
    e = engine()
    _slots(monkeypatch, primary=_primary(e.url),
           secondary=_primary(e.url, alias="secondary", supports_vision="true"))
    rows = await mi.verify_models()
    assert [r["image_status"] for r in rows] == ["unchecked", "unchecked"]
    assert all(r["supports_vision"] is False for r in rows), "only literal True claims it"
    assert e.image_requests() == 0 and e.chat_bodies == []


@pytest.mark.asyncio
async def test_the_engine_answer_decides_accepted_then_refused(engine, monkeypatch):
    e = engine(200, 400)
    _slots(monkeypatch, primary=_primary(e.url, supports_vision=True))
    rows = await mi.verify_models()
    assert rows[0]["status"] == "ok" and rows[0]["image_status"] == "ok"
    rows = await mi.verify_models()
    assert rows[0]["status"] == "ok"                        # the right model...
    assert rows[0]["image_status"] == "REFUSED"             # ...without its tower
    assert "does not support image input" in rows[0]["image_detail"]
    assert e.image_requests() == 2
    body = e.chat_bodies[0]
    assert body["model"] == "primary" and body["max_tokens"] == 1
    assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_refused_is_distinct_from_down_and_from_other_errors(engine, monkeypatch):
    e = engine(500)
    _slots(monkeypatch, primary=_primary(e.url, supports_vision=True))
    rows = await mi.verify_models()
    assert rows[0]["image_status"] == "unknown", "a 5xx is not a refusal"
    e.close()
    rows = await mi.verify_models()
    assert rows[0]["image_status"] == "unreachable"


def test_the_identity_route_carries_the_verdict_on_every_row(engine, monkeypatch):
    """A caller learns the flag lies from the response alone — no image of its own."""
    from app.routers import models as models_router

    e = engine(400)
    _slots(monkeypatch, primary=_primary(e.url, supports_vision=True),
           secondary={"alias": "secondary", "base_url": ""})
    app = FastAPI()
    app.include_router(models_router.router)
    body = TestClient(app).get("/api/models/identity?refresh=1").json()
    assert all("image_status" in r for r in body["models"])
    by = {r["alias"]: r for r in body["models"]}
    assert by["primary"]["image_status"] == "REFUSED"
    assert by["secondary"]["image_status"] == "unchecked"
    assert body["image_refused"] == ["primary"]


@pytest.mark.asyncio
async def test_the_boot_sweep_logs_a_refusal_in_the_mismatch_ladder(engine, monkeypatch, caplog):
    e = engine(400)
    _slots(monkeypatch, primary=_primary(e.url, supports_vision=True))
    with caplog.at_level(logging.INFO, logger="lloyd-server"):
        rows = await mi.verify_models_with_retry(attempts=1)
    assert rows[0]["image_status"] == "REFUSED"
    errors = [r.getMessage() for r in caplog.records
              if r.levelno == logging.ERROR and r.name == "lloyd-server"]
    assert len(errors) == 1, errors
    assert "image input REFUSED" in errors[0]
    assert "models.primary.supports_vision" in errors[0]
    assert "LANGUAGE_MODEL_ONLY=0" in errors[0]
