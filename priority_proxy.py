#!/usr/bin/env python3
"""Priority-injecting HTTP proxy for vLLM.

Intercepts requests and injects a `priority` field into the JSON body before
forwarding to the upstream vLLM endpoint.  vLLM's priority scheduler
(--scheduling-policy priority) treats lower numbers as higher priority — 0
is highest.

Interactive sessions → direct vLLM endpoint (default / priority 0)
Background tasks (pipeline, autonomy) → this proxy (priority 1)

Usage:
    python priority_proxy.py --port 8097 --upstream http://127.0.0.1:8096 --priority 1
    python priority_proxy.py --port 8092 --upstream http://127.0.0.1:8091 --priority 1
"""

import argparse
import json
import logging

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("priority-proxy")

# Set by main() before uvicorn starts — read by route handler at call time.
UPSTREAM: str = ""
PRIORITY: int = 1

_HOP_BY_HOP = frozenset([
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
])

app = FastAPI(docs_url=None, redoc_url=None)


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
)
async def proxy(request: Request, path: str):
    body = await request.body()

    # Inject priority into JSON request bodies only
    ct = request.headers.get("content-type", "")
    if body and "application/json" in ct:
        try:
            data = json.loads(body)
            data["priority"] = PRIORITY
            body = json.dumps(data).encode()
        except (json.JSONDecodeError, ValueError):
            pass  # Non-JSON body — forward unchanged

    upstream_url = f"{UPSTREAM}/{path}"
    if request.url.query:
        upstream_url += f"?{request.url.query}"

    fwd_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP | {"host"}
    }
    if body:
        fwd_headers["content-length"] = str(len(body))

    logger.debug("→ %s /%s (priority=%d, %d bytes)", request.method, path, PRIORITY, len(body))

    client = httpx.AsyncClient(timeout=None)
    req = client.build_request(
        method=request.method,
        url=upstream_url,
        headers=fwd_headers,
        content=body,
    )
    resp = await client.send(req, stream=True)

    resp_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

    async def body_stream():
        try:
            async for chunk in resp.aiter_bytes(chunk_size=4096):
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        content=body_stream(),
        status_code=resp.status_code,
        headers=resp_headers,
        media_type=resp.headers.get("content-type"),
    )


def main():
    global UPSTREAM, PRIORITY

    parser = argparse.ArgumentParser(description="Priority-injecting proxy for vLLM")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")
    parser.add_argument("--upstream", required=True, help="Upstream vLLM base URL (e.g. http://127.0.0.1:8096)")
    parser.add_argument("--priority", type=int, default=1, help="Priority to inject into requests (default: 1)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    args = parser.parse_args()

    UPSTREAM = args.upstream.rstrip("/")
    PRIORITY = args.priority

    logger.info(
        "Priority proxy: %s:%d → %s (injecting priority=%d)",
        args.host, args.port, UPSTREAM, PRIORITY,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
