"""LSP WebSocket proxy — bridges Monaco's language client to local LSP servers.

Clients connect to:
    ws://.../api/lsp/{language}?workspace=<abs_path>

Each connection spawns its own LSP subprocess. v1 trade-off: one process
per browser tab is wasteful but stateless multiplexing across clients is
LSP-incorrect (the server thinks one client is editing each file). At a
single-user scale this is fine.

Wire format: each WebSocket message is one JSON-RPC envelope (no
Content-Length header on the WS side — vscode-ws-jsonrpc handles framing
on the client side). On the LSP side we add/strip the Content-Length
header. monaco-languageclient closes the WS on shutdown which terminates
the proc.

Currently supported:
    python      → pyright-langserver --stdio  (from the lloyd venv)
    typescript  → typescript-language-server --stdio  (via npx -y)
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("lloyd-server.lsp")

router = APIRouter()

# Resolve absolute paths to the binaries so subprocess launches don't
# depend on whoever launched the FastAPI process having the right PATH.
_LLOYD_VENV_BIN = Path(__file__).resolve().parent.parent.parent / ".venvs" / "lloyd" / "bin"
_PYRIGHT_LANGSERVER = _LLOYD_VENV_BIN / "pyright-langserver"


def _resolve_typescript_ls() -> Optional[list[str]]:
    """Locate the typescript-language-server invocation.

    Prefer a direct binary on PATH; fall back to `npx -y typescript-language-server`
    so first-run still works on a clean system (npx will fetch from cache).
    """
    direct = shutil.which("typescript-language-server")
    if direct:
        return [direct, "--stdio"]
    npx = shutil.which("npx")
    if npx:
        return [npx, "-y", "typescript-language-server", "--stdio"]
    return None


_LANGUAGES: dict[str, callable] = {
    # Each entry returns the argv list to spawn, or None if unavailable.
    "python": lambda: (
        [str(_PYRIGHT_LANGSERVER), "--stdio"]
        if _PYRIGHT_LANGSERVER.exists() else None
    ),
    "typescript": _resolve_typescript_ls,
}


async def _spawn_ls(language: str, workspace: str) -> Optional[asyncio.subprocess.Process]:
    resolver = _LANGUAGES.get(language)
    if resolver is None:
        return None
    argv = resolver()
    if argv is None:
        logger.warning("lsp: %s server not available", language)
        return None
    cwd = workspace if workspace and os.path.isdir(workspace) else None
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
    except Exception as e:
        logger.warning("lsp: spawn %s failed: %s", language, e)
        return None
    logger.info("lsp: spawned %s (pid=%s, workspace=%s)", language, proc.pid, cwd)
    return proc


def _parse_content_length(headers: bytes) -> Optional[int]:
    for line in headers.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            try:
                return int(line.split(b":", 1)[1].strip())
            except ValueError:
                return None
    return None


@router.websocket("/api/lsp/{language}")
async def lsp_ws(ws: WebSocket, language: str, workspace: str = ""):
    await ws.accept()

    proc = await _spawn_ls(language, workspace)
    if proc is None:
        await ws.close(code=1011, reason=f"language {language} unavailable")
        return

    assert proc.stdin is not None and proc.stdout is not None

    async def ws_to_proc() -> None:
        """Read JSON-RPC text frames from WS, frame them as LSP, write to stdin."""
        try:
            while True:
                msg = await ws.receive_text()
                body = msg.encode("utf-8")
                header = f"Content-Length: {len(body)}\r\n\r\n".encode()
                proc.stdin.write(header + body)
                await proc.stdin.drain()
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug("lsp ws→proc loop ended: %s", e)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    async def proc_to_ws() -> None:
        """Read framed LSP messages from stdout, strip header, send as WS text."""
        try:
            while True:
                # Accumulate header up through the blank line.
                header = b""
                while b"\r\n\r\n" not in header:
                    chunk = await proc.stdout.readline()
                    if not chunk:
                        return
                    header += chunk
                content_length = _parse_content_length(header)
                if content_length is None:
                    logger.warning("lsp proc→ws: no Content-Length, dropping frame")
                    continue
                body = await proc.stdout.readexactly(content_length)
                await ws.send_text(body.decode("utf-8"))
        except asyncio.IncompleteReadError:
            pass
        except Exception as e:
            logger.debug("lsp proc→ws loop ended: %s", e)

    async def stderr_drain() -> None:
        """Forward LSP stderr to logs so misconfigurations are visible."""
        if proc.stderr is None:
            return
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                logger.debug("lsp[%s] stderr: %s", language, line.decode("utf-8", "replace").rstrip())
        except Exception:
            pass

    t1 = asyncio.create_task(ws_to_proc())
    t2 = asyncio.create_task(proc_to_ws())
    t3 = asyncio.create_task(stderr_drain())

    try:
        # Whichever pump exits first ends the session. The other gets cancelled.
        done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
    finally:
        try:
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
        except Exception:
            pass
        t3.cancel()
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("lsp: closed %s session (pid=%s)", language, proc.pid)
