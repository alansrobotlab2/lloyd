"""Lloyd MCP Server: IDE drive — tell the user's IDE what to do.

Three tools:
  ide_open_folder — change the IDE's open folder.
  ide_open_file   — open a file in a new editor tab (or focus existing).
  ide_close_tab   — close an IDE editor tab.

These tools push commands into the FastAPI backend's MC bus, which fans
out to subscribed frontends over SSE. Read-side IDE state (what's open,
what's visible) is already surfaced via mc_get_state and via the
<ide_state> block injected by prefetch — we don't duplicate that here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
from mcp.server import Server
from mcp.types import Tool, TextContent

from agent_mcp._shared import _err, _wrap, ErrorCode

LLOYD_API = os.environ.get("LLOYD_API_URL", "http://127.0.0.1:8080")

app = Server("lloyd-ide")


def _expand(path: str) -> str:
    if not path:
        return path
    return os.path.expanduser(os.path.expandvars(path))


def _ensure_abs(path: str, field: str = "path") -> tuple[str, dict | None]:
    expanded = _expand(path or "")
    if not expanded:
        return "", _err(f"{field} is required", ErrorCode.MISSING_PARAM)
    if not os.path.isabs(expanded):
        return "", _err(
            f"{field} must be absolute, got {expanded!r}",
            ErrorCode.INVALID_PARAM,
        )
    return os.path.normpath(expanded), None


async def _ide_open_folder(params: dict) -> dict:
    raw = params.get("path") or ""
    abs_path, err = _ensure_abs(raw, "path")
    if err:
        return err
    p = Path(abs_path)
    if not p.exists():
        return _err(f"folder does not exist: {abs_path}", ErrorCode.NOT_FOUND)
    if not p.is_dir():
        return _err(f"path is not a directory: {abs_path}", ErrorCode.INVALID_PARAM)

    # 1) Push the open_folder action into the MC bus so any subscribed
    #    frontend swaps its IDE folder.
    # 2) Switch the user's current tab to the IDE so they actually see it.
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(
                f"{LLOYD_API}/api/mc/ide_action",
                json={"kind": "open_folder", "path": abs_path},
            )
            r.raise_for_status()
            await client.post(
                f"{LLOYD_API}/api/mc/navigate",
                json={"tab": "ide"},
            )
    except Exception as e:
        return _err(f"ide_open_folder failed: {e}", ErrorCode.INTERNAL)

    # Top-level listing as a confirmation hint — saves a follow-up round-trip
    # when the agent wants to immediately reason about what's in the folder.
    root_entries: list[dict[str, Any]] = []
    try:
        with os.scandir(p) as it:
            for de in it:
                try:
                    is_dir = de.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                root_entries.append({"name": de.name, "isDir": is_dir})
                if len(root_entries) >= 50:
                    break
    except OSError:
        pass
    root_entries.sort(key=lambda e: (0 if e["isDir"] else 1, e["name"].lower()))

    return {"open_folder": abs_path, "root_entries": root_entries}


async def _ide_open_file(params: dict) -> dict:
    raw = params.get("path") or ""
    abs_path, err = _ensure_abs(raw, "path")
    if err:
        return err
    p = Path(abs_path)
    if not p.exists():
        return _err(f"file does not exist: {abs_path}", ErrorCode.NOT_FOUND)
    if p.is_dir():
        return _err(
            f"path is a directory; use ide_open_folder for folders: {abs_path}",
            ErrorCode.INVALID_PARAM,
        )

    # Re-use the existing pendingFocus channel: mc_navigate(tab=ide,
    # focus_id=<path>) makes the frontend's IdePage open the requested
    # file. Same wire shape as memory tab opening a vault path.
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(
                f"{LLOYD_API}/api/mc/navigate",
                json={"tab": "ide", "focus_id": abs_path},
            )
            r.raise_for_status()
    except Exception as e:
        return _err(f"ide_open_file failed: {e}", ErrorCode.INTERNAL)

    language = _language_for(abs_path)
    return {"path": abs_path, "opened": True, "language": language}


async def _ide_close_tab(params: dict) -> dict:
    raw = params.get("path") or ""
    abs_path, err = _ensure_abs(raw, "path")
    if err:
        return err
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(
                f"{LLOYD_API}/api/mc/ide_action",
                json={"kind": "close_tab", "path": abs_path},
            )
            r.raise_for_status()
    except Exception as e:
        return _err(f"ide_close_tab failed: {e}", ErrorCode.INTERNAL)
    return {"path": abs_path, "closed": True}


_LANG_BY_EXT = {
    ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".py": "python",
    ".rs": "rust",
    ".go": "go",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".java": "java",
    ".rb": "ruby",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml",
    ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "scss",
    ".sql": "sql",
    ".xml": "xml",
}


def _language_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return _LANG_BY_EXT.get(ext, "plaintext")


_handlers = {
    "ide_open_folder": _ide_open_folder,
    "ide_open_file": _ide_open_file,
    "ide_close_tab": _ide_close_tab,
}


@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="ide_open_folder",
            description=(
                "Set the folder shown in the IDE tab's file tree. The user's "
                "current tab is also switched to IDE so they immediately see "
                "the change. Returns the new open_folder and a small "
                "root_entries hint of what's inside.\n\n"
                "Path must be absolute. ~ and $VARS are expanded."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute folder path to open in the IDE.",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="ide_open_file",
            description=(
                "Open a file in the IDE tab — adds an editor tab and focuses "
                "it (or just focuses if already open). The user's IDE folder "
                "should usually contain this file, but any absolute path "
                "works. Returns the language Monaco will use.\n\n"
                "Use this whenever the user asks to \"pull up\", \"show\", "
                "\"open\" a file. For directories, use ide_open_folder."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute file path to open in a new editor tab.",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="ide_close_tab",
            description=(
                "Close the IDE editor tab for the given path. No-op if no "
                "tab matches that path; if the closed tab is currently "
                "visible, the IDE picks an adjacent tab to show."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute file path of the tab to close.",
                    },
                },
                "required": ["path"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    handler = _handlers.get(name)
    if not handler:
        return _wrap(_err(f"Unknown tool: {name}", ErrorCode.UNKNOWN_TOOL))
    result = await handler(arguments or {})
    return _wrap(result)
