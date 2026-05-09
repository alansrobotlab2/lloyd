"""inotify-based file watcher for the IDE tab.

Watches the IDE's currently-open folder recursively and publishes
`file_changed` events to the MC SSE bus when files inside it are
modified, created, deleted, or renamed.

Single-rooted: when the user changes the open folder, the watcher
rebinds. Cross-process safety isn't needed — there's one Lloyd backend
and one Observer thread.

Skips the obvious noise dirs (node_modules, .git, .venv, __pycache__,
dist, build) at scheduling time so the descriptor count stays bounded
even on big repos.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from app import mc_state

logger = logging.getLogger("lloyd-server.file_watcher")

# Names that are nearly always pure noise inside an IDE folder. Skipping
# them at scheduling time avoids both the inotify descriptor cost and the
# blizzard of events from build/install activity.
_SKIP_NAMES = {
    "node_modules", ".git", ".venv", ".venvs", "venv",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "dist", "build", ".next", ".turbo", ".cache",
    "target",  # Rust
}

# Per-path debounce so atomic-save bursts (write-temp + rename + chmod)
# collapse into one published event.
_DEBOUNCE_SECONDS = 0.15


class _Handler(FileSystemEventHandler):
    """Translates watchdog events into mc_state.publish_file_changed calls."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self._loop = loop
        self._lock = threading.Lock()
        # path -> {"timer": Timer, "deleted": bool}
        self._pending: dict[str, dict] = {}

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._enqueue(event.src_path, deleted=False)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._enqueue(event.src_path, deleted=False)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._enqueue(event.src_path, deleted=True)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        # Treat moves as delete-old + create-new so any open tab matching
        # the old path is told the file is gone, and the new path
        # publishes as a normal change.
        self._enqueue(event.src_path, deleted=True)
        dest = getattr(event, "dest_path", "") or ""
        if dest:
            self._enqueue(dest, deleted=False)

    def _enqueue(self, path: str, *, deleted: bool) -> None:
        if not path:
            return
        # Filter out noise dirs in the path lineage too — events can fire
        # before our scheduling-time filter has had a chance to skip them
        # (e.g. on initial recursive setup).
        parts = Path(path).parts
        if any(p in _SKIP_NAMES for p in parts):
            return

        with self._lock:
            existing = self._pending.get(path)
            if existing is not None:
                existing["timer"].cancel()
            timer = threading.Timer(
                _DEBOUNCE_SECONDS,
                self._flush,
                args=(path,),
            )
            # If a delete and a modify race for the same path within the
            # debounce window, prefer the most recent intent.
            self._pending[path] = {"timer": timer, "deleted": deleted}
            timer.daemon = True
            timer.start()

    def _flush(self, path: str) -> None:
        with self._lock:
            entry = self._pending.pop(path, None)
        if entry is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                mc_state.publish_file_changed(path, deleted=entry["deleted"]),
                self._loop,
            )
        except RuntimeError:
            # Loop is gone (shutdown) — drop silently.
            pass


class _Manager:
    """Singleton owning the current Observer.

    Rebinds atomically when the open folder changes. Safe to call
    `bind(None)` to fully detach.
    """

    def __init__(self) -> None:
        self._observer: Optional[Observer] = None
        self._handler: Optional[_Handler] = None
        self._root: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the FastAPI event loop so the handler can dispatch coros into it."""
        self._loop = loop

    def bind(self, root: Optional[str]) -> None:
        """Switch the watcher to a new folder (or detach if root is None).

        No-op if `root` is the same as the currently-watched folder.
        """
        with self._lock:
            if root == self._root:
                return
            self._tear_down_locked()
            if not root:
                return
            if self._loop is None:
                logger.warning("file_watcher.bind called before attach_loop — skipping")
                return
            try:
                root_p = Path(root)
                if not root_p.exists() or not root_p.is_dir():
                    logger.info("file_watcher.bind: not a directory, skipping: %s", root)
                    return
                handler = _Handler(self._loop)
                observer = Observer()
                # `recursive=True` plus our path-lineage filter in _enqueue
                # is simpler than walking the tree to skip noise dirs at
                # schedule time — watchdog still sees them, but the events
                # get dropped on the floor before publishing.
                observer.schedule(handler, str(root_p), recursive=True)
                observer.start()
                self._observer = observer
                self._handler = handler
                self._root = root
                logger.info("file_watcher: watching %s", root)
            except Exception as e:
                logger.warning("file_watcher.bind failed for %s: %s", root, e)
                self._tear_down_locked()

    def shutdown(self) -> None:
        with self._lock:
            self._tear_down_locked()

    def _tear_down_locked(self) -> None:
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=2.0)
            except Exception as e:
                logger.debug("file_watcher: tear-down: %s", e)
        self._observer = None
        self._handler = None
        self._root = None


_manager = _Manager()


def attach_loop(loop: asyncio.AbstractEventLoop) -> None:
    _manager.attach_loop(loop)


def bind(root: Optional[str]) -> None:
    _manager.bind(root)


def shutdown() -> None:
    _manager.shutdown()
