"""The two log lines a reader of an idle loop needs, each said once per change.

An empty implement pool declined silently, so a loop starved of confirmed
items read in the log exactly like one that was not running (66 min on
2026-11-09 with `_loop_is_free` passing every minute and nothing enqueued). And
the stray-worktree line was said at INFO on every 60 s look.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from scripts.automod import backlog as B, state as S, worktree as W
from workers.sources import DECLINED
from workers.sources import autocode as I


class _Queue:
    def has_live(self, key):
        return False


@pytest.fixture
def quiet_loop(monkeypatch, caplog):
    monkeypatch.setattr(I, "_last_decline", {"why": "", "at": 0.0})
    monkeypatch.setitem(I._boot_settled, "done", True)
    monkeypatch.setattr(I, "_housekeeping_due", lambda q, c: False)
    monkeypatch.setattr(I, "_release_stranded_quietly", lambda: None)
    monkeypatch.setattr(I, "_maybe_flush", lambda *a, **k: None)
    caplog.set_level(logging.DEBUG, logger=I.logger.name)
    return caplog


def _said(caplog, level, text):
    return [r for r in caplog.records if r.levelno == level and text in r.getMessage()]


def test_an_empty_implement_pool_is_a_logged_decline_said_once(quiet_loop, monkeypatch):
    monkeypatch.setattr(I, "_loop_is_free", lambda depth=None: (True, "free"))
    monkeypatch.setattr(B, "select_confirmed", lambda ledger: None)

    for _ in range(3):
        assert asyncio.run(I.enqueue_if_due(_Queue(), {})) == DECLINED

    assert len(_said(quiet_loop, logging.INFO, "nothing ready")) == 1
    assert len(_said(quiet_loop, logging.DEBUG, "nothing ready")) == 2


def test_the_stray_worktree_line_is_said_when_the_set_changes(monkeypatch, caplog):
    monkeypatch.setattr(I, "_last_strays", {"paths": set()})
    monkeypatch.setattr(S, "is_enabled", lambda root: True)
    monkeypatch.setattr(S, "is_halted", lambda: False)
    monkeypatch.setattr(S, "is_broken", lambda: False)
    monkeypatch.setattr(S, "read_current", lambda: None)
    monkeypatch.setattr(S, "read_rollback_request", lambda: None)
    monkeypatch.setattr(S, "landing_lock", lambda: None)
    monkeypatch.setattr(I, "_rounds_about_to_land", lambda owned: [])
    strays = {"paths": ["/tmp/wt1"]}
    monkeypatch.setattr(W, "prune_orphans", lambda root: list(strays["paths"]))
    caplog.set_level(logging.INFO, logger=I.logger.name)

    for _ in range(3):
        assert I._loop_is_free(depth=2) == (True, "free")
    assert len(_said(caplog, logging.INFO, "ignoring 1 worktree")) == 1
    strays["paths"] = ["/tmp/wt1", "/tmp/wt2"]
    I._loop_is_free(depth=2)
    I._loop_is_free(depth=2)
    assert len(_said(caplog, logging.INFO, "ignoring 2 worktree")) == 1
