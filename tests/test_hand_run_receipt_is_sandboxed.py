"""A `hand_run.receipt` is item front matter that the backend executes, so it runs
in the read-only sandbox the bench sessions get, never bare.

Until 2026-11-10 `_receipt_passes` ran `/bin/bash -c <receipt>` in the backend,
on the autocode poll, outside every tool-layer guard. These nodes pin the three
properties that replace that: a write made by a receipt does not reach the
machine, a receipt that cannot be sandboxed fails closed (holds the item) rather
than running bare, and a reading is reused instead of re-run on every poll.
"""

from __future__ import annotations

import shutil

import pytest

from agent_mcp import _tool_sandbox as T
from scripts.automod import owed as O

needs_bwrap = pytest.mark.skipif(
    not shutil.which("bwrap") or not T.sandbox_available()[0],
    reason="no usable bubblewrap on this host")


@pytest.fixture(autouse=True)
def _fresh_cache():
    O.clear_receipt_cache()
    yield
    O.clear_receipt_cache()


@needs_bwrap
def test_exit_codes_still_decide(tmp_path):
    assert O._receipt_passes("exit 0")[0] is True
    ok, reading = O._receipt_passes("echo held; exit 3")
    assert ok is False and reading.startswith("exited 3") and "held" in reading


@needs_bwrap
def test_a_receipt_cannot_write_to_the_machine(tmp_path):
    home_target = O._live_root() / ".receipt-sandbox-probe"
    assert not home_target.exists()
    ok, reading = O._receipt_passes(f"touch {home_target} && exit 0")
    assert ok is False, "a write into the checkout succeeded inside the sandbox"
    assert not home_target.exists()


@needs_bwrap
def test_a_write_under_tmp_lands_in_the_sandbox_not_on_the_host(tmp_path):
    target = tmp_path / "written-by-a-receipt"
    O._receipt_passes(f"touch {target}; exit 0")
    assert not target.exists()


def test_no_sandbox_means_not_run_and_held(monkeypatch, tmp_path):
    marker = tmp_path / "ran"
    monkeypatch.setattr(T, "sandbox_available", lambda: (False, "bwrap is not installed"))
    ok, reading = O._receipt_passes(f"touch {marker}; exit 0")
    assert ok is False and "sandbox is unavailable" in reading
    assert not marker.exists(), "the receipt ran bare when no sandbox could be built"


def test_a_reading_is_reused_within_the_cache_window(monkeypatch):
    monkeypatch.delenv(O.RECEIPT_CACHE_ENV, raising=False)
    calls = []
    monkeypatch.setattr(O, "_run_receipt", lambda r: calls.append(r) or (True, ""))
    for _ in range(5):
        assert O._receipt_passes("exit 0") == (True, "")
    assert calls == ["exit 0"]
    monkeypatch.setenv(O.RECEIPT_CACHE_ENV, "0")
    O._receipt_passes("exit 0")
    assert len(calls) == 2


def test_the_data_root_overlay_is_mounted_before_the_masks_over_it(monkeypatch, tmp_path):
    """bwrap applies mounts in order: an overlay of the data root placed after the
    `/dev/null` masks would mount over them and uncover a socket or a credential
    file that lives under the data root."""
    import app.data_root as DR
    data = tmp_path / "lloyd-data"
    data.mkdir()
    sock, secret = str(data / "agent.sock"), str(data / "token")
    monkeypatch.setattr(T, "sandbox_available", lambda: (True, ""))
    monkeypatch.setattr(T, "_host_socket_paths", lambda *a, **k: [sock])
    monkeypatch.setattr(T, "_credential_paths", lambda *a, **k: [secret])
    monkeypatch.setattr(DR, "resolve_data_root_for_tree", lambda tree=None: data)

    argv, why = O._receipt_argv("exit 0")

    assert argv is not None, why
    overlay = argv.index("--overlay-src")
    assert argv[overlay + 1] == str(data.resolve())
    for masked in (sock, secret):
        mask = argv.index(masked)
        assert argv[mask - 1] == "/dev/null", argv[mask - 2:mask + 1]
        assert overlay < mask, f"the overlay is mounted after the mask over {masked}"
