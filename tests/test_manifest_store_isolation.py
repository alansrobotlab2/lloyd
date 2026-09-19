"""A pytest run never writes the operator's live request-manifest store (#581).

Different in kind from the three manifest test files, which each own a store:
this file deliberately sets **no** `LLOYD_MANIFEST_STORE` of its own, so what it
measures is where a request goes when nobody redirected it. That is the only way
to test the `tests/conftest.py` guard added in this diff, and the guard is what
stops the suite from filling the real store.

The measured fact that made this a test and not a comment: the first full-suite
run with `app/component_manifest.py` in its tree wrote 48 lines into
`~/.local/state/lloyd-request-manifests/manifests/2026-09-19.ndjson` — 38 from
`app/harness/finalizer.py::run_finalizer`, 6 from `app/secondary_models.py`, 4
from `app/harness/client.py::stream_chat` — every one of them a fixture string
that no model was ever asked. #581's closing clause is read off that directory
("24 h of mixed traffic read off the live store: zero manifest-write errors"), so
an unguarded suite does not merely clutter the store; it writes the evidence a
person is going to read, and writes it wrong.
"""

from __future__ import annotations

import os
from pathlib import Path

from app import component_manifest as cm

#: The variable conftest is supposed to have redirected, and the two it already
#: redirected before this diff existed. The second pair is the positive control:
#: if the conftest pass did not run at all, the first assertion below would pass
#: for an unset variable instead of a set one.
REDIRECTED = "LLOYD_MANIFEST_STORE"
CONTROLS = ("LLOYD_AUTOMOD_STATE", "LLOYD_GUARDIAN_STATE")


def _production_state_home() -> Path:
    return Path.home() / ".local" / "state"


def _outside_production_state(value: str) -> bool:
    """False iff `value` sits under the machine's real state dir.

    Not "is under /tmp": a caller that chose its own state dir keeps it, and
    conftest says so — the gate's `_child_env` points `LLOYD_AUTOMOD_STATE` at
    `<round>/gate-state/automod`, which is neither the production path nor the
    temp scratch, and is correct. The claim under test is only that a test run
    never lands in the operator's own state directory, so that is what is
    asserted. (The first version of this test asserted `_under_temp` for the two
    controls and failed exactly that gate run for exactly that reason.)
    """
    real = Path(value).resolve()
    home = _production_state_home().resolve()
    return home not in real.parents and real != home


def test_conftest_redirects_the_manifest_store_default() -> None:
    """The default store a test resolves is not the machine's live one.

    Nothing here calls the module; the claim is entirely about the environment
    conftest built, which is exactly what the leak was. The controls prove the
    redirect pass ran, so `REDIRECTED` being set is a consequence of that pass
    and not of the machine's shell.
    """
    for control in CONTROLS:
        assert os.environ.get(control), (
            f"{control} is unset, so conftest's redirect pass did not run and "
            f"the assertion about {REDIRECTED} below would prove nothing")
        assert _outside_production_state(os.environ[control]), os.environ[control]

    got = os.environ.get(REDIRECTED)
    assert got, (
        "conftest left LLOYD_MANIFEST_STORE unset, so any test that sends a "
        "request without its own store writes the operator's live manifest "
        "directory — the 48-line pollution this guard was added for")
    assert _outside_production_state(got), (
        f"{got} is inside the machine's state dir, so an unredirected test write "
        f"would land under {_production_state_home()} — the store the human "
        "reads for the 24 h clause")
    assert cm.store_root() == Path(got), (
        "the module and the environment disagree about where the store is")


def test_a_request_with_no_store_of_its_own_lands_in_the_scratch_dir() -> None:
    """The redirect is not just set, it is the store the writer actually uses.

    An env var that nothing reads would satisfy the test above and still leak, so
    this drives a real `record_request` — with no `monkeypatch.setenv` anywhere in
    this file — and reads the line back out of the redirected root by its
    `request_id`. Clause 5 (no component text in any written line) is pinned in
    `tests/test_component_manifest.py::test_no_written_line_carries_component_text`;
    this test is only about *where* the line went.

    The store under test is conftest's redirected scratch dir and nothing else:
    deliberately no `tmp_path` and no `setenv` here, because passing a directory
    this test never hands to the writer would read as though it were the one being
    written to. The scratch root is re-derived below from the same env conftest
    set, and the assertion is against that.
    """
    before = _count_live_default_this_test()
    line = cm.record_request(
        base_url="http://stub:8096/v1", model="stub-model",
        payload={"model": "stub-model",
                 "messages": [{"role": "user", "content": "isolation probe"}]},
        session_id="store-isolation", send_site=SEND_SITE)
    assert line and line.get("request_id"), (
        "record_request returned no line, so nothing is being proved "
        f"(stats: {cm.stats()})")
    request_id = line["request_id"]
    cm.flush(timeout=8.0)

    rows = [row for row in _lines_under(cm.store_root())
            if row.get("request_id") == request_id]
    assert len(rows) == 1, (
        f"the redirected store {cm.store_root()} did not receive the line")
    assert rows[0]["send_site"] == SEND_SITE
    assert _count_live_default_this_test() == before, (
        "this test's own line appeared in the operator's live manifest store: the "
        "conftest redirect is set but not in effect. Counted by `send_site`, not "
        "by file size, so a real request from the running backend cannot make "
        "this fail for the wrong reason.")


#: Tagged with the test file rather than a module function, so the leak check
#: below can select this test's lines out of a store that legitimately holds
#: everyone else's once the module is deployed.
SEND_SITE = "tests/test_manifest_store_isolation.py"


def _live_default() -> Path:
    """The production default: what `store_root()` answers with no redirect.

    Same expression `component_manifest.store_root()` uses, so this is the path
    the guard exists to keep writes out of — not a guess at it.
    """
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / cm.DEFAULT_SUBDIR


def _live_default_lines():
    import json

    root = _live_default() / "manifests"
    if not root.is_dir():
        return
    for path in sorted(root.glob("*.ndjson")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    continue


def _count_live_default_this_test() -> int:
    return sum(1 for line in _live_default_lines()
               if line.get("send_site") == SEND_SITE)



def _lines_under(root: Path) -> list[dict]:
    import json

    out: list[dict] = []
    manifests = root / "manifests"
    if manifests.is_dir():
        for path in sorted(manifests.glob("*.ndjson")):
            out += [json.loads(r) for r in
                    path.read_text(encoding="utf-8").splitlines() if r.strip()]
    return out
