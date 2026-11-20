"""The Background brief counts what the Background tab withholds (#4267).

`mc_navigate` answers "I moved you to the Background tab" with the dict
`app/routers/mc_ui.py::_summarize_background` builds. The tab itself is served by
`GET /api/background/sessions` → `app/routers/sessions.py::_collect_background_sessions`,
which since #2648 refuses a bench-family row by default (`include_bench`) and says
how many it refused under `bench_hidden`. The brief was a second, private walk over
the same directory and had no bench filter at all, so on the live store on 2026-11-18
the two surfaces answered for ONE directory: `sessions_total: 395` with `'bench': 40`
in `by_source` here, against `count 150, total 9143, bench_hidden 8` there. The model
was handed a number that was neither the panel's row window nor the directory, and
named a class of transcript the panel had been built to hide.

The identity that can hold between a bounded listing and a count of a pass over it is

    brief["sessions_total"] + brief["bench_hidden"] == envelope["total"]

— `total` is what the collector's pass settled (docstring at
`app/routers/sessions.py:454-459`), and it INCLUDES the trials it withheld.
`count` is rows RETURNED and is never the other side of this equation: on any store
bigger than the fixture below the panel's window, not the directory, decides `count`,
which is exactly why the two numbers drifted in production and why an assertion
against `count` would pin the wrong thing.

The fixture is deliberately small (7 background heads, 2 chats, 1 sidecar) so that
neither the brief's `SESSION_SCAN_CEILING` walk nor the collector's
`_BACKGROUND_PLATFORM_SCAN_CEILING` pass truncates — an identity pinned across a
truncation would be a coincidence about two windows that happened to stop in the same
place.
"""
import json
import os

import pytest

from app.harness.microcompact import REDUCTION_SIDECAR_SUFFIX

#: Ordinary unattended runs — four-part ids, machine platforms.
ORDINARY = {
    "20261118_100000_autocode_a1": "autocode",
    "20261118_100100_worker_b2": "worker",
    "20261118_100200_autotriage_c3": "autonomy-task:80",
}
#: The bench family, named exactly as its two writers stamp `source`:
#: `scripts/autoresearch/bench_runner_sdk.py` and `workers/sources/bench_mine.py`.
BENCH = {
    "20261118_100300_bench_d4": "bench",
    "20261118_100400_benchmine_e5": "bench-mine",
}
#: `sessions_io.is_bench_source` compares EXACT source names, so the eval corpus's
#: own `benchmarked` is not a trial and stays visible. One near-miss row in the
#: fixture is what stops "hide the bench family" being implemented as `startswith`.
NEAR_MISS = {"20261118_100500_benchmarked_f6": "benchmarked"}
#: #1064's class: four-part id, human platform, shown by no listing but Background's.
#: It is in the collector's `total` (the mis-labelled exemption) and must be in the
#: brief's count too, or the identity moves by one in the wrong direction.
ORPHAN = {"20261118_100600_autocode_g7": "autocode"}
#: Conversations — in neither surface's count.
CHATS = {"20261118_110000_a1b2c3": "mission-control",
         "20261118_110100_d4e5f6": "browser"}

#: What the panel will draw, and what the collector settles: the same rows, plus the
#: trials it withholds.
VISIBLE = list(ORDINARY) + list(NEAR_MISS) + list(ORPHAN)
SETTLED = len(VISIBLE) + len(BENCH)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """One sessions directory holding all five classes plus one reduction sidecar.

    Each site is patched where IT binds the directory: `mc_ui` holds its own
    module-level `SESSIONS_DIR`, and the collector takes the directory as an
    argument, which is why the fixture returns the path. mtimes are distinct and in
    one fixed order, so no ordering difference can mask or fake a count.
    """
    d = tmp_path / "sessions"
    d.mkdir()

    def write(sid, source=None, platform="worker"):
        body = {"session_id": sid, "title": sid, "preview": "p",
                "platform": platform, "model": "test-model", "messages": []}
        if source:
            body["source"] = source
        (d / f"{sid}.json").write_text(json.dumps(body), encoding="utf-8")

    for group, platform in ((ORDINARY, "worker"), (BENCH, "worker"),
                            (NEAR_MISS, "worker"), (ORPHAN, "mission-control")):
        for sid, source in group.items():
            write(sid, source=source, platform=platform)
    for sid, platform in CHATS.items():
        write(sid, platform=platform)
    # #3815: a reduction sidecar sits in the same `*.json` glob and is not a
    # session. It is the NEWEST file in the store by name, so a walk that counts
    # files instead of heads has to be wrong about `sessions_total` to get here.
    sidecar = "20261118_110200_a1b2c3" + REDUCTION_SIDECAR_SUFFIX
    (d / sidecar).write_text(json.dumps(
        {"schema": "microcompact-reduced.v1", "cleared_call_ids": ["toolu_01A"]}),
        encoding="utf-8")

    for i, path in enumerate(sorted(d.iterdir(), reverse=True)):
        stamp = 1_763_000_000 + i * 60
        os.utime(path, (stamp, stamp))

    from app.routers import mc_ui
    monkeypatch.setattr(mc_ui, "SESSIONS_DIR", d)
    return d


def _brief():
    from app.routers import mc_ui
    return mc_ui._summarize_background()


def _envelope(d):
    from app.routers.sessions import _collect_background_sessions
    return _collect_background_sessions(d, 150)


def test_the_brief_withholds_bench_rows_the_way_the_tab_does(store):
    """Clause 1: no bench-family row reaches `by_source` or `sessions_total`, and the
    withheld count travels beside them under `bench_hidden`.

    Before #4267 this fixture came back `sessions_total: 7` with `'bench': 1,
    'bench-mine': 1` in `by_source` — the panel's default request shows neither row,
    and `mc_navigate` was telling the model it had moved the user to a tab holding
    them.
    """
    out = _brief()
    assert out["bench_hidden"] == len(BENCH), out
    assert out["sessions_total"] == len(VISIBLE), out
    assert set(out["by_source"]) == set(ORDINARY.values()) | {
        "benchmarked", *ORPHAN.values()}, out["by_source"]
    for trial in BENCH.values():
        assert trial not in out["by_source"], (
            f"{trial} is counted in a brief about a tab that hides it")
    assert out["by_source"]["benchmarked"] == 1, (
        "the near-miss source was withheld with the family — is_bench_source "
        "compares exact names, so `benchmarked` is a visible run")


def test_no_bench_row_is_in_the_brief_s_recent_rows(store):
    """The rows the brief lists are rows the panel draws, not only its counts."""
    out = _brief()
    ids = {r["id"] for r in out["recent"]}
    leaked = ids & set(BENCH)
    assert not leaked, f"withheld trials listed anyway: {sorted(leaked)}"
    assert ids == set(VISIBLE), sorted(ids)


def test_brief_total_plus_bench_hidden_equals_the_envelope_total(store):
    """Clause 2, the identity: `sessions_total + bench_hidden == total`.

    `total` (`app/routers/sessions.py:472`) is what the collector's pass settled and
    it includes the trials, so here 5 + 2 == 7. `count` is rows returned — 5 in this
    fixture, where the whole store fits inside the 150-row window — and it is NOT the
    other side of the equation: on the live store the same pair read 150 against a
    brief total of 395 (measured 2026-11-18), because the window, not the directory,
    decides `count`.
    """
    brief, env = _brief(), _envelope(store)
    assert brief["sessions_total"] + brief["bench_hidden"] == env["total"], (
        f"the brief said {brief['sessions_total']} + {brief['bench_hidden']} withheld "
        f"= {brief['sessions_total'] + brief['bench_hidden']}; the tab settled "
        f"{env['total']} for the same directory")
    # Both withheld counts agree HERE, because this store is small enough that
    # neither window bites. They are not the same measurement in general: the
    # collector withholds a trial only inside the ordinary four-part machine-platform
    # set it settled, so a bench row that is also #1064-mislabelled is in the brief's
    # `bench_hidden` and not the envelope's — and on a store the size of the live one
    # the two windows differ anyway (measured 2026-11-20: brief 150 withheld inside its
    # 400-read budget, envelope 71 inside its 150-row page of a 9,469-row directory).
    assert env["bench_hidden"] == brief["bench_hidden"], (
        "on a fixture neither walk truncates, the two surfaces must withhold the same "
        "rows")
    assert env["total"] != env["count"], (
        f"a fixture where `total` equals `count` cannot tell withholding from a short "
        f"window: total={env['total']} count={env['count']}")


def test_the_sidecar_is_in_neither_count(store):
    """#3815 across both surfaces at once: the sidecar belongs to no count, and
    excluding it does not move the identity either.
    """
    brief, env = _brief(), _envelope(store)
    assert brief["by_source"].get("unknown", 0) == 0, brief["by_source"]
    assert not [r for r in brief["recent"]
                if REDUCTION_SIDECAR_SUFFIX in r["id"]], brief["recent"]
    assert brief["sessions_total"] + brief["bench_hidden"] == env["total"]


def test_a_store_of_only_trials_counts_zero_and_reports_them_all(store):
    """The case the identity exists for, and the proof it can fail.

    All 7 background heads re-stamped `source: bench` — one detached grid, the shape
    #2648 was written for (273 transcripts on 2026-10-07, more than the tab's whole
    150-row window). The brief must report an empty tab with the trials named beside
    it. The old private walk reported `sessions_total: 7` here, which is the number
    the model would have contradicted the screen about.
    """
    from app.routers import mc_ui
    for sid in list(ORDINARY) + list(NEAR_MISS) + list(ORPHAN):
        path = mc_ui.SESSIONS_DIR / f"{sid}.json"
        body = json.loads(path.read_text(encoding="utf-8"))
        body["source"] = "bench"
        path.write_text(json.dumps(body), encoding="utf-8")
    brief, env = _brief(), _envelope(store)
    assert brief["sessions_total"] == 0, brief
    assert brief["by_source"] == {}, brief
    assert brief["bench_hidden"] == SETTLED, brief
    assert brief["sessions_total"] + brief["bench_hidden"] == env["total"]
