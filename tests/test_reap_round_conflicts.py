"""#3750's read-only reap-conflict probe: `scripts/maintenance/reap_round_conflicts.py`.

The probe re-derives #3600's recount, so what is worth pinning is not "it prints
something" but that its numbers survive the two ways a ledger recount goes wrong:
reading a population the figure does not belong to, and re-inventing the
promotion-liveness rule instead of borrowing the engine's. Every node here plants a
tiny ledger in `tmp_path` and reads the answer back, so nothing depends on how many rows
the live file happens to hold today — the live figures live in the probe's docstring as
dated readings, deliberately not asserted here.

The liveness rule under test is the engine's own `_unsettled_promotions` in
`workers/sources/board_steward.py`: a promotion stays live until a `settled` row naming its
COMMIT appears, and a `settled` row carries
no `round_id` at all, so a probe that scoped the settle test by round would see every
promotion as unsettled and report conflicts that never happened.
"""
from __future__ import annotations

import ast
import gzip
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "scripts" / "maintenance" / "reap_round_conflicts.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("reap_round_conflicts", PROBE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = _load_probe()


def _ts(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def _row(event: str, when: str, **fields) -> dict:
    """A ledger row shaped like the real ones: `ts` and `created_at` always, plus
    whatever that event carries. `settled` gets no `round_id` because it has none."""
    return {"ts": _ts(when), "created_at": when, "event": event, **fields}


def _plant(base: Path, *, live: list[dict],
           archive: list[dict] | None = None) -> Path:
    """A state dir holding `promotions.jsonl` and, when asked, one rotation beside it —
    the layout `state.ledger_archives` globs for."""
    base.mkdir(parents=True, exist_ok=True)
    ledger = base / "promotions.jsonl"
    ledger.write_text("".join(json.dumps(r) + "\n" for r in live), encoding="utf-8")
    if archive:
        with gzip.open(base / "promotions-archive-202609.jsonl.gz", "wt",
                       encoding="utf-8") as fh:
            for r in archive:
                fh.write(json.dumps(r) + "\n")
    return ledger


def _docstring_prose() -> str:
    """The probe's module docstring, whitespace-normalised and lower-cased. Prose
    wraps: a node that pinned a phrase to one physical line would fail on a rewrap that
    changed nothing, which is how a doc test becomes a lint rule nobody keeps."""
    doc = ast.get_docstring(ast.parse(PROBE.read_text(encoding="utf-8"))) or ""
    return " ".join(doc.split()).lower()


def _parser_flags() -> set[str]:
    """The long flags the probe's OWN parser registers, read off the syntax tree from
    `add_argument` calls — not from a text search. Text is the wrong instrument here:
    this file and the probe both QUOTE `--record` and `--schedule` to explain why they
    are absent, and `git log --format=...` appears in a comment, so a grep over the
    source finds three flags the probe does not have and misses nothing that it does."""
    tree = ast.parse(PROBE.read_text(encoding="utf-8"))
    flags = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                        and arg.value.startswith("--"):
                    flags.add(arg.value.split("=")[0])
    return flags


#: One item, two rounds: `SM_B` promoted at 09:00, `SM_A` abandoned at 10:00 with no
#: settle in between — the collision #3600 stopped, in four rows. Dated before the fix
#: instant on purpose, so a plain `main()` over this ledger exits 0.
_ABANDON = _row("round_abandoned", "2026-10-01T10:00:00Z", round_id="SM_A", item_id=42,
                reason="implement turn ended (stop) with the round still open")
_PROMOTE = _row("promoted", "2026-10-01T09:00:00Z", round_id="SM_B", commit="c0ffee1234")
_ATTEMPT_A = _row("backlog_implement", "2026-10-01T08:00:00Z", phase="finished",
                  item_id=42, round_id="SM_A")
_ATTEMPT_B = _row("backlog_implement", "2026-10-01T07:00:00Z", phase="finished",
                  item_id=42, round_id="SM_B")
_COLLISION = [_ABANDON, _PROMOTE, _ATTEMPT_A, _ATTEMPT_B]


# ── clause 1: what it prints ──────────────────────────────────────────────────────

def test_the_report_names_the_population_and_each_conflict_row(tmp_path):
    """A conflict is reported with the fields #3600's recount quotes, not just a count
    — a bare `conflicts=1` cannot be cross-checked against the ledger by anyone who did
    not run the probe themselves."""
    ledger = _plant(tmp_path, live=_COLLISION)

    result = probe.recount(ledger, merged=True)
    body = "\n".join(probe.format_report(result))

    assert result["population"] == "merged"
    assert result["abandons"] == 1 and len(result["conflicts"]) == 1
    assert result["conflicts"][0] == {
        "created_at": "2026-10-01T10:00:00Z", "ts": _ABANDON["ts"], "item_id": 42,
        "reaped_round": "SM_A", "promoted_round": "SM_B",
        "promoted_created_at": "2026-10-01T09:00:00Z", "commit": "c0ffee1234"}
    for token in ("population=merged", "abandons=1", "conflicts=1",
                  "created_at=2026-10-01T10:00:00Z", "item=42", "reaped=SM_A",
                  "promoted=SM_B", "commit=c0ffee12"):
        assert token in body, f"the report never prints `{token}`:\n{body}"


def test_a_reaped_promotion_and_a_settled_one_are_not_conflicts(tmp_path):
    """Two non-events, because a probe that only ever fires is as useless as one that
    never does: the same commit reverted, and the same commit settled, both end the
    promotion's claim on the item."""
    reverted = _plant(tmp_path / "rv", live=[
        *_COLLISION,
        _row("rollback_succeeded", "2026-10-01T09:30:00Z", commit="c0ffee1234"),
    ])
    settled = _plant(tmp_path / "st", live=[
        *_COLLISION,
        _row("settled", "2026-10-01T09:30:00Z", commit="c0ffee1234"),
    ])

    assert probe.recount(reverted, merged=True)["conflicts"] == []
    assert probe.recount(settled, merged=True)["conflicts"] == []


def test_the_abandon_carries_the_round_that_was_reaped_not_the_one_still_live(tmp_path):
    """The two round ids are different and must not be interchangeable: #3600's defect
    was reaping the round whose promotion was still under observation, so the report has
    to say which one it reaped and which one stayed live."""
    ledger = _plant(tmp_path, live=_COLLISION)

    conflict = probe.recount(ledger, merged=True)["conflicts"][0]

    assert conflict["reaped_round"] == "SM_A" and conflict["promoted_round"] == "SM_B"


# ── clause 2: the population switch ───────────────────────────────────────────────

def test_merged_is_the_default_and_live_only_excludes_the_rotations(tmp_path):
    """The figure #3600 quotes for item 2694 lives in a rotation today: its `promoted`
    row is archive-only while its abandon is still live. A probe that defaulted to the
    live file alone would report no conflict for a collision that happened, and a probe
    that could not show both populations would make every older reading look wrong."""
    ledger = _plant(tmp_path, live=[_ABANDON, _ATTEMPT_A, _ATTEMPT_B],
                    archive=[_PROMOTE])

    merged = probe.recount(ledger, merged=True)
    live = probe.recount(ledger, merged=False)

    assert [c["promoted_round"] for c in merged["conflicts"]] == ["SM_B"]
    assert live["conflicts"] == [], "the live population saw a rotated row"
    assert (merged["population"], live["population"]) == ("merged", "live")


def test_every_printed_number_rides_its_population_name(tmp_path):
    """`abandons=374` beside someone else's `317` reads as ledger loss unless the line
    says which store it came from, so the label is on every line rather than once in a
    header — 317 live against 374 merged on 2026-11-04 is a 57-row gap."""
    ledger = _plant(tmp_path, live=_COLLISION)

    lines = probe.format_report(probe.recount(ledger, merged=True))

    assert lines, "the probe printed nothing"
    assert all(ln.startswith("population=merged ") for ln in lines), lines
    counted = [ln for ln in lines if any(
        f" {k}=" in ln for k in ("abandons", "conflicts", "conflicts_since_fix"))]
    assert len(counted) >= 2, f"expected the counts and the since-fix figure: {lines}"


def test_the_parser_takes_live_only_and_no_way_to_schedule_or_record():
    """`--record` is the trap this item names by hand: #2759 shipped
    `gate_to_land_lag.py` with a `--record` and no caller, and #2858 had to file the
    caller afterwards. Owed-check ruled on #3600's owed entry 2 at 2026-11-04T02:14:31
    that the recount stays an on-demand probe, so the absence of the flag is the proof —
    a repo grep proves less, because a probe gets scheduled from
    `~/obsidian/autonomy/*.md` plus a skill, not only from the repo."""
    for flag in ("--record", "--schedule", "--nightly", "--write"):
        result = subprocess.run([sys.executable, str(PROBE), flag],
                                capture_output=True, text=True, cwd=REPO_ROOT)
        assert result.returncode != 0, f"`{flag}` was accepted: {result.stdout}"
        assert "writes nothing" in result.stderr, f"`{flag}` refused without a reason"
    assert _parser_flags() == {"--live-only", "--ledger"}, (
        f"the probe registers {sorted(_parser_flags())}, not the population switch and the "
        "test hook alone — an extra flag is how an on-demand probe becomes a scheduled "
        "row six weeks from now, which is exactly what owed-check ruled against")


def test_the_probe_has_no_write_path_of_any_kind():
    """The contract is `read-only, no write path`, checked on the syntax tree rather
    than on the docstring — a comment claiming read-only is what every script that later
    grew a journal used to say."""
    tree = ast.parse(PROBE.read_text(encoding="utf-8"))
    writers = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name in {"write_text", "write_bytes", "mkdir", "touch", "remove", "unlink",
                    "rmtree", "open"}:
            writers.append((node.lineno, name))
    assert not writers, f"the probe writes somewhere: {writers}"


# ── clause 3: the predicate is borrowed, not restated ─────────────────────────────

def test_one_planted_abandon_over_one_unsettled_promotion_is_one_conflict(tmp_path):
    """The question the probe exists to answer: an abandon plus a promotion of the same
    item that no settle reached, at T = the abandon's own instant. One row in, one
    conflict out, and the report names the planted rows."""
    ledger = _plant(tmp_path, live=_COLLISION)

    result = probe.recount(ledger, merged=True)
    body = "\n".join(probe.format_report(result))

    assert len(result["conflicts"]) == 1, result["conflicts"]
    assert "population=merged abandons=1 conflicts=1 items=42" in body


def test_a_promotion_at_the_very_instant_of_the_abandon_is_still_as_of_t(tmp_path):
    """`ts <= T`, at the boundary, because clause 3 says so and the two readings differ
    only here.

    A mutation of the as-of-T test from `> t` to `>= t` — which drops a promotion that
    landed at the same instant the reaper wrote its row — left all seventeen other nodes
    green: the fixtures space the promotion an hour before the abandon, so the difference
    between "before T" and "before or at T" was never reachable. Same ledger, same
    rows, one timestamp.
    """
    at = "2026-10-01T10:00:00Z"
    ledger = _plant(tmp_path, live=[
        _ATTEMPT_A, _row("round_abandoned", at, round_id="SM_A", item_id=42),
        _ATTEMPT_B, _row("promoted", at, round_id="SM_B", commit="c0ffee1234"),
    ])

    result = probe.recount(ledger, merged=True)

    assert len(result["conflicts"]) == 1, (
        f"a promotion at exactly the abandon's own instant is not as-of-T: "
        f"{len(result['conflicts'])} counted — the comparison has to be `ts <= T`, and "
        "the strict form is only reachable at this instant, which is why it needs a node")


def test_the_round_the_reaper_reaped_is_never_its_own_conflict(tmp_path):
    """The reaped round's own promotion is #3600's real shape, not a fiction: a round can
    be promoted AND reaped — SM_20261021_172454 was promoted at 17:50:10Z and abandoned at
    18:03:41Z, and item 3112 is a conflict only because ANOTHER round's promotion is
    unsettled. Pairing an item with its own reaped round would report the guard's own
    round as the thing it collided with.

    The companion guard on this line is `not other` — a `started` row carries no
    `round_id`, so it names a round called ``. Half of that cannot be reached from any
    ledger, because the map it looks up is built under `if d.get("round_id")` and so can
    never hold an empty key; only `other == reaped` changes behaviour, and only that half
    is tested here.
    """
    ledger = _plant(tmp_path, live=[
        _ATTEMPT_A, _ATTEMPT_B, _PROMOTE,
        _row("promoted", "2026-10-01T08:30:00Z", round_id="SM_A", commit="beef0000aa"),
        _ABANDON,
    ])

    result = probe.recount(ledger, merged=True)

    assert len(result["conflicts"]) == 1, (
        f"the reaped round's own promotion was counted too: {result['conflicts']}")
    assert result["conflicts"][0]["promoted_round"] == "SM_B", (
        f"the conflict names the wrong round: {result['conflicts'][0]['promoted_round']}")
    assert all(r["reaped_round"] != r["promoted_round"] for r in result["conflicts"]), (
        f"a round was reported as colliding with itself: {result['conflicts']}")


def test_the_flag_selects_the_population_and_the_default_does_not(tmp_path):
    """The wiring between argv and the population is its own process boundary, and a
    node that calls `recount(..., merged=False)` directly walks straight past it.

    Mutating `merged=not args.live_only` to `merged=True` left the other fourteen nodes
    green and printed `population=merged` for a `--live-only` run: every number was still
    labelled with a population, which is all clause 2's wording invites a test to check.
    A label is not a selection. One ledger, whose planted conflict lives in the ARCHIVE,
    is run through the CLI twice so the two populations disagree by exactly that row.
    """
    ledger = _plant(tmp_path,
                      live=[_ATTEMPT_A, _ABANDON,
                            _row("settled", "2026-10-01T09:30:00Z",
                                 commit="fffffffffff")],
                      archive=[_ATTEMPT_B, _PROMOTE])
    default = subprocess.run([sys.executable, str(PROBE), "--ledger", str(ledger)],
                             capture_output=True, text=True, cwd=REPO_ROOT)
    live = subprocess.run([sys.executable, str(PROBE), "--ledger", str(ledger),
                           "--live-only"], capture_output=True, text=True, cwd=REPO_ROOT)

    assert "population=merged" in default.stdout, (
        f"the default population is not merged: {default.stdout!r}")
    assert "population=merged abandons=1 conflicts=1 items=42" in default.stdout, (
        f"the archived row was not counted by the default population: {default.stdout!r}")
    assert "population=live abandons=1 conflicts=0 items=-" in live.stdout, (
        f"`--live-only` still counted the archived row, or did not report its own "
        f"abandon count under its own population name: {live.stdout!r}")


def test_a_settled_row_suppresses_the_conflict_matched_on_commit_not_round(tmp_path):
    """The trap #3750's triage named, made executable: `settled` rows carry no
    `round_id`, so a promotion's commit is the only key that closes it. A probe scoping
    the settle test by round passes the node above and then reports a conflict where the
    ledger shows none — so the settle row here is written with the round id absent."""
    unsettled = _plant(tmp_path / "u", live=_COLLISION)
    settled = _plant(tmp_path / "s", live=[
        *_COLLISION,
        _row("settled", "2026-10-01T09:30:00Z", commit="c0ffee1234"),
    ])
    assert "round_id" not in settled_rows(settled)

    assert len(probe.recount(unsettled, merged=True)["conflicts"]) == 1
    assert probe.recount(settled, merged=True)["conflicts"] == []


def settled_rows(ledger: Path) -> list[str]:
    """The keys a planted `settled` row actually carries — read here, not assumed, so
    the node above cannot pass by asserting against a row it wrote wrongly."""
    return sorted(json.loads(line).keys() for line in
                  ledger.read_text(encoding="utf-8").splitlines()
                  if json.loads(line)["event"] == "settled")[0]


def test_a_settle_after_the_reap_instant_does_not_clear_the_conflict(tmp_path):
    """The settle is scoped to `ts <= T` as well as to the commit: the settle that
    followed #3600's reaper landed ten minutes after the reaper itself, and a probe that
    counted settles at any later time would erase history by waiting long enough."""
    ledger = _plant(tmp_path, live=[
        *_COLLISION,
        _row("settled", "2026-10-01T11:00:00Z", commit="c0ffee1234"),
    ])

    assert len(probe.recount(ledger, merged=True)["conflicts"]) == 1


def test_the_probe_borrows_the_engine_s_history_and_liveness_helpers():
    """`no liveness arithmetic re-invented`: the probe never parses a ledger line
    itself, and its three questions go to the functions that already answer them for the
    engine. Reading JSON directly would let it keep working after the engine's rule
    moved, which is the failure this recount exists to avoid."""
    source = PROBE.read_text(encoding="utf-8")

    for needed in ("B.implement_history", "B._reverted_commits", "B._ledger_events"):
        assert needed in source, f"the probe stopped using {needed}"
    for invented in ("json.loads", "read_text(", "splitlines(", "gzip.open"):
        assert invented not in source, (
            f"the probe reads the ledger itself via `{invented}` instead of through the "
            "engine's helpers, so the engine's rule could move underneath it")


# ── clause 4: conflicts_since_fix drives the exit code ────────────────────────────

def test_the_fix_instant_and_its_commit_are_one_constant_not_a_pair_to_drift():
    """`2026-10-30T18:36:03Z` is #3600's COMMITTER stamp — `git log -1 --format='%cI'
    68425af5` answers 2026-10-30T11:36:03-07:00. Its landed note records settle at
    18:46:21Z, ten minutes later; a probe pinned to the settle would silently widen the
    window by that much and let a conflict inside the gap read as historical."""
    assert probe.FIX_INSTANT == "2026-10-30T18:36:03Z"
    assert probe.FIX_COMMIT == "68425af5"
    assert probe.FIX_TS == _ts("2026-10-30T18:36:03Z")
    source = PROBE.read_text(encoding="utf-8")
    assert probe.FIX_COMMIT in source and probe.FIX_INSTANT in source, (
        "the constant and the commit it came from have to be readable side by side")


def test_the_historical_shape_exits_zero_and_a_post_fix_conflict_exits_one(tmp_path,
                                                                          capsys):
    """Exit is `conflicts_since_fix == 0`, not `conflicts == 0`. The four pre-fix
    conflicts are the reason the guard exists and are not this probe's alarm; a conflict
    dated after the instant is the alarm, and it has to reach the exit status."""
    historical = _plant(tmp_path / "h", live=_COLLISION)
    after = _plant(tmp_path / "a", live=_COLLISION + [
        _row("round_abandoned", "2026-11-01T10:00:00Z", round_id="SM_C", item_id=42,
             reason="implement turn ended (stop) with the round still open"),
    ])

    assert probe.main(["--ledger", str(historical)]) == 0
    assert "conflicts_since_fix=0" in capsys.readouterr().out
    assert probe.main(["--ledger", str(after)]) == 1
    out = capsys.readouterr().out
    assert "conflicts_since_fix=1" in out and "population=merged" in out


# ── clause 5: the baseline in the docstring ───────────────────────────────────────

def test_the_docstring_states_the_merged_baseline_as_a_dated_merged_figure():
    """The figure and its population travel together: `conflicts=4` is merged over
    2026-11-04, on items 1685, 2694, 3112 and 3524. A reader who found 4 without the
    word "merged" would go looking for a regression in the live file, which reads 2."""
    doc = _docstring_prose()

    assert "conflicts=4" in doc and "merged" in doc
    for item in ("1685", "2694", "3112", "3524"):
        assert item in doc, f"the baseline lost item {item}"
    assert "2026-11-04" in doc, "the baseline has to carry the day it was measured"


def test_the_docstring_says_the_live_figure_decays_rather_than_calling_it_a_baseline():
    """The live reading is the perishable one, and this is the item's own failure mode
    turned around: rotation is per-row and keyed on the row's own month, so
    `SM_20261021_172454`'s promotion going into an archive drops the live count to 1
    with nothing broken. A docstring that called 2 a baseline would read as a regression
    a month from now."""
    doc = _docstring_prose()

    assert "conflicts=2" in doc
    for token in ("dated reading", "rotat", "archive"):
        assert token in doc, f"the docstring never says `{token}` about the live figure"
    assert "not a baseline" in doc, (
        "the live figure has to be refused the word baseline in the same breath it is "
        "quoted, or a rotation reads as a regression")

def test_nothing_in_the_engine_or_the_scheduler_names_the_probe():
    """Clause 6's repo half. The parser node above is the real proof, and this is the
    cheap backstop: a wiring commit that aimed this probe at a job would have to name it,
    and it would fail here rather than ship a schedule nobody ruled for."""
    hit = subprocess.run(["git", "-C", str(REPO_ROOT), "grep", "-l",
                          "reap_round_conflicts", "--", "workers/", "scripts/automod/"],
                         capture_output=True, text=True)
    assert hit.returncode == 1, (
        f"the probe is named in engine or scheduler code:\n{hit.stdout.strip()}")
