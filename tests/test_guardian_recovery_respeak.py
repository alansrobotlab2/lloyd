"""#2264: the recovery announcement names the alerts the outage swallowed, and says one again.

`agent-tts` answers `/health` about 96 s after a supervisor restart, so the alert that
announces a landing routinely dies inside the gap the landing itself opened: on
2026-10-05 the `Landed:` utterance was refused `[Errno 111]` at 18:13:26, and at
18:14:20 the guardian alerted `Recovered: agent-tts` with the body `agent-tts was
agent-tts: STOPPED without an intentional stop.\\nRestarted; started;
http://127.0.0.1:8090/health answering.` — not one word about the two utterances
`voice-loss.md` was naming at that same instant. Nothing said them again, and the freed
suppression slot (#2256 clause 4) sat unused all evening: the next two keys in
`voice_spoken.json` are new alerts at 18:41:55 and 18:58:08.

So this is the recovery path reading a record that already exists. `speak.read_loss_record`
parses the count and the lost texts and was written for exactly that read; before this
file, `guardian.py` had zero hits for it and `git log -S'read_loss_record'` named one
commit ever (#1904's backlog escalator).

**What these nodes do NOT use `voice.log` for.** Every `_log` call in `speak_now` sits on
a failure branch — quiet hours, a refused synthesis, no audio back, a playback error — so a
successful re-speak adds no line, and `grep -c "<the lost text>" voice.log` stays at the
one `synth failed` line after a correct fix. No line numbers are quoted for that here,
deliberately: the review rung caught the first version of this file citing the positions
from the pre-change tree, ~121 lines stale by the time this diff landed, and which branch
writes is a claim about `speak_now`'s shape, not about an address in it. The witnesses that
move are `voice_spoken.json` (the re-speak stamps its own text at recovery time) and the
recovery alert's body.

**Where the timing report comes from.** The premise is a 96 s accident, so its numbers are
load-bearing, and the review of round SM_20261006_023308 graded the clause that owns them
`partial`: the only committed supervisord bytes were #2256's witness, which stops at the
SIGTERM row and cannot yield a spawn time. The two files that fix that are committed in the
vault (`backlog/data/2026-10-05.2264-supervisord-restart-window.log` and
`…-alert-md-recovery-witness.md`) and re-derived by
`test_the_committed_restart_window_witness_still_sums_to_the_reported_gap` — no figure in
this file is quoted from prose.

**Where the restart itself lives.** These nodes stub `Guardian.recover_service` to return
success, because what is under test here is what the recovery does with the loss record,
not the supervisor conversation — the restart, the endpoint confirmation and the flap
bound are pinned in `tests/test_guardian_rollback.py::
test_a_down_synthesiser_is_restarted_and_confirmed_by_its_endpoint` and its neighbours.
What is NOT stubbed is the voice route's own process boundary: `subprocess.Popen` is
captured, so the re-speak is asserted as the argv it hands the detached worker, which is
the same seam `tests/test_guardian_speak.py::
test_dispatch_spawns_detached_and_returns_immediately` pins for a first-time alert.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import pytest

GUARDIAN_DIR = Path(__file__).resolve().parent.parent / "agent-services" / "guardian"
sys.path.insert(0, str(GUARDIAN_DIR))

import speak  # noqa: E402


#: Quiet hours are wall-clock, so every dispatch-taking node here pins them open the way
#: `tests/test_guardian_speak.py::_AWAKE` does. Without it a 02:00 suite run withholds the
#: speech and the re-speak nodes fail on the hour rather than on the code.
_AWAKE = lambda *a, **k: False  # noqa: E731

#: The utterance the outage ate, in the shape `_record_loss` stores it: already composed
#: by `utterance_for`, so this is exactly what must reach the worker's `--text`.
LOST = "Warning. Landed: #2258 round 2: satisfy clause 6 and gate."
LOST_2 = "Info. Alerts: agent-supervisord unreachable."

#: What `recover_service` returns as its detail: no trailing period, because the alert
#: template inside `_recover_infra` adds the one (`Restarted; {detail}.`), so the
#: strings below are the values it is handed, not the sentence it writes.
DETAIL = "started; http://127.0.0.1:8090/health answering"

#: The liveness reason `_recover_infra` is handed. No trailing period either, for the
#: same reason as `DETAIL`: the template adds it, and the live ALERT.md row of
#: 2026-10-05T18:14:20-07:00 ends each sentence with exactly one.
REASON = "agent-tts: STOPPED without an intentional stop"

#: Today's body, byte for byte, as the `self.alert(...)` call inside it writes it at `f63bdf2b` — the
#: live `ALERT.md` row named above carries this text.
BASE_BODY = f"agent-tts was {REASON}.\nRestarted; {DETAIL}."


def _guardian(tmp_path, monkeypatch):
    """A Guardian whose recovery SUCCEEDS, whose alerts are collected, and whose voice
    channel is awake, unmuted and recorded at the `Popen` boundary."""
    import guardian as G

    args = types.SimpleNamespace(
        repo=str(tmp_path), state=str(tmp_path / "state"),
        guardian_state=str(tmp_path / "gdir"), supervisor_sock="/nonexistent",
        backend_url="http://127.0.0.1:1/health", mcp_url="http://127.0.0.1/health",
        programs="agent-tts", interval=5.0,
    )
    g = G.Guardian(args)
    g.alerts: list = []
    g.alert = lambda level, title, body, **kw: g.alerts.append(
        {"level": level, "title": title, "body": body, **kw})
    g._beat = lambda: None
    monkeypatch.setattr(g, "recover_service", lambda program, reason: (True, DETAIL))
    monkeypatch.setenv("LLOYD_VOICE_ALERTS", "1")
    monkeypatch.setattr(speak, "in_quiet_hours", _AWAKE)
    spawned: list = []
    monkeypatch.setattr(speak.subprocess, "Popen", lambda *a, **k: spawned.append(a[0]))
    g.spawned = spawned
    return g


def _lose(state_dir, *texts, at=None):
    """Write a real burst into `voice-loss.md` through the real writer.

    Not a hand-made file: `_record_loss` decides what a burst IS (one window, one
    `burst_started`), and a fixture that invented the fields could pin a record the
    writer never produces — which is the shape of every false-green premise in this repo's
    own history. Two texts in = `occurrences: 2` with the newest first in `said`.

    `at` is the clock `_record_loss` sees, for the node that needs a SECOND burst. Left to
    itself the writer stamps `burst_started` with `time.time()` at three decimals, so two
    bursts written in the same millisecond arrive with the same start and the cursor test
    silently compares a burst with itself. Only the clock moves, never a field of the
    record, so what lands on disk is exactly what the writer produces for that instant.
    The patch is restored in `finally` rather than by `monkeypatch` teardown, because the
    rest of the node — the recovery, the stamp renewal — has to run on the real clock.
    """
    real = time.time
    if at is not None:
        speak.time.time = lambda: float(at)
    try:
        for t in texts:
            speak._record_loss(state_dir, t, "ConnectionRefusedError: [Errno 111]")
    finally:
        if at is not None:
            speak.time.time = real
    return speak.read_loss_record(state_dir)


def _worker_text(argv):
    return argv[argv.index("--text") + 1]


# ── clause 1: the recovery body names the loss ────────────────────────
def test_the_recovery_alert_names_the_alerts_the_outage_swallowed(tmp_path, monkeypatch):
    """`Recovered: <program>` cannot be the whole story when the record names losses.

    The title stays exactly as it is — a person or a dashboard matching on
    `Recovered: agent-tts` must still match — and the count goes in the body, because
    that is where the retraction pass and `ALERT.md` already carry the detail.
    """
    g = _guardian(tmp_path, monkeypatch)
    rec = _lose(g.gdir, LOST, LOST_2)
    assert rec["occurrences"] == 2, rec

    assert g._recover_infra(REASON) == \
        "recovered"

    assert len(g.alerts) == 1, g.alerts
    alert = g.alerts[0]
    assert alert["title"] == "Recovered: agent-tts", alert
    assert "2 spoken alerts did not reach the speakers" in alert["body"], alert["body"]
    # The pre-existing sentences are still there in the same order: the loss is added to
    # the report, never replaces it.
    assert alert["body"].startswith(BASE_BODY.split("\nRestarted; ")[0]), alert["body"]
    assert "Restarted; started;" in alert["body"], alert["body"]


def test_the_body_names_the_count_as_a_number_and_not_only_in_words(tmp_path, monkeypatch):
    """One lost alert is the common case, and its count has to read as a count.

    The noun after the digit agrees with it, because `1 spoken alerts` is the kind of
    wording that makes a reader distrust the rest of the sentence.

    NOT because this sentence is spoken. `utterance_for` composes the utterance from
    `body.split(chr(10))[0]` alone (`agent-services/guardian/speak.py:534`), and the loss
    sentence is the body's SECOND line — so what the synthesiser the guardian just
    recovered actually says is the first line plus `Restarted; started;`, and this count
    reaches the room only inside the loss-naming re-speak of clause 2, never in the
    recovery alert itself. `test_only_the_first_body_line_is_spoken` pins that split, and
    it is a design fact this round did not change: the recovery alert is a written record,
    and the words are carried by the replay.
    """
    g = _guardian(tmp_path, monkeypatch)
    _lose(g.gdir, LOST)

    g._recover_infra(REASON)

    body = g.alerts[0]["body"]
    assert "1 spoken alert did not reach the speakers" in body, body
    assert "1 spoken alerts" not in body, body


def test_only_the_first_body_line_is_spoken():
    """Where the recovery alert stops being a written record and becomes speech.

    #2264 adds its loss sentence as the body's SECOND line, and `utterance_for` composes
    from `body.split(chr(10))[0]` (`agent-services/guardian/speak.py:534`, the line that
    builds `first_body`), so the utterance the recovered synthesiser will play is the title
    line plus the first body line and no more. That is the reason clause 1 and clause 2 are
    two clauses and not one: the count is the written witness, and the only thing that
    reaches the room's ears is the replay of the lost words.

    The node is a check on THIS diff's boundary, not a restatement of the helper: a future
    `utterance_for` that read the whole body would start playing multi-line alert bodies —
    every newline in every guardian report — and would have to answer to this node first.
    """
    body = ("agent-tts was agent-tts: STOPPED without an intentional stop.\n"
            "Restarted; started; http://127.0.0.1:8090/health answering.\n"
            "2 spoken alerts did not reach the speakers during the outage and are being "
            "said again.")

    composed = speak.utterance_for("info", "Recovered: agent-tts", body)

    assert "Recovered: agent-tts" in composed, composed
    assert "agent-tts was agent-tts: STOPPED without an intentional stop." in composed, composed
    assert "2 spoken alerts did not reach the speakers" not in composed, (
        f"the loss sentence reached the spoken utterance, so the split this round's "
        f"wording depends on has moved: {composed!r}")
    assert "http://127.0.0.1:8090" not in composed, (
        "the second line is not spoken either; if it ever is, a URL is being read aloud "
        f"and the answer to that is _speakable's job, not this node's: {composed!r}")


# ── clause 2: the newest lost utterance is spoken again, stamp or not ──
def test_the_recovery_re_speaks_the_newest_lost_utterance_once(tmp_path, monkeypatch):
    """The words are on disk and never reached the room; recovery is the room's last chance.

    `said` is newest-first by construction (`_record_loss` prepends), so the newest is
    `said[0]`, and it goes to the worker verbatim — re-composing it through
    `utterance_for(level, title, body)` would be a second opinion about a text whose
    whole record is the words that were lost.
    """
    g = _guardian(tmp_path, monkeypatch)
    _lose(g.gdir, LOST, LOST_2)

    g._recover_infra(REASON)

    texts = [_worker_text(a) for a in g.spawned]
    assert LOST_2 in texts, f"the newest loss was not re-dispatched: {texts}"
    assert texts.count(LOST_2) == 1, f"the loss was re-said more than once: {texts}"
    # One utterance, not the burst: `occurrences` says how much was lost, and the re-say
    # is the newest thing only. Reading out the whole `said` list is #2264 owed 2 —
    # Alan's ruling after one real recovery — and re-saying a stale line is the outcome
    # he was asked to weigh, so the older text staying un-spoken is the shipped policy.
    assert LOST not in texts, f"the replay spoke more than the newest loss: {texts}"


def test_the_re_speak_argv_replays_through_the_worker_cli_byte_for_byte(tmp_path,
                                                                        monkeypatch):
    """The argv captured above is fed back through `speak.main`, because argv is not speech.

    The re-speak crosses a process boundary, and the assertion that it crossed with the
    right WORDS is only half of it: `dispatch` writes `--text <text> --key <key>` into an
    argv list, `main` parses it back with argparse, and #1904's `--key` bug was a defect in
    exactly that pair. Reading the captured argv proves what the parent intended; replaying
    it proves the child receives the same bytes.

    One seam is stubbed — `speak.speak_now`, the boundary under test — so everything above
    it (`main`'s argparse, the state-dir resolution, the exit code) runs for real, and the
    synthesiser and player below it stay `tests/test_guardian_speak.py`'s subject. Calling
    `speak_now` here instead would reproduce the ~25 s stall the file docstring above is
    about, for no additional boundary.

    The last two assertions are a property of the route rather than of this dispatch. An
    option VALUE whose first character is `-` is the one shape argparse can refuse as
    `--text`, and the reason that cannot happen here is that every text the loss record can
    hold came out of `utterance_for`, which always puts a lead word first — so that is what
    is pinned, in both directions: the utterance a real level composes, and the text this
    replay actually handed the worker.
    """
    g = _guardian(tmp_path, monkeypatch)
    record = _lose(g.gdir, LOST, LOST_2)

    g._recover_infra(REASON)

    assert len(g.spawned) == 1, g.spawned
    argv = list(g.spawned[0])
    assert argv[0] == speak._worker_python(), argv[0]
    assert argv[1] == str(Path(speak.__file__).resolve()), argv[1]

    seen: dict = {}

    def fake_speak_now(text, cfg, state_dir, key=None):
        seen.update(text=text, key=key, state=str(state_dir),
                    voice=bool(cfg.get("api_url")))
        return True

    monkeypatch.setattr(speak, "speak_now", fake_speak_now)
    rc = speak.main(argv[2:])          # argv[0] is the interpreter, argv[1] the script

    assert rc == 0, "the worker CLI reported a failed utterance"
    assert seen.get("text") == record["said"][0], (
        f"the text changed crossing the CLI: {seen.get('text')!r} != {record['said'][0]!r}")
    assert seen.get("key") == seen.get("text"), (
        "the slot the worker would give back on failure is not the utterance it was told "
        f"to speak: {seen.get('key')!r}")
    assert seen.get("state") == str(g.gdir), seen.get("state")
    # The one dash assert that is load-bearing here: `dispatch` puts `composed` into argv
    # as a value of `--text`, and a value that BEGINS with a dash would be a live argv if
    # argparse ever moved --text behind `--`. This is the argv-crossing node, so it is where
    # that belongs. What this node deliberately does NOT do is sweep `utterance_for`'s
    # levels: the leading token is always `Critical.` / `Guardian alert.` / the config's
    # `lead` (`speak.py:528-531`), so no level can produce a dash, and a loop over four
    # levels asserting it cannot was the review's advisory on round SM_20261006_023308 —
    # ceremony that passes whatever the diff does.
    assert not seen["text"].startswith("-"), seen["text"]


def test_the_re_speak_is_not_suppressed_by_a_stamp_for_its_own_text(tmp_path, monkeypatch):
    """A failed attempt must not mute the retry, which is the #2256 class at the replay.

    `voice_spoken.json` still carries pre-#2256 residue — the live store held
    `Landed: #2258 round 2: satisfy clause 6 …` stamped 18:05:36, spent by the OLD
    guardian and never released — and `policy.VOICE_REPEAT_SECONDS` is one hour, so a
    plain `dispatch` of the identical text would be swallowed before it spawned anything.
    The record itself is the proof that the text was never heard, so it outranks the
    stamp; and the stamp is then renewed at recovery time, which is the live witness the
    item's acceptance names.
    """
    g = _guardian(tmp_path, monkeypatch)
    _lose(g.gdir, LOST)
    stale = time.time() - 60.0
    (g.gdir / speak.SPOKEN_NAME).write_text(json.dumps({LOST: stale}), encoding="utf-8")

    g._recover_infra(REASON)

    texts = [_worker_text(a) for a in g.spawned]
    assert LOST in texts, (
        f"a stale stamp suppressed the replay, which is the #2256 bug wearing a new hat: "
        f"{texts}")
    seen = json.loads((g.gdir / speak.SPOKEN_NAME).read_text(encoding="utf-8"))
    assert seen[LOST] > stale, f"the re-speak did not renew its own slot: {seen}"


def test_the_re_speak_honours_the_master_mute(tmp_path, monkeypatch):
    """The replay rides the same route, so `LLOYD_VOICE_ALERTS=0` still means mute.

    If the recovery path reached around `dispatch` and played the stored text itself, a
    muted box would start talking — the one property `voice_enabled` exists to hold.
    """
    g = _guardian(tmp_path, monkeypatch)
    _lose(g.gdir, LOST)
    monkeypatch.setenv("LLOYD_VOICE_ALERTS", "0")

    g._recover_infra(REASON)

    assert [_worker_text(a) for a in g.spawned] == [], (
        "the replay bypassed the master mute")
    assert "1 spoken alert did not reach the speakers" in g.alerts[0]["body"], (
        "the count still belongs in the body even when nothing can be re-said")


# ── clause 3: at most once per burst ─────────────────────────────────
def test_a_second_recovery_over_the_same_burst_dispatches_nothing(tmp_path, monkeypatch):
    """The cursor is the difference between one re-said alert and a loop.

    Two recoveries of the same burst are the normal shape of a flapping `agent-tts` — and
    `RECOVERABLE_INFRA` is restarted from `tick()` every time the liveness read names it,
    so a replay with no cursor would re-speak the same stale `Landed:` line on every
    flap. The cursor records the burst it spent, not the text, because the burst is what
    `voice-loss.md` is a record OF: the record is a per-burst overwrite, so there is no
    backlog to drain later and one burst can only ever be said once.
    """
    g = _guardian(tmp_path, monkeypatch)
    rec = _lose(g.gdir, LOST, LOST_2)
    reason = "agent-tts: STOPPED without an intentional stop."

    g._recover_infra(reason)
    first = [_worker_text(a) for a in g.spawned]
    assert LOST_2 in first, f"the newest of the two losses was not re-said: {first}"

    cursor = json.loads((g.gdir / speak.REPLAY_CURSOR_NAME).read_text(encoding="utf-8"))
    assert cursor["burst_started"] == pytest.approx(rec["burst_started"]), cursor

    g.spawned.clear()
    g2 = _guardian(tmp_path, monkeypatch)     # a fresh process, same state dir
    g2._recover_infra(reason)
    assert [_worker_text(a) for a in g2.spawned] == [], (
        "the same burst was re-said by the second recovery")
    # The alert still says what happened, with its count: the cursor stops the SPEECH,
    # not the report.
    assert "2 spoken alerts did not reach the speakers" in g2.alerts[0]["body"], \
        g2.alerts[0]["body"]


def test_a_later_burst_is_replayed_again(tmp_path, monkeypatch):
    """The cursor bounds ONE burst; the next outage is new news.

    `LOSS_WINDOW` is an hour, so a burst that starts after the window closes has a new
    `burst_started` and a re-speak is owed again — a cursor that swallowed every future
    burst would put this fix into permanent service after its first use.
    """
    g = _guardian(tmp_path, monkeypatch)
    first = _lose(g.gdir, LOST)
    g._recover_infra(REASON)
    assert LOST in [_worker_text(a) for a in g.spawned]

    # The second hour's outage, written through the same writer with the clock moved
    # past `LOSS_WINDOW` — not by deleting the file, which would fabricate a state the
    # writer never produces. Crossing the window is what makes it a new burst: the count
    # restarts and the named list restarts with it (#1913), so the cursor left by the
    # first replay must not be allowed to swallow it.
    second = _lose(g.gdir, LOST_2, at=time.time() + speak.LOSS_WINDOW + 60.0)
    assert second["burst_started"] > first["burst_started"], (first, second)
    assert second["occurrences"] == 1, f"not a new burst: {second}"
    g.spawned.clear()

    g._recover_infra(REASON)

    assert LOST_2 in [_worker_text(a) for a in g.spawned], g.spawned


# ── clause 4: nothing lost, nothing changed ──────────────────────────
def test_a_recovery_with_no_loss_record_is_worded_exactly_as_before(tmp_path, monkeypatch):
    """The overwhelmingly common recovery must not start reporting a loss.

    No `voice-loss.md` at all is the normal state, and a body that said `0 spoken alerts`
    would be a claim about an outage nobody had. This node pins the shipped bytes: title
    and body as the `self.alert(...)` call inside it writes them today, and no re-speak dispatched.
    """
    g = _guardian(tmp_path, monkeypatch)
    assert speak.read_loss_record(g.gdir) is None

    assert g._recover_infra(REASON) == \
        "recovered"

    assert len(g.alerts) == 1, g.alerts
    assert g.alerts[0]["title"] == "Recovered: agent-tts", g.alerts
    assert g.alerts[0]["body"] == BASE_BODY, g.alerts[0]["body"]
    assert [_worker_text(a) for a in g.spawned] == [], g.spawned


def test_a_zero_occurrence_record_changes_nothing_and_dispatches_nothing(tmp_path,
                                                                        monkeypatch):
    """`occurrences: 0` is not a loss, and the parser answers on a file that lost the line.

    Two shapes reach the same no-op: a record stamped with a count of zero, and a body
    whose `occurrences` line has gone — `read_loss_record` returns None for that one
    (`_parse_loss_body`'s missing-count branch: a file that lost its count is not a record, and treating it as
    one would file an alarm whose number is invented). Both have to leave the recovery
    exactly as it reads today.
    """
    g = _guardian(tmp_path, monkeypatch)
    (g.gdir / speak.LOSS_NAME).write_text(
        f"{speak.LOSS_HEADING}\n\noccurrences: 0\n"
        "last_seen: 2026-10-05T18:13:26-0700\nburst_started: 1791248006.000\n"
        "window_s: 3600.0\n\n## What did not get said\n\n"
        f"- \"{LOST}\"\n", encoding="utf-8")
    assert speak.read_loss_record(g.gdir)["occurrences"] == 0

    g._recover_infra(REASON)

    assert g.alerts[0]["body"] == BASE_BODY, g.alerts[0]["body"]
    assert [_worker_text(a) for a in g.spawned] == [], g.spawned

    (g.gdir / speak.LOSS_NAME).write_text(
        f"{speak.LOSS_HEADING}\n\nlast_seen: 2026-10-05T18:13:26-0700\n", encoding="utf-8")
    assert speak.read_loss_record(g.gdir) is None
    g.spawned.clear()

    g._recover_infra(REASON)

    assert g.alerts[-1]["body"] == BASE_BODY, g.alerts[-1]["body"]
    assert [_worker_text(a) for a in g.spawned] == [], g.spawned


# ── clause 5: the numbers the item reports come from committed bytes ─────────
#
# The review of round SM_20261006_023308 graded this clause `partial` and named the
# reason: the only committed supervisord bytes were #2256's witness, which "stops exactly
# at the SIGTERM row cited as :2164, so the item's own headline citation — lines
# 2177-2178, agent-tts spawned 18:13:40 and RUNNING 18:13:50 — lies beyond the committed
# bytes and the ~56 s/~40 s report cannot be re-derived from them". The round that is
# being graded now had no node for clause 5 at all; this is it.
#
# The bytes are two NEW vault files, not an append to #2256's. Extending
# `backlog/data/supervisord.log` was tried first and reverted (`2b7f6d04` by `72a6488b`)
# after reproducing the damage: #2256 pins that file's sha256 AND asserts no `spawned:`
# row between 15:08:12 and 18:14:00, which is exactly the window the restart occupies —
# `tests/test_guardian_rollback.py::test_the_witness_log_still_shows_the_stop_with_no_spawn_after_it`
# goes red the moment the restart's rows are inside it. Two items cannot pin one file with
# opposite claims about it, so this node reads `…2264-supervisord-restart-window.log` and
# asserts the seam between the two witnesses instead (below): row 1 here IS row 2164 there.
#
# Runs under the gate, unmarked: the automod round home symlinks `~/obsidian` across
# (`scripts/automod/worktree.py:78-120` — "every entry but `lloyd` is symlinked across"),
# so these committed bytes are readable in the round home, which is also how #2256's node
# works. It is the mutable vault that needs `live_vault`; `backlog/data/` is durable.

WINDOW_WITNESS = "backlog/data/2026-10-05.2264-supervisord-restart-window.log"
ALERT_WITNESS = "backlog/data/2026-10-05.2264-alert-md-recovery-witness.md"
PINNED_WITNESS = "backlog/data/supervisord.log"

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) ")


def _vault() -> Path:
    return Path(os.environ.get("LLOYD_VAULT") or (Path.home() / "obsidian"))


def _ts(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S,%f")


def test_the_committed_restart_window_witness_still_sums_to_the_reported_gap(tmp_path,
                                                                            monkeypatch):
    """The whole `~56 s / ~40 s` report, re-derived from bytes somebody can `wc -l`.

    The item's premise is a timing accident, so the numbers are the load-bearing part of
    it, and the review refused a version of this clause that quoted figures living only in
    an appended `## Findings` note. Everything asserted here comes from two committed
    vault files; `voice.log` is not one of them and is not read — `speak_now` logs only
    failures, so a successful re-speak leaves no line there.

    The window file is 15 rows of the live supervisord log (its lines 2164-2178, `sed -n`,
    no header and no blank line, so `wc -l` on it is the row count and every figure below
    has a row number a reader can check):

      row  8  18:12:45,454  stopped: lloyd-mcp     the landing restart begins
      row 14  18:13:40,724  spawned: 'agent-tts'   55.27 s later — the item's "~56 s to spawn"
      row 15  18:13:50,990  success: RUNNING       10.27 s after spawn
      ALERT  written: 18:14:20.926               40.20 s after spawn — "~40 s more"

    Two honesty notes the node states rather than smooths over. (1) Row 15 is a `success:`
    row whose own text says "> than 10 seconds (startsecs)": it cannot report a spawn-to-
    RUNNING figure under that floor, so 10.27 s is a LOWER bound and the true interval is
    unknown from these bytes — which is also why the restart→RUNNING sum (65.54 s) is
    asserted as a sum and never as a measured readiness. (2) `/health` answering is in no
    log at all; `ALERT.md` is its only source, which is why that file is committed too.
    """
    vault = _vault()
    window = vault / WINDOW_WITNESS
    alert_w = vault / ALERT_WITNESS
    assert window.is_file(), f"{WINDOW_WITNESS} is not on disk — clause 5 has no bytes"
    assert alert_w.is_file(), f"{ALERT_WITNESS} is not on disk — clause 5 has no bytes"

    rows = window.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 15, f"the witness is {len(rows)} rows, not the 15 it was cut at"
    assert all(TS.match(r) for r in rows), "a row that is not a supervisord timestamp row"
    assert all(r.startswith("2026-10-05") for r in rows), "one witness, one day"

    def row_containing(needle: str) -> tuple[int, str]:
        hits = [(i + 1, r) for i, r in enumerate(rows) if needle in r]
        assert len(hits) == 1, f"{needle!r} matched {len(hits)} rows, expected exactly 1"
        return hits[0]

    n_restart, restart = row_containing("2026-10-05 18:12:45,454 WARN stopped: lloyd-mcp")
    n_spawn, spawn = row_containing("spawned: 'agent-tts' with pid 3696388")
    n_running, running = row_containing("success: agent-tts entered RUNNING state")
    assert (n_restart, n_spawn, n_running) == (8, 14, 15), (n_restart, n_spawn, n_running)

    t_restart = _ts(TS.match(restart).group(1))
    t_spawn = _ts(TS.match(spawn).group(1))
    t_running = _ts(TS.match(running).group(1))

    to_spawn = (t_spawn - t_restart).total_seconds()
    assert abs(to_spawn - 55.27) <= 0.01, to_spawn
    # The claim the item makes is "~56 s to spawn"; the test is an inequality with the
    # measured figure in its message, so this fails on the CLAIM (a future witness whose
    # spawn is 4 s later no longer matches "~56 s"), not on one float's spelling.
    assert 30 <= to_spawn <= 75, f"the '~56 s to spawn' report is wrong: {to_spawn} s"

    to_running = (t_running - t_spawn).total_seconds()
    assert abs(to_running - 10.27) <= 0.01, to_running
    assert "startsecs" in running and "> than 10 seconds" in running, running
    assert to_running >= 10.0, (
        "a success row cannot report less than its own startsecs floor; if it did, this "
        f"witness is not the log's shape any more: {to_running} s")

    # The sum, because readiness itself is unknown inside that floor.
    to_ready = (t_running - t_restart).total_seconds()
    assert abs(to_ready - (to_spawn + to_running)) <= 0.01, to_ready

    # `/health` answering: one row in the ALERT witness, and it is unique, so this cannot
    # be reading a `written:` mention in the surrounding prose.
    written = re.findall(r"(?m)^written: (\S+)$", alert_w.read_text(encoding="utf-8"))
    assert len(written) == 1, f"expected one `written:` row, found {written!r}"
    t_health = datetime.fromisoformat(written[0])
    assert t_health.tzinfo is not None, (
        f"{written[0]!r} carries no offset, so it is naive and every later reader would "
        "read it as UTC — the class of bug recorded in loaded memory on 2026-09-21")

    # supervisord writes naive LOCAL stamps; the alert writes an offset. Subtracting one
    # from the other raises TypeError, and the lazy repair — dropping the offset — is the
    # bug that class names, because it silently reads a `-07:00` instant as UTC. So the
    # alert's own offset is attached to the log stamps: both artifacts come off this one
    # host in this one local zone, the alert is what NAMES that zone, and the arithmetic
    # below is only the gap between two instruments if that is stated rather than assumed.
    tz = t_health.tzinfo
    t_spawn = t_spawn.replace(tzinfo=tz)
    t_restart = t_restart.replace(tzinfo=tz)

    assert t_spawn <= t_health, (
        "the alert says /health was answering BEFORE the process was even spawned; the "
        f"committed bytes disagree: spawn {t_spawn} vs {t_health}")
    to_health = (t_health - t_spawn).total_seconds()
    assert abs(to_health - 40.20) <= 0.01, to_health
    assert 20 <= to_health <= 120, f"the '~40 s more' report is wrong: {to_health} s"

    # And the accident itself, which is what the whole item is about. The burst coalescer's
    # window is `window_s: 3600.0` (the live `voice-loss.md` front matter, and
    # `speak.LOSS_HEADING`'s own field list), and the restart from the first stopped row to
    # `/health` answering is ~2% of it — so the burst that had just recorded the landing
    # alert was certain to still be open when the recovery alert was written. Two asserts,
    # one per side of that overlap: the restart fits inside a window, and the witness spans
    # at least one window, so a reader who doubts the arithmetic can do it on the row count.
    burst_open = (t_health - _ts(TS.match(rows[0]).group(1)).replace(tzinfo=tz)).total_seconds()
    assert to_ready <= 3_600 <= burst_open, (
        "the restart no longer straddles the 3600 s window, so the overlap this item is "
        f"about has stopped happening: {to_ready} s of restart inside {burst_open} s of "
        "witness")
    assert to_ready / 3_600 <= 0.05, (
        f"the restart is {to_ready / 3_600:.0%} of the burst window, no longer the small "
        "fraction inside it that makes the overlap unavoidable")

    # The seam with #2256's witness: row 1 here is byte-identical to the last row there.
    # Without it the two files could drift into describing different outages while both
    # stayed green, which is the failure mode of splitting one log across two witnesses.
    pinned = vault / PINNED_WITNESS
    assert pinned.is_file(), f"{PINNED_WITNESS} is gone — #2256's witness moved"
    last_pinned = pinned.read_text(encoding="utf-8").splitlines()[-1]
    assert rows[0] == last_pinned, (
        "this window no longer starts where #2256's no-spawn window ends:\n"
        f"  window row 1: {rows[0]!r}\n  pinned last row: {last_pinned!r}")
    # And that the pinned witness still ends where #2256 cut it, which is the row THIS
    # window starts from. Its sha256 and its "no spawn inside the hour" claim belong to
    # `tests/test_guardian_rollback.py::
    # test_the_witness_log_still_shows_the_stop_with_no_spawn_after_it`, and are not
    # re-pinned here: a second copy of that sha is a second number that has to move in
    # step, and rows BEFORE 15:08:12 contain plenty of legitimate `spawned:` rows — the
    # first draft of this line asserted the last eight rows were spawn-free and was simply
    # wrong about what the file contains.
    pinned_rows = pinned.read_text(encoding="utf-8").splitlines()
    assert len(pinned_rows) == 2164, f"#2256's witness is {len(pinned_rows)} rows, not 2164"
    assert "WARN stopped: agent-tts (terminated by SIGTERM)" in last_pinned, last_pinned

    # The pre-fix shape, in the artifact clause 1 changes: the committed recovery alert
    # names the restart and no loss. If this ever passes because the copy was regenerated
    # AFTER the fix shipped, it stops being a premise witness — so the row count of the
    # copy is pinned too, and it is 9 lines because that is all the file holds.
    copied = alert_w.read_text(encoding="utf-8").split("---\n")[-1]
    assert "Recovered: agent-tts" in copied, copied[:200]
    assert "Restarted; started;" in copied, copied[:200]
    assert "did not reach the speakers" not in copied, (
        "the committed ALERT.md copy names a loss, so it is no longer the pre-fix "
        "artifact this clause is about")

    # A recovery against THIS record shape behaves as clause 1 requires: the count in the
    # body is the number of rows in the loss record, and it is a digit. One real loss row
    # is used because the window witness documents one swallowed alert's timing; the
    # multi-loss wording is the other node's job.
    g = _guardian(tmp_path, monkeypatch)
    _lose(g.gdir, LOST)
    g._recover_infra(REASON)
    assert "1 spoken alert did not reach the speakers" in g.alerts[0]["body"]
