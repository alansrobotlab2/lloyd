"""Speak an alert aloud in Lloyd's cloned voice — the guardian's sixth channel.

Why this file lives in `agent-services/guardian/` rather than beside
`tts_shaping.py`: `guardian-stage.sh` stages `agent-services/guardian/*.py`
and nothing else into the pinned snapshot. A module the guardian imports
must be in this directory or it simply will not exist at runtime, and the
failure would appear only in production.

Three properties, in priority order:

1. **It must never delay the loop.** The unit sets `WatchdogSec=90` against a
   5s tick. Synthesis takes seconds and playback takes as long as the
   sentence, so `dispatch` spawns a detached worker and returns immediately.
   It reports *dispatched*, never *heard* — the honest thing a fire-and-forget
   channel can claim.
2. **It must never raise.** Same contract as every other channel in
   `notify.py`.
3. **Shaping is a soft dependency.** The presence EQ and the WSOLA speed live
   in `agent-services/tts_shaping.py`, which needs numpy and sits outside the
   snapshot. When it imports, the alert sounds like Lloyd; when it does not,
   the alert is still spoken, just flat and at 1.0. The guardian's own failure
   domain never widens, because the worker is a separate process and the other
   five channels have already fired before it starts.

Suppression is on **disk**, not in memory, because the two producers are
different processes: the long-lived guardian daemon and the 15-minute
`lloyd-guardian-nag` oneshot. `guardian.py`'s in-memory `_alert_seen` cannot
see the nag, so without a shared record a persistent BROKEN state would say
the same sentence out loud four times an hour forever, which is how you teach
someone to unplug the speakers. See `policy.VOICE_REPEAT_SECONDS`.

Failures are logged to `voice.log` in the guardian state dir, each line stamped
with the LOCAL wall clock and an explicit UTC offset (`%z`, #1808) — marked, not
converted, because the state dir's other artefacts stamp UTC and a reader that
assumed it here would be hours wrong. An alerting
channel that fails silently is the exact anti-pattern `notify.py`'s own
docstring is about — and a *record* was all that `synth failed` had, which on
2026-09-28 was the wrong amount: three utterances refused with `[Errno 111]
Connection refused` at 11:50:51 and 11:51:18, the second of them the alert
announcing that `agent-supervisord` was unreachable, and the only trace that the
sound never came out was a line in the log of the channel that had just failed.
So a refused synthesis now does two more things. It writes a durable loss record
to `voice-loss.md` beside this log — a plain filesystem write, because the API
that would otherwise file an item lives in `[program:lloyd-backend]`, the same
supervised tree as `[program:agent-tts]`, so a backlog-bound report dies exactly
when it is most needed — and it hands the local player a short chirp built in
this process from the stdlib `math`/`array` pair, because what survives that
outage is the *player* (`/usr/bin/pw-play`), not the synthesiser. In-process
speech is not on offer on this box: `espeak`, `espeak-ng`, `spd-say`, `festival`,
`flite` and `pico2wave` are all absent. The words are already on the toast, the
journal, the daily note and now this record; the chirp's job is only that the
room is not silent at the moment the alert was supposed to land.
"""

from __future__ import annotations

import array
import fcntl
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

try:                                    # policy is staged beside us
    import policy as _policy
    _REPO = _policy.REPO
except Exception:                       # standalone / test import
    _REPO = "/home/alansrobotlab/lloyd"

# Mirrors config.yaml `livekit.tts`. These are defaults, not the source of
# truth: `agent-services/bin/sync-voice-config.py` writes voice.json from
# config.yaml at stage time so the two cannot drift. They exist so a guardian
# with no synced config still speaks in the right voice.
DEFAULTS: dict = {
    "api_url": "http://127.0.0.1:8090",
    "model": "qwen3-tts",
    "voice": "clone:dave_cullen",
    "speed": 1.22,
    "sample_rate": 24000,
    "tail_silence_ms": 250,
    "presence_eq": True,
    "shelves": [
        {"freq": 1800, "gain_db": 2.75, "q": 1.0},
        {"freq": 8000, "gain_db": 4.0, "q": 0.7},
    ],
    "shaping_module": f"{_REPO}/agent-services/tts_shaping.py",
    "synth_timeout": 60.0,
    "max_chars": 240,
    # Two ceilings on *consuming* a response, both distinct from `synth_timeout`.
    # That one is per blocking socket operation, so a stream that keeps handing
    # over bytes more often than every 60 s never trips it and `read()` keeps
    # accumulating: on 2026-09-28 19:51 a decoder that never reached EOS played
    # 655.57 s of scrambled audio — 31,467,360 bytes of s16le at 24000 Hz — for a
    # 160-character alert, and `budget = len(pcm) / (2 * sr) + 15.0` in `play`
    # turned that garbage into a 670.6 s speaker booking. Neither number is a
    # guess: `max_chars: 240` caps the utterance, which is ~20 s of speech at
    # `speed: 1.22`, so 60 s of audio is ~3x the longest legitimate alert and
    # under a tenth of the incident; 90 s of wall clock is above `synth_timeout`
    # (so a stalled socket still trips *that* bound first, as it has six times in
    # voice.log) and 7x under the broadcast. Both stay DEFAULTS-only:
    # `sync-voice-config.py` projects six keys out of config.yaml and this pair
    # has no config.yaml path, exactly like `synth_timeout` and `max_chars`.
    "max_audio_seconds": 60.0,
    "synth_wall_clock": 90.0,
    # Spoken alerts only. This must never reach livekit_worker: voice mode has
    # to stay conversational at any hour, and a Lloyd who goes mute mid-answer
    # at 23:00 is a bug, not a courtesy. That is why the setting lives under
    # `guardian.voice` in config.yaml rather than `livekit.tts`.
    "quiet_hours": {"enabled": True, "start": 23, "end": 7, "allow_critical": False},
}

VENV_PYTHON = f"{_REPO}/.venvs/lloyd/bin/python"

CONFIG_NAME = "voice.json"
SPOKEN_NAME = "voice_spoken.json"
LOG_NAME = "voice.log"
_LOG_CAP = 256 * 1024
# Room around a playback attempt: process start-up, device contention, the
# player's own teardown. It was a bare `15.0` inside a budget that grew without
# bound, where nobody had to say what it was for; now it is the only thing
# besides the ceiling that sets that budget, so it has a name.
_PLAY_SLACK_S = 15.0

# ── when the synthesiser is the thing that broke ──────────────────────
# The durable record of a lost utterance (#1806). Deliberately a *second file*
# rather than more lines in `voice.log`: `voice.log` is the channel that just
# failed, it is self-capped at `_LOG_CAP` (it truncates itself to empty at
# 256 KiB, so a burst that outlives a maintenance pass erases the evidence), and
# the item's whole complaint is that a lost alert left only that line behind.
LOSS_NAME = "voice-loss.md"
# One record per burst. `notify.py` hands `dispatch` a `voice_window` of 3600 s
# (`notify.py:122`) and `policy.VOICE_REPEAT_SECONDS` is the same hour, so an
# outage long enough to swallow several alerts is exactly the span during which
# five lines about it are noise and one escalating record is the report.
LOSS_WINDOW = 3600.0
# What one record names. Bounded because this file is written by a failure path
# that must not grow without limit — the same mistake `_LOG_CAP` exists to stop.
LOSS_TEXT_KEEP = 5
LOSS_TEXT_CHARS = 200
LOSS_HEADING = "# Voice alert lost"
# Which burst the recovery already re-spoke (#2264). A burst, not a text, because
# a burst is what `voice-loss.md` is a record OF and the record is a per-burst
# overwrite: the five losses of 2026-10-05 15:15-16:17 were already gone from the
# file, replaced by the 18:05/18:13 pair, so there is no backlog to drain later
# and one burst can only ever be said once. In the guardian state dir rather than
# in memory because the process that recovers the synthesiser is the one that may
# be restarting: a cursor a restart forgets re-says the same stale `Landed:` line
# on the next flap, which is the same "a repeating reader must not re-act" rule
# `voiceloss.CURSOR_NAME` exists for one file over.
REPLAY_CURSOR_NAME = "voice_replay_cursor.json"

# The locally-generated attention chirp. Under a second so that
# `_FALLBACK_PLAY_TIMEOUT` can hold the whole failure path inside the 2.0 s the
# watchdog contract needs, and a rising minor-third rather than one steady tone:
# a steady tone is what a smoke alarm makes, and a room learns to ignore those.
TONE_SECONDS = 0.9
TONE_HZ = (740.0, 988.0)
TONE_AMPLITUDE = 0.25          # of full scale: audible across a room, not a shriek
FALLBACK_PLAY_TIMEOUT = 1.5    # ceiling on the chirp's own playback, see _audible_fallback


def voice_enabled() -> bool:
    """Master mute, checked at dispatch time.

    `LLOYD_VOICE_ALERTS=0` keeps every other channel and drops only the
    speech. Two callers need this and neither is hypothetical: the test suite,
    which must never make noise on a developer's machine, and anyone who wants
    the toasts without the talking at two in the morning.
    """
    return str(os.environ.get("LLOYD_VOICE_ALERTS", "1")).strip().lower() not in (
        "0", "false", "no", "off", "")


# ── config ────────────────────────────────────────────────────────────
def load_config(state_dir: Path) -> dict:
    """DEFAULTS overlaid with voice.json, which never has to exist."""
    cfg = dict(DEFAULTS)
    try:
        raw = (Path(state_dir) / CONFIG_NAME).read_text(encoding="utf-8")
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            for k, v in loaded.items():
                if v is None:
                    continue
                # One level of merge: {"quiet_hours": {"start": 22}} must move
                # the hour without silently dropping `enabled` and `end`.
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k] = {**cfg[k], **v}
                else:
                    cfg[k] = v
    except Exception:
        pass
    return cfg


def _log(state_dir: Path, message: str) -> None:
    # The stamp carries an explicit UTC offset (#1808) and the digits stay the
    # LOCAL wall clock: `%z` marks which zone those digits are in, it does not
    # convert them. Writing `datetime.now(timezone.utc)` here would satisfy the
    # offset and move every line seven hours, which is the opposite of the fix —
    # the 2026-09-28 outage is understood by lining these lines up with a human's
    # memory of the evening. The marker matters because the file is machine-read
    # against artefacts in the same state dir that DO mark UTC (`gstate.py:26`
    # and `memwatch.py:210` both write a trailing `Z`), and because the three
    # utterances refused at 11:50:51 and 11:51:18 that day read, taken as UTC, as
    # an outage beginning 59 minutes before the outage.
    try:
        p = Path(state_dir) / LOG_NAME
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and p.stat().st_size > _LOG_CAP:
            p.write_text("", encoding="utf-8")
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}\n")
    except Exception:
        pass


_LOSS_COUNT_RE = re.compile(r"^occurrences:\s*(\d+)\s*$", re.M)
_LOSS_STARTED_RE = re.compile(r"^burst_started:\s*([0-9.]+)\s*$", re.M)
_LOSS_SAID_RE = re.compile(r'^- "(.*)"$', re.M)
_LOSS_LAST_SEEN_RE = re.compile(r"^last_seen:\s*(\S+)\s*$", re.M)
_LOSS_FIRST_SEEN_RE = re.compile(r"^first_seen:\s*(\S+)\s*$", re.M)


def _parse_loss_body(body: str) -> dict | None:
    """The fields of one loss record, or None when the text is not one.

    ONE definition of the format, shared by the writer and the reader. The writer
    needs it because it decides, from the bytes it holds the lock on, whether this
    failure opens a record or refreshes one; the reader is #1904's escalator, and a
    second parser would be a second opinion about a format whose whole contract is
    that the two agree.

    `occurrences` is the only field a body must have to count as a record — a file
    that lost it is not a record, and treating it as one would file an alarm whose
    count is invented. `burst_started` is what bounds a burst, so a record without
    it can be read but never refreshed.
    """
    m_count = _LOSS_COUNT_RE.search(body)
    if m_count is None:
        return None
    m_started = _LOSS_STARTED_RE.search(body)
    m_last, m_first = _LOSS_LAST_SEEN_RE.search(body), _LOSS_FIRST_SEEN_RE.search(body)
    return {
        "occurrences": int(m_count.group(1)),
        "burst_started": float(m_started.group(1)) if m_started else None,
        "last_seen": m_last.group(1) if m_last else None,
        "first_seen": m_first.group(1) if m_first else None,
        "said": _LOSS_SAID_RE.findall(body),
    }


def read_loss_record(state_dir: Path) -> dict | None:
    """The live loss record in `LOSS_NAME`, or None when there is nothing to read.

    #1904's reader. Takes no lock: the writer holds `LOCK_EX` only for the moment it
    reads-and-rewrites, and a reader that blocked on that lock would be a guardian
    tick waiting on a failure path in a detached child, which is the wrong way
    round. The cost is that a read racing a rewrite can catch a truncated file, and
    the answer to that is the one this function already gives — None, try again on
    the next tick — because a guardian tick is a repeating reader and an
    under-counted alarm never was.
    """
    try:
        body = (Path(state_dir) / LOSS_NAME).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    return _parse_loss_body(body)


def _read_replay_cursor(state_dir: Path) -> dict:
    """The last burst the recovery re-spoke, or `{}` when there is no such record.

    Absent, unreadable and unparseable are one answer, and it is the same one
    `gstate.read_json` gives: nothing has been replayed. That is safe here in the
    direction that matters — the worst case is a burst said twice, and the file is
    written atomically below so the torn read that would cause it is not a thing
    this reader can actually see.
    """
    try:
        cur = json.loads((Path(state_dir) / REPLAY_CURSOR_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return cur if isinstance(cur, dict) else {}


def _write_replay_cursor(state_dir: Path, burst_started: float, text: str) -> None:
    """Record the burst as said, atomically, and never raise.

    One small JSON, tmp-then-`os.replace`, which is `gstate.write_json_atomic`'s shape
    and deliberately a copy of three lines rather than an import: `speak.py` is pulled
    in by `eval/speaker_embed_eval.py` and `scripts/voice/tts_bakeoff.py` as a
    standalone module, and it has never imported a sibling from the guardian directory.
    """
    d = Path(state_dir)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "burst_started": float(burst_started),
        # Local digits with an explicit offset, the %z rule #1808 put on `voice.log`
        # and #1912 on ALERT.md: this dir's other artefacts stamp UTC, and a reader
        # that assumed that here would be seven hours wrong about when an alert was
        # re-said.
        "replayed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "text": text[:LOSS_TEXT_CHARS],
    }
    try:
        tmp = d / f"{REPLAY_CURSOR_NAME}.tmp{os.getpid()}"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, d / REPLAY_CURSOR_NAME)
    except OSError:
        pass


#: What `loss_replay_state` answers, and the only three things that are true about a
#: burst and the voice that owes it a re-saying.
REPLAY_OWED, REPLAY_SPOKEN, REPLAY_NONE = "owed", "spoken", "none"


def loss_replay_state(record: dict | None, state_dir: Path) -> str:
    """`owed`, `spoken` or `none` for a loss record already read (#2264).

    One call, one answer, three states rather than a bool, because the recovery has to
    SAY which of them it is in: a body that promised "none of them has been said again"
    on the second recovery of a burst the first recovery already re-said would be a
    false statement about the room, written by the one process that knows the difference.

    * `owed` — the record counts at least one lost utterance, names words to say, and
      the cursor has not spent its burst. This is the re-speak's cue.
    * `spoken` — an earlier recovery already re-said this burst. The losses stay facts
      worth reporting; the speech does not happen twice.
    * `none` — no re-speak is possible or owed: no record, a count of zero, a record
      with no words in it, or a burst with no start to bound. A `Recovered:` body that
      reported `0 spoken alerts` on an uneventful day would be announcing an outage
      nobody had, and the box is in this state nearly always.

    `spoken` means the CURSOR says this process's predecessors said it, which is the only
    sense in which the recovery may claim it out loud. A record whose `burst_started` is
    missing therefore lands in `none`: `_parse_loss_body` allows the field to be absent —
    such a record "can be read but never refreshed" — and a cursor cannot bound a burst
    with no start, so the only available guard would be "re-speak on every recovery",
    which on a flapping `agent-tts` that `tick()` recovers once per tick is the same
    stale line read out once per flap. `none` is also the honest wording for that case:
    the count still goes in the body, and "none of them has been said again" is true
    about it, whereas `spoken` would assert a re-saying that never happened.
    """
    if not record or int(record.get("occurrences") or 0) < 1:
        return REPLAY_NONE
    if not [s for s in list(record.get("said") or []) if str(s).strip()]:
        return REPLAY_NONE
    started = record.get("burst_started")
    if started is None:
        return REPLAY_NONE
    done = _read_replay_cursor(state_dir).get("burst_started")
    if isinstance(done, (int, float)) and float(started) <= float(done):
        return REPLAY_SPOKEN
    return REPLAY_OWED


def replay_latest_loss(record: dict, state_dir: Path, *, window: float) -> bool:
    """Say the newest lost utterance again, and record this burst as said.

    `record` is the one the caller already put to `loss_replay_state`; re-reading the
    file here would be a second opinion about which burst is live, and that same caller
    is composing an alert body from this record one line later — the two must be about
    the same burst.

    Newest is `said[0]` because `_record_loss` prepends, and newest is the right pick on
    merit too: the oldest thing a burst can name is a `Rolled back` line about a deploy
    that is now hours stale, and re-saying that is worse than the chirp that already
    sounded (#2264 owed 2 is Alan's call on whether the whole burst should ever be read
    out — one alert is the shipped policy).

    The cursor moves ONLY on a successful spawn. Quiet hours and the master mute both
    return False without saying anything, and stamping a burst that was never re-said
    would mark a debt paid and lose the alert permanently — the rule #1904's escalator
    already states from the other side: "the cursor is NOT advanced on that path, so the
    escalation is still owed".

    `force` is what gets the words out when the store still holds a stamp for them (see
    `should_speak`), and the slot is the utterance itself, so the re-speak leaves its own
    mark in `voice_spoken.json` — the artefact a later reader checks to see whether an
    alert was said twice.
    """
    said = [s for s in list(record.get("said") or []) if str(s).strip()]
    if not said:
        return False
    text = said[0]
    if not dispatch("info", "Re-said after a voice outage", "", state_dir,
                    window=window, key=text, verbatim=text, force=True):
        return False
    started = record.get("burst_started")
    if started is not None:
        _write_replay_cursor(state_dir, float(started), text)
    return True


def _record_loss(state_dir: Path, text: str, reason: str) -> None:
    """Record a lost utterance somewhere other than `voice.log` (#1806).

    One record per burst: a failure inside `LOSS_WINDOW` of the open one
    REFRESHES it — `occurrences` goes up, the new utterance joins the named list,
    `last_seen` moves — rather than appending a second report. That is the shape
    #1798 landed for a dropped daily-note line, and it is what a burst needs:
    the 2026-09-23 cluster was five refusals between 14:47 and 16:03, and five
    records would have made the outage look like five incidents while burying the
    one number that matters, how many times it tried. Past the window the burst is
    over and the next failure starts a NEW record — count back to 1, and the named
    list restarts with its own utterance, because a record that counted one
    incident and quoted another is evidence nobody can act on (#1913).

    Written under the same exclusive lock `should_speak` uses, because the two
    producers are separate processes (the daemon and the nag oneshot) and a lost
    record has to survive both of them. It is a direct filesystem write with no
    HTTP anywhere: the backlog API that #1798 posts to is
    `[program:lloyd-backend]`, which shares `agent-supervisord` with the TTS
    server that just refused, so an outage-shaped failure would take the report
    down with the sound. Never raises — a failure path that can raise is how a
    voice channel ends up taking a watchdog with it.

    A record is not an alarm. Nothing read this file for the two days it existed,
    which is #1904: the escalator lives in the guardian's loop (`voiceloss.py`),
    reading through `read_loss_record`, so the post happens in the one process that
    outlives the outage that silenced the speaker — and the stamps below carry an
    explicit offset, because a file with a reader is a machine-facing payload.
    """
    now = time.time()
    try:
        d = Path(state_dir)
        d.mkdir(parents=True, exist_ok=True)
        line = " ".join(str(text).split())[:LOSS_TEXT_CHARS]
        with open(d / LOSS_NAME, "a+", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.seek(0)
            prev = _parse_loss_body(fh.read())
            count, started = 1, now
            in_window = bool(prev and prev["burst_started"] is not None
                             and now - prev["burst_started"] <= LOSS_WINDOW)
            if in_window:
                count = prev["occurrences"] + 1
                started = prev["burst_started"]
            # One window test decides BOTH halves of a new burst (#1913). Counting
            # from 1 while quoting the ended burst's utterances made the new record
            # evidence for the outage that already finished: the repro's burst B,
            # one lost alert three hours after burst A's two, read
            # `occurrences: 1` and named all three utterances. A record is the
            # report of one incident, so what it names is bounded by the same
            # window that bounds what it counts.
            prior = prev["said"] if in_window else []
            said = [line] + [s for s in prior if s != line][:LOSS_TEXT_KEEP - 1]
            fh.seek(0)
            fh.truncate()
            fh.write(
                # Both stamps carry an explicit UTC offset, the same %z #1808 put on
                # `_log` (:212) and for the same reason: the digits stay the LOCAL
                # wall clock and the marker says which zone they are in. Writing
                # `time.gmtime` here would satisfy the offset and move both stamps
                # seven hours, which is the opposite of the fix. It matters now
                # rather than only for tidiness because #1904 put a reader on this
                # file, and a naive local stamp handed to a UTC-assuming parser is
                # the class that made the 2026-09-28 outage look like it began 59
                # minutes early.
                f"{LOSS_HEADING}\n"
                f"\noccurrences: {count}"
                f"\nlast_seen: {time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(now))}"
                f"\nburst_started: {started:.3f}"
                f"\nfirst_seen: {time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(started))}"
                f"\nwindow_s: {LOSS_WINDOW:.1f}"
                f"\nlast_error: {reason}"
                "\n\nThese alerts were dispatched to the voice channel and never "
                "reached the speakers. The words themselves went to the toast, the "
                "journal, ALERT.md and the daily note; this is the record that the "
                "spoken one was lost.\n"
                "\n## What did not get said\n"
                + "".join(f'- "{s}"\n' for s in said)
            )
    except Exception:
        pass


# ── what to actually say ──────────────────────────────────────────────
# A hash, not merely a long run of hex-legal characters: it must contain at
# least one a-f. Without that lookahead "SM_20260906_a1b2" loses its date and
# every 7-digit number in a body is silently eaten.
_HEX = re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b", re.I)
_PATH = re.compile(r"[~\w.-]*(?:/[\w.-]+)+/?")
_EMPTY_BRACKET = re.compile(r"\(\s*\)|\[\s*\]|\{\s*\}")
_WS = re.compile(r"\s+")
_FILLER_TAIL = re.compile(r"\b(?:to|from|into|and|the|at|on|of|with|by)\s*$", re.I)


def speakable(text: str) -> str:
    """Turn alert prose into something worth hearing out loud.

    Commit hashes are the main offender: "Rolled back 1a2b3c4d → 5e6f7a8b"
    read aloud is sixteen letters of noise, and the hash is in the toast, the
    ledger and the journal for anyone who needs it. Arrows, backticks, paths
    and newlines get the same treatment for the same reason — this is the one
    channel that cannot be skimmed.
    """
    t = text.replace("→", " to ").replace("->", " to ")
    t = t.replace("`", " ").replace("*", " ").replace("_", " ")
    t = _HEX.sub("", t)
    t = _PATH.sub(" ", t)                           # filesystem paths
    t = _EMPTY_BRACKET.sub(" ", t)                  # "(1a2b3c4d)" -> "()" -> gone
    t = _WS.sub(" ", t)
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)         # "good ." after a hash left
    t = t.strip(" .,;:")
    # Removing a hash strips the noun but leaves the preposition that pointed
    # at it: "Rolled back 1a2b3c4d -> 5e6f7a8b" collapses to "Rolled back to",
    # which sounds like the sentence was cut off mid-word. Drop the dangler.
    prev = None
    while prev != t:
        prev = t
        t = _FILLER_TAIL.sub("", t).strip(" .,;:")
    return t


def utterance_for(level: str, title: str, body: str = "", *,
                  max_chars: int = 240) -> str:
    """One or two sentences. Long enough to know what happened, short enough
    that nobody is waiting for it to finish."""
    lead = "Critical." if level == "critical" else "Guardian alert."
    head = speakable(title or "Something needs attention")
    parts = [f"{lead} {head}."]
    rest = speakable((body or "").split("\n")[0])
    if rest and len(rest) > 12:
        parts.append(f"{rest}.")
    out = " ".join(parts)
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0] + "."
    return out


def in_quiet_hours(cfg: dict, now: float | None = None) -> bool:
    """True when the hour says to withhold speech.

    Only the *sound* is withheld. The toast, journal, ledger, vault note and
    backlog task all still fire, so nothing is lost — the record is intact and
    waiting in the morning. That is what makes this safe to default on: it
    drops an accompaniment, never an alert.

    Local time, because the point is what hour it is in the room. A window
    that wraps midnight is the normal case (23 to 7), so `start > end` is
    handled, and `start == end` means no window rather than a full day of
    silence — the reading of an empty range that cannot mute the box forever.
    """
    qh = cfg.get("quiet_hours") or {}
    if not qh.get("enabled", False):
        return False
    try:
        start, end = int(qh["start"]), int(qh["end"])
    except (KeyError, TypeError, ValueError):
        return False
    if start == end:
        return False
    hour = time.localtime(now).tm_hour
    return start <= hour < end if start < end else (hour >= start or hour < end)


# ── suppression, shared across processes ──────────────────────────────
def should_speak(key: str, state_dir: Path, window: float,
                 now: float | None = None, *, force: bool = False) -> bool:
    """True at most once per `window` for a given key, across processes.

    Records the decision under an exclusive lock *before* speaking, so the
    daemon and the nag oneshot racing on the same tick produce one utterance
    rather than two overlapping ones.

    `force` speaks anyway inside the window and still takes the stamp. It exists
    for #2264's replay: the proof that a stored utterance was never heard is the
    loss record that names it, and the stamp beside that record is the residue of
    an attempt that made no sound — the same confusion #2256 fixed for a live
    alert, where a failed synthesis muted the identical text for an hour on the
    strength of a call that produced nothing. `forget_spoken` is the other answer
    and is the wrong one here: it presumes a key this code spent, and pre-#2256
    residue in the live store (`Landed: #2258 round 2: …` stamped 18:05:36 by the
    OLD guardian, never released) is precisely a key the process now holding it
    cannot release. Stamping as well is what makes the replay visible in
    `voice_spoken.json`, which is where a later reader looks to see whether an
    alert was said twice.
    """
    now = time.time() if now is None else now
    try:
        d = Path(state_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / SPOKEN_NAME
        with open(path, "a+", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.seek(0)
            try:
                seen = json.loads(fh.read() or "{}")
                if not isinstance(seen, dict):
                    seen = {}
            except Exception:
                seen = {}
            last = seen.get(key)
            if not force and isinstance(last, (int, float)) and now - last < window:
                return False
            seen[key] = now
            # Keep the file from growing without bound; anything outside the
            # window can no longer suppress anything.
            seen = {k: v for k, v in seen.items()
                    if isinstance(v, (int, float)) and now - v < max(window * 4, 86400)}
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(seen))
        return True
    except Exception:
        # A broken suppression file must not mute the channel.
        return True


def forget_spoken(key: str, state_dir: Path) -> bool:
    """Undo a `should_speak` reservation for `key`, so the same text is speakable again.

    #2256: `should_speak` records the slot BEFORE the words exist, because two processes
    on one tick must produce one utterance. When the synthesiser then refuses, the alert
    was never heard but the hour-long slot is spent, so the guardian's next attempt at the
    identical text is suppressed by a failed attempt — and on 2026-10-05 that was the
    difference between an alert spoken once `agent-tts` came back and one never spoken at
    all. Only a *recorded loss* clears it (see `speak_now`): a playback failure means the
    words were synthesised and something downstream dropped them, which a retry would
    likely repeat into a second half-spoken alert.

    Returns whether the key was there to remove. Never raises — a suppression store that
    cannot be written must not swallow the failure record that got us here, and it must
    not mute the channel either, which is the same rule `should_speak`'s handler applies.
    """
    try:
        path = Path(state_dir) / SPOKEN_NAME
        with open(path, "a+", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.seek(0)
            try:
                seen = json.loads(fh.read() or "{}")
            except Exception:
                return False
            if not isinstance(seen, dict) or key not in seen:
                return False
            seen.pop(key)
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(seen))
        return True
    except Exception:
        return False


# ── synthesis ─────────────────────────────────────────────────────────
# Read granularity on the wire: the size `probes.py:32` already reads at. Big
# enough that a healthy eight-second utterance arrives in a couple of pieces,
# small enough that the loop gets to test its ceilings often.
_READ_CHUNK = 65536


def _pcm_seconds(n_bytes: int, sample_rate: int) -> float:
    """Seconds of mono s16le PCM — the one unit a bound on a *stream* is kept in."""
    return n_bytes / float(2 * int(sample_rate))


def _read_bounded(resp, cfg: dict, state_dir: Path | None = None) -> bytes:
    """Consume a streaming response under two independent ceilings, keeping the head.

    `resp.read()` with no argument was the whole defect: one call that bounds
    neither how many bytes arrive nor how long they take. `synth_timeout` is a
    *per blocking socket operation* timeout, so a stream dripping bytes more
    often than every 60 s never trips it, and the 2026-09-28 19:51 runaway bought
    itself 655.57 s of speakers that way. So:

    * the wall-clock budget is tested **before a read is started**, which is what
      makes it independent of the per-socket timeout — a drip can never buy one
      more read once `synth_wall_clock` is spent. Total cost is that budget plus
      at most one chunk already in flight, never an unbounded read.
    * the byte ceiling is tested on what has arrived, so the answer is a
      truncation rather than a mute: the decoder emits the real sentence first
      and hallucinates after it, so the first `max_audio_seconds` are the alert.

    Which ceiling fired goes to voice.log with the seconds it was set at and the
    seconds received. An alert silently shortened is the same anti-pattern
    `shape()` logs its fallback for.
    """
    sr = int(cfg["sample_rate"])
    ceiling_s = float(cfg["max_audio_seconds"])
    wall_s = float(cfg["synth_wall_clock"])
    cap = int(ceiling_s * sr * 2)
    cap -= cap % 2                                    # whole samples only
    started = time.monotonic()
    parts: list[bytes] = []
    got = 0
    fired: str | None = None
    while True:
        if time.monotonic() - started >= wall_s:
            fired = "wall-clock"
            break
        chunk = resp.read(_READ_CHUNK)
        if not chunk:
            break                                     # the stream ended on its own
        parts.append(chunk)
        got += len(chunk)
        if got >= cap:
            fired = "duration"
            break
    pcm = b"".join(parts)[:cap]
    if fired:
        try:                                          # stop draining the socket now
            resp.close()
        except Exception:
            pass
        recv, kept = _pcm_seconds(got, sr), _pcm_seconds(len(pcm), sr)
        if fired == "duration":
            _log(state_dir, f"synth duration ceiling fired: received {recv:.2f}s of "
                            f"audio, ceiling max_audio_seconds={ceiling_s:.2f}s — kept "
                            f"the first {kept:.2f}s and stopped reading")
        else:
            _log(state_dir, f"synth wall-clock deadline fired: received {recv:.2f}s of "
                            f"audio in {time.monotonic() - started:.2f}s, ceiling "
                            f"synth_wall_clock={wall_s:.2f}s — kept the first {kept:.2f}s")
    return pcm


def synthesize(text: str, cfg: dict, state_dir: Path | None = None) -> bytes | None:
    """Raw s16le PCM from the Qwen3-TTS server, unshaped.

    Requests `stream: true` and `speed: 1.0` to match `livekit_worker`
    exactly: that is the path the voice was tuned against, and the server
    silently drops `speed` there anyway — we apply it ourselves in `shape`.

    The body arrives through `_read_bounded`, not `resp.read()`, and `state_dir`
    is where an over-run is recorded — the same optional argument `shape()` takes.
    """
    payload = json.dumps({
        "model": cfg["model"],
        "input": text,
        "voice": cfg["voice"],
        "response_format": "pcm",
        "stream": True,
        "speed": 1.0,
    }).encode()
    req = urllib.request.Request(
        f"{str(cfg['api_url']).rstrip('/')}/v1/audio/speech",
        data=payload, headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(cfg["synth_timeout"])) as resp:
        if not (200 <= resp.status < 300):
            return None
        return _read_bounded(resp, cfg, state_dir)


def _load_shaping_module(cfg: dict):
    """Import tts_shaping by absolute path.

    By path rather than by name because the worker runs with the snapshot dir
    on sys.path and the repo lives somewhere else entirely; putting the whole
    repo on sys.path to reach one module is a bigger door than this needs.
    """
    try:
        import importlib.util
        path = Path(cfg["shaping_module"])
        if not path.is_file():
            return None
        spec = importlib.util.spec_from_file_location("lloyd_tts_shaping", path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def shape(pcm: bytes, cfg: dict, state_dir: Path | None = None) -> bytes:
    """Presence EQ + WSOLA speed, degrading in tiers rather than all at once.

    The EQ needs scipy; the stretch needs only numpy. A worker that fell back
    to the system interpreter can therefore still fix the *pace* even where it
    cannot fix the presence band, which is worth more than either-or.

    Which tier ran is written to voice.log. A shaping fallback that reports
    nothing is indistinguishable from shaping that worked — that is exactly
    how the first cut of this module shipped unshaped audio while looking
    healthy, and the log line is what caught it.
    """
    mod = _load_shaping_module(cfg)
    if mod is None:
        _log(state_dir, "shaping module unavailable — speaking unshaped")
        return pcm
    tiers = [bool(cfg["presence_eq"])]
    if tiers[0]:
        tiers.append(False)          # scipy missing: keep the pace, lose the EQ
    last: Exception | None = None
    for presence in tiers:
        try:
            shaper = mod.OutputShaper(
                sample_rate=int(cfg["sample_rate"]),
                speed=float(cfg["speed"]),
                presence_eq=presence,
                shelves=cfg["shelves"],
            )
            if not shaper.enabled:
                return pcm
            out = shaper.process(pcm) + shaper.flush()
            if presence != bool(cfg["presence_eq"]):
                _log(state_dir, f"shaping degraded to: {shaper.describe()}")
            return out
        except Exception as exc:
            last = exc
    _log(state_dir, f"shaping failed ({type(last).__name__}: {last}) — speaking unshaped")
    return pcm


def _with_tail(pcm: bytes, cfg: dict) -> bytes:
    ms = int(cfg.get("tail_silence_ms") or 0)
    if ms <= 0:
        return pcm
    return pcm + b"\x00" * (int(cfg["sample_rate"]) * ms // 1000 * 2)


# ── playback ──────────────────────────────────────────────────────────
def _player(path: str) -> list[str] | None:
    for cmd in (["paplay", path],
                ["pw-play", path],
                ["aplay", "-q", path],
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]):
        if shutil.which(cmd[0]):
            return cmd
    return None


def attention_tone(cfg: dict) -> bytes:
    """A short chirp built here, from the stdlib alone: mono s16le PCM (#1806).

    Nothing in this function opens a socket, which is the entire reason it
    exists. Every failure the clauses are about is a failure *of* the TTS
    endpoint — the 2026-09-28 refusals were reporting the outage of the same
    `agent-supervisord` tree that hosts `[program:agent-tts]` — so the degraded
    attempt cannot ask that endpoint for help. In-process *speech* would be the
    better thing and is not available on this box: `espeak`, `espeak-ng`,
    `spd-say`, `festival`, `flite` and `pico2wave` are all absent. The player is
    not: `_player` resolves `/usr/bin/pw-play` / `paplay`, separate binaries that
    answered throughout that outage.

    Two ascending notes with a 12 ms envelope each, `TONE_AMPLITUDE` of full
    scale. A steady tone is what a smoke alarm makes and a room learns to ignore
    it; the alert's words are on four other surfaces, so this only has to say
    "something is trying to reach you" without waking the house.
    """
    sr = max(1, int(cfg["sample_rate"]))
    amp = TONE_AMPLITUDE * 32767.0
    fade = max(1, int(0.012 * sr))
    per = max(1, int(TONE_SECONDS / len(TONE_HZ) * sr))
    out = array.array("h", bytes(2 * per * len(TONE_HZ)))
    i = 0
    for hz in TONE_HZ:
        for n in range(per):
            env = min(1.0, n / fade, (per - n) / fade)
            out[i] = int(amp * env * math.sin(2.0 * math.pi * hz * n / sr))
            i += 1
    return out.tobytes()


def _audible_fallback(cfg: dict, state_dir: Path) -> bool:
    """Try the speakers anyway, locally, when the synthesiser refused (#1806).

    Returns whether the *chirp* reached a player — deliberately not the same
    question as `speak_now`'s return value, which stays False because the words
    were not spoken. Bounded by `FALLBACK_PLAY_TIMEOUT` rather than by `play`'s
    normal budget (`audio + tail + 15 s of slack`), because this runs on the
    path a watchdog-adjacent process takes after the endpoint already failed and
    15 s of slack for a noise that lasts under one second is not a bound.
    """
    try:
        return play(attention_tone(cfg), cfg, timeout_cap=FALLBACK_PLAY_TIMEOUT)
    except Exception as exc:
        _log(state_dir, f"fallback chirp failed ({type(exc).__name__}: {exc})")
        return False


def play(pcm: bytes, cfg: dict, timeout_cap: float | None = None) -> bool:
    """Write a WAV and hand it to whatever player this box has.

    `timeout_cap` bounds the booking from above regardless of how long the audio
    is; it is how the fallback chirp stays inside its own short budget while the
    normal path keeps the ceiling-derived one (#1806).
    """
    sr = int(cfg["sample_rate"])
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="lloyd-guardian-")
    os.close(fd)
    try:
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm)
        cmd = _player(path)
        if cmd is None:
            return False
        # The player's timeout used to be `len(pcm) / (2 * sr) + 15.0`, which
        # scales with exactly the quantity a runaway decoder controls — so the
        # 31,467,360-byte artefact of 2026-09-28 booked 670.6 s of speakers for
        # itself. The ceiling bounds it from above; a short alert still gets a
        # short timeout, and only the tail it carries is added on top.
        spoken_s = min(_pcm_seconds(len(pcm), sr), float(cfg["max_audio_seconds"]))
        tail_s = int(cfg.get("tail_silence_ms") or 0) / 1000.0
        budget = spoken_s + tail_s + _PLAY_SLACK_S
        if timeout_cap is not None:
            budget = min(budget, float(timeout_cap))
        proc = subprocess.run(cmd, capture_output=True,
                              timeout=budget)
        return proc.returncode == 0
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ── the two entry points ──────────────────────────────────────────────
def _worker_python() -> str:
    """The lloyd venv interpreter when it is usable, otherwise our own.

    The detached worker is the one place the guardian's stdlib-only rule does
    not apply, and it is worth being explicit about why. That rule exists so
    the watchdog cannot be taken down by the thing it watches. This child runs
    *after* all five reliable channels have already fired, nothing waits on
    its exit, and its failure cannot reach the loop — so spending the venv
    here buys the presence EQ (scipy, which the system interpreter does not
    have) at no cost to the property the rule protects.

    A wrecked venv therefore costs a duller voice, never an alert.
    """
    venv = Path(VENV_PYTHON)
    try:
        if venv.is_file() and os.access(venv, os.X_OK):
            return str(venv)
    except OSError:
        pass
    return sys.executable


def _release_slot(key: str | None, state_dir: Path) -> None:
    """Give back the suppression slot a failed synthesis spent, and log whether that
    worked. Log-only on failure: the loss record is already written, and a suppression
    store we cannot write is also a suppression store that cannot suppress."""
    if not key:
        return
    if not forget_spoken(key, state_dir):
        _log(state_dir, f"could not release the speech slot for {key!r} after a synth "
                        f"failure — this alert stays suppressed for the repeat window")


def speak_now(text: str, cfg: dict, state_dir: Path,
              key: str | None = None) -> bool:
    """Synthesise and play, blocking. This is what the detached worker runs.

    Returns whether the *words* reached a speaker. Two ways to fail at that are
    recorded beyond `voice.log` and answered with a chirp (#1806): the endpoint
    refusing, and the endpoint answering with no audio — an HTTP 500 from a
    half-alive TTS server is the same dead silence from the room's side. The
    return value stays False in both cases: a chirp is not the alert, and the
    loss record is the report.

    `key` is the suppression slot `dispatch` reserved before spawning this worker.
    On both recorded-loss paths it is given back (#2256), because the reservation was
    made for an utterance that never existed: `dispatch` returns True — "dispatched" —
    and the alerting layer reasonably stops asking, so unless this process releases the
    slot, the identical alert is muted for `policy.VOICE_REPEAT_SECONDS` (one hour) from
    an attempt that produced no sound. A playback failure deliberately does NOT release
    it: the audio was made, something downstream lost it, and re-speaking a
    half-heard alert is a different bug than a dead synthesiser.
    """
    try:
        pcm = synthesize(text, cfg, state_dir)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        _log(state_dir, f"synth failed ({reason}) for {text[:60]!r}")
        _record_loss(state_dir, text, reason)
        _release_slot(key, state_dir)
        _audible_fallback(cfg, state_dir)
        return False
    if not pcm:
        _log(state_dir, f"synth returned nothing for {text[:60]!r}")
        _record_loss(state_dir, text, "synth returned no audio")
        _release_slot(key, state_dir)
        _audible_fallback(cfg, state_dir)
        return False
    try:
        audio = _with_tail(shape(pcm, cfg, state_dir), cfg)
        ok = play(audio, cfg)
        if not ok:
            _log(state_dir, "playback failed — no usable player, or it exited nonzero")
        return ok
    except Exception as exc:
        _log(state_dir, f"playback error ({type(exc).__name__}: {exc})")
        return False


def dispatch(level: str, title: str, body: str, state_dir: Path, *,
             window: float, key: str | None = None,
             verbatim: str | None = None, force: bool = False) -> bool:
    """Suppression-check, then hand off to a detached worker.

    Returns whether an utterance was *dispatched*. False means suppressed or
    un-spawnable, never "the speaker did not work" — nobody is waiting long
    enough to find that out. See `voice.log` for what happened next.

    `verbatim` skips `utterance_for` and hands the worker these exact words. Only
    #2264's replay uses it: the text is already composed, and the loss record's
    entire content is the words that were lost, so re-deriving them from a
    level/title/body triple would be a second opinion about the one thing that has
    to be reproduced byte for byte. `force` passes the suppression gate and still
    takes the stamp — see `should_speak`.

    Both new keywords default to the shipped behaviour, so every existing caller
    (`notify.py::_speak`, the only caller in production) is unchanged.
    """
    if not voice_enabled():
        return False
    try:
        state_dir = Path(state_dir)
        cfg = load_config(state_dir)
        # Checked BEFORE should_speak, which records as it decides. Recording a
        # quiet-hours drop would spend the hourly slot on an utterance nobody
        # heard, so the 08:00 repeat of a 03:00 alert would stay silent too.
        if in_quiet_hours(cfg) and not (
                level == "critical" and cfg["quiet_hours"].get("allow_critical")):
            _log(state_dir, f"quiet hours — withholding speech for {title!r}")
            return False
        # One expression for the slot, and it crosses the worker boundary with the text:
        # the process that discovers the synthesiser is dead is not the process that
        # reserved the hour, so without `--key` it cannot know what to give back (#2256).
        slot = key or title or "alert"
        if not should_speak(slot, state_dir, window, force=force):
            return False
        text = (verbatim if verbatim else
                utterance_for(level, title, body, max_chars=int(cfg["max_chars"])))
        subprocess.Popen(
            [_worker_python(), str(Path(__file__).resolve()),
             "--state-dir", str(state_dir), "--text", text, "--key", slot],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )
        return True
    except Exception as exc:
        _log(Path(state_dir), f"dispatch failed ({type(exc).__name__}: {exc})")
        return False


def main(argv: list[str] | None = None) -> int:
    import argparse
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser(description="Speak one line in Lloyd's voice.")
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--text", required=True)
    # Optional, and it has to stay optional: a worker invoked without it (a hand run, an
    # older caller) still speaks — it just cannot hand the slot back if the endpoint is
    # dead, which is #2256's bug rather than a reason to refuse the alert.
    ap.add_argument("--key", default=None,
                    help="suppression slot to release if synthesis fails")
    args = ap.parse_args(argv)
    state_dir = Path(args.state_dir)
    return 0 if speak_now(args.text, load_config(state_dir), state_dir,
                          key=args.key) else 1


if __name__ == "__main__":
    raise SystemExit(main())
