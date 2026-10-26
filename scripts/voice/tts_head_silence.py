#!/usr/bin/env python
"""How much of a synthesis is silence *before* the first audible sample? (#3366)

A byte does not equal audio. The provider case LiveKit measured was a dashboard
reporting 210 ms time-to-first-byte while shipping up to 3/4 s of silence behind it,
against a caller's budget of ~1.5 s from last syllable to first audible sound
(500-600 ms where it "really starts to feel human-like"). Lloyd's own latency number,
`eos_to_audio` (`voice/timeline.py`, one row per spoken turn since #2273), is
`first_played - speech_end`, and `first_played` is the playout of the first *frame* —
so silence sitting inside that audio is attributed to nothing, and every voice-latency
decision graded against it (`eval/lloyd_profile.md` standing problem 8, the owed
`EOS_TO_AUDIO_P90_BOUND_S`, #1163, #1166) is optimistic by exactly however much head
silence exists. Nobody had measured it for the local clone. This does.

    # production config, 44 built-in fixtures, clause + reply populations:
    .venvs/lloyd/bin/python scripts/voice/tts_head_silence.py

    # the same over your own lines, one reply per line, as data:
    .venvs/lloyd/bin/python scripts/voice/tts_head_silence.py \
        --fixture-file lines.txt --json --out ~/lloyd-data/voice/head-silence.jsonl

Needs the TTS service `livekit.tts.api_url` points at (production: `agent-tts-omni` on
:8093, GPU 3). It synthesises; it does not restart or reconfigure anything.

WHAT IS MEASURED. For each fixture the worker's own request body is POSTed to
`<api_url>/v1/audio/speech` (identical to `livekit_worker.py:641-656`, including
`stream_format: "audio"`, which is what stops vLLM-Omni answering SSE, and `speed: 1.0`,
because the server's streaming path drops speed and the worker stretches instead). The
response is decoded as s16le and head silence is measured twice over the *same* bytes:

    raw_ms      the server's PCM as it arrived — padding the vocoder produced
    shaped_ms   after `tts_shaping.OutputShaper` at `livekit.tts.speed` with the
                configured shelves — padding the client-side path produced, or removed

Reporting one column and not the other is what makes the number unactionable: the fix
for inherited padding is a server decode setting (`silence_ban_frames`), and the fix for
shaping-created padding is in `tts_shaping.py`. They are different files.

The onset is the first whole 10 ms window whose RMS exceeds -40 dBFS (0.01), the Coval
detector recorded in `knowledge/research/2026-10-18-whether-client-side-recv-buffer-sizes-and-tcp-coalescing-hid.md`.
Quantised down to the window grid, so it can under-report the true onset by less than one
window and never over-reports it; `None` when no window clears, never 0.0, because a
clip that never became audible is not the fastest result in the sample. Held for a whole
window on purpose: with a shorter window this is a peak detector, and the whole reason
there is no trim in the shaping path is that this floor reads a soft fricative onset
("s", "f", a nasal "n") as silence — see the outliers this run prints as `worst`.

WHAT MOVES BETWEEN RUNS, measured the same day. The command reproduces the measurement; it
does not reproduce the numbers, because Qwen3-TTS samples its tokens — the same text gets a
different head each call. Two runs of this identical 44-fixture set put a different raw head
on 71 of 88 rows: median absolute delta 20 ms, max 600 ms, and the maximum moved fixture
(one reply measured 130 ms then 650 ms; the 510 ms reply measured 20 ms the second time). So
p50 is the stable statistic and a p90/p95 from one run is a small-sample draw, not a
property — `--repeats k`, or several `--out` files merged, is how to tighten the tail.

    RUN A  2026-10-26T07:46:18Z                RUN B  2026-10-26T08:22Z
    first clause  raw    p50  20.0 p90  60.0   first clause  raw    p50  20.0 p90  97.0
                         p95  78.5 max 210.0                        p95 117.0 max 170.0
                  shaped p50  10.0 p90  50.0   shaped p50  10.0 p90  77.0 max 140.0
                         max 180.0                                p95  97.0
    whole reply   raw    p50  20.0 p90 117.0   whole reply   raw    p50  40.0 p90 248.0
                         p95 198.0 max 510.0                        p95 486.0 max 650.0
                  shaped p50  10.0 p90  94.0   shaped p50  30.0 p90 198.0 max 530.0
                         max 410.0                                p95 392.0

Both runs: 44 fixtures x (first clause + whole reply) = 88 syntheses, voice
`clone:dave_cullen` on :8093, shaping 1.22x WSOLA + presence EQ, n=44 and 0 unmeasurable in
every one of the eight columns. What survived the movement is the ruling and the one open
number: raw p50 is 20-40 ms either way, and the whole-reply raw p90 was past the ~100 ms
re-open line in both runs (117 ms, then 248 ms) — but that population is not what production
streams on a spoken turn, whose first clause measured p90 60 ms and 97 ms.

p50 20 ms is under the item's own ~60 ms no-code-change threshold, so the shipped answer
is no trim in the shaping path and no `first_audible` entry in `voice/timeline.py`'s
`STAGES`: at that size the mark would move the missing-stage denominator and
`tests/test_voice_turn_trend.py` / `tests/test_retention_sweep.py` to move `eos_to_audio`
by 0.6% of its 3.20 s median. The shaped column sits ~10 ms *below* raw almost everywhere,
because `WsolaStretch` reads its first frame at `buf[radius]` — 11 ms in — so the client
path removes head silence rather than creating it (the item's step-2 question, answered).
The long heads are soft onsets, not padding: the worst three all measure a non-zero
sub-floor peak in their first window (0.0004-0.0006 = a fricative or a breath), and a
trim to this floor would clip real speech to buy 20 ms. What this run does leave open is
the whole-reply raw p90 of 117 ms, past the ~100 ms line #3366 set for re-opening the
mark — that population is the non-streamed path (fillers, summaries, a typed turn spoken
whole), and re-opening is a decision for whoever next runs this, not for the round that
measured it. Written up in the "Measured in Lloyd" section of
`knowledge/youtube/AI_Engineer/20261026-how-to-build-voice-agents-that-dont-fall-apart-when-you-interrupt-jesse-hall-liv.md`.
Pinning: `tests/test_tts_head_silence.py`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "agent-services") not in sys.path:
    sys.path.insert(0, str(ROOT / "agent-services"))

import tts_shaping  # noqa: E402  (agent-services/tts_shaping.py, no LiveKit deps)
import tts_text  # noqa: E402    (the same spoken-form rewrite `speak()` applies)


def _load_voice_module(name: str):
    """Load `agent-services/voice/<name>.py` by path.

    Not `sys.path.insert(agent-services/voice)`: that directory also holds `turn.py`,
    `pipeline.py` and `asr.py`, which would shadow those names for every later import in
    this process. `voice/__init__.py` is LiveKit-free but the package would still pull
    the sibling stages in, and this script needs one function."""
    path = ROOT / "agent-services" / "voice" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_{name}_for_head_silence", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{path} cannot be loaded as a module")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


speakable = _load_voice_module("speakable")

#: RMS a 10 ms window must exceed to count as audible: -40 dBFS.
FLOOR_RMS = 0.01
#: How long it must be held for. A burst shorter than this is a transient, not audio.
WINDOW_MS = 10.0
#: Bytes per chunk fed to the shaper, the size `OpenAIRunner` reads. The shaper buffers
#: across chunks, so this only decides where the WSOLA search window happens to be.
CHUNK_BYTES = 4096

#: The transport. Tests swap this; production uses urllib.
OPENER: Callable = urllib.request.urlopen

#: 44 conversational replies of the shape Lloyd speaks, written for this measurement.
#: Deliberately varied at the part that matters — the onset. Soft fricative and nasal
#: openings ("Sure", "So", "Honestly", "Yeah", "Sixty"), short first clauses cut at a
#: comma, clauseless sentences, numbers and dates, and the two openings that produced
#: the longest heads in the 2026-10-26 triage, kept here as the regression case.
REPLIES = [
    "Sure, I've added the dentist appointment to your calendar for Thursday at three.",
    "The build finished cleanly, and all four hundred tests passed on the first run.",
    "Honestly? I'd wait until the morning, because the backup is still running.",
    "Your package from the hardware store should arrive sometime tomorrow afternoon.",
    "It's sixty two degrees outside right now, with light rain expected after six.",
    "I checked the logs, and the engine restarted twice overnight without losing a turn.",
    "Okay, the living room lights are off and the thermostat is set to sixty eight.",
    "There are three new emails, but only the one from the school looks urgent.",
    "Give me a second to look that up, it's buried in last week's notes.",
    "That should do it, let me know if the printer gives you any more trouble.",
    "I checked the GPU, and card three is the one that's been running hot.",
    "One moment while I read the file.",
    "So the short answer is yes, but only after the queue drains.",
    "Yes, the reboot took the wake word down with it, and that's on me.",
    "Yeah, that's the config change I mentioned, it landed at nine oh four.",
    "Fine, I'll leave the dry run going and tell you when it stops.",
    "Sixty percent of the rows came back empty, which is not a coincidence.",
    "Search done: eleven notes mention the supervisor conf, four of them recently.",
    "The wake word model is trained, and the retrain finished at oh six twenty.",
    "Nothing is on fire, but the retention sweep is due and hasn't run in a week.",
    "Two things changed since yesterday, and only one of them was deliberate.",
    "Let me put that on Thursday at nine, with a reminder the morning before.",
    "Hang on, the transcript is still coming through from the recorder.",
    "Right, the meeting moved to four, so I've shifted your walk to five.",
    "Here's the summary: three merged, two refused, one still sitting in review.",
    "Careful, that path doesn't exist any more, it moved under the data root.",
    "The TTS server answers in about a tenth of a second on the first clause.",
    "Which one do you mean, the nightly reflection or the morning brief?",
    "Mostly yes. The first two thirds are fine and the tail is guesswork.",
    "I can do that, but the gate would need the fixture set first.",
    "Reading it now, it's a long one, about forty pages plus the appendix.",
    "Not yet, the run is still going, and it started about four minutes ago.",
    "Understood, I'll stop the retry loop and leave the alarm standing.",
    "Back to your question about the embedding: it was timing out, not rejecting.",
    "Hmm, that number is from before the rebuild, so treat it as a rough one.",
    "Well, that explains the gap in the log, but it doesn't explain the crash.",
    "There you go, the calendar entry is created and the reminder is set.",
    "Just so you know, the second monitor is showing the last turn, not this one.",
    "Before you ask, no, I did not restart the engine, I only read the log.",
    "Ah, that explains it, the quiet hours window covers exactly that hour.",
    "Say the word and I'll file it, but I'd rather you saw the diff first.",
    "Oh, that's the fallback server on eight oh nine oh, not the production one.",
    "Three quarters of the way through, and nothing has failed so far.",
    "Anyway, the short version is that the probe works and the number is small.",
]


# ── the detector ────────────────────────────────────────────────────────────

def _samples(pcm) -> np.ndarray:
    """s16le little-endian bytes as float in [-1, 1); an ndarray is taken as already
    float samples, so a caller holding decoded audio can measure it without re-encoding."""
    if isinstance(pcm, np.ndarray):
        return pcm.astype(np.float64, copy=False)
    buf = bytes(pcm)
    buf = buf[: len(buf) - (len(buf) % 2)]
    return np.frombuffer(buf, dtype="<i2").astype(np.float64) / 32768.0


def onset_ms(pcm, sample_rate: int, *, floor: float = FLOOR_RMS,
             window_ms: float = WINDOW_MS) -> Optional[float]:
    """Milliseconds of audio before the first audible `window_ms` window, or None.

    A window clears when its RMS exceeds `floor` (default 0.01 = -40 dBFS). Windows are
    whole and non-overlapping on a grid from sample 0, so:

      * the answer is quantised down to the grid — it can under-report the true onset by
        less than one window, and never over-reports it, which is the direction that
        matters when the decision is whether to trim;
      * audio shorter than one window has no measurable head and returns None;
      * `None` means "no window ever cleared the floor", and is never 0.0 — 0.0 means
        the first window was already audio, the best result a clip can get.

    The floor is what makes a soft fricative onset read as silence, which is why the
    shipped answer to a long head here is "listen to it", not "cut to the floor".
    """
    x = _samples(pcm)
    width = max(1, int(round(sample_rate * window_ms / 1000.0)))
    usable = (len(x) // width) * width
    if usable == 0:
        return None
    blocks = x[:usable].reshape(-1, width)
    rms = np.sqrt(np.mean(blocks * blocks, axis=1))
    clearing = np.flatnonzero(rms > floor)
    if clearing.size == 0:
        return None
    return float(clearing[0]) * float(window_ms)


def peak(pcm) -> float:
    """Largest |sample|, so a "silent" head can be checked for sub-floor speech in it."""
    x = _samples(pcm)
    return float(np.max(np.abs(x))) if x.size else 0.0


# ── synthesis: the worker's own request ─────────────────────────────────────

class NonAudioResponse(RuntimeError):
    """The speech endpoint answered with something that is not waveform bytes, so
    decoding it as s16le would be measuring framing instead of audio."""


def speech_body(text: str, model: str, voice: str) -> dict:
    """The worker's body, field for field (`agent-services/livekit_worker.py:641-656`).

    `stream_format: "audio"` is what makes vLLM-Omni send raw s16le; without it the same
    request answers `text/event-stream` carrying base64 of it. `speed: 1.0` on the wire
    because the server's streaming path drops the field and the worker applies the
    stretch itself — asking in both places would stretch this probe's audio differently
    from production's."""
    return {"model": model, "input": text, "voice": voice,
            "response_format": "pcm", "stream": True, "stream_format": "audio",
            "speed": 1.0}


def synthesize(text: str, *, api_url: str, model: str, voice: str,
               opener: Optional[Callable] = None, timeout: float = 120.0) -> bytes:
    """One whole reply as raw s16le PCM bytes. The head is already in these bytes: the
    worker streams them, and streaming changes when a byte arrives, never what it is."""
    url = api_url.rstrip("/") + "/v1/audio/speech"
    req = urllib.request.Request(
        url, data=json.dumps(speech_body(text, model, voice)).encode(),
        headers={"Content-Type": "application/json"})
    with (opener or OPENER)(req, timeout=timeout) as resp:
        ct = (resp.headers.get("Content-Type") if resp.headers else "") or ""
        base = ct.split(";")[0].strip().lower()
        if base.startswith("text/"):
            # Refuse `text/*` rather than requiring an `audio/` prefix: the wrong answer
            # is text/event-stream and the right one is audio/pcm on this engine but
            # application/octet-stream on others. Measured in scripts/voice/tts_bakeoff.py:
            # 61921 SSE bytes decoded as s16le read rms -3.83 dBFS against the real PCM's
            # -19.1 — framing with a plausible head-silence number attached.
            raise NonAudioResponse(
                f"{url} answered Content-Type {ct!r}, not waveform: refusing to decode "
                f"it as s16le PCM (is `stream_format: audio` reaching the engine?)")
        return resp.read()


def shape(pcm: bytes, *, sample_rate: int, speed: float, presence_eq: bool = True,
          shelves: Optional[Sequence[dict]] = None) -> bytes:
    """The same bytes as the worker pushed them, through `OutputShaper`.

    This is the second column's whole purpose: `WsolaStretch` sets its first read
    pointer to `buf[radius]` with `radius` = 11 ms (`tts_shaping.py:142,206`) and
    `PresenceEQ.reset()` runs on zero filter state (`:107-112`), so the shaping path can
    move the head — the question this script exists to answer was whether it moves it
    later or earlier, and per fixture rather than in the median."""
    shaper = tts_shaping.OutputShaper(sample_rate, speed=speed,
                                     presence_eq=presence_eq, shelves=shelves)
    out = bytearray()
    for i in range(0, len(pcm), CHUNK_BYTES):
        out += shaper.process(pcm[i:i + CHUNK_BYTES])
    out += shaper.flush()
    return bytes(out)


# ── fixtures: what production would actually send ───────────────────────────

def first_clause(text: str, cfg: dict) -> str:
    """The first clause `livekit_worker` would synthesise for this reply.

    Default `stream` runs the production splitter — `voice.speakable.ClauseStream` at
    `livekit.voice_turn.first_clause_chars` (20), which is the rule and not a length: 20
    chars is when a soft boundary first counts, so the emitted clause is often longer.
    `truncate` is the 2026-10-26 triage probe's blunt `text[:20]`, kept because that is
    the number the item quotes and this script has to be able to reproduce it."""
    limit = int(cfg.get("first_clause_chars", 20))
    if cfg.get("clause_mode") == "truncate":
        return text[:limit]
    cs = speakable.ClauseStream(first_soft_chars=limit)
    out = cs.feed(text) + cs.flush()
    return out[0] if out else text


def _speech_form(text: str, cfg: dict) -> str:
    """`tts_text.for_speech`, as `speak()` applies it — to the clause, not the reply,
    which is production's order: the splitter sees the model's text and only the clause
    that came out of it is rewritten for the vocoder (`livekit_worker.py:328-329`)."""
    return tts_text.for_speech(text) if cfg.get("normalise_text", True) else text


def fixtures(texts: Iterable[str], cfg: dict) -> list[dict]:
    """Two items per reply: the first clause (what the listener's budget is spent on)
    and the whole reply (what a non-streamed utterance costs). `sent` is the exact
    string that will be POSTed; `text` is the fixture it came from."""
    items: list[dict] = []
    for raw in texts:
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        items.append({"kind": "clause", "text": text,
                      "sent": _speech_form(first_clause(text, cfg), cfg)})
        items.append({"kind": "reply", "text": text, "sent": _speech_form(text, cfg)})
    return items


def load_cfg(path: Optional[Path] = None) -> dict:
    """`livekit.tts` + the clause rule, with the worker's own defaults for any key
    config.yaml does not carry, so this cannot silently probe a different voice than
    production speaks."""
    cfg_yaml = yaml.safe_load((path or ROOT / "config.yaml").read_text()) or {}
    lk = cfg_yaml.get("livekit", {}) or {}
    tts = lk.get("tts", {}) or {}
    shaping = tts.get("shaping") or {}
    return {
        "api_url": tts.get("api_url") or "http://127.0.0.1:8093",
        "model": tts.get("model", "qwen3-tts"),
        "voice": tts.get("voice", "clone:cullen"),
        "sample_rate": int(tts.get("sample_rate", 24000)),
        "speed": float(tts.get("speed", 1.0)),
        "presence_eq": bool(shaping.get("presence_eq", True)),
        "shelves": shaping.get("shelves"),
        "first_clause_chars": int((lk.get("voice_turn") or {}).get("first_clause_chars", 20)),
        "normalise_text": bool(tts.get("normalise_text", False)),
    }


# ── measurement and report ──────────────────────────────────────────────────

def measure(items: Sequence[dict], cfg: dict, *, opener: Optional[Callable] = None,
            floor: float = FLOOR_RMS, window_ms: float = WINDOW_MS,
            on_row: Optional[Callable] = None) -> list[dict]:
    """Synthesise every item and report its head twice. Raises `NonAudioResponse` or
    `OSError` rather than returning a short list: an empty population printed as a
    population is how a broken run becomes a quoted percentile."""
    sr = cfg["sample_rate"]
    rows: list[dict] = []
    for it in items:
        raw = synthesize(it["sent"], api_url=cfg["api_url"], model=cfg["model"],
                         voice=cfg["voice"], opener=opener)
        shaped = shape(raw, sample_rate=sr, speed=cfg["speed"],
                       presence_eq=cfg["presence_eq"], shelves=cfg["shelves"])
        row = {
            "kind": it["kind"],
            "text": it["text"],
            "sent": it["sent"],
            "raw_head_ms": onset_ms(raw, sr, floor=floor, window_ms=window_ms),
            "shaped_head_ms": onset_ms(shaped, sr, floor=floor, window_ms=window_ms),
            # Length of what actually came back, so a truncated or empty response is
            # visible beside its head rather than masquerading as a short one.
            "raw_audio_ms": round(1000.0 * (len(raw) // 2) / sr, 1),
            "shaped_audio_ms": round(1000.0 * (len(shaped) // 2) / sr, 1),
            # Peak of the first window, which is what distinguishes padded silence from a
            # sub-floor soft onset in the `worst` list.
            "raw_head_peak": round(peak(raw[:int(sr * window_ms / 1000) * 2]), 4),
        }
        rows.append(row)
        if on_row is not None:
            on_row(row)
    return rows


def _percentile(values: Sequence[float], q: float) -> float:
    """Percentile of a NON-EMPTY sample. The empty case is the caller's to report as
    `None`: a percentile of no samples is not a number, and returning None here would
    put the empty check inside the arithmetic where a caller can forget it."""
    return float(np.percentile(np.asarray(values, dtype=float), q * 100.0))


def _column(rows: Sequence[dict], key: str, q: float = 0.9) -> dict:
    """One column's own sample. The n is of this column, not of the row set: a fixture
    whose shaped audio never cleared the floor belongs to raw_n and not to shaped_n."""
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))
            and not isinstance(r.get(key), bool)]
    return {"n": len(vals),
            "p50": None if not vals else round(_percentile(vals, 0.5), 1),
            "p90": None if not vals else round(_percentile(vals, q), 1),
            "p95": None if not vals else round(_percentile(vals, 0.95), 1),
            "max": None if not vals else round(max(vals), 1),
            "unmeasurable": len(rows) - len(vals)}


def summarise(rows: Sequence[dict]) -> dict:
    """{kind: {"raw": {...}, "shaped": {...}}} for every kind actually measured. A kind
    with no rows gets no block rather than a block of Nones, so a partial run cannot
    print a population that does not exist."""
    out: dict[str, dict] = {}
    for kind in ("clause", "reply"):
        krows = [r for r in rows if r.get("kind") == kind]
        if not krows:
            continue
        out[kind] = {"n_rows": len(krows),
                     "raw": _column(krows, "raw_head_ms"),
                     "shaped": _column(krows, "shaped_head_ms")}
    return out


def _ms(v) -> str:
    return "n/a" if v is None else f"{v:.1f}"


def render(rows: Sequence[dict], report: dict, cfg: dict, *, worst: int = 3,
           floor: float = FLOOR_RMS, window_ms: float = WINDOW_MS) -> str:
    lines = [
        f"TTS head silence — {cfg['api_url']} model={cfg['model']} voice={cfg['voice']}",
        f"shaping: {cfg['speed']:g}x WSOLA"
        + (f" + presence EQ ({len(cfg['shelves'])} shelves)" if cfg["presence_eq"] and cfg["shelves"]
           else (" + presence EQ" if cfg["presence_eq"] else " (off)")),
        f"onset: first {window_ms:g} ms window with RMS > {floor:g} "
        f"({20 * np.log10(floor):+.1f} dBFS); n/a = no window cleared it",
        "",
        f"{'kind':7} {'raw_ms':>8} {'shaped_ms':>10} {'audio_s':>8}  fixture",
    ]
    for r in rows:
        lines.append(f"{r['kind']:7} {_ms(r['raw_head_ms']):>8} "
                     f"{_ms(r['shaped_head_ms']):>10} "
                     f"{r['raw_audio_ms'] / 1000.0:>8.2f}  {r['text']}")
    for kind, block in report.items():
        lines += ["", f"{kind}: n_rows={block['n_rows']}"]
        lines.append(f"{'column':8} {'n':>4} {'p50':>8} {'p90':>8} {'p95':>8} {'max':>8} {'none':>5}")
        for name in ("raw", "shaped"):
            c = block[name]
            lines.append(f"{name:8} {c['n']:>4} {_ms(c['p50']):>8} {_ms(c['p90']):>8} "
                         f"{_ms(c['p95']):>8} {_ms(c['max']):>8} {c['unmeasurable']:>5}")
    worst_rows = sorted((r for r in rows if r["raw_head_ms"] is not None),
                        key=lambda r: -r["raw_head_ms"])[:worst]
    if worst_rows:
        lines += ["", f"worst {len(worst_rows)} raw heads — check these by ear before "
                      f"believing the floor:"]
        for r in worst_rows:
            lines.append(f"  {_ms(r['raw_head_ms']):>7} ms  peak in first "
                         f"{WINDOW_MS:g} ms = {r['raw_head_peak']:.4f}  "
                         f"[{r['kind']}] {r['text']}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Head silence of a synthesis: raw server PCM vs post-shaping.")
    ap.add_argument("--fixture-file", type=Path,
                    help="one reply per line (# comments allowed); replaces the built-ins")
    ap.add_argument("--replies", type=int, default=0,
                    help="use only the first N built-in fixtures (0 = all %d)" % len(REPLIES))
    ap.add_argument("--kinds", default="clause,reply",
                    help="comma list of clause,reply (default both, in one run)")
    ap.add_argument("--api-url")
    ap.add_argument("--model")
    ap.add_argument("--voice")
    ap.add_argument("--speed", type=float, help="override livekit.tts.speed (1.0 = no WSOLA)")
    ap.add_argument("--no-eq", action="store_true", help="skip the presence EQ")
    ap.add_argument("--clause-mode", choices=("stream", "truncate"), default="stream",
                    help="how the first clause is cut: production ClauseStream, or [:N]")
    ap.add_argument("--raw-text", action="store_true",
                    help="skip the for_speech spoken-form rewrite speak() applies")
    ap.add_argument("--floor", type=float, default=FLOOR_RMS)
    ap.add_argument("--window-ms", type=float, default=WINDOW_MS)
    ap.add_argument("--worst", type=int, default=3)
    ap.add_argument("--json", action="store_true", help="machine-readable instead of the table")
    ap.add_argument("--out", type=Path, help="append one JSON line per run to this file")
    args = ap.parse_args(argv)

    cfg = load_cfg()
    if args.api_url:
        cfg["api_url"] = args.api_url
    if args.model:
        cfg["model"] = args.model
    if args.voice:
        cfg["voice"] = args.voice
    if args.speed is not None:
        cfg["speed"] = args.speed
    if args.no_eq:
        cfg["presence_eq"] = False
    if args.raw_text:
        cfg["normalise_text"] = False
    cfg["clause_mode"] = args.clause_mode

    if args.fixture_file:
        texts = args.fixture_file.read_text().splitlines()
    else:
        texts = REPLIES[:args.replies] if args.replies else REPLIES
    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    unknown = kinds - {"clause", "reply"}
    if unknown:
        print(f"error: unknown --kinds {sorted(unknown)}, expected clause and/or reply",
              file=sys.stderr)
        return 2
    items = [i for i in fixtures(texts, cfg) if i["kind"] in kinds]
    if not items:
        print("error: no fixtures to synthesise", file=sys.stderr)
        return 2

    try:
        rows = measure(items, cfg, floor=args.floor, window_ms=args.window_ms)
    except NonAudioResponse as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (urllib.error.URLError, OSError) as exc:   # URLError is an OSError
        print(f"error: {cfg['api_url']}: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print("error: measured 0 fixtures, which is not a result", file=sys.stderr)
        return 2

    report = summarise(rows)
    payload = {"at": __import__("datetime").datetime.now(
                   __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
               "cfg": {k: cfg[k] for k in ("api_url", "model", "voice", "sample_rate",
                                           "speed", "presence_eq", "first_clause_chars",
                                           "clause_mode", "normalise_text")},
               "floor_rms": args.floor, "window_ms": args.window_ms,
               "rows": rows, "summary": report}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(rows, report, cfg, worst=args.worst, floor=args.floor,
                     window_ms=args.window_ms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
