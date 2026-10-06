"""Where a spoken turn's time goes, one mark per seam.

"1.5-2.3 s from end of speech to first audio" was all anyone could say about a
turn, because the only timestamps in the worker's log were the inject and the
first clause. That number is a sum of nine stages, and every lever in the
latency plan moves a different one: a shorter VAD silence moves the first, a
parallel Smart Turn/ASR/embed the next three, prewarm the model's, an earlier
first clause the TTS request. So a turn carries a `TurnTimeline` and the worker
logs its breakdown as one `[latency]` line — pipecat's
`LatencyBreakdown.contributions`, in Lloyd's seams:

    speech_end      the speaker stopped (VAD close minus its closing silence)
    vad_close       the segmenter closed the utterance
    turn_verdict    Smart Turn answered
    asr_done        the transcript is in
    embed_done      the speaker embedding is in (wake / continuation only)
    inject_sent     POST /api/voice/inject left the worker
    voice_turn      the backend's first frame came back
    first_delta     the model's first text delta
    first_clause    the first clause went to TTS
    first_tts_byte  the TTS server's first PCM
    first_pushed    the first frame went into the LiveKit source
    first_played    that frame's estimated playout (push + audio already queued)

Marks are first-writer-wins, so a stage that happens many times (TTS bytes, a
frame per 100 ms) records only its first occurrence. In the logged line
`stage+ms` is the time since the previous stage and `stage@ms` (the three
parallel post-VAD stages) the time since the VAD close. A turn also records the
longest stretch of silence between audio it spoke (`max_gap_s`): the plan's
bound for a tool turn is 8 s, and the 2026-09-24 turn that prompted it was
silent for 44.

Two readings of those marks: `summary()` is the line the worker logs, and
`as_dict()` is the same numbers as data (#2273) — `app/voice_turns.py` appends
one such record per spoken turn to `<data root>/voice/turns.jsonl`, and
`scripts/maintenance/voice_turn_trend.py` is where they become percentiles.
Before that, every latency figure this project quoted was sampled from
`scripts/voice/e2e_voice.py`'s synthetic rig, because the log line was the only
sink and nothing could be sorted or filtered across turns.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

#: The order stages happen in, which is also the order the breakdown prints.
STAGES = (
    "speech_end", "vad_close", "turn_verdict", "asr_done", "embed_done",
    "inject_sent", "voice_turn", "first_delta", "first_clause",
    "first_tts_byte", "first_pushed", "first_played",
)
#: The stages that run at the same time after the VAD closes.
PARALLEL = frozenset({"turn_verdict", "asr_done", "embed_done"})


class TurnTimeline:
    """Monotonic marks for one spoken turn. Cheap; never raises."""

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.marks: dict[str, float] = {}
        #: Audio clock of the last frame this turn pushed: when it will have
        #: finished playing. Gaps are measured between that and the next push.
        self._audio_until: Optional[float] = None
        self.max_gap_s = 0.0
        self.gaps: list[float] = []

    def mark(self, stage: str, at: Optional[float] = None) -> None:
        if stage not in self.marks:
            self.marks[stage] = time.monotonic() if at is None else float(at)

    def has(self, stage: str) -> bool:
        return stage in self.marks

    def audio_pushed(self, push_at: float, play_from: float, duration_s: float) -> None:
        """A frame of this turn's audio went out; it plays from `play_from`."""
        if self._audio_until is not None and play_from > self._audio_until:
            gap = play_from - self._audio_until
            if gap >= 0.5:  # frame jitter and clause seams are not "silence"
                self.gaps.append(gap)
                self.max_gap_s = max(self.max_gap_s, gap)
        self._audio_until = max(self._audio_until or 0.0, play_from) + duration_s
        self.mark("first_pushed", push_at)
        self.mark("first_played", play_from)

    def contributions(self) -> list[tuple[str, float]]:
        """(stage, seconds since the previous recorded stage), in order.

        Smart Turn, the recogniser and the embedding run side by side, so each
        of those is measured from the VAD close, and the stage after them from
        whichever finished last — a sequential delta between two overlapping
        stages is a negative number that means nothing."""
        out: list[tuple[str, float]] = []
        prev: Optional[float] = None
        parallel_end: Optional[float] = None
        for stage in STAGES:
            t = self.marks.get(stage)
            if t is None:
                continue
            if stage in PARALLEL and "vad_close" in self.marks:
                out.append((stage, t - self.marks["vad_close"]))
                parallel_end = max(parallel_end or t, t)
                continue
            if parallel_end is not None:
                prev, parallel_end = max(prev or parallel_end, parallel_end), None
            if prev is not None:
                out.append((stage, t - prev))
            prev = t
        return out

    def total(self, start: str = "speech_end", end: str = "first_played") -> Optional[float]:
        a, b = self.marks.get(start), self.marks.get(end)
        if a is None or b is None:
            return None
        return b - a

    def summary(self) -> str:
        """The one-line breakdown the worker logs."""
        parts = [f"{stage}{'@' if stage in PARALLEL else '+'}{dt * 1000:.0f}"
                 for stage, dt in self.contributions()]
        total = self.total()
        head = f"eos→audio={total:.2f}s" if total is not None else "eos→audio=?"
        gap = f" max_gap={self.max_gap_s:.1f}s" if self._audio_until is not None else ""
        return f"{head}{gap} | " + " ".join(parts)

    def as_dict(self, *, epoch: Optional[float] = None) -> dict:
        """This turn as data, for the store `app/voice_turns.py` keeps (#2273).

        `summary()` is the human half of the same measurement and stays the log line
        `scripts/voice/e2e_voice.py` greps; this is the machine half, and the two are
        built from one set of marks so a table and a log line can never disagree.

          `stages`      one delta in seconds for every stage this turn reached, in
                        `STAGES` order, with the first recorded stage carrying its own
                        0.0. `contributions()` deliberately omits the origin for want
                        of an earlier mark to measure from; a record that omitted it
                        too would read as though `speech_end` never happened, which is
                        the survivorship half of the mistake the trend is built to
                        avoid.
          `eos_to_audio`  end of speech to first audible audio, or `None` when either
                        end is missing — a typed turn has no `speech_end`, a turn
                        cancelled before any audio has no `first_played`. `None` and
                        never 0.0: a zero there is the fastest turn on the box.
          `max_gap_s`   the longest silence inside the reply.
          `label`       `voice` or `typed:user`. Their stage sets differ, so the trend
                        splits on it rather than pooling them.
          `at`, `epoch` one instant spelled twice: ISO-8601 with the zone written on
                        it, and the numeric half for sorting and arithmetic. The marks
                        here are monotonic and cannot be joined to a session record, so
                        the record carries wall clock — and a naive local timestamp
                        would be read as UTC by every later machine reader, hence the
                        explicit `Z`. `epoch` omitted means now, which for the worker is
                        the moment the row is built.
        """
        if epoch is None:
            epoch = time.time()
        pairs = list(self.contributions())
        first = next((stage for stage in STAGES if stage in self.marks), None)
        if first is not None and all(stage != first for stage, _ in pairs):
            pairs.append((first, 0.0))
        order = {stage: i for i, stage in enumerate(STAGES)}
        return {
            "label": self.label,
            "at": datetime.fromtimestamp(epoch, timezone.utc).isoformat(
                timespec="milliseconds").replace("+00:00", "Z"),
            "epoch": epoch,
            "stages": {stage: delta for stage, delta in
                       sorted(pairs, key=lambda sd: order.get(sd[0], len(order)))},
            "eos_to_audio": self.total("speech_end", "first_played"),
            "max_gap_s": self.max_gap_s,
        }
