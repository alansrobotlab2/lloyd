"""Was that said TO Lloyd? The question an open conversation has to ask.

The wake word answers it by construction: a sentence that starts with "hey
Lloyd" is addressed. Inside a conversation there is no wake word, and every
shipped assistant that drops one gates on *directedness*, not on a timer —
Alexa's follow-up mode declines when it is not confident the speech was meant
for it, Echo Show's conversation mode fuses head pose with an audio
device-directedness model. No open device-directed-speech classifier exists, so
this asks djev (`app/djev.py`, ~40 ms on GPU 2): one `noul` question over the
utterance, Lloyd's last sentence, how long ago he said it, and who is talking.
Apple's work on device-directed speech found the prior turn is the context that
matters most (20-40% fewer false accepts at a 10% false-reject rate).

Two modes, `livekit.conversation.addressee`:

  shadow   ask, log the verdict beside the gate's own decision, change nothing;
  enforce  inside an open conversation but past the short follow-up window,
           an utterance reaches the model only when djev says it is addressed.

It never gates the short follow-up window (`wake.continuation_seconds` after
Lloyd stops speaking or the user's last turn) — that path is today's proven
behaviour and stays exactly as it was. What enforce adds is the rest of the
conversation, where today everything is dropped. So a djev that is down, slow
or wrong can only ever cost what the conversation mode added, never what the
wake word already had: an unreachable engine answers `None`, and `None` is a
drop there, which is what the gate did before this existed.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

LOG = logging.getLogger("lloyd-agent-worker.addressee")

_REPO = Path(__file__).resolve().parents[2]

#: The question, frozen: djev's thresholds belong to one schema, option order
#: and wording included (architecture/djev.md), so changing a word here is a
#: re-measurement, not an edit.
QUESTION = {
    "addressed": {
        "type": "noul",
        "instructions": (
            "Lloyd is a voice assistant in a room. Is the SPEAKER'S LATEST "
            "UTTERANCE directed at Lloyd — a request, question, answer or reply "
            "meant for the assistant to act on or respond to — rather than "
            "speech to another person, to a phone, to themselves, background "
            "audio, or a mere acknowledgement like 'yeah' or 'okay'?"
        ),
    },
}


@dataclass(frozen=True)
class Verdict:
    addressed: bool
    probability: float
    latency_ms: float
    label_mass: float = 0.0


def state_text(utterance: str, last_agent: str, since_agent_s: Optional[float],
               speaker: Optional[str], mentions_name: bool) -> str:
    since = ("never" if since_agent_s is None
             else f"{since_agent_s:.0f} seconds ago")
    return (
        f"Lloyd last said ({since}): \"{(last_agent or '(nothing yet)')[-300:]}\"\n"
        f"Speaker: {speaker or 'the person Lloyd is talking with'}\n"
        f"Speaker's latest utterance: \"{utterance[:300]}\"\n"
        f"The utterance {'mentions' if mentions_name else 'does not mention'} "
        f"the name Lloyd."
    )


class AddresseeClassifier:
    def __init__(self, cfg: dict) -> None:
        self.mode = str(cfg.get("addressee", "shadow")).lower()
        self.threshold = float(cfg.get("addressee_threshold", 0.5))
        self.timeout_s = float(cfg.get("addressee_timeout_ms", 1500)) / 1000.0
        self._djev = None
        self._import_failed = False

    @property
    def enforcing(self) -> bool:
        return self.mode == "enforce"

    @property
    def active(self) -> bool:
        return self.mode in ("shadow", "enforce")

    def _client(self):
        if self._djev is None and not self._import_failed:
            try:
                if str(_REPO) not in sys.path:
                    sys.path.append(str(_REPO))
                from app import djev
                self._djev = djev
            except Exception as e:  # noqa: BLE001
                self._import_failed = True
                LOG.warning("addressee: djev client unavailable (%s)", e)
        return self._djev

    async def judge(self, utterance: str, last_agent: str,
                    since_agent_s: Optional[float], speaker: Optional[str],
                    mentions_name: bool) -> Optional[Verdict]:
        """djev's verdict, or None when it could not be asked. Never raises."""
        if not self.active or not utterance.strip():
            return None
        djev = self._client()
        if djev is None:
            return None
        t0 = time.monotonic()
        try:
            out = await asyncio.wait_for(
                djev.ask(state_text(utterance, last_agent, since_agent_s,
                                    speaker, mentions_name),
                         QUESTION, seam="voice:addressee",
                         timeout=self.timeout_s),
                timeout=self.timeout_s + 0.2)
        except Exception as e:  # noqa: BLE001
            LOG.debug("addressee: djev failed: %s", e)
            return None
        if out is None:
            return None
        ans = out.get("addressed")
        if ans is None:
            return None
        p = float(ans.value)
        return Verdict(p >= self.threshold, p, (time.monotonic() - t0) * 1000,
                       float(ans.label_mass))
