"""The one client for djev's structured-decision server.

djev is DiffusionGemma 26B-A4B NVFP4 on GPU 2 (`github.com/mmastrac/djev-spark`),
serving `POST :8011/v1/systemone`: typed questions — yes/no (`noul`), one-of-N
(`choice`), ordered scale (`score`) — read off one diffusion canvas in ~40 ms
plus ~0.25 ms per prompt token. `architecture/djev.md` is the long version.

Stdlib at its core, on `urllib.request`, because the callers are hot paths and
leaves: `agent_mcp/backlog_similar.py` sits on the `backlog_write_task` write
path, `agent_mcp/vault.py` on every recall, and a sweep script runs outside the
backend entirely. `app/backlog_status.py` and `app/qmd_health.py` are the
precedent. httpx appears only behind the function-local import in `ask()`, the
`agent_mcp/_shared.py::make_http_client` trick, so importing this module costs
nothing.

WHAT THIS CLIENT IS FOR, AND WHAT IT IS NOT FOR
-----------------------------------------------
Measured on 2026-09-20 over 40 labelled pairs, four framings of one question:
AUC held at 0.72-0.83 across every framing, so the ORDERING djev produces is a
real signal. But merely reversing the option order moved P by 0.324 on average
(max 0.851), and each framing had a different optimal threshold — 0.39, 0.03,
0.30, 0.01. **A fixed 0.5 cutoff is meaningless.** Ranking, ordering and
shortlisting are safe; a yes/no gate ships with a threshold calibrated on its
own frozen schema or it does not ship.

So this module returns values and two trust flags and decides nothing. There
is no `is_same()` here and there must not be one: a helper that answers a
yes/no from a hardcoded cutoff is the exact mistake the measurement rules out.

THREE-VALUED, LIKE `semantic_candidates`
----------------------------------------
`None` on ANY failure — unreachable, non-200, malformed, slot switched off.
Never `[]`, never an exception. `backlog_similar.semantic_candidates` has the
same contract for the same reason: a caller must be able to tell "djev had no
opinion" from "djev did not answer", because only the first is evidence. Every
caller in this tree treats `None` as "carry on unchanged".

THE TRAPS THIS NORMALIZES
-------------------------
The server's `jev_answer` returns three different shapes, and two of them lose
something on the way out:

* `noul` returns `{"type": "noul", "noul": p}` and NOTHING else — no
  `confidence`, no `probabilities`. A caller that reads `confidence` off a
  mixed answer set gets `None` for exactly the yes/no questions and,
  unguarded, treats an absent confidence as zero.
* `score` returns BOTH `score` (the 0-based expected value over the levels)
  and `confidence` (the modal probability). They are different numbers and
  they answer different questions; collapsing them is how "how severe is this"
  becomes "how sure are you".
* `label_mass` and `argmax_is_label` — the only honest confidence signals the
  server produces — are buried in `diagnostics.questions.<id>`, one nesting
  level away from the answers, so the natural read of the response misses
  them entirely.

`label_mass` is how much of the model's probability actually landed on legal
label tokens. The returned `probabilities` are renormalized over the label set
regardless, so they still sum to 1 and still look like confident scores while
92-98% of the mass sat somewhere else. Measured listwise: 1.000 at 8
candidates, 0.965 at 16, 0.873 at 32, 0.303 at 48, 0.005 at 64. Every answer
here carries it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

logger = logging.getLogger("lloyd-djev")

#: Where the structured server answers when config.yaml cannot be read. The
#: same literal as `djev.structured_url`, not a second opinion about it: a
#: config import that fails must cost the config, never the client.
DEFAULT_STRUCTURED_URL = "http://127.0.0.1:8011"

#: Client-side bound, deliberately far under the server's own 600 s upstream
#: read. There is no 504 on that path — a hung read surfaces as a 500 carrying
#: a `repr()` — so the only thing standing between a wedged engine and a
#: blocked caller is this number. djev is `--max-num-seqs 1`, so a queue in
#: front of a call is real and 15 s leaves room for one cold 20k-token state
#: (6.0 s measured) plus a neighbour ahead of it.
DEFAULT_TIMEOUT_S = 15.0

#: `djev_rank`'s default and hard ceiling, and the reason both numbers exist.
#: Listwise `label_mass` measured 1.000 at 8 and 0.965 at 16 in one run, 0.807
#: and 0.446 at 16 in two earlier ones — 16 is the EDGE of the safe window,
#: not its middle. Above 32 questions the server splits the canvas into
#: separate shared contexts (measured: 48 -> chunks of 33 and 15, 64 -> 33, 30
#: and 1) and upstream states plainly that a partitioned listwise score is not
#: comparable across chunks. The lone one-item chunk at n=64 scored its
#: candidate against nothing and ranked it first.
RANK_DEFAULT_N = 12
RANK_MAX_N = 16

#: The canvas split. A ranking that crosses it is not a ranking.
CANVAS_CHUNK_QUESTIONS = 32

#: Recent latencies kept for `djev_status`, per seam. A ring, not a counter:
#: the question a status route is asked is "is it answering, and how fast
#: lately", which a lifetime mean stops being able to answer after a day.
_LATENCY_RING = 64
_lock = threading.Lock()
_latency: dict[str, deque] = {}
_calls: dict[str, dict[str, int]] = {}


class DjevUnavailable(Exception):
    """Raised only inside this module. Callers see `None`."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def structured_url() -> str:
    """`djev.structured_url` from config.yaml, or the hard default.

    Lazy import: `app.config` pulls in the whole config surface, and this
    module is imported by aggregator leaves that must stay cheap.
    """
    try:
        from app.config import CONFIG
        url = ((CONFIG or {}).get("djev") or {}).get("structured_url")
        return str(url).rstrip("/") if url else DEFAULT_STRUCTURED_URL
    except Exception:  # noqa: BLE001 — fail open to the default
        return DEFAULT_STRUCTURED_URL


def enabled() -> bool:
    """Is the djev slot switched on?

    Through `app.llm_slots.is_enabled`, which is the ONE definition of "should
    this program be running" and already reads `djev.enabled`. A second read
    of the same flag here is how six surfaces came to disagree about the
    secondary — see that module's docstring.
    """
    try:
        from app.llm_slots import is_enabled
        return bool(is_enabled("agent-djev"))
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Answer:
    """One question's answer, with the server's shape differences ironed out.

    `value` is the number or name you asked for, and it means something
    different per type — P(yes) for `noul`, the chosen option's name for
    `choice`, the 0-based expected level for `score`. `confidence` is always
    the modal probability, which for `score` is a DIFFERENT number from
    `value` and is not interchangeable with it.
    """

    id: str
    type: str
    value: Any
    label: str
    confidence: float
    probabilities: dict[str, float]
    label_mass: float
    argmax_is_label: bool
    #: `label_mass` fell below this schema's measured floor. False whenever no
    #: floor is set, which is the starting state for every schema — see
    #: `eval/djev/schemas.py`.
    low_trust: bool = False
    #: Every answer in this set carried the same value. A floor cannot catch
    #: that shape: the n=4 listwise run returned all 0.0 with `label_mass`
    #: 0.987, so the mass was legal and the answer was empty.
    uninformative: bool = False
    #: The level names for a `score`, by their 0-based index.
    legend: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "id": self.id, "type": self.type, "value": self.value,
            "label": self.label, "confidence": self.confidence,
            "probabilities": self.probabilities, "label_mass": self.label_mass,
            "argmax_is_label": self.argmax_is_label,
            "low_trust": self.low_trust, "uninformative": self.uninformative,
            **({"legend": self.legend} if self.legend else {}),
        }


@dataclass(frozen=True)
class Answers:
    """One decision: the answers, and what the server said about making them."""

    answers: dict[str, Answer]
    latency_ms: float
    server_ms: float
    prompt_tokens: int
    #: The server's own canvas split. More than one chunk means the answers
    #: were produced in separate shared contexts and MUST NOT be sorted
    #: against each other.
    chunks: list[list[str]]
    #: Every answer carried the same value — see `Answer.uninformative`.
    uninformative: bool
    #: A floor was configured for this call at all. `low_trust` is always
    #: False without one, and "no answer was low-trust" and "nothing has been
    #: calibrated yet" must not read the same.
    floor: float | None
    seam: str = ""

    def __len__(self) -> int:
        return len(self.answers)

    def __iter__(self):
        return iter(self.answers.values())

    def __getitem__(self, key: str) -> Answer:
        return self.answers[key]

    def get(self, key: str, default=None):
        return self.answers.get(key, default)

    @property
    def cross_chunk(self) -> bool:
        """Did the server split this decision across canvas chunks?"""
        return len(self.chunks) > 1

    @property
    def min_label_mass(self) -> float:
        return min((a.label_mass for a in self.answers.values()), default=0.0)

    def as_dict(self) -> dict:
        return {
            "answers": {k: a.as_dict() for k, a in self.answers.items()},
            "latency_ms": round(self.latency_ms, 1),
            "server_ms": round(self.server_ms, 1),
            "prompt_tokens": self.prompt_tokens,
            "chunks": self.chunks,
            "cross_chunk": self.cross_chunk,
            "uninformative": self.uninformative,
            "min_label_mass": round(self.min_label_mass, 4),
            "floor": self.floor,
            "seam": self.seam,
        }


def _normalize(qid: str, raw: Mapping | None, diag: Mapping) -> Answer | None:
    """One server answer plus its diagnostics row into an `Answer`.

    `None` for a question the server skipped through `ask_if`, which it
    reports as a null answer rather than as an omission.
    """
    if not isinstance(raw, Mapping):
        return None
    kind = str(raw.get("type") or "")
    mass = float(diag.get("label_mass", 0.0) or 0.0)
    argmax_ok = bool(diag.get("argmax_is_label", False))
    if kind == "noul":
        # The server sends P(yes) alone here. Filling the other two fields is
        # what lets a caller read `confidence` and `probabilities` off a mixed
        # answer set without branching on the type first.
        p = float(raw.get("noul", 0.0) or 0.0)
        return Answer(id=qid, type="noul", value=p,
                      label="yes" if p >= 0.5 else "no",
                      confidence=max(p, 1.0 - p),
                      probabilities={"yes": p, "no": 1.0 - p},
                      label_mass=mass, argmax_is_label=argmax_ok)
    probs = {str(k): float(v) for k, v in (raw.get("probabilities") or {}).items()}
    conf = float(raw.get("confidence", 0.0) or 0.0)
    if kind == "choice":
        name = str(raw.get("choice") or "")
        return Answer(id=qid, type="choice", value=name, label=name,
                      confidence=conf, probabilities=probs,
                      label_mass=mass, argmax_is_label=argmax_ok)
    if kind == "score":
        legend = {str(k): str(v) for k, v in (raw.get("legend") or {}).items()}
        # `score` is the expected level index and `confidence` is the modal
        # probability. Keeping both is the point; see the class docstring.
        expected = float(raw.get("score", 0.0) or 0.0)
        modal = max(probs, key=probs.get) if probs else ""
        return Answer(id=qid, type="score", value=expected,
                      label=legend.get(modal, modal), confidence=conf,
                      probabilities=probs, label_mass=mass,
                      argmax_is_label=argmax_ok, legend=legend)
    return None


def _build(payload: Mapping, latency_ms: float, floor: float | None,
           seam: str) -> Answers:
    diag = payload.get("diagnostics") or {}
    per_q = diag.get("questions") or {}
    built: dict[str, Answer] = {}
    for qid, raw in (payload.get("answers") or {}).items():
        ans = _normalize(str(qid), raw, per_q.get(qid) or {})
        if ans is not None:
            built[str(qid)] = ans

    # "Every candidate got the same score" is a property of the SET, and it is
    # the failure a floor structurally cannot see. Only meaningful over a
    # ranking — two or more answers of one type — so a single question and a
    # mixed ticket schema are never flagged.
    values = [a.value for a in built.values()]
    types = {a.type for a in built.values()}
    uninformative = (
        len(built) >= 2 and len(types) == 1
        and len({v if isinstance(v, str) else round(float(v), 6) for v in values}) == 1
    )

    if floor is not None or uninformative:
        built = {
            k: Answer(**{**a.as_dict(), "legend": a.legend,
                         "low_trust": floor is not None and a.label_mass < floor,
                         "uninformative": uninformative})
            for k, a in built.items()
        }

    timing = diag.get("timing") or {}
    usage = payload.get("usage") or {}
    return Answers(
        answers=built,
        latency_ms=latency_ms,
        server_ms=float(timing.get("total_ms", 0.0) or 0.0),
        prompt_tokens=int(usage.get("input_tokens") or 0),
        chunks=[list(c) for c in (diag.get("chunks") or [])],
        uninformative=uninformative,
        floor=floor,
        seam=seam,
    )


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------

def _body(state: Any, questions: Mapping, *, samples: int | None,
          instructions: str | None, seed: int | None) -> dict:
    body: dict[str, Any] = {"model": "djev", "state": state,
                            "questions": dict(questions)}
    if samples is not None:
        body["samples"] = samples
    if instructions:
        body["instructions"] = instructions
    if seed is not None:
        body["seed"] = int(seed)
    return body


def _record(seam: str, outcome: str, latency_ms: float | None) -> None:
    with _lock:
        counts = _calls.setdefault(seam or "-", {})
        counts[outcome] = counts.get(outcome, 0) + 1
        if latency_ms is not None:
            ring = _latency.setdefault(seam or "-", deque(maxlen=_LATENCY_RING))
            ring.append(latency_ms)


# ---------------------------------------------------------------------------
# Replay: one answer per request inside one comparison (eval arms only)
# ---------------------------------------------------------------------------
#
# djev does not give the same answer twice. Measured 2026-09-21 on a 32-row
# recall rank replayed straight at vLLM: the argmax at all 123 canvas positions
# was identical on every run, while the label logprobs the rank score is built
# from moved by 1-3 nats between identical requests. Not the prefix cache (a
# fresh `cache_salt` per request varies as much), not CUDA graphs or compile
# (`--enforce-eager` varies), not async scheduling, not the seed canvas (applied
# identically) and not sampling (the read-only path draws nothing): the kernels.
#
# That is harmless for a recall and fatal for a paired eval. The regression
# check runs the same questions through two arms and reads any difference as
# the change; with djev in the loop two arms running identical code differed by
# a query, and two promotions that touched no retrieval code were rolled back
# for it. So inside one comparison an identical request is answered once: each
# arm names itself, one arm is the anchor, and a request the anchor already
# asked is replayed from its answer. A request only this arm asks (the change
# moved the ranker's input) is a fresh draw, and counted as one, so the caller
# knows the ranker's own noise is in the comparison. Failures are counted and
# never cached. Unset in production; `LLOYD_DJEV_REPLAY` is the eval's switch.

REPLAY_ENV = "LLOYD_DJEV_REPLAY"            # sqlite file shared by one comparison's arms
REPLAY_ARM_ENV = "LLOYD_DJEV_REPLAY_ARM"    # this arm's name
REPLAY_ANCHOR_ENV = "LLOYD_DJEV_REPLAY_ANCHOR"  # the arm whose answers every arm reuses

#: Outcomes that mean djev did not rank: the recall fell back to the
#: cross-encoder, so the arm measured a different ranker for a reason that is
#: not the code under test.
REPLAY_FAILURES = ("unreachable", "http_5xx", "malformed")


def replay_env(path: Any, arm: str, anchor: str = "baseline") -> dict[str, str]:
    """The environment that puts one eval arm under replay."""
    return {REPLAY_ENV: str(path), REPLAY_ARM_ENV: arm, REPLAY_ANCHOR_ENV: anchor}


def _replay_conf() -> tuple[str, str, str] | None:
    path = os.environ.get(REPLAY_ENV)
    if not path:
        return None
    arm = os.environ.get(REPLAY_ARM_ENV) or "arm"
    return path, arm, os.environ.get(REPLAY_ANCHOR_ENV) or arm


def _replay_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT, arm TEXT, payload TEXT, "
                 "PRIMARY KEY (key, arm))")
    conn.execute("CREATE TABLE IF NOT EXISTS outcomes (arm TEXT, outcome TEXT, n INTEGER, "
                 "PRIMARY KEY (arm, outcome))")
    return conn


def _replay_key(url_path: str, data: bytes) -> str:
    return hashlib.sha256(url_path.encode() + b"\0" + data).hexdigest()


def _replay_lookup(conf: tuple[str, str, str], key: str) -> tuple[dict, str] | None:
    path, arm, anchor = conf
    try:
        with _replay_db(path) as conn:
            for who, source in ((anchor, "replayed_anchor"), (arm, "replayed_own")):
                row = conn.execute("SELECT payload FROM answers WHERE key=? AND arm=?",
                                   (key, who)).fetchone()
                if row:
                    return json.loads(row[0]), source
    except Exception as exc:  # noqa: BLE001 — replay is an eval aid, never a failure
        logger.warning("djev replay lookup failed: %s", exc)
    return None


def _replay_note(conf: tuple[str, str, str], outcome: str, key: str | None = None,
                 payload: dict | None = None) -> None:
    path, arm, _anchor = conf
    try:
        with _replay_db(path) as conn:
            if key is not None and payload is not None:
                conn.execute("INSERT OR REPLACE INTO answers VALUES (?, ?, ?)",
                             (key, arm, json.dumps(payload)))
            conn.execute("INSERT INTO outcomes VALUES (?, ?, 1) ON CONFLICT (arm, outcome) "
                         "DO UPDATE SET n = n + 1", (arm, outcome))
    except Exception as exc:  # noqa: BLE001
        logger.warning("djev replay write failed: %s", exc)


def replay_stats(path: Any) -> dict[str, dict[str, int]]:
    """`{arm: {outcome: n}}` for one comparison's replay file; `{}` if absent.

    `fresh` is a request djev answered for this arm and no earlier arm asked;
    `replayed_anchor` / `replayed_own` were answered from the file; the
    `REPLAY_FAILURES` outcomes are requests djev did not answer. An arm with no
    row at all ran code that predates replay, and says nothing either way.
    """
    out: dict[str, dict[str, int]] = {}
    try:
        if not os.path.exists(str(path)):
            return out
        with _replay_db(str(path)) as conn:
            for arm, outcome, n in conn.execute("SELECT arm, outcome, n FROM outcomes"):
                out.setdefault(arm, {})[outcome] = int(n)
    except Exception as exc:  # noqa: BLE001
        logger.warning("djev replay stats unreadable: %s", exc)
    return out


def ask_sync(state: Any, questions: Mapping[str, Mapping], *,
             timeout: float = DEFAULT_TIMEOUT_S, seam: str = "",
             floor: float | None = None, samples: int | None = None,
             instructions: str | None = None,
             seed: int | None = None) -> Answers | None:
    """One structured decision. `None` on any failure at all.

    `floor` is this schema's measured `label_mass` floor, from
    `eval/djev/schemas.py`. Leaving it `None` is the honest state for a schema
    nothing has calibrated yet: every answer still carries its `label_mass`,
    and `low_trust` stays False rather than being decided by a number nobody
    measured. A fixed 0.5 would trip on real requests inside the window this
    design calls safe — the two earlier n=16 runs read 0.446 and 0.807.
    """
    if not questions:
        return None
    if not enabled():
        _record(seam, "disabled", None)
        return None
    url = structured_url() + "/v1/systemone"
    data = json.dumps(_body(state, questions, samples=samples,
                            instructions=instructions, seed=seed)).encode()
    replay = _replay_conf()
    key = _replay_key("/v1/systemone", data) if replay else None
    if replay:
        hit = _replay_lookup(replay, key)
        if hit is not None:
            payload, source = hit
            _replay_note(replay, source)
            try:
                out = _build(payload, 0.0, floor, seam)
            except Exception:  # noqa: BLE001 — only a payload that built once is stored
                return None
            _record(seam, "replayed", 0.0)
            return out if out.answers else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        # 422 is a schema this client built wrong and is worth seeing; 500 is
        # the shape a hung upstream read takes, since there is no 504 here.
        _record(seam, "http_error", None)
        if replay:
            _replay_note(replay, "http_5xx" if exc.code >= 500 else "http_4xx")
        logger.debug("djev %s: HTTP %s", seam or "-", exc.code)
        return None
    except Exception as exc:  # noqa: BLE001 — every failure is `None`
        _record(seam, "unreachable", None)
        if replay:
            _replay_note(replay, "unreachable")
        logger.debug("djev %s: %s", seam or "-", exc)
        return None
    latency_ms = (time.perf_counter() - t0) * 1e3
    try:
        out = _build(payload, latency_ms, floor, seam)
    except Exception as exc:  # noqa: BLE001 — a malformed body is a failure
        _record(seam, "malformed", None)
        if replay:
            _replay_note(replay, "malformed")
        logger.debug("djev %s: malformed response: %s", seam or "-", exc)
        return None
    if replay:
        # Stored even when it holds no answers: `empty` is djev's answer to
        # this input, not a failure to give one.
        _replay_note(replay, "fresh", key, payload)
    if not out.answers:
        _record(seam, "empty", latency_ms)
        return None
    _record(seam, "ok", latency_ms)
    return out


async def ask(state: Any, questions: Mapping[str, Mapping], *,
              timeout: float = DEFAULT_TIMEOUT_S, seam: str = "",
              floor: float | None = None, samples: int | None = None,
              instructions: str | None = None,
              seed: int | None = None) -> Answers | None:
    """The async twin, for callers already on the event loop.

    httpx behind a function-local import so this module stays a stdlib leaf.
    A tool handler that reached for `ask_sync` on the loop would block it for
    as long as the engine takes, and djev serves one sequence at a time.
    """
    if not questions:
        return None
    if not enabled():
        _record(seam, "disabled", None)
        return None
    url = structured_url() + "/v1/systemone"
    body = _body(state, questions, samples=samples, instructions=instructions,
                 seed=seed)
    t0 = time.perf_counter()
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=body)
        if resp.status_code != 200:
            _record(seam, "http_error", None)
            logger.debug("djev %s: HTTP %s", seam or "-", resp.status_code)
            return None
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        _record(seam, "unreachable", None)
        logger.debug("djev %s: %s", seam or "-", exc)
        return None
    latency_ms = (time.perf_counter() - t0) * 1e3
    try:
        out = _build(payload, latency_ms, floor, seam)
    except Exception as exc:  # noqa: BLE001
        _record(seam, "malformed", None)
        logger.debug("djev %s: malformed response: %s", seam or "-", exc)
        return None
    if not out.answers:
        _record(seam, "empty", latency_ms)
        return None
    _record(seam, "ok", latency_ms)
    return out


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

#: The levels a rank question scores against. Ordered worst to best, because
#: `score` is an expected value over the level INDEX — reversing the list
#: reverses the ranking silently.
RANK_LEVELS = ("irrelevant", "tangential", "partly answers it",
               "directly answers it")


def rank_questions(candidates: Sequence[str], *,
                   levels: Sequence[str] = RANK_LEVELS) -> dict[str, dict]:
    """One `score` question per candidate, ids `c0`..`cN`."""
    return {f"c{i}": {"type": "score",
                      "instructions": f"How well does candidate [{i}] answer the query?",
                      "criteria": list(levels)}
            for i in range(len(candidates))}


def rank_state(query: str, candidates: Sequence[str], *,
               chars: int = 1200) -> str:
    """The canvas state for a ranking: the query, then the numbered pool."""
    body = "\n\n".join(f"[{i}] {str(c)[:chars]}" for i, c in enumerate(candidates))
    return f"Query: {query}\n\nCandidates:\n{body}"


def rank(query: str, candidates: Sequence[str], *,
         timeout: float = DEFAULT_TIMEOUT_S, seam: str = "rank",
         floor: float | None = None,
         levels: Sequence[str] = RANK_LEVELS,
         chars: int = 1200, samples: int | None = None,
         max_n: int = RANK_MAX_N) -> list[dict] | None:
    """`[{index, score, label_mass, ...}]` best first, or `None`.

    Refuses more than `RANK_MAX_N` candidates rather than truncating: a caller
    that hands over 40 rows means to rank 40, and quietly scoring 16 of them
    would return a confident ordering of a slice nobody chose. It also refuses
    an answer the SERVER split across canvas chunks, which is the same failure
    arriving from the other side — different chunks are different shared
    contexts, so their scores are not comparable and sorting the union
    produces an artefact that looks exactly like a ranking.

    `max_n` raises the cap for a caller that measured a wider window, never past
    `CANVAS_CHUNK_QUESTIONS`: the vault recall ranks up to 32 (#1336), measured
    against qmd's cross-encoder on the same pools. `chars` is how much of each
    candidate the state carries and `samples` the number of reads; that recall
    measured 160 chars and one read as both the fastest and the best-ordered
    of the shapes it tried (full text and `samples: "auto"` were slower and
    ranked worse). One read's top pick is stable; its lower ranks are not
    exactly repeatable (see Replay, above).
    """
    if max_n > CANVAS_CHUNK_QUESTIONS:
        raise ValueError(f"max_n {max_n} is past the canvas split ({CANVAS_CHUNK_QUESTIONS})")
    n = len(candidates)
    if n == 0:
        return []
    if n > max_n:
        raise ValueError(
            f"djev ranks at most {max_n} candidates in one request "
            f"(got {n}); shortlist first")
    out = ask_sync(rank_state(query, candidates, chars=chars),
                   rank_questions(candidates, levels=levels),
                   timeout=timeout, seam=seam, floor=floor, samples=samples)
    if out is None:
        return None
    if out.cross_chunk:
        logger.debug("djev %s: refusing a ranking split across %d canvas chunks",
                     seam, len(out.chunks))
        return None
    rows = []
    for i in range(n):
        a = out.get(f"c{i}")
        if a is None:
            return None
        rows.append({"index": i, "score": float(a.value),
                     "label": a.label, "confidence": a.confidence,
                     "label_mass": a.label_mass,
                     "argmax_is_label": a.argmax_is_label,
                     "low_trust": a.low_trust})
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def reachable(timeout: float = 2.0) -> bool:
    """A live `GET /health`. Never called from `list_tools()` — see
    `agent_mcp/djev.py`."""
    try:
        with urllib.request.urlopen(structured_url() + "/health",
                                    timeout=timeout) as resp:
            return json.loads(resp.read()).get("status") == "ok"
    except Exception:  # noqa: BLE001
        return False


def stats() -> dict:
    """Call counts and recent latency per seam, for `djev_status` and
    `/state`. Offline — it reads process memory and touches no socket."""
    with _lock:
        per_seam = {}
        for seam, counts in _calls.items():
            ring = list(_latency.get(seam) or ())
            per_seam[seam] = {
                **counts,
                "recent_n": len(ring),
                "recent_avg_ms": round(sum(ring) / len(ring), 1) if ring else None,
                "recent_max_ms": round(max(ring), 1) if ring else None,
            }
        return {"url": structured_url(), "enabled": enabled(), "seams": per_seam}


def reset_stats() -> None:
    """For tests. Production never calls it."""
    with _lock:
        _calls.clear()
        _latency.clear()
