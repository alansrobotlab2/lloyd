#!/usr/bin/env python3
"""Qwen3.8's context-rot curve, and the compaction trigger it implies (P7).

`compaction.microcompact.trigger_fraction` (0.72 of the 210,144-token
truncation threshold, ~151k) was set on 2026-09-10 as a KV cost knob. Nothing
has ever measured where the primary's recall actually starts to fall with
prompt length, so the number is a guess on the quality side. This runner
measures it and turns the curve into a recommendation.

WHAT ONE REQUEST IS
-------------------
A haystack of a known length (the engine's own token count) with one or more
needles at a known depth, then the question LAST, thinking off,
temperature 0, `max_tokens` 64, priority 1:

  shape    repo     one user message of this tree's code and docs
                    (`bench-admission-stall.py::corpus()`), the question
                    appended after it.
           session  Lloyd-shaped history (`run_compaction_recall_eval.
                    build_session`: Read/Grep/Bash call-result pairs cut from
                    this tree), its own planted turn and salt lines removed,
                    the question as the final user message.
  needles  single     the billing-east relay's current port.
           multi3     that port, the release codename, the maintenance-window
                      time — three lines within ±1% of the depth.
           distract4  the port needle plus four salted distractors at least
                      0.2 of the haystack away: the relay's OLD port (stated
                      as pre-migration) and three sibling relays' ports, all
                      digit permutations of the answer.

Needles and distractors are `# ops note` lines inserted at a line boundary of
a tool result (session) or of the document (repo), exactly as the compaction
eval salts its filler. The grid is {50k,100k,150k,200k,240k} × depth
{0.1,0.3,0.5,0.7,0.9} × the three needle conditions × 3 seeds = 225
requests per shape.

Every request is sent twice: COLD, with a fresh nonce at position 0 of the
system message so nothing is prefix-cached, then WARM, byte-identical, so
the whole prompt is. TTFT is taken to the first content delta through the
harness's own `stream_chat` and `_merge_usage`. The cold answer is graded;
the warm one is recorded for agreement.

GUARDS
------
* `app.vllm_metrics.wait_idle(quiet_s=5)` before EVERY request; an engine
  that does not go quiet exits the run before anything is sent.
* The worker pool is paused through `POST /api/workers/pause` (an operator
  pause, so a landing's backend restart cannot lift it mid-run) and resumed
  in a `finally`, SIGTERM/SIGHUP included. A pool that was already paused
  when the run started is never resumed by it. If the automod promoter holds
  its own pause at the end, the resume waits for it to clear (an operator
  resume lifts both) and otherwise leaves the pool paused and says so.
* `LLOYD_DJEV_SHADOW=0`, priority 1 (a real chat still outranks it).

DECISION (`--decide`, also run at the end of a live run)
--------------------------------------------------------
`A(L)` is accuracy under distractors. `L*` is the largest grid length such
that every length up to it has `A(L) >= 0.9·A(50k)` and no depth under
`0.8·A(50k)` (a prefix rule: a noisy pass at 200k after a fail at 150k does
not count). Per shape, and the smaller of the shapes decides.

  L* >= 200k        keep 0.72 / 0.52.
  151k <= L* < 200k keep: the curve holds through today's trigger.
  L* < 151k         trigger = floor(L*/threshold·20)/20, target = trigger − 0.20,
                    then the compaction recall eval and a 3-day soak; adopt only
                    if extra compactions/day × cold TTFT < 5% of engine-busy
                    seconds and landings per round-hour stay in the prior two
                    weeks' range (`round scorecard`).

The cost side reads `usage.db` READ-ONLY: turns in the last 14 days whose
peak prompt (`input_tokens`, or a relief pass's `used_before` from the
`compaction` column, whichever is larger) falls in [new trigger, current
trigger) — each one a compaction the new value adds — times the measured cold
TTFT at the new target; engine-busy seconds are the union of turn intervals.
The 3-day `prefix_misses` / `reprefill_tokens` baseline is printed for the
soak to compare against.

Re-run whenever `models.primary.expect_model` changes.

    .venvs/lloyd/bin/python eval/run_context_rot_eval.py --dry-run
    .venvs/lloyd/bin/python eval/run_context_rot_eval.py --lengths 50000 --limit 6
    .venvs/lloyd/bin/python eval/run_context_rot_eval.py              # both shapes, 450 cold
    .venvs/lloyd/bin/python eval/run_context_rot_eval.py --decide     # latest json
"""

from __future__ import annotations

import os

# Before any app import: the eval's mute for djev's shadow rows.
os.environ.setdefault("LLOYD_DJEV_SHADOW", "0")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import secrets  # noqa: E402
import signal  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Awaitable, Callable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MEASUREMENTS = ROOT / "eval" / "measurements"
BACKEND = "http://127.0.0.1:8080"
DEFAULT_BASE = "http://127.0.0.1:8096"
MODEL = "primary"
PRIORITY = 1
MAX_TOKENS = 64
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}

LENGTHS = (50_000, 100_000, 150_000, 200_000, 240_000)
DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)
CONDITIONS = ("single", "multi3", "distract4")
SEEDS = (1, 2, 3)
SHAPES = ("repo", "session")

MULTI_SPREAD = 0.01          # multi3's outer needles sit this far either side
DISTRACTOR_STEP = 0.2        # distractors at depth + k·step (cyclic), k = 1..4
DEPTH_TOLERANCE = 0.02       # what the tests hold the placement to

PASS_MEAN = 0.9              # A(L) >= 0.9·A(base)
PASS_POSITION = 0.8          # every depth >= 0.8·A(base)
KEEP_AT = 200_000            # L* at or past this keeps today's values
BAND = 0.20                  # target = trigger − BAND
COST_CEILING = 0.05          # extra compaction seconds / busy seconds

_SALT_RE = re.compile(r"\n# ops: [\w-]+ relay listens on port \d+")


def _load(name: str, path: Path):
    """Import a sibling script by path (the bench's name has dashes)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def bench():
    return _load("bench_admission_stall", ROOT / "agent-services" / "bin" / "bench-admission-stall.py")


def recall_eval():
    return _load("run_compaction_recall_eval", ROOT / "eval" / "run_compaction_recall_eval.py")


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    shape: str
    length: int
    seed: int
    depth: float
    condition: str

    @property
    def key(self) -> str:
        return f"{self.shape}-{self.length // 1000}k-s{self.seed}-d{int(round(self.depth * 100))}-{self.condition}"


def build_grid(shapes=SHAPES, lengths=LENGTHS, depths=DEPTHS,
               conditions=CONDITIONS, seeds=SEEDS) -> list[Cell]:
    """Deterministic order: shape, length, seed, depth, condition — so one
    haystack serves fifteen consecutive cells and is then dropped."""
    return [Cell(sh, L, s, d, c) for sh in shapes for L in lengths for s in seeds
            for d in depths for c in conditions]


# ---------------------------------------------------------------------------
# Needles
# ---------------------------------------------------------------------------


@dataclass
class Needles:
    codename: str
    port: str
    old_port: str
    siblings: dict[str, str]
    window: str

    def wrong_ports(self) -> set[str]:
        return {self.old_port, *self.siblings.values()} - {self.port}


def make_needles(seed: int, length: int) -> Needles:
    """The compaction eval's planted facts (codename, a current port whose
    siblings and old value are digit permutations of it) plus a time."""
    rng = random.Random(f"context-rot:{seed}:{length}")
    p = recall_eval().make_planted(rng)
    sib = dict(list(p.distractor_ports.items())[:3])
    window = f"{rng.randint(0, 23):02d}:{rng.choice([0, 15, 30, 45]):02d}"
    return Needles(codename=p.passphrase, port=p.port, old_port=p.old_port,
                   siblings=sib, window=window)


def needle_lines(n: Needles) -> dict[str, str]:
    return {
        "port": f"# ops note (Halcyon relay migration): the billing-east relay now listens on port {n.port}.",
        "codename": f"# ops note (Halcyon release): this release ships under the codename {n.codename}.",
        "window": f"# ops note (Halcyon maintenance): the maintenance window opens at {n.window} UTC.",
    }


def distractor_lines(n: Needles) -> list[str]:
    out = [f"# ops note (pre-migration inventory, before the relay migration): "
           f"the billing-east relay listens on port {n.old_port}."]
    out += [f"# ops note: the {svc} relay listens on port {p}." for svc, p in n.siblings.items()]
    return out


QUESTION_PORT = (
    "Question: according to the notes in this conversation, which port does the "
    "billing-east relay listen on now, after the relay migration? Answer with "
    "exactly one line:\nPORT: <value>")
QUESTION_MULTI = (
    "Question: from the notes in this conversation, give the Halcyon release "
    "codename, the port the billing-east relay listens on now (after the relay "
    "migration), and the time the Halcyon maintenance window opens. Answer with "
    "exactly three lines:\nCODENAME: <value>\nPORT: <value>\nWINDOW: <HH:MM>")


def facts_for(condition: str) -> tuple[str, ...]:
    return ("codename", "port", "window") if condition == "multi3" else ("port",)


def question_for(condition: str) -> str:
    return QUESTION_MULTI if condition == "multi3" else QUESTION_PORT


def placements(cell: Cell, n: Needles) -> list[tuple[float, str, str]]:
    """(depth, kind, line) for everything planted in this cell."""
    lines = needle_lines(n)
    d = cell.depth
    if cell.condition == "multi3":
        out = [(max(0.01, d - MULTI_SPREAD), "codename", lines["codename"]),
               (d, "port", lines["port"]),
               (min(0.99, d + MULTI_SPREAD), "window", lines["window"])]
    else:
        out = [(d, "port", lines["port"])]
    if cell.condition == "distract4":
        rng = random.Random(f"distract:{cell.key}")
        dl = distractor_lines(n)
        rng.shuffle(dl)
        for k, line in enumerate(dl, start=1):
            pos = (d + k * DISTRACTOR_STEP) % 1.0
            out.append((min(0.98, max(0.02, pos)), "distractor", line))
    return out


# ---------------------------------------------------------------------------
# Haystacks
# ---------------------------------------------------------------------------


@dataclass
class Haystack:
    shape: str
    length: int
    seed: int
    messages: list[dict[str, Any]]     # OpenAI shape, string content
    host: set[int]                     # indices a needle may be inserted into
    tokens: int = 0                    # engine count (or estimate in a dry run)
    counted: str = "estimate"
    meta: dict[str, Any] = field(default_factory=dict)


def _flat(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


REPO_SEED_OFFSET = 200_000   # chars: each seed reads a different stretch of the tree


def _repo_messages(seed: int, est_tokens: int) -> tuple[list[dict], set[int]]:
    b = bench()
    skip = (seed - 1) * REPO_SEED_OFFSET
    text = b.corpus(b.A_GLOBS + b.B_GLOBS, int(est_tokens * 4) + skip)[skip:]
    text = text[text.find("\n") + 1:] if skip else text
    # Trim to the estimator's size, so the dry run and the calibration agree
    # on what "est_tokens" means for both shapes.
    from app.compaction import estimate_tokens
    n = max(estimate_tokens(text), 1)
    text = text[: int(len(text) * est_tokens / n)]
    return [{"role": "user", "content": text}], {0}


def _session_messages(seed: int, est_tokens: int) -> tuple[list[dict], set[int]]:
    R = recall_eval()
    s = R.build_session(seed, est_tokens, 0.0, corpus=_corpus_files())
    msgs = list(s.messages)
    # Its own planted turn (user, assistant call, the notes result, reply) out,
    # and its sibling-port salt lines: this eval plants its own.
    del msgs[s.planted_index - 2: s.planted_index + 2]
    out: list[dict] = []
    host: set[int] = set()
    for m in msgs:
        mm = {k: v for k, v in m.items() if k != "content"}
        text = _SALT_RE.sub("", _flat(m.get("content")))
        mm["content"] = text
        if m.get("role") == "tool" and text:
            host.add(len(out))
        out.append(mm)
    return out, host


_CORPUS: list[Path] | None = None


def _corpus_files() -> list[Path]:
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = recall_eval()._corpus_files(ROOT)
    return _CORPUS


def build_haystack(shape: str, length: int, seed: int, est_tokens: int) -> Haystack:
    if shape == "repo":
        msgs, host = _repo_messages(seed, est_tokens)
    elif shape == "session":
        msgs, host = _session_messages(seed, est_tokens)
    else:
        raise ValueError(f"unknown shape {shape!r}")
    return Haystack(shape=shape, length=length, seed=seed, messages=msgs, host=host,
                    meta={"est_target": est_tokens})


def haystack_chars(messages: list[dict]) -> int:
    return sum(len(m.get("content") or "") for m in messages)


def plant(hay: Haystack, items: list[tuple[float, str, str]]) -> tuple[list[dict], list[dict]]:
    """Insert each line at a line boundary nearest its depth (by characters
    of message content), inside a host message. Returns the new messages and
    where each line actually landed. The haystack itself is not modified."""
    msgs = [dict(m) for m in hay.messages]
    total = haystack_chars(msgs)
    spans: list[tuple[int, int, int]] = []     # (msg index, start, end)
    pos = 0
    for i, m in enumerate(msgs):
        n = len(m.get("content") or "")
        if i in hay.host and n:
            spans.append((i, pos, pos + n))
        pos += n
    if not spans:
        raise RuntimeError("haystack has nowhere to plant")
    targets = []
    for depth, kind, line in items:
        want = depth * total
        i, a, b = min(spans, key=lambda s: 0 if s[1] <= want <= s[2]
                      else min(abs(want - s[1]), abs(want - s[2])))
        text = msgs[i]["content"]
        local = int(min(max(want - a, 0), b - a))
        # nearest newline to `local` (or the ends of the message)
        before = text.rfind("\n", 0, local)
        after = text.find("\n", local)
        cands = [c for c in (before, after, 0, len(text)) if c >= 0]
        at = min(cands, key=lambda c: abs(c - local))
        targets.append((i, at, kind, line))
    # Highest offset first, so earlier offsets stay valid.
    for i, at, _kind, line in sorted(targets, key=lambda t: (t[0], t[1]), reverse=True):
        text = msgs[i]["content"]
        msgs[i]["content"] = text[:at] + "\n" + line + "\n" + text[at:]
    return msgs, landed(msgs, items)


def landed(msgs: list[dict], items: list[tuple[float, str, str]]) -> list[dict]:
    blob = "".join(m.get("content") or "" for m in msgs)
    out = []
    for depth, kind, line in items:
        at = blob.find(line)
        out.append({"kind": kind, "want": round(depth, 4),
                    "got": round(at / max(len(blob), 1), 4) if at >= 0 else None})
    return out


def request_messages(hay: Haystack, planted: list[dict], cell: Cell, nonce: str) -> list[dict]:
    system = {"role": "system", "content": (
        f"[probe {nonce}] You are answering a question about the material in "
        "this conversation. Answer only from it, and exactly in the format asked.")}
    q = question_for(cell.condition)
    if hay.shape == "repo":
        doc = planted[0]["content"]
        return [system, {"role": "user", "content": f"{doc}\n\n---\n\n{q}"}]
    return [system, *planted, {"role": "user", "content": q}]


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

_LINES = {
    "codename": re.compile(r"CODENAME\s*[:=]\s*(.+)", re.I),
    "port": re.compile(r"PORT\s*[:=]\s*(.+)", re.I),
    "window": re.compile(r"WINDOW\s*[:=]\s*(.+)", re.I),
}


def grade(text: str, n: Needles, condition: str) -> dict[str, Any]:
    """`hit` / `wrong` / `undecided` per fact, code only (undecided = miss).
    The port is graded with every salted value as a wrong answer, and an
    answer naming the right port AND a wrong one has not answered."""
    fv = recall_eval()._field_verdict
    text = text or ""
    verdicts: dict[str, str] = {}
    for fact in facts_for(condition):
        m = None
        for m in _LINES[fact].finditer(text):
            pass
        line = m.group(1) if m else text
        if fact == "port":
            v = fv(line, n.port, n.wrong_ports(), digits=True, labelled=bool(m))
        elif fact == "codename":
            v = fv(line, n.codename, set(), digits=False, labelled=bool(m))
        else:
            v = fv(line, n.window, set(), digits=True, labelled=bool(m))
        verdicts[fact] = v
    hits = sum(v == "hit" for v in verdicts.values())
    return {"verdicts": verdicts, "accuracy": hits / len(verdicts)}


# ---------------------------------------------------------------------------
# The engine and the pool (live only)
# ---------------------------------------------------------------------------


class EngineBusy(RuntimeError):
    """The engine never went quiet: nothing was sent."""


async def ensure_idle(base: str, *, limit_s: float, client=None) -> None:
    from app import vllm_metrics
    try:
        await vllm_metrics.wait_idle(base, quiet_s=5.0, limit_s=limit_s, client=client)
    except TimeoutError as exc:
        raise EngineBusy(f"refusing to send: {exc}") from exc


async def count_tokens(client, base: str, messages: list[dict]) -> int:
    r = await client.post(f"{base}/tokenize", json={
        "model": MODEL, "messages": messages, "add_generation_prompt": True,
        **NO_THINK}, timeout=300)
    r.raise_for_status()
    return int(r.json()["count"])


async def one_request(base: str, messages: list[dict]) -> dict:
    from app.harness.client import stream_chat
    from app.harness.loop import _merge_usage
    t0 = time.monotonic()
    first = None
    usage: dict = {}
    text: list[str] = []
    body = {**NO_THINK, "max_tokens": MAX_TOKENS, "temperature": 0}
    async for chunk in stream_chat(base_url=base, model=MODEL, messages=messages,
                                   tools=None, extra_body=body, cancel_event=None,
                                   timeout_s=1800, priority=PRIORITY):
        if u := chunk.get("usage"):
            usage = _merge_usage(usage, u)
        for choice in chunk.get("choices") or []:
            content = (choice.get("delta") or {}).get("content")
            if content:
                first = first or time.monotonic()
                text.append(content)
    end = time.monotonic()
    return {"ttft_s": round((first or end) - t0, 3), "total_s": round(end - t0, 3),
            "prompt_tokens": usage.get("input_tokens"),
            "cached_tokens": usage.get("cache_read"),
            "output_tokens": usage.get("output_tokens"), "text": "".join(text)}


class PoolPause:
    """Pause the worker pool for the run; resume only a pause we took.

    An operator pause, so it survives a landing's backend restart. Resuming as
    operator also lifts an automod pause, so while the promoter holds one we
    wait for it to clear rather than un-pause a landing's drain.
    """

    def __init__(self, backend: str, client, *, automod_wait_s: float = 900.0,
                 poll_s: float = 10.0):
        self.backend = backend.rstrip("/")
        self.client = client
        self.ours = False
        self.automod_wait_s = automod_wait_s
        self.poll_s = poll_s
        self.log: list[str] = []

    async def status(self) -> dict | None:
        try:
            r = await self.client.get(f"{self.backend}/api/workers/status", timeout=10)
            body = r.json()
        except Exception:  # noqa: BLE001
            return None
        pool = body.get("pool") if isinstance(body, dict) else None
        return pool if isinstance(pool, dict) else None

    async def _post(self, paused: bool) -> dict:
        r = await self.client.post(f"{self.backend}/api/workers/pause",
                                   json={"paused": paused}, timeout=10)
        r.raise_for_status()
        return r.json()

    async def __aenter__(self) -> "PoolPause":
        st = await self.status()
        if st is None or "paused" not in st:
            raise SystemExit("cannot read /api/workers/status — refusing to run on "
                             "an engine the pool may be using (pass --no-pool-pause "
                             "if the backend is down)")
        if st.get("paused"):
            self.log.append(f"pool already paused by {st.get('paused_by')}; leaving it alone")
            return self
        got = await self._post(True)
        if not got.get("paused"):
            raise SystemExit(f"pool pause did not take: {got}")
        self.ours = True
        self.log.append("pool paused by this run")
        return self

    async def __aexit__(self, *exc) -> None:
        if not self.ours:
            return
        deadline = time.monotonic() + self.automod_wait_s
        while True:
            st = await self.status() or {}
            if "automod" not in (st.get("paused_by") or []):
                break
            if time.monotonic() >= deadline:
                msg = ("!! the automod promoter still holds a pool pause; NOT resuming "
                       "(an operator resume would lift it). Resume by hand once the "
                       "landing is done: curl -X POST localhost:8080/api/workers/pause "
                       "-H 'content-type: application/json' -d '{\"paused\": false}'")
                print(msg, file=sys.stderr)
                self.log.append(msg)
                return
            await asyncio.sleep(self.poll_s)
        try:
            await self._post(False)
            self.log.append("pool resumed")
        except Exception as e:  # noqa: BLE001
            msg = f"!! pool resume FAILED ({e}); the pool is still paused"
            print(msg, file=sys.stderr)
            self.log.append(msg)


class _NoPause:
    ours = False
    log = ["pool pause skipped (--no-pool-pause)"]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

Counter = Callable[[list[dict]], Awaitable[int]]


async def sized_haystack(shape: str, length: int, seed: int, counter: Counter | None) -> Haystack:
    """A haystack whose full request (system, haystack, question) is `length`
    tokens: built at the estimator's size, counted, rescaled once, counted
    again. With no counter (dry run) the estimate stands."""
    from app.compaction import estimate_conversation_tokens
    probe = Cell(shape, length, seed, 0.5, "multi3")

    def full(h: Haystack) -> list[dict]:
        return request_messages(h, h.messages, probe, "0" * 16)

    est = length
    hay = build_haystack(shape, length, seed, est)
    if counter is None:
        hay.tokens = estimate_conversation_tokens(full(hay))
        return hay
    for _ in range(2):
        n = await counter(full(hay))
        if abs(n - length) <= max(500, length * 0.005):
            break
        est = max(1_000, int(est * length / max(n, 1)))
        hay = build_haystack(shape, length, seed, est)
    hay.tokens = await counter(full(hay))
    hay.counted = "engine"
    hay.meta["est_target"] = est
    return hay


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


async def engine_meta(client, base: str) -> dict:
    from app import vllm_metrics
    out: dict[str, Any] = {}
    try:
        r = await client.get(f"{base}/v1/models", timeout=10)
        data = (r.json().get("data") or [{}])[0]
        out["served"] = data.get("id")
        out["root"] = data.get("root")
        out["max_model_len"] = data.get("max_model_len")
    except Exception as e:  # noqa: BLE001
        out["models_error"] = str(e)
    try:
        text = (await client.get(f"{base}/metrics", timeout=10)).text
        out["cache_config"] = vllm_metrics.cache_config_from_text(text)
    except Exception as e:  # noqa: BLE001
        out["metrics_error"] = str(e)
    return out


def _expect_model() -> str:
    try:
        from app.config import _get_model_cfg
        return str(_get_model_cfg(MODEL).get("expect_model") or "")
    except Exception:  # noqa: BLE001
        return ""


async def run_cells(cells: list[Cell], *, base: str, client, write: Callable[[list[dict]], None],
                    first_idle_limit_s: float, idle_limit_s: float,
                    warm: bool = True) -> list[dict]:
    rows: list[dict] = []
    await ensure_idle(base, limit_s=first_idle_limit_s, client=client)

    async def counter(msgs):
        await ensure_idle(base, limit_s=idle_limit_s, client=client)
        return await count_tokens(client, base, msgs)

    hay: Haystack | None = None
    for i, cell in enumerate(cells):
        if hay is None or (hay.shape, hay.length, hay.seed) != (cell.shape, cell.length, cell.seed):
            hay = await sized_haystack(cell.shape, cell.length, cell.seed, counter)
        n = make_needles(cell.seed, cell.length)
        items = placements(cell, n)
        planted, where = plant(hay, items)
        msgs = request_messages(hay, planted, cell, secrets.token_hex(8))
        await ensure_idle(base, limit_s=idle_limit_s, client=client)
        cold = await one_request(base, msgs)
        row: dict[str, Any] = {
            "cell": cell.key, "shape": cell.shape, "length": cell.length,
            "seed": cell.seed, "depth": cell.depth, "condition": cell.condition,
            "haystack_tokens": hay.tokens, "placed": where,
            "needles": {"port": n.port, "codename": n.codename, "window": n.window,
                        "wrong_ports": sorted(n.wrong_ports())},
            "cold": cold, **grade(cold["text"], n, cell.condition),
        }
        if warm:
            await ensure_idle(base, limit_s=idle_limit_s, client=client)
            w = await one_request(base, msgs)
            w["same_answer"] = w["text"].strip() == cold["text"].strip()
            row["warm"] = w
        rows.append(row)
        write(rows)
        print(f"[{i + 1}/{len(cells)}] {cell.key}: acc={row['accuracy']:.2f} "
              f"ttft cold={cold['ttft_s']}s"
              + (f" warm={row['warm']['ttft_s']}s" if warm else "")
              + f" prompt={cold['prompt_tokens']}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Summary and decision (pure)
# ---------------------------------------------------------------------------


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def _p50(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 3) if xs else None


def summarize(rows: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for shape in sorted({r["shape"] for r in rows}):
        rs = [r for r in rows if r["shape"] == shape]
        lengths = sorted({r["length"] for r in rs})
        by_len: dict[str, Any] = {}
        by_depth: dict[str, Any] = {}
        ttft: dict[str, Any] = {}
        for L in lengths:
            rl = [r for r in rs if r["length"] == L]
            by_len[str(L)] = {c: _mean([r["accuracy"] for r in rl if r["condition"] == c])
                              for c in CONDITIONS if any(r["condition"] == c for r in rl)}
            dr = [r for r in rl if r["condition"] == "distract4"]
            by_depth[str(L)] = {str(d): _mean([r["accuracy"] for r in dr if r["depth"] == d])
                                for d in sorted({r["depth"] for r in dr})}
            ttft[str(L)] = {
                "cold_p50": _p50([r["cold"]["ttft_s"] for r in rl]),
                "warm_p50": _p50([r["warm"]["ttft_s"] for r in rl if r.get("warm")]),
                "prompt_tokens_p50": _p50([r["cold"]["prompt_tokens"] for r in rl
                                           if r["cold"].get("prompt_tokens")]),
                "warm_same_answer": _mean([1.0 if r["warm"]["same_answer"] else 0.0
                                           for r in rl if r.get("warm")]),
            }
        out[shape] = {"lengths": lengths, "accuracy": by_len,
                      "distract4_by_depth": by_depth, "ttft": ttft, "n": len(rs)}
    return out


def l_star(shape_summary: dict[str, Any]) -> dict[str, Any]:
    """The prefix rule: the largest length L such that every grid length up to
    L holds A >= 0.9·A(base) under distractors and no depth under 0.8·A(base)."""
    lengths = shape_summary["lengths"]
    acc = shape_summary["accuracy"]
    depth = shape_summary["distract4_by_depth"]
    if not lengths:
        return {"L_star": None, "reason": "no rows"}
    base_len = lengths[0]
    base = (acc.get(str(base_len)) or {}).get("distract4")
    if not base:
        return {"L_star": None, "base_length": base_len, "base": base,
                "reason": "no distractor accuracy at the base length"}
    best = None
    fails: list[dict] = []
    for L in lengths:
        a = (acc.get(str(L)) or {}).get("distract4")
        per = {d: v for d, v in (depth.get(str(L)) or {}).items() if v is not None}
        low = {d: v for d, v in per.items() if v < PASS_POSITION * base}
        ok = a is not None and a >= PASS_MEAN * base and not low
        if not ok:
            fails.append({"length": L, "A": a, "low_positions": low})
            break
        best = L
    return {"L_star": best, "base_length": base_len, "base": base,
            "first_fail": fails[0] if fails else None,
            "reason": None if best else "fails at the base length itself"}


def recommend(summary: dict[str, Any], *, threshold: int, current_trigger: float,
              current_target: float, full_grid: bool = True) -> dict[str, Any]:
    per = {shape: l_star(s) for shape, s in summary.items()}
    stars = [p["L_star"] for p in per.values()]
    current_tokens = int(current_trigger * threshold)
    rec: dict[str, Any] = {"per_shape": per, "threshold": threshold,
                           "current": {"trigger": current_trigger, "target": current_target,
                                       "trigger_tokens": current_tokens},
                           "partial_grid": not full_grid}
    if not stars or any(s is None for s in stars):
        rec.update(action="inconclusive", L_star=None,
                   why="no length passes the rule in at least one shape; do not move the trigger on this run")
        return rec
    L = min(stars)
    rec["L_star"] = L
    if L >= KEEP_AT:
        rec.update(action="keep", trigger=current_trigger, target=current_target,
                   why=f"L* = {L:,} >= {KEEP_AT:,}: recall holds past any trigger we would set")
    elif L >= current_tokens:
        rec.update(action="keep", trigger=current_trigger, target=current_target,
                   why=f"L* = {L:,} is at or past today's trigger ({current_tokens:,})")
    else:
        trig = math.floor(L / threshold * 20) / 20
        tgt = round(trig - BAND, 2)
        clamped = tgt < 0.10
        rec.update(action="lower", trigger=trig, target=max(0.10, tgt),
                   trigger_tokens=int(trig * threshold), target_tokens=int(max(0.10, tgt) * threshold),
                   target_clamped=clamped,
                   why=(f"L* = {L:,} < today's trigger ({current_tokens:,}): "
                        f"trigger = floor(L*/{threshold:,}·20)/20"),
                   next=["run eval/run_compaction_recall_eval.py with the new values",
                         "3-day soak: usage.db prefix_misses / reprefill_tokens against the baseline below",
                         f"adopt only if the cost side stays under {COST_CEILING:.0%} of engine-busy "
                         "seconds and landings per round-hour stay in the prior two weeks' range "
                         "(`round scorecard`)"])
    return rec


def interp_ttft(ttft: dict[str, Any], tokens: int) -> float | None:
    pts = sorted((int(L), v["cold_p50"]) for L, v in ttft.items() if v.get("cold_p50") is not None)
    if not pts:
        return None
    if tokens <= pts[0][0]:
        return round(pts[0][1] * tokens / pts[0][0], 3)
    for (a, ya), (b, yb) in zip(pts, pts[1:]):
        if a <= tokens <= b:
            return round(ya + (yb - ya) * (tokens - a) / (b - a), 3)
    return round(pts[-1][1] * tokens / pts[-1][0], 3)


# ---------------------------------------------------------------------------
# Cost side (usage.db, read-only)
# ---------------------------------------------------------------------------


def _peak(row: dict) -> int:
    peak = int(row["input_tokens"] or 0)
    comp = row["compaction"]
    if comp:
        try:
            rec = json.loads(comp)
            for p in rec.get("relief") or []:
                peak = max(peak, int(p.get("used_before") or 0))
        except (ValueError, TypeError, AttributeError):
            pass
    return peak


def _union_seconds(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    end = None
    start = None
    for a, b in sorted(intervals):
        if end is None or a > end:
            if end is not None:
                total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    if end is not None:
        total += end - start
    return total


def cost_side(db_path: Path, *, new_trigger_tokens: int | None, current_trigger_tokens: int,
              cold_ttft_s: float | None, days: int = 14, soak_days: int = 3,
              now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    since = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    soak_since = (now - timedelta(days=soak_days)).strftime("%Y-%m-%dT%H:%M:%S")
    if not Path(db_path).exists():
        return {"error": f"{db_path} not found"}
    import usage_store
    rows = usage_store.read_rows_readonly(
        db_path, ["ts", "session_id", "input_tokens", "duration_ms",
                  "compaction", "prefix_misses", "reprefill_tokens"], since)

    def ts(r) -> float:
        return datetime.strptime(r["ts"][:19], "%Y-%m-%dT%H:%M:%S").timestamp()

    busy = _union_seconds([(ts(r) - (r["duration_ms"] or 0) / 1000.0, ts(r))
                           for r in rows if r["duration_ms"]])
    peaks = [(_peak(r), r) for r in rows]
    bands = {}
    edges = [0, *LENGTHS, 10**9]
    for a, b in zip(edges, edges[1:]):
        bands[f"{a // 1000}k-{b // 1000 if b < 10**9 else 'inf'}k"] = sum(a <= p < b for p, _ in peaks)
    out: dict[str, Any] = {
        "window_days": days, "turns": len(rows), "busy_seconds": round(busy),
        "busy_seconds_per_day": round(busy / days), "peak_bands": bands,
        "current_trigger_tokens": current_trigger_tokens,
    }
    soak = [r for r in rows if r["ts"] >= soak_since]
    out["soak_baseline"] = {
        "days": soak_days, "turns": len(soak),
        "prefix_misses": sum(int(r["prefix_misses"] or 0) for r in soak),
        "reprefill_tokens": sum(int(r["reprefill_tokens"] or 0) for r in soak),
        "turns_with_misses": sum(1 for r in soak if (r["prefix_misses"] or 0) > 0),
    }
    if new_trigger_tokens is None or new_trigger_tokens >= current_trigger_tokens:
        out["extra_compactions"] = None
        out["verdict"] = "not applicable: the trigger does not move"
        return out
    hit = [(p, r) for p, r in peaks if new_trigger_tokens <= p < current_trigger_tokens]
    per_day = len(hit) / days
    bg = sum(1 for _, r in hit if len(str(r["session_id"] or "").split("_")) >= 4)
    out.update(extra_compactions=len(hit), extra_per_day=round(per_day, 2),
               extra_background=bg, extra_user=len(hit) - bg,
               cold_ttft_s=cold_ttft_s)
    if cold_ttft_s is None or not busy:
        out["verdict"] = "cannot price: no cold TTFT or no busy time"
        return out
    frac = per_day * cold_ttft_s / (busy / days)
    out["cost_fraction"] = round(frac, 4)
    out["verdict"] = ("within budget" if frac < COST_CEILING
                      else f"over budget (>= {COST_CEILING:.0%} of engine-busy seconds)")
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _current_policy() -> tuple[int, float, float]:
    from app.compaction import get_context_window, truncation_threshold
    threshold = truncation_threshold(get_context_window(MODEL))
    trig, tgt = 0.72, 0.52
    try:
        from app.config import CONFIG
        mc = (CONFIG.get("compaction") or {}).get("microcompact") or {}
        trig = float(mc.get("trigger_fraction", trig))
        tgt = float(mc.get("target_fraction", tgt))
    except Exception:  # noqa: BLE001
        pass
    return threshold, trig, tgt


def decide_report(doc: dict[str, Any], *, usage_db: Path | None) -> dict[str, Any]:
    rows = doc.get("rows") or []
    summary = summarize(rows)
    threshold, trig, tgt = _current_policy()
    grid = doc.get("meta", {}).get("grid") or {}
    full = (sorted(grid.get("lengths") or []) == sorted(LENGTHS)
            and len(rows) >= len(build_grid(shapes=tuple(summary) or SHAPES)))
    rec = recommend(summary, threshold=threshold, current_trigger=trig,
                    current_target=tgt, full_grid=full)
    ttft_src = summary.get("session") or next(iter(summary.values()), {})
    new_tok = rec.get("trigger_tokens") if rec.get("action") == "lower" else None
    cold = interp_ttft(ttft_src.get("ttft") or {}, rec["target_tokens"]) if new_tok else None
    if usage_db is None:
        from app.paths import USAGE_DB
        usage_db = USAGE_DB
    cost = cost_side(usage_db, new_trigger_tokens=new_tok,
                     current_trigger_tokens=int(trig * threshold), cold_ttft_s=cold)
    return {**doc, "summary": summary, "decision": rec, "cost": cost}


def render_md(doc: dict[str, Any]) -> str:
    meta = doc.get("meta") or {}
    rec = doc.get("decision") or {}
    cost = doc.get("cost") or {}
    out = [f"# Context rot: {meta.get('expect_model') or 'primary'}, {meta.get('date', '')} (P7)", ""]
    act = rec.get("action", "?")
    if act == "lower":
        out.append(f"**Verdict: lower the trigger to {rec['trigger']:.2f} / {rec['target']:.2f}** "
                   f"(pending the recall eval and the soak). {rec.get('why', '')}")
    elif act == "keep":
        out.append(f"**Verdict: keep {rec['trigger']:.2f} / {rec['target']:.2f}.** {rec.get('why', '')}")
    else:
        out.append(f"**Verdict: {act}.** {rec.get('why', '')}")
    if rec.get("partial_grid"):
        out.append("")
        out.append("_Partial grid: this run is not the measurement the rule was written for._")
    out += ["", f"Runner: `eval/run_context_rot_eval.py`. Commit `{meta.get('commit', '')[:10]}`. "
            f"Engine: {json.dumps(meta.get('engine') or {})}. Pool: {'; '.join(meta.get('pool') or [])}.",
            ""]
    for shape, s in (doc.get("summary") or {}).items():
        out += [f"## {shape}", "", "| length | single | multi3 | distract4 | cold TTFT p50 | warm TTFT p50 | prompt p50 |",
                "|---|---|---|---|---|---|---|"]
        for L in s["lengths"]:
            a = s["accuracy"].get(str(L)) or {}
            t = s["ttft"].get(str(L)) or {}
            out.append(f"| {L:,} | {a.get('single')} | {a.get('multi3')} | {a.get('distract4')} | "
                       f"{t.get('cold_p50')} | {t.get('warm_p50')} | {t.get('prompt_tokens_p50')} |")
        depths = sorted({d for v in s["distract4_by_depth"].values() for d in v}, key=float)
        if depths:
            out += ["", "distract4 by depth:", "",
                    "| length | " + " | ".join(depths) + " |", "|---|" + "---|" * len(depths)]
            for L in s["lengths"]:
                v = s["distract4_by_depth"].get(str(L)) or {}
                out.append(f"| {L:,} | " + " | ".join(str(v.get(d)) for d in depths) + " |")
        p = (rec.get("per_shape") or {}).get(shape) or {}
        out += ["", f"L* ({shape}) = {p.get('L_star')}; base A({p.get('base_length')}) = {p.get('base')}; "
                f"first fail: {json.dumps(p.get('first_fail'))}", ""]
    out += ["## Cost side (usage.db, last 14 days, read-only)", "", "```", json.dumps(cost, indent=2), "```", ""]
    if rec.get("next"):
        out += ["## Next", ""] + [f"- {n}" for n in rec["next"]] + [""]
    out.append("Re-run whenever `models.primary.expect_model` changes.")
    return "\n".join(out) + "\n"


def write_doc(doc: dict[str, Any], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = json_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str))
    tmp.replace(json_path)


def latest_json() -> Path | None:
    got = sorted(MEASUREMENTS.glob("context-rot-*.json"))
    return got[-1] if got else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _ints(s: str) -> tuple[int, ...]:
    return tuple(int(x.replace("_", "").replace("k", "000")) for x in s.split(",") if x.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shape", choices=("repo", "session", "both"), default="both")
    ap.add_argument("--lengths", type=_ints, default=LENGTHS,
                    help="comma list, e.g. 50000,100k (default: the full grid)")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N cells")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the grid and estimated token counts; touch neither engine nor pool")
    ap.add_argument("--decide", nargs="?", const="", default=None, metavar="JSON",
                    help="recompute summary, decision and cost side from a run (default: latest)")
    ap.add_argument("--base", default=None, help="primary base URL (default: config)")
    ap.add_argument("--backend", default=BACKEND)
    ap.add_argument("--no-warm", action="store_true", help="cold requests only")
    ap.add_argument("--no-pool-pause", action="store_true",
                    help="do not pause the pool (only when the backend is down)")
    ap.add_argument("--first-idle-limit", type=float, default=3600.0,
                    help="seconds to wait for the pool to drain before the first request")
    ap.add_argument("--idle-limit", type=float, default=600.0)
    ap.add_argument("--usage-db", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="json path (default eval/measurements/context-rot-<date>.json)")
    return ap.parse_args(argv)


def select_cells(args: argparse.Namespace) -> list[Cell]:
    shapes = SHAPES if args.shape == "both" else (args.shape,)
    cells = build_grid(shapes=shapes, lengths=tuple(args.lengths))
    return cells[: args.limit] if args.limit else cells


def _base_url(args) -> str:
    if args.base:
        return args.base.rstrip("/")
    try:
        from app.config import _get_model_cfg
        return str(_get_model_cfg(MODEL).get("base_url") or DEFAULT_BASE).rstrip("/")
    except Exception:  # noqa: BLE001
        return DEFAULT_BASE


def dry_run(args) -> int:
    cells = select_cells(args)
    print(f"grid: {len(cells)} cells ({'cold only' if args.no_warm else 'cold + warm'} = "
          f"{len(cells) * (1 if args.no_warm else 2)} requests); nothing is sent")
    seen: dict[tuple, Haystack] = {}
    for c in cells:
        k = (c.shape, c.length, c.seed)
        if k not in seen:
            seen[k] = asyncio.run(sized_haystack(c.shape, c.length, c.seed, None))
            h = seen[k]
            print(f"  {c.shape:8s} {c.length:>7,} seed {c.seed}: est {h.tokens:>7,} tokens, "
                  f"{haystack_chars(h.messages):>9,} chars, {len(h.messages)} messages")
    by = {}
    for c in cells:
        by[c.condition] = by.get(c.condition, 0) + 1
    print("  by condition:", by)
    c = cells[0] if cells else None
    if c:
        n = make_needles(c.seed, c.length)
        _msgs, where = plant(seen[(c.shape, c.length, c.seed)], placements(c, n))
        print(f"  sample {c.key}: answer port {n.port}, salts {sorted(n.wrong_ports())}, placed {where}")
    return 0


async def live(args) -> int:
    import httpx

    cells = select_cells(args)
    base = _base_url(args)
    date = datetime.now().strftime("%Y-%m-%d")
    json_path = args.out or MEASUREMENTS / f"context-rot-{date}.json"
    meta: dict[str, Any] = {
        "date": date, "commit": _git_head(), "base": base, "expect_model": _expect_model(),
        "priority": PRIORITY, "max_tokens": MAX_TOKENS,
        "grid": {"shapes": sorted({c.shape for c in cells}), "lengths": list(args.lengths),
                 "depths": list(DEPTHS), "conditions": list(CONDITIONS), "seeds": list(SEEDS),
                 "limit": args.limit, "cells": len(cells)},
    }
    doc: dict[str, Any] = {"meta": meta, "rows": [], "complete": False}

    def write(rows):
        doc["rows"] = rows
        write_doc(doc, json_path)

    # SIGTERM / SIGHUP cancel the run so the pool's `finally` still resumes.
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            loop.add_signal_handler(sig, task.cancel)
        except (NotImplementedError, RuntimeError):
            pass

    async with httpx.AsyncClient() as client:
        pause = _NoPause() if args.no_pool_pause else PoolPause(args.backend, client)
        try:
            async with pause:
                meta["engine"] = await engine_meta(client, base)
                await run_cells(cells, base=base, client=client, write=write,
                                first_idle_limit_s=args.first_idle_limit,
                                idle_limit_s=args.idle_limit, warm=not args.no_warm)
        finally:
            meta["pool"] = list(pause.log)
            write(doc["rows"])
    doc["complete"] = True
    doc = decide_report(doc, usage_db=args.usage_db)
    write_doc(doc, json_path)
    json_path.with_suffix(".md").write_text(render_md(doc))
    print(f"wrote {json_path} and {json_path.with_suffix('.md')}")
    print(json.dumps(doc["decision"], indent=1, default=str))
    return 0


def decide_main(args) -> int:
    path = Path(args.decide) if args.decide else latest_json()
    if not path or not path.exists():
        print("no context-rot json to decide from", file=sys.stderr)
        return 2
    doc = decide_report(json.loads(path.read_text()), usage_db=args.usage_db)
    write_doc(doc, path)
    path.with_suffix(".md").write_text(render_md(doc))
    print(json.dumps({"decision": doc["decision"], "cost": doc["cost"]}, indent=1, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.decide is not None:
        return decide_main(args)
    if args.dry_run:
        return dry_run(args)
    try:
        return asyncio.run(live(args))
    except EngineBusy as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("interrupted; rows so far are in the json, the pool was handled on the way out",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
