#!/usr/bin/env python3
"""Engine output-integrity probe for the primary (#1268).

    .venvs/lloyd/bin/python eval/engine_output_probe.py run [--label L] [--out-dir D]
    .venvs/lloyd/bin/python eval/engine_output_probe.py floor RUN.json RUN.json ... [--out F]
    .venvs/lloyd/bin/python eval/engine_output_probe.py compare --current C.json [--reference R.json] [--floor F]
    .venvs/lloyd/bin/python eval/engine_output_probe.py preempt [--load-requests N --load-prompt-words W --load-max-tokens T]

Why this exists
---------------
Every check this box runs on the primary is config-shaped:
`flash-next-bootfacts.sh` greps the boot log for the KV dtype and pool size,
`app/model_identity.py` re-reads the same two facts off `/metrics`, and
`tests/test_flash_next_launcher.py` reads launcher arguments. None of them
reads a token. AI21's two vLLM bugs (both in the Mamba state cache this
hybrid model runs on, `mamba_cache_mode="align"`) produced confident wrong
output with no crash, no warning and no error line — nothing the guardian,
which reads `server.err`, could ever see. The automod gate's tool-choice eval
cannot either: its n=20 floor is 0.05, so a 1-in-1000 corruption is ~25x under
its noise.

So this sends a committed corpus (`eval/engine_output/corpus.yaml`) at
temperature 0 with `logprobs: 1`, and keeps, per prompt, the sampled token
list and each token's logprob. A record from one engine build is the
reference for the next; `compare` reports, per prompt, top-1 agreement, the
first divergent token index, the median-logprob delta and the largest
per-token logprob delta over the agreed prefix — worst prompt first.

The threshold is a file, not a constant
---------------------------------------
Continuous batching, MTP speculative decoding and an FP8 KV cache can each
make temperature-0 output non-reproducible, so "identical" is a hypothesis to
measure, not an assumption. `floor` reads idle runs and records, per prompt,
the worst pairwise agreement and the largest deltas they produced —
`eval/engine_output/floor.json`. `compare` decides against that file and exits
2 without deciding when it is missing, exactly as `compare_tool_choice.py`
does with its noise floor.

Measured on this engine (2026-09-24, eval/measurements/engine-output-floor-
2026-09-24.md): idle greedy output is bitwise reproducible for every prompt
under ~1.8k tokens and is NOT for prompts past ~2.7k, cold or warm, so the
floor splits prompts into two tiers. An `exact` prompt is decided on
everything — one flipped token or one moved logprob is past the floor. A
`jitter` prompt is decided only on the logprobs of tokens both runs agree on,
past `jitter_margin` x its own idle maximum; where it diverges, and the median
of a different continuation, are idle noise there and would fire on an
unchanged engine.

This build returns the LEGACY completions logprobs shape
(`logprobs.tokens` / `logprobs.token_logprobs`), not the chat
`logprobs.content` list; `parse_completion` reads the legacy keys and falls
back to `content` so a future build that moves does not read as empty.

The preemption arm
------------------
`vllm:num_preemptions_total` read 0.0 on ~25,000 requests at triage, so the
preempt-and-recompute path — where AI21's scheduler-ordering bug sat — had
never executed here. `preempt` refuses to start unless the engine reports
`num_requests_running == 0`, drives concurrent long requests, runs the corpus
beside them, and writes `num_preemptions_before`, `num_preemptions_after` and
an explicit `preemptions_reached: true|false`. "Not reached" is a stated result,
never a silent pass. Reaching it on this pool (844,969 FP8 tokens,
`--max-num-seqs 8`) probably needs a KV-shrunk engine arm, which is a restart
of the only primary and a human's call.

Exit codes (the `flash-next-bootfacts.sh` convention)
-----------------------------------------------------
0  ok — a record was written / nothing diverged past the floor
1  divergence past the floor (compare); `preempt` only records — compare its
   `probe` against a reference to judge it, knowing batch variance is not in
   the idle floor
2  the instrument has nothing to decide on: engine unreachable or busy, no
   logprobs in the answer, floor or reference missing, corpora differ
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

LLOYD_HOME = Path(__file__).resolve().parent.parent
if str(LLOYD_HOME) not in sys.path:
    sys.path.insert(0, str(LLOYD_HOME))

from app.paths import EVAL_BASELINES_DIR  # noqa: E402
from app.vllm_metrics import parse_prometheus  # noqa: E402

SCHEMA = 1
DEFAULT_BASE_URL = "http://127.0.0.1:8096"
DEFAULT_MODEL = "primary"
HERE = Path(__file__).resolve().parent / "engine_output"
DEFAULT_CORPUS = HERE / "corpus.yaml"
DEFAULT_FLOOR = HERE / "floor.json"
# Routine runs are runtime data; the committed idle runs and reference live
# under eval/engine_output/ because a run was pointed there with --out-dir.
DEFAULT_OUT_DIR = EVAL_BASELINES_DIR / "engine_output"
REQUEST_TIMEOUT_S = 300.0


class ProbeRefused(Exception):
    """The instrument cannot produce an honest record. Exit 2, write nothing."""


# --------------------------------------------------------------------------
# HTTP — one tiny client, injectable so tests never touch an engine.
# --------------------------------------------------------------------------

class EngineClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = REQUEST_TIMEOUT_S):
        import httpx

        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=httpx.Timeout(timeout, connect=5.0))

    def get_text(self, path: str) -> str:
        resp = self._http.get(self.base_url + path)
        resp.raise_for_status()
        return resp.text

    def post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = self._http.post(self.base_url + path, json=body)
        resp.raise_for_status()
        return resp.json()


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------

def _expand(spec: dict[str, Any]) -> str:
    cycle = spec.get("cycle") or {}
    out = []
    for i in range(1, int(spec["count"]) + 1):
        values = {k: v[(i - 1) % len(v)] for k, v in cycle.items()}
        out.append(spec["template"].format(i=i, **values))
    return "".join(out)


def _assemble(parts: list[Any], corpus: dict[str, Any], *, depth: int = 0) -> str:
    if depth > 2:
        raise ValueError("corpus prefixes nest too deep")
    out = []
    for part in parts:
        if isinstance(part, str):
            out.append(part)
        elif isinstance(part, dict) and "prefix" in part:
            out.append(_assemble(corpus["prefixes"][part["prefix"]], corpus, depth=depth + 1))
        elif isinstance(part, dict) and "expand" in part:
            out.append(_expand(corpus["expansions"][part["expand"]]))
        else:
            raise ValueError(f"unrecognised corpus part: {part!r}")
    return "".join(out)


def load_corpus(path: Path = DEFAULT_CORPUS) -> dict[str, Any]:
    """Read and assemble the corpus. `sha256` covers the file's bytes, so any
    edit — a prompt, max_tokens, an expansion — makes old records incomparable."""
    import yaml

    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    prompts = []
    seen = set()
    for p in data["prompts"]:
        if p["id"] in seen:
            raise ValueError(f"duplicate prompt id {p['id']!r}")
        seen.add(p["id"])
        prompts.append({"id": p["id"], "tags": list(p.get("tags") or []),
                        "text": _assemble(p["parts"], data)})
    try:
        shown = str(path.resolve().relative_to(LLOYD_HOME))
    except ValueError:
        shown = str(path)
    return {"path": shown, "sha256": hashlib.sha256(raw).hexdigest(),
            "max_tokens": int(data.get("max_tokens", 48)), "prompts": prompts}


# --------------------------------------------------------------------------
# Engine identity and gauges
# --------------------------------------------------------------------------

def _gauge(series: dict[str, list[tuple[dict[str, str], float]]], name: str) -> float | None:
    rows = series.get(name)
    if not rows:
        return None
    return float(sum(v for _, v in rows))


def read_gauges(client: Any) -> dict[str, Any]:
    series = parse_prometheus(client.get_text("/metrics"))
    return {
        "num_requests_running": _gauge(series, "vllm:num_requests_running"),
        "num_requests_waiting": _gauge(series, "vllm:num_requests_waiting"),
        "num_preemptions_total": _gauge(series, "vllm:num_preemptions_total"),
        "kv_cache_usage_perc": _gauge(series, "vllm:kv_cache_usage_perc"),
        "_series": series,
    }


def find_engine_venv(port: int, proc_root: Path = Path("/proc")) -> str | None:
    """The venv of the process serving `port`, from its argv[0].

    `/metrics` does not publish which venv is running, and two venvs can serve
    this slot (SETUP.md: `vllm-flash-next-main` and its `-0910` revert), so
    read it off the live process rather than restating the supervisor conf.
    None when no readable vLLM process names that port.
    """
    needle = f"--port\x00{port}\x00"
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if b"vllm" not in raw or needle.encode() not in raw + b"\x00":
            continue
        argv0 = raw.split(b"\x00", 1)[0].decode(errors="replace")
        exe = Path(argv0)
        if exe.parent.name == "bin":
            return str(exe.parent.parent)
        return argv0
    return None


def read_engine_identity(client: Any, *, port: int | None = None) -> dict[str, Any]:
    """Name the engine a record came from. Raises ProbeRefused when the
    engine does not answer, because a record that cannot say which build it
    measured cannot be a reference for the next one."""
    try:
        metrics_text = client.get_text("/metrics")
    except Exception as exc:  # noqa: BLE001 — any transport failure is "unreachable"
        raise ProbeRefused(f"engine unreachable: /metrics failed ({type(exc).__name__}: {exc})") from exc
    series = parse_prometheus(metrics_text)
    cfg_rows = series.get("vllm:cache_config_info") or []
    labels = cfg_rows[0][0] if cfg_rows else {}

    def _int(key: str) -> int | None:
        try:
            return int(float(labels.get(key, "")))
        except (TypeError, ValueError):
            return None

    version = None
    try:
        version = json.loads(client.get_text("/version")).get("version")
    except Exception:  # noqa: BLE001 — an engine without /version still has a record
        pass
    model_root = model_name = None
    try:
        models = json.loads(client.get_text("/v1/models")).get("data") or []
        if models:
            model_root = models[0].get("root")
            model_name = models[0].get("id")
    except Exception:  # noqa: BLE001
        pass
    if port is None:
        try:
            port = int(client.base_url.rsplit(":", 1)[1].split("/", 1)[0])
        except (AttributeError, IndexError, ValueError):
            port = None
    return {
        "base_url": getattr(client, "base_url", None),
        "venv": find_engine_venv(port) if port else None,
        "vllm_version": version,
        "served_model": model_name,
        "model_root": model_root,
        "kv_cache_dtype": labels.get("cache_dtype") or None,
        "kv_cache_size_tokens": _int("kv_cache_size_tokens"),
        "block_size": _int("block_size"),
        "mamba_cache_mode": labels.get("mamba_cache_mode") or None,
        "cache_config_published": bool(cfg_rows),
    }


# --------------------------------------------------------------------------
# The probe
# --------------------------------------------------------------------------

def parse_completion(resp: dict[str, Any]) -> dict[str, Any]:
    """Tokens and sampled-token logprobs out of one /v1/completions answer.

    Reads the legacy `logprobs.tokens` / `logprobs.token_logprobs` arrays this
    build returns; falls back to the chat-style `logprobs.content` list. A
    response with neither raises: an answer with no logprobs is not a clean
    run, it is a run this instrument cannot read.
    """
    choice = (resp.get("choices") or [{}])[0]
    lp = choice.get("logprobs") or {}
    tokens = lp.get("tokens")
    logprobs = lp.get("token_logprobs")
    if tokens is None and isinstance(lp.get("content"), list):
        tokens = [c.get("token") for c in lp["content"]]
        logprobs = [c.get("logprob") for c in lp["content"]]
    if not isinstance(tokens, list) or not isinstance(logprobs, list) or len(tokens) != len(logprobs):
        raise ProbeRefused("engine returned no readable logprobs (neither tokens/token_logprobs nor content)")
    logprobs = [None if v is None else float(v) for v in logprobs]
    usage = resp.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "tokens": list(tokens),
        "token_logprobs": logprobs,
        "text": choice.get("text"),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
    }


def _median(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def probe_one(client: Any, prompt: dict[str, Any], *, model: str, max_tokens: int,
              cache_salt: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "prompt": prompt["text"], "temperature": 0,
                            "logprobs": 1, "max_tokens": max_tokens}
    if cache_salt:
        body["cache_salt"] = cache_salt
    t0 = time.monotonic()
    try:
        resp = client.post_json("/v1/completions", body)
    except Exception as exc:  # noqa: BLE001
        raise ProbeRefused(f"engine unreachable: completion for {prompt['id']!r} failed "
                           f"({type(exc).__name__}: {exc})") from exc
    parsed = parse_completion(resp)
    parsed.update({
        "id": prompt["id"], "tags": prompt["tags"],
        "prompt_sha256": hashlib.sha256(prompt["text"].encode()).hexdigest(),
        "median_logprob": _median(parsed["token_logprobs"]),
        "elapsed_s": round(time.monotonic() - t0, 3),
    })
    return parsed


def run_probe(client: Any, corpus: dict[str, Any], *, label: str, model: str = DEFAULT_MODEL,
              cache_salt_mode: str = "none", only: set[str] | None = None) -> dict[str, Any]:
    """One pass over the corpus -> one run record. Raises ProbeRefused on an
    unreachable engine, before or during the pass."""
    engine = read_engine_identity(client)
    before = read_gauges(client)
    started = dt.datetime.now(dt.timezone.utc)
    prompts = []
    for i, prompt in enumerate(corpus["prompts"]):
        if only and prompt["id"] not in only:
            continue
        salt = None
        if cache_salt_mode == "fresh":
            # a salt per request: every prefill is cold, no prefix-cache hit
            salt = f"{label}-{started.timestamp():.0f}-{i}"
        # What else the engine was running as this prompt went in: an idle
        # floor measured beside somebody's turn is a batch-variance floor.
        try:
            others = read_gauges(client)["num_requests_running"]
        except Exception:  # noqa: BLE001 — the completion below names an unreachable engine
            others = None
        row = probe_one(client, prompt, model=model, max_tokens=corpus["max_tokens"], cache_salt=salt)
        row["running_before"] = others
        prompts.append(row)
    after = read_gauges(client)
    return {
        "schema": SCHEMA,
        "kind": "engine_output_probe",
        "label": label,
        "started": started.isoformat(timespec="seconds"),
        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "engine": engine,
        "corpus": {"path": corpus["path"], "sha256": corpus["sha256"],
                   "n": len(corpus["prompts"]), "max_tokens": corpus["max_tokens"]},
        "params": {"temperature": 0, "logprobs": 1, "model": model, "cache_salt": cache_salt_mode},
        "gauges_before": {k: v for k, v in before.items() if not k.startswith("_")},
        "gauges_after": {k: v for k, v in after.items() if not k.startswith("_")},
        "prompts": prompts,
    }


def write_record(record: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = record["started"].replace(":", "").replace("-", "").replace("+0000", "Z")
    path = out_dir / f"{stamp}_{record['label']}.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    return path


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def compare_prompt(ref: dict[str, Any], cur: dict[str, Any]) -> dict[str, Any]:
    """Per-prompt distance between two answers to the same prompt.

    agreement         common-prefix tokens / the longer answer (1.0 = identical)
    first_divergence  index of the first differing token, None when identical
    median_lp_delta   |median sampled logprob, current - reference|
    token_lp_delta    largest |logprob delta| over the agreed prefix — the one
                      a single shifted logprob moves even when the median does not
    """
    rt, ct = ref["tokens"], cur["tokens"]
    common = 0
    for a, b in zip(rt, ct):
        if a != b:
            break
        common += 1
    longest = max(len(rt), len(ct))
    identical = common == len(rt) == len(ct)
    deltas = [abs(a - b) for a, b in zip(ref["token_logprobs"][:common], cur["token_logprobs"][:common])
              if a is not None and b is not None]
    rm, cm = _median(ref["token_logprobs"]), _median(cur["token_logprobs"])
    return {
        "id": ref["id"],
        "agreement": 1.0 if longest == 0 else common / longest,
        "first_divergence": None if identical else common,
        "median_lp_delta": abs(rm - cm) if rm is not None and cm is not None else None,
        "token_lp_delta": max(deltas) if deltas else 0.0,
        "ref_tokens": len(rt),
        "cur_tokens": len(ct),
    }


def _by_id(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {p["id"]: p for p in record["prompts"]}


def build_floor(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """The idle floor: every pairwise comparison among `runs`, reduced per
    prompt to its worst case. Refuses runs over different corpora or engines,
    since a floor that mixes builds measures the build change, not the noise."""
    if len(runs) < 2:
        raise ProbeRefused("a floor needs at least two runs")
    shas = {r["corpus"]["sha256"] for r in runs}
    if len(shas) != 1:
        raise ProbeRefused("floor runs span different corpora")
    engines = {json.dumps({k: r["engine"].get(k) for k in ("venv", "vllm_version", "kv_cache_dtype",
                                                           "kv_cache_size_tokens")}, sort_keys=True)
               for r in runs}
    if len(engines) != 1:
        raise ProbeRefused("floor runs span different engines")
    ids = [p["id"] for p in runs[0]["prompts"]]
    per: dict[str, dict[str, Any]] = {i: {"agreement_min": 1.0, "median_lp_delta_max": 0.0,
                                          "token_lp_delta_max": 0.0, "first_divergence_min": None,
                                          "pairs_same_tokens": 0, "pairs_bitwise": 0, "pairs": 0}
                                      for i in ids}
    for a, b in itertools.combinations(runs, 2):
        ma, mb = _by_id(a), _by_id(b)
        for pid in ids:
            if pid not in ma or pid not in mb:
                raise ProbeRefused(f"floor run is missing prompt {pid!r}")
            c = compare_prompt(ma[pid], mb[pid])
            f = per[pid]
            f["pairs"] += 1
            f["agreement_min"] = min(f["agreement_min"], c["agreement"])
            f["median_lp_delta_max"] = max(f["median_lp_delta_max"], c["median_lp_delta"] or 0.0)
            f["token_lp_delta_max"] = max(f["token_lp_delta_max"], c["token_lp_delta"])
            if c["first_divergence"] is None:
                f["pairs_same_tokens"] += 1
                if c["token_lp_delta"] == 0.0 and not c["median_lp_delta"]:
                    f["pairs_bitwise"] += 1
            elif f["first_divergence_min"] is None or c["first_divergence"] < f["first_divergence_min"]:
                f["first_divergence_min"] = c["first_divergence"]
    # "bitwise": same tokens AND the same logprobs in every pair; "same
    # tokens" allows the logprobs to jitter while greedy output holds.
    reproducible = sum(1 for f in per.values() if f["pairs_bitwise"] == f["pairs"])
    same_tokens = sum(1 for f in per.values() if f["pairs_same_tokens"] == f["pairs"])
    for f in per.values():
        f["tier"] = "exact" if f["pairs_bitwise"] == f["pairs"] else "jitter"
    margin, loo_ratio = _jitter_margin(runs, per)
    return {
        "schema": SCHEMA,
        "kind": "engine_output_floor",
        "made": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "runs": [r["label"] for r in runs],
        "n_runs": len(runs),
        "corpus_sha256": runs[0]["corpus"]["sha256"],
        "engine": runs[0]["engine"],
        "params": runs[0]["params"],
        "summary": {
            "prompts": len(ids),
            "prompts_bitwise_reproducible": reproducible,
            "prompts_token_reproducible": same_tokens,
            "agreement_min": min(f["agreement_min"] for f in per.values()),
            "median_lp_delta_max": max(f["median_lp_delta_max"] for f in per.values()),
            "token_lp_delta_max": max(f["token_lp_delta_max"] for f in per.values()),
        },
        "cache_salt_modes": sorted({str(r["params"].get("cache_salt")) for r in runs}),
        # A jitter prompt is past its floor when its token-logprob delta beats
        # `jitter_margin` x its own idle maximum. The margin is measured, not
        # chosen: leave-one-out over these runs, the worst held-out ratio, x1.25.
        "jitter_margin": margin,
        "jitter_loo_worst_ratio": loo_ratio,
        "prompts": per,
    }


def _jitter_margin(runs: list[dict[str, Any]], per: dict[str, dict[str, Any]]) -> tuple[float, float | None]:
    """Leave-one-out: for each held-out run, a floor from the others, and the
    held-out run's token-logprob delta against each of them as a multiple of
    that floor. The worst ratio is how far a fresh idle run landed past a
    floor measured without it; x1.25, rounded up to 0.5, is the margin."""
    jitter = [pid for pid, f in per.items() if f["tier"] == "jitter"]
    if not jitter or len(runs) < 3:
        return 1.0, None
    worst = 0.0
    for i, held in enumerate(runs):
        rest = runs[:i] + runs[i + 1:]
        rest_max: dict[str, float] = {pid: 0.0 for pid in jitter}
        for a, b in itertools.combinations(rest, 2):
            ma, mb = _by_id(a), _by_id(b)
            for pid in jitter:
                rest_max[pid] = max(rest_max[pid], compare_prompt(ma[pid], mb[pid])["token_lp_delta"])
        mh = _by_id(held)
        for ref in rest:
            mr = _by_id(ref)
            for pid in jitter:
                d = compare_prompt(mr[pid], mh[pid])["token_lp_delta"]
                if rest_max[pid] > 0:
                    worst = max(worst, d / rest_max[pid])
    return max(1.0, -(-worst * 1.25 // 0.5) * 0.5), round(worst, 4)


def compare_records(reference: dict[str, Any], current: dict[str, Any],
                    floor: dict[str, Any]) -> dict[str, Any]:
    """Every prompt against the reference, each decided against its OWN floor
    row. A prompt past its floor on any axis is `exceeds`. Rows are sorted
    worst first: exceeders, then lowest agreement, then largest token delta."""
    if reference["corpus"]["sha256"] != current["corpus"]["sha256"]:
        raise ProbeRefused("reference and current were run over different corpora")
    if floor.get("corpus_sha256") != current["corpus"]["sha256"]:
        raise ProbeRefused("floor was measured over a different corpus; re-measure it")
    ref, cur = _by_id(reference), _by_id(current)
    rows = []
    for pid, r in ref.items():
        if pid not in cur:
            raise ProbeRefused(f"current record has no answer for {pid!r}")
        f = floor["prompts"].get(pid)
        if f is None:
            raise ProbeRefused(f"floor has no row for {pid!r}")
        c = compare_prompt(r, cur[pid])
        c["tier"] = f.get("tier", "exact")
        reasons = []
        if c["tier"] == "exact":
            # Bitwise reproducible idle: the floor row's numbers are the threshold.
            if c["agreement"] < f["agreement_min"]:
                reasons.append(f"agreement {c['agreement']:.3f} < floor {f['agreement_min']:.3f}")
            if (c["median_lp_delta"] or 0.0) > f["median_lp_delta_max"]:
                reasons.append(f"median logprob delta {c['median_lp_delta']:.4g} > floor "
                               f"{f['median_lp_delta_max']:.4g}")
            if c["token_lp_delta"] > f["token_lp_delta_max"]:
                reasons.append(f"token logprob delta {c['token_lp_delta']:.4g} > floor "
                               f"{f['token_lp_delta_max']:.4g}")
        else:
            # Not reproducible idle: where a later token diverges, and the
            # median of a different continuation, are noise here (measured —
            # leave-one-out, a max-of-pairs floor on those axes fired on 66 of
            # 90 unchanged-engine comparisons). What stays decidable is the
            # logprob of tokens both runs agree on, past the margin. Not the
            # first token either: a held-out idle run flipped sys_summary's
            # first token although none of the 45 floor pairs had.
            margin = floor.get("jitter_margin", 1.0)
            if c["token_lp_delta"] > margin * f["token_lp_delta_max"]:
                reasons.append(f"token logprob delta {c['token_lp_delta']:.4g} > "
                               f"{margin:g} x floor {f['token_lp_delta_max']:.4g}")
        c["exceeds"] = bool(reasons)
        c["reasons"] = reasons
        rows.append(c)
    rows.sort(key=lambda c: (not c["exceeds"], c["agreement"], -c["token_lp_delta"],
                             -(c["median_lp_delta"] or 0.0)))
    return {
        "reference": reference["label"], "current": current["label"],
        "reference_engine": reference["engine"], "current_engine": current["engine"],
        "diverged": sum(1 for c in rows if c["exceeds"]),
        "rows": rows,
    }


def render_comparison(result: dict[str, Any]) -> str:
    re_, ce = result["reference_engine"], result["current_engine"]
    lines = [
        f"reference {result['reference']}: {re_.get('venv')} vllm {re_.get('vllm_version')} "
        f"kv {re_.get('kv_cache_dtype')} pool {re_.get('kv_cache_size_tokens')}",
        f"current   {result['current']}: {ce.get('venv')} vllm {ce.get('vllm_version')} "
        f"kv {ce.get('kv_cache_dtype')} pool {ce.get('kv_cache_size_tokens')}",
        f"{result['diverged']} of {len(result['rows'])} prompts past their idle floor",
        "",
        f"{'prompt':<20} {'tier':<6} {'agree':>6} {'1st div':>7} {'d median lp':>12} {'d token lp':>11}  verdict",
    ]
    for c in result["rows"]:
        fd = "-" if c["first_divergence"] is None else str(c["first_divergence"])
        md = "-" if c["median_lp_delta"] is None else f"{c['median_lp_delta']:.4g}"
        lines.append(f"{c['id']:<20} {c.get('tier', 'exact'):<6} {c['agreement']:>6.3f} {fd:>7} {md:>12} {c['token_lp_delta']:>11.4g}  "
                     + ("PAST FLOOR: " + "; ".join(c["reasons"]) if c["exceeds"] else "within floor"))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Preemption arm
# --------------------------------------------------------------------------

_LOAD_WORDS = ("amber basalt cobalt delta ember fjord granite harbor indigo jasper kelp lumen "
               "meadow nickel onyx prism quartz russet sierra tundra umber vector willow xenon "
               "yarrow zephyr").split()


def load_prompt(index: int, words: int) -> str:
    """A unique long prompt per load request, so no two share a cached prefix
    (a shared prefix is one KV copy, which is the opposite of pressure)."""
    n = len(_LOAD_WORDS)
    body = " ".join(_LOAD_WORDS[(index * 7 + i * (index + 3)) % n] + str((i * 31 + index) % 997)
                    for i in range(words))
    return f"Load request {index}. Repeat the following list back verbatim.\n{body}\n"


def run_preempt(client: Any, corpus: dict[str, Any], *, label: str, model: str = DEFAULT_MODEL,
                load_requests: int = 8, load_prompt_words: int = 20000, load_max_tokens: int = 4000,
                poll_s: float = 1.0, settle_s: float = 5.0,
                sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Drive concurrent long requests, run the corpus beside them, and record
    whether the preempt path was reached. Refuses a busy engine (exit 2): this
    stresses the only primary, and a live turn caught under it would be the
    turn that pays."""
    engine = read_engine_identity(client)
    before = read_gauges(client)
    running = before["num_requests_running"]
    if running is None:
        raise ProbeRefused("engine publishes no vllm:num_requests_running; cannot prove it idle")
    if running != 0 or (before["num_requests_waiting"] or 0) != 0:
        raise ProbeRefused(f"engine busy: num_requests_running={running:g}, "
                           f"num_requests_waiting={before['num_requests_waiting'] or 0:g}; refusing to load it")

    peak = {"kv_cache_usage_perc": before["kv_cache_usage_perc"] or 0.0,
            "num_requests_running": 0.0, "num_requests_waiting": 0.0}
    stop = threading.Event()

    def _watch() -> None:
        while not stop.is_set():
            try:
                g = read_gauges(client)
            except Exception:  # noqa: BLE001 — a missed poll loses a sample, not the run
                g = {}
            for k in peak:
                if g.get(k) is not None:
                    peak[k] = max(peak[k], g[k])
            stop.wait(poll_s)

    def _load(i: int) -> dict[str, Any]:
        body = {"model": model, "prompt": load_prompt(i, load_prompt_words), "temperature": 0,
                "max_tokens": load_max_tokens, "ignore_eos": True, "logprobs": 1}
        t0 = time.monotonic()
        try:
            resp = client.post_json("/v1/completions", body)
            p = parse_completion(resp)
            return {"index": i, "ok": True, "finish_reason": p["finish_reason"],
                    "prompt_tokens": p["prompt_tokens"], "completion_tokens": len(p["tokens"]),
                    "median_logprob": _median(p["token_logprobs"]),
                    "elapsed_s": round(time.monotonic() - t0, 1)}
        except Exception as exc:  # noqa: BLE001
            return {"index": i, "ok": False, "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_s": round(time.monotonic() - t0, 1)}

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()
    probe_record = None
    probe_error = None
    with ThreadPoolExecutor(max_workers=max(1, load_requests)) as pool:
        futures = [pool.submit(_load, i) for i in range(load_requests)]
        sleep(settle_s)  # let the load be admitted before the corpus joins it
        try:
            probe_record = run_probe(client, corpus, label=f"{label}-under-load", model=model)
        except ProbeRefused as exc:
            probe_error = str(exc)
        load_results = [f.result() for f in futures]
    stop.set()
    watcher.join(timeout=poll_s * 3 + 1)
    after = read_gauges(client)

    pb, pa = before["num_preemptions_total"], after["num_preemptions_total"]
    reached = pb is not None and pa is not None and pa > pb
    return {
        "schema": SCHEMA,
        "kind": "engine_output_preempt",
        "label": label,
        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "engine": engine,
        "load": {"requests": load_requests, "prompt_words": load_prompt_words,
                 "max_tokens": load_max_tokens, "results": load_results},
        "num_preemptions_before": pb,
        "num_preemptions_after": pa,
        "preemptions_reached": reached,
        "peak": peak,
        "probe": probe_record,
        "probe_error": probe_error,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _latest_reference(out_dir: Path, exclude: Path) -> Path | None:
    cands = sorted(p for p in out_dir.glob("*.json") if p.resolve() != exclude.resolve())
    return cands[-1] if cands else None


def main(argv: list[str] | None = None, *, client_factory: Callable[[str], Any] = EngineClient) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--base-url", default=DEFAULT_BASE_URL)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
        p.add_argument("--label", default="probe")
        p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    p_run = sub.add_parser("run", help="one pass over the corpus -> one run record")
    _common(p_run)
    p_run.add_argument("--cache-salt", choices=("none", "fresh"), default="none",
                       help="fresh: a salt per request, so every prefill is cold")
    p_run.add_argument("--only", action="append", help="probe only this prompt id (repeatable)")

    p_floor = sub.add_parser("floor", help="reduce idle runs to the floor file")
    p_floor.add_argument("runs", nargs="+", type=Path)
    p_floor.add_argument("--out", type=Path, default=DEFAULT_FLOOR)

    p_cmp = sub.add_parser("compare", help="current record vs reference, decided against the floor")
    p_cmp.add_argument("--current", type=Path, required=True)
    p_cmp.add_argument("--reference", type=Path,
                       help="default: the newest other record in the current's directory")
    p_cmp.add_argument("--floor", type=Path, default=DEFAULT_FLOOR)
    p_cmp.add_argument("--json", action="store_true")

    p_pre = sub.add_parser("preempt", help="load driver + corpus under load, preemptions recorded")
    _common(p_pre)
    p_pre.add_argument("--load-requests", type=int, default=8)
    p_pre.add_argument("--load-prompt-words", type=int, default=20000)
    p_pre.add_argument("--load-max-tokens", type=int, default=4000)
    p_pre.add_argument("--settle-s", type=float, default=5.0)

    args = ap.parse_args(argv)
    try:
        if args.cmd == "run":
            client = client_factory(args.base_url)
            record = run_probe(client, load_corpus(args.corpus), label=args.label, model=args.model,
                               cache_salt_mode=args.cache_salt, only=set(args.only) if args.only else None)
            path = write_record(record, args.out_dir)
            print(f"wrote {path}  ({len(record['prompts'])} prompts, engine "
                  f"{record['engine'].get('vllm_version')} {record['engine'].get('kv_cache_dtype')})")
            return 0
        if args.cmd == "floor":
            floor = build_floor([_load_json(p) for p in args.runs])
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(floor, indent=1) + "\n")
            s = floor["summary"]
            print(f"wrote {args.out}: {s['prompts_bitwise_reproducible']}/{s['prompts']} prompts bitwise "
                  f"reproducible, {s['prompts_token_reproducible']}/{s['prompts']} same tokens, over "
                  f"{floor['n_runs']} runs; worst agreement {s['agreement_min']:.3f}, "
                  f"median-lp spread {s['median_lp_delta_max']:.4g}, token-lp spread {s['token_lp_delta_max']:.4g}")
            return 0
        if args.cmd == "compare":
            if not args.floor.exists():
                raise ProbeRefused(f"no floor file at {args.floor}; measure one with `floor` before deciding")
            if not args.current.exists():
                raise ProbeRefused(f"no current record at {args.current}")
            ref_path = args.reference or _latest_reference(args.current.parent, args.current)
            if ref_path is None or not ref_path.exists():
                raise ProbeRefused("no reference record to compare against")
            result = compare_records(_load_json(ref_path), _load_json(args.current), _load_json(args.floor))
            print(json.dumps(result, indent=1) if args.json else render_comparison(result))
            return 1 if result["diverged"] else 0
        if args.cmd == "preempt":
            client = client_factory(args.base_url)
            record = run_preempt(client, load_corpus(args.corpus), label=args.label, model=args.model,
                                 load_requests=args.load_requests, load_prompt_words=args.load_prompt_words,
                                 load_max_tokens=args.load_max_tokens, settle_s=args.settle_s)
            args.out_dir.mkdir(parents=True, exist_ok=True)
            path = args.out_dir / f"preempt_{record['finished'].replace(':', '')}_{args.label}.json"
            path.write_text(json.dumps(record, indent=1) + "\n")
            print(f"wrote {path}: preemptions {record['num_preemptions_before']} -> "
                  f"{record['num_preemptions_after']}, preemptions_reached={str(record['preemptions_reached']).lower()}, "
                  f"peak kv {record['peak']['kv_cache_usage_perc']:.3f}")
            if record["probe"] is None:
                raise ProbeRefused(f"corpus under load did not complete: {record['probe_error']}")
            return 0
    except ProbeRefused as exc:
        print(f"engine_output_probe: cannot decide — {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
