#!/usr/bin/env python3
"""Does djev follow an option's NAME or its DEFINITION? (#1452)

    .venvs/lloyd/bin/python eval/djev_name_prior_probe.py                  # pinned corpus -> eval/djev/name_prior_<date>.json
    .venvs/lloyd/bin/python eval/djev_name_prior_probe.py --samples 3 --corpus eval/djev/name_prior_corpus.jsonl
    .venvs/lloyd/bin/python eval/djev_name_prior_probe.py --build-corpus eval/djev/name_prior_corpus.jsonl

The name-prior experiment of *Type-Safe Is Not Error-Free* (arXiv 2609.26758),
run against Lloyd's own typed-decision engine. That paper holds each option's
rubric byte-identical, renames the option keys, and finds a trained decision
head's argmax follows the conventional meaning of the label rather than the
definition bound to it. djev is a prompted general model pressed into typed
output, not a trained head, so whether the effect transfers is an empirical
question — and every production djev seam passes semantically loaded keys
(`different`/`same`, ten edge-type names, `RANK_LEVELS`).

For every prompt in the pinned corpus the probe sends one `choice` question
four times, `state` and `instructions` fixed and the option DESCRIPTIONS
byte-identical in the same order every time:

    control         the production keys
    nonce           `opt_1..opt_N` — the neutral control
    inverted        the production keys reversed across the descriptions, so
                    `same` is bound to "different findings..." and
                    `directly answers it` to "has nothing to do with the query"
    control_repeat  the control request again, byte for byte

and compares argmax POSITIONS (which description won), never keys and never
probabilities. Three comparisons, each a flip rate with a Wilson 95% CI:

    repeat    control vs control_repeat — the noise floor. #1357 measured label
              logprobs moving 1–8+ nats between byte-identical requests and a
              cold read flipping top-1, so a name effect is only a name effect
              when it flips MORE than a replay does. `exceeds_repeat_floor` is
              true only when the treatment's CI lies wholly above the repeat's.
    nonce     control vs nonce
    inverted  control vs inverted

The repeat is sent LAST for each prompt, so its cache distance from the control
read is at least as large as either treatment's — a floor measured warmer than
the treatments would understate the noise and credit it to the name.

Design rules, each one a way this probe could otherwise lie:

* **Argmax only.** djev's `probabilities` are renormalised over the label set
  and look confident whatever happened (`architecture/djev.md` §3.2), and the
  pilot on this item saw .98 on an inverted answer and .99997 on an aligned
  one. The report carries argmax positions and counts, and no probability,
  confidence or score anywhere — there is no magnitude in it to compare.
* **A tie is not a decision.** A missing answer, a probability vector that does
  not name exactly the variant's keys, or a top two within `TIE_EPS` makes that
  read *indeterminate*; a pair with an indeterminate side is counted under
  `ties` and left out of the flip denominator. `flip_rate` is null over zero
  decided pairs, never 0.
* **A dead engine is never 0 flips.** `:8011/health` is asked first; no answer,
  or a connection that dies mid-run, prints `engine unreachable`, writes no
  report and exits 3. This is deliberately the opposite of
  `scripts/djev_determinism_probe.py`, which exits 0 there because a bisect on
  a host with no djev has nothing to fail; here a missing report must be
  distinguishable from a clean one.
* **The rank shape is posed as `choice`.** Production ranks with `score`, whose
  criteria are bare level names with no description to hold fixed, so the
  probe asks the same four `RANK_LEVELS` in the same worst-to-best order as a
  `choice` with a one-line definition each. It measures whether those NAMES
  pull the answer, which is the property a nonce-name policy would change.

Scope: this measures. Whether production call sites adopt nonce option names
if the flips are above the floor is a separate, human decision (#1452's human
clause 2) — it renames `RANK_LEVELS` and moves the live recall ranker.

The pinned corpus is `eval/djev/name_prior_corpus.jsonl`, built once from the
seam rows in `~/.local/state/lloyd-djev/shadow.jsonl` (read only) plus the edges
replay corpus by `--build-corpus`; until that file exists the probe falls back
to the committed synthetic fixture `eval/djev/name_prior_fixture.jsonl` and the
report says so in `corpus.kind`. The engine is read-only here (no restart, no
variant boot), but it is GPU 2 and shared with the production recall ranker:
do not run it while something holds `regression.lock`.

Exit codes: 0 report written, 2 bad corpus or an answer not in the shape
expected (no report), 3 engine unreachable (no report).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
LLOYD_HOME = HERE.parent
if str(LLOYD_HOME) not in sys.path:
    sys.path.insert(0, str(LLOYD_HOME))

from app import djev  # noqa: E402
from eval.djev import schemas  # noqa: E402
from eval.stats import wilson_ci  # noqa: E402

PROBE = "djev_name_prior"
SCHEMA_VERSION = 1

DEFAULT_URL = djev.DEFAULT_STRUCTURED_URL
DEFAULT_SAMPLES = 3
DEFAULT_TIMEOUT_S = 60.0
REPORT_DIR = HERE / "djev"
PINNED_CORPUS = REPORT_DIR / "name_prior_corpus.jsonl"
FIXTURE_CORPUS = REPORT_DIR / "name_prior_fixture.jsonl"

#: The corpus floor the item's acceptance names. Below it a Wilson interval on
#: a flip rate is too wide to separate a treatment from the repeat floor.
MIN_PROMPTS = 30

#: Two label probabilities closer than this are a tie, not a decision. The
#: server renormalises over the label set, so an exact 0.5/0.5 is what a read
#: with no opinion looks like; 1e-6 absorbs float noise and nothing more.
TIE_EPS = 1e-6

VARIANTS = ("control", "nonce", "inverted", "control_repeat")
#: (comparison name, treatment variant). Each is judged against `control`.
COMPARISONS = (("repeat", "control_repeat"), ("nonce", "nonce"),
               ("inverted", "inverted"))

QID = "decision"


# ---------------------------------------------------------------------------
# The four seam shapes, each one question with the production keys in order
# ---------------------------------------------------------------------------

def _from_schema(schema: schemas.Schema) -> dict:
    q = next(iter(schema.spec.values()))
    return {"instructions": q["instructions"], "criteria": dict(q["criteria"])}


#: One-line definitions for the rank levels. Production's `score` question has
#: none (its criteria are the bare names), so these are the probe's own, held
#: byte-identical across every variant — what is under test is the NAMES.
RANK_DESCRIPTIONS = (
    "the candidate has nothing to do with the query",
    "the candidate touches the query's topic but does not address what was asked",
    "the candidate answers part of what was asked",
    "the candidate answers what was asked",
)

SHAPES: dict[str, dict] = {
    "dedupe": _from_schema(schemas.DEDUPE),
    "entity": _from_schema(schemas.ENTITY),
    "edges": _from_schema(schemas.EDGES),
    "rank": {
        # The production wording (`app.djev.rank_questions`) for one candidate.
        "instructions": "How well does candidate [0] answer the query?",
        "criteria": dict(zip(djev.RANK_LEVELS, RANK_DESCRIPTIONS)),
    },
}


def variant_keys(shape: str, variant: str) -> list[str]:
    """The option keys a variant binds to the shape's descriptions, in order."""
    keys = list(SHAPES[shape]["criteria"])
    if variant in ("control", "control_repeat"):
        return keys
    if variant == "nonce":
        return [f"opt_{i + 1}" for i in range(len(keys))]
    if variant == "inverted":
        return list(reversed(keys))
    raise ValueError(f"unknown variant {variant!r}")


def paired_schemas(shape: str) -> dict[str, dict]:
    """`{variant: question}` for one shape — descriptions identical, keys not."""
    base = SHAPES[shape]
    descriptions = list(base["criteria"].values())
    out = {}
    for variant in VARIANTS:
        keys = variant_keys(shape, variant)
        out[variant] = {"type": "choice", "instructions": base["instructions"],
                        "criteria": dict(zip(keys, descriptions))}
    return out


def request_body(state: str, question: Mapping, *, samples: int) -> dict:
    return {"model": "djev", "state": state, "questions": {QID: dict(question)},
            "samples": samples}


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

class CorpusError(Exception):
    pass


def load_corpus(path: Path) -> list[dict]:
    rows, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise CorpusError(f"{path}:{n}: not JSON ({exc})") from None
        rid, shape, state = row.get("id"), row.get("shape"), row.get("state")
        if not rid or rid in seen:
            raise CorpusError(f"{path}:{n}: missing or duplicate id {rid!r}")
        if shape not in SHAPES:
            raise CorpusError(f"{path}:{n}: unknown shape {shape!r}")
        if not isinstance(state, str) or not state.strip():
            raise CorpusError(f"{path}:{n}: empty state")
        seen.add(rid)
        rows.append(row)
    if len(rows) < MIN_PROMPTS:
        raise CorpusError(f"{path}: {len(rows)} prompts, the probe needs >= {MIN_PROMPTS}")
    missing = sorted(set(SHAPES) - {r["shape"] for r in rows})
    if missing:
        raise CorpusError(f"{path}: no prompt of shape {', '.join(missing)}")
    return rows


def default_corpus() -> tuple[Path, str]:
    if PINNED_CORPUS.exists():
        return PINNED_CORPUS, "pinned"
    return FIXTURE_CORPUS, "fixture"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

class Unreachable(Exception):
    """Nothing answered: refused, reset, timed out, or DNS."""


def _http(method: str, url: str, body: dict | None = None, *,
          timeout: float) -> tuple[int, bytes]:
    """`(status, body)`, or `Unreachable`. An answered non-200 is a status."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read() or b""
    except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError,
            OSError) as exc:
        raise Unreachable(str(getattr(exc, "reason", exc))) from None


def argmax_position(payload: Mapping, keys: Sequence[str]) -> tuple[int | None, str]:
    """`(position, "")`, or `(None, why)` when the read is indeterminate.

    The position is into the DESCRIPTIONS, which is what makes variants with
    different keys comparable. Nothing about the probabilities leaves here.
    """
    ans = ((payload or {}).get("answers") or {}).get(QID)
    if not isinstance(ans, Mapping):
        return None, "no_answer"
    probs = ans.get("probabilities")
    if not isinstance(probs, Mapping) or set(probs) != set(keys):
        return None, "label_set"
    try:
        vec = [float(probs[k]) for k in keys]
    except (TypeError, ValueError):
        return None, "label_set"
    order = sorted(range(len(vec)), key=lambda i: vec[i], reverse=True)
    if len(order) > 1 and vec[order[0]] - vec[order[1]] <= TIE_EPS:
        return None, "tie"
    return order[0], ""


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _ci(k: int, n: int) -> list[float] | None:
    if n <= 0:
        return None
    lo, hi = wilson_ci(k, n)
    return [round(lo, 4), round(hi, 4)]


def compare(rows: Sequence[Mapping], treatment: str) -> dict:
    """Flip counts on argmax position, control vs `treatment`."""
    n = ties = flips = 0
    for r in rows:
        a, b = r["argmax"].get("control"), r["argmax"].get(treatment)
        n += 1
        if a is None or b is None:
            ties += 1
            continue
        flips += int(a != b)
    decided = n - ties
    return {"n": n, "decided": decided, "ties": ties, "flips": flips,
            "flip_rate": round(flips / decided, 4) if decided else None,
            "ci95": _ci(flips, decided)}


def summarise(rows: Sequence[Mapping]) -> dict:
    comps = {name: compare(rows, variant) for name, variant in COMPARISONS}
    floor = comps["repeat"]["ci95"]
    for name in ("nonce", "inverted"):
        ci = comps[name]["ci95"]
        comps[name]["exceeds_repeat_floor"] = (
            None if ci is None or floor is None else ci[0] > floor[1])
    return comps


# ---------------------------------------------------------------------------
# The report and its schema
# ---------------------------------------------------------------------------

#: The documented output shape. `validate_report` is its only reader, and the
#: test grades a report built through `main` against it, so the two cannot
#: drift. `None` in a type tuple means the value may be null.
_STATS = {"n": (int,), "decided": (int,), "ties": (int,), "flips": (int,),
          "flip_rate": (float, int, type(None)), "ci95": (list, type(None))}
REPORT_SCHEMA: dict[str, Any] = {
    "probe": (str,), "schema_version": (int,), "date": (str,), "generated_at": (str,),
    "engine": {"url": (str,), "samples": (int,)},
    "corpus": {"path": (str,), "kind": (str,), "sha256": (str,), "n_prompts": (int,),
               "by_shape": (dict,)},
    "method": {"variants": (list,), "tie_eps": (float,), "argmax_only": (bool,)},
    "comparisons": {name: _STATS for name, _ in COMPARISONS},
    "by_shape": (dict,),
    "indeterminate": (dict,),
    "rows": (list,),
    "notes": (list,),
}
_ROW_KEYS = {"id", "shape", "argmax", "indeterminate"}
#: Words that would mean the report carries a magnitude. Argmax only.
_MAGNITUDE_KEYS = {"probabilities", "probability", "confidence", "score", "scores",
                   "label_mass", "logprob", "logprobs"}


def _check(obj: Any, spec: Any, where: str, errs: list[str]) -> None:
    if isinstance(spec, dict):
        if not isinstance(obj, dict):
            errs.append(f"{where}: expected an object")
            return
        for key, sub in spec.items():
            if key not in obj:
                errs.append(f"{where}.{key}: missing")
            else:
                _check(obj[key], sub, f"{where}.{key}", errs)
    elif not isinstance(obj, spec) or (isinstance(obj, bool) and bool not in spec):
        errs.append(f"{where}: {type(obj).__name__} is not {'/'.join(t.__name__ for t in spec)}")


def _keys_anywhere(obj: Any) -> Iterable[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys_anywhere(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys_anywhere(v)


def validate_report(report: Any) -> list[str]:
    """Every way `report` departs from `REPORT_SCHEMA`; empty when it conforms."""
    errs: list[str] = []
    _check(report, REPORT_SCHEMA, "report", errs)
    if errs:
        return errs
    if report["probe"] != PROBE:
        errs.append(f"report.probe: {report['probe']!r}")
    if len(report["rows"]) != report["corpus"]["n_prompts"]:
        errs.append("report.rows: one row per corpus prompt")
    for i, row in enumerate(report["rows"]):
        if not isinstance(row, dict) or set(row) != _ROW_KEYS:
            errs.append(f"report.rows[{i}]: keys must be {sorted(_ROW_KEYS)}")
            continue
        if set(row["argmax"]) != set(VARIANTS):
            errs.append(f"report.rows[{i}].argmax: one entry per variant")
    for name, stats in report["comparisons"].items():
        if stats["decided"] + stats["ties"] != stats["n"]:
            errs.append(f"report.comparisons.{name}: decided + ties != n")
        if (stats["flip_rate"] is None) != (stats["decided"] == 0):
            errs.append(f"report.comparisons.{name}: flip_rate is null iff nothing decided")
    leaked = _MAGNITUDE_KEYS & set(_keys_anywhere(report))
    if leaked:
        errs.append(f"report carries magnitudes: {sorted(leaked)}")
    return errs


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

class BadAnswer(Exception):
    pass


def run(corpus: Sequence[Mapping], *, url: str, samples: int,
        timeout: float, log: Callable[[str], None] = print) -> list[dict]:
    """One row per prompt: argmax position per variant. Raises `Unreachable`
    or `BadAnswer`; either one means no report."""
    out = []
    for i, prompt in enumerate(corpus):
        qs = paired_schemas(prompt["shape"])
        row = {"id": prompt["id"], "shape": prompt["shape"], "argmax": {},
               "indeterminate": {}}
        for variant in VARIANTS:  # control_repeat last, on purpose
            status, raw = _http("POST", url + "/v1/systemone",
                                request_body(prompt["state"], qs[variant], samples=samples),
                                timeout=timeout)
            if status != 200:
                raise BadAnswer(f"{prompt['id']} {variant}: HTTP {status}")
            try:
                payload = json.loads(raw)
            except ValueError:
                raise BadAnswer(f"{prompt['id']} {variant}: response is not JSON") from None
            pos, why = argmax_position(payload, variant_keys(prompt["shape"], variant))
            row["argmax"][variant] = pos
            if why:
                row["indeterminate"][variant] = why
        out.append(row)
        log(f"  [{i + 1:>3}/{len(corpus)}] {prompt['shape']:<6} {prompt['id']:<24} "
            + " ".join(f"{v}={row['argmax'][v]}" for v in VARIANTS))
    return out


def build_report(rows: Sequence[Mapping], *, corpus_path: Path, corpus_kind: str,
                 url: str, samples: int, when: dt.datetime) -> dict:
    by_shape_count: dict[str, int] = {}
    for r in rows:
        by_shape_count[r["shape"]] = by_shape_count.get(r["shape"], 0) + 1
    indeterminate: dict[str, dict[str, int]] = {v: {} for v in VARIANTS}
    for r in rows:
        for v, why in r["indeterminate"].items():
            indeterminate[v][why] = indeterminate[v].get(why, 0) + 1
    notes = [
        "Flip = the winning DESCRIPTION differs from control's; argmax only, "
        "no score magnitudes are compared or reported.",
        "A name effect is claimed only where exceeds_repeat_floor is true: the "
        "treatment's Wilson 95% CI lies wholly above the control-vs-control repeat's.",
    ]
    if corpus_kind != "pinned":
        notes.append("Corpus is the synthetic fixture, not the pinned seam corpus; "
                     "build it with --build-corpus before citing these numbers.")
    return {
        "probe": PROBE, "schema_version": SCHEMA_VERSION,
        "date": when.date().isoformat(),
        "generated_at": when.isoformat(timespec="seconds"),
        "engine": {"url": url, "samples": samples},
        "corpus": {"path": _rel(corpus_path), "kind": corpus_kind,
                   "sha256": _sha256(corpus_path), "n_prompts": len(rows),
                   "by_shape": by_shape_count},
        "method": {"variants": list(VARIANTS), "tie_eps": TIE_EPS, "argmax_only": True},
        "comparisons": summarise(rows),
        "by_shape": {s: summarise([r for r in rows if r["shape"] == s])
                     for s in sorted(by_shape_count)},
        "indeterminate": indeterminate,
        "rows": [dict(r) for r in rows],
        "notes": notes,
    }


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(LLOYD_HOME))
    except ValueError:
        return str(path)


def report_path(out_dir: Path, date: str) -> Path:
    return out_dir / f"name_prior_{date}.json"


def _print_summary(report: Mapping) -> None:
    print(f"\n{'comparison':<10} {'n':>4} {'decided':>8} {'ties':>5} {'flips':>6} "
          f"{'rate':>7}  95% CI")
    for name, s in report["comparisons"].items():
        rate = "—" if s["flip_rate"] is None else f"{s['flip_rate']:.3f}"
        ci = "—" if s["ci95"] is None else f"[{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]"
        extra = ""
        if "exceeds_repeat_floor" in s:
            extra = f"  above repeat floor: {s['exceeds_repeat_floor']}"
        print(f"{name:<10} {s['n']:>4} {s['decided']:>8} {s['ties']:>5} {s['flips']:>6} "
              f"{rate:>7}  {ci}{extra}")


# ---------------------------------------------------------------------------
# Building the pinned corpus (read-only over the seam logs)
# ---------------------------------------------------------------------------

def _jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict):
                yield rec


def _vault_text(rel: str, chars: int = 1200) -> str:
    from app.paths import VAULT_ROOT
    try:
        text = (VAULT_ROOT / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if text.startswith("---"):
        text = text.split("---", 2)[-1]
    return text.strip()[:chars]


def build_corpus(per_shape: int = 10, *, shadow_path: Path | None = None,
                 readers: Mapping[str, Callable] | None = None) -> list[dict]:
    """Seam-shaped prompts rebuilt from what the seams recorded.

    `shadow.jsonl` stores each decision's `meta`, not its state, so the state is
    rebuilt the way the seam built it (`schemas.pair_state`, `djev.rank_state`)
    from the same sources the replay reads. `readers` overrides those sources
    for tests; nothing here writes anywhere but the returned list.
    """
    if readers is None:
        from eval.djev import replay
        readers = {"backlog_head": replay._backlog_head,
                   "definition": replay._definition,
                   "vault_text": _vault_text,
                   "edges": replay.corpus_edges}
        shadow_path = shadow_path or replay.SHADOW_LOG
    assert shadow_path is not None
    out: dict[str, list[dict]] = {s: [] for s in SHAPES}
    seen: set[str] = set()

    def add(shape: str, state: str, source: dict) -> None:
        digest = hashlib.sha1(f"{shape}\0{state}".encode()).hexdigest()[:12]
        if len(out[shape]) >= per_shape or digest in seen or not state.strip():
            return
        seen.add(digest)
        out[shape].append({"id": f"{shape}-{digest}", "shape": shape,
                           "state": state, "source": source})

    for rec in _jsonl(shadow_path):
        seam, meta = rec.get("seam"), rec.get("meta") or {}
        if seam == "dedupe" and meta.get("name"):
            for cand in meta.get("candidates") or []:
                head = readers["backlog_head"](cand.get("id"))
                if head:
                    add("dedupe", schemas.pair_state(
                        meta["name"], "", f"#{head['id']} {head['title']}", head["text"]),
                        {"seam": "dedupe", "ts": rec.get("ts"), "candidate": head["id"]})
        elif seam == "entity" and meta.get("a") and meta.get("b"):
            da, db = readers["definition"](meta["a"]), readers["definition"](meta["b"])
            if da and db:  # the gate's own rule: never judge from name shape
                add("entity", schemas.pair_state(meta["a"], da, meta["b"], db),
                    {"seam": "entity", "ts": rec.get("ts")})
        elif seam == "rerank" and meta.get("query"):
            for rel in (rec.get("actual") or [])[:3]:
                text = readers["vault_text"](rel)
                if text:
                    add("rank", djev.rank_state(meta["query"], [text]),
                        {"seam": "rerank", "ts": rec.get("ts"), "path": rel})
    for r in readers["edges"](per_shape * 4):
        add("edges", f"Source: {r['a_title']}\nTarget: {r['b_title']}\n"
                     f"Quote: {r['quote'][:400]}", {"corpus": "classified-v4-batch"})
    return [row for shape in SHAPES for row in out[shape]]


# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default=DEFAULT_URL, help="djev structured server")
    ap.add_argument("--corpus", type=Path, default=None,
                    help="JSONL of {id, shape, state}; default the pinned corpus, "
                         "else the synthetic fixture")
    ap.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    ap.add_argument("--date", default=None, help="report date stamp (default today)")
    ap.add_argument("--build-corpus", type=Path, default=None, metavar="OUT",
                    help="write the pinned corpus from the seam logs and exit")
    ap.add_argument("--per-shape", type=int, default=10)
    args = ap.parse_args(argv)

    if args.build_corpus:
        rows = build_corpus(args.per_shape)
        counts = {s: sum(r["shape"] == s for r in rows) for s in SHAPES}
        args.build_corpus.write_text("".join(json.dumps(r) + "\n" for r in rows),
                                     encoding="utf-8")
        print(f"wrote {len(rows)} prompts to {args.build_corpus}: {counts}")
        return 0 if len(rows) >= MIN_PROMPTS and all(counts.values()) else 2

    if args.corpus is not None:
        corpus_path, kind = args.corpus, ("pinned" if args.corpus.resolve()
                                          == PINNED_CORPUS.resolve() else "custom")
    else:
        corpus_path, kind = default_corpus()
    try:
        corpus = load_corpus(corpus_path)
    except (CorpusError, OSError) as exc:
        print(f"bad corpus: {exc}", file=sys.stderr)
        return 2

    url = args.url.rstrip("/")
    try:
        status, _ = _http("GET", url + "/health", timeout=min(args.timeout, 5.0))
        if status != 200:
            raise Unreachable(f"/health answered HTTP {status}")
        print(f"djev {url}: {len(corpus)} prompts ({kind}), samples={args.samples}")
        rows = run(corpus, url=url, samples=args.samples, timeout=args.timeout)
    except Unreachable as exc:
        print(f"engine unreachable: {url} ({exc}) — no report written, no flip "
              f"rate measured", file=sys.stderr)
        return 3
    except BadAnswer as exc:
        print(f"engine answered out of shape: {exc} — no report written", file=sys.stderr)
        return 2

    when = dt.datetime.now().astimezone()
    report = build_report(rows, corpus_path=corpus_path, corpus_kind=kind, url=url,
                          samples=args.samples, when=when)
    if args.date:
        report["date"] = args.date
    errs = validate_report(report)
    if errs:
        print("report failed its own schema: " + "; ".join(errs), file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = report_path(args.out_dir, report["date"])
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_summary(report)
    print(f"\nreport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
