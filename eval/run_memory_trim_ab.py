#!/usr/bin/env python3
"""Behavioural A/B of the 2026-09-14 USER.md trim (#1425).

The trim took `lloyd/USER.md` from 95,302 B to its 16,384 B ceiling, and the only
fidelity evidence on record was a per-entry word-overlap proxy. This asks the
task-level question instead: for a line the trim cut, does restoring it change
what the model does?

Design, and the reason for each part:

* **Arms differ by one line and nothing else.** SOUL.md, MEMORY.md and USER.md
  are copied once into a frozen `canonical` overlay; each probe's `restored`
  overlay is that copy with ONE archive line put back, verbatim, at the end of
  the `## ` section it was cut from. `check_arm_equality` builds both system
  prompts through the real `prompt_builder.build_system_prompt` and refuses to
  run unless removing that one line from the restored prompt gives the
  canonical prompt byte for byte. MEMORY.md grew 21.9 KB -> 72 KB after the
  trim; without the freeze the two arms could differ by whatever it wrote
  between trials.
* **Probes come from the ledger's `move` rows only** (`load_probes`). A probe
  citing a `condense` row, a line not verbatim in the archive, or a line
  already in the loaded surface is refused at load — each would measure
  something other than "a cut line".
* **A noise floor is part of the design, not a follow-up.** Every probe also
  runs the canonical arm a second time (`canonical_rep`), so the
  canonical-vs-restored discordance is read against canonical-vs-canonical.
* **One fixed grader, one explicit rule per probe.** Each answer is graded
  PASS/FAIL against the probe's `criterion`, blind to its arm, by the primary
  at temperature 0 with thinking off (`grade_answer`). The pair verdict is
  derived: `lost` (restored PASS, canonical FAIL), `gained` (the reverse),
  `same`.
* **The report keeps populations apart and refuses incomplete pairs**
  (`summarize`): a `move` rate and a `condense` rate bucketed by the ledger's
  word overlap, never pooled, and `IncompletePairs` rather than a rate over
  fewer pairs than were run.

Every trial is a real harness turn through `bench_runner_sdk.run_bench_sdk`,
so it runs under the aggregator's read-only tool sandbox (a `bench` session id)
and refuses to start without it. The live USER.md is never written; its size
is recorded before and after the run.

Usage (under the primary lock — this is ~60 agent turns):
    flock <scratch>/primary.lock .venvs/lloyd/bin/python eval/run_memory_trim_ab.py \\
        --out eval/measurements/memory-trim-ab-2026-09-24
    .venvs/lloyd/bin/python eval/run_memory_trim_ab.py --check   # load + arm equality only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from app.paths import VAULT_ROOT  # noqa: E402

MEMORY_DIR = VAULT_ROOT / "lloyd"
AUDIT_PATH = MEMORY_DIR / "reviews" / "2026-09-14-user-md-audit.md"
ARCHIVE_PATH = MEMORY_DIR / "reviews" / "2026-09-14-user-md-trim-archive.md"
PROBES_PATH = HERE / "memory_trim_ab_probes.yaml"
SURFACE_FILES = ("SOUL.md", "MEMORY.md", "USER.md")

ARMS = ("canonical", "canonical_rep", "restored")
POPULATIONS = ("move", "condense")

#: Grader config, recorded on every verdict so "a single fixed grader" is
#: checkable from the results file rather than asserted.
GRADER = {"model": "primary", "temperature": 0.0, "thinking": False, "version": 1}

#: A probe turn is a short question; the cap keeps a wandering investigation
#: from dominating the run, and is the same for every arm. 8 was too tight: the
#: first run's arms hit it on 5 of 30 trials mid-investigation, and a turn cut
#: off there ends on a preamble ("I'll check...") that grades FAIL for budget,
#: not for what the model knew.
MAX_AGENT_TURNS = 16
PER_TRIAL_TIMEOUT = 600  # the engine is shared; 300 s timed out under a 20-deep queue


class ProbeRejected(ValueError):
    """A probe does not measure a cut line."""


class ArmMismatch(RuntimeError):
    """The two arms' system prompts differ by more than the restored line."""


class IncompletePairs(RuntimeError):
    """A rate was asked for over pairs that lack an output or a verdict."""


# --------------------------------------------------------------------------
# Ledger + archive
# --------------------------------------------------------------------------


@dataclass
class LedgerRow:
    section: str
    decision: str
    prefix: str            # the ledger's first-62-chars, trailing ellipsis removed
    overlap: float | None  # only condense rows carry one


_OVERLAP = re.compile(r"word overlap ([0-9.]+)")


def load_ledger(text: str) -> list[LedgerRow]:
    """The audit's per-entry ledger rows (`| section | decision | entry | … |`)."""
    rows: list[LedgerRow] = []
    in_ledger = False
    for line in text.splitlines():
        if line.startswith("## Per-entry ledger"):
            in_ledger = True
            continue
        if in_ledger and line.startswith("## "):
            break
        if not (in_ledger and line.startswith("| ")):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) < 5 or cells[1] not in ("keep", "condense", "move"):
            continue
        m = _OVERLAP.search(cells[3])
        rows.append(LedgerRow(section=cells[0], decision=cells[1],
                              prefix=cells[2].rstrip("…").rstrip(),
                              overlap=float(m.group(1)) if m else None))
    return rows


def load_archive_bullets(text: str) -> list[tuple[str, str]]:
    """`(section, line)` for every top-level bullet of the archived USER.md.

    The archive opens with its own front matter and preamble; the copy of
    USER.md starts at its H1, so bullets are read from there on.
    """
    out: list[tuple[str, str]] = []
    started = False
    section = ""
    for line in text.splitlines():
        if line.startswith("# User (Alan)"):
            started = True
            continue
        if not started:
            continue
        if line.startswith("## "):
            section = line[3:].strip()
        elif line.startswith("- ") and section:
            out.append((section, line))
    return out


@dataclass
class Probe:
    id: str
    section: str
    entry: str
    prompt: str
    criterion: str
    kind: str = "rule"
    population: str = "move"
    gist_in_live: bool = False
    # resolved at load
    line: str = ""
    overlap: float | None = None


def resolve_probe(raw: dict[str, Any], ledger: list[LedgerRow],
                  bullets: list[tuple[str, str]], loaded: dict[str, str]) -> Probe:
    """Bind a probe to exactly one ledger row and one archive line, or refuse."""
    fields = {k: raw[k] for k in ("id", "section", "entry", "prompt", "criterion") if k in raw}
    missing = {"id", "section", "entry", "prompt", "criterion"} - set(fields)
    if missing:
        raise ProbeRejected(f"probe {raw.get('id', '?')}: missing {sorted(missing)}")
    probe = Probe(**fields, kind=raw.get("kind", "rule"),
                  population=raw.get("population", "move"),
                  gist_in_live=bool(raw.get("gist_in_live", False)))
    if probe.population not in POPULATIONS:
        raise ProbeRejected(f"{probe.id}: unknown population {probe.population!r}")
    entry = probe.entry.strip()
    rows = [r for r in ledger if r.section == probe.section
            and (r.prefix.startswith(entry) or entry.startswith(r.prefix))]
    if len(rows) != 1:
        raise ProbeRejected(f"{probe.id}: {len(rows)} ledger rows match {entry!r} "
                            f"in ## {probe.section}")
    row = rows[0]
    if row.decision != probe.population:
        raise ProbeRejected(f"{probe.id}: ledger row is `{row.decision}`, probe claims "
                            f"`{probe.population}` — a {row.decision} row is not a cut line")
    if probe.population == "condense":
        # Several originals were merged into one condensed line, so "restore the
        # original" is a swap whose target this ledger does not identify.
        raise ProbeRejected(f"{probe.id}: condense arms are not implemented (swap target "
                            "is not identifiable from the ledger)")
    lines = [ln for sec, ln in bullets if sec == probe.section and ln[2:].startswith(row.prefix)]
    if len(lines) != 1:
        raise ProbeRejected(f"{probe.id}: {len(lines)} archive lines start with the ledger "
                            f"prefix {row.prefix!r}")
    line = lines[0]
    for name, text in loaded.items():
        if line in text or line[2:] in text:
            raise ProbeRejected(f"{probe.id}: line is already loaded (verbatim in {name})")
    probe.line = line
    probe.overlap = row.overlap
    return probe


def load_probes(probes_path: Path = PROBES_PATH, *, audit_path: Path = AUDIT_PATH,
                archive_path: Path = ARCHIVE_PATH,
                surface_dir: Path = MEMORY_DIR) -> list[Probe]:
    import yaml

    raw = yaml.safe_load(probes_path.read_text(encoding="utf-8")) or {}
    ledger = load_ledger(audit_path.read_text(encoding="utf-8"))
    bullets = load_archive_bullets(archive_path.read_text(encoding="utf-8"))
    loaded = {n: (surface_dir / n).read_text(encoding="utf-8") for n in SURFACE_FILES
              if (surface_dir / n).exists()}
    probes = [resolve_probe(p, ledger, bullets, loaded) for p in raw.get("probes") or []]
    ids = [p.id for p in probes]
    if len(set(ids)) != len(ids):
        raise ProbeRejected("duplicate probe ids")
    return probes


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------


def restore_line(user_md: str, section: str, line: str) -> str:
    """USER.md with `line` added as the last line of `## section`.

    Appends a new section at the end when the live file no longer has it.
    """
    lines = user_md.split("\n")
    start = next((i for i, ln in enumerate(lines) if ln.strip() == f"## {section}"), None)
    if start is None:
        return user_md.rstrip("\n") + f"\n\n## {section}\n{line}\n"
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
               len(lines))
    last = end - 1
    while last > start and not lines[last].strip():
        last -= 1
    return "\n".join(lines[:last + 1] + [line] + lines[last + 1:])


def freeze_surface(src: Path, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for name in SURFACE_FILES:
        shutil.copyfile(src / name, dst / name)
    return dst


def build_restored_arm(canonical: Path, probe: Probe, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for name in SURFACE_FILES:
        shutil.copyfile(canonical / name, dst / name)
    user = (canonical / "USER.md").read_text(encoding="utf-8")
    (dst / "USER.md").write_text(restore_line(user, probe.section, probe.line), encoding="utf-8")
    return dst


def check_arm_equality(prompt_canonical: str, prompt_restored: str, line: str) -> None:
    """Raise unless the restored prompt is the canonical one plus exactly `line`."""
    if prompt_restored.count(line) != prompt_canonical.count(line) + 1:
        raise ArmMismatch("restored line does not appear exactly once more in the restored arm")
    idx = prompt_restored.find(line + "\n")
    candidate = (prompt_restored[:idx] + prompt_restored[idx + len(line) + 1:]
                 if idx >= 0 else prompt_restored.replace(line, "", 1))
    if candidate != prompt_canonical:
        raise ArmMismatch("arms differ by more than the restored line")


def verify_arms(canonical: Path, restored: dict[str, Path], probes: list[Probe],
                build=None) -> dict[str, int]:
    """Build every arm's system prompt the way a trial does and check equality."""
    if build is None:
        from prompt_builder import build_system_prompt as build
    base = build(overlay_dir=canonical)
    sizes = {"canonical": len(base)}
    for p in probes:
        other = build(overlay_dir=restored[p.id])
        check_arm_equality(base, other, p.line)
        sizes[p.id] = len(other) - len(base)
    return sizes


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------

GRADER_PROMPT = """You are grading one assistant answer against one explicit rule.

USER REQUEST:
{prompt}

RULE (the whole test — apply it literally):
{criterion}

TOOLS THE ASSISTANT CALLED (name — its own caption):
{tools}

ASSISTANT'S FINAL ANSWER:
{answer}

Reply with a JSON object only: {{"verdict": "PASS" or "FAIL", "reason": "<one sentence>"}}"""


def _grader_prompt(probe: Probe, trace: dict[str, Any]) -> str:
    calls = trace.get("tool_calls") or []
    tools = "\n".join(f"- {c.get('name')} — {c.get('summary') or ''}" for c in calls[:20]) or "(none)"
    return GRADER_PROMPT.format(prompt=probe.prompt, criterion=probe.criterion, tools=tools,
                                answer=(trace.get("final_text") or "(empty)")[-6000:])


def parse_grade(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    verdict = str(obj.get("verdict", "")).upper()
    if verdict not in ("PASS", "FAIL"):
        return None
    return {"verdict": verdict, "reason": str(obj.get("reason", ""))[:400]}


def _call_grader(prompt: str, timeout: int = 300) -> str | None:
    import requests

    from app.config import _get_model_cfg, resolve_model_alias
    from scripts.autoresearch.common import AUTORESEARCH_PRIORITY

    name = resolve_model_alias(GRADER["model"])
    cfg = _get_model_cfg(name) or {}
    base = (cfg.get("base_url") or cfg.get("env", {}).get("ANTHROPIC_BASE_URL", "")).rstrip("/")
    resp = requests.post(
        f"{base}/v1/chat/completions",
        headers={"Authorization": "Bearer no-key-required"},
        json={"model": name, "messages": [{"role": "user", "content": prompt}],
              "temperature": GRADER["temperature"], "max_tokens": 400,
              "response_format": {"type": "json_object"},
              "chat_template_kwargs": {"enable_thinking": GRADER["thinking"]},
              "priority": AUTORESEARCH_PRIORITY},
        timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")


def grade_answer(probe: Probe, trace: dict[str, Any], call=_call_grader) -> dict[str, Any] | None:
    """One PASS/FAIL against the probe's rule, blind to the arm. None = ungraded."""
    if trace.get("status") != "success" or not (trace.get("final_text") or "").strip():
        return None
    for _ in range(3):
        try:
            graded = parse_grade(call(_grader_prompt(probe, trace)))
        except Exception:  # noqa: BLE001 — an unreachable grader leaves the pair ungraded
            graded = None
        if graded:
            return {**graded, "grader": dict(GRADER)}
    return None


# --------------------------------------------------------------------------
# Pairs and the report
# --------------------------------------------------------------------------


def pair_verdict(a: str, b: str) -> str:
    """`a` = canonical-side verdict, `b` = the other arm's."""
    if a == b:
        return "same"
    return "lost" if b == "PASS" else "gained"


def _wilson(k: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    from eval.stats import wilson_ci
    lo, hi = wilson_ci(k, n)
    return [round(lo, 3), round(hi, 3)]


def _sign_test_p(k: int, n: int) -> float | None:
    """Two-sided exact binomial p for k of n discordant pairs going one way."""
    if n == 0:
        return None
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def _rates(records: list[dict[str, Any]], other: str) -> dict[str, Any]:
    verdicts = [pair_verdict(r["grades"]["canonical"]["verdict"],
                             r["grades"][other]["verdict"]) for r in records]
    n = len(verdicts)
    div = sum(v != "same" for v in verdicts)
    lost = verdicts.count("lost")
    gained = verdicts.count("gained")
    return {"n": n, "divergent": div, "rate": round(div / n, 3) if n else None,
            "ci95": _wilson(div, n), "lost": lost, "gained": gained,
            "sign_test_p": _sign_test_p(lost, lost + gained)}


def _complete(r: dict[str, Any]) -> bool:
    return all((r.get("outputs") or {}).get(a) is not None
               and ((r.get("grades") or {}).get(a) or {}).get("verdict") in ("PASS", "FAIL")
               for a in ARMS)


def _cost(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    trials = [r.get("trials") for r in records]
    if not records or not all(trials):
        return None
    out = {}
    for a in ARMS:
        ts = [t.get(a) or {} for t in trials]
        out[a] = {"mean_tool_calls": round(sum(len(t.get("tools") or []) for t in ts) / len(ts), 2),
                  "mean_seconds": round(sum(float(t.get("duration_seconds") or 0) for t in ts)
                                        / len(ts), 1),
                  # A turn that hit the iteration cap usually ends on a preamble
                  # ("I'll check…"), so its FAIL is a budget outcome, not a verdict
                  # about what it knew — counted here so the report can say so.
                  "hit_max_turns": sum(t.get("stop_reason") == "max_turns" for t in ts)}
    return out


def _decile(overlap: float) -> str:
    lo = min(9, int(overlap * 10)) / 10
    return f"{lo:.1f}-{lo + 0.1:.1f}"


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-population rates; refuses to compute over incomplete pairs.

    Deliberately has no pooled rate: a move pair contrasts cut vs absent, a
    condense pair contrasts original vs condensed wording, and one number over
    both answers neither question.
    """
    bad = [r.get("probe_id", "?") for r in records if not _complete(r)]
    if bad:
        raise IncompletePairs(f"{len(bad)} pair(s) lack an output or a verdict: {bad}")
    out: dict[str, Any] = {}
    move = [r for r in records if r["population"] == "move"]
    out["move"] = {
        "effect": _rates(move, "restored"),
        "noise_floor": _rates(move, "canonical_rep"),
        "pass_rate": {a: round(sum(r["grades"][a]["verdict"] == "PASS" for r in move)
                               / len(move), 3) if move else None for a in ARMS},
        # Secondary: what the arm spent to get there. A cut line whose content
        # is still retrievable can cost searches rather than correctness.
        "cost": _cost(move),
    }
    cond = [r for r in records if r["population"] == "condense"]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in cond:
        buckets.setdefault(_decile(r["overlap"]), []).append(r)
    by_bucket = {b: _rates(rs, "restored")["rate"] for b, rs in sorted(buckets.items())}
    out["condense"] = {"n": len(cond),
                       "by_overlap_decile": {b: _rates(rs, "restored")
                                             for b, rs in sorted(buckets.items())}}
    out["overlap_orders_divergence"] = _overlap_statement(by_bucket)
    return out


def _overlap_statement(by_bucket: dict[str, float | None]) -> str:
    rates = [r for r in by_bucket.values() if r is not None]
    if len(rates) < 2:
        return (f"undetermined: {len(rates)} condense overlap bucket(s) measured, "
                "at least 2 are needed to say whether overlap orders divergence")
    falling = all(a >= b for a, b in zip(rates, rates[1:]))
    return ("yes: divergence falls monotonically as word overlap rises" if falling
            else "no: divergence is not monotone in word overlap")


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def _needs_retry(t: dict[str, Any] | None) -> bool:
    """An engine blip or a turn cut off by the iteration cap is not a verdict."""
    return (t is None or t.get("status") != "success"
            or not (t.get("final_text") or "").strip()
            or t.get("stop_reason") == "max_turns")


async def run(probes: list[Probe], canonical: Path, restored: dict[str, Path], *,
              model: str, out_dir: Path, run_bench=None, grade=grade_answer,
              max_parallel: int = 3, probe_parallel: int = 1) -> list[dict[str, Any]]:
    if run_bench is None:
        from scripts.autoresearch.bench_runner_sdk import run_bench_sdk as run_bench
    records: list[dict[str, Any]] = []
    raw_path = out_dir / "trials.jsonl"
    gate = asyncio.Semaphore(max(1, probe_parallel))

    async def _probe(probe: Probe) -> None:
        async with gate:
            await _one_probe(probe)

    async def _one_probe(probe: Probe) -> None:
        task = {"id": probe.id, "prompt": probe.prompt, "category": "memory-trim-ab"}
        variants = [("canonical", canonical), ("canonical_rep", canonical),
                    ("restored", restored[probe.id])]
        traces = await run_bench(None, variants, [task], model, max_parallel=max_parallel,
                                 per_task_timeout=PER_TRIAL_TIMEOUT,
                                 max_agent_turns=MAX_AGENT_TURNS)
        by_arm = {t["variant_id"]: t for t in traces}
        # One retry per arm: an incomplete pair cannot be rated, and the retry
        # runs the same arm, so it can only replace a non-answer.
        for arm, odir in variants:
            if _needs_retry(by_arm.get(arm)):
                again = await run_bench(None, [(arm, odir)], [task], model, max_parallel=1,
                                        per_task_timeout=PER_TRIAL_TIMEOUT,
                                        max_agent_turns=MAX_AGENT_TURNS)
                if again and (again[0].get("status") == "success"
                              or (by_arm.get(arm) or {}).get("status") != "success"):
                    by_arm[arm] = again[0]
        rec = {"probe_id": probe.id, "population": probe.population, "kind": probe.kind,
               "gist_in_live": probe.gist_in_live, "overlap": probe.overlap,
               "section": probe.section, "restored_line": probe.line,
               "outputs": {}, "grades": {}, "trials": {}}
        for arm in ARMS:
            t = by_arm.get(arm) or {}
            ok = t.get("status") == "success" and (t.get("final_text") or "").strip()
            rec["outputs"][arm] = t.get("final_text") if ok else None
            rec["grades"][arm] = (await asyncio.to_thread(grade, probe, t)) if ok else None
            rec["trials"][arm] = {k: t.get(k) for k in ("status", "session_id", "turns",
                                                          "duration_seconds", "stop_reason",
                                                          "error")}
            rec["trials"][arm]["tools"] = [c.get("name") for c in t.get("tool_calls") or []]
        v = [(rec["grades"][a] or {}).get("verdict") for a in ARMS]
        rec["pair"] = (pair_verdict(v[0], v[2]) if v[0] and v[2] else None)
        records.append(rec)
        with raw_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({**rec, "traces": {a: by_arm.get(a) for a in ARMS}},
                                default=str) + "\n")
        print(f"{probe.id}: " + " ".join(f"{a}={x}" for a, x in zip(ARMS, v))
              + f" -> {rec['pair']}", flush=True)

    await asyncio.gather(*[_probe(p) for p in probes])
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=HERE / "measurements" / "memory-trim-ab")
    ap.add_argument("--model", default="primary")
    ap.add_argument("--only", nargs="*", help="probe ids to run")
    ap.add_argument("--check", action="store_true", help="load probes, build and verify arms, stop")
    ap.add_argument("--max-parallel", type=int, default=3)
    ap.add_argument("--probe-parallel", type=int, default=1,
                    help="probes in flight at once (each runs its three arms)")
    ap.add_argument("--resume", action="store_true",
                    help="keep probes already recorded in <out>/trials.jsonl")
    args = ap.parse_args()

    user_md = MEMORY_DIR / "USER.md"
    size_before = user_md.stat().st_size
    probes = load_probes()
    if args.only:
        probes = [p for p in probes if p.id in set(args.only)]
    args.out.mkdir(parents=True, exist_ok=True)
    arms_dir = args.out / "arms"
    # A resumed run keeps the snapshot it started from: re-freezing would let
    # MEMORY.md's overnight growth in between the two halves of the probe set.
    canonical = arms_dir / "canonical"
    if not (args.resume and all((canonical / n).exists() for n in SURFACE_FILES)):
        canonical = freeze_surface(MEMORY_DIR, canonical)
    restored = {p.id: build_restored_arm(canonical, p, arms_dir / p.id) for p in probes}
    deltas = verify_arms(canonical, restored, probes)
    print(f"{len(probes)} probes; arms verified (canonical prompt {deltas['canonical']} chars)")
    if args.check:
        return 0

    # --resume: probes already in trials.jsonl are kept, not re-run. A 45-minute
    # lock hold on a shared engine does not always fit 60 turns, and a pair is
    # only ever recorded whole, so resuming cannot splice half a pair.
    done: list[dict[str, Any]] = []
    raw_path = args.out / "trials.jsonl"
    if args.resume and raw_path.exists():
        wanted = {p.id for p in probes}
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            rec.pop("traces", None)
            # An incomplete pair is re-run whole, never patched arm by arm.
            if (rec["probe_id"] in wanted and _complete(rec)
                    and rec["probe_id"] not in {d["probe_id"] for d in done}):
                done.append(rec)
    elif raw_path.exists():
        raw_path.unlink()
    todo = [p for p in probes if p.id not in {d["probe_id"] for d in done}]
    print(f"{len(done)} probes resumed from {raw_path.name}, {len(todo)} to run")

    started = time.time()
    records = done + asyncio.run(run(todo, canonical, restored, model=args.model,
                                     out_dir=args.out, max_parallel=args.max_parallel,
                                     probe_parallel=args.probe_parallel))
    order = {p.id: i for i, p in enumerate(probes)}
    records.sort(key=lambda r: order[r["probe_id"]])
    size_after = user_md.stat().st_size
    result = {
        "item": 1425, "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - started, 1), "model": args.model,
        "grader": GRADER, "max_agent_turns": MAX_AGENT_TURNS,
        "arm_prompt_deltas_chars": deltas,
        "scope_guard": {"user_md_bytes_before": size_before, "user_md_bytes_after": size_after,
                        "unchanged": size_before == size_after},
        "probes": [asdict(p) for p in probes],
        "records": records,
    }
    try:
        result["summary"] = summarize(records)
    except IncompletePairs as exc:
        result["summary_error"] = str(exc)
    (args.out / "results.json").write_text(json.dumps(result, indent=2, default=str),
                                           encoding="utf-8")
    print(json.dumps(result.get("summary") or result.get("summary_error"), indent=2))
    return 0 if "summary" in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
