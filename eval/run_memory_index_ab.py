#!/usr/bin/env python3
"""Behavioural A/B of MEMORY.md as a typed index (review 2026-09-24, P4).

Generalises `eval/run_memory_trim_ab.py` (#1425) from "one restored line" to "a
whole overlay": an arm is a complete SOUL.md / MEMORY.md / USER.md directory, plus
the `memory/<slug>.md` topic files the indexed arm pulls from.

Arms, and the reason for each:

* `canonical` — today's three files, frozen once into `<out>/arms/canonical` so
  the nightly writers cannot move the baseline between trials.
* `canonical_rep` — the same overlay again. The A/A discordance is the noise the
  indexed arm's losses are read against (decision (a)).
* `indexed` — the frozen canonical put through
  `scripts/memory/consolidate_memory_index.py`: a ~20 KB typed index (80% of
  `prompt_surface.MEMORY_MD_INDEX_CEILING_BYTES`) plus topic files. Built from the
  frozen copy, never from the live vault, and validated (`validate_memory_index`
  full mode) before any trial runs.

`check_arms` builds both system prompts through the real
`prompt_builder.build_system_prompt` and refuses to run unless swapping the
MEMORY.md section is the ONLY difference — SOUL.md and USER.md reach both arms
byte-identical.

**`memory_read` answers from the arm, not from the vault.** A trial is a real
harness turn through `bench_runner_sdk.run_bench_sdk` (read-only tool sandbox,
recorded session), and the aggregator's `memory_read` reads the live
`~/obsidian/lloyd` — where no topic file exists until the index is deployed. So
each arm's hook registry carries the production safety gate plus a PreToolUse
deliverer (`memory_read_deliverer`) that answers every `memory_read` from that
arm's overlay, through the same file-name grammar the tool uses
(`app.memory_ceiling.topic_slug`). The call still appears in the trace as a
`memory_read` tool call, which is what decision (c) counts.

Probes: `eval/memory_index_probes.yaml`, 30 — 10 answer-in-index, 10
answer-in-topic (`objective_checks: tool_called: memory_read`), 10 `feedback`
rulings — each anchored on verbatim MEMORY.md text that `check_probe_anchors`
locates in the index or a topic file before anything runs; a topic probe's
`answer_terms` must also be absent from the index, SOUL.md and USER.md. `--with-trim-probes`
adds the 20 #1425 trim probes as a regression set (their USER.md lines are the
same in every arm, so they measure collateral damage).

Grading is `run_memory_trim_ab.grade_answer`: blind to the arm, one PASS/FAIL per
answer against the probe's criterion, primary at temperature 0, thinking off.

Decision (plan §3.5 P4), computed by `decide` and written to results.json:
  (a) indexed `lost − gained` ≤ the A/A discordance count;
  (b) 0 of 10 feedback probes lost;
  (c) ≥ 8 of 10 topic probes PASS on the indexed arm with a `memory_read` call;
  (d) live `memory_read` calls per user turn ≤ 0.5 — measured after deploy from
      event logs, so it is reported `pending` here, never assumed.
Promote only if (a)–(c) hold here and (d) holds in the week after. Otherwise keep
the ceiling, keep typed entries + topic files (additive), re-tune the consolidator.

Token side: each arm's `memories` chars and whole-prompt chars
(`arm_prompt_chars`), the same breakdown the per-turn `PROMPT_BUDGET` line logs.

Do not run casually: 3 arms × 30 probes = 90 agent turns on the primary. Pause
the worker pool first (`POST /api/workers/enable {"enabled": false}`) and hold
the primary lock:

    flock <scratch>/primary.lock .venvs/lloyd/bin/python eval/run_memory_index_ab.py \\
        --out eval/measurements/memory-index-ab-<date>
    .venvs/lloyd/bin/python eval/run_memory_index_ab.py --check --out <dir>  # arms + probes only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from app.paths import VAULT_ROOT  # noqa: E402

MEMORY_DIR = VAULT_ROOT / "lloyd"
PROBES_PATH = HERE / "memory_index_probes.yaml"
TRIM_PROBES_PATH = HERE / "memory_trim_ab_probes.yaml"
SURFACE_FILES = ("SOUL.md", "MEMORY.md", "USER.md")

ARMS = ("canonical", "canonical_rep", "indexed")
KINDS = ("index", "topic", "feedback", "trim")
PROBES_PER_KIND = 10

MAX_AGENT_TURNS = 16      # same budget as the trim A/B, same reason
PER_TRIAL_TIMEOUT = 600

#: Decision thresholds, recorded on the result so the verdict is checkable.
TOPIC_READS_REQUIRED = 8
FEEDBACK_LOSSES_ALLOWED = 0
LIVE_READS_PER_TURN_MAX = 0.5


class ProbeRejected(ValueError):
    """A probe's anchor is not where its kind says the answer lives."""


class ArmMismatch(RuntimeError):
    """The arms differ by more than their MEMORY.md."""


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------


@dataclass
class Probe:
    id: str
    kind: str
    prompt: str
    criterion: str
    anchor: str = ""
    objective_checks: dict[str, Any] = field(default_factory=dict)
    answer_terms: list[str] = field(default_factory=list)  # topic probes only
    # resolved at check time
    topic: str = ""


def load_probes(path: Path = PROBES_PATH, *, with_trim: bool = False,
                trim_path: Path = TRIM_PROBES_PATH) -> list[Probe]:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    probes: list[Probe] = []
    for p in raw.get("probes") or []:
        missing = {"id", "kind", "prompt", "criterion", "anchor"} - set(p)
        if missing:
            raise ProbeRejected(f"probe {p.get('id', '?')}: missing {sorted(missing)}")
        if p["kind"] not in KINDS[:3]:
            raise ProbeRejected(f"{p['id']}: unknown kind {p['kind']!r}")
        terms = [str(t) for t in p.get("answer_terms") or []]
        if p["kind"] == "topic" and not terms:
            raise ProbeRejected(f"{p['id']}: a topic probe names its answer_terms")
        probes.append(Probe(id=p["id"], kind=p["kind"], prompt=p["prompt"].strip(),
                            criterion=p["criterion"].strip(), anchor=p["anchor"],
                            objective_checks=dict(p.get("objective_checks") or {}),
                            answer_terms=terms))
    counts = {k: sum(p.kind == k for p in probes) for k in KINDS[:3]}
    if any(n != PROBES_PER_KIND for n in counts.values()):
        raise ProbeRejected(f"expected {PROBES_PER_KIND} probes per kind, got {counts}")
    if with_trim:
        # The #1425 probes, as a regression set: their prompt and rule only. The
        # line each one restores is in USER.md, which every arm here shares.
        trim = yaml.safe_load(trim_path.read_text(encoding="utf-8")) or {}
        for p in trim.get("probes") or []:
            probes.append(Probe(id=f"trim_{p['id']}", kind="trim",
                                prompt=p["prompt"].strip(), criterion=p["criterion"].strip()))
    ids = [p.id for p in probes]
    if len(set(ids)) != len(ids):
        raise ProbeRejected("duplicate probe ids")
    return probes


def check_probe_anchors(probes: list[Probe], canonical: Path, indexed: Path) -> None:
    """Refuse any probe whose answer is not where its kind claims, in these arms."""
    from app.memory_ceiling import TOPICS_SUBDIR

    canon = (canonical / "MEMORY.md").read_text(encoding="utf-8")
    index = (indexed / "MEMORY.md").read_text(encoding="utf-8")
    topics = {p.stem: p.read_text(encoding="utf-8")
              for p in sorted((indexed / TOPICS_SUBDIR).glob("*.md"))}
    prompt = index + "".join((indexed / n).read_text(encoding="utf-8")
                             for n in ("SOUL.md", "USER.md") if (indexed / n).exists())
    for p in probes:
        if p.kind == "trim":
            continue
        if p.anchor not in canon:
            raise ProbeRejected(f"{p.id}: anchor is not in canonical MEMORY.md")
        in_index = p.anchor in index
        if p.kind == "topic":
            if in_index:
                raise ProbeRejected(f"{p.id}: topic anchor is in the index itself")
            homes = [slug for slug, text in topics.items() if p.anchor in text]
            if len(homes) != 1:
                raise ProbeRejected(f"{p.id}: anchor is in {len(homes)} topic files")
            p.topic = homes[0]
            # The anchor is a verbatim sentence, so it is trivially absent from a
            # clipped hook that still paraphrases the answer. The terms the
            # criterion turns on are what the prompt — the index, and the SOUL.md
            # and USER.md every arm shares — must not carry (2026-09-25).
            for term in p.answer_terms:
                if term.lower() in prompt.lower():
                    raise ProbeRejected(f"{p.id}: answer term {term!r} is in the index "
                                        "or the shared prompt files")
                if term not in topics[p.topic]:
                    raise ProbeRejected(f"{p.id}: answer term {term!r} is not in "
                                        f"topics/{p.topic}")
        elif not in_index:
            raise ProbeRejected(f"{p.id}: {p.kind} anchor is not in the index")
        elif p.kind == "feedback":
            line = next(ln for ln in index.split("\n") if p.anchor in ln)
            if not line.startswith("- [feedback] "):
                raise ProbeRejected(f"{p.id}: anchor's index line is not [feedback]")


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------


def freeze_surface(src: Path, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for name in SURFACE_FILES:
        shutil.copyfile(src / name, dst / name)
    return dst


def build_indexed_arm(canonical: Path, dst: Path) -> dict[str, Any]:
    """The consolidator's overlay, built from the frozen canonical copy."""
    from scripts.memory.consolidate_memory_index import write_overlay

    report = write_overlay(canonical, dst)  # copies SOUL.md and USER.md verbatim
    if not report["validation"]["ok"]:
        raise ArmMismatch(f"indexed overlay fails validation: {report['validation']['errors']}")
    return report


def _memory_section(memory_md: str) -> str:
    """The `## MEMORY.md` block `_load_memories` renders from this file's text."""
    return f"## MEMORY.md\n{memory_md.strip()}"


def check_arms(canonical: Path, indexed: Path, build=None) -> dict[str, dict[str, int]]:
    """Raise unless the two prompts differ only in their MEMORY.md section."""
    if build is None:
        from prompt_builder import build_system_prompt as build
    a = build(overlay_dir=canonical)
    b = build(overlay_dir=indexed)
    ca = _memory_section((canonical / "MEMORY.md").read_text(encoding="utf-8"))
    cb = _memory_section((indexed / "MEMORY.md").read_text(encoding="utf-8"))
    if a.count(ca) != 1 or b.count(cb) != 1:
        raise ArmMismatch("MEMORY.md is not rendered exactly once, verbatim, in each arm")
    if a.replace(ca, cb) != b:
        raise ArmMismatch("arms differ by more than their MEMORY.md section")
    return {"canonical": {"prompt_chars": len(a), "memory_md_chars": len(ca)},
            "indexed": {"prompt_chars": len(b), "memory_md_chars": len(cb)}}


# --------------------------------------------------------------------------
# memory_read, answered from the arm
# --------------------------------------------------------------------------


def read_from_overlay(overlay: Path, file: str) -> dict[str, Any]:
    """What `agent_mcp.session._memory_read` returns, with `overlay` as the root."""
    from app.memory_ceiling import TOPICS_SUBDIR, topic_path, topic_slug

    file = (file or "MEMORY.md").strip()
    if file in ("MEMORY.md", "USER.md"):
        path = overlay / file
    else:
        slug = topic_slug(file)
        if slug is None:
            return {"error": "Invalid file. Must be MEMORY.md, USER.md, or topics/<slug>",
                    "code": "INVALID_PARAM"}
        path = topic_path(overlay, slug)
    if not path.exists():
        tdir = overlay / TOPICS_SUBDIR
        names = sorted(f"topics/{p.stem}" for p in tdir.glob("*.md")) if tdir.is_dir() else []
        return {"content": "", "file": file, "exists": False, "topics": names}
    return {"content": path.read_text(encoding="utf-8"), "file": file}


def memory_read_deliverer(overlay: Path):
    """A PreToolUse callback answering `memory_read` from `overlay`."""
    async def _cb(input_dict: dict[str, Any], _tool_use_id: str | None, _ctx: Any) -> dict:
        name = str(input_dict.get("tool_name") or "")
        if not (name == "memory_read" or name.endswith("__memory_read")):
            return {}
        args = input_dict.get("tool_input") or {}
        result = read_from_overlay(overlay, str(args.get("file") or "MEMORY.md"))
        return {"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "skillDeliver": {"skill": "", "label": "memory_read(arm overlay)",
                             "content": json.dumps(result, ensure_ascii=False)},
        }}
    return _cb


def hooks_factory_for(overlay: Path):
    def _factory():
        from app.harness import HookRegistry, install_default_safety_hook

        hooks = HookRegistry()
        install_default_safety_hook(hooks)
        hooks.add_pre_tool_use(None, memory_read_deliverer(overlay))
        return hooks
    return _factory


# --------------------------------------------------------------------------
# Objective checks, pairs, decision
# --------------------------------------------------------------------------


def memory_reads(trace: dict[str, Any]) -> list[str]:
    """The `file` of every memory_read call in a trace, in order."""
    out = []
    for c in trace.get("tool_calls") or []:
        name = str(c.get("name") or "")
        if name == "memory_read" or name.endswith("__memory_read"):
            out.append(str((c.get("args") or {}).get("file") or "MEMORY.md"))
    return out


def objective(probe: Probe, trace: dict[str, Any]) -> dict[str, Any]:
    reads = memory_reads(trace)
    out: dict[str, Any] = {"memory_reads": reads,
                           "topic_read": any(r.startswith("topics/") for r in reads)}
    want = (probe.objective_checks or {}).get("tool_called")
    if want:
        out["tool_called"] = any(str(c.get("name") or "").endswith(want)
                                 for c in trace.get("tool_calls") or [])
    return out


def _verdict(rec: dict[str, Any], arm: str) -> str | None:
    return ((rec.get("grades") or {}).get(arm) or {}).get("verdict")


def pair_verdict(canonical: str, other: str) -> str:
    """`lost` = canonical PASS and the other arm FAIL — the reverse of the trim
    A/B's orientation, where the canonical arm was the one missing a line."""
    if canonical == other:
        return "same"
    return "lost" if canonical == "PASS" else "gained"


def decide(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Decision (a)–(c) over complete records; (d) is a live measurement, pending."""
    complete = [r for r in records if all(_verdict(r, a) for a in ARMS)]
    incomplete = [r["probe_id"] for r in records if r not in complete]

    def pairs(other: str, rs: list[dict[str, Any]]) -> dict[str, int]:
        vs = [pair_verdict(_verdict(r, "canonical"), _verdict(r, other)) for r in rs]
        return {"n": len(vs), "lost": vs.count("lost"), "gained": vs.count("gained"),
                "discordant": sum(v != "same" for v in vs)}

    effect = pairs("indexed", complete)
    noise = pairs("canonical_rep", complete)
    fb = [r for r in complete if r["kind"] == "feedback"]
    fb_lost = pairs("indexed", fb)["lost"]
    topic = [r for r in complete if r["kind"] == "topic"]
    topic_ok = sum(1 for r in topic if _verdict(r, "indexed") == "PASS"
                   and (r.get("objective") or {}).get("indexed", {}).get("tool_called"))
    by_kind = {k: {"effect": pairs("indexed", [r for r in complete if r["kind"] == k]),
                   "noise": pairs("canonical_rep", [r for r in complete if r["kind"] == k]),
                   "pass_rate": {a: (round(sum(_verdict(r, a) == "PASS" for r in rs) / len(rs), 3)
                                     if (rs := [r for r in complete if r["kind"] == k]) else None)
                                 for a in ARMS}}
               for k in KINDS}
    a_ok = effect["lost"] - effect["gained"] <= noise["discordant"]
    b_ok = fb_lost <= FEEDBACK_LOSSES_ALLOWED
    c_ok = topic_ok >= TOPIC_READS_REQUIRED
    return {
        "complete": len(complete), "incomplete": incomplete,
        "effect": effect, "noise_floor": noise, "by_kind": by_kind,
        "criteria": {
            "a_net_loss_within_noise": {"ok": a_ok,
                                        "net_lost": effect["lost"] - effect["gained"],
                                        "aa_discordant": noise["discordant"]},
            "b_no_feedback_lost": {"ok": b_ok, "lost": fb_lost,
                                   "allowed": FEEDBACK_LOSSES_ALLOWED},
            "c_topic_answered_by_read": {"ok": c_ok, "count": topic_ok,
                                         "of": len(topic), "required": TOPIC_READS_REQUIRED},
            "d_live_reads_per_user_turn": {"ok": None, "max": LIVE_READS_PER_TURN_MAX,
                                           "status": "pending: measure from event logs "
                                                     "for 7 d after deploy"},
        },
        # Promote is conditional on (d) by construction: the eval can only say
        # "not refuted yet" or "refuted".
        "promote_if_live_d_holds": bool(a_ok and b_ok and c_ok and not incomplete),
    }


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def _needs_retry(t: dict[str, Any] | None) -> bool:
    return (t is None or t.get("status") != "success"
            or not (t.get("final_text") or "").strip()
            or t.get("stop_reason") == "max_turns")


async def run(probes: list[Probe], arms: dict[str, Path], *, model: str, out_dir: Path,
              run_bench=None, grade=None, probe_parallel: int = 1) -> list[dict[str, Any]]:
    if run_bench is None:
        from scripts.autoresearch.bench_runner_sdk import run_bench_sdk as run_bench
    if grade is None:
        from eval.run_memory_trim_ab import grade_answer as grade
    records: list[dict[str, Any]] = []
    raw_path = out_dir / "trials.jsonl"
    gate = asyncio.Semaphore(max(1, probe_parallel))

    async def _arm(arm: str, task: dict[str, Any]) -> dict[str, Any] | None:
        odir = arms[arm]
        for _ in range(2):  # one retry: a non-answer is not a verdict
            got = await run_bench(None, [(arm, odir)], [task], model, max_parallel=1,
                                  per_task_timeout=PER_TRIAL_TIMEOUT,
                                  max_agent_turns=MAX_AGENT_TURNS,
                                  hooks_factory=hooks_factory_for(odir))
            t = got[0] if got else None
            if not _needs_retry(t):
                return t
        return t

    async def _probe(probe: Probe) -> None:
        async with gate:
            task = {"id": probe.id, "prompt": probe.prompt, "category": "memory-index-ab"}
            traces = await asyncio.gather(*[_arm(a, task) for a in ARMS])
            by_arm = dict(zip(ARMS, traces))
            rec: dict[str, Any] = {"probe_id": probe.id, "kind": probe.kind,
                                   "topic": probe.topic, "outputs": {}, "grades": {},
                                   "objective": {}, "trials": {}}
            for arm in ARMS:
                t = by_arm.get(arm) or {}
                ok = t.get("status") == "success" and (t.get("final_text") or "").strip()
                rec["outputs"][arm] = t.get("final_text") if ok else None
                rec["grades"][arm] = (await asyncio.to_thread(grade, probe, t)) if ok else None
                rec["objective"][arm] = objective(probe, t)
                rec["trials"][arm] = {k: t.get(k) for k in ("status", "session_id", "turns",
                                                              "duration_seconds",
                                                              "stop_reason", "error")}
                rec["trials"][arm]["tools"] = [c.get("name") for c in t.get("tool_calls") or []]
                rec["trials"][arm]["usage"] = t.get("usage")
            records.append(rec)
            with raw_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({**rec, "traces": by_arm}, default=str) + "\n")
            print(f"{probe.id}: " + " ".join(f"{a}={_verdict(rec, a)}" for a in ARMS)
                  + f" reads={rec['objective']['indexed']['memory_reads']}", flush=True)

    await asyncio.gather(*[_probe(p) for p in probes])
    return records


def prepare(out: Path, *, resume: bool = False) -> tuple[dict[str, Path], dict[str, Any]]:
    """Freeze canonical, build indexed, verify both. Never writes the vault."""
    arms_dir = out / "arms"
    canonical = arms_dir / "canonical"
    if not (resume and all((canonical / n).exists() for n in SURFACE_FILES)):
        freeze_surface(MEMORY_DIR, canonical)
    indexed = arms_dir / "indexed"
    report = build_indexed_arm(canonical, indexed)
    sizes = check_arms(canonical, indexed)
    return {"canonical": canonical, "canonical_rep": canonical, "indexed": indexed}, {
        "consolidation": {k: v for k, v in report.items() if k != "topics"},
        "topic_files": len(report["topics"]), "arm_prompt_chars": sizes}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=HERE / "measurements" / "memory-index-ab")
    ap.add_argument("--model", default="primary")
    ap.add_argument("--only", nargs="*", help="probe ids to run")
    ap.add_argument("--with-trim-probes", action="store_true",
                    help="add the 20 #1425 trim probes as a regression set")
    ap.add_argument("--check", action="store_true",
                    help="build and verify arms and probe anchors, run nothing")
    ap.add_argument("--probe-parallel", type=int, default=1)
    ap.add_argument("--resume", action="store_true",
                    help="keep complete probes already in <out>/trials.jsonl")
    args = ap.parse_args()

    memory_md = MEMORY_DIR / "MEMORY.md"
    size_before = memory_md.stat().st_size
    args.out.mkdir(parents=True, exist_ok=True)
    arms, meta = prepare(args.out, resume=args.resume)
    probes = load_probes(with_trim=args.with_trim_probes)
    check_probe_anchors(probes, arms["canonical"], arms["indexed"])
    if args.only:
        probes = [p for p in probes if p.id in set(args.only)]
    print(f"{len(probes)} probes; arms verified: {json.dumps(meta['arm_prompt_chars'])}")
    if args.check:
        return 0

    raw_path = args.out / "trials.jsonl"
    done: list[dict[str, Any]] = []
    if args.resume and raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            rec.pop("traces", None)
            if all(_verdict(rec, a) for a in ARMS) and rec["probe_id"] not in {
                    d["probe_id"] for d in done}:
                done.append(rec)
    elif raw_path.exists():
        raw_path.unlink()
    todo = [p for p in probes if p.id not in {d["probe_id"] for d in done}]
    started = time.time()
    records = done + asyncio.run(run(todo, arms, model=args.model, out_dir=args.out,
                                     probe_parallel=args.probe_parallel))
    order = {p.id: i for i, p in enumerate(probes)}
    records.sort(key=lambda r: order.get(r["probe_id"], 1 << 30))
    result = {
        "item": "review-2026-09-24-P4",
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - started, 1), "model": args.model,
        "max_agent_turns": MAX_AGENT_TURNS, **meta,
        "scope_guard": {"memory_md_bytes_before": size_before,
                        "memory_md_bytes_after": memory_md.stat().st_size},
        "probes": [asdict(p) for p in probes],
        "decision": decide(records),
        "records": records,
    }
    (args.out / "results.json").write_text(json.dumps(result, indent=2, default=str),
                                           encoding="utf-8")
    print(json.dumps(result["decision"]["criteria"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
