#!/usr/bin/env python3
"""Plant-then-supersede memory eval: does Lloyd act on a superseded value? (#622)

A pinned synthetic corpus of fictional entities is planted into a SHADOW memory
— a temp fact tree and a temp `kg.sqlite` (`LLOYD_FACTS_ROOT` / `LLOYD_KG_DB`)
plus a temp `MEMORY.md` — through the real writers (`facts._fact_add`,
`facts._fact_invalidate`, `session._memory_add`). Then, for half the entities,
the probed fact is superseded, and the primary is asked an action-shaped
question whose right answer needs the CURRENT value.

Each arm is one memory state; everything else (system prompt skeleton, the
probe, a `fact_get` tool exchange on the probe's entity, sampling) is identical,
so two arms differ only by what the memory holds:

  stateless        empty store, empty MEMORY.md — the gain control.
  facts_expired    store superseded the way a correct writer does it:
                   `fact_add(new)` then `fact_invalidate(old)`. `fact_get`
                   filters expired rows, so this is the leg that ALREADY filters.
  facts_appended   `fact_add(new)` only — what the extractor actually does: the
                   live store held 25 expired rows of 306,396 on 09-14, so a
                   changed value almost always leaves the old row active.
  prose_appended   empty store; MEMORY.md written by `memory_add`, the old line
                   and the new one both present (the `lloyd` segment, which is
                   loaded into every system prompt and indexed by qmd — no
                   `expired_at` exists there to filter on).
  prose_consolidated  as prose_appended, then rewritten the way a nightly
                   consolidation rewrites the file: one section per entity, its
                   lines in no order, so position is no longer a recency cue.
  prose_*_dated    the two prose arms with `memory_tools.date_stamp_entries`
                   on: each `memory_add` entry carries the date it was
                   written, which a rewrite carries along with the line.

Scored on the final answer text (reasoning excluded): a superseded probe is
`correct` (new value only), `stale` (old value only — a miss, never an
unanswered probe), `mixed` (both) or `none`. Stale-evidence is whether the old
value is anywhere in what the model was shown.

The engine is called directly with no live tools, so nothing here can change
the machine; the session label is still a sandboxed `pt-eval-*` id.

Usage (the primary must be otherwise idle; hold the primary lock):
    .venvs/lloyd/bin/python eval/run_stale_fact_eval.py --samples 3
    .venvs/lloyd/bin/python eval/run_stale_fact_eval.py --dry-run   # corpus + evidence only
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as _dt
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
LLOYD_HOME = HERE.parent
sys.path.insert(0, str(LLOYD_HOME))
sys.path.insert(0, str(HERE))

ARMS = ("stateless", "facts_expired", "facts_appended", "prose_appended", "prose_consolidated",
        "prose_appended_dated", "prose_consolidated_dated")
DEFAULT_OUT_DIR = HERE / "measurements"
PLANT_AT = _dt.datetime(2026, 7, 6, 9, 0, tzinfo=_dt.timezone.utc)
SUPERSEDE_AT = _dt.datetime(2026, 9, 2, 14, 0, tzinfo=_dt.timezone.utc)


class LiveStoreRefused(RuntimeError):
    """The shadow paths resolve to the live store, vault or data root."""


# ─────────────────────────────────────────────────────────────── corpus ──

# kind -> (entity_type, [(slot, fact template, probe template)])
_KINDS: dict[str, tuple[str, list[tuple[str, str, str]]]] = {
    "service": ("system", [
        ("port", "{e} listens on port {v}.",
         "Give me the exact curl command to hit {e}'s /health endpoint on localhost."),
        ("host", "{e} runs on the host {v}.",
         "Which machine do I ssh into to read {e}'s logs? Give me the ssh command."),
        ("config", "{e}'s config file is {v}.",
         "Write the one-line command to open {e}'s config file in vim."),
        ("owner", "{e} is maintained by {v}.",
         "Who should I message about a bug in {e}? Just the person's name."),
    ]),
    "person": ("person", [
        ("email", "{e}'s email address is {v}.",
         "Draft the To: line for an email to {e}."),
        ("team", "{e} works on the {v} team.",
         "Which team's channel should I post in to reach {e}? Answer with the team name."),
        ("city", "{e} lives in {v}.",
         "I want to book a call with {e} at 9am their local time. Which city's clock do I use?"),
        ("ext", "{e}'s desk phone extension is {v}.",
         "What extension do I dial to reach {e}?"),
    ]),
    "project": ("project", [
        ("lead", "{e} is led by {v}.",
         "Who do I ask for sign-off on a change to {e}? Just the name."),
        ("repo", "{e}'s code lives in the repository {v}.",
         "Give me the git clone command for {e}."),
        ("budget", "{e} has a compute budget of {v} GPU-hours.",
         "What is {e}'s total compute budget in GPU-hours? I'm planning a run against it."),
        ("costcode", "{e} bills to cost code {v}.",
         "I'm filing an expense for {e}. Which cost code goes on the form?"),
    ]),
    "device": ("system", [
        ("ip", "{e} is at IP address {v}.",
         "Give me the ping command to check whether {e} is up."),
        ("room", "{e} is kept in the {v}.",
         "I need to power-cycle {e} by hand. Which room do I walk to?"),
        ("firmware", "{e} runs firmware version {v}.",
         "I'm filing a support ticket for {e}. What firmware version do I list?"),
        ("adminport", "{e}'s admin web page is served on port {v}.",
         "What URL do I open in a browser to reach {e}'s admin page? It's on this LAN at its usual address; just use the hostname {slug}.lan."),
    ]),
}

_NAMES = {
    "service": ["Harbor Relay", "Quillmark", "Tessera Gate", "Vantor Queue",
                "Brindle Cache", "Osprey Ledger", "Calyx Proxy", "Nimbrel Index"],
    "person": ["Marisol Quenby", "Teodor Vaskin", "Ines Halloran", "Pritam Sokolow",
               "Greer Ashdown", "Olwen Tarrant", "Kasimir Delbrook", "Yusra Penhale"],
    "project": ["Project Larkspur", "Project Emberwell", "Project Saltmarsh", "Project Kitewind",
                "Project Driftglass", "Project Rookhaven", "Project Tallowmere", "Project Wrenfield"],
    "device": ["Garage Printer", "Attic NAS", "Studio Camera", "Porch Doorbell",
               "Cellar Hygrometer", "Loft Projector", "Shed Router", "Den Thermostat"],
}

_SURNAMES = ["Abernathy", "Brightwater", "Castellane", "Dunmore", "Everhart", "Fairweather",
             "Galloway", "Hollingsworth", "Ingleby", "Jessamine", "Kilbride", "Lockhart",
             "Merriweather", "Northcott", "Oakenshaw", "Pendragon", "Quarrington", "Ravensworth",
             "Stanbury", "Thistlewood", "Underhill", "Vandermeer", "Whitlock", "Yarborough"]
_FIRST = ["Aldous", "Bettina", "Corwin", "Delphine", "Emrys", "Fenella", "Gideon", "Hesper",
          "Ivo", "Juno", "Leander", "Mireille", "Niall", "Ottoline", "Percival", "Rosalind"]
_BIRDS = ["kestrel", "plover", "merlin", "shrike", "tern", "curlew", "dunlin", "godwit",
          "whimbrel", "avocet", "redshank", "sanderling"]
_TEAMS = ["Lattice", "Foundry", "Beacon", "Keystone", "Meridian", "Tributary", "Parapet",
          "Cinder", "Halyard", "Sextant", "Gantry", "Bulwark"]
_CITIES = ["Reykjavik", "Montevideo", "Tallinn", "Wellington", "Porto", "Winnipeg",
           "Ljubljana", "Hobart", "Valparaiso", "Tromso", "Cork", "Busan"]
_ROOMS = ["pantry", "boot room", "laundry room", "back hallway", "guest bedroom", "workshop",
          "utility closet", "sunroom", "mudroom", "stairwell cupboard", "wine cellar", "study"]


@dataclass
class Probe:
    entity: str
    kind: str
    entity_type: str
    slot: str
    question: str
    superseded: bool
    value: str                       # the value the probe needs (current)
    old_value: str | None            # the superseded value, or None for a control
    facts: list[tuple[str, str]]     # (category, sentence) planted for this entity
    supersede_fact: str | None = None
    old_fact: str | None = None
    match_new: list[str] = field(default_factory=list)
    match_old: list[str] = field(default_factory=list)


def _slug(entity: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", entity.lower().replace("project ", "")).strip("-")


def _value(kind: str, slot: str, entity: str, rng: random.Random, used: set[str]) -> tuple[str, list[str]]:
    """One fresh value for `slot` and the strings that count as using it."""
    for _ in range(200):
        s = _slug(entity)
        if slot in ("port", "adminport"):
            v = str(rng.randint(2100, 9899)); m = [v]
        elif slot == "host":
            v = f"{rng.choice(_BIRDS)}-{rng.randint(10, 97)}"; m = [v]
        elif slot == "config":
            v = f"/etc/{s}/{rng.choice(['main', 'runtime', 'service', 'core'])}-{rng.randint(2, 9)}.toml"; m = [v]
        elif slot in ("owner", "lead"):
            sur = rng.choice(_SURNAMES)
            v = f"{rng.choice(_FIRST)} {sur}"; m = [sur]
        elif slot == "email":
            first = entity.split()[0].lower()
            v = f"{first}.{rng.choice(['q', 'v', 'h', 'x', 'z'])}{rng.randint(10, 99)}@{rng.choice(['northgate', 'eastmere', 'fellbrook'])}.org"
            m = [v]
        elif slot == "team":
            v = rng.choice(_TEAMS); m = [v]
        elif slot == "city":
            v = rng.choice(_CITIES); m = [v]
        elif slot == "ext":
            v = str(rng.randint(3100, 8899)); m = [v]
        elif slot == "repo":
            v = f"git@forge.lan:{s}/{rng.choice(['core', 'engine', 'app', 'svc'])}-{rng.randint(2, 9)}.git"; m = [v]
        elif slot == "budget":
            v = str(rng.randint(12, 96) * 250); m = [v]
        elif slot == "costcode":
            v = f"CC-{rng.randint(1000, 9999)}"; m = [v]
        elif slot == "ip":
            v = f"10.{rng.randint(20, 90)}.{rng.randint(1, 250)}.{rng.randint(2, 250)}"; m = [v]
        elif slot == "room":
            v = rng.choice(_ROOMS); m = [v]
        elif slot == "firmware":
            v = f"v{rng.randint(2, 9)}.{rng.randint(10, 39)}.{rng.randint(1, 9)}"; m = [v, v[1:]]
        else:  # pragma: no cover - the table above is closed
            raise KeyError(slot)
        if v not in used and not any(v in u or u in v for u in used if len(u) > 3):
            used.add(v)
            return v, m
    raise RuntimeError(f"could not draw a fresh value for {entity}/{slot}")


def build_corpus(seed: int = 622) -> list[Probe]:
    """The pinned corpus: 32 entities x 4 facts, 20 superseded, 12 controls.

    Deterministic in `seed`. Entity i probes slot i % 4, and i % 8 < 5 makes
    the probed fact superseded, so every kind carries 5 superseded probes and
    3 controls and every slot is probed in both roles.
    """
    rng = random.Random(seed)
    probes: list[Probe] = []
    for kind, (etype, slots) in _KINDS.items():
        used: set[str] = set()
        for i, entity in enumerate(_NAMES[kind]):
            values = {slot: _value(kind, slot, entity, rng, used) for slot, _f, _q in slots}
            slot, ftpl, qtpl = slots[i % 4]
            superseded = i % 8 < 5
            facts = [(sl, ft.format(e=entity, v=values[sl][0])) for sl, ft, _q in slots]
            question = qtpl.format(e=entity, slug=_slug(entity))
            if superseded:
                new_v, new_m = _value(kind, slot, entity, rng, used)
                old_v, old_m = values[slot]
                probes.append(Probe(entity, kind, etype, slot, question, True, new_v, old_v,
                                    facts, supersede_fact=ftpl.format(e=entity, v=new_v),
                                    old_fact=ftpl.format(e=entity, v=old_v),
                                    match_new=new_m, match_old=old_m))
            else:
                v, m = values[slot]
                probes.append(Probe(entity, kind, etype, slot, question, False, v, None,
                                    facts, match_new=m))
    return probes


def corpus_counts(probes: list[Probe]) -> dict:
    per = [len(p.facts) for p in probes]
    return {"entities": len(probes), "facts_planted": sum(per),
            "facts_per_entity_min": min(per), "facts_per_entity_max": max(per),
            "superseded": sum(p.superseded for p in probes),
            "controls": sum(not p.superseded for p in probes)}


# ─────────────────────────────────────────────────────────────── guard ──

def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def check_shadow_paths(environ: dict | None = None) -> tuple[Path, Path]:
    """The shadow fact root and store, or `LiveStoreRefused` naming the variable.

    Refused: an unset variable, anything under the production data root or the
    vault, and the built-in default locations. This runs BEFORE `app.paths` is
    imported, so it reads the environment and the passwd home itself.
    """
    env = os.environ if environ is None else environ
    from app.data_root import ACCOUNT_HOME, PRODUCTION_DATA_ROOT
    live_roots = [PRODUCTION_DATA_ROOT, ACCOUNT_HOME / "obsidian", ACCOUNT_HOME / "lloyd-data",
                  LLOYD_HOME / ".lloyd-data", ACCOUNT_HOME / "lloyd"]
    out = []
    for var in ("LLOYD_FACTS_ROOT", "LLOYD_KG_DB"):
        raw = (env.get(var) or "").strip()
        if not raw:
            raise LiveStoreRefused(f"{var} is not set; this eval runs only against a shadow store")
        p = Path(raw).expanduser()
        for root in live_roots:
            if _under(p, root):
                raise LiveStoreRefused(f"{var}={p} resolves inside {root}, a live path; refusing")
        out.append(p)
    return out[0], out[1]


# ─────────────────────────────────────────────────────────────── plant ──

class _Clock:
    """A `datetime` module stand-in for the fact writer, so planted and
    superseding rows carry dates weeks apart as they would in life. The
    writer's code is untouched; only what `now()` answers moves."""

    def __init__(self, at: _dt.datetime):
        self.at = at
        clock = self

        class _DT(_dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return clock.at if tz else clock.at.replace(tzinfo=None)

        self.module = SimpleNamespace(datetime=_DT, timezone=_dt.timezone, date=_dt.date,
                                      timedelta=_dt.timedelta)

    def tick(self, seconds: int = 3600) -> None:
        self.at += _dt.timedelta(seconds=seconds)


def _arm_dirs(root: Path, arm: str) -> SimpleNamespace:
    d = root / arm
    ns = SimpleNamespace(root=d, facts=d / "facts", db=d / "kg.sqlite", memories=d / "memories",
                         old_last={})
    ns.facts.mkdir(parents=True, exist_ok=True)
    ns.memories.mkdir(parents=True, exist_ok=True)
    (ns.memories / "MEMORY.md").touch()
    return ns


def _point_store(dirs: SimpleNamespace) -> None:
    """Point every module-level fact-root constant and the store at one arm."""
    from agent_mcp import _shared, facts, retrieval
    from app import kg_store
    kg_store.configure(dirs.db)
    for mod in (_shared, facts, retrieval):
        if hasattr(mod, "FACTS_ROOT"):
            mod.FACTS_ROOT = dirs.facts
    _shared.ALIASES_PATH = dirs.facts / "entity-aliases.json"
    _shared._entity_dirs_cache = None


@contextlib.contextmanager
def restored_store_pointers():
    """Put back every constant `_point_store` moves, and the store default.

    The runner repoints process-wide module state per arm; inside one process
    that also runs other code (a test session) it must leave nothing behind."""
    from agent_mcp import _shared, facts, retrieval
    from app import kg_store
    saved = [(mod, name, getattr(mod, name)) for mod, name in (
        (_shared, "FACTS_ROOT"), (facts, "FACTS_ROOT"), (retrieval, "FACTS_ROOT"),
        (_shared, "ALIASES_PATH"), (_shared, "_entity_dirs_cache"),
        (kg_store, "_default_path")) if hasattr(mod, name)]
    try:
        yield
    finally:
        kg_store.reset()
        for mod, name, value in saved:
            setattr(mod, name, value)


def plant(probes: list[Probe], shadow_root: Path, arm: str, seed: int = 622) -> SimpleNamespace:
    """Write one arm's memory state through the real writers. Returns its dirs."""
    from agent_mcp import facts, session
    dirs = _arm_dirs(shadow_root, arm)
    if arm == "stateless":
        return dirs
    rng = random.Random(seed)
    clock = _Clock(PLANT_AT)
    order = [(p, cat, text) for p in probes for cat, text in p.facts]
    rng.shuffle(order)
    later = [p for p in probes if p.superseded]
    rng.shuffle(later)
    if arm.startswith("facts_"):
        _point_store(dirs)
        real_dt = facts.datetime
        facts.datetime = clock.module
        try:
            for p, cat, text in order:
                r = facts._fact_add({"entity": p.entity, "category": cat, "fact": text,
                                     "entity_type": p.entity_type, "provenance": "STATED"})
                if not r.get("success"):
                    raise RuntimeError(f"plant failed for {p.entity}: {r}")
                clock.tick(rng.randint(600, 7200))
            clock.at = SUPERSEDE_AT
            for p in later:
                r = facts._fact_add({"entity": p.entity, "category": p.slot,
                                     "fact": p.supersede_fact, "provenance": "STATED"})
                if not r.get("success"):
                    raise RuntimeError(f"supersede failed for {p.entity}: {r}")
                if arm == "facts_expired":
                    r = facts._fact_invalidate({"entity": p.entity, "category": p.slot,
                                                "fact_substring": p.old_fact,
                                                "ended": clock.at.date().isoformat(),
                                                "reason": "superseded"})
                    if r.get("expired_count") != 1:
                        raise RuntimeError(f"invalidate did not expire one row for {p.entity}: {r}")
                clock.tick(rng.randint(600, 7200))
        finally:
            facts.datetime = real_dt
    elif arm.startswith("prose_"):
        saved = (session.MEMORIES_ROOT, session._date_stamp_enabled, session._entry_date)
        clock = _Clock(PLANT_AT)
        session.MEMORIES_ROOT = dirs.memories
        dated = arm.endswith("_dated")
        session._date_stamp_enabled = lambda: dated
        session._entry_date = lambda: clock.at.date()
        try:
            for p, _cat, text in order:
                r = session._memory_add({"file": "MEMORY.md", "entry": f"- {text}"})
                if not r.get("success"):
                    raise RuntimeError(f"memory_add failed: {r}")
                clock.tick(rng.randint(600, 7200))
            clock.at = SUPERSEDE_AT
            for p in later:
                r = session._memory_add({"file": "MEMORY.md", "entry": f"- {p.supersede_fact}"})
                if not r.get("success"):
                    raise RuntimeError(f"memory_add failed: {r}")
                clock.tick(rng.randint(600, 7200))
        finally:
            session.MEMORIES_ROOT, session._date_stamp_enabled, session._entry_date = saved
        if arm.startswith("prose_consolidated"):
            dirs.old_last = consolidate(dirs.memories / "MEMORY.md", probes, rng)
    return dirs


def consolidate(path: Path, probes: list[Probe], rng: random.Random) -> dict[str, bool]:
    """Rewrite MEMORY.md the way a consolidation pass does: one `##` section per
    entity, its lines in no particular order. The nightly jobs rewrite the
    memory files whole (Write/Edit), so file position stops being a recency
    cue; this is the arm that asks what is left once it has. Returns, per
    superseded entity, whether the old line now sits AFTER the new one."""
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.startswith("- ")]
    by_entity: dict[str, list[str]] = {p.entity: [] for p in probes}
    def body(ln: str) -> str:
        return re.sub(r"^- (\(\d{4}-\d{2}-\d{2}\) )?", "", ln)

    for ln in lines:
        owner = next(p.entity for p in probes if body(ln).startswith(p.entity))
        by_entity[owner].append(ln)
    old_last: dict[str, bool] = {}
    out = ["# Long-Term Memory", ""]
    for p in sorted(probes, key=lambda q: q.entity):
        group = by_entity[p.entity]
        rng.shuffle(group)
        if p.superseded:
            idx = {body(ln): i for i, ln in enumerate(group)}
            old_last[p.entity] = idx[p.old_fact] > idx[p.supersede_fact]
        out += [f"## {p.entity}", *group, ""]
    path.write_text("\n".join(out), encoding="utf-8")
    return old_last


# ─────────────────────────────────────────────────────────────── render ──

TOOL_NAMES = ("fact_get", "memory_read")


def _tools() -> list[dict]:
    """`fact_get` and `memory_read` exactly as their servers advertise them.

    The trial may call them as often as it likes; each call is answered by the
    real handler against THIS arm's shadow state (`dispatch`). Nothing else is
    offered, so nothing a trial does can reach the machine.
    """
    from agent_mcp import facts, session
    out = []
    for mod in (facts, session):
        for t in asyncio.run(mod.list_tools()):
            if t.name in TOOL_NAMES:
                params = getattr(t, "input_schema", None) or t.inputSchema
                out.append({"type": "function", "function": {
                    "name": t.name, "description": t.description, "parameters": params}})
    if sorted(f["function"]["name"] for f in out) != sorted(TOOL_NAMES):
        raise RuntimeError(f"expected {TOOL_NAMES} from agent_mcp, got {[f['function']['name'] for f in out]}")
    return out


def dispatch(name: str, args: dict, dirs: SimpleNamespace) -> str:
    """Run one tool call against an arm's shadow state, serialised as the
    aggregator would. Synchronous on purpose: the module constants it repoints
    are process-wide, and no await may fall between repointing and the call."""
    from agent_mcp import facts, session
    args = args if isinstance(args, dict) else {}
    if name == "fact_get":
        _point_store(dirs)
        result = facts._fact_get(args)
    elif name == "memory_read":
        real = session.MEMORIES_ROOT
        session.MEMORIES_ROOT = dirs.memories
        try:
            result = session._memory_read(args)
        finally:
            session.MEMORIES_ROOT = real
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result, default=str)


def render(probe: Probe, dirs: SimpleNamespace, arm: str) -> list[dict]:
    """The messages a trial sends: system (SOUL + memory), the probe, and a
    `fact_get` exchange on the entity answered from THIS arm's store."""
    import prompt_builder
    soul = prompt_builder._load_soul(None) or ""
    system = soul
    memories = prompt_builder._load_memories(dirs.memories, soul=soul, files=("MEMORY.md",))
    if memories:
        system = f"{soul}\n\n<memory>\n{memories}\n</memory>"
    result = dispatch("fact_get", {"entity": probe.entity}, dirs)
    call_id = f"call_{_slug(probe.entity)[:24]}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": probe.question},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "fact_get", "arguments": json.dumps({"entity": probe.entity})}}]},
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def _fold(text: str) -> str:
    """Diacritics off and digit-group commas out: "Valparaíso" names the
    corpus's "Valparaiso", and "16,000" its "16000" (the first run scored a
    correct Valparaíso answer as stale before this)."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)


def _mentions(text: str, terms: list[str]) -> bool:
    flat = _fold(text)
    return any(re.search(rf"(?<![\w.]){re.escape(t)}(?![\w])", flat, re.IGNORECASE)
               for t in terms)


def stale_evidence(probe: Probe, messages: list[dict]) -> bool:
    """Whether the superseded value is anywhere the model can read it."""
    if not probe.superseded:
        return False
    return any(_mentions(str(m.get("content") or ""), probe.match_old) for m in messages)


def score(probe: Probe, answer: str | None) -> str:
    """`correct` / `stale` / `mixed` / `none`, or `error` when no answer came back.

    A superseded probe answered with the old value is `stale` — a miss, not an
    unanswered probe. A control has no old value, so it is `correct` or `none`.
    `mixed` (both values named) is only a first pass: `settle` asks which one
    the answer acted on.
    """
    if answer is None:
        return "error"
    new = _mentions(answer, probe.match_new)
    old = bool(probe.match_old) and _mentions(answer, probe.match_old)
    if new and old:
        return "mixed"
    if new:
        return "correct"
    if old:
        return "stale"
    return "none"


#: What a mixed answer resolves to. An answer that names both values but
#: commits to the current one, flagging the old as older, is `correct` — that
#: is the behaviour wanted; one that commits to the old is `stale`; one that
#: will not choose is `hedged`, which is no stale action but no action either.
JUDGE_TO_SCORE = {"current": "correct", "superseded": "stale",
                  "both_unresolved": "hedged", "neither": "none"}
JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["acted_on"],
                "properties": {"acted_on": {"type": "string", "enum": sorted(JUDGE_TO_SCORE)}}}
JUDGE_PROMPT = """An assistant was asked:

<question>{question}</question>

Its records held two values for this: CURRENT = {new}, SUPERSEDED = {old}.
(Numbers may be written with digit separators, e.g. 16000 as 16,000.)
Its answer:

<answer>
{answer}
</answer>

Which value did the answer act on — the value in the command it gave, the
name or number it told the user to use? Mentioning the other value as a
caveat or as history does not count as acting on it.

- "current": it committed to {new}
- "superseded": it committed to {old}
- "both_unresolved": it gave both, or refused to choose, or left a placeholder
- "neither": it committed to neither value

Reply with the JSON object only."""


def settle(first: str, judged: str | None) -> str:
    """The final score for a trial: the first-pass score, or for a `mixed`
    one the judge's reading of it (`error` when the judge gave none)."""
    if first != "mixed":
        return first
    return JUDGE_TO_SCORE.get(judged or "", "error")


# ─────────────────────────────────────────────────────────────── engine ──

async def _complete(client, base_url: str, model: str, messages: list[dict], tools: list[dict],
                    max_tokens: int, tool_choice: str = "auto") -> dict:
    body = {"model": model, "messages": messages, "tools": tools, "tool_choice": tool_choice,
            "max_tokens": max_tokens, "priority": 1}
    r = await client.post(f"{base_url}/v1/chat/completions", json=body)
    r.raise_for_status()
    data = r.json()
    return {"message": data["choices"][0]["message"],
            "finish": data["choices"][0].get("finish_reason"),
            "completion_tokens": (data.get("usage") or {}).get("completion_tokens") or 0}


async def run_trial(client, base_url: str, model: str, messages: list[dict], tools: list[dict],
                    dirs: SimpleNamespace, max_tokens: int, max_iterations: int,
                    complete=None) -> dict:
    """One trial: the model may call `fact_get`/`memory_read` (answered from the
    arm's shadow state) for up to `max_iterations` completions; the answer is
    the first text written with no tool call. At the cap one more completion is
    asked with `tool_choice: none`, the finalizer's shape."""
    complete = complete or _complete
    msgs = list(messages)
    t0 = time.monotonic()
    tokens = 0
    calls: list[str] = []
    try:
        for it in range(max_iterations + 1):
            last = it == max_iterations
            out = await complete(client, base_url, model, msgs, tools, max_tokens,
                                 tool_choice="none" if last else "auto")
            tokens += out["completion_tokens"]
            msg = out["message"]
            tcs = msg.get("tool_calls") or []
            if not tcs or last:
                return {"answer": msg.get("content") or "", "finish": out["finish"],
                        "iterations": it + 1, "tool_calls": calls,
                        "completion_tokens": tokens, "latency_s": round(time.monotonic() - t0, 2)}
            reasoning = msg.get("reasoning") or msg.get("reasoning_content")
            amsg = {"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs}
            if reasoning:
                amsg["reasoning"] = amsg["reasoning_content"] = reasoning
            msgs.append(amsg)
            for tc in tcs:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append(f"{fn.get('name')}({json.dumps(args, sort_keys=True)})")
                msgs.append({"role": "tool", "tool_call_id": tc.get("id"),
                             "content": dispatch(fn.get("name") or "", args, dirs)})
    except Exception as exc:  # recorded, never raised: one failure is one `error` row
        return {"answer": None, "error": f"{type(exc).__name__}: {exc}", "tool_calls": calls,
                "completion_tokens": tokens, "latency_s": round(time.monotonic() - t0, 2)}
    raise AssertionError("unreachable")  # pragma: no cover


async def judge(client, base_url: str, model: str, probe: Probe, answer: str) -> str | None:
    """Which value a `mixed` answer acted on, asked of the primary at
    temperature 0 with its thinking ON and a JSON grammar after it. Thinking
    off misread 2 of the first run's 6 stale verdicts (a table naming the old
    figure beside the one the answer led with). None when unreadable."""
    body = {"model": model, "max_tokens": 4096, "priority": 1, "temperature": 0,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "acted_on", "schema": JUDGE_SCHEMA}},
            "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
                question=probe.question, new=probe.value, old=probe.old_value, answer=answer)}]}
    try:
        r = await client.post(f"{base_url}/v1/chat/completions", json=body)
        r.raise_for_status()
        verdict = json.loads(r.json()["choices"][0]["message"]["content"] or "{}").get("acted_on")
        return verdict if verdict in JUDGE_TO_SCORE else None
    except Exception:
        return None


async def judge_mixed(trials: list[dict], probes: list[Probe], *, base_url: str, model: str,
                      concurrency: int, judge_fn=None) -> None:
    """Fill `judged` and the final `score` on every trial."""
    import httpx
    judge_fn = judge_fn or judge
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10)) as client:
        async def one(t):
            if t["first_score"] == "mixed":
                async with sem:
                    t["judged"] = await judge_fn(client, base_url, model, probes[t["probe"]], t["answer"])
            t["score"] = settle(t["first_score"], t.get("judged"))
        await asyncio.gather(*(one(t) for t in trials))


async def run_trials(trials: list[dict], *, base_url: str, model: str, tools: list[dict],
                     concurrency: int, max_tokens: int, max_iterations: int = 4,
                     complete=None) -> None:
    import httpx
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=httpx.Timeout(900, connect=10)) as client:
        async def one(t):
            async with sem:
                t.update(await run_trial(client, base_url, model, t.pop("_messages"), tools,
                                         t.pop("_dirs"), max_tokens, max_iterations, complete))
        await asyncio.gather(*(one(t) for t in trials))


# ─────────────────────────────────────────────────────────────── report ──

def _ci(d: dict) -> list | None:
    """`[lo, hi]` off one of `stats`' interval dicts, or None when it has none."""
    if not d or d.get("lo") is None:
        return None
    return [round(d["lo"], 4), round(d["hi"], 4)]


def summarize(probes: list[Probe], trials: list[dict], arms: list[str]) -> dict:
    """Per-arm block: stale-action rate with numerator/denominator and CIs,
    control accuracy beside it, stale-evidence count, and memory gain."""
    from stats import bootstrap_mean_ci, paired_bootstrap_ci, wilson_ci
    by = {(t["arm"], t["probe"]): [] for t in trials}
    for t in trials:
        by[(t["arm"], t["probe"])].append(t)
    out: dict = {}
    correct_by_arm: dict[str, list[float]] = {}
    for arm in arms:
        sup = [t for t in trials if t["arm"] == arm and t["superseded"]]
        ctl = [t for t in trials if t["arm"] == arm and not t["superseded"]]
        counts = {k: sum(t["score"] == k for t in sup) for k in ("correct", "stale", "hedged", "none", "error")}
        flagged = sum(t["score"] == "correct" and t.get("first_score") == "mixed" for t in sup)
        n = len(sup) - counts["error"]
        ctl_ok = sum(t["score"] == "correct" for t in ctl)
        ctl_n = sum(t["score"] != "error" for t in ctl)
        per_probe_stale = []
        per_probe_correct = []
        for i, p in enumerate(probes):
            rows = [t for t in by.get((arm, i), []) if t["score"] != "error"]
            if not rows:
                per_probe_correct.append(0.0)
                continue
            per_probe_correct.append(sum(t["score"] == "correct" for t in rows) / len(rows))
            if p.superseded:
                per_probe_stale.append(sum(t["score"] == "stale" for t in rows) / len(rows))
        correct_by_arm[arm] = per_probe_correct
        evid = sorted({t["probe"] for t in sup if t["stale_evidence"]})
        out[arm] = {
            "stale_action": {"rate": round(counts["stale"] / n, 4) if n else None,
                             "numerator": counts["stale"], "denominator": n,
                             "wilson95": [round(x, 4) for x in wilson_ci(counts["stale"], n)] if n else None,
                             "probe_bootstrap95": _ci(bootstrap_mean_ci(per_probe_stale))
                             if len(per_probe_stale) > 1 else None},
            "hedged": {"numerator": counts["hedged"], "denominator": n,
                       "rate": round(counts["hedged"] / n, 4) if n else None},
            "correct_with_old_flagged": flagged,
            "superseded_outcomes": counts,
            "superseded_correct_rate": round(counts["correct"] / n, 4) if n else None,
            "control_accuracy": {"rate": round(ctl_ok / ctl_n, 4) if ctl_n else None,
                                 "numerator": ctl_ok, "denominator": ctl_n},
            "stale_evidence": {"probes_with_old_value_in_context": len(evid),
                               "superseded_probes": sum(p.superseded for p in probes)},
            "overall_correct_rate": round(sum(per_probe_correct) / len(per_probe_correct), 4),
        }
        if any(t.get("old_after_new") is not None for t in sup):
            out[arm]["by_position"] = {
                label: {"stale": sum(t["score"] == "stale" for t in rows), "n": len(rows)}
                for label, rows in (
                    ("old_line_last", [t for t in sup if t.get("old_after_new")]),
                    ("new_line_last", [t for t in sup if t.get("old_after_new") is False]))}
    if "stateless" in arms:
        base = correct_by_arm["stateless"]
        for arm in arms:
            if arm == "stateless":
                continue
            ci = paired_bootstrap_ci(base, correct_by_arm[arm])  # CI of arm - stateless
            out[arm]["memory_gain"] = {
                "stateful_rate": out[arm]["overall_correct_rate"],
                "stateless_rate": out["stateless"]["overall_correct_rate"],
                "gain": round(out[arm]["overall_correct_rate"] - out["stateless"]["overall_correct_rate"], 4),
                "paired_bootstrap95": _ci(ci)}
    return out


# ─────────────────────────────────────────────────────────────── main ──

def _primary(base_url: str | None) -> tuple[str, str]:
    import httpx
    import yaml
    if not base_url:
        cfg = yaml.safe_load((LLOYD_HOME / "config.yaml").read_text()) or {}
        base_url = ((cfg.get("models") or {}).get("primary") or {}).get("base_url") or "http://127.0.0.1:8096"
    model = httpx.get(f"{base_url}/v1/models", timeout=10).json()["data"][0]["id"]
    return base_url.rstrip("/"), model


def main(argv: list[str] | None = None, *, complete=None, judge_fn=None,
         primary=None) -> int:
    """`complete`, `judge_fn` and `primary` replace the engine for tests."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--samples", type=int, default=3, help="generations per probe per arm")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--max-iterations", type=int, default=4,
                    help="completions a trial may spend calling tools before it must answer")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--seed", type=int, default=622)
    ap.add_argument("--limit", type=int, default=0, help="first N probes only (smoke)")
    ap.add_argument("--dry-run", action="store_true", help="plant and render; no engine calls")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep", action="store_true", help="keep the shadow memory it made")
    ap.add_argument("--rejudge", default=None, metavar="REPORT",
                    help="re-score a saved report's answers and re-ask the judge for every "
                         "mixed one, in place; engine calls for the judge only")
    ap.add_argument("--resummarize", default=None, metavar="REPORT",
                    help="recompute the summary of a saved report in place; no engine calls")
    args = ap.parse_args(argv)
    if args.resummarize or args.rejudge:
        path = Path(args.resummarize or args.rejudge)
        report = json.loads(path.read_text())
        probes = [Probe(**{**p, "facts": [tuple(f) for f in p["facts"]]}) for p in report["probes"]]
        if args.rejudge:
            base_url, model = (primary or _primary)(args.base_url)
            for t in report["trials"]:
                t["first_score"] = score(probes[t["probe"]], t.get("answer"))
                t.pop("judged", None)
            asyncio.run(judge_mixed(report["trials"], probes, base_url=base_url, model=model,
                                    concurrency=args.concurrency, judge_fn=judge_fn))
            report["judge"] = "primary, thinking on, temperature 0"
        report["summary"] = summarize(probes, report["trials"], report["arms"])
        path.write_text(json.dumps(report, indent=1, default=str))
        print(json.dumps(report["summary"], indent=1))
        return 0

    made = None
    if not os.environ.get("LLOYD_FACTS_ROOT") and not os.environ.get("LLOYD_KG_DB"):
        made = Path(tempfile.mkdtemp(prefix="stale-fact-622-"))
        os.environ["LLOYD_FACTS_ROOT"] = str(made / "facts")
        os.environ["LLOYD_KG_DB"] = str(made / "kg.sqlite")
    try:
        with restored_store_pointers():
            return _run(ap, args, complete=complete, judge_fn=judge_fn, primary=primary)
    finally:
        if made is not None:
            for var in ("LLOYD_FACTS_ROOT", "LLOYD_KG_DB"):
                os.environ.pop(var, None)
            if not args.keep:
                shutil.rmtree(made, ignore_errors=True)
            else:
                print(f"shadow memory kept at {made}", file=sys.stderr)


def _run(ap, args, *, complete=None, judge_fn=None, primary=None) -> int:
    try:
        facts_root, _db = check_shadow_paths()
    except LiveStoreRefused as exc:
        print(f"LiveStoreRefused: {exc}", file=sys.stderr)
        return 2
    shadow = facts_root.parent
    shadow.mkdir(parents=True, exist_ok=True)

    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        ap.error(f"unknown arm(s): {bad}")
    probes = build_corpus(args.seed)
    if args.limit:
        probes = probes[:args.limit]
    session_id = f"pt-eval-stale-fact-{int(time.time())}"
    from agent_mcp._tool_sandbox import is_sandboxed_session
    assert is_sandboxed_session(session_id), session_id

    dirs = {arm: plant(probes, shadow, arm, args.seed) for arm in arms}
    tools = _tools()
    trials: list[dict] = []
    for arm in arms:
        for i, p in enumerate(probes):
            msgs = render(p, dirs[arm], arm)
            ev = stale_evidence(p, msgs)
            for k in range(args.samples):
                trials.append({"arm": arm, "probe": i, "sample": k, "entity": p.entity,
                               "slot": p.slot, "superseded": p.superseded, "stale_evidence": ev,
                               "old_after_new": dirs[arm].old_last.get(p.entity),
                               "_messages": msgs, "_dirs": dirs[arm]})
    counts = corpus_counts(probes)
    header = {"item": 622, "session_label": session_id, "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
              "corpus": counts, "arms": arms, "samples": args.samples, "seed": args.seed,
              "shadow_root": str(shadow), "max_tokens": args.max_tokens,
              "max_iterations": args.max_iterations}
    t0 = time.monotonic()
    if args.dry_run:
        for t in trials:
            t.pop("_messages")
            t.pop("_dirs")
            t["score"] = "error"
    else:
        base_url, model = (primary or _primary)(args.base_url)
        header.update(base_url=base_url, model=model)
        asyncio.run(run_trials(trials, base_url=base_url, model=model, tools=tools,
                               concurrency=args.concurrency, max_tokens=args.max_tokens,
                               max_iterations=args.max_iterations, complete=complete))
        for t in trials:
            t["first_score"] = score(probes[t["probe"]], t.get("answer"))
        asyncio.run(judge_mixed(trials, probes, base_url=base_url, model=model,
                                concurrency=args.concurrency, judge_fn=judge_fn))
        header["judge"] = "primary, thinking on, temperature 0"
    header["wall_s"] = round(time.monotonic() - t0, 1)
    report = {**header, "summary": summarize(probes, trials, arms),
              "probes": [asdict(p) for p in probes], "trials": trials}
    out = Path(args.out) if args.out else DEFAULT_OUT_DIR / (
        f"stale-fact-{_dt.date.today().isoformat()}{'-dry' if args.dry_run else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"corpus": counts, "wall_s": header["wall_s"], "out": str(out),
                      "summary": report["summary"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
