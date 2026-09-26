#!/usr/bin/env python3
"""Build LloydMemEval's frozen question set from real facts (#1480).

The retained conversational corpus cannot supply a memory eval: ~13 human chat
sessions survive the 2026-09-22 wipe, 6 with three user turns or more. So the
set is SYNTHETIC, and grounded: every question is built from one or two real
fact records of the knowledge graph (a snapshot of `_pipeline/vault-derived/
facts`), the primary writes the dated conversations that carry those facts and
the question over them, and a mechanical check refuses anything whose answer is
not in the fact it came from, not in the conversation that is its evidence, or
already in the question. The facts make the answers checkable; the synthetic
sessions make the categories exist.

Three stages, each resumable, raw artifacts under ~/lloyd-data/eval/1480/:

  sample    pick grounded episodes (category, facts, session dates) — no model
  generate  the primary writes sessions + question per episode (thinking off,
            JSON grammar); every draft is validated, a refusal is retried
  freeze    write eval/memory_eval/<version>/: dev/ and holdout/ questions and
            sessions, and a manifest with a sha256 per file, the generating
            model, and the holdout split hash

Categories (LongMemEval's five):
  single_session    one fact, said by the user mid-session among other topics
  multi_session     two facts from two sessions; the answer needs both
  knowledge_update  an earlier session states an invented old value, a later one
                    the real (current) one; action-shaped probe, old value = anti
  temporal          dates carry the answer: which came first, days between, or
                    what was said on a given date
  preference        a real preference fact stated by the user; the probe is a
                    request the preference should shape (a "use" probe)

The generator is the primary; `run_memory_eval.py` refuses to judge with it.

Usage (hold the primary lock for `generate`):
    python eval/memory_eval_build.py sample
    flock -s ~/.local/state/lloyd-automod/primary.lock python eval/memory_eval_build.py generate
    python eval/memory_eval_build.py freeze --version v1
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLOYD_HOME = HERE.parent
sys.path.insert(0, str(LLOYD_HOME))
sys.path.insert(0, str(HERE))

WORK = Path(os.path.expanduser("~/lloyd-data/eval/1480"))
CORPUS = WORK / "corpus"
EPISODES = WORK / "episodes.jsonl"
DRAFTS = WORK / "drafts.jsonl"
OUT_ROOT = HERE / "memory_eval"

CATEGORIES = ("single_session", "multi_session", "knowledge_update", "temporal", "preference")
PLAN = {"single_session": 74, "multi_session": 72, "knowledge_update": 72,
        "temporal": 74, "preference": 68}
TEMPORAL_KINDS = ("order", "days_between", "on_date")
HOLDOUT_FRACTION = 0.2
HOLDOUT_SALT = "lloydmemeval-holdout-v1"
MAX_PER_ENTITY = 2
SEED = 1480

_BAD_ENTITY = re.compile(r"^(\d[\d\-_. ]*|#.*|.*\.md|\d{4}-\d{2}-\d{2}.*)$")
_PRIVATE = re.compile(r"@|\+?\d[\d\s\-()]{8,}\d|password|passwd|token|api[_ ]?key|secret", re.I)


# ─────────────────────────────────────────────────────────────── text ──

# One normaliser for "the gold is in the fact" here and "the gold is in the
# answer" in the runner.
from run_memory_eval import contains, fold  # noqa: E402,F401


# ─────────────────────────────────────────────────────────────── sample ──

def _anchored(fact: str, entity: str) -> bool:
    """A fact with something checkable in it: a number, a code-ish token, or a
    capitalised name other than the entity's own."""
    if re.search(r"\d", fact):
        return True
    if re.search(r"\b\w+[_/.:]\w+", fact):
        return True
    ent = set(entity.lower().split())
    caps = [w for w in re.findall(r"(?<!^)(?<![.!?] )\b[A-Z][\w\-]+", fact)
            if w.lower() not in ent]
    return len(caps) >= 1


def load_corpus(root: Path) -> dict[str, list[dict]]:
    """entity -> active facts (with `_file`), off a facts-tree snapshot."""
    import yaml
    out: dict[str, list[dict]] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or _BAD_ENTITY.match(d.name) or len(d.name) < 3:
            continue
        facts: list[dict] = []
        for f in sorted(d.glob("*.md")):
            try:
                raw = f.read_text(errors="replace")
                if not raw.startswith("---"):
                    continue
                meta = yaml.safe_load(raw.split("---", 2)[1]) or {}
            except Exception:  # noqa: BLE001 — one unreadable file is one skipped file
                continue
            for x in meta.get("facts") or []:
                if not isinstance(x, dict) or x.get("expired_at") or x.get("invalid_at"):
                    continue
                text = str(x.get("fact") or "").strip()
                if not text or not x.get("id"):
                    continue
                facts.append({"id": str(x["id"]), "fact": text,
                              "confidence": float(x.get("confidence") or 0.0),
                              "category": str(x.get("category") or ""),
                              "created_at": str(x.get("created_at") or ""),
                              "_file": f"{d.name}/{f.name}"})
        if facts:
            out[d.name] = facts
    return out


def _conf_rank(facts: list[dict], fid: str, ffile: str) -> int:
    """Where prefetch's confidence sort puts this fact (0-based; stable)."""
    order = sorted(facts, key=lambda f: f.get("confidence", 0.0), reverse=True)
    for i, f in enumerate(order):
        if f["id"] == fid and f["_file"] == ffile:
            return i
    return -1


def sample(seed: int = SEED, plan: dict | None = None, corpus: Path = CORPUS) -> list[dict]:
    """Pick the episodes. Deterministic in `seed` and the corpus snapshot."""
    os.environ.setdefault("LLOYD_FACTS_ROOT", str(corpus / "facts"))
    os.environ.setdefault("LLOYD_KG_DB", str(corpus / "kg.sqlite"))
    from agent_mcp.retrieval import extract_entities_from_query

    plan = dict(plan or PLAN)
    rng = random.Random(seed)
    ents = load_corpus(corpus / "facts")
    usable: dict[str, list[dict]] = {}
    for e, facts in ents.items():
        if not (3 <= len(facts) <= 400):
            continue
        good = [f for f in facts if 40 <= len(f["fact"]) <= 240
                and not _PRIVATE.search(f["fact"]) and _anchored(f["fact"], e)]
        if good:
            usable[e] = good
    names = sorted(usable)
    rng.shuffle(names)

    reach_cache: dict[str, bool] = {}

    def reachable(e: str) -> bool:
        if e not in reach_cache:
            got = [x[0].lower() for x in extract_entities_from_query(f"Tell me about {e}")[:2]]
            reach_cache[e] = e.lower() in got
        return reach_cache[e]

    used: dict[str, int] = {}
    episodes: list[dict] = []

    def take(e: str) -> bool:
        if used.get(e, 0) >= MAX_PER_ENTITY or not reachable(e):
            return False
        used[e] = used.get(e, 0) + 1
        return True

    def fact_ref(e: str, f: dict) -> dict:
        return {"entity": e, "id": f["id"], "file": f["_file"], "fact": f["fact"],
                "confidence": f["confidence"], "category": f["category"],
                "conf_rank": _conf_rank(ents[e], f["id"], f["_file"]),
                "entity_facts": len(ents[e])}

    def dates(n: int) -> tuple[list[str], str]:
        base = _dt.date(2026, 5, 1) + _dt.timedelta(days=rng.randrange(0, 120))
        out = [base]
        for _ in range(n - 1):
            out.append(out[-1] + _dt.timedelta(days=rng.randrange(2, 13)))
        asked = out[-1] + _dt.timedelta(days=rng.randrange(1, 8))
        return [d.isoformat() for d in out], asked.isoformat()

    pref = [e for e in names if any(f["category"] == "preference" for f in usable[e])]
    cursor = {c: 0 for c in CATEGORIES}
    tk = 0
    for cat in CATEGORIES:
        want = plan.get(cat, 0)
        pool = pref if cat == "preference" else names
        got = 0
        while got < want and cursor[cat] < len(pool):
            e = pool[cursor[cat]]
            cursor[cat] += 1
            facts = usable[e]
            ep: dict | None = None
            if cat in ("single_session", "knowledge_update"):
                if cat == "knowledge_update":
                    facts = [f for f in facts if re.search(r"\d", f["fact"])] or []
                if not facts or not take(e):
                    continue
                f = rng.choice(facts)
                d, asked = dates(2 if cat == "knowledge_update" else 1)
                ep = {"facts": [fact_ref(e, f)], "dates": d, "asked_on": asked}
            elif cat == "preference":
                pf = [f for f in facts if f["category"] == "preference"]
                if not pf or not take(e):
                    continue
                d, asked = dates(1)
                ep = {"facts": [fact_ref(e, rng.choice(pf))], "dates": d, "asked_on": asked}
            elif cat == "multi_session":
                if len(facts) < 2 or not take(e):
                    continue
                a, b = rng.sample(facts, 2)
                d, asked = dates(2)
                ep = {"facts": [fact_ref(e, a), fact_ref(e, b)], "dates": d, "asked_on": asked}
            elif cat == "temporal":
                kind = TEMPORAL_KINDS[tk % len(TEMPORAL_KINDS)]
                if kind == "on_date":
                    if len(facts) < 2 or not take(e):
                        continue
                    a, b = rng.sample(facts, 2)
                    refs = [fact_ref(e, a), fact_ref(e, b)]
                else:
                    # a second entity, disjoint, from further down the list
                    other = next((o for o in names[::-1] if o != e and used.get(o, 0) < MAX_PER_ENTITY
                                  and reachable(o)), None)
                    if other is None or not take(e):
                        continue
                    used[other] = used.get(other, 0) + 1
                    refs = [fact_ref(e, rng.choice(facts)), fact_ref(other, rng.choice(usable[other]))]
                d, asked = dates(2)
                ep = {"facts": refs, "dates": d, "asked_on": asked, "temporal_kind": kind}
                if kind == "days_between":
                    d0, d1 = (_dt.date.fromisoformat(x) for x in d)
                    ep["days_between"] = (d1 - d0).days
                tk += 1
            if ep is None:
                continue
            ep["category"] = cat
            ep["id"] = f"lme-{cat[:2]}-{got + 1:03d}"
            episodes.append(ep)
            got += 1
    return episodes


# ─────────────────────────────────────────────────────────────── generate ──

GEN_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["sessions", "question", "gold", "aliases", "anti", "evidence", "probe"],
    "properties": {
        "sessions": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["sid", "turns"],
            "properties": {
                "sid": {"type": "string"},
                "turns": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["role", "text"],
                    "properties": {"role": {"type": "string", "enum": ["user", "assistant"]},
                                   "text": {"type": "string"}}}}}}},
        "question": {"type": "string"},
        "gold": {"type": "array", "items": {"type": "string"}},
        "aliases": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
        "anti": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "probe": {"type": "string", "enum": ["recall", "action"]},
    },
}

_COMMON = """You are building one item of a memory benchmark for Lloyd, Alan's personal AI
assistant. You write short, realistic past chat sessions between Alan (role "user")
and Lloyd (role "assistant"), and ONE later question from Alan that can only be
answered by remembering those sessions.

Rules for every item:
- Sessions are casual and specific, 6-10 turns each, and each also covers one or two
  unrelated everyday topics (a bug, dinner plans, a hardware order, the weather)
  so the memory is buried, not the whole point of the chat.
- Alan says the remembered fact in his OWN words in a user turn. Lloyd may
  acknowledge it but must not restate the key value more than once.
- The question must NOT contain the answer, must name the subject ("{entity}")
  naturally, and must read like something Alan would type days later.
- `gold` is the SHORT key value(s) the answer must contain — a number, name, version,
  path, word or short phrase — copied EXACTLY as it appears in the fact text below.
  Never a sentence: at most 6 words, ideally 1-3. `aliases` gives, per gold entry, 0-3 other ways to write it
  (e.g. "8 GB" for "8GB"). `evidence` lists the sid(s) holding the answer.
- Use sids exactly: {sids}. Session dates (for your reference): {dates}.
  The question is asked on {asked_on}.
"""

_RECIPES = {
    "single_session": """Category: single-session recall.
One session ({sids}). The fact below comes up in passing, mid-session, between other topics.
Fact about {entity}: "{f0}"
Write a question asking for the key value of that fact. probe = "recall". anti = [].""",
    "multi_session": """Category: multi-session recall.
Two sessions ({sids}), on different days, each about other things too. Session s1 carries
fact A, session s2 carries fact B (both about {entity}).
Fact A: "{f0}"
Fact B: "{f1}"
Write ONE question whose full answer needs BOTH facts (e.g. "what two things did I tell
you about {entity}'s ...", or a question that combines them). gold = [key value of A,
key value of B], each copied from its fact. probe = "recall". anti = [].""",
    "knowledge_update": """Category: knowledge update.
Two sessions. In s1 (earlier) Alan states an OLDER value for this fact that you invent: same
kind of value, clearly different, plausible. In s2 (later) he says it changed and gives the
CURRENT value, which is the one in the real fact:
Fact about {entity} (current): "{f0}"
Write an ACTION-shaped question (ask Lloyd to do something that needs the value: write a
command, fill a form field, pick an option, compute with it) — not "what is X now".
gold = [the current key value from the fact]. anti = [the old value you invented].
evidence = ["s2"]. probe = "action".""",
    "temporal_order": """Category: temporal reasoning (order).
Two sessions: s1 on {d0} mentions fact A (about {entity}); s2 on {d1} mentions fact B
(about {entity2}). Each session happens on its date; Alan refers to what he did or found
"today" or "this morning".
Fact A: "{f0}"
Fact B: "{f1}"
Pick a short label for each topic (1-4 words, copied from its fact, e.g. "{entity}" and
"{entity2}") and ask which of the two Alan brought up FIRST, naming both by EXACTLY those
labels, without hinting at the order. gold = [A's label]. anti = [B's label]. aliases may add
other short names for A that appear in fact A. probe = "recall".""",
    "temporal_days_between": """Category: temporal reasoning (elapsed time).
Two sessions: s1 on {d0} mentions fact A (about {entity}); s2 on {d1} mentions fact B
(about {entity2}). Alan does not state the dates in the chat text; the session dates are
the only clock.
Fact A: "{f0}"
Fact B: "{f1}"
Ask how many days passed between when he told Lloyd about A and when he told Lloyd about B.
Set gold = ["{days}"] and aliases = [["{days} days"]]. anti = []. probe = "recall".""",
    "temporal_on_date": """Category: temporal reasoning (by date).
Two sessions about {entity}: s1 on {d0} carries fact A, s2 on {d1} carries fact B.
Fact A: "{f0}"
Fact B: "{f1}"
Ask what Alan told Lloyd about {entity} on {d0_words} (name that date, not the other).
gold = [the key value of fact A]. anti = [the key value of fact B]. evidence = ["s1"].
probe = "recall".""",
    "preference": """Category: preference.
One session ({sids}). Alan states this preference or rule about {entity} as his own view:
Preference: "{f0}"
Write a REQUEST (recommend, choose, draft, plan) where a good answer should APPLY that
preference without being reminded of it. gold = [the short word or phrase from the
preference that an answer applying it would use]. anti = [the short word or phrase a
generic answer that ignores the preference would use instead, if there is an obvious one,
else leave empty]. probe = "action".""",
}


def _words(d: str) -> str:
    x = _dt.date.fromisoformat(d)
    return x.strftime("%A, %B ") + str(x.day)


def render_prompt(ep: dict) -> str:
    n = len(ep["dates"])
    sids = [f"s{i + 1}" for i in range(n)]
    f = ep["facts"]
    key = ep["category"] if ep["category"] != "temporal" else f"temporal_{ep['temporal_kind']}"
    fill = {"entity": f[0]["entity"], "entity2": f[-1]["entity"], "f0": f[0]["fact"],
            "f1": f[-1]["fact"], "sids": ", ".join(sids),
            "dates": ", ".join(f"{s}={d}" for s, d in zip(sids, ep["dates"])),
            "asked_on": ep["asked_on"], "d0": ep["dates"][0], "d1": ep["dates"][-1],
            "d0_words": _words(ep["dates"][0]), "days": ep.get("days_between", "")}
    return (_COMMON.format(**fill) + "\n" + _RECIPES[key].format(**fill)
            + "\n\nReply with the JSON object only.")


def validate(ep: dict, draft: dict) -> list[str]:
    """Why this draft cannot be frozen ([] = accepted). Mechanical, no model."""
    errs: list[str] = []
    n = len(ep["dates"])
    sids = [f"s{i + 1}" for i in range(n)]
    got = [s.get("sid") for s in draft.get("sessions") or []]
    if got != sids:
        errs.append(f"sessions {got} != {sids}")
        return errs
    for s in draft["sessions"]:
        turns = s.get("turns") or []
        if len(turns) < 4 or not any(t["role"] == "user" for t in turns):
            errs.append(f"{s['sid']}: too short")
    gold = [str(g).strip() for g in draft.get("gold") or [] if str(g).strip()]
    want = 2 if ep["category"] == "multi_session" else 1
    if len(gold) != want:
        errs.append(f"gold has {len(gold)} entries, want {want}")
        return errs
    aliases = draft.get("aliases") or []
    q = draft.get("question") or ""
    text = {s["sid"]: " ".join(t["text"] for t in s["turns"]) for s in draft["sessions"]}
    evidence = [e for e in draft.get("evidence") or [] if e in text] or sids
    ev_text = " ".join(text[e] for e in evidence)
    all_text = " ".join(text.values())
    kind = ep.get("temporal_kind")
    for i, g in enumerate(gold):
        forms = [g] + [a for a in (aliases[i] if i < len(aliases) else []) if a]
        if len(g) > 50 or len(g.split()) > 6:
            errs.append(f"gold {i} is a sentence, not a value (<=6 words)")
        if kind == "days_between":
            if fold(g) != str(ep["days_between"]):
                errs.append("days_between gold is not the computed day count")
            continue
        fact = ep["facts"][i if ep["category"] == "multi_session" else 0]["fact"]
        if not any(contains(fact, x) for x in forms):
            errs.append(f"gold {g!r} not in its fact")
        hay = all_text if ep["category"] == "multi_session" else ev_text
        if not any(contains(hay, x) for x in forms):
            errs.append(f"gold {g!r} not in the evidence session")
        if kind == "order":
            # both options are named by design; the answer is the ORDER
            if not contains(q, g) or not all(contains(q, a) for a in draft.get("anti") or []):
                errs.append("order question must name both labels verbatim")
        elif any(contains(q, x) for x in forms):
            errs.append(f"question leaks gold {g!r}")
    anti = [str(a).strip() for a in draft.get("anti") or [] if str(a).strip()]
    for a in anti:
        if any(contains(a, g) or contains(g, a) for g in gold):
            errs.append(f"anti {a!r} overlaps gold")
    if ep["category"] == "knowledge_update":
        if not anti:
            errs.append("knowledge_update needs the invented old value as anti")
        elif not contains(text["s1"], anti[0]):
            errs.append("old value not stated in s1")
        elif contains(ep["facts"][0]["fact"], anti[0]):
            errs.append("old value is in the real fact")
    if ep["category"] == "temporal" and kind in ("order", "on_date") and not anti:
        errs.append("temporal order/on_date needs the other side as anti")
    return errs


async def _generate_one(client, base_url: str, model: str, ep: dict, attempts: int = 3,
                        seed_offset: int = 0) -> dict:
    prompt = render_prompt(ep)
    last: dict = {}
    for attempt in range(attempts):
        body = {"model": model, "max_tokens": 6000, "priority": 1, "temperature": 0.8,
                "seed": SEED + seed_offset + attempt, "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "memory_item", "schema": GEN_SCHEMA}},
                "messages": [{"role": "user", "content": prompt}]}
        try:
            r = await client.post(f"{base_url}/v1/chat/completions", json=body)
            r.raise_for_status()
            draft = json.loads(r.json()["choices"][0]["message"]["content"] or "{}")
        except Exception as exc:  # noqa: BLE001 — recorded, retried
            last = {"id": ep["id"], "ok": False, "attempt": attempt, "errors": [repr(exc)[:300]]}
            continue
        errs = validate(ep, draft)
        last = {"id": ep["id"], "ok": not errs, "attempt": attempt, "errors": errs,
                "draft": draft, "model": model}
        if not errs:
            return last
    return last


async def generate(episodes: list[dict], *, base_url: str, model: str, concurrency: int,
                   out: Path = DRAFTS, seed_offset: int = 0) -> None:
    import httpx
    done = {}
    if out.exists():
        for ln in out.read_text().splitlines():
            row = json.loads(ln)
            if row.get("ok"):
                done[row["id"]] = row
    todo = [ep for ep in episodes if ep["id"] not in done]
    print(f"generate: {len(done)} already accepted, {len(todo)} to go")
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10)) as client:
        async def one(ep):
            async with sem:
                row = await _generate_one(client, base_url, model, ep, seed_offset=seed_offset)
            async with lock:
                with out.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"  {ep['id']}: {'ok' if row.get('ok') else 'REFUSED ' + '; '.join(row.get('errors') or [])[:160]}")
        await asyncio.gather(*(one(ep) for ep in todo))


# ─────────────────────────────────────────────────────────────── freeze ──

def holdout_ids(ids_by_cat: dict[str, list[str]], fraction: float = HOLDOUT_FRACTION) -> set[str]:
    """Stratified, hash-ordered: the split is a function of the ids alone, so it
    cannot be re-picked after results are seen."""
    out: set[str] = set()
    for cat, ids in ids_by_cat.items():
        ranked = sorted(ids, key=lambda i: hashlib.sha256(f"{HOLDOUT_SALT}:{i}".encode()).hexdigest())
        out.update(ranked[:round(len(ids) * fraction)])
    return out


def freeze(version: str, *, episodes: list[dict], drafts: dict[str, dict],
           out_root: Path = OUT_ROOT, corpus_note: str = "") -> Path:
    import yaml
    from run_memory_eval import write_manifest
    root = out_root / version
    if root.exists():
        raise SystemExit(f"{root} exists — a frozen version is never rewritten; pick a new one")
    accepted = [ep for ep in episodes if drafts.get(ep["id"], {}).get("ok")]
    by_cat: dict[str, list[str]] = {}
    for ep in accepted:
        by_cat.setdefault(ep["category"], []).append(ep["id"])
    held = holdout_ids(by_cat)
    models = sorted({drafts[ep["id"]].get("model", "") for ep in accepted})
    for leg in ("dev", "holdout"):
        (root / leg / "sessions").mkdir(parents=True)
        rows = []
        for ep in accepted:
            if (ep["id"] in held) != (leg == "holdout"):
                continue
            d = drafts[ep["id"]]["draft"]
            sess = [{"sid": f"{ep['id']}-{s['sid']}", "date": date, "turns": s["turns"]}
                    for s, date in zip(d["sessions"], ep["dates"])]
            rel = f"sessions/{ep['id']}.json"
            (root / leg / rel).write_text(json.dumps({"episode": ep["id"], "sessions": sess},
                                                     indent=1, ensure_ascii=False) + "\n")
            gold = [str(g).strip() for g in d["gold"]]
            aliases = [[str(a) for a in (d["aliases"][i] if i < len(d["aliases"]) else []) if str(a).strip()]
                       for i in range(len(gold))]
            row = {
                "id": ep["id"], "category": ep["category"], "prompt": d["question"].strip(),
                "asked_on": ep["asked_on"], "probe": d["probe"],
                "accept": {"all_of": [[g] + a for g, a in zip(gold, aliases)],
                           "none_of": [str(a).strip() for a in d.get("anti") or [] if str(a).strip()],
                           "rule": ("the answer must state or act on every all_of value; "
                                    "acting on a none_of value is not correct")},
                "evidence": [f"{ep['id']}-{e}" for e in (d.get("evidence") or [])],
                "source": {"session_file": rel,
                           "facts": [f"{f['file']}#{f['id']}" for f in ep["facts"]]},
                "grounding": [{"entity": f["entity"], "fact": f["fact"], "confidence": f["confidence"],
                               "conf_rank": f["conf_rank"], "entity_facts": f["entity_facts"]}
                              for f in ep["facts"]],
            }
            if ep.get("temporal_kind"):
                row["temporal_kind"] = ep["temporal_kind"]
            rows.append(row)
        (root / leg / "questions.yaml").write_text(
            yaml.safe_dump({"version": version, "leg": leg, "questions": rows},
                           sort_keys=False, allow_unicode=True, width=100))
    write_manifest(root, version=version, generator_model=models[0] if len(models) == 1 else ",".join(models),
                   notes=corpus_note)
    return root


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("sample", "generate", "freeze", "report"))
    ap.add_argument("--version", default="v1")
    ap.add_argument("--base-url", default="http://127.0.0.1:8096")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--only", default="", help="comma-separated episode ids (generate)")
    ap.add_argument("--out-root", default=str(OUT_ROOT), help="where freeze writes <version>/")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="a retry pass over refused episodes draws new samples")
    args = ap.parse_args(argv)
    WORK.mkdir(parents=True, exist_ok=True)
    if args.stage == "sample":
        eps = sample()
        EPISODES.write_text("".join(json.dumps(e) + "\n" for e in eps))
        from collections import Counter
        print(f"sampled {len(eps)} episodes: {dict(Counter(e['category'] for e in eps))}")
        return 0
    episodes = _read_jsonl(EPISODES)
    if args.stage == "generate":
        import httpx
        model = httpx.get(f"{args.base_url}/v1/models", timeout=10).json()["data"][0]["id"]
        if args.only:
            keep = set(args.only.split(","))
            episodes = [e for e in episodes if e["id"] in keep]
        asyncio.run(generate(episodes, base_url=args.base_url, model=model,
                             concurrency=args.concurrency, seed_offset=args.seed_offset))
        return 0
    drafts: dict[str, dict] = {}
    for row in _read_jsonl(DRAFTS):
        if row.get("ok") or row["id"] not in drafts:
            drafts[row["id"]] = row
    if args.stage == "report":
        from collections import Counter
        ok = Counter(e["category"] for e in episodes if drafts.get(e["id"], {}).get("ok"))
        bad = Counter(e["category"] for e in episodes if not drafts.get(e["id"], {}).get("ok"))
        print("accepted", dict(ok), "sum", sum(ok.values()))
        print("refused/missing", dict(bad))
        return 0
    root = freeze(args.version, episodes=episodes, drafts=drafts, out_root=Path(args.out_root),
                  corpus_note="facts snapshot ~/lloyd-data/eval/1480/corpus (copied 2026-09-25 "
                              "from _pipeline/vault-derived/facts)")
    print(f"froze {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
