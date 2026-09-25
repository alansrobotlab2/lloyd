#!/usr/bin/env python3
"""scripts/skill_pseudo_queries.py — Skill2Query pseudo-queries per skill (#1490).

For every live skill, ask the primary for a handful of short requests a user
might type when that skill's procedure is the job, and cache them keyed by a
hash of the skill's own text. `agent_mcp.skills` indexes the cached queries as
a fifth token set when `skills.pseudo_queries.enabled` is on.

**Leakage rule.** The prompt carries the skill's name, description, tags and
body — nothing else. No eval query, no session transcript, no example turn is
ever shown to the generator, so the skill-match eval stays a held-out set.

The cache is a JSON file:

    {"prompt_version": 1, "model": "...",
     "skills": {"<name>": {"hash": "<sha1 of name/desc/tags/body>",
                           "queries": ["...", ...]}}}

A skill whose hash moved is regenerated; one that is gone is dropped. Nothing
here runs at import time and nothing in a turn calls the engine.

    # needs the primary; hold its lock (shared: this is not a timing run)
    flock -s -w 7200 ~/.local/state/lloyd-automod/primary.lock \\
      .venvs/lloyd/bin/python scripts/skill_pseudo_queries.py --out PATH
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROMPT_VERSION = 1
N_QUERIES = 8
BODY_CHARS = 6000

PROMPT = """You are helping index a library of assistant skills so the right one is found for a user's message.

Below is one skill: its name, description, tags and instructions. Write {n} different short messages a user might send to a personal AI assistant in chat when running THIS skill's procedure is exactly the right thing to do.

Rules:
- Write them the way a person types in chat: casual, first person, often terse, sometimes a question.
- Vary the wording. Use the everyday words a user would use for the task, not only the skill's own jargon. At most two may contain the skill's name.
- Each must be a request whose job is this skill, not a question ABOUT the skill's subject.
- One message per line, no numbering, no quotes, no commentary.

Skill name: {name}
Description: {description}
Tags: {tags}

Instructions:
{body}
"""


def skill_hash(skill: dict) -> str:
    """What the queries were generated from; a change regenerates them."""
    from agent_mcp.skills import skill_text_hash
    return skill_text_hash(skill)


def build_prompt(skill: dict, n: int = N_QUERIES) -> str:
    tags = skill.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    return PROMPT.format(n=n, name=skill.get("name") or "",
                         description=str(skill.get("description") or "").strip(),
                         tags=", ".join(map(str, tags)),
                         body=(skill.get("body") or "")[:BODY_CHARS])


def parse_queries(text: str, n: int = N_QUERIES) -> list[str]:
    """One query per non-empty line, list markers and quotes stripped."""
    out: list[str] = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"').strip("'").strip()
        if len(line) < 4 or line.endswith(":"):
            continue
        if line not in out:
            out.append(line)
    return out[:n]


def _generate(skill: dict, base_url: str, model: str, timeout: float,
              seed: int = 1490) -> list[str]:
    import httpx
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(skill)}],
        "max_tokens": 600,
        "temperature": 0.7,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
        "priority": 5,
    }
    r = httpx.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    return parse_queries(msg.get("content") or "")


def load_cache(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {"prompt_version": PROMPT_VERSION, "skills": {}}
    if data.get("prompt_version") != PROMPT_VERSION:
        return {"prompt_version": PROMPT_VERSION, "skills": {}}
    data.setdefault("skills", {})
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", default="", help="cache file (default: app.paths.SKILL_PSEUDO_QUERIES_PATH)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8096")
    ap.add_argument("--model", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N stale skills")
    ap.add_argument("--seed", type=int, default=1490, help="sampling seed (temperature 0.7)")
    args = ap.parse_args(argv)

    from agent_mcp.skills import _iter_skills
    if args.out:
        out = Path(args.out)
    else:
        from app.paths import SKILL_PSEUDO_QUERIES_PATH
        out = SKILL_PSEUDO_QUERIES_PATH
    model = args.model
    if not model:
        import httpx
        model = httpx.get(f"{args.base_url}/v1/models", timeout=10).json()["data"][0]["id"]

    cache = load_cache(out)
    skills = list(_iter_skills())
    live = {s["name"] for s in skills}
    for gone in set(cache["skills"]) - live:
        del cache["skills"][gone]
    stale = [s for s in skills
             if (cache["skills"].get(s["name"]) or {}).get("hash") != skill_hash(s)]
    if args.limit:
        stale = stale[:args.limit]
    print(f"{len(skills)} skills, {len(stale)} to generate -> {out}")

    t0 = time.time()
    failed = 0

    def one(skill):
        try:
            return skill, _generate(skill, args.base_url, model, args.timeout, args.seed), None
        except Exception as exc:  # noqa: BLE001
            return skill, None, exc

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for skill, queries, err in ex.map(one, stale):
            if err is not None or not queries:
                failed += 1
                print(f"  FAILED {skill['name']}: {err or 'no queries parsed'}", file=sys.stderr)
                continue
            cache["skills"][skill["name"]] = {"hash": skill_hash(skill), "queries": queries}

    cache["model"] = model
    cache["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    tmp.replace(out)
    print(f"done in {time.time() - t0:.0f}s, {failed} failed, "
          f"{len(cache['skills'])}/{len(skills)} cached")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
