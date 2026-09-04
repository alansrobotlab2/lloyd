"""Definition-aware merge gate for the entity-resolution sweep.

The sweep's suffix tier decided that `X` and `X System` / `X Agent` / `X SDK` /
`X Pipeline` are "basically always the same thing". They are not: on 2026-09-03
that rule merged Lloyd's news scanner into Intel Corporation, a fact-lookup
layer into a robotics action tokenizer, and a robot's training pipeline into
the robot. Name shape cannot tell those apart. Definitions can.

This gate shows both entities' definitions to local models and asks SAME or
DIFFERENT. Two judges, and a merge is only allowed when EVERY judge answers
SAME; anything else routes the pair to hand review with the verdicts attached.
Measured on the 78 suffix pairs from that day: each judge alone caught 14 of
17 verified-bad merges, with complementary misses; requiring agreement caught
16 and left ~33 pairs for review.

Verdicts are cached by (a, b, definition-hash) so re-runs are free.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.paths import VAULT_FACTS_ROOT  # noqa: E402

DEFAULT_CACHE = Path.home() / "lloyd" / "_pipeline" / "memory-graph" / "semantic-verdicts.jsonl"

SYSTEM_PROMPT = (
    "You decide whether two entity names from a personal knowledge graph refer to the SAME "
    "real-world thing or to DIFFERENT things. Use the definitions. A product and its SDK are "
    "different. A company and a project named after it are different. A robot and its training "
    "pipeline are different. A general concept and a specific system are different. A system and "
    "the same system with the word 'System' or 'Pipeline' appended are the same. "
    "Answer with exactly one line: SAME or DIFFERENT, then ' — ' and a reason under 15 words."
)


def default_judges() -> list[tuple[str, str]]:
    """(model_alias, chat-completions URL) pairs, honouring `secondary_enabled`."""
    env = os.environ.get("KG_JUDGES")
    if env:
        out = []
        for part in env.split(","):
            alias, url = part.split("=", 1)
            out.append((alias.strip(), url.strip()))
        return out
    judges = [("primary", "http://127.0.0.1:8096/v1/chat/completions")]
    try:
        from app.config import resolve_model_alias
        if resolve_model_alias("secondary") == "secondary":
            judges.append(("secondary", "http://127.0.0.1:8091/v1/chat/completions"))
    except Exception:
        pass
    return judges


def entity_definition(name: str, root: Path = VAULT_FACTS_ROOT) -> str:
    """The overview's `definition:` field, else the first prose line of its Summary."""
    p = root / name / f"{name}-overview.md"
    if not p.exists():
        return ""
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            m = re.search(r"^definition:\s*(.+?)(?=^\S|\Z)", parts[1], re.M | re.S)
            if m:
                d = " ".join(line.strip() for line in m.group(1).splitlines()).strip()
                if d and d.lower() not in ("null", "''", '""'):
                    return d[:500]
            body = parts[2]
        else:
            body = text
    else:
        body = text
    for line in body.splitlines():
        s = line.strip()
        if s and not s.startswith(("#", "-", "*", "|", "**")) and len(s) > 30:
            return s[:500]
    return ""


def _http_judge(model: str, url: str, timeout: float = 60.0) -> Callable[[str], str]:
    def ask(user: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user}],
            "max_tokens": 300, "temperature": 0.0,
            "priority": 2,                       # batch work yields to chat (0) and autonomy (1)
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        return (d["choices"][0]["message"].get("content") or "").strip()
    return ask


def parse_verdict(raw: str) -> str:
    u = raw.strip().upper()
    if u.startswith("SAME"):
        return "SAME"
    if u.startswith("DIFFERENT"):
        return "DIFFERENT"
    return "UNPARSED"


class SemanticGate:
    """Ask every judge; allow a merge only on unanimous SAME.

    `judge_fns` overrides the HTTP judges (tests pass fakes). A judge that
    errors counts as REVIEW — the gate fails closed.
    """

    def __init__(self, root: Path = VAULT_FACTS_ROOT,
                 cache_path: Optional[Path] = DEFAULT_CACHE,
                 judges: Optional[list[tuple[str, str]]] = None,
                 judge_fns: Optional[dict[str, Callable[[str], str]]] = None):
        self.root = root
        self.cache_path = cache_path
        if judge_fns is not None:
            self.judge_fns = judge_fns
        else:
            self.judge_fns = {alias: _http_judge(alias, url) for alias, url in (judges or default_judges())}
        self._cache: dict[str, dict] = {}
        if cache_path and cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec
                except Exception:
                    continue
        self.calls = 0

    @staticmethod
    def _key(a: str, b: str, da: str, db: str) -> str:
        h = hashlib.sha1((da + "\x1f" + db).encode("utf-8", "replace")).hexdigest()[:12]
        return f"{a}\x1f{b}\x1f{h}"

    def verdict(self, a: str, b: str) -> dict:
        """{"decision": "SAME"|"REVIEW", "judges": {alias: {"verdict","reason"}}, "cached": bool}"""
        da, db = entity_definition(a, self.root), entity_definition(b, self.root)
        # No definition, no verdict. Asked about a name with nothing on file, the
        # judges answer from the name's shape — the very signal this gate exists
        # to distrust. Seen 2026-09-03: 92 of 129 "SAME" verdicts were for pairs
        # where one side's directory had just been restored and had no overview
        # yet. Overviews regenerate on the next extraction pass; the pair will be
        # judged properly on the run after that. This check sits before the cache
        # on purpose so a cached shape-only verdict can never be reused.
        missing = [n for n, d in ((a, da), (b, db)) if not d]
        if missing:
            return {"decision": "REVIEW", "cached": False,
                    "judges": {"gate": {"verdict": "NO_DEFINITION",
                                        "reason": f"no definition on file for {', '.join(missing)}; "
                                                  f"judge after overviews regenerate"}},
                    "def_a": da[:160], "def_b": db[:160]}
        key = self._key(a, b, da, db)
        if key in self._cache:
            rec = dict(self._cache[key]); rec["cached"] = True
            return rec
        user = (f"Entity A: {a}\nDefinition A: {da or '(no definition on file)'}\n\n"
                f"Entity B: {b}\nDefinition B: {db or '(no definition on file)'}\n\nSAME or DIFFERENT?")
        judges: dict[str, dict] = {}
        for alias, fn in self.judge_fns.items():
            self.calls += 1
            try:
                raw = fn(user)
                v = parse_verdict(raw)
                reason = raw.split("—", 1)[-1].strip() if "—" in raw else raw
                judges[alias] = {"verdict": v, "reason": reason[:200]}
            except Exception as e:
                judges[alias] = {"verdict": "ERROR", "reason": f"{type(e).__name__}: {e}"[:200]}
        decision = "SAME" if judges and all(j["verdict"] == "SAME" for j in judges.values()) else "REVIEW"
        rec = {"key": key, "a": a, "b": b, "decision": decision, "judges": judges,
               "def_a": da[:160], "def_b": db[:160]}
        self._cache[key] = rec
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out = dict(rec); out["cached"] = False
        return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: entity_semantic_gate.py 'Entity A' 'Entity B'"); sys.exit(2)
    g = SemanticGate()
    print(json.dumps(g.verdict(sys.argv[1], sys.argv[2]), indent=2, ensure_ascii=False))
