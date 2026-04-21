#!/usr/bin/env python3
"""
Entity Overview Generator — Next-Gen Memory System

Generates a `{entity}-overview.md` file for each entity in ~/obsidian/facts/,
containing a one-line definition (frontmatter) and a synthesized summary
(markdown body). Uses the local primary model and, for non-personal entities,
augments the prompt with DuckDuckGo search snippets.

Note: filename uses `-overview.md` (not `-profile.md`) because `profile` is
already used by the fact extractor as a category name.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

HOME = Path.home()
FACTS_DIR = HOME / "obsidian" / "facts"

# Entities whose facts fall into these categories are treated as personal
# and NOT sent to DuckDuckGo. `preference` is the narrow "this is a sentient
# entity" signal — tools, projects, and concepts don't have preferences.
PERSONAL_CATEGORIES = {"preference"}

# DDGS uses the `primp` Rust HTTP client which deadlocks when called from
# multiple Python threads (shared native state). We dodge this by running
# each search in a short-lived subprocess — each process has its own primp
# state, so N subprocesses run truly parallel.
_DDGS_SUBPROCESS_TIMEOUT = 20  # seconds per search
_DDGS_WORKER_SCRIPT = """
import json, sys
from ddgs import DDGS
query = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
try:
    results = list(DDGS().text(query, max_results=n))
    print(json.dumps({"ok": True, "results": results}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": str(exc)}))
"""

PROFILE_PROMPT = """You are writing an entity profile for a knowledge graph.

Given the atomic facts about "{entity}" below{web_note}, produce:
1. A single-sentence DEFINITION (<=25 words) describing what this entity IS.
2. A coherent SUMMARY paragraph (<=300 words) synthesizing what is known.

CRITICAL: Prefer in-vault facts over web snippets when they conflict.
Facts are authoritative; web snippets are supplementary context only.

Facts:
{facts}
{web_block}
Return strict JSON with exactly this shape, and nothing else:
{{"definition": "one sentence", "summary": "paragraph"}}
"""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_frontmatter(path: Path) -> tuple[dict, str]:
    """Return (frontmatter_dict, body_text). Both empty on parse failure."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return {}, ""
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except Exception:
        fm = {}
    return fm, parts[2].lstrip("\n")


class ProfileGenerator:
    def __init__(self, model_port: int = 8096, facts_dir: Path = FACTS_DIR):
        self.model_port = model_port
        self.facts_dir = facts_dir

    # ── Fact loading ────────────────────────────────────────────────────

    def _fact_files(self, entity_dir: Path) -> list[Path]:
        """All `{entity}-*.md` fact files except the overview itself."""
        return [
            p for p in sorted(entity_dir.glob(f"{entity_dir.name}-*.md"))
            if p.name != f"{entity_dir.name}-overview.md"
        ]

    def _load_facts(self, entity_dir: Path) -> list[dict]:
        """Flatten all facts across the entity's category files."""
        facts = []
        for f in self._fact_files(entity_dir):
            fm, _ = _parse_frontmatter(f)
            for item in (fm.get("facts") or []):
                if isinstance(item, dict):
                    facts.append({
                        "fact": item.get("fact", ""),
                        "category": item.get("category", fm.get("category", "")),
                        "confidence": item.get("confidence", 1.0),
                        "event_date": item.get("event_date"),
                    })
        return facts

    def compute_source_hash(self, entity_dir: Path) -> str:
        """SHA256 over sorted fact bodies. Changes iff facts change."""
        facts = self._load_facts(entity_dir)
        bodies = sorted(f["fact"] for f in facts if f.get("fact"))
        return _sha256("\n".join(bodies))

    def is_personal_entity(self, entity_dir: Path) -> bool:
        """True if the entity has a file whose top-level category is personal.

        Checks file-level category only (e.g. {entity}-preference.md). Per-fact
        categories are too noisy — the extractor often labels individual facts
        as 'preference' inside state/relationship files on non-person entities.
        """
        for f in self._fact_files(entity_dir):
            fm, _ = _parse_frontmatter(f)
            if fm.get("category") in PERSONAL_CATEGORIES:
                return True
        return False

    # ── Web augmentation ────────────────────────────────────────────────

    def web_augment(self, entity: str) -> tuple[str, list[str]]:
        """Query DuckDuckGo via a subprocess; return (concatenated snippets, source URLs).

        Silent failure: returns ("", []) on any error. We never want a web
        hiccup to break profile generation. Subprocess isolation lets many
        workers run DDGS in parallel without hitting primp's threading bug.
        """
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _DDGS_WORKER_SCRIPT, entity, "5"],
                capture_output=True, text=True,
                timeout=_DDGS_SUBPROCESS_TIMEOUT,
            )
            if proc.returncode != 0:
                print(f"  [web] subprocess rc={proc.returncode} for {entity!r}: {proc.stderr[:200]}", file=sys.stderr)
                return "", []
            payload = json.loads(proc.stdout)
            if not payload.get("ok"):
                print(f"  [web] search failed for {entity!r}: {payload.get('error')}", file=sys.stderr)
                return "", []
            results = payload["results"]
        except subprocess.TimeoutExpired:
            print(f"  [web] timeout for {entity!r}", file=sys.stderr)
            return "", []
        except Exception as exc:
            print(f"  [web] search failed for {entity!r}: {exc}", file=sys.stderr)
            return "", []
        if not results:
            return "", []
        lines = []
        urls = []
        for r in results:
            title = r.get("title", "").strip()
            snippet = r.get("body", "").strip()
            url = r.get("href", "").strip()
            if url:
                urls.append(url)
            if snippet:
                lines.append(f"- [{title}] {snippet}")
        return "\n".join(lines), urls

    # ── LLM ─────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """Call the local LLM. Uses alias 'primary' so we survive model upgrades."""
        url = f"http://localhost:{self.model_port}/v1/chat/completions"
        payload = {
            "model": "primary",
            "messages": [
                {"role": "system", "content": "You write concise, accurate entity profiles. Always return strict JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1500,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

    def _parse_llm_json(self, text: str) -> dict:
        """Strip code fences and parse. Returns {} on failure."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: find first {...} block.
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return {}
            try:
                obj = json.loads(match.group())
            except json.JSONDecodeError:
                return {}
        if not isinstance(obj, dict):
            return {}
        return obj

    # ── Regeneration gate ───────────────────────────────────────────────

    def profile_path(self, entity: str) -> Path:
        return self.facts_dir / entity / f"{entity}-overview.md"

    def needs_regeneration(self, entity: str) -> bool:
        entity_dir = self.facts_dir / entity
        if not entity_dir.is_dir():
            return False
        if not self._fact_files(entity_dir):
            # No facts — nothing to summarize.
            return False
        prof = self.profile_path(entity)
        if not prof.exists():
            return True
        fm, _ = _parse_frontmatter(prof)
        if fm.get("manually_edited"):
            return False
        return fm.get("source_hash") != self.compute_source_hash(entity_dir)

    # ── Generation ──────────────────────────────────────────────────────

    def generate(self, entity: str) -> dict | None:
        """Generate and write the profile. Returns the profile dict or None on failure."""
        entity_dir = self.facts_dir / entity
        if not entity_dir.is_dir():
            print(f"  [skip] {entity}: directory missing", file=sys.stderr)
            return None

        facts = self._load_facts(entity_dir)
        if not facts:
            print(f"  [skip] {entity}: no facts", file=sys.stderr)
            return None

        personal = self.is_personal_entity(entity_dir)
        if personal:
            web_text, web_urls = "", []
        else:
            web_text, web_urls = self.web_augment(entity)

        # Build prompt
        facts_str = "\n".join(
            f"- ({f.get('category','?')}) {f.get('fact','')}"
            for f in facts if f.get("fact")
        )
        if web_text:
            web_note = " and the web search snippets"
            web_block = f"\nWeb search snippets:\n{web_text}\n"
        else:
            web_note = ""
            web_block = ""

        prompt = PROFILE_PROMPT.format(
            entity=entity,
            facts=facts_str,
            web_note=web_note,
            web_block=web_block,
        )

        try:
            raw = self._call_llm(prompt)
        except Exception as exc:
            print(f"  [fail] {entity}: LLM call failed: {exc}", file=sys.stderr)
            return None

        parsed = self._parse_llm_json(raw)
        definition = (parsed.get("definition") or "").strip()
        summary = (parsed.get("summary") or "").strip()
        if not definition or not summary:
            print(f"  [fail] {entity}: empty definition/summary", file=sys.stderr)
            return None

        source_hash = self.compute_source_hash(entity_dir)
        profile = {
            "type": "overview",
            "entity": entity,
            "category": "overview",
            "definition": definition,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_fact_count": len(facts),
            "source_hash": source_hash,
            "manually_edited": False,
            "web_searched": bool(web_urls) and not personal,
            "web_sources": web_urls,
        }
        self._write_profile(entity, profile, summary)
        return profile

    def _write_profile(self, entity: str, profile: dict, summary: str) -> Path:
        entity_dir = self.facts_dir / entity
        path = self.profile_path(entity)
        yaml_text = yaml.dump(profile, default_flow_style=False, sort_keys=False, allow_unicode=True)
        body = f"# Summary\n\n{summary.strip()}\n"
        path.write_text(f"---\n{yaml_text}---\n\n{body}", encoding="utf-8")
        return path

    # ── Batch driver ────────────────────────────────────────────────────

    def _process_one(self, entity: str, force: bool, print_lock: threading.Lock) -> bool:
        """Generate one entity. Returns True if a profile was written."""
        try:
            if not force and not self.needs_regeneration(entity):
                return False
            result = self.generate(entity)
            if result:
                with print_lock:
                    print(f"  ✓ {entity}")
                return True
        except Exception as exc:
            with print_lock:
                print(f"  ✗ {entity}: {exc}", file=sys.stderr)
        return False

    def regenerate_all(self, force: bool = False, workers: int = 1) -> int:
        """Walk FACTS_DIR and regenerate profiles that need it.

        Parallel-safe: each entity writes to its own `{entity}-overview.md`
        under `{entity}/`, so workers never touch the same file. The local
        vLLM server handles continuous batching, so higher `workers` trades
        GPU utilization for wall-clock time.
        """
        if not self.facts_dir.is_dir():
            return 0

        entities = [
            d.name for d in sorted(self.facts_dir.iterdir())
            if d.is_dir() and d.name != "templates"
        ]
        if not entities:
            return 0

        print_lock = threading.Lock()

        if workers <= 1:
            return sum(self._process_one(e, force, print_lock) for e in entities)

        count = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._process_one, e, force, print_lock): e for e in entities}
            for fut in as_completed(futures):
                if fut.result():
                    count += 1
        return count


# ── CLI ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate entity profiles")
    parser.add_argument("--entity", help="Generate profile for a single entity (bypasses needs_regeneration check)")
    parser.add_argument("--all", action="store_true", help="Generate profiles for all changed entities")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if source hash unchanged")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default 1). Try 8 for full GPU utilization.")
    parser.add_argument("--port", type=int, default=8096, help="LLM port (default 8096)")
    args = parser.parse_args()

    gen = ProfileGenerator(model_port=args.port)

    if args.entity:
        print(f"Generating profile for {args.entity!r}...")
        result = gen.generate(args.entity)
        if result:
            print(json.dumps(result, indent=2))
            return 0
        return 1

    if args.all:
        print(f"Regenerating entity overviews (workers={args.workers})...")
        t0 = datetime.now()
        n = gen.regenerate_all(force=args.force, workers=args.workers)
        elapsed = (datetime.now() - t0).total_seconds()
        print(f"Generated {n} overviews in {elapsed:.1f}s")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
