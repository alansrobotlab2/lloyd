#!/usr/bin/env python3
"""
Fact Extraction Pipeline - Next-Gen Memory System

Extracts atomic facts from documents using the local 2B LLM.
Integrates with periodic-memory-capture system.
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Shared entity normalization (case-insensitive alias resolution + self-register).
# The facts tree uses <VAULT_FACTS_ROOT>/entity-aliases.json as the source of truth;
# without normalization, writers accumulate duplicate dirs (gr00t/ + GR00T/).
_LLOYD_ROOT = Path(__file__).resolve().parents[3]
if str(_LLOYD_ROOT) not in sys.path:
    sys.path.insert(0, str(_LLOYD_ROOT))
from app.entity_naming import normalize_and_register as _entity_normalize
from app.entity_naming import known_entities_in_text as _known_entities
from app.entity_naming import looks_like_junk_entity as _is_junk_entity
from app.atomic_io import atomic_write_text, locked_file
from app.fact_ids import assign_ids as _assign_fact_ids
from app.kg_store import StoreUnavailable, store as _kg_store

# The seven fallback classes that used to sit here (a hand-rolled YAML parser,
# a pass-through entity normaliser, a never-junk predicate) each turned a
# missing dependency into silently wrong output: unparseable frontmatter read
# as an empty fact list, which `write_fact_file` then wrote back as truth.
# An ImportError is the correct outcome — the extractor cannot do its job
# without these.

import yaml

# Constants
HOME = Path.home()
VAULT = HOME / "obsidian"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from app.paths import VAULT_FACTS_ROOT as FACTS_DIR

# Chunked extraction. The old hard `content[:3000]` cap dropped ~40% of the median
# vault doc (70% of docs exceed 3000 chars) — including the structured Tools /
# GitHub / Papers sections at the bottom of YouTube notes. We now window the doc
# so most files (≤ CHUNK_SIZE) are still a single LLM call, while longer docs get
# full coverage across a bounded number of chunks.
CHUNK_SIZE = 8000          # chars per chunk (~2000 tokens); 90% of docs fit in one
CHUNK_OVERLAP = 200        # carry a little context across the cut so facts aren't split
MAX_CHUNKS = 6             # bound cost on huge docs (sessions can be 100KB+)

# The category vocabulary. 287 distinct category spellings existed on
# 2026-09-03 — `state`, `States`, `current state`, `state/config` and so on —
# because the model's free-text answer was written through verbatim. Each
# spelling makes its own fact file, so `fact_get(entity, category="state")`
# missed most of the entity's state facts. Unknown answers map to the nearest
# by token overlap, else `general`.
CATEGORY_VOCAB = (
    "state", "event", "decision", "preference", "goal", "skill",
    "relationship", "capability", "constraint", "configuration",
    "hardware", "research", "general",
)
_CATEGORY_TOKENS = {c: set(re.findall(r"[a-z]+", c)) for c in CATEGORY_VOCAB}
# Spellings seen in the tree that map cleanly onto the vocabulary.
_CATEGORY_ALIASES = {
    "status": "state", "current state": "state", "states": "state",
    "config": "configuration", "settings": "configuration",
    "preferences": "preference", "prefs": "preference",
    "relationships": "relationship", "relations": "relationship",
    "events": "event", "history": "event", "timeline": "event",
    "goals": "goal", "objectives": "goal", "skills": "skill",
    "capabilities": "capability", "constraints": "constraint",
    "decisions": "decision", "temporary": "state", "fact": "general",
    "facts": "general", "info": "general", "notes": "general",
}


class ExtractionFailed(RuntimeError):
    """The model call failed or returned nothing usable.

    Raised rather than returning an empty fact list. An empty list was
    indistinguishable from `this document genuinely has no facts`, so a
    transient vLLM error marked the document extracted and its content hash
    was saved — the document was then never revisited.
    """


def normalize_category(raw: str | None) -> str:
    """Map a model-supplied category onto CATEGORY_VOCAB."""
    c = (raw or "").strip().lower()
    c = re.sub(r"[^a-z ]+", " ", c).strip()
    if not c:
        return "general"
    if c in _CATEGORY_TOKENS:
        return c
    if c in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[c]
    toks = set(c.split())
    best, best_score = "general", 0.0
    for cat, cat_toks in _CATEGORY_TOKENS.items():
        overlap = len(toks & cat_toks)
        if not overlap:
            continue
        score = overlap / len(toks | cat_toks)
        if score > best_score:
            best, best_score = cat, score
    return best if best_score >= 0.3 else "general"

EXTRACTION_PROMPT = """You are a fact extraction engine. Analyze the following content and extract
atomic facts about entities mentioned.

Rules:
1. Each fact must be a single, unambiguous statement
2. Include temporal context if mentioned ("moved last week" → event_date)
3. A document usually covers SEVERAL distinct entities (people, projects, tools,
   papers, systems). Attribute EACH fact to the specific entity it is about via
   that fact's "entity" field — do NOT collapse everything onto one entity.
4. Categorize each fact: preference, relationship, event, state, skill, goal, temporary
5. Note confidence level (0.0-1.0)
6. If a fact contradicts a known fact, flag it as an update

CRITICAL GUARDRAILS:
- NEVER extract "session" as an entity. Session metadata (duration, message count,
  triviality, health checks, emptiness) is NOT a valid entity.
- NEVER create entities named "session", "session-distill", "session_<timestamp>",
  "session_pong5", or any session identifier. Sessions are the SOURCE of data,
  not entities to be extracted.
- If content is from a trivial/empty session (<5 messages, health check, routine
  maintenance), return an empty facts list. Do NOT fabricate observations like
  "the session was short" or "had no unresolved threads".
- Only extract facts about actual domain knowledge, tools, decisions, people,
  systems, and concepts discussed — never about the session container itself.

Content:
{content}

Known entities already in the knowledge graph that appear in this content.
When a fact is about one of these, put this EXACT name in its "entity" field —
do not add or drop words like "System", "Pipeline", "Agent", "SDK", "App", and
do not re-spell or re-case it. Only coin a new entity name when none of these
is the thing the fact is about:
{known_entities}

Known facts about relevant entities:
{existing_facts}

Extract facts as structured JSON. Set the top-level "entity" to the single most
central entity of the content; give EVERY fact its own "entity" naming the
specific thing that fact is about (it may differ from the top-level entity):
{{
  "entity": "primary_entity_name",
  "category": "category_name",
  "facts": [
    {{
      "entity": "entity this fact is about",
      "fact": "Fact statement",
      "confidence": 0.95,
      "event_date": null,
      "category": "state"
    }}
  ]
}}
"""

class FactExtractor:
    """Extracts facts from documents using LLM."""
    
    def __init__(self, model_port: int = 8096):
        self.model_port = model_port
        self.facts_dir = FACTS_DIR
    
    def _chunk_content(self, content: str) -> list:
        """Window long docs so most files stay a single LLM call but long docs get
        full coverage. Returns [content] unchanged for docs <= CHUNK_SIZE."""
        if len(content) <= CHUNK_SIZE:
            return [content]
        chunks = []
        start = 0
        step = CHUNK_SIZE - CHUNK_OVERLAP
        while start < len(content) and len(chunks) < MAX_CHUNKS:
            chunks.append(content[start:start + CHUNK_SIZE])
            start += step
        if start < len(content):
            covered = (MAX_CHUNKS - 1) * step + CHUNK_SIZE
            print(f"  ⚠️ doc is {len(content)} chars; capped at {MAX_CHUNKS} chunks "
                  f"(~{covered} chars covered)")
        return chunks

    def _parse_response(self, response: str) -> dict:
        """Parse one LLM response into {entity, category, facts}. Tolerant of code
        fences and list-vs-object shapes."""
        try:
            if response.startswith("```"):
                response = re.sub(r'^```\w*\n?', '', response)
                response = re.sub(r'\n?```$', '', response)
            result = json.loads(response)
        except json.JSONDecodeError:
            print(f"Failed to parse LLM response as JSON: {response[:200]}")
            return {"entity": None, "category": None, "facts": []}

        if isinstance(result, list):
            if not result:
                return {"entity": None, "category": None, "facts": []}
            base = result[0] if isinstance(result[0], dict) else {"entity": None, "category": None, "facts": []}
            for extra in result[1:]:
                if isinstance(extra, dict) and extra.get("facts"):
                    base.setdefault("facts", []).extend(extra["facts"])
            result = base
        elif not isinstance(result, dict):
            return {"entity": None, "category": None, "facts": []}
        return result

    def extract_from_document(self, doc_path: Path, content: str,
                              existing_facts: str = "") -> dict:
        """Extract facts from a document.

        Long docs are windowed (see CHUNK_SIZE) and extracted chunk-by-chunk;
        facts from all chunks are merged and de-duplicated by text. Each fact
        carries its own sanitized "entity" so a multi-entity doc fans out to
        per-entity fact files instead of collapsing onto one primary entity.
        Return shape stays {entity, category, facts} for backward compatibility.
        """
        existing = existing_facts[:1000] if existing_facts else "None"
        primary_entity = None
        primary_category = None
        all_facts = []
        seen_fact_text = set()

        for chunk in self._chunk_content(content):
            known = _known_entities(chunk, 60)
            known_block = "\n".join(f"- {k}" for k in known) if known else "(none recognised)"
            prompt = EXTRACTION_PROMPT.format(content=chunk, existing_facts=existing,
                                              known_entities=known_block)
            parsed = self._parse_response(self._call_llm(prompt))

            if primary_entity is None and parsed.get("entity"):
                primary_entity = self._sanitize_entity(parsed["entity"])
                # A category is a vocabulary term, not an entity. Running it
                # through _sanitize_entity registered every distinct spelling
                # as a canonical entity in the alias table.
                primary_category = normalize_category(parsed.get("category"))

            for f in parsed.get("facts", []):
                if not isinstance(f, dict):
                    continue
                text = (f.get("fact") or "").strip()
                if not text or text in seen_fact_text:
                    continue
                seen_fact_text.add(text)
                # Resolve each fact to its own entity (falls back to the doc primary
                # at write time when absent).
                if f.get("entity"):
                    f["entity"] = self._sanitize_entity(f["entity"])
                f["category"] = normalize_category(f.get("category"))
                all_facts.append(f)

        return {
            "entity": primary_entity or "general",
            "category": primary_category or "general",
            "facts": all_facts,
        }
    
    def _call_llm(self, prompt: str) -> str:
        """Call the local model. Raises ExtractionFailed rather than faking
        an empty extraction — see the class docstring."""
        import urllib.request

        url = f"http://localhost:{self.model_port}/v1/chat/completions"
        payload = {
            "model": "primary",
            "messages": [
                {"role": "system", "content": "You are a fact extraction engine."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 6000,
            "chat_template_kwargs": {"enable_thinking": False},
            # vLLM --scheduling-policy priority: chat sends 0, autonomy 1. This
            # ran at the default and competed with both.
            "priority": 2,
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode('utf-8'))
                text = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise ExtractionFailed(f"LLM call failed: {e}") from e
        if not text or not text.strip():
            raise ExtractionFailed("LLM returned empty content")
        return text

    def get_existing_facts(self, entity: str, category: str) -> str:
        """Load existing facts for an entity/category."""
        # Sanitize entity and category to prevent nested path creation
        entity = self._sanitize_entity(entity)
        category = normalize_category(category)
        if not entity:
            return ""
        fact_file = self.facts_dir / entity / f"{entity}-{category}.md"
        if not fact_file.exists():
            return ""
        
        content = fact_file.read_text()
        # Extract YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 2:
                return parts[1].strip()
        return ""
    
    def _sanitize_entity(self, entity: str, source_doc: str | None = None) -> str:
        """Sanitize an entity name and resolve it to its canonical form.

        Sanitization strips path characters; the junk predicate runs BEFORE
        registration, so a leaked filename or a pipeline run name never
        enters the alias table. (It used to be registered first and rejected
        at write time, which is why 921 run-named canonicals existed.)
        Returns "" for a rejected name; callers drop the fact.
        """
        if not entity:
            return ""
        # Take last path component if slashes present
        entity = entity.strip().split("/")[-1].split("\\")[-1]
        # Remove any remaining path-unsafe characters
        entity = re.sub(r'[<>:"|?*]', '', entity)
        # Collapse whitespace
        entity = re.sub(r'\s+', ' ', entity).strip()
        if not entity or _is_junk_entity(entity, source_doc):
            return ""
        # Alias-resolve + self-register. Safe on unknowns (pass-through).
        return _entity_normalize(entity)

    def write_fact_file(self, entity: str, category: str, facts_data: dict,
                        *, source_doc: str | None = None,
                        source_hash: str | None = None) -> Path | None:
        """Write or update a fact file.

        Every fact written here carries `created_at`, `source_doc`,
        `source_hash` and `provenance`, so it can be dated, attributed and
        selectively reverted. 99.7% of the 205k facts in the pre-2026-09 tree
        had none of those, which is why nothing could tell a fact extracted
        from a real vault note from one extracted out of the pipeline's own
        exhaust.

        Returns the written path, `None` when the entity is rejected as junk.
        """
        entity = self._sanitize_entity(entity, source_doc)
        category = normalize_category(category)

        if not entity:
            print(f"  ⤫ skipped junk entity ({len(facts_data.get('facts', []))} facts dropped)")
            return None

        entity_dir = self.facts_dir / entity
        entity_dir.mkdir(parents=True, exist_ok=True)
        fact_file = entity_dir / f"{entity}-{category}.md"

        now_iso = datetime.now(timezone.utc).isoformat()
        new_facts = facts_data.get("facts", [])
        for nf in new_facts:
            if not isinstance(nf, dict):
                continue
            nf.setdefault("provenance", "EXTRACTED")
            nf.setdefault("created_at", now_iso)
            if source_doc:
                nf.setdefault("source_doc", source_doc)
            if source_hash:
                nf.setdefault("source_hash", source_hash)
            # An event the fact itself dates is when it was true, not when we
            # read it. `created_at` stays the extraction time either way.
            if nf.get("event_date") and not nf.get("valid_at"):
                nf["valid_at"] = str(nf["event_date"])
            nf.setdefault("expired_at", None)
            nf.setdefault("invalid_at", None)

        # The lock covers the whole read-modify-write. Four extractor threads
        # and `fact_add` from a chat turn all target the same file; without it
        # the later writer silently drops the earlier one's facts.
        with locked_file(fact_file):
            existing_facts = self._read_existing_facts(fact_file)
            if existing_facts is None:
                return None            # quarantined; do not write over it

            merged_facts = self._merge_facts(existing_facts, new_facts)
            merged_facts = _assign_fact_ids(merged_facts, category)

            frontmatter = {
                "type": "facts",
                "entity": entity,
                "category": category,
                "facts": merged_facts,
                "last_extracted": now_iso,
                "last_updated": now_iso,
            }
            if source_doc:
                frontmatter["source_doc"] = source_doc

            yaml_content = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
            markdown_body = self._generate_markdown_body(entity, category, merged_facts)
            atomic_write_text(fact_file, f"---\n{yaml_content}---\n\n{markdown_body}")

        self._index_and_link(entity, category, fact_file, new_facts, source_doc)
        return fact_file

    def _read_existing_facts(self, fact_file: Path) -> list | None:
        """Existing facts, or None when the file is corrupt and was quarantined.

        A YAML error used to fall through to `existing_facts = []`, and the
        very next statement wrote the file back with only the new facts in it.
        One unparseable character therefore deleted an entity's whole history.
        The file is renamed instead, and this extraction is skipped.
        """
        if not fact_file.exists():
            return []
        try:
            content = fact_file.read_text(encoding="utf-8")
        except OSError as e:
            print(f"  ⤫ cannot read {fact_file.name}: {e}")
            return None
        if not content.strip():
            return []
        if not content.startswith("---"):
            return self._quarantine(fact_file, "no frontmatter fence")
        parts = content.split("---", 2)
        if len(parts) < 3:
            return self._quarantine(fact_file, "unterminated frontmatter")
        try:
            frontmatter = yaml.safe_load(parts[1])
        except Exception as e:
            return self._quarantine(fact_file, f"YAML error: {e}")
        if not isinstance(frontmatter, dict):
            return self._quarantine(fact_file, "frontmatter is not a mapping")
        facts = frontmatter.get("facts")
        if facts is None:
            return []
        if not isinstance(facts, list):
            return self._quarantine(fact_file, "`facts` is not a list")
        return [f for f in facts if isinstance(f, dict)]

    @staticmethod
    def _quarantine(fact_file: Path, why: str) -> None:
        """Rename a corrupt fact file aside and return None."""
        stamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        dest = fact_file.with_name(f"{fact_file.name}.corrupt-{stamp}")
        try:
            fact_file.rename(dest)
            print(f"  ⚠ quarantined {fact_file.name} ({why}) → {dest.name}")
        except OSError as e:
            print(f"  ⚠ {fact_file.name} is corrupt ({why}) and could not be moved: {e}")
        return None

    def _index_and_link(self, entity: str, category: str, fact_file: Path,
                        new_facts: list, source_doc: str | None) -> None:
        """Index the file and emit the edges its facts imply.

        This is the graph's growth path. Before it, edges only appeared when
        someone ran `seed_relationship_edges.py` by hand, which is why node
        coverage sat at 13.7% and the nightly chain added no edges at all.

        A `mentions` edge is emitted for every OTHER known entity a fact names
        — `relationship`-category facts included, which is where the densest
        signal is. The v4 classifier upgrades them to typed relations.
        """
        try:
            st = _kg_store()
        except StoreUnavailable as e:
            print(f"  ⚠ store unavailable, not indexing {fact_file.name}: {e}")
            return
        try:
            st.entities.register(entity)
            st.facts_idx.update_file(fact_file, root=self.facts_dir)
            with st.transaction():
                for f in new_facts:
                    if not isinstance(f, dict):
                        continue
                    text = (f.get("fact") or "").strip()
                    if not text:
                        continue
                    subject = f.get("entity") or entity
                    for target in _known_entities(text, 12):
                        if target == subject:
                            continue
                        try:
                            st.edges.add({
                                "source": subject, "target": target, "type": "mentions",
                                "confidence": float(f.get("confidence", 0.8) or 0.8),
                                "provenance": "EXTRACTED",
                                "source_doc": source_doc,
                                "evidence": text[:500],
                            }, origin="extractor")
                        except ValueError:
                            continue   # self-loop or blank endpoint
        except Exception as e:  # the markdown is written; the index is derived
            print(f"  ⚠ index/link failed for {fact_file.name}: {e}")

    def _merge_facts(self, existing: list, new: list) -> list:
        """Merge new facts with existing, avoiding duplicates."""
        # Simple dedup by fact text similarity
        existing_texts = {f.get("fact", "") for f in existing}
        merged = existing.copy()
        
        for fact in new:
            fact_text = fact.get("fact", "")
            # Simple duplicate check
            if fact_text not in existing_texts:
                merged.append(fact)
                existing_texts.add(fact_text)
        
        return merged
    
    def _assign_ids(self, facts: list, category: str) -> list:
        """Kept as a method for callers; the scheme lives in app.fact_ids."""
        return _assign_fact_ids(facts, category)

    def _generate_markdown_body(self, entity: str, category: str, 
                                facts: list) -> str:
        """Generate human-readable markdown body for fact file."""
        lines = [
            f"# {entity.title()} - {category.title()}",
            "",
            f"**Entity:** {entity}",
            f"**Category:** {category}",
            f"**Fact Count:** {len(facts)}",
            "",
            "## Facts",
            ""
        ]
        
        for fact in facts:
            fact_id = fact.get("id", "unknown")
            fact_text = fact.get("fact", "")
            confidence = fact.get("confidence", 0.0)
            status = fact.get("status", "current")
            
            lines.extend([
                f"### {fact_id}",
                "",
                f"**Fact:** {fact_text}",
                f"**Confidence:** {confidence}",
                f"**Status:** {status}",
                ""
            ])
        
        return "\n".join(lines)


if __name__ == "__main__":
    print("Fact Extraction Pipeline — imported by nightly_extraction.py.")
    print("There is no standalone entry point: run the nightly extraction, "
          "which owns the corpus selection, the content-hash gate and the "
          "failure accounting this module deliberately does not.")
