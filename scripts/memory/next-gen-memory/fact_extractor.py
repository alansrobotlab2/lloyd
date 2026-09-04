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
from datetime import datetime
from pathlib import Path
from typing import Any

# Shared entity normalization (case-insensitive alias resolution + self-register).
# The facts tree uses <VAULT_FACTS_ROOT>/entity-aliases.json as the source of truth;
# without normalization, writers accumulate duplicate dirs (gr00t/ + GR00T/).
_LLOYD_ROOT = Path(__file__).resolve().parents[3]
if str(_LLOYD_ROOT) not in sys.path:
    sys.path.insert(0, str(_LLOYD_ROOT))
try:
    from app.entity_naming import normalize_and_register as _entity_normalize
    from app.entity_naming import known_entities_in_text as _known_entities
    from app.entity_naming import looks_like_junk_entity as _is_junk_entity
except Exception:
    def _known_entities(text: str, limit: int = 60) -> list:  # fallback: none known
        return []

    def _entity_normalize(name: str) -> str:  # fallback: pass-through
        return name

    def _is_junk_entity(name: str) -> bool:  # fallback: never junk
        return False

# Try to import yaml, provide fallback if not available
try:
    import yaml
except ImportError:
    # Simple YAML parser for frontmatter
    class yaml:
        @staticmethod
        def safe_load(text):
            """Minimal YAML parser for frontmatter."""
            result = {}
            current_key = None
            current_list = False
            for line in text.strip().split('\n'):
                line = line.rstrip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('- '):
                    if current_key and current_list:
                        item = line[2:].strip().strip('"').strip("'")
                        result[current_key].append(item)
                elif ':' in line:
                    key, _, value = line.partition(':')
                    key = key.strip()
                    value = value.strip()
                    if value == '':
                        result[key] = []
                        current_key = key
                        current_list = True
                    elif value.startswith('[') and value.endswith(']'):
                        # Inline list
                        items = value[1:-1].split(',')
                        result[key] = [item.strip().strip('"').strip("'") for item in items if item.strip()]
                        current_key = None
                        current_list = False
                    else:
                        result[key] = value.strip('"').strip("'")
                        current_key = None
                        current_list = False
            return result

        @staticmethod
        def dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True):
            """Simple YAML dumper for frontmatter."""
            lines = []
            for key, value in data.items():
                if isinstance(value, list):
                    lines.append(f"{key}:")
                    for item in value:
                        if isinstance(item, dict):
                            lines.append(f"- id: {item.get('id', '')}")
                            for k, v in item.items():
                                if k != 'id':
                                    lines.append(f"  {k}: {v}")
                        else:
                            lines.append(f"- {item}")
                elif isinstance(value, dict):
                    lines.append(f"{key}:")
                    for k, v in value.items():
                        lines.append(f"  {k}: {v}")
                else:
                    lines.append(f"{key}: {value}")
            return '\n'.join(lines) + '\n'

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
                primary_category = self._sanitize_entity(parsed.get("category") or "general")

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
                if f.get("category"):
                    f["category"] = self._sanitize_entity(f["category"])
                all_facts.append(f)

        return {
            "entity": primary_entity or "general",
            "category": primary_category or "general",
            "facts": all_facts,
        }
    
    def _call_llm(self, prompt: str) -> str:
        """Call the local 2B model."""
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

        response_text = None
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode('utf-8'))
                response_text = data["choices"][0]["message"]["content"]
                return response_text
        except Exception as e:
            print(f"LLM call failed: {e}")
            # Fallback: try to extract first complete JSON object from truncated response
            if response_text:
                match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text)
                if match:
                    try:
                        return match.group()
                    except Exception:
                        pass

            # Return empty extraction
            return json.dumps({
                "entity": "unknown",
                "category": "general",
                "facts": []
            })
    
    def get_existing_facts(self, entity: str, category: str) -> str:
        """Load existing facts for an entity/category."""
        # Sanitize entity and category to prevent nested path creation
        entity = self._sanitize_entity(entity)
        category = self._sanitize_entity(category)
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
    
    def _sanitize_entity(self, entity: str) -> str:
        """Sanitize entity name and resolve to canonical form.

        Sanitization strips path characters; after that we run it through
        the alias map so casing/punct variants collapse to the existing
        canonical (gr00t → GR00T). Unknown names are registered as
        self-identity so the next writer sees the same canonical.
        """
        if not entity:
            return "general"
        # Take last path component if slashes present
        entity = entity.strip().split("/")[-1].split("\\")[-1]
        # Remove any remaining path-unsafe characters
        entity = re.sub(r'[<>:"|?*]', '', entity)
        # Collapse whitespace
        entity = re.sub(r'\s+', ' ', entity).strip()
        if not entity:
            return "general"
        # Alias-resolve + self-register. Safe on unknowns (pass-through).
        return _entity_normalize(entity)
    
    def write_fact_file(self, entity: str, category: str, facts_data: dict) -> Path | None:
        """Write or update a fact file.

        Returns the written path, or ``None`` when the entity is rejected as
        junk (a leaked filename / code fragment — see
        ``app.entity_naming.looks_like_junk_entity``).
        """
        # Sanitize entity and category to prevent nested path creation
        entity = self._sanitize_entity(entity)
        category = self._sanitize_entity(category)

        # Skip leaked filenames / code fragments so they never become entity dirs.
        if _is_junk_entity(entity):
            print(f"  ⤫ skipped junk entity '{entity}' ({len(facts_data.get('facts', []))} facts dropped)")
            return None

        entity_dir = self.facts_dir / entity
        entity_dir.mkdir(parents=True, exist_ok=True)
        
        fact_file = entity_dir / f"{entity}-{category}.md"
        
        # Load existing facts if file exists
        existing_facts = []
        if fact_file.exists():
            content = fact_file.read_text()
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 2:
                    try:
                        frontmatter = yaml.safe_load(parts[1])
                        existing_facts = frontmatter.get("facts", [])
                    except:
                        pass
        
        # Merge new facts with existing — tag provenance on new LLM-extracted facts
        new_facts = facts_data.get("facts", [])
        for nf in new_facts:
            if "provenance" not in nf:
                nf["provenance"] = "EXTRACTED"
        merged_facts = self._merge_facts(existing_facts, new_facts)
        
        # Generate sequential IDs
        merged_facts = self._assign_ids(merged_facts, category)
        
        # Build frontmatter
        frontmatter = {
            "type": "facts",
            "entity": entity,
            "category": category,
            "facts": merged_facts,
            "last_extracted": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "relationships": []
        }
        
        # Write file
        yaml_content = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        markdown_body = self._generate_markdown_body(entity, category, merged_facts)
        
        full_content = f"---\n{yaml_content}---\n\n{markdown_body}"
        fact_file.write_text(full_content)
        
        return fact_file
    
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
        """Assign sequential IDs to facts."""
        # Get next ID based on existing facts
        prefix_map = {
            "preferences": "pref",
            "configuration": "conf", 
            "hardware": "hard",
            "status": "status",
            "relationships": "rel",
            "events": "event",
            "skills": "skill",
            "goals": "goal",
            "general": "fact"
        }
        prefix = prefix_map.get(category, "fact")
        
        # Sort by ID if exists
        facts_with_id = []
        next_id = 1
        for fact in facts:
            if "id" in fact:
                facts_with_id.append(fact)
            else:
                fact["id"] = f"{prefix}-{next_id:03d}"
                next_id += 1
                facts_with_id.append(fact)
        
        return facts_with_id
    
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


def extract_from_daily_notes():
    """Extract facts from recent daily notes."""
    extractor = FactExtractor()
    
    # Get recent daily notes
    memory_dir = VAULT / "memory"
    daily_notes = sorted(memory_dir.glob("2026-*.md"))[-7:]  # Last 7 days
    
    for note in daily_notes:
        print(f"Processing: {note.name}")
        content = note.read_text()
        
        # Extract date from filename
        date_match = re.match(r"(\d{4}-\d{2}-\d{2})", note.name)
        doc_date = date_match.group(1) if date_match else None
        
        # Extract facts
        result = extractor.extract_from_document(
            note, 
            content,
            existing_facts=""
        )
        
        if result.get("facts"):
            # Write to appropriate fact file
            entity = result.get("entity", "general")
            category = result.get("category", "events")
            
            fact_file = extractor.write_fact_file(
                entity, category, result
            )
            if fact_file:
                print(f"  → Wrote {len(result['facts'])} facts to {fact_file.name}")


if __name__ == "__main__":
    print("Fact Extraction Pipeline - Next-Gen Memory")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--daily-notes":
        extract_from_daily_notes()
    else:
        print("Usage: fact_extractor.py [--daily-notes]")
        print("\nExtracts facts from documents using the 2B LLM.")
