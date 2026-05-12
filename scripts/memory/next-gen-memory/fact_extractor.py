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
except Exception:
    def _entity_normalize(name: str) -> str:  # fallback: pass-through
        return name

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
EXTRACTION_PROMPT = """You are a fact extraction engine. Analyze the following content and extract
atomic facts about entities mentioned.

Rules:
1. Each fact must be a single, unambiguous statement
2. Include temporal context if mentioned ("moved last week" → event_date)
3. Identify the entity each fact is about (person, project, system)
4. Categorize: preference, relationship, event, state, skill, goal, temporary
5. Note confidence level (0.0-1.0)
6. If a fact contradicts a known fact, flag it as an update

Content:
{content}

Known facts about relevant entities:
{existing_facts}

Extract facts as structured JSON with the following format:
{{
  "entity": "entity_name",
  "category": "category_name",
  "source": "document_path",
  "document_date": "YYYY-MM-DD",
  "facts": [
    {{
      "fact": "Fact statement",
      "confidence": 0.95,
      "event_date": null,
      "category": "preference"
    }}
  ]
}}
"""

class FactExtractor:
    """Extracts facts from documents using LLM."""
    
    def __init__(self, model_port: int = 8096):
        self.model_port = model_port
        self.facts_dir = FACTS_DIR
    
    def extract_from_document(self, doc_path: Path, content: str, 
                              existing_facts: str = "") -> dict:
        """Extract facts from a document."""
        # Build prompt
        prompt = EXTRACTION_PROMPT.format(
            content=content[:3000],  # Limit context
            existing_facts=existing_facts[:1000] if existing_facts else "None"
        )
        
        # Call local LLM (primary model on port 8096)
        response = self._call_llm(prompt)
        
        # Parse JSON response
        try:
            # Strip markdown code blocks if present
            if response.startswith("```"):
                # Remove ```json and ``` markers
                response = re.sub(r'^```\w*\n?', '', response)
                response = re.sub(r'\n?```$', '', response)
            result = json.loads(response)
            # Handle array responses
            if isinstance(result, list):
                if not result:
                    return {"entity": None, "category": None, "facts": []}
                # Take first element as base, merge facts from remaining
                base = result[0] if isinstance(result[0], dict) else {"entity": None, "category": None, "facts": []}
                for extra in result[1:]:
                    if isinstance(extra, dict) and extra.get("facts"):
                        base.setdefault("facts", []).extend(extra["facts"])
                result = base
            elif not isinstance(result, dict):
                return {"entity": None, "category": None, "facts": []}
            
            # Sanitize entity to prevent nested path creation
            if result.get("entity"):
                result["entity"] = self._sanitize_entity(result["entity"])
            if result.get("category"):
                result["category"] = self._sanitize_entity(result["category"])
            
            return result
        except json.JSONDecodeError:
            print(f"Failed to parse LLM response as JSON: {response[:200]}")
            return {"entity": None, "category": None, "facts": []}
    
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
            "max_tokens": 4000,
            "chat_template_kwargs": {"enable_thinking": False}
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
    
    def write_fact_file(self, entity: str, category: str, facts_data: dict) -> Path:
        """Write or update a fact file."""
        # Sanitize entity and category to prevent nested path creation
        entity = self._sanitize_entity(entity)
        category = self._sanitize_entity(category)
        
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
            print(f"  → Wrote {len(result['facts'])} facts to {fact_file.name}")


if __name__ == "__main__":
    print("Fact Extraction Pipeline - Next-Gen Memory")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--daily-notes":
        extract_from_daily_notes()
    else:
        print("Usage: fact_extractor.py [--daily-notes]")
        print("\nExtracts facts from documents using the 2B LLM.")
