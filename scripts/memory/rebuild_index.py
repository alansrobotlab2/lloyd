#!/usr/bin/env python3
"""
Rebuild Index Script - Next-Gen Memory System

Rebuilds both relations-index.json and facts-index.json to ensure consistency.
This is Task #15 from the autonomy idler agent.
"""

import json
import sys
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

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

HOME = Path.home()
VAULT = HOME / "obsidian"
MEMORY_DIR = VAULT / "memory"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.paths import VAULT_FACTS_ROOT as FACTS_DIR
RELATIONS_INDEX = Path(__file__).resolve().parent.parent.parent / "_pipeline" / "relations-index.json"
FACTS_INDEX = Path(__file__).resolve().parent.parent.parent / "_pipeline" / "facts-index.json"


def rebuild_relations_index() -> dict:
    """Rebuild the relations index using semantic relationships."""
    print("Rebuilding relations-index.json...")
    
    # Scan documents
    documents = []
    scan_dirs = [
        VAULT / "memory",
        VAULT / "agents",
        VAULT / "projects",
        VAULT / "work",
        VAULT / "personal",
        VAULT / "knowledge"
    ]
    
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for md_file in scan_dir.rglob("*.md"):
            if not md_file.is_file():
                continue
            try:
                content = md_file.read_text(errors='ignore')
                rel_path = str(md_file.relative_to(VAULT))
                
                # Extract frontmatter
                frontmatter = {}
                frontmatter_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
                if frontmatter_match:
                    try:
                        frontmatter = yaml.safe_load(frontmatter_match.group(1)) or {}
                    except:
                        pass
                
                # Extract wiki-links
                wiki_links = re.findall(r'\[\[([^\]]+)\]\]', content)
                
                documents.append({
                    "path": rel_path,
                    "type": frontmatter.get("type", "unknown"),
                    "tags": frontmatter.get("tags", []),
                    "wiki_links": wiki_links
                })
            except:
                pass
    
    # Build relationships
    relationships = []
    
    # 1. Wiki-link relationships
    link_map = defaultdict(list)
    for doc in documents:
        for link in doc.get("wiki_links", []):
            link_map[link.lower()].append(doc["path"])
    
    for link, sources in link_map.items():
        if len(sources) >= 2:
            for i, s1 in enumerate(sources):
                for s2 in sources[i+1:]:
                    relationships.append({
                        "source": s1,
                        "target": s2,
                        "type": "wiki-link",
                        "reason": f"Both link to [[{link}]]",
                        "score": 100
                    })
    
    # 2. Tag-based relationships
    tag_docs = defaultdict(list)
    for doc in documents:
        for tag in doc.get("tags", []):
            tag_key = str(tag).strip() if not isinstance(tag, str) else tag.strip()
            if tag_key:
                tag_docs[tag_key].append(doc["path"])
    
    tag_doc_sets = {tag: set(docs) for tag, docs in tag_docs.items()}
    all_tags = list(tag_docs.keys())
    
    for i, t1 in enumerate(all_tags):
        for t2 in all_tags[i+1:]:
            shared_docs = tag_doc_sets[t1] & tag_doc_sets[t2]
            if len(shared_docs) >= 2:
                docs_list = list(shared_docs)
                for j, d1 in enumerate(docs_list):
                    for d2 in docs_list[j+1:]:
                        relationships.append({
                            "source": d1,
                            "target": d2,
                            "type": "tag-cluster",
                            "reason": f"Share tags: {t1}, {t2}",
                            "score": 80
                        })
    
    # Write index
    index_data = {
        "relationships": relationships,
        "total_relationships": len(relationships),
        "documents_indexed": len(documents),
        "last_updated": datetime.now().isoformat()
    }
    
    RELATIONS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    RELATIONS_INDEX.write_text(json.dumps(index_data, indent=2))
    
    print(f"  → Relations index rebuilt: {len(relationships)} relationships from {len(documents)} documents")
    return {"total_relationships": len(relationships), "documents": len(documents), "status": "rebuilt"}


def rebuild_facts_index() -> dict:
    """Rebuild the facts index by scanning all fact files."""
    print("Rebuilding facts-index.json...")
    
    facts_dir = FACTS_DIR
    facts = []
    errors = []
    
    if not facts_dir.exists():
        print("  → Facts directory not found, creating empty index")
        facts = []
    else:
        # Scan all fact files (skip directories)
        for entity_dir in facts_dir.iterdir():
            if not entity_dir.is_dir():
                continue
            entity_name = entity_dir.name
            for fact_file in entity_dir.glob("*.md"):
                if not fact_file.is_file():
                    continue
                try:
                    content = fact_file.read_text()
                    if content.startswith("---"):
                        parts = content.split("---", 2)
                        if len(parts) >= 2:
                            frontmatter = yaml.safe_load(parts[1])
                            if frontmatter:
                                # Convert datetime objects to strings
                                last_updated = frontmatter.get("last_updated", "unknown")
                                if hasattr(last_updated, 'isoformat'):
                                    last_updated = last_updated.isoformat()
                                
                                facts.append({
                                    "path": str(fact_file.relative_to(VAULT)),
                                    "entity": frontmatter.get("entity", entity_name),
                                    "category": frontmatter.get("category", "unknown"),
                                    "fact_count": len(frontmatter.get("facts", [])),
                                    "last_updated": last_updated
                                })
                except Exception as e:
                    errors.append(f"{fact_file.name}: {str(e)[:50]}")
    
    # Report errors (limit to first 10)
    if errors:
        print(f"  → Skipped {len(errors)} files with parsing errors (first 10):")
        for err in errors[:10]:
            print(f"      {err}")
    
    # Write facts index
    facts_index = {
        "facts": facts,
        "total_facts": len(facts),
        "last_updated": datetime.now().isoformat(),
        "skipped_files": len(errors)
    }
    
    FACTS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    FACTS_INDEX.write_text(json.dumps(facts_index, indent=2))
    
    print(f"  → Facts index rebuilt: {len(facts)} fact files indexed, {len(errors)} skipped")
    return {"total_facts": len(facts), "skipped": len(errors), "status": "rebuilt"}


def main():
    print("=" * 60)
    print("Rebuild Index Script - Next-Gen Memory System")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Rebuild relations index
    relations_result = rebuild_relations_index()
    print()
    
    # Rebuild facts index
    facts_result = rebuild_facts_index()
    print()
    
    # Summary
    print("=" * 60)
    print("Rebuild Complete")
    print("=" * 60)
    print(f"Relations index: {relations_result}")
    print(f"Facts index: {facts_result}")
    print(f"Files updated:")
    print(f"  - {RELATIONS_INDEX}")
    print(f"  - {FACTS_INDEX}")


if __name__ == "__main__":
    main()
