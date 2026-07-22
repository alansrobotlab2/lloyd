#!/usr/bin/env python3
"""
Semantic Relationship Discovery - Next-Gen Memory System

Discovers meaningful document relationships using content-based heuristics:
- Shared tags (from frontmatter)
- Entity mentions (extracted from content)
- Document type clustering
- Cross-references (wiki-links in content)

Avoids O(n²) path-based explosion; focuses on high-value relationships.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Tuple

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


VAULT = Path.home() / "obsidian"
MEMORY_DIR = VAULT / "memory"
RELATIONS_FILE = MEMORY_DIR / "semantic-relationships.md"


class SemanticRelationshipDiscoverer:
    """Discovers meaningful relationships between documents."""
    
    def __init__(self):
        self.vault = VAULT
        self.documents: List[Dict] = []
        self.relationships: List[Dict] = []
        self.stats = defaultdict(int)
        
        # Heuristics thresholds
        self.MIN_SHARED_TAGS = 2
        self.MIN_ENTITY_OVERLAP = 3
    
    def discover(self) -> List[Dict]:
        """Run full relationship discovery pipeline."""
        print("Phase 1: Scanning documents...")
        self._scan_documents()
        print(f"  Found {len(self.documents)} documents")
        
        print("\nPhase 2: Extracting relationships...")
        self._find_tag_based_relationships()
        self._find_entity_based_relationships()
        self._find_cross_reference_relationships()
        self._find_type_based_relationships()
        
        print(f"\nPhase 3: Found {len(self.relationships)} meaningful relationships")
        return self.relationships
    
    def _scan_documents(self):
        """Scan vault for documents and extract metadata."""
        scan_dirs = [
            self.vault / "memory",
            self.vault / "agents",
            self.vault / "projects",
            self.vault / "work",
            self.vault / "personal",
            self.vault / "knowledge"
        ]
        
        for scan_dir in scan_dirs:
            if not scan_dir.exists():
                continue
            for md_file in scan_dir.rglob("*.md"):
                try:
                    content = md_file.read_text(errors='ignore')
                    rel_path = str(md_file.relative_to(self.vault))
                    
                    # Extract frontmatter
                    frontmatter = {}
                    content_body = content
                    frontmatter_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
                    if frontmatter_match:
                        try:
                            frontmatter = yaml.safe_load(frontmatter_match.group(1)) or {}
                            content_body = content[frontmatter_match.end():]
                        except:
                            pass
                    
                    # Extract wiki-links from content
                    wiki_links = re.findall(r'\[\[([^\]]+)\]\]', content_body)
                    
                    # Extract entities from path (skip common dirs)
                    entities = self._extract_entities_from_path(rel_path)
                    
                    raw_tags = frontmatter.get("tags", [])
                    if isinstance(raw_tags, str):
                        raw_tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
                    elif not isinstance(raw_tags, list):
                        raw_tags = []
                    self.documents.append({
                        "path": rel_path,
                        "size": md_file.stat().st_size,
                        "type": frontmatter.get("type", "unknown"),
                        "tags": raw_tags,
                        "entities": entities,
                        "wiki_links": wiki_links,
                        "frontmatter": frontmatter
                    })
                except Exception as e:
                    pass
    
    def _extract_entities_from_path(self, path: str) -> Set[str]:
        """Extract entity names from file path."""
        entities = set()
        skip_dirs = {"memory", "agents", "projects", "work", "personal", "knowledge", 
                     "obsidian", "next-gen-memory", "scripts", "ai-models", "ai-frameworks"}
        
        for part in Path(path).parts:
            part_lower = part.lower()
            if part_lower not in skip_dirs and part_lower not in {"md", "mdx"}:
                # Normalize: remove dates, numbers, special chars
                entity = re.sub(r'^\d{4}-\d{2}-\d{2}-?', '', part_lower)
                entity = re.sub(r'[-_]', '', entity)
                if entity and len(entity) > 2:
                    entities.add(entity)
        
        return entities
    
    def _find_tag_based_relationships(self):
        """Find documents sharing multiple tags."""
        tag_docs: Dict[str, List[str]] = defaultdict(list)
        
        for doc in self.documents:
            for tag in doc.get("tags", []):
                tag_key = str(tag).strip() if not isinstance(tag, str) else tag.strip()
                if tag_key:
                    tag_docs[tag_key].append(doc["path"])
        
        # Find documents sharing 2+ tags
        path_tag_map: Dict[str, Set[str]] = {
            doc["path"]: set(
                str(t).strip() if not isinstance(t, str) else t.strip()
                for t in doc.get("tags", []) if str(t).strip()
            )
            for doc in self.documents
        }
        
        paths = list(path_tag_map.keys())
        for i, p1 in enumerate(paths):
            for p2 in paths[i+1:]:
                shared = path_tag_map[p1] & path_tag_map[p2]
                if len(shared) >= self.MIN_SHARED_TAGS:
                    self._add_relationship(
                        p1, p2, "related-to",
                        f"Shared tags: {', '.join(str(t) for t in sorted(shared, key=str)[:5])}",
                        score=len(shared) * 10
                    )
    
    def _find_entity_based_relationships(self):
        """Find documents mentioning same entities."""
        entity_docs: Dict[str, List[str]] = defaultdict(list)
        
        for doc in self.documents:
            for entity in doc.get("entities", set()):
                entity_docs[entity].append(doc["path"])
        
        # Find documents sharing 3+ entities
        path_entity_map: Dict[str, Set[str]] = {doc["path"]: doc.get("entities", set()) 
                                                 for doc in self.documents}
        
        paths = list(path_entity_map.keys())
        for i, p1 in enumerate(paths):
            for p2 in paths[i+1:]:
                shared = path_entity_map[p1] & path_entity_map[p2]
                if len(shared) >= self.MIN_ENTITY_OVERLAP:
                    self._add_relationship(
                        p1, p2, "related-to",
                        f"Shared entities: {', '.join(str(e) for e in sorted(shared, key=str)[:5])}",
                        score=len(shared) * 5
                    )
    
    def _find_cross_reference_relationships(self):
        """Find documents that link to each other."""
        link_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        
        for doc in self.documents:
            for link in doc.get("wiki_links", []):
                # Normalize link to potential path
                link_lower = link.lower().replace(" ", "-")
                for target in self.documents:
                    target_name = Path(target["path"]).stem.lower()
                    if link_lower == target_name or link_lower in target_name:
                        link_counts[(doc["path"], target["path"])] += 1
        
        for (source, target), count in link_counts.items():
            if count >= 1:
                self._add_relationship(
                    source, target, "references",
                    f"Linked {count} time(s)",
                    score=count * 20
                )
    
    def _find_type_based_relationships(self):
        """Group documents by type for potential relationships."""
        type_docs: Dict[str, List[str]] = defaultdict(list)
        
        for doc in self.documents:
            doc_type = doc.get("type", "unknown")
            if doc_type not in {"facts", "index", "unknown"}:
                type_docs[doc_type].append(doc["path"])
        
        # Only create relationships for types with few documents (focused groups)
        for doc_type, paths in type_docs.items():
            if 2 <= len(paths) <= 20:  # Small, focused groups
                for i, p1 in enumerate(paths):
                    for p2 in paths[i+1:]:
                        self._add_relationship(
                            p1, p2, "same-type",
                            f"Both are {doc_type}",
                            score=3
                        )
    
    def _add_relationship(self, source: str, target: str, rel_type: str, reason: str, score: int):
        """Add a relationship with scoring."""
        self.relationships.append({
            "source": source,
            "target": target,
            "type": rel_type,
            "reason": reason,
            "score": score,
            "status": "pending"
        })
        self.stats[rel_type] += 1
    
    def write_report(self):
        """Write relationship discovery report."""
        # Sort by score
        sorted_rels = sorted(self.relationships, key=lambda x: -x["score"])
        
        lines = [
            "# Semantic Relationship Discovery Report",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Documents Analyzed:** {len(self.documents)}",
            f"**Relationships Found:** {len(self.relationships)}",
            "",
            "## Statistics",
            "",
        ]
        
        for rel_type, count in sorted(self.stats.items(), key=lambda x: -x[1]):
            lines.append(f"- **{rel_type}:** {count}")
        
        lines.extend([
            "",
            "## Top Relationships (by score)",
            "",
        ])
        
        # Show top 20 overall
        for i, rel in enumerate(sorted_rels[:20], 1):
            lines.extend([
                f"### {i}. `{rel['source']}` → `{rel['target']}`",
                "",
                f"- **Type:** {rel['type']}",
                f"- **Score:** {rel['score']}",
                f"- **Reason:** {rel['reason']}",
                f"- **Status:** {rel['status']}",
                "",
            ])
        
        # Show sample of tag-based relationships
        tag_rels = [r for r in sorted_rels if r['type'] == 'related-to' and 'Shared tags' in r['reason']]
        if tag_rels:
            lines.extend([
                "## Sample Tag-Based Relationships",
                "",
            ])
            for i, rel in enumerate(tag_rels[:15], 1):
                lines.extend([
                    f"### {i}. `{rel['source']}` → `{rel['target']}`",
                    "",
                    f"- **Type:** {rel['type']}",
                    f"- **Score:** {rel['score']}",
                    f"- **Reason:** {rel['reason']}",
                    f"- **Status:** {rel['status']}",
                    "",
                ])
        
        # Show sample of entity-based relationships
        entity_rels = [r for r in sorted_rels if r['type'] == 'related-to' and 'Shared entities' in r['reason']]
        if entity_rels:
            lines.extend([
                "## Sample Entity-Based Relationships",
                "",
            ])
            for i, rel in enumerate(entity_rels[:10], 1):
                lines.extend([
                    f"### {i}. `{rel['source']}` → `{rel['target']}`",
                    "",
                    f"- **Type:** {rel['type']}",
                    f"- **Score:** {rel['score']}",
                    f"- **Reason:** {rel['reason']}",
                    f"- **Status:** {rel['status']}",
                    "",
                ])
        
        lines.append(f"**Total Relationships:** {len(sorted_rels)} (full list available on request)")
        
        RELATIONS_FILE.write_text("\n".join(lines))
        print(f"\nReport written to {RELATIONS_FILE}")
    
    def get_summary(self) -> Dict:
        """Get discovery summary."""
        return {
            "documents_analyzed": len(self.documents),
            "relationships_found": len(self.relationships),
            "by_type": dict(self.stats),
            "avg_score": sum(r["score"] for r in self.relationships) / len(self.relationships) if self.relationships else 0
        }


def main():
    print("Semantic Relationship Discovery - Next-Gen Memory")
    print("=" * 50)
    
    discoverer = SemanticRelationshipDiscoverer()
    discoverer.discover()
    discoverer.write_report()
    
    summary = discoverer.get_summary()
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
