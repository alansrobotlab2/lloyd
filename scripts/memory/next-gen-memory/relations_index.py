#!/usr/bin/env python3
"""
Relations Index Generator - Next-Gen Memory System

Generates and maintains typed relationships between documents.
Parses relations frontmatter blocks and builds a queryable index.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
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


# Relation type inverses (bidirectional mapping)
INVERSE_RELATIONS = {
    "implements": "designed-by",
    "designed-by": "implements",
    "supersedes": "superseded-by",
    "superseded-by": "supersedes",
    "depends-on": "required-by",
    "required-by": "depends-on",
    "derived-from": "produces",
    "produces": "derived-from",
    "related-to": "related-to",  # Symmetric
    "conflicts-with": "conflicts-with",  # Symmetric
}

VALID_RELATION_TYPES = set(INVERSE_RELATIONS.keys())


class RelationsIndexGenerator:
    """Generate and maintain document relationships index."""
    
    def __init__(self):
        self.vault = Path.home() / "obsidian"
        self.index_file = Path.home() / "obsidian" / "memory" / "_pipeline" / "relations-index.json"
        self.index_data: Dict[str, Any] = {
            "edges": [],
            "stale": [],
            "built_at": None
        }
        self._doc_cache: Dict[str, dict] = {}  # Cache for parsed documents
    
    def rebuild(self) -> dict:
        """Rebuild the relationships index from scratch.
        
        Scans all vault markdown files for relations: frontmatter blocks,
        parses typed relations, and compiles into relations-index.json.
        
        Returns:
            Index summary with relationship counts
        """
        print("  → Scanning vault for documents with relations...")
        
        # Initialize fresh index
        self.index_data = {
            "edges": [],
            "stale": [],
            "built_at": datetime.now().isoformat()
        }
        
        edge_set = set()  # Deduplicate edges
        stale_set = set()
        
        # Scan all markdown files in vault
        md_files = list(self.vault.rglob("*.md"))
        print(f"  → Found {len(md_files)} markdown files")
        
        for md_file in md_files:
            try:
                # Validate path is within vault
                rel_path = str(md_file.relative_to(self.vault))
                
                # Skip system/excluded directories
                path_str = str(md_file)
                if any(skip in path_str for skip in [
                    "/.git/", "/node_modules/", "/.venv/", "/.cache/",
                    "/__pycache__/", ".pyc"
                ]):
                    continue
                
                content = md_file.read_text(errors='ignore')
                relations = self._parse_relations(content)
                
                if relations:
                    # Process each relation type
                    for rel_type, targets in relations.items():
                        if rel_type not in VALID_RELATION_TYPES:
                            continue
                        
                        for target in targets:
                            # Normalize target path
                            target_path = self._normalize_path(target, rel_path)
                            if not target_path:
                                continue
                            
                            # Create edge (source -> target with type)
                            edge = {
                                "source": rel_path,
                                "target": target_path,
                                "type": rel_type,
                                "origin": "manual"
                            }
                            
                            edge_key = (rel_path, target_path, rel_type)
                            if edge_key not in edge_set:
                                edge_set.add(edge_key)
                                self.index_data["edges"].append(edge)
                            
                            # Track stale documents (superseded-by)
                            if rel_type == "superseded-by":
                                stale_set.add(rel_path)
                            
                            # Add inverse relation
                            inverse_type = INVERSE_RELATIONS.get(rel_type)
                            if inverse_type and inverse_type != rel_type:
                                inverse_edge = {
                                    "source": target_path,
                                    "target": rel_path,
                                    "type": inverse_type,
                                    "origin": "inverse"
                                }
                                inverse_key = (target_path, rel_path, inverse_type)
                                if inverse_key not in edge_set:
                                    edge_set.add(inverse_key)
                                    self.index_data["edges"].append(inverse_edge)
                
            except Exception as e:
                print(f"  ⚠️ Error processing {md_file}: {e}")
                continue
        
        self.index_data["stale"] = sorted(list(stale_set))
        
        # Write index file
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        self.index_file.write_text(json.dumps(self.index_data, indent=2))
        
        total_edges = len(self.index_data["edges"])
        print(f"  → Built index with {total_edges} edges ({len(stale_set)} stale docs)")
        
        return {
            "total_relationships": total_edges,
            "stale_documents": len(stale_set),
            "status": "rebuilt",
            "built_at": self.index_data["built_at"]
        }
    
    def _parse_relations(self, content: str) -> Optional[Dict[str, List[str]]]:
        """Parse relations frontmatter block from document content.
        
        Handles both:
        - relations: {type: [targets]} (typed relations)
        - related: [targets] (legacy format)
        
        Args:
            content: Full document content
            
        Returns:
            Dictionary of relation_type -> list of targets, or None
        """
        # Extract frontmatter
        frontmatter_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
        if not frontmatter_match:
            return None
        
        try:
            frontmatter = yaml.safe_load(frontmatter_match.group(1)) or {}
        except:
            return None
        
        relations = {}
        
        # Parse typed relations block
        if "relations" in frontmatter and isinstance(frontmatter["relations"], dict):
            for rel_type, targets in frontmatter["relations"].items():
                if rel_type in VALID_RELATION_TYPES:
                    if isinstance(targets, str):
                        targets = [targets]
                    elif not isinstance(targets, list):
                        continue
                    relations[rel_type] = [t for t in targets if isinstance(t, str)]
        
        # Parse legacy related field (convert to related-to)
        if "related" in frontmatter and frontmatter["related"]:
            related = frontmatter["related"]
            if isinstance(related, str):
                related = [related]
            elif not isinstance(related, list):
                related = []
            relations["related-to"] = [r for r in related if isinstance(r, str)]
        
        return relations if relations else None
    
    def _normalize_path(self, target: str, source_path: str) -> Optional[str]:
        """Normalize target path to be relative to vault.
        
        Handles:
        - Relative paths (already vault-relative)
        - Paths with ./ prefix
        - Wiki-link style [[target]]
        
        Args:
            target: Target path from relations
            source_path: Source document path
            
        Returns:
            Normalized vault-relative path or None if invalid
        """
        # Remove wiki-link brackets if present
        target = re.sub(r'^\[\[|\]\]$', '', target)
        
        # Remove ./ prefix
        target = target.lstrip('./')
        
        # Ensure it's a valid path
        if not target or target.startswith('/'):
            return None
        
        # Validate it's a markdown file (add .md if missing)
        if not target.endswith('.md'):
            target = target + '.md'
        
        return target
    
    def add_relation(self, source: str, target: str, rel_type: str) -> bool:
        """Add a relationship between two documents.
        
        Adds to in-memory index, writes to disk, and optionally
        updates frontmatter of both documents (bidirectional).
        
        Args:
            source: Source document path (vault-relative)
            target: Target document path (vault-relative)
            rel_type: Relationship type (must be valid relation type)
            
        Returns:
            True if relation was added, False if duplicate or invalid
        """
        # Validate relation type
        if rel_type not in VALID_RELATION_TYPES:
            print(f"  ⚠️ Invalid relation type: {rel_type}")
            return False
        
        # Normalize paths
        source = self._normalize_path(source, "")
        target = self._normalize_path(target, "")
        
        if not source or not target:
            return False
        
        # Check for duplicate
        edge_key = (source, target, rel_type)
        existing_edges = {(e["source"], e["target"], e["type"]) for e in self.index_data["edges"]}
        
        if edge_key in existing_edges:
            return False  # Duplicate
        
        # Add to index
        edge = {
            "source": source,
            "target": target,
            "type": rel_type,
            "origin": "manual"
        }
        self.index_data["edges"].append(edge)
        
        # Add inverse relation (unless symmetric)
        inverse_type = INVERSE_RELATIONS.get(rel_type)
        if inverse_type and inverse_type != rel_type:
            inverse_edge = {
                "source": target,
                "target": source,
                "type": inverse_type,
                "origin": "inverse"
            }
            self.index_data["edges"].append(inverse_edge)
        
        # Write updated index
        self.index_file.write_text(json.dumps(self.index_data, indent=2))
        
        # Optionally update frontmatter (bidirectional)
        self._update_frontmatter(source, target, rel_type, add=True)
        if inverse_type and inverse_type != rel_type:
            self._update_frontmatter(target, source, inverse_type, add=True)
        
        return True
    
    def _update_frontmatter(self, doc_path: str, target: str, rel_type: str, add: bool = True):
        """Update document frontmatter with relation.
        
        Args:
            doc_path: Document to update
            target: Target document path
            rel_type: Relation type
            add: True to add, False to remove
        """
        doc_file = self.vault / doc_path
        if not doc_file.exists():
            return
        
        try:
            content = doc_file.read_text()
            
            # Check if has frontmatter
            if not content.startswith("---"):
                return
            
            # Parse frontmatter
            parts = content.split("---", 2)
            if len(parts) < 3:
                return
            
            frontmatter_text = parts[1]
            try:
                frontmatter = yaml.safe_load(frontmatter_text) or {}
            except:
                return
            
            # Ensure relations block exists
            if "relations" not in frontmatter or not isinstance(frontmatter["relations"], dict):
                frontmatter["relations"] = {}
            
            relations = frontmatter["relations"]
            
            if add:
                # Add relation
                if rel_type not in relations:
                    relations[rel_type] = []
                elif not isinstance(relations[rel_type], list):
                    relations[rel_type] = [relations[rel_type]]
                
                if target not in relations[rel_type]:
                    relations[rel_type].append(target)
            else:
                # Remove relation
                if rel_type in relations and isinstance(relations[rel_type], list):
                    if target in relations[rel_type]:
                        relations[rel_type].remove(target)
                    if not relations[rel_type]:
                        del relations[rel_type]
            
            # Write back
            new_content = f"---\n{yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)}---\n{parts[2]}"
            doc_file.write_text(new_content)
            
        except Exception as e:
            print(f"  ⚠️ Error updating frontmatter for {doc_path}: {e}")
    
    def scan_documents(self) -> list:
        """Scan documents for potential relationships.
        
        Returns:
            List of document metadata with parsed frontmatter
        """
        documents = []
        
        # Scan common vault directories
        scan_dirs = [
            self.vault / "memory",
            self.vault / "agents",
            self.vault / "projects",
            self.vault / "procedures",
            self.vault / "work",
            self.vault / "personal",
            self.vault / "knowledge",
            self.vault
        ]
        
        for scan_dir in scan_dirs:
            if not scan_dir.exists():
                continue
            for md_file in scan_dir.rglob("*.md"):
                try:
                    # Skip excluded paths
                    path_str = str(md_file)
                    if any(skip in path_str for skip in [
                        "/.git/", "/node_modules/", "/.venv/", "/.cache/",
                        "/__pycache__/", "/facts/", "/_pipeline/"
                    ]):
                        continue
                    
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
                    
                    # Extract tags
                    tags = frontmatter.get("tags", [])
                    if isinstance(tags, str):
                        tags = [tags]
                    
                    # Extract entity mentions from path
                    entities = self._extract_entities_from_path(rel_path)
                    
                    documents.append({
                        "path": rel_path,
                        "size": md_file.stat().st_size,
                        "frontmatter": frontmatter,
                        "type": frontmatter.get("type", "unknown"),
                        "segment": frontmatter.get("segment", "unknown"),
                        "tags": tags,
                        "wiki_links": wiki_links,
                        "entities": entities,
                        "relations": frontmatter.get("relations", {})
                    })
                except Exception as e:
                    pass  # Skip unreadable files
        
        return documents
    
    def _extract_entities_from_path(self, path: str) -> Set[str]:
        """Extract entity names from file path."""
        entities = set()
        skip_dirs = {"memory", "agents", "projects", "work", "personal", "knowledge", 
                     "obsidian", "next-gen-memory", "scripts", "ai-models", "ai-frameworks",
                     "lloyd", "architecture", "next-gen-memory-subsystem"}
        
        for part in Path(path).parts:
            part_lower = part.lower()
            if part_lower not in skip_dirs and part_lower not in {"md", "mdx"}:
                # Normalize: remove dates, numbers, special chars
                entity = re.sub(r'^\d{4}-\d{2}-\d{2}-?', '', part_lower)
                entity = re.sub(r'[-_]', '', entity)
                if entity and len(entity) > 2:
                    entities.add(entity)
        
        return entities
    
    def _calculate_similarity(self, doc1: dict, doc2: dict) -> float:
        """Calculate document similarity using proven heuristics.
        
        Based on semantic_relationships.py heuristics:
        - Wiki-link co-occurrence (strongest signal, score 100-180)
        - Tag overlap (2+ shared tags, score 80-90)
        - Entity co-occurrence (shared entity mentions, score 15)
        
        Args:
            doc1: First document metadata
            doc2: Second document metadata
            
        Returns:
            Normalized similarity score (0.0-1.0)
        """
        score = 0.0
        
        # 1. Wiki-link co-occurrence (strongest signal)
        links1 = set(doc1.get("wiki_links", []))
        links2 = set(doc2.get("wiki_links", []))
        shared_links = links1 & links2
        
        if shared_links:
            # Score: 100-180 based on number of shared links
            link_score = min(100 + len(shared_links) * 20, 180)
            score += link_score
        
        # 2. Tag overlap (2+ shared tags required)
        tags1 = set(doc1.get("tags", []))
        tags2 = set(doc2.get("tags", []))
        shared_tags = tags1 & tags2
        
        if len(shared_tags) >= 2:
            # Score: 80-90 based on number of shared tags
            tag_score = min(80 + len(shared_tags) * 5, 90)
            score += tag_score
        
        # 3. Entity co-occurrence (shared entity mentions)
        entities1 = doc1.get("entities", set())
        entities2 = doc2.get("entities", set())
        shared_entities = entities1 & entities2
        
        if shared_entities:
            # Score: 15 per shared entity
            entity_score = len(shared_entities) * 15
            score += entity_score
        
        # Normalize to 0.0-1.0
        # Max theoretical score ~250 (180 + 90 + ~80 for entities)
        normalized = min(score / 250.0, 1.0)
        
        return normalized
    
    def get_relations_for_doc(self, doc_path: str) -> List[dict]:
        """Get all relations for a specific document.
        
        Args:
            doc_path: Document path (vault-relative)
            
        Returns:
            List of relation dicts with source, target, type
        """
        # Reload index if needed
        if not self.index_data["edges"]:
            self.rebuild()
        
        return [
            edge for edge in self.index_data["edges"]
            if edge["source"] == doc_path or edge["target"] == doc_path
        ]
    
    def load_index(self) -> dict:
        """Load index from disk.
        
        Returns:
            Index data dictionary
        """
        if self.index_file.exists():
            try:
                self.index_data = json.loads(self.index_file.read_text())
            except:
                pass
        return self.index_data


def main():
    """Main entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Relations Index Generator")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from scratch")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--query", type=str, help="Query relations for document")
    
    args = parser.parse_args()
    
    generator = RelationsIndexGenerator()
    
    if args.test:
        print("Running tests...")
        _run_tests()
    elif args.rebuild:
        print("Rebuilding relations index...")
        result = generator.rebuild()
        print(f"\nResult: {json.dumps(result, indent=2)}")
    elif args.query:
        generator.load_index()
        relations = generator.get_relations_for_doc(args.query)
        print(f"\nRelations for {args.query}:")
        print(json.dumps(relations, indent=2))
    else:
        # Default: just load and show summary
        generator.load_index()
        print(f"Index loaded: {len(generator.index_data.get('edges', []))} edges")
        print(f"Stale docs: {len(generator.index_data.get('stale', []))}")


def _run_tests():
    """Run basic tests to validate implementation."""
    print("\n=== Relations Index Tests ===\n")
    
    generator = RelationsIndexGenerator()
    
    # Test 1: Path normalization
    print("Test 1: Path normalization...")
    assert generator._normalize_path("test.md", "") == "test.md"
    assert generator._normalize_path("./test.md", "") == "test.md"
    assert generator._normalize_path("[[test]]", "") == "test.md"
    print("  ✓ Path normalization works")
    
    # Test 2: Parse relations
    print("\nTest 2: Relations parsing...")
    test_content = """---
relations:
  depends-on:
    - target1.md
    - target2.md
  related-to:
    - target3.md
related:
  - legacy1.md
---
Content here
"""
    relations = generator._parse_relations(test_content)
    assert relations is not None
    assert "depends-on" in relations
    assert "related-to" in relations
    assert "legacy1.md" in relations["related-to"]
    print("  ✓ Relations parsing works")
    
    # Test 3: Similarity calculation
    print("\nTest 3: Similarity calculation...")
    doc1 = {
        "path": "test1.md",
        "wiki_links": ["common", "shared"],
        "tags": ["tag1", "tag2", "tag3"],
        "entities": {"entity1", "entity2"}
    }
    doc2 = {
        "path": "test2.md",
        "wiki_links": ["common", "other"],
        "tags": ["tag2", "tag4"],
        "entities": {"entity2", "entity3"}
    }
    similarity = generator._calculate_similarity(doc1, doc2)
    assert 0.0 <= similarity <= 1.0
    print(f"  ✓ Similarity calculation works (score: {similarity:.3f})")
    
    # Test 4: Rebuild index
    print("\nTest 4: Index rebuild...")
    result = generator.rebuild()
    assert "total_relationships" in result
    assert result["status"] == "rebuilt"
    print(f"  ✓ Index rebuilt with {result['total_relationships']} relationships")
    
    # Test 5: Load index
    print("\nTest 5: Index load...")
    generator2 = RelationsIndexGenerator()
    loaded = generator2.load_index()
    assert "edges" in loaded
    print(f"  ✓ Index loaded with {len(loaded['edges'])} edges")
    
    print("\n=== All tests passed ===\n")


if __name__ == "__main__":
    main()
