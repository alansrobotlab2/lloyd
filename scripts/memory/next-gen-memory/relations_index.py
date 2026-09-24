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


def _proposal_type_to_index_spelling(type_: str) -> str:
    """Fold a proposal's relation type into this index's own spelling (#1161).

    Two boundaries spell the same relations two ways, on purpose. The edge store
    normalizes to `related_to`; this index — because the vault frontmatter it
    mirrors does — is hyphenated, and `related-to:` is the key a person
    hand-writes in a note. #1161 made the conversation linker emit the store's
    spelling, so without this fold the merge read proposal types against
    `VALID_RELATION_TYPES` exactly and every new proposal silently skipped
    itself: zero merged, no error, nothing in the index to show for the run.

    Normalizing at each boundary's own edge is what keeps the index bytes and
    the frontmatter keys unchanged.
    """
    return (type_ or "").strip().lower().replace("_", "-")


# ── One file, one owner, one schema (backlog #1148) ─────────────────────────
# Two scripts used to write the *same* path with two different schemas: this
# module's :meth:`RelationsIndexGenerator.rebuild` wrote
# ``{edges, stale, built_at}`` while
# ``scripts/memory/rebuild_index.py::rebuild_relations_index()`` wrote
# ``{relationships, total_relationships, documents_indexed, last_updated}``.
# Both run inside scheduled task #24 (Data Pipeline, 6x-daily) in a fixed
# order — this module from ``nightly_extraction.py`` Step 1, ``rebuild_index.py``
# as Step 2 — so the file was not raced, it was deterministically overwritten by
# the last step of every cycle. This module's own readers then found neither
# shape they were written for: the no-arg CLI summary printed ``Index loaded: 0
# edges`` against a ~100 MB file, and ``--query`` raised ``KeyError: 'edges'``.
#
# Ownership, one writer each:
#   ``relations-index.json``        — written ONLY by ``scripts/memory/rebuild_index.py``.
#       ``relationships`` rows: wiki-link / tag-cluster co-occurrence.
#       Read-only from this module.
#   ``relations-index-typed.json``  — written ONLY by this module.
#       ``edges`` rows: the typed frontmatter relations, plus merged
#       conversation proposals.
#
# Which script owns which path is pinned by
# ``tests/test_relations_index_read_only.py::test_exactly_one_writer_per_index``
# (a checkout-wide scan), and what this module may write is pinned statically in
# the same file — so pointing a write back at the derived path fails a test
# instead of quietly re-opening the clobber. Consolidating the two schemas into
# one index is a retrieval design call and is deliberately NOT made here.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.paths import PIPELINE_DIR  # noqa: E402
DERIVED_INDEX_FILE = PIPELINE_DIR / "relations-index.json"
TYPED_INDEX_FILE = PIPELINE_DIR / "relations-index-typed.json"


def index_rows(index_data: Any) -> List[dict]:
    """Relation rows out of either schema, without a KeyError on either.

    ``edges`` is this module's typed shape; ``relationships`` is the shape
    ``rebuild_index.py`` leaves at the derived path. Before #1148 the readers
    here picked one key each — ``.get('edges', [])`` in the summary,
    ``index_data["edges"]`` in the query — so the same file read as a silently
    empty graph from one method and raised from the other, depending on which
    got there first.
    """
    if not isinstance(index_data, dict):
        return []
    rows = index_data.get("edges") or index_data.get("relationships") or []
    return rows if isinstance(rows, list) else []


class RelationsIndexGenerator:
    """Generate and maintain document relationships index.

    READ-ONLY with respect to the vault: every path here reads vault ``.md``
    files and the only thing it writes is ``self.typed_index_file``. Relations are
    authored in vault frontmatter by hand (or, for conversation-derived ones,
    land as ``relations:`` frontmatter through the normal vault route) and are
    *read* from here by :meth:`rebuild`; nothing in this module mutates a vault
    source document.

    That boundary is load-bearing, not stylistic. This class used to carry a
    second pair of methods — an "add this relation" entry point and the private
    frontmatter rewriter behind it. The rewriter rebuilt a note's whole
    frontmatter with ``yaml.dump`` and rejoined the body with
    ``f"---\\n{dump}---\\n{body}"``, a round-trip that reorders keys, drops
    quoting and appends one newline byte per call: it changed bytes without
    changing content. The vault's content-hash gate
    (``scripts/memory/content_hasher.py``) selects files on sha256 of bytes, so
    every such write re-triggered extraction of a note whose content had not
    changed. ``memory/vault-maintenance/vault-maintenance-2026-09-03.md`` is
    where that pattern got its name — ":167 byte-churn family confirmed",
    counting ``memory/2026-02-22.md`` from 5 reprocesses that day (:167) to its
    "10th reprocess today" (:223), with the run-to-run count also logged at :211.
    ``vault-maintenance-2026-08-30.md`` had already watched the same note come
    back as changed on consecutive runs (:63, :84, :105, :124) without naming it;
    there is no 08-31 log. No caller survived
    commit f36c522, which deleted the last production use, so both methods went
    the way of commit 0b3f00b ("delete what nothing runs") rather than staying
    as an unguarded writer aimed at the gate for the next caller to arm. The
    history, the clauses and the test are backlog #484,
    ``tests/test_relations_index_read_only.py``. If relation-writing comes
    back, it has to splice the ``relations:`` key through as text and skip the
    write when the result compares equal to what was parsed — and that test
    belongs in that file.
    """

    def __init__(self):
        self.vault = Path.home() / "obsidian"
        # Read-only: the derived index owned by scripts/memory/rebuild_index.py
        # (#1148). Kept under its original attribute name because it is the file
        # this module's readers consult; the module's own write target below is
        # what `rebuild()` may put bytes on.
        self.index_file = DERIVED_INDEX_FILE
        self.typed_index_file = TYPED_INDEX_FILE
        self.proposals_file = PIPELINE_DIR / "conversation-relation-proposals.json"
        self.index_data: Dict[str, Any] = {
            "edges": [],
            "stale": [],
            "built_at": None
        }
        # Rows read from `self.index_file`, whatever schema that file carries.
        self.derived_rows: List[dict] = []
        self._doc_cache: Dict[str, dict] = {}  # Cache for parsed documents
    
    def rebuild(self) -> dict:
        """Rebuild this module's typed index and write it to ``typed_index_file``.

        Scans all vault markdown files for relations: frontmatter blocks,
        parses typed relations, and compiles them — plus any approved
        conversation proposals — into ``_pipeline/relations-index-typed.json``.

        That file is the only thing this method writes (#1148). It used to write
        ``_pipeline/relations-index.json``, which
        ``scripts/memory/rebuild_index.py::rebuild_relations_index()`` also
        writes with a different schema, both inside scheduled task #24; the
        second writer won every cycle and this module's readers then could not
        read the file they had just been handed. The derived file is now read
        only (:meth:`load_index`), which is also what keeps merged conversation
        proposals from being dead on arrival: they are rows in a file nothing
        else rebuilds.

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

        # Merge approved conversation-derived proposals
        merged = self._merge_approved_proposals(edge_set)

        # Write THIS module's index file. `self.index_file` — the derived
        # relations-index.json owned by scripts/memory/rebuild_index.py — is
        # never a write target here (#1148); `self.typed_index_file` is.
        self.typed_index_file.parent.mkdir(parents=True, exist_ok=True)
        self.typed_index_file.write_text(json.dumps(self.index_data, indent=2))

        total_edges = len(self.index_data["edges"])
        print(f"  → Built index with {total_edges} edges ({len(stale_set)} stale docs, "
              f"{merged} from conversation proposals) → {self.typed_index_file}")

        return {
            "total_relationships": total_edges,
            "stale_documents": len(stale_set),
            "conversation_proposals_merged": merged,
            "status": "rebuilt",
            "built_at": self.index_data["built_at"]
        }
    
    def _merge_approved_proposals(self, edge_set: set) -> int:
        """Merge approved conversation-derived proposals into the index.

        Reads conversation-relation-proposals.json and adds any proposal with
        status "approved" as edges with origin "conversation".  Deduplicates
        against edge_set (modified in-place so subsequent calls stay clean).

        Returns:
            Number of new edges added from proposals.
        """
        if not self.proposals_file.exists():
            return 0

        try:
            data = json.loads(self.proposals_file.read_text())
        except Exception as e:
            print(f"  ⚠️ Could not read proposals file: {e}")
            return 0

        proposals = data.get("proposals", [])
        added = 0

        for p in proposals:
            if p.get("status") != "approved":
                continue

            source = self._normalize_path(p.get("source", ""), "")
            target = self._normalize_path(p.get("target", ""), "")
            # The linker that writes these proposals now speaks the edge store's
            # canonical `related_to`; this index keeps the hyphenated convention
            # of the frontmatter it mirrors, so fold at this boundary (#1161).
            rel_type = _proposal_type_to_index_spelling(str(p.get("type", "")))

            if not source or not target or rel_type not in VALID_RELATION_TYPES:
                continue

            edge_key = (source, target, rel_type)
            if edge_key in edge_set:
                continue

            edge_set.add(edge_key)
            self.index_data["edges"].append({
                "source": source,
                "target": target,
                "type": rel_type,
                "origin": "conversation",
                "reason": p.get("reason", ""),
                "confidence": p.get("confidence", 0.0),
            })
            added += 1

            # Add inverse unless symmetric
            inverse_type = INVERSE_RELATIONS.get(rel_type)
            if inverse_type and inverse_type != rel_type:
                inv_key = (target, source, inverse_type)
                if inv_key not in edge_set:
                    edge_set.add(inv_key)
                    self.index_data["edges"].append({
                        "source": target,
                        "target": source,
                        "type": inverse_type,
                        "origin": "conversation-inverse",
                        "reason": p.get("reason", ""),
                        "confidence": p.get("confidence", 0.0),
                    })

        if added:
            print(f"  → Merged {added} approved conversation proposals")

        return added

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
        
        Heuristics inherited from the retired semantic_relationships.py:
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
    
    def all_rows(self) -> List[dict]:
        """Every relation row this module can see: typed edges + derived rows."""
        return list(index_rows(self.index_data)) + list(self.derived_rows)

    def get_relations_for_doc(self, doc_path: str) -> List[dict]:
        """Get all relations for a specific document.

        Args:
            doc_path: Document path (vault-relative)

        Returns:
            List of relation dicts with source, target, type. Rows from both
            index files, whichever schema each one carries.

        This is a pure read (#1148). It used to open with
        ``if not self.index_data["edges"]: self.rebuild()``, so querying from a
        generator that had not called :meth:`load_index` — which is exactly what
        ``--test`` builds — scanned the live vault and rewrote the ~100 MB
        production index: a read path that could put bytes on production state.
        An empty index is now simply an empty answer.
        """
        return [
            row for row in self.all_rows()
            if doc_path in (row.get("source"), row.get("target"))
        ]

    @staticmethod
    def _read_json_dict(path: Path) -> dict:
        """Parse a JSON object, or ``{}`` when the file is missing or unreadable."""
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ⚠️ Could not read {path}: {exc}")
            return {}
        return data if isinstance(data, dict) else {}

    def load_index(self) -> dict:
        """Load both index files into the read views. Writes nothing (#1148).

        ``self.index_data``    <- this module's typed index (``edges`` shape).
        ``self.derived_rows``  <- rows from the derived index owned by
        ``scripts/memory/rebuild_index.py``, whose rows live under
        ``relationships``; :func:`index_rows` reads either key, so a file in
        either shape loads with its real count instead of reading as empty.

        Returns:
            The typed index data dictionary
        """
        self.index_data = {
            "edges": [],
            "stale": [],
            "built_at": None,
            **self._read_json_dict(self.typed_index_file),
        }
        self.derived_rows = index_rows(self._read_json_dict(self.index_file))
        print(f"Loaded {len(index_rows(self.index_data))} typed edges "
              f"({self.typed_index_file.name}), {len(self.derived_rows)} derived rows "
              f"({self.index_file.name})")
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
        # Default: just load and show summary. Both counts are printed under
        # the name of the file they came from (#1148): reading only `edges` out
        # of one merged number is what made a 331,977-row index print
        # `Index loaded: 0 edges`.
        generator.load_index()
        typed = index_rows(generator.index_data)
        print(f"Typed edges ({generator.typed_index_file}): {len(typed)}")
        print(f"Derived rows ({generator.index_file}): {len(generator.derived_rows)}")
        print(f"Stale docs: {len(generator.index_data.get('stale', []))}")


def _sandbox_generator(root: Path) -> RelationsIndexGenerator:
    """A generator whose every file lives under ``root``: scratch vault, derived
    index, typed index and proposals sidecar.

    #1148: the self-test used to build a bare ``RelationsIndexGenerator()``,
    whose default paths are the real vault and ``_pipeline/relations-index.json``,
    so ``python relations_index.py --test`` was a developer command that scanned
    the live vault and overwrote the production index. Every path it can reach
    now goes through here, and
    ``tests/test_relations_index_read_only.py::test_test_flag_leaves_the_live_index_alone``
    runs ``--test`` under a scratch HOME and requires the ``_pipeline`` files
    there not to move.
    """
    generator = RelationsIndexGenerator()
    generator.vault = root / "vault"
    generator.index_file = root / "pipeline" / DERIVED_INDEX_FILE.name
    generator.typed_index_file = root / "pipeline" / TYPED_INDEX_FILE.name
    generator.proposals_file = (
        root / "pipeline" / "conversation-relation-proposals.json"
    )
    return generator


def _run_tests():
    """Run basic tests to validate implementation.

    Everything with a filesystem side effect happens in a temp dir under a
    ``relations-index-selftest-`` prefix, never against ``~/obsidian`` or
    ``_pipeline`` (#1148). ``tempfile`` is imported here rather than at module
    top because ``tests/test_yaml_fix_skill_claims.py`` pins the line the
    fallback ``class yaml`` sits at, and the skill prose cites it by number.
    """
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="relations-index-selftest-"))
    print(f"\n=== Relations Index Tests ===\nSandbox: {root}\n")

    generator = _sandbox_generator(root)

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
    
    # Test 4: Rebuild index — the write-boundary check, so the sandbox vault is
    # deliberately EMPTY: this command's job here is to prove rebuild() puts
    # bytes on exactly one file and it is the typed one. Creating fixture notes
    # would make the self-test a writer of vault notes, which #484 forbids this
    # module from being at all. Edge production from frontmatter is pinned by
    # tests/test_relations_index_read_only.py::test_rebuild_edge_count_is_the_document_relations.
    print("\nTest 4: Index rebuild (empty sandbox vault)...")
    generator.vault.mkdir(parents=True, exist_ok=True)
    result = generator.rebuild()
    assert "total_relationships" in result
    assert result["status"] == "rebuilt"
    assert result["total_relationships"] == 0, (
        f"rebuild over an empty sandbox vault produced "
        f"{result['total_relationships']} edges; the sandbox is not as empty as "
        "the self-test assumes"
    )
    assert generator.typed_index_file.exists(), "rebuild() did not write the typed index"
    assert not generator.index_file.exists(), (
        f"the self-test wrote {generator.index_file.name}, the derived index owned "
        "by scripts/memory/rebuild_index.py (#1148)"
    )
    print(f"  ✓ Index rebuilt into {generator.typed_index_file.name}, "
          f"{generator.index_file.name} untouched")

    # Test 5: Load index — a fresh sandbox generator reading what Test 4 wrote.
    print("\nTest 5: Index load...")
    reader = _sandbox_generator(root)
    loaded = reader.load_index()
    assert "edges" in loaded, "load_index() returned a payload with no edges key"
    assert len(index_rows(loaded)) == result["total_relationships"], (
        "the index that was written and the index that was read back disagree"
    )
    assert reader.get_relations_for_doc("knowledge/a.md") == []
    print(f"  ✓ Index round-tripped: {len(index_rows(loaded))} edges written, "
          f"{len(index_rows(loaded))} read back")

    # Test 6: rows from either schema are readable, and a read puts nothing on
    # disk. The `relationships` shape is what scripts/memory/rebuild_index.py
    # leaves at the derived path; before #1148 the query path raised
    # KeyError: 'edges' against exactly that file. The shape is exercised in
    # memory here so the self-test writes no file it does not own — the
    # file-level version of the same read is pinned by
    # tests/test_relations_index_read_only.py.
    print("\nTest 6: Schema-agnostic read, and a read that writes nothing...")
    derived_rows = index_rows({
        "relationships": [{"source": "a.md", "target": "b.md", "type": "wiki-link"}],
        "total_relationships": 1,
    })
    assert len(derived_rows) == 1, "a relationships-shaped index yielded no rows"
    assert not index_rows({"edges": [], "stale": [], "built_at": None})
    assert index_rows("not a dict") == []

    empty = _sandbox_generator(root / "empty")
    assert empty.get_relations_for_doc("knowledge/a.md") == [], (
        "a generator that loaded nothing should answer with nothing, not rebuild"
    )
    assert not empty.typed_index_file.exists(), (
        "get_relations_for_doc rebuilt and wrote an index: a read must not put "
        "bytes on production state (#1148)"
    )
    print(f"  ✓ Read {len(derived_rows)} derived row(s); an empty read wrote nothing")

    print("\n=== All tests passed ===\n")


if __name__ == "__main__":
    main()
