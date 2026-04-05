#!/usr/bin/env python3
"""
Relationship Proposal Generator - Next-Gen Memory System

Generates relationship proposals for all documents based on content analysis.
"""

import json
import sys
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

VAULT = Path.home() / "obsidian"
MEMORY_DIR = VAULT / "memory"
PROPOSALS_FILE = MEMORY_DIR / "relationship-proposals.md"

sys.path.insert(0, str(VAULT / "agents" / "memory" / "scripts" / "next-gen-memory"))

from relations_index import RelationsIndexGenerator

class RelationshipProposalGenerator:
    """Generates relationship proposals for documents."""
    
    def __init__(self):
        self.rel_generator = RelationsIndexGenerator()
        self.proposals = []
        self.stats = defaultdict(int)
    
    def generate_proposals(self):
        """Generate relationship proposals for all documents."""
        print("Scanning documents...")
        documents = self.rel_generator.scan_documents()
        print(f"Found {len(documents)} documents")
        
        # Get existing relationships
        existing_rels = set()
        if (Path.home() / "lloyd" / "_pipeline" / "relations-index.json").exists():
            index = json.loads((Path.home() / "lloyd" / "_pipeline" / "relations-index.json").read_text())
            for rel in index.get("relationships", []):
                existing_rels.add((rel["source"], rel["target"], rel["type"]))
        
        # Generate proposals based on various heuristics
        print("\nGenerating proposals...")
        
        # 1. Path-based relationships (documents in same directory)
        path_groups = defaultdict(list)
        for doc in documents:
            path = Path(doc["path"])
            parent = str(path.parent)
            if parent != ".":
                path_groups[parent].append(doc["path"])
        
        for parent, paths in path_groups.items():
            if len(paths) > 1:
                for i, p1 in enumerate(paths):
                    for p2 in paths[i+1:]:
                        if (p1, p2, "related-to") not in existing_rels:
                            self._add_proposal(p1, p2, "related-to", 
                                f"Same directory: {parent}")
        
        # 2. Entity-based relationships (documents mentioning same entity)
        entity_docs = defaultdict(list)
        for doc in documents:
            # Extract entity from path
            path_parts = Path(doc["path"]).parts
            for part in path_parts:
                if part not in ["obsidian", "agents", "memory", "projects"]:
                    entity_docs[part.lower()].append(doc["path"])
        
        for entity, paths in entity_docs.items():
            if len(paths) > 1:
                for i, p1 in enumerate(paths):
                    for p2 in paths[i+1:]:
                        if (p1, p2, "related-to") not in existing_rels:
                            self._add_proposal(p1, p2, "related-to",
                                f"Related to entity: {entity}")
        
        # 3. Type-based relationships (same document type)
        type_groups = defaultdict(list)
        for doc in documents:
            try:
                content = (VAULT / doc["path"]).read_text()
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 2:
                        frontmatter = yaml.safe_load(parts[1])
                        doc_type = frontmatter.get("type", "unknown")
                        type_groups[doc_type].append(doc["path"])
            except:
                pass
        
        for doc_type, paths in type_groups.items():
            if len(paths) > 1 and doc_type not in ["facts", "index"]:
                for i, p1 in enumerate(paths):
                    for p2 in paths[i+1:]:
                        if (p1, p2, "related-to") not in existing_rels:
                            self._add_proposal(p1, p2, "related-to",
                                f"Same type: {doc_type}")
        
        # 4. Derives relationships (projects from specifications)
        spec_docs = [p for p in documents if "specs" in p["path"].lower() or "spec" in p["path"].lower()]
        project_docs = [p for p in documents if "project" in p["path"].lower() or "-project" in p["path"].lower()]
        
        for spec in spec_docs:
            for proj in project_docs:
                # Check if project name matches spec
                spec_name = Path(spec["path"]).stem.lower()
                proj_name = Path(proj["path"]).stem.lower()
                if spec_name in proj_name or proj_name in spec_name:
                    self._add_proposal(spec["path"], proj["path"], "produces",
                        "Specification produces project")
        
        print(f"\nGenerated {len(self.proposals)} relationship proposals")
        return self.proposals
    
    def _add_proposal(self, source, target, rel_type, reason):
        """Add a relationship proposal."""
        proposal = {
            "source": source,
            "target": target,
            "type": rel_type,
            "reason": reason,
            "status": "pending"
        }
        self.proposals.append(proposal)
        self.stats[rel_type] += 1
    
    def write_proposals_file(self):
        """Write proposals to review file."""
        lines = [
            "# Relationship Proposals",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Total Proposals:** {len(self.proposals)}",
            "",
            "## Statistics",
            "",
        ]
        
        for rel_type, count in sorted(self.stats.items(), key=lambda x: -x[1]):
            lines.append(f"- **{rel_type}:** {count}")
        
        lines.extend([
            "",
            "## Proposals",
            "",
        ])
        
        for i, prop in enumerate(self.proposals[:100], 1):  # Limit to 100 for readability
            lines.extend([
                f"### {i}. {prop['source']} → {prop['target']}",
                "",
                f"- **Type:** {prop['type']}",
                f"- **Reason:** {prop['reason']}",
                f"- **Status:** {prop['status']}",
                "",
            ])
        
        if len(self.proposals) > 100:
            lines.append(f"*...and {len(self.proposals) - 100} more proposals*")
        
        PROPOSALS_FILE.write_text("\n".join(lines))
        print(f"\nProposals written to {PROPOSALS_FILE}")
    
    def get_stats(self):
        """Get relationship statistics."""
        return {
            "total_proposals": len(self.proposals),
            "by_type": dict(self.stats),
            "documents_processed": len(self.rel_generator.scan_documents())
        }


def main():
    print("Relationship Proposal Generator - Next-Gen Memory")
    print("=" * 50)
    
    generator = RelationshipProposalGenerator()
    generator.generate_proposals()
    generator.write_proposals_file()
    
    stats = generator.get_stats()
    print("\nStatistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
