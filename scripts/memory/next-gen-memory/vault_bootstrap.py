#!/usr/bin/env python3
"""
Full Vault Bootstrap - Phase 10

Processes all 872 existing documents for facts, relations, and indexes.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

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
FACTS_DIR = MEMORY_DIR / "facts"

sys.path.insert(0, str(VAULT / "agents" / "memory" / "scripts" / "next-gen-memory"))

from fact_extractor import FactExtractor
from relations_index import RelationsIndexGenerator


class VaultBootstrap:
    """Full vault bootstrap for Phase 10."""
    
    def __init__(self):
        self.extractor = FactExtractor()
        self.rel_generator = RelationsIndexGenerator()
        self.log_file = MEMORY_DIR / "vault-bootstrap.log"
        self.status_file = MEMORY_DIR / "bootstrap-status.json"
    
    def run_full_bootstrap(self):
        """Run complete vault bootstrap."""
        start_time = datetime.now()
        
        log_lines = [
            f"\n{'='*60}",
            f"Full Vault Bootstrap Started: {start_time.isoformat()}",
            f"{'='*60}"
        ]
        
        results = {
            "started_at": start_time.isoformat(),
            "steps": {},
            "errors": []
        }
        
        try:
            # Step 1: Classification (513 untyped docs)
            log_lines.append("\n[Step 1] Full Vault Classification")
            classified = self._classify_untyped_documents()
            results["steps"]["classification"] = {"classified": classified}
            log_lines.append(f"  → Classified {classified} documents")
            
            # Step 2: Fact Extraction (872 docs)
            log_lines.append("\n[Step 2] Full Vault Fact Extraction")
            facts_extracted = self._extract_all_facts_batch()
            results["steps"]["fact_extraction"] = {"facts": facts_extracted}
            log_lines.append(f"  → Extracted {facts_extracted} facts")
            
            # Step 3: Relationship Proposals
            log_lines.append("\n[Step 3] Full Vault Relationship Proposals")
            relations_proposed = self._propose_all_relations()
            results["steps"]["relationship_proposals"] = {"proposed": relations_proposed}
            log_lines.append(f"  → Proposed {relations_proposed} relationships")
            
            # Step 4: Index Rebuild
            log_lines.append("\n[Step 4] Rebuild All Indexes")
            index = self.rel_generator.rebuild()
            results["steps"]["index_rebuild"] = {"relationships": index['total_relationships']}
            log_lines.append(f"  → Rebuilt index with {index['total_relationships']} relationships")
            
            # Summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results["completed_at"] = end_time.isoformat()
            results["duration_seconds"] = duration
            results["success"] = True
            
            log_lines.extend([
                f"\n{'='*60}",
                f"Full Vault Bootstrap Complete: {end_time.isoformat()}",
                f"Duration: {duration:.1f} seconds",
                f"{'='*60}\n"
            ])
            
            # Write outputs
            self.log_file.write_text("\n".join(log_lines))
            self.status_file.write_text(json.dumps(results, indent=2))
            
            return results
            
        except Exception as e:
            results["errors"].append(str(e))
            results["success"] = False
            log_lines.append(f"\nERROR: {str(e)}")
            self.log_file.write_text("\n".join(log_lines))
            self.status_file.write_text(json.dumps(results, indent=2))
            raise
    
    def _classify_untyped_documents(self) -> int:
        """Classify documents without type in frontmatter."""
        classified = 0
        
        for md_file in VAULT.rglob("*.md"):
            # Strict vault path validation
            path_str = str(md_file)
            if not path_str.startswith(str(VAULT) + "/"):
                continue
            # Skip excluded dirs
            if any(part in str(md_file) for part in ["node_modules", ".venv", ".cache", "facts"]):
                continue
            
            try:
                content = md_file.read_text()
                
                if not content.startswith("---"):
                    continue
                
                parts = content.split("---", 2)
                if len(parts) < 2:
                    continue
                
                frontmatter = yaml.safe_load(parts[1])
                
                # Check if already typed
                if "type" in frontmatter:
                    continue
                
                # Classify (simple heuristic)
                segment = self._infer_segment(md_file)
                doc_type = self._infer_type(md_file, frontmatter)
                
                # Update frontmatter
                frontmatter["type"] = doc_type
                frontmatter["segment"] = segment
                
                # Write back
                yaml_content = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
                new_content = f"---\n{yaml_content}---\n{parts[2]}"
                md_file.write_text(new_content)
                
                classified += 1
                
            except Exception as e:
                print(f"Error classifying {md_file}: {e}")
        
        return classified
    
    def _infer_segment(self, file_path: Path) -> str:
        """Infer document segment from path."""
        path_str = str(file_path)
        
        if "/agents/" in path_str:
            return "agents"
        elif "/personal/" in path_str:
            return "personal"
        elif "/work/" in path_str:
            return "work"
        elif "/projects/" in path_str:
            return "projects"
        elif "/knowledge/" in path_str:
            return "knowledge"
        elif "/memory/" in path_str:
            return "memory"
        elif "/skills/" in path_str:
            return "skills"
        
        return "projects"
    
    def _infer_type(self, file_path: Path, 
                    frontmatter: dict) -> str:
        """Infer document type."""
        filename = file_path.name.lower()
        
        # Check existing metadata
        if "title" in frontmatter:
            title = frontmatter["title"].lower()
            if "fact" in title or "fact" in filename:
                return "facts"
            elif "profile" in title:
                return "profile"
            elif "index" in title:
                return "index"
        
        # Check filename patterns
        if filename.endswith("-notes.md"):
            return "notes"
        elif filename.endswith("-specs.md") or "spec" in filename:
            return "reference"
        elif filename.endswith("-complete.md") or "complete" in filename:
            return "project-notes"
        elif filename.startswith("20") and filename.endswith(".md"):
            return "notes"
        elif "readme" in filename:
            return "notes"
        
        # Default based on segment
        segment = self._infer_segment(file_path)
        if segment in ["projects", "work"]:
            return "project-notes"
        elif segment == "knowledge":
            return "reference"
        else:
            return "notes"
    
    def _extract_all_facts_batch(self) -> int:
        """Batch extract facts from all documents with progress tracking and resume."""
        total_facts = 0
        processed = 0
        errors = 0
        
        # Process in batches
        batch_size = 50
        docs = [f for f in VAULT.rglob("*.md") 
         if not any(d in str(f) for d in ["/proc/", "/sys/", "/run/", "/etc/", "/bin/", "/sbin/", "/lib/", "/lib64/", "/usr/", "/dev/", "/var/"])]
        
        # Filter to relevant docs
        docs = [d for d in docs if "node_modules" not in str(d) 
                and ".venv" not in str(d) 
                and "facts" not in str(d)
                and ".cache" not in str(d)]
        
        total_docs = len(docs)
        total_batches = (total_docs + batch_size - 1) // batch_size
        
        # Load resume state if exists
        resume_file = MEMORY_DIR / "fact-extraction-progress.json"
        processed_docs = set()
        if resume_file.exists():
            try:
                progress_data = json.loads(resume_file.read_text())
                processed_docs = set(progress_data.get("processed_docs", []))
                print(f"  Resuming from batch {len(processed_docs)//batch_size + 1}/{total_batches}")
            except:
                pass
        
        for batch_num, i in enumerate(range(0, len(docs), batch_size), 1):
            batch = docs[i:i+batch_size]
            
            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} docs)...")
            
            for doc in batch:
                doc_path = str(doc)
                
                # Skip already processed
                if doc_path in processed_docs:
                    processed += 1
                    continue
                
                try:
                    doc_content = doc.read_text()
                    
                    # Extract facts
                    result = self.extractor.extract_from_document(
                        doc, doc_content, existing_facts=""
                    )
                    
                    if result.get("facts") and len(result["facts"]) > 0:
                        entity = result.get("entity", "general")
                        category = result.get("category", "general")
                        
                        fact_file = self.extractor.write_fact_file(
                            entity, category, result
                        )
                        
                        total_facts += len(result["facts"])
                    
                    # Mark as processed
                    processed_docs.add(doc_path)
                    processed += 1
                    
                except Exception as e:
                    print(f"    Error processing {doc.name}: {e}")
                    errors += 1
                    processed_docs.add(doc_path)  # Don't retry failed docs
                    processed += 1
            
            # Save progress after each batch
            progress_data = {
                "processed_docs": list(processed_docs),
                "total_facts": total_facts,
                "errors": errors,
                "last_batch": batch_num
            }
            resume_file.write_text(json.dumps(progress_data, indent=2))
            
            # Progress report
            progress_pct = (processed / total_docs) * 100
            print(f"    Progress: {processed}/{total_docs} docs ({progress_pct:.1f}%), {total_facts} facts, {errors} errors")
        
        return total_facts
    
    def _propose_all_relations(self) -> int:
        """Propose relationships for all documents."""
        documents = self.rel_generator.scan_documents()
        proposals = 0
        
        # Simple similarity-based proposal
        for i, doc1 in enumerate(documents[:100]):  # Limit for performance
            for doc2 in documents[i+1:100]:
                similarity = self._calculate_similarity(doc1, doc2)
                
                if similarity > 0.6:
                    # Propose relation
                    if self.rel_generator.add_relation(
                        doc1["path"], doc2["path"], "related-to"
                    ):
                        proposals += 1
        
        return proposals
    
    def _calculate_similarity(self, doc1: dict, doc2: dict) -> float:
        """Calculate document similarity."""
        words1 = set(doc1.get("path", "").lower().split())
        words2 = set(doc2.get("path", "").lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        return overlap / max(len(words1), len(words2))


def main():
    print("Full Vault Bootstrap - Phase 10")
    print("=" * 50)
    
    bootstrap = VaultBootstrap()
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        print("Dry run mode - would process all documents")
    else:
        result = bootstrap.run_full_bootstrap()
        print("\nBootstrap Result:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
