#!/usr/bin/env python3
"""
Nightly Deep Extraction - Next-Gen Memory System

Runs at 2 AM PST with the primary model for comprehensive extraction.
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Content hashing for incremental processing
sys.path.insert(0, str(Path.home() / "lloyd" / "scripts" / "memory"))
try:
    from content_hasher import ContentHasher
    _HAS_HASHER = True
except ImportError:
    _HAS_HASHER = False

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from app.paths import VAULT_FACTS_ROOT as FACTS_DIR
VAULT = Path.home() / "obsidian"
MEMORY_DIR = VAULT / "memory"
INDEX_FILE = Path(__file__).resolve().parent.parent.parent.parent / "_pipeline" / "relations-index.json"

# Import local modules
sys.path.insert(0, str(VAULT / "agents" / "memory" / "scripts" / "next-gen-memory"))

from fact_extractor import FactExtractor
from relations_index import RelationsIndexGenerator
from profile_generator import ProfileGenerator


class NightlyExtraction:
    """Nightly deep extraction with primary model."""

    def __init__(self):
        self.extractor = FactExtractor(model_port=8096)  # Uses primary for deep extraction
        self.rel_generator = RelationsIndexGenerator()
        self.profile_generator = ProfileGenerator(model_port=8096)
        self.log_file = Path.home() / "lloyd" / "_pipeline" / "nightly-extraction.log"
        # Thread-safe locks per entity for parallel processing
        self.entity_locks = {}
        self.locks_lock = threading.Lock()
    
    def _get_entity_lock(self, entity: str) -> threading.Lock:
        """Get or create a lock for an entity."""
        with self.locks_lock:
            if entity not in self.entity_locks:
                self.entity_locks[entity] = threading.Lock()
            return self.entity_locks[entity]
    
    def normalize_entity_name(self, entity: str) -> str:
        """Normalize entity name using alias registry.
        
        Loads entity-registry.json and checks if the entity is an alias.
        If it is an alias, returns the canonical_name; otherwise returns
        the entity as-is. Handles missing registry gracefully.
        
        Args:
            entity: Entity name to normalize
            
        Returns:
            Canonical entity name or original if not in registry
        """
        registry_path = FACTS_DIR / "entity-registry.json"
        
        if not registry_path.exists():
            # Fallback to no normalization if registry missing
            return entity
        
        try:
            import json
            registry = json.loads(registry_path.read_text())
            canonical_mapping = registry.get("canonical_mapping", {})
            
            # Check if entity is an alias
            if entity in canonical_mapping:
                return canonical_mapping[entity].get("canonical_name", entity)
            
            # Check if entity matches any alias in the registry
            for canonical, data in canonical_mapping.items():
                if entity in data.get("aliases", []):
                    return canonical
            
            # Entity not found in registry, return as-is
            return entity
            
        except Exception as e:
            print(f"Warning: Failed to load entity registry: {e}")
            return entity

    def check_preflight_dedup(self) -> dict:
        """Pre-flight check for duplicate fact groups before extraction.
        
        Scans facts/ directory for duplicate groups using entity normalization.
        This is a prevention mechanism to detect any new duplicates that may
        have been created since the last cleanup.
        
        Returns:
            Dictionary with duplicate_groups count and list of affected entities
        """
        import json
        from collections import defaultdict
        
        # Load entity registry
        registry_path = FACTS_DIR / "entity-registry.json"
        canonical_mapping = {}
        
        if registry_path.exists():
            try:
                registry = json.loads(registry_path.read_text())
                canonical_mapping = registry.get("canonical_mapping", {})
            except Exception as e:
                print(f"Warning: Failed to load entity registry: {e}")
        
        # Build reverse mapping: alias -> canonical
        alias_to_canonical = {}
        for canonical, data in canonical_mapping.items():
            for alias in data.get("aliases", []):
                alias_to_canonical[alias] = canonical
        
        # Scan facts directory and group by canonical entity
        entity_groups = defaultdict(list)
        
        if FACTS_DIR.exists():
            for entity_dir in FACTS_DIR.iterdir():
                if not entity_dir.is_dir() or entity_dir.name == "templates":
                    continue
                
                # Normalize the entity name
                canonical = self.normalize_entity_name(entity_dir.name)
                entity_groups[canonical].append(entity_dir.name)
        
        # Identify groups with multiple entities (potential duplicates)
        duplicate_groups = []
        for canonical, entities in entity_groups.items():
            if len(entities) > 1:
                duplicate_groups.append({
                    "canonical": canonical,
                    "entities": entities,
                    "count": len(entities)
                })
        
        return {
            "duplicate_groups": len(duplicate_groups),
            "groups": duplicate_groups,
            "checked_at": datetime.now().isoformat()
        }
    
    def clean_facts_directory(self):
        """Wipe facts directory (preserve entity-registry.json)."""
        if not FACTS_DIR.exists():
            print("  → Facts directory does not exist, skipping clean")
            return
        
        preserved_file = FACTS_DIR / "entity-registry.json"
        preserved_data = None
        
        # Save registry if it exists
        if preserved_file.exists():
            preserved_data = preserved_file.read_text()
            print(f"  → Preserving entity-registry.json")
        
        # Remove all entity subdirectories
        for item in FACTS_DIR.iterdir():
            if item.name == "templates":
                continue
            if item.is_dir():
                import shutil
                shutil.rmtree(item)
                print(f"  → Removed {item.name}/")
            elif item.name != "entity-registry.json":
                item.unlink()
                print(f"  → Removed {item.name}")
        
        # Restore registry if it existed
        if preserved_data:
            preserved_file.write_text(preserved_data)
        
        print(f"  → Facts directory cleaned")
    
    def run_full_extraction(self, full_mode=False, workers=1, clean=False, limit=0, force=False):
        """Run complete nightly extraction pipeline."""
        start_time = datetime.now()
        log_lines = [
            f"\n{'='*60}",
            f"Nightly Extraction Started: {start_time.isoformat()}",
            f"Mode: {'Full Vault' if full_mode else '24h Window'}",
            f"Workers: {workers}",
            f"Clean: {'Yes' if clean else 'No'}",
            f"{'='*60}"
        ]
        
        try:
            # Clean facts directory if requested
            if clean:
                log_lines.append("\n[Pre-Flight] Cleaning Facts Directory")
                self.clean_facts_directory()
            
            # Pre-flight check: scan for duplicate groups
            log_lines.append("\n[Pre-Flight] Duplicate Group Detection")
            preflight = self.check_preflight_dedup()
            dup_count = preflight["duplicate_groups"]
            if dup_count > 0:
                log_lines.append(f"  ⚠️ WARNING: Found {dup_count} duplicate group(s)!")
                for group in preflight["groups"]:
                    log_lines.append(f"    - {group['canonical']}: {group['entities']}")
                log_lines.append("  → These should be merged before proceeding")
            else:
                log_lines.append(f"  ✓ No duplicate groups detected (prevention working)")
            
            # Step 1: Full vault fact extraction
            log_lines.append("\n[Step 1] Full Vault Fact Extraction")
            self.last_files_processed = 0
            facts_extracted = self._extract_all_facts(full_mode=full_mode, workers=workers, limit=limit)
            files_processed = getattr(self, "last_files_processed", 0)
            log_lines.append(f"  → Processed {files_processed} files, extracted {facts_extracted} new facts")

            # Gate: Steps 2-5 (derives, relation discovery, index rebuild, overview
            # generation) only have new material to chew on when extraction touched
            # at least one file. Running them every poll on an unchanged vault is the
            # expense the outer skill used to guard with a git-commit watermark — now
            # the content-hash result is the authoritative signal. --full/--clean/--force
            # always run the complete pipeline.
            if files_processed == 0 and not full_mode and not clean and not force:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                log_lines.append("  → No new/changed files; skipping relation, index, and overview steps")
                log_lines.append(f"\nNightly Extraction Complete (noop): {end_time.isoformat()} ({duration:.1f}s)")
                self.log_file.write_text("\n".join(log_lines))
                return {
                    "success": True,
                    "noop": True,
                    "files_processed": 0,
                    "facts_extracted": 0,
                    "derives_created": 0,
                    "relations_found": 0,
                    "total_relationships": None,
                    "overviews_generated": 0,
                    "duration_seconds": duration,
                }

            # Step 2: Derives relationship inference
            log_lines.append("\n[Step 2] Derives Relationship Inference")
            derives_created = self._infer_derives_relationships()
            log_lines.append(f"  → Created {derives_created} derives relationships")
            
            # Step 3: Automated relation discovery
            log_lines.append("\n[Step 3] Automated Relation Discovery")
            relations_found = self._discover_relations()
            log_lines.append(f"  → Found {relations_found} new relations")
            
            # Step 4: Index rebuild
            log_lines.append("\n[Step 4] Index Rebuild")
            index = self.rel_generator.rebuild()
            log_lines.append(f"  → Rebuilt index with {index['total_relationships']} relationships")

            # Step 5: Entity overview generation (change-triggered)
            log_lines.append("\n[Step 5] Entity Overview Generation")
            overviews_generated = self._regenerate_entity_overviews()
            log_lines.append(f"  → Generated/updated {overviews_generated} entity overviews")

            # Summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            log_lines.extend([
                f"\n{'='*60}",
                f"Nightly Extraction Complete: {end_time.isoformat()}",
                f"Duration: {duration:.1f} seconds",
                f"{'='*60}\n"
            ])

            # Write log
            self.log_file.write_text("\n".join(log_lines))

            return {
                "success": True,
                "noop": False,
                "files_processed": files_processed,
                "duration_seconds": duration,
                "facts_extracted": facts_extracted,
                "derives_created": derives_created,
                "relations_found": relations_found,
                "total_relationships": index['total_relationships'],
                "overviews_generated": overviews_generated,
            }
            
        except Exception as e:
            log_lines.append(f"\nERROR: {str(e)}")
            self.log_file.write_text("\n".join(log_lines))
            raise
    
    def _process_single_file(self, md_file, full_mode, index, total):
        """Process a single file for fact extraction. Thread-safe."""
        processed = 0
        facts_count = 0
        
        try:
            mtime = datetime.fromtimestamp(md_file.stat().st_mtime)
            # Skip if not modified in 24h and not full mode
            if not full_mode and (datetime.now() - mtime).total_seconds() > 86400:
                return 0, 0
            
            content = md_file.read_text()
            doc_path = str(md_file.relative_to(VAULT))
            
            # Extract facts
            result = self.extractor.extract_from_document(
                md_file, content, existing_facts=""
            )
            
            if result.get("facts"):
                entity = result.get("entity", "general")
                category = result.get("category", "general")
                
                # Thread-safe write with entity-specific lock
                lock = self._get_entity_lock(entity)
                with lock:
                    fact_file = self.extractor.write_fact_file(
                        entity, category, result
                    )
                
                facts_count += len(result["facts"])
                processed = 1
                print(f"[{index}/{total}] Processing: {doc_path}")
        
        except Exception as e:
            print(f"Error processing {md_file}: {e}")
        
        return processed, facts_count
    
    def _extract_all_facts(self, full_mode=False, workers=1, limit=0) -> int:
        """Extract facts from all documents."""
        total_facts = 0
        processed = 0
        
        # Collect eligible files
        eligible_files = []
        for md_file in VAULT.rglob("*.md"):
            # Skip directories (e.g. facts/ entities stored as .md directories)
            if not md_file.is_file():
                continue

            # Strict vault path validation
            path_str = str(md_file)
            if not path_str.startswith(str(VAULT) + "/"):
                continue
            
            # Skip system directories that might be symlinked into vault
            if any(sys_dir in path_str for sys_dir in [
                "/proc/", "/sys/", "/run/", "/etc/", "/bin/", "/sbin/", 
                "/lib/", "/lib64/", "/usr/", "/dev/", "/var/"
            ]):
                continue
            
            # Skip other excluded dirs
            if any(part in path_str for part in ["node_modules", ".venv", ".cache"]):
                continue
            
            # Skip fact files, pipeline artifacts, skills, and agent workspace files.
            # /autonomy/ is excluded too: the scheduler rewrites task frontmatter
            # (last_run/next_run) every ~15 min, so those files always look "changed"
            # — they'd keep the content-hash gate from ever going noop and produce
            # junk entities like "Autonomy Task #32". Task defs are not knowledge.
            if any(skip in path_str for skip in ["/facts/", "/_pipeline/", "/skills/", "/agents/", "/autonomy/"]):
                continue
            
            # Check modification time (unless full mode)
            if not full_mode:
                mtime = datetime.fromtimestamp(md_file.stat().st_mtime)
                if (datetime.now() - mtime).total_seconds() >= 86400:
                    continue
            
            eligible_files.append(md_file)
        
        total_files = len(eligible_files)
        print(f"Found {total_files} eligible files")

        # Filter by content hash - skip files already extracted (unchanged content).
        # Applied in BOTH window and full mode so --full is a resumable backfill that
        # skips already-done files rather than reprocessing the whole corpus.
        hasher = ContentHasher() if _HAS_HASHER else None
        if hasher is not None:
            changed_files = hasher.get_changed_files(eligible_files)
            skipped_unchanged = len(eligible_files) - len(changed_files)
            eligible_files = changed_files
            if skipped_unchanged > 0:
                print(f"Skipped {skipped_unchanged} unchanged files (content hash match)")

        # Optional per-run cap so a long backfill makes durable, bounded progress and
        # exits cleanly before the bash timeout (use with --full).
        if limit and limit > 0 and len(eligible_files) > limit:
            print(f"Limiting this run to {limit} of {len(eligible_files)} eligible files")
            eligible_files = eligible_files[:limit]

        # Incremental checkpoint: persist hashes for already-processed files every
        # CHECKPOINT_EVERY files so a timeout/kill never loses the whole run's work.
        CHECKPOINT_EVERY = 25
        pending: list = []

        def _flush_checkpoint():
            if hasher is not None and pending:
                try:
                    hasher.update_hashes(pending)
                    hasher.save()
                except Exception as e:
                    print(f"Warning: checkpoint failed: {e}")
                pending.clear()

        if workers == 1:
            # Sequential processing (default)
            for index, md_file in enumerate(eligible_files, 1):
                p, f = self._process_single_file(md_file, full_mode, index, total_files)
                processed += p
                total_facts += f
                pending.append(md_file)
                if len(pending) >= CHECKPOINT_EVERY:
                    _flush_checkpoint()
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(self._process_single_file, md_file, full_mode, index, total_files):
                    (index, md_file)
                    for index, md_file in enumerate(eligible_files, 1)
                }

                # Collect results
                for future in as_completed(futures):
                    index, md_file = futures[future]
                    try:
                        p, f = future.result()
                        processed += p
                        total_facts += f
                        pending.append(md_file)
                        if len(pending) >= CHECKPOINT_EVERY:
                            _flush_checkpoint()
                    except Exception as e:
                        print(f"Error processing {md_file}: {e}")

        print(f"Processed {processed} documents, extracted {total_facts} facts")

        # Final checkpoint for any remaining processed files
        _flush_checkpoint()
        if hasher is not None:
            print(f"Updated content hashes for {processed} processed files")

        # Expose file-processed count so run_full_extraction can gate the
        # expensive downstream steps on whether any new/changed file was seen.
        self.last_files_processed = processed
        return total_facts
    
    def _infer_derives_relationships(self) -> int:
        """Infer derives relationships between facts."""
        # Load all facts
        facts_by_entity = {}
        
        for entity_dir in FACTS_DIR.iterdir():
            if not entity_dir.is_dir() or entity_dir.name == "templates":
                continue
            
            entity = entity_dir.name
            facts_by_entity[entity] = []
            
            for fact_file in entity_dir.glob("*.md"):
                if not fact_file.is_file():
                    continue
                content = fact_file.read_text()
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 2:
                        try:
                            frontmatter = yaml.safe_load(parts[1])
                            for fact in frontmatter.get("facts", []):
                                facts_by_entity[entity].append({
                                    "file": fact_file.name,
                                    **fact
                                })
                        except:
                            pass
        
        # Simple derives inference
        derives_count = 0
        
        for entity, facts in facts_by_entity.items():
            for i, f1 in enumerate(facts):
                if not isinstance(f1, dict):
                    continue
                for f2 in facts[i+1:]:
                    if not isinstance(f2, dict):
                        continue
                    if "fact" not in f1 or "fact" not in f2:
                        continue
                    # Check if f2 derives from f1
                    if self._check_derives(f1, f2):
                        # Add relation
                        source = f"memory/facts/{entity}/{f1['file']}"
                        target = f"memory/facts/{entity}/{f2['file']}"
                        
                        # Skip if file doesn't exist or is a directory
                        if not os.path.isfile(source) or not os.path.isfile(target):
                            continue
                        
                        if self.rel_generator.add_relation(source, target, "derived-from"):
                            derives_count += 1
        
        return derives_count
    
    def _check_derives(self, fact1: dict, fact2: dict) -> bool:
        """Check if fact2 derives from fact1."""
        # Guard against None/empty inputs from corrupt fact data
        if not fact1 or not fact2:
            return False
        fact1_text = fact1.get("fact", "")
        fact2_text = fact2.get("fact", "")
        if not fact1_text or not fact2_text:
            return False
        # Simple heuristic: check if fact2 references fact1's content
        text1 = fact1_text.lower()
        text2 = fact2_text.lower()
        
        # Check for shared key terms
        words1 = set(text1.split())
        words2 = set(text2.split())
        shared = words1 & words2
        
        return len(shared) >= 2 and len(text2) > len(text1)
    
    def _regenerate_entity_overviews(self) -> int:
        """Regenerate entity overview files whose source facts have changed."""
        return self.profile_generator.regenerate_all(workers=8)

    def _discover_relations(self) -> int:
        """Discover new document relationships."""
        # Scan documents for potential relations
        documents = self.rel_generator.scan_documents()
        
        # Similarity-based discovery using relations_index implementation
        new_relations = 0
        
        for i, doc1 in enumerate(documents[:50]):  # Limit for performance
            for doc2 in documents[i+1:50]:
                # Check similarity using relations_index implementation
                similarity = self.rel_generator._calculate_similarity(doc1, doc2)
                
                if similarity > 0.5:
                    # Add relation
                    if self.rel_generator.add_relation(
                        doc1["path"], doc2["path"], "related-to"
                    ):
                        new_relations += 1
        
        return new_relations


def main():
    print("Nightly Deep Extraction - Next-Gen Memory")
    print("=" * 50)
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Nightly fact extraction for Next-Gen Memory System"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Process ALL files regardless of modification time (default: 24h window)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe facts directory before extraction"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Just rebuild index (existing behavior)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run the full downstream pipeline (relations/index/overviews) even "
             "when no new files were processed (overrides the noop short-circuit)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N eligible files this run (0 = no cap). "
             "Use with --full for durable, bounded backfill that checkpoints "
             "progress and exits cleanly before the bash timeout."
    )

    args = parser.parse_args()
    
    extraction = NightlyExtraction()
    
    if args.quick:
        # Quick mode: just rebuild index
        print("Quick mode: rebuilding index...")
        index = extraction.rel_generator.rebuild()
        print(f"Total relationships: {index['total_relationships']}")
    else:
        # Full extraction with options
        result = extraction.run_full_extraction(
            full_mode=args.full,
            workers=args.workers,
            clean=args.clean,
            limit=args.limit,
            force=args.force,
        )
        print("\nResult:", json.dumps(result, indent=2))
        # Single-line, grep-friendly summary for the autonomy-data-pipeline skill's
        # gate. status=noop means nothing changed → caller should skip downstream work.
        print(
            f"PIPELINE_RESULT files_processed={result.get('files_processed', 0)} "
            f"facts={result.get('facts_extracted', 0)} "
            f"relations={result.get('relations_found', 0)} "
            f"status={'noop' if result.get('noop') else 'ran'}"
        )


if __name__ == "__main__":
    main()
