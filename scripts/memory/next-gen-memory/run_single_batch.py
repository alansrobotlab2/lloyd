#!/usr/bin/env python3
"""
Fact Extraction Runner - Process ONE batch only
"""

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path.home() / "obsidian" / "agents" / "memory" / "scripts" / "next-gen-memory"))

from fact_extractor import FactExtractor

VAULT = Path.home() / "obsidian"
MEMORY_DIR = VAULT / "memory"
FACTS_DIR = MEMORY_DIR / "facts"

def main():
    extractor = FactExtractor()
    
    # Get all documents
    docs = [f for f in VAULT.rglob("*.md") 
         if not any(d in str(f) for d in ["/proc/", "/sys/", "/run/", "/etc/", "/bin/", "/sbin/", "/lib/", "/lib64/", "/usr/", "/dev/", "/var/"])]
    docs = [d for d in docs if all(part not in str(d) for part in ["node_modules", ".venv", "facts", ".cache", "__pycache__"])]
    
    batch_size = 20
    
    # Load resume state
    progress_file = MEMORY_DIR / "fact-extraction-progress.json"
    processed_docs = set()
    start_fact_count = 0
    
    if progress_file.exists():
        try:
            progress_data = json.loads(progress_file.read_text())
            processed_docs = set(progress_data.get("processed_docs", []))
            start_fact_count = progress_data.get("total_facts", 0)
            print(f"Resuming: {len(processed_docs)} documents already processed, {start_fact_count} facts")
        except Exception as e:
            print(f"Warning: Could not load progress: {e}")
    
    remaining_docs = [d for d in docs if str(d) not in processed_docs]
    
    print(f"Remaining: {len(remaining_docs)} docs")
    print(f"Processing first {batch_size} docs...")
    print()
    
    total_facts = start_fact_count
    errors = 0
    
    batch = remaining_docs[:batch_size]
    
    for doc in batch:
        doc_path = str(doc)
        
        try:
            content = doc.read_text()
            
            # Skip files without frontmatter
            if not content.startswith("---"):
                processed_docs.add(doc_path)
                continue
            
            result = extractor.extract_from_document(doc, content, existing_facts="")
            
            if result.get("facts") and len(result["facts"]) > 0:
                entity = result.get("entity", "general")
                category = result.get("category", "general")
                fact_file = extractor.write_fact_file(entity, category, result)
                total_facts += len(result["facts"])
                if fact_file:
                    print(f"  + {len(result['facts'])} facts -> {fact_file.name}")
            
            processed_docs.add(doc_path)
            
        except Exception as e:
            print(f"  ERROR {doc.name}: {e}")
            errors += 1
            processed_docs.add(doc_path)
    
    # Save progress
    progress_data = {
        "processed_docs": list(processed_docs),
        "total_facts": total_facts,
        "errors": errors,
        "last_batch": 1,
        "timestamp": datetime.now().isoformat()
    }
    progress_file.write_text(json.dumps(progress_data, indent=2))
    
    print(f"\nBatch complete: {len(batch)} docs processed")
    print(f"Total facts: {total_facts}, Errors: {errors}")

if __name__ == "__main__":
    main()
