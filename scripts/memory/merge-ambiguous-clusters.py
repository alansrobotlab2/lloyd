#!/usr/bin/env python3
"""Direct merge of the 6 AMBIGUOUS entity clusters from the sweep."""
import json
import shutil
from pathlib import Path
from datetime import datetime

VAULT = Path.home() / "obsidian"
FACTS = VAULT / "facts"
RELATIONS = FACTS / "_relationships.json"
ALIASES = FACTS / "entity-aliases.json"

# The 6 AMBIGUOUS clusters with their canonical names
MERGES = {
    "openclaw": {
        "canonical": "openclaw",
        "variants": ["OpenClaw", "OpenClaw System", "openclaw_agent"]
    },
    "assistant": {
        "canonical": "assistant", 
        "variants": ["Assistant"]
    },
    "vaultmaintenance": {
        "canonical": "vault-maintenance",
        "variants": ["Vault Maintenance System", "Vault Maintenance Pipeline", "vault-maintenance-system"]
    },
    "memory": {
        "canonical": "Memory System",
        "variants": ["Memory Agent", "memory_system"]
    },
    "vaultaccessmodel": {
        "canonical": "Vault Access Model",
        "variants": ["Vault Access Model Research"]
    },
    "missioncontrolchat": {
        "canonical": "mission-control-chat",
        "variants": ["Mission Control Chat System"]
    }
}

def main():
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    
    # Backup
    backup_dir = FACTS / f"backups/{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(RELATIONS, backup_dir / "_relationships.json.bak")
    shutil.copy(ALIASES, backup_dir / "entity-aliases.json.bak")
    print(f"✓ Backed up to {backup_dir}")
    
    # Load data
    with open(RELATIONS) as f:
        relations = json.load(f)
    with open(ALIASES) as f:
        aliases = json.load(f)
    
    edges = relations.get("edges", [])
    updated_edges = []
    edge_changes = 0
    
    # Merge edges
    for edge in edges:
        from_entity = edge.get("source")
        to_entity = edge.get("target")
        
        new_from, new_to = from_entity, to_entity
        changed = False
        
        for canonical, data in MERGES.items():
            if from_entity in data["variants"]:
                new_from = data["canonical"]
                changed = True
            if to_entity in data["variants"]:
                new_to = data["canonical"]
                changed = True
        
        if changed:
            edge_changes += 1
            edge["source"] = new_from
            edge["target"] = new_to
        
        updated_edges.append(edge)
    
    relations["edges"] = updated_edges
    
    # Update aliases - simple mapping: variant -> canonical
    for canonical, data in MERGES.items():
        for variant in data["variants"]:
            aliases[variant] = canonical
    
    # Deduplicate edges
    seen = set()
    deduped_edges = []
    for edge in updated_edges:
        key = (edge["source"], edge["target"], edge.get("type", ""))
        if key not in seen:
            seen.add(key)
            deduped_edges.append(edge)
    
    relations["edges"] = deduped_edges
    
    # Write
    with open(RELATIONS, "w") as f:
        json.dump(relations, f, indent=2)
    with open(ALIASES, "w") as f:
        json.dump(aliases, f, indent=2)
    
    print(f"✓ Merged {len(MERGES)} clusters")
    print(f"✓ Updated {edge_changes} edges")
    print(f"✓ Deduplicated {len(updated_edges) - len(deduped_edges)} duplicate edges")
    print(f"✓ Final edge count: {len(deduped_edges)}")
    
    # Move fact directories
    moved = 0
    for canonical, data in MERGES.items():
        for variant in data["variants"]:
            variant_dir = FACTS / variant
            if variant_dir.exists():
                shutil.move(str(variant_dir), str(FACTS / canonical))
                moved += 1
                print(f"  Moved {variant} → {canonical}")
    
    print(f"✓ Moved {moved} fact directories")
    print("✓ Done!")

if __name__ == "__main__":
    main()
