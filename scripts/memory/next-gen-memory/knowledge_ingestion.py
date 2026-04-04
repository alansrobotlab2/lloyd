#!/usr/bin/env python3
"""
Knowledge Ingestion - Next-Gen Memory System

Integrates research output and project status into memory system.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

VAULT = Path.home() / "obsidian"
MEMORY_DIR = VAULT / "memory"
FACTS_DIR = MEMORY_DIR / "facts"
RESEARCH_DIR = VAULT / "projects" / "lloyd" / "research"

sys.path.insert(0, str(VAULT / "agents" / "memory" / "scripts" / "next-gen-memory"))

from fact_extractor import FactExtractor
from mcp_tools import MemoryMCPTools


class KnowledgeIngestion:
    """Ingests research and project status into memory."""
    
    def __init__(self):
        self.extractor = FactExtractor()
        self.tools = MemoryMCPTools()
        self.research_dir = RESEARCH_DIR
    
    def ingest_research(self, research_path: str, 
                        project: str = None) -> dict:
        """Ingest research output into memory."""
        research_file = VAULT / research_path
        
        if not research_file.exists():
            return {"error": f"Research file not found: {research_path}"}
        
        content = research_file.read_text()
        
        # Extract facts from research
        result = self.extractor.extract_from_document(
            research_file, content, existing_facts=""
        )
        
        if not result.get("facts"):
            return {"error": "No facts extracted", "research_path": research_path}
        
        # Write facts
        entity = result.get("entity", "research")
        category = result.get("category", "findings")
        
        fact_file = self.extractor.write_fact_file(entity, category, result)
        
        return {
            "success": True,
            "research_path": research_path,
            "facts_added": len(result["facts"]),
            "fact_file": str(fact_file.relative_to(VAULT))
        }
    
    def track_project_status(self, project_name: str, 
                             status_data: dict) -> dict:
        """Track project status as facts."""
        entity = "work"
        category = f"{project_name}-status"
        
        # Convert status data to facts
        facts = []
        
        if "completion" in status_data:
            facts.append({
                "fact": f"{project_name} is {status_data['completion']} complete",
                "confidence": 0.95,
                "category": "state"
            })
        
        if "blockers" in status_data:
            for blocker in status_data["blockers"]:
                facts.append({
                    "fact": f"Blocker: {blocker}",
                    "confidence": 0.9,
                    "category": "state"
                })
        
        if "next_milestone" in status_data:
            facts.append({
                "fact": f"Next milestone: {status_data['next_milestone']}",
                "confidence": 0.9,
                "category": "goal"
            })
        
        if not facts:
            return {"error": "No status data to ingest"}
        
        # Write facts
        result = {
            "entity": entity,
            "category": category,
            "source": f"projects/{project_name}/status",
            "document_date": datetime.now().strftime("%Y-%m-%d"),
            "facts": facts
        }
        
        fact_file = self.extractor.write_fact_file(entity, category, result)
        
        return {
            "success": True,
            "project": project_name,
            "facts_added": len(facts),
            "fact_file": str(fact_file.relative_to(VAULT))
        }
    
    def detect_knowledge_gaps(self) -> dict:
        """Detect knowledge gaps in the memory system."""
        gaps = []
        
        # Define expected entities
        expected_entities = ["alan", "lloyd", "alfie", "stompy", "yoshi", "work"]
        
        # Check entity coverage
        for entity in expected_entities:
            entity_dir = FACTS_DIR / entity
            if not entity_dir.exists():
                gaps.append({
                    "type": "missing_entity",
                    "entity": entity,
                    "severity": "high"
                })
                continue
            
            # Check categories
            categories = [f.stem for f in entity_dir.glob("*.md")]
            
            # Check for outdated facts
            for fact_file in entity_dir.glob("*.md"):
                content = fact_file.read_text()
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 2:
                        try:
                            frontmatter = yaml.safe_load(parts[1])
                            last_updated = frontmatter.get("last_updated", "")
                            
                            if last_updated:
                                try:
                                    updated_date = datetime.fromisoformat(
                                        last_updated.replace("Z", "+00:00")
                                    )
                                    days_old = (datetime.now() - updated_date).days
                                    
                                    if days_old > 30:
                                        gaps.append({
                                            "type": "outdated_facts",
                                            "entity": entity,
                                            "file": fact_file.name,
                                            "days_old": days_old,
                                            "severity": "medium"
                                        })
                                except:
                                    pass
                        except:
                            pass
        
        # Check for low coverage categories
        low_coverage_categories = ["skills", "goals", "relationships"]
        
        for entity in expected_entities:
            entity_dir = FACTS_DIR / entity
            if entity_dir.exists():
                for category in low_coverage_categories:
                    if not (entity_dir / f"{entity}-{category}.md").exists():
                        gaps.append({
                            "type": "missing_category",
                            "entity": entity,
                            "category": category,
                            "severity": "low"
                        })
        
        return {
            "gaps": gaps,
            "total_gaps": len(gaps),
            "by_severity": {
                "high": len([g for g in gaps if g["severity"] == "high"]),
                "medium": len([g for g in gaps if g["severity"] == "medium"]),
                "low": len([g for g in gaps if g["severity"] == "low"])
            }
        }
    
    def generate_gap_report(self) -> str:
        """Generate human-readable knowledge gap report."""
        gaps_data = self.detect_knowledge_gaps()
        gaps = gaps_data["gaps"]
        
        lines = [
            "# Knowledge Gap Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Total Gaps: {len(gaps)}",
            "",
            "## By Severity",
            f"- High: {gaps_data['by_severity']['high']}",
            f"- Medium: {gaps_data['by_severity']['medium']}",
            f"- Low: {gaps_data['by_severity']['low']}",
            "",
            "## Gaps",
            ""
        ]
        
        for gap in sorted(gaps, key=lambda g: {"high": 0, "medium": 1, "low": 2}[g["severity"]]):
            gap_type = gap["type"].replace("_", " ").title()
            severity = gap["severity"]
            
            if gap["type"] == "missing_entity":
                lines.append(f"- **[{severity.upper()}]** Missing entity: {gap['entity']}")
            elif gap["type"] == "outdated_facts":
                lines.append(f"- **[{severity.upper()}]** Outdated: {gap['entity']}/{gap['file']} ({gap['days_old']} days)")
            elif gap["type"] == "missing_category":
                lines.append(f"- **[{severity.upper()}]** Missing: {gap['entity']}/{gap['category']}")
        
        return "\n".join(lines)


def main():
    print("Knowledge Ingestion - Next-Gen Memory")
    print("=" * 50)
    
    ingestion = KnowledgeIngestion()
    
    import sys
    if len(sys.argv) < 2:
        print("Usage: knowledge_ingestion.py <command> [args]")
        print("\nCommands:")
        print("  ingest <research_path> [project]")
        print("  track-status <project> <json_data>")
        print("  detect-gaps")
        print("  gap-report")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "ingest":
        research_path = sys.argv[2]
        project = sys.argv[3] if len(sys.argv) > 3 else None
        result = ingestion.ingest_research(research_path, project)
        print(json.dumps(result, indent=2))
    
    elif command == "track-status":
        project = sys.argv[2]
        status_data = json.loads(sys.argv[3])
        result = ingestion.track_project_status(project, status_data)
        print(json.dumps(result, indent=2))
    
    elif command == "detect-gaps":
        result = ingestion.detect_knowledge_gaps()
        print(json.dumps(result, indent=2))
    
    elif command == "gap-report":
        report = ingestion.generate_gap_report()
        print(report)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
