#!/usr/bin/env python3
"""
Groundskeeper Weekly Summary Script

Reads the iteration log and produces a weekly summary markdown file.
"""

import os
import json
import glob
from datetime import datetime, timedelta
from collections import defaultdict

LOG_FILE = "/home/alansrobotlab/obsidian/memory/_pipeline/groundskeeper-log.jsonl"
OUTPUT_FILE = "/home/alansrobotlab/obsidian/memory/_pipeline/groundskeeper-weekly-summary.md"
QUEUE_FILE = "/home/alansrobotlab/obsidian/memory/_pipeline/groundskeeper-queue.json"

def parse_log_entries():
    """Read and parse the JSONL log file."""
    entries = []
    
    if not os.path.exists(LOG_FILE):
        return entries
    
    with open(LOG_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
    
    return entries

def get_week_entries(entries, weeks=1):
    """Filter entries from the last N weeks."""
    cutoff = datetime.now() - timedelta(weeks=weeks)
    return [e for e in entries if e.get('processed_at', '') > cutoff.isoformat()]

def compute_stats(entries):
    """Compute statistics from log entries."""
    stats = {
        'total': len(entries),
        'by_type': defaultdict(int),
        'by_status': defaultdict(int),
        'dates': []
    }
    
    for entry in entries:
        stats['by_type'][entry.get('type', 'unknown')] += 1
        stats['by_status'][entry.get('status', 'unknown')] += 1
        if entry.get('processed_at'):
            stats['dates'].append(entry['processed_at'][:10])
    
    return stats

def get_health_score_trend():
    """Get health score trend from queue snapshots."""
    if not os.path.exists(QUEUE_FILE):
        return None
    
    try:
        with open(QUEUE_FILE, 'r') as f:
            data = json.load(f)
        
        health = data.get('health_score', {})
        return {
            'latest': health.get('overall', 0),
            'computed_at': health.get('computed_at', ''),
            'dimensions': health.get('dimensions', {})
        }
    except (json.JSONDecodeError, KeyError):
        return None

def generate_markdown(entries, health_trend):
    """Generate the weekly summary markdown."""
    week_entries = get_week_entries(entries, weeks=1)
    stats = compute_stats(week_entries)
    
    lines = [
        "# Groundskeeper Loop — Weekly Summary",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "## Overview",
        "",
        f"**Items Processed This Week:** {len(week_entries)}",
        "",
    ]
    
    if stats['by_type']:
        lines.append("## Items by Type")
        lines.append("")
        for type_name, count in sorted(stats['by_type'].items()):
            lines.append(f"- **{type_name}**: {count}")
        lines.append("")
    
    if stats['by_status']:
        lines.append("## Items by Status")
        lines.append("")
        for status, count in sorted(stats['by_status'].items()):
            label = status.capitalize()
            lines.append(f"- **{label}**: {count}")
        lines.append("")
    
    if health_trend:
        lines.append("## Health Score Trend")
        lines.append("")
        lines.append(f"**Latest Score:** {health_trend['latest']}")
        lines.append(f"**Computed:** {health_trend['computed_at']}")
        lines.append("")
        
        if health_trend['dimensions']:
            lines.append("**Dimensions:**")
            lines.append("")
            for dim, data in health_trend['dimensions'].items():
                score = data.get('score', 0)
                count = data.get('count', 0)
                total = data.get('total', 0)
                lines.append(f"- **{dim}**: {score} ({count}/{total} issues)")
            lines.append("")
    
    # Success rate
    if len(week_entries) > 0:
        done_count = stats['by_status'].get('done', 0)
        skip_count = stats['by_status'].get('skipped', 0)
        success_rate = (done_count / len(week_entries)) * 100 if week_entries else 0
        
        lines.append("## Success Rate")
        lines.append("")
        lines.append(f"- **Completed:** {done_count}")
        lines.append(f"- **Skipped:** {skip_count}")
        lines.append(f"- **Success Rate:** {success_rate:.1f}%")
        lines.append("")
    
    # Recent activity
    if week_entries:
        lines.append("## Recent Activity")
        lines.append("")
        for entry in week_entries[-10:]:  # Last 10 entries
            item_id = entry.get('item_id', 'unknown')[:30]
            item_type = entry.get('type', 'unknown')
            status = entry.get('status', 'unknown')
            reason = entry.get('reason', '')[:50]
            processed = entry.get('processed_at', '')[:16]
            lines.append(f"- [{processed}] **{item_type}**: `{item_id}` → {status} ({reason})")
        lines.append("")
    
    lines.append("---")
    lines.append("*Generated by Groundskeeper Loop Weekly Summary Script*")
    
    return "\n".join(lines)

def main():
    """Generate weekly summary."""
    print("Groundskeeper Weekly Summary Starting...")
    
    # Parse log entries
    entries = parse_log_entries()
    print(f"  Found {len(entries)} log entries")
    
    # Get health trend
    health_trend = get_health_score_trend()
    if health_trend:
        print(f"  Health score: {health_trend['latest']}")
    
    # Generate markdown
    markdown = generate_markdown(entries, health_trend)
    
    # Write output
    with open(OUTPUT_FILE, 'w') as f:
        f.write(markdown)
    
    print(f"\nSummary complete!")
    print(f"  Output: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
