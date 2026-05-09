"""Main entry point for the intelligence pipeline."""

import sys
import argparse
from datetime import datetime
from pathlib import Path

from .profile import load_profile, PROFILE_FILE
from .scoring import run_scoring_pipeline
from . import state


def main():
    """Run the intelligence pipeline."""
    parser = argparse.ArgumentParser(description="Intelligence Pipeline")
    parser.add_argument("--scan", action="store_true", help="Run scanners only")
    parser.add_argument("--score", action="store_true", help="Run scoring only")
    parser.add_argument("--write", action="store_true", help="Run vault writer only")
    parser.add_argument("--date", type=str, default=None, help="Date string (YYYY-MM-DD) for scoring/writing")
    
    args = parser.parse_args()
    
    # Default: run everything if no flags specified
    run_all = not (args.scan or args.score or args.write)
    
    print("=== Intelligence Pipeline ===\n")
    
    # Load profile
    profile = load_profile()
    print(f"Loaded profile from: {PROFILE_FILE}")
    
    today = datetime.utcnow().strftime("%Y-%m-%d")
    date_str = args.date if args.date else today
    
    # Run scanners
    if run_all or args.scan:
        print("\n--- Running Scanners ---\n")
        
        # Import and run GitHub scanner
        try:
            from .scanners.github_scanner import scan_github_repos
            github_items = scan_github_repos()
            print(f"GitHub scanner: {len(github_items)} items")
        except Exception as e:
            print(f"GitHub scanner error: {e}")
            github_items = []
        
        # Import and run YouTube scanner
        try:
            from .scanners.youtube_scanner import scan_youtube_channels
            youtube_items = scan_youtube_channels()
            print(f"YouTube scanner: {len(youtube_items)} items")
        except Exception as e:
            print(f"YouTube scanner error: {e}")
            youtube_items = []
        
        # Combine all scanned items
        all_items = github_items + youtube_items
        print(f"\nTotal scanned items: {len(all_items)}")
        
        # Save raw items
        if all_items:
            raw_path = state.get_raw_path(today)
            state.save_raw_items(all_items, today)
            print(f"Saved raw items to: {raw_path}")
    
    # Run scoring
    if run_all or args.score:
        print("\n--- Running Scoring Pipeline ---\n")
        
        # Load raw items for today
        raw_items = state.load_raw_items(today)
        print(f"Loaded {len(raw_items)} raw items")
        
        if raw_items:
            # Score items
            from .models import FeedItem, ScoredItem
            scored = run_scoring_pipeline(raw_items, profile)
            print(f"Scored {len(scored)} items")
            
            # Save scored items
            from ._paths import FEEDS_DIR
            intel_path = FEEDS_DIR / f"intel-{today}.jsonl"
            with open(intel_path, "w") as f:
                for item in scored:
                    f.write(item.to_json() + "\n")
            print(f"Saved scored items to: {intel_path}")
            
            # Display top results
            print("\n=== Top Results ===")
            for item in sorted(scored, key=lambda x: x.relevance, reverse=True)[:5]:
                print(f"\n[{item.urgency.upper()}] {item.title}")
                print(f"  Source: {item.source}")
                print(f"  Relevance: {item.relevance}/10")
                print(f"  Category: {item.category}")
                print(f"  Why: {item.why}")
                print(f"  URL: {item.url}")
    
    # Run vault writer
    if run_all or args.write:
        print("\n--- Running Vault Writer ---\n")
        
        from .vault_writer import write_all_to_vault
        count = write_all_to_vault(date_str)
        print(f"Vault writer: {count} items written")
    
    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
