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
    
    # One day key for the whole invocation (backlog #853). The scoring stage used
    # to key its raw read and its scored write on `today` while the write stage
    # keyed on `date_str`, so `--date D --score --write` scored today into
    # `intel-<today>.jsonl` and then had `load_scored_items(D)` read
    # `intel-D.jsonl`: the day just scored never reached the vault, and a stale
    # day's scored file could be re-published in its place. A malformed `--date`
    # is refused here rather than becoming a filename two directories deep.
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if args.date:
        try:
            datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print(f"Invalid --date {args.date!r}: expected YYYY-MM-DD "
                  f"(e.g. {today})")
            sys.exit(2)
    date_str = args.date if args.date else today

    # Run scanners
    youtube_coverage = None  # None until the YouTube scanner reports its feeds
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
        youtube_items = []
        try:
            from .scanners.youtube_scanner import scan_youtube_channels
            youtube_items, youtube_coverage = scan_youtube_channels()
            print(f"YouTube scanner: {len(youtube_items)} items")
        except Exception as e:
            print(f"YouTube scanner error: {e}")
            youtube_items = []
        
        # Combine all scanned items
        all_items = github_items + youtube_items
        print(f"\nTotal scanned items: {len(all_items)}")

        if youtube_coverage is not None:
            print(f"YouTube feed coverage: {youtube_coverage.describe()}")
        
        # NOTE: each scanner saves its own items to the raw JSONL already
        # (github_scanner / youtube_scanner call state.save_raw_items internally).
        # Do NOT save the combined list here — that would double-write every item.
        # The scanners key that write on their OWN `today`, which `--date` does not
        # move, so this line prints today even when `--date D` asked for another
        # day. Naming the day here would print a path the run never wrote; the raw
        # write key is a scanner-side change this round did not make (#853 findings).
        print(f"Scanners saved raw items to: {state.get_raw_path(today)}")
    
    # Run scoring
    if run_all or args.score:
        print("\n--- Running Scoring Pipeline ---\n")
        
        # Load raw items for the day this invocation named — the same key the write
        # stage uses, which is the whole of #853.
        raw_items = state.load_raw_items(date_str)
        print(f"Loaded {len(raw_items)} raw items for {date_str}")

        if not raw_items:
            # A day with nothing to score gets said out loud. Writing an empty
            # `intel-<D>.jsonl` here would be worse than saying it: for a day whose
            # raw file has since rotated away, that file is the last record of what
            # was scored, and a following `--write` would then publish nothing.
            print(f"NO RAW ITEMS FOR {date_str}: no scored file written for it. A "
                  f"--write in this same run re-publishes whatever "
                  f"intel-{date_str}.jsonl already holds, or nothing if it does not "
                  f"exist (raw: {state.get_raw_path(date_str)})")
        else:
            # Score items
            from .models import FeedItem, ScoredItem
            scored = run_scoring_pipeline(raw_items, profile)
            print(f"Scored {len(scored)} items")
            
            # Save scored items
            from ._paths import FEEDS_DIR
            intel_path = FEEDS_DIR / f"intel-{date_str}.jsonl"
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
    
    # A run that could not reach most of the configured YouTube feeds must not
    # finish saying `Pipeline Complete` at exit 0: on 2026-09-10 the scanner lost
    # 55 of 64 feeds to upstream 404/500 and the run still reported success
    # (backlog #739). Checked after scoring/writing so the GitHub half of the run
    # still lands; only the success banner is withheld.
    if youtube_coverage is not None and youtube_coverage.degraded:
        print(f"\nYouTube stage DEGRADED: {youtube_coverage.describe()} "
              f"(threshold: at least half of the feeds attempted)")
        sys.exit(1)

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
