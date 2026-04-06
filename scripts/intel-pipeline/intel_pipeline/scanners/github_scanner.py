"""GitHub repository scanner for releases, commits, and issues."""

import requests
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
import time

from ..models import FeedItem
from .. import state
from ..profile import load_profile


# GitHub API base URL
GITHUB_API_URL = "https://api.github.com"
# State keys for GitHub scanner
GITHUB_STATE_KEY = "github_repos"


def load_github_repos_config() -> List[Dict[str, Any]]:
    """Load GitHub repos configuration."""
    import yaml
    from pathlib import Path
    
    config_path = Path.home() / "lloyd/scripts/intel-pipeline/config/github-repos.yml"
    
    if not config_path.exists():
        return []
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config.get("repos", [])


def get_repo_state_key(owner: str, repo: str, state_type: str) -> str:
    """Get state key for a repo."""
    return f"{owner}/{repo}:{state_type}"


def fetch_releases(owner: str, repo: str, last_tag: Optional[str] = None) -> List[Dict]:
    """Fetch releases for a repo."""
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/releases"
    params = {"per_page": 10}
    
    try:
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": "lloyd-intel-pipeline"},
            timeout=30
        )
        response.raise_for_status()
        releases = response.json()
        
        # Filter to only new releases since last check
        if last_tag:
            releases = [r for r in releases if r.get("tag_name") != last_tag]
        
        return releases[:5]  # Limit to 5 most recent new releases
    except requests.RequestException as e:
        print(f"Error fetching releases for {owner}/{repo}: {e}")
        return []


def fetch_commits(owner: str, repo: str, last_sha: Optional[str] = None) -> List[Dict]:
    """Fetch recent commits for a repo."""
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/commits"
    params = {"per_page": 10}
    
    try:
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": "lloyd-intel-pipeline"},
            timeout=30
        )
        response.raise_for_status()
        commits = response.json()
        
        # Filter to only commits since last check
        if last_sha:
            # Find position of last_sha and take only newer commits
            found = False
            filtered = []
            for commit in commits:
                if commit.get("sha") == last_sha:
                    found = True
                    break
                filtered.append(commit)
            commits = filtered
        
        return commits[:10]  # Limit to 10 commits
    except requests.RequestException as e:
        print(f"Error fetching commits for {owner}/{repo}: {e}")
        return []


def fetch_issues(owner: str, repo: str, last_ts: Optional[str] = None) -> List[Dict]:
    """Fetch recently updated issues/PRs for a repo."""
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/issues"
    params = {
        "state": "open",
        "sort": "updated",
        "per_page": 10
    }
    
    try:
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": "lloyd-intel-pipeline"},
            timeout=30
        )
        response.raise_for_status()
        issues = response.json()
        
        # Filter to only recently updated issues
        if last_ts:
            try:
                last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                filtered = []
                for issue in issues:
                    updated_at = issue.get("updated_at", "")
                    if updated_at:
                        try:
                            updated_dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                            if updated_dt > last_dt:
                                filtered.append(issue)
                        except ValueError:
                            continue
                issues = filtered
            except ValueError:
                pass
        
        return issues[:5]  # Limit to 5 issues
    except requests.RequestException as e:
        print(f"Error fetching issues for {owner}/{repo}: {e}")
        return []


def scan_github_repos() -> List[FeedItem]:
    """
    Scan configured GitHub repositories for new activity.
    
    Returns:
        List of FeedItem objects
    """
    repos = load_github_repos_config()
    if not repos:
        print("No GitHub repos configured")
        return []
    
    # Load current state
    current_state = state.load_state()
    if GITHUB_STATE_KEY not in current_state:
        current_state[GITHUB_STATE_KEY] = {}
    repo_state = current_state[GITHUB_STATE_KEY]
    
    all_items = []
    today = datetime.utcnow().strftime("%Y-%m-%d")
    
    for repo_config in repos:
        owner = repo_config.get("owner", "")
        repo = repo_config.get("repo", "")
        track = repo_config.get("track", ["releases", "commits", "issues"])
        
        if not owner or not repo:
            continue
        
        repo_key = f"{owner}/{repo}"
        print(f"\nScanning {repo_key}...")
        
        # Get stored state for this repo
        stored_state = repo_state.get(repo_key, {})
        last_tag = stored_state.get("last_release_tag")
        last_sha = stored_state.get("last_commit_sha")
        last_ts = stored_state.get("last_issue_check_ts")
        
        # Fetch releases
        if "releases" in track:
            releases = fetch_releases(owner, repo, last_tag)
            for release in releases:
                tag = release.get("tag_name", "")
                item_id = f"github:{owner}/{repo}:release:{tag}"
                
                # Skip if already seen
                if state.is_seen(item_id, current_state):
                    continue
                
                title = f"Release {tag}: {release.get('name', tag)}"
                summary = release.get("body", "")[:500] if release.get("body") else "No description"
                url = release.get("html_url", "")
                
                item = FeedItem(
                    id=item_id,
                    source="github",
                    title=title,
                    url=url,
                    summary=summary,
                    discovered_at=datetime.utcnow().isoformat() + "Z",
                    authors=[release.get("author", {}).get("login", "")] if release.get("author") else [],
                    source_tags=[f"release", tag]
                )
                all_items.append(item)
                state.mark_seen(item_id, current_state)
            
            # Update state with latest tag
            if releases:
                stored_state["last_release_tag"] = releases[0].get("tag_name", "")
        
        # Fetch commits
        if "commits" in track:
            commits = fetch_commits(owner, repo, last_sha)
            for commit in commits:
                sha = commit.get("sha", "")
                item_id = f"github:{owner}/{repo}:commit:{sha[:8]}"
                
                # Skip if already seen
                if state.is_seen(item_id, current_state):
                    continue
                
                commit_info = commit.get("commit", {})
                message = commit_info.get("message", "")
                first_line = message.split("\n")[0][:100] if message else "Unknown commit"
                summary = message[:500] if message else ""
                url = commit.get("html_url", "")
                
                item = FeedItem(
                    id=item_id,
                    source="github",
                    title=first_line,
                    url=url,
                    summary=summary,
                    discovered_at=datetime.utcnow().isoformat() + "Z",
                    authors=[commit_info.get("author", {}).get("name", "")] if commit_info.get("author") else [],
                    source_tags=["commit"]
                )
                all_items.append(item)
                state.mark_seen(item_id, current_state)
            
            # Update state with latest SHA
            if commits:
                stored_state["last_commit_sha"] = commits[0].get("sha", "")
        
        # Fetch issues
        if "issues" in track:
            issues = fetch_issues(owner, repo, last_ts)
            for issue in issues:
                number = issue.get("number", 0)
                item_id = f"github:{owner}/{repo}:issue:{number}"
                
                # Skip if already seen
                if state.is_seen(item_id, current_state):
                    continue
                
                title = issue.get("title", "")
                body = issue.get("body", "") or ""
                summary = body[:500] if body else "No description"
                url = issue.get("html_url", "")
                
                item = FeedItem(
                    id=item_id,
                    source="github",
                    title=title,
                    url=url,
                    summary=summary,
                    discovered_at=datetime.utcnow().isoformat() + "Z",
                    authors=[issue.get("user", {}).get("login", "")] if issue.get("user") else [],
                    source_tags=["issue"] if not issue.get("pull_request") else ["pr"]
                )
                all_items.append(item)
                state.mark_seen(item_id, current_state)
            
            # Update state with current timestamp
            stored_state["last_issue_check_ts"] = datetime.utcnow().isoformat() + "Z"
        
        # Save updated state for this repo
        repo_state[repo_key] = stored_state
    
    # Save updated state
    current_state[GITHUB_STATE_KEY] = repo_state
    state.save_state(current_state)
    
    # Save raw items
    if all_items:
        state.save_raw_items(all_items, today)
        print(f"\nSaved {len(all_items)} items to raw JSONL")
    
    return all_items


if __name__ == "__main__":
    # Example usage
    items = scan_github_repos()
    print(f"\n=== GitHub Scan Complete ===")
    print(f"Found {len(items)} new items")
    
    for item in items[:5]:
        print(f"\n[{item.source}] {item.title}")
        print(f"  URL: {item.url}")
        print(f"  Summary: {item.summary[:100]}...")
