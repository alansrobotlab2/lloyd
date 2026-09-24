"""GitHub repository scanner for releases, commits, and issues."""

import urllib.error
import urllib.request
import json
import os
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import time

from ..models import FeedItem
from ..body import clip_body
from .. import state
from ..profile import load_profile


# GitHub API base URL
GITHUB_API_URL = "https://api.github.com"
# State keys for GitHub scanner
GITHUB_STATE_KEY = "github_repos"

# Module-level so a test (or a relocation) can move it.
CONFIG_PATH = Path.home() / "lloyd/scripts/intel-pipeline/config/github-repos.yml"

# Unauthenticated GitHub API calls are capped at 60 requests/hour per source IP.
# This scanner makes ~3 calls per repo across 4 repos at runs_per_day: 3, which
# leaves no headroom for anything else on this box — and until 2026-09-11 a 403
# from that ceiling was swallowed by `except Exception`, printed as a fetch
# error, and yielded 0 items: indistinguishable in the run report from a repo
# with nothing new (backlog #570, defect 5).
UNAUTHENTICATED_HOURLY_LIMIT = 60


class GitHubRateLimitError(Exception):
    """GitHub refused the request for rate limiting (HTTP 403 / 429).

    A named exception so a quota cannot be mistaken for an idle repo, and so the
    fetch helpers' broad `except Exception` cannot quietly turn it into [].
    """

    def __init__(self, message: str, status: int, remaining: Optional[str] = None,
                 limit: Optional[str] = None, reset: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.remaining = remaining
        self.limit = limit
        self.reset = reset


def load_github_token(config_path: Optional[Path] = None) -> Optional[str]:
    """GitHub token: `GITHUB_TOKEN` in the environment, else `token:` in config.

    None when neither is set — the honest unauthenticated state, not an error.
    """
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        return token

    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        return None
    text = path.read_text()
    try:
        import yaml
        data = yaml.safe_load(text)
        if isinstance(data, dict):
            value = data.get("token")
            if value:
                return str(value).strip()
        return None
    except ImportError:
        m = re.search(r'^\s*token:\s*(.+?)\s*$', text, re.MULTILINE)
        return m.group(1).strip().strip('"').strip("'") if m else None


def _api_headers(token: Optional[str] = None) -> Dict[str, str]:
    """Request headers, with `Authorization` only when a token is configured."""
    headers = {
        "User-Agent": "lloyd-intel-pipeline",
        "Accept": "application/vnd.github+json",
    }
    token = load_github_token() if token is None else token
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _yaml_load_repos(path: str) -> List[Dict]:
    """Parse a YAML file containing a list of repo dicts."""
    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        if data and "repos" in data:
            return data["repos"]
        return data if isinstance(data, list) else []
    except ImportError:
        # Fallback to regex parser if PyYAML is not available
        with open(path) as f:
            text = f.read()
        repos = []
        for m in re.finditer(r'-\s*owner:\s*(.+?)(?:\n|$)', text):
            owner = m.group(1).strip().strip('"').strip("'")
            block_start = m.start()
            next_dash = re.search(r'\n\s+-\s+owner:', text[block_start + 1:])
            end = block_start + 1 + (next_dash.start() if next_dash else len(text) - block_start - 1)
            block = text[block_start:end]
            lines = block.strip().split('\n')
            repo = {"owner": owner}
            for line in lines[1:]:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                kv = line.split(':', 1)
                if len(kv) == 2:
                    k, v = kv[0].strip(), kv[1].strip()
                    repo[k] = _yaml_parse_val(v)
            repos.append(repo)
        return repos


def _yaml_parse_val(s: str) -> Any:
    s = s.strip().strip('"').strip("'")
    if s.startswith('[') and s.endswith(']'):
        inner = s[1:-1]
        if not inner.strip():
            return []
        return [v.strip() for v in inner.split(',')]
    if s.lower() == 'true':
        return True
    if s.lower() == 'false':
        return False
    try:
        return int(s)
    except ValueError:
        pass
    return s


def load_github_repos_config() -> List[Dict[str, Any]]:
    """Load GitHub repos configuration."""
    if not CONFIG_PATH.exists():
        return []
    return _yaml_load_repos(str(CONFIG_PATH))


def get_repo_state_key(owner: str, repo: str, state_type: str) -> str:
    """Get state key for a repo."""
    return f"{owner}/{repo}:{state_type}"


def _http_get(url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None, timeout: int = 30) -> Any:
    """HTTP GET using stdlib urllib. Returns parsed JSON or raises.

    A 403/429 becomes GitHubRateLimitError with the X-RateLimit headers attached,
    because "quota exhausted" and "this repo had nothing new" have to be tellable
    apart in the run report.
    """
    from urllib.parse import urlencode
    if params:
        url = url + "?" + urlencode(params)
    req = urllib.request.Request(url, headers=headers if headers is not None else _api_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (403, 429):
            hdrs = e.headers or {}
            raise GitHubRateLimitError(
                f"HTTP {e.code} from {url} — X-RateLimit-Remaining="
                f"{hdrs.get('X-RateLimit-Remaining', '?')} of "
                f"{hdrs.get('X-RateLimit-Limit', '?')}",
                status=e.code,
                remaining=hdrs.get("X-RateLimit-Remaining"),
                limit=hdrs.get("X-RateLimit-Limit"),
                reset=hdrs.get("X-RateLimit-Reset"),
            ) from e
        raise Exception(str(e))
    except Exception as e:
        raise Exception(str(e))


def fetch_releases(owner: str, repo: str, last_tag: Optional[str] = None) -> List[Dict]:
    """Fetch releases for a repo."""
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/releases"

    try:
        releases = _http_get(url, params={"per_page": 10}, headers=_api_headers(), timeout=30)

        # Filter to only new releases since last check
        if last_tag:
            releases = [r for r in releases if r.get("tag_name") != last_tag]

        return releases[:5]  # Limit to 5 most recent new releases
    except GitHubRateLimitError:
        raise  # never fold a quota into "nothing new"
    except Exception as e:
        print(f"Error fetching releases for {owner}/{repo}: {e}")
        return []


def fetch_commits(owner: str, repo: str, last_sha: Optional[str] = None) -> List[Dict]:
    """Fetch recent commits for a repo."""
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/commits"

    try:
        commits = _http_get(url, params={"per_page": 10}, headers=_api_headers(), timeout=30)

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
    except GitHubRateLimitError:
        raise
    except Exception as e:
        print(f"Error fetching commits for {owner}/{repo}: {e}")
        return []


def fetch_issues(owner: str, repo: str, last_ts: Optional[str] = None) -> List[Dict]:
    """Fetch recently updated issues/PRs for a repo."""
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/issues"

    try:
        issues = _http_get(url, params={"state": "open", "sort": "updated", "per_page": 10}, headers=_api_headers(), timeout=30)

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
    except GitHubRateLimitError:
        raise
    except Exception as e:
        print(f"Error fetching issues for {owner}/{repo}: {e}")
        return []


def _fetch_track(track: str, fn, *args, rate_limited: Dict[str, str], repo_key: str) -> List[Dict]:
    """Run one fetch, recording a quota hit against this repo instead of returning [].

    The point is that the caller still gets a list and the scan still finishes,
    but a 403 is named — and the caller refuses to advance that repo's stored
    state, so items missed during the quota window are still reported later.
    """
    try:
        return fn(*args)
    except GitHubRateLimitError as e:
        rate_limited[repo_key] = f"{track}: {e}"
        print(f"  GITHUB_RATE_LIMIT on {repo_key} ({track}): HTTP {e.status} "
              f"[{e.remaining} of {e.limit} requests left]")
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

    if load_github_token() is None:
        print(f"No GitHub token configured (GITHUB_TOKEN, or `token:` in {CONFIG_PATH}) "
              f"— running against the unauthenticated {UNAUTHENTICATED_HOURLY_LIMIT} "
              f"requests/hour ceiling")

    # Load current state
    current_state = state.load_state()
    if GITHUB_STATE_KEY not in current_state:
        current_state[GITHUB_STATE_KEY] = {}
    repo_state = current_state[GITHUB_STATE_KEY]

    all_items = []
    today = datetime.utcnow().strftime("%Y-%m-%d")
    # repo_key -> "track: reason", so the run report can say which repos were
    # rate limited rather than reporting a quiet day.
    rate_limited: Dict[str, str] = {}

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
            releases = _fetch_track("releases", fetch_releases, owner, repo, last_tag,
                                    rate_limited=rate_limited, repo_key=repo_key)
            for release in releases:
                tag = release.get("tag_name", "")
                item_id = f"github:{owner}/{repo}:release:{tag}"

                # Skip if already seen
                if state.is_seen(item_id, current_state):
                    continue

                title = f"Release {tag}: {release.get('name', tag)}"
                summary = clip_body(release.get("body")) or "No description"
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
            commits = _fetch_track("commits", fetch_commits, owner, repo, last_sha,
                                   rate_limited=rate_limited, repo_key=repo_key)
            for commit in commits:
                sha = commit.get("sha", "")
                item_id = f"github:{owner}/{repo}:commit:{sha[:8]}"

                # Skip if already seen
                if state.is_seen(item_id, current_state):
                    continue

                commit_info = commit.get("commit", {})
                message = commit_info.get("message", "")
                first_line = message.split("\n")[0][:100] if message else "Unknown commit"
                summary = clip_body(message)
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
            issues = _fetch_track("issues", fetch_issues, owner, repo, last_ts,
                                  rate_limited=rate_limited, repo_key=repo_key)
            for issue in issues:
                number = issue.get("number", 0)
                item_id = f"github:{owner}/{repo}:issue:{number}"

                # Skip if already seen
                if state.is_seen(item_id, current_state):
                    continue

                title = issue.get("title", "")
                body = issue.get("body", "") or ""
                summary = clip_body(body) or "No description"
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

            # Update state with current timestamp — only if we actually read it.
            # Stamping a repo we were rate limited on would silently skip every
            # issue updated during the quota window.
            if repo_key not in rate_limited:
                stored_state["last_issue_check_ts"] = datetime.utcnow().isoformat() + "Z"

        # Save updated state for this repo, unless a quota hit means what we
        # fetched was not the whole picture.
        if repo_key not in rate_limited:
            repo_state[repo_key] = stored_state

    # Save updated state
    current_state[GITHUB_STATE_KEY] = repo_state
    state.save_state(current_state)

    if rate_limited:
        # The reason this block exists: before 2026-09-11 a quota 403 printed a
        # per-repo fetch error and yielded 0 items, and the run still ended with
        # `Pipeline Complete` — so a rate limit and a quiet day were the same
        # report (backlog #570, defect 5).
        print(f"\n=== GITHUB_RATE_LIMIT: {len(rate_limited)} of {len(repos)} repos ===")
        for repo_key, detail in rate_limited.items():
            print(f"GITHUB_RATE_LIMIT {repo_key} — {detail}")
        print("GITHUB_RATE_LIMIT: 0 items from the repos above is a quota failure, "
              "NOT a quiet day. Set GITHUB_TOKEN or `token:` in "
              f"{CONFIG_PATH} to raise the ceiling from "
              f"{UNAUTHENTICATED_HOURLY_LIMIT} to 5000 requests/hour.")

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
