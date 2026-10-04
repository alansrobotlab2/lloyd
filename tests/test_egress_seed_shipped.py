"""The shipped egress seed (#2123): telemetry on, enforcement off, a reviewed list.

The allow-list was empty until 2026-10-04, so `enforce` could not be armed without
refusing every search and fetch the unattended fleet makes. The seed is the hosts
those sessions reached three or more times in 14 days; what is deliberately left
off it — deep-research's long tail, and one unexplained file-proxy host — is in
`~/obsidian/knowledge/harness/egress-destinations.md`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from agent_mcp import egress

ROOT = Path(__file__).resolve().parents[1]
POLICY = yaml.safe_load((ROOT / "config.yaml").read_text())["harness"]["egress_policy"]


def test_the_shipped_policy_records_and_does_not_enforce():
    assert POLICY["telemetry"] is True and POLICY["enforce"] is False


def test_every_shipped_entry_validates_and_says_why():
    entries = egress.allow_entries(POLICY["allow"])
    assert len(entries) == len(POLICY["allow"]), "an entry was dropped as malformed"
    assert all(e["reason"] for e in entries)
    # All scope-less and unexpiring today; the number #628 says must be reported.
    assert egress.count_permanent_entries(POLICY["allow"]) == len(entries)


def test_the_seed_covers_the_subdomains_it_claims_and_not_a_sibling_domain():
    entries = egress.allow_entries(POLICY["allow"])
    now = datetime.now(timezone.utc)

    def covered(host: str) -> bool:
        return any(egress._covers(e, host=host, scope="worker:deep-research", at=now)
                   for e in entries)

    for host in ("html.duckduckgo.com", "export.arxiv.org", "api.github.com",
                 "raw.githubusercontent.com", "en.wikipedia.org", "www.youtube.com"):
        assert covered(host), host
    assert not covered("githubusercontent.com.example.org")


def test_an_unexplained_storage_host_is_not_seeded_by_a_parent_domain():
    """`routify-file-proxy-sg.oss-ap-southeast-1.aliyuncs.com` was fetched 31 times
    by deep-research with no caller naming it; it stays off the list until someone
    has read what was fetched, and no entry may cover it by suffix."""
    entries = egress.allow_entries(POLICY["allow"])
    now = datetime.now(timezone.utc)
    host = "routify-file-proxy-sg.oss-ap-southeast-1.aliyuncs.com"
    assert not any(egress._covers(e, host=host, scope="", at=now) for e in entries)
