"""The two client trees must not describe a middleware that no longer exists.

Both `chrome-extension/src/background/lloyd-client.ts` and `web/src/api.ts` told
a reader that loopback works because it "bypasses the mTLS middleware at
server.py:76-113". Client-certificate auth was dropped from this backend on
2026-06-14 (iOS Chrome cannot present a keychain identity), and the peer-address
gate that replaced it landed on 2026-09-20: `ApiPeerGate` calls
`_is_trusted_peer` on `scope["client"]` and refuses header-based evidence
outright. So the comment named a layer that is not there, at a line range that
holds something else, in the exact place a person editing a client looks.

The corpus is `web/src/api.ts` alone. The service-worker file carried the same
sentence verbatim and is in the same shape, but `chrome-extension/**` is not in
`scripts/automod/spec.py`'s `ALLOWED_GLOBS`, so no round may write it — #1722
carries that comment edit and the header node that goes with it. Add the path
back to `CORPUS` when it lands: a corpus narrowed by a scope rule is the one
thing here that is not about the comments.

These tests pin the corrected wording and then forbid the stale shape, and the
corpus is asserted tracked and non-empty before anything is searched, with a
repo-wide positive control beside it, so a 0-hit result can only mean the
comments are clean — never that the grep searched nothing.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

WEB_API = "web/src/api.ts"

#: The files a client-side editor reads, whose comments must describe the gate
#: that actually answers them. `git grep -n mTLS` over exactly these paths is the
#: check the item recorded; see the module docstring for why one of the two is
#: missing today.
CORPUS = (WEB_API,)

API_BASE_ANCHOR = "const API_BASE"

#: Case-insensitive: the point is the mechanism being claimed, not one spelling.
STALE_MECHANISM = re.compile(r"mtls", re.IGNORECASE)
#: A `server.py:<digits>` citation is what rotted — the symbol is the durable
#: reference, the line number is the one that moves under an unrelated edit.
LINE_CITATION = re.compile(r"server\.py\s*:\s*\d")


def _tracked_text(rel_path: str) -> str:
    """Read one corpus file, asserting first that it is really in the corpus.

    A missing or empty file would make every absence-assertion below pass for
    the wrong reason, so the guard runs before any searching: the path must be
    tracked in this worktree and must have bytes in it.
    """
    path = REPO / rel_path
    assert path.is_file(), f"{rel_path} is not a file — the corpus moved"
    assert path.stat().st_size > 0, f"{rel_path} is empty — nothing to grep"
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--error-unmatch", rel_path],
        capture_output=True, text=True,
    )
    assert tracked.returncode == 0, f"{rel_path} is not tracked by git: {tracked.stderr.strip()}"
    return path.read_text(encoding="utf-8")


def _comment_block_above(text: str, anchor: str) -> str:
    """The contiguous `//` comment lines sitting directly above `anchor`."""
    lines = text.splitlines()
    idx = next((i for i, ln in enumerate(lines) if anchor in ln), None)
    assert idx is not None, f"{anchor!r} vanished from the corpus"
    start = idx
    while start > 0 and lines[start - 1].lstrip().startswith("//"):
        start -= 1
    block = "\n".join(lines[start:idx])
    assert block.strip(), f"no comment block above {anchor!r} — the anchor moved"
    return block


def test_web_api_base_comment_states_the_peer_rule_for_loopback():
    """Clause 2: the comment above `API_BASE` states the same peer-address rule
    for the loopback case, and no longer claims a Vite-injected client-cert
    header is what the main web app relies on — `web/vite.config.ts` stopped
    requesting a client cert in 5e1351f3, so no such header reaches the backend
    from a browser."""
    block = _comment_block_above(_tracked_text(WEB_API), API_BASE_ANCHOR).lower()
    assert "_is_trusted_peer" in block, "the comment must name the real gate"
    assert "loopback" in block, "the comment must cover the loopback case"
    assert "peer address" in block, "the comment must state the rule is the peer address"
    assert "inject" not in block, "the comment must not claim the Vite proxy injects headers"
    assert "cert header" not in block, "the comment must not rely on a forwarded cert header"


def test_web_api_comment_names_the_gate_and_that_loopback_needs_no_cert():
    """Clause 1: the two facts a client editor needs beside the URL — which
    symbol answers a loopback call, and that no certificate is involved. The
    original clause put this on the service-worker header as well; that file is
    outside the loop's writable set, so #1722 carries that half.
    """
    block = _comment_block_above(_tracked_text(WEB_API), API_BASE_ANCHOR).lower()
    assert "server.py" in block, "the comment must name the module the gate lives in"
    assert "_is_trusted_peer" in block, "the comment must name the gate that answers loopback"
    assert "no client certificate" in block, "the comment must say loopback needs no cert"


def test_web_api_file_carries_no_stale_mechanism_or_line_citation():
    """Clause 3 (the `web/src/api.ts` half): the grep that catches the next stale
    rewrite. Neither the mechanism string nor a `server.py:<line>` range may
    appear anywhere in the file — including the section header over the
    still-live `/api/system/*` endpoints, which used to read
    `LAN access / mTLS`. The corpus is proven non-empty first and a positive
    control proves the pattern still matches real files elsewhere in the repo, so
    a 0-hit result is not a grep that searched nothing."""
    for rel_path in CORPUS:
        text = _tracked_text(rel_path)
        assert not STALE_MECHANISM.search(text), f"{rel_path} still mentions the dropped mechanism"
        assert not LINE_CITATION.search(text), f"{rel_path} still cites a server.py line range"

    listing = subprocess.run(
        ["git", "-C", str(REPO), "grep", "-i", "-l", "mtls"],
        capture_output=True, text=True,
    )
    assert listing.returncode == 0, f"`git grep -i -l mtls` failed: {listing.stderr.strip()}"
    hits = {ln.strip() for ln in listing.stdout.splitlines() if ln.strip()}
    outside = hits - set(CORPUS) - {f"tests/{Path(__file__).name}"}
    assert len(outside) >= 3, (
        f"positive control: only {sorted(outside)} mention the mechanism repo-wide — "
        "the pattern is matching too little to trust the 0 hits above"
    )
    assert any(name == "server.py" or name.startswith("architecture/") for name in outside), (
        "the live description of the drop (server.py / architecture/) must remain findable"
    )
