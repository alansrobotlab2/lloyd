"""Client code and the architecture docs must not describe a certificate
exemption that no longer exists.

Both `chrome-extension/src/background/lloyd-client.ts` and `web/src/api.ts` told
a reader that loopback works because it "bypasses the mTLS middleware at
server.py:76-113". Client-certificate auth was dropped from this backend on
2026-06-14 (iOS Chrome cannot present a keychain identity), and the peer-address
gate that replaced it landed on 2026-09-20: `ApiPeerGate` calls
`_is_trusted_peer` on `scope["client"]` and refuses header-based evidence
outright. So the comment named a layer that is not there, at a line range that
holds something else, in the exact place a person editing a client looks.

The corpus is both client files, named verbatim. The service-worker client's
header was corrected by hand (#1722) — `chrome-extension/**` stays outside the
loop's writable set, since there is no JS/TS test runner a round could be graded
on — and its header node below pins the peer-address wording there too.

The corpus widened to two architecture docs (#1759): `architecture/browser-side-panel.md`
carried the same false mechanism ("`server.py` skips mTLS for loopback") in the
section a reader goes to for "why is plain HTTP to :8080 allowed", and
`architecture/authority-surfaces.md` — the doc whose own last line warns that
"two docs describing one guard in full is how one gets corrected and the other
does not" — carried it too. Those two are a SEPARATE corpus with a different ban,
because the true history must stay readable in them: `infrastructure.md`,
`mission-control.md` and the Loopback row all say "mTLS was dropped on
2026-06-14", which is prose about a dead mechanism and has to survive. What is
forbidden there is the claim shape — an exemption for loopback, or an origin that
still has to present a certificate — not the word.

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
SW_CLIENT = "chrome-extension/src/background/lloyd-client.ts"

#: The files a client-side editor reads, whose comments must describe the gate
#: that actually answers them. `git grep -n mTLS` over exactly these paths is the
#: check the item recorded. Named verbatim, never a directory glob: a glob over
#: `chrome-extension/` would drag in prose this guard was not written for.
CORPUS = (WEB_API, SW_CLIENT)

API_BASE_ANCHOR = "const API_BASE"

#: Case-insensitive: the point is the mechanism being claimed, not one spelling.
STALE_MECHANISM = re.compile(r"mtls", re.IGNORECASE)
#: A `server.py:<digits>` or `messages.py:<digits>` citation is what rotted —
#: the symbol is the durable reference, the line number is the one that moves
#: under an unrelated edit. `messages.py` joined in #1766: both client comments
#: that justify awaiting the kickoff POST sent a reader to line ranges that had
#: drifted onto the turn's error path and an unrelated helper.
LINE_CITATION = re.compile(r"(?:server|messages)\.py\s*:\s*\d")
#: Only the `messages.py` half, for the doc corpus below: `architecture/` keeps
#: some correct line citations, and this guard is about one rotted range.
MESSAGES_CITATION = re.compile(r"messages\.py\s*:\s*\d")

#: The sentences #1766 found, verbatim as they shipped, plus the `server.py`
#: one this guard was written for — kept so `LINE_CITATION` is proven to fire on
#: real wording rather than on a paraphrase. The first is
#: `chrome-extension/src/background/service-worker.ts` and the second
#: `chrome-extension/src/background/lloyd-client.ts`, both at `a7a22bfa`; the
#: third is `architecture/automod.md`'s `_turn_budget` sentence at the same
#: commit.
LINE_CITATION_SAMPLES = (
    "`/api/message/stream` enqueues the turn before it hands back the "
    "StreamingResponse (app/routers/messages.py:1827, then :1837), so once this "
    "POST has answered the backend already counts the session as active",
    "Lloyd's /api/message/stream explicitly survives client disconnect "
    "(messages.py:10, 924-935) — the consumer keeps running on the server even "
    "though we never read the stream.",
    "`messages._turn_budget` (`app/routers/messages.py:132`) clamps whatever a "
    "worker asks for to `agent.max_turns_ceiling` (120).",
)
SERVER_CITATION_SAMPLE = (
    "All calls hit http://127.0.0.1:8080 directly — the FastAPI mTLS middleware "
    "at server.py:76-113 skips loopback, so no client cert is required."
)

AUTOMOD_DOC = "architecture/automod.md"

BROWSER_PANEL = "architecture/browser-side-panel.md"
AUTHORITY = "architecture/authority-surfaces.md"

#: The two docs that restate the network boundary (#1759). Separate from
#: `CORPUS` on purpose: the correct drop history has to stay findable in them, so
#: what is banned here is the claim, not the mechanism's name.
#: `authority-surfaces.md` owns the rule and `browser-side-panel.md` answers "why
#: is plain HTTP to :8080 allowed" — which is why the copy in the panel doc is
#: load-bearing and why it may not copy the list.
ARCH_DOCS = (BROWSER_PANEL, AUTHORITY)

#: "server.py skips mTLS for loopback": an exemption for one peer from a check
#: that asks nobody anything. The mechanism died 2026-06-14; `ApiPeerGate` has
#: decided on `scope["client"]` since 2026-09-20.
STALE_EXEMPTION = re.compile(
    r"skip\w*\s+mTLS|mTLS[^.]{0,60}skip\w*|bypass\w*[^.]{0,60}\bcert", re.IGNORECASE)
#: The surviving half of the same claim: that some other origin is still required
#: to produce a certificate, and so that the certless panel call is an exception.
STALE_CERT_REQUIRED = re.compile(
    r"must\s+present\s+(?:one|a\s+cert|a\s+certificate|a\s+client\s+cert)", re.IGNORECASE)
#: A CIDR literal. The trusted-network list has one home
#: (`server.trusted_networks`, described once in `authority-surfaces.md`); a
#: second copy in prose is the thing that went stale here.
CIDR = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}/\d{1,2}\b")

#: The two sentences as this item found them, kept so the patterns above are
#: proven to fire on the wording that actually shipped rather than on a
#: paraphrase. The first is `browser-side-panel.md` at `07f90d8b`
#: (`git grep -n "skips mTLS" architecture/`); the second is the older Loopback
#: bullet of `authority-surfaces.md`, quoted from the item.
STALE_SAMPLES = (
    "`server.py` skips mTLS for loopback, so loopback is the only origin that "
    "reaches the API without a cert, and an extension's service worker cannot "
    "present one.",
    "so `chrome-extension`'s service worker can call `http://127.0.0.1:8080` "
    "with no client certificate ([[browser-side-panel]]) while every other "
    "origin must present one.",
)


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


def _section(text: str, heading: str, stops: tuple[str, ...] = ("\n## ",)) -> str:
    """One block of a markdown doc, from `heading` to the earliest of `stops`.

    Asserted rather than returned empty — a renamed or deleted heading would
    otherwise hand the caller a blank string and a test that passes on it. The
    caller names the stop so a bullet-level claim is not satisfied by prose in
    the bullets or paragraphs sitting after it.
    """
    start = text.find(heading)
    assert start >= 0, f"{heading!r} vanished from the doc — the heading was renamed"
    rest = text[start:]
    ends = [rest.find(stop, len(heading)) for stop in stops]
    ends = [e for e in ends if e >= 0]
    return rest[:min(ends)] if ends else rest


def _leading_comment_block(text: str, must_mention: str) -> str:
    """A file's opening `//` block — the header a reader sees first."""
    lines = text.splitlines()
    end = 0
    while end < len(lines) and lines[end].lstrip().startswith("//"):
        end += 1
    block = "\n".join(lines[:end])
    assert block.strip(), "the file has no opening comment block — the header moved"
    assert must_mention in block.lower(), (
        f"the opening block does not mention {must_mention!r} — wrong block, not the header"
    )
    return block


def test_sw_client_header_names_the_peer_address_gate_and_no_cert():
    """#1722 clause 1: the service-worker client's header must say loopback is
    accepted by the peer-address gate, name the trusted set, and say no client
    certificate is required — which is the true reason a loopback call works
    without a cert."""
    block = _leading_comment_block(_tracked_text(SW_CLIENT), "127.0.0.1:8080").lower()
    assert "_is_trusted_peer" in block, (
        "the header must name the gate that actually answers loopback"
    )
    assert "peer address" in block, "the header must state the rule is the peer address"
    assert "trusted" in block, "the header must name the trusted-network set"
    assert "loopback" in block and "server.trusted_networks" in block, (
        "the header must say what the trusted set is: loopback plus the configured networks")
    assert "no client certificate" in block, "the header must say loopback needs no cert"


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
    service-worker header carries the same two facts, pinned by
    `test_sw_client_header_names_the_peer_address_gate_and_no_cert`.
    """
    block = _comment_block_above(_tracked_text(WEB_API), API_BASE_ANCHOR).lower()
    assert "server.py" in block, "the comment must name the module the gate lives in"
    assert "_is_trusted_peer" in block, "the comment must name the gate that answers loopback"
    assert "no client certificate" in block, "the comment must say loopback needs no cert"


def test_client_files_carry_no_stale_mechanism_or_line_citation():
    """Clause 3 (both client files, #1722): the grep that catches the next stale
    rewrite. Neither the mechanism string nor a `server.py:<line>` range may
    appear anywhere in either file — including the section header over the
    still-live `/api/system/*` endpoints, which used to read
    `LAN access / mTLS`. The corpus is proven non-empty first and a positive
    control proves the pattern still matches real files elsewhere in the repo, so
    a 0-hit result is not a grep that searched nothing."""
    # Every corpus file proven tracked and non-empty, and the pattern proven
    # live repo-wide, before a single absence is asserted.
    texts = {rel_path: _tracked_text(rel_path) for rel_path in CORPUS}
    assert set(texts) == {WEB_API, SW_CLIENT}, f"the corpus drifted: {sorted(texts)}"

    listing = subprocess.run(
        ["git", "-C", str(REPO), "grep", "-i", "-l", "mtls"],
        capture_output=True, text=True,
    )
    assert listing.returncode == 0, f"`git grep -i -l mtls` failed: {listing.stderr.strip()}"
    hits = {ln.strip() for ln in listing.stdout.splitlines() if ln.strip()}
    outside = hits - set(CORPUS) - {f"tests/{Path(__file__).name}"}
    assert len(outside) >= 3, (
        f"positive control: only {sorted(outside)} mention the mechanism repo-wide — "
        "the pattern is matching too little to trust the 0 hits below"
    )
    assert any(name == "server.py" or name.startswith("architecture/") for name in outside), (
        "the live description of the drop (server.py / architecture/) must remain findable"
    )

    for rel_path, text in texts.items():
        assert not STALE_MECHANISM.search(text), f"{rel_path} still mentions the dropped mechanism"
        cited = LINE_CITATION.search(text)
        assert not cited, (
            f"{rel_path} still cites a python line range ({cited.group(0)!r}) — "
            "cite the symbol")


def test_the_line_citation_pattern_fires_on_the_wording_that_shipped():
    """#1766 clause 2: the widened pattern matches every shipped sentence
    verbatim, and still matches the `server.py` one it was written for — so the
    absence assertions over the corpus cannot pass by matching nothing."""
    assert len(LINE_CITATION_SAMPLES) == 3, "the fixtures are the three shipped sentences"
    for sample in LINE_CITATION_SAMPLES:
        assert LINE_CITATION.search(sample), f"LINE_CITATION went vacuous on: {sample!r}"
        assert MESSAGES_CITATION.search(sample), f"MESSAGES_CITATION went vacuous on: {sample!r}"
    assert LINE_CITATION.search(SERVER_CITATION_SAMPLE), (
        "widening to messages.py dropped the server.py case")
    assert not MESSAGES_CITATION.search(SERVER_CITATION_SAMPLE), (
        "the doc-corpus pattern is meant to be the messages.py half alone")


def test_automod_doc_names_turn_budget_by_symbol_not_a_line_range():
    """#1766 clause 3: `architecture/automod.md` refers to
    `messages._turn_budget` by symbol, with no `messages.py:<digits>` range —
    its `:132` pointed at a `return 0.0` four lines above the def. The doc is
    proven tracked and non-empty, the symbol present, and the pattern live on
    real architecture docs before the absence is asserted."""
    text = _tracked_text(AUTOMOD_DOC)
    assert "messages._turn_budget" in text, (
        f"{AUTOMOD_DOC} no longer names `messages._turn_budget` — the clamp moved or "
        "the sentence was deleted, and the absence below would be vacuous")

    listing = subprocess.run(
        ["git", "-C", str(REPO), "grep", "-l", "-E", r"messages\.py[[:space:]]*:[[:space:]]*[0-9]",
         "--", "architecture/"],
        capture_output=True, text=True,
    )
    assert listing.returncode == 0, (
        f"positive control: no architecture doc carries a messages.py line citation at "
        f"all, so the pattern cannot be shown live: {listing.stderr.strip()}")
    carriers = {ln.strip() for ln in listing.stdout.splitlines() if ln.strip()}
    assert carriers - {AUTOMOD_DOC}, "the positive control found only the doc under test"
    for rel_path in sorted(carriers - {AUTOMOD_DOC}):
        assert MESSAGES_CITATION.search(_tracked_text(rel_path)), (
            f"git grep found a citation in {rel_path} that MESSAGES_CITATION misses")

    cited = MESSAGES_CITATION.search(text)
    assert not cited, (
        f"{AUTOMOD_DOC} cites a messages.py line range ({cited.group(0)!r}); refer to "
        "the symbol")


# ── #1759: the same stale claim, in the docs that describe the guard ────────── #


def test_the_architecture_doc_bans_fire_on_the_wording_that_shipped():
    """Negative control for the two claim patterns: they must match the sentences
    this item found on the tree, verbatim. Without this the 0-hit assertions
    below could be satisfied by a pattern that matches nothing, which is the
    failure mode the doc drift itself is an instance of."""
    assert len(STALE_SAMPLES) == 2, "the fixtures are the two shipped sentences; do not pad them"
    assert STALE_EXEMPTION.search(STALE_SAMPLES[0]), (
        "`skips mTLS for loopback` no longer trips the exemption ban — the ban went vacuous")
    assert STALE_CERT_REQUIRED.search(STALE_SAMPLES[1]), (
        "`every other origin must present one` no longer trips the requirement ban "
        "— the ban went vacuous")
    assert STALE_CERT_REQUIRED.search(STALE_SAMPLES[0]) or STALE_EXEMPTION.search(STALE_SAMPLES[0]), (
        "the panel-doc sentence must trip at least one ban")


def test_no_architecture_doc_claims_a_certificate_skip_or_a_cert_requirement():
    """Clause 3 (the drift ban), and the acceptance check written down: no doc
    under `architecture/` may claim that client-certificate auth is skipped or
    bypassed for a peer, or that some origin still has to present a certificate.

    The positive control is the half that makes a 0-hit meaningful: the mechanism
    name must STILL be findable under `architecture/`, because
    `infrastructure.md` and `mission-control.md` carry the true history ("mTLS
    was dropped on 2026-06-14") and a later sweep that 'fixes' those into silence
    would turn every assertion here green and the docs unreadable."""
    for rel_path in ARCH_DOCS:
        text = _tracked_text(rel_path)
        exempted = STALE_EXEMPTION.search(text)
        assert not exempted, (
            f"{rel_path} describes a certificate exemption again: {exempted.group(0)!r} — "
            "the control is ApiPeerGate's peer-address rule, which skips nothing")
        required = STALE_CERT_REQUIRED.search(text)
        assert not required, (
            f"{rel_path} says some origin must still present a certificate: "
            f"{required.group(0)!r} — the mechanism was dropped 2026-06-14")

    listing = subprocess.run(
        ["git", "-C", str(REPO), "grep", "-i", "-l", "mtls", "--", "architecture/"],
        capture_output=True, text=True,
    )
    assert listing.returncode == 0, (
        f"`git grep -i -l mtls -- architecture/` found nothing at all: {listing.stderr.strip()}")
    hits = {ln.strip() for ln in listing.stdout.splitlines() if ln.strip()}
    carriers = hits - set(ARCH_DOCS)
    assert carriers, (
        "no architecture doc outside the two under ban mentions the dropped mechanism: the "
        "true history ('mTLS was dropped on 2026-06-14') has to stay findable somewhere, or "
        "the 0 hits above are a doc set gone silent rather than a claim corrected"
    )
    assert "architecture/infrastructure.md" in carriers or "architecture/mission-control.md" in carriers, (
        f"the docs that carry the drop history are not among the hits: {sorted(hits)}")


def test_panel_doc_names_the_peer_gate_that_admits_the_loopback_call():
    """Clause 1: §Getting to the backend must name the live control by symbol —
    `ApiPeerGate`, deciding through `_is_trusted_peer` on the peer address — since
    that section is the answer to "why does the panel get to use plain HTTP".
    Naming the symbols is also what keeps the sentence true when the line numbers
    around them move."""
    # Flattened: markdown wraps, and a phrase pinned across a hard wrap would
    # fail for a reason that has nothing to do with its claim.
    section = " ".join(_section(_tracked_text(BROWSER_PANEL),
                               "## Getting to the backend").split()).lower()
    assert "apipeergate" in section, "the section must name ApiPeerGate as what admits the call"
    assert "_is_trusted_peer" in section, "the section must name the decision function"
    assert "loopback" in section, "the section must cover the loopback case"
    assert "peer address" in section, "the section must say the rule is the peer address"
    assert "no client certificate" in section, (
        "the section must say the panel presents no certificate and none is asked")


def test_panel_doc_defers_the_network_list_and_the_cert_history():
    """Clause 4: the panel doc points at `[[authority-surfaces]]` for the boundary
    and `[[mission-control]]` for the certificate history, and copies neither. A
    CIDR literal in this file is a second definition of the trusted set, which is
    exactly what drifted the first time."""
    text = _tracked_text(BROWSER_PANEL)
    copied = CIDR.search(text)
    assert not copied, (
        f"{BROWSER_PANEL} restates the trusted-network list ({copied.group(0)!r}) instead "
        "of deferring it to [[authority-surfaces]]")
    assert "[[authority-surfaces]]" in text, "the network rule belongs to [[authority-surfaces]]"
    assert "[[mission-control]]" in text, "the certificate history belongs to [[mission-control]]"

    # Control over the ban itself: the pattern does fire on a CIDR, and the one
    # place it is allowed to fire is the doc that owns the list.
    authority = _tracked_text(AUTHORITY)
    assert CIDR.search(authority), (
        f"{AUTHORITY} no longer carries the trusted-network default, so the ban above "
        "is firing on an empty pattern rather than on a doc that kept its copy")


def test_authority_loopback_bullet_states_the_peer_address_rule():
    """Clause 2: the row that owns the boundary has to state it as the peer
    address — loopback, or a network in `server.trusted_networks` — and name both
    halves of the live control (`ApiPeerGate`, deciding through
    `_is_trusted_peer`). This is the bullet `browser-side-panel.md` defers to, so
    if it goes vague the deferral points at nothing."""
    bullet = " ".join(_section(_tracked_text(AUTHORITY), "- **Loopback.**",
                              stops=("\n- ", "\n## ")).split()).lower()
    assert "apipeergate" in bullet, "the bullet must name the gate"
    assert "_is_trusted_peer" in bullet, "the bullet must name the decision function"
    assert "peer address" in bullet, "the bullet must say the rule is the peer address"
    assert "loopback" in bullet, "the bullet must cover the loopback case"
    assert "server.trusted_networks" in bullet, (
        "the bullet must name the config key that widens the trusted set")
