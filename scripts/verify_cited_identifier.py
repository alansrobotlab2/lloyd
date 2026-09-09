#!/usr/bin/env python3
"""Resolve a cited identifier against its primary source before a claim rests on it.

Backlog #447. A 2026-09-06 research card cited vLLM **PR #34223** as evidence for
"GDN + MoE shared-expert overlap". The API record is
``[Kernels] Make GGUF linear method allow 3d inputs`` — closed **unmerged** on
2026-07-13 as a conflicting draft, touching two GGUF CUDA files. The overlap work
the card described sat in three other open PRs, none landed. A number that
resolves to a different title is not minor drift: the described work does not
exist at that number, so every claim leaning on it is unsupported.

The same class appeared in memory: commit ``96472427`` credited with the ``.bak``
rotation fix does not exist in the repo (the pickaxe names ``bbe5689``).

This is the mechanical half of the ``cited-identifier-verification`` skill — the
part that must not be left to recall. It resolves the record and compares it to
the claim; a human or a model still decides what to do about the verdict.

Usage
-----
    verify_cited_identifier.py pr <owner/repo> <number> [--claim "what you assert"]
                                             [--assert-landed] [--json]

Exit codes
----------
    0  VERIFIED        — the record matches the claim (or no claim was given to check)
    1  MISMATCH        — it resolves, but to something the claim does not describe
    2  NOT_LANDED      — the claim asserts landed work and the record says it never merged
    3  UNRESOLVED      — could not fetch: 404, rate limit, no network. NOT a pass.

``UNRESOLVED`` deliberately exits non-zero. Treating "couldn't check" as "checked
and fine" is how a bad citation survives its own verification step.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.request

API = "https://api.github.com"
UA = "lloyd-cited-identifier-verification/1.0"

OK, MISMATCH, NOT_LANDED, UNRESOLVED = 0, 1, 2, 3

#: Words that carry no signal when comparing a free-text claim to a PR title.
#: Citation verbs first: a claim that says "the PR added X" must not fail to
#: match a title that says "Add X" because of the framing word.
_STOPWORDS = frozenset(
    """
    a an the of for and or to in on with at by from is are was were be been being
    that this these those it its as than then which who whom whose
    pr pull request issue ticket patch commit merge merged prs
    claim claims cited citation cites says said stated the card note research
    work works change changes support supports added adds add adding
    now also still just very really completely actually
    """.split()
)

#: Minimum share of the claim's content words that must appear in the title for
#: the citation to count as describing the same work. Set low on purpose: a
#: paraphrased claim ("GDN + MoE shared-expert overlap") will not reproduce every
#: word of a real title, so this only has to reject citations that share almost
#: nothing with what they point at — which is exactly the #34223 shape (0 words).
_MATCH_THRESHOLD = 0.30


def _stem(token: str) -> str:
    """Cheap suffix strip, so 'overlapping' matches 'overlap'."""
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if len(token) - len(suffix) >= 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _tokens(text: str) -> set[str]:
    """Content-word stems, lowercased and accent-folded."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    raw = re.findall(r"[a-z0-9][a-z0-9+#.\-]*", text)
    out = set()
    for tok in raw:
        tok = tok.strip(".-")
        if not tok or tok in _STOPWORDS:
            continue
        out.add(_stem(tok))
    return out


def _shares(a: str, pool: set[str]) -> bool:
    """A claim token counts if it stems into the title pool, or is a substring of
    a sufficiently long title token (and vice versa). Covers ``Qwen3.5-27B`` vs
    ``Qwen3.5`` and ``quantization`` vs ``quantize``."""
    if a in pool:
        return True
    if len(a) < 5:
        return False
    return any(a in b or b in a for b in pool if len(b) >= 5)


def title_matches_claim(title: str, claim: str,
                        threshold: float = _MATCH_THRESHOLD) -> tuple[bool, list[str]]:
    """Compare a claim to a resolved title.

    Returns ``(matched, unmatched_tokens)``. An empty claim has nothing to
    falsify, so it matches — the caller decides whether an unchecked citation is
    acceptable (it is not, per the skill; that is a gate the skill enforces, not
    this function)."""
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return True, []
    title_tokens = _tokens(title)
    unmatched = sorted(t for t in claim_tokens if not _shares(t, title_tokens))
    matched = len(claim_tokens) - len(unmatched)
    return (matched / len(claim_tokens)) >= threshold, unmatched


def github_token(env: dict | None = None) -> str:
    env = os.environ if env is None else env
    return env.get("GITHUB_TOKEN") or env.get("GH_TOKEN") or ""


def fetch_record(owner_repo: str, number: int, token: str = "") -> dict:
    """GET the PR record. Returns a dict with ``found`` set, or ``error``."""
    url = f"{API}/repos/{owner_repo}/pulls/{number}"
    headers = {"User-Agent": UA, "Accept": "application/vnd.github+json"}
    tok = token or github_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as exc:
        # 404 is an answer (no such PR); 403 is usually a rate limit and is not.
        return {"found": False, "number": number, "repo": owner_repo,
                "status": exc.code,
                "error": ("no such PR in that repo" if exc.code == 404
                          else f"HTTP {exc.code} from the API")}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"found": False, "number": number, "repo": owner_repo,
                "error": f"could not reach the API: {exc}"}
    return {
        "found": True,
        "repo": owner_repo,
        "number": body.get("number", number),
        "title": body.get("title", ""),
        "state": body.get("state", ""),
        "merged": bool(body.get("merged")),
        "merged_at": body.get("merged_at"),
        "html_url": body.get("html_url", ""),
    }


def check_pr_claim(record: dict, claim: str = "",
                   assert_landed: bool = False) -> tuple[list[str], list[str]]:
    """Judge a claim against a resolved record.

    Returns ``(reasons, notes)``. Non-empty ``reasons`` means the citation does
    not support the claim — this is the signal the research skills' pre-synthesis
    step is supposed to act on.
    """
    reasons: list[str] = []
    notes: list[str] = []
    if not record.get("found"):
        return [record.get("error") or "identifier did not resolve"], notes

    title = record.get("title", "")
    if claim:
        matched, unmatched = title_matches_claim(title, claim)
        if not matched:
            reasons.append(
                "title does not describe the claim"
                + (f" (no claim content words appear in it: {unmatched})" if unmatched else ""))
        elif unmatched:
            notes.append(f"claim words absent from the title: {unmatched}")

    if assert_landed and not record.get("merged"):
        state = record.get("state", "unknown")
        closed = record.get("merged_at") or record.get("closed_at") or "never closed"
        reasons.append(f"the claim asserts landed work; this PR is {state}, unmerged ({closed})")
    return reasons, notes


def _verdict_line(repo: str, number: int, reasons: list[str], notes: list[str]) -> tuple[str, int]:
    if reasons:
        head = "MISMATCH"
        code = NOT_LANDED if any("landed work" in r for r in reasons) else MISMATCH
        return head, code
    return "VERIFIED", OK


def cmd_pr(args) -> int:
    owner_repo = args.repo.strip().strip("/")
    if owner_repo.endswith(".git"):
        owner_repo = owner_repo[:-4]
    record = fetch_record(owner_repo, args.number, args.token)
    reasons, notes = ([], [])
    if record.get("found"):
        reasons, notes = check_pr_claim(record, args.claim, args.assert_landed)
    else:
        reasons = [record.get("error") or "could not resolve"]

    if args.json:
        print(json.dumps({"record": record, "claim": args.claim,
                          "reasons": reasons, "notes": notes}, indent=2))
        return OK if not reasons else UNRESOLVED if not record.get("found") else (
            NOT_LANDED if any("landed work" in r for r in reasons) else MISMATCH)

    if not record.get("found"):
        print(f"UNRESOLVED  {owner_repo}#{args.number}: {reasons[0]}")
        print("            This is not a pass. Open the source another way, or write")
        print("            the claim as unresolved. Do not hedge it and move on.")
        return UNRESOLVED

    print(f"{'VERIFIED' if not reasons else 'CAUGHT'}  "
          f"{owner_repo}#{record['number']}: {record['title']}")
    print(f"          state={record['state']} merged={record['merged']} "
          f"merged_at={record['merged_at']}")
    print(f"          {record['html_url']}")
    if args.claim:
        print(f"claim: {args.claim}")
    for r in reasons:
        print(f"  - {r}")
    for n in notes:
        print(f"  note: {n}")
    if reasons:
        print("VERDICT: the citation does not support the claim. Drop the claim or")
        print("         write it as unresolved, and record the resolved title + date")
        print("         where the citation stood.")
        return NOT_LANDED if any("landed work" in r for r in reasons) else MISMATCH
    print("VERDICT: resolved title matches the claim. Record the title and today's")
    print("         date next to the citation so the next reader sees the check.")
    return OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verify_cited_identifier.py",
        description="Resolve a cited PR/commit identifier and compare it to the claim.")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("pr", help="resolve a GitHub pull request by number")
    pr.add_argument("repo", help="owner/repo, e.g. vllm-project/vllm")
    pr.add_argument("number", type=int)
    pr.add_argument("--claim", default="",
                    help="what the note asserts about this PR, in your own words")
    pr.add_argument("--assert-landed", action="store_true",
                    help="fail if the PR never merged (the claim says the work shipped)")
    pr.add_argument("--token", default="", help="GitHub token; defaults to $GITHUB_TOKEN")
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=cmd_pr)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
