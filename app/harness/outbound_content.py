"""Deterministic outbound content gate (#1136).

Every guard on Lloyd's dispatch path answers a different question.
`app/harness/policy.py` answers *who is authorised* to make a side-effecting
call; `app/harness/safety.py` answers *which Bash command is catastrophic*.
Neither reads a payload, so until this module an authorised, correctly-tiered
`email_send` whose body carried a private key or a phone number dispatched
unchanged. This is the third question: **what is in the arguments**.

The design is PostHog's, from the AI Engineer talk that produced the backlog
item ("We let an AI agent execute Bash and lived to talk about it", 2026-09-14,
[[20260914-we-let-an-ai-agent-execute-bash-and-lived-to-talk-about-it-sarah-sanders-posthog]]):
every rule carries a **direction** — content flowing *into* the agent versus
content the agent is *writing* — and Lloyd had only the inbound half, in
prose. This module is the outbound half, and three of that talk's choices are
load-bearing here:

1. **No model on the enforcement path.** Every rule is a compiled regex plus a
   plain-Python guard. Same input, same output, no call, no latency, no
   day where a bad model releases a secret. An LLM triage arm is explicitly
   out of scope and, if it is ever added, it may only suppress *reported*
   findings after this pass — never release a refusal.
2. **Detection and enforcement are separate jobs.** `scan_outbound_payload`
   returns findings and stops; the hook decides. Callers that want a report
   can call the scanner directly.
3. **Severity tracks measured impact, not scariness, and the false-positive
   class is the one that actually kills the tool.** Sanders' operational
   problem was demo login screens and documentation prose, not attacks. So
   every rule here ships its negative patterns *inside the rule object*
   (`positive` / `negative`), they are executed by the suite, and
   `validate_rule_table` refuses to import a rule that has no negative test —
   "that negative test is the first line of defense against false positives".

Actions, and why they are asymmetric
-----------------------------------
- `action="deny"` — credential-shaped. A private key or a live API token in an
  outbound argument is never needed to accomplish a task, so refusing is
  nearly free and the leak is irreversible.
- `action="report"` — PII-shaped. An unattended job that emails a phone number
  pulled from the contacts list is **normal operation**; blocking it would be
  the false-positive death this item's own risk clause names. Reported
  findings land in a JSONL ledger (`findings_path()`) redacted so the ledger
  is not itself the leak, and the call proceeds.

Scope: direction `out` over string arguments only
-------------------------------------------------
The scanner is handed `tool_input` and nothing else. It never sees a tool
*result*: the moment a rule reads a `contacts_get` response or a fetched page
it starts firing on legitimate research, which is the inbound problem and is
a separate item. Argument values are walked recursively because a sender
payload is nested (`attachments`, `messages`); keys are not scanned, since a
parameter name is chosen by the schema and not authored under pressure.

Ablation
--------
One table, one installer. `install_outbound_content_gate` is the only wiring;
`EXEMPT_SCOPES` (scope → written reason, same convention as
`skill_lint.PHANTOM_EXEMPT`) turns it off for one scope; setting every rule's
action to `"report"` makes the whole module observational without touching a
call site.
"""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.harness.hooks import HookRegistry
from app.harness.policy import current_scope, effective_tier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule format
# ---------------------------------------------------------------------------

ACTIONS = ("deny", "report")
SEVERITIES = ("critical", "high", "medium", "low")
#: Every rule is direction "out" by construction, and a test pins it. A rule
#: with any other direction does not belong in this module: scanning content
#: flowing *into* the agent is a different gate with a different FP profile.
DIRECTIONS = ("out",)


@dataclass(frozen=True)
class ContentRule:
    """One row of the table. Metadata first, then patterns, then the test corpus.

    `condition` is the prose form of YARA's condition of the same name: what
    has to be true of the match for it to count. `guards` names the plain
    functions that implement it, resolved against `_GUARDS` at import — a
    typo in a guard name is an import error, not a silently-disabled rule.
    """

    name: str
    description: str
    severity: str
    category: str
    action: str
    direction: str
    regexes: tuple[str, ...]
    condition: str
    guards: tuple[str, ...] = ()
    positive: tuple[str, ...] = ()
    negative: tuple[str, ...] = ()


#: Matches excluded before any guard runs: documentation's own example
#: credentials. These are the strings a scanner is *taught* to print in a
#: README, so a rule that fires on them has misfired. Every entry names where
#: it comes from; a literal with no provenance is not allowed in here.
#:
#: **An entry that carries a live-looking prefix must be split across a `+`**,
#: as the Stripe pair below is. This module's job is to hold
#: credential-shaped strings and GitHub's push protection scans the raw file:
#: on 2026-09-20 these two entries blocked a push of 34 commits (`GH013`,
#: commit `904f0bac`). Because protection scans every commit in a push,
#: cleaning it up in a LATER commit does not unblock the earlier one — the
#: only ways out are a per-secret unblock URL or rewriting published history,
#: and here history is what the automod ledger, the LKG and every rollback
#: target are keyed on. Splitting the prefix from the body is
#: runtime-identical and defeats a prefix-anchored scanner.
#: `tests/test_outbound_content_gate.py` pins both halves: that the set still
#: contains the joined strings, and that the source does not.
EXEMPT_LITERALS = frozenset({
    "AKIAIOSFODNN7EXAMPLE",          # AWS IAM documentation access key id
    "ASIAIOSFODNN7EXAMPLE",          # the same, STS form
    "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",  # AWS docs secret key
    "AKIAIOSFODNN7EXAMPLEKEY",       # the docs key id padded to a full pair
    "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",   # GitHub's token placeholder
    "ghp_yourtokenhere",             # the placeholder people actually paste
    # Stripe's published documentation key body. Only the `sk_test_` form is
    # the key Stripe itself publishes; the `sk_live_` entry is that same body
    # under the live prefix, kept because a doc that pastes one often pastes
    # the other. That prefix is what scanners rate highest, which is why this
    # pair is what blocked the push.
    "sk_live_" + "4eC39HqLyjWDarjtT1zdp7dc",
    "sk_test_" + "4eC39HqLyjWDarjtT1zdp7dc",
})

# Guards — a function of the candidate token; True keeps the finding.
def _guard_present(token: str) -> bool:
    """No condition beyond 'the pattern matched'. Named so the table reads
    uniformly and a rule is never silently missing its condition."""
    return True


_HEX_ONLY = re.compile(r"[0-9a-fA-F]+$")
_PLACEHOLDER_MARKS = (
    "example", "placeholder", "changeme", "change_me", "your", "redacted",
    "dummy", "sample", "fixture", "xxxxxxxx", "todo", "lorem", "deadbeef",
    "0123456789", "abcdefghij",
)


def _guard_not_hex_digest(token: str) -> bool:
    """A hex-only run of 32+ characters is a digest or a git object id, not a
    credential. This is the single most important guard on this machine: the
    vault cites commit shas constantly (40 hex chars is *the* git sha shape),
    and without this rule 1 would deny every email that quotes a commit.
    Cost: a real secret written as pure hex is missed. That cost is the right
    one to pay — see the module docstring on false-positive death."""
    return not _HEX_ONLY.fullmatch(token)


def _guard_high_entropy(token: str) -> bool:
    """Character-class diversity, the cheap stand-in for Shannon entropy.

    A generated credential has a digit, a lowercase letter and an uppercase
    letter, and few repeated characters. Prose and identifiers do not, so this
    is what keeps a sentence that happens to contain a long word — or a
    snake_case identifier quoted next to the word `token` — from becoming a
    refusal. stdlib only: the vendored `shannon` lives under `.venvs/` and is
    not an install requirement, so it cannot be a dependency of the
    enforcement path.
    """
    if len(token) < 32:
        return False
    if not re.search(r"[0-9]", token):
        return False
    if not re.search(r"[a-z]", token):
        return False
    if not re.search(r"[A-Z]", token):
        return False
    return (len(set(token)) / len(token)) >= 0.5


def _guard_not_placeholder(token: str) -> bool:
    low = token.lower()
    return not any(mark in low for mark in _PLACEHOLDER_MARKS)


_GUARDS = {
    "present": _guard_present,
    "not-hex-digest": _guard_not_hex_digest,
    "high-entropy": _guard_high_entropy,
    "not-placeholder": _guard_not_placeholder,
}


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------
#
# String arguments only, direction "out", tier-2/tier-3 senders only. Ordering
# is presentation order in a refusal reason, not precedence: all rules run and
# the findings are returned together.

CONTENT_RULES: tuple[ContentRule, ...] = (
    ContentRule(
        name="private-key-material",
        description=(
            "PEM/OpenSSH/PGP private-key material, an AWS credentials-file "
            "secret, or a browser/Thunderbird login-store dump. Nothing in "
            "Lloyd's legitimate work needs to send one: a key is installed, "
            "not emailed."
        ),
        severity="critical",
        category="credential",
        action="deny",
        direction="out",
        regexes=(
            # `-----BEGIN RSA PRIVATE KEY-----`, `-----BEGIN OPENSSH PRIVATE
            # KEY-----`, `-----BEGIN ENCRYPTED PRIVATE KEY-----`. PUBLIC KEY
            # does not match, which is its negative test.
            r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----",
            r"-----BEGIN PGP PRIVATE KEY BLOCK-----",
            # ~/.aws/credentials, pasted or read off disk.
            r"(?im)^\s*aws_secret_access_key\s*[=:]\s*[A-Za-z0-9/+=]{16,}",
            # Thunderbird logins.json / browser logins dump.
            r"(?i)\"encrypted(?:username|password)\"\s*:",
        ),
        condition="any of them",
        guards=("present",),
        positive=(
            "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEA\n",
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n",
            "aws_secret_access_key = wJalrXUtnFEMI7MDENGkPxRfiCYz-example\n",
            '{"hostname":"https://x","encryptedUsername":"MEQCIAB="}',
        ),
        negative=(
            # A public key is the thing people legitimately paste.
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPublicKeyValueNotSecret alan@goliath",
            "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQFOz6d-----END PUBLIC KEY-----",
            # A note *describing* the header, which is what the vault full of.
            "If the file starts with a PEM private key header, do not commit it.",
            "The config wants an aws credentials file with an access key and secret.",
        ),
    ),
    ContentRule(
        name="provider-token",
        description=(
            "A provider-prefixed API token: AWS key id, GitHub PAT, "
            "OpenAI/Anthropic secret key, Slack token, Google API key, GitLab "
            "PAT, Stripe secret. The prefix is the whole precision budget — a "
            "prefix nobody hand-writes in prose means the shape itself is the "
            "signal, so no adjacency is required."
        ),
        severity="high",
        category="credential",
        action="deny",
        direction="out",
        regexes=(
            r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b",
            r"\bgh[pousr]_[0-9A-Za-z]{36,}\b",
            r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b",
            r"\bsk-[A-Za-z0-9]{20,}\b",
            r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b",
            r"\bAIza[0-9A-Za-z_\-]{35}\b",
            r"\bglpat-[0-9A-Za-z_\-]{20,}\b",
            r"\b(?:sk|pk)_live_[0-9A-Za-z]{16,}\b",
        ),
        condition="any of them, and the match is not a documentation example",
        guards=("not-placeholder",),
        positive=(
            "AWS_ACCESS_KEY_ID=AKIAQZ7AB3CD5EFG6HIJ",
            "ghp_1a2B3c4D5e6F7g8H9i0JkLmNoPqRsTuVwXyZ",
            "sk-ant-api03-Qz7Kp2Rm9Xn4Vb1Ld8Fg3Hs6Jt0Wy5",
            "glpat-9aB_cDeFgHiJkLmNoPqRs",
        ),
        negative=(
            # Documentation's own example, and the placeholder shapes.
            "AKIAIOSFODNN7EXAMPLE",
            "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "export OPENAI_API_KEY=***",
            "The skill uses a sk- style key from the provider console.",
            "sk-1 is the schedule kind in the queue schema.",
        ),
    ),
    ContentRule(
        name="assigned-high-entropy-secret",
        description=(
            "A credential-shaped value in an assignment position: a ≥32-char "
            "token bound to a name containing key/token/secret/password/"
            "credential, in either direction (`token = VALUE` or "
            "`SOME_API_KEY=VALUE`). Assignment is the requirement — a long "
            "random-looking string that merely *appears* in prose is not a "
            "secret being transmitted."
        ),
        severity="high",
        category="credential",
        action="deny",
        direction="out",
        regexes=(
            r"(?i)\b(?:api[ _-]?key|access[ _-]?token|access[ _-]?key[ _-]?id"
            r"|secret[ _-]?access[ _-]?key|secret[ _-]?key|client[ _-]?secret"
            r"|auth[ _-]?token|bearer[ _-]?token|refresh[ _-]?token"
            r"|api[ _-]?secret|password|passwd|secret|token|credential)s?"
            r"\b[\"']?[ \t]{0,4}[=:][ \t]{0,4}[\"']?"
            r"(?P<token>[A-Za-z0-9]{32,})",
            r"(?im)^[ \t]*(?:export[ \t]+)?(?P<name>[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD"
            r"|CREDENTIAL)[A-Z0-9_]*)[ \t]*=[ \t]*(?P<token>[A-Za-z0-9]{32,})",
        ),
        condition=(
            "any of them, and the captured token is not a hex digest, is "
            "character-class diverse, and is not a placeholder"
        ),
        guards=("not-hex-digest", "high-entropy", "not-placeholder"),
        positive=(
            "ANTHROPIC_API_KEY=Qz7Kp2Rm9Xn4Vb1Ld8Fg3Hs6Jt0Wy5Zu2Ci7Po4Ka1Em\n",
            'body = {"access_token": "Qz7Kp2Rm9Xn4Vb1Ld8Fg3Hs6Jt0Wy5Zu2Ci7Po4Ka1Em"}',
            "the config file said api_key: Qz7Kp2Rm9Xn4Vb1Ld8Fg3Hs6Jt0Wy5Zu2Ci7Po4Ka1Em",
        ),
        negative=(
            # The sha256 of a released artifact, quoted in a release note — the
            # most common 64-char alphanumeric run this machine ever sends.
            "sha256sum: 9f2c1b7a4d6e3f0a5c8b2d1e7f4a9c6b3d0e8f5a2c7b4d1e9f6a3c0b7d4e1f8a  dist.tar.gz",
            # A git object id: 40 hex chars, cited constantly.
            "landed as 5caabaef6dd318c3d324a241c584b15612ded27e yesterday",
            # Prose that contains the word and a long word, no assignment.
            "The token endpoint rotates the signing token every ninety days.",
            # A placeholder a README tells you to replace.
            "API_KEY=your_api_key_here_placeholder_value",
            # A snake_case identifier: long, next to TOKEN, but no digit and no
            # mixed case, so it is a name and not a value.
            "TOKEN=LLOYD_OUTBOUND_CONTENT_GATE_FINDINGS_PATH",
        ),
    ),
    ContentRule(
        name="phone-number-e164",
        description=(
            "An E.164 phone number in an outbound argument. Reported, never "
            "denied: sending a contact's number is what a mail agent does, and "
            "a rule that breaks the nightly brief is a rule that gets turned "
            "off, taking the credential rules with it."
        ),
        severity="medium",
        category="pii",
        action="report",
        direction="out",
        regexes=(r"(?<![\d.+-])(?P<token>\+[1-9][0-9]{6,14})(?!\d)",),
        # ^ a 1-tuple, not a bare string: iterating a str yields characters,
        # and validate_rule_table refuses a bare string for exactly that reason.
        condition="any of them",
        guards=("present",),
        positive=(
            "Alan: +14155559876\nDana: +442071838750",
            "call me at +15551234567",
        ),
        negative=(
            "14155559876",              # national form: not what the rule claims
            "version +1.2.3 and +1.20.30 of the package",
            "the diff adds +4 lines and removes -2 in patch 3.14.159",
            "UTC offset +00:00",
        ),
    ),
    ContentRule(
        name="tailnet-address",
        description=(
            "A private/CGNAT IPv4 or a `.taile<NNN>.ts.net` tailnet hostname. "
            "Reported: these are in every infrastructure note, and the useful "
            "alert is a reviewer noticing that an *external* recipient is "
            "being handed the network map, not a blocked brief."
        ),
        severity="low",
        category="network-topology",
        action="report",
        direction="out",
        regexes=(
            r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
            r"|192\.168\.\d{1,3}\.\d{1,3}"
            r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
            r"|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3})\b",
            r"\b[\w.-]*\.taile[0-9a-f]+\.ts\.net\b",
        ),
        condition="any of them",
        guards=("present",),
        positive=(
            "ssh alansrobotlab@100.105.113.88",
            "browse https://goliath.taile37041.ts.net:5173/",
            "the desktop sits at 192.168.1.42 behind the router",
        ),
        negative=(
            # Public and documentation ranges.
            "resolver 8.8.8.8",
            "example traffic goes to 203.0.113.5 per RFC 5737",
            "upgrade to v100.64.2 of the parser, build 1.100.2.3",
            "tailscale status shows the peer by name only",
        ),
    ),
)


# ---------------------------------------------------------------------------
# Table validation
# ---------------------------------------------------------------------------


class RuleTableError(ValueError):
    """The table is malformed. Raised at import: a rule whose guard name is a
    typo, or which ships no negative test, must stop the process rather than
    run silently unchecked."""


def validate_rule_table(rules: Iterable[ContentRule] = None) -> None:
    """Every structural promise the suite relies on, in one place.

    Lives in the module rather than in a test so a mutation of the table —
    dropping a negative pattern, inventing a severity, naming a guard that
    does not exist — is refused at import *and* asserted by the suite. A test
    that only iterates the shipped table would pass a rule that never
    contained a negative pattern at all; this function is what makes that
    shape impossible.
    """
    rows = tuple(CONTENT_RULES if rules is None else rules)
    seen: set[str] = set()
    for rule in rows:
        where = f"rule {rule.name!r}"
        if not rule.name:
            raise RuleTableError("every rule needs a name: a refusal that "
                                 "cannot name its rule is un-debuggable")
        if rule.name in seen:
            raise RuleTableError(f"{where}: duplicate rule name")
        seen.add(rule.name)
        if rule.action not in ACTIONS:
            raise RuleTableError(f"{where}: action must be one of {ACTIONS}")
        # A credential is never a finding-only category. Without this the way
        # to silence a noisy credential rule is to demote it to `report`, which
        # is the outcome this table exists to make impossible: the asymmetry
        # ('credential-shaped -> refuse, PII-shaped -> record') is the design,
        # so the validator enforces it and one author cannot re-decide it alone.
        if rule.category == "credential" and rule.action != "deny":
            raise RuleTableError(
                f"{where}: category 'credential' must action 'deny' — a "
                "credential match is refused, never merely recorded")
        if rule.severity not in SEVERITIES:
            raise RuleTableError(f"{where}: severity must be one of {SEVERITIES}")
        if rule.direction not in DIRECTIONS:
            raise RuleTableError(f"{where}: direction must be one of {DIRECTIONS} "
                                 "— inbound scanning is a different gate")
        if not rule.description.strip() or not rule.condition.strip():
            raise RuleTableError(f"{where}: description and condition are both "
                                 "required; they are what a reviewer reads")
        if not rule.category.strip():
            raise RuleTableError(f"{where}: category is required")
        if isinstance(rule.regexes, str):
            raise RuleTableError(
                f"{where}: regexes is a bare string, so iterating it yields "
                "characters — the table would 'validate' and match nothing")
        patterns = tuple(rule.regexes)
        if not patterns:
            raise RuleTableError(f"{where}: no patterns")
        for pat in patterns:
            try:
                re.compile(pat)
            except re.error as exc:
                raise RuleTableError(f"{where}: pattern does not compile: {exc}") from exc
        for guard in rule.guards:
            if guard not in _GUARDS:
                raise RuleTableError(f"{where}: guard {guard!r} is not in "
                                     f"{sorted(_GUARDS)} — a rule with an "
                                     "unresolvable condition is a rule that "
                                     "silently matches nothing")
        if not rule.guards:
            raise RuleTableError(f"{where}: every rule names its condition guards")
        if not rule.positive:
            raise RuleTableError(f"{where}: no positive test pattern")
        # The clause this whole function exists for: a rule without a negative
        # test is a rule whose false-positive rate is unknown, and unknown is
        # the number that ends gates.
        if not rule.negative:
            raise RuleTableError(f"{where}: no negative test pattern — every "
                                 "rule ships the benign strings that must not "
                                 "fire, because that negative test is the "
                                 "first line of defense against false positives")


validate_rule_table()


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One rule matched one argument. `excerpt` is already redacted."""

    rule: str
    severity: str
    category: str
    action: str
    arg: str
    excerpt: str
    matched_sha256: str = ""

    def as_row(self) -> dict[str, str]:
        return {
            "rule": self.rule, "severity": self.severity,
            "category": self.category, "action": self.action,
            "arg": self.arg, "excerpt": self.excerpt,
            "matched_sha256": self.matched_sha256,
        }


def redact(text: str) -> str:
    """Keep the shape, lose the value.

    The findings ledger is written by an unattended job and read by whoever
    triages it; an unredacted excerpt would move the secret from a payload
    into a file that outlives it. First 4 and last 2 characters are enough to
    tell one finding from another and nowhere near enough to reconstruct a
    key.
    """
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}{'*' * (len(text) - 6)}{text[-2:]}"


def _arg_paths(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    """Yield `(path, string)` for every string in a nested argument tree.

    Keys are skipped deliberately: a parameter name comes from the tool
    schema, so scanning names would fire on `body_contains_token` forever.
    """
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, sub in value.items():
            yield from _arg_paths(sub, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple, set)):
        for i, sub in enumerate(value):
            yield from _arg_paths(sub, f"{prefix}[{i}]")


def _candidate_text(match: re.Match) -> str:
    """The token a guard judges: the named `token` group when the pattern has
    one (the assignment position), else the whole match."""
    token = None
    if "token" in (match.re.groupindex or {}):
        token = match.group("token")
    return token if token else match.group(0)


#: Longest single argument value the scan reads. The scan is synchronous and
#: sits in front of every tier-2/tier-3 call, so an unbounded regex pass over a
#: multi-megabyte body would make the gate the slowest thing on the dispatch
#: path — and a guard that makes a job slow is a guard that gets turned off.
#: Truncation is *reported*, never passed quietly: a call whose payload the
#: gate could not read in full is a call with a hole in its verdict, and the
#: ledger is where somebody who cares can go look for it.
MAX_SCAN_CHARS = 65_536

#: Synthetic finding (not a rule, so it is in no table and can never deny)
#: naming an argument this bound truncated. Same standing as
#: `_no_string_arguments`.
OVERSIZED_FINDING = "_skipped_oversized"


def scan_outbound_payload(args: Any) -> list[Finding]:
    """Findings for one call's arguments. Detection only — it decides nothing.

    Pure and synchronous: no I/O, no model, no clock, so the same arguments
    always produce the same findings and a replay is a real measurement.
    """
    findings: list[Finding] = []
    oversized: list[tuple[str, int]] = []
    for path, full in _arg_paths(args or {}):
        text = full
        if len(text) > MAX_SCAN_CHARS:
            oversized.append((path, len(text)))
            text = text[:MAX_SCAN_CHARS]
        for rule in CONTENT_RULES:
            for pattern in _COMPILED[rule.name]:
                for match in pattern.finditer(text):
                    token = _candidate_text(match)
                    if token.upper() in EXEMPT_LITERALS or token in EXEMPT_LITERALS:
                        continue
                    if not all(_GUARDS[g](token) for g in rule.guards):
                        continue
                    findings.append(Finding(
                        rule=rule.name, severity=rule.severity,
                        category=rule.category, action=rule.action, arg=path,
                        excerpt=redact(token),
                        matched_sha256=hashlib.sha256(
                            token.encode("utf-8", "replace")).hexdigest()[:12],
                    ))
                    # One finding per (rule, argument): a PEM block quoted six
                    # times is one finding, and a ledger of repeats is how a
                    # reviewer stops reading the ledger.
                    break
                else:
                    continue
                break
    for path, size in oversized:
        findings.append(Finding(
            rule=OVERSIZED_FINDING, severity="low", category="coverage",
            action="report", arg=path,
            excerpt=f"argument is {size} "
                    f"characters; the first {MAX_SCAN_CHARS} were scanned and "
                    f"the rest were not",
        ))
    return findings


_COMPILED: dict[str, tuple[re.Pattern[str], ...]] = {
    rule.name: tuple(re.compile(p) for p in rule.regexes)
    for rule in CONTENT_RULES
}


# ---------------------------------------------------------------------------
# Findings ledger
# ---------------------------------------------------------------------------

FINDINGS_ENV = "LLOYD_OUTBOUND_GATE_FINDINGS"


def findings_path() -> Path:
    """`~/.local/state/lloyd-outbound-gate/findings.jsonl`.

    Outside every git-tracked tree, the way the automod ledger and the effect
    ledger are: a file an unattended run writes must not be able to dirty the
    vault, and this one holds redacted-but-real PII shapes.
    `LLOYD_OUTBOUND_GATE_FINDINGS` overrides for tests and replays.
    """
    raw = os.environ.get(FINDINGS_ENV)
    if raw:
        return Path(str(raw)).expanduser()
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "lloyd-outbound-gate" / "findings.jsonl"


def record_findings(findings: Sequence[Finding], *, scope: str, tool: str,
                    denied: bool = False,
                    path: str | Path | None = None) -> None:
    """Append one JSONL row. Best-effort on purpose.

    A ledger that cannot be written is a reporting failure, never a reason to
    block — the enforcement decision is already made by the time this is
    called, and a `report`-only finding must not become a denial because a
    directory is missing. That asymmetry is the difference between a gate and
    an outage.
    """
    rows = [f.as_row() for f in (findings or ())]
    if not rows:
        return
    target = Path(str(path)) if path else findings_path()
    row = {
        "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "scope": scope, "tool": tool, "findings": rows,
    }
    # Only a refusal gets the flag: an absent key is the allow, so an allow
    # replay stays the boring default and `grep '"denied": true'` over the file
    # is the whole list of calls this gate stopped. Without it the ledger says
    # 'a credential matched' identically for a blocked send and an allowed one.
    if denied:
        row["denied"] = True
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001 — see docstring
        logger.warning("[outbound-content] findings ledger unwritable (%s: %s); "
                       "recording %d finding(s) in the log instead",
                       exc.__class__.__name__, exc, len(rows))
        for r in rows:
            logger.warning("[outbound-content] finding rule=%s severity=%s "
                           "arg=%s excerpt=%s", r["rule"], r["severity"],
                           r["arg"], r["excerpt"])


# ---------------------------------------------------------------------------
# Exempt list
# ---------------------------------------------------------------------------
#
# Same convention as `scripts/skill_lint.py`'s PHANTOM_EXEMPT: an entry is a
# scope plus the written reason it is exempt, so an exemption is a claim a
# reviewer can argue with rather than a silent hole.
#
# What an exemption means here, stated once because a comment that promises
# more than the code does is the defect the review rung caught on the first
# attempt: an exempt scope is **not scanned at all**. The callback returns
# before `scan_outbound_payload`, so nothing about that call is recorded — not
# "scanned, reported, and unblocked". Skipping the scan is the point (a scope
# whose traffic is all false positives should cost nothing), but it means an
# exemption removes the *visibility* as well as the block, which is exactly why
# each entry needs a reason a person can argue with, in its own commit.
#
# It ships empty because the measurement that would justify an entry does not
# exist yet — triage counted 0 tier-2/tier-3 side effects in `workers.db` and 0
# `email_send` calls in 2,826 transcripts, so there is no benign corpus to have
# been burned by. Both directions are pinned by tests in
# `tests/test_outbound_content_gate.py`: a listed scope dispatches unchanged
# even with a private key in its body, and an entry whose reason is blank or
# stub-length is NOT honoured, so the call is scanned and denied anyway.
#
# A malformed entry is treated as not-exempt rather than honoured: a row that
# cannot state why it exists may not open the gate, and the failure is logged
# rather than silent.

EXEMPT_SCOPES: dict[str, str] = {}


def validate_exempt_list(table: Mapping[str, str] | None = None) -> None:
    """Every exemption carries a scope and a written reason, or it is refused.

    The reason is the whole point of an exempt list: an entry that says only
    which scope is skipped is a hole with a name on it. Same convention as
    `scripts/skill_lint.py`'s `PHANTOM_EXEMPT`, which pairs each skill with the
    fact that its job is to name fakes — the reviewer has to be able to argue
    with the reason without going to ask for it.
    """
    live = EXEMPT_SCOPES if table is None else table
    for scope, reason in live.items():
        if not str(scope or "").strip():
            raise RuleTableError("EXEMPT_SCOPES entry with an empty scope "
                                 "(an empty scope matches nothing and reads as "
                                 "a global exemption)")
        text = str(reason or "").strip()
        if len(text) < 20:
            raise RuleTableError(
                f"EXEMPT_SCOPES[{scope!r}] has no usable reason "
                f"(got {text!r}); an exemption needs the sentence that justifies "
                "it, in the same commit that adds it")


def is_exempt(scope: str, exempt: Mapping[str, str] | None = None) -> str | None:
    """The reason this scope is exempt, or None. Exact match only: a prefix
    match would let `worker` exempt the whole fleet, which is what the
    per-task grant system exists to prevent.

    An entry with a blank reason is treated as *not* exempt rather than
    honoured: a malformed row may fail the gate open over nobody's argument.
    """
    table = EXEMPT_SCOPES if exempt is None else exempt
    reason = str(table.get(str(scope or "").strip(), "") or "").strip()
    if len(reason) < 20:
        if reason:
            logger.warning("[outbound-content] exempt entry for scope=%s has no "
                           "usable reason, so it was NOT applied", scope)
        return None
    return reason


# ---------------------------------------------------------------------------
# Hook wiring, and the guard that the wiring is complete
# ---------------------------------------------------------------------------

#: Production dispatch paths that arm this gate, and how each one does it. The
#: set is *derived* from the tree by `_wiring_findings` (real syntax, real
#: scopes), and this list is the claim that derivation re-measures on every
#: suite run through `stale_gate_arm_points` and
#: `dispatch_files_missing_from_arm_points` — never inherited.
#:
#: The count is asserted by a test, because #869's finding was not that a guard
#: was missing but that three guards were installed on one dispatch path and on
#: none of the others, which no test inside either file could see.
GATE_ARM_POINTS: tuple[str, ...] = (
    "app/harness/safety.py",                       # the floor every turn takes
    "agent_mcp/builtin_task.py",                   # Task subagent turns
    "app/routers/messages.py",                     # the 3 chat/ambient dispatch sites
    "app/routers/voice.py",                        # spoken turns
    "autonomy.py",                                 # autonomy task turns
    "workers/sources/_common.py",                  # every worker slot (worker + task)
    "scripts/autoresearch/bench_runner_sdk.py",    # the scored bench path
    "eval/run_preserve_thinking_eval.py",          # live eval, real MCP tools
    "eval/run_tool_choice_eval.py",                # live eval, real MCP tools
    "eval/run_compaction_recall_eval.py",          # live eval, production kwargs
    "eval/run_prefetch_cost_eval.py",              # live eval, real MCP tools
    "eval/decision_replay_588.py",                 # live replay, real MCP tools
)

#: Which dispatch files are outside the roster because their turns cannot reach a
#: sender tool is NOT a list — a hand-maintained allowlist over an open set is
#: how every entry in the catalogued 'a check that reads its own missing input'
#: class got written. It is derived: `sender_unreachable_dispatch_files()` below
#: reads the same syntax as everything else here, and
#: `_TurnBuilder.sender_reachable` is False only for a build that passes no
#: `mcp_servers` and no splat that could. `app/routers/ide.py`'s two single-turn
#: IDE helpers are those builds today; the day one of them registers a server it
#: becomes an unarmed path and a roster gap in the same pass, with no list to
#: update and nobody's permission to ask.

#: The one installer every production turn already calls: the gate is installed
#: inside it, so a scope that calls the floor arms the gate through it.
FLOOR_INSTALLER = "install_default_safety_hook"
GATE_INSTALLER = "install_outbound_content_gate"
INSTALLERS: tuple[str, ...] = (FLOOR_INSTALLER, GATE_INSTALLER)

#: The module that arms by *being called* rather than by building a turn.
#: Its own entry in the roster can never be judged by "does it build a turn
#: it arms", because it builds none — it is the floor the other entries call.
FLOOR_MODULE = "app/harness/safety.py"

#: Package roots the finder reads, and the only places it looks. Bounding by
#: directory rather than walking the tree and filtering is the difference
#: between a guard and an outage: an unbounded recursive walk of a live checkout
#: parses every file under `.venvs/` (tens of thousands, minutes per call) and,
#: worse, would report a vendored dependency that ever built a `RunOptions` as
#: an unarmed dispatch path. A path outside these roots has no installer today,
#: and the suite says so if one appears.
PACKAGE_ROOTS = ("app", "agent_mcp", "workers", "scripts", "eval")

#: `RunOptions` is a dataclass whose `hooks` defaults to `None`, and
#: `run_query` fires no PreToolUse callback for a turn built that way. So the
#: unit of coverage is not a file containing some literal — it is **the call
#: that constructs the turn**, and what that call passes as `hooks`.
#:
#: An earlier revision of this finder attributed coverage to *files* that
#: contained both the literal `HookRegistry()` and the literal `RunOptions(`.
#: That rule was blind in both directions, and the review rung found it: a file
#: building a turn with no `hooks` argument at all (`app/routers/voice.py`
#: before this round) was never even considered, while a file that merely
#: mentioned both strings — in a comment, in a fixture, in prose — counted as
#: armed. Attribution by syntax is what replaces it: see `_turn_builders`.
RUN_OPTIONS_CTOR = "RunOptions"


def _attr(name: ast.expr | None) -> str:
    if isinstance(name, ast.Name):
        return name.id
    if isinstance(name, ast.Attribute):
        return name.attr
    return ""


def _package_files(root: Path) -> Iterable[tuple[str, Path]]:
    # Top level first: `autonomy.py` is a dispatch path and lives at the root,
    # so a scan that only walked packages would have called the autonomy
    # runner unarmed-by-attribution — the failure mode this whole function is
    # here to prevent, wearing its opposite.
    for path in sorted(root.glob("*.py")):
        if not path.name.startswith("test_"):
            yield path.name, path
    for pkg in PACKAGE_ROOTS:
        base = root / pkg
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if path.name.startswith("test_") or ".venvs" in path.parts:
                continue
            yield str(path.relative_to(root)), path


def _reads(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


#: Parsed trees, keyed by (path, mtime_ns, size). The finder's public helpers each
#: walk every package file, and the guard test calls six of them; parsing the tree
#: six times took the test module from seconds to minutes. The key is the file's
#: own stat, so an edit invalidates its entry — the cache can serve a stale tree
#: only within one filesystem timestamp, and a guard that reads a file written in
#: the same nanosecond it was checked was never going to see it anyway.
_PARSE_CACHE: dict[tuple[str, int, int], "ast.Module | None"] = {}
_PARSE_CACHE_CAP = 4096


def _parse(rel: str, path: Path) -> ast.Module | None:
    """Parse one source file, or return None if it is not parseable.

    A file the finder cannot parse is skipped rather than reported: a syntax
    error in somebody else's in-flight file is not this guard's outage to cause.
    It is also the reason this is not a grep — grep has no equivalent of "I
    could not read this", it just does not match.
    """
    try:
        st = path.stat()
        key: tuple[str, int, int] | None = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if key is not None and key in _PARSE_CACHE:
        return _PARSE_CACHE[key]
    try:
        tree: ast.Module | None = ast.parse(_reads(path), filename=rel)
    except (SyntaxError, ValueError, OSError, RecursionError):
        tree = None
    if key is not None:
        if len(_PARSE_CACHE) >= _PARSE_CACHE_CAP:
            _PARSE_CACHE.clear()
        _PARSE_CACHE[key] = tree
    return tree


@dataclass(frozen=True)
class _TurnBuilder:
    """One `RunOptions(...)` call, plus the scope it sits in.

    `hooks_value` is what the call wrote as its `hooks` argument: a bare name,
    the string `"None"` for the literal `None`, or the argument node's class
    name for anything computed. `None` is a *value* meaning "no registry" — the
    shape #534 removed from the worker and autonomy paths — so it is unarmed,
    not absent.

    `scope` and `params` are what make the attribution local: `armed_in_scope`
    is the set of registry names an installer was called on in that same
    function (or at module level), and `params` are the names the enclosing
    function received — a registry handed in by the caller is the caller's to
    arm, and the caller's own call is itself in this scan.
    """
    file: str
    line: int
    function: str
    scope: int
    params: frozenset[str]
    armed_in_scope: frozenset[str]
    hooks_value: str
    has_hooks_kwarg: bool
    #: True when a `**splat` sits in the same call: the keys it contributes are
    #: not visible at the call site, so a splat cannot be counted as arming.
    splatted: bool = False
    #: What the call wrote as `mcp_servers`: a name/expression class, or `""`
    #: when the argument is absent. `RunOptions.mcp_servers` defaults to an
    #: empty dict, so a turn built without it registers no MCP server and
    #: therefore exposes no tier-2/tier-3 sender at all — which is the one
    #: reason a dispatch path may be outside the roster without being a hole.
    mcp_servers_value: str = ""

    @property
    def sender_reachable(self) -> bool:
        """Could a `email_send`-class tool be called on this turn?

        Over-approximating by design: a `**splat` that might carry the key
        counts as reachable, because the safe direction for a guard is to
        report a hole that is not there and have a human say so, never to hide
        one that is. The only build treated as unreachable is the one that
        names no MCP server and offers no splat that could: `app/routers/ide.py`
        with its two single-turn IDE helpers.
        """
        return bool(self.splatted or self.mcp_servers_value)


def _attr_name(node: ast.expr | None) -> str:
    return _attr(node)


def _hooks_written(kw: ast.keyword | None) -> str:
    """The written `hooks` argument as a name, `'None'`, a node class, or `''`."""
    if kw is None:
        return ""
    val = kw.value
    if isinstance(val, ast.Constant) and val.value is None:
        return "None"
    if isinstance(val, ast.Name):
        return val.id
    return type(val).__name__


def _analyze(rel: str, tree: ast.Module) -> list[_TurnBuilder]:
    """Every `RunOptions(...)` in one file, each with its local arming facts.

    One pass, real scopes. The previous revision of this module answered the
    same question with two `in text` substring tests over the whole file — and
    the review rung was right that it could not fail: co-locate the literals
    `HookRegistry()` and `RunOptions(` anywhere in a file, in a docstring or a
    comment, and the file reads as a dispatch path that arms its own gate. This
    reads the syntax: which call built the turn, which name it passed, and
    whether an installer was called on that name in that function.
    """
    scopes: dict[int, set[str]] = {0: set()}
    builders: list[_TurnBuilder] = []
    pending: list[tuple[ast.Call, tuple[int, ...], str, set[str]]] = []

    def walk(node: ast.AST, chain: tuple[int, ...], fname: str,
             params: set[str]) -> None:
        for child in ast.iter_child_nodes(node):
            child_chain, child_fn, child_params = chain, fname, params
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                child_chain = chain + (id(child),)
                child_fn = child.name
                a = child.args
                child_params = params | {p.arg for p in
                                         list(a.posonlyargs) + list(a.args)
                                         + list(a.kwonlyargs)}
                if a.vararg:
                    child_params = child_params | {a.vararg.arg}
                if a.kwarg:
                    child_params = child_params | {a.kwarg.arg}
                scopes.setdefault(id(child), set())
            if isinstance(child, ast.Call):
                callee = _attr_name(child.func)
                if callee == RUN_OPTIONS_CTOR:
                    pending.append((child, child_chain, child_fn,
                                    set(child_params)))
                elif callee in INSTALLERS and child.args and isinstance(
                        child.args[0], ast.Name):
                    # Arm the scope the call sits in. A builder in a *nested*
                    # function reaches it through `chain`, which is how
                    # `eval/run_preserve_thinking_eval.py` arms once in the outer
                    # function and builds its `RunOptions` in a closure.
                    scopes.setdefault(chain[-1], set()).add(child.args[0].id)
            walk(child, child_chain, child_fn, child_params)

    walk(tree, (0,), "module", set())

    for call, chain, fname, params in pending:
        kw = next((k for k in call.keywords if k.arg == "hooks"), None)
        srv = next((k for k in call.keywords if k.arg == "mcp_servers"), None)
        # Every enclosing scope, innermost first, down to module level: Python
        # resolves a name through that whole chain, so a registry armed in an
        # outer function is armed for the closure that hands it to the turn.
        armed: set[str] = set()
        for s in chain:
            armed |= scopes.get(s, set())
        builders.append(_TurnBuilder(
            file=rel, line=call.lineno, function=fname, scope=chain[-1],
            params=frozenset(params), armed_in_scope=frozenset(armed),
            hooks_value=_hooks_written(kw), has_hooks_kwarg=kw is not None,
            splatted=any(k.arg is None for k in call.keywords),
            mcp_servers_value=("" if srv is None else _hooks_written(srv))))
    return builders


def _turn_builders(rel: str, tree: ast.Module) -> list[_TurnBuilder]:
    """Compatibility name for the scan; see `_analyze`."""
    return _analyze(rel, tree)


def _signatures(trees: Mapping[str, ast.Module]) -> dict[str, tuple[str, ...]]:
    """Positional parameter names per function name, across the scanned tree.

    Needed so a positional `f(hooks)` can be credited like `hooks=hooks`; see
    `_caller_armed`. Later definitions win, matching Python's own binding at the
    cost of a shadowed name reading as the last one written.
    """
    out: dict[str, tuple[str, ...]] = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                names = [x.arg for x in list(a.posonlyargs) + list(a.args)]
                if names:
                    out.setdefault(node.name, tuple(names))
    return out


def _caller_armed(tree: ast.Module,
                signatures: Mapping[str, tuple[str, ...]]
                ) -> set[tuple[str, str]]:
    """`(callee, keyword)` for each call that passes on an already-armed registry.

    This is the evidence behind a pass-through builder's exemption, and it is the
    half the review rung's clause-5 finding was about. The attribution the first
    round shipped asked whether the *file* contained an installer call, and both
    halves of that went wrong together: `def build_options(hooks): return
    RunOptions(hooks=hooks)` read as covered with nothing calling it, while
    `app/routers/voice.py` — which builds a spoken turn's options and holds no
    installer call and no registry at all — was never even judged, since the rule
    that could have cleared it also required the two literals `HookRegistry()` and
    `RunOptions(` to sit in the same file.

    Here a parameter is credited only where some call site hands *that parameter*
    a registry an installer ran on, and `_wiring_findings` unions this across every
    file before judging anything — which is what lets a caller in one module
    discharge a builder in another, the seam the voice hole lay in. A helper
    nothing calls, or one called with a registry nobody armed, is reported: for a
    guard the expensive direction is the false clear, since a false report gets
    read and dismissed and a false clear ships the hole.
    """
    scopes: dict[tuple[int, ...], set[str]] = {(): set()}
    out: set[tuple[str, str]] = set()

    def walk(node: ast.AST, chain: tuple[int, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            child_chain = chain
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                child_chain = chain + (id(child),)
                scopes.setdefault(child_chain, set())
            if isinstance(child, ast.Call):
                live: set[str] = set(scopes.get((), set()))
                for depth in range(1, len(child_chain) + 1):
                    live |= scopes.get(child_chain[:depth], set())
                callee = _attr_name(child.func)
                if callee in INSTALLERS and child.args and isinstance(
                        child.args[0], ast.Name):
                    scopes.setdefault(child_chain, set()).add(child.args[0].id)
                elif callee:
                    for kw in child.keywords:
                        if kw.arg and isinstance(kw.value, ast.Name) \
                                and kw.value.id in live:
                            out.add((callee, kw.arg))
                    # Positional: `_worker_run_options(hooks)` passes the registry
                    # the same way `hooks=hooks` does, and crediting only the
                    # keyword form would report the worker path as a hole. Resolved
                    # through the callee's own parameter list when this scan can see
                    # the definition; an unresolvable callee earns nothing, which is
                    # the safe direction.
                    params = signatures.get(callee)
                    if params:
                        for i, pos in enumerate(child.args):
                            if i < len(params) and isinstance(pos, ast.Name) \
                                    and pos.id in live:
                                out.add((callee, params[i]))
            walk(child, child_chain)

    walk(tree, ())
    return out


@dataclass(frozen=True)
class WiringFinding:
    """A dispatch path that can reach a sender tool with the gate unarmed.

    `reason` is one of the `FINDING_*` constants; `detail` says what the call
    actually passed, so a failure is actionable without opening the file.
    """
    file: str
    line: int
    reason: str
    detail: str

    def key(self) -> tuple[str, int, str]:
        return (self.file, self.line, self.reason)


#: A `RunOptions` whose `hooks` argument is a name no installer was called on —
#: `None`, or something computed. The gate is not on this turn's registry.
FINDING_UNARMED = "hooks-not-armed"
#: A `RunOptions` built with no `hooks` argument at all (and no splat that could
#: be supplying it): the turn has no registry, so nothing fires and nothing can
#: be armed. This is the exact shape the file-literal rule was blind to —
#: `app/routers/voice.py` was a spoken turn with no hooks and no installer in
#: the file, and the old finder never even considered the file, because the file
#: contained neither literal it was looking for.
FINDING_NO_REGISTRY = "options-built-without-hooks"
#: A file that builds turns but is not in `GATE_ARM_POINTS`: the roster is a
#: claim about the tree, and a claim nobody re-measures is how every entry in
#: the catalogued 'a check that reads its own missing input' class got written.
FINDING_INCOMPLETE_ROSTER = "dispatch-file-missing-from-arm-point-list"


def _judged(b: _TurnBuilder,
            callers_armed: frozenset[tuple[str, str]] = frozenset()
            ) -> WiringFinding | None:
    """One turn-building call → the finding it raises, or None if it is covered.

    Four ways a call raises nothing, and only four: it registers no MCP server
    and so can reach no sender tool at all (`app/routers/ide.py`'s two
    single-turn helpers — the argument is the evidence, not a comment); the name
    it passed had an installer called on it in the same scope or at module
    level; the name is a parameter of the enclosing function, so the registry
    came from a caller whose own call site is in this same scan *and arms the
    registry it passes*. There is no fifth reading in which a file 'mentions
    hooks'.
    """
    if not b.sender_reachable:
        return None
    if not b.has_hooks_kwarg:
        detail = (f"{RUN_OPTIONS_CTOR}(...) in {b.function}() passes no hooks "
                  "argument" + (", and the only `**splat` in the call supplies "
                                "no hooks key" if b.splatted else "")
                  + f", so the turn has no HookRegistry and "
                    f"{GATE_INSTALLER} can never run on it")
        return WiringFinding(b.file, b.line, FINDING_NO_REGISTRY, detail)
    if b.hooks_value in b.armed_in_scope:
        return None
    if b.hooks_value and b.hooks_value in b.params \
            and (b.function, b.hooks_value) in callers_armed:
        # Pass-through: the registry arrived as an argument, so whoever built it
        # is elsewhere. Credited only by a call site that demonstrably hands this
        # parameter an armed registry — the name of a parameter is not evidence of
        # anything, which is what `_caller_armed` exists to stop. The previous
        # revision stopped at `in b.params`, exempting a helper nothing called.
        return None
    shown = b.hooks_value or ("<computed>" if not b.splatted
                              else "<splat only>")
    return WiringFinding(
        b.file, b.line, FINDING_UNARMED,
        f"{RUN_OPTIONS_CTOR}(...) in {b.function}() passes hooks={shown}, "
        f"which no {GATE_INSTALLER} or {FLOOR_INSTALLER} call is made on in "
        "this scope, and which is not a parameter the caller supplies")


def _wiring_findings(base: Path) -> list[WiringFinding]:
    findings: list[WiringFinding] = []
    trees: dict[str, ast.Module] = {}
    for rel, path in _package_files(base):
        tree = _parse(rel, path)
        if tree is None:
            continue
        trees[rel] = tree
    # Union before judging: a pass-through builder is discharged by a caller this
    # scan can see, and that caller may live in another module.
    signatures = _signatures(trees)
    callers_armed = frozenset(set().union(
        *(_caller_armed(t, signatures) for t in trees.values())) if trees else ())
    for rel, tree in trees.items():
        builders = [b for b in _analyze(rel, tree) if b.sender_reachable]
        if not builders:
            continue
        if rel not in GATE_ARM_POINTS:
            findings.append(WiringFinding(
                rel, builders[0].line, FINDING_INCOMPLETE_ROSTER,
                f"{len(builders)} turn build(er) in a file the gate's roster "
                "does not name"))
        for b in builders:
            judged = _judged(b, callers_armed)
            if judged is not None:
                findings.append(judged)
    return findings


def unarmed_dispatch_paths(root: str | Path | None = None
                           ) -> list[WiringFinding]:
    """Every dispatch path the gate does not cover, with the reason.

    Non-empty here means a tier-2/tier-3 side effect is reachable with no
    content check. The pin-count guard fails the suite on this output, which is
    what makes the guard unfakeable: a new dispatch path either arms the gate or
    is refused at the review rung, instead of shipping a hole no existing test
    could see — the exact shape #869 found in three other guards.
    """
    base = Path(str(root)) if root else Path(__file__).resolve().parents[2]
    return [f for f in _wiring_findings(base)
            if f.reason in (FINDING_UNARMED, FINDING_NO_REGISTRY)]


def find_unarmed_dispatch_paths(root: str | Path | None = None
                                ) -> list[tuple[str, int]]:
    """The same findings, reduced to `(file, line)` for the guard's assertion.

    Kept as the tuple form because the assertion reads as a list of places to
    fix; `unarmed_dispatch_paths` is what a failure message should print.
    """
    return [(f.file, f.line) for f in unarmed_dispatch_paths(root)]


def all_turn_builds(root: str | Path | None = None) -> list[_TurnBuilder]:
    """Every `RunOptions(...)` build in the package roots, armed or not.

    The unfiltered denominator, kept public so a test can show a sender-unreachable
    build being excluded rather than assert the exclusion in prose.
    """
    base = Path(str(root)) if root else Path(__file__).resolve().parents[2]
    out: list[_TurnBuilder] = []
    for rel, path in _package_files(base):
        tree = _parse(rel, path)
        if tree is None:
            continue
        out.extend(_turn_builders(rel, tree))
    return out


def dispatch_registry_sites(root: str | Path | None = None
                            ) -> list[tuple[str, int]]:
    """Every production dispatch site a tier-2/tier-3 sender is reachable from.

    The denominator the arm-point roster has to cover. A build that registers no
    MCP server is not in it — see `_TurnBuilder.sender_reachable` — because
    listing a turn that cannot address a sender tool is how a roster becomes
    decoration. The line is the `RunOptions` call itself, so a failure names the
    place to fix rather than the module.
    """
    return sorted((b.file, b.line) for b in all_turn_builds(root)
                  if b.sender_reachable)


def dispatch_registry_sites_files(root: str | Path | None = None) -> set[str]:
    """Just the file names, for set comparisons in the guard tests."""
    return {rel for rel, _line in dispatch_registry_sites(root)}


def armed_files(root: str | Path | None = None) -> set[str]:
    """Dispatch files whose every turn build passes an armed registry.

    `install_default_safety_hook(hooks)` counts as arming because the gate is
    installed inside that function — one call, every turn — which is the
    mechanism that keeps this from becoming five per-call-site conventions.
    """
    base = Path(str(root)) if root else Path(__file__).resolve().parents[2]
    bad = {f.file for f in _wiring_findings(base)
           if f.reason in (FINDING_UNARMED, FINDING_NO_REGISTRY)}
    return dispatch_registry_sites_files(base) - bad


def dispatch_files_missing_from_arm_points(root: str | Path | None = None
                                           ) -> list[str]:
    """Files that build turns but are not named in `GATE_ARM_POINTS`."""
    base = Path(str(root)) if root else Path(__file__).resolve().parents[2]
    return sorted({f.file for f in _wiring_findings(base)
                   if f.reason == FINDING_INCOMPLETE_ROSTER})


def _calls_installer(rel: str, path: Path) -> bool:
    """Does this file contain a *call* to either installer (AST, not a grep)?

    A grep for `install_default_safety_hook(` is satisfied by the module that
    *defines* it, which would let the roster keep an entry whose only remaining
    mention of the installer is the definition nobody reaches. `def` is not `call`.
    """
    tree = _parse(rel, path)
    if tree is None:
        return False
    return any(isinstance(n, ast.Call) and _attr(n.func) in INSTALLERS
               for n in ast.walk(tree))


def sender_unreachable_dispatch_files(root: str | Path | None = None
                                      ) -> list[str]:
    """Dispatch files whose every turn build registers no MCP server.

    Derived, never listed: see the note above `GATE_ARM_POINTS`. These are the
    files that may sit outside the roster without being holes, and only because
    the syntax says their turns cannot address a sender tool.
    """
    base = Path(str(root)) if root else Path(__file__).resolve().parents[2]
    out: list[str] = []
    for rel, path in _package_files(base):
        tree = _parse(rel, path)
        if tree is None:
            continue
        builders = _turn_builders(rel, tree)
        if builders and not any(b.sender_reachable for b in builders):
            out.append(rel)
    return sorted(out)


def stale_gate_arm_points(root: str | Path | None = None) -> list[str]:
    """Roster entries that no longer earn their place.

    An entry is stale when the file no longer calls an installer, when it calls
    one and still leaves one of its own turn builds unarmed, or when none of its
    turn builds can reach a sender tool at all — the third reading being the one
    that keeps `app/routers/ide.py` off a roster whose entries are supposed to be
    places a secret can leave. The floor module is exempt from the third test
    because it arms by being called from the other entries, not by building
    turns of its own.

    Every clause here is a re-measurement of the tree in the same call that
    prints the verdict, which is the standing rule for a counting guard: a
    baseline the guard itself wrote is not a baseline.
    """
    base = Path(str(root)) if root else Path(__file__).resolve().parents[2]
    unreachable = set(sender_unreachable_dispatch_files(base))
    builds_turns = {rel for rel, line in dispatch_registry_sites(base)}
    armed = armed_files(base)
    findings = _wiring_findings(base)
    bad = {f.file for f in findings
           if f.reason in (FINDING_UNARMED, FINDING_NO_REGISTRY)}
    stale: list[str] = []
    for rel in GATE_ARM_POINTS:
        path = base / rel
        if not _calls_installer(rel, path):
            stale.append(rel)
            continue
        if rel in bad or (rel in builds_turns and rel not in armed):
            stale.append(rel)
            continue
        if rel != FLOOR_MODULE and rel in unreachable:
            stale.append(rel)
    return stale


def _deny(reason: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


#: Rule name → its plain-English description. `Finding` stays lean (it goes to
#: the ledger, where a repeated paragraph per row is noise) and the refusal
#: message still explains itself: this lookup, not a field on the finding.
DESCRIPTIONS: dict[str, str] = {rule.name: rule.description for rule in CONTENT_RULES}


def deny_reason(tool_name: str, findings: Sequence[Finding]) -> str:
    """Names the rule, the argument and the severity, and says what to do.

    A refusal that does not name its rule cannot be acted on: the caller has
    to guess which of its own strings was the problem, and the guess is
    usually 'remove the content the task needed'. So the argument path is
    here too.

    Nothing here may raise. The callback calls this on its deny path, and
    `HookRegistry.fire_pre_tool_use` swallows a raising callback and keeps
    going — so an exception in the refusal message would open the gate it is
    reporting through. Unknown rule names therefore degrade to an empty
    description rather than an AttributeError.
    """
    blocking = [f for f in findings if f.action == "deny"]
    first = blocking[0] if blocking else findings[0]
    more = (f" (+{len(blocking) - 1} more credential match(es): "
            + ", ".join(sorted({f.rule for f in blocking[1:]})) + ")") \
        if len(blocking) > 1 else ""
    description = DESCRIPTIONS.get(first.rule, "")
    explained = f" — {description}" if description else ""
    return (
        f"outbound-content: rule '{first.rule}' ({first.severity}/"
        f"{first.category}) matched argument '{first.arg}' "
        f"of '{tool_name}'{explained}{more} "
        f"Redacted match: {first.excerpt}. The call was not made. Remove the "
        f"matched content (or reference the file instead of inlining it) and "
        f"send again; the rule table is app/harness/outbound_content.py and a "
        f"scope-level exemption needs a written reason in EXEMPT_SCOPES."
    )


def install_outbound_content_gate(
    hooks: HookRegistry, *, scope: str | None = None,
    findings_ledger: str | Path | None = None,
    exempt_scopes: Mapping[str, str] | None = None,
) -> None:
    """Refuse credential-shaped payloads in tier-2/tier-3 sender arguments.

    `scope` names whose authority this turn is borrowing, exactly as
    `install_policy_hook` takes it; omit it and the callback reads
    `policy.current_scope`, which the worker pool and the autonomy runner bind
    per job. The gate is armed by `install_default_safety_hook` for every turn
    that installs the default floor, and explicitly at the two dispatch paths
    (`autonomy.run_task`, `workers/sources/_common.py`) that build a registry
    without it — `GATE_ARM_POINTS` is the list, and it is pinned by a test.

    Like the policy hook, this callback *denies* rather than raises:
    `HookRegistry.fire_pre_tool_use` treats a raising callback as a pass,
    which is right for an observer and exactly wrong here — a broken scan
    would otherwise open the gate it was meant to close.
    """

    async def _content_pretool_cb(input_data: dict[str, Any],
                                  _tool_use_id: str | None,
                                  _ctx: Any) -> dict:
        tool_name = str(input_data.get("tool_name", ""))
        # Tier 1 tools have no durable recipient; scanning them is the
        # false-positive surface that gets a gate turned off.
        #
        # Per-call, not per-name. `autonomy_write_task` went tier 2 with #724
        # because six of its short fields decide whether a task runs; the same
        # tool also carries the run-record note the skills tell every unattended
        # run to append to its own task file — free prose, no recipient,
        # deliberately outside the grant gate. Arming on the name would put this
        # fail-closed scan in front of that note, and would answer tier 1 for a
        # call that `check_grants`, on the same registry with the same
        # arguments, answers tier 2. `effective_tier` is the one number both
        # gates read for one call.
        if effective_tier(tool_name, input_data.get("tool_input") or {}) < 2:
            return {}
        active_scope = str(scope or current_scope.get() or "")
        reason_exempt = is_exempt(active_scope, exempt_scopes)
        if reason_exempt:
            logger.debug("[outbound-content] skipped scope=%s tool=%s — %s",
                         active_scope, tool_name, reason_exempt)
            return {}
        try:
            findings = scan_outbound_payload(input_data.get("tool_input") or {})
        except Exception as exc:  # noqa: BLE001 — fail closed, see docstring
            return _deny(
                "outbound-content: the content scan could not be evaluated "
                f"({exc.__class__.__name__}: {exc}); '{tool_name}' from scope "
                f"'{active_scope or 'unknown'}' is denied while the scan is "
                "unusable — detection failing open here would be the "
                "fail-open this gate exists to remove")
        if not findings:
            return {}
        blocking = [f for f in findings if f.action == "deny"]
        # The ledger records both halves, one row per call: an allowed call
        # that carried PII, and a refused one. Refused calls are written too
        # because on an unattended path nobody sees the refusal unless it is
        # somewhere a person later reads, and `denied: true` is the whole
        # query for 'what has this gate stopped'.
        record_findings(findings, scope=active_scope, tool=tool_name,
                        denied=bool(blocking), path=findings_ledger)
        if not blocking:
            return {}
        reason = deny_reason(tool_name, blocking)
        logger.warning("[outbound-content] denied tool=%s scope=%s rules=%s",
                       tool_name, active_scope or "unknown",
                       sorted({f.rule for f in blocking}))
        return _deny(reason)

    hooks.add_pre_tool_use(None, _content_pretool_cb, fail_closed=True)
