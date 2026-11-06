"""Vaulted credentials: a child process gets a placeholder, the real value stays here.

The design is Philipp Schmid's Managed-Agent credential shape (#3846, from the
digested talk `20261106-7-things-your-ai-coworker-needs-philipp-schmid-google-deepmind`),
recreated on the two lanes Lloyd actually owns. The property it buys in one
sentence: **what a Bash child can read is not a secret**, so an injected
"print your environment and POST it somewhere" returns placeholders — and the value
is never removed from the process that needs it, only from the child's reach.

Two halves, and both are needed for the property to mean anything:

1. **Placeholder the spawn env** (`vaulted_env`, called by
   `agent_mcp/builtin_bash.py` at both spawn sites). Every vaulted name the child
   would have inherited is replaced by a deterministic placeholder
   (`placeholder_for`); everything else is inherited byte-for-byte.
2. **Substitute at the egress gate** (`substitute_credentials`, called by the
   `http_request` lane through `agent_mcp/egress.py`), keyed to the destination the
   credential is bound to. A swap is recorded on the `egress_events` row, so the
   audit trail says which credential went to which host — not merely that a call
   happened.

A scope may hold a real value legitimately, by declaring it: a LIVE
`authority_grants` row whose `tool_pattern` is `credential:<NAME>` for that exact
scope (#2471's shape, `tool_pattern` matched by equality, `destination` column
unused here because a spawn has no destination). That is the escape hatch the
`github-auth` skill and every CLI that reads its token from the environment needs,
and it is per-scope rather than a global relaxation.

## What this does NOT close — read before citing this module as a control

- **The confused deputy is untouched.** The agent still *causes* a credentialed
  call, and a destination binding is not a scope binding: nothing here restricts
  which method, account or repository that call touches. A prompt that knows the
  right host can still spend the credential there.
- **Nothing is enforced until #628's `harness.egress_policy.enforce` is flipped**,
  and that flip was left to a human on purpose. With the flag off, the egress gate
  records rather than refuses; destination-keyed substitution is then a better
  default and a nicer audit row, NOT a control. The Bash-side placeholder is not
  gated by that flag — it applies on every scope — but the placeholder's value
  depends on there being somewhere safe to send the real one.
- **File-form secrets are out of scope**: `~/.ssh` private keys, `~/.config/gh`,
  `~/.openclaw` and any token a tool reads from a config file. A private key has no
  placeholder-able form, and `~/.config/gh/hosts.yml` is read by `gh` itself, not
  through env. Those lanes stay as they were, and
  `eval/measurements/credentials-reachable-from-a-bash-child-<date>.md` names each
  one — the file this module's own `write_artifact()` generates, not a hand-written
  summary of it.
- **`config.yaml` is a second credential lane** this module does not move: the
  `${LIVEKIT_API_KEY}` / `${LIVEKIT_API_SECRET}` expansions resolve inside
  `app/config.py`, so anything reading the parsed config gets the real value
  regardless of what the child's environment says. That is a named residual in the
  census, not a silent gap.
- **This is not TLS interception.** Only the lanes that hand headers to this module
  can be swapped; `curl` inside Bash gets the placeholder and no proxy exists to
  upgrade it. That is deliberate — building a MITM proxy for CLI traffic is a
  bigger and riskier change than this item, and the placeholder means the CLI path
  fails loudly rather than quietly succeeding with a real token.

Fail-safe direction, throughout: an unreadable registry widens NOTHING and leaks
nothing — every secret-shaped name still gets a placeholder, because the name set
is derived from the shape of the name (`_SECRET_SHAPE`) and not from a file that
might be missing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path

#: The registry: credential name → its bindings. Deliberately holds NO secret
#: material — only names, destination bindings and a note — so landing this module
#: does not duplicate a secret onto disk. The real values stay in `~/lloyd/.env`
#: (already a `protected_read_roots` entry) and in the aggregator's own environment.
#: Its path is a `protected_read_roots` entry too, so the agent's read tools refuse
#: it; what they would learn from it is which credentials exist and where they are
#: allowed, which is a map of the attack surface and no part of a tool's job.
STORE_TEMPLATE = "~/lloyd-data/secrets/credentials.json"

#: Prefix on the placeholder. Not a secret, not random, and unmistakable in an
#: audit log or a `docker exec` shell: a reader who sees it knows the name is
#: vaulted rather than unset, which is the difference between "misconfigured" and
#: "protected" when a legitimate consumer fails.
PLACEHOLDER_PREFIX = "lloyd:vaulted:"

#: The grant prefix on the existing `authority_grants.tool_pattern` column. A row
#: with `tool_pattern = "credential:LIVEKIT_API_SECRET"` is the only way a scope
#: gets that value for real.
GRANT_PREFIX = "credential:"

#: Names ending in one of these are secrets by shape. The set is what makes the
#: placeholder fail-safe: a credential added to `~/lloyd/.env` tomorrow is
#: placeholder-ed tomorrow, with no code change and no registry edit, because
#: forgetting to edit a list is exactly how a leak like #3846's survives a fix.
_SECRET_SHAPE = re.compile(
    r"(?:^|_)(?:KEY|TOKEN|SECRET|SECRETS|PASSWORD|PASSWD|PASSPHRASE|CREDENTIAL"
    r"|CREDENTIALS|APIKEY|ACCESS_KEY|PRIVATE_KEY)$", re.IGNORECASE)

#: Names that are vaulted-shaped but provably not credentials, so placeholder-ing
#: them would break a consumer for no gain. Each entry needs a reason here, and
#: `tests/test_credential_census.py` requires this set to stay named in the census.
NOT_A_CREDENTIAL: frozenset[str] = frozenset({
    # A URL, and the only one in `~/lloyd/.env`: `ANTHROPIC_BASE_URL` names the
    # gateway the agent is *told* to talk to and is echoed in the tool schema.
    "ANTHROPIC_BASE_URL",
    # Model-selection strings, not credentials. Both are printed in dashboards.
    "ANTHROPIC_CUSTOM_MODEL_OPTION",
    "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME",
})


def store_path() -> Path:
    return Path(os.path.expanduser(STORE_TEMPLATE))


def env_file_path() -> Path:
    """`~/lloyd/.env`, the file `app/config.py` loads into the process env."""
    return Path.home() / "lloyd" / ".env"


def env_file_values() -> dict[str, str]:
    """Parse `~/lloyd/.env` without importing `app.config` (which would run the
    whole config loader, including its side-effecting `os.environ.setdefault`)."""
    path = env_file_path()
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key:
            out[key] = value.strip().strip("\"'")
    return out


def registry() -> dict[str, dict]:
    """The registry file, or `{}` when there is none. Never raises.

    Malformed JSON is `{}` rather than a partial parse: an unreadable registry must
    leave the placeholder behaviour intact (it comes from `_SECRET_SHAPE`, not from
    here) and widen no grant.
    """
    try:
        data = json.loads(store_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    creds = data.get("credentials")
    if not isinstance(creds, dict):
        return {}
    return {str(k): (v if isinstance(v, dict) else {}) for k, v in creds.items()}


def secret_shaped(name: str) -> bool:
    """Whether a variable NAME is a credential by shape."""
    return bool(_SECRET_SHAPE.search(str(name or "")))


def _env_file_secret_names() -> tuple[str, ...]:
    """The secret-shaped names in the env file, re-read on every call.

    NOT cached, and the absence is load-bearing in two directions. A cached read
    of a mutable secret file is a fail-open with a long latency: a name added to
    `~/lloyd/.env` after a long-lived process started would never be vaulted by
    that process, which is exactly the "a credential added tomorrow is placeholder-ed
    tomorrow" promise the module docstring makes. And a cache keyed on nothing at all
    cannot survive a changed `$HOME`, so the vault could not be tested against bytes
    a test wrote — every node would measure whichever home the first caller happened
    to see. The file is six lines; this runs once per Bash spawn.
    """
    return tuple(sorted(n for n in env_file_values()
                        if secret_shaped(n) and n not in NOT_A_CREDENTIAL))


def vaulted_names() -> frozenset[str]:
    """Every name whose value must never reach a child.

    Three sources, unioned, so no one of them being missing opens the guard:

    1. the registry, for credentials with destination bindings;
    2. the secret-shaped names in `~/lloyd/.env`;
    3. the secret-shaped names in THIS process's environment.

    Source 3 is the one that makes the guard fail closed rather than fail open. A
    secret can sit in the environment without being in the file — set by a
    supervisor `environment=` line, exported by a launcher, or simply present
    because `app/config.py` loaded the file into `os.environ` and a later root
    change made the file unreachable. The thing a Bash child inherits is the
    environment, so a vault defined only over the FILE would placeholder nothing
    while looking configured. Derived from NAME shape, never from a value heuristic:
    an empty or obvious-fixture value under a credential name is still vaulted,
    because deciding by value would let a real secret keep its name by being long.
    """
    return frozenset(set(registry())
                     | set(_env_file_secret_names())
                     | {n for n in os.environ if secret_shaped(n)
                        and n not in NOT_A_CREDENTIAL})


def placeholder_for(name: str) -> str:
    """The deterministic stand-in for `name`: derived from the NAME, never the value.

    Deterministic on purpose. A per-boot random token would make a child's failure
    unreproducible, and a placeholder derived from the value would put the secret in
    a hash the census then has to reason about. Truncated to 16 hex characters:
    long enough that nothing collides by accident, short enough to read in `env`
    output.
    """
    return PLACEHOLDER_PREFIX + hashlib.sha256(str(name).encode()).hexdigest()[:16]


def is_placeholder(value: str) -> bool:
    return str(value or "").startswith(PLACEHOLDER_PREFIX)


#: A placeholder appearing inside a larger value. This is the shape that actually
#: reaches the egress lane: `Authorization: Bearer lloyd:vaulted:…`. Matching only
#: a whole value would make the substitution fire on the one header form nobody
#: writes and never on the one everybody does.
_PLACEHOLDER_TOKEN_RE = re.compile(re.escape(PLACEHOLDER_PREFIX) + r"[0-9a-f]{16}")


def placeholder_tokens(value: str) -> list[str]:
    """Every placeholder token inside one value, in order of appearance.

    A value can hold two of them (a mTLS-style header carrying both a key and a
    secret), which is why this returns a list rather than the first hit: substituting
    only the first would send a half-real credential and record one swap for it.
    """
    return _PLACEHOLDER_TOKEN_RE.findall(str(value or ""))


def name_of_placeholder(value: str) -> str | None:
    """Which credential a placeholder stands for, or None if it is not one.

    Solved by recomputing rather than by decoding: the digest is one-way by
    construction, so the only reader is the one that already knows the name set.
    """
    if not is_placeholder(value):
        return None
    for name in sorted(vaulted_names()):
        if placeholder_for(name) == value:
            return name
    return None


def real_value(name: str) -> str | None:
    """The live value: the aggregator's environment first, then `~/lloyd/.env`.

    Environment-first is what keeps a supervisord-set value winning over the file,
    the same precedence `app/config.py`'s `_load_env_file` documents. Reading the
    file second is what makes this work in a process that started before the file
    was loaded (a test, a one-shot script) rather than reporting "unset".
    """
    value = os.environ.get(name)
    if value:
        return value
    return env_file_values().get(name) or None


def destinations_for(name: str) -> tuple[str, ...]:
    """`scheme://host` bindings this credential may be substituted for.

    Empty by default, which is fail-closed: a credential with no binding is never
    swapped in, so adding a name to the registry without a destination cannot leak
    anything.
    """
    raw = registry().get(name, {}).get("destinations") or []
    if isinstance(raw, str):
        raw = [raw]
    return tuple(str(d).strip().rstrip("/") for d in raw if str(d).strip())


def bound_for(name: str, destination: str) -> bool:
    """Whether `destination` matches the binding, on the same key the egress gate uses.

    Exact `scheme://host` comparison — no suffix or substring match, because
    `https://api.example.com` must not authorise `https://evil.example.com`, which
    is the `endswith` bug in every hand-rolled host check.
    """
    dest = str(destination or "").strip().rstrip("/")
    return bool(dest) and dest in destinations_for(name)


def granted_credentials(scope: str) -> frozenset[str]:
    """Names a live grant gives THIS scope, from `authority_grants.tool_pattern`.

    Fail-safe: an unreadable or absent store grants nothing, which means a child
    gets placeholders — the direction that leaks nothing. An empty scope grants
    nothing either, so an attended chat turn with no row sees placeholders.
    """
    if not scope:
        return frozenset()
    try:
        from app.harness.policy import default_store
        rows = default_store().live(scope=scope)
    except Exception:  # noqa: BLE001 - a dead store is the no-grant case, by design
        return frozenset()
    return frozenset(
        str(r.get("tool_pattern") or "").strip()[len(GRANT_PREFIX):]
        for r in rows
        if str(r.get("tool_pattern") or "").startswith(GRANT_PREFIX))


def vaulted_env(env: "dict[str, str] | None", *, scope: str = "",
                granted: "frozenset[str] | None" = None
                ) -> "dict[str, str] | None":
    """The spawn environment: vaulted names placeholder-ed, everything else untouched.

    `None` means "inherit the process environment", the same convention
    `_resource_ceiling.launcher_env` and `_path_sandbox.child_env` use — so it is
    materialised FROM `os.environ` rather than returned as a partial dict. #3768's
    lesson is the reason: a one-key environment handed to `create_subprocess_exec`
    strips `PATH` from every Bash child on the machine, and the `systemd-run`
    launcher then fails open into an unbounded spawn.

    A granted name gets its real value, so `gh` and the nightly writers keep
    working in a scope that declares them. Unset vaulted names are NOT created: a
    placeholder is written only where a value would otherwise have been inherited,
    so this function never invents a variable.
    """
    names = vaulted_names()
    if granted is None:
        granted = granted_credentials(scope)
    base = dict(os.environ if env is None else env)
    for name in names:
        if name not in base:
            continue
        if name in granted:
            live = real_value(name)
            if live:
                base[name] = live
            continue
        base[name] = placeholder_for(name)
    return base


def substitute_headers(headers: "dict[str, str] | None", destination: str
                       ) -> "tuple[dict[str, str], list[tuple[str, str]]]":
    """Swap placeholder → real where the destination is bound. Returns the headers to
    actually send and `[(credential, decision)]` for the audit column.

    Decisions are recorded even when nothing is swapped — `out-of-binding` is the
    row that shows an attempt to spend a credential somewhere it was not bound,
    which is the signal a human reading this table wants most. Nothing is refused
    here: refusal is the destination gate's job (#628), and with `enforce` off this
    module only decides which bytes go on the wire.
    """
    out = dict(headers or {})
    swaps: list[tuple[str, str]] = []
    for key, value in list(out.items()):
        # Substitution is per TOKEN, not per header, because the credential arrives
        # inside a larger value ("Bearer <placeholder>", "Basic <placeholder>") and
        # the transport needs the rest of that value left alone.
        for token in placeholder_tokens(value):
            name = name_of_placeholder(token)
            if name is None:
                # A prefix-shaped string too short, too long or too corrupt to be
                # any credential's: leave it on the wire. Recording it as a swap
                # would put a name on a row that names nothing real.
                continue
            if bound_for(name, destination):
                live = real_value(name)
                if live:
                    out[key] = str(out[key]).replace(token, live)
                    swaps.append((name, "applied"))
                else:
                    # The placeholder is real but the vault has no value: send the
                    # placeholder and say so, rather than dropping the header into a
                    # 401 that names nothing.
                    swaps.append((name, "no-value"))
            else:
                swaps.append((name, "out-of-binding"))
    return out, swaps


# ---------------------------------------------------------------- census

#: Keys whose VALUE is a credential when it is a literal string. Deliberately
#: narrower than SECRET_NAME_RE: a config key called `url` or `room_prefix` next to
#: a token is not one, and calling it one would put the census's second lane full
#: of non-secrets and teach a reader to skim the list.
CONFIG_SECRET_KEYS = ("token", "api_key", "api_secret", "secret", "password",
                      "passwd", "passphrase", "client_secret", "app_secret",
                      "webhook_secret", "access_token", "auth_token",
                      "refresh_token")

#: An indirected value names an environment variable in dollars-and-braces form,
#: resolved at load (`app/config.py::_expand_vars`). Naming the indirection is what
#: makes the census honest: an indirected key is NOT a second copy of the secret,
#: and a reader who thinks it is will go looking for a literal that does not exist.
_INDIRECTION = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def config_secret_literals(path: Path | None = None) -> list[dict]:
    """Every literal credential written into `config.yaml`, with its line number.

    Read with a regex over the raw text rather than through `yaml.safe_load`
    because the answer is a line number, and a parse tree does not carry one. The
    scan reports `` indirections as a separate kind, since the thing to fix
    for those is the env lane, not the file.
    """
    p = path or (Path(__file__).resolve().parents[2] / "config.yaml")
    out: list[dict] = []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    key_re = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")
    for no, line in enumerate(lines, 1):
        if line.lstrip().startswith("#"):
            continue
        m = key_re.match(line)
        if not m or m.group(2) not in CONFIG_SECRET_KEYS:
            continue
        raw = m.group(3).strip().strip("'\"")
        ind = _INDIRECTION.match(raw)
        if ind:
            out.append({"line": no, "key": m.group(2), "kind": "indirection",
                        "names": ind.group(1)})
        elif raw:
            out.append({"line": no, "key": m.group(2), "kind": "literal",
                        "length": len(raw)})
        else:
            out.append({"line": no, "key": m.group(2), "kind": "empty"})
    return out


def measure() -> dict:
    """The census as data: names, counts and line numbers — never a value.

    This is the function both the dated artifact and `tests/test_credential_census.py`
    read, so the prose and the test cannot disagree about what was counted. The
    hard rule inside it: no returned field may contain a secret. Values are
    reported as a length and, where a comparison is needed, as a sha256 prefix —
    which is enough to prove a child received the real thing and is what a census
    file that a model can read has to be limited to.
    """
    import hashlib
    env_vals = env_file_values()
    child_secret_names = sorted(k for k in os.environ if secret_shaped(k))
    vaulted = sorted(vaulted_names())
    return {
        "generated_by": "app.harness.credential_store measure (#3846)",
        "env_file_name_count": len(env_vals),
        "env_file_secret_names": sorted(_env_file_secret_names()),
        "env_file_other_names": sorted(
            k for k in env_vals if not secret_shaped(k)),
        "value_lengths": {k: len(v) for k, v in sorted(env_vals.items())},
        "value_hash_prefixes": {
            k: hashlib.sha256(v.encode()).hexdigest()[:12]
            for k, v in sorted(env_vals.items())},
        "child_secret_shaped_names": child_secret_names,
        "child_total_vars": len(os.environ),
        "vaulted_names": vaulted,
        "config_secret_literals": [
            r for r in config_secret_literals() if r["kind"] == "literal"],
        "config_indirections": [
            r for r in config_secret_literals() if r["kind"] == "indirection"],
        "registry_path": str(store_path()),
        "registry_exists": store_path().exists(),
        "registry_read_denied": _registry_is_read_denied(),
        "spawn_branches_vaulted": _vaulted_spawn_branches(),
    }


def _registry_is_read_denied() -> bool:
    """Whether the registry's own path is refused by the read-deny rail.

    Asked of `read_deny_reason`, the exact predicate the rail calls, rather than of
    the root list — a census entry that re-implemented the matching would keep
    passing if the matching changed. Imported inside the function because
    `app.harness.protected_paths` imports THIS module for one constant, so a
    module-level import here would be a cycle.
    """
    from app.harness.protected_paths import read_deny_reason
    return read_deny_reason(str(store_path())) is not None


def _vaulted_spawn_branches() -> list[str]:
    """Which spawn sites route their env through the placeholder helper.

    Read from the source text, because the thing worth pinning is that there are
    three call sites and none of them was skipped — an import-time check would
    only prove the module loaded.
    """
    src = (Path(__file__).resolve().parents[2] / "agent_mcp"
           / "builtin_bash.py").read_text(encoding="utf-8")
    return [name for name, present in (
        ("sandboxed-foreground", "env=_vault_for_child(_resource_ceiling"
                                 ".launcher_env(None))"),
        ("foreground", "env=_vault_for_child("),
        ("background", "env = _vault_for_child(env)"),
    ) if present in src]


#: Per-name answers to the two questions clause 1 asks of every row: what reads
#: this value today, and whether a substitution lane could serve that consumer
#: instead. Keyed by NAME because the census is a measurement of THIS box; a name
#: that shows up in the environment and is not looked up renders as `UNRESOLVED`
#: rather than as an empty cell, so an un-enumerated credential is visible in the
#: artifact instead of being the silence clause 1 exists to prevent.
UNRESOLVED_CONSUMER = ("UNRESOLVED — this census has not enumerated a reader for "
                       "it; treat it as an open question, not as an unused name")
UNRESOLVED_LANE = ("UNRESOLVED — no lane judgement recorded; the egress header "
                   "swap serves only credentials bound to one `scheme://host`")
CONSUMERS: dict[str, tuple[str, str]] = {
    "ANTHROPIC_API_KEY": (
        "The aggregator's own LLM client (server-side), plus any CLI a turn invokes",
        "Yes for egress-shaped use — a header swap bound to one host. No for a CLI "
        "that reads its token from env"),
    "LIVEKIT_API_KEY": (
        "`agent-services/livekit_worker.py`, `app/routers/voice.py`, the voice e2e "
        "scripts",
        "No — a LiveKit key signs SDK tokens; it is never a header on one host, so a "
        "grant is its only lane"),
    "LIVEKIT_API_SECRET": (
        "The same three readers; signs room tokens",
        "No — same reason; a grant is its only lane"),
}

ARTIFACT_STEM = "credentials-reachable-from-a-bash-child"


def census_rows(m: "dict | None" = None) -> list[dict]:
    """One row per vaulted name: the name, its length, its consumer, its judgement.

    The unit is the NAME, never the file — `~/lloyd/.env` is a config file that
    happens to hold three secrets, and a census of the file would list a base URL as
    a credential. Names that come from the environment rather than the file get a
    row too, which is the point: the child inherits the environment.
    """
    m = m or measure()
    lengths = m["value_lengths"]
    rows = []
    for name in m["vaulted_names"]:
        consumer, lane = CONSUMERS.get(name, (UNRESOLVED_CONSUMER, UNRESOLVED_LANE))
        rows.append({"name": name,
                     "reached_before": "real value",
                     "length": lengths.get(name, len(os.environ.get(name, ""))),
                     "consumer": consumer,
                     "lane": lane,
                     "after": "Placeholder; real value only on a live grant"})
    return rows


def render_artifact(m: "dict | None" = None,
                    today: "str | None" = None) -> str:
    """The census as a dated markdown artifact — the file clause 1's deliverable is.

    Generated rather than hand-written so the table cannot disagree with `measure()`:
    the version of this census that was wrong about its own headline number was a
    prose file, and prose is what a re-run cannot correct. The named residuals are
    prose still, deliberately — they are scope decisions, and a generator that
    invented them would be asserting a judgement nothing measured.
    """
    m = m or measure()
    day = today or date.today().isoformat()
    rows = census_rows(m)
    skipped = m["env_file_other_names"]
    out = [f"# Credentials reachable from a Bash child — measured {day}", "",
           f"Generated by `{m['generated_by']}`; it prints names, counts, line "
           "numbers and sha256 prefixes and NEVER a value. Re-run:", "",
           "    cd ~/lloyd && .venvs/lloyd/bin/python -m app.harness."
           "credential_store --write", "",
           f"## The env lane: {len(rows)} vaulted name(s) inside a "
           f"{m['child_total_vars']}-variable child environment", "",
           "| Env name | Reached a child before the fix | Value length | Consumer "
           "today | Can a substitution lane serve it? | After the fix |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| `{r['name']}` | {r['reached_before']} | {r['length']} | "
                   f"{r['consumer']} | {r['lane']} | {r['after']} |")
    out += ["", f"`{env_file_path()}` supplies {m['env_file_name_count']} name(s); "
            f"{len(m['env_file_secret_names'])} are secret-shaped and vaulted. The "
            "rest share the file without being credentials, so the child keeps them "
            "unchanged, and each is named rather than omitted:", ""]
    for name in skipped:
        out.append(f"- `{name}` — configuration, listed in `NOT_A_CREDENTIAL`")
    out += ["", "## Lane 2 — `config.yaml`", "",
            f"Literal credentials in the file: **{len(m['config_secret_literals'])}**"
            " (a raw-text scan, because the answer needed is a line number). "
            "Indirected `${NAME}` values, which resolve at config LOAD against the "
            "same environment this census placeholder-s:", ""]
    for r in m["config_indirections"]:
        out.append(f"- `config.yaml:{r['line']}` key `{r['key']}` → env name "
                   f"`{r['names']}` — a second REFERENCE to a name listed above, "
                   "not a second copy of a secret")
    if not m["config_indirections"]:
        out.append("- none")
    out += ["", "## Registry", "",
            f"- path `{m['registry_path']}` — exists: {m['registry_exists']}; "
            f"refused by the read-deny rail: {m['registry_read_denied']}",
            f"- spawn sites routing their env through the placeholder: "
            f"{_branch_list(m['spawn_branches_vaulted'])}", "",
            "## Named residuals", "",
            "1. **File-form secrets** (`~/.ssh`, `~/.config/gh`, any CLI token "
            "cache): no env var holds them, so there is no placeholder to "
            "substitute. Out of scope, named so the module docstring cannot be read "
            "as claiming credential isolation generally.",
            "2. **The confused deputy**: a destination binding says WHERE a "
            "credential may be spent, not which account or method it may touch.",
            "3. **`harness.egress_policy.enforce` is off** (#628, left to a human): "
            "until it is on, the destination gate records rather than refuses, so "
            "substitution is a reduced-blast-radius default and not yet a control.",
            ""]
    return "\n".join(out)


def _branch_list(branches: "list[str]") -> str:
    """The spawn sites as a readable list, with a loud default when there are none.

    Zero branches is the state of a tree WITHOUT the fix, so it must read as such in
    the artifact rather than as an empty cell a reader skips.
    """
    return ", ".join(branches) if branches else (
        "NONE — this tree does not route any Bash spawn through the vault")


def write_artifact(out: "Path | None" = None, today: "str | None" = None) -> Path:
    """Write the dated census and return its path.

    `out` exists so a test can point the generator at a scratch directory: a node
    that wrote into `eval/measurements/` would leave a dated file in the tree on
    every run, which reads as a real census to the next reader.
    """
    day = today or date.today().isoformat()
    target = out or (Path(__file__).resolve().parents[2] / "eval" / "measurements"
                     / f"{ARTIFACT_STEM}-{day}.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_artifact(today=day), encoding="utf-8")
    return target


if __name__ == "__main__":
    import sys
    if "--write" in sys.argv:
        rest = [a for a in sys.argv[1:] if not a.startswith("--")]
        print(write_artifact(Path(rest[0]) if rest else None))
    else:
        print(json.dumps(measure(), indent=2, sort_keys=True))
