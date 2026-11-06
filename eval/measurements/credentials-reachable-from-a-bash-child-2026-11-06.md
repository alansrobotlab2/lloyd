# Credentials reachable from a Bash child — measured 2026-11-06

Step 1's census for #3846. Written against HEAD `21411475` before the fix and
then updated with what the fix closed, so this file is a before/after and not a
claim. Everything below is a measurement on this box; where the item's own
coordinates are wrong, they are named wrong rather than repeated.

Reproduce all of it (prints names, counts, line numbers and sha256 prefixes —
never a value):

    cd ~/lloyd && .venvs/lloyd/bin/python -m app.harness.credential_store

`tests/test_credential_census.py` reads the same `measure()` function this prose
quotes, so the artifact and the suite cannot drift apart.

## The answer step 1 asked for: three names, not thirty

The whole env lane is **3** vaulted names inside a **223**-variable child
environment. That is what makes placeholder-ing the spawn env a landable control
rather than a project, and it is why the residual list below can be named in
full. The census deliberately did not vault the whole file: see the shape rule
under lane 1.

## Lane 1 — the environment (3 vaulted names)

`~/lloyd/.env` supplies **6** names. Three are secret-shaped and vaulted; three
are configuration that merely shares a file with secrets. A fourth secret-shaped
name appearing anywhere in a child's 223 variables would make
`tests/test_credential_census.py` fail, which is the "no entry left unnamed"
half of clause 1.

| Env name | Reached a child before this round | Value length | Consumer today | Can a substitution lane serve it? | After this round |
|---|---|---|---|---|---|
| `ANTHROPIC_API_KEY` | real value | 15 | The aggregator's own LLM client (server-side), plus any CLI a turn invokes | Yes for egress-shaped use — header swap bound to one host. No for a CLI that reads env | Placeholder; real value only on a live grant |
| `LIVEKIT_API_KEY` | real value | 18 | `agent-services/livekit_worker.py`, `app/routers/voice.py`, the voice e2e scripts | No — a LiveKit key signs SDK tokens, it is not a header on one host | Placeholder; real value only on a live grant |
| `LIVEKIT_API_SECRET` | real value | 48 | The same three readers; signs room tokens | No — same reason | Placeholder; real value only on a live grant |

The three names above are the whole env lane. Three further names come from the
same file and are NOT credentials, so they are not vaulted and the child keeps
them unchanged:

- `ANTHROPIC_BASE_URL` — endpoint selection. Placeholder-ing a URL a client parses
  is an outage, not a control.
- `ANTHROPIC_CUSTOM_MODEL_OPTION` — a model id.
- `ANTHROPIC_CUSTOM_MODEL_OPTION_NAME` — a display label.

This is why the census keys on NAME SHAPE and not on the file. `~/lloyd/.env` is
not a secrets file that happens to hold config; it is a config file that happens to
hold three secrets, and vaulting it wholesale would break the model selector.

No secret-shaped name outside `.env` reaches a child: the same probe over all 223
variables of the aggregator's environment returns exactly the three rows above, and
`agent-services/*.conf` carries no secret-shaped `environment=` line (0 hits).

## Lane 2 — the config-literal lane: zero literals, and the item is wrong about it

#3846's clause 1 asks the census to name "the config.yaml literals at
:339/:1040-1041". Measured at HEAD `21411475`, and at the live tree — `config.yaml`
is md5-identical between the two (`d8969b925ca62220b0d3871e628545c3`) — there are
**no literal credentials in `config.yaml` at all**. The scan
(`credential_store.config_secret_literals`, a raw-text pass over the 2647-line file
because the answer needed is a line number) returns an empty list.

What those three coordinates actually hold:

| Coordinate | Key | Value form | Consumer today | Could a substitution lane serve it? |
|---|---|---|---|---|
| config.yaml:339 | `token` | empty — an explicitly blank webhook token | The inbound webhook receiver, which therefore accepts nothing | Nothing to vault; a placeholder where an empty string is the documented "disabled" value would read as a configured token |
| config.yaml:1040 | `api_key` | indirected to the env name `LIVEKIT_API_KEY` | `livekit_worker.py`, `app/routers/voice.py` | Already on lane 1: this is a second REFERENCE to a name the census already lists, not a second copy of a secret |
| config.yaml:1041 | `api_secret` | indirected to the env name `LIVEKIT_API_SECRET` | The same readers | Same — the env lane is where this secret is reachable, and that is where it is now placeholder-ed |

Three more coordinates worth naming so nobody re-derives them: `livekit.url` at
:1039 is a loopback ws URL (config, not a credential) and `livekit.room_prefix` at
:1042 is a string prefix.

The distinction that matters for the fix: indirected values are resolved at config
LOAD by `app/config.py::_expand_vars`, which reads the process environment. Once
the env lane holds a placeholder, the resolved config a child process builds
inherits the placeholder too — so the indirection lane needed no second mechanism,
and building one would have been a second source of truth for one secret.

## Named residuals — what is still reachable, unchanged

Nothing in this list is a secret-shaped env var, and each is named rather than
left as an absence:

1. **File-form secrets.** `~/.ssh` private keys, `~/.config/gh/hosts.yml`, any CLI
   token cache. No env var holds them, so there is no placeholder to substitute;
   they remain behind the read-deny roots and bwrap's read-binds. A private key has
   no placeholder-able form — this is #3846's own scope decision, restated so the
   module docstring cannot be read as claiming general credential isolation.
2. **The bwrap branch narrows nothing that read-binds the tree.** The bubblewrap
   sandbox mounts the home tree read-only, so a bench or eval turn can still READ
   the files in residual 1. What this round changed on that branch is that its
   child no longer receives the three env credentials, which was never bwrap's job.
3. **The confused deputy.** A destination binding says where a credential may be
   spent, not which account or method it may touch. An injected turn that causes a
   granted call still causes it. Not solved here and not claimed here.
4. **The `enforce` flip (#628, left to a human).** Until `harness.egress_policy.enforce`
   is on, the destination gate records rather than refuses, so substitution is a
   reduced-blast-radius default and not yet a control. This round does not touch it
   (`config.yaml:799-801` unchanged).
5. **Whether the three config-lane indirections should move into the registry.**
   Deferred: `config.yaml` lands only outside `spec.CONFIG_DENIED_KEYS`, so moving
   them is a gate-scope call this census cannot settle. Named, not silent.

## Where the vault reads its names from — and why not only from this file

The census's headline is `~/lloyd/.env`, but a guard defined over that file alone
would fail **open**: a secret can be in the environment without being in the file
(a supervisor `environment=` line, a launcher export, or a root change that made
the file unreachable after `app/config.py` had already loaded it into
`os.environ`). What a child inherits is the environment, so `vaulted_names()` is
the union of registry ∪ file ∪ **this process's secret-shaped environment names**,
pinned by `test_a_secret_only_in_the_environment_is_still_vaulted`.

Membership is decided by NAME shape, never by a value heuristic — an empty or
obvious-fixture value under a credential name is still vaulted, because judging by
value would let a real secret keep its name just by being long. `NOT_A_CREDENTIAL`
is the one exclusion, and it is the three config names listed above.
