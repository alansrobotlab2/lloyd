"""What a self-modification round is allowed to change.

With no human review tier, this file is the entire control surface, and a
control surface that can quietly modify itself is not a control surface. Hence
three tiers rather than two:

  * **allowed** — ordinary code. Lands automatically once the gate passes.
  * **protected** — the gate, the guardian, and the restart/health path the
    rollback itself depends on. Lloyd may edit these, but the change must
    additionally survive a live rollback drill (gate rung 6). A bound on *how*
    it changes, not on *whether* — which is the difference between guarding a
    capability and amputating it.
  * **denied** — never landed by this loop, whatever the run spec says.
    `config.yaml` and `data/tool_overrides.yaml` because a round could disable
    the agent's own Bash/Edit tools and lock itself out with no Python change
    at all; `pytest.ini` and `.gitignore` because they define what the gate
    even sees; `.env*` because secrets. Under `web/`, the dependency set and
    the build configuration (`package.json`, the lockfile, `node_modules`,
    `vite.config.*`, `tsconfig*.json`) — the frontend rung type-checks and
    builds the candidate against the LIVE tree's `node_modules`, which is only
    the candidate's dependency set if a round cannot change it.

`web/src/**`, `web/index.html` and `web/public/**` are *allowed* since
2026-09-07, for the same reason `requirements.txt` is: gate rung `frontend`
runs `tsc --noEmit` (new errors only — the tree carries pre-existing ones) and
a full `vite build` from the worktree. Before that rung existed `web/**` was
denied outright, and the first unattended implement round (#278) spent nine
iterations discovering that its contract needed a Browser page it could not
touch.

Note `requirements.txt` / `requirements.lock` are *allowed*, but only because
gate rung 3 builds a throwaway venv from them (btrfs reflink clone + `uv pip
install` of the delta) and boots the canary against it. Without that rung they
would belong in `denied`.

`config.yaml` stays *denied* by path, with two content-judged lanes out of
the denial. The first (2026-09-28) cannot change a value: a diff whose YAML
token stream is identical before and after, so only comments and layout
moved (`COMMENT_ONLY_GLOBS`, `comment_only_change`). Refusing comments too
left every stale comment in the file to a hand edit: five items sat
confirmed and unlandable at once on 2026-09-28.

The second (2026-10-05) changes a value, inside a fence. Over the three weeks
before it, thirteen items stalled as `human-only: config.yaml` and every one
was approved by hand as written; most wanted a tunable moved —
`harness.finalizer.max_tokens`, `workers.sources.autoresearch.max_duration_
seconds`, `harness.edit_diagnostics.blast_radius`, `harness.egress_policy`,
`workers.sources.autocode.reasoning_bank`. Alan's ruling that day: automod
goes with its own recommendation unless a step is physical. So
`config_value_change` admits a diff whose both sides parse to a mapping, whose
changed key paths all fall outside `CONFIG_DENIED_KEYS` (prefixes: the tool
pool and `disabled_tools`, the engine slots, the services and ports a
rollback has to reach, the guardian, the loop's own switch and landing block,
and its three worker sources — the config half of what `PROTECTED_GLOBS` is
for code), whose changed leaves are none of `CONFIG_DENIED_LEAVES` (an
endpoint, a device, a credential, anywhere in the tree) and carry no `${`
placeholder (secrets reach the file only that way), and which removes no
top-level key. The refusal names the key and the rule. What lands moves to
the `config_value` bucket, and `requires_drill` turns the rollback drill on
for it: a config change is boot-affecting, so the bound is on HOW it lands,
not whether — the same shape as `protected`. The canary rungs boot from the
round's worktree, so the candidate file is the one they exercise. The
lock-out the denial was written for (`mcp_servers.*.disabled_tools`, an
`enabled` switch on the agent's own tools) is a denied prefix, and so stays
exactly as unreachable as before.
"""

from __future__ import annotations

import fnmatch
from pathlib import PurePosixPath

ALLOWED_GLOBS: tuple[str, ...] = (
    "app/**",
    "agent_mcp/**",
    "workers/**",
    "eval/**",
    "tests/**",
    "scripts/**",
    "architecture/**",
    # The one module left at the repo root: the backend entrypoint supervisord
    # runs by path. `prompt_builder`, `prompt_surface`, `autonomy`, `prefetch`
    # and `usage_store` used to sit beside it and each needed its own line here;
    # #1242 was three rounds (SM_20260911_190850, SM_20260914_114935,
    # SM_20260918_145241) refused at rung 0 because one of them was never
    # listed. They live in `app/` now, so `app/**` above covers them.
    "server.py",
    "requirements.txt",
    "requirements.lock",
    # #1073 (2026-09-22): the settled home for an optional solver — a package
    # that must NOT become a container-rebuild dependency. Admitting it is safe
    # because of two facts, both pinned by `test_the_dev_requirements_file_is_
    # never_an_install_target`: `touches_requirements` below keys on exactly the
    # two names above, and `rung_venv` installs from `requirements.lock` else
    # `requirements.txt` (gate.py:1800, :1825-1826). A third requirements file
    # can therefore neither trigger the candidate-venv rung nor be installed by
    # it, which is the whole point: a solver listed only here can never diverge
    # the live venv from the candidate.
    "requirements-dev.txt",
    "CLAUDE.md",
    "README.md",
    # Also #1073: the item's contract puts the dependency decision in SETUP.md,
    # and this path was unlisted, so no round could have landed it — the same
    # defect #1242 was for the root-level `prompt_surface.py`, found the same way.
    "SETUP.md",
    # #1449 / #1444 (2026-09-24): the wake-miss corpus writer, and only that
    # file — never a directory glob. `agent-services/**` would admit every
    # launcher and conf a round may not touch (the 2026-09-21 agent-llm-primary
    # restart is what those rails exist for); tests/test_automod_spec.py's
    # #1376 section pins that a widening here must name its path verbatim.
    "agent-services/livekit_worker.py",
    # #1883 / #1878 (2026-09-30): the tracked Qwen3-TTS patch, and again one
    # verbatim file rather than `agent-services/services/tts/*.patch` — the
    # #1376 rail refuses a wildcard spelling of a single file, and
    # tests/test_automod_spec.py pins both halves. Admitted for a reason the
    # line above does not share: `agent-services/services/tts/qwen3-tts/` is an
    # untracked vendored clone, so no launcher, conf or weight under it is in
    # the index, and this `.patch` is the only artefact of that integration a
    # commit can carry and a gate can see. #1878's round SM_20260930_063800
    # wrote the frame cap into it and died at rung 0 with the path bucketed
    # `unlisted` — the sole rung that ran. Applying the patch to the live clone
    # and restarting `agent-tts` stays a human action (SETUP.md); this entry
    # only makes the tracked half of that fix landable by a round.
    "agent-services/services/tts/qwen3-tts-local.patch",
    # #1301 / #1298 (2026-10-03): the tracked qmd collection template, one
    # verbatim path again and not `agent-services/conf/**` — that directory also
    # holds the tracked `livekit.yaml` and `livekit.yaml.runtime`, the 0600
    # live-credentials copy `.gitignore:95` keeps out of the index, so a
    # directory grant would widen a secret-bearing launcher config to buy one
    # text file. Admitted for the reason the two above do not share: the guard on
    # a stale template is a report, not a wall. The nightly check compares this
    # file with the config the daemon reads and records drift as "a report entry
    # and never an exit code"
    # (scripts/maintenance/qmd_index_maintenance.py:85-86, restated at :889),
    # then prints `cp ~/.config/qmd/index.yml agent-services/conf/qmd-index.yml`
    # (:156) for a person to run — and the `facts` drift sat reported-but-unfixed
    # for ten days, 2026-09-19 to 09-28, because no round could run it. What this
    # entry does NOT buy: the live file is outside the repo and stays unreachable,
    # so the direction stays live -> template, and once the tracked half is
    # writable the only thing between a bad rewrite and a stale contract is
    # tests/test_qmd_index_template.py.
    "agent-services/conf/qmd-index.yml",
    "web/src/**",
    "web/index.html",
    "web/public/**",
)

PROTECTED_GLOBS: tuple[str, ...] = (
    "scripts/automod/**",
    "agent-services/guardian/**",
    "agent-services/systemd/**",
    "agent-services/supervisor/**",
    "agent-services/bin/**",
    "app/routers/health.py",
    "app/routers/automod.py",
    "app/supervisor_client.py",
    "app/lifecycle.py",
    "app/gitinfo.py",
)

DENIED_GLOBS: tuple[str, ...] = (
    ".gitignore",
    "pytest.ini",
    "config.yaml",
    "data/**",
    ".env",
    ".env.*",
    "web/package.json",
    "web/package-lock.json",
    "web/node_modules/**",
    "web/dist/**",
    "web/vite.config.*",
    "web/tsconfig*.json",
    "web/.env",
    "web/.env.*",
    ".git/**",
    ".venvs/**",
)

# Denied paths a round may still change when the change is comment-only.
# Verbatim names, never a glob: each entry must be a YAML file, because
# `comment_only_change` is a YAML check.
COMMENT_ONLY_GLOBS: tuple[str, ...] = ("config.yaml",)


def _yaml_tokens(text: str) -> list[tuple]:
    """The token stream with comments and positions dropped: what a value is."""
    import yaml
    return [(type(t).__name__, getattr(t, "value", None), getattr(t, "style", None))
            for t in yaml.scan(text)]


def comment_only_change(before: str, after: str) -> tuple[bool, str]:
    """Is `before` → `after` a change to comments and layout only?

    Two independent checks, both required: the scanner's token stream (which
    carries no comments) is identical, so no key, value, quoting style or
    ordering moved; and the parsed documents compare equal. A file that does
    not parse on either side is refused.
    """
    import yaml
    try:
        if _yaml_tokens(before) != _yaml_tokens(after):
            return False, "a key or value changed, not only comments"
        if yaml.safe_load(before) != yaml.safe_load(after):
            return False, "the parsed document changed"
    except yaml.YAMLError as exc:
        return False, f"does not parse: {str(exc)[:200]}"
    return True, "comments and layout only"


# Dotted key-path PREFIXES a round may never change in `config.yaml` (2026-10-05).
# A changed path matches when it equals a prefix or starts with `prefix + "."`.
# This is the config half of PROTECTED_GLOBS: the lock-out, the rollback path,
# the engine's identity, and the loop's own three sources.
CONFIG_DENIED_KEYS: tuple[str, ...] = (
    "mcp_servers",                    # the tool pool, `enabled`, `disabled_tools`: the lock-out itself
    "models",                         # engine slots, `expect_model`: what answers a turn
    "model",                          # the default model a turn routes to
    "subagents",                      # a Task child's model and tool set
    "server",                         # the backend's bind address: the rollback's health probe
    "services",                       # the service URLs the rollback and the canary reach
    "guardian",                       # the thing that performs every rollback
    "automod.enabled",                # the loop's master switch: a round may not switch itself off or on
    "automod.landing",                # restart/drain/squash: how a landing happens — the rollback path
    "workers.enabled",                # the pool that runs the loop
    "workers.slots",                  # the pool's depth: rounds + triages + 1, a restart-sized change
    "workers.sources.autocode",       # the loop's own three sources, like protected code:
    "workers.sources.autotriage",     #   a round editing its own budget, depth or model
    "workers.sources.owed-check",     #   is a control surface modifying itself
)

# Leaf key NAMES denied anywhere in the tree: an endpoint, a device, a credential.
CONFIG_DENIED_LEAVES: frozenset[str] = frozenset({
    "base_url", "url", "host", "port", "expect_model",
    "device", "devices", "gpu", "cuda_visible_devices",
    "token", "api_key", "password", "secret",
})

#: How many changed paths an admitting reason lists before "…".
_CONFIG_REASON_PATHS = 10


def _config_denied_prefix(path: str) -> str | None:
    for prefix in CONFIG_DENIED_KEYS:
        if path == prefix or path.startswith(prefix + "."):
            return prefix
    return None


def _diff_paths(before, after, prefix: str = "") -> dict[str, tuple]:
    """Changed dotted key paths → (before, after). A list is a leaf."""
    out: dict[str, tuple] = {}
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before:
                out[path] = (None, after[key])
            elif key not in after:
                out[path] = (before[key], None)
            else:
                out.update(_diff_paths(before[key], after[key], path))
    elif before != after:
        out[prefix] = (before, after)
    return out


def _has_placeholder(value) -> bool:
    if isinstance(value, str):
        return "${" in value
    if isinstance(value, dict):
        return any(_has_placeholder(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_placeholder(v) for v in value)
    return False


def config_value_change(before: str, after: str) -> tuple[bool, str, list[str]]:
    """Is `before` → `after` a value change to `config.yaml` inside the fence?

    Returns (ok, reason, changed_paths). Pure: no git, no file system. Both
    sides must parse to a mapping. The changed paths are the recursive diff
    of the two documents — added, removed and modified leaves, a list
    compared whole. Refused when any changed path sits under a
    `CONFIG_DENIED_KEYS` prefix, names a `CONFIG_DENIED_LEAVES` leaf, carries a
    `${` placeholder on either side, or removes a top-level key; the reason
    names every offending path and the rule that caught it.
    """
    import yaml
    docs = []
    for side, text in (("before", before), ("after", after)):
        try:
            doc = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            return False, f"{side} does not parse: {str(exc)[:200]}", []
        if not isinstance(doc, dict):
            return False, f"{side} is not a mapping", []
        docs.append(doc)
    diff = _diff_paths(docs[0], docs[1])
    changed = sorted(diff)
    if not changed:
        return False, "no value changed", []
    offences: list[str] = []
    for path in changed:
        old, new = diff[path]
        prefix = _config_denied_prefix(path)
        if prefix is not None:
            offences.append(f"`{path}` is under the denied key `{prefix}`")
        leaf = path.rsplit(".", 1)[-1]
        if leaf in CONFIG_DENIED_LEAVES:
            offences.append(f"`{path}` is a denied leaf name (`{leaf}`)")
        if _has_placeholder(old) or _has_placeholder(new):
            offences.append(f"`{path}` carries a `${{` placeholder — secrets reach "
                            f"config.yaml only that way")
        if "." not in path and new is None:
            offences.append(f"`{path}` is a top-level key and the change removes it")
    if offences:
        return False, "; ".join(offences), changed
    shown = ", ".join(changed[:_CONFIG_REASON_PATHS])
    if len(changed) > _CONFIG_REASON_PATHS:
        shown += f", … (+{len(changed) - _CONFIG_REASON_PATHS} more)"
    return True, f"value change within the fence: {shown}", changed


def _match(path: str, globs: tuple[str, ...]) -> bool:
    for pattern in globs:
        if fnmatch.fnmatch(path, pattern):
            return True
        # fnmatch's `*` crosses `/`, but `a/**` should also match `a/b/c.py`
        # and, for a directory pattern, the directory's own entries.
        if pattern.endswith("/**") and (
            path == pattern[:-3] or path.startswith(pattern[:-2])
        ):
            return True
    return False


def normalize(path: str) -> str | None:
    """Return a clean repo-relative POSIX path, or None if it escapes the repo."""
    if not path or not isinstance(path, str):
        return None
    p = path.strip().replace("\\", "/")
    if not p or p.startswith("/"):
        return None
    parts = PurePosixPath(p).parts
    if any(part == ".." for part in parts):
        return None
    parts = tuple(part for part in parts if part not in (".", ""))
    if not parts:
        return None
    return "/".join(parts)


def classify(path: str) -> str:
    """Return 'denied' | 'protected' | 'allowed' | 'unlisted' for one path.

    Order is the whole point: **denied beats everything**, including an
    explicit entry in a run spec's `writable_paths`. A round cannot widen its
    own permissions by asking nicely.
    """
    norm = normalize(path)
    if norm is None:
        return "denied"
    if _match(norm, DENIED_GLOBS):
        return "denied"
    if _match(norm, PROTECTED_GLOBS):
        return "protected"
    if _match(norm, ALLOWED_GLOBS):
        return "allowed"
    return "unlisted"


def classify_all(paths: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"allowed": [], "protected": [], "denied": [], "unlisted": []}
    for p in paths:
        out[classify(p)].append(p)
    return out


def check_scope(paths: list[str], *, contents=None
                ) -> tuple[bool, str, dict[str, list[str]]]:
    """Gate rung 0's diff-scope check.

    Returns (ok, reason, buckets). `ok` is False if anything is denied or
    unlisted. Protected paths are permitted here — they are what turns on the
    rollback drill in rung 6, handled by the caller.

    `contents(path) -> (before, after)` lets a denied path in
    `COMMENT_ONLY_GLOBS` through on content: `comment_only_change` first (the
    path moves to the `comment_only` bucket), else `config_value_change` (the
    path moves to `config_value`, which turns the drill on — see
    `requires_drill`). Without it, or when both checks fail, the path stays
    denied and the reason carries the key-level explanation.
    """
    buckets = classify_all(paths)
    buckets["comment_only"] = []
    buckets["config_value"] = []
    why_not: list[str] = []
    for p in list(buckets["denied"]):
        if contents is None or normalize(p) not in COMMENT_ONLY_GLOBS:
            continue
        try:
            before, after = contents(p)
        except Exception as exc:  # noqa: BLE001 — unreadable is not comment-only
            why_not.append(f"{p}: cannot read both sides: {exc}")
            continue
        ok, why = comment_only_change(before, after)
        if ok:
            buckets["denied"].remove(p)
            buckets["comment_only"].append(p)
            continue
        ok, why_value, _changed = config_value_change(before, after)
        if ok:
            buckets["denied"].remove(p)
            buckets["config_value"].append(p)
            continue
        why_not.append(f"{p}: {why}; as a value change: {why_value}")
    # Both refusals name what to do instead. A round that needs a path the
    # loop may never touch used to be told only that it could not have it,
    # and the move it then reached for was `git add -f` — which defeats the
    # scope check rather than reporting past it. `data/**` and `.gitignore`
    # are both denied, so a JSON data record has no home in the repo at all:
    # the automod state dir is where one goes.
    _MOVE = ("Leave it out of the diff, land the rest, and report it under "
             "`human_paths` in your outcome so owed-check can route it. NEVER "
             "`git add -f`. If you need to record structured data, write it "
             "to the automod state dir (~/.local/state/lloyd-automod/), not "
             "into the repo.")
    if buckets["denied"]:
        only = (f" A comment-only edit to {', '.join(COMMENT_ONLY_GLOBS)} is "
                f"allowed, and so is a value change outside the denied keys "
                f"(spec.CONFIG_DENIED_KEYS / CONFIG_DENIED_LEAVES); this one is "
                f"not ({'; '.join(why_not)})." if why_not else "")
        return False, (f"denied paths in diff: {sorted(buckets['denied'])}.{only} "
                       f"{_MOVE}"), buckets
    if buckets["unlisted"]:
        return False, (f"paths outside the writable set: "
                       f"{sorted(buckets['unlisted'])}. {_MOVE}"), buckets
    return True, "in scope", buckets


def requires_drill(paths: list[str], *, buckets: dict[str, list[str]] | None = None) -> bool:
    """True when the diff touches the rollback path and must pass the drill.

    By paths alone: any `protected` path. With rung 0's `buckets` passed, a
    non-empty `config_value` bucket also arms it: a value change to
    `config.yaml` is boot-affecting, so the bound on it is how it lands —
    through a live rollback drill — the same shape `protected` has for code.
    The bucket is content-judged, so the paths-only call cannot see it.
    """
    if buckets is not None and buckets.get("config_value"):
        return True
    return bool(classify_all(paths)["protected"])


def touches_requirements(paths: list[str]) -> bool:
    return any(normalize(p) in ("requirements.txt", "requirements.lock") for p in paths)


def validate_code_run_spec(spec: dict) -> str | None:
    """Validate a code round's run_spec. Returns None on success, else a reason.

    Layers over `scripts.autoresearch.common.validate_run_spec`, which only
    hard-checks that `mutation_scope.writable_paths` is a list. For code rounds
    the entries are repo-relative globs rather than the absolute .md paths the
    prompt pipeline produces, so they need their own shape check.
    """
    try:
        from scripts.autoresearch.common import validate_run_spec
    except Exception:  # pragma: no cover - import guard
        validate_run_spec = None  # type: ignore
    if validate_run_spec is not None:
        base = validate_run_spec(spec)
        if base:
            return base

    scope = (spec.get("mutation_scope") or {}).get("writable_paths")
    if not isinstance(scope, list):
        return "mutation_scope.writable_paths must be a list"
    for entry in scope:
        if not isinstance(entry, str):
            return f"writable_paths entry is not a string: {entry!r}"
        if normalize(entry.replace("**", "x")) is None:
            return f"writable_paths entry is not a safe relative path: {entry!r}"

    code = spec.get("code")
    if not isinstance(code, dict):
        return "missing required key: code"
    for key in ("base_commit", "branch"):
        if not code.get(key):
            return f"code.{key} is required"
    return None
