"""#3352 — the automod config fence must also stand on #40's nightly write path.

Autonomy task #40 (`nightly-reflection-config`) edits `config.yaml` directly on
`main`, unattended, with `yaml.safe_load` as its only stated validation, and
`skills/nightly-reflection-config/SKILL.md` forbids the work branch that would
make such a commit revertable by review. Before this round the deny-set
(`CONFIG_DENIED_KEYS`, `CONFIG_DENIED_LEAVES` and the differ
`config_value_change` in `scripts/automod/spec.py`) had exactly one production
caller — `check_scope`, inside `scripts/automod/` — so it only ever judged a
self-modification *round's* config edit, never a nightly job's.

These tests pin the three things #3352 asked for:

  1. `check_config_edit` answers for the on-disk `config.yaml` against `HEAD`,
     and a denied value is refused naming the offending key path and the rule.
  2. A comment-only edit is allowed, reported as `no_value_changed` — an
     outcome distinguishable from a refusal, because the differ underneath
     answers `ok=False` for both.
  3. The refusal is reached from the route the nightly job's commit uses
     (`scripts/config_edit_guard.py commit`), not only from inside the gate.

The process boundary is real here — the job reaches this through Bash, and this
through `git` — so the commit-route tests drive the CLI as a subprocess against
a throwaway repo rather than calling the function in-process. Each refusal test
has a positive control beside it (the benign value under the same route does
commit) so the guard cannot pass by always refusing.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import config_edit_guard as guard
from scripts.automod import spec

ROOT = Path(__file__).resolve().parent.parent
GUARD_CLI = ROOT / "scripts" / "config_edit_guard.py"

#: The committed baseline of the fixture repo. `workers.slots` sits under a denied
#: prefix, `discord.token` is a denied leaf name, `discord.home_channel` and
#: `discord.owner_id` are denied exact paths (#3472), and `logging.level`,
#: `reflection.confidence_floor` and `discord.require_mention` are free. The keys
#: the leaf and path rules name exist at `HEAD` so those tests judge a VALUE change;
#: a denied path that exists only in the edit is the added-block case, which is what
#: #3472 closed — `_diff_paths` used to record an added key at the node that was
#: added and never descend, so a whole new `discord:` block diffed as `discord` and
#: was admitted. That case is `test_a_denied_block_added_whole_is_refused_naming…`.
BASE_CONFIG = """\
# Lloyd engine config — fixture baseline.
workers:
  enabled: true
  slots: 4
discord:
  home_channel: ""
  owner_id: ""
  require_mention: true
  token: ""
logging:
  level: info
reflection:
  confidence_floor: 0.4
"""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A throwaway repo on `main` holding one commit of `BASE_CONFIG`."""
    root = tmp_path / "repo"
    root.mkdir()
    assert _git(root, "init", "-b", "main").returncode == 0
    _git(root, "config", "user.name", "Config Guard Test")
    _git(root, "config", "user.email", "guard-test@example.invalid")
    (root / guard.CONFIG_PATH).write_text(BASE_CONFIG)
    assert _git(root, "add", "--", guard.CONFIG_PATH).returncode == 0
    assert _git(root, "commit", "-m", "baseline").returncode == 0
    return root


def _edit(repo: Path, old: str, new: str) -> None:
    text = (repo / guard.CONFIG_PATH).read_text()
    assert old in text, f"fixture line {old!r} is not in config.yaml"
    (repo / guard.CONFIG_PATH).write_text(text.replace(old, new))


def _run_cli(*args: str, repo: Path) -> subprocess.CompletedProcess:
    """Cross the real boundary: the job runs this through a shell."""
    return subprocess.run(
        [sys.executable, str(GUARD_CLI), *args],
        capture_output=True,
        text=True,
        cwd=str(repo),
    )


# --- clause 1: the callable answers, and refuses by naming path and rule -------


def test_a_denied_prefix_is_refused_naming_the_key_path_and_the_rule(repo: Path) -> None:
    """`workers.slots` is a denied prefix, so flipping it is refused by name."""
    _edit(repo, "  slots: 4", "  slots: 16")

    verdict = guard.check_config_edit(repo)

    assert verdict.allowed is False
    assert verdict.status == guard.DENIED
    assert "workers.slots" in verdict.reason, verdict.reason
    assert "denied key" in verdict.reason, verdict.reason
    assert "workers.slots" in verdict.changed_paths


def test_a_denied_leaf_is_refused_naming_the_leaf_rule(repo: Path) -> None:
    """A `token` leaf is denied anywhere in the tree, not only under a prefix —
    which is the shape of the edit #40 is most likely to want: filling in a
    credential that is configured empty."""
    _edit(repo, '  token: ""', '  token: "abc123"')

    verdict = guard.check_config_edit(repo)

    assert verdict.allowed is False
    assert verdict.status == guard.DENIED
    assert "discord.token" in verdict.reason, verdict.reason
    assert "denied leaf name" in verdict.reason, verdict.reason


def test_a_benign_value_change_is_allowed_and_names_its_path(repo: Path) -> None:
    """Positive control: the guard is not a blanket refusal of config edits."""
    _edit(repo, "  confidence_floor: 0.4", "  confidence_floor: 0.7")

    verdict = guard.check_config_edit(repo)

    assert verdict.allowed is True
    assert verdict.status == guard.VALUE_CHANGED
    assert verdict.changed_paths == ("reflection.confidence_floor",)


# --- clause 2: a comment-only edit is an outcome, not a denial -----------------


def test_a_comment_only_edit_is_allowed_and_reported_as_no_value_changed(
    repo: Path,
) -> None:
    """The trap #3352 names: `config_value_change` answers `ok=False` for a
    comment-only edit with the reason `no value changed`, which is admission,
    not refusal. The guard's `status` is what separates it from a denial, so a
    caller reading the differ's flag alone could never tell them apart."""
    text = (repo / guard.CONFIG_PATH).read_text()
    (repo / guard.CONFIG_PATH).write_text(
        text.replace(
            "# Lloyd engine config — fixture baseline.",
            "# Lloyd engine config — fixture baseline. Comment reworded only.",
        )
    )

    after = (repo / guard.CONFIG_PATH).read_text()
    differ_ok, differ_reason, differ_paths = spec.config_value_change(BASE_CONFIG, after)
    assert differ_ok is False, "the fixture stopped exercising the trap"
    assert differ_reason == guard.NO_VALUE_CHANGED_REASON
    assert differ_paths == []

    verdict = guard.check_config_edit(repo)

    assert verdict.allowed is True, "a comment-only edit must not read as denied"
    assert verdict.status == guard.NO_VALUE_CHANGED
    assert verdict.status != guard.DENIED
    assert verdict.reason == differ_reason


def test_the_cli_reports_a_comment_edit_and_a_denial_as_different_outcomes(
    repo: Path,
) -> None:
    """The verdict the skill's instruction depends on, read off the CLI."""
    text = (repo / guard.CONFIG_PATH).read_text()
    (repo / guard.CONFIG_PATH).write_text(text + "# a comment the run added\n")
    comment_run = _run_cli("--repo", str(repo), "--json", repo=repo)
    assert comment_run.returncode == guard.EXIT_ALLOWED, comment_run.stderr
    comment_verdict = json.loads(comment_run.stdout)
    assert comment_verdict["allowed"] is True
    assert comment_verdict["status"] == guard.NO_VALUE_CHANGED

    _edit(repo, "  slots: 4", "  slots: 16")
    denied_run = _run_cli("--repo", str(repo), "--json", repo=repo)
    assert denied_run.returncode == guard.EXIT_DENIED, denied_run.stdout
    denied_verdict = json.loads(denied_run.stdout)
    assert denied_verdict["allowed"] is False
    assert denied_verdict["status"] == guard.DENIED
    assert denied_verdict["status"] != comment_verdict["status"]


# --- #3472: the two measured holes in the fence this guard inherits -----------


@pytest.mark.parametrize("key, old, new", [
    ("home_channel", '  home_channel: ""', '  home_channel: "1234567890123456789"'),
    ("owner_id", '  owner_id: ""', '  owner_id: "1234567890123456789"'),
])
def test_the_two_discord_routing_keys_are_refused_naming_path_and_rule(
    repo: Path, key: str, old: str, new: str
) -> None:
    """Where the fleet's alerts go is Alan's call, not a nightly run's.

    `skills/nightly-reflection-config/SKILL.md` forbids a `config.yaml` edit to
    `discord.home_channel`/`token`/`owner_id`, and until #3472 the fence behind that
    sentence backed one of the three: `token` is caught only because `token` happens
    to be a denied LEAF name, while these two came back as a value change inside the
    fence. `discord.home_channel: null` in the live file is a ruling, not an unset
    credential — see that skill's pinned row and
    `topics/retired-decisions-alan-s-rulings-do-not` — which is why filling it in
    unattended has to be refused rather than merely advised against.
    """
    _edit(repo, old, new)

    verdict = guard.check_config_edit(repo)

    assert verdict.allowed is False
    assert verdict.status == guard.DENIED
    assert f"discord.{key}" in verdict.reason, verdict.reason
    assert "denied key" in verdict.reason, verdict.reason
    assert f"discord.{key}" in verdict.changed_paths, verdict.changed_paths


def test_a_sibling_discord_tunable_still_lands_through_the_commit_route(
    repo: Path,
) -> None:
    """Positive control on the same block: `require_mention` is a tunable, so
    fencing two keys of `discord` must not freeze the section. The route is the
    job's own `commit`, so a fence that had widened to the whole block would show up
    here refusing an edit #40 is allowed to make."""
    _edit(repo, "  require_mention: true", "  require_mention: false")

    run = _run_cli("--repo", str(repo), "commit",
                   "-m", "nightly-reflection: config require_mention off", repo=repo)

    assert run.returncode == guard.EXIT_ALLOWED, f"{run.stdout}\n{run.stderr}"
    head = guard.head_config(repo)
    assert head is not None and "require_mention: false" in head, head
    assert _git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "2"


def test_a_denied_block_added_whole_is_refused_naming_the_key_inside_it(
    repo: Path,
) -> None:
    """Hole 1 on #40's own route, across the boundary the job actually uses.

    The differ records an added key at the node that was added, so until #3472 a
    whole `observability:` block carrying a collector credential diffed as
    `observability` — no denied prefix, and `observability` is not a denied leaf
    name — and this guard committed it to `main`. The refusal now names the key
    inside the block, which is the one a reviewer has to see.
    """
    cfg = repo / guard.CONFIG_PATH
    cfg.write_text(cfg.read_text() + "observability:\n  otel:\n"
                   "    sample_rate: 0.1\n    token: 'collector-secret'\n")

    run = _run_cli("--repo", str(repo), "--json", repo=repo)

    assert run.returncode == guard.EXIT_DENIED, run.stdout
    verdict = json.loads(run.stdout)
    assert verdict["status"] == guard.DENIED
    assert "observability.otel.token" in verdict["reason"], verdict
    assert "observability.otel.token" in verdict["changed_paths"], verdict

    # Positive control: the same block, credential removed, is a value change.
    cfg.write_text(cfg.read_text().replace(
        "    token: 'collector-secret'\n", "    endpoint_name: otel-collector\n"))
    allowed = _run_cli("--repo", str(repo), "--json", repo=repo)
    assert allowed.returncode == guard.EXIT_ALLOWED, allowed.stdout
    assert json.loads(allowed.stdout)["status"] == guard.VALUE_CHANGED


def test_an_untouched_tree_and_a_comment_edit_stay_no_value_changed(
    repo: Path,
) -> None:
    """Clause 5, after the recursion: an untouched tree and a comment-only edit are
    admitted outcomes, not denials, and `status` is what separates them from a
    refusal — the differ underneath answers `ok=False` for all three of them."""
    untouched = _run_cli("--repo", str(repo), "--json", repo=repo)
    assert untouched.returncode == guard.EXIT_ALLOWED, untouched.stderr
    assert json.loads(untouched.stdout)["status"] == guard.NO_VALUE_CHANGED

    cfg = repo / guard.CONFIG_PATH
    cfg.write_text(cfg.read_text() + "# a comment the run added\n")
    commented = _run_cli("--repo", str(repo), "--json", repo=repo)
    assert commented.returncode == guard.EXIT_ALLOWED, commented.stderr
    assert json.loads(commented.stdout)["status"] == guard.NO_VALUE_CHANGED

    _edit(repo, "  slots: 4", "  slots: 16")
    denied = _run_cli("--repo", str(repo), "--json", repo=repo)
    assert denied.returncode == guard.EXIT_DENIED, denied.stdout
    assert json.loads(denied.stdout)["status"] == guard.DENIED


# --- clause 6: the skill's prose and this fence are one rule, not two ----------

#: The Discord keys `skills/nightly-reflection-config/SKILL.md` refuses and the
#: fence has to back: `token` by denied leaf name, `home_channel` and `owner_id`
#: by denied exact path (#3472 — until then only `token` was, because `token`
#: happens to be a leaf name). One tuple read by BOTH halves of the test below, so
#: prose and fence are graded against the same list and a rename in either trips it.
DISCORD_ROUTING_KEYS = ("discord.home_channel", "discord.token", "discord.owner_id")

#: Where the skill's enumeration of refused keys begins, in its own words. Located
#: by wording rather than by line number or heading, which both drift.
_FENCE_BULLET_MARKER = "The refused keys are"


def _nightly_config_skill() -> str:
    """The live `nightly-reflection-config` skill — the file job #40 reads.

    `Path.home() / "obsidian"`, deliberately not `app.data_root.vault_root()`: a
    suite run that redirects the vault would otherwise judge a fixture's prose
    instead of the instruction the nightly run actually follows. Fails rather than
    passing an absence, the shape of `test_ambient_inject.py::_skill` — a test that
    returned early on a missing vault would report prose and fence as agreeing on
    the strength of there being no prose at all.
    """
    skill = Path.home() / "obsidian" / "skills" / "nightly-reflection-config" / "SKILL.md"
    if not skill.exists():
        pytest.fail(f"no live skill at {skill} — cannot check prose against the fence")
    return skill.read_text(encoding="utf-8")


#: A line that starts a bullet, at any indent. Everything after it until the next
#: one is that bullet's own prose, however the file wraps it.
_BULLET_START = re.compile(r"^\s*- ")


def _bullets(text: str) -> list[str]:
    """Every bullet of the document, each wrapped lines joined into one string.

    The skill wraps at ~100 columns, so a sentence searched for in raw lines is
    two sentences: matching `The refused keys are` against the file as read fails
    when the phrase happens to straddle a newline, which is a reflow away from a
    red test that says nothing about the fence.
    """
    out: list[str] = []
    for line in text.splitlines():
        if _BULLET_START.match(line) or not out:
            out.append(line.strip())
        else:
            out[-1] = f"{out[-1]} {line.strip()}".strip()
    return out


def _fence_bullet(text: str) -> str:
    """The one bullet that enumerates the keys the guard refuses. Located by its
    own wording, so editing anywhere else in the skill cannot move the window."""
    hits = [b for b in _bullets(text) if _FENCE_BULLET_MARKER in b]
    if len(hits) != 1:
        pytest.fail(f"expected exactly one bullet enumerating the refused keys, found "
                    f"{len(hits)} — the prose the fence has to match has moved or gone")
    return hits[0]


def test_the_skills_fence_bullet_and_the_fence_name_the_same_discord_keys() -> None:
    """Clause 6, pinned from both sides.

    #3472 was filed because that skill refused three keys in prose and the fence
    behind the sentence enforced one of them, so an unattended run read a rule that
    was not there and could repoint every alert the fleet emits. Checking the key
    list twice — that the fence refuses each one, and that the skill's bullet names
    each one — is what makes the two files unable to drift apart silently, which is
    the failure a vault-side edit outside the graded tree would otherwise hide.
    """
    bullet = _fence_bullet(_nightly_config_skill())

    for key in DISCORD_ROUTING_KEYS:
        # The fence side: a value change to a key the prose refuses is refused,
        # naming that key as the offending path.
        ok, reason, paths = spec.config_value_change(f"{key}: old\n", f"{key}: new\n")
        assert not ok, f"the fence admits `{key}`, which the skill refuses in prose"
        assert key in reason and key in paths, (key, reason, paths)
        # The prose side: the bullet still names it.
        assert f"`{key}`" in bullet, f"the skill's fence bullet no longer names `{key}`"


# --- clause 3: the nightly commit route reaches the refusal -------------------


def test_the_commit_step_refuses_a_denied_value_and_it_never_reaches_head(
    repo: Path,
) -> None:
    """A denied value edited into the working tree is refused by the very step
    the job's commit uses, stays out of `HEAD`, and is left in place for the
    run to revert — the guard commits, it never reverts."""
    head_before = guard.head_config(repo)
    _edit(repo, "  slots: 4", "  slots: 16")

    run = _run_cli(
        "--repo", str(repo), "commit", "-m", "nightly-reflection: config test", repo=repo
    )

    assert run.returncode == guard.EXIT_DENIED, run.stdout
    assert "REFUSED" in run.stderr, run.stderr
    assert "workers.slots" in run.stderr, run.stderr
    assert guard.head_config(repo) == head_before, "the denied value reached HEAD"
    assert "slots: 16" in (repo / guard.CONFIG_PATH).read_text()
    assert "config.yaml" in _git(repo, "status", "--porcelain").stdout
    assert _git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "1"


def test_the_commit_step_commits_a_benign_value_on_main_and_nothing_else(
    repo: Path,
) -> None:
    """Positive control across the same boundary: an allowed change is actually
    committed on `main`, and only `config.yaml` travels — another writer's
    staged file is left staged, so this route cannot become an `git add -A`."""
    _edit(repo, "  confidence_floor: 0.4", "  confidence_floor: 0.7")
    (repo / "stray.txt").write_text("someone else's work\n")
    assert _git(repo, "add", "--", "stray.txt").returncode == 0

    run = _run_cli(
        "--repo",
        str(repo),
        "commit",
        "-m",
        "nightly-reflection: config confidence floor 0.7",
        repo=repo,
    )

    assert run.returncode == guard.EXIT_ALLOWED, f"{run.stdout}\n{run.stderr}"
    assert "ALLOWED" in run.stdout, run.stdout
    assert "reflection.confidence_floor" in run.stdout, run.stdout
    head = guard.head_config(repo)
    assert head is not None and "confidence_floor: 0.7" in head
    committed = _git(repo, "show", "--name-only", "--pretty=format:%s", "HEAD").stdout
    assert committed.splitlines()[0].startswith("nightly-reflection: config")
    assert "config.yaml" in committed
    assert "stray.txt" not in committed, "the commit swept another writer's file"
    assert "A  stray.txt" in _git(repo, "status", "--porcelain").stdout
    assert guard.current_branch(repo) == "main"


def test_the_commit_step_refuses_when_head_is_not_on_main(repo: Path) -> None:
    """#3352 keeps the commit on `main` rather than restoring a work branch:
    with `HEAD` parked on a branch, even an allowed change is not committed."""
    assert _git(repo, "checkout", "-b", "nightly-improvement-leftover").returncode == 0
    _edit(repo, "  confidence_floor: 0.4", "  confidence_floor: 0.7")

    run = _run_cli(
        "--repo", str(repo), "commit", "-m", "nightly-reflection: config test", repo=repo
    )

    assert run.returncode == guard.EXIT_DENIED, run.stdout
    assert guard.OFF_MAIN in run.stderr, run.stderr
    assert "nightly-improvement-leftover" in run.stderr, run.stderr
    assert _git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "1"


def test_a_repo_with_no_config_at_head_refuses_closed(tmp_path: Path) -> None:
    """No committed document to diff against is not a pass."""
    root = tmp_path / "bare"
    root.mkdir()
    assert _git(root, "init", "-b", "main").returncode == 0
    (root / "README").write_text("no config here\n")
    _git(root, "config", "user.name", "Config Guard Test")
    _git(root, "config", "user.email", "guard-test@example.invalid")
    assert _git(root, "add", "--", "README").returncode == 0
    assert _git(root, "commit", "-m", "baseline without config").returncode == 0

    verdict = guard.check_config_edit(root)

    assert verdict.allowed is False
    assert verdict.status == guard.DENIED
    assert "HEAD" in verdict.reason, verdict.reason
