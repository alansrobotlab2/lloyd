"""An active SKILL.md may document a script, and the script must be there.

The tool-name guard (`tests/test_skill_tool_names.py`) checks which *tools* a
skill names. It has never checked which *files*, and that is the wider class:
three `status: active` skills — `browser-session-extract`, `file-processor`,
`subagent-orchestrate` — document a Python executable in their own directory
that has never existed, 29 references between them (`grep -c` on each SKILL.md:
10, 10, 9), every documented command dying on its first line. All three are
`status: active`, so a session can
pick one up and follow it into a wall; `subagent-orchestrate` is the sharpest,
because `skills_search` on its own trigger phrase ranks it #1, ahead of the
working `subagent-driven-development`.

Two checks look like they might already cover this and do not — measured in the
tree this commit lands in, not inferred, because the naive version of the claim
("lint has no file-existence rule") is false and would not survive a reader:

  * `scripts/skill_lint.py` *does* have a script-path rule: `check_script_paths`
    (:367), reported under `missing_script` at :472. Its regex (:336-339) is
    anchored at `~/lloyd/` or `$HOME/lloyd/` and only over repo code roots
    (`_REPO_CODE_ROOTS`, :334), so a path under `~/obsidian/skills/` is not a
    shape it can match. Calling it directly returns `[]` for all three offenders
    and two repo paths for `system-health-check`, both already in its
    `KNOWN_ABSENT_SCRIPTS` ledger (:357) — the positive control that shows the
    rule is live and simply cannot see this directory. "Two" is the tree this
    file joined: the second of them,
    `tests/test_system_health_check_frontend_endpoint.py`, stopped being
    reported on 2026-09-23 when that skill's lone citation of it lost its
    `~/lloyd/` anchor, and #1417 retired the ledger entry the orphan left
    behind. `test_the_retired_repo_ledger_entry_is_not_an_allowlist_any_more`
    below is what keeps that retirement from becoming a blind spot.
  * the vault gate runs the real loaders, but `loader_errors` asks
    `skill_load_defect` about a touched skill (:180, :185) — that skill's own
    front matter and `SKILL.md`. It never opens a path the prose mentions, and
    the comment at :174-179 records why it deliberately does not treat
    `_load_skill(...) is None` as damage. A skill whose sibling script is
    missing loads fine, which is exactly how it ships with no script beside it.

A third check has to be conceded rather than argued around, because it is in the
suite this file joins: `tests/test_skill_tool_names.py`'s path guard already
lists all three of these scripts — as three rows of its `PATH_KNOWN_UNFIXED`
debt ledger, one of 22 entries there covering eight unrelated path shapes. So
the honest statement of what this file adds is not "nothing has ever seen these
paths" but that the class had no *test*: those three rows sit in a ledger whose
own comments record a fence-blind matcher and `~`-forms resolved against the
repo root, where one more debt row and one more deletion move them silently.
`MISSING_SCRIPT_EXEMPT` below is the same shape both lint and that ledger
already use — a debt with a name on every entry — narrowed to exactly one
subject and three rows, so a new absent path fails and an entry cannot leave
the ledger without the corpus noticing (clause 4 asks for exactly this set).

The matcher is a home-anchored path into `skills/<slug>/<file>.(py|sh)`, and it
charges every mention: a `## Location` heading, a fenced usage block, or a
backticked path all count. That choice was measured, not assumed, and the
measurement came out the opposite way from the intuition that a wide matcher
buys false positives. Re-running the scan with fenced blocks stripped, and
again with backticked spans stripped too, finds **zero** of the three offenders
— because all 29 mentions are inside fenced code blocks (`grep -o` per file: 10
fenced / 0 backticked / 0 bare prose, 10/0/0, 9/0/0). So the narrow variants
are not a quieter version of this check, they are a check that never fires on
the only defect the corpus actually has, and "run this command" is precisely
where a documented script belongs. Strictness here costs nothing either: the
widest matcher returns exactly those three skills and no others, so no
false-positive pressure was traded away to get them. The only exclusion in the
pattern is a lookbehind, so an embedded occurrence such as
`~/vault/obsidian/skills/…` reads as a note path, not an executable here.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VAULT = Path.home() / "obsidian"
SKILLS_DIR = VAULT / "skills"

#: A documented executable: a path anchored at `~`, `$HOME` or the real home
#: directory, pointing at a `.py` or `.sh` inside `skills/<slug>/`. The
#: lookbehind rejects an embedded occurrence (`~/vault/obsidian/skills/…`),
#: which is a note path and not an executable in the skills tree.
_SCRIPT_MENTION = re.compile(
    r"(?<![\w/.~-])"
    r"(?:~|\$HOME|/home/[A-Za-z0-9_.-]+)"
    r"/obsidian/skills/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+\.(?:py|sh)"
)

#: Empty since #409 archived the three skills whose documented script had
#: never been on disk (`browser-session-extract`, `file-processor`,
#: `subagent-orchestrate`, 2026-09-23). Keep it empty: a new entry is a
#: regression, and `test_the_exempt_set_stays_empty` below says so.
MISSING_SCRIPT_EXEMPT: dict[str, str] = {}


def _skill_bodies() -> list[tuple[str, str]]:
    """(skill name, SKILL.md text) for every skill the prompt advertises.

    Enumerated through `agent_mcp.skills.iter_active_skills` — the one walker
    the prompt index, `GET /api/skills`, Mission Control and skill_lint all
    share (#1294) — so a skill retired to `skills/.archived/` or carrying a
    quarantine `status:` drops out of this check automatically. A guard that
    kept its own directory walk would start failing on archived skills, which
    is the trap #409's original acceptance walked into: it grepped
    `~/obsidian/skills/` wholesale and therefore grepped the archive too.
    """
    from agent_mcp.skills import iter_active_skills

    out: list[tuple[str, str]] = []
    for skill in iter_active_skills():
        skill_file = skill.directory / "SKILL.md"
        if skill_file.is_file():
            out.append((skill.name,
                        skill_file.read_text(encoding="utf-8", errors="replace")))
    return out


def _documented_scripts(body: str) -> set[str]:
    """Vault-relative script paths this skill tells someone to run."""
    return {m.group(0).split("/obsidian/")[1]
            for m in _SCRIPT_MENTION.finditer(body)}


def _missing_scripts(exempt: frozenset[str] = frozenset()
                     ) -> dict[str, list[str]]:
    """skill name -> vault-relative script paths it names that are not on disk."""
    missing: dict[str, list[str]] = {}
    for name, body in _skill_bodies():
        for rel in sorted(_documented_scripts(body)):
            if rel in exempt:
                continue
            if not (VAULT / rel).exists():
                missing.setdefault(name, []).append(rel)
    return missing


def test_every_documented_skill_script_exists():
    """An active skill that says `python3 ~/obsidian/skills/<slug>/x.py` owes
    the reader an `x.py`. Nothing in lint or the vault gate checks that, so
    this is the check."""
    missing = _missing_scripts(exempt=frozenset(MISSING_SCRIPT_EXEMPT))
    assert missing == {}, (
        "active skills document scripts that are not on disk: "
        f"{missing}. Either write the script, name one that exists, or archive "
        "the skill."
    )


def test_the_exempt_set_stays_empty():
    """The ledger was exactly three paths, all owned by #409, and #409 archived
    all three skills (#410 clause 4). Asserted against the raw scan too, so the
    ledger cannot be re-grown into a quiet allowlist: a new absent script has to
    be written, repointed, or archived, not exempted.
    """
    assert MISSING_SCRIPT_EXEMPT == {}, (
        "MISSING_SCRIPT_EXEMPT grew again; fix the skill instead of exempting it")
    raw_paths = {rel for paths in _missing_scripts().values() for rel in paths}
    assert raw_paths == set(), (
        f"active skills document scripts that are not on disk: {sorted(raw_paths)}")


def test_a_newly_documented_absent_script_fails_the_check(tmp_path, monkeypatch):
    """The check must be able to fail, or it is decoration (#410 clause 4).

    A `Location` heading path, a fenced invocation, and a backticked mention of
    a different skill's script: all three are charged, which is the strictest of
    the three strictness levels measured for this commit — the live corpus is
    clean under it, so none of these is an exemption being exercised.
    """
    from types import SimpleNamespace

    slug_dir = tmp_path / "fresh-mined-skill"
    slug_dir.mkdir()
    (slug_dir / "SKILL.md").write_text(
        "---\nstatus: active\n---\n"
        "# SKILL: fresh-mined-skill\n\n"
        "## Location\n\n"
        "~/obsidian/skills/fresh-mined-skill/fresh_mined_skill.py\n\n"
        "```\npython3 ~/obsidian/skills/fresh-mined-skill/fresh_mined_skill.py run\n```\n\n"
        "It mirrors `~/obsidian/skills/some-other-skill/some_other_skill.py`.\n",
        encoding="utf-8")
    import agent_mcp.skills as skills_mod
    monkeypatch.setattr(
        skills_mod, "iter_active_skills",
        lambda *a, **k: iter([SimpleNamespace(name="fresh-mined-skill",
                                              directory=slug_dir)]))
    missing = _missing_scripts()
    assert missing == {"fresh-mined-skill": [
        "skills/fresh-mined-skill/fresh_mined_skill.py",
        "skills/some-other-skill/some_other_skill.py",
    ]}, f"a skill documenting an absent script must be reported: {missing}"


def test_the_mispathed_sibling_is_pinned_to_the_real_filename():
    """The three failures are a script that never existed. The fourth shape
    #923's survey reported — `system-health-check` running `system_check.py`
    while its directory holds `system_health_check.py` — is a wrong *filename*
    beside a real file, and it is not among today's three, so that reference
    has since been corrected. Pinning the corrected form is the half of this
    that cannot silently go false again: a skill that renames its script and
    leaves the prose behind fails here even though the old path still resolves
    to nothing in exempted form.
    """
    bodies = dict(_skill_bodies())
    skill = bodies.get("system-health-check")
    if skill is None:
        pytest.skip("system-health-check is not installed on this machine")
    named = {rel.split("/")[-1] for rel in _documented_scripts(skill)
             if rel.startswith("skills/system-health-check/")}
    assert "system_health_check.py" in named, (
        f"system-health-check must document its real script; it names {sorted(named)}")
    assert "system_check.py" not in named, (
        "system-health-check documents system_check.py again, which is not the "
        "file sitting in that directory")
    on_disk = {p.name for p in (SKILLS_DIR / "system-health-check").glob("*.py")}
    assert on_disk <= named, (
        f"scripts on disk that the skill never names: {sorted(on_disk - named)}")


def test_the_check_cannot_be_silently_widened_into_a_no_op():
    """`iter_active_skills` is the only walker, so if it ever yields almost
    nothing, the two live-corpus tests here would pass on an empty set."""
    bodies = _skill_bodies()
    assert len(bodies) > 100, (
        f"only {len(bodies)} active skills enumerated — the path check is "
        "vacuous, not green")


# ── the repo-side ledger's retirement route (#1417) ──────────────────────────

def _load_skill_lint():
    """Load `scripts/skill_lint.py` the way its callers do: by path.

    It is not an importable module (no package beside it on the path for the
    nightly invocation `python …/scripts/skill_lint.py`), which is why
    `tests/test_skills_single_walk.py` and
    `tests/test_fact_identity_one_action_one_fact.py` each `importlib` it rather
    than importing it. The code graph cannot see any of these reads — a
    spec-from-file location is a string, not an edge — so a change to the
    ledger's contents looks local to the graph and is not.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "skill_lint_under_existence_test", ROOT / "scripts" / "skill_lint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_retired_repo_ledger_entry_is_not_an_allowlist_any_more(tmp_path):
    """A skill citing the #1417-retired path is now reported, with no excuse.

    `check_script_paths` answers with `known_stale`: the ledger's note when a
    path is allowed, `""` otherwise, and `test_no_shipped_skill_names_an_absent_repo_script`
    charges only the empty-note hits. So an entry that outlives its citation is
    not merely untidy — it is a permanently open permit for that exact absent
    path, granted by a line nobody can point to in the corpus any more. Retiring
    `tests/test_system_health_check_frontend_endpoint.py` therefore has to be
    *observable*: before this commit the synthetic citation below came back with
    the note `"system-health-check; test never landed"` and was excused; after
    it, the same citation is an offender. That flip is the clause, and it is
    asserted as a whole-record comparison rather than `any(...)` so a hit with a
    note still present fails rather than passing a membership test.
    """
    skill_lint = _load_skill_lint()
    retired = "tests/test_system_health_check_frontend_endpoint.py"
    assert retired not in skill_lint.KNOWN_ABSENT_SCRIPTS, (
        "the entry is back, so the path is silently allowed again")

    hits = skill_lint.check_script_paths(
        f"Run `~/lloyd/{retired}` for the frontend certificate.\n",
        skill_dir=tmp_path, repo_root=ROOT)
    assert hits == [{"path": retired, "known_stale": ""}], (
        f"a skill citing the retired path must be reported as an unexcused "
        f"absent script: {hits}")
