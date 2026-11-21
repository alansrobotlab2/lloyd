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
    *observable*, and it has stayed observable through a second flip: the entry
    began as an excuse for an absent path, #1417 removed it once the skill's lone
    citation lost its `~/lloyd/` anchor — which made that same citation an
    unexcused offender, the flip this node shipped with — and #2129 then landed the
    file, after which the citation is no finding at all, because a path that exists
    is not drift and an allowance for it would permit nothing.

    So the claim is two records over ONE citation and two roots, and they only mean
    anything together. Over a tree without the file the scan must still return the
    unexcused hit: that is what proves the rule is not blind, since a scan that
    reported nothing at all would satisfy "no findings" for entirely the wrong
    reason. Over the real checkout, where the file's existence is asserted beside
    it as a node-for-node fact, the scan must return nothing. Both are whole-record
    comparisons rather than `any(...)`, so a hit that came back carrying a ledger
    note fails here instead of slipping through a membership test.
    """
    skill_lint = _load_skill_lint()
    retired = "tests/test_system_health_check_frontend_endpoint.py"
    assert retired not in skill_lint.KNOWN_ABSENT_SCRIPTS, (
        "the entry is back, so the path is silently allowed again")
    citation = f"Run `~/lloyd/{retired}` for the frontend certificate.\n"

    empty_checkout = tmp_path / "checkout"
    (empty_checkout / "tests").mkdir(parents=True)
    hits = skill_lint.check_script_paths(
        citation, skill_dir=tmp_path, repo_root=empty_checkout)
    assert hits == [{"path": retired, "known_stale": ""}], (
        f"the citation must be reported as an unexcused absent script whenever the "
        f"file is not in the tree it is scanned against: {hits}")

    assert (ROOT / retired).is_file(), (
        f"{retired} is the file #2129 was opened to write; with it gone, the empty "
        "result below would be the rule going blind, not the citation being met")
    assert skill_lint.check_script_paths(
        citation, skill_dir=tmp_path, repo_root=ROOT) == []


# ── #4367: the absolute path a command tells the reader to open ──────────────
#
# The class above resolves a cited *repo script* against the checkout. It cannot
# see a command that names a file by absolute path, because `_SCRIPT_PATH_RE`
# needs a `~/lloyd/` anchor and a code extension, and
# `git -C ~/lloyd grep -n 'shutil.which|os.path.exists' -- scripts/skill_lint.py`
# returns nothing: no shipped rule ever stats an absolute operand. Two live
# instances cost a step each until vault commit 07379a6ac (2026-11-20 09:42:55):
#
#   * `service-health-check/SKILL.md:463` ran `ls -la
#     /home/alansrobotlab/lloyd/autonomy.db` — a store that left the code tree
#     with #1381, so the step answered `No such file or directory` on every
#     following. It now reads `ls -1t ~/lloyd-data/autonomy-runs | head`.
#   * `subtitle-processing/SKILL.md:70` fell back to
#     `/home/alansrobotlab/.local/bin/yt-dlp`, a path that has never held the
#     binary here (`which yt-dlp` answers `/usr/bin/yt-dlp`).
#
# Both were the instruction, not the model's improvisation: the reader ran the
# line and the line was wrong. That is the failure the paper in #4367 names —
# a skill stores a program from past experience, the environment moves, the
# agent compensates at run time, and the stored state is never updated. The
# runtime join that would price the compensation measured a null (15 attributed
# failures across 57 days of traces, 0 surviving inspection), so what is
# buildable today is the static half: report the operand, with the line that
# authored it.
#
# The rule is deliberately narrow, and every narrowing is a measured noise class
# from the 18 distinct dead-under-home absolutes the shipped walker reads over the
# live corpus (2026-11-20), all 18 of which stay silent while the control below
# still reports: a name carrying a date or timestamp
# is an EXAMPLE of a per-run artefact the reader must substitute (`kg-health-
# 2026-09-20T061402Z.json`), a field carrying `$VAR`, `$(…)`, a glob, `…/x` or an
# `=` is not a literal operand, a `>` target is written rather than read, an
# output transcript's `ls:` / `Cause:` is a prefix rather than a verb, and a
# non-shell fence (`json`, `yaml`, `xml`, `sql`, `python`) is not a command this
# box would run. What survives is one shape: a shell read verb, in command
# position, naming a path under this box's home root that is not on disk.

#: One fenced block, so a test can name the line a finding must be reported at:
#: the prefix below is 5 lines, the fence marker is line 6, so the first command
#: line is line 7.
_BODY_PREFIX = "---\nstatus: active\n---\n# demo-skill\n\n"


def _fenced(lines, lang: str = "bash") -> str:
    return "".join(f"```{lang}\n" + "".join(f"{ln}\n" for ln in lines) + "```\n")


def _operand_hits(hits) -> list[dict]:
    """Only the #4367 command-operand records, so the repo-script class above
    cannot satisfy these nodes by answering with a `~/lloyd/...` citation."""
    return [h for h in hits if h.get("kind") == "command-operand"]


def test_a_dead_absolute_operand_is_reported_as_skill_path_and_line(tmp_path,
                                                                   monkeypatch):
    """`ls -la /home/<user>/lloyd/autonomy.db` in a bash fence is the instruction
    that was live until 2026-11-20, and `check_script_paths` cannot see it.

    Whole-record comparison, because the four fields are the clause: `skill` and
    `path` are not enough to fix it — the reader needs the line and the command
    that produced it. `kind` marks the class, `known_stale` stays empty.
    """
    skill_lint = _load_skill_lint()
    dead = Path.home() / "lloyd" / "autonomy.db"
    assert not dead.is_file(), (
        f"{dead} has reappeared in the code tree; the report path would then be "
        "testing a hit that is not a finding")
    body = _BODY_PREFIX + _fenced([f"ls -la {dead}"])

    hits = _operand_hits(skill_lint.check_script_paths(
        body, skill_dir=tmp_path, repo_root=tmp_path))
    assert hits == [{"kind": "command-operand", "path": str(dead), "line_no": 7,
                     "command": f"ls -la {dead}", "known_stale": ""}], (
        f"the operand must be reported as path + line + command, with no excuse: "
        f"{hits}")

    # …and through the report row, which is where a human reads it. The row must
    # carry all four fields: the skill, the path, the line and the command. The
    # MISSING_SCRIPT render path had a defect class of its own before (#3678's
    # `test_the_lint_report_and_totals_line_carry_the_missing_script_count` names
    # it: a run whose only defect was an absent script once printed
    # `All skills pass lint`), so an operand that dies between `check_script_paths`
    # and the rendered table has not been reported.
    report = skill_lint.render_report({
        "generated_at": "2026-11-20T00:00:00", "total": 1,
        "dead": [], "missing_desc": [], "drift": [], "duplicates": [],
        "stale": [], "phantom": [],
        "missing_script": [{"name": "demo-skill", "path": "/x/SKILL.md",
                            "scripts": hits}],
    })
    row = [ln for ln in report.splitlines() if "autonomy.db" in ln]
    assert len(row) == 1 and row[0].startswith("| `demo-skill` |"), (
        f"the operand must appear as exactly one MISSING_SCRIPT row naming its "
        f"skill: {row}")
    assert f"`{dead}`" in row[0] and ":7" in row[0] and f"`ls -la {dead}`" in row[0], (
        f"the row must name path, line and command, not just the path: {row[0]}")

    # …and the ledger cannot excuse it. `KNOWN_ABSENT_SCRIPTS` is the repo-script
    # class's dated-debt list; an entry naming this operand must not silence it,
    # because `lint()` charges only the empty-`known_stale` hits and the corpus-
    # wide assertion in `test_no_shipped_skill_names_an_absent_repo_script` is
    # what turns a dead operand red. The exempt set stays empty for this class.
    monkeypatch.setitem(skill_lint.KNOWN_ABSENT_SCRIPTS, str(dead),
                        "an excuse that must not apply")
    excused = _operand_hits(skill_lint.check_script_paths(
        body, skill_dir=tmp_path, repo_root=tmp_path))
    assert excused == hits, (
        "a KNOWN_ABSENT_SCRIPTS entry silenced a command operand, which is the "
        f"permit this class must never take: {excused}")


def test_the_measured_noise_classes_produce_no_finding(tmp_path):
    """Every noise class the clause names stays silent — numbered 1-8 below, the
    clause's seven items with the example home and the `path/to/<x>` template split
    apart — and the one real defect sitting beside them still reports.

    The control is in the same body on purpose: a scan that went blind would
    satisfy "no findings" for the wrong reason, which is the failure mode
    `test_the_retired_repo_ledger_entry_is_not_an_allowlist_any_more` names.
    """
    skill_lint = _load_skill_lint()
    home = str(Path.home())
    control = f"{home}/lloyd/autonomy.db"
    body = _BODY_PREFIX + _fenced([
        "# 1 scratch under /tmp, named per run by the run itself",
        "tail -3 /tmp/4367-no-such-run.txt",
        "# 2 another account's example home, and 3 a template path",
        "cat /home/user/lloyd-data/_pipeline/never.json path/to/<x>.json",
        "# 4 a field carrying $VAR, a glob, or $(…)",
        f"tail -3 {home}/lloyd-data/_pipeline/reflection/signals-latest-$STAMP.md",
        f"ls {home}/lloyd-data/_pipeline/metrics/kg-health-*.json",
        f"cat $(echo {home}/nowhere/x.json)",
        "# 5 a dated example artefact: the reader substitutes its own name",
        f"ls -l {home}/lloyd-data/_pipeline/metrics/kg-health-2026-09-20T061402Z.json",
        "# 6 an elided …/x head, and 7 a redirect's write target",
        f"cat {home}/lloyd/.../kg.sqlite",
        f"echo done > {home}/lloyd-data/_pipeline/reflection/4367-note.md",
        "# the control — a literal, undated operand that is not on disk",
        f"ls -la {control}",
    ]) + (
        "8. a retirement note quoting a moved path, in prose, with no command: "
        f"its old root was `{home}/lloyd/_pipeline` (#1381), and the diagnostic "
        f"used to print `ls: cannot access '{home}/lloyd/_pipeline/"
        "vault-derived/kg.sqlite'`.\n")

    hits = _operand_hits(skill_lint.check_script_paths(
        body, skill_dir=tmp_path, repo_root=tmp_path))
    assert [h["path"] for h in hits] == [control], (
        "exactly the control may be reported; every other shape above is a "
        f"noise class measured over the live corpus: {hits}")
    assert hits[0]["line_no"] == 21, (
        f"the control is line 21 of the body above, so a finding that names "
        f"another line does not name the instruction: {hits}")


def test_a_non_shell_first_token_in_a_fence_produces_no_finding(tmp_path):
    """A fence is only a command this box can run if it is a shell fence, and an
    operand is only a claim if the verb in front of it reads.

    Both halves matter: `cat` inside a `sql` or `python` fence is sample text in
    someone else's language, and a first token that is not a read command —
    `defunct-analyzer`, a bare binary — is the PATH question #4367 puts off, not
    a dead operand. Nothing here stats the binary.

    The fifth block is the same dead path behind a shell `cat`, so this node cannot
    pass by never matching: exactly that one reports, at its own line.
    """
    skill_lint = _load_skill_lint()
    gone = f"{Path.home()}/lloyd-data/_pipeline/4367-not-here.tsv"
    assert not Path(gone).exists()
    body = _BODY_PREFIX + (
        _fenced([f"cat {gone}"], lang="sql")
        + _fenced([f"cat {gone}"], lang="python")
        + _fenced([f"defunct-analyzer {gone}"], lang="bash")
        + _fenced([f"defunct-analyzer {Path.home()}/no-such-binary-at-all"])
        # The control. _BODY_PREFIX is 5 lines and every _fenced() block is 3
        # (marker, command, close), so the fifth block's command is line 19.
        + _fenced([f"cat {gone}"], lang="bash"))

    hits = _operand_hits(skill_lint.check_script_paths(
        body, skill_dir=tmp_path, repo_root=tmp_path))
    assert hits == [{"kind": "command-operand", "path": gone, "line_no": 19,
                     "command": f"cat {gone}", "known_stale": ""}], (
        "only the shell-fence control may be reported: a non-shell fence, or a "
        f"first token that is not a read command, is not an instruction this rule "
        f"may name — and if the control is missing, the rule is blind, not "
        f"restrained: {hits}")


def test_no_dead_command_operand_survives_in_the_live_skill_library():
    """The rule is ON and the live corpus is clean: zero findings over every
    skill `lint()` walks, and the two instructions vault 07379a6ac repointed run.

    The corpus is `iter_active_skills()` — 208 skills on 2026-11-20, out of 218 skill
    directories at the top of the tree (219 entries counting `.archived`) — because
    that is the set the shipped `lint()` walks and the only set a model can reach.
    Counting every archived skill, `rglob` finds 381 SKILL.md files there, so the
    number asserted below is a floor that a two-skill scan would fail, not a count.
    The two shapes this rule would report in the 10 non-active directories —
    `.archived/bash-permission-handling:40` `ls -la /home/alansrobotlab/target-dir/
    2>&1`, and `.archived/nightly-skills-path-resolution:107` `ls -la
    /home/alansrobotlab/lloyd/mcp-servers/backlog.py  # Verify first` — are recorded
    on #4367 rather than silenced: an archived skill cannot mislead a turn, which is
    `lint()`'s own stated reason for not walking them, and the sibling assertion
    `test_no_shipped_skill_names_an_absent_repo_script` is scoped the same way.
    """
    from agent_mcp.skills import iter_active_skills

    skill_lint = _load_skill_lint()
    records = list(iter_active_skills())
    assert len(records) >= 200, (
        f"only {len(records)} active skills — the scan is vacuous, not green")
    assert len(list(SKILLS_DIR.rglob("SKILL.md"))) >= 218, (
        f"only {len(list(SKILLS_DIR.rglob('SKILL.md')))} SKILL.md files under "
        f"{SKILLS_DIR} — the tree moved, this assertion needs re-pointing")

    findings: dict[str, list[dict]] = {}
    for record in records:
        folder = Path(record.directory)
        body = skill_lint.skill_folder_text(
            folder, (folder / "SKILL.md").read_text(encoding="utf-8",
                                                     errors="replace"))
        hits = _operand_hits(skill_lint.check_script_paths(body, skill_dir=folder))
        if hits:
            findings[record.name] = hits
    assert findings == {}, (
        "live skills tell a reader to open a path this box does not have: "
        f"{findings}")

    # A witness that the matcher reaches command position in a real body rather
    # than passing by never arriving: `service-health-check` runs `ls -la` on a
    # store that IS here, so the same line with one name changed must report.
    # Without this, a rule that silently matched nothing in the corpus would look
    # identical to a corpus that has no defect.
    health = (SKILLS_DIR / "service-health-check" / "SKILL.md").read_text(
        encoding="utf-8", errors="replace")
    live = [(ln, m.group(1)) for ln, txt in enumerate(health.splitlines(), 1)
            if (m := re.match(r"\s*ls -la (/\S+)$", txt))
            and Path(m.group(1)).exists()]
    assert live, ("service-health-check no longer holds an `ls -la /…` step, so "
                  "this witness needs a live absolute operand to be built on")
    line_no, real_path = live[0]
    mutated = "\n".join(
        (f"ls -la {Path.home()}/4367-not-on-this-box.db" if n == line_no else txt)
        for n, txt in enumerate(health.splitlines(), 1))
    witness = _operand_hits(skill_lint.check_script_paths(
        mutated, skill_dir=SKILLS_DIR / "service-health-check"))
    assert [h["path"] for h in witness] == [f"{Path.home()}/4367-not-on-this-box.db"], (
        f"the corpus scan found nothing anywhere, including in a body built to "
        f"match — the matcher is blind, not the corpus clean: {witness}")
    assert witness[0]["line_no"] == line_no and witness[0]["command"].startswith("ls -la "), (
        f"the witness must name line {line_no} (`{real_path}`), not another: "
        f"{witness}")

    # The two repoints, run out of the files rather than from a copy typed into
    # this test, so what is proven is the instruction a session actually reads.
    for skill, needle in (("service-health-check", "autonomy-runs"),
                          ("subtitle-processing", "yt-dlp")):
        body = (SKILLS_DIR / skill / "SKILL.md").read_text(
            encoding="utf-8", errors="replace")
        step = [ln.strip() for ln in body.splitlines()
                if needle in ln and ln.strip().startswith(("ls ", "which "))]
        assert step, f"{skill} no longer carries a `{needle}` step to run"
        command = step[0]
        import subprocess
        done = subprocess.run(command, shell=True, cwd=str(Path.home()),
                              capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, (
            f"{skill}: `{command}` exits {done.returncode} on this box — "
            f"{done.stderr.strip()[:200]}")


def test_the_scan_writes_nothing_under_the_skills_tree(tmp_path):
    """Report-only, per #882's 2026-10-03 ruling for this surface: the scan names
    an offender, it never repairs one. A digest of the whole skills tree — every
    path, its size and its mtime — before and after the scan runs over it.
    """
    import hashlib

    def digest() -> str:
        h = hashlib.sha256()
        for p in sorted(SKILLS_DIR.rglob("*")):
            st = p.stat()
            h.update(f"{p}|{st.st_size}|{st.st_mtime_ns}|{'d' if p.is_dir() else 'f'}"
                     .encode())
        return h.hexdigest()

    before = digest()
    test_no_dead_command_operand_survives_in_the_live_skill_library()
    assert digest() == before, (
        "the scan changed something under ~/obsidian/skills; it is a report, not "
        "a repair")
