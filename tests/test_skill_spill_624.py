"""The five sampled skills are spilled into sibling files, and nothing was lost (#624).

"Your skill is really a folder": the SKILL.md body is the index and the rules, the
detail lives beside it and is read when its index line says so. A spill can go
wrong three ways that nobody notices until a nightly run misbehaves, and each has
a test here:

* a section is dropped rather than moved — every `##`/`###` heading the pre-spill
  body carried must still be in the body or verbatim in a sibling;
* a sibling is written but never pointed at — each one needs an index line that
  names it and says when to read it, inside the chat injector's cut
  (`prefetch.SKILL_BODY_MAX`), or the capped route never learns it exists;
* a guardrail is moved out of the body — an upper-case hard-constraint line
  (`prefetch._HARD_CONSTRAINT_RE`, the carry-forward rule) must still be in the
  body, with its text intact (hard wraps may be undone).

The last corpus node is the spill's SIZE claim, and it guards two quantities rather
than one: the body-line collapse in each sampled skill, and the char saving across the
sample as a whole. Per-skill chars against the fixed `SPILL_BASELINE_BEFORE` instant
guard neither — that reading is what #2230 and #2406 each went red on; see the node.

The vault is live and shared, so the corpus tests carry `live_vault` and the gate
skips them. What the gate does see is the hermetic half: `skill_folder_text`, which
makes the lint read the folder, and the collapse rule of the SIZE node, replayed on a
synthetic vault built to put a body-line fall and a byte growth in the same row.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "skill_lint_spill", ROOT / "scripts" / "skill_lint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sl = _load()

VAULT = Path.home() / "obsidian"
_HEADING = re.compile(r"^#{2,3}\s+\S")


def _unfenced(text: str):
    fence = False
    for line in text.splitlines():
        if line.strip().startswith(("```", "~~~")):
            fence = not fence
            continue
        if not fence:
            yield line


def _norm(text: str) -> str:
    return " ".join(text.split())


def _baseline_rev() -> str:
    out = subprocess.run(["git", "-C", str(VAULT), "rev-list", "-1",
                          f"--before={sl.SPILL_BASELINE_BEFORE}", "HEAD"],
                         capture_output=True, text=True, timeout=30)
    rev = out.stdout.strip()
    if out.returncode != 0 or not rev:
        pytest.skip(f"no vault commit before {sl.SPILL_BASELINE_BEFORE}")
    return rev


def _before(name: str) -> str:
    out = subprocess.run(["git", "-C", str(VAULT), "show",
                          f"{_baseline_rev()}:skills/{name}/SKILL.md"],
                         capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        pytest.skip(f"skills/{name}/SKILL.md absent at the baseline")
    return out.stdout


def _now(name: str) -> tuple[str, list[Path]]:
    d = VAULT / "skills" / name
    if not (d / "SKILL.md").is_file():
        pytest.skip(f"{d} not present on this machine")
    siblings = sorted(p for p in d.glob("*.md") if p.name != "SKILL.md")
    return (d / "SKILL.md").read_text(encoding="utf-8"), siblings


def _body(text: str) -> str:
    return sl.parse_frontmatter(text)[1].strip("\n")


@pytest.fixture(scope="module", autouse=False)
def vault_git():
    if not (VAULT / ".git").exists():
        pytest.skip("the vault is not a git checkout here")


# ── the corpus: the five sampled skills as they are on disk ──────────────────

@pytest.mark.live_vault
@pytest.mark.parametrize("name", sl.SPILL_SAMPLE)
def test_the_body_is_within_the_ceiling(name, vault_git):
    text, _ = _now(name)
    lines = len(_body(text).splitlines())
    assert lines <= sl.MAX_BODY_LINES, f"{name}: {lines} body lines"


@pytest.mark.live_vault
@pytest.mark.parametrize("name", sl.SPILL_SAMPLE)
def test_every_removed_heading_is_verbatim_in_a_sibling(name, vault_git):
    before = _before(name)
    text, siblings = _now(name)
    body_heads = {ln.strip() for ln in _unfenced(_body(text)) if _HEADING.match(ln)}
    sibling_lines = {ln.strip() for p in siblings
                     for ln in p.read_text(encoding="utf-8").splitlines()}
    removed = [ln.strip() for ln in _unfenced(_body(before))
               if _HEADING.match(ln) and ln.strip() not in body_heads]
    lost = [h for h in removed if h not in sibling_lines]
    assert not lost, f"{name}: headings in neither the body nor a sibling: {lost}"


@pytest.mark.live_vault
@pytest.mark.parametrize("name", sl.SPILL_SAMPLE)
def test_each_sibling_has_an_index_line_inside_the_chat_cut(name, vault_git):
    text, siblings = _now(name)
    assert siblings, f"{name} has no sibling files — the spill has not been done"
    head = text[:sl.CHAT_SKILL_CUT]
    for sibling in siblings:
        lines = [ln for ln in head.splitlines() if sibling.name in ln]
        assert lines, (f"{name}: no line inside the first {sl.CHAT_SKILL_CUT} chars "
                       f"names {sibling.name}")
        # The condition: "read … when/before/after/only …" on the same line.
        assert any(re.search(r"\b[Rr]ead\b.*\b(when|before|after|only|at)\b", ln)
                   for ln in lines), (
            f"{name}: {sibling.name} is named but no line says when to read it: {lines}")


@pytest.mark.live_vault
@pytest.mark.parametrize("name", sl.SPILL_SAMPLE)
def test_hard_constraints_stay_in_the_body(name, vault_git):
    from app import prefetch
    before = _before(name)
    text, _ = _now(name)
    body = _norm(_body(text))
    hard = [ln.strip() for ln in _unfenced(_body(before))
            if prefetch._HARD_CONSTRAINT_RE.search(ln)]
    moved = [ln for ln in hard if _norm(ln) not in body]
    assert not moved, f"{name}: hard-constraint lines moved out of the body: {moved}"


def _spill_row_defect(row: dict) -> str | None:
    """Why one `spill_delta` row does NOT record a spill, or None if it does.

    The quantity is the body, counted in lines. A spill's whole act is moving detail
    out of `SKILL.md` into a sibling the index names, the ceiling that binds a landing
    is `MAX_BODY_LINES` — lines — and `vault_round.skill_body_findings` refuses a land
    on lines and never on chars, so a fall in body lines is both what a spill is and
    the one size number anything enforces. Whole-file chars against the pinned instant
    are not that quantity: the baseline is the last vault commit before the spill
    (`SPILL_BASELINE_BEFORE`) while the capped body keeps taking legitimate commits, so
    a byte-for-byte claim against it says "this file has never exceeded its 2026-09-24
    size", which a live skill cannot hold.

    A function and not inline asserts because the hermetic node below has to apply this
    same rule to a synthetic vault: a second copy of the predicate there would be a
    second rule, free to drift from the one the corpus is actually judged by.
    """
    before, after = row["before_body_lines"], row["after_body_lines"]
    if before is None or after is None:
        return (f"no body-line count on one side of the spill "
                f"(before={before}, after={after})")
    if after >= before:
        return f"the body did not collapse: {before} -> {after} lines"
    return None


@pytest.mark.live_vault
def test_the_spill_delta_is_nonzero_and_a_saving(vault_git):
    """What #624's spill can be claimed on, per skill and across the sample.

    Per skill: the body collapsed, by `_spill_row_defect`. Per sample: the chars are a
    net saving, which is a library-level claim — one skill's own file is free to grow.

    Free to grow because the baseline is a fixed instant and a spilled body is not a
    frozen file. `nightly-reflection-knowledge-write` measures +2,558 chars against the
    26,277 it had at that baseline (28,835 on disk) while its body sits at 309 -> 90
    lines: the guardrail commits that arrived after the spill — `3d119a0a` (#2367),
    `ff7e28fc` and `1f5900e1` (#2381, which reflowed so the new rule fit the pinned
    ceiling) — made the surviving 90 lines richer, which is exactly what a skill that
    has been capped and spilled is for. Asserting `delta_chars < 0` per skill instead
    guarded the byte size, and it has now gone red on that reading twice: #2230 at +84,
    repaired by moving prose to a sibling (vault `e3864ebf`), and #2406 at +2,558 three
    days later. That repair cannot be repeated anyway: `test_skill_lint_size.py`'s
    `SAMPLE_BASELINE` and `CAP_HEADROOM` pin each sampled skill's exact body-line count,
    so moving a line out to buy chars back turns that node red.

    A skill whose body never collapsed is still red here, a library that was never
    spilled is still red on the total, and `tests/test_skill_lint_size.py`
    `::test_spill_delta_of_zero_says_the_spill_has_not_been_done` already pins that an
    untouched vault totals 0. Both halves are shown on a synthetic vault below, where
    the rule can be made to fail without editing a live skill.
    """
    delta = sl.spill_delta(VAULT)
    rows = {r["name"]: r for r in delta["skills"]}
    assert set(rows) == set(sl.SPILL_SAMPLE)
    unrecorded: dict[str, str] = {}
    for name, row in sorted(rows.items()):
        defect = _spill_row_defect(row)
        if defect is not None:
            unrecorded[name] = defect
        assert row["siblings"], name
    assert not unrecorded, (
        "sampled skill(s) whose spill is not in the row: "
        + "; ".join(f"{name}: {why} (delta_chars {rows[name]['delta_chars']:+}, "
                    f"{len(rows[name]['siblings'])} sibling(s))"
                    for name, why in unrecorded.items()))
    assert delta["total_delta_chars"] < 0, (
        f"the sample is not a net saving: total_delta_chars "
        f"{delta['total_delta_chars']}. A library where the spill was never done looks "
        "exactly like this, which is the world this assert exists to refuse")


# ── the collapse rule, on a vault nobody has to edit to falsify it ───────────

_GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
            "GIT_AUTHOR_DATE": "2026-09-20T00:00:00",
            "GIT_COMMITTER_DATE": "2026-09-20T00:00:00",
            "PATH": "/usr/bin:/bin"}


def _synth_skill(root: Path, name: str, body_lines: int, pad: int) -> Path:
    """A synthetic `SKILL.md` with `body_lines` body lines of about `pad`+18 chars
    each, so a test can move lines and bytes in opposite directions — which is the
    whole question the delta node answers."""
    d = root / name
    d.mkdir(parents=True)
    body = "\n".join(f"line {i} of {name} " + "x" * pad for i in range(body_lines))
    (d / "SKILL.md").write_text(
        "---\n"
        f"description: Use this skill when testing {name} spills.\n"
        "tags: [test]\n"
        "---\n\n" + body + "\n", encoding="utf-8")
    return d / "SKILL.md"


def _committed_vault(vault: Path) -> Path:
    """`git init` + one commit dated 2026-09-20, the same throwaway-repo recipe as
    `tests/test_skill_lint_size.py::test_spill_delta_reads_before_from_history_and_after_from_disk`,
    so the baseline `spill_delta` reads is history and not the working tree."""
    subprocess.run(["git", "init", "-q", str(vault)], check=True)
    for args in (["add", "-A"], ["commit", "-qm", "baseline"]):
        subprocess.run(["git", "-C", str(vault), *args], check=True,
                       capture_output=True, env=_GIT_ENV)
    return vault


def _spill_two_ways(vault: Path) -> None:
    """Fill `vault` with three skills and commit them: `grown` and `shrunk` at 240 body
    lines each, `flat` at 120 and never touched again. Then spill the first two —
    `grown` reflowed into 90 lines that carry MORE bytes than the 240 they replace
    (guardrail prose added), `shrunk` cut to 90 lines with the detail moved to a
    sibling, so it saves bytes. `flat` is the skill nobody spilled."""
    skills = vault / "skills"
    grown = _synth_skill(skills, "grown", 240, pad=20)
    shrunk = _synth_skill(skills, "shrunk", 240, pad=20)
    _synth_skill(skills, "flat", 120, pad=20)
    _committed_vault(vault)
    for path in (grown, shrunk):
        head, _, rest = path.read_text(encoding="utf-8").partition("\n\n")
        if path is grown:
            kept = [f"line {i} of grown " + "x" * 100 for i in range(89)]
        else:
            kept = rest.splitlines()[:89]
        path.write_text(head + "\n\n" + "\n".join(kept)
                        + "\n- `detail.md`: read when the detail is needed\n",
                        encoding="utf-8")
        (path.parent / "detail.md").write_text(
            "\n".join(rest.splitlines()) + "\n", encoding="utf-8")


def test_the_collapse_rule_passes_a_collapsed_body_with_grown_chars_and_refuses_a_flat_one(tmp_path):
    """The rule `test_the_spill_delta_is_nonzero_and_a_saving` applies, on a vault
    built to separate its two halves.

    `grown` is the shape that made #2406 red on the live vault and must be GREEN here:
    its body fell 240 -> 90 lines while the whole file ended up larger than its baseline
    char count, because a guardrail was added to the 90 lines that survived. `flat` is
    the shape that must stay RED: its body never moved, so no spill happened, and the
    only reason to call it one is that a char count exists for it at all.

    The two are in one vault so the third reading is visible too: the sample total is
    still a saving while one member is a growth — which is precisely the distinction the
    node exists to make and the per-skill char assert could not.
    """
    vault = tmp_path / "vault"
    _spill_two_ways(vault)
    delta = sl.spill_delta(vault, names=("grown", "shrunk", "flat"),
                           before="2026-09-21T00:00:00")
    rows = {r["name"]: r for r in delta["skills"]}

    grown = rows["grown"]
    assert (grown["before_body_lines"], grown["after_body_lines"]) == (240, 90), grown
    assert grown["delta_chars"] > 0, (
        f"the fixture is meant to be the grown-chars case, and it measured "
        f"{grown['delta_chars']:+} — the PASS below would be vacuous")
    assert _spill_row_defect(grown) is None, (
        f"a collapsed body with grown chars is a completed spill, but the rule "
        f"returned: {_spill_row_defect(grown)}")

    flat = rows["flat"]
    assert flat["delta_chars"] == 0 and flat["before_body_lines"] == flat["after_body_lines"]
    defect = _spill_row_defect(flat)
    assert defect is not None and "did not collapse" in defect, (
        f"an untouched skill passed the spill rule: {defect!r}")

    assert _spill_row_defect(rows["shrunk"]) is None, "the byte-saving spill must pass too"
    assert delta["total_delta_chars"] < 0, (
        f"the library must still read as a net saving while one member grew: "
        f"{delta['total_delta_chars']}")

    never_root = tmp_path / "never-spilled"
    _synth_skill(never_root / "skills", "flat", 120, pad=20)
    never = sl.spill_delta(_committed_vault(never_root),
                           names=("flat",), before="2026-09-21T00:00:00")
    assert never["total_delta_chars"] == 0, (
        f"a vault that was never spilled into totals {never['total_delta_chars']} chars, "
        "not 0 — and 0 is what the corpus node's `total_delta_chars < 0` guard has to "
        "refuse, because it is the shape of a world where the spill never happened")
    assert _spill_row_defect(never["skills"][0]) is not None


# ── the lint reads the folder, so a spill hides nothing from it ──────────────

def test_skill_folder_text_appends_every_sibling_markdown(tmp_path):
    d = tmp_path / "demo"
    d.mkdir()
    (d / "SKILL.md").write_text("body\n", encoding="utf-8")
    (d / "b.md").write_text("second\n", encoding="utf-8")
    (d / "a.md").write_text("first\n", encoding="utf-8")
    (d / "notes.txt").write_text("not markdown\n", encoding="utf-8")
    text = sl.skill_folder_text(d, "body\n")
    assert text.index("body") < text.index("first") < text.index("second")
    assert "not markdown" not in text
    assert text.count("body") == 1, "SKILL.md must not be read twice"


def test_lint_finds_a_phantom_tool_that_moved_into_a_sibling(tmp_path):
    """The counterfactual for reading the folder: the same phantom name, moved
    out of SKILL.md into a sibling, is still a finding."""
    from agent_mcp.skills import ActiveSkill
    d = tmp_path / "skills" / "spilled"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\ndescription: Use when testing a spill.\ntags: [t]\n---\n\n"
        "Read `detail.md` when you need the detail.\n", encoding="utf-8")
    (d / "detail.md").write_text("Call `web_search` for every lookup.\n", encoding="utf-8")
    rec = ActiveSkill(name="spilled", directory=d, frontmatter={})
    result = sl.lint(skill_records=[rec])
    assert [p["name"] for p in result["phantom"]] == ["spilled"], result["phantom"]


@pytest.mark.live_vault
def test_the_block_the_body_defers_to_is_actually_where_the_body_says_it_is():
    """#2188: the pointer's PROMISE, which no other node in this file makes.

    `test_each_sibling_has_an_index_line_inside_the_chat_cut` already owns the two
    halves of clause 4 that are about the index — the sibling is NAMED inside the
    `CHAT_SKILL_CUT`, and the naming line carries a `Read … when/before/after/only`
    condition — and it runs over every sibling of every sampled skill, so this node
    does not re-do that work. What nothing else checks is the other side of a deferral:
    the body's §2e–2f paragraph tells the reader that `### 2e. Pattern Output Files`
    and `### 2f. Knowledge Log` are in `steps-2b-2f.md`. A naming line says a file
    exists; this says the two specific sections the body shed are the ones a reader
    will find on opening it. Move either heading elsewhere, or drop the paragraph that
    quotes them, and this is red while the index node stays green — which is the drift
    a spill leaves behind: prose that defers to a place that no longer holds the thing.
    """
    name = "nightly-reflection-knowledge-write"
    text, siblings = _now(name)
    body = _body(text)
    # The deferral paragraph, located by the section that makes it rather than by the
    # first line containing the filename — `## Files in this skill` names the same file
    # on its index bullet, and that bullet is the index node's business, not this one's.
    section = body.split("\n### 2e", 1)
    assert len(section) == 2, "the body has no §2e section left to defer with"
    pointer = "### 2e" + section[1].split("\n## ", 1)[0]
    assert "steps-2b-2f.md" in pointer, (
        f"the body's §2e–2f paragraph no longer says where the detail went: {pointer[:120]}")
    # The promise is matched on collapsed whitespace: the body wraps its prose, and a
    # heading quoted mid-sentence can break across lines. What has to survive the wrap
    # is the NAME of the section, which is what a reader greps for in the sibling.
    promised = _norm(pointer)
    held = "\n".join(p.read_text(encoding="utf-8") for p in siblings)
    for heading in ("### 2e. Pattern Output Files", "### 2f. Knowledge Log"):
        assert _norm(heading) in promised, (
            f"the body's deferral paragraph does not name {heading} — it points at the "
            "file but not at the section, so a reader opening it has nothing to find")
        assert heading in held, (
            f"{heading} is not the verbatim heading of any sibling — the body points at "
            "a section that is not there")
