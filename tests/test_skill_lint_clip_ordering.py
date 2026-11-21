"""#4374 — the lint arm for a description clipped *before* its trigger.

`DESCRIPTION` measures one property: `len(index_desc) > clip`. Length is exactly
the property the index's clip preserves or destroys, so a rewrite that moves a
when-to-use clause from character 172 to character 0 — which is what #4373 does
by hand — changes no count in this report anywhere, and `check_description_drift`
cannot catch it either (its Path 2 passes any ≤10-character opener, verb or not,
which is why `CATEGORY_TRUST["DRIFT"]` is a `no`). `BURIED_TRIGGER` asks the
question length cannot: of the skills the index *actually renders* with a clipped
description, which have no `TRIGGER_WORDS` marker inside the visible window?

Three arms, and they are not equally solid — which is what the trust cell says:
(a) the rendered index line carries a description, measured through the
    renderer's own grant arithmetic (`_skills_index_lines`); (b) the folded
    description exceeds `_skills_index_settings()["max_description_chars"]`;
    (c) no `TRIGGER_WORDS` token occurs inside that many leading characters —
    advisory, a word list over prose, not a parse.

Hermetic except for the last node, which re-derives the live library's firing
set from the renderer and demands the lint agree with it.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent_mcp.skills as skills_mod  # noqa: E402
from app import prompt_builder as pb  # noqa: E402
from app.config import CONFIG  # noqa: E402


def _load():
    """`scripts/skill_lint.py` by path, so this file measures that file."""
    spec = importlib.util.spec_from_file_location(
        "skill_lint_clip_ordering", ROOT / "scripts" / "skill_lint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sl = _load()

#: 171 + 1 + 45 = 217 chars. The only TRIGGER_WORDS tokens (`use` at 172, `when`
#: at 176) sit past the 100-character clip, so the window the model sees —
#: `Rebuild the entity-vector index the semantic seeding leg reads, publish its
#: rows and live counts to ` — holds none of them.
_BURIED_HEAD = ("Rebuild the entity-vector index the semantic seeding leg reads, "
                "publish its rows and live counts to the vault report, and "
                "reconcile the counter table written by the sweep.")
_BURIED_TAIL = "Use when the seeding leg reads a stale index."
BURIED_DESC = _BURIED_HEAD + " " + _BURIED_TAIL

#: The same 217 characters with the trigger sentence moved to the front: same
#: length, same words, count 1 → count 0. This is #4373's reorder.
FRONT_DESC = _BURIED_TAIL + " " + _BURIED_HEAD

#: 206 chars, no marker inside the first 100 (`use` sits at 155). The second
#: buried description exists so the names-only test can fail arm (a) ALONE: the
#: names-only skill has to be a skill that arms (b) and (c) would convict, or the
#: test would pass for any reason at all — which is how the first draft of this
#: file passed while the shipped arm tested dict membership instead of the value.
BURIED2_DESC = ("Publish one row per tracked channel into the digest table, refresh the "
                "fetch timestamps, and prune the rows older than the retention window "
                "set in config. Use when the digest table is missing a channel row.")

#: 151 chars whose only marker (`whenever` at 81) is inside the 100-char window,
#: so it does not fire at today's clip and does fire at a clip of 40.
MID_MARKER_DESC = ("Rebuild the entity-vector index the semantic seeding leg reads, "
                   "publish its rows whenever the vault report is missing, and "
                   "reconcile the counter table.")

#: 63 chars with no marker at all: under today's clip, so the length arm stops it.
SHORT_DESC = "Rebuild the entity-vector index the semantic seeding leg reads."


def _skill(root: Path, name: str, description: str) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    fm = ["---", f"name: {name}", f"description: {json.dumps(description)}",
          "tags: [x]", "---", "", f"BODY of {name}."]
    (d / "SKILL.md").write_text("\n".join(fm) + "\n", encoding="utf-8")


def _active():
    return list(skills_mod.iter_active_skills())


def _budget_describing_only(active, counts: dict, chosen: str, max_desc: int) -> int:
    """The smallest budget at which exactly `chosen` is granted a description.

    Found by asking the renderer, not by arithmetic: `_skills_index_lines` tests
    `total + len(line) + len(" — ") + max_desc` against the budget while `total`
    already holds that line's names-only cost, so hand-deriving the marginal price
    of one description gets it wrong by a whole line — the first version of this
    helper did, and the node it fed passed anyway. Probing is the only reading of
    the rule that cannot drift from the rule.

    Raises if the probe never isolates `chosen`: a fixture whose budget arithmetic
    silently describes nobody would make the names-only node vacuous.
    """
    start = len("\n".join(pb._skills_index_lines(active, counts, budget=1,
                                                 max_desc=max_desc)))
    for budget in range(start, start + 4000):
        described = set()
        for line in pb._skills_index_lines(active, counts, budget=budget,
                                           max_desc=max_desc)[1:]:
            name, sep, _ = line[2:].partition(" — ")
            if sep:
                described.add(name)
        if described == {chosen}:
            return budget
    raise AssertionError(f"no budget isolates {chosen!r}; the fixture cannot test arm (a)")


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """One fixture root, `skills.index` on, the rank alphabetical."""
    root = tmp_path / "skills"
    monkeypatch.setattr(skills_mod, "SKILLS_DIRS", [root])
    monkeypatch.setattr(pb, "_CANON_SKILLS_DIRS", None)
    monkeypatch.delenv("LLOYD_OVERLAY_DIR", raising=False)
    skills_cfg = dict(CONFIG.get("skills") or {})
    skills_cfg["index"] = {"descriptions": True, "budget_chars": 12000,
                           "max_description_chars": 100}
    monkeypatch.setitem(CONFIG, "skills", skills_cfg)
    monkeypatch.setattr(pb, "_skill_rank_counts", lambda: {})
    return root


def _fired(result: dict) -> dict[str, dict]:
    return {r["name"]: r for r in result["buried_trigger"]}


# ── the fixtures ─────────────────────────────────────────────────────────────

def test_the_fixture_descriptions_hold_the_positions_the_tests_assert():
    """A fixture whose offsets drifted would make every node below vacuous.

    Measured here against the module's own word list and the renderer's own clip,
    so editing one of these strings into an accidental `use` fails this node
    rather than quietly turning a firing test into a passing one.
    """
    clip = pb._skills_index_settings()["max_description_chars"]
    assert len(BURIED_DESC) == len(FRONT_DESC) == 217, "the reorder pair stopped being length-neutral"
    assert len(MID_MARKER_DESC) > clip >= len(SHORT_DESC)
    assert sl.trigger_marker_in_window(BURIED_DESC, clip) is False
    assert sl.trigger_marker_in_window(FRONT_DESC, clip) is True
    assert sl.trigger_marker_in_window(MID_MARKER_DESC, clip) is True
    assert sl.trigger_marker_in_window(SHORT_DESC, clip) is False
    assert sl.trigger_marker_in_window(BURIED2_DESC, clip) is False
    assert sl.first_trigger_offset(BURIED_DESC) == (172, "use")
    assert sl.first_trigger_offset(BURIED2_DESC) == (155, "use")
    assert sl.first_trigger_offset(SHORT_DESC) is None


# ── clause 1: all three arms, and only all three ─────────────────────────────

def test_the_buried_row_fires_and_a_front_loaded_or_short_one_does_not(tree):
    """Arms (b) and (c) each veto on their own, with every line described.

    Three skills, all rendered with a description (the budget is generous), so
    arm (a) holds for all three and cannot be what separates them: only the
    buried one fires. `FRONT_DESC` is the same 217 characters as `BURIED_DESC`
    with its trigger at the front, so nothing but position differs; `SHORT_DESC`
    is under the clip, which is the existing DESCRIPTION arm's business, not
    this one's.
    """
    _skill(tree, "aaa_buried", BURIED_DESC)
    _skill(tree, "bbb_front", FRONT_DESC)
    _skill(tree, "ccc_short", SHORT_DESC)

    rows = _fired(sl.lint())
    assert set(rows) == {"aaa_buried"}, rows
    row = rows["aaa_buried"]
    assert row["chars"] == 217
    assert row["window"] == BURIED_DESC[:100]
    assert row["first_marker_at"] == 172 and row["first_marker"] == "use"
    # Arm (a) held for all three: membership is the renderer's, not a guess.
    membership = sl.rendered_index_descriptions(_active())
    assert all(membership[n] for n in ("aaa_buried", "bbb_front", "ccc_short"))


def test_a_names_only_index_line_does_not_fire(tree, monkeypatch):
    """Arm (a): over the cap is not the same as rendered.

    The names-only skill carries `BURIED2_DESC` — over the clip, no marker inside
    its first 100 characters — so it satisfies arms (b) and (c) exactly as the
    firing skill does, and arm (a) is the only thing that can hold it back. Give it
    a description that already leads with its trigger and this node would pass even
    if arm (a) did not exist: that is the trap the first draft fell into.

    The budget is found by probing the renderer for the one value that grants a
    description to the ranked skill and to nobody else, so the split is the
    renderer's own decision and not a number this file asserts about it. That gap
    is the 191-versus-68 split #4374's triage measured.
    """
    _skill(tree, "aaa_buried", BURIED_DESC)
    _skill(tree, "zzz_names_only", BURIED2_DESC)
    monkeypatch.setattr(pb, "_skill_rank_counts", lambda: {"aaa_buried": 5})
    CONFIG["skills"]["index"] = dict(
        CONFIG["skills"]["index"],
        budget_chars=_budget_describing_only(_active(), {"aaa_buried": 5},
                                             "aaa_buried", 100))

    # Arms (b) and (c) hold for the skill that must not fire; only (a) differs.
    assert len(BURIED2_DESC) > 100
    assert sl.trigger_marker_in_window(BURIED2_DESC, 100) is False

    membership = sl.rendered_index_descriptions(_active())
    assert membership["aaa_buried"] and membership["zzz_names_only"] == ""

    rows = _fired(sl.lint())
    assert set(rows) == {"aaa_buried"}, rows
    # The old arm still sees both: the new count is a subset, not a replacement.
    assert {d["name"] for d in sl.lint()["description"] if d["reason"] == "clipped"} == {
        "aaa_buried", "zzz_names_only"}


def test_all_three_arms_are_conjunctive_in_one_run(tree, monkeypatch):
    """One firing row in a four-skill library that fails one arm each.

    `ccc_short` is under the clip, `bbb_front` leads with its trigger, and
    `zzz_names_only` fails only the rendered-membership arm; `aaa_buried` is the
    single skill satisfying all three, so a conjunction that had weakened to a
    disjunction — or an arm (a) testing dict membership instead of the value, since
    every scanned skill is a key — would return more than this one name. The two
    denominators are asserted too: a count of 1 over a denominator of 4 is a
    different instrument from a count of 1 over 1.
    """
    _skill(tree, "aaa_buried", BURIED_DESC)
    _skill(tree, "bbb_front", FRONT_DESC)
    _skill(tree, "ccc_short", SHORT_DESC)
    _skill(tree, "zzz_names_only", BURIED2_DESC)
    monkeypatch.setattr(pb, "_skill_rank_counts", lambda: {"aaa_buried": 5})
    CONFIG["skills"]["index"] = dict(
        CONFIG["skills"]["index"],
        budget_chars=_budget_describing_only(_active(), {"aaa_buried": 5},
                                             "aaa_buried", 100))

    result = sl.lint()
    assert set(_fired(result)) == {"aaa_buried"}
    ctx = result["buried_trigger_context"]
    assert ctx["rendered_clipped"] == 1, "the denominator is the population the model half-sees"
    assert ctx["index_described"] == 1 and ctx["clip"] == 100
    assert ctx["descriptions_on"] is True
    assert ctx["clause_free"] == 0, "BURIED_DESC has a marker; it is at 172, not absent"


# ── clause 2: the instrument can fail, and can be satisfied ─────────────────

def test_moving_the_trigger_inside_the_window_moves_the_count_from_one_to_zero(tree):
    """The #4373 shape: a length-neutral reorder is the only thing that changes.

    No count anywhere else in this report moves for this edit — that is the
    defect #4374 exists to close — so this pair is what makes the category
    non-vacuous rather than a permanent 20.
    """
    _skill(tree, "one_skill", BURIED_DESC)
    assert set(_fired(sl.lint())) == {"one_skill"}

    _skill(tree, "one_skill", FRONT_DESC)
    assert _fired(sl.lint()) == {}


# ── clause 4: the cap is the renderer's, never a local copy ─────────────────

def test_retreating_the_clip_changes_which_skills_fire(tree):
    """`MID_MARKER_DESC`'s only marker sits at char 81.

    At the shipped 100-character clip the window covers it, so nothing fires.
    Retreating the clip to 40 — through `skills.index.max_description_chars`,
    the same key `_load_skills_index` reads — puts the marker outside the window
    and the row appears. A lint holding its own copy of `100` would keep printing
    0 here, and the nightly and the prompt would disagree about who is cut, which
    is the drift `skill_index_description()` exists to prevent on the other seam.
    """
    _skill(tree, "mid_marker", MID_MARKER_DESC)
    assert _fired(sl.lint()) == {}

    CONFIG["skills"]["index"] = dict(CONFIG["skills"]["index"],
                                     max_description_chars=40)
    result = sl.lint()
    rows = _fired(result)
    assert set(rows) == {"mid_marker"}, rows
    assert result["description_clip"] == 40
    assert rows["mid_marker"]["first_marker_at"] == 81
    assert rows["mid_marker"]["window"] == MID_MARKER_DESC[:40]


def test_a_names_only_index_makes_the_count_zero_and_says_why(tree):
    """`descriptions: false` renders no description at all, so the 0 is not health.

    `_load_skills_index` does not call `_skills_index_lines` on that path, so
    membership is empty by construction; the context block is what tells a reader
    that, instead of a clean-looking zero.
    """
    _skill(tree, "aaa_buried", BURIED_DESC)
    CONFIG["skills"]["index"] = dict(CONFIG["skills"]["index"], descriptions=False)

    assert sl.rendered_index_descriptions(_active()) == {}
    result = sl.lint()
    assert _fired(result) == {}
    assert result["buried_trigger_context"]["descriptions_on"] is False


# ── clause 5 + report surface ────────────────────────────────────────────────

def test_the_category_is_registered_in_category_trust_with_an_honest_cause():
    """The table looks its cause up by name (`skill_lint.py`'s `trust_cell`), so a
    row without an entry raises KeyError on the first render. The cause must say
    which arms are measured and which is a scan, the way DRIFT's and STALE's do.
    """
    verdict, cause = sl.CATEGORY_TRUST[sl.BURIED_TRIGGER_CATEGORY]
    assert verdict in sl.TRUST_VERDICTS and verdict != "no"
    assert verdict == "mostly"
    lowered = cause.lower()
    assert "measured" in lowered and "advisory" in lowered
    # The exact reading of a 0, said the way DRIFT says its own.
    assert "never" in lowered and "marker" in lowered
    assert sl.trust_cell(sl.BURIED_TRIGGER_CATEGORY).startswith("⚠️ mostly")
    assert sl.BURIED_TRIGGER_CATEGORY not in sl.untrustworthy_categories()


def test_the_rows_print_in_the_existing_md_and_json_report_surfaces(tree):
    """No new report file: the row, the section and the payload key are the
    surfaces job 70 already publishes (`skill-lint-report.md`/`.json`)."""
    _skill(tree, "aaa_buried", BURIED_DESC)
    result = sl.lint()
    report = sl.render_report(result)
    assert f"| {sl.BURIED_TRIGGER_CATEGORY} (" in report
    assert f"## {sl.BURIED_TRIGGER_CATEGORY}" in report
    assert "`aaa_buried`" in report
    assert sl.BURIED_TRIGGER_CATEGORY in sl.CATEGORY_TRUST
    # The JSON sibling is `json.dumps(lint())`, so the payload key is its surface.
    assert json.loads(json.dumps(result, default=str))["buried_trigger"]

    # A result that never measured it — the shape every older fixture and stored
    # payload has — must still render, with an unqualified zero as the exception.
    # A payload that carries neither key — one stored before this category existed,
    # or a hand-built one like every older fixture in this suite — keeps its table row
    # (the table is unconditional, and a category row disappearing is how a check
    # goes unnoticed) and gets no section: the renderer will not describe a
    # population it was never handed.
    bare = {k: v for k, v in result.items()
            if k not in ("buried_trigger", "buried_trigger_context")}
    report = sl.render_report(bare)
    assert f"| {sl.BURIED_TRIGGER_CATEGORY} (" in report
    assert f"## {sl.BURIED_TRIGGER_CATEGORY}" not in report


def test_a_zero_after_a_measured_run_is_explained_and_not_read_as_clean(tree):
    """A check that fires on nothing prints why it could have, the way the
    module's own per-rule INJECTION table does: a rule that ran and found nothing
    and a rule that never ran must not read the same."""
    _skill(tree, "bbb_front", FRONT_DESC)
    result = sl.lint()
    assert _fired(result) == {}
    report = sl.render_report(result)
    section = report[report.index(f"## {sl.BURIED_TRIGGER_CATEGORY}"):]
    # The zero prints its own denominator — one rendered-clipped skill out of one
    # described — so a reader can tell "the window covered every marker" from "the
    # sweep had nothing to measure". Quoted from the section's own first line.
    assert "Measured over 1 of the 1 skills" in section
    assert "no rendered-clipped" in section.lower()


# ── the live library: the lint and the renderer must agree ───────────────────

def test_the_live_library_count_equals_the_renderers_own_derivation():
    """Independent recomputation of the same population, from the renderer.

    This is #4373's acceptance command as an assertion: membership is taken from
    `_skills_index_lines` with the same rank counts handed to `lint`, then the
    three arms are applied by hand here, in the open, with no call into the
    module's own loop. If the two ever disagree, the nightly and the prompt
    disagree about who is cut — the failure #4374 was opened for.
    """
    active = list(skills_mod.iter_active_skills(roots=pb._skill_walk_roots()))
    counts = pb._skill_rank_counts()
    settings = pb._skills_index_settings()
    clip = settings["max_description_chars"]
    lines = pb._skills_index_lines(active, counts, budget=settings["budget_chars"],
                                   max_desc=clip)
    shown: dict[str, str] = {}
    for line in lines[1:]:
        name, sep, desc = line[2:].partition(" — ")
        shown[name] = desc if sep else ""
    folded = {rec.name: pb.skill_index_description(rec.frontmatter) for rec in active}
    rendered_clipped = {n for n, d in shown.items() if d and len(folded[n]) > clip}
    expect = {n for n in rendered_clipped
              if not set(re.findall(r"\b\w+\b", folded[n][:clip].lower()))
              & set(sl.TRIGGER_WORDS)}

    result = sl.lint(index_counts=counts)
    fired = set(_fired(result))
    assert fired == expect, sorted(expect ^ fired)
    assert fired <= rendered_clipped, sorted(fired - rendered_clipped)
    assert result["buried_trigger_context"]["rendered_clipped"] == len(rendered_clipped)
