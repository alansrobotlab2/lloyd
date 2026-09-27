"""knowledge-health-report.py — the `## Suggested Research Questions` section (#954).

Every thin entity has **exactly 1** active fact by definition (`active_facts <
THIN_ENTITY_MAX_FACTS`), so `thin_entities.sort(key=lambda x: x[1]["active_facts"])`
was a total tie across all 4,246 of them. Python's sort is stable and
`load_entities` fills its dict from `sorted(facts_dir.iterdir())`, so the order that
survived was alphabetical by entity directory name, and the section rendered the first
20 of it. On the live store that made the section a frozen ASCII-first slice: in
`_pipeline/reflection/knowledge-health-2026-09-19.md` the 20 questions equal
`sorted(table_names)[:20]` exactly, and 20 of 20 name an extraction artifact (`#160`,
`#165`, `#222` … — each of which resolves to an existing
`~/obsidian/backlog/<id>-*.md`). Zero of the 20 was a researchable gap, and the
section was byte-identical on 09-13, 09-14, 09-15 and 09-16.

Each node below pins one clause of #954: recency breaks the tie (1); no
artifact-shaped name reaches the section (2); too few survivors is a printed finding
rather than twenty junk questions (3); and the `## Thin Entities` table keeps listing
artifact-shaped entities, because the filter belongs to the questions and not to the
hygiene count (4). The four nodes at the bottom pin the four clauses of #1542, which
widens the same filter to the two shapes #954 never covered: an underscore-bearing
dotted path and a bare number. The artifact shapes are matched here with regexes
written independently of the module's own, so the test does not grade the
implementation against itself.
"""
import importlib.util
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("khr_rq", ROOT / "scripts/memory/knowledge-health-report.py")
khr = importlib.util.module_from_spec(_spec); sys.modules["khr_rq"] = khr; _spec.loader.exec_module(khr)

# A fixed clock so every `latest_created` below is an exact number of days ago.
NOW = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)
THRESH = khr.STALE_DAYS_THRESHOLD

# The four name shapes the acceptance names, written independently of the module.
ARTIFACT_RE = re.compile(r"^#\d|^--|\.md$|^\d{4}-\d{2}-\d{2}")

# The entity names the extractor keeps minting: all five appeared in the live
# report's top 20, and every `#NNN` among them is a backlog id.
ARTIFACT_NAMES = ["#441", "#160", "--continue",
                  "03-linear-representation-hypothesis.md", "2026-09-11"]

# The two shapes #1542 adds, also written independently of the module: a dotted
# path with an underscore on either side of a dot (a pytest node id `a_b.c_d`, a
# dotted config key `inner_voice.todo_stewardship.enabled`) and a bare number.
# Whitespace stops the dotted leg, because a code path is one token: a name with
# a space in it is prose, and is left alone.
CODE_SHAPE_RE = re.compile(r"[^.\s]*_[^.\s]*\.|\.[^.\s]*_|^\d+$")

# Six rows of the live 2026-09-26 `## Thin Entities` table that are code, not
# concepts: the first two are the pair that reached the briefing that morning (as
# `- Is **test_iv_loop_guards.pytest_observer_resolves_to_the_primary_endpoint**
# still relevant?` and the same for `inner_voice.todo_stewardship.enabled`), and
# `255` is the table's row 40. Of the table's 50 listed rows the widened rule
# rejects 7 of them, which is what these names are here to pin.
CODE_SHAPED_NAMES = [
    "test_iv_loop_guards.pytest_observer_resolves_to_the_primary_endpoint",
    "inner_voice.todo_stewardship.enabled",
    "inner_voice.todo_stewardship",
    "observer_prompt._tool_call_labels",
    "guards._strip_cd_prefix",
    "255",
]

# The dotted name the widened rule must spare: a nix flake, so a dot is legitimate
# and neither segment carries an underscore. Row 27 of the same live table. That
# single fixture pair — this name asked about twice, `guards._strip_cd_prefix`
# never — is the whole discriminator.
DOTTED_NOT_CODE = "jail.nix"


def _write(root: Path, name: str, days_ago_list: list[int]) -> None:
    """One entity dir holding one overview file, one active fact per day-count.

    The fact's text is the entity name, so a fixture can be read as a table:
    `{"Zebra": [1], "Alpha": [12]}` means Zebra's newest fact is 1 day old.
    """
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    facts = [{"fact": f"note about {name} #{i}", "entity": name, "confidence": 0.9,
              "category": "overview", "id": f"fact-{i:03d}",
              "created_at": (NOW - timedelta(days=days)).isoformat()}
             for i, days in enumerate(days_ago_list)]
    fm = {"type": "facts", "entity": name, "category": "overview", "facts": facts}
    (d / f"{name}-overview.md").write_text(
        f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {name} - overview\n", encoding="utf-8")


def _load(root: Path, spec: dict[str, list[int]]) -> dict:
    for name, days in spec.items():
        _write(root, name, days)
    return khr.load_entities(root)


def _report(entities: dict) -> str:
    return khr.generate_report(
        khr.compute_entity_stats(entities),
        khr.compute_relationship_stats([], entities),
        [],
        khr.find_stale_facts(entities, NOW, THRESH),
        NOW,
        stale_unevaluable=khr.stale_coverage(entities),
    )


def _section(report: str, heading: str) -> str:
    start = report.index(heading)
    rest = report[start + len(heading):]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


def _questions(report: str) -> list[str]:
    """The entity names the section asks about, in the order it asks about them.

    Parsed with the same pattern `skills/research-queue-generator/SKILL.md` reads,
    so this is the question set as its one consumer sees it.
    """
    return re.findall(r"^- What does \*\*(.+?)\*\* relate to\?$",
                      _section(report, "## Suggested Research Questions"), re.M)


def _table_names(report: str) -> list[str]:
    return re.findall(r"^\| (.+?) \| \d+ \|$", _section(report, "## Thin Entities"), re.M)


# ── clause 1: ties break on `latest_created`, not on the entity's first byte ──

# Twelve thin entities, newest first. Their alphabetical order is exactly this
# list reversed, so a section rendered in this order cannot have come from a name
# sort. Twelve is above MIN_USABLE_RESEARCH_QUESTIONS: below it the section
# correctly prints the unevaluable verdict instead of questions (clause 3), which
# is what a 3-name fixture would have pinned here instead.
RECENCY_SPEC = [
    ("Zebra-Embedding-Cache", 1), ("Midas-Query-Planner", 2),
    ("Hotel-Sparse-Index", 3), ("India-KV-Cache", 4),
    ("Golf-Chunker", 5), ("Foxtrot-Tokenizer", 6),
    ("Juliett-Router", 7), ("Echo-Reranker", 8),
    ("Delta-Cache-Layer", 9), ("Cuda-Graph-Capture", 10),
    ("Bloom-Filter-Registry", 11), ("Alpha-Vector-Store", 12),
]


def test_thin_entities_rank_newest_first_when_recency_and_alphabet_differ(tmp_path):
    """Every tie is broken by `latest_created`, and the result is not a name order."""
    root = tmp_path / "facts"
    entities = _load(root, {name: [days] for name, days in RECENCY_SPEC})
    stats = khr.compute_entity_stats(entities)

    # The premise: every one of these is thin on the same key, so only the
    # tie-break decides the order — and the tie-break field is the newest fact date.
    assert {s["active_facts"] for s in stats.values()} == {1}
    assert (stats["Zebra-Embedding-Cache"]["latest_created"]
            > stats["Midas-Query-Planner"]["latest_created"]
            > stats["Alpha-Vector-Store"]["latest_created"])

    questions = _questions(_report(entities))

    assert questions == [name for name, _ in RECENCY_SPEC]
    # The defect's own signature: the rendered order equalled the alphabetically
    # first rows of the table, i.e. `sorted(names)`. The spec above is not that
    # order — Zebra leads and Hotel precedes India — so this equality failing is
    # the acceptance check itself flipping.
    assert questions != sorted(questions)


def test_latest_created_is_the_newest_fact_date_not_the_first_one(tmp_path):
    """`latest_created` is a max: an older fact listed first must not win."""
    root = tmp_path / "facts"
    entities = _load(root, {"Kestrel-Retrieval": [40, 3]})  # oldest listed first

    stats = khr.compute_entity_stats(entities)

    assert stats["Kestrel-Retrieval"]["active_facts"] == 2  # thin or not, the field stands
    assert stats["Kestrel-Retrieval"]["latest_created"] == NOW - timedelta(days=3)


# ── clause 2: extraction-artifact names never become a question ───────────────

def test_artifact_shaped_names_never_reach_the_questions(tmp_path):
    """11 usable names + 5 artifacts: the artifacts are dropped, the rest ranked."""
    root = tmp_path / "facts"
    usable = {f"Gap-Topic-{i:02d}": [(i * 3) % 11] for i in range(11)}
    # `latest_created` order is the day-count order, and it deliberately does not
    # agree with the name order, so this cannot pass on a name sort either.
    expected = sorted(usable, key=lambda n: usable[n][0])
    assert expected != sorted(usable)

    entities = _load(root, {**usable, **{n: [100 + i] for i, n in enumerate(ARTIFACT_NAMES)}})
    report = _report(entities)
    questions = _questions(report)

    assert questions == expected
    assert [q for q in questions if ARTIFACT_RE.search(q)] == []


# ── clause 3: too little surviving signal is a printed finding, not junk ──────

def test_unevaluable_when_fewer_than_ten_names_survive(tmp_path):
    """7 of 12 thin entities are artifacts, leaving 5 — below the floor of 10."""
    root = tmp_path / "facts"
    entities = _load(root, {
        **{n: [1] for n in ARTIFACT_NAMES},                       # 5
        **{f"#9{i:02d}": [2] for i in range(2)},                   # 2 more, total 7
        **{f"Live-Topic-{i}": [3] for i in range(5)},              # 5 usable
    })
    section = _section(_report(entities), "## Suggested Research Questions")

    assert ("RESEARCH_QUESTIONS_UNEVALUABLE: 7 of 12 thin entities carry an "
            "artifact-shaped name") in section
    assert "- What does" not in section
    assert "- Is **" not in section


def test_ten_survivors_is_enough_and_renders_no_verdict_line(tmp_path):
    """The boundary: exactly 10 usable names is a ranking, not a verdict."""
    root = tmp_path / "facts"
    entities = _load(root, {
        **{n: [50 + i] for i, n in enumerate(ARTIFACT_NAMES)},
        **{f"Real-Topic-{i}": [i] for i in range(10)},
    })
    report = _report(entities)
    section = _section(report, "## Suggested Research Questions")

    assert "RESEARCH_QUESTIONS_UNEVALUABLE" not in section
    assert len(_questions(report)) == 10


# ── clause 4: the filter covers the questions, not the hygiene table ──────────

def test_thin_entities_table_still_lists_the_artifact_names(tmp_path):
    """The same 16 thin entities: dropped from the questions, still in the table.

    The table is what makes a regrowth in `#NNN` minting visible; filtering the
    questions must not also hide the population it was drawn from.
    """
    root = tmp_path / "facts"
    usable = {f"Gap-Topic-{i:02d}": [i] for i in range(11)}
    entities = _load(root, {**usable, **{n: [100 + i] for i, n in enumerate(ARTIFACT_NAMES)}})
    report = _report(entities)
    table = _table_names(report)

    assert set(ARTIFACT_NAMES) <= set(table)
    assert len(table) <= khr.SECTION_ROW_CAP
    # The count is the whole population, not the filtered one.
    assert "**16** in total" in _section(report, "## Thin Entities")
    # And it is the questions that lost them, not the table: no artifact is a
    # question here either.
    assert [q for q in _questions(report) if ARTIFACT_RE.search(q)] == []


# ── #1542 clause 1: a code-shaped name is not a research subject ──────────────

def test_code_shaped_names_never_reach_the_questions(tmp_path):
    """12 usable names + 6 code-shaped: 12 questions, and not one of them is code.

    12 usable sits above MIN_USABLE_RESEARCH_QUESTIONS (10), so the section is
    rendering an ordinary ranking here — the rejection has to hold while the
    section works, not only when it gives up (that case is the last node). Each
    name would otherwise have produced two lines, `- What does **…** relate to?`
    and `- Is **…** still relevant?`, which is what the briefing reads.
    """
    root = tmp_path / "facts"
    usable = {f"Gap-Shape-{i:02d}": [i] for i in range(12)}
    entities = _load(root, {**usable,
                            **{n: [100 + i] for i, n in enumerate(CODE_SHAPED_NAMES)}})
    report = _report(entities)
    section = _section(report, "## Suggested Research Questions")
    questions = _questions(report)

    assert len(questions) == 12
    assert set(questions) == set(usable)
    assert [q for q in questions if CODE_SHAPE_RE.search(q)] == []
    # Not merely absent from the parsed questions: absent from the section, so
    # neither of its two line forms is rendered for a code-shaped name.
    for name in CODE_SHAPED_NAMES:
        assert f"**{name}**" not in section
    # And the section rendered questions rather than surrendering: 12 survivors
    # is above the floor, so no verdict line is due.
    assert "RESEARCH_QUESTIONS_UNEVALUABLE" not in section


# ── #1542 clause 2: the underscore is the discriminator, not the dot ──────────

def test_a_dotted_name_without_an_underscore_keeps_both_questions(tmp_path):
    """`jail.nix` is a nix flake and a real tool: it keeps both of its questions.

    The widened rule rejects dotted code-shaped names, not every name containing
    a dot. The fixture holds `jail.nix` next to `guards._strip_cd_prefix`, so the
    only difference between the name asked about twice and the name never asked
    about is the underscore: a rule written as "the name contains a dot" fails
    this node, and the underscore is what makes a dotted name a code path.
    """
    root = tmp_path / "facts"
    usable = {f"Gap-Shape-{i:02d}": [i] for i in range(11)}
    entities = _load(root, {**usable, DOTTED_NOT_CODE: [4],
                            **{n: [100 + i] for i, n in enumerate(CODE_SHAPED_NAMES)}})
    report = _report(entities)
    section = _section(report, "## Suggested Research Questions")

    assert [ln for ln in section.splitlines() if f"**{DOTTED_NOT_CODE}**" in ln] == [
        "- What does **jail.nix** relate to?",
        "- Is **jail.nix** still relevant?",
    ]
    # Same section, opposite verdict: the underscore-bearing dotted name next to
    # it is gone, and 12 usable names (11 + jail.nix) still clear the floor.
    assert "guards._strip_cd_prefix" not in section
    assert "RESEARCH_QUESTIONS_UNEVALUABLE" not in section


# ── #1542 clause 3: the widened filter belongs to the questions only ──────────

def test_thin_entities_table_still_lists_the_code_shaped_names(tmp_path):
    """All 18 thin entities stay in the table and keep the stated total at 18.

    The same 12 usable + 6 code-shaped fixture as clause 1: the questions drop to
    12, but the table lists every one of the 6 rejections and its total is the
    unfiltered thin-entity population — 18, not the 12 the section was allowed to
    ask about. This is the signal #743/#1535 read: hide these rows and a regrowth
    in junk minting stops being visible.
    """
    root = tmp_path / "facts"
    usable = {f"Gap-Shape-{i:02d}": [i] for i in range(12)}
    entities = _load(root, {**usable,
                            **{n: [100 + i] for i, n in enumerate(CODE_SHAPED_NAMES)}})
    report = _report(entities)
    table = _table_names(report)

    assert set(CODE_SHAPED_NAMES) <= set(table)
    assert "**18** in total" in _section(report, "## Thin Entities")
    # The table kept them; only the questions lost them.
    assert [q for q in _questions(report) if CODE_SHAPE_RE.search(q)] == []


# ── #1542 clause 4: the widened rejections drive the existing verdict ─────────

def test_code_shaped_rejections_alone_can_drive_the_unevaluable_verdict(tmp_path):
    """4 usable of 14: the widened shapes alone take the section under the floor.

    None of the 10 rejected names here matches any of the four #954 shapes — no
    `#`, no leading `--`, no `.md`, no leading date — so the count of 10 and the
    4 survivors come entirely from the two #1542 shapes. The section prints the
    guard that already existed, unchanged, instead of padding with the four
    names that survived: same marker, same wording, one code path.
    """
    root = tmp_path / "facts"
    entities = _load(root, {
        **{f"Live-Shape-{i}": [1] for i in range(4)},                 # 4 usable
        **{n: [2] for n in CODE_SHAPED_NAMES},                        # 6
        **{f"test_iv_guard_{i}.pytest_case_{i}": [3] for i in range(2)},  # 2 node ids
        **{"1204": [4], "77": [4]},                                   # 2 bare numbers
    })
    section = _section(_report(entities), "## Suggested Research Questions")

    assert len(CODE_SHAPED_NAMES) == 6
    assert ("RESEARCH_QUESTIONS_UNEVALUABLE: 10 of 14 thin entities carry an "
            "artifact-shaped name") in section
    assert "- What does" not in section
    assert "- Is **" not in section
