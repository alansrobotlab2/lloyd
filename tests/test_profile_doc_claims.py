"""Claims `eval/lloyd_profile.md` makes to the sessions that *propose* work.

This file is not documentation. `workers/sources/youtube_digest.py`'s
`PROFILE_PATH` puts it in front of the digest session that decides whether a
video holds an idea worth filing, and `skills/ai-engineer-monitor` names the
same path, so every sentence in it is a premise a backlog item inherits. Backlog
#599 was filed against a KG graph arm in the pre-turn prefetch that no code has
ever had, and #671 restated the identical sentence four days later; the scale
counts in the neighbouring bullet were an order of magnitude out and nothing
refreshed them (#1076). Hence the three rules pinned here: attribute graph
expansion to the path that actually has it, put no un-refreshed count in front
of the idea generator, and date a metric snapshot to the baseline file it came
from — a file that exists where the prose says it does — and to the ceiling
latency is graded against.

The code-side assertions are the point, not decoration: each one re-reads the
module the sentence names, so the sentence cannot outlive the behaviour by
silence in either direction.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

import prefetch as PF
from agent_mcp.vault import _vault_recall_base as _vault_recall
from app import data_root as DR
from app import paths as P
from app.kg_store import configure, store
from workers.sources import automod_regression as ARE

ROOT = Path(inspect.getsourcefile(PF)).resolve().parent
PROFILE = ROOT / "eval" / "lloyd_profile.md"
AI_ENGINEER_SKILL = Path.home() / "obsidian" / "skills" / "ai-engineer-monitor" / "SKILL.md"

# The shapes of magnitude this file used to carry: an approximation with a
# unit (`~23.6k`, `~12k`), a bare k-count (`4k active`), or a count written
# beside the noun it counts (`305002 facts`, `20 queries`).
_COUNT_PATTERNS = (
    re.compile(r"~\s*\d[\d.]*\s*[kKmM]?\b"),
    re.compile(r"\b\d[\d.]*[kKmM]\b"),
    re.compile(r"\b\d[\d,]*\s+(?:entities|edges|aliases|facts|documents|queries)\b",
               re.IGNORECASE),
)

# The keys `store().stats()` answers with, as the Knowledge-graph bullet names
# them. A key renamed in code and left in the prose sends a reader to a call
# that does not answer.
KG_STAT_KEYS = ("entities", "aliases", "edges_total", "edges_active", "facts")


def _bullet(starts_with: str) -> str:
    """The markdown list item whose text begins `starts_with`, with its
    continuation lines."""
    lines = PROFILE.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(f"- {starts_with}"):
            j = i + 1
            while j < len(lines) and lines[j].startswith("  ") and lines[j].strip():
                j += 1
            return "\n".join(lines[i:j])
    raise AssertionError(f"no bullet starting '- {starts_with}' in {PROFILE}")


def _stale_counts(text: str) -> list[str]:
    found: list[str] = []
    for pattern in _COUNT_PATTERNS:
        found.extend(m.group(0).strip() for m in pattern.finditer(text))
    return found


# ── clause 1: graph expansion belongs to vault_recall, not to prefetch ──────

def test_prefetch_is_not_credited_with_graph_expansion():
    text = PROFILE.read_text()
    assert "plus KG seed-and-expand" not in text, (
        "the sentence that made #599 and #671 file a non-existent graph arm is "
        "back: it tells the idea generator the pre-turn prefetch expands the KG"
    )
    bullet = _bullet("Retrieval runs on two paths")
    assert "no graph arm" in bullet, "the retrieval paragraph no longer states the negative"
    assert "seed-and-expand" in bullet, "graph seed-and-expand is attributed to nothing"
    assert "vault_recall" in bullet, "seed-and-expand is not attributed to vault_recall"


def test_the_graph_arm_sits_where_the_profile_puts_it():
    """The code half of the same claim, so the prose cannot drift either way.

    `prefetch.py` is the path that runs before every turn; if a graph leg is
    ever added to it this test goes red on purpose — the paragraph the idea
    generator reads would then be wrong again, in the opposite direction.
    """
    source = inspect.getsource(PF)
    assert "kg_store" not in source, (
        "prefetch.py now imports the knowledge-graph module; the profile's "
        "never-imports-`app.kg_store` clause and this test both have to be "
        "re-read before either is trusted"
    )
    assert "app.kg_store" not in source
    assert "neighbor" not in inspect.getsource(PF._search_facts).lower(), (
        "the fact leg has grown a neighbour walk, which is exactly what the "
        "profile says it does not do"
    )
    submitted = re.findall(r"\(\"([a-z_]+)\",", inspect.getsource(PF._prefetch_run))
    assert submitted, "no named legs to read out of _prefetch_run"
    assert not [n for n in submitted if "graph" in n or n.startswith("kg")], submitted
    # The call site, not the file: vault.py imports the arm from
    # `agent_mcp.retrieval`, so a name merely present in the module proves
    # nothing about who walks the graph when the model asks for recall.
    assert "graph_weighted_neighbors" in inspect.getsource(_vault_recall), (
        "the seed-and-expand the profile credits to vault_recall is no longer "
        "called inside `_vault_recall` — re-read the retrieval path before "
        "trusting either the prose or this assertion"
    )


def test_every_prefetch_leg_the_profile_names_is_a_leg_the_code_submits():
    bullet = _bullet("Retrieval runs on two paths")
    src = inspect.getsource(PF._prefetch_run)
    for leg in ("skills", "facts", "sessions", "backlog"):
        assert leg in bullet, f"profile dropped the {leg} leg"
        assert f'("{leg}",' in src, f"prefetch no longer submits a {leg} leg"


# ── clause 2: no un-refreshed magnitude in front of the idea generator ──────

def test_knowledge_graph_bullet_prints_no_count_that_nothing_refreshes():
    bullet = _bullet("Knowledge graph:")
    assert not _stale_counts(bullet), (
        f"hard-coded KG counts are back ({_stale_counts(bullet)}); the bullet "
        "must name store().stats() instead of carrying a number the nightly "
        "extractor will outrun"
    )


def test_knowledge_graph_bullet_names_the_sanctioned_source():
    bullet = _bullet("Knowledge graph:")
    assert "app.kg_store" in bullet
    assert "store().stats()" in bullet, "the pointer that replaced the counts is gone"


def test_the_kg_scale_pointer_resolves_through_the_module(tmp_path):
    """`store().stats()` must answer with the keys the bullet sends a reader to.

    Reaches the store only as `app.kg_store` allows — `configure` + `store()` on
    a throwaway path, never `sqlite3` on the live `kg.sqlite`.
    """
    configure(tmp_path / "kg.sqlite")
    try:
        stats = store().stats()
    finally:
        from app import kg_store
        kg_store.reset()
    for key in KG_STAT_KEYS:
        assert f"`{key}`" in _bullet("Knowledge graph:"), f"bullet no longer names {key}"
        assert key in stats, f"stats() no longer returns {key}"
    assert stats["edges_active"] <= stats["edges_total"]


def test_retrieval_bullet_prints_no_count_that_nothing_refreshes():
    assert not _stale_counts(_bullet("Retrieval runs on two paths"))


# ── clause 3: the metric snapshot is dated, sourced, and priced ─────────────

def test_metric_snapshot_is_dated_to_the_nightly_baseline_it_came_from():
    bullet = _bullet("Nightly retrieval eval")
    named = re.search(r"eval/baselines/nightly-(\d{8})-\d{8}-\d{6}\.json", bullet)
    assert named, "the snapshot names no eval/baselines/nightly-*.json run"
    prose_dates = ["".join(g) for g in re.findall(r"(20\d\d)-(\d\d)-(\d\d)", bullet)]
    assert named.group(1) in prose_dates, (
        "the prose is dated to a different night than the baseline file it names"
    )


def test_latency_ceiling_is_the_named_budget_and_still_the_live_one():
    """The ceiling quoted here is read out of the module that prices it.

    The old sentence said `~1.6 s per query`, which no rung owned and which the
    nightly had already left 2.7x behind; #1129 named the constant that grades
    latency, so the profile cites the constant and this test checks the number
    against it rather than against a remembered figure.
    """
    text = PROFILE.read_text()
    bullet = _bullet("Nightly retrieval eval")
    assert "~1.6" not in text, "the un-owned latency claim is back"
    assert "LATENCY_BUDGET_MS" in bullet and "CONTEXT_NIGHTLY" in bullet
    assert "workers/sources/automod_regression.py" in bullet
    ceiling = ARE.latency_budget(ARE.CONTEXT_NIGHTLY)
    assert ceiling > 0
    assert f"{int(ceiling)} ms" in bullet, (
        f"profile does not quote the live nightly ceiling of {int(ceiling)} ms; "
        "re-read LATENCY_BUDGET_MS[CONTEXT_NIGHTLY] rather than editing this test"
    )


def _baseline_dirs() -> list[Path]:
    """Every directory a nightly baseline can legitimately be read from here.

    `app.paths.EVAL_BASELINES_DIR` is *this process's* root, which the gate and
    the suite repoint at a scratch `LLOYD_DATA`; the run a published snapshot
    quotes was written by the production nightly, so the production root is the
    other candidate. Both come out of `app.data_root` — never a home path
    spelt out here, which is how the profile's own pointer went stale.
    """
    return list(dict.fromkeys([
        P.EVAL_BASELINES_DIR,
        DR.production_data_root() / "eval" / "baselines",
    ]))


def test_metric_snapshot_numbers_match_the_file_they_name():
    """The bullet's figures are re-read from the baseline JSON it cites.

    This is the seam the whole item sits on: `eval/run_eval.py` writes the
    record, the profile restates it by hand, and across that boundary the
    numbers went a month without anyone checking — a "~1.6 s per query" that
    the nightly had left 2.7x behind. Dating the snapshot without re-reading
    the file it names would pin the citation and still let the claim drift.

    Skips only when the named file is genuinely absent: baselines are
    gitignored mutable data, and the 2026-09-22 corpus wipe emptied the
    directory once (#1377). The naming and date checks run either way.
    """
    bullet = _bullet("Nightly retrieval eval")
    named = re.search(r"eval/baselines/(nightly-\d{8}-\d{8}-\d{6}\.json)", bullet)
    assert named, "the snapshot names no nightly baseline file"
    name = named.group(1)
    path = next((d / name for d in _baseline_dirs() if (d / name).is_file()), None)
    if path is None:
        pytest.skip(f"{name} is not on disk under any of {_baseline_dirs()}")
    overall = json.loads(path.read_text())["summary"]["overall"]
    for key in ("entity_hit_rate", "doc_hit_rate", "entity_recall_avg",
                "doc_recall_avg", "mrr_doc", "ndcg10", "fact_entity_recall_avg"):
        assert key in overall, (
            f"{name} has no {key}: the bullet quotes a metric the run that "
            "produced it did not emit"
        )
        assert str(overall[key]) in bullet, (
            f"bullet does not carry {key}={overall[key]} from {name}; re-read "
            "the baseline and fix the prose, not this assertion"
        )
    assert re.search(rf"latency_ms_avg\s+{int(round(overall['latency_ms_avg']))}\s*ms",
                     bullet), (
        f"bullet's latency is not {round(overall['latency_ms_avg'])} ms, the "
        f"value {name} records under summary.overall"
    )


def test_the_baselines_pointer_names_a_directory_that_resolves():
    """The nightly baselines have not lived under `~/lloyd` since the
    data-home cutover: `app.paths` puts every mutable artifact under
    `DATA_ROOT`, so the directory is `EVAL_BASELINES_DIR` — `~/lloyd-data/eval/
    baselines` in production. The Live-measurements bullet said "under
    `~/lloyd`", which is the same defect as a stale count one level up: a
    pointer that resolves nowhere is read as "the nightly never ran", and the
    session that follows it stops looking.
    """
    section = PROFILE.read_text().split("## Live measurements", 1)[1]
    pointers = [b for b in section.split("\n- ") if b.startswith("**Retrieval**")]
    assert pointers, "the Live-measurements retrieval pointer is gone"
    prose = " ".join(pointers[0].split())
    assert "EVAL_BASELINES_DIR" in prose, (
        "the pointer must name app.paths.EVAL_BASELINES_DIR, the one name that "
        "resolves in production, in the gate's scratch root and in a worktree"
    )
    assert "under `~/lloyd`" not in prose, (
        "the baselines are not in the code checkout; that claim sent every "
        "reader to a directory that does not exist"
    )
    assert P.EVAL_BASELINES_DIR.name == "baselines"
    assert P.EVAL_BASELINES_DIR.parent.name == "eval"


# ── clause 4: the store is reached through the module, never the file ───────

def test_this_file_reaches_the_store_only_through_app_kg_store():
    src = Path(__file__).read_text()
    # Built from parts: written as literals these very lines would trip them.
    for forbidden in ("import " + "sqlite3", "sqlite" + "3.connect", "py" + "sqlite"):
        assert forbidden not in src, (
            f"the store is reached through app.kg_store, never by opening the "
            f"database file ({forbidden!r})"
        )


# ── the seam: who actually reads this file, and the copy they get ───────────

def test_the_digest_worker_reads_the_file_this_test_guards():
    """`workers/sources/youtube_digest.py` interpolates a *path* into the
    digest session's prompt, so nothing at that boundary checks that the file
    at the end of it is the one whose claims were verified. If it moves, every
    assertion above silently guards the wrong document.
    """
    from workers.sources import youtube_digest as YD

    assert Path(YD.PROFILE_PATH).resolve() == PROFILE, (
        "the idea-generating worker reads a different file than these claims"
    )
    assert PROFILE.is_file()


@pytest.mark.live_vault
@pytest.mark.skipif(not AI_ENGINEER_SKILL.exists(), reason="vault not present")
def test_the_ai_engineer_skill_names_the_same_profile():
    """The other reader of this document is a vault skill, so it is marked
    live_vault: a nightly skill job can rewrite it between rounds."""
    assert "eval/lloyd_profile.md" in AI_ENGINEER_SKILL.read_text(), (
        "skills/ai-engineer-monitor points at some other profile; the claims "
        "pinned here are not the ones that session reads"
    )


# ── clause 5: the Standing-problems list is the useful half; leave it alone ─

def test_standing_problems_list_still_has_its_nine_items():
    text = PROFILE.read_text()
    assert "## Standing problems worth solving" in text
    section = text.split("## Standing problems worth solving", 1)[1].split("\n## ", 1)[0]
    numbered = re.findall(r"^\d+\.\s", section, re.MULTILINE)
    assert len(numbered) == 9, f"expected nine numbered problems, found {len(numbered)}"
    assert "Entity identification in retrieval" in section
    assert "Research pipeline" in section
