"""skill-lint's report carries its own answer to "is this 0 trustworthy?" (#903).

Task #70's 2026-09-10 calibration found two categories that cannot flag
anything today — DRIFT passes 53% of the library through an untested length
heuristic and STALE returns before its age check for every skill — and wrote
that qualification into the report by hand. `main` rewrites the report
wholesale, so the next run dropped it and the file ended "All skills pass
lint". These tests pin the qualification to `render_report` itself, on
synthetic results, so no vault is read.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "skill_lint_report_trust", ROOT / "scripts" / "skill_lint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sl = _load()


def _result(**over):
    base = {"generated_at": "2026-09-24T00:00:00", "total": 194,
            "dead": [], "missing_desc": [], "drift": [], "duplicates": [],
            "stale": [], "phantom": [], "missing_script": [], "injection": []}
    base.update(over)
    return base


def _table_rows(report: str) -> dict[str, list[str]]:
    lines = report.splitlines()
    start = lines.index("| category | count | action | is this count trustworthy? |")
    rows = {}
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows[cells[0].split(" (")[0]] = cells
    return rows


def test_every_category_row_carries_a_trust_cell_from_the_table():
    rows = _table_rows(sl.render_report(_result()))
    # Both directions: a new row without an entry, or an entry with no row,
    # is a category whose zero would print unqualified (or a stale claim).
    assert set(rows) == set(sl.CATEGORY_TRUST)
    for category, cells in rows.items():
        assert len(cells) == 4, cells
        assert cells[3] == sl.trust_cell(category)
        verdict, cause = sl.CATEGORY_TRUST[category]
        assert verdict in sl.TRUST_VERDICTS and cause.strip()


def test_drift_and_stale_say_untrustworthy_and_name_their_cause():
    rows = _table_rows(sl.render_report(_result()))
    drift, stale = rows["DRIFT"][3], rows["STALE"][3]
    assert drift.startswith(sl.TRUST_MARK["no"])
    assert "first token" in drift and "never tested" in drift
    assert stale.startswith(sl.TRUST_MARK["no"])
    assert "status: active" in stale and "before the age check" in stale
    assert sl.untrustworthy_categories() == ["DRIFT", "STALE"]


def test_all_zero_result_renders_a_qualified_verdict_not_clean():
    report = sl.render_report(_result())
    assert "## ✅ Clean" not in report
    assert "All skills pass lint" not in report
    verdict = report.split("## No findings on the checks that can fail", 1)[1]
    for category in ("DRIFT", "STALE"):
        assert f"**{category}** — 0 is not trustworthy" in verdict


def test_phantom_only_result_keeps_its_section_and_no_clean_verdict():
    report = sl.render_report(_result(
        phantom=[{"name": "websearch", "tools": ["web_search"]}]))
    assert "## PHANTOM_TOOL — 1 skills naming tools that do not exist" in report
    assert "| `websearch` | `web_search` |" in report
    assert "## ✅ Clean" not in report
    assert "All skills pass lint" not in report
    assert "## No findings on the checks that can fail" not in report


def test_drift_path_two_comment_promises_no_calibration_check():
    source = inspect.getsource(sl.check_description_drift)
    assert "calibration check below" not in source


# ── #1603: the STALE bucket's reason, and the sentence that used to stand in ──
#
# Until #1603 the STALE section explained its own emptiness with a promise: the
# usage measurement it wanted "requires injection-level telemetry not yet emitted
# — see #334 follow-ups". That stopped being true on 2026-09-24, when #435 landed
# as `3774e27b` and `prefetch._emit_skill_match_events` started writing one row per
# offered skill per turn. The sentence survived in three places — the rendered
# section, `CATEGORY_TRUST["STALE"]`, and `scripts/skill_lint.py`'s copy of the
# claim — because nothing read them against the emitter. A skill-lint job that has
# run nightly since then has been printing, to whoever reads the report, that a
# measurement exists which does not, which is the exact reason `CATEGORY_TRUST` was
# built.

#: The retired claims, assembled rather than written out so this file never
#: contains the sentence it forbids — a later reader who adds a scan over the test
#: suite itself must not find the ban tripped by the ban's own list. The pieces are
#: greppable in the pre-#1603 `scripts/skill_lint.py` (`git show HEAD~:scripts/skill_lint.py`
#: at the round's commit, or any history before `automod/SM_20260928_123118`).
RETIRED_UNEMITTED_CLAIMS = (
    "not yet " + "emitted",
    "telemetry that is " + "not emitted",
    "no injection " + "event for it",
)


def _unemitted_claims_in(text: str) -> list[str]:
    return [c for c in RETIRED_UNEMITTED_CLAIMS if c in text]


def _stale_section(report: str) -> str:
    start = report.index("## STALE")
    rest = report[start + len("## STALE"):]
    end = rest.find("\n## ")
    return rest if end < 0 else rest[:end]


def test_the_ban_on_the_unemitted_telemetry_claim_bites():
    """Positive control before the three bans below. A `not in report` assertion
    whose phrase list is empty, or whose matcher is broken, passes on any text —
    including a report that still carries the retired sentence. So the matcher is
    exercised on synthetic text first, and the list is held non-empty.
    """
    assert len(RETIRED_UNEMITTED_CLAIMS) == 3, RETIRED_UNEMITTED_CLAIMS
    for claim in RETIRED_UNEMITTED_CLAIMS:
        assert _unemitted_claims_in(f"usage needs telemetry {claim}") == [claim] or \
               claim in "usage needs telemetry " + claim, claim
        assert _unemitted_claims_in(f"## STALE\\nx {claim}") == [claim], claim
    assert _unemitted_claims_in("measured from prefetch.skill_match") == []


def test_stale_reason_names_both_live_halves_and_no_longer_blames_telemetry():
    """#1603 clause 4: `CATEGORY_TRUST["STALE"]` still says the count is not a
    measurement, and now says why in terms of the two things that are actually
    inert — the exemption that returns before the age is read, and an age ceiling
    nothing has reached — instead of a telemetry gap that closed four days later.
    """
    verdict, reason = sl.CATEGORY_TRUST["STALE"]
    assert verdict == "no", (
        "the STALE count is still 0 by construction, so a `yes` here would be the "
        "falsest trust label in the table")
    assert not _unemitted_claims_in(reason), reason

    # Both live halves, named as mechanisms rather than as counts: the counts move
    # with the library, the mechanisms are what a reader needs to act on.
    assert "status: active" in reason, reason
    assert "before" in reason.lower(), reason
    assert "check_stale" in reason, reason
    assert "max_age_days" in reason, reason
    # And the measured half is pointed at, so the `no` is not a dead end.
    assert "prefetch.skill_match" in reason, reason
    assert sl.USAGE_UNMEASURED_LABEL in reason, reason


def test_rendered_stale_section_names_both_halves_and_prints_their_numbers():
    """Clause 4's other half: the report text itself. Rendered from a fixture so the
    numbers below are the fixture's — the point is that the section prints the two
    measurements `stale_context` carries rather than reciting a threshold.
    """
    result = _result()
    result["stale_context"] = {"threshold_days": sl.STALE_DAYS,
                               "marked_active": 189, "unmarked": 4, "total": 193,
                               "max_age_days": 34, "oldest_could_trip": False}
    result["usage"] = {"error": "read failed: OSError: none"}
    report = sl.render_report(result)
    sec = _stale_section(report)
    assert not _unemitted_claims_in(sec), sec
    assert f"{sl.STALE_DAYS}" in sec, sec
    assert "status: active" in sec and "check_stale" in sec, sec
    assert "189 of 193" in sec, sec
    assert "34 days old" in sec, sec
    assert "false" in sec, "the ceiling claim must be printed as a measured verdict"
    # An unreadable store says so, in the section, rather than printing nothing.
    assert "NOT MEASURED" in sec, sec
    # Positive control on the section's own unreachability: the old gate was
    # `if n_stale:`, and `n_stale` is 0 by construction, so a section gated on it
    # printed nothing at all. Assert the section exists with an empty stale list.
    assert result["stale"] == [] and sec, sec


def test_stale_context_measures_both_halves_from_the_walk_not_from_prose():
    """The numbers the section prints, measured: `stale_context` must report the
    exemption's coverage and the oldest file's age over the SAME scan, and must
    disagree with `check_stale` exactly where the exemption is the reason.

    The bug class here is a reason string that has rotted into a guess — the entry
    it replaces asserted "194/194 carry `status: active`" while the live count was
    189 of 193, and no test could tell because the count was never computed. So the
    fixture uses a marked-active skill that is older than the threshold: the verdict
    is still not-stale, and `stale_context` still reports its age.
    """
    ages = [34, 10, 5]
    ctx = sl.stale_context(ages, marked_active=189, total=193)
    assert ctx == {"threshold_days": sl.STALE_DAYS, "marked_active": 189,
                   "unmarked": 4, "total": 193, "max_age_days": 34,
                   "oldest_could_trip": False}, ctx
    assert sl.stale_context([], 0, 0)["max_age_days"] == 0
    assert sl.stale_context([200], 0, 1)["oldest_could_trip"] is True

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        p = Path(td) / "SKILL.md"
        p.write_text("---\nstatus: active\n---\n\nbody\n", encoding="utf-8")
        import os
        old = 1_600_000_000
        os.utime(p, (old, old))
        fm = {"status": "active"}
        assert sl.check_stale(p, fm) == (False, 0), (
            "check_stale stopped returning early for a marked skill, so the "
            "exemption half of the reason string is now false prose")
        age = sl.skill_mtime_age_days(p)
        assert age is not None and age > sl.STALE_DAYS, (
            f"exemption hid the measurement too (age={age}): the report prints this "
            "number precisely because check_stale never reads it")


def test_the_walk_itself_produces_the_two_stale_measurements(tmp_path):
    """Clause 4's numbers come off `lint()`'s own scan, not from a caller's guess.

    The test above hands `stale_context` a list; this one proves the list is
    populated by the real walk, over temp skill roots, with `lint()`'s own record
    shape. The shape that has to survive is the disagreement: one skill is marked
    `status: active` AND older than the threshold, so the bucket stays empty while
    the age measurement is not 0. If `stale_context` were dropped from the result —
    or computed from `stale` instead of from the files — the report would fall back
    to the bare sentence and the section would again be asserting a threshold with
    no measurement beside it.
    """
    import os
    from agent_mcp.skills import iter_active_skills

    for name, status in (("marked-old", "active"), ("marked-new", "active"),
                         ("unmarked-new", "draft")):
        d = tmp_path / name
        d.mkdir()
        f = d / "SKILL.md"
        f.write_text(f"---\nname: {name}\n"
                     "description: Use this skill when testing the stale walk.\n"
                     f"tags: [demo]\nstatus: {status}\n---\n# {name}\n",
                     encoding="utf-8")
        if name == "marked-old":
            old = (dt.datetime.now() - dt.timedelta(days=sl.STALE_DAYS + 40)).timestamp()
            os.utime(f, (old, old))

    result = sl.lint(skill_records=list(iter_active_skills(roots=[tmp_path])))
    assert result["stale"] == [], result["stale"]
    ctx = result["stale_context"]
    assert ctx["marked_active"] == 2 and ctx["total"] == 3, ctx
    assert ctx["unmarked"] == 1, ctx
    assert ctx["max_age_days"] >= sl.STALE_DAYS + 30, ctx
    assert ctx["oldest_could_trip"] is True, ctx
    # The names the usage block subtracts from, measured over the same scan.
    assert result["names"] == ["marked-new", "marked-old", "unmarked-new"], (
        result["names"])
    sec = _stale_section(sl.render_report(result))
    assert "2 of 3" in sec, sec
