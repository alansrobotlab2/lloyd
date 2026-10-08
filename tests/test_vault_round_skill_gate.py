"""#711: the vault-landing skill rails — activation enforcing since #2148, the
body-line ceiling enforcing since #2158.

A tmp vault, fixture skills and a fixture corpus: the candidate is a real
SKILL.md on disk and the current text is the vault's HEAD, exactly as a
consolidation landing sees them, so this runs in the gate's `not live_vault`
rung. Both flags now ship on, so the enforcement nodes stand on the shipped
default with no monkeypatch of them, and the log-only behaviour the
off-by-default nodes used to cover is pinned in each rail with the flag stood
down explicitly (#2148 clause 5, mirrored for the ceiling by #2158). Two nodes
here read the real tree rather than the tmp one: the activation witness node
re-derives the ledger figures that flip was ruled on from the bytes the vault
committed, tolerating an absent vault root the way #2046's witness node does,
and the last one re-reads `vault_round.py` and `architecture/skills.md` to pin
that no surface still defers the ceiling flip to a person.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import skill_activation as A
from scripts.automod import state as S, vault_round as V

ROOT = Path(__file__).resolve().parent.parent

TEA = "tea-brewing"
RECORDS = [
    {"id": "p1", "turn": "how long should I steep oolong", "expected_skills": [TEA]},
    {"id": "p2", "turn": "what temperature for green leaves", "expected_skills": [TEA]},
    {"id": "p3", "turn": "steep the oolong again please", "expected_skills": [TEA]},
    {"id": "p4", "turn": "brew a pot of tea for me", "expected_skills": [TEA]},
    {"id": "p5", "turn": "the green leaves taste bitter, what temperature", "expected_skills": [TEA]},
    {"id": "n1", "turn": "make coffee in the french press", "expected_skills": ["coffee-making"]},
    {"id": "n2", "turn": "espresso shot is too sour", "expected_skills": ["coffee-making"]},
    {"id": "n3", "turn": "grind coffee beans coarse for french press", "expected_skills": ["coffee-making"]},
    {"id": "n4", "turn": "restart the backend server now", "expected_skills": [], "expected_empty": True},
    {"id": "n5", "turn": "what's the weather tomorrow afternoon", "expected_skills": [], "expected_empty": True},
]
BASE = "steep oolong leaves"                                       # recall 0.6, 0 false
BETTER = "steep oolong and green leaves at the right temperature"  # recall 1.0, 0 false
NOISY = BASE + ", coffee espresso french press"                    # recall 0.6, 2 false
WORSE = "brewing guide"                                            # recall 0.2, 0 false
# Different bytes, the same scores as BASE (recall 0.6, 0 false triggers): the
# reword a consolidation run actually writes, which the rail must not mistake for
# a regression just because the file changed.
UNCHANGED = "steep oolong leaves for three minutes"


def _md(desc, name=TEA):
    return f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\n"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


@pytest.fixture
def vault(tmp_path, monkeypatch):
    r = tmp_path / "obsidian"
    for slug in (TEA, "uncovered"):
        (r / "skills" / slug).mkdir(parents=True)
    (r / "skills" / TEA / "SKILL.md").write_text(_md(BASE))
    (r / "skills" / "uncovered" / "SKILL.md").write_text(_md("knit wool scarves", "uncovered"))
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com"); git(r, "config", "user.name", "t")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "base")
    monkeypatch.setattr(V, "VAULT", r)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(V, "loader_errors", lambda paths: [])
    coffee = A.skill_from_text("coffee-making",
                               _md("coffee french press espresso grinder beans", "coffee-making"))
    monkeypatch.setattr(A, "load_spec", lambda *a, **k: {
        "skills": {TEA: {"recall_floor": 0.6}}, "extra_records": []})
    monkeypatch.setattr(A, "load_records", lambda *a, **k: RECORDS)
    monkeypatch.setattr(A, "base_skills", lambda: [coffee])
    return r


def _rewrite(vault, desc):
    (vault / "skills" / TEA / "SKILL.md").write_text(_md(desc))


def test_both_skill_rails_ship_enforcing_and_are_ruled_separately():
    """#2148 clause 5 (first half) and #2158: the two flags' shipped values, off the module.

    Both rails now enforce, and the node says so because each was ruled on its
    own evidence — the two flags do not travel together, and a later reader who
    assumed they did could stand the wrong one down. The activation rail is on
    because nine days of its own `skill_gate` rows said it would have refused
    nothing — 137 rows, 0 would-refusals, the bytes the witness node of this
    file re-derives (#2148). The body-line ceiling is on because the log #1985
    shipped caught two real over-cap landings — one 103-line body that landed
    on 2026-10-03 and had to be hand-fixed, and #1534 at 116 lines whose only
    red check was a `live_vault` node the gate deselects — and #2158 ruled the
    five-skill sample enforcing over that log on 2026-10-04.
    """
    assert V.SKILL_ACTIVATION_ENFORCE is True
    assert V.SKILL_BODY_ENFORCE is True


@pytest.mark.parametrize("desc, why", [(NOISY, "false triggers 0 -> 2"),
                                       (WORSE, "recall 0.2 under the floor 0.6")])
def test_the_shipped_default_refuses_a_degraded_skill_and_puts_it_back(vault, desc, why):
    """#2148 clauses 1 and 2, at the shipped value: no monkeypatch of the flag here.

    The flag is off the module, not off a fixture, because what is being shipped
    is the default. Everything the refusal is claimed to do is read out of the
    ledger file rather than out of the raised message's shape: the row must be the
    refusal row (`ok: False`), it must carry the same `skill_gate` rows a passing
    landing carries, and the reason in them must name the rail that fired — false
    triggers or recall under the floor — which is the half #2148 exists to fix,
    since the gate used to run only after validation had already passed.
    """
    rel = f"skills/{TEA}/SKILL.md"
    _rewrite(vault, desc)
    # Measured BEFORE the land, and that order is load-bearing: the land puts the
    # file back, so the same call afterwards reports "no regression" and any
    # comparison made then would pass against a row that named nothing.
    before = V.skill_activation_findings([rel])
    [finding] = before
    assert finding["would_refuse"] is True and why in finding["reason"]
    with pytest.raises(V.VaultRoundError, match="skill activation") as raised:
        V.land([rel], "degrade tea")
    assert rel in str(raised.value), "the refusal names the path it refused"
    assert (vault / rel).read_text() == _md(BASE), "the candidate is gone from the tree"
    assert git(vault, "show", f"HEAD:{rel}").stdout == (vault / rel).read_text(), \
        "the tracked SKILL.md is not byte-identical to HEAD"
    assert git(vault, "status", "--porcelain").stdout == "", "nothing left staged behind it"
    [row] = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "vault_land"]
    assert row["ok"] is False and row["reverted"] == [rel], row
    assert row["skill_gate"] == before, (
        "the refusal row does not carry the rows the passing row would carry")
    [gate] = row["skill_gate"]
    assert {"skill", "has_eval", "would_refuse", "reason"} <= set(gate), sorted(gate)
    assert gate["skill"] == TEA and gate["has_eval"] is True and gate["would_refuse"] is True
    assert why in gate["reason"], f"the row does not name the rail: {gate['reason']}"


def test_the_shipped_default_lands_a_reword_the_scores_do_not_move(vault):
    """#2148 clause 1, the unchanged leg: new bytes, the same recall and the same
    false-trigger count, so nothing is a regression and the landing goes through."""
    rel = f"skills/{TEA}/SKILL.md"
    _rewrite(vault, UNCHANGED)
    out = V.land([rel], "reword the tea skill")
    assert out["ok"] is True
    [gate] = out["skill_gate"]
    assert gate["would_refuse"] is False and gate["reason"] == "no regression"
    assert gate["candidate"] == gate["current"], "measured, and identical to HEAD"


def test_the_shipped_default_lands_an_improved_skill(vault):
    """#2148 clause 1, the improved leg: the whole point of the rail is to let
    this through, so it is pinned at the shipped default and not under a flag."""
    rel = f"skills/{TEA}/SKILL.md"
    _rewrite(vault, BETTER)
    out = V.land([rel], "sharpen tea")
    [gate] = out["skill_gate"]
    assert gate["would_refuse"] is False
    assert gate["candidate"]["recall"] == 1.0 and gate["current"]["recall"] == 0.6


def test_the_flag_stood_down_by_hand_lands_a_would_refuse_and_logs_it(vault, monkeypatch):
    """#2148 clause 5, second half: the log-only behaviour the old default node covered.

    The node this replaced let a degraded body through *because the module shipped
    off*, and asserted that off-ness. The module no longer ships off, so the
    behaviour stays pinned the only way that still means something: the flag stood
    down in this node, explicitly, rather than inherited from the default. What it
    must still do is land the change and record what it would have refused — that
    is the log the flip was ruled on, so it cannot go silent with the flip.
    """
    monkeypatch.setattr(V, "SKILL_ACTIVATION_ENFORCE", False)
    _rewrite(vault, NOISY)
    out = V.land([f"skills/{TEA}/SKILL.md"], "degrade tea")
    assert out["ok"] is True
    [row] = out["skill_gate"]
    assert row["skill"] == TEA and row["would_refuse"] is True
    assert "false triggers 0 -> 2" in row["reason"]
    ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "vault_land"][-1]
    assert ev["skill_gate"] == out["skill_gate"], "the committed row says the same thing"


def test_a_skill_with_no_eval_is_never_blockable(vault):
    """#2148 clause 3: the corpus, not a list, decides what enforcement can reach.

    No monkeypatch: enforcement is the shipped default now, and this is the bound
    that made shipping it narrow. In this fixture one slug carries a floor, so
    only it is blockable; in the real `eval/skill_activation_cases.yaml` five
    slugs carry one and the other 124 of the 137 witness rows the last node reads
    are exactly the row asserted below — `has_eval: False`, never refused.
    """
    (vault / "skills" / "uncovered" / "SKILL.md").write_text(_md("anything at all", "uncovered"))
    out = V.land(["skills/uncovered/SKILL.md"], "edit uncovered")
    [row] = out["skill_gate"]
    assert row == {"skill": "uncovered", "has_eval": False, "would_refuse": False,
                   "reason": "no activation eval for this skill"}


def test_a_gate_that_breaks_does_not_block(vault, monkeypatch):
    """#2148 clause 4: turning the rail on did not make it able to block on its own
    failure. No monkeypatch of the flag — this is the enforcing default meeting a
    scoring call that raises, which is the case a person would otherwise fear."""
    monkeypatch.setattr(A, "base_skills", lambda: (_ for _ in ()).throw(OSError("disk")))
    _rewrite(vault, NOISY)
    out = V.land([f"skills/{TEA}/SKILL.md"], "degrade tea")
    [row] = out["skill_gate"]
    assert row["would_refuse"] is False and row["reason"].startswith("gate unavailable")


# ── #2148: the ledger bytes the flip was ruled on, as committed evidence ──────
#
# The comment above `SKILL_ACTIVATION_ENFORCE` quotes one report: over the window
# the rail logged, it would have refused nothing. That report lived in
# `~/.local/state/lloyd-automod/promotions.jsonl` — one machine's state dir, folded
# by the groundskeeper — so #2148's clause 6 asked for the rows themselves to be
# committed. They are, at the vault path below: 137 lines, one per `skill_gate`
# entry, each beside its landing's `created_at`, `commit`, `item_id` and skill
# paths. A dated filename, not the `backlog/data/` promotions mirror that clause 6
# names: that path is the wholesale copy #2050 retired and #2064 deleted, and
# `test_no_reader_under_tests_or_scripts_opens_the_retired_mirror` in
# `tests/test_automod_vault_round.py` refuses a new file naming it — the same wall
# round SM_20261002_233736 hit with the same clause shape on #2079, which resolved
# it the same way.

WITNESS_VAULT_PATH = "backlog/data/2026-10-04.2148-skill-activation-witness.jsonl"


def test_the_committed_witness_bytes_hold_the_rows_the_flip_is_ruled_on():
    """#2148 clause 6: every number the flag is set on, re-derived from committed bytes.

    The figures in `SKILL_ACTIVATION_ENFORCE`'s comment are asserted here against
    the vault's copy of the ledger extract — 137 rows, 0 of them a would-refusal,
    13 of them on a covered skill — so the flip rests on bytes in the vault's
    history rather than on a claim about one machine's state dir. That is what
    makes the report falsifiable: a reader with the file and no ledger can run the
    same counts, and a witness rewritten to flatter the rail reddens this node.
    The durable leg tolerates an absent vault root the way #2046's witness node
    does, because the gate's tree may not carry one; the figures themselves are
    pinned here, so an absent root says "this leg did not run", never "clean".
    """
    import collections
    import json

    import scripts.automod.review as RV

    roots = [r for r in RV.REVIEW_EVIDENCE_ROOTS if r.is_dir()]
    if not roots:
        pytest.skip("no vault root on this tree, so the durable witness leg cannot be read")
    witness = roots[0] / WITNESS_VAULT_PATH
    assert witness.is_file(), (
        f"the witness bytes are not on the vault's main: {witness} — the flip's "
        f"report has no evidence behind it until they are re-landed")
    lines = witness.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 137, f"{len(lines)} rows, not the 137 the item quotes"
    rows = [json.loads(line) for line in lines]
    assert all({"skill", "has_eval", "would_refuse", "reason"} <= set(r) for r in rows), \
        "a witness row that does not carry the four fields the ledger row carries"
    assert sum(1 for r in rows if r["would_refuse"]) == 0, (
        "a would-refusal in these bytes: the ruling below is no longer 0 refusals "
        "on 137 rows, and the flag's comment has to be re-measured, not re-quoted")
    assert sum(1 for r in rows if r["has_eval"]) == 13, "covered-skill rows"
    assert collections.Counter(r["skill"] for r in rows if r["has_eval"]) == \
        collections.Counter({"system-health-check": 10, "voice-mode": 2,
                             "local-llm-gotchas": 1})
    assert sum(1 for r in rows
               if r["reason"] == "no activation eval for this skill") == 124
    stamps = sorted(r["created_at"] for r in rows)
    assert (stamps[0], stamps[-1]) == ("2026-09-25T01:22:44Z", "2026-10-03T23:26:57Z"), \
        "the nine days the comment names are not the window these bytes cover"
    assert len({r["commit"] for r in rows}) == 116, "one row per gate row, 116 landings"
    assert len({r["skill"] for r in rows}) == 47, "47 distinct skills touched"


# ── #1985: the body-line ceiling on a spill-sampled skill, at the writer ─────
#
# #1985 shipped the ceiling log-only and #2158 (2026-10-04) ruled it enforcing
# on the five `SPILL_SAMPLE` skills; every node below stands on the shipped
# default except the one that pins the log-only behaviour, which — exactly as
# #2148's mirror does for the activation rail — stands the flag down itself.

from scripts import skill_lint as SL

SPILLED = SL.SPILL_SAMPLE[0]


def _sized(slug: str, body_lines: int) -> str:
    """A body of exactly `body_lines` lines, all of them under one `## Steps`
    heading — so `largest_block` names a real heading, the way a real over-cap
    skill's does, and the refusal's spill advice is checkable."""
    body = "\n".join([f"# {slug}", "## Steps"]
                     + [f"step {i}" for i in range(1, body_lines - 1)])
    return f"---\nname: {slug}\ndescription: knit wool scarves\n---\n{body}\n"


def _write_skill(vault, slug: str, body_lines: int) -> str:
    d = vault / "skills" / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(_sized(slug, body_lines))
    return f"skills/{slug}/SKILL.md"


def _ledger_rows():
    import json
    return [json.loads(l) for l in S.LEDGER_PATH.read_text().splitlines() if l.strip()]


def test_a_spilled_skill_at_the_cap_has_no_size_finding(vault):
    rel = _write_skill(vault, SPILLED, SL.MAX_BODY_LINES)
    assert V.skill_body_findings([rel]) == []
    out = V.land([rel], "rewrite at the cap")
    assert out["ok"] and "skill_body" not in out


def test_one_line_past_the_cap_is_a_finding_naming_the_skill_and_its_count(vault):
    rel = _write_skill(vault, SPILLED, SL.MAX_BODY_LINES + 1)
    [row] = V.skill_body_findings([rel])
    assert row["skill"] == SPILLED and row["body_lines"] == SL.MAX_BODY_LINES + 1
    assert str(SL.MAX_BODY_LINES + 1) in row["reason"]


def test_the_shipped_default_refuses_an_over_cap_rewrite_and_ledgers_the_finding(vault):
    """#2158 clause 1, at the shipped value: no monkeypatch of the flag here.

    The node this replaced pinned the opposite — `errors == []`, the over-cap
    rewrite committing, the finding merely carried on the row — because the
    module then shipped log-only. #2158 ruled the ceiling enforcing on the
    `SPILL_SAMPLE` five, so what must be true now is the refusal: `validate`
    answers with exactly the one `body-line ceiling` error, `land` raises and
    commits nothing, and the ledger row carries the `skill_body` finding beside
    its `ok: False`. That row is computed before the revert for the reason the
    activation rail's is: after `revert_paths` the sampled file is back under
    the cap or gone, and a row written afterwards would state the absence of
    the very over-cap state it is refusing.
    """
    rel = _write_skill(vault, SPILLED, SL.MAX_BODY_LINES + 1)
    errors, _ = V.validate([rel])
    assert len(errors) == 1, f"not the one ceiling error: {errors}"
    assert "body-line ceiling" in errors[0] and rel in errors[0]
    with pytest.raises(V.VaultRoundError, match="body-line ceiling") as raised:
        V.land([rel], "push it over")
    assert rel in str(raised.value), "the refusal names the path it refused"
    assert not (vault / rel).exists(), "the untracked candidate was not left standing"
    assert git(vault, "status", "--porcelain").stdout == "", "nothing committed or staged"
    [row] = [r for r in _ledger_rows() if r.get("event") == "vault_land"]
    assert row["ok"] is False and row["reverted"] == [rel], row
    [finding] = row["skill_body"]
    assert finding["skill"] == SPILLED
    assert finding["body_lines"] == SL.MAX_BODY_LINES + 1
    assert finding["would_refuse"] is True


def test_the_refusal_names_the_ceiling_and_the_block_to_spill(vault):
    """#2158 clause 3: a refusal has to be actionable where it fires.

    The error string carries the skill path, the measured count, the 100-line
    ceiling, and the clearing action naming the largest `##`/`###` block — the
    heading `skill_lint.skill_size`'s `_largest_block` picks and its docstring
    calls "the first spill candidate" — so the writer knows which section to
    move into a sibling file without recomputing anything. The same heading
    rides on the finding row, so the ledger says it too.
    """
    rel = _write_skill(vault, SPILLED, SL.MAX_BODY_LINES + 1)
    f = vault / rel
    content = f.read_text(encoding="utf-8")
    size = SL.skill_size(SPILLED, f, content, SL.parse_frontmatter(content)[1])
    heading = size["largest_block"]["heading"]
    assert heading == "## Steps"
    assert size["largest_block"]["lines"] == SL.MAX_BODY_LINES
    errors, _ = V.validate([rel])
    [error] = errors
    assert rel in error
    assert f"{SL.MAX_BODY_LINES + 1} lines" in error, error
    assert f"{SL.MAX_BODY_LINES}-line" in error, error
    assert heading in error, f"the refusal does not name the block to spill: {error}"
    [finding] = V.skill_body_findings([rel])
    assert finding["largest_block"]["heading"] == heading


def test_the_ceiling_measures_the_resulting_state_not_the_shrink(vault):
    """#2158 clause 4: the docstring's state-based rule, pinned both directions.

    HEAD here holds a 120-line body, committed straight through git because the
    enforcing rail now refuses any landing that would produce it. A rewrite
    that cuts 19 lines and lands at 101 SHRANK the skill and is still refused:
    the measure is the candidate's resulting state, never its delta. A rewrite
    that lands at exactly the cap is allowed however it got there. And the
    revert between the two legs restores the HEAD state — 120 lines — not the
    candidate, which is what "did not commit" has to mean for a tracked file.
    """
    rel = _write_skill(vault, SPILLED, 120)
    git(vault, "add", "-A")
    git(vault, "commit", "-q", "-m", "a 120-line body, pre-flip bytes")
    _write_skill(vault, SPILLED, SL.MAX_BODY_LINES + 1)
    [finding] = V.skill_body_findings([rel])
    assert finding["body_lines"] == SL.MAX_BODY_LINES + 1, "measured on the candidate"
    with pytest.raises(V.VaultRoundError, match="body-line ceiling"):
        V.land([rel], "cut 19 lines, still over the cap")
    assert (vault / rel).read_text() == _sized(SPILLED, 120), "HEAD's 120 lines restored"
    rel = _write_skill(vault, SPILLED, SL.MAX_BODY_LINES)
    assert V.validate([rel])[0] == [], "a shrink that lands at the cap must be allowed"
    assert V.land([rel], "spill until it fits")["ok"] is True


def test_the_body_rail_stood_down_by_hand_lands_an_over_cap_rewrite_and_logs_it(vault,
                                                                                monkeypatch):
    """#2158 clause 1's other half, in #2148's shape: the log-only behaviour, pinned
    with the flag stood down by hand rather than inherited from a retired default.

    The module no longer ships log-only, so "an over-cap rewrite lands and the
    row carries the finding" only still means something with
    `SKILL_BODY_ENFORCE` stood down inside the node. The behaviour itself is
    unchanged — validate is silent, the landing commits, the `skill_body`
    finding rides on the row — and that is the log #2158's ruling was made
    from, so the flip may not take it silent with it.
    """
    monkeypatch.setattr(V, "SKILL_BODY_ENFORCE", False)
    rel = _write_skill(vault, SPILLED, SL.MAX_BODY_LINES + 1)
    assert V.validate([rel])[0] == []
    out = V.land([rel], "push it over, flag stood down")
    assert out["ok"] and git(vault, "status", "--porcelain").stdout == "", "committed"
    row = [r for r in _ledger_rows() if r.get("event") == "vault_land"][-1]
    assert row["ok"] is True and "errors" not in row
    assert row["skill_body"][0]["skill"] == SPILLED
    assert row["skill_body"][0]["body_lines"] == SL.MAX_BODY_LINES + 1


def test_a_skill_outside_the_spill_sample_is_never_measured(vault):
    """#1985's scope rule, at the shipped default since #2158: no monkeypatch.

    Three times the cap and still no row: enforcement reaches
    `skill_lint.SPILL_SAMPLE` and nothing else. That scope is ruled twice over —
    #2158 set it, and #2334 ruled library-wide enforcement NO on 2026-10-08 — so
    no count of the library is quoted here. The figure this docstring carried was
    measured on 2026-10-01 and had moved by the 2026-10-07 re-run; the count that
    would decide a library-wide rule is `skill_lint.render_size`'s to print on
    every sweep, so this node keeps the scope and points at the sweep instead of
    transcribing a number that ages in a docstring.
    """
    assert "uncovered" not in SL.SPILL_SAMPLE
    rel = _write_skill(vault, "uncovered", SL.MAX_BODY_LINES * 3)
    assert V.skill_body_findings([rel]) == []
    assert V.validate([rel])[0] == []
    out = V.land([rel], "a long unsampled skill")
    assert out["ok"] and "skill_body" not in out


def test_no_surface_still_carries_the_library_wide_question_as_open():
    """#2436 clause 4: the ruling is settled, so a surface that still defers is the drift.

    This node's job inverted when #2334 ruled on 2026-10-08. It used to refuse any
    surface that took the ceiling decision out of a person's hands; the person has
    answered — SIZE stays advisory library-wide, `MAX_BODY_LINES` (100) is the
    `SPILL_SAMPLE` spill target and not a library ceiling, the p90 is rejected as a
    ceiling because it moves — so the drift is now a surface that keeps sending a
    session to a person who has already ruled. It pins the settled content in all three
    places the question was ever deferred (`skill_lint.py`'s size section and its
    constant comment are the third, graded in `tests/test_skill_lint_size.py`), and it
    pins the other half that keeps this from being won by erasure: the doc must still
    name the spill-sample scope, the constant, and the fact that widening enforcement
    needs a NEW ruled item rather than an edit.

    The section comment and constant comment in `scripts/skill_lint.py` carry the same
    ruling and are graded by `tests/test_skill_lint_size.py`, so this node reads only
    the two surfaces it is the rail for: the enforce comment in the module under test,
    and the doc that comment sends a reader to.
    """
    src = (ROOT / "scripts" / "automod" / "vault_round.py").read_text(encoding="utf-8")
    doc = (ROOT / "architecture" / "skills.md").read_text(encoding="utf-8")
    here = Path(__file__).read_text(encoding="utf-8")

    # Nothing defers the flip any more, on any of the three surfaces.
    assert "human's change" not in src, "the module defers the flip again"
    assert "person" not in doc.lower(), (
        "architecture/skills.md defers the library-wide question to a person again, and "
        "#2334 settled it on 2026-10-08")
    assert "#2334" in doc and "2026-10-08" in doc, (
        "the settled ruling has to arrive dated and named, or a reader cannot tell it "
        "from the open question it replaced")

    # What the ruling IS, in the doc's words.
    dflat = " ".join(doc.split())
    for ruling in ("advisory library-wide", "`SPILL_SAMPLE` spill target", "ruled NO",
                   "MAX_BODY_LINES"):
        assert ruling in dflat, (
            f"the doc no longer states the ruling ({ruling!r}), so the surfaces that "
            f"point at it are pointing at nothing: {dflat}")

    # The scope that survives the ruling, still stated and still narrow.
    assert "cap is enforced on vault landings" in doc, (
        "the spill-sample scope is gone rather than settled: the rail would be "
        "unenforced and this doc silent about it")
    assert "fresh ruled item" in dflat, (
        "widening past SPILL_SAMPLE must read as a new ruled item's decision — without "
        "that sentence 'advisory library-wide' reads as a suggestion a round can ignore")
    assert "SPILL_SAMPLE" in src, (
        "the enforce scope is unchanged, and the comment that states it has to keep "
        "naming the set it covers")

    # And this file may not quietly restore the old rail. The needle is built at
    # runtime, because a literal here would be the deferral this assert refuses.
    deferral = "remains a " + "person"
    offending = [ln for ln in here.split(chr(10)) if deferral in ln]
    assert not offending, (
        "a surface in this file still defers the ceiling flip: " + offending[0])


#: The knowledge-write skill as the nightly skills pass maintains it, in the live
#: vault itself — the writer whose unred-checked commit reddened `main` on
#: 2026-10-04 by rewriting an index line a code test pins.
KNOWLEDGE_WRITE = "skills/nightly-reflection-knowledge-write/SKILL.md"
LIVE_VAULT = Path.home() / "obsidian"

@pytest.mark.live_vault
def test_the_knowledge_write_skill_carries_a_memory_index_pre_flight():
    """Clauses 3, 4 and 5 read off the live skill the job actually follows.

    Marked `live_vault` deliberately, which is the ruling this item exists to make:
    the skill body is owned by the nightly skills pass, and a node that reads it
    unmarked is the very failure mode #2173 is fixing — it reddens the next author's
    gate for the previous writer's edit. So this is the reporting copy, run with
    `-m live_vault`, while the *enforcement* of clause 5 is the landing rail above,
    graded on fixtures and in the gate's own selection.

    Run it with:
        .venvs/lloyd/bin/python -m pytest -m live_vault \\
            tests/test_vault_round_skill_gate.py -k pre_flight
    """
    text = (LIVE_VAULT / KNOWLEDGE_WRITE).read_text(encoding="utf-8")
    body = SL.parse_frontmatter(text)[1].strip("\n")
    assert len(body.splitlines()) <= SL.MAX_BODY_LINES, (
        f"the pre-flight pushed the body to {len(body.splitlines())} lines; the "
        f"vault-landing route refuses a SPILL_SAMPLE skill over "
        f"{SL.MAX_BODY_LINES}, so this commit could not land")

    pre = text.index("Memory-index pre-flight")
    commit = text.index("## Step 3: Commit")
    assert pre < commit, "the pre-flight sits after the commit step it must gate"
    step = text[pre:commit]

    # Clause 3: it runs the validator over the whole index, and the commit is
    # forbidden on a report that is not ok — not "consider fixing".
    assert "scripts/memory/validate_memory_index.py --mode full" in step, step
    assert '"ok": false' in step and "FORBIDDEN" in step, step
    assert "2026-10-04" in step, "the step must say which commit it exists for"

    # Clause 4: a pinned index line is found before it is rewritten, and the fix
    # goes to the detail file rather than into the line.
    assert "~/lloyd/tests/" in step, step
    assert "BYTE-IDENTICAL" in step and "lloyd/memory/<slug>.md" in step, step


def test_the_enforce_comment_names_the_doc_by_the_path_this_file_resolves():
    """#2334 clause 3: a citation two files answer to is a citation to neither.

    The comment above `SKILL_BODY_ENFORCE` recorded the ruling as standing in
    "`architecture/skills.md`", meaning the repo copy this file's own size-ruling rail
    opens as `ROOT / "architecture" / "skills.md"`. The bare name also matches the
    vault's `~/obsidian/architecture/skills.md`, and a reader who followed it there
    landed on 113 lines whose front matter still claims "34 current custom skills"
    against 197 live ones and which contain no SIZE section, no spill paragraph and
    no `MAX_BODY_LINES` at all — so the sentence that says "the doc records the
    ruling" was true of one file and false of the file the name reaches from the
    vault. #1985 and #2148 already cite that bare path with line numbers (":411",
    ":441") which exist only in the repo copy; naming the tree is the whole fix, so
    the comment has to keep naming it.
    """
    src = (ROOT / "scripts" / "automod" / "vault_round.py").read_text(encoding="utf-8")
    cut = src.find("SKILL_BODY_ENFORCE = True")
    assert cut > 0, "the switch is still on, which the node above is the ruling for"
    head = src[:cut]
    comment = head[head.rfind("# #1985 shipped"):]
    assert comment.lstrip().startswith("#"), "the citation lives in the comment above it"
    flat = " ".join(comment.split())
    assert 'ROOT / "architecture" / "skills.md"' in flat, (
        "the comment must name the doc by the expression the test that checks it uses: "
        + flat)
    assert "REPO-root" in flat or "repo-root" in flat or "repo root" in flat, flat
    assert "obsidian" in flat.lower(), (
        f"it must say which other file the bare name also reaches: {flat}")
    assert "#2334" in flat and "#2158" in flat and "2026-10-08" in flat, (
        "the spill-sample ruling (#2158) and the library-wide ruling (#2334, dated) "
        "stay distinguishable in the comment that cites the doc: " + flat)

    # The path it names resolves, and is the page that actually carries both halves.
    doc = (ROOT / "architecture" / "skills.md").read_text(encoding="utf-8")
    assert "cap is enforced on vault landings" in doc, (
        "the repo copy must still carry the #2158 scope the comment cites")
    # ...and the page carries the #2334 ruling the comment now cites beside that
    # scope. A doc that quietly reverted to "this question is open" would leave the
    # comment pointing at a page that defers, which is the drift this node refuses.
    dflat = " ".join(doc.split())
    for ruling in ("advisory library-wide", "#2334", "2026-10-08",
                   "`SPILL_SAMPLE` spill target", "ruled NO", "fresh ruled item"):
        assert ruling in dflat, (
            f"the comment cites a page that no longer carries {ruling!r}: " + dflat)
    assert "person" not in doc.lower(), (
        "the doc defers the library-wide question to someone again; #2334 settled it "
        "on 2026-10-08, and this comment sends a reader there")

    # And the ambiguity the comment warns about is real rather than invented: the same
    # relative path exists in the vault and is NOT that page. If the vault arch-doc
    # cleanup (#2330's family) ever moves the SIZE ruling into the vault copy, this
    # fails and the comment gets updated — it does not silently go stale.
    twin = Path.home() / "obsidian" / "architecture" / "skills.md"
    if twin.exists():
        twin_text = twin.read_text(encoding="utf-8")
        assert "MAX_BODY_LINES" not in twin_text and "### SIZE" not in twin_text, (
            "the vault copy now carries the SIZE ruling too, so the comment's claim "
            "that only the repo copy does is stale: rewrite the comment, do not delete it")

