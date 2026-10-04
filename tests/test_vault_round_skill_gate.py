"""#711: the vault-landing skill-activation gate — enforcing since #2148.

A tmp vault, fixture skills and a fixture corpus: the candidate is a real
SKILL.md on disk and the current text is the vault's HEAD, exactly as a
consolidation landing sees them, so this runs in the gate's `not live_vault`
rung. The flag now ships on, so the enforcement nodes stand on the shipped
default with no monkeypatch of it, and the log-only behaviour the off-by-default
node used to cover is pinned with the flag stood down explicitly (#2148 clause
5). One node here reads the vault rather than a tmp tree — the last, which
re-derives the ledger figures the flip was ruled on from the bytes the vault
committed, tolerating an absent vault root the way #2046's witness node does.
"""
from __future__ import annotations

import subprocess

import pytest

from scripts import skill_activation as A
from scripts.automod import state as S, vault_round as V

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


def test_the_activation_rail_ships_enforcing_and_the_body_ceiling_does_not():
    """#2148 clause 5, first half: the two flags' shipped values, read off the module.

    The activation rail is on because nine days of its own `skill_gate` rows said
    it would have refused nothing — 137 rows, 0 would-refusals, the bytes the last
    node of this file re-derives. The body-line ceiling is off because
    `architecture/skills.md` still calls its 100-line cap advisory and #1985 left
    the flip to a person. Two flags, two rulings, and one node that keeps a later
    reader from flipping the wrong one by assuming they travel together.
    """
    assert V.SKILL_ACTIVATION_ENFORCE is True
    assert V.SKILL_BODY_ENFORCE is False


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

from scripts import skill_lint as SL

SPILLED = SL.SPILL_SAMPLE[0]


def _sized(slug: str, body_lines: int) -> str:
    body = "\n".join([f"# {slug}"] + [f"step {i}" for i in range(1, body_lines)])
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


def test_off_by_default_an_over_cap_rewrite_lands_and_the_row_carries_the_finding(vault):
    assert V.SKILL_BODY_ENFORCE is False
    rel = _write_skill(vault, SPILLED, SL.MAX_BODY_LINES + 1)
    errors, _ = V.validate([rel])
    assert errors == []
    out = V.land([rel], "push it over")
    assert out["ok"] and git(vault, "status", "--porcelain").stdout == "", "committed"
    row = [r for r in _ledger_rows() if r.get("event") == "vault_land"][-1]
    assert row["ok"] is True and "errors" not in row
    assert row["skill_body"][0]["skill"] == SPILLED
    assert row["skill_body"][0]["body_lines"] == SL.MAX_BODY_LINES + 1


def test_enforcing_refuses_the_over_cap_rewrite_and_names_the_ceiling(vault, monkeypatch):
    monkeypatch.setattr(V, "SKILL_BODY_ENFORCE", True)
    rel = _write_skill(vault, SPILLED, SL.MAX_BODY_LINES + 1)
    errors, _ = V.validate([rel])
    assert len(errors) == 1
    assert "body-line ceiling" in errors[0] and rel in errors[0]
    assert f"{SL.MAX_BODY_LINES}-line" in errors[0]
    with pytest.raises(V.VaultRoundError, match="body-line ceiling"):
        V.land([rel], "push it over")


def test_a_skill_outside_the_spill_sample_is_never_measured(vault, monkeypatch):
    monkeypatch.setattr(V, "SKILL_BODY_ENFORCE", True)
    assert "uncovered" not in SL.SPILL_SAMPLE
    rel = _write_skill(vault, "uncovered", SL.MAX_BODY_LINES * 3)
    assert V.skill_body_findings([rel]) == []
    assert V.validate([rel])[0] == []
    out = V.land([rel], "a long unsampled skill")
    assert out["ok"] and "skill_body" not in out
