"""#711: the vault-landing skill-activation gate — advisory by default.

A tmp vault, fixture skills and a fixture corpus: the candidate is a real
SKILL.md on disk and the current text is the vault's HEAD, exactly as a
consolidation landing sees them, so this runs in the gate's `not live_vault`
rung.
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


@pytest.mark.parametrize("desc, why", [(NOISY, "false triggers 0 -> 2"),
                                       (WORSE, "recall 0.2 under the floor 0.6")])
def test_enforcing_refuses_a_degraded_skill_and_puts_it_back(vault, monkeypatch, desc, why):
    monkeypatch.setattr(V, "SKILL_ACTIVATION_ENFORCE", True)
    _rewrite(vault, desc)
    with pytest.raises(V.VaultRoundError, match="skill activation"):
        V.land([f"skills/{TEA}/SKILL.md"], "degrade tea")
    assert (vault / "skills" / TEA / "SKILL.md").read_text() == _md(BASE)
    [row] = V.skill_activation_findings([f"skills/{TEA}/SKILL.md"])
    assert row["would_refuse"] is False, "reverted: the tree is back at HEAD"
    _rewrite(vault, desc)
    [row] = V.skill_activation_findings([f"skills/{TEA}/SKILL.md"])
    assert row["would_refuse"] is True and why in row["reason"]


def test_enforcing_allows_an_improved_skill(vault, monkeypatch):
    monkeypatch.setattr(V, "SKILL_ACTIVATION_ENFORCE", True)
    _rewrite(vault, BETTER)
    out = V.land([f"skills/{TEA}/SKILL.md"], "sharpen tea")
    [row] = out["skill_gate"]
    assert row["would_refuse"] is False
    assert row["candidate"]["recall"] == 1.0 and row["current"]["recall"] == 0.6


def test_off_by_default_it_lands_and_records_what_it_would_have_refused(vault):
    assert V.SKILL_ACTIVATION_ENFORCE is False
    _rewrite(vault, NOISY)
    out = V.land([f"skills/{TEA}/SKILL.md"], "degrade tea")
    assert out["ok"] is True
    [row] = out["skill_gate"]
    assert row["skill"] == TEA and row["would_refuse"] is True
    assert "false triggers 0 -> 2" in row["reason"]
    ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "vault_land"][-1]
    assert ev["skill_gate"] == out["skill_gate"]


def test_a_skill_with_no_eval_is_never_blockable(vault, monkeypatch):
    monkeypatch.setattr(V, "SKILL_ACTIVATION_ENFORCE", True)
    (vault / "skills" / "uncovered" / "SKILL.md").write_text(_md("anything at all", "uncovered"))
    out = V.land(["skills/uncovered/SKILL.md"], "edit uncovered")
    [row] = out["skill_gate"]
    assert row == {"skill": "uncovered", "has_eval": False, "would_refuse": False,
                   "reason": "no activation eval for this skill"}


def test_a_gate_that_breaks_does_not_block(vault, monkeypatch):
    monkeypatch.setattr(V, "SKILL_ACTIVATION_ENFORCE", True)
    monkeypatch.setattr(A, "base_skills", lambda: (_ for _ in ()).throw(OSError("disk")))
    _rewrite(vault, NOISY)
    out = V.land([f"skills/{TEA}/SKILL.md"], "degrade tea")
    [row] = out["skill_gate"]
    assert row["would_refuse"] is False and row["reason"].startswith("gate unavailable")


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
