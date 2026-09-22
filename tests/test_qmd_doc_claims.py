"""`architecture/qmd.md` §3's account of the embed backfill stays true after #1367.

The doc is where this item came from and where it will get re-filed from. §3
described `models: embed` as inert ("nothing re-embeds by itself") while pending
is counted per configured model, and once that was corrected (06ccbe28) the same
section described task #81's backfill as having "no ceiling" — accurate at the
time, and an open gap the moment the cap lands. An open gap in an architecture
doc is a to-do list, and a stale entry in one reads like a decision nobody made,
so the phrase is pinned out of the file here.

What replaces it is pinned the same way: the guard's name and its fraction are
asserted against `scripts/maintenance/qmd_index_maintenance.py`, so prose cannot
keep citing a constant that was renamed or re-tuned, and the doc's threshold and
the code's cannot drift apart.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "qmd.md"
SCRIPT = ROOT / "scripts" / "maintenance" / "qmd_index_maintenance.py"

#: The module constant the doc is allowed to name, and the only place the
#: threshold is defined. Read from the source text rather than imported so this
#: file does not depend on the module's import-time side effects (it stats the
#: live index at import in other contexts).
GUARD = "EMBED_PENDING_MAX_RATIO"


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def _section(heading: str) -> str:
    """From `heading` to the next `## ` line — one numbered section, sliced out."""
    text = _doc()
    i = text.index(heading)
    j = text.find("\n## ", i + len(heading))
    return text[i: j if j != -1 else len(text)]


def _reembed_bullet() -> str:
    """§3's bullet on changing the embed model — the one #1367 is about.

    Asserts its own absence, so a doc that reworded the heading fails here with a
    reason instead of at an index.
    """
    body = _section("## 3. Models")
    hits = [b for b in body.split("\n- ") if "full re-embed" in b]
    assert len(hits) == 1, f"expected exactly one 'full re-embed' bullet in §3, got {len(hits)}"
    return hits[0]


def _script_constant() -> float:
    hits = re.findall(rf"^{GUARD} = ([0-9.eE+]+)$", SCRIPT.read_text(encoding="utf-8"), re.M)
    assert len(hits) == 1, f"{SCRIPT.name}: expected exactly one {GUARD} assignment, got {hits}"
    return float(hits[0])


# --- clause 4: the doc stops describing the backfill as uncapped --------------

def test_the_doc_no_longer_describes_the_backfill_as_having_no_ceiling():
    """"no ceiling" was true on 2026-09-22 and must not survive the cap.

    Checked over the whole file, review log included: a dated entry that still
    says the job is uncapped is the sentence an architecture review re-files the
    item from, and the review log is read as a description of the job as often as
    it is read as history.
    """
    text = _doc()
    assert "no ceiling" not in text, "the doc still describes #81's backfill as uncapped"
    assert "uncapped" not in text.lower(), (
        "the backfill is described as uncapped somewhere in the doc")


def test_the_doc_names_the_guard_and_its_threshold_fraction():
    """§3 now says what refuses, by the name the code uses, with the fraction."""
    bullet = _reembed_bullet()
    assert GUARD in bullet, "§3 does not name the constant that caps the backfill"
    assert "model_change_suspected" in bullet, (
        "§3 names the cap but not the finding the run records, which is the half "
        "a reader of the report is looking for")
    assert re.search(r"0\.25|quarter", bullet), "§3 does not state the threshold fraction"


def test_the_docs_threshold_is_the_ones_the_code_carries():
    """The fraction in prose and the constant in the module are one number.

    Two places that can be edited separately is how a doc ends up describing a
    threshold nobody tuned.
    """
    code_value = _script_constant()
    assert code_value == 0.25, f"{SCRIPT.name} retuned the cap; update §3 in the same change"
    assert f"{code_value:.2f}" in _reembed_bullet() or "quarter" in _reembed_bullet()


def test_the_guard_is_attributed_to_the_job_that_runs_it_not_the_watcher():
    """The sentence that names the cap must not credit the watcher with it.

    #1367 capped task #81's backfill and deliberately not
    `agent-services/scripts/qmd-watcher.sh`, which embeds every cycle and is
    still the first responder to an embed-model edit. Prose that says otherwise
    would tell the next operator the machine is covered when it is not. A round
    that caps the watcher too rewrites this sentence and this test together.
    """
    for sentence in re.split(r"(?<=[.!?])\s+", _reembed_bullet()):
        if GUARD in sentence:
            assert "watcher" not in sentence.lower(), sentence


def test_the_doc_still_says_which_path_is_left_unprotected():
    """The residual gap is stated, because the fix is partial by decision.

    Only the mutating branch of the maintenance job embeds, and only one of the
    two unattended surfaces is capped; a doc that reads as "fixed" here is how
    the watcher's path survives the next six months.
    """
    bullet = _reembed_bullet()
    assert "watcher" in bullet.lower(), "§3 no longer says which embed path is uncapped"


def test_section_8_lists_the_test_that_pins_this_claim():
    """The doc's own test index carries this file, so the claim is findable."""
    assert "test_qmd_doc_claims.py" in _section("## 8.")


# --- the config the guard reads is the config the doc describes ---------------

def test_the_embed_model_the_live_config_template_names_still_parses_the_way_the_guard_reads_it():
    """`models: embed` is a mapping key, not a scalar under some other spelling.

    The guard's only useful field in the report is the model name, and it comes
    from `data["models"]["embed"]` in the file the daemon reads. The tracked
    template is byte-identical to that file today on this key, so it is the
    checkable stand-in — a rename upstream (a `model:` list, a `embedding:`
    nesting) would leave the guard reporting None for every run, which is the
    failure the report is supposed to make visible rather than cause.
    """
    cfg = yaml.safe_load(
        (ROOT / "agent-services" / "conf" / "qmd-index.yml").read_text(encoding="utf-8"))
    models = cfg["models"]
    assert isinstance(models, dict) and isinstance(models.get("embed"), str) and models["embed"]
