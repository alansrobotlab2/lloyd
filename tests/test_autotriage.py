"""Backlog triage: a stale item is a hypothesis that failed, and that is a win.

The property this file exists to protect is that triage cannot start work from
an unverified premise. A backlog going back to February contains items whose
premise no longer holds, and acting on those produces the worst available
outcome: a confident, tested, gated change that solves a problem nobody has.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from scripts.automod import backlog as B
from workers.sources import autotriage as M


def write_item(tmp_path, item_id, *, status="draft", days_old=100, body="Do the thing.",
               name="A thing", priority="medium", board="lloyd"):
    created = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    fm = {"status": status, "priority": priority, "created": created,
          "board": board, "tags": ["backlog"]}
    path = tmp_path / f"{item_id}-{name.lower().replace(' ', '-')}.md"
    path.write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# {name}\n\n{body}\n",
        encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def backlog_dir(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    return d


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def test_only_open_items_are_candidates(backlog_dir, tmp_path):
    write_item(backlog_dir, 1, status="done")
    write_item(backlog_dir, 2, status="draft")
    write_item(backlog_dir, 3, status="closed")
    assert [i.id for i in B.open_items()] == [2]


def test_oldest_untriaged_is_selected_first(backlog_dir, tmp_path):
    """Age is the best proxy for staleness, and finding out which old items are
    still real is the whole point of the pass."""
    write_item(backlog_dir, 10, days_old=30)
    write_item(backlog_dir, 11, days_old=200)
    write_item(backlog_dir, 12, days_old=90)
    assert B.select_candidate(tmp_path / "none.jsonl").id == 11


def test_priority_overrides_age(backlog_dir, tmp_path):
    # Inverted on 2026-09-16. Oldest-first was the rule while the board was a
    # stale pile nobody had read; after the sweep read all of it, Alan's rule
    # is that the human's `priority` tag is honoured in every pool. Age still
    # breaks ties within one priority (`tests/test_backlog_priority.py`).
    write_item(backlog_dir, 20, days_old=10, priority="high")
    write_item(backlog_dir, 21, days_old=300, priority="low")
    assert B.select_candidate(tmp_path / "none.jsonl").id == 20


def test_an_already_triaged_item_is_not_reselected(backlog_dir, tmp_path):
    write_item(backlog_dir, 30, days_old=300)
    write_item(backlog_dir, 31, days_old=200)
    ledger = tmp_path / "l.jsonl"
    ledger.write_text(json.dumps(
        {"event": "backlog_triage", "item_id": 30, "verdict": "stale"}) + "\n")
    assert B.select_candidate(ledger).id == 31


def test_nothing_left_to_triage_returns_none(backlog_dir, tmp_path):
    write_item(backlog_dir, 40)
    ledger = tmp_path / "l.jsonl"
    ledger.write_text(json.dumps(
        {"event": "backlog_triage", "item_id": 40, "verdict": "confirmed"}) + "\n")
    assert B.select_candidate(ledger) is None


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def test_a_retiring_verdict_closes_the_item(backlog_dir, tmp_path):
    path = write_item(backlog_dir, 50)
    item = B.load_item(path)
    B.record_verdict(item, "stale", "The module was deleted in abc1234.",
                     check="grep -n foo app/x.py", close=True)
    fm = yaml.safe_load(path.read_text().split("---\n")[1])
    assert fm["status"] == "done"
    assert fm["autotriage_retired"] == "stale"


def test_a_confirmed_verdict_leaves_the_item_open(backlog_dir, tmp_path):
    """Confirmed means there is real work — not that the work is finished."""
    path = write_item(backlog_dir, 51)
    item = B.load_item(path)
    B.record_verdict(item, "confirmed", "Still reproduces on HEAD.", close=True)
    fm = yaml.safe_load(path.read_text().split("---\n")[1])
    assert fm["status"] == "up_next"
    assert "autotriage_retired" not in fm


def test_the_evidence_is_always_written_not_just_the_conclusion(backlog_dir):
    """An item closed with no stated reason is indistinguishable from one closed
    by mistake, and auditability is the whole value of the pass."""
    path = write_item(backlog_dir, 52)
    item = B.load_item(path)
    B.record_verdict(item, "stale", "No caller remains; removed in abc1234.",
                     check="rg -n 'old_fn' app/", close=True)
    text = path.read_text()
    assert "No caller remains" in text
    assert "rg -n 'old_fn' app/" in text
    assert "Automod triage" in text
    fm = yaml.safe_load(text.split("---\n")[1])
    assert any("stale" in e for e in fm["activity_log"])


def test_an_unknown_verdict_is_refused(backlog_dir):
    path = write_item(backlog_dir, 53)
    item = B.load_item(path)
    with pytest.raises(ValueError):
        B.record_verdict(item, "probably_fine", "vibes")


def test_the_original_body_survives_annotation(backlog_dir):
    path = write_item(backlog_dir, 54, body="Original description worth keeping.")
    item = B.load_item(path)
    B.record_verdict(item, "confirmed", "Reproduced.")
    assert "Original description worth keeping." in path.read_text()


# ---------------------------------------------------------------------------
# Verdict-block parsing
# ---------------------------------------------------------------------------

def test_a_well_formed_block_parses():
    parsed = M.parse_verdict(
        "prose...\nVERDICT: already_done\nCHECK: pytest tests/test_x.py\n"
        "EVIDENCE: Fixed in 90aa609; the test now passes.\nACCEPTANCE: -")
    assert parsed["verdict"] == "already_done"
    assert parsed["check"] == "pytest tests/test_x.py"


def test_the_last_block_wins_if_the_model_restates_itself():
    parsed = M.parse_verdict(
        "VERDICT: confirmed\nCHECK: a\nEVIDENCE: first\nACCEPTANCE: x\n"
        "on reflection...\nVERDICT: stale\nCHECK: b\nEVIDENCE: second\nACCEPTANCE: -")
    assert parsed["verdict"] == "stale" and parsed["evidence"] == "second"


def test_an_invented_verdict_is_rejected():
    assert M.parse_verdict(
        "VERDICT: looks_fine\nCHECK: x\nEVIDENCE: y\nACCEPTANCE: -") is None


def test_no_block_at_all_is_none():
    assert M.parse_verdict("I had a good look and it seems fine really.") is None


@pytest.mark.parametrize("verdict", B.VERDICTS)
def test_every_declared_verdict_parses(verdict):
    parsed = M.parse_verdict(f"VERDICT: {verdict}\nCHECK: c\nEVIDENCE: e\nACCEPTANCE: -")
    assert parsed and parsed["verdict"] == verdict


# ---------------------------------------------------------------------------
# The property that matters most
# ---------------------------------------------------------------------------

def test_triage_never_starts_a_round():
    """Implementation must be a separate, explicit act.

    If triage could open a round, a wrong `confirmed` would send the gate after
    a problem that does not exist — the exact failure this pipeline is designed
    to prevent.
    """
    import inspect
    src = inspect.getsource(M)
    for forbidden in ("automod_start", "round.start", "R.start(", "promote("):
        assert forbidden not in src, f"triage must not call {forbidden}"


def test_the_prompt_states_that_retiring_is_a_good_outcome():
    """A pipeline that only counts code as progress turns a stale backlog into
    a pile of unnecessary changes."""
    assert "good outcome" in M.PROMPT
    assert "Never guess" in M.PROMPT
    assert "read-only" in M.PROMPT


def test_the_prompt_demands_evidence_and_an_acceptance_check():
    assert "Quote your evidence" in M.PROMPT
    assert "ACCEPTANCE:" in M.PROMPT


# ---------------------------------------------------------------------------
# #1738: the prompt must bound the scan, because nothing else does
#
# Before-count, from the 2026-09-28 signal report's shape-gated scan (408 session
# files, mtime >= 2026-09-27T05:00:00Z): 27 payload-gated `timed out after`
# results, 26 naming 120000 ms, 15 of them autotriage — the job on top. Triage's
# wider re-count (every unbounded Bash call, 655 files in the same window) got 58,
# 26 of them autotriage, and 19 payload-gated with 9 autotriage in the 4.5 hours
# after filing. "Payload-gated" means the call carried neither `timeout` nor
# `run_in_background`, so it died on `agent_mcp/builtin_bash.py:53`
# DEFAULT_TIMEOUT_MS = 120_000.
#
# The prompt is the only surface that reaches this job. `build_skill_prompt`
# (`workers/sources/_common.py:62`) is called only by `deep_research.py:425` and
# `youtube_digest.py:336`, so a SKILL.md edit under `skills/backlog-premise-triage/`
# is a placebo; and `app/prompt_builder.py:471-479` already injects a generic
# "pass run_in_background=true for any long-running command" into every session
# including these, which plainly is not enough — it never names the ceiling the
# command is crossing, so nothing tells the model that a repo-wide grep is on the
# far side of it.
# ---------------------------------------------------------------------------

CEILING_HEAD = "- **Bound a scan that can outlive the Bash call's 120 s ceiling.**"


def _ceiling_rule(text: str) -> str:
    """The Bash-ceiling bullet of `text`, running to the next bullet.

    Reading the bullet rather than the whole prompt is what makes the assertions
    about a *bound* mean something: a `600000` elsewhere in a 13,682-character
    prompt would be coincidence, and a bound stated in the guidance is not.
    """
    assert CEILING_HEAD in text, (
        "the prompt carries no Bash-ceiling budget line at all")
    start = text.index(CEILING_HEAD)
    tail = text[start + len(CEILING_HEAD):]
    end = tail.find("\n- ")
    return text[start:start + len(CEILING_HEAD) + (len(tail) if end == -1 else end)]


def test_the_prompt_names_the_bash_ceiling_by_number_and_its_remedy():
    """Clause 1: the number is the missing half, not the existence of backgrounding.

    `app/prompt_builder.py:471-479` has told every session about
    `run_in_background=true` all along and 27 calls still died in-window, so a
    line that only repeated it would have been the same placebo in a different
    file. The remedy has to be consumed too, which is what `output_file` is for —
    a backgrounded scan nobody reads back is a scan that did not happen.
    """
    rule = _ceiling_rule(M.PROMPT)
    assert "120 s" in rule, "the ceiling has to be named in the unit the model plans in"
    assert "120000 ms" in rule, "and in the unit the error message and the arg use"
    assert "run_in_background=true" in rule
    assert "output_file" in rule, "the remedy has to say where the answer turns up"
    assert "Read" in rule, "and that reading it back is part of it"


def test_the_ceiling_rule_survives_rendering_even_over_a_truncated_body(backlog_dir, ledger):
    """Clause 2: the guidance reaches the model, not just the module constant.

    The body is truncated to `body_chars` (default 30,000) by `render_prompt`, so a
    rule that lived inside the candidate's text would be cut here; rendering an
    item whose body runs to 40,000 characters with `body_chars=200` is what proves
    the line is part of the template every triage turn gets. The two strings are
    compared byte for byte, so a formatting difference — a stray line continuation
    lost through `.format`, say — cannot slip through as a passing substring test.
    """
    write_item(backlog_dir, 1738, body="Do it.\n\n" + ("x" * 40_000), name="Long item")
    text = M.render_prompt(B.item_by_id(1738), ledger=ledger, body_chars=200)
    assert _ceiling_rule(text) == _ceiling_rule(M.PROMPT)


def test_the_prompt_bounds_the_scan_and_never_answers_a_ceiling_with_a_bigger_one():
    """Clause 3: option (b) — a smarter platform default — is out of scope, and a
    prompt that told the model to raise `timeout` would have shipped it anyway.

    Asserted as the set of millisecond figures the rule contains: exactly the
    120000 default it is warning about. Any 6-digit ms bound added as guidance
    ("or pass `timeout: 600000`", the tool's own cap) makes that set disagree, and
    the second half keeps the figure out of the rest of the prompt too.
    """
    rule = _ceiling_rule(M.PROMPT)
    assert set(re.findall(r"\b\d{6,}\b", rule)) == {"120000"}, rule
    assert "600000" not in M.PROMPT
    assert "run_in_background=true" in rule, (
        "with the ceiling the only bound, the background route must be the remedy")


def test_the_ceiling_rule_names_the_two_command_shapes_that_actually_time_out():
    """Clause 4: recognition is the whole point of putting this in the prompt.

    Sampled from the 9 in-window autotriage timeouts: `grep -rn "timed out after"
    --include=*.py .` issued at `~/lloyd`, where the walk sweeps the vendored
    trees, and a `grep -l ... *.json` over the 2,590-file
    `~/lloyd-data/sessions`. A rule naming only a number and a flag is advice the
    model cannot match to the command it is about to press Return on — and the
    bullet two above this one already scopes those same greps for *correctness*
    (`git grep` instead of `grep -r`), which is why this one has to say the shapes
    are still too big to wait for even when scoped.
    """
    rule = _ceiling_rule(M.PROMPT)
    assert "grep -r" in rule, "the recursive grep shape"
    assert "`~/lloyd`" in rule, "under Lloyd's own tree"
    assert "~/lloyd-data/sessions" in rule and "*.json" in rule, "and the sessions scan"
    assert "vendored" in rule, "the vendored walk is why scoping does not fix the time"


# ---------------------------------------------------------------------------
# #1800: the sweep path gets the same ceiling bullet, out of the same constant
#
# The bullet above shipped in `PROMPT` only (1750f88d, #1738), and the owed-check
# measurement that asked for the sweep copy counted the gap: 52/52 post-landing
# single-item sessions carried the bullet, 0/110 sweep sessions did, and 0 of those
# 110 ever passed `run_in_background=true`. Sweep-attributable payload-gated
# timeouts went 2/44 -> 2/29 sessions across that landing (0.045 -> 0.069 per
# session, i.e. UP) while single-item fell 24/137 -> 2/52 (0.175 -> 0.038), which
# is why the sweep is the source to fix next and not the group path: group ran 1
# session in the same 2026-09-24 -> 2026-09-29 window, with 0 timeouts.
# ---------------------------------------------------------------------------


def test_the_sweep_prompt_carries_the_same_ceiling_and_the_same_remedy():
    """Clause 1: a sweep greps for evidence too and dies on the same default.

    Asserted through `_ceiling_rule`, the extractor the single-item tests use, so
    the sweep copy is held to the same content as the single-item one: the ceiling
    in both units, and backgrounding-plus-reading-back as the only remedy.
    """
    rule = _ceiling_rule(M.SWEEP_PROMPT)
    assert "120 s" in rule, "the unit the model plans in"
    assert "120000 ms" in rule, "and the unit the error message and the arg use"
    assert "run_in_background=true" in rule
    assert "output_file" in rule and "Read" in rule, (
        "the remedy is to background the scan and read back what it wrote")


def test_the_sweep_prompt_names_the_shapes_that_cross_the_ceiling_in_a_sweep():
    """Clause 2: a sweep's own commands, not the single-item pass's.

    The two shapes measured in the sweep's own timeouts: a chained evidence
    command (`Read` the item or vault note, then grep under `~/lloyd` in the same
    call), and an over-many-files scan of a corpus thousands of files deep.
    Asserted against the whole prompt because they sit in the sweep's own sentence,
    which follows the shared bullet — and the last assertion keeps that sentence out
    of the shared text, so it cannot arrive in `PROMPT` by the back door.
    """
    for probe in ("chained evidence command", "over-many-files scan", "grep -rn",
                  "vault note", "~/lloyd", "~/obsidian/backlog",
                  "~/lloyd-data/sessions"):
        assert probe in M.SWEEP_PROMPT, f"sweep prompt names no {probe!r} shape"
    assert "over-many-files" not in _ceiling_rule(M.SWEEP_PROMPT), (
        "the shapes are the sweep's own sentence, not part of the shared bullet")


def test_the_ceiling_bullet_is_one_constant_both_prompts_share_verbatim():
    """Clause 3: the two copies could not be edited apart even by accident.

    `src.count(CEILING_HEAD) == 1` is the source-level half of the acceptance check
    (`git grep -n "Bound a scan"` showing one hit); the equalities are the rendered
    half — each prompt carries the constant byte for byte, so editing one copy is
    editing both, which is the drift a second hand-copied bullet would reopen.
    """
    import inspect
    src = inspect.getsource(M)
    assert src.count(CEILING_HEAD) == 1, (
        "the ceiling text exists more than once in the source: someone hand-copied "
        "it instead of reusing BASH_CEILING_BULLET")
    assert M.BASH_CEILING_BULLET in M.PROMPT
    assert M.BASH_CEILING_BULLET in M.SWEEP_PROMPT
    assert _ceiling_rule(M.PROMPT) == M.BASH_CEILING_BULLET
    assert _ceiling_rule(M.SWEEP_PROMPT) == M.BASH_CEILING_BULLET


def test_the_sweep_prompt_still_ends_with_its_verdict_block_after_the_bullet():
    """Clause 4: guidance landing after "nothing after it" is guidance ignored.

    `_SWEEP_LINE`/`_SWEEP_WORTH`/`_SWEEP_SIZE` parse the `SWEEP_VERDICTS:` block,
    and the template is what the sweep is told to finish on, so the bullet has to
    precede both — inserted between the steps and the `Rules:` paragraph.
    """
    bullet_at = M.SWEEP_PROMPT.index(M.BASH_CEILING_BULLET)
    assert bullet_at < M.SWEEP_PROMPT.index("SWEEP_VERDICTS:")
    assert bullet_at < M.SWEEP_PROMPT.index(
        "Finish with exactly this block and nothing after it:")
    assert M.SWEEP_PROMPT.rstrip().endswith("(one line per item; every item listed)")


def test_a_two_item_sweep_render_carries_the_bullet_and_no_bigger_bound(backlog_dir):
    """Clause 5, first half: the guidance reaches the sweep turn, not just the module.

    Rendered the way `_execute_sweep` renders it — `SWEEP_PROMPT.format` over
    `_render_cluster` (`workers/sources/autotriage.py:1384`) — for a two-item batch.
    The millisecond set is #1738's guard carried over: exactly the 120000 default the
    bullet warns about, so no bigger bound entered the sweep text either, and the
    regex keeps an instruction like `timeout: 600000` out of it in prose form too.
    """
    write_item(backlog_dir, 1800, name="Sweep ceiling", body="Bound the scan.")
    write_item(backlog_dir, 1801, name="Another item", body="An unrelated claim.")
    members = [B.item_by_id(1800), B.item_by_id(1801)]
    assert [m.id for m in members] == [1800, 1801], "the two fixtures did not land"
    rendered = M.SWEEP_PROMPT.format(
        n=2, batch_id=B.sweep_batch_id([m.id for m in members]),
        items=M._render_cluster(members, 1500))
    assert M.BASH_CEILING_BULLET in rendered, "the bullet did not survive .format"
    assert set(re.findall(r"\b\d{6,}\b", rendered)) == {"120000"}, rendered
    assert not re.search(r"timeout\s*[:=]\s*[\"']?\d", rendered), (
        "the prompt must not answer a ceiling by asking for a bigger one")
    assert "600000" not in rendered
    assert rendered.rstrip().endswith("(one line per item; every item listed)"), (
        "the rendered prompt no longer closes on the verdict template")


def test_the_group_prompt_stays_without_the_bullet_and_the_ruling_is_in_code():
    """Clause 5, second half: near-zero traffic, so the omission is a decision.

    The group path ran 1 session in the 2026-09-24 -> 2026-09-29 window with 0
    payload-gated timeouts, so there is nothing for that prompt to prevent. The
    ruling is asserted as text in the module, above the literal, because an
    unexplained absence reads as an oversight and gets "fixed" by whichever session
    notices next — spending prompt tokens on a path with no observed failure.
    """
    import inspect
    src = inspect.getsource(M)
    assert CEILING_HEAD not in M.GROUP_PROMPT
    assert M.BASH_CEILING_BULLET not in M.GROUP_PROMPT
    assert "run_in_background" not in M.GROUP_PROMPT and "120000" not in M.GROUP_PROMPT
    note_start = src.index("# Deliberately WITHOUT the Bash-ceiling bullet")
    assert note_start < src.index('GROUP_PROMPT = """'), "the ruling must sit above the literal"
    note = src[note_start:src.index('GROUP_PROMPT = """')]
    assert "0 payload-gated timeouts" in note, "the note must carry the measurement"
    assert "BASH_CEILING_BULLET" in note, (
        "the note must say how to add it later: reuse the constant, not a copy")

def test_summarize_counts_retirements_separately(backlog_dir, tmp_path):
    for i in (60, 61, 62):
        write_item(backlog_dir, i)
    ledger = tmp_path / "l.jsonl"
    ledger.write_text("\n".join(json.dumps(
        {"event": "backlog_triage", "item_id": i, "verdict": v})
        for i, v in ((60, "stale"), (61, "already_done"), (62, "confirmed"))) + "\n")
    s = B.summarize(ledger)
    assert s["retired"] == 2 and s["confirmed"] == 1


# ---------------------------------------------------------------------------
# Board scoping
#
# The backlog is shared. Of 53 open items, 3 are Alfie (robot firmware) and 1
# sits on an Architecture board — legitimately out of scope for a
# self-modification pass. The `board` field says so for free, and spending an
# LLM turn per item to rediscover it is waste: verified against #38 "Alfie —
# Fix mecanum wheels behavior", where a full triage turn correctly concluded
# `not_code` from something the frontmatter already knew.
# ---------------------------------------------------------------------------

def test_only_the_lloyd_board_is_in_scope_by_default(backlog_dir, tmp_path):
    write_item(backlog_dir, 70, board="lloyd", days_old=100)
    write_item(backlog_dir, 71, board="alfie", days_old=300)
    write_item(backlog_dir, 72, board="Architecture", days_old=250)
    assert [i.id for i in B.open_items()] == [70]


def test_an_older_out_of_scope_item_does_not_get_selected(backlog_dir, tmp_path):
    """Oldest-first must not drag in another board's work."""
    write_item(backlog_dir, 80, board="lloyd", days_old=50)
    write_item(backlog_dir, 81, board="alfie", days_old=400)
    assert B.select_candidate(tmp_path / "none.jsonl").id == 80


def test_boards_none_means_everything(backlog_dir):
    write_item(backlog_dir, 90, board="lloyd")
    write_item(backlog_dir, 91, board="alfie")
    assert len(B.open_items(None)) == 2


def test_board_matching_is_case_insensitive(backlog_dir):
    write_item(backlog_dir, 100, board="Lloyd")
    assert [i.id for i in B.open_items()] == [100]


def test_an_item_with_no_board_is_out_of_scope(backlog_dir):
    """Absent board is not the same as Lloyd's board — say so explicitly."""
    write_item(backlog_dir, 110, board="")
    assert B.open_items() == []
    assert len(B.open_items(None)) == 1


def test_the_summary_names_the_boards_it_counted(backlog_dir, tmp_path):
    write_item(backlog_dir, 120, board="lloyd")
    write_item(backlog_dir, 121, board="alfie")
    s = B.summarize(tmp_path / "none.jsonl")
    assert s["boards"] == ["lloyd"] and s["open_items"] == 1


# ---------------------------------------------------------------------------
# Provenance: the prompt says where the item came from
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402

from scripts.automod import state as S  # noqa: E402
from workers.sources import _common as C  # noqa: E402


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(S, "LEDGER_PATH", path)
    return path


def _ev(ledger, **row):
    S.append_event(row, path=ledger)


def test_the_origin_block_names_who_filed_it_the_keep_prior_filings_and_findings(backlog_dir, ledger):
    write_item(backlog_dir, 7, status="done", name="Parent")
    body = "Do it.\n\n## Findings (round A)\n\n- x\n\n## Findings (triage B)\n\n- y\n"
    write_item(backlog_dir, 42, body=body, name="Child")
    _ev(ledger, event="backlog_triage", item_id=7, verdict="stale", spawned=[42], merged=[])
    _ev(ledger, event="backlog_group_triage", cluster_id="c-1", judged={"42": "keep", "43": "fold"})
    _ev(ledger, event="backlog_triage", item_id=42, verdict="incomplete", spawned=[601], merged=[])
    item = B.item_by_id(42)
    text = M.render_prompt(item, ledger=ledger)
    origin = text[text.index("<origin"):text.index("</origin>")]
    assert "filed by triage of #7 (stale)" in origin
    assert 'parent="#7 (done)"' in origin
    assert "group triage c-1 judged it `keep` on" in origin
    assert "earlier triages of this item filed or appended to: #601" in origin
    assert "2 Findings section(s)" in origin
    assert 'tags="backlog"' in origin


def test_a_human_item_says_so(backlog_dir, ledger):
    write_item(backlog_dir, 5)
    text = M.render_prompt(B.item_by_id(5), ledger=ledger)
    assert "a human's item, or a writer outside the loop" in text


def test_spawn_origin_reads_all_three_producers(ledger):
    _ev(ledger, event="backlog_implement", item_id=3, phase="finished", spawned=[30])
    _ev(ledger, event="arch_review", unit="doc:automod", verdict="stale", filed=[31])
    assert B.spawn_origin(ledger, 30)["by"] == "autocode"
    assert B.spawn_origin(ledger, 31) == {"by": "arch-review", "parent": "doc:automod",
                                          "ts": B.spawn_origin(ledger, 31)["ts"], "verdict": "stale"}
    assert B.spawn_origin(ledger, 99) is None


def test_prior_triage_spawned_keeps_ledger_order_and_skips_the_item_itself(ledger):
    _ev(ledger, event="backlog_triage", item_id=9, verdict="incomplete", spawned=[602, 9], merged=[])
    _ev(ledger, event="backlog_triage", item_id=9, verdict="stale", spawned=[601], merged=[602, 400])
    assert B.prior_triage_spawned(ledger, 9) == [602, 601, 400]


# ---------------------------------------------------------------------------
# The implement-pool depth gate
# ---------------------------------------------------------------------------

def _ready(backlog_dir, ledger, n, start=1000, **kw):
    for i in range(start, start + n):
        write_item(backlog_dir, i, status="up_next", name=f"ready {i}")
        _ev(ledger, event="backlog_triage", item_id=i, verdict="confirmed",
            acceptance=kw.get("acceptance", "it passes"))


class _QItem:
    def __init__(self, payload=None):
        self.payload = payload or {}


def _stub_turn(monkeypatch, verdict="unverifiable", acceptance="none"):
    async def turn(prompt, **kw):
        turn.calls.append(prompt)
        return {"text": f"VERDICT: {verdict}\nSURFACE: code\nCHECK: grep -n x app.py\nEVIDENCE: e\n"
                        f"ACCEPTANCE: {acceptance}\nSPAWNED: none\n",
                "session_id": "s", "stop_reason": "stop", "num_turns": 3, "errors": []}
    turn.calls = []
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    return turn


def test_with_holding_off_a_full_pool_pauses_single_triage_and_says_both_numbers(backlog_dir, ledger, monkeypatch):
    """The kill switch restores the 2026-09-13 gate exactly."""
    _ready(backlog_dir, ledger, 40)
    write_item(backlog_dir, 7, days_old=300)
    turn = _stub_turn(monkeypatch)
    out = asyncio.run(M.execute(_QItem({"group_triage": False, "implement_pool_floor": 40,
                                        "hold_confirmations": False})))
    assert out["status"] == "skipped"
    assert "single-item triage paused: 40 ready in up_next ≥ bound 40" in out["summary"]
    assert "0 items landed in 7 d, floor 40" in out["summary"]
    assert turn.calls == []
    assert [e for e in S.read_events(path=ledger) if e.get("item_id") == 7] == [], \
        "a skipped run is not a triage"
    assert B.select_candidate(ledger).id == 7, "the draft is still a candidate"


def test_items_autocode_would_not_take_do_not_fill_the_pool(backlog_dir, ledger, monkeypatch):
    _ready(backlog_dir, ledger, 39)
    _ready(backlog_dir, ledger, 3, start=2000, acceptance="human-only: config.yaml")
    for i in (3000, 3001):     # grouped members parked in up_next
        p = write_item(backlog_dir, i, status="up_next")
        B.update_frontmatter(p, {"group": 1})
        _ev(ledger, event="backlog_triage", item_id=i, verdict="confirmed", acceptance="x")
    write_item(backlog_dir, 7, days_old=300)
    turn = _stub_turn(monkeypatch)
    out = asyncio.run(M.execute(_QItem({"group_triage": False, "implement_pool_floor": 40})))
    assert out["status"] == "success" and out["item_id"] == 7
    assert len(turn.calls) == 1


def test_the_bound_is_the_floor_until_landings_exceed_it(ledger):
    now = datetime.now(timezone.utc).timestamp()
    for i, item in enumerate((1, 2, 3)):
        _ev(ledger, event="vault_land", ok=True, item_id=item, commit=f"c{i}")
    assert B.implement_pool_bound(ledger, floor=20, now=now + 1) == {
        "bound": 20, "floor": 20, "landed_items_7d": 3}
    assert B.implement_pool_bound(ledger, floor=2, now=now + 1)["bound"] == 3


def test_landed_items_are_distinct_items_not_rounds(ledger):
    """#487 landed several times in three days; a pool sized by rows is three
    times too deep."""
    for commit in ("a", "b", "c"):
        _ev(ledger, event="vault_land", ok=True, item_id=487, commit=commit)
    _ev(ledger, event="vault_land", ok=False, item_id=488, commit="d")
    _ev(ledger, event="backlog_implement", item_id=9, phase="finished", round_id="SM_1")
    _ev(ledger, event="promoted", round_id="SM_1", commit="sha1")
    _ev(ledger, event="settled", commit="sha1")
    assert B.landed_items_trailing(ledger, 7) == 2
    later = datetime.now(timezone.utc).timestamp() + 8 * 86400
    assert B.landed_items_trailing(ledger, 7, now=later) == 0


def test_group_triage_still_runs_under_a_full_pool(backlog_dir, ledger, monkeypatch):
    from scripts.automod import cluster as CL
    _ready(backlog_dir, ledger, 40)
    monkeypatch.setattr(B, "select_cluster", lambda *a, **k: ({"id": "c-9"}, ["m"]))
    monkeypatch.setattr(CL, "load_clusters", lambda *a, **k: {"clusters": []})

    async def group(item, cluster, members):
        return {"status": "success", "cluster_id": cluster["id"]}
    monkeypatch.setattr(M, "_execute_group", group)
    out = asyncio.run(M.execute(_QItem({"group_triage": True, "implement_pool_floor": 20})))
    assert out == {"status": "success", "cluster_id": "c-9"}


def test_the_floor_rides_in_the_payload():
    import inspect
    src = inspect.getsource(M.enqueue_if_due)
    assert '"implement_pool_floor"' in src and '"spawn_cap"' in src


# ---------------------------------------------------------------------------
# Held confirmations: the gate holds a verdict instead of pausing the pass
# ---------------------------------------------------------------------------

def _held(backlog_dir, ledger, iid, **kw):
    """An item triage confirmed into a full pool, exactly as `execute` leaves it."""
    path = write_item(backlog_dir, iid, name=f"held {iid}", **kw)
    B.record_verdict(B.load_item(path), "confirmed", "still real", acceptance="it passes", hold=True)
    _ev(ledger, event="backlog_triage", item_id=iid, verdict="confirmed",
        acceptance="it passes", held=True)
    return path


def _status(backlog_dir, iid):
    item = B.item_by_id(iid)
    return item.status, item.tags


def test_a_full_pool_holds_a_confirmation_instead_of_pausing(backlog_dir, ledger, monkeypatch):
    """The first cut returned before the turn and stopped retirements with the
    confirmations. Now the turn runs, and only the move into up_next waits."""
    _ready(backlog_dir, ledger, 40)
    write_item(backlog_dir, 7, days_old=300)
    turn = _stub_turn(monkeypatch, verdict="confirmed", acceptance="it passes")
    out = asyncio.run(M.execute(_QItem({"group_triage": False, "implement_pool_floor": 40})))
    assert out["status"] == "success" and out["item_id"] == 7 and out["held"] is True
    assert "held: implement pool full" in out["summary"]
    assert len(turn.calls) == 1
    status, tags = _status(backlog_dir, 7)
    assert status == "draft" and B.HELD_TAG in tags
    row = [e for e in S.read_events(path=ledger) if e.get("item_id") == 7][-1]
    assert row["verdict"] == "confirmed" and row["held"] is True
    assert set(B.held_confirmations(ledger)) == {7}
    assert len(B.ready_confirmed(ledger)) == 40, "a held item does not deepen the pool"
    assert B.desired_statuses(ledger)[7][0] == "draft"
    assert B.reconcile_statuses(ledger) == [], "the reconciler leaves a held item where it is"
    assert "**Held:**" in B.item_by_id(7).path.read_text()


def test_a_full_pool_still_retires(backlog_dir, ledger, monkeypatch):
    """The reason the pass must keep running: stale and already_done were the
    loop's largest closer."""
    _ready(backlog_dir, ledger, 40)
    write_item(backlog_dir, 7, days_old=300)
    _stub_turn(monkeypatch, verdict="stale", acceptance="-")
    out = asyncio.run(M.execute(_QItem({"group_triage": False, "implement_pool_floor": 40})))
    assert out["status"] == "success" and out["closed"] is True and out["held"] is False
    assert B.item_by_id(7).status == "done"


def test_room_in_the_pool_confirms_straight_into_up_next(backlog_dir, ledger, monkeypatch):
    _ready(backlog_dir, ledger, 3)
    write_item(backlog_dir, 7, days_old=300)
    _stub_turn(monkeypatch, verdict="confirmed", acceptance="it passes")
    out = asyncio.run(M.execute(_QItem({"group_triage": False, "implement_pool_floor": 40})))
    assert out["held"] is False
    status, tags = _status(backlog_dir, 7)
    assert status == "up_next" and B.HELD_TAG not in tags
    assert B.held_confirmations(ledger) == {}


def test_a_human_only_confirmation_is_never_held(backlog_dir, ledger, monkeypatch):
    """It never enters the pool anyway, so holding it would park it twice."""
    _ready(backlog_dir, ledger, 40)
    write_item(backlog_dir, 7, days_old=300)
    _stub_turn(monkeypatch, verdict="confirmed", acceptance="human-only: config.yaml")
    out = asyncio.run(M.execute(_QItem({"group_triage": False, "implement_pool_floor": 40})))
    assert out["held"] is False and B.held_confirmations(ledger) == {}


def test_a_human_only_confirmation_stays_in_draft_owed_to_owed_check(backlog_dir, ledger,
                                                                    monkeypatch):
    """09-27 to 09-28 every one of these was orphaned: triage moved it to
    `up_next`, the reconciler moved it back, and that move's take-back erased
    the `decide` entry — a triaged draft that no pool and no job would read."""
    from scripts.automod import owed as O
    _ready(backlog_dir, ledger, 3)
    write_item(backlog_dir, 7, days_old=300)
    _stub_turn(monkeypatch, verdict="confirmed", acceptance="human-only: config.yaml")
    asyncio.run(M.execute(_QItem({"group_triage": False, "implement_pool_floor": 40})))
    assert _status(backlog_dir, 7)[0] == "draft"
    B.reconcile_statuses(ledger)
    fm, _ = B._split_frontmatter(B.item_by_id(7).path.read_text())
    assert fm["status"] == "draft"
    assert [e["kind"] for e in O.entries_of(fm)] == ["decide"], "owed-check must still see it"


def test_a_move_to_draft_takes_back_only_what_was_tagged(backlog_dir, ledger):
    """The reconciler passes the tag on every move; the owed `decide` entry goes
    only with a real take-back (the tag on the item, or a move into the pool)."""
    from scripts.automod import owed as O
    write_item(backlog_dir, 7, days_old=3, status="up_next")
    path = B.item_by_id(7).path
    O.add_owed(path, ["decide this"], kind="decide")
    assert B.set_status(7, "draft", "not for the loop", remove_tags=(B.NEEDS_HUMAN_TAG,))
    assert O.entries_of(B._split_frontmatter(path.read_text())[0]), "untagged: kept"
    assert B.set_status(7, "up_next", "back in the pool", remove_tags=(B.NEEDS_HUMAN_TAG,))
    assert not O.entries_of(B._split_frontmatter(path.read_text())[0]), "into the pool: taken back"


def test_room_releases_held_items_oldest_first_and_only_as_many_as_fit(backlog_dir, ledger):
    _ready(backlog_dir, ledger, 2)
    for iid in (10, 11, 12):
        _held(backlog_dir, ledger, iid)
    out = B.release_held_confirmations(ledger, floor=3)
    assert [r["item_id"] for r in out] == [10] and out[0]["moved"] is True
    assert "2 ready < bound 3" in out[0]["reason"]
    status, tags = _status(backlog_dir, 10)
    assert status == "up_next" and B.HELD_TAG not in tags
    assert set(B.held_confirmations(ledger)) == {11, 12}
    assert [_status(backlog_dir, i)[0] for i in (11, 12)] == ["draft", "draft"]
    assert B.release_held_confirmations(ledger, floor=3) == [], "the pool is full again"
    assert {i.id for i, _ in B.ready_confirmed(ledger)} >= {10}
    assert B.reconcile_statuses(ledger) == [], "released and held items agree with the ledger"
    moves = [e for e in S.read_events(path=ledger) if e.get("event") == "status_moved"]
    assert [(m["item_id"], m["to"]) for m in moves] == [(10, "up_next")]


def test_a_held_item_moved_by_hand_is_released_where_it_stands(backlog_dir, ledger):
    """A human moving it is the decision; the reconciler must not undo it."""
    _ready(backlog_dir, ledger, 40)
    _held(backlog_dir, ledger, 10)
    assert B.set_status(10, "up_next", "by hand")
    moved = B.reconcile_statuses(ledger)
    assert all(m["item_id"] != 10 for m in moved)
    status, tags = _status(backlog_dir, 10)
    assert status == "up_next" and B.HELD_TAG not in tags
    assert B.held_confirmations(ledger) == {}
    rel = [e for e in S.read_events(path=ledger) if e.get("event") == "backlog_confirm_released"]
    assert rel[-1]["item_id"] == 10 and rel[-1]["moved"] is False


def test_switching_holding_off_releases_everything_so_nothing_strands(backlog_dir, ledger):
    _ready(backlog_dir, ledger, 40)
    for iid in (10, 11):
        _held(backlog_dir, ledger, iid)
    out = B.release_held_confirmations(ledger, floor=20, enabled=False)
    assert [r["item_id"] for r in out] == [10, 11]
    assert [_status(backlog_dir, i)[0] for i in (10, 11)] == ["up_next", "up_next"]


def test_a_later_confirmation_that_is_not_held_supersedes_the_hold(backlog_dir, ledger):
    _held(backlog_dir, ledger, 10)
    _ev(ledger, event="backlog_triage", item_id=10, verdict="confirmed", acceptance="a")
    assert B.held_confirmations(ledger) == {}
    assert B.desired_statuses(ledger)[10][0] == "up_next"


def test_triage_releases_room_before_it_triages(backlog_dir, ledger, monkeypatch):
    """A confirmation held on an earlier run enters the pool before this run
    adds another behind it."""
    _ready(backlog_dir, ledger, 1)
    _held(backlog_dir, ledger, 10, days_old=1)
    write_item(backlog_dir, 7, days_old=300)
    _stub_turn(monkeypatch)
    asyncio.run(M.execute(_QItem({"group_triage": False, "implement_pool_floor": 5})))
    assert B.item_by_id(10).status == "up_next"


def test_holding_rides_in_the_payload():
    import inspect
    assert '"hold_confirmations"' in inspect.getsource(M.enqueue_if_due)


def test_autocode_housekeeping_releases_with_triages_floor(monkeypatch, tmp_path):
    """The path that still works with triage switched off."""
    from workers.sources import autocode as A
    seen = {}

    def release(ledger, **kw):
        seen.update(kw)
        return []
    monkeypatch.setattr(B, "release_held_confirmations", release)
    monkeypatch.setattr(A, "_source_cfg", lambda name: {"implement_pool_floor": 7,
                                                        "hold_confirmations": False}
                        if name == "autotriage" else {})
    for name in ("reap_abandoned_rounds",):
        monkeypatch.setattr(A, name, lambda *a, **k: [])
    monkeypatch.setattr(B, "close_settled_items", lambda *a, **k: [])
    monkeypatch.setattr(B, "unfold_spent_umbrellas", lambda *a, **k: [])
    monkeypatch.setattr(B, "retriage_spent_items", lambda *a, **k: [])
    monkeypatch.setattr(B, "reconcile_statuses", lambda *a, **k: [])
    monkeypatch.setattr(B, "expire_stale_spawns", lambda *a, **k: [])
    A._housekeeping({})
    assert seen == {"floor": 7, "enabled": False}


# ── a high item is picked up next (2026-09-16) ──────────────────────────────
#
# Alan: "I want to be able to submit a high priority backlog item and have it
# be picked up next." The tag has to beat every rule about the pile: the
# sweep batch, the cluster, the depth gate's pause and its hold.

def test_a_high_draft_is_confirmed_straight_into_up_next_over_a_full_pool(backlog_dir, ledger, monkeypatch):
    _ready(backlog_dir, ledger, 40)
    write_item(backlog_dir, 7, days_old=300)                     # medium, older
    write_item(backlog_dir, 8, days_old=1, priority="high", name="Urgent thing")
    turn = _stub_turn(monkeypatch, verdict="confirmed", acceptance="it passes")
    out = asyncio.run(M.execute(_QItem({"group_triage": False, "implement_pool_floor": 40})))
    assert out["status"] == "success" and out["item_id"] == 8 and out["held"] is False
    assert len(turn.calls) == 1 and 'priority="high"' in turn.calls[0]
    status, tags = _status(backlog_dir, 8)
    assert status == "up_next" and B.HELD_TAG not in tags
    assert B.held_confirmations(ledger) == {}
    picked, _ = B.select_confirmed(ledger)
    assert picked.id == 8, "and the next round takes it ahead of the 40 already ready"


def test_a_high_draft_runs_before_a_sweep_batch_and_before_a_cluster(backlog_dir, ledger, monkeypatch):
    from scripts.automod import cluster as CL
    monkeypatch.setattr(B, "sweep_enabled", lambda: True)
    for i in (1, 2, 3):
        write_item(backlog_dir, i, days_old=200, name=f"Low {i}", priority="low")
    write_item(backlog_dir, 8, days_old=1, priority="high", name="Urgent thing")
    monkeypatch.setattr(CL, "load_clusters", lambda *a, **k: {"clusters": [
        {"id": "c1", "item_ids": [1, 2, 3], "duplicates": []}]})
    ran = {}

    async def fake_sweep(item, members):
        ran["sweep"] = [m.id for m in members]
        return {"status": "ok", "summary": "sweep"}

    async def fake_group(item, cluster, members):
        ran["group"] = [m.id for m in members]
        return {"status": "ok", "summary": "group"}
    monkeypatch.setattr(M, "_execute_sweep", fake_sweep)
    monkeypatch.setattr(M, "_execute_group", fake_group)
    turn = _stub_turn(monkeypatch, verdict="confirmed", acceptance="it passes")
    payload = {"group_triage": True, "sweep": True, "sweep_batch": 8, "implement_pool_floor": 40,
               "group_min_items": 2, "group_max_items": 4}
    out = asyncio.run(M.execute(_QItem(payload)))
    assert out["status"] == "success" and out["item_id"] == 8 and not ran, \
        "the high draft went first; neither the sweep nor the cluster ran"
    assert [i.id for i in B.sweep_pool(ledger)] == [1, 2, 3], "the sweep never reads a high item"
    # With the high one confirmed, the sweep resumes as before.
    out = asyncio.run(M.execute(_QItem(payload)))
    assert out["summary"] == "sweep" and ran["sweep"] == [1, 2, 3]
    assert len(turn.calls) == 1


def test_a_high_draft_is_triaged_even_when_holding_is_off_and_the_pool_is_full(backlog_dir, ledger, monkeypatch):
    _ready(backlog_dir, ledger, 40)
    write_item(backlog_dir, 8, days_old=1, priority="high", name="Urgent thing")
    turn = _stub_turn(monkeypatch, verdict="confirmed", acceptance="it passes")
    out = asyncio.run(M.execute(_QItem({"group_triage": False, "implement_pool_floor": 40,
                                        "hold_confirmations": False})))
    assert out["status"] == "success" and out["item_id"] == 8 and len(turn.calls) == 1
    assert _status(backlog_dir, 8)[0] == "up_next"
