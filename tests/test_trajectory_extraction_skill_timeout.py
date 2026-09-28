r"""Backlog #1728 — the skill must not price the extractor at the autonomy run's clock.

Why this file exists
--------------------
`skills/trajectory-extraction/SKILL.md` Guardrails told the next worker to
"Budget **at least 300 s** for the run; it is not a sub-minute job", quoted a
"**219 s**" peak, said "a finish under 60 s happens on a quiet day, not by
design", and closed with "treat 219 s as a floor for the new shape". Every one
of those figures is the **autonomy run wall clock** — `duration_seconds` of a
whole agent turn, LLM turns and tool round-trips included — and not the cost of
`scripts/extract-trajectories.py`. Measured with `time` at the prescribed
window on 2026-09-28, the script is `real 0m8.478s` (1,778 session files
selected, 1,515 parsed, 263 tool-less skipped, 0 failed; 8.3 s and 9.5 s on two
earlier runs the same day). So the file the extraction worker actually reads
described its own healthy 8-second run as anomalous, and any future re-sizing
of `timeout_seconds` started from the wrong quantity. The same conflation was
the stated basis of task #56's 900 s cap ("900 s = 4.1 × the 219 s peak",
#1523) — and a turn count is meaningless for a subprocess, which is what
`run_56_20260928_080127` reporting `duration_seconds: 271.5` with
`num_turns: 23` proves.

The cap itself was never the problem: 900 s is headroom over a 271.5 s turn
peak, and the historic `timed out after 300s` lines are turn timeouts too. What
had to change was the basis and the expectation, so that is what these tests
pin. The prose landed on the vault's main at `b273f12c` (plus `cc52d33d`,
which names the 219 s quantity in its own sentence); this module is the
enforcement half.

What each test holds
--------------------
1. Guardrails stop calling a fast extraction anomalous (clause 1).
2. Guardrails state a measured script wall clock and cite the command that
   measures it, so the number is reproducible rather than remembered (clause 2).
3. Guardrails label 219 s and 271.5 s as the run wall clock and say the task's
   900 s timeout bounds that quantity, not the script (clause 3).
4. Task #56's derivation names which quantity 219 s is, quotes the current run
   peak, and its front matter keeps `timeout_seconds: 900` (clause 4).

The two checks that read prose go through the readers that actually consume it:
`agent_mcp.skills._load_skill` (whose `body` is what prefetch injects into a
run) and `app.autonomy._parse_task_file` (whose `timeout_seconds` the scheduler
clamps and runs against). `OLD_GUARDRAILS` / `OLD_DERIVATION` are the pre-change
text verbatim, from vault `80821277`, and the last two tests push the real pages
through them so no assertion here can pass by being unfalsifiable.

Measured denominators, so a later reader can tell a real 0 from an empty
pattern: `grep -cE "sub-minute|quiet day"` over the SKILL.md went 1 → 0;
`grep -c "at least 300"` 1 → 0; `grep -c "floor"` 1 → 0; the proving command
`grep -c 'time \.venvs/lloyd/bin/python scripts/extract-trajectories'` 0 → 1;
`grep -c "271\.5"` in the task file 0 → 2. The run id `run_56_20260928_080127`
was already present in the task file before this change (once, in the activity
log), which is exactly why the derivation check is scoped to the
`## Timeout budget` section: a whole-file grep would pass on a log line and pin
nothing.

These assertions read the live vault, so they can go red from a nightly skills
pass rather than from the change under review. That is deliberate and is the
same trade `tests/test_brief_triage_clock_skill.py`,
`tests/test_skill_tool_names.py` and `tests/test_yaml_fix_skill_claims.py`
make: the gate runner hardcodes `-m "not live_vault"`, so a marked check is
deselected from the run meant to enforce it and pins nothing. Nothing here
skips, and a missing vault fails the read rather than passing an absence.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

VAULT = Path.home() / "obsidian"
SKILL = VAULT / "skills" / "trajectory-extraction" / "SKILL.md"
TASK = VAULT / "autonomy" / "56-nightly-trajectory-extraction.md"

#: The command #1728 requires the skill to cite, so the script figure can be
#: re-measured instead of remembered. Cited verbatim, not paraphrased.
PROVING_COMMAND = ("time .venvs/lloyd/bin/python scripts/extract-trajectories.py"
                   " --agent all --days 3")

#: Current run peak: `duration_seconds: 271.5`, `num_turns: 23`.
RUN_PEAK_ID = "run_56_20260928_080127"
RUN_PEAK_SECONDS = "271.5"

#: Stale framings, phrase by phrase. "at least 300 s" is the instruction that
#: told a worker to reserve minutes for an 8-second script.
STALE_FRAMINGS = ("sub-minute", "quiet day", "at least 300 s")

SECONDS_FIGURE = re.compile(r"(\d+(?:\.\d+)?)\s*s\b")
RUN_CLOCK_LABEL = re.compile(r"run wall clock", re.I)
TURN_EVIDENCE = re.compile(r"agent[- ]turn|LLM turns|duration_seconds", re.I)

# ── the pre-change prose, verbatim from vault 80821277 ───────────────────────

OLD_GUARDRAILS = """
- This is data collection only — do not author skills,do not analyze errors
- Budget **at least 300 s** for the run; it is not a sub-minute job. The wall clock scales with the window, and measured successful runs peak at **219 s** (2026-09-18) with 215 s and 212 s either side of it — a finish under 60 s happens on a quiet day, not by design. Task #56 declares `timeout_seconds: 900` as of 2026-09-26; the round 300 s cap it replaces had already been blown five times — four at 300 s, one at the older 120 s cap (see the derivation section in `autonomy/56-nightly-trajectory-extraction.md`).
- The widening to three days is **1.25×**, not a big one, and it fits the cap above: measured 2026-09-27 with the same `mtime >= now - days` test the script applies, the two-day window selected 1,263 session files and the three-day window selects 1,582. Every timing above was measured at the shorter window, so treat 219 s as a floor for the new shape rather than the answer — the first two cycles after a widening are the ones to watch for a timeout line in the run record.
- If the script fails,log the error and exit — do not retry within this run
""".strip()

OLD_DERIVATION = """
The front-matter `timeout_seconds` was raised on 2026-09-26 to **900** — derived, not rounded.
The cap it replaces was the round 300. The basis below comes from this file's own activity log, so a later reviewer can re-derive the number without re-running anything:

- **Measured basis — highest successful run: 219 s.** `Run run_56_20260918_082207`, 2026-09-18; next 215 s (2026-09-05) and 212 s (2026-09-25). Against the old cap that peak was 73 % of the wall clock.
- **Five timeouts in this log**: four at the 300 s cap (2026-09-04 ×3 — the third one `DISABLED after 3 consecutive failures` — plus 2026-09-05) and one at the older 120 s cap (2026-09-03).
- **900 s = 4.1 × the 219 s peak** — the same 4× margin applied to #57, which is the headroom the job's own growing-corpus design assumes.
- **Ceiling, verified at `~/lloyd` HEAD on 2026-09-26, so no `config.yaml` change is needed.** The `scheduled-task` worker source declares a 3600 s budget (`config.yaml:2116`, block `scheduled-task:` at `config.yaml:2102`); `workers/sources/scheduled_task.py:549` reads it and passes it as `max_duration=` into `run_task` at `:553`; `autonomy.py:3153` then clamps the declared value with `timeout = max(60, min(declared_timeout, int(max_duration) - _POOL_TIMEOUT_MARGIN))`, the margin being 30 s (`autonomy.py:398`). Effective cap = **3570 s**, so 900 s here and 1200 s in #57 are honoured exactly as declared.
- **Regression check — run it over activity-log lines only, so this prose is not counted:** `grep -cE '^- [0-9]{4}-.*timed out after' <this file>` reads **5** at land time. That count must not grow over the next two nightly runs; if it does at 900 s, the cause is a runaway loop, not sizing.
""".strip()


def _require_live_vault(path: Path) -> None:
    if not path.exists():
        pytest.fail(f"the live vault is not at {path} — these pins read the "
                    "vault, they do not skip it")


def _skill_body() -> str:
    """The skill page as the harness serves it: `agent_mcp.skills._load_skill`
    is what builds `skill["body"]`, and `body` is the text prefetch injects into
    an autonomy run. Reading that instead of the raw file is what makes this a
    check on what a run reads, not on bytes in a file."""
    _require_live_vault(SKILL)
    from agent_mcp import skills

    loaded = skills._load_skill(SKILL.parent)
    if not loaded:
        pytest.fail(f"{SKILL} no longer loads through "
                    "agent_mcp.skills._load_skill — the run would read nothing")
    return loaded["body"]


def _task_text() -> str:
    _require_live_vault(TASK)
    return TASK.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """Body of the first `## <heading…>` section, up to the next `## `.

    Returns "" rather than failing: the predicates below report a missing
    section as a problem, so a control that deletes a heading is observable
    instead of aborting the test that is trying to prove it can fail.
    """
    out: list[str] = []
    inside = False
    for line in text.splitlines():
        if line.startswith("## "):
            if inside:
                break
            inside = line[3:].strip().lower().startswith(heading.lower())
            continue
        if inside:
            out.append(line)
    return "\n".join(out).strip()


def _bullets(section: str) -> list[str]:
    """Top-level `- ` bullets with their continuation lines folded in, so a
    check runs on the sentence unit a reader reads rather than on one wrapped
    physical line."""
    out: list[str] = []
    for line in section.splitlines():
        if line.startswith("- "):
            out.append(line)
        elif out and line.strip():
            out[-1] = out[-1] + " " + line.strip()
    return out


# ── predicates: empty list means the page is conforming ──────────────────────

def _skill_problems(body: str) -> list[str]:
    problems: list[str] = []
    guardrails = _section(body, "Guardrails")
    if not guardrails:
        return ["the skill has no `## Guardrails` section for a worker to read"]
    bullets = _bullets(guardrails)

    # clause 1 — the run-clock framing is gone, and replaced, not just deleted
    for phrase in STALE_FRAMINGS:
        if phrase in body:
            problems.append(f"stale clock framing is back in the skill: {phrase!r}")
    if "normal, healthy" not in guardrails:
        problems.append("Guardrails no longer say a fast extraction is the "
                        "normal, healthy result — the correction was deleted "
                        "rather than replaced, so a sub-30 s finish reads as "
                        "anomalous again")

    # clause 2 — a measured script clock with the command that measures it
    commanding = [b for b in bullets if PROVING_COMMAND in b]
    if not commanding:
        problems.append(f"no Guardrails bullet cites the proving command "
                        f"{PROVING_COMMAND!r}, so the script figure is "
                        "remembered rather than reproducible")
    else:
        bullet = commanding[0]
        if "script" not in bullet.lower():
            problems.append("the bullet carrying the proving command never says "
                            "it is measuring the script")
        figures = [float(m) for m in SECONDS_FIGURE.findall(bullet)]
        if not figures:
            problems.append("the bullet carrying the proving command states no "
                            "seconds figure for the script")
        elif max(figures) >= 60.0:
            problems.append(
                f"the script wall clock is stated as {max(figures):g} s, which "
                "is a run-clock order of magnitude — the conflation #1728 is about")

    # clause 3 — the run figures are labelled, and the cap bounds the run
    for figure in ("219", RUN_PEAK_SECONDS):
        if figure not in guardrails:
            problems.append(f"{figure} s has dropped out of Guardrails, so the "
                            "run clock the cap is sized against is unstated")
        for bullet in [b for b in bullets if figure in b]:
            if not RUN_CLOCK_LABEL.search(bullet):
                problems.append(f"the bullet quoting {figure} s does not name it "
                                "as the autonomy run wall clock")
            if "floor" in bullet.lower():
                problems.append(f"{figure} s is called a floor again, which is "
                                "exactly the sentence that made an 8 s run look "
                                "like a silent failure")
    caps = [b for b in bullets if "900" in b]
    if not caps:
        problems.append("Guardrails never mention the task's 900 s timeout, so "
                        "the reader cannot tell what bounds what")
    elif not any("900" in b and "run" in b.lower()
                 and ("never the script" in b or "not the script" in b)
                 for b in caps):
        problems.append("the 900 s timeout is not said to bound the run wall "
                        "clock and NOT the script")
    return problems


def _derivation_problems(derivation: str) -> list[str]:
    problems: list[str] = []
    if not derivation:
        return ["task #56 has no `## Timeout budget` derivation section"]
    bullets = _bullets(derivation)

    # clause 4 — the peak is the current one, and the quantity is named
    peak = [b for b in bullets if RUN_PEAK_ID in b and RUN_PEAK_SECONDS in b]
    if not peak:
        problems.append(f"the derivation does not quote the current run peak "
                        f"{RUN_PEAK_SECONDS} s from {RUN_PEAK_ID}, so the cap is "
                        "still sized off a stale figure")
    elif not any("highest successful run" in b or "Measured basis" in b
                 for b in peak):
        problems.append(f"{RUN_PEAK_SECONDS} s is present but not presented as "
                        "the measured run peak")
    if not RUN_CLOCK_LABEL.search(derivation):
        problems.append("the derivation never names its figures as the autonomy "
                        "run wall clock")
    if not TURN_EVIDENCE.search(derivation):
        problems.append("the derivation does not say the run clock is the agent "
                        "turn (LLM turns / duration_seconds), not script time")
    for bullet in [b for b in bullets if "219" in b]:
        if not RUN_CLOCK_LABEL.search(bullet):
            problems.append("the bullet quoting 219 s does not name which "
                            "quantity it is")
        if "floor" in bullet.lower():
            problems.append("219 s is called a floor in the derivation")
    if re.search(r"4\.1\s*×\s*the 219", derivation):
        problems.append("the margin is still read off the stale 219 s peak "
                        "('4.1 × the 219'), which is the arithmetic #1728 says "
                        "was right for a turn budget but labelled wrong")
    return problems


# ── clause 1 ─────────────────────────────────────────────────────────────────

def test_guardrails_no_longer_call_a_fast_extraction_anomalous():
    """The skill must stop telling a worker that the extractor is a multi-minute
    job and that a quick finish is suspect. Zero hits, not 'every hit is
    qualified': the phrase is what the next run pastes its expectation from."""
    problems = [p for p in _skill_problems(_skill_body())
                if "stale clock framing" in p or "normal, healthy" in p]
    assert problems == [], problems


# ── clause 2 ─────────────────────────────────────────────────────────────────

def test_guardrails_cite_the_measured_script_clock_and_its_proving_command():
    """A seconds figure and the command that produces it, in the same bullet,
    with every figure in that bullet small enough to be script time."""
    problems = [p for p in _skill_problems(_skill_body())
                if "proving command" in p or "script wall clock is stated" in p
                or "measuring the script" in p or "seconds figure" in p]
    assert problems == [], problems


# ── clause 3 ─────────────────────────────────────────────────────────────────

def test_guardrails_label_the_run_clock_and_say_what_the_900_s_cap_bounds():
    """219 s and 271.5 s are the turn's, the cap bounds the turn, and neither
    figure is a floor for the script."""
    problems = [p for p in _skill_problems(_skill_body())
                if "dropped out of Guardrails" in p or "run wall clock" in p
                or "floor" in p or "900 s timeout" in p]
    assert problems == [], problems


# ── clause 4: the prose half, and the cap that must not have moved ───────────

def test_task_56_derivation_names_the_run_clock_and_the_current_run_peak():
    problems = _derivation_problems(_section(_task_text(), "Timeout budget"))
    assert problems == [], problems


def test_task_56_front_matter_still_declares_the_900_second_cap():
    """The item is explicit that the cap was adequate: only the stated basis
    changed. Read through `app.autonomy._parse_task_file`, the same parser the
    scheduler uses to clamp and enforce `timeout_seconds`, so this pins the
    value the engine runs against rather than a line in a file."""
    _require_live_vault(TASK)
    from app import autonomy

    task = autonomy._parse_task_file(TASK)
    assert task, f"{TASK} no longer parses through _parse_task_file"
    assert int(task.get("timeout_seconds") or 0) == 900, (
        f"timeout_seconds moved: {task.get('timeout_seconds')!r} — #1728 "
        "re-labels the basis and does not touch the cap")
    assert task.get("skill_name") == "trajectory-extraction", (
        "the task no longer points at the skill these clauses are about")


# ── falsifiability: the predicates must reject the prose that was here before ─

def test_the_guardrail_predicate_fires_on_the_prose_this_item_fixed():
    body = _skill_body()
    current = _section(body, "Guardrails")
    old = body.replace(current, OLD_GUARDRAILS)
    assert old != body, "the substitution found nothing to replace"
    problems = _skill_problems(old)
    assert any("stale clock framing" in p for p in problems), problems
    assert any("proving command" in p for p in problems), problems
    assert any("floor" in p for p in problems), problems
    assert any("normal, healthy" in p for p in problems), problems

    # …and each correction, removed on its own, is enough to go red.
    def _mutate(anchor: str, replacement: str) -> list[str]:
        mutated = body.replace(anchor, replacement)
        assert mutated != body, f"anchor {anchor!r} is not in the skill anymore"
        return _skill_problems(mutated)

    assert any("proving command" in p for p in
               _mutate(PROVING_COMMAND, "time <the extractor script>"))
    assert any("script wall clock is stated" in p for p in
               _mutate("i.e. **8.5 s**", "i.e. **219 s**"))
    assert any("normal, healthy" in p for p in
               _mutate("a finish under 30 s is the normal, healthy result",
                       "a finish is what it is"))
    # Every mention of the peak reverts to the stale figure: the run clock the
    # cap is sized against is no longer stated anywhere in Guardrails.
    assert any(RUN_PEAK_SECONDS in p for p in _mutate(RUN_PEAK_SECONDS, "219"))
    assert any("floor" in p for p in
               _mutate("watch for a `timed out after` line",
                       "treat 219 s as a floor for the new shape and watch for "
                       "a `timed out after` line"))
    assert any("900 s timeout" in p for p in
               _mutate("never the script", "and the script too"))


def test_the_derivation_predicate_fires_on_the_prose_this_item_fixed():
    text = _task_text()
    current = _section(text, "Timeout budget")

    def _derive(page: str) -> list[str]:
        # Re-scope after every edit: the Activity Log below this section now
        # itself contains the phrase "run wall clock" (the 09-28 land entry), so
        # grading a whole-page mutation would find a label the derivation itself
        # no longer carries.
        return _derivation_problems(_section(page, "Timeout budget"))

    old = text.replace(current, OLD_DERIVATION)
    assert old != text, "the substitution found nothing to replace"
    problems = _derive(old)
    assert any(RUN_PEAK_ID in p for p in problems), problems
    assert any("run wall clock" in p for p in problems), problems
    assert any("4.1 ×" in p for p in problems), problems

    def _mutate(anchor: str, replacement: str) -> list[str]:
        mutated = text.replace(anchor, replacement)
        assert mutated != text, f"anchor {anchor!r} is not in the task file anymore"
        return _derive(mutated)

    assert any(RUN_PEAK_ID in p for p in
               _mutate(RUN_PEAK_ID, "run_56_20260918_082207"))
    assert any("not presented as" in p for p in
               _mutate("Measured basis — highest successful run: "
                       f"{RUN_PEAK_SECONDS} s",
                       f"Some run once measured {RUN_PEAK_SECONDS} s"))
    assert any("does not name which quantity it is" in p for p in
               _mutate("an autonomy run wall clock, agent-turn time, not "
                       "script time", "the same kind of figure"))
    assert any("4.1 ×" in p for p in
               _mutate("3.3 × the 271.5 s run peak", "4.1 × the 219 s peak"))


def test_an_empty_page_is_reported_as_a_problem_not_a_pass():
    """The degenerate case a doc predicate gets wrong: with nothing to read it
    must complain, not return a clean sheet."""
    assert _skill_problems("") != []
    assert _derivation_problems("") != []
    assert _skill_problems("# nothing\n\nno guardrails here\n") != []
    assert _derivation_problems("- **Measured basis — highest successful run: "
                                "219 s.** old prose\n") != []
