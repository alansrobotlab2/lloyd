"""The architecture docs' source tables name exactly what the pool polls.

#897: `gap-fill` stayed in both docs' rosters as a live source, registered and
polling every 300 s, while it had never once run. The docs read the same for a
source that is idle and one that cannot fire, so the tables are held to the
registry: a retired source has to leave them, and a new one has to arrive —
`board-steward` had been missing from `architecture/workers.md` since it
shipped. Retired sources are §7's business in `workers-jobs.md`, which this
does not read.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

import workers.sources as sources

ROOT = Path(__file__).resolve().parent.parent
ARCH = ROOT / "architecture"

_ROW = re.compile(r"^\| `([a-z][a-z0-9-]*)` \|")


def _table_after(path: Path, heading: str) -> set[str]:
    """Source names in the first table under `heading`."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(heading))
    names: set[str] = set()
    in_table = False
    for ln in lines[start + 1:]:
        if ln.startswith("|"):
            in_table = True
            m = _ROW.match(ln)
            if m:
                names.add(m.group(1))
        elif in_table:
            break
    return names


def _registry() -> set[str]:
    return set(sources.SOURCE_REGISTRY)


def _section(heading: str) -> str:
    """One `## ` section of `workers-jobs.md`, heading line excluded.

    #1713: §2 of that doc carries a claim §6 already corrected, and the two
    sections disagree in the same file. A whole-file grep cannot tell which
    section is wrong — `42%` is a stale present-tense rate at line 97 and the
    right dated history at line 763 — so every §2 claim is graded inside §2.
    """
    text = (ARCH / "workers-jobs.md").read_text(encoding="utf-8")
    start = text.index(heading) + len(heading)
    rest = text[start:]
    end = rest.find("\n## ")
    return rest if end < 0 else rest[:end]


SEC2 = "## 2. What actually ran"
SEC6 = "## 6. Mining"
SEC7 = "## 7. Retired and renamed"


def test_gap_fill_is_retired():
    assert "gap-fill" not in _registry()
    assert not (ROOT / "workers" / "sources" / "gap_fill.py").exists()


def test_workers_jobs_roster_is_the_registry():
    assert _table_after(ARCH / "workers-jobs.md", "## 1. The roster") == _registry()


def test_workers_jobs_families_cover_the_registry():
    text = (ARCH / "workers-jobs.md").read_text(encoding="utf-8")
    block = text[text.index("| § | family |"):text.index("## 1. The roster")]
    named = set(re.findall(r"`([a-z][a-z0-9-]*)`", block))
    assert named == _registry()


def test_workers_md_source_table_is_the_registry():
    got = _table_after(ARCH / "workers.md", "| source | prio | what it does |")
    assert got == _registry()


def test_config_configures_only_registered_sources():
    """An unregistered block is inert (the pool iterates the registry), which
    is exactly why it would outlive its source unnoticed."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert set(cfg["workers"]["sources"]) <= _registry()


# ---------------------------------------------------------------- §2, #1713
#
# §2 "What actually ran" and §6 "Mining" describe the same two mining sources,
# and since #896/#1460 landed they describe them differently: §6 is past-tense
# and dated, §2 called `session-distill`'s turn budget a literal and quoted the
# 2026-09-04→09-11 failure rates in the present tense. So the doc disagreed with
# itself about whether the mining budget can be moved — and a reader of §2 alone
# would go looking for a literal that is not there. These nodes hold §2 to the
# tree (the keys it names must exist and be read) and to one dated measurement.

_MINING = ("session-distill", "bench-mine")


def _mining_bullets(sec2: str) -> list[str]:
    """§2's bullets that report a mining source's run failures."""
    bullets = [b for b in sec2.split("\n- ")
               if "`session-distill`" in b and "fail" in b]
    assert bullets, (
        "§2 no longer reports the mining sources' failure shape at all — the "
        "bullet this file pins was deleted rather than dated")
    return bullets


def _section7_retired_names() -> set[str]:
    """Names in §7's retirement table that are no longer registered."""
    sec7 = _section(SEC7)
    table = sec7[sec7.index("| in `runs` |"):]
    table = table[:table.index("\n\n")]
    named: set[str] = set()
    for ln in table.splitlines():
        if ln.startswith("| `"):
            named |= set(re.findall(r"`([a-z][a-z0-9-]*)`", ln.split("|")[1]))
    return named - _registry()


def test_section2_names_the_live_turn_budget_keys():
    """Clause 1: §2 must not call either mining budget a literal, and the
    sentence that replaces it has to name the keys that moved them."""
    sec2 = _section(SEC2)
    lowered = sec2.lower()
    assert "literal" not in lowered, (
        "§2 still calls a turn budget a literal. Both mining budgets are config "
        "since #896 (bench-mine) and #1460 (session-distill); §6 says so in the "
        "past tense, and §2 restating it in the present is the disagreement "
        "#1713 is about.")
    for src, issue in (("session-distill", "#1460"), ("bench-mine", "#896")):
        assert f"workers.sources.{src}.max_turns" in sec2, (
            f"§2 has to name the key that moves {src}'s turn budget")
        assert issue in sec2, f"§2 has to credit {issue} for it"


def test_the_turn_budget_keys_section2_names_exist_and_are_read():
    """The seam: the doc's key -> `config.yaml` -> the source that reads it.

    A doc that names a config key is making a claim about two other files. This
    is the compiler for that claim — #1713 started because §2 asserted the
    opposite of it ("still a literal") while `config.yaml` carried the key.
    """
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    for src in _MINING:
        assert cfg["workers"]["sources"][src].get("max_turns"), (
            f"workers.sources.{src}.max_turns is absent from config.yaml, so "
            f"architecture/workers-jobs.md §2 names a key that does not exist")
        body = (ROOT / "workers" / "sources" / f"{src.replace('-', '_')}.py"
                ).read_text(encoding="utf-8")
        assert 'src_cfg.get("max_turns"' in body, (
            f"{src} does not read its budget from config, so §2's sentence about "
            "it is wrong in the other direction")


def test_section2_states_no_undated_mining_failure_rate():
    """Clause 2: no 42%/82% in §2, and the rate it does carry is dated."""
    sec2 = _section(SEC2)
    for stale in ("42%", "82%"):
        assert stale not in sec2, (
            f"§2 still quotes {stale} — the 2026-09-04→09-11 mining failure "
            "rate, which reproduces in no later window (3.4% and 19.5% over the "
            "7 days to 2026-09-28) and has no rows behind it on this machine: "
            "the earliest `runs.completed_at` in `workers.db` is 2026-09-22.")


def test_section2_mining_bullet_is_attributed_to_a_window():
    """The other half of clause 2: deleting the stale rate is not enough — the
    shape it described has to be reported as a dated past, not a present tense."""
    for bullet in _mining_bullets(_section(SEC2)):
        assert re.search(r"2026-\d{2}-\d{2}", bullet), (
            "§2's mining-failure bullet states no window, so a reader cannot "
            "tell whether it is current. The 2026-09-11 rates are history (§6); "
            "§2 has to say which days it means.")
        assert not re.search(r"fails \d+% of its runs", bullet), (
            "§2 is back to reporting a mining failure rate in the present tense")


def test_section2_table_holds_no_retired_source():
    """Clause 3: `gap-fill` left the roster in #897 and stayed in §2's table.

    §2's table is re-measured per-source, so a row is a claim that the source
    ran — for a retired one it is a claim that cannot be true, and the reason
    #897 pulled `gap-fill` out of §1's roster in the first place.
    """
    rows = _table_after(ARCH / "workers-jobs.md", SEC2)
    assert rows, "§2 has no per-source table to check"
    retired = _section7_retired_names()
    assert "gap-fill" in retired, (
        "§7's retirement table no longer names `gap-fill`, so this test's "
        "control is broken — §2 losing its row must not delete the retirement")
    overlap = rows & retired
    assert not overlap, (
        f"§2's run table still carries retired source(s) {sorted(overlap)} — "
        "they are §7's business, and #897 deleted `gap-fill` from a roster for "
        "exactly this reason")


def test_section6_keeps_the_dated_history_section2_lost():
    """The control on the two nodes above: §6 must stay exactly as corrected.

    §2's bans are scoped to §2 because the same strings are correct in §6 —
    `42%`/`82%` there are attributed to the 7 days to 2026-09-11, and the
    literal that §2 wrongly still asserts is §6's past-tense history. A fix
    that broadened the ban would delete the corrected record to pass the test.
    """
    sec6 = _section(SEC6)
    assert "42%" in sec6 and "82%" in sec6, (
        "§6's dated mining failure rates are gone — they are the measurement "
        "#1713's §2 fix is measured against, not a stale claim")
    assert "started with a literal at the call site" in sec6, (
        "§6's history of the turn budget is gone")
    assert "still carries those numbers in the present tense" not in sec6, (
        "§6 still points at §2 as unfixed: #1713 re-measured §2 on 2026-09-28, "
        "so the forward reference is the stale sentence now")


def test_section2_and_section6_report_one_measurement():
    """Clause 4: §2's table has to say the date it was measured to, and it has
    to be the same window §6 quotes, or the two sections disagree again."""
    sec2_dates = re.findall(r"[Dd]ays to (\d{4}-\d{2}-\d{2})", _section(SEC2))
    assert sec2_dates, "§2 states no measurement window for its table"
    sec6_dates = re.findall(r"[Dd]ays to (\d{4}-\d{2}-\d{2})", _section(SEC6))
    assert sec6_dates, "§6 states no window; the control is broken"
    assert max(sec2_dates) == max(sec6_dates), (
        f"§2's table is measured to {max(sec2_dates)} but §6's current figures "
        f"are the 7 days to {max(sec6_dates)} — one doc, two measurements")


def stale_not_present_tense(sec2: str):
    """Guard for the deleted-rate assertion above, not a test: returns None."""
    return None


# ---------------------------------------------------------------- `autocode`'s
# Inner-Voice claim (#1689). The module docstring of `workers/sources/autocode.py`
# asserted that `automod_start` refuses a turn with no Inner Voice attached — a
# gate inert since 2026-09-24 — and §4 of this doc copied the sentence, so the
# doc was describing a refusal that does not fire. The four nodes below pin the
# source, the traceability of its retirement, the sweep for any other carrier,
# and the pointer the fix had to remove rather than move.

import workers.sources.autocode as autocode  # noqa: E402

SEC4 = "## 4. Self-modification — the loop that changes Lloyd's own code"

_REFUSAL_VERB = re.compile(r"\brefus\w*", re.I)
#: A refusal *for want of* the observer, which is the claim and the only thing
#: worth scanning for: a bare "refused" within 200 chars of the word "observer"
#: matched six unrelated sentences on the first try — the Bash sandbox's write
#: refusal, `autonomy.revert_run_writes` answering per path, the observer's own
#: bench harness — none of which says anything about `automod_start`.
_IV_REFUSAL = re.compile(
    r"refus\w*[^.!?]{0,60}?(?:with\s+no|without|for\s+want\s+of|lacking|missing)"
    r"[^.!?]{0,40}?(?:inner\s+voice|observer)", re.I)
_IV_REFUSAL_REV = re.compile(
    r"(?:\bno|\bwithout|lacking)\s+(?:an?\s+)?(?:inner\s+voice|observer)"
    r"[^.!?]{0,60}?refus\w*", re.I)
#: The retired sentence itself, verbatim, so the scan's own sight is checked
#: against the thing it exists to catch rather than only against the tree.
RETIRED_SENTENCE = ("`automod_start` refuses a turn with no Inner Voice attached, "
                    "and the chat path is the only one that attaches it.")
#: A sentence/paragraph that says the rule is past.
_RETIRED = re.compile(r"retir\w*|no longer|does not|did not|would not|used to"
                      r"|any more|until 20\d\d", re.I)
#: The switch that would restore the gate, and the only thing that makes a
#: present-tense description of it safe to leave in a doc.
_IV_KEY = "require_inner_voice"
#: A retirement has to say when, or it is a rumour.
_DATED = re.compile(r"20\d\d-\d\d-\d\d")


def _scope() -> list[Path]:
    """The live tree's `architecture/*.md` + `workers/sources/*.py`.

    `.claude/worktrees/agent-*/` holds gitignored snapshots of both and each one
    still carries the old sentence; walking the directory tree would report four
    copies of a claim that is not in the tree, so this names the two live
    directories explicitly instead.
    """
    return sorted(ARCH.glob("*.md")) + sorted((ROOT / "workers" / "sources").glob("*.py"))


def _refusal_windows(text: str) -> list[str]:
    """±200 chars around every refusal that is ascribed to the missing observer.

    A window rather than a sentence: the retirement record for
    `architecture/automod.md`'s rule sits in the paragraph *above* the rule it
    retires, so a sentence-level split flags corrected history as a live claim —
    the #1713 failure this file already has a control test for. A window errs
    the other way, toward letting text through, which is the cheaper mistake.
    """
    return [text[max(0, m.start() - 200):m.end() + 200]
            for rx in (_IV_REFUSAL, _IV_REFUSAL_REV)
            for m in rx.finditer(text)]


def test_autocode_docstring_states_the_live_reason_not_the_retired_refusal():
    """Clause 1: the docstring was the ORIGIN of the false claim, not a copy.

    `architecture/workers-jobs.md` §4 got its sentence about a refusal from
    here, which is why correcting the doc alone left the loop able to re-derive
    the same false sentence on the next review. Checked in both directions so
    the paragraph cannot simply be deleted: the retired claim is gone, and the
    reason that IS true is still stated — the chat endpoint is the only place
    Inner Voice attaches, it is on for this source again since 2026-09-25, and
    it is where a round's transcript can be read.
    """
    doc = " ".join((autocode.__doc__ or "").split())
    assert doc, "autocode has no module docstring to check"
    assert "no Inner Voice attached" not in doc
    windows = _refusal_windows(doc)
    assert not windows, (
        f"the module docstring still couples a refusal with the missing observer "
        f"(#1689 retired that gate 2026-09-24): {windows[0][:160]!r} — the live "
        "reason to use /api/message/stream is the chat attach point, not a refusal")
    for live in ("chat endpoint", "2026-09-25", "transcript"):
        assert live in doc, (
            f"the true reason no longer states {live!r}: rewriting the retired "
            "claim must not delete why the source uses /api/message/stream")


def test_autocode_docstring_names_the_key_that_would_restore_the_gate():
    """Clause 2: a retirement is only traceable if it names its own switch.

    The key is checked against the code that reads it rather than against the
    docstring, because the failure this item is about is a claim that outlived
    the thing it described: if `agent_mcp/automod.py` stopped reading
    `automod.require_inner_voice`, a docstring pointing at it would be the next
    generation of the same bug.
    """
    doc = " ".join((autocode.__doc__ or "").split())
    assert "automod.require_inner_voice" in doc, (
        "the docstring says the refusal is gone without naming what would bring "
        "it back, so a reader cannot check the retirement")
    src = (ROOT / "agent_mcp" / "automod.py").read_text(encoding="utf-8")
    assert 'get("require_inner_voice", False)' in src, (
        "the docstring names a key the gate no longer reads, or one that no "
        "longer defaults false — the retirement note has outlived the code")
    assert "defaults false" in doc, (
        "the docstring names the key without its default, which is the half "
        "that says the gate is off now")


def test_no_live_file_asserts_the_inner_voice_refusal():
    """Clause 3: the sweep the item asked for, over the live tree only.

    A file may describe the rule in the present tense only if it also carries
    the retirement record — the key and a date — so `architecture/automod.md`
    keeps explaining how the gate worked while nothing may state it as current.
    The walk is paired with its positive control two ways: it must find a
    refusal window somewhere (a pattern empty against the corpus proves
    nothing), and the file that carries that window has to be one that records
    the retirement, or this test is passing on an unchecked exemption.
    """
    files = _scope()
    assert len(files) > 10, f"the walk found only {len(files)} files; it is not scanning"
    assert _refusal_windows(RETIRED_SENTENCE), (
        "the scan no longer matches the sentence #1689 was filed from, so an "
        "empty result below would be the pattern dying, not the claim dying")
    violations, windows, carriers = [], 0, set()
    for path in files:
        text = path.read_text(encoding="utf-8")
        if _IV_KEY in text and _DATED.search(text) and _RETIRED.search(text):
            carriers.add(path.name)
        for window in _refusal_windows(text):
            windows += 1
            if _RETIRED.search(window):
                continue
            if path.name not in carriers:
                violations.append((path.name, " ".join(window.split())[:160]))
    assert windows, (
        "the scan found no refusal-and-observer coupling anywhere under "
        "architecture/ or workers/sources/, so it cannot tell a fix from a "
        "broken pattern — architecture/automod.md is meant to still carry one")
    assert not violations, (
        f"{len(violations)} unlabelled refusal claim(s) remain: {violations} — "
        "state a retired rule only beside its key and date (#1689)")
    assert "automod.md" in carriers, (
        "architecture/automod.md no longer records the retirement, which is "
        "what licenses its description of the retired rule; a test that passes "
        "because every file is exempt is not a sweep")


def test_workers_jobs_dropped_the_pointer_at_the_docstring_it_fixed():
    """Clause 4: the doc used to say `autocode.py:25-29` still makes the claim.

    That sentence was true the day it was written and false the moment the
    docstring was fixed — the same carried-forward claim one layer down, which
    is the failure #1689 exists to end. So §4 keeps the retirement (key and
    date) and names no other file's contents: a section that says what another
    file says is a claim that rots every time that file is edited, and the test
    below is what notices instead of the prose.
    """
    sec4 = _section(SEC4)
    assert sec4, f"§4 {SEC4!r} not found — the heading moved and this test is blind"
    assert "autocode.py" not in sec4, (
        "§4 names the module docstring's contents again; #1689 deleted that "
        "pointer precisely because the next edit to the docstring falsifies it")
    assert "require_inner_voice" in sec4 and "2026-09-24" in sec4, (
        "§4 lost the retirement record along with the pointer — the dated fact "
        "is the part that has to stay")
