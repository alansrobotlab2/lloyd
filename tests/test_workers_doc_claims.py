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

import json
import re
import subprocess
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
SEC3 = "## 3. Dispatch — one door onto the autonomy fleet"
SEC5 = "## 5. Intake — the outside world in"
SEC6 = "## 6. Mining"
SEC7 = "## 7. Retired and renamed"

#: §Dispatch's invariant exactly as #1682 wrote it (#1744). The clause is kept
#: verbatim so the bans below can be shown to bite: a ban on a phrase the retired
#: text does not contain is a ban on nothing, which is the vacuity this file's
#: other nodes control for before they assert.
DISPATCH_INVARIANT_BEFORE_1744 = """  - **The invariant across both** (#938): an outage of the model server pauses
    dispatch, never the watching. The vLLM gate stays here because it is dispatch;
    the detectors above are read-only and never probe the model server.
"""


def test_intake_states_where_the_infra_path_spacing_lives_and_its_bound():
    """#1714: §Intake's own rule left a hole, and the hole is the doc's claim.

    The third rule says the retry lives outside the queue — each intake source
    keeps its own registry (`seen.json`, `research.db`) and decides there when to
    offer the work again. `youtube-digest`'s infra-shaped branch deliberately
    touches NEITHER registry (no `--fail`, so a primary outage is not charged to
    the video), which under that sentence left the re-offer spaced by
    `interval_seconds` alone: every 300 s, forever, one fresh row with
    `attempts=1` each time. The paragraph now has to say three things or it is
    describing a system that re-offers on every tick: the field a source uses to
    ask for spacing (`defer_seconds`), where the wait is stored (the queue's
    `not_before`, on the same row, which keeps its `dedup_key`), and what caps the
    re-offers once each wait passes (`max_attempts`, because `claim_next` raises
    `attempts` on every claim).
    """
    sec5 = _section(SEC5)
    assert sec5, f"§5 {SEC5!r} not found — the heading moved and this test is blind"
    flat = " ".join(sec5.split())

    for fact, why in (
        ("defer_seconds", "the field a source asks for spacing with"),
        ("not_before", "where the wait is actually stored, so a reader can find it"),
        ("dedup_key", "the row keeps its key, or the next tick mints a fresh row"),
        ("INFRA_DEFER_SECONDS", "which source sets it, named"),
        ("max_attempts", "what caps the re-offers once each wait passes"),
        ("seen.json", "the registry the branch still refuses to touch"),
    ):
        assert fact in flat, (
            f"§Intake no longer states {why} ({fact!r} is gone): the rule reads as "
            "'the source owns its retry', which is the statement #1714 was filed "
            "against for a branch that owns nothing")

    m = re.search(r"`INFRA_DEFER_SECONDS`[^.]*?\((\d+)\s*s", flat)
    assert m, (
        "§Intake names INFRA_DEFER_SECONDS without its number of seconds beside it, "
        "so nothing here can be compared with the constant — 'a longer interval' is "
        "not a bound")
    assert int(m.group(1)) == 900, (
        f"§Intake says {m.group(1)} s while workers/sources/youtube_digest.py sets "
        "INFRA_DEFER_SECONDS = 900: one statement, two places, and the doc is the "
        "one a reader trusts")


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


# ------------------------------------------------------- `bench-mine`'s cap
# (#1712).
#
# `MAX_ENQUEUE_PER_TICK`'s comment said "per tick, per input" while
# `enqueue_if_due` passed the resolved `max_enqueue_per_tick` value to the
# failed-runs selector alone; the ledger input ran at `_recent_ledger_losers`'s
# own default 5. §6 of this doc was then rewritten (#a42640eb, 2026-09-27) to
# state exactly that — "on the failed-runs input only ... neither that constant
# nor the `max_enqueue_per_tick` key bounds it (filed)" — so the code fix makes
# THE DOC the false statement unless the same sha repairs it. Nothing else in
# this file reads that paragraph, which is why the two nodes below exist: a
# missed doc edit would otherwise reach no test at all.

_BENCH_MINE_SRC = ROOT / "workers" / "sources" / "bench_mine.py"


#: The false sentence itself, verbatim, so the ban below is proven to have eyes:
#: a `not in` over a phrase the scan could never have matched is a passing test
#: about nothing (#1689's `RETIRED_SENTENCE`, same reason).
RETIRED_CAP_SENTENCE = ("**Wakes** every 7200 s with two deliberately independent inputs, "
                        "capped at `MAX_ENQUEUE_PER_TICK` (3) on the failed-runs input "
                        "only — the ledger input is offered by `_recent_ledger_losers` at "
                        "its own default `limit=5`, so neither that constant nor the "
                        "`max_enqueue_per_tick` key bounds it (filed)")


def _cap_sentence(flat_sec: str) -> str:
    """The sentence of `flat_sec` that states the enqueue cap, whitespace-flattened.

    Scoped to one sentence rather than the section because the key
    `max_enqueue_per_tick` appears in the RETIRED prose too — "neither that
    constant nor the `max_enqueue_per_tick` key bounds it" — so a section-wide
    token check could be satisfied by the very sentence this node exists to
    delete. The cap sentence is the only place the true scope may be asserted.
    """
    hits = [s for s in re.split(r"(?<=[.!?]) ", flat_sec) if "capped at" in s]
    assert hits, (
        "§6 no longer has a sentence stating the enqueue cap ('capped at'), so "
        "there is nothing here to check the scope of — the paragraph was deleted, "
        "not corrected")
    assert len(hits) == 1, (
        f"§6 states the cap in {len(hits)} sentences; one sentence is the scope "
        "this node grades, and two would let the false half hide in the other")
    return hits[0]


def _const_comment(src: str, name: str) -> str:
    """The `#:` comment block sitting directly above a module constant.

    Block-anchored rather than a whole-file grep, because "per input" appearing
    anywhere else in the module would otherwise make the ban unenforceable and a
    key named in an unrelated comment would make the requirement free.
    """
    m = re.search(r"((?:^#:[^\n]*\n)+)^" + re.escape(name) + r" = ", src, re.M)
    assert m, (
        f"no `#:` comment block directly above `{name}` — deleted rather than "
        "corrected, or the constant moved off the column-0 form this extracts")
    return m.group(1)


def test_bench_mine_cap_comment_names_the_key_that_overrides_it():
    """Clause 4, code half: the comment's two false halves, pinned both ways.

    "per input" was false for the one input the cap did not reach, and the
    comment never named the config key that overrides the constant — so an
    operator reading it had no way to find the knob or to learn it was
    half-read. Both directions are asserted: the retired phrase is gone, and the
    scope plus the key are actually stated, so deleting the comment cannot pass.
    """
    src = _BENCH_MINE_SRC.read_text(encoding="utf-8")
    # Positive control that the extractor has eyes: a sibling constant in the
    # same block is found too. An extractor matching nothing makes every
    # `not in` below vacuously true.
    assert "max_turns" in _const_comment(src, "DEFAULT_MAX_TURNS").lower(), (
        "the comment extractor matched a block with no key in it, so the checks "
        "below are reading the wrong text")

    comment = _const_comment(src, "MAX_ENQUEUE_PER_TICK")
    assert "per input" not in comment.lower(), (
        "the comment is back to claiming the cap covers every input by its "
        "spelling — #1712 is the input it did not reach")
    assert "max_enqueue_per_tick" in comment, (
        "the comment no longer names the config key that overrides the constant, "
        "so an operator cannot find the knob from the number")
    assert "both" in comment.lower(), (
        "the comment states the key but not that one value bounds both of the "
        "source's inputs, which is the scope #1712 fixed")
    assert 'src_cfg.get("max_enqueue_per_tick", MAX_ENQUEUE_PER_TICK)' in src, (
        "the comment names a key the code stopped reading, or read a different "
        "way — the note has outlived the resolution it describes")


def test_workers_jobs_states_the_cap_over_both_inputs():
    """Clause 4, doc half: §6's sentence must follow the code, not the bug.

    The negative checks are the sentence #a42640eb wrote on purpose; it was true
    when written and is the false statement now. The positive checks are the
    paired requirement from #1689/#1713: a correction has to say the true scope
    in the present tense, not merely drop the lie — and the threading it claims
    is re-checked against the source, because prose about a call site rots the
    moment that call site moves.
    """
    sec6 = _section(SEC6)
    assert sec6, f"§6 {SEC6!r} not found — the heading moved and this test is blind"
    # Both sides graded on whitespace-normalised prose: the retired sentence
    # wrapped across four source lines, so a raw check would let a re-flow of the
    # same false claim through.
    flat = " ".join(sec6.split())
    retired = ("on the failed-runs input only", "nor the `max_enqueue_per_tick` key")
    # Control before bans: the retired text has to trip both phrases and the
    # extractor has to see the sentence they sit in. An empty pattern against an
    # empty corpus passes, and that is what this line is for.
    old = " ".join(RETIRED_CAP_SENTENCE.split())
    assert all(r in old for r in retired), (
        "the verbatim retired sentence no longer trips its own bans, so the two "
        "checks below are vacuous and would pass on any text")
    assert "capped at" in old and "max_enqueue_per_tick" in _cap_sentence(old), (
        "the cap-sentence extractor cannot see the retired sentence, so scoping "
        "§6 to it would silently grade nothing")

    for phrase in retired:
        assert phrase not in flat, (
            f"§6 still says {phrase!r} — true until #1712 threaded the cap, and "
            "the sentence that has to be rewritten in the same sha as the fix")

    # The live scope is graded inside the cap sentence alone: the key's name
    # occurs in the retired prose as well, so a section-wide token check could
    # read as satisfied by the sentence the node just banned.
    cap = _cap_sentence(flat)
    for live in ("each capped at", "`MAX_ENQUEUE_PER_TICK`", "max_enqueue_per_tick",
                 "FAILURE_WINDOW_DAYS", "both selectors"):
        assert live in cap, (
            f"the cap sentence no longer states {live!r}: {cap[:220]!r} — the "
            "false sentence must be replaced by the true one, not deleted")

    src = _BENCH_MINE_SRC.read_text(encoding="utf-8")
    assert "_enqueue_ledger_losers(queue, src_cfg, limit)" in src, (
        "§6 says the resolved value reaches both selectors; the ledger input no "
        "longer receives it, so the paragraph is false again")
    assert re.search(r"days=FAILURE_WINDOW_DAYS,\s*limit=limit", src), (
        "§6 says the ledger selector gets FAILURE_WINDOW_DAYS as well as the "
        "cap; the call site stopped passing it as named arguments, so either the "
        "prose or the threading has moved")


#: The clause #1682 wrote the vLLM gate with, and therefore the anchor for the
#: bullet #1744 has to name the co-watcher inside.
GATE_BULLET_MARK = "The vLLM gate stays here"


def _invariant_bullet(sec: str) -> str:
    """The §Dispatch invariant bullet — the one carrying the vLLM-gate sentence.

    #1744 clause 1 puts the co-watcher BESIDE that sentence, so the pair is
    graded inside this bullet and not across §3: a section-wide token check
    passes with the filename two bullets away from the gate it belongs to, and
    "named somewhere in the section" is precisely the state §3 was in for
    `workers/fleet_watchdog.py` and not for the probe.
    """
    if GATE_BULLET_MARK not in sec:
        return ""
    rest = sec[sec.index(GATE_BULLET_MARK):]
    end = rest.find("\n\n")
    return rest if end < 0 else rest[:end]


def test_dispatch_names_the_probe_as_co_watcher_of_the_primary_port():
    """#1744 clause 1: §Dispatch has to name BOTH watchers of :8096, with the
    pair of thresholds that make one outage two lines.

    #1682 rewrote this section and carried `workers/fleet_watchdog.py` into it
    while leaving `workers/service_probe.py` out, so the section that exists to
    explain who watches what read as "nothing watches the model server" — the
    exact #1683 defect reproduced inside the doc that triggered it. The pair is
    pinned to the constants it describes rather than to the prose alone: a doc
    that keeps the right sentence while a number moves underneath it is the
    failure mode this file exists for, and the ordering the bullet claims (the
    probe's line first) is only true because the grace is strictly below the
    gate's threshold.
    """
    from workers.service_probe import GRACE_S, PRIMARY_ENGINE
    from workers.sources.scheduled_task import _VLLM_DOWN_ALERT_SECONDS

    sec3 = _section(SEC3)
    assert sec3, f"§3 {SEC3!r} not found — the heading moved and this test is blind"
    flat = " ".join(sec3.split())

    # Control before bans: the retired bullet has to trip the phrase this node
    # forbids, and must genuinely be the version that named only one watcher.
    old = " ".join(DISPATCH_INVARIANT_BEFORE_1744.split())
    assert "never probe the model server" in old, (
        "the retired invariant no longer trips its own ban, so the check below "
        "is vacuous and would pass on any text")
    assert "service_probe" not in old, (
        "the quoted bullet already names the co-watcher, so it is not the "
        "#1682 text this node is meant to exclude")
    assert "never probe the model server" not in flat, (
        "§Dispatch is back to denying that anything probes the model server — "
        "the sentence a fresh arch review reads as an open coverage gap (#1683)")

    # Graded inside the bullet that carries the gate sentence, for the reason in
    # `_invariant_bullet`: the clause is about the pairing, not the section.
    gate = " ".join(_invariant_bullet(sec3).split())
    assert "the vllm gate stays here" in gate.lower(), (
        f"{GATE_BULLET_MARK!r} has left §Dispatch, so this node has lost its "
        "anchor and would be grading an empty window")

    assert "workers/service_probe.py" in gate, (
        "§Dispatch no longer names the co-watcher of the primary's port beside "
        "the vLLM-gate sentence — the pairing is what stops the next arch review "
        "reading §3 as an unprobed model server (#1683)")
    assert ":8096" in gate, (
        "§Dispatch names a second watcher without saying which port they share")
    assert "TWO watchers" in gate, (
        "§Dispatch does not say the gate is one of two watchers, which is the "
        "claim the next arch review needs to see stated")

    # The pair, each with its own threshold and its own surface.
    assert "45 min" in gate and "`logger.error`" in gate and "discord" in gate, (
        f"the gate sentence no longer states its threshold and its two surfaces: {gate!r}")
    assert "30 min grace" in gate and "journal" in gate and "toast" in gate, (
        f"the co-watcher sentence no longer states the grace and that it lands "
        f"in the journal and a toast: {gate!r}")

    # ...and the numbers are the code's, not the doc's memory of them.
    assert f"{GRACE_S[PRIMARY_ENGINE] // 60} min grace" in gate, (
        f"§Dispatch says 30 min grace but `GRACE_S[{PRIMARY_ENGINE!r}]` is "
        f"{GRACE_S[PRIMARY_ENGINE] // 60} min — the prose has to follow the "
        "constant, and only their agreement is the ordering claim")
    assert f"{_VLLM_DOWN_ALERT_SECONDS // 60} min" in gate, (
        f"§Dispatch says 45 min but `_VLLM_DOWN_ALERT_SECONDS` is "
        f"{_VLLM_DOWN_ALERT_SECONDS // 60} min")
    assert GRACE_S[PRIMARY_ENGINE] < _VLLM_DOWN_ALERT_SECONDS, (
        "the probe's grace no longer sits strictly below the gate's threshold, "
        "so the bullet's 'the probe line first' is false — and one outage would "
        "read as two incidents")


def test_every_probe_mention_in_the_doc_names_the_probe_file():
    """#1744 clause 2: in a doc that finally names the probe, no line may get to
    say the word without naming it.

    The old bullet's damage was one verb — "never probe the model server" — in a
    section that omitted the thing it named. Deleting that sentence alone would
    leave the same trap one edit away, so the pin is the shape rather than the
    sentence: every line of this doc that contains "probe" is a line that
    identifies `workers/service_probe.py`. The invariant the bullet carries is
    asserted at the same time, because clause 2 rewords it ("never touch the
    model server") rather than dropping it — deleting the read-only guarantee to
    get past the ban would satisfy this node's letter and undo its point.
    """
    text = (ARCH / "workers-jobs.md").read_text(encoding="utf-8")
    hits = [(n, ln) for n, ln in enumerate(text.splitlines(), 1)
            if "probe" in ln.lower()]
    # Positive control: the doc still mentions the probe at all. Zero hits makes
    # the loop below vacuously true, and zero hits would itself be the regression.
    assert hits, (
        "architecture/workers-jobs.md mentions no probe anywhere, so the loop "
        "below proves nothing — and a doc that lost the co-watcher sentence "
        "altogether is precisely the #1744 regression")
    # #1981 registered a source whose own name contains the word: the
    # `frontend-probe-canary` source runs `scripts/automod/frontend_probe_canary.py`,
    # which measures the gate's FRONTEND probe and has nothing to do with the
    # service probe. The rule is unchanged in kind — a line that says "probe" names
    # which probe file it means — with a second file a line may name. A bare
    # "probe" still fails.
    named = ("service_probe", "frontend_probe", "frontend-probe-canary")
    assert any("service_probe" in ln for _, ln in hits), (
        "no line names workers/service_probe.py any more — the frontend canary's "
        "lines must not be what keeps this node from being vacuous")
    for n, ln in hits:
        assert any(name in ln for name in named), (
            f"architecture/workers-jobs.md:{n} uses the word probe without "
            f"naming the probe file: {ln.strip()!r}")
    flat = " ".join(text.split())
    assert "read-only and never touch the model server" in flat, (
        "the §Dispatch invariant is gone rather than reworded: clause 2 asks "
        "that 'probe' stop being a verb about the model server, not that the "
        "detectors' read-only guarantee stop being stated")


# ---------------------------------------------------------------------------
# #1769 clause 5 — the docstring that stops the next re-measurement from being
# #1710 again.
#
# A stale staged note (no `calibration.task_id`) is re-measured by parsing its
# BODY with `_candidate_frontmatter` and passing `task=`. Handing the
# calibration entry point the note's PATH re-reads the staging envelope and
# spends ten GPU trials on a document that is not a task. That route exists
# nowhere a future caller can trip over it except in `bench_mine`'s module
# docstring, so the docstring is the enforcement — which makes it prose that has
# to be graded like the tables above.
# ---------------------------------------------------------------------------

def _module_docstring(path: Path) -> str:
    """The MODULE docstring only, whitespace-flattened.

    Extracted through the AST rather than read off the file text, because
    `_candidate_frontmatter`, `task=` and `calibrate_candidate(path)` all appear
    in this module's CODE as well — a whole-file `in` check would be satisfied by
    the very call sites the docstring exists to warn about (#1689's rule for a
    `not in` with no eyes, applied to a `in`).
    """
    import ast

    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    assert doc, f"{path.name} has no module docstring — the extractor found nothing"
    return " ".join(doc.split())


def test_bench_mine_docstring_states_the_route_for_a_stale_note():
    """#1769: a reader who decides to re-measure the relabelled notes has to find
    the `task=` route, the fact that `_candidate_frontmatter` takes text, and the
    reason the path form is wrong — in that module, where they will look.

    Pinned on the sentence, not the docstring, for the ban: the docstring also
    *describes* the path fallback correctly, so a doc-wide "never" check could be
    satisfied by an unrelated sentence while the actual instruction went back to
    recommending the path form.
    """
    doc = _module_docstring(_BENCH_MINE_SRC)
    # Positive control that the extractor has the right text: the module's own
    # opening words, which no other string in the file carries.
    assert doc.startswith("bench-mine source"), (
        f"the extractor returned something that is not this module's docstring: "
        f"{doc[:60]!r}")

    assert "_candidate_frontmatter" in doc, (
        "the function a stale note must be parsed with is unnamed again, so the "
        "route is not findable from the module that owns it")
    assert "TEXT, not a path" in doc, (
        "`_candidate_frontmatter` takes text; without that stated, the obvious "
        "call is the one that passes the Path and re-opens #1710")
    assert "task=" in doc, "the argument the parsed task has to travel in is gone"
    assert "uncalibrated" in doc and "stale_envelope" in doc, (
        "the docstring no longer says how a note that was never measured of its "
        "task is labelled, so a reader cannot tell a stale note from a real one")

    ban = [s for s in re.split(r"(?<=[.!?]) ", doc) if "calibrate_candidate(path)" in s]
    assert len(ban) == 1, (
        f"the docstring states the path form in {len(ban)} sentences; one "
        "sentence is the instruction this node grades, and two would let a "
        "recommendation hide beside the warning")
    assert "never" in ban[0].lower(), (
        f"the sentence naming `calibrate_candidate(path)` no longer forbids it: "
        f"{ban[0]!r}")
    for fact in ("_load_candidate", "FIRST front matter block", "envelope"):
        assert fact in ban[0], (
            f"the warning sentence lost {fact!r}, which is the reason a later "
            "reader needs to obey it rather than 'simplify' the call back to a "
            "path — the exact edit #1710 exists to prevent")
    assert "#1710" in doc, (
        "the docstring stopped attributing the envelope-reading bug to the item "
        "that fixed it, so the warning has no failure a reader can go read")


# ---------------------------------------------------------------------------
# #1774 clause 2 — the marker set reaches the SCAN, and only one caller builds it.
#
# #1711 taught the ledger input to read its `done:` markers but applied them after
# `rows[:limit]`, so a retired row still spent the per-tick budget; the live cost
# on 2026-09-29 was 20.9 hours of silence against 44 unmined losers in the window.
# Moving the test into the scan means the caller can no longer do its own
# filtering, and the failure this node prevents is the quiet re-growth of that: a
# second marker set built in the caller, or the `done=` argument dropped from the
# call (which the scan's default absorbs silently, exactly as #1712's `or
# MAX_ENQUEUE_PER_TICK` fallback absorbed a dropped `limit`).
#
# Graded on the SOURCE TEXT, like the doc tables in this file, because the fact is
# about how the call reads: a behaviour test cannot see a `done=` that is absent,
# only the slate it produces, and a fixture wide enough to hide it is a fixture
# that no longer tests the budget.
# ---------------------------------------------------------------------------

#: The retired comment block, verbatim from the caller before #1774. Kept so the
#: bans in the next node have a control: this text trips every one of them, so they
#: are not empty patterns passing on empty corpora (#1689's rule).
RETIRED_POST_SLICE_FILTER = (
    "Filtered after the slice, on purpose. Marked rows still consume the "
    "selection budget, so a tick whose whole slate is marked enqueues nothing")


def _func_src(src: str, name: str) -> str:
    """The source of one top-level function, by name.

    Scoped to a function body rather than the whole module because
    `wm_keys(NAME)` is legitimately built twice in this file — the ledger input
    and the failed-runs input each build the set — and a module-wide count would
    either fail on that or drift with an unrelated caller.
    """
    import ast

    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            seg = ast.get_source_segment(src, node)
            assert seg, f"{name} has no extractable source segment"
            return seg
    raise AssertionError(f"{name} is gone from the module — this node is blind")


def test_the_marker_set_is_passed_into_the_scan_and_built_once():
    """#1774: `_recent_ledger_losers(days=…, limit=…, done=…)` is called with all
    three named, from a caller that keeps its shape and builds exactly one set.

    The three pins are the three ways this fix quietly un-fixes itself: drop
    `done=done` and the scan's default filters nothing while every existing
    behaviour test still passes on a small fixture; build a second set in the
    caller and the two halves disagree about what was mined; change the caller's
    signature and §6's sentence about the resolved cap reaching both selectors is
    false again (#1712's pin, still holding).
    """
    src = _BENCH_MINE_SRC.read_text(encoding="utf-8")
    caller = _func_src(src, "_enqueue_ledger_losers")

    # Control first: the retired comment must trip the ban it is banned by, and
    # the extractor has to be looking at the function that held it.
    retired = " ".join(RETIRED_POST_SLICE_FILTER.split())
    assert "Filtered after the slice" in retired and "selection budget" in retired, (
        "the retired text no longer trips its own bans, so the checks below are "
        "vacuous and would pass on any module")
    assert "Filtered after the slice" not in caller, (
        "the caller is filtering after the slice again — that is the placement "
        "#1711's owed entry ruled against, and the 20.9-hour silence that followed "
        "it: retired rows spending a budget they cannot spend twice")

    assert re.search(
        r"async def _enqueue_ledger_losers\(queue: WorkQueue, src_cfg: dict, "
        r"limit: int\) -> None:", src), (
        "the ledger caller no longer takes (queue, src_cfg, limit): §6's cap "
        "sentence claims the resolved value reaches both selectors, and this is "
        "the seam that carries it")
    assert re.search(
        r"_recent_ledger_losers,\s*days=FAILURE_WINDOW_DAYS,\s*limit=limit,\s*"
        r"done=done", caller), (
        "the scan is no longer handed FAILURE_WINDOW_DAYS, the resolved cap and "
        "the marker set by name — a dropped `done=done` is invisible to every "
        "behaviour test on a small fixture, which is how #1712's budget fallback "
        "hid for a month")
    assert caller.count("wm_keys(NAME)") == 1, (
        f"the ledger caller builds its marker set {caller.count('wm_keys(NAME)')} "
        "times; the scan owns the filtering now, and a second set in the caller is "
        "a second opinion about what was already mined")
    assert "done = {k[len(\"done:\"):] for k in queue.wm_keys(NAME)" in caller, (
        "the caller no longer builds the one set it passes in, so `done=done` "
        "names something this function does not construct")


def test_workers_jobs_says_the_filter_runs_before_the_slice():
    """§6's sentence has to follow the placement, not just the existence, of the
    filter: it currently promises the retired-rows-consume-budget behaviour as
    live policy and defers the question to an owed-check that has since answered.

    Both halves again: the retired claim banned (with a control that it trips the
    ban), and the live claim required in the same section, because deleting a false
    sentence is not the same edit as writing the true one (#1689/#1713).
    """
    flat = " ".join(_section(SEC6).split())

    retired_claims = ("marked rows still consume the slice",
                      "or the filter belongs before the slice, is still owed")
    old = ("What that fix does not do is widen the selector's `limit = 5` — marked "
           "rows still consume the slice, so with all five current rows marked the "
           "input offers nothing until they age out of the 7-day window (~09-29/30) "
           "or new sub-0.6 rows appear. Whether that silence is acceptable, or the "
           "filter belongs before the slice, is still owed (#1711's owed-check).")
    assert all(c in old for c in retired_claims), (
        "the retired §6 text no longer trips its own bans, so the check below is "
        "vacuous — it would pass on a section that never mentioned the slice")
    for claim in retired_claims:
        assert claim not in flat, (
            f"§6 still states {claim!r} as the live behaviour: #1774 moved the "
            "filter inside the scan, and this paragraph is what a re-triaging "
            "reader would trust instead of the source")

    live = [s for s in re.split(r"(?<=[.!?]) ", flat) if "rows[:limit]" in s]
    assert live, (
        "§6 no longer names `rows[:limit]` anywhere, so nothing here ties the "
        "section to where the filter actually runs")
    assert any("before" in s and "#1774" in s for s in live), (
        "the slice sentence does not state that the retired pair is dropped before "
        f"it and attribute the change: {[s[:90] for s in live]}")
    budget = [s for s in re.split(r"(?<=[.!?]) ", flat)
              if "max_enqueue_per_tick" in s and "#1774" in s]
    assert budget, (
        "§6 does not state, in the same breath as the change, that the budget did "
        "NOT move — the half a reader is most likely to over-read as 'and now it "
        "fetches a wider slate'")


# ---------------------------------------------------------------- `runs` §2 — #1775
#
# §2's table is a measurement with three failure modes: the sentence that explains
# a row's absence can rot into an inference the code contradicts
# (`automod-regression`: "no row to read, which is the design working", when
# `start_runner()` skips precisely because no promotion is waiting); an unbounded
# SQL predicate plus a count can go stale while looking dated (the window has no
# upper bound, and `queue-maintenance`'s count grows every 900-second sweep); and
# the read stamp can be left behind by the pool, which is how §2 came to report
# `automod-regression` as nothing at all thirteen hours after its first row.
#
# The nodes below are §2-scoped because `_section()` is the whole reason that
# matters: a section-local check fails when the fact moves elsewhere in the doc or
# vanishes, where a whole-file grep is silent about both. None of them quotes a
# run count — `runs` is pruned at 30 days (`app/routers/workers.py:93`), so a
# number in a test is a green suite holding a dead claim.
#
# And they deliberately do NOT read `~/lloyd-data/workers.db` to check the table is
# still true, which is the obvious next node and is unsound here: since 2026-09-22
# `gate._child_env` points `$HOME` at the round's own home (`tests/conftest.py:140`),
# and that home has its own EMPTY `lloyd-data/workers.db` — first measured by
# tripping over it, where the gate's store held 654 autotriage rows and the round
# home held 0. A doc test reaching for the live store is one env change away from
# asserting that a real measurement is a lie. Freshness of §2's numbers is the
# owed-check job's, over a window of real churn, with the store in front of it.

#: A conclusion is banned, not the words that carried it. The two below were
#: §2's readings of an empty row and of a missing row; each was wrong, and each is
#: wrong wherever the doc puts it, so both are banned across the FILE while the
#: facts they dressed up are required inside §2.
#:
#: Deliberately absent from this tuple: "the design working". §2 keeps that phrase
#: twice after #1775 — of `automod-regression` declining to offer at all
#: (`enqueue_if_due` → `_runner_needed()`), and of `queue-maintenance` writing rows
#: without being a work source — and both uses are now backed by the code the node
#: below re-reads.
#: Banning the phrase would have been the wrong fix: it is a legitimate conclusion,
#: it was only ever illegitimate as an inference from an absence. What is banned is
#: the two specific absences-inferred-as-health, each of which was a false sentence
#: wherever it appeared.
BANNED_SECTION2_CONCLUSIONS = (
    "whether it should is an open question",
    "has no row to read",
)

#: The window §2's table was measured over. Clause 5's "window end at 2026-09-28"
#: is about the END of the prose window; the SQL predicate keeps its unbounded
#: `>=` and its date, and §6's copy of the same date keeps its prose ("the 7 days
#: to 2026-09-11", "days to 2026-09-28"), which `test_section2_and_section6_report_one_measurement`
#: holds equal across the two sections.
SECTION2_WINDOW_END = "2026-09-28"
SECTION2_SQL_FLOOR = "2026-09-21"


def _sec2_prose() -> str:
    """§2 with its table rows dropped, so a count-check can run over the prose.

    The table is exempt because its header stamp governs every cell in it, and the
    node that pins that stamp is below; the prose is where a count outlives its
    measurement.
    """
    return "\n".join(ln for ln in _section(SEC2).splitlines()
                     if not ln.lstrip().startswith("|"))


def test_section2_says_why_queue_maintenance_is_not_a_row():
    """Clause 1 + 2: the open question is gone, replaced by the settled reason.

    Self-checking on the corpus: `BANNED_SECTION2_CONCLUSIONS` is asserted non-empty
    up front, so a refactor that drops the banned strings from the tuple cannot
    quietly turn this node into a no-op.
    """
    assert BANNED_SECTION2_CONCLUSIONS, "the ban list itself was emptied"

    sec2 = _section(SEC2)
    doc = " ".join((ARCH / "workers-jobs.md").read_text(encoding="utf-8").split())
    for banned in BANNED_SECTION2_CONCLUSIONS:
        assert " ".join(banned.split()) not in doc, (
            f"the doc still concludes {banned!r} somewhere: it is wrong wherever "
            "it lands — an unregistered name's absence is a design fact, and a "
            "zero row is only ever a skip")

    assert "queue-maintenance" in sec2, (
        "§2 stopped mentioning the one unregistered name that writes rows, so the "
        "14 names in the store against 13 rows reads as an omission again")

    flat = " ".join(sec2.split())
    assert "registered" in flat, (
        "§2 no longer says what a row of its table is, which is the half of the "
        "reason a 14th name has no row")
    assert "workers/maintenance.py" in sec2, (
        "the settled reason has to name the module that records the rows, or a "
        "reader cannot check it")
    assert "nothing enqueues into it" in flat, (
        "§2 lost the maintenance module's own words for why it is not a work "
        "source — the reason has to be quotable, not paraphrased away")
    for token in ("work source", "workers.sources"):
        assert token in flat, (
            f"§2's queue-maintenance reason lost {token!r}: the point is that it "
            "is deliberately not a work source AND has no config block, and either "
            "half alone lets a reader re-file the open question")


def test_section2_table_names_only_registered_sources():
    """#1775: the ruling that `queue-maintenance` gets no row of §2's table, as a check.

    The store holds rows under a name the registry has never carried — `run_sweep()`
    books one into `runs` from `workers/maintenance.py`, from the scheduler's own
    sweep rather than from a runner claim. §2's table is the one place in this doc
    whose rows read as "this source ran through the queue", so that name is a standing
    temptation to add a row to it, and #1775's ruling was that none is added: the sweep
    is deliberately not a work source, nothing enqueues into it, and §2's prose bullet —
    the one that dates its count — IS the record. Prose in a closed item stays prose;
    this node is the same ruling as an assertion, so the row now fails a test instead
    of passing a suite.

    The two asserts are deliberately asymmetric, and the asymmetry is the ruling. The
    name has to be in §2's prose and must not be in its parsed table, both at once.
    That is also why the check cannot be a ban on the name anywhere in the section:
    §2's text names it twice, both times in that bullet, so
    `queue-maintenance not in _section(SEC2)` is satisfied by the very sentence the
    ruling points at — and would still pass on a doc that had gone on to add the row.
    `_sec2_prose()` drops table rows, so the bullet assert cannot be licensed by a row;
    `_table_after` parses the table and nothing else, so the table assert cannot be
    satisfied by prose.

    Subset, never equality. A registered source that ran nothing inside §2's single
    dated window legitimately has no row, so `rows == _registry()` would go red on a
    quiet week over a doc that is telling the truth; equality against the registry is
    `test_workers_jobs_roster_is_the_registry`'s claim, on §1's roster, where every
    source belongs whatever it did that week. At this table's stamp the two sets do
    coincide, which is a fact about one window and the reason it is not asserted here.

    Names only. Every column but the first is a run count from a window that moves on
    every sweep, and `test_section2_carries_no_undated_count` is the node that deals
    with numbers; this one reads the first column and nothing else.
    """
    rows = _table_after(ARCH / "workers-jobs.md", SEC2)
    assert rows, "§2 has no per-source table to check"
    registered = _registry()

    unregistered = rows - registered
    assert not unregistered, (
        f"§2's run table carries {sorted(unregistered)}, which SOURCE_REGISTRY does "
        "not name. A row is a claim that a queued source ran; #1775 ruled that an "
        "unregistered bookkeeping name like `queue-maintenance` is recorded in this "
        "section's prose bullet instead — and "
        "test_config_configures_only_registered_sources is already there to refuse "
        "the other way round, a config block for a name no runner claims")

    assert "queue-maintenance" in _sec2_prose(), (
        "§2's prose no longer records `queue-maintenance`, which is where #1775 put "
        "the record: clearing the bullet would leave the table's silence looking "
        "tidy while destroying the only account of why the name has rows but no row")


def test_section2_explains_what_a_zero_in_the_automod_regression_row_means():
    """Clause 3: the placeholder zero is gone and the inference is now the code's.

    Cross-checked against the source it describes: `start_runner()` really is this
    source's poll, and it really returns `_skipped(...)` when `pending_promotions()`
    is empty. Without that cross-check this is a better-worded guess, and the
    previous wording was exactly as fluent and exactly as wrong.
    """
    sec2 = _section(SEC2)

    # The bullet is found by the function it is required to name, not by the source
    # name: §2 mentions `automod-regression` three times before any bullet — in the
    # header stamp's own cautionary sentence and in the table row — so selecting on
    # the source name returns the table and grades the wrong text (found while
    # falsifying this node: the pre-fix §2 passed an early version of it).
    bullet = [b for b in sec2.split("\n- ") if "start_runner" in b]
    assert len(bullet) == 1, (
        f"§2 has {len(bullet)} bullets naming `start_runner()`, which is the "
        "clause-3 bullet's identity — with none the bullet is gone, and with two "
        "this node cannot tell which one to grade")
    b = " ".join(bullet[0].split())

    assert "start_runner()" in b or "start_runner`" in b, (
        "the bullet no longer names the function in backticks-as-a-call, so its "
        "line cite cannot be followed")
    assert "_runner_needed" in b, (
        "§2 lost the poll-side gate, so its zero reads only as a recorded skip: "
        "`_runner_needed()` declines the OFFER, which writes no row at all, while "
        "`start_runner()`'s skip does write one — the difference between a zero "
        "with rows behind it and a zero nobody can see")
    assert "pending_promotions()" in b, (
        "the condition behind the skip is gone: a zero means an empty "
        "`pending_promotions()` result, and nothing else")
    assert "no promotion was pending" in b, (
        "the bullet lost the sentence that IS clause 3: what a zero means")

    # "the design working" is not banned outright — §2's queue-maintenance bullet
    # uses it as a settled conclusion, correctly. In THIS bullet it may only appear
    # as a quotation being refuted, which is mechanical to check: inside quotes.
    hits = [k for k in range(len(b)) if b.startswith("design working", k)]
    assert hits, (
        "the bullet no longer quotes the reading it exists to refute, so the loop "
        "below has nothing to grade and the false conclusion could return "
        "unopposed — keep the refutation in the bullet (#1689's rule for a loop "
        "over a possibly-empty set)")
    for i in hits:
        before, after = b[:i], b[i + len("design working"):]
        assert before.count('"') % 2 == 1 and after.count('"') % 2 == 1, (
            "the automod-regression bullet asserts \"the design working\" as its "
            f"own conclusion rather than quoting it as the refuted reading: …{b[max(0, i - 90):i + 40]}…")

    src = (ROOT / "workers" / "sources" / "automod_regression.py").read_text(
        encoding="utf-8")
    assert "def _runner_needed(" in src and "def start_runner(" in src, (
        "one of the two functions §2's bullet cites is gone from the module, so "
        "the bullet and the code disagree about what this source's poll is")
    assert "no promotion is waiting to be measured" in src, (
        "start_runner's skip reason changed wording, so §2's quotation of it is "
        "stale — re-read the module, do not edit the doc to match this test")


def test_section2_carries_no_undated_count():
    """Clause 4, generalized: a count in §2 has to be dated or gone.

    Two halves, because clause 4 offers a choice and a one-sided test can be passed
    by the half this item did not take. (a) If §2 still states its
    queue-maintenance count it must sit beside an `as of <ISO>Z>` stamp —
    "19 of them here" was true for about eleven hours and grew on every 900-second
    sweep. (b) No other bare count appears in §2's prose: the table is exempt
    because the header stamp governs its cells and the node below checks it against
    the store.

    The scan runs per bullet, not per section: §2's bullets sit in one block
    separated by `- ` lines, so splitting only on blank lines would let the one
    dated bullet license every undated count in the list — the vacuity this node
    exists to catch. Dated means a WINDOW or a READ STAMP, not any date string: the
    bullet this clause was filed against opened "The store on this box begins
    2026-09-22T20:03Z", so `2026-..-..T` as the marker would have licensed the very
    count that went stale. A wipe date is not a measurement window.
    """
    prose = _sec2_prose()

    qm_bullets = [b for b in prose.split("\n- ") if "queue-maintenance" in b]
    assert qm_bullets, "§2 no longer mentions queue-maintenance at all"
    qm_bullet = " ".join(qm_bullets[0].split())
    stated = re.search(r"\b(\d+)\s+of them\b", qm_bullet)
    if stated:
        assert re.match(r"[^0-9]{0,12}as of\s*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z",
                        qm_bullet[stated.end():stated.end() + 40]), (
            f"§2's queue-maintenance count ({stated.group(1)}) is undated again: it "
            "grows every sweep interval, and the clause's alternative was to drop it")

    UNITS = {"", "s", "ms", "runs", "run", "ok", "of", "items", "notes", "rows",
             "days", "%"}
    DATED = re.compile(r"(?:days to|as of|read|through|until)\s*`?\d{4}-\d{2}-\d{2}")
    bare = re.compile(r"(?<![\w.`/#:-])(\d{3,})(?!\d)([^0-9]{1,3})")
    undated = []
    for block in prose.split("\n\n"):
        for para in block.split("\n- ")[1:]:        # per bullet
            if DATED.search(para):
                continue
            for m in bare.finditer(para):
                if m.group(2).strip() in UNITS:
                    undated.append(m.group(0).strip())
        if not block.lstrip().startswith("- ") and not DATED.search(block):
            for m in bare.finditer(block):
                if m.group(2).strip() in UNITS:
                    undated.append(m.group(0).strip())
    assert not undated, (
        f"undated count(s) in §2 prose: {undated} — every number in a store that is "
        "pruned at 30 days and written continuously is a measurement, and a "
        "measurement with neither a window nor a read stamp is the defect this item "
        "was filed for")


def test_section2_stamps_its_read_and_keeps_the_remeasure_order():
    """Clause 5: the stamp, the window, and the instruction that makes them useful.

    The stamp must not merely exist: a stamp that equals the window end is the old
    bug wearing the fix, because it claims the read happened at the boundary
    instead of saying when it did.
    """
    sec2 = _section(SEC2)
    assert "re-measure before quoting this table" in " ".join(sec2.split()), (
        "§2 lost the instruction that tells a reader the table is a snapshot — the "
        "stamps below are decoration without it")
    assert SECTION2_WINDOW_END in sec2, (
        f"§2's window end moved off {SECTION2_WINDOW_END}, which is the date §6 "
        "and `test_section2_and_section6_report_one_measurement` quote")
    assert SECTION2_SQL_FLOOR in sec2, "§2 lost its SQL floor, so its window is unnamed"
    stamp = re.search(r"read\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})Z", sec2)
    assert stamp, (
        "§2 restated no read stamp — the sentence 're-measure' instructs but does "
        "not date, which is what let the automod-regression zero age thirteen hours")
    assert not stamp.group(1).startswith(SECTION2_WINDOW_END), (
        f"§2's read stamp ({stamp.group(1)}) is the window end, which is a claim "
        "about when the read happened being replaced by a claim about the window")


# ---------------------------------------------------------------------------
# #1783 — the `backlog-cluster` cadence: two gates, and all three prose surfaces
# described one.
#
# `architecture/workers-jobs.md`, the module docstring in
# `workers/sources/backlog_cluster.py` and the `config.yaml` comment above
# `backlog-cluster:` each said the source runs *only* when `clusters.json` is
# older than `min_age_seconds`, which reads as "nightly, once a day". The ledger
# says otherwise: 94 of the 98 `backlog_cluster` rows logged 2026-09-11→29 carry
# `trigger: exhausted`, exactly one is `nightly` (2026-09-19T06:12:40Z), and the
# median gap is 3.13 h against a 20 h gate. The exhausted path is structurally
# sticky — `execute` calls `write_clusters` on every pass, so each rebuild resets
# the very age the gates measure, and an empty clusterable pool therefore never
# reaches 72 000 s. The cadence is correct behaviour (ruled 2026-09-29) and the
# prose was the defect, so each surface is graded in the file a reader actually
# opens, and every duration in the prose is re-derived from the constant or the
# config value it describes: a number may not rot beside the thing it names.
# ---------------------------------------------------------------------------

#: The retired §4 "Wakes" sentence, verbatim. Kept so the bans below have a
#: control: this text trips every one of them, so none is an empty pattern
#: passing on an empty corpus (#1689's rule for a `not in` with no eyes).
RETIRED_CLUSTER_WAKES = (
    "**Wakes** hourly but runs only when `clusters.json` is older than "
    "`min_age_seconds` (20 h), so \"nightly\" is the age of the output rather "
    "than a wall-clock hour and a restart never doubles it up.")

#: The retired `config.yaml` comment, `#` markers stripped, same purpose.
RETIRED_CLUSTER_COMMENT = (
    "\"Nightly\" is the age of the last output: polled hourly, runs when the "
    "file is older than min_age_seconds, so a restart never doubles it up.")

CLUSTER_SUBSEC = "### `backlog-cluster`"


def _subsec(prefix: str) -> str:
    """One `### ` subsection of `workers-jobs.md`, heading line excluded.

    Not `_section`: that cuts at the next `## `, so a §4 subsection would come
    back carrying the remaining seven sources as well, and `exhausted` sitting in
    the `owed-check` or `autocode` prose would then satisfy a cadence claim about
    `backlog-cluster` for free — the free pass #1713 built `_section` to deny.
    """
    text = (ARCH / "workers-jobs.md").read_text(encoding="utf-8")
    assert prefix in text, (
        f"{prefix!r} is not a heading in workers-jobs.md — it moved, and a check "
        "reading the wrong span is worse than one that fails loudly")
    rest = text[text.index(prefix) + len(prefix):]
    ends = [x for x in (rest.find("\n## "), rest.find("\n### ")) if x >= 0]
    return rest[:min(ends)] if ends else rest


def _comment_block_above(key: str, path: Path | None = None) -> str:
    """The `#` comment lines sitting directly above `key:`, markers stripped.

    Block-anchored rather than a whole-file grep: `config.yaml` is thousands of
    comment lines, so `min_age_seconds` occurring anywhere in the file would make
    the two-gate requirement free and the ban on the retired claim unenforceable.
    """
    lines = (path or ROOT / "config.yaml").read_text(encoding="utf-8").splitlines()
    hits = [i for i, ln in enumerate(lines) if ln.strip() == key]
    assert len(hits) == 1, (
        f"{key!r} matched {len(hits)} lines; the anchor reads exactly one, or the "
        "comment graded below is not the one beside the key")
    i = hits[0]
    block: list[str] = []
    while i > 0 and lines[i - 1].lstrip().startswith("#"):
        block.append(lines[i - 1].lstrip().lstrip("#").strip())
        i -= 1
    assert block, f"no comment block directly above {key} — deleted, not corrected"
    return " ".join(" ".join(reversed(block)).split())


def _cluster_cfg() -> dict:
    """The live `backlog-cluster` source block, parsed from `config.yaml`."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    block = cfg["workers"]["sources"]["backlog-cluster"]
    # If a gate key vanished the lookups below raise, which is the point: a test
    # that cannot read the value cannot contradict the prose about it.
    assert "min_age_seconds" in block and "exhausted_min_age_seconds" in block, block
    return block


def _hours(seconds: int) -> str:
    """How all three prose surfaces spell a duration: `20 h`, `2 h`."""
    assert seconds % 3600 == 0, f"{seconds} s is not a whole number of hours"
    return f"{seconds // 3600} h"


def test_workers_jobs_backlog_cluster_names_both_rebuild_gates():
    """Clause 1: §Self-modification → `backlog-cluster` stated one of two gates.

    Both halves are graded, because the fix is a correction and not a deletion:
    the exclusive sentence is gone, AND the gate that actually runs the job is
    stated with its key, its floor and its condition on `select_cluster`. The
    durations are then re-derived from the module's own defaults, so "(2 h)" in
    the prose and `DEFAULT_EXHAUSTED_MIN_AGE_SECONDS = 2 * 3600` cannot drift
    apart without one of them going red.
    """
    from workers.sources import backlog_cluster as SRC

    sec = " ".join(_subsec(CLUSTER_SUBSEC).split())
    old = " ".join(RETIRED_CLUSTER_WAKES.split())
    # Controls before bans: the retired sentence has to trip them, and the
    # extractor has to be looking at a real section.
    assert "runs only when" in old, (
        "the retired sentence no longer trips its own ban, so the check below is "
        "an empty pattern with nothing left to bite")
    assert "exhausted" not in old, (
        "the retired text already named the exhausted gate, so requiring it in "
        "the section would be free")
    assert len(sec) > 400, (
        f"§backlog-cluster extracted only {len(sec)} chars — the span is wrong "
        "and every check below is reading someone else's source")

    assert "runs only when" not in sec, (
        "§backlog-cluster is back to one gate: it also rebuilds a used-up "
        "`clusters.json` at the exhausted floor, which is 94 of 98 ledger rows")
    for live in ("`min_age_seconds` (20 h)", "`exhausted_min_age_seconds` (2 h)",
                 "trigger: nightly|exhausted", "nothing left to take", "floor",
                 "every 2-3 h", "resets its age"):
        assert live in sec, (
            f"§backlog-cluster no longer states {live!r} — the false sentence has "
            f"to be replaced by the true one, not trimmed. Section opens "
            f"{sec[:120]!r}")

    # A run count in this file is a measurement of a store written continuously,
    # so it travels with its window or it goes (#1713's §2 rule, same shape).
    for m in re.finditer(r"\d+ of (?:the |its )?\d+ runs", sec):
        tail = sec[m.end():m.end() + 60]
        assert re.search(r"\d{4}-\d{2}-\d{2}", tail), (
            f"§backlog-cluster states a run count with no date within 60 chars of "
            f"it: {tail!r} — an undated count is the defect #1713 was filed for")

    assert _hours(SRC.DEFAULT_MIN_AGE_SECONDS) == "20 h", (
        "the nightly gate stopped being 20 h, so the doc's '(20 h)' is the stale "
        "half of this node and the prose has to move with the constant")
    assert _hours(SRC.DEFAULT_EXHAUSTED_MIN_AGE_SECONDS) == "2 h", (
        "the exhausted floor stopped being 2 h, so the doc's '(2 h)' is the lie")


def test_config_comment_beside_the_cluster_ages_describes_both_gates():
    """Clause 3: the comment a person reads while editing the two ages.

    The clause's other half — `#`-prefixed lines only, YAML token stream
    byte-identical — is enforced by the gate itself: `scripts/automod/spec.py
    :comment_only_change` moves a `config.yaml` diff into the `comment_only`
    bucket only when `yaml.scan`'s token stream and the parsed document are both
    identical, and refuses it otherwise, so the value the comment sits beside is
    pinned by the rung rather than by a self-vacuating diff check here. What this
    node pins is the durable half: the durations the comment states are the
    durations the parsed keys carry, so moving `exhausted_min_age_seconds`
    without touching the sentence beside it — the drift #1783 exists to end —
    goes red here instead of shipping a comment that misdescribes its own knob.
    """
    from workers.sources import backlog_cluster as SRC

    block = _comment_block_above("backlog-cluster:")
    old = " ".join(RETIRED_CLUSTER_COMMENT.split())
    assert ("polled hourly, runs when the file is older than min_age_seconds" in old
            and "so a restart never doubles it up" in old), (
        "the retired comment no longer trips its own bans, so the two checks "
        "below would pass on any text")

    for retired in ("polled hourly, runs when the file is older than "
                    "min_age_seconds, so a restart never doubles it up",
                    "runs only when"):
        assert retired not in block, (
            f"the comment is back to stating the 20 h gate as the only one "
            f"({retired!r}) — the sentence 94 `trigger: exhausted` rows contradict")
    for live in ("min_age_seconds", "exhausted_min_age_seconds", "select_cluster",
                 "trigger: nightly|exhausted", "2-3 h", "resets"):
        assert live in block, (
            f"the comment no longer names {live!r}: {block[:180]!r} — it has to "
            "describe both gates, not merely both keys")

    cfg = _cluster_cfg()
    assert _hours(cfg["min_age_seconds"]) == _hours(SRC.DEFAULT_MIN_AGE_SECONDS), (
        "config and the source's default no longer agree on the nightly gate, so "
        "one of the three prose surfaces is wrong whatever this node does")
    assert _hours(cfg["exhausted_min_age_seconds"]) == _hours(
        SRC.DEFAULT_EXHAUSTED_MIN_AGE_SECONDS), (
        "config and the source's default no longer agree on the exhausted floor")
    for stated in (_hours(cfg["min_age_seconds"]),
                   _hours(cfg["exhausted_min_age_seconds"])):
        assert stated in block, (
            f"the comment never states {stated}, the value the key beside it now "
            f"holds ({block[:180]!r}) — naming a key without its magnitude is how "
            "7200 becomes 10800 while the prose still says 2 h")


def test_the_cluster_docs_agree_on_the_triggers_and_the_exhausted_floor():
    """Clause 4: `workers-jobs.md` sends the reader to [[automod]] §clustering,
    so the short and the long version have to describe one set of triggers.

    `architecture/automod.md` was already right — it has named both gates and
    `trigger: nightly|exhausted` since 2026-09-14 — so it is the fixed point and
    this round does not edit it. Alignment is graded over the passage that
    carries the exhausted gate in each doc, not over the whole file: automod.md
    spells `exhausted` twice more, about a stalled queue and about a triage cap,
    and either mention would licence a whole-file token check that the cadence
    claim itself could still be wrong under.
    """
    from workers.sources import backlog_cluster as SRC

    jobs = " ".join(_subsec(CLUSTER_SUBSEC).split())
    automod = (ARCH / "automod.md").read_text(encoding="utf-8")
    paras = [p for p in (" ".join(p.split()) for p in automod.split("\n\n"))
             if "exhausted_min_age_seconds" in p]
    assert len(paras) == 1, (
        f"automod.md states the exhausted floor in {len(paras)} paragraphs; this "
        "node grades one passage, and two would let the disagreement hide in the "
        "other half — the §2/§6 split #1713 exists to catch, in a second file")
    long_version = paras[0]

    for shared in ("exhausted_min_age_seconds", "trigger: nightly|exhausted"):
        assert shared in long_version, (
            f"the model doc stopped stating {shared!r}, so the fixed point moved "
            "and both docs have to be re-read before this node means anything")
        assert shared in jobs, f"§backlog-cluster lost {shared!r} again"

    def floor_hours(text: str) -> str:
        m = re.search(r"`exhausted_min_age_seconds` \((\d+) h\)", text)
        assert m, (
            f"a doc names the exhausted floor with no hours figure beside it: "
            f"{text[:140]!r} — the shared number this node compares is gone")
        return f"{m.group(1)} h"

    got = (floor_hours(jobs), floor_hours(long_version))
    assert got[0] == got[1] == _hours(SRC.DEFAULT_EXHAUSTED_MIN_AGE_SECONDS), (
        f"the two docs state different exhausted floors {got}, or neither states "
        f"{_hours(SRC.DEFAULT_EXHAUSTED_MIN_AGE_SECONDS)} — which of the three is "
        "wrong should not be a reader's guess, and #1783 was filed precisely "
        "because it was")


# ── #2077: §8's bench-mine block records a measurement, not a prediction ──────
#
# `architecture/workers.md` §8 closed the bench-mine bullet with "until `SELECT
# count(*) FROM queue WHERE source='bench-mine' AND kind='mine'` is non-zero,
# nothing here has been demonstrated end to end". Production answered that on
# 2026-09-22 and the page never recorded it, while its sibling
# `architecture/workers-jobs.md` did — so the stale caution was the only
# surviving record of a resolved question, which is the failure this file's dated
# counts exist to prevent.
#
# These nodes read `ARCH / "workers.md"` directly and slice the bullet themselves:
# the file's `_section()` helper is hard-wired to `workers-jobs.md`, so reusing it
# here would grade a different document than the one the claim lives in.

#: The bullet's first line, as the doc writes it. Anchored on the leading `- ` so
#: the slice starts at the bullet and not at a mention of the source elsewhere.
_BENCH_MINE_BULLET = "- **`bench-mine` advertised an input"

#: What §8 used to predict, and the prediction's own framing sentence. Both must
#: be gone: the first is the falsified claim, the second is what made it a
#: prediction ("Production has to answer whether the input now fires").
_BENCH_MINE_PREDICTIONS = (
    "nothing here has been demonstrated end to end",
    "Production has to answer whether the input now fires",
)

#: The witness: the 212 `queue` rows for source='bench-mine' AND kind='mine', read
#: out of the live `~/lloyd-data/workers.db` at 2026-10-02T22:18:17Z, one JSON
#: object per row. Committed because a live store has no history: the figures the
#: doc quotes are re-derivable from these bytes after the store has moved on.
BENCH_MINE_WITNESS = ROOT / "tests" / "fixtures" / "workers_bench_mine_2077.jsonl"
VAULT_BENCH_MINE_WITNESS = (Path.home() / "obsidian" / "backlog" / "data"
                            / "2026-10-02.2077-bench-mine-queue-witness.jsonl")


def _bench_mine_block() -> str:
    """§8's bench-mine bullet of `workers.md`, from its first line to the next bullet.

    Sliced on `\n- ` because the bullet's continuation paragraphs are indented, so
    a line-beginning `- ` is only ever the next bullet. The block is long on
    purpose: the run-failure paragraphs clause 4 protects are separate indented
    paragraphs INSIDE this bullet, so a slice that stopped at the first blank line
    could not see what the change was not allowed to touch.
    """
    text = (ARCH / "workers.md").read_text(encoding="utf-8")
    assert "## 8. Known limits" in text, "§8 is gone, so this section's claims moved"
    start = text.index(_BENCH_MINE_BULLET)
    rest = text[start:]
    end = rest.index("\n- ", 10)
    block = rest[:end]
    assert "bench-mine" in block and len(block) > 1_500, (
        f"the slice is {len(block)} chars, which is not the bench-mine bullet — "
        "a slice that catches no prose makes every assert below vacuous")
    return block


def _witness_rows() -> list[dict]:
    raw = BENCH_MINE_WITNESS.read_text(encoding="utf-8")
    rows = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
    assert rows, "the witness fixture is empty, so it can witness nothing"
    return rows


def test_workers_md_no_longer_predicts_the_bench_mine_input_is_unproven():
    """Clause 1: the falsified prediction is out of the file, not just out of §8.

    Whole-file, because the clause counts occurrences in `architecture/workers.md`
    and a prediction that survives a §8 rewrite by moving to §5 is still a
    prediction. Non-vacuity first: the banned strings are asserted non-empty and
    the block is asserted to still be about bench-mine (inside `_bench_mine_block`),
    so the node cannot be satisfied by deleting the bullet.
    """
    assert len(_BENCH_MINE_PREDICTIONS) == 2, "the ban list itself was edited"
    doc = (ARCH / "workers.md").read_text(encoding="utf-8")
    flat = " ".join(doc.split())
    for banned in _BENCH_MINE_PREDICTIONS:
        assert " ".join(banned.split()) not in flat, (
            f"workers.md still predicts {banned!r}, which production refuted on "
            "2026-09-22; the sibling doc carries the dated observation already")

    block = " ".join(_bench_mine_block().split())
    assert "kind='mine'" in block, (
        "the block no longer names the queue rows its claim was about, so the "
        "absence asserted above is a deletion and not a correction")


def test_workers_md_records_the_bench_mine_input_as_a_dated_measurement():
    """Clause 2: the replacement is past tense and carries the date it was measured.

    Three things have to be true at once for a reader to have a measurement rather
    than a prediction: the enqueueing is stated as something that HAS happened, the
    since-date is named, and the figures are stated with their read stamp. A "will"
    or "has to" replacement sentence would satisfy clause 1 and fail this one.
    """
    block = " ".join(_bench_mine_block().split())
    assert re.search(r"has (therefore )?been enqueueing `kind='mine'` rows "
                     r"since 2026-09-22", block), (
        "the block does not state in the past tense that the input has been "
        f"enqueueing since 2026-09-22: …{block[:200]}…")
    for stated in ("212 queue rows", "211 `completed`", "1 `quarantined`",
                   "`2026-09-22T22:03:16.477369+00:00`",
                   "`2026-10-02T21:07:21.183481+00:00`"):
        assert stated in block, f"the measurement lost {stated!r}"


def test_the_dated_bench_mine_counts_sit_beside_a_date():
    """Clause 3: no queue-row or run count in the block is undated.

    Per sentence, not per block: the block is ~50 lines and one dated sentence must
    not license every count in it. `DATED` requires a WINDOW or a READ STAMP attached
    to the count's own sentence (`as of`, `since`, `to`, `read`, `at` before an ISO
    date), which is the same distinction #1713's `test_section2_carries_no_undated_count`
    draws — a date string anywhere is not a measurement date.

    Scope, stated honestly: this guards counts of QUEUE ROWS and RUNS, which is what
    clause 3 names. It does not scan every integer in the bullet, because clause 4
    forbids altering the trace_status-guard sentence, whose "all 77 `error` baseline
    rows" is undated and stays exactly as written — dating it would be the edit the
    same clause prohibits. The new measurement sentence is what clause 3's "adds no
    undated live count" is about, and it is checked here and pinned to its bytes
    below.
    """
    block = _bench_mine_block()
    COUNTED = re.compile(r"\b\d[\d,]*\s+(queue rows|queue row|runs|run)\b")
    DATED = re.compile(r"(?:as of|since|to|read|at|through)\s*`?\d{4}-\d{2}-\d{2}")
    sentences = re.split(r"(?<=[.!])\s+", " ".join(block.split()))
    checked = 0
    for s in sentences:
        if not COUNTED.search(s):
            continue
        checked += 1
        assert DATED.search(s), f"undated count in §8's bench-mine block: {s[:160]!r}"
    assert checked >= 2, (
        f"only {checked} count-bearing sentence(s) were checked: the doc's queue "
        "and run counts moved somewhere this node no longer reads")
    assert "as of 2026-09-19" in " ".join(block.split()), (
        "the pre-fix zero is no longer stated with the date it was read, which is "
        "the only reason it can coexist with the 2026-10-02 measurement")


def test_the_quoted_bench_mine_figures_are_re_derivable_from_the_committed_bytes():
    """Clauses 2 and 3, tied to bytes: the doc's figures ARE the fixture's figures.

    The doc quotes a live store, and a live store has no history — so the numbers
    are read out of the doc and recomputed from the committed extract. Edit either
    side and this fails, which is the only defence against the page drifting back
    into an unfalsifiable claim about a count that grows on every enqueue.
    """
    rows = _witness_rows()
    assert {r["source"] for r in rows} == {"bench-mine"}, "witness rows of another source"
    assert {r["kind"] for r in rows} == {"mine"}, "witness rows of another kind"
    states: dict[str, int] = {}
    for r in rows:
        states[r["state"]] = states.get(r["state"], 0) + 1
    earliest = min(r["enqueued_at"] for r in rows)
    newest = max(r["enqueued_at"] for r in rows)

    block = " ".join(_bench_mine_block().split())
    total = re.search(r"holds (\d[\d,]*) queue rows", block)
    completed = re.search(r"`source='bench-mine' AND kind='mine'`: (\d+) `completed`", block)
    quarantined = re.search(r"(\d+) `quarantined`", block)
    assert total and completed and quarantined, (
        f"the measurement sentence no longer states its total, completed and "
        f"quarantined figures in the checked shape (total={bool(total)}, "
        f"completed={bool(completed)}, quarantined={bool(quarantined)})")
    assert int(total.group(1).replace(",", "")) == len(rows), (
        f"the doc says {total.group(1)} queue rows; the committed bytes hold {len(rows)}")
    assert int(completed.group(1)) == states.get("completed", 0), (
        f"the doc says {completed.group(1)} completed; the bytes hold "
        f"{states.get('completed', 0)}")
    assert int(quarantined.group(1)) == states.get("quarantined", 0), (
        f"the doc says {quarantined.group(1)} quarantined; the bytes hold "
        f"{states.get('quarantined', 0)}")
    assert f"`{earliest}`" in block and f"`{newest}`" in block, (
        f"the doc's earliest/newest stamps are not the bytes' {earliest!r} / {newest!r}")


def test_the_bench_mine_block_keeps_the_caveats_it_was_not_licensed_to_touch():
    """Clause 4: the four protected passages are here, verbatim, unaltered.

    Quoted in full rather than by keyword, because the failure this guards is a
    rewrite that keeps the keywords and drops the evidence — "eight-day gap in
    which no row was appended" is the mtime caveat's whole content, and the
    `trace_status` guard's reason is the `judge_trace` clause, not the word
    `trace_status`. Whitespace-normalised, since the doc wraps at ~80 columns.
    """
    block = " ".join(_bench_mine_block().split())
    protected = (
        'Note too that mtime cannot be read as "the ledger is fresh": a watermark '
        "matched the file's `stat` across an eight-day gap in which no row was "
        "appended, which only proves the inode was touched.",
        "#625 made the comparison case-insensitive (`BM.BASELINE_ID_PREFIX`)",
        "a loser may only be a trial whose `trace_status` is `success`, because "
        "`judge_trace` (`scripts/autoresearch/judge.py`) zeroes the composite of "
        "any trace that did not complete",
        "empty response (stop_reason=max_turns) — nothing written",
        "60 runs since 2026-09-09: 6 success, 5 `skipped`",
        "`bench-mine` 123 failures of 146 runs, **120 of them `stop_reason=max_turns`**",
    )
    for kept in protected:
        assert " ".join(kept.split()) in block, (
            f"§8's bench-mine block lost or reworded a passage this round was not "
            f"licensed to change: {kept[:70]!r}…")
    assert (ROOT / "workers" / "sources" / "bench_mine.py").is_file(), (
        "the source the block describes is gone, so every figure above is about a "
        "module that no longer exists")


def test_the_vault_copy_of_the_bench_mine_witness_is_the_same_bytes_as_the_fixture():
    """Clause 5, the durability half: the vault duplicate is the fixture's bytes.

    The vault holds the copy because the repo's copy is the thing the doc cites and
    the vault is where a report is read from; the two are only one witness if they
    are the same bytes, so the digest is compared and the vault file's presence in
    the vault's own git is asserted. A reader with neither the live store nor the
    repo can still re-derive 212 / 211 / 1 from `backlog/data`.
    """
    assert VAULT_BENCH_MINE_WITNESS.is_file(), (
        f"{VAULT_BENCH_MINE_WITNESS} is not on disk, so §8 cites a witness that "
        "exists only in the repo")
    fixture = BENCH_MINE_WITNESS.read_bytes()
    vault = VAULT_BENCH_MINE_WITNESS.read_bytes()
    assert fixture == vault, (
        f"the vault copy diverged from the fixture: {len(vault)} bytes against "
        f"{len(fixture)}")
    tracked = subprocess.run(
        ["git", "-C", str(Path.home() / "obsidian"), "ls-files", "--error-unmatch",
         "backlog/data/2026-10-02.2077-bench-mine-queue-witness.jsonl"],
        capture_output=True, text=True)
    assert tracked.returncode == 0, (
        "the vault copy is not tracked on the vault's main, so `git log -- ` over "
        "it finds nothing and it has no history either: " + tracked.stderr.strip()[:160])
