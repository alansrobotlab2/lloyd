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

import importlib
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
# #3090 — the other half of the #1769 ruling: what an ABSENT key states.
#
# #1769 ruled twice and only one ruling reached the code. The relabel route is
# pinned by the node above; the second ruling — `measured_against` stays a
# void-only marker and the writer never stamps it — was made 2026-09-30 (the
# owed-check run `20260929_194724_owedcheck_5e0b`), reaffirmed 2026-10-20 (run
# `20261020_135123_owedcheck_bd8d`, the pass that filed #3090), and then lived
# only in #1769's owed ledger. That ledger sits on a `status: done` item, so no
# later round re-derives it, while the staging tree keeps showing a reader what
# looks like a missing field: the live sweep on 2026-10-21 finds every
# with-`task_id` note carrying no `measured_against` at all (296 of them that
# day, and `--dry-run` is the only honest count — a number written here is
# false by the next night, which is the class #705 removed). The fix for a
# reader who reads "missing" is a backfill, and #1769's ruling is that a
# backfill would invent a provenance the trials never had. Graded sentence by
# sentence for the reason the node above gives: a doc-wide `in` can be satisfied
# by an unrelated sentence while the claim itself is the one that went back to
# recommending the opposite.
# ---------------------------------------------------------------------------

def _ma_sentences(doc: str) -> list[str]:
    """The docstring sentences that name `measured_against`, flattened.

    One sentence per claim, because the ruling is two claims — what an absent
    key means, and who may write it — and a check that let a single sentence
    satisfy both would let either half be deleted with the other looking
    untouched.
    """
    return [s for s in re.split(r"(?<=[.!?]) ", doc) if "measured_against" in s]


def test_bench_mine_docstring_says_an_absent_key_means_the_named_task():
    """#3090 clause 1: a staged note with `calibration.task_id` and NO
    `measured_against` was measured against the task its `task_id` names.

    Without this sentence the only statement of the invariant is a closed
    item's ledger line, and the observable in the queue — hundreds of notes with
    a `task_id` and no provenance key — reads as an omission.
    """
    doc = _module_docstring(_BENCH_MINE_SRC)
    assert doc.startswith("bench-mine source"), (
        f"the extractor returned something that is not this module's docstring: "
        f"{doc[:60]!r}")

    inv = [s for s in _ma_sentences(doc)
           if "task_id" in s and "NO ``measured_against``" in s]
    assert len(inv) == 1, (
        f"{len(inv)} sentence(s) state what an absent `measured_against` means "
        "next to `task_id`; exactly one may carry that claim, and zero is the "
        "regression this node exists for — the ruling retreats to #1769's owed "
        "ledger, on an item nobody re-opens")
    assert "measured against the task" in inv[0], (
        f"the sentence no longer names what an unmarked note's mean describes: "
        f"{inv[0]!r}")
    for fact in ("staged", "band verdict", "body"):
        assert fact in inv[0], (
            f"the invariant lost {fact!r}, and each one is a different misread: "
            "without 'staged' it looks like a rule about ledger rows, without "
            "'band verdict' the verdict itself is unaccounted for, and without "
            "'body' a reader can still conclude the mean describes the staging "
            "envelope — which is #1710")


def test_bench_mine_docstring_says_only_the_relabel_script_stamps_the_key():
    """#3090 clause 2: `measured_against` is stamped by
    `scripts/maintenance/relabel_stale_bench_calibration.py` alone, and only to
    void a band verdict — never by this writer.

    The prohibition is the load-bearing half. A docstring that said only "the
    relabel writes it" would still leave a round free to add the key to the
    calibration block "for completeness", which is exactly the writer change
    #1769's ruling refused and #3090's scope excludes.
    """
    doc = _module_docstring(_BENCH_MINE_SRC)
    assert doc.startswith("bench-mine source"), (
        f"the extractor returned something that is not this module's docstring: "
        f"{doc[:60]!r}")

    stamp = [s for s in _ma_sentences(doc) if "relabel_stale_bench_calibration" in s]
    assert len(stamp) == 1, (
        f"{len(stamp)} sentence(s) name the script allowed to stamp the key; one "
        "may own that bound, and zero means the key has no stated owner again")
    s = stamp[0]
    assert "never by this writer" in s, (
        "the writer's own prohibition is gone — that is the sentence a future "
        "'stamp measured_against going forward' round has to overrule, and "
        "#1769's ruling (2026-09-30, reaffirmed by #3090) is no")
    assert "alone" in s and "void" in s, (
        f"the bound needs BOTH the sole writer and the one purpose; 'a marker "
        f"the relabel also uses' still leaves room for a second writer: {s!r}")


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

#: The window §2's table was measured over, as re-read from `workers.db` on
#: 2026-10-29. Clause 5's "window end" is about the END of the prose window, and it
#: is the date `test_section2_and_section6_report_one_measurement` holds equal to the
#: latest window §6 quotes (whose current figures are the 7 days to 2026-10-28); the
#: SQL floor is the other end of the same sentence. The predicate §2 now states is
#: bounded at both ends (`>= '2026-10-22'` AND `< '2026-10-29'`), where the read it
#: replaced had only the `>=` — so these two constants are the whole named window,
#: and a doc that quietly moves one of them without moving its read stamp is what
#: the node below still catches.
SECTION2_WINDOW_END = "2026-10-28"
SECTION2_SQL_FLOOR = "2026-10-22"


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


#: What a count has to be ABOUT before this block demands a date beside it
#: (#3287 widened the noun list). `queue rows`/`runs` are what #2077 wrote this
#: for; `failures` and `failed runs` are what #3287 added, because the sentence
#: it filed — "104 eligible failures in the last 7 days" — named a failure count
#: and no noun here recognised it, so the node ran green over the one unstamped
#: figure in a bullet whose own last line demands a stamp on every figure.
#: `rows` ALONE is deliberately absent. The trace_status-guard sentence's
#: "all 77 `error` baseline rows" is undated on purpose — its date is #625's, not
#: a reading of a live store — and this file forbids editing that sentence, so
#: adding bare `rows` would make the guard demand an edit it is not allowed to
#: ask for. That is a scope limit, not an oversight, and the node below asserts
#: the exclusion is still in force.
_BENCH_COUNTED = re.compile(
    r"\b\d[\d,]*(?:\s+[\w`'-]+){0,2}\s+(queue rows?|runs?|failures|failed runs)\b")

#: A count is dated only by a WINDOW or READ STAMP on its own sentence: `as of`,
#: `since`, `to`, `read`, `at`, `through` immediately before an ISO date. A date
#: string somewhere in the sentence is not a measurement date — #1713's
#: `test_section2_carries_no_undated_count` draws the same line.
_BENCH_DATED = re.compile(r"(?:as of|since|to|read|at|through)\s*`?\d{4}-\d{2}-\d{2}")

#: Sentence boundaries after the block's whitespace is collapsed. Splitting on
#: `.` only is what makes this per-sentence rather than per-block, and the block
#: is full of `…jsonl` — paths whose period is followed by a backtick, not by a
#: space, so they do not end a sentence.
_BENCH_SENTENCE = re.compile(r"(?<=[.!])\s+")


def _bench_mine_count_sentences(block: str) -> tuple[list[str], list[str]]:
    """(count-bearing sentences, of those the undated ones) for a bench-mine block.

    Takes the text rather than reading the page so a caller can hand it a mutated
    block: a guard that can only be exercised on the live tree has never been
    shown to be able to fail, and #3287's whole claim is that this guard stopped
    the next occurrence of a retired sentence. So the retired sentence is fed
    back in below and this function has to report it.
    """
    checked: list[str] = []
    undated: list[str] = []
    for s in _BENCH_SENTENCE.split(" ".join(block.split())):
        if not _BENCH_COUNTED.search(s):
            continue
        checked.append(s)
        if not _BENCH_DATED.search(s):
            undated.append(s)
    return checked, undated


def test_the_dated_bench_mine_counts_sit_beside_a_date():
    """No run, failure or queue-row count in the block goes without a stamp.

    Per sentence, not per block: the block is ~50 lines and one dated sentence
    must not license every count in it.

    Two things changed here for #3287, and both are asserted rather than assumed.
    The noun list now recognises a FAILURE count, and the pattern tolerates up to
    two words between the number and the noun — "104 eligible failures" is the
    shape that walked past the old one-word-adjacent form, which is why an
    unstamped figure sat in this bullet since 2026-09-09 while a node named
    "counts sit beside a date" passed over it. The capability of that widening is
    asserted below against the retired sentence itself, so narrowing the pattern
    again trips this node instead of quietly reopening the hole.

    Scope, stated honestly, is unchanged in one respect: `rows` is still not a
    counted noun, because the trace_status-guard sentence's "all 77 `error`
    baseline rows" is undated on purpose and this file forbids editing that
    sentence. That is asserted too, so the exclusion is a checked fact and not a
    regex accident waiting for someone to simplify the alternation.
    """
    block = _bench_mine_block()
    retired = ("gated by nothing the ledger does — 104 eligible failures "
               "in the last 7 days.")
    assert _BENCH_COUNTED.search(retired), (
        "`_BENCH_COUNTED` no longer recognises a count with a word between the "
        f"number and the noun ({retired!r} is the sentence #3287 was filed over), "
        "so the widening is gone and an unstamped failure count can pass this "
        "node again")

    checked, undated = _bench_mine_count_sentences(block)
    assert not undated, (
        "undated count in §8's bench-mine block: "
        + " | ".join(s[:160] for s in undated))
    assert len(checked) >= 2, (
        f"only {len(checked)} count-bearing sentence(s) were checked: the doc's "
        "queue, run and failure counts moved somewhere this node no longer reads")
    assert "as of 2026-09-19" in " ".join(block.split()), (
        "the pre-fix zero is no longer stated with the date it was read, which is "
        "the only reason it can coexist with the 2026-10-02 measurement")

    seventy_seven = "all 77 `error` baseline rows"
    assert seventy_seven in " ".join(block.split()), (
        "the trace_status-guard sentence lost its undated `rows` figure, which is "
        "the one number in this bullet this file is not allowed to touch")
    assert not any(seventy_seven in s for s in checked), (
        "`rows` alone became a counted noun, so this node now demands a date "
        "beside a figure whose clause forbids dating it")


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


# ── #2309: `failure-ledger` is scheduled, and §6 names what it is measured on ──
#
# #2079 registered the source without giving it a `workers.sources` block, so
# `WorkerPool._scheduler_pass` read `{}` for it (`workers/pool.py:1176`) and stepped
# past it at `:1177` without calling `enqueue_if_due`. Three days passed with the sweep
# running only inside its own tests, and nothing here could have caught it: the two
# nodes that touch the roster pin one row per registered source and config ⊆ registry,
# and BOTH hold for a registered-but-unconfigured source. The rail was missing the
# other direction, which is the direction the fleet actually fell.


def _src_cfg() -> dict:
    """`workers.sources` from `config.yaml`, parsed from disk.

    Not the live `CONFIG` mapping: nodes in this file compare prose to the file a
    round commits, and `CONFIG` is mutated in place by the runtime toggles other tests
    exercise, so a value read off it is a measurement of whichever test ran last.
    """
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    return cfg["workers"]["sources"]


def _roster_row(name: str) -> list[str]:
    """§1's row for `name`, cells stripped. Index: 1 family, 2 prio, 3 cadence,
    4 inflight, 5 runner, 6 IV, 8 **on**."""
    text = (ARCH / "workers-jobs.md").read_text(encoding="utf-8")
    for ln in text.splitlines():
        if ln.startswith(f"| `{name}` |"):
            return [c.strip() for c in ln.strip().strip("|").split("|")]
    raise AssertionError(f"§1 of workers-jobs.md has no `{name}` row")


def _failure_ledger_para() -> str:
    """§6's scheduling block: #2079's closing entry, from its bold lead-in on.

    Spans both paragraphs of the entry — the one that says the block landed and names
    the denominator, and the one that says what decides the judgement — and stops at
    the next `### `, if any (this is §6's last entry). Grading one paragraph at a time
    here would let a claim move between the two and dodge its own node.
    """
    six = _section(SEC6)
    rest = six[six.index("**Registered, and scheduled"):]
    end = rest.find("\n### ")
    return rest if end < 0 else rest[:end]


def test_the_failure_ledger_source_ships_a_block_the_scheduler_reads():
    """#2309 clause 1: registered AND configured, at the values pool.py consumes.

    Every key is pinned to the line that reads it, because a block whose keys nothing
    reads is a comment: `enabled` and `interval_seconds` at `workers/pool.py:1176-1179`
    (the skip, then the poll), `min_interval_seconds` at
    `workers/sources/failure_ledger.py:228` (the source's own watermark), `max_inflight`
    at `workers/pool.py:1333`, `max_duration_seconds` at `:1376`.

    The registry-symmetry half runs on `get_sources_config()` — the live mapping the
    scheduler iterates — because that is the exact expression the item names as its
    check. The reverse direction reads the registry against the file, like
    `test_config_configures_only_registered_sources` above it.
    """
    assert not set(sources.SOURCE_REGISTRY) - set(sources.get_sources_config()), (
        "registered sources with no `workers.sources` block: `_scheduler_pass` reads "
        "`{}` for each and steps past it without calling `enqueue_if_due` "
        f"(workers/pool.py:1176-1177) — the #2079 shape: "
        f"{sorted(set(sources.SOURCE_REGISTRY) - set(sources.get_sources_config()))}")

    blk = dict(_src_cfg().get("failure-ledger") or {})
    assert blk.get("enabled") is True, (
        "the block exists but `enabled` is not true, so workers/pool.py:1177 still "
        f"steps past the source: {blk or 'no block at all'}")
    assert blk.get("interval_seconds") == 3600, blk
    assert blk.get("min_interval_seconds") == 86400, blk
    assert blk.get("max_inflight") == 1, blk
    assert blk.get("max_duration_seconds") == 900, blk
    assert blk.get("max_turns") == 12, blk
    assert "model" not in blk, (
        f"a `model` key joined the block: {sorted(blk)}. The handler hands its prompt "
        "to `run_prompt_on_primary`, whose signature takes a prompt, a turn ceiling "
        "and a source name and no model, so the key would name an engine nothing "
        "reads — a knob attached to nothing is what this file keeps refusing.")

    body = (ROOT / "workers" / "sources" / "failure_ledger.py").read_text(
        encoding="utf-8")
    assert "run_prompt_on_primary" in body, (
        "the handler no longer goes direct to the primary, which is the only reason "
        "this block carries no `model` key")
    assert re.search(r"\bmodel\s*=", body) is None, (
        "the handler started passing a model, so the absent `model` key is a gap "
        "rather than a fact about the call")
    for key in ("min_interval_seconds", "max_turns"):
        assert f'src_cfg.get("{key}"' in body or f'payload.get("{key}"' in body, (
            f"nothing reads `{key}` any more, so config.yaml's comment about it is "
            "the claim and the code is the decoration")

    # Two of the six values restate a module default rather than overriding it, and
    # config.yaml's comment says so in those words. A default that moves while the
    # block stands makes that comment the first thing to break, so the pair is pinned
    # rather than left for the next reader's diff to explain.
    from workers.sources import failure_ledger as fl

    assert blk["min_interval_seconds"] == fl.DEFAULT_INTERVAL_SECONDS, (
        f"the block says {blk['min_interval_seconds']}s while "
        f"DEFAULT_INTERVAL_SECONDS is {fl.DEFAULT_INTERVAL_SECONDS}: whichever drifted "
        "left the other one lying, and config.yaml claims they are the same number")
    # `max_turns` is the key that does NOT reach the turn today: `execute` reads it off
    # the queue payload (`payload.get("max_turns")`) and `sweep_and_dispatch` writes no
    # such field, so the ceiling in force is the module default. Pinning config to that
    # default is what stops the key advertising a number that is not what runs; wiring
    # the key through is recorded on #2309, not done here.
    assert blk["max_turns"] == fl.DEFAULT_MAX_TURNS == 12, (
        f"config says max_turns {blk['max_turns']} while the ceiling in force is "
        f"DEFAULT_MAX_TURNS {fl.DEFAULT_MAX_TURNS}")
    assert 'payload.get("max_turns")' in body, (
        "`execute` no longer reads the turn ceiling off the payload, so the comment "
        "above this assert and the one in config.yaml describe a wire that is not "
        "there — either wire it or delete the key")


def test_the_failure_ledger_roster_row_reads_scheduled():
    """#2309 clause 2: §1's row, its cadence column, and the prose above it.

    Three surfaces carried the same fact, so three asserts: the **on** column, the
    cadence column, and the paragraph that explained to a reader WHY the column said
    **no** — a paragraph that outlives its own reason for existing reads as true to
    the next person, which is the hardest kind of false to notice.

    The cadence column is compared to the configured poll in SECONDS, because the one
    thing that must not happen again is the table and the block disagreeing while each
    looks fine alone. `test_section2_names_the_live_turn_budget_keys` is this file's
    precedent for the same shape: a doc that names a number is a claim about a file.
    """
    one = _section("## 1. The roster")
    # Both count sentences are pinned as a PAIR on the same word, and the word is
    # the registry's size. `cost-investigate` (#2510, 2026-10-10) moved both from
    # fifteen to sixteen, which is the point: a new source has to be said in the
    # doc twice, in the same breath, before a node will let the number stand.
    assert "sixteen have a block under" in one, (
        "§1 no longer states that every registered source has a block, which is what "
        "is now true: " + one.strip().splitlines()[0][:90])
    for gone in ("Fourteen of them have a block", "no config block"):
        assert gone not in one, f"§1 still carries the retired claim {gone!r}"
    assert "Sixteen sources are registered." in one, (
        "the count sentence changed its shape, and "
        "test_workers_jobs_counts_and_rosters_every_registered_source in "
        "test_automod_doc_claims.py parses it out of §1's first line")

    assert _src_cfg()["failure-ledger"]["enabled"] is True
    row = _roster_row("failure-ledger")
    assert row[8] == "yes", (
        f"the **on** column still says {row[8]!r} for a source config enables. The "
        "column answers 'does it run on a timer', and it does now")
    cadence = row[3]
    assert "no config block" not in cadence and "skips" not in cadence, cadence
    assert _hours(int(_src_cfg()["failure-ledger"]["interval_seconds"])) in cadence, (
        f"the cadence column {cadence!r} does not state the configured poll in the "
        "form every other row uses — §1 and config.yaml disagree about how often the "
        "pool asks, and each reads fine alone")
    assert "one sweep a day" in cadence, (
        f"{cadence!r} lost the half that matters most here: the pool asks hourly and "
        "`min_interval_seconds` keeps the sweep itself to one a day, so a row stating "
        "only the poll reads as a 24x increase in ledger re-reads")


def test_section6_stops_saying_the_failure_ledger_needs_a_person():
    """#2309 clause 3, and the replacement for the excuse that was never true.

    The retired paragraph gave three reasons nothing ran the sweep: no block,
    "`config.yaml` is outside what the automod loop may write", and the guardian not
    owning the job (#2080). The first is now false on this tree. The second was
    retired on 2026-10-05 by the `config_value_change` lane, and this round is that
    lane's first use for a `workers.sources.*` subtree — which made that sentence the
    falsest one in the file: it told the next reader the work needed a person when the
    loop had long been allowed to do it.

    The bans therefore run over all of §6, not just the paragraph: a deleted excuse
    explains nothing either. And the replacement is checked against the fence it says
    is absent — `spec.py` is read, not quoted.
    """
    six = _section(SEC6)
    assert "**Registered, and scheduled" in _failure_ledger_para(), (
        "§6 lost the paragraph that says the source runs on a timer")
    for gone in ("not scheduled", "unscheduled", "no config block", "scheduler skips",
                 "outside what the automod loop may write", "awaiting a person"):
        assert gone not in six, f"§6 still says {gone!r}, which this change falsified"
    para = _failure_ledger_para()
    assert "config_value_change" in para and "CONFIG_DENIED_KEYS" in para, (
        "the paragraph no longer names the lane that may write this block and the "
        "fence it cannot cross, so the retirement left an absence, not a replacement")

    from scripts.automod import spec

    assert "workers.sources.failure-ledger" not in spec.CONFIG_DENIED_KEYS, (
        "the fence now covers this subtree, so §6's new sentence is the false one")
    assert spec._config_denied_prefix("workers.sources.failure-ledger.enabled") is None, (
        f"the prefix fence denies the block's own `enabled` leaf: "
        f"{spec._config_denied_prefix('workers.sources.failure-ledger.enabled')}")
    assert "failure-ledger" not in spec.CONFIG_LOOP_SOURCES, (
        "the source became one of the loop's own three, whose control leaves are "
        "fenced by LEAF rather than prefix — a different rule than the one §6 states")


def test_section6_names_the_runs_table_as_the_effectiveness_denominator():
    """#2309 clause 4: the denominator, and the literals this edit had to spare.

    #2079 owed entry 4. "How much did the ledger do" has three candidate measures —
    events counted, queue rows, run rows — and `app/failure_ledger.py` already picks
    one in `reconcile()`'s own docstring: "`runs_total`: a queue row is an intention, a
    run row is a thing that happened". §6 promised an effectiveness read while naming
    no denominator, so the reader reaches for the count a sweep can inflate by running.

    The rest of this node is a keep-list, because §6 is also the file's only measured
    history: `42%`, `82%` and the literal-at-the-call-site sentence have their own
    nodes, and restating them here is what makes THIS node the whole clause. The
    §2/§6 window equality is why `test_section2_and_section6_report_one_measurement`
    exists — one edit must not move one of the two dates.
    """
    six, para = _section(SEC6), _failure_ledger_para()
    for frag in ("runs_total", "runs.status", "workers.db", "'failure-ledger'"):
        assert frag in para, f"the denominator paragraph never names {frag}"
    ledger = (ROOT / "app" / "failure_ledger.py").read_text(encoding="utf-8")
    assert "FROM runs WHERE source = ? GROUP BY status" in ledger, (
        "`reconcile()` stopped reading the runs table, so the denominator §6 names is "
        "no longer the one the code computes")
    assert '"runs_total"' in ledger, "reconcile() no longer reports runs_total"
    for keep in ("42%", "82%", "started with a literal at the call site"):
        assert keep in six, f"§6 lost {keep!r}, which is measured history"
    # The LATEST window each section quotes, which is the rule
    # `test_section2_and_section6_report_one_measurement` uses and the only correct
    # one: §6 is the dated history, so it legitimately quotes several windows (its
    # earliest is the 7 days to 2026-09-11 that `42%`/`82%` belong to) and a
    # first-match compare would fail on a doc that is not wrong.
    win = r"[Dd]ays to (\d{4}-\d{2}-\d{2})"
    two, sixwin = re.findall(win, _section(SEC2)), re.findall(win, six)
    assert two and sixwin and max(two) == max(sixwin), (
        f"§2's table is measured to {max(two) if two else '?'} and §6's current "
        f"figures to {max(sixwin) if sixwin else 'nothing'}: one doc, two "
        "measurements — and this edit added a dated sentence to §6, so it is the "
        "likely cause")


def test_section6_names_the_day_growth_reviews_and_what_the_watermark_means():
    """#2893 clause 5: two sentences §6 never had, each checked against the code.

    The missing one that cost the bursts: §6 said the sweep runs once a day, which is
    true, and never said WHICH day it counts — and that is the half that decides
    whether a detector can fire at all. The missing one about the watermark:
    `last_enqueued` reads like "the last row enqueued" and is in fact stamped after
    every sweep, so a reader who trusts the name concludes a source that never
    dispatched never ran.

    Nothing here greps for a phrase and stops. The lag the doc states is compared to
    `GROWTH_REVIEW_LAG_DAYS`, the helper it names is called, and the source file is
    read to confirm the call really passes the shifted day AND the pass's own day as
    the stamp — the pair the paragraph says cannot move apart.
    """
    sec = _section(SEC6)
    sub = sec[sec.index("### `failure-ledger`"):]
    from app import failure_ledger as app_fl

    assert "growth_review_day" in sub and "GROWTH_REVIEW_LAG_DAYS" in sub, (
        "§6 stopped saying which day growth counts, which is the fact that decides "
        "whether the detector can ever see a burst")
    lag = re.search(r"`GROWTH_REVIEW_LAG_DAYS`[^\d\n]{0,8}(\d+)", sub)
    assert lag and int(lag.group(1)) == app_fl.GROWTH_REVIEW_LAG_DAYS, (
        "§6 no longer states the review lag with the number the code uses "
        f"(code: {app_fl.GROWTH_REVIEW_LAG_DAYS})")
    assert app_fl.growth_review_day("2026-10-03") == "2026-10-02", (
        "the helper the doc names no longer returns the day before the pass")

    src = (ROOT / "workers" / "sources" / "failure_ledger.py").read_text(
        encoding="utf-8")
    one_line = re.sub(r"\s+", " ", src)
    assert "fl.detect_growth(conn, day=fl.growth_review_day(day), detected_on=day)" \
        in one_line, (
        "the pass no longer counts the day before and stamps its own, which is the "
        "pair §6's bullet describes: count D-1, stamp D, or dispatch never sees it")
    assert "WHERE detected_on = ?" in sub, (
        "§6 lost the reason the two days cannot move apart")

    assert "last_enqueued" in sub and "last sweep" in sub, (
        "§6 stopped saying what `last_enqueued` actually records")
    assert "last_enqueue_ok" in sub, (
        "§6 lost the pointer to the stamp that DOES say a pass completed")
    assert one_line.rindex("sweep_and_dispatch, queue=queue") < one_line.index(
        "queue.wm_set"), (
        "the source no longer stamps `last_enqueued` after the sweep returns, so the "
        "sentence saying it means 'last swept' is the one that needs rewriting")

    # §6's two `app/failure_ledger.py:` line citations moved with this edit
    # (`detect_growth` grew a `detected_on` parameter and the constant block above it
    # grew the review-day rule), and a citation nobody re-checks is the kind that
    # reads authoritative while pointing at a comment. Each is compared to the line
    # the code actually has.
    ledger_src = (ROOT / "app" / "failure_ledger.py").read_text(encoding="utf-8")
    lines = ledger_src.splitlines()

    def _cited(name: str) -> int:
        m = re.search(rf"`app/failure_ledger\.py:(\d+)` `{name}", sub)
        assert m, f"§6 no longer cites `app/failure_ledger.py:<line>` for {name}"
        return int(m.group(1))

    assert "def reconcile(" in lines[_cited("reconcile()") - 1], (
        f"§6 cites line {_cited('reconcile()')} for reconcile(), which is now "
        f"{lines[_cited('reconcile()') - 1]!r}")
    cited_runs_total = re.search(r"reports `runs_total` \(`:(\d+)`\)", sub)
    assert cited_runs_total and '"runs_total"' in lines[int(cited_runs_total.group(1)) - 1], (
        "§6's `runs_total` line citation no longer lands on the dict key it names")


def test_section6_states_both_growth_arms_and_the_module_stopped_denying_them():
    """#3281 clause 5: the bar §6 never stated, and the comment that denied the new one.

    §6's growth bullet explained WHICH day is counted and never said what the count is
    compared against, so a reader could not see that a family with no history was being
    thrown away. The module comment and `median_active`'s docstring DID say so, and
    called it the design ("never a trip") — prose that reads as a reason for behaviour
    the code has since dropped is the failure this file exists for.

    No assert here stops at a phrase. The two numbers the new bullet states are read
    back off the constants they name, the arm is checked in the code that has to
    implement it (a `median is None` branch that EMITS, and no `median or 0`, which is
    the coercion the bullet forbids), and the retired sentence is checked absent from
    the module rather than merely unfavourable.
    """
    from app import failure_ledger as app_fl

    sub = _section(SEC6)
    sub = sub[sub.index("### `failure-ledger`"):]
    assert "two arms" in sub, (
        "§6 stopped saying growth has two arms, which is the fact that decides whether "
        "a burst from silence is a finding or a discard")

    bullet = sub[sub.index("**What the counted day is compared against"):]
    mult = re.search(r"`GROWTH_MEDIAN_MULTIPLE`[^\d\n]{0,8}(\d+(?:\.\d+)?)", bullet)
    assert mult and float(mult.group(1)) == app_fl.GROWTH_MEDIAN_MULTIPLE, (
        f"§6's growth bullet states no multiple the code uses (code: "
        f"{app_fl.GROWTH_MEDIAN_MULTIPLE})")
    minimum = re.search(r"`GROWTH_MIN_COUNT`[^\d\n]{0,8}(\d+)", bullet)
    assert minimum and int(minimum.group(1)) == app_fl.GROWTH_MIN_COUNT, (
        f"§6's silent-baseline arm states no floor the code uses (code: "
        f"{app_fl.GROWTH_MIN_COUNT})")
    assert "`GROWTH_WINDOW_DAYS`" in bullet, (
        "§6 lost the width of the window the median is taken over")
    for frag in ("median_active", "burst-from-silence", "median coerced to zero"):
        assert frag in bullet, f"§6's growth arm no longer names {frag!r}"

    ledger = (ROOT / "app" / "failure_ledger.py").read_text(encoding="utf-8")
    assert "never a trip" not in ledger, (
        "the comment that called the discard the design is back: an empty trailing "
        "window is an arm, not a reason to stay silent")
    growth_src = ledger[ledger.index("def detect_growth("):]
    growth_src = growth_src[:growth_src.index("\n\ndef ")]
    assert "if median is None:" in growth_src, (
        "the silent-baseline arm is no longer a branch of its own, so the doc's second "
        "arm has nothing behind it")
    none_arm = growth_src[growth_src.index("if median is None:"):]
    assert 'kind="growth"' in none_arm[:none_arm.index("elif")], (
        "the None branch stopped emitting a growth finding")
    # The coercion is named in the module and in §6 precisely in order to forbid it, so
    # what gets checked is WHERE it is named: a `median or 0` on a non-comment line is
    # the threshold every family trips on, not a warning about one.
    for ln in ledger.splitlines():
        if "median or 0" in ln:
            assert ln.strip().startswith("#"), (
                "the coercion the doc forbids became executable code: "
                f"{ln.strip()!r}")
    assert "median or 0" in bullet, (
        "§6 stopped naming the coercion it forbids, so the next reader has no reason "
        "to keep the two arms separate")


def _deciding_paragraph() -> str:
    """§6's deciding-criterion paragraph, flattened to one line.

    Markdown wraps a paragraph across lines and the count/date pairing this file's
    rails check is a SENTENCE property, so prose that wraps mid-sentence reads as
    several dateless fragments. The one existing node that reads this paragraph
    (`test_section6_names_the_day_growth_reviews_and_what_the_watermark_means`)
    avoids the problem by substring-matching; a node that matches per-sentence
    cannot. Joining lines is the same normalisation the skill's `_flat` does on the
    vault side.
    """
    sec6 = _section(SEC6)
    start = sec6.find("**What decides it")
    assert start != -1, (
        "§6 no longer holds a 'What decides it' paragraph — either it was renamed or "
        "the failure-ledger criterion moved, and the node that reads this one has to "
        "say which before its assertions mean anything")
    para = sec6[start:].split("\n\n")[0]
    return " ".join(para.split("\n"))


#: The three strings that made the retired §6 claim what it was. #3532's own
#: triage records them verbatim, and `git log -S"no pass can review it now"` returns
#: exactly one sha — HEAD's 08bb1eee, written by the arch-review job itself. That is
#: the self-refiling loop: arch-review writes "no arrival condition" out of the same
#: 0-runs measurement, a later triage reads it as an incident, and #3529 (high) is
#: what came of it on 2026-10-28.
RETIRED_REACHABILITY = ("neither detector can reach this store",
                        "no pass can review it now",
                        "no arrival condition")


#: Count nouns this store is actually measured in. The §2 rail's alternation at
#: `test_section2_carries_no_undated_count` (:1290) omits `events`, `event days` and
#: `signatures`, which is why the first version of this round's node read green while
#: an undated `214 events` clause sat in the same paragraph — a guard that cannot see
#: its own defect is the vacuity this file already names at :2433. Broadening is
#: deliberate and one-directional: §2's rail stays exactly as #1713 left it, and only
#: this paragraph is policed by the wider set.
#: A figure that was READ out of a store. A threshold is not one: `>=3 events` and
#: `at least 4 signatures` state the rule's bar, which lives in the source and does not
#: move between reads, so stamping them would train the reader to expect a measurement
#: where there is none. The two lookbehinds are what keeps the floor's own figures out
#: of this rail without weakening it for the counts that do move.
_COUNTED_FIGURE = re.compile(r"(?<![\u2265\u2264<>])(?<!least )\b\d[\d,]*\s*"
                             r"(?:`?rows`?|`?runs`?|`?failures`?|`?events`?"
                             r"|`?event days?`?|`?signatures`?|signatures|notices?"
                             r"|signature-days?)")

#: A date attached to a word that says the figure was READ. Two refusals produced
#: this form. The first added a looser `_DATED` alternative to the pairing, and
#: `_STAMPED` ⊂ `_DATED`, so the conjunction reduced to the loose test and the strict
#: one never ran. The second refused the loose prepositions this first draft accepted
#: (`at`, `on`, `since`, `through`), because "ruled on 2026-10-27T03:03Z" or any
#: incidental `on <date>` then qualified as a stamp on a count it is not attached to,
#: and a mutation that deleted the real stamp left the sentence still passing. So:
#: reading verbs only, a word boundary in front so `utton 2026-…` cannot qualify, and
#: the seconds group present so this and the mutation below cannot disagree.
_STAMPED = re.compile(r"\b(?:measured|read|counted|as of)\s`?20\d\d-\d\d-\d\d"
                      r"(?:T\d\d(?::\d\d){1,2}(?:Z|[+-]\d\d:\d\d)?)?", re.I)


def _undated_counts(flat: str) -> tuple[list[str], list[str]]:
    """(count-bearing sentences, of those the unstamped ones) for one paragraph.

    A function, not inline asserts, because the same pairing has to run over the
    spliced paragraph in the guard node below: claiming the date rail is the thing
    that separates the two wordings, while testing the splice with local substring
    checks, asserts the claim transitively and not at all.

    The predicate is `_STAMPED` ALONE. The first version ended `and not
    _DATED.search(s)`, which the review rung refused on 2026-10-29 because
    `_STAMPED` ⊂ `_DATED`: the conjunction reduces to the looser test, so the strict
    regex was decorative and any sentence carrying ANY ISO date passed — including
    one whose date belongs to a different fact. `_DATED` stays in this file for §2
    and §3, which are content to accept a bare date; a count of a store that grows
    needs the word that says it was read.
    """
    checked = [s for s in _BENCH_SENTENCE.split(flat) if _COUNTED_FIGURE.search(s)]
    unpaired = [s for s in checked if not _STAMPED.search(s)]
    return checked, unpaired


def _armed_claims(flat: str) -> list[str]:
    """The forward-looking claims §6's deciding paragraph must make, as failures.

    Empty means every claim is present. Returns a list rather than asserting because
    the guard node has to run these SAME claims over a paragraph with the retired
    sentence welded back into it: the finding #3532 rests on is that nothing but the
    retired-phrase check separates the two wordings, and that can only be shown by
    grading the spliced text with the live node's own checks.
    """
    bad: list[str] = []
    if not re.search(r"qualifying[^.]{0,70}since[^.]{0,70}(?:arm|landed)", flat, re.I):
        bad.append("no 'nothing qualifying has arrived since the arm landed' claim, "
                   "which is the sentence that separates 'quiet' from 'cannot fire'")
    arrival = [s for s in _BENCH_SENTENCE.split(flat)
               if "GROWTH_MIN_COUNT" in s and "detect_growth" in s]
    if not arrival:
        bad.append("no sentence names both GROWTH_MIN_COUNT and detect_growth, so the "
                   "arrival condition is a mood rather than a checkable rule")
    else:
        for pat, why in ((r"≥\s*3|>=\s*3|\bthree\b", "the floor is not stated as a number"),
                         (r"silent", "the arm that fires is not named, so which of the "
                                     "two growth arms is meant stays unsaid"),
                         (r"trailing[- ]?7", "the empty-trailing-7-day precondition is "
                                             "gone, and that quiet window is the arm")):
            if not re.search(pat, arrival[0], re.I):
                bad.append(why)
    d1 = [s for s in _BENCH_SENTENCE.split(flat) if "GROWTH_REVIEW_LAG_DAYS" in s]
    if len(d1) != 1:
        bad.append(f"{len(d1)} sentences cite GROWTH_REVIEW_LAG_DAYS; the constraint "
                   f"is stated once here, and a second copy is where a forward-looking "
                   f"claim hides")
    else:
        # The first version's regex required the negation to sit BEFORE the word
        # `burst` and a fixed verb phrase after it, so it allowed exactly what it
        # existed to catch: "a burst that closed before <date> can never be caught",
        # and the live bounded sentence survived only because it spells `re-reviewed`
        # where the pattern wanted `be reviewed`. What makes a forever-claim legal here
        # is a bound — a named day the unreachable window is pinned to — so that is
        # what the check requires beside each negation, in either order.
        for m in re.finditer(r"\b(cannot|can't|never|unreachable|impossible|forever"
                             r"|no|not)\b", d1[0], re.I):
            win = d1[0][max(0, m.start() - 70):m.end() + 70]
            if not re.search(r"burst|arrival|arrives|qualifying day|re-?review\w*"
                             r"|reviewed|caught|reachab", win, re.I):
                continue
            if re.search(r"(?:closed|settled|ended|before|through|up to)\s*`?"
                         r"20\d\d-\d\d-\d\d", win, re.I):
                continue
            bad.append(f"the D-1 sentence carries {m.group(0)!r} beside a burst/review "
                       f"word with no named day bounding it — a forward-looking "
                       f"unreachability claim, which is the retired premise with one "
                       f"word changed: ...{win[:170]!r}...")
    return bad


def test_section6_dates_the_counts_behind_the_shape_not_seen_zero():
    """Clause 4 (with clause 3's quiet): every count in the rewritten paragraph
    carries the date it was read, and the zero has its cause named beside it.

    §2's rail at `test_section2_carries_no_undated_count` exists because a count of a
    live store is a claim about a moment, not a property of the file — that node
    records 9 of 12 §6 counts undated. So the *absence* of a stamp is the defect, and
    the node has to be able to see one: the first version asked only sentences
    containing the literal `all-time` for their dates, and the review rung proved by
    mutation that deleting the dates from the nine-signature-days sentence left it
    green — a rail wired to one phrase polices one sentence, not a paragraph. This
    version pairs every count-bearing sentence in the block, over the store's own
    count nouns, and proves both directions with a splice.

    A stale count is still possible after this: no test can query the live store (the
    skill's own rule — a green suite that asserts about a runtime store is a claim
    with no evidence), so the paragraph's numbers are pinned by their date stamps and
    the store-side recount is what catches one going stale.
    """
    para = _deciding_paragraph()
    flat = " ".join(para.split())

    # clause 3: the quiet, stated outright, with its cause — not its mechanism.
    quiet = [s for s in _BENCH_SENTENCE.split(flat)
             if re.search(r"no qualifying (?:burst|day)[^.]{0,70}(?:since|landed)", s, re.I)]
    assert quiet, (
        "the paragraph no longer says outright that the store has produced no "
        "qualifying burst since the arm landed — the one sentence that separates "
        "'quiet' from 'cannot fire', which is the whole of #3532")
    assert "2026-10-21" in flat and "2026-10-25" in flat, (
        "the quiet is asserted without the two dates that bound it (the last "
        "qualifying day, and the day the arm landed), so a reader cannot tell it "
        "from a stall")
    # The last qualifying day, dated by its OWN reading verb. The refused version
    # paired the date with `_DATED` — the bare ISO regex the date satisfies by
    # construction, inside a sentence an earlier assert had already required the
    # literal in — so the element could not go red. This one asks for a word saying
    # the figure was READ, then shows the ask bites by deleting that stamp.
    last_day = [s for s in _BENCH_SENTENCE.split(flat)
                if "2026-10-21" in s and _STAMPED.search(s)]
    assert last_day, (
        "the last qualifying day 2026-10-21 is named by no sentence carrying a stamp "
        "word attached to a date, so the figure has no stated reading time — an "
        "undated count of a store that gains rows daily, which is the §2 rail's defect "
        "sitting in the paragraph that replaced the retired claim")
    unstamp = _STAMPED.sub("counted once", last_day[0], count=1)
    assert unstamp != last_day[0], (
        f"`_STAMPED` matched nothing in {last_day[0][:170]!r}, so the assert "
        "above is being satisfied by something that is not a reading verb on a date")
    assert not _STAMPED.search(unstamp), (
        "deleting the count's own stamp left the same sentence still `_STAMPED`, so "
        "what the assert above really pairs the date with is a date attached to some "
        "other fact in that sentence — the refused shape wearing a stricter regex")

    # the zero, and why it is a zero.
    zero = [s for s in _BENCH_SENTENCE.split(flat) if re.search(r"0 `?runs`?|0 runs", s)]
    assert zero, (
        "the 0-runs/0-queue measurement left the paragraph, and #3281's seven-day "
        "criterion needs that denominator stated to be judged against")
    assert any(re.search(r"no qualifying (?:burst|day)|shape not seen", s, re.I)
               for s in zero), (
        "the 0 rows are stated without their cause, so the paragraph invites the "
        "reader to attribute them to the detector — the inference that filed #3529 as "
        "a high incident out of a store that had merely been quiet")
    assert any("all-time" in s for s in zero), (
        "the 0-runs sentence no longer says its count is all-time, so a reader "
        "arriving after one dispatch will take the seven-day criterion as met")

    # The live authority on the residual zero, by name and with its next date: this
    # is what lets a later reader tell 'the mechanism cannot fire' (retired) from
    # 'the shape has not been seen yet, and someone is watching' (true).
    assert "#3281" in flat and "2026-11-03" in flat, (
        "the paragraph stopped pointing at #3281's owed entry (recheck 2026-11-03). "
        "It is the live authority on this zero, and #3532's point is that the answer "
        "is already on the record — owed-check ruled it at 2026-10-27T03:03Z")

    # clause 5: the pairing, over the WHOLE paragraph, on the store's own nouns.
    checked, unpaired = _undated_counts(flat)
    # Two, not the three the paragraph happened to hold on the day this was written:
    # a floor set at the current count reddens the node for any reword that folds a
    # count noun into a sentence — the same error as testing a rate by its own number,
    # which is what the review rung flagged here. Two is the minimum at which "every
    # count carries its reading date" is a claim with content, and the probes in
    # _undated_counts are what prove the predicate bites rather than this number.
    assert len(checked) >= 2, (
        f"only {len(checked)} count-bearing sentence(s) in the paragraph, so the "
        f"pairing has almost nothing to bite on — a guard checking nothing still "
        f"reports clean, and 2026-10-27's arch-review run proves a green suite does "
        f"not mean a correct paragraph")
    assert not unpaired, (
        f"{len(unpaired)} of {len(checked)} count-bearing sentence(s) in the "
        f"deciding-criterion paragraph carry no date ({unpaired[0][:150]!r}) — the "
        "store gains rows daily and an undated count is a claim about a moment")
    assert re.search(r"9 signature-days|nine signature-days", flat, re.I), (
        "the paragraph stopped giving the quiet its rate. Nine qualifying "
        "signature-days is the number that says a burst arrives anyway; with no "
        "arrival *rate* the 'quiet now' sentence reads like the retired 'quiet "
        "forever' one")

    # The rail must be able to see a regression, not merely report the clean case, and
    # each probe below is fed through `_undated_counts` — the node's own paragraph-wide
    # predicate — rather than a local substring check. The refused version demonstrated
    # this with `flat.replace("roughly one a week", "roughly one a week")`, a no-op the
    # assert could not tell from an edit that took, plus one splice the node's earlier
    # asserts would also have caught, which is #1713's ruled defect against #1595's rail
    # reproduced inside the node that replaced that rail.
    UNSTAMPED = "The store also holds 214 events across 12 signatures."
    # Dates present, none of them attached to the count: what the refused conjunction
    # accepted, because `not _DATED` is satisfied by a date sitting beside `first` and
    # `last` rather than beside a reading verb. Every date here is deliberately out of
    # reach of `_STAMPED`'s loose prepositions too — `on 2026-10-21` would itself be a
    # stamp-shaped pair, and a probe that accidentally reads as stamped proves nothing.
    WRONG_DATE = ("The store holds 214 events (first 2026-09-06, last 2026-10-21) "
                  "across 12 signatures.")
    STAMPED_OK = ("The store holds 214 events, counted 2026-10-30T00:28Z, across 12 "
                  "signatures.")
    for probe, why in ((UNSTAMPED, "a count carrying no date at all"),
                       (WRONG_DATE, "a count whose only dates sit beside other "
                                    "facts, never beside the count")):
        _p_checked, _p_unpaired = _undated_counts(flat + " " + probe)
        assert any(probe[:18] in s for s in _p_unpaired), (
            f"`_undated_counts` passed the probe for {why}: {probe!r} is exactly the "
            "sentence the paragraph is not allowed to grow, and accepting it is how "
            "the retired paragraph's '9 signature-days … 27 event days' survived four "
            "edits")
    assert _DATED.search(WRONG_DATE) and not _STAMPED.search(WRONG_DATE), (
        "the probe that stands for the refused conjunction is no longer shaped like "
        "it: it needs a bare ISO date the loose `_DATED` accepts and a `_STAMPED` that "
        "refuses, because `not _DATED` ANDed onto `_STAMPED` reduces to `not _DATED` "
        "(`_STAMPED` is a subset of `_DATED`) and made the strict regex decorative")
    _g_checked, _g_unpaired = _undated_counts(flat + " " + STAMPED_OK)
    assert not any(s.startswith("The store holds 214 events, counted")
                   for s in _g_unpaired), (
        "the pairing rejects a properly stamped count, which would push the fix toward "
        "refusing figures instead of refusing undated ones")


#: How the doc may say what `last_enqueue_ok` is: a stamp the pool puts on every
#: ASK. Without those words the key reads as "a pass completed", which is the
#: misreading that made #3528 call this source's diagnosis turn unbounded and #3529's
#: triage read a daily stamp as evidence work was happening.
#: The irreducible qualifier: a sentence may cite a pool watermark only if it frames
#: that key as an ASK. The first draft listed set phrases ("per pool ask", "each ask",
#: "stamped on every ask") and rejected §6's own valid wording — "says the pool is
#: still asking" — which makes the guard a trap for the doc rather than a reader of it.
#: Any phrasing that names the ask is enough; what is not enough is citing the stamp
#: and inferring from it.
_PER_ASK = re.compile(r"\bask(?:s|ed|ing)?\b", re.I)
#: `complet\w*` rather than `completes`: §6's replacement sentence disowns the retired
#: claim in the words `neither one is "a pass completed"`, and a pattern that matched
#: only the present tense would let `_affirmed_pass` off the field there — the negation
#: would then be guarding a phrase the regex never saw.
_COMPLETE_PASS = re.compile(r"a pass complet\w*|pass completes daily|"
                            r"a completed (?:pass|diagnosis)|pass (?:that )?actually "
                            r"complet", re.I)
#: Both pool keys, since the first version policed only `last_enqueue_ok` and left
#: `last_enqueue_check` — stamped on EVERY ask, clean return or not, so the more
#: misleading of the two — free to be quoted as evidence the source is keeping up.
_WM_PAIR = re.compile(r"last_enqueue_ok|last_enqueue_check")

#: Who may disown a completion phrase. §6's replacement says the opposite of the
#: retired wording in the same breath — `neither one is "a pass completed"` — so the
#: phrase alone cannot be the ban: that would refuse the paragraph that retired it.
_PASS_NEGATORS = re.compile(r"\b(?:neither|nor|not|never|nothing|no)\b", re.I)


def _affirmed_pass(sentence: str) -> bool:
    """A completed-pass conclusion the sentence does not disown.

    The ask framing alone is not enough, and the splice below proved it: the retired
    sentence it inserts contains "the pool asks hourly", so that sentence satisfies the
    ask test and still concludes `a pass completes daily`. Both halves are required —
    name the key as an ask, and draw no completed pass out of it.
    """
    m = _COMPLETE_PASS.search(sentence)
    if not m:
        return False
    return not _PASS_NEGATORS.search(sentence[max(0, m.start() - 45):m.start()])


def test_section6_says_which_watermark_is_a_sweep_and_which_is_the_pool_ask():
    """Clause 5: the schedule evidence names what each stamp records, and no citation
    of `last_enqueue_ok` is allowed to read as a completed pass.

    Three keys, two facts. `workers/sources/failure_ledger.py:260` stamps the
    source's `last_enqueued` unconditionally after `sweep_and_dispatch` returns, so it
    is the sweep's clock and says nothing about dispatch — 0 `failure_dispatches` rows
    sat against it at 2026-10-30T00:28Z. The pool's `last_enqueue_check`
    (`workers/pool.py:1636`) is stamped on every ask and `last_enqueue_ok`
    (`workers/pool.py:1647`) on a clean return only; `enqueue_if_due` returns early
    inside `min_interval_seconds` (`workers/sources/failure_ledger.py:243`) and that
    early return is clean, so the pair advances once per hourly ask whether or not the
    source swept. The retired paragraph quoted `last_enqueue_ok` as 'a pass completes
    daily', which is why #3532 had to be told the store was merely quiet: a stamp that
    moves hourly is not a daily pass, and a stamp that moves daily is not a dispatch.
    """
    sec6 = _section(SEC6)
    assert sec6 and len(sec6) > 400, (
        "no §6 body found, so the watermark reads below are the scan dying rather than "
        "the claim dying")
    flat6 = " ".join(sec6.split())
    sentences = _BENCH_SENTENCE.split(flat6)

    def _sents(chunk: str) -> list[str]:
        return _BENCH_SENTENCE.split(" ".join(chunk.split()))

    # Unconditional, and over BOTH pool keys. The first version read
    # `assert _PER_ASK.search(s) or _COMPLETE_PASS.search(s) is None`, whose second arm
    # is true for most citations, so the loop fired only on a sentence already claiming
    # a completed pass. `The stamp `last_enqueue_ok` read 2026-11-01T04:59Z, so the
    # source is keeping up.` walked through it — and that sentence is the misreading
    # clause 5 names: `last_enqueue_check` is stamped on every ask, clean return or not
    # (`workers/pool.py:1636`), so neither key says anything about this source unless
    # the sentence carrying it says so.
    pair_says = [s for s in sentences if _WM_PAIR.search(s)]
    assert pair_says, (
        "§6 no longer cites either pool watermark, so the keys that move once per pool "
        "ask have gone undocumented and the next reader reaches for them again")
    for s in pair_says:
        assert _PER_ASK.search(s) and not _affirmed_pass(s), (
            f"{_WM_PAIR.search(s).group(0)!r} is cited as {s[:170]!r} without naming "
            "the ask, or while drawing a completed pass out of the stamp — the reading "
            "that made a key moving hourly look like a diagnosis turn running daily")
    PROBE = ("The stamp `last_enqueue_ok` read 2026-11-01T04:59Z, so the source is "
             "keeping up.")
    assert _WM_PAIR.search(PROBE) and not _PER_ASK.search(PROBE), (
        "the probe that must be red no longer has the shape of the misreading: a pool "
        "key cited with no per-ask qualifier is exactly what the loop above let through")
    QUALIFIED = ("`last_enqueue_ok`, stamped once per pool ask, read "
                 "2026-11-01T04:59Z, so the pool is asking.")
    assert _PER_ASK.search(QUALIFIED) and not _affirmed_pass(QUALIFIED), (
        "the pair of tests rejects wording §6 itself uses, so the loop above is a trap "
        "for the doc rather than a guard for its reader")
    NEGATED = ('The pool asks hourly and `last_enqueue_check` is stamped on every ask; '
               'neither one is "a pass completed".')
    assert _COMPLETE_PASS.search(NEGATED) and not _affirmed_pass(NEGATED), (
        "`_affirmed_pass` reads a disowned completion phrase as a claim, which would "
        "refuse the paragraph that retired the wording")

    enq_says = [s for s in sentences if "last_enqueued" in s]
    assert enq_says, (
        "§6 stopped naming `last_enqueued` as the sweep's stamp, which is the only key "
        "that answers whether this source swept")
    assert any("sweep" in s for s in enq_says), (
        "the `last_enqueued` sentences never call it a sweep stamp, so a reader has no "
        "way to tell it from the pool's ask pair")

    # The retired sentence, spliced back into the section: the reading this clause
    # exists to forbid, run through the same predicate rather than a local echo. The
    # #1713 ruling applies here — a splice no live reader would ever meet, or one a
    # no-op replace pretends to make, proves nothing.
    START = "and its stamps are not the same fact:"
    END = "whether it dispatched."
    i, j = flat6.find(START), flat6.find(END)
    assert i != -1 and j != -1 and j > i, (
        f"the live schedule sentence this node splices into has moved (found START="
        f"{i != -1}, END={j != -1}), so the assert below would be vacuous rather than "
        "red")
    spliced = (flat6[:i]
               + "the pool asks hourly (`last_enqueue_check` 2026-10-28T04:12Z), a "
                 "pass completes daily (`last_enqueue_ok` 2026-10-27T04:59Z), and "
                 "`failure_findings` is still empty. "
               + flat6[j + len(END):])
    assert spliced != flat6, (
        "the splice did not take, so its assert below is vacuous rather than red")
    assert any("last_enqueue_ok" in s
               and (_affirmed_pass(s) or not _PER_ASK.search(s))
               for s in _sents(spliced)), (
        "the retired 'a pass completes daily (`last_enqueue_ok` …)' wording sailed "
        "through the check above, so the check is not the thing keeping it out")

    # The code seams, read rather than cited from memory: the doc's line numbers are
    # claims about the tree, and a doc-claims file that only greps the doc grades
    # prose against prose.
    pool = (ROOT / "workers" / "pool.py").read_text(encoding="utf-8").splitlines()
    assert "last_enqueue_check" in pool[1635], (
        f"workers/pool.py:1636 no longer stamps the pool's attempt key ({pool[1635]!r}) "
        "while §6 cites that line for it")
    assert "OK_WM_KEY" in pool[1646], (
        f"workers/pool.py:1647 no longer stamps the clean-return key ({pool[1646]!r})")
    src = (ROOT / "workers" / "sources" / "failure_ledger.py").read_text(
        encoding="utf-8").splitlines()
    assert "WATERMARK_LAST_ENQUEUED" in src[259], (
        f"workers/sources/failure_ledger.py:260 is no longer the unconditional sweep "
        f"stamp ({src[259]!r}) — the bullet says it fires whether or not a dispatch "
        "happened, and if that moved the paragraph is describing a different program")
    assert "WATERMARK_LAST_ENQUEUED" in src[242], (
        f"workers/sources/failure_ledger.py:243 no longer reads the watermark for the "
        f"interval gate ({src[242]!r}), which is the early return that makes the "
        "pool's clean-return stamp advance without a sweep")


def test_section6_says_the_growth_detector_is_armed_not_unreachable():
    """Clauses 1, 2 and 4: §6 says the detector is armed, and cannot be walked back
    into saying it cannot reach the store.

    The retired paragraph is the interesting kind of wrong — every number in it was
    accurate (0 runs, 0 queue, `failure_findings` empty) and the inference drawn from
    them was not. `git log -S"no pass can review it now" -- architecture/workers-jobs.md`
    returns one sha, HEAD's own 08bb1eee, authored by the arch-review job: it re-derived
    'no arrival condition' from a store that had simply been quiet, and a later triage
    filed that as a high incident (#3529, 2026-10-28). Since #3281 the silent-baseline arm
    is in main (`d4eddcf5`, settled 2026-10-25T03:04:41Z, `app/failure_ledger.py:576`,
    `GROWTH_MIN_COUNT = 3` at `:482`) and a replay over a COPY of the live store returns
    a finding for day='2026-10-21' — the burst §6 said nothing could reach.

    So the guard is not 'the paragraph reads nicely'. A splice test welds the retired
    sentence back into the block's head and requires the absence check to survive it,
    and the forward claims are re-graded on the spliced text: the finding is that both
    still PASS, so the phrase check is the only thing standing between this paragraph
    and #3529-mark-2. That is worth recording in the failure message, because it says
    the next editor must keep the three phrases absent, not merely re-word around them.
    """
    para = _deciding_paragraph()
    flat = " ".join(para.split())
    for phrase in RETIRED_REACHABILITY:
        assert phrase not in para, (
            "§6's deciding-criterion paragraph carries the retired claim "
            f"{phrase!r} again. Since #3281 (2026-10-25) the silent-baseline arm is in "
            "main and reaches the store, and the pass on 2026-10-22 really did review "
            "the 2026-10-21 n=8 burst. It is what generated #3529 (high, 2026-10-28) "
            "out of a store that had merely been quiet — the arch-review job writing "
            "'no arrival condition' and a later triage reading it as an incident is "
            "the self-refiling loop #3532 was split out of #3529 to close")

    # clauses 2 and 4, graded by the same function the splice guard runs below.
    bad = _armed_claims(flat)
    assert not bad, "§6's deciding paragraph lost a forward-looking claim: " + "; ".join(bad)

    # clause 4's receipts: the corrected paragraph restates the retired one's three
    # bursts rather than silently dropping two of them — `04f90c87b03a` 09-25 n=23 and
    # `a1afbe2e3b04` 09-06 n=6 are named inside the D-1 sentence, and `detect_growth`
    # is named as what fires today. So the surviving figures are 0 runs / 0 queue / 9
    # signature-days / 27 event days / last-qualifying 2026-10-21 / n=8.
    assert "04f90c87b03a" in para and "a1afbe2e3b04" in para, (
        "the two bursts the retired sentence enumerated have left the paragraph. The "
        "D-1 statement still refers to 'a burst that closed before 2026-10-25', and "
        "with nothing named a later editor cannot tell those days are bounded from "
        "the detector being broken — which is how the retired claim got written")
    assert "detect_growth" in para, (
        "§6 stopped naming the function that fires, so 'the detector is armed' is a "
        "claim about a paragraph rather than about code a reader can go and read")

    # clause 1's guard: the retired sentence, spliced into the block's head, must be
    # caught by the check above — and must NOT break anything else.
    retired = ("**Measured 2026-10-28 that first real run has still not happened: 0 "
               "`runs` rows for `source='failure-ledger'` in `workers.db` all-time, and "
               "0 queue rows.** `failure_findings` is still empty because neither "
               "detector can reach this store: while growth counts only the day before "
               "the pass, so every burst the store has ever produced closed before "
               "#3281's burst-from-silence arm landed on 2026-10-25 and no pass can "
               "review it now. The seven-day criterion and #543's clock are therefore "
               "not early: they wait on a burst with no arrival condition.")
    spliced = retired + " " + para
    assert any(p in spliced for p in RETIRED_REACHABILITY), (
        "the splice fixture no longer contains the retired claim, so the check above "
        "has to be re-pointed at the current wording")
    hit = [p for p in RETIRED_REACHABILITY
           if any(p in s for s in _BENCH_SENTENCE.split(spliced))]
    assert len(hit) == len(RETIRED_REACHABILITY), (
        f"the per-sentence absence scan sees {len(hit)} of "
        f"{len(RETIRED_REACHABILITY)} retired phrases in the spliced block, so a "
        f"sentence-level scan would let the claim back in through the paragraph head")
    spliced_bad = _armed_claims(" ".join(spliced.split()))
    assert not spliced_bad, (
        "the spliced text also lost a forward-looking claim ("
        + "; ".join(spliced_bad) + "), so this fixture is not a clean regression: the "
        "finding below is that only the phrase check separates the two wordings, and "
        "it is only that finding if everything else survives the splice")


def test_workers_md_does_not_restate_the_retired_unscheduled_claim():
    """The second carrier of the same retired claim, in the second workers doc.

    `architecture/workers.md`'s source table explained `failure-ledger` with "and
    unscheduled until a person adds its config block" — the same claim §6 carried,
    falsified by the same block, and the only node that reads that file
    (`test_workers_md_source_table_is_the_registry`) checks which names are in it, not
    what the row says about them. Pinned here rather than in a new file because the
    fact under test is one claim with two homes, and the half that rots is the half
    nobody reads.
    """
    text = (ARCH / "workers.md").read_text(encoding="utf-8")
    row = next((ln for ln in text.splitlines()
                if ln.startswith("| `failure-ledger`")), None)
    assert row is not None, "architecture/workers.md lost its failure-ledger row"
    for gone in ("unscheduled", "until a person", "no config block"):
        assert gone not in row, f"the row still says {gone!r}: {row[:120]}"
    assert "#2309" in row, (
        "the row no longer says what put the source on a timer, so a reader has no "
        "record of when it started")


# --------------------------------------------------------------------------- #
# The queue discriminator: which arm reaches `mark_failed` (item #3087)
#
# §5's in-band-failure bullet and
# `architecture/research-pipeline.md` §3 state the same rule from two sides, and
# until now nothing read `workers/pool.py` from either file to check it —
# `git grep -nw -e mark_completed -e mark_failed` over this file and
# `tests/test_research_doc_claims.py` was empty when the item was filed, and every
# node above that cites `workers/pool.py` does so in prose without opening it. The
# BEHAVIOUR was never
# unguarded: `test_an_in_band_failure_does_not_send_the_item_back_for_a_retry` in
# `tests/test_workers_pool.py` drives a real pool over a source that returns
# `{"status": "failed"}` and asserts the queue row comes back `completed`. What
# nothing held was this page's prose against that code.

#: The pool's source, read as text. The claim is about which queue call sits on
#: which arm of one `try`, which an import cannot answer.
POOL_SRC = ROOT / "workers" / "pool.py"

#: Slice anchors. Symbols, never line numbers. The sentence `25364e54` deleted from
#: `architecture/research-pipeline.md` §3 anchored this rule to three of them — `:717`,
#: `:744`, `:773` — and on that commit's `workers/pool.py` not one of the three named a
#: queue call: `awk 'NR==717||NR==744||NR==773' <(git show 25364e54^:workers/pool.py)`
#: gives `poll_idle_seconds: float = 2.0,`, a comment about boot reconciliation, and
#: `"engaged_since": None,`. A number is not a citation. An anchor that stops being
#: unique fails the node rather than silently reading a different slice, which is the
#: same failure in the other direction.
#: One anchor opens the region, and the arms inside it are found STRUCTURALLY rather
#: than by a second and third needle — because neither could be anchored to occur once:
#: `grep -c 'finally:' workers/pool.py` is 2 and `grep -c 'except Exception as e:'` is
#: 12, so the previous commit had to search for them forward from a fixed point, which
#: is exactly what made its ordering assertion true by construction. Indentation is the
#: third anchor instead, and it also keeps the returned arm's inner `except Exception:`
#: (the `cost_outlier.arm_after_run` wrapper, nested one level deeper) out of the list.
_POOL_VERDICT = 'run_status = norm["status"]'        # opens the try whose arms follow

#: The handler expressions this guard expects to be exactly the raised arms of that
#: `try`, in the order the source states them. Compared against what `_pool_raised_arms`
#: parses out of `workers/pool.py`, and NOT against literals the helper itself inserted:
#: the previous commit's `assert set(arms) == {"asyncio.TimeoutError", "Exception"}`
#: checked the dict's keys against the two strings that helper had just built it from,
#: so no source could have made it fail. A third handler, a renamed one, a dropped one
#: or a reordered pair all reach this line.
POOL_RAISED_ARMS = ("asyncio.TimeoutError", "Exception")


def _pool_lines() -> list[str]:
    return POOL_SRC.read_text(encoding="utf-8").splitlines()


def _pool_line_once(lines: list[str], needle: str) -> int:
    hits = [i for i, ln in enumerate(lines) if needle in ln]
    assert len(hits) == 1, (
        f"`workers/pool.py` now has {len(hits)} lines containing {needle!r}, so this "
        "guard's slice boundary is ambiguous. Re-point it at something the file says "
        f"once. The alternative — indexing by line number — is what the sentence "
        "`25364e54` deleted got wrong, and the claim at stake here is "
        f"{DOC_RAISED_ARM!r}.")
    return hits[0]


def _pool_indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _pool_enclosing_try(lines: list[str], inside: int) -> tuple[int, int]:
    """The nearest `try:` above `inside` that is INDENTED LESS than it, with its
    indentation. Shallower and nearest: the innermost enclosing `try` is the one whose
    handlers are this job's raised arms, and a `try` nested inside the body has handlers
    of its own that are not ours to read."""
    ind = _pool_indent(lines[inside])
    for i in range(inside - 1, -1, -1):
        if lines[i].strip() == "try:" and _pool_indent(lines[i]) < ind:
            return i, _pool_indent(lines[i])
    raise AssertionError(
        f"`workers/pool.py` has no `try:` above {lines[inside].strip()!r}, so the arms "
        f"this guard reads as 'raised' are no longer an exception's doing — and the §5 "
        f"clause they are pinned against is {DOC_RAISED_ARM!r}.")


def _pool_try_clauses(lines: list[str], try_at: int,
                      ind: int) -> tuple[list[tuple[str, int]], int]:
    """Every clause of the `try` at `try_at` in source order as (clause text, line), and
    the line where the statement ends.

    A clause is a line at the `try`'s OWN indentation beginning `except`, `else:` or
    `finally:`; anything deeper is a body, and the first same-or-shallower line that is
    not one of those closes the statement. Indentation, not a needle, is what keeps the
    returned arm's nested `except Exception:` out of this list — and if the pool ever
    grows a third handler it shows up here rather than being skipped past.
    """
    clauses: list[tuple[str, int]] = []
    end = len(lines)
    for i in range(try_at + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            continue
        if _pool_indent(line) > ind:
            continue
        stripped = line.strip()
        if stripped.startswith(("except", "else:", "finally:")):
            clauses.append((stripped, i))
            continue
        end = i
        break
    return clauses, end


def _pool_clause_kind(clause: str) -> str:
    """`except asyncio.TimeoutError as e:` → `asyncio.TimeoutError`: the handler named
    as the source writes it, with no binding and no colon, because that is the form
    `POOL_RAISED_ARMS` states and the form a reader searches for."""
    return clause[len("except "):].split(" as ")[0].rstrip(":").strip()


def _pool_raised_arms(lines: list[str]) -> dict[str, str]:
    """The arms that run when the job RAISES, keyed by the handler expression the source
    actually states and in the order it states them, each slice running to the clause
    that follows it — the `finally:` releasing the semaphore on this `try`, or the end of
    the statement if it ever grows no `finally`.

    Keys and boundaries are both read out of the file. Nothing here inserts a key and
    then checks the result for that key: the two assertions that survived from the
    previous commit were exactly that, and the review rung named both. The handler NAMES
    need no uniqueness assertion either — CPython rejects a duplicate handler for the
    same type on one `try` ("duplicate handler for type"), so the dict below cannot lose
    an entry to a collision, and an assertion that the lengths match would be the third
    guard that can never fire. A file that somehow did state one is a file that does not
    import, and `workers/pool.py` failing to import fails the suite by itself.
    """
    verdict = _pool_line_once(lines, _POOL_VERDICT)
    try_at, ind = _pool_enclosing_try(lines, verdict)
    clauses, end = _pool_try_clauses(lines, try_at, ind)
    handlers = [(kind, at) for kind, at in clauses if kind.startswith("except")]
    assert handlers, (
        "the `try` that reads the returned verdict in `workers/pool.py` has no handler "
        "at all, so there is no raised arm left to tell apart from a returned one, and "
        f"§5's rule is {DOC_RAISED_ARM!r}.")
    assert verdict < handlers[0][1], (
        "the returned-verdict line is no longer in that `try`'s body — it sits inside "
        "its first handler — so the slice this file's doc describes as a RAISED "
        f"exception is the arm that handles a RETURNED verdict: {DOC_RAISED_ARM!r}.")
    return {_pool_clause_kind(kind):
            "\n".join(lines[at:next((a for _, a in clauses if a > at), end)])
            for kind, at in handlers}


#: The clause, verbatim from §5's in-band-failure bullet, asserted to still be that
#: page's wording by the node below and quoted in the messages of the node above it.
DOC_RAISED_ARM = "only a *raised* exception reaches `mark_failed`"


def test_both_raised_arms_release_the_queue_with_mark_failed():
    """Clause 2: the raised arms are the only thing that fails a queue row.

    Every handler the pool has on that `try` — asserted below to be exactly the two
    `POOL_RAISED_ARMS` names, in the order the source states them, read out of the file —
    calls `mark_failed` and never `mark_completed`, and each still writes the run row
    `status="failed"` — which is the half that makes this a discriminator rather than a
    call count: on this page's own evidence the run row reads `failed` on both arms, so
    the queue call is the only thing separating an in-band failure (schedule intact) from
    a raised one (backoff, attempts).
    """
    arms = _pool_raised_arms(_pool_lines())
    # Tuple, not set: the ORDER is a claim about the source too, and `_pool_raised_arms`
    # returns the handlers in the order the file states them, so a swapped pair lands
    # here instead of passing. Both sides of this comparison come from different places —
    # the left is parsed from `workers/pool.py`, the right is this file's expectation —
    # which is the property the previous commit's version lacked.
    assert tuple(arms) == POOL_RAISED_ARMS, (
        f"`workers/pool.py`'s raised arms are {tuple(arms)}, not {POOL_RAISED_ARMS}, so "
        "the pair this guard pins is not the pair the code has: a third handler would "
        "need the same `mark_failed` before §5's rule still read as the whole rule, and a "
        f"renamed or dropped one means this file is quoting {DOC_RAISED_ARM!r} against "
        "arms that no longer exist.")
    for name, arm in arms.items():
        assert "mark_failed" in arm, (
            f"the {name} arm no longer calls `mark_failed`, so a job that raises stops "
            f"being retried by the queue. The page this file guards says the opposite: "
            f"{DOC_RAISED_ARM!r}.")
        assert "mark_completed" not in arm, (
            f"`mark_completed` reached the {name} arm, so the rule is no longer which "
            "arm a failure took but only whether it took one — and the doc's "
            f"contrast collapses: {DOC_RAISED_ARM!r}.")
        assert 'status="failed"' in arm, (
            f"the {name} arm no longer writes the run row `failed` inside the handler, "
            "so `runs.status` stopped being the thing both arms share. This guard's "
            "point is that the QUEUE call is the discriminator precisely because "
            "`runs.status` says the same word on every arm; if that stops being true, "
            "say so in §5 as well as here.")


def test_workers_jobs_md_names_the_queue_call_each_arm_takes():
    """Clause 4: §5's in-band bullet has to keep naming the calls, in the right arms.

    Read as one `## ` section, not paragraph by paragraph: this doc states the rule as a
    bullet inside a bullet, so it spans several blank-line blocks and reading one of them
    would see half a sentence. What has to be unique is the SECTION — §5, Intake, is the
    queue's surface and the only place the two calls belong together.
    """
    text = (ARCH / "workers-jobs.md").read_text(encoding="utf-8")
    # One `## ` section may state the rule. Not one paragraph: this doc writes its
    # claims as bullets inside a bullet, so the rule is spread over several `\n\n`
    # blocks and splitting there would read half a bullet as a contradiction. A second
    # SECTION naming both calls is a real risk — two places, two rules — so that is
    # what has to be unique.
    sections = [s for s in re.split(r"(?m)^## .*$", text)
                if "mark_completed" in s and "mark_failed" in s]
    assert len(sections) == 1, (
        f"{len(sections)} sections of `architecture/workers-jobs.md` name both queue "
        "calls, so the discriminator now has two homes in one page and this node reads "
        "whichever comes first. Keep the rule where §5 has it.")
    flat = " ".join(sections[0].split())
    assert DOC_RAISED_ARM in flat, (
        f"§5 lost {DOC_RAISED_ARM!r}, the clause the code-side assertions above quote. "
        "The behaviour test whose failure message this passage explains is "
        "tests/test_workers_pool.py::test_an_in_band_failure_does_not_send_the_item_"
        "back_for_a_retry, and it asserts the queue row comes back `completed`.")
    assert ("calls `mark_completed` **regardless** — " + DOC_RAISED_ARM) in flat, (
        "§5 no longer states the two arms in one breath: the returned verdict settled "
        "with `mark_completed` regardless, and only a raised exception reaching "
        "`mark_failed`. Split across sentences it stops being a discriminator — which is "
        "the shape `architecture/research-pipeline.md`'s deleted §3 sentence had, where a "
        "`mark_completed` mention sat "
        "beside a `failed` branch and read as if the failed arm were settled differently.")
    assert ("So a source that returns `failed` and expects a retry does not get one "
            "from the queue.") in flat, (
        "§5 stopped naming the consequence for the returned arm, so the asymmetry is "
        "left as two call names instead of what each arm costs the schedule.")
    assert "and the queue's backoff" in flat, (
        "§5 no longer says what a raised exception reaches besides the call — the "
        "backoff is the reason the split matters to a reader who runs a job on a timer.")


# --------------------------------------------------------------------------- #
# §1's scheduler-seat list and §2's shared-table count: the two enumerations
# #3286 found hand-maintained and unpinned.
#
# Both had rotted by 2026-10-25, and the pass that fixed the PROSE left nothing
# to keep it fixed. §1 stated the 60 s tick "carries two deterministic chores"
# while `WorkerPool._scheduler_loop` awaited five seats ahead of
# `_scheduler_pass` — among them the dispatch watch (#1681) and the fleet
# watchdog (#1682), which exist precisely because their alarms had been disarmed
# from inside a source's own `enqueue_if_due`, so a page naming two of the five
# under-reports the thing watching the pool. §2 stated `.tables` shows seven
# while `pause_journal` (#2615) had already made it eight. A corrected paragraph
# with no pin is a paragraph that rots again, so each enumeration is now held to
# the tree it describes, in the shape of `tests/test_kv_gate_docs.py`: read the
# live thing, parse the prose with a BOUNDED capture, compare in BOTH directions.
#
# Two hazards decided the shape.
#   * `workers/queue.py` carries prose: `apply_grant_ddl`'s docstring quotes the
#     literal `CREATE TABLE IF NOT EXISTS` while explaining why that statement is
#     not a migration, so a pattern stopping at the identifier reads `IF` as a
#     table name and the count comes out one too high out of a comment.
#     `_ddl_tables` therefore requires the name to be followed by `(` — a
#     statement, not a sentence — and
#     `test_a_prose_create_table_is_not_counted_as_a_table` feeds it that real
#     sentence instead of asserting the guard works. The sentence is located
#     from its own text and the symbols around it, never from a line number: a
#     numeric citation to that file is banned corpus-wide (#2971, widened #3233)
#     precisely because the numbers rot, and this file is inside the ban.
#   * The page's count words are not evidence of anything. §1 still carries the
#     retired "two" inside the sentence explaining the fix, §7 and §8 hold
#     unrelated counts ("seven task ids", "all nine"), and §8's dated review rows
#     record "the seventh shared table" and "now eight" AS HISTORY. A node that
#     grepped the file for a number would be vacuous or permanently red, so §1
#     and §2 are compared as LISTS, each sliced inside its own `## ` heading, and
#     §8 is never opened.


#: The module attributes that own the shared database's DDL. Each is read as the
#: live module sees it rather than re-typed here: a fourth `_SCHEMA` appearing in
#: a new module is the event this pin is for, and a hand-maintained list of
#: modules in the test would be a third copy of the very enumeration #3286 is
#: about. `workers.queue._SCHEMA` already concatenates `GRANT_DDL`, which is why
#: `authority_grants` and `grant_dispatch` come out of that one attribute.
DDL_SCHEMAS = ("workers.queue", "agent_mcp._tool_effects", "agent_mcp.egress")

#: A `CREATE TABLE` statement. The trailing `(` is the whole guard: it is what
#: separates a definition from `apply_grant_ddl`'s docstring, whose sentence
#: "`CREATE TABLE IF NOT EXISTS` alone is not a migration" would otherwise
#: contribute the table `IF`.
# The name class is case-wide on BOTH sides — `_TABLE_DDL` here and `_DOC_TABLE`
# below — because a class restricted to lowercase hides a table declared as
# `CREATE TABLE Audit_Log(...)` from the derived set AND from the page's parsed
# list, so count and list would still agree with each other while both were
# wrong. `_shared_tables` then refuses an uppercase name outright rather than
# folding it in: SQLite resolves table names case-insensitively, the eight
# declared today are all lowercase, and a mixed-case one is a naming decision a
# human has to make before the page can be written to match.
_TABLE_DDL = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(")

#: A table name inside §2's parenthesised enumeration.
_DOC_TABLE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")

#: §2's count claim and the list it counts, as ONE bounded capture. Bounded at
#: the closing paren because §2 backticks plenty of things that are not tables —
#: `GRANT_DDL`, `_SCHEMA`, `expires_at`, `holder_kind`, `actor`, `hold_id`,
#: `.tables` — and any of them would otherwise satisfy or trip a check written
#: over the section's whole text. Anchored on the `.tables` sentence, the only
#: place in the file where a count of the shared tables is a PRESENT-TENSE claim.
_TABLE_SITE = re.compile(
    r"So `\.tables`\s+shows\s+(?P<count>[a-z]+)\s+\((?P<tables>[^)]*)\)")

_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
                "twelve": 12}


def _ddl_tables(sql: str) -> set[str]:
    """Every table a SQL string actually creates, statements only."""
    return set(_TABLE_DDL.findall(sql))


def _shared_tables() -> set[str]:
    """The union of the three modules' own `_SCHEMA` strings."""
    out: set[str] = set()
    for mod_name in DDL_SCHEMAS:
        mod = importlib.import_module(mod_name)
        sql = getattr(mod, "_SCHEMA", None)
        assert sql, (f"{mod_name} no longer declares a module-level `_SCHEMA`, so "
                     "this pin's source of the shared tables has moved — find it "
                     "and name it in DDL_SCHEMAS rather than dropping this entry")
        found = _ddl_tables(sql)
        assert found, f"{mod_name}._SCHEMA creates no table: the extractor read nothing"
        mixed = sorted(n for n in found if n != n.lower())
        assert not mixed, (
            f"{mod_name}._SCHEMA declares {mixed} with a mixed-case name. Every "
            "table named in `architecture/workers.md` §2 is lowercase, SQLite "
            "resolves these names case-insensitively, and the page's enumeration is "
            "what a human reads off — settle the spelling in the schema before "
            "widening this assertion rather than letting the count silently miss a "
            "table")
        out |= found
    return out


def _workers_md() -> str:
    return (ARCH / "workers.md").read_text(encoding="utf-8")


def _md_section(heading: str, doc: str) -> str:
    """One `## ` section of `workers.md`, heading line excluded, out of `doc`.

    Takes the text rather than reading the file so a mutated page can be handed
    in: an enumeration pin that can only be exercised on the live tree has never
    been shown to be able to fail. Same reason `_table_after` and `_section`
    slice rather than grep.
    """
    assert heading in doc, f"{heading} is gone from `architecture/workers.md`"
    start = doc.index(heading) + len(heading)
    rest = doc[start:]
    end = rest.find("\n## ")
    return rest if end < 0 else rest[:end]


#: `_scheduler_loop`'s seat list, read out of the source instead of re-listed
#: here. A `SEATS = {...}` literal in this file would be a second hand-maintained
#: copy of the list, which is the rot #3286 exists to stop: as written, a seat
#: added to the pool has to be named on the page and nowhere else.
_LOOP_OPEN = "async def _scheduler_loop"
_PASS_CALL = "await self._scheduler_pass()"
_AWAIT_SEAT = re.compile(r"await self\.(_[a-z][a-z0-9_]*)\(\)")


def _seats_awaited(src: str) -> list[str]:
    """Every `await self._<seat>()` the scheduler loop runs ahead of its pass."""
    assert src.count(_LOOP_OPEN) == 1, (
        f"`{_LOOP_OPEN!r}` occurs {src.count(_LOOP_OPEN)} times in the source handed "
        "to the extractor, so the seat region has no single opening. Re-point it at "
        "something the file says once — a line number would be worse.")
    body = src[src.index(_LOOP_OPEN):]
    assert body.count(_PASS_CALL) == 1, (
        f"`{_PASS_CALL!r}` occurs {body.count(_PASS_CALL)} times after the loop opens, "
        "so the extractor cannot tell where 'ahead of the pass' ends.")
    seats = _AWAIT_SEAT.findall(body[:body.index(_PASS_CALL)])
    assert seats, (
        "no `await self._…()` call ahead of `_scheduler_pass`: the extractor read an "
        "empty region, which would make every comparison below pass by comparing "
        "nothing.")
    assert len(seats) == len(set(seats)), f"a seat is awaited twice: {seats}"
    return seats


#: §1's seat enumeration, bounded between the colon that opens it and the period
#: that closes it. `_scheduler_pass` is named in the sentence BEFORE that colon,
#: so the one seat-shaped identifier on the page that is not a seat sits outside
#: the capture and can neither satisfy it nor trip it.
_SEAT_SITE = re.compile(
    r"cannot silence them:\s*(?P<seats>.*?)an alarm that its own silencing "
    r"cannot report\.", re.S)
_SEAT_NAME = re.compile(r"`(_[a-z][a-z0-9_]*)`")


def _seats_named(sec1: str) -> set[str]:
    match = _SEAT_SITE.search(sec1)
    assert match, (
        "§1 of `architecture/workers.md` no longer states the seat enumeration "
        "between 'cannot silence them:' and 'an alarm that its own silencing cannot "
        "report.' Re-point `_SEAT_SITE` at the new sentence: a capture that reads "
        "nothing would let the seat pin pass on an empty set.")
    named = set(_SEAT_NAME.findall(match.group("seats")))
    assert named, (
        "§1's seat enumeration names no backticked `_name`, so the pin has nothing "
        "to compare — the page describes the seats in prose the pool cannot be "
        "compared to. Name each seat by the method `_scheduler_loop` awaits.")
    return named


def test_the_scheduler_runs_more_than_one_seat_before_its_pass():
    """Non-vacuity for the seat pin, the way #1341 held its own first check.

    Two is the floor, not the count: five seats run today and the nodes below
    compare LISTS rather than that number, because a sixth seat with the page
    updated is a passing change and a pinned five would make the pin itself the
    thing that has to be edited whenever the pool grows — which is how the list
    on the page got left behind in the first place.
    """
    seats = _seats_awaited(POOL_SRC.read_text(encoding="utf-8"))
    assert len(seats) >= 2, (
        "the pool runs one seat, so 'every seat is named on the page' is one name "
        "and the enumeration checks below are meaningless")


def test_workers_md_sec1_names_every_awaited_seat_and_nothing_else():
    """Clause 1: the page's seat list IS the pool's, in both directions."""
    seats = _seats_awaited(POOL_SRC.read_text(encoding="utf-8"))
    named = _seats_named(_md_section("## 1. Shape", _workers_md()))
    missing = sorted(set(seats) - named)
    assert not missing, (
        f"`_scheduler_loop` awaits {missing} before `_scheduler_pass` and §1 does "
        "not name them. These seats are the readers kept outside the pass's `try` "
        "because a source cannot be trusted to report on itself — the page that "
        "under-reports them under-reports what is watching the pool.")
    extra = sorted(named - set(seats))
    assert not extra, (
        f"§1 names {extra} as a seat the 60 s tick carries, and `_scheduler_loop` "
        "awaits none of them before `_scheduler_pass`. The page promises a "
        "deterministic reader that the loop no longer runs.")


def test_the_seat_pin_bites_in_both_directions():
    """Clause 2: the seat node can fail, and fails on either side moving alone.

    Three mutations, each one edit to one side, because the live tree is a single
    observation: an unmutated green is also what a node that cannot fail looks
    like. Every case is the comparison the live node above runs, on the same
    helpers, so nothing here is a second and weaker copy of that check.
    """
    src = POOL_SRC.read_text(encoding="utf-8")
    doc = _workers_md()
    seats, named = _seats_awaited(src), _seats_named(_md_section("## 1. Shape", doc))
    assert set(seats) == named, "the live tree is inconsistent; fix that first"

    # (a) a seat added to the pool and not to the page: clause 1's failure.
    added_src = src.replace(
        "            await self._watch_fleet()",
        "            await self._watch_fleet()\n"
        "            await self._watch_brand_new_thing()", 1)
    assert added_src != src, "the source mutation found no seat to follow"
    after = set(_seats_awaited(added_src))
    assert "_watch_brand_new_thing" in after, "the extractor missed the new seat"
    assert after - named == {"_watch_brand_new_thing"}, (
        "a seat added to the pool is not reported as missing from §1, so the "
        "forward direction of the pin is not a check")

    # (b) a seat dropped from the page and left running: clause 2's failure, the
    # same node's other assert firing. `re.sub` rather than a literal replace
    # because the enumeration is hard-wrapped, so a name and the backticked
    # module after it do not necessarily sit on one line.
    dropped_doc = re.sub(r"`_watch_fleet`,\s*", "", doc, count=1)
    assert dropped_doc != doc, "the page mutation found no seat name to drop"
    dropped = _seats_named(_md_section("## 1. Shape", dropped_doc))
    assert dropped == named - {"_watch_fleet"}, (
        "dropping `_watch_fleet` from §1 left the parsed list alone, so the capture "
        "is reading somewhere other than the enumeration")
    assert set(seats) - dropped == {"_watch_fleet"}, (
        "a seat the pool still runs can vanish from the page without the pin "
        "reporting it")

    # (c) a seat named on the page that the pool never awaited: the reverse
    # direction, which no amount of consistency in the source can produce.
    phantom_doc = re.sub(r"(the poison sweep \()",
                         r"\1`_watch_phantom_seat`, ", doc, count=1)
    assert phantom_doc != doc, "the page mutation found no enumeration to widen"
    phantom = _seats_named(_md_section("## 1. Shape", phantom_doc))
    assert "_watch_phantom_seat" in phantom and phantom - set(seats) == {
        "_watch_phantom_seat"}, (
        "a name on the page that no source awaits is not reported, so the reverse "
        "direction of the pin is not a check")


def test_the_shared_table_set_reaches_every_declaring_schema():
    """Non-vacuity for §2's pin: the union is the union, not the first file.

    §2's own account splits the eight across three modules, and the drift that
    made the count wrong was a LATER addition (`pause_journal`, #2615) in the
    same `_SCHEMA` as the queue's three tables. So the set is checked to contain
    the grant pair — proof `GRANT_DDL` was reached through the concatenation
    rather than read past — and the journal, which is the table the count missed.
    """
    tables = _shared_tables()
    for must_have in ("queue", "runs", "watermarks", "authority_grants",
                      "grant_dispatch", "pause_journal", "tool_effects",
                      "egress_events"):
        assert must_have in tables, (
            f"`{must_have}` is not in the set built from {DDL_SCHEMAS}, so §2's "
            "comparison below is being run against a partial read of the DDL")


def test_workers_md_sec2_counts_and_names_exactly_the_declared_tables():
    """Clause 3: §2's stated count and its enumerated list are the declared set.

    Count and list are both graded, and the list in both directions: a ninth
    table in `_SCHEMA` has to fail this, and so has a name on the page that no
    source creates. The count is read as a word because that is how the sentence
    states it, and compared to the length of the derived set rather than to a
    literal this file carries.
    """
    tables = _shared_tables()
    match = _TABLE_SITE.search(_md_section("## 2. The queue", _workers_md()))
    assert match, (
        "§2 of `architecture/workers.md` no longer states the shared-table count "
        "as `So `.tables` shows <word> (<list>)`. Re-point `_TABLE_SITE`: the "
        "alternative is a pin that silently compares nothing.")
    word = match.group("count")
    assert word in _COUNT_WORDS, f"§2's count {word!r} is not a word this node reads"
    assert _COUNT_WORDS[word] == len(tables), (
        f"§2 says `.tables` shows {word} ({_COUNT_WORDS[word]}) but the schemas "
        f"declare {len(tables)}: {sorted(tables)}. §2's 2026-09-25 review row says "
        "the count missed a table once already, which is the drift this node exists "
        "for.")
    listed = set(_DOC_TABLE.findall(match.group("tables")))
    assert len(listed) == len(match.group("tables").split(",")), (
        "§2's table list has a name this node cannot parse, so a table added to a "
        "schema could hide in it")
    missing = sorted(tables - listed)
    assert not missing, (
        f"§2's enumeration omits {missing}, which the declared schemas create. The "
        "count and the list are the same claim stated twice; a reader who learns "
        "the tables from the list stops reading the number.")
    invented = sorted(listed - tables)
    assert not invented, (
        f"§2 enumerates {invented}, which no schema in {DDL_SCHEMAS} creates")
    assert len(listed) == _COUNT_WORDS[word], (
        "§2's list and its own count word disagree with each other")


def test_a_prose_create_table_is_not_counted_as_a_table():
    """The `(` in `_TABLE_DDL` is load-bearing, demonstrated on the real prose.

    `apply_grant_ddl`'s docstring in `workers/queue.py` quotes the literal
    `CREATE TABLE IF NOT EXISTS` while explaining why that statement is not a
    migration. A pattern that stopped at the identifier would return the table
    `IF` from a comment and hand §2 a count one too high, so the guard is
    exercised on that sentence rather than asserted — and the sentence is found
    from its own text between two symbols, not from a line number, because the
    numeric spelling of a citation into that file is banned across every tracked
    path (#2971, widened #3233) after `691df24b` moved `depth_by_source` from
    1226 to 1253 and left every citation naming its old line pointing somewhere
    else.
    """
    src = (ROOT / "workers" / "queue.py").read_text(encoding="utf-8")
    # Whitespace-normalised: the sentence is hard-wrapped in the docstring, and a
    # node quoting it with the wrong newline is a node that fails on a re-wrap.
    flat = " ".join(src.split())
    anchor = "`CREATE TABLE IF NOT EXISTS` alone is not a migration"
    assert anchor in flat, (
        "the prose literal this node exists to guard has moved out of "
        "`apply_grant_ddl`'s docstring; re-find such a sentence, or drop this node "
        "only if no DDL file quotes the statement in prose any more")
    clause = flat[flat.index("def apply_grant_ddl"):flat.index("conn.executescript")]
    assert anchor in clause, "the slice is not the docstring that carries the prose"
    assert _ddl_tables(clause) == set(), (
        f"`_ddl_tables` read {sorted(_ddl_tables(clause))} out of a docstring "
        "sentence, so the count it feeds §2's comparison is inflated by prose")
    queue_sql = importlib.import_module("workers.queue")._SCHEMA
    assert _ddl_tables(queue_sql + "\n" + clause) == _ddl_tables(queue_sql), (
        "appending prose to the schema changed the derived table set: the guard is "
        "not the trailing `(`")
    assert _ddl_tables(src) == _ddl_tables(queue_sql), (
        "regexing the whole of `workers/queue.py` no longer yields just the declared "
        "tables, so the trailing-`(` guard has stopped doing its job and the count "
        "§2 is graded against is one reading away from prose")


def test_the_table_pin_bites_in_both_directions():
    """Clause 3's other half: either side moving alone fails §2's node."""
    tables = _shared_tables()
    match = _TABLE_SITE.search(_md_section("## 2. The queue", _workers_md()))
    assert match, "the live §2 claim is unreadable; the node above says what to fix"
    listed = set(_DOC_TABLE.findall(match.group("tables")))
    assert listed == tables, "the live tree is inconsistent; fix that first"

    # (a) a ninth table declared by a schema, page untouched: clause 3's failure.
    ninth = "\nCREATE TABLE IF NOT EXISTS brand_new_ledger (\n  id INTEGER\n);\n"
    grown = _ddl_tables(importlib.import_module("workers.queue")._SCHEMA + ninth)
    assert grown - listed == {"brand_new_ledger"}, (
        "a table added to `_SCHEMA` is not reported as missing from §2, so a new "
        "shared table can ship with the count a month behind")
    assert _COUNT_WORDS[match.group("count")] != len(grown), (
        "the count comparison cannot see a ninth table")

    # (b) a name on the page no schema creates: the reverse direction.
    phantom = match.group("tables").replace(
        "`queue`", "`queue`, `phantom_ledger`", 1) + "\n"
    phantom_doc = _workers_md().replace(match.group("tables"), phantom, 1)
    re_search = _TABLE_SITE.search(_md_section("## 2. The queue", phantom_doc))
    assert re_search is not None, "the page mutation broke the capture, not the check"
    phantom_listed = set(_DOC_TABLE.findall(re_search.group("tables")))
    assert phantom_listed - tables == {"phantom_ledger"}, (
        "a table named on the page that no schema creates is not reported, so the "
        "reverse direction of the pin is not a check")
    assert len(phantom_listed) != _COUNT_WORDS[re_search.group("count")], (
        "the count word still matches a widened list, so §2 can enumerate nine and "
        "say eight")


#: §1's own count of the seats it then enumerates — the page states it bold, and
#: the bold is what makes the capture safe. `\s+` rather than a space because the
#: page is hard-wrapped. The retired count the fix replaced is the string `said
#: "two" for a month`, two sentences later and unbolded, which this capture
#: cannot reach — which is the whole reason a count word is worth reading at this
#: address when a count word read anywhere else in the file is worthless.
_SEC1_COUNT = re.compile(r"carries\s+\*\*(?P<count>[a-z]+)\*\*\s+deterministic seats")


def test_the_two_pins_pass_on_the_tree_as_it_stands_and_stray_counts_cannot_move_them():
    """Clause 4: the live tree satisfies both pins, and the file's other numbers
    are inert with respect to them.

    Three claims, three checks.

    (i) The tree as it stands. `WorkerPool._scheduler_loop` awaits five seats, §1
    enumerates those five AND states the count word `five` for them in the same
    sentence; §2 states `eight` and names the eight the three schemas declare.
    Both counts are compared to the parsed lists rather than to a literal typed
    here, so this node does not have to be edited the next time the pool grows —
    §1's count word does, which is the entire point.

    (ii) The unrelated numbers are inert. §1 still carries `said "two" for a
    month` in the sentence explaining the fix; §2's own body carries `seven
    youtube`; `seven task ids` and `all nine` sit further into the page; and the
    Review log records `the seventh shared table` and `now eight`. Each is
    mutated below to a word that belongs to the OTHER pin, and both parsed sets
    come back identical — which is what "anchored to §1's and §2's own
    sentences" has to mean before the phrase is worth claiming. A node that
    grepped the file for a number would be vacuous or permanently red here, which
    is why the item ruled that shape out.

    (iii) The Review log rows still say what their dates recorded. They are
    measurements, not promises, and the way a count-word pin destroys history is
    by getting a later editor to retype an old row to match the new reality.
    """
    src = POOL_SRC.read_text(encoding="utf-8")
    doc = _workers_md()
    sec1, sec2 = _md_section("## 1. Shape", doc), _md_section("## 2. The queue", doc)
    tables = _shared_tables()

    seats = _seats_awaited(src)
    named = _seats_named(sec1)
    assert set(seats) == named, "clause 1's node already names the disagreement"
    sec1_count = _SEC1_COUNT.search(sec1)
    assert sec1_count, (
        "§1 no longer states its own count in the form `carries **<word>** "
        "deterministic seats`; re-point `_SEC1_COUNT` rather than dropping this "
        "comparison — a list on the page with no number beside it is how `two` "
        "survived a month")
    assert _COUNT_WORDS[sec1_count.group("count")] == len(seats), (
        f"§1 says the tick carries {sec1_count.group('count')!r} seats and then "
        f"enumerates {len(seats)}: the count word and the list are two claims and "
        "only one of them was updated")
    assert sec1_count.group("count") != "two", (
        "§1's seat count reads `two` — the retired value, which stayed on the page "
        "for a month while the loop already awaited five")

    sec2_match = _TABLE_SITE.search(sec2)
    assert sec2_match, "clause 3's node already names an unreadable §2 claim"
    listed = set(_DOC_TABLE.findall(sec2_match.group("tables")))
    assert listed == tables, "clause 3's node already names the disagreement"
    assert _COUNT_WORDS[sec2_match.group("count")] == len(tables)
    assert len(seats) >= 2 and len(tables) >= 2, "the non-vacuity nodes say what to fix"

    # (ii) Each stray pushed to a number the other pin owns.
    strays = {}
    for needle, replacement in (
            ('said "two" for a month', 'said "eight" for a month'),
            ("seven youtube", "eight youtube"),
            ("seven task ids", "eight task ids"),
            ("all nine", "all eight"),
            ("the seventh shared table", "the eighth shared table"),
            ("now eight", "now five")):
        assert needle in doc, (
            f"{needle!r} has left the page, so either a dated record was edited (it "
            "must not be) or this node's idea of the file's stray numbers is stale")
        strays[needle] = doc.replace(needle, replacement, 1)
        assert strays[needle] != doc, f"could not mutate {needle!r}"
    for needle, mutated in strays.items():
        assert _seats_named(_md_section("## 1. Shape", mutated)) == named, (
            f"changing {needle!r} moved §1's parsed seat list, so the seat pin is "
            "reading a count word rather than the enumeration")
        mutated_match = _TABLE_SITE.search(_md_section("## 2. The queue", mutated))
        assert mutated_match is not None, (
            f"changing {needle!r} made §2's count claim unreadable, so the capture is "
            "not bounded to the sentence that makes the claim")
        assert (set(_DOC_TABLE.findall(mutated_match.group("tables"))),
                mutated_match.group("count")) == (listed, sec2_match.group("count")), (
            f"changing {needle!r} moved §2's list or its own count word, so the table "
            "pin can be satisfied or tripped from prose the schemas do not declare")

    # (iii) The dated rows, still saying what their dates measured.
    assert "the seventh shared table" in doc and "now eight" in doc, (
        "a dated Review log row was retyped to satisfy a count check. §2 is the "
        "present-tense claim and the only one a pin may edit; the log records what "
        "the measurement was on its date")


def test_workers_md_no_longer_states_that_the_pin_is_owed():
    """Clause 5: §1 stops promising a test that this file now contains.

    Bounded to §1 on purpose. §8's 2026-10-25 review row records "the pin is
    #3286" as the thing that review FILED, and a dated history row is a record,
    not a promise — editing it to satisfy a ban would destroy the measurement it
    carries. §1 is the present-tense page, and that is where a forward reference
    reads as work still owed.
    """
    doc = _workers_md()
    sec1 = _md_section("## 1. Shape", doc)
    for owed in ("asks for a test", "needs a test", "no test yet", "a test that keeps"):
        assert owed not in sec1, (
            f"§1 still says the seat list {owed!r}. The pin is in "
            "tests/test_workers_doc_claims.py; §1 should name it, not promise it.")
    assert "test_workers_doc_claims" in sec1, (
        "§1 no longer points a reader at the pin that holds the seat list, so the "
        "sentence that replaced the forward reference has stopped saying where the "
        "enforcement lives")
    assert "#3286" not in sec1, (
        "#3286 is cited in §1 in the present tense; it is closed by this round, and "
        "a live forward reference is what let the list rot once")
    # The ban is §1's, not the file's: the Review log's 2026-10-25 row records
    # "the pin is #3286" as the thing that review FILED, and that row is evidence.
    assert "#3286" in doc, (
        "`#3286` has vanished from the page entirely, so the ban above was widened "
        "into a deletion and a dated review row was edited to satisfy a test")


# ── #3288: the evidence denominator is read off a route, never off this page ─

#: The six counters, and the six only. A rename on either surface lands here.
_EVIDENCE_FIELDS = ("runs_with_bundle", "runs_without_bundle", "claims_checked",
                    "claims_verified", "claims_refuted", "claims_insufficient")

#: Shapes a measured bundle/claim figure takes in prose, and what each one is.
#: Every pattern is spliced-and-proved below, because a ban that matches nothing
#: would pass this node and re-license the very sentence #3288 was filed over.
_EVIDENCE_BANS = (
    (r"\d[\d,]*\s+(?:rows|runs)\b[^.]{0,80}?\bbundle",
     "a bundled-row count"),
    (r"\bbundled?\b[^.]{0,60}?\d[\d,]*\s+(?:rows|runs)\b",
     "a bundled-row count stated the other way round"),
    (r"\d[\d,]*\s+bundles?\b", "a bundle count"),
    (r"\d[\d,]*\s+claims\b", "a claim tally"),
    (r"\d[\d,]*\s+(?:verified|refuted|insufficient)\b", "a claim tally by verdict"),
    (r"(?:runs_with_bundle|runs_without_bundle|claims_checked|claims_verified"
     r"|claims_refuted|claims_insufficient|EVIDENCE_PILOT_TASK_IDS)[^\n]{0,25}\d",
     "a figure sitting next to a field name"),
)


def _evidence_sec() -> str:
    """§3 of `workers.md`, heading line excluded, asserted to hold the pilot.

    The whole section, not just its `### Evidence` subsection: the ban is on §3
    carrying a measured figure, and a slice that stopped at the next `###` would
    let the same sentence be pasted one heading down and read as compliant.
    """
    doc = _workers_md()
    sec = _md_section("## 3.", doc)
    assert "### Evidence" in sec, (
        "`### Evidence` has vanished from `## 3. What a source is` — the pilot's "
        "rules and its denominator pointer lived there, so a section shuffle "
        "that dropped it silently unmakes #525's contract")
    return sec


def test_the_evidence_denominators_are_read_off_the_routes_not_off_the_page():
    # Raw docstring: this one quotes the ban's own regex (`#\d+`), and in a
    # plain string `\d` is an invalid escape sequence — the static rung refuses
    # the SyntaxWarning, and the prose has to show the pattern as written.
    r"""Clause 5: §3 names the route and the six fields, and carries no count.

    Half of this node is a positive assertion. The pilot has exactly one
    readable size — the DB — and §3 is the page an operator reads before
    querying it, so the section has to send them somewhere: `GET
    /api/workers/health`, and the six field names it now serves per source. A
    section that only said "read the route" without the names would be a pointer
    to a payload nobody knows the shape of, which is the reason the prose count
    was written in the first place.

    The other half is the ban, and it is the reason this item exists. §3 once
    read "3 rows in `runs` carry one … every rate the health view derives from a
    bundle rests on a sample of three", and that sentence stood for six weeks
    after it stopped being true — widened to seven tasks and 202 rows by #2675
    with no code failing anywhere. #3288's arch-review pass moved the numbers
    into the route and *still* left both measurements in prose, hedged with "Read
    the denominator rather than this sentence". A hedge is not a rail: nobody
    re-runs a dated figure that is sitting right there looking like the answer.
    So a measured bundle count or claim tally in this section is now a failure,
    not an invitation.

    Item ids stay legal — `#\d+` is stripped before the patterns run, and a
    clause that banned citing the item that filed it would be banned itself.
    The pilot's seven task ids are likewise not banned here (this is not a
    no-digits-in-prose rule); they are simply not in this section any more,
    because they belong to `app/autonomy.py`, pinned verbatim by
    tests/test_worker_evidence.py and quoted in `architecture/autonomy-jobs.md`.
    """
    ev = _evidence_sec()

    assert "GET /api/workers/health" in ev, (
        "§3 stopped naming the route the denominator is read from, so the only "
        "place left to state it is prose again")
    for field in _EVIDENCE_FIELDS:
        assert f"`{field}`" in ev, (
            f"`{field}` is no longer named in the Evidence section: an operator "
            "reading this page before querying the store would not know the "
            "payload's shape")
    assert "denominator" in ev.lower()

    stripped = re.sub(r"#\d+", "", ev)
    for pattern, what in _EVIDENCE_BANS:
        hit = re.search(pattern, stripped)
        assert hit is None, (
            f"§3's Evidence section carries {what} as prose again: "
            f"…{ev[max(0, hit.start() - 60):hit.end() + 60]}… — the whole point "
            "of #3288 is that this figure has a route (`GET "
            "/api/workers/health`), because the prose version of it stood for "
            "six weeks after it stopped being true")


def test_the_ban_on_measured_evidence_figures_actually_bites():
    """Every pattern above must catch the real sentence it was written for.

    These examples are not invented for the test: each is the exact wording a
    dated pass left in this section — the 2026-10-25 arch-review paragraph that
    replaced the stale one while still quoting both measurements, and the
    sentence before it. They are spliced into the section IN MEMORY and the ban
    re-run, so the node proves the rail holds without leaving a single one of
    the figures in the committed doc. Without this, one mistyped escape in
    `_EVIDENCE_BANS` — `\\b` in the wrong place, a greedy `[^.]` that can never
    reach the word — turns the node above into a green light over the exact
    regression it names, and that is the failure this file's own header warns
    about: a doc test that asserts a shape rather than the fact.
    """
    ev = _evidence_sec()
    banned_examples = (
        "202 rows in `runs` carry a bundle as measured 2026-10-25",
        "3 rows in `runs` carry one, so every rate the health view derives "
        "from a bundle rests on a sample of three",
        "the 7-day fleet read 78 bundles over 253 runs",
        "919 claims checked, 877 verified, 16 refuted, 26 insufficient",
        "`runs_with_bundle` is 78 today, over 253 runs",
    )
    for example in banned_examples:
        spliced = ev + "\n" + example + "\n"
        stripped = re.sub(r"#\d+", "", spliced)
        hits = [what for pattern, what in _EVIDENCE_BANS
                if re.search(pattern, stripped)]
        assert hits, (
            f"the §3 prose ban accepted {example!r}, which is the measured "
            f"figure this node exists to refuse; every pattern missed it "
            f"({[p for p, _ in _EVIDENCE_BANS]})")


# ---------------------------------------------------------- config.yaml's own
# # Inner-Voice comments (#3285). Three sentences in `config.yaml` read as a
# fleet-wide Inner-Voice off while the same file says the opposite two blocks
# away: the comment above `workers.sources.board-steward.inner_voice: false` was
# "Off, like every other unattended turn since cut 1 of senses-not-supervision
# (2026-09-12)", and `arch-review` and `youtube-digest` each carried a copy of
# one paragraph calling themselves "OFF for unattended turns since 2026-09-12".
# The values were always right — `autotriage.inner_voice` and
# `autocode.inner_voice` are both true, re-armed 2026-09-25 — and
# `architecture/workers-jobs.md` §1 was corrected on 2026-09-25, so the config
# comment was the last carrier. `config.yaml` was never inside the existing
# sweep's reach: `test_no_live_file_asserts_the_inner_voice_refusal` walks
# `_scope()`, which is `architecture/*.md` + `workers/sources/*.py` and does not
# open `config.yaml` at all. These nodes read `config.yaml`'s own source blocks.

#: The board-steward sentence #3285 was filed from, verbatim. The file wraps it
#: over four comment lines; the readers below normalise a block, so both sides
#: of every comparison here are whitespace-normalised.
RETIRED_FLEET_WIDE_COMMENT = (
    "Off, like every other unattended turn since cut 1 of senses-not-supervision "
    "(2026-09-12). It landed the same day with no key and inherited the old True "
    "fallback, so it was the one source still watched on the primary every 900 s "
    "(#1015).")

#: The paragraph `arch-review` and `youtube-digest` each copied from the other,
#: verbatim, in the form that states an off as a property of unattended turns
#: rather than of the source whose key it sits beside.
RETIRED_CUT_ONE_PARAGRAPH = (
    "OFF for unattended turns since 2026-09-12 (plan: senses-not-supervision, "
    "cut 1). Measured effect on rounds was negative: #874 abandoned at iteration "
    "38 with 44 min left on an invented premise, 16 false repetition fires in one "
    "day, an inject into #875 with no room to answer. Stall rescue, budget and "
    "context pressure are deterministic anchors now; the gate is the real check. "
    "Stays on for chat, where a human reads it and its value was measured there.")

#: The stale verdict line both of those blocks opened with while their key read
#: `false` — the armed/disarmed pair inside one comment block.
RETIRED_VERDICT_LINE = ("On: it evaluates untrusted transcripts against Lloyd's "
                        "own profile and files backlog items from the answer.")

#: A comment that states the observer's state as a property of unattended turns
#: in general rather than of the source whose value it licenses. Each pattern is
#: shown to bite in the first node below — on the two verbatim retired sentences
#: and, for the third, on the same claim in the shape a later editor would write
#: it — because a ban whose pattern matches nothing is a ban on nothing (#1689).
_IV_FLEET_WIDE_BANS = (
    (r"(?:like|as\s+with|as\s+in)\s+(?:every|all)\s+(?:other\s+)?(?:unattended|background|scheduled)",
     "one source's off compared to every other unattended turn"),
    (r"\b(?:off|on)\s+for\s+unattended\s+turns\b",
     "an off stated as a property of unattended turns"),
    (r"\b(?:off|on|disabled|switched off)\s+for\s+(?:every|all|the\s+whole|entire)"
     r"\s+(?:unattended|background|scheduled)\s+turns?",
     "an off stated as applying to all unattended turns"),
)

#: A comment block's own verdict line, in the house style (`# Off: …`,
#: `# On: …`). Colon-terminated on purpose: `# OFF from 2026-09-12 …` is a dated
#: history clause, not this source's verdict.
_IV_VERDICT = re.compile(r"^(On|Off)\s*:", re.I)

_SRC_HEAD = re.compile(r"^    ([a-z][a-z0-9-]*):(?:\s*#.*)?$")


def _config_lines() -> list[str]:
    return (ROOT / "config.yaml").read_text(encoding="utf-8").splitlines()


def _keyed_inner_voice_sources() -> dict[str, bool]:
    """Every `workers.sources` block that carries an `inner_voice` key, parsed.

    Parsed rather than grepped, so the values are the shipped ones and a comment
    can never be checked against a line the loader would not read.
    """
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    src = cfg["workers"]["sources"]
    return {k: bool(v["inner_voice"]) for k, v in src.items()
            if isinstance(v, dict) and "inner_voice" in v}


def _inner_voice_comment(name: str, required: bool = True) -> tuple[str, list[str]]:
    """The `#` lines directly above `workers.sources.<name>.inner_voice`.

    Block-anchored per source for the reason `_comment_block_above` gives:
    `inner_voice:` sits on seven source blocks here, so a whole-file grep cannot
    say which block a sentence is licensing. Returns the normalised block and its
    raw lines (the verdict check needs the line starts back).
    """
    lines = _config_lines()
    si = next((i for i, ln in enumerate(lines) if ln == "  sources:"), None)
    assert si is not None, "config.yaml has no `  sources:` line; this reader is blind"
    heads = {i: _SRC_HEAD.match(ln) for i, ln in enumerate(lines)
             if i > si and _SRC_HEAD.match(ln)}
    start = next((i for i, m in sorted(heads.items()) if m.group(1) == name), None)
    assert start is not None, f"{name!r} has no block under `workers.sources`"
    end = next((i for i in sorted(heads) if i > start), len(lines))
    hits = [i for i in range(start + 1, end)
            if lines[i].startswith("      ")
            and lines[i].strip().startswith("inner_voice:")]
    assert len(hits) == 1, (
        f"{name} carries {len(hits)} `inner_voice` keys in its block; the reader "
        "cannot pick which comment belongs to the value")
    i = hits[0]
    raw: list[str] = []
    while i > 0 and lines[i - 1].lstrip().startswith("#"):
        raw.append(lines[i - 1].lstrip().lstrip("#").strip())
        i -= 1
    assert raw or not required, (
        f"no comment above {name}.inner_voice — the block was deleted, not corrected")
    return " ".join(" ".join(reversed(raw)).split()), list(reversed(raw))


def _config_comment_paragraphs() -> list[str]:
    """Every contiguous `#` comment block in `config.yaml`, whitespace-normalised.

    Paragraphs rather than lines because the retired sentence wrapped over four
    of them: a line-level scan would miss any claim that a later wrap split.
    """
    paras, cur = [], []
    for ln in _config_lines():
        if ln.lstrip().startswith("#"):
            cur.append(ln.lstrip().lstrip("#").strip())
        elif cur:
            paras.append(" ".join(" ".join(cur).split()))
            cur = []
    if cur:
        paras.append(" ".join(" ".join(cur).split()))
    return paras


def test_no_config_comment_claims_inner_voice_off_for_every_unattended_turn():
    """Clause 1 (#3285): the file may not state the observer's off state as a
    rule about unattended turns in general.

    The scan is the whole file's comment corpus, because the three carriers sat
    in three different source blocks and only one of them was the item's title
    sentence. Its sight is checked before its verdict is trusted: the corpus has
    to be big enough to be a scan, and every ban has to bite on the retired text
    it was written from — an empty result with a dead pattern reads exactly like
    a fixed file (#1689's rule, which this file already enforces twice above).
    """
    paras = _config_comment_paragraphs()
    assert len(paras) > 200, (
        f"the corpus is only {len(paras)} comment blocks; config.yaml has "
        "hundreds — the reader is not scanning the file")
    retired = (" ".join(RETIRED_FLEET_WIDE_COMMENT.split()),
               " ".join(RETIRED_CUT_ONE_PARAGRAPH.split()),
               "Inner Voice is off for every unattended turn since the re-arm")
    for pattern, what in _IV_FLEET_WIDE_BANS:
        assert any(re.search(pattern, r, re.I) for r in retired), (
            f"the ban on {what} matches none of the retired sentences it was "
            f"written from ({pattern!r}) — a pattern that matches nothing is a "
            "ban on nothing, so the empty result below would prove nothing")
    assert "like every other unattended turn" not in (ROOT / "config.yaml").read_text(
        encoding="utf-8"), "the exact sentence #3285 was filed from is still in the file"
    violations = [(p[:150], what) for p in paras for pattern, what in _IV_FLEET_WIDE_BANS
                  if re.search(pattern, p, re.I)]
    assert not violations, (
        f"{len(violations)} comment(s) still state Inner Voice's off state as a "
        f"property of unattended turns rather than of their own source: "
        f"{violations} — say what this source's value is, and where the "
        "on/off decision for it is made (#3285)")


def test_board_steward_inner_voice_comment_states_the_rule_its_value_encodes():
    """Clause 2 (#3285): the comment above `board-steward.inner_voice: false`
    has to state the rule that value encodes, and name its owner.

    The rule is not paraphrased here — it lives in `source_inner_voice`
    (`workers/sources/_common.py`), whose docstring carries the #1015 story this
    comment repeats. The comment points at that reader, at the fallback an
    unkeyed session source inherits, at where on/off is decided, and at the
    boundary that makes the key meaningless on a `run_prompt_on_primary` source.
    The tokens are checked literally because a comment that merely sounds like
    the rule is what this item is: the retired sentence was fluent, dated, and
    false two blocks away in the same file.
    """
    value = _keyed_inner_voice_sources()["board-steward"]
    assert value is False, (
        "board-steward is no longer off, so this node is pinning a comment to a "
        "value it never described — move the node with the value")
    block, _ = _inner_voice_comment("board-steward")
    assert " ".join(RETIRED_FLEET_WIDE_COMMENT.split()) not in block, (
        "the fleet-wide sentence is back; it is false two blocks away in this "
        "same file, which is what #3285 was filed from")
    for required in ("Off:", "per-source", "source_inner_voice", "unkeyed",
                     "#1015", "run_prompt_on_primary", "app/routers/messages.py",
                     "2026-09-25"):
        assert required in block, (
            f"the comment no longer names {required!r}: {block[:200]!r} — the "
            "half that says where the decision is made is the half a reader "
            "cannot re-derive from the value alone")


def test_every_cut_one_inner_voice_comment_carries_the_rearm_date():
    """Clause 3 (#3285): a comment that justifies its `false` with the 2026-09-12
    cut has to carry the 2026-09-25 re-arm beside it.

    That is the difference between a dated history and a rumour of a rule: cut 1
    did switch the keyed session sources off, and `autotriage` and `autocode`
    have been on again since 2026-09-25, so the date alone is the sentence that
    reads as fleet-wide. The same node holds each block's own verdict line to its
    value, because `arch-review` and `youtube-digest` both opened with `# On:`
    above `inner_voice: false` — the pair this item is about, one layer down.
    Both halves are shown to bite: on the verbatim retired paragraph, and on the
    verbatim retired `# On:` line.
    """
    keyed = _keyed_inner_voice_sources()
    assert len(keyed) >= 5, f"only {len(keyed)} sources key inner_voice; not a sweep"
    assert ("2026-09-12" in RETIRED_CUT_ONE_PARAGRAPH
            and "2026-09-25" not in RETIRED_CUT_ONE_PARAGRAPH), (
        "the retired cut-1 paragraph no longer trips this node's own rule, so an "
        "empty result below would be the pattern dying, not the claim dying")
    legacy_verdict = _IV_VERDICT.match(RETIRED_VERDICT_LINE)
    assert legacy_verdict and (legacy_verdict.group(1).lower() == "on"), (
        "the retired `# On:` line no longer parses as a verdict, so the "
        "verdict/value check below cannot be shown to bite")
    cut_one, verdicts, stale = [], [], []
    for name, value in sorted(keyed.items()):
        block, raw = _inner_voice_comment(name, required=False)
        if "2026-09-12" in block:
            cut_one.append(name)
            assert "2026-09-25" in block, (
                f"{name}'s comment still gives the 2026-09-12 cut as the reason "
                f"for inner_voice: {str(value).lower()} with no re-arm beside it "
                f"— {block[:160]!r}: cut 1 was not the end of the rule, and a "
                "reader of one block cannot see the sources that came back")
        seen = [m.group(1) for m in map(_IV_VERDICT.match, raw) if m]
        if seen:
            verdicts.append(name)
            if (seen[0].lower() == "on") is not value:
                stale.append((name, seen[0], value))
    assert len(cut_one) >= 3, (
        f"only {cut_one} cite 2026-09-12; the rule is reading an empty corpus")
    assert len(verdicts) >= 3, (
        f"only {verdicts} carry a verdict line; the check is reading an empty corpus")
    assert not stale, (
        f"{stale}: a block's own verdict line disagrees with the value beneath it "
        "— #3285 is exactly this pair, and `# On:` above `inner_voice: false` is "
        "how the fleet-wide reading was written in the first place")


# ── #3287: §8's failure count is a dated record, never a standing number ─────

#: The retired live-reading framings, quoted byte-exact from the three files that
#: carried them (#522's era, read 2026-09-09, `a4e52633`). Kept as constants here
#: so the ban and the non-vacuity splice quote the SAME strings rather than two
#: paraphrases of them.
_BENCH_LIVE_CLAIMS = (
    "104 eligible failures in the last 7 days",
    "104 failed runs in the last 7 days",
    "104 failed runs in 7 days",
    "4,044 run files on disk",
    "4,044 unused failure transcripts",
)


def _bench_mine_sentences() -> list[str]:
    """§8's bench-mine bullet as sentences, whitespace normalised."""
    return _BENCH_SENTENCE.split(" ".join(_bench_mine_block().split()))


def test_the_selector_is_described_and_its_era_count_is_historical():
    """Clause 1: no present-tense standing failure count, and no thinner prose.

    The sentence "It now scans `AUTONOMY_RUNS_DIR/**/run_*.md` for
    `status: failed` on every tick" is the only place on this page that states
    what the source's second input reads, so it has to survive whatever happens
    to the number beside it — deleting the clause is only allowed if it leaves
    the selector standing. The clause being banned is a count attached to that
    tick-scoped present-tense sentence, because that pairing is what made a
    2026-09-09 reading read like a property of the code: the era it measured is
    deleted by the 30-day retention sweep
    (`scripts/groundskeeper/retention-sweep.py`) and cannot be re-stamped, so the
    only honest forms are past tense with its own date, or silence.

    The count itself is allowed to remain only as a dated record. That half is
    not re-checked here — `_bench_mine_count_sentences` in the node above already
    refuses any undated run/failure count in this block — so this node asks only
    the tense question: is there a count sitting inside the sentence that says
    the source does this on every tick?
    """
    sentences = _bench_mine_sentences()
    joined = " ".join(sentences)

    assert "`AUTONOMY_RUNS_DIR/**/run_*.md`" in joined, (
        "§8's bench-mine bullet no longer names the path the failed-run input "
        "walks, so the clause was deleted along with the sentence it lived in")
    assert "`status: failed`" in joined and "on every tick" in joined, (
        "the selector sentence went with it: the reader is left with a number "
        "and no statement of what was counted")

    standing = [s for s in sentences if "on every tick" in s
                and _BENCH_COUNTED.search(s)]
    assert not standing, (
        "a count is still attached to the tick-scoped present-tense sentence, so "
        f"it reads as a standing property of the code again: {standing[0][:200]!r}")
    for claim in _BENCH_LIVE_CLAIMS:
        assert claim not in joined, (
            f"{claim!r} is back verbatim in §8's bench-mine block — the era it "
            "measured is pruned and cannot be re-read, so it can only be quoted "
            "as a dated record in its own past-tense sentence")


def test_the_ban_on_a_standing_failure_count_actually_bites():
    """The two §8 checks above must refuse the exact text #3287 was filed over.

    A regex that cannot fire is a green light, and both checks here are pure
    text predicates, so they can be exercised on a block that does not exist any
    more: the retired sentence is spliced back into the real bullet IN MEMORY —
    nothing is written — and both the tick-pairing test and the widened
    undated-count guard have to report it. Without this, widening
    `_BENCH_COUNTED`'s noun list or narrowing the tick-pairing predicate would
    silently un-make the rail this item exists to install.
    """
    block = " ".join(_bench_mine_block().split())
    retired = ("It now scans `AUTONOMY_RUNS_DIR/**/run_*.md` for `status: failed` "
               "on every tick, gated by nothing the ledger does — 104 eligible "
               "failures in the last 7 days.")
    spliced = block.replace(
        "It now scans `AUTONOMY_RUNS_DIR/**/run_*.md` for `status: failed` on "
        "every tick,", retired, 1)
    assert spliced != block, (
        "the selector sentence this splice targets is no longer the sentence it "
        "was when #3287 was filed, so the splice no longer proves anything and "
        "has to be re-pointed at the current wording")

    spliced_sentences = _BENCH_SENTENCE.split(" ".join(spliced.split()))
    standing = [s for s in spliced_sentences
                if "on every tick" in s and _BENCH_COUNTED.search(s)]
    assert standing, (
        "the spliced retired sentence passed the tick-pairing check, so that "
        "check cannot catch the regression it names")

    _checked, undated = _bench_mine_count_sentences(spliced)
    assert any("104 eligible failures" in s for s in undated), (
        "the widened undated-count guard let the retired sentence through, so "
        "the widening this round installed is not real")


def test_the_retired_counts_are_dated_records_in_the_source_and_the_jobs_page():
    """Clause 4: the two other copies of the same figure stop claiming liveness.

    `workers/sources/bench_mine.py` opened its own module docstring with "104
    failed runs in the last 7 days against 4,044 run files on disk" and
    `enqueue_if_due` repeated "4,044 unused failure transcripts";
    `architecture/workers-jobs.md` carried "104 failed runs in 7 days against
    4,044 run files on disk" inside the #522 paragraph. All three were read
    2026-09-09 and none said so, and the 4,044 was 4x the live count on disk
    today — a module docstring is the most-freely-quoted prose in the tree, so
    an unstamped figure there propagates faster than one on a docs page.

    Docstrings are read with `ast`, not by importing: this file's own rule is
    that a doc claim must not pull a module's import side effects into the
    suite, and `bench_mine` opens a queue and a config on import.

    What must SURVIVE is the lesson, not the numbers — that `enqueue_if_due`
    returned early and the input stayed unreachable, and that the queue held
    zero rows for its whole life — so those phrases are pinned as well. A pass
    that deleted the paragraphs to make the ban green would fail here, which is
    the difference between removing a claim and removing a record.
    """
    import ast  # precedent at :775 and :955 of this file, not a module import

    src = (ROOT / "workers" / "sources" / "bench_mine.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    mod_doc = ast.get_docstring(tree) or ""
    enqueue_doc = next(
        (ast.get_docstring(node) or "" for node in tree.body
         if isinstance(node, ast.AsyncFunctionDef) and node.name == "enqueue_if_due"),
        None)
    assert enqueue_doc, (
        "`enqueue_if_due` lost its docstring or changed kind, so this node has "
        "nothing to grade and the 4,044 copy inside it is ungraded again")

    jobs = (ARCH / "workers-jobs.md").read_text(encoding="utf-8")
    anchor = "#522 found it enabled"
    assert anchor in jobs, (
        "the #522 paragraph anchor moved on the jobs page, so the slice below is "
        "no longer the passage that carried the figure")
    jobs_para = jobs[jobs.index(anchor):jobs.index(anchor) + 900]

    for where, text in (("bench_mine.py module docstring", mod_doc),
                        ("bench_mine.py `enqueue_if_due` docstring", enqueue_doc),
                        ("architecture/workers-jobs.md's #522 paragraph", jobs_para)):
        for claim in _BENCH_LIVE_CLAIMS:
            assert claim not in text, f"{where} still asserts {claim!r} as a count"
        for sent in _BENCH_SENTENCE.split(" ".join(text.split())):
            if re.search(r"\b(?:104|4,044)\b", sent) and re.search(r"run|failure", sent):
                assert _BENCH_DATED.search(sent), (
                    f"{where} quotes one of the #522-era figures with no date on "
                    f"its own sentence: {sent[:170]!r} — the class rule this item "
                    "is an instance of is that a number without the moment it was "
                    "read becomes a standing claim")

    assert "never once enqueued" in mod_doc and "never opened the directory" in mod_doc, (
        "the module docstring's account of WHY the input never fired was removed "
        "with the figure; the mechanism is the part worth keeping")
    assert "returned early" in enqueue_doc, (
        "`enqueue_if_due` no longer says its first cut returned early, which is "
        "the bug its call order is arranged around")
    assert "never once enqueued" in jobs_para and \
        "the directory was never opened" in jobs_para, (
        "the jobs page's #522 paragraph was emptied rather than de-dated: the "
        "finding it records is worth keeping, only the number beside it was not")


# ── #3527: a published failure RATE needs the span of the failures it counts ─
#
# #3287's `_BENCH_COUNTED` guard above asks one question of a published figure: was
# it dated? It was written for a COUNT (`failed 40`), where the honest repair is one
# clause with one number in it. A rate is a different object. `fail_rate` is a
# quotient over a rolling window, so its numerator is a POPULATION — a set of rows
# with a first day and a last day — and a date on the sentence quoting the quotient
# dates the READING, not what was counted. That distinction is what #3287 cannot
# see, and it cost a round its judgement: this page's §5 youtube-digest bullet read
# "Measured 2026-10-28 over `/api/workers/health?days=7`, unclamped … 100 runs, 51
# failed — **a 51 % fail rate**". Every word of it was true. The failures it counted
# ran 2026-10-18 through 2026-10-23, and `2a5778b6` had reverted the two-stage pilot
# that produced them five days before the read — so the sentence published a closed
# burst as the source's standing rate. The reader it misled was the reviewer of
# `group:workers-jobs:Intake`, which WROTE those lines and filed #3523 at `high`
# against a source whose burst had already closed, because `git grep` over the
# payload found no field that could have said otherwise. #3527 adds
# `last_failure` and `consecutive_failures` to that payload; this is the prose side
# of the same fix: a sentence on this page publishing a worker-source failure RATE
# measured over a stated window must also end-date the failure population it counts.
# "Measured 2026-10-28" does not qualify — that is the day of the reading.
#
# Scope is narrow on purpose, in the way #3134 taught. The trigger is a sentence
# naming a `/api/workers/health` window or a `days=N` / N-day window AND carrying a
# rate token beside a failure word. An unscoped "any percentage on this page" rule
# would fire on the judge-flip's `agreement 50% of 2 judged` 500 lines up and on the
# eval bar's 90 %, and a guard that cries wolf is a guard someone deletes.

_FAIL_RATE_NEAR_A_FAILURE = re.compile(
    r"\b(?:fail|failure|failures|failing|failed)\b[^.]{0,80}?"
    r"(?:\d+(?:\.\d+)?\s?%|\bfail_rate\b)"
    r"|(?:\d+(?:\.\d+)?\s?%|\bfail_rate\b)[^.]{0,80}?"
    r"\b(?:fail|failure|failures|failing)\b", re.IGNORECASE)
_MEASURED_WINDOW = re.compile(
    r"/api/workers/health|\bdays=\d+\b|\b\d+-day\s+(?:rolling\s+)?window\b"
    r"|\b\d+\s*days?\s+to\s+20\d{2}-\d{2}-\d{2}\b", re.IGNORECASE)
_POPULATION_END = re.compile(
    r"(?:through|up\s+to|until|ended|ends|stopped|to)\s*20\d{2}-\d{2}-\d{2}"
    r"|\b(?:last_failure|consecutive_failures)\b", re.IGNORECASE)
_JOBS_PAGE = ARCH / "workers-jobs.md"
_RATE_LOOKAHEAD_CHARS = 400


def _sentences(page: str) -> list[str]:
    """Split on a sentence-ending period followed by whitespace.

    This page writes the measurement across three sentences of one bullet — the
    window claim, then "100 runs, 51 failed — a 51 % fail rate", then the count of
    the rows — so a rule keyed to "the sentence holding the %" finds nothing to
    check on the live page and reports that as a pass.
    """
    return re.split(r"(?<=[.!?])\s+", page)


def _rate_guard_hits(page: str) -> list[tuple[int, str]]:
    """(index, sentence) for every sentence this guard takes as publishing a rate.

    A candidate must (a) name a `/api/workers/health` window or a `days=N` / N-day
    window, and (b) have a rate token within eighty characters of a failure word, in
    that sentence or the one before — before, because that is how the page writes it.
    The failure word is what keeps the guard off the page's other rates (the judge
    flip's agreement percentage, the eval bar), none of which has ever misled anyone.
    """
    parts = _sentences(page)
    return [(i, s) for i, s in enumerate(parts)
            if _MEASURED_WINDOW.search(s)
            and _FAIL_RATE_NEAR_A_FAILURE.search(" ".join(parts[max(0, i - 1):i + 2]))]


def _rate_guard_candidates(page: str) -> list[str]:
    """The triggered sentences, unnamed — the guard's own witness of what it covers.

    Exposed so a test can assert this list is non-empty and of a known size. A rule
    that silently matches nothing is the failure this file has been caught by twice
    (#3134's retired-name list, #3106's regex written against an absent `::`), and it
    is invisible from a green check mark: `all(x for x in [])` is True forever.
    """
    return [s for _i, s in _rate_guard_hits(page)]


def _rates_missing_a_population_end(page: str) -> list[str]:
    """Triggered sentences that publish their rate with no end-date on the population.

    Judged on a 3-sentence / 400-character lookahead rather than on the candidate
    alone: the span is normally given just after the number (the live bullet names it
    two sentences on), so reading only the candidate would pass the exact misreading
    this exists to catch while failing honest prose that states the number last. The
    400-character bound is the other half — long enough for those two sentences,
    short enough that an end-date three bullets away cannot stand in for the one this
    sentence's own population needs.
    """
    parts = _sentences(page)
    return [s for i, s in _rate_guard_hits(page)
            if not _POPULATION_END.search(
                " ".join(parts[i:i + 3])[:_RATE_LOOKAHEAD_CHARS])]


def test_every_workers_jobs_failure_rate_names_the_span_it_counts():
    """No worker-source failure rate on this page is published as a standing property.

    This node fails if a later round re-quotes a `/api/workers/health` rate without
    saying when the failures behind it stopped — the sentence that opened #3523 — and
    it also fails if the trigger stops matching anything, because at that point the
    rule has quietly stopped existing while still passing.
    """
    page = _JOBS_PAGE.read_text(encoding="utf-8")
    triggered = _rate_guard_candidates(page)
    assert len(triggered) == 3, (
        f"the rate guard's trigger matched {len(triggered)} sentences, not the "
        f"three this node was written against: {[t[:110] for t in triggered]} — a "
        "count rather than a name-list so that honest prose ADDED to the page moves "
        "it in the open instead of letting the guard go quiet (#3134). Re-check each "
        "one and adjust the number here, in the assertion, not by widening the "
        "trigger.")
    offenders = _rates_missing_a_population_end(page)
    assert not offenders, (
        "a worker-source failure rate is published here with no end-date on the "
        "population it counts. A rate over a rolling window is not a property of the "
        "source: #3523 was filed at `high` off 'a 51 % fail rate' whose failures all "
        "sat behind a revert five days old, and the only way to see that was to open "
        "workers.db by hand. Name the span the failures ran (`…through 2026-10-23`), "
        "or read the payload's own `last_failure`. Offending sentences: "
        f"{[o[:150] for o in offenders]}")


_RETIRED_51_PERCENT_BULLET = (
    "Measured 2026-10-28 over `/api/workers/health?days=7`, unclamped "
    "(`window_clamped_to_hours: null`, `oldest_input: null`, so a real 7 days): "
    "100 runs, 51 failed — **a 51 % fail rate**, 5.96 h GPU. Counting the 50 "
    "`failed` rows in `runs`: 47 read `turn ended (stop, N iterations) without a "
    "note at <path>` with N between 3 and 13, 2 read `turn produced nothing` (one "
    "of them the backend answering `All connection attempts failed`), and **0** are "
    "the `--fetch rc=2` argparse burst, last seen 2026-09-09, that this line called "
    "the whole of its 92 failures. The retry design below is working exactly as "
    "described and still loses about half the digests. Filed as #3523.")


def test_the_retired_51_percent_bullet_is_gone_and_still_trips_the_guard():
    """The corrected reading of the burst, plus the negative control for the guard.

    Both halves of the misreading are asserted gone by string. `**a 51 % fail
    rate**` published a quotient as a property. `still loses about half the digests`
    was false about the outcome as well as the rate, and the corrected bullet now
    carries the accounting instead: the retries landed, `uTrY8slys0w` reached its
    note on the 8th attempt (vault commit `470a9aae`), the two rows that reached
    `quarantined` were re-offered on fresh rows and digested, and `YvQ4BVTWfHI` — 12
    failed / 0 success on 2026-10-21 and 2026-10-22, no note under
    `~/obsidian/knowledge/youtube/` — is the one confirmed permanent loss of the
    burst that ran 2026-10-18 through 2026-10-23.

    Then the retired text is spliced back IN MEMORY and handed to the live guard.
    Without that second half this node is an absence check that outlives its own
    rule: the day the trigger regex rots, the string assertions still pass and the
    guard is dead, which is #3287's lesson about a proving command that returns zero.
    """
    page = _JOBS_PAGE.read_text(encoding="utf-8")
    assert "**a 51 % fail rate**" not in page, (
        "the bolded standing rate is back: that phrasing is what read a closed burst "
        "as a live defect and cost #3523 its `high` priority")
    assert "still loses about half the digests" not in page, (
        "the page again claims the retry design loses half the digests. It does not: "
        "the failures were re-offered and landed, and one video is the loss")
    assert "YvQ4BVTWfHI" in page, (
        "the corrected bullet has to name the ONE real loss by id, or the next reader "
        "re-derives 'about half' off a rolling fail_rate and reopens #3523")

    anchor = "- Its failure population is a closed burst"
    assert anchor in page, (
        "the youtube-digest bullet's anchor moved, so the splice below splices "
        "nothing and this node's negative control is ungraded")
    start = page.index(anchor)
    end = page.index("\n- ", start + len(anchor))
    spliced = (page[:start] + "- " + _RETIRED_51_PERCENT_BULLET + "\n"
               + page[end + 1:])
    offenders = _rates_missing_a_population_end(spliced)
    assert offenders, (
        "the retired 51 % bullet was spliced back onto the page and the guard above "
        "found nothing to complain about, so the rule is a no-op and everything in "
        "this node is decoration")
