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
    for n, ln in hits:
        assert "service_probe" in ln, (
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
