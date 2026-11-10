"""P5 — the skill index can say what each skill is for; bodies can be pulled.

`prompt_builder._load_skills_index` renders the `<available_skills>` block that
sits in the cached system-prompt prefix. With `skills.index.descriptions` off
(the default) it must be today's names-only line byte for byte; on, it is one
`- name — description` line per skill under `skills.index.budget_chars`, the
most-used skills (30-day telemetry) getting descriptions first and the lines
rendered alphabetically whatever the rank. `prefetch.skills.push: false` turns
off body injection and swaps the index's closing note for a pull instruction.

Hermetic: skill trees under `tmp_path`, config keys set through `monkeypatch`.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))

import agent_mcp.skills as skills_mod  # noqa: E402
from app import prefetch  # noqa: E402
from app import prompt_builder as pb  # noqa: E402
from app.config import CONFIG  # noqa: E402

#: The real ranker, captured before the fixture stubs it out.
_REAL_RANK = pb._skill_rank_counts

# Today's rendering, spelled out rather than imported: if the builder's copy of
# either string changes, this file is the thing that must notice.
OLD_NOTE = ("Note: relevant skill content is automatically injected into each "
            "user message as <context> when matched.")


def _skill(root: Path, name: str, description: str | None) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    fm = ["---", f"name: {name}"]
    if description is not None:
        fm.append(f"description: {json.dumps(description)}")
    fm += ["tags: [x]", "---", "", f"BODY of {name}."]
    (d / "SKILL.md").write_text("\n".join(fm) + "\n", encoding="utf-8")


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """One root; `skills.index` / `prefetch.skills` reset to the shipped defaults."""
    root = tmp_path / "skills"
    monkeypatch.setattr(skills_mod, "SKILLS_DIRS", [root])
    monkeypatch.setattr(pb, "_CANON_SKILLS_DIRS", None)
    monkeypatch.delenv("LLOYD_OVERLAY_DIR", raising=False)
    skills_cfg = dict(CONFIG.get("skills") or {})
    skills_cfg["index"] = {"descriptions": False, "budget_chars": 12000,
                           "max_description_chars": 100}
    monkeypatch.setitem(CONFIG, "skills", skills_cfg)
    monkeypatch.setitem(CONFIG, "prefetch", {"skills": {"push": True}})
    # No telemetry unless a test seeds some: the rank is alphabetical.
    monkeypatch.setattr(pb, "_skill_rank_counts", lambda: {})
    return root


def _set_index(**kw):
    CONFIG["skills"]["index"] = dict(CONFIG["skills"]["index"], **kw)


def _skills_component(**kw) -> str:
    prompt = pb.build_system_prompt(**kw)
    start = prompt.index("<available_skills>")
    return prompt[start:]


def test_off_is_bytewise_today(tree):
    """`descriptions: false` renders the names-only line and today's note —
    the literal old block, reconstructed here — so shipping this changes no
    byte of any production prompt."""
    for name, desc in (("zeta", "Use when zeta."), ("alpha", "Use when alpha."),
                       ("mid", None)):
        _skill(tree, name, desc)

    index = pb._load_skills_index()
    assert index == "Available skills: alpha, mid, zeta"
    old_block = f"<available_skills>\n{index}\n</available_skills>\n{OLD_NOTE}"
    assert _skills_component().startswith(old_block)
    assert pb._skills_index_note(True) == OLD_NOTE


def test_fits_budget_with_189_skills(tree):
    """The live library's shape (189 skills, descriptions ~190 chars on
    2026-09-24): every skill is listed, the index stays under the budget, some
    skills get descriptions and none is longer than the clip."""
    for i in range(189):
        _skill(tree, f"skill-{i:03d}-{'x' * (i % 12)}",
               f"Use when task number {i} needs doing. " + "detail " * 25)
    _set_index(descriptions=True)

    index = pb._load_skills_index()
    lines = index.splitlines()[1:]
    assert len(index) <= 12000, len(index)
    assert len(lines) == 189, "a skill was dropped to make room"
    described = [ln for ln in lines if " — " in ln]
    assert 40 < len(described) < 189, len(described)
    for ln in described:
        assert len(ln.split(" — ", 1)[1]) <= 100
        assert ln.endswith("…")


def test_names_alone_over_budget_still_lists_every_skill(tree):
    for i in range(30):
        _skill(tree, f"s{i:02d}", "Use when needed.")
    lines = pb._skills_index_lines(list(skills_mod.iter_active_skills()), {},
                                   budget=50, max_desc=100)
    assert len(lines) == 31 and not any(" — " in ln for ln in lines[1:])


def test_missing_description_renders_the_name(tree):
    _skill(tree, "bare", None)
    _skill(tree, "empty", "")
    _skill(tree, "full", "Use when full.")
    _set_index(descriptions=True)

    lines = pb._load_skills_index().splitlines()[1:]
    assert lines == ["- bare", "- empty", "- full — Use when full."]


def test_rendered_order_is_alphabetical_regardless_of_rank(tree):
    """Rank decides WHO gets a description — here only the top-used skill fits
    — and never where a line sits."""
    for name in ("apple", "mango", "zebra"):
        _skill(tree, name, f"Use when {name} " + "y" * 60)
    active = list(skills_mod.iter_active_skills())
    counts = {"zebra": 50, "mango": 5}
    names_only = len("\n".join(pb._skills_index_lines(active, {}, budget=1, max_desc=100)))
    one_more = names_only + len(" — ") + len("Use when zebra " + "y" * 60)
    lines = pb._skills_index_lines(active, counts, budget=one_more, max_desc=100)
    assert len("\n".join(lines)) == one_more

    assert [ln[2:].split(" — ")[0] for ln in lines[1:]] == ["apple", "mango", "zebra"]
    assert [ln for ln in lines[1:] if " — " in ln] == [lines[3]], lines
    assert lines[3].startswith("- zebra — ")

    # No telemetry: the alphabetical head gets it instead.
    lines = pb._skills_index_lines(active, {}, budget=one_more, max_desc=100)
    assert lines[1].startswith("- apple — ") and " — " not in lines[3]


def test_index_and_api_skills_agree(tree):
    """#1294: the description the model is shown is the one the Skills page
    shows — the route's field, whitespace folded and clipped, never another
    reading of the front matter."""
    from app.routers.skills import get_skills

    _skill(tree, "short", "Use when short.")
    _skill(tree, "long", "Use when long.\n" + "word " * 40)
    _skill(tree, "none", None)
    _set_index(descriptions=True)

    rows = json.loads(asyncio.run(get_skills()).body)["workspace"]
    route = {r["name"]: r["description"] for r in rows}
    index = pb._load_skills_index().splitlines()[1:]
    assert [ln[2:].split(" — ")[0] for ln in index] == sorted(route)
    for ln in index:
        name, _, shown = ln[2:].partition(" — ")
        expect = pb.clip_skill_description(" ".join(route[name].split()), 100)
        assert shown == expect, (name, shown, expect)


def test_pull_arm_changes_the_note_and_plans_no_body(tree):
    _skill(tree, "alpha", "Use when alpha.")
    CONFIG["prefetch"]["skills"]["push"] = False

    assert "skills_read(name)" in _skills_component()
    assert OLD_NOTE not in _skills_component()
    scored = [(9.0, {"name": "alpha", "raw": "alpha body"}),
              (8.0, {"name": "beta", "raw": "beta body"})]
    assert prefetch._skill_injection_plan(scored) == []
    assert "<skill" not in prefetch._format_context(scored, [], show_skill_hint=False)

    CONFIG["prefetch"]["skills"]["push"] = True
    assert [m for _, _, m in prefetch._skill_injection_plan(scored)] == ["body", "excerpt"]


def test_unreadable_settings_keep_today(tree, monkeypatch):
    CONFIG["skills"]["index"] = {"descriptions": "yes", "budget_chars": -1,
                                 "max_description_chars": True}
    s = pb._skills_index_settings()
    # `advertise_top_n: None` IS today's behaviour, not a fourth knob switched on: None means
    # every active skill gets an index line, which is what production has always rendered and
    # what #4027's pruned arm exists to deviate from. Garbage in that key cannot reach a number
    # through it either, same as the other three.
    assert s == {"descriptions": False, "budget_chars": 12000, "max_description_chars": 100,
                 "advertise_top_n": None, "chat_scope": "all"}
    CONFIG["skills"]["index"]["advertise_top_n"] = "seven"
    assert pb._skills_index_settings()["advertise_top_n"] is None
    monkeypatch.setitem(CONFIG, "prefetch", {"skills": {"push": "no"}})
    assert pb.skills_push_enabled() is True   # only a literal false switches it


def test_advertise_top_n_prunes_the_index_and_nothing_else(tree):
    """#4027: the set-restricting seam is subtractive at the INDEX and grants nothing else.

    `_load_skills_index(overlay=…)` only ever ADDED skills, so a pruned arm needed a new seam —
    and a new seam in the prompt builder is exactly where an unintended capability creeps in. The
    invariants: the cut keeps the top N by the same rank the description grant walks (ties by
    name, so it is reproducible); it applies to the names-only rendering too, or an arm could be
    measured on an index it never pruned; every other skill is still returned by the walker,
    because a pruned skill has to stay loadable by `skills_read` for a tail query to be a
    deferred-task control; and the budget cannot be what does the cutting — an oversized budget
    must still yield exactly N names.
    """
    # Names chosen so alphabetical order (alpha, mango, zebra) is NOT the rank order
    # (zebra 30, alpha 20, mango 10): a node whose fixture sorts the same both ways cannot
    # tell a rank cut from an alphabetical cut, which is the difference between this seam and
    # the budget's own behaviour.
    for name, desc in (("zebra", "Z"), ("alpha", "A"), ("mango", "M")):
        d = tree / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\nbody {name}\n")
    CONFIG["skills"]["index"] = {"descriptions": True, "budget_chars": 12000,
                                 "max_description_chars": 100}
    real = pb._skill_rank_counts
    pb._skill_rank_counts = lambda: {"zebra": 30, "alpha": 20, "mango": 10}
    try:
        full = pb._load_skills_index(overlay=tree)
        assert [ln for ln in full.splitlines() if ln.startswith("- ")] == [
            "- alpha — A", "- mango — M", "- zebra — Z"], full
        cut = pb._load_skills_index(overlay=tree, advertise_top_n=2)
        assert [ln for ln in cut.splitlines() if ln.startswith("- ")] == [
            "- alpha — A", "- zebra — Z"], cut      # mango (rank 10) is the one that went
        # The budget is generous here, so only the seam can have removed a name.
        assert pb._load_skills_index(overlay=tree, advertise_top_n=99) == full
        assert pb._load_skills_index(overlay=tree, advertise_top_n=None) == full
        # Names-only rendering is pruned by the same cut, or the two renderings disagree about
        # who is advertised and a pruned arm could be scored against an unpruned surface.
        names_only = pb._load_skills_index(overlay=tree, descriptions=False, advertise_top_n=1)
        assert names_only == "Available skills: zebra", names_only
        # Nothing left the library: the pruned name is still walked, still readable.
        from agent_mcp.skills import iter_active_skills
        roots = [tree] + [d for d in pb._skill_walk_roots() if d != tree]
        assert {s.name for s in iter_active_skills(roots=roots)} >= {"zebra", "alpha", "mango"}
        assert (tree / "mango" / "SKILL.md").is_file()
    finally:
        pb._skill_rank_counts = real


def test_rank_counts_join_offers_loads_and_reads_over_chat_sessions_only(tree, monkeypatch):
    """The rank is offers + loaded + loaded_by_read over 30 days, read from CHAT
    sessions only (2026-11-09: 97% of offers were worker traffic)."""
    import app.skill_telemetry as tel

    seen = {}

    def fake(root, days, *, include=None):
        seen["days"] = days
        seen["include"] = include
        return {"skills": {"a": {"offers": 3, "loaded": 1}},
                "loaded_by_read": {"a": 2, "b": 4}}

    monkeypatch.setattr(tel, "skill_injection_counts", fake)
    assert pb._compute_skill_rank_counts() == {"a": 6, "b": 4}
    assert seen["days"] == 30
    assert seen["include"]("20261109_101500") is True                  # a chat id
    assert seen["include"]("20261109_101500_autotriage_0fa8") is False  # a worker run


def test_the_rank_is_never_computed_on_the_callers_thread(tree, monkeypatch, tmp_path):
    """2026-11-09: the scan ran inline in the synchronous prompt build (13.2 s, on the
    event loop). Now a call returns at once — yesterday's persisted rank, or {} — and
    one background refresh per day computes, persists and publishes today's."""
    import threading

    import app.paths as paths

    monkeypatch.setattr(paths, "SKILLS_INDEX_RANK_PATH", tmp_path / "rank.json")
    (tmp_path / "rank.json").write_text(json.dumps({"date": "2000-01-01", "counts": {"old": 1}}))
    gate = threading.Event()
    calls = []

    def slow():
        calls.append(1)
        gate.wait(5)
        return {"new": 7}

    monkeypatch.setattr(pb, "_compute_skill_rank_counts", slow)
    pb._rank_cache.clear()
    pb._rank_refresh_started.clear()
    try:
        assert _REAL_RANK() == {"old": 1}, "a call waited for the scan"
        assert _REAL_RANK() == {"old": 1}
        gate.set()
        for t in threading.enumerate():
            if t.name == "skills-index-rank":
                t.join(5)
        assert calls == [1], "more than one refresh started in a day"
        assert _REAL_RANK() == {"new": 7}
        stored = json.loads((tmp_path / "rank.json").read_text())
        assert stored["counts"] == {"new": 7} and stored["date"] != "2000-01-01"
        # Another process that day reads the file instead of scanning again.
        pb._rank_cache.clear()
        assert _REAL_RANK() == {"new": 7} and calls == [1]
    finally:
        gate.set()
        pb._rank_cache.clear()
        pb._rank_refresh_started.clear()


def test_a_failed_refresh_keeps_the_previous_rank(tree, monkeypatch, tmp_path):
    import app.paths as paths

    monkeypatch.setattr(paths, "SKILLS_INDEX_RANK_PATH", tmp_path / "rank.json")

    def boom():
        raise OSError("event log unreadable")

    monkeypatch.setattr(pb, "_compute_skill_rank_counts", boom)
    pb._rank_cache.clear()
    try:
        pb._refresh_skill_rank("2099-01-01", {"kept": 2})
        assert pb._rank_cache == {"2099-01-01": {"kept": 2}}
        assert not (tmp_path / "rank.json").exists()
    finally:
        pb._rank_cache.clear()


def test_chat_scope_discoverable_hides_task_bound_skills_from_chat_only(tree, monkeypatch):
    """`skills.index.chat_scope: discoverable` (2026-11-09): a chat prompt leaves out
    the skills an autonomy task binds by name and those whose `surface:` omits chat,
    and says how many in one tail line; a worker prompt still lists every skill;
    `all` (the default) is today's index exactly."""
    _skill(tree, "chatty", "Use when chatting.")
    _skill(tree, "nightly-job", "Use when the nightly job runs.")
    d = tree / "worker-only"
    d.mkdir()
    (d / "SKILL.md").write_text("---\ndescription: Use when a worker.\nsurface: [worker]\n---\nB\n")
    monkeypatch.setattr(pb, "autonomy_bound_skill_names", lambda: {"nightly-job"})
    _set_index(descriptions=True)

    default = pb._load_skills_index()
    assert "nightly-job" in default and "more skills" not in default

    _set_index(chat_scope="discoverable")
    chat = pb._load_skills_index()
    assert "- chatty — Use when chatting." in chat
    assert "nightly-job" not in chat and "worker-only" not in chat
    assert chat.endswith(pb._skills_index_tail(2))
    assert len(chat) <= 12000

    for platform in ("worker", "autonomy"):
        assert pb._load_skills_index(platform=platform) == default
    names_only = pb._load_skills_index(descriptions=False)
    assert names_only == "Available skills: chatty\n" + pb._skills_index_tail(2)


def test_autonomy_bound_skill_names_reads_task_front_matter(tmp_path, monkeypatch):
    import app.data_root as data_root

    auto = tmp_path / "vault" / "autonomy"
    auto.mkdir(parents=True)
    (auto / "a.md").write_text("---\nskill_name: slug-one\n---\n")
    (auto / "b.md").write_text("---\nskill_path: ~/obsidian/skills/slug-two/SKILL.md\n---\n")
    (auto / "c.md").write_text("---\nskill_name: null\n---\n")
    monkeypatch.setattr(data_root, "vault_root", lambda: tmp_path / "vault")
    assert pb.autonomy_bound_skill_names() == {"slug-one", "slug-two"}


def test_the_eval_offers_both_arms_and_scores_the_right_skill():
    import run_prefetch_cost_eval as ev

    assert {ev.ARM_DESC_PUSH, ev.ARM_DESC_PULL, ev.ARM_SCOPED, ev.ARM_NO_INDEX} <= set(ev.ARMS)
    assert ev.SYSTEM_VARIANTS[ev.ARM_SCOPED]["chat_scope"] == "discoverable"
    assert ev.SYSTEM_VARIANTS[ev.ARM_NO_INDEX]["index"] is False
    assert ev.ARM_DESC_PUSH not in ev.DEFAULT_ARMS   # a bare run is still #562's
    assert ev.SYSTEM_VARIANTS[ev.ARM_DESC_PULL] == {"descriptions": True, "push": False}
    rendered = prefetch._format_context([(9.0, {"name": "alpha", "raw": "a"})], [],
                                        show_skill_hint=False) + "\n\nq"
    assert "<skill" not in ev.arm_prompt(rendered, "q", ev.ARM_DESC_PULL)
    assert ev.arm_prompt(rendered, "q", ev.ARM_DESC_PUSH) == rendered
    assert ev.right_skill_reached(rendered, [], ["alpha"]) is True
    assert ev.right_skill_reached("q", ["alpha"], ["alpha"]) is True
    assert ev.right_skill_reached("q", ["beta"], ["alpha"]) is False
    assert ev.right_skill_reached("q", [], []) is None


def test_skill_lint_description_bucket(tree):
    """skill_lint's DESCRIPTION bucket measures what the index can say, with the
    index's own rule and clip: missing is a finding, clipped is advisory."""
    import skill_lint as sl

    _skill(tree, "bare", None)
    _skill(tree, "long", "Use when long. " + "z" * 150)
    _skill(tree, "fine", "Use when fine.")
    result = sl.lint()
    rows = {d["name"]: d for d in result["description"]}
    assert set(rows) == {"bare", "long"}
    assert rows["bare"]["reason"] == "missing" and rows["bare"]["index_text"] == ""
    assert rows["long"]["reason"] == "clipped" and len(rows["long"]["index_text"]) == 100
    report = sl.render_report(result)
    assert "| DESCRIPTION (" in report and "## DESCRIPTION" in report


def test_eval_summary_counts_what_pull_lost_against_push():
    import run_prefetch_cost_eval as ev

    def rec(qid, push, pull):
        return {"id": qid, "arms": {
            ev.ARM_DESC_PUSH: {"right_skill_reached": push, "prefill_tokens": 100},
            ev.ARM_DESC_PULL: {"right_skill_reached": pull, "prefill_tokens": 80}}}

    s = ev.summarize([rec("a", True, False), rec("b", True, True), rec("c", False, True),
                      rec("d", True, False)])
    assert s["pull_lost_vs_push"] == 2 and s["pull_gained_vs_push"] == 1
    assert s["arms"][ev.ARM_DESC_PUSH]["right_skill_reached_rate"] == 0.75
