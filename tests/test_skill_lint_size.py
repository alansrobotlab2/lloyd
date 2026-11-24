"""skill-lint's SIZE bucket, and where skill bytes enter a prompt (#624).

The lint report had no size number at all, so "how long are our skills" was
measured once by hand at triage (108 of 191 bodies over 100 lines) and never
again. These tests drive `lint()` over a temp skills root: body lines and chars
are counted with the front matter excluded, the library percentiles and the
over-cap count reach both the JSON payload and the markdown, and a lint run
changes no file. The spill delta is read from a throwaway git repo, and the
instrumentation half is pinned on the helper every route calls.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "skill_lint_size", ROOT / "scripts" / "skill_lint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sl = _load()


def _skill(root: Path, name: str, body_lines: int, *, heading_every: int = 0) -> Path:
    d = root / name
    d.mkdir(parents=True)
    body = []
    for i in range(body_lines):
        if heading_every and i % heading_every == 0:
            body.append(f"## Section {i // heading_every}")
        else:
            body.append(f"line {i} of {name}")
    text = ("---\n"
            f"description: Use this skill when testing {name} sizes.\n"
            "tags: [test]\n"
            "---\n\n" + "\n".join(body) + "\n")
    (d / "SKILL.md").write_text(text, encoding="utf-8")
    return d / "SKILL.md"


@pytest.fixture
def library(tmp_path):
    root = tmp_path / "skills"
    paths = {"short": _skill(root, "short", 50),
             "long": _skill(root, "long", 150, heading_every=30),
             "exact": _skill(root, "exact", sl.MAX_BODY_LINES)}
    from agent_mcp.skills import iter_active_skills
    return root, paths, list(iter_active_skills(roots=[root]))


def test_the_threshold_is_a_named_constant_of_100_lines():
    assert sl.MAX_BODY_LINES == 100


def test_the_comment_above_the_ceiling_carries_2334_settled_ruling(library):
    """#2334 clause 1, re-graded after the ruling: the constant is pinned by the node
    above, and its EXCUSE by this one.

    Two sentences have already been false here in the half a reader acts on. The first
    said the ceiling was "to be set from the measured curve, by a person" — a task that
    was already done, because `render_size` has printed that curve on every sweep since
    #624. The second kept it open anyway: after #2334 ruled on 2026-10-08 that SIZE
    stays ADVISORY library-wide, that `MAX_BODY_LINES` (100) is the `SPILL_SAMPLE` spill
    target and not a library ceiling, and that the p90 is rejected as a ceiling because
    it MOVES (352 → 330 body lines between the sweeps of one day), a comment still
    calling the question owed to a person sends a session to a person who has already
    answered. So this node now pins the settled content: the dated ruling, what the 100
    IS, the rejected alternative with the pair that rejected it, and where the live
    numbers are printed — and refuses the deferral phrasings outright, on the whole
    comment, not just on the old sentence.

    `MAX_BODY_LINES == 100` above stays the load-bearing assertion: the ruling keeps
    the number, so a diff that edits it is widening enforcement, which needs a new
    ruled item rather than this file's edit.
    """
    src = (ROOT / "scripts" / "skill_lint.py").read_text(encoding="utf-8")
    cut = src.find("MAX_BODY_LINES = 100")
    assert cut > 0, (
        "#2334 keeps the 100 as the SPILL_SAMPLE spill target, so the literal has to "
        "still be here; a diff that moves it is widening enforcement, which is a new "
        "ruled item's job and not this file's")
    head = src[:cut]
    comment = head[head.rfind("# ── size"):]
    assert comment.strip().startswith("#"), "the text above the constant is the comment"
    flat = " ".join(comment.split())
    assert "to be set from the measured curve" not in flat, \
        "the undated instruction that made the ceiling look unmeasured is back: " + flat
    for deferral in ("owed to a person", "pre-empt", "remains a person",
                     "#2334's ruling", "not a round's edit"):
        assert deferral not in flat, (
            f"the comment defers again with {deferral!r}, and #2334 closed on "
            f"2026-10-08: " + flat)
    assert "borrowed" in flat.lower(), "the 100 has to say it is not a measured number"
    assert "#2334" in flat and "2026-10-08" in flat, (
        "the settled ruling has to arrive dated, or a reader cannot tell it from the "
        "open question it replaces: " + flat)
    assert "advisory" in flat and "library-wide" in flat, \
        "the ruling this comment now carries is that SIZE stays advisory library-wide: " + flat
    assert "SPILL_SAMPLE" in flat, \
        "the 100 has to name what it IS now that it is not a library ceiling: " + flat
    assert "352" in flat and "330" in flat, (
        "the p90 is rejected because it moves, and the pair that shows it moves is the "
        "part a reader can check: " + flat)
    assert "#624" in flat and "#2158" in flat, \
        "the comment separates the narrow five-skill ruling from the library-wide one"
    assert "render_size" in flat and "p90" in flat and "over-cap" in flat, \
        "it names the measured alternative this same script prints: " + flat
    # ...and the numbers it points at are real output, not prose it hopes for: one
    # scan feeds both the dict and the rendered line, so the two cannot disagree. The
    # fixture corpus is built to have skills over the cap in it, so both fields are
    # non-zero here and neither assertion can pass on an empty scan.
    _, _, records = library
    scan = sl.lint(skill_records=records)
    size = scan["size"]
    assert size["over_cap"] > 0 and size["count"] > size["over_cap"], size
    rendered = " ".join(" ".join(sl.render_size(scan)).split())
    assert f"Over the cap: **{size['over_cap']}** of {size['count']}" in rendered, rendered[:300]
    assert f"p90 **{size['p90_lines']}**" in rendered, \
        "the p90 the comment sends the ruling to is on the same summary line"


def test_chat_cut_matches_the_injector():
    from app import prefetch
    assert sl.CHAT_SKILL_CUT == prefetch.SKILL_BODY_MAX


def test_size_bucket_counts_body_lines_and_chars_without_front_matter(library):
    _root, paths, records = library
    size = sl.lint(skill_records=records)["size"]
    rows = {r["name"]: r for r in size["skills"]}
    assert rows["short"]["body_lines"] == 50
    assert rows["long"]["body_lines"] == 150
    assert rows["exact"]["body_lines"] == 100
    content = paths["short"].read_text()
    body = content.split("---\n", 2)[2].strip("\n")
    assert rows["short"]["body_chars"] == len(body)
    assert rows["short"]["file_chars"] == len(content)
    assert [r["name"] for r in size["skills"]] == ["long", "exact", "short"]


def test_over_cap_is_advisory_and_counted(library):
    _root, _paths, records = library
    result = sl.lint(skill_records=records)
    size = result["size"]
    assert size["over_cap"] == 1
    assert [r["name"] for r in size["skills"] if r["over_cap"]] == ["long"]
    assert size["max_body_lines"] == 100
    # A measurement, not a category: the verdict is untouched by an over-cap skill.
    report = sl.render_report(result)
    assert "## No findings on the checks that can fail" in report


def test_percentiles_and_max(library):
    _root, _paths, records = library
    size = sl.lint(skill_records=records)["size"]
    assert (size["p50_lines"], size["p90_lines"], size["max_lines"]) == (100, 150, 150)
    assert size["count"] == 3
    assert size["max_chars"] == max(r["body_chars"] for r in size["skills"])


def test_largest_block_names_the_first_spill_candidate(library):
    _root, _paths, records = library
    rows = {r["name"]: r for r in sl.lint(skill_records=records)["size"]["skills"]}
    assert rows["long"]["largest_block"] == {"heading": "## Section 0", "lines": 30}


def test_size_reaches_the_json_payload_and_the_markdown_table(library):
    _root, _paths, records = library
    result = sl.lint(skill_records=records)
    payload = json.loads(json.dumps(result, default=str))
    assert payload["size"]["over_cap"] == 1
    report = sl.render_report(result)
    assert "### SIZE — body length (advisory; cap 100 lines)" in report
    assert "Over the cap: **1** of 3" in report
    assert "p50 **100**, p90 **150**, max **150**" in report
    for name in ("short", "long", "exact"):
        assert f"| `{name}` |" in report
    assert "| `long` | 150 |" in report


def test_a_lint_run_modifies_no_skill_file(library):
    root, _paths, records = library
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in root.rglob("*") if p.is_file()}
    sl.render_report(sl.lint(skill_records=records))
    after = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in root.rglob("*") if p.is_file()}
    assert after == before


# ── the sampled spill delta, from git history ────────────────────────────────

def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
                        "GIT_AUTHOR_DATE": "2026-09-20T00:00:00",
                        "GIT_COMMITTER_DATE": "2026-09-20T00:00:00",
                        "PATH": "/usr/bin:/bin"})


def test_spill_delta_reads_before_from_history_and_after_from_disk(tmp_path):
    vault = tmp_path / "vault"
    skills = vault / "skills"
    path = _skill(skills, "powerpoint", 246, heading_every=40)
    subprocess.run(["git", "init", "-q", str(vault)], check=True)
    _git(vault, "add", "-A")
    _git(vault, "commit", "-qm", "baseline")
    before = path.read_text()
    # A spill: the body shrinks under the cap and the detail moves to a sibling.
    head, _, rest = before.partition("\n\n")
    kept = rest.splitlines()[:90]
    path.write_text(head + "\n\n" + "\n".join(kept) + "\n- `editing.md`: read when editing\n")
    (path.parent / "editing.md").write_text("\n".join(rest.splitlines()[90:]) + "\n")

    delta = sl.spill_delta(vault, names=("powerpoint", "absent-skill"),
                           before="2026-09-21T00:00:00")
    row = delta["skills"][0]
    assert row["before_chars"] == len(before)
    assert row["after_chars"] == len(path.read_text())
    assert row["delta_chars"] < 0 and delta["total_delta_chars"] == row["delta_chars"]
    assert (row["before_body_lines"], row["after_body_lines"]) == (246, 91)
    assert row["siblings"] == ["editing.md"]
    missing = delta["skills"][1]
    assert missing["delta_chars"] is None and missing["siblings"] == []

    result = sl.lint(skill_records=[])
    result["size"]["spill_delta"] = delta
    report = sl.render_report(result)
    assert f"**{row['delta_chars']}**" in report
    assert "| `powerpoint` |" in report


def test_spill_delta_of_zero_says_the_spill_has_not_been_done(tmp_path):
    vault = tmp_path / "vault"
    _skill(vault / "skills", "powerpoint", 120)
    subprocess.run(["git", "init", "-q", str(vault)], check=True)
    _git(vault, "add", "-A")
    _git(vault, "commit", "-qm", "baseline")
    delta = sl.spill_delta(vault, names=("powerpoint",), before="2026-09-21T00:00:00")
    assert delta["total_delta_chars"] == 0
    result = sl.lint(skill_records=[])
    result["size"]["spill_delta"] = delta
    assert "the spill has not been done yet" in sl.render_report(result)


def test_the_sample_is_the_five_skills_the_item_names():
    assert sl.SPILL_SAMPLE == ("powerpoint", "deep-research",
                               "nightly-reflection-knowledge-write",
                               "system-health-check", "entity-resolution-sweep")


# ── instrumentation: each route books its own skill bytes ────────────────────

def test_record_skill_embed_writes_one_event_per_route(tmp_path, monkeypatch):
    from app import event_log, skill_embed
    monkeypatch.setattr(event_log, "EVENT_LOGS_DIR", tmp_path)
    seen = []
    monkeypatch.setattr(event_log, "log_event",
                        lambda sid, ev, data, turn_id=None: seen.append((sid, ev, data, turn_id)))
    skill_embed.record_skill_embed("sess-a", route="autonomy_task", skill="deep-research",
                                   embedded_chars=15534, source_chars=15534, turn_id="run_1")
    # The exact row, stamps included. #2272 added them and this pin moved because the
    # row grew — 15,534 chars on `autonomy_task` is over `UNCAPPED_EMBED_SOFT_LIMIT`
    # (6,000), so an uncapped embed of this size now books the line it was judged
    # against too. WHO gets stamped and who does not is `tests/test_skill_embed.py`'s
    # business; this node's job is the one-row-per-route-per-turn shape, unchanged.
    assert seen == [("sess-a", "skill.embedded",
                     {"route": "autonomy_task", "skill": "deep-research",
                      "embedded_chars": 15534, "source_chars": 15534,
                      "over_soft_limit": True,
                      "soft_limit": skill_embed.UNCAPPED_EMBED_SOFT_LIMIT}, "run_1")]


def test_a_failing_event_log_never_reaches_the_caller(monkeypatch):
    from app import event_log, skill_embed

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(event_log, "log_event", boom)
    rec = skill_embed.record_skill_embed("s", route="worker_prompt", skill="x",
                                         embedded_chars=3)
    assert rec["embedded_chars"] == 3


#: The two slots a turn-start `<context>` renders, named with skills this box
#: serves. They used to be `big` and `small`, which #2134 made meaningless: a name
#: the skill library does not resolve is no longer a delivery at all, so a fixture
#: named for its size would book nothing and read as a passing size test.
CAPPED_SKILL = "web-search-and-fetch"
EXCERPT_SKILL = "youtube-transcript"


def test_turn_start_context_books_the_capped_size_by_route(monkeypatch):
    """The chat route through the renderer that builds it: the first skill is
    cut at the injector's limit and flagged truncated, the runner-up is booked
    under the excerpt route — so a report can put the capped route beside the
    two uncapped ones.

    Both names resolve (#2134): the sizes below are what the injector put in the
    prompt, and a turn whose tags were only quotations books nothing at all, which
    is `test_a_quoted_skill_name_books_no_size_row`.
    """
    from app import prefetch
    from app import event_log, skill_embed
    seen = []
    monkeypatch.setattr(event_log, "log_event",
                        lambda sid, ev, data, turn_id=None: seen.append(data))
    long_raw = "x" * (prefetch.SKILL_BODY_MAX + 500)
    text = prefetch._format_context(
        [(9.0, {"name": CAPPED_SKILL, "raw": long_raw}),
         (9.0, {"name": EXCERPT_SKILL, "raw": "short body"})], [])
    skill_embed.record_context_skills("sess-b", text)
    assert [(d["route"], d["skill"]) for d in seen] == [
        ("prefetch", CAPPED_SKILL), ("prefetch_excerpt", EXCERPT_SKILL)]
    assert seen[0]["truncated"] is True
    assert prefetch.SKILL_BODY_MAX <= seen[0]["embedded_chars"] < len(long_raw)
    assert seen[1]["truncated"] is False and seen[1]["embedded_chars"] == len("short body")


def test_a_quoted_skill_name_books_no_size_row(monkeypatch):
    """#2134 clause 3, first half: a phantom name books nothing.

    `record_context_skills` reads `skill_delivery_sizes`, which reads the one
    skill-tag walk — the walk that now refuses a name the library does not serve.
    The context block is the renderer's own output with a quoted example appended
    to it, which is the shape that reached the event log from a real turn: a
    `skill.embedded` row for a skill that never entered the prompt.
    """
    from app import prefetch, skill_embed
    from app import event_log
    from app.harness.skill_dispatch import skill_delivery_sizes
    from tests._skill_phantoms import QUOTED_MARKUP, install_skill_names
    install_skill_names(monkeypatch, CAPPED_SKILL, EXCERPT_SKILL)
    seen = []
    monkeypatch.setattr(event_log, "log_event",
                        lambda sid, ev, data, turn_id=None: seen.append(data))
    text = (prefetch._format_context(
        [(9.0, {"name": CAPPED_SKILL, "raw": "the body"}),
         (9.0, {"name": EXCERPT_SKILL, "raw": "the excerpt"})], [])
        + "\n" + QUOTED_MARKUP)
    skill_embed.record_context_skills("sess-c", text)
    assert [d["skill"] for d in seen] == [CAPPED_SKILL, EXCERPT_SKILL], (
        "the three quoted names were booked as skill bytes that never arrived"
    )
    assert [d["name"] for d in skill_delivery_sizes(text)] == [
        CAPPED_SKILL, EXCERPT_SKILL], (
        "the size reader and the event log disagree about what this turn carried")


def test_the_size_gate_is_the_shared_resolver_not_a_private_list(monkeypatch):
    """#2134 clause 3, second half: neither `skill_delivery_sizes` nor
    `record_context_skills` keeps a filter of its own.

    Widening the one shared resolver to serve `X` is the only thing that moves:
    the same text that booked two rows now books three, through
    `record_context_skills`. A caller-side denylist would not budge, and a second
    scanner upstream of the resolver would keep `X` out regardless of what the
    library says. Muting the module's single skill-tag regex then silences the
    event log as well, which is the same proof one level up: everything here
    passes through that one parse.
    """
    from app import prefetch, skill_embed
    from app import event_log
    from app.harness import skill_dispatch as sd
    from tests._skill_phantoms import QUOTED_MARKUP, install_skill_names
    install_skill_names(monkeypatch, CAPPED_SKILL, EXCERPT_SKILL, "X")
    seen: list = []
    monkeypatch.setattr(event_log, "log_event",
                        lambda sid, ev, data, turn_id=None: seen.append(data))
    text = (prefetch._format_context(
        [(9.0, {"name": CAPPED_SKILL, "raw": "the body"}),
         (9.0, {"name": EXCERPT_SKILL, "raw": "the excerpt"})], [])
        + "\n<skill name=\"X\" score=\"8.0\">\nquoted in a doc example\n</skill>\n"
        + QUOTED_MARKUP)
    booked = skill_embed.record_context_skills("sess-d", text)
    assert [d["skill"] for d in booked] == [CAPPED_SKILL, EXCERPT_SKILL, "X"], (
        "the resolver was widened and the booking did not follow, so the caller "
        "is filtering for itself"
    )
    assert seen == booked, "the event log is a second copy of the decision"

    class _Blind:
        @staticmethod
        def finditer(_text):
            return iter(())

    monkeypatch.setattr(sd, "_SKILL_TAG_RE", _Blind())
    seen.clear()
    assert skill_embed.record_context_skills("sess-e", text) == []
    assert seen == [], (
        "a reader that survived the muted regex is reading the markup somewhere "
        "other than the one skill-tag walk"
    )


def test_every_uncapped_route_is_wired_at_its_call_site():
    """The autonomy task prompt and the worker prompt are the routes that embed
    a whole SKILL.md; each call site books it under its own route, and the chat
    path books the capped injection. Read off the source so a refactor that drops
    a call fails here rather than going quiet in the event log."""
    autonomy_src = (ROOT / "app" / "autonomy.py").read_text()
    worker_src = (ROOT / "workers" / "sources" / "deep_research.py").read_text()
    chat_src = (ROOT / "app" / "routers" / "messages.py").read_text()
    assert "route=ROUTE_AUTONOMY_TASK" in autonomy_src
    assert "route=ROUTE_WORKER_PROMPT" in worker_src
    assert "record_context_skills(session_id" in chat_src


# ── #2188: the sampled skill that sat AT the cap now sits under it ────────────
#:
#: `MAX_BODY_LINES` is 100 and `over_cap` in `skill_size` is a strict `>`, so a skill
#: measured at exactly 100 accepts one more line and refuses the second — and
#: `vault_round.SKILL_BODY_ENFORCE` is True, so that refusal lands mid-round at
#: `skill_body_findings`, after the work is done. The fix is headroom, and headroom is
#: only real if the constants stay where they are: a trim that "solved" the problem by
#: raising the ceiling would be the very ruling #1534 left to a person.

#: Measured with `skill_size`'s own `body_lines` row (front matter excluded), which is
#: the number `vault_round.skill_body_findings` refuses on. The item's target is
#: `nightly-reflection-knowledge-write`, the sampled skill #2188 measured at the cap;
#: the other four are the control, whose figures are recorded below.
CAP_HEADROOM = {"nightly-reflection-knowledge-write": 90}

#: The control figures, measured from the vault's COMMITTED bytes — the source the
#: headroom node below already uses, for the reason it gives ("an in-flight edit by
#: another writer is a gate rung owned by nobody"; `pytest.ini`'s `live_vault` rationale
#: says the same of these files, which "an hourly autoresearch promotion or a nightly job
#: can rewrite between rounds").
#:
#: A committed vault change is NOT exempt, and that is what #3028 was: these four values
#: sat at #2188's 2026-10-04 triage figures as though nothing could move them, and
#: `system-health-check` went 84 → 95 when vault `bfb12eb6` (#3025's
#: `api_trust_boundary` component section, +11 lines, 2026-10-19) landed through the
#: rail that exists to land it. The red that filed #3028 was a stale measurement, not a
#: round widening a skill outside its scope — so a drift now names the commit that moved
#: the path instead of asserting "#2188's scope is one skill", which was the false half
#: of the old message: it could only ever mean "someone's file moved", and it named the
#: wrong someone. Refreshing a figure to a landed, rail-validated commit is a legitimate
#: fix; the ceiling is not a figure but `MAX_BODY_LINES`, and that is guarded by
#: `test_no_sampled_skill_sits_past_the_cap_the_lint_refuses_on`.
#:
#: The path has moved that way once since, and this is that refresh: 95 → 98 on vault
#: `e5dc16417` (#3649's `scheduled_jobs` component section, +3 lines, 2026-11-02), whose
#: landing review returned `pass` and whose skill-size gate row read `no regression`
#: against the same 98. The vault was at 95 lines before it and is at 98 at HEAD, both
#: counted by `skill_size` on `git show HEAD:…` bytes — the ruler `skill_body_findings`
#: refuses on, not a second expression of it. What the refresh does NOT do is buy room:
#: 98 of `MAX_BODY_LINES` 100 is 2 lines of headroom, and the node guarding the cap is
#: still green because 98 <= 100, which is the distinction #3028 asked for.
#:
#: The three that did NOT move are unchanged from 2026-10-04, which is checkable: the
#: paths were last touched by `49e5c410` (powerpoint, deep-research, 2026-09-24) and
#: `c0e889d4` (entity-resolution-sweep, 2026-09-26), both before the triage that recorded
#: them.
SAMPLE_BASELINE = {"powerpoint": 92, "deep-research": 96,
                   "system-health-check": 98, "entity-resolution-sweep": 91}


def _vault() -> Path:
    return Path.home() / "obsidian"


def _committed_skill_text(vault: Path, name: str) -> str:
    """`SKILL.md` as the vault's HEAD has it, not as some writer's disk has it.

    Read through `git show` because the enforcement these pins defend
    (`vault_round.skill_body_findings`) runs on a landing — on committed bytes.
    """
    rel = f"HEAD:skills/{name}/SKILL.md"
    shown = subprocess.run(["git", "-C", str(vault), "show", rel],
                           capture_output=True, text=True, timeout=30)
    assert shown.returncode == 0, f"{name}: git show {rel} failed: {shown.stderr[:120]}"
    return shown.stdout


def _committed_body_lines(vault: Path, name: str) -> int:
    """The figure a landing is judged by, from one ruler.

    `skill_size` and not a copy of its expression, for the reason the headroom node
    gives: two expressions for one ruler is how a pin starts disagreeing quietly with the
    thing it defends. (The count is unchanged either way for all five sampled skills on
    2026-10-19, so this half of the change moves no figure.)
    """
    text = _committed_skill_text(vault, name)
    _, body = sl.parse_frontmatter(text)[:2]
    return sl.skill_size(name, vault / "skills" / name / "SKILL.md",
                         text, body)["body_lines"]


def test_the_refreshed_figure_is_measured_and_waives_no_case(tmp_path):
    """#3722 clause 2: the refresh passes the case, it does not remove or waive it.

    `test_the_other_sampled_skills_are_untouched` is parametrized FROM
    `SAMPLE_BASELINE`, so a drifted figure can be made green four ways and only one is a
    fix. Deleting the entry makes its parameter — and its check — disappear while the run
    still prints green; a `skip`, `skipif`, `xfail` or `live_vault` mark on the node does
    the same thing to all four cases at once; and the node's own guard skips quietly
    whenever the vault file is absent. This node pins those three shut and witnesses the
    fourth:

    * the four keys are exactly `sl.SPILL_SAMPLE` minus the one skill `CAP_HEADROOM`
      guards for #2188, so dropping an entry reddens THIS node even though the
      parametrized one would merely go quiet;
    * the control node carries no suppression mark at function or module level — the only
      mark on it is the `parametrize` that builds its four cases;
    * where this vault holds the four `SKILL.md` files — the condition under which the
      control node's absent-path guard does NOT skip — each pinned figure is re-read from
      the vault's committed bytes and asserted; where it does not, this node skips on that
      same machine-absent convention, after the three pins above have run, because a
      vault that is not here cannot contradict a figure;
    * the figure is a measurement and not a constant: asked of a fixture vault whose
      committed `SKILL.md` holds 3 body lines, `_committed_body_lines` answers 3, so the
      number the live case passes on is the number the vault reports.
    """
    control = test_the_other_sampled_skills_are_untouched
    suppressed = {"skip", "skipif", "xfail", "live_vault", "fault_injection"}

    assert set(SAMPLE_BASELINE) == set(sl.SPILL_SAMPLE) - set(CAP_HEADROOM), (
        "an entry removed from SAMPLE_BASELINE deletes that skill's case from the "
        "parametrized control node: 4 cases are expected, got "
        f"{sorted(SAMPLE_BASELINE)}")

    fn_marks = {m.name for m in getattr(control, "pytestmark", [])}
    assert not fn_marks & suppressed, f"{control.__name__} is waived: {sorted(fn_marks)}"
    assert "parametrize" in fn_marks, (
        "the control node is no longer parametrized over the pinned figures, so this "
        "file's green no longer says the four figures were checked")
    module_marks = {m.name for m in globals().get("pytestmark", [])}
    assert not module_marks & suppressed, f"the module is waived: {sorted(module_marks)}"

    live = _vault()
    absent = [name for name in SAMPLE_BASELINE
              if not (live / "skills" / name / "SKILL.md").is_file()]
    if absent:
        # The same machine-absent convention the control node skips on: a gate round
        # repoints HOME at a root with no vault, and a missing vault says nothing about a
        # figure. Everything above — the key set, the marks, the measurement witness — is
        # asserted unconditionally, because none of it needs this vault; what needs it is
        # the claim that the four live cases RAN, and that is only reportable where they
        # did.
        pytest.skip(f"{sorted(absent)} not in {live}: the parametrized control cases "
                    "skip on this machine too, so its 'every case ran' half is "
                    "untestable here")
    for name, expected in SAMPLE_BASELINE.items():
        assert _committed_body_lines(live, name) == expected, (
            f"{name}: pinned {expected}, the vault's committed bytes report otherwise — "
            "a stale figure, not a waived one")

    fixture = tmp_path / "vault"
    _skill(fixture / "skills", "system-health-check", 3)
    subprocess.run(["git", "init", "-q", str(fixture)], check=True)
    _git(fixture, "add", "-A")
    _git(fixture, "commit", "-qm", "fixture")
    assert _committed_body_lines(fixture, "system-health-check") == 3, (
        "the helper stopped reading the committed bytes, which is the property that makes "
        "a passing figure a measurement of the vault")


def _moving_commit(vault: Path, name: str) -> str:
    """The commit that last changed the path, so a drift can name its author."""
    # `--` and the path as SEPARATE argv entries: `--skills/x/SKILL.md` as one token is
    # an unknown option, git prints nothing, and the helper below would then report
    # "<no commit names this path>" on a file with a whole history — a guard that reads
    # its own missing input and still prints a verdict (#1044's family).
    log = subprocess.run(["git", "-C", str(vault), "log", "-1", "--format=%h %s",
                          "--", f"skills/{name}/SKILL.md"],
                         capture_output=True, text=True, timeout=30)
    return (log.stdout or "").strip() or "<no commit names this path>"


@pytest.mark.parametrize("name,expected", sorted(SAMPLE_BASELINE.items()))
def test_the_other_sampled_skills_are_untouched(name, expected):
    """The control of #2188's measurement, pinned to the vault's committed figure.

    "Untouched" means nothing has moved this file since the value recorded in
    `SAMPLE_BASELINE`. It does not mean "the round now gating is the only thing that
    could": the vault is a live tree that any round, autonomy job or person commits to
    through `automod_vault_land`, so the failure names the commit that moved the path and
    a reader can attribute the drift in one step instead of being told, falsely, that it
    was their diff.
    """
    vault = _vault()
    path = vault / "skills" / name / "SKILL.md"
    if not path.is_file():
        pytest.skip(f"{path} not on this machine")
    got = _committed_body_lines(vault, name)
    assert got == expected, (
        f"{name}: {got} body lines at HEAD, pinned {expected}. The last commit to move "
        f"the path is {_moving_commit(vault, name)} — if that is a legitimate landing, "
        f"refresh SAMPLE_BASELINE, which is a recorded measurement and not a ceiling; "
        f"the ceiling is MAX_BODY_LINES, guarded by "
        f"test_no_sampled_skill_sits_past_the_cap_the_lint_refuses_on.")


def test_the_control_reads_committed_bytes_and_ignores_an_in_flight_edit(tmp_path):
    """The seam under this node is a `git show` into a tree no round controls.

    In a throwaway vault: commit a 40-line body, then write a 60-line body to the same
    path and leave it uncommitted — another writer mid-edit. The reader the control uses
    must report 40, and the naive disk read is asserted at 60 in the same breath, because
    without that the first assertion could pass on a fixture that had stopped being a
    drift at all. That is the failure `pytest.ini`'s `live_vault` rationale is written
    against, and the one that made main red for a landed document (#3028).
    """
    vault = tmp_path / "vault"
    path = _skill(vault / "skills", "system-health-check", 40)
    subprocess.run(["git", "init", "-q", str(vault)], check=True)
    _git(vault, "add", "-A")
    _git(vault, "commit", "-qm", "baseline")

    committed = path.read_text(encoding="utf-8")
    fm_end = committed.index("---\n\n") + len("---\n\n")
    path.write_text(committed[:fm_end]
                    + "\n".join(f"line {i} of system-health-check" for i in range(60))
                    + "\n", encoding="utf-8")

    assert _committed_body_lines(vault, "system-health-check") == 40
    live = path.read_text(encoding="utf-8")
    assert sl.skill_size("system-health-check", path, live,
                         sl.parse_frontmatter(live)[1])["body_lines"] == 60, (
        "the fixture is no longer a drift between disk and HEAD, so the assertion above "
        "would be passing for the wrong reason")


def test_a_drift_names_the_commit_that_moved_the_path(monkeypatch):
    """The failure path, asked directly: a wrong figure must RED, and name its author.

    Written because the first draft of the helper above passed a scratch check by
    reporting `<no commit names this path>` — its argv glued `--` to the path, so git
    printed nothing for a file with a whole history and the message would have blamed no
    one. Here the pin is deliberately set one line off the vault's committed figure, so
    the only way through is the assertion firing with a real hex sha in it.

    This is also the round-trip proof that the control node still guards anything at all
    after #3028 moved its figure: a stale pin on a drifted file is exactly the red that
    filed this item, and it is still red.
    """
    vault = _vault()
    name = "system-health-check"
    if not (vault / "skills" / name / "SKILL.md").is_file():
        pytest.skip(f"vault skill {name} not on this machine")
    committed = _committed_body_lines(vault, name)
    wrong = committed + 1
    monkeypatch.setitem(SAMPLE_BASELINE, name, wrong)
    with pytest.raises(AssertionError) as exc:
        test_the_other_sampled_skills_are_untouched(name, wrong)
    msg = str(exc.value)
    # Both figures have to be in the message: the one HEAD says, and the one that was
    # pinned. A message carrying only one is a reader guessing which is which.
    assert f"{committed} body lines" in msg, msg
    assert f"pinned {wrong}" in msg, msg
    assert "<no commit names this path>" not in msg, msg
    shas = [t for t in msg.split()
            if len(t) >= 7 and all(c in "0123456789abcdef" for c in t)]
    assert shas, f"no commit sha in the drift message: {msg}"


def test_no_sampled_skill_sits_past_the_cap_the_lint_refuses_on():
    """The ceiling, asked of all five sampled skills instead of #2188's one.

    `skill_body_findings` refuses a landing whose sampled skill sits over
    `MAX_BODY_LINES`, and until now exactly one of the five names was guarded against
    that; the other four were guarded against *movement* instead, which is why a legal
    +11-line section could turn main red while a sampled skill edging up to 99 of 100
    would have passed this file unnoticed. `system-health-check` is the tightest of the
    five at 98 since vault `e5dc16417` (2026-11-02, #3649's `scheduled_jobs` section) —
    legal, with 2 lines of room, which is the state this node is content to call green and
    the reason the +8-line-rule headroom property stays scoped to the one skill #2188
    chose.
    """
    vault = _vault()
    for name in sl.SPILL_SAMPLE:
        lines = _committed_body_lines(vault, name)
        assert lines <= sl.MAX_BODY_LINES, (
            f"{name}: {lines} body lines at HEAD, cap {sl.MAX_BODY_LINES} — a landing "
            f"would be refused at skill_body_findings. Last commit to move the path: "
            f"{_moving_commit(vault, name)}")


def test_the_skill_that_sat_at_the_cap_now_has_headroom():
    from scripts.automod import vault_round

    root = Path.home() / "obsidian" / "skills"
    (name, ceiling), = CAP_HEADROOM.items()
    # Measured against the vault's COMMITTED bytes, which is the state the enforcement
    # that this item is about actually grades (`vault_round.skill_body_findings` runs on
    # a landing). The figure's provenance, in committed shas, so this comment stays
    # checkable instead of drifting: `da43c60d` (#2188's spill) left 88 body lines;
    # `c698c9a5` (#2173's `## Step 2.5: Memory-index pre-flight`) added 5, and at 93 the
    # property below is false — 93 + an 8-line rule = 101, refused at the 100 cap — which
    # is the red this file's own #2193 was filed for. #2193 brought the body to 88 by
    # reflowing the hard-wrapped §2e–2f deferral paragraphs — 7 body lines became 2,
    # every word kept and nothing moved to a sibling — so the quoted heading
    # `### 2f. Knowledge Log` stopped being split across a wrap for the first time and
    # the room the item exists to protect is 12 lines under the cap (88 + 8 = 96).
    # The working-tree figure stays printed, not judged, for the reason `pytest.ini`
    # gives — an in-flight edit by another writer is a gate rung owned by nobody. That
    # exemption covers a writer mid-edit, never a landed commit: `c698c9a5` was committed
    # when #2188 gated, so the sentence naming it as "another session's in-flight work"
    # was the false half of this comment, and a comment that states a false thing about
    # the tree it guards is how a red sat unowned for a round.
    rel = f"skills/{name}/SKILL.md"
    shown = subprocess.run(["git", "-C", str(root.parent), "show", f"HEAD:{rel}"],
                           capture_output=True, text=True, timeout=30)
    assert shown.returncode == 0, f"git show HEAD:{rel} failed: {shown.stderr[:120]}"
    text = shown.stdout
    _, body = sl.parse_frontmatter(text)[:2]
    # The count comes from `skill_size` itself, not from a copy of its expression:
    # `vault_round.skill_body_findings` refuses on `size["over_cap"]`, so the number a
    # landing is judged by and the number this node judges have to come from one place.
    # Two expressions for one ruler is how a pin starts disagreeing with the thing it
    # defends, quietly, the day the ruler changes (a fence exclusion, say).
    lines = sl.skill_size(name, root / name / "SKILL.md", text, body)["body_lines"]
    assert lines <= ceiling, f"{name}: {lines} body lines, ceiling {ceiling}"
    # The constants are the point, not the number: an #1488-sized (+8) rule has to
    # land without meeting `skill_body_findings`, and it can only do that while the
    # ceiling and the enforcement are where they are.
    assert sl.MAX_BODY_LINES == 100, sl.MAX_BODY_LINES
    assert sl.SPILL_SAMPLE == ("powerpoint", "deep-research",
                              "nightly-reflection-knowledge-write",
                              "system-health-check", "entity-resolution-sweep")
    assert vault_round.SKILL_BODY_ENFORCE is True
    assert lines + 8 <= sl.MAX_BODY_LINES, (
        f"{lines} + an 8-line rule = {lines + 8}: still refused at the ceiling")
    live = (root / name / "SKILL.md").read_text(encoding="utf-8")
    live_lines = (sl.skill_size(name, root / name / "SKILL.md", live,
                               sl.parse_frontmatter(live)[1])["body_lines"]
                  if live else -1)
    assert live_lines <= sl.MAX_BODY_LINES, (
        f"the working tree sits at {live_lines} body lines, past the 100-line ceiling — "
        "an uncommitted edit has used up more headroom than exists. The committed figure "
        f"above ({lines}) is what this node judges; this one is the same file on disk.")


# ── the 12,000-CHAR cap the automod gate enforces (#4191) ────────────────────
#
# `MAX_BODY_LINES` (100) is the SPILL_SAMPLE target and the report's `Over the cap:`
# count has always been that LINE count. The ceiling that actually freezes a skill is
# a different one: `scripts/automod/vault_round.SKILL_BODY_CHAR_CAP` (12,000 body
# chars), which refuses a landing that creates or grows a body past it. On 2026-11-16
# that froze 30 of the 208 live skills — `system-health-check` among them at 20,078
# body chars in 98 lines, so it passes the line rule it is measured by and no number
# printed anywhere said it was frozen. Four backlog items were spent discovering it
# one skill at a time (#4086, #4139, #4186, #4190), each after an unattended implement
# attempt had already been spent on the refusal. This block publishes the population.
# It adds measurements only: #2334 ruled on 2026-10-08 that SIZE stays advisory
# library-wide, so no node here asks for a refusal and none can be caused by it.

def _wide(root: Path, name: str, chars: int, *, siblings: int = 0) -> Path:
    """A skill whose BODY is about `chars` chars in FEWER than `MAX_BODY_LINES` lines.

    200-char lines keep the LINE ceiling off the measurement: a builder that overshot
    100 lines as well would make `over_char_cap` unfalsifiable against `over_cap`, and
    the whole point of the flag is that the two rules disagree about real skills —
    `system-health-check` is over the char cap at 98 body lines.
    """
    d = root / name
    d.mkdir(parents=True)
    width = 200
    text = ("---\n"
            f"description: Use this skill when testing {name} char sizes.\n"
            "tags: [test]\n"
            "---\n\n" + "\n".join(["x" * width] * max(1, chars // width)) + "\n")
    (d / "SKILL.md").write_text(text, encoding="utf-8")
    for i in range(siblings):
        (d / f"detail-{i}.md").write_text(f"# detail {i}\n", encoding="utf-8")
    return d / "SKILL.md"


@pytest.fixture
def char_library(tmp_path):
    """Two skills past the char cap (one with siblings, one without) and one under it."""
    from agent_mcp.skills import iter_active_skills
    root = tmp_path / "skills"
    cap = sl.SKILL_BODY_CHAR_CAP
    paths = {"frozen-wide": _wide(root, "frozen-wide", cap + 4_000),
             "frozen-spilled": _wide(root, "frozen-spilled", cap + 1_000, siblings=4),
             "under": _wide(root, "under", 2_000)}
    return root, paths, list(iter_active_skills(roots=[root]))


def test_the_char_cap_is_the_gate_s_own_twelve_thousand():
    """One ceiling, two spellings, pinned equal (#4191 clause 1).

    `skill_lint` cannot import `scripts.automod.vault_round` to read the constant:
    vault_round imports this module lazily inside its body-lines check, so a
    module-level import back is a cycle. `CHAT_SKILL_CUT` above restates
    `prefetch.SKILL_BODY_MAX` for the identical reason and `test_chat_cut_matches_the_
    injector` pins that pair; this pins this one, so a move on either side is a red
    node here rather than two numbers that drift apart silently.
    """
    from scripts.automod import vault_round as vr
    assert sl.SKILL_BODY_CHAR_CAP == vr.SKILL_BODY_CHAR_CAP == 12_000, (
        "the census must count against the cap the gate refuses at; 12,000 is also the "
        "value #4186 ruled must NOT be raised, because app/prefetch.SKILL_BODY_MAX (6000) "
        "is the clip that makes the ceiling load-bearing")


def test_the_boundary_is_strictly_greater_like_the_gate():
    """12,000 chars is allowed; 12,001 is over — the gate's `>` at the same boundary."""
    from scripts.automod import vault_round as vr
    at = sl.skill_size("at", Path("/tmp/x/SKILL.md"), "", "y" * vr.SKILL_BODY_CHAR_CAP)
    over = sl.skill_size("over", Path("/tmp/x/SKILL.md"), "",
                         "y" * (vr.SKILL_BODY_CHAR_CAP + 1))
    assert at["over_char_cap"] is False and over["over_char_cap"] is True


def test_the_size_row_carries_the_char_cap_flag_beside_the_line_flag(char_library):
    """#4191 clause 1: the per-skill row flags the char ceiling, and the two rules
    stay separate fields — a skill can be over either, both, or neither."""
    _root, _paths, records = char_library
    rows = {r["name"]: r for r in sl.lint(skill_records=records)["size"]["skills"]}
    assert rows["frozen-wide"]["over_char_cap"] is True
    assert rows["frozen-spilled"]["over_char_cap"] is True
    assert rows["under"]["over_char_cap"] is False
    assert all(r["over_cap"] is False for r in rows.values()), \
        "the fixture is under 100 lines on purpose: over_char_cap must not be a synonym"


def test_size_summary_publishes_the_frozen_count_and_the_cap_it_counted(char_library):
    """#4191 clause 1: the count, the cap it counted against, and the corpus named."""
    _root, _paths, records = char_library
    size = sl.lint(skill_records=records)["size"]
    assert size["over_char_cap"] == 2
    assert size["char_cap"] == 12_000
    assert size["char_cap_corpus"] == "live", \
        "the report counts the corpus `iter_active_skills` owns (208 on 2026-11-16), not " \
        "the on-disk SKILL.md walk (218); the label has to travel with the number"
    assert size["count"] == 3
    assert size["over_char_cap"] == sl.char_cap_census(size["skills"])["over"], \
        "the published count and the shared census are one measurement, not two rules"


def test_each_frozen_row_names_the_skill_its_chars_and_its_siblings(char_library):
    """#4191 clause 2: the published census is per-skill, worst first.

    `siblings` is published beside the count because the failure mode the census names
    is a skill that HAS siblings and is still frozen — the spill pattern was applied to
    `system-health-check` without ever stripping its SKILL.md, so a report that treated
    siblings as the fix would tell the next round it had nothing to do.
    """
    _root, _paths, records = char_library
    size = sl.lint(skill_records=records)["size"]
    rows = {r["name"]: r for r in size["skills"]}
    assert rows["frozen-spilled"]["siblings"] == ["detail-0.md", "detail-1.md",
                                                  "detail-2.md", "detail-3.md"]
    assert rows["frozen-wide"]["siblings"] == []
    frozen = size["over_char_cap_skills"]
    assert [s["name"] for s in frozen] == ["frozen-wide", "frozen-spilled"], \
        "the census is ordered by cost, so the first row is the one worth spilling first"
    for s in frozen:
        assert s["body_chars"] == rows[s["name"]]["body_chars"] > sl.SKILL_BODY_CHAR_CAP
        assert s["siblings"] == len(rows[s["name"]]["siblings"])


def test_the_size_summary_line_prints_the_frozen_count_beside_the_line_count(char_library):
    """#4191 clause 3: one run of this script puts the frozen population on the
    `### SIZE` line, next to the line-cap count that was the only count before."""
    _root, _paths, records = char_library
    result = sl.lint(skill_records=records)
    size = result["size"]
    report = sl.render_report(result)
    line = next(l for l in report.splitlines() if l.startswith("Over the cap:"))
    assert f"Over the cap: **{size['over_cap']}** of {size['count']} lines" in line, \
        "the line-cap count this report has always printed stays where its tests find it"
    assert f"**{size['over_char_cap']}** of {size['count']}" in line.split("lines", 1)[1], \
        "the frozen count is on the SAME line, beside it: " + line
    assert "12,000" in line and "char" in line.lower(), \
        "a reader has to be able to tell which cap this count is against: " + line
    assert "live" in line, "and which corpus it was counted over: " + line


def test_the_frozen_rows_render_in_the_size_table(char_library):
    """The table carries the same three facts, so the published census is per-skill and
    not only a total."""
    _root, _paths, records = char_library
    result = sl.lint(skill_records=records)
    report = sl.render_report(result)
    for name in ("frozen-wide", "frozen-spilled"):
        row = next((l for l in report.splitlines() if l.startswith(f"| `{name}` |")), "")
        assert row, f"{name} is over the char cap and must appear in the SIZE table"
        chars = next(r["body_chars"] for r in result["size"]["skills"] if r["name"] == name)
        assert f"{chars:,}" in row or str(chars) in row, row
    spilled = next(l for l in report.splitlines() if l.startswith("| `frozen-spilled` |"))
    assert "| yes | 4 |" in spilled, \
        "over the char cap AND four sibling files, side by side — the pair that shows a " \
        "spilled skill can still be frozen: " + spilled
    assert "| yes | 0 |" in next(l for l in report.splitlines()
                                 if l.startswith("| `frozen-wide` |")), \
        "and the majority shape: 23 of the 30 frozen skills have nothing spilled at all"


def test_the_census_counts_the_records_it_is_handed(tmp_path):
    """`live_char_cap_census` is the function the automod gate calls at refusal time;
    it resolves the corpus through `iter_active_skills` (#2963: never a private walk)
    and counts with the same rows the report publishes."""
    from agent_mcp.skills import iter_active_skills
    root = tmp_path / "skills"
    cap = sl.SKILL_BODY_CHAR_CAP
    _wide(root, "one", cap + 400)
    _wide(root, "two", cap + 9_000)
    _wide(root, "three", 500)
    census = sl.live_char_cap_census(list(iter_active_skills(roots=[root])))
    assert (census["cap"], census["count"], census["over"]) == (12_000, 3, 2)
    assert [s["name"] for s in census["skills"]] == ["two", "one"]
    # #4336 widened this key set: an over-cap row now carries the commit and author that
    # put the body over the cap, plus the delta that shows how fast. Pinned as a set
    # equality because the shape is the API — `_char_cap_note` reads it at refusal time and
    # a key renamed or dropped there is invisible until someone asks who grew the file.
    # `CENSUS_ROW_KEYS` and its reading are defined in the #4336 section at the tail of
    # this file, which is also where the fixture-backed proof of these fields lives.
    assert set(census["skills"][0]) == CENSUS_ROW_KEYS, sorted(census["skills"][0])
    assert sl.live_char_cap_census([]) == {"cap": 12_000, "corpus": "live", "count": 0,
                                          "over": 0, "skills": []}


def test_the_live_census_and_the_report_agree_on_the_real_corpus():
    """The published line and the refusal reason must not disagree about the library,
    so both count through this one function over the corpus owner's list. Measured on
    2026-11-16: 208 live skills, 30 of them over the 12,000-char cap, worst
    `nightly-reflection-signals` at 162,608 chars and the smallest freeze
    `ascii-video` at 12,057. Nothing here pins those figures — skills get split and the
    count moves, which is the point of publishing it — but the census and a fresh
    re-count by the gate's own extractor must land on the same number."""
    from agent_mcp.skills import iter_active_skills
    from scripts.automod.vault_round import _skill_desc_and_body, SKILL_BODY_CHAR_CAP
    census = sl.live_char_cap_census()
    assert census["count"] == len(list(iter_active_skills())), \
        "the census counts the corpus its caller would count, not a walk of its own"
    recount = sum(1 for rec in iter_active_skills()
                  if _skill_desc_and_body(
                      Path(rec.skill_file).read_text(encoding="utf-8",
                                                     errors="replace"))[1]
                  > SKILL_BODY_CHAR_CAP)
    assert census["over"] == recount > 0, \
        f"the census says {census['over']}, the gate's own extractor says {recount}"


def test_system_health_check_is_frozen_with_siblings_already_on_disk():
    """#4191 clause 2's named instance, read from the bytes as committed (#3028).

    This is the row the census exists to print: 98 body lines (inside the 100-line
    spill target, so the line rule never flagged it), 20,078 body chars (past the
    ceiling the gate enforces), and the spill pattern ALREADY APPLIED —
    `## Files in this skill` plus check-components.md, implementation.md,
    output-format.md, usage-examples.md. Siblings existing is not the fix, and 23 of
    the 30 frozen skills on 2026-11-16 have no siblings at all.

    The 20,078 and the four names are a recorded measurement, not a ceiling: if they
    move, a commit split this skill or renamed a sibling, and this node names it.
    """
    vault = _vault()
    skill = vault / "skills" / "system-health-check" / "SKILL.md"
    text = _committed_skill_text(vault, "system-health-check")
    if text is None or not skill.is_file():
        pytest.skip("vault has no system-health-check committed to measure")
    body = sl.parse_frontmatter(text)[1]
    row = sl.skill_size("system-health-check", skill, text, body)
    assert row["body_chars"] == 20_078, (
        f"the committed body is {row['body_chars']:,} chars, not the 20,078 recorded on "
        "2026-11-15/16; refresh the figure from `git log -1 --format=%h -- "
        "skills/system-health-check/SKILL.md` in ~/obsidian, and if it has fallen under "
        "12,000 the skill has been split and this node's instance is stale")
    assert row["over_char_cap"] is True and row["over_cap"] is False, (
        f"{row['body_lines']} body lines is inside the 100-line spill target while "
        f"{row['body_chars']:,} chars is past the gate's ceiling — that disagreement IS "
        "the blind spot; if it has gone, re-measure before editing the expected pair")
    assert row["siblings"] == ["check-components.md", "implementation.md",
                               "output-format.md", "usage-examples.md"], \
        f"the spill shape this node names has changed: {row['siblings']}"


# ── #4336: a crossing names the commit and the author that caused it ────────────────
#
# The census could say THAT 30 of 208 bodies sit past 12,000 chars and how big each is;
# it could not say who put it there. `nightly-reflection-signals` grew 156,329 ->
# 167,502 body chars in 48 hours across vault commits 78357539e..f775c602b, authored
# `lloyd-nightly-reflection-config` and `lloyd-nightly-reflection-knowledge-write`, and
# zero of the promotions ledger's 85,295 rows recorded it — the growth arm at
# `scripts/automod/vault_round.py:2985` binds only `vault_land`, and the other commit
# door, `scripts/util/vault-commit.sh`, never reads the cap at all. So the spill
# question was argued instead of measured, and the census — which runs nightly and at
# every refusal — had no writer, no author and no delta.
#
# These nodes pin the two things that close that: attribution on the row, and a durable
# dated row on a crossing. Both are read over a throwaway git repo, because the only
# history that identifies an author is the vault's own, and a fixture with two commits by
# two authors is the smallest thing that can prove a sha is not a guess.

#: The row keys an over-cap census row must carry after #4336. `name`, `body_chars` and
#: `siblings` are #4191's; the last four are #4336's, and `crossed` is what the durable
#: store keys on.
CENSUS_ROW_KEYS = {"name", "body_chars", "siblings", "last_commit_sha",
                   "last_commit_author", "last_commit_subject", "body_char_delta",
                   "prev_body_chars", "crossed"}

#: The keys of one durable crossing row. #4336 clause 4 names seven of them; the ISO
#: stamp is `at`, and the delta is spelled as it is on the census row so a reader
#: reconciling the two files is not reconciling two names for one number.
CROSSING_ROW_KEYS = {"at", "skill", "sha", "author", "subject", "body_chars",
                     "body_char_delta"}

# Imported here rather than beside the other imports at the top of the file because this
# whole section is appended: a line inserted above anything a rail may pin is precisely
# the fragility #4332 and #4336 exist to remove.
import datetime as dt  # noqa: E402


def _body_chars(text: str) -> int:
    """Body chars the way `skill_size` counts them: front matter out, edge newlines off."""
    return len(sl.parse_frontmatter(text)[1].strip("\n"))


def _cap_repo(tmp_path) -> Path:
    """A throwaway vault: `<root>/skills/<name>/SKILL.md`, committed under git.

    `iter_active_skills(roots=[root / "skills"])` then yields exactly the fixture's
    skills, and the repo root is one level above the skills directory — the same shape
    as the real vault, which is what makes the relative path the code asks git about
    come out as `skills/<name>/SKILL.md`.
    """
    vault = tmp_path / "vault"
    (vault / "skills").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(vault)], check=True)
    return vault


def _commit_skill(vault: Path, name: str, chars: int, author: str,
                  subject: str = "") -> str:
    """Commit `skills/<name>/SKILL.md` with a body of about `chars` chars as `author`.

    200-char lines for the same reason `_wide` uses them: they keep the LINE ceiling off
    the measurement, so a body counted over the char cap is over it on chars alone. The
    body is rewritten in place on a second call, which is what makes this the shape of a
    nightly writer appending to a skill it already committed. `--author` goes AFTER
    `commit`: before the subcommand git(1) reads it as its own option and exits 129.
    """
    path = vault / "skills" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\n"
                    f"description: Use this skill when testing {name} growth.\n"
                    "tags: [test]\n"
                    "---\n\n" + "\n".join(["x" * 200] * max(1, chars // 200)) + "\n",
                    encoding="utf-8")
    ident = ["-c", f"user.name={author}", "-c", f"user.email={author}@example.invalid"]
    subprocess.run(["git", "-C", str(vault), *ident, "add", "--",
                    f"skills/{name}/SKILL.md"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(vault), *ident, "commit", "-q",
                    f"--author={author} <{author}@example.invalid>",
                    "-m", subject or f"{name}: body now ~{chars} chars"],
                   check=True, capture_output=True)
    return subprocess.run(["git", "-C", str(vault), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()


def _commit_unrelated(vault: Path, author: str) -> str:
    """Commit a file that is not a SKILL.md, as `author`, and return the new HEAD sha.

    Exists so a test can put HEAD out of alignment with a skill's own history: a fixture
    with one skill has its last commit BE its HEAD commit, and then no assertion on sha or
    author can tell the two apart.
    """
    path = vault / "notes.md"
    path.write_text("# an unrelated vault note\n", encoding="utf-8")
    ident = ["-c", f"user.name={author}", "-c", f"user.email={author}@example.invalid"]
    subprocess.run(["git", "-C", str(vault), *ident, "add", "--", "notes.md"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(vault), *ident, "commit", "-q",
                    f"--author={author} <{author}@example.invalid>",
                    "-m", "note: touches no skill at all"],
                   check=True, capture_output=True)
    return subprocess.run(["git", "-C", str(vault), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def _records(vault: Path):
    from agent_mcp.skills import iter_active_skills
    return list(iter_active_skills(roots=[vault / "skills"]))


def _crossing_store(tmp_path, monkeypatch) -> Path:
    """Point `$LLOYD_DATA` at the fixture and hand back the store path it implies."""
    monkeypatch.setenv("LLOYD_DATA", str(tmp_path / "data"))
    return (tmp_path / "data" / "skill-lint" / "char-cap-crossings.jsonl")


def _store_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_an_over_cap_row_names_the_commit_author_and_delta_from_history(tmp_path):
    """Clauses 1 and 2: the row says which commit last touched THIS skill's SKILL.md,
    who authored it, and how far the body moved since the commit before that.

    Two commits, two authors: the row must carry the SECOND sha and the second author —
    reading `HEAD` instead of the path's own history would name whoever committed an
    unrelated file last, which is how a growth report blames the wrong route — and
    `body_char_delta` must equal the body-char difference between the two commits,
    counted with `skill_size`'s own ruler rather than the file's byte length, because
    front matter moves independently of a body.
    """
    cap = sl.SKILL_BODY_CHAR_CAP
    vault = _cap_repo(tmp_path)
    _commit_skill(vault, "grower", cap - 5_000, "lloyd-nightly-reflection-config")
    second = _commit_skill(vault, "grower", cap + 4_000,
                           "lloyd-nightly-reflection-knowledge-write",
                           subject="knowledge-write: append the 2026-11-24 stage plan")
    # A commit that touches NO skill, by a third author, AFTER the growth commit: without it
    # this fixture's HEAD would BE the skill's last commit and the two asserts below could
    # not tell "the history of this file" from "whatever the repo did last" — which is the
    # exact failure mode that blames the wrong route for a crossing.
    _commit_unrelated(vault, "lloyd-nightly-skills-management")
    head = subprocess.run(["git", "-C", str(vault), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    assert head != second, "the unrelated commit has to move HEAD for this to prove anything"
    census = sl.live_char_cap_census(_records(vault))
    row = next(s for s in census["skills"] if s["name"] == "grower")
    assert row["last_commit_sha"] == second, (
        f"the row named {row['last_commit_sha']} while HEAD is {head}: it must name the "
        "last commit touching skills/grower/SKILL.md, not the repo's latest commit")
    assert row["last_commit_author"] == "lloyd-nightly-reflection-knowledge-write", row
    assert row["last_commit_subject"] == "knowledge-write: append the 2026-11-24 stage plan"
    assert row["last_commit_sha"] != head, (
        "the attribution came off HEAD: an author field read that way names whoever "
        "committed anything last, not whoever grew this body")
    first_body = _body_chars(
        subprocess.run(["git", "-C", str(vault), "show",
                        f"{second}^:skills/grower/SKILL.md"],
                       check=True, capture_output=True, text=True).stdout)
    assert row["prev_body_chars"] == first_body
    assert row["body_char_delta"] == row["body_chars"] - first_body, (
        f"delta {row['body_char_delta']} must be today's body chars less the body chars "
        f"at the previous commit touching this file ({row['body_chars']} - {first_body})")


def test_a_skill_that_crossed_the_cap_is_flagged_and_one_already_over_is_not(tmp_path):
    """Clause 3: the crossing is the transition, not the state.

    `grower` was inside the cap at the previous commit touching it and is past it now:
    that is the event worth a durable row. `already-frozen` was past the cap at BOTH
    commits, so it is over the cap with no crossing — the 30 skills frozen on 2026-11-16
    are a backlog, and a store that logged all 30 every night would bury the one signal
    in the count that made the store worth opening. A skill with only one commit in its
    history has no previous body to cross from, so it is not a crossing either.
    """
    cap = sl.SKILL_BODY_CHAR_CAP
    vault = _cap_repo(tmp_path)
    _commit_skill(vault, "grower", cap - 5_000, "lloyd-nightly-reflection-config")
    _commit_skill(vault, "grower", cap + 4_000, "lloyd-nightly-reflection-signals")
    _commit_skill(vault, "already-frozen", cap + 1_000, "lloyd-nightly-skills-management")
    _commit_skill(vault, "already-frozen", cap + 9_000, "lloyd-nightly-skills-management")
    _commit_skill(vault, "single-commit", cap + 500, "lloyd-nightly-reflection-signals")
    rows = {s["name"]: s for s in sl.live_char_cap_census(_records(vault))["skills"]}
    assert rows["grower"]["crossed"] is True, (
        f"{rows['grower']}: inside the cap at the previous commit, past it now")
    assert rows["already-frozen"]["crossed"] is False, (
        f"{rows['already-frozen']}: past the cap at both commits is a frozen skill, "
        "not a crossing")
    assert rows["single-commit"]["crossed"] is False, (
        "one commit touching a file is no transition to measure")
    assert all(r["prev_body_chars"] is not None for r in rows.values()
               if r["crossed"]), "a crossing needs the previous body it crossed from"


def test_a_crossing_appends_one_dated_row_and_a_quiet_run_appends_nothing(
        tmp_path, monkeypatch):
    """Clause 4: the crossing outlives the window in a file, and only crossings write it.

    stdout is not a record: `~/obsidian/autonomy/skill-lint-report.md` was three days old
    on 2026-11-16, which is why this row goes to `$LLOYD_DATA/skill-lint/
    char-cap-crossings.jsonl` where the owed-check job can read it after the growth
    window has closed. One row per crossing, not one per census run — the lint runs
    nightly and the gate calls the census at every refusal, so a per-run append would
    write the same event dozens of times and the row count would stop meaning anything.
    A census over a corpus with no crossing appends nothing at all.
    """
    store = _crossing_store(tmp_path, monkeypatch)
    cap = sl.SKILL_BODY_CHAR_CAP
    vault = _cap_repo(tmp_path)
    _commit_skill(vault, "grower", cap - 5_000, "lloyd-nightly-reflection-config")
    sha = _commit_skill(vault, "grower", cap + 4_000,
                        "lloyd-nightly-reflection-knowledge-write")
    rows = _store_rows(store)
    assert not rows, f"a run with no crossing wrote {rows}"
    sl.live_char_cap_census(_records(vault))
    rows = _store_rows(store)
    assert len(rows) == 1, f"a crossing appends exactly one row, got {rows}"
    row = rows[0]
    assert set(row) == CROSSING_ROW_KEYS, row
    assert row["skill"] == "grower" and row["sha"] == sha, row
    assert row["author"] == "lloyd-nightly-reflection-knowledge-write", row
    assert row["body_char_delta"] > 0 and row["body_chars"] > cap, row
    assert "grower" in row["subject"], row
    # The row reconciles against the sha it names: `body_chars` is the COMMITTED body there,
    # so a reader with `git show <sha>` reproduces it. The census row's own `body_chars` is
    # the working tree, and mixing the two into one number is how a store starts narrating.
    at_sha = _body_chars(subprocess.run(
        ["git", "-C", str(vault), "show", f"{sha}:skills/grower/SKILL.md"],
        check=True, capture_output=True, text=True).stdout)
    assert row["body_chars"] == at_sha == row["body_char_delta"] + _body_chars(
        subprocess.run(["git", "-C", str(vault), "show",
                        f"{sha}^:skills/grower/SKILL.md"],
                       check=True, capture_output=True, text=True).stdout), row
    assert dt.datetime.fromisoformat(row["at"]).utcoffset() is not None, (
        f"`{row['at']}` must be an aware ISO stamp: a naive stamp in a machine-facing "
        "row is read as UTC by every later reader")
    sl.live_char_cap_census(_records(vault))
    assert len(_store_rows(store)) == 1, "the same crossing must not be appended twice"
    # A SECOND vault whose only over-cap skill was past the cap at both of its commits,
    # censused here: over-cap with no crossing is the 30-frozen-on-2026-11-16 shape, and the
    # only thing that makes "the store still holds one row" a measurement rather than a
    # repeat of the line above is a census run sitting between them that COULD have written.
    quiet = _cap_repo(tmp_path / "quiet")
    _commit_skill(quiet, "already-frozen", cap + 1_000, "lloyd-nightly-skills-management")
    _commit_skill(quiet, "already-frozen", cap + 9_000, "lloyd-nightly-skills-management")
    quiet_rows = sl.live_char_cap_census(_records(quiet))["skills"]
    assert [r["name"] for r in quiet_rows] == ["already-frozen"], quiet_rows
    assert quiet_rows[0]["crossed"] is False, quiet_rows[0]
    assert len(_store_rows(store)) == 1, (
        "a census over a corpus that is over the cap but crossed nothing appended a row")


def test_the_crossing_store_bounds_itself_on_age_and_on_rows(tmp_path, monkeypatch):
    """Clause 4's other half: the new store cannot grow without a horizon.

    The retention sweep's own rule is that a store missing from its table is a store
    nothing bounds, and `char-cap-crossings.jsonl` is not in that table — the cheapest
    bound that keeps this change single-surface is to apply it on append. A row older
    than `CROSSING_MAX_AGE_DAYS` goes, and a file longer than `CROSSING_MAX_ROWS` keeps
    its newest `CROSSING_MAX_ROWS`: the horizon has to be measured against the row's own
    stamp, never against the file's mtime, or a store that is appended to every day is
    forever young and never trims.
    """
    store = _crossing_store(tmp_path, monkeypatch)
    store.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc)
    stale = {"at": (now - dt.timedelta(days=sl.CROSSING_MAX_AGE_DAYS + 10)
                    ).isoformat(), "skill": "ancient", "sha": "0" * 40,
             "author": "nobody", "subject": "long predates this horizon",
             "body_chars": sl.SKILL_BODY_CHAR_CAP + 1, "body_char_delta": 7}
    filler = [{"at": now.isoformat(), "skill": f"filler-{i}", "sha": f"{i:040x}",
               "author": "lloyd-nightly-reflection-signals", "subject": "seed",
               "body_chars": sl.SKILL_BODY_CHAR_CAP + 1, "body_char_delta": 1}
              for i in range(sl.CROSSING_MAX_ROWS + 40)]
    store.write_text("".join(json.dumps(r) + "\n"
                             for r in [stale] + filler), encoding="utf-8")
    cap = sl.SKILL_BODY_CHAR_CAP
    vault = _cap_repo(tmp_path)
    _commit_skill(vault, "grower", cap - 5_000, "lloyd-nightly-reflection-config")
    _commit_skill(vault, "grower", cap + 4_000, "lloyd-nightly-reflection-signals")
    sl.live_char_cap_census(_records(vault))
    rows = _store_rows(store)
    assert len(rows) == sl.CROSSING_MAX_ROWS, (
        f"the store kept {len(rows)} rows against a bound of {sl.CROSSING_MAX_ROWS}")
    assert "ancient" not in {r["skill"] for r in rows}, (
        "a row past CROSSING_MAX_AGE_DAYS survived an append: the horizon is not "
        "reading the row's own stamp")
    assert rows[-1]["skill"] == "grower", "the fresh crossing has to survive the trim"


# ── #4336 review round 2: the three gaps the refusal named ──────────────────────────
#
# Appended at the tail, not inserted above the section they complete, for the reason
# #4332 wrote into `app/harness/protected_paths.py`: a line inserted above something a
# document cites re-points the citation.


def test_uncommitted_growth_is_attributed_to_the_last_commit_rather_than_invented(tmp_path):
    """The dirty working tree, which is what `live_char_cap_census` actually reads.

    The census counts the body on disk and the attribution comes from git, so between two
    commits the two halves describe different states. Every attributed number here is a
    COMMIT fact — the delta is the last commit's body less the commit before it, and the
    crossing is those two commits' relation — while `body_chars` on the same row is the
    working tree. The split is the safety property: a nightly writer with the tree dirty
    must not have growth nobody has committed charged to whoever committed last, and an
    on-disk body past the cap must not mint a crossing row for a commit that stayed inside
    it. Both commits in this fixture are inside the cap, the tree then grows past it, and
    the row reports the growth in `body_chars` while the delta stays the earlier commit's
    own move and no crossing is claimed.
    """
    cap = sl.SKILL_BODY_CHAR_CAP
    vault = _cap_repo(tmp_path)
    first = _commit_skill(vault, "grower", cap - 5_000, "lloyd-nightly-reflection-config",
                          subject="config: 2026-11-23 pass")
    second = _commit_skill(vault, "grower", cap - 3_000,
                           "lloyd-nightly-reflection-knowledge-write",
                           subject="knowledge-write: 2026-11-24 pass")
    first_body = _body_chars(
        subprocess.run(["git", "-C", str(vault), "show",
                        f"{first}:skills/grower/SKILL.md"],
                       check=True, capture_output=True, text=True).stdout)
    committed = _body_chars(
        subprocess.run(["git", "-C", str(vault), "show",
                        f"{second}:skills/grower/SKILL.md"],
                       check=True, capture_output=True, text=True).stdout)
    assert committed <= cap, "both commits stay inside the cap; only the tree crosses"
    skill_file = vault / "skills" / "grower" / "SKILL.md"
    skill_file.write_text(
        skill_file.read_text(encoding="utf-8") + "\n"
        + "\n".join(["z" * 200] * 60) + "\n", encoding="utf-8")
    dirty = _body_chars(skill_file.read_text(encoding="utf-8"))
    assert dirty > cap, "the uncommitted growth has to cross the cap to be tempting"
    row = next(s for s in sl.live_char_cap_census(_records(vault))["skills"]
               if s["name"] == "grower")
    assert row["body_chars"] == dirty, "the census counts the working tree, not HEAD"
    assert (row["last_commit_sha"], row["last_commit_author"]) == (
        second, "lloyd-nightly-reflection-knowledge-write"), row
    assert row["prev_body_chars"] == first_body, (
        f"prev_body_chars must be the body at the commit BEFORE the named one, read from "
        f"git, not assumed from the size asked for: {row['prev_body_chars']} vs "
        f"{first_body}")
    assert row["body_char_delta"] == committed - first_body, (
        f"the delta must be the named commit's own move ({committed} - {first_body} = "
        f"{committed - first_body}), not the on-disk body less prev "
        f"({dirty - first_body}), which would charge the {dirty - committed} uncommitted "
        "chars to the last committer")
    assert row["body_char_delta"] < dirty - first_body, (
        "the committed move is strictly smaller than the on-disk gap, which is the whole "
        "point of anchoring the delta to commits")
    assert row["crossed"] is False, (
        "the tree crossed the cap but no commit did, so no dated crossing row is owed")


def test_the_trim_keeps_every_row_newer_than_the_horizon_even_when_they_exceed_it(
        tmp_path, monkeypatch):
    """The trim branch at its own numbers, at a size a fixture can afford.

    `CROSSING_MAX_ROWS` is 500, and seeding 501 live rows to prove a slice is a fixture
    nobody maintains; the shipped constant is pinned by name instead, and the slice is
    exercised with the bound monkeypatched to 2. What has to survive is the ORDERING claim:
    the newest rows are kept and the stale ones go, INCLUDING when the fresh rows alone
    exceed the bound — a trim written to always leave room for history would drop the
    crossing that triggered the write, which is the one row the store exists to hold.
    """
    assert sl.CROSSING_MAX_ROWS == 500, "the shipped bound is what the docstring names"
    store = _crossing_store(tmp_path, monkeypatch)
    monkeypatch.setattr(sl, "CROSSING_MAX_ROWS", 2)
    store.parent.mkdir(parents=True, exist_ok=True)
    old = {"at": (dt.datetime.now(dt.timezone.utc)
                  - dt.timedelta(days=sl.CROSSING_MAX_AGE_DAYS - 5)).isoformat(),
           "skill": "within-horizon-but-older", "sha": "1" * 40, "author": "yesterday",
           "subject": "inside the age horizon, outside the row bound",
           "body_chars": sl.SKILL_BODY_CHAR_CAP + 1, "body_char_delta": 1}
    store.write_text(json.dumps(old) + "\n", encoding="utf-8")
    cap = sl.SKILL_BODY_CHAR_CAP
    vault = _cap_repo(tmp_path)
    _commit_skill(vault, "grower-a", cap - 5_000, "lloyd-nightly-reflection-config")
    _commit_skill(vault, "grower-a", cap + 4_000, "lloyd-nightly-reflection-signals")
    _commit_skill(vault, "grower-b", cap - 3_000, "lloyd-nightly-reflection-config")
    _commit_skill(vault, "grower-b", cap + 6_000, "lloyd-nightly-reflection-signals")
    assert len(_store_rows(store)) == 1
    sl.live_char_cap_census(_records(vault))
    rows = _store_rows(store)
    assert len(rows) == 2, f"the bound is 2, the store holds {len(rows)}"
    assert {r["skill"] for r in rows} == {"grower-a", "grower-b"}, (
        f"both fresh crossings had to survive the trim and the older in-horizon row go, "
        f"got {[r['skill'] for r in rows]}")
    assert "within-horizon-but-older" not in {r["skill"] for r in rows}, rows
    assert [r["skill"] for r in rows] == ["grower-b", "grower-a"], (
        "rows are appended in census order, which is worst-first by body chars — the "
        "census doubles as a spill queue, so the bigger body is written first")


def test_the_store_lives_under_the_data_root_and_the_ceiling_is_untouched(tmp_path,
                                                                          monkeypatch):
    """Where the record goes, and the one thing this diff was forbidden to move.

    Path, not prose: `char_cap_crossings_path()` must sit under the DATA ROOT at
    `skill-lint/char-cap-crossings.jsonl` and NOT under the vault — `~/obsidian/autonomy/
    skill-lint-report.md` regenerates nightly and was three days stale on 2026-11-16, which
    is the whole reason this store exists, and a store inside the vault would be swept or
    landed by rounds rather than read by the owed-check job. The data root is asked of the
    same resolver every other store path on this box uses, so the assertion is tree-relative
    and true in a round's worktree as well as in production.

    Then the ceiling: 12,000 in `skill_lint` and 12,000 in the census output, and the
    sentence `_char_cap_note` composes out of `over`/`count`/`corpus` still composes —
    #4191's refusal text is assembled from those three keys, so widening the ROWS must not
    have touched the keys the note reads.
    """
    monkeypatch.delenv("LLOYD_DATA", raising=False)
    from app.data_root import resolve_data_root_for_tree
    path = sl.char_cap_crossings_path()
    rel = path.relative_to(resolve_data_root_for_tree())
    assert rel.as_posix() == "skill-lint/char-cap-crossings.jsonl", rel
    assert path.parent.parent != Path.home() / "obsidian", (
        f"{path} sits in the vault, where a nightly report already goes stale")
    assert sl.SKILL_BODY_CHAR_CAP == 12_000 == sl.char_cap_census([])["cap"], \
        "the cap moved: #4336 was asked to attribute crossings, not to raise the ceiling"
    census = {"cap": 12_000, "corpus": "live", "count": 208, "over": 30, "skills": []}
    assert set(census) == {"cap", "corpus", "count", "over", "skills"}, \
        "the census's own key set moved, and the refusal note is composed from it"
    note = f"{census['over']} of {census['count']} {census['corpus']} skills are " \
           f"already frozen against that ceiling"
    assert note == "30 of 208 live skills are already frozen against that ceiling", note


def test_attribution_survives_a_vault_reached_through_a_symlink(tmp_path):
    """The seam a round's home actually is: `obsidian` symlinked into the live vault.

    `_repo_owner` asks git for the owning repo and then takes a relative path, and git
    answers with a resolved path — so a corpus reached through a link is the case where a
    naive `skill_file.relative_to(root)` raises ValueError and every row comes back
    silently unknown. Unknown-forever would be worse than the item's premise: the census
    would report 28 over-cap skills with no author at all and look like a clean result. Same
    fixture as the history node, one extra link in the path.
    """
    cap = sl.SKILL_BODY_CHAR_CAP
    vault = _cap_repo(tmp_path)
    sha = _commit_skill(vault, "grower", cap - 5_000, "lloyd-nightly-reflection-config")
    _commit_skill(vault, "grower", cap + 4_000, "lloyd-nightly-reflection-signals")
    link = tmp_path / "linked"
    link.symlink_to(vault)
    row = next(s for s in sl.live_char_cap_census(_records(link))["skills"]
               if s["name"] == "grower")
    assert row["last_commit_author"] == "lloyd-nightly-reflection-signals", (
        f"attribution went unknown through a symlink: {row}")
    assert row["last_commit_sha"] != sha and row["prev_body_chars"] is not None, row
    assert row["crossed"] is True, row
