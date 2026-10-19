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
#: The three that did NOT move are unchanged from 2026-10-04, which is checkable: the
#: paths were last touched by `49e5c410` (powerpoint, deep-research, 2026-09-24) and
#: `c0e889d4` (entity-resolution-sweep, 2026-09-26), both before the triage that recorded
#: them.
SAMPLE_BASELINE = {"powerpoint": 92, "deep-research": 96,
                   "system-health-check": 95, "entity-resolution-sweep": 91}


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
    would have passed this file unnoticed. `system-health-check` is at 95 on 2026-10-19 —
    legal, with 5 lines of room, which is the state this node is content to call green and
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
