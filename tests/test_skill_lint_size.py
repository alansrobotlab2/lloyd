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
    assert seen == [("sess-a", "skill.embedded",
                     {"route": "autonomy_task", "skill": "deep-research",
                      "embedded_chars": 15534, "source_chars": 15534}, "run_1")]


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
#: the other four are the control — this round moves nothing in them, so their figures
#: are pinned to the values triage recorded on 2026-10-04, and a drift in one is a
#: drift this round caused.
CAP_HEADROOM = {"nightly-reflection-knowledge-write": 90}
SAMPLE_BASELINE = {"powerpoint": 92, "deep-research": 96,
                   "system-health-check": 84, "entity-resolution-sweep": 91}


@pytest.mark.parametrize("name,expected", sorted(SAMPLE_BASELINE.items()))
def test_the_other_sampled_skills_are_untouched(name, expected):
    root = Path.home() / "obsidian" / "skills"
    path = root / name / "SKILL.md"
    if not path.is_file():
        pytest.skip(f"{path} not on this machine")
    _, body = sl.parse_frontmatter(path.read_text(encoding="utf-8"))[:2]
    got = len(body.strip("\n").splitlines())
    assert got == expected, (
        f"{name}: {got} body lines, was {expected} — #2188's scope is one skill")


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
