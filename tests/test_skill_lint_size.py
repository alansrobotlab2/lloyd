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
    import prefetch
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


def test_turn_start_context_books_the_capped_size_by_route(monkeypatch):
    """The chat route through the renderer that builds it: the first skill is
    cut at the injector's limit and flagged truncated, the runner-up is booked
    under the excerpt route — so a report can put the capped route beside the
    two uncapped ones."""
    import prefetch
    from app import event_log, skill_embed
    seen = []
    monkeypatch.setattr(event_log, "log_event",
                        lambda sid, ev, data, turn_id=None: seen.append(data))
    long_raw = "x" * (prefetch.SKILL_BODY_MAX + 500)
    text = prefetch._format_context(
        [(9.0, {"name": "big", "raw": long_raw}),
         (9.0, {"name": "small", "raw": "short body"})], [])
    skill_embed.record_context_skills("sess-b", text)
    assert [(d["route"], d["skill"]) for d in seen] == [
        ("prefetch", "big"), ("prefetch_excerpt", "small")]
    assert seen[0]["truncated"] is True
    assert prefetch.SKILL_BODY_MAX <= seen[0]["embedded_chars"] < len(long_raw)
    assert seen[1]["truncated"] is False and seen[1]["embedded_chars"] == len("short body")


def test_every_uncapped_route_is_wired_at_its_call_site():
    """The autonomy task prompt and the worker prompt are the routes that embed
    a whole SKILL.md; each call site books it under its own route, and the chat
    path books the capped injection. Read off the source so a refactor that drops
    a call fails here rather than going quiet in the event log."""
    autonomy_src = (ROOT / "autonomy.py").read_text()
    worker_src = (ROOT / "workers" / "sources" / "deep_research.py").read_text()
    chat_src = (ROOT / "app" / "routers" / "messages.py").read_text()
    assert "route=ROUTE_AUTONOMY_TASK" in autonomy_src
    assert "route=ROUTE_WORKER_PROMPT" in worker_src
    assert "record_context_skills(session_id" in chat_src
