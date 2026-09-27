"""Backlog #1225 (members #1201, #1223) — the GitHub body a vault entry carries.

The intel pipeline wrote the raw upstream body into `knowledge/tools/<repo>/*.md`,
and it failed two ways on the one field: `github_scanner` cut it at `[:500]` with no
word or block boundary (`isaaclab/releases.md` carried `> [!NO`, a callout marker cut
in half), and `vault_writer` pasted whatever came back, so an unfilled PR template
was published verbatim — 42 times in `isaaclab/prs.md`. `intel_pipeline/body.py` is
now the one contract between the two: `clip_body` in the scanner, `clean_body` in the
writer. One test per acceptance clause, in order. Nothing here touches the real
vault: the in-process tests rebind the package's paths, and the last one runs the
`python -m intel_pipeline --write` subprocess under a scratch HOME.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INTEL_DIR = REPO_ROOT / "scripts" / "intel-pipeline"
if str(INTEL_DIR) not in sys.path:
    sys.path.insert(0, str(INTEL_DIR))

from intel_pipeline import body as body_mod  # noqa: E402
from intel_pipeline import state as state_mod  # noqa: E402
from intel_pipeline import vault_writer as vw_mod  # noqa: E402
from intel_pipeline.models import ScoredItem  # noqa: E402
from intel_pipeline.scanners import github_scanner as gh_mod  # noqa: E402
from intel_pipeline.scanners import youtube_scanner as yt_mod  # noqa: E402

LIMIT = body_mod.SUMMARY_LIMIT
ELL = body_mod.ELLIPSIS
PHRASE = "Thank you for your interest in sending a pull request"

# The isaaclab PR template exactly as upstream ships it, left unfilled.
UNFILLED_TEMPLATE = (
    "# Description\n\n<!--\n" + PHRASE + ". Please make sure to check the "
    "contribution guidelines.\n\nLink: https://isaac-sim.github.io/IsaacLab/main/"
    "source/refs/contributing.html\n\n💡 Please try to keep PRs small and focused. "
    "Large PRs are harder to review and merge.\n-->\n\nFixes # (issue)\n\n"
    "## Type of change\n\n- [ ] Bug fix\n- [ ] New feature\n\n## Checklist\n\n"
    "- [ ] I have read the contributing guidelines\n"
)


def _words(n: int) -> str:
    """`n` characters of prose built from distinct words, no trailing space."""
    out = []
    i = 0
    while len(" ".join(out)) < n:
        out.append(f"word{i:03d}")
        i += 1
    return " ".join(out)[:n].rstrip()


def _assert_word_boundary_cut(original: str, clipped: str) -> None:
    assert clipped.endswith(ELL), clipped[-40:]
    kept = clipped[: -len(ELL)]
    assert len(kept) <= LIMIT
    assert original.startswith(kept)
    # The character after the kept text is whitespace: nothing was cut mid-word.
    assert original[len(kept)].isspace(), repr(original[len(kept) - 10: len(kept) + 10])


# ── clause 1 — cut at the last whitespace before the limit, with an ellipsis ──

def test_over_limit_body_is_cut_on_a_word_boundary_with_an_ellipsis():
    text = _words(LIMIT + 300)
    # Put the limit squarely inside a word, which is what `[:500]` did.
    assert not text[LIMIT - 1].isspace() and not text[LIMIT].isspace()

    clipped = body_mod.clip_body(text)

    _assert_word_boundary_cut(text, clipped)
    assert body_mod.clip_body("short body stays whole") == "short body stays whole"


# ── clause 2 — no half-emitted callout, heading or checkbox ──────────────────

@pytest.mark.parametrize("marker_line", [
    "> [!NOTE] The callout that used to render as `> [!NO` in releases.md",
    "## Breaking changes in the scheduler and the KV connector API surface",
    "- [ ] <!-- backport-active-release --> Backport this pull request to the branch",
])
def test_a_cut_never_leaves_a_partial_marker_line(marker_line):
    lead = _words(LIMIT - 20)
    text = lead + "\n" + marker_line + "\n" + _words(200)
    # The marker line straddles the limit.
    assert len(lead) < LIMIT < len(lead) + 1 + len(marker_line)

    clipped = body_mod.clip_body(text)

    assert clipped.endswith(ELL)
    lines = clipped[: -len(ELL)].rstrip().splitlines()
    assert not lines[-1].lstrip().startswith((">", "#", "- [")), lines[-1]
    assert "[!NO" not in clipped
    assert clipped[: -len(ELL)].rstrip() == lead


# ── clause 3 — release body, commit message and PR body share the helper ─────

def test_all_three_scanner_shapes_route_through_the_one_helper(tmp_path, monkeypatch):
    long_body = _words(LIMIT + 400)
    monkeypatch.setattr(state_mod, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(state_mod, "STATE_FILE", tmp_path / "scanner-state.json")
    monkeypatch.setattr(gh_mod, "load_github_repos_config",
                        lambda: [{"owner": "o", "repo": "r"}])
    monkeypatch.setattr(gh_mod, "load_github_token", lambda *a, **k: "t")
    monkeypatch.setattr(gh_mod, "fetch_releases", lambda *a: [
        {"tag_name": "v1", "name": "v1", "body": long_body, "html_url": "u1"}])
    monkeypatch.setattr(gh_mod, "fetch_commits", lambda *a: [
        {"sha": "abcdef0123", "html_url": "u2",
         "commit": {"message": "Subject line\n\n" + long_body}}])
    monkeypatch.setattr(gh_mod, "fetch_issues", lambda *a: [
        {"number": 7, "title": "A PR", "body": long_body, "html_url": "u3",
         "pull_request": {"url": "u3"}}])

    items = gh_mod.scan_github_repos()

    by_tag = {it.source_tags[0]: it.summary for it in items}
    assert set(by_tag) == {"release", "commit", "pr"}
    for shape, summary in by_tag.items():
        original = long_body if shape != "commit" else "Subject line\n\n" + long_body
        _assert_word_boundary_cut(original, summary)


# ── clauses 4 and 5 — the writer's side ──────────────────────────────────────

def _item(title: str, summary: str, item_id: str = "gh1") -> ScoredItem:
    return ScoredItem(
        id=f"github:o/r:issue:{item_id}", source="github", title=title,
        url=f"https://github.com/o/r/pull/{item_id}", summary=summary,
        discovered_at=datetime.now(timezone.utc).isoformat(), authors=[],
        source_tags=["pr"], relevance=6, urgency="morning", why="matches vllm",
        projects=[], category="ai-llms")


@pytest.mark.parametrize("summary, reason_word", [
    (UNFILLED_TEMPLATE, "template"),
    ("<!--\nPlease include a summary of the change.\n-->\n", "template"),
    ("No description", "no description"),
    ("# Description\n\n## Motivation\n", "headings"),
    ("fix typo", "under"),
])
def test_template_or_sub_floor_body_is_written_as_none_with_a_reason(
        summary, reason_word):
    rendered = vw_mod._entry_body(_item("Add a scheduler knob", summary))

    assert rendered.startswith("None — "), rendered
    assert reason_word in rendered
    assert "\n" not in rendered
    assert PHRASE not in rendered and "Please include a summary" not in rendered


def test_a_filled_body_under_a_template_comment_is_kept_without_the_comment():
    filled = (UNFILLED_TEMPLATE.split("Fixes #")[0]
              + "Extend the visualization markers supported in the Kit visualizer "
                "to the Newton visualizers.\n")
    rendered = vw_mod._entry_body(_item("Extend markers", filled))
    assert "Newton visualizers" in rendered
    assert PHRASE not in rendered and "<!--" not in rendered


def test_a_body_that_is_or_begins_with_the_title_is_omitted():
    title = "Fix the KV connector leak on abort"
    assert vw_mod._entry_body(_item(title, title)) == ""
    assert vw_mod._entry_body(_item(title, title + "\n\nSigned-off-by: a")) == ""
    # What a commit says beyond its subject line is kept, without the subject.
    rest = "The connector held a block reference after the request was aborted."
    kept = vw_mod._entry_body(_item(title, f"{title}\n\n{rest}"))
    assert kept == rest


def test_an_omitted_body_leaves_no_blank_body_line(tmp_path, monkeypatch):
    vault = tmp_path / "obsidian"
    (vault / "knowledge").mkdir(parents=True)
    monkeypatch.setattr(vw_mod, "KNOWLEDGE_DIR", vault / "knowledge")
    monkeypatch.setattr(vw_mod, "VAULT_ROOT", vault)
    monkeypatch.setattr(vw_mod, "SCORED_FEEDS_DIR", tmp_path / "feeds")
    title = "Fix the KV connector leak on abort"
    item = _item(title, title)
    target = vault / "knowledge" / "ai-llms" / "notes.md"
    monkeypatch.setattr(vw_mod, "determine_vault_path", lambda i, p: target)
    monkeypatch.setattr(vw_mod, "_digest_target", lambda i, p: p)

    assert vw_mod.write_item_to_vault(item, {}) is True

    text = target.read_text()
    assert f"**Relevance:** 6/10\n\n[Link]({item.url})" in text
    assert text.count(title) == 1


# ── clause 6 — a HOME-redirected write appends zero template phrases ─────────

def test_writing_template_bodied_items_to_a_redirected_vault_adds_no_template_phrase(
        tmp_path):
    home = tmp_path / "home"
    feeds = home / "lloyd-data" / "_pipeline" / "vault-derived" / "memory" / "feeds"
    (feeds / "raw").mkdir(parents=True)
    vault = home / "obsidian"
    (vault / "knowledge").mkdir(parents=True)
    (vault / "interests.md").write_text(
        "---\ntitle: Interests\n---\n\n## AI & LLMs\ninference, vllm\n", encoding="utf-8")
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    records = [
        _item(f"vllm scheduler change {n}", UNFILLED_TEMPLATE, item_id=str(n)).to_dict()
        for n in range(3)
    ]
    (feeds / f"intel-{day}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "intel_pipeline", "--write", "--date", day],
        cwd=str(INTEL_DIR),
        env=dict(os.environ, HOME=str(home), LLOYD_DATA=str(home / "lloyd-data")),
        capture_output=True, text=True, timeout=180)

    assert proc.returncode == 0, proc.stderr[-2000:]
    written = [p for p in (vault / "knowledge").rglob("*.md")]
    assert written, proc.stdout[-2000:]
    text = "".join(p.read_text(encoding="utf-8") for p in written)
    for n in range(3):
        assert f"vllm scheduler change {n}" in text, proc.stdout[-2000:]
    assert text.count(PHRASE) == 0
    assert text.count("None — the upstream body is an unfilled PR template") == 3


# ── #1509 — a body that is only a git trailer is not written ─────────────────

TRAILER_TITLE = "refactor(llm-core): consolidate tool argument validation tests (#158254)"
TRAILER_BODY = TRAILER_TITLE + "\n\nCo-authored-by: Vincent Koc <vincentkoc@ieee.org>"


def test_the_incident_body_renders_as_a_trailer_only():
    """The 2026-09-25 openclaw entry: the trailer alone clears clean_body's floor."""
    rendered = vw_mod._entry_body(_item(TRAILER_TITLE, TRAILER_BODY))
    assert rendered == "Co-authored-by: Vincent Koc <vincentkoc@ieee.org>"
    assert vw_mod.is_trailer_only(rendered)
    assert vw_mod.lacks_body(_item(TRAILER_TITLE, TRAILER_BODY))


def test_prose_with_a_trailer_is_a_body():
    assert vw_mod.is_trailer_only("Co-Authored-By: A <a@x>\nSigned-off-by: B <b@x>")
    assert not vw_mod.is_trailer_only(
        "Moves the validation tests under llm-core.\n\nCo-authored-by: A <a@x>")
    assert not vw_mod.is_trailer_only("")
    rest = "The connector held a block reference after the request was aborted."
    assert not vw_mod.lacks_body(
        _item(TRAILER_TITLE, f"{TRAILER_TITLE}\n\n{rest}\n\nCo-authored-by: A <a@x>"))


def _redirect_vault(tmp_path, monkeypatch):
    vault = tmp_path / "obsidian"
    (vault / "knowledge").mkdir(parents=True)
    feeds = tmp_path / "feeds"
    feeds.mkdir()
    monkeypatch.setattr(vw_mod, "KNOWLEDGE_DIR", vault / "knowledge")
    monkeypatch.setattr(vw_mod, "VAULT_ROOT", vault)
    monkeypatch.setattr(vw_mod, "SCORED_FEEDS_DIR", feeds)
    monkeypatch.setattr(vw_mod, "VAULT_WRITTEN_STATE", tmp_path / "written.json")
    monkeypatch.setattr(vw_mod, "load_profile", lambda: {})
    monkeypatch.setattr(vw_mod, "state_loss_detected", lambda *a, **k: False)
    target = vault / "knowledge" / "tools" / "openclaw" / "updates.md"
    monkeypatch.setattr(vw_mod, "determine_vault_path", lambda i, p: target)
    return feeds, target


def test_the_direct_route_refuses_a_trailer_only_body(tmp_path, monkeypatch):
    _feeds, target = _redirect_vault(tmp_path, monkeypatch)
    assert vw_mod.write_item_to_vault(_item(TRAILER_TITLE, TRAILER_BODY), {}) is False
    assert not target.exists()


def test_the_batch_counts_skipped_no_body(tmp_path, monkeypatch, capsys):
    feeds, target = _redirect_vault(tmp_path, monkeypatch)
    day = "2026-09-25"
    good = _item("Add a scheduler knob",
                 "Adds a knob that bounds how many requests the scheduler admits "
                 "per step.", item_id="2")
    rows = [_item(TRAILER_TITLE, TRAILER_BODY, item_id="1").to_dict(), good.to_dict()]
    (feeds / f"intel-{day}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    assert vw_mod.write_all_to_vault(day) == 1
    out = capsys.readouterr().out
    assert "skipped (no body): 1" in out
    text = target.read_text(encoding="utf-8")
    assert "Add a scheduler knob" in text
    assert TRAILER_TITLE not in text and "Co-authored-by" not in text
    # Not marked written: the drop is a refusal, not a publication.
    assert json.loads((tmp_path / "written.json").read_text())["written"] == [good.id]


# ── #1561: the YouTube branch published the feed description verbatim ─────────
#
# The four GitHub clauses above all held, and the YouTube branch was never in this
# file: `_entry_body` returned `summary` as-is for any non-GitHub source, so the
# channel's own promotional footer — a `____` rule, `My Links 🔗`, an arrow and a
# Twitter handle — reached `knowledge/` as knowledge prose, and the `[:500]` slice
# upstream meant the prose that survived was cut wherever 500 bytes fell.
#
# The tests below drive the three surfaces the fix crosses: the renderer
# (`_entry_body`), the scanner that builds the summary off the wire, and the
# published file. The one-helper contract they extend is #1225's, which until now
# covered only release/commit/PR.

#: `youtube:UCqcbQf6yw5KzRoDDcZ_wBSw:V3KeMw2nIDA`'s `summary` field, copied out of
#: `lloyd-data/_pipeline/vault-derived/memory/feeds/raw/2026-09-26.jsonl`: 500
#: characters, ending in the channel's `____` rule, its `My Links` line and its
#: Twitter link. Both defects are in this one string — the upstream description is
#: longer than 500 chars, so this is what the slice left, and the footer is what
#: the last 100 characters of it are.
WESROTH_DESCRIPTION = (
    "OpenAI’s alarm went off. The automatic shutdown didn’t. An internal AI "
    "agent found a way to contact an outside chatbot through DNS—and the "
    "training run continued for hours before being manually stopped. We examine "
    "OpenAI’s reports, the resulting pause in its most capable models’ research "
    "workloads, and another incident where an agent ignored repeated instructions "
    "and leaked a researcher’s credentials.\n\n"
    "______________________________________________\n"
    "My Links 🔗\n"
    "➡️ Twitter: https://x.com/WesRothMon")

#: The same channel's footer, alone: the body #1509's counter waved through because
#: `is_trailer_only` matches git trailers, and a `____` rule plus two link lines is
#: not a git trailer. This is the row the run logged as a write and not as
#: `skipped (no body)`.
FOOTER_ONLY = ("______________________________________________\n"
               "My Links 🔗\n"
               "➡️ Twitter: https://x.com/WesRothMon\n"
               "➡️ Instagram: https://instagram.com/wesroth")

_RULE_AT_LINE_START = re.compile(r"^\s*[_=]{5,}\s*$", re.MULTILINE)


def _yt(id: str = "youtube:UCtest:vid1", *, summary: str = "", why: str = "",
        title: str = "OpenAI paused all training runs... ALIGNMENT FAILURE",
        relevance: int = 8) -> ScoredItem:
    """A scored YouTube row, the shape `--score` hands the writer for a video whose
    channel has no monitor note — the branch #1269 clause 2 routes to `summary`."""
    return ScoredItem(id=id, source="youtube", title=title,
                      url=f"https://www.youtube.com/watch?v={id.rsplit(':', 1)[-1]}",
                      summary=summary, discovered_at="2026-09-27T06:00:00+00:00",
                      relevance=relevance, why=why, category="ai-llms")


@pytest.fixture
def intel_state(tmp_path, monkeypatch):
    """Move the writer's and the scanner's state into `tmp_path`.

    `_paths` resolves at import time and each module imported the result by name, so
    rebinding the per-module names is what actually moves them — otherwise driving
    the scanner would append a day of fake rows to the live feed raw file.
    """
    feeds = tmp_path / "feeds"
    (feeds / "raw").mkdir(parents=True)
    vault = tmp_path / "vault"
    (vault / "knowledge").mkdir(parents=True)
    monkeypatch.setattr(state_mod, "RAW_DIR", feeds / "raw")
    monkeypatch.setattr(state_mod, "STATE_FILE", feeds / "scanner-state.json")
    monkeypatch.setattr(vw_mod, "SCORED_FEEDS_DIR", feeds)
    monkeypatch.setattr(vw_mod, "VAULT_WRITTEN_STATE", feeds / "vault-written.json")
    monkeypatch.setattr(vw_mod, "KNOWLEDGE_DIR", vault / "knowledge")
    monkeypatch.setattr(vw_mod, "VAULT_ROOT", vault)
    monkeypatch.setenv("INTEL_DISABLE_LLM", "1")
    return tmp_path


#: A profile whose one topic matches on `agent`, so a YouTube item is routed to
#: `knowledge/ai-llms/youtube-digest.md` — the file the item names as the surface. With
#: no matching topic the writer files the same item under
#: `knowledge/feeds/youtube-uncategorized.md`, which is a real destination but not the
#: one under test.
DIGEST_PROFILE = {"topics": [{"name": "ai-llms", "weight": 0.9, "keywords": ["agent"]}]}

#: Where `DIGEST_PROFILE` sends a YouTube item, under the redirected vault.
DIGEST_FILE = "knowledge/ai-llms/youtube-digest.md"


def _publish(tmp_path, *items, profile: dict | None = None) -> str:
    """Write `items` through `write_item_to_vault` and return everything that landed
    under the redirected `knowledge/` — the file is the artefact the item names, so
    the assertion runs on it and not only on the rendered string."""
    profile = DIGEST_PROFILE if profile is None else profile
    for item in items:
        vw_mod.write_item_to_vault(item, profile)
    return "\n".join(p.read_text() for p in
                     sorted((tmp_path / "vault" / "knowledge").rglob("*.md")))


def test_recorded_youtube_description_reaches_the_note_without_its_link_footer(intel_state):
    """Clause 1 (#1561): a feed description whose tail is a separator rule and the
    link lines under it renders without that block, on the renderer and in the
    published file. The recorded WesRoth description has to come back ending at its
    own last sentence, with no rule, no `My Links` and no `Twitter:`."""
    assert len(WESROTH_DESCRIPTION) == 500, "the fixture stopped being the recorded row"
    rendered = vw_mod._entry_body(_yt(summary=WESROTH_DESCRIPTION))
    assert rendered.endswith("leaked a researcher’s credentials.")
    assert "My Links" not in rendered and "Twitter:" not in rendered
    assert not _RULE_AT_LINE_START.search(rendered)
    # What survives is the channel's own prose, not a head-of-file stub: the strip
    # has to remove the block and keep everything above it.
    assert rendered.startswith("OpenAI’s alarm went off.")
    assert len(rendered) > 400

    # The two shapes a `____` can be, and the one thing that tells them apart. Under a
    # paragraph or heading with no blank between, `______` is markdown's setext underline
    # and removing it would re-render the surviving line as a heading; set off by a
    # blank, it is the divider over a channel's link block. Both are in a real corpus,
    # so the branch has to be decided by the blank and not left to whichever shape the
    # sample happened to carry.
    setext = "Prose about the run.\n####Subhead\n______"
    assert body_mod.strip_link_footer(setext) == setext, "a setext underline was stripped"
    promo = "Prose about the run.\n\n______\nMy Links 🔗\n➡️ Twitter: https://x.com/a"
    assert body_mod.strip_link_footer(promo) == "Prose about the run."

    written = _publish(intel_state, _yt(summary=WESROTH_DESCRIPTION))
    digest = intel_state / "vault" / DIGEST_FILE
    assert digest.is_file(), f"the item never reached {DIGEST_FILE}: {sorted(written)[:60]}"
    body = digest.read_text()
    assert "My Links" not in body and "Twitter:" not in body
    assert not _RULE_AT_LINE_START.search(body)
    assert "leaked a researcher’s credentials." in body


def test_youtube_summary_at_the_char_cap_is_cut_on_a_word_boundary_and_marked(intel_state,
                                                                             monkeypatch):
    """Clause 2 (#1561): the YouTube summary passes through `body.clip_body` — the
    same helper `_words` bodies go through above, and asserted with the same helper,
    `_assert_word_boundary_cut` — so a description longer than the cap reaches the
    note cut at a word boundary and marked with the package's ellipsis. Never inside
    a token, which is what the raw `[:500]` slice produced: the byte at the cap here
    is inside `word08`, so the old code's summary ended mid-word and unmarked."""
    description = " ".join(f"word{i:02d}" for i in range(400))     # 3,099 chars, no footer
    # The fixture has to be one the OLD code failed: the char at the cap is inside a
    # token, so `[:500]` ended the summary mid-word and said nothing about it.
    assert not description[LIMIT - 1].isspace() and not description[LIMIT].isspace()
    atom = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns:media="http://search.yahoo.com/mrss/"
      xmlns:yt="http://www.youtube.com/xml/schemas/2015"
      xmlns="http://www.w3.org/2005/Atom">
 <entry>
  <id>yt:video:LONG1111</id>
  <yt:videoId>LONG1111</yt:videoId>
  <yt:channelId>UClong</yt:channelId>
  <title>a channel with no monitor note</title>
  <link rel="alternate" href="https://www.youtube.com/watch?v=LONG1111"/>
  <media:description>{description}</media:description>
 </entry>
</feed>"""
    monkeypatch.setattr(yt_mod, "load_youtube_channels_config", lambda: [
        {"handle": "@wesroth", "name": "Wes Roth", "channel_id": "UClong"}])
    monkeypatch.setattr(yt_mod, "_http_get", lambda url, headers=None, timeout=None: atom)

    items, coverage = yt_mod.scan_youtube_channels()

    assert coverage.fetched == 1 and len(items) == 1
    summary = items[0].summary
    # The release/commit/PR shapes above assert this cut with `_assert_word_boundary_cut`
    # (ellipsis, kept text a prefix, width inside the cap, and the character after the
    # cut whitespace); the YouTube summary now answers to the same check.
    _assert_word_boundary_cut(description, summary)
    kept = summary[:-len(body_mod.ELLIPSIS)]
    assert len(kept) > 450, f"clip width {len(kept)} is not the cap, it is a stub"
    # The boundary moved BACK to the previous word, not forward past the next one:
    # at this fixture's width a byte slice keeps 500 characters and a word-boundary
    # clip keeps the whole word before the cap, which is a shorter string.
    assert len(kept) < LIMIT
    # No footer is fabricated out of a description that has none: the prose is the
    # feed's, whole up to the cut.
    assert "My Links" not in summary and "_" not in summary


def test_a_youtube_description_under_the_cap_is_still_passed_through_untouched(intel_state):
    """The other half of #1269 clause 2, which the strip and the clip must not
    disturb: a description with no footer and room under the cap reaches the entry as
    the channel wrote it, uncut, unmarked and not replaced by the scorer's line.

    The second case below is the one that keeps this honest: a footer-less summary
    that does NOT end in punctuation — what an older row cut mid-sentence by the
    `[:500]` slice looks like — is still published as the summary, because the
    emptiness ruling in `_entry_body` judges what THE STRIP removed, not whether the
    writer happens to like the sentence. Prefer `why` over that and this file would be
    quietly rewriting #1269's clause."""
    description = ("Two-week cadence on the humanoid stack, with the gait weights "
                   "checked in and the sim-to-real gap measured on the bench.")
    assert len(description) < LIMIT
    item = _yt(id="youtube:UCtest:vid3", summary=description, why="Scores 8/10: robotics")
    assert vw_mod._entry_body(item) == description
    assert description in _publish(intel_state, item)

    cut_mid_sentence = ("The gait weights were checked in and the sim-to-real gap was "
                        "measured on the bench before they started talking abo")
    assert not cut_mid_sentence.rstrip().endswith((".", "!", "?", "…"))
    older = _yt(id="youtube:UCtest:vid4", summary=cut_mid_sentence,
                why="Scores 8/10: robotics")
    assert vw_mod._entry_body(older) == cut_mid_sentence


def test_footer_only_youtube_summary_falls_through_to_the_scored_why(intel_state):
    """Clause 3 (#1561): when stripping the footer leaves nothing a sentence ends,
    the writer treats the summary as empty and the entry carries `ScoredItem.why`,
    so a body that is nothing but a channel's link block is never published. #1509's
    counter cannot catch this shape — `is_trailer_only` matches git trailers."""
    assert vw_mod.is_trailer_only(FOOTER_ONLY) is False, (
        "the positive control broke: if a footer were trailer-only, #1509 would "
        "already have refused this body and this clause would be testing nothing")

    item = _yt(summary=FOOTER_ONLY, why="Scores 8/10: covers the agent-safety keyword set")
    rendered = vw_mod._entry_body(item)
    assert rendered == "Scores 8/10: covers the agent-safety keyword set"
    assert "My Links" not in rendered and not _RULE_AT_LINE_START.search(rendered)

    written = _publish(intel_state, item)
    assert "Scores 8/10: covers the agent-safety keyword set" in written
    assert "My Links" not in written and "Twitter:" not in written
    assert "Instagram" not in written

    # And a body that does end a sentence is kept, footer or no footer: the
    # emptiness test is about the strip's remainder, not a licence to drop every
    # YouTube summary in favour of the scoring note.
    kept = vw_mod._entry_body(_yt(id="youtube:UCtest:vid2", summary=WESROTH_DESCRIPTION,
                                  why="Scores 8/10: covers the agent-safety keyword set"))
    assert kept.endswith("leaked a researcher’s credentials.")


def test_github_template_and_trailer_only_bodies_keep_their_rulings(intel_state, capsys):
    """Clause 4 (#1561): the GitHub branch is untouched by the YouTube fix. An
    unfilled PR template still renders `None — <reason>` rather than the raw
    headings, and a commit whose body is only a `Co-authored-by:` trailer is still
    refused by the `skipped (no body)` counter instead of reaching the note."""
    assert vw_mod._entry_body(_item("Add a feature", UNFILLED_TEMPLATE, "t1")).startswith("None — ")

    day = "2026-09-27"
    trailer_commit = ScoredItem(
        id="github:openclaw:commit:t1", source="github", title="Bump the pin",
        url="https://github.com/openclaw/openclaw/commit/t1",
        summary="Co-authored-by: someone <someone@users.noreply.github.com>",
        discovered_at="2026-09-27T06:00:00+00:00", source_tags=["commit"],
        relevance=9, category="tools")
    (intel_state / "feeds" / f"intel-{day}.jsonl").write_text(
        json.dumps(trailer_commit.to_dict()) + "\n", encoding="utf-8")

    assert vw_mod.write_all_to_vault(day) == 0
    out = capsys.readouterr().out
    assert "skipped (no body): 1" in out
    assert not list((intel_state / "vault" / "knowledge").rglob("*.md")), \
        "a trailer-only commit body reached the note"
