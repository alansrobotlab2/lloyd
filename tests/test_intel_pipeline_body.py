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


#: The two footer-less descriptions #1269 clause 2 rules on, bound here so #1819
#: clause 4 can pin the same two strings from the strip's side as well as the
#: writer's. Names carry, values are the strings that were inline below.
NO_FOOTER_DESCRIPTION = ("Two-week cadence on the humanoid stack, with the gait weights "
                         "checked in and the sim-to-real gap measured on the bench.")

#: An older row the pre-#1561 `[:500]` slice left ending mid-sentence, with no
#: terminal punctuation at all — #1269 clause 2 publishes it as the summary.
NO_FOOTER_CUT_MID_SENTENCE = ("The gait weights were checked in and the sim-to-real gap "
                              "was measured on the bench before they started talking abo")


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
    description = NO_FOOTER_DESCRIPTION
    assert len(description) < LIMIT
    item = _yt(id="youtube:UCtest:vid3", summary=description, why="Scores 8/10: robotics")
    assert vw_mod._entry_body(item) == description
    assert description in _publish(intel_state, item)

    cut_mid_sentence = NO_FOOTER_CUT_MID_SENTENCE
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


# ── #1819: a promotional footer with NO separator rule above it ──────────────
#
# #1561 anchored its strip on a separator rule and never looked past it:
# `strip_link_footer` scans backwards for `_RULE_LINE_RE`, and AI Revolution publishes
# no rule line at all — a blank line, two emoji contact lines, a blank, then a
# `What You'll See:` heading and its `0:00 —` chapter list. With no candidate rule the
# function returns its input unchanged, `_entry_body`'s `stripped == summary` branch
# (vault_writer.py:427-428) hands the whole description back, and three sections of
# `knowledge/ai-llms/youtube-digest.md` — `## 2026-09-27`, `## 2026-09-28`,
# `## 2026-09-29` — carry the channel's contact block and chapter list as knowledge
# prose. The strip has to be anchored on the footer's own labels as well as on a rule.

#: The news paragraph of the 2026-09-29 row `The First Real RSI is Here and It's
#: Evolving Fast`, and the two halves of the footer under it — split out so a clause
#: can name the half it is asserting on. Copied out of
#: `lloyd-data/_pipeline/vault-derived/memory/feeds/raw/2026-09-29.jsonl`.
RSI_NEWS = (
    "Weco’s AIDE² just showed what it calls the first real evidence of recursive "
    "self-improvement, redesigning itself in eight days and beating the human-built "
    "version. Meanwhile OpenAI’s Astra hits Critical cyber capability, Claude Sonnet "
    "5.5 launches, and Gemini 4 nears release.")

RSI_CONTACT = ("📩 Brand Deals & Partnerships: collabs@nouralabs.com\n"
               "✉️ General Inquiries: airevolutionofficial@gmail.com")

#: The chapter list, ending where the scanner's 500-character clip ran out — inside
#: the list, which is the half of the defect the footer strip removes at source.
RSI_CHAPTERS = ("What You'll See:\n"
                "0:00 — Intro\n"
                "0:41 — How Weco’s AIDE² redesigned itself in eight days and beat its "
                "human-built…")

#: The stored `summary` field of that row, verbatim: 495 characters, no rule line.
RSI_RULE_FREE_ROW = f"{RSI_NEWS}\n\n{RSI_CONTACT}\n\n{RSI_CHAPTERS}"

#: The same footer with nothing above it: the body that was only a channel's contact
#: block, which the strip empties so the writer's footer-only ruling reaches it.
RSI_RULE_FREE_FOOTER_ONLY = f"{RSI_CONTACT}\n\n{RSI_CHAPTERS}"

#: What the feed actually sent for that video: the same footer over the same news,
#: with the chapter list uncut. The stored row above is this clipped to 500 chars.
RSI_RAW_DESCRIPTION = (
    f"{RSI_NEWS}\n\n{RSI_CONTACT}\n\n"
    "What You'll See:\n"
    "0:00 — Intro\n"
    "0:41 — How Weco’s AIDE² redesigned itself in eight days and beat its human-built "
    "version.\n"
    "2:31 — Why OpenAI’s Astra hitting Critical cyber capability matters.\n")

#: What a reader must never see again in a digest section: the strings the grep that
#: closed #1561, and the one chapter marker, all four in the recorded footer.
RSI_FOOTER_MARKS = ("Brand Deals", "collabs@", "gmail.com", "What You'll See:", "0:00 —")


def test_a_promo_footer_with_no_separator_rule_is_stripped_by_shape():
    """Clause 1 (#1819): the recorded rule-free shape — news paragraph, blank, the two
    contact lines, blank, `What You'll See:` and its `0:00 —` chapter lines, no
    `____`/`====` anywhere — comes back as the news paragraph and exactly that.
    #1561's rule anchor is blind to it, which is how the run published it whole."""
    assert len(RSI_RULE_FREE_ROW) == 495, "the fixture stopped being the recorded row"
    # The positive control that this really is the rule-FREE shape: #1561's anchor does
    # not occur in the fixture, so whatever this strip removes was removed by the label
    # anchor and not by a rule the old code could already have seen.
    assert not _RULE_AT_LINE_START.search(RSI_RULE_FREE_ROW), \
        "the fixture grew a separator rule and is no longer the shape under test"

    stripped = body_mod.strip_link_footer(RSI_RULE_FREE_ROW)
    assert stripped == RSI_NEWS, repr(stripped)
    for mark in RSI_FOOTER_MARKS:
        assert mark not in stripped, mark

    # And a body that WAS nothing but the rule-free footer strips to nothing, which is
    # what lets #1561's existing footer-only ruling cover the new anchor class too.
    assert body_mod.strip_link_footer(RSI_RULE_FREE_FOOTER_ONLY) == ""


#: `DIGEST_PROFILE` matches on `agent`, which this row's title and summary do not say:
#: routing scores `title + " " + summary` only (`determine_vault_path`,
#: vault_writer.py:245), never `why`. So the profile here matches on the row's own
#: phrase, which routes it to the same `DIGEST_FILE` the real run used.
RSI_DIGEST_PROFILE = {"topics": [{"name": "ai-llms", "weight": 0.9,
                                  "keywords": ["self-improvement"]}]}


def test_the_recorded_rule_free_row_lands_in_the_digest_as_its_news_paragraph(intel_state):
    """Clause 2 (#1819): that same scored YouTube item written through
    `write_item_to_vault` publishes the news paragraph as the entry's body, and the
    published digest carries no contact line and no chapter line anywhere — the grep
    that closed #1561 has to come back empty on a section written after this fix."""
    why = "Scores 8/10: first claimed evidence of recursive self-improvement"
    item = _yt(id="youtube:UCnouralabs:rsi29", summary=RSI_RULE_FREE_ROW, why=why,
               title="The First Real RSI is Here and It’s Evolving Fast")
    written = _publish(intel_state, item, profile=RSI_DIGEST_PROFILE)
    digest = intel_state / "vault" / DIGEST_FILE
    assert digest.is_file(), f"the item never reached {DIGEST_FILE}: {written[:200]!r}"
    assert RSI_NEWS in written
    assert why not in written, "the body fell through to `why` instead of the kept prose"
    for mark in RSI_FOOTER_MARKS:
        assert mark not in written, mark


def test_the_1561_footer_rulings_are_unmoved_by_the_label_anchor():
    """Clause 3 (#1819): the label anchor is additive, so every ruling #1561 pinned
    still holds byte-for-byte — the rule + `My Links` strip, the setext-underline
    abstention, and a footer-only summary rendering the scorer's `why`. The new scan
    runs only on text with no separator rule in it at all."""
    assert body_mod.strip_link_footer(
        "Prose about the run.\n\n______\nMy Links 🔗\n➡️ Twitter: https://x.com/a") \
        == "Prose about the run."
    setext = "Prose about the run.\n####Subhead\n______"
    assert body_mod.strip_link_footer(setext) == setext, "a setext underline was stripped"
    assert vw_mod._entry_body(
        _yt(id="youtube:UCtest:vid5", summary=FOOTER_ONLY,
            why="Scores 8/10: covers the agent-safety keyword set")
    ) == "Scores 8/10: covers the agent-safety keyword set"


def test_a_footer_less_description_passes_through_byte_for_byte_both_ways():
    """Clause 4 (#1819): the anchor list is a closed list of promo labels and not a
    classifier that prefers `why` over genuine channel copy. Both footer-less
    descriptions #1269 clause 2 rules on — the one ending in a period and the `[:500]`
    relic ending with no punctuation at all, pinned at :502-504 — pass
    `strip_link_footer` and `_entry_body` byte-for-byte, with `why` not substituted."""
    for text in (NO_FOOTER_DESCRIPTION, NO_FOOTER_CUT_MID_SENTENCE):
        assert body_mod.strip_link_footer(text) == text, repr(text[-40:])
        rendered = vw_mod._entry_body(
            _yt(id="youtube:UCtest:vid6", summary=text,
                why="Scores 9/10: the robotics roundup"))
        assert rendered == text, repr(rendered[-40:])


def test_the_scanner_spends_its_500_characters_on_prose_not_the_rule_free_footer(
        intel_state, monkeypatch):
    """The process boundary #1819 crosses: `scan_youtube_channels` calls
    `strip_link_footer` BEFORE `clip_body` (youtube_scanner.py:407), so the footer sat
    inside the 500-character budget and the clip ran out inside the chapter list —
    which is where the stored row's `human-built…` tail came from. With the label
    anchor the footer is gone before the clip, the whole news paragraph survives, and
    nothing is cut at all at this length."""
    assert len(RSI_RAW_DESCRIPTION) > LIMIT, "the fixture is no longer over the cap"
    # The control that this fixture reproduces the incident: clipped without the strip,
    # the footer is what the budget bought.
    unstripped = body_mod.clip_body(RSI_RAW_DESCRIPTION)
    assert "Brand Deals" in unstripped and "What You'll See:" in unstripped

    atom = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns:media="http://search.yahoo.com/mrss/"
      xmlns:yt="http://www.youtube.com/xml/schemas/2015"
      xmlns="http://www.w3.org/2005/Atom">
 <entry>
  <id>yt:video:RSI29</id>
  <yt:videoId>RSI29</yt:videoId>
  <yt:channelId>UCnouralabs</yt:channelId>
  <title>The First Real RSI is Here and It’s Evolving Fast</title>
  <link rel="alternate" href="https://www.youtube.com/watch?v=RSI29"/>
  <media:description>{RSI_RAW_DESCRIPTION.replace("&", "&amp;")}</media:description>
 </entry>
</feed>"""
    monkeypatch.setattr(yt_mod, "load_youtube_channels_config", lambda: [
        {"handle": "@ai revolution", "name": "AI Revolution",
         "channel_id": "UCnouralabs"}])
    monkeypatch.setattr(yt_mod, "_http_get", lambda url, headers=None, timeout=None: atom)

    items, coverage = yt_mod.scan_youtube_channels()

    assert coverage.fetched == 1 and len(items) == 1
    summary = items[0].summary
    for mark in RSI_FOOTER_MARKS:
        assert mark not in summary, mark
    # The prose arrives whole: the footer no longer spends the budget, so at this
    # length there is nothing left to cut and no ellipsis to mark it with.
    assert summary == RSI_NEWS, repr(summary[-60:])
    assert not summary.endswith(ELL)
    # The prose is the channel's own, kept byte-for-byte as clause 1 keeps it.
    assert summary.startswith("Weco’s AIDE² just showed")


# ── #1861: the channel's link block sits MID-description, and neither anchor reaches it ──
#
# #1561 anchored its strip on a separator rule and #1819 on a closed list of footer
# labels, and both then required the block's WHOLE tail to be link lines — which is
# #1561's abstention and stays. Nate B. Jones publishes no rule line, his `Full post:` +
# Substack link and his `My Links 🔗` block sit at lines 2-9 of 41, and genuine prose
# follows them, so both anchors abstain correctly and the block was published into
# `knowledge/ai-llms/youtube-digest.md` (`## 2026-09-29`, SEC 2) as knowledge prose. It
# also spent the scanner's clip budget, so the prose after it never reached the note.
#
# What removes it is not a wider label list — a list is how #1819 was cut to the labels
# of one channel and the next channel reproduced the failure. It is that every line of
# the block is a link and none of them is a sentence, which is already the property
# #1561 tests a footer's tail for. A run holding one prose line is left byte-for-byte
# alone: the same protection those clauses pin, applied wherever the block sits.

#: The description `youtube:gbbPBr3OraQ` sent on 2026-09-29, verbatim from
#: `yt-dlp --skip-download --print "%(description)s"`, minus that command's own trailing
#: newline: 2218 characters, 41 lines. The channel's two link runs sit at lines 2-3 and
#: 5-9; the prose the video is about runs from line 11; `Chapters:` and its list are
#: lines 23-35.
MUSE_RAW_DESCRIPTION = (
    "I gave Meta's new AI assistant, Muse, the most boring job I had: find the subscriptions I forgot to cancel. It found $5,350 a year, and it has cancelled $1,285 a year of that so far. That same small job is why I think Amazon locked Muse out of its store.\n"
    "\n"
    "Full post:\n"
    "https://natesnewsletter.substack.com/p/meta-muse-agentic-shopping?utm_source=youtube&utm_medium=video&utm_campaign=free-to-paid&utm_content=description\n"
    "\n"
    "My Links 🔗\n"
    "👉🏻 Nate's Library MCP: https://unlock-ai.natebjones.com/guides/how-to-connect-nates-library?utm_source=youtube&utm_medium=video&utm_campaign=free-to-paid&utm_content=description\n"
    "👉🏻 X: https://x.com/natebjones\n"
    "👉🏻 TikTok: https://www.tiktok.com/@nate.b.jones\n"
    "👉🏻 Instagram: https://www.instagram.com/nate.b.jones\n"
    "\n"
    "What's really happening inside Meta's Muse?\n"
    "\n"
    "The common story is that the company with the smartest model wins, but the real question is which assistant you trust with the recurring work of your life.\n"
    "\n"
    "In this video, I share the inside scoop on why Muse took off and who it threatens:\n"
    "- What people are handing Muse, from subscriptions to insurance to kids' schedules\n"
    "- How Muse makes AI usable for people who never cared about AI\n"
    "- Why I read Amazon's block as a fight over its ad business\n"
    "- Where Meta's money comes from if ordinary Muse use stays free\n"
    "\n"
    "Muse is early and asks for real trust with your data, and it is the clearest case yet that everyday usefulness decides who owns your attention.\n"
    "\n"
    "Chapters:\n"
    "00:00 The subscriptions Muse found and cancelled\n"
    "01:30 Real jobs people are handing to Muse\n"
    "04:24 Beyond money: family schedules and logistics\n"
    "05:58 Why Muse is easy for nontechnical people\n"
    "11:17 Why Amazon locked Muse out\n"
    "13:38 How Meta plans to make money from Muse\n"
    "15:13 Amazon versus Walmart on AI shopping\n"
    "17:54 What forgotten subscriptions are worth to businesses\n"
    "19:14 Why the smartest model does not automatically win\n"
    "22:36 The team Zuckerberg built to ship Muse\n"
    "24:41 The platform Zuckerberg has always wanted\n"
    "27:36 What Muse means for everyone else\n"
    "\n"
    "Listen to this video as a podcast.\n"
    "\n"
    "Spotify: https://open.spotify.com/show/0gkFdjd1wptEKJKLu9LbZ4\n"
    "Apple Podcasts: https://podcasts.apple.com/us/podcast/ai-news-strategy-daily-with-nate-b-jones/id1877109372")

#: The opening paragraph — the only part of that description the note was meant to carry.
MUSE_OPENING = "I gave Meta's new AI assistant, Muse, the most boring job I had: find the subscriptions I forgot to cancel. It found $5,350 a year, and it has cancelled $1,285 a year of that so far. That same small job is why I think Amazon locked Muse out of its store."

#: The question the video goes on to ask, below the channel's block. It has to SURVIVE
#: the strip, and it is why a tail-only rule cannot work here: the block is not the tail,
#: it is the middle of the description.
MUSE_PROSE_HEAD = "What's really happening inside Meta's Muse?"

#: The stored `summary` field of that row, verbatim: 454 characters, cut by `clip_body`
#: inside the last arrow line so that line's URL is gone (`👉🏻 Nate's Library MCP:…`).
#: This is the string a replay of `2026-09-29.jsonl` hands the writer.
MUSE_STORED_SUMMARY = (
    "I gave Meta's new AI assistant, Muse, the most boring job I had: find the subscriptions I forgot to cancel. It found $5,350 a year, and it has cancelled $1,285 a year of that so far. That same small job is why I think Amazon locked Muse out of its store.\n"
    "\n"
    "Full post:\n"
    "https://natesnewsletter.substack.com/p/meta-muse-agentic-shopping?utm_source=youtube&utm_medium=video&utm_campaign=free-to-paid&utm_content=description\n"
    "\n"
    "My Links 🔗\n"
    "👉🏻 Nate's Library MCP:…")

#: The trailer the item names; all three marks are in the recorded row.
MUSE_LINK_MARKS = ("Full post:", "My Links", "👉")

#: `DIGEST_PROFILE` matches on `agent`, which this row's title and summary do not say
#: (`determine_vault_path` scores `title + " " + summary`, never `why`), so this profile
#: matches on the row's own word and routes to the same `DIGEST_FILE`.
MUSE_DIGEST_PROFILE = {"topics": [{"name": "ai-llms", "weight": 0.9,
                                  "keywords": ["subscriptions"]}]}


def test_a_set_off_link_run_with_no_prose_line_is_removed_wherever_it_sits():
    """Clause 1: a set-off run of two or more non-blank lines that holds no prose line —
    every line a URL, a `_FOOTER_LABEL_RE` label, an arrow-prefixed pointer, or a line
    that is only a short colon-terminated label — is removed from the middle of channel
    text, and every line after the run comes back unchanged.
    """
    text = ("The video explains the rollout in three parts.\n\n"
            "My Links 🔗\n"
            "👉🏻 Repo: https://github.com/example/rollout\n"
            "➡️ Docs: https://example.com/rollout\n\n"
            "The second part covers the benchmark.")
    assert body_mod.strip_link_footer(text) == (
        "The video explains the rollout in three parts.\n\n"
        "The second part covers the benchmark.")
    # The run has to be a RUN. One label line above a paragraph is how a channel heads
    # its own sections, and the two-line floor is what keeps it: this text differs from
    # the one above only in that the arrow line is gone.
    one_line = ("The video explains the rollout in three parts.\n\n"
                "My Links 🔗\n\n"
                "The second part covers the benchmark.")
    assert body_mod.strip_link_footer(one_line) == one_line, \
        "a single link line was removed: the floor is two or more non-blank lines"


def test_the_recorded_muse_description_keeps_its_prose_and_loses_its_link_block():
    """Clause 2: the recorded `gbbPBr3OraQ` description comes back with no `Full post:`,
    no `My Links` and no `👉` line, and still carrying its opening paragraph and the
    sentence `What's really happening inside Meta's Muse?` that sits BELOW the removed
    block. Neither anchor can see it: #1561's needs a separator rule (the description has
    none) and #1819's needs the block to be the whole tail (prose follows it)."""
    assert len(MUSE_RAW_DESCRIPTION) == 2218, "the fixture stopped being the recorded row"
    assert not _RULE_AT_LINE_START.search(MUSE_RAW_DESCRIPTION), \
        "the fixture grew a separator rule and is no longer the shape under test"
    stripped = body_mod.strip_link_footer(MUSE_RAW_DESCRIPTION)
    for mark in MUSE_LINK_MARKS:
        assert mark not in stripped, mark
    assert stripped.startswith(MUSE_OPENING)
    assert MUSE_PROSE_HEAD in stripped, "the prose after the block did not survive"


def test_the_clip_budget_reaches_the_prose_once_the_mid_description_block_is_gone():
    """The scanner-side half of the defect, across the boundary that decides it:
    `scan_youtube_channels` calls `strip_link_footer` before `clip_body`
    (youtube_scanner.py:407), so while the block survived the strip it spent the
    500-character budget and the clip ended inside the block — which is why the stored
    row is 454 characters of link block with no `What's really happening` prose in it.
    With the block gone first, the budget buys the channel's prose instead."""
    assert len(MUSE_STORED_SUMMARY) < len(MUSE_RAW_DESCRIPTION)
    unstripped = body_mod.clip_body(MUSE_RAW_DESCRIPTION)
    assert "My Links" in unstripped and MUSE_PROSE_HEAD not in unstripped, \
        "the control: without the strip the clip still spends its budget on links"
    summary = body_mod.clip_body(body_mod.strip_link_footer(MUSE_RAW_DESCRIPTION))
    for mark in MUSE_LINK_MARKS:
        assert mark not in summary, mark
    assert MUSE_PROSE_HEAD in summary, repr(summary[:200])
    # Still a clip, and still inside its budget. `clip_body` cuts at the last word
    # boundary at or under `limit` and THEN appends its marker, so the shipped ceiling is
    # `limit` plus that one character — a shape of that function, not of this round.
    assert summary.endswith(body_mod.ELLIPSIS)
    assert len(summary) <= body_mod.SUMMARY_LIMIT + len(body_mod.ELLIPSIS)


def test_a_link_run_with_a_prose_line_inside_it_is_returned_byte_for_byte():
    """Clause 3: the run rule is additive, and the protection it rests on is the one
    #1561 and #1819 already pin. A set-off run with ONE prose line among its links is
    returned exactly as sent, and the earlier rulings hold — a rule over `My Links` plus
    `➡️ Twitter:` is stripped, a `____` with a prose paragraph under it keeps everything,
    and a setext underline is not a promo rule."""
    prose_in_run = ("The video explains the rollout in three parts.\n\n"
                    "My Links 🔗\n"
                    "👉🏻 Repo: https://github.com/example/rollout\n"
                    "The benchmark section is the best part of it.\n\n"
                    "Closing thoughts here.")
    assert body_mod.strip_link_footer(prose_in_run) == prose_in_run, \
        "a run holding a prose line was removed"
    assert body_mod.strip_link_footer(
        "Prose about the run.\n\n______\nMy Links 🔗\n➡️ Twitter: https://x.com/a") \
        == "Prose about the run."
    under_the_rule = ("Prose about the run.\n\n____\n\n"
                      "A paragraph the channel wrote about the topic itself.")
    assert body_mod.strip_link_footer(under_the_rule) == under_the_rule
    setext = "Prose about the run.\n####Subhead\n______"
    assert body_mod.strip_link_footer(setext) == setext, "a setext underline was stripped"


def test_replaying_the_stored_muse_row_writes_no_link_block_to_the_digest(intel_state):
    """Clause 4: the stored, already-clipped `summary` — 454 characters, ending inside an
    arrow line with that line's URL cut off — loses its `My Links 🔗` + `👉🏻` run to the
    same call, and written through `write_item_to_vault` the digest carries the opening
    paragraph and none of the three link marks. A replay of `2026-09-29.jsonl` therefore
    cannot reproduce the section this item was filed for.
    """
    assert len(MUSE_STORED_SUMMARY) == 454, "the fixture stopped being the stored row"
    stripped = body_mod.strip_link_footer(MUSE_STORED_SUMMARY)
    for mark in MUSE_LINK_MARKS:
        assert mark not in stripped, mark
    assert stripped == MUSE_OPENING, repr(stripped)

    item = _yt(id="youtube:UC0C-17n9iuUQPylguM1d-lQ:gbbPBr3OraQ",
               summary=MUSE_STORED_SUMMARY,
               why="Scores 7/10: an assistant doing the subscription audit",
               title="I Gave Meta's Muse The Most Boring Job I Had. "
                     "It Found $5,350 A Year.")
    written = _publish(intel_state, item, profile=MUSE_DIGEST_PROFILE)
    digest = intel_state / "vault" / DIGEST_FILE
    assert digest.is_file(), f"the item never reached {DIGEST_FILE}: {written[:200]!r}"
    assert MUSE_OPENING in written
    for mark in MUSE_LINK_MARKS:
        assert mark not in written, mark


# ── #1900: the two ad shapes that are not a TAIL, so no backwards anchor reaches them ──
#
# #1561, #1819 and #1861 all strip from the END: a rule line with links under it, a
# footer label with links under it, a run of link lines. Two shapes ran into the digest on
# 2026-09-30 that are not tails at all. TheAIGRID's standing self-intro sits at offset 0
# and is the WHOLE description, so `vault_writer.py:426` sees `stripped == summary`, takes
# the no-footer branch, and publishes the ad as 100 % of the body — the video's subject
# appears nowhere in the note. The Manus row's `👉 Join the free GPT-6 Astra Crash Course`
# is ONE arrow line, and `_link_run_bounds`'s floor of two link lines is what makes a run
# a run, so a lone sign-off line was never a candidate.
#
# The fix is two more closed lists, not a classifier, and the tests below are mostly
# abstentions: `👉 My repo: https://…` is a pointer to the video and stays, `👉 Join the
# course` with no URL stays, a greeting above real prose stays. Frequency first, as the
# item asked: 1 greeting and 1 signup line in 1078 stored youtube rows
# (`~/lloyd-data/_pipeline/vault-derived/memory/feeds/raw/*.jsonl`, 2026-09-22→09-30).

#: TheAIGRID's description for `youtube:UCbY9xX3_jW5c2fjlZVBI4cg:o3YTzebEs18`, verbatim
#: from the `summary` field of `raw/2026-09-30.jsonl`: 500 characters, one paragraph, the
#: `…` being `clip_body`'s own cut marker. All three greeting anchors occur in it.
THEAIGRID_GREETING = (
    "Welcome to TheAIGRID — the place to learn AI for free. I create simple, practical "
    "videos that help beginners, creators, entrepreneurs, and business owners understand "
    "artificial intelligence, AI tools, automation, AI agents, robotics, ChatGPT, Claude, "
    "Gemini, and the future of technology. Whether you want AI tutorials, tool breakdowns, "
    "beginner guides, or explanations of the latest breakthroughs, this channel gives you "
    "the knowledge you need to stay ahead. Subscribe to start learning AI for free…")

#: TheAIGRID's title and its scorer's `why`, both verbatim: the title from `summary`'s
#: sibling `title` field in `raw/2026-09-30.jsonl`, the `why` from the model-graded row of
#: the same video in `intel-2026-09-30.jsonl` (whose bytes are committed for this item at
#: `backlog/data/intel-2026-09-30.jsonl`, 15 lines). The `why` is what the entry must carry
#: once the greeting is treated as footer-only-equivalent, and it mentions neither
#: `Welcome to` nor `Subscribe`, so the digest assertion below has something to prove.
THEAIGRID_TITLE = "OpenAI Just Revealed Dots… This Changes ChatGPT Forever"
THEAIGRID_WHY = ("The video discusses a major update to ChatGPT, which is directly "
                 "relevant to the user's interest in AI and LLMs, though it lacks "
                 "specific focus on robotics, voice, or hardware.")

#: A `why` that says what the video is about and names neither the reader nor a profile:
#: the kind #1155 ruled is a body. And the named no-body form #2011 gives a `why` that
#: only rates the item, which is what `THEAIGRID_WHY` above does.
DESCRIPTIVE_WHY = "OpenAI showed Dots, a ChatGPT interface that keeps several threads live."
NO_BODY = "None — the feed carried no description of this item"

#: The news paragraph of `youtube:UC5l7RouTQ60oUjLjt1Nh-UQ:L75zF2WKiVE`: 279 characters of
#: the video's actual subject, from the same stored file. It has to survive uncut.
MANUS_NEWS = (
    "Manus 2.0 just turned AI agents into something much closer to digital people, giving "
    "Cue agents their own phone, email, wallet and computer. Meanwhile Tencent secretly "
    "tests an AI gaming companion, while Claude Sonnet 5.5 beats Opus 5.5 on agentic "
    "coding at half the token price.")

#: The signup line the same description ends with: 82 characters, arrow-prefixed, one URL.
#: The whole of shape 2, and the only arrow-CTA in the 1078-row corpus.
MANUS_CTA = "👉 Join the free GPT-6 Astra Crash Course here: https://links.outskill.com/AIRVOCT1"

#: The stored `summary` of that row verbatim: news, blank line, CTA. 363 characters.
MANUS_STORED_SUMMARY = MANUS_NEWS + "\n\n" + MANUS_CTA

#: Title and `why` verbatim from the two stored files, by the same route as above. Both
#: rows carry `relevance: 6`, which is what the fixtures below pass.
MANUS_TITLE = ("New Manus 2.0 is Fully Autonomous, New Sonnet 5.5 Beats Opus, "
               "Tencent AI Companion & More AI News")
MANUS_WHY = ("The item focuses on autonomous AI agents and LLM benchmarks, which aligns "
             "with the ai-llms interest, though it lacks specific robotics or hardware "
             "details.")

#: `DIGEST_PROFILE`'s keyword is `agent`, and `match_keywords` tests a single-word keyword
#: WHOLE-WORD (`profile.py:180`): both stored rows say `agents`, neither says `agent`, so
#: with that profile they route to `knowledge/feeds/youtube-uncategorized.md` and never
#: reach the file this item is about. These two keywords are each a word one of the stored
#: titles really contains — `determine_vault_path` scores `title + " " + summary`, never
#: `why` — which is what sends both rows to `DIGEST_FILE`.
STORED_ROW_PROFILE = {"topics": [{"name": "ai-llms", "weight": 0.9,
                                  "keywords": ["chatgpt", "manus"]}]}

#: What the CTA rule must never touch: a pointer to the video's own subject, an ask with
#: no destination, and an ask welded into a paragraph instead of set off by a blank.
POINTER_NOT_A_SIGNUP = "👉 My repo: https://github.com/example/rollout"
SIGNUP_WITHOUT_URL = "👉 Join the waitlist for the next cohort"
CTA_HEADING = "Prose about the model. Two sentences, both about the video."


def test_a_description_that_is_only_a_channel_greeting_strips_to_nothing_and_renders_why():
    """Clause 1 (#1900): the standing self-intro IS the description, so the strip returns
    "" — the value the writer already understands as "the body WAS the footer".

    `ends_a_sentence("")` is False, so `_entry_body` falls through to the scorer's `why`
    and the digest carries a reason instead of a subscriber appeal. Without the "" the
    writer's `stripped == summary` branch at `vault_writer.py:426` returns the greeting
    verbatim, which is precisely what reached the note on 2026-09-30. The re-flowed form is
    in the same assertion because the rule asks the question of every non-blank LINE: an
    intro broken over two paragraphs is still an intro and nothing else, because each of
    its lines carries one of the three phrases.
    """
    assert len(THEAIGRID_GREETING) == 500, "the fixture stopped being the stored row"
    assert "this channel gives you" in THEAIGRID_GREETING
    assert body_mod.strip_link_footer(THEAIGRID_GREETING) == ""
    wrapped = THEAIGRID_GREETING.replace(
        "Whether you want AI tutorials", "breakthroughs.\n\nWhether you want AI tutorials")
    assert body_mod.strip_link_footer(wrapped) == ""

    # A `why` that describes the video is what the entry carries, byte for byte.
    item = _yt(id="youtube:UCbY9xX3_jW5c2fjlZVBI4cg:o3YTzebEs18", relevance=6,
               summary=THEAIGRID_GREETING, why=DESCRIPTIVE_WHY, title=THEAIGRID_TITLE)
    assert vw_mod._entry_body(item) == DESCRIPTIVE_WHY
    for ad in ("Welcome to", "Subscribe"):
        assert ad not in vw_mod._entry_body(item), ad
    # The stored `why` of this row rates it against the interest profile instead
    # (#2011), so with it the entry names that it has no body. The ad is refused
    # either way.
    rated = _yt(id="youtube:UCbY9xX3_jW5c2fjlZVBI4cg:o3YTzebEs18", relevance=6,
                summary=THEAIGRID_GREETING, why=THEAIGRID_WHY, title=THEAIGRID_TITLE)
    assert vw_mod._entry_body(rated) == NO_BODY
    for ad in ("Welcome to", "Subscribe"):
        assert ad not in vw_mod._entry_body(rated), ad


def test_a_greeting_above_real_prose_is_returned_byte_for_byte():
    """Clause 2 (#1900): the greeting rule fires on a description that is NOTHING but the
    intro and on nothing else. Three shapes, all of them prose the strip must hand back
    untouched:

    - the intro above the video's subject, below a blank line — TheAIGRID's own phrases
      followed by a real story;
    - the same intro with the news on the NEXT LINE and no blank between, which is one
      paragraph to a human and one line more to the rule, and the news line carries none
      of the three phrases;
    - a description that opens with `Welcome to` and is a single paragraph of news
      throughout — `Welcome to my deep dive today, which covers how Manus 2.0 gave Cue
      agents their own phone` — where the anchor's presence means nothing, because a
      standing boilerplate says all three things and a first sentence says one.

    A strip that preferred `why` on any of these would be doing the thing #1819 clause 4
    forbids and #856 made expensive: editing the opening of the channel's copy on a guess.
    """
    greeting_then_prose = THEAIGRID_GREETING + "\n\n" + MANUS_NEWS
    assert body_mod.strip_link_footer(greeting_then_prose) == greeting_then_prose

    no_blank = THEAIGRID_GREETING + "\n" + MANUS_NEWS
    assert body_mod.strip_link_footer(no_blank) == no_blank, \
        "news on the line under a greeting was deleted with it"

    one_phrase = ("Welcome to my deep dive today, which covers how Manus 2.0 gave Cue "
                  "agents their own phone, email, wallet and computer, and why that "
                  "matters")
    assert len(body_mod._greeting_kinds(one_phrase)) == 1, \
        "the counterexample grew a second phrase"
    assert body_mod.strip_link_footer(one_phrase) == one_phrase, \
        "a single-phrase `Welcome to` opening was treated as the standing intro"

    item = _yt(summary=greeting_then_prose, relevance=6, why=THEAIGRID_WHY,
               title=THEAIGRID_TITLE)
    assert vw_mod._entry_body(item) == greeting_then_prose, \
        "a greeting above real prose was replaced by the scorer's `why`"


def test_a_set_off_signup_line_is_dropped_and_every_other_line_survives_byte_for_byte():
    """Clause 3 (#1900): one arrow-prefixed line whose first word after the arrow is a
    signup verb and which carries a URL is dropped, and every other line of the description
    comes back byte-for-byte. `_link_run_bounds` needs two link lines to call them a run,
    and that floor of two is exactly what let a single sign-off line through.

    The three abstentions are the rule's whole edge: a pointer to the video's own subject
    opens with a word that is not a signup verb; an ask with no URL has no destination to
    advertise; and a line with no blank above it is inside someone's paragraph, where
    dropping it would edit prose. All three come back unchanged.

    Spacing is part of the clause, not decoration: the blank that set the dropped line off
    goes with it when the line ended its paragraph, and stays when it heads a link run,
    because there the blank is the run's separator — eating it would weld the run onto the
    prose and move it out of the run rule's reach.
    """
    mid_body = CTA_HEADING + "\n\n" + MANUS_CTA + "\n\n" + MANUS_NEWS
    assert body_mod.strip_link_footer(mid_body) == CTA_HEADING + "\n\n" + MANUS_NEWS
    assert body_mod.strip_link_footer(MANUS_STORED_SUMMARY) == MANUS_NEWS
    for arrow_line in ("📌 Sign up for the cohort: https://example.com/cohort",
                       "➡️ Register here: https://example.com/waitlist",
                       "👉🏻 Join the course: https://example.com/course"):
        assert body_mod.strip_link_footer("Real prose about the video.\n\n" + arrow_line) \
            == "Real prose about the video."

    for abstention in (POINTER_NOT_A_SIGNUP, SIGNUP_WITHOUT_URL):
        text = CTA_HEADING + "\n\n" + abstention
        assert body_mod.strip_link_footer(text) == text, abstention
    welded = CTA_HEADING + "\n" + MANUS_CTA
    assert body_mod.strip_link_footer(welded) == welded, \
        "a CTA line with no blank above it was dropped out of a paragraph"

    # A signup line that HEADS a link run is the interesting case, and the blank under it
    # decides what happens: that blank is the RUN's separator, not the line's, so it stays,
    # the remaining two link lines are still set off, and #1861's run rule removes them too.
    # A dropped line's own separator is taken only when it ENDED its paragraph.
    run_head = (CTA_HEADING + "\n\n" + MANUS_CTA
                + "\n🔴 Subscribe: https://youtube.com/@example"
                + "\n🌐 Website: https://example.com")
    assert body_mod.strip_link_footer(run_head) == CTA_HEADING, \
        "a signup line heading a link run left its separator above the run"
    run_mid = run_head + "\n\n" + MANUS_NEWS
    assert body_mod.strip_link_footer(run_mid) == CTA_HEADING + "\n\n" + MANUS_NEWS, \
        "a run removed below the fold took the prose under it"


def test_replaying_the_two_stored_rows_writes_neither_ad_to_the_digest(intel_state):
    """Clause 4 (#1900): both stored 2026-09-30 descriptions through the real writer. The
    greeting row publishes neither greeting phrase (nor, since #2011, its `why`, which
    rates the video against the interest profile: it carries the named no-body form); the Manus row
    publishes its news paragraph WHOLE — equal to it, not merely containing it — and
    neither the course URL nor its arrow. A replay of `2026-09-30.jsonl` therefore cannot
    reproduce the two sections this item was filed for, which is the only claim here that
    is about the file on disk and not about a rule.
    """
    assert len(MANUS_STORED_SUMMARY) == 363 and len(MANUS_NEWS) == 279 \
        and len(MANUS_CTA) == 82, "the fixture stopped being the stored row"

    greeting_item = _yt(id="youtube:UCbY9xX3_jW5c2fjlZVBI4cg:o3YTzebEs18", relevance=6,
                        summary=THEAIGRID_GREETING, why=THEAIGRID_WHY, title=THEAIGRID_TITLE)
    news_item = _yt(id="youtube:UC5l7RouTQ60oUjLjt1Nh-UQ:L75zF2WKiVE", relevance=6,
                    summary=MANUS_STORED_SUMMARY, why=MANUS_WHY, title=MANUS_TITLE)
    # The expected value comes out of the stored INPUT, not out of the constant that built
    # it: everything before the first blank line is what the news paragraph is, and its
    # length is the number the item quotes. `MANUS_NEWS` is then checked against that, so
    # the assertion fails if either the fixture or the strip drifts.
    news_only = MANUS_STORED_SUMMARY.split("\n\n")[0]
    assert len(news_only) == 279 == len(MANUS_NEWS) and news_only == MANUS_NEWS
    assert vw_mod._entry_body(greeting_item) == NO_BODY      # #2011: its `why` is a rating
    assert vw_mod._entry_body(news_item) == news_only

    # The OTHER caller of the strip is the scanner, and it crosses a real process
    # boundary: `youtube_scanner.py:407` stores `clip_body(strip_link_footer(desc))` into
    # the row's `summary`, so a greeting-only description is recorded EMPTY and every later
    # reader of that JSONL row — stage 2, a replay, this replay — sees the same thing the
    # writer sees. The Manus row keeps its news paragraph there too. The empty-summary
    # shape is not new to the pipeline: stage 1 left `summary` empty for every YouTube row
    # until #1155, and `why` is what the writer renders for it.
    assert body_mod.clip_body(body_mod.strip_link_footer(THEAIGRID_GREETING)) == ""
    assert body_mod.clip_body(body_mod.strip_link_footer(MANUS_STORED_SUMMARY)) \
        == MANUS_NEWS

    written = _publish(intel_state, greeting_item, news_item, profile=STORED_ROW_PROFILE)
    digest = intel_state / "vault" / DIGEST_FILE
    assert digest.is_file(), f"neither item reached {DIGEST_FILE}: {written[:200]!r}"
    for ad in ("Welcome to", "Subscribe", "links.outskill.com", "👉"):
        assert ad not in written, ad
    assert MANUS_NEWS in written
    assert THEAIGRID_WHY not in written and NO_BODY in written


def test_the_1269_pin_survives_both_new_rules():
    """Clause 5 (#1900): the two descriptions #1269 clause 2 rules on are still published
    as the summary, after these rules exist and because they do not apply.

    `NO_FOOTER_DESCRIPTION` and `NO_FOOTER_CUT_MID_SENTENCE` are the strings
    `test_a_youtube_description_under_the_cap_is_still_passed_through_untouched` already
    pins, imported here by name rather than restated so this node cannot drift from it.
    What this node adds is the interaction: a greeting-shaped or CTA-shaped string could
    have made the writer prefer `why` for those rows too, and it must not — the strip's
    abstention is what #1269's ruling is made of, and #1819 clause 4 pinned it for the
    footer anchors.
    """
    assert body_mod.strip_link_footer(NO_FOOTER_DESCRIPTION) == NO_FOOTER_DESCRIPTION
    assert body_mod.strip_link_footer(NO_FOOTER_CUT_MID_SENTENCE) \
        == NO_FOOTER_CUT_MID_SENTENCE
    plain = _yt(id="youtube:UCtest:vid5", summary=NO_FOOTER_DESCRIPTION,
                why="Scores 8/10: robotics")
    assert vw_mod._entry_body(plain) == NO_FOOTER_DESCRIPTION
    cut = _yt(id="youtube:UCtest:vid6", summary=NO_FOOTER_CUT_MID_SENTENCE,
              why="Scores 8/10: robotics")
    assert vw_mod._entry_body(cut) == NO_FOOTER_CUT_MID_SENTENCE


# ── #1925: the headings the floor discounted came back in the paste ─────────────
#
# `clean_body` dropped every line matching `_SCAFFOLD_LINE_RE` into `content`, measured
# `content` against `BODY_FLOOR`, and returned `body` — the text from BEFORE that loop.
# A filled PR body therefore cleared the floor on its prose and published the author's
# `# Description` anyway: on 2026-09-30, 450 of the 4,159 `## 2026-MM-DD` dated sections
# across `knowledge/tools/*/{prs,updates}.md` carried such a line (10.8 %), and the
# entry this run wrote for PR #8128 opened its body with an h1 that outranks the dated
# heading it sits under — the digest's own section structure, flattened by upstream
# authors rather than by Lloyd.
#
# `PR_8128_BODY` and `COMMIT_8183_BODY` below are the two bodies that run published.
# The four tests after them are the item's four acceptance clauses, in order.

PR_8128_TITLE = "Make uv the only installation workflow"
#: The opening paragraph of isaac-sim/IsaacLab PR #8128 as it reached the digest. 202
#: characters: over `BODY_FLOOR`, and under `SUMMARY_LIMIT`, so no assertion below
#: depends on `clip_body` having cut anything.
PR_8128_PROSE = (
    "Make uv the only supported Isaac Lab installation workflow. Remove "
    "`isaaclab.sh`, `isaaclab.bat`, the Python CLI installer "
    "(-i / --install), conda provisioning, and legacy environment "
    "creation commands.")
PR_8128_BODY = "# Description\n\n" + PR_8128_PROSE + "\n"

COMMIT_8183_TITLE = "Remove obsolete ROS 2 Docker image (#8183)"
COMMIT_8183_BODY = (
    "## Summary\n\n"
    "Remove the obsolete ROS 2 Docker image and its dedicated files: "
    "`Dockerfile.ros2`, `.env.ros2`, and DDS configuration.\n\n"
    "## Type of change\n\n"
    "Breaking change: the `ros2` container profile is gone.\n")

#: The clause 1-3 rail: the heading-marker half of `_SCAFFOLD_LINE_RE`, spelled as the
#: acceptance clause spells it. A level sign followed by a space, or nothing.
_HEADING_LINE = re.compile(r"^#{1,6}(\s|$)")


# ── clause 1 — the floor's own discount applies to what gets pasted ─────────────

def test_a_body_over_the_floor_returns_without_the_heading_it_was_measured_without():
    assert len(PR_8128_PROSE) >= body_mod.BODY_FLOOR, "the fixture must clear the floor"

    body, reason = body_mod.clean_body(PR_8128_BODY, PR_8128_TITLE)

    assert reason is None, reason
    first = next(line for line in body.splitlines() if line.strip())
    assert first == PR_8128_PROSE, first
    assert body == PR_8128_PROSE, body
    assert not [ln for ln in body.splitlines() if _HEADING_LINE.match(ln)], body
    # The pre-fix return value, asserted absent by name rather than by the shape that
    # would happen to exclude it: `# Description` was the line that shipped.
    assert "# Description" not in body, body


# ── clause 2 — nothing but a heading marker may be removed ──────────────────────

#: A filled release body carrying every shape that is NOT a heading: checkbox lines with
#: text on them (the trap in `return "\n".join(content)`, which drops the whole line), a
#: hashtag line, an ordinary bullet, a bare URL — plus one HTML comment and one bare
#: template phrase, which are stripped before this clause is ever asked.
FILLED_CHECKLIST_BODY = "\n".join([
    "<!--",
    "Please include a summary of the change and which issue is fixed.",
    "-->",
    "",
    "# Description",
    "",
    "Kit visualizer markers now reach the Newton visualizers.",
    "",
    "## Type of change",
    "",
    "- [ ] Bug fix",
    "- [ ] New feature",
    "- [ ] Added Newton visualizer support",
    "",
    "Please include a summary of the change with before and after numbers.",
    "",
    "- `isaaclab.sh` and `isaaclab.bat` are gone",
    "#shipit: the hashtag line is content, not a heading",
    "",
    "https://github.com/isaac-sim/IsaacLab/pull/8128",
])

#: Every line of `FILLED_CHECKLIST_BODY` the floor counts, in the order upstream wrote
#: them. `#shipit…` is a level sign with no space after it, so it is a hashtag and not a
#: heading.
FILLED_CHECKLIST_KEPT = [
    "Kit visualizer markers now reach the Newton visualizers.",
    "- `isaaclab.sh` and `isaaclab.bat` are gone",
    "#shipit: the hashtag line is content, not a heading",
    "https://github.com/isaac-sim/IsaacLab/pull/8128",
]

#: The three checkbox lines of that body. #1925 kept them in the published text for the
#: sake of the third, which carries real content; #2012 takes all three out, because the
#: floor has never counted a checkbox line and an unfilled checklist was clearing it as
#: prose. The loss of `- [ ] Added Newton visualizer support` is the price, pinned here
#: so it is a decision on the record and not something a later reader finds.
FILLED_CHECKLIST_DROPPED = [
    "- [ ] Bug fix",
    "- [ ] New feature",
    "- [ ] Added Newton visualizer support",
]

_CHECKBOX_LINE = re.compile(r"^\s*[-*+]\s*\[[ xX]?\]")


def test_no_line_the_floor_counts_is_removed_and_no_checkbox_line_is_returned():
    body, reason = body_mod.clean_body(FILLED_CHECKLIST_BODY, "Extend Kit markers to Newton")

    assert reason is None, reason
    rows = body.splitlines()
    for line in FILLED_CHECKLIST_KEPT:
        assert line in rows, f"{line!r} missing from:\n{body}"
    # …verbatim AND in order, not merely present.
    positions = [rows.index(line) for line in FILLED_CHECKLIST_KEPT]
    assert positions == sorted(positions), positions
    for line in FILLED_CHECKLIST_DROPPED:
        assert line not in rows, f"{line!r} still published:\n{body}"
    assert not [ln for ln in rows if _CHECKBOX_LINE.match(ln)], body
    assert not [ln for ln in rows if _HEADING_LINE.match(ln)], body
    # The comment and the bare template phrase are gone, as they were before this fix.
    assert "<!--" not in body and "Please include a summary" not in body, body


def test_a_hash_comment_inside_a_fence_goes_with_the_headings_and_says_so():
    """The one content cost of a line-based strip, pinned here rather than found later.

    `# install the pinned toolchain` is a shell comment inside a fenced block, and to a
    line-based pattern it is indistinguishable from `# Description`. It goes — the same
    line the floor measurement has always discounted when it judged this body worth
    publishing, so the measurement and the publication stay consistent at the cost of one
    comment line. The command below it, which is the reason the block exists, survives.
    """
    fenced = ("Moves the install docs onto uv. Everything below is what a reader copies "
              "into a shell.\n\n```bash\n# install the pinned toolchain\n"
              "uv sync --extra isaacsim\n```\n")

    body, reason = body_mod.clean_body(fenced, "Install docs onto uv")

    assert reason is None, reason
    assert "# install the pinned toolchain" not in body, body
    assert "uv sync --extra isaacsim" in body, body
    assert "```bash" in body and "```" in body, body


# ── clause 3 — the three refusal rulings are unmoved ────────────────────────────

@pytest.mark.parametrize("text, reason", [
    (UNFILLED_TEMPLATE, "the upstream body is an unfilled PR template"),
    ("# Description\n\n## Motivation\n",
     "the upstream body is template headings with nothing under them"),
    ("fix typo", "the upstream body is under 40 characters"),
])
def test_the_three_refusal_rulings_survive_the_heading_strip(text, reason):
    """Same headings, same refusal — the strip only changes what a PASSED body looks like.

    Every fixture below still fails the floor on `content_text`, which is computed
    exactly as it was: `# Description` and `## Motivation` were discounted before the
    comparison and are discounted now. Removing the heading from the returned value must
    not move a body across the boundary, so the reason strings are pinned whole and not
    by keyword.
    """
    body, got = body_mod.clean_body(text, "Add a scheduler knob")

    assert body is None, body
    assert got == reason, got


# ── clause 4 — the seam: a real `--write` run into a redirected vault ───────────

def test_a_write_run_publishes_those_bodies_with_no_heading_under_the_dated_section(
        tmp_path):
    """Process boundary: scanner jsonl → `python -m intel_pipeline --write` → vault file.

    `clean_body` and `vault_writer` share a process, but the digest they build is the
    artefact, and only a subprocess with `HOME` redirected proves the entry that reaches
    disk carries no heading — which is what 450 sections of the live corpus say it did not
    do on 2026-09-30. Each entry appends its own `## <date>` section, so the section count
    is the entry count: exactly one per run.
    """
    home = tmp_path / "home"
    feeds = home / "lloyd-data" / "_pipeline" / "vault-derived" / "memory" / "feeds"
    (feeds / "raw").mkdir(parents=True)
    vault = home / "obsidian"
    (vault / "knowledge").mkdir(parents=True)
    (vault / "interests.md").write_text(
        "---\ntitle: Interests\n---\n\n## AI & LLMs\ninference, vllm\n", encoding="utf-8")
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _write(records):
        (feeds / f"intel-{day}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        return subprocess.run(
            [sys.executable, "-m", "intel_pipeline", "--write", "--date", day],
            cwd=str(INTEL_DIR),
            env=dict(os.environ, HOME=str(home), LLOYD_DATA=str(home / "lloyd-data")),
            capture_output=True, text=True, timeout=180)

    def _digest_for(title):
        hits = [p for p in (vault / "knowledge").rglob("*.md")
                if title in p.read_text(encoding="utf-8")]
        assert len(hits) == 1, f"{title!r} in {hits}"
        return hits[0]

    first = _write([_item(PR_8128_TITLE, PR_8128_BODY, item_id="8128").to_dict()])
    assert first.returncode == 0, first.stderr[-2000:]

    target = _digest_for(PR_8128_TITLE)
    text = target.read_text(encoding="utf-8")
    # The entry sits under its own dated section, and the body line under it is the
    # prose — the heading that used to sit between them is not there.
    assert re.search(
        rf"^## {day}\n\n### {re.escape(PR_8128_TITLE)}\n", text, re.M), text
    assert f"**Source:** github | **Relevance:** 6/10\n\n{PR_8128_PROSE}" in text, text
    assert not re.findall(r"^#{1,3} (Description|Summary|Type of change)$", text, re.M)
    assert len(re.findall(r"^## 2026-", text, re.M)) == 1, text

    second = _write([_item(COMMIT_8183_TITLE, COMMIT_8183_BODY,
                           item_id="8183").to_dict()])
    assert second.returncode == 0, second.stderr[-2000:]

    # One more run, one more entry, and exactly one more dated section in the same file:
    # the strip does not swallow the entry, duplicate it, or nest it under the last one.
    assert _digest_for(COMMIT_8183_TITLE) == target
    text2 = target.read_text(encoding="utf-8")
    assert len(re.findall(r"^## 2026-", text2, re.M)) == 2, text2
    assert re.search(
        rf"^## {day}\n\n### {re.escape(COMMIT_8183_TITLE)}\n", text2, re.M), text2
    assert not re.findall(r"^#{1,3} (Description|Summary|Type of change)$", text2, re.M)
    assert "## Summary" not in text2 and "## Type of change" not in text2, text2


# ── #2012 — what the floor does not count is not published ────────────────────
#
# IsaacLab commit ae5d4e39 was written to `knowledge/tools/isaaclab/updates.md` on
# 2026-10-01 carrying `Fixes # (issue)` and `- [x]  Backport this pull request to the`
# under a paragraph about a calibration photo: the author's PR checklist, published as
# knowledge. The floor discounted the checkbox line and the return pasted it back.

#: The `summary` of that row in `intel-2026-10-01.jsonl`, byte for byte (its trailing
#: `…` is the scanner's own clip).
SO101_TITLE = "[Docs] Add SO-101 leader calibration pose image (#8219)"
SO101_BODY = (
    "[Docs] Add SO-101 leader calibration pose image (#8219)\n\n# Description\n\n"
    "Adds a photo of the SO-101 leader arm in the mid-range calibration pose\n"
    "to the SO-101\njoint teleop example in the Isaac Teleop docs, plus a tip to set the\n"
    "gripper to its\n  midpoint during that step.\n\nFixes # (issue)\n\n"
    "## Type of change\n\n- Documentation update\n\n## Release backport\n\n"
    "- [x]  Backport this pull request to the\n"
    "active release branch after it merges into `develop`\n\n## Checklist\n\n"
    "Docker and GPU tests run on demand.…")


def test_the_real_so101_body_keeps_its_prose_and_loses_the_checklist():
    body, reason = body_mod.clean_body(SO101_BODY, SO101_TITLE)

    assert reason is None, reason
    rows = [ln for ln in body.splitlines() if ln.strip()]
    assert rows[0] == ("Adds a photo of the SO-101 leader arm in the mid-range "
                       "calibration pose"), rows[0]
    assert not [ln for ln in rows if body_mod._SCAFFOLD_LINE_RE.match(ln)], body
    assert "Fixes # (issue)" not in rows, body
    # The wrapped tail of the checkbox line goes with the line it belongs to.
    assert "Backport this pull request" not in body, body
    assert "active release branch" not in body, body


def test_a_checkbox_line_and_its_wrapped_tail_do_not_clear_the_floor():
    """The tail of a hard-wrapped checkbox was counted as prose: 51 characters, over
    the floor on its own. With nothing else in the body there is nothing to publish."""
    text = ("## Release backport\n\n- [x]  Backport this pull request to the\n"
            "active release branch after it merges into `develop`\n")

    body, reason = body_mod.clean_body(text, "Some PR")

    assert body is None, body
    assert reason == "the upstream body is template headings with nothing under them", reason


def test_a_list_item_after_a_checkbox_line_is_not_its_tail():
    text = ("The scheduler knob moves into the per-task config, with a migration.\n\n"
            "- [x] tested\n- `sched.max_inflight` replaces the global\n")

    body, reason = body_mod.clean_body(text, "Scheduler knob")

    assert reason is None, reason
    assert body.splitlines()[-1] == "- `sched.max_inflight` replaces the global", body
    assert "tested" not in body, body


@pytest.mark.parametrize("trailer", [
    "Fixes # (issue)", "fixes #(issue)", "Closes # (issue number)", "Resolves # ()",
    "Fixed # (issue)", "Fixes: # (issue)",
])
def test_an_unfilled_issue_trailer_is_in_neither_the_floor_nor_the_body(trailer):
    prose = "Moves the camera presets into the task config so each task can pick its own."
    body, reason = body_mod.clean_body(f"{prose}\n\n{trailer}\n", "Camera presets")
    assert reason is None, reason
    assert body == prose, body

    # Not in the floor's line set either: beside prose under the floor it adds nothing.
    short, why = body_mod.clean_body(f"tidy the launch file\n\n{trailer}\n", "Launch cleanup")
    assert short is None and why == "the upstream body is under 40 characters", (short, why)


@pytest.mark.parametrize("trailer", ["Fixes #8219", "Closes #12 (the camera one)",
                                     "Fixes # (issue 8219)"])
def test_a_filled_issue_trailer_is_kept_in_both(trailer):
    prose = "Moves the camera presets into the task config so each task can pick its own."
    body, reason = body_mod.clean_body(f"{prose}\n\n{trailer}\n", "Camera presets")
    assert reason is None, reason
    assert body.splitlines()[-1] == trailer, body

    # Counted towards the floor: 20 + a filled trailer of 27 is a body.
    counted, why = body_mod.clean_body(f"tidy the launch file\n\nCloses #12 (the camera one)\n",
                                       "Launch cleanup")
    assert why is None and "Closes #12" in counted, (counted, why)


def test_the_trailer_is_recognised_by_shape_and_the_phrase_list_is_unchanged():
    assert body_mod.TEMPLATE_PHRASES == (
        "thank you for your interest in sending a pull request",
        "please include a summary",
        "please make sure to check the contribution guidelines",
        "please try to keep prs small and focused",
    )


def test_the_docs_no_longer_promise_checkbox_lines_come_back():
    """`clean_body`'s docstring and the `_HEADING_LINE_RE` comment both said a checkbox
    line is returned byte-for-byte. Neither may say it now."""
    import inspect
    doc = inspect.getdoc(body_mod.clean_body)
    assert "byte-for-byte" not in doc, doc
    assert "checkbox" in doc                       # it says what happens to them instead
    src = Path(body_mod.__file__).read_text(encoding="utf-8")
    comment = src[src.index("#: The heading half of that class"):src.index("_HEADING_LINE_RE = re.compile")]
    assert "byte-for-byte" not in comment, comment
    assert "no longer published" in comment, comment


# ── #2011 — a `why` that rates the item against the interest profile is no body ──
#
# Stage 2's `why` answers "why does this match my interests". On 2026-10-01 TheAIGRID's
# description stripped to nothing and the entry's whole body became `Directly addresses
# the user's interest in AI/LLMs by covering a major new model release, despite the
# generic channel description.` — a sentence about the rubric. The same text reaches the
# digest two more ways: the empty `summary` stage 1 leaves on a YouTube row, and the
# `— <why>` suffix of an `Already noted:` line.

#: Every interest-profile sentence `knowledge/ai-llms/youtube-digest.md` held on
#: 2026-10-01, verbatim: bodies and `Already noted:` suffixes both.
PUBLISHED_PROFILE_SENTENCES = [
    "Directly addresses advanced AI reasoning techniques, which is core to the user's "
    "AI/LLM interest.",
    "The event features key players in LLMs (DeepMind, Hugging Face) and voice AI "
    "(ElevenLabs), directly matching the user's core interests.",
    "The video focuses on AI capabilities and NVIDIA hardware, which aligns with the "
    "user's interests in AI and hardware, though it lacks specific robotics or voice/TTS "
    "depth.",
    "Directly addresses AI/LLM advancements, specifically recursive self-improvement, "
    "which is core to the user's interest in AI-LLMs.",
    "Directly addresses robotics and AI advancements, aligning perfectly with the user's "
    "top-weighted interests.",
    "Directly addresses AI/LLM agent architecture and agentic coding workflows, which are "
    "core to the user's stated interests.",
    "Directly addresses AI-LLM agent workflows and software engineering automation, which "
    "aligns with the ai-llms interest, though it lacks specific relevance to robotics, "
    "voice, or hardware.",
    "The item discusses a major update to ChatGPT, which is directly relevant to the "
    "user's interest in AI and LLMs, though it lacks specific technical depth or "
    "hardware/robotics connections.",
    "The item focuses on AI metrics in software engineering workflows, which is "
    "tangentially related to AI interests but lacks specific relevance to robotics, "
    "voice, or hardware.",
    "Directly addresses the user's interest in AI/LLMs by covering a major new model "
    "release, despite the generic channel description.",
    # --- the 17 the item's own acceptance grep missed -------------------------
    #
    # Harvested the same day by running `is_interest_profile_prose` over every
    # prose line of the digest instead of grepping the file for the defect's
    # phrasing. None of these contains "the user's", "interest profile",
    # "tangential" or "lacks specific relevance" — the four patterns that grep
    # was built from — and every one names the interest profile anyway: "aligns
    # with the high-weight ai-llms interest", "top-weighted interests", "lacks
    # specific focus", "lacks relevance to robotics". That is how the first
    # sweep reported a clean digest at 0 hits with 17 rubric sentences still
    # published, so the corpus check is the classifier and not a pattern. The
    # comment above each entry is its line number in the file as harvested.
    # :840
    "It is a general AI news roundup covering LLMs and voice features, but lacks specific "
    "focus on robotics or hardware.",
    # :852
    "vLLM is a core library for LLM inference, directly relevant to the ai-llms interest.",
    # :866
    "Directly addresses AI applications in manufacturing and CAD, which intersects with "
    "hardware and robotics interests.",
    # :904
    "Highly relevant to AI-LLMs as it covers RAG and context management for agents, though "
    "it lacks direct robotics or hardware focus.",
    # :916
    "Directly addresses AI engineering and LLM tooling (Mistral, Langfuse), which aligns "
    "with the high-weight ai-llms interest.",
    # :928
    "Directly intersects robotics (giving AI a body) and AI/LLMs, aligning perfectly with "
    "top-weighted interests.",
    # :952
    "Directly addresses the intersection of AI/LLMs (VLM/VLA) and robotics (embodied "
    "agents), matching two core interests.",
    # :964
    "Focuses on AI agents and document intelligence, which aligns with AI/LLM interests "
    "but lacks direct relevance to robotics, voice, or hardware.",
    # :1014
    "Discusses advanced AI concepts (world models/causality) relevant to AI/LLM interests, "
    "but lacks specific connection to robotics, voice, or hardware.",
    # :1106
    "Directly addresses LLM identification and behavior analysis, which is core to the "
    "ai-llms interest, though it lacks the robotics or hardware components.",
    # :1120
    "Directly addresses AI-LLM capabilities in code generation, which is highly relevant "
    "to the ai-llms interest, though less so for robotics or hardware.",
    # :1148
    "Directly addresses the core interest in AI/LLMs by detailing a new open-weight "
    "model's performance and release strategy.",
    # :1162
    "Directly addresses AI agent architecture and LLM workflow optimization, which is core "
    "to the ai-llms interest, though it lacks specific robotics or hardware focus.",
    # :1190
    "Directly addresses AI agent orchestration and LLM workflow management, which is core "
    "to the ai-llms interest.",
    # :1295
    "Directly addresses the ai-llms interest through a specific multi-agent LLM "
    "architecture application.",
    # :1369
    "Directly addresses advanced LLM representation geometry, highly relevant to AI/LLM "
    "interests.",
    # :1397
    "Directly addresses advanced AI agent orchestration and multi-agent system "
    "architecture, which is core to the AI-LLMs interest.",
]

#: The 2026-10-01 row's stored `why`, by index rather than `[-1]`: the list above
#: grew by the 17 sentences the first sweep missed, and a `[-1]` here would silently
#: re-point this name at some other row's rubric sentence.
ARGON_WHY = PUBLISHED_PROFILE_SENTENCES[9]


@pytest.mark.parametrize("sentence", PUBLISHED_PROFILE_SENTENCES + [THEAIGRID_WHY, MANUS_WHY])
def test_every_published_interest_profile_sentence_is_classified_as_one(sentence):
    assert body_mod.is_interest_profile_prose(sentence) is True, sentence


@pytest.mark.parametrize("why", [
    "Scores 8/10: covers the agent-safety keyword set",
    "Scores 8/10: robotics",
    DESCRIPTIVE_WHY,
    "Isaac Lab adds a camera preset per task; the note walks through the config.",
    "An interesting comparison of two KV cache layouts under load.",
    "",
])
def test_a_why_that_describes_or_merely_scores_is_not_interest_profile_prose(why):
    assert body_mod.is_interest_profile_prose(why) is False, why


def test_an_interest_profile_why_is_never_the_body_by_either_route():
    """Route one: the recorded 2026-10-01 description, all channel boilerplate, strips
    to "". Route two: the empty `summary` stage 1 leaves on every YouTube row, which
    never reaches the strip at all. Both used to publish the `why`."""
    assert body_mod.strip_link_footer(THEAIGRID_GREETING) == ""
    stripped_to_nothing = _yt(summary=THEAIGRID_GREETING, why=ARGON_WHY)
    never_had_one = _yt(summary="", why=ARGON_WHY)

    for item in (stripped_to_nothing, never_had_one):
        body = vw_mod._entry_body(item)
        assert body == NO_BODY, body
        assert body.startswith("None — ")
        assert "Welcome to" not in body and "Subscribe" not in body
        assert not body_mod.is_interest_profile_prose(body)
    # The entry is still written: a named no-body line is a body to `lacks_body`.
    assert vw_mod.is_trailer_only(NO_BODY) is False


def test_a_descriptive_why_is_still_the_body_byte_for_byte():
    for why in ("Scores 8/10: robotics", DESCRIPTIVE_WHY):
        assert vw_mod._entry_body(_yt(summary="", why=why)) == why
        assert vw_mod._entry_body(_yt(summary=THEAIGRID_GREETING, why=why)) == why
    # And nothing here prefers the ad back: no `why` at all is the old placeholder.
    assert vw_mod._entry_body(_yt(summary=THEAIGRID_GREETING, why="")) == "(No description)"


def test_the_already_noted_line_drops_an_interest_profile_suffix_and_keeps_a_descriptive_one(
        tmp_path):
    note = tmp_path / "knowledge" / "youtube" / "Chan" / "20260921-a-video.md"
    note.parent.mkdir(parents=True)
    note.write_text("# a video\n", encoding="utf-8")
    digest = tmp_path / "knowledge" / "ai-llms" / "youtube-digest.md"
    standing = ("The YouTube channel monitor holds the full note for this video; this "
                "digest indexes it instead of restating it.")

    rated = vw_mod._note_pointer(_yt(why=PUBLISHED_PROFILE_SENTENCES[0]), note, digest)
    first, _, rest = rated.partition("\n\n")
    assert first.startswith("**Already noted:** [") and first.endswith(
        "](../youtube/Chan/20260921-a-video.md)"), first
    assert " — " not in first and "the user" not in rated, rated
    assert rest == standing

    described = vw_mod._note_pointer(_yt(why=DESCRIPTIVE_WHY), note, digest)
    assert described.partition("\n\n")[0].endswith(
        f"](../youtube/Chan/20260921-a-video.md) — {DESCRIPTIVE_WHY}"), described
    assert described.partition("\n\n")[2] == standing

# --- the corpus-level shape, pinned so a blind sweep cannot report clean again --

#: The pattern this item's acceptance clause proposed for verifying the corpus,
#: transcribed verbatim so the comparison below cannot drift from it.
ITEM_ACCEPTANCE_GREP = re.compile(
    r"the user'?s|interest profile|tangential|lacks (specific )?relevance", re.I)


def test_the_item_s_acceptance_grep_is_narrower_than_the_guard_it_verifies():
    """The blind spot, pinned as a number over the fixture and not over the vault.

    All 27 sentences in `PUBLISHED_PROFILE_SENTENCES` are flagged by the shipped
    guard; the hand-written pattern finds exactly 10 of them and misses exactly 17.
    Every miss names the interest profile without using any of the four phrases the
    pattern was built from — "aligns with the high-weight ai-llms interest", "top-
    weighted interests", "lacks specific focus", "lacks relevance to robotics" —
    which is how a sweep built from the defect's phrasing reported a clean digest
    while 17 rubric sentences were still published.

    The assertion is directional on purpose: the guard is a strict superset of the
    pattern, never the reverse. Write the check the other way round — filter the
    corpus with the pattern, then assert the guard agrees — and this node goes red.
    """
    assert len(PUBLISHED_PROFILE_SENTENCES) == 27, len(PUBLISHED_PROFILE_SENTENCES)
    caught = [s for s in PUBLISHED_PROFILE_SENTENCES if ITEM_ACCEPTANCE_GREP.search(s)]
    assert len(caught) == 10, (
        f"the acceptance pattern now catches {len(caught)} of 27; the blind spot "
        f"this node pins has changed shape, so update the count and the docstring")
    for sentence in PUBLISHED_PROFILE_SENTENCES:
        assert body_mod.is_interest_profile_prose(sentence) is True, sentence
    for sentence in caught:
        assert ITEM_ACCEPTANCE_GREP.search(sentence), sentence


def test_replaying_the_committed_witness_rows_publishes_no_interest_profile_prose(
        tmp_path):
    """Real stored feed rows through BOTH emission paths, not fixtures invented for it.

    `WITNESS_ROWS` is the four YouTube rows of the committed feed witness, embedded as
    constants so this stays a unit test with no read of the vault. All four stored
    `why` fields name the interest profile — that is what stage 2 answers when asked
    "why does this match my interests" — so every one of them was a candidate leak,
    and this is the node the first fix lacked: it was written against a single
    hand-picked fixture and so never showed that the corpus held 27 of these.

    Three of the four descriptions survive `strip_link_footer` as a publishable
    sentence, so their body is the description byte-for-byte (and the guard must not
    touch it — that is #1155's ruling); the fourth, the 2026-10-01 trigger, strips
    500 -> 0 and renders the named no-body form. Both ways, the rubric sentence stays
    out of the note, and no `Already noted:` line carries one as its suffix.
    """
    note = tmp_path / "knowledge" / "youtube" / "Chan" / "20261001-a-video.md"
    note.parent.mkdir(parents=True)
    note.write_text("# a video\n", encoding="utf-8")
    digest = tmp_path / "knowledge" / "ai-llms" / "youtube-digest.md"
    standing = ("The YouTube channel monitor holds the full note for this video; this "
                "digest indexes it instead of restating it.")

    for row in WITNESS_ROWS:
        assert body_mod.is_interest_profile_prose(row["why"]) is True, row["id"]
        item = _yt(id=row["id"], summary=row["summary"], why=row["why"],
                   title=row["title"])
        body = vw_mod._entry_body(item)
        assert body_mod.is_interest_profile_prose(body) is False, (
            f"{row['id']}: _entry_body published the rubric — {body}")
        assert "Welcome to" not in body and "Subscribe" not in body, (
            f"{row['id']}: the channel ad came back instead of the guard — {body}")

        pointer = vw_mod._note_pointer(item, note, digest)
        first, _, rest = pointer.partition("\n\n")
        assert first.startswith("**Already noted:** [") and first.endswith(
            "](../youtube/Chan/20261001-a-video.md)"), f"{row['id']}: {first}"
        assert " — " not in first, (
            f"{row['id']}: the Already-noted line kept a ` — <why>` suffix, and every "
            f"stored why here is a rating: {first}")
        assert rest == standing, f"{row['id']}: lost the standing sentence: {rest}"

    # The 500 -> 0 trigger row is the one route the guard exists for, and the three
    # rows with a real description must keep it, unedited: a guard that replaced
    # those with `None — …` would be quietly reverting #1155's fallback.
    trigger, *kept = WITNESS_ROWS[2:] + WITNESS_ROWS[:2]
    assert vw_mod._entry_body(
        _yt(id=trigger["id"], summary=trigger["summary"], why=trigger["why"],
            title=trigger["title"])) == NO_BODY
    for row in kept:
        assert vw_mod._entry_body(
            _yt(id=row["id"], summary=row["summary"], why=row["why"],
                title=row["title"])).startswith(row["summary"][:60]), row["id"]


WITNESS_ROWS = [
    {
        "id": "youtube:UC0C-17n9iuUQPylguM1d-lQ:otMlZycCBZ4",
        "title": "Meta's Muse can get your money back #AI #meta #muse #agent",
        "why":
        "The item focuses on an AI agent for consumer tasks like billing and "
        "subscriptions, which does not align with the user's specific interests "
        "in robotics, LLMs, voice tech, or hardware.",
        "summary":
        "Early Muse users are handing it the jobs nobody wants. One user, Chris "
        "Abraham, says it found a book subscription his family forgot to cancel, "
        "cancelled it, checked the refund policy, and recovered a year of "
        "payments. Another, Brad, says it spent 98 minutes on the phone with AT&T"
        " and compared offers from Verizon and T-Mobile, saving him more than "
        "three hours of calls. The bills, the subscriptions, and the charges you "
        "never had time to fight were problems long before AI. Now something can\u2026",
    },
    {
        "id": "youtube:UCajiMK_CY9icRhLepS8_3ug:A5pLkQBQ8EQ",
        "title": "The First Thing AI Found Wasn\u2019t an Overcharge",
        "why":
        "The item focuses on AI agent capabilities and privacy handling, which "
        "aligns with AI/LLM interests but lacks relevance to robotics, voice, or "
        "hardware.",
        "summary":
        "I asked Perplexity Computer to compare six months of bank statements and"
        " internet bills against a service agreement, then look for better "
        "internet plans. When its Privacy Gate flagged personal information, I "
        "chose to process the files on my Mac.\n\nThe financial documents are "
        "sample data with made-up names and account numbers. The task shown in "
        "the video is a real run.",
    },
    {
        "id": "youtube:UCbY9xX3_jW5c2fjlZVBI4cg:FfAYjDA35gY",
        "title": "Googles New Gemini 4 Argon is Now The Worlds Smartest AI",
        "why":
        "Directly addresses the user's interest in AI/LLMs by covering a major "
        "new model release, despite the generic channel description.",
        "summary":
        "Welcome to TheAIGRID \u2014 the place to learn AI for free. I create simple, "
        "practical videos that help beginners, creators, entrepreneurs, and "
        "business owners understand artificial intelligence, AI tools, "
        "automation, AI agents, robotics, ChatGPT, Claude, Gemini, and the future"
        " of technology. Whether you want AI tutorials, tool breakdowns, beginner"
        " guides, or explanations of the latest breakthroughs, this channel gives"
        " you the knowledge you need to stay ahead. Subscribe to start learning "
        "AI for free\u2026",
    },
    {
        "id": "youtube:UC5l7RouTQ60oUjLjt1Nh-UQ:y12lBg43rE0",
        "title": "OpenAI Just Dropped Its Biggest Agent Upgrade Yet",
        "why":
        "Directly addresses AI-LLMs and agent capabilities, which are core "
        "interests, though it lacks specific hardware or robotics focus.",
        "summary":
        "OpenAI just turned ChatGPT into something much closer to a digital "
        "worker with Dots, an always-on AI agent with its own computer and "
        "browser. Meanwhile GPT-6.1 Astra gets pulled over deception concerns, "
        "while GPT-6.1 Sol gets close to Astra at a fraction of the price.",
    },
]
