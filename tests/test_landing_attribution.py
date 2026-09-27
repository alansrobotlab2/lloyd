"""A landing credits Alan (author) and Lloyd (the one co-author), nobody else.

#1238's landing (adfd8022, 2026-09-19) carried a Co-Authored-By line the model
invented — `Lloyding <69833984+Lloyding@users.noreply.github.com>` — and GitHub
resolved the number to a real account, which then showed as a contributor to
the public repo. #1241's single-commit landing carried another
(`lllydeon@proton.me`). The squash copied the first; the second was never
squashed at all. These pin that no path onto `main` carries a model-written
credit line, and that rewording one never changes what was gated.
"""
from __future__ import annotations

import subprocess

import pytest

from scripts.automod import promote as P, worktree as W

LLOYD = "Co-Authored-By: Lloyd <lloyd@local>"
INVENTED = "Co-authored-by: Lloyding <69833984+Lloyding@users.noreply.github.com>"


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "r"
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com"); git(r, "config", "user.name", "t")
    (r / "a.py").write_text("A = 1\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "base")
    base = git(r, "rev-parse", "HEAD").stdout.strip()
    git(r, "checkout", "-q", "-b", "automod/SM_X")
    return r, base


def commit(r, msg, *, name="lloyd", email="lloyd@local", n=[0]):
    n[0] += 1
    (r / f"f{n[0]}.txt").write_text(f"{n[0]}\n")
    git(r, "add", "-A")
    git(r, "-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "-q", "-m", msg)
    return git(r, "rev-parse", "HEAD").stdout.strip()


def body(r, rev="HEAD"):
    return git(r, "log", "-1", "--format=%B", rev).stdout


def test_attributed_drops_every_credit_line_and_adds_lloyds():
    msg = ("fix the thing\n\nWhy it broke.\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n"
           f"{INVENTED}\nco-authored-by:lllydeon <lllydeon@proton.me>\n")
    out = P.attributed(msg, "Lloyd <lloyd@local>")
    assert out == f"fix the thing\n\nWhy it broke.\n\n{LLOYD}\n"


def test_prose_that_mentions_the_trailer_is_not_a_credit_and_stays():
    msg = "trailers\n\nCo-authored-by lines render through clean_body as the subject.\n"
    assert "Co-authored-by lines render" in P.attributed(msg, "Lloyd <lloyd@local>")


def test_a_single_commit_round_is_reworded_with_the_same_tree_and_author(repo):
    r, base = repo
    old = commit(r, f"feat(certs): #1241\n\nbody\n\n{INVENTED}\n")
    old_tree = git(r, "rev-parse", "HEAD^{tree}").stdout.strip()
    new, note = W.reword_onto(r, base, P.attributed, keep_ref="refs/automod/rounds/SM_X")
    assert new and new != old and "reworded 1 of 1" in note
    assert W.head(r) == new
    assert git(r, "rev-parse", "HEAD^{tree}").stdout.strip() == old_tree, "what was gated is what lands"
    assert git(r, "rev-parse", "HEAD^").stdout.strip() == base
    assert git(r, "log", "-1", "--format=%an <%ae>").stdout.strip() == "lloyd <lloyd@local>"
    assert "69833984" not in body(r) and body(r).rstrip().endswith(LLOYD)
    assert git(r, "rev-parse", "refs/automod/rounds/SM_X").stdout.strip() == old
    assert P.foreign_coauthors(r, base, "Lloyd <lloyd@local>") == []


def test_every_commit_is_reworded_when_squash_is_off_and_clean_ones_keep_their_sha(repo):
    r, base = repo
    clean = commit(r, f"first\n\n{LLOYD}\n")
    commit(r, f"second\n\n{INVENTED}\n")
    commit(r, "third\n")
    old_tree = git(r, "rev-parse", "HEAD^{tree}").stdout.strip()
    new, note = W.reword_onto(r, base, lambda m: P.attributed(m, "Lloyd <lloyd@local>"))
    assert new and "reworded 2 of 3" in note
    assert git(r, "rev-parse", "HEAD~2").stdout.strip() == clean, "an already-right prefix is untouched"
    assert git(r, "rev-parse", "HEAD^{tree}").stdout.strip() == old_tree
    for rev in ("HEAD", "HEAD~1", "HEAD~2"):
        assert body(r, rev).lower().count("co-authored-by") == 1 and body(r, rev).rstrip().endswith(LLOYD)


def test_a_branch_already_right_is_left_exactly_as_gated(repo):
    r, base = repo
    old = commit(r, f"one\n\n{LLOYD}\n")
    new, note = W.reword_onto(r, base, lambda m: P.attributed(m, "Lloyd <lloyd@local>"))
    assert new is None and "already" in note and W.head(r) == old


def test_foreign_coauthors_names_what_would_reach_main(repo):
    r, base = repo
    commit(r, f"one\n\n{LLOYD}\n{INVENTED}\n")
    assert P.foreign_coauthors(r, base, "Lloyd <lloyd@local>") == [" ".join(INVENTED.split())]


def test_settle_attribution_refuses_a_landing_it_could_not_clean(repo, monkeypatch):
    r, base = repo
    commit(r, f"one\n\n{INVENTED}\n")
    monkeypatch.setattr(W, "reword_onto", lambda *a, **k: (None, "refused for the test"))
    events = []
    monkeypatch.setattr(P.S, "append_event", events.append)
    with pytest.raises(P.PromoteError, match="Lloyding"):
        P.settle_attribution("SM_X", r, base)
    assert events and events[0]["event"] == "land_failed"


def test_the_configured_coauthor_is_used_and_a_malformed_one_falls_back(monkeypatch):
    monkeypatch.setattr(P.S, "landing_cfg", lambda *_: {"coauthor": "Lloyd <1+lloyd@users.noreply.github.com>"})
    assert P.landing_coauthor() == "Lloyd <1+lloyd@users.noreply.github.com>"
    monkeypatch.setattr(P.S, "landing_cfg", lambda *_: {"coauthor": "no address here"})
    assert P.landing_coauthor() == P.DEFAULT_COAUTHOR
    monkeypatch.setattr(P.S, "landing_cfg", lambda *_: {})
    assert P.landing_coauthor() == "Lloyd <lloyd@local>"


def test_the_shipped_config_names_lloyd():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
    assert cfg["automod"]["landing"]["coauthor"] == "Lloyd <lloyd@local>"


def test_settle_attribution_authors_a_single_commit_round_as_the_repos_identity(repo):
    """#1577 and #1543 landed on 2026-09-26 authored `Lloyd <lloyd@localhost>`
    after the credit fix: their messages were already right, so nothing was
    rewritten, and a single-commit round skips the squash that commits as the
    repo. The repo's configured identity (`t <t@e.com>` here) is the author."""
    r, base = repo
    commit(r, f"one\n\n{LLOYD}\n", name="Lloyd", email="lloyd@localhost")
    old_tree = git(r, "rev-parse", "HEAD^{tree}").stdout.strip()
    new, note = P.settle_attribution("SM_X", r, base)
    assert new and "reworded 1 of 1" in note
    assert git(r, "log", "-1", "--format=%an <%ae>").stdout.strip() == "t <t@e.com>"
    assert git(r, "rev-parse", "HEAD^{tree}").stdout.strip() == old_tree
    assert body(r).rstrip().endswith(LLOYD)


def test_settle_attribution_leaves_a_branch_already_authored_right(repo):
    r, base = repo
    old = commit(r, f"one\n\n{LLOYD}\n", name="t", email="t@e.com")
    new, note = P.settle_attribution("SM_X", r, base)
    assert new is None and "already" in note and W.head(r) == old
