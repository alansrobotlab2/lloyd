"""`bless` pins the commit whose CODE is running, not the sha that matches.

The guard existed because only `/health.commit` proves the service — a working
tree can be anywhere. But it compared shas, and what it is *for* is narrower:
`app/gitinfo.py` and `app/routers/health.py` both state the property as "only
`/health.commit == last_known_good` proves **the service** changed".

`arch-review` made the difference routine. It commits one `architecture/*.md`
per run, up to `daily_max` times a day, so HEAD sits documentation-ahead of the
served commit most of the time — and a sha-equality guard made `bless`
unreachable without a backend restart whose only purpose was to load a markdown
file nothing reads at runtime.

The carve-out is an allowlist of inert paths and fails closed, because it
guards a rollback target: a diff that cannot be read, or that moves one `.py`,
refuses exactly as before.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.automod import round as R


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    r = tmp_path / "lloyd"
    (r / "architecture").mkdir(parents=True)
    (r / "app").mkdir()
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "l@l")
    git(r, "config", "user.name", "lloyd")
    (r / "app" / "m.py").write_text("V = 1\n")
    (r / "architecture" / "workers.md").write_text("# Workers\n\nBody.\n")
    (r / "CLAUDE.md").write_text("# Context\n")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "base")
    monkeypatch.setattr(R, "LIVE_ROOT", r)
    return r


def _head(r: Path) -> str:
    return git(r, "rev-parse", "HEAD").strip()


def _commit(r: Path, rel: str, text: str, msg: str) -> str:
    (r / rel).parent.mkdir(parents=True, exist_ok=True)
    (r / rel).write_text(text)
    git(r, "add", "-A")
    git(r, "commit", "-qm", msg)
    return _head(r)


# ── the predicate ────────────────────────────────────────────────────────────


def test_a_documentation_only_delta_leaves_the_served_code_equal(repo):
    served = _head(repo)
    head = _commit(repo, "architecture/workers.md", "# Workers\n\nCorrected.\n", "doc")
    same, differing = R._served_code_is_head(served, head)
    assert same is True
    assert differing == ["architecture/workers.md"]


def test_a_markdown_delta_anywhere_counts_as_documentation(repo):
    """`arch-review` writes under `architecture/`, but CLAUDE.md, SETUP.md and
    the per-directory notes are the same kind of file and the same non-effect
    on the running process."""
    served = _head(repo)
    _commit(repo, "CLAUDE.md", "# Context\n\nMore.\n", "claude")
    head = _commit(repo, "architecture/workers.md", "# Workers\n\nx\n", "doc")
    same, _ = R._served_code_is_head(served, head)
    assert same is True


def test_one_code_path_makes_the_whole_delta_code(repo):
    served = _head(repo)
    _commit(repo, "architecture/workers.md", "# Workers\n\nx\n", "doc")
    head = _commit(repo, "app/m.py", "V = 2\n", "code")
    same, differing = R._served_code_is_head(served, head)
    assert same is False
    assert differing == ["app/m.py"], "and it names the code path, not the doc"


@pytest.mark.parametrize("rel,text", [
    ("app/m.py", "V = 3\n"),
    ("config.yaml", "a: 1\n"),
    ("workers/sources/x.py", "NAME = 'x'\n"),
    ("tests/test_x.py", "def test_x(): pass\n"),
])
def test_anything_not_markdown_is_code_until_proven_otherwise(repo, rel, text):
    """An allowlist, never a denylist: a file type nobody thought about is
    code, so a new one can never slip through as inert."""
    served = _head(repo)
    head = _commit(repo, rel, text, "change")
    same, _ = R._served_code_is_head(served, head)
    assert same is False


def test_an_unreadable_diff_fails_closed(repo):
    """It guards a rollback target. A served commit that has been
    garbage-collected, or git unavailable, must refuse rather than assume."""
    same, differing = R._served_code_is_head("0" * 40, _head(repo))
    assert same is False and differing == []


def test_identical_commits_are_equal(repo):
    assert R._served_code_is_head(_head(repo), _head(repo)) == (True, [])


# ── bless itself ─────────────────────────────────────────────────────────────


@pytest.fixture
def blessable(repo, monkeypatch, tmp_path):
    """`bless` with its two collaborators stubbed: the health probe and the
    automod state it writes."""
    written = {}
    monkeypatch.setattr(R.S, "write_lkg", lambda c: written.setdefault("lkg", c) or {"commit": c})
    monkeypatch.setattr(R.S, "read_current", lambda: None)
    events = []
    monkeypatch.setattr(R.S, "append_event", lambda e, **kw: events.append(e))

    def serve(commit):
        import scripts.automod.promote as P
        monkeypatch.setattr(P, "_get", lambda url, timeout=5.0: (200, {"commit": commit}))
    return {"serve": serve, "written": written, "events": events}


def test_bless_pins_head_when_only_docs_are_ahead(repo, blessable):
    served = _head(repo)
    head = _commit(repo, "architecture/workers.md", "# Workers\n\nCorrected.\n", "doc")
    blessable["serve"](served)

    out = R.bless("after arch-review")
    assert out["docs_only_ahead"] is True
    assert blessable["written"]["lkg"] == head, (
        "HEAD, not the served sha: rolling back to HEAD restores the running "
        "code AND keeps the doc, where the older sha would discard it")
    ev = blessable["events"][-1]
    assert ev["commit"] == head and ev["served_commit"] == served
    assert ev["docs_only_ahead"] is True
    assert "documentation-only ahead of served" in ev["note"], (
        "the gap is recorded, not silent — it is the whole justification")


def test_bless_still_refuses_when_code_is_ahead(repo, blessable):
    served = _head(repo)
    _commit(repo, "app/m.py", "V = 2\n", "code")
    blessable["serve"](served)

    with pytest.raises(RuntimeError) as exc:
        R.bless()
    assert "app/m.py" in str(exc.value), "it names what actually differs"
    assert "restart the backend before blessing" in str(exc.value)
    assert "lkg" not in blessable["written"]


def test_bless_is_unchanged_when_the_shas_match(repo, blessable):
    head = _head(repo)
    blessable["serve"](head)
    out = R.bless()
    assert out["docs_only_ahead"] is False
    assert blessable["written"]["lkg"] == head
    assert blessable["events"][-1]["docs_only_ahead"] is False


def test_bless_still_refuses_over_a_promotion_under_observation(repo, blessable, monkeypatch):
    """Unchanged, and checked after the code comparison: a settling promotion
    is a different objection and outranks a clean tree."""
    served = _head(repo)
    head = _commit(repo, "architecture/workers.md", "# W\n\nx\n", "doc")
    blessable["serve"](served)
    monkeypatch.setattr(R.S, "read_current", lambda: {"round_id": "SM_x", "commit": head})
    with pytest.raises(RuntimeError, match="under observation"):
        R.bless()
    assert "lkg" not in blessable["written"]
