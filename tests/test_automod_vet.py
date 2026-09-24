"""`scripts/automod/vet.py` — the deterministic change-set vet (backlog #679).

Four clauses, four groups:

  1. `empty_file` is base-relative, so the zero-byte files `main` already
     carries (seven legitimate `__init__.py` plus
     `scripts/intel-pipeline/config/interest-profile.yml`, empty since the
     initial snapshot) cannot produce a finding.
  2. `binary_artifact` fires on a newly-added NUL-bearing path and stays quiet
     inside the trees that hold binaries today (five tracked `.onnx` models,
     `web/public/**`, `tests/fixtures/**`).
  3. `diff_too_large` compares total changed lines against the ceiling read as
     `automod.gate.max_diff_lines`, whose default lives in code.
  4. A git read that fails is `status="unevaluated"`, never an empty violation
     list, so "could not check" is never reported as "clean".

Every test here runs against a scratch git repo built by the fixture below —
`main` is never mutated, and the false-positive controls use the same relative
paths the real tree does so a change to the allowlist has to be a decision.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.automod import vet

# The five `.onnx` models tracked on `main` as of this round, spelled as they
# are on disk. Used as the false-positive control: if one of these ever starts
# to be flagged, the allowlist and the repo have drifted apart.
TRACKED_ONNX = (
    "agent-services/models/openwakeword/embedding_model.onnx",
    "agent-services/models/openwakeword/melspectrogram.onnx",
    "agent-services/models/silero-vad/silero_vad.onnx",
    "agent-services/models/wakeword/Lloyd.onnx",
    "agent-services/models/wakeword/hey_lloyd.onnx",
)

#: Zero-byte shapes planted in the scratch repo, drawn from the paths the real
#: tree carries (eight `__init__.py` and one empty YAML config — the exact
#: number moves as packages are added, so the live assertion below *discovers*
#: them instead of pinning a count). A check phrased as "is this file empty?"
#: would fire on every one of them; the base-relative check must fire on none.
ZERO_BYTE_PATHS = (
    "app/__init__.py",
    "app/routers/__init__.py",
    "app/harness/tests/__init__.py",
    "agent_mcp/__init__.py",
    "eval/djev/__init__.py",
    "scripts/meta_review/__init__.py",
    "scripts/intel-pipeline/intel_pipeline/__init__.py",
    "scripts/intel-pipeline/config/interest-profile.yml",
)

#: The only shapes a zero-byte tracked file may have on the live tree. A new
#: empty file of some other shape is exactly what the empty check exists to
#: catch, so this is a ceiling on the repo, not on the test.
LIVE_ZERO_BYTE_OK_SUFFIXES = ("__init__.py",)
LIVE_ZERO_BYTE_OK_EXACT = ("scripts/intel-pipeline/config/interest-profile.yml",)

PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00"
WAV = b"RIFF\x00\x00\x00\x00WAVEfmt \x00\x00\x00\x00\x00"

#: The one non-empty file in the fixture's base commit: what gets emptied.
SERVICE_SRC = "def handle(event):\n    return event\n"


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=False)


def write(repo: Path, rel: str, content) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    data = content if isinstance(content, bytes) else str(content).encode("utf-8")
    p.write_bytes(data)


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    r = git(repo, "commit", "-q", "-m", message)
    assert r.returncode == 0, r.stdout + r.stderr
    return git(repo, "rev-parse", "HEAD").stdout.strip()


class ScratchRepo:
    """A scratch git tree plus the sha every test diffs from.

    Not a `Path` subclass: `pathlib` objects carry `__slots__`, so
    `path.base_sha = sha` raises `AttributeError` — and a fixture that dies in
    `setup` looks exactly like a broken module import.
    """

    def __init__(self, root: Path, base_sha: str):
        self.root = root
        self.base_sha = base_sha

    def write(self, rel: str, content) -> None:
        write(self.root, rel, content)

    def remove(self, rel: str) -> None:
        (self.root / rel).unlink()

    def chmod(self, rel: str, mode: int) -> None:
        (self.root / rel).chmod(mode)

    def config(self, key: str, value: str) -> None:
        assert git(self.root, "config", key, value).returncode == 0

    def commit(self, message: str) -> str:
        return commit_all(self.root, message)

    def vet(self, base: str | None = None):
        return vet.vet_change_set(base or self.base_sha, self.root)


#: The base commit's content for `app/service.py`: what gets emptied.
SERVICE_SRC = "def handle(event):\n    return event\n"


@pytest.fixture()
def repo(tmp_path):
    """A scratch repo whose base holds one real file and eight zero-byte ones.

    HEAD then adds five more things that must all score zero: three of the
    zero-byte shapes as NEW files, a mode-only (`chmod +x`) change to two that
    already existed at base, and an ordinary new text file. `base_sha` is the
    commit every test diffs from.
    """
    root = tmp_path / "wt"
    root.mkdir()
    git(tmp_path, "init", "-q", "-b", "main", str(root))
    git(root, "config", "user.email", "vet@example.invalid")
    git(root, "config", "user.name", "vet")
    write(root, "app/service.py", SERVICE_SRC)
    for rel in ZERO_BYTE_PATHS:
        write(root, rel, "")
    base_sha = commit_all(root, "base")

    # Everything in this commit must be invisible to the vet.
    for rel in ("newpkg/__init__.py", "newpkg/sub/__init__.py",
                "eval/fixtures/blank.json"):
        write(root, rel, "")
    for rel in ("app/__init__.py", "scripts/intel-pipeline/config/interest-profile.yml"):
        (root / rel).chmod(0o755)   # mode-only: in the diff, 0 bytes at both ends
    write(root, "app/new_text.py", "X = 1\n")
    commit_all(root, "head: new empty files, two mode changes, one text file")
    return ScratchRepo(root, base_sha)



# ---------------------------------------------------------------------------
# Clause 1 — empty files, relative to base
# ---------------------------------------------------------------------------

def test_the_shapes_main_already_carries_score_zero(repo):
    """Eight zero-byte tracked files, plus three newly-added empty files, plus
    two mode-only changes to files that were already empty: zero findings.

    Each of the three shapes is a way a naive check would cry wolf. A file that
    is empty at base AND empty at HEAD only reaches the diff through a mode
    change, which is why the fixture bothers to `chmod` them — "did the diff
    touch this file" and "did its bytes change" are not the same question.
    """
    res = repo.vet()
    assert res.evaluated, res.reason
    assert res.violations == [], [v.label() for v in res.violations]
    assert res.totals["files"] == 6      # 3 added empty + 2 mode changes + 1 added text


def test_a_file_emptied_since_base_is_reported_once_and_by_name(repo):
    repo.write("app/service.py", "")
    repo.commit("clobber: emptied the handler")

    res = repo.vet()
    empties = [v for v in res.violations if v.kind == vet.EMPTY_FILE]
    assert [v.path for v in empties] == ["app/service.py"]
    # The whole list, so the eight base-empty shapes and the three newly-added
    # empty files are excluded by this same assertion, not by luck.
    assert [v.label() for v in res.violations] == ["empty_file:app/service.py"]
    # The finding carries the size it was at base, so a reader can tell a
    # 400 KB clobber from a 30-byte one without opening git.
    assert f"non-empty ({len(SERVICE_SRC)} B) at base" in empties[0].detail
    assert "0 bytes at HEAD" in empties[0].detail


def test_a_file_emitted_empty_by_a_rename_is_not_an_emptied_file(repo):
    """A deleted path has no HEAD blob, so the check does not see it at all."""
    repo.remove("app/service.py")
    repo.commit("moved away, not emptied")

    res = repo.vet()
    assert [v.kind for v in res.violations if v.kind == vet.EMPTY_FILE] == []
    assert res.evaluated


# ---------------------------------------------------------------------------
# Clause 2 — stray binary artifacts
# ---------------------------------------------------------------------------

def test_a_new_binary_outside_the_allowlist_is_named_and_the_rest_are_not(repo):
    """One planted `.bin` under `app/` is the only finding among eight new
    binaries, because five of them are the paths `main` tracks today."""
    repo.write("app/blob.bin", b"\x7fELF\x02\x01\x01\x00\x00\x00\x00")
    for rel in TRACKED_ONNX:
        repo.write(rel, b"\x08\x00\x00onnx-model-bytes\x00\x01")
    repo.write("web/public/apple-touch-icon.png", PNG)
    repo.write("web/public/lloyd.jpg", b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01")
    repo.write("tests/fixtures/voice/hey_lloyd_16k.wav", WAV)
    repo.write("app/migrations/0001_add_users.sql", "CREATE TABLE users (id int);\n")
    repo.commit("planted binary alongside the repo's real binary trees")

    res = repo.vet()
    bins = [v for v in res.violations if v.kind == vet.BINARY_ARTIFACT]
    assert [v.path for v in bins] == ["app/blob.bin"]
    assert res.violations == bins, [v.label() for v in res.violations]
    assert "NUL" in bins[0].detail


def test_a_new_text_file_is_not_a_binary(repo):
    """The NUL test, not an extension test: `.bin` with no NUL is prose."""
    repo.write("app/payload.bin", "print('still text')\n")
    repo.commit("misleading name, no NUL")

    res = repo.vet()
    assert res.violations == [], [v.label() for v in res.violations]


def test_a_moved_binary_is_named_and_rename_detection_is_not_what_says_so(repo):
    """A `git mv` of an allowlisted model into `app/` is an ADDITION — and `--no-renames` is the flag that makes that deterministic.

    The allowlist is path-scoped, so a model that moves into a source tree has
    arrived somewhere it was never allowed. With rename detection on, git
    collapses the move into one `R100 old new` line, which is neither an add nor
    a delete: the same committed change would be a finding in one environment and
    silence in another, depending on a `diff.renames` setting nobody is thinking
    about at gate time, and on whether the moved content happens to pair with
    something else that vanished in the same commit. The flag fixes it to two
    lines with fixed letters, so the `status == "A"` branch is the same code path
    for `git mv`, for `cp models/x.onnx app/ && git add`, and for a fresh
    download — three ways the blob reaches `app/`, only one of which git calls a
    rename. Asserted against this git, so the test fails if the flag is dropped.
    """
    repo.write("agent-services/models/silero-vad/silero_vad.onnx", b"ONNX\x00model\x00")
    start = repo.commit("the model lands (allowlisted)")
    assert repo.vet(start).violations == []

    git(repo.root, "mv", "agent-services/models/silero-vad/silero_vad.onnx",
        "app/blob.onnx")
    head = repo.commit("git mv: the model moves into app/")

    spec = f"{start}...{head}"
    enumerated = vet._name_status(repo.root, spec)        # what the check calls
    assert ("A", "app/blob.onnx") in enumerated, enumerated
    assert ("D", "agent-services/models/silero-vad/silero_vad.onnx") in enumerated
    assert not [st for st, _p in enumerated if st.startswith("R")], enumerated

    repo.config("diff.renames", "true")
    unflagged = git(repo.root, "diff", "--name-status", spec).stdout.splitlines()
    assert any(ln.startswith("R") for ln in unflagged), (
        "this git does not report the move as a rename at all, so the fixture no "
        "longer exercises what the flag suppresses")

    res = repo.vet(start)
    assert [v.label() for v in res.violations] == ["binary_artifact:app/blob.onnx"], \
        res.violations


def test_a_binary_replaced_in_place_is_not_reported_as_added(repo):
    """The check keys on status `A`, not on "is there a NUL in the tree".

    A retrained model landing over its own path is the repo's normal business —
    five `.onnx` files are tracked precisely so it can happen — and the defect
    this check names is a blob appearing where no blob was. The fixture's base
    has no models at all, so the test lands them first and diffs from there.
    """
    for rel in TRACKED_ONNX:
        repo.write(rel, b"\x08\x00onnx-model-bytes\x00\x01")
    with_models = repo.commit("the models land (allowlisted additions)")
    assert repo.vet().violations == []

    for rel in TRACKED_ONNX:
        repo.write(rel, b"\x08\x00retrained-model-bytes\x00\x02")
    repo.commit("the models are replaced in place")

    res = repo.vet(with_models)
    assert res.violations == [], [v.label() for v in res.violations]


# ---------------------------------------------------------------------------
# Clause 3 — the changed-line ceiling
# ---------------------------------------------------------------------------

def _ceiling(monkeypatch, value):
    """Set `automod.gate.max_diff_lines` the way config.yaml would."""
    import app.config
    monkeypatch.setattr(app.config, "CONFIG",
                        {"automod": {"gate": {"max_diff_lines": value}}})


def test_over_the_ceiling_is_a_violation_and_at_it_is_not(repo, monkeypatch):
    """One line over the ceiling is a finding; exactly the ceiling is not.

    The boundary is taken from the instrument's own total rather than written
    as a literal: the diff here is cumulative from the fixture's base commit, so
    the fixture's own head commit contributes to it, and a hard-coded 40 would
    be an assertion about the fixture, not about the check.
    """
    repo.write("app/bulk.py", "\n".join(f"X{i} = {i}" for i in range(40)) + "\n")
    repo.commit("40 new lines")

    total = repo.vet().totals["changed_lines"]
    assert total >= 40, total          # 40 here, plus the fixture's own head commit

    _ceiling(monkeypatch, total - 1)
    res = repo.vet()
    big = [v for v in res.violations if v.kind == vet.DIFF_TOO_LARGE]
    assert [v.label() for v in res.violations] == ["diff_too_large"], res.violations
    assert big[0].path == "", "the size finding is about the change set, not a path"
    assert f"{total} changed lines over the ceiling of {total - 1}" in big[0].detail
    assert res.totals["changed_lines"] == total
    assert res.totals["max_diff_lines"] == total - 1

    # At the ceiling: `>`, not `>=`.
    _ceiling(monkeypatch, total)
    assert repo.vet().violations == []
    _ceiling(monkeypatch, total + 1000)
    assert repo.vet().violations == []


def test_the_ceiling_default_is_in_code_so_no_config_edit_is_needed(repo, monkeypatch):
    """`config.yaml` is denied to the loop, so a check that only exists when
    config says so is a check that is off forever. Read the default with no
    `automod` key at all, and then prove the default actually bites."""
    import app.config
    monkeypatch.setattr(app.config, "CONFIG", {})
    assert vet.max_diff_lines() == vet.DEFAULT_MAX_DIFF_LINES

    over = vet.DEFAULT_MAX_DIFF_LINES + 5
    repo.write("app/flood.py", "\n".join(f"Y{i} = {i}" for i in range(over)) + "\n")
    repo.commit(f"{over} new lines: past the code default")

    res = repo.vet()
    assert [v.kind for v in res.violations] == [vet.DIFF_TOO_LARGE]
    assert res.totals["max_diff_lines"] == vet.DEFAULT_MAX_DIFF_LINES


def test_a_garbage_ceiling_falls_back_to_the_code_default(monkeypatch):
    import app.config
    monkeypatch.setattr(app.config, "CONFIG",
                        {"automod": {"gate": {"max_diff_lines": "lots"}}})
    assert vet.max_diff_lines() == vet.DEFAULT_MAX_DIFF_LINES
    monkeypatch.setattr(app.config, "CONFIG",
                        {"automod": {"gate": {"max_diff_lines": 0}}})
    assert vet.max_diff_lines() == vet.DEFAULT_MAX_DIFF_LINES


# ---------------------------------------------------------------------------
# Clause 4 — a check that could not run is never reported as clean
# ---------------------------------------------------------------------------

def test_an_unreadable_base_ref_is_unevaluated_not_clean(repo, tmp_path):
    res = vet.vet_change_set("0" * 40, repo.root)
    assert not res.evaluated
    assert res.status == vet.UNEVALUATED
    assert res.violations == []          # empty, but the status says why
    assert res.reason and "merge-base" in res.reason
    assert res.to_dict()["status"] == "unevaluated"


def test_a_missing_worktree_is_unevaluated(tmp_path):
    res = vet.vet_change_set("HEAD", tmp_path / "no-such-tree")
    assert res.status == vet.UNEVALUATED
    assert "no worktree" in res.reason


@pytest.mark.parametrize("reader", [
    "_name_status", "_changed_lines", "_paths_and_sizes", "_has_nul",
])
def test_each_git_read_the_vet_depends_on_fails_to_unevaluated(repo, monkeypatch, reader):
    """Four reads, four ways to be blind, and none of them may report clean.

    The `_has_nul` case is the sharp one: the fixture's HEAD adds a text file,
    so the sniff runs, and a failed read there is exactly the shape of "I found
    no NUL byte" if it is written as a bare `except: return False`.
    """
    monkeypatch.setattr(vet, reader, lambda *a, **k: None)
    res = repo.vet()
    assert res.status == vet.UNEVALUATED, f"{reader} failure read as clean"
    assert res.violations == []
    assert res.reason


def test_a_diff_entry_missing_from_heads_tree_is_unevaluated(repo, monkeypatch):
    """The tree moved while the vet was reading it: inconsistent inputs, so no
    verdict — not an empty list that reads as "nothing wrong here"."""
    real = vet._paths_and_sizes

    def heads_are_missing_them(root, ref):
        sizes = real(root, ref)
        return {p: s for p, s in sizes.items() if p != "app/new_text.py"} \
            if sizes is not None and ref == "HEAD" else sizes

    monkeypatch.setattr(vet, "_paths_and_sizes", heads_are_missing_them)
    res = repo.vet()
    assert res.status == vet.UNEVALUATED
    assert "app/new_text.py" in res.reason


# ---------------------------------------------------------------------------
# The false-positive controls, measured against the real tree
# ---------------------------------------------------------------------------

LIVE = Path(__file__).resolve().parent.parent


def test_the_live_tree_itselves_the_two_false_positive_classes_the_checks_must_survive():
    """The controls #679 names, asserted on the real repo by discovery.

    Why not `vet_change_set("main", ROOT).violations == []`: that assertion runs
    against whatever the round under test happened to write, through the `tests`
    rung, and a soak that is required to block nothing cannot be enforced by a
    test that blocks. So the control is stated as what the vet actually reads:

      * the size parser sees every zero-byte tracked file as zero bytes — the
        one place a parsing bug in `ls-tree -l` would turn them into findings —
        and each of them has a shape the repo accepts (a package marker, or the
        one config that has been empty since the initial snapshot). The count is
        discovered, not pinned: it was 8 at #679's triage commit and 9 now, and
        a new package adds one more on any ordinary day.
      * each tracked `.onnx` really does contain a NUL, so the only thing between
        it and a finding is the allowlist, which is exactly clause 2's claim.
    """
    sizes = vet._paths_and_sizes(LIVE, "HEAD")
    assert sizes is not None, "ls-tree unreadable on the live tree"

    zero_byte = sorted(p for p, s in sizes.items() if s[0] == 0)
    assert len(zero_byte) >= 8, f"only {len(zero_byte)} zero-byte files: control absent"
    for path in zero_byte:
        ok = path.endswith(LIVE_ZERO_BYTE_OK_SUFFIXES) or path in LIVE_ZERO_BYTE_OK_EXACT
        assert ok, f"unexpected zero-byte tracked file {path}"

    models = sorted(p for p in sizes if p.endswith(".onnx"))
    assert len(models) >= 5, f"only {len(models)} tracked .onnx: control absent"
    for path in models:
        assert vet.binary_allowed(path), f"{path} is tracked but not allowlisted"
        assert vet._has_nul(LIVE, sizes[path][1]) is True, \
            f"{path} has no NUL, so it no longer proves the allowlist works"


def test_no_recent_landing_on_main_produces_an_empty_file_finding():
    """Check (a) against real traffic: the 12 most recent commits on `main`.

    The item's own soak question, taken as far as a unit test can: not "does
    HEAD's own diff look clean", which on a round that empties nothing and
    touches no model is true whatever the check does, but "do 12 ordinary
    merges and squash-merges of real rounds produce an empty-file finding",
    which a check that had regressed into flagging legitimate landings would not
    survive. The structural controls (the zero-byte files read as zero bytes, the
    `.onnx` models really carrying a NUL so the allowlist is what protects them)
    are in the test above, asserted by discovery rather than by count.
    """
    shas = subprocess.run(["git", "-C", str(LIVE), "rev-list", "-12", "main"],
                          capture_output=True, text=True).stdout.split()
    assert len(shas) == 12, shas
    empties = []
    for sha in shas:
        res = vet.vet_change_set(f"{sha}^", LIVE)
        assert res.evaluated, f"{sha[:8]}: {res.reason}"
        empties += [(sha[:8], v.path) for v in res.violations
                    if v.kind == vet.EMPTY_FILE]
    assert empties == [], f"empty-file findings on ordinary landings: {empties}"


def test_an_unreadable_added_blob_is_unevaluated_rather_than_clean(repo, monkeypatch):
    """Clause 4 at the blob seam: a NUL probe that cannot answer is never "no NUL".

    A missing object is a real shape on a shared worktree — a concurrent gc, a
    half-written alternates object — and the naive `cat-file | read(8 KiB)`
    reports it as an empty read, which reads as an empty file, which reads as
    text. `_has_nul` asks for the object's size first, so a blob git cannot
    resolve is `None`, and the check declines to evaluate rather than reporting a
    clean binary pass.
    """
    assert vet._has_nul(repo.root, "0" * 40) is None, "a missing object read as text"
    assert vet._has_nul(repo.root, "HEAD~1:app/service.py") is False, \
        "a readable text blob must still answer, not come back unevaluated"

    repo.write("app/blob.dat", b"whatever")
    repo.commit("an added path whose blob then cannot be read")
    monkeypatch.setattr(vet, "_has_nul", lambda *a, **k: None)
    res = repo.vet()
    assert res.status == vet.UNEVALUATED
    assert "app/blob.dat" in res.reason, res.reason
