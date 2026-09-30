"""OKF's two *reserved* filenames are not concept documents (backlog #450).

OKF §3.1 reserves `index.md` and `log.md` **at any level** of the tree, and §8
says an index file "contains no frontmatter" — frontmatter is permitted only in
a bundle-root `index.md`, and only for `okf_version`. So a spec-conformant index
or log is, by construction, a file with no frontmatter.

Both gate scripts used to treat a concept `.md` with no frontmatter as a
violation and to backfill frontmatter into one. `index.md` and `log.md` were
absent from `EXCLUDE_FILES` in both (`validate_okf.py:37`, `okf_migrate.py:57`
held only `tags.md`), so the gate FAILED a conformant index while certifying
`projects/inner-voice-paper/index.md`, which passes only because it carries
frontmatter and so breaks §8 — the gate certified the non-conformant file and
rejected the conformant one, and `okf_migrate --apply` would have written
frontmatter into the two files the spec forbids it in.

Everything here drives the two scripts as the weekly gate and autonomy task #80
drive them — `subprocess` on the real CLI — because the defect lived in the CLI
verdict, not in a function. `--root` is what makes a fixture tree gradable on
that same command line; before it, the only way to exercise the reserved-name
skip was to write an index.md into the live vault.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATE = ROOT / "scripts" / "vault" / "validate_okf.py"
MIGRATE = ROOT / "scripts" / "vault" / "okf_migrate.py"

# Spec-conformant reserved files: prose only, NO frontmatter (§8).
INDEX = "# Test index\n\n## research\n- [Foo](foo.md) — a page about foo\n"
LOG = "# Test log\n\n## 2026-09-22\n- nothing happened here\n"
# A concept document with no frontmatter: a real §3 violation.
ORPHAN = "# Orphan\n\nthis page has no frontmatter and must be reported\n"


def run(script: Path, *args: str) -> subprocess.CompletedProcess:
    """Invoke one gate CLI the way the conformance task does."""
    return subprocess.run([sys.executable, str(script), *args],
                          capture_output=True, text=True, cwd=ROOT)


def reserved_tree(tmp_path: Path) -> Path:
    """A fixture vault holding ONLY the reserved names, at three depths.

    `index.md` at the root, `index.md` nested, `log.md` nested — §3.1 reserves
    both names at any level, so the skip must be by filename, not by path.
    """
    root = tmp_path / "vault"
    (root / "knowledge" / "ai").mkdir(parents=True)
    (root / "index.md").write_text(INDEX, encoding="utf-8")
    (root / "knowledge" / "index.md").write_text(INDEX, encoding="utf-8")
    (root / "knowledge" / "ai" / "log.md").write_text(LOG, encoding="utf-8")
    return root


def scanned_count(stdout: str) -> int:
    """N from either script's scan line: 'scanned N concept files' / '(N files scanned)'."""
    m = re.search(r"scanned (\d+) concept files", stdout) or \
        re.search(r"\((\d+) files scanned\)", stdout)
    assert m, f"no scan count in output:\n{stdout}"
    return int(m.group(1))


# ── clause 1: reserved files scan clean at any depth ─────────────────────────

def test_reserved_files_are_not_okf_violations(tmp_path):
    """A tree whose only .md files are a frontmatter-less index.md and log.md,
    nested and at the root, validates clean and exits 0.

    Before the fix this printed `VIOLATIONS : 3` with 'no parseable frontmatter
    block' for each of the three files and exited 1 — i.e. the gate was red on a
    tree that OKF itself calls conformant.
    """
    root = reserved_tree(tmp_path)
    proc = run(VALIDATE, "--root", str(root))
    assert proc.returncode == 0, (
        f"validator exited {proc.returncode} on a spec-conformant reserved-only "
        f"tree:\n{proc.stdout}\n{proc.stderr}")
    assert "VIOLATIONS : 0" in proc.stdout, proc.stdout
    assert scanned_count(proc.stdout) == 0, (
        "a reserved file was counted as a concept document")


# ── clause 2: non-vacuity — the scan still reports a real violation ──────────

def test_a_non_reserved_page_beside_reserved_files_is_still_a_violation(tmp_path):
    """Control for clause 1: adding one frontmatter-less non-reserved page to the
    SAME tree makes the same scan exit 1 and name only that page.

    Without this control, clause 1 passes as easily when the scan skips the whole
    tree as when it skips two filenames.
    """
    root = reserved_tree(tmp_path)
    (root / "knowledge" / "orphan.md").write_text(ORPHAN, encoding="utf-8")
    proc = run(VALIDATE, "--root", str(root))
    assert proc.returncode == 1, (
        f"validator exited {proc.returncode}; a frontmatter-less concept page is a "
        f"real §3 violation:\n{proc.stdout}")
    assert "VIOLATIONS : 1" in proc.stdout, proc.stdout
    assert "knowledge/orphan.md: no parseable frontmatter block" in proc.stdout
    reported = proc.stdout.split("OKF violations", 1)[-1]
    for reserved in ("index.md", "log.md"):
        assert reserved not in reported, (
            f"the non-vacuity control had to report only knowledge/orphan.md, but "
            f"{reserved} is named in the violation list too:\n{proc.stdout}")


# ── clause 3: the migrator leaves reserved files byte-identical under --apply ─

def test_migrator_changes_no_reserved_file(tmp_path):
    """`okf_migrate.py --apply` against a fixture root containing a
    frontmatter-less index.md and log.md leaves every reserved file byte-identical
    and creates no frontmatter in either.

    Before the fix a dry run over this same tree printed `created fm : 4` — all
    four files — and `--apply` wrote `type: note` into `index.md` and `log.md`,
    the two files OKF §8 forbids frontmatter in.

    The non-reserved page in the same tree is the control: it MUST gain
    frontmatter, which proves `--apply` actually wrote and that the reserved
    files were left alone by name, not because the run did nothing.
    """
    root = reserved_tree(tmp_path)
    orphan = root / "knowledge" / "orphan.md"
    orphan.write_text(ORPHAN, encoding="utf-8")
    before = {p: p.read_bytes() for p in sorted(root.rglob("*.md"))}

    proc = run(MIGRATE, "--root", str(root), "--apply")
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"

    for rel in ("index.md", "knowledge/index.md", "knowledge/ai/log.md"):
        path = root / rel
        assert path.read_bytes() == before[path], (
            f"okf_migrate --apply rewrote reserved file {rel}")
        assert not path.read_text(encoding="utf-8").startswith("---"), (
            f"frontmatter was created in reserved file {rel}, which §8 forbids")

    assert "created fm   : 1" in proc.stdout, (
        f"exactly one scannable page is left in this tree, the control, and it must "
        f"have gained frontmatter:\n{proc.stdout}")
    assert orphan.read_text(encoding="utf-8").startswith("---"), (
        "the control page did not gain frontmatter, so --apply wrote nothing and "
        "the byte-identity assertions above would pass vacuously")


# ── clause 4: --root on both CLIs, defaulting to the live vault ──────────────

def test_both_clis_take_a_root_and_scan_the_tree_they_are_given(tmp_path):
    """`--root PATH` is accepted by both scripts and makes them scan that tree
    instead of `~/obsidian`, so clauses 1-3 are graded on the same command line
    the weekly gate and autonomy task #80 use. Omitting it must still read the
    live vault root.

    The live-vault control is `--dir autonomy`: a small directory that really is
    in `~/obsidian`. A fixture root scans 0 files there (it has no `autonomy/`),
    the default root scans the real ones — so the two runs differ by *which tree*
    was read, not by whether reading the vault works at all.
    """
    root = reserved_tree(tmp_path)

    assert run(VALIDATE, "--root", str(root)).returncode == 0
    fixture_scan = run(VALIDATE, "--root", str(root), "--dir", "autonomy")
    assert scanned_count(fixture_scan.stdout) == 0, (
        "--root did not redirect the validator's scan:\n" + fixture_scan.stdout)
    live_scan = run(VALIDATE, "--dir", "autonomy")
    assert scanned_count(live_scan.stdout) > 0, (
        "with no --root the validator must still scan the live vault root:\n"
        + live_scan.stdout)

    dry = run(MIGRATE, "--root", str(root), "--apply")
    assert dry.returncode == 0, f"{dry.stdout}\n{dry.stderr}"
    fixture_mig = run(MIGRATE, "--root", str(root), "--dir", "autonomy")
    assert scanned_count(fixture_mig.stdout) == 0, (
        "--root did not redirect the migrator's scan:\n" + fixture_mig.stdout)
    live_mig = run(MIGRATE, "--dir", "autonomy")
    assert scanned_count(live_mig.stdout) > 0, (
        "with no --root the migrator must still scan the live vault root:\n"
        + live_mig.stdout)


# ── #1934: `backlog/data/` is exempt from the OKF gate by path prefix ────────

#: A witness sidecar: prose with NO front matter block. This is the shape
#: `backlog/data/usage.db.witness.md` has, and it was the only OKF violation in
#: `backlog` — 1,877 files scanned, 1 violation, measured 2026-09-30. The four
#: other sidecars there carry a block and omit only `segment:`/`tags:`, which the
#: segment scan counts; both halves clear on the one skip in `iter_md`.
WITNESS = ("# usage.db witness\n\n"
           "Frozen extract; its provenance is the commit and md5 named in the "
           "body, not a front matter key.\n")


def witness_tree(tmp_path: Path) -> Path:
    """A fixture vault whose only note is a frontmatter-less sidecar in `backlog/data/`."""
    root = tmp_path / "vault"
    (root / "backlog" / "data").mkdir(parents=True)
    (root / "backlog" / "data" / "usage.db.witness.md").write_text(
        WITNESS, encoding="utf-8")
    return root


def test_backlog_data_is_not_scanned_for_conformance(tmp_path):
    """The gate skips the whole `backlog/data/` subtree, so a sidecar with no
    front matter is neither a violation nor a counted concept file — on a
    full-tree scan and on the `--dir backlog` scan the conformance task runs.

    Before #1934 the same tree printed `VIOLATIONS : 1` naming that file and
    exited 1. The ruling is that a witness sidecar is not a concept document at
    all — its body is the load-bearing frozen extract and its provenance is a
    checksum — so the subtree is exempt, rather than having keys stamped into
    it after the fact by a nightly.
    """
    root = witness_tree(tmp_path)
    for args in (("--root", str(root)),
                 ("--root", str(root), "--dir", "backlog")):
        proc = run(VALIDATE, *args)
        assert proc.returncode == 0, (
            f"validator exited {proc.returncode} on `{' '.join(args)}` with only "
            f"an exempt sidecar in the tree:\n{proc.stdout}\n{proc.stderr}")
        assert "VIOLATIONS : 0" in proc.stdout, proc.stdout
        assert scanned_count(proc.stdout) == 0, (
            f"a file under backlog/data was counted as a concept document on "
            f"`{' '.join(args)}`:\n{proc.stdout}")


def test_no_path_outside_backlog_data_leaves_the_okf_gate(tmp_path):
    """Control: the exemption is the path `backlog/data`, not a directory named `data`.

    Three byte-identical frontmatter-less notes at three places. Only the one
    under `backlog/data/` is skipped; the one directly under `backlog/` and the
    one in `knowledge/data/` are both reported, so every path outside the exempt
    subtree is still graded. The second of the two is what a bare `"data"` in
    `EXCLUDE_DIRS` would have let through — that set matches a directory name at
    any depth, so only a vault-relative path prefix leaves the gate whole over
    every other tree in the vault.
    """
    root = witness_tree(tmp_path)
    (root / "backlog" / "1934-item.md").write_text(WITNESS, encoding="utf-8")
    (root / "knowledge" / "data").mkdir(parents=True)
    (root / "knowledge" / "data" / "orphan.md").write_text(
        WITNESS, encoding="utf-8")

    proc = run(VALIDATE, "--root", str(root))
    assert proc.returncode == 1, (
        "two frontmatter-less concept notes outside the exempt subtree are real §3 "
        f"violations:\n{proc.stdout}")
    assert "VIOLATIONS : 2" in proc.stdout, proc.stdout
    assert "backlog/1934-item.md: no parseable frontmatter block" in proc.stdout
    assert "knowledge/data/orphan.md: no parseable frontmatter block" in proc.stdout
    reported = proc.stdout.split("OKF violations", 1)[-1]
    assert "backlog/data/" not in reported, (
        f"the exempt sidecar was reported beside the two real violations:\n"
        f"{proc.stdout}")
