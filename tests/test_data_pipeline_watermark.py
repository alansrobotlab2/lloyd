"""The data_pipeline watermark goes through one validated atomic writer (#4245).

`~/lloyd-data/_pipeline/autonomy-watermarks.json` was found on 2026-11-17 at 23:09Z holding
67 bytes whose entire content was the JSON-encoded absolute path of itself, written one second
after the data-pipeline run's own `json.dump` and one second before the read-back that caught
it. The individual writer was never named (no change-ledger pre-image for `sha1(realpath)`
`3909b7248a050238984981417b8f11d4f2fb4a19`; the vault audit log holds vault paths only; zero
of 172,280 small files under the data root contain their own path, which is what rules out the
tombstone-pointer and lock-helper candidates — `app/atomic_io.py`'s `_flock_exclusive` opens a
lock `O_RDWR` and never writes a byte). What the item leaves owed is therefore the *writer
class*, and the class it describes is a heredoc: `skills/autonomy-data-pipeline/SKILL.md`
re-typed by hand in two branches, doing a bare `json.load` with no type check and an
`open(wm,"w")` truncate-then-write, committing whatever the re-typing produced.

Reproducing the 67-byte payload byte-for-byte against the reader the item assumed would
self-heal refutes its impact claim: `scripts/autonomy/self_improve.py` `load_watermarks`
(:1146-1156) returns the bare `str` without error, because its `except Exception` catches only
syntax errors, and `check_watermark` then raises `AttributeError: 'str' object has no attribute
'get'` (:1185) while `update_watermark` raises `TypeError: 'str' object does not support item
assignment` (:1164, outside its `try` at :1173). A clobber is an unhandled crash, and the file
can never be repaired by the lane that reads it — so the defence has to be the write path.

`scripts/autonomy/data_pipeline_watermark.py` is that path, and every node below runs it as
production does — a subprocess whose `argv` is what the skill will shell out to — except the
two that have to see *inside* one call: the rename witness and the eaten-write replay, which
patch `os.replace` in-process because the thing they prove happens during the commit.

One seam is pinned by node `test_the_default_path_comes_from_app_paths`: the skill invokes the
writer with no `--path` at all, so the file it lands on is decided by `app.paths.PIPELINE_DIR`
in the child process, exactly as it is for the extraction script next door.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "autonomy" / "data_pipeline_watermark.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.skill_folder import disk_skill_folder  # noqa: E402

VAULT = Path.home() / "obsidian"
SKILL = VAULT / "skills" / "autonomy-data-pipeline" / "SKILL.md"

#: The name is asserted, not imported, so a node can fail on the spelling and not on a
#: constant the code under test chose.
WATERMARK_NAME = "autonomy-watermarks.json"
#: The block the writer owns, and the four keys that are its whole canonical shape —
#: `skills/autonomy-data-pipeline/SKILL.md`'s Post-Flight table is the source of this list.
BLOCK = "data_pipeline"
CANON = ["last_run", "status", "last_result", "last_log"]

#: The 2026-11-17 artifact, byte for byte: the store's own absolute path, JSON-encoded. The
#: item measured it at 67 bytes and `python3 -c "...type(json.load(...))..."` answered `str`;
#: `_self_path_artifact` pins that length so every "67" below is a measurement, not a guess.
SELF_PATH = "/home/alansrobotlab/lloyd-data/_pipeline/autonomy-watermarks.json"

#: A post-flight run's own provenance, in the shape Step 1's log carries it.
RESULT = "files_processed=16 facts=1134 failed=0 truncated=3 status=ran"
LOG = "/tmp/data-pipeline-run-20261117-144338494524540.txt"

#: A sibling top-level block this writer must not touch. It is the one the item names as the
#: cost of the clobber ("the whole top-level dict, including the `self_improvement` block this
#: pipeline must not touch"), and the only other key the live file has ever carried.
OTHER_BLOCK = {"self_improvement": {"last_run": "2026-09-03T09:00:00+00:00",
                                     "last_session_mtime": 1789000000.0}}


def _self_path_artifact() -> bytes:
    """The exact 67 bytes found on 2026-11-17: `json.dump(path, f)` of the store's own path."""
    artifact = json.dumps(SELF_PATH).encode()
    assert len(artifact) == 67, f"the item measured 67 bytes; this fixture is {len(artifact)}"
    return artifact


def _healthy_file(path: Path) -> bytes:
    """A watermark holding the canonical block plus a sibling block, as the live file does.

    Returns the seeded bytes so a caller can compare after a refused write.
    """
    payload = {"data_pipeline": {"last_run": "2026-11-17T23:10:25.337093+00:00",
                                 "status": "completed",
                                 "last_result": RESULT,
                                 "last_log": LOG},
               **OTHER_BLOCK}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path.read_bytes()


def _run(args: list[str], *, stdin: str | None = None,
         root: Path | None = None) -> subprocess.CompletedProcess:
    """The writer as the skill runs it: a fresh `python3`, argv on the command line."""
    env = dict(os.environ)
    if root is not None:
        env["LLOYD_DATA"] = str(root)
    return subprocess.run([sys.executable, str(SCRIPT), *args], input=stdin,
                          capture_output=True, text=True, timeout=120, env=env)


def _load_module():
    """The writer as an importable module, for the two nodes that must see inside a commit."""
    spec = importlib.util.spec_from_file_location("data_pipeline_watermark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tmp_writer_args(tmp_path: Path) -> list[str]:
    return ["--path", str(tmp_path / WATERMARK_NAME)]


# ── clause 1: temp file + os.replace, and no stray survives ──────────────────────────────

def test_the_commit_is_a_rename_so_no_reader_sees_a_partial_file_and_no_tmp_survives(
        tmp_path, monkeypatch):
    """The old heredoc opened the target `O_TRUNC` and wrote into it, so a reader in the
    window saw an empty file or a prefix — and that window is exactly where the 2026-11-17
    write was eaten. A rename has no such window: nothing ever opens the destination for
    writing. Witnessed at the instant of `os.replace` itself, and the temp that carried the
    new bytes is gone once the call returns.
    """
    module = _load_module()
    target = tmp_path / WATERMARK_NAME
    seed = _healthy_file(target)
    assert set(json.loads(seed)) == {BLOCK, "self_improvement"}

    real_replace = os.replace
    seen: dict[str, object] = {}

    def spy(src, dst, *args, **kwargs):
        seen["tmp"] = Path(src)
        seen["tmp_bytes"] = Path(src).read_bytes()
        seen["dst_bytes_at_replace"] = (Path(dst).read_bytes() if Path(dst).exists() else None)
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", spy)
    code = module.main([*_tmp_writer_args(tmp_path), "--status", "completed",
                        "--last-result", RESULT, "--last-log", LOG])

    assert code == 0, "post-flight must commit"
    assert seen, "no os.replace means the commit is not a rename — clause 1"
    # The destination still held the WHOLE previous file at the moment of the rename: a
    # reader at any point up to that instant read the old watermark, never a prefix of it.
    assert seen["dst_bytes_at_replace"] == seed
    # And the temp held the whole new file, so the rename moved a complete document.
    committed_mid = json.loads(seen["tmp_bytes"])  # type: ignore[arg-type]
    assert list(committed_mid[BLOCK]) == CANON
    assert committed_mid[BLOCK]["last_result"] == RESULT
    assert committed_mid[BLOCK]["last_log"] == LOG

    # No half-written sibling is left behind: the directory holds exactly the watermark.
    assert sorted(p.name for p in tmp_path.iterdir()) == [WATERMARK_NAME]
    assert not list(tmp_path.glob("*.tmp"))
    # The sibling block the item says the clobber cost survives, values untouched.
    after = json.loads(target.read_text())
    assert after["self_improvement"] == OTHER_BLOCK["self_improvement"]
    assert after[BLOCK]["status"] == "completed"
    assert after[BLOCK]["last_run"].endswith("+00:00"), (
        "a naive stamp is local time wearing UTC's clothes; this box runs at -0700")


# ── clause 2: a payload that is not the canonical shape is refused, bytes untouched ───────

def test_replaying_the_67_byte_self_path_payload_is_refused_with_a_named_file_error(tmp_path):
    """The acceptance check's own replay: the artifact that bricked the reader is now a
    refused *write*, and the refusal names the file rather than raising from a reader three
    calls later. Exit non-zero, and the bytes on disk are the ones already there — a refusal
    that rewrote the file would be the clobber with better manners.
    """
    target = tmp_path / WATERMARK_NAME
    artifact = _self_path_artifact()
    target.write_bytes(artifact)

    proc = _run([*_tmp_writer_args(tmp_path), "--payload-file", "-"],
                stdin=artifact.decode())

    assert proc.returncode != 0, "a self-path payload must never be committable"
    assert target.read_bytes() == artifact, "a refusal must leave the bytes untouched"
    assert "REFUSED" in proc.stderr and str(target) in proc.stderr, proc.stderr
    assert "not a dict" in proc.stderr, proc.stderr
    assert not list(tmp_path.glob("*.tmp"))


def test_a_block_that_does_not_hold_exactly_the_four_keys_is_refused(tmp_path):
    """Key sprawl is the other half of the shape rule. As of 2026-09-27 the live file carried
    8 keys in this block — `last_snapshot`, `last_files_processed`, `last_facts`, `last_failed`
    minted by hand — and each one is a second surface for a number that can disagree with the
    first. A block with a fifth key, and a block missing one, are both refused by name.
    """
    target = tmp_path / WATERMARK_NAME
    seed = _healthy_file(target)
    healthy = json.loads(seed)

    sprawled = json.loads(json.dumps(healthy))
    sprawled[BLOCK]["last_facts"] = 1134
    short = json.loads(json.dumps(healthy))
    del short[BLOCK]["last_log"]

    for payload, named in ((sprawled, "last_facts"), (short, "last_log")):
        proc = _run([*_tmp_writer_args(tmp_path), "--payload-file", "-"],
                    stdin=json.dumps(payload))
        assert proc.returncode != 0, f"{named} must not be committable: {proc.stdout}"
        assert target.read_bytes() == seed, "a refusal must leave the bytes untouched"
        assert named in proc.stderr, f"the refusal must name {named}: {proc.stderr}"


def test_a_noop_cannot_carry_counts_and_a_run_cannot_commit_neither(tmp_path):
    """Two refusals that keep the block honest. A noop has no counts of its own, so passing
    `--last-result` with it is a run attributing an earlier run's numbers to itself — the
    misreading the skill's noop branch exists to stop. And a run with no `--status` has nothing
    true to write.
    """
    target = tmp_path / WATERMARK_NAME
    seed = _healthy_file(target)

    proc = _run([*_tmp_writer_args(tmp_path), "--status", "noop", "--last-result", RESULT])
    assert proc.returncode != 0, proc.stdout
    assert "noop" in proc.stderr and target.read_bytes() == seed, proc.stderr

    proc = _run(_tmp_writer_args(tmp_path))
    assert proc.returncode != 0, "no --status and no --payload-file means nothing to write"
    assert target.read_bytes() == seed


# ── clause 3: a non-dict file is repaired AND reported ────────────────────────────────────

def test_a_file_that_parses_to_a_non_dict_is_replaced_and_the_clobber_is_reported(tmp_path):
    """The clobber itself, met head-on: the writer still leaves a valid dict holding this
    run's block, because refusing here would leave the store bricked to every reader, but it
    says so first with the path and the byte count it found. Silence is what made 2026-11-17 a
    week-old discovery instead of a one-line report.
    """
    target = tmp_path / WATERMARK_NAME
    target.write_bytes(_self_path_artifact())

    proc = _run([*_tmp_writer_args(tmp_path), "--status", "completed",
                 "--last-result", RESULT, "--last-log", LOG])

    assert proc.returncode == 0, f"{proc.stdout}{proc.stderr}"
    repaired = json.loads(target.read_text())
    assert isinstance(repaired, dict) and list(repaired[BLOCK]) == CANON
    assert repaired[BLOCK]["last_result"] == RESULT
    assert repaired[BLOCK]["last_log"] == LOG

    report = [line for line in proc.stdout.splitlines() if line.startswith("CLOBBER")]
    assert len(report) == 1, f"the clobber must be reported on stdout, got: {proc.stdout}"
    assert str(target) in report[0], report[0]
    assert "67 bytes" in report[0], f"the byte count found must be in the report: {report[0]}"


def test_a_write_that_is_eaten_after_the_rename_is_reported_not_clean(tmp_path, monkeypatch,
                                                                      capsys):
    """The 2026-11-17 failure, replayed at the only seam it can be replayed at: something
    replaces the file in the window after the commit. Task #24 caught that by hand, with the
    read-back step this item's own report says it ran; here the read-back is inside the writer,
    so the run comes back non-zero naming the file instead of printing a clean watermark over a
    67-byte tombstone.
    """
    module = _load_module()
    target = tmp_path / WATERMARK_NAME
    _healthy_file(target)
    artifact = _self_path_artifact()

    real_replace = os.replace

    def eat_the_write(src, dst, *args, **kwargs):
        Path(src).write_bytes(artifact)  # the second writer wins the window
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", eat_the_write)
    code = module.main([*_tmp_writer_args(tmp_path), "--status", "completed",
                        "--last-result", RESULT, "--last-log", LOG])

    captured = capsys.readouterr()
    assert code != 0, f"a watermark that is not on disk must not exit 0: {captured.out}"
    assert "VERIFY FAILED" in captured.err, captured.err
    assert str(target) in captured.err, "the failure has to name the file, not the reader"
    assert "WATERMARK" not in captured.out, (
        f"the run must not also report a clean watermark: {captured.out}")
    assert not list(tmp_path.glob("*.tmp"))


def test_the_default_path_comes_from_app_paths(tmp_path):
    """The skill passes no `--path`, so where the watermark lives is decided by
    `app.paths.PIPELINE_DIR` in the child — the same resolution as every other store under the
    data root, and the reason a scratch root redirects this writer too. It also has to create
    `_pipeline/` itself: `app.paths` creates nothing under `DATA_ROOT` on import (#712), so a
    writer that assumed the directory died on a fresh root before it wrote anything.
    """
    root = tmp_path / "data-root"
    proc = _run(["--status", "noop", "--last-result", "files_processed=0", "--last-log", LOG],
                root=root)
    # A noop over an empty root has no counts to carry forward, so it is refused — that is
    # clause 2 working — and it is refused about the file app.paths chose.
    assert proc.returncode != 0
    assert str(root / "_pipeline" / WATERMARK_NAME) in proc.stderr, proc.stderr

    proc = _run(["--status", "completed", "--last-result", RESULT, "--last-log", LOG],
                root=root)
    assert proc.returncode == 0, f"{proc.stdout}{proc.stderr}"
    landed = root / "_pipeline" / WATERMARK_NAME
    assert landed.is_file(), "the writer must create _pipeline/ rather than die in open()"
    assert list(json.loads(landed.read_text())[BLOCK]) == CANON


# ── clause 4: the skill calls the writer instead of re-typing a heredoc ───────────────────

def _section(text: str, heading: str) -> str:
    """The span under one heading, up to the next heading of the same level or higher.

    Section-scoped on purpose: a whole-file scan would pass on a `json.dump` parked under a
    different heading while the branch that writes the watermark still said `open(wm,"w")`.
    """
    start = text.index(heading)
    level = len(heading) - len(heading.lstrip("#"))
    body = text[start + len(heading):]
    following = re.search(rf"^#{{1,{level}}}\s", body, re.MULTILINE)
    return body[:following.start()] if following else body


def test_the_skill_branches_call_the_writer_and_read_the_four_keys_back():
    """Every run re-emitted the write snippet by hand, so the payload was whatever that
    re-typing produced — the writer class this item asked for. Both branches now shell out to
    one module, and the post-flight read-back is no longer optional prose.
    """
    assert SKILL.is_file(), f"{SKILL} is the surface clause 4 is about"
    # The skill as a FOLDER, not one file: #4307 moved the noop branch and the
    # Post-Flight section into `step1-fact-extraction.md` and `post-flight.md` under their
    # same headings, so reading `SKILL.md` alone reported a moved instruction as a deleted
    # one — this node went red on item #4496 for exactly that. `disk_skill_folder` splices
    # each sibling's body under the heading it shares with `SKILL.md`, so `_section` below
    # still cuts one branch's span and a `json.dump` parked under some other heading still
    # fails it; what changes is that the branch's text is found where it now lives.
    text = disk_skill_folder(SKILL.parent, encoding="utf-8")[1]
    noop = _section(text, "### GATE — if `status=noop` (files_processed=0)")
    post = _section(text, "## Post-Flight: Watermark Update")

    for name, section in (("noop", noop), ("post-flight", post)):
        assert "data_pipeline_watermark.py" in section, (
            f"the {name} branch must invoke the writer, not re-type it")
        assert "json.dump" not in section, (
            f"the {name} branch still carries an inline json.dump heredoc")
        assert not re.search(r"""open\([^)]*['"]w['"]""", section), (
            f"the {name} branch still opens a file for writing itself — the truncate-then-write "
            "is the half a reader could catch empty")

    # A noop writes no counts, so it must not be given any: it is the writer that carries
    # last_result/last_log forward from the last run that had them.
    assert "--status noop" in noop, noop[:400]
    assert "--last-result" not in noop and "--last-log" not in noop

    # Post-flight still proves its own numbers and still reads all four keys back.
    assert "PIPELINE_RESULT" in post, "post-flight must still derive last_result from the log"
    assert re.search(r"read[- ]back", post), (
        "the four-key read-back has to stay a step, not an idea")
    assert "WATERMARK" in post, "the writer's read-back line has to be the thing it reads"
    quoted = [ln for ln in post.splitlines() if "WATERMARK" in ln and "data_pipeline" in ln]
    assert len(quoted) == 1, f"post-flight must quote one read-back line, got {quoted}"
    for key in CANON:
        assert f"{key}=" in quoted[0], f"{key} is not in the line post-flight reads back"
