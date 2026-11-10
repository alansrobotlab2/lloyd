"""Tests for `scripts/maintenance/clear_data_root_cruft.py` (#3422).

The program exists because #3210's arrivals leg can SEE a 0-byte cruft stub at the data root
and is forbidden to touch it (#3109): `false` and `sqlite3` stood there for weeks, reported
hourly, deleted by nothing. Everything worth testing here is on the deletion side of that
line, so the nodes are built the way the risk is — a name gets unlinked, or the program
refuses and says why:

* Every node that can delete goes through `main`, as a SUBPROCESS. The dry-run default is a
  property of the parsed command line; patched into this process it would still pass with
  `--apply` inverted in the source, and a reviewer would have no way to notice.
* The fixture root is a real directory carrying a real `.lloyd-data-root` marker, and the
  live root is never opened. Most nodes name it with `--data-root`, which is the seam a test
  uses; `test_the_resolver_route_deletes_under_the_root_it_resolved` and the two refusal nodes
  beside it run with NO root flag at all, so the resolver clause 1 names
  (`app.data_root.resolve_data_root_for_tree`) is what actually picks the directory in them —
  with `LLOYD_DATA` stripped from the child's environment, because rule 1 of that resolver is
  that variable and the harness sets it per node.
* The names that must survive are the ones the live root actually holds today — a populated
  `usage.db` with a live `-wal`, a 0-byte `kg.sqlite` shadowing a populated nested store, a
  0-byte `workers.db-wal` left by a checkpoint, `next-session-notes.json` plus its 0-byte
  `.lock`. An empty-file test that only ever sees `false` proves the mechanism and nothing
  about the judgement.

`lsof` is real in these nodes: the handle probe is the one refusal that asks a question of
the machine rather than of a name, and stubbing it would leave the claim "proven inert"
carried by the fake. The case where the probe cannot answer is reached by emptying the CHILD's
`PATH`, not by patching this process — that is the shape a cron entry with a bare `PATH`
actually faces, and a patched `subprocess.run` would only prove the branch exists.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/maintenance/clear_data_root_cruft.py"

# The two cruft names as filed, so a node reads as the incident rather than as a string.
CRUFT = ("false", "sqlite3")

# The #4025 name. `sqlite3.connect(os.path.expanduser("~/lloyd-data/workers.db") + "?mode=ro")`
# — a URI with no `file:` scheme and no `uri=True` — reads the whole string as a FILENAME, and the
# default read-write-create open leaves a 0-byte stub whose name is that string. The live root has
# carried `workers.db?mode=ro` since 2026-11-09 15:05, created by an owed-check session 56 s into
# its own run. `?` is therefore the one glob character a genuine data-root name carries, so the
# rule about it is about literal presence, not about shape. This fixture member is named for the
# store it sits beside, exactly as the live one is.
URI_STUB = "usage.db?mode=ro"


def _run(root: Path, *args: str, state_dir: "Path | None" = None) -> subprocess.CompletedProcess:
    """One invocation of the real program against a fixture root."""
    argv = [sys.executable, str(SCRIPT), f"--data-root={root}", *args]
    if state_dir is not None:
        argv.insert(2, f"--state-dir={state_dir}")
    return subprocess.run(argv, capture_output=True, text=True, timeout=180,
                          cwd=str(REPO))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A fixture data root holding the live root's whole 0-byte population, correctly shaped.

    `top_level_regular_files` reports regular files at any depth below the top level, so a
    nested populated store is in the same scan the sidecar test is answered from — which is
    what makes `kg.sqlite` at the root look, to a naive pairing test, like that store's
    sidecar. It is refused anyway, and by the rule that names its class.
    """
    r = tmp_path / "lloyd-data"
    r.mkdir()
    (r / ".lloyd-data-root").write_text("fixture\n", encoding="utf-8")
    (r / "false").write_bytes(b"")
    (r / "sqlite3").write_bytes(b"")
    (r / "usage.db").write_bytes(b"SQLite format 3\x00" + b"x" * 4096)
    (r / "usage.db-wal").write_bytes(b"w" * 32)
    nested = r / "_pipeline/vault-derived"
    nested.mkdir(parents=True)
    (nested / "kg.sqlite").write_bytes(b"SQLite format 3\x00" + b"y" * 4096)
    (r / "kg.sqlite").write_bytes(b"")
    (r / "workers.db").write_bytes(b"SQLite format 3\x00" + b"z" * 4096)
    (r / "workers.db-wal").write_bytes(b"")          # 0 B: a just-checkpointed live WAL
    notes = r / "next-session-notes.json"
    notes.write_text('{"held": true}\n', encoding="utf-8")
    (r / "next-session-notes.json.lock").write_bytes(b"")
    # Two more declared stores, refused here as 0 bytes ON PURPOSE. `usage.db` and `workers.db`
    # above carry bytes, so the size rule answers for them and the name rule is never reached;
    # a 0-byte `failure-ledger.db` is the case the name rule exists for — a create that died
    # mid-way is exactly what looks like cruft — and it is the shape `kg.sqlite` above is.
    # `quarantine` is the live root's fourth member and a declared name too
    # (`QUARANTINE_DIR_RELATIVE`), so it is a directory here as it is there.
    (r / "failure-ledger.db").write_bytes(b"")
    (r / "research.db").write_bytes(b"")
    (r / "quarantine").mkdir()
    # Two shapes the live root does NOT carry today, added because no live name reaches the
    # rule that is supposed to answer for them. `test_the_rule_that_answers_is_the_rule_that_is_
    # tested` pins WHICH rule answers each name, and `mc-state.json.lock` is declined by the lock
    # rule before the base-file rule is ever consulted, so without a non-`.lock` name beside a
    # live file the derived-name rule had no test at all. A 0-byte `X.bak` beside a live `X` is
    # the shape that rule names ("a name beside a file belongs to it"), and a 0-byte `X` with a
    # live `X-wal` is clause 1's "no -wal/-shm/-journal sibling of its own" — the pairing test
    # cannot catch that one, because `X` is the store, not the sidecar. Both are named for the
    # live shapes they generalise, and both are declared here as fixture-only.
    (r / "mc-state.json").write_text('{"v": 1}\n', encoding="utf-8")
    (r / "mc-state.json.bak").write_bytes(b"")
    (r / "cache").write_bytes(b"")
    (r / "cache-wal").write_bytes(b"w" * 16)
    return r


def _zero_byte_names(root: Path) -> "set[str]":
    return {p.name for p in root.iterdir()
            if p.is_file() and p.stat().st_size == 0}


def test_the_dry_run_is_the_default_and_deletes_nothing(root: Path):
    """The first thing to pin is that an argument list is not a deletion.

    Both cruft names are named, cleanly acceptable, and the exit is 0 — the dry run's job is
    to tell the operator what the real run would do, in the same words, and to change
    nothing. `usage.db`'s bytes are compared as well as its name: a fixture that only checks
    the file still exists would not notice a truncation.
    """
    before = (root / "usage.db").read_bytes()
    proc = _run(root, *CRUFT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "dry run" in proc.stdout, proc.stdout
    assert _zero_byte_names(root) >= set(CRUFT), (
        f"the default deleted: {_zero_byte_names(root)}\n{proc.stdout}")
    assert (root / "usage.db").read_bytes() == before
    assert "--apply" in proc.stdout, (
        "a dry run that never prints the flag it declined is not a rehearsal")


def test_apply_removes_exactly_the_two_named_cruft_stubs_and_nothing_else(root: Path):
    """#3422 clause 1+3: `--apply false sqlite3` empties the class down to its live members.

    The survivors are named individually rather than counted, because the count is the same
    (2 → 0 removed of 2 named, 6 others untouched) whether the program deleted the right two
    names or swept the class.
    """
    proc = _run(root, "--apply", *CRUFT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not (root / "false").exists() and not (root / "sqlite3").exists(), proc.stdout
    survivors = {
        "usage.db": "a declared store with live bytes",
        "usage.db-wal": "that store's live WAL",
        "kg.sqlite": "the 0-byte shadow file clause 3 says must still exist",
        "workers.db": "another declared store",
        "workers.db-wal": "a 0-byte sidecar of a live store",
        "next-session-notes.json": "a live file with a live lock",
        "next-session-notes.json.lock": "a 0-byte lock",
        "mc-state.json.bak": "the derived-name class: 0 bytes beside a live file",
        "cache": "a 0-byte store with a live -wal of its own",
        "cache-wal": "that store's WAL",
    }
    for name, why in survivors.items():
        assert (root / name).exists(), f"{name} was deleted, and it is {why}:\n{proc.stdout}"
    assert (root / "usage.db").stat().st_size > 0


def test_a_second_apply_of_the_same_two_names_is_a_noop_exiting_zero(root: Path):
    """#3422 clause 3: the run is safe to repeat, which is what makes an attended run calm.

    "Already gone" is not a failure: an operator who re-runs the command after a typo, or
    after the hourly leg's alert is still on screen, must not be greeted with a red exit that
    reads as "the deletion went wrong".
    """
    first = _run(root, "--apply", *CRUFT)
    assert first.returncode == 0, first.stdout + first.stderr
    second = _run(root, "--apply", *CRUFT)
    assert second.returncode == 0, (
        f"a repeat apply exited {second.returncode}:\n{second.stdout}{second.stderr}")
    assert "nothing to remove" in second.stdout, second.stdout


@pytest.mark.parametrize("name, why", [
    ("usage.db", "declared store with bytes in it"),
    ("workers.db", "declared store with bytes in it"),
    ("kg.sqlite", "declared store name AND a database suffix"),
    ("failure-ledger.db", "declared store name"),
    ("research.db", "declared store name"),
    ("workers.db-wal", "0-byte sidecar of a store in the same scan"),
    ("next-session-notes.json.lock", "0-byte lock beside its live base file"),
    ("next-session-notes.json", "a live file with bytes and a live lock"),
    (".lloyd-data-root", "the root's own marker"),
    ("quarantine", "a directory: never recursive"),
])
def test_a_name_a_live_store_or_sidecar_or_lock_claims_is_refused_by_name(
        root: Path, name: str, why: str):
    """#3422 clause 2: every refused class has a member in this fixture, so none is inert.

    `workers.db-wal` and `next-session-notes.json.lock` are the two that a naive version of
    this program deletes: each is 0 bytes, neither is refused by the candidate's own
    `-wal/-shm/-journal` sibling test (`workers.db-wal-wal` does not exist) nor by the
    database-suffix test (`'workers.db-wal'.endswith('.db')` is False). They are caught by
    pairing and by the lock rule, which is why they are in the fixture rather than in a
    comment.
    """
    existed = (root / name).exists()
    proc = _run(root, "--apply", name)
    assert proc.returncode == 1, (
        f"{name} ({why}) exited {proc.returncode}; a refused name must exit non-zero "
        f"\n{proc.stdout}{proc.stderr}")
    assert f"refused: {name}" in proc.stdout, proc.stdout
    assert proc.stdout.strip().splitlines()[-1].startswith("result:"), proc.stdout
    if existed:
        assert (root / name).exists(), f"{name} ({why}) was unlinked despite the refusal"


def test_the_refusal_names_the_class_it_refused_for(root: Path):
    """A refusal that only says "no" is not the deliverable: the reason is the product.

    Each of these strings is the thing an operator needs to decide the next action — "a
    suffix class the shadow leg owns" means go read that alert, "sidecar of a store in the
    same scan" means this is a database, not cruft.

    Asserted per NAME, on that name's own line, not anywhere in stdout. A whole-output
    substring test is how a rule that never fires can look covered: `workers.db-wal`'s refusal
    mentions "sidecar", so `"sidecar" in out` passes without the pairing rule being the thing
    that answered, and `next-session-notes.json.lock` contains the word "lock" in its OWN name,
    so a substring check on it cannot fail whatever the program says. Each class therefore
    gets the fragment only ITS rule writes, matched against the line for its own name. Two of
    these names exist to be reachable at all: `orphan.sqlite3` is the suffix class with no
    declared owner (`kg.sqlite` is refused as a declared name first), and
    `mc-state.json.bak` / `cache` are the derived-name and own-sidecar classes, which no live
    root name reaches — `.lock` shadows the derived-name rule on every real pair today, and
    `cache` is a store whose own `-wal` stands beside it, which the pairing test cannot see
    because `cache` is the store rather than the sidecar.
    """
    (root / "orphan.sqlite3").write_bytes(b"")
    wanted = {
        "kg.sqlite": "declared store",
        "orphan.sqlite3": "data_root_shadows",
        "workers.db-wal": "of a store standing in the same scan",
        "next-session-notes.json.lock": "owned by whatever takes it",
        "mc-state.json.bak": "derived from 'mc-state.json'",
        "cache": "sidecar(s) beside it: cache-wal",
    }
    proc = _run(root, "--apply", *wanted)
    assert proc.returncode == 1, proc.stdout
    lines = {ln.split(": ", 1)[1].split(":")[0]: ln
             for ln in proc.stdout.splitlines() if ln.startswith("refused: ")}
    for name, fragment in wanted.items():
        line = lines.get(name, "")
        assert line, f"no `refused: {name}:` line in:\n{proc.stdout}"
        assert fragment in line, (
            f"{name} was refused, but not by the rule that owns it. Expected {fragment!r} on "
            f"its own line, got:\n  {line}\n"
            f"(A refusal by an EARLIER rule leaves this class untested: the fixture still "
            f"survives, so nothing else would notice.)")





def test_a_path_or_glob_argument_is_refused_without_being_resolved(root: Path):
    """The reach rule is about this program's own hands: names, never a sweep.

    `sub` here is a directory that CONTAINS a perfectly removable cruft stub, so the refusal
    is not about the file's shape — it is that the argument named a path, and the operator
    who typed it meant a class. `*` is refused without ever being looked up.
    """
    sub = root / "sub"
    sub.mkdir()
    (sub / "false").write_bytes(b"")
    # Two different reds, and the difference is the point: 2 is a usage error meaning NO name
    # was looked at, 1 is a declined name meaning the accept-set judged it. A malformed
    # argument must not be reported as the second, because then `sub/false` beside a name that
    # really was removed reads as "the sweep was declined".
    for bad in ("sub/false", "*", "false sqlite3", "./false", "false*"):
        proc = _run(root, "--apply", bad)
        assert proc.returncode == 2, f"{bad!r} exited {proc.returncode}:\n{proc.stdout}"
        assert "refused" in proc.stdout, f"{bad!r}: {proc.stdout}"
        assert "NO name was looked at" in proc.stdout, f"{bad!r}: {proc.stdout}"
    # `..` is a bare dot-name, so it IS a name and the accept-set is what declines it.
    proc = _run(root, "--apply", "..")
    assert proc.returncode == 1, f"`..` exited {proc.returncode}:\n{proc.stdout}"
    assert "refused: .." in proc.stdout, proc.stdout
    assert (sub / "false").exists(), "a path argument reached into a subdirectory"
    assert (root / "false").exists(), f"a glob deleted the real one:\n{proc.stdout}"


def test_a_uri_shaped_stub_is_named_by_a_dry_run_rather_than_refused_as_a_glob(root: Path):
    """#4025 clause 1: the remover can name the stub a mis-spelled SQLite URI leaves behind.

    `usage.db?mode=ro` is 0 bytes, is a regular file at 644, is not a declared store's basename
    (`app.paths` declares `usage.db`), does not end in a database suffix (`_DB_SUFFIXES` is
    `.db`/`.sqlite`/`.sqlite3` and this ends `=ro`), and has no `-wal`/`-shm` sibling of its own —
    every property of the class this program exists to clear. Before this change it could not be
    handed to the program at all: `?` sat in `_FORBIDDEN_CHARS`, so the run printed "were not bare
    names, so NO name was looked at" and exited 2, which meant neither the unlink nor the
    arrivals-baseline prune could run and the hourly alert could only be closed by a hand-written
    `rm`. An operator reads the dry run before the real one, so the deliverable is the same two
    lines the `false` class gets: the mode, and the name with its inertness reason beside it.
    """
    (root / URI_STUB).write_bytes(b"")
    proc = _run(root, URI_STUB)
    assert proc.returncode == 0, (
        f"a literal ?-bearing name exited {proc.returncode}; it must reach the accept-set, not "
        f"the usage error\n{proc.stdout}{proc.stderr}")
    assert "were not bare names" not in proc.stdout, proc.stdout
    assert "mode: dry run — pass --apply to delete anything" in proc.stdout, proc.stdout
    assert (f"would remove: {URI_STUB}: 0-byte regular file, no declared owner, no sidecar, "
            "no handle") in proc.stdout, proc.stdout
    assert (root / URI_STUB).exists(), f"the rehearsal deleted:\n{proc.stdout}"
    assert (root / "usage.db").stat().st_size > 0, "the store beside it was touched"


def test_apply_unlinks_the_uri_shaped_stub_alone_and_prunes_it_from_the_baseline(
        root: Path, tmp_path: Path):
    """#4025 clause 2: `--apply` clears exactly that name, and a recurrence stays nameable.

    The prune is half of the deletion, not a courtesy. `write_arrivals_baseline` merges
    monotonically, so a `seen` key that survives its own unlink mutes the class forever — the
    arrivals leg that reported `workers.db?mode=ro` on 2026-11-09 could never report it twice.
    The survivors are named individually rather than counted, because a run that swept the root
    would leave the same count as one that named one file. The second `--apply` is here because
    the same command is what an operator types again when the first one scrolled past.
    """
    (root / URI_STUB).write_bytes(b"")
    state = tmp_path / "guardian-state"
    state.mkdir()
    sys.path.insert(0, str(REPO / "agent-services/guardian"))
    import datawatch as dw
    dw.write_arrivals_baseline(state, {URI_STUB: 1.0, "usage.db": 2.0}, now=100.0,
                               data_root=str(root))

    first = _run(root, "--apply", URI_STUB, state_dir=state)
    assert first.returncode == 0, first.stdout + first.stderr
    assert f"removed: {URI_STUB}: 0-byte regular file, no declared owner, no sidecar, " \
           "no handle" in first.stdout, first.stdout
    assert f"arrivals baseline pruned: {URI_STUB}" in first.stdout, first.stdout
    assert not (root / URI_STUB).exists(), f"reported removed but still standing:\n{first.stdout}"
    for name in ("usage.db", "usage.db-wal", "workers.db", "workers.db-wal", "kg.sqlite",
                 "next-session-notes.json", "next-session-notes.json.lock", "false", "sqlite3",
                 "mc-state.json.bak", "cache", "cache-wal"):
        assert (root / name).exists(), (
            f"{name} was deleted by a run that named exactly one name:\n{first.stdout}")
    seen = dw.read_arrivals_baseline(state, str(root))
    assert seen == {"usage.db": 2.0}, f"the prune was not exact: {seen}"

    second = _run(root, "--apply", URI_STUB, state_dir=state)
    assert second.returncode == 0, (
        f"a repeat apply of the same name exited {second.returncode}:\n"
        f"{second.stdout}{second.stderr}")
    assert "nothing to remove" in second.stdout, second.stdout


@pytest.mark.parametrize("bad, why", [
    ("fals?", "it resolves to `false`, which is standing right here — the sweep this rule is for"),
    ("usage.db-wa?", "it resolves to the live `usage.db-wal`: the sidecar a ? glob would take"),
    ("*", "the whole-class sweep"),
    ("[cf]alse", "a bracket class, refused on shape alone as always"),
])
def test_a_glob_that_is_not_a_name_here_is_refused_before_any_deletion(
        root: Path, bad: str, why: str):
    """#4025 clause 3: relaxing `?` did not relax the reach rule the `?` rule was protecting.

    Each argument is handed alongside `false`, a perfectly removable 0-byte stub, so a program
    that answered the bad spelling per-name would delete `false` on the way to refusing it. It
    must not: a pattern the shell was meant to expand is a bad INVOCATION, and the exit is 2 with
    "NO name was looked at", never the per-name refusal with exit 1 that reads as "the accept-set
    judged these". `fals?` is the case the `?` relaxation had to keep refusing, and it is refused
    from the root's own scan, which is the same listing the sidecar test and the hourly leg read.
    """
    proc = _run(root, "--apply", "false", bad)
    assert proc.returncode == 2, (
        f"{bad!r} ({why}) exited {proc.returncode}; expected 2, no name looked at\n"
        f"{proc.stdout}{proc.stderr}")
    assert "refused before any name was read" in proc.stdout, proc.stdout
    assert "NO name was looked at" in proc.stdout, proc.stdout
    assert (root / "false").exists(), (
        f"{bad!r} ({why}) was answered after `false` had already been unlinked:\n{proc.stdout}")
    assert (root / "sqlite3").exists(), f"{bad!r} swept the class instead of naming it"


def test_a_question_bearing_name_matching_nothing_at_all_is_the_same_no_op_as_false(
        root: Path):
    """A `?` spelling that resolves to nothing is an absent name, not a sweep — exit 0, nothing gone.

    The two halves of the `?` rule are told apart by evidence, not by intent: `fals?` is a sweep
    because something here answers to it, while `usage.db?mode=ro` in a root holding no such file
    (and this fixture deliberately holds none, unlike the live root) is a name that is simply not
    here — the identical situation as handing `false` after a successful `--apply`. Both must read
    as `nothing to remove` at exit 0, because a run that answered them as a usage error would make
    the repeat `--apply` of clause 2 fail, and an operator who re-types a command whose first run
    scrolled past is the ordinary case. Rehearsed rather than applied here, since the legitimate
    `false` beside it is supposed to be deleted when `--apply` is passed; the state where nothing
    may be touched is the sweep node above.
    """
    proc = _run(root, "false", URI_STUB)
    assert proc.returncode == 0, (
        f"an absent ?-bearing name exited {proc.returncode} instead of reading as a name that "
        f"is not here\n{proc.stdout}{proc.stderr}")
    assert "were not bare names" not in proc.stdout, proc.stdout
    assert f"nothing to remove: {URI_STUB}" in proc.stdout, proc.stdout
    assert "would remove: false" in proc.stdout, proc.stdout
    assert (root / "false").exists(), f"the rehearsal deleted:\n{proc.stdout}"


@pytest.mark.parametrize("sibling, mode, fragment", [
    ("-wal", 0o644, "sidecar(s) beside it: usage.db?mode=ro-wal"),
    ("-shm", 0o644, "sidecar(s) beside it: usage.db?mode=ro-shm"),
    ("", 0o666, "world-writable"),
])
def test_a_literal_question_mark_buys_no_second_acceptance(
        root: Path, sibling: str, mode: int, fragment: str):
    """#4025 clause 4: literal presence is the ONLY thing accepting `?` grants.

    `usage.db?mode=ro` beside its own `-wal` is a store in use, not a stub, and one at mode 666 is
    not the residue of a create that died — the same two refusals, in the same words, with the
    same exit 1, that `false` would get. The fragment is matched on the line for THIS name and
    nowhere else in the output, because a refusal by an earlier rule leaves the class untested
    while the fixture still survives: nothing but this assertion would notice.
    """
    victim = root / URI_STUB
    victim.write_bytes(b"")
    if sibling:
        (root / (URI_STUB + sibling)).write_bytes(b"w" * 16)
    if mode != 0o644:
        os.chmod(victim, mode)
    proc = _run(root, "--apply", URI_STUB)
    assert proc.returncode == 1, (
        f"a {sibling or 'world-writable'} {URI_STUB} exited {proc.returncode}:\n"
        f"{proc.stdout}{proc.stderr}")
    line = next((ln for ln in proc.stdout.splitlines()
                 if ln.startswith(f"refused: {URI_STUB}")), "")
    assert line, f"no `refused: {URI_STUB}:` line in:\n{proc.stdout}"
    assert fragment in line, (
        f"{URI_STUB} was refused, but not by the rule that owns it. Expected {fragment!r} on "
        f"its own line, got:\n  {line}")
    assert victim.exists(), f"{URI_STUB} was unlinked despite the refusal"
    assert (root / "usage.db").stat().st_size > 0, "the populated store beside it was touched"


def test_a_question_bearing_name_claimed_by_a_store_or_missing_from_the_scan_is_refused(
        root: Path):
    """The two clause-4 halves no fixture on disk can hold, asked of `classify` itself.

    A name cannot be both `usage.db?mode=ro` and one of the basenames `declared_store_names()`
    reads out of `app.data_root` and `app/paths.py`, and a file that stands at the root is by
    definition in the scan taken from it — so those two rules are unreachable through a
    subprocess, and were the reason a `?` rule that only re-checked the accept-set could look
    covered while quietly waving a store through. Both are therefore asked of the real function
    with the one argument a fixture cannot supply: the store set, and the scan mapping. The scan
    is taken with the leg's own `top_level_regular_files`, which is also the proof that the
    `?`-bearing name is visible to the hourly walk at all.
    """
    sys.path.insert(0, str(REPO / "scripts/maintenance"))
    import clear_data_root_cruft as crc
    (root / URI_STUB).write_bytes(b"")
    scan = crc.DW.top_level_regular_files(str(root))
    assert URI_STUB in scan, "the leg's own scan cannot see a ?-bearing name"
    stores = crc.declared_store_names()
    assert URI_STUB not in stores, (
        f"{URI_STUB} is a declared store basename today, so the hypothetical below is not one")

    verdict, reason = crc.classify(URI_STUB, root, scan, stores | {URI_STUB})
    assert verdict == "refuse" and "declared store" in reason, (
        f"a store-claimed ?-bearing name answered {verdict}: {reason}")
    stripped = {k: v for k, v in scan.items() if k != URI_STUB}
    verdict2, reason2 = crc.classify(URI_STUB, root, stripped, stores)
    assert verdict2 == "refuse" and "not in the scan" in reason2, (
        f"a ?-bearing name outside the scan answered {verdict2}: {reason2}")
    assert (root / URI_STUB).exists(), "classify is not a deletion route"


def test_a_nonempty_or_special_file_is_refused(root: Path):
    """Size 0 and "regular file" are separate refusals, and both are checked before any unlink.

    The fifo matters more than it looks: `os.stat` reports size 0 for it, so a checker that
    reads only the size accepts a named pipe as a stub.
    """
    (root / "notcruft").write_text("payload\n", encoding="utf-8")
    os.mkfifo(root / "afifo")
    os.symlink(root / "usage.db", root / "db-link")
    for name in ("notcruft", "afifo", "db-link"):
        proc = _run(root, "--apply", name)
        assert proc.returncode == 1, f"{name}:\n{proc.stdout}"
        assert (root / name).lstat() is not None, f"{name} was deleted"
    assert (root / "usage.db").read_bytes(), "following a symlink would have emptied a store"


def test_an_open_handle_is_a_refusal_even_for_a_perfect_cruft_stub(tmp_path: Path):
    """The one refusal that asks the machine, asked for real: `lsof`, not a stub.

    The name is `false`, freshly created, 0 bytes, at the root of the thing — every property
    of a provably-inert stub except the one property a name cannot carry. The test process
    holds it open; the probe is a real subprocess and sees this process's pid.
    """
    r = tmp_path / "lloyd-data"
    r.mkdir()
    (r / ".lloyd-data-root").write_text("fixture\n", encoding="utf-8")
    victim = r / "false"
    victim.write_bytes(b"")
    sys.path.insert(0, str(REPO / "scripts/maintenance"))
    import clear_data_root_cruft as crc
    fd = os.open(str(victim), os.O_RDONLY)
    try:
        held, reason = crc.has_open_handle(victim)
    finally:
        os.close(fd)
    assert held, f"the probe reported no handle on a file this process has open: {reason}"
    assert str(os.getpid()) in reason, reason
    assert victim.exists()


def test_a_handle_probe_that_cannot_answer_refuses_rather_than_passes(
        root: Path, monkeypatch):
    """"Could not ask" is not "nobody is using it", so no answer keeps the file.

    This is the only node here that stubs the probe, and it stubs it in-process because the
    condition is a missing `lsof`, which cannot be arranged from a subprocess without editing
    the PATH out from under the interpreter too.
    """
    sys.path.insert(0, str(REPO / "scripts/maintenance"))
    import clear_data_root_cruft as crc
    monkeypatch.setattr(crc.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no lsof here")))
    held, reason = crc.has_open_handle(root / "false")
    assert held, "a probe that raised was read as no handle"
    assert "could not ask" in reason, reason


def test_the_store_refusals_are_read_from_the_declaring_modules_not_a_list_here():
    """The refuse-list must extend itself when a store is added, or it is a second opinion.

    Asserted against the modules rather than against literals, with the literals named in the
    message so a reader sees what the four attributes are today. A copy of the names in the
    script would pass this node and fail the day someone adds a fifth store.
    """
    sys.path.insert(0, str(REPO / "scripts/maintenance"))
    import clear_data_root_cruft as crc
    import app.paths as P
    from app.data_root import KG_DB_RELATIVE
    got = crc.declared_store_names()
    assert Path(str(KG_DB_RELATIVE)).name in got, got
    for attr in ("USAGE_DB", "WORKERS_DB", "FAILURE_LEDGER_DB", "RESEARCH_DB"):
        assert Path(str(getattr(P, attr))).name in got, (attr, sorted(got))
    src = SCRIPT.read_text(encoding="utf-8")
    for literal in ('"usage.db"', "'usage.db'", '"workers.db"', "'workers.db'",
                    '"failure-ledger.db"', "'failure-ledger.db'", '"research.db"',
                    "'research.db'"):
        assert literal not in src, (
            f"the script carries {literal} as a literal: the refuse-list is read from "
            "app.data_root and app/paths so a new store extends it by itself")


def test_the_prune_fires_only_after_a_real_removal_and_names_the_pruned_name(
        root: Path, tmp_path: Path):
    """Deletion and bookkeeping in one run, or the removal silently defeats the leg.

    The baseline merges monotonically, so an unpruned `seen` key means the hourly alert can
    never name a recurrence of the name it exists to catch. This is the wiring node: after
    `--apply`, the fixture baseline must no longer hold `false`/`sqlite3`, and the program
    must say which names it pruned.
    """
    state = tmp_path / "guardian-state"
    state.mkdir()
    sys.path.insert(0, str(REPO / "agent-services/guardian"))
    import datawatch as dw
    dw.write_arrivals_baseline(state, {"false": 1.0, "sqlite3": 2.0, "usage.db": 3.0,
                                      "usage.db-wal": 4.0}, now=100.0, data_root=str(root))
    proc = _run(root, "--apply", *CRUFT, state_dir=state)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # Read for the fixture root, the same one the baseline was written against: the reader
    # refuses a root mismatch by contract, so asking it about the default root answers `None`.
    seen = dw.read_arrivals_baseline(state, str(root))
    assert set(seen) == {"usage.db", "usage.db-wal"}, (
        f"the prune was not exact: {sorted(seen)}")
    assert proc.stdout.count("arrivals baseline pruned:") == 2, proc.stdout


def test_a_dry_run_never_touches_the_arrivals_baseline(root: Path, tmp_path: Path):
    """A rehearsal must be a rehearsal in every store it can reach.

    Pruning the baseline is the one write this program makes besides the unlink, and a dry
    run that performed it would make the next real run's alert look like a recurrence that
    never happened.
    """
    state = tmp_path / "guardian-state"
    state.mkdir()
    sys.path.insert(0, str(REPO / "agent-services/guardian"))
    import datawatch as dw
    dw.write_arrivals_baseline(state, {"false": 1.0}, now=100.0, data_root=str(root))
    proc = _run(root, *CRUFT, state_dir=state)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "false" in dw.read_arrivals_baseline(state, str(root)), (
        "the dry run pruned, so a real recurrence would now be muted")


def _run_no_data_root(tree: Path, *args: str, state_dir: "Path | None" = None,
                      with_lloyd_data: "str | None" = None
                      ) -> subprocess.CompletedProcess:
    """One invocation with NO `--data-root`, so `app.data_root` is what picks the root.

    `LLOYD_DATA` is stripped unless given: the test harness sets it to a throwaway directory
    per node, and rule 1 of the resolver is that variable, so leaving it set would have this
    node measure the environment instead of the route clause 1 names.
    """
    argv = [sys.executable, str(SCRIPT)]
    env = {k: v for k, v in os.environ.items() if k != "LLOYD_DATA"}
    if with_lloyd_data is not None:
        env["LLOYD_DATA"] = with_lloyd_data
    if tree is not None:
        argv.append(f"--tree={tree}")
    if state_dir is not None:
        argv.append(f"--state-dir={state_dir}")
    argv.extend(args)
    return subprocess.run(argv, capture_output=True, text=True, timeout=180,
                          cwd=str(REPO), env=env)


def test_an_unreachable_lsof_refuses_the_name_rather_than_calling_it_inert(
        root: Path, tmp_path: Path):
    """The handle probe across the boundary it is really crossed: no `lsof` on this PATH.

    The in-process node above fakes a failing probe, which proves the branch exists but not the
    thing the branch is for — that a program run from a shell or a cron entry whose `PATH` has
    no `lsof` in it declines instead of deleting. So this node removes the binary from the
    child's view by emptying its `PATH` (the interpreter is invoked by absolute path, and the
    program needs nothing else external), and the perfect cruft stub `false` is handed to it
    with `--apply`. The failure mode being refused is the quiet one: `subprocess` raising
    `FileNotFoundError` and a naive `except: pass` reading as "no handle reported, therefore
    inert".

    The empty `PATH` is also why `sys.executable`'s directory is put back: without a working
    interpreter on the PATH the child never starts, and `FileNotFoundError` would be a fact
    about the test harness rather than about `lsof`.
    """
    probe_dir = tmp_path / "empty-bin"
    probe_dir.mkdir()
    env = dict(os.environ)
    env["PATH"] = f"{probe_dir}:{os.path.dirname(sys.executable)}"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), f"--data-root={root}", "--state-dir",
         str(tmp_path / "state"), "--apply", "false"],
        capture_output=True, text=True, timeout=180, cwd=str(REPO), env=env)
    assert proc.returncode == 1, (
        f"an absent lsof did not refuse: exit {proc.returncode}\n{proc.stdout}{proc.stderr}")
    assert "refused: false" in proc.stdout, proc.stdout
    assert "handle" in proc.stdout, proc.stdout
    assert (root / "false").exists(), "a name was deleted with the probe unanswerable"


def test_the_resolver_route_deletes_under_the_root_it_resolved(tmp_path: Path):
    """#3422 clause 1's resolution, run as the route rather than as an override.

    Every other node in this file hands the program `--data-root`, which is the seam the test
    uses and NOT the path a person on cron takes: with no flag, the root comes from
    `app.data_root.resolve_data_root_for_tree`, which is what the clause names and `app.paths`
    is not allowed to be. A checkout of its own resolves to `<tree>/.lloyd-data` (rule 3), so
    a fixture checkout is enough to watch that function choose, and the proof it chose is a
    deletion at the path ONLY it could have produced — `--data-root` was never passed, and no
    environment variable was left pointing anywhere.
    """
    tree = tmp_path / "checkout"
    data = tree / ".lloyd-data"
    data.mkdir(parents=True)
    (data / ".lloyd-data-root").write_text("fixture\n", encoding="utf-8")
    (data / "false").write_bytes(b"")
    (data / "usage.db").write_bytes(b"SQLite format 3\x00" + b"x" * 512)

    proc = _run_no_data_root(tree, "--apply", "false",
                             state_dir=str(tmp_path / "state"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"data root: {data}" in proc.stdout, (
        f"the resolver did not answer with <tree>/.lloyd-data:\n{proc.stdout}")
    assert not (data / "false").exists(), f"resolved but did not delete:\n{proc.stdout}"
    assert (data / "usage.db").exists(), "the resolver route swept instead of naming"


def test_the_resolver_route_refuses_an_unmarked_root(tmp_path: Path):
    """Without `--data-root`, a root the resolver answers with is still refused without its marker.

    The route has two halves, and a node that passed `--data-root` tested neither: the resolver
    picks the directory (`<tree>/.lloyd-data` for a checkout that is not production), and then
    the root's own `.lloyd-data-root` marker is what licenses deleting inside it. Here the
    resolver DOES answer — the directory exists — and is refused a second later. A half-built
    checkout or a `--tree` typo therefore gets an exit 2 and a sentence, not an `ls` with
    deletions in it.
    """
    tree = tmp_path / "half-built"
    data = tree / ".lloyd-data"
    data.mkdir(parents=True)                        # the resolver's answer, unmarked
    (data / "false").write_bytes(b"")
    proc = _run_no_data_root(tree, "--apply", "false",
                             state_dir=str(tmp_path / "state"))
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert ".lloyd-data-root" in proc.stdout, proc.stdout
    assert (data / "false").exists(), "an unmarked root was deleted in"


def test_the_environment_route_refuses_a_root_with_no_marker(tmp_path: Path):
    """The first resolver rule, reached with no flag at all — the shape cron actually runs.

    `LLOYD_DATA` is the resolver's top rule, and a scheduled job has it set while a person at a
    shell usually does not, so both rules have to gate on the marker rather than on the
    variable being present. This is the case where the marker is missing AND `--tree` was never
    typed: everything came from the environment, and the answer is still exit 2.
    """
    data = tmp_path / "env-data"
    data.mkdir()
    (data / "sqlite3").write_bytes(b"")
    proc = _run_no_data_root(None, "--apply", "sqlite3",
                             state_dir=str(tmp_path / "state"),
                             with_lloyd_data=str(data))
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert ".lloyd-data-root" in proc.stdout, proc.stdout
    # The refusal names the root it refused, so the operator can see the typo: this is the
    # difference between "I looked in your directory and declined" and a silent exit 2.
    assert str(data) in proc.stdout, proc.stdout
    assert (data / "sqlite3").exists(), "an unmarked root was deleted in"


def test_the_resolver_refuses_an_unmarked_production_root_instead_of_answering(tmp_path: Path):
    """The property this program depends on from `app.data_root`: a refusal is a raise, not a value.

    `app.paths.DATA_ROOT` always answers, including in the state the watchdog exists to alert on
    — a production root that lost its marker — and a writer that borrowed that constant would
    delete in `<home>/lloyd-data` on a guess. `resolve_data_root_for_tree` raises
    `DataRootMissing` there instead, which is why `run()` wraps the call in exit 2. The module
    exposes the three rules as a pure function precisely so a test can stand in a state no
    checkout on this machine is in — a production tree whose root has no marker — so that is
    what is called here, with the live checkout as both home and checkout and an unmarked
    production root. `data_root_for_tree` is asserted in the same breath because it is rule 3,
    the branch the `--tree` node above relies on to land under `<tree>/.lloyd-data`.
    """
    import app.data_root as DR

    live = DR.live_checkout()
    unmarked = tmp_path / "lloyd-data"
    unmarked.mkdir()
    with pytest.raises(DR.DataRootMissing):
        DR.resolve_data_root(env=None, lloyd_home=live, is_worktree=False,
                             live_checkout=live, production_root=unmarked)
    # The same tree WITH the marker answers with that root: the refusal above is about the
    # marker, not about the checkout, so the two nodes together pin rule 2 in both directions.
    (unmarked / DR.DATA_ROOT_MARKER).write_text("marked\n", encoding="utf-8")
    assert DR.resolve_data_root(env=None, lloyd_home=live, is_worktree=False,
                                live_checkout=live, production_root=unmarked) == unmarked
    # Rule 3, the branch `--tree` takes for a tree that is not the production checkout.
    assert DR.data_root_for_tree(tmp_path / "checkout") == tmp_path / "checkout/.lloyd-data"


def test_no_guardian_route_reaches_the_remover_or_deletes_a_data_root_name():
    """The report-only contract of #3109, stated from the remover's side.

    That this program is safe is not what keeps the watchdog from using it — the watchdog has
    no route to it, and this node pins the two ways a route could appear. `guardian.py`
    already shells out to scripts elsewhere in the repo, so a `subprocess` naming this file
    would be a deletion the watchdog performed on its own initiative: exactly the decision
    #3109 reserved for a person. And `datawatch.py` already deletes — its `quarantine_inert`
    unlinks into `quarantine/tree-strays` — so a bare "no unlink in the guardian" claim would
    be false and would have to be weakened to survive; what holds is the DISTINCTION, which
    the third assertion states by naming the one deletion the guardian does have and the one
    direction it moves. Clause 5's own half of this lives in `tests/test_guardian_datawatch.py`
    as the surviving report-only and needs_human nodes.
    """
    guardian_dir = REPO / "agent-services" / "guardian"
    for py in sorted(guardian_dir.glob("*.py")):
        src = py.read_text(encoding="utf-8")
        # `guardian.py` is unchanged by this round, which is why its share of the clause is
        # asserted from a run rather than a diff line: `tests/test_guardian_datawatch.py::
        # test_the_guardians_only_mover_moves_code_tree_residue_into_the_root`
        # reads it, and this loop reads every module in the directory, changed or not.
        assert "clear_data_root_cruft" not in src, (
            f"{py.name} names the remover, so the watchdog can reach a data-root unlink")
    dw = (guardian_dir / "datawatch.py").read_text(encoding="utf-8")
    # The prune is allowed to exist here — it edits the guardian's OWN state file, which is
    # what clause 4 asks for — but it may not be given a path, and no name may be unlinked.
    assert "prune_arrivals_baseline" in dw, "clause 4's prune must live in the guardian module"
    assert "unlink" in dw, (
        "the node's own premise: datawatch already deletes somewhere, or the claim below is "
        "trivially true and guards nothing")
    assert "clear_data_root_cruft" not in dw
    assert "scripts/maintenance" not in dw, (
        "a subprocess path to the remover would be a guardian that deletes a data-root name")
    # The one deletion the guardian has: code-tree residue moved INTO the data root, never
    # a data-root name removed from it. That direction is the whole of #3109's contract.
    assert "def quarantine_inert(tree" in dw
    assert "def prune_arrivals_baseline" in dw
    prune_src = dw.split("def prune_arrivals_baseline", 1)[1].split("\n\n\ndef ", 1)[0]
    for verb in ("os.unlink", "os.remove", "os.rename", "shutil.move", "quarantine_inert("):
        assert verb not in prune_src, (
            f"the prune gained `{verb}`: it edits a JSON state file and nothing else, and "
            "gaining a file deletion is how clause 5's contract is lost through clause 4")
