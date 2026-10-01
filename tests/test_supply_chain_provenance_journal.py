"""#1839: the install-provenance decision journal.

`check_install_provenance` decided in log lines, and log lines do not survive:
`server.err` + `.1..10` is 11 × ~10 MB spanning 2.6 days as measured on 2026-09-29,
and a busy automod round closes one in minutes. The deny branch is worse than
rotated — it emits no `lloyd-supply-chain` line at all, so a refusal exists only as
the caller's boundary line with the command cut to an 80-char excerpt and the failed
registry fact buried in label text that no later reader can group by distribution.
The trend #688 step 6 judges ("blocks per week, overrides used") had no source.

These nodes read the journal off disk. Nothing here consults a log, because the
point of the file is that it is the surface that outlives the logs; a node that
reached for `caplog` to find a decision would be testing the thing this replaces.

The registry is always injected, as in the sibling files: the gate runs with no
outbound access, and a live GET is a flake with a network in it. Every node points
`LLOYD_SUPPLY_CHAIN_CACHE_DIR` at its own `tmp_path`, which relocates the journal
with the cache — the same single override the code uses.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import pytest

from app.harness import safety, supply_chain as sc
from app.harness.supply_chain import (
    JOURNAL_OUTCOMES, MappingRegistry, RegistryFacts, check_install_provenance,
)

BACKGROUND = "20260924_054120_autonomy_688"
NOW = dt.datetime(2026, 9, 24, 12, tzinfo=dt.timezone.utc)
#: `httpx` is in the dependency set the fixtures pass, so it is the name that must
#: read `declared` — allowed by the declaration, never looked up.
DEPENDENCY_SET = {"httpx": "requirements.txt"}


@pytest.fixture
def journal(monkeypatch, tmp_path):
    """The journal inside this test's own scratch dir, relocated by the one override.

    Also clears `LLOYD_OSV_API_URL`, so nothing in the guard reaches for a live
    advisory endpoint while a test is deciding provenance.
    """
    monkeypatch.setenv("LLOYD_SUPPLY_CHAIN_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv(sc.OSV_API_URL_ENV, raising=False)
    return tmp_path / sc.PROVENANCE_JOURNAL_NAME


def _rows(path: Path) -> list[dict]:
    """Every row in the journal. Asserts the file exists, because a missing file and
    an empty file are the same 0 rows to a `len()` assertion, and only one of them
    means "this decision wrote nothing".
    """
    assert path.exists(), f"no journal at {path}: the decision wrote nothing at all"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _registry(**facts):
    """`_registry(graphy=("missing",), freshy=(True, "2018-01-01T00:00:00+00:00", 1))`."""
    payload = {}
    for name, spec in facts.items():
        if spec[0] == "missing":
            payload[name] = RegistryFacts(name=name, exists=False, source="test",
                                          observed_at=NOW)
        else:
            exists, first, count = spec
            payload[name] = RegistryFacts(
                name=name, exists=exists,
                first_release=dt.datetime.fromisoformat(first).replace(
                    tzinfo=dt.timezone.utc) if first else None,
                release_count=count, source="test", observed_at=NOW)
    return MappingRegistry(payload, name="test")


def _check(command, registry, *, session=BACKGROUND):
    return check_install_provenance(command, session, registry=registry,
                                    dependency_set=DEPENDENCY_SET, now=NOW)


# ── clause 1: one row per decision, nothing for a command that installs nothing ──

def test_one_row_per_unattended_decision_carries_the_stamp_class_and_one_entry_per_name(
        journal):
    """The row's own shape, read back as JSON: one line per call, an offset-bearing
    UTC stamp, the session class, and one entry per parsed name drawn from the five
    legal outcomes.

    `pip install httpx graphy` is deliberately two outcomes in one command: `httpx`
    is declared so nothing looked it up, `graphy` was looked up and refused. One row
    with two entries is the only shape that says both "exactly one line per call" and
    "one entry per name" — two rows would satisfy the second half and break the first.
    """
    before = dt.datetime.now(dt.timezone.utc)
    _check("pip install httpx graphy", _registry(graphy=("missing",)))

    rows = _rows(journal)
    assert len(rows) == 1, f"one decision must be one line, got {len(rows)} rows"
    row = rows[0]
    assert row["session_class"] == "unattended", row
    assert row["session"] == BACKGROUND, row

    stamp = dt.datetime.fromisoformat(row["at"])
    assert stamp.tzinfo is not None and stamp.utcoffset() == dt.timedelta(0), (
        f"the stamp {row['at']!r} carries no UTC offset, so a later reader has to "
        "guess whether it was written at noon UTC or noon local")
    assert before <= stamp <= dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=5), (
        f"the stamp {row['at']!r} is not this decision's time")

    by_name = {e["name"]: e["outcome"] for e in row["names"]}
    assert by_name == {"httpx": "declared", "graphy": "denied"}, row
    assert set(by_name.values()) <= set(JOURNAL_OUTCOMES), by_name
    entry = row["names"][1]
    assert entry["ecosystem"] == "pypi" and entry["verb"] == "pip install", entry


def test_a_command_that_parses_no_install_request_appends_nothing(journal):
    """Clause 1's other half, and the false-positive half of the whole guard: prose
    that mentions an install is not a decision, so it must not become a row.

    These are the same three commands the dispatch-seam file uses for the matcher's
    non-firing half; a journal that filled up with `grep` lines would make the trend
    it exists to measure meaningless.
    """
    for command in ["grep -rn 'pip install graphy' docs/",
                    "echo 'remember to pip install graphy later'",
                    "sed -i 's/pip install old/pip install graphy/' setup.sh"]:
        result = _check(command, _registry())
        assert not result.refusals and not result.cleared, command
    assert not journal.exists(), (
        "a command that parsed no install request still wrote a row, so the file "
        "counts prose about installs as decisions")


def test_an_attended_turn_writes_no_row(journal):
    """The attended case, decided rather than stumbled into (owed ruling 3 on the
    item). A chat turn returns at the session-class branch before any registry fact
    exists, so a row for it would carry a name with no outcome to record — and would
    put a person's own terminal commands into a durable file nobody asked for.

    Pinning the absence is what keeps the ruling open: without it, the next pass
    that moves the journal write above that branch would answer the privacy question
    by accident.
    """
    _check("pip install graphy", _registry(graphy=("missing",)), session="chat-abc123")

    assert not journal.exists(), (
        "an attended turn now writes a row; that is owed ruling 3's decision to make, "
        "not this file's to take silently")


# ── clause 2: the override reason, verbatim, with the dispatch still cleared ─────

def test_an_override_still_clears_the_dispatch_and_the_row_carries_the_reason_verbatim(
        journal, monkeypatch):
    """Driven through `safety.check_bash_command`, the boundary every Bash call
    crosses, exactly as `test_supply_chain_dispatch_seam.py` drives it — so this is
    the shipped seam writing the file, not a helper calling a helper.

    Both halves of the override are graded together: the command must still get
    through (a journal that turned an override into a refusal would be a rollback),
    and the row must carry the reason byte for byte. Truncating it here is how a
    trend ends up counting overrides nobody can justify afterwards.
    """
    def fake_lookup(self, name):  # noqa: ANN001 — patching the fetch, as the seam file does
        return RegistryFacts(name=name, exists=False, source="test", observed_at=NOW)
    monkeypatch.setattr(sc.PypiRegistry, "lookup", fake_lookup)

    out = safety.check_bash_command(
        'LLOYD_DEP_OVERRIDE="new solver for the routing eval" pip install graphy',
        session_id=BACKGROUND)

    assert out is None, f"the override stopped clearing dispatch: {out}"
    rows = _rows(journal)
    assert len(rows) == 1, rows
    entries = [e for e in rows[0]["names"] if e["name"] == "graphy"]
    assert len(entries) == 1, rows[0]["names"]
    assert entries[0]["outcome"] == "overridden", (
        f"an overridden install also sits in `cleared`; the row must report the "
        f"decision that needed a reason, not the fact it was let through: {entries[0]}")
    assert entries[0]["override"] == "new solver for the routing eval", entries[0]


def test_an_override_that_was_not_needed_still_becomes_a_row(journal):
    """The reflexive-override habit step 6's trend is about. `httpx` is declared, so
    nothing needed the override — `check_install_provenance` records it as
    `("not-needed", reason)` and the journal keeps that entry under the same spelling.
    Folding these into "nothing happened" is what makes the habit invisible.
    """
    _check('LLOYD_DEP_OVERRIDE="belt and braces" pip install httpx', _registry())

    entries = {e["name"]: e for e in _rows(journal)[0]["names"]}
    assert entries["httpx"]["outcome"] == "declared", entries
    assert entries["not-needed"]["outcome"] == "overridden", entries
    assert entries["not-needed"]["override"] == "belt and braces", entries


# ── clause 3: a denial's failed registry fact is readable from the file ──────────

def test_a_denied_name_keeps_its_failed_registry_fact_in_the_file(journal):
    """The reason a refusal can be audited with no log access, and the two kinds of
    denial in one command: `graphy` is not published at all, `freshy` exists but has
    one release. Each entry must carry its OWN fact and its OWN reason — one shared
    "policy denied" would make the count of blocks per fact unrecoverable, which is
    the failure this file exists to fix.

    Before #1839 the deny branch emitted no log line whatsoever, so this row is not
    a second copy of a witness; it is the only one.
    """
    _check("pip install graphy freshy",
           _registry(graphy=("missing",),
                     freshy=(True, "2018-01-01T00:00:00+00:00", 1)))

    entries = {e["name"]: e for e in _rows(journal)[0]["names"]}
    assert set(entries) == {"graphy", "freshy"}, entries
    for name in entries:
        assert entries[name]["outcome"] == "denied", entries[name]
    assert entries["graphy"]["fact"] == "not-published", entries["graphy"]
    assert "graphy" in entries["graphy"]["reason"], entries["graphy"]
    assert "not published on pypi.org" in entries["graphy"]["reason"], entries["graphy"]
    assert entries["freshy"]["fact"] == "too-few-releases", entries["freshy"]
    assert "graphy" not in entries["freshy"]["reason"], (
        f"the two refusals share a reason string, so per-fact counts are impossible: "
        f"{entries['freshy']['reason']}")


def test_a_denial_still_reaches_the_caller_unchanged_by_the_journal(journal):
    """Writing the row must not change the answer. Same command, same injected
    registry, refusal text identical to the caller's — the journal is a record of the
    decision, never an input to it.
    """
    with_journal = _check("pip install graphy", _registry(graphy=("missing",)))

    assert with_journal.refusal and "graphy" in with_journal.refusal
    assert _rows(journal)[0]["names"][0]["reason"] == with_journal.refusal, (
        "the journal and the caller report different reasons for one refusal")


# ── clause 4: the journal is fail-open like the rest of the guard ───────────────

def test_an_unwritable_journal_costs_a_row_and_not_the_command(monkeypatch, tmp_path,
                                                                caplog):
    """The guard's contract is that it never becomes an outage, and the journal is
    now part of the guard.

    "Unwritable" here is a regular FILE standing where the journal's directory has to
    be, not a `chmod`: root ignores a mode bits request, so a permission-based fixture
    passes on this box and fails on a machine that honours it, and a test that cannot
    fail is worse than no test.
    """
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory\n", encoding="utf-8")

    good_env = tmp_path / "good"
    monkeypatch.setenv("LLOYD_SUPPLY_CHAIN_CACHE_DIR", str(good_env))
    monkeypatch.delenv(sc.OSV_API_URL_ENV, raising=False)
    registry = _registry(graphy=("missing",))
    writable = _check("pip install graphy", registry)

    monkeypatch.setenv("LLOYD_SUPPLY_CHAIN_CACHE_DIR", str(blocked / "sub"))
    with caplog.at_level(logging.WARNING, logger="lloyd-supply-chain"):
        unwritable = _check("pip install graphy", registry)

    assert unwritable == writable, (
        "the same command decided differently depending on whether a file could be "
        f"written: {unwritable!r} vs {writable!r}")
    assert (unwritable.refusals and not unwritable.cleared) == (
        writable.refusals and not writable.cleared), (
        "both results refused, but not the same name for the same reason")
    assert unwritable.overrides == writable.overrides == [], (
        "the refusal carried an override entry, so this node is not comparing what it "
        "claims to")
    assert unwritable.refusal and "graphy" in unwritable.refusal, (
        "a broken journal must not silence the refusal either")
    assert not (blocked / "sub").exists(), (
        "the guard created the directory it could not write into, which is the "
        "side effect a fail-open write is not supposed to have")
    assert (good_env / sc.PROVENANCE_JOURNAL_NAME).exists(), (
        "the writable half of this node never wrote anything either, so the "
        "equality above compares two runs that both did nothing")
    assert any("provenance journal write failed" in r.getMessage()
               for r in caplog.records), (
        "the loss was swallowed whole; a guard whose journal breaks silently is a "
        "guard whose records stop existing with nobody noticing")


def test_a_journal_that_cannot_build_its_row_loses_the_row_and_not_the_dispatch(
        monkeypatch, tmp_path, caplog):
    """Clause 4's other failure surface: the projection, not the write.

    Half one is the tripwire for a future branch of `check_install_provenance` that
    invents a sixth outcome — the row is still written and the odd outcome is named in
    the log, because suppressing it would hide the bug the log line exists to report.
    Half two is the one that matters for dispatch: a projection that raises has to be
    caught by the same `try` that guards `mkdir`, or a journal bug becomes a crash on
    every Bash call in a background session.
    """
    monkeypatch.setenv("LLOYD_SUPPLY_CHAIN_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv(sc.OSV_API_URL_ENV, raising=False)
    path = tmp_path / sc.PROVENANCE_JOURNAL_NAME

    monkeypatch.setattr(sc, "journal_entries",
                        lambda result: [{"name": "graphy", "outcome": "shrugged"}])
    with caplog.at_level(logging.WARNING, logger="lloyd-supply-chain"):
        sc._journal_decision("pip install graphy", BACKGROUND, "unattended",
                             sc.ProvenanceResult())
    rows = _rows(path)
    assert rows[0]["names"] == [{"name": "graphy", "outcome": "shrugged"}], rows
    assert any("outside" in r.getMessage() and "shrugged" in r.getMessage()
               for r in caplog.records), (
        "an outcome outside the five was journalled silently, so the day a branch "
        "invents one the trend just starts reading wrong with no trace")

    def _boom(result):
        raise KeyError("outcome")
    monkeypatch.setattr(sc, "journal_entries", _boom)
    before = len(_rows(path))
    with caplog.at_level(logging.WARNING, logger="lloyd-supply-chain"):
        sc._journal_decision("pip install graphy", BACKGROUND, "unattended",
                             sc.ProvenanceResult())
    assert len(_rows(path)) == before, "a raising projection still wrote a row"
    assert any("provenance journal write failed" in r.getMessage()
               for r in caplog.records), (
        "the projection raised and nothing said so; the record stops existing quietly")


# ── clause 5: CLAUDE.md names the file the counts come from ─────────────────────

def test_claude_md_names_the_journal_as_the_source_of_the_counts():
    """The paragraph used to say an override "is recorded per distribution name, so
    the overrides are countable rather than folklore". What recorded them was one
    `logger.warning` inside a 2.6-day rotation window, and denials were not recorded
    at all — a claim about a surface that did not exist.

    So the doc must name the file, and it must no longer claim the env var alone
    makes anything countable. Both halves: naming the path and dropping the old
    assertion, which is the alarm-that-outlives-its-fix shape.
    """
    doc = (Path(__file__).resolve().parents[1] / "CLAUDE.md").read_text(encoding="utf-8")
    paragraph = [p for p in doc.split("\n\n") if "LLOYD_DEP_OVERRIDE" in p]
    assert paragraph, "the install-provenance paragraph is gone from CLAUDE.md"
    text = "\n".join(paragraph)

    assert "supply-chain/provenance.jsonl" in text, (
        "the doc still does not name the file the counts actually come from, so the "
        "next reader looks for overrides in a log that rotated three days ago")
    assert "so the overrides are countable rather than folklore" not in text, (
        "the sentence that made the env var itself sound like the record is still "
        "standing next to the new one")


def test_the_path_the_doc_names_is_the_path_the_code_writes(monkeypatch):
    """The doc and the code resolve the same file with no override set — the doc says
    `$DATA_ROOT/supply-chain/provenance.jsonl`, and this asserts that is what
    `provenance_journal_path()` returns, not just that both strings appear somewhere.

    A doc that names a plausible file in the right shape and is wrong by one directory
    is worse than no doc, because it reads like an instruction that was followed.
    """
    from app.paths import DATA_ROOT

    monkeypatch.delenv("LLOYD_SUPPLY_CHAIN_CACHE_DIR", raising=False)

    assert sc.provenance_journal_path() == (
        Path(DATA_ROOT) / "supply-chain" / sc.PROVENANCE_JOURNAL_NAME), (
        f"the guard writes {sc.provenance_journal_path()}, which is not the "
        "$DATA_ROOT/supply-chain/ file CLAUDE.md points a reader at")


# ── #1956: one row per decision, and a lockfile is one entry ─────────────────

def test_the_hook_and_the_dispatch_of_one_call_write_one_row(journal):
    """A Bash call reaches the guard twice — PreToolUse hook, then dispatch,
    in two processes about 11 ms apart — and each wrote a row: 15 of the 19
    decisions in the live journal on 2026-10-01 were there twice. Driven through
    the boundary both callers use, once as each."""
    command = "pip install httpx"
    for at_dispatch in (False, True):
        assert safety.check_bash_command(command, session_id=BACKGROUND,
                                         at_dispatch=at_dispatch) is None
    rows = _rows(journal)
    assert len(rows) == 1, [r["command"] for r in rows]
    assert rows[0]["session"] == BACKGROUND and rows[0]["command"] == command


def test_a_denied_command_still_has_exactly_one_row(journal):
    """Why the latch and not "write from dispatch only": a hook denial never
    reaches dispatch, so that route would stop journalling denials at all."""
    registry = _registry(graphy=("missing",))
    result = _check("pip install graphy", registry)
    assert result.refusals, "the premise: this command is denied"
    rows = _rows(journal)
    assert len(rows) == 1 and rows[0]["names"][0]["outcome"] == "denied"
    _check("pip install graphy", registry)       # the model retrying at once
    assert len(_rows(journal)) == 1


def test_the_same_command_run_again_later_is_a_new_decision(journal, monkeypatch):
    """The latch is a window, not a set: past it, an identical command is a
    second decision and gets its own row. And inside it, a different session or
    a different command is never folded."""
    registry = _registry()
    t0 = dt.datetime(2026, 10, 1, 12, tzinfo=dt.timezone.utc)

    def write(command, session, seconds):
        result = sc.ProvenanceResult()
        result.declared.extend(sc.find_install_commands(command))
        sc._journal_decision(command, session, "unattended", result,
                             now=t0 + dt.timedelta(seconds=seconds))

    write("pip install httpx", BACKGROUND, 0)
    write("pip install httpx", BACKGROUND, 0.011)
    assert len(_rows(journal)) == 1
    write("pip install httpx", BACKGROUND, sc.JOURNAL_LATCH_SECONDS + 1.5)
    assert len(_rows(journal)) == 2, "more than 60 s apart is a second decision"
    write("pip install httpx", "20260924_054120_autonomy_999", sc.JOURNAL_LATCH_SECONDS + 2)
    write("pip install rich", BACKGROUND, sc.JOURNAL_LATCH_SECONDS + 2)
    assert len(_rows(journal)) == 4
    assert sc.JOURNAL_LATCH_SECONDS == 60.0


def test_a_requirements_file_is_one_entry_with_a_count(journal, tmp_path, monkeypatch):
    """`pip install -r <lockfile>` wrote one entry per name: 2,168 of the live
    journal's 2,191 entries, 180 to a row. The parser still expands the file —
    every name is decided on its own — and the journal names the file once.

    The size budget is over the row with the lockfile's path cut out of it, because
    that path is the only part of the row the environment controls, and the row
    carries it three times: in `command`, as the collapsed entry's `name`
    (`-r <path>`), and as its `file`. Measuring the whole row made the ceiling
    `600 + 3 × how deep tmp_path is`, and the same commit cleared it in one tree and
    missed it in another — 500 bytes under `/tmp`, 647 under the gate's per-round
    `TMPDIR` (`~/lloyd-work/.t/<10 hex>`, capped at 48 bytes by `MAX_CHILD_TMPDIR` in
    `scripts/automod/gate.py`). That is #2031: red at the gate and green on the same
    commit anywhere else, from the day #1956 landed this node. With the path stripped
    the row is 266 bytes in both trees, so the budget is a property of the shape and
    not of the temp dir — and it still bites on the thing it was always about: with
    the fold taken back out these 40 names are a row of 4,067 path-free bytes, and
    `assert 4067 < 600` is what the node prints (verified 2026-10-01 by disabling the
    fold, in both temp roots — the un-collapsed entries each carry the path too, and
    stripping it discounts all 43 copies, so both figures are the same number in both
    trees)."""
    names = [f"pkg{i}" for i in range(40)]
    req = tmp_path / "requirements.lock"
    req.write_text("\n".join(f"{n}==1.0" for n in names) + "\n")
    command = f"pip install -r {req}"

    parsed = sc.find_install_commands(command)
    assert [r.name for r in parsed] == names, "the parser still yields every name"
    assert {r.source for r in parsed} == {str(req)}

    result = check_install_provenance(
        command, BACKGROUND, registry=_registry(), now=NOW,
        dependency_set={n: "requirements.lock" for n in names})
    assert [r.name for r in result.declared] == names, (
        "the decision is per name and unchanged")

    row = _rows(journal)[0]
    # First, so a regression that stops folding prints this rather than the shape
    # asserts below it. `command` is truncated to 200 chars, so on an absurdly deep
    # `tmp_path` one copy can survive the strip short and add at most the ~200 bytes
    # of the cut; the row's path-free 266 plus that remainder is still half the
    # ceiling, so the budget holds wherever the temp dir is.
    assert len(json.dumps(row).replace(str(req), "")) < 600, (
        "the row outgrew its budget once the lockfile's path is discounted, so what "
        "grew is the entries, not the temp dir: 40 names used to be 40 entries")
    assert len(row["names"]) == 1, row["names"]
    entry = row["names"][0]
    assert entry["file"] == str(req) and entry["count"] == 40
    assert entry["outcome"] == "declared" and entry["name"] == f"-r {req}"


def test_a_denied_name_from_a_requirements_file_keeps_its_own_entry(journal, tmp_path):
    """The collapse never hides the one thing worth a row: a name the registry
    refused is written by name, with its fact, beside the file's count."""
    req = tmp_path / "requirements.txt"
    req.write_text("httpx==0.27\ngraphy==0.1\n")
    result = _check(f"pip install -r {req} rich", _registry(
        graphy=("missing",), rich=(True, "2018-01-01T00:00:00+00:00", 40)))
    assert [r.name for r, _v in result.refusals] == ["graphy"]

    by_name = {e["name"]: e for e in _rows(journal)[0]["names"]}
    assert by_name[f"-r {req}"]["count"] == 1 and by_name[f"-r {req}"]["outcome"] == "declared"
    assert by_name["graphy"]["outcome"] == "denied" and by_name["graphy"]["fact"]
    assert by_name["graphy"]["file"] == str(req)
    assert by_name["rich"]["outcome"] == "cleared" and "count" not in by_name["rich"]
