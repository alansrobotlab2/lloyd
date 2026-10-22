"""The memory epoch in the request manifest (#3168).

SOUL.md, MEMORY.md and USER.md are three files that `build_system_prompt` reads
with three separate `read_text` calls and no lock between them — SOUL.md first, in
`prompt_builder._load_soul`, then the members `_memory_files_for(platform)` selects
one per iteration in `prompt_builder._load_memories` — while
`harness.prompt_layout.freeze_memory` is false, so a vault write can land in the
gap. Each file is individually pristine
(the vault route writes through `atomic_write_text` + `os.replace`), so nothing
could tell that a turn's SOUL.md came from one instant and its MEMORY.md from the
next, and the string SOUL.md holds is what `_memory_duplicate_share` compares
MEMORY.md against: a tear there changes WHICH LINES get dropped from the prompt,
not only which bytes were served.

This file pins what makes that countable:

* **clause 1** — a manifest line carries one `memory_epoch` entry per member the
  build actually read for that platform, each with the `sha256` and `bytes` of the
  bytes as read (`test_a_build_stamps_one_epoch_entry_per_memory_member_it_read`,
  `test_the_epoch_carries_only_the_members_this_platform_loads`,
  `test_the_epoch_reaches_every_iteration_line_of_the_turn`);
* **clause 2** — each entry carries the member's `mtime_ns` and `size` from both
  sides of the read, and `torn: true` when the two differ
  (`test_the_entry_carries_the_stat_reading_from_both_sides_of_the_read`,
  `test_a_member_replaced_mid_read_is_stamped_torn`,
  `test_a_torn_read_reaches_the_store_marked_torn`);
* **clause 3** — the entries carry digests and numbers and no memory text, even
  against a caller that tried
  (`test_a_rogue_epoch_entry_cannot_bring_memory_text`; the sentinel node in
  `tests/test_component_manifest.py` is the other half of this clause);
* **clause 4** — `memory_epoch_report` counts per session at TURN granularity, so
  111 iteration lines of one user turn are one turn, and it reports both the turns
  whose epoch differs from the session's first turn's epoch and the turns carrying a
  torn member. Its nodes, all of them, since this is the clause a reader is most
  likely to want to check by name:
  `test_one_user_turn_spanning_many_iterations_counts_as_one_turn` (the 111-lines-
  are-one-turn case itself, plus the epoch-change-per-session-not-per-line count),
  `test_a_session_whose_epoch_never_moved_reports_zero_not_noise`,
  `test_the_reader_names_the_torn_member_and_counts_its_turn_once`,
  `test_an_epoch_that_moves_inside_one_turn_is_reported_as_movement`,
  `test_a_line_from_before_the_epoch_landed_is_counted_and_not_guessed`,
  `test_the_reader_bounds_by_day_file_and_sums_across_sessions`,
  `test_a_line_with_no_session_id_is_not_folded_into_someone_elses_session`,
  `test_lines_that_give_no_turn_boundary_signal_are_counted_as_a_floor`.

Against the commit before this diff every node here fails at the first assertion
or earlier: `git grep -n "memory_epoch" -- '*.py'` in `~/lloyd` returned zero
hits at triage on 2026-10-22 — no field, no `note_components` parameter to carry
it, no reader to count it.

What this file does NOT pin, because the item gates it on the number this
instrument produces: the re-scoping of `freeze_memory` from per-session-forever to
per-identity-epoch, and giving `memory_delta_note` the text of removed lines
(`app/memory_snapshot.py:93-108` still prints `-{n} lines`). Both are recorded as
owed on #3168, deliberately out of this diff: a prompt-content change riding in
with an instrumentation change is how rounds get refused.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from app import component_manifest as cm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

#: Frontmatter on SOUL.md is deliberate. `_load_soul` strips it, so the epoch's
#: digest of the raw bytes and the component's digest of the processed string
#: provably differ — which is what makes the epoch a record of the FILE rather
#: than a second copy of a number `components` already carries.
SOUL_TEXT = ("---\ntype: note\n---\n\n"
             "# SOUL\n\nSOUL-EPOCH-SENTINEL-3d41 identity frame.\n")
MEMORY_TEXT = "# MEMORY.md\n\nMEMORY-EPOCH-SENTINEL-9ab2 index line.\n"
USER_TEXT = "# USER.md\n\nUSER-EPOCH-SENTINEL-7c55 facts about the user.\n"


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    """A store of our own, and clean module state around every test."""
    monkeypatch.setenv("LLOYD_MANIFEST_STORE", str(tmp_path / "manifest-store"))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    cm.reset_stats()
    cm._reset_registry()
    cm._reset_template_cache()
    cm._reset_tools_memo()
    yield tmp_path / "manifest-store"
    cm.flush(timeout=5.0)
    cm.reset_stats()
    cm._reset_registry()
    cm._reset_template_cache()
    cm._reset_tools_memo()


def _surface(tmp_path: Path) -> Path:
    """An overlay holding all three memory-surface members, sentinels included.

    An overlay rather than the live vault: the vault's own bytes change every day
    by design, and a test that reads them can go red on a memory write that has
    nothing to do with this code.
    """
    overlay = tmp_path / "prompts"
    overlay.mkdir(parents=True, exist_ok=True)
    (overlay / "SOUL.md").write_text(SOUL_TEXT, encoding="utf-8")
    (overlay / "MEMORY.md").write_text(MEMORY_TEXT, encoding="utf-8")
    (overlay / "USER.md").write_text(USER_TEXT, encoding="utf-8")
    return overlay


def _build(session_id: str, overlay: Path, **kw) -> dict:
    """Run a real build and hand back the registry entry it left behind."""
    from app import prompt_builder

    prompt_builder.build_system_prompt(session_id=session_id, overlay_dir=overlay,
                                       session_state="system_head", **kw)
    return cm.components_for(session_id)


def _record(session_id: str, *, iteration: int = 1, turns: int = 1) -> dict:
    """One manifest line for `session_id`, `turns` user messages deep.

    Raises rather than returning None: every caller subscripts the line, and a
    node whose line came back None would fail on a `TypeError` about NoneType
    instead of on the assertion it is actually making.
    """
    messages = [{"role": "user", "content": f"turn {n}"} for n in range(1, turns + 1)]
    line = cm.record_request(
        base_url="http://stub:8000", model="m", session_id=session_id,
        iteration=iteration, send_site="tests/test_component_manifest_memory_epoch.py",
        payload={"model": "m", "messages": messages})
    assert line is not None, (
        "record_request wrote nothing, so every assertion on its line would be "
        "about None rather than about the epoch")
    return line


def _epoch_names(line: dict) -> "list[str]":
    return [row["name"] for row in line.get("memory_epoch") or []]


def _write_lines(store: Path, day: str, lines: "list[dict]") -> None:
    """Put lines in a dated store file directly, bypassing the writer thread.

    The reader under test reads dated files, and the node that bounds it by date
    needs files whose names are not today's — which the writer cannot produce.
    """
    manifests = store / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    with open(manifests / f"{day}.ndjson", "a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(cm.canonical_json(line) + "\n")


def _line(*, session_id: str, iteration: "int | None", turns: int = 1,
          epoch: "list[dict] | None", messages: bool = True) -> dict:
    """A stored line in the shape `record_request` writes, epoch optional.

    `epoch=None` stands for a line written before #3168 landed: the field absent,
    not empty. `turns=0` with `messages=False` is a line whose payload carries no
    message list at all, which is a real shape and not the same thing as an empty
    one — see `test_lines_that_give_no_turn_boundary_signal_are_counted_as_a_floor`.
    """
    line = {"schema": cm.SCHEMA, "ts": 1_760_000_000.0 + (iteration or 0),
            "request_id": f"r{session_id}-{iteration}", "session_id": session_id,
            "iteration": iteration,
            "components_captured": "turn_start"}
    if messages:
        line["messages"] = [
            {"index": n, "role": "user", "sha256": cm.digest_text(f"u{n}"),
             "bytes": 3} for n in range(1, turns + 1)]
    if epoch is not None:
        line["memory_epoch"] = epoch
    return line


def _epoch(marker: str, *, torn: "str | None" = None) -> "list[dict]":
    """A two-member epoch for `MEMORY.md`/`SOUL.md`, one optionally torn."""
    rows = [{"name": "SOUL.md", "sha256": cm.digest_text(f"soul-{marker}"),
             "bytes": 100, "mtime_ns": 1, "size": 100,
             "mtime_ns_after": 1, "size_after": 100, "torn": False}]
    rows.append({"name": "MEMORY.md", "sha256": cm.digest_text(f"mem-{marker}"),
                 "bytes": 200, "mtime_ns": 2, "size": 200,
                 "mtime_ns_after": 2, "size_after": 200,
                 "torn": torn == "MEMORY.md"})
    if torn and torn != "MEMORY.md":
        rows[0]["torn"] = True
    return rows


# ── clause 1: one entry per member the build read ─────────────────────────


def test_a_build_stamps_one_epoch_entry_per_memory_member_it_read(tmp_path):
    """Every line of a turn carries the three members, digested as they were read.

    Read order is the order `build_system_prompt` performs the reads in, so a
    reordered assembly would be visible in the field rather than only in the code.
    """
    overlay = _surface(tmp_path)
    assert _build("stamp", overlay)  # the build ran and noted something
    line = _record("stamp")
    epoch = line["memory_epoch"]
    assert [row["name"] for row in epoch] == ["SOUL.md", "MEMORY.md", "USER.md"]
    for row in epoch:
        raw = (overlay / row["name"]).read_text(encoding="utf-8")
        assert row["sha256"] == cm.digest_text(raw)
        assert row["bytes"] == len(raw.encode("utf-8"))


def test_the_epoch_digests_the_file_not_the_processed_component(tmp_path):
    """The epoch is about the bytes on disk, which is not what `components` hashed.

    SOUL.md carries frontmatter and `_load_soul` strips it, so the two digests of
    SOUL.md must differ. Were they equal, the epoch would be a second copy of a
    number the store already had and could not answer the item's question at all.
    """
    overlay = _surface(tmp_path)
    _build("processed", overlay)
    line = _record("processed")
    soul_epoch = [r for r in line["memory_epoch"] if r["name"] == "SOUL.md"][0]
    soul_component = [r for r in line["components"] if r["name"] == "SOUL.md"][0]
    assert soul_epoch["sha256"] != soul_component["sha256"]
    assert soul_epoch["sha256"] == cm.digest_text(
        (overlay / "SOUL.md").read_text(encoding="utf-8"))
    assert soul_epoch["bytes"] > soul_component["bytes"], "the frontmatter is the difference"


def test_the_epoch_carries_only_the_members_this_platform_loads(tmp_path):
    """A worker build does not read USER.md, so a worker line must not claim it did.

    `_memory_files_for` drops USER.md for the platforms in
    `sessions_io.NON_USER_PLATFORMS`, and the clause says "per member actually
    read" — a field that listed a file the build never opened would be the exact
    false-positive this instrument cannot afford, because its only job is to say
    whether a mixed identity was ever served.
    """
    overlay = _surface(tmp_path)
    assert (overlay / "USER.md").is_file(), "the member exists and is simply not loaded"
    _build("worker-1", overlay, platform="autonomy")
    line = _record("worker-1")
    assert _epoch_names(line) == ["SOUL.md", "MEMORY.md"]

    _build("worker-2", overlay, platform="")          # a user platform, same files
    assert _epoch_names(_record("worker-2")) == ["SOUL.md", "MEMORY.md", "USER.md"]


def test_a_frozen_snapshot_build_carries_an_epoch_of_soul_only(tmp_path):
    """`memories_text` skips the two memory reads, so the epoch must skip them too.

    The epoch records reads, not the state of the vault. A snapshot turn that
    listed MEMORY.md and USER.md would blame an epoch for bytes it never touched.
    """
    overlay = _surface(tmp_path)
    _build("frozen", overlay, memories_text="<memory>\nsnapshot body\n</memory>")
    assert _epoch_names(_record("frozen")) == ["SOUL.md"]


def test_the_epoch_reaches_every_iteration_line_of_the_turn(tmp_path):
    """Built once per turn, stamped on every line of it — the registry handoff.

    This is what makes clause 4's turn collapsing legitimate: the epoch a turn's
    lines carry is one reading, so 111 copies of it are one reading, not 111.
    """
    overlay = _surface(tmp_path)
    _build("iterations", overlay)
    lines = [_record("iterations", iteration=n) for n in (1, 2, 3)]
    first = lines[0]["memory_epoch"]
    assert [l["memory_epoch"] for l in lines] == [first, first, first]
    assert lines[0]["components_captured"] == "turn_start"


def test_a_send_site_line_with_no_registry_entry_carries_no_epoch(tmp_path):
    """The honest absence: a request that never went through the builder.

    Absent, not `[]` — `[]` is a recorded reading meaning "the builder ran and read
    no member", and folding the two together would make a pre-landing line
    indistinguishable from a memoryless turn.
    """
    line = cm.record_request(base_url="http://stub:8000", model="m",
                             session_id="never-built", iteration=1,
                             send_site="tests/test_component_manifest_memory_epoch.py",
                             components={"SOUL.md": "inline body"},
                             payload={"model": "m",
                                      "messages": [{"role": "user", "content": "go"}]})
    assert line is not None, "nothing was written, so the absence below proves nothing"
    assert "memory_epoch" not in line


# ── clause 2: both stat readings, and `torn` when they differ ─────────────


def test_the_entry_carries_the_stat_reading_from_both_sides_of_the_read(tmp_path):
    """Undisturbed, the two readings agree — and both are present anyway.

    Asserted against a fresh `stat` so the numbers are the filesystem's, not the
    module's restatement of itself.
    """
    overlay = _surface(tmp_path)
    _build("quiet", overlay)
    epoch = {r["name"]: r for r in _record("quiet")["memory_epoch"]}
    for name, row in epoch.items():
        st = (overlay / name).stat()
        assert row["mtime_ns"] == st.st_mtime_ns == row["mtime_ns_after"]
        assert row["size"] == st.st_size == row["size_after"]
        assert row["torn"] is False


def test_a_member_replaced_mid_read_is_stamped_torn(tmp_path, monkeypatch):
    """The tear itself, produced by the vault's own writer at the moment of reading.

    `read_text` for MEMORY.md hands back the old bytes and THEN replaces the file
    through `app.atomic_io.atomic_write_text` — the route every memory write in
    production takes — so the `stat` taken after the read sees a different file.
    The prompt keeps the bytes it read; the entry is the only trace that anything
    happened, which is exactly the property the item needed.
    """
    overlay = _surface(tmp_path)
    real_read = Path.read_text

    def tearing_read(self, *args, **kwargs):
        text = real_read(self, *args, **kwargs)
        if self.name == "MEMORY.md":
            from app.atomic_io import atomic_write_text

            atomic_write_text(self, MEMORY_TEXT + "A SECOND LINE, LONGER.\n")
        return text

    monkeypatch.setattr(Path, "read_text", tearing_read)
    _build("torn", overlay)
    epoch = {r["name"]: r for r in _record("torn")["memory_epoch"]}

    assert epoch["MEMORY.md"]["torn"] is True
    assert epoch["MEMORY.md"]["mtime_ns"] != epoch["MEMORY.md"]["mtime_ns_after"]
    assert epoch["MEMORY.md"]["size_after"] > epoch["MEMORY.md"]["size"]
    assert epoch["SOUL.md"]["torn"] is False, "the member read before the tear is clean"
    assert epoch["USER.md"]["torn"] is False


def test_a_torn_read_reaches_the_store_marked_torn(tmp_path, monkeypatch):
    """Same tear, verified in the bytes that were written, not in the returned dict.

    The store is the artefact every later reader has; a `torn` flag that lived only
    on the object `record_request` handed back would count for nothing.
    """
    overlay = _surface(tmp_path)
    real_read = Path.read_text

    def tearing_read(self, *args, **kwargs):
        text = real_read(self, *args, **kwargs)
        if self.name == "USER.md":
            from app.atomic_io import atomic_write_text

            atomic_write_text(self, USER_TEXT + "AND MORE TEXT THAN BEFORE.\n")
        return text

    monkeypatch.setattr(Path, "read_text", tearing_read)
    _build("torn-store", overlay)
    _record("torn-store")
    cm.flush(timeout=5.0)

    day_files = sorted((tmp_path / "manifest-store" / "manifests").glob("*.ndjson"))
    assert day_files, "the store wrote nothing, so this node would pass vacuously"
    lines = [json.loads(raw) for path in day_files
             for raw in path.read_text(encoding="utf-8").splitlines() if raw.strip()]
    row = [r for r in lines[0]["memory_epoch"] if r["name"] == "USER.md"][0]
    assert row["torn"] is True
    assert row["size_after"] > row["size"]


# ── clause 3: no memory text, including from a caller that tried ──────────


def test_a_rogue_epoch_entry_cannot_bring_memory_text(tmp_path):
    """The allow-list is the retention guard, not the caller's good manners.

    `note_components` is called from the same frame that holds the three loaded
    memory strings, so an entry that smuggled text through an unexpected key, an
    over-long `name`, or the `sha256` slot itself has to be stopped where it is
    accepted. `sha256` in particular is validated rather than stringified: a
    caller that put the file body where a digest belongs gets a marker naming its
    type, because a 30 KB paragraph is not a sha256 and truncating one is worse.
    """
    smuggled = "MEMORY-TEXT-IN-THE-EPOCH-4f8e"
    # Typed as a bare `list` on purpose: the point of the node is that a caller can
    # hand over a row that is not a mapping at all, and the annotation the function
    # declares is exactly the promise this test is trying to break.
    rogue: list = [
        {"name": "SOUL.md", "sha256": cm.digest_text("soul"), "bytes": 4,
         "text": smuggled, "content": smuggled,
         "body": {"line_one": smuggled}},
        {"name": "USER.md" + "x" * 200, "sha256": smuggled, "bytes": 4,
         "mtime_ns": "not a number", "size": 7, "torn": 1},
        "not-a-mapping",
    ]
    cm.note_components("rogue", {"SOUL.md": "# soul"}, memory_epoch=rogue)
    line = _record("rogue")
    blob = cm.canonical_json(line)
    assert smuggled not in blob
    epoch = line["memory_epoch"]
    assert len(epoch) == 2, "the non-mapping row is dropped, not stored as a marker"
    assert set(epoch[0]) <= set(cm._EPOCH_KEYS), sorted(epoch[0])
    assert epoch[0]["sha256"] == cm.digest_text("soul")
    assert len(epoch[1]["name"]) == cm._EPOCH_NAME_MAX
    assert epoch[1]["sha256"] == "unhashable:str"
    assert "mtime_ns" not in epoch[1], "a non-numeric mtime stays absent, never becomes 0"
    assert epoch[1]["torn"] is True, "1 is a truthy reading, stored as the boolean it means"
    assert cm.stats()["hash_errors"] >= 1


# ── clause 4: the reader counts turns, not lines ──────────────────────────


def test_one_user_turn_spanning_many_iterations_counts_as_one_turn(tmp_path):
    """111 lines of one turn are one turn, and the epoch change behind them is one.

    The false zero the item's triage measured: the five sessions on 2026-10-22 that
    straddled the window's only memory write were 24, 38, 47, 86 and 111
    `turn_start` lines long, and each was ONE user turn — `build_system_prompt`
    runs once per turn and every iteration of it repeats the same epoch. A reader
    that counted lines would have called that an epoch change 111 times over, or
    zero times over, depending on which naive rule it used; neither number is the
    truth. Here the truth is one turn before the write and one after it.
    """
    before = _epoch("A")
    after = _epoch("B")
    _write_lines(tmp_path / "manifest-store", "2026-10-22", [
        _line(session_id="round", iteration=n, epoch=before) for n in range(1, 112)
    ] + [
        _line(session_id="round", iteration=n, turns=2, epoch=after)
        for n in range(1, 6)
    ])
    rep = cm.memory_epoch_report(store=tmp_path / "manifest-store",
                                 session_id="round")
    assert rep["totals"]["lines"] == 116
    assert rep["totals"]["turns"] == 2, rep["totals"]
    assert rep["totals"]["epoch_changes"] == 1
    row = rep["sessions"][0]
    assert row["turns"] == 2 and row["epoch_changes"] == 1
    assert row["torn_turns"] == 0 and row["lines_without_epoch"] == 0


def test_a_session_whose_epoch_never_moved_reports_zero_not_noise(tmp_path):
    """The number the item is waiting for: turns 2, epoch changes 0.

    A green run of a long round looks exactly like this. It is the answer the
    freeze_memory ruling is gated on, so the shape has to be pinned on the
    negative case as well as the positive one.
    """
    epoch = _epoch("A")
    _write_lines(tmp_path / "manifest-store", "2026-10-22", [
        _line(session_id="steady", iteration=n, epoch=epoch) for n in range(1, 40)
    ] + [
        _line(session_id="steady", iteration=n, turns=2, epoch=epoch)
        for n in range(1, 21)
    ])
    rep = cm.memory_epoch_report(store=tmp_path / "manifest-store")
    assert rep["totals"]["turns"] == 2
    assert rep["totals"]["epoch_changes"] == 0
    assert rep["totals"]["sessions_with_epoch_change"] == 0


def test_the_reader_names_the_torn_member_and_counts_its_turn_once(tmp_path):
    """A torn member is one finding per turn, with the member named.

    Five iteration lines all carrying the same `torn: true` is one torn read, the
    same way five lines carrying the same epoch is one epoch.
    """
    _write_lines(tmp_path / "manifest-store", "2026-10-22", [
        _line(session_id="torn-round", iteration=n,
              epoch=_epoch("A", torn="MEMORY.md")) for n in range(1, 6)
    ])
    rep = cm.memory_epoch_report(store=tmp_path / "manifest-store")
    assert rep["totals"]["turns"] == 1
    assert rep["totals"]["torn_turns"] == 1
    assert rep["totals"]["sessions_with_torn_read"] == 1
    row = rep["sessions"][0]
    assert row["torn_members"] == {"MEMORY.md": 1}, row


def test_an_epoch_that_moves_inside_one_turn_is_reported_as_movement(tmp_path):
    """The mixed-identity serve itself, if it ever happens: one turn, two epochs.

    `note_components` runs once per turn, so two lines of ONE turn describing two
    memory surfaces means something rebuilt the prompt mid-turn and read a
    different memory surface while doing it. That is the event #3168 exists to
    catch, and it is counted as an epoch change too: a turn that moved is not a
    turn that held still.
    """
    _write_lines(tmp_path / "manifest-store", "2026-10-22", [
        _line(session_id="mixed", iteration=1, epoch=_epoch("A")),
        _line(session_id="mixed", iteration=2, epoch=_epoch("A")),
        _line(session_id="mixed", iteration=3, epoch=_epoch("B")),
    ])
    rep = cm.memory_epoch_report(store=tmp_path / "manifest-store")
    assert rep["totals"]["turns"] == 1
    assert rep["totals"]["intra_turn_movement"] == 1
    assert rep["totals"]["epoch_changes"] == 1


def test_a_line_from_before_the_epoch_landed_is_counted_and_not_guessed(tmp_path):
    """Every pre-landing line is this shape, and the reader says how many it saw.

    The item's caveat is that the 40-day window before this stays
    commit-time-approximate permanently. A line with no `memory_epoch` is therefore
    the commonest thing in the store on the day it lands, and counting it as a
    turn with an unchanged epoch would quietly turn the blind spot into a clean
    bill of health.
    """
    _write_lines(tmp_path / "manifest-store", "2026-10-21", [
        _line(session_id="ancient", iteration=n, epoch=None) for n in range(1, 8)
    ])
    rep = cm.memory_epoch_report(store=tmp_path / "manifest-store")
    row = rep["sessions"][0]
    assert row["lines"] == 7 and row["turns"] == 1
    assert row["lines_without_epoch"] == 7
    assert row["epoch"] is None and row["epoch_changes"] == 0
    assert rep["totals"]["sessions_measurable"] == 0


def test_the_reader_bounds_by_day_file_and_sums_across_sessions(tmp_path):
    """Two sessions in two days: `since`/`until` select files, totals sum rows.

    The owed post-landing check is "at least 7 days of the store", so the bound has
    to be the file name's date and not a timestamp inside the line — the retention
    window already ages files by that name, and a second clock would disagree.
    """
    _write_lines(tmp_path / "manifest-store", "2026-10-20", [
        _line(session_id="early", iteration=1, epoch=_epoch("A")),
    ])
    _write_lines(tmp_path / "manifest-store", "2026-10-23", [
        _line(session_id="late", iteration=1, epoch=_epoch("A")),
        _line(session_id="late", iteration=2, turns=2, epoch=_epoch("B")),
    ])
    store = tmp_path / "manifest-store"
    assert cm.memory_epoch_report(store=store, since="2026-10-22")["totals"]["lines"] == 2
    rep = cm.memory_epoch_report(store=store, since="2026-10-20", until="2026-10-23")
    assert [r["session_id"] for r in rep["sessions"]] == ["early", "late"]
    assert rep["totals"]["lines"] == 3
    assert rep["totals"]["turns"] == 3
    assert rep["totals"]["epoch_changes"] == 1


def test_a_line_with_no_session_id_is_not_folded_into_someone_elses_session(tmp_path):
    """An unkeyed sub-call is not a session, and must not be counted as one.

    1,306 of the 7,987 lines in the live store's 2026-10-22 file are one-shot
    `app/secondary_models.py` and `app/inner_voice/observer.py` calls whose
    `session_id` is the empty string, whose `iteration` is null and whose
    `components_captured` is `send_site`. Folding them under the empty key prints
    them as ONE session of ONE turn — a number about nothing, which is the false
    zero this reader exists to avoid, in a costume. They are counted in
    `lines_without_session` instead, and no epoch can ever reach them: the registry
    that carries an epoch is keyed by session id, so `_memory_epoch_sink` returns
    None without one.
    """
    _write_lines(tmp_path / "manifest-store", "2026-10-22", [
        _line(session_id="", iteration=None, epoch=_epoch("A")),
        _line(session_id="", iteration=None, epoch=_epoch("B")),
        _line(session_id="real", iteration=1, epoch=_epoch("A")),
    ])
    rep = cm.memory_epoch_report(store=tmp_path / "manifest-store")
    assert [r["session_id"] for r in rep["sessions"]] == ["real"]
    assert rep["totals"]["lines_without_session"] == 2
    assert rep["totals"]["lines"] == 1, "the unkeyed lines are not silently in the totals"


def test_lines_that_give_no_turn_boundary_signal_are_counted_as_a_floor(tmp_path):
    """Where no boundary can be derived, the turn count is a floor and says so.

    A payload with no message list and no integer `iteration` is the other real
    shape in the store. Folding such lines into the turn already open is the right
    reading of them — they carry nothing that could separate two turns — but it
    would be a lie to present the resulting number as a census, so the lines that
    contributed no signal are counted beside it.
    """
    epoch = _epoch("A")
    _write_lines(tmp_path / "manifest-store", "2026-10-22", [
        _line(session_id="blind", iteration=None, epoch=epoch, messages=False)
        for _ in range(5)
    ])
    rep = cm.memory_epoch_report(store=tmp_path / "manifest-store")
    row = rep["sessions"][0]
    assert row["turns"] == 1
    assert row["lines_without_turn_index"] == 5
    assert rep["totals"]["lines_without_turn_index"] == 5


def test_the_epoch_on_a_line_survives_the_writer_unchanged(tmp_path):
    """The report reads the same epoch the builder wrote, across the writer thread.

    The seam this crosses is a queue and a file: `record_request` hands a line to a
    writer thread that serialises it with `canonical_json`, and the reader
    re-parses it. Both halves are pinned here against the real store — a field that
    only survives in the returned object would satisfy the builder's unit test and
    leave clause 4 counting nothing forever.
    """
    overlay = _surface(tmp_path)
    _build("round-trip", overlay)
    written = _record("round-trip", iteration=1)
    assert cm.flush(timeout=5.0)
    day_files = sorted((tmp_path / "manifest-store" / "manifests").glob("*.ndjson"))
    stored = [json.loads(raw) for path in day_files
              for raw in path.read_text(encoding="utf-8").splitlines() if raw.strip()]
    assert [s["memory_epoch"] for s in stored] == [written["memory_epoch"]]
    rep = cm.memory_epoch_report(store=tmp_path / "manifest-store",
                                 session_id="round-trip")
    assert rep["sessions"][0]["epoch"] == cm.canonical_json(written["memory_epoch"])
    assert rep["totals"]["epoch_changes"] == 0


def test_a_session_whose_epoch_moves_off_its_first_turn_counts_the_move(tmp_path):
    """Clause 4's positive half: the count of turns that DIFFER from turn one.

    The collapse node pins `epoch_changes == 0`, which a reader could reach by never
    counting anything, so this node pins the number moving. Three turns over one
    session: turn one's epoch, a second turn still on it, and a third that came from
    a different memory surface — one changed turn, not the two later lines, and the
    row still names the session's first epoch as the thing it moved off.
    """
    first, later = _epoch("A"), _epoch("B")
    _write_lines(tmp_path / "manifest-store", "2026-10-22", [
        _line(session_id="moved", iteration=1, turns=1, epoch=first),
        _line(session_id="moved", iteration=2, turns=1, epoch=first),
        _line(session_id="moved", iteration=1, turns=2, epoch=first),
        _line(session_id="moved", iteration=1, turns=3, epoch=later),
        _line(session_id="moved", iteration=2, turns=3, epoch=later),
    ])
    rep = cm.memory_epoch_report(store=tmp_path / "manifest-store")
    row = rep["sessions"][0]
    assert row["turns"] == 3, row
    assert row["epoch_changes"] == 1, (
        "turns 2 and 3 are two lines of ONE changed turn; counting lines here is "
        "the false positive the turn fold exists to prevent")
    assert row["epoch"] == cm.canonical_json(first), (
        "the reported baseline is the session's FIRST turn's epoch, which is what "
        "clause 4 compares every later turn against")
    assert rep["totals"]["sessions_with_epoch_change"] == 1
    assert rep["totals"]["epoch_changes"] == 1


def test_a_torn_member_and_a_moved_epoch_are_counted_separately(tmp_path):
    """The two findings the clause asks a reader to report are not one number.

    One session, two turns: the first torn and on epoch A, the second clean on
    epoch B. `torn_turns` and `epoch_changes` must each read 1 — a fold that
    reported a tear as an epoch change (or lost one) would tie them at the same
    value and read as consistent.
    """
    torn = _epoch("A", torn="MEMORY.md")
    _write_lines(tmp_path / "manifest-store", "2026-10-22", [
        _line(session_id="both", iteration=1, turns=1, epoch=torn),
        _line(session_id="both", iteration=1, turns=2, epoch=_epoch("B")),
    ])
    rep = cm.memory_epoch_report(store=tmp_path / "manifest-store")
    row = rep["sessions"][0]
    assert (row["turns"], row["torn_turns"], row["epoch_changes"]) == (2, 1, 1), row
    assert row["torn_members"] == {"MEMORY.md": 1}, row
    assert rep["totals"]["sessions_with_torn_read"] == 1
    assert rep["totals"]["sessions_with_epoch_change"] == 1
