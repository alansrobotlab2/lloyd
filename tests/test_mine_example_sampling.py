"""#1231 — a mined pattern's printed examples must be a rotating sample, not a
frozen first-N payload, and every candidate must name the sample's denominator.

Why this file exists
--------------------
`scripts/mine-trajectories.py` built each pattern's example list with
`if len(examples) < N: append`, so the emitted payload was the first N instances
in scan order forever, while `sessions:`/`occurrences:` and the
`## Sessions Affected` list kept growing cumulatively. `load_trajectories`
iterates a sorted glob, so "first N" meant the *oldest* N in the window.

Measured 2026-09-18 over `_pipeline/skills/candidates/` (901 keys with 2+ dated
snapshots): 625 keys carried a byte-identical evidence section from their oldest
to their newest snapshot, with `sessions` growing median 1.40x and up to 22.2x,
and 611 of them already satisfied the `sessions >= 3` persistence gate at both
ends. `seq-2-read-write` held one sha1 (`8af59f3972`) across all 5 of its
snapshots while its sessions went 38 -> 275, and `seq-2-read-edit` held
`d3a1609e87` from 5 -> 111. The counter was honest — the printed
`## Sessions Affected` bullets equalled `sessions` — it was the *adjudicable
evidence* that was frozen, which is what consolidation's Phase 1.3 "2+ dated
snapshots — persistent, not a one-off" test reads.

Sequence fixtures here carry an error-recovery n-gram (`edit:ERR → read`), not
the plain `write → read` the item measured, because `is_emittable` has
suppressed `has_error_recovery: false` sequences since #1181 — a non-recovering
n-gram is still mined and counted, but `write_candidate_file` writes nothing for
it, so the only sequence whose examples a human ever adjudicates is one with a
recovery step in it.

A file with a dash in its name is not importable, so it is loaded by path (the
same shape `tests/test_trajectory_extraction.py` uses).
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_mspec = importlib.util.spec_from_file_location(
    "mine_trajectories_sampling", _ROOT / "scripts" / "mine-trajectories.py"
)
mt = importlib.util.module_from_spec(_mspec)
_mspec.loader.exec_module(mt)

# The n-gram every sequence test below mines: a failed Edit followed by a Read.
RECOVERY_SEQ = "edit:ERR → read"
RECOVERY_KEY = "seq-2-edit-err-read"


# ── fixtures / helpers ───────────────────────────────────────────────────────

class _FixedNow(datetime):
    """`datetime` whose `now()` is pinned, so two mining runs can be given two
    different emission dates inside one test. `fromisoformat` is inherited and
    still returns a plain datetime, which is all the miners ask of it."""
    NOW = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.NOW if tz is None else cls.NOW.astimezone(tz)


@pytest.fixture
def fixed_now(monkeypatch):
    monkeypatch.setattr(mt, "datetime", _FixedNow)
    return _FixedNow


@pytest.fixture
def no_verdicts(tmp_path):
    """An empty verdict store, so `status_block` cannot read the live
    `skill-verdicts.jsonl` and turn a candidate into `superseded_by_verdict`
    depending on what a human decided about the real key last week."""
    store = tmp_path / "verdicts.jsonl"
    store.write_text("", encoding="utf-8")
    return store


def seq_session(key: str, date: str) -> dict:
    """One session carrying the `edit:ERR → read` bigram. `result_summary`
    carries the session key and the date, so every instance of the n-gram prints
    visibly different evidence — which is what makes a frozen sample detectable
    by a hash."""
    return {
        "session_key": key,
        "timestamp": f"{date}T12:00:00Z",
        "tools": [
            {"name": "Edit", "sequence": 1, "is_error": True,
             "error_type": "not_found",
             "params_summary": {"path": f"/tmp/{key}.py"},
             "result_summary": f"File does not exist: /tmp/{key}.py ({date})"},
            {"name": "Read", "sequence": 2, "is_error": False,
             "params_summary": {"file_path": f"/tmp/{key}.py"},
             "result_summary": f"read /tmp/{key}.py in {key} on {date}"},
        ],
    }


def error_session(key: str, date: str) -> dict:
    """One corroborated `Bash`/`network` failure, keyed the same way every other
    session keys it. `failure_class` is set so `is_corroborated_error` decides
    on the field #500 introduced rather than on the legacy fallback."""
    return {
        "session_key": key,
        "timestamp": f"{date}T12:00:00Z",
        "error_tools": [
            {"name": "Bash", "sequence": 1, "error_type": "network",
             "failure_class": "network",
             "params_summary": {"command": f"curl https://example.test/{key}"},
             "result_summary": f"connection refused for {key} on {date}"},
        ],
    }


def success_session(key: str, date: str) -> dict:
    """One keyable non-error call. The signature is a value-shape key, not a
    `*_signature` bucket, because #391's `is_emittable` suppresses the latter
    and this pattern has to survive far enough to be inspected."""
    return {
        "session_key": key,
        "timestamp": f"{date}T12:00:00Z",
        "tools": [
            {"name": "http_fetch", "sequence": 1, "is_error": False,
             "params_summary": {"url": f"https://example.test/{key}"},
             "result_summary": f"fetched {key} on {date}"},
        ],
    }


def ascending_sessions(n: int, start_day: int = 1):
    """`n` sessions in the scan order the live miner uses: `load_trajectories`
    iterates `sorted(glob)`, so file order is chronological ascending and the
    first ones scanned are the oldest."""
    return [seq_session(f"s{i}", f"2026-09-{start_day + i - 1:02d}")
            for i in range(1, n + 1)]


def mine_sequences(rows, threshold=2):
    return mt.mine_sequence_patterns(rows, threshold=threshold)


def pattern_for(patterns, sequence_str=RECOVERY_SEQ):
    hits = [p for p in patterns if p["sequence_str"] == sequence_str]
    assert len(hits) == 1, [p["sequence_str"] for p in patterns]
    return hits[0]


def frontmatter(text: str) -> str:
    return text.split("---")[1]


def field(text: str, name: str) -> str:
    m = re.search(rf"^{name}:\s*(\d+)$", frontmatter(text), re.MULTILINE)
    assert m, f"{name}: absent from frontmatter:\n{frontmatter(text)}"
    return m.group(1)


def evidence_section(text: str) -> str:
    """The adjudicable payload: the headed example section, from its heading to
    the next `## `. This is what the triage hashed, and what consolidation's
    persistence test compares across dated snapshots."""
    for heading in ("## Concrete Examples", "## Error Examples",
                    "## Usage Examples"):
        start = text.find(heading)
        if start != -1:
            rest = text[start + len(heading):]
            end = rest.find("\n## ")
            return heading + (rest if end == -1 else rest[:end])
    raise AssertionError(f"no example section in:\n{text}")


def section_sha(text: str) -> str:
    return hashlib.sha256(evidence_section(text).encode()).hexdigest()


def printed_examples(text: str) -> int:
    return len(re.findall(r"^### Example \d+ ", evidence_section(text),
                          re.MULTILINE))


# ── clause 1: the sequence path samples the newest instances ─────────────────

def test_a_sequence_key_over_the_cap_shows_its_newest_sessions_not_its_first_three():
    """6 sessions carry the `edit:ERR → read` bigram. Pre-fix the emitted
    examples were s1/s2/s3 — the three oldest scanned — and stayed those three
    no matter how many sessions arrived; now the pool evicts the oldest held
    instance, so the sample is s4/s5/s6 and every one of them is outside the
    first three scanned."""
    pat = pattern_for(mine_sequences(ascending_sessions(6)))

    assert len(pat["examples"]) == 3, "the cap must still bind the sample size"
    shown = [ex["session_key"] for ex in pat["examples"]]
    assert shown == ["s4", "s5", "s6"], shown
    first_three_scanned = {"s1", "s2", "s3"}
    assert set(shown) - first_three_scanned, (
        "every printed example is one of the first three scanned for the key — "
        "the sample is still first-N-wins")


def test_a_sequence_key_at_or_under_the_cap_still_shows_every_session_it_has():
    """A rotation must not silently shrink a small key: 2 sessions print 2
    examples, so `examples_shown` can equal the denominator instead of lying."""
    pat = pattern_for(mine_sequences(ascending_sessions(2)))
    assert [ex["session_key"] for ex in pat["examples"]] == ["s1", "s2"]
    assert len(pat["examples"]) == len(pat["sessions"])


def test_the_sequence_offers_are_deduplicated_within_the_sessions_they_count():
    """A sequence key has no instance counter distinct from `sessions`, so the
    denominator clause means printing `examples_shown` beside `sessions` — which
    only means something if offers never exceed sessions. The same session_key
    arriving in two rows must contribute one offer, not two."""
    rows = ascending_sessions(6)
    rows.append(seq_session("s3", "2026-09-03"))  # duplicate row for s3
    pat = pattern_for(mine_sequences(rows))
    assert len(pat["examples"]) <= len(pat["sessions"]) == 6


# ── clause 2: the error and success paths rotate too ─────────────────────────

def test_an_error_key_over_its_cap_of_five_shows_an_instance_outside_the_first_five():
    sessions = [error_session(f"s{i}", f"2026-09-{i:02d}") for i in range(1, 9)]
    pat = [p for p in mt.mine_error_patterns(sessions, threshold=2)
           if p["tool_name"] == "Bash"][0]

    assert pat["total_calls"] == 8
    assert len(pat["examples"]) == 5, "the error cap is 5 and must still bind"
    shown = [ex["session_key"] for ex in pat["examples"]]
    assert shown == ["s4", "s5", "s6", "s7", "s8"], shown
    assert set(shown) - {f"s{i}" for i in range(1, 6)}


def test_a_success_key_over_its_cap_of_three_shows_an_instance_outside_the_first_three():
    sessions = [success_session(f"s{i}", f"2026-09-{i:02d}") for i in range(1, 7)]
    pat = [p for p in mt.mine_success_patterns(sessions, threshold=2)
           if p["tool_name"] == "http_fetch"][0]

    assert pat["total_calls"] == 6
    assert len(pat["examples"]) == 3, "the success cap is 3 and must still bind"
    shown = [ex["session_key"] for ex in pat["examples"]]
    assert shown == ["s4", "s5", "s6"], shown
    assert set(shown) - {"s1", "s2", "s3"}


def test_keys_under_their_cap_keep_every_example_on_the_error_and_success_paths():
    err = [error_session(f"s{i}", f"2026-09-0{i}") for i in range(1, 4)]
    ok = [success_session(f"s{i}", f"2026-09-0{i}") for i in range(1, 4)]
    err_pat = mt.mine_error_patterns(err, threshold=2)[0]
    ok_pat = mt.mine_success_patterns(ok, threshold=2)[0]

    assert len(err_pat["examples"]) == err_pat["total_calls"] == 3
    assert len(ok_pat["examples"]) == ok_pat["total_calls"] == 3


# ── clause 3: the emitted numerator sits beside the denominator ──────────────

def test_a_sequence_candidate_prints_examples_shown_against_its_session_count(
        tmp_path, no_verdicts):
    """The frozen-payload corpus is exactly this shape: 3 printed examples under
    a `sessions:` count of 111, with nothing on the page saying which."""
    pat = pattern_for(mine_sequences(ascending_sessions(6)))
    path = mt.write_candidate_file(pat, tmp_path, verdict_store=no_verdicts)
    text = Path(path).read_text(encoding="utf-8")

    assert field(text, "examples_shown") == "3"
    assert field(text, "sessions") == "6"
    assert printed_examples(text) == int(field(text, "examples_shown")), (
        "examples_shown must describe the printed sections, not the key")


def test_a_small_sequence_candidate_prints_examples_shown_equal_to_its_sessions(
        tmp_path, no_verdicts):
    """"3 of 111" and "3 of 3" have to be distinguishable. For a 2-session key
    they are the same number, which is the claim a reader needs."""
    pat = pattern_for(mine_sequences(ascending_sessions(2)))
    path = mt.write_candidate_file(pat, tmp_path, verdict_store=no_verdicts)
    text = Path(path).read_text(encoding="utf-8")

    assert field(text, "examples_shown") == field(text, "sessions") == "2"
    assert printed_examples(text) == 2


def test_an_error_candidate_prints_examples_shown_against_its_occurrences(
        tmp_path, no_verdicts):
    sessions = [error_session(f"s{i}", f"2026-09-{i:02d}") for i in range(1, 9)]
    pat = mt.mine_error_patterns(sessions, threshold=2)[0]
    path = mt.write_candidate_file(pat, tmp_path, verdict_store=no_verdicts)
    text = Path(path).read_text(encoding="utf-8")

    assert field(text, "examples_shown") == "5"
    assert field(text, "occurrences") == "8"
    assert printed_examples(text) == 5


def test_a_small_error_candidate_prints_examples_shown_equal_to_its_occurrences(
        tmp_path, no_verdicts):
    sessions = [error_session(f"s{i}", f"2026-09-0{i}") for i in range(1, 4)]
    pat = mt.mine_error_patterns(sessions, threshold=2)[0]
    path = mt.write_candidate_file(pat, tmp_path, verdict_store=no_verdicts)
    text = Path(path).read_text(encoding="utf-8")

    assert field(text, "examples_shown") == field(text, "occurrences") == "3"


def test_a_success_candidate_prints_examples_shown_against_its_occurrences(
        tmp_path, no_verdicts):
    """Hand-built because #391's `is_emittable` suppresses any mined success key
    ending in `_signature`, and a suppressed pattern never reaches the writer.
    `total_calls` is 9 against 3 printed examples: the "3 of 9" case."""
    pat = {"type": "success", "tool_name": "http_fetch",
           "params_signature": "url_host_example_test",
           "sessions": {"s1", "s2", "s3", "s4"},
           "examples": [{"session_key": f"s{i}", "date": f"2026-09-0{i}",
                         "tool": "http_fetch", "params_summary": {},
                         "result_summary": "ok", "is_error": False,
                         "sequence": i} for i in (2, 3, 4)],
           "dates": {"2026-09-01", "2026-09-04"}, "total_calls": 9,
           "error_count": 0, "error_rate": 0.0,
           "first_seen": "2026-09-01", "last_seen": "2026-09-04"}
    path = mt.write_candidate_file(pat, tmp_path, verdict_store=no_verdicts)
    text = Path(path).read_text(encoding="utf-8")

    assert field(text, "examples_shown") == "3"
    assert field(text, "occurrences") == "9"
    assert printed_examples(text) == 3


# ── clause 4: re-emission over a grown instance set changes the evidence ─────

def test_re_emitting_the_recovery_key_over_two_more_sessions_changes_its_evidence(
        tmp_path, fixed_now, no_verdicts):
    """The item's acceptance check, across the real miner and the real writer.

    Key `seq-2-edit-err-read`. Run 1, 2026-09-19, seen in 4 sessions →
    `candidate-seq-2-edit-err-read-20260919.md`. Run 2, 2026-09-20, the same
    corpus plus two newer sessions → `candidate-seq-2-edit-err-read-20260920.md`.
    The sha256 of the two `## Concrete Examples` sections must differ.

    Pre-fix both files carried s1/s2/s3 and the hashes were equal, which is the
    measurement that made 625 of 901 keys byte-identical first-to-last."""
    older = ascending_sessions(4)
    newer = older + [seq_session("s5", "2026-09-05"), seq_session("s6", "2026-09-06")]

    _FixedNow.NOW = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)
    first = pattern_for(mine_sequences(older))
    path_a = mt.write_candidate_file(first, tmp_path, verdict_store=no_verdicts)

    _FixedNow.NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
    second = pattern_for(mine_sequences(newer))
    path_b = mt.write_candidate_file(second, tmp_path, verdict_store=no_verdicts)

    assert Path(path_a).name == f"candidate-{RECOVERY_KEY}-20260919.md", Path(path_a).name
    assert Path(path_b).name == f"candidate-{RECOVERY_KEY}-20260920.md", Path(path_b).name

    text_a = Path(path_a).read_text(encoding="utf-8")
    text_b = Path(path_b).read_text(encoding="utf-8")
    assert field(text_a, "sessions") == "4"
    assert field(text_b, "sessions") == "6"
    assert section_sha(text_a) != section_sha(text_b), (
        f"{Path(path_a).name} and {Path(path_b).name} carry identical "
        f"{RECOVERY_KEY} evidence — the sample is frozen")


def test_re_emission_still_changes_the_evidence_when_the_new_sessions_cannot_all_fit(
        tmp_path, fixed_now, no_verdicts):
    """The same key, one day apart, 6 sessions then 8: the pool slides by two
    and only one example survives the re-emission. A sample that rotated on the
    first growth but froze afterwards would pass the two-run test above and fail
    here."""
    _FixedNow.NOW = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)
    six = pattern_for(mine_sequences(ascending_sessions(6)))
    path_a = mt.write_candidate_file(six, tmp_path, verdict_store=no_verdicts)

    _FixedNow.NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
    eight = pattern_for(mine_sequences(ascending_sessions(8)))
    path_b = mt.write_candidate_file(eight, tmp_path, verdict_store=no_verdicts)

    text_a = Path(path_a).read_text(encoding="utf-8")
    text_b = Path(path_b).read_text(encoding="utf-8")
    assert section_sha(text_a) != section_sha(text_b)
    assert "s8" in evidence_section(text_b)
    assert "s6" in evidence_section(text_a)


def test_the_sample_does_not_move_with_the_caller_s_list_order(tmp_path):
    """#1131 made candidate emission order-independent; the rotating sample must
    not reintroduce that. Every instance's order key is its own date plus its
    position in the key's scan, so feeding the same 5 sessions back-to-front
    selects the same three instances — the date dominates, and only a same-date
    tie is broken by scan position."""
    forward = ascending_sessions(5)

    a = pattern_for(mine_sequences(forward))
    b = pattern_for(mine_sequences(list(reversed(forward))))
    assert [e["session_key"] for e in a["examples"]] == \
           [e["session_key"] for e in b["examples"]], (
        "the printed sample moved with the caller's list order")


def test_an_unchanged_instance_set_re_emits_the_same_evidence_on_a_later_date(
        tmp_path, fixed_now, no_verdicts):
    """Rotation is driven by new instances, not by the calendar. The same 6
    sessions mined on 09-19 and again on 09-20 must print byte-identical
    `## Concrete Examples` sections — otherwise the clause-4 hash test above
    could be satisfied by a sampler that churns its sample every run, and a
    consolidation pass re-adjudicating an old key would see "new evidence" that
    is the same six legs re-shuffled. `sessions:` still differs between the two
    files, so this is not the frozen-payload case returning."""
    rows = ascending_sessions(6)

    _FixedNow.NOW = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)
    first = pattern_for(mine_sequences(rows))
    path_a = mt.write_candidate_file(first, tmp_path, verdict_store=no_verdicts)

    _FixedNow.NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
    second = pattern_for(mine_sequences(rows))
    path_b = mt.write_candidate_file(second, tmp_path, verdict_store=no_verdicts)

    assert Path(path_a).name == f"candidate-{RECOVERY_KEY}-20260919.md", Path(path_a).name
    assert Path(path_b).name == f"candidate-{RECOVERY_KEY}-20260920.md", Path(path_b).name

    text_a = Path(path_a).read_text(encoding="utf-8")
    text_b = Path(path_b).read_text(encoding="utf-8")
    assert section_sha(text_a) == section_sha(text_b), (
        f"{Path(path_a).name} and {Path(path_b).name} differ with no new "
        f"instance for {RECOVERY_KEY} — the sample churns instead of rotating")
    assert field(text_a, "examples_shown") == field(text_b, "examples_shown") == "3"
