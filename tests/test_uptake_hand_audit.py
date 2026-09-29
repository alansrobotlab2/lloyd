"""Tests for `scripts/uptake_hand_audit.py` — the #1849 hand-audit packet builder.

Each test names the clause of item #1849 it pins. The builder exists because the
uptake gate (`app/uptake.py`, `PRECISION_FLOOR` 0.70 / `RECALL_FLOOR` 0.50) has
no gradable label corpus: `hand-2026-09-11.json` is `spent` (0 of 46 turns
resolve), and rebuilding the set is a human judgement. So every test here is
about the two things the machine is allowed to do — make the reading cheap, and
never pretend to have done the judgement — plus the floors that stop it
publishing a set with no positive class.

Fixture note: a candidate turn is one the CUE screen (`uptake.candidate_disputes`)
screens in — text matching `_DISPUTE_CUES` longer than 12 characters, or a bare
re-prompt. Non-candidates are built from plain requests with no cue word in them.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app import uptake  # noqa: E402

SCRIPT = REPO / "scripts" / "uptake_hand_audit.py"

_spec = importlib.util.spec_from_file_location("uptake_hand_audit_for_tests", SCRIPT)
assert _spec and _spec.loader
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

#: The six keys a packet item carries and nothing else. Spelled out HERE rather
#: than read from the module, so a change to `audit.PACKET_KEYS` goes red instead
#: of being definitionally true.
ITEM_KEYS = {"turn_id", "ts", "prev_assistant", "user_text", "reason", "label"}

#: Cue-screened text (matches `_DISPUTE_CUES` on `no,` and `wrong`) and text that
#: matches nothing in it. Both are longer than the 12-character screen floor.
CUE_TEXT = "no, that is wrong, run the migration for real this time {i}"
PLAIN_TEXT = "please summarize the deploy checklist for release {i}"


#: The last turn of every fixture session, so `measured_growth` reads a span that
#: is entirely in the past — a corpus whose newest turn is next week would make
#: every projected date in these tests meaningless.
LATEST = datetime(2026, 9, 28, 10, 0, tzinfo=timezone.utc)


def _msgs(n_cand: int, n_plain: int) -> list[dict]:
    """`n_cand` screened-in turns and `n_plain` screened-out ones, one per day.

    Timestamps run backwards from `LATEST` and are one day apart, so
    `measured_growth` has a real span to read: `n` turns over `n-1` days is a
    rate the projection tests can compute by hand.
    """
    texts = ([PLAIN_TEXT.format(i=i) for i in range(n_plain)]
             + [CUE_TEXT.format(i=i) for i in range(n_cand)])
    out: list[dict] = []
    for i, text in enumerate(texts):
        ts = (LATEST - timedelta(days=len(texts) - 1 - i)).isoformat()
        out.append({"role": "assistant",
                    "content": f"Here is result {i}. " + "x" * 40,
                    "timestamp": ts})
        out.append({"role": "user", "content": text, "timestamp": ts})
    return out


def _write_store(root: Path, name: str, messages: list[dict]) -> Path:
    """A live conversation transcript: `source` unset, no synthetic prefix."""
    doc = {"session_id": name, "id": name, "title": "", "source": None,
           "created_at": messages[0]["timestamp"],
           "last_active": messages[-1]["timestamp"], "messages": messages}
    p = root / "sessions" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))
    return p


def _corpus(root: Path) -> list[uptake.Turn]:
    return uptake.human_turns(root=root, days=audit.CORPUS_DAYS)


def _run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    rc = audit.main(list(argv), out=out, err=err)
    return rc, out.getvalue(), err.getvalue()


@pytest.fixture()
def store(tmp_path: Path) -> Path:
    """A store clearing every floor: 20 candidates among 70 human turns.

    Split across three sessions (`s_a` long, `s_b` and `s_c` short) so a sample
    that only ever read the head of the corpus is visible, not the default. The
    counts are asserted HERE because every test downstream reads the packet's
    numbers as a measurement of this fixture.
    """
    _write_store(tmp_path, "s_a", _msgs(14, 40))
    _write_store(tmp_path, "s_b", _msgs(3, 5))
    _write_store(tmp_path, "s_c", _msgs(3, 5))
    turns = _corpus(tmp_path)
    cands = uptake.candidate_disputes(turns)
    assert len(turns) == 70, len(turns)
    assert len(cands) == 20, len(cands)
    return tmp_path


@pytest.fixture()
def small_store(tmp_path: Path) -> Path:
    """40 human turns but only 3 candidates: over the item floor, under it."""
    _write_store(tmp_path, "s_a", _msgs(3, 37))
    assert len(_corpus(tmp_path)) == 40
    assert len(uptake.candidate_disputes(_corpus(tmp_path))) == 3
    return tmp_path


def _labels(root: Path) -> Path:
    return root / "labels"


# ---------------------------------------------------------------- clause 1 --

def test_emit_writes_the_packet_with_the_documented_item_shape(store, tmp_path):
    """Clause 1: >= 15 candidates among >= 40 human turns -> a packet whose items
    carry EXACTLY the six documented keys, and EVERY candidate turn appears."""
    rc, out, err = _run("--root", str(store), "--labels-dir", str(_labels(store)),
                        "--date", "2026-11-19")
    assert rc == 0, (out, err)
    path = _labels(store) / "packet-2026-11-19.json"
    doc = json.loads(path.read_text())
    items = doc["items"]

    for it in items:
        assert set(it) == ITEM_KEYS, sorted(it)
    assert not any("engine" in k for it in items for k in it), \
        "a packet must not carry engine_raw / engine_predicted"

    cand_ids = {t.turn_id for t in uptake.candidate_disputes(_corpus(store))}
    assert cand_ids, "the fixture has no candidates to find"
    packet_ids = {it["turn_id"] for it in items}
    assert cand_ids <= packet_ids, sorted(cand_ids - packet_ids)
    assert doc["n_candidates"] == len(cand_ids) == 20, doc["n_candidates"]
    assert len(items) == doc["n_items"] >= audit.MIN_ITEMS, len(items)


def test_the_default_labels_dir_is_eval_uptake_labels_under_the_repo(store,
                                                                    monkeypatch):
    """Clause 1's PATH half: with no `--labels-dir` the packet goes to
    `eval/uptake/labels/packet-<date>.json`, which is the directory
    `app.uptake.LABEL_GLOB` reads. REPO is redirected to a scratch tree so the
    real `eval/uptake/labels/` is never touched."""
    scratch = store / "repo"
    monkeypatch.setattr(audit, "REPO", scratch)
    rc, out, err = _run("--root", str(store), "--date", "2026-11-19")
    assert rc == 0, (out, err)
    assert (scratch / "eval" / "uptake" / "labels"
            / "packet-2026-11-19.json").is_file(), out


def test_the_sample_is_spread_across_the_corpus_not_a_head_slice(store):
    """Clause 1's "stratified" half. Sessions sort by name, so `s_a`'s 54 turns
    come first and a head-slice would never reach the last turn of `s_c`. The
    systematic sample takes endpoints, so the store's LAST turn must be in the
    packet — a `pool[:want]` reading of "sample" fails here."""
    rc, out, err = _run("--root", str(store), "--labels-dir", str(_labels(store)),
                        "--date", "2026-11-19")
    assert rc == 0, (out, err)
    turns = _corpus(store)
    non_ids = [t.turn_id for t in turns
               if t.turn_id not in {c.turn_id for c in
                                    uptake.candidate_disputes(turns)}]
    packet_ids = {it["turn_id"] for it in
                  json.loads((_labels(store) / "packet-2026-11-19.json")
                             .read_text())["items"]}
    assert non_ids[-1] in packet_ids, non_ids[-1]
    sampled_sessions = {tid.split("#")[0] for tid in packet_ids
                        if tid in set(non_ids)}
    assert len(sampled_sessions) >= 2, sorted(sampled_sessions)


def test_rebuilding_a_packet_from_an_unchanged_store_is_byte_identical(store):
    """The docstring promises a rebuilt packet is byte-identical so either copy
    can be handed to a labeler. A seeded shuffle would also satisfy "deterministic"
    while breaking this, and a resample would silently change which turns were
    read — which is the whole reason the label set is auditable."""
    a = _labels(store) / "a"
    b = _labels(store) / "b"
    assert _run("--root", str(store), "--labels-dir", str(a),
                "--date", "2026-11-19")[0] == 0
    assert _run("--root", str(store), "--labels-dir", str(b),
                "--date", "2026-11-19")[0] == 0
    assert (a / "packet-2026-11-19.json").read_text() \
        == (b / "packet-2026-11-19.json").read_text()


# ---------------------------------------------------------------- clause 2 --

def test_emit_writes_no_label_even_with_a_verdict_available(store, monkeypatch,
                                                            tmp_path):
    """Clause 2: with the cue screen marking candidates and a classifier verdict
    AVAILABLE, every emitted item still has `label: null` and `reason: null`.

    "Available" is made real by patching the engine to a loud wrong answer: the
    builder must not merely decline to write it, it must not call it at all, so
    the patch raises. An engine call would also be a per-candidate request
    charged against a run whose whole product is a blank form.
    """
    def boom(*a, **k):
        raise AssertionError("the packet builder must not call the classifier")

    monkeypatch.setattr(uptake, "classify_dispute", boom)
    monkeypatch.setattr(uptake, "classify_dispute_raw", boom)
    rc, out, err = _run("--root", str(store), "--labels-dir", str(_labels(store)),
                        "--date", "2026-11-19")
    assert rc == 0, (out, err)
    items = json.loads((_labels(store) / "packet-2026-11-19.json")
                       .read_text())["items"]
    assert items, "no items to check means the null-label claim is vacuous"
    assert all(it["label"] is None for it in items)
    assert all(it["reason"] is None for it in items)
    # The cue screen is what marks a turn for reading, so the packet has to be
    # full of the very turns a classifier would have opinions about.
    cand_ids = {t.turn_id for t in uptake.candidate_disputes(_corpus(store))}
    assert cand_ids <= {it["turn_id"] for it in items}


def test_a_packet_written_by_a_previous_run_is_never_rewritten(store, tmp_path):
    """The refusal to auto-label is worthless if a second run can clobber a packet
    a human is half-way through filling in."""
    labels = _labels(store)
    assert _run("--root", str(store), "--labels-dir", str(labels),
                "--date", "2026-11-19")[0] == 0
    path = labels / "packet-2026-11-19.json"
    doc = json.loads(path.read_text())
    doc["items"][0]["label"] = 1
    doc["items"][0]["reason"] = "contradicted a delivered result"
    path.write_text(json.dumps(doc))

    rc, out, err = _run("--root", str(store), "--labels-dir", str(labels),
                        "--date", "2026-11-19")
    assert rc == audit.EXIT_BELOW_FLOOR
    assert "already exists" in err, err
    assert json.loads(path.read_text())["items"][0]["label"] == 1, \
        "a human's filled-in label was overwritten"


# ---------------------------------------------------------------- clause 3 --

def test_emit_refuses_below_the_candidate_floor_and_writes_nothing(small_store):
    """Clause 3: 40 human turns but 3 candidates -> non-zero, no packet, both
    measured counts printed, and a projected date."""
    labels = _labels(small_store)
    rc, out, err = _run("--root", str(small_store), "--labels-dir", str(labels),
                        "--date", "2026-11-19")
    assert rc != 0, err
    assert rc == audit.EXIT_BELOW_FLOOR
    assert not labels.exists() and not list(small_store.rglob("packet-*.json")), \
        "a refused run must write nothing"
    both = out + err
    assert "human_turns(days=900) = 40" in both, both
    assert "candidates = 3" in both, both
    assert "wrote nothing" in both, both
    assert "projected ready date" in both, both


def test_emit_refuses_below_the_human_turn_floor_too(tmp_path):
    """Clause 3's other half: enough candidates, too few human turns. A packet of
    only disputes has no negative class, so precision would be graded against
    nothing."""
    root = tmp_path / "store"
    _write_store(root, "s_a", _msgs(20, 5))
    labels = tmp_path / "labels"
    rc, out, err = _run("--root", str(root), "--labels-dir", str(labels))
    assert rc == audit.EXIT_BELOW_FLOOR, (out, err)
    assert "packet items 25 < 40" in err, err
    assert not labels.exists()


@pytest.mark.parametrize("n_turns,n_cand,span_days,want_days", [
    # yield 1/100 = 1% => the 15-candidate floor wants 1500 turns, past the
    # 500-turn trigger; rate 100/10 = 10/day => ceil(1400/10) = 140 days.
    (100, 1, 10, 140),
    # yield 20/100 = 20% => 75 turns would clear the candidate floor, so the
    # 500-turn trigger binds: ceil(400/10) = 40 days.
    (100, 20, 10, 40),
])
def test_the_projection_is_derived_from_the_measured_counts(n_turns, n_cand,
                                                           span_days, want_days):
    """Clause 3's "a projected date derived from them": the printed date is the
    counts divided by the measured rate, not a date anyone wrote down."""
    day0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
    step = span_days / (n_turns - 1)
    turns = [uptake.Turn(session="s", ts=(day0 + timedelta(days=step * i)).isoformat(),
                         user_text=f"turn {i}", ordinal=i + 1)
             for i in range(n_turns)]
    p = audit.projection(turns, n_cand, today=datetime(2026, 10, 1,
                                                       tzinfo=timezone.utc))
    assert p["days"] == want_days, p
    assert p["ready_date"] == (date(2026, 10, 1) + timedelta(days=want_days)
                               ).isoformat(), p


def test_a_fallback_growth_rate_is_named_as_one_and_an_empty_store_gets_no_date():
    """Two shapes of the same honesty rule. One turn with no timestamp has no span
    to measure, so the ~8/day constant from triage is used — and the printed basis
    must SAY `fallback`, because a date resting on a carried constant is not a
    measurement. An EMPTY store gets no date at all: applying that rate to zero
    turns would publish "ready in 63 days" over a tree with no transcripts, which
    is #1679's shape (a probe measuring an unmounted root and reading as health)."""
    lonely = [uptake.Turn(session="s", ts=None, user_text="only turn", ordinal=1)]
    p = audit.projection(lonely, 0, today=datetime(2026, 10, 1, tzinfo=timezone.utc))
    assert p["ready_date"] is not None, p
    assert "fallback" in p["rate_basis"], p["rate_basis"]

    empty = audit.projection([], 0, today=datetime(2026, 10, 1, tzinfo=timezone.utc))
    assert empty["ready_date"] is None, empty
    assert empty["human_turns"] == 0, empty


def test_an_empty_store_refuses_without_publishing_a_projection_date(tmp_path):
    """The empty-corpus refusal is what a round worktree or an unmounted data root
    looks like from here, and it must not read as "ready in 63 days"."""
    (tmp_path / "sessions").mkdir()
    labels = tmp_path / "labels"
    rc, out, err = _run("--root", str(tmp_path), "--labels-dir", str(labels))
    assert rc == audit.EXIT_BELOW_FLOOR, (out, err)
    assert "none — no growth rate measurable" in err, err
    assert not labels.exists()


def test_help_names_the_turn_trigger_and_carries_no_calendar_date(capsys):
    """Clause 3's last half: `--help` states the trigger as
    `human_turns(days=900)` >= about 500 and never as a date. The corpus grows, so
    a date in the help is false the week usage changes — and a date is exactly the
    form a future reader would schedule on, which is how #1624's remedy ended up
    quoted against a calendar for three weeks."""
    with pytest.raises(SystemExit) as exc:
        audit.main(["--help"])
    assert exc.value.code == 0, exc.value.code
    text = capsys.readouterr().out
    assert "human_turns(days=900)" in text, text
    assert "500" in text, text
    assert not re.search(r"\b\d{4}-\d{2}-\d{2}\b", text), \
        f"--help advertises a calendar date, not the volume trigger:\n{text}"


def test_the_cli_exit_status_and_refusal_survive_a_real_subprocess(store, tmp_path):
    """The refusal is a CLI contract (`automod`/an operator reads the exit status,
    not a return value), so it is checked across the process boundary: real argv,
    real stdout, real exit code. `small_store` is rebuilt inline to keep this
    subprocess call honest about what it refuses."""
    root = tmp_path / "store"
    _write_store(root, "s_a", _msgs(2, 38))
    labels = tmp_path / "labels"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root),
         "--labels-dir", str(labels)],
        capture_output=True, text=True, timeout=180, cwd=str(REPO))
    assert proc.returncode == audit.EXIT_BELOW_FLOOR, proc.stderr
    assert "human_turns(days=900) = 40" in proc.stdout + proc.stderr
    assert not labels.exists()


# ---------------------------------------------------------------- clause 4 --

def _fill(path: Path, *, positives: int, reason: str = "contradicts the "
                                                      "delivered result") -> Path:
    """Label a packet the way the human pass does: 1s and 0s with reasons."""
    doc = json.loads(path.read_text())
    for i, it in enumerate(doc["items"]):
        it["label"] = 1 if i < positives else 0
        it["reason"] = reason if it["label"] == 1 else "new request, no reply yet"
    path.write_text(json.dumps(doc))
    return path


def test_merge_writes_the_live_set_with_the_supplied_labeled_by(store):
    """Clause 4: a completed packet plus an explicit `--labeled-by` writes
    `hand-<date>.json` whose `labeled_by` is EXACTLY the supplied value."""
    labels = _labels(store)
    assert _run("--root", str(store), "--labels-dir", str(labels),
                "--date", "2026-11-19")[0] == 0
    packet = _fill(labels / "packet-2026-11-19.json", positives=23)

    rc, out, err = _run("--root", str(store), "--labels-dir", str(labels),
                        "--merge", str(packet), "--date", "2026-11-20",
                        "--labeled-by", "hand:alan-turns-2026-11-20")
    assert rc == 0, (out, err)
    doc = json.loads((labels / "hand-2026-11-20.json").read_text())
    assert doc["labeled_by"] == "hand:alan-turns-2026-11-20", doc["labeled_by"]
    assert doc["n_positives"] == 23 == sum(1 for i in doc["items"]
                                          if i["label"] == 1), doc
    assert all(i["labeled_by"] == "hand:alan-turns-2026-11-20" for i in doc["items"])
    # The merged set must be the thing the gate can grade: same item keys plus
    # provenance, `status: live`, and every item re-anchoring to a real turn.
    assert doc["status"] == "live", doc["status"]
    for it in doc["items"]:
        assert set(it) == ITEM_KEYS | {"labeled_by"}, sorted(it)
    assert doc["n_resolved"] == doc["n_items"], doc["n_resolved"]
    assert "re-resolved" in out, out


def test_the_merged_set_is_what_load_labels_resolves(store):
    """The seam that matters: the builder's output is consumed by
    `app.uptake.load_labels` in a different process, through the
    `hand-*.json` glob and the `status` front matter — not by any function this
    diff calls. So the file is read back through the real loader."""
    labels = _labels(store)
    assert _run("--root", str(store), "--labels-dir", str(labels),
                "--date", "2026-11-19")[0] == 0
    packet = _fill(labels / "packet-2026-11-19.json", positives=21)
    assert _run("--root", str(store), "--labels-dir", str(labels),
                "--merge", str(packet), "--date", "2026-11-20",
                "--labeled-by", "hand:alan-turns-2026-11-20")[0] == 0

    root = store / "repo"
    (root / "eval" / "uptake" / "labels").mkdir(parents=True, exist_ok=True)
    for name in ("hand-2026-11-20.json", "packet-2026-11-19.json"):
        (root / "eval" / "uptake" / "labels" / name).write_text(
            (labels / name).read_text())

    assert uptake.labels_path(root=root).name == "hand-2026-11-20.json"
    got = uptake.load_labels(root=root)
    assert len(got) == 45 == len(json.loads(
        (root / "eval" / "uptake" / "labels" / "hand-2026-11-20.json")
        .read_text())["items"]), len(got)
    assert sum(1 for i in got if i["label"] == 1) == 21
    st = uptake.labels_status(root=root)
    assert st["status"] == uptake.LABEL_STATUS_LIVE and st["spent"] is False, st


def test_merge_refuses_an_explicit_labeled_by_absence(store):
    """Clause 4: no `--labeled-by` -> non-zero, nothing written. `LABELER` in
    `app/uptake.py` is dead and no reader validates the string, so this flag is
    the only provenance control there is: a default would be a fabricated
    attestation, and the test below catches one being added."""
    labels = _labels(store)
    assert _run("--root", str(store), "--labels-dir", str(labels),
                "--date", "2026-11-19")[0] == 0
    packet = _fill(labels / "packet-2026-11-19.json", positives=25)
    rc, out, err = _run("--root", str(store), "--labels-dir", str(labels),
                        "--merge", str(packet), "--date", "2026-11-20")
    assert rc == audit.EXIT_MERGE_REFUSED, (out, err)
    assert "--labeled-by" in err, err
    assert "wrote nothing" in err, err
    assert not (labels / "hand-2026-11-20.json").exists()
    assert not (labels / "hand-2026-11-19.json").exists()
    assert sorted(p.name for p in labels.iterdir()) == ["packet-2026-11-19.json"]


def test_merge_refuses_a_blank_labeled_by(store):
    """The no-default rule has a second shape: a flag present but empty. Written
    as its own node because a `labeled_by or DEFAULT` implementation would pass
    the absence test and fail this one."""
    labels = _labels(store)
    assert _run("--root", str(store), "--labels-dir", str(labels),
                "--date", "2026-11-19")[0] == 0
    packet = _fill(labels / "packet-2026-11-19.json", positives=25)
    rc, out, err = _run("--root", str(store), "--labels-dir", str(labels),
                        "--merge", str(packet), "--labeled-by", "   ")
    assert rc == audit.EXIT_MERGE_REFUSED, (out, err)
    assert not list(labels.glob("hand-*.json")), list(labels.iterdir())


def test_merge_refuses_below_twenty_positives_and_names_the_shortfall(store):
    """Clause 4's floor: 19 positives is one short of the 20 that
    `tests/test_uptake.py::test_hand_labeled_corpus_covers_the_item_s_minimum`
    pins on the LIVE file, and `labels_path` returns only the newest `hand-*.json`
    — so a merged set that lands 19 positives does not supplement
    `hand-2026-09-11.json`, it shadows it with a red suite."""
    labels = _labels(store)
    assert _run("--root", str(store), "--labels-dir", str(labels),
                "--date", "2026-11-19")[0] == 0
    packet = _fill(labels / "packet-2026-11-19.json", positives=19)
    rc, out, err = _run("--root", str(store), "--labels-dir", str(labels),
                        "--merge", str(packet),
                        "--labeled-by", "hand:alan-turns-2026-11-20")
    assert rc == audit.EXIT_MERGE_REFUSED, (out, err)
    assert "19 of" in err, err
    assert "short by 1" in err, err
    assert "20" in err, err
    assert not list(labels.glob("hand-*.json")), list(labels.iterdir())


def test_merge_accepts_exactly_twenty_positives(store):
    """The boundary, from the other side: 19 refuses and 20 writes. Without this
    node a `> MIN_POSITIVES` off-by-one would satisfy the refusal test and the
    happy-path test (which uses 23) alike."""
    labels = _labels(store)
    assert _run("--root", str(store), "--labels-dir", str(labels),
                "--date", "2026-11-19")[0] == 0
    packet = _fill(labels / "packet-2026-11-19.json", positives=20)
    rc, out, err = _run("--root", str(store), "--labels-dir", str(labels),
                        "--date", "2026-11-20", "--merge", str(packet),
                        "--labeled-by", "hand:alan-turns-2026-11-20")
    assert rc == 0, (out, err)
    doc = json.loads((labels / "hand-2026-11-20.json").read_text())
    assert doc["n_positives"] == 20, doc


def test_merge_refuses_an_incomplete_packet(store):
    """A packet with a null label still in it is not a labelling pass; merging it
    would write blanks into the file the gate grades. The same refusal covers a
    label that is neither 0 nor 1, which must be reported and not coerced: a
    `"yes"` silently read as 1 puts a judgement the labeler did not make into
    the corpus the gate trusts."""
    labels = _labels(store)
    assert _run("--root", str(store), "--labels-dir", str(labels),
                "--date", "2026-11-19")[0] == 0
    path = labels / "packet-2026-11-19.json"
    doc = json.loads(path.read_text())
    for i, it in enumerate(doc["items"]):
        it["label"] = 1 if i < 25 else None
        it["reason"] = "contradicted the result" if it["label"] == 1 else None
    path.write_text(json.dumps(doc))
    rc, out, err = _run("--root", str(store), "--labels-dir", str(labels),
                        "--merge", str(path),
                        "--labeled-by", "hand:alan-turns-2026-11-20")
    assert rc == audit.EXIT_MERGE_REFUSED, (out, err)
    assert "unlabelled" in err, err
    assert not list(labels.glob("hand-*.json"))

    doc["items"][30]["label"] = "yes"
    doc["items"][30]["reason"] = "clearly a dispute"
    for i, it in enumerate(doc["items"]):
        if it["label"] is None:
            it["label"] = 0
            it["reason"] = "new request"
    path.write_text(json.dumps(doc))
    rc, out, err = _run("--root", str(store), "--labels-dir", str(labels),
                        "--merge", str(path),
                        "--labeled-by", "hand:alan-turns-2026-11-20")
    assert rc == audit.EXIT_MERGE_REFUSED, (out, err)
    assert "not 0 or 1" in err, err
    assert not list(labels.glob("hand-*.json")), list(labels.iterdir())


def test_merge_never_invents_a_label(store):
    """Clause 2's second mode: the labels in a merged set are the packet's,
    byte-for-byte, in the packet's order — the script contributes provenance and
    counts, never a verdict. Written as an identity check over every item so a
    single quietly-flipped label is caught, not just the totals."""
    labels = _labels(store)
    assert _run("--root", str(store), "--labels-dir", str(labels),
                "--date", "2026-11-19")[0] == 0
    packet = _fill(labels / "packet-2026-11-19.json", positives=22)
    before = [(i["turn_id"], i["label"], i["reason"])
              for i in json.loads(packet.read_text())["items"]]
    assert _run("--root", str(store), "--labels-dir", str(labels),
                "--merge", str(packet), "--date", "2026-11-20",
                "--labeled-by", "hand:alan-turns-2026-11-20")[0] == 0
    after = [(i["turn_id"], i["label"], i["reason"])
             for i in json.loads((labels / "hand-2026-11-20.json")
                                 .read_text())["items"]]
    assert after == before, [(a, b) for a, b in zip(after, before) if a != b]


def test_merge_refuses_to_overwrite_an_existing_label_set(store):
    """A label set is the historical record of one human pass. Overwriting it
    would be the same loss the packet-overwrite refusal prevents."""
    labels = _labels(store)
    assert _run("--root", str(store), "--labels-dir", str(labels),
                "--date", "2026-11-19")[0] == 0
    packet = _fill(labels / "packet-2026-11-19.json", positives=24)
    args = ["--root", str(store), "--labels-dir", str(labels), "--merge",
            str(packet), "--date", "2026-11-20",
            "--labeled-by", "hand:alan-turns-2026-11-20"]
    assert _run(*args)[0] == 0
    rc, out, err = _run(*args)
    assert rc == audit.EXIT_MERGE_REFUSED, (out, err)
    assert "already exists" in err, err


def test_merge_refuses_a_packet_that_is_not_a_packet(store):
    """`--merge /etc/passwd` and a JSON file that is not a packet must be a
    refusal with a reason, not a traceback on an operator's terminal."""
    labels = _labels(store)
    labels.mkdir(parents=True)
    bad = labels / "packet-notreal.json"
    bad.write_text(json.dumps({"items": [], "n_items": 0}))
    rc, out, err = _run("--root", str(store), "--labels-dir", str(labels),
                        "--merge", str(bad),
                        "--labeled-by", "hand:alan-turns-2026-11-20")
    assert rc == audit.EXIT_MERGE_REFUSED, (out, err)
    assert "no items" in err, err

    bad.write_text("this is not json")
    rc, out, err = _run("--root", str(store), "--labels-dir", str(labels),
                        "--merge", str(bad),
                        "--labeled-by", "hand:alan-turns-2026-11-20")
    assert rc == audit.EXIT_MERGE_REFUSED, (out, err)
    assert "cannot read packet" in err, err
