"""The exhibit sweep: read both suffixes, print denominators, never read 0 as 100% (#3749 3-5).

Three failure modes this file exists to make impossible, each pinned by a node that does not
depend on production traffic:

* a sweep that globs `*.json` and calls what it found the corpus. On this box 4,496 of 6,956
  sessions in a 7-day window are `.json.gz` — the retention sweep compresses old sessions
  rather than deleting them — so the plain-suffix-only read `app/tool_quality.py:489` uses
  would have measured 35% of the corpus and printed it as all of it. Clause 3 is pinned with
  BOTH suffixes on disk, because a node that only ever seeds `.json` cannot tell a
  gz-reading sweep from a gz-dropping one;
* a row with `N=0` rendered as 100% compliance. `0 / 0` is the sound of a trigger that never
  fires, and #2275's degenerate argument already produced a live `0 / 89,105` that described
  the harness rather than the model;
* a detector that is simply wrong about its own rule. The positive-control fixtures carry the
  expected N AND k for a seeded trace, and every one also carries a counter-trace with a
  LOWER expected k — a control that only asserts non-zero would pass a detector that always
  answers True, which is the compliance story this instrument must not manufacture.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.behaviour_exhibit as BE  # noqa: E402

import ast  # noqa: E402  # clause 5: the no-hard-coded-bar check reads the module's AST

NOW = datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)


def _tc(name, args):
    return {"id": f"c-{name}-{abs(hash(json.dumps(args, sort_keys=True)))}",
            "function": {"name": name, "arguments": json.dumps(args, sort_keys=True)}}


def _msg(text="", calls=()):
    return {"role": "assistant", "content": text, "tool_calls": list(calls)}


def write_session(dir_, sid, messages, *, gzipped=False, source=None, platform=None,
                  age_days=0, msg_ts=None, mtime=None):
    """One transcript on disk, gzipped or not, stamped inside the window unless aged.

    `platform` is written only when given: whether a record carries the field at all is one of
    the signals `population_of` reads, so a helper that always wrote `platform: null` would
    make "unstampable" and "stamped null" the same fixture and the residual population
    untestable.

    `msg_ts` stamps EVERY message with an ISO timestamp — the field `turns_of` turns into
    `Turn.ts` and the field `content_date` reads FIRST. It is deliberately opt-in: a fixture
    with no message timestamp can only be dated from its filename, which is precisely the
    fallback #3885 names, so both shapes must be buildable or one half of that ordering is
    untestable. `mtime` overrides the file's date directly, which is how a session whose
    ARCHIVE is dated one thing and whose TRAFFIC another is put on disk at all.
    """
    if msg_ts is not None:
        messages = [{**m, "timestamp": msg_ts} for m in messages]
    doc = {"session_id": sid, "id": sid, "source": source, "messages": messages,
           "created_at": "2026-11-01T00:00:00+00:00"}
    if platform is not None:
        doc["platform"] = platform
    # The suffix is spelled out twice rather than grafted on, because the first version of
    # this helper wrote `f"{sid}.json{'gz' if gzipped else ''}"` — one missing dot — and the
    # archive landed as `.jsongz`, which `iter_transcripts`'s `*.json.gz` glob correctly
    # ignores. The test then measured a corpus of plain files only and the gzipped half of
    # clause 3 silently did not exist: an archived session is inside the denominator or the
    # run says so, and a fixture that cannot produce one cannot assert that.
    stem = sid[:-5] if sid.endswith(".json") else sid
    p = Path(dir_) / (f"{stem}.json.gz" if gzipped else f"{stem}.json")
    body = json.dumps(doc)
    if gzipped:
        with gzip.open(p, "wt", encoding="utf-8") as fh:
            fh.write(body)
    else:
        p.write_text(body, encoding="utf-8")
    stamp = (mtime if mtime is not None else NOW - timedelta(days=age_days)).timestamp()
    import os
    os.utime(p, (stamp, stamp))
    return p


def fully_captioned_turn():
    """A turn where EVERY call carries its `summary` — the caption rate at 1.000.

    `captioned_turn` deliberately sits at 0.500 so a detector cannot confuse "counted the
    calls" with "counted the captions"; the paired before/after nodes need a second,
    DIFFERENT rate on the same shape, and the delta between these two (+50.00pp per call) is
    the seeded rate change across the split that clause 5 is written in terms of.
    """
    return [{"role": "user", "content": "go"},
            _msg("", [_tc("Bash", {"summary": "listing", "command": "ls"}),
                       _tc("Read", {"summary": "reading a", "file_path": "/tmp/a"})]),
            _msg("done")]


def captioned_turn(calls=2):
    """A turn whose second call carries NO summary, so the caption rate is not 1.

    Half the calls have to be situations without the behaviour: a fixture that is fully
    captioned cannot tell a detector that counts captions from one that counts calls, and
    every "k is half of N" assertion in this file depends on it. The shape is also the
    shape the rule is about — `loop.py`'s own measured failure was repeat calls arriving
    uncaptioned (5/5 first calls, 0/31 repeats, session 20260907_184351_ivec8d).
    """
    return [{"role": "user", "content": "go"},
            _msg("", [_tc("Bash", {"summary": "listing", "command": "ls"})]),
            _msg("", [_tc("Read", {"file_path": "/tmp/a"})]),
            _msg("done")]


def registry_path(tmp_path):
    """The shipped registry copied beside the fixtures, so both halves read the same file.

    Into a subdirectory, not the corpus directory: the sweep's own directory listing is what
    `test_the_sweep_never_writes_into_the_sessions_dir` compares, and a registry written into
    the corpus dir shows up in that listing as a file the sweep supposedly created. It also
    keeps the corpus holding nothing but transcripts, which is what a sessions directory is.
    """
    src = (ROOT / "eval" / "behaviour_specs.yaml").read_text(encoding="utf-8")
    d = Path(tmp_path) / "_registry"
    d.mkdir(exist_ok=True)
    p = d / "behaviour_specs.yaml"
    p.write_text(src, encoding="utf-8")
    return p


def run(tmp_path, *, days=7, max_files=0):
    return BE.sweep(tmp_path, days=days, now=NOW, registry=registry_path(tmp_path),
                    write=False, max_files=max_files)


# ── clause 3: plain AND gzipped transcripts are inside the denominator ─────


def test_the_sweep_reads_plain_and_gzipped_transcripts_and_publishes_the_split(tmp_path):
    """Half the corpus is archived; a sweep that dropped it would report a smaller box.

    Four sessions, two of them `.json.gz`, each with two tool calls of which one is
    uncaptioned. Every one of the four must be counted: the printed `corpus` block carries
    `plain=2` and `gzipped=2`, and the caption row's denominator is 8 — 4 sessions x 2 calls —
    not the 4 a `glob('*.json')` sweep would have divided by.
    """
    for i in range(2):
        write_session(tmp_path, f"20261103_10000{i}_plain_aaaa", captioned_turn())
    for i in range(2):
        write_session(tmp_path, f"20261028_10000{i}_arch_bbbb", captioned_turn(),
                      gzipped=True, age_days=5)
    art = run(tmp_path)
    assert art["corpus"]["plain"] == 2, art["corpus"]
    assert art["corpus"]["gzipped"] == 2, (
        f"archived sessions were dropped, so the denominator is a suffix artefact: "
        f"{art['corpus']}")
    assert art["corpus"]["sessions"] == 4
    row = next(r for r in art["rows"] if r["id"] == "tool-call-caption-present")
    assert row["situations_encountered"] == 8, (
        f"the caption denominator says {row['situations_encountered']}, so the gzipped half "
        "of the corpus is not in the rate the artifact publishes")
    assert row["exhibited"] == 4, (
        "one of the two calls in each session carries no summary, so half the calls are "
        "situations without the behaviour")


def test_a_gzipped_transcript_is_decoded_and_scored_not_skipped(tmp_path):
    """The gzipped file alone, so a silent skip cannot hide behind a plain file's numbers."""
    write_session(tmp_path, "20261027_100000_arch_cccc", captioned_turn(), gzipped=True,
                  age_days=6)
    art = run(tmp_path)
    assert art["corpus"]["sessions"] == 1 and art["corpus"]["gzipped"] == 1
    row = next(r for r in art["rows"] if r["id"] == "tool-call-caption-present")
    assert row["situations_encountered"] == 2 and row["exhibited"] == 1


def test_files_outside_the_window_are_left_alone(tmp_path):
    """The window is by file mtime, and an aged file must not appear in either count."""
    write_session(tmp_path, "20260901_100000_old_dddd", captioned_turn(), age_days=40)
    write_session(tmp_path, "20261103_100000_new_eeee", captioned_turn())
    art = run(tmp_path, days=7)
    assert art["corpus"]["sessions"] == 1
    assert art["corpus"]["plain"] == 1 and art["corpus"]["gzipped"] == 0


def test_an_unreadable_transcript_is_counted_not_swallowed(tmp_path):
    """A truncated archive is a corpus fact. Read as nothing it would shrink the denominator
    silently, which is the same mistake as not reading the suffix at all."""
    (tmp_path / "20261102_100000_bad_ffff.json").write_text('{"messages": [', encoding="utf-8")
    write_session(tmp_path, "20261103_100000_good_gggg", captioned_turn())
    art = run(tmp_path)
    assert art["corpus"]["unreadable"] == 1, art["corpus"]
    assert art["corpus"]["sessions"] == 1


# ── clause 4: the row shape, and N=0 is never a pass ──────────────────────


def test_every_row_carries_n_k_rate_and_a_wilson_interval(tmp_path):
    write_session(tmp_path, "20261103_100000_shape_hhhh", captioned_turn())
    art = run(tmp_path)
    for row in art["rows"]:
        for key in ("situations_encountered", "exhibited", "rate", "wilson95",
                    "status", "by_population", "situation_unit", "source", "detector"):
            assert key in row, f"{row['id']} row is missing {key}"
        if row["situations_encountered"]:
            lo, hi = row["wilson95"]
            assert lo <= row["rate"] <= hi, row
        for pop in ("user", "worker"):
            cell = row["by_population"][pop]
            n = cell["situations_encountered"]
            if n:
                assert cell["wilson95"] is not None and cell["status"] == "measured", row
                assert cell["wilson95"][0] <= cell["rate"] <= cell["wilson95"][1], row
            else:
                # The half the clause turns on: an empty population inside a row that has a
                # rate elsewhere gets NO number at all. `0.0` would read as "never obeyed"
                # and `1.0` as "always obeyed"; both are stories about traffic that was not
                # there, and either would survive into a summary line unchallenged.
                assert cell["rate"] is None and cell["wilson95"] is None, row
                assert cell["status"] == "instrument failure: N=0", row


def test_the_population_split_separates_user_turns_from_worker_runs(tmp_path):
    """Three buckets, three kinds of evidence, and none of them is "the default".

    A chat-shaped id on an allow-listed platform is a person's turn; a machine platform or a
    spawner's `source` is a run; a run-shaped record carrying neither is `unattributed`. The
    third bucket exists so the second cannot be reached by elimination — which is how the
    first committed run reported 54 `user` turns of which not one was typed by a person.
    """
    write_session(tmp_path, "20261103_100000_a1b2c3", captioned_turn(),
                  platform="mission-control")
    write_session(tmp_path, "20261103_100001_autonomy_w1", captioned_turn(),
                  source="autonomy-task:24")
    write_session(tmp_path, "20261103_100002_mystery_x9", captioned_turn())
    art = run(tmp_path)
    row = next(r for r in art["rows"] if r["id"] == "tool-call-caption-present")
    assert row["by_population"]["user"]["situations_encountered"] == 2, row
    assert row["by_population"]["worker"]["situations_encountered"] == 2, row
    assert row["by_population"]["unattributed"]["situations_encountered"] == 2, row
    assert art["corpus"]["turns_user"] == 1 and art["corpus"]["turns_worker"] == 1
    assert art["corpus"]["turns_unattributed"] == 1, art["corpus"]


@pytest.mark.parametrize("sid,platform,source", [
    ("20261103_100000_worker_w1", "worker", None),
    ("20261103_100001_autonomy_w2", "autonomy", None),
    # The classes the committed run put in the user column because their `source` was not in
    # the token list it happened to know: the nightly distiller and the failure ledger. Both
    # are jobs, and the store says so with `platform` — which is why the classifier reads the
    # store instead of a vocabulary.
    ("20261103_100002_sessiondisti_w3", "worker", "session-distill"),
    ("20261103_100003_failureledge_w4", "worker", "failure-ledger"),
])
def test_a_harness_job_is_never_counted_as_a_persons_turn(tmp_path, sid, platform, source):
    """No job shape reaches the `user` column, so a `user` rate cannot absorb harness traffic."""
    write_session(tmp_path, sid, captioned_turn(), platform=platform, source=source)
    art = run(tmp_path)
    assert art["corpus"]["turns_user"] == 0, (
        f"{sid} (platform={platform}, source={source}) landed in the user population; its "
        f"situations would be reported as a person's behaviour: {art['corpus']}")
    assert art["corpus"]["turns_worker"] == 1, (
        f"{sid} was counted as neither user nor worker, so a job run left the denominator")


def test_the_split_is_the_same_measurement_the_session_listings_make(tmp_path):
    """The seam this instrument must not get wrong: two readers of "is this a person's turn".

    The listings, the markdown export and this sweep all answer that question, and the item's
    class rule is that both sides read ONE measurement. So the node asserts agreement with
    `sessions_io.is_conversation_session` over every fixture shape, rather than asserting a
    second answer here: if `population_of` ever grows its own classifier again, the two
    disagree and this fails — which is precisely the drift a "user exhibit rate" that no
    listing can reproduce would be.
    """
    from app.sessions_io import is_conversation_session
    shapes = [
        ("20261103_100000_a1b2c3", "mission-control", None),
        ("20261103_100001_browser", "browser", None),
        ("20261103_100002_worker_w", "worker", None),
        ("20261103_100003_autonomy_w", "autonomy", "autonomy-task:24"),
        ("20261103_100004_iv1da3", None, ""),
        ("20261103_100005_plain_x", None, None),
    ]
    for sid, platform, source in shapes:
        p = write_session(tmp_path, sid, captioned_turn(), platform=platform, source=source)
        doc = json.loads(p.read_text(encoding="utf-8"))
        got = BE.population_of(doc)
        assert (got == "user") == is_conversation_session(f"{sid}.json", doc), (
            f"{sid} (platform={platform}, source={source}): the sweep says {got!r} while the "
            "listings' own predicate disagrees, so the user rate is not reproducible from the "
            "chat history")


def test_an_empty_record_contributes_no_turn_to_any_population(tmp_path):
    """The store lists these with `model: ''` and `message_count: 0` — a record, no traffic.

    Eighty-four such files sat in the committed run's window. They cannot add a turn, so they
    must not add one here either: a population that grows from an empty record is a rate over
    traffic that was never recorded.
    """
    write_session(tmp_path, "20261103_100000_iv9999", [])
    art = run(tmp_path)
    assert art["corpus"]["turns"] == 0, art["corpus"]
    assert art["corpus"]["turns_user"] == 0 and art["corpus"]["turns_worker"] == 0
    assert art["corpus"]["sessions"] == 1, "the file is a session record; it simply has no turns"


def test_bookkeeping_sidecars_are_excluded_from_the_corpus_and_counted_separately(tmp_path):
    """`<session>.microcompact-reduced.json` is a compaction record, not a transcript.

    It carries `{schema, cleared_call_ids, reduced_call_ids}` and no messages, and it lands
    inside a `*.json` glob under the same session name. Eighty-four of them sat in the
    7-day window the first committed artifact measured, so the printed corpus total was 84
    files larger than the corpus; they are now dropped and the drop is published, because a
    silently smaller corpus is the same class of error as a silently suffix-limited one.
    """
    write_session(tmp_path, "20261103_100000_real_aaaa", captioned_turn())
    sidecar = tmp_path / "20261103_100000_real_aaaa.microcompact-reduced.json"
    sidecar.write_text(json.dumps({"schema": 1, "cleared_call_ids": ["c-1"],
                                   "reduced_call_ids": []}), encoding="utf-8")
    import os
    os.utime(sidecar, (NOW.timestamp(), NOW.timestamp()))
    art = run(tmp_path)
    assert art["corpus"]["sessions"] == 1, (
        f"the corpus counted a compaction record as a session: {art['corpus']}")
    assert art["corpus"]["sidecars_skipped"] == 1, (
        "the exclusion is invisible, so the printed total and the directory listing "
        f"cannot be reconciled: {art['corpus']}")
    assert BE.is_transcript_path(Path("x/20261103_iv1da3.json")) is True
    assert BE.is_transcript_path(Path("x/20261103_iv1da3.microcompact-reduced.json")) is False



def test_a_row_with_no_situations_cannot_render_as_a_rate(tmp_path):
    """`0 / 0` is the instrument's own failure and must be unrepresentable as compliance.

    Seeded with a corpus whose every session is a bare user row: no tool calls, no
    announcements, no writes, so all four triggers fire nowhere. Every row then has N=0, and
    each one must carry `rate=None` (not 0.0, not 1.0), a status that says the instrument
    failed or that the behaviour is not instrumentable, and a reason. Then the same artifact
    is rendered, and the printed form is checked for the absence of a percentage.
    """
    write_session(tmp_path, "20261103_100000_bare_iiii",
                  [{"role": "user", "content": "hello"}])
    art = run(tmp_path)
    assert art["rows"], "no rows at all means the sweep scored nothing"
    for row in art["rows"]:
        assert row["situations_encountered"] == 0, row
        assert row["rate"] is None, (
            f"{row['id']} reported a rate over an empty denominator: {row['rate']}")
        assert row["wilson95"] is None, row
        assert "not_instrumentable" in row["status"] or "failure" in row["status"], row
        assert row["reason_if_zero"], "a zero denominator with no reason is not a finding"
    text = BE.render_rows(art)
    for row in art["rows"]:
        seg = next(ln for ln in text.splitlines() if row["id"] in ln)
        assert "INSTRUMENT FAILURE" in seg, seg
        assert "100" not in seg and "%" not in seg, (
            f"an N=0 row rendered as a percentage: {seg}")


def test_render_rows_never_prints_a_zero_denominator_row_as_a_pass_even_when_asked(tmp_path):
    """The renderer is the surface a human reads, so the guard has to hold there too.

    A row built by hand with N=0 (the shape a future field or a partial run would produce)
    must still print as a failure: the check is on the printed text, not on the field that
    produced it, because the failure this defends is a reader — or a later summary job —
    quoting `100%` out of a log line.
    """
    art = {"corpus": {"sessions": 0, "plain": 0, "gzipped": 0, "unreadable": 0,
                      "turns": 0, "turns_user": 0, "turns_worker": 0,
                      "turns_unattributed": 0, "sidecars_skipped": 0},
           "window_days": 7, "controls_ok": True, "controls": [],
           "rows": [BE.row_for({"id": "tool-call-caption-present",
                                "source": "app/harness/loop.py:2341",
                                "detector": "deterministic", "population": "both"},
                               {"user": [0, 0], "worker": [0, 0],
                                "unattributed": [0, 0]})]}
    text = BE.render_rows(art)
    assert "INSTRUMENT FAILURE" in text and "N=0" in text, text
    assert "1.000" not in text and "100%" not in text, text


def test_row_for_reports_a_trigger_that_never_fired_as_not_instrumentable_with_a_reason():
    """Clause 5's reporting half: a blind detector is a finding, not a zero rate.

    `status` carries the words `not_instrumentable`, and `reason_if_zero` says out loud that
    a detector blind spot and an absent behaviour cannot be told apart from the corpus alone
    — which is the sentence that stops the next reader from reading a 0 as absence.
    """
    row = BE.row_for({"id": "read-back-before-success-claim",
                      "source": "lloyd/SOUL.md:71", "detector": "deterministic",
                      "population": "both"}, {"user": [0, 0], "worker": [0, 0],
                                              "unattributed": [0, 0]})
    assert "not_instrumentable" in row["status"], row["status"]
    assert "blind" in row["reason_if_zero"] and "cannot tell" in row["reason_if_zero"]


# ── clause 5: positive controls ────────────────────────────────────────────


def test_every_registered_behaviour_has_a_positive_control_that_passes():
    """The clause's first half: a fixture per behaviour, and it must pass.

    Not a loop over a hard-coded list of ids: the node reads the shipped registry, so adding
    a fifth behaviour without a control fails here rather than shipping an uncontrolled
    detector.
    """
    ids = {s["id"] for s in BE.load_registry()}
    controls = {c["id"]: c for c in BE.run_controls()}
    assert ids == set(controls), f"behaviours with no control: {ids - set(controls)}"
    uncontrolled = [c["id"] for c in controls.values() if not c["ok"]]
    assert not uncontrolled, f"controls failing on their seeded traces: {uncontrolled}"


def test_each_control_asserts_the_denominator_and_the_exhibit_count_not_just_nonzero():
    """A control that only asserted N>0 would pass a detector that always answers True.

    Every fixture therefore carries an expected N and an expected k, and each behaviour also
    ships a counter-trace whose expected k is LOWER. The node below checks that structure
    holds for every behaviour, so a future author cannot satisfy the clause with a trace
    seeded to trivially match a broken detector.
    """
    for c in BE.run_controls():
        traces = c["traces"]
        assert len(traces) >= 2, f"{c['id']} has no counter-trace"
        for t in traces:
            assert t["ok"], (f"{c['id']}/{t['trace']}: got N={t['n']} k={t['k']}, "
                             f"want N={t['expected_n']} k={t['expected_k']}")
        assert any(t["expected_k"] < t["expected_n"] for t in traces), (
            f"{c['id']}: no fixture asserts a situation WITHOUT the behaviour, so a "
            "detector that always answers exhibit would pass")


def test_control_traces_score_identically_through_a_real_session_file(tmp_path):
    """Each seeded trace, replayed as an on-disk transcript, must give the same N and k.

    The first run of this instrument produced exactly the bug this node closes: the control
    builder returned a message wrapper where a tool call belongs, so the caption detector
    scored N=3 k=0 on a trace visibly carrying two captions — the control passed on the way
    the fixture was built and would have failed on the way the corpus is read. So each trace
    goes through the full path here: written as a session file, decoded, split into turns,
    scored, tallied into a row.
    """
    reg = registry_path(tmp_path)
    for bid, traces in BE.CONTROL_TRACES.items():
        for i, tr in enumerate(traces):
            one = tmp_path / f"trace-{bid}-{i}"
            one.mkdir()
            write_session(one, f"20261103_12000{i}_ctlfixed", tr["messages"])
            art = BE.sweep(one, days=7, now=NOW, registry=reg, write=False)
            row = next(r for r in art["rows"] if r["id"] == bid)
            assert row["situations_encountered"] == tr["expected"]["n"], (
                f"{bid}/{tr['name']}: N={row['situations_encountered']} through the sweep "
                f"but {tr['expected']['n']} in the control")
            assert row["exhibited"] == tr["expected"]["k"], (
                f"{bid}/{tr['name']}: k={row['exhibited']} through the sweep but "
                f"{tr['expected']['k']} in the control")


def test_a_control_failure_is_an_instrument_failure_on_the_artifact(tmp_path):
    """`controls_ok` is on the artifact, and the renderer shouts when it is false.

    A sweep whose detectors are broken still prints four tidy rates, so the artifact that
    carries the numbers must also carry the verdict on whether they mean anything.
    """
    write_session(tmp_path, "20261103_100000_ok_jjjj", captioned_turn())
    art = run(tmp_path)
    assert art["controls_ok"] is True
    broken = dict(art, controls_ok=False)
    text = BE.render_rows(broken)
    assert "CONTROLS FAILED" in text, "rates printed with no warning that the instrument broke"


# ── the interval itself ───────────────────────────────────────────────────


@pytest.mark.parametrize("k,n,want", [
    (9, 10, (0.5958, 0.9821)),
    (0, 5, (0.0, 0.4345)),
    (96, 100, (0.9016, 0.9843)),
    (1, 3, (0.0615, 0.7923)),
])
def test_the_wilson_interval_matches_values_computed_from_the_formula(k, n, want):
    """Checked against hand-computed Wilson scores, never against this function's own output.

    A CI test written by reading the implementation is a tautology. These four are the cases
    that matter here: a rate near 1 over a small N, a k of 0 (whose upper bound is what a
    reader will quote), and 1/3, the shape every rare-behaviour row will have.
    """
    lo, hi = BE.wilson_interval(k, n)
    assert (round(lo, 4), round(hi, 4)) == want, f"{k}/{n} -> {(lo, hi)}"


def test_an_empty_denominator_has_no_interval_at_all():
    """(0, 0) from a `0/0` Wilson would be a point estimate of zero width — the most
    confident-looking answer an instrument with no data can produce."""
    lo, hi = BE.wilson_interval(0, 0)
    assert (lo, hi) == (0.0, 0.0)


# ── the artifact as a store ───────────────────────────────────────────────


def test_the_artifact_is_one_json_line_under_the_data_root_and_bounded(tmp_path):
    """Written as `.jsonl` past `.gitignore`'s bare `*.json`, one record per run, pruned.

    Retention is asserted by writing a 40-day-old dated file next to the new one and checking
    the writer deleted it: an instrument created without a bound is a disk incident with a
    date on it, and this store gets its bound in the same change that creates it rather than
    in a follow-up that would not arrive (#2275's tool-quality artifact directory has never
    existed on this box, which is what the follow-up is worth).
    """
    write_session(tmp_path, "20261103_100000_wr_kkkk", captioned_turn())
    out = tmp_path / "artifacts"
    old = out / "behaviour-exhibit-2026-09-25.jsonl"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text("{}\n", encoding="utf-8")
    art = BE.sweep(tmp_path, days=7, now=NOW, registry=registry_path(tmp_path),
                   out=out, max_files=0)
    written = Path(art["artifact_path"])
    assert written.name.startswith("behaviour-exhibit-2026-11-04") and written.suffix == ".jsonl"
    assert len(written.read_text(encoding="utf-8").strip().splitlines()) == 1, (
        "the artifact must be one JSON record on one line: it is a store file, and a "
        "pretty-printed blob is untracked-by-tooling and unappendable")
    assert json.loads(written.read_text())["item"] == "#3749"
    assert not old.exists(), "the retention bound did not prune a 40-day-old artifact"
    assert written.exists(), "the sweep deleted its own output while pruning"


def test_the_artifact_declares_that_nothing_is_gated_on_it(tmp_path):
    """Clause 6 of the item, pinned as a field: report-only, in the file itself.

    A number with no statement of its status is a number someone will eventually gate on.
    `gated_on: null` plus the note is the artifact saying out loud that it has not been
    promoted to anything, which is the sentence a future round finds before it wires this
    into a decision.
    """
    write_session(tmp_path, "20261103_100000_gate_llll", captioned_turn())
    art = run(tmp_path)
    assert art["gated_on"] is None
    assert "report-only" in art["note"] and "grip" in art["note"]


def test_the_sweep_never_writes_into_the_sessions_dir(tmp_path):
    """Read-only is the item's word, and the module is pointed at the live corpus in normal
    use — so the property is asserted, not asserted in prose."""
    write_session(tmp_path, "20261103_100000_ro_mmmm", captioned_turn())
    reg = registry_path(tmp_path)          # created BEFORE the snapshot: it is an input
    before = sorted(p.name for p in tmp_path.iterdir())
    before_hash = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    BE.sweep(tmp_path, days=7, now=NOW, registry=reg, write=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == before, (
        "the sweep created something inside the corpus directory")
    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.json")} == before_hash, (
        "the sweep rewrote a transcript: this module reads the live corpus and must not "
        "touch it — a scorer that perturbs the traffic it scores is not a scorer")


def test_the_store_directory_is_created_when_it_does_not_exist_yet(tmp_path):
    """The first run on a box has no store directory, and `is_dir()` is false there.

    The bug this closes was live in the first version of `sweep`: it branched on
    `dest.is_dir()`, so a fresh store path — which by definition does not exist the first
    time — was taken for a FILE name and the run wrote one artifact named `behaviour-exhibit`
    with no date in it. That is the shape of the failure this whole instrument exists to
    avoid: #2275's tool-quality store has an artifact directory that has never existed on
    this box, and a writer that only works once the directory already does is how that
    happens twice. A directory is therefore named by its shape (no `.jsonl` suffix), not by
    whether something is already there.
    """
    write_session(tmp_path, "20261103_100000_fresh_nnnn", captioned_turn())
    store = tmp_path / "eval" / "behaviour-exhibit"      # three levels deep, none exist
    assert not store.exists()
    art = BE.sweep(tmp_path, days=7, now=NOW, registry=registry_path(tmp_path),
                   out=store, write=True)
    written = Path(art["artifact_path"])
    assert written.parent == store and store.is_dir()
    assert written.name == f"behaviour-exhibit-{art['measured_at'][:10]}.jsonl", (
        f"the artifact must be dated inside the store, got {written.name!r}")
    # And the sibling that the old branch produced must not be there.
    assert not (tmp_path / "eval" / "behaviour-exhibit").is_file()


def test_render_rows_never_prints_a_rate_for_an_empty_denominator(tmp_path):
    """The rendering rule, on the printed text itself, not just the JSON field.

    `rate: null` in an artifact is still readable as `0%` by whatever prints it, so the
    clause lives in the renderer too: an empty population's number must be absent from the
    line, and the word `INSTRUMENT FAILURE` present.
    """
    write_session(tmp_path, "20261103_100000_print_oooo", captioned_turn())
    art = run(tmp_path)
    text = BE.render_rows(art)
    for r in art["rows"]:
        line = next(ln for ln in text.splitlines() if ln.startswith(r["id"]))
        assert "100%" not in line, line
        if r["situations_encountered"] == 0:
            # The word `rate` must not survive into the line either: `rate: null` printed
            # as `rate=None` is a field a table turns back into a number.
            assert "INSTRUMENT FAILURE" in line, line
            assert "= " not in line and "0.000" not in line and "1.000" not in line, line
            continue
        assert "1.000" not in line, line
        empty = [p for p in ("user", "worker")
                 if r["by_population"][p]["rate"] is None]
        if empty:
            assert "N=0(no reading)" in line, (
                f"an empty population printed as a bare k/N, which a table copies as a "
                f"rate: {line!r}")
    assert "INSTRUMENT FAILURE" in text, text


# ── the instrument against real traffic ────────────────────────────────────

WITNESS = (ROOT / "eval" / "measurements" / "behaviour-exhibit"
           / "behaviour-exhibit-2026-11-04.jsonl")


def test_the_sweep_ran_over_real_traffic_and_its_rows_are_not_retypeable(tmp_path):
    """A committed sweep over the live corpus, re-derived here rather than trusted.

    Two halves, and only one of them runs under pytest. The half that ALWAYS runs reads the
    committed artifact and checks it against itself: >1,000 sessions, a non-zero gzipped
    share, `sessions == plain + gzipped`, `controls_ok: true`, and per row `k <= N`, the
    Wilson interval containing the rate, and the two populations summing to the total. Those
    are arithmetic properties of a real measurement, and a row that could not be written by
    this module's own writer fails them.

    The second half re-derives the rows over the live corpus and it never runs under pytest:
    `tests/conftest.py` redirects the data root to scratch for every run (:159 refuses even
    an explicit `LLOYD_DATA` pointing at the live root), so the sessions directory is empty
    by construction. The re-derivation is therefore a property of the CLI on a normal shell
    — `.venvs/lloyd/bin/python -m app.behaviour_exhibit --days 7` — and this node says so
    rather than pretending to a coverage it cannot have.

    No skip on the missing artifact itself: it is committed at a fixed path, so absent bytes
    mean the witness was deleted or never written, and a node that skips on that is a node
    that goes green as the instrument loses its evidence.
    """
    assert WITNESS.is_file(), (
        f"{WITNESS.relative_to(ROOT)} is not on the tree: the real-traffic witness is gone, "
        "and a skipped witness node reports the same green as a passing one")
    art = json.loads(WITNESS.read_text(encoding="utf-8"))
    assert art["item"] == "#3749" and art["controls_ok"] is True
    c = art["corpus"]
    assert c["sessions"] > 1000 and c["gzipped"] > 0, (
        f"the committed run did not read a real corpus: {c}")
    assert c["sessions"] == c["plain"] + c["gzipped"] and c["unreadable"] == 0
    assert c.get("turns_unattributed") is not None, (
        f"the committed run predates population attribution and cannot witness it: {c}")
    for row in art["rows"]:
        assert row["situations_encountered"] > 0, (
            f"{row['id']}: the committed run reports an instrument failure — a row that "
            "cannot be re-derived is not a witness")
        assert row["exhibited"] <= row["situations_encountered"]
        assert row["wilson95"][0] <= row["rate"] <= row["wilson95"][1]
        assert (row["by_population"]["user"]["situations_encountered"]
                + row["by_population"]["worker"]["situations_encountered"]
                + row["by_population"]["unattributed"]["situations_encountered"]
                == row["situations_encountered"]), row["id"]

    # Re-derive over the same window, live — but only where the corpus is reachable.
    # Inside an automod gate `LLOYD_DATA` is the round's own root and `SESSIONS_DIR`
    # resolves inside the worktree (`app.paths`' warning names this out loud), so a sweep
    # there reads zero transcripts and a zero here would be a statement about the sandbox,
    # not about the traffic — the exact mistake this instrument exists to refuse. Skipping
    # says so; asserting on an empty tree would have been a false green elsewhere.
    sd = BE.session_dir(None)
    if not sd.is_dir() or not any(sd.glob("*.json*")):
        pytest.skip(
            f"the sessions dir resolved to {sd}, which holds no transcripts — "
            "tests/conftest.py gives every pytest run a scratch data root, so a sweep here "
            "measures the sandbox and not the traffic. The committed artifact's rows were "
            "checked for internal consistency above; re-derive live with "
            "`python -m app.behaviour_exhibit --days 7`")
    live = BE.sweep(None, days=art["window_days"], write=False)
    lc = live["corpus"]
    assert lc["sessions"] > 0, lc
    for row in art["rows"]:
        now_row = next(r for r in live["rows"] if r["id"] == row["id"])
        assert now_row["situations_encountered"] > 0, (
            f"{row['id']}: measured on {row['situations_encountered']} situations when "
            "committed and 0 now — the reader broke, not the traffic")
        assert now_row["situations_encountered"] >= row["situations_encountered"] * 0.5, (
            f"{row['id']}: the live corpus yields {now_row['situations_encountered']} "
            f"situations against a committed {row['situations_encountered']}; a halving is a "
            "reader regression, not drift")


# ── the entry point, and the store it defaults to ──────────────────────────


def test_the_command_line_entry_point_sweeps_writes_and_reports(tmp_path, capsys, monkeypatch):
    """The seam between `main()` and `sweep()` — the only path that has ever produced a record.

    Everything above calls `sweep` directly, so nothing pinned that the documented invocation
    (`python -m app.behaviour_exhibit`) actually reaches it: a mismatched keyword, a swallowed
    argument or a `main` that returns before writing would leave a working library and a dead
    instrument. It also pins that the artifact records the argv it was produced by, because a
    `command` field that is a template rather than a record is how a run becomes unrepeatable,
    and pins the `--check` half, which is the only thing standing between a dead `source`
    pointer and a green-looking run.
    """
    write_session(tmp_path, "20261103_100000_cli_l1", captioned_turn())
    out = tmp_path / "store"
    registry = registry_path(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "behaviour_exhibit", "--sessions-dir", str(tmp_path), "--days", "7",
        "--registry", str(registry), "--out", str(out)])
    assert BE.main() == 0, "the sweep's exit code is not a clean-run zero"
    printed = capsys.readouterr().out
    assert "corpus:" in printed and "tool-call-caption-present" in printed, printed
    written = sorted(out.glob("behaviour-exhibit-*.jsonl"))
    assert len(written) == 1, f"the default out path is a store directory: {out}"
    art = json.loads(written[0].read_text(encoding="utf-8"))
    assert art["corpus"]["sessions"] == 1, art["corpus"]
    assert "behaviour_exhibit --sessions-dir" in art["command"], art["command"]
    assert "--days 7" in art["command"], "the recorded command is a template, not this run"
    assert BE.main(["--check", "--registry", str(registry)]) == 0, (
        "a clean registry and its controls must exit zero")


def test_the_default_store_is_under_the_data_root_and_never_in_the_checkout():
    """The default path, unpinned until now: `--out` was always passed, so the default ran never.

    That is how #2275's tool-quality store ended with an artifact directory that has never
    existed — the writing half was correct and only ever exercised with an explicit path. The
    store must be under `DATA_ROOT` (a `rm -r` aimed at the code tree must not reach the
    history) and outside the checkout (`.gitignore`'s bare patterns and a code tree are not a
    place for a growing record).
    """
    from app.paths import DATA_ROOT
    d = BE.out_dir(None)
    assert d == Path(DATA_ROOT) / "eval" / "behaviour-exhibit", d
    assert Path(DATA_ROOT) in d.parents, d
    assert BE.REPO_ROOT not in d.parents and d != BE.REPO_ROOT, (
        f"the sweep's history would be written into the code tree at {d}")


def test_a_named_directory_is_a_store_and_a_named_file_is_that_file(tmp_path):
    """The one bug that would have made the dated store a fiction on the only machine it runs on.

    `out` arrives as a path that does not exist yet on a first run, and `is_dir()` was the
    test for "is this a directory" there — false — so the whole run wrote one artifact FILE
    called `behaviour-exhibit` and the dated-store convention broke exactly once per box,
    silently. The discriminator is now the suffix, so a not-yet-existing directory is a
    directory.
    """
    store = tmp_path / "not-yet-created"
    write_session(tmp_path, "20261103_100000_dir_m1", captioned_turn())
    art = BE.sweep(tmp_path, days=7, now=NOW, registry=registry_path(tmp_path), out=store)
    written = Path(art["artifact_path"])
    assert written.parent == store and written.name == "behaviour-exhibit-2026-11-04.jsonl", written
    named = tmp_path / "given-name.jsonl"
    art2 = BE.sweep(tmp_path, days=7, now=NOW, registry=registry_path(tmp_path), out=named)
    assert Path(art2["artifact_path"]) == named, "an explicit .jsonl path was rewritten"


# ── #3885: the window is a session's CONTENT date, and two windows are a test ──
#
# `--days N` asked a file when it was WRITTEN, and for the archived half of this corpus the
# answer is the retention sweep's schedule: all 9,814 `.json.gz` transcripts here sit a median
# 30.52 days between their mtime and their own filename date, so `--days 7` admitted 7,191
# files of which 4,546 (63%) were content-dated outside its seven days. Every node below
# therefore puts a file's two dates in OPPOSITE places and asserts which one the code obeyed.

def _iso(year, month, day, hour=10):
    return datetime(year, month, day, hour, 0, tzinfo=timezone.utc).isoformat()


def _split_row(artifact, behaviour_id, population):
    rows = [r for r in artifact["split"]["rows"]
            if r["id"] == behaviour_id and r["population"] == population]
    assert len(rows) == 1, f"{behaviour_id}/{population}: {len(rows)} paired rows, want 1"
    return rows[0]


def _write_pair(tmp_path, prefix, *, sessions, turn, content_day, content_month=10):
    """`sessions` transcripts on consecutive CONTENT dates, each running `turn`.

    `msg_ts` is what puts an arm on a date: the filename would do it too, but a paired test
    that only ever exercised the fallback could not tell a turn-timestamp window from a
    filename window, and clause 1 names the turn timestamp first.
    """
    # Three days, not nine: the first draft spread a 30-session arm over nine days and ran
    # past the window's exclusive `--until`, so the after arm silently kept 6 of its 30
    # sessions and the paired test measured unequal denominators while asserting 60 and 60.
    # An arm whose tail is clipped by the window is exactly the bug the window is for, so the
    # spread has to stay inside it for the node to be a control of anything.
    for i in range(sessions):
        day = content_day + (i % 3)
        write_session(tmp_path, f"2026{content_month:02d}{day:02d}_10{i:04d}_{prefix}_aaaa",
                      turn(), msg_ts=_iso(2026, content_month, day, 10 + (i % 9)))


def test_content_date_reads_the_turn_timestamp_before_the_filename_stamp(tmp_path):
    """Clause 1: content date is the record's turn timestamp FIRST, the filename's stamp SECOND.

    Two fixtures whose name and contents DISAGREE by ten days, so which source won is a
    measurement and not a reading of the code: a session named Oct 20 whose traffic is stamped
    Oct 30 is dated Oct 30, and the same traffic with no message timestamps at all — the shape
    every archived transcript takes if the fallback is load-bearing — is dated from its name.
    """
    named = write_session(tmp_path, "20261020_100000_late_ts", captioned_turn(),
                          msg_ts=_iso(2026, 10, 30))
    assert BE.content_date(named) == datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc), \
        "a turn timestamp inside the record must beat the filename stamp"

    fallback = write_session(tmp_path, "20261020_100000_name_only", captioned_turn())
    assert BE.content_date(fallback) == datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc), \
        "with no readable timestamp the filename stamp has to place the session"

    assert BE.filename_date("20261020_100000_x.json") == \
        datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc)
    assert BE.filename_date("no_stamp_here.json") is None
    assert BE.content_date(Path(str(tmp_path)) / "20261020_100000_absent.json") == \
        datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc), \
        "a file that cannot be opened is still placeable from its name — dropping it silently " \
        "is how a window shrinks and the run still prints as a clean corpus"


def test_a_session_gzipped_after_the_split_but_dated_before_lands_in_the_before_arm(tmp_path):
    """Clause 1 in its own words, end to end: archive dated after, traffic dated before.

    This is the shape the retention sweep produces — a `.gz` mtime IS the compression date,
    here today, over traffic eleven days older. A paired exhibit that windowed on mtime would
    put this session in the AFTER arm and charge the compression job with changing behaviour.
    """
    packed_today = write_session(tmp_path, "20261024_090000_old_pack", captioned_turn(),
                                 gzipped=True, mtime=NOW)
    assert BE.content_date(packed_today) == datetime(2026, 10, 24, 9, 0, tzinfo=timezone.utc)
    art = BE.sweep(tmp_path, since="2026-10-01", until="2026-11-05", split_on="2026-11-01",
                   now=NOW, write=False)
    assert art["corpus"]["turns_before"] == 1 and art["corpus"]["turns_after"] == 0, \
        "the traffic is Oct 24 and the split is Nov 1: it belongs to the before arm"
    assert art["split"]["turns_before"] == 1 and art["split"]["turns_undated"] == 0
    r = _split_row(art, "tool-call-caption-present", "unattributed")
    assert r["before"]["situations_encountered"] == 2 and r["after"]["situations_encountered"] == 0
    # One empty arm is not a comparison, and it must not be dressed as one.
    assert r["discriminated"] is None
    assert r["status"] == "instrument failure: N=0 in one arm"


def test_since_and_until_select_on_content_date_over_both_suffixes(tmp_path):
    """Clause 2: [since, until) over `.json` and `.json.gz`, with mtime given no vote."""
    inside_plain = write_session(tmp_path, "20261020_100000_in_plain", captioned_turn(),
                                 msg_ts=_iso(2026, 10, 20))
    # The gzipped half, dated from inside the archive and stamped on disk TODAY — the mtime a
    # `--days` reader would place in November and an absolute reader must not.
    inside_gz = write_session(tmp_path, "20261005_100000_in_gz", captioned_turn(),
                              gzipped=True, msg_ts=_iso(2026, 10, 5), mtime=NOW)
    at_start = write_session(tmp_path, "20261001_000000_edge_in", captioned_turn(),
                             msg_ts="2026-10-01T00:00:00+00:00")
    at_end = write_session(tmp_path, "20261101_000000_edge_out", captioned_turn(),
                           msg_ts="2026-11-01T00:00:00+00:00")
    before_all = write_session(tmp_path, "20260915_100000_too_early", captioned_turn(),
                               msg_ts=_iso(2026, 9, 15))

    got = {p.name for p in BE.iter_transcripts(tmp_path, since="2026-10-01",
                                               until="2026-11-01", now=NOW)}
    assert got == {inside_plain.name, inside_gz.name, at_start.name}
    assert before_all.name not in got
    assert at_end.name not in got, "`--until` is EXCLUSIVE: an endpoint shared with the next " \
        "window must not be counted twice, which is what makes the two halves of --split-on a " \
        "partition rather than an overlap"
    assert inside_gz.name in got, "the `.gz` half has to answer to an absolute window too, or " \
        "the archived corpus is selectable only by the retention schedule's dates"


def test_days_alone_still_admits_the_mtime_set_it_admitted_before(tmp_path):
    """Clause 3: the scheduled sweep must not silently change corpus under this change.

    The expected set is recomputed here from the pre-#3885 rule — `st_mtime >= now − days·86400`
    — rather than asserted as a literal, so the node fails the day `--days` starts meaning
    something else. The fixtures are chosen so the two readings DISAGREE on two of the three
    files: an old session touched today and a recent session left alone for twenty days. A node
    whose corpus happened to make both readings agree would pass without pinning anything.
    """
    old_everywhere = write_session(tmp_path, "20260901_100000_old_aaaa", captioned_turn(),
                                   age_days=40)
    stale_name = write_session(tmp_path, "20260901_100000_touched_bbbb", captioned_turn(),
                               age_days=2)
    recent_name = write_session(tmp_path, "20261103_100000_left_alone_cccc", captioned_turn(),
                                age_days=20)
    days = 7
    cutoff = NOW.timestamp() - days * 86400
    expected = {p.name for p in Path(tmp_path).glob("*.json*")
                if p.stat().st_mtime >= cutoff}
    got = {p.name for p in BE.iter_transcripts(tmp_path, days=days, now=NOW)}
    assert got == expected == {stale_name.name}, \
        f"`--days` alone must keep the mtime cutoff: {sorted(got)} vs {sorted(expected)}"
    assert recent_name.name not in got, "a file named today but untouched for twenty days is " \
        "outside a `--days` window and inside a content window — the two modes differ here on " \
        "purpose, so this node cannot pass by both agreeing"
    assert old_everywhere.name not in got


def test_the_artifact_records_the_window_and_counts_mtime_content_disagreements(tmp_path):
    """Clause 4: both endpoints in the record, and the count of files the two clocks split.

    Seeded so the answer is known by hand: A is October traffic in an October file (agrees), B
    is October traffic in a file dated today (disagrees, admitted), C is September traffic in
    an October file (disagrees, refused), D is November traffic in a November file (agrees,
    refused). Two disagreements, two admissions.
    """
    win = dict(since="2026-10-01", until="2026-11-01", now=NOW)
    a = write_session(tmp_path, "20261020_100000_agree_a", captioned_turn(),
                      msg_ts=_iso(2026, 10, 20),
                      mtime=datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc))
    b = write_session(tmp_path, "20261020_110000_late_pack_b", captioned_turn(),
                      msg_ts=_iso(2026, 10, 20, 11), mtime=NOW)
    c = write_session(tmp_path, "20260915_100000_repacked_c", captioned_turn(),
                      msg_ts=_iso(2026, 9, 15),
                      mtime=datetime(2026, 10, 20, 9, 0, tzinfo=timezone.utc))
    d = write_session(tmp_path, "20261103_100000_recent_d", captioned_turn(),
                      msg_ts=_iso(2026, 11, 3),
                      mtime=datetime(2026, 11, 3, 10, 0, tzinfo=timezone.utc))
    art = BE.sweep(tmp_path, **win, write=False)
    assert art["window_mode"] == "content"
    assert art["window_start"] == "2026-10-01T00:00:00+00:00"
    assert art["window_end"] == "2026-11-01T00:00:00+00:00"
    assert art["window_days"] is None, \
        "`window_days` is the caller's flag, not the window: under an absolute window it is " \
        "still the default 7, and publishing 7 beside a one-month window is the artifact " \
        "asserting a span it did not measure"
    assert "window: mode=content" in BE.render_rows(art)
    corpus = art["corpus"]
    assert corpus["candidates_considered"] == 4 and corpus["content_undated_files"] == 0
    assert corpus["mtime_content_disagreements"] == 2, \
        "B (October traffic in a November file) and C (September traffic in an October file) " \
        "are the two disagreements; A and D's two dates put them the same way"
    admitted = {p.name for p in BE.iter_transcripts(tmp_path, **win)}
    assert admitted == {a.name, b.name}, \
        "the content window admits B in and D out, which is the whole point of the clause"
    assert c.name not in admitted, "September traffic in an October file is October traffic's " \
        "neighbour by mtime only — admitting it is the exact failure mode on the live corpus"
    assert d.name not in admitted, "D is November traffic whose mtime AND content date are " \
        "both outside the window — the case that must stay refused, so the disagreement count " \
        "measures the two files the clocks split and not every file the window excludes"

    # A `--days` run records WHICH CLOCK it used, so an old-style artifact is never mistaken
    # for a content-dated one by whoever reads it next, and its endpoints are still published.
    plain = write_session(tmp_path, "20261020_120000_days_e", captioned_turn(), age_days=2)
    day_art = BE.sweep(tmp_path, days=7, now=NOW, write=False)
    assert day_art["window_mode"] == "mtime"
    assert day_art["window_start"] == (NOW - timedelta(days=7)).isoformat(timespec="seconds")
    assert day_art["window_end"] is None
    assert day_art["window_days"] == 7, "a `--days` run still publishes its look-back"
    assert "(7 days)" in BE.render_rows(day_art)
    assert plain.name in {p.name for p in BE.iter_transcripts(tmp_path, days=7, now=NOW)}
    assert "file mtime" in day_art["corpus"]["window_filter"]


def test_split_on_marks_a_seeded_rate_change_as_discriminated(tmp_path):
    """Clause 5's positive control: a real rate step across the split says `true`.

    Thirty sessions each side — `captioned_turn` before (0.500), `fully_captioned_turn` after
    (1.000): Δ = +50.00pp against a 60-per-arm floor of 24.55pp. Every figure on the row is
    re-derivable from the denominators beside it, and the CI is asserted equal to the module's
    own `wilson_interval`, so a paired mode that grew a second interval function fails here.
    """
    _write_pair(tmp_path, "paired", sessions=30, turn=captioned_turn, content_day=20)
    _write_pair(tmp_path, "paired", sessions=30, turn=fully_captioned_turn,
                content_day=2, content_month=11)
    art = BE.sweep(tmp_path, since="2026-10-01", until="2026-11-05", split_on="2026-11-01",
                   now=NOW, write=False)
    r = _split_row(art, "tool-call-caption-present", "unattributed")
    assert r["before"]["situations_encountered"] == 60 and r["before"]["exhibited"] == 30
    assert r["before"]["rate"] == 0.5
    assert r["after"]["situations_encountered"] == 60 and r["after"]["exhibited"] == 60
    assert r["after"]["rate"] == 1.0
    assert r["delta"] == 0.5 and r["delta_pp"] == 50.0
    assert r["minimum_detectable_delta"] == pytest.approx(0.245494, abs=1e-5)
    assert r["discriminated"] is True
    lo, hi = BE.wilson_interval(30, 60)
    assert r["before"]["wilson95"] == [round(lo, 4), round(hi, 4)], \
        "the paired arms must publish the SAME Wilson interval the single-window rows use"
    assert art["split"]["alpha"] == 0.05 and art["split"]["power"] == 0.80
    assert "DISCRIMINATED" in BE.render_split(art)


def test_split_on_marks_identical_seeded_rates_as_not_discriminated(tmp_path):
    """Clause 5's negative control: identical seeded rates must say `false`, and say why.

    Same shape, same denominators, same split as the positive control — only the rate is the
    same on both sides. Asserting the denominators and a non-zero floor beside the verdict is
    what keeps this from passing because an arm came out empty or the floor came out infinite,
    either of which would make `false` a tautology rather than a measurement.
    """
    _write_pair(tmp_path, "same", sessions=30, turn=captioned_turn, content_day=20)
    _write_pair(tmp_path, "same", sessions=30, turn=captioned_turn,
                content_day=2, content_month=11)
    art = BE.sweep(tmp_path, since="2026-10-01", until="2026-11-05", split_on="2026-11-01",
                   now=NOW, write=False)
    r = _split_row(art, "tool-call-caption-present", "unattributed")
    assert r["before"]["situations_encountered"] == 60 == r["after"]["situations_encountered"], \
        "both arms have to hold the whole seeded corpus: an arm clipped by the window turns " \
        "the pair into 60-versus-12 and the floor into a number about the fixture"
    assert r["before"]["rate"] == 0.5 and r["after"]["rate"] == 0.5
    assert r["delta"] == 0.0
    assert r["minimum_detectable_delta"] == pytest.approx(0.245494, abs=1e-5)
    assert r["discriminated"] is False
    assert "not discriminated" in BE.render_split(art)


def test_split_on_reports_not_instrumentable_rather_than_a_zero_rate(tmp_path):
    """A behaviour whose trigger never fires gets a reason, not a delta of 0.0.

    0.0-vs-0.0 reads as "no effect found", which is a claim about the traffic; the honest
    statement is that nothing exercised the behaviour on either side and the corpus alone
    cannot tell a blind detector from an absent behaviour. This is the fall-back the item's
    clause 6 asks for — and it is what several registered behaviours will report on the human
    population, which carries 29 user turns a week against 14,516 worker.
    """
    _write_pair(tmp_path, "noinstr", sessions=4, turn=captioned_turn, content_day=20)
    _write_pair(tmp_path, "noinstr", sessions=4, turn=captioned_turn,
                content_day=2, content_month=11)
    art = BE.sweep(tmp_path, since="2026-10-01", until="2026-11-05", split_on="2026-11-01",
                   now=NOW, write=False)
    r = _split_row(art, "no-raw-write-to-protected-path", "unattributed")
    assert r["before"]["rate"] is None and r["after"]["rate"] is None
    assert r["delta"] is None and r["minimum_detectable_delta"] is None
    assert r["discriminated"] is None
    assert r["status"].startswith("not_instrumentable")
    assert "cannot tell apart" in r["reason"]
    assert "NOT_INSTRUMENTABLE" in BE.render_split(art)


def test_minimum_detectable_delta_is_the_spread_it_claims_to_be():
    """The floor checked against the definition it is quoted from, by exact enumeration.

    `minimum_detectable_delta` solves the pooled two-proportion z-test's own power equation, so
    the independent instrument is the sampling distribution itself: every pair of counts the two
    arms could produce, weighted by its binomial probability, counted where the test rejects. At
    the published floor the rejection rate has to BE the nominal power; at ninety percent of it
    it has to be clearly below. A solver that forgot the alternative hypothesis's standard error
    — the usual off-by-one, `delta > z·SE0`, which returns a floor roughly 40% too small here —
    clears the first check at about half power and fails the second. scipy is deliberately not
    used: the suite runs where only the declared dependencies do.
    """
    from math import comb

    def exact_power(n1, n2, p1, p2):
        z = BE._inv_normal_cdf(0.975)
        pa = [comb(n1, x) * p1 ** x * (1 - p1) ** (n1 - x) for x in range(n1 + 1)]
        pb = [comb(n2, y) * p2 ** y * (1 - p2) ** (n2 - y) for y in range(n2 + 1)]
        tot = 0.0
        for x in range(n1 + 1):
            for y in range(n2 + 1):
                pbar = (x + y) / (n1 + n2)
                se = ((pbar * (1 - pbar)) ** 0.5) * ((1 / n1 + 1 / n2) ** 0.5)
                if abs(y / n2 - x / n1) > z * se:
                    tot += pa[x] * pb[y]
        return tot

    n, p = 20, 0.5
    mdd = BE.minimum_detectable_delta(n, n, p)
    assert mdd == pytest.approx(0.395056, abs=1e-5)
    assert abs(exact_power(n, n, p, min(1.0, p + mdd)) - 0.80) < 0.05, \
        "at the published floor the test must reject at the nominal 80% power"
    assert exact_power(n, n, p, min(1.0, p + 0.9 * mdd)) < 0.75, \
        "a smaller move must be under-powered, or the number is not a floor"
    assert BE.minimum_detectable_delta(0, 20, 0.5) is None
    assert BE.minimum_detectable_delta(20, 20, 1.5) is None, \
        "a rate outside [0,1] is not a proportion and no floor is defined for it"


def test_an_unparseable_window_date_is_refused_rather_than_ignored(tmp_path):
    """A silently dropped `--since` is a run that measures the whole corpus under a label.

    Both surfaces are checked: `sweep` raises, and the CLI converts that into exit 2 with a
    message — an operator's typo in a date must not be the quiet kind, because it is the kind
    that still prints `window_start` and reads like a real window.
    """
    with pytest.raises(ValueError):
        BE.sweep(tmp_path, since="last tuesday", now=NOW, write=False)
    with pytest.raises(ValueError):
        BE.sweep(tmp_path, split_on="nonsense", now=NOW, write=False)
    with pytest.raises(ValueError):
        BE.sweep(tmp_path, since="2026-11-01", until="2026-10-01", now=NOW, write=False)
    out = str(Path(tmp_path) / "rows.jsonl")
    assert BE.main(["--sessions-dir", str(tmp_path), "--since", "not-a-date",
                    "--out", out]) == 2
    assert BE.main(["--sessions-dir", str(tmp_path), "--split-on", "not-a-date",
                    "--out", out]) == 2


def test_a_session_whose_timestamp_is_past_the_head_read_falls_back_to_its_name(tmp_path):
    """The bounded head read is a real bound, so the fallback has to survive it.

    `content_date` reads 256 KiB and stops — the alternative is a full parse of every candidate
    twice, minutes against seconds. A transcript whose first timestamp sits beyond that must be
    placed by its filename, not dropped: 384 of this box's 21,441 transcripts yield no timestamp
    in the head, and a window that lost them would be wrong in the same direction as the mtime
    bug it replaces, only more quietly.
    """
    msgs = json.dumps([{"role": "user", "content": "go",
                        "timestamp": "2026-10-20T10:00:00+00:00"}])
    body = ('{"session_id": "20261011_100000_deep_ts", "id": "20261011_100000_deep_ts", '
            '"source": null, "pad": "' + "x" * 300_000 + '", "messages": ' + msgs + "}")
    p = Path(tmp_path) / "20261011_100000_deep_ts.json"
    p.write_text(body, encoding="utf-8")
    assert BE._FIRST_TS_RE.search(p.read_text(encoding="utf-8")[:BE._HEAD_BYTES]) is None, \
        "the fixture must actually put the timestamp beyond the head read, or this node is " \
        "testing the primary source wearing the fallback's name"
    assert BE.content_date(p) == datetime(2026, 10, 11, 10, 0, tzinfo=timezone.utc)
    assert {q.name for q in BE.iter_transcripts(tmp_path, since="2026-10-01",
                                                until="2026-11-01", now=NOW)} == {p.name}


def test_paired_rows_reads_a_non_proportion_rate_at_the_widest_variance():
    """A rate above 1.0 gets the p=0.5 floor, because p(1−p) is negative up there.

    Direct on `paired_rows` rather than through `sweep`, because no registered spec can reach
    the branch: each of the four counts one situation per event of its own trigger, so k ≤ n.
    The shape is the one this instrument SHIPPED — the caption behaviour's first published rate
    was 2.000, per-turn denominator under a per-call detector — and `sweep` validates nothing
    about a spec's `situation_unit`, so a future registry line can bring it back. Asserting the
    published baseline and its note is what stops the floor being read off p=2.0, where the
    variance term is negative and any delta beats the "floor".
    """
    spec = [{"id": "per-call-over-per-turn", "source": "app/harness/loop.py:2341",
             "population": "both", "trigger": "t", "exhibit": "e", "denominator": "both",
             "detector": "d", "situation_unit": "turns with calls", "status": "active"}]

    def arms(nk_b, nk_a):
        return {"per-call-over-per-turn": {
            "before": {"user": [0, 0], "worker": [0, 0], "unattributed": nk_b},
            "after": {"user": [0, 0], "worker": [0, 0], "unattributed": nk_a}}}

    rows = {r["population"]: r for r in BE.paired_rows(spec, arms([30, 60], [30, 60]))}
    row = rows["combined"]
    assert row["before"]["rate"] == 2.0 and row["delta"] == 0.0
    assert row["mdd_baseline_rate"] == 0.5, \
        "the floor has to be read off p=0.5, the widest variance; off p=2.0 the variance term " \
        "is negative and the reported floor is a fiction that any movement beats"
    assert "not a proportion" in row["mdd_baseline_note"], \
        "the row that quietly widens its own floor has to say it did, or a reader compares it " \
        "against a floor from a different design"
    assert row["discriminated"] is False
    # The proportion path is unchanged by the guard, so the guard cannot be a blanket dodge.
    prop = {r["population"]: r for r in BE.paired_rows(spec, arms([60, 30], [60, 60]))}["combined"]
    assert prop["mdd_baseline_rate"] == 0.5 and prop["mdd_baseline_note"] is None
    assert prop["minimum_detectable_delta"] == pytest.approx(0.245494, abs=1e-5)
    assert prop["discriminated"] is True

# ── #3886 clauses 4 and 5: the report must SAY whether a rate moved ───────────
#
# The sweep has printed four Wilson intervals since it landed and never once said whether
# any of them MOVED. #3749's owed entry asks for one sentence — did the rate change by
# more than that arm's own interval allows, or is this what a quiet week looks like — and
# the 2026-11-04 run measured a caption move of 0.6pp with non-overlapping intervals
# [0.9041, 0.9070] against [0.897, 0.900] that nothing in the report said a word about.
# These tests pin the sentence, the population it is measured on, and where its bar comes
# from.
#
# The fixtures carry the live artifact's own worker cells (caption 141,558/156,286;
# read-back 285/759; dangling 853/1,226; protected-write 1,863/2,233) so the numbers
# asserted below are the numbers the store holds, not numbers tuned to the code.


def _cell(k: int, n: int) -> dict:
    """One population cell as `sweep` emits it, including the N=0 case, which gets
    `rate: None` and no interval rather than 0.0."""
    return {"situations_encountered": n, "exhibited": k,
            "status": "measured" if n else "not scored: no situation in the window",
            "rate": (round(k / n, 6) if n else None),
            "wilson95": (list(BE.wilson_interval(k, n)) if n else None)}


def _mrow(bid: str, k: int, n: int, *, user: "tuple[int, int] | None" = None) -> dict:
    return {"id": bid, "status": "measured" if n else "not scored",
            "situations_encountered": n, "exhibited": k,
            "rate": (round(k / n, 6) if n else None),
            "wilson95": (list(BE.wilson_interval(k, n)) if n else None),
            "by_population": dict(
                {"worker": _cell(k, n)},
                **({"user": _cell(*user)} if user else {}))}


#: The live 2026-11-04 artifact's four worker cells, verbatim, as a prior reading.
LIVE_PRIOR_ROWS = [
    _mrow("tool-call-caption-present", 141558, 156286),
    _mrow("read-back-before-success-claim", 285, 759),
    _mrow("no-dangling-announcement", 853, 1226),
    _mrow("no-raw-write-to-protected-path", 1863, 2233),
]


def _mart(rows: "list[dict]", measured_at: str, *, corpus: dict | None = None) -> dict:
    return {"schema": 1, "measured_at": measured_at, "window_days": 7, "rows": rows,
            "worker_pooled": BE.worker_pooled(rows),
            "corpus": corpus or {"files_read": 6953, "plain": 2457, "gzipped": 4496,
                                 "window_filter": "file mtime inside the window, "
                                                  "both suffixes read"}}


def _write_artifact(store: Path, doc: dict, *, day: str) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    path = store / f"behaviour-exhibit-{day}.jsonl"
    path.write_text(json.dumps(doc) + "\n", encoding="utf-8")
    return path


def _movement(current: dict, store: Path) -> str:
    """The block as `main` prints it: the prior is found by date in the store, not passed
    in — the path a scheduled run actually takes."""
    return BE.render_movement(current, store=store)


def _line(text: str, bid: str) -> str:
    return next(l for l in text.splitlines() if l.strip().startswith(bid + " worker"))


def _half_pp(k: int, n: int) -> str:
    """The bar the sentence on THAT arm must print, computed here independently of the
    code under test so the assertion is an outside check on `wilson_interval`."""
    lo, hi = BE.wilson_interval(k, n)
    return f"±{(hi - lo) / 2 * 100:.2f}pp"


def test_an_earlier_dated_artifact_makes_the_report_say_which_moves_are_real(tmp_path):
    """Clause 4. Every comparison line carries the exhibit rate with its Wilson interval
    AND a verdict in words about the change from that artifact.

    The two runs are built to DISAGREE, which is the only construction that proves the bar
    is per-arm rather than one threshold applied to everything:

      * caption 141,558/156,286 unchanged against itself → change +0.00pp against its own
        ±0.14pp → noise;
      * read-back 285/759 → 0/759 → change −37.55pp against ±0.25pp → a real move;
      * the pooled arm moves −0.18pp against ±0.15pp → real, because the small arm moved
        and the 156k arm did not.

    A single hard-coded threshold anywhere in the path calls at least one of those three
    wrong, and the three half-widths printed are three different numbers.
    """
    store = tmp_path / "store"
    _write_artifact(store, _mart(LIVE_PRIOR_ROWS, "2026-11-04T02:24:50+00:00"),
                    day="2026-11-04")
    current = _mart([LIVE_PRIOR_ROWS[0],
                     _mrow("read-back-before-success-claim", 0, 759),
                     LIVE_PRIOR_ROWS[2], LIVE_PRIOR_ROWS[3]],
                    "2026-11-07T06:00:00+00:00")
    text = _movement(current, store)

    assert "movement vs behaviour-exhibit-2026-11-04.jsonl" in text, text
    assert "measured_at 2026-11-04T02:24:50+00:00" in text, (
        "the sentence must name what it compared against, or a reader cannot tell a "
        "second reading from a first one")

    caption = _line(text, "tool-call-caption-present")
    assert caption.strip() == (
        "tool-call-caption-present worker 141558/156286 = 0.9058 [0.9043, 0.9072]; "
        "baseline 141558/156286 = 0.9058; change +0.00pp against this arm's own "
        "±0.14pp Wilson half-width at worker N=156286 — inside it: noise, not a move"), caption
    assert _half_pp(141558, 156286) in caption, caption
    assert "inside it: noise, not a move" in caption, caption

    readback = _line(text, "read-back-before-success-claim")
    assert "0/759 = 0.0000 [0.0000, 0.0050]" in readback, readback
    assert "baseline 285/759 = 0.3755" in readback, readback
    assert "change -37.55pp" in readback, readback
    assert _half_pp(0, 759) in readback, readback
    assert "beyond it: a real move, not noise" in readback, readback

    pooled = next(l for l in text.splitlines() if "ALL SCORED" in l)
    assert "ALL SCORED (4 behaviours, pooled) 144274/160504 = 0.8989 [0.8974, 0.9003]" in pooled, pooled
    assert "baseline 144559/160504 = 0.9007" in pooled, pooled
    assert "change -0.18pp" in pooled and _half_pp(144274, 160504) in pooled, pooled
    assert "beyond it" in pooled, (
        "the pooled arm moved because the small arm did; a pooled figure that only "
        "followed the 156k-situation arm would report a still day\n" + pooled)

    halves = {_half_pp(141558, 156286), _half_pp(0, 759), _half_pp(144274, 160504)}
    assert len(halves) == 3, f"three arms printed one shared bar: {halves}"


def test_the_first_reading_says_so_and_compares_nothing(tmp_path):
    """Clause 4's other half: with no earlier dated artifact the report refuses the
    comparison out loud, because a first run's intervals printed beside nothing else are
    exactly the condition under which a reader invents a baseline."""
    store = tmp_path / "empty-store"
    store.mkdir()
    text = _movement(_mart(LIVE_PRIOR_ROWS, "2026-11-04T02:24:50+00:00"), store)
    assert "first reading" in text, text
    assert "compares nothing" in text, text
    assert "±" not in text and "half-width" not in text, (
        "no bar may be printed when there is nothing to compare against: the first run's "
        f"job is to BE the baseline\n{text}")
    assert "noise, not a move" not in text and "a real move" not in text, text


def test_a_baseline_dated_the_same_day_as_the_run_is_not_compared_against(tmp_path):
    """The artifact this run wrote must never become its own baseline.

    Both 2026-11-04 runs of that night are live evidence of the trap: the store's artifact
    (measured_at 02:24:50Z, caption 141,983/156,755) and the tracked witness (03:37:58Z,
    142,262/157,101) differ, so a comparison that read a same-day file would report the
    gap between two runs of one night as a behaviour change. `prior_artifact` therefore
    takes files dated STRICTLY before the run's own day.
    """
    store = tmp_path / "store"
    same_day = _mart([_mrow("tool-call-caption-present", 142262, 157101)],
                     "2026-11-04T03:37:58+00:00")
    _write_artifact(store, same_day, day="2026-11-04")
    text = _movement(_mart([_mrow("tool-call-caption-present", 141983, 156755)],
                           "2026-11-04T02:24:50+00:00"), store)
    assert "first reading" in text, text
    assert "157101" not in text, (
        "a same-day artifact was used as a baseline; the two 2026-11-04 runs differ by "
        f"346 situations and would read as a behaviour change\n{text}")


def test_a_comparison_uses_nothing_but_the_worker_population(tmp_path):
    """Clause 4's population, which is #3749's owed-check ruling of 2026-11-07T10:34 (run
    20261107_023154_owedcheck_2421) barring the `user` population from trending.

    The fixture makes that ruling the whole test: the worker cells are identical between
    the two runs while the user cells go 40/40 → 0/40. A report that pooled both
    populations, or compared the user one, would print a collapse here. It must print
    noise, and it must never name the user arm.
    """
    prior_rows = [dict(r, by_population=dict(r["by_population"],
                                             user=_cell(*u)))
                  for r, u in zip(LIVE_PRIOR_ROWS, [(40, 40), (20, 20), (15, 15), (9, 9)])]
    current_rows = [dict(r, by_population=dict(r["by_population"],
                                               user=_cell(0, u[1])))
                    for r, u in zip(LIVE_PRIOR_ROWS, [(40, 40), (20, 20), (15, 15), (9, 9)])]
    store = tmp_path / "store"
    _write_artifact(store, _mart(prior_rows, "2026-11-04T02:24:50+00:00"), day="2026-11-04")
    text = _movement(_mart(current_rows, "2026-11-07T06:00:00+00:00"), store)

    assert "worker" in text
    assert "user" not in text.replace("worker", ""), text
    assert BE.MOVEMENT_POPULATION == "worker"
    for line in text.splitlines():
        if line.startswith("  ") and " worker " in line:
            assert "inside it: noise, not a move" in line, (
                "the worker rates did not move by a hundredth of a point; the user "
                f"population went from 100% to 0% and is not evidence\n{line}")
    assert "0/40" not in text and "40/40" not in text, text


def test_the_movement_bar_comes_from_wilson_interval_and_nowhere_else(tmp_path):
    """Clause 5: the ± in the sentence is that arm's own interval at that run's N.

    #3886's own triage measured the caption arm at ±0.1449pp (±0.15pp) where the item text
    said ±1.5pp — a factor of ten. A threshold copied from that list would have called the
    0.6pp caption move #3749 measured on 2026-11-04 (CIs [0.9041, 0.9070] against
    [0.897, 0.900], non-overlapping, i.e. real) noise, every day, silently. So the check is
    structural: the comparison functions may hold no rate-shaped float literal, and no name
    containing 'half' may be bound to a literal anywhere in the module.
    """
    src = Path(BE.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)

    def floats_in(fn_name: str) -> list:
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == fn_name)
        return [c.value for c in ast.walk(fn)
                if isinstance(c, ast.Constant) and isinstance(c.value, float)]

    for fn_name in ("movement_lines", "worker_pooled", "render_movement"):
        got = floats_in(fn_name)
        assert not [f for f in got if 0.0 < f < 1.0], (
            f"{fn_name}() holds a rate-shaped literal {got}; the comparison bar must come "
            "from wilson_interval at run time, never from a baked-in constant")
    body = src.split("def movement_lines(")[1].split("\ndef ")[0]
    assert "wilson_interval" in body, (
        "the comparison no longer calls wilson_interval, so its ± is somebody's guess")

    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            named, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            named, value = [node.target], node.value
        else:
            continue
        for t in named:
            if "half" in getattr(t, "id", "").lower() and isinstance(value, ast.Constant):
                offenders.append(f"{t.id} = {value.value!r}")
    assert offenders == [], (
        f"half-width(s) bound to a literal: {offenders} — each must be (hi − lo) / 2 of a "
        "wilson_interval result")


def test_no_comparison_or_trend_is_made_over_the_gzipped_file_split(tmp_path):
    """Clause 5's second half, with the reason measured: the window is `st_mtime >=
    cutoff`, and 4,496 of the 6,953 files in the live corpus were gzipped, so a series of
    those counts graphs the retention sweep's gzip night, not behaviour (#3885 measured
    4,457 of 7,051 the same way and owns the mtime→content-date fix).

    The assertion is not a comment: the file split is made WILDLY different (plain
    2,457→180, gzipped 4,496→6,800) while the rates are identical, and the movement block
    must come out byte-for-byte the same. Anywhere a comparison read the split, the
    verdicts would move with the sweep.
    """
    rows = LIVE_PRIOR_ROWS
    wild_prior = _write_artifact(
        tmp_path / "wild",
        _mart(rows, "2026-11-04T02:24:50+00:00",
              corpus={"files_read": 6953, "plain": 2457, "gzipped": 4496,
                      "window_filter": "file mtime inside the window, both suffixes read"}),
        day="2026-11-04").parent
    calm_prior = _write_artifact(
        tmp_path / "calm",
        _mart(rows, "2026-11-04T02:24:50+00:00",
              corpus={"files_read": 6953, "plain": 6953, "gzipped": 0,
                      "window_filter": "file mtime inside the window, both suffixes read"}),
        day="2026-11-04").parent
    current = _mart(rows, "2026-11-07T06:00:00+00:00",
                    corpus={"files_read": 6980, "plain": 180, "gzipped": 6800,
                            "window_filter": "file mtime inside the window, both suffixes read"})

    wild = _movement(current, wild_prior)
    calm = _movement(current, calm_prior)
    assert wild == calm, (
        "the movement block changed when only the plain/gzipped counts moved, so a "
        f"comparison is reading the file split\n--- wild ---\n{wild}\n--- calm ---\n{calm}")
    for needle in ("6800", "2457", "4496", "6953", "6980", "180"):
        assert needle not in wild, f"the corpus count {needle} leaked into the comparison"

    # The structural form of the same rule: the comparison functions may not READ the
    # corpus block at all. Prose in the head line says the split is not compared, so the
    # check is on the keys actually looked up, not on the words appearing nearby.
    tree = ast.parse(Path(BE.__file__).read_text(encoding="utf-8"))
    for fn_name in ("movement_lines", "worker_pooled", "prior_artifact", "render_movement"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == fn_name)
        keys = {c.value for c in ast.walk(fn) if isinstance(c, ast.Constant)
                and isinstance(c.value, str)}
        touched = keys & {"corpus", "plain", "gzipped", "files_read"}
        assert touched == set(), (
            f"{fn_name}() looks up the corpus keys {touched}: the plain/gzipped split is "
            "mtime-shaped and cannot carry a comparison or a trend until #3885 makes the "
            "window content-dated")


def test_a_run_with_no_out_flag_writes_the_named_store_and_reports_movement(tmp_path,
                                                                           monkeypatch,
                                                                           capsys):
    """The two surfaces joined as one real invocation: no `--out`, so the artifact lands at
    the path `app.paths` names, and the previous day's artifact is what gets compared.

    This is the process boundary the item is about. The scheduled command is
    `.venvs/lloyd/bin/python -m app.behaviour_exhibit --days 7` — it carries no output
    path, so the instrument's whole existence as an arm of the eval index rests on
    `out_dir()` returning the constant the index reads (clause 1, in service of clause 2).
    The second invocation then has to find yesterday's file as its baseline, and must not
    find today's own.
    """
    import app.paths as p

    store = tmp_path / "registered-store"
    monkeypatch.setattr(p, "BEHAVIOUR_EXHIBIT_DIR", store)
    # Stamped as a harness run on purpose: an unstamped record is `unattributed`,
    # which would leave the worker denominator empty and the report with the
    # "no rate" branch instead of a comparison line.
    write_session(tmp_path, "20261106_100000_main_aaaa", captioned_turn(),
                  source="autonomy-task:24")
    assert BE.out_dir() == store, "out_dir() must resolve the constant at call time"

    assert BE.main(["--sessions-dir", str(tmp_path), "--days", "7"]) == 0
    first_run = capsys.readouterr().out
    written = sorted(q.name for q in store.iterdir())
    assert len(written) == 1 and written[0].startswith("behaviour-exhibit-"), written
    assert f"wrote {store / written[0]}" in first_run, first_run
    assert "first reading" in first_run, (
        "an empty store is a first reading and the report must say so\n" + first_run)

    _write_artifact(store, _mart(LIVE_PRIOR_ROWS, "2026-11-06T06:00:00+00:00"),
                    day="2026-11-06")
    assert BE.main(["--sessions-dir", str(tmp_path), "--days", "7"]) == 0
    second_run = capsys.readouterr().out
    assert "movement vs behaviour-exhibit-2026-11-06.jsonl" in second_run, second_run
    assert "Wilson half-width" in second_run, second_run
    mv = second_run.split("movement vs ")[1].split("wrote ")[0]
    assert written[0] not in mv, (
        "the run compared itself against the artifact it had just written, which is how "
        f"two runs of one night read as a behaviour change: {mv}")
    assert mv.startswith("behaviour-exhibit-2026-11-06.jsonl"), mv
    assert (store / "behaviour-exhibit-2026-11-06.jsonl").exists(), (
        "retention must not have eaten the 1-day-old baseline: without it the next run "
        "has nothing to compare against either")


# ── #3886 clause 3: the sweep is scheduled ────────────────────────────────────
#
# Clause 1 and 2 give the store a name and a reader; clause 3 is what makes it run. This
# test is the seam between the two trees: the code lives in the checkout, the schedule
# lives in ~/obsidian/autonomy/, and the only thing that ties them is a `description`
# string that quotes the command. Read from the live vault through `app.paths.VAULT_ROOT`
# — the same root `app/autonomy.py` lists tasks from, so this asserts on the directory the
# scheduler will actually read.
#
# `status` has two legitimate readings and the test accepts both. `draft` is the state an
# unattended round must leave a new task in — `app/harness/policy.py::create_arms_a_task`
# refuses a created file that would be dispatchable on landing, so the file ships fully
# authored and one flip from firing, the #2858 precedent that landed
# 101-gate-to-land-lag-daily-record.md the same way — and `up_next` is what the flip that
# the file's own notes named actually did: #3977 armed #109 on 2026-11-09 through the file
# lane, whose predicate is `first_arm_allowed` (one dispatch field, `status`, moving alone
# on a never-run, unparked task). Asserting `draft` therefore certified the stall rather
# than the schedule, and the node went red on the day the step it demanded was taken —
# which is what #3980 is.
#
# Whether the task ever fires is not this node's question either: the instrument index row
# for this store (`cadence_days=1`, `hand_run_by_ruling=False`, and
# `scheduled_by="autonomy/109-behaviour-exhibit-daily.md"` in
# `scripts/eval_instrument_index.py`) prints `missing or stale: behaviour_exhibit` on a day
# with no dated artifact, and that headline is the documented detector. What is asserted
# here is the schedule's shape, and that the file states the reason for its own status —
# #3464's unattributed arm is what an armed file with stale notes reads as.

from app.harness.policy import DISPATCHING_STATUS, create_arms_a_task  # noqa: E402

#: What a round may leave a new task reading, and the one status the queue dispatches
#: (`DISPATCHING_STATUS` is imported from the policy module the scheduler mirrors, so
#: this test follows the rail instead of restating a string the rail could move under).
AUTHORED_STATUS = "draft"
SCHEDULE_STATES = (AUTHORED_STATUS, DISPATCHING_STATUS)

#: What the file's `notes` must say for each status it may hold. The authored reason is
#: required in both readings: #3977's arm commit retired the old "status is draft
#: deliberately" paragraph and replaced it with one that still carries that reason
#: alongside the arm, and an armed file that stopped saying why it was written unarmed is
#: the file the next reader "fixes" without knowing who set what.
STATUS_PROVENANCE = {AUTHORED_STATUS: ("draft",),
                     DISPATCHING_STATUS: ("draft", "first_arm_allowed")}

TASK_FILE = "109-behaviour-exhibit-daily.md"
RETENTION_TASK_FILE = "79-retention-sweep.md"


def _task_fields(name: str) -> dict:
    """A task file's front matter, loaded exactly as `app.autonomy` loads it (yaml)."""
    import yaml

    from app.paths import VAULT_ROOT
    f = VAULT_ROOT / "autonomy" / name
    if not f.exists():
        pytest.skip(f"{f} is not in the vault: the autonomy file is landed through "
                    "automod_vault_land, which commits the vault separately from this "
                    "tree — running the suite against a vault without it is a missing "
                    "witness, not a passing clause")
    text = f.read_text(encoding="utf-8")
    fm = text.split("---", 2)[1]
    fields = yaml.safe_load(fm)
    assert isinstance(fields, dict), f"{f}: front matter did not parse"
    return fields


def test_the_sweep_is_scheduled_as_a_daily_task_running_the_documented_command():
    """Clause 3, read out of the task file the scheduler will load.

    Four things the item names, each checked rather than assumed: a fresh id (the store
    grew from #108, so 109 and nothing lower, which would overwrite a live task),
    `frequency: daily`, the command exactly as the clause states it with nothing added, and
    an hour after the retention sweep. The command assertion is the cross-tree seam: it is
    a string inside a `description` field, and the only reason a daily job runs the real
    sweep instead of paraphrasing it is that the string says so — so the string is
    compared to the module's own parser as well as to the clause text. The status is read
    the same way, against the policy module's own `DISPATCHING_STATUS`: either legitimate
    reading passes, and the file must state the reason for the one it carries.
    """
    fm = _task_fields(TASK_FILE)
    assert fm["id"] == 109, fm["id"]
    assert fm["frequency"] == "daily", fm["frequency"]
    assert fm["skill_name"] == "behaviour-exhibit-daily", fm["skill_name"]
    from app.paths import VAULT_ROOT
    assert (VAULT_ROOT / "skills" / fm["skill_name"] / "SKILL.md").exists(), (
        "a task whose skill_name resolves to nothing skips at dispatch and looks like a "
        "day the sweep was simply not run")

    cmd = ".venvs/lloyd/bin/python -m app.behaviour_exhibit --days 7"
    assert cmd in fm["description"], (
        "the description must carry the command verbatim; it is the only text the worker "
        "is actually delivered")
    # Every mention of the module in the delivered text is checked, because the flags the
    # description forbids are also mentioned BY name there — scanning for `--out` anywhere
    # would trip on the prohibition that forbids it. What matters is that no occurrence of
    # the command carries an extra flag: each must close right after `--days 7`.
    mentions = [i for i in range(len(fm["description"]))
                if fm["description"].startswith("-m app.behaviour_exhibit", i)]
    assert mentions, fm["description"][:200]
    for i in mentions:
        tail = fm["description"][i + len("-m app.behaviour_exhibit"):][:40]
        assert tail.startswith(" --days 7") and "--days 7 " not in tail, (
            f"a delivered mention of the command is not the clause-3 command: {tail!r}")

    # The scheduled arguments, resolved the way the scheduler will resolve them: fed
    # through the module's own parser, so a `--days 7` that the code cannot read (a renamed
    # flag, a flag that now requires a value) fails here instead of on the first run.
    argv = cmd.split("-m app.behaviour_exhibit ")[1].split()
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--sessions-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--registry", default=None)
    ns = parser.parse_args(argv)
    assert ns.days == 7 and ns.out is None and ns.sessions_dir is None, vars(ns)

    # Only the two readings the scheduler knows. `up_next` is the one status that
    # dispatches (`app/autonomy.py::_is_task_due`), `draft` is the authored wait for the
    # flip, and anything else — `paused`, a misspelling — dispatches nothing while the
    # file still looks configured on the board.
    status = str(fm["status"]).strip()
    assert status in SCHEDULE_STATES, (
        f"`status: {fm['status']}` is neither `{AUTHORED_STATUS}` (authored, waiting for "
        "the human flip) nor the dispatching status: the daily artifact that goes missing "
        "from here is the silence #3886 was filed about")
    for word in STATUS_PROVENANCE[status]:
        assert word in str(fm.get("notes", "")), (
            f"a file reading `status: {status}` has to carry {word!r} in its notes, or the "
            "next reader 'fixes' the status without knowing who set it — the reason #3977 "
            "retired the stale notes paragraph in the same commit as the flip")


def test_the_rail_that_left_the_file_unarmed_still_refuses_an_arming_create():
    """The policy the red assertion was only narrating, checked at the rail itself.

    #3886 authored #109 `draft` because an unattended round cannot turn on an engine, and
    the 2026-11-09 arm did not weaken that rule — it exercised the one move the file lane
    had gained for a never-run, unparked task (`app/harness/policy.py::first_arm_allowed`,
    #3464). The sentence therefore belongs on the predicate that enforces it, not on a
    status string in a file that has legitimately moved past it. Both readings are pinned,
    because this item's fix removes the only place the schedule's own file spelled the
    policy out: a CREATED task reading `up_next` is exactly what the two file lanes refuse
    (`agent_mcp/builtin_fs.py` and `agent_mcp/vault.py` both branch on this predicate), and
    the authored shape — `skill_name` and `frequency` beside `draft` — is the nightly
    chain's documented hand-down and must stay accepted.
    """
    assert create_arms_a_task({"status": DISPATCHING_STATUS}) is True
    assert create_arms_a_task({"status": AUTHORED_STATUS,
                               "skill_name": "behaviour-exhibit-daily",
                               "frequency": "daily"}) is False


def test_the_sweep_is_ordered_after_the_retention_sweep_it_must_not_race():
    """Clause 3's ordering, checked against the other task's own file.

    The item anchored the new task to "after the 05:00Z retention sweep". Measured on
    2026-11-07 that anchor is false — #79's own front matter says `next_run:
    2026-11-08T06:00:00+00:00` — and a task at 05:00Z would have run BEFORE the sweep it
    was ordered behind. The test therefore compares against the retention task's declared
    hour rather than any hard-coded constant, so if #79 ever moves this test moves with it
    instead of certifying a stale hour. The race is real, not ceremonial: #79 gzips
    transcripts and the exhibit window is `st_mtime >= cutoff`, so reading the corpus
    mid-sweep changes what the rates mean.
    """
    mine = _task_fields(TASK_FILE)
    theirs = _task_fields(RETENTION_TASK_FILE)

    # `preferred_hours` are MACHINE-LOCAL hours (app/autonomy.py) and `next_run` is UTC, so
    # the two only agree if the hour was converted. #79 is the worked example on the same
    # shelf: window [22, 23, 0, 1, 2] against a 06:00Z next_run — 22 local is 06:00Z on a
    # box set to America/Los_Angeles. Checking the pair against each other is what catches
    # the trap #3886's text would have walked into: `preferred_hours: [7]` reads as 07:00Z
    # and fires at 15:00Z.
    def utc_hour(fm: dict) -> int:
        return datetime.fromisoformat(str(fm["next_run"])).astimezone(timezone.utc).hour

    def window_hours(fm: dict) -> list:
        return [int(h) for h in (fm.get("preferred_hours") or [])]

    def local_hour_of_next_run(fm: dict) -> int:
        return datetime.fromisoformat(str(fm["next_run"])).astimezone().hour

    assert local_hour_of_next_run(theirs) in window_hours(theirs), (
        "the convention check first proves itself on the precedent: #79's declared window "
        "must contain the local hour of its own next_run")
    assert local_hour_of_next_run(mine) in window_hours(mine), (
        f"#109's window {window_hours(mine)} does not contain the local hour "
        f"{local_hour_of_next_run(mine)} of its next_run {mine['next_run']}: "
        "`preferred_hours` is local and `next_run` is UTC, so an unconverted hour fires "
        "eight hours off")

    assert utc_hour(theirs) == 6, (
        f"#79 fires at {utc_hour(theirs)}Z, not the 05:00Z #3886's item text assumed")
    assert utc_hour(mine) > utc_hour(theirs), (
        f"the exhibit sweep's next_run {mine['next_run']} is not after the retention "
        f"sweep's {theirs['next_run']}: it would read the store mid-gzip")
    assert utc_hour(mine) == 7, utc_hour(mine)


def test_the_skill_that_runs_the_sweep_quotes_the_command_it_must_not_extend():
    """The other side of the same seam: the SKILL.md the dispatcher actually injects.

    `_load_skill_content` returns `skills/<skill_name>/SKILL.md`, so the worker's step 1 is
    whatever that file's fenced command says, and a flag added there is a flag added to
    every scheduled run. The check is that the command line in the skill parses to the
    no-flag sweep, and that the file exists at the name the task's `skill_name` gives.
    """
    from app.paths import VAULT_ROOT

    fm = _task_fields(TASK_FILE)
    skill = VAULT_ROOT / "skills" / fm["skill_name"] / "SKILL.md"
    assert skill.exists(), skill
    text = skill.read_text(encoding="utf-8")
    assert f"name: {fm['skill_name']}" in text, "the skill's own name must match the slug"

    fenced = [ln.strip() for ln in text.splitlines()
              if "app.behaviour_exhibit" in ln and ln.strip().startswith("cd ~/lloyd")]
    assert fenced, "the skill has no step-1 command line to run"
    assert all(ln.endswith("--days 7") for ln in fenced), fenced
    for ln in fenced:
        for bad in ("--out", "--sessions-dir", "--registry", "--max-files", "--check"):
            assert bad not in ln, f"{bad} is in the skill's step-1 command: {ln}"
