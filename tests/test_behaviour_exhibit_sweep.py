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

import gzip
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.behaviour_exhibit as BE  # noqa: E402

NOW = datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)


def _tc(name, args):
    return {"id": f"c-{name}-{abs(hash(json.dumps(args, sort_keys=True)))}",
            "function": {"name": name, "arguments": json.dumps(args, sort_keys=True)}}


def _msg(text="", calls=()):
    return {"role": "assistant", "content": text, "tool_calls": list(calls)}


def write_session(dir_, sid, messages, *, gzipped=False, source=None, platform=None,
                  age_days=0):
    """One transcript on disk, gzipped or not, stamped inside the window unless aged.

    `platform` is written only when given: whether a record carries the field at all is one of
    the signals `population_of` reads, so a helper that always wrote `platform: null` would
    make "unstampable" and "stamped null" the same fixture and the residual population
    untestable.
    """
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
    stamp = (NOW - timedelta(days=age_days)).timestamp()
    import os
    os.utime(p, (stamp, stamp))
    return p


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
