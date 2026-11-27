"""#1495: the log join that prices contention on qmd's own card (eval/qmd_contention.py).

The card number is not hardcoded in here. qmd's own supervisord program is what
decides it, so `test_the_docstring_names_the_card_the_daemon_conf_pins` reads that
conf and turns this suite red the day the module's prose and the pin disagree (#4474).

The rest of this file is the join's dating, which is what made its figure
non-reproducible: the runner's stamps are local and cross the DST boundary, the
daemon's are UTC and undated, and the daemon log is rotated (#4631). Every test here
that dates something dates it from a fixture it constructs, so none of it depends on
what today's log happens to contain.
"""
import configparser
import datetime as dt
import json
import re
from pathlib import Path

import pytest

from eval import qmd_contention
from eval.qmd_contention import (analyse, generation_span, in_window, merge,
                                parse_daemon, parse_pin_windows)
from scripts.automod.evalpin import (QMD_PROGRAM_CONF, QMD_PROGRAM_SECTION,
                                     _program_environment)

ROOT = Path(__file__).resolve().parent.parent
CARD = ROOT / "eval" / "measurements" / "qmd-contention-gpu4-2026-11-27.md"

# A card mention in prose: "GPU" then a space or a hyphen then a card index. Spelled
# as a character class and with no worked example, so this pattern's own bytes never
# match the files it scans — the suite stays free of the card labels it is checking for.
_CARD_MENTION = re.compile(r"GPU[- ](\d+)")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
# Wording that would place a service co-resident with qmd on qmd's own card.
_CO_RESIDENT = re.compile(r"(same|this|that)\s+card|shares? (?:the |qmd)?\s*card|on the card",
                          re.I)
# The card qmd's daemon left on 2026-11-10, when a372c876 moved the stack. Held as a
# bare digit so this file stays free of the label it forbids.
LEFT_CARD = "0"

DAEMON = """QMD MCP server listening on http://localhost:8181/mcp
23:59:58.000 POST /query 2 queries (50ms) [fts=2 embed=7 vec=30 chunk=2]
00:00:30.000 POST /query 2 queries (300ms) [fts=2 embed=160 vec=30 chunk=2]
00:00:31.000 POST /query 1 queries (5ms) [fts=2 vec=0 chunk=2]
00:05:00.000 POST /query 2 queries (900ms) [fts=2 embed=9 vec=40 chunk=2 rerank=800]
"""
REG = """2026-09-25 17:00:10,000 [INFO] lloyd-workers.automod-regression: measuring abcdef12 against 12345678 (1 pending)
2026-09-25 17:01:00,000 [INFO] lloyd-workers.automod-regression: abcdef12: success — no regression
"""


def pinned_card(conf: Path = QMD_PROGRAM_CONF):
    """CUDA_VISIBLE_DEVICES as qmd's daemon program sets it.

    ``None`` means there is no pin to agree with in this checkout -- the conf is
    absent or carries no ``[program:agent-qmd-daemon]`` -- which is a skip. A conf
    that has the section but no ``CUDA_VISIBLE_DEVICES`` comes back as ``""``, so it
    fails rather than quietly skipping: that daemon is on no card the prose can name.
    """
    if not conf.exists():
        return None
    cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=(" ;",),
                                   strict=False)
    cp.read(conf, encoding="utf-8")
    if not cp.has_section(QMD_PROGRAM_SECTION):
        return None
    env = _program_environment(cp.get(QMD_PROGRAM_SECTION, "environment", fallback=""))
    return env.get("CUDA_VISIBLE_DEVICES", "")


def _sentences(text: str) -> list[str]:
    return [s.replace("\n", " ").strip() for s in _SENTENCE_BREAK.split(text)]


def _row(elapsed_h: float, ms: int = 200, phases: str = "fts=6 embed=8 vec=14 chunk=2") -> str:
    """One `/query` line whose stamp is `elapsed_h` after midnight, mod 24 h — the shape
    of an undated log: the stamp only ever says where in the day the request landed."""
    t = dt.timedelta(hours=elapsed_h) % dt.timedelta(days=1)
    h, rem = divmod(int(t.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    return (f"{h:02d}:{m:02d}:{s:02d}.000 POST /query 2 queries ({ms}ms) [{phases}]")


def test_the_clock_wrap_is_the_next_day_and_phases_parse():
    rows = parse_daemon(DAEMON, dt.date(2026, 9, 25))
    assert [r["ts"].day for r in rows] == [25, 26, 26, 26]
    assert rows[1]["embed"] == 160 and "rerank" in rows[3]


def test_pin_windows_are_local_time_and_classify_requests():
    rows = parse_daemon(DAEMON, dt.date(2026, 9, 25))
    wins = parse_pin_windows(REG)
    assert len(wins) == 1 and wins[0][0].utcoffset() == dt.timedelta(hours=-7)
    out = analyse(rows, wins)
    assert out["recall_doc_leg__pin"]["total_ms"]["n"] == 1      # 00:00:30Z = 17:00:30 PDT
    assert out["recall_doc_leg__no_pin"]["total_ms"]["n"] == 1
    assert out["rerank__no_pin"]["rerank_ms"]["p50"] == 800


# --- the runner's stamps are dated by the zone that wrote them (#4631) ----------

STAMPS_ACROSS_DST = """2026-10-15 01:00:00,000 [INFO] lloyd-workers.automod-regression: measuring aaaaaaaa against 11111111 (1 pending)
2026-10-15 01:04:00,000 [INFO] lloyd-workers.automod-regression: aaaaaaaa: success — no regression
2026-11-15 01:00:00,000 [INFO] lloyd-workers.automod-regression: measuring bbbbbbbb against 22222222 (1 pending)
2026-11-15 01:09:00,000 [INFO] lloyd-workers.automod-regression: bbbbbbbb: success — no regression
"""


def test_a_pin_window_is_placed_by_its_own_stamp_date():
    """One regression log, two offsets: the runner's clock springs forward.

    The window this join prices spans the DST boundary in any run long enough to matter,
    and the whole file used to be read at one offset. A stamp of 2026-11-15 01:00 is
    09:00Z and one of 2026-10-15 01:00 is 08:00Z; read at a single offset every November
    window sat an hour early against the UTC daemon log, which moves which `/query` rows
    are 'inside' a pin — 1,026 lex-only rows against 564 on the real logs (#4631).
    """
    wins = parse_pin_windows(STAMPS_ACROSS_DST)
    assert len(wins) == 2
    oct_start, nov_start = wins[0][0], wins[1][0]
    assert (oct_start.year, oct_start.month, oct_start.day) == (2026, 10, 15)
    assert oct_start.astimezone(dt.timezone.utc).strftime("%H:%M:%S") == "08:00:00"
    assert (nov_start.year, nov_start.month, nov_start.day) == (2026, 11, 15)
    assert nov_start.astimezone(dt.timezone.utc).strftime("%H:%M:%S") == "09:00:00"
    assert nov_start.utcoffset() != oct_start.utcoffset(), \
        "one offset was applied to both stamps, which is #4631"


def test_no_fixed_utc_offset_survives_in_the_join():
    """The offset must come from a zone, so no per-stamp arithmetic can drift back in.

    #4631 filed this because `local_offset_h: int = -7` was a default nobody re-checked
    after 2026-11-01: a literal offset in the module is the defect, whatever it is set to.
    """
    src = Path(qmd_contention.__file__).read_text()
    assert "local_offset" not in src, "the join is taking an offset from its caller again"
    assert not re.search(r"hours\s*=\s*-\s*\d", src), (
        "a negative fixed offset is back in eval/qmd_contention.py; RUNNER_TZ is what "
        "places a stamp, per its own date")


def test_the_join_reads_stamps_in_the_machines_own_zone():
    """`RUNNER_TZ` is a name for the box's clock, so it has to stay that way.

    Same shape as the card rail above: the module hardcodes a fact about the machine, so
    the machine is asked. A box moved to another zone makes the runner's stamps mean
    something else, and this is the failing test that says so.
    """
    now = dt.datetime.now(dt.timezone.utc)
    assert now.astimezone(qmd_contention.RUNNER_TZ).utcoffset() == now.astimezone().utcoffset(), (
        f"eval/qmd_contention.py reads stamps as {qmd_contention.RUNNER_TZ} but this box's "
        "clock is something else")


# --- the daemon's own log: a quiet stretch is a day, not six hours (#4631) ------

#: Hourly traffic for a day, 30 h of nothing, then hourly traffic for another day. The
#: true span is 75 h; the stamps only ever move 51 h, because a 30 h silence shows up as
#: the 6 h the clock advanced.
QUIET_ROWS = [float(h) for h in range(0, 25)] + [float(h) for h in range(54, 76)]
QUIET_LOG = "\n".join(_row(h) for h in QUIET_ROWS) + "\n"


def test_a_quiet_stretch_longer_than_a_day_gains_its_day_back():
    """A 75 h log whose clock moves 51 h held a day of silence, not a faster day.

    `parse_daemon` advances on a backwards clock only, and a 30 h silence moves the stamp
    forward by 6 h, so the stretch is invisible by construction: the rows come out on
    three calendar days spanning 51 h when the file itself ran 75 h. Given the file's own
    duration the same text is four calendar days spanning 75 h (#4631). The wrap rule the
    older node pins is untouched — the hourly steps here still roll the day over, twice.
    """
    plain = parse_daemon(QUIET_LOG, dt.date(2026, 11, 10))
    assert sorted({r["ts"].date() for r in plain}) == [dt.date(2026, 11, d) for d in
                                                       (10, 11, 12)]
    assert (plain[-1]["ts"] - plain[0]["ts"]).total_seconds() == 51 * 3600

    dated = parse_daemon(QUIET_LOG, dt.date(2026, 11, 10), observed_span_s=75 * 3600)
    assert len(dated) == len(QUIET_ROWS)
    assert sorted({r["ts"].date() for r in dated}) == [dt.date(2026, 11, d) for d in
                                                       (10, 11, 12, 13)]
    assert dated[0]["ts"] == dt.datetime(2026, 11, 10, tzinfo=dt.timezone.utc)
    assert dated[-1]["ts"] == dt.datetime(2026, 11, 13, 3, tzinfo=dt.timezone.utc)
    # the day is given back where the silence is, not smeared over the busy hours
    stamps = [r["ts"] for r in dated]
    busy = [b - a for a, b in zip(stamps, stamps[1:]) if b - a < dt.timedelta(hours=2)]
    assert len(busy) == 45 and all(d == dt.timedelta(hours=1) for d in busy), busy


def test_an_observed_span_shorter_than_the_clock_moves_adds_nothing():
    """The correction is one-directional: a file that ran less than its stamps claim
    must not have a day taken off, or a quiet stretch would be 'fixed' into the past."""
    same = parse_daemon(QUIET_LOG, dt.date(2026, 11, 10), observed_span_s=40 * 3600)
    assert (same[-1]["ts"] - same[0]["ts"]).total_seconds() == 51 * 3600


# --- the rotations are readable, and each is dated by itself (#4631) ------------

def _fake_stats(table):
    """Replace `birth_and_mtime` with a table of (creation, last-write) epochs per name,
    which is the only way to exercise generation dating: a test filesystem under /tmp is
    tmpfs, and its creation times are all 'now'."""
    def fake(path: Path):
        return table[path.name]
    return fake


def test_covering_windows_agrees_with_testing_them_one_by_one():
    """`analyse` asks `in_window(merge(wins), ts)` for every row; 350k rows against a few
    hundred windows was nine minutes of `any(...)` per row, which is why the rotated join
    could not be re-run at all. The fast form must answer identically, overlapping
    windows and all: two windows that touch are one stretch of pin, and a request in the
    overlap is inside either way."""
    d0 = dt.date(2026, 11, 10)
    wins = [(dt.datetime.combine(d0, dt.time(), tzinfo=dt.timezone.utc) + dt.timedelta(
        minutes=m), dt.datetime.combine(d0, dt.time(), tzinfo=dt.timezone.utc)
        + dt.timedelta(minutes=m + 40)) for m in (0, 20, 100, 105, 300, 295)]
    covered = merge(list(wins))
    probes = [dt.datetime.combine(d0, dt.time(), tzinfo=dt.timezone.utc)
              + dt.timedelta(minutes=m) for m in
              (-1, 0, 10, 19, 20, 39, 40, 41, 60, 99, 100, 140, 141, 250, 294, 340, 341)]
    for ts in probes:
        assert in_window(covered, ts) == any(a <= ts <= b for a, b in wins), ts
    assert len(covered) == 3, covered          # 0-40+20-60, 100-145, 295-340


def test_a_generation_is_dated_in_utc_and_from_its_own_creation_time(monkeypatch):
    """The rotation lands whenever the log fills, and 00:34Z is 16:34 here the day before.

    One shared `--first-day` for a concatenated set meant at most one generation could be
    right; measured against their own mtimes, every generation on this box read 86,400 s
    off. The date is therefore taken from the file's creation instant *in UTC*, which is
    the only clock the daemon log is written in (#4631).
    """
    rotated = dt.datetime(2026, 11, 25, 0, 34, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(qmd_contention, "birth_and_mtime",
                        _fake_stats({"gen.err": (rotated.timestamp(),
                                                 rotated.timestamp() + 3600)}))
    g = generation_span(Path("/nowhere/gen.err"))
    assert g["first_day_utc"] == dt.date(2026, 11, 25)
    assert g["observed_span_s"] == 3600
    assert rotated.astimezone().date() == dt.date(2026, 11, 24), \
        "the fixture no longer shows a UTC date the local date disagrees with"
    assert g["first_day_utc"] != rotated.astimezone().date()


def test_a_generation_with_no_creation_time_is_refused_not_guessed(monkeypatch, tmp_path):
    """`stat`'s %W is 0 where the filesystem carries no creation time. Zero is the epoch,
    and 1970 would silently move a window by 56 years, so the caller has to say a date."""
    p = tmp_path / "flat.err"
    p.write_text(_row(1.0) + "\n")
    monkeypatch.setattr(qmd_contention, "birth_and_mtime",
                        _fake_stats({"flat.err": (None, 1795777925)}))
    with pytest.raises(SystemExit) as refused:
        qmd_contention.read_daemon_rows([str(p)])
    assert "creation time" in str(refused.value)
    rows = qmd_contention.read_daemon_rows([str(p)], first_day=dt.date(2026, 11, 27))
    assert rows[0]["ts"].date() == dt.date(2026, 11, 27)


def test_rotated_generations_join_without_a_shared_first_day(tmp_path, monkeypatch, capsys):
    """The CLI seam: two generations, one pattern, no `--first-day` anywhere.

    Each file is dated from its own creation instant, so a row from the older one keeps
    the older day and a row from the newer keeps the newer, and the pin window written in
    local time classifies them together. Both files are read through one glob and the
    rows come out in clock order, which is what makes the figure re-derivable from the
    rotated logs rather than from one person's `--first-day` judgement (#4631).
    """
    def epoch(y, m, d, hh=0, mm=0):
        return dt.datetime(y, m, d, hh, mm, tzinfo=dt.timezone.utc).timestamp()

    (tmp_path / "gen.err.2").write_text(_row(3 + 10 / 60) + "\n" + _row(5.0) + "\n")
    (tmp_path / "gen.err.1").write_text(_row(0 + 40 / 60) + "\n" + _row(1 + 20 / 60) + "\n")
    monkeypatch.setattr(qmd_contention, "birth_and_mtime", _fake_stats({
        "gen.err.2": (epoch(2026, 11, 20, 3), epoch(2026, 11, 20, 8)),
        "gen.err.1": (epoch(2026, 11, 24, 0, 34), epoch(2026, 11, 24, 2)),
    }))
    reg = tmp_path / "regression.log"
    # Local stamps: 2026-11-23 16:00 PST is 2026-11-24T00:00Z, so this window holds the
    # newer generation's two rows and none of the older one's.
    reg.write_text(
        "2026-11-23 16:00:00,000 [INFO] lloyd-workers.automod-regression: measuring "
        "abcdef12 against 12345678 (1 pending)\n"
        "2026-11-23 17:30:00,000 [INFO] lloyd-workers.automod-regression: abcdef12: "
        "success — no regression\n")
    out = tmp_path / "nested" / "join.json"

    rc = qmd_contention.main(["--daemon-log-glob", str(tmp_path / "gen.err*"),
                              "--regression-log", str(reg), "--json", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["query_rows"] == 4
    assert data["log_span_utc"][0].startswith("2026-11-20")
    assert data["log_span_utc"][1].startswith("2026-11-24")
    assert data["recall_doc_leg__pin"]["total_ms"]["n"] == 2
    assert data["recall_doc_leg__no_pin"]["total_ms"]["n"] == 2
    assert json.loads(capsys.readouterr().out)["query_rows"] == 4


# --- who else is on the card (#4631, over #4474's wording) ----------------------

def test_the_docstring_names_the_card_the_daemon_conf_pins():
    """#4474: the module's prose and the daemon's own pin cannot disagree.

    This is the rail a card move spends. Nothing here knows which card qmd runs on:
    the number comes from the conf, so moving the daemon off it is what turns this red.
    """
    card = pinned_card()
    if card is None:
        pytest.skip(f"{QMD_PROGRAM_CONF} is absent here, so nothing pins a card to agree with")
    assert card.isdigit(), (f"{QMD_PROGRAM_CONF.name} sets CUDA_VISIBLE_DEVICES={card!r}, "
                            "which is not a card index this docstring could name")
    named = set(_CARD_MENTION.findall(qmd_contention.__doc__ or ""))
    assert named == {card}, (
        f"eval/qmd_contention.py names card(s) {sorted(named)}, but "
        f"{QMD_PROGRAM_CONF.name} pins {card}")


def test_no_file_of_the_join_teaches_the_card_qmd_left():
    """#4474: the retired pairing must not survive in either file of the join.

    This file's own bytes are in scope: its docstring taught the retired pairing once.
    Runs even with the conf absent -- what is asserted is about the prose, not the card.
    """
    if pinned_card() == LEFT_CARD:
        pytest.skip("qmd is back on the card this test retires")
    for f in (Path(qmd_contention.__file__), Path(__file__)):
        named = set(_CARD_MENTION.findall(f.read_text()))
        assert LEFT_CARD not in named, (
            f"{f.name} still names card {LEFT_CARD}, which qmd's daemon left on 2026-11-10")


def test_the_docstring_names_the_watcher_as_a_co_resident_tenant():
    """#4631, over #4474's wording: the pin is not the card's only other tenant.

    The node before this one required a sentence matching 'only ... tenant', so the
    docstring stayed wrong while the suite stayed green: `agent-qmd-watcher` sets the
    same CUDA_VISIBLE_DEVICES as the daemon in its own supervisord conf, and its
    re-embeds are load on the card that no `/query` line shows. What must be pinned is
    that the watcher is named as co-resident, and that TTS is not placed here at all.
    """
    doc = qmd_contention.__doc__ or ""
    for s in _sentences(doc):
        if "tts" in s.lower():
            assert not _CO_RESIDENT.search(s), f"TTS is still placed on qmd's card: {s!r}"
    tenants = [s for s in _sentences(doc) if "watcher" in s.lower()]
    assert tenants, ("the docstring does not mention agent-qmd-watcher at all, and it is "
                     "one of the two things co-resident with production qmd")
    assert any(_CO_RESIDENT.search(s) for s in tenants), (
        f"no sentence places the watcher on qmd's card: {tenants}")
    assert any("pin" in s.lower() for s in tenants), (
        f"the tenant sentence names the watcher but not the eval pin beside it: {tenants}")


# --- the recorded figure (#4631) ------------------------------------------------

def test_the_gpu_4_figure_is_recorded_and_says_how_it_was_dated():
    """The card is the deliverable; this keeps it honest as a record.

    Every figure below is the corrected join over 2026-11-10T00:02Z→2026-11-27T10:42Z:
    352,281 rows, 339 windows, duty cycle 0.0481, the recall doc leg 198 ms p50 and mean
    213.2 ms outside. The needles chosen are the parts a later reader cannot re-derive —
    the rerank arm's sample size, how an undated rotated log got its dates, and that the
    machine-readable rows live in `$LLOYD_DATA/eval` and `/tmp` is tmpfs, so nothing
    survives there to be checked against.
    """
    assert CARD.exists(), f"{CARD} is where the GPU-4 join's figure belongs"
    card = CARD.read_text()
    for needle in ("2026-11-10T00:02", "2026-11-27T10:42", "418.6", "0.0481", "352,281",
                   "213.2", "339", "not priced", "tmpfs", "--daemon-log-glob",
                   "stat -c %W", "$LLOYD_DATA/eval"):
        assert needle in card, f"the card no longer states {needle!r}"
    assert "p95" in card, "the card must give p95 beside p50, or it hides the tail"
    assert re.search(r"22 (of them |rows )?(landed inside|inside a pin|against)", card) \
        or "22 rows" in card, "the rerank arm's sample size is unpinned"
    assert re.search(r"86,400", card), \
        "the card no longer states how far the old per-file dating was off"
