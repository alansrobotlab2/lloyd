"""The #4201 on-screen arms scorer: two counts, no model in the loop.

`scripts/youtube_onscreen_arms.py` exists to answer one question about the inert keyframe
leg (`e30ce708`) before that leg reaches the nightly digest: does reading text off
keyframes give the digest names it could not verify, and does it hand the digest names that
were only ever on the screen? Those are `missing_or_unverifiable` and `contamination`, and
item #4201 step 5 says the second one closes the arm at any nonzero value. So the numbers
these nodes pin are not decorative — a scorer that over-counts contamination kills a
research leg that works, and one that under-counts it ships the failure mode the leg
exists to avoid. Every node here is therefore written as a pair: the count it must produce,
and the shape it must refuse to count.

Import style is `sys.path.insert` of `scripts/` (the repo's own house pattern,
`tests/test_youtube_channel_monitor.py:26-32`, because `scripts/` is not a package), and the
sibling leg is reached as `workers.sources.youtube_digest` the way
`tests/test_youtube_digest_source.py:25` reaches it — that import is what lets these
nodes pin the scorer against the real names and the real sidecar row shape instead of
against a copy this file could get wrong.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for _p in (str(SCRIPTS), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import youtube_onscreen_arms as arms   # noqa: E402


#: The three inputs for one video, as small as the shapes can be while still holding every
#: case the two counts have. `Gemini 2.5 Pro` is in the transcript (folded, hyphen included)
#: and in no sidecar row, so it is neither missing nor contaminated. `Swarmskills` is in the
#: sidecar at 12.0s and in no caption, and the note says the speaker says it: contamination.
#: `Kernel-Bench` is screen-only as well but attributes itself to the slide at 0:41, the
#: timestamp that row really carries: not contamination. `Vidmancer` is in neither source:
#: missing. The fifth line is the prose the scorer must not read as claims — ordinary words,
#: two of them capitalised by sentence-start rather than by naming anything.
NOTE = (
    "Google shipped `Gemini 2.5 Pro` and the Antigravity agent this week.\n"
    "The speaker says `Swarmskills` is the repo behind it.\n"
    "`Kernel-Bench` appears on the slide at 0:41.\n"
    "`Vidmancer` is never in either source at all.\n"
    "It Uses a harness and Model weights shipped Friday.\n"
)
TRANSCRIPT = ("google shipped gemini 2.5 pro and the antigravity agent this week. "
              "the harness runs a benchmark suite.")
SIDECAR_ROWS = [
    '{"ts": 41.0, "frame": "000041.jpg", "frame_index": 1, '
    '"literals": [{"text": "Kernel-Bench", "kind": "title"}], "dropped": 0}',
    '{"ts": 12.0, "frame": "000012.jpg", "frame_index": 2, '
    '"literals": [{"text": "Swarmskills", "kind": "title"}], "dropped": 0}',
]


def _bundle(tmp_path: Path, *, rows: list[str] | None = None,
            transcript: str = TRANSCRIPT) -> Path:
    """A scratch bundle directory in the shape the monitor writes: `transcript.txt` plus a
    sidecar when `rows` is given. Never a real bundle — the real ones under
    `~/.local/share/ai-engineer/bundles` are inputs to a comparison, never an output of a
    test, and `test_the_scorer_changes_nothing_in_the_bundle_directory` is the node that
    keeps that true of the code rather than of this helper."""
    bundle = tmp_path / "Sc6TriRAGnI"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "transcript.txt").write_text(transcript, encoding="utf-8")
    if rows is not None:
        (bundle / "onscreen.jsonl").write_text("\n".join(rows), encoding="utf-8")
    return bundle


def _note(tmp_path: Path, text: str = NOTE) -> Path:
    note = tmp_path / "note.md"
    note.write_text(text, encoding="utf-8")
    return note


def _score(tmp_path: Path, note_text: str = NOTE, *, rows: list[str] | None = None,
           transcript: str = TRANSCRIPT) -> dict:
    bundle = _bundle(tmp_path, rows=SIDECAR_ROWS if rows is None else rows,
                     transcript=transcript)
    return arms.score_bundle(_note(tmp_path, note_text), bundle)


# --- clause 1: the scorer's shape, and that getting the shape costs no network ----------

def test_the_scorer_returns_both_counts_for_one_video(tmp_path):
    """`score_bundle` answers both counts for one video, and says how it got them.

    The two counts are the whole reason this file exists, so they are asserted by name —
    the item names them `missing_or_unverifiable` and `contamination`, and the artifact
    #4201 step 4 asks for has a column per count per video per arm, which only works if the
    keys are those words.

    `onscreen_rows` and `transcript_tokens` ride along because a count with no denominator
    cannot be read: `missing_or_unverifiable: 1` is reassuring over a 4-literal note and
    damning over a 40-literal one, and the paired comparison has to publish the size of the
    thing it counted, not just the count.
    """
    result = _score(tmp_path)
    assert result["missing_or_unverifiable"] == 1, result
    assert result["contamination"] == 1, result
    assert result["onscreen_rows"] == 2, result
    assert result["transcript_tokens"] > 0, result
    assert result["video_id"] == "Sc6TriRAGnI", result


def test_importing_the_scorer_opens_no_socket_and_calls_no_engine(monkeypatch):
    """Importing the module touches no network, no vision engine, no model.

    The paired run scores 20 notes in a loop, and the number is only comparable to itself if
    no part of the measurement asked a model anything — an LLM judging an LLM's provenance
    is how a leg that contaminates gets a clean report. Clause 1 asks for this property
    explicitly, so it is tested by forbidding the calls rather than by timing the import.

    Reloaded, not merely re-imported: the module is already in `sys.modules` by the time
    this node runs, so without `importlib.reload` it would pass against a cached module and
    say nothing about what an import actually does.

    The patched set is the ways out of a process that matter here — a raw socket, a helper
    that opens one, a DNS lookup that precedes either, and the two HTTP stacks the vision
    pass and the engine client use. `general_vocab()` is the one load-on-first-use import in
    the module (`scripts/youtube_onscreen_arms.py:107`), and it is why the assertion is about
    calls rather than about module identity: reaching the leg's vocabulary list pulls
    `workers.sources.youtube_digest` and everything it reaches, so a timing check or a
    `sys.modules` diff could be satisfied by a version of this file that stayed stdlib-only,
    while a forbidden call cannot be.
    """
    def forbidden(name):
        def raise_it(*args, **kwargs):
            raise AssertionError(f"{name} was called during an import: {args[:2]}")
        return raise_it

    for target in ("socket.socket", "socket.create_connection", "socket.getaddrinfo",
                   "urllib.request.urlopen", "http.client.HTTPConnection.request",
                   "http.client.HTTPSConnection.request"):
        monkeypatch.setattr(target, forbidden(target))
    # The reload has to happen with the leg NOT already in the cache, or the assertion below
    # passes on somebody else's leftover import. Hoisting `general_vocab()`'s leg import to
    # module level would open no socket and trip no tripwire; it would only make importing
    # this scorer cost the leg's whole chain, engine client included. That hoist is visible
    # here as an import-cache fact and nothing else.
    evicted = {name: module for name, module in list(sys.modules.items())
               if name.startswith("workers.") or name == "youtube_digest"}
    for name in evicted:
        del sys.modules[name]
    try:
        assert importlib.reload(arms) is arms
        assert arms.ONSCREEN_FILENAME == "onscreen.jsonl"
        assert "workers.sources.youtube_digest" not in sys.modules, (
            "importing the scorer pulled in the leg, so `general_vocab()` is no longer lazy")
        assert "code" in arms.general_vocab(), (
            "the leg's vocabulary is unreachable, so the fold silently lost its source")
        assert "workers.sources.youtube_digest" in sys.modules, (
            "asking for the vocabulary never loaded the leg, so the fold is not its source")
    finally:
        sys.modules.update(evicted)


# --- clause 2: missing_or_unverifiable, and the fold that must not over-count -----------

def test_a_note_with_three_literals_counts_exactly_one_missing_or_unverifiable(tmp_path):
    """Three asserted literals, one of them in neither source: exactly 1, not 2 and not 0.

    The clause is this sentence, and it is the pair that decides whether the treatment arm
    is any good: `Gemini 2.5 Pro` is in the transcript in lower case and split at the
    decimal, so the fold has to bridge case and punctuation to spare it; `Swarmskills` is in
    the sidecar only, so it is verified as seen-but-not-heard and must NOT land in the
    missing column either; `Vidmancer` is in nothing, and it is the one number the clause
    names.

    The note's last line is part of the count and not decoration — `It Uses`,
    `Model weights` and `Friday` are ordinary words that sentence-start capitalises, and a
    scorer that called them literals would report a clean arm as a bad one, which is the
    failure that would close a working leg. `test_ordinary_prose_is_never_a_literal` owns
    that half; here it is what makes the expected value 1 rather than 5.
    """
    result = _score(tmp_path)
    assert result["literals"] == 4, result
    assert result["missing_or_unverifiable"] == 1, result
    assert result["missing_literals"] == ["Vidmancer"], result


def test_a_literal_present_in_either_source_after_the_fold_is_never_counted(tmp_path):
    """`KERNELBENCH` on a frame and `Kernel-Bench` in the note are one name, not two.

    The fold's whole job, stated as the case that would break it. A vision model reading a
    slide caption in caps hands back one token where the note's hyphen hands back two, and
    auto-captions — the reason the leg exists at all — split a proper noun across a space, so
    a comparison that only matched exact spellings would report the leg's successes as
    missing literals and score the treatment arm worse than the control for doing its work.

    `Vidmancer` stays in the same note as the control on the control: the count must fall to
    exactly the one literal that really is in neither source, not to zero because the whole
    note stopped being readable.
    """
    rows = ['{"ts": 41.0, "frame": "000041.jpg", "frame_index": 1, '
            '"literals": [{"text": "KERNELBENCH", "kind": "title"}], "dropped": 0}']
    note = ("`Kernel-Bench` appears on the slide at 0:41.\n"
            "`Vidmancer` is never in either source at all.\n")
    result = _score(tmp_path, note, rows=rows)
    assert result["missing_or_unverifiable"] == 1, result
    assert result["missing_literals"] == ["Vidmancer"], result


def test_ordinary_prose_is_never_a_literal(tmp_path):
    """The false-positive guard: capitalised ordinary words are not claims about a video.

    Three shapes prose produces constantly, none of which a name-shape rule can tell from a
    name on shape alone: `It Uses` (two ordinary English words, the first capitalised by
    starting the sentence), `Model weights` (an ordinary word capitalised by starting a
    clause, the other lower case, so the run dies at one word) and `Friday` (a single
    capitalised word with nothing marked about it). Any of them in the missing column would
    turn a clean digest into a failing one, and #4201 step 5 reads its verdict per video — one
    prose false positive would close a leg that works. The reading comes from `GENERAL_VOCAB`,
    which knows `it`, `uses`, `model` and `weights` and does not know `ai` or `engineer`.

    `AI Engineer` is in this node as the case the rule must NOT swallow, and it is kept in both
    positions deliberately. Sentence position only breaks a tie among words the vocabulary list
    already knows; a word the list does not know is a name wherever it stands, because a note
    writing `AI Engineer is the channel.` really is asserting a name and this scorer's job is to
    go check it, not to decide it was prose. The loop is also the node that breaks if the
    tie-breaker were ever widened into a blanket sentence-start rule, and both spellings come
    back as one literal — the longest form only, which is what stops the acronym `AI` inside
    `AI Engineer` from being counted a second time.
    """
    assert arms.extract_literals(
        "It Uses a harness and Model weights shipped Friday.\n") == [], (
        arms.extract_literals("It Uses a harness and Model weights shipped Friday.\n"))
    for text in ("the AI Engineer episode", "AI Engineer is the channel."):
        assert arms.extract_literals(text) == ["AI Engineer"], (
            text, arms.extract_literals(text))


# --- clause 3: contamination, and the attribution that clears it ------------------------

def test_a_screen_only_literal_asserted_as_spoken_is_contamination(tmp_path):
    """A name only the screen showed, written as something the speaker said: 1.

    This is the number that closes the arm, so its evidence is named and not just counted:
    `Swarmskills` is in the sidecar at frame 12.0s, in no caption, and the note says "The
    speaker says `Swarmskills` is the repo behind it" — an assertion about a person's words
    that the video cannot support, which is the failure mode the leg is gated on.

    The companion in the same note is not counted: `Kernel-Bench` is screen-only too, but the
    sentence that carries it names the slide and gives 0:41, the timestamp the sidecar
    actually holds for it. One note, one count, one cleared and one booked — the rule
    separates them, and a rule that booked both would be reporting the format rather than
    the behaviour.
    """
    result = _score(tmp_path)
    assert result["contamination"] == 1, result
    assert [row["literal"] for row in result["contaminated_literals"]] == ["Swarmskills"], result
    assert result["contaminated_literals"][0]["frame_ts"] == [12.0], result


def test_a_screen_only_literal_with_a_frame_timestamp_the_sidecar_carries_is_not_counted(tmp_path):
    """The same literal, attributed: 0. This is the whole exemption, with nothing else changed.

    "seen on screen at 0:12, per the keyframe" is the provenance the prompt already teaches
    (`workers/sources/youtube_digest.py:740`: what comes off the sidecar "was seen on screen,
    not heard"), and 12.0s is a row the sidecar really has. A scorer that counted this would
    make the treatment arm fail for doing the one thing that makes it safe to run, and the
    comparison would then be measuring the scorer.

    Deliberately a tolerance case and not an exact one: 0:12 against a row at 12.0 is inside
    `TS_TOLERANCE_SECONDS` either way, and the frame a note cites is the nearest sample, not
    a byte offset.
    """
    note = NOTE.replace("The speaker says `Swarmskills` is the repo behind it.",
                        "Swarmskills is on screen at 0:12, per the keyframe.")
    result = _score(tmp_path, note)
    assert result["contamination"] == 0, result
    assert result["missing_or_unverifiable"] == 1, result


def test_a_screen_citation_to_a_frame_the_sidecar_does_not_carry_is_contamination(tmp_path):
    """A marker with a timestamp no frame has is a claim about the video, not a source: 1.

    The exemption has to be about the sidecar and not about the sentence, or writing the word
    "slide" in front of an invented name buys it a pass. 9:99 is 6059 seconds; the sidecar
    holds 41.0 and 12.0, so nothing here is within tolerance of anything, and the count stays
    at 1 while the previous node's identical sentence with a real timestamp is 0.
    """
    note = NOTE.replace("The speaker says `Swarmskills` is the repo behind it.",
                        "Swarmskills is on screen at 9:99, per the keyframe.")
    result = _score(tmp_path, note)
    assert result["contamination"] == 1, result
    assert [row["literal"] for row in result["contaminated_literals"]] == ["Swarmskills"], result


def test_a_literal_the_transcript_contains_is_never_contamination(tmp_path):
    """Heard and seen is not contamination: the video supports "the speaker said it".

    `Kernel-Bench` moves from sidecar-only to audible by adding two words to the transcript,
    and the note then asserts it as spoken with no marker at all. The count is 0. This is the
    exclusion that keeps the number about the leg rather than about the note's style: a
    literal the captions carry was in the video as sound, so attributing it to the speaker is
    not an invented claim, and booking it would contaminate every ordinary sentence about an
    ordinary named thing.
    """
    rows = ['{"ts": 41.0, "frame": "000041.jpg", "frame_index": 1, '
            '"literals": [{"text": "Kernel-Bench", "kind": "title"}], "dropped": 0}']
    result = _score(tmp_path,
                    "The speaker says `Kernel-Bench` twice in one breath.\n",
                    rows=rows, transcript=TRANSCRIPT + " kernel bench")
    assert result["contamination"] == 0, result


def test_a_sidecar_row_of_ordinary_vocabulary_is_not_a_name(tmp_path):
    """A sidecar holding the word `code` is a malformed row, and must not book the note.

    `filter_onscreen_literals` cannot have kept it: `code` is in `GENERAL_VOCAB`, so it would
    have been dropped before the row was written
    (`workers/sources/youtube_digest.py:249-262`). Its presence therefore means a sidecar
    from something other than that filter, and the safe reading of that is "I do not know what
    this row is", not "the note is contaminated" — the alternative would flag a digest for
    writing English, which is a false positive on the number that closes the leg.
    """
    rows = ['{"ts": 5.0, "frame": "000005.jpg", "frame_index": 1, '
            '"literals": [{"text": "code", "kind": "body"}], "dropped": 0}']
    result = _score(tmp_path, "It uses code everywhere in the harness.\n", rows=rows)
    assert result["contamination"] == 0, result


# --- clause 4: the CLI, and everything this round must not have touched -----------------

def test_the_cli_prints_both_counts_for_one_video(tmp_path, capsys):
    """`--note` plus `--bundle` prints the two counts for one video, in both forms.

    The two keys are printed as their own names rather than as prose because the owed
    10-bundle run pipes this into an artifact with a column per count, and a reader of that
    artifact has to be able to grep it. The sizes follow on a second line for the same
    reason as clause 1: a bare count with no denominator cannot be doubted.

    The control arm is part of this node and not an afterthought: with no sidecar in the
    bundle the CLI still exits 0 and reports `contamination: 0`, because on the control arm
    nothing was only on the screen. A CLI that errored there could not run the comparison it
    exists for.
    """
    bundle = _bundle(tmp_path, rows=SIDECAR_ROWS)
    assert arms.main(["--note", str(_note(tmp_path)), "--bundle", str(bundle)]) == 0
    out = capsys.readouterr().out
    assert "missing_or_unverifiable: 1" in out, out
    assert "contamination: 1" in out, out
    assert "note literals: 4" in out, out
    assert out.count("(") == out.count(")"), out
    capsys.readouterr()
    assert arms.main(["--note", str(_note(tmp_path)), "--bundle", str(bundle),
                      "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["missing_or_unverifiable"] == 1, payload
    assert payload["contamination"] == 1, payload

    run = subprocess.run([sys.executable, str(SCRIPTS / "youtube_onscreen_arms.py"),
                          "--note", str(_note(tmp_path)), "--bundle", str(bundle)],
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    assert "missing_or_unverifiable: 1" in run.stdout, run.stdout


def test_the_cli_refuses_a_video_whose_sources_are_not_there(tmp_path, capsys):
    """A missing note or transcript exits 2 and says which file, instead of scoring nothing.

    Absent inputs are the one failure this tool can get silently wrong: with an empty
    transcript every literal is unverifiable, so a run that could not find `transcript.txt`
    would print a confident worst-case score and the artifact would carry it as a result.
    Exit 2 with the path named is the difference between a failed measurement and a bad one.

    A missing `onscreen.jsonl` is deliberately NOT in this node — that is the control arm,
    and the node above pins that it scores.
    """
    bundle = tmp_path / "nope"
    bundle.mkdir()
    assert arms.main(["--note", str(_note(tmp_path)), "--bundle", str(bundle)]) == 2
    err = capsys.readouterr().err
    assert "transcript.txt" in err, err
    assert str(bundle / "onscreen.jsonl") not in err, err


def test_the_scorer_changes_nothing_in_the_bundle_directory(tmp_path):
    """Scoring leaves the bundle exactly as it found it — same names, same mtimes.

    The owed run scores the treatment arm against a sidecar copied into scratch, but this
    tool is pointed at `~/.local/share/ai-engineer/bundles` by the CLI's own help text, and
    696 real bundles are what the next measurement needs intact. A side-effecting scorer
    (dropping a normalised sidecar, writing a report next to the transcript) would quietly
    change what the treatment arm sees mid-comparison, which is a paired run whose two arms
    are no longer paired.

    mtimes and not just names, because a rewrite with identical bytes is exactly what an
    idempotent-looking writer does.
    """
    bundle = _bundle(tmp_path, rows=SIDECAR_ROWS)
    before = {(p.name, p.stat().st_mtime_ns, p.stat().st_size)
              for p in sorted(bundle.iterdir())}
    assert before and arms.main(["--note", str(_note(tmp_path)), "--bundle", str(bundle)]) == 0
    after = {(p.name, p.stat().st_mtime_ns, p.stat().st_size)
             for p in sorted(bundle.iterdir())}
    assert after == before, (before, after)


def test_the_scorer_leaves_the_prompt_builder_and_the_live_leg_alone():
    """`build_prompt` gains no seam, and the live digest and monitor files are byte-identical.

    The arms are two directories: `onscreen_field()` already chooses between them from
    `sidecar.is_file()` (`workers/sources/youtube_digest.py:833-847`), so an `onscreen=`
    kwarg would be a second writer of a decision the filesystem makes — and it would put an
    offline experiment's switch inside the prompt the nightly job builds for every video. The
    item forbids that, and this node is what forbids it to a later round that finds the kwarg
    convenient: the assertion is against `HEAD`, so the file is compared as committed rather
    than as this worktree might have left it.

    Read through `git show` rather than the working tree because the working tree is what a
    bad change would have edited; `HEAD` is the thing this round did not touch.
    """
    for relative in ("scripts/youtube_channel_monitor.py",
                     "workers/sources/youtube_digest.py"):
        committed = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{relative}"],
                                   capture_output=True, text=True, check=True).stdout
        assert committed == (ROOT / relative).read_text(encoding="utf-8"), relative


def test_the_two_arms_are_two_directories_and_not_a_new_seam(tmp_path):
    """The control arm is a directory without a sidecar, and the live code already says so.

    Same bundle copied twice, one copy given the sidecar: `onscreen_field()` answers `none`
    for the first and the sidecar's path for the second, with nothing in this diff involved.
    That is the whole reason the scorer needs no seam to be run — the arm is an input, chosen
    by which files are where, and the control arm is what all 696 bundles already get on the
    nightly path.

    Pinned against `workers.sources.youtube_digest` rather than against a local copy of the
    rule: if that function ever stops reading the filesystem, the paired comparison stops
    being a comparison of two directories, and this node is where that would surface.
    """
    from workers.sources import youtube_digest

    control = _bundle(tmp_path, rows=None)
    treatment = _bundle(tmp_path / "t", rows=SIDECAR_ROWS)
    assert youtube_digest.onscreen_field({"bundle_dir": str(control)}) == "none"
    field = youtube_digest.onscreen_field({"bundle_dir": str(treatment)})
    assert field == str(treatment / "onscreen.jsonl"), field


def test_the_scorer_spells_the_sidecar_name_as_the_leg_does():
    """One spelling of `onscreen.jsonl`, checked against the leg that writes it.

    The scorer opens the file another module creates, and the two are in different trees of
    this repo (`scripts/` and `workers/sources/`) with no import between them —
    `general_vocab()` is loaded lazily so that an import of this file stays cheap, and the
    filename constant is the one fact from that module worth taking eagerly by assertion
    instead of by copy. Renamed on either side, the treatment arm would read as a control:
    no sidecar found, `contamination: 0`, and the comparison would report a clean leg.
    """
    from workers.sources import youtube_digest

    assert arms.ONSCREEN_FILENAME == youtube_digest.ONSCREEN_FILENAME, arms.ONSCREEN_FILENAME
    assert arms.TRANSCRIPT_FILENAME == "transcript.txt", arms.TRANSCRIPT_FILENAME


def test_a_sidecar_the_leg_wrote_is_scored_as_the_leg_wrote_it(tmp_path):
    """The scorer is checked against the leg's own filter output, not against a copy of its rules.

    Two modules decide what a literal is: the leg on the way in
    (`workers/sources/youtube_digest.py:268` — drop UI chrome, then subtract general
    vocabulary) and this scorer on the way out. If they disagree about one string, the paired
    comparison measures the disagreement instead of the leg, and the disagreement is invisible
    because both sides look self-consistent. So the fixture is `filter_onscreen_literals`'s own
    return value over three strings, one of which it drops: `the model` never reaches the
    sidecar, so a note mentioning a model cannot be contaminated by it, while `Kernel-Bench`,
    which the leg did keep and the transcript does not carry, is contamination once the note
    puts it in the speaker's mouth. `Gemini 2.5 Pro` is kept as well and is audible, so it is
    neither missing nor contaminated — the third arm of the same row.

    This is also the only node that would notice the row shape changing under `read_onscreen`:
    that function tolerates a malformed row instead of crashing, which is what a run over 696
    bundles needs and a poor witness, so the count is asserted together with `onscreen_rows`,
    and a row that stopped parsing would move both.
    """
    from workers.sources import youtube_digest

    kept = youtube_digest.filter_onscreen_literals(
        ["Kernel-Bench", "the model", "Gemini 2.5 Pro"])
    assert kept["dropped"] == 1, kept
    row = {"ts": 41.0, "frame": "000041.jpg", "frame_index": 1,
           "literals": kept["literals"], "dropped": kept["dropped"]}
    result = _score(tmp_path,
                    "The speaker says `Kernel-Bench` twice and `Gemini 2.5 Pro` once.\n",
                    rows=[json.dumps(row)])
    assert result["contamination"] == 1, result
    assert [r["literal"] for r in result["contaminated_literals"]] == ["Kernel-Bench"], result
    assert result["onscreen_rows"] == 1, result


def test_a_sidecar_the_leg_actually_wrote_is_what_the_scorer_reads(tmp_path):
    """The writer and the reader on one bundle directory: `run_onscreen_pass` then `score`.

    The node above pins the scorer against the leg's FILTER. This one pins it against the
    leg's FILE, which is the boundary that matters: `run_onscreen_pass` is what creates
    `onscreen.jsonl` on a real bundle
    (`workers/sources/youtube_digest.py:484-569`), and if its row shape and `read_onscreen`'s
    reader ever part ways the treatment arm degrades into the control silently — a sidecar
    that parses to zero rows scores `contamination: 0` on every video, which is the passing
    result for the wrong answer and exactly what a comparison of two arms cannot see.

    Frames are fake bytes and the vision model is a lambda answering with the OCR envelope
    the leg expects, one row per frame; the file it writes is not fake. The asserted literal
    is screen-only and the note puts it in the speaker's mouth, so the count must come out 1
    through the real writer, and the row the leg wrote must carry the timestamp the exemption
    rule needs.
    """
    from workers.sources import youtube_digest

    bundle = _bundle(tmp_path, rows=None)
    (bundle / "frames").mkdir()
    manifest = []
    for name, ts in (("000012.jpg", 12.0), ("000041.jpg", 41.0)):
        (bundle / "frames" / name).write_bytes(b"\xff\xd8\xff\xe0fakejpegbytes")
        manifest.append({"path": f"frames/{name}", "ts": ts})
    (bundle / "frames.jsonl").write_text(
        "\n".join(json.dumps(row) for row in manifest) + "\n", encoding="utf-8")

    def fake_vision(prompt, urls):
        # The envelope `parse_ocr_answer` reads: `{"frames": [{"index", "literals"}]}`
        # (`workers/sources/youtube_digest.py:372-404`). Shaped by the parser rather than
        # guessed at: a reply in a shape it cannot use would have this node assert a sidecar
        # full of nothing and read the leg as broken when only the fake was wrong.
        return json.dumps({"frames": [
            {"index": 0, "literals": [{"text": "Swarmskills", "kind": "title"}]},
            {"index": 1, "literals": [{"text": "Kernel-Bench", "kind": "title"}]}]}), None

    meta = {"video_id": "FAKEVIDEO", "bundle_dir": str(bundle)}
    result, error = youtube_digest.run_onscreen_pass(meta, log=lambda *_a: None,
                                                     vision=fake_vision)
    assert error is None, error
    assert result["path"] == str(bundle / "onscreen.jsonl"), result
    assert (bundle / "onscreen.jsonl").is_file(), "the leg wrote no sidecar to score"

    scored = arms.score_bundle(_note(tmp_path), bundle)
    assert scored["onscreen_rows"] == 2, scored
    assert scored["contamination"] == 1, scored
    assert [row["literal"] for row in scored["contaminated_literals"]] == ["Swarmskills"], scored
    assert scored["contaminated_literals"][0]["frame_ts"] == [12.0], scored
