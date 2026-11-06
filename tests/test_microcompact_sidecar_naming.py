"""One rule for what a reduction sidecar is (#3815, acceptance clause 1).

`app/harness/microcompact.reduction_sidecar_path` writes
`sessions/<sid>.microcompact-reduced.json` into the SAME `*.json` glob as the
transcripts it records — the COUNT moves (~2 a
night here, so it is quoted with the tree it was read from): 107 sidecars against
11,877 real session heads at the item's triage (`656be6dc`), re-measured on 2026-11-06 at
`38eec6e5` as 111 against 11,933, so every `SESSIONS_DIR.glob("*.json")` denominator on the
box was reading 11,984. Excluding them needs a predicate, and a predicate written twice
is how the exclusion goes missing at the tenth site: before this change the harness had
`REDUCTION_SIDECAR_SUFFIX`, `app/behaviour_exhibit.py` had its own literal
`SIDECAR_MARKER = ".microcompact-reduced"`, and exactly ONE of the ten readers filtered
anything. These nodes pin that there is one rule, that the exhibit module reads the
writer's spelling rather than keeping a second one, and that the one restatement that
must exist — the stdlib-only groundskeeper script, which cannot import the harness —
has not drifted from it.
"""
import re
from pathlib import Path

from app.harness.microcompact import (REDUCTION_SIDECAR_SUFFIX,
                                      is_reduction_sidecar,
                                      reduction_sidecar_path)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: A chat-shaped session id and its sidecar, shaped exactly like the two chat-shaped
#: files on disk the day this was measured (oldest
#: `20261029_092125_iv1da3.microcompact-reduced.json`).
HEAD_ID = "20261029_092125_iv1da3"
HEAD_NAME = HEAD_ID + ".json"
SIDECAR_NAME = HEAD_ID + REDUCTION_SIDECAR_SUFFIX


def test_the_predicate_distinguishes_the_two_names_it_exists_for():
    """`is_reduction_sidecar` is True for the sidecar and False for the head (#3815
    clause 1) — the two names the whole item turns on.

    The negative cases are the ones a sloppy `endswith(".json")`-shaped rule would get
    wrong: the bare stem (no extension at all), the head, and a name that merely
    CONTAINS the marker rather than ending with it, which is what makes a `in name`
    test like the exhibit module's old one a different rule from this one.
    """
    assert is_reduction_sidecar(SIDECAR_NAME) is True
    assert is_reduction_sidecar(HEAD_NAME) is False
    assert is_reduction_sidecar(HEAD_ID) is False
    # A transcript whose own id happens to carry the marker substring is still a
    # transcript: the rule is the SUFFIX, not the substring.
    assert is_reduction_sidecar("20261029_092125_microcompact-reduced.json") is False
    assert is_reduction_sidecar("") is False


def test_the_predicate_is_the_writers_own_suffix_end_to_end():
    """Every path `reduction_sidecar_path` can emit is True under the predicate.

    The predicate and the writer read one constant, so this holds by construction — and
    it is the construction that is pinned: a change to the suffix that made the writer
    emit a name the predicate rejects would leave every #3815 exclusion filtering
    nothing while the writer kept producing the file.
    """
    emitted = reduction_sidecar_path(HEAD_ID)
    assert emitted is not None
    assert emitted.name == SIDECAR_NAME
    assert is_reduction_sidecar(emitted.name) is True
    # The head it sits beside is not it.
    assert is_reduction_sidecar(f"{HEAD_ID}.json") is False


def test_the_exhibit_marker_is_imported_from_the_writer_not_respelled():
    """`app.behaviour_exhibit.SIDECAR_MARKER` IS `microcompact.REDUCTION_SIDECAR_SUFFIX`
    (#3815 clause 1), and the module holds no second spelling of it.

    This is the clause's whole point: the exhibit module's private literal is what let
    it stay the only filtered site for as long as it took to accumulate
    the ~111 unfiltered files on this box (107 at the item's triage —
    a count read twice, each with the tree it came from, because it moves)
    files. The identity check catches a re-hardened value; the source scan catches a
    new literal anywhere else in the file, including one hidden in `is_transcript_path`
    itself.
    """
    import app.behaviour_exhibit as be

    assert be.SIDECAR_MARKER is REDUCTION_SIDECAR_SUFFIX
    assert be.SIDECAR_MARKER == ".microcompact-reduced.json"

    src = (REPO_ROOT / "app" / "behaviour_exhibit.py").read_text(encoding="utf-8")
    # Strip comments so the prose about the old literal is not read as a rule.
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    strays = [ln for ln in code.splitlines()
              if ".microcompact-reduced" in ln
              and "REDUCTION_SIDECAR_SUFFIX" not in ln]
    assert strays == [], (
        "app/behaviour_exhibit.py spells the sidecar suffix somewhere other than the "
        f"imported name: {strays}")


def test_is_transcript_path_and_the_predicate_never_disagree():
    """The exhibit filter agrees with the predicate on both real names, and the
    predicate is a SUFFIX rule where the exhibit's retired marker was a SUBSTRING rule —
    which is the drift this node exists to catch.
    """
    import app.behaviour_exhibit as be

    assert be.is_transcript_path(Path(be.SIDECAR_MARKER)) is False
    assert be.is_transcript_path(Path(SIDECAR_NAME)) is False
    assert be.is_transcript_path(Path(HEAD_NAME)) is True

    # The names that separate the two possible rules. The retired exhibit marker was the
    # SHORT substring `.microcompact-reduced`, tested with `in`; the predicate is the
    # writer's SUFFIX, tested with `endswith`. A name that ends in something else — an
    # archive the sweep writes, an editor's temp file — is a sidecar to the substring
    # rule and not to the suffix rule, so a name list of files both rules answer alike
    # (`HEAD_NAME`, `SIDECAR_NAME`, `notes.json`, `HEAD_ID`) cannot tell a re-hardened
    # short marker from the imported suffix. These can. The assertion is on the
    # predicate's own answer, not on a comparison with `is_transcript_path`, which now
    # delegates to the predicate: the two cannot disagree by construction, so a loop over
    # them restates the implementation instead of testing it.
    def short_marker_rule(name: str) -> bool:
        return ".microcompact-reduced" in name

    drift_names = [
        f"{HEAD_ID}.microcompact-reduced.json.gz",   # the sweep's co-archive name
        f"{HEAD_ID}.microcompact-reduced.json.tmp",  # an editor's or tool's temp write
    ]
    for name in drift_names:
        assert is_reduction_sidecar(name) is False, (
            f"{name} is not the name the reducer writes, so it is not a sidecar")
        assert be.is_transcript_path(Path(name)) is True, (
            f"the exhibit filter no longer follows the suffix rule on {name}")
    # The drift witness itself: on these two names the retired rule and the shipped rule
    # answer differently, which is what makes the asserts above a pin rather than a
    # restatement. If the predicate is ever widened back to a substring test, this is the
    # line that goes red.
    assert [short_marker_rule(n) != is_reduction_sidecar(n)
            for n in drift_names] == [True, True], (
        "the short-substring rule and the suffix rule now answer alike on the names that "
        "used to separate them — the predicate has been widened back to a substring test")


def test_the_groundskeeper_restatement_has_not_drifted():
    """`scripts/groundskeeper/retention-sweep.py` restates the suffix because it is
    stdlib-only and must not import the harness it is cleaning up — so the restatement
    is pinned to the writer's constant, the way
    `tests/test_session_platform_checks.py` pins that script's `NON_USER_PLATFORMS`.

    A drifted copy is worse than no copy: the sweep would keep reporting an archive
    count while archiving nothing, or refuse to archive real sessions.
    """
    src = (REPO_ROOT / "scripts" / "groundskeeper" / "retention-sweep.py").read_text(
        encoding="utf-8")
    m = re.search(r'^SIDECAR_SUFFIX\s*=\s*"([^"]*)"|^SIDECAR_SUFFIX\s*=\s*'
                  r"'([^']*)'", src, re.MULTILINE)
    assert m, ("retention-sweep.py no longer names SIDECAR_SUFFIX, or renamed it into a "
               "form this pin cannot read")
    restated = next(g for g in m.groups() if g is not None)
    assert restated == REDUCTION_SIDECAR_SUFFIX, (
        f"retention-sweep.py says {restated!r}, the writer says "
        f"{REDUCTION_SIDECAR_SUFFIX!r}")
