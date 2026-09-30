"""#1878 — the Qwen3-TTS server-side frame cap, pinned against the tracked patch.

The defect: `api/backends/optimized_backend.py` calls the two vendored streaming
methods — `stream_generate_voice_clone` and `stream_generate_custom_voice` — without
`max_frames`, so every request inherits the vendored default `max_frames=10000`, an
~800 s audio budget at the model's 12.5 Hz frame rate. `~/lloyd-data/logs/services/
agent-tts.err` shows that budget consumed exactly twice: 2026-09-23 14:19:34
(`total=419.94s audio=800.00s chunks=1667`) and 2026-09-28 19:51:04 (`total=378.74s
audio=800.00s chunks=1667`). 1667 chunks is `ceil(10000 / 6)` at the service's
`emit_every_frames: 6`, which is the signature of the cap, not of the decoder
finishing. Healthy requests on the same log top out at `audio=19.68s chunks=41`.

Why the test reads the patch and not the module: `agent-services/services/tts/
qwen3-tts/` is an untracked clone (`SETUP.md:838-841` documents clone → checkout
783bf0e → `git apply ../qwen3-tts-local.patch`), so an in-tree edit yields no diff
and no gate artefact. `qwen3-tts-local.patch` is the only tracked thing this change
can touch, and until a human re-applies it (HUMAN_CLAUSE 1) the running service
still has the old behaviour. So the artefact under test is the patch text, and what
this file does is *execute the code the patch adds*: it lifts the added lines out of
the diff, builds the helper functions and the three backend methods they define into
a live namespace, and calls them. Nothing here asserts on a string the test itself
built — the numbers come from code extracted from the file under review, so breaking
the derivation in the patch turns these nodes red.

The four clauses and the nodes that pin them:

  1. every streaming call site passes `max_frames`
     → `test_the_patch_adds_a_max_frames_argument_to_every_streaming_call`
  2. a 240-character utterance gets 60-120 s (750-1500 frames) and no input gets
     more than 2000 frames
     → `test_a_240_character_utterance_gets_between_sixty_and_120_seconds`
     → `test_no_input_can_buy_the_vendored_eight_hundred_second_budget`
  3. the numbers hold with no cap keys in `optimization.streaming`, and the keys
     override the defaults when present
     → `test_the_defaults_hold_on_a_tree_that_never_reapplied_the_config_hunk`
     → `test_the_config_keys_override_the_in_code_defaults`
  4. ending at the cap logs a WARNING naming the cap and the frames emitted
     → `test_a_cap_hit_warns_naming_the_cap_and_the_frames_emitted`
     → `test_a_run_that_finishes_at_eos_stays_silent`
     → `test_the_witness_counts_the_loop_it_is_wired_into_and_fires_after_it`

The warmup calls are capped too, but by a rule with no ceiling: a warmup cut short leaves
torch.compile work undone, which is the opposite of what warmup is for, and no text a
caller does not control reaches one — so `test_no_warmup_call_gets_a_budget_its_own_text_
could_exhaust` checks the budget against the warmup's own text rather than against a
number. And because the two warmup methods never receive the `optimization.streaming`
block, that budget call takes no config argument at all:
`test_the_warmup_budget_needs_no_argument_the_warmup_never_binds` sweeps the patched file
for a name no enclosing scope binds, which is what passing one there would have been — a
`NameError` swallowed by the warmup's own `except`, i.e. a model that boots uncompiled.
`test_the_regenerated_patch_applies_to_the_pristine_clone_and_caps_every_call` applies the
whole patch to a reconstructed pristine clone and AST-walks the result, so the hunk is
graded as a patch and not only as text.

Two things this file deliberately does not claim. It does not measure the live
service — between landing and HUMAN_CLAUSE 1 the running server is still uncapped,
and the seven-day `agent-tts.err` witness is owed after that step. And it does not
rule on lowering the vendored `max_frames: int = 10000` default itself
(`qwen_tts/inference/qwen3_tts_model.py:700`, `:1024`), which is an upstream-facing
edit to an untracked clone (#1878 owed entry 3).
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from app.paths import VAULT_ROOT

REPO = Path(__file__).resolve().parents[1]
PATCH = REPO / "agent-services" / "services" / "tts" / "qwen3-tts-local.patch"
BACKEND = "api/backends/optimized_backend.py"
CONFIG = "config.yaml"

#: The vendored budget, and the arithmetic that turns frames into seconds. Both are
#: measurements, not preferences: 10000 is the literal at
#: `qwen_tts/inference/qwen3_tts_model.py:700`, and 12.5 Hz falls out of the log —
#: the largest healthy line is `audio=19.68s chunks=41`, i.e. 41 x 6 = 246 frames for
#: 19.68 s of audio, and the runaway is 10000 frames for exactly 800.00 s.
VENDORED_FRAMES = 10_000
FRAMES_PER_SECOND = 12.5
OBSERVED_HEALTHY_FRAMES = 246          # audio=19.68s, chunks=41
RUNAWAY_CHUNKS = 1667                  # ceil(10000 / 6) at emit_every_frames: 6

#: Clause 2's two bounds, stated in the unit the clause uses.
CEILING_FRAMES = 2_000
ALERT_CHARS = 240                      # the length of a Guardian alert


# ── reading the patch ─────────────────────────────────────────────────────

def _sections(patch_text: str) -> "dict[str, list[str]]":
    """Split a git patch into per-file line lists, keyed by the path `git apply -p1`
    will land it at.

    The key comes from the `+++` line, not the `diff --git` header: this patch is
    generated inside the untracked clone by the re-sync rule at SETUP.md:879-883, so
    its headers carry the clone's own `i/` and `w/` prefixes rather than the usual
    `a/`/`b/`, and a parser that assumed `b/` would find no files at all. Stripping
    one leading path component is what `-p1` does, so the key here and the file the
    patch touches agree by construction.
    """
    out: "dict[str, list[str]]" = {}
    current = None
    for line in patch_text.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            current = path.split("/", 1)[1] if "/" in path else path
            out.setdefault(current, [])
        elif line.startswith("diff --git "):
            current = None                      # wait for this file's +++ line
        elif current is not None:
            out[current] = out[current] + [line]
    return out


def _added(section: "list[str]") -> "list[str]":
    """The lines the patch adds, with the leading `+` removed, in order."""
    return [ln[1:] for ln in section if ln.startswith("+") and not ln.startswith("+++")]


def _hunk_postimages(section: "list[str]") -> "list[list[str]]":
    """Each hunk's post-image: its context lines plus its added lines, in source order.

    This is what a def has to be extracted from. The added lines alone are not a
    piece of source — they are every hunk of the file concatenated, so a body read
    out of them runs straight on into a later hunk's `max_frames=` argument. A
    hunk's post-image *is* contiguous file content, which makes a dedent the end of
    a body again.
    """
    hunks: "list[list[str]]" = []
    current = None
    for ln in section:
        if ln.startswith("@@"):
            current = []
            hunks.append(current)
        elif current is not None and (ln.startswith("+") or ln.startswith(" ")):
            current.append(ln[1:])
    return hunks


@pytest.fixture(scope="module")
def patch_sections() -> "dict[str, list[str]]":
    assert PATCH.exists(), f"the tracked patch vanished: {PATCH}"
    return _sections(PATCH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def backend_added(patch_sections) -> "list[str]":
    assert BACKEND in patch_sections, (
        f"{BACKEND} is not in the tracked patch, so nothing caps the streaming "
        f"calls: {sorted(patch_sections)}")
    return _added(patch_sections[BACKEND])


def _block(hunks: "list[list[str]]", start_pattern: str) -> "list[str]":
    """One `def`'s source, taken from whichever hunk post-image declares it.

    The body runs to the first non-blank line indented no deeper than the `def`
    itself — the rule Python's own grammar uses — except while brackets from the
    block are still open, because a multi-line signature closes its parenthesis in
    column zero and that is not the end of the definition. Matched across every hunk
    and asserted unique, so a rename that left two definitions of the same helper is a
    failure here rather than whichever one a search happened to find first.
    """
    found = []
    for hunk in hunks:
        for i, ln in enumerate(hunk):
            if not re.match(start_pattern, ln):
                continue
            def_indent = len(ln) - len(ln.lstrip())
            body = [ln]
            # Seed the bracket depth with the def line itself: a signature that
            # wraps onto its own lines leaves the paren open on line 1, and a
            # counter that starts at zero breaks out at the `) -> int:` line.
            depth = (ln.count("(") + ln.count("[") + ln.count("{")
                     - ln.count(")") - ln.count("]") - ln.count("}"))
            for following in hunk[i + 1:]:
                # A dedent ends the block only with no bracket still open from the
                # lines before it — so the open-count is tested as of the *previous*
                # line, before this line's own parentheses are folded in. Testing it
                # after would end a wrapped signature at its own `) -> int:` line.
                if following.strip() == "":
                    body.append(following)
                    continue
                if depth <= 0 and len(following) - len(following.lstrip()) <= def_indent:
                    break
                depth += (following.count("(") + following.count("[")
                          + following.count("{") - following.count(")")
                          - following.count("]") - following.count("}"))
                body.append(following)
            while body and body[-1].strip() == "":
                body.pop()
            found.append(body)
    assert len(found) == 1, (
        f"pattern {start_pattern!r} matched {len(found)} defs across the patch's "
        "hunks, expected exactly 1 — the code this test executes is not uniquely "
        "where the patch says it is")
    assert len(found[0]) > 1, f"{start_pattern}: a def with no body"
    return found[0]


def _source(lines: "list[str]") -> str:
    """Join extracted lines back into source, with the newline the split removed."""
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def backend_hunks(patch_sections) -> "list[list[str]]":
    return _hunk_postimages(patch_sections[BACKEND])


@pytest.fixture(scope="module")
def tts_cap(backend_hunks):
    """The patch's added code, executed.

    Module-level constants and the two free functions are exec'd as they are, and
    the three `OptimizedQwen3TTSBackend` methods are lifted into a shim class —
    they touch no `self` state but `logger` and the constants, so the shim needs no
    model. A recording logger stands in for the module's, which is what lets the
    clause-4 nodes assert on the WARNING the code emits rather than on the string
    literal that formats it.
    """
    consts = [ln for hunk in backend_hunks for ln in hunk
              if re.match(r"^_[A-Z0-9_]+ = ", ln)]
    funcs = (_block(backend_hunks, r"^def _frame_budget\(")
             + _block(backend_hunks, r"^def _frame_cap_message\("))
    methods = (_block(backend_hunks, r"^    def _frame_cap\(")
               + _block(backend_hunks, r"^    def _warmup_frame_cap\(")
               + _block(backend_hunks, r"^    def _frame_cap_hit\("))
    assert consts and funcs and methods, (
        "the patch adds none of the frame-cap helpers — no cap exists to test")

    class _RecordingLogger:
        def __init__(self):
            self.warnings: "list[str]" = []

        def warning(self, msg, *args):
            self.warnings.append(msg % args if args else str(msg))

        def info(self, msg, *args):
            pass

    log = _RecordingLogger()
    namespace: dict = {"math": __import__("math"), "logger": log}
    exec("import math\nfrom pathlib import Path\n" + _source(consts + funcs),
         namespace)   # noqa: S102
    dedented = [ln[4:] if ln.startswith("    ") else ln for ln in methods]
    exec(_source(["class _Backend:"] + ["    " + d for d in dedented]), namespace)

    def _budget(text_len: int, streaming_opts: "dict | None" = None) -> int:
        """The patch's pure `_frame_budget` evaluated at the bounds the patch ships.

        `_frame_budget` takes its bounds as arguments — that is what makes it a pure
        function, and what the clause-2 nodes evaluate — so the defaults have to be
        supplied by the caller, and they are taken from the constants in the same
        patch rather than retyped here. `_frame_cap` is the same arithmetic with a
        `optimization.streaming` dict in front of it, which is what the clause-3 nodes
        exercise against the defaults and against overriding keys.
        """
        opts = streaming_opts or {}
        return namespace["_frame_budget"](
            text_len,
            float(opts.get("max_frames_per_char",
                           namespace["_DEFAULT_MAX_FRAMES_PER_CHAR"])),
            int(opts.get("max_frames_floor", namespace["_DEFAULT_MAX_FRAMES_FLOOR"])),
            int(opts.get("max_frames_ceiling",
                         namespace["_DEFAULT_MAX_FRAMES_CEILING"])))

    class Cap:
        """Everything the test needs, with the real callables in it."""

        budget = staticmethod(_budget)
        message = staticmethod(namespace["_frame_cap_message"])
        backend = namespace["_Backend"]()
        logger = log
        default_floor = namespace["_DEFAULT_MAX_FRAMES_FLOOR"]
        default_ceiling = namespace["_DEFAULT_MAX_FRAMES_CEILING"]
        default_per_char = namespace["_DEFAULT_MAX_FRAMES_PER_CHAR"]
        default_fps = namespace["_DEFAULT_FRAMES_PER_SECOND"]
        warmup_per_char = namespace["_WARMUP_FRAMES_PER_CHAR"]

    return Cap()


def _seconds(frames: int) -> float:
    return frames / FRAMES_PER_SECOND


# ── clause 1: every streaming call site is capped ─────────────────────────

def test_the_patch_adds_a_max_frames_argument_to_every_streaming_call(backend_added):
    """Clause 1. Eight streaming calls, eight added `max_frames=` arguments.

    `api/backends/optimized_backend.py` has exactly eight calls into the two vendored
    streaming methods: three in `_warmup_base_model`, three in
    `_warmup_customvoice_model`, one serving CustomVoice path
    (`generate_speech_streaming`) and one serving voice-clone path
    (`generate_voice_clone_streaming`). Every one of them kept the vendored 10000-frame
    budget before this patch.

    The count is of added *argument* lines — a line whose whole content is
    `max_frames=<expr>,` — so the cap-hit warning's message text, which also contains
    the word `max_frames`, cannot be mistaken for a capped call. And the expression
    must be one of the two derivations: `cap_frames` on a serving path, or
    `_warmup_frame_cap(<that call's own text expression>, streaming_opts)` on a warmup
    one, which is the half that stops a six-padded warmup utterance being clipped by a
    budget derived from its un-padded length.
    """
    args = [ln.strip() for ln in backend_added
            if re.match(r"^ {4,}max_frames=\w.+,$", ln)]
    assert len(args) == 8, (
        f"the patch adds {len(args)} max_frames= arguments; the backend has 8 "
        f"streaming calls (3 + 3 warmup, 1 CustomVoice serving, 1 voice-clone "
        f"serving) and each must pass one, or that request keeps the vendored "
        f"{VENDORED_FRAMES}-frame (~{_seconds(VENDORED_FRAMES):.0f} s) budget: {args}")
    serving = [a for a in args if a == "max_frames=cap_frames,"]
    warmup = [a for a in args if "_warmup_frame_cap(" in a]
    assert len(serving) == 2, (
        f"expected the 2 serving paths to pass the pre-computed cap_frames; got "
        f"{serving}. Deriving it inline per chunk would re-read the config per "
        "iteration of the token loop.")
    assert len(warmup) == 6, (
        f"expected the 6 warmup calls to derive their own budget; got {warmup}")
    assert len(serving) + len(warmup) == len(args), (
        f"a max_frames= argument that is neither the serving cap nor a warmup "
        f"derivation: {set(args) - set(serving) - set(warmup)}")


def test_no_warmup_call_gets_a_budget_its_own_text_could_exhaust(
        backend_hunks, tts_cap):
    """Warmup must never be clipped, so its cap has to clear what its text can emit.

    `_warmup_frame_cap` exists because the warmup passes are the one place a cap could
    do harm instead of good: they exist to compile the CUDA graphs and stabilise the GPU
    power state, and a warmup cut short leaves the first real request paying for it. So
    the serving derivation is not used on them — it floors at 750 frames for any short
    text, which is generous here, but its ceiling of 1500 would be a real bound on a
    long synthetic input. The warmup derivation is `4.0` frames per character with a
    2-second floor, so its answer is a multiple of the text it is given and can never
    be less than the text.

    The literals are read out of the patch rather than retyped: a warmup string edited
    to be longer has to still clear the bar, and a bar built from a copy of today's
    strings would not notice.
    """
    texts = set()
    for hunk in backend_hunks:
        for ln in hunk:
            hit = re.search(r"_warmup_frame_cap\(\"([^\"]*)\"", ln)
            if hit:
                texts.add(hit.group(1))
    assert len(texts) == 6, (
        f"expected the 6 warmup literals the backend passes to `_warmup_frame_cap`, "
        f"found {len(texts)} in the patch: {sorted(texts)}")
    for text in sorted(texts):
        cap = tts_cap.backend._warmup_frame_cap(text, {})
        assert cap >= int(len(text) * 4.0), (
            f"warmup cap {cap} is below 4 frames/char for a {len(text)}-character "
            "warmup string: the budget stopped tracking the text it has to speak")
        assert cap >= int(2 * tts_cap.default_fps), (
            f"warmup cap {cap} is under the 2-second floor")
        assert cap > len(text), (
            f"warmup cap {cap} does not exceed the {len(text)} characters of its own "
            "text: at one frame per character — the most any of the healthy requests "
            f"came close to, 41 frames for a whole alert — a warmup pass could be cut "
            "off before it finished speaking")
        assert cap <= tts_cap.backend._frame_cap(text, {}), (
            f"warmup cap {cap} exceeds what the serving path would allow for the "
            "same text, so warmup is asking for more than any real request gets")


def test_a_240_character_utterance_gets_between_sixty_and_120_seconds(tts_cap):
    """Clause 2, first half: a Guardian alert is un-clippable.

    240 characters is roughly the longest alert the guardian speaks (#1478 capped it
    at 240 chars). The clause asks for at least 60 s and at most 120 s of audio at
    12.5 Hz — 750 to 1500 frames. The floor is not an arbitrary round number: 60 s is
    the client ceiling `max_audio_seconds: 60.0` that 7a60c0e1 landed in
    `agent-services/guardian/speak.py:88`, so server and client stop at the same length
    of audio, and it is 3x the longest healthy request ever recorded on this box
    (`audio=19.68s`, `chunks=41`).
    """
    cap = tts_cap.backend._frame_cap("x" * ALERT_CHARS, {})
    assert 750 <= cap <= 1500, (
        f"a {ALERT_CHARS}-character utterance gets {cap} frames "
        f"({_seconds(cap):.1f} s); clause 2 requires 750-1500 frames (60-120 s)")
    assert cap > OBSERVED_HEALTHY_FRAMES, (
        f"the cap {cap} frames is under the {OBSERVED_HEALTHY_FRAMES} frames the "
        "longest healthy request on record actually emitted: a real alert would be "
        f"cut at {_seconds(cap):.1f} s")


def test_a_one_word_alert_still_gets_the_floor(tts_cap):
    """Clause 2's floor is what makes short text safe, not just long text.

    A three-character utterance derives 9 frames by the multiplier alone, which is
    0.72 s of audio — the multiplier on its own would clip an alert like "Done." to
    nothing. The floor is the half of the derivation that carries short text, so it
    has to be reachable by the shortest plausible input, not only by a long one.
    """
    assert tts_cap.backend._frame_cap("Done.", {}) == tts_cap.default_floor, (
        "a short utterance does not get the floor, so text-length scaling is "
        "deciding budgets that the floor exists to protect")
    assert tts_cap.backend._frame_cap("", {}) == tts_cap.default_floor


def test_no_input_can_buy_the_vendored_eight_hundred_second_budget(tts_cap):
    """Clause 2, second half: no input reaches 800 s, and none exceeds 2000 frames.

    Tested over inputs an alert path could actually send — the guardian's own longest
    observed messages and far past them — plus the empty string and a whitespace-only
    one. The bound that matters is not the clause's 2000 frames but the vendor's
    10,000: the point of the cap is that a decoder which never emits EOS stops inside
    two minutes of audio instead of thirteen, so an input that could still reach the
    vendored number would leave the defect exactly where it was.
    """
    for n in (0, 1, 3, 40, 120, 240, 500, 1000, 5000, 200_000, 10**7):
        cap = tts_cap.backend._frame_cap("a" * n, {})
        assert cap <= CEILING_FRAMES, (
            f"a {n}-character input buys {cap} frames "
            f"({_seconds(cap):.1f} s); clause 2 caps any input at "
            f"{CEILING_FRAMES} frames ({_seconds(CEILING_FRAMES):.0f} s)")
        assert cap < VENDORED_FRAMES, (
            f"a {n}-character input still reaches the vendored "
            f"{VENDORED_FRAMES}-frame budget the runaway consumed")
    assert tts_cap.default_ceiling < CEILING_FRAMES <= VENDORED_FRAMES, (
        f"the in-code ceiling {tts_cap.default_ceiling} does not sit below the "
        f"clause bound {CEILING_FRAMES}, so the bound above is not doing the work")


# ── clause 3: defaults and overrides ──────────────────────────────────────

def test_the_defaults_hold_on_a_tree_that_never_reapplied_the_config_hunk(tts_cap):
    """Clause 3, first half: the config keys are optional, in both directions.

    `config.yaml` lives inside the untracked clone. A tree that applied an older
    version of this patch, or that was set up from `SETUP.md` and never re-synced, has
    `optimization.streaming` with `decode_window_frames` and `emit_every_frames` and no
    cap keys at all — and the running service must still be bounded there. Two shapes
    stand in for that state: the streaming section as it exists today, and no section
    at all, which is what an unpatched or hand-edited config gives the code.
    """
    today = {"decode_window_frames": 80, "emit_every_frames": 6}
    for streaming_opts in ({}, today, {"emit_every_frames": 6}):
        opts = streaming_opts
        got = tts_cap.backend._frame_cap("x" * ALERT_CHARS, opts)
        want = tts_cap.budget(ALERT_CHARS, {})
        assert got == want, (
            f"with streaming section {opts!r} the cap is {got} frames, but the "
            f"defaults alone give {want}: a key-less tree is reading a different "
            "budget from a patched-config tree")
        assert 750 <= got <= 1500, (
            f"a key-less tree yields {got} frames for a {ALERT_CHARS}-char alert, "
            "outside clause 2's 750-1500")
    # and the empty-section path cannot fall back to an unbounded default
    assert tts_cap.backend._frame_cap("a" * 200_000, {}) <= CEILING_FRAMES


def test_the_config_keys_override_the_in_code_defaults(tts_cap):
    """Clause 3, second half: an operator who sets the keys gets the number set.

    Three keys, one per bound, read from `optimization.streaming` the way
    `emit_every_frames` already is (`streaming_opts.get(name, default)`). Each is
    checked with a value that is *different* from the default, so a code path that
    silently ignored the key could not pass. `max_frames_floor` has to raise the floor
    for short text; `max_frames_ceiling` has to clamp long text below the default
    ceiling; `max_frames_per_char` has to change the slope in the middle of the range
    where the slope, not a bound, decides.
    """
    mid = "x" * 400                       # 1200 frames: between floor and ceiling
    assert tts_cap.budget(len(mid), {}) == 1200, (
        "sanity: 400 characters should be slope-led at the default multiplier "
        f"({tts_cap.default_per_char}/char), not bounded by floor "
        f"{tts_cap.default_floor} or ceiling {tts_cap.default_ceiling}")
    # The floor is lifted here only so the multiplier is what decides: at the default
    # floor of 750 a 1.0/char budget over 400 chars would land on the floor (400 ->
    # 750) and the assertion would pass whether or not the key was read.
    assert tts_cap.backend._frame_cap(
        mid, {"max_frames_per_char": 1.0, "max_frames_floor": 100}) == 400, (
        "max_frames_per_char=1.0 did not change the slope, so the multiplier is "
        "being read from the defaults and not from config")
    assert tts_cap.backend._frame_cap("Done.", {"max_frames_floor": 1200}) == 1200, (
        "max_frames_floor did not raise the short-text budget")
    assert tts_cap.backend._frame_cap("x" * 5000, {"max_frames_ceiling": 900}) == 900, (
        "max_frames_ceiling did not clamp a long input")
    # a config that raises the ceiling beyond the clause bound still cannot reach
    # the vendored budget, because the serving code passes what config says
    assert tts_cap.backend._frame_cap("x" * 5000, {"max_frames_ceiling": 3000}) == 3000


def test_the_tracked_config_hunk_ships_keys_the_code_agrees_with(patch_sections, tts_cap):
    """Clause 3's two halves meet in the tracked `config.yaml` hunk.

    The keys the patch writes into `optimization.streaming` must parse as numbers and
    must not contradict the in-code defaults — a config hunk saying `99999` while the
    code says 750 means every re-applied tree behaves differently from one that was
    never re-applied, which is exactly the split clause 3 exists to prevent. This reads
    the hunk as text because `config.yaml` itself is in the untracked clone.
    """
    cfg = patch_sections.get(CONFIG)
    assert cfg, "the patch no longer carries a config.yaml hunk"
    added = {m.group(1): m.group(2).strip()
             for ln in _added(cfg)
             if (m := re.match(r"^\s+([a-z_]+):\s*(\S+)", ln))}
    for key in ("max_frames_per_char", "max_frames_floor", "max_frames_ceiling"):
        assert key in added, (
            f"{key} is missing from the config hunk, so a re-applied tree cannot "
            "tune the cap the way the code lets it")
    assert float(added["max_frames_floor"]) == tts_cap.default_floor, (
        f"config floor {added['max_frames_floor']} != in-code floor "
        f"{tts_cap.default_floor}")
    assert float(added["max_frames_ceiling"]) == tts_cap.default_ceiling, (
        f"config ceiling {added['max_frames_ceiling']} != in-code ceiling "
        f"{tts_cap.default_ceiling}")
    assert float(added["max_frames_per_char"]) == tts_cap.default_per_char
    assert float(added["max_frames_floor"]) == int(60 * FRAMES_PER_SECOND), (
        "the floor is no longer 60 s of audio, the client ceiling from "
        "agent-services/guardian/speak.py's max_audio_seconds: 60.0")


# ── clause 4: a cap hit is greppable ──────────────────────────────────────

def test_a_cap_hit_warns_naming_the_cap_and_the_frames_emitted(tts_cap):
    """Clause 4: the runaway becomes a greppable event instead of a silent INFO.

    Before this, the only trace of a 378-second runaway was
    `Voice clone stream done: total=378.74s audio=800.00s chunks=1667` at INFO — a
    line that looks like a long request, and is greppable for nothing in particular.
    The warning has to name the cap in frames and the frames emitted, because the
    question a reader has is "was the cap too tight, or did the decoder run away",
    and only the pair of numbers answers it: cap == emitted is a runaway, cap >> is
    the padding arithmetic.

    Asserted on the emitted string through a recording logger, so renaming the
    placeholder or dropping a number from the message fails here even though the code
    still runs.
    """
    tts_cap.logger.warnings.clear()
    assert tts_cap.backend._frame_cap_hit(750, 750, {}) is None
    assert len(tts_cap.logger.warnings) == 1, (
        f"an exact cap hit logged {len(tts_cap.logger.warnings)} warnings; the "
        "runaway has to leave exactly one witness line")
    msg = tts_cap.logger.warnings[0]
    assert "cap reached" in msg, (
        f"the message is not greppable for the phrase the item names: {msg!r}")
    assert "750" in msg and "750" in msg, (
        f"the warning does not carry both numbers: {msg!r}")
    assert msg.count("750") >= 2, (
        f"the warning names one number twice by coincidence rather than both the "
        f"cap and the frames emitted: {msg!r}")
    tts_cap.logger.warnings.clear()
    tts_cap.backend._frame_cap_hit(1500, 1500, {})
    assert "1500" in tts_cap.logger.warnings[0], (
        "the cap in the message is a constant, not the cap that was hit")
    assert "= 120s" in tts_cap.logger.warnings[0], (
        "the message gives frames but not the seconds they are worth, so a reader "
        "of a `cap reached` line cannot tell 1500 frames of audio from 1500 chunks")


def test_a_run_that_finishes_at_eos_stays_silent(tts_cap):
    """Clause 4's other edge: nine-tenths of a cap is not a cap hit.

    A healthy request ends at EOS with far fewer frames than it was allowed, and
    1,295 of those lines exist in the log against 2 runaways. Warning at 90% of the
    budget would bury the one signal in noise and get the warning switched off, so
    the predicate must be `emitted >= cap` and nothing looser.
    """
    tts_cap.logger.warnings.clear()
    for emitted in (0, 6, 744, 749, 749.999):
        tts_cap.backend._frame_cap_hit(750, emitted, {})
    assert tts_cap.logger.warnings == [], (
        f"a stream that stopped short of the cap warned: {tts_cap.logger.warnings}")
    # And the silence is the *method's* decision, not a property of the formatter:
    # `_frame_cap_message` is reached only on a cap hit, so it must not be able to
    # talk itself out of one by returning None for an awkward pair.
    assert tts_cap.message(750, 749, 12.5).startswith(
        "streaming frame cap reached"), (
        "the formatter returned something other than the witness for an under-cap "
        "pair, which is harmless on its own, but a `None` here would mean a cap hit "
        "could be silently dropped by whichever caller reached for it")


def test_the_witness_counts_the_loop_it_is_wired_into_and_fires_after_it(
        patch_sections, backend_hunks):
    """Clause 4's wiring: the counter is in the body, and the check is after the loop.

    Two halves, both graded, and both read out of a hunk's **post-image** because only
    a post-image is contiguous source — the order of lines there is the order they
    execute. `streamed_frames += emit_every_frames` has to sit in the generator's body
    at the same indentation as the `yield` it precedes, so a chunk is counted even when
    the consumer abandons the stream mid-flight — the case where the decoder is burning
    GPU and nobody is listening. And `self._frame_cap_hit(...)` has to sit at *lower*
    indentation than that `yield`, i.e. outside the loop, because the vendored method
    enforces `max_frames` by returning and the total is only known when iteration ends:
    a witness inside the loop would fire on the first chunk and then on every one
    after it.

    Both serving generators must carry the pair, which is why the counts are 2 and not
    1 — the two runaway lines in `agent-tts.err` were both voice-clone requests, and
    nothing says the next one is not a CustomVoice request.
    """
    added = _added(patch_sections[BACKEND])
    counters = [ln.strip() for ln in added
                if ln.strip() == "streamed_frames += emit_every_frames"]
    assert len(counters) == 2, (
        f"the chunk counter is incremented {len(counters)} time(s); both serving "
        "generators must count, or one path emits its witness from a counter that "
        "stayed at zero")
    hits = [ln.strip() for ln in added
            if ln.strip() == "self._frame_cap_hit(cap_frames, streamed_frames, "
                             "streaming_opts)"]
    assert len(hits) == 2, (
        f"the witness is called with the serving signature {len(hits)} time(s), "
        f"expected 2 (CustomVoice and voice-clone): {hits}")

    checked = 0
    for hunk in backend_hunks:
        idx = lambda pred: next((i for i, ln in enumerate(hunk)
                                 if pred(ln.strip())), None)
        counter = idx(lambda t: t == "streamed_frames += emit_every_frames")
        if counter is None:
            continue
        yielded = idx(lambda t: t.startswith("yield chunk, sr"))
        witness = idx(lambda t: t.startswith("self._frame_cap_hit("))
        checked += 1
        assert yielded is not None and witness is not None, (
            f"a hunk with the counter has no `yield chunk, sr` or no witness line to "
            f"order against: {[ln.strip()[:50] for ln in hunk]}")
        ind = lambda i: len(hunk[i]) - len(hunk[i].lstrip())
        assert counter < yielded, (
            f"the counter at column {ind(counter)} comes after the yield at column "
            f"{ind(yielded)}: chunks streamed before a consumer stops iterating would "
            "not be counted")
        assert ind(counter) == ind(yielded), (
            f"the counter is indented {ind(counter)} and the yield {ind(yielded)}: "
            "they are not in the same block, so the increment is not in the loop body")
        assert ind(witness) < ind(yielded), (
            f"the witness is indented {ind(witness)}, not outside the loop holding "
            f"the yield at {ind(yielded)}: it would fire per chunk instead of once "
            "when generation ends")
    assert checked == 2, (
        f"only {checked} hunk(s) carry the counter, expected 2 — one serving path is "
        "unwired")


# ── the process boundary: does this patch actually apply? ─────────────────

def _patched_clone(tmp_path):
    """Reconstruct the pristine clone, forward-apply the tracked patch, return the tree.

    Returns None when no Qwen3-TTS clone exists on this machine. The clone is
    untracked (`SETUP.md:1351`), so it exists only where somebody ran the SETUP clone:
    this checkout, or — when the suite runs from an automod worktree, which is where the
    gate runs it — one of the other worktrees `git worktree list` reports. Skipping in a
    worktree would leave this, the only node across a real process boundary, never
    running anywhere.

    The pristine file is recovered by reverse-applying the patch **as it stood before
    this round**, not the working copy's. The live clone is the post-image of whichever
    committed patch a human last applied (`SETUP.md:838-841`), and it stays that way
    until owed step 1 re-applies and restarts — so the moment the tracked patch changes,
    reverse-applying the changed one asserts about a tree that cannot exist yet, and the
    only way to make such a node pass would be to edit the running service from the
    round. Committed versions are tried newest-first and the first that undoes cleanly is
    the one the clone was built with; the tracked patch then goes on forward, which is
    exactly what owed step 1 will do.

    Every file either patch names is copied: a round that adds a hunk to a file the
    previous patch already touched would otherwise have the forward apply die on a file
    that was never put in the scratch tree.
    """
    git = shutil.which("git")
    assert git, "no git on PATH to apply the patch with"
    rel = Path("agent-services") / "services" / "tts"
    roots = [REPO, Path(__file__).resolve().parent.parent]
    listing = subprocess.run([git, "worktree", "list", "--porcelain"],
                             capture_output=True, text=True)
    # Every listed worktree, main first. The gate runs the suite from an automod
    # worktree, whose checkout holds no clone, so the main tree is the candidate that
    # matters — and `--porcelain` is the only listing this box's git (2.55) answers: a
    # probe against an unsupported flag prints nothing and silently drops the one
    # candidate with the clone in it, which reads as "no clone on this machine".
    for line in listing.stdout.splitlines():
        if line.startswith("worktree "):
            roots.append(Path(line.split(" ", 1)[1]))
    clone = next((c / rel / "qwen3-tts" for c in roots
                  if (c / rel / "qwen3-tts" / BACKEND).is_file()), None)
    if clone is None:
        return None

    prior = _previously_applied_patch(git, clone, tmp_path)
    if prior is None:
        return None

    touched = _patch_paths(PATCH) | _patch_paths(prior)
    assert touched, "neither patch names a file, so there is nothing to reconstruct"
    tree = tmp_path / "pristine"
    tree.mkdir()
    for rel_path in sorted(touched):
        src = clone / rel_path
        if not src.is_file():
            return None
        dst = tree / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    subprocess.run([git, "-C", str(tree), "init", "-q"], check=True)
    subprocess.run([git, "-C", str(tree), "add", "-A"], check=True)
    subprocess.run([git, "-C", str(tree), "-c", "user.email=t@t", "-c",
                    "user.name=t", "commit", "-qm", "live"], check=True)
    reverse = subprocess.run([git, "-C", str(tree), "apply", "-R", "-p1", str(prior)],
                             capture_output=True, text=True)
    assert reverse.returncode == 0, (
        "the live untracked clone is the post-image of no committed patch version, so "
        f"no pristine file can be reconstructed here: {reverse.stderr[-400:]}")
    subprocess.run([git, "-C", str(tree), "-c", "user.email=t@t", "-c",
                    "user.name=t", "add", "-A"], check=True)
    subprocess.run([git, "-C", str(tree), "-c", "user.email=t@t", "-c",
                    "user.name=t", "commit", "-qm", "pristine"], check=True)
    check = subprocess.run([git, "-C", str(tree), "apply", "-p1", "--check", str(PATCH)],
                           capture_output=True, text=True)
    assert check.returncode == 0, (
        f"the regenerated patch does not apply to a pristine clone: "
        f"{check.stderr[-600:]}")
    subprocess.run([git, "-C", str(tree), "apply", "-p1", str(PATCH)], check=True)
    return tree


def _patch_paths(patch: Path) -> set:
    """The paths a `git diff`-format patch touches, resolved the way `-p1` resolves them.

    Read off the `+++` lines, which carry the destination name under whatever prefix the
    diff was generated with — this patch uses `i/`/`w/`, not git's usual `a/`/`b/`,
    because `SETUP.md:838-841` applies it with `-p1`. One path component is stripped to
    match, so the names line up with the clone's own layout.
    """
    out = set()
    for line in patch.read_text(encoding="utf-8").splitlines():
        if line.startswith("+++ ") and not line.startswith("+++ /dev/null"):
            name = line[4:]
            out.add(name.split("/", 1)[1] if "/" in name else name)
    return out


def _previously_applied_patch(git: str, clone: Path, tmp_path: Path):
    """Newest committed version of the patch that reverse-applies to the live clone.

    Written under `tmp_path` and returned — never beside the clone, which lives inside
    the live tree and is not this suite's to write into. None when the clone matches no
    committed version: the honest "cannot reconstruct here" answer, not a green run.
    """
    rel_patch = str(PATCH.relative_to(REPO))
    history = subprocess.run([git, "-C", str(REPO), "log", "--format=%H", "--",
                             rel_patch], capture_output=True, text=True)
    for sha in history.stdout.split()[:8]:
        shown = subprocess.run([git, "-C", str(REPO), "show", f"{sha}:{rel_patch}"],
                               capture_output=True, text=True)
        if shown.returncode != 0 or not shown.stdout.strip():
            continue
        probe = tmp_path / f"committed-{sha[:8]}.patch"
        probe.write_text(shown.stdout, encoding="utf-8")
        undo = subprocess.run([git, "-C", str(clone), "apply", "-R", "-p1", "--check",
                              str(probe)], capture_output=True, text=True)
        if undo.returncode == 0:
            return probe
    return None


def test_the_regenerated_patch_applies_to_the_pristine_clone_and_caps_every_call(tmp_path):
    """Clause 1 at the boundary: real `git apply` on a fresh clone, then the real parser.

    A text-level assertion is satisfiable by a hunk that merely describes the right
    change; this one requires the patch to apply and the patched file to carry
    `max_frames` on all eight streaming calls, warmups included.
    """
    tree = _patched_clone(tmp_path)
    if tree is None:
        pytest.skip(
            "no qwen3-tts clone on this machine, so the patch cannot be exercised "
            "against the tree it was generated from")

    import ast
    src = (tree / BACKEND).read_text(encoding="utf-8")
    ast.parse(src)
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") in
             ("stream_generate_voice_clone", "stream_generate_custom_voice")]
    assert len(calls) == 8, (
        f"the reconstructed file has {len(calls)} streaming calls, expected 8 — the "
        "clone is not the file the patch was written against")
    uncapped = [f"{n.lineno}:{getattr(n.func, 'attr')}" for n in calls
                if not any(k.arg == "max_frames" for k in n.keywords)]
    assert not uncapped, (
        f"after applying the patch these streaming calls still take the vendored "
        f"{VENDORED_FRAMES}-frame budget: {uncapped}")


def test_the_warmup_budget_needs_no_argument_the_warmup_never_binds(tmp_path):
    """A warmup that cannot resolve its own budget argument fails the model load.

    `streaming_opts` is a local of `warmup()` (`optimized_backend.py:193`) and of the two
    serving methods (`:512`, `:611`). It is NOT a local of `_warmup_base_model` or
    `_warmup_customvoice_model`, where four of the six warmup calls live. Passing it
    there raises `NameError`, and because the vendored call that carries it sits inside a
    `try/except Exception` that logs and keeps loading, the failure surfaces as a warmup
    that silently did nothing — the model boots uncompiled and the first real request
    pays the compile. Nothing in this repo boots the service, so this node is the only
    thing between that bug and production.

    It walks every function body in the patched file and requires that no name it reads
    is unbound in the scopes Python would consult: the function's own parameters and
    bindings, its enclosing function's, the module's, the builtins. A class body is
    deliberately NOT in its methods' scope — a method that writes `streaming_opts` bare
    and means `self.streaming_opts` is exactly the NameError being looked for.
    """
    import ast
    import builtins

    tree = _patched_clone(tmp_path)
    if tree is None:
        pytest.skip(
            "no qwen3-tts clone on this machine, so the patch cannot be exercised "
            "against the tree it was generated from")

    root = ast.parse((tree / BACKEND).read_text(encoding="utf-8"))

    def _bindings(node, names: set) -> None:
        """Names the statements directly in `node` bind, not those a nested scope binds."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                                  ast.ClassDef)):
                continue
            if isinstance(child, ast.Name) and isinstance(
                    child.ctx, (ast.Store, ast.Del)):
                names.add(child.id)
            elif isinstance(child, ast.ExceptHandler) and child.name:
                names.add(child.name)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(child, (ast.Global, ast.Nonlocal)):
                names.update(child.names)
            _bindings(child, names)

    def _params(fn, names: set) -> None:
        a = fn.args
        for arg in [*getattr(a, "posonlyargs", []), *a.args, *a.kwonlyargs,
                    *([a.vararg] if a.vararg else []),
                    *([a.kwarg] if a.kwarg else [])]:
            names.add(arg.arg)

    module_names: set = set(dir(builtins))
    _bindings(root, module_names)
    for stmt in root.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_names.add(stmt.name)

    offenders: list = []

    def _walk_body(body, scope):
        for stmt in body:
            _walk(stmt, scope)

    def _walk(node, scope):
        # Only bodies are walked, so a function's decorators, defaults and annotations —
        # which Python evaluates in the ENCLOSING scope, at def time — are never graded
        # against the function's own locals. That would report a NameError that cannot
        # happen, and a check with false failures gets silenced.
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Name):
                if isinstance(child.ctx, ast.Load) and child.id not in scope:
                    offenders.append(f"line {child.lineno} reads {child.id!r}")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _check_fn(child, scope, is_method=False)
            elif isinstance(child, ast.Lambda):
                local: set = set()
                _params(child, local)
                _walk(child.body, frozenset(scope | local))
            elif isinstance(child, ast.ClassDef):
                class_scope: set = set(scope)
                _bindings(child, class_scope)
                for stmt in child.body:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                         ast.ClassDef)):
                        class_scope.add(stmt.name)
                for stmt in child.body:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # A method's enclosing scope is the module: Python skips the
                        # class body when resolving a bare name inside a method.
                        _check_fn(stmt, frozenset(module_names), is_method=True)
                    elif isinstance(stmt, ast.ClassDef):
                        _walk(stmt, frozenset(class_scope))
                    else:
                        _walk(stmt, frozenset(class_scope))
            else:
                _walk(child, scope)

    def _check_fn(fn, base_scope, *, is_method: bool):
        local: set = set()
        _params(fn, local)
        _bindings(fn, local)
        scope = frozenset(module_names | local) if is_method else \
            frozenset(base_scope | local)
        _walk_body(fn.body, scope)

    for stmt in root.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _check_fn(stmt, frozenset(module_names), is_method=False)
        else:
            _walk(stmt, frozenset(module_names))

    assert not offenders, (
        "the patched backend reads names no enclosing scope binds: "
        + "; ".join(sorted(set(offenders)))
        + " — a NameError inside a warmup's `try/except` is a warmup that quietly did "
        "nothing, which is the failure this clause must not ship")

    defined = {f.name for f in ast.walk(root)
               if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert {"_warmup_base_model", "_warmup_customvoice_model"} <= defined, (
        "the patched file no longer defines both warmup methods, so this node's premise "
        "— that the sweep covers the methods four of the six warmup calls live in — is "
        "stale")


# ── clause 5: the witness bytes have a history now ────────────────────────


WITNESS = "backlog/data/voice.log"


def test_the_committed_voice_log_witness_carries_the_timeout_axis_figure():
    """Clause 5: the log the TimeoutError axis was quoted from is committed bytes.

    The item asked for first-byte and voice-prompt latency probes because six
    `TimeoutError` lines kept appearing against a healthy ~0.27 s TTFB. Triage
    replaced that clause with a causal finding — the guardian gives up on a request
    whose decoder has run away to the vendored budget — and the finding is about a
    rotating operational file, so it needs bytes a later reader can recount. This
    node is that recount: the counts come from the committed file, not from the live
    one, which has no history and had already grown by the time the round ran.

    Two numbers and one line, all re-derived here rather than carried: 300 lines, 6
    TimeoutErrors, and the 09-28 19:45:47 entry that sits 60 s after the request
    `agent-tts.err:6490-6493` shows running to `audio=800.00s chunks=1667` — the
    server-side runaway and the client-side timeout, same axis, two files.
    """
    witness = VAULT_ROOT / WITNESS
    assert witness.is_file(), (
        f"{witness} is not in the vault, so the 6-TimeoutError figure this item's "
        "instrumentation clause was replaced by has no citable source")
    lines = witness.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 300, (
        f"the committed witness holds {len(lines)} lines, not the 300 the clause "
        "quotes — `wc -l < backlog/data/voice.log` is the figure the item states")
    timeouts = [ln for ln in lines if "TimeoutError" in ln]
    assert len(timeouts) == 6, (
        f"{len(timeouts)} TimeoutError lines in the witness, not the 6 quoted: "
        f"{[t[:20] for t in timeouts]}")
    assert any(ln.startswith("2026-09-28T19:45:47") for ln in timeouts), (
        "the witness no longer contains the 09-28 19:45:47 timeout, which is the "
        "line that ties the client's timeout to the 19:44:47 request agent-tts.err "
        "shows consuming the whole 10,000-frame budget")
