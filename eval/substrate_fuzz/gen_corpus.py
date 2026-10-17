"""The program that emits `corpus.json`, with every expected value computed (#2620).

Arm A's postconditions are literals in a data file, and a literal somebody typed is a
liability: a mistyped digest makes a task red forever, and a digest quietly re-typed to
match whatever the box produced turns the instrument into the #644 defect class — a green
assertion that asserts nothing. So the expected values here are derived, in this file,
from fixtures this file builds: a PNG assembled from bytes, a base64 encoding, a sha256 of
generated noise. `corpus.json` is then the checked-in *output*, reviewed as data.

The one class of expected value this generator cannot compute is what a tool's own answer
looks like — the shape of `rg`'s or bash's error text — so those tasks assert a substring
that was read off a real run of the tool on this box (the file records which), and never
a digest of a rendering the harness owns.

Run it and commit the result:

    cd ~/lloyd && .venvs/lloyd/bin/python -m eval.substrate_fuzz.gen_corpus
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import zlib
from pathlib import Path

from app.harness.tool_result_spill import SPILL_THRESHOLD_CHARS
from eval.substrate_fuzz.corpus import POSTCONDITION_KINDS, sha as sha_b

OUT = Path(__file__).resolve().parent / "corpus.json"

# ── fixtures ────────────────────────────────────────────────────────────────

def png() -> bytes:
    """A real, small PNG — 4x3 RGB, no palette, one IDAT.

    Not a random blob with a .png name: the point of this fixture is that the file
    legitimately contains NUL bytes and high-bit bytes, which is what silently truncates
    when a boundary decides text is involved. Hand-assembled rather than drawn so the
    digest is stable across machines and zlib versions (fixed level, one flush).
    """
    w, h = 4, 3
    raw = b"".join(b"\x00" + bytes(((y * w + x) * 37) % 256 for x in range(w) for c in range(3))
                   for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


PDF = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n"
#: The null byte Orellana names as the second interface fault: "a null byte can be a
#: zero, it could also be a marker in some coding languages that just means that the
#: string is complete". Anything that truncates at it produces a *shorter* answer that
#: still reads as success.
NUL_TEXT = b"before\x00after\n"
EMOJI_TEXT = "face \U0001F600 grinning\n".encode("utf-8")
#: The bytes `printf 'cafe\xcc\x81\n'` actually emits: ASCII `cafe`, then U+0301 COMBINING
#: ACUTE ACCENT in its DECOMPOSED form (`cc 81`), then a newline — 7 bytes. The row that reads
#: this used to declare `"cafe\u0301 combining\n"` as its expected payload, a 16-byte string
#: carrying a word the step never prints, so it went red against a substrate that had answered
#: faithfully (declared `84720c67…`, observed `dcc49242…`, both of them correct digests of
#: different strings) and the first arm-A run reported it as a normalisation finding. It is a
#: corpus bug, and the constant now states the payload the step emits (#2929). The row still
#: detects the fault it was written for: a boundary that precomposed the sequence would answer
#: `caf\xc3\xa9\n`, 6 bytes, and fail this digest.
COMBINING_TEXT = b"cafe\xcc\x81\n"
CJK_TEXT = "日本語 nbsp\u00a0X\n".encode("utf-8")
CRLF = b"a\r\nb\r\n"
PLAIN = b"plain payload\n"
WRITTEN_TEXT = "written by arm A: café \U0001F600 end\n"
SCRIPT = "#!/bin/sh\necho script-ran\n"


def noise(n: int, seed: str) -> bytes:
    """`n` deterministic bytes, no RNG. A fixture whose bytes come from the clock would
    make its own expected digest unreproducible on the next run."""
    out = bytearray()
    i = 0
    while len(out) < n:
        out += hashlib.sha256(f"{seed}:{i}".encode()).digest()
        i += 1
    return bytes(out[:n])


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_s(s: str) -> str:
    return sha(s.encode("utf-8", "surrogateescape"))


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def lines_file(count: int, width: int, tail: str) -> str:
    """A file of fixed-width lines whose LAST line ends in `tail`.

    Fixed width because the task's whole argument is size: a grep over this file has to
    land above `SPILL_THRESHOLD_CHARS` on one task and below it on another, and the only
    way to place those two on either side of the line without guessing is to choose the
    counts from the width. The tail marker has to be on the last line because a result
    clipped to a prefix keeps the head and loses exactly that.
    """
    body = [f"{i:06d}:" + "x" * max(0, width - 7) for i in range(count - 1)]
    body.append(tail)
    return "\n".join(body) + "\n"


TAIL_BIG = "SUBFUZZ-TAIL-BIG-9d41c2"
TAIL_SMALL = "SUBFUZZ-TAIL-SMALL-4be0a7"
TAIL_SECOND = "SUBFUZZ-TAIL-SECOND-77e1f3"

# ── tasks ───────────────────────────────────────────────────────────────────

TASKS: list[dict] = []


def task(**row) -> None:
    TASKS.append(row)


def ctl(tid: str, domain: str, base: str, control: str, *, title: str, reason: str,
        expected_layers: list[str] | None = None) -> None:
    """Materialise a control by COPYING the base task's row, then adding the break.

    The copy is the mechanism: a control that restated its own steps or its own expectation
    could quietly claim to prove a task it does not run, and the report would still count it
    as that domain's detector. `steps` and the postcondition therefore cannot be passed in —
    they are taken from the base verbatim, and the only thing a control contributes is the
    name of the break it seeds at the tool boundary. The expectation stays the UNSPILLED,
    UNBROKEN one, because the point is that the shipped postcondition goes red for it.
    """
    row = next((r for r in reversed(TASKS) if r["id"] == base), None)
    if row is None:
        raise SystemExit(f"control {tid} names control_for={base!r}, which is not a task "
                         "declared before it — a control pointing at nothing reads as "
                         "coverage and is a hole")
    out = json.loads(json.dumps(row))
    out.update({"id": tid, "domain": domain, "title": title, "reason": reason,
                "control_for": base, "control": control, "control_at": "tool_boundary",
                "expected_layers": expected_layers or []})
    TASKS.append(out)


# ══ byte_roundtrip ══════════════════════════════════════════════════════════
PNG, PDF_B, NUL_B = png(), PDF, NUL_TEXT

task(id="roundtrip-png", domain="byte_roundtrip",
     title="a PNG piped as text through the tool boundary comes back byte-identical",
     reason="the video's headline fault, stated exactly as it happened there: an image goes "
            "in, the same image comes out, nothing errors, the log looks fine, and the bytes "
            "differ. This row therefore reads the fixture raw rather than base64'd, and its "
            "expected value is the digest of the fixture's own bytes — the claim is byte "
            "fidelity and nothing else. Whether it passes is the instrument's answer, not the "
            "corpus's assertion: a row that pre-bakes the answer into its own expectation "
            "cannot find the fault it was written for, and the digest of the base64 of a file "
            "is exactly such a pre-bake.",
     setup=[{"path": "png.bin", "fixture": True}],
     steps=[{"tool": "Bash", "args": {"command": "cat {fixture}", "cwd": "{root}"}}],
     expected_layers=["interface"],
     postcondition={"kind": "stdout_sha256", "expected": sha_b(PNG)})

task(id="rt-pdf-bytes", domain="byte_roundtrip",
     title="a small PDF survives the same round trip",
     reason="a second container format, so one special-cased MIME type cannot explain a "
            "pass.",
     setup=[{"path": "pdf.bin", "fixture": True}],
     steps=[{"tool": "Bash", "args": {"command": "base64 -w0 {fixture}",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha_s(b64(PDF_B))})

task(id="rt-nul-text-bytes", domain="byte_roundtrip",
     title="a file containing a NUL byte round-trips without truncation",
     reason="null bytes are the second recorded interface fault; a boundary that stops at "
            "one returns a shorter answer that still reads as success.",
     setup=[{"path": "nul.bin", "fixture": True}],
     steps=[{"tool": "Bash", "args": {"command": "base64 -w0 {fixture}",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha_s(b64(NUL_B))})

task(id="rt-utf8-emoji-bytes", domain="byte_roundtrip",
     title="a UTF-8 file with an emoji round-trips byte for byte",
     reason="a high-bit sequence that a latin-1 or a `errors=replace` path silently "
            "rewrites, which changes the digest without changing the visible text.",
     setup=[{"path": "emoji.bin", "fixture": True}],
     steps=[{"tool": "Bash", "args": {"command": "base64 -w0 {fixture}",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha_s(b64(EMOJI_TEXT))})

task(id="rt-crlf-length-preserved", domain="byte_roundtrip",
     title="CRLF line endings are not normalised on the way to disk",
     reason="a write path that rewrites \\r\\n as \\n shortens the file by one byte per "
            "line, which a text comparison never notices and a length always does.",
     setup=[{"path": "crlf.bin", "fixture": True}],
     steps=[{"tool": "Bash", "args": {"command": "wc -c < {fixture}",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha_s(f"{len(CRLF)}\n")})

task(id="rt-digest-on-disk", domain="byte_roundtrip",
     title="the bytes on disk hash to the fixture's digest",
     reason="the substrate half of the same claim: the base64 task above proves what the "
            "tool reported, this one proves what the filesystem kept.",
     setup=[{"path": "png.bin", "fixture": True}],
     steps=[{"tool": "Bash", "args": {
         "command": "python3 -c \"import hashlib,sys;"
                    "sys.stdout.write(hashlib.sha256(open(sys.argv[1],'rb')"
                    ".read()).hexdigest())\" {fixture}",
         "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha_s(sha(PNG))})

task(id="rt-shown-side-carries-payload", domain="byte_roundtrip",
     title="the text that would reach messages still carries the payload",
     reason="a postcondition on the tool's own answer can pass while the string appended "
            "to the conversation is something else; this one reads the shown side.",
     setup=[{"path": "nul.bin", "fixture": True}],
     steps=[{"tool": "Bash", "args": {"command": "base64 -w0 {fixture}",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "shown_contains", "expected": b64(NUL_B)})

task(id="rt-write-then-read-back-nul", domain="byte_roundtrip",
     title="content written through Write lands byte-identical, NUL included",
     reason="the write direction of the same boundary, which is where the harness has an "
            "argument rather than a return value to carry the bytes.",
     # Read back with `cat`, not `base64 -w0`, for a reason worth keeping: this domain's
     # seeded break is `strip_at_nul`, and a base64 payload contains no NUL byte to strip,
     # so the base64 version of this row shipped a control that could never fire — a
     # detector that reported itself alive, which is the failure #644 warned about. The
     # expectation is the digest of the bytes this row's own Write step puts on disk, and
     # `cat` is measured on this box to return them unchanged, NUL included: the NUL is not
     # what this transport loses.
     steps=[{"tool": "Write", "args": {"file_path": "{root}/written-nul.bin",
                                       "content": NUL_B.decode("utf-8", "surrogateescape")}},
            {"tool": "Bash", "args": {"command": "cat {root}/written-nul.bin",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha_b(NUL_B)})

ctl("ctl-rt-write-nul-stripped", "byte_roundtrip", "rt-write-then-read-back-nul",
    "strip_at_nul",
    title="control: strip the payload at its NUL byte",
    reason="the seeded break for byte fidelity, and the exact fault the video names. If "
           "this control ever comes back green, the digest check is not looking at the "
           "bytes it claims to.",)

def decode_task(tid: str, fname: str, fx_name: str, fx: bytes, title: str,
                why: str) -> None:
    """One encode-transport-decode round trip whose postcondition is the DIGEST OF A FILE.

    The corpus's other byte_roundtrip tasks check what a tool *answers*; these check the
    other side of the same boundary — the bytes that land on disk after payload has been
    through a transport encoding — which is where the video's fault actually lived, and
    where a sha256 is the strongest thing arm A can ask.
    """
    task(id=tid, domain="byte_roundtrip", title=title, reason=why,
         setup=[{"path": fx_name, "fixture": True}],
         steps=[
             {"tool": "Write", "args": {"file_path": "{root}/" + fname + ".b64",
                                        "content": b64(fx)}},
             {"tool": "Bash", "args": {
                 "command": "base64 -d " + fname + ".b64 > " + fname + ".out",
                 "cwd": "{root}"}}],
         postcondition={"kind": "file_sha256", "path": fname + ".out",
                        "expected": sha(fx)})


decode_task("rt-b64-decode-png-to-disk", "png-dec", "png.bin", PNG,
            "a PNG survives base64 transport and lands byte-identical on disk",
            "the video's fault in its exact shape: the payload goes out through an "
            "encoding layer and comes back as a file, and the check is that the file's "
            "digest is the original's — the encoding is where it broke upstream")
decode_task("rt-b64-decode-pdf-to-disk", "pdf-dec", "pdf.bin", PDF_B,
            "a PDF survives base64 transport and lands byte-identical on disk",
            "a second container format, because a boundary that mangles one codec usually "
            "gets the other one by accident")
decode_task("rt-b64-decode-nul-to-disk", "nul-dec", "nul.bin", NUL_B,
            "text with a NUL byte survives transport and lands with its NUL intact",
            "Orellana's null-byte fault on the disk side rather than the answer side: a "
            "decoder that treats NUL as a terminator produces a shorter file that still "
            "writes without complaint, and only a digest notices")
decode_task("rt-b64-decode-emoji-to-disk", "emoji-dec", "emoji.bin", EMOJI_TEXT,
            "a multi-byte emoji payload survives transport to disk unchanged",
            "the same round trip with a payload whose bytes are all non-ASCII, which is "
            "what a text-assuming encode layer silently rewrites")

task(id="rt-dd-copy-preserves-digest", domain="byte_roundtrip",
     title="a block-level copy of a binary file is byte-identical",
     reason="the purest form of the question — no encoding, no tool logic, just the "
            "filesystem moving bytes — so a failure here is the host and nothing else, "
            "which is what makes the domain's layer split decidable at all",
     setup=[{"path": "png.bin", "fixture": True}],
     steps=[{"tool": "Bash", "args": {
         "command": "dd if=png.bin of=png-dd.bin bs=1 status=none", "cwd": "{root}"}}],
     postcondition={"kind": "file_sha256", "path": "png-dd.bin", "expected": sha(PNG)})

# ══ filesystem ══════════════════════════════════════════════════════════════
DEEP = "a/b/c/d/deep.txt"
SPACY = "sp ace-é.txt"

task(id="fs-nested-mkdir", domain="filesystem",
     title="nested directories can be created and written under the run root",
     reason="mount and permission basics from the video's primitive list: if the substrate "
            "will not let a task build a path, nothing downstream matters.",
     steps=[{"tool": "Bash", "args": {
         "command": f"mkdir -p {DEEP.rsplit('/', 1)[0]} && printf 'deep\\n' > {DEEP}",
         "cwd": "{root}"}}],
     postcondition={"kind": "file_sha256", "path": DEEP, "expected": sha_s("deep\n")})

task(id="fs-copy-is-byte-identical", domain="filesystem",
     title="a filesystem copy of a binary fixture is byte-identical",
     reason="persistent storage and the copy primitive, with no harness layer between the "
            "two files — the control's break lands on the harness side, never here.",
     setup=[{"path": "png.bin", "fixture": True}],
     steps=[{"tool": "Bash", "args": {"command": "cp {fixture} copy.bin",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "file_sha256", "path": "copy.bin", "expected": sha(PNG)})

task(id="fs-filename-with-space-and-accent", domain="filesystem",
     title="a filename with a space and a non-ASCII letter round-trips",
     reason="paths cross the same boundary as payloads, and a quoting bug shows up first "
            "on a name nobody typed by hand.",
     steps=[{"tool": "Bash", "args": {"command": f"printf 'x\\n' > './{SPACY}'",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "file_sha256", "path": SPACY, "expected": sha_s("x\n")})

task(id="fs-preserve-exec-bit-on-copy", domain="filesystem",
     title="permission bits survive a copy",
     reason="filesystem permissions are named in the video as a determinant of task "
            "success; mode is metadata, so it needs its own check rather than a digest.",
     setup=[{"path": "exe.sh", "content": SCRIPT, "mode": "755"}],
     steps=[{"tool": "Bash", "args": {
         "command": "cp exe.sh exe-copy.sh && stat -c '%a' exe-copy.sh",
         "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha_s("755\n")})

task(id="fs-write-roundtrip-exact", domain="filesystem",
     title="what Write writes is what the shell reads back from the file",
     reason="the write direction, read back through the boundary rather than trusted from the "
            "tool's own summary — the summary is the thing under test, and the bytes `cat` "
            "returns are the bytes a model would be shown. Hashing the file from disk was also "
            "a postcondition this domain's control could never redden: the seeded break lands "
            "on the payload crossing the boundary, and the file on disk is whatever the tool "
            "faithfully wrote.",
     steps=[{"tool": "Write", "args": {"file_path": "{root}/written.txt",
                                       "content": WRITTEN_TEXT}},
            {"tool": "Bash", "args": {"command": "cat written.txt", "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha_s(WRITTEN_TEXT)})

task(id="fs-write-into-proc-denied", domain="filesystem",
     title="a write into /proc is denied by the operating system",
     reason="the substrate's own ceiling, with no path component the harness's guard "
            "claims — it is the kernel that says no, which is why the layer here is host.",
     steps=[{"tool": "Bash", "args": {"command": "printf x > /proc/substrate-fuzz-nope",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "exit_shape", "is_error": True},
     expected_layers=["host"])

task(id="fs-write-into-readonly-dir", domain="filesystem",
     title="a write into a read-only directory fails with a filesystem error",
     reason="Orellana's third fault: files that are unavailable where they were written. "
            "Here the ceiling is a mount flag, and the report has to say so.",
     setup=[{"path": "rodir", "dir": True, "mode": "500"}],
     steps=[{"tool": "Bash", "args": {"command": "printf x > rodir/inside.txt",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "exit_shape", "is_error": True},
     expected_layers=["host"])

ctl("ctl-fs-write-flipped", "filesystem", "fs-write-roundtrip-exact", "flip_one_byte",
    title="control: flip the first byte of the payload coming back from the read-back",
    reason="one byte is the whole point: the fault the video found was innocuous to a "
           "developer reading logs, and a digest is the only thing that sees it.",)

# ══ process ═════════════════════════════════════════════════════════════════
task(id="proc-spawn-and-wait", domain="process",
     title="a child process is spawned, waited for, and its output returned",
     reason="whether the container may spawn a process at all is named in the video as a "
            "primitive that decides task success before any model does.",
     steps=[{"tool": "Bash", "args": {"command": "sleep 0.2; echo waited",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha_s("waited\n")})

task(id="proc-backgrounded-children-reaped", domain="process",
     title="backgrounded children are reaped before the result is returned",
     reason="`wait` has to actually wait: a tool that returns while grandchildren still "
            "hold the process group leaks a subtree per call (#2578).",
     steps=[{"tool": "Bash", "args": {
         "command": "sleep 0.2 & sleep 0.2 & wait; echo reaped", "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha_s("reaped\n")})

task(id="proc-nonzero-exit-is-an-error", domain="process",
     title="a non-zero exit is reported as an error result",
     reason="an exit code that arrives as success is the interface swallowing the "
            "substrate's answer, which is the failure mode this whole instrument is for.",
     steps=[{"tool": "Bash", "args": {"command": "printf partial; exit 3",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "exit_shape", "is_error": True})

task(id="proc-timeout-is-reported", domain="process",
     title="a command outliving its timeout is reported rather than left running",
     reason="the reported ceiling matters more than the ceiling: a killed command that "
            "answers as success tells the next caller the file is there.",
     steps=[{"tool": "Bash", "args": {"command": "sleep 5", "timeout": 1,
                                      "cwd": "{root}"}}],
     postcondition={"kind": "exit_shape", "is_error": True},
     expected_layers=["host"])

task(id="proc-shell-metachars-in-literal", domain="process",
     title="shell metacharacters in a quoted literal are not expanded away",
     reason="a `$(` that expands on the way in is an argument-injection path and also "
            "destroys the payload; the digest catches either.",
     steps=[{"tool": "Bash", "args": {
         "command": "printf '%s\\n' '$HOME `id` $(id)'", "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256",
                    "expected": sha_s("$HOME `id` $(id)\n")})

task(id="proc-env-passthrough", domain="process",
     title="an environment variable set for one command reaches it",
     reason="credential injection is the first item in the video's preparation list; the "
            "check is that the value arrives with nothing added, including no newline. A "
            "child shell reads it, not the shell that set it: in `VAR=x printf '%s' \"$VAR\"` "
            "the expansion happens BEFORE the assignment applies to that same simple command, "
            "so the answer is empty and the row reddens on shell semantics instead of on the "
            "substrate — which is what it did before this note was written.",
     steps=[{"tool": "Bash", "args": {
         "command": "SUBFUZZ_PROBE=abc123 sh -c 'printf \"%s\" \"$SUBFUZZ_PROBE\"'",
         "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha_s("abc123")})

ctl("ctl-proc-spawn-clipped", "process", "proc-spawn-and-wait", "clip_to_prefix",
    title="control: clip the returned stdout to a prefix",
    reason="models read a prefix as a complete answer more readily than as a truncated "
           "one, so a clipped result has to be caught by the digest.",)

# ══ size_spill ══════════════════════════════════════════════════════════════
BIG_LINES, SMALL_LINES, LINE_W = 2000, 400, 40
_BIG_FILE = lines_file(BIG_LINES, LINE_W, TAIL_BIG)
_SMALL_FILE = lines_file(SMALL_LINES, LINE_W, TAIL_SMALL)
_SECOND_FILE = lines_file(BIG_LINES, LINE_W, TAIL_SECOND)
assert len(_BIG_FILE) > SPILL_THRESHOLD_CHARS, len(_BIG_FILE)
assert len(_SMALL_FILE) * 2 < SPILL_THRESHOLD_CHARS, len(_SMALL_FILE)

task(id="spill-large-result-resolves", domain="size_spill",
     title="a result over the spill threshold resolves back to its LAST bytes",
     reason="the item's own sharp case: a `<persisted-output>` pointer is not a lost "
            "result, but only if it resolves to the full original bytes. A pointer whose "
            "target is a prefix reads to the model like a short answer.",
     setup=[{"path": "big.txt", "content": _BIG_FILE}],
     steps=[{"tool": "Grep", "args": {"pattern": "^\\d{6}:|SUBFUZZ-TAIL",
                                      "path": "{root}/big.txt",
                                      "output_mode": "content"}}],
     # `content` mode with a pattern every line matches, measured on this box, returns the
     # file's own bytes: 81,983 chars out for 81,983 bytes in, no headings, no line numbers.
     # So the expected payload is the fixture this row writes, not a rendering anybody
     # typed — which is the line this generator will not cross. The claim is the spill
     # round trip: the pointer the model is shown has to resolve back to THESE bytes.
     postcondition={"kind": "stdout_sha256", "expected": sha_s(_BIG_FILE)})

task(id="spill-shown-carries-a-pointer", domain="size_spill",
     title="the shown text for a large result is a pointer, not the payload",
     reason="the other side of the same fact. Without it, a passing "
            "`spill-large-result-resolves` could mean the threshold never fired at all.",
     setup=[{"path": "big.txt", "content": _BIG_FILE}],
     steps=[{"tool": "Grep", "args": {"pattern": "^\\d{6}:|SUBFUZZ-TAIL",
                                      "path": "{root}/big.txt",
                                      "output_mode": "content"}}],
     postcondition={"kind": "shown_contains", "expected": "<persisted-output>"})

task(id="spill-under-threshold-is-inline", domain="size_spill",
     title="a result under the threshold is shown in full",
     reason="a spill layer that fires early turns every ordinary result into a file the "
            "model has to go and read; the threshold has to be a real line.",
     setup=[{"path": "small.txt", "content": _SMALL_FILE}],
     steps=[{"tool": "Grep", "args": {"pattern": "^\\d{6}:|SUBFUZZ-TAIL",
                                      "path": "{root}/small.txt",
                                      "output_mode": "content"}}],
     postcondition={"kind": "shown_contains", "expected": TAIL_SMALL})

task(id="spill-two-calls-last-pointer", domain="size_spill",
     title="two large results in one task keep their own payloads",
     reason="spills are keyed by call id; a collision hands the model the first result as "
            "the answer to the second call, which is a wrong answer that looks complete.",
     setup=[{"path": "big.txt", "content": _BIG_FILE},
            {"path": "second.txt", "content": _SECOND_FILE}],
     steps=[{"tool": "Grep", "args": {"pattern": "^\\d{6}:|SUBFUZZ-TAIL",
                                      "path": "{root}/big.txt",
                                      "output_mode": "content"}},
            {"tool": "Grep", "args": {"pattern": "^\\d{6}:|SUBFUZZ-TAIL",
                                      "path": "{root}/second.txt",
                                      "output_mode": "content"}}],
     postcondition={"kind": "shown_contains", "expected": TAIL_SECOND})

ctl("ctl-spill-payload-clipped", "size_spill", "spill-large-result-resolves",
    "clip_to_prefix",
    title="control: clip the spilled payload to a prefix",
    reason="this is the shape #1052 was about — a length that describes the pointer or the "
           "preview instead of the answer — seen from the writer's side.",)

# ══ binaries ════════════════════════════════════════════════════════════════
task(id="fs-readonly-dir-refused", domain="binaries",
     title="python3 is on PATH and its stdout arrives unmodified",
     reason="whether the binaries a task needs exist, and where, is the primitive the "
            "video says decides success before the model does anything.",
     steps=[{"tool": "Bash", "args": {
         "command": "python3 -c 'print(\"SUBFUZZ-PROBE-7f3c\")'", "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha_s("SUBFUZZ-PROBE-7f3c\n")})

task(id="bin-python-stdout", domain="binaries",
     title="the interpreter the corpus leans on prints a marker to stdout unchanged",
     reason="half the tasks in three domains shell out to python3, so a substrate whose "
            "interpreter mangles or refuses stdout would make them all red at once, and "
            "this row is the one that says which of those it is",
     steps=[{"tool": "Bash", "args": {
         "command": "python3 -c 'print(\"SUBFUZZ-OK\")'"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha_s("SUBFUZZ-OK\n")})

task(id="bin-python-imports-json", domain="binaries",
     title="a stdlib import inside the tool's interpreter works",
     reason="'the downloaded package lands in the module path the kernel imports from' is "
            "the video's example; the stdlib is the cheapest version of that claim.",
     steps=[{"tool": "Bash", "args": {
         "command": "python3 -c 'import json; print(json.dumps({\"k\": 1}))'",
         "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha_s("{\"k\": 1}\n")})

task(id="bin-python-stderr-and-exit-code", domain="binaries",
     title="a program that fails on stderr reports its own text",
     reason="a missing binary is only useful if the error text survives, which is what "
            "makes the failure actionable instead of a stack trace.",
     steps=[{"tool": "Bash", "args": {
         "command": "python3 -c 'import sys; sys.stderr.write(\"bad\\n\"); sys.exit(4)'",
         "cwd": "{root}"}}],
     postcondition={"kind": "exit_shape", "is_error": True})

task(id="bin-missing-binary-error-shape", domain="binaries",
     title="a missing binary answers with a not-found error, not a stack trace",
     reason="node for yt-dlp, pandoc, ffmpeg: this box has failed sessions over exactly "
            "this, and the item's requirement is an error shape a caller can read.",
     steps=[{"tool": "Bash", "args": {"command": "definitely-not-a-real-binary-xyz",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "exit_shape", "is_error": True},
     expected_layers=["host"])

task(id="bin-shell-substitution-works", domain="binaries",
     title="command substitution runs inside the spawned shell",
     reason="the spawn is a real shell with the usual facilities, not an exec of argv; the "
            "digest says the substitution ran rather than being passed through.",
     steps=[{"tool": "Bash", "args": {"command": "echo $(printf ok)",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha_s("ok\n")})

task(id="bin-exec-script-in-run-root", domain="binaries",
     title="a script written into the run root is executable from it",
     reason="'whether binaries sit where the execute step looks' — the run root is the "
            "place arm A puts things, so it has to be on the exec path.",
     setup=[{"path": "run.sh", "content": SCRIPT, "mode": "755"}],
     steps=[{"tool": "Bash", "args": {"command": "./run.sh", "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha_s("script-ran\n")})

ctl("ctl-bin-python-flipped", "binaries", "bin-python-stdout", "flip_one_byte",
    title="control: flip one byte of python's stdout",
    reason="the fault that started the video's corpus was one image coming back "
           "different; a single appended byte is that fault in miniature.",)

# ══ unicode_text ════════════════════════════════════════════════════════════
task(id="uni-emoji-in-stdout", domain="unicode_text",
     title="an emoji written by a child reaches the harness as UTF-8",
     reason="the encoding layer at the boundary is where the video's first fault lived; an "
            "emoji is four bytes that no latin-1 path survives.",
     steps=[{"tool": "Bash", "args": {"command": "printf '\\xf0\\x9f\\x98\\x80\\n'",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha("\U0001F600\n".encode())})

task(id="uni-combining-marks-in-stdout", domain="unicode_text",
     title="a combining sequence is not precomposed on the way through",
     reason="a normalising boundary turns e + combining acute into é: identical on screen, "
            "different bytes, and the digest is the only witness.",
     steps=[{"tool": "Bash", "args": {"command": "printf 'cafe\\xcc\\x81\\n'",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha(COMBINING_TEXT)})

task(id="uni-cjk-and-nbsp", domain="unicode_text",
     title="CJK text and a non-breaking space survive the boundary",
     reason="U+00A0 is the byte a well-meaning whitespace normaliser rewrites, and it "
            "changes the digest while the rendered text looks the same.",
     steps=[{"tool": "Bash", "args": {"command": "printf '\\u65e5\\u672c\\u8a9e "
                                                 "nbsp\\xc2\\xa0X\\n'",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha(CJK_TEXT)})

task(id="uni-utf8-filename-read-back", domain="unicode_text",
     title="a file whose name carries UTF-8 is written and read back under that name",
     reason="the name crosses the boundary before the content does, and a mangled name "
            "lands the bytes somewhere the postcondition never looks.",
     steps=[{"tool": "Bash", "args": {"command": f"printf 'caf\\xc3\\xa9\\n' > './{SPACY}'",
                                      "cwd": "{root}"}},
            {"tool": "Bash", "args": {"command": f"cat './{SPACY}'",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha(b"caf\xc3\xa9\n")})

task(id="uni-nul-in-stdout-survives", domain="unicode_text",
     title="a NUL byte printed to stdout is not silently dropped",
     reason="the probe the corpus exists for: Orellana's null-byte fault is a stdout fault "
            "as often as a file fault, and this box has never measured it.",
     steps=[{"tool": "Bash", "args": {"command": "printf 'a\\0b'", "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha(b"a\x00b")})

ctl("ctl-uni-emoji-clipped", "unicode_text", "uni-emoji-in-stdout", "clip_to_prefix",
    title="control: clip the emoji output to nothing useful",
    reason="halving four bytes of UTF-8 also produces a surrogate, which is how a clip "
           "becomes a decode fault — the same bytes, two different faults, and one digest "
           "that sees either.",)

# ══ resource_limits ═════════════════════════════════════════════════════════
BIG_BYTES = 4 * 1024 * 1024
NOISE_BYTES = 2 * 1024 * 1024
NOISE = noise(NOISE_BYTES, "res-noise")
NOISE_DIGEST = sha(NOISE)
ARG_PAYLOAD = "a" * 200_000

task(id="res-large-file-write-4mb", domain="resource_limits",
     title="a 4 MB file can be written and is the size it claims",
     reason="disk and size ceilings, reported as a length rather than as an absence.",
     steps=[{"tool": "Bash", "args": {
         "command": "python3 -c \"open('big.bin','wb').write(b'x' * 4194304)\"",
         "cwd": "{root}"}}],
     postcondition={"kind": "file_length", "path": "big.bin", "expected": BIG_BYTES})

task(id="res-large-file-digest-round-trip", domain="resource_limits",
     title="a 2 MB generated file hashes to the digest its generator computed",
     reason="size plus fidelity: a truncating layer that only bites on large payloads is "
            "the one kind that never shows up in a hand-written unit test.",
     setup=[{"path": "noise.bin", "gen": "noise", "bytes": NOISE_BYTES}],
     steps=[{"tool": "Bash", "args": {
         "command": "python3 -c \"import hashlib,sys;"
                    "sys.stdout.write(hashlib.sha256(open(sys.argv[1],'rb')"
                    ".read()).hexdigest())\" {root}/noise.bin",
         "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha_s(NOISE_DIGEST)})

task(id="res-oversized-arg-list", domain="resource_limits",
     title="a 200 KB payload inside one command line survives to the child",
     reason="argument-list limits are a substrate ceiling, not a model failure, and the "
            "count in the output is the proof the payload arrived whole.",
     steps=[{"tool": "Bash", "args": {"command": f"printf '%s' '{ARG_PAYLOAD}' | wc -c",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "stdout_sha256", "expected": sha_s(f"{len(ARG_PAYLOAD)}\n")})

task(id="res-memory-ceiling-reported", domain="resource_limits",
     title="hitting a memory limit is reported as a failure",
     reason="the video's worst fault was a sandbox that starved its sessions of memory and "
            "'signaled that as a complete failure'. A ceiling must be visible as itself.",
     steps=[{"tool": "Bash", "args": {
         "command": "python3 -c 'import resource;"
                    "resource.setrlimit(resource.RLIMIT_AS,(6<<30,6<<30));"
                    "bytearray(8<<30)'", "cwd": "{root}"}}],
     postcondition={"kind": "exit_shape", "is_error": True},
     expected_layers=["host"])

task(id="res-tool-output-cap-is-marked", domain="resource_limits",
     title="the tool's own output cap announces itself",
     reason="Bash truncates at 30,000 characters below the harness's spill threshold; the "
            "marking is the difference between a short answer and a truncated one.",
     steps=[{"tool": "Bash", "args": {
         "command": "python3 -c \"print('x' * 40000)\"", "cwd": "{root}"}}],
     postcondition={"kind": "shown_contains", "expected": "[... output truncated ...]"})

ctl("ctl-res-oversized-flipped", "resource_limits", "res-oversized-arg-list",
    "flip_one_byte",
    title="control: append one byte to the command carrying the 200 KB payload",
    reason="a byte added at the end of a pipeline is the fault class that only shows in the "
           "count — the visible half of the answer is unchanged.",)

# ══ refusal_shapes ══════════════════════════════════════════════════════════
#: The three substrings below were read off real answers from this box's own tools
#: (`bash: …: command not found`, `…: Permission denied` from a denied write, and the same
#: `Permission denied` from a mode-500 directory) plus the harness's own refusal line in
#: `agent_mcp/main.py`. They are the error SHAPE the item asks for; pinning a whole message
#: would pin bash's version too.
#:
task(id="rf-missing-binary-error-text", domain="refusal_shapes",
     title="a missing binary's answer says not found",
     reason="the shape, not a stack trace — a caller has to be able to tell 'not "
            "installed' from 'crashed'.",
     steps=[{"tool": "Bash", "args": {"command": "definitely-not-a-real-binary-xyz",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "shown_contains", "expected": "not found"},
     expected_layers=["host"])

#: The path is `/proc/version` and not a name this corpus invented. The row used to write to
#: `/proc/substrate-fuzz-nope`, which does not exist, so the kernel answered ENOENT —
#: `/bin/sh: line 1: /proc/substrate-fuzz-nope: No such file or directory` — and the row's
#: `Permission denied` needle could never appear; the first arm-A run reported it as a host
#: defect. ENOENT arrives before any permission check, so a denied-permission needle needs a
#: target that IS there and IS unwritable: `/proc/version`, mode 444 owned by `nobody`, which
#: uid 1000 cannot open for writing (measured: `bash: line 1: /proc/version: Permission denied`,
#: exit 1, nothing written). The loader refuses the invented-path shape on sight
#: (`corpus.errno_needle_reason` measures the redirect target).
task(id="rf-write-into-proc-error-text", domain="refusal_shapes",
     title="a denied write says permission denied",
     reason="the error text is the only thing that lets a caller attribute the refusal to "
            "the kernel instead of to itself, so the target has to be a file that exists and "
            "refuses — a missing path yields ENOENT and tests nothing about permission.",
     steps=[{"tool": "Bash", "args": {"command": "printf x > /proc/version",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "shown_contains", "expected": "Permission denied"},
     expected_layers=["host", "harness"])

#: The readonly row's needle used to be `Read-only file system`, and it could never have
#: fired: that is EROFS, the message for a read-only MOUNT, while this row's fixture is a
#: mode-500 DIRECTORY, which is EACCES. Measured on this box as uid 1000 —
#: `mkdir -p /tmp/rotestcheck/rodir && chmod 500 $_ && printf x > $_/inside.txt` prints
#: `bash: line 1: …/inside.txt: Permission denied`. Arm A runs inside an ordinary writable
#: tree and cannot stage a read-only mount, so the row now asserts the error its own setup
#: causes, and the loader refuses the old needle outright
#: (`corpus.errno_needle_reason`).
task(id="rf-readonly-dir-error-text", domain="refusal_shapes",
     title="a directory with no write permission says permission denied",
     reason="Orellana's filesystem-permission class: the fault is invisible in a "
            "boolean and obvious in a string. The string has to be the one this setup "
            "causes — a mode-500 directory is EACCES, and the row used to demand EROFS's "
            "text, which no fixture arm A can stage would ever print.",
     setup=[{"path": "rodir", "dir": True, "mode": "500"}],
     steps=[{"tool": "Bash", "args": {"command": "printf x > rodir/inside.txt",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "shown_contains", "expected": "Permission denied"},
     expected_layers=["host"])

task(id="rf-destructive-command-refused", domain="refusal_shapes",
     title="the harness refuses a destructive command before it runs",
     reason="Lloyd's own guard is part of the substrate a task meets, so its refusal has "
            "to be visible in the artifact as `harness` and not absorbed into the model's "
            "own failure.",
     steps=[{"tool": "Bash", "args": {"command": "rm -rf {root}",
                                      "cwd": "{root}"}}],
     postcondition={"kind": "shown_contains", "expected": "harness safety"},
     expected_layers=["harness"])

task(id="rf-read-missing-file-errors", domain="refusal_shapes",
     title="reading a file that is not there is an error result",
     reason="a missing file answered as empty text is how a 0-hit lookup becomes a wrong "
            "answer — the class rule about positive controls exists because of exactly "
            "this ambiguity.",
     steps=[{"tool": "Read", "args": {"file_path": "{root}/no-such-file.txt"}}],
     postcondition={"kind": "exit_shape", "is_error": True},
     expected_layers=["host"])

ctl("ctl-rf-notfound-clipped", "refusal_shapes", "rf-missing-binary-error-text",
    "clip_to_prefix",
    title="control: clip the not-found error to a prefix",
    reason="a clipped error string keeps the command's name and loses the reason, which is "
           "indistinguishable from a tool that answered nothing. The detector has to see "
           "that difference.",)


# ── emit ────────────────────────────────────────────────────────────────────

#: Fixture file name -> the bytes the run root gets. A task that names one of these in a
#: `{"fixture": true}` setup entry carries those same bytes in its own row, so the expected
#: values in it are checkable from the row alone and a reader with `corpus.json` in one hand
#: can recompute every digest in the other.
FIXTURE_BY_PATH = {"png.bin": PNG, "pdf.bin": PDF_B, "nul.bin": NUL_B,
                   "emoji.bin": EMOJI_TEXT, "crlf.bin": CRLF}


def build() -> dict:
    for row in TASKS:
        for item in row.get("setup") or []:
            if isinstance(item.get("content"), str) and not item.get("dir"):
                item["sha256"] = sha(item["content"].encode("utf-8", "surrogateescape"))
        for item in row.get("setup") or []:
            if not item.get("fixture"):
                continue
            f = FIXTURE_BY_PATH[item["path"]]
            # The row's declared setup digest is the one the runner re-checks after it
            # writes, so a fixture the runner mis-materialises fails as a host fault
            # instead of a hundred-byte-long round-trip mismatch.
            item["sha256"] = sha(f)
            row["fixture"] = b64(f)
            row["fixture_sha256"] = sha(f)
    return {"version": 1, "item": 2620,
            "generated_by": "eval/substrate_fuzz/gen_corpus.py — edit the generator, "
                            "not this file",
            "postcondition_kinds": list(POSTCONDITION_KINDS),
            "tasks": TASKS}


def main() -> int:
    from eval.substrate_fuzz.corpus import corpus_summary, load_corpus

    doc = build()
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tasks = load_corpus(OUT)
    real = [t for t in tasks if not t.is_control]
    print(f"{OUT}: {len(real)} real tasks + {len(tasks) - len(real)} controls, "
          f"{len({t.domain for t in real})} domains")
    for domain, counts in corpus_summary(tasks).items():
        print(f"  {domain:<16} {counts['real']} real, {counts['controls']} control")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
