"""Clause 4 (#581): `prompt-diff A B` names the differing component, in position order.

The CLI is a separate process that reads the store the writer thread fills, so the
test drives it with `subprocess` against a real store rather than calling
`format_diff` in-process — the bug class it would miss otherwise is the one where
the module formats fine and the CLI's own argument handling or store lookup does
not.

Two synthetic pairs do the naming work the item was filed for: a request pair that
differs only in the tools array, and one that differs only in a single message. Each
has to report exactly that one component and nothing else — a diff that answers
"something moved" is the same answer `cached_tokens` already gives, at more cost.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import component_manifest as cm  # noqa: E402

CLI = ROOT / "scripts" / "meta_review" / "prompt_diff.py"
TOOLS_A = [{"type": "function",
            "function": {"name": "Bash", "description": "Run a shell command",
                         "parameters": {"type": "object"}}}]
TOOLS_B = [{"type": "function",
            "function": {"name": "Bash", "description": "Run a shell command now",
                         "parameters": {"type": "object"}}}]


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    monkeypatch.setenv("LLOYD_MANIFEST_STORE", str(tmp_path / "manifest-store"))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    cm.reset_stats()
    cm._reset_registry()
    cm._reset_template_cache()
    cm._reset_tools_memo()
    yield
    cm.flush(timeout=5.0)


def _record(*, tools=None, user_text="go", components=None,
            session_id="diff-session", extra_messages=(),
            iteration=1) -> str:
    """One manifest line from the real writer, with the parts a test chooses."""
    cm._reset_tools_memo()
    if components is not None:
        cm.note_components(session_id, components)
    messages = [{"role": "user", "content": user_text}, *extra_messages]
    line = cm.record_request(
        base_url="http://127.0.0.1:8096", model="primary", session_id=session_id,
        iteration=iteration, send_site="tests/test_prompt_diff.py",
        payload={"model": "primary", "temperature": 0.7, "messages": messages,
                 **({"tools": tools} if tools is not None else {})})
    assert line, "the writer produced nothing, so the diff has no input"
    cm.flush(timeout=8.0)
    return line["request_id"]


def _changed(proc: subprocess.CompletedProcess) -> "list[str]":
    """The differing-position lines, leading alignment stripped.

    A helper rather than a per-test substring because a status word also appears
    inside the `moved:` detail (`Bash changed`), and a test that counted that as a
    differing position would pass on a diff that reported the change twice.
    """
    out = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if parts and parts[0] in ("changed", "added", "removed"):
            out.append(parts[1] if len(parts) > 1 else "")
    return out


def _run(*args: str) -> subprocess.CompletedProcess:
    """The CLI as a child process, against the store this test owns."""
    return subprocess.run([sys.executable, str(CLI), *args],
                          capture_output=True, text=True, timeout=90, cwd=str(ROOT))


def _pair(left: dict, right: dict) -> tuple[str, str]:
    """Two request ids, each recorded from the parts its own dict names.

    Both calls pass explicit dicts rather than a set of `_b`-suffixed overrides: a
    pair whose difference is a naming convention is a pair whose difference a
    reader has to reconstruct, and the whole clause is about being legible.
    """
    return _record(**left), _record(**right)


def test_a_pair_differing_only_in_the_tools_array_names_only_tools():
    """The clause's first half, exactly: one component differs, one is reported.

    Same message, same components, same engine — only a tool's description text
    moved. The array digest has to carry the whole change and say which definition
    did it, because "did the model get a different tool?" is the question.
    """
    same = {"components": {"SOUL.md": "identity", "memories": "facts"}}
    left, right = _pair({**same, "tools": TOOLS_A}, {**same, "tools": TOOLS_B})
    proc = _run(left, right)
    assert proc.returncode == 0, proc.stderr
    changed = _changed(proc)
    assert len(changed) == 1, proc.stdout
    assert changed[0].startswith("tools:"), proc.stdout
    assert "moved: Bash changed" in proc.stdout, proc.stdout
    # Everything that did not move stays unmentioned — the diff's value is that it
    # does not list the parts that were the same.
    for untouched in ("SOUL.md", "memories", "message[0]"):
        assert untouched not in proc.stdout, proc.stdout


def test_a_pair_differing_only_in_one_message_names_only_that_message():
    """The clause's second half: "was it just my message?" answers with a position.

    Two messages, the second one changed. Only `message[1]` may appear — and the
    index has to be the changed one, since the whole value of position order is
    pointing at the message that moved rather than at the request as a whole.
    """
    left, right = _pair(
        {"user_text": "same opener",
         "extra_messages": [{"role": "assistant", "content": "before"}]},
        {"user_text": "same opener",
         "extra_messages": [{"role": "assistant", "content": "after"}]})
    proc = _run(left, right)
    assert proc.returncode == 0, proc.stderr
    changed = _changed(proc)
    assert changed[0].startswith("message[1]:"), proc.stdout
    assert len(changed) == 1, proc.stdout
    assert "message[0]" not in proc.stdout, proc.stdout


def test_differences_are_printed_in_position_order():
    """Two components move and the print order is render order, not dict order.

    Tools render ahead of the prompt components in Qwen's chat template, so the
    first line must be `tools` and the second `SOUL.md` — the same claim
    `component_positions` documents, proved through the file and the CLI rather
    than in memory.
    """
    left, right = _pair(
        {"tools": TOOLS_A,
         "components": {"SOUL.md": "identity v1", "memories": "facts"}},
        {"tools": TOOLS_B,
         "components": {"SOUL.md": "identity v2", "memories": "facts"}})
    proc = _run(left, right)
    assert proc.returncode == 0, proc.stderr
    order = [ln.split()[0].rstrip(":") for ln in _changed(proc)]
    assert order == ["tools", "system.SOUL.md"], proc.stdout


def test_identical_requests_say_so_and_a_missing_id_exits_nonzero():
    """The two ways this CLI can be pointed at nothing, told apart.

    Two requests that hash the same is a real and useful answer ("the harness sent
    the same thing twice"), so it prints rather than erroring. An id that is not in
    the store is a mistake by whoever typed the command, and has to fail loudly:
    a diff that silently prints nothing is indistinguishable from a clean pair,
    which is exactly the reading that would hide a prefix break.
    """
    left, right = _pair({"tools": TOOLS_A, "components": {"SOUL.md": "x"}},
                        {"tools": TOOLS_A, "components": {"SOUL.md": "x"}})
    proc = _run(left, right)
    assert proc.returncode == 0, proc.stderr
    assert "no component differs" in proc.stdout, proc.stdout

    missing = _run(left, "r-00000000000000-00000000")
    assert missing.returncode != 0, missing.stdout
    assert "no manifest line for" in (missing.stderr + missing.stdout)


def test_the_json_output_carries_the_same_positions_in_the_same_order():
    """`--json` is what another tool reads, so it agrees with the printed lines.

    The machine-readable form must report the same positions in the same order and
    keep the moved-definition names, or a script consuming it loses the detail the
    human output has — and a future reporting surface is exactly where a dropped
    field goes unnoticed.
    """
    left, right = _pair({"tools": TOOLS_A, "components": {"SOUL.md": "identity v1"}},
                        {"tools": TOOLS_B, "components": {"SOUL.md": "identity v2"}})
    proc = _run("--json", left, right)
    assert proc.returncode == 0, proc.stderr
    rows = json.loads(proc.stdout)
    assert [row["position"] for row in rows] == ["tools", "system.SOUL.md"], rows
    assert rows[0]["moved"] == ["Bash changed"], rows[0]
    assert rows[0]["a"] != rows[0]["b"], rows[0]


SESSION = "20260919_120000_seampair"


def _record_turn(*, iteration: int, tools=TOOLS_A, user_text="go") -> str:
    """One iteration of one session: same components, caller's iteration and tools."""
    return _record(session_id=SESSION, iteration=iteration, tools=tools,
                   user_text=user_text, components={"SOUL.md": "identity"})


def _headers(stdout: str) -> "list[str]":
    """The pair header lines, one per diff printed — the denominator of every count below."""
    return [ln for ln in stdout.splitlines() if "(iteration " in ln and " vs " in ln]


def _changed_lines(text: str) -> "list[str]":
    """Differing-position lines of one diff block, status column normalised.

    The CLI right-pads the status to seven characters, so a literal
    `"changed tools:"` never appears in its output; `_changed` above already folds
    that padding for the two-id tests, and this is the same fold for a block
    extracted from `--session` output, where the status word is the thing being
    compared.
    """
    return [" ".join(ln.split()) for ln in text.splitlines()
            if ln.strip().startswith(("changed", "added", "removed"))]


def test_session_mode_diffs_consecutive_requests_oldest_first():
    """`--session <id>`, the mode the module docstring names for a prefix cliff.

    Three recorded iterations whose tools array moves between the second and the
    third: the CLI must print one diff per consecutive pair, oldest pair first, the
    first saying nothing differs and the second naming `tools` as the position that
    did. That is the whole advertised move — locate the iteration where
    `prefix_misses` jumps and read the pair that straddles it — and without this
    test the mode a reader is pointed at is the one mode nothing exercises.
    """
    _record_turn(iteration=1)
    _record_turn(iteration=2)
    _record_turn(iteration=3, tools=TOOLS_B)
    # A second session recorded into the same store, with a component of its own
    # left out of every pair: the control that the output below is this session's
    # pairs and not whatever happens to be in the store.
    _record(session_id="20260919_120000_otherchat", iteration=9,
            tools=TOOLS_A, components={"SOUL.md": "someone else's identity"})

    proc = _run("--session", SESSION)
    assert proc.returncode == 0, proc.stderr
    headers = _headers(proc.stdout)
    assert len(headers) == 2, proc.stdout
    assert "(iteration 1, " in headers[0] and "(iteration 2, " in headers[0], headers
    assert "(iteration 2, " in headers[1] and "(iteration 3, " in headers[1], headers
    # Oldest pair first: the pairs are sorted by iteration, not by file order, and
    # reversing them would point a reader at the wrong boundary.
    assert "iteration 1" in proc.stdout.split("iteration 3")[0], proc.stdout
    first, second = proc.stdout.split(headers[1])
    assert "no component differs" in first, first
    changed = _changed_lines(second)
    assert len(changed) == 1, second
    assert changed[0].startswith("changed tools:"), second
    # The other session stayed out of it.
    assert "someone else" not in proc.stdout, proc.stdout


def test_session_last_prints_only_the_final_pair():
    """`--last 1` is how a long turn is read: the boundary, not the whole turn.

    Four recorded iterations make three pairs; `--last 1` must print exactly the
    last one — the pair straddling the break — and dropping the cap would bury it
    under three diffs of nothing.
    """
    for iteration in (1, 2, 3):
        _record_turn(iteration=iteration)
    _record_turn(iteration=4, tools=TOOLS_B)

    proc = _run("--session", SESSION, "--last", "1")
    assert proc.returncode == 0, proc.stderr
    headers = _headers(proc.stdout)
    assert len(headers) == 1, proc.stdout
    assert "(iteration 3, " in headers[0] and "(iteration 4, " in headers[0], headers
    assert _changed_lines(proc.stdout) and _changed_lines(proc.stdout)[0].startswith(
        "changed tools:"), proc.stdout
    assert "no component differs" not in proc.stdout, proc.stdout


def test_session_mode_with_fewer_than_two_records_exits_nonzero():
    """A session with one request has no pair, and must not print an empty success.

    Same rule as a missing request id: nothing compared is not the same as nothing
    changed, and exit 0 here would let a script reading this mode conclude that a
    turn's requests were identical.
    """
    _record_turn(iteration=1)
    proc = _run("--session", SESSION)
    assert proc.returncode != 0, proc.stdout
    assert "no manifest pair" in (proc.stderr + proc.stdout), proc.stdout
