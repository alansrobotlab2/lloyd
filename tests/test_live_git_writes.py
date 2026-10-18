"""The two decisions the #2604 lane cannot delegate: *who* is a person, and *where*
a `git` runs.

`tests/test_safety_gate_discrimination.py` and `tests/test_protected_paths.py`
exercise the lane through `safety.check_bash_command`, which is the right level for
the refuse/allow tables the item's clauses are written as. This file goes underneath
them, at the two helpers whose failure mode is invisible from the outside:

  * `attended_session`, which decides whether a session id is evidence of a human.
    A miss in the permissive direction is a commit on production `main`; a miss in
    the strict direction takes a person's own `git commit` away, and the module's
    whole asymmetry — an *empty* id is attended, an *unrecognised* id is not — reads
    like a typo unless someone has written down which way each case goes and why.
  * `_is_live_root` and the segment walk, which decide whether a command's working
    directory is the production tree. Both fail silently: a prefix comparison calls
    the sibling `lloyd-data` production, and a wrapper the unwrapper does not know
    answers None, which is indistinguishable from "not a write".

Run:
  /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python -m pytest tests/test_live_git_writes.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LLOYD_HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLOYD_HOME))

from app.harness.live_git_writes import (  # noqa: E402
    _is_live_root,
    attended_session,
    check_live_git_writes,
)

#: A tree that is not this box's, so nothing here can pass by accident against
#: the real production checkout — the matcher compares paths, it does not require
#: a repository to exist.
_LIVE = "/srv/prod/lloyd"

_CHAT = "20260928_091452_4f2a9c"
_BG = "20261011_233337_owedcheck_ac94"


# ── Who counts as a person ────────────────────────────────────────────────────

def test_attended_session_needs_positive_evidence_and_says_which_way_each_id_goes():
    """Every arm is a shape that exists on this box, and the two directions are
    asymmetric on purpose: an *empty* id is attended (the desktop terminal guard
    calls `check_bash_command` with no id for text a person is typing), while an id
    that simply carries no evidence of a person — a platform-prefixed worker id, an
    id with no record — is refused. Reversing either one is a bug someone would
    have to discover at runtime, which is what this table is for."""
    cases = {
        # No id is a human at a terminal, not an anonymous automation.
        "": True,
        None: True,
        # The minted web-chat shape (`app/routers/messages.py`): three parts, six
        # hex characters. Positive evidence, and the only shape that is.
        _CHAT: True,
        # A recognised background run: four parts, date and time, a source slug.
        _BG: False,
        "20261012_051633_autotriage_1a2b": False,
        # A surface prefix is a machine route; a colon is never in a minted id.
        "worker:arch-review:abc": False,
        "autonomy-task:40": False,
        # An id with no record and no shape is nobody's evidence of a person.
        "client-9f2c1e": False,
    }
    for sid, expected in cases.items():
        assert attended_session(sid) is expected, repr(sid)


def test_a_session_record_with_a_person_platform_is_the_other_kind_of_evidence(
        tmp_path, monkeypatch):
    """The branch that reads across processes: a client-supplied id with no mint
    gets its answer from the record the *backend* wrote, at
    `~/lloyd-data/sessions/<id>.json`. Nothing else in the suite writes one, so
    without this node the attended-by-record path is code that has never run in a
    test — and it is the one path where a stale cache or a renamed platform field
    would silently hand a background turn its commit back."""
    import app.paths as P

    for sid, platform, expected in (
            ("client-a1", "mission-control", True),   # the web UI
            ("client-a2", "browser", True),           # the Chrome side panel
            ("client-a3", "worker", False),           # a machine surface
            ("client-a4", "autonomy", False),
            ("client-a5", "", False),                 # a record that says nothing
            ("client-a6", None, False)):              # no record at all
        if platform is not None:
            (tmp_path / f"{sid}.json").write_text(
                json.dumps({"id": sid, "platform": platform, "messages": []}))
    monkeypatch.setattr(P, "SESSIONS_DIR", tmp_path)

    for sid, platform, expected in (
            ("client-a1", "mission-control", True), ("client-a2", "browser", True),
            ("client-a3", "worker", False), ("client-a4", "autonomy", False),
            ("client-a5", "", False), ("client-a6", None, False)):
        assert attended_session(sid) is expected, (sid, platform)
        # And the decision is the lane's too, not just the classifier's.
        refused = check_live_git_writes(f"git -C {_LIVE} commit -m x", sid,
                                        live=_LIVE)
        assert (refused is None) is expected, (sid, refused)


def test_a_task_subagent_is_attended_only_through_a_resolved_attended_parent():
    """A `task:` id answers for nothing on its own. With a resolver it inherits the
    parent's surface, one level at a time; without one it is refused, which is the
    opposite of `classify_session`'s `UNKNOWN` handling and deliberate — on this
    axis the guard's own guess is the safer answer, and the registry it would
    resolve through is documented as best-effort."""
    assert attended_session("task:sub-1") is False
    assert attended_session("task:sub-1", parent_of=lambda _s: _CHAT) is True
    assert attended_session("task:sub-1", parent_of=lambda _s: _BG) is False
    assert attended_session("task:sub-1", parent_of=lambda s: s) is False

    def chain(sid: str) -> str:
        return {"task:sub-1": "task:sub-2", "task:sub-2": _CHAT}.get(sid, "")

    assert attended_session("task:sub-1", parent_of=chain) is True


# ── Where a git actually runs ────────────────────────────────────────────────

def test_the_live_tree_is_matched_by_the_path_and_not_by_its_string(tmp_path):
    """Containment, the sibling trap, and symlinks — each in both directions,
    because the same one-character decision (`startswith(root)` vs
    `startswith(root + "/")`) is what makes `~/lloyd-data` either production or a
    scratch tree."""
    assert _is_live_root(_LIVE, _LIVE)
    assert _is_live_root(_LIVE + "/", _LIVE)
    assert _is_live_root(_LIVE + "/app/harness", _LIVE)
    assert _is_live_root(_LIVE + "/../lloyd", _LIVE)
    # The sibling whose name starts with the live path as a string.
    assert not _is_live_root("/srv/prod/lloyd-data", _LIVE)
    assert not _is_live_root("/srv/prod/lloyd-data/scratch", _LIVE)
    assert not _is_live_root("/srv/prod/lloyd-work/SM_x/home/lloyd", _LIVE)
    assert not _is_live_root("/tmp/clone", _LIVE)

    real = tmp_path / "real" / "lloyd"
    real.mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real)
    assert _is_live_root(str(link), str(real))
    assert _is_live_root(str(link / "app"), str(real))


def test_the_walk_refuses_through_the_wrappers_a_turn_actually_uses():
    """Each of these reaches `git` in program position after one or two layers, and
    the corpus this item measured contains the `bash -c` and `cd` forms in
    particular. A miss here answers None — the direction a matcher may not fail in,
    because the reviewer of a refused command sees a refusal and the author of an
    unrefused one sees nothing."""
    for cmd in (
            f"bash -c 'cd {_LIVE} && git commit -m x'",
            f"sh -c 'git -C {_LIVE} reset --hard HEAD~1'",
            f"echo hi && git -C {_LIVE} commit -m x",
            f"if git -C {_LIVE} commit -m x; then true; fi",
            f"env FOO=bar git -C {_LIVE} commit -m x",
            f"timeout -k 5 30 git -C {_LIVE} commit -m x",
            f"nice -n 10 git -C {_LIVE} commit -m x",
            f"docker ps >/dev/null; git -C {_LIVE} commit -m x",
            f"msg=$(git -C {_LIVE} rev-parse HEAD) && git -C {_LIVE} commit -m x",
    ):
        assert check_live_git_writes(cmd, _BG, live=_LIVE), cmd


def test_the_walk_stops_at_the_depth_it_documents():
    """`_MAX_WALK` is a real bound and this pins it rather than letting it drift:
    two nestings of `bash -c` are refused, a third is not. No shape in the corpus
    nests at all, and the bound is what stops a self-referential `-c` payload from
    looping the walk — so the gap is a stated limit, not an accident waiting to be
    read as one."""
    assert check_live_git_writes(f"bash -c 'bash -c \"git -C {_LIVE} commit\"'",
                                 _BG, live=_LIVE)
    assert check_live_git_writes(
        f"bash -c 'bash -c \"bash -c ''git -C {_LIVE} commit''\"'", _BG, live=_LIVE) is None


def test_a_write_verb_inside_a_non_shells_argument_is_data_not_a_command():
    """The precision half of the matcher, and the one that keeps the lane from
    being switched off in a week: a `git commit` inside a `python -c` string, a
    `grep` pattern or a script file is text, and the fleet reads about commits far
    more often than it writes them. A non-shell interpreter's `-c` payload is
    therefore *not* walked — the same limit `check_service_control` states for a
    script that starts a service."""
    for cmd in (
            "python -c \"print('git commit')\"",
            f"python3 -c \"import subprocess; print('cd {_LIVE} && git commit')\"",
            "perl -e 'system(\"git commit\")'",
            f"grep -rn 'git commit' {_LIVE}/app/",
            f"bash {_LIVE}/scripts/whatever.sh",
            f"echo 'git -C {_LIVE} commit -m x' | tee /tmp/note",
            f"docker run -w {_LIVE} alpine git commit -m x",
    ):
        assert check_live_git_writes(cmd, _BG, live=_LIVE) is None, cmd


def test_a_variable_the_command_never_assigns_still_falls_back_to_the_dispatch_cwd():
    """#2801 clause 4: the fail-closed direction survives the fix.

    Substituting an assignment is only safe while *unresolved* stays what it was. A
    name the command never assigns still reaches `_norm` with its `$` intact
    (`os.path.expandvars` finds it in neither the process environment nor the
    command), still gets joined onto the dispatch cwd, and so still refuses when that
    cwd is inside the live tree — which is the answer a `cd $UNSET` gets from a shell
    too, in the only sense that matters here: it does not quietly become nobody's
    directory. The control arm below is what makes that a measurement and not an
    assumption: the identical four commands, same unresolved names, dispatched from a
    scratch directory, are not writes to production.
    """
    for name, cmd in (
            ("cd an unset name", "cd $UNSET && git commit -m x"),
            ("the -C spelling of an unset name", 'git -C "$UNSET" commit -m x'),
            ("add and commit after an unset cd",
             "cd $UNSET && git add f && git commit -m x"),
            # An unrelated assignment earlier in the string must not become an
            # excuse for the name that was never assigned.
            ("an unset name beside an assigned one",
             "W=/tmp/g; cd $UNSET && git commit -m x"),
    ):
        refused = check_live_git_writes(cmd, _BG, cwd=f"{_LIVE}/app/harness",
                                        live=_LIVE)
        assert refused, (name, cmd)
        assert _LIVE in refused, (name, refused)
        assert check_live_git_writes(cmd, _BG, cwd="/tmp/scratch",
                                     live=_LIVE) is None, (name, cmd)


def test_which_assignments_the_lane_believes_is_the_shells_own_rule():
    """Where `_assignments` draws its line, and why each side of it is the shell's
    line rather than the matcher's convenience.

    Every arm is chosen so that the guard's answer agrees with the directory the
    running shell would actually have put the `git` in:

      * A **prefix** assignment (`W=… git add f`) is not believed, and the arm is
        quieter than that: the lane never reads that `git` at all, because the matcher
        is anchored on a program position and `_unwrap` strips assignments only behind a
        wrapper it knows (`env FOO=bar git …`, pinned above). The shell puts a prefix
        assignment in *one command's* environment and expands that command's arguments
        first, so the `$W` in them is empty; believing it would resolve a variable that
        never existed, which is #2801's bug wearing the opposite sign.
      * A value **cut by a command substitution** (`W=/srv/prod/lloyd/$(echo sub)`)
        is not believed either, because the only thing the tokeniser can see is the
        prefix up to `(`, and a prefix is not the directory. `_assignments` drops the
        whole assignment, so the lane falls back to the dispatch cwd exactly as it did
        before the fix. `W=$(mktemp -d)`, the fleet's scratch-clone recipe, is in this
        class — which is what keeps 2,000-odd real commands answering unchanged.
      * `$WD` is not eaten by an assigned `W`: the reference needs a word boundary,
        or the first assignment would resolve its neighbour's name.
      * `${W}` and a value built from an earlier name (`X=…; W=$X`) are believed,
        because a shell believes them.
    """
    scratch = "/tmp/scratch"

    # Prefix assignments: not believed, so the dispatch cwd is the whole story.
    assert check_live_git_writes(f"W={_LIVE} git add f", _BG, cwd=scratch,
                                 live=_LIVE) is None
    assert check_live_git_writes(f"W={_LIVE} cd $W && git add f", _BG, cwd=scratch,
                                 live=_LIVE) is None
    # …and the same tree reached through a *delimiter*, dispatched from inside the live
    # checkout, does refuse. That is the control that keeps the two arms above from
    # being vacuous: the assignment is what stopped being believed, not the segment
    # being unread — the delimiter form is read, followed, and refused.
    assert check_live_git_writes(f"W={_LIVE}; git add f", _BG, cwd=f"{_LIVE}/app",
                                 live=_LIVE)

    # Values cut by a substitution: dropped whole, so they fall back to the cwd.
    assert check_live_git_writes(f"W={_LIVE}/$(echo sub); cd $W && git add f", _BG,
                                 cwd=scratch, live=_LIVE) is None
    assert check_live_git_writes(f"W=$({ _LIVE }); cd $W && git add f", _BG,
                                 cwd=scratch, live=_LIVE) is None

    # The word boundary, and the two spellings a shell follows.
    assert check_live_git_writes(f"W=/tmp/g; WD={_LIVE}; cd $WD && git commit -m x",
                                 _BG, cwd=scratch, live=_LIVE)
    assert check_live_git_writes(f"W={_LIVE}; cd ${{W}} && git add f", _BG,
                                 cwd=scratch, live=_LIVE)
    assert check_live_git_writes(f"X={_LIVE}; W=$X; cd $W && git add f", _BG,
                                 cwd=scratch, live=_LIVE)


# ── #2987: a heredoc body is data, and a substitution is read for what it names ──
#
# 10 of the 13 rows this lane wrote to ~/lloyd-data/safety/denials.jsonl after it
# landed were refusals of commands aimed OUTSIDE the live checkout (the window
# 2026-10-16T13:05:10Z..2026-10-18T22:47:57Z splits 3 correct / 9 heredoc-body / 1
# command-substitution), and 3 of those 13 were owed-check turns refused for writing a
# probe about THIS guard inside a `<<'PY'` body. Both arms below are the lane reading
# text that the shell never executes as if it were shell, so the pair is pinned in one
# section: the stripping that fixes the first must not reach into the second.


def test_a_heredoc_body_for_a_shell_is_still_walked_as_shell_code():
    """Clause 2. `bash <<'EOF'` hands the body to the shell, so the body IS shell.

    `_split_heredocs` keeps a shell recipient's body in the stream it returns, and a
    fix that stripped bodies wholesale would have started missing this: the shell
    executes `cd <live> && git commit -m x` the moment the heredoc is fed to it, which
    is exactly the write the lane exists to refuse. The arm that makes this a
    measurement of the SPLIT and not of the whole lane is the non-shell twin below it
    in tests/test_safety_gate_discrimination.py, whose identical body line is program
    text for `cat` and answers None — one byte of difference, the recipient, and the
    verdict has to move.
    """
    refused = check_live_git_writes(
        "bash <<'EOF'\n" + f"cd {_LIVE} && git commit -m x\n" + "EOF", _BG,
        cwd=_LIVE, live=_LIVE)
    assert refused, "a shell-recipient heredoc body was dropped as data"
    assert refused.startswith("`git commit` in the live checkout"), refused[:120]
    # An unquoted delimiter with expansion inside it is still a shell body.
    assert check_live_git_writes(
        "sh <<EOF\n" + f"git -C {_LIVE} reset --hard\n" + "EOF", _BG,
        cwd="/tmp/scratch", live=_LIVE)


def test_a_variable_assigned_only_inside_a_stripped_body_is_not_believed():
    """Clause 4: the assignment map comes from the SHELL text, not the raw command.

    `_assignments` used to run on the raw string, so a `W=<live>` line that exists only
    as data inside a python heredoc body was believed as an assignment, and the real
    shell statement after the body (`cd $W && git add -A`, dispatched from a scratch
    directory) resolved to the live tree. The shell never ran that line — the file was
    written, never executed — so the target was invented from program text, and the
    dispatch cwd, which is the only real fact about where the `git add` would have
    happened, was overridden by it.

    The arm that keeps this from being an over-fix is that a real assignment before the
    heredoc is still believed: strip the body, keep every assignment the SHELL made.
    """
    body = (f"cat > /tmp/p2987.py <<'PY'\nW={_LIVE}\nPY\n"
            "cd $W && git add -A")
    assert check_live_git_writes(body, _BG, cwd="/tmp/scratch", live=_LIVE) is None, (
        "an assignment that exists only inside a stripped heredoc body still "
        "retargeted the git write onto the live tree")
    # Positive control on the same shape: the same assignment, made by the shell
    # OUTSIDE any body, is still believed — stripping a body must not un-teach the
    # lane what an assignment is.
    assert check_live_git_writes(
        f"W={_LIVE}\n" + body.splitlines()[-1], _BG, cwd="/tmp/scratch",
        live=_LIVE), "the shell's own assignment stopped being believed"
    # And the same file, written by a heredoc, being executed IS judged: the body
    # reaching python through an interpreter is a payload, and the payload is not
    # walked, but nothing about the file's contents can retarget the next statement.
    assert check_live_git_writes(
        f"cat > /tmp/p2987.py <<'PY'\nW={_LIVE}\nPY\n"
        f"cd {_LIVE} && git add -A", _BG, cwd="/tmp/scratch",
        live=_LIVE), "the operator line after a body stopped being judged"


def test_a_command_substitution_is_outside_only_when_it_cannot_name_the_live_tree():
    """Clause 5: the scratch-clone recipe, and the two arms that must not come with it.

    `d=$(mktemp -d /tmp/ci-XXXX); cd "$d"; git init; git add -A` is the fleet's own
    recipe for a scratch clone — the same one `tests/test_live_git_writes.py`'s header
    and `_assignments`' own comment name as the reason command substitutions were never
    resolved. Dispatched from the live checkout it refused, because an unresolved `$d`
    is joined onto the dispatch cwd and the cwd was live.

    The rule that fixes it without becoming a bypass is that a substitution counts as
    an outside location ONLY WHEN its own text names at least one absolute path and
    none of the paths it names is inside the live tree. `mktemp -d /tmp/ci-XXXX` names
    `/tmp/ci-XXXX`: outside, believed. `$(echo <live>)` names the live tree: not
    believed, and the fall-back to the dispatch cwd stands, which is what keeps a
    substitution from being a way to smuggle a live path past the lane — measured at the
    base commit, that arm refused, and it has to refuse for the same reason afterwards.
    A name the command NEVER assigns is untouched by all of this and still falls back
    (#2801 clause 4), which the third arm re-pins in this clause's own words.
    """
    recipe = ('d=$(mktemp -d /tmp/ci-XXXX); cd "$d"; git init -q -b main .; '
              "git add -A")
    assert check_live_git_writes(recipe, _BG, cwd=_LIVE, live=_LIVE) is None, (
        "the scratch-clone recipe was refused from a live dispatch cwd even though "
        "its own text names only /tmp")
    # The same recipe with the reference UNQUOTED is the spelling the journalled row
    # actually used, and `_assignments` stores the value raw, so the map lookup — not
    # the quoting of `$d` — is what resolves it.
    assert check_live_git_writes(
        'd=$(mktemp -d /tmp/ci-XXXX); cd $d; git init -q -b main .; git add -A',
        _BG, cwd=_LIVE, live=_LIVE) is None, (
        "the same recipe with an unquoted $d was refused")
    # The substitution that DOES name the live tree: unresolved, so the dispatch cwd
    # decides, and the dispatch cwd is live.
    smuggled = check_live_git_writes(
        f"W=$(echo {_LIVE}); cd \"$W\" && git add -A", _BG, cwd=_LIVE, live=_LIVE)
    assert smuggled, "a substitution naming the live tree stopped falling back"
    assert _LIVE in smuggled, smuggled[:160]
    # A substitution naming no path at all is not believed either — `$(pwd)` and
    # `$(git rev-parse --show-toplevel)` are named at the repo root and answer the
    # ROOT, and widening to them is an owed ruling on post-landing traffic, not a
    # judgement this diff gets to make.
    assert check_live_git_writes(
        "W=$(pwd); cd \"$W\" && git add -A", _BG, cwd=_LIVE, live=_LIVE), (
        "a substitution naming no path started being believed as outside")
    # …and the same is true of the BARE `mktemp -d`, which names no template and so
    # gives the lane nothing to read. #2801 pinned this command's twin from a scratch
    # cwd (an unresolved name there falls back to a scratch cwd, so it is not a write
    # to production); this is its fail-closed half from a live one.
    assert check_live_git_writes(
        'W=$(mktemp -d); cd "$W" && git add -A', _BG, cwd=_LIVE, live=_LIVE), (
        "a substitution naming no path at all was believed as outside")
    # A name the command never assigns: unchanged, and still fail-closed.
    unassigned = check_live_git_writes("cd $D_2987 && git add -A", _BG, cwd=_LIVE,
                                       live=_LIVE)
    assert unassigned and _LIVE in unassigned, unassigned
    assert check_live_git_writes("cd $D_2987 && git add -A", _BG, cwd="/tmp/scratch",
                                 live=_LIVE) is None


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            import inspect

            if inspect.signature(_fn).parameters:
                continue          # needs a fixture: run it under pytest
            _fn()
            print(f"ok {_name}")
