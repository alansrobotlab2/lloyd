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
    """#2987 clause 5: the scratch-clone recipe, and the arms that must not come with it.

    #3244 widened the producer test this node pins (its clause 1 pins the bare
    `mktemp -d` here and its clause 4 pins the arms that stay refused), so every
    sentence below says which item's clause it is.

    `d=$(mktemp -d /tmp/ci-XXXX); cd "$d"; git init; git add -A` is the fleet's own
    recipe for a scratch clone — the same one `tests/test_live_git_writes.py`'s header
    and `_assignments`' own comment name as the reason command substitutions were never
    resolved. Dispatched from the live checkout it refused, because an unresolved `$d`
    is joined onto the dispatch cwd and the cwd was live.

    The rule that fixes it without becoming a bypass has two halves, and both read the
    substitution's WHOLE TEXT. A text naming any path inside the live tree is never
    believed, whatever its producer; a text naming at least one absolute path and none
    of them live is believed (#2987); and since #3244 a text naming no path at all is
    believed when its producer is `mktemp`. `mktemp -d /tmp/ci-XXXX` names
    `/tmp/ci-XXXX`: outside, believed. `$(echo <live>)` names the live tree: not
    believed, and the fall-back to the dispatch cwd stands, which is what keeps a
    substitution from being a way to smuggle a live path past the lane — measured at the
    base commit, that arm refused, and it has to refuse for the same reason afterwards.
    A name the command NEVER assigns is untouched by all of this and still falls back
    (#2801 clause 4), which the last arm re-pins in this clause's own words.
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
    # A substitution naming no path at all is believed ONLY when its producer is
    # `mktemp`, and the arms that are not mktemp are the point of pinning the pair.
    # `$(pwd)` and `$(git rev-parse --show-toplevel)` answer the repo root — the live
    # tree from a live dispatch cwd — so they stay unresolved. Widening belief to them
    # is #2987's owed entry 2, a ruling on post-landing traffic that #3244 deliberately
    # did not make: only the `mktemp` head was ruled on.
    assert check_live_git_writes(
        "W=$(pwd); cd \"$W\" && git add -A", _BG, cwd=_LIVE, live=_LIVE), (
        "a non-mktemp substitution naming no path started being believed as outside")
    # The same two producers in the INLINE form, which is the arm that discriminates the
    # new `_retarget_substituted_cd` rather than `_assignments`: with no assignment to
    # read, the only thing standing between `cd $(pwd)` and a None verdict is the
    # producer test applied to the `cd`'s own argument, so deleting that check would let
    # all four of these through. `_MKTEMP_PRODUCER` is what says no, and these are its
    # proof — the mktemp inline arms above clear on the same call site.
    for inline in ("cd $(pwd) && git add -A",
                   "cd $(git rev-parse --show-toplevel) && git add -A",
                   'cd "$(pwd)" && git add -A',
                   'cd "$(git rev-parse --show-toplevel)" && git add -A'):
        refused_inline = check_live_git_writes(inline, _BG, cwd=_LIVE, live=_LIVE)
        assert refused_inline, f"a non-mktemp inline cd was retargeted: {inline}"
        assert _LIVE in refused_inline, refused_inline[:160]
    # …and the BARE `mktemp -d` now answers None where #2987 pinned a refusal. That
    # assertion is REPLACED here, not kept beside it (#3244 clause 1): the command is
    # the fleet's own CI recipe with its template dropped, and `mktemp -d` creates a
    # fresh directory under `TMPDIR` or `/tmp` — it cannot hand back the checkout, so
    # the refusal was only ever the lane reading "I cannot parse this" as "this is
    # live". #2801's twin from a scratch cwd is untouched: an unresolved name there was
    # never a write to production either.
    assert check_live_git_writes(
        'W=$(mktemp -d); cd "$W" && git add -A', _BG, cwd=_LIVE, live=_LIVE) is None, (
        "a substitution whose producer is mktemp was still refused from a live cwd")
    # The fail-open arm a producer test opens, and that closes it: `mktemp` is the
    # head, but the TEXT names the live tree, so the text test runs first and the
    # producer test never gets to believe it. Both spellings, because `TMPDIR` is how a
    # turn redirects a scratch dir and a template is how it names one.
    for smuggled_head in (
            f'W=$(TMPDIR={_LIVE} mktemp -d); cd "$W" && git add -A',
            f'W=$(mktemp -d {_LIVE}/x.XXXX); cd "$W" && git add -A'):
        refused = check_live_git_writes(smuggled_head, _BG, cwd=_LIVE, live=_LIVE)
        assert refused, f"a mktemp naming live was believed: {smuggled_head}"
        assert _LIVE in refused, refused[:160]
    # A mktemp redirected to an OUTSIDE TMPDIR is believed, which is what makes the two
    # arms above a measurement of the live-path test and not of the producer test.
    assert check_live_git_writes('W=$(TMPDIR=/tmp mktemp -d); cd "$W" && git add -A',
                                 _BG, cwd=_LIVE, live=_LIVE) is None
    # …and the arms the exact match exists for, which a name-anything-after-`mktemp`
    # test would have believed: `mktemp` honours a RELATIVE name by creating it in the
    # current directory, so each of these answers a directory INSIDE the live tree from a
    # live dispatch cwd while naming no absolute path for the live-path test to catch.
    for relative in ('W=$(mktemp -d -p .); cd "$W" && git add -A',
                     'W=$(mktemp -d x.XXXX); cd "$W" && git add -A',
                     'W=$(TMPDIR=. mktemp -d); cd "$W" && git add -A',
                     "cd $(mktemp -d -p .) && git add -A"):
        refused_relative = check_live_git_writes(relative, _BG, cwd=_LIVE, live=_LIVE)
        assert refused_relative, f"a relative mktemp target was believed: {relative}"
        assert _LIVE in refused_relative, refused_relative[:160]
    # A name the command never assigns: unchanged, and still fail-closed.
    unassigned = check_live_git_writes("cd $D_2987 && git add -A", _BG, cwd=_LIVE,
                                       live=_LIVE)
    assert unassigned and _LIVE in unassigned, unassigned
    assert check_live_git_writes("cd $D_2987 && git add -A", _BG, cwd="/tmp/scratch",
                                 live=_LIVE) is None


# ── #3244: a target the command states, in the two shapes #2987 left unread ────────
#
# 5 of the 6 refusals this lane journalled between `d43c2072` (2026-10-18T23:40Z) and
# 2026-10-24 pointed OUTSIDE the live checkout, and none of the five was a shape the
# lane could read: 3 named their repository with `--git-dir=`/`--work-tree=`, 1 assigned
# a bare `R=$(mktemp -d)`, and 1 wrote `cd $(mktemp -d)` with no assignment at all. One
# of the five was a review rung's own probe (`set -e; R=$(mktemp -d); cd $R; git init
# -q -b main; …`), which is how a round grading a diff about this guard became the
# guard's own false positive. All three shapes say where the write goes; all three fell
# back to the dispatch cwd, which for an unattended round IS the live tree. The arms
# below are the item's, in the item's spellings, at cwd = the live checkout.


def test_a_scratch_directory_the_command_makes_for_itself_is_not_the_dispatch_cwd():
    """#3244 clauses 1 and 2: one /tmp directory, three spellings, one verdict.

    The assigned form (`T=$(mktemp -d); cd $T; git add -A`) and the bare-`set -e` form
    the review rung ran are resolved by `_assignments` through the producer test; the
    INLINE form (`cd $(mktemp -d) && git init -q r && cd r && git add -A`) has no
    assignment to find, and is retargeted by `_retarget_substituted_cd` because the
    tokeniser hands `_walk` a `cd` with no argument at all — `$(…)` is a boundary to it.

    The control that makes each arm a measurement and not a stub is the `mktemp -d
    /tmp/ci-XXXX` recipe from #2987, which was already None and has to stay None, and
    the two refusals below: a rewrite that retargeted everything would clear the third
    arm of each pair as well, and those must still refuse.
    """
    # Clause 1, the two commands the item names, plus the quoted reference.
    for cmd in (
            'T=$(mktemp -d); cd $T; git add -A',
            'T=$(mktemp -d); cd "$T"; git add -A',
            "set -e; R=$(mktemp -d); cd $R; git init -q -b main; git add -A",
            # #2987's recipe, unchanged: the template-named twin of the arms above.
            'd=$(mktemp -d /tmp/ci-XXXX); cd "$d"; git init -q -b main .; git add -A',
    ):
        assert check_live_git_writes(cmd, _BG, cwd=_LIVE, live=_LIVE) is None, cmd
    # Clause 2, the non-assignment form, unquoted and quoted, and the `cd`-then-commit
    # shape the same recipe reaches for.
    for cmd in (
            "cd $(mktemp -d) && git init -q r && cd r && git add -A",
            'cd "$(mktemp -d)" && git init -q r && cd r && git add -A',
            "cd $(mktemp -d) && git commit -q -m x",
    ):
        assert check_live_git_writes(cmd, _BG, cwd=_LIVE, live=_LIVE) is None, cmd
    # A `cd` with no argument at all still falls back to the dispatch cwd. `cd ;
    # mktemp -d && git add -A` tokenises to the SAME segments as the unquoted arm above
    # (`cd`, then `mktemp -d`, then the git), so this pins that the retargeting works off
    # the raw text and does not fire when the `cd` has no argument: the shell goes to
    # `$HOME` and the write stays where it was dispatched. That is all it pins — the
    # command carries no substitution, so `_CD_SUBST_ARG` never matches it whatever the
    # producer test does. The arm that discriminates the producer test on a `cd`'s own
    # argument is `cd $(pwd)`, pinned in
    # `test_a_command_substitution_is_outside_only_when_it_cannot_name_the_live_tree`.
    fallback = check_live_git_writes("cd ; mktemp -d && git add -A", _BG, cwd=_LIVE,
                                     live=_LIVE)
    assert fallback, "a cd with no argument stopped falling back to the dispatch cwd"
    assert _LIVE in fallback, fallback[:160]
    # And retargeting the `cd` does not launder a live target that arrives later in the
    # same command by its own explicit `-C`.
    assert check_live_git_writes(f"cd $(mktemp -d) && git -C {_LIVE} add -A", _BG,
                                 cwd=_LIVE, live=_LIVE), (
        "a believed cd hid a live -C that followed it")


def test_the_git_dir_and_work_tree_the_command_names_are_its_write_target():
    """#3244 clauses 3 and 4: the flags that say where, read for their values.

    `--git-dir` and `--work-tree` sit in `_GIT_VALUE_FLAGS`, so the walk used to skip
    each with its value and answer the dispatch cwd. That shape — a write aimed wholly
    at /tmp — is 3 of the 5 false refusals in the 5.5-day window to 2026-10-24, one of
    them with both values held in believed `mktemp` variables, so even #2987's
    substitution rule could not save it: the lane never read the flags to put the values
    anywhere.

    The refusals below are not the lane failing to read the flags: each command states a
    tree or a repository inside the checkout, and the lane refuses the mutating family on
    the target the command gives it. Measured on this box, the two flags write different
    things depending on the subcommand they are handed — `git --git-dir=<repo>/.git
    --work-tree=<T> add -A` leaves `<T>`'s file mtimes alone and writes `<repo>`'s index,
    while the same flags with `reset --hard HEAD` rewrite `<T>`'s tracked files outright;
    so a live `--work-tree` is a live write for part of the family and a live read for the
    rest, which is why the refusal is on the stated tree and not re-derived per
    subcommand. The rest, checked the same way: `--git-dir=<live>/.git` names the live
    index and refs whatever the work tree says; `--git-dir=.git` is that arm spelled
    relatively against a live cwd; `--git-dir=/tmp/r.git` with NO `--work-tree` leaves the
    work tree where the process stands, which is the live checkout; and
    `--work-tree=/tmp/w` with no `--git-dir` leaves the REPOSITORY where it stands —
    `git --work-tree=/tmp/w add -A` from inside a checkout rewrote that checkout's own
    index.
    """
    for cmd in (
            "git --git-dir=/tmp/r.git --work-tree=/tmp/w add -A",
            'R=$(mktemp -d /tmp/a.XXXX); W=$(mktemp -d /tmp/b.XXXX); '
            'git --git-dir="$R" --work-tree="$W" add -A',
            "R=$(mktemp -d); git --git-dir=$R --work-tree=/tmp/w add -A",
    ):
        assert check_live_git_writes(cmd, _BG, cwd=_LIVE, live=_LIVE) is None, cmd
    for cmd in (
            f"git --git-dir=/tmp/r.git --work-tree={_LIVE} add -A",
            f"git --git-dir={_LIVE}/.git --work-tree=/tmp/w add -A",
            "git --git-dir=.git --work-tree=/tmp/w add -A",
            "git --git-dir=/tmp/r.git add -A",
            "git --work-tree=/tmp/w add -A",
    ):
        refused = check_live_git_writes(cmd, _BG, cwd=_LIVE, live=_LIVE)
        assert refused, f"a flag-named live write was let through: {cmd}"
        assert _LIVE in refused, refused[:160]
    # The same outside flags from a scratch dispatch cwd are nobody's live write — the
    # control that keeps the two None arms above from being the dispatch cwd agreeing
    # by accident.
    assert check_live_git_writes("git --git-dir=/tmp/r.git --work-tree=/tmp/w add -A",
                                 _BG, cwd="/tmp/scratch", live=_LIVE) is None


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            import inspect

            if inspect.signature(_fn).parameters:
                continue          # needs a fixture: run it under pytest
            _fn()
            print(f"ok {_name}")
