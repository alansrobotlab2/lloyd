"""Read-only reader for the committed gate-shape skip inventories (#2637).

Two files, both written by the runner this module documents at the head of a
round whose diff converts live-state-conditional skips into two-branch nodes:

    tests/fixtures/skip_inventory_before_20261013.jsonl
    tests/fixtures/skip_inventory_after_20261013.jsonl

Each line is one `SKIPPED [n] path:line: reason` row folded to
`{"path", "reason", "n"}` — the three fields the three claims in #2637 need,
and nothing else. The reason text is kept verbatim because clause 2's claim is
about the NAME of a machine fact, and a normalised reason could hide a rename.

Why a committed artifact and not a live suite run: the rung already runs the
whole suite, and a node that re-ran it would double the round. What a suite run
cannot do is preserve the BEFORE state — by the time the after-inventory exists,
the tree it must be compared against is gone. So the two inventories are the
evidence, committed, and these nodes read them.

The runner that produced them is `/tmp/inv2637.py` at round time, reconstructed
here because the file itself was scratch and the claim has to be re-runnable:
build the tests rung's OWN environment through
`scripts/automod/gate.py::Gate._child_env(isolate_home=True)` — which is what
decides the round home, `LLOYD_DATA`, `TMPDIR` and the mark expression — then
run, from the round's worktree:

    <venv>/bin/python -m pytest -q -m "not live_vault and not fault_injection" \\
        -n 8 --dist loadfile -rs

and fold each summary line. Two details that cost a run each to learn: the round
id argument must be the REAL round id, because `ensure_round_home` refuses a
worktree that does not exist and silently falls back to the real home, which is
not gate shape and counts skips against a machine instead of a round; and the
interpreter must be the live venv, because the round worktree has no `.venvs`.
"""

from __future__ import annotations

import ast
import json
import re
from functools import lru_cache
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
BEFORE = FIXTURES / "skip_inventory_before_20261013.jsonl"

#: Read out of the artifact by `read_inventory`, and pinned there so the
#: evidence cannot be edited to flatter a later delta. 40 skip instances on the
#: base tree against a ceiling of 40: the round that measured this was one node
#: away from refused, which is the whole of #2637.
BEFORE_TOTAL = 40
#: 21, not the 16 the broken classifier first reported: `turn-1-mini model not
#: present` x4 and `streaming model not present` x1 are machine facts their reason
#: texts name (each ends `not present`, which the character-class bug in
#: `_MACHINE_RE` could not match through its space), and all five were measured
#: skipping in the gate-shaped round home on 2026-10-13. `test_the_machine_fact_population_is_a_
#: closed_committed_set` pins the 23 by file and instance count against a literal,
#: so this number is not whatever the regex happens to produce.
BEFORE_MACHINE = 21


def _git(root: Path, *args: str) -> str:
    import subprocess
    r = subprocess.run(["git", "-C", str(root), *args],
                       capture_output=True, text=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def round_base(root: Path) -> str:
    """The commit this round's diff is measured FROM, as a rev.

    `git diff --name-only HEAD` names nothing the moment a round commits, and
    every automod round commits before its gate runs — so a diff read that way is
    the empty set at the only moment the pins below are graded, and each "this
    diff touches none of those files" assertion built on it would pass without
    looking at a single file. The merge-base with `main` survives the commit and
    is the same base the gate's own scope check and the review rung diff against.
    Diffing the WORKTREE against it (no `..HEAD`) also covers the uncommitted half,
    so the reader is correct mid-round and after the commit alike. If `main`
    cannot be named, fall back to HEAD's parent: one commit short of the real base
    is still a diff, where `HEAD` is none.
    """
    return (_git(root, "merge-base", "HEAD", "main")
            or _git(root, "rev-parse", "HEAD^") or "HEAD")


@lru_cache(maxsize=8)
def changed_paths(root: Path) -> frozenset[str]:
    """Every repo-relative path this round's diff names, from two git calls.

    Tracked changes come from `git diff` against the round base; the files a round
    ADDS are untracked until it commits, and a reader that missed them would call a
    brand-new test file untouched — which is the same blindness the pins below exist
    to catch, just earlier in the round. Cached per root: the pins ask about ~700
    test files, and a git subprocess each would add half a minute to the `tests`
    rung for nothing.
    """
    base = round_base(root)
    out = _git(root, "diff", "--name-only", base)
    out += "\n" + _git(root, "ls-files", "--others", "--exclude-standard")
    return frozenset(ln.strip() for ln in out.splitlines() if ln.strip())


def git_touches(root: Path, path: Path) -> bool:
    """True when the working round's diff names `path`.

    Reads `git diff --name-only <round_base>`, the round's own change set against
    its base, which is what the review rung and the gate's scope check both read,
    so headroom is attributed against the same set a human would audit.
    """
    try:
        return str(path.resolve().relative_to(root.resolve())) in changed_paths(root)
    except ValueError:
        return False

#: The conversions this round made that the before-inventory can attribute to a
#: NODE, named by the `(path, reason)` row it recorded rather than by a line number,
#: because a conversion shifts every line below it in the same file and a reason is
#: the one field that survives. `test_the_converted_sites_are_rows_of_the_committed_
#: inventory` reads these back out of the artifact, so the table cannot name a skip
#: the measurement never saw.
#:
#: The two `tests/test_trajectory_extraction.py` conversions are deliberately NOT
#: listed: both nodes and the surviving `_live_buckets` guard emit the SAME reason
#: (`no trajectories dir on this machine`), and a folded row carries no line number,
#: so the artifact cannot tell a converted node from the guard clause 4 requires to
#: stay. Those two are pinned by structure instead — `CONVERTED` in
#: `tests/test_skip_inventory.py` names them and `_body_of` reads their bodies.
#: Listing them here would let the tie below pass on the surviving guard's row.
CONVERTED: tuple[tuple[str, str], ...] = (
    ("tests/test_eval_scorer.py", "no baseline runs recorded yet"),
    ("tests/test_automod_regression.py", "noise floor not measured"),
    ("tests/test_uptake.py", "secondary_enabled is false"),
)

#: A machine fact is a property of THIS box that no test on this round can
#: change: a model that was never fetched, a browser that is not installed, a
#: desktop that is not attached. Those skips are honest and clause 2 keeps them.
#: A live-state skip is the opposite: it reports on mutable state outside the
#: worktree (production `~/lloyd-data`, the working vault, a tracked config
#: switch), so it can flip between two runs of one unchanged tree — which is how
#: this suite oscillated 39/40/41 against a ceiling of 40 and refused rounds
#: that had failed nothing.
#: The machine-fact population, as a NAME-match over the skip reason.
#:
#: Written once, with the alternation terms separated by a real `|` and each
#: intra-word separator spelled `[-_\s]`: this pattern used to be built by
#: `.replace(" ", "")` on a re.X string so that `[-_ ]` could be written readably,
#: and that strip also deleted the space *inside* every character class. The class
#: became `[-_]`, so no reason written with a space matched it — `recording mode
#: only`, `streaming model not present` and `turn-1-mini model not present` (7 skip
#: instances in the 2026-10-13 before-inventory) were all classified live-state.
#: That is the review's clause-2 finding on #2637's first round: a machine-fact
#: reason that reads as live state is a reason a later round can "convert" away.
#: The fix added `[-_\s]` inside every class and the two voice-model alternatives
#: the stripped pattern could not reach (`streaming model`, `turn-1-mini`), which
#: moves 5 instances from live to machine. It deliberately does NOT add
#: `recording mode`: `recording mode only (LLOYD_RECORD_*=1)` names an env flag an
#: operator sets, not a fact about this box's hardware, and counting it as a machine
#: fact would exempt two skips a later round could genuinely convert. It is NOT the
#: case that they were converted and no longer skip — `tests/test_turn_options.py:331`
#: and `tests/test_worker_capability_allowlist.py:163` both still carry the marker and
#: both still skip in gate shape, which is exactly why their two rows are in the
#: committed before-inventory. An earlier draft of this comment claimed they were gone;
#: the classification stands, for the reason above and not that one.
#: `test_the_machine_fact_population_is_a_closed_committed_set` in
#: `tests/test_skip_inventory.py` pins this population against a literal, not
#: against this regex, so widening or narrowing it goes red there.
_MACHINE_RE = re.compile(
    r"""parakeet
      | smart[-_\s]?turn
      | streaming[-_\s]asr
      | streaming[-_\s]model
      | turn[-_\s]?1[-_\s]?mini
      | silero
      | cam\+\+
      | chromium | firefox | webkit | ms[-_\s]playwright | playwright
      | huggingface | hf[-_\s]cache | from[-_\s]hf | node_modules
      | live[-_\s]desktop | desktop
      | cuda | gpu | whisper | piper | \btts\b | \bwake\b
      | not[-_\s]fetched | not[-_\s]present | not[-_\s]installed
    """,
    re.I | re.X,
)


def is_machine_fact(reason: str) -> bool:
    """True when a skip reason names a machine fact rather than live state.

    Deliberately a name-match over the reason text, because that is exactly the
    test clause 2 states ("every node whose skip reason names a machine fact").
    It errs toward machine-fact: a reason matching stays out of the live-state
    population, so a conversion can never quietly be justified by a vague reason.
    
    The classifier is a name-match, so it is a policy stated in one place and
    auditable from outside: `test_the_machine_fact_population_is_a_closed_committed_
    set` in `tests/test_skip_inventory.py` pins the population this produces against
    a literal committed set of reason texts, so a regex widened to swallow a
    live-state reason (or narrowed to drop a machine fact, which is how #2637's
    first round failed) has to move that literal too.
    """
    return bool(_MACHINE_RE.search(reason))


def read_inventory(path: Path) -> tuple[tuple[str, str, int], ...]:
    """One inventory as `(path, reason, count)` rows, order preserved."""
    rows: list[tuple[str, str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append((d["path"], d["reason"], int(d.get("n", 1))))
    return tuple(rows)


def machine_rows(rows):
    return [(p, r, n) for (p, r, n) in rows if is_machine_fact(r)]


def live_rows(rows):
    return [(p, r, n) for (p, r, n) in rows if not is_machine_fact(r)]


def total(rows) -> int:
    return sum(n for (_, _, n) in rows)


#: The call names that report a node as skipped, keyed on the LAST element of the
#: dotted name so `pytest.skip`, `mark.skipif` and a locally-imported `skip` are
#: all seen. `xfail` is deliberately absent: the rung counts `tests_skipped`, and an
#: xfail spends none of the 40, so flagging one would blur the units #2637 is about.
_SKIP_TERMINAL = {"skip", "skipif", "importorskip"}


def _dotted(node: ast.AST) -> str:
    """An `ast.Attribute` chain as text (`pytest.mark.skipif`); '' if it is not one."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return ""
    parts.append(cur.id)
    return ".".join(reversed(parts))


def _decorator_name(d: ast.expr) -> str:
    """The name a decorator APPLIES, with its arguments dropped.

    `@pytest.mark.skipif(cond, reason="no node")` → `pytest.mark.skipif`;
    `@_SKIP_UNLESS_NODE` → `_SKIP_UNLESS_NODE`; a marker built by a call
    (`@factory("skipif")`) → '' , which is the documented hole below. Dropping the
    arguments is the point: a marker's reason is a string, and the word inside a
    reason is prose about a skip rather than a skip.
    """
    return _dotted(d.func if isinstance(d, ast.Call) else d)


def skip_sites(src: str) -> list[str]:
    """Every way the code in `src` can report a node as SKIPPED, read out of its SYNTAX.

    Three shapes reach `tests_skipped`, and a text search sees one of them:

      * an inline call — `pytest.skip(...)`, `pytest.importorskip("x")`;
      * a marker — `@pytest.mark.skip`, `@pytest.mark.skipif(cond, reason=...)`;
      * nothing, which is the passing case.

    `"pytest.skip" in body` matches only the first, because neither spelling of the
    marker contains that substring: `pytest.mark.skipif` has `.mark.` where the call
    has nothing. #2637's second review (round `SM_20261013_173548`) named exactly
    that: clause 3's skipif half read as passing while it could not see a marker at
    all, and re-adding a data-root guard as `@pytest.mark.skipif` at a converted
    node — the precise regression the clause exists to catch — would have gone
    undetected. So this reads the syntax: markers come off `decorator_list`, calls off
    the dotted name of the callee. This is the ONE definition of "what is a skip" for
    #2637's pins — `tests/test_skip_inventory.py` reads a converted node with it and
    `tests/test_intel_pipeline_body.py::test_no_data_root_skip_is_left_on_the_witness_source_nodes`
    reads its whole file with it, because two spellings of the same rule is two things
    to get wrong, and the first one of them to go blind reads as a clean suite.

    Deliberate exclusions and one admitted hole:

      * `pytest.raises(pytest.skip.Exception)` is an ASSERTION about a skip, not a
        skip, and must not be flagged: ~12 nodes in `tests/test_trajectory_extraction.py`
        assert that the `_live_buckets` guard raises it, and clause 4 REQUIRES that
        guard to stay. `pytest.raises` is a call named `raises`, and
        `pytest.skip.Exception` is an attribute, so neither rule fires on it. That
        asymmetry is what `test_the_skip_needle_sees_a_marker_and_leaves_an_assertion_alone`
        pins in both directions.
      * `pytest.fail(...)` is a red, and a red is the outcome #2637 wants more of.
      * A marker bound to a name that does not mention skip and is not built inside
        `src` (`@_needs_node`, produced elsewhere by `pytest.mark.skipif`) is invisible
        to any per-node reader, including this one: `_decorator_name` reads the name a
        decorator applies, deliberately not its argument strings, because a marker's
        REASON is a string and a reader that scanned arguments would redden on
        `@pytest.mark.parametrize` over a table of skip reasons — which is real
        coverage of the `_live_buckets` guard clause 4 protects
        (`ABSENCE_EDGES` in `tests/test_trajectory_extraction.py`). Naming the name
        buys the false-positive-free half; beyond it the authority stays what it
        already is — the gate's own `tests` rung count and the committed inventories
        this module reads.
    """
    tree = ast.parse(src)
    out: list[str] = []

    def seg(n: ast.AST) -> str:
        return ast.get_source_segment(src, n) or _dotted(n) or ast.dump(n)

    # `@pytest.mark.skipif(cond, reason=...)` is ONE skip and TWO interesting nodes:
    # the decorator, which the marker rule must see, and the call `pytest.mark.skipif(…)`,
    # which the call rule would see again. Decorator subtrees are therefore collected
    # first and excluded from the call pass, so a site list is a list of skips rather
    # than a list of nodes — `len()` of it is quoted in assertions and in review
    # findings, and a double count there is an arithmetic error waiting for a reader.
    in_decorator: set[int] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in n.decorator_list:
                in_decorator.update(id(m) for m in ast.walk(d))
                name = _decorator_name(d)
                # The NAME the decorator resolves to, not its whole source: a marker
                # carries its reason as a string argument, and `@pytest.mark.parametrize`
                # over a table of skip reasons is a real shape in this suite
                # (`ABSENCE_EDGES` in `tests/test_trajectory_extraction.py`). A rule
                # that read arguments would redden on coverage of the very guard
                # clause 4 protects. What is read instead is `pytest.mark.skipif`,
                # `pytest.mark.skip`, or the bound identifier `@_SKIP_UNLESS_NODE`.
                if "skip" in name.lower():
                    out.append(f"@{seg(d)}")

    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and id(n) not in in_decorator:
            name = _dotted(n.func)
            if name.split(".")[-1] in _SKIP_TERMINAL:
                out.append(seg(n))
    return out
