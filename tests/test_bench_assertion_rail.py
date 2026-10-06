"""The bench assertion rail: a task may not be written unless it can be graded (#2286).

Seven backlog items in five weeks — #1589, #1724 (three tasks), #1968, #2174 (two),
#2228 (two), #2285 — each opened when
`tests/test_autoresearch_judge.py::test_every_live_bench_task_has_an_assertion_set` went
red because a new `~/obsidian/lloyd/bench/bench_NN_*.md` had arrived with no key in
`eval/autoresearch_assertions.yaml`. Seven times the fix was to add rows; none of them
closed the property, because the only thing enforcing it was that test, which runs after
every writer. Consequence on the tree before #2285: `judge.assertions_for` returned
`None` for `bench_028_contradiction_two_kinds`, and `judge._judge_rubric` — with
`rubric_mode` at its default `binary` — scored that task's rubric leg on the SCALAR judge
while its 27 keyed siblings scored binary, and `aggregate_variant` averaged the two into
one promotion decision.

This file pins the rail that refuses such a write on all four lanes that can land one:

| lane | entry point |
|---|---|
| `Write` / `Edit` | `agent_mcp/builtin_fs.py`, `_bench_assertion_refusal` |
| `vault_write` | `agent_mcp/vault.py`, `_bench_assertion_refusal` |
| Bash | `app/harness/safety.py` → `protected_paths.check_bash_write_denied` |
| worker-artifact promotion | `app/routers/workers.py`, `_bench_assertion_denial` |

The last is the lane the recurrence used — `workers/sources/bench_mine.py:537` gives the
spawned session a PROMPT, not a code path, and the promote route writes the artifact with
`Path.write_text`, touching none of the tool lanes. The 2026-09-22 class rule ("a guard
that lives on one of two write surfaces is not a guard") is why the rail is on all four,
and why two nodes below are about the guard's own shape rather than about any one lane.

Everything runs against a scratch `$HOME`. `autoresearch.bench_dir` is
`~/obsidian/lloyd/bench` and `_expand` runs `os.path.expanduser` on every
`load_config()` call, so pointing `HOME` at a tmp directory relocates the corpus that
`app.harness.bench_corpus.corpus_target` resolves to — the lanes are driven against a
real corpus directory in the scratch home, not against a stubbed resolver. The assertion
table is NOT stubbed for the deny/allow nodes: it is the real
`eval/autoresearch_assertions.yaml`, because a rail proven against a fake table proves
nothing about whether tomorrow's writer gets through.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_mcp import builtin_fs as FS  # noqa: E402
from agent_mcp import main as M  # noqa: E402
from agent_mcp import vault as V  # noqa: E402
from app.harness import bench_authoring as RAIL  # noqa: E402
from app.harness import bench_corpus  # noqa: E402
from app.routers import workers as router  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from scripts.autoresearch import judge  # noqa: E402

SID = "20261006_2286_rail_test"
CORPUS_REL = "obsidian/lloyd/bench"

#: A real id in the real table: #2285 authored 5 `{id, text}` rows for it on
#: 2026-10-06, so the allow half is tested against an entry the judge really resolves.
KEYED_ID = "bench_028_contradiction_two_kinds"

#: An id in no table on any branch — the shape a bench-mine run produces the night after
#: this one lands, before anyone has authored its rubric.
KEYLESS_ID = "bench_999_rail_probe"

TASK_FM = {
    "type": "note",
    "category": "synthetic",
    "id": "",
    "prompt": "Say something short and true.",
    "objective_checks": [{"type": "regex", "value": "yes|no", "gap_class": "probe"}],
    "rubric_criteria": ["answers_the_ask", "conciseness"],
    "safety_critical": False,
}

STAGING_FM = {
    "source": "bench-mine",
    "confidence": 0.5,
    "review_status": "pending",
    "rationale": "mined to exercise the #2286 rail",
    "generated_at": "2026-10-06T11:00:00+00:00",
}


def _task_text(task_id: str) -> str:
    return ("---\n" + yaml.dump(dict(TASK_FM, id=task_id), default_flow_style=False,
                                allow_unicode=True)
            + "---\n\nThe task body.\n")


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    """Off the change ledger: these nodes assert on the bytes on disk, not on who is
    recorded as having changed them."""
    from agent_mcp import _change_ledger
    monkeypatch.setattr(_change_ledger, "enabled", lambda: False)
    FS.reset_read_records()
    yield
    FS.reset_read_records()


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A scratch `$HOME` whose vault holds the bench corpus the config points at.

    One home, every lane: `HOME` alone moves `autoresearch.bench_dir` (expanded per
    `load_config()` call), and the two module-level vault roots the vault lane and the
    promote route resolve against are aimed at the same directory — so a file refused
    here is refused at the address the rail actually watches, and a file allowed lands
    where a real promotion would land it.
    """
    h = tmp_path / "home"
    (h / CORPUS_REL).mkdir(parents=True)
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setattr(FS, "get_bound_session", lambda: SID)
    monkeypatch.setattr(V, "VAULT", h / "obsidian")
    # The vault lane appends an audit line per write; pointed at tmp so a rail test
    # cannot append to the real vault's audit log.
    audit = tmp_path / "audit"
    monkeypatch.setattr(V, "AUDIT_LOG_DIR", audit)
    monkeypatch.setattr(V, "AUDIT_LOG_FILE", audit / "writes.jsonl")
    monkeypatch.setattr(router, "PENDING_ROOT", h / "pending-research")
    monkeypatch.setattr(router, "VAULT_ROOT", h / "obsidian")
    return h


def corpus(home: Path) -> Path:
    return home / CORPUS_REL


def _bash_refusal(command: str, home: Path) -> str:
    """The Bash lane's refusal as one string, "" when it allows the command.

    `check_bash_command` returns `(label, excerpt)` or None, so unwrapping the label is
    the only adaptation. Driven through it and not through `check_bash_write_denied`
    because it is the function the Bash tool calls, and the deny-set is one of five
    checks inside it — driving the inner function would leave the wiring untested, which
    is how a lane stays open.
    """
    from app.harness.safety import check_bash_command
    found = check_bash_command(command, cwd=str(home))
    return "" if found is None else found[0]


def _pair(res) -> tuple[str, bool]:
    """`(text, is_error)` from whatever the seam handed back.

    `main.call_tool` normally returns a `CallToolResult`, but a PostToolUse hook that
    rewrites the result hands back `(text, is_error)` instead — which is what the vault
    lane produced here. Normalising is the only way the node asserts the refusal rather
    than the shape it travelled in.
    """
    if isinstance(res, tuple) and len(res) == 2:
        body, is_error = res
        text = body[0].text if hasattr(body, "__getitem__") else str(body)
        return text, bool(is_error)
    text = res.content[0].text
    flag = getattr(res, "is_error", None)
    if flag is None:
        flag = getattr(res, "isError", False)
    return text, bool(flag)


def _text(res) -> str:
    return _pair(res)[0]


def _refused(res) -> bool:
    return _pair(res)[1]


def _error(res) -> str:
    try:
        return json.loads(_text(res)).get("error", "")
    except (ValueError, TypeError, AttributeError):
        return _text(res)


def _staged(home: Path, task_id: str) -> Path:
    """One staged bench-mine artifact, the shape `write_staging_note` writes."""
    d = home / "pending-research" / "bench-mine" / "2026-10-06"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{task_id}.md"
    task_block = ("---\n" + yaml.dump(dict(TASK_FM, id=task_id),
                                      default_flow_style=False, allow_unicode=True)
                  + "---\n\nMined to exercise the rail.\n")
    p.write_text("---\n" + yaml.dump(STAGING_FM, default_flow_style=False,
                                     allow_unicode=True)
                 + "---\n\n" + task_block, encoding="utf-8")
    return p


def _promote(home: Path, src: Path) -> dict:
    app = FastAPI()
    app.include_router(router.router)
    r = TestClient(app).post("/api/workers/pending/promote", json={"path": str(src)})
    return {"status": r.status_code, **(r.json() if r.content else {})}


def _assert_names_the_key(reason: str, task_id: str) -> None:
    """Clause 1's second half: the refusal says WHICH key is missing and what to do.

    A refusal that only says "not allowed" sends the writer back to the test file to
    work out what it did — the six-node diagnosis step this item exists to remove.
    """
    assert task_id in reason, reason
    assert "eval/autoresearch_assertions.yaml" in reason, reason
    assert "id:" in reason and "text:" in reason, (
        f"the refusal must also say what a correct entry looks like, or the writer "
        f"repeats the attempt in a different shape: {reason}")


# ── clause 1: a keyless task file is refused, on every lane, and nothing lands ──

async def test_the_write_lane_refuses_a_keyless_bench_task_and_creates_no_file(home):
    target = corpus(home) / f"{KEYLESS_ID}.md"
    res = await FS.call_tool("Write", {"file_path": str(target),
                                       "content": _task_text(KEYLESS_ID)})
    assert res.is_error is True, _text(res)
    _assert_names_the_key(_error(res), KEYLESS_ID)
    assert not target.exists(), (
        "a refusal that leaves the file behind has not refused anything — the corpus is "
        "exactly the set of files on disk")


async def test_the_vault_write_lane_refuses_a_keyless_bench_task_and_creates_no_file(home):
    """Driven through `main.call_tool`, the aggregator trip, not `V._vault_write`
    directly: the refusal must arrive as an error RESULT on that seam, and a session id
    must be on it — a sessionless state-changing call is refused earlier by #1053's
    guard, which would have this node asserting the wrong refusal."""
    await M.list_tools()
    res = await M.call_tool("vault_write", {
        "path": f"lloyd/bench/{KEYLESS_ID}.md", "content": _task_text(KEYLESS_ID)},
        {"lloyd/session_id": SID})
    assert _refused(res) is True, _text(res)
    _assert_names_the_key(_error(res), KEYLESS_ID)
    assert not (corpus(home) / f"{KEYLESS_ID}.md").exists()


async def test_the_write_lane_writes_a_task_whose_id_has_real_rows(home):
    """Clause 2 on the `Write` lane: legitimate bench authoring still works, and the
    bytes the writer asked for are the bytes that land.

    The node either side of this one is the point. A rail that refuses everything
    satisfies clause 1 loudly and clause 2 never, and the failure mode that costs a
    corpus is the quiet one where the guard blocks the writer it was never about.
    """
    target = corpus(home) / f"{KEYED_ID}.md"
    text = _task_text(KEYED_ID)
    res = await FS.call_tool("Write", {"file_path": str(target), "content": text})
    assert res.is_error is False, _text(res)
    assert target.is_file(), "the call was allowed and nothing landed"
    assert target.read_text(encoding="utf-8") == text


async def test_the_vault_write_lane_writes_a_task_whose_id_has_real_rows(home):
    """Clause 2 on the `vault_write` lane, through the aggregator trip.

    This is the exact call a bench-mine session makes once its task has a row, so it is
    also the node that would catch a rail mis-wired to refuse the whole `lloyd/bench/`
    prefix instead of the keyless id.
    """
    await M.list_tools()
    res = await M.call_tool("vault_write", {
        "path": f"lloyd/bench/{KEYED_ID}.md", "content": _task_text(KEYED_ID)},
        {"lloyd/session_id": SID})
    assert _refused(res) is False, _text(res)
    assert (corpus(home) / f"{KEYED_ID}.md").is_file()


def test_the_bash_lane_refuses_a_keyless_bench_task_whichever_guard_closes_it(home):
    """The lane this recurrence arrived through is not a tool call, and the Bash tool is
    the one path that can write the corpus without passing any of the three rails — so it
    is measured rather than assumed shut.

    `assert`ed as a property with two admissible closings, because which guard fires is
    configuration: the write deny-set closes the whole vault (label "obsidian (vault)"),
    and if that is ever narrowed to let a vault path through, the bench rail is the thing
    that must answer. A node that asserted only one of the two would go red on a
    configuration change that weakens nothing, and — worse — would go GREEN if the lane
    opened while the assertion still expected the deny-set's wording. Both closings are
    refusals; the node's job is to fail if the command is ever allowed.
    """
    target = corpus(home) / f"{KEYLESS_ID}.md"
    staged = home / "candidate.md"
    staged.write_text(_task_text(KEYLESS_ID), encoding="utf-8")
    why = _bash_refusal(f"cat {staged} > {target}", home)
    assert why, (
        "the Bash lane wrote into the bench corpus with nothing refusing it: the write "
        "deny-set does not cover the vault here, so the bench rail at "
        "`app/harness/protected_paths.check_bash_write_denied` is the only thing standing "
        "between a shell redirect and a corpus the judge scores two ways")
    _assert_names_the_key(why, KEYLESS_ID)
    assert not target.exists()


def test_the_bash_lane_never_refuses_a_keyed_task_for_the_rail(home):
    """The allow half, stated so it can only fail one way: whatever the Bash lane does
    with a task that HAS real rows, it must not do it because of the bench rail.

    The deny-set closing the vault wholesale is a separate policy with its own test
    (`tests/test_bash_write_guard.py`); a refusal here that names the missing key would
    mean the rail had become the reason legitimate authoring fails — clause 2's
    violation on the lane nobody was watching.
    """
    target = corpus(home) / f"{KEYED_ID}.md"
    staged = home / "candidate.md"
    staged.write_text(_task_text(KEYED_ID), encoding="utf-8")
    why = _bash_refusal(f"cp {staged} {target}", home)
    assert "no assertion set" not in why, (
        f"a task with real rows was refused BY THE RAIL: {why}")
    assert KEYED_ID not in why or "assertion" not in why, (
        f"a task with real rows was refused BY THE RAIL: {why}")


# ── clause 3: the lane the recurrence actually used ───────────────────────────

def test_the_promotion_route_that_lands_bench_tasks_refuses_a_keyless_one(home):
    """#2285's filing says the file reached the corpus as unattributed dirty state
    (vault `dc72ec91`), and `workers/sources/bench_mine.py:537` is a PROMPT, not a
    writer. The promote route is the code that lands a mined candidate and it calls
    none of the three tool lanes — this node is what "demonstrated for that lane, not
    only for vault_write" means."""
    src = _staged(home, KEYLESS_ID)
    out = _promote(home, src)
    assert out["status"] == 400, out
    _assert_names_the_key(out.get("detail", ""), KEYLESS_ID)
    assert not (corpus(home) / f"{KEYLESS_ID}.md").exists(), (
        "the refusal must not have landed the task")
    assert src.exists(), (
        "and must not have consumed the candidate: the round that authors the row must "
        "still be able to promote it")


def test_the_promotion_route_still_lands_a_task_whose_id_has_real_rows(home):
    src = _staged(home, KEYED_ID)
    out = _promote(home, src)
    assert out["status"] == 200, out
    assert (corpus(home) / f"{KEYED_ID}.md").is_file(), out
    assert not src.exists(), "the promoted artifact is consumed by the move"


# ── the guard's own shape: one predicate, no second source of truth ───────────

def test_the_rail_answers_from_the_judge_and_not_from_its_own_copy_of_the_table(
        home, monkeypatch):
    """The refusal must move when the judge's table moves, or it is a second source of
    truth that will disagree with the thing it protects.

    The real table is read once for the fixture precondition, then the lane is asked
    twice with `judge.load_assertions` redirected to a table where today's keyed id has
    no key and the keyless id has real rows. Both verdicts flip, which is the only
    proof available in-process that the rail calls `judge.load_assertions` /
    `judge.assertions_for` rather than restating them.
    """
    real = judge.load_assertions()
    assert KEYED_ID in real and KEYLESS_ID not in real, (
        "the fixture ids have drifted from the table; a node that asserts a flip "
        "against a table that already agrees with it proves nothing")
    flipped = {k: v for k, v in real.items() if k != KEYED_ID}
    flipped[KEYLESS_ID] = [{"id": "probe_row", "text": "The reply says the thing."}]
    monkeypatch.setattr(judge, "load_assertions", lambda *a, **k: flipped)
    assert RAIL.bench_write_defect(str(corpus(home) / f"{KEYED_ID}.md")), (
        "the rail still allowed a task the redirected table has no key for — it is "
        "answering from something other than judge.load_assertions")
    assert RAIL.bench_write_defect(str(corpus(home) / f"{KEYLESS_ID}.md")) == "", (
        "the rail still refused a task the redirected table has rows for")


def test_the_rail_refuses_a_graded_marker_the_coverage_node_would_let_through(home):
    """`test_every_live_bench_task_has_an_assertion_set` passes an entry that is a
    `graded: true` MAPPING, and #1968's
    `test_bench_023_resolves_to_authored_checks_and_not_a_graded_marker` exists because
    that marker is the escape hatch which keeps a task on the scalar judge while the
    coverage node reads green. A rail that copied the node's predicate would reproduce
    the bug it exists to prevent, so the rail asks `assertions_for` — which returns
    None for a mapping — and this node pins the difference. No task in the table is
    marked `graded` today; the point is that the writing side cannot later be widened
    into the old shape without this node saying so."""
    real = judge.load_assertions()
    marker = {**real, KEYLESS_ID: {"graded": True}}
    reason = ""
    previous = judge.load_assertions
    judge.load_assertions = lambda *a, **k: marker
    try:
        reason = RAIL.bench_write_defect(str(corpus(home) / f"{KEYLESS_ID}.md"))
        allowed = RAIL.assertion_key_defect(KEYED_ID)
    finally:
        judge.load_assertions = previous
    assert reason, "a `graded: true` mapping sailed through the rail"
    assert "mapping" in reason or "graded" in reason, reason
    assert allowed == "", "the same table read must still allow a real assertion set"


@pytest.mark.parametrize("broken", ["empty", "unreadable", "unimportable"],
                         ids=["table-reads-empty", "table-raises",
                              "table-unimportable"])
def test_the_rail_fails_closed_when_it_cannot_read_its_own_input(broken, home,
                                                                 monkeypatch):
    """Every other guard on these lanes refuses when its checker cannot load
    (`agent_mcp/vault.py`'s deny-set, `builtin_fs`'s) and says so in the refusal:
    "every write silently proceeded while the checker was broken" is the hole #1049
    exists to close. A rail that reads a table has one more way to be broken — a
    zero-key read — and reading "no keys" as "nothing denied" would open every lanes
    exactly when the table went missing: the counting-guard failure again, a check
    whose denominator can be zero is not a check."""
    if broken == "empty":
        monkeypatch.setattr(judge, "load_assertions", lambda *a, **k: {})
    elif broken == "unreadable":
        def _boom(*a, **k):
            raise OSError("table unreadable")
        monkeypatch.setattr(judge, "load_assertions", _boom)
    else:
        # `from scripts.autoresearch import judge` binds the PARENT'S attribute, so
        # putting None in sys.modules is not the simulation it looks like — the real
        # submodule still resolves. An object without the judge's API is.
        import scripts.autoresearch as pkg
        monkeypatch.setattr(pkg, "judge", object())

    reason = RAIL.bench_write_defect(str(corpus(home) / f"{KEYLESS_ID}.md"))
    assert reason, "a rail that could not read its table reported a pass"
    assert "not being written" in reason, reason
    assert "Report this" in reason, (
        f"a fail-closed refusal must not read like a policy the writer can satisfy: "
        f"{reason}")


def test_the_rail_is_inert_on_every_path_it_was_not_about(home):
    """The three ways this rail could silently become a vault-wide write blocker, all
    answered on the sandbox corpus: a non-task file INSIDE the corpus, a task-shaped name
    OUTSIDE it, and the corpus directory itself.

    Matters because the Bash lane now asks the predicate for every write target in every
    command. A predicate that keyed on the directory instead of the filename would stop
    `notes.md` writes in the corpus, and one that keyed on the filename alone would stop
    `~/lloyd/bench/bench_999_x.md` in the repo tree — an authoring scratch location that
    is not graded at all. Both are refusals of legitimate work, which is the failure mode
    clause 2 exists to catch, on the lane nobody watches.
    """
    dir_ = corpus(home)
    assert RAIL.bench_write_defect(str(dir_ / "notes.md")) == ""
    assert RAIL.bench_write_defect(str(dir_ / "bench_mine_staging.json")) == ""
    stray = home / "lloyd" / "bench" / f"{KEYLESS_ID}.md"
    stray.parent.mkdir(parents=True, exist_ok=True)
    assert RAIL.bench_write_defect(str(stray)) == "", (
        "the rail reached outside the corpus: an ungraded scratch file is not a task")
    assert RAIL.bench_write_defect(str(dir_)) == "", (
        "the rail answered for the directory itself, which is every write under it")
    assert RAIL.bench_write_defect(str(dir_ / f"{KEYLESS_ID}.md"))


def test_the_bench_corpus_the_rail_watches_is_the_corpus_the_judge_reads():
    """The one assumption every lane shares: the directory the rail guards is the one
    `scripts.autoresearch.common.load_bench_tasks` is handed, which is what the coverage
    node enumerates. If a later change repoints `autoresearch.bench_dir`, moves the
    vault, or starts a second corpus directory, the rail goes on refusing files nobody
    grades and passing files nobody reads — and every other node here stays green while
    it does, because they all live inside whatever the config says. This node reads the
    config from outside.

    `corpus_roots()` lists two sources — the configured key and the vault's
    conventional `lloyd/bench`, deliberately both, so the read deny survives the config
    key moving. What must be in that list is the directory the judge reads; what must
    resolve is a probe inside it, since `corpus_target` accepts only a `realpath`
    contained in a root and resolving the other way would have the rail guard a path no
    writer can name.
    """
    from scripts.autoresearch.common import load_config

    configured = Path(load_config().paths.bench_dir)
    assert configured.is_absolute(), configured
    roots = [Path(r) for r in bench_corpus.corpus_roots()]
    assert configured.resolve() in roots, (
        f"the judge reads {configured} and the rail guards {roots} — it would be "
        f"guarding a corpus that is not graded, or grading one it does not guard")
    probe = configured / f"{KEYLESS_ID}.md"
    assert bench_corpus.corpus_target(str(probe)) == str(probe.resolve()), (
        "the rail's resolver does not answer for the corpus it just claimed")
    assert RAIL.frontmatter_id(_task_text(KEYED_ID)) == KEYED_ID, (
        "the vault lane's declared-id check reads the FIRST frontmatter block; if that "
        "stops parsing, the lane silently checks only the stem")


def test_no_lane_holds_a_list_of_permitted_bench_ids():
    """Clause 5's "no exception list anywhere in the round's diff", as a check that
    survives its next reader. An allowlist would be a hand-maintained set over an
    open-set corpus — the defect the 2026-09-20 class rule names, and why this item is
    a rail rather than a skill instruction. Checked on the syntax tree: a container
    literal holding ≥2 bench-shaped ids in any of the five files that carry the rail.
    Ids in prose are history (an item number, the task this rail was written about) and
    are not something a writer can slip past."""
    files = ["app/harness/bench_authoring.py", "agent_mcp/vault.py",
             "agent_mcp/builtin_fs.py", "app/harness/protected_paths.py",
             "app/routers/workers.py"]
    offenders = []
    for rel in files:
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.List, ast.Set, ast.Tuple)):
                continue
            ids = [e.value for e in node.elts
                   if isinstance(e, ast.Constant) and isinstance(e.value, str)
                   and RAIL.BENCH_TASK_FILE_RE.match(e.value + ".md")]
            if len(ids) >= 2:
                offenders.append(f"{rel}:{node.lineno} {ids}")
    assert not offenders, "a bench-id exception list appeared: " + "; ".join(offenders)


def test_every_lane_that_can_write_into_the_corpus_asks_the_rail():
    """The recurrence's actual failure mode, as a node: not a wrong verdict, but a lane
    that never asked. #2285's file arrived through a lane no guard was on, and the
    2026-09-22 class rule is that you find the surfaces by enumerating them — so the
    enumeration is here, and a fifth lane that appears without joining it, or one that
    stops calling, says which.

    A source scan rather than a call, because the thing at risk is the call site: the
    helper could keep its name and become a second source of truth about the table,
    which is how a rail quietly stops being one.
    """
    lanes = {
        "agent_mcp/builtin_fs.py": "_bench_assertion_refusal",
        "agent_mcp/vault.py": "_bench_assertion_refusal",
        "app/harness/protected_paths.py": "_bench_assertion_denial",
        "app/routers/workers.py": "_bench_assertion_denial",
    }
    for rel, symbol in lanes.items():
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert f"def {symbol}" in src, f"{rel} no longer defines {symbol}"
        body = src.split(f"def {symbol}", 1)[1]
        assert "bench_write_defect" in body, (
            f"{symbol} in {rel} no longer asks the one predicate — it has become its "
            f"own answer about the table")
        assert f"{symbol}(" in src.replace(f"def {symbol}", "", 1), (
            f"{symbol} is defined and never called in {rel}")
    assert len(lanes) == 4, (
        "the lane enumeration changed shape; the module docstring's table and the "
        "Bash-lane node have to be re-read against it")
