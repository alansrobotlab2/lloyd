"""#2272: an ACCOUNTING ceiling on the two skill-embed routes that have no cap.

`app/skill_embed.py` is the one module both uncapped routes already import, and it is
the only place a shared char count can live: a limit computed independently at each
call site drifts by exactly the thing the count is meant to catch, which is why
`app/prefetch.py` owns the one number its splices share (`SKILL_BODY_MAX`, 6,000 chars,
`app/prefetch.py:54`) rather than letting each pick its own (#2208), and #2272's soft
limit is a SECOND number beside it: deliberately not aliased, because a limit that
cannot move independently of the hard cut is not a proposal about the uncapped routes.

This file pins the four code halves of #2272 — the constant and the command that
re-derives it, the stamp on the two uncapped routes, the stamp's scope (a route with a
hard cut does not need a soft line), and the fact that the accounting changes no prompt
by one byte and adds no switch. The behaviour half — whether to cap or spill those
routes — stays a separate ruling made on the traffic this records, so no node here
asserts what an over-limit embed should DO.
"""
from __future__ import annotations

import ast
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import app.skill_embed as SE  # noqa: E402
from app import event_log as EL  # noqa: E402
from app.harness import skill_dispatch as SD  # noqa: E402

#: The two routes #2272 is about, taken from their owner rather than restated as
#: literals here: a test carrying its own copy of a route name keeps passing when the
#: constant changes, and then stamps a route that no longer exists.
UNCAPPED = (SD.ROUTE_AUTONOMY_TASK, SD.ROUTE_WORKER_PROMPT)

#: The four routes the recorder knows, likewise from their owner.
ALL_ROUTES = (SD.ROUTE_PREFETCH, SD.ROUTE_PREFETCH_EXCERPT,
              SD.ROUTE_AUTONOMY_TASK, SD.ROUTE_WORKER_PROMPT)

#: The worst embed the two uncapped routes have produced on this machine: 69,639
#: chars, the whole body of `autonomy-data-pipeline`, booked by `app/autonomy.py` as
#: `len(skill_content)`. The clause-4 nodes embed a skill THIS size rather than
#: limit+1, because what is worth catching is a change to a payload that really runs,
#: not a change to the arithmetic three characters around the threshold.
WORST_EMBED_CHARS = 69_639

_ids = itertools.count()


def _sid(kind: str) -> str:
    """A session id no earlier run of this file wrote to.

    `log_event` appends, so a fixed id makes "the row this call wrote" mean "some row
    with this event name", and a second parametrization of the same route would read
    the first one's row and pass for the wrong reason.
    """
    return f"skill-embed-{kind}-{next(_ids)}"


def _rows_for(session_id: str) -> list[dict]:
    """The `skill.embedded` rows written for one session, read from the real log file.

    Read through `app.event_log`'s own `EVENT_LOGS_DIR`, not a re-derived path: the
    point is that the stamp reached the file the re-derivation command sums over, and
    a test that guessed at the directory would pass on a scratch root and on nothing
    else.
    """
    # Read through the module, not a name bound at import: `tests/conftest.py`'s
    # autouse `_isolate_background_records` repoints `EVENT_LOGS_DIR` at a fresh tmp
    # dir per test, and the recorder writes where THAT says.
    path = Path(EL.EVENT_LOGS_DIR) / f"{session_id}.events.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "skill.embedded":
            out.append(row)
    return out


def _record(route: str, embedded_chars: int, *, kind: str, **kw) -> dict:
    """One `record_skill_embed` call through the real writer, plus what it wrote."""
    sid = _sid(kind)
    rec = SE.record_skill_embed(route=route, skill=kw.pop("skill", "some-skill"),
                                embedded_chars=embedded_chars,
                                source_chars=kw.pop("source_chars", 999_999),
                                truncated=kw.pop("truncated", False),
                                session_id=sid)
    return {"rec": rec, "rows": _rows_for(sid), "session_id": sid}


def _soft_limit_doc() -> tuple[str, str]:
    """The constant's own comment block and the command inside it, dedented.

    Read by AST-locating the assignment and taking the `#:` comment run that sits
    immediately above it, so a reworded passage elsewhere in the file cannot satisfy it
    and a block that drifts away from the value it documents fails: the number and the
    command that re-derives it have to be read together or the number becomes folklore,
    which is what #2134's 4,000/6,000 pair became.

    Returns `(block, command)`; `command` is the program after the
    `$ python3 - <<'PY'` line up to the `PY` terminator, with the `#:` marker and one
    following space removed from each line and nothing else touched — the indentation
    inside the heredoc is the program's own, so the text a reader would paste is
    exactly what the file holds.
    """
    src = (_REPO_ROOT / "app" / "skill_embed.py").read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)
    assign = next((n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name)
                           and t.id == "UNCAPPED_EMBED_SOFT_LIMIT" for t in n.targets)),
                  None)
    assert assign is not None, (
        "app/skill_embed.py must define UNCAPPED_EMBED_SOFT_LIMIT at module level")
    i = min(getattr(t, "lineno", assign.lineno) for t in assign.targets) - 2
    block: list[str] = []
    while i >= 0:
        if not lines[i].lstrip().startswith("#"):
            break
        block.insert(0, lines[i].lstrip())
        i -= 1
    assert block, "the constant must carry a comment block of its own above it"

    def uncomment(ln: str) -> str:
        # Strip the `#:` marker and EXACTLY one following space, never more: the
        # indentation inside the heredoc is the program's own, and flattening it
        # rebuilds a script that cannot compile.
        body = ln[1:]
        if body.startswith(":"):
            body = body[1:]
        return body[1:] if body.startswith(" ") else body

    text = "\n".join(uncomment(ln) for ln in block)

    marker = next((k for k, ln in enumerate(text.splitlines())
                   if ln.strip().startswith("$ python3 - <<'PY'")), None)
    assert marker is not None, (
        "the constant's block must carry the re-derivation command introduced by "
        "`$ python3 - <<'PY'`, not a prose description of it")
    body = []
    for ln in text.splitlines()[marker + 1:]:
        if ln.strip() == "PY":
            break
        body.append(ln)
    assert body, "the heredoc command has no body"
    return text, "\n".join(body)


# --------------------------------------------------------------------------- #
# clause 1: the constant, and the command that re-derives it
# --------------------------------------------------------------------------- #

def test_the_soft_limit_is_a_char_int_whose_comment_names_the_route_sum_command(
        tmp_path):
    """The number is a ruling on traffic, so its comment must carry its derivation.

    #2272 defers WHAT to do about an uncapped embed, so the durable thing this module
    can commit is how the number was arrived at. Pinned two ways: the constant is a
    plain `int`, and the command in its comment is actually RUN — over a synthetic log
    tree whose sums are known in advance — rather than grepped for words about it.

    The command must name the two uncapped routes. `prefetch` sizes come out of
    `skill_dispatch._walk_deliveries`, which slices
    `context_text[body_start:close if close >= 0 else None]`, so a `<skill …>` tag
    quoted in prose with no closing tag books every character after it as that skill's
    size (live example: a 10,115-char `prefetch` row for a tag quoted inside a
    sentence, session `20261003_210909_owedcheck_731d`). A sum over all routes inherits
    that phantom; the two call-site routes, booked at the splice, cannot have one.
    """
    assert isinstance(SE.UNCAPPED_EMBED_SOFT_LIMIT, int) and not isinstance(
        SE.UNCAPPED_EMBED_SOFT_LIMIT, bool), SE.UNCAPPED_EMBED_SOFT_LIMIT
    assert SE.UNCAPPED_EMBED_SOFT_LIMIT > 0

    block, cmd = _soft_limit_doc()
    assert "embedded_chars" in block, "the block must name the field the sum runs over"
    assert "lloyd-data/event_logs" in block and "events.jsonl" in block, (
        "the block must name the log files the command reads, by their real path")
    for route in UNCAPPED:
        assert route in block, (
            f"the comment must name {route}: summed over every route the figure "
            "inherits the tag-walk phantom instead of measuring the routes with no cap")

    over = SE.UNCAPPED_EMBED_SOFT_LIMIT + 10
    rows = [{"event": "skill.embedded",
             "data": {"route": r, "skill": "s", "embedded_chars": over}}
            for r in UNCAPPED]
    # A capped route holding far more than the limit must not move the figure: it is
    # not what this constant is a ceiling for, and it is the contaminated half.
    rows.append({"event": "skill.embedded",
                 "data": {"route": SD.ROUTE_PREFETCH, "skill": "s",
                          "embedded_chars": over + 1_000_000}})
    logs = tmp_path / "event_logs"
    logs.mkdir()
    (logs / "s1.events.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    proc = subprocess.run([sys.executable, "-"], input=cmd, capture_output=True,
                          text=True, timeout=180,
                          env={"SKILL_EMBED_LOG_GLOB": str(logs / "*.events.jsonl"),
                               "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, f"the documented command failed: {proc.stderr[-600:]}"
    got = json.loads(proc.stdout)
    assert set(got) == set(UNCAPPED), (
        f"the command reported {sorted(got)}; it must cover exactly the two uncapped "
        "routes and no capped route at all")
    for route in UNCAPPED:
        assert got[route] == {"n": 1, "sum": over, "max": over}, (route, got[route])


def test_the_soft_limit_equals_the_capped_route_s_cut_today():
    """The value is `prefetch.SKILL_BODY_MAX`'s number, by measurement and not by alias.

    Two halves, because the module's comment claims both. Equal today: 6,000 chars is
    the number `app/prefetch.py:54` holds for one skill on the route that DOES cut, so
    the soft line sits where the engine already drew a line rather than at a percentile
    of the traffic under review. And NOT bound to it: if `skill_embed.py` imported
    `SKILL_BODY_MAX`, the soft limit would stop being a proposal about the uncapped
    routes and become the hard cut wearing a second name, and the pair could never be
    seen to move apart when the ruling in #2272 clause 5 comes.
    """
    import app.prefetch as PF

    assert isinstance(SE.UNCAPPED_EMBED_SOFT_LIMIT, int), type(
        SE.UNCAPPED_EMBED_SOFT_LIMIT)
    assert not isinstance(SE.UNCAPPED_EMBED_SOFT_LIMIT, bool)
    assert SE.UNCAPPED_EMBED_SOFT_LIMIT == PF.SKILL_BODY_MAX == 6_000, (
        f"soft={SE.UNCAPPED_EMBED_SOFT_LIMIT} hard={PF.SKILL_BODY_MAX}: the module "
        "comment says the soft line is the number the cut route already holds, so a "
        "change to either is a change to that claim")
    # AST, not a text grep: the name has to stay free in EXECUTABLE statements, and
    # prose legitimately names it (the module docstring's opening paragraph already
    # spelled the cap `prefetch.SKILL_BODY_MAX` before this item existed, and a
    # comment-stripping grep cannot tell that prose from a binding).
    mod = (_REPO_ROOT / "app" / "skill_embed.py").read_text(encoding="utf-8")
    tree = ast.parse(mod)
    doc = ast.get_docstring(tree) or ""
    named = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value == doc:
            continue                      # the module docstring is prose, not a binding
        for attr in ("module", "name", "attribute"):
            v = getattr(n, attr, None)
            if isinstance(v, str):
                named.add(v)
    assert "SKILL_BODY_MAX" not in named, (
        "the soft limit must not be ALIASED to the hard cut in code: the comment may "
        "name the number, an import or attribute access may not bind it")
    assert "SKILL_BODY_MAX" in doc or "SKILL_BODY_MAX" in mod, (
        "the comment is supposed to name the number this one equals — if the prose "
        "went too, the reader of 6_000 has no origin")


@pytest.mark.parametrize("route", UNCAPPED, ids=list(UNCAPPED))
def test_an_over_limit_uncapped_embed_is_stamped_on_the_record_and_the_event(
        tmp_path, route):
    """The whole point of the accounting: a reader of the log can find these runs.

    Three things have to hold together or the ledger is useless. The returned record
    carries `over_soft_limit: True`; it carries `soft_limit` set to the LIMIT and not
    to `embedded_chars`, which is already on the row and whose repetition would let a
    reader mistake one for the other; and the `skill.embedded` line in the session log
    carries both keys too. `app/autonomy.py` and `workers/sources/deep_research.py`
    both hand the returned dict to `log_event` as the event's data, so the record half
    and the event half are one write and both are asserted here.
    """
    over = SE.UNCAPPED_EMBED_SOFT_LIMIT + 1
    out = _record(route, over, kind=f"over-{route}", skill="big-skill")
    rec = out["rec"]
    assert rec["over_soft_limit"] is True, rec
    assert rec["soft_limit"] == SE.UNCAPPED_EMBED_SOFT_LIMIT, rec
    assert rec["embedded_chars"] == over, rec
    assert rec["route"] == route, rec
    assert rec["truncated"] is False, rec

    rows = out["rows"]
    assert len(rows) == 1, rows
    data = rows[0]["data"]
    assert data["over_soft_limit"] is True, data
    assert data["soft_limit"] == SE.UNCAPPED_EMBED_SOFT_LIMIT, data


def test_the_stamp_reaches_the_file_the_re_derivation_command_reads(tmp_path):
    """The process boundary the accounting exists to cross: recorder → session log.

    `record_skill_embed` returns a dict, and a stamp that lived only in that dict would
    be invisible in the one corpus clause 1's command sums. So this node reads the
    bytes back off disk through `app.event_log`'s own directory and runs the DOCUMENTED
    command over them: an over-limit `autonomy_task` embed must be visible to the same
    tool a human would use to ask the question.
    """
    _block, cmd = _soft_limit_doc()
    over = SE.UNCAPPED_EMBED_SOFT_LIMIT + 1
    out = _record(SD.ROUTE_AUTONOMY_TASK, over, kind="roundtrip")
    assert out["rows"], "the recorder wrote no row at all — nothing to round-trip"
    # A row on EACH uncapped route: the command's output is then required to carry both
    # keys, so a program that silently dropped one route cannot pass on the one left.
    _record(SD.ROUTE_WORKER_PROMPT, over, kind="roundtrip-worker")
    assert out["rows"][0]["data"]["over_soft_limit"] is True, out["rows"]

    proc = subprocess.run([sys.executable, "-"], input=cmd, capture_output=True,
                          text=True, timeout=300,
                          env={"SKILL_EMBED_LOG_GLOB":
                               str(Path(EL.EVENT_LOGS_DIR) / "*.events.jsonl"),
                               "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, f"the documented command failed: {proc.stderr[-600:]}"
    got = json.loads(proc.stdout)

    # Independent count over the same files, so "the command saw my row" is not
    # inferred from a number that was already large: this scratch log directory holds
    # rows from every other node in this file too, capped routes included, and the
    # command must agree with a plain count of those files route by route.
    import glob as _glob
    mine = {r: 0 for r in UNCAPPED}
    for f in _glob.glob(str(Path(EL.EVENT_LOGS_DIR) / "*.events.jsonl")):
        for line in Path(f).read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = row.get("data") or {}
            if row.get("event") == "skill.embedded" and d.get("route") in mine:
                mine[d["route"]] += 1
    assert set(got) == set(UNCAPPED), sorted(got)
    for route in UNCAPPED:
        assert got[route]["n"] == mine[route], (route, got[route], mine[route])
    assert got[SD.ROUTE_AUTONOMY_TASK]["max"] >= over, got


# --------------------------------------------------------------------------- #
# clause 3: the stamp is scoped to the uncapped routes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("route", UNCAPPED, ids=list(UNCAPPED))
def test_an_embed_at_or_below_the_limit_carries_neither_key(route):
    """Below the line stays ordinary, and the boundary itself is pinned.

    Both edges are driven because `>` and `>=` differ at exactly one value, and a
    ceiling whose boundary a reader cannot reproduce is worse than none: an embed of
    EXACTLY the limit is not over it.
    """
    for n in (1, SE.UNCAPPED_EMBED_SOFT_LIMIT):
        out = _record(route, n, kind=f"at-or-below-{route}-{n}")
        assert "over_soft_limit" not in out["rec"], (n, out["rec"])
        assert "soft_limit" not in out["rec"], (n, out["rec"])
        assert out["rows"], (n, out["rows"])
        assert "over_soft_limit" not in out["rows"][0]["data"], (n, out["rows"])
        assert "soft_limit" not in out["rows"][0]["data"], (n, out["rows"])


@pytest.mark.parametrize("route", [SD.ROUTE_PREFETCH, SD.ROUTE_PREFETCH_EXCERPT],
                         ids=["prefetch", "prefetch_excerpt"])
def test_a_capped_route_above_the_same_number_is_not_stamped(route):
    """A route with a hard cut does not need a soft line, and stamping it would lie.

    `prefetch` clips at `app/prefetch.py`'s `SKILL_BODY_MAX` before booking, so every
    number it reports is already bounded by code — `over_soft_limit` there would report
    a condition that cannot recur and would bury the uncapped traffic the constant
    exists to count. Both capped names are driven, not just the one that dominates the
    log.
    """
    out = _record(route, SE.UNCAPPED_EMBED_SOFT_LIMIT + 1, kind=f"capped-{route}")
    assert "over_soft_limit" not in out["rec"], out["rec"]
    assert "soft_limit" not in out["rec"], out["rec"]
    assert "over_soft_limit" not in out["rows"][0]["data"], out["rows"]
    assert "soft_limit" not in out["rows"][0]["data"], out["rows"]


def test_only_these_two_routes_are_treated_as_uncapped():
    """The set is the item's, and drift in it is otherwise silent.

    A third name would start stamping a route that has a cut; a dropped one would stop
    counting the traffic the owed ruling waits on. Asserted on the set itself and
    against the constants' owner, so the recorder cannot quietly rename or widen it and
    so the literals here cannot drift from `app/harness/skill_dispatch.py`.
    """
    assert set(SE.UNCAPPED_ROUTES) == set(UNCAPPED), SE.UNCAPPED_ROUTES
    assert SE.UNCAPPED_ROUTES == (SD.ROUTE_AUTONOMY_TASK, SD.ROUTE_WORKER_PROMPT), (
        "the recorder's uncapped set must name the same routes as its owner")


def test_both_uncapped_routes_book_the_whole_skill_body_as_embedded_chars():
    """The number the stamp compares is the whole prompt body, on both routes.

    Clause 3's scope is only meaningful if what arrives at the recorder is the whole
    skill: an `embedded_chars` that reported a clipped excerpt would make the stamp
    under-count silently, and the call sites are the only place that is decided. Read
    from their source, because they are one-argument expressions with no branch to
    drive.
    """
    auton = (_REPO_ROOT / "app" / "autonomy.py").read_text(encoding="utf-8")
    assert "embedded_chars=len(skill_content)" in auton, (
        "app/autonomy.py must book the whole skill body; if it starts booking a clip, "
        "UNCAPPED_ROUTES is wrong and this node is the alarm")
    worker = (_REPO_ROOT / "workers" / "sources" / "deep_research.py").read_text(
        encoding="utf-8")
    assert "embedded_chars=len(skill)" in worker, (
        "the worker route must book the whole skill body, as `build_skill_prompt`'s own "
        "docstring says it does")
    common = (_REPO_ROOT / "workers" / "sources" / "_common.py").read_text(
        encoding="utf-8")
    # The body is one element of a `"\n".join([...])`, not a bare `return`, so the pin
    # is the element plus the sentence that says what it does: `skill_text` appears in
    # the joined list unmodified, and the docstring states it goes in whole.
    assert "\n        skill_text,\n" in common and "goes in whole, uncapped" in common, (
        "`build_skill_prompt` no longer splices the text whole; clause 4's byte-identity "
        "node has to be rewritten to the new shape, not dropped")


# --------------------------------------------------------------------------- #
# clause 4: accounting only — the prompt does not move by one byte
# --------------------------------------------------------------------------- #

def test_the_autonomy_prompt_with_a_worst_case_skill_is_unchanged_by_the_accounting():
    """`_build_task_prompt` still ships the whole body of a 69,639-char skill.

    The shape is asserted exactly — hint, header, blank line, body, nothing after it —
    so an edit that truncated the body, appended a warning line, or wrapped it in
    anything is caught, not only one that shrank it.
    """
    from app import autonomy as A

    body = "w" * WORST_EMBED_CHARS
    assert len(body) > SE.UNCAPPED_EMBED_SOFT_LIMIT, (
        "this node's subject is an over-limit skill; if the limit is ever raised above "
        "the worst embed on record, raise this number with it rather than weaken it")
    prompt = A._build_task_prompt({"id": 42, "name": "autonomy-data-pipeline"}, body)
    header = ('[SYSTEM: You are executing autonomy task #42: "autonomy-data-pipeline". '
              "Follow the skill instructions below.]")
    assert header in prompt
    assert prompt.endswith(f"\n\n{body}"), (
        "the skill body must still be the last thing in the prompt, whole")
    assert prompt.count(body) == 1
    assert prompt == prompt[:prompt.index(header)] + header + f"\n\n{body}", (
        "nothing may sit between the header and the body, and nothing may follow it")


def test_the_worker_prompt_returns_the_skill_whole_and_unchanged():
    """`build_skill_prompt` still ships the body whole, between its header and its task.

    `workers/sources/_common.py:77` takes `(skill_text, *, job, task_block)` and joins
    header, blank, body, blank, task_block. The exact shape is pinned, so an edit that
    clipped the body or added a line beside it fails here — the accounting change this
    item makes is not allowed to touch any of it.
    """
    from workers.sources._common import build_skill_prompt

    body = "x" * (SE.UNCAPPED_EMBED_SOFT_LIMIT + 1)
    task = "Investigate the thing and write the note."
    prompt = build_skill_prompt(body, job="deep-research", task_block=task)
    assert prompt == (
        '[SYSTEM: You are running the "deep-research" worker job. Follow the skill '
        "below, applied to the task at the end. Work autonomously and do not ask for "
        f'confirmation.]\n\n{body}\n\n{task}'), prompt[:200]
    assert prompt.count(body) == 1, "the body appears exactly once, whole"


def test_record_skill_embed_raises_nothing_on_an_over_limit_uncapped_embed():
    """No truncation, no refusal, no raise: an over-limit embed is booked and returned.

    Every route the recorder knows is driven at the same over-limit size, on both sides
    of `truncated`, and each call must return a dict. The accounting is loud in the log
    and silent everywhere else, because the module's own docstring says the shape of a
    bad embed is a human's call and #2272 defers it.
    """
    over = SE.UNCAPPED_EMBED_SOFT_LIMIT + 1
    for route in ALL_ROUTES:
        for trunc in (False, True):
            rec = SE.record_skill_embed(
                route=route, skill="s", embedded_chars=over, source_chars=over,
                truncated=trunc, session_id=_sid(f"loud-{route}-{trunc}"))
            assert isinstance(rec, dict), (route, trunc, rec)
            assert rec["embedded_chars"] == over, (route, trunc, rec)


def test_the_module_offers_no_switch_that_could_cap_or_spill_those_routes():
    """#2272's clause 5 is the half that must stay unimplemented, so it is pinned.

    Whether to cap or spill the two routes is a separate ruling made on the traffic this
    records, so a flag defaulting to off would pre-empt it quietly — "off today" is
    exactly how #2134's shadow flag reached "flip it" with no eval behind it. The check
    is on top-level names against a word list, with the names this module legitimately
    carries on an allow-list, so the node cannot be satisfied by renaming a switch into
    something vaguer.
    """
    tree = ast.parse((_REPO_ROOT / "app" / "skill_embed.py").read_text(encoding="utf-8"))
    top: set[str] = set()
    for n in tree.body:
        if isinstance(n, ast.Assign):
            top |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top.add(n.name)
    allowed = {"UNCAPPED_EMBED_SOFT_LIMIT", "UNCAPPED_ROUTES",
               "record_skill_embed", "record_context_skills"}
    switch_words = {"enforce", "cap", "spill", "hard", "truncate", "truncation",
                    "enabled", "shadow", "switch", "flag", "allow", "block"}
    offenders = {
        name for name in top
        if name not in allowed and not name.startswith("_")
        and (switch_words & set(name.lower().split("_")))
    }
    assert not offenders, (
        f"app/skill_embed.py grew a name that reads like a switch: {sorted(offenders)}"
        " — capping or spilling the two uncapped routes is a separate ruling, and a"
        " flag that exists will be flipped (#2134)")


# --------------------------------------------------------------------------- #
# clause 5: the doc half
# --------------------------------------------------------------------------- #

def test_the_size_paragraph_says_the_line_ceiling_does_not_bound_the_uncapped_routes():
    """The #624 paragraph must state the non-bound as a measured pair.

    Today it says size is a cost only on the uncapped routes and then describes a
    100-line ceiling that measures something else, which reads as though that cost were
    bounded. The correction is the inversion, and numbers are the only form that carries
    it: `nightly-reflection-knowledge-write` at 88 body lines embedded 25,856 chars
    while `backlog-triage` at 107 lines embedded 9,977 — 19 MORE lines for a third of
    the cost, so no line count can order these two by what they cost a prompt.
    """
    text = (_REPO_ROOT / "architecture" / "skills.md").read_text(encoding="utf-8")
    i = text.find("Size is a cost only on the uncapped routes")
    assert i >= 0, "the #624 SIZE paragraph must still be in architecture/skills.md"
    para = text[i:text.find("\n\n", i)]
    for needle in ("88", "25,856", "107", "9,977",
                   "nightly-reflection-knowledge-write", "backlog-triage"):
        assert needle in para, f"{needle!r} missing from the SIZE paragraph: {para}"
    assert "does not bound" in para, para
    assert "100" in para and "line" in para, (
        f"the sentence has to name the 100-line ceiling it is limiting: {para}")


def test_the_size_paragraph_carries_its_counts_as_dated_and_points_at_the_sweep():
    """#2334 clause 2: the deferral stays a person's, and its EVIDENCE stays live.

    The sentence that deferred the library-wide ceiling also froze a count in prose —
    "106 of 197 skills were over the cap on 2026-10-01" — and nothing re-checks a
    sentence, so by the sweep re-run on 2026-10-07 the page was two counts wrong while
    reading as the authority for a ruling that has to be made on current numbers. The
    same pair sat quoted in `vault_round`'s comment and in
    `tests/test_vault_round_skill_gate.py`'s docstring: three frozen quotations, one
    measurement. This node holds the two halves of the fix. The paragraph must still
    DEFER to a person — #624's owed entry reserves the ceiling to one, and the node
    below the rail in `test_vault_round_skill_gate.py` refuses a round that deletes
    the sentence without a ruling — and it must send the reader to where the pair is
    PRINTED rather than carrying it: the `### SIZE` summary line of the skill-lint
    report, written by `skill_lint.render_size`, which prints the cap it counted
    against in its own heading. And any count it does keep has to arrive dated, because
    an undated count is a claim nobody will re-run.
    """
    text = (_REPO_ROOT / "architecture" / "skills.md").read_text(encoding="utf-8")
    i = text.find("library-wide, whether SIZE becomes a lint failure")
    assert i >= 0, "the library-wide SIZE deferral has left the SIZE paragraph entirely"
    seg = text[i:text.find("What that ceiling does NOT do", i)]
    assert "person's call" in seg, (
        "the ceiling is a person's ruling; the paragraph may not quietly take it: " + seg)
    assert "#2334" in seg, "the open item that owns the ruling has to be named: " + seg
    assert "MAX_BODY_LINES" in seg, (
        "the paragraph has to say which constant the ruling would move: " + seg)
    assert "autonomy/skill-lint-report.md" in seg and "### SIZE" in seg, (
        f"it must name where the live over-cap count and p90 are printed: {seg}")
    assert "render_size" in seg, "and the function that prints them: " + seg
    # The 2026-10-01 figure survives as a DATED measurement, not as the state of the
    # corpus, and the date it was overtaken by is named beside it.
    assert "2026-10-01" in seg and "2026-10-07" in seg, (
        f"the frozen count must read as dated and superseded: {seg}")
    assert "DATED" in seg.upper(), f"say of the figure that it is dated: {seg}"
    j = seg.find("106 of 197")
    assert j >= 0 and "2026-10-01" in seg[j:j + 260], (
        f"the 2026-10-01 count must carry its own date beside it: {seg[j - 80:j + 260]}")
