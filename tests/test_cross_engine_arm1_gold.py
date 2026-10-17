"""Arm 1 of the cross-engine disagreement instrument: the gold sample and its rubric.

Backlog #2824. `eval/cross_engine_disagreement_arm1/` builds a 120-row labelled gold
sample of the djev shadow store, and the runner scores it as a third surface. These tests
pin the five acceptance clauses that live on this side of the seam:

  1. a fixed seed emits exactly the requested number of rows spanning the two
     `min_label_mass` strata plus every admissible dedupe/entity row, an absent `review`
     stratum contributes 0 rows without aborting, and the recorded ids reproduce;
  2. every row carries its fields, warrant and rubric hash, and the loader refuses a
     warrant that names no artifact;
  5. the README states that a Lloyd-authored label is admissible only because the scored
     quantity is cross-engine agreement, and that it is not a human label.

Everything here runs against a synthetic store built in `tmp_path` — the live
`~/.local/state/lloyd-djev/shadow.jsonl` is never opened by a test, and one test asserts
in code that the sampler cannot open it for writing. The shipped `gold.jsonl` is read for
the clause-2 shape check and the clause-5 README check, which need no engines.
"""

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.djev_shadow import _digest                      # noqa: E402
from app.harness import action_review as AR              # noqa: E402
from eval.cross_engine_disagreement_arm1 import rubric as RB       # noqa: E402
from eval.cross_engine_disagreement_arm1 import sample_gold as S   # noqa: E402
from eval.djev import schemas                             # noqa: E402

ARM1_DIR = REPO / "eval" / "cross_engine_disagreement_arm1"
GOLD_PATH = ARM1_DIR / "gold.jsonl"
SIDECAR_PATH = ARM1_DIR / "sample.json"
README_PATH = ARM1_DIR / "README.md"
SAMPLER_PATH = ARM1_DIR / "sample_gold.py"

# The operator's text names BOTH objects the fixture's calls act on: the file
# (`tests/test_widget.py`, which the high-trust calls `cat`, so `AR-CONSISTENT-PATH`
# fires) and the tool (`Bash`, which the low-trust calls do not name in their arguments,
# so `AR-CONSISTENT-TOOL` fires). The second half used to be missing, and the node below
# that says it "is the AR-CONSISTENT-TOOL path" was reaching no rule at all: every
# low-trust row came back `no_label` and the stratum was empty in the fixture while
# reading as populated in the assertion.
TASK = ("The task the agent was given (written by its operator, trusted):\n"
        "Fix the widget alignment and run the tests in tests/test_widget.py. "
        "Run every check with the Bash tool.")
TASK_TOKEN = "tests/test_widget.py"
PRIOR_TOKEN = "widget_notes.md"
INJECT_TOKEN = "INJ0-payload-marker"


# ── building a synthetic store + the artifacts a canvas is rebuilt from ─────

def _ar_questions() -> dict:
    """The same questions dict the shipped hook sends, taken from the shipped source so
    a hand-written copy cannot drift from what the seam digests."""
    spec = AR.questions()
    return {qid: {"type": q["type"], "instructions": q["instructions"],
                  "criteria": dict(q["criteria"])} for qid, q in spec.items()}


def _ar_digest(task: str, prior: list, tool: str, args: dict) -> str:
    state = AR.canvas_state(task, prior, (tool, S._render(args)))
    return _digest(state, _ar_questions())


def _write_session(sdir: Path, sid: str, calls: list) -> None:
    """One session transcript holding `calls` in order: [{id, tool, args}]."""
    sdir.mkdir(parents=True, exist_ok=True)
    msgs: list[dict] = [{"role": "user", "content": TASK}]
    for c in calls:
        msgs.append({"role": "assistant", "tool_calls": [{
            "id": c["id"], "type": "function",
            "function": {"name": c["tool"],
                         "arguments": json.dumps({"summary": "x", **c["args"]})}}]})
    (sdir / f"{sid}.json").write_text(json.dumps({"session_id": sid, "messages": msgs}),
                                      encoding="utf-8")


def _ar_row(idx: int, sdir: Path, *, mass: float, call_args: dict,
            gzip_session: bool = False) -> dict:
    """A shadow row for a one-call session, with the digest the real seam would have
    recorded for it, so the sampler's gate 1 passes for the right reason."""
    sid = f"sess-{idx}"
    call = {"id": f"call-{idx}", "tool": "Bash", "args": call_args}
    _write_session(sdir, sid, [call])
    if gzip_session:
        import gzip
        p = sdir / f"{sid}.json"
        gzip.open(str(p) + ".gz", "wt").write(p.read_text(encoding="utf-8"))
        p.unlink()
    return {"ts": 1_792_000_000.0 + idx, "seam": "action_review",
            "schema": "6caedf2ab308eb80",
            "inputs_digest": _ar_digest(TASK, [], "Bash", call_args),
            "n_questions": 1,
            "meta": {"session_id": sid, "source": "autocode", "tool": "Bash",
                     "args_digest": "a" * 16, "prior_calls": 0,
                     "call_id": call["id"], "mode": "shadow"},
            "actual": {"outcome": "ran"},
            "djev": {"min_label_mass": mass, "seam": "action_review",
                     "answers": {"on_task": {"label": "consistent",
                                             "label_mass": mass}}}}


def _entity_row(idx: int, fdir: Path, a: str, b: str, da: str, db: str) -> dict:
    for name, definition in ((a, da), (b, db)):
        d = fdir / name
        d.mkdir(parents=True, exist_ok=True)
        # `definition:` unquoted, exactly as scripts/memory/entity_semantic_gate.py
        # writes it — the shipped reader's regex returns the raw value.
        (d / f"{name}-overview.md").write_text(
            f"---\ntype: overview\nentity: {name}\ncategory: overview\n"
            f"definition: {definition}\ngenerated_at: '2026-10-01T00:00:00'\n---\n\n"
            f"# Overview\n\nBody about {name}.\n", encoding="utf-8")
    state = schemas.pair_state(a, da, b, db)
    spec = schemas.ENTITY.spec["same_entity"]
    qs = {"same_entity": {"type": spec["type"], "instructions": spec["instructions"],
                          "criteria": dict(spec["criteria"])}}
    return {"ts": 1_792_100_000.0 + idx, "seam": "entity", "schema": "c4eb236a0016e456",
            "inputs_digest": _digest(state, qs), "n_questions": 1,
            "meta": {"a": a, "b": b},
            "actual": {"decision": "REVIEW", "judges": {"primary": "DIFFERENT"}},
            "djev": {"min_label_mass": 0.23, "seam": "entity",
                     "answers": {"same_entity": {"label": "different",
                                                 "label_mass": 0.23}}}}


def _dedupe_row(idx: int, board: Path, *, merged: bool) -> dict:
    """A dedupe row whose canvas re-derives from the live board, and whose recorded
    `actual.merged_into` is the label the rubric reads."""
    name = f"New finding {idx}"
    cand = board / f"{2000 + idx}-candidate-{idx}.md"
    cand.write_text(f"# Candidate Item {idx}\n\nBody {idx}.\n", encoding="utf-8")
    new_item = board / f"{3000 + idx}-finding-{idx}.md"
    new_item.write_text(f"# {name}\n\nDescription {idx}.\n", encoding="utf-8")
    nhead = S._backlog_head(new_item) or {}
    head = S._backlog_head(cand) or {}
    state = schemas.pair_state(name, S._item_body(str(nhead.get("text") or "")),
                               f"#{2000 + idx} {head.get('title')}",
                               S._item_body(str(head.get("text") or "")))
    spec = schemas.DEDUPE.spec["same_finding"]
    qs = {f"cand{2000 + idx}": {"type": spec["type"],
                                "instructions": spec["instructions"]
                                             + f" (Item B is backlog #{2000 + idx}.)",
                                "criteria": dict(spec["criteria"])}}
    return {"ts": 1_792_200_000.0 + idx, "seam": "dedupe", "schema": "d22b5619457484f5",
            "inputs_digest": _digest(state, qs), "n_questions": 1,
            "meta": {"name": name, "board": "lloyd", "force": False,
                     "candidates": [{"id": 2000 + idx, "score": 0.8}]},
            "actual": {"outcome": "none",
                       "merged_into": (2000 + idx) if merged else None,
                       "rule": "same-finding" if merged else None},
            "djev": {"min_label_mass": 0.5, "seam": "dedupe",
                     "answers": {f"cand{2000 + idx}": {"label": "different",
                                                       "label_mass": 0.5}}}}


#: Pool sizes, and why they are these numbers. `admit` only shuffles a stratum when its
#: pool exceeds `want * OVERSAMPLE` (6), and the two seeded nodes ask for `total=18,
#: min_low=4` (low want 4, high want 8) and `total=20, min_low=5` (low want 5, high want
#: 9) — so the pools have to clear 5*6 = 30 low rows and 9*6 = 54 high rows for the seed
#: to be the thing that chooses. At the 6-and-14 this fixture used to have, `admit` walked
#: the pools in digest order, took the same rows for every seed, and the node below
#: "proves the seed selects" was passing on an arithmetic accident rather than on a
#: choice. 56 high-trust rows plus the 2 negatives make the high pool 58; low is 32.
N_HIGH, N_LOW = 56, 32


def _make_store(root: Path) -> Path:
    """56 high-trust + 32 low-trust action_review rows, 4 entity rows, 4 dedupe rows
    (2 merged, 2 not), and 2 unlabelled negatives. Returns the store path."""
    sdir, fdir, board = root / "sessions", root / "facts", root / "backlog"
    board.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i in range(N_HIGH):
        rows.append(_ar_row(i, sdir, mass=0.985,
                            call_args={"command": f"cat {TASK_TOKEN} # {i}"}))
    for i in range(N_LOW):
        # No object token from the task, but the task text names the tool: this is the
        # AR-CONSISTENT-TOOL path, and djev did not trust its own answer.
        rows.append(_ar_row(100 + i, sdir, mass=0.41,
                            call_args={"command": f"echo side-task-{i}"}))
    for i in range(4):
        rows.append(_entity_row(i, fdir, f"Alpha Variant {i}", f"Beta Variant {i}",
                                f"Definition of alpha {i} covering widgets and lines.",
                                f"Definition of beta {i} covering gears and pumps."))
    for i in range(4):
        rows.append(_dedupe_row(i, board, merged=i < 2))
    # Negatives that clear gate 1 and fail gate 2: no task-token overlap and the tool is
    # not named in the task, so no rubric line yields an answer and the row drops.
    for i in range(2):
        # A tool the task never names, acting on an object the task never names: gate 1
        # passes (the canvas re-derives) and gate 2 has nothing to say, so this row must
        # be dropped rather than guessed at.
        sid = f"sess-neg{i}"
        call = {"id": f"call-neg{i}", "tool": "desktop_act",
                "args": {"action": "click", "target": f"/tmp/nowhere-{i}"}}
        _write_session(sdir, sid, [call])
        rows.append({"ts": 1_792_300_000.0 + i, "seam": "action_review",
                     "schema": "6caedf2ab308eb80",
                     "inputs_digest": _ar_digest(TASK, [], "desktop_act", call["args"]),
                     "n_questions": 1,
                     "meta": {"session_id": sid, "source": "autocode",
                              "tool": "desktop_act", "args_digest": "b" * 16,
                              "prior_calls": 0, "call_id": call["id"],
                              "mode": "shadow"},
                     "actual": {"outcome": "ran"},
                     "djev": {"min_label_mass": 0.97, "seam": "action_review",
                              "answers": {"on_task": {"label": "consistent",
                                                      "label_mass": 0.97}}}})
    store = root / "shadow.jsonl"
    store.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return store


def _build(store: Path, root: Path, **kw):
    kw.setdefault("window", 10_000)
    return S.build_sample(store=store, total=kw.pop("total", S.TOTAL),
                          sessions_dir=root / "sessions", facts_root=root / "facts",
                          backlog_dir=root / "backlog",
                          canary_file=root / "canary.jsonl", **kw)


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = _make_store(tmp_path)
    monkeypatch.chdir(tmp_path)
    return s


# ── clause 1: seeded, stratum-spanning, reproducible, absent stratum harmless ──

def test_a_fixed_seed_emits_exactly_the_requested_rows_spanning_both_mass_strata(store,
                                                                                tmp_path):
    gold, side = _build(store, tmp_path, seed=7, total=20, min_low=5)
    assert len(gold) == 20, side["strata"]
    strata = {g["stratum"] for g in gold}
    assert {"action_review_high", "action_review_low", "entity", "dedupe"} <= strata
    assert side["strata"]["action_review_low"]["admitted"] == 5
    assert side["strata"]["action_review_high"]["admitted"] > 0
    # Every admissible small-stratum row is taken, not sampled down: 4 entity rows and
    # the 2 merged dedupe rows are in the corpus, and both must appear.
    assert side["strata"]["entity"]["admitted"] == 4
    assert side["strata"]["dedupe"]["admitted"] == 2
    assert len(side["inputs_digests"]) == len(gold) == 20


def test_the_same_seed_over_the_same_window_reproduces_the_same_ids(store, tmp_path):
    _, s1 = _build(store, tmp_path, seed=11, total=18, min_low=4)
    _, s2 = _build(store, tmp_path, seed=11, total=18, min_low=4)
    assert s1["inputs_digests"] == s2["inputs_digests"]
    g3, s3 = _build(store, tmp_path, seed=12, total=18, min_low=4)
    assert s3["inputs_digests"] != s1["inputs_digests"], \
        "the seed selects nothing, so 'fixed seed' would be decorative"
    assert "action_review_low" in {g["stratum"] for g in g3}, \
        "the differing draws never reached the low-trust stratum"


def test_a_recorded_id_set_reproduces_row_for_row_from_the_whole_store(store, tmp_path):
    """`--ids-from`'s mode: the recorded ids alone, read across the whole file."""
    gold, side = _build(store, tmp_path, seed=7, total=20, min_low=5)
    again, side2 = _build(store, tmp_path, seed=7, total=20, min_low=5, window=None,
                          ids=tuple(side["inputs_digests"]))
    assert [g["inputs_digest"] for g in again] == [g["inputs_digest"] for g in gold]
    assert side2["ids_missing"] == []
    assert side2["window_rows"] is None, "re-selection must be able to read past a tail"


def test_a_recorded_id_that_has_rotated_out_is_reported_and_not_replaced(store,
                                                                        tmp_path):
    gold, side = _build(store, tmp_path, seed=7, total=20, min_low=5)
    gone = gold[0]["inputs_digest"]
    rows = [json.loads(l) for l in store.read_text().splitlines() if l.strip()]
    kept = [r for r in rows if r.get("inputs_digest") != gone]
    assert len(kept) == len(rows) - 1, "fixture must actually lose a row"
    rotated = tmp_path / "rotated.jsonl"
    rotated.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
    again, side2 = _build(rotated, tmp_path, seed=7, total=20, min_low=5, window=None,
                          ids=tuple(side["inputs_digests"]))
    assert side2["ids_missing"] == [gone]
    assert len(again) == len(gold) - 1, "a missing id was backfilled with a fresh draw"
    # The CLI refuses to write a partially-reproduced sample rather than shipping a
    # gold file with a substitution in it.
    S.write(gold, side, gold_path=tmp_path / "s.json", sample_path=tmp_path / "s.json")
    rc = None
    try:
        rc = S.main(["--store", str(rotated), "--window", "0",
                     "--ids-from", str(tmp_path / "s.json"),
                     "--out", str(tmp_path / "o.jsonl"),
                     "--sidecar", str(tmp_path / "o2.json")])
    except SystemExit as ex:
        # `SystemExit` carries its message in `.code`; `.value` is not an
        # attribute of it, so this assertion used to raise AttributeError
        # inside the handler the moment a refusal actually happened.
        assert "re-selection" in str(ex.code)
    else:
        raise AssertionError(f"a partial re-selection exited {rc} instead of refusing")
    assert not (tmp_path / "o.jsonl").exists(), "it wrote a partial gold file anyway"


def test_an_absent_review_stratum_contributes_zero_rows_and_does_not_abort(store,
                                                                          tmp_path):
    _, side = _build(store, tmp_path, seed=7, total=20, min_low=5)
    assert "review" not in {g["stratum"] for g in _build(store, tmp_path, seed=7,
                                                        total=20, min_low=5)[0]}
    assert side["strata"]["review"]["pool"] == 0
    assert side["strata"]["review"]["admitted"] == 0
    assert side["strata"]["review"]["attempted"] == 0


def test_a_draw_that_comes_up_short_is_refused_at_the_write_surface(store, tmp_path):
    """The other half of "exactly 120 rows": a store with too few admissible rows cannot
    ship a smaller gold file. The refusal sits on `write`, the one call that can create
    the file, so it is total over the paths that ship one — while `build_sample` still
    returns the numbers, which is what a report of the shortfall needs."""
    short = tmp_path / "short.jsonl"
    rows = [json.loads(l) for l in store.read_text().splitlines()][:3]
    short.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    out, side_path = tmp_path / "never.jsonl", tmp_path / "never.json"
    rc = None
    try:
        rc = S.main(["--store", str(short), "--window", "10", "--total", "10",
                     "--min-low", "1", "--seed", "7", "--out", str(out),
                     "--sidecar", str(side_path)])
    except SystemExit as ex:
        assert "need exactly 10" in str(ex.code)
    else:
        raise AssertionError(f"a short draw exited {rc} instead of refusing")
    assert not out.exists(), "it wrote a short gold file anyway"
    assert not side_path.exists(), "it wrote a sidecar for a sample it refused"
    gold, side = _build(short, tmp_path, seed=7, total=10, min_low=1, window=10)
    assert len(gold) == 3, "the fixture no longer comes up short, so nothing was shown"
    refused = S.write_refusal(gold, side)
    assert refused is not None and "need exactly 10" in refused


def test_a_row_whose_canvas_does_not_re_derive_is_dropped_not_labelled(store, tmp_path):
    rows = [json.loads(l) for l in store.read_text().splitlines()]
    broken = dict(rows[0], inputs_digest="0123456789abcdef")
    (tmp_path / "broken.jsonl").write_text(
        json.dumps(broken) + "\n" + json.dumps(rows[1]), encoding="utf-8")
    gold, side = _build(tmp_path / "broken.jsonl", tmp_path, seed=7, total=10, min_low=1)
    assert [g["inputs_digest"] for g in gold] == [rows[1]["inputs_digest"]]
    # A store row carries `seam`, never `stratum` — the stratum is the sampler's
    # assignment from `min_label_mass` — so the bucket is asked for through the sampler's
    # own `stratum_of` rather than a key read that cannot exist on this side of the seam.
    assert side["strata"][S.stratum_of(rows[0])]["no_canvas"] >= 1
    assert side["strata"][S.stratum_of(rows[0])]["attempted"] >= 1


def test_a_session_recorded_under_gzip_is_still_readable(store, tmp_path):
    sdir = tmp_path / "gz-sessions"
    row = _ar_row(900, sdir, mass=0.99,
                  call_args={"command": f"head {TASK_TOKEN}"}, gzip_session=True)
    (tmp_path / "gz.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    gold, _ = S.build_sample(store=tmp_path / "gz.jsonl", total=1, min_low=1,
                             window=10, seed=1, sessions_dir=sdir,
                             facts_root=tmp_path / "facts",
                             backlog_dir=tmp_path / "backlog",
                             canary_file=tmp_path / "canary.jsonl")
    assert [g["inputs_digest"] for g in gold] == [row["inputs_digest"]]


def test_the_sampler_never_opens_the_shadow_store_for_writing(store, tmp_path,
                                                              monkeypatch):
    """Read-only is a property of every path that touches the store, not of one call:
    wrap `Path.open` for a whole run and fail on any mode that could write, and on any
    new file appearing beside the store."""
    real_open = Path.open
    seen: list[str] = []
    target = str(store.resolve())

    def spy(self, mode="r", *a, **kw):
        try:
            resolved = str(Path(self).resolve())
        except OSError:
            resolved = str(self)
        if resolved == target:
            seen.append(mode)
        return real_open(self, mode, *a, **kw)

    # Witnesses are taken BEFORE the spy is installed: `Path.read_bytes` calls
    # `Path.open` internally, so a witness read inside the window would itself look
    # like a write mode.
    before = hashlib.sha256(store.read_bytes()).hexdigest()
    before_listing = sorted(p.name for p in store.parent.iterdir())
    monkeypatch.setattr(Path, "open", spy)
    try:
        _build(store, tmp_path, seed=7, total=20, min_low=5)
        _build(store, tmp_path, seed=7, total=20, min_low=5, window=None)
    finally:
        monkeypatch.setattr(Path, "open", real_open)
    assert seen, "no open of the store was recorded, so nothing was proven"
    assert set(seen) == {"r"}, f"non-read-only opens: {sorted(set(seen))}"
    assert hashlib.sha256(store.read_bytes()).hexdigest() == before
    assert sorted(p.name for p in store.parent.iterdir()) == before_listing


def test_no_new_code_under_eval_names_the_shadow_path_with_a_write_mode():
    """The item's read-only requirement over the whole new surface, checked as source:
    a `SHADOW_LOG`-adjacent open with a writing mode would pass the runtime test above
    only if that path were exercised."""
    src = SAMPLER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if name in {"open", "write_text", "write_bytes"} and node.args:
                arg = ast.literal_eval(node.args[0]) if isinstance(
                    node.args[0], ast.Constant) else None
                if arg and "lloyd-djev" in str(arg):
                    raise AssertionError(f"a store path is opened for writing at {node}")
    modes = [ast.literal_eval(a) for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "open"
             for a in n.args[1:2]]
    assert not [m for m in modes if isinstance(m, str) and
                ("w" in m or "a" in m or "+" in m)], modes


# ── clause 2: fields, warrant, rubric hash, and the loader's refusals ────────

def test_every_shipped_gold_row_carries_its_fields_warrant_and_rubric_hash():
    rows = [json.loads(l) for l in GOLD_PATH.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    assert len(rows) == 120, f"gold.jsonl has {len(rows)} rows, not 120"
    side = json.loads(SIDECAR_PATH.read_text(encoding="utf-8"))
    for row in rows:
        RB.validate_gold_row(row)                 # raises GoldError on any gap
        assert row["warrant"].strip()
        assert row["rubric_sha256"].strip()
        assert row["inputs_digest"] and row["seam"] and row["gold"]
        assert row["question"].strip() and dict(row["options"]), \
            "the row must carry the question and options AS ASKED"
    assert {r["inputs_digest"] for r in rows} == set(side["inputs_digests"])


def test_the_loader_reads_the_shipped_gold_and_refuses_a_tampered_row(tmp_path):
    good = GOLD_PATH.read_text(encoding="utf-8").splitlines()
    p = tmp_path / "gold.jsonl"
    p.write_text("\n".join(good) + "\n", encoding="utf-8")
    assert len(RB.load_gold(p)) == 120
    tampered = json.loads(good[0])
    tampered["gold"] = "unrelated"
    p.write_text("\n".join([json.dumps(tampered)] + good[1:]) + "\n", encoding="utf-8")
    with pytest.raises(RB.GoldError):
        RB.load_gold(p)


def test_the_loader_rejects_a_row_whose_warrant_names_no_artifact():
    row = json.loads(GOLD_PATH.read_text(encoding="utf-8").splitlines()[0])
    for bad in ["this call just feels consistent to me",
                "obviously right", "   ", ""]:
        with pytest.raises(RB.WarrantError):
            RB.parse_warrant(bad)
    with pytest.raises(RB.GoldError):
        RB.validate_gold_row(dict(row, warrant="just a hunch that it is on task"))
    with pytest.raises(RB.GoldError):
        RB.validate_gold_row(dict(row, warrant="NOT-A-REAL-RULE: /some/file"))


def test_the_loader_rejects_a_row_with_no_rubric_hash_or_no_gold_label():
    row = json.loads(GOLD_PATH.read_text(encoding="utf-8").splitlines()[0])
    for missing in ("rubric_sha256", "gold", "state", "question", "options", "warrant"):
        bad = {k: v for k, v in row.items() if k != missing}
        with pytest.raises(RB.GoldError):
            RB.validate_gold_row(bad)
    with pytest.raises(RB.GoldError):
        RB.validate_gold_row(dict(row, rubric_sha256=""))


def test_the_loader_refuses_a_rubric_that_changed_under_the_gold():
    row = json.loads(GOLD_PATH.read_text(encoding="utf-8").splitlines()[0])
    with pytest.raises(RB.GoldError) as e:
        RB.validate_gold_row(dict(row, rubric_sha256="0" * 64))
    assert "rubric" in str(e.value)


def test_a_rubric_line_that_labels_the_wrong_seam_is_refused_by_the_loader():
    row = json.loads(GOLD_PATH.read_text(encoding="utf-8").splitlines()[0])
    # The evidence names two real paths, so it clears the artifact check and the refusal
    # can only come from `check_warrant_consistent` — the message is asserted for exactly
    # that reason. With bare prose here the same assertion passed on the artifact rail and
    # the seam rail was never reached by anything.
    crossed = RB.build_warrant(
        "EN-DIFFERENT-VAULT-RECORDS",
        "two overview files, facts/djev/djev-overview.md and "
        "facts/djev_shadow/djev_shadow-overview.md, each with its own definition")
    with pytest.raises(RB.GoldError) as e:
        RB.validate_gold_row(dict(row, seam="action_review", warrant=crossed))
    assert "a rule for seam" in str(e.value), \
        "refused before the seam check, so the seam rail was not the thing measured"


def test_build_warrant_and_parse_warrant_are_inverse_and_the_hash_covers_the_lines():
    sha = RB.rubric_sha256()
    assert len(sha) == 64
    # Example evidence names the two overview files it describes, because that is what
    # `parse_warrant` demands of any warrant: a reader must be able to open something.
    w = RB.build_warrant(
        "EN-DIFFERENT-VAULT-RECORDS",
        "two overview files, facts/djev/djev-overview.md and "
        "facts/djev_shadow/djev_shadow-overview.md, each with its own definition")
    parsed = RB.parse_warrant(w)
    assert parsed.rule_id == "EN-DIFFERENT-VAULT-RECORDS"
    assert parsed.evidence.startswith("two overview files")
    assert parsed.rule["label"] == "different"
    assert parsed.rule["seam"] == "entity"
    # A reader can recompute the hash the gold's `rubric_sha256` must equal.
    recomputed = hashlib.sha256(json.dumps(RB.RULES, sort_keys=True, ensure_ascii=False,
                                          separators=(",", ":")).encode()).hexdigest()
    assert recomputed == sha
    for rule_id, rule in RB.RULES.items():
        assert rule["artifact"], rule_id
        assert rule["text"], rule_id
        assert rule["seam"] in ("action_review", "dedupe", "entity"), rule_id


# ── clause 5: the README's provenance statement ─────────────────────────────

def test_the_readme_states_that_a_lloyd_label_is_admissible_only_for_agreement():
    text = README_PATH.read_text(encoding="utf-8").lower()
    assert "lloyd-authored" in text or "lloyd authored" in text
    assert "human label" in text
    assert "cross-engine agreement" in text
    assert "not a human label" in text, \
        "the README must say in as many words that this is not a human label"


def test_the_readme_states_what_the_arm_does_not_measure():
    text = README_PATH.read_text(encoding="utf-8").lower()
    for phrase in ("does not measure", "not evidence", "not licenses", "not license"):
        if phrase in text:
            break
    else:
        raise AssertionError("the README must name what the arm does NOT license")
    assert "abstain" in text or "unmeasurable" in text
    assert "label mass" in text or "label_mass" in text


def test_the_readme_records_the_shipped_sample_s_own_numbers():
    text = README_PATH.read_text(encoding="utf-8")
    side = json.loads(SIDECAR_PATH.read_text(encoding="utf-8"))
    for key in ("seed", "rubric_sha256"):
        assert str(side[key]) in text, f"README omits the shipped {key}"
    for stratum in ("action_review_high", "action_review_low", "entity"):
        assert str(side["strata"][stratum]["admitted"]) in text, stratum
    assert str(side["store"]["window_rows_read"]) in text
    assert str(side["strata"]["dedupe"]["admitted"]) in text, \
        "an empty stratum is stated in the README, not hidden"


def test_the_readme_s_numbers_are_the_ones_the_shipped_gold_contains():
    rows = [json.loads(l) for l in GOLD_PATH.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    text = README_PATH.read_text(encoding="utf-8")
    for seam, count in ((s, sum(1 for r in rows if r["seam"] == s))
                        for s in ("action_review", "entity", "dedupe")):
        if count:
            assert str(count) in text, f"README does not state the {seam} row count"


def test_the_readme_states_the_replay_s_shelf_life_and_names_the_rows_it_refused():
    """The documented `--ids-from` replay refused at 115 of 120 when re-run on
    2026-10-17T02:15:46Z: five `entity` rows fail gate 1 because the vault-derived
    overview page a label was read against has since been regenerated, while all 100
    action_review rows reproduced from their session records. The README must carry that
    as a dated measurement, name the ids it names truthfully, and say `gold.jsonl` is the
    durable artifact — never imply the replay can regenerate the sample at will."""
    text = README_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "shelf life" in lowered, \
        "the README must not present the recorded-ids replay as permanent"
    assert "rows: 115" in text, "the refused replay count must be the measured one"
    assert "2026-10-17t02:15:46z" in lowered, \
        "the refusal is a dated measurement, not a bare claim about the future"
    assert "no_canvas" in text and "pool 20" in text, \
        "the refusal is gate 1 on a moved canvas, not rows leaving the store"
    refused = ("19cd7959284555b8", "3d9dbc6da24e0a1a", "630dc90fde40117e",
               "8a37cf0c37bd3717", "8cddeef9514f1db7")
    assert all(i in text for i in refused), "all five refused ids must be named"
    shipped = {json.loads(line)["inputs_digest"]
               for line in GOLD_PATH.read_text(encoding="utf-8").splitlines()
               if line.strip()}
    assert set(refused) <= shipped, \
        "the README names an id that is not a row of the shipped gold"
    assert "durable artifact" in lowered, "it must say which file is the durable one"


# ── clause 5, and the runner surface's registration (clauses 3 and 4 live in
#    tests/test_cross_engine_disagreement.py; this node is the README's) ─────

def test_the_readme_states_the_caveat_in_prose_and_in_a_field():
    """Clause 5: the caveat must say all three things, and one must be a question —
    the admissibility is conditional on the scored quantity, not blanket."""
    text = README_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "not a human label" in lowered
    assert "lloyd-authored" in lowered or "lloyd authored" in lowered
    assert "cross-engine agreement" in lowered
    # the conditionality, in the order the clause states it: admissible ONLY BECAUSE
    assert "only because" in lowered
    # and the field the README points at really exists on the shipped gold
    rows = RB.load_gold(GOLD_PATH)
    assert all("not a human label" in r["labelled_by"] for r in rows)
    assert "labelled_by" in text
