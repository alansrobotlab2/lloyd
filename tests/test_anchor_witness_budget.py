"""#3218 clause 4: the nightly-witness family has a budget, and it refuses loudly.

The four rounds this item exists to stop each committed ONE baseline into the vault
(#2685→#2954, and #2935 for a `gate.json`), and the route that now does that copy
itself has to stop somewhere: 31 source baselines exist, 15 of them cited with no
committed copy, so a guard that copies on demand would keep copying every night
forever into a `backlog/data` that triage measured at 271 files and 127 MB. The
budget is where it stops.

The two things a budget can fail at are opposite, and both are pinned here: a cap
that trips early refuses evidence nobody asked to delete, and a cap that reads its
own measurement as zero is a cap that passes on the machine it is meant to bound —
so a `git ls-files` that cannot answer is a refusal, never a `0`.

What is NOT tested: deleting or rewriting a tracked witness. The route has no code
path that does it — no `git rm`, no expiry, no `git checkout` — so `test_a_refusal_
touches_no_tracked_witness` pins that by hashing the family before and after a
refusal, which is the only form of "never" a test can check here.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.automod import backlog as B          # noqa: E402


#: A baseline body with the figure fields the route quotes, so the note it emits is
#: the shape the eval job actually writes (`summary.overall`, and the rate legs
#: `scripts/eval_trend_stats.py` grades).
def _baseline(n: int = 7, lines: int = 4) -> str:
    rows = [{"q": f"q{i}", "entity_hit_rate": 0.5, "doc_hit_rate": 0.25}
            for i in range(lines)]
    return json.dumps({
        "label": f"nightly-20261024-run{n}",
        "rows": rows,
        "summary": {"overall": {
            "n_queries": 81, "entity_hit_rate": 0.5185185185185185,
            "doc_hit_rate": 0.4074074074074074, "mrr_doc": 0.346, "ndcg10": 0.398}},
    }, indent=2)


def _make_vault(tmp: Path):
    """A git tree with `backlog/data` and one commit, so `HEAD` exists.

    `git init -b main` because `rev-parse HEAD` below needs a head to compare
    against, and a fresh init without a commit has none — the shape of a vault clone
    mid-checkout, which this route must not mistake for a successful archive.
    """
    vault = tmp / "vault"
    (vault / "backlog" / "data").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(vault)], check=True)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(vault), "config", k, v], check=True)
    (vault / "backlog" / "data" / "seed.json").write_text('{"seed": 1}\n')
    subprocess.run(["git", "-C", str(vault), "add", "--", "backlog/data/seed.json"],
                   check=True)
    subprocess.run(["git", "-C", str(vault), "commit", "-q", "-m", "seed"], check=True)
    return vault


def _seed_tracked(vault: Path, name: str, body: str) -> None:
    """Commit a `nightly-*` witness into the vault, as a previous night would have."""
    (vault / "backlog" / "data" / name).write_text(body)
    subprocess.run(["git", "-C", str(vault), "add", "--", f"backlog/data/{name}"],
                   check=True)
    subprocess.run(["git", "-C", str(vault), "commit", "-q", "-m", f"seed {name}"],
                   check=True)


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """One temporary vault, reached through BOTH seams.

    `_vault_witness_dir` resolves the COPY through `app.paths.VAULT_ROOT`; the commit goes
    through `vault_round.land()`, which takes its tree from the module attribute
    `vault_round.VAULT` when it runs. Patching only the first writes the copy here and
    attempts the commit in `~/obsidian`, where nothing was staged — which is exactly the
    split `commit_anchor_witness` now refuses on rather than reporting quietly.
    """
    v = _make_vault(tmp_path)
    import app.paths as P
    from scripts.automod import vault_round as VR
    monkeypatch.setattr(P, "VAULT_ROOT", str(v))
    monkeypatch.setattr(VR, "VAULT", v)
    return v


@pytest.fixture
def src(tmp_path):
    d = tmp_path / "lloyd-data" / "eval" / "baselines"
    d.mkdir(parents=True)
    p = d / "nightly-20261024-20261024-060000.json"
    p.write_text(_baseline())
    return p


def _head_tree(vault: Path) -> str:
    return subprocess.run(["git", "-C", str(vault), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def _family(vault: Path) -> "list[str]":
    out = subprocess.run(["git", "-C", str(vault), "ls-files", "--", "backlog/data"],
                         capture_output=True, text=True).stdout
    return sorted(n for n in out.splitlines()
                  if Path(n).name.startswith("nightly-"))


def test_the_file_count_cap_refuses_and_names_count_and_budget(vault, src):
    """20 tracked `nightly-*` witnesses is the cap, and the refusal says 20 + 19.

    Seeded by committing 19 and refusing the 20th at `max_files=19`, which is the
    same arithmetic (`count + 1 > max_files`) with nineteen files instead of twenty
    planted on disk: the cap's shape is that the LAST admissible copy is the one
    where `count + 1 == max_files`, so a node that planted 20 and expected the 21st
    to pass would be testing a different rule.
    """
    for i in range(19):
        _seed_tracked(vault, f"nightly-202609{i:02d}-x.json", _baseline(i))
    before_head, before_family = _head_tree(vault), _family(vault)

    res = B.commit_anchor_witness(src, "backlog/data/nightly-20261024-20261024-060000.json",
                                  max_files=19)

    assert res["ok"] is False and res["reason"] == "too_many_files"
    assert res["sha"] == "" and res["wc"] == ""
    # Names the CURRENT count (19) and the budget (20) the note is read for, and says
    # which family it is talking about.
    assert "19" in res["note"] and str(B.ANCHOR_WITNESS_MAX_FILES) in res["note"], \
        res["note"]
    assert "nightly-" in res["note"] and res["note"].strip()
    assert _head_tree(vault) == before_head, "a refusal committed a sha"
    assert _family(vault) == before_family, "a refusal added a witness"


def test_the_byte_cap_refuses_and_names_count_and_budget(vault, src, monkeypatch):
    """8 MB total is the byte cap: a vault at 8,388,608 B minus 1 refuses the copy.

    The ceiling is on the TOTAL after the copy (`total + size > max_total_bytes`), so
    seeding the family to `max_total_bytes - size + 1` is the exact first byte that
    must refuse, and the node also runs the boundary one byte lower to prove the cap
    is arithmetic and not a coincidence of one planted number.
    """
    size = src.stat().st_size
    budget = 8 * 1_048_576
    assert size < budget, "the fixture must fit under the ceiling it is testing"
    body = _baseline()
    filler = "x" * (budget - size - len(body.encode()) + 1)
    _seed_tracked(vault, "nightly-20260901-fat.json",
                  body.rstrip("}") + ',"filler":"' + filler + '"}')
    total = sum((vault / n).stat().st_size for n in _family(vault))
    assert total + size > budget, f"fixture does not straddle the budget: {total}+{size}"
    before_head = _head_tree(vault)

    res = B.commit_anchor_witness(src, "backlog/data/nightly-20261024-20261024-060000.json")

    assert res["ok"] is False and res["reason"] == "too_many_bytes"
    assert f"{total:,}" in res["note"], "the note omits the current total"
    assert f"{budget:,}" in res["note"], "the note omits the budget"
    assert _head_tree(vault) == before_head
    assert len(_family(vault)) == 1, "a refusal added a witness"


def test_a_copy_under_both_caps_is_the_one_that_lands(vault, src):
    """The positive control for both caps: with the family empty the copy commits.

    Without this the two refusal nodes above could pass on a route that refuses
    everything — and a route that always refuses is indistinguishable from a budget
    set to zero, which is the failure the next night's traffic would reveal.
    """
    res = B.commit_anchor_witness(src, "backlog/data/nightly-20261024-20261024-060000.json")
    assert res["ok"] is True, res["note"]
    assert res["wc"], "the archive landed with no figure quoted"
    assert len(_family(vault)) == 1


def test_a_refusal_touches_no_tracked_witness(vault, src):
    """Refusing must not delete or rewrite what the vault already carries.

    The route's own comment says the cap is a human call, not a licence to expire a
    witness, and the only test for that is a hash of the family before and after.
    """
    _seed_tracked(vault, "nightly-20260901-x.json", _baseline(1))
    _seed_tracked(vault, "nightly-20260902-x.json", _baseline(2))
    hashes_before = [(vault / n).read_bytes() for n in _family(vault)]
    before_head = _head_tree(vault)

    res = B.commit_anchor_witness(src, "backlog/data/nightly-20261024-20261024-060000.json",
                                  max_files=1)

    assert res["ok"] is False and res["note"]
    after = [(vault / n).read_bytes() for n in _family(vault)]
    assert after == hashes_before, "a refusal rewrote a tracked witness"
    assert _head_tree(vault) == before_head
    assert not (vault / "backlog" / "data" / src.name).exists(), \
        "a refused copy left the bytes in the vault untracked: the file is on disk " \
        "with no history, which reads as archived to anyone who only looks"


def test_an_unmeasurable_family_is_a_refusal_not_a_zero(tmp_path, monkeypatch, src):
    """`git ls-files` that cannot answer must refuse, not report an empty family.

    Done by pointing `VAULT_ROOT` at a directory that is NOT a git tree, so the
    count and the byte total are both unknown. Asserting the refusal here is the
    positive control for the cap arithmetic: a `return (0, 0)` on failure would sail
    every node in this file green while bounding nothing.
    """
    plain = tmp_path / "not-a-vault"
    (plain / "backlog" / "data").mkdir(parents=True)
    import app.paths as P
    monkeypatch.setattr(P, "VAULT_ROOT", str(plain))
    heads_before = sorted(plain.rglob("*"))

    res = B.commit_anchor_witness(src, "backlog/data/nightly-20261024-20261024-060000.json")

    assert res["ok"] is False and res["reason"] == "split_vault"
    assert res["sha"] == "" and res["note"].strip()
    assert "git" in res["note"].lower()
    assert sorted(plain.rglob("*")) == heads_before, \
        "a refused archive still wrote bytes into a tree with no history"


def test_the_guard_routes_a_budget_refusal_to_the_owed_note(vault, src, monkeypatch):
    """The refusal surfaces where a human will read it — and no copy demand beside it.

    This is the whole difference between a silent no-op and a refusal: the clause
    count goes to zero and the owed list gains one line naming the count and the
    budget. A route that swallowed the refusal would close the item with nothing
    committed and nothing owed, which is worse than the round this item replaced.
    """
    # The cap is brought to the fixture instead of the fixture built up to the cap:
    # 20 real baselines at ~400 KB each is 8 MB of tmpfs per node to demonstrate an
    # `if`, and the constant is what the note quotes, so the note still names 3/3.
    monkeypatch.setattr(B, "ANCHOR_WITNESS_MAX_FILES", 3)
    for i in range(3):
        _seed_tracked(vault, f"nightly-202609{i:02d}-x.json", _baseline(i))
    text = f"See `{src}` for the figure."

    out = B.add_witness_artifact_clause([], text, route_to_owed=True, item_id=3218)
    clauses, owed = out

    assert clauses == [], "a budget refusal still booked the copy it refused"
    assert len(owed) == 1, owed
    assert "#3218 archived nothing" in owed[0]
    assert "3" in owed[0], owed[0]      # the current count and the budget, both 3 here
