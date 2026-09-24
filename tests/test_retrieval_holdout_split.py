"""The retrieval eval's holdout leg: its manifest, and the rule that nothing else reads it (#1412).

Two properties, and the second is the one that makes the first worth anything:

* the split is pinned — `split_hash` is recomputed, never trusted, and a manifest
  whose id pool or content moved after the hash was written is refused (the
  `scripts/autoresearch/bench_split.py` discipline, #549);
* no non-gate path reads a reserved id — not `run_eval.py`'s default corpus, not the
  paired arms' `--queries`, not a nightly artifact, not any tracked file. A holdout a
  job can read is decorative.

No test here reads a reserved id out of anything but the holdout file itself, and no
assertion message prints one.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from eval import retrieval_holdout as H

ROOT = Path(__file__).resolve().parent.parent


def _copy_split(tmp_path: Path) -> tuple[Path, Path]:
    """The committed holdout file and dev file, copied, with a fresh manifest beside them."""
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    holdout = eval_dir / H.HOLDOUT_FILENAME
    dev = eval_dir / H.DEV_FILENAME
    holdout.write_text(H.HOLDOUT_QUERIES.read_text())
    dev.write_text(H.DEV_QUERIES.read_text())
    manifest = H.compute_manifest(holdout, dev)
    (eval_dir / H.MANIFEST_FILENAME).write_text(json.dumps(manifest))
    return holdout, dev


# ---------------------------------------------------------------------------
# clause 2 — the manifest and its split_hash
# ---------------------------------------------------------------------------

def test_the_committed_manifest_verifies_against_the_committed_holdout_file():
    manifest = H.load_manifest(ROOT)
    assert manifest is not None, (
        "eval/vault_recall_holdout_manifest.json is absent or no longer describes the "
        "holdout file — `python -m eval.retrieval_holdout write` after an intended change")
    assert manifest["n"] >= 20 and manifest["leg"] == "holdout"
    assert sorted(manifest["reserved_ids"]) == sorted(H.query_ids(H.HOLDOUT_QUERIES))
    assert manifest["split_hash"] and len(manifest["split_hash"]) == 64


def test_the_hash_is_recomputed_so_an_edited_id_pool_is_refused(tmp_path):
    holdout, _ = _copy_split(tmp_path)
    manifest = json.loads((holdout.parent / H.MANIFEST_FILENAME).read_text())
    assert H.verify(manifest, holdout)

    dropped = dict(manifest, reserved_ids=manifest["reserved_ids"][1:])
    assert not H.verify(dropped), "a pool edited after the hash must not verify"
    swapped = dict(manifest, reserved_ids=manifest["reserved_ids"][:-1] + ["swapped-in"])
    assert not H.verify(swapped)
    assert not H.verify({k: v for k, v in manifest.items() if k != "split_hash"})


def test_a_holdout_file_edited_after_the_hash_is_refused(tmp_path):
    """Adding a query, dropping one or relabelling one all refuse the manifest —
    the split cannot be re-picked after its results were seen."""
    holdout, _ = _copy_split(tmp_path)
    original = yaml.safe_load(holdout.read_text())
    manifest = json.loads((holdout.parent / H.MANIFEST_FILENAME).read_text())

    added = {"queries": original["queries"] + [
        {"id": "added-after-results", "query": "q", "category": "single",
         "expect_entities": [], "expect_docs": ["lloyd"]}]}
    holdout.write_text(yaml.safe_dump(added))
    assert not H.verify(manifest, holdout)
    assert H.load_manifest(tmp_path) is None, "a refused manifest reads as absent"

    relabelled = {"queries": [dict(q) for q in original["queries"]]}
    relabelled["queries"][0]["expect_docs"] = ["an/easier/target.md"]
    holdout.write_text(yaml.safe_dump(relabelled))
    assert not H.verify(manifest, holdout), "a relabel keeps the ids and must still refuse"

    holdout.write_text(yaml.safe_dump(original))
    assert H.verify(manifest, holdout), "comment/layout changes are not a new split"


def test_a_tranche_that_overlaps_dev_is_refused_by_count(tmp_path):
    holdout, dev = _copy_split(tmp_path)
    dev_spec = yaml.safe_load(dev.read_text())["queries"][0]
    specs = yaml.safe_load(holdout.read_text())["queries"] + [dev_spec]
    holdout.write_text(yaml.safe_dump({"queries": specs}))
    with pytest.raises(ValueError) as exc:
        H.compute_manifest(holdout, dev)
    assert "1 holdout id(s)" in str(exc.value)
    assert str(dev_spec["id"]) not in str(exc.value)


def test_a_refused_manifest_still_reserves_the_file_ids(tmp_path):
    """Tampering must not make the reserve rule forget what it was reserving."""
    holdout, _ = _copy_split(tmp_path)
    (holdout.parent / H.MANIFEST_FILENAME).write_text("{}")
    assert H.load_manifest(tmp_path) is None
    assert H.reserved_ids(tmp_path) == set(H.query_ids(holdout))


# ---------------------------------------------------------------------------
# clause 3 — no non-gate path reads a reserved id
# ---------------------------------------------------------------------------

def test_run_eval_defaults_to_the_dev_corpus():
    from eval.run_eval import build_parser
    default = Path(build_parser().parse_args([]).queries)
    assert default.name == H.DEV_FILENAME
    assert not H.is_holdout_corpus(H.load_specs(default), ROOT)


def test_run_eval_refuses_the_holdout_corpus_outside_the_holdout_leg():
    specs = H.load_specs(H.HOLDOUT_QUERIES)
    reserved = H.reserved_ids(ROOT)
    no_leg = H.refusal(specs, "adhoc", {}, ROOT)
    assert no_leg and H.HOLDOUT_LEG_ENV in no_leg
    assert H.refusal(specs, "holdout-check", {H.HOLDOUT_LEG_ENV: "1"}, ROOT) is None
    nightly = H.refusal(specs, "nightly-20260924", {H.HOLDOUT_LEG_ENV: "1"}, ROOT)
    assert nightly and "nightly" in nightly, "a holdout run may never write a nightly record"
    # One reserved query smuggled into an otherwise-dev corpus is still refused.
    dev_specs = H.load_specs(H.DEV_QUERIES)
    assert H.refusal(dev_specs, "nightly-x", {}, ROOT) is None
    assert H.refusal(dev_specs + specs[:1], "nightly-x", {}, ROOT)
    for msg in (no_leg, nightly):
        assert not any(rid in msg for rid in reserved), "a refusal must name no reserved id"


def test_run_eval_main_exits_2_on_the_holdout_corpus_before_scoring(monkeypatch, capsys):
    import eval.run_eval as RE
    monkeypatch.delenv(H.HOLDOUT_LEG_ENV, raising=False)
    monkeypatch.setattr("sys.argv", ["run_eval.py", "--queries", str(H.HOLDOUT_QUERIES),
                                     "--label", "adhoc"])
    monkeypatch.setattr(RE, "_corpus_provenance",
                        lambda: pytest.fail("scored a reserved corpus outside the holdout leg"))
    assert RE.main() == 2
    err = capsys.readouterr().err
    assert H.HOLDOUT_LEG_ENV in err
    assert not any(rid in err for rid in H.reserved_ids(ROOT))


class _Proc:
    returncode = 1
    stdout = "per-query table naming every id"
    stderr = ""


def _capture_arm(monkeypatch, R) -> list:
    seen: list = []

    def fake_run(argv, **kw):
        seen.append({"argv": list(argv), "env": dict(kw.get("env") or {})})
        return _Proc()
    monkeypatch.setattr(R.subprocess, "run", fake_run)
    return seen


def test_the_paired_arms_pass_the_holdout_path_only_on_the_holdout_leg(monkeypatch, tmp_path):
    from workers.sources import automod_regression as R
    seen = _capture_arm(monkeypatch, R)
    assert R._run_arm(tmp_path, "automod-check", {}) is None
    assert R._run_arm(tmp_path, R.HOLDOUT_LABEL_CURRENT, {}, leg="holdout") is None
    dev_call, hold_call = seen
    q = lambda call: Path(call["argv"][call["argv"].index("--queries") + 1])  # noqa: E731
    assert q(dev_call) == R.LIVE_QUERIES and q(dev_call).name == H.DEV_FILENAME
    assert H.HOLDOUT_LEG_ENV not in dev_call["env"]
    assert q(hold_call) == R.LIVE_HOLDOUT_QUERIES
    assert hold_call["env"][H.HOLDOUT_LEG_ENV] == "1"
    # Both resolved from the LIVE tree: a round cannot swap its own file in.
    assert R.LIVE_HOLDOUT_QUERIES.parent == R.LIVE_QUERIES.parent
    with pytest.raises(ValueError):
        R._run_arm(tmp_path, "x", {}, leg="whatever")


def test_a_failed_holdout_arm_logs_no_runner_output(monkeypatch, tmp_path, caplog):
    from workers.sources import automod_regression as R
    _capture_arm(monkeypatch, R)
    with caplog.at_level("ERROR"):
        R._run_arm(tmp_path, R.HOLDOUT_LABEL_CURRENT, {}, leg="holdout")
        R._run_arm(tmp_path, "automod-check", {})
    hold, dev = [r.getMessage() for r in caplog.records][-2:]
    assert "holdout leg" in hold and "per-query" not in hold
    assert "per-query" in dev, "the dev leg keeps its diagnostic tail"


def _tracked_files() -> list[Path]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"],
                         capture_output=True, text=True, check=True).stdout
    return [ROOT / p for p in out.split("\0") if p]


# Files that may name the holdout file or its module. Everything else that reads the
# eval corpus — the nightly, the prefetch eval, the counterfactual, fact improvement,
# the trend audit — reads the dev file by its own name, and this list is how that stays
# true.
READERS_ALLOWED = {
    "eval/retrieval_holdout.py",               # the manifest and the reserve rule
    "eval/run_eval.py",                        # enforces the rule
    "workers/sources/automod_regression.py",   # the holdout leg itself
    "eval/vault_recall_holdout_queries.yaml",
    "eval/vault_recall_holdout_manifest.json",
    ".gitignore",
}


def test_no_other_tracked_code_names_the_holdout_file_or_module():
    offenders = []
    for path in _tracked_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel in READERS_ALLOWED or rel.startswith("tests/") or rel.startswith("architecture/"):
            continue
        if path.suffix not in {".py", ".sh", ".yaml", ".yml", ".json", ".md", ".ts", ".tsx", ".mjs"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if ("vault_recall_holdout" in text or "retrieval_holdout" in text
                or "vault_recall_*" in text):
            offenders.append(rel)
    # eval/measurements may cite the leg by name in prose; it never carries an id
    # (the next test), so it is a report, not a reader.
    offenders = [o for o in offenders if not o.startswith("eval/measurements/")]
    assert not offenders, f"tracked files outside the holdout leg name it: {offenders}"


def test_no_reserved_id_appears_in_any_other_tracked_file():
    reserved = H.reserved_ids(ROOT)
    assert len(reserved) >= 20
    hits = []
    for path in _tracked_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel in {"eval/vault_recall_holdout_queries.yaml",
                   "eval/vault_recall_holdout_manifest.json"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        if any(rid in text for rid in reserved):
            hits.append(rel)
    assert not hits, f"{len(hits)} tracked file(s) carry a reserved holdout id: {hits}"


def test_no_nightly_artifact_carries_a_reserved_id():
    """The nightly trend report reads `nightly-*.json`; none may hold a holdout row."""
    from app.paths import production_data_root
    baselines = production_data_root() / "eval" / "baselines"
    nightlies = sorted(baselines.glob("nightly-*.json")) if baselines.is_dir() else []
    if not nightlies:
        pytest.skip(f"no nightly artifacts under {baselines}")
    reserved = H.reserved_ids(ROOT)
    leaked = 0
    for p in nightlies:
        try:
            blob = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        ids = {str(r.get("id")) for r in blob.get("records") or []}
        leaked += len(ids & reserved)
        assert blob.get("leg", "dev") == "dev", p.name
    assert leaked == 0, f"{leaked} reserved id(s) in nightly artifacts"
