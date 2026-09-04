#!/usr/bin/env python3
"""Rebuild the fact tree from the vault, then swap it in behind a gate.

Two defects in the current tree cannot be repaired, only re-extracted:

  * 0.37% of 205,573 facts carry `created_at` and `source_doc`. Nothing else
    can be dated, attributed, or selectively reverted.
  * 15,825 fact files hold duplicate fact IDs, because the extractor
    restarted its numbering on every run. Anything that addresses a fact by
    ID acts on whichever it finds first.

Repairing those in place means inventing provenance for facts whose source is
unknown, which is worse than not having it. Re-extracting is honest: every
fact in the new tree came from a named document at a known time.

The extraction writes to a SEPARATE tree and a SEPARATE store (LLOYD_FACTS_ROOT
/ LLOYD_KG_DB), so the live system keeps serving throughout. Nothing swaps until
`gate` passes.

    kg_rebuild.py freeze     # pause writers, snapshot, record the before-state
    kg_rebuild.py export     # carry over what re-extraction cannot reproduce
    kg_rebuild.py extract    # the long part; resumable, safe to re-run
    kg_rebuild.py import     # apply the carry-over to the new tree
    kg_rebuild.py gate       # every check, as JSON. Refuses to lie.
    kg_rebuild.py swap       # two renames + reindex. Requires a passing gate.
    kg_rebuild.py rollback   # reverse the renames, restore the store
    kg_rebuild.py status     # where a run got to
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLOYD = HERE.parent.parent
sys.path.insert(0, str(LLOYD))
sys.path.insert(0, str(HERE))

from app.paths import VAULT_DERIVED_ROOT, VAULT_FACTS_ROOT, VAULT_KG_DB  # noqa: E402
from app.kg_store import KGStore  # noqa: E402
from _invocation import invocation_ledger  # noqa: E402

REBUILD_FACTS = VAULT_DERIVED_ROOT / "facts-rebuild"
REBUILD_DB = VAULT_DERIVED_ROOT / "kg-rebuild.sqlite"
RUN_ROOT = LLOYD / "_pipeline" / "backups"
MEMORY_GRAPH = LLOYD / "_pipeline" / "memory-graph"
STATE_PATH = VAULT_DERIVED_ROOT / "rebuild-state.json"

# The gate. Every one of these must hold before the new tree replaces the old.
# `eval_*` compare against the numbers recorded by `freeze`.
GATE = {
    "provenance_pct": 100.0,       # every fact says where it came from and when
    "duplicate_id_files": 0,
    "contamination_dirs": 0,
    "junk_entity_dirs": 0,
    "node_coverage_pct": 30.0,     # was 14% before the extractor emitted edges
    "corpus_coverage_pct": 98.0,   # documents actually extracted, of the corpus
    "eval_mrr_slack": 0.0,         # may not fall below the frozen baseline
    "eval_ndcg_slack": 0.0,
    "eval_category_regression": 0.05,
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(**kw) -> dict:
    state = load_state()
    state.update(kw)
    state["updated_at"] = _now()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def _rebuild_env() -> dict:
    """Environment that points every child process at the REBUILD tree."""
    return dict(os.environ,
                LLOYD_FACTS_ROOT=str(REBUILD_FACTS),
                LLOYD_KG_DB=str(REBUILD_DB))


def _venv_python() -> str:
    return str(LLOYD / ".venvs" / "lloyd" / "bin" / "python")


# ── freeze ───────────────────────────────────────────────────────────────────

PAUSED_TASKS = (24, 48, 74)


def cmd_freeze(args) -> int:
    """Stop the writers, snapshot, and record what we are measured against."""
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUN_ROOT / f"rebuild-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    st = KGStore(VAULT_KG_DB)
    before = st.stats()
    backup = st.backup(run_dir / "kg-before.sqlite")
    st.export_json(run_dir / "json-before")
    st.close()
    print(f"store snapshot: {backup}  {before}")

    paused = []
    for task_id in PAUSED_TASKS:
        for path in (Path.home() / "obsidian" / "autonomy").glob(f"{task_id}-*.md"):
            text = path.read_text(encoding="utf-8")
            if "\nstatus: paused\n" in text:
                paused.append({"task": task_id, "path": str(path), "was": "paused"})
                continue
            import re
            new, n = re.subn(r"^status: .*$", "status: paused", text, count=1, flags=re.M)
            if n and not args.dry_run:
                path.write_text(new, encoding="utf-8")
            was = re.search(r"^status: (.*)$", text, re.M)
            paused.append({"task": task_id, "path": str(path),
                           "was": was.group(1) if was else "?"})
            print(f"  paused #{task_id} ({path.name}, was {was.group(1) if was else '?'})")

    # The write flag `_fact_add` reads. A chat turn can still add a fact while
    # the rebuild runs; without this it lands in a tree about to be renamed.
    if not args.dry_run and not args.keep_writes:
        _set_write_enabled(False)

    baseline = _run_eval("rebuild-before", run_dir)
    state = save_state(run_dir=str(run_dir), frozen_at=_now(),
                       store_before=before, paused=paused,
                       eval_before=baseline, ledger=invocation_ledger())
    (run_dir / "freeze.json").write_text(json.dumps(state, indent=2))
    print(f"\nfrozen. run dir: {run_dir}")
    print(f"baseline: MRR {baseline.get('mrr_doc')} NDCG@10 {baseline.get('ndcg10')}")
    return 0


def _set_write_enabled(enabled: bool) -> None:
    """Flip `knowledge_graph.write_enabled` in config.yaml."""
    import yaml
    from app.atomic_io import atomic_write_text
    path = LLOYD / "config.yaml"
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    data.setdefault("knowledge_graph", {})["write_enabled"] = enabled
    # Rewrite only that key, so the file's comments survive.
    import re
    if re.search(r"^knowledge_graph:", text, re.M):
        text = re.sub(r"(^knowledge_graph:\n(?:[ \t]+.*\n)*?[ \t]+write_enabled: )\w+",
                      lambda m: m.group(1) + ("true" if enabled else "false"),
                      text, count=1, flags=re.M)
    else:
        text = text.rstrip("\n") + (
            "\n\n# Fact writes through the fact_add MCP tool. The rebuild sets this\n"
            "# false so a chat turn cannot add a fact to a tree about to be renamed.\n"
            f"knowledge_graph:\n  write_enabled: {'true' if enabled else 'false'}\n")
    atomic_write_text(path, text)
    print(f"  config.yaml knowledge_graph.write_enabled = {enabled}")


def _corpus_size() -> int:
    """How many documents the allow-list selects. The denominator for
    corpus coverage — read from the same code the extractor uses, so the
    two cannot drift."""
    import importlib.util
    ngm = HERE / "next-gen-memory"
    if str(ngm) not in sys.path:
        sys.path.insert(0, str(ngm))     # nightly_extraction imports its siblings by name
    spec = importlib.util.spec_from_file_location(
        "ne_corpus", ngm / "nightly_extraction.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ne_corpus"] = mod
    spec.loader.exec_module(mod)
    return len(mod.NightlyExtraction()._eligible_files(full_mode=True))


def _run_eval(label: str, run_dir: Path) -> dict:
    """Run the retrieval eval against whatever is live now."""
    try:
        out = subprocess.run(
            [_venv_python(), str(LLOYD / "eval" / "run_eval.py"),
             "--label", label, "--notes", f"kg rebuild: {label}"],
            cwd=str(LLOYD), capture_output=True, text=True, timeout=1800)
        if out.returncode != 0:
            print(f"  [eval] failed rc={out.returncode}: {out.stderr[-500:]}")
            return {}
        newest = max((LLOYD / "eval" / "baselines").glob(f"{label}-*.json"),
                     key=lambda p: p.stat().st_mtime)
        rec = json.loads(newest.read_text())
        shutil.copy2(newest, run_dir / newest.name)
        summary = rec.get("summary", {})
        return {**summary.get("overall", {}),
                "by_category": summary.get("by_category", {}),
                "matches_production_defaults": rec.get("matches_production_defaults"),
                "file": str(newest)}
    except Exception as exc:
        print(f"  [eval] {type(exc).__name__}: {exc}")
        return {}


# ── export ───────────────────────────────────────────────────────────────────

CARRY_PROVENANCE = ("STATED", "INFERRED", "AMBIGUOUS")


def cmd_export(args) -> int:
    """Collect what re-extraction cannot reproduce."""
    state = load_state()
    run_dir = Path(state.get("run_dir") or (RUN_ROOT / f"rebuild-{dt.datetime.now():%Y%m%dT%H%M%SZ}"))
    out = run_dir / "carryover"
    out.mkdir(parents=True, exist_ok=True)

    st = KGStore(VAULT_KG_DB)

    # (a) Hand-stated and inferred facts. The extractor will never produce
    #     these: they came from a conversation, not a document.
    rows = st._query(
        "SELECT * FROM facts_idx WHERE provenance IN (?,?,?) OR source_doc LIKE '%session%'",
        CARRY_PROVENANCE)
    facts = [{k: r[k] for k in r.keys()} for r in rows]
    (out / "facts.json").write_text(json.dumps(facts, indent=2, default=str))

    # (b) Experiment records, verbatim — written by the autoresearch worker,
    #     not extracted from anything.
    exp_src = VAULT_FACTS_ROOT / "Experiments"
    n_exp = 0
    if exp_src.is_dir():
        shutil.copytree(exp_src, out / "Experiments", dirs_exist_ok=True)
        n_exp = sum(1 for _ in (out / "Experiments").rglob("*.md"))

    # (c) Aliases a human or a judge decided. Case and punct aliases are
    #     mechanical and the sweep will re-derive them.
    aliases = [a for a in st.aliases.rows() if a["kind"] in ("semantic", "suffix", "manual")]
    (out / "aliases.json").write_text(json.dumps(aliases, indent=2, default=str))

    # (d) Review state: judge verdicts, ambiguous reports, applied merges.
    review = out / "review"
    review.mkdir(exist_ok=True)
    n_review = 0
    for pattern in ("semantic-verdicts*.jsonl", "semantic-proposals*.jsonl",
                    "entity-ambiguous-*.md", "entity-merges-applied-*.json",
                    "entity-merges-reverted-*.json", "graph-baseline.json",
                    "junk-entities-review.json"):
        for p in MEMORY_GRAPH.glob(pattern):
            if p.is_file():
                shutil.copy2(p, review / p.name)
                n_review += 1

    # (e) Stated edges. Extracted ones are re-derived; a stated edge is a claim
    #     someone made.
    edges = [e for e in st.edges.all(include_expired=False)
             if (e.get("provenance") or "") in CARRY_PROVENANCE
             or (e.get("origin") or "") in ("fact_relate", "manual")]
    (out / "edges.json").write_text(json.dumps(edges, indent=2, default=str))
    st.close()

    manifest = {"exported_at": _now(), "facts": len(facts), "experiments_files": n_exp,
                "aliases": len(aliases), "review_files": n_review, "edges": len(edges)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    save_state(run_dir=str(run_dir), carryover=str(out), export=manifest)
    print(json.dumps(manifest, indent=2))
    print(f"\ncarry-over: {out}")
    return 0


# ── extract ──────────────────────────────────────────────────────────────────

def _hashed_count() -> int:
    """Documents the rebuild has actually extracted.

    A failed extraction is deliberately not hashed, so this is the count of
    successes — which is exactly what the gate's corpus-coverage check reads.
    """
    try:
        return int(json.loads(
            (VAULT_DERIVED_ROOT / "rebuild-content-hashes.json").read_text())["file_count"])
    except Exception:
        return 0


def cmd_extract(args) -> int:
    """Extract the corpus into the rebuild tree.

    Resumable: the content-hash index lives with the rebuild, so a re-run
    continues where it stopped AND retries whatever failed — a failed
    document is never hashed.

    Runs up to `--passes` times, stopping early when a pass extracts nothing
    new. Individual documents time out under concurrency (3 of the first 533
    at 8 workers, all 120s LLM timeouts on long documents), and each of those
    is a document the gate's 98% coverage floor would otherwise block the
    swap on. Sweeping them up is mechanical, so it should not be manual.
    """
    REBUILD_FACTS.mkdir(parents=True, exist_ok=True)
    env = _rebuild_env()
    # A hash index of its own, or the rebuild would skip every file the LIVE
    # tree has already extracted.
    env["LLOYD_CONTENT_HASHES"] = str(VAULT_DERIVED_ROOT / "rebuild-content-hashes.json")

    print(f"  LLOYD_FACTS_ROOT={REBUILD_FACTS}")
    print(f"  LLOYD_KG_DB={REBUILD_DB}")

    rc = 0
    t_all = time.perf_counter()
    for attempt in range(1, max(1, args.passes) + 1):
        before = _hashed_count()
        cmd = [_venv_python(), "nightly_extraction.py", "--full",
               "--workers", str(args.workers)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        print(f"\n== pass {attempt}/{args.passes} — {before:,} documents extracted so far ==")
        print(f"$ {' '.join(cmd)}")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(HERE / "next-gen-memory"), env=env)
        rc = proc.returncode
        after = _hashed_count()
        gained = after - before
        print(f"   pass {attempt}: rc={rc}, +{gained:,} documents in {time.perf_counter() - t0:.0f}s")
        if rc != 0:
            print("   stopping: the extractor exited non-zero")
            break
        if gained == 0:
            print("   stopping: nothing new extracted — retries are not making progress")
            break
        if args.limit:
            break     # a bounded run is a bounded run

    st = KGStore(REBUILD_DB)
    stats = st.stats()
    st.close()
    extracted = _hashed_count()
    corpus = _corpus_size()
    save_state(extract_last_rc=rc,
               extract_elapsed_s=round(time.perf_counter() - t_all, 1),
               extracted=extracted, corpus=corpus,
               rebuild_stats=stats)
    pct = round(100.0 * extracted / corpus, 2) if corpus else 0.0
    print(f"\nrc={rc} in {time.perf_counter() - t_all:.0f}s")
    print(f"corpus coverage: {extracted:,}/{corpus:,} = {pct}% "
          f"(gate wants >= {GATE['corpus_coverage_pct']}%)")
    print(f"rebuild store: {stats}")
    if pct < GATE["corpus_coverage_pct"]:
        print(f"\n{corpus - extracted:,} documents still unextracted. Re-run `extract` "
              "to retry them; if the count stops moving, the failures are not transient "
              "and want looking at (check the log for `FAILED:`).")
    return rc


# ── import ───────────────────────────────────────────────────────────────────

def cmd_import(args) -> int:
    """Apply the carry-over to the rebuilt tree."""
    state = load_state()
    carry = Path(state.get("carryover") or "")
    if not carry.is_dir():
        print("no carry-over export found; run `export` first", file=sys.stderr)
        return 2
    if not REBUILD_FACTS.is_dir():
        print(f"no rebuild tree at {REBUILD_FACTS}; run `extract` first", file=sys.stderr)
        return 2

    os.environ["LLOYD_FACTS_ROOT"] = str(REBUILD_FACTS)
    os.environ["LLOYD_KG_DB"] = str(REBUILD_DB)
    import importlib
    import app.paths
    importlib.reload(app.paths)
    import app.kg_store
    importlib.reload(app.kg_store)
    from app.kg_store import store as rebuild_store
    import agent_mcp._shared as shared
    importlib.reload(shared)
    import agent_mcp.facts as facts_mod
    importlib.reload(facts_mod)

    st = rebuild_store()
    stats = {"facts": 0, "facts_skipped": 0, "aliases": 0, "edges": 0, "experiments": 0}

    # Facts go back through fact_add, so they get the new ID scheme and land
    # in the new tree's files rather than being copied as raw markdown.
    for f in json.loads((carry / "facts.json").read_text()):
        res = facts_mod._fact_add({
            "entity": f["entity"], "category": f["category"], "fact": f["fact"],
            "confidence": f.get("confidence") or 0.9,
            "provenance": f.get("provenance") or "STATED",
            "source_doc": f.get("source_doc"), "valid_at": f.get("valid_at"),
        })
        if res.get("success"):
            stats["facts"] += 1
        else:
            stats["facts_skipped"] += 1

    exp = carry / "Experiments"
    if exp.is_dir():
        shutil.copytree(exp, REBUILD_FACTS / "Experiments", dirs_exist_ok=True)
        stats["experiments"] = sum(1 for _ in (REBUILD_FACTS / "Experiments").rglob("*.md"))
        st.facts_idx.reindex(list((REBUILD_FACTS / "Experiments").rglob("*.md")),
                             root=REBUILD_FACTS)

    with st.transaction():
        for a in json.loads((carry / "aliases.json").read_text()):
            st.aliases.set(a["surface"], a["canonical"], kind=a["kind"],
                           origin=a["origin"], report_path=a.get("report_path"))
            stats["aliases"] += 1
        for e in json.loads((carry / "edges.json").read_text()):
            payload = {k: v for k, v in e.items() if k not in ("id", "superseded_edge_id")}
            try:
                st.edges.add(payload, origin=e.get("origin") or "migration")
                stats["edges"] += 1
            except ValueError:
                pass

    review = carry / "review"
    if review.is_dir():
        for p in review.iterdir():
            if p.is_file() and not (MEMORY_GRAPH / p.name).exists():
                shutil.copy2(p, MEMORY_GRAPH / p.name)

    st.entities.backfill_kinds()
    save_state(import_stats=stats, rebuild_stats=st.stats())
    print(json.dumps(stats, indent=2))
    return 0


# ── gate ─────────────────────────────────────────────────────────────────────

def cmd_gate(args) -> int:
    """Every check, as JSON. Exit 0 only if all pass."""
    if not REBUILD_FACTS.is_dir():
        print(f"no rebuild tree at {REBUILD_FACTS}", file=sys.stderr)
        return 2
    state = load_state()
    run_dir = Path(state.get("run_dir") or RUN_ROOT)
    results: dict = {"checked_at": _now(), "facts_root": str(REBUILD_FACTS),
                     "db": str(REBUILD_DB), "checks": {}}

    st = KGStore(REBUILD_DB)
    stats = st.stats()
    results["store"] = stats

    def record(name, ok, got, want, note=""):
        results["checks"][name] = {"pass": bool(ok), "got": got, "want": want, "note": note}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, want {want}"
              + (f" — {note}" if note else ""))

    # 1. Provenance
    total = stats["facts"]
    both = st._query("SELECT COUNT(*) n FROM facts_idx "
                     "WHERE created_at IS NOT NULL AND source_doc IS NOT NULL")[0]["n"]
    pct = round(100.0 * both / total, 2) if total else 0.0
    record("provenance_pct", pct >= GATE["provenance_pct"], pct, f">= {GATE['provenance_pct']}",
           "every fact must say where it came from and when")

    # 2. Duplicate fact IDs
    dup = st._query(
        "SELECT COUNT(*) n FROM (SELECT file_path, fact_id FROM facts_idx "
        "WHERE fact_id IS NOT NULL GROUP BY file_path, fact_id HAVING COUNT(*) > 1)")[0]["n"]
    record("duplicate_id_files", dup <= GATE["duplicate_id_files"], dup,
           f"<= {GATE['duplicate_id_files']}")

    # 3/4. Hygiene
    sys.path.insert(0, str(HERE))
    import kg_hygiene
    contamination = kg_hygiene.contamination(REBUILD_FACTS)
    record("contamination_dirs", contamination["dirs"] <= GATE["contamination_dirs"],
           contamination["dirs"], f"<= {GATE['contamination_dirs']}",
           "a directory holding facts about another entity means a merge went wrong")
    near = kg_hygiene.near_duplicates(REBUILD_FACTS)
    results["near_duplicates"] = near

    # 5. Junk entity names
    from app.entity_naming import looks_like_junk_entity
    src = {r["entity"]: r["source_doc"] for r in st._query(
        "SELECT entity, MIN(source_doc) source_doc FROM facts_idx "
        "WHERE source_doc IS NOT NULL GROUP BY entity")}
    junk = [d.name for d in REBUILD_FACTS.iterdir()
            if d.is_dir() and looks_like_junk_entity(d.name, src.get(d.name))]
    record("junk_entity_dirs", len(junk) <= GATE["junk_entity_dirs"], len(junk),
           f"<= {GATE['junk_entity_dirs']}", f"e.g. {junk[:3]}" if junk else "")

    # 6. Corpus coverage. Without this the gate would pass a tree built from
    #    80% of the vault: every structural check is a RATIO, and a rebuild
    #    that stopped early looks just as clean as one that finished.
    #    Documents that failed extraction are deliberately not content-hashed,
    #    so the hash index is the count of documents actually extracted.
    hashes = VAULT_DERIVED_ROOT / "rebuild-content-hashes.json"
    try:
        extracted = int(json.loads(hashes.read_text())["file_count"])
    except Exception:
        extracted = 0
    corpus = _corpus_size()
    cov = round(100.0 * extracted / corpus, 2) if corpus else 0.0
    record("corpus_coverage_pct", cov >= GATE["corpus_coverage_pct"], cov,
           f">= {GATE['corpus_coverage_pct']}",
           f"{extracted:,} of {corpus:,} documents extracted")
    results["corpus"] = {"extracted": extracted, "corpus": corpus,
                         "failed_or_pending": max(0, corpus - extracted)}

    # 7. Node coverage
    coverage = round(100.0 * len(st.edges.nodes()) / stats["entities"], 2) if stats["entities"] else 0.0
    record("node_coverage_pct", coverage >= GATE["node_coverage_pct"], coverage,
           f">= {GATE['node_coverage_pct']}", "share of entities in at least one edge")
    st.close()

    # 8. Retrieval, against the tree we are about to swap in.
    before = state.get("eval_before") or {}
    after = _run_eval("rebuild-after", run_dir) if not args.skip_eval else {}
    results["eval_before"], results["eval_after"] = before, after
    if after and before:
        for key, gate_key in (("mrr_doc", "eval_mrr_slack"), ("ndcg10", "eval_ndcg_slack")):
            got, want = after.get(key), before.get(key)
            ok = got is not None and want is not None and got >= want - GATE[gate_key]
            record(f"eval_{key}", ok, got, f">= {want}")
        worst = None
        for cat, b in (before.get("by_category") or {}).items():
            a = (after.get("by_category") or {}).get(cat, {})
            if b.get("mrr_doc") is None or a.get("mrr_doc") is None:
                continue
            delta = a["mrr_doc"] - b["mrr_doc"]
            if worst is None or delta < worst[1]:
                worst = (cat, delta)
        if worst:
            record("eval_category_regression", worst[1] >= -GATE["eval_category_regression"],
                   f"{worst[0]} {worst[1]:+.3f}", f">= -{GATE['eval_category_regression']}")
    elif not args.skip_eval:
        record("eval", False, "no result", "a completed eval run",
               "the gate cannot pass without a retrieval measurement")

    results["pass"] = all(c["pass"] for c in results["checks"].values())
    out = run_dir / "gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    save_state(gate=results["pass"], gate_file=str(out))
    print(f"\ngate: {'PASS' if results['pass'] else 'FAIL'}  ->  {out}")
    return 0 if results["pass"] else 1


# ── swap / rollback ──────────────────────────────────────────────────────────

def _facts_written_since_export(state: dict) -> list[dict]:
    """Hand-stated facts added to the LIVE tree after the carry-over export.

    The rebuild runs for hours and the system stays usable throughout, so a
    fact stated in a chat turn tonight lands in the tree that `swap` renames
    to facts-quarantine-<ts>. Anything found here has to be re-exported
    before the swap or it is lost.
    """
    exported_at = (state.get("export") or {}).get("exported_at")
    if not exported_at:
        return []
    st = KGStore(VAULT_KG_DB)
    try:
        rows = st._query(
            "SELECT entity, category, fact, created_at, provenance FROM facts_idx "
            "WHERE created_at > ? AND (provenance IN ('STATED','INFERRED','AMBIGUOUS') "
            "OR source_doc LIKE '%session%') ORDER BY created_at",
            (exported_at,))
        return [{k: r[k] for k in r.keys()} for r in rows]
    finally:
        st.close()


def cmd_swap(args) -> int:
    """Two renames and a reindex. Refuses without a passing gate."""
    state = load_state()
    if not state.get("gate") and not args.force:
        print("gate has not passed; run `gate` first (or --force, which you should not)",
              file=sys.stderr)
        return 2
    if not REBUILD_FACTS.is_dir():
        print(f"no rebuild tree at {REBUILD_FACTS}", file=sys.stderr)
        return 2

    missed = _facts_written_since_export(state)
    if missed and not args.force:
        print(f"REFUSING: {len(missed)} hand-stated fact(s) were added to the live tree "
              f"after the carry-over export at {(state.get('export') or {}).get('exported_at')}.",
              file=sys.stderr)
        for m in missed[:10]:
            print(f"  {m['created_at']}  {m['entity']}/{m['category']}: {(m['fact'] or '')[:70]}",
                  file=sys.stderr)
        if len(missed) > 10:
            print(f"  … and {len(missed) - 10} more", file=sys.stderr)
        print("\nRe-run `export` then `import`, and swap after that. Those facts came "
              "from a conversation and re-extraction cannot reproduce them.", file=sys.stderr)
        return 3

    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%SZ")
    quarantine = VAULT_DERIVED_ROOT / f"facts-quarantine-{ts}"
    old_db = VAULT_DERIVED_ROOT / f"kg-quarantine-{ts}.sqlite"

    if args.dry_run:
        print(f"would move {VAULT_FACTS_ROOT} -> {quarantine}")
        print(f"would move {REBUILD_FACTS} -> {VAULT_FACTS_ROOT}")
        print(f"would move {VAULT_KG_DB} -> {old_db}")
        print(f"would move {REBUILD_DB} -> {VAULT_KG_DB}")
        return 0

    VAULT_FACTS_ROOT.rename(quarantine)
    REBUILD_FACTS.rename(VAULT_FACTS_ROOT)
    if VAULT_KG_DB.exists():
        VAULT_KG_DB.rename(old_db)
    for suffix in ("-wal", "-shm"):
        stale = Path(str(VAULT_KG_DB) + suffix)
        if stale.exists():
            stale.unlink()
    REBUILD_DB.rename(VAULT_KG_DB)

    st = KGStore(VAULT_KG_DB)
    st.facts_idx.reindex(root=VAULT_FACTS_ROOT)
    stats = st.stats()
    st.close()

    _set_write_enabled(True)
    save_state(swapped_at=_now(), quarantine=str(quarantine), quarantine_db=str(old_db),
               after_swap=stats)
    print(f"swapped. quarantine: {quarantine}\nstore: {stats}")
    print("\nRestart the backend and the aggregator when idle:")
    print("  supervisorctl -c agent-services/supervisor/supervisord.conf "
          "restart lloyd-mc:lloyd-mcp lloyd-mc:lloyd-backend")
    print("Then un-pause #24, #48, #74.")
    return 0


def cmd_rollback(args) -> int:
    state = load_state()
    quarantine = Path(state.get("quarantine") or "")
    old_db = Path(state.get("quarantine_db") or "")
    if not quarantine.is_dir():
        print("no quarantined tree recorded", file=sys.stderr)
        return 2
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%SZ")
    VAULT_FACTS_ROOT.rename(VAULT_DERIVED_ROOT / f"facts-rolledback-{ts}")
    quarantine.rename(VAULT_FACTS_ROOT)
    if old_db.exists():
        if VAULT_KG_DB.exists():
            VAULT_KG_DB.rename(VAULT_DERIVED_ROOT / f"kg-rolledback-{ts}.sqlite")
        old_db.rename(VAULT_KG_DB)
    save_state(rolled_back_at=_now())
    print("rolled back. Restart the backend and the aggregator.")
    return 0


def cmd_status(args) -> int:
    state = load_state()
    print(json.dumps(state, indent=2, default=str))
    if REBUILD_FACTS.is_dir():
        dirs = sum(1 for d in REBUILD_FACTS.iterdir() if d.is_dir())
        files = sum(1 for _ in REBUILD_FACTS.rglob("*.md"))
        print(f"\nrebuild tree: {dirs:,} entity dirs, {files:,} fact files")
    if REBUILD_DB.exists():
        st = KGStore(REBUILD_DB)
        print(f"rebuild store: {st.stats()}")
        st.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("freeze")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-writes", action="store_true",
                   help="Leave fact writes enabled. The extraction runs for hours and "
                        "the system stays useful; `swap` refuses if anything was stated "
                        "in the meantime and not re-exported.")
    p.set_defaults(fn=cmd_freeze)
    sub.add_parser("export").set_defaults(fn=cmd_export)
    p = sub.add_parser("extract")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--passes", type=int, default=3,
                   help="Re-run to sweep up documents that timed out (default 3). "
                        "Stops early when a pass extracts nothing new.")
    p.set_defaults(fn=cmd_extract)
    sub.add_parser("import").set_defaults(fn=cmd_import)
    p = sub.add_parser("gate"); p.add_argument("--skip-eval", action="store_true")
    p.set_defaults(fn=cmd_gate)
    p = sub.add_parser("swap")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_swap)
    sub.add_parser("rollback").set_defaults(fn=cmd_rollback)
    sub.add_parser("status").set_defaults(fn=cmd_status)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
