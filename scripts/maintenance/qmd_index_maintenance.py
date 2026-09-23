#!/usr/bin/env python3
"""
qmd index maintenance — nightly orphan prune + embedding backfill.

WHY THIS EXISTS (2026-08-06, backlog #380):
qmd never prunes its own index. Orphaned embedding chunks — rows in
`content_vectors` whose `hash` no longer exists in `content` — accumulate on
every re-index. Left alone for months they reached **2,948,805 of 2,962,138
rows (99.5%)**, a 24 GB index, and 700 ms of single-core CPU per vector query
(the vec search brute-force scans every vector and applies the collection
filter afterwards, so cost is O(total vectors) regardless of query scope).

It is not only a speed problem: orphans displace real results. A vec-only query
returned 4 hits before the first cleanup and 20 after, and eval multi-hop MRR
went 0.022 → 0.220.

Separately, `qmd status` can report documents that are indexed but never
embedded. 406 such documents had accumulated because embedding silently stalled
while the vec leg was crash-looping.

SAFETY CONTRACT — this runs unattended:
  * The daemon is NOT stopped. It was, until 2026-09-19: published 2.8.3's
    cleanup wanted the index to itself, and the stop stayed after the daemon
    moved to the fork, whose `cleanup` and `embed` are safe beside a running
    daemon (the watcher embeds under it all day, and
    `lloyd-qmd-cleanup.timer` cleans under it every night). Pending embeddings
    are never zero while the automod loop is writing, so "only when there is
    real work" meant every run: eight of the last eight took retrieval down
    and brought it back cold -- models reloaded, vector index rebuilt -- to
    embed one to four documents the watcher would have reached anyway.
  * A daemon found unhealthy after the mutating branch is still restarted, and
    being left unhealthy is still the one thing that makes the exit code non-zero.
    A daemon found unhealthy on a run that did nothing is NOT restarted: this run
    did not break it, and a quiet night is not the job's moment to restart a
    service it never touched. It reports and exits 1.
  * Every subprocess has a timeout.
  * The embedding backfill refuses a corpus-wide re-embed (#1367). Pending is
    counted per *configured model*, so editing one line — `models: embed` in
    ~/.config/qmd/index.yml — makes every hash in the index read as unembedded,
    and `pending_embeddings() > 0` used to turn that into an hour of
    `qmd embed` against the live index beside the serving daemon, unattended,
    nightly. Past EMBED_PENDING_MAX_RATIO of the index the run now records
    `model_change_suspected` and declines to embed; verify the switch was
    deliberate and run `qmd embed` by hand. A refused guard exits 0 — it is the
    check working, not a failed job — and is a report entry like drift.
  * Exit code is non-zero only when the daemon is left unhealthy — a failed
    prune with a healthy daemon is a warning, not a page. Every run probes for it,
    including one that decided there was nothing to do (#958): before that, the
    no-op path returned 0 having never measured, which was the case that quietly
    passed. "Unhealthy" means the daemon answered no HTTP request on
    DAEMON_PROBE_URL within NOOP_HEALTH_RETRIES tries -- a listening-but-wedged
    daemon still reads healthy, and this job's probe does not claim otherwise.
  * Every run also compares the tracked qmd collection template with the config
    the daemon actually reads, and records it as `config_drift` in the dated
    report (#1298). Drift is a report entry and never an exit code: the job that
    fixes embeddings must not start failing over a stale tracked copy.

Usage:
  python scripts/maintenance/qmd_index_maintenance.py            # act if needed
  python scripts/maintenance/qmd_index_maintenance.py --dry-run  # report only
  python scripts/maintenance/qmd_index_maintenance.py --force    # prune anyway
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.paths import PIPELINE_DIR  # noqa: E402

# The fork in ~/lloyd/qmd -- the build the daemon serves this index with, and
# since 2026-09-19 the only qmd on the machine. Until then this was the
# published @tobilu/qmd under ~/.bun: same version string (2.8.3), different
# commit, so the index had two writers that were not the reader.
# tests/test_qmd_single_build.py pins every caller to this one path.
QMD_CLI = Path.home() / "lloyd/qmd/dist/cli/qmd.js"
INDEX = Path.home() / ".cache/qmd/index.sqlite"
SUPERVISORCTL = Path.home() / ".local/share/uv/tools/supervisor/bin/supervisorctl"
SUPERVISOR_CONF = Path.home() / "lloyd/agent-services/supervisor/supervisord.conf"
SERVICE = "agent-qmd-daemon"
REPORT_DIR = PIPELINE_DIR / "reflection"

# The two qmd collection definitions. `LIVE_CONFIG` is the one the daemon reads
# and serves; `TEMPLATE_CONFIG` is the tracked copy an operator, a restore or a
# new host reads (SETUP.md "Collections"). They are separate files with no
# installer between them — nothing at runtime opens the template — so the only
# thing keeping them agreeing is the manual re-sync SETUP.md:631-635 prescribes,
# and between 2026-09-07 and 2026-09-19 nobody ran it: the template kept a
# `facts` collection the live file dropped and pointed `sessions` at
# ~/obsidian/sessions while the daemon had been indexing
# ~/lloyd/_pipeline/vault-derived/sessions the whole time (#1298).
REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_CONFIG = REPO_ROOT / "agent-services/conf/qmd-index.yml"
LIVE_CONFIG = Path.home() / ".config/qmd/index.yml"
# The documented re-sync, in the direction that reconciles a diverged template:
# the daemon's file is what is true, so it is copied *onto* the template
# (SETUP.md:631-635). Run from the repo root.
RESYNC_COMMAND = "cp ~/.config/qmd/index.yml agent-services/conf/qmd-index.yml"
RESYNC_DIRECTION = "live -> template (the file the daemon reads is the truth)"

# Prune when orphans exceed this share of all vectors. The threshold was set
# when a cleanup cost a daemon stop; it no longer does, and the nightly
# `lloyd-qmd-cleanup.timer` prunes unconditionally, so this is the backstop for
# a day that outruns it (2026-09-19: 17% by evening, fifteen hours after the
# 04:45 cleanup) rather than the primary.
ORPHAN_RATIO_TRIGGER = 0.20
# Floor, in absolute rows, so a nearly-empty index doesn't trip the ratio on
# noise. It is a floor and not a second gate: the two were ANDed at 50,000,
# which on a ~21,000-vector live corpus meant the ratio had to reach ~70%
# before the AND could pass — the absolute trigger sat above the entire live
# index, so `ORPHAN_RATIO_TRIGGER` was unreachable and the prune never ran on
# ratio alone. Found 2026-09-07 at 65% orphans (38,910 of 59,897) with
# `need_prune: false`, costing ~49% of every vec query. A threshold that can
# only fire when the corpus is mostly garbage is not a safety margin.
ORPHAN_ABS_TRIGGER = 2_000
# Cap on the unattended embedding backfill (#1367), as a fraction of the
# `documents` count this job already reports. It exists because `qmd status`'s
# pending number is counted per *configured model*:
# `getHashesNeedingEmbedding` (qmd/src/store.ts:2561-2580) LEFT-JOINs on
# `model` + `embed_fingerprint` rather than on the content hash, so editing the
# single `models: embed` line in ~/.config/qmd/index.yml to any other
# same-dimension model makes every hash read as unembedded. Measured read-only
# against the live index at triage (2026-09-22): 10,798 pending = every active
# distinct hash, over 16,127 `documents` rows — ratio 0.67. The ordinary nightly
# figure in the three most recent reports is 0, 3 or 2, so this fraction sits
# orders of magnitude clear of real work and is nowhere near tuning away: what
# it separates is a handful of new documents from a full rewrite of the live
# vector table while the daemon serves from it. Two models cannot share
# `vectors_vec` — it is keyed `hash_seq` with no model column, and only a
# *dimension* change is caught (qmd/src/store.ts:1518-1523) — so an interrupted
# same-dimension re-embed leaves both models' vectors in one cosine table,
# indistinguishable at query time. That is the state being protected, not the
# hour of GPU.
EMBED_PENDING_MAX_RATIO = 0.25

QMD_ENV = {
    "HOME": str(Path.home()),
    "PATH": f"/opt/cuda/bin:{Path.home()}/.local/bin:/usr/local/bin:/usr/bin:/bin",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "0",
    "CUDA_PATH": "/opt/cuda",
    "CUDA_HOME": "/opt/cuda",
    "LD_LIBRARY_PATH": "/usr/lib:/opt/cuda/lib64",
    "QMD_VEC_BACKEND": "bit",
}


def _sh(cmd: list[str], timeout: int, env: dict | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return 1, repr(e)


def supervisor(action: str, timeout: int = 120) -> tuple[int, str]:
    return _sh([str(SUPERVISORCTL), "-c", str(SUPERVISOR_CONF), action, SERVICE], timeout)


def inspect_index() -> dict:
    """Read orphan counts straight from SQLite (read-only, daemon can be up)."""
    out: dict = {"index_bytes": INDEX.stat().st_size if INDEX.exists() else 0}
    if not INDEX.exists():
        out["error"] = "index missing"
        return out
    try:
        con = sqlite3.connect(f"file:{INDEX}?mode=ro", uri=True, timeout=30)
        q = lambda s: con.execute(s).fetchone()[0]  # noqa: E731
        out["vectors_total"] = q("select count(*) from content_vectors")
        out["vectors_orphaned"] = q(
            "select count(*) from content_vectors v "
            "left join content c on v.hash = c.hash where c.hash is null"
        )
        out["documents"] = q("select count(*) from documents")
        con.close()
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
        return out
    tot = out.get("vectors_total") or 0
    out["vectors_live"] = tot - out.get("vectors_orphaned", 0)
    out["orphan_ratio"] = round(out["vectors_orphaned"] / tot, 4) if tot else 0.0
    return out


def pending_embeddings() -> int:
    """Docs indexed but not embedded, per `qmd status`. -1 if unreadable."""
    rc, out = _sh(["/usr/bin/node", str(QMD_CLI), "status"], 180, env=QMD_ENV)
    if rc != 0:
        return -1
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Pending:"):
            digits = "".join(ch for ch in s if ch.isdigit())
            return int(digits) if digits else 0
    return 0


def configured_embed_model(path: Path = LIVE_CONFIG) -> str | None:
    """The embed model from the config's `models: embed`, or None if unreadable.

    Read from the file the daemon actually reads — `~/.config/qmd/index.yml`,
    which overrides `QMD_EMBED_MODEL` (src/llm.ts::resolveEmbedModel). This is
    the one line whose edit turns the whole index pending, so it is what a
    reader of the report needs; None is a legitimate answer and never a reason to
    skip the guard.
    """
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception:  # noqa: BLE001 — unreadable config is not this check's failure
        return None
    if not isinstance(data, dict):
        return None
    models = data.get("models")
    if not isinstance(models, dict):
        return None
    model = models.get("embed")
    return str(model) if model else None


def embed_backfill_guard(pend: int, documents: int | None,
                         configured_model: str | None) -> dict:
    """Decide whether `pend` is a backfill or a whole-index re-embed (#1367).

    The denominator is `documents` — the row count `inspect_index` already puts
    in the report — not the pending count itself, so the ratio stays auditable
    from the file alone. It is deliberately the *rows* figure and not the
    distinct-hash figure the pending number counts (16,127 rows over 10,798
    distinct hashes on the live index: many documents share one hash), which is
    why the denominator is named in the verdict rather than implied.

    An index of zero documents with something pending trips: with nothing to
    compare against there is no ratio, and the unattended path is the one that
    has to fail closed.
    """
    docs = documents or 0
    max_pending = int(EMBED_PENDING_MAX_RATIO * docs)
    return {
        "pending_embeddings": pend,
        "denominator": "documents (rows in the `documents` table, from inspect_index)",
        "denominator_value": docs,
        "ratio": round(pend / docs, 4) if docs else None,
        "threshold_fraction": EMBED_PENDING_MAX_RATIO,
        "max_pending": max_pending,
        "configured_embed_model": configured_model,
        "tripped": pend > max_pending,
    }


#: The daemon endpoint both health probes hit. It is a named constant because the
#: no-op path probes it too (#958), and the test that pins "the health line came
#: from a real probe" has to assert on the same string the probe sends.
DAEMON_PROBE_URL = "http://localhost:8181/mcp"

#: Tries the *no-op* path spends on that probe. The mutating path's default of 10
#: (with a 3 s sleep between tries) exists to wait out a restart it just performed;
#: a run that pruned and embedded nothing restarted nothing, so the ~30 s that
#: default can cost was pure waiting on a daemon that was already down. A few tries
#: still ride out a transient blip instead of turning a 3 s hiccup into a failed
#: nightly run.
NOOP_HEALTH_RETRIES = 3


def daemon_healthy(retries: int = 10) -> bool:
    """True once the daemon answers an HTTP request on its MCP endpoint.

    *Answers*, not "retrieval verified": curl exits 0 on any HTTP response, and the
    MCP endpoint replies 405 to a GET, so a listening-but-wedged daemon reads
    healthy here. Telling that apart needs a real query (`qmd vsearch`), which is a
    different change than the one that asked for this probe.
    """
    for _ in range(retries):
        rc, _ = _sh(["curl", "-s", "-m", "3", "-o", "/dev/null", DAEMON_PROBE_URL], 10)
        if rc == 0:
            return True
        time.sleep(3)
    return False


def _qmd_collections(
    path: Path,
) -> tuple[dict[str, dict] | None, str | None, list[str]]:
    """Read a qmd config's `collections:` block.

    Returns `(collections, note, malformed)`. `collections` is None when there is
    nothing to compare — the file is absent, or it could not be parsed — with the
    reason in `note`; a dict (possibly empty) when it read. An absent file and an
    empty collections block are different answers, so they do not come back the
    same. `malformed` names the collections whose body is not a mapping: they are
    carried with no path, so the caller has to say it skipped them rather than
    let two of them read as an agreement.
    """
    if not path.exists():
        return None, f"no qmd config at {path}", []
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception as e:  # noqa: BLE001 — a broken config is reported, never raised here
        return None, f"unparsable qmd config at {path}: {e!r}", []
    if not isinstance(data, dict):
        return None, f"qmd config at {path} is not a mapping", []
    colls = data.get("collections") or {}
    if not isinstance(colls, dict):
        return None, f"qmd config at {path} has no collections mapping", []
    # A collection whose body is not a mapping — `vault: qmd` where
    # `vault:\n    path: ...` belongs — is a shape this has to survive (found in
    # review, 2026-09-20): `dict(spec)` raises ValueError on a string, so one
    # mistyped line took the whole nightly job down. Keep the collection with no
    # known path and name it, rather than raising or quietly dropping it.
    out: dict[str, dict] = {}
    malformed: list[str] = []
    for name, spec in colls.items():
        if isinstance(spec, dict):
            out[name] = dict(spec)
        else:
            out[name] = {}
            malformed.append(str(name))
    return out, None, malformed


def config_drift(template: Path = TEMPLATE_CONFIG, live: Path = LIVE_CONFIG) -> dict:
    """Compare the committed template with the config the daemon actually reads.

    Both files define the same set of qmd collections by hand, and only by hand:
    SETUP.md:619 installs template -> live and SETUP.md:631-635 re-syncs live ->
    template, and no code enforces either. So the failure mode is silent — the
    2026-09-19 live edit retargeted `sessions` onto the real export directory
    (~650 indexed documents) and deleted `facts`, and the tracked copy still
    described the 2026-09-07 world. A reindex from the stale template drops that
    collection from both the FTS and the vector legs.

    Compared: which collections exist, and each one's `path`.
    Deliberately NOT compared: collection *order* and comments — a re-sync
    copies whole files, so those reconcile themselves and calling them drift
    would cry wolf every time the block is re-sorted — nor `pattern`/`ignore`,
    which this check has never claimed to cover.

    Never raises and never changes the job's exit code: with either file missing
    there is nothing to compare, and that is a note in the report, not a failure.
    A malformed file — unparseable, or a `collections:` block that is not a
    mapping — is the same kind of answer, and so is a single collection whose
    body will not read: it is named in `malformed` and skipped, never scored as
    agreement.
    """
    out: dict = {
        "template": str(template),
        "live": str(live),
        "drift": [],
        "drift_count": 0,
        "resync_command": RESYNC_COMMAND,
        "resync_direction": RESYNC_DIRECTION,
    }
    tmpl, tmpl_note, tmpl_bad = _qmd_collections(template)
    livec, live_note, live_bad = _qmd_collections(live)
    if tmpl_note:
        out["note"] = tmpl_note
        return out
    if live_note:
        out["note"] = live_note
        return out
    out["comparable"] = True
    if tmpl_bad or live_bad:
        out["malformed"] = {"template": tmpl_bad, "live": live_bad}

    for name in sorted(set(tmpl) | set(livec)):
        # Order matters. Which collections exist is answerable even when one
        # body will not read, so presence is judged first; only a collection
        # declared on both sides with an unreadable body is genuinely
        # uncomparable, and that is never scored as agreement.
        if name not in livec:
            out["drift"].append(
                {"collection": name, "kind": "template_only",
                 "template_path": tmpl[name].get("path"), "live_path": None}
            )
        elif name not in tmpl:
            out["drift"].append(
                {"collection": name, "kind": "live_only",
                 "template_path": None, "live_path": livec[name].get("path")}
            )
        elif name in tmpl_bad or name in live_bad:
            # Declared on both sides, but one side's path is unknown, so no
            # comparison of it means anything. Name it rather than reporting
            # either "no drift" or a path change that was never made.
            out["drift"].append(
                {"collection": name, "kind": "uncomparable_body",
                 "template_path": tmpl[name].get("path"),
                 "live_path": livec[name].get("path")}
            )
        elif tmpl[name].get("path") != livec[name].get("path"):
            out["drift"].append(
                {"collection": name, "kind": "path_differs",
                 "template_path": tmpl[name].get("path"),
                 "live_path": livec[name].get("path")}
            )

    out["drift_count"] = len(out["drift"])
    out["in_sync"] = out["drift_count"] == 0
    return out


def _write_report(report: dict, started: datetime) -> Path:
    """Land the dated JSON report, on every run that did something or nothing.

    It used to be written only when a prune or an embed actually ran, which is a
    handful of nights a month — enough that a reader looking for the nightly
    drift verdict mostly found no file at all. The config comparison is only
    worth running nightly if its answer is on disk nightly.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"qmd-index-maintenance-{started:%Y-%m-%d}.json"
    out.write_text(json.dumps(report, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="qmd index maintenance (#380)")
    ap.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    ap.add_argument("--force", action="store_true", help="prune regardless of triggers")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    started = datetime.now()
    report: dict = {"ran_at": started.isoformat(), "actions": []}

    before = inspect_index()
    report["before"] = before
    pend = pending_embeddings()
    report["pending_embeddings"] = pend
    # Read the two collection definitions fresh, from the module-level paths, so
    # a caller (or a test) that moves those moves this check with them.
    report["config_drift"] = config_drift(TEMPLATE_CONFIG, LIVE_CONFIG)

    need_prune = args.force or (
        before.get("orphan_ratio", 0) >= ORPHAN_RATIO_TRIGGER
        and before.get("vectors_orphaned", 0) >= ORPHAN_ABS_TRIGGER
    )
    # A dead env var is worse than no env var: QMD_VEC_BACKEND is set in
    # QMD_ENV and qmd 2.8.3 reads it nowhere (it appears in no dist/*.js).
    # Left in place deliberately — removing it is a separate change — but do
    # not add tuning here expecting it to take effect.
    need_embed = pend > 0
    # The pending number is per configured model, so it is not a count of new
    # documents — see EMBED_PENDING_MAX_RATIO. Read fresh, from the same module
    # global the drift check reads, so a caller that moves LIVE_CONFIG moves this.
    guard = embed_backfill_guard(pend, before.get("documents"),
                                 configured_embed_model(LIVE_CONFIG))
    report["embed_guard"] = guard
    if need_embed and guard["tripped"]:
        need_embed = False
        report["model_change_suspected"] = {
            "pending_embeddings": guard["pending_embeddings"],
            "denominator": guard["denominator"],
            "denominator_value": guard["denominator_value"],
            "ratio": guard["ratio"],
            "threshold_fraction": guard["threshold_fraction"],
            "max_pending": guard["max_pending"],
            "configured_embed_model": guard["configured_embed_model"],
            "why": ("pending is counted per configured model, so a number this "
                    "large means `models: embed` names a model the index's "
                    "existing vectors were not built with — not that new "
                    "documents are waiting. An interrupted same-dimension "
                    "re-embed leaves two models' vectors in one `vectors_vec` "
                    "table, which has no model column and only catches a "
                    "dimension change."),
            "what_to_do": ("confirm the model switch was deliberate, then run "
                           "`qmd embed` by hand against ~/.cache/qmd/index.sqlite "
                           "(or rebuild a side copy and swap, as the 2026-09-21 "
                           "switch did); this job will not do it unattended."),
        }
        report["actions"].append(
            f"embed SKIPPED — model_change_suspected: pending "
            f"{guard['pending_embeddings']:,} exceeds "
            f"{guard['threshold_fraction']:.0%} of "
            f"{guard['denominator']}: {guard['denominator_value']:,} "
            f"(max {guard['max_pending']:,}); configured embed model "
            f"{guard['configured_embed_model']}")
    report["need_prune"], report["need_embed"] = need_prune, need_embed

    if args.dry_run or not (need_prune or need_embed):
        if args.dry_run:
            report["actions"].append("dry-run")
        elif not guard["tripped"]:
            # A refused guard already recorded its own reason; "nothing to do"
            # would contradict the line above it in the same report.
            report["actions"].append("none — nothing to do")
        # Doing nothing is not evidence that retrieval is up, so this branch
        # measured it too: it used to `return 0` without ever calling
        # `daemon_healthy()`, whose only call sat in the mutating section's
        # `finally`, and the exit line's `True` default turned that absence into a
        # pass. A daemon that died on a quiet night was a green nightly run (#958).
        # One curl, NOOP_HEALTH_RETRIES tries, no lock, no stop, no restart: a
        # daemon found unhealthy on a no-work day is reported and exited on, and
        # the decision to restart it stays with the branch that touched the index.
        report["daemon_healthy"] = daemon_healthy(retries=NOOP_HEALTH_RETRIES)
        # --dry-run promises to change nothing, so it only prints. Everything
        # else that runs leaves a report, drift included, whether or not it
        # needed the index. A probe changes nothing either, which is why `--dry-run`
        # shares this branch instead of getting its own health semantics.
        if not args.dry_run:
            _write_report(report, started)
        _emit(report, args.json)
        return 0 if report["daemon_healthy"] else 1

    # ---- Mutating section. The daemon stays up throughout (see the contract). ----
    try:
        if need_prune:
            t = time.time()
            rc, out = _sh(["/usr/bin/node", str(QMD_CLI), "cleanup"], 5400, env=QMD_ENV)
            report["actions"].append(
                f"cleanup rc={rc} in {time.time()-t:.0f}s :: {out.strip().splitlines()[-1] if out.strip() else ''}"
            )
            report["cleanup_ok"] = rc == 0
        if need_embed:
            t = time.time()
            rc, out = _sh(["/usr/bin/node", str(QMD_CLI), "embed"], 5400, env=QMD_ENV)
            report["actions"].append(f"embed rc={rc} in {time.time()-t:.0f}s")
            report["embed_ok"] = rc == 0
    finally:
        healthy = daemon_healthy()
        if not healthy:
            # Not something this run did to it -- but leaving retrieval down is
            # the one outcome worse than a bloated index, whoever caused it.
            rc, _ = supervisor("restart")
            report["actions"].append(f"daemon unhealthy -> restart rc={rc}")
            healthy = daemon_healthy()
        report["daemon_healthy"] = healthy

    report["after"] = inspect_index()
    report["elapsed_s"] = round((datetime.now() - started).total_seconds(), 1)

    _write_report(report, started)

    _emit(report, args.json)
    # Only fail loudly if retrieval is actually down. Indexed, not `.get(..., True)`:
    # the `finally` above always sets the key, and both exits now read a measured
    # boolean, so a missing measurement crashes instead of passing.
    return 0 if report["daemon_healthy"] else 1


def _emit(r: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(r, indent=2))
        return
    b, a = r.get("before", {}), r.get("after")
    mb = lambda n: f"{(n or 0)/1e6:.0f} MB"  # noqa: E731
    print("qmd index maintenance")
    print(f"  index size        {mb(b.get('index_bytes'))}"
          + (f"  →  {mb(a.get('index_bytes'))}" if a else ""))
    print(f"  vectors total     {b.get('vectors_total', 0):,}"
          + (f"  →  {a.get('vectors_total', 0):,}" if a else ""))
    print(f"  orphaned          {b.get('vectors_orphaned', 0):,} "
          f"({100*b.get('orphan_ratio', 0):.1f}%)"
          + (f"  →  {a.get('vectors_orphaned', 0):,}" if a else ""))
    print(f"  pending embeds    {r.get('pending_embeddings')}")
    g = r.get("embed_guard")
    if g:
        # The denominator has to be printed with the number: pending counts
        # distinct hashes and `documents` counts rows, and a ratio whose base is
        # unnamed cannot be audited afterwards from the run's own output.
        print(f"  embed cap         {g['threshold_fraction']:.0%} of "
              f"{g['denominator']}: {g['denominator_value']:,} "
              f"= max {g['max_pending']:,}"
              + (f"  → REFUSED (model_change_suspected); configured embed model "
                 f"{g['configured_embed_model']}" if g["tripped"] else ""))
    print(f"  prune needed      {r.get('need_prune')}   embed needed {r.get('need_embed')}")
    cd = r.get("config_drift")
    if cd:
        if cd.get("note"):
            print(f"  qmd config        not compared: {cd['note']}")
        elif cd.get("drift_count"):
            print(f"  qmd config drift  {cd['drift_count']} collection difference(s)")
            for d in cd["drift"]:
                print(f"    · {d['collection']} [{d['kind']}] "
                      f"template={d.get('template_path')} live={d.get('live_path')}")
            print(f"    re-sync ({cd.get('resync_direction')}):")
            print(f"      {cd.get('resync_command')}")
        else:
            print("  qmd config        template and live collections agree")
    for act in r["actions"]:
        print(f"    · {act}")
    if "daemon_healthy" in r:
        print(f"  daemon healthy    {r['daemon_healthy']}")
    if "elapsed_s" in r:
        print(f"  elapsed           {r['elapsed_s']}s")


if __name__ == "__main__":
    raise SystemExit(main())
