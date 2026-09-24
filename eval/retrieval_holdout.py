"""The retrieval eval's never-read holdout leg: its manifest and its reserve rule (#1412).

`vault_recall_queries.yaml` (the `dev` leg) is what every nightly, every trend claim
and every post-landing paired check scores, so it is the set retrieval changes are
selected against. `vault_recall_holdout_queries.yaml` is a disjoint tranche that only
the holdout leg of the paired check reads, and only as aggregates. A promotion that
gains on dev past the floor while losing on holdout past it is recorded
`overfit_suspected` (`workers/sources/automod_regression.py`).

This is `scripts/autoresearch/bench_split.py` (#549) ported to the other corpus, and
the mechanism is the same on purpose — "implement one split and share it":

* the reserved ids are written to a manifest with a `split_hash` over them, so the
  split cannot be re-picked after results are seen;
* `verify` RECOMPUTES the hash and never trusts the recorded one, and a manifest
  whose id pool or content was edited after the hash was written is refused;
* a refused manifest is reported as absent, and absent means the holdout leg does
  not run (the stricter side), never that it runs on an unverified set.

Unlike the bench split there is no per-round rotation: the tranche is small (~27) and
hand-authored, and rotating it would re-measure the transfer gap on a different
denominator every promotion. The sanctioned change is to retire the whole tranche into
dev and author a fresh one, then `python -m eval.retrieval_holdout write`.

Stdlib + PyYAML only: `eval/run_eval.py` imports it inside the eval subprocess, and the
paired check imports it from the backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
DEV_FILENAME = "vault_recall_queries.yaml"
HOLDOUT_FILENAME = "vault_recall_holdout_queries.yaml"
MANIFEST_FILENAME = "vault_recall_holdout_manifest.json"

DEV_QUERIES = HERE / DEV_FILENAME
HOLDOUT_QUERIES = HERE / HOLDOUT_FILENAME
MANIFEST = HERE / MANIFEST_FILENAME

# The one switch that lets `eval/run_eval.py` score the holdout corpus. An env var,
# not a CLI flag: the paired check's BASELINE arm runs the parent commit's
# `run_eval.py`, and a parent that predates a new flag would exit 2 on it (the
# failure `autonomy/82-nightly-retrieval-eval.md` records for the nightly).
HOLDOUT_LEG_ENV = "LLOYD_EVAL_HOLDOUT_LEG"

# Labels the nightly and its trend report glob. A holdout run may never carry one,
# or its per-query rows would land in the files every nightly reader opens.
FORBIDDEN_LABEL_PREFIXES = ("nightly",)

RESERVE_RULE = ("only the holdout leg of the post-landing paired check scores these ids; "
                "it records aggregate deltas, never per-query rows or ids; no nightly, "
                "trend report, prompt or tuning job reads them")


def _digest(*parts: str) -> str:
    """`bench_split._digest`, same framing, so the two hashes are one idea."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def load_specs(path: Path) -> list[dict]:
    return list((yaml.safe_load(Path(path).read_text()) or {}).get("queries") or [])


def query_ids(path: Path) -> list[str]:
    return [str(s.get("id")) for s in load_specs(path)]


def corpus_sha(specs: list[dict]) -> str:
    """Content hash over the parsed specs — comments and layout do not count.

    A relabel, a reworded question or an added query all change it; a comment edit
    does not, because the comments are documentation and the specs are the split.
    """
    canon = sorted((dict(s) for s in specs), key=lambda s: str(s.get("id")))
    return _digest(json.dumps(canon, sort_keys=True, default=str))


def _payload(reserved_ids: list[str], content_sha: str) -> dict[str, Any]:
    return {"leg": "holdout", "reserved_ids": sorted(reserved_ids),
            "n": len(reserved_ids), "corpus_sha": content_sha,
            "reserve_rule": RESERVE_RULE}


def compute_manifest(holdout: Path = HOLDOUT_QUERIES, dev: Path = DEV_QUERIES) -> dict[str, Any]:
    """The manifest for `holdout`. Pure; refuses a tranche that overlaps dev."""
    specs = load_specs(holdout)
    ids = [str(s.get("id")) for s in specs]
    if not ids:
        raise ValueError(f"{holdout} holds no queries — refusing to reserve an empty leg")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{holdout} repeats {len(ids) - len(set(ids))} id(s)")
    overlap = set(ids) & set(query_ids(dev))
    if overlap:
        # A count, never the ids: this message lands in logs and test output.
        raise ValueError(f"{len(overlap)} holdout id(s) also appear in {dev} — the legs "
                         "must be disjoint")
    manifest = _payload(ids, corpus_sha(specs))
    manifest["split_hash"] = _digest(json.dumps(_payload(ids, corpus_sha(specs)), sort_keys=True))
    return manifest


def verify(manifest: dict[str, Any], holdout: Path | None = None) -> bool:
    """Does `split_hash` still describe this manifest — and, given the file, the file?

    Recomputed, never trusted (`bench_split.verify`). With `holdout`, the file's ids
    and content must also be exactly what the manifest reserved: an id added, dropped
    or relabelled after the hash was written is a refusal, not a new split.
    """
    recorded = manifest.get("split_hash")
    if not recorded:
        return False
    ids = [str(i) for i in (manifest.get("reserved_ids") or [])]
    body = _payload(ids, str(manifest.get("corpus_sha") or ""))
    if _digest(json.dumps(body, sort_keys=True)) != recorded:
        return False
    if holdout is not None:
        try:
            specs = load_specs(holdout)
        except (OSError, yaml.YAMLError):
            return False
        if sorted(str(s.get("id")) for s in specs) != sorted(ids):
            return False
        if corpus_sha(specs) != manifest.get("corpus_sha"):
            return False
    return True


def load_manifest(root: Path | None = None) -> dict[str, Any] | None:
    """The verified manifest under `root/eval/`, or None (absent or refused).

    `root` is a checkout; the paired check passes the LIVE tree, so a candidate round
    cannot swap in its own holdout file mid-check (the `LIVE_QUERIES` rail).
    """
    base = (Path(root) / "eval") if root is not None else HERE
    try:
        manifest = json.loads((base / MANIFEST_FILENAME).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    return manifest if verify(manifest, base / HOLDOUT_FILENAME) else None


def reserved_ids(root: Path | None = None) -> set[str]:
    """Ids no non-holdout path may carry. Falls back to the file when unverified.

    The fallback is the stricter side: a tampered manifest must not make the reserve
    rule forget what it was reserving.
    """
    manifest = load_manifest(root)
    if manifest:
        return set(manifest.get("reserved_ids") or [])
    base = (Path(root) / "eval") if root is not None else HERE
    try:
        return set(query_ids(base / HOLDOUT_FILENAME))
    except (OSError, yaml.YAMLError):
        return set()


def is_holdout_corpus(specs: list[dict], root: Path | None = None) -> bool:
    """Does a corpus about to be scored carry any reserved id?"""
    reserved = reserved_ids(root)
    return bool(reserved & {str(s.get("id")) for s in specs})


def refusal(specs: list[dict], label: str, env: dict, root: Path | None = None) -> str | None:
    """Why `run_eval` must not score this corpus under this label, or None.

    Holds the rule in one callable so `run_eval.main` and its test agree. Names counts
    and the label, never an id.
    """
    if not is_holdout_corpus(specs, root):
        return None
    if str(env.get(HOLDOUT_LEG_ENV) or "") != "1":
        return (f"this corpus carries reserved holdout queries; only the holdout leg of the "
                f"paired check may score it ({HOLDOUT_LEG_ENV}=1). #1412's reserve rule: "
                f"{RESERVE_RULE}")
    if any(str(label).startswith(p) for p in FORBIDDEN_LABEL_PREFIXES):
        return (f"label {label!r} is one the nightly readers glob; a holdout run must never "
                "write a record they open")
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("write", "verify"))
    args = ap.parse_args(argv)
    if args.action == "write":
        manifest = compute_manifest()
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"wrote {MANIFEST}: {manifest['n']} reserved ids, split_hash "
              f"{manifest['split_hash'][:16]}")
        return 0
    ok = load_manifest() is not None
    print(f"{MANIFEST}: {'verified' if ok else 'REFUSED (absent, or edited after its hash)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
