"""Cause-labelled corpus of grader-versus-reality disagreements (#4026).

The review rung (`scripts/automod/review.py`) grades each acceptance clause and
records `evidence_path` / `evidence_line` / `test_node_id` behind the verdict. When
that verdict disagrees with reality the dispute has always been written down — one
bug report at a time, each closed on its own. Nothing reported **which kind** of
dispute it was, so every grader change was aimed at the last incident rather than the
modal one, and the item that would have fixed a whole class kept being filed as the
next instance of it.

This module is the bounded instrument, not a new grader. It fixes a five-class cause
taxonomy, loads a labelled corpus, and publishes the distribution:

  grader_defect                 the verdict's own citation does not resolve in the
                                graded tree and never could: a line past EOF (#1254),
                                a test node whose file is absent there (#2083), a path
                                in no commit at all, or a graded range that does not
                                contain the surface the clause named (#2240).
  graded_wrong_input            the citation could only have come from another tree: an
                                absolute path into the vault, or a path standing in the
                                live checkout but absent from the graded commit — a rail
                                that read the running system instead of the round's
                                worktree (#1755, #2494).
  spec_ambiguity                the clause is unsatisfiable as written or admits two
                                readings (#1169, #1202, #1349). Fixed by rewriting the
                                clause; it is never evidence against the grader.
  real_unmet_work               the grader was right: the author changed the diff and
                                the same round's next attempt, at a different commit,
                                passed. Routes back to the author.
  grader_wrong_on_approved_work one round and one identical commit recorded BOTH a
                                refusal and a pass — #1903's class, the only one whose
                                remedy is a confirmation pass rather than a softer
                                grader.
  unknown                       fits none of the five. Its share is a reported number:
                                at or above 60% the taxonomy is wrong and gets fixed
                                before anything else does.

**The labelling rule.** The labeler here is the same engine whose grader is being
judged, so two of the six labels are legal only when a deterministic check fails.
`citation_failure()` must return a reason, and that reason must be the arm the class
claims: `load_corpus` refuses a `grader_defect` or `graded_wrong_input` row whose own
citations resolve against the commit it says it was graded at, and refuses either
label over a failure belonging to the other arm. An uncheckable label is not a label
— it is an opinion about the grader written into a file the next round cites. The
other four classes are not gated because none of them can license softening a gate.

**Row shape.** Every row carries exactly six keys, so the fields a re-check needs
travel inside `contrary_evidence`, in a tail this file both writes and parses
(`graded_commit=<sha>`, optional `graded_parent=<sha>`, and `cites=<path>:<line>` /
`<file>::<node>` / `range=<path>`). A reader holding one row can re-run the check on
the row as written; no side-channel key was added to the schema to make that possible.

**Scope of the two gated classes.** Only code-tree rounds are labelled mechanically. A
vault-surface round (`VC_…`) grades `~/obsidian`, whose commits the ledger does not
record, so its citations would "fail" against the code checkout for the wrong reason;
such rows are skipped and counted in `vault_surface_rows_skipped` rather than labelled
(every review row in today's ledger is an `SM_` code round, which is what keeps that
counter at zero in the open hand). Absolute vault paths *cited by* a code round are a
different thing, and the commonest finding: that is the `path_outside_graded_tree` arm
of `graded_wrong_input` (#1755, #2494).

**Survivor bias.** Only disputes somebody wrote down are in here. A wrong `met` that
was never challenged appears nowhere, so this is the distribution of *noticed*
disagreements — it over-weights refusals, is not a false-negative rate, and carries no
precision number. `summarize()` puts that sentence in the artifact, and `citation_rows`
counts the unnoticed `pass` rows it found rather than hiding them.

Nothing here changes grading behaviour and no model is called: every row is derived
from a record that already exists (a `promotions.jsonl` review row, a recorded refusal,
a backlog item naming the round it blocked) or seeded by hand from one.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


from app.data_root import vault_root
LIVE_ROOT = Path(__file__).resolve().parent.parent.parent

#: The five named classes, in reporting order, then `unknown`.
CAUSE_CLASSES = ("grader_defect", "graded_wrong_input", "spec_ambiguity",
                 "real_unmet_work", "grader_wrong_on_approved_work")
UNKNOWN = "unknown"
FIXED_CAUSES = CAUSE_CLASSES + (UNKNOWN,)

#: Every row carries exactly these keys — no more, no fewer.
ROW_KEYS = ("case_id", "source", "surface", "verdict_given",
            "contrary_evidence", "cause_class", "resolvable_now")

#: Where the corpus and its hand-seeded half live.
CORPUS_PATH = LIVE_ROOT / "eval" / "review_calibration" / "cause_corpus.jsonl"
SEED_PATH = LIVE_ROOT / "eval" / "review_calibration" / "cause_corpus_seed.jsonl"

#: The committed corpus is capped per derived arm so it stays reviewable; the full
#: population each arm selected from is returned in the census, not discarded.
MAX_CITATION_ROWS = 22
MAX_CLAUSE_ROWS = 10
MAX_UNMET_ROWS = 6
MAX_APPROVED_ROWS = 4

#: A row's `source` names a record that exists: a backlog item, or a ledger round.
SOURCE_RE = re.compile(r"^(?:backlog/(\d+)|round:([A-Za-z0-9_]+))$")

#: The machine tail inside `contrary_evidence`.
COMMIT_RE = re.compile(r"graded_commit=([0-9a-f]{7,40})")
PARENT_RE = re.compile(r"graded_parent=([0-9a-f]{7,40})")
CITES_RE = re.compile(r"cites=([^)]+)")
SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")

#: How a backlog item must phrase itself before its own title counts as evidence of a
#: clause defect: clause-specific wording only, so an incidental "unsatisfiable"
#: mention in some other item's body cannot recruit it into this class.
CLAUSE_DEFECT_RE = re.compile(
    r"\bclause[s]?\b[^.\n]{0,80}?(unsatisfiabl|not satisfiable|unamendabl|nonexistent|"
    r"names a (?:measurement|fixture|nonexistent)|two readings)", re.I)

#: A row's `surface`: which part of the loop produced the verdict in dispute — the code
#: review rung, the vault review grader, or the calibration harness.
SURFACES = ("review", "vault_review", "calibrate")


class CorpusError(ValueError):
    """A corpus row that must not be loaded: bad shape, bogus source, or a
    mechanically-gated cause whose citation check does not fire."""


# ── git and the working tree, read-only ──────────────────────────────────

def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=180,
                       check=False)
    r.stdout = (r.stdout or b"").decode("utf-8", "replace")
    r.stderr = (r.stderr or b"").decode("utf-8", "replace")
    return r


def commit_reachable(repo: Path, sha: str, _cache: dict = {}) -> bool:
    """Is `<sha>^{commit}` an object in this repo? A refused round whose branch was
    pruned has a head nobody can read any more, and a citation cannot be judged
    against a tree that is gone: that row is archaeology, not a disagreement."""
    if not sha or not SHA_RE.match(sha):
        return False
    if sha not in _cache:
        _cache[sha] = _git(repo, "cat-file", "-e", f"{sha}^{{commit}}").returncode == 0
    return _cache[sha]


def lines_at_commit(repo: Path, sha: str, path: str, _cache: dict = {}) -> int:
    """How many lines the file has at `sha`; -1 when that tree does not hold it."""
    key = (sha, path)
    if key not in _cache:
        if _git(repo, "cat-file", "-e", f"{sha}:{path}").returncode:
            _cache[key] = -1
        else:
            _cache[key] = len(_git(repo, "show", f"{sha}:{path}").stdout.splitlines())
    return _cache[key]


def path_in_range(repo: Path, parent: str, commit: str, path: str) -> bool:
    """Does `git diff --name-only parent..commit` list `path`? The graded surface is a
    range, so a clause whose named surface sits outside it was never gradeable."""
    out = _git(repo, "diff", "--name-only", f"{parent}..{commit}").stdout
    return path in {ln.strip() for ln in out.splitlines() if ln.strip()}


def in_live_tree(repo: Path, path: str) -> bool:
    """Does the path stand in the live checkout's working tree right now? That is the
    wrong-input witness: a citation resolving here but not in the graded tree was read
    off the running system, not off the round (#1755, #2494)."""
    try:
        return (repo / path).is_file()
    except OSError:
        return False


# ── the legality check ───────────────────────────────────────────────────

def outside_graded_tree(path: str) -> bool:
    """True for a path no commit of the graded repo can contain: an absolute path, a
    `~`-prefixed one, one escaping with `../`, or an embedded `/abs/path`."""
    p = str(path or "").strip()
    return bool(p) and (p.startswith("/") or p.startswith("~") or p.startswith("../")
                        or ":/" in p)


def citation_failure(repo: Path, graded_commit: str, citations: list[dict],
                     parent: str | None = None) -> str:
    """Why this row's own citations fail to resolve against the graded commit.

    A citation may carry `path` (with optional `line`), `node` (a pytest node id whose
    file part is checked), and `range` (a surface the clause named that must sit inside
    the graded `parent..commit`). Returns "" when every citation resolves — which is
    exactly the case in which a `grader_defect` or `graded_wrong_input` label is
    ILLEGAL. A row whose graded commit is not a reachable object reports
    `graded_commit_unreachable`: an unreadable tree evidences nothing, in either
    direction.
    """
    if not citations:
        return "no_citation"
    if not commit_reachable(repo, graded_commit):
        return "graded_commit_unreachable"
    for c in citations:
        p = str(c.get("path") or "").split(":")[0]
        node = str(c.get("node") or "").split("::")[0]
        if outside_graded_tree(p) or outside_graded_tree(node):
            return "path_outside_graded_tree"
        if p:
            n = lines_at_commit(repo, graded_commit, p)
            if n < 0:
                # Absent from the graded tree: another tree's path if it is standing in
                # the live checkout, otherwise a path that was never in any tree.
                return "live_tree_only" if in_live_tree(repo, p) else "missing_path"
            line = c.get("line")
            if isinstance(line, int) and line > 0 and line > n:
                return "line_past_eof"
        if node and lines_at_commit(repo, graded_commit, node) < 0:
            return "missing_node_file"
        rp = str(c.get("range") or "")
        if rp:
            if not parent:
                return "range_parent_missing"
            if not path_in_range(repo, parent, graded_commit, rp):
                return "graded_range_mismatch"
    return ""


GATED_CAUSES = ("grader_defect", "graded_wrong_input")

#: Which failing reason makes which gated class legal, per the item's own wording: a
#: graded range or tree that is not the one the clause named is `grader_defect`, while
#: evidence that exists only in some other tree is `graded_wrong_input`.
REASON_CLASS = {
    "path_outside_graded_tree": "graded_wrong_input",
    "live_tree_only": "graded_wrong_input",
    "missing_path": "grader_defect",
    "line_past_eof": "grader_defect",
    "missing_node_file": "grader_defect",
    "graded_range_mismatch": "grader_defect",
    # `range_parent_missing` is deliberately absent: a range claim with no named parent
    # cannot be checked at all, and an unverifiable claim is not evidence that a label is
    # legal ("If the labels for those two classes can't be checked, they aren't labels").
}


def classify_failure(reason: str) -> str | None:
    """The gated cause a failing citation supports, or None when it supports neither
    (`""`, `no_citation`, `graded_commit_unreachable`)."""
    return REASON_CLASS.get(str(reason or ""))


def evidence_tail(graded_commit: str, citations: list[dict] | None = None,
                  graded_parent: str | None = None) -> str:
    """Render the machine tail the builder writes and `evidence_fields` reads back."""
    parts = [f"graded_commit={graded_commit}"]
    if graded_parent:
        parts.append(f"graded_parent={graded_parent}")
    toks: list[str] = []
    for c in citations or []:
        if c.get("range"):
            toks.append(f"range={c['range']}")
        if c.get("node"):
            toks.append(str(c["node"]))
        if c.get("path"):
            toks.append(f"{c['path']}:{c['line']}" if c.get("line") else str(c["path"]))
    if toks:
        parts.append("cites=" + " | ".join(toks))
    return "(" + " ".join(parts) + ")"


def evidence_fields(text: str) -> dict:
    """Pull the machine tail out of a row's `contrary_evidence`.

    Reads `graded_commit=<sha>`, `graded_parent=<sha>` and `cites=` tokens: `range=<p>`,
    a token containing `::` is a pytest node id, `<p>:<n>` is a path with a line, and a
    bare token is a path. A row with no tail yields empty fields.
    """
    t = str(text or "")
    out: dict = {"graded_commit": "", "graded_parent": "", "citations": []}
    if (m := COMMIT_RE.search(t)):
        out["graded_commit"] = m.group(1)
    if (m := PARENT_RE.search(t)):
        out["graded_parent"] = m.group(1)
    if (m := CITES_RE.search(t)):
        for tok in re.split(r"[|,]", m.group(1)):
            tok = tok.strip().rstrip(".").strip()
            if not tok:
                continue
            if tok.startswith("range="):
                out["citations"].append({"range": tok[len("range="):].strip()})
            elif "::" in tok:
                out["citations"].append({"node": tok})
            else:
                head, _, tail = tok.rpartition(":")
                if tail.isdigit() and head:
                    out["citations"].append({"path": head, "line": int(tail)})
                else:
                    out["citations"].append({"path": tok})
    return out


def legality(row: dict, *, repo: Path = LIVE_ROOT) -> dict:
    """The deterministic verdict on one row's label: `{"cause","gated","reason",
    "legal","why"}`. For a gated cause `legal` means the check fired with the reason
    that class claims; for the other four it is trivially legal, because none of them
    can be used to soften a grader."""
    cause = str(row.get("cause_class") or "")
    if cause not in GATED_CAUSES:
        return {"cause": cause, "gated": False, "reason": "", "legal": True,
                "why": "not a mechanically-gated cause"}
    f = evidence_fields(row.get("contrary_evidence") or "")
    reason = citation_failure(repo, f["graded_commit"], f["citations"],
                              parent=f["graded_parent"] or None)
    wanted = classify_failure(reason)
    if wanted is None:
        detail = reason or (f"every cited path, line and node resolves at "
                           f"{f['graded_commit'] or 'no commit'}")
        return {"cause": cause, "gated": True, "reason": reason, "legal": False,
                "why": f"its own citations do not fail as a {cause} ({detail}); the label "
                       f"needs the verdict's own citation to fail against the commit it names"}
    if wanted != cause:
        return {"cause": cause, "gated": True, "reason": reason, "legal": False,
                "why": f"{reason} is the {wanted} arm, not {cause}"}
    return {"cause": cause, "gated": True, "reason": reason, "legal": True,
            "why": f"{reason}: the citation does not resolve against the graded tree"}


# ── sources and loading ──────────────────────────────────────────────────

def _ledger_events(ledger: Path):
    """Every event in the promotion ledger, rotated monthly archives included — most
    recorded refusals are in the archives, so a scan of the live file alone would
    report the population as far smaller than it is."""
    import gzip
    paths = [ledger] + sorted(ledger.parent.glob(f"{ledger.stem}-*.jsonl.gz"))
    for p in paths:
        if not p.exists():
            continue
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", errors="replace") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if isinstance(ev, dict):
                    yield ev


#: The records the corpus's rows were derived from: the promotion ledger (whose monthly
#: rotation holds most recorded refusals in the archive beside it) and the backlog.
#: `state.LEDGER_PATH` and `backlog.BACKLOG_DIR` are what the loop runs on, but the suite
#: re-points both — `tests/conftest.py` sends `LLOYD_AUTOMOD_STATE` to a scratch directory
#: so no test can write near the production ledger, and the `world` fixture points
#: `BACKLOG_DIR` at its own one-item tree. A corpus row can only be verified against the
#: record it cites, so the artifact-level tests name these two paths and READ them: the
#: conftest invariant guards writers, and nothing in this module writes to a ledger.
ARTIFACT_LEDGER = (Path.home() / ".local" / "state" / "lloyd-automod"
                   / "promotions.jsonl")
#: Via the resolver, never `Path.home() / "obsidian"`: `app.data_root.vault_root()` is the
#: one place that honours both `LLOYD_VAULT_ROOT` and `LLOYD_VAULT`, and a reader that
#: spelled the home path itself would read the live vault while its caller sandboxed with
#: the name it ignored (`tests/test_no_runtime_paths_in_code.py` exists for exactly that).
ARTIFACT_BACKLOG = vault_root() / "backlog"


def load_committed_corpus() -> list[dict]:
    """The corpus as committed, validated against the records its rows cite."""
    return load_corpus(CORPUS_PATH, ledger=ARTIFACT_LEDGER, backlog_dir=ARTIFACT_BACKLOG)


def load_ledger_round_ids(ledger: Path) -> set[str]:
    """Every `round_id` named anywhere in the ledger. A row whose source is not one of
    these was never a round, so it cannot be a disagreement the loop had."""
    return {str(ev["round_id"]) for ev in _ledger_events(ledger) if ev.get("round_id")}


def source_resolves(source: str, *, backlog_dir: Path, round_ids: set[str]) -> tuple[bool, str]:
    """Does a row's `source` name a record that exists? `backlog/<id>` needs a
    `<id>-*.md` item file; `round:<RID>` needs to appear in the promotion ledger."""
    m = SOURCE_RE.match(str(source or "").strip())
    if not m:
        return False, "must be backlog/<id> or round:<ROUND_ID>"
    if m.group(1):
        iid = m.group(1)
        hit = sorted(backlog_dir.glob(f"{iid}-*.md")) if backlog_dir.exists() else []
        return (bool(hit), f"backlog/{iid}" + ("" if hit else " has no item file"))
    rid = m.group(2)
    return (rid in round_ids, f"round:{rid}" + ("" if rid in round_ids else " is not in the ledger"))


def load_corpus(path: Path = CORPUS_PATH, *, repo: Path = LIVE_ROOT,
                backlog_dir: Path | None = None, ledger: Path | None = None,
                round_ids: set[str] | None = None, enforce_legality: bool = True) -> list[dict]:
    """Read and validate the corpus, refusing a row that cannot justify itself.

    Refuses (`CorpusError`): JSON that is not an object; a row whose keys are not
    exactly `ROW_KEYS`; a `cause_class` outside `FIXED_CAUSES`; an unknown `surface`; a
    `source` resolving to no existing record; a non-`unknown` row that does not name the
    commit it was graded at; a duplicate `case_id`; and — the rule that keeps this from
    being the grader grading itself — a gated cause whose own citations do not fail
    against the commit the row names.
    """
    from scripts.automod import backlog as B, state as S
    backlog_dir = backlog_dir or B.BACKLOG_DIR
    ledger = ledger or S.LEDGER_PATH
    if round_ids is None:
        round_ids = load_ledger_round_ids(ledger) if ledger.exists() else set()
    rows: list[dict] = []
    seen: set[str] = set()
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    for ln, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise CorpusError(f"{path.name}:{ln}: not JSON ({exc})") from exc
        if not isinstance(row, dict):
            raise CorpusError(f"{path.name}:{ln}: row is not an object")
        missing = [k for k in ROW_KEYS if k not in row]
        extra = [k for k in row if k not in ROW_KEYS]
        if missing or extra:
            raise CorpusError(f"{path.name}:{ln} ({row.get('case_id')}): keys must be exactly "
                             f"{list(ROW_KEYS)}; missing={missing} unexpected={extra}")
        cause = row["cause_class"]
        if cause not in FIXED_CAUSES:
            raise CorpusError(f"{path.name}:{ln} ({row['case_id']}): cause_class {cause!r} is not "
                             f"in {list(FIXED_CAUSES)}")
        if not str(row["case_id"]).strip() or not str(row["contrary_evidence"]).strip():
            raise CorpusError(f"{path.name}:{ln}: case_id and contrary_evidence must be non-empty")
        if not isinstance(row["resolvable_now"], bool):
            raise CorpusError(f"{path.name}:{ln} ({row['case_id']}): resolvable_now must be true/false")
        if row["surface"] not in SURFACES:
            raise CorpusError(f"{path.name}:{ln} ({row['case_id']}): surface {row['surface']!r} "
                             f"is not one of {list(SURFACES)}")
        ok, why = source_resolves(row["source"], backlog_dir=backlog_dir, round_ids=round_ids)
        if not ok:
            raise CorpusError(f"{path.name}:{ln} ({row['case_id']}): source {row['source']!r} "
                             f"does not resolve — {why}")
        if cause != UNKNOWN and not COMMIT_RE.search(str(row["contrary_evidence"])):
            raise CorpusError(f"{path.name}:{ln} ({row['case_id']}): a {cause} row must name the "
                             f"commit it was graded at, as graded_commit=<sha> inside "
                             f"contrary_evidence")
        if enforce_legality:
            leg = legality(row, repo=repo)
            if not leg["legal"]:
                raise CorpusError(f"{path.name}:{ln} ({row['case_id']}): {cause} refused — "
                                  f"{leg['why']}")
        if row["case_id"] in seen:
            raise CorpusError(f"{path.name}:{ln}: duplicate case_id {row['case_id']}")
        seen.add(row["case_id"])
        rows.append(row)
    if not rows:
        raise CorpusError(f"{path}: no rows")
    return rows


# ── the distribution ─────────────────────────────────────────────────────

SURVIVOR_NOTE = (
    "Every row is a disagreement somebody wrote down. A wrong `met` that was never "
    "challenged appears nowhere, so this is the distribution of NOTICED "
    "disagreements: it over-weights refusals, is not a false-negative rate, and no "
    "precision number may be read off it.")


def per_cause_counts(rows: list[dict]) -> dict[str, dict]:
    """Count per cause, with every class present even at zero so that an empty class is
    a reported finding rather than an absent key."""
    c = Counter(str(r.get("cause_class")) for r in rows)
    return {k: {"count": int(c.get(k, 0)), "empty": c.get(k, 0) == 0} for k in FIXED_CAUSES}


def summarize(rows: list[dict]) -> dict:
    """The artifact body: counts, the unknown share, the modal cause with its number,
    the empty classes, the sum check, and the survivor-bias note."""
    by = per_cause_counts(rows)
    total = len(rows)
    named = {k: by[k]["count"] for k in CAUSE_CLASSES}
    modal = max(named, key=lambda k: named[k]) if named else ""
    modal_count = named.get(modal, 0)
    unknown = by[UNKNOWN]["count"]
    s = sum(v["count"] for v in by.values())
    pct = (lambda n: round(100.0 * n / total, 1) if total else 0.0)
    return {
        "event": "review_cause_audit",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": total,
        "counts_sum": s,
        "counts_sum_matches_rows": s == total,
        "by_cause": by,
        "unknown": unknown,
        "unknown_share_pct": pct(unknown),
        "taxonomy_broken_if_unknown_ge_pct": 60.0,
        "modal_cause": modal,
        "modal_cause_count": modal_count,
        "modal_cause_sentence": (
            f"Of {total} recorded grader-versus-reality disagreements, the modal cause is "
            f"{modal} at {modal_count} rows ({pct(modal_count)}%), and the next "
            f"grader-behaviour change has to argue against that number."),
        "classes_empty": [k for k in FIXED_CAUSES if by[k]["empty"]],
        "survivor_bias_note": SURVIVOR_NOTE,
    }


def render_counts(rows: list[dict]) -> str:
    """One line per class in taxonomy order; an empty class is printed as empty."""
    by = per_cause_counts(rows)
    out = []
    for k in FIXED_CAUSES:
        n = by[k]["count"]
        if n:
            ex = next((str(r["case_id"]) for r in rows if r.get("cause_class") == k), "")
            out.append(f"cause {k}: {n} row(s), e.g. {ex}")
        else:
            out.append(f"cause {k}: EMPTY — no resolvable example in the corpus")
    out.append(f"unknown: {by[UNKNOWN]['count']} of {len(rows)} rows "
               f"({summarize(rows)['unknown_share_pct']}%)")
    return "\n".join(out)


def agreement_by_cause(cases: list[dict], results: list[dict]) -> dict[str, dict]:
    """Per-cause agreement over calibration cases. A case carries its cause in
    `cause_class`; a case without one is reported under `unlabelled`, never folded into
    `unknown` — an unlabelled fixture is a gap in the corpus, not a row that fit none of
    the five classes."""
    cause_of = {str(c.get("name")): str(c.get("cause_class") or "unlabelled") for c in cases}
    out: dict[str, dict] = {}
    for r in results:
        cause = cause_of.get(str(r.get("name")), "unlabelled")
        d = out.setdefault(cause, {"ok": 0, "n": 0})
        d["n"] += 1
        d["ok"] += 1 if r.get("ok") else 0
    return dict(sorted(out.items()))


def render_agreement(by_cause: dict[str, dict]) -> str:
    """The per-cause lines `calibrate` prints ABOVE its aggregate. Counts only, never a
    two-decimal rate: a handful of cases per class cannot carry a percentage, and no
    confidence interval the sample cannot support is printed (#696)."""
    return "\n".join(f"cause {c}: {v['ok']}/{v['n']} agree" for c, v in by_cause.items())


# ── building the corpus from records that already exist ──────────────────

def _clause_citations(cl: dict) -> list[dict]:
    """Where a recorded verdict pointed, as citations the legality check can read.

    One citation per location: an `evidence_path` and a `test_node_id` are separate
    claims about the tree, and collapsing them into one dict would let the row render
    only one of them — which is how a first build produced a row whose recorded failure
    lived in the dropped half and that the loader then correctly refused.
    """
    out: list[dict] = []
    p = str(cl.get("evidence_path") or "")
    if p:
        out.append({"path": p, "line": cl.get("evidence_line")})
    node = str(cl.get("test_node_id") or "")
    if node:
        out.append({"node": node})
    return out


def _review_events(ledger: Path):
    for ev in _ledger_events(ledger):
        if ev.get("event") == "review" and ev.get("ok") and ev.get("round_id") and ev.get("head"):
            yield ev


def citation_rows(*, ledger: Path, repo: Path = LIVE_ROOT, backlog_ids: set[int],
                  max_rows: int = MAX_CITATION_ROWS) -> tuple[list[dict], dict]:
    """Rows for the two mechanically-labelled classes: every recorded verdict whose OWN
    citation fails to resolve against the commit it was graded at.

    The cause is not chosen here — `classify_failure` reads it off the reason, so a row
    cannot be talked into either label. A refusal (`kind: retry` / `unsound`) is a
    disagreement the loop felt and sorts first; a `pass` whose citation does not resolve
    is one the loop never noticed, and is counted in the census as the survivor-bias
    population rather than padded into the corpus. `vault` reviews are skipped: their
    graded tree is the vault, whose commits the ledger does not record, and a vault
    citation checked against the code checkout would fail for the wrong reason.
    """
    bases = {str(e["round_id"]): str(e.get("base") or "")
             for e in _ledger_events(ledger)
             if e.get("event") == "round_start" and e.get("round_id")}
    found: list[dict] = []
    seen: set[tuple] = set()
    census = {"vault_surface_rows_skipped": 0, "unreachable_head_rows": 0,
              "candidate_rows": 0, "noticed": 0, "unnoticed_pass_rows": 0,
              "by_reason": {}, "by_cause": {}}
    for ev in _review_events(ledger):
        rid = str(ev["round_id"])
        head = str(ev["head"])
        cl_list = ev.get("clauses")
        if not isinstance(cl_list, list):
            continue
        if rid.startswith("VC"):
            # A vault-surface round grades ~/obsidian, whose commits the ledger does not
            # record; its citations would "fail" against the code checkout for the wrong
            # reason. Every review round in today's ledger is an SM_ code round, so this
            # counter is the witness that the skip is not silently eating rows.
            census["vault_surface_rows_skipped"] += len(cl_list)
            continue
        if not commit_reachable(repo, head):
            census["unreachable_head_rows"] += len(cl_list)
            continue
        for cl in cl_list:
            if not isinstance(cl, dict):
                continue
            cites = _clause_citations(cl)
            if not cites:
                continue
            reason = citation_failure(repo, head, cites, parent=bases.get(rid) or None)
            cause = classify_failure(reason)
            if not cause:
                continue
            census["candidate_rows"] += 1
            noticed = bool(ev.get("kind")) and ev.get("kind") != "pass"
            census["noticed" if noticed else "unnoticed_pass_rows"] += 1
            census["by_reason"][reason] = census["by_reason"].get(reason, 0) + 1
            census["by_cause"][cause] = census["by_cause"].get(cause, 0) + 1
            key = (rid, cl.get("clause"), reason)
            if key in seen:
                continue
            seen.add(key)
            iid = ev.get("item_id")
            cited = ", ".join(filter(None, [
                str(c.get("path") or "") + (f" line {c['line']}" if c.get("line") else "")
                for c in cites] + [str(c.get("node") or "") for c in cites]))
            found.append({
                "case_id": f"cite-{rid}-c{cl.get('clause')}-{reason}",
                "source": (f"backlog/{iid}" if isinstance(iid, int) and iid in backlog_ids
                           else f"round:{rid}"),
                "surface": "review",
                "verdict_given": f"the {rid} review rung returned `{ev.get('kind')}` on attempt "
                                 f"{ev.get('attempt')}, clause {cl.get('clause')} "
                                 f"`{cl.get('verdict')}`",
                "contrary_evidence": (
                    f"the promotions.jsonl review row for {rid} attempt {ev.get('attempt')} cites "
                    f"{cited} of the commit it graded, and that citation fails as {reason}"
                    + " " + evidence_tail(head, cites, bases.get(rid) or None)),
                "cause_class": cause,
                "resolvable_now": True,
                "_noticed": noticed,
            })
    found.sort(key=lambda r: (not r["_noticed"], r["case_id"]))
    return found[:max_rows], census


def graded_commit_of(*, round_id: str, review_events: list[dict],
                     bases: dict[str, str]) -> str:
    """The commit a round's clauses were graded against.

    A round that reached review names the head that was graded. A round that never got
    there — it died on an earlier rung — never produced a head, and its `base` *is* the
    tree its clauses were graded against, so naming it is naming the graded commit
    rather than inventing one.
    """
    for ev in review_events:
        if ev.get("head"):
            return str(ev["head"])
    return str(bases.get(round_id) or "")


def clause_defect_rows(*, backlog_dir: Path, ledger: Path,
                       max_rows: int = MAX_CLAUSE_ROWS) -> list[dict]:
    """Rows for `spec_ambiguity`, taken from the items that filed it: a backlog item
    whose own title says a clause was unsatisfiable, unamendable, or named something
    that does not exist, and whose body names a round the ledger knows about.

    The class is the item's own claim about itself, quoted; the graded commit comes from
    the ledger row for the round it names. Nothing here decides that the clause was
    badly written — that ruling is in the item, made by whoever hit it. The routing is
    the point: this class is repaired by rewriting the clause (`spec_ambiguity` goes to
    the clause author, never to the grader), so it must not be counted as evidence
    against the grader."""
    heads: dict[str, list[dict]] = {}
    for ev in _review_events(ledger):
        heads.setdefault(str(ev["round_id"]), []).append(ev)
    bases = {str(e["round_id"]): str(e.get("base") or "")
             for e in _ledger_events(ledger)
             if e.get("event") == "round_start" and e.get("round_id")}
    out: list[dict] = []
    for f in sorted(backlog_dir.glob("*.md")) if backlog_dir.exists() else []:
        m = re.match(r"^(\d+)-(.+)\.md$", f.name)
        if not m:
            continue
        body = f.read_text(encoding="utf-8", errors="replace")
        title = body.split("# ", 1)[-1].splitlines()
        title = title[0].strip() if title else f.stem
        if not CLAUSE_DEFECT_RE.search(title):
            continue
        named = [r for r in re.findall(r"SM_[0-9A-Za-z_]+", body) if r in heads or r in bases]
        rid = next((r for r in named if r in heads), "") or next(iter(named), "")
        if not rid:
            continue
        head = graded_commit_of(round_id=rid, review_events=heads.get(rid, []), bases=bases)
        if not head:
            continue
        reached_review = rid in heads
        status = re.search(r"^status:\s*(\S+)", body, re.M)
        out.append({
            "case_id": f"clause-{m.group(1)}",
            "source": f"backlog/{m.group(1)}",
            "surface": "review",
            "verdict_given": (f"the {rid} review rung returned `{heads[rid][0].get('kind')}` "
                              f"on attempt {heads[rid][0].get('attempt')}" if reached_review
                              else f"{rid} never reached the review rung, so no clause "
                                   f"verdict was ever returned for this item"),
            "contrary_evidence": (
                f"backlog #{m.group(1)} (status "
                f"{status.group(1) if status else 'unstated'}) filed the clause, not the grader, "
                f"as the defect: \u201c{title}\u201d "
                + ("the clause was graded at " if reached_review
                   else "the round died before review, so its base is the tree it was graded at ")
                + evidence_tail(head, [], None)),
            "cause_class": "spec_ambiguity",
            "resolvable_now": True,
        })
    return sorted(out, key=lambda r: r["case_id"])[:max_rows]


def retry_then_pass_rows(*, ledger: Path, backlog_ids: set[int],
                         max_rows: int = MAX_UNMET_ROWS) -> list[dict]:
    """Rows for `real_unmet_work`: a round the review rung refused at head X whose next
    recorded attempt passed at a DIFFERENT head Y. The author changed the diff and the
    same clauses then graded met, so the grader was right and the class routes back to
    the author — it argues for nothing at all in the grader.
    """
    by_round: dict[str, list[dict]] = {}
    for ev in _review_events(ledger):
        if ev.get("kind"):
            by_round.setdefault(str(ev["round_id"]), []).append(ev)
    out: list[dict] = []
    for rid, evs in sorted(by_round.items()):
        refused = [e for e in evs if e.get("kind") == "retry"]
        if not refused:
            continue
        passed = [e for e in evs if e.get("kind") == "pass"
                  and str(e.get("head")) != str(refused[-1].get("head"))]
        if not passed:
            continue
        r0, p0 = refused[0], passed[-1]
        iid = r0.get("item_id")
        head = str(r0.get("head"))
        if not (isinstance(iid, int) and iid in backlog_ids):
            continue
        flagged = sorted(c.get("clause") for c in (r0.get("clauses") or [])
                         if isinstance(c, dict) and c.get("verdict") != "met")
        out.append({
            "case_id": f"unmet-{rid}",
            "source": f"backlog/{iid}",
            "surface": "review",
            "verdict_given": f"the {rid} review rung returned `retry` on attempt "
                             f"{r0.get('attempt')} with clauses {flagged or 'unnamed'} not met",
            "contrary_evidence": (
                f"the author then changed the diff: attempt {p0.get('attempt')} at head "
                f"{p0.get('head')} — a different commit — recorded `{p0.get('kind')}`, so the "
                f"refusal at {head} described work that was genuinely not finished "
                + evidence_tail(head, [], None)),
            "cause_class": "real_unmet_work",
            "resolvable_now": True,
        })
    return sorted(out, key=lambda r: r["case_id"])[:max_rows]


def refused_then_passed_rows(*, ledger: Path, backlog_ids: set[int],
                             max_rows: int = MAX_APPROVED_ROWS) -> list[dict]:
    """Rows for `grader_wrong_on_approved_work`: one round and one identical commit
    recorded BOTH a refusal and a pass. The same tree, graded twice, answered two ways,
    and the pass is another reading (a fresh session, its own context) of exactly the
    work the refusal blocked. That is #1903's class — the only class whose remedy is a
    confirmation pass, which is why this row names #1903 instead of building a second
    reader here."""
    kinds: dict[tuple, list[dict]] = {}
    for ev in _review_events(ledger):
        if ev.get("kind"):
            kinds.setdefault((str(ev["round_id"]), str(ev["head"])), []).append(ev)
    out: list[dict] = []
    for (rid, head), evs in sorted(kinds.items()):
        ks = {str(e.get("kind")) for e in evs}
        if "pass" not in ks or not (ks - {"pass"}):
            continue
        r0 = next(e for e in evs if e.get("kind") != "pass")
        p0 = next(e for e in evs if e.get("kind") == "pass")
        iid = r0.get("item_id")
        if not (isinstance(iid, int) and iid in backlog_ids):
            continue
        out.append({
            "case_id": f"approved-{rid}",
            "source": f"backlog/{iid}",
            "surface": "review",
            "verdict_given": f"the {rid} review rung returned `{r0.get('kind')}` on attempt "
                             f"{r0.get('attempt')} for head {head}",
            "contrary_evidence": (
                f"a second grading of that identical commit inside the same round "
                f"(attempt {p0.get('attempt')}) recorded `pass`, so the clauses the refusal held "
                f"unmet were graded met by another reading of the same tree — the class #1903 "
                f"names, whose remedy is a confirmation pass and not a softer grader "
                + evidence_tail(head, [], None)),
            "cause_class": "grader_wrong_on_approved_work",
            "resolvable_now": True,
        })
    return sorted(out, key=lambda r: r["case_id"])[:max_rows]


def seed_rows(path: Path = SEED_PATH) -> list[dict]:
    """The hand-seeded rows, each taken from a record that already exists: a calibration
    fixture beside what the recorded `calibrate` runs answered, and disputes where the
    grader never produced a clause verdict at all. Kept in their own committed file so
    regenerating the derived half never overwrites them."""
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8",
                                                    errors="replace").splitlines() if ln.strip()]


def build_cause_corpus(*, out: Path = CORPUS_PATH, ledger: Path | None = None,
                 repo: Path = LIVE_ROOT, backlog_dir: Path | None = None) -> dict:
    """Write the corpus: the four derived arms plus the seeded rows, each reduced to the
    six row keys (the machine fields live inside `contrary_evidence`, where a reader
    re-runs the legality check from). Returns the per-arm counts and the census of the
    population each cap selected from."""
    from scripts.automod import backlog as B, state as S
    backlog_dir = backlog_dir or B.BACKLOG_DIR
    ledger = ledger or S.LEDGER_PATH
    backlog_ids = {int(m.group(1))
                   for p in (backlog_dir.glob("*.md") if backlog_dir.exists() else [])
                   if (m := re.match(r"^(\d+)-", p.name))}
    cite_rows, census = citation_rows(ledger=ledger, repo=repo, backlog_ids=backlog_ids)
    parts = {
        "citation": cite_rows,
        "clause": clause_defect_rows(backlog_dir=backlog_dir, ledger=ledger),
        "unmet": retry_then_pass_rows(ledger=ledger, backlog_ids=backlog_ids),
        "approved": refused_then_passed_rows(ledger=ledger, backlog_ids=backlog_ids),
        "seed": seed_rows(),
    }
    rows: list[dict] = []
    for v in parts.values():
        rows.extend(v)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: r[k] for k in ROW_KEYS}, sort_keys=True) + "\n")
    return {"rows": len(rows), "arms": {k: len(v) for k, v in parts.items()},
            "census": census, "path": str(out)}


def write_audit(*, out_dir: Path | None = None, path: Path = CORPUS_PATH,
                date: str | None = None, corpus: list[dict] | None = None) -> Path:
    """Publish the dated distribution artifact under the runtime data root — never in the
    code tree, which is code only."""
    root = Path(out_dir) if out_dir else (Path.home() / "lloyd-data" / "eval"
                                          / "review-calibration")
    root.mkdir(parents=True, exist_ok=True)
    rows = corpus if corpus is not None else load_corpus(path)
    day = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    p = root / f"cause-audit-{day}.json"
    body = summarize(rows)
    body["cases"] = [{"case_id": r["case_id"], "cause_class": r["cause_class"],
                      "source": r["source"], "surface": r["surface"]} for r in rows]
    p.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p
