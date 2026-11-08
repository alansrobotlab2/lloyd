#!/usr/bin/env python3
"""Report file cites in the always-loaded memory files that no longer resolve (#882).

`lloyd/MEMORY.md`, `lloyd/USER.md` and `memory/mental-models.md` are rendered
into every system prompt, and they lean on backticked paths as proof: "see
`knowledge/software/mc-frontend-https-probe.md`". When the file behind a cite
is gone the line is an instruction to go read nothing, and nothing noticed —
`skill_lint.check_script_paths` covers SKILL.md only, the groundskeeper covers
wikilinks. This resolves every backticked path and `path:N` line cite in those
files against the live tree and reports the ones that do not resolve.

It REPORTS; it never edits a cited file. An agent rewriting durable memory from
a regex is the wrong direction — a dead cite is for a person or a vault round
to fix. The only file it may write is the report named by `--report`.

Kinds and verdicts, one record per cite:
  kind     path | line          (`line` = `file.py:N`, checked against EOF)
  roots    `~/` -> home; `/…` as is; otherwise repo root, vault root, data
           root (`~/lloyd-data`), the citing file's directory, and for a bare
           `name.py:N` the one tracked file of that name
  verdict  resolves | dangling | swept | exempt
  how_checked names the check behind every verdict — a `resolves` without one
  would be a pass nobody can re-run.

Precision is the whole risk (the item's step-4 ablation found 13 of 18 naive
hits were noise), so only backticked spans are read, fenced code is skipped,
and slash commands, `/v1/...` endpoints, templates (`YYYY-MM-DD.md`, `<id>`,
globs), URLs, commands with spaces and truncated ids are never extracted.

A cite is `exempt` when its line says the target is gone on purpose ("no longer
exists", "retired", and the two spellings #3944 found missing: an attributive
`deleted` before the path, and the shell's own `No such file or directory`), when
it carries `<!-- ri:known-absent -->`, or when its target is in `KNOWN_ABSENT` —
the same shape as skill_lint's
`KNOWN_ABSENT_SCRIPTS`: absent path -> why that is fine. Exempt cites do not
count toward the exit code.

A cite is `swept` when it names an autonomy run record (`autonomy-runs/<id>/
run_*.md`) that the groundskeeper's retention sweep has already deleted: the file
is missing, its own name-timestamp is past `RUN_RECORD_MAX_AGE_DAYS`, and the
directory holding its newer records is still there. That is a third thing, not a
fourth spelling of the two above — an exemption is a target that is *meant* to be
gone, and `dangling` is a target a reader could still re-point. A swept record has
no bytes to re-point at: `memory/mental-models.md` cited
`autonomy-runs/74/run_74_20260929_230732.md` as a rule's evidence and the cite went
dead on its own one night, arriving in knowledge health report #60 on 2026-10-31 as
`NEWLY DANGLING: 1` beside cites a person can repair (#3637). Swept cites are
reported and counted, but they leave the exit code and the newly-dangling delta,
because both of those exist to say "a human can still fix this".

Not covered yet: symbol, config-key and heading cites (a `path:N` beyond EOF is
the only drift checked), and no nightly task runs it — arming one is a person's
call (the item's human clause 2).

    python -m scripts.maintenance.referential_integrity [--json]
        [--report ~/obsidian/autonomy/referential-integrity-latest.md]
        [--expect-dangling TARGET ...]

Exit 0 clean, 1 at least one dangling cite, 2 an `--expect-dangling` target was
not reported dangling (a self-check that has stopped catching its own fixture).
`exempt` and `swept` cites never move the exit code — and narrowing it to
`dangling` must not soften its purpose: a cite that is not a swept run record and
does not resolve is still a red run, so classing swept records out can never mask a
moved or renamed module (#3637 clause 4).
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
# Run by path (`python …/scripts/maintenance/referential_integrity.py`), which puts this file's own directory on
# `sys.path[0]` and not the checkout: without the insertion below, `app.data_root`
# raises `ModuleNotFoundError` and the reader silently never runs. Same reason as
# `scripts/skill_lint.py`'s insertion. (#3403)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from app.data_root import vault_root  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# Vault-relative; these are the files the prompt builder loads whole.
CORPUS = ("lloyd/USER.md", "lloyd/MEMORY.md", "memory/mental-models.md")

KNOWN_ABSENT: dict[str, str] = {}

EXEMPT_MARKER = "<!-- ri:known-absent -->"
# The vocabulary held the postpositive form ("`x` was deleted") but not the two
# spellings a person reaching for a sentence about a gone path actually write, so
# `memory/mental-models.md:2407` ("…run a watchdog under a deleted
# `lloyd-work/SM_20261104_123801/` round root…") and :2199 ("`ls
# ~/.local/state/lloyd-fleet-watchdog/` → `No such file or directory`") printed as
# dangling records a human could not repair (#3944). `\bdeleted\b` subsumes
# `was deleted|were deleted`, which stay readable in the alternation. Deliberately
# no bare `\bremoved\b`: `was removed|were removed` are already here, and the two
# cites this widening newly exempts are both caught above, so a wider `removed`
# would weaken the one guard that keeps a moved cite red for no cite needing it.
_ABSENCE_PROSE = re.compile(
    r"no longer exist|does not exist|doesn't exist|did not exist|\bdeleted\b|"
    r"was removed|were removed|no such file or directory|\bretired\b|\bis gone\b|"
    r"\bare gone\b|\bnever landed\b|\bnever existed\b|\babsent\b|\bsuperseded\b|"
    r"\brenamed\b|\bmissing\b|\bneither is\b",
    re.IGNORECASE,
)

#: Public spelling of the guard. `scripts/automod/prose_consumers.py` (#3538) asks
#: the inverse question — a retired name, the durable files still naming it — and
#: needs the same ruling on when a line is a MENTION rather than a consumer. It
#: imports this rather than carrying its own copy, which is #3309's whole point: two
#: copies of a rule drift until the nightly sweep and the gate disagree about the
#: same sentence in the same note, and both read as green.
ABSENCE_PROSE = _ABSENCE_PROSE

_BACKTICK = re.compile(r"`([^`\n]+)`")
_LINE_SUFFIX = re.compile(r":(\d+)(?:[-–]\d+)?$")
_EXTENSIONS = (".md", ".py", ".json", ".jsonl", ".yaml", ".yml", ".sh", ".ts",
               ".tsx", ".toml", ".txt", ".sqlite", ".db", ".conf", ".service",
               ".timer", ".patch", ".log", ".js", ".mjs", ".csv")
_CODE_EXTENSIONS = (".py", ".ts", ".tsx", ".js", ".mjs", ".sh", ".yaml", ".yml")
# An absolute path is a filesystem cite only under a real root; `/goal`,
# `/state` and `/v1/messages` share the leading slash and are not paths.
_FS_ROOTS = ("/home/", "/tmp/", "/etc/", "/usr/", "/var/", "/opt/", "/run/", "/dev/")
_TEMPLATE = re.compile(r"YYYY|MM-DD|\bDD\b|<|>|\{|\}|\*|\?|\$|\.\.\.|…|\bN\b|\bXXX|\bNNN")

#: ── the retention-swept arm (#3637) ─────────────────────────────────────────────
#: Where the horizon lives. `scripts/groundskeeper/retention-sweep.py` is the thing
#: that deletes run records, so it is the only place the number may be read from; the
#: file has a hyphen and cannot be imported by name, so it loads by path the way
#: `scripts/maintenance/guard_vacuity.py` loads its own mutation targets.
SWEEP_SOURCE = REPO_ROOT / "scripts" / "groundskeeper" / "retention-sweep.py"

#: A cite of an autonomy run record. `run_` is required, not any `.md`: the sweeper's
#: own glob is `run_*.md`, and a cite whose name it would never have touched (a jsonl
#: sidecar, a hand-written summary) has no sweep to blame.
_RUN_RECORD_CITE = re.compile(r"(?:^|/)autonomy-runs/(\d{1,6})/(run_[^/]*\.md)$")

#: The UTC instant the engine wrote into the name: `run_<task>_<YYYYMMDD>_<HHMMSS>.md`.
#: Same clock the sweeper ages on — it reads a frontmatter stamp first and this file
#: name's own digits are that stamp, so an age read off the name is the age the
#: deletion was made at.
_RUN_STAMP = re.compile(r"run_\d{1,6}_(\d{8}_\d{6})\.md$")


@functools.lru_cache(maxsize=4)
def _run_record_max_age_days(source: Path) -> int | None:
    """`RUN_RECORD_MAX_AGE_DAYS` off the sweeper itself; `None` when it cannot be read.

    Imported rather than restated: a copy of `30` here is a claim that goes false the
    night the sweeper's window changes, which is the same failure #3637 is about — a
    rule nobody checks against the rule it describes. `None` keeps every cite
    `dangling`, so a load failure costs precision it had already, never a verdict the
    resolver cannot support.
    """
    try:
        spec = importlib.util.spec_from_file_location("_ri_sweep_source", source)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        days = int(getattr(mod, "RUN_RECORD_MAX_AGE_DAYS"))
    except (OSError, ValueError, TypeError, AttributeError, ImportError):
        return None
    return days if days > 0 else None


def _source_name(source: Path) -> str:
    """How `how_checked` names the sweeper: repo-relative when it is in the checkout,
    the path as given otherwise — a verdict must not name a file it did not read.
    """
    try:
        return str(source.relative_to(REPO_ROOT))
    except ValueError:
        return str(source)


def _record_age_days(name: str, now: datetime) -> float | None:
    """How old the run record's own name says it is, in days; `None` when the name
    carries no instant it could have been aged on — including digits that are not a
    calendar date, which the sweeper could not age either.
    """
    m = _RUN_STAMP.search(name)
    if m is None:
        return None
    try:
        stamp = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return (now - stamp).total_seconds() / 86400.0


def _swept_run_record(target: str, missing: list[Path], *, now: datetime | None = None,
                      source: Path | None = None) -> str | None:
    """Why this missing cite is a record the retention sweep deleted; `None` if not.

    Three conditions, each one the sweeper's behaviour and not a guess: the cite names
    an `autonomy-runs/<id>/run_*.md` record, that record's name-timestamp is past the
    horizon, and a candidate parent directory still exists — the sweep ages files out
    of a task's directory and leaves the directory, so a directory that is itself gone
    is some other loss (a wiped data root) and stays `dangling`. All three are needed:
    inside the horizon the file went missing for a reason a reader can act on, and
    without the parent it was not this sweep that did it.
    """
    m = _RUN_RECORD_CITE.search(target)
    if m is None:
        return None
    days = _record_age_days(m.group(2), now or datetime.now(timezone.utc))
    if days is None:
        return None
    horizon = _run_record_max_age_days(source or SWEEP_SOURCE)
    if horizon is None or days < horizon:
        return None
    parent = next((p.parent for p in missing if p.parent.is_dir()), None)
    if parent is None:
        return None
    return (f"retention sweep: {parent} still holds its newer records and the cited "
            f"record's own name-timestamp is {int(days)} days old, past "
            f"RUN_RECORD_MAX_AGE_DAYS = {horizon} read from "
            f"{_source_name(source or SWEEP_SOURCE)} — deleted, not moved, so there is "
            f"nothing here to re-point")


def extract_target(span: str) -> tuple[str, int | None] | None:
    """The (path, line) a backticked span cites, or None when it is not a path."""
    s = span.strip().rstrip(".,;")
    if not s or " " in s or "://" in s or _TEMPLATE.search(s):
        return None
    s = s.split("::", 1)[0]                    # pytest node id -> its file
    line = None
    m = _LINE_SUFFIX.search(s)
    if m:
        line = int(m.group(1))
        s = s[: m.start()]
    if s.endswith("-"):                        # truncated uuid / hash prefix
        return None
    if s.startswith("/"):
        if not s.startswith(_FS_ROOTS):
            return None
    elif not s.startswith("~/") and "/" not in s:
        # A bare `notes.md` could be anywhere and a search would guess; a bare
        # `notify.py:246` is a code cite, looked up by basename in the repo.
        if line is None or not s.endswith(_CODE_EXTENSIONS):
            return None
    if s.startswith(("./", "../", ":")):
        return None
    if not s.startswith(("/", "~/")) and "/" not in s.rstrip("/") and s.endswith("/"):
        return None                            # `subliminal/`: a prefix, not a place
    # Something path-shaped must be there: an extension or a trailing slash.
    tail = s.rstrip("/").rsplit("/", 1)[-1]
    if not (s.endswith("/") or tail.endswith(_EXTENSIONS)):
        return None
    return s, line


@functools.lru_cache(maxsize=8)
def _tracked_files(repo: Path) -> tuple[str, ...]:
    try:
        return tuple(subprocess.run(
            ["git", "-C", str(repo), "ls-files"], capture_output=True, text=True,
            timeout=30, check=True).stdout.splitlines())
    except (OSError, subprocess.SubprocessError):
        return tuple(str(p.relative_to(repo)) for p in repo.rglob("*")
                     if p.is_file() and ".git" not in p.parts)


def _repo_files_named(repo: Path, name: str) -> list[Path]:
    """Tracked files in ``repo`` with this basename (a walk when not a git tree)."""
    return [repo / f for f in _tracked_files(repo) if f.rsplit("/", 1)[-1] == name]


def _candidates(target: str, citing: Path, *, vault: Path, repo: Path, home: Path,
                data: Path) -> list[tuple[str, Path]]:
    if target.startswith("~/"):
        return [("home", home / target[2:])]
    if target.startswith("/"):
        return [("absolute", Path(target))]
    out = [("repo_root", repo / target), ("vault_root", vault / target),
           ("data_root", data / target), ("citing_dir", citing.parent / target)]
    if "/" not in target:
        # A bare `notify.py:246` means the one file of that name in the tree;
        # two of them is a cite nobody can follow, so it stays unresolved.
        named = _repo_files_named(repo, target)
        if len(named) == 1:
            out.append(("repo_basename", named[0]))
    return out


def resolve(target: str, line: int | None, citing: Path, *, vault: Path,
            repo: Path, home: Path, data: Path | None = None) -> tuple[str, str]:
    data = data or home / "lloyd-data"
    candidates = _candidates(target, citing, vault=vault, repo=repo,
                             home=home, data=data)
    tried = []
    for root, path in candidates:
        tried.append(root)
        if not path.exists():
            continue
        if line is None:
            return "resolves", f"stat:{root}"
        if not path.is_file():
            return "dangling", f"stat:{root} is not a file, cite names line {line}"
        try:
            n = sum(1 for _ in path.open("rb"))
        except OSError as exc:
            return "dangling", f"stat:{root} unreadable ({exc.__class__.__name__})"
        if line <= n:
            return "resolves", f"stat:{root}+line<=eof({n})"
        return "dangling", f"stat:{root}, line {line} beyond eof ({n})"
    how = "stat:" + "|".join(tried) + " (none exist)"
    if "/" not in target and len(named := _repo_files_named(repo, target)) > 1:
        how += f"; ambiguous: {len(named)} tracked files named {target}"
    # Runtime data left the tree on 2026-09-22 (architecture/data-home.md):
    # `~/lloyd/X` became `~/lloyd-data/X`. Still dangling — the cite is
    # wrong — but the report says where the file went.
    if target.startswith("~/lloyd/"):
        moved = data / target[len("~/lloyd/"):].rstrip("/")
        if moved.exists():
            how += f"; moved to {moved}"
    # Last arm, so every check above still owns its own case: this one only fires for
    # a cite the sweeper's own rule explains (#3637).
    swept = _swept_run_record(target, [p for _, p in candidates])
    if swept:
        return "swept", swept
    return "dangling", how


def scan_file(rel: str, *, vault: Path, repo: Path, home: Path,
              data: Path | None = None) -> list[dict]:
    citing = vault / rel
    try:
        text = citing.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [{"file": rel, "line": 0, "kind": "path", "target": rel,
                 "verdict": "dangling", "how_checked": "corpus file unreadable"}]
    out: list[dict] = []
    in_fence = False
    for lineno, raw in enumerate(text.splitlines(), 1):
        if raw.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for span in _BACKTICK.findall(raw):
            got = extract_target(span)
            if got is None:
                continue
            target, cited_line = got
            rec = {"file": rel, "line": lineno,
                   "kind": "line" if cited_line is not None else "path",
                   "target": target if cited_line is None else f"{target}:{cited_line}"}
            verdict, how = resolve(target, cited_line, citing,
                                   vault=vault, repo=repo, home=home, data=data)
            if verdict == "dangling":
                if target in KNOWN_ABSENT:
                    verdict, how = "exempt", f"KNOWN_ABSENT: {KNOWN_ABSENT[target]}"
                elif EXEMPT_MARKER in raw:
                    verdict, how = "exempt", f"marker {EXEMPT_MARKER}"
                elif (m := _ABSENCE_PROSE.search(raw)):
                    verdict, how = "exempt", f"absence prose: {m.group(0)!r}"
            rec.update(verdict=verdict, how_checked=how)
            out.append(rec)
    return out


def scan(*, vault: Path, repo: Path = REPO_ROOT, home: Path | None = None,
         data: Path | None = None, corpus: tuple[str, ...] = CORPUS) -> list[dict]:
    home = home or Path.home()
    data = data or home / "lloyd-data"
    records: list[dict] = []
    for rel in corpus:
        records.extend(scan_file(rel, vault=vault, repo=repo, home=home, data=data))
    return records


def _key(rec: dict) -> str:
    return f"{rec['file']} -> {rec['target']}"


_STATE_RE = re.compile(r"<!-- ri:dangling (\[.*?\]) -->", re.S)


def previous_dangling(report: Path) -> set[str] | None:
    try:
        m = _STATE_RE.search(report.read_text(encoding="utf-8"))
    except OSError:
        return None
    if not m:
        return None
    try:
        return set(json.loads(m.group(1)))
    except ValueError:
        return None


#: Where a report of this sweep lives, and what its frontmatter says. The file is a vault
#: file like any other, so `scripts/vault/validate_okf.py` walks it: on 2026-10-07 that
#: gate was RED on exactly four files — `autonomy/referential-integrity-latest.md` and the
#: three dated copies autonomy task #94 `cp`s from it — solely because this generator
#: emitted no `---` block at all (#2326). `scripts/skill_lint.py` already carries its own
#: report for the same reason; this is that precedent, one edit, reaching `latest` and
#: every future dated copy together because the copies are a `cp` of these bytes.
#:
#: The keys are the ones the gate reads plus the two a reader needs: `segment` is the only
#: value that can say where this sits, and `generated_at` is the run's UTC stamp, quoted so
#: it stays a string rather than PyYAML's `datetime` resolver turning it into an object a
#: later timestamp check has to special-case. `type: note` is a value the gate already
#: accepts — `okf_taxonomy.KNOWN_TYPES` has `note` and no `report`, so `type: report` would
#: be the unknown-type warning that `--strict` turns into exit 2 — and it does not make this
#: a task either way: the Mission Control autonomy tab keys on the `NN-` filename through
#: `autonomy_task_files()` (#1594), which `tests/test_mc_summarize_autonomy_gate.py` pins
#: against these very bytes.
REPORT_SEGMENT = "autonomy"
REPORT_TAGS = ["referential-integrity", "loaded-memory", "dangling-cites"]


def render_frontmatter(stamp: str, total: int, dangling: int) -> str:
    """The `---` block `render_report` puts on top of every report it writes.

    The gate `yaml.safe_load`s this block whole (`validate_okf.py:237-242`: `STRICT_FM_RE`
    match at `:155`, then a parse whose failure is itself a violation — the numbers are
    #2402's re-measure, and they were already ~45 lines adrift before it: this cite named
    184-191, which is `main()`'s argparse, because nothing resolves a `path:N` cite inside
    a source docstring), so the shapes here are chosen
    against what a YAML resolver does to them rather than against a text count. `segment`
    and `type` are plain scalars. `generated_at` is quoted because PyYAML resolves a bare
    `2026-10-07T03:07:45Z` to a `datetime`, which makes `json.dumps(frontmatter)` raise
    `TypeError` for any reader that serialises the block; the vault's own generated notes
    quote it the same way. `tags` is flow style so it parses to a list on one line, and
    `summary` is quoted because it contains a colon.
    """
    return "\n".join([
        "---",
        "type: note",
        f"segment: {REPORT_SEGMENT}",
        f"tags: [{', '.join(REPORT_TAGS)}]",
        f'generated_at: "{stamp}"',
        f'summary: "Referential-integrity sweep over loaded memory: {total} cites, '
        f'{dangling} dangling. Generated, do not hand-edit."',
        "---",
        "",
    ])


def render_report(records: list[dict], previous: set[str] | None) -> str:
    dangling = [r for r in records if r["verdict"] == "dangling"]
    exempt = [r for r in records if r["verdict"] == "exempt"]
    swept = [r for r in records if r["verdict"] == "swept"]
    keys = sorted({_key(r) for r in dangling})
    new = keys if previous is None else [k for k in keys if k not in previous]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        render_frontmatter(stamp, len(records), len(dangling)).rstrip("\n"),
        "",
        "# Referential integrity — loaded memory",
        "",
        f"Run {stamp} by `scripts/maintenance/referential_integrity.py` (#882). "
        "Report only: nothing cited here was edited.",
        "",
        f"- total cites: {len(records)}",
        f"- dangling: {len(dangling)}",
        f"- exempt: {len(exempt)}",
        f"- swept run records: {len(swept)}",
        f"- newly dangling since the previous report: "
        f"{len(new) if previous is not None else 'n/a (no previous report)'}",
        "",
        "## Dangling",
        "",
    ]
    lines += [f"- `{r['file']}:{r['line']}` → `{r['target']}` — {r['how_checked']}"
              + (" **(new)**" if previous is not None and _key(r) in new else "")
              for r in dangling] or ["- none"]
    lines += ["", "## Exempt", ""]
    lines += [f"- `{r['file']}:{r['line']}` → `{r['target']}` — {r['how_checked']}"
              for r in exempt] or ["- none"]
    # Its own section, not a footnote on either of the two above: a swept cite is
    # neither re-pointable (Dangling) nor absent on purpose (Exempt), and the row is
    # the only surviving statement of what the rule cited before the window closed.
    lines += ["", "## Swept by the retention sweep", ""]
    lines += [f"- `{r['file']}:{r['line']}` → `{r['target']}` — {r['how_checked']}"
              for r in swept] or ["- none"]
    lines += ["", f"<!-- ri:dangling {json.dumps(keys)} -->", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--vault", type=Path, default=vault_root())
    ap.add_argument("--repo", type=Path, default=REPO_ROOT)
    ap.add_argument("--home", type=Path, default=Path.home())
    ap.add_argument("--data", type=Path, default=None,
                    help="runtime data root (default <home>/lloyd-data)")
    ap.add_argument("--report", type=Path, help="write the markdown report here")
    ap.add_argument("--json", action="store_true", help="print every record as JSON")
    ap.add_argument("--expect-dangling", action="append", default=[],
                    metavar="TARGET", help="a known-dead cite that must be reported")
    args = ap.parse_args(argv)

    records = scan(vault=args.vault, repo=args.repo, home=args.home, data=args.data)
    if args.report:
        text = render_report(records, previous_dangling(args.report))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")

    if args.json:
        print(json.dumps(records, indent=1))
    else:
        for r in records:
            if r["verdict"] != "resolves":
                print(f"{r['verdict']:8} {r['file']}:{r['line']} -> {r['target']}"
                      f"  [{r['how_checked']}]")
        n = sum(r["verdict"] == "dangling" for r in records)
        print(f"{len(records)} cites, {n} dangling, "
              f"{sum(r['verdict'] == 'exempt' for r in records)} exempt, "
              f"{sum(r['verdict'] == 'swept' for r in records)} swept run records")

    dangling_targets = {r["target"] for r in records if r["verdict"] == "dangling"}
    missed = [t for t in args.expect_dangling if t not in dangling_targets]
    if missed:
        print("expected dangling but not reported: " + ", ".join(missed), file=sys.stderr)
        return 2
    return 1 if dangling_targets else 0


if __name__ == "__main__":
    sys.exit(main())
