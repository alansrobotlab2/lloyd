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
  verdict  resolves | dangling | exempt
  how_checked names the check behind every verdict — a `resolves` without one
  would be a pass nobody can re-run.

Precision is the whole risk (the item's step-4 ablation found 13 of 18 naive
hits were noise), so only backticked spans are read, fenced code is skipped,
and slash commands, `/v1/...` endpoints, templates (`YYYY-MM-DD.md`, `<id>`,
globs), URLs, commands with spaces and truncated ids are never extracted.

A cite is `exempt` when its line says the target is gone on purpose ("no longer
exists", "retired", ...), when it carries `<!-- ri:known-absent -->`, or when
its target is in `KNOWN_ABSENT` — the same shape as skill_lint's
`KNOWN_ABSENT_SCRIPTS`: absent path -> why that is fine. Exempt cites do not
count toward the exit code.

Not covered yet: symbol, config-key and heading cites (a `path:N` beyond EOF is
the only drift checked), and no nightly task runs it — arming one is a person's
call (the item's human clause 2).

    python -m scripts.maintenance.referential_integrity [--json]
        [--report ~/obsidian/autonomy/referential-integrity-latest.md]
        [--expect-dangling TARGET ...]

Exit 0 clean, 1 at least one dangling cite, 2 an `--expect-dangling` target was
not reported dangling (a self-check that has stopped catching its own fixture).
"""
from __future__ import annotations

import argparse
import functools
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Vault-relative; these are the files the prompt builder loads whole.
CORPUS = ("lloyd/USER.md", "lloyd/MEMORY.md", "memory/mental-models.md")

KNOWN_ABSENT: dict[str, str] = {}

EXEMPT_MARKER = "<!-- ri:known-absent -->"
_ABSENCE_PROSE = re.compile(
    r"no longer exist|does not exist|doesn't exist|did not exist|was deleted|"
    r"were deleted|was removed|were removed|\bretired\b|\bis gone\b|\bare gone\b|"
    r"\bnever landed\b|\bnever existed\b|\babsent\b|\bsuperseded\b|\brenamed\b|"
    r"\bmissing\b|\bneither is\b",
    re.IGNORECASE,
)

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
    tried = []
    for root, path in _candidates(target, citing, vault=vault, repo=repo,
                                  home=home, data=data):
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


def render_report(records: list[dict], previous: set[str] | None) -> str:
    dangling = [r for r in records if r["verdict"] == "dangling"]
    exempt = [r for r in records if r["verdict"] == "exempt"]
    keys = sorted({_key(r) for r in dangling})
    new = keys if previous is None else [k for k in keys if k not in previous]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# Referential integrity — loaded memory",
        "",
        f"Run {stamp} by `scripts/maintenance/referential_integrity.py` (#882). "
        "Report only: nothing cited here was edited.",
        "",
        f"- total cites: {len(records)}",
        f"- dangling: {len(dangling)}",
        f"- exempt: {len(exempt)}",
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
    lines += ["", f"<!-- ri:dangling {json.dumps(keys)} -->", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--vault", type=Path, default=Path.home() / "obsidian")
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
              f"{sum(r['verdict'] == 'exempt' for r in records)} exempt")

    dangling_targets = {r["target"] for r in records if r["verdict"] == "dangling"}
    missed = [t for t in args.expect_dangling if t not in dangling_targets]
    if missed:
        print("expected dangling but not reported: " + ", ".join(missed), file=sys.stderr)
        return 2
    return 1 if dangling_targets else 0


if __name__ == "__main__":
    sys.exit(main())
