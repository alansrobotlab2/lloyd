"""Landing a change to the Obsidian vault through the self-modification loop.

The vault (`~/obsidian`) is not code and cannot be handled like it. It is a
separate git repo and a *live* tree that Lloyd, the nightly jobs and Alan all
write into at once — 65 files were dirty the day this was written — and
nothing reads it from a worktree: `prompt_builder` reads SOUL.md and the
skills straight from the live tree, `autonomy` reads its tasks the same way.
So there is no candidate to gate in isolation; an edit is live the moment it
is saved.

What the loop can still guarantee is the shape of its promise — nothing
lands unverified, and what lands is one commit that can be reverted alone.
The vault route is therefore validate → commit only these paths → revert on
failure:

  1. **Scope.** `.obsidian/**` (the app's own config), `.git/**` and
     `.trash/**` are denied; everything else is writable. Paths that feed a
     prompt or the scheduler — `skills/**`, `lloyd/**`, `autonomy/**` — are
     *validated*: the loaders below actually run against them.
  2. **Front matter.** Every changed `.md` that opens with `---` must parse
     to a mapping. A task file whose YAML broke is a task the scheduler
     silently drops; a skill whose front matter broke is a skill nobody can
     find.
  3. **Loaders**, for validated paths only and only for what changed: the
     system prompt must still build (SOUL.md, memories, the skills index),
     each touched skill must still load through `agent_mcp.skills`, each
     touched task must still parse through `autonomy`. Scoped to the diff on
     purpose — a pre-existing broken file elsewhere in the vault must not
     block every round, the same delta principle as pyflakes and tsc in the
     code gate.
  4. **A failure reverts the round's paths** — tracked ones back to HEAD, new
     ones deleted. The vault is live, so "nothing lands" has to mean "nothing
     stays".
  5. **Success commits exactly these paths**, on `main` (the branch guard
     mirrors `scripts/util/vault-commit.sh`, which every nightly writer
     uses), and records a `vault_land` ledger event with the sha.
     `revert(sha)` is the rollback, and it is a plain `git revert`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.selfmod import spec, state as S

LLOYD_HOME = Path(__file__).resolve().parent.parent.parent
VAULT = Path(os.environ.get("LLOYD_VAULT") or (Path.home() / "obsidian"))
PYTHON = Path(sys.executable)

DENIED_GLOBS: tuple[str, ...] = (".obsidian/**", ".git/**", ".trash/**")
VALIDATED_GLOBS: tuple[str, ...] = ("skills/**", "lloyd/**", "autonomy/**")


class VaultRoundError(RuntimeError):
    pass


def classify(path: str) -> str:
    norm = spec.normalize(path)
    if norm is None or spec._match(norm, DENIED_GLOBS):
        return "denied"
    if spec._match(norm, VALIDATED_GLOBS):
        return "validated"
    return "allowed"


def check_scope(paths: list[str]) -> tuple[bool, str, dict[str, list[str]]]:
    buckets: dict[str, list[str]] = {"denied": [], "validated": [], "allowed": []}
    for p in paths:
        buckets[classify(p)].append(p)
    if buckets["denied"]:
        return False, f"denied vault paths: {sorted(buckets['denied'])}", buckets
    return True, "in scope", buckets


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(VAULT), *args],
                          capture_output=True, text=True, check=False)


def frontmatter_error(path: Path) -> str | None:
    """None if the file has no front matter or it parses to a mapping."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"unreadable: {exc}"
    if not text.startswith("---"):
        return None
    lines = text.splitlines()
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return "front matter never closes"
    try:
        fm = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError as exc:
        return f"front matter is not valid YAML: {str(exc).splitlines()[0]}"
    if not isinstance(fm, dict):
        return "front matter is not a mapping"
    return None


_LOADER_SCRIPT = r"""
import json, sys
from pathlib import Path
paths = json.loads(sys.argv[1]); vault = Path(sys.argv[2]); errs = []
if any(p.startswith(("lloyd/", "skills/")) for p in paths):
    from prompt_builder import build_system_prompt
    prompt = build_system_prompt()
    if not isinstance(prompt, str) or len(prompt) < 500:
        errs.append("system prompt failed to build or came back empty")
skills = sorted({p.split("/")[1] for p in paths if p.startswith("skills/") and p.count("/") >= 2})
if skills:
    from agent_mcp.skills import _load_skill
    for name in skills:
        d = vault / "skills" / name
        if d.is_dir() and _load_skill(d) is None:
            errs.append(f"skills/{name}: does not load")
tasks = [p for p in paths if p.startswith("autonomy/") and p.endswith(".md")]
if tasks:
    from autonomy import _parse_task_file
    for p in tasks:
        f = vault / p
        if f.exists() and _parse_task_file(f) is None:
            errs.append(f"{p}: autonomy cannot parse it")
print(json.dumps(errs))
"""


def loader_errors(paths: list[str]) -> list[str]:
    """Run the real loaders in a fresh interpreter, scoped to `paths`."""
    r = subprocess.run([str(PYTHON), "-c", _LOADER_SCRIPT, json.dumps(paths), str(VAULT)],
                       cwd=str(LLOYD_HOME), capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        return [f"loader crashed: {(r.stdout + r.stderr).strip()[-600:]}"]
    try:
        return list(json.loads(r.stdout.strip().splitlines()[-1]))
    except (ValueError, IndexError):
        return [f"loader produced no verdict: {(r.stdout + r.stderr).strip()[-300:]}"]


def validate(paths: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """(errors, buckets). Empty errors means the change may land."""
    ok, why, buckets = check_scope(paths)
    if not ok:
        return [why], buckets
    errors: list[str] = []
    for p in paths:
        f = VAULT / p
        if f.exists() and f.suffix == ".md":
            err = frontmatter_error(f)
            if err:
                errors.append(f"{p}: {err}")
    if not errors and buckets["validated"]:
        errors.extend(loader_errors(buckets["validated"]))
    return errors, buckets


def revert_paths(paths: list[str]) -> list[str]:
    """Put the round's paths back: tracked ones to HEAD, new ones removed."""
    undone: list[str] = []
    for p in paths:
        tracked = _git("cat-file", "-e", f"HEAD:{p}").returncode == 0
        if tracked:
            _git("checkout", "HEAD", "--", p)
            undone.append(p)
        elif (VAULT / p).exists():
            (VAULT / p).unlink()
            undone.append(p)
    return undone


def _ensure_main() -> None:
    branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch == "main":
        return
    # Same policy as vault-commit.sh (#341): nothing commits to a stranded
    # experiment branch. Plain checkout, not -f: if main cannot be reached
    # without discarding work, that is a human's problem, not this round's.
    r = _git("checkout", "main")
    if r.returncode != 0:
        raise VaultRoundError(f"vault HEAD is on {branch!r} and main cannot be checked out: "
                              f"{r.stderr.strip()[:200]}")


def land(paths: list[str], message: str, *, item_id: int | None = None) -> dict:
    norm = []
    for p in paths:
        n = spec.normalize(p)
        if n is None:
            raise VaultRoundError(f"not a safe vault-relative path: {p!r}")
        norm.append(n)
    if not norm:
        raise VaultRoundError("no paths given")
    if not (message or "").strip():
        raise VaultRoundError("a commit message is required")

    errors, buckets = validate(norm)
    if errors:
        undone = revert_paths(norm) if not buckets["denied"] else []
        S.append_event({"event": "vault_land", "ok": False, "item_id": item_id,
                        "paths": norm, "errors": errors[:10], "reverted": undone})
        raise VaultRoundError("validation failed; the change was reverted: "
                              + "; ".join(errors[:5]))

    _ensure_main()
    add = _git("add", "-A", "--", *norm)
    if add.returncode != 0:
        raise VaultRoundError(f"git add failed: {add.stderr.strip()[:300]}")
    if _git("diff", "--cached", "--quiet").returncode == 0:
        raise VaultRoundError("nothing to commit on those paths")
    commit = _git("commit", "-q", "-m", message.strip())
    if commit.returncode != 0:
        _git("reset", "-q", "--", *norm)
        raise VaultRoundError(f"git commit failed: {(commit.stdout + commit.stderr).strip()[:300]}")
    sha = _git("rev-parse", "HEAD").stdout.strip()
    S.append_event({"event": "vault_land", "ok": True, "item_id": item_id, "commit": sha,
                    "paths": norm, "validated": buckets["validated"],
                    "message": message.strip()[:200]})
    return {"ok": True, "commit": sha, "paths": norm, "validated": buckets["validated"]}


def revert(sha: str, reason: str = "manual") -> dict:
    sha = (sha or "").strip()
    if not sha or _git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
        raise VaultRoundError(f"no such commit in the vault: {sha!r}")
    _ensure_main()
    r = _git("revert", "--no-edit", sha)
    if r.returncode != 0:
        _git("revert", "--abort")
        raise VaultRoundError(f"revert conflicted and was aborted: {r.stderr.strip()[:300]}")
    new = _git("rev-parse", "HEAD").stdout.strip()
    S.append_event({"event": "vault_revert", "reverted": sha, "commit": new, "reason": reason[:300]})
    return {"ok": True, "reverted": sha, "commit": new}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="vault_round")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("validate"); a.add_argument("paths", nargs="+")
    b = sub.add_parser("land"); b.add_argument("-m", "--message", required=True)
    b.add_argument("--item", type=int); b.add_argument("paths", nargs="+")
    c = sub.add_parser("revert"); c.add_argument("sha"); c.add_argument("--reason", default="manual")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "validate":
            errors, buckets = validate(args.paths)
            print(json.dumps({"ok": not errors, "errors": errors, "buckets": buckets}, indent=2))
            return 0 if not errors else 1
        if args.cmd == "land":
            print(json.dumps(land(args.paths, args.message, item_id=args.item), indent=2))
            return 0
        print(json.dumps(revert(args.sha, args.reason), indent=2))
        return 0
    except VaultRoundError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
