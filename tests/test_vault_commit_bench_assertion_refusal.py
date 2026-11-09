"""The bench assertion rail at the `vault-commit.sh` door (#3974 — #3870's second door).

`app/harness/bench_authoring.py` refuses a numbered bench card whose stem has no assertion
set in `eval/autoresearch_assertions.yaml`, on four writer lanes; since #3870 (`b6feeef0`)
`scripts/automod/vault_round.py::land()` refuses it at the landing door too. The third way
onto the vault's main had no bench reference of any kind: `scripts/util/vault-commit.sh`,
whose `git add -A` of whatever the tree holds runs nightly under the subject
`pre-flight … (unattributed dirty state)`. That door carried bench_024–028 in unrefused —
`d1c89cf81`, `bf8be644e`, `dc72ec91a` — and #3870's own land-rail comment said so out loud:
it "does NOT close `scripts/util/vault-commit.sh`, the second door".

Triage reproduced the asymmetry rather than inferring it. A scratch vault with
`lloyd/bench/bench_99_premise_repro.md` staged ran clean through the sweep door (exit 0,
`create mode 100644 lloyd/bench/bench_99_premise_repro.md` in the commit), and the same
path fed to `bench_card_landing_defect(…, exists_at_head=False)` returned a refusal. Two
doors, one input, opposite answers. A guard on only one of two write surfaces is not a
guard (#2468).

What the fix is, and the two places it deliberately differs from its siblings — both
pinned below rather than asserted in prose.

* **The decision is the rail's.** `scripts/util/bench_assertion_findings.py` imports
  `app.harness.bench_authoring` and answers through `RAIL.BENCH_TASK_FILE_RE` and
  `RAIL.bench_card_landing_defect(path, exists_at_head=…, table_path=…)` — the same two
  calls `vault_round.py::_bench_card_assertion_errors` makes. No stem pattern and no table
  read is restated at the second door, because a cloned rule is a rule that drifts from the
  one the writers apply, and then "covered at both doors" means whatever the weaker copy
  says.
* **DROP, not refuse.** The orphan-topic and timezone rungs beside it can exit 1, because
  their remedy is a vault file the same commit can carry (stage the `lloyd/MEMORY.md` hook
  line). This remedy is not: `eval/autoresearch_assertions.yaml` is in the LLOYD REPO, and a
  nightly vault snapshot can neither carry it nor wait for it. `git reset` + `exit 1` there
  leaves the card dirty, the next bare invocation re-stages it, and every pre-flight commit
  afterwards is refused for a file no job inside the vault can fix — the nightly-wide red
  #2415's owed ruling 2 withheld, which #3870's ruling named as the thing to avoid. So the
  sidecar rung's shape (#3236): `git reset -q --` the path (bytes untouched on disk), one
  column-0 line naming it and the table it needs, and the rest of the commit goes through.
* **Fails closed on its own missing rail**, where the four siblings fail open by ruling. An
  unimportable `app.harness.bench_authoring` (exit 4) is the guard going blind: never
  transient, never a writer's fault, and committing through it is a clean-looking reading
  of a check that looked at nothing. So that one code refuses the commit; every other
  exit past 1 keeps the siblings' `committing whatever the index holds` branch, because a
  `git diff --cached` that will not answer can be transient and a freeze here is every
  later job's snapshot.

Scope is `git diff --cached --diff-filter=A` over `lloyd/bench/` — this commit's own adds,
never a whole-tree count — with `exists_at_head` from `git cat-file -e HEAD:<path>`, so an
edit to a card main already holds is never refused even while its key is missing. Nothing
on disk is violated today (`log --diff-filter=A --since=2026-11-06 -- lloyd/bench/` is
empty; the table's keys diff clean against the vault's card stems), so this file is the
whole verification, over scratch vaults; the live-window cross-read stays owed on #3870.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts" / "util"
_WRAPPER = _SCRIPTS / "vault-commit.sh"
_SHIM = _SCRIPTS / "bench_assertion_findings.py"

sys.path.insert(0, str(_REPO_ROOT))
from app.harness import bench_authoring as RAIL  # noqa: E402
from scripts.autoresearch import judge  # noqa: E402

#: The card the item and #3870's triage both spell this way, so the acceptance check and
#: this file cannot drift apart.
KEYLESS_CARD = "bench_99_premise_repro"
CARD_PATH = f"lloyd/bench/{KEYLESS_CARD}.md"
CARD_BODY = "---\ntype: note\n---\n\n# premise repro\n"


def _keyed_stem() -> str:
    """A bench-shaped stem the LLOYD REPO's own table has real rows for.

    Read out of the table rather than hardcoded, because a hardcoded name is a fixture that
    silently stops testing anything the day the card is renamed or its rows are dropped —
    and this is the only arm in the file that cannot use a scratch table: the shim resolves
    its own table to the checkout it runs from (no seam, on purpose, so no caller can aim
    it at an easier file), so the accept arm has to be a real key.
    """
    table = judge.load_assertions()
    for stem in sorted(table):
        entry = table[stem]
        if isinstance(entry, list) and entry and RAIL.BENCH_TASK_FILE_RE.match(stem + ".md"):
            return stem
    raise AssertionError(
        f"{judge.ASSERTIONS_PATH} holds no bench-shaped key with rows, so the accept arm "
        "cannot be pinned here — that is the table going wrong, not this fixture")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)


@pytest.fixture()
def vault_repo(tmp_path, monkeypatch):
    """A scratch vault with one commit, so ADD and EDIT are different answers.

    `exists_at_head` is a git read against HEAD, so an empty directory cannot express the
    clause it decides; and the wrapper refuses to commit a vault that is not on `main`, so
    the branch is named for it.
    """
    repo = tmp_path / "vault"
    (repo / "lloyd" / "bench").mkdir(parents=True)
    monkeypatch.setenv("LLOYD_VAULT", str(repo))
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@e")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "seed.md").write_text("seed\n", encoding="utf-8")
    (repo / "notes.md").write_text("notes\n", encoding="utf-8")
    _git(repo, "add", "-A")
    assert _git(repo, "commit", "-qm", "base").returncode == 0
    return repo


def _run_wrapper(vault: Path, msg: str, paths: "list[str] | None" = None,
                 python: "str | None" = None) -> subprocess.CompletedProcess:
    """The door, with this interpreter handed to it the way the nightly's is.

    `VAULT_DIR` is what the wrapper reads; `LLOYD_VAULT` is what `app.paths` reads. A rung
    reached from a bash door needs both pointed at the scratch tree, or the guard runs
    against the live vault — of all things. `LLOYD_PYTHON` is the wrapper's own interpreter
    knob (`vault-commit.sh` resolves `$PY` from it first): the rung imports
    `app.harness.bench_authoring`, which needs the checkout's dependencies, and the
    wrapper's fallback (`command -v python3`) is a system interpreter that cannot import
    that package — under the fallback the rung answers exit 4 and the door refuses, which
    is this file's own fail-closed branch firing at the fixture.
    """
    env = dict(os.environ, LLOYD_VAULT=str(vault), VAULT_DIR=str(vault),
               LLOYD_PYTHON=python or sys.executable)
    env.pop("LLOYD_JOB_WRITES", None)
    argv = ["bash", str(_WRAPPER), msg]
    if paths is not None:
        argv.append("--")
        argv.extend(paths)
    return subprocess.run(argv, capture_output=True, text=True, timeout=180, env=env)


def _run_shim(vault: Path, *extra: str, python: "str | None" = None,
              shim: "Path | None" = None,
              env: "dict | None" = None) -> subprocess.CompletedProcess:
    return subprocess.run([python or sys.executable, str(shim or _SHIM),
                           "--vault", str(vault), *extra],
                          capture_output=True, text=True, timeout=180,
                          env=dict(os.environ, **(env or {})))


def _commit_paths(repo: Path) -> list[str]:
    if _git(repo, "rev-parse", "--verify", "-q", "HEAD~1").returncode != 0:
        return []
    out = _git(repo, "show", "--name-only", "--format=", "HEAD").stdout
    return sorted(l for l in out.splitlines() if l.strip())


def _commit_message(repo: Path) -> str:
    return _git(repo, "log", "-1", "--format=%B").stdout


def _index_paths(repo: Path) -> list[str]:
    return sorted(l for l in _git(repo, "ls-files").stdout.splitlines() if l.strip())


def _head_has(repo: Path, rel: str) -> bool:
    return _git(repo, "cat-file", "-e", f"HEAD:{rel}").returncode == 0


def _unattributed_paths(stderr: str) -> list[str]:
    """The paths listed under stderr's `unattributed dirty state:` header (#1070)."""
    lines = stderr.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("vault-commit.sh: unattributed dirty state:"):
            out = []
            for following in lines[i + 1:]:
                if following.startswith("    "):
                    out.append(following.strip())
                else:
                    break
            return out
    return []


def _line_index(stderr: str, predicate) -> int:
    """Index of the first stderr line matching `predicate`, or -1."""
    for i, line in enumerate(stderr.splitlines()):
        if predicate(line):
            return i
    return -1


def _add_card(vault: Path, name: str = CARD_PATH, body: str = CARD_BODY) -> Path:
    p = vault / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


# ── clause 1: a keyless ADD does not reach the commit, and the rest of the commit
# ── still happens
def test_a_keyless_bench_card_add_never_reaches_the_commit(vault_repo):
    """The acceptance check, end to end through the door the nightly jobs use."""
    _add_card(vault_repo)
    (vault_repo / "notes.md").write_text("notes changed\n", encoding="utf-8")
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: pre-flight 2026-11-09 "
                                    "(unattributed dirty state)")
    assert proc.returncode == 0, proc.stderr
    assert CARD_PATH not in _commit_paths(vault_repo), \
        f"the keyless card reached main:\n{_commit_message(vault_repo)}"


def test_the_other_staged_paths_still_commit_around_a_dropped_card(vault_repo):
    """A drop costs the card its entry, not the job its snapshot — #2245's stranded
    commits are the failure mode this shape exists to avoid."""
    _add_card(vault_repo)
    (vault_repo / "notes.md").write_text("notes changed\n", encoding="utf-8")
    (vault_repo / "memory").mkdir(exist_ok=True)
    (vault_repo / "memory" / "note.md").write_text("fresh\n", encoding="utf-8")
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: pre-flight 2026-11-09")
    assert proc.returncode == 0, proc.stderr
    committed = _commit_paths(vault_repo)
    assert "notes.md" in committed and "memory/note.md" in committed, committed


def test_a_dropped_card_keeps_its_bytes_and_leaves_the_index(vault_repo):
    """`git reset -q --`, not a deletion: the writer's note survives to be re-staged.

    The index half is the clause the commit-body check cannot make: `git commit` commits
    the INDEX, so a dropped path left staged becomes whichever job commits next, and that
    job's snapshot carries a card it never wrote.
    """
    card = _add_card(vault_repo)
    (vault_repo / "notes.md").write_text("notes changed\n", encoding="utf-8")
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: pre-flight 2026-11-09")
    assert proc.returncode == 0, proc.stderr
    assert card.read_text(encoding="utf-8") == CARD_BODY, \
        "the drop rewrote the card's bytes"
    assert CARD_PATH not in _index_paths(vault_repo)
    assert CARD_PATH not in _git(vault_repo, "diff", "--cached", "--name-only").stdout


# ── clause 4: the report is loud, column 0, actionable from this door, and the path
# ── appears nowhere in what the commit says about itself
def test_the_drop_line_names_the_path_the_table_and_where_the_rows_must_land(vault_repo):
    _add_card(vault_repo)
    (vault_repo / "notes.md").write_text("notes changed\n", encoding="utf-8")
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: pre-flight 2026-11-09")
    lines = proc.stderr.splitlines()
    drop = [l for l in lines if "BENCH ASSERTION DROP" in l and CARD_PATH in l]
    assert drop, f"no column-0 drop line naming {CARD_PATH}:\n{proc.stderr}"
    line = drop[0]
    assert line.startswith("vault-commit.sh: BENCH ASSERTION DROP"), \
        f"the drop line is not at column 0: {line[:60]!r}"
    assert "eval/autoresearch_assertions.yaml" in line, line
    assert "lloyd repo" in line, line
    assert "first" in line.lower(), line
    # The false instruction this item's own draft carried, and which the triage struck:
    # a sweep commit has no round and no repo, so "in the SAME commit" is not a remedy
    # anyone here can perform.
    assert "same commit" not in line.lower(), line
    assert "same round" not in line.lower(), line


def test_the_rail_says_its_own_sentence_for_the_dropped_card(vault_repo):
    """The rung's stderr carries `bench_card_landing_defect`'s text unedited, so the two
    doors describe one defect in one set of words."""
    _add_card(vault_repo)
    rail_text = RAIL.bench_card_landing_defect(CARD_PATH, exists_at_head=False)
    assert rail_text and "has no assertion set" in rail_text
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: pre-flight 2026-11-09")
    assert rail_text in proc.stderr, proc.stderr


def test_the_dropped_path_is_absent_from_the_commit_body_and_the_unattributed_list(vault_repo):
    """#1070's own accounting must not name a path this wrapper decided not to take —
    which is why the rung runs before the index readback, not after it."""
    _add_card(vault_repo)
    (vault_repo / "notes.md").write_text("notes changed\n", encoding="utf-8")
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: pre-flight 2026-11-09")
    assert proc.returncode == 0, proc.stderr
    assert "unattributed dirty state:" in proc.stderr, \
        "the fixture must produce the #1070 block, or this node proves nothing"
    assert CARD_PATH not in _commit_message(vault_repo), _commit_message(vault_repo)
    assert CARD_PATH not in _unattributed_paths(proc.stderr), proc.stderr
    assert "notes.md" in _unattributed_paths(proc.stderr)
    drop_at = _line_index(proc.stderr, lambda l: "BENCH ASSERTION DROP" in l)
    block_at = _line_index(proc.stderr, lambda l: l.startswith(
        "vault-commit.sh: unattributed dirty state:"))
    assert 0 <= drop_at < block_at, \
        "the drop must be reported before the index readback, or the block can name it"


# ── clause 3: the one-way arm — an existing card is never newly refused
def test_a_card_whose_stem_the_table_keys_is_committed_unchanged(vault_repo):
    """The positive control: with a real key behind it, the same door takes the card.

    Without this node every refusal above could be "the rung drops everything".
    """
    stem = _keyed_stem()
    rel = f"lloyd/bench/{stem}.md"
    _add_card(vault_repo, name=rel, body=f"---\ntype: note\n---\n\n# {stem}\n")
    (vault_repo / "notes.md").write_text("notes changed\n", encoding="utf-8")
    proc = _run_wrapper(vault_repo, "nightly: knowledge write 2026-11-09")
    assert proc.returncode == 0, proc.stderr
    assert rel in _commit_paths(vault_repo), proc.stderr
    assert "BENCH ASSERTION DROP" not in proc.stderr, proc.stderr


def test_editing_a_card_head_already_holds_is_never_dropped(vault_repo):
    """`exists_at_head` from `git cat-file -e HEAD:<path>`: refusing an edit would make
    every existing card unfixable by exactly the commit that would fix it."""
    assert not _head_has(vault_repo, CARD_PATH)
    _add_card(vault_repo)
    # Commit the keyless card BYPASSING the door, which is how it got onto main in the
    # first place (d1c89cf81, bf8be644e, dc72ec91a): a raw `git add -A && git commit`.
    _git(vault_repo, "add", "-A")
    assert _git(vault_repo, "commit", "-qm", "someone else's snapshot").returncode == 0
    assert _head_has(vault_repo, CARD_PATH)
    (vault_repo / CARD_PATH).write_text(CARD_BODY + "\nEdited body.\n", encoding="utf-8")
    proc = _run_wrapper(vault_repo, "nightly: fix a card 2026-11-09")
    assert proc.returncode == 0, proc.stderr
    assert CARD_PATH in _commit_paths(vault_repo), proc.stderr
    assert "Edited body." in _git(vault_repo, "show",
                                  f"HEAD:{CARD_PATH}").stdout
    assert "BENCH ASSERTION DROP" not in proc.stderr, proc.stderr


def test_a_pre_existing_keyless_card_never_blocks_an_unrelated_commit(vault_repo):
    """Never a whole-tree count: the rung answers for this commit's adds, so the cards
    already on main cannot redden a batch that did not create them (#2415 owed ruling 2)."""
    _add_card(vault_repo)
    _git(vault_repo, "add", "-A")
    assert _git(vault_repo, "commit", "-qm", "someone else's snapshot").returncode == 0
    assert _head_has(vault_repo, CARD_PATH), "fixture must leave a keyless card ON MAIN"
    (vault_repo / "notes.md").write_text("notes changed\n", encoding="utf-8")
    shim = _run_shim(vault_repo, "--json")
    assert shim.returncode == 0, shim.stderr
    assert shim.stdout.strip() == "[]", \
        f"the rung counted the tree, not this commit's adds:\n{shim.stdout}"
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: 2026-11-09")
    assert proc.returncode == 0, proc.stderr
    assert "notes.md" in _commit_paths(vault_repo)


# ── clause 2: one matcher, two doors — the shim restates nothing
def test_the_door_shells_out_to_the_rung_and_the_rung_asks_the_rail():
    """`$PY` `$SCRIPT_DIR/bench_assertion_findings.py` `--vault` `$VAULT`, and the decision
    made by the module the writers use — checked as text, because the seam between a bash
    door and a python rung is exactly where a duplicated rule hides."""
    wrapper = _WRAPPER.read_text(encoding="utf-8")
    assert 'BENCH_RUNG="$SCRIPT_DIR/bench_assertion_findings.py"' in wrapper
    assert '"$PY" "$BENCH_RUNG" --vault "$VAULT"' in wrapper
    assert '"$SCRIPT_DIR/bench_assertion_findings.py"' in wrapper
    # Runs BEFORE the index readback, so the commit body and the #1070 block never see it.
    rung_at = wrapper.index('BENCH_PATHS="$("$PY" "$BENCH_RUNG" --vault "$VAULT")"')
    readback_at = wrapper.index('git diff --cached --name-only -z > "$TMP_STAGED"')
    assert rung_at < readback_at
    assert 'git -C "$VAULT" reset -q -- "$BENCH_PATH"' in wrapper


def test_the_shim_restates_no_stem_pattern_and_no_table_lookup():
    """Clause 2's second half, over the CODE (the docstring may name the table — the code
    may not read it)."""
    src = _SHIM.read_text(encoding="utf-8")
    code = src.replace(ast.get_docstring(ast.parse(src)) or "", "")
    assert "from app.harness import bench_authoring" in code
    assert "RAIL.BENCH_TASK_FILE_RE" in code
    assert "bench_card_landing_defect(" in code
    assert "exists_at_head=" in code and "table_path=" in code
    assert "RAIL.TABLE_REL" in code, "the table must come from the rail, not a local path"
    assert "--diff-filter=A" in code and '"--diff-filter=A"' in code
    for restated in ("re.compile", "bench_[", "load_assertions", "import yaml",
                     "autoresearch_assertions.yaml", "BENCH_TASK_FILE_RE ="):
        assert restated not in code, f"the second door restates {restated!r}"


def test_the_shim_scopes_to_this_commit_s_staged_adds_under_lloyd_bench(vault_repo):
    """`--diff-filter=A` over `lloyd/bench/`, and `exists_at_head` read from the vault the
    rung was pointed at — the two arguments that make the rail's answer correct here."""
    _add_card(vault_repo)
    (vault_repo / "notes.md").write_text("changed\n", encoding="utf-8")
    (vault_repo / "lloyd" / "bench" / "index.md").write_text("# index\n", encoding="utf-8")
    _git(vault_repo, "add", "-A")
    out = _run_shim(vault_repo, "--json")
    assert out.returncode == 1, out.stderr
    assert out.stdout.count('"path"') == 1, out.stdout
    assert CARD_PATH in out.stdout
    assert "lloyd/bench/index.md" not in out.stdout, \
        "a non-card file under lloyd/bench/ reached a refusal line"
    assert '"exists_at_head": false' in out.stdout


def test_the_shim_answers_cleanly_when_the_commit_adds_no_card(vault_repo):
    (vault_repo / "notes.md").write_text("changed\n", encoding="utf-8")
    _git(vault_repo, "add", "-A")
    out = _run_shim(vault_repo, "--json")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout


def test_the_shim_asks_the_resolver_for_the_vault_and_never_derives_it_from_home():
    """Which vault the rung reads is a seam, and the shim may not own its answer.

    The door always passes `--vault "$VAULT"`, so the fallback chain only runs for a hand
    invocation — and the failure it can produce is the worst one this rung has: a caller that
    believes it sandboxed itself under one override name gets a reading of the LIVE vault,
    which has no keyless card staged, so the check reports clean and the card rides the next
    snapshot into main. `tests/test_no_runtime_paths_in_code.py::
    test_no_module_outside_the_resolver_computes_the_vault_root_from_home` is the rule that
    caught the first draft spelling `Path.home() / "obsidian"` here; this node pins the same
    rule from the rung's side, so a regression names this door rather than a repo-wide scan.
    """
    src = _SHIM.read_text(encoding="utf-8")
    code = src.replace(ast.get_docstring(ast.parse(src)) or "", "")
    assert "from app.data_root import vault_root" in code
    assert "vault_root()" in code, "the resolver has to be called, not merely imported"
    assert 'os.environ.get("VAULT_DIR")' in code, \
        "$VAULT_DIR is the name THIS door's script reads; ignoring it reads the live vault"
    # The home-joined root, checked on the SYNTAX rather than as a substring: this file's own
    # comments quote the spelling it forbids, and a prose match would fail on the warning
    # instead of the defect. Same shape the neighbour check walks — a path expression joining
    # a `home()`/`expanduser()` read with an `"obsidian"` segment.
    spelled_roots = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, (ast.BinOp, ast.Subscript)):
            continue
        subs = list(ast.walk(node))
        home = any(isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute)
                   and s.func.attr in ("home", "expanduser") for s in subs)
        vault_seg = any(isinstance(s, ast.Constant) and isinstance(s.value, str)
                        and s.value.strip("/ ") == "obsidian" for s in subs)
        if home and vault_seg:
            spelled_roots.append(node.lineno)
    assert not spelled_roots, \
        f"the second door derives its own vault root at line(s) {spelled_roots}"


def test_the_shim_answers_about_the_sandbox_named_by_any_of_the_three_env_names(vault_repo):
    """Behavioural half of the same seam: with no `--vault` and a cwd that is neither vault,
    each sandbox name reaches that sandbox — so no caller silently reads the house."""
    _add_card(vault_repo)
    _git(vault_repo, "add", "-A")
    for name in ("VAULT_DIR", "LLOYD_VAULT", "LLOYD_VAULT_ROOT"):
        env = dict(os.environ)
        for other in ("VAULT_DIR", "LLOYD_VAULT", "LLOYD_VAULT_ROOT"):
            env[other] = str(vault_repo) if other == name else ""
        out = subprocess.run([sys.executable, str(_SHIM), "--json"],
                             capture_output=True, text=True, timeout=180, env=env,
                             cwd="/tmp")
        assert out.returncode == 1, f"{name}: exit {out.returncode}\n{out.stderr}"
        assert CARD_PATH in out.stdout, \
            f"{name}: the rung read some other vault, not the sandbox named by it"


def test_no_named_vault_and_no_resolver_is_a_failed_check_not_a_clean_one(tmp_path):
    """The one branch where the resolver is reached AND unreachable: nothing named a root, so
    the rung must say it did not read (exit 3) and hand the door no paths, never answer `[]`.
    Exit 1 with empty stdout would be worse than either — the door's drop branch would fire on
    nothing, print nothing, and commit the card."""
    scratch = tmp_path / "stripped-checkout" / "scripts" / "util"
    scratch.mkdir(parents=True)
    shim = scratch / "bench_assertion_findings.py"
    shim.write_text(_SHIM.read_text(encoding="utf-8"), encoding="utf-8")
    # `PYTHONPATH` is cleared for the same reason as in the exit-4 node below, and the gate's
    # first run shows why it is not decoration: with the checkout on `PYTHONPATH` the stripped
    # copy DID reach `app.data_root`, the resolver answered `~/obsidian` — the live vault —
    # the index held no keyless card, and the node got exit 0, a clean verdict from a check
    # that had been aimed outside its sandbox. That is the failure this rung exists to make
    # impossible, arriving inside its own test.
    env = dict(os.environ, VAULT_DIR="", LLOYD_VAULT="", LLOYD_VAULT_ROOT="",
               PYTHONPATH="")
    out = subprocess.run([sys.executable, str(shim), "--json"], capture_output=True,
                         text=True, timeout=180, env=env, cwd="/tmp")
    assert out.returncode == 3, f"exit {out.returncode}\n{out.stderr}"
    assert out.stdout.strip() == "", out.stdout
    assert "no vault root" in out.stderr, out.stderr
    assert "CHECK FAILED" in out.stderr, out.stderr


# ── clause 5: a rail that cannot be read stops the commit; the header says so
def test_the_shim_exits_above_one_naming_bench_authoring_when_the_rail_is_unimportable(tmp_path):
    """The copy sits in a tree with no `app` package beside it and no inherited path to one,
    so the import the rung depends on fails the way it can fail in the world rather than
    because a test asserted it."""
    scratch = tmp_path / "stripped-checkout" / "scripts" / "util"
    scratch.mkdir(parents=True)
    scratch.joinpath("bench_assertion_findings.py").write_text(
        _SHIM.read_text(encoding="utf-8"), encoding="utf-8")
    # `PYTHONPATH` is cleared because the state under test is "no `app` package reachable",
    # and the launcher can hand one over from outside the fixture. The gate's first run of
    # this node inherited a `PYTHONPATH` naming the checkout, so the copy in a stripped tree
    # imported the REAL rail, never reached the exit-4 branch, and answered `git ls-files:
    # fatal: not a git repository` — exit 3. A node whose premise the environment silently
    # falsifies is not testing the branch it names.
    out = _run_shim(tmp_path, shim=scratch / "bench_assertion_findings.py",
                    env={"PYTHONPATH": ""})
    assert out.returncode > 1, f"a missing rail read as clean: {out.stdout}"
    assert out.returncode == 4, out.stderr
    assert "bench_authoring" in out.stderr, out.stderr
    assert out.stdout.strip() == "", \
        "an unusable check must not hand the door paths to unstage"
    # Bounded, on any layout: at most one attempt (a machine with no venv candidate prints
    # none, one with a candidate prints exactly one), and a second failure under the flag
    # is reported instead of retried — an unbounded re-exec loop under a door that fails
    # closed is a door that never commits.
    assert out.stderr.count("re-running under") <= 1, out.stderr
    again = _run_shim(tmp_path, shim=scratch / "bench_assertion_findings.py",
                      env={"LLOYD_BENCH_RAIL_REEXEC": "1", "PYTHONPATH": ""})
    assert again.returncode == 4, again.stderr
    assert "re-running under" not in again.stderr, again.stderr
    assert "bench_authoring" in again.stderr, again.stderr


def test_the_rung_still_answers_under_the_door_s_last_resort_python3(vault_repo):
    """`command -v python3` — the `$PY` fallback — has yaml but not `httpx`, and
    `app/harness/__init__.py` imports `app.harness.loop`, which imports `httpx`. A rung that
    called that exit-4 would freeze EVERY commit made by a caller that did not export
    `LLOYD_PYTHON`, which is most of this door's other test files and any hand invocation:
    the freeze clause 5 asks for would be the nightly-wide red #2415's owed ruling 2
    withheld. So a `ModuleNotFoundError` re-execs once under a venv candidate and answers.

    This is the node that keeps the sibling `vault-commit.sh` suites green; without it the
    closed branch reads as a general refusal to run without an env var.
    """
    sys_py = Path("/usr/bin/python3")
    if not sys_py.is_file():
        pytest.skip("no system python3 to stand in for the door's fallback")
    probe = subprocess.run([str(sys_py), "-c", "import httpx"],
                           capture_output=True, text=True)
    if probe.returncode == 0:
        pytest.skip("this box's system python3 has the app's deps, so it cannot show the gap")
    _add_card(vault_repo)
    _git(vault_repo, "add", "-A")          # the rung reads the INDEX, not the worktree
    out = _run_shim(vault_repo, "--json", python=str(sys_py), env={"LLOYD_PYTHON": ""})
    assert out.returncode == 1, f"exit {out.returncode}\n{out.stderr}"
    assert CARD_PATH in out.stdout, out.stdout
    assert "re-running under" in out.stderr, out.stderr
    assert "BENCH ASSERTION CHECK FAILED" not in out.stderr, out.stderr


def test_the_door_refuses_the_commit_when_the_rung_reports_no_rail(vault_repo, tmp_path):
    """Fail CLOSED where the four siblings fail open: no verdict, no commit.

    The rung's exit code is produced by an interpreter stand-in, which is the honest way
    across this seam — the thing under test is the door's branch on exit 4, and the rung's
    own exit 4 is pinned by the node above.
    """
    stub = tmp_path / "python-without-the-rail"
    stub.write_text(
        "#!/bin/bash\n"
        'for a in "$@"; do\n'
        '  case "$a" in\n'
        '    *bench_assertion_findings.py)\n'
        '      echo "BENCH ASSERTION CHECK FAILED (no rail): '
        'app.harness.bench_authoring is unimportable (ModuleNotFoundError: '
        "No module named 'app')\" >&2\n"
        '      exit 4;;\n'
        '  esac\n'
        "done\n"
        f'exec "{sys.executable}" "$@"\n',
        encoding="utf-8")
    stub.chmod(0o755)
    _add_card(vault_repo)
    (vault_repo / "notes.md").write_text("notes changed\n", encoding="utf-8")
    before = _git(vault_repo, "rev-parse", "HEAD").stdout.strip()
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: pre-flight 2026-11-09",
                        python=str(stub))
    assert proc.returncode == 1, f"{proc.returncode}\n{proc.stderr}"
    closed = [l for l in proc.stderr.splitlines() if "CHECK FAILED CLOSED" in l]
    assert closed, proc.stderr
    assert closed[0].startswith("vault-commit.sh: bench assertion CHECK FAILED CLOSED")
    assert "bench_authoring" in closed[0], closed[0]
    assert _git(vault_repo, "rev-parse", "HEAD").stdout.strip() == before, \
        "the door committed under a check that had no verdict"
    # Read the content HEAD holds, not the file list of the commit: this refusal happens on
    # the fixture's only commit, so there is no `HEAD~1` for a name-list to diff against and
    # a name-list helper reads as empty whether or not anything was committed. `notes.md` was
    # rewritten to "notes changed" before the door ran, so the original bytes surviving in
    # HEAD is the assertion that can fail.
    assert _git(vault_repo, "show", "HEAD:notes.md").stdout == "notes\n", \
        "HEAD still holds the pre-run bytes, so nothing was committed"


def test_the_header_gains_one_enforcement_item_and_the_three_report_only_lines_are_unchanged():
    """Clause 6 / clause 5's header half: item 5e is new; the autonomy-status (:39-42),
    skill-path (:43-45) and witness (:46-49) blocks stay byte for byte, so a reader cannot
    mistake the new rung for one more reporter."""
    wrapper = _WRAPPER.read_text(encoding="utf-8")
    for pinned in (
            "#      never blocks: it exists so a dispatch-killing `status` flip cannot be\n"
            "#      certified by a commit message that talks about something else. #1127.",
            "#      checkout does not have (the skill-path block below). Reports, never\n"
            "#      blocks. #1969.",
            "#      Reports, never blocks, and prints nothing when nothing matches. #2381."):
        assert pinned in wrapper, f"a report-only header block changed:\n{pinned}"
    assert "#   5e." in wrapper
    item = wrapper.split("#   5e.", 1)[1].split("#   6.", 1)[0]
    assert "bench" in item and "eval/autoresearch_assertions.yaml" in item
    assert "Reports, never blocks" not in item, \
        "the new item must not describe itself as a reporter"


def test_the_wrapper_still_parses_and_the_new_rung_is_silent_on_ordinary_traffic(vault_repo):
    """`bash -n`, and the cheap guarantee every other `vault-commit.sh` node depends on: a
    commit that stages no bench card sees no bench output at all."""
    assert subprocess.run(["bash", "-n", str(_WRAPPER)],
                          capture_output=True, text=True).returncode == 0
    (vault_repo / "notes.md").write_text("notes changed\n", encoding="utf-8")
    (vault_repo / "memory").mkdir(exist_ok=True)
    (vault_repo / "memory" / "x.md").write_text("x\n", encoding="utf-8")
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: 2026-11-09")
    assert proc.returncode == 0, proc.stderr
    assert "BENCH ASSERTION" not in proc.stderr, proc.stderr
    assert "CHECK FAILED" not in proc.stderr, proc.stderr
    assert "memory/x.md" in _commit_paths(vault_repo)
