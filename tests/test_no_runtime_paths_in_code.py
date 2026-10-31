"""No code resolves a runtime-data path off the code tree.

Since 2026-09-22 runtime data lives in `~/lloyd-data` (`architecture/data-home.md`)
and every writer reaches it through `app.paths.DATA_ROOT`. Before the move, some
60 sites built their own path — `Path.home()/"lloyd"/"sessions"`,
`Path(__file__).parent/"usage.db"`, `~/lloyd/workers.db` in config — and one
missed site is enough to start a second, silent copy of the data inside the tree
that every delete aimed at the code reaches again. The guardian names such a
copy an hour after it appears (`datawatch.stray_in_tree`); this names the code
that would write it, before it lands.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RUNTIME = (r"(?:sessions|event_logs|_pipeline|autonomy-runs|logs|usage\.db|workers\.db|"
           r"research\.db|mc-state\.json|voice_profiles|agent-services/logs|eval/baselines|"
           # `eval/djev` joined with #2455, when the djev name-prior probe's dated reports
           # became a data-root store (`paths.EVAL_DJEV_REPORTS_DIR`). The trailing
           # `(?![\w.-])` on every pattern is what keeps this out of firing on the probe
           # SCRIPT (`~/lloyd/eval/djev_name_prior_probe.py`, where `_` is a word char) and
           # on the tracked corpus package the probe reads.
           r"eval/djev|data/tool_overrides\.yaml|ww_diag)")
# `ww_diag` joined the list with #1444, and the enumeration alone would not have
# caught it: the corpus was spelled as a hidden dot-directory in the account home
# (`~/.lloyd/` plus the name), which no pattern below looked for because every
# one of them expected the `lloyd/` segment of a path inside the code tree. So
# the sweep needed a shape of its own. It is deliberately narrow — the account
# home's dot-directory plus a data name, not "any dot-directory" — because
# `~/.local/state/lloyd-*` and `~/.cache/lloyd-*` are state that lives outside
# the root on purpose (`architecture/data-home.md`), and a pattern that alarmed
# on those would be routed around.
DOTDIR = re.compile(r"(?:~|\$HOME|\$\{HOME\}|%h|/home/[a-z_][a-z0-9_-]*)/\.lloyd/" + RUNTIME + r"(?![\w.-])")
_Q = r"""["']"""
PATTERNS = [
    # ~/lloyd/X, $HOME/lloyd/X, /home/<user>/lloyd/X, %h/lloyd/X
    re.compile(r"(?:~|\$HOME|\$\{HOME\}|%h|/home/[a-z_][a-z0-9_-]*)/lloyd/" + RUNTIME + r"(?![\w.-])"),
    # Path.home() / "lloyd" / "X"
    re.compile(r"Path\.home\(\)\s*/\s*" + _Q + r"lloyd" + _Q + r"\s*/\s*" + _Q + RUNTIME + _Q),
    # ~/.lloyd/X — the account home's dot-directory (#1444)
    DOTDIR,
    # Path.home() / ".lloyd" / "X" — the same place, spelled in Python
    re.compile(r"Path\.home\(\)\s*/\s*" + _Q + r"\.lloyd" + _Q + r"\s*/\s*" + _Q + RUNTIME + _Q),
    # LLOYD_HOME / "X", LIVE_CHECKOUT / "X", REPO / "X"
    re.compile(r"\b(?:LLOYD_HOME|LIVE_CHECKOUT|LIVE_ROOT|_LLOYD_ROOT|_LLOYD_HOME)\s*/\s*"
               + _Q + RUNTIME + _Q),
    # Path(__file__)…parent / "usage.db" etc.
    re.compile(r"__file__\)[\w.()\[\]]*\s*/\s*" + _Q + RUNTIME + _Q),
]

SUFFIXES = {".py", ".sh", ".conf", ".yaml", ".yml", ".service", ".timer", ".ts", ".tsx"}
#: Files whose job is naming the old layout.
ALLOWED = {
    "scripts/migrate_data_home.py",
    "scripts/maintenance/rewrite_vault_data_paths.py",
    "scripts/maintenance/cutover_data_home.sh",
    "tests/test_no_runtime_paths_in_code.py",
}
#: Dated measurement scripts are a record of how a number was taken, not code
#: that runs; tests carry the old layout as fixture text.
ALLOWED_PREFIXES = ("tests/", "eval/measurements/", "agent-services/llm/")


def _tracked() -> list[str]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True,
                         text=True, check=True).stdout.split("\n")
    return [f for f in out if f and Path(f).suffix in SUFFIXES
            and f not in ALLOWED and not f.startswith(ALLOWED_PREFIXES)]


def test_no_tracked_code_builds_a_runtime_path_off_the_tree():
    hits = []
    for rel in _tracked():
        path = ROOT / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if any(p.search(line) for p in PATTERNS):
                hits.append(f"{rel}:{n}: {line.strip()[:140]}")
    assert not hits, ("runtime data resolved off the code tree — use app.paths "
                      "(DATA_ROOT and its constants) or ${LLOYD_DATA} in config:\n  "
                      + "\n  ".join(hits))


def test_the_patterns_catch_every_spelling_that_shipped():
    shipped = [
        'DB_PATH = Path(__file__).parent / "usage.db"',
        '_STATE_PATH = LLOYD_HOME / "mc-state.json"',
        'SESSIONS_DIR = Path.home() / "lloyd" / "sessions"',
        'raw = cfg.get("db_path") or "~/lloyd/workers.db"',
        'stdout_logfile=/home/alansrobotlab/lloyd/logs/server.log',
        'SESSIONS="$HOME/lloyd/_pipeline/vault-derived/sessions"',
        "ExecStart=%h/lloyd/logs/x",
        # Both spellings WakeMissCapture shipped with until #1444: the corpus in
        # the account home's dot-directory, in a unit and in a reader. The first
        # is built by concatenation because #1444's clause-4 grep forbids the
        # literal in any tracked `.py`, a test's own fixture text included — the
        # assembled string is what the pattern must still catch.
        'DIAG_DIR = Path("~/.lloyd' + '/ww_diag").expanduser()',
        'DIAG = Path.home() / ".lloyd" / "ww_diag"',
        # The spelling #2455 took out of the djev probe's reach. Without `eval/djev` in
        # RUNTIME this line is inert, so it is what makes the enumeration addition
        # load-bearing rather than decorative.
        'REPORT = Path.home() / "lloyd" / "eval/djev" / "name_prior_2026-10-11.json"',
    ]
    for line in shipped:
        assert any(p.search(line) for p in PATTERNS), line
    for fine in ('Path.home() / "lloyd" / "scripts"', "~/lloyd/.venvs/lloyd/bin/python",
                 "~/lloyd-data/sessions", 'LLOYD_HOME / "config.yaml"',
                 # The root's own copy of the corpus, and the state that lives
                 # outside the root deliberately: none is a finding.
                 'DIAG = Path("~/lloyd-data/ww_diag").expanduser()',
                 # #2455's two fine cases: the data root at this very leaf is the correct
                 # spelling, and the probe's own tracked script and corpus directory are
                 # not stores — the word-boundary lookahead is what keeps them out.
                 'REPORT = Path("~/lloyd-data' + '/eval/djev").expanduser()',
                 'CORPUS = ROOT / "eval" / "djev" / "name_prior_corpus.jsonl"',
                 'python ~/lloyd/eval/djev_name_prior_probe.py',
                 '~/.local/state/lloyd-automod/promotions.jsonl',
                 '~/.cache/lloyd-voice-eval/wake-tts'):
        assert not any(p.search(fine) for p in PATTERNS), fine


# ─────────────────  #3403: the vault root is a resolved root, everywhere  ─────────
#
# The rule this file exists for is "one runtime path, one resolver". The vault was the one
# runtime root still computed by its readers: `Path.home() / "obsidian"` appeared in
# nineteen modules, `app.paths.VAULT_ROOT` honoured no override at all, and
# `agent_mcp/vault.py` and `scripts/automod/vault_round.py` each honoured a DIFFERENT one.
# #3403 calls that the dangerous arrangement: a trial that set the name a given reader
# ignores gets a fixture from half the tree and Alan's real vault from the other half, and
# the writer that escaped is invisible to the isolation check that was supposed to catch it.

_VAULT_RESOLVER = "app/data_root.py"
#: The two override names. A stdlib-only reader (`agent-services/guardian/`) cannot import
#: `app.data_root`, so it may keep its own fallback only if it consults BOTH names —
#: honouring one and not the other is the half-root this scan exists to fail.
_VAULT_OVERRIDES = ("LLOYD_VAULT_ROOT", "LLOYD_VAULT")
_HOME_CALLS = {"home", "expanduser"}


def _docstring_lines(tree: ast.AST) -> set[int]:
    """Lines occupied by docstrings, which are prose about a root, not a root."""
    out: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        # `isinstance(body, list)` because `IfExp.body` is a single expression, and this
        # walks every node type rather than only the three that own a docstring.
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(
                body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            d = body[0]
            out.update(range(d.lineno, (d.end_lineno or d.lineno) + 1))
    return out


def _computes_the_vault_from_home(tree: ast.AST) -> int | None:
    """First line where CODE joins a home directory to `obsidian`, or None.

    AST and not regex, because the same characters appear in prose this rule must not
    trip: a docstring explaining the old spelling, or a refusal message quoting it, is not
    a module resolving a root. What does trip it is any path expression — a `BinOp` or a
    `Subscript` — whose subtree contains BOTH a standalone `"obsidian"` segment and a
    `Path.home()` / `expanduser("~")` / `$HOME` read: the shapes a reader can write through.
    """
    doc = _docstring_lines(tree)

    def has_home_read(node: ast.AST) -> bool:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                    and sub.func.attr in _HOME_CALLS:
                return True
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                    and re.match(r"^(HOME|\$HOME)$", sub.value):
                return True
        return False

    def has_vault_segment(node: ast.AST) -> bool:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                    and sub.value.strip("/ ") == "obsidian" and sub.lineno not in doc:
                return True
        return False

    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.Subscript)) \
                and has_vault_segment(node) and has_home_read(node):
            return node.lineno
    return None


def test_no_module_outside_the_resolver_computes_the_vault_root_from_home():
    hits = []
    for rel in _tracked():
        if rel == _VAULT_RESOLVER or not rel.endswith(".py"):
            continue
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (OSError, SyntaxError):
            continue
        line = _computes_the_vault_from_home(tree)
        if line is None:
            continue
        if all(name in text for name in _VAULT_OVERRIDES):
            continue      # a stdlib reader that honours both names, by design
        hits.append(f"{rel}:{line}")
    assert not hits, (
        "these modules resolve the vault from a home directory without consulting both "
        "override names, so a caller that sandboxed itself with the name the reader "
        f"ignores writes the live vault: {hits}")


def test_the_scan_fires_on_the_spelling_it_exists_to_catch():
    """A scan that catches nothing is a green light painted over a red one."""
    caught = (
        'VAULT = Path.home() / "obsidian"\n',
        'return Path.home() / "obsidian" / "lloyd"\n',
        'vault = Path(os.path.expanduser("~")) / "obsidian"\n',
    )
    for src in caught:
        assert _computes_the_vault_from_home(ast.parse(src)) is not None, src
    # and stays silent on the resolver's own fallback, on docstring prose, and on a
    # path built from the resolver.
    # The resolver's own fallback line IS caught by the helper — the scan exempts it by
    # PATH (`_VAULT_RESOLVER`), not by shape, so the exemption is visible in the loop
    # rather than hidden in the predicate.
    for fine in ('VAULT = vault_root() / "lloyd"\n',
                 'def f():\n    """Path.home() / "obsidian" used to live here."""\n'):
        assert _computes_the_vault_from_home(ast.parse(fine)) is None, fine


def test_the_three_vault_readers_return_the_same_path_under_one_override(tmp_path):
    """`app.paths`, `app.data_root` and `vault_round` agree in one fresh interpreter.

    The clause is the agreement, not the resolution: a fixture the runner points three
    readers at three places is a trial whose `vault_write` lands in the house while its KG
    row lands in the copy, and the reward still says the sandbox held.
    """
    fixture = tmp_path / "obsidian"
    fixture.mkdir()
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "import app.paths as p, app.data_root as d\n"
        "from scripts.automod import vault_round as vr\n"
        "print(repr((str(p.VAULT_ROOT), str(d.vault_root()), str(vr.VAULT))))"
        % str(ROOT))
    env = {k: v for k, v in os.environ.items()
           if k not in ("LLOYD_VAULT_ROOT", "LLOYD_VAULT")}
    env["LLOYD_VAULT_ROOT"] = str(fixture)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, env=env, cwd=str(ROOT), timeout=180)
    assert out.returncode == 0, out.stderr[-900:]
    paths = list(ast.literal_eval(out.stdout.strip()))
    assert paths == [str(fixture)] * 3, (
        f"the vault readers disagree under one override: {paths}")


def test_a_symlinked_vault_root_is_reported_as_not_isolated():
    """A gate round's `$HOME/obsidian` is a SYMLINK into the live vault.

    `scripts/automod/worktree.py` deliberately leaves it that way (`HOME_LINK_SKIP` covers
    `lloyd` and `lloyd-data`, not `obsidian`), so a predicate comparing path STRINGS calls
    that link an isolated copy and a scoped trial pointed at it writes Alan's real vault
    while its flags report a sandbox. `is_production_vault` realpaths both sides, so the
    answer is True — production, not isolated — which is what the runner's pre-dispatch
    assertion needs to refuse on.
    """
    from app.data_root import PRODUCTION_VAULT, is_production_vault
    link = Path(tempfile.mkdtemp(prefix="vaultlink-")) / "obsidian"
    link.symlink_to(PRODUCTION_VAULT)
    assert is_production_vault(link) is True, (
        "a symlink to the live vault read as isolated")
    assert is_production_vault(PRODUCTION_VAULT) is True


def test_a_fixture_copy_is_reported_as_isolated(tmp_path):
    from app.data_root import is_production_vault
    copy = tmp_path / "obsidian"
    copy.mkdir()
    assert is_production_vault(copy) is False


def test_no_reader_binds_a_local_named_after_the_function_it_calls():
    """The failure the #3403 sweep invented, and the one a rename to a call cannot see at
    the call site. Readers used to write `VAULT_ROOT / x` and could also hold a local named
    `vault_root = VAULT_ROOT.resolve()`; replacing the CONSTANT with the FUNCTION of that
    near-identical name turned such an assignment into a function-local binding for the
    whole scope, so the earlier call raised `UnboundLocalError` — which is what `app/routers/mc_ui.py`
    did to every Mission Control memory-tab focus with a path-shaped id until this round
    caught it in the static rung.

    Scanned by AST rather than grep, because the pattern is a binding and a call in one
    scope, which a line-based search cannot relate. `nonlocal`/`global` are not exempt: a
    `global vault_root` makes every later call in that module read the global, which is the
    same hazard by another route."""
    import ast

    offenders: list[str] = []
    for pkg in ("app", "agent_mcp", "scripts", "agent-services"):
        for py in (ROOT / pkg).rglob("*.py"):
            try:
                tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue

            funcs = [n for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for fn in funcs:
                called = any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                             and c.func.id == "vault_root" for c in ast.walk(fn))
                if not called:
                    continue
                bound = ""
                for n in ast.walk(fn):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) \
                            and n.id == "vault_root":
                        bound = bound or "assignment"
                    elif isinstance(n, (ast.Global, ast.Nonlocal)) \
                            and "vault_root" in n.names:
                        bound = "global/nonlocal"
                if bound:
                    offenders.append(f"{py.relative_to(ROOT)}:{fn.lineno} "
                                     f"{fn.name} ({bound})")
    assert offenders == [], "a reader would raise UnboundLocalError on the vault root: " + \
        ", ".join(offenders)


#: Every python rung `scripts/util/vault-commit.sh` runs, read out of the hook that runs
#: it. Each is resolved to `$SCRIPT_DIR/<name>` and started BY PATH, so the interpreter
#: puts `scripts/util/` on `sys.path[0]` and not the checkout.
_RUNG_RE = re.compile(r'^[A-Za-z_]+="\$SCRIPT_DIR/([^"]+\.py)"', re.M)

#: Packages that sit at the checkout root, and so cannot be imported by a script the
#: interpreter started somewhere else.
_ROOT_PKGS = ("app", "agent_mcp", "scripts", "eval")


def _import_root_pkg(node) -> str | None:
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").split(".")[0]
    if isinstance(node, ast.Import):
        return (node.names[0].name or "").split(".")[0]
    return None


def test_a_rung_the_vault_commit_hook_starts_by_path_can_import_the_app_package():
    """A rung that reaches the vault through `app.paths` must put the checkout on `sys.path`.

    #3403 routed the vault root through `app.paths`/`app.data_root`, and the reader sweep
    did that to 19 modules. Inside `app/` and `agent_mcp/` that costs nothing. In a module
    the commit hook starts by path it is a `ModuleNotFoundError` at import — and the hook
    answers an unavailable rung by printing `CHECK SKIPPED` and committing anyway, so a
    rung that can no longer import the resolver it was just handed goes green. That is a
    blind check reading as a clean tree, which is the failure this file's header is about.

    The corpus is read from the hook instead of being listed here, because a hand list
    would be a second writer of a fact the `.sh` owns: a rung added at
    `scripts/util/vault-commit.sh` is covered on the day it lands, and one renamed out of
    my list cannot hide. A rung whose module imports a checkout-root package at top level
    must insert the checkout onto `sys.path` ABOVE that import — the idiom is
    `scripts/skill_lint.py:66-68`, which exists for exactly this reason, and the bug that
    earned it is the same one: 194 skills linted, five of them retired, because the import
    that would have said so had died.
    """
    hook = (ROOT / "scripts" / "util" / "vault-commit.sh").read_text(encoding="utf-8")
    rungs = _RUNG_RE.findall(hook)
    assert len(rungs) >= 4, (
        f"the enumeration found {len(rungs)} rung(s): {rungs}. The hook names five; a "
        "changed quoting spelling there would leave this node grading an empty corpus "
        "and passing, which is the outcome it exists to refuse")

    offenders = []
    for name in rungs:
        mod = ROOT / "scripts" / "util" / name
        if not mod.exists():
            continue                       # a rung that ships without its file
        src = mod.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue                       # not this node's problem to grade
        lines = src.splitlines()
        tops = [n.lineno for n in tree.body
                if isinstance(n, (ast.Import, ast.ImportFrom))
                and _import_root_pkg(n) in _ROOT_PKGS]
        if not tops:
            continue                       # imports nothing at the root: startable as-is
        first = min(tops)
        if not any("sys.path.insert" in l for l in lines[:first - 1]):
            offenders.append(f"{name}:{first}  {lines[first - 1].strip()}")
    assert not offenders, (
        "rung(s) started by path that import a checkout-root package with nothing "
        "putting the checkout on sys.path first:\n  " + "\n  ".join(offenders)
        + "\nFix: `sys.path.insert(0, str(Path(__file__).resolve().parents[N]))` above "
          "the import, as scripts/skill_lint.py does.")


def _three_reader_paths(env: dict[str, str]) -> list[str]:
    """`app.paths.VAULT_ROOT`, `app.data_root.vault_root()` and `vault_round.VAULT`,
    read in one fresh interpreter under `env`, as strings.

    A subprocess and not an in-process import because all three bind at import time:
    inside pytest the first importer's environment is already frozen into the module,
    and a test that re-imports to dodge that would be measuring `importlib`, not the
    roots the runner's child processes actually see.
    """
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "import app.paths as p, app.data_root as d\n"
        "from scripts.automod import vault_round as vr\n"
        "print(repr((str(p.VAULT_ROOT), str(d.vault_root()), str(vr.VAULT))))"
        % str(ROOT))
    base = {k: v for k, v in os.environ.items()
            if k not in ("LLOYD_VAULT_ROOT", "LLOYD_VAULT")}
    base.update(env)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, env=base, cwd=str(ROOT), timeout=180)
    assert out.returncode == 0, out.stderr[-900:]
    return list(ast.literal_eval(out.stdout.strip()))


def test_the_vault_readers_agree_under_both_names(tmp_path):
    """The precedence, not just the agreement: both names set, to two different places.

    `scripts/automod/vault_round.py` keeps its own one-line fallback (it is started by
    path by a gate rung, and importing `app` from a round's worktree can bind the wrong
    tree — #2190), so what keeps that copy honest is this node: the two names must order
    the same way in all three readers. `test_the_three_vault_readers_return_the_same_path_under_one_override`
    sets one name and cannot see an ordering disagreement; a reader that preferred the
    fallback name would pass that node and still hand a trial its real vault the moment a
    caller set both — which is the half-root this whole sweep exists to fail.
    """
    root = tmp_path / "root-vault"
    fallback = tmp_path / "fallback-vault"
    for one in (root, fallback):
        one.mkdir()
    both = _three_reader_paths({"LLOYD_VAULT_ROOT": str(root),
                                "LLOYD_VAULT": str(fallback)})
    assert both == [str(root)] * 3, (
        f"the readers disagree on which override wins: {both}")
    alone = _three_reader_paths({"LLOYD_VAULT": str(fallback)})
    assert alone == [str(fallback)] * 3, (
        f"`LLOYD_VAULT` alone no longer reaches every reader: {alone}")
    plain = _three_reader_paths({})
    assert plain == [str(Path.home() / "obsidian")] * 3, (
        f"with no override the readers must land on the account's vault: {plain}")
