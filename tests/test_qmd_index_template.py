"""The tracked qmd collection template, as an operator reads it.

`agent-services/conf/qmd-index.yml` is a hand-maintained copy of the config the
qmd daemon actually reads (`~/.config/qmd/index.yml`). Nothing opens it at
runtime — there is no installer — so its whole job is to be true when a human, a
restore or a new host reads it. Between 2026-09-07 (`91a59f9`, the template's
only commit) and 2026-09-19 the live file was edited twice without the copy being
re-synced: `sessions` moved from `~/obsidian/sessions` (a directory that exists
and is empty) to `~/lloyd/_pipeline/vault-derived/sessions`, which is where the
session exports really are and where ~650 indexed documents were being served, and
`facts` was dropped. Following SETUP.md's install direction during that window
retargeted a live collection at an empty directory (#1298).

Why a patch file is checked in here: `agent-services/**` and `SETUP.md` are in
neither `ALLOWED_GLOBS` nor `PROTECTED_GLOBS` (`scripts/automod/spec.py`), so
`spec.check_scope` refuses a round that edits them — the change has to be applied
by a person or by the backlog task the round spawns (#1301). A pointer in prose
would be the same claim with more steps, so the exact patch the round authored is
the artifact, and every test here applies it to copies of HEAD's two files.
"""
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "agent-services/conf/qmd-index.yml"
SETUP = ROOT / "SETUP.md"
PATCH = ROOT / "scripts/maintenance/qmd-index-template-1298.patch"

# Where the session exports actually live: app/paths.py:71 VAULT_SESSIONS_DIR,
# written by app/post_capture.py, indexed as the `sessions` collection.
SESSIONS_PATH = "/home/alansrobotlab/lloyd/_pipeline/vault-derived/sessions"
# The one collection whose direction a person still has to decide: the daemon
# dropped it in the 2026-09-19 edit, SETUP.md says facts reach retrieval through
# the knowledge graph rather than qmd. Until that call is made it stays in the
# template and stays reported as drift — which is the check working.
UNDECIDED = "facts"


def _head_file(rel: str) -> str:
    """The file as it stands at HEAD — not the working tree this round edited."""
    return subprocess.run(
        ["git", "-C", str(ROOT), "show", f"HEAD:{rel}"],
        capture_output=True, text=True, check=True,
    ).stdout


def _reconcile(dest: Path) -> None:
    """Reconcile the copies under `dest`, accepting either starting state.

    Two states are legitimate and the suite has to survive both. Patch pending:
    HEAD's files are pre-reconciliation and the patch applies. Patch landed by a
    person (#1301 is a human path, so this is the steady state the day someone
    runs it): the files already contain the change and applying again fails — a
    suite that went red at exactly that moment would be an argument against ever
    doing the thing it exists to verify. Anything else is real damage: the patch
    fits neither way means HEAD moved away from it, and every assertion below
    would then be graded against text nobody wrote.
    """
    apply = subprocess.run(
        ["git", "apply", "-p1", "--whitespace=nowarn", str(PATCH)],
        cwd=dest, capture_output=True, text=True,
    )
    if apply.returncode == 0:
        return
    landed = subprocess.run(
        ["git", "apply", "-p1", "--reverse", "--check", str(PATCH)],
        cwd=dest, capture_output=True, text=True,
    )
    assert landed.returncode == 0, (
        "scripts/maintenance/qmd-index-template-1298.patch fits HEAD's files "
        "neither forwards nor in reverse, so HEAD moved away from it and this "
        f"suite is asserting against stale text.\n{apply.stderr[:500]}"
    )


def _patched(root: Path) -> tuple[Path, Path]:
    """HEAD's template and SETUP.md under `root/lloyd`, reconciled.

    Returns the two paths. Everything runs against copies because an assertion
    written against the working tree would pass on a round that landed nothing,
    and one written against HEAD alone would fail on a round that did: copies plus
    the patch make every assertion about the file a person would actually have.
    """
    dest = root / "lloyd"
    dest.mkdir(parents=True)
    written = []
    for target in (TEMPLATE, SETUP):
        rel = str(target.relative_to(ROOT))
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_head_file(rel))
        written.append(p)
    _reconcile(dest)
    return written[0], written[1]


def _collections_section(setup_text: str) -> str:
    """The Collections section, up to the next `##` part heading."""
    start = setup_text.index("### Collections")
    nxt = re.search(r"^## ", setup_text[start + 10:], re.M)
    return setup_text[start:start + 10 + (nxt.start() if nxt else len(setup_text))]


def _live_like(tmp_path: Path, colls: dict, drop: tuple[str, ...] = ()) -> Path:
    """A config in the shape the daemon's file has after a documented re-sync."""
    p = tmp_path / "live.yml"
    lines = ["collections:"]
    for name, spec in colls.items():
        if name in drop:
            continue
        lines.append(f"  {name}:")
        lines.append(f"    path: {spec['path']}")
    p.write_text("\n".join(lines) + "\n")
    return p


# --- the patch itself ------------------------------------------------------

def test_the_reconciliation_patch_applies_to_the_head_files(tmp_path):
    """The patch is the artifact a person applies, so it must apply to HEAD.

    `git apply` is context-matched: this fails the moment SETUP.md's Collections
    section or the template's `sessions` block moves under it, which is the point.
    A hand-off patch that silently stops applying is how an unreconciled template
    stays unreconciled. `_reconcile` raises when the patch fits neither forwards
    nor in reverse, which is the only state that makes the assertions below
    meaningless.
    """
    tmpl, setup = _patched(tmp_path)
    assert tmpl.exists() and setup.exists()
    assert isinstance(yaml.safe_load(tmpl.read_text())["collections"], dict)


def test_a_landed_reconciliation_does_not_turn_the_suite_red(tmp_path):
    """#1301 is a person's edit; the day they perform it must not be a regression.

    A suite that goes red exactly when its own human path is carried out argues
    against the path, so `_reconcile` accepts "the patch is already in the file".
    Here that branch is exercised on files that carry the change — HEAD's state
    after #1301 — and the reconciled copies still satisfy the same clause, rather
    than merely failing to raise.
    """
    staged = tmp_path / "staged"
    dest = tmp_path / "landed"
    for target in (TEMPLATE, SETUP):
        rel = str(target.relative_to(ROOT))
        for d in (staged, dest):
            p = d / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(_head_file(rel))
    # Land it in the first tree by hand, then ask the helper to reconcile the
    # identical second tree: forwards now fails, reverse-check must pass.
    subprocess.run(
        ["git", "apply", "-p1", "--whitespace=nowarn", str(PATCH)],
        cwd=staged, capture_output=True, text=True, check=True,
    )
    for target in (TEMPLATE, SETUP):
        rel = str(target.relative_to(ROOT))
        (dest / rel).write_text((staged / rel).read_text())

    _reconcile(dest)

    tmpl = yaml.safe_load((dest / str(TEMPLATE.relative_to(ROOT))).read_text())
    assert tmpl["collections"]["sessions"]["path"] == SESSIONS_PATH
    setup = (dest / str(SETUP.relative_to(ROOT))).read_text()
    assert "15 collections" not in _collections_section(setup)


def test_the_patch_changes_only_the_two_files_it_names(tmp_path):
    """A reconciliation hand-off that also touched something else is a surprise."""
    names = set()
    for line in PATCH.read_text().splitlines():
        if line.startswith("diff --git "):
            names.add(line.split(" b/")[-1])
    assert names == {"SETUP.md", "agent-services/conf/qmd-index.yml"}, names


# --- clause 1: the template says where sessions really is ------------------

def test_the_patched_template_points_sessions_at_the_real_export_directory(tmp_path):
    tmpl, _ = _patched(tmp_path)
    colls = yaml.safe_load(tmpl.read_text())["collections"]
    assert colls["sessions"]["path"] == SESSIONS_PATH
    assert "/home/alansrobotlab/obsidian/sessions" not in tmpl.read_text()


def test_the_patch_retargets_sessions_and_no_other_collection(tmp_path):
    """`sessions` moves to where the exports are; the vault roots must not move."""
    before = yaml.safe_load(_head_file(str(TEMPLATE.relative_to(ROOT))))["collections"]
    after = yaml.safe_load(_patched(tmp_path)[0].read_text())["collections"]
    assert set(after) == set(before)
    moved = {n for n in after if after[n].get("path") != before[n].get("path")}
    assert moved == {"sessions"}, f"patch moved collections beyond sessions: {moved}"


# --- clause 4: the contract is visible in the file it describes ------------

def test_the_patched_template_declares_which_file_the_daemon_reads(tmp_path):
    tmpl, _ = _patched(tmp_path)
    header = "\n".join(
        line for line in tmpl.read_text().splitlines()[:22]
        if line.lstrip().startswith("#")
    )
    assert "~/.config/qmd/index.yml" in header
    assert "cp ~/.config/qmd/index.yml agent-services/conf/qmd-index.yml" in header


# --- clause 2: SETUP.md matches the template it documents -----------------

def _documented_paths(section: str, colls: dict) -> set[str]:
    """Every collection path the section states, absolute, tilde-form included.

    SETUP.md writes paths as `~/lloyd/...`; the template writes them absolute.
    Both name the same directory, so the comparison expands rather than
    string-matches — a test that only accepted one form would fail the doc for a
    style choice and pass it for a wrong path.
    """
    home = str(Path.home())
    found = set()
    for spec in colls.values():
        p = str(spec["path"])
        tilde = "~" + p.removeprefix(home)
        if p in section or tilde in section:
            found.add(p)
    return found


def test_the_patched_setup_md_accounts_for_every_collection_the_template_defines(tmp_path):
    """Every name appears, and every path is derivable from what the section says.

    The section states the two checkout-rooted collections' paths in full and
    roots the rest under `~/obsidian`, so derivable means: for each vault-rooted
    name, `~/obsidian/<name>` is what a reader would write — which is checked
    against the template rather than asserted, so the doc cannot drift from the
    config and still pass.
    """
    tmpl, setup = _patched(tmp_path)
    colls = yaml.safe_load(tmpl.read_text())["collections"]
    section = _collections_section(setup.read_text())
    home = str(Path.home())
    for name, spec in colls.items():
        assert f"`{name}`" in section, f"{name} not accounted for"
        p = str(spec["path"])
        if p.startswith(home + "/obsidian/") and name != "subliminal":
            assert "`~/obsidian`" in section, "the vault root the names hang off"
            assert p == f"{home}/obsidian/{name}", (
                f"{name} is only accounted for by being under ~/obsidian, "
                f"but the template points it at {p}")
        else:
            tilde = "~" + p.removeprefix(home)
            assert p in section or tilde in section, (
                f"{name}'s path {p} is not stated in the Collections section")


def test_the_patched_setup_md_stops_calling_sessions_empty(tmp_path):
    """The claim that misled a reader: 'Two of them — facts and sessions — are empty'."""
    tmpl, setup = _patched(tmp_path)
    colls = yaml.safe_load(tmpl.read_text())["collections"]
    section = _collections_section(setup.read_text())
    assert SESSIONS_PATH in _documented_paths(section, colls)
    assert "`facts` and `sessions` — are **empty**" not in section
    assert "15 collections" not in section


def test_the_patched_setup_md_keeps_the_command_the_drift_check_prints(tmp_path):
    """The nightly report tells an operator to run this; SETUP.md must define it."""
    from scripts.maintenance.qmd_index_maintenance import RESYNC_COMMAND
    section = _collections_section(_patched(tmp_path)[1].read_text())
    assert RESYNC_COMMAND in section


def test_the_undecided_facts_collection_is_named_and_its_direction_left_open(tmp_path):
    tmpl, setup = _patched(tmp_path)
    colls = yaml.safe_load(tmpl.read_text())["collections"]
    assert colls[UNDECIDED]["path"] == "/home/alansrobotlab/obsidian/facts"
    section = _collections_section(setup.read_text())
    assert UNDECIDED in section and "#1298" in section


# --- the seam between the two halves --------------------------------------

def test_the_drift_check_reads_the_reconciled_template_as_in_sync(tmp_path):
    """config_drift over the real seam: patched template vs patched-template-live.

    Same collections, same paths, in the order a documented `cp` produces them —
    the state the acceptance check demands — so the answer must be "in sync".
    """
    from scripts.maintenance.qmd_index_maintenance import config_drift
    tmpl, _ = _patched(tmp_path)
    colls = yaml.safe_load(tmpl.read_text())["collections"]
    out = config_drift(tmpl, _live_like(tmp_path, colls))
    assert out["comparable"] is True, out
    assert out["in_sync"] is True and out["drift_count"] == 0, out


def test_the_drift_check_still_names_facts_until_its_direction_is_decided(tmp_path):
    """Against a daemon that dropped `facts`, the check reports exactly that."""
    from scripts.maintenance.qmd_index_maintenance import config_drift
    tmpl, _ = _patched(tmp_path)
    colls = yaml.safe_load(tmpl.read_text())["collections"]
    out = config_drift(tmpl, _live_like(tmp_path, colls, drop=(UNDECIDED,)))
    assert [d["collection"] for d in out["drift"]] == [UNDECIDED], out
    assert out["drift"][0]["kind"] == "template_only"
