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

The reconciliation #1298 wrote as a patch (`agent-services/**` and `SETUP.md`
are outside what a round may land) was applied by hand on 2026-09-22, together
with the data-home move that put both checkout-rooted collections under
`~/lloyd-data` (`architecture/data-home.md`). These tests read the files as
they now stand.
"""
import re
from pathlib import Path

import yaml

from app.paths import ACCOUNT_HOME

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "agent-services/conf/qmd-index.yml"
SETUP = ROOT / "SETUP.md"

# Where the session exports actually live: app.paths.VAULT_SESSIONS_DIR under the
# production data root, written by app/post_capture.py, indexed as `sessions`.
SESSIONS_PATH = "/home/alansrobotlab/lloyd-data/_pipeline/vault-derived/sessions"
# The one collection whose direction a person still has to decide: the daemon
# dropped it in the 2026-09-19 edit, SETUP.md says facts reach retrieval through
# the knowledge graph rather than qmd. Until that call is made it stays in the
# template and stays reported as drift — which is the check working.
UNDECIDED = "facts"


def _patched(root: Path) -> tuple[Path, Path]:
    """The template and SETUP.md as landed (the name is kept from the patch era)."""
    return TEMPLATE, SETUP


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

# --- clause 1: the template says where sessions really is ------------------

def test_the_patched_template_points_sessions_at_the_real_export_directory(tmp_path):
    tmpl, _ = _patched(tmp_path)
    colls = yaml.safe_load(tmpl.read_text())["collections"]
    assert colls["sessions"]["path"] == SESSIONS_PATH
    assert "/home/alansrobotlab/obsidian/sessions" not in tmpl.read_text()


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
    home = str(ACCOUNT_HOME)  # the template names paths absolutely, off the account
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
    home = str(ACCOUNT_HOME)  # the template names paths absolutely, off the account
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
