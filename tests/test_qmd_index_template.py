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

The reconciliation #1298 wrote as a patch was applied by hand on 2026-09-22,
together with the data-home move that put both checkout-rooted collections under
`~/lloyd-data` (`architecture/data-home.md`). It could not have been landed by a
round: at that point `agent-services/**` was outside the writable set entirely.
#2136 admitted this one file — the tracked template and nothing else beside it,
so `livekit.yaml` and the untracked 0600 `livekit.yaml.runtime` in the same
directory are still a human's, and the daemon's own file is outside the repo and
reaches nobody — which is what makes the next drift a round's job instead of a
ten-day wait for a person (the 09-19 to 09-28 gap #1652 eventually closed). These
tests read the files as they now stand.
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
# Dropped by the daemon in the 2026-09-19 edit and from the template on
# 2026-09-28 (#1652): facts reach retrieval through the knowledge graph, and
# `~/obsidian/facts` is an empty directory.
DROPPED = "facts"


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


def test_the_dropped_facts_collection_stays_dropped(tmp_path):
    tmpl, setup = _patched(tmp_path)
    colls = yaml.safe_load(tmpl.read_text())["collections"]
    assert DROPPED not in colls and len(colls) == 14, sorted(colls)
    assert "/home/alansrobotlab/obsidian/facts" not in tmpl.read_text()
    section = _collections_section(setup.read_text())
    assert "still lists `facts`" not in section and "open in #1298" not in section


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


def test_the_drift_check_names_a_collection_the_daemon_dropped(tmp_path):
    """Against a daemon missing one template collection, the check names it."""
    from scripts.maintenance.qmd_index_maintenance import config_drift
    tmpl, _ = _patched(tmp_path)
    colls = yaml.safe_load(tmpl.read_text())["collections"]
    out = config_drift(tmpl, _live_like(tmp_path, colls, drop=("memory",)))
    assert [d["collection"] for d in out["drift"]] == ["memory"], out
    assert out["drift"][0]["kind"] == "template_only"


# --- the #2136 grant: what a round may now rewrite, and what it still may not --

GRANTED_TEMPLATE = "agent-services/conf/qmd-index.yml"


def test_the_file_a_round_may_now_rewrite_is_the_one_the_drift_check_reads():
    """#2136: the grant, the check's template side and the printed re-sync name
    one path, and the live file stays outside all three.

    #1301's owed decision 2 widened `ALLOWED_GLOBS` by exactly this one file so
    that the report's own prescription becomes a round's job. A grant aimed at a
    lookalike buys nothing, so the three names have to be one string: the entry
    spec.py admits, the file `config_drift` compares from, and the destination of
    the `cp` the nightly report prints. All three are read from their modules
    rather than restated, so a `REPO_ROOT` derivation that moved (`qmd_index_maintenance.py:150`,
    `parents[2]` of its own `__file__`) or a re-pointed `RESYNC_COMMAND` goes red
    here instead of leaving a round to land a reconcile nobody reads.

    The fourth assertion is the grant's boundary: `LIVE_CONFIG` is
    `~/.config/qmd/index.yml` (:152), outside the repo, so admitting the tracked
    half cannot make the daemon's file writable and the direction stays
    live -> template.

    Drift itself is *measured* at this head — `comparable: True, drift: [],
    in_sync: True` against this box's live file, which is why the facts drift of
    2026-09-19 to 09-28 is not in this file's future — but deliberately not
    asserted. `config_drift` is "a report entry and never an exit code" (:85-86,
    restated :889) by design, and the failure mode a hard assert would install is
    worse than the one it would catch: the live file is edited by hand, drift sat
    non-empty for ten days last time, and a red node here would refuse *every*
    round in the loop over a stale copy of a doc file — the exact outcome that
    rule exists to prevent. What the suite does hold a rewrite to is everything
    checkable without the live file: `test_the_patched_setup_md_accounts_for_every_collection_the_template_defines`
    (SETUP.md must still describe exactly the collections the template defines) and
    `test_the_dropped_facts_collection_stays_dropped` (14 collections, no `facts`).
    The comparable assertion below is the denominator control: it fails if either
    side stops being readable, because a drift check that compared nothing would
    otherwise report zero drift forever.
    """
    from scripts.automod import spec
    from scripts.maintenance import qmd_index_maintenance as qm

    rel = TEMPLATE.relative_to(ROOT).as_posix()
    assert rel == GRANTED_TEMPLATE
    assert spec.classify(rel) == "allowed", (
        "the file this file guards is not writable by a round, so the drift report "
        "is still a human's errand and #1301's decision did not land")
    assert qm.TEMPLATE_CONFIG == TEMPLATE, (
        f"the drift check compares {qm.TEMPLATE_CONFIG}, not the granted "
        f"{TEMPLATE} — two different files, so the grant protects nothing")
    assert qm.RESYNC_COMMAND.split()[-1] == GRANTED_TEMPLATE, qm.RESYNC_COMMAND
    assert qm.RESYNC_DIRECTION.startswith("live -> template"), qm.RESYNC_DIRECTION
    assert qm.RESYNC_COMMAND in _collections_section(SETUP.read_text()), (
        "the command a round can now run is not the command SETUP.md prescribes")
    assert not qm.LIVE_CONFIG.is_relative_to(ROOT), (
        f"{qm.LIVE_CONFIG} moved inside the repo, where a grant over the tracked "
        f"half reaches it — the live file being unreachable is what keeps the "
        f"reconcile one-directional")
    out = qm.config_drift()
    assert out.get("comparable") is True, out
    assert out["resync_command"].split()[-1] == GRANTED_TEMPLATE, out
