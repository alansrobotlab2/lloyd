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


# --- #2311: the sessions: comment states no count, and says what replaced it --

# A live document count in any spelling an editor reaches for: with or without the
# `~` that means "approximately", with or without thousands separators, and with or
# without the word `indexed` — the last one because `~650 indexed documents` is the
# exact string this template carried until #2089 removed it in `aa47d6ec`, so a
# pattern that missed it would be a pattern fitted to the wrong crime.
COUNT_PHRASE = re.compile(r"~?\d[\d,]*\s*(?:indexed\s+)?documents", re.I)

# The removed sentence, byte for byte as it shipped (the deleted line of
# `aa47d6ec`): the control is the real string, not a paraphrase of it.
REMOVED_COUNT_SENTENCE = (
    "    # ~/obsidian/sessions, which exists but is empty. ~650 indexed documents")


def _sessions_comment_block() -> str:
    """The comment lines the template carries under its own `sessions:` key.

    Read as text rather than through `yaml.safe_load`, which discards comments: the
    object under test is what an operator reads and no other node in this file can
    see, because every one of them goes through the parser. The block is the run of
    `#` lines between `  sessions:` and its `path:`, so another collection's comment
    is not folded into it. The assert inside is the denominator: an extractor
    returning `""` matches no count phrase and contains no literal either, and would
    turn all four claims below into passes earned from nothing.
    """
    lines = TEMPLATE.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.rstrip() == "  sessions:")
    block = []
    for ln in lines[start + 1:]:
        if not ln.lstrip().startswith("#"):
            break
        block.append(ln.strip())
    assert block, (
        "the `sessions:` collection carries no comment block, so every statement "
        "below about what it says would be a statement about nothing")
    return "\n".join(block)


def test_the_sessions_comment_states_no_document_count_and_keeps_its_measure():
    """#2311 clauses 1-3: the count is banned, the query is mandatory, and the
    refusal is proven on the string that actually shipped.

    WHY A NUMBER MAY NOT BE WRITTEN HERE. The comment asserted `~650 indexed
    documents` from 2026-09-07 until #2089 took it out (`aa47d6ec`, 2026-10-04); by
    then #1064 had retracted 514 non-conversation exports on 2026-09-27 and the
    data-home move had re-rooted the collection under `~/lloyd-data`, so the
    reassuring figure described a world two changes dead. A stale count is worse
    than no count: it is what let a reader approve a re-sync from the template
    without measuring, which is the failure that dropped ~650 session documents in
    the 2026-09-19 drift. So `:68` ("No count is stated here, because one goes
    stale") and the query beside it are one device, and this node holds both halves
    — a ban that keeps the prohibition but loses the query invites the number back,
    and a query kept beside a re-added number is a contradiction nobody enforces.

    Nothing else in the suite can catch the re-add: `test_the_patched_template_points_sessions_at_the_real_export_directory`
    and its neighbours assert on parsed paths, SETUP.md prose, the dropped `facts`
    and the drift seam, and all of them see the file only after `yaml.safe_load`
    has thrown the comments away.
    """
    block = _sessions_comment_block()

    # Control first, and through the same pattern object: `search` answers None
    # just as quietly for a block that is empty as for a regex that lost its
    # `documents`, so the pattern's power is shown before its verdict is trusted.
    control = COUNT_PHRASE.search(REMOVED_COUNT_SENTENCE)
    assert control, (
        "the pattern does not reject the sentence the template really carried, "
        "so the assertion below would pass on a regex that matches nothing")
    assert control.group().lower().endswith("documents"), control.group()

    stale = COUNT_PHRASE.search(block)
    assert stale is None, (
        f"the `sessions:` comment states a live document count ({stale.group()!r}) "
        f"— that is the defect #2089 removed: measure with the query already in "
        f"this block instead, and see the 514-export retraction of 2026-09-27 for "
        f"how fast such a figure decays")

    assert "No count is stated here" in block, (
        "the sentence that forbids a count here is gone, so the next hand edit has "
        "no reason written down to argue with")
    assert "collection='sessions'" in block and "active=1" in block, (
        f"the block lost the measuring query ({block!r}); no-count without a "
        f"replacement is how the number comes back")


def test_the_sessions_comment_names_the_background_split_it_does_not_show():
    """#2311 clause 4: the directory an operator counts is not this one.

    `app/paths.py` sends worker and background transcripts to a sibling directory
    (`VAULT_BACKGROUND_SESSIONS_DIR`) precisely so the qmd watcher does not embed
    them, and `app/post_capture.py` writes there. The template's `sessions:` block
    named neither, so the tree read as a leak: the indexed collection is a small
    minority of what sits under `vault-derived/`, and the only figure a curious
    reader can compute by counting files is the big one, which makes the corpus
    look like it lost most of its documents. Naming the split is the fix; naming it
    with a count would recreate the clause-1 defect, so the count ban is re-run on
    the same block rather than trusted to intention.

    Both names must be the real ones. `sessions-background` is asserted as
    `paths.VAULT_BACKGROUND_SESSIONS_DIR.name` and not as a literal, and the two
    directories must still share a parent, so a future relocation of the constant
    reddens the comment that describes it instead of leaving the comment to assert
    a directory that no longer exists.
    """
    from app import paths

    block = _sessions_comment_block()

    assert "VAULT_BACKGROUND_SESSIONS_DIR" in block, (
        "the block does not name app.paths.VAULT_BACKGROUND_SESSIONS_DIR, so a "
        "reader comparing file counts across the two directories has nothing to "
        "reconcile them with")
    background = paths.VAULT_BACKGROUND_SESSIONS_DIR
    assert background.name in block, (
        f"the block names a background directory that is not the one the code "
        f"writes to ({background}) — a comment describing a moved path is worse "
        f"than no comment")
    assert background.parent == paths.VAULT_SESSIONS_DIR.parent, (
        f"{background} is no longer beside the indexed sessions export "
        f"({paths.VAULT_SESSIONS_DIR}); the block says it is")

    # The comment's central claim is about a shell script, not about Python: the
    # split exists because `qmd-watcher.sh` embeds one directory and not the other,
    # and nothing in the AST graph or the YAML parse reaches that line. Read it
    # here, because a watcher that started watching both directories would retire
    # the reason for the split while the comment went on asserting it.
    watcher = (ROOT / "agent-services/scripts/qmd-watcher.sh").read_text()
    watched = next(ln for ln in watcher.splitlines()
                   if ln.startswith("SESSIONS="))
    # Compared as a path tail, not absolutely: `app.paths` re-anchors every
    # derived-root constant inside a git worktree (it warns about exactly that on
    # import), so an absolute comparison here would pass on this box and go red in
    # the gate's throwaway worktree — a failure that describes the harness, not the
    # seam. The tail is the same string in both trees.
    tail = "/" + paths.VAULT_SESSIONS_DIR.relative_to(
        paths.VAULT_DERIVED_ROOT.parent.parent).as_posix() + '"'
    assert watched.rstrip().endswith(tail), (
        f"qmd-watcher.sh watches {watched!r}, which no longer ends in the "
        f"collection's own path tail {tail} — the template comment and the watcher "
        f"disagree about what is embedded")
    assert "sessions-background" not in watcher, (
        f"the watcher gained {background.name}, so background transcripts are "
        f"embedded after all and this template's comment is now a false account of "
        f"why they are not")

    stale = COUNT_PHRASE.search(block)
    assert stale is None, (
        f"the background split is documented with a document count "
        f"({stale.group()!r}); the split is the point, and the count is what this "
        f"file's own history says not to write")
