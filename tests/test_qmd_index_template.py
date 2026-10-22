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
    """A config in the shape the daemon's file has after a documented re-sync.

    Every field the template declares, not just `path`: the re-sync is `cp
    ~/.config/qmd/index.yml agent-services/conf/qmd-index.yml`, which copies the
    whole file, so a live file that lost every `pattern:` line is not a re-synced
    box — it is a fiction this helper invented while `config_drift` could only
    read paths. #2543 made `pattern` and `ignore` drift fields, which turned the
    fiction into a false `drift_count` here (`in_sync: True` over a pair that now
    differs on 14 collections' `pattern`), and dropping it is what keeps
    `test_the_drift_check_reads_the_reconciled_template_as_in_sync` asserting the
    thing its name says instead of the shape of an old helper.
    """
    p = tmp_path / "live.yml"
    lines = ["collections:"]
    for name, spec in colls.items():
        if name in drop:
            continue
        lines.append(f"  {name}:")
        lines.append(f"    path: {spec['path']}")
        if "pattern" in spec:
            lines.append(f"    pattern: \"{spec['pattern']}\"")
        if "ignore" in spec:
            lines.append("    ignore:")
            lines.extend(f"      - {entry}" for entry in spec["ignore"])
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

    Same collections, same paths, same patterns and ignores, in the order a
    documented `cp` produces them — the state the acceptance check demands — so
    the answer must be "in sync". `drift_count` here is 0 for the pair this
    helper builds, and that is the point of the whole check; it is never asserted
    against the box's real live file, which a person edits by hand (#2543).
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


# --- #2728: `subliminal` is root-level only, and the reason is in the block -----
#
# #2180 measured what a vault-wide root costs: with `pattern: "**/*.md"` rooted at
# `~/obsidian`, every note belonging to one of the eleven segment collections
# nested under it was a `subliminal` row as well, so the same file stood in
# `documents` and in `documents_fts` twice and arrived at the merge as two
# candidates. The ruling of 2026-10-07 (carried by #2351) was to narrow this
# collection to the vault root and keep the eleven, not to retire them. What lands
# here is the tracked half of that ruling; the daemon's own file, being a person's
# edit, was the owed half until #2728's operator leg ran it on 2026-10-15. Both
# sides now carry `pattern: "*.md"`, so `config_drift` has no `pattern` row to
# report for it — but the comparison still runs on every nightly
# (`scripts/maintenance/qmd_index_maintenance.py:84-87`), and the instruction that
# comes with it is unchanged: should a `pattern` row for `subliminal` ever come
# back, the wrong ways to silence it are widening this glob back or running the
# re-sync this file's own header documents, which copies the live file over the
# template.

#: The vault root. `subliminal` is the only collection rooted here rather than
#: inside one of its segments, which is the whole basis of the #2180 finding.
VAULT_ROOT = "/home/alansrobotlab/obsidian"

#: Every collection's `pattern` as this template stands. The clause #2728 has to
#: hold is "nothing else moved", and only a snapshot says what the "else" was: an
#: assertion aimed at `subliminal` alone would also pass on a file where `memory`
#: had quietly gone root-level. A deliberate change to another collection edits
#: this dict in the same commit — that friction is the pin working.
EXPECTED_PATTERNS = {
    "memory": "**/*.md",
    "knowledge": "**/*.md",
    "projects": "**/*.md",
    "lloyd": "**/*.md",
    "personal": "**/*.md",
    "work": "**/*.md",
    "skills": "**/*.md",
    "people": "**/*.md",
    "subliminal": "*.md",
    "backlog": "**/*.md",
    "autonomy": "*.md",
    "autonomy-runs": "*/run_*.md",
    "sessions": "**/*.md",
    "architecture": "**/*.md",
}

#: What the `subliminal:` comment block has to carry for the clause to be worth
#: anything: the ruling's item, the ruling's date, what it decided, the glob it
#: replaced, and the instruction against undoing it. A comment saying only
#: "narrowed" leaves a reader without the thing they need at the moment of the
#: edit, which is what the reported drift will tempt them to undo.
BLOCK_EVIDENCE = ("#2180", "2026-10-07", "Narrowed, not retired", "**/*.md",
                  "DO NOT WIDEN IT BACK")


def _subliminal_comment_block() -> str:
    """The comment lines the template carries inside its own `subliminal:` block.

    The reason `_sessions_comment_block` gives, and the reason it is written the
    same way: `yaml.safe_load` discards comments, and the object under test is the
    paragraph an operator reads before deciding to widen a glob. The block is the
    run of `#` lines between `  subliminal:` and its `path:`, so another
    collection's comment is not folded into it. The assert inside is the
    denominator: an extractor returning `""` contains no `#2180` and no date, and
    every claim about the block below would be a pass earned from nothing.
    """
    lines = TEMPLATE.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.rstrip() == "  subliminal:")
    block = []
    for ln in lines[start + 1:]:
        if not ln.lstrip().startswith("#"):
            break
        block.append(ln.strip())
    assert block, (
        "the `subliminal:` collection carries no comment block, so the ruling this "
        "file is supposed to hand a reader is not in the file at all")
    return "\n".join(block)


def test_the_subliminal_collection_is_root_level_only_and_nothing_else_moved():
    """#2728 clause 1: one glob changed, and it is the glob #2180 ruled on.

    Three assertions, one per thing a reader acts on: the glob, the root it is
    relative to, and the ignores. `path` and `ignore` are not decoration beside the
    pattern — the ruling narrows the pattern and nothing else, and a rewrite that
    moved the root or dropped `agents/**` would change which documents the
    collection serves while leaving `pattern: "*.md"` in place, which is the only
    line a pattern-only test can see.
    """
    colls = yaml.safe_load(TEMPLATE.read_text())["collections"]
    assert set(colls) == set(EXPECTED_PATTERNS), (
        f"{sorted(set(colls) ^ set(EXPECTED_PATTERNS))} of the collections this file "
        "documents is not the set the template defines, so the snapshot below has no "
        "denominator")
    for name, pattern in EXPECTED_PATTERNS.items():
        assert colls[name]["pattern"] == pattern, (
            f"{name}'s pattern is {colls[name]['pattern']!r}, not {pattern!r}: #2728 "
            "narrowed `subliminal`, and nothing in the 2026-10-07 ruling said "
            f"anything about {name}")
    sub = colls["subliminal"]
    assert sub["path"] == VAULT_ROOT, (
        f"`subliminal` is rooted at {sub['path']!r}. Root-level `*.md` means the vault "
        "root's own loose notes, so a moved root quietly changes the file set the "
        "surviving glob matches")
    assert sub["ignore"] == ["agents/**", "templates/**"], sub["ignore"]


def test_the_subliminal_block_hands_a_reader_the_ruling_and_the_refusal():
    """#2728 clause 2: the reason not to widen the glob is in the file, not only
    in the backlog entry that recorded it.

    The clause is not "a comment exists". It is that the next reader of a reported
    `pattern` drift has the argument in front of them, in the file they are about
    to edit — which is the exact moment the 2026-09-07 to 09-19 window went wrong
    for `sessions`, when the reconcile that was run had nothing beside it saying
    which side was true. Each item of `BLOCK_EVIDENCE` is therefore asserted by
    name, and the control that these are assertable at all is the denominator
    inside `_subliminal_comment_block`.
    """
    block = _subliminal_comment_block()
    missing = [needle for needle in BLOCK_EVIDENCE if needle not in block]
    assert not missing, (
        f"the `subliminal:` block no longer carries {missing}; the narrowing without "
        "its reason is how the vault-wide glob comes back as a drift 'fix'")


def test_the_narrow_is_block_scoped_and_the_documented_resync_would_break_it():
    """#2728 clause 3: this file was edited inside one block, not overwritten.

    The alternative is not hypothetical — it is the command this file's own header
    and `scripts/automod/spec.py:138` document, and the nightly drift report
    prints: `cp ~/.config/qmd/index.yml agent-services/conf/qmd-index.yml`. The
    daemon's file is shorter than this one and carries no provenance and none of
    these comment blocks, so running it now would delete exactly the three
    structures pinned below: the header that names which file the daemon reads and
    which direction a re-sync runs, the `autonomy-runs` block explaining why only
    `run_*.md` is indexed, and the `sessions` block #2311 wrote to keep a stale
    count out of the corpus. They share one node because a file that lost them is
    not partly broken — it is a different file, and the last collection count or
    path a reader could restore from would be gone with it.

    The 14-collection assertion is the fourth corner: the `facts` collection went
    out of the live file silently in 2026-09-19 and out of here only on 2026-09-28
    (#1652), so the number of collections is the thing that catches a copy that
    lost one rather than one that lost its comments.
    """
    text = TEMPLATE.read_text()

    header = text.split("\ncollections:", 1)[0]
    assert "TRACKED REFERENCE COPY" in header, (
        "the header line that says what this file is is gone, which is the shape of "
        "the daemon's file copied over it")
    assert "~/.config/qmd/index.yml" in header, (
        "the header no longer names the file the daemon actually reads, so a reader "
        "has no way to know this one is not it")
    assert "cp ~/.config/qmd/index.yml agent-services/conf/qmd-index.yml" in header, (
        "the header lost the re-sync command and its direction — the sentence that "
        "makes a restore from this file safe")
    assert "#1298" in header, (
        "the header no longer cites the incident that made it say any of this, so it "
        "reads as decoration and gets dropped by the next hand edit")

    lines = text.splitlines()
    at = next(i for i, ln in enumerate(lines) if ln.rstrip() == "  autonomy-runs:")
    above = []
    for ln in reversed(lines[:at]):
        if not ln.lstrip().startswith("#"):
            break
        above.append(ln.strip())
    assert above, (
        "the `autonomy-runs:` key lost the comment block sitting above it — the "
        "~19 MB of run records it exists to explain")
    runs = "\n".join(reversed(above))
    assert "architecture/autonomy-system.md" in runs and "run_*.md" in runs, (
        f"the `autonomy-runs` block no longer says what it indexes and why ({runs!r})")

    assert "No count is stated here" in _sessions_comment_block(), (
        "the `sessions` block lost the sentence that keeps a live document count out "
        "of it — #2311's rule, and the first thing the daemon's file does not have")

    colls = yaml.safe_load(text)["collections"]
    assert len(colls) == 14, sorted(colls)


# --- #3160: every collection the memory route names is one this file declares --
#
# `app/routers/memory.py` kept its own seven-name segment list, and one of the
# seven — `agents` — is not a collection in this template at all: the string
# appears here only on `subliminal`'s `ignore:` list. qmd answers an undeclared
# collection with HTTP 200 and `{"results": []}` rather than an error (measured on
# the live daemon 2026-10-22), so the divergence spent a search leg and reported
# nothing wrong anywhere. `tests/test_qmd_query_shape.py` pins what the route asks
# for today; this is the check that stops *any* name the route names from being a
# name the daemon has never heard of — the same shape, forward-proof, and the one
# a hand-synced list cannot give.

def _undeclared(names, colls: dict) -> list[str]:
    """The names for which this template declares no collection of that name."""
    return [n for n in names if n not in colls]


def _memory_search_collections(monkeypatch, scope: str = "") -> list[str]:
    """The `collections` the Mission Control memory route puts on the wire.

    Read out of the payload the route hands `qmd_query`, not out of a module
    constant: a check aimed at a name the route no longer uses would pass on a
    route that had quietly gone back to its own list. The `len(seen) == 1` assert
    is the denominator — a route that asked qmd nothing yields an empty list, and
    an empty list is a subset of anything.
    """
    import asyncio

    from agent_mcp import vault
    from app.routers import memory as mem

    seen = []

    def fake_query(payload, **kw):
        seen.append(payload)
        return {"results": []}

    monkeypatch.setattr(vault, "qmd_query", fake_query)
    asyncio.run(mem.memory_search(q="collections drift", limit=3, scope=scope))
    assert len(seen) == 1, (
        "the memory route named no payload at all, so this check would have no numerator")
    return list(seen[0]["collections"])


def test_every_collection_the_memory_route_names_is_declared_by_the_template(monkeypatch):
    """#3160 clause 2: no name the memory search can send is a name qmd lacks.

    Three shapes of the route's own doing, because the unrestricted branch is the
    one Mission Control takes (`web/src/api.ts:1468` sends `q` and `limit` only)
    while the scoped branches are where a caller's typo would land: the default
    set, a scope naming one undeclared thing, and a scope naming a path inside a
    real collection.
    """
    colls = yaml.safe_load(TEMPLATE.read_text())["collections"]
    for scope in ("", "agents", "knowledge/youtube", "agents,notes"):
        named = _memory_search_collections(monkeypatch, scope=scope)
        assert _undeclared(named, colls) == [], (
            f"scope={scope!r}: the memory route would send {named}, and "
            f"{_undeclared(named, colls)} of that is not a collection this template "
            "declares — qmd answers such a name with 200 and no rows, so the search "
            "comes back empty with no signal anywhere (#3160)")

    # The check bites. `agents` is the name this incident was about, and this file
    # does not declare it: without this line an `_undeclared` that always returned
    # `[]` would make the loop above unconditionally green.
    assert _undeclared(["agents"], colls) == ["agents"], (
        "`_undeclared` no longer flags `agents`, the one name this template must never "
        "be asked for, so the check above proves nothing")
