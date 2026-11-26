"""SD-stranded-1's ruled-set record must say what is ruled, executed, and checkable.

`~/obsidian/skills/autonomy-data-pipeline/SKILL.md` Step 2.5 is the only place the
stranded-entity apply decision lives, and it is read by every task #24 run. Three things
were wrong in it at once, all measured on 2026-10-18/19 and all fixed by vault commit
`6efe5b94` for #2998:

* **The set-check could not see the ruling.** `RULED` held only `PS-1` (5 pairs) while the
  2026-10-18T223227Z dump printed 6 — the extra pair `False Absence Guard Pattern -> Guard`,
  which #2986's owed-check ruling of 2026-10-18T23:41:30Z had already accepted. The snippet
  printed `novel=['False Absence Guard Pattern -> Guard'] covered_by=none`, which by
  SKILL.md's own trigger obliges the run to FILE a carrier, every ~4h12m, forever. Four
  carriers went through in twelve hours on 2026-10-18 — #2970 at 12:48Z, #2986 at 22:32Z,
  #2989 at 23:41Z, #2998 at 00:16Z — and each closed inside an hour of being triaged
  confirmed, so nothing downstream ever read the ruling that had already been made.
* **The scope paragraph sent an executor to a dead end.** It said "this record does not pick
  one" and pointed at "an open owed-check ruling on #2853" — #2853 is `status: done`, and
  #2986 had already ruled #2829's 3-of-5 scope MOOT because `--apply` has no per-pair
  exclusion. An apply that needed a choice could not find that the choice was made.
* **Two required verification records the code cannot emit.** `link_stranded_entities.py`'s
  apply branch takes a backup, writes `proposed` in one transaction, prints
  `applied N edges; backup -> ...`, and writes the `--out` proposals JSONL. It emits no
  `mode: "apply"` artefact and no per-apply `stranded-applied-*.jsonl`, yet the skill told the
  executor to expect "the dry-run rows plus exactly one" apply row in
  `stranded-entities-*.json` and forbade hand-writing one — a verification that could be
  neither satisfied nor falsified.

Each node below reads the live vault (resolved as `Path.home() / "obsidian"`, never
`app.paths`, which re-anchors to a round worktree where no skills tree exists) and fails
loud naming the path when it is missing, after `NO_VAULT` in
`tests/test_yaml_fix_skill_claims.py:96`. The one node that measures the live kg store
anchors it on `app.data_root.production_data_root()` instead: the gate runs the suite under
the round's own HOME, where `~/lloyd-data` is an empty scaffold and a clean-looking answer
would be a measurement of nothing. Nothing here skips, and nothing here is marked: the
gate runner hardcodes `-m "not live_vault and not fault_injection"`
(`scripts/automod/gate.py:641`), so a marked node is deselected from the run meant to enforce
it and pins nothing. The nodes that exist to be able to fail are
`test_deleting_ps_2_reopens_the_sixth_carrier`, `test_the_stale_claim_detector_fires_on_the
_prose_it_replaces` and `test_the_marker_checker_refuses_a_state_the_store_contradicts`, which
run the checks against readings that are wrong — the pre-fix text in the first two cases, and
a RULED marker aged out of step with the store in the third.

Which brings the store into this file's scope, and it needs saying plainly: every other node
here is a read of one vault file and cannot go red on its own account. The three nodes added
by #3031 additionally read `kg.sqlite` under `PRODUCTION_DATA_ROOT`, because a marker's
execution state is a claim about that store and nothing else. They are the first nodes in this
file that can go red because the vault and the fact diverged rather than because a sentence
went missing — which is why all three name the store path in the message they refuse with, and
why `_live_name_embedding_pairs` asserts the file is there before opening it instead of
letting sqlite invent an empty one.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.data_root import PRODUCTION_DATA_ROOT

ROOT = Path(__file__).resolve().parent.parent
BACKLOG_PY = ROOT / "scripts" / "automod" / "backlog.py"
THIS_FILE = "tests/test_skill_stranded_ruled_sets.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.skill_folder import disk_skill_folder  # noqa: E402

# Never app.paths.VAULT_ROOT: a round's worktree has no vault.
VAULT = Path.home() / "obsidian"
SKILL_PATH = VAULT / "skills" / "autonomy-data-pipeline" / "SKILL.md"
COMMITTED_PS2 = VAULT / "backlog/data/stranded-proposals-2026-10-18T223227Z.jsonl"
# NOT Path.home(): the gate runs the suite under the round's own HOME
# (`lloyd-work/<SM_*>/home`), whose `lloyd-data` is an empty scaffold, and the node then
# reports "no live store" for a store that is sitting right there. production_data_root()
# ignores $HOME and LLOYD_DATA by design — the reader-side anchor from the
# `worktree-data-anchor` skill. The VAULT paths above deliberately stay on Path.home():
# the round home's `obsidian` entry is the real vault, and these are vault-reading nodes.
LIVE_STORE = PRODUCTION_DATA_ROOT / "_pipeline/vault-derived/kg.sqlite"

TS_PS2 = "2026-10-18T223227Z"
TS_PS1 = "2026-10-16T191722Z"

# The six pairs PS-2 must hold, transcribed from the 22:32Z dump the item names — this file's
# own copy of the truth, so the vault dict and the vault witness bytes are checked against a
# third source and not merely against each other.
PS_2_PAIRS = {("False Absence Guard Pattern", "Guard"),
              ("Failure Taxonomy Induction", "Failure Taxonomy"),
              ("Faros AI", "Faros"),
              ("Isaac-Lift-Soft-Franka", "Franka"),
              ("Perplexity Computer", "Perplexity"),
              ("giodl73-repo", "Repo")}
PS_1_PAIRS = {p for p in PS_2_PAIRS if p[0] != "False Absence Guard Pattern"}

# The witness bytes #2998 committed: 6 lines, 1,748 bytes, sha256 of the committed blob.
PS2_BYTES = 1748
PS2_SHA256 = "3c769602276750e57a31d2dd46edf76065c127f1169da6fc476a84bf2438fbf9"

STEP_HEADING = "## Step 2.5:"
HEADING_RE = re.compile(r"^## ", re.M)
FENCE_RE = re.compile(r"^[ \t]*```bash\n(.*?)\n[ \t]*```", re.S | re.M)
BULLET_RE = re.compile(r"^(?=[-*] )", re.M)
RULED_CITE_RE = re.compile(r"backlog\.py:(\d+)")
# The two lines of `scripts/automod/backlog.py` the Step 2.5 prose may legitimately name by
# number, each pinned here by an ANCHOR that has to occur exactly once rather than by a
# number written down: the constant's own assignment, and the line inside
# `_rederive_command` that builds the read-only-URI sqlite open whose command the witness
# bullet quotes. #3228: the guard used to admit ONE cite in the whole 820-line section and
# demand it be the constant's, so a prose edit that legitimately cited the emitter failed a
# test outside its own diff.
WITNESS_CONSTANT_ANCHOR = "WITNESS_MAX_BYTES = 1_048_576"
READONLY_URI_ANCHOR = "sqlite3.connect(f'file:"
READONLY_URI_MODE = "?mode=ro"
# The number this file's falsifier leg (a) re-points the section's cite at. It is the figure
# the guard was BUILT on — 28 lines off the constant — and it must not coincide with either
# derived line, which the node that uses it asserts before it substitutes.
UNDERIVED_CITE = "1240"
STORE_FIGURE_RE = re.compile(r"vault-derived/kg\.sqlite` is \*\*([\d,]+)\*\* bytes")
STAMP_RE = re.compile(r"read at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z)")

# Claims that an apply leaves a record of its own. Every pattern here was in the skill text
# #2998 replaced; none of them may survive in it.
STALE_APPLY_CLAIMS = {
    "the apply writes its own artefact": re.compile(
        r"writes its own[^\n]{0,80}(?:mode|artefact|artifact)", re.I),
    "the grep expects one apply row among the dry-run rows": re.compile(
        r"plus exactly one", re.I),
    "the verification list demands a new stranded-applied file": re.compile(
        r"one new[^\n]{0,40}stranded-applied", re.I),
    "the verification list demands a new backup only": re.compile(
        r"one new[^\n]{0,40}kg-stranded", re.I),
}

# The apply that wrote `PS-2`'s and `PS-3`'s edges: an attended by-hand `--apply` at this
# instant, recorded by #3029 on the vault side and named by those two keys' RULED markers on
# this side. It is a constant here rather than something scraped from the comment because the
# point of comparing them is that a marker which LOSES the date is a failure — a marker
# reading `EXECUTED` with no instant cannot be aged, re-checked, or attributed to the
# `store-backups/kg-stranded-20261019T164420Z.sqlite` it is stored beside.
APPLY_INSTANT = "2026-10-19T16:44:20Z"

# It is no longer the only apply: #3060's carrier was applied by hand 16h31m later, and that
# write is what `PS-4`'s marker has to name. So a ruled key's marker is weighed against THAT
# key's instant, held per key here — one constant for all keys could only ever describe the
# set applied first, and the only way to make `PS-4` agree with it would be to stamp it with
# the instant of a backup it is not stored in.
PS_4_APPLY_INSTANT = "2026-10-20T09:15:39Z"

# The third ruled apply: carrier #3704's four pairs (`PS-9`) went in by hand on 2026-11-19,
# and the write is on disk to be checked against — `store-backups/
# kg-stranded-20261119T010306Z.sqlite` exists and its `stat -c %Y` is 1795050187. The instant
# is taken from that artefact, never from the marker this file checks it against, for the
# reason on `APPLY_INSTANT` above: a date scraped off the claim cannot falsify the claim.
PS_9_APPLY_INSTANT = "2026-11-19T01:03:06Z"

# The fourth ruled apply, and the fourth KEY that had to be added rather than excused: carrier
# #4257's pair `agent-supervisord.service -> supervisord` went in by hand on 2026-11-19T15:46:35Z
# and its write is on disk to be checked against — `store-backups/kg-stranded-20261119T154635Z.sqlite`,
# whose `stat -c %Y` is 1795103195, the marker's instant to the second. This pair used to be this
# file's own counter-example, and the wrong one: the paragraph that sat directly above
# `PS_9_APPLY_INSTANT` — deleted by #4519, and pinned absent by
# `test_ps_10_is_a_ruled_key_whose_pair_the_set_check_covers_by_name` — declared that a key the
# skill did not carry could not be weighed, that adding one here would authorise a decision no
# human had made, and that the finding lived on #4480. All three readings were wrong in the
# direction that costs something. `RULED` is what `covered_by=` is computed against, so while the
# skill's literal stopped at `PS-9` a dry run answered
# `novel=['agent-supervisord.service -> supervisord'] covered_by=none` for a pair whose edge was
# LIVE: the filing branch was one edge-expiry away from opening another carrier for a decision that
# had been ruled, applied and written. The ruling that covers it is SD-stranded-1 (2026-10-16) plus
# #2986's owed-check ruling of 2026-10-18T23:41:30Z, whose operative scope is "EVERY pair a fresh
# dry run prints", and #4257 held this pair as its pending `hand_run:`; `PS-9` is the same shape — a
# carrier whose apply ran, with no separate ruling. #4519 is the round that put the key in the skill,
# this instant in this map, and the pair in both. The finding's home is #4519, not #4480: #4480 is
# `status: done`, as is #4257, and a closed item is not an owner.
#
# Which makes it the rule rather than a one-off: a by-hand apply that ran and whose edges are live
# takes a `RULED` key in the skill AND an entry in `APPLY_INSTANTS` in the same pass — `PS-11` and
# `PS-12`, below, are the first pair of keys added under it. A settled set the record cannot see stays
# invisible until one of its edges expires, and edges expire rather than delete.
PS_10_APPLY_INSTANT = "2026-11-19T15:46:35Z"

#: PS-10's whole set, from carrier #4257 — the pair the three runs of 2026-11-19 between `PS-9`'s
#: apply and this one printed as `proposed edges 1`. Like `PS_9_WRITTEN_PAIRS`, a closed historical
#: fact about one write, which is why it may be written down where a store count may not.
PS_10_RULING_PAIR = frozenset({("agent-supervisord.service", "supervisord")})

#: Derived from the instant, never spelled out — the same rule as `PS_9_BACKUP`.
PS_10_BACKUP = f"kg-stranded-{PS_10_APPLY_INSTANT.replace('-', '').replace(':', '')}.sqlite"

# The fifth ruled apply, and the rule above applied to two keys at once: carrier #4560's set `PS-12`
# (`Dual pulley servo wheel -> Servo`, `Kimchi by Cast AI -> Cast AI`) went in by an attended by-hand
# `--apply` at 2026-11-26T17:35:28Z, and its subset `PS-11` — carrier #4503, the Kimchi pair alone,
# filed three days earlier — was written by that same apply, never on its own. So both keys carry this
# one instant, as `PS-2` and `PS-3` share `APPLY_INSTANT`, and `PS-11` ⊂ `PS-12` is the shape
# `PS-1` ⊂ `PS-2` ⊂ `PS-3` already has. The witness is on disk:
# `store-backups/kg-stranded-20261126T173528Z.sqlite`, `stat -c %Y` 1795714528, the instant to the
# second; the receipt read `ACTIVE_NAME_EMBEDDING=28 PAIRS_LIVE=2/2 MISSING=0`, exit 0; and the vault
# carries the before/after proposals at commit `54246a2ad`. Both carriers are `status: done`.
PS_12_APPLY_INSTANT = "2026-11-26T17:35:28Z"

#: PS-12's whole set, from carrier #4560, and PS-11's, from carrier #4503 — closed facts about one
#: write, like `PS_10_RULING_PAIR`.
PS_12_WRITTEN_PAIRS = frozenset({("Dual pulley servo wheel", "Servo"),
                                 ("Kimchi by Cast AI", "Cast AI")})
PS_11_RULING_PAIR = frozenset({("Kimchi by Cast AI", "Cast AI")})

#: Derived from the instant, never spelled out — the same rule as `PS_9_BACKUP`.
PS_12_BACKUP = f"kg-stranded-{PS_12_APPLY_INSTANT.replace('-', '').replace(':', '')}.sqlite"

#: The four pairs that apply wrote, from carrier #3704. A closed historical fact about one write,
#: unlike a store count that moves with the next apply, so it is legitimate to write down: the
#: node that reads it holds `RULED` to the set the apply is recorded as having made.
PS_9_WRITTEN_PAIRS = frozenset({
    ("Browser UI Test Flakiness", "Test Flakiness"),
    ("Capture-recapture population estimation", "Capture-recapture"),
    ("LiteLLM Router", "LiteLLM"),
    ("write tool", "Write"),
})

#: The pre-apply backup, named by deriving its name from the apply instant rather than spelling
#: it out — the same instant the marker declares, so the two cannot drift apart unnoticed.
PS_9_BACKUP = f"kg-stranded-{PS_9_APPLY_INSTANT.replace('-', '').replace(':', '')}.sqlite"

#: Every ruled key except `PS-1` (excused: its status rides on `PS-2`'s receipt sentence) and
#: the apply instant its marker must declare. A key missing from this map fails the node that
#: reads it and names itself, rather than being skipped or borrowing a date. A NEW `RULED` key
#: in the skill therefore reds that node until its own apply's artefact is named here.
APPLY_INSTANTS = {"PS-2": APPLY_INSTANT, "PS-3": APPLY_INSTANT,
                  "PS-4": PS_4_APPLY_INSTANT, "PS-9": PS_9_APPLY_INSTANT,
                  "PS-10": PS_10_APPLY_INSTANT,
                  "PS-11": PS_12_APPLY_INSTANT, "PS-12": PS_12_APPLY_INSTANT}


def _missing(path: Path) -> str:
    """Fail-loud, after NO_VAULT in tests/test_yaml_fix_skill_claims.py:96."""
    return (f"no vault file at {path}: these nodes police that file, so they report "
            "rather than skip")


def _skill_text() -> str:
    """The skill's text as a FOLDER: `SKILL.md` spliced with every sibling beside it.

    #4307 folded this skill to 9,968 chars, moving each step's text into a named sibling
    under its heading, so the Step 2.5 section these 34 nodes grade — its Purpose
    paragraph, its `SD-STRANDED SETCHECK` heredoc, its ruled-set bullets — is in
    `step2-5-denominators.md` while `SKILL.md` keeps the heading and a pointer. Reading
    one file reported that as prose this skill no longer carries (the 26 nodes here, item #4496), which is the #4190 misreading `tests/skill_folder.py` exists to prevent and
    this module had not been routed through. Splicing is by heading title, so the span
    under `## Step 2.5:` holds main's own text plus the sibling's under that heading and
    nothing else — the scoping that stops a node here pinning somebody else's paragraph
    still bites, and a heading that is genuinely gone still extracts nothing.
    """
    assert SKILL_PATH.is_file(), _missing(SKILL_PATH)
    return disk_skill_folder(SKILL_PATH.parent, encoding="utf-8")[1]


def _step_2_5_section(text: str) -> str:
    """From the `## Step 2.5:` heading to the next `## ` heading."""
    starts = [m.start() for m in HEADING_RE.finditer(text)]
    heading = [s for s in starts if text[s:].startswith(STEP_HEADING)]
    assert len(heading) == 1, (
        f"expected exactly one {STEP_HEADING} section in {SKILL_PATH}, found "
        f"{len(heading)} — the block these nodes pin has moved or duplicated, and an "
        "empty extraction would satisfy them by saying nothing")
    start = heading[0]
    following = [s for s in starts if s > start]
    return text[start:following[0]] if following else text[start:]


def _paragraphs(section: str) -> list[str]:
    """Prose units split on blank lines — keeps fenced blocks inside their own unit."""
    return [p for p in section.split("\n\n") if p.strip()]


def _step_2_5_purpose(text: str) -> str:
    """The Step 2.5 `**Purpose:**` paragraph, which is the site that states today's count.

    Isolated rather than graded over the whole section, because clause 3 names THIS paragraph: a
    `now returns **N**` stated anywhere else in Step 2.5 must not satisfy it, and the paragraph
    is also where the applies that wrote the count have to be named.
    """
    units = [p for p in _paragraphs(_step_2_5_section(text))
             if p.lstrip().startswith("**Purpose:**")]
    assert len(units) == 1, (
        f"expected exactly one `**Purpose:**` paragraph in the Step 2.5 section of {SKILL_PATH}, "
        f"found {len(units)} — the paragraph clause 3 grades has moved or merged, and an empty "
        "extraction would satisfy that clause by saying nothing")
    return units[0]


def _bullets(section: str) -> list[str]:
    """Bullet-aligned units: a markdown bullet keeps its wrapped lines and its fenced blocks.

    `_paragraphs` alone cannot find a bullet, because a bullet with a code sample in it is
    split into three blank-line units and the surrounding prose merges across bullets.
    """
    return [b for b in BULLET_RE.split(section) if b.strip()]


def _one(items: list[str], what: str, found: int | None = None) -> str:
    assert len(items) == 1, (
        f"expected exactly one {what} in the Step 2.5 section of {SKILL_PATH}, found "
        f"{found if found is not None else len(items)} — the prose this file pins has moved "
        "or duplicated, and silently taking the first would let a stale copy pass")
    return items[0]


def _setcheck_script(section: str) -> str:
    """The Python the Step 2.5 set-check actually runs, out of its bash heredoc."""
    blocks = [b for b in FENCE_RE.findall(section) if "SD-STRANDED SETCHECK" in b]
    assert len(blocks) == 1, (
        f"expected exactly one fenced block printing SD-STRANDED SETCHECK in the "
        f"Step 2.5 section of {SKILL_PATH}, found {len(blocks)} — this file runs that "
        "snippet, so it must be findable and unique")
    body = blocks[0]
    assert "<<'EOF'" in body, f"the set-check block at {SKILL_PATH} has no heredoc body"
    script = body.split("<<'EOF'\n", 1)[1].rsplit("\nEOF", 1)[0]
    assert "RULED = {" in script and "covered_by" in script, (
        "the extracted block is not the set-check: no RULED dict or covered_by print")
    return script


def _ruled_dict(script: str) -> dict[str, set[tuple[str, str]]]:
    """The RULED literal, parsed rather than grepped, so a typo cannot hide in a string."""
    for node in ast.walk(ast.parse(script)):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "RULED" for t in node.targets):
            return {k: {tuple(p) for p in v}
                    for k, v in ast.literal_eval(node.value).items()}
    raise AssertionError("the set-check snippet no longer assigns RULED as a literal")


def _pairs_in(text: str) -> set[tuple[str, str]]:
    out = set()
    for line in text.splitlines():
        if line.strip():
            e = json.loads(line)
            out.add((e["source"], e["target"]))
    return out


def _committed_ps2_bytes() -> bytes:
    assert COMMITTED_PS2.is_file(), _missing(COMMITTED_PS2)
    return COMMITTED_PS2.read_bytes()


def _run_setcheck(script: str, ts: str, payload: str, tmp_path: Path) -> str:
    """Run the snippet as the run does: a subprocess reading its proposals file from HOME."""
    home = tmp_path / f"home-{ts}"
    stage = home / "lloyd-data/_pipeline"
    stage.mkdir(parents=True, exist_ok=True)
    (stage / f"stranded-proposals-{ts}.jsonl").write_text(payload, encoding="utf-8")
    env = {"HOME": str(home), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    r = subprocess.run([sys.executable, "-", ts], input=script, cwd=str(home),
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, (
        f"the set-check snippet exited {r.returncode} on {ts}:\n{r.stderr[-600:]}")
    return r.stdout.strip()


def _scope_paragraph(section: str) -> str:
    """The paragraph that answers which scope is operative."""
    # A bullet that is prose, not any unit naming the ruling: the RULED comment inside the
    # set-check's heredoc cites the same ruling with the same words, and quoting from THAT
    # would grade a code comment instead of the scope answer an executor reads.
    hits = [b for b in _bullets(section) if b.startswith("- ")
            and "SD-STRANDED SETCHECK" not in b
            and "2026-10-18T23:41" in b and "MOOT" in b.upper()]
    return _one(hits, "bullet giving the operative scope (2026-10-18T23:41 + the moot ruling)")


def _witness_paragraph(section: str) -> str:
    """The bullet that cites WITNESS_MAX_BYTES against the live store's byte size."""
    hits = [b for b in _bullets(section)
            if "WITNESS_MAX_BYTES" in b and "vault-derived/kg.sqlite" in b]
    return _one(hits, "bullet citing WITNESS_MAX_BYTES beside the live store path")


def _stale_apply_claims(text: str) -> list[str]:
    """Which 'the apply leaves its own record' claims survive in `text`."""
    return [label for label, rx in STALE_APPLY_CLAIMS.items() if rx.search(text)]


# ── a RULED key's execution marker, weighed against the live store ───────────
#
# `RULED` records what has been RULED, never what has been WRITTEN, and a key stays in it
# after its apply — so the comment beside a key is the ONLY place a reader of the set-check
# learns whether that set's edges exist. #3029 re-cut PS-2's and PS-3's markers from
# `NOT EXECUTED` to `EXECUTED` when the attended by-hand `--apply` ran, and the reading that
# marker gives has to be weighed against the thing it describes rather than against more
# prose: a stale `pending` invites a second apply of edges that are already live, and a
# hollow `executed` is the false green #854 was filed for. Hence one function that reads the
# marker and the store together, called by the clause nodes and by the can-fail node alike.

def _live_name_embedding_pairs() -> tuple[int, set[tuple[str, str]]]:
    """Active `name-embedding:` edges in the live store: `(count, {(source, target)})`.

    The same read the skill's own `receipt:` probe makes, through `app.kg_store` — whose
    constructor is the read-only URI — anchored on `PRODUCTION_DATA_ROOT` and not
    `Path.home()`, for the reason spelled out at `LIVE_STORE` above: the gate runs the suite
    under the round's scaffolded HOME, where a clean-looking answer would be a measurement of
    nothing. The count comes back with the pairs because the skill's receipt condition is a
    floor on it (the pilot's 12 edges plus `PS-3`'s 7 = 19)."""
    from app.data_root import KG_DB_RELATIVE
    from app.kg_store import KGStore
    assert LIVE_STORE.is_file(), (
        f"no live store at {LIVE_STORE} (the production data root, not the round's) to weigh "
        "a RULED execution marker against")
    rows = [(r["source"], r["target"]) for r in
            KGStore(PRODUCTION_DATA_ROOT / KG_DB_RELATIVE).edges.all_rows(
                include_expired=False)
            if str(r["evidence"] or "").startswith("name-embedding:")]
    return len(rows), set(rows)


def _ruled_comment(section: str, key: str) -> str:
    """The comment a RULED key carries: from its own `# <key>` line up to the set it opens.

    Cut by position, not by line number, so re-wrapping the comment in the vault cannot make
    this read the neighbouring key's text — which is the failure mode that would have an
    execution-marker assertion grading the wrong set."""
    cut = section.split(f'"{key}"')[0].rsplit(f"# {key}", 1)
    assert len(cut) == 2, f"{key}'s own comment line is gone from the RULED block"
    return cut[1]


def _declared_execution(mark: str, key: str) -> tuple[str, str]:
    """What a marker claims: `("executed", instant)` or `("pending", "")` — or it refuses.

    `NOT EXECUTED` contains `EXECUTED`, so the pending reading is taken first and a comment
    carrying both is refused rather than resolved by precedence: that ambiguity is the thing
    #3029 removed, and a checker that silently picked a side would let it come back. The
    instant is required *beside* the word, on the same line, because PS-2's comment also
    carries its ruling instant `2026-10-18T23:41:30Z` three lines lower — accepting any
    instant in the comment would let a pending set read as applied on a date that only names
    the ruling."""
    up = mark.upper()
    pending = "NOT EXECUTED" in up
    executed = "EXECUTED" in up.replace("NOT EXECUTED", "")
    if pending and executed:
        raise AssertionError(
            f"{key}'s RULED marker declares both states: it carries `NOT EXECUTED` and a "
            f"second `EXECUTED`: {mark[:160]!r}")
    if pending:
        return "pending", ""
    if not executed:
        raise AssertionError(
            f"{key}'s RULED marker declares no execution state: {mark[:160]!r}. A key whose "
            "apply is neither claimed nor disclaimed cannot be weighed against the store at "
            "all, which is the sentence PS-1 sat without.")
    hit = re.search(r"EXECUTED[^\n]{0,80}?(20\d\d-\d\d-\d\dT\d\d:\d\d:\d\dZ)", up)
    if not hit:
        raise AssertionError(
            f"{key}'s RULED marker says EXECUTED without an instant beside it: "
            f"{mark[:160]!r} — an applied set with no date is indistinguishable from one "
            "whose apply is still owed.")
    return "executed", hit.group(1)


def _execution_claim(section: str, key: str,
                     live: set[tuple[str, str]]) -> tuple[str, str]:
    """The state `key`'s marker declares, after the live store has been asked about it.

    Refuses three ways: a marker declaring neither state or both (inside
    ``_declared_execution``), an `EXECUTED` with no instant beside it, and any marker the
    store contradicts in either direction — `executed` while one of that set's own pairs is
    not live at exact `source`+`target` under `expired_at IS NULL`, or `pending` while every
    pair of it is. Both directions are load-bearing here: `PS-3` ⊇ `PS-2` ⊇ `PS-1`, so one
    apply makes three sets live and a marker left stale on either of the other two is a
    reader being told to re-apply edges the store already holds.
    """
    mark = _ruled_comment(section, key)
    state, instant = _declared_execution(mark, key)
    pairs = _ruled_dict(_setcheck_script(section))[key]
    missing = sorted(f"{a} -> {b}" for a, b in pairs if (a, b) not in live)
    if state == "executed" and missing:
        raise AssertionError(
            f"{key} declares itself EXECUTED at {instant}, but the live store does not hold "
            f"{missing}: the marker is ahead of the write. The `receipt:` probe in "
            "SD-stranded-1 is the command that settles it.")
    if state == "pending" and not missing:
        raise AssertionError(
            f"{key} declares its apply still owed, but all {len(pairs)} of its pairs are "
            "live in the store: the marker is behind the write, and a run that believes it "
            "is being told to apply edges that already exist.")
    return state, instant


# ---------------------------------------------------------------- clause 1: PS-2 in RULED

def test_ruled_holds_ps_2_with_the_six_pairs_the_dump_printed():
    """`RULED` gained PS-2 with the six pairs the dump printed, and its marker now has to
    declare the apply that #3029 recorded as run.

    "`RULED` means RULED — not executed" stayed true of the *key* and stopped being true of
    the *apply* on 2026-10-19T16:44:20Z, when the attended by-hand `--apply` wrote PS-3 ⊇
    PS-2 ⊇ PS-1. So this node no longer demands `NOT EXECUTED` — it demands that the marker
    state agree with the live store in whichever direction it points, which is the property
    that survives the apply. PS-1 staying ruled beside PS-2 is unchanged, and so is the
    requirement that PS-2's comment carry its 2026-10-18T23:41 ruling instant."""
    section = _step_2_5_section(_skill_text())
    ruled = _ruled_dict(_setcheck_script(section))
    assert "PS-2" in ruled, (
        f"PS-2 is absent from RULED in {SKILL_PATH}: a run whose set contains the pair "
        "#2986 already accepted prints novel and is obliged to file a carrier")
    assert ruled["PS-2"] == PS_2_PAIRS, (
        f"PS-2 holds {sorted(ruled['PS-2'])}, expected the six pairs "
        f"{sorted(PS_2_PAIRS)} the {TS_PS2} dump printed")
    assert "PS-1" in ruled and ruled["PS-1"] == PS_1_PAIRS, (
        "PS-1 must stay ruled alongside PS-2: the nine runs that printed the five-pair "
        "set printed a set that is still ruled, and deleting it would call them novel")
    # The comment must still carry the ruling instant, and it must declare which side of the
    # apply the set is on — PS-2 was `NOT EXECUTED` from 2026-10-18T23:41:30Z until the
    # attended by-hand `--apply` of 2026-10-19T16:44:20Z, which wrote PS-3 and so wrote all
    # six of these pairs with it. Demanding the old wording would be demanding a marker that
    # tells a reader to re-apply edges the store already holds.
    mark = _ruled_comment(section, "PS-2")
    assert "2026-10-18T23:41" in mark, (
        f"PS-2's RULED comment says {mark[:120]!r}: it must still carry the ruling instant "
        "2026-10-18T23:41, whichever state it declares")
    state, instant = _execution_claim(section, "PS-2", _live_name_embedding_pairs()[1])
    assert state == "executed" and instant == APPLY_INSTANT, (
        f"PS-2's RULED marker declares {state}" + (f" at {instant}" if instant else "") +
        f"; the apply that wrote it is {APPLY_INSTANT}, and the live store holds all six of "
        "PS-2's pairs — a marker reading `NOT EXECUTED` here is a run being told to write "
        "edges that already exist")


def test_the_ps_2_witness_bytes_are_committed_in_the_vault_tree():
    """The ruling's bytes travel with it: committed, under the ceiling, and equal to the dict."""
    assert COMMITTED_PS2.is_file(), _missing(COMMITTED_PS2)
    blob = COMMITTED_PS2.read_bytes()
    assert len(blob) == PS2_BYTES, (
        f"{COMMITTED_PS2} is {len(blob)} bytes, expected {PS2_BYTES} as committed by "
        "#2998 — a witness named for one dump must keep that dump's bytes")
    assert hashlib.sha256(blob).hexdigest() == PS2_SHA256, (
        f"{COMMITTED_PS2} is no longer the bytes #2998 committed "
        f"(sha256 {hashlib.sha256(blob).hexdigest()[:8]}… expected "
        f"{PS2_SHA256[:8]}…)")
    ceiling = re.search(r"WITNESS_MAX_BYTES = ([\d_]+)",
                        BACKLOG_PY.read_text(encoding="utf-8"))
    assert ceiling, f"WITNESS_MAX_BYTES is not defined where this node expects it: {BACKLOG_PY}"
    limit = int(ceiling.group(1).replace("_", ""))
    assert len(blob) <= limit, (
        f"the witness is {len(blob)} bytes over WITNESS_MAX_BYTES = {limit}")
    assert _pairs_in(blob.decode()) == PS_2_PAIRS, (
        f"the committed witness at {COMMITTED_PS2} no longer holds the six pairs PS-2 claims")
    ruled = _ruled_dict(_setcheck_script(_step_2_5_section(_skill_text())))
    assert ruled["PS-2"] == _pairs_in(blob.decode()), (
        "PS-2's dict and its committed witness disagree, so a run reading one cannot "
        "check the other")


def test_the_setcheck_covers_the_22_32z_set_as_ps_2(tmp_path):
    """The acceptance check: the snippet over that dump prints no novel pair, covered by PS-2.

    The snippet is run the way the run runs it — a subprocess reading
    `~/lloyd-data/_pipeline/stranded-proposals-$TS.jsonl` — against the vault-committed
    witness bytes placed at that path under a temporary HOME.
    """
    script = _setcheck_script(_step_2_5_section(_skill_text()))
    out = _run_setcheck(script, TS_PS2, _committed_ps2_bytes().decode(), tmp_path)
    assert out == (f"SD-STRANDED SETCHECK {TS_PS2}: pairs=6 novel=none "
                   "covered_by=['PS-2', 'PS-3']"), (
        f"the set-check reads {out!r} over the witness #2998 committed; the filing trigger "
        "at SKILL.md's novel=/covered_by= bullets turns anything naming a novel pair into "
        "another carrier item, which is the loop that produced #2970 #2986 #2989 #2998. "
        "PS-3 (#3017) is PS-2 plus `OCR-Assisted ASR -> ASR`, so from this round's change "
        "onward a six-pair run names TWO covering sets; the ledger field still names the "
        "smallest, which is PS-2.")


def test_deleting_the_sets_that_cover_the_sixth_pair_reopens_that_carrier(tmp_path):
    """The node above can fail: with every set covering the sixth pair stripped, the same
    bytes print the novel pair.

    This is the pre-fix reading, measured 2026-10-19T02:48Z before the edit:
    `pairs=6 novel=['False Absence Guard Pattern -> Guard'] covered_by=none`. Without this
    node, a RULED dict quietly emptied of the sets that cover it — or a snippet that never
    reads the file — could keep the node above green. **What #3017 changed about this node:**
    it used to strip `PS-2` alone, and PS-3 (`PS-2` plus `OCR-Assisted ASR -> ASR`, added to
    `RULED` for #3017) is a superset of `PS-2`, so stripping `PS-2` alone now leaves the six
    pairs covered and the node would measure PS-3 instead of the mechanism. What is stripped
    here is therefore every ruled set that contains the sixth pair — `PS-2` and `PS-3` — which
    is the same hole the node was written to probe.
    """
    script = _setcheck_script(_step_2_5_section(_skill_text()))
    ruled = _ruled_dict(script)
    sixth = ("False Absence Guard Pattern", "Guard")
    covering = {k for k, v in ruled.items() if sixth in set(v)}
    assert covering == {"PS-2", "PS-3"}, (
        f"only {sorted(covering)} cover {sixth[0]} -> {sixth[1]}; this node strips the sets "
        "that cover it, so a new covering set has to be named here rather than ignored")
    stripped = {k: v for k, v in ruled.items() if k not in covering}
    head, sep, tail = script.partition("RULED = {")
    assert sep, "the snippet no longer assigns RULED as a literal"
    _first, sep2, rest = tail.partition("\nnovel = ")
    assert sep2, "the RULED literal is not followed by the novel= computation"
    mutated = head + "RULED = " + repr({k: set(v) for k, v in stripped.items()}) + sep2 + rest
    out = _run_setcheck(mutated, TS_PS2, _committed_ps2_bytes().decode(), tmp_path)
    assert "novel=['False Absence Guard Pattern -> Guard']" in out and "covered_by=none" in out, (
        f"with PS-2 removed the set-check reads {out!r}, not the pre-fix novel reading — "
        "the coverage these nodes celebrate is then coming from somewhere other than PS-2")


def test_a_five_pair_run_is_still_covered_and_names_the_smallest_set(tmp_path):
    """PS-2 is a superset, so the older set must stay covered and the ledger stay one-valued."""
    script = _setcheck_script(_step_2_5_section(_skill_text()))
    committed_ps1 = VAULT / "backlog/data/stranded-proposals-2026-10-16T191722Z.jsonl"
    assert committed_ps1.is_file(), _missing(committed_ps1)
    out = _run_setcheck(script, TS_PS1, committed_ps1.read_text(encoding="utf-8"), tmp_path)
    assert "novel=none" in out and "covered_by=['PS-1', 'PS-2', 'PS-3']" in out, (
        f"a run reprinting PS-1 now reads {out!r}: all three sets cover it in dict order " \
        "(PS-3 ⊇ PS-2 ⊇ PS-1 as of #3017), and "
        "the skill's receipt rule says the ledger names the SMALLEST covering set or a "
        "later reader cannot tell which decision licensed the silence")
    assert "smallest" in _step_2_5_section(_skill_text()).lower(), (
        "the skill prints two covering set ids but never says which one to write into the "
        "ledger field")


# ---------------------------------------------------------- clause 2: the scope is settled

def test_the_scope_paragraph_records_a_settled_ruling(tmp_path):
    """No 'this record does not pick one', no executor sent to a closed item's open ruling."""
    section = _step_2_5_section(_skill_text())
    assert "does not pick one" not in section, (
        f"{SKILL_PATH} still tells an executor the record does not choose a scope")
    assert "open owed-check ruling on #2853" not in section, (
        "the paragraph sends an executor to wait on a #2853 ruling again; #2853 is "
        "status: done, so the wait never ends")
    para = _scope_paragraph(section)
    assert "no per-pair exclusion" in para, (
        f"the scope paragraph does not say WHY the narrower ruling is moot: {para[:200]!r}")
    assert "False Absence Guard Pattern" in para, (
        "the scope answer must name the pair the ruling included, since that is the pair "
        "a run will otherwise call novel")
    assert "every pair" in para.lower(), (
        "the scope paragraph does not state the operative scope in one clause")
    flat = " ".join(para.split()).lower()   # markdown wraps mid-phrase; compare unwrapped
    for bar in ("never a round", "never an autonomy task"):
        assert bar in flat, (
            f"settling the scope must not move the class of session: {bar!r} is gone from "
            "the paragraph that settles it")


def test_every_quoted_phrase_in_the_scope_paragraph_is_verbatim_from_an_item():
    """The scope answer quotes four rulings; each quote must exist in the corpus it cites.

    A paraphrase wearing quote marks is how a fabricated scope survives review: #2829's
    3-of-5 list was already unrecoverable from disk, so a quote nobody can re-read is worse
    than no quote. Whitespace is collapsed on both sides because the item files wrap
    mid-sentence.
    """
    section = _step_2_5_section(_skill_text())
    para = _scope_paragraph(section)
    quotes = [q for q in re.findall(r'"([^"]{20,})"', para)]
    assert len(quotes) >= 4, (
        f"the scope paragraph carries {len(quotes)} quoted rulings, expected at least 4 "
        "(the moot ruling, the operative scope, the accepted-pairs bound, the reasoning for "
        f"including the novel pair): {para[:240]!r}")
    corpus = ""
    files = 0
    for item in ("2765", "2829", "2853", "2986", "2989", "2998"):
        for f in sorted((VAULT / "backlog").glob(f"{item}-*.md")):
            files += 1
            corpus += " " + " ".join(f.read_text(encoding="utf-8",
                                                 errors="replace").split())
    assert files >= 5, (
        f"only {files} of the six item files this paragraph cites are present under "
        f"{VAULT / 'backlog'} — an empty corpus would make every quote 'found'")
    for q in quotes:
        needle = " ".join(q.split())
        assert needle in corpus, (
            f"quoted phrase is not verbatim in any cited item file: {q!r} — fix the quote "
            "in the skill or attribute it to the ruling that actually says it")


# ------------------------------------------- clause 3: no artefact an apply cannot produce

def test_no_surviving_claim_that_an_apply_leaves_a_record_of_its_own():
    """Neither the artefact JSON nor a per-apply JSONL: the skill must stop asking for them."""
    section = _step_2_5_section(_skill_text())
    stale = _stale_apply_claims(section)
    assert not stale, (
        f"{SKILL_PATH} still asserts: {stale}. link_stranded_entities.py's apply branch "
        "takes the backup, writes `proposed` in one transaction, prints the applied line, "
        "and writes --out; it emits no artefact JSON, so demanding one leaves the executor "
        "with a check it can neither satisfy nor falsify")
    # "four readings" alone is not enough to find it: :649 in the same section uses that
    # phrase for the fact_holding_entities denominators.
    hits = [p for p in _bullets(section) if "12 + N" in p and "MTIME" in p]
    para = _one(hits, "bullet stating the readings an apply actually leaves (12 + N + MTIME)")
    for needed in ("applied N edges", "--out", "MTIME", "12 + N"):
        assert needed in para, (
            f"the apply-verification paragraph omits {needed!r}: the record is the printed "
            "line, the --out proposals JSONL, the backup matched by MTIME, and the store count")
    assert "--sample 0" in para and "fals" in para, (
        "--sample 0 prints no pair list (`if proposed and args.sample:`, 0 is falsy), so the "
        "bullet must say N and the pairs come from --out and not from stdout")


def test_the_stale_claim_detector_fires_on_the_prose_it_replaces():
    """The detector above is not vacuous: the two paragraphs #2998 deleted trip all four."""
    old = (
        "- **How an apply of this decision is verified:** active `name-embedding:` = 12 + N, "
        "one new `stranded-applied-*.jsonl`, one new `kg-stranded-<ts>.sqlite`, and the "
        "next dry run printing `proposed edges` 0. The applying run writes its own "
        "`mode: \"apply\"` artefact, never a hand-made one.\n\n"
        "**What an apply writes, and where.** The applying run writes its own artefact "
        "under the same per-run pattern as a dry run, and the check is the dry-run rows "
        "plus exactly one `mode: \"apply\"`.\n")
    stale = _stale_apply_claims(old)
    assert len(stale) == 4, (
        f"the detector found only {stale} in text that contains all four stale claims — a "
        "detector that cannot fire on the prose it was written for pins nothing")


def test_a_proposals_jsonl_is_not_an_apply_receipt():
    """What the retired demand confused: both witnesses are proposals, with no mode field."""
    keys = None
    for ts in (TS_PS1, TS_PS2):
        f = VAULT / f"backlog/data/stranded-proposals-{ts}.jsonl"
        assert f.is_file(), _missing(f)
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert rows, f"{f} holds no rows"
        assert "mode" not in rows[0], (
            f"{f} gained a mode field: a proposals JSONL is the accepted pair set, and the "
            "skill's verification depends on it not being an apply receipt")
        assert keys in (None, set(rows[0])), f"{f} changed shape from the PS-1 witness"
        keys = set(rows[0])
    assert keys == {"source", "target", "type", "confidence", "provenance", "origin",
                    "created_at", "evidence"}, f"unexpected proposal keys {keys}"


# --------------------------------------------------------- clause 4: the two dead cites

def test_the_live_store_figure_is_a_measurement_not_a_backup_listing():
    """The byte count is `wc -c` on the live path with its instant, and the old one is named."""
    section = _step_2_5_section(_skill_text())
    para = _witness_paragraph(section)
    figure = STORE_FIGURE_RE.search(" ".join(para.split()))
    assert figure, (
        "the witness bullet no longer states the live store size as "
        "'`vault-derived/kg.sqlite` is **N** bytes', so nothing here checks the number")
    stated = int(figure.group(1).replace(",", ""))
    assert stated != 153_206_784, (
        "the live store figure is back to 153,206,784, which was never the live path — it is "
        "the size of the Oct-16 nightly backup store-backups/kg-v4-20261016T193937Z.sqlite")
    assert STAMP_RE.search(para), (
        "the store figure lost its measurement instant: this store grows on every nightly "
        "write, which is how the last figure went stale")
    assert "wc -c" in para, "the store figure must name the command that produced it"
    assert LIVE_STORE.is_file(), (
        f"no live store at {LIVE_STORE} (the production data root, not the round's) to "
        "check the cited figure against")
    live = LIVE_STORE.stat().st_size
    assert stated <= live, (
        f"the skill cites {stated:,} bytes but the live store now reads {live:,}: the "
        "figure is stale, or the store was swapped — re-measure with wc -c and re-stamp it")
    assert stated > 153_206_784, (
        f"{stated:,} is below the Oct-16 backup's 153,206,784: either the store shrank "
        "(a rebuild swap) and the cite needs re-measuring, or a backup size moved back in")
    old = section.find("153,206,784")
    assert old != -1, (
        "the superseded figure was deleted outright: its provenance — that it is the Oct-16 "
        "nightly backup's size, never the live path — is the part worth keeping")
    assert "kg-v4-20261016T193937Z.sqlite" in section[old:old + 400], (
        "the superseded figure is no longer tied to the file it was actually measured on")


def _backlog_lines() -> list[str]:
    """`scripts/automod/backlog.py` as lines, the tree every derived cite resolves against."""
    return BACKLOG_PY.read_text(encoding="utf-8").splitlines()


def _derived_backlog_line(lines: list[str], what: str, *anchors: str) -> str:
    """The one line of `backlog.py` carrying every string in `anchors`, as a cite.

    DERIVED rather than written down, so a round that inserts module-level lines above the
    constant or above the emitter moves the number the assertion expects instead of failing
    a prose edit outside its own diff (#3228). Deriving costs nothing in strength: the
    anchor must sit on EXACTLY one line — a constant that was renamed or deleted, or an
    emitter whose read-only open doubled, names itself as the failure rather than going
    green — and the number the prose carries must still EQUAL the derived line, so a stale
    number cannot pass on being a plausible line number.
    """
    hits = [i + 1 for i, l in enumerate(lines) if all(a in l for a in anchors)]
    assert hits, (
        f"no line of {BACKLOG_PY} carries {anchors}: {what} has no line for the skill's "
        "cite to resolve to, and a rename or a deletion is not fixed by re-measuring the "
        "number the prose carries")
    assert len(hits) == 1, (
        f"{anchors} occurs on {len(hits)} lines of {BACKLOG_PY} ({hits}), so the cite for "
        f"{what} has no single line to resolve to")
    return str(hits[0])


def _witness_ceiling_lines(lines: list[str]) -> tuple[str, str, str]:
    """The three line numbers the ceiling's prose may name, all three derived from the tree.

    `(constant, emitter, applying_check)`. The third is cited in the bullet's shorthand
    spelling — `` `:1473` ``, filename dropped — which is precisely why it is never a
    `RULED_CITE_RE` match and is checked as text beside the cite set rather than inside it.
    """
    return (_derived_backlog_line(lines, "the WITNESS_MAX_BYTES constant",
                                  WITNESS_CONSTANT_ANCHOR),
            _derived_backlog_line(lines, "the read-only-URI open _rederive_command builds",
                                  READONLY_URI_ANCHOR, READONLY_URI_MODE),
            _derived_backlog_line(lines, "the check that applies the ceiling",
                                  "size <= WITNESS_MAX_BYTES"))


def _assert_ceiling_cites_resolve(section: str, lines: list[str]) -> None:
    """Grade a Step 2.5 section's `backlog.py:<n>` cites against the lines just derived.

    Factored out of the node below so the two falsifier legs can put a SYNTHETIC section —
    live prose with one cite edited, built in memory, never written to the vault — through
    the same rule the real prose is graded on. Three things must hold, and each is a
    different way the prose can be wrong:

    * every fully-qualified `backlog.py:<n>` cite anywhere in the section must be ONE OF
      the two derived lines — a subset, not a one-element list, because the constant and
      the emitter are both real and the section is 820 lines long (#3228). It is a set
      because the rule is over WHICH lines are named: one true line cited twice is one
      fact stated twice, and the thing worth refusing is a number no line of this tree
      carries;
    * the witness bullet must still carry the constant's OWN cite, and that cite must
      EQUAL the derived constant line, so a stale or invented number is still caught even
      though a second cite is no longer one;
    * the bullet must still name the check that APPLIES the ceiling, in its shorthand
      spelling, so the companion cite cannot quietly go missing.
    """
    constant, emitter, size_check = _witness_ceiling_lines(lines)
    derived = {constant, emitter}
    cites = set(RULED_CITE_RE.findall(" ".join(section.split())))
    assert cites <= derived, (
        f"the Step 2.5 section cites backlog.py {sorted(cites - derived)}, which is neither "
        f"the WITNESS_MAX_BYTES constant (scripts/automod/backlog.py:{constant}) nor the "
        f"read-only-URI open _rederive_command builds (:{emitter}) — a cite that is neither "
        "derived line is stale or invented, so move the cite, not the constant")
    para = _witness_paragraph(section)
    para_cites = set(RULED_CITE_RE.findall(" ".join(para.split())))
    assert constant in para_cites, (
        f"the witness bullet cites backlog.py {sorted(para_cites)} but not "
        f"backlog.py:{constant}, where WITNESS_MAX_BYTES actually sits — admitting the "
        "emitter's cite was never a licence to drop the constant's")
    assert f":{size_check}" in para, (
        f"the skill cites the applying check as something other than :{size_check}")


def _section_citing(*numbers: str) -> str:
    """The live Step 2.5 section, with its ONE fully-qualified cite rewritten to name `numbers`.

    A synthetic section built this way is the real prose — same bullets, same fenced blocks,
    the same `_witness_paragraph` — with the cite text substituted and NOTHING else, so a leg
    that disagrees with it disagrees about a number and not about a fixture that drifted from
    the file. The vault is READ, never written: this returns a string built in memory.
    """
    section = _step_2_5_section(_skill_text())
    constant = _witness_ceiling_lines(_backlog_lines())[0]
    cite = f"backlog.py:{constant}"
    assert section.count(cite) == 1, (
        f"the Step 2.5 section names {cite} {section.count(cite)} times, so a synthetic "
        "section built from it would be rewriting the wrong text")
    return section.replace(cite, "` and `scripts/automod/".join(f"backlog.py:{n}"
                                                                for n in numbers), 1)


def test_the_witness_ceiling_cite_resolves_in_the_tree_it_names():
    """`backlog.py:1240` was 28 lines off; the cite must land on a line derived from here.

    Every number is DERIVED from the tree — the constant's assignment, the read-only-URI
    open `_rederive_command` builds, and the applying check as it already was — rather than
    written down here. Deriving them is what makes the prose editable: #3027 added a
    documented constant block above the constant and `da1ab7bb` added 493 module lines, so
    any such insertion moves a number the bullet has to carry, and a hard-coded one turns
    that edit into a red test in a round that never touched this file. Nothing is weakened
    by it either: the number the prose carries must still EQUAL the derived line, so
    `:1240` cannot pass, an anchor that went missing or doubled names itself instead of
    going green, and the emitter line must still sit inside the function whose command the
    bullet quotes."""
    lines = _backlog_lines()
    _assert_ceiling_cites_resolve(_step_2_5_section(_skill_text()), lines)
    emitter = _witness_ceiling_lines(lines)[1]
    fn = next((n for n in ast.walk(ast.parse(BACKLOG_PY.read_text(encoding="utf-8")))
               if isinstance(n, ast.FunctionDef) and n.name == "_rederive_command"), None)
    assert fn is not None, (
        f"no `_rederive_command` in {BACKLOG_PY}: the derived emitter line :"
        f"{emitter} is supposed to be the read-only open that function builds, and the "
        "witness bullet quotes that function BY NAME")
    assert fn.lineno <= int(emitter) <= fn.end_lineno, (
        f"the derived emitter line :{emitter} is outside `_rederive_command` "
        f"(:{fn.lineno}-:{fn.end_lineno}): the read-only-URI anchor matched somewhere else, "
        "so the cite the prose carries does not name the command the bullet quotes")


def test_a_witness_ceiling_cite_of_both_derived_lines_passes():
    """Falsifier leg (b) of #3228: a second cite naming the real emitter must be ADMITTED.

    `aaa98599` (#3225) added exactly this — a TRUE cite for the read-only-URI branch of
    `_rederive_command`, at `scripts/automod/backlog.py:1437-1441` — and the one-cite rule
    could not hold it: the guard red-ed the base probe of round `SM_20261024_035952` at
    `cade1351`, which filed #3226, and #3226's fix was pure prose (name the function, drop
    the line range). This node is the half that fix never had: with the section carrying
    BOTH derived lines, the guard has to pass.
    """
    lines = _backlog_lines()
    constant, emitter, _ = _witness_ceiling_lines(lines)
    assert emitter != constant, (
        f"the emitter and the constant both derived to line :{constant}, so this leg would "
        "be passing one cite wearing two names")
    _assert_ceiling_cites_resolve(_section_citing(constant, emitter), lines)


def test_a_witness_ceiling_cite_of_an_underived_number_reds():
    """Falsifier leg (a) of #3228: the guard must still bite on the number it was built for.

    The pair to the node above — same rule, same fixture, opposite verdict — so neither leg
    is satisfied by a check that simply never fires. Two underived numbers are tried,
    because the guard holds two things over the bullet: a fully-qualified cite that is
    neither derived line (`backlog.py:1240`, 28 lines off the constant, the error that made
    this guard necessary) must red, and the bullet's companion cite to the check that
    APPLIES the ceiling — the shorthand `:1473`, spelled without the filename and so
    invisible to `RULED_CITE_RE` — must not be replaceable by one either.
    """
    lines = _backlog_lines()
    constant, emitter, size_check = _witness_ceiling_lines(lines)
    assert UNDERIVED_CITE not in (constant, emitter, size_check), (
        f"a derived line has moved onto :{UNDERIVED_CITE}, the number this leg falsifies "
        "with; pick another, or the leg would be calling a real line a fake one")
    with pytest.raises(AssertionError, match=UNDERIVED_CITE):
        _assert_ceiling_cites_resolve(_section_citing(UNDERIVED_CITE), lines)
    section = _section_citing(constant)
    para = _witness_paragraph(section)
    assert f":{size_check}" in para, (
        f"the witness bullet does not carry the companion cite :{size_check}, so there is "
        "nothing here to drop and this leg would be asserting nothing about it")
    dropped = section.replace(para, para.replace(f":{size_check}", f":{UNDERIVED_CITE}", 1), 1)
    with pytest.raises(AssertionError, match=rf":{size_check}"):
        _assert_ceiling_cites_resolve(dropped, lines)


# --------------------------------------------- clause 5: the record an empty run leaves

def test_an_empty_run_prints_no_covering_set_and_the_skill_says_what_it_records(tmp_path):
    """`pairs=0` prints covered_by=none by construction, so the case needs its own label.

    The half that runs the snippet is the reason the prose exists: the `covered =`
    comprehension is guarded by `if run`, so the empty set — a subset of every ruled set —
    can never name a covering set. Neither filing branch reaches it, which is the gap
    #2998 clause 5 closes.
    """
    section = _step_2_5_section(_skill_text())
    script = _setcheck_script(section)
    out = _run_setcheck(script, "post-apply-empty", "", tmp_path)
    assert out == "SD-STRANDED SETCHECK post-apply-empty: pairs=0 novel=none covered_by=none", (
        f"an empty proposals file reads {out!r}: the `if run` guard is what makes "
        "covered_by unprintable on the empty set, and this node's prose half depends on it")
    assert "no-filing post-apply-empty" in section, (
        f"{SKILL_PATH} still does not name what an empty run records, so the run has no "
        "branch to follow: novel=none means no filing duty and covered_by=none means no "
        "set-identical receipt")
    bullet = _one([b for b in _bullets(section) if "no-filing post-apply-empty" in b],
                  "bullet naming the empty run's record")
    assert "SD-stranded-1" in bullet, (
        "the empty-run record must be written beside the standing decision it belongs to")
    assert re.search(r"proposed edges.{0,40}0|pairs=0", bullet, re.S), (
        f"the record is not tied to the reading that produces it: {bullet[:160]!r}")
    assert "novel=none" in bullet and "covered_by=none" in bullet, (
        "the record must name the printed triple it is keyed on, or a reader cannot tell "
        "which branch the run was in")
    assert "Before the apply" in bullet or "before the apply" in bullet, (
        "an empty set BEFORE the apply is a different fact — zero admissible candidates with "
        "an unexecuted authorization — and must not borrow the post-apply label")


def test_the_empty_run_label_is_what_the_ledger_line_carries():
    """The label is not just prose: it is the literal field the run appends."""
    section = _step_2_5_section(_skill_text())
    printf = [b for b in FENCE_RE.findall(section) if "no-filing post-apply-empty" in b]
    assert len(printf) == 1, (
        f"expected one fenced printf appending no-filing post-apply-empty, found "
        f"{len(printf)} — the record has to be appendable verbatim or runs will paraphrase it")
    assert "stranded-standing-decision.log" in printf[0], (
        "the empty-run record must go to SD-stranded-1's ledger, the same receipt file the "
        "set-identical case appends to")


# ------------------------------------------------------------------ this file's own rails

def _pytest(extra: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "pytest", *extra],
                          cwd=str(ROOT), capture_output=True, text=True, timeout=600)


def test_the_gate_mark_expression_still_collects_every_node_here():
    """A node the gate's `-m` expression deselects enforces nothing, so none may be marked.

    `scripts/automod/gate.py:641` runs the suite with
    `-m "not live_vault and not fault_injection"`; these nodes read the live vault, so
    marking one would hide it from the very run meant to enforce it.
    """
    plain = _pytest(["--collect-only", "-q", THIS_FILE])
    marked = _pytest(["--collect-only", "-q", "-m",
                      "not live_vault and not fault_injection", THIS_FILE])
    assert plain.returncode == 0, plain.stdout[-600:] + plain.stderr[-600:]
    assert marked.returncode == 0, marked.stdout[-600:] + marked.stderr[-600:]

    def count(out: str) -> int:
        return len([l for l in out.splitlines() if l.startswith(THIS_FILE) and "::" in l])

    assert count(plain.stdout) >= 10, (
        f"only {count(plain.stdout)} nodes collected from {THIS_FILE}; this file is meant "
        "to pin six clauses, so a collection collapse is itself the regression")
    assert count(plain.stdout) == count(marked.stdout), (
        f"{count(plain.stdout)} nodes collect plainly but {count(marked.stdout)} under the "
        "gate's mark expression — a marked node here is a clause with no enforcement")


# ── PS-3 as a RULED-but-UNEXECUTED set, and the hand_run block that holds it (#3017) ──
#
# Eight carriers of the stranded apply went `status: done` with the write unrun, the last
# of them (#3011) at 2026-10-19T09:06:29Z, answered by `ruling: see entry 1 — one entry
# only`. #3017 gave a ruled-but-unexecuted hand-run a shape the loop cannot close — a
# `hand_run:` front-matter block whose `receipt` command must exit 0 before any closer
# fires — and PS-3 (PS-2 plus `OCR-Assisted ASR -> ASR`, the set the 2026-10-19T070503Z
# run printed) is the decision that shape was built for. These nodes read the skill's own
# records; none of them runs `--apply`, and none of them opens the live store, which is
# still owed to an attended session by SD-stranded-1's executor rule at §3.

PS_3_PAIRS = PS_2_PAIRS | {("OCR-Assisted ASR", "ASR")}

#: PS-3's pairs as `hand_run.set` spells them, which is the spelling `STRANDED step 2.5:`
#: prints and therefore the spelling the filing branch compares sets in.
PS_3_SPELLED = [f"{s} -> {t}" for s, t in sorted(PS_3_PAIRS)]


def _sd_stranded_1_section(text: str) -> str:
    """The standing-decision section, from its heading to the next heading."""
    head = "**Standing apply decision SD-stranded-1"
    i = text.index(head)
    rest = text[i + len(head):]
    end = rest.find("\n## ")
    return text[i:] if end < 0 else text[i:i + len(head) + end]


def _ps3_seven_pair_payload() -> str:
    """PS-2's committed witness rows plus a seventh row for `OCR-Assisted ASR -> ASR`, in
    the shape `link_stranded_entities.py` writes. Synthesised rather than read out of
    `_pipeline/`, because that directory has no history, the gate runs under a scaffolded
    HOME, and the seventh pair's own artefact is on #3011's owed list, not in the tree."""
    rows = [json.loads(l) for l in
            _committed_ps2_bytes().decode().splitlines() if l.strip()]
    extra = dict(rows[0])
    extra.update({"source": "OCR-Assisted ASR", "target": "ASR",
                  "evidence": "name-embedding:0.913"})
    return "".join(json.dumps(r) + "\n" for r in rows + [extra])


def test_ps_3_is_ruled_in_the_setcheck_and_a_seven_pair_run_reads_covered(tmp_path):
    """#3017 clause 5, the machine-readable half: `RULED` gains PS-3 = PS-2 plus the
    seventh pair, and the set it names is covered — so the run that printed it does not
    file a ninth carrier.

    Its marker is checked too, and it no longer says what this docstring used to claim.
    `NOT EXECUTED` was the state at 2026-10-19T09:51Z, when the store read 12 active
    `name-embedding:` edges and 0 live for all seven pairs; the attended by-hand `--apply`
    of 2026-10-19T16:44:20Z wrote them, and the same probe now prints
    `ACTIVE_NAME_EMBEDDING=19 PAIRS_LIVE=7/7 MISSING=0` and exits 0. So the node demands the
    state the STORE reports, with the apply instant beside it, rather than the pending wording
    — which is what keeps a seventh carrier from being filed by a reader who trusts a marker
    that has fallen behind the write."""
    section = _step_2_5_section(_skill_text())
    script = _setcheck_script(section)
    ruled = _ruled_dict(script)
    assert "PS-3" in ruled, f"RULED holds only {sorted(ruled)}"
    assert ruled["PS-3"] == PS_3_PAIRS, (
        f"PS-3 reads {sorted(ruled['PS-3'])}, expected PS-2 plus "
        "`OCR-Assisted ASR -> ASR`: "
        f"{sorted(PS_3_PAIRS)}")
    assert ruled["PS-2"] < ruled["PS-3"], "PS-3 is not a superset of PS-2"
    assert ruled["PS-3"] - ruled["PS-2"] == {("OCR-Assisted ASR", "ASR")}
    state, instant = _execution_claim(section, "PS-3", _live_name_embedding_pairs()[1])
    assert state == "executed" and instant == APPLY_INSTANT, (
        f"PS-3's RULED marker declares {state}" + (f" at {instant}" if instant else "") +
        f": a ruled set has to declare which side of the apply it is on, and the store holds "
        f"all seven of its pairs, written at {APPLY_INSTANT}. Declaring neither state is the "
        "sentence PS-1 sat without for eleven days; declaring `NOT EXECUTED` now is a run "
        "being told to re-apply edges that are live.")

    out = _run_setcheck(script, "ps3-seven-pair", _ps3_seven_pair_payload(), tmp_path)
    assert out.endswith("pairs=7 novel=none covered_by=['PS-3']"), (
        f"a run reprinting PS-3's seven pairs reads {out!r}; `novel=` naming a pair is the "
        "filing trigger, and a ninth carrier for a set this file has already ruled is the "
        "loop that produced #2970 #2986 #2989 #2998 #3011")


def test_the_filing_branch_suppresses_a_filing_for_a_pending_hand_run(tmp_path):
    """#3017 clause 4: the setcheck records `no-filing pending-hand-run-<id>` against the
    carrier instead of FILED, and the read that decides it is a board read the run performs
    from the checkout — never an import inside the set-check snippet.

    The second half is not a style rule. `_run_setcheck` runs the snippet with only `HOME`
    and `PATH` in its environment and the staged home as cwd, which is how the run runs it,
    and a `from scripts.automod import owed` inside it dies with `ModuleNotFoundError: No
    module named 'scripts'` — the exact red this node replaces. So the branch must ask the
    board OUTSIDE the snippet, and the snippet's printed line keeps the two fields the
    other nodes in this file pin.
    """
    section = _step_2_5_section(_skill_text())
    script = _setcheck_script(section)
    assert "no-filing pending-hand-run-" in section, (
        "the filing branch no longer records a pending hand-run: a suppression that is a "
        "run's judgement rather than a line in this file dies with the run — #2998's pass "
        "wrote `no-filing novel-set-covered-by-open-carrier-2998` into the ledger, no rule "
        "here defined it, and the next run filed #3011")
    assert "hand_runs()" in section and "cd ~/lloyd" in section, (
        "the branch does not say where the pending run is read from")
    assert "scripts.automod" not in script and "hand_runs" not in script, (
        "the set-check snippet itself imports the repo, so it cannot run under the "
        "environment the run and `_run_setcheck` give it")
    out = _run_setcheck(script, TS_PS2, _committed_ps2_bytes().decode(), tmp_path)
    assert out == (f"SD-STRANDED SETCHECK {TS_PS2}: pairs=6 novel=none "
                   "covered_by=['PS-2', 'PS-3']"), (
        f"the printed line changed shape ({out!r}); the pending-hand_run field belongs to "
        "the branch, not to the print the rest of this file pins")


def test_the_standing_decision_records_the_hand_run_block_for_ps_3():
    """#3017 clause 5, the record a newly filed carrier is stamped from.

    `owed.hand_run_of` reads {set, cmd, receipt, since} and `owed.hand_run_pending` runs
    `receipt`; this node pins that what SD-stranded-1 records is the same shape and the
    same seven pairs, that `cmd` is the SD-stranded-1 apply command, and that `receipt` is
    the 12 + 7 / every-pair-live probe rather than anything that could write: it must name
    `app.data_root.PRODUCTION_DATA_ROOT` (#2998's round `SM_20261019_024744` lost a gate
    anchoring a probe on `Path.home()`, fixed in `2cb58091`), and must NOT contain
    `--apply`, because the thing that proves a write happened can never be the write.

    It is never executed here: the gate runs the suite under a scaffolded HOME where
    `~/lloyd-data` is empty, so a reading taken in this process would be a measurement of
    nothing. Its measured form, taken 2026-10-19T09:51Z against the live store by hand
    (0.21 s, exit 1, store md5 unchanged), is quoted in the skill beside the block.
    """
    import yaml
    section = _sd_stranded_1_section(_skill_text())
    # FENCE_RE at the top of this file matches ```bash only, and this block is yaml.
    yaml_fences = re.compile(r"^[ \t]*```yaml\n(.*?)\n[ \t]*```", re.S | re.M)
    block = [x for x in yaml_fences.findall(section) if x.lstrip().startswith("hand_run:")]
    assert len(block) == 1, (
        f"expected exactly one fenced `hand_run:` block in SD-stranded-1, found "
        f"{len(block)} — this is the shape every newly filed carrier is stamped with")
    hr = yaml.safe_load(block[0])["hand_run"]
    assert sorted(hr) == ["cmd", "receipt", "set", "since"], (
        f"the block carries {sorted(hr)}; `owed.hand_run_of` reads exactly "
        "set, cmd, receipt, since")
    assert len(hr["set"]) == 7 and set(hr["set"]) == set(PS_3_SPELLED), (
        f"the block's set is {hr['set']}, expected the seven pairs PS-3 rules, spelled "
        f"{PS_3_SPELLED} — the spelling Step 2.5 prints, which is what the filing branch "
        "compares a run's set against. Order carries nothing and is not asserted: the branch "
        "compares `set(h['set'])` against the run's set, and this list is written in the "
        "order the artefact printed them.")
    assert all(" -> " in x for x in hr["set"]), "a pair is not in `source -> target` spelling"
    assert hr["cmd"].startswith("cd ~/lloyd") and hr["cmd"].rstrip().endswith("--apply")
    assert "--min-facts 10" in hr["cmd"] and hr["cmd"].count("--apply") == 1
    r = hr["receipt"]
    assert "app.data_root import PRODUCTION_DATA_ROOT" in r, (
        "the receipt anchors on something other than the production data root — #2998's "
        "round lost a gate on exactly this (round SM_20261019_024744, fixed in 2cb58091)")
    assert "KG_DB_RELATIVE" in r and "name-embedding:" in r, (
        "the receipt does not count active `name-embedding:` edges")
    assert "include_expired=False" in r, (
        "the receipt does not read live edges only, which is `expired_at IS NULL` in the "
        "probe #3011's bullet 10 spells")
    assert "len(rows)>=19" in r.replace(" ", ""), (
        "the receipt's count is not 12 + 7 = 19: the pilot wrote 12 on 2026-10-05T14:22:23Z "
        "and PS-3 adds 7")
    assert all(s in r and t in r for s, t in PS_3_PAIRS), (
        "the receipt does not name every one of the seven pairs, so it could go green with "
        "a pair still unwritten")
    assert "--apply" not in r and "SystemExit" in r, (
        "the receipt is not a read-only probe: it must exit non-zero until the by-hand run "
        "has happened, and it must never be the run itself")


# ── the marker checker must be able to fail ─────────────────────────────────
#
# A node that asserts "the marker and the store agree" passes for exactly one reason: they
# agree. If the checker could not see a disagreement, the two clause nodes above would be a
# record of today's state and not a guard on tomorrow's. So the checker runs here against
# readings that are each wrong in one specific way — three of them states this file has
# actually held, and the fourth a store reading that really was measured.

def test_the_marker_checker_refuses_a_state_the_store_contradicts():
    """Every refusal direction of `_execution_claim`, on text and readings that are wrong in
    one named way each.

    (a) `PS-3`'s comment put back to `NOT EXECUTED`, the state #3017 published and the true
        one until 16:44:20Z. All seven of its pairs are live now, so a checker that let this
        through would be licensing a second apply of edges the store already holds — which is
        the hazard #2986 exists because of.
    (b) The live reading from before that apply, missing `OCR-Assisted ASR -> ASR`, against
        the marker as it stands. `EXECUTED` while the store cannot produce the pair is the
        false green #854 was filed for, in the one place a person reads before pushing work
        they believe is written.
    (c) Markers that declare both states, neither, or an `EXECUTED` with no instant: the three
        shapes that make the state unreadable rather than merely wrong.
    """
    section = _step_2_5_section(_skill_text())
    count, live = _live_name_embedding_pairs()

    # The positive control first: the checker accepts the real pair of readings, or the three
    # refusals below would be proving that it always raises.
    assert _execution_claim(section, "PS-3", live) == ("executed", APPLY_INSTANT), (
        f"the live store ({count} active `name-embedding:` edges) and PS-3's marker both "
        "read executed-at-apply, so a refusal here is the checker, not the file")

    stale = section.replace("# PS-3 — EXECUTED", "# PS-3 — NOT EXECUTED", 1)
    assert stale != section, "PS-3's marker is not in the shape this check can age"
    try:
        _execution_claim(stale, "PS-3", live)
        raise AssertionError(
            "a `NOT EXECUTED` marker on a set whose seven pairs are all live was accepted: "
            "the stale-pending direction is unguarded")
    except AssertionError as exc:
        assert "still owed" in str(exc), (
            f"refused for the wrong reason — expected the stale-pending refusal, got: {exc}")

    pre_apply = live - {("OCR-Assisted ASR", "ASR")}
    try:
        _execution_claim(section, "PS-3", pre_apply)
        raise AssertionError(
            "an `EXECUTED` marker was accepted while `OCR-Assisted ASR -> ASR` is absent from "
            "the store: the hollow-executed direction is unguarded")
    except AssertionError as exc:
        assert "OCR-Assisted ASR -> ASR" in str(exc), (
            f"refused for the wrong reason — expected the missing-pair refusal naming the "
            f"pair, got: {exc}")

    for mark, what in (
        ("# PS-9 — EXECUTED by an attended by-hand --apply", "without an instant"),
        ("# PS-9 — EXECUTED 2026-10-19T16:44:20Z; the apply is NOT EXECUTED",
         "declaring both states"),
        ("# PS-9 — ruled, supersedes PS-3", "declaring no state"),
    ):
        try:
            _declared_execution(mark, "PS-9")
            raise AssertionError(f"a marker {what} was accepted")
        except AssertionError as exc:
            assert "PS-9" in str(exc), (
                f"the {what} refusal did not name the key it refused: {exc}")


def test_every_ruled_key_besides_ps_1_declares_a_state_the_store_agrees_with():
    """The agreement check runs over `RULED`'s keys, not over names written into a node.

    `RULED` is an open set — a further ruled set is the whole point of having a dict — so a
    check that enumerates its keys by name leaves the next key born unchecked, which is the
    class of guard that cannot hold. `PS-1` is the single named exception and it is justified
    in the file it is excused by: PS-2's comment carries that set's own receipt
    (`PS1_LIVE=5/5`, the pre-apply probe of #2998's bullet 10), so its status is recorded on
    the page, beside a key that declares its own. Nothing else may be silent.

    What each key is measured against is now its OWN entry in `APPLY_INSTANTS`, and that is
    #3065's point: `PS-2` and `PS-3` were written by the apply of 2026-10-19T16:44:20Z, while
    `PS-4`'s two edges went in at 2026-10-20T09:15:39Z, so a single module constant on the
    right-hand side would read `PS-4`'s true marker as a lie — and the only edit that would
    silence it is the false one, stamping a set with the instant of a backup that does not
    hold it. A key with no entry fails here and names itself.
    """
    section = _step_2_5_section(_skill_text())
    ruled = _ruled_dict(_setcheck_script(section))
    count, live = _live_name_embedding_pairs()
    assert "PS1_LIVE=5/5" in _ruled_comment(section, "PS-2"), (
        "PS-1 is excused from declaring a state only because PS-2's comment carries its "
        "receipt; without that sentence the exception is a hole and not a ruling")
    for key in sorted(ruled):
        if key == "PS-1":
            continue
        assert key in APPLY_INSTANTS, (
            f"{key} is in `RULED` with {len(ruled[key])} pair(s) but has no entry in "
            f"APPLY_INSTANTS: {sorted(APPLY_INSTANTS)}. Add the instant of the apply that "
            "wrote ITS edges — every settled set's edges are live, so without its own date "
            "this node could only skip it or accuse it")
        state, instant = _execution_claim(section, key, live)
        assert state == "executed" and instant == APPLY_INSTANTS[key], (
            f"{key} ({len(ruled[key])} pairs) declares {state}"
            + (f" at {instant}" if instant else "")
            + f", and the live store holds all {len(ruled[key])} of its pairs — every ruled "
            f"set whose edges are live has to name its own apply, {APPLY_INSTANTS[key]}, "
            f"the run whose `store-backups/kg-stranded-*.sqlite` holds it")
    assert count >= 19, (
        f"{count} active `name-embedding:` edges: the pilot wrote 12 on 2026-10-05T14:22:23Z "
        f"and the PS-3 apply added 7, so a count under 19 is the store losing written work")


# ── #3065: the record left standing after the SECOND apply ──────────────────
#
# Everything above this line was written while the store read 19 active `name-embedding:`
# edges, and a present-tense sentence about the store gets read as "today" by whoever finds it.
# #3060's carrier was applied by hand at 2026-10-20T09:15:39Z — receipt
# `ACTIVE_NAME_EMBEDDING=21 PAIRS_LIVE=2/2 MISSING=0`, exit 0 — and the skill is the only party
# that did not move with it: `RULED` never gained the set those two edges belong to, two
# sentences still described 19 edges as the state of the store, and three of the verification
# readings it tells an executor to take were never sound. The nodes below pin the corrected
# record. None of them runs `--apply` or opens the store for write: §3's executor rule is
# unchanged by this, and the store is the constant this diff must leave where #3060 left it.

PS_4_PAIRS = {("Automod Scorecard", "automod"), ("automod tests rung", "automod")}

#: The Step 2.5 run that printed PS-4's set — `novel=['Automod Scorecard -> automod',
#: 'automod tests rung -> automod'] covered_by=none`, `covered_by` being none precisely because
#: no key held them. A runtime artefact under `_pipeline/`, which has no history and is empty
#: under the gate's scaffolded HOME, so the same two rows are synthesised as well and the file
#: is used only when the box still has it.
PS_4_PROPOSALS = (Path.home() / "lloyd-data/_pipeline"
                  / "stranded-proposals-2026-10-20T084644Z.jsonl")


def _ps10_pair_payload() -> str:
    """PS-10's one proposal row in the shape `link_stranded_entities.py` writes it.

    Built off the committed PS-2 witness's row shape for the reason spelled out on
    `_ps4_pair_payload`, and it is ONE row: the pair the key had to cover is one pair, and a
    payload with a second row would test a set the record does not claim.
    """
    rows = [json.loads(l) for l in
            _committed_ps2_bytes().decode().splitlines() if l.strip()]
    source, target = sorted(PS_10_RULING_PAIR)[0]
    r = dict(rows[0])
    r.update({"source": source, "target": target,
              "evidence": f"name-embedding: '{target}' is a registered entity name "
                          f"inside '{source}'"})
    return json.dumps(r) + "\n"


def _without_key(script: str, key: str) -> str:
    """`script` with `key`'s entry and its comment lines taken out — the record before it had one.

    The mutated text must still parse and still hold every OTHER key, or a "refusal" printed by a
    snippet that crashed on the mutation would be scored as the key having been load-bearing.
    """
    entry = r"(?:\n[ ]*#[^\n]*)*\n[ ]*\"" + re.escape(key) + r"\": \{[^}]*\}"
    # A key with a neighbour after it takes its own trailing comma with it; the LAST key hands
    # its closing brace back to the entry before it. Never both: the literal has one last key.
    out, n = re.subn(entry + r",", "", script, count=1)
    if n == 0:
        out, n = re.subn(r"\}," + entry + r"\}", "}}", script, count=1)
    assert n == 1, (f"the mutation removed no `{key}` entry: the RULED literal no longer has "
                    f"the shape this expects, so the mutant proves nothing")
    return out


def test_ps_10_is_a_ruled_key_whose_pair_the_set_check_covers_by_name(tmp_path):
    """#4519 clauses 1-4: the applied pair is a ruled key, and the snippet now says so by name.

    The bite was latent and the mechanism was not. `covered_by=` is computed against `RULED`
    alone, so while the literal stopped at `PS-9` the set-check answered
    `novel=['agent-supervisord.service -> supervisord'] covered_by=none` for a pair whose edge was
    LIVE — which reads as a new decision to the branch that opens a carrier. Today's silence is
    the linker's `edge already live` branch keeping the pair out of the proposals JSONL, and an
    expiry is the one thing the record elsewhere says an edge does rather than delete: one
    `expired_at` and the missing key fired on its own. So the pair belongs in `RULED` for the same
    reason `PS-4` does, and this node is the same shape as `test_ps_4_is_ruled_and_a_reprint_of_
    its_pairs_is_covered_not_novel` on purpose — one ruled set, its exact pairs, and a reprint of
    those pairs that must come back covered.

    The exactness is the safety half, not decoration: a key holding a second pair would answer
    `covered_by=['PS-10']` for a proposal nobody ruled on, which is a fabricated authorization
    (#2829's over-inclusion list is the precedent). The epoch is the honesty half: PS-9's node
    established that an instant copied from a comment can only ever agree with the comment that
    copied it, so the number here is read off `stat` and compared, and the mutant that moves it by
    one second has to be caught.
    """
    text = _skill_text()
    section = _step_2_5_section(text)
    script = _setcheck_script(section)
    ruled = _ruled_dict(script)

    # Clause 1: the key, its place, and the pair set it is allowed to cover.
    assert "PS-10" in ruled, (
        f"`RULED` carries {sorted(ruled)}: carrier #4257's apply of 2026-11-19T15:46:35Z is "
        "still not a key, so the set-check calls a live edge novel")
    assert ruled["PS-10"] == PS_10_RULING_PAIR, (
        f"`RULED['PS-10']` holds {sorted(ruled['PS-10'])}, not carrier #4257's one pair "
        f"{sorted(PS_10_RULING_PAIR)} — a subset re-proposes the rest, a superset authorises "
        "proposals nobody ruled")
    keys = list(ruled)
    assert keys.index("PS-10") == keys.index("PS-9") + 1, (
        f"`RULED` is in dict order and `PS-10` sits at {keys.index('PS-10')} with "
        f"`PS-9` at {keys.index('PS-9')}: {keys}")

    # Its marker, weighed the way every other applied key's is.
    count, live = _live_name_embedding_pairs()
    state, instant = _execution_claim(section, "PS-10", {(s, t) for s, t in live})
    assert (state, instant) == ("executed", PS_10_APPLY_INSTANT), (
        f"`PS-10` declares {state}" + (f" at {instant}" if instant else "")
        + f" while the store holds its pair; it has to carry `EXECUTED` with "
        f"{PS_10_APPLY_INSTANT} on the same line, the shape PS-4 and PS-9 carry")
    assert APPLY_INSTANTS["PS-10"] == PS_10_APPLY_INSTANT

    # Clause 1's provenance: the named artefact, matched by epoch, in the entry itself.
    comment = _ruled_comment(section, "PS-10")
    assert PS_10_BACKUP in comment and "stat -c %Y" in comment, (
        f"`PS-10`'s entry does not name both the artefact and the command that reads it: "
        f"{comment!r}")
    backup = LIVE_STORE.parent / "store-backups" / PS_10_BACKUP
    assert backup.is_file(), (
        f"{backup} is absent, so the instant `PS-10` declares has no witness that is not the "
        "claim itself — the pre-apply backup is the only one")
    actual_epoch = int(backup.stat().st_mtime)
    assert actual_epoch == 1795103195, (
        f"{PS_10_BACKUP} reports {actual_epoch} "
        f"({datetime.fromtimestamp(actual_epoch, timezone.utc).isoformat()}), not the epoch "
        "clause 1 quotes: the artefact moved, or the pair was written by a different run")
    assert _epoch_claims(comment) == {str(actual_epoch)}, (
        f"`PS-10`'s entry claims epoch(s) {sorted(_epoch_claims(comment))} where the file says "
        f"{actual_epoch} — one of the two is a copy of a claim rather than a measurement")
    assert not re.search(r"Nov 19 0[78]", comment), (
        f"`PS-10`'s entry carries an `ls`-shaped date, which no later run can re-check: {comment!r}")
    moved = comment.replace(str(actual_epoch), str(actual_epoch + 1))
    assert _epoch_claims(moved) != {str(actual_epoch)}, (
        "moving the quoted epoch by one second left it agreeing with the artefact, so the "
        "comparison above never compared anything")

    # Clause 4: the set-check, run as Step 2.5 runs it — HOME and PATH and nothing else.
    payload = _ps10_pair_payload()
    out = _run_setcheck(script, "ps10-covered", payload, tmp_path)
    assert out.endswith("pairs=1 novel=none covered_by=['PS-10']"), (
        f"the set-check answered {out!r} for a reprint of the pair PS-10 rules; the novel-set "
        "branch reads `novel=['...'] covered_by=none` as a new decision and files a carrier")

    removed = _without_key(script, "PS-10")
    assert "PS-10" not in _ruled_dict(removed) and len(_ruled_dict(removed)) == len(ruled) - 1, (
        "the mutant still carries PS-10, so the refusal below is not the key's doing")
    out2 = _run_setcheck(removed, "ps10-key-removed", payload, tmp_path)
    assert out2.endswith("pairs=1 novel=['agent-supervisord.service -> supervisord'] "
                         "covered_by=none"), (
        f"with the key removed the snippet answered {out2!r}; it must go back to calling the "
        "pair novel, which is the behaviour #4519 exists to end — and the only reason it is not "
        "firing today is the live edge")

    # Clause 2: the figure, derived, attributed per apply, and unchanged where it counts the store.
    n_pairs = _ruled_pair_total(text)
    assert n_pairs == 16 == len(set().union(*ruled.values())), (
        f"the distinct pairs `RULED` enumerates is {len(set().union(*ruled.values()))} while the "
        f"figure checkers derive {n_pairs}: PS-1..PS-3 are nested and PS-11 sits inside PS-12, so "
        "they contribute nothing new and the sum is the size of the union — 14 through PS-10, "
        "plus PS-12's two")
    assert _wrong_ruled_pair_figures(text, n_pairs) == [], (
        f"a site still states a ruled-pair figure other than {n_pairs}")
    assert len(re.findall(RULED_PAIR_FIGURE_RE, " ".join(text.split()))) == 2, (
        "the enumerated figure must be stated at exactly two sites — the ledger and the rule — "
        "and a third copy is the one that will not be updated")
    flat = " ".join(text.split())
    for key in ("PS-3", "PS-4", "PS-9", "PS-10", "PS-12"):
        want = _number_word(len(ruled[key]))
        assert f"`{key}`'s {want} " in flat, (
            f"the per-apply attribution does not name `{key}`'s {want} pair(s): a total with no "
            "denominator is how this record reached nineteen")
    assert _stale_store_readings(flat, len(live)) == [], (
        "the store-count sites moved with the ruled-pair figure: they count the `name-embedding:` "
        "edges that are live, and PS-10's is one of them, not an extra")

    # Clause 3: the module's own half — nothing in here may still call this set unruled. Each
    # phrase is JOINED AT RUNTIME for the reason this module's own bans are spelled
    # `(?:PS|SD)[-_]stranded`: a forbidden literal written whole into the loop that searches for
    # it is present by construction, and the node can only ever fail.
    here = Path(__file__).read_text(encoding="utf-8")
    for head, tail in (("applied but NOT a ", "`RULED` key"),
                       ("applied but not ", "ruled"),
                       ("applied-but-", "unruled"),
                       ("inventing the key ", "here"),
                       ("nobody ruled ", "PS-10")):
        stale = head + tail
        assert stale not in here, (
            f"{THIS_FILE} still says {stale!r}: PS-10 is a ruled key with an apply instant in "
            "APPLY_INSTANTS, and a comment that says otherwise is read by the next run as a "
            "reason not to add the next one")
    assert "#4519" in here, (
        "the new comment cites no open carrier: #4480 and #4257 are both `status: done`, and a "
        "closed item is not an owner")


def _pair_payload(pairs) -> str:
    """Proposal rows for `pairs` in the shape `link_stranded_entities.py` writes, one per pair.

    Built off the committed PS-2 witness's row shape for the reason on `_ps4_pair_payload`."""
    base = json.loads(next(l for l in _committed_ps2_bytes().decode().splitlines() if l.strip()))
    out = []
    for source, target in sorted(pairs):
        r = dict(base)
        r.update({"source": source, "target": target,
                  "evidence": f"name-embedding: '{target.lower()}' is a registered entity name "
                              f"inside '{source}'"})
        out.append(json.dumps(r) + "\n")
    return "".join(out)


def test_ps_11_and_ps_12_are_ruled_keys_the_one_apply_of_2026_11_26_wrote(tmp_path):
    """The PS-10 rule applied to a nested pair: one attended apply, two carriers, two keys.

    Carrier #4503 held `Kimchi by Cast AI -> Cast AI` alone (`PS-11`); #4560's run printed that pair
    plus `Dual pulley servo wheel -> Servo` (`PS-12`), and the apply of 2026-11-26T17:35:28Z wrote
    the superset. So both keys are in `RULED`, each holding EXACTLY its carrier's set, nested the way
    `PS-1` ⊂ `PS-2` ⊂ `PS-3` are, both weighed against the one instant, and the instant is read off
    the backup's `stat`, never off the comment that states it. The removal mutants prove the keys
    are what makes a reprint covered: without `PS-12` the servo pair is novel again, and without
    both the Kimchi pair is too.
    """
    text = _skill_text()
    section = _step_2_5_section(text)
    script = _setcheck_script(section)
    ruled = _ruled_dict(script)

    assert ruled.get("PS-11") == PS_11_RULING_PAIR and ruled.get("PS-12") == PS_12_WRITTEN_PAIRS, (
        f"`RULED` holds PS-11={sorted(ruled.get('PS-11', ()))} PS-12={sorted(ruled.get('PS-12', ()))}"
        f"; carriers #4503 and #4560 ruled {sorted(PS_11_RULING_PAIR)} and "
        f"{sorted(PS_12_WRITTEN_PAIRS)} — a subset re-proposes the rest, a superset authorises "
        "proposals no carrier held")
    assert ruled["PS-11"] < ruled["PS-12"], "PS-11 is not inside PS-12"
    keys = list(ruled)
    assert keys[-3:] == ["PS-10", "PS-11", "PS-12"], (
        f"`RULED` is in dict order, and the smallest covering set is named first only if the "
        f"subset precedes its superset: {keys}")

    _count, live = _live_name_embedding_pairs()
    for key in ("PS-11", "PS-12"):
        assert _execution_claim(section, key, live) == ("executed", PS_12_APPLY_INSTANT), (
            f"`{key}` must declare `EXECUTED {PS_12_APPLY_INSTANT}` on its marker line — both "
            "keys were written by that one apply")

    comment = _ruled_comment(section, "PS-12")
    assert PS_12_BACKUP in comment and "stat -c %Y" in comment, (
        f"`PS-12`'s entry does not name both the artefact and the command that reads it: "
        f"{comment!r}")
    backup = LIVE_STORE.parent / "store-backups" / PS_12_BACKUP
    assert backup.is_file(), (
        f"{backup} is absent, so the instant `PS-12` declares has no witness that is not the "
        "claim itself")
    actual_epoch = int(backup.stat().st_mtime)
    assert actual_epoch == 1795714528 and datetime.fromtimestamp(
        actual_epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") == PS_12_APPLY_INSTANT, (
        f"{PS_12_BACKUP} reports {actual_epoch}, not the epoch of the 2026-11-26T17:35:28Z apply")
    assert _epoch_claims(comment) == {str(actual_epoch)}, (
        f"`PS-12`'s entry claims epoch(s) {sorted(_epoch_claims(comment))} where the file says "
        f"{actual_epoch}")

    both = _pair_payload(PS_12_WRITTEN_PAIRS)
    kimchi = _pair_payload(PS_11_RULING_PAIR)
    assert _run_setcheck(script, "ps12-covered", both, tmp_path).endswith(
        "pairs=2 novel=none covered_by=['PS-12']")
    assert _run_setcheck(script, "ps11-covered", kimchi, tmp_path).endswith(
        "pairs=1 novel=none covered_by=['PS-11', 'PS-12']"), (
        "a reprint of PS-11's one pair must name both covering sets, smallest first")

    no_12 = _without_key(script, "PS-12")
    assert sorted(_ruled_dict(no_12)) == sorted(set(ruled) - {"PS-12"})
    assert _run_setcheck(no_12, "ps12-removed", both, tmp_path).endswith(
        "pairs=2 novel=['Dual pulley servo wheel -> Servo'] covered_by=none")
    no_both = _without_key(no_12, "PS-11")
    assert sorted(_ruled_dict(no_both)) == sorted(set(ruled) - {"PS-11", "PS-12"})
    assert _run_setcheck(no_both, "ps11-removed", kimchi, tmp_path).endswith(
        "pairs=1 novel=['Kimchi by Cast AI -> Cast AI'] covered_by=none"), (
        "with both keys removed the Kimchi pair must read novel again — the reading #4503 was "
        "filed on")


def _ps4_pair_payload() -> str:
    """PS-4's two proposal rows in the shape `link_stranded_entities.py` writes them.

    Synthesised off the committed PS-2 witness's row shape for the reason spelled out on
    `_ps3_seven_pair_payload`: `_pipeline/` has no history, so a node that read only the
    runtime file would silently check nothing wherever the gate runs it.
    """
    rows = [json.loads(l) for l in
            _committed_ps2_bytes().decode().splitlines() if l.strip()]
    out = []
    for source in sorted(s for s, _ in PS_4_PAIRS):
        r = dict(rows[0])
        r.update({"source": source, "target": "automod",
                  "evidence": f"name-embedding: 'automod' is a registered entity name "
                              f"inside '{source}'"})
        out.append(r)
    return "".join(json.dumps(r) + "\n" for r in out)


def _verify_bullet(section: str) -> str:
    """The bullet that states the readings an apply leaves — the file's verification step."""
    return _one([b for b in _bullets(section) if "12 + N" in b],
                "bullet stating the readings an apply actually leaves (12 + N)")


def test_ps_4_is_ruled_and_a_reprint_of_its_pairs_is_covered_not_novel(tmp_path):
    """#3065 clause 1: `RULED` gains PS-4, so Step 2.5 stops re-printing a decided set.

    `RULED` records what has been RULED, and a set whose edges are live belongs in it for
    exactly the reason PS-2 and PS-3 do: without the key the set-check answers `covered_by=none`
    for a pair set a human has already settled, which is the filing trigger. That is not
    hypothetical — Step 2.5's run of 2026-10-20T084644Z printed these two pairs and filed
    #3060, which then ran to the same apply this record already describes. The severity today is
    low because the live edges themselves keep both entities out of the degree-zero pool, so a
    reprint needs one of those edges to expire first; the missing key is what makes that reprint
    a new decision rather than a covered one.

    The pairs must be EXACTLY these two. A key carrying a third pair would answer
    `covered_by=['PS-4']` for proposals nobody ruled on, which is a fabricated authorization —
    the failure #2829's over-inclusion list is the precedent for.
    """
    section = _step_2_5_section(_skill_text())
    script = _setcheck_script(section)
    ruled = _ruled_dict(script)
    assert "PS-4" in ruled, (
        f"RULED holds {sorted(ruled)}: PS-4's two edges have been live in the store since "
        "2026-10-20T09:15:39Z, so the next run reprinting that pair set still reads "
        "covered_by=none and files a carrier for a decision already made")
    assert ruled["PS-4"] == PS_4_PAIRS, (
        f"PS-4 reads {sorted(ruled['PS-4'])}, expected exactly {sorted(PS_4_PAIRS)} — the two "
        "pairs Step 2.5 printed at 2026-10-20T084644Z and nothing else")
    assert ruled["PS-4"].isdisjoint(ruled["PS-3"]), (
        "PS-4 shares a pair with PS-3, so covered_by= would name two sets for one pair and "
        "the smallest-covering-set answer this snippet is designed to give stops meaning "
        "anything")

    state, instant = _execution_claim(section, "PS-4", _live_name_embedding_pairs()[1])
    assert state == "executed" and instant == PS_4_APPLY_INSTANT, (
        f"PS-4's RULED marker declares {state}" + (f" at {instant}" if instant else "") +
        f": the store holds both of its pairs, written by the apply of "
        f"{PS_4_APPLY_INSTANT}, and a marker that denies that invites a re-apply of live "
        "edges while a bare `EXECUTED` with no instant cannot be attributed to a backup")

    out = _run_setcheck(script, "ps4-two-pair", _ps4_pair_payload(), tmp_path)
    assert out.endswith("pairs=2 novel=none covered_by=['PS-4']"), (
        f"the snippet reads a reprint of PS-4's pairs as {out!r}; `novel=` naming a pair is "
        "the filing trigger, and `covered_by=none` for a settled set is what filed #3060")

    if PS_4_PROPOSALS.is_file():
        real = PS_4_PROPOSALS.read_text(encoding="utf-8")
        assert _pairs_in(real) == PS_4_PAIRS, (
            f"{PS_4_PROPOSALS} prints {sorted(_pairs_in(real))}, not the two pairs PS-4 rules "
            "— either this set drifted or the node is reading the wrong run's artefact")
        out = _run_setcheck(script, "ps4-runtime", real, tmp_path)
        assert out.endswith("pairs=2 novel=none covered_by=['PS-4']"), (
            f"the snippet over the real {PS_4_PROPOSALS.name} reads {out!r}: a reprint of "
            "that file must come back covered")


def test_a_ruled_marker_is_weighed_against_its_own_key_instant_not_one_constant():
    """#3065 clause 2: with two applies on the record, one constant cannot weigh both markers.

    The agreement node used to compare every key's declared instant against `APPLY_INSTANT`,
    the 2026-10-19 write. That was true of every key then and stopped being true the moment
    #3060's apply landed: `PS-4`'s edges went in 16h31m later, so the honest marker
    `EXECUTED 2026-10-20T09:15:39Z` fails a comparison against the older date, and the only
    edit that would satisfy it is the false one — stamping a set with the instant of the apply
    that did not write it, which then points a reader at
    `store-backups/kg-stranded-20261019T164420Z.sqlite` for edges that are not in it.

    Pinned three ways: the table's own values, the two instants differing (a single-constant
    comparison could only ever describe the earlier apply), and the node's source reading the
    per-key entry rather than the module constant.
    """
    import inspect

    assert APPLY_INSTANTS["PS-2"] == APPLY_INSTANTS["PS-3"] == "2026-10-19T16:44:20Z", (
        f"PS-2 and PS-3 are weighed against {APPLY_INSTANTS}: both sets were written by the "
        "one attended apply of 2026-10-19T16:44:20Z, whose backup is "
        "`store-backups/kg-stranded-20261019T164420Z.sqlite`, and moving this value would "
        "make their true markers read as lies")
    assert APPLY_INSTANTS["PS-4"] == "2026-10-20T09:15:39Z", (
        f"PS-4 is weighed against {APPLY_INSTANTS['PS-4']}, but #3060's carrier was applied at "
        "2026-10-20T09:15:39Z — receipt ACTIVE_NAME_EMBEDDING=21 PAIRS_LIVE=2/2 MISSING=0, "
        "exit 0")
    assert PS_4_APPLY_INSTANT != APPLY_INSTANT, (
        "the two applies have collapsed onto one instant: a per-key table with one value is "
        "the module constant again, and the node it feeds cannot tell PS-4's marker from PS-3's")

    src = inspect.getsource(
        test_every_ruled_key_besides_ps_1_declares_a_state_the_store_agrees_with)
    assert "APPLY_INSTANTS[key]" in src, (
        "the agreement node is back to comparing every ruled key against one constant, which "
        "is the comparison that cannot hold across two applies")
    assert "instant == APPLY_INSTANT," not in src, (
        "the bare-constant comparison survived inside the node that reads the table")


# ── the two figures the record states about itself (#4480) ───────────────────
#
# Both of these numbers have a source outside the prose: the store decides one, the `RULED`
# literal decides the other. Writing either down in this file is what filed #4480 — vault
# commit `86b1eeeca` (#4352) corrected the skill's present-tense count from 21 to 26 on
# 2026-11-19, a vault-surface landing cannot edit a file under `~/lloyd`, and the pinned 21
# here stayed: every `tests` rung since reproduced two red nodes with no code bug behind them.
# A pinned present-tense figure is the SAME claim held by two owners and only one of them gets
# corrected, so the expected values below are measured and any figure but the measured one is
# the failure. That is also what keeps the nodes able to fail: a checker that cannot see a
# stale number is a photograph of today, not a guard, which is why the falsifier node feeds
# both checkers readings this record has actually carried.

#: Every shape in which the record states today's `name-embedding:` count as a bare number, with
#: the site that carries it: the `pairs=0` label branch and SD-stranded-1's scope settle are the
#: two present-tense sentences, the Step 2.5 purpose paragraph states it as `now returns **N**`,
#: and the pilot bullet states it as `today's N live …`. There are four sites because a store
#: count stated in prose and the same count pinned in this file are ONE claim with TWO owners,
#: and #4480 is what that costs: the two applies of 2026-11-19 moved the store from 21 to 26 and
#: only the prose had an owner, so the pinned half went red while the false half stayed green.
#: Every site here is therefore graded against `_live_name_embedding_pairs()`, never a literal.
#:
#: The patterns deliberately span only these shapes. The readings that must KEEP their numbers are
#: past tense (`the store read 12 active`) or carry a qualifier no pattern here reaches
#: (`Measured 2026-10-20T10:05Z: the live store reads 21 active …`), so a dated reading can never
#: be swept up by a check on today's.
STORE_COUNT_SITES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("the `pairs=0` label branch",
     re.compile(r"the store at (\d+) active `name-embedding:` edges")),
    ("SD-stranded-1's scope settle",
     re.compile(r"the store reads (\d+) active `name-embedding:` edges")),
    ("the Step 2.5 purpose paragraph",
     re.compile(r"now returns \*\*(\d+)\*\*")),
    ("the pilot bullet",
     re.compile(r"today's (\d+) live `name-embedding:` edges")),
)

#: The sites' names, for a message that has to list what is missing.
SITE_NAMES: tuple[str, ...] = tuple(label for label, _ in STORE_COUNT_SITES)

#: Any count the record states as "N ruled pairs", spelled or in digits, so a miscount written
#: as a word is as catchable as one written as a number.
RULED_PAIR_FIGURE_RE = re.compile(r"(?i)\b([a-z]+|\d+) ruled pairs\b")

_WORD_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
              "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
              "seventeen", "eighteen", "nineteen")
_WORD_TENS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def _number_word(n: int) -> str:
    """Spell `n` (0-99) the way the record spells its counts: 13 → "thirteen".

    The section states its arithmetic in words, so a derived figure has to be spelled before
    it can be compared with the prose. Outside 0-99 this raises rather than compare against a
    string the record would never use.
    """
    assert 0 <= n < 100, f"{n} is outside the range _number_word spells"
    if n < 20:
        return _WORD_ONES[n]
    ten, one = divmod(n, 10)
    return _WORD_TENS[ten - 2] + (f"-{_WORD_ONES[one]}" if one else "")


def _stale_store_readings(text: str, count: int) -> list[str]:
    """Every site in `text` whose stated count is not `count`, named and quoted."""
    flat = " ".join(text.split())
    return [f"{label} states {found.group(0)!r}"
            for label, pattern in STORE_COUNT_SITES
            for found in pattern.finditer(flat)
            if found.group(1) != str(count)]


def _sites_carrying(text: str, count: int) -> set[str]:
    """Which of the four sites state exactly `count`, so a reworded site is caught as missing."""
    flat = " ".join(text.split())
    want = str(count)
    return {label for label, pattern in STORE_COUNT_SITES
            if want in {found.group(1) for found in pattern.finditer(flat)}}


def _wrong_ruled_pair_figures(text: str, want: int) -> list[str]:
    """Every "N ruled pairs" in `text` whose value is not `want`, quoted back."""
    spelled, digits = _number_word(want), str(want)
    return [f"{fig} ruled pairs" for fig in
            RULED_PAIR_FIGURE_RE.findall(" ".join(text.split()))
            if fig.lower() != spelled and fig != digits]


def _ruled_pair_total(script_section: str) -> int:
    """How many distinct pairs `RULED` enumerates, counted from the literal itself.

    `PS-1` ⊂ `PS-2` ⊂ `PS-3` and contributes no pair of its own, so the union — not a sum over
    keys — is the figure the record's arithmetic sentence has to carry.
    """
    ruled = _ruled_dict(_setcheck_script(_step_2_5_section(script_section)))
    return len(set().union(*ruled.values())) if ruled else 0


def test_no_present_tense_reading_claims_nineteen_active_edges_and_the_pasts_stay():
    """#3065 clause 3, re-anchored by #4480 and extended by #4352 clauses 1-3: every place the
    record states today's count must equal the store's own count, and every past reading keeps
    its number.

    Four sites state today's count, and `STORE_COUNT_SITES` holds them: "the store at N active
    `name-embedding:` edges" (Step 2.5's `pairs=0` label branch), "the store reads N active
    `name-embedding:` edges" (SD-stranded-1's scope settle), "now returns **N**" (the Step 2.5
    purpose paragraph) and "today's N live `name-embedding:` edges" (the pilot bullet). N was 19
    until the apply of 2026-10-20T09:15:39Z and 21
    until the two applies of 2026-11-19 — `PS-9`'s four pairs at 01:03:06Z and `PS-10`'s one at
    15:46:35Z — after which the store measured 26, and 28 after `PS-12`'s apply of
    2026-11-26T17:35:28Z. This node asserted 21 as a literal, #4352
    moved the vault to 26, and no vault landing can reach this file: that is #4480, and it is
    why the expected figure is now read through `_live_name_embedding_pairs()` instead of
    written down. A present-tense count in prose and a present-tense count pinned here are one
    claim with two owners.

    The arithmetic is attributed per set, and the enumerated ruled pairs are SIXTEEN — `PS-3`'s
    seven, `PS-4`'s two, `PS-9`'s four, `PS-10`'s one and `PS-12`'s two — because `PS-1` ⊂ `PS-2`
    ⊂ `PS-3` and `PS-11` ⊂ `PS-12` contribute no pair of their own. `PS-10`'s one is in that sum only because the key is in
    `RULED`: #4519 added it, and until then the record enumerated thirteen while its own prose
    named a fourteenth pair that had been ruled and applied — the figure and the dict moved
    together in one landing, which is why neither could be corrected on its own. That figure is derived from the
    `RULED` literal for the same reason the store figure is derived from the store. This
    replaces three literal bans (`ten pairs`, `10 pairs`, `ten ruled pairs`), each naming one
    wrong figure; `_wrong_ruled_pair_figures` refuses every figure but the derived one, in
    digits or in words, so it is not a narrower check.

    Clause 3 of #4352 is graded in the same node because it is the same claim in a different
    shape: the Step 2.5 purpose paragraph states today's count as `now returns **N**`, and it has
    to name the applies that wrote it — including `PS-9`'s of 2026-11-19T01:03:06Z, the one the
    record did not carry when the item was filed. The node does not pin HOW MANY applies there
    have been: that count moves with the next apply exactly as the store count moved, and `RULED`
    cannot yield it (its keys count settled SETS, not writes — `PS-2` and `PS-3` name the one apply
    of 2026-10-19T16:44:20Z and `PS-1` names none),
    so a pinned apply-count would be the next figure of this class rather than the last.

    What must NOT move is the history and the receipt. `ACTIVE_NAME_EMBEDDING=19 PAIRS_LIVE=7/7
    MISSING=0` is #3011's probe's measured output on 2026-10-19, "exits 0 only at 19 active
    `name-embedding:` edges" is that probe's own pass condition, and `len(rows)>=19` is the live
    receipt's floor: a floor is not a statement about today's count, and rewriting it would
    rewrite the command owed.py runs to decide whether a carrier may close. The four dated
    readings keep their numbers too — 12 at 2026-10-19T00:35Z, 19 at 2026-10-20T10:05Z, 21 at
    2026-10-24T04:26Z and the `PS-4` transition "19 → 21".
    """
    text = _skill_text()
    count, _live = _live_name_embedding_pairs()
    n_ruled_pairs = _ruled_pair_total(text)

    stale = _stale_store_readings(text, count)
    assert not stale, (
        f"{SKILL_PATH} states today's store count as {stale}, but the store measures {count} "
        "active `name-embedding:` edges through `app.kg_store` right now "
        f"(measured at {LIVE_STORE}): a present-tense store count is read as today, and a "
        "false one is worse than none because it closes the report that would have caught it")
    quiet = _sites_carrying(text, count)
    assert quiet == set(SITE_NAMES), (
        f"the sites carrying the measured {count} are {sorted(quiet)}, not all four: "
        f"{[n for n in SITE_NAMES if n not in quiet]} is missing it — a figure the node cannot "
        "see is a figure the node cannot keep honest, and a half-corrected record is stale prose")

    # Clause 3, amended: the Step 2.5 purpose paragraph carries today's count as `now returns
    # **N**` and names the applies that wrote it. `PS-9`'s apply must be named there with its
    # instant, because it is the one the record did not carry when this item was filed. What the
    # node does NOT pin is how many applies there have been: that count moves with the next
    # apply exactly as the store count did, and `RULED` cannot yield it — its keys count settled
    # SETS, not writes, since `PS-2` and `PS-3` name one apply and `PS-1` names none — so no
    # measurement here could tell a true 'five' from a false 'six'. Recorded as a finding on #4352.
    purpose = _step_2_5_purpose(text)
    assert f"now returns **{count}**" in " ".join(purpose.split()), (
        "the Step 2.5 purpose paragraph does not state the measured count: clause 3 asks it to "
        f"say `now returns **{count}**`, the figure `_live_name_embedding_pairs()` returns")
    purpose_flat = " ".join(purpose.split())
    assert "`PS-9`" in purpose_flat and PS_9_APPLY_INSTANT in purpose_flat, (
        f"the Step 2.5 purpose paragraph must name `PS-9`'s apply at {PS_9_APPLY_INSTANT} among "
        "the applies that wrote the count; a reader who cannot see which applies are being "
        "totalled cannot check the total")

    assert _wrong_ruled_pair_figures(text, n_ruled_pairs) == [], (
        f"{SKILL_PATH} states the enumerated ruled-pair count as "
        f"{_wrong_ruled_pair_figures(text, n_ruled_pairs)}, but `RULED` enumerates "
        f"{n_ruled_pairs} distinct pairs")
    flat_all = " ".join(text.split())
    assert (f"{_number_word(n_ruled_pairs)} ruled pairs" in flat_all
            or f"{n_ruled_pairs} ruled pairs" in flat_all), (
        f"neither \"{_number_word(n_ruled_pairs)} ruled pairs\" nor "
        f"\"{n_ruled_pairs} ruled pairs\" is in the record: the arithmetic has to be stated, "
        "not merely free of a wrong figure")

    section = _sd_stranded_1_section(text)
    flat = " ".join(section.split())
    assert "`PS-3`'s seven" in flat and "`PS-4`'s two" in flat and "`PS-9`'s four" in flat, (
        "the count has to be attributed to the applies that wrote it — PS-3's seven pairs at "
        "2026-10-19T16:44:20Z, PS-4's two at 2026-10-20T09:15:39Z and PS-9's four at "
        "2026-11-19T01:03:06Z — or a reader cannot tell which backup holds which edges")

    assert "ACTIVE_NAME_EMBEDDING=19 PAIRS_LIVE=7/7 MISSING=0" in section, (
        "the 2026-10-19 measured form of #3011's probe is gone: it is the past-tense record of "
        "what the PS-3 apply left, and deleting it leaves only the today figure")
    assert "exits 0 only at **19 active `name-embedding:` edges**" in flat, (
        "the receipt's own pass condition stopped saying 19: that number is the probe's floor "
        "(12 pilot + PS-3's 7), not a claim about today, and the probe is run by owed.py")
    assert "len(rows)>=19" in section.replace(" ", ""), (
        "the live receipt's floor was edited: it is a floor, still true at 26, and it is the "
        "reading that decides whether a carrier may close")


def test_the_figure_checkers_refuse_a_stale_count_and_a_miscounted_set():
    """Both derived figures above can fail, on readings this record has actually held.

    The positive control first: the live skill agrees with the live store and with its own
    `RULED`, or the refusals below would be proving that the checkers always raise. Then four
    wrong readings, each wrong in one named way — the 21 the vault carried until #4352 (the
    figure that red-coded #4480), that same 21 left in ONE sentence of a two-sentence record
    (the half-corrected case the message has to name), a ruled-pair count that counts
    `RULED` read one pair higher than its own keys, and the same count written in digits when
    the prose spells it.
    """
    text = _skill_text()
    count, _live = _live_name_embedding_pairs()
    n_ruled_pairs = _ruled_pair_total(text)
    assert _stale_store_readings(text, count) == [], (
        "the live skill already disagrees with the live store, so this node cannot use the "
        "live skill as its positive control — fix the clause node's subject first")
    assert _wrong_ruled_pair_figures(text, n_ruled_pairs) == [], (
        "the live skill already misstates its own RULED count — same reason")
    assert _sites_carrying(text, count) == set(SITE_NAMES), (
        f"the live record's sites carrying {count} are {sorted(_sites_carrying(text, count))}, "
        f"not all four — positive control, so the mutations below cannot be blamed on a site "
        "the record stopped carrying")

    drifted = " ".join(text.split()).replace(
        f"the store at {count} active", f"the store at {count - 5} active")
    assert drifted != " ".join(text.split()), (
        f"the record does not carry `the store at {count} active` — the mutation this node "
        "needs cannot be built, so the checker's ability to refuse is unproven")
    refused = _stale_store_readings(drifted, count)
    assert refused == ["the `pairs=0` label branch states "
                       + repr(f"the store at {count - 5} active `name-embedding:` edges")], (
        f"a present-tense count five edges behind the store came back as {refused!r}: the "
        "stale-record direction is unguarded, which is exactly how #4480 happened")

    half = " ".join(text.split()).replace(
        f"the store reads {count} active", f"the store reads {count - 7} active")
    assert half != " ".join(text.split()), "the second reading is not in the record"
    assert _stale_store_readings(half, count) == [
        "SD-stranded-1's scope settle states "
        + repr(f"the store reads {count - 7} active `name-embedding:` edges")], (
        "a record with one sentence corrected and one left behind passed: the "
        "half-corrected direction is unguarded")

    # Clause 3's site, mutated on its own: the two prose sentences corrected while the Step 2.5
    # purpose paragraph keeps the figure from before the applies. A checker over the whole file
    # would see four figures, three of them right, and pass.
    purpose_stale = " ".join(text.split()).replace(
        f"now returns **{count}**", f"now returns **{count - 5}**")
    assert purpose_stale != " ".join(text.split()), "`now returns **N**` is not in the record"
    assert _stale_store_readings(purpose_stale, count) == [
        "the Step 2.5 purpose paragraph states "
        + repr(f"now returns **{count - 5}**")], (
        "the Step 2.5 purpose paragraph kept a stale count unnoticed: that is the site clause 3 "
        "names, and the only figure in it that a store measurement can check")

    # The reworded-site direction: a site that stops stating the figure at all has to be caught
    # as MISSING, not waved through because no stale number was found.
    reworded = " ".join(text.split()).replace(
        f"today's {count} live `name-embedding:` edges",
        "today's live `name-embedding:` edges")
    assert reworded != " ".join(text.split()), "the pilot bullet does not state the count"
    assert _stale_store_readings(reworded, count) == [], (
        "dropping the number left a stale-looking fragment behind; the site pattern is matching "
        "something it should not")
    assert "the pilot bullet" not in _sites_carrying(reworded, count), (
        "a site that states no figure still counted as carrying it: every wrong figure would "
        "then pass, because only the measured one is ever compared")

    # The record states the ruled-pair figure twice — Step 2.5's post-apply watermark block and
    # SD-stranded-1's verification bullet — so the mutation lands on both and both must be
    # refused. A checker that flagged only the first would leave the second site unguarded.
    spelled = _number_word(n_ruled_pairs)
    over = " ".join(text.split()).replace(f"{spelled} ruled pairs",
                                          f"{_number_word(n_ruled_pairs + 1)} ruled pairs")
    assert over != " ".join(text.split()), "the ruled-pair figure is not spelled as expected"
    over_refused = _wrong_ruled_pair_figures(over, n_ruled_pairs)
    assert over_refused and set(over_refused) == {
        f"{_number_word(n_ruled_pairs + 1)} ruled pairs"}, (
        f"counting one pair more than `RULED` enumerates as a ruled pair came back as "
        f"{over_refused!r}: a spelled miscount is invisible, and that figure is what a reader "
        "takes as the authorised scope")
    n_sites = " ".join(text.split()).count(f"{spelled} ruled pairs")
    assert len(over_refused) == n_sites, (
        f"the record states the figure at {n_sites} sites but {len(over_refused)} refusals came "
        "back: a checker that stops at the first leaves the rest unguarded")
    assert n_sites == 2, (
        f"the figure is stated at {n_sites} sites, not the two this node documents — Step 2.5's "
        "watermark block and SD-stranded-1's verification bullet")

    digits = " ".join(text.split()).replace(f"{spelled} ruled pairs",
                                           f"{n_ruled_pairs} ruled pairs")
    assert _wrong_ruled_pair_figures(digits, n_ruled_pairs) == [], (
        "the same count written in digits was refused: the check is on the value, not on the "
        "style the prose happens to write it in")


def _text_beside(text: str, needle: str, radius: int = 200) -> list[str]:
    """Flat windows of `text` around each occurrence of `needle`, for claims stated in the
    same breath as a named artefact."""
    flat = " ".join(text.split())
    return [flat[max(0, hit.start() - radius):hit.end() + radius]
            for hit in re.finditer(re.escape(needle), flat)]


def _epoch_claims(window: str) -> set[str]:
    """Every bare 9-11 digit number in `window` — the shape a `stat -c %Y` epoch takes."""
    return set(re.findall(r"(?<![\d.])(\d{9,11})(?![\d.])", window))


def test_the_ps_9_rule_names_the_four_pairs_it_wrote_and_its_backup_proves_the_instant():
    """#4352 clause 5's provenance half: `PS-9`'s entry has to be checkable, not merely dated.

    The previous round made this clause true in the prose and the review still graded it PARTIAL,
    because nothing but the prose was checking it: the marker's instant was weighed against the
    store (every pair present, so it must be), but no test opened the artefact the instant is
    supposed to come from. An instant copied from a comment can only ever agree with the comment
    that copied it — that is the rule on `APPLY_INSTANTS`, and this is the node that applies it.

    So three things are checked against each other rather than quoted. `RULED`'s `PS-9` must
    enumerate exactly the four pairs carrier #3704 wrote — one key, the whole set, no more. The
    backup the comment names must exist under the store-backups directory, its name derived here
    from the marker's own instant so the two cannot drift apart. And the epoch the comment claims
    for it must be the epoch the filesystem reports for that file, read by `stat` and compared as
    a number: the comment says "matched by its `stat -c %Y` epoch", so an `ls`-shaped string that
    no later run could re-check is exactly what may not stand.

    The file itself is never opened, let alone ordered: at 200,245,248 bytes it sits over the
    WITNESS_MAX_BYTES ceiling, and #4352's own owed note says a whole copy must not be taken.
    `stat` reads the directory entry, which is the only thing this claim is about.
    """
    text = _skill_text()
    section = _step_2_5_section(text)
    ruled = _ruled_dict(_setcheck_script(section))
    assert "PS-9" in ruled, (
        f"`RULED` has lost the key for the 2026-11-19T01:03:06Z apply ({sorted(ruled)}); the four "
        "edges it wrote would start printing as fresh proposals again")
    assert set(ruled["PS-9"]) == set(PS_9_WRITTEN_PAIRS), (
        f"`PS-9` enumerates {sorted(ruled['PS-9'])} but the apply it names wrote "
        f"{sorted(PS_9_WRITTEN_PAIRS)}: a key holding a subset of its own write re-proposes the "
        "rest, and one holding more authorises pairs nobody ruled")

    _count, live = _live_name_embedding_pairs()
    state, instant = _execution_claim(section, "PS-9", {(s, t) for s, t in live})
    assert (state, instant) == ("executed", PS_9_APPLY_INSTANT), (
        f"`PS-9`'s marker declares {state} at {instant}, not executed at {PS_9_APPLY_INSTANT}, "
        "and `APPLY_INSTANTS` holds "
        f"{APPLY_INSTANTS.get('PS-9')!r}: the comment, the constant and the store have to agree")
    assert APPLY_INSTANTS["PS-9"] == PS_9_APPLY_INSTANT

    backup = LIVE_STORE.parent / "store-backups" / PS_9_BACKUP
    assert backup.is_file(), (
        f"{backup} is absent, so the instant `PS-9` declares has no artefact to be checked "
        "against — the pre-apply backup is the only witness that is not the claim itself")
    actual_epoch = int(backup.stat().st_mtime)
    windows = _text_beside(text, PS_9_BACKUP)
    assert windows, (
        f"nothing in {SKILL_PATH} names the backup {PS_9_BACKUP} by which its own apply instant "
        "is matched: a date with no artefact attached is a date nobody can re-check")
    for window in windows:
        assert "stat -c %Y" in window, (
            f"the entry names {PS_9_BACKUP} but does not say the instant is matched by "
            f"`stat -c %Y` — an `ls` string is how this record would come to state an instant "
            "no later run can reproduce")
        assert _epoch_claims(window) == {str(actual_epoch)}, (
            f"{PS_9_BACKUP} is quoted with epoch claim(s) {sorted(_epoch_claims(window))} while "
            f"the file reports {actual_epoch} "
            f"({datetime.fromtimestamp(actual_epoch, timezone.utc).isoformat()}): the comment's "
            "epoch and the filesystem's disagree, so one of them is a copy of a claim rather "
            "than a measurement")

    # The comparison has to bite. An epoch one second off is the smallest error this record could
    # carry and still look exactly as correct as it does now, so it is the one worth proving
    # against: if rewriting it were invisible, the node would be a photograph of the comment.
    tampered = windows[0].replace(str(actual_epoch), str(actual_epoch + 1))
    assert _epoch_claims(tampered) != {str(actual_epoch)}, (
        "moving the quoted epoch by one second left it agreeing with the artefact, which means "
        "the claim was never compared with it")


def test_the_pairs_zero_paragraph_records_the_measured_current_reading():
    """#4352 clause 5's other half: `proposed edges 0` is a branch a run can be in, never
    today's expected one.

    The paragraph under the `pairs=0` heading used to call an empty run "the expected one" over a
    store it stated as 21 active `name-embedding:` edges "so the residue is gone". The store
    disagrees with that reading, and so does this decision's own ledger: the three runs of
    2026-11-19 that fell between `PS-9`'s apply and `PS-10`'s — `011109Z`, `053554Z`, `100236Z` —
    each printed `proposed edges 1` over `agent-supervisord.service -> supervisord`, the pair
    carrier #4257 held as a pending `hand_run:`, and took `no-filing pending-hand-run-4257`. The
    candidate set is re-derived from Step 1's fresh mints every run, so `proposed edges` reports
    that run's extraction; the paragraph must say so and record the measured reading, while the
    `pairs=0` branch itself stays valid.
    """
    section = _sd_stranded_1_section(_skill_text())
    flat = " ".join(section.split())
    assert "and it is the expected one" not in flat, (
        "the `pairs=0` paragraph calls an empty run today's expected reading again; the set is "
        "re-derived every run, so no reading is standing")
    assert "residue is gone" not in flat, (
        "the paragraph claims the residue is gone as a present fact; the store's shape is a "
        "measurement, and `PS-10`'s pair was novel after the apply that sentence called final")
    for required in ("NOT today's expected reading", "`proposed edges 1`",
                     "`agent-supervisord.service -> supervisord`", "#4257",
                     "`no-filing pending-hand-run-4257`"):
        assert required in flat, (
            f"the paragraph must record the measured current reading ({required!r} is missing): "
            "three runs of 2026-11-19 printed `proposed edges 1` on this very ledger")
    assert "no-filing post-apply-empty" in flat and "printf" in flat, (
        "the `pairs=0` branch itself must survive: an empty set on a store whose `receipt:` probe "
        "exits 0 still records `no-filing post-apply-empty`")


def test_the_copy_reading_names_the_backup_method_and_says_what_a_lone_main_file_reads():
    """#3065 clause 4: copying `kg.sqlite` alone is not copying the store.

    The instruction used to be "a fresh read-only copy through `app.kg_store`", which an
    executor follows as `shutil.copy2` of the one file named. That reads the PRE-WRITE count and
    looks like a refutation of a write that succeeded: reproduced 2026-10-20T10:05Z on this box,
    where the live store read 21 active `name-embedding:` edges through `app.kg_store` and a
    copy of `kg.sqlite` alone — 159,481,856 bytes, with 8,157,632 bytes of uncheckpointed
    `kg.sqlite-wal` beside it — read 19. Both figures are that instant's, not today's, and the
    store measures more than either now: #4480 is about what a present-tense figure costs. The
    same trap was what #3060's triage hit three minutes
    after its own apply printed `applied 2 edges`.

    A stale snapshot that reads like a refutation is worse than no snapshot, because the action
    it invites is the write that is already done. So the step now names `KGStore.backup(dest_path)`
    — one required destination, no default — and says in the same breath what the shortcut reads.
    """
    section = _sd_stranded_1_section(_skill_text())
    bullet = _verify_bullet(section)
    assert "KGStore.backup(dest_path)" in bullet, (
        "the verification bullet still sends an executor to 'a fresh read-only copy' without "
        "naming the method that takes a snapshot coherently: `KGStore.backup(dest_path)`, "
        "`app/kg_store.py:412`")
    assert "kg.sqlite-wal" in bullet and "kg.sqlite-shm" in bullet, (
        "the bullet must name what a hand copy has to take beside the main file, or the "
        "alternative it offers is the trap it is warning about")
    assert "PRE-WRITE" in bullet, (
        "the sentence names the method but not the failure: a main-file-only copy reads the "
        "count from before the write, which reads as a refutation")
    assert "a fresh read-only copy through `app.kg_store`" not in section, (
        "the unsound instruction is still in the section verbatim, wherever the sound one was "
        "put beside it")


def test_the_backup_is_matched_by_epoch_and_an_empty_dry_run_disowns_being_a_test():
    """#3065 clause 5: the two readings that could not bear their weight, disarmed in place.

    **MTIME.** The step said to match the new backup against the pilot's by MTIME and never
    named a command, and the string form of mtime on this box is a trap: the system runs
    UTC−7, so `ls --time-style=+%Y-%m-%dT%H:%M:%SZ` prints LOCAL wall-clock wearing a literal
    `Z`. Measured 2026-10-20T10:05Z: `store-backups/kg-stranded-20261020T091539Z.sqlite` prints
    `2026-10-20T02:15:39Z`, seven hours BEFORE the apply that made it, while `stat -c %Y` gives
    1792487739, which `date -u -d @1792487739` converts to 2026-10-20T09:15:39Z — the filename's
    own instant. The two applies are 16h31m apart and their backups' epochs agree with that; only
    the printed strings invert. The pilot's file is the one
    whose name and mtime agree, because its stamp was written on the local clock — so the string
    comparison ranks the two backups correctly only by accident, and it can never tie either to
    the apply it belongs to. Hence: epoch, or the run's own printed
    `applied N edges; backup → …` line.

    **`proposed edges` 0.** The step said the next dry run prints 0 and calls that verification.
    It cannot be one: the candidate set is re-derived every run from Step 1's fresh mints, so the
    number reports today's extraction, not yesterday's write. `~/lloyd-data/_pipeline/`
    `stranded-standing-decision.log` printed `proposed=0` at 2026-10-19T195646Z,
    2026-10-20T001216Z and 2026-10-20T043126Z and then `proposed=2` at 2026-10-20T084644Z with
    no apply between the four — the reappearance was a fresh set, not a lost write. The sound
    reading is the pair-level one, the carrier's `hand_run.receipt`.
    """
    section = _sd_stranded_1_section(_skill_text())
    bullet = _verify_bullet(section)

    assert "stat -c %Y" in bullet, (
        "MTIME is still named without the command that measures it, which is how the string "
        "trap got into the step at all")
    assert "LOCAL wall-clock" in bullet and "literal `Z`" in bullet, (
        "the bullet must say why the ls string cannot be trusted: on this box it is local "
        "wall-clock carrying a Z that says UTC")
    assert "date -u" in bullet, (
        "the bullet names the epoch but not the conversion that ties it to the filename, so a "
        "reader cannot check the reading without knowing the offset")
    assert "applied N edges" in bullet, (
        "the printed line is the other sound route to the same backup and the step's first "
        "listed reading; it must survive the mtime rewrite")

    pilot = _one([b for b in _bullets(section) if "Authorization — the pilot" in b],
                 "authorization bullet for the pilot")
    assert "stat -c %Y" in pilot, (
        "the pilot's own artefacts are still to be matched by mtime with no command named, "
        "and its filename stamp is the local one — the worst case for a string comparison")

    lower = bullet.lower()
    assert "not a closure test" in lower, (
        "`proposed edges` 0 is still offered as the check that an apply worked; the standing "
        "log shows 0, 0, 0 then 2 with no apply between, so it reports the extraction")
    assert "hand_run.receipt" in bullet, (
        "the disarmament has to say what to read instead: the pair-level receipt, which asks "
        "this store for every pair of its own set at exact source+target")
