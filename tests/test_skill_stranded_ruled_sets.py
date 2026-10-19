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
it and pins nothing. The two nodes that exist to be able to fail are
`test_deleting_ps_2_reopens_the_sixth_carrier` and `test_the_stale_claim_detector_fires_on_the
_prose_it_replaces`, which run the checks against the pre-fix text.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from app.data_root import PRODUCTION_DATA_ROOT

ROOT = Path(__file__).resolve().parent.parent
BACKLOG_PY = ROOT / "scripts" / "automod" / "backlog.py"
THIS_FILE = "tests/test_skill_stranded_ruled_sets.py"

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


def _missing(path: Path) -> str:
    """Fail-loud, after NO_VAULT in tests/test_yaml_fix_skill_claims.py:96."""
    return (f"no vault file at {path}: these nodes police that file, so they report "
            "rather than skip")


def _skill_text() -> str:
    assert SKILL_PATH.is_file(), _missing(SKILL_PATH)
    return SKILL_PATH.read_text(encoding="utf-8", errors="replace")


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


# ---------------------------------------------------------------- clause 1: PS-2 in RULED

def test_ruled_holds_ps_2_with_the_six_pairs_the_dump_printed():
    """`RULED` gained PS-2, and `RULED` means RULED — not executed."""
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
    # Ruled, not executed: the dict comment carries the ruling instant and the pending apply,
    # so a reader cannot mistake PS-2 for work already written.
    ps2_comment = section.split('"PS-2"')[0].rsplit("# PS-2", 1)
    assert len(ps2_comment) == 2, "PS-2's own comment line is gone from the RULED block"
    mark = ps2_comment[1]
    assert "NOT EXECUTED" in mark.upper() and "2026-10-18T23:41" in mark, (
        f"PS-2's RULED comment says {mark[:120]!r}: it must carry the ruling instant "
        "2026-10-18T23:41 and that the apply is still pending, as PS-1 sat unapplied")


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


def test_the_witness_ceiling_cite_resolves_in_the_tree_it_names():
    """`backlog.py:1240` was 28 lines off; the cite must land on the constant it names."""
    section = _step_2_5_section(_skill_text())
    cites = set(RULED_CITE_RE.findall(" ".join(_witness_paragraph(section).split())))
    assert cites == {"1268"}, (
        f"the witness bullet cites backlog.py {sorted(cites)}; WITNESS_MAX_BYTES is the "
        "constant at scripts/automod/backlog.py:1268 and :1240 is not")
    assert RULED_CITE_RE.findall(" ".join(section.split())) == ["1268"], (
        f"a second backlog.py line cite appeared in the Step 2.5 section: "
        f"{RULED_CITE_RE.findall(' '.join(section.split()))}")
    lines = BACKLOG_PY.read_text(encoding="utf-8").splitlines()
    assert "WITNESS_MAX_BYTES = 1_048_576" in lines[1268 - 1], (
        f"scripts/automod/backlog.py:1268 is {lines[1268 - 1]!r}, not the "
        "WITNESS_MAX_BYTES constant the skill cites — move the cite, not the constant")
    size_check = [i + 1 for i, l in enumerate(lines) if "size <= WITNESS_MAX_BYTES" in l]
    assert size_check, (
        f"no `size <= WITNESS_MAX_BYTES` check in {BACKLOG_PY}: the skill's companion "
        "cite to the check that applies the ceiling has nothing to point at")
    assert f":{size_check[0]}" in _witness_paragraph(section), (
        f"the skill cites the applying check as something other than :{size_check[0]}")


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
    file a ninth carrier. PS-3 is marked RULED, NOT EXECUTED, in PS-2's shape, because the
    apply is still owed: measured 2026-10-19T09:51Z the store reads 12 active
    `name-embedding:` edges and 0 live edges for every one of the seven pairs."""
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
    mark = section.split('"PS-3"')[0].rsplit("# PS-3", 1)
    assert len(mark) == 2, "PS-3's own comment line is gone from the RULED block"
    assert "NOT EXECUTED" in mark[1].upper(), (
        f"PS-3's RULED comment says {mark[1][:120]!r}: a ruled set whose apply is owed has "
        "to say so, which is the sentence PS-1 sat without for eleven days")

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
