"""Witness artefacts: the clause that orders a live store's bytes copied into the
vault, the dated name those bytes go to, and the anchor witness a round commits.

Moved out of `scripts/automod/backlog.py` (2026-11-23, the third cut of the split
assessed in `architecture/backlog.md`): `_witness_artifact` and the naming rules
around it (`_RETIRED_WITNESS_NAMES`, `_ROLLING_STATE_NAMES`, the dated-sibling
probes), `commit_anchor_witness` and its `ANCHOR_WITNESS_*` budget,
`_over_bound_extract_demand`, and `add_witness_artifact_clause`. The witness
constants above that block — `WITNESS_EXTS`, `WITNESS_PATH_RX`,
`WITNESS_ARTIFACT_DIR`, `WITNESS_MAX_BYTES` — and the helpers `_rederive_command`,
`_witness_within_size_bound` and the digest readers stay in `backlog`: prose in
this repo and in the vault's skills cites their lines there by absolute number.
The bodies are unchanged except that every reference to a backlog name — this
module's own names included — reads `B.<name>`, so a `monkeypatch.setattr(backlog,
"_ROLLING_STATE_NAMES", ...)` reaches this code as it did before. The foot of this
module binds every name it defines back onto `backlog`.
"""
from __future__ import annotations

from pathlib import Path
import json
import re

from scripts.automod import backlog as B


#: A witness name belongs to its first occupant, and this generator was the thing
#: destroying them. `backlog/data/gate.json` held #1883's pinned witness for round
#: `SM_20260930_063800` (vault `cfe9a113`, 34 newline characters / 1,143 bytes).
#: Items #2228 and #2383 were each handed that same path by a clause like the one
#: below and stopped, writing a dated sibling instead (`2026-10-05.2228-gate-witness.json`,
#: and the node `test_the_2383_gate_witness_is_a_dated_sibling_because_gate_json_was_taken`
#: is that round's record of it). On 2026-10-10 nobody stopped: #2512's triage emitted
#: "copy `~/.local/state/lloyd-automod/rounds/SM_20261010_204036/gate.json` to
#: `backlog/data/gate.json`" — the artifact name was the SOURCE file's basename — the
#: round obeyed it at vault `d5f9f542`, and #1883's witness was gone, which is what
#: turned three tests on main red (#2513). A witness is an immutable extract of one
#: moment, so a name already on disk is taken: the two demands now name a dated
#: sibling instead of ordering a clobber.
def _vault_witness_dir(artifact_dir: str) -> Path:
    """`artifact_dir` under the vault root, read live so a test can move the vault.

    Read through the `app.paths` MODULE at call time, never captured into a default
    argument: the whole value of this probe is that a test can point it at a
    temporary vault, and a name bound at `def` time cannot be monkeypatched.
    """
    from app import paths
    return Path(paths.VAULT_ROOT) / artifact_dir


def _artifact_taken(artifact_dir: str, name: str) -> bool:
    """True when the vault already holds bytes at `artifact_dir/name`.

    An `OSError` (no such directory, unreadable tree, a round run with `HOME`
    pointed away from the vault) answers "not taken", which is the pre-fix
    behaviour: this probe chooses a NAME and must never become a reason the demand
    goes silent. Under the gate, where `~/obsidian` does not exist, every name
    therefore reads free — and that is the same graceful-degradation shape the
    vault-reading test nodes describe: corroboration when the vault is there, no
    veto when it is not.
    """
    try:
        return (B._vault_witness_dir(artifact_dir) / name).is_file()
    except OSError:
        return False


#: Vault slots a witness name may never be aimed at again, keyed on BASENAME.
#:
#: `promotions.jsonl` was the ledger mirror until #2050 owed 5 retired the
#: working-tree copy and #2054 re-aimed the figures at vault history, so every one of
#: the five figures now names the live store. #2064 then deleted its 28,678 rows from
#: the vault (vault `ffc04ce5`) and #2243 moved the last 44 witness rows that had
#: re-landed there to a dated name (vault `7786c598`), which leaves the path absent from
#: HEAD — and three assertions in `tests/test_failure_ledger_witness.py` (`:122`,
#: `:326`, `:471`) keep it absent from the working tree. It is therefore a name the
#: vault must not hold, while `_artifact_taken` can only ever answer "free" for it: the
#: two facts together are what made an over-bound extract demand against the live ledger
#: order the retired slot (#2444 owed 1, measured 2026-10-11), and obeying that order is
#: what broke a rail three times before it — #2175→#2178, #2383→#2387 and #2385→#2523,
#: each a whole follow-up item.
#:
#: KEYED ON THE BASENAME, never on the artifact directory joined to it: that literal
#: path is allowed in exactly eight files under `tests/` and `scripts/` — the allowlist
#: `_MIRROR_MENTION_FILES` in
#: tests/test_automod_vault_round.py::test_no_reader_under_tests_or_scripts_opens_the_retired_mirror
#: — and `scripts/automod/backlog.py` is not one of them, so spelling the path here would
#: turn that rail red on the file whose job is to stop the demand.
# Two names, two different reasons, one ban on the bare `backlog/data/` slot:
#   * `promotions.jsonl` was COMMITTED and then deleted (#2064 took its rows out),
#     and three assertions in `tests/test_failure_ledger_witness.py` keep it
#     absent — retiring it is closing a door that was already shut.
#   * `landed_outcomes.jsonl` was never committed at all: it is the live rolling
#     append-only journal under the automod state dir, growing a row a day. Its
#     bytes sit under the witness bound, so `_artifact_taken` would call the bare
#     slot free and hand out a demand for `backlog/data/<basename>` — an UNDATED
#     snapshot of a file that keeps moving, whose own `wc -l -c` re-derivation
#     goes false the day the journal gains a row. Here the ban is what forces the
#     cut date into the name, and the date is the witness.
# The set is keyed on BASENAME, and `landed_outcomes.jsonl` was the second member
# for a reason that is NOT retirement — it is a live rolling store. A third such
# basename arrived on 2026-10-26 (`data_watch.json`, #3398), which is the shape this
# comment predicted: the rule is no longer "a list of retired names" but "a live
# rolling store's vault copy is always dated". That rule now lives in
# `_ROLLING_STATE_NAMES` below, and this set keeps only its own judgement — a slot the
# vault must never hold again, which `_artifact_taken` structurally cannot see.
_RETIRED_WITNESS_NAMES = frozenset(
    {"promotions.jsonl", "landed_outcomes.jsonl"})


#: Rolling state sources, keyed on BASENAME exactly as the retired ban above is: the
#: files a live process REWRITES IN PLACE, so bytes copied out of one are a snapshot of
#: something that keeps moving. `_artifact_taken` calls every one of them a free slot,
#: and the bare name it then hands out is a demand for a witness that stops matching
#: its own source on the next write — the `wc -l -c < backlog/data/data_watch.json`
#: re-derivation an owed note orders goes false at the next persist
#: (`PERSIST_EVERY_SECONDS = 60.0`, agent-services/guardian/vaultwatch.py:66, fired at
#: :530). Measured while triaging #3398: the live `data_watch.json` read 626 bytes at
#: 11:57, 625 at 11:59 and 626 again at 12:00, while `backlog/3095-…md:90` and
#: `backlog/3153-…md:119` were already publishing an UNDATED demand for that slot and
#: `wc -l -c` figures for it. Putting the day in the NAME is what makes the claim
#: permanent: the date is the moment the snapshot was cut.
#:
#: KEYED ON A DECLARED BASENAME, never on a "sits under a state dir" pattern.
#: `tests/test_witness_artifact_clause.py` pins `state/qmd/index.db` and
#: `state/lloyd-automod/sessions.jsonl` as BARE names under the same free probe, and
#: `sessions.jsonl` is a rolling store in real life — a directory heuristic would
#: silently rewrite both, which is why membership here is a declaration, not a
#: location. Each member is therefore named by the line that declares it:
#:   `data_watch.json`          datawatch.py:1275 (`state_file`, the data watcher)
#:   `vault_watch.json`         vaultwatch.py:58 (`STATE_FILE`, the vault watcher)
#:   `data-root-arrivals.json`  datawatch.py:1098 (`ARRIVALS_STATE_FILE`)
#:   `heartbeat.json`           policy.py:696 (`HEARTBEAT_NAME`), written every loop at guardian.py:570
#:   `logcursors.json`          guardian.py:463 (`logtail.LogCursor(self.gdir / …)`)
#:   `signatures.json`          guardian.py:1028 (the chronic-error cache, re-written when it refreshes)
#:   `voice.json`               speak.py:126 (`CONFIG_NAME`), re-written into the state dir at every guardian stage by bin/sync-voice-config.py:28
#:   `conversation-relation-proposals.json`  conversation_relations.py:45 (`PROPOSALS_FILE`), rewritten by `save_proposals` at :807 — under `commit_lock` and via `atomic_write_text` at :856, which replaces the same path — on every #51 run, so the file's counts move nightly
#:
#: The guardian cannot be IMPORTED for this (`agent-services/guardian/` has no
#: `__init__.py`; `scripts/` reads those files by path only, as promote.py:533 does), so
#: the registry is declared here and its members are traceable to the lines above. A
#: fifth store is dated by adding ITS basename to this set: the banned-name set above is
#: a different judgement, and
#: `tests/test_witness_artifact_clause.py::test_a_fifth_rolling_source_dates_without_touching_the_ban`
#: is the rail that keeps the two from collapsing back into one hand-list.
_ROLLING_STATE_NAMES = frozenset({
    "data_watch.json",
    "vault_watch.json",
    "data-root-arrivals.json",
    "heartbeat.json",
    "logcursors.json",
    "signatures.json",
    "voice.json",
    "conversation-relation-proposals.json",
})


def _witness_name_must_be_dated(name: str) -> bool:
    """True when a witness copied from a source named `name` may not take the bare slot.

    Two reasons, one predicate, so `_witness_artifact` has a single question to ask:
    the vault must never hold this path again (`_RETIRED_WITNESS_NAMES`, #2560/#2879),
    or the source is a file a live process rewrites and an undated copy of it is a
    figure that goes false on the next write (`_ROLLING_STATE_NAMES`, #3398). Both
    fire INDEPENDENT of `_artifact_taken`: for a retired slot the probe is always wrong
    in the safe-looking direction, and for a rolling one it is right about the slot and
    wrong about the bytes.
    """
    return name in B._RETIRED_WITNESS_NAMES or name in B._ROLLING_STATE_NAMES


def _witness_artifact(source, artifact_dir: str = B.WITNESS_ARTIFACT_DIR, *,
                      taken=None, day: str = "", item_id=None) -> str:
    """The vault path witness bytes should go to — never a path it would clobber.

    The free-name case is unchanged and stays the bare basename #1756's convention
    looks like on disk (`backlog/data/iv-metrics.jsonl`), so the overwhelmingly
    common demand is not rewritten into a dated monster. Two things move a name into
    the dated-sibling form the two items that hit this by hand settled on —
    `<date>.<item>-<source directory>-<name>`, e.g.
    `backlog/data/2026-10-10.2512-SM_20261010_204036-gate.json`: a name the vault
    already holds (#2513), and a basename `_witness_name_must_be_dated` rules out —
    either a retired slot (#2560) or a rolling state file a live process rewrites
    (#3398) — which fires even when the free-name probe says the slot is empty, because
    for a retired slot the probe is always wrong in the safe-looking direction, and for
    a rolling one it reads the SLOT correctly and the BYTES wrongly. The source
    directory is what keeps two items copying two DIFFERENT `gate.json` files on
    one day apart; the item id is what makes the name readable from the item that
    owes it, and it is optional because the emitter is also called by tests and by
    owed-check paths that have no item in hand. Re-triaging the same source is
    idempotent — same day, same directory, same id — so the demand keeps pointing
    at the bytes its own last run wrote rather than spawning a fresh copy each pass,
    and for a retired name that is the whole fix: the second demand asks for the
    extract the first one published, not a second copy of it.
    """
    src = B._witness_target(source)
    name = src.name
    probe = taken if taken is not None else B._artifact_taken
    if not B._witness_name_must_be_dated(name) and not probe(artifact_dir, name):
        return f"{artifact_dir}/{name}"
    day = day or B.now_stamp()[:10]
    # One rule for the tag, shared with `_witness_family_pattern`: the sibling probe
    # asks the vault for a family by the same spelling the emitter writes, and two
    # copies of a 48-char substitution are how a glob and a name stop agreeing.
    tag = B._witness_tag(src)
    stem = f"{day}.{item_id}-{tag}" if item_id else f"{day}.{tag}"
    return f"{artifact_dir}/{stem}-{name}"


# The shape `_witness_artifact` emits once a plain name is taken: a 10.2 MB
# `promotions.jsonl` demanded by five items over nine days produced
# `2026-10-14.2444-lloyd-automod-promotions.jsonl` and its siblings — all of them one
# store. The family tail is what survives between two such demands, because the day
# and the item id are precisely the two parts that move.
_DATED_WITNESS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.(?:\d+-)?(.+)$")


def _witness_tag(src: Path) -> str:
    """The store-identifying component of a dated witness name.

    Extracted from `_witness_artifact` rather than duplicated beside it: the 48-char
    cap and the character substitution are what make a name like
    `2026-10-14.2471-lloyd-data-workers.db` identify `workers.db`, and the sibling
    probe below has to ask for exactly what the emitter writes.
    """
    return re.sub(r"[^\w.\-+]", "-", src.parent.name).strip("-.")[:48] or "src"


def _witness_family_tail(target: Path) -> str:
    """`-lloyd-data-workers.db`: the part of a dated witness name that IS the store.

    `_witness_artifact` builds `{day}.{item_id}-{tag}-{name}`, so from the hyphen
    before the tag down the name identifies the store, and nothing above it does.
    Derived from the SOURCE through `_witness_tag` rather than by parsing an emitted
    name: the day component itself contains hyphens, so a `split("-", 1)` on
    `2026-10-23.2703-lloyd-data-workers.db` cuts inside the DATE and yields a family
    that matches nothing while looking like one.
    """
    return f"-{B._witness_tag(target)}-{target.name}"


def _witness_family_pattern(target: Path) -> str:
    """`*-lloyd-data-workers.db` — the glob for one store's dated sibling family.

    Shown to the reader, not used to search: `_committed_witness_siblings` matches the
    tail with `endswith`, because a store name carrying `[` or `?` would otherwise
    widen its own family through the glob compiler.
    """
    return f"*{B._witness_family_tail(target)}"


def _committed_witness_siblings(path, artifact_dir: str = B.WITNESS_ARTIFACT_DIR) -> int:
    """How many extracts of `path` the vault already holds under `artifact_dir`.

    Read through `_vault_witness_dir`, which resolves `app.paths.VAULT_ROOT` from the
    module at CALL time — the same seam `_artifact_taken` uses — so a test can point
    the whole probe at a temporary vault. An unreadable tree counts as none, which is
    the #2513 degradation: this count exists so a demand can refuse itself against
    bytes already committed, and under the gate (no `~/obsidian`) it has to go quiet
    rather than raise. A demand that goes silent is how a store stops being witnessed.

    `endswith` on the store's own tail is what makes the number mean extracts of THIS
    store: `…-lloyd-data-users.db` shares the parent directory and does not match, and
    `2026-10-14.2471-lloyd-data-workers-witness.md` is a note ABOUT an extract, ending
    `-workers-witness.md`, so it cannot be counted as one either.
    """
    return len(B._committed_witness_sibling_names(path, artifact_dir))


def _committed_witness_sibling_names(path,
                                     artifact_dir: str = B.WITNESS_ARTIFACT_DIR) -> list[str]:
    """The NAMES of the extracts of `path` the vault holds, sorted, oldest first.

    #3872 factored the walk out of `_committed_witness_siblings` because a count is no
    longer the only thing asked of it: the over-bound note asks `if siblings:` (#3175),
    which a count answers, while the whole-file demand has to NAME the copy it is
    re-aiming the reader at. Two call sites, one walk — and above all one definition of
    "an extract of this source". A second walk is exactly where a second, subtly
    different answer to that question comes from, which is the class this module keeps
    re-meeting. The count keeps its own entry point so both existing callers and their
    nodes keep reading a number.
    """
    tail = B._witness_family_tail(B._witness_target(path))
    try:
        vdir = B._vault_witness_dir(artifact_dir)
        return sorted(p.name for p in vdir.iterdir()
                      if p.name.endswith(tail) and p.is_file())
    except OSError:
        return []


# ── the loop does the cp itself (#3218) ───────────────────────────────────────
#
# WHY A COPY ROUTE EXISTS. For every round of this the guard's answer was a DEMAND:
# "copy it into `backlog/data` and quote `wc -l -c` off the copy". The demand was
# correct, and the rounds that obeyed it were spent on a `cp`. The promotions ledger
# holds four `vault_land` events (#2685→#2954 and #2935 for a `gate.json`) whose whole
# path list is one `backlog/data/nightly-*.json` plus the item that asked for it, each
# closed `acceptance_source: round`. Since `9f5dc0cd` (#2289) the demand reaches the owed
# list rather than the graded contract, so what those rounds were really buying is the
# copy a human clause implies. Where the bytes are already decided, the guard can put
# them in history and stop asking.
#
# WHY ONLY ONE FAMILY. The guard above is generic: it fires on any cited path that is on
# disk, under `WITNESS_MAX_BYTES`, and in no git tree — a `gate.json`, a `gate.log`, a
# classify batch, an index. Those witnesses need an item to CHOOSE something (an extract,
# which arm, which window), and a route that copied them would be the loop writing its
# own evidence. A nightly retrieval-eval baseline is quoted WHOLE by construction:
# `summary.overall` is the rate table every trend claim in an item reads, and its
# basename is unique per run, so there is nothing to choose and nothing to rewrite. So
# the family test names both the basename AND the root the eval job writes under, and it
# is the same test the budget below counts against — a cap that counted only `nightly-*`
# while the route accepted any basename would let a 404,430-byte blob into a 127 MB
# `backlog/data` through whichever item happened to cite it.
#
# WHY THE COMMIT IS NOT `vault_round.land()`. `land()` refuses a caller with no
# `round_id` and no `item_id`: its own probe rail raises for want of "the second
# reader's verdict", and this route is not a round — it is the copy that makes a round
# unnecessary. What it shares with `land()` is the commit discipline that the same file
# had to learn the hard way: `--` before every pathspec (#2454: without it a path
# starting with `-` is read as a rev), no `-f` anywhere (an ignored target is refused,
# not forced), one path per commit, and the figures read back out of the commit rather
# than off the disk that produced it.
ANCHOR_WITNESS_PREFIX = "nightly-"
#: The source root the eval job writes baselines into. Required alongside the prefix:
#: `~/lloyd-data/eval/baselines/` is not a git tree (`git -C ~/lloyd-data rev-parse` →
#: `fatal: not a git repository`), which is the whole reason the guard fires on these
#: files every night, and it is what keeps a `nightly-something.json` cited from
#: somewhere else out of a route scoped to this one store.
ANCHOR_WITNESS_ROOT = ("eval", "baselines")
#: What the family may cost the vault. Measured at triage (2026-10-24): 9 tracked
#: `nightly-*` witnesses totalling 3,129,787 B in a 271-file / 127 MB `backlog/data`,
#: and 15 cited-but-uncommitted sources — so the condition recurs nightly, and 20 files
#: is roughly a fortnight of them at the 31-file source inventory's rate. A tracked
#: witness is never deleted or rewritten by this route: the budget can only ever be
#: reached by a NEW copy, so refusing is the whole of the enforcement.
ANCHOR_WITNESS_MAX_FILES = 20
ANCHOR_WITNESS_MAX_TOTAL_BYTES = 8 * 1_048_576
#: The figures an item quotes off a baseline, and the only ones re-derived from the
#: blob: `scripts/eval_trend_stats.py` grades `entity_hit_rate`/`doc_hit_rate` as its
#: rate legs and `ndcg10`/`mrr_doc` as its continuous ones (its `CONT_LEGS`, :125), over
#: the `n_queries` denominator. Anything else in `summary.overall` is a knob, not a
#: result, and an item that needs it can read the blob by the sha this route names.
ANCHOR_WITNESS_FIGURES = ("n_queries", "entity_hit_rate", "doc_hit_rate",
                          "mrr_doc", "ndcg10")

#: `reason` values that mean "this witness already has history" — no demand, no note.
#: A plain basename the vault holds can only be OUR earlier copy: `_witness_artifact`
#: hands out a dated sibling name once the plain one is taken, and this family's
#: `nightly-*` names are in neither `_RETIRED_WITNESS_NAMES` nor `_ROLLING_STATE_NAMES`,
#: so taken ⟺ archived by a previous pass.
ANCHOR_WITNESS_DONE = "taken"
#: `reason` values that mean "the copy failed for a reason that is not a guard's
#: judgement" — the demand falls back, because a witness that silently stops being
#: witnessed is the failure this whole guard exists to prevent. The four named refusals
#: above (ceiling / ignored target / taken name / budget) do NOT fall back: ordering a
#: copy the guard has just refused is how a round gets booked for nothing.
_ANCHOR_WITNESS_FALLBACKS = frozenset(
    {"unreadable", "copy", "git", "add", "commit", "no_sha", "blob", "split_vault"})


def _is_anchor_witness(path) -> bool:
    """True for a cited nightly eval baseline: basename `nightly-*` under `eval/baselines/`.

    Both halves, and neither is redundant. The prefix alone would sweep in any other
    `nightly-` file an item cites; the root alone would sweep in a whole directory the
    eval job does not write. Checked on the resolved path's components, so `~`,
    `$HOME/lloyd-data` and a relative `eval/baselines/` all read the same way.
    """
    parts = B._witness_target(path).parts
    if not parts or not parts[-1].startswith(B.ANCHOR_WITNESS_PREFIX):
        return False
    return any(parts[i:i + 2] == B.ANCHOR_WITNESS_ROOT for i in range(len(parts) - 1))


def _anchor_witness_family(vault_root) -> "tuple[int | None, int | None]":
    """(tracked files, total bytes) of the nightly witnesses already in history.

    `git ls-files` and not `iterdir()`: the budget is about what the vault CARRIES, and
    a `nightly-*.json` sitting untracked in `backlog/data` is not in the history the
    budget is bounding — it is the gap this route is trying to close. Either figure is
    `None` when git cannot answer, which the caller must treat as "unknown", never as
    zero: a probe that reads an absent vault as an empty one is how a cap passes
    silently on the machine it is meant to protect.
    """
    import subprocess
    try:
        proc = subprocess.run(["git", "-C", str(vault_root), "ls-files", "-z", "--",
                               B.WITNESS_ARTIFACT_DIR],
                              capture_output=True, text=True, timeout=60)
    except Exception:
        return (None, None)
    if proc.returncode != 0:
        return (None, None)
    names = [n for n in (proc.stdout or "").split("\0")
             if Path(n).name.startswith(B.ANCHOR_WITNESS_PREFIX)]
    total = 0
    for n in names:
        try:
            total += (Path(vault_root) / n).stat().st_size
        except OSError:
            # A tracked path with no bytes under it (deleted in the worktree, still in
            # HEAD) counts as 0 bytes and still counts as a file. The cap that matters
            # for the vault's size is the byte one, and it errs the other way here — so
            # say so rather than quietly understating what is there.
            continue
    return (len(names), total)


def _blob_figure(vault_root, sha: str, artifact: str) -> "tuple[str, str]":
    """(`wc -l -c` of the blob, the rate figures read off the blob).

    Both read through `git show <sha>:<path>` and nothing else. The reason is #2954's:
    the figure a round quoted was the figure of whatever bytes were on disk when it
    looked, so the number in an item and the number in history could disagree and
    nothing caught it. `wc` runs over the blob on stdin, so the two integers this
    returns ARE `wc -l -c < backlog/data/<name>` for anyone who re-runs it — the
    comparison is on the numbers, not on the whitespace `wc` pads when it is given a
    file operand.
    """
    import subprocess
    proc = subprocess.run(["git", "-C", str(vault_root), "show", f"{sha}:{artifact}"],
                          capture_output=True, timeout=120)
    if proc.returncode != 0:
        return ("", "")
    blob = proc.stdout
    # `wc` gets the blob on STDIN with no `text=True`: the bytes are what is being
    # measured, and decoding them to count characters would be a different measurement
    # than the `wc -c < file` a reader re-runs on disk.
    wc = subprocess.run(["wc", "-l", "-c"], input=blob, capture_output=True,
                        timeout=60)
    bits = (wc.stdout or b"").decode(errors="replace").split()
    figures = ""
    try:
        overall = json.loads(blob.decode("utf-8")).get("summary", {}).get("overall", {})
        got = [f"{k}={overall[k]:.6g}" if isinstance(overall[k], float)
               else f"{k}={overall[k]}"
               for k in B.ANCHOR_WITNESS_FIGURES if k in overall]
        figures = ", ".join(got)
    except Exception:
        figures = ""
    if wc.returncode != 0 or len(bits) != 2:
        return ("", figures)
    return (f"{bits[0]} {bits[1]}", figures)


def commit_anchor_witness(source, artifact: str, *,
                          max_files: "int | None" = None,
                          max_total_bytes: "int | None" = None,
                          item_id: "int | None" = None,
                          session_id: "str | None" = None) -> dict:
    """Archive a cited nightly eval baseline into the vault and quote it from history.

    Copies the WHOLE source file byte-identically to `artifact` under the vault witness
    dir, leaves the source exactly where it is (no extract, no move, no truncate), and
    commits exactly one revertable sha touching that one path. Returns
    `{"ok", "reason", "sha", "artifact", "note", "wc", "figures"}`; on any refusal
    nothing is committed and `note` says why in the terms a human needs — `note` is
    empty only for `reason="not_anchor"`, which is the caller's cue to emit the demand
    clause this route exists to replace.

    The vault is resolved AT CALL TIME through `_vault_witness_dir`, which reads
    `app.paths.VAULT_ROOT` from the module — the constraint `scripts/autoresearch/promote.py:1208`
    records as the failure mode of resolving it at import: a test that cannot move the
    vault root is a test that runs `git add` against `~/obsidian`. `max_files` and
    `max_total_bytes` are parameters so the budget's arithmetic is testable without
    planting eight megabytes of baselines.
    """
    import shutil
    import subprocess

    # The budget is read HERE, from the module, rather than as a signature default: a
    # default argument is bound at import, and the note below has to quote the budget it
    # actually applied. An explicit argument still wins.
    max_files = B.ANCHOR_WITNESS_MAX_FILES if max_files is None else max_files
    max_total_bytes = (B.ANCHOR_WITNESS_MAX_TOTAL_BYTES if max_total_bytes is None
                       else max_total_bytes)

    def refuse(reason: str, note: str = "") -> dict:
        return {"ok": False, "reason": reason, "sha": "", "artifact": artifact,
                "note": note, "wc": "", "figures": ""}

    if not B._is_anchor_witness(source):
        # Not this family: no refusal to surface, the caller's demand stands as #2289
        # and #2382 left it.
        return refuse("not_anchor")

    src = B._witness_target(source)
    artifact_dir = str(Path(artifact).parent)
    name = Path(artifact).name
    vdir = B._vault_witness_dir(artifact_dir)
    root = vdir
    for _ in Path(artifact_dir).parts:
        root = root.parent          # `backlog/data` → the vault itself
    # One vault, both ends. The tree this function commits in is a git tree under the
    # root `_vault_witness_dir` names — there is no second `git -C` to disagree with, so
    # the failure this rules out is a witness DIRECTORY that is not inside a vault at
    # all: under the gate `~/obsidian` is absent and `app.paths.VAULT_ROOT` resolves to
    # `/`, which is a git tree for no value of artifact_dir. `rev-parse --show-toplevel`
    # is the answer rather than an equality test against `vault_round.VAULT` because it
    # reads the tree that is actually there, and because both spellings of the vault are
    # movable by a test — `LLOYD_VAULT` rebinds `vault_round.VAULT` (`vault_round.py:64`)
    # while `app.paths.VAULT_ROOT` is monkeypatched, and the one that is captured at
    # import time is the one `promote.py:1208` says must never be used to `git add`.
    top = None
    try:
        # The dir a fresh vault clone may not have yet. Guarded because the failure
        # this function must never produce is a traceback: with the vault absent (the
        # gate sandbox, a round run with `HOME` elsewhere) this is a permission error
        # on `/backlog`, and the rev-parse below is what turns that into a note.
        vdir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    try:
        top = subprocess.run(["git", "-C", str(vdir), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=60)
    except Exception:
        top = None
    answer = (top.stdout or "").strip() if top is not None else ""
    if (top is None or top.returncode != 0 or not answer
            or not str(root.resolve()).startswith(str(Path(answer).resolve()))):
        return refuse("split_vault",
                      f"#3218 archived nothing: `{artifact_dir}` does not resolve into "
                      f"a git tree (`git -C {vdir} rev-parse --show-toplevel` "
                      f"{'exited ' + str(top.returncode) if top is not None and top.returncode != 0 else 'answered nothing'}`), "
                      f"so there is no vault to commit `{artifact}` into and the "
                      f"witness still has no history.")
    root = Path(answer).resolve()

    # ONE vault, both ends. The copy is written under the root `_vault_witness_dir` names
    # and the commit is made by `vault_round.land()`, which takes its tree from the module
    # attribute `vault_round.VAULT`. Those are two independent seams — `app.paths.VAULT_ROOT`
    # and `VR.VAULT` — and on this box they are the same directory only because nothing
    # moved them. When they differ the copy lands in one tree, `land` stages nothing in
    # the other, and the route either refuses with a message about a path it never wrote
    # or, worse, commits a sha whose blob is not the file that was measured. Refuse here,
    # naming both, rather than report bytes archived that no history holds.
    from scripts.automod import vault_round as _VR
    _land_root = Path(_VR.VAULT).expanduser().resolve()
    # Containment, not a string prefix: `…/vault2` starts with `…/vault` as TEXT and the
    # two are unrelated trees, which is how a prefix test passes on exactly the split it
    # exists to catch. `is_relative_to` asks the question about path components.
    if root != _land_root and not (
            hasattr(root, "is_relative_to") and root.is_relative_to(_land_root)):
        return refuse("split_vault",
                      f"#3218 archived nothing: the witness directory resolves to "
                      f"`{root}` but the landing route commits in `{_land_root}`, so the "
                      f"copy would be written where no commit is made and `{artifact}` "
                      f"would still have no history.")

    size = B._witness_size(source)
    if size is None:
        return refuse("unreadable",
                      f"#3218 could not read `{src}` to archive it, so the copy "
                      f"route committed nothing and the witness still has no history.")
    if not B._witness_within_size_bound(source):
        return refuse("over_ceiling",
                      f"#3218 archived nothing: `{src}` is {size:,} bytes, over the "
                      f"WITNESS_MAX_BYTES = {B.WITNESS_MAX_BYTES:,} ceiling on what a "
                      f"witness clause may demand whole. Nothing was committed. An "
                      f"extract is still owed, and only an item can choose it.")
    # Idempotence, keyed on the SOURCE's own basename and NOT on `artifact`: on a second
    # triage of the same item `_witness_artifact` has already disambiguated the slot
    # (#2513), so the artifact handed to this function is a dated sibling that is
    # perfectly free — copying again would plant one more copy of a single baseline every
    # time its item is re-triaged, which is the guard committing unbounded witnesses
    # instead of stopping rounds. The question being asked is "has THIS baseline already
    # been archived?", and the answer is whether its own name is tracked.
    own = subprocess.run(["git", "-C", str(root), "ls-files", "--",
                          f"{B.WITNESS_ARTIFACT_DIR}/{src.name}"],
                         capture_output=True, text=True, timeout=60)
    tracked = own.stdout.strip() if own.returncode == 0 else ""
    if tracked:
        # A tracked name is only "already archived" if the tracked BYTES are this
        # file's. One name can hold two documents — two runs that picked the same
        # basename, or an item filed against a baseline that was later re-run — and
        # calling that done would silently withhold a witness nobody has committed.
        shown = subprocess.run(["git", "-C", str(root), "show", f"HEAD:{tracked}"],
                               capture_output=True, timeout=120)
        if shown.returncode == 0 and shown.stdout == src.read_bytes():
            return {"ok": True, "reason": B.ANCHOR_WITNESS_DONE, "sha": "",
                    "artifact": tracked, "note": "", "wc": "", "figures": ""}


    if B._artifact_taken(artifact_dir, name):
        return refuse(B.ANCHOR_WITNESS_DONE,
                      f"#3218 archived nothing: `{artifact_dir}/{name}` already holds "
                      f"bytes. The name is never overwritten (#2513 — a demand aimed "
                      f"at a slot the vault holds is what made main red), and the "
                      f"witness already has history, so nothing is owed.")
    # An ignored target cannot be committed without `-f`, which is refused here and
    # everywhere: `.gitignore:141` ignores `eval/baselines/*` wholesale with one
    # carve-out, and a `backlog/data` rule added tomorrow must stop this route rather
    # than make it force a path past the ignore. `check-ignore` exits 0 on a match, 1
    # on none, 128 on error — and "cannot prove it is addable" is a refusal, not a pass.
    # `-v` and NOT `-q`: git 2.56 refuses to have both ("cannot have both --quiet and
    # --verbose", exit 128), and a 128 here is indistinguishable from a broken tree.
    try:
        chk = subprocess.run(["git", "-C", str(root), "check-ignore", "-v", "--",
                              artifact], capture_output=True, text=True, timeout=60)
    except Exception:
        return refuse("git",
                      f"#3218 archived nothing: `git check-ignore` for `{artifact}` "
                      f"could not run, so the target's ignore state is unknown and the "
                      f"copy would have to force it. The witness has no history.")
    if chk.returncode == 0:
        return refuse("ignored",
                      f"#3218 archived nothing: `git check-ignore -v` accepts "
                      f"`{artifact}` (matched `{(chk.stdout or '').strip()}`), so "
                      f"committing it needs `-f` — which this route never passes. "
                      f"Nothing was committed; the witness still has no history.")
    if chk.returncode != 1:
        return refuse("git",
                      f"#3218 archived nothing: `git check-ignore` for `{artifact}` "
                      f"exited {chk.returncode}, so the target's ignore state is "
                      f"unknown. Nothing was committed.")

    count, total = B._anchor_witness_family(root)
    if count is None:
        return refuse("git",
                      f"#3218 archived nothing: the tracked nightly-witness count in "
                      f"`{B.WITNESS_ARTIFACT_DIR}` could not be read, so the "
                      f"{max_files}-file / "
                      f"{max_total_bytes:,}-byte family budget could not "
                      f"be checked and is not assumed to pass.")
    if count + 1 > max_files:
        return refuse("too_many_files",
                      f"#3218 archived nothing: `{B.WITNESS_ARTIFACT_DIR}` already "
                      f"carries {count} tracked `nightly-*` witnesses and the family "
                      f"budget is {max_files} files "
                      f"(these would be {count + 1}). Nothing was committed, and "
                      f"nothing tracked was deleted or rewritten — the cap is a "
                      f"human call on the family, not a licence to expire a witness.")
    if total + size > max_total_bytes:
        return refuse("too_many_bytes",
                      f"#3218 archived nothing: {count} tracked `nightly-*` witnesses "
                      f"total {total:,} bytes and the family budget is "
                      f"{max_total_bytes:,} bytes (this file is {size:,}). Nothing was "
                      f"committed, and nothing tracked was deleted or rewritten.")

    try:
        vdir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, vdir / name)
    except OSError as exc:
        return refuse("copy",
                      f"#3218 archived nothing: the copy of `{src}` to "
                      f"`{artifact}` failed ({exc}). The witness has no history.")
    import filecmp
    if not filecmp.cmp(str(src), str(vdir / name), shallow=False):
        (vdir / name).unlink(missing_ok=True)
        return refuse("copy",
                      f"#3218 archived nothing: the copy at `{artifact}` is not "
                      f"byte-identical to `{src}`, so the copy was removed and nothing "
                      f"was committed.")

    def head() -> str:
        try:
            p = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                               capture_output=True, text=True, timeout=60)
            return (p.stdout or "").strip() if p.returncode == 0 else ""
        except Exception:
            return ""

    msg = (f"#3218 archive the nightly eval baseline an item quotes: {name}\n\n"
           f"Source: `{src}` (left in place; the whole file, not an extract).\n"
           f"Filed automatically by the unarchived-witness guard, which found this file "
           f"cited by an item with no git tree behind it.\n")

    # The commit goes through the loop's own vault write route. `vault_round.land()` is
    # what the `automod_vault_land` tool runs: the only door that validates the paths,
    # runs the vault's guards and appends the `vault_land` row to `promotions.jsonl`. That
    # row is what the owed-check reads to decide whether this route ever archived
    # anything, and what `promote.py:vault_commits_for` reads to decide whether a mixed
    # item's vault half landed at all; a raw `git add`/`git commit` would put bytes in the
    # vault with no ledger event behind them, which is the un-ledgered half of a landing.
    #
    # Read at CALL time, like the vault root: `land()` takes its tree from the module
    # attribute `vault_round.VAULT` when it runs, so a test that rebinds that attribute
    # works against a temporary repo and no node stages a thing against `~/obsidian` — the
    # failure `promote.py:1208` records for an import-time binding.
    from scripts.automod import vault_round as VR

    before = head()
    try:
        landed = VR.land([artifact], msg, item_id=item_id, session_id=session_id)
    except Exception as exc:
        # `land` refuses two ways: a validation/scope/guard failure comes back as
        # `{"ok": false, ...}`, and "nothing staged on those paths" is RAISED as
        # `VaultRoundError` (`vault_round.py:1879`). Both are refusals, and neither may
        # escape as a traceback — this function is called from triage, on every item
        # text a nightly job reads, and an exception there loses the whole triage pass.
        (vdir / name).unlink(missing_ok=True)
        return refuse("commit",
                      f"#3218 archived nothing: the vault's landing route rejected "
                      f"`{artifact}` ({type(exc).__name__}: `{str(exc)[:240]}`), so no "
                      f"sha exists to quote and the witness still has no history.")
    if not landed.get("ok"):
        # Everything `land` refuses on it refuses before staging, so a refusal committed
        # nothing — but the copied bytes are sitting in the vault's working tree RIGHT
        # NOW, and an untracked witness there is a file with no history that reads as
        # archived to anyone who only lists the directory. Take it back.
        (vdir / name).unlink(missing_ok=True)
        detail = (landed.get("validation") or landed.get("note")
                  or landed.get("message") or landed.get("error") or "")
        code = landed.get("refuse_code") or landed.get("reason") or "refused"
        return refuse("commit",
                      f"#3218 archived nothing: the vault's landing route refused "
                      f"`{artifact}` ({code}: `{str(detail)[:240]}`), so no sha exists "
                      f"to quote and the witness still has no history.")
    # The route names its commit `commit`, not `sha` (`vault_round.py:1943`); reading the
    # wrong key would report a landed archive as a commit-less one.
    after = str(landed.get("commit") or "")
    if not after:
        (vdir / name).unlink(missing_ok=True)
        return refuse("no_sha",
                      f"#3218 archived nothing: the landing route reported success for "
                      f"`{artifact}` without naming a commit, so any figure quoted below "
                      f"would come from a commit nobody can revert.")
    if after == before:
        return refuse("no_sha",
                      f"#3218 archived nothing: the landing route returned the HEAD it "
                      f"started at (`{after}`), so nothing was committed and any figure "
                      f"quoted below would name bytes that are not in history.")

    # The tree the commit landed in is the route's, and it is only the same directory as
    # the one `_vault_witness_dir` named because nothing moved it — `head()` above and
    # `VR.VAULT` below disagree the moment a caller rebinds one and not the other, and
    # then every later check would be reading a repo that does not hold the file.
    root = Path(VR.VAULT)
    touched = [ln for ln in (subprocess.run(
        ["git", "-C", str(root), "show", "--pretty=format:", "--name-only", after],
        capture_output=True, text=True, timeout=60).stdout or "").splitlines()
        if ln.strip()]
    if touched != [artifact]:
        return refuse("split_commit",
                      f"#3218 landed `{after}` for `{artifact}` but that sha also names "
                      f"{touched} — the sha this route reports must touch one path and "
                      f"nothing else, because it is the sha a human reverts. Revert it "
                      f"(`git -C {root} revert {after}`) before trusting the figures.")
    wc, figures = B._blob_figure(root, after, artifact)
    if not wc:
        return refuse("blob",
                      f"#3218 committed `{after}` for `{artifact}` but `git show "
                      f"{after}:{artifact}` gave no bytes to measure, so no figure is "
                      f"quoted and the archive must be checked by hand.")
    ext = Path(name).suffix.lower().lstrip(".")
    note = (f"#3218 archived the witness itself, no round needed: `{src}` -> "
            f"`{artifact}` in the vault at `{after}` (whole file, source left in place, "
            f"{size:,} bytes). Re-derived from that blob, not from the disk it came "
            f"from: `{B._rederive_command(artifact, ext)}` -> `{wc}`"
            + (f"; summary.overall {figures}" if figures else
               "; summary.overall figures not present in the blob"))
    return {"ok": True, "reason": "archived", "sha": after, "artifact": artifact,
            "note": note, "wc": wc, "figures": figures}


def _over_bound_extract_demand(path, artifact_dir: str = B.WITNESS_ARTIFACT_DIR,
                              item_id=None) -> str:
    """The owed note for a witness too large to be asked for whole (#2382).

    What #2267 ruled out is ordering the STORE: bytes over `WITNESS_MAX_BYTES` into
    `backlog/data`, which already holds 108 MB, on an item that will be re-triaged
    more than once. What #2267 owed, and this delivers, is the other reading — serve
    the claim, not the file. So the note asks for an EXTRACT: the subset of that
    store which reproduces the figure the item quotes, committed under the same
    convention, and proved by the same re-derive command the under-bound clause
    carries. It states the ceiling as the constant and as a number, because the
    reader has to know what "under" means without opening this file, and it closes
    with the fallback that keeps the bound a bound: when the figure cannot be
    reproduced under it, say so — do not order the store anyway.

    Two wordings, chosen by budget, because the property has to hold over path
    lengths this rule cannot enumerate: `clean_clauses` truncates an owed entry at
    `CLAUSE_MAX_CHARS` FROM THE END, so on a long path the sentence that disappears is
    the fallback — the one instruction that stops a round ordering the store anyway.
    The roomy form is what a normal path gets; when it will not fit, the cheap
    characters go first (the artifact needs no separate mention once the deriving
    command carries it, and `the subset` is the half the compact lead drops because the
    roomy sentence has already said it) and the mandatory halves — source path, ceiling
    by constant and by number, the subset, the read-instant cap, the command, the
    fallback — are what survive. Both forms end on the same sentence, so which one was
    chosen never changes what the item asks for.

    Both forms also carry the cap, and #3894 is why it could not live in one of them.
    The demand asks for "the subset that reproduces the figure the item quotes", and on
    a live store that subset has an expiry date: the figure was READ at an instant, and
    a cut taken later is a different set of rows. The board-steward witness is the
    exhibited case — `judgement_aggregate()` in workers/dispatch_watch.py filters
    `completed_at >= since` with no upper bound, because the function that reads it,
    `list_runs_joined()` in workers/queue.py, stops at that predicate; and `since` is
    built FROM the read instant, so one command read now / +3d / +7d on
    `~/lloyd-data/workers.db` returns 163 / 107 / 0 ticks and rates 0.8414 / 0.8537 /
    None, and #3602's figure drifted 406 → 407 → 446 rows while a round tried to satisfy
    the uncapped wording. So the note names the instant: the extract is the store `as at
    the reading`, which bounds the cut above as well as below. The bound is in the
    DEMAND prose and not in `_rederive_command`, because that command recomputes over the
    COMMITTED extract bytes and is time-invariant once the extract exists — the
    time-dependence is entirely in choosing the subset from the live store, which
    nothing but this note constrains.

    Deliberately absent from both: the store's own byte count. `~/lloyd-data`
    witnesses rotate and grow, so a size written into the item's front matter is a
    figure that goes stale on purpose, and the demand does not need it — the ceiling is
    the number that governs the work.

    The artifact it names comes from `_witness_artifact`, so a store whose basename
    `_witness_name_must_be_dated` answers yes for — a retired slot, or a rolling state
    file — is asked for at a dated sibling rather than at the free
    slot the vault must not hold (#2444 owed 1, 2026-10-11: the demand against the live
    ledger named `promotions.jsonl`, a path three tests keep absent and whose deletion
    is why the free-name probe could not see it).

    How far that name repeats is narrower than it looks, and #3175 is what measured
    it. `_witness_artifact`'s stem is `{day}.{item_id}-{tag}`, so this note orders the
    same file ONLY for the same day AND the same item id — one item, re-triaged twice
    in one day. Both other pairs name different paths: the same item asked again
    tomorrow gets tomorrow's date, and two demands about ONE store from two different
    items get two files. #3174 is that second case at the real vault: it ordered
    `workers.db` schema whose 98,304 bytes and 21 sqlite_master objects
    `backlog/data/2026-10-14.2471-lloyd-data-workers.db` already held. So the note
    now says what is already there — the family pattern and how many committed
    extracts match it — and asks for the figure to be measured over those before a new
    file is published, which is the only reuse #2769's measurements support: it found
    zero byte-identical pairs among 83 committed artifacts, so `_witness_artifact` and
    `_artifact_taken` stay as they were and the pointing happens in the text.

    #3942 moved that pointing the rest of the way, for the half of it that was a
    contradiction and not a convenience. The text said "measure those first; publish a new
    file only if none reproduces the figure" while the destination it ordered was a fresh
    date-today name — the day of TRIAGE, since `now_stamp()[:10]` is what stamps it and no
    `day=` reaches this builder — and the bytes a round later cuts land on whatever day
    THAT round runs. #3547's demand asked for `backlog/data/2026-10-29.3547-lloyd-data-
    workers.db` and the vault holds only `2026-11-06.3547-…`; a scan of the vault's demand
    sentences on 2026-11-08 counted 170 demanded paths that never existed across 151
    items, 17 of them the SAME item's artifact under another day (#3280, #3308, #3365,
    #3375, #3418, #3447, #3461, #3518, #3547, #3549, #3607, #3619, #3643, #3684, #3726). A
    round that obeyed the pointer half reused a committed sibling and left the demanded
    path absent forever, which is exactly how a published witness reads as missing. So when
    the vault already commits a family for this source the destination IS one of those
    bytes — the newest committed sibling, named identically in the sentence and in the
    `proved by` command, so the note can no longer order a file nobody was told to write.
    Newest rather than oldest because #3894's "as at the reading" is closest to the
    snapshot nearest now; the under-bound whole-file demand keeps `sibs[0]` (#3872) and its
    own node pins that, so the two branches are left un-harmonised on purpose. With no
    family the fresh dated destination stands, because the round that obeys writes its
    bytes to exactly that path — which is also why "resolve the command through the glob"
    was rejected for that branch: with no family the glob matches zero files, so it could
    only teach the command to glob a hole. The swap is in place, never appended, and it is
    length-neutral (day and item id are fixed width): the emitted `workers.db` note carries
    2 characters of headroom at 29 committed siblings, and `clean_clauses` truncates FROM
    THE END, which is where #3175's pointer and #2382's fallback live. ONE case is not
    neutral, and the last rung below settles it: where the free-name slot is open the fresh
    destination is the bare basename, so a dated sibling is ~15 characters dearer, and at
    the longest store names that overshoots the cap — there the fresh name stands, because
    the sentence a truncation takes is the one that stops a round ordering the store
    anyway. Nothing committed is renamed — the harm being undone is a MISSING witness, so a
    demand that made its path true by MOVING bytes would be a worse bug than the one it
    fixes.
    """
    target = B._witness_target(path)
    # #3942: ONE listing of `backlog/data`, taken before the destination is chosen, so the
    # count the reader is told to check and the name the demand orders cannot come from two
    # answers. Names only: the bytes are never touched.
    sib_names = B._committed_witness_sibling_names(path, artifact_dir)
    # #2513: the same name rule as the under-bound clause, because the over-bound
    # note orders a vault write to exactly the same directory and would clobber a
    # pinned witness just as deadeningly. A 30.7 MB store is the shape that reaches
    # this branch, and the extract it asks for lands in `backlog/data/`, among the
    # witnesses already delivered there — so the destination needs the same test for
    # a name somebody else holds.
    # #3942: and when that test finds the vault ALREADY commits an extract of this source,
    # the destination is those bytes rather than a fresh date-today name the vault will
    # never hold. The names are date-prefixed, so their lexicographic order IS chronological
    # and `[-1]` is the most recent — the closest snapshot to the reading the figure came
    # from. No family, no re-aim: `_witness_artifact`'s dated name is the right answer then,
    # because the round that obeys writes its bytes to exactly that path.
    sibling = f"{artifact_dir}/{sib_names[-1]}" if sib_names else ""
    fresh = B._witness_artifact(path, artifact_dir, item_id=item_id)
    ext = target.suffix.lower().lstrip(".")
    limit = f"{B.WITNESS_MAX_BYTES:,}"
    # Identical tail in both forms, and it is last in both: the sentence that must
    # survive the budget is the sentence the budget is for.
    fallback = ("Prefer an extract under it; if the figure cannot be reproduced "
                "under it, say so rather than ordering the store.")
    # #3894: the extract is cut at an instant, not over a window's open end. The store
    # this demand keeps firing on is `judgement_aggregate`, and its read has a lower
    # bound and no upper one — `list_runs_joined()` in workers/queue.py filters
    # `r.completed_at >= ?` with nothing above it, while
    # `judgement_aggregate()` builds `since` FROM the read instant — so the same command
    # run an hour after a
    # reading counts ticks that landed since it and loses the oldest tail: the subset
    # that "reproduces the figure the item quotes" stops existing the moment the store
    # moves. #3602's figure went 406 → 407 → 446 rows while a round was told to
    # reproduce it, and #3330 §1, #3546, #3680, #3685 and #3842 are five independent
    # rediscoveries of the same dead end in thirteen days. `as at the reading` is the
    # bound in three words — the state of the store AT the instant the quoted figure was
    # read, which excludes later rows and earlier-pruned ones alike. Both forms spend
    # it; the roomy form pays for it with `on what this rule may demand` (29 characters,
    # scaffolding the ceiling already says) and the compact with `Publish`→`Owed:`, the
    # dropped `so` and `the figure the item quotes`→`the quoted figure`, because the
    # emitted `workers.db` note measured 595 of `CLAUSE_MAX_CHARS` and an APPENDED cap
    # would have bought a pointer's truncation.
    cap = "as at the reading"

    # #3175: what to measure BEFORE publishing. A path list cannot live here — the
    # longest form of this note has 2 characters of headroom under `CLAUSE_MAX_CHARS`
    # at the `workers.db` spelling with 29 committed siblings, which
    # `tests/test_witness_artifact_clause.py` re-measures, and `workers.db` carries 52
    # characters a path per sibling — so the pointer is the family PATTERN plus a count, and it rides INSIDE the fallback
    # rather than beside it: the fallback is 111 characters ending in the words the
    # ceiling exists to produce, and a reader who is told to publish a new extract is
    # exactly the reader who needs "…if none can, say so rather than ordering the
    # store" to stay attached to that instruction. The count is what makes the glob checkable — run it, get a different number, and you
    # have a contradiction to report rather than advice to ignore.
    # #3942: the count is the same listing the destination was chosen from, so a reader who
    # runs the glob either gets this number or has a contradiction to report — it cannot be
    # a number that agreed with a name the note no longer orders.
    # `_committed_witness_siblings` stays as the standalone count probe (this file's fixtures
    # read it as a second instrument on the directory); it is no longer called from here
    # because calling both is asking the vault twice for one answer.
    sibs = len(sib_names)
    family = f"{artifact_dir}/{B._witness_family_pattern(target)}"
    if sibs:
        plural = "extract" if sibs == 1 else "extracts"
        tail = (f"Measure the {sibs} committed {plural} under `{family}` first; publish "
                f"a new file only if none reproduces the figure, else say so "
                f"rather than ordering the store.")
    else:
        tail = fallback
    def demand(artifact: str) -> str:
        """Both wordings of the demand for ONE destination — #2382's ladder.

        Hoisted into a closure by #3942 because the destination is now a choice with two
        candidates (the committed extract, the fresh dated name), and the ladder has to run
        over each to know which of them the note can afford. The two forms and their order
        are unchanged: roomy first, then the compact lead, then dropping the pointer.
        """
        cmd = B._rederive_command(artifact, ext)
        roomy = (f"`{path}` is over the WITNESS_MAX_BYTES = {limit} byte ceiling: do not "
                 f"order it whole. What is owed is an extract of it — the subset that "
                 f"reproduces the figure the item quotes, {cap} — committed to "
                 f"`{artifact}` and re-derived from it with `{cmd}`. {tail}")
        if len(roomy) <= B.CLAUSE_MAX_CHARS:
            return roomy
        # The compact lead trades phrasing, never a mandatory half: `on what this rule may
        # demand` and `of the subset` are gone because the roomy sentence has said them,
        # and #3894's `as at the reading` is bought with `Publish`→`Owed:`, the dropped `so`
        # and `the figure the item quotes`→`the quoted figure`. What survives the trade is
        # the ceiling, the artifact, the command, the cap, the pointer and the closing words
        # — and the trade is bounded: the cap is in BOTH forms, so no path is left without
        # the bound, which is the half that used to be missing.
        compact = (f"`{path}` over the WITNESS_MAX_BYTES = {limit} byte ceiling: do not "
                   f"order it whole. Owed: an extract reproducing the quoted figure {cap}, "
                   f"proved by `{cmd}`. {tail}")
        if len(compact) <= B.CLAUSE_MAX_CHARS or not sibs:
            return compact
        # A store name long enough that even the compact note cannot carry a pointer keeps
        # #2382's text verbatim: an unspoken convenience, never a truncated fallback,
        # because `clean_clauses` cuts from the END and the end is the bound's purpose.
        return compact.replace(tail, fallback)

    # #3942's own last rung, and it is a trade, not a preference. Re-aiming at a committed
    # extract costs characters exactly when the source's free-name slot is OPEN, because
    # then `_witness_artifact`'s fresh destination is the bare basename and a dated sibling
    # carries 26 characters more (`2026-10-02.222-reflection-` before the store's own name) —
    # the longest such name (`a-quite-long-live-store-name-that-keeps-going.sqlite`, the one
    # `_rederive_command`'s 233-character budget is stated against) lands at 611 of 600 with
    # the pointer already dropped. A destination is not
    # worth #2382's fallback: `clean_clauses` cuts from the END, so the thing that
    # disappears is "…say so rather than ordering the store", the one sentence that stops a
    # round committing a 30 MB store into a vault that already holds 108 MB. So when naming
    # the committed extract does not fit and the fresh name does, the fresh name is taken —
    # and #3175's "measure those first" is what still points at the existing bytes. For the
    # stores this demand actually fires on (`workers.db`, whose basename
    # `_witness_name_must_be_dated` answers yes for) both candidates are dated and the swap
    # is length-neutral, 598 → 598 at this file's binding shape.
    note = demand(sibling or fresh)
    if sibs and len(note) > B.CLAUSE_MAX_CHARS and fresh != sibling:
        affordable = demand(fresh)
        if len(affordable) <= B.CLAUSE_MAX_CHARS:
            return affordable
    return note




def _tree_basenames(haystack, probe) -> "set[str]":
    """Basenames of every path in `haystack` that the probe places IN a git tree.

    The comparison the basename skip needs, kept separate so the rule sits in one
    place: an item's bytes are pinned when some path the item names is inside a
    tree's history, and the only thing two spellings of the same evidence have in
    common is the basename. Note what makes this not a blanket skip: the candidate
    itself can never reach the result through the probe — it was just told "outside
    every tree" — and any other path must be on disk AND inside a tree to suppress
    anything, so a mention of a name that does not exist pins nothing.

    `haystack` is the clauses joined with the item text, and every token in it is
    asked rather than only the ones already examined: the clause that cites the
    pinned copy is the case #2267 clause 1 is about, and a clause can sit on either
    side of the body's reference to the live file.
    """
    out: "set[str]" = set()
    for tok in B._PINNED_PATH_RX.findall(str(haystack or "")):
        for target in B._pinned_candidates(tok):
            if not B._witness_is_on_disk(target) or not probe(target):
                continue
            out.add(target.name)
            break
    return out


def _pinned_candidates(tok: str):
    """The spellings of a cited path worth probing: as given, then repo-relative.

    A clause cites what it pinned the way a test file cites it — `tests/fixtures/
    x.json`, no leading home and no leading slash — which resolves only against the
    repository, not against whatever directory triage happened to be invoked from.
    The absolute/home spellings are still tried first so a token that means what it
    says is never shadowed by an accident of CWD.
    """
    raw = str(tok)
    yield B._witness_target(raw)
    if raw.startswith(("./", "../")) or not raw.startswith(("/", "~", "$")):
        yield Path(__file__).resolve().parents[2] / raw.lstrip("./")


def _copy_demand(witness: str, artifact: str, ext: str, *,
                 hash_quoted: bool = False) -> str:
    """The whole-file copy demand, in the shape that can pin the figure quoted.

    Two wordings, one per kind of figure, because the re-derive command has to be able to
    reproduce what the item actually claimed. With no digest quoted the text is the one
    #2289/#2382/#2513 left, byte for byte — clause 3 of #4342 is that it does not change.
    With a matching digest quoted, an extract "reproducing the same numbers" is a
    non-sequitur (a digest has no numbers to reproduce except itself) and `wc -l -c` cannot
    answer a hash claim, so the demand asks for the file whole and for `sha256sum` of the
    bytes in the vault (#4342 clause 2). Neither wording issues a `git` command or names a
    repository: the demand orders bytes into the vault, and the vault route an item takes
    to land them is the item's own to choose, so a wording that read "commit this" would
    ask a code round for a commit its diff can never contain (#2289).
    """
    if hash_quoted:
        return (f"The witness bytes have no history: copy `{witness}` to `{artifact}` in "
                f"the vault, whole — an extract cannot carry a digest — and re-derive the "
                f"quoted digest from the bytes there with `sha256sum {artifact}`; the "
                f"digest the item quotes is that output.")
    return (f"The witness bytes have no history: copy `{witness}` to `{artifact}` in "
            f"the vault (an extract reproducing the same numbers is fine) and "
            f"re-derive the quoted report from the committed bytes with "
            f"`{B._rederive_command(artifact, ext)}` — that output is the figure the "
            f"item quotes.")


def _sibling_copy_demand(first: str, ext: str, artifact_dir: str, *,
                         hash_quoted: bool = False) -> str:
    """The re-aimed demand, for a witness the vault already holds an extract of (#3872).

    Same two wordings for the same reason: the cheap check is "re-derive from the copy that
    is already committed", and what re-deriving means depends on the figure the item quoted.
    """
    if hash_quoted:
        return (f"The witness bytes have no history: `{artifact_dir}` already commits a "
                f"copy of them — re-derive the quoted digest from `{first}`, and publish "
                f"your own extract only if that output does not reproduce the digest the "
                f"item quotes: `sha256sum {first}`")
    return (f"The witness bytes have no history: `{artifact_dir}` already commits a copy "
            f"of them — re-derive the quoted report from `{first}`, and publish "
            f"your own extract only if that output does not reproduce the figure the "
            f"item quotes: `{B._rederive_command(first, ext)}`")


def add_witness_artifact_clause(clauses, text, *, probe=B.in_git_tree,
                                artifact_dir: str = B.WITNESS_ARTIFACT_DIR,
                                route_to_owed: bool = False, item_id=None):
    """One clause asking for the witness bytes, when a named witness has none.

    `clauses` is the contract as it stands on the way to the item; `text` is
    everything the item states (body, evidence, acceptance prose). The input
    list is never mutated. Fires only when all three hold:

      * no clause already names where the bytes go (`_ARCHIVE_MENTION_RX`);
      * the bytes are on this disk: a token that resolves to no file is
        dropped, because the clause would order a round to copy nothing. #1886
        got one on `/manifest.json` — a token the old unanchored pattern cut out
        of the relative `chrome-extension/manifest.json`, which then walked
        through `in_git_tree` as "in no tree" for the separate reason that the
        path does not exist;
      * `probe` says such a path is in no git tree — those bytes exist in no
        history anywhere;
      * the file is no larger than `WITNESS_MAX_BYTES` (#2267 clause 3): no CLAUSE
        is generated for a token over the bound, so the generator stops ordering
        3 MB of a daily-rotating service log into a vault directory that already
        holds 108 MB. It is not skipped into silence the way an absent file is: an
        over-bound token that clears the two probes above publishes one extract
        demand to `owed` under `route_to_owed` (#2382), and never a clause;
      * its basename is not shared by some other path in the clauses or the body
        that `probe` places IN a git tree (#2267 clause 1, `_tree_basenames`).
        Those bytes are pinned under this very name already — #2248's row was
        tracked at `tests/fixtures/vllm_prefix_miss_2026-10-01.json` while the
        clause still ordered a copy of the same name out of `~/lloyd-data`, and the
        re-triage stamp that did it had cleared the front matter, so no cap was in
        the way. Compared across paths by basename, never by substring, and never
        against the candidate itself;
      * the 64-hex digest it states as that file's hash still matches the bytes on
        disk (#4342, `_witness_digest_state`). An item that quotes a witness BY digest
        and whose digest has since moved on cannot be served by any copy: the bytes a
        copy can carry now are a later instant's, so the demand would order a witness
        that contradicts the claim it is meant to pin. The token is dropped and the
        owed list is told why — a witness that quietly stops being asked for is
        indistinguishable from a guard that never fired;
      * the contract has room: at `MAX_CLAUSES` the list comes back UNCHANGED,
        because a generated clause must never evict, truncate or reorder one
        that was authored.

    Every one of these skips is per-token and none may become a blanket skip: each
    leaves every other token in the text eligible. The opposite direction — a
    witness that must still get its clause — is pinned by
    `test_a_witness_whose_basename_no_tree_path_names_still_gains_one_clause` for
    the basename, by the under-bound control inside
    `test_a_witness_beyond_the_size_bound_is_skipped_like_an_absent_one` for size, and
    by `test_one_stale_digest_witness_does_not_skip_the_live_one_beside_it` for the
    digest.

    Exactly one clause goes on, for the first out-of-tree path in reading order,
    worded over committed bytes with no time shape — the owed route publishes it as
    written, and a demand phrased as a future observation ("after a day of traffic")
    would read as an instruction to wait rather than an instruction to copy. Its
    command is `wc -l -c` for a file witness — the line count a jsonl's report is
    quoted from, plus the byte count, because every canonical extract in this
    convention is ONE line and `wc -l` alone printed `1` for all six
    `tests/fixtures/vllm_prefix_miss_*.json` whose bytes ranged 169,348 to 263,748,
    pinning nothing (#2267 clause 4) — and a sqlite count for a store. A witness the
    text quotes by a digest that still matches its bytes is the exception no count can
    serve: its command is `sha256sum` of the artifact in the vault, and its demand asks
    for the file whole, because the figure to reproduce IS a digest (#4342 clause 2).

    At most ONE over-bound note goes out per call, for the first such token in
    reading order, and independently of whether a clause also went out: a body
    citing a 30 MB ledger and a 200 KB jsonl gets the jsonl's clause AND the
    ledger's extract note, because the two claims are served by different bytes
    and the silence #2382 is about was the ledger's half. The note is owed-only and
    never joins `every` — it asks for a vault commit like the clause does, and it
    is capped by `MAX_CLAUSES` exactly like the clause is, since an item whose
    contract is at its budget has spent it on both lists (#2289 clause 4).

    `route_to_owed=True` changes only the DESTINATION, and that is the whole of
    #2289. The demand orders a commit on the vault, and a code round's diff never
    contains the vault (`vault_round.py:1161`): its only validated route is
    `automod_vault_land`, which runs after the round, so putting the demand in
    `acceptance_clauses` asked a round to land the change it was still proposing —
    #2283's round SM_20261006_102005 was refused twice on one such clause while the
    rail it exists to repair was what refused it. With the flag the function returns
    `(clauses, owed)`, where `owed` holds the copy demand and/or the one over-bound
    extract note (#2382) and nothing else; the caller owes them, and every other
    rule here — trigger, wording, the two #2267 skips, `_ARCHIVE_MENTION_RX`, the
    on-disk gate and the cap — applies unchanged. With the flag off (the default,
    every other caller) there is no owed destination, so neither demand is
    generated and the answer is the plain list, as before.

    WHERE the bytes go is `_witness_artifact`'s decision, not a bare basename
    (#2513). The free name is still `backlog/data/<basename>`, the #1756 shape; a
    name the vault ALREADY holds becomes a dated sibling, because a witness is an
    immutable extract and overwriting one destroys the evidence another item's test
    reads. This is the third time that mattered: #2228 and #2383 were handed the
    taken path `backlog/data/gate.json` and stopped, while #2512's round obeyed the
    clause literally at vault `d5f9f542` and deleted #1883's witness, which is what
    turned three tests on main red. `item_id` is the id whose triage is emitting the
    demand — it goes into the sibling's name so a reader can get from the name back
    to the item — and `record_verdict` is the only caller that has one, so every
    other caller leaves the date-and-directory form.

    What this function cannot do is undo a clause it already emitted. A
    `vault`-surface item has no amendment route: `amend_clause` resolves the
    graded review through `last_graded_review(ledger, round_id)`, which reads
    code-round `review` rows, and a vault landing has no round and so no
    `round_id` to resolve. `amend_clause` says so by name
    (`VAULT_NO_AMENDMENT_ROUTE`) rather than claiming the review did not judge
    the clause. A bad generated clause on a vault item is cleared by editing
    the item, which is what #1889 does for #1886, or by filing a blocker.
    """
    every = list(clauses or ())
    owed: list[str] = []

    def _answer():
        """The list, or `(list, owed)` when the caller asked for the owed route.

        Two shapes rather than always a tuple because the other callers are triage
        passes holding a clause list, and rewriting every one of them to unpack a
        value it never asked for is a wider change than moving one destination.
        #2289.
        """
        return (every, owed) if route_to_owed else every

    if any(B._ARCHIVE_MENTION_RX.search(str(c)) for c in every):
        return _answer()
    seen: set[str] = set()
    witness = ""
    # True when the chosen witness is the file whose digest this text quotes and that
    # digest still matches its bytes. It decides which re-derive command the demand names:
    # a digest claim is reproduced by `sha256sum`, and a `wc -l -c` demand attached to a
    # hash quote claims to pin a hash with a line count (#4342 clause 2, and the note the
    # triage left for whoever implemented this).
    hash_quoted = False
    # The first token, in reading order, that is on disk, over the size bound and in
    # no tree — the one the owed extract note is about (#2382). Kept apart from
    # `witness` because the two demands are not alternatives: a body can cite a big
    # store and a small one, and the clause for the small one does not answer for the
    # big one. Recorded only once, so three over-bound tokens still yield one note.
    over_bound = ""
    # Where a pinned copy may be cited: the contract's own clauses AND the item
    # text. #2267 clause 1 is precisely the case where the clause is the only place
    # the tracked path appears, so scanning the body alone would miss it.
    pinned_haystack = "\n".join([*[str(c) for c in every], str(text or "")])
    # Basenames of the paths in that haystack that the probe says ARE in a git
    # tree. Resolved lazily, only once a candidate needs the question, because the
    # probe walks the filesystem and the common case is no candidate at all.
    tree_basenames: "set[str] | None" = None
    # #4342: whether this text quotes ANY witness by digest at all. Computed once and
    # consulted per token, because the question below ("does the quoted digest still
    # match?") can only be answered by reading the file, and a text with no 64-hex in it
    # must not cost a read of anything. One text, one answer: the digest set does not
    # change between tokens of the same call.
    digests_in_text = bool(B._WITNESS_DIGEST_RX.search(str(text or "")))
    # The first token skipped for a stale quoted digest, kept so the skip is reportable
    # rather than silent — a witness that quietly stops being asked for is
    # indistinguishable from a guard that never fired, which is #2382's failure mode in a
    # new guise.
    stale_quoted = ""
    for tok in B.WITNESS_PATH_RX.findall(str(text or "")):
        if tok in seen:
            continue
        seen.add(tok)
        if not B._witness_is_on_disk(tok):
            continue
        # Asked before the tree probe, because it decides WHICH of the two demands
        # this token is eligible for, not whether the token is eligible at all.
        too_big = not B._witness_within_size_bound(tok)
        if not probe(tok):
            if tree_basenames is None:
                tree_basenames = B._tree_basenames(pinned_haystack, probe)
            if Path(B._witness_target(tok)).name in tree_basenames:
                continue
            if too_big:
                # Bytes no round may be asked to copy whole, and bytes whose byte
                # count could not be read at all are not "over the bound" — they are
                # unmeasurable, and an unmeasurable witness is owed no note claiming a
                # figure about it. Hence the second look at `_witness_size`.
                if not over_bound and B._witness_size(tok) is not None:
                    over_bound = tok
                continue
            state = (B._witness_digest_state(tok, str(text or ""))
                     if digests_in_text else "absent")
            if state == "stale":
                # The instant this item quoted is gone: bytes copied now are a different
                # file, so the copy would carry this item's figure and prove nothing. The
                # token is dropped exactly as an absent file is, and every OTHER token in
                # the text stays eligible — this is the per-token skip #4342 asks for, not
                # a blanket one (clause 5 pins one stale plus one live = one demand).
                # Only the token is RECORDED here; the owed note is emitted below the
                # clause cap, where every other destination is gated.
                stale_quoted = stale_quoted or tok
                continue
            if not witness:
                # No `break`: the loop keeps going so an over-bound token LATER in
                # reading order is still seen, and stops on its own once both
                # destinations have a candidate.
                witness = tok
                # "match": the quoted digest still equals these bytes, so the demand has
                # to be re-derivable as a digest and not as a line count (#4342 clause 2).
                hash_quoted = state == "match"
        if witness and over_bound:
            break
    # The cap gates BOTH destinations, and now literally so: a note accumulated
    # inside the loop that escaped this line would restart exactly the quiet growth
    # past the item's own prose that the budget exists to stop (#1909, #2289
    # clause 4, and #2382 clause 5 for the note).
    if len(every) >= B.MAX_CLAUSES:
        return _answer()
    if over_bound and route_to_owed:
        owed.append(B._over_bound_extract_demand(over_bound, artifact_dir,
                                              item_id=item_id))
    if stale_quoted and route_to_owed:
        owed.append(B._stale_digest_note(stale_quoted))
    if not witness:
        return _answer()
    # #2513: never a name the vault already holds. The basename used to be lifted
    # straight off the SOURCE file, so a source named `gate.json` was ordered onto
    # `backlog/data/gate.json` — the slot #1883's witness occupies — and the round
    # that obeyed the clause is what made main red.
    artifact = B._witness_artifact(witness, artifact_dir, item_id=item_id)
    ext = Path(witness).suffix.lower().lstrip(".")
    # #3218: the bytes of a nightly eval baseline are already decided — the file is
    # quoted whole, its basename is unique per run, and nothing but its own run rewrites
    # it — so the guard puts them in history instead of ordering a round to `cp` them.
    # Every other unarchived witness falls through to the demand below, which is emitted
    # and routed exactly as #2289/#2382/#2513 left it.
    if B._is_anchor_witness(witness):
        sink = owed if route_to_owed else every
        res = B.commit_anchor_witness(witness, artifact, item_id=item_id)
        if res["reason"] == B.ANCHOR_WITNESS_DONE:
            # Either this exact baseline is already tracked, or the slot holds another
            # item's bytes and is refused with a note (#2513). Only the refused half has
            # something to say; the archived half must not add an empty owed line, which
            # would read to a human as a refusal with no reason.
            if res["note"]:
                sink.append(res["note"])
            return _answer()
        if res["ok"]:
            sink.append(res["note"])          # the archive record, figures and all
            return _answer()
        if res["reason"] not in B._ANCHOR_WITNESS_FALLBACKS:
            # A guard refused it (ceiling / ignored target / family budget). Say so
            # loudly and do NOT also order the copy that was just refused: a demand the
            # route knows is unexecutable is the round this item exists to stop.
            sink.append(res["note"])
            return _answer()
        # The copy failed for a reason that is not a judgement — git would not answer,
        # the commit did not land. Fall through to the demand so the witness is still
        # witnessed, with the failure named beside it rather than swallowed.
        sink.append(res["note"])
    # The vault may already hold an extract of THESE bytes. #3175 taught the over-bound
    # note above to ask; this is the whole-file demand every witness UNDER the ceiling
    # reaches, and it never asked — it ordered a copy whatever the vault held. The
    # asymmetry is the whole of #3872: one round's `gate.json` is ONE file, while every
    # item quoting it gets a demand carrying ITS OWN item number, so two item ids naming
    # one report are two free slots to `_artifact_taken` and the vault gains identical
    # bytes under two names — the #3161/#3396 near-miss, which no later reader
    # distinguishes. Aimed at the committed copy instead, the demand keeps a new file
    # available as a CONDITIONAL in #3175's shape: a sibling's tail says it came from this
    # source, not that its figures answer the one this item quotes, so re-deriving from it
    # is the cheap check that decides. Unlike the copy demand it replaces, it names no
    # source path and books no fresh slot — that is what stops the second copy.
    sibs = B._committed_witness_sibling_names(witness, artifact_dir)
    if sibs:
        first = f"{artifact_dir}/{sibs[0]}"
        (owed if route_to_owed else every).append(
            B._sibling_copy_demand(first, ext, artifact_dir, hash_quoted=hash_quoted))
        return _answer()
    # Past the `MAX_CLAUSES` check above, which is where both destinations are
    # gated. #2289 clause 4 pinned that gate for the copy demand; #2382 clause 5
    # pins it for the note.
    (owed if route_to_owed else every).append(
        B._copy_demand(witness, artifact, ext, hash_quoted=hash_quoted))
    return _answer()


# Bound onto `backlog`, so `backlog.<name>`, `from scripts.automod.backlog import
# <name>`, a bare call from code still in backlog, and a monkeypatch on backlog
# all keep meaning what they meant before the move.
WITNESS_NAMES = (
    "_vault_witness_dir",
    "_artifact_taken",
    "_RETIRED_WITNESS_NAMES",
    "_ROLLING_STATE_NAMES",
    "_witness_name_must_be_dated",
    "_witness_artifact",
    "_DATED_WITNESS_RE",
    "_witness_tag",
    "_witness_family_tail",
    "_witness_family_pattern",
    "_committed_witness_siblings",
    "_committed_witness_sibling_names",
    "ANCHOR_WITNESS_PREFIX",
    "ANCHOR_WITNESS_ROOT",
    "ANCHOR_WITNESS_MAX_FILES",
    "ANCHOR_WITNESS_MAX_TOTAL_BYTES",
    "ANCHOR_WITNESS_FIGURES",
    "ANCHOR_WITNESS_DONE",
    "_ANCHOR_WITNESS_FALLBACKS",
    "_is_anchor_witness",
    "_anchor_witness_family",
    "_blob_figure",
    "commit_anchor_witness",
    "_over_bound_extract_demand",
    "_tree_basenames",
    "_pinned_candidates",
    "_copy_demand",
    "_sibling_copy_demand",
    "add_witness_artifact_clause",
)
for _name in WITNESS_NAMES:
    setattr(B, _name, globals()[_name])
