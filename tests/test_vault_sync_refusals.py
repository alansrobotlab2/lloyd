"""#539 — the two things Obsidian Sync refuses, and the tree that must not hold them.

`~/obsidian`'s only off-box copy is Obsidian Sync (`obsidian-headless`, supervisor program
`agent-obsidian-sync`). That transport refuses two different classes of path, and refused
is not the same as failed: the client logs one line, the supervisor keeps reporting
`RUNNING … uptime 2 days`, every endpoint probe stays green, and the file sits single-copy
forever. Measured at triage on 2026-09-13 while the drop was happening.

  1. **any file over 5 MiB** — a plan property of the hosted service, printed by the
     client as `File too large to sync (5.43 MB, max 5.00 MB)
     projects/lloyd/voice/references/dave_cullen/source_nS8PvZv3v0U.webm`. Not a config
     anywhere on this box, so nothing local can raise it. Exactly one file hit it: the
     5691071-byte Dave Cullen voice-source recording — the asset whose audio was already
     lost once on the 08-22 rebuild.
  2. **`.git/` in its entirety** — sync *configuration*, printed by
     `ob sync-status --path ~/obsidian` as `Excluded folders: .git`. 60 MB of vault
     history, and `git -C ~/obsidian remote -v` returns 0 lines, so nothing else carries
     it. Provisioning that copy needs credentials and a person; it is NOT delivered here,
     and neither the note nor `~/vault-external/README.md` claims otherwise.
  3. **everything the first two do not name, by file type** — also configuration, and
     the boundary this guard must not overclaim. `ob`'s own banner, reprinted at every
     sync start, reads `File types: image, audio, pdf, video`: markdown plus four media
     categories, an allow-list. A `.jsonl`, `.db`, `.log`, `.ndjson` or `.patch` is in
     none of them, so it is refused by configuration exactly as `.git` is and no size
     ceiling can apply to it — it was never eligible to upload. That is #1935: the
     ceiling probes were red on `backlog/data/promotions.jsonl` (10,917,874 bytes, the
     ledger extract #1903 pinned), a file whose bytes the transport has never carried,
     against a witness route that mints extracts with no size bound. Both probes now
     prune non-markdown files under `backlog/data/` and nothing else — every `.md` and
     every media file, wherever they sit, is still graded. The banner line the prune
     stands on is pinned by `test_the_transport_carries_only_the_types_its_prune_assumes`,
     so widening the transport's categories turns this file red rather than quiet.

What the fix did, and what these tests hold:

  * the blob moved byte-identically (sha256 verified before the vault copy was removed) to
    `~/vault-external/projects/lloyd/voice/references/dave_cullen/`, a named out-of-vault
    path, and `~/obsidian/.../source-blob-location.md` records where it is, its sha256,
    the source id `nS8PvZv3v0U` and both cut ranges — so removing the blob loses no
    provenance, while the `.wav`/`.lab` references that *do* sync stay untouched;
  * `knowledge/software/obsidian-headless-sync-quota-silent-failure.md` states both
    refusals by mechanism, each with the command that proves it — because #539 first
    inferred refusal 2 from `grep -c '\\.git/'` over the sync log, a count that reads 0
    both when `.git` is excluded *and* when the log rotated.

Process boundaries crossed, each with the test that crosses it: the hosted transport's own
emitted line is parsed, not restated
(`test_ceiling_in_the_note_is_the_clients_number_not_the_authors`); the live vault tree is
read by a shell `find` run as a subprocess, the command the acceptance itself names
(`test_acceptance_find_pipeline_reports_no_over_cap_file`); the sync client's stderr log,
written by a *different process* and rotated at 10 MB, is the input to
`test_no_refused_path_still_lives_in_the_vault`; and the vault's git index, which the file
tools never touch, is read through `git ls-files` in a subprocess by
`test_the_move_reached_the_vault_git_index`. The transport's *configuration*, which no
file on this box states, is read from the banner the client prints into the supervisor
program's stdout and parsed out of the retained windows by
`test_the_transport_carries_only_the_types_its_prune_assumes`.

One consequence to keep in plain sight, since it is the honest cost of prune 3: files
outside those four categories have no off-box copy at any size, so this guard was never
giving the witness extracts durability — `backlog/data/` is single-copy on this box
today, 10 MB ceiling or not, alongside the 60 MB of `.git`. Recording that as a fact is
what #1935 does; provisioning a real off-box copy for either is a person's action and
stays owed.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

import yaml

from app.paths import ACCOUNT_HOME, production_data_root

# The vault is read through `$HOME` (a symlink to the real one inside a gate); the
# sync log and the out-of-vault blob are production files named absolutely, so they
# come off the account home, which a gate's `HOME=<round>/home` does not move.
VAULT = Path.home() / "obsidian"
SYNC_LOG_DIR = production_data_root() / "logs" / "services"
SYNC_ERR = SYNC_LOG_DIR / "agent-obsidian-sync.err"

#: Obsidian Sync's per-file ceiling. The client divides by 1024**2: it printed a
#: 5691071-byte file as "5.43 MB" (5691071 / 1048576 = 5.4274); a decimal megabyte would
#: have printed 5.69. Pinned by test_ceiling_in_the_note_is_the_clients_number_not_the_authors.
SYNC_MAX_BYTES = 5 * 1024 * 1024

#: The subtree an item's owed-witness clause commits frozen extracts into —
#: `voice.log`, `usage.db`, `*.ndjson`, `promotions.jsonl`, and the `.md` sidecar
#: that names each one's commit and md5.
WITNESS_DIR_REL = ("backlog", "data")

#: The non-markdown categories `ob` prints on its own banner line
#: `File types: image, audio, pdf, video` — an allow-list, not a deny-list. The
#: categories this guard's prune assumes are NOT carried; pinned by
#: `test_the_transport_carries_only_the_types_its_prune_assumes`.
CARRIED_NONMD_TYPES = {"image", "audio", "pdf", "video"}


def witness_extract(rel: Path) -> bool:
    """True for a machine-written frozen extract: non-markdown under `backlog/data/`.

    Exempt from the ceiling probe on the same ground `.git` is exempt — the
    transport is *configured* not to carry it. The client's own banner, printed by
    every sync start into `agent-obsidian-sync.log`, reads
    `File types: image, audio, pdf, video`: markdown plus four media categories.
    `.jsonl`, `.db`, `.log`, `.ndjson` and `.patch` are in none of them, so
    `backlog/data/promotions.jsonl` — 10,917,874 bytes, vault `0d96fdb0`, the
    26,903-row ledger extract #1903 pinned — was never a candidate for a 5 MiB
    upload, exactly as the 60 MB of `.git` never was. What this guard is about is
    what the transport *would* carry and might refuse; counting what it cannot carry
    makes the check unsatisfiable against a witness route that mints extracts with
    no size bound. That is how #1935 went red: the file landed at 16:42 on
    2026-09-30 and both ceiling nodes have been red since, with nothing any round
    could do about it except delete the witness #1903 exists to pin.

    Scoped to the subtree, deliberately NOT to a list of exempt extensions:
    exempting by type would mean enumerating every image/audio/video suffix the
    client recognises, and one forgotten suffix — `m4a`? `webp`? — would quietly
    take a real, carryable, over-cap asset out of the guard, which is the standing
    class rule about hand-maintained allow-lists over an open-set corpus. Under this
    rule the only way to escape the probe is to put a file under `backlog/data/`,
    and markdown *there* is still graded — a keyless sidecar is exempt from the
    concept-document gates (#1934) but a 6 MiB `.md` is still something Sync would
    refuse. Both halves pinned by
    `test_python_walk_agrees_with_the_find_and_detects_a_planted_offender`, and the
    config line the whole thing rests on pinned by
    `test_the_transport_carries_only_the_types_its_prune_assumes`.
    """
    return rel.parts[:2] == WITNESS_DIR_REL and rel.suffix.lower() != ".md"


#: The same predicate in the acceptance's own `find` syntax, so the shell probe and
#: the Python walker prune one set rather than two that can drift. `-prune` on a
#: plain file is a no-op; the `-o` is what drops it, and the pattern needs
#: `backlog/data/` in the path, so a directory named `data` anywhere else — the
#: hole #1934 had to avoid in the OKF gate — is unaffected.
WITNESS_PRUNE = r"\( -path '*/backlog/data/*' ! -name '*.md' -prune \) -o "

REF_DIR = "projects/lloyd/voice/references/dave_cullen"
BLOB_REL = f"{REF_DIR}/source_nS8PvZv3v0U.webm"
POINTER_REL = f"{REF_DIR}/source-blob-location.md"
POINTER = VAULT / POINTER_REL
OUT_OF_VAULT_BLOB = ACCOUNT_HOME / "vault-external" / BLOB_REL
SOURCE_NOTE = VAULT / "projects/lloyd/voice/voice-source-dave-cullen.md"
KNOWLEDGE_NOTE = Path(os.environ.get("LLOYD_SYNC_NOTE") or
                      VAULT / "knowledge/software/obsidian-headless-sync-quota-silent-failure.md")
SYNC_PROGRAM_CONF = (Path(__file__).resolve().parent.parent
                     / "agent-services" / "supervisor" / "conf.d" / "agent-obsidian-sync.conf")

#: The reference set the voice clone actually loads. The move had to touch none of it, so
#: each is pinned by size *and* content hash — a size-only check would not notice a file
#: re-encoded "to make it fit", which is the other half of this item's fix list.
DERIVED_SET = {
    "dave_cullen_001.wav": (1538286,
                            "f4ece0677797ca5559a8818211c039dc4348a8b7630a0710e308056d614eaf3b"),
    "dave_cullen_002.wav": (1241934,
                            "14d6d0d2218fa0dc14e5b7111785260a5f3d4c64fbfcfc1449bb93814dcd1ae7"),
    "dave_cullen_001.lab": (218,
                            "4f2217e4745ba1ec1c375dda867b21d8862ae8b491554cd76bb5809cb06e8bba"),
    "dave_cullen_002.lab": (131,
                            "49047f7fc732a64bc78e9eb50aa53c04156c85dcf1f53c8009b3de748dcd6c7f"),
    "dave_cullen_source.json": (2502,
                               "989f04aa3a91197d1b33b1134550ab65bb770650dd2226237d4a4955309f0f40"),
}


def _frontmatter(path: Path) -> dict:
    """The pointer note's YAML block — the machine-readable half of its claim."""
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "---", f"{path} must open with front matter, got {lines[0]!r}"
    end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    fm = yaml.safe_load("\n".join(lines[1:end]))
    assert isinstance(fm, dict), f"{path} front matter is not a mapping"
    return fm


def _bash(pipeline: str) -> subprocess.CompletedProcess:
    """Run a pipeline exactly as a person would type it — the acceptance is shell."""
    return subprocess.run(["bash", "-lc", pipeline], capture_output=True, text=True,
                          timeout=180)


def _find_over_cap(root_expr: str, prune: str = WITNESS_PRUNE) -> str:
    """The acceptance's own `find` over `root_expr`, with the `.git` prune and `prune`.

    `root_expr` is passed unquoted so `~/obsidian` expands the way the acceptance is
    typed; `-H` is what follows the symlink a gate's round home puts there. The
    default carries `WITNESS_PRUNE`, and the node that needs the unpruned denominator
    passes `prune=""`.
    """
    return (f'find -H {root_expr} -path {root_expr}/.git -prune -o {prune}'
            f'-type f -size +{SYNC_MAX_BYTES}c -printf \'%s\\t%p\\n\'')


def _find_offenders(pipeline: str) -> list[Path]:
    run = _bash(pipeline)
    assert run.returncode == 0, f"find failed: {run.stderr[:400]}"
    return [Path(line.split("\t", 1)[1])
            for line in run.stdout.splitlines() if line.strip()]


#: The shapes a ceiling guard has to tell apart, as relative paths and sizes. One
#: definition, read by both the shell node and the walker node, so the two
#: denominators cannot be pinned against two different fixtures.
CEILING_TREE: dict[str, int] = {
    "big.webm": SYNC_MAX_BYTES + 1,                      # media: carried, refused
    "edge.wav": SYNC_MAX_BYTES,                          # exactly at the cap: syncs
    ".git/objects/pack/pack-x.pack": SYNC_MAX_BYTES * 40,  # excluded folder
    "backlog/data/pinned.jsonl": SYNC_MAX_BYTES * 3,     # extract: type not carried
    "backlog/data/big-sidecar.md": SYNC_MAX_BYTES + 1,   # markdown IS carried
    "backlog/1935-big-item.md": SYNC_MAX_BYTES + 1,      # a note one directory up
}

#: What each of those three must be reported as, written out rather than derived: an
#: expectation computed from `witness_extract` would move with the predicate it is
#: meant to catch, which is the shape of test this file exists to prevent.
CEILING_REPORTED = {"big.webm", "backlog/data/big-sidecar.md", "backlog/1935-big-item.md"}


def _plant_ceiling_tree(root: Path) -> None:
    """Plant `CEILING_TREE` under `root`.

    Files are `truncate`d, not written: only `st_size` matters to either side, and a
    sparse 209 MB pack file costs the tmp directory nothing.
    """
    for name, size in CEILING_TREE.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            handle.truncate(size)


def over_cap_files(root: Path, limit: int = SYNC_MAX_BYTES) -> list[tuple[int, Path]]:
    """Files under `root` bigger than `limit`, skipping what the transport cannot carry.

    Two skips, one per reason. The `.git` skip mirrors the acceptance's
    `-path ./.git -prune`: `.git` is refused by *configuration* (refusal 2), whose
    remedy is a git remote — a different fix from the per-file ceiling (refusal 1).
    Reporting it here would make this check unsatisfiable, which is why the
    acceptance prunes it and not why it is safe. The `witness_extract` skip is the
    same shape one level further in: `File types: image, audio, pdf, video` is an
    allow-list, so the frozen extracts under `backlog/data/` are refused by that same
    configuration and no ceiling applies to a file that was never eligible to upload.

    Everything else is graded at full width: every markdown file anywhere in the
    tree, and every non-markdown file outside `backlog/data/` — including media,
    which the transport does carry and does refuse over 5 MiB.
    """
    offenders: list[tuple[int, Path]] = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if rel.parts[:1] == (".git",):
            continue
        if witness_extract(rel):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        size = path.stat().st_size
        if size > limit:
            offenders.append((size, path))
    return offenders


# ── clause 1: nothing left in the tree is silently un-syncable ─────────────────────────


def test_acceptance_find_pipeline_reports_no_over_cap_file():
    """No file under `~/obsidian` that the transport could carry is over its 5 MiB ceiling.

    The probe is the acceptance's own `find` with one added clause — `WITNESS_PRUNE`,
    the non-markdown extracts under `backlog/data/` that
    `File types: image, audio, pdf, video` excludes — and the same command without that
    clause runs beside it, so the addition cannot widen in silence: every file it hides
    has to be something `witness_extract` exempts on its own terms. The vault today
    hides exactly one file that way, `backlog/data/promotions.jsonl` at 10,917,874
    bytes; an offender of any other shape — a clip in a note tree, a big `.md` in the
    witness subtree — leaves this node red.
    """
    graded = _find_offenders(_find_over_cap("~/obsidian"))
    assert graded == [], f"files over Sync's ceiling are still in the vault:\n{graded}"

    # The prune's own control: the same walk without it. Anything it reports that the
    # walker's predicate does not exempt is a file this check has no business hiding.
    for path in _find_offenders(_find_over_cap("~/obsidian", prune="")):
        assert witness_extract(path.relative_to(VAULT)), (
            f"the ceiling probe hides {path}, which `witness_extract` does not exempt: "
            "the denominator has widened past what the transport excludes")

    # A green verdict on an empty walk proves nothing — #1028 is exactly this shape — so the
    # same command without the size filter has to show a real tree behind it.
    walk = _bash('find -H ~/obsidian -path ~/obsidian/.git -prune -o -type f -printf "%p\\n" | wc -l')
    n_files = int(walk.stdout.strip())
    assert n_files > 1000, f"the vault walk saw only {n_files} files — this check saw nothing"


def test_shell_probe_and_python_walker_prune_the_same_tree(tmp_path):
    """The pair's claim is agreement, so test agreement: both denominators, one fixture.

    `find` with `WITNESS_PRUNE`, rooted at the fixture exactly as the acceptance roots
    it at the vault, and `over_cap_files` must report the same three relative paths out
    of `CEILING_TREE`. Before #1935 the shell clause and the Python skip were written
    twice, in two languages, and only the two live-vault runs kept them in step — on a
    tree with one over-cap file, where any prune that hides that one file passes. Here
    an exempt extract, a graded `.md` in the same directory, a graded media file, a
    graded note one directory up, the at-cap edge case and a 40x `.git` pack exist at
    once, so widening either side on its own — dropping `-name '*.md'` from the shell
    clause, or broadening `witness_extract` to any directory named `data` — breaks the
    equality rather than passing both halves.
    """
    _plant_ceiling_tree(tmp_path)
    shell = {p.relative_to(tmp_path).as_posix()
             for p in _find_offenders(_find_over_cap(str(tmp_path)))}
    walker = {p.relative_to(tmp_path).as_posix() for _, p in over_cap_files(tmp_path)}
    assert shell == CEILING_REPORTED, (
        f"the shell probe reported {sorted(shell)}, not {sorted(CEILING_REPORTED)}")
    assert walker == CEILING_REPORTED, (
        f"the walker reported {sorted(walker)}, not {sorted(CEILING_REPORTED)}")


def test_python_walk_agrees_with_the_find_and_detects_a_planted_offender(tmp_path):
    """The walker is falsifiable on sizes: it reports each of `CEILING_REPORTED`, and
    skips the 40x pack under `.git`, the file sitting *exactly* at the cap, and the 3x
    extract under `backlog/data/`.

    Sizes are the axis this node owns — a walker reporting the at-cap file or the pack
    is measuring the wrong denominator — while `witness_extract`'s own boundary (a
    graded `.md` inside the exempt subtree, a graded media file outside it) is pinned
    against the shell clause by `test_shell_probe_and_python_walker_prune_the_same_tree`.
    """
    _plant_ceiling_tree(tmp_path)
    found = {p.relative_to(tmp_path).as_posix(): size
             for size, p in over_cap_files(tmp_path)}
    assert found == {name: CEILING_TREE[name] for name in sorted(CEILING_REPORTED)}, (
        f"expected exactly {sorted(CEILING_REPORTED)} at their planted sizes; a walker "
        "that reports the 40x .git pack file, the file sitting exactly at the cap, or "
        f"the extract under backlog/data is measuring the wrong denominator: {found}")
    assert over_cap_files(VAULT) == [], f"over-cap files in the live vault: {over_cap_files(VAULT)}"


# ── #1935 clause: the config the two prunes rest on, read off the transport ───────────


def _sync_banner_field(field: str) -> str:
    """The most recent `field: value` the sync client printed into its own log.

    The client prints its configuration banner (`File types:`, `Excluded folders:`)
    on every start, into the supervisor program's stdout, which rotates — so the
    newest window that holds the banner is the live answer, the same rotation rule
    `test_every_diagnostic_one_liner_names_its_evidence_and_rotation_bound` enforces
    on the note. Windows are the conf's own bound, read from `SYNC_PROGRAM_CONF`.
    """
    windows = sorted(SYNC_LOG_DIR.glob("agent-obsidian-sync.log*"),
                     key=lambda p: p.stat().st_mtime)
    assert windows, f"no sync log window under {SYNC_LOG_DIR} — the check lost its input"
    latest, seen = None, 0
    for window in windows:
        with window.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip().startswith(f"{field}:"):
                    seen += 1
                    latest = line.split(":", 1)[1].strip()
    assert latest is not None, (
        f"no `{field}:` banner in any of the {len(windows)} retained sync log windows — "
        "the transport no longer prints the configuration this guard's prunes assume, "
        "so they are unverified, not true")
    assert seen >= 1, f"`{field}:` was read but counted {seen}"
    return latest


def test_the_transport_carries_only_the_types_its_prune_assumes():
    """The one config line both prunes stand on, read from the client's own banner.

    `File types:` is an allow-list of what Obsidian Sync carries beyond markdown.
    Measured 2026-09-30: `image, audio, pdf, video`, printed 11 times across the
    retained log windows, beside `Excluded folders: .git`. That is why
    `backlog/data/promotions.jsonl` is not a silently-un-syncable file: a `.jsonl`
    is in none of those categories, so it was never eligible to upload — the same
    configuration-class refusal the `.git` prune already concedes.

    Falsifiable in the direction that matters: if a person turns on the category
    that holds machine files (Obsidian calls it `other`, and a catch-all like `all`
    is the same hole), or the banner stops being printed, this node goes red and the
    `witness_extract` exemption has to be withdrawn and the extracts moved out of the
    vault instead. It does not fail on a *narrowing* — dropping `video` would only
    make this guard stricter than the transport.
    """
    carried = {c.strip().lower() for c in _sync_banner_field("File types").split(",")
               if c.strip()}
    assert carried, "the banner printed an empty File types list"
    assert not carried & {"all", "other", "others", "files", "all files"}, (
        f"the transport now carries {sorted(carried)}, which includes the category "
        "machine-written extracts fall in — `witness_extract` must stop exempting them")
    assert carried <= CARRIED_NONMD_TYPES, (
        f"an unrecognised category in `File types: {sorted(carried)}`; this guard's "
        "prune was written against image/audio/pdf/video and has to be re-read")

    excluded = _sync_banner_field("Excluded folders")
    assert ".git" in [f.strip() for f in excluded.split(",")], (
        f"`Excluded folders: {excluded}` no longer names .git, so the walker's other "
        "skip is no longer a configuration refusal")


# ── clause 2: a path the transport refused must not still exist ────────────────────────


def _refused_paths() -> list[str]:
    """Paths named `File too large to sync` across the live `.err` and its rotated siblings.

    The file is written by `agent-obsidian-sync`, another process, and rotates at 10 MB —
    measured at 10,239,373 B — so one window is not the whole history and an empty window
    is not a pass. The durable guard for this clause is the filesystem walk above; this one
    reads what the transport actually said.
    """
    windows = [SYNC_ERR] + sorted(SYNC_LOG_DIR.glob("agent-obsidian-sync.err*"))
    names: set[str] = set()
    for window in windows:
        if not window.exists():
            continue
        run = _bash(f"grep -o 'File too large to sync.*' {window} | sort -u")
        assert run.returncode in (0, 1), f"grep failed on {window}: {run.stderr[:300]}"
        for line in run.stdout.splitlines():
            if line.strip():
                names.add(line.split(")")[-1].strip())
    return sorted(names)


def test_no_refused_path_still_lives_in_the_vault():
    """Every path the sync client named as too large is gone from `~/obsidian`."""
    assert SYNC_ERR.exists(), f"{SYNC_ERR} is missing — the check lost its input entirely"
    refused = _refused_paths()
    still_there = [rel for rel in refused if (VAULT / rel).exists()]
    assert still_there == [], f"refused-and-still-in-the-vault: {still_there}"


# ── clause 3: provenance survives the blob ─────────────────────────────────────────────


def test_pointer_note_names_the_new_home_and_both_cut_ranges():
    """The synced note holds the new path, the source id and the exact cut ranges."""
    assert POINTER.exists(), f"{POINTER} is missing — the blob has no pointer"
    text = POINTER.read_text(encoding="utf-8")
    fm = _frontmatter(POINTER)

    assert fm["source_video_id"] == "nS8PvZv3v0U", fm.get("source_video_id")
    recorded = Path(fm["source_blob_path"])
    assert recorded.is_absolute(), f"an out-of-vault home must be named absolutely: {recorded}"
    assert recorded == OUT_OF_VAULT_BLOB, recorded
    assert recorded.is_file(), f"{recorded} is not on disk — the note points at nothing"
    assert str(recorded) in text, "the path has to be in the prose too, not only front matter"

    for token in ("597.12", "614.56", "107.92", "122.00", "5691071"):
        assert token in text, f"the pointer note lost {token!r}"
    # The two cuts survive in the vault as the derived references, and the note says which.
    for name in ("dave_cullen_001.wav", "dave_cullen_002.wav"):
        assert name in text, f"the pointer note lost the link to {name}"
    assert "yt-dlp" in text and "-f 250" in text, "the re-derive command is gone"
    # `test -f` on the named path is the acceptance's own wording; say it out loud.
    assert subprocess.run(["test", "-f", str(recorded)]).returncode == 0


def test_pointer_notes_sha_and_size_match_the_blob_it_points_at():
    """The claim is checkable: re-hash the file the note points at and compare."""
    fm = _frontmatter(POINTER)
    blob = Path(fm["source_blob_path"])
    assert blob.stat().st_size == int(fm["source_blob_size_bytes"]), "the blob's size drifted"
    digest = hashlib.sha256(blob.read_bytes()).hexdigest()
    assert digest == fm["source_blob_sha256"], (
        f"{blob} is no longer the file the note describes: sha256 {digest}")
    assert not (VAULT / BLOB_REL).exists(), (
        "the over-cap blob is back in the vault — the pointer note's whole premise is that "
        "it is not, because Obsidian Sync would refuse it again")


def test_ceiling_in_the_note_is_the_clients_number_not_the_authors():
    """`(5.43 MB, max 5.00 MB)` is MiB arithmetic on the recorded size, not an invention.

    The note states a *mebibyte* ceiling on the strength of the client's printed pair.
    Recompute it from the byte size the note itself records: on decimal megabytes the
    client would have written 5.69, and `SYNC_MAX_BYTES` would be 5000000, not 5242880.
    """
    fm = _frontmatter(POINTER)
    text = POINTER.read_text(encoding="utf-8")
    assert "File too large to sync (5.43 MB, max 5.00 MB)" in text, "the quoted log line"
    size = int(fm["source_blob_size_bytes"])
    assert f"{size / 1024 / 1024:.2f} MB" == "5.43 MB"
    assert f"{size / 1000 / 1000:.2f} MB" == "5.69 MB", "decimal MB is not what was printed"
    assert int(fm["sync_ceiling_bytes"]) == SYNC_MAX_BYTES == 5242880


def test_the_source_note_no_longer_lists_the_blob_as_an_in_vault_asset():
    """`voice-source-dave-cullen.md` must send a reader to the new home, not to the vault."""
    lines = [ln for ln in SOURCE_NOTE.read_text(encoding="utf-8").splitlines()
             if "source_nS8PvZv3v0U.webm" in ln]
    assert lines, f"{SOURCE_NOTE.name} lost every mention of the source blob"
    for line in lines:
        assert ("vault-external" in line or POINTER_REL in line), (
            f"a line still presents the blob as living in the vault: {line[:160]}")


# ── clause 4: the derived reference set is not collateral damage ───────────────────────


def test_the_derived_reference_set_is_byte_for_byte_unchanged():
    ref_dir = VAULT / REF_DIR
    for name, (size, digest) in DERIVED_SET.items():
        f = ref_dir / name
        assert f.is_file(), f"{f} is missing — the fix removed a file it must not have"
        assert f.stat().st_size == size, f"{f} is {f.stat().st_size} B, expected {size} B"
        assert hashlib.sha256(f.read_bytes()).hexdigest() == digest, f"{f} changed content"


# ── clause 5: the knowledge note states both refusals as mechanisms ────────────────────


def test_knowledge_note_states_both_refusals_with_their_proving_commands():
    assert KNOWLEDGE_NOTE.exists(), f"{KNOWLEDGE_NOTE} is missing"
    text = KNOWLEDGE_NOTE.read_text(encoding="utf-8")
    # refusal 1: the per-file ceiling, proven from the client's own log line
    assert 'grep -o "File too large to sync.*"' in text, "refusal 1 lost its proving command"
    assert "5,242,880" in text and "5 MiB" in text, "the ceiling is not stated in bytes"
    # refusal 2: .git as sync CONFIGURATION, not an inference from a log grep
    assert "ob sync-status --path ~/obsidian" in text, "refusal 2 lost its proving command"
    assert "Excluded folders: .git" in text, "refusal 2 does not quote what sync-status prints"
    assert "fence-blind" in text, (
        "the note no longer says why the old `grep -c '.git/'` inference was wrong "
        "evidence for a true fact")
    # the payload #538/#1031's check asserts on
    assert "-size +5242880c" in text, "the named payload probe is gone"
    assert POINTER_REL in text, "the note does not name where the payload went"


def test_the_move_reached_the_vault_git_index():
    """Sync carries the *committed* tree; the file tools only move files on disk.

    `automod_vault_land` stages exactly the named paths (`git add -A -- <paths>`), and
    that step is what turns a working-tree deletion into a synced deletion and a new note
    into a synced note. Read through git itself, in a subprocess.
    """
    def ls_files(rel: str) -> str:
        run = subprocess.run(["git", "-C", str(VAULT), "ls-files", "--", rel],
                             capture_output=True, text=True, timeout=60)
        assert run.returncode == 0, run.stderr[:300]
        return run.stdout.strip()

    assert ls_files(BLOB_REL) == "", (
        f"{BLOB_REL} is still tracked: the deletion never reached the vault index, so the "
        "blob is still vault history even though it is gone from disk")
    assert ls_files(POINTER_REL) == POINTER_REL, (
        f"{POINTER_REL} is not committed, so Obsidian Sync would never carry the provenance")
    assert ls_files("knowledge/software/obsidian-headless-sync-quota-silent-failure.md"), \
        "the knowledge note is not committed, so the two refusals stay on this box"


# ------------------------------------------------------------------ #1141 clause 5
# "Each of the note's Diagnostic one-liners names its evidence source and the
#  rotation bound that limits it." A log count of 0 is also what a rotated window
#  reads, which is exactly how #539 first inferred a true fact from fence-blind
#  evidence. The bound is read from the supervisor program's own conf, so the note
#  cannot keep a number the conf has moved away from.

def _rotation_bound() -> str:
    conf = SYNC_PROGRAM_CONF.read_text(encoding="utf-8")
    sizes = set(re.findall(r"^std(?:out|err)_logfile_maxbytes\s*=\s*(\S+)", conf, re.M))
    assert len(sizes) == 1, f"stdout and stderr rotate differently: {sizes}"
    backups = set(re.findall(r"^std(?:out|err)_logfile_backups\s*=\s*(\d+)", conf, re.M))
    assert len(backups) <= 1, backups
    # supervisord's documented default when the program sets none.
    return f"{sizes.pop()} per file, {backups.pop() if backups else '10'} backups"


def _one_liners() -> list[tuple[list[str], str]]:
    text = KNOWLEDGE_NOTE.read_text(encoding="utf-8")
    section = text[text.index("## Diagnostic one-liners"):]
    block = section[section.index("```bash") + len("```bash"):]
    block = block[:block.index("```")]
    pairs, comments = [], []
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments.append(line)
        else:
            pairs.append((comments, line))
            comments = []
    return pairs


def test_every_diagnostic_one_liner_names_its_evidence_and_rotation_bound():
    bound = _rotation_bound()
    pairs = _one_liners()
    assert len(pairs) >= 8, pairs
    for comments, command in pairs:
        evidence = [c for c in comments if c.startswith("# evidence:")]
        assert len(evidence) == 1, f"{command!r} names no evidence source: {comments}"
        source, sep, rotation = evidence[0].partition(" · rotation: ")
        assert sep and source.strip() != "# evidence:", evidence[0]
        logs = [name for name in ("agent-obsidian-sync.log", "agent-obsidian-sync.err")
                if name in command]
        if logs:
            for name in logs:
                assert name in source, f"{command!r} reads {name} but cites {source!r}"
            assert rotation.startswith(bound), (
                f"{command!r} reads a rotated log; its bound must be the conf's "
                f"{bound!r}, got {rotation!r}")
        else:
            assert rotation.startswith("none"), (command, rotation)


def test_the_non_rotating_last_sync_record_is_one_of_the_one_liners():
    commands = [c for _, c in _one_liners()]
    assert any("--component vault_sync" in c and "last_observed_sync" in c for c in commands)
    assert not any("last_observed_sync" in c and ".log" in c for c in commands)
