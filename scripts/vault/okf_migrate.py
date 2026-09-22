#!/usr/bin/env python3
"""okf_migrate.py — bring ~/obsidian authored markdown into OKF v0.1 conformance.

OKF (Google Open Knowledge Format) v0.1 requires every non-reserved concept `.md`
to have parseable YAML frontmatter with a non-empty `type`, and recommends
`title` / `description` / `resource` / `tags` / `timestamp`.

This migrator (per the approved plan cheerful-purring-narwhal):

  Phase 1 (--repair-only): fix frontmatter that breaks STRICT parsers — the glued
    closing fence (`type: notes---`) and the orphaned-tags / duplicate-key / stacked
    -block corruption families. Re-emit ONE canonical `---\\n...\\n---\\n` block.
    No field values are added or renamed.

    The stacked-block half (#960) is a SECOND frontmatter block stranded in the
    BODY: its keys merge up into the leading block and the block is excised. The
    FIRST block wins any repeated key — the rule `repair_skill_frontmatter.py:111`
    has always applied to `skills/`, and the reason a July stamp's `type: notes`
    never overwrites the `type: book-note` it buried (205 of those contradictions
    are still on disk). Detection is `scripts/vault/okf_stranded.py`, shared with
    `validate_okf.py`, so gate and repairer cannot disagree about the family.
    Until #960 this promise was false: `split_leniently()` returned `parts[1],
    parts[2]` and only `parts[1]` was parsed, so repair-only reported
    `repaired: 0` over 2,574 files. One refusal: a body block whose own text will
    not parse — #961's flow-glue family — is reported `stranded/undecodable` and
    left byte-identical, because merging GUESSED values into the strict-parseable
    leading block is not the same trade as repairing a block already broken.

  Phase 2 (default, full): everything in repair, PLUS the field migration —
    * backfill a non-empty `type` (directory+content heuristic, catch-all `note`)
    * rename `summary` -> `description` (code-safe: nothing reads a `summary:` key)
    * add `timestamp` (ISO 8601) derived from the best existing date field, or the
      file mtime as a last resort
  Load-bearing keys are preserved verbatim: `updated`, `last_updated`, `facts`,
  `relations`, `event_date`, `valid_at`, `created_at`, `segment`, `source_type`.

Minimal-churn: a file is rewritten ONLY when it needs a fence repair or an actual
field change. Already-conformant files are left byte-for-byte untouched. Idempotent:
a second run is a no-op.

Parsing mirrors the lenient scheduler split (`content.split("---\\n", 2)`), which
already recovers the glued fence, then the shared graduated-recovery parser
(`agent_mcp._shared.parse_frontmatter_text`) for the corruption families. Output is
validated with `yaml.safe_load` AND the strict `^---\\n(.*?)\\n---\\n` regex (the
form `relations_index.py` requires) before any write.

Usage:
    okf_migrate.py [--repair-only] [--dir NAME] [--apply] [--limit N] [--report PATH]
                   [--root PATH]

    (default = DRY RUN over the whole vault; prints a summary + proposed `type`
     assignments. Add --apply to write. --dir scopes to one top-level dir for
     reviewable per-directory batches.)
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent_mcp._shared import _fold_orphaned_tag_items  # noqa: E402
from app.paths import VAULT_ROOT  # noqa: E402
# The same detector `validate_okf.py` runs, so the gate and this repairer can
# never disagree about which files have a second block (#960).
from scripts.vault.okf_stranded import find_stranded_frontmatter  # noqa: E402

# Dirs/files that are utility-only or OKF-reserved — never treated as concept docs.
# `index.md` / `log.md` are reserved at any depth (§3.1) and §8 forbids frontmatter
# in them, so backfilling a `type` into one is a spec violation this tool would
# write: a dry run over a tree holding only a conformant index and log proposed
# `created index.md [type=note]` / `created log.md [type=note]` before #450. Kept
# identical to EXCLUDE_FILES in validate_okf.py — the two scripts must agree.
EXCLUDE_DIRS = {"templates", "images", ".git", ".obsidian", ".trash"}
EXCLUDE_FILES = {"tags.md", "index.md", "log.md"}
STRICT_FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

# Date fields consulted (highest priority first) to derive OKF `timestamp`.
DATE_PRIORITY = [
    "updated", "last_updated", "updated_at", "last_verified", "last_synthesized",
    "date", "created_at", "created", "researched_at", "published", "captured",
    "generated_at", "generated", "document_date",
]

# Known frontmatter keys — used to disambiguate flow-list-glue corruption
# (`tags: [ai,youtubesource: …` -> the glued key is a known key `source`).
KNOWN_KEYS = {
    "type", "id", "board", "status", "priority", "tags", "blocked", "assigned",
    "created", "updated", "completed", "date", "source", "sources", "video_id",
    "summary", "description", "segment", "title", "domain", "source_type",
    "channel", "published", "category", "last_verified", "last_synthesized",
    "researched_at", "version", "author", "relations", "related", "folder",
    "generated", "generated_at", "entity", "confidence", "last_updated",
    "document_date", "name", "license", "metadata", "timestamp", "resource",
}
_KNOWN_ALT = "|".join(
    re.escape(k) for k in sorted(KNOWN_KEYS, key=len, reverse=True)
)
# A line that opens a flow list which is NOT closed on the same line, with a
# glued known key: `... [a,b,cKEY: value`  ->  split before KEY.
_FLOW_GLUE_RE = re.compile(
    r"^(?P<pre>.*\[[^\]\n]*?)(?P<key>" + _KNOWN_ALT + r"):(?P<rest>\s.*|)$"
)


# ── frontmatter extraction / emission ────────────────────────────────────────

def split_leniently(content: str):
    """Recover (fm_text, body) using the scheduler's lenient split, which resolves
    the glued closing fence. Returns (None, content) if there is no leading block."""
    if not content.startswith("---"):
        return None, content
    parts = content.split("---\n", 2)
    if len(parts) < 3:
        return None, content
    return parts[1], parts[2]


def _split_flow_glue(text: str) -> str:
    """Close unterminated flow lists that swallowed a following known key.

    `tags: [ai,youtubesource: https://…`  ->  `tags: [ai,youtube]\\nsource: https://…`
    Only splits at a KNOWN key boundary, so tag content is never mistaken for a
    key. Non-matching lines pass through untouched.
    """
    out = []
    for line in text.split("\n"):
        if "[" in line and "]" not in line[line.index("["):]:
            m = _FLOW_GLUE_RE.match(line)
            if m:
                pre = m.group("pre").rstrip(", ")
                line = f"{pre}]\n{m.group('key')}:{m.group('rest')}"
        out.append(line)
    return "\n".join(out)


def _quote_colon_scalars(text: str) -> str:
    """Quote bare single-line scalar values that contain YAML-breaking `: ` /
    ` #` (e.g. `summary: Overview of frameworks: GStack`). Lossless. Values that
    are already quoted / structured / multi-line are left alone (the embedded-
    quote multi-line family stays quarantined rather than being lossy-guessed)."""
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        mo = re.match(r"^(\s*)([\w-]+):[ \t]+(\S.*?)[ \t]*$", line)
        if mo:
            indent, key, val = mo.groups()
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            is_continuation = bool(re.match(r"^\s+\S", nxt)) and \
                not re.match(r"^\s*[\w-]+:(\s|$)", nxt) and \
                not re.match(r"^\s*-\s", nxt)
            if val[0] not in "\"'[{|>&*!#%@`" and not is_continuation and (
                ": " in val or " #" in val or val.endswith(":")
            ):
                esc = val.replace("\\", "\\\\").replace('"', '\\"')
                line = f'{indent}{key}: "{esc}"'
        out.append(line)
    return "\n".join(out)


def _strip_stray_fence(text: str) -> str:
    """De-glue a stray `---` fused onto a key line inside the frontmatter, e.g.
    `---type: backlog` -> `type: backlog` (an artifact of double-fence mangling)."""
    out = []
    for line in text.split("\n"):
        mo = re.match(r"^---(?=[\w-]+:(\s|$))(.*)$", line)
        if mo:
            line = mo.group(2)
        out.append(line)
    return "\n".join(out)


def _close_bare_flow(text: str) -> str:
    """Close an unterminated flow list whose content ends on its own line, when
    the next line is not a block-list continuation: `tags: [backlog` -> `[backlog]`."""
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        if "[" in line and "]" not in line[line.index("["):]:
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if not re.match(r"^\s*-\s", nxt):  # not a block-item continuation
                line = line.rstrip() + "]"
        out.append(line)
    return "\n".join(out)


def repair_fm_text(fm_text: str) -> str:
    """Deterministic, non-lossy repairs (order matters). Returns candidate text;
    the caller re-parses and only ACCEPTS it if it yields a valid mapping — so an
    over-aggressive repair can only downgrade a file to quarantine, never corrupt
    it on disk."""
    t = _strip_stray_fence(fm_text)    # de-glue `---key:` -> `key:`
    t = _split_flow_glue(t)            # split `[a,bKEY:` at known-key boundary
    t = _close_bare_flow(t)            # close `key: [a,b` at line end
    t = _fold_orphaned_tag_items(t)    # fold orphan `- item` lines into the list
    t = _quote_colon_scalars(t)        # quote bare scalars containing ': '
    return t


def load_fm(fm_text: str):
    """Parse frontmatter. Returns (dict|None, repaired: bool).

    dict is None when the text cannot be safely recovered to a mapping — the
    caller QUARANTINES those (never lossy-guesses). `repaired` is True when the
    deterministic repair layer was needed to make it parse.
    """
    try:
        fm = yaml.safe_load(fm_text)
        if isinstance(fm, dict):
            return fm, False
    except yaml.YAMLError:
        pass
    try:
        fm = yaml.safe_load(repair_fm_text(fm_text))
        if isinstance(fm, dict):
            return fm, True
    except yaml.YAMLError:
        pass
    return None, False


def emit(fm: dict, body: str) -> str:
    dumped = yaml.safe_dump(
        fm, sort_keys=False, allow_unicode=True, default_flow_style=False
    ).rstrip()
    return "---\n" + dumped + "\n---\n" + body.lstrip("\n")


# ── the second frontmatter block stranded in a body (#960) ───────────────────

def collapse_stranded_frontmatter(content: str, fm: dict) -> str | None:
    """Merge every body-block stranded in `content` up into the parsed `fm`.

    Returns the rewritten document (leading block plus stranded keys, body with
    the stranded blocks excised), or None when there is nothing to merge OR the
    merge is not safe. Detection lives in `scripts/vault/okf_stranded.py`, shared
    with `validate_okf.py`; the merge rule is the one
    `scripts/repair_skill_frontmatter.py:111-124` has applied to `skills/` since
    it was written: the FIRST block wins any repeated key, so the stranded copy
    contributes only the keys the leading block lacks. A July stamp's
    `type: notes` therefore never overwrites the `type: book-note` it buried —
    which is the whole reason #478 counts 205 contradictions.

    None (change nothing) is the answer whenever a stranded block's own text will
    not parse as a mapping. That is #961's flow-glue family (`tags: [a,b,cdate:
    2025-07-09`), and this tool already refuses to lossy-guess a corrupt block
    elsewhere — it QUARANTINES it. Inventing values here would write them into the
    one place every downstream reader trusts, so the file keeps its
    `stranded/undecodable` verdict until #478's data pass resolves the glue.
    """
    found = find_stranded_frontmatter(content)
    if not found:
        return None
    merged = dict(fm)
    for block in found:
        try:
            values = yaml.safe_load(block.inner)
        except yaml.YAMLError:
            return None
        if not isinstance(values, dict) or not values:
            return None
        for key, val in values.items():
            merged.setdefault(key, val)
    # Slice the body directly rather than re-splitting: `split_leniently` counts
    # `---\\n` occurrences, and excising a block changes how many there are, so a
    # re-split could resolve the fences differently than the ones that produced
    # `fm`. Offsets are into `content`, and every stranded block lies strictly
    # after the leading block, so translating by the body's own start is exact.
    fm_text, body = split_leniently(content)
    base = len(content) - len(body)
    for block in reversed(found):
        body = body[:block.start - base] + body[block.end - base:]
    return emit(merged, body)


# ── type heuristic ───────────────────────────────────────────────────────────

def infer_type(path: Path, fm: dict, body: str,
               root: Path | None = None) -> str:
    # `None` resolves to this module's VAULT_ROOT at CALL time — a captured
    # default would freeze the live vault and break the test that points the
    # global at a fixture. The tree matters: the heuristic below keys on the
    # top-level directory, so a --root fixture must be resolved against itself.
    rel = path.relative_to(VAULT_ROOT if root is None else root)
    top = rel.parts[0] if len(rel.parts) > 1 else ""
    name = path.name
    stem = path.stem

    if top == "autonomy":
        return "autonomy"
    if name == "SKILL.md" or top == "skills":
        return "skill"
    if top == "people":
        return "person"
    if top == "knowledge":
        # Canonical values only — the 12 in scripts/vault/okf_taxonomy.py (#370).
        # This branch used to end in the catch-all `note` and emit
        # `source-summary` for source pages: two spellings the vault had just
        # been renamed away from (vault 6eed1445), so every --apply minted fresh
        # fragmentation and `validate_okf --strict` counted it as an unknown
        # type. The two spellings live on in TYPE_ALIASES for READING old files;
        # nothing writes them. Outside knowledge/ the catch-all is unchanged —
        # the 12-value taxonomy is knowledge/'s, and #442 owns retiring `note`.
        if stem == path.parent.name:
            return "infrastructure"      # dir hub: <dir>/<dir>.md
        if "sources" in rel.parts:
            return "reference"           # a source summary is reference material
        if fm.get("video_id") or fm.get("channel"):
            return "video-note"
        if fm.get("research-depth") or "research" in stem:
            return "research"
        return "notes"
    if top == "memory":
        return "note"
    # dir "hub" file: <dir>/<dir>.md
    if stem == path.parent.name:
        return "hub"
    return "note"


# ── timestamp derivation ─────────────────────────────────────────────────────

def _to_iso(val) -> str | None:
    if isinstance(val, dt.datetime):
        return val.isoformat()
    if isinstance(val, dt.date):
        return val.isoformat()
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d"):
            try:
                return dt.datetime.strptime(s.replace("Z", "+0000"), fmt).isoformat()
            except ValueError:
                continue
        return s  # keep whatever it is; better than dropping provenance
    return None


def derive_timestamp(fm: dict, path: Path) -> str:
    for key in DATE_PRIORITY:
        if key in fm and fm[key] not in (None, ""):
            iso = _to_iso(fm[key])
            if iso:
                return iso
    return dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


# ── per-file processing ──────────────────────────────────────────────────────

def looks_like_fm(fm_text: str) -> bool:
    """True if the block's first non-blank line looks like a YAML key — i.e. it
    is genuinely (corrupt) frontmatter, not stray markdown / an empty block."""
    for line in fm_text.split("\n"):
        if line.strip():
            return bool(re.match(r"^[\w-]+:", line))
    return False


def process(path: Path, *, repair_only: bool, root: Path | None = None):
    """Return (action, new_text|None, inferred_type|None).

    `root` is the tree `path` sits in — pass it when scanning anything other than
    the live vault, or the type heuristic resolves the file against the wrong
    tree (see `infer_type`).

    action ∈ {ok, repaired, migrated, created, quarantine, stranded/undecodable}
      ok         — already conformant, no write
      repaired   — fence/parse fix only, no field changes
      migrated   — field changes (type/description/timestamp)
      created    — had no real frontmatter block; one was added (full mode only)
      quarantine — genuinely-corrupt frontmatter we won't lossy-guess; untouched
      stranded/undecodable — a second frontmatter block sits in the body and its
                   values will not parse, so nothing was merged and nothing was
                   written (see `collapse_stranded_frontmatter`)
    """
    disk = content = path.read_text(encoding="utf-8")
    fm_text, body = split_leniently(content)

    repaired_parse = False
    if fm_text is None:
        # No leading '---' block at all -> the whole file is body.
        fm: dict = {}
        body, had_fm, strict_ok = content, False, False
    else:
        parsed, repaired_parse = load_fm(fm_text)
        if parsed is None:
            if looks_like_fm(fm_text):
                return "quarantine", None, None   # real but unrecoverable frontmatter
            # A spurious/empty leading block (markdown, not frontmatter) -> treat
            # the whole file as body and prepend fresh frontmatter. Drop the one
            # stray leading `---` fence so we don't emit a double fence.
            body = re.sub(r"^---\n", "", content, count=1)
            fm, had_fm, strict_ok = {}, False, False
        else:
            fm, had_fm = parsed, True
            strict_ok = bool(STRICT_FM_RE.match(content)) and not repaired_parse

    # ── a second frontmatter block stranded in the body (#960) ───────────────
    # Before this, `split_leniently` handed every stage only parts[1] and the
    # body where the stacked block lives was read by nothing — so --repair-only
    # reported `repaired: 0` over a tree with 334 stranded files. Collapsing
    # happens BEFORE the field logic so a `type:` the stranded block carries is
    # visible to the checks below, and the merge keeps the leading block's value
    # on any repeated key.
    stranded_collapsed = False
    if had_fm:
        collapsed = collapse_stranded_frontmatter(content, fm)
        if collapsed is None and find_stranded_frontmatter(content):
            return "stranded/undecodable", None, None
        if collapsed is not None:
            content = collapsed
            fm_text, body = split_leniently(content)
            fm, repaired_parse = load_fm(fm_text)
            had_fm = isinstance(fm, dict)
            strict_ok = bool(STRICT_FM_RE.match(content)) and not repaired_parse
            stranded_collapsed = True

    if repair_only and not had_fm:
        return "ok", None, None  # nothing to repair; Phase 2 will add frontmatter

    changed_fields = False
    inferred = None

    if not repair_only:
        # type
        if not str(fm.get("type", "") or "").strip():
            inferred = infer_type(path, fm, body, root)
            fm["type"] = inferred
            changed_fields = True
        # summary -> description
        if "summary" in fm:
            summ = fm.pop("summary")
            if not str(fm.get("description", "") or "").strip() and \
                    str(summ or "").strip():
                fm["description"] = summ
            changed_fields = True
        # timestamp (additive)
        if not str(fm.get("timestamp", "") or "").strip():
            fm["timestamp"] = derive_timestamp(fm, path)
            changed_fields = True

    needs_fence_fix = had_fm and (repaired_parse or not strict_ok)

    if not had_fm and not repair_only:
        return "created", emit(fm, body), inferred
    if not (changed_fields or needs_fence_fix or stranded_collapsed):
        return "ok", None, inferred
    new_text = emit(fm, body)
    # Safety: verify the emitted block is strict-parseable and round-trips.
    m = STRICT_FM_RE.match(new_text)
    if not m:
        return "quarantine", None, inferred
    try:
        yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return "quarantine", None, inferred
    if new_text == disk:
        # A collapse that emit() normalised back to the bytes already on disk is
        # already in the fixed state — reporting it as `repaired` would promise a
        # write that does not happen, and the next run would report it again.
        return "ok", None, inferred
    action = "migrated" if changed_fields else "repaired"
    return action, new_text, inferred


def iter_md(root: Path, only_dir: str | None):
    base = root / only_dir if only_dir else root
    for p in sorted(base.rglob("*.md")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if p.name in EXCLUDE_FILES or p.name.startswith("_"):
            continue
        yield p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repair-only", action="store_true",
                    help="Phase 1: fix fences/parse only, no field changes")
    ap.add_argument("--dir", default=None,
                    help="scope to one top-level vault dir (e.g. knowledge)")
    ap.add_argument("--apply", action="store_true", help="write changes (default dry run)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N files")
    ap.add_argument("--report", default=None, help="write a text report to this path")
    ap.add_argument("--quarantine-report", default=None,
                    help="write the full quarantined-file list to this path")
    ap.add_argument("--root", default=None,
                    help="tree to migrate instead of the live vault root "
                         "(app.paths.VAULT_ROOT) — for a fixture, or a copy under "
                         "review before anything is written to ~/obsidian")
    args = ap.parse_args()
    root = Path(args.root) if args.root else VAULT_ROOT

    counts = Counter()
    type_assign = Counter()
    quarantined: list[str] = []
    stranded_undecodable: list[str] = []
    quarantine_by_dir = Counter()
    samples: list[str] = []
    n = 0

    for path in iter_md(root, args.dir):
        if args.limit and n >= args.limit:
            break
        n += 1
        rel = path.relative_to(root)
        try:
            action, new_text, inferred = process(path, repair_only=args.repair_only,
                                                root=root)
        except Exception as e:  # noqa: BLE001
            counts["error"] += 1
            quarantined.append(f"{rel}  (exception: {e})")
            quarantine_by_dir[rel.parts[0] if len(rel.parts) > 1 else "<root>"] += 1
            continue
        counts[action] += 1
        if inferred:
            type_assign[inferred] += 1
        if action == "quarantine":
            quarantined.append(str(rel))
            quarantine_by_dir[rel.parts[0] if len(rel.parts) > 1 else "<root>"] += 1
        if action == "stranded/undecodable":
            stranded_undecodable.append(str(rel))
        if action in ("repaired", "migrated", "created"):
            if len(samples) < 25:
                samples.append(f"  {action:8} {rel}"
                               + (f"  [type={inferred}]" if inferred else ""))
            if args.apply and new_text is not None:
                path.write_text(new_text, encoding="utf-8")

    mode = "APPLY" if args.apply else "DRY RUN"
    phase = "REPAIR-ONLY" if args.repair_only else "FULL"
    lines = [
        f"[okf_migrate] {mode} / {phase}"
        + (f" / dir={args.dir}" if args.dir else "") + f"  ({n} files scanned)",
        f"  ok/unchanged : {counts['ok']}",
        f"  repaired     : {counts['repaired']}",
        f"  migrated     : {counts['migrated']}",
        f"  created fm   : {counts['created']}",
        f"  QUARANTINED  : {counts['quarantine']}",
        f"  stranded     : {counts['stranded/undecodable']}",
        f"  errors       : {counts['error']}",
    ]
    if type_assign:
        lines.append("  backfilled type assignments:")
        for t, c in type_assign.most_common():
            lines.append(f"      {t:16} {c}")
    if samples:
        lines.append("  sample changes:")
        lines.extend(samples)
    if stranded_undecodable:
        lines.append("  stranded in body but its values will not parse "
                     "(nothing merged, nothing written):")
        lines.extend(f"      {p}" for p in stranded_undecodable[:25])
        if len(stranded_undecodable) > 25:
            lines.append(f"      ... and {len(stranded_undecodable) - 25} more")
    if quarantine_by_dir:
        lines.append("  quarantined by dir (pre-existing corruption, untouched):")
        for d, c in quarantine_by_dir.most_common():
            lines.append(f"      {d:16} {c}")

    report = "\n".join(lines)
    print(report)
    if args.report:
        Path(args.report).write_text(report + "\n", encoding="utf-8")
    if args.quarantine_report and quarantined:
        Path(args.quarantine_report).write_text(
            "\n".join(sorted(quarantined)) + "\n", encoding="utf-8")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
