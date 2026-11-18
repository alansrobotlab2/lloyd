#!/usr/bin/env python3
"""The one writer of the ``data_pipeline`` watermark block: validate, rename, read back.

Why this module exists (#4245). On 2026-11-17 at 23:09Z, ``_pipeline/autonomy-watermarks.json``
under the data root was found holding 67 bytes whose entire content was the JSON-encoded
absolute path of itself — ``"/home/alansrobotlab/lloyd-data/_pipeline/autonomy-watermarks.json"`` —
one second after the data-pipeline run's own write and one second before the read-back that
caught it. The individual writer was never named: no change-ledger pre-image exists for
``sha1(realpath)`` ``3909b7248a050238984981417b8f11d4f2fb4a19``, the vault audit log records
vault paths only, and none of 172,280 small files under the data root (nor 483 under
``backlog/data``) contains its own path, which rules out the tombstone and lock-helper
candidates outright — ``app/atomic_io.py`` ``_flock_exclusive`` opens a lock ``O_RDWR`` and
never writes a byte.

What IS reproducible is the damage shape, and it is worse than "the next run rewrites it".
Feeding that 67-byte payload to ``scripts/autonomy/self_improve.py`` (the module the item
called the only writer; it is DEPRECATED, task #29 archived 2026-09-03): ``load_watermarks``
(:1146-1156) ``json.load``s successfully and its ``except Exception: return {}`` catches only
syntax errors, so it returns the bare ``str``; ``check_watermark`` then raises
``AttributeError: 'str' object has no attribute 'get'`` (:1185) and ``update_watermark`` raises
``TypeError: 'str' object does not support item assignment`` (:1164, outside its ``try`` at
:1173). A clobbered watermark therefore cannot be repaired by the lane that reads it, and the
run that would repair it dies naming the reader, not the file.

So the defence is the write path, not the reader:

* **One payload shape is committable.** A dict whose ``data_pipeline`` block holds exactly
  the four canonical keys (:data:`CANONICAL_KEYS`) and nothing else. Anything else is refused
  with exit 2 and the existing bytes are left exactly as they were.
* **A non-dict file on disk is reported, not quietly overwritten.** Its path and the byte
  count found are printed before it is replaced by a valid dict, so a clobber leaves a record
  instead of leaving only a symptom.
* **The commit is a rename.** ``app.atomic_io.write_text_durable`` writes a temp and
  ``os.replace``s it, so a concurrent reader never observes a truncated file, and the temp is
  gone when the call returns.
* **The result is read back and re-validated** (exit 3 if it is not what was committed). That
  is the step task #24 ran by hand and that caught the clobber; here it is not skippable, so
  a watermark that is eaten in the window after the rename is reported as a failure rather
  than printed as a clean run.

No lock is taken. ``app.atomic_io``'s locks order *cooperative* writers, and whatever wrote
the 67 bytes took no lock at all, so a lock would not have refused it; what refuses it here is
the payload check plus the read-back.

Exit codes: 0 committed and verified, 2 refused (bytes untouched), 3 committed but the
read-back does not match the contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.atomic_io import write_text_durable  # noqa: E402
from app.paths import PIPELINE_DIR  # noqa: E402

#: The file's name inside ``app.paths.PIPELINE_DIR``. One spelling, so no lane resolves
#: this store by basename — the two ``autonomy-watermarks.json`` files on this box (the live
#: one and the frozen vault copy under ``backlog/data``) are why that matters.
WATERMARK_NAME = "autonomy-watermarks.json"
#: The block this writer owns. Other top-level keys are preserved untouched.
BLOCK_KEY = "data_pipeline"
#: The canonical shape, spelled in the skill's Post-Flight table: four keys and no others.
CANONICAL_KEYS = ("last_run", "status", "last_result", "last_log")
#: The only two states a data_pipeline run may record. A noop has no counts of its own, so
#: it says so instead of implying it with an earlier run's numbers.
KNOWN_STATUSES = ("completed", "noop")

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_VERIFY_FAILED = 3


def watermark_path() -> Path:
    """The live watermark, resolved through ``app.paths`` like every other store.

    Hanging off ``PIPELINE_DIR`` rather than a spelled-out ``~/lloyd-data`` string is what
    makes a scratch ``LLOYD_DATA`` root redirect it, which is the only reason this module is
    testable against a real path instead of a mock.
    """
    return Path(PIPELINE_DIR) / WATERMARK_NAME


def describe_non_dict(parsed: object) -> str:
    """Why this parsed value is not a watermark: used in both the clobber and refuse lines."""
    return f"parsed to {type(parsed).__name__}, not a dict"


def read_existing(path: Path) -> tuple[dict, int, str | None]:
    """Return ``(base_dict, bytes_found, clobber_note)`` for ``path``.

    A file that is absent, unreadable, unparseable or parsed to a non-dict yields ``{}`` plus
    a note naming what was found — the writer keeps going in every one of those cases,
    because the point is to leave a valid dict behind, but it never does so silently.
    """
    if not path.exists():
        return {}, 0, None
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return {}, 0, f"unreadable ({exc.strerror or type(exc).__name__})"
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {}, len(raw), f"not JSON ({type(exc).__name__})"
    if not isinstance(parsed, dict):
        return {}, len(raw), describe_non_dict(parsed)
    return parsed, len(raw), None


def build_block(base: dict, status: str, last_result: str | None,
                last_log: str | None) -> dict:
    """The new ``data_pipeline`` block, assembled to exactly the canonical four keys.

    ``last_run`` is always this instant, in UTC with an explicit offset: a naive stamp is
    local time wearing UTC's clothes and every later reader parses it as UTC (this box runs
    at -0700, which is how the 2026-10-02 run came out seven hours wrong against its own
    file's mtime). ``last_result`` and ``last_log`` come from the caller when it has counts
    and from the block already on disk when it does not — a noop must not touch the counts, so
    it must also not lose them. Anything else that was in the block is dropped, which is how a
    key an earlier run minted by hand dies on the next run instead of piling up. A carried-
    forward key the file does not have stays ``None``, and validation refuses the payload
    rather than inventing a number.
    """
    previous = base.get(BLOCK_KEY)
    previous = previous if isinstance(previous, dict) else {}
    block = {
        "last_run": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "last_result": last_result if last_result is not None else previous.get("last_result"),
        "last_log": last_log if last_log is not None else previous.get("last_log"),
    }
    return block


def validate(payload: object) -> str | None:
    """``None`` if this payload may be committed, else the reason it may not."""
    if not isinstance(payload, dict):
        return f"payload {describe_non_dict(payload)}"
    block = payload.get(BLOCK_KEY)
    if not isinstance(block, dict):
        return f"{BLOCK_KEY} {describe_non_dict(block)}"
    missing = [key for key in CANONICAL_KEYS if key not in block]
    extra = sorted(set(block) - set(CANONICAL_KEYS))
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing {', '.join(missing)}")
        if extra:
            parts.append(f"unexpected {', '.join(extra)}")
        return (f"{BLOCK_KEY} does not hold exactly {', '.join(CANONICAL_KEYS)} "
                f"({'; '.join(parts)})")
    for key in CANONICAL_KEYS:
        value = block[key]
        if not isinstance(value, str) or not value.strip():
            return f"{BLOCK_KEY}[{key}] is not a non-empty string (got {value!r})"
    return None


def commit(path: Path, payload: dict) -> int:
    """Validate, atomically replace, then read back. Returns the process exit code."""
    reason = validate(payload)
    if reason is not None:
        found = path.stat().st_size if path.exists() else 0
        print(f"REFUSED {path}: {reason} — existing {found} bytes left untouched",
              file=sys.stderr)
        return EXIT_REFUSED

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    write_text_durable(path, text, fsync=True)

    # The read-back task #24 ran by hand, made unskippable: this catches the write that is
    # eaten after the rename, which is exactly the shape of the 2026-11-17 clobber.
    written = path.read_bytes()
    try:
        parsed = json.loads(written.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"VERIFY FAILED {path}: the {len(written)} bytes on disk after the replace are "
              f"not JSON ({type(exc).__name__}) — do not report this run clean",
              file=sys.stderr)
        return EXIT_VERIFY_FAILED
    verify_reason = validate(parsed)
    if verify_reason is not None:
        print(f"VERIFY FAILED {path}: the {len(written)} bytes on disk after the replace "
              f"{verify_reason} — do not report this run clean", file=sys.stderr)
        return EXIT_VERIFY_FAILED

    block = parsed[BLOCK_KEY]
    print(f"WATERMARK {path}: {BLOCK_KEY} read back with {len(CANONICAL_KEYS)} keys: "
          + " ".join(f"{key}={block[key]}" for key in CANONICAL_KEYS))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data_pipeline_watermark.py",
        description="Validate and atomically replace the data_pipeline watermark block.",
    )
    parser.add_argument("--path", default=None,
                        help="Watermark file. Defaults to PIPELINE_DIR/"
                             f"{WATERMARK_NAME} resolved through app.paths.")
    parser.add_argument("--status", choices=KNOWN_STATUSES, default=None,
                        help="completed carries this run's counts; noop carries none.")
    parser.add_argument("--last-result", default=None,
                        help="This run's PIPELINE_RESULT counts with the leading token "
                             "stripped, derived from its own log — never typed by hand.")
    parser.add_argument("--last-log", default=None,
                        help="The run-log path those counts came from.")
    parser.add_argument("--payload-file", default=None, metavar="PATH|-",
                        help="Commit this JSON document verbatim as the whole top-level "
                             "payload instead of building one, so a suspect artifact can be "
                             "replayed against the writer. Use '-' for stdin.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.path) if args.path else watermark_path()

    if args.payload_file is not None:
        if args.status is not None:
            print(f"REFUSED {path}: --payload-file commits a payload verbatim, so --status "
                  f"{args.status} would be ignored — give one or the other",
                  file=sys.stderr)
            return EXIT_REFUSED
        raw = (sys.stdin.read() if args.payload_file == "-"
               else Path(args.payload_file).read_text())
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"REFUSED {path}: payload file {args.payload_file} is not JSON "
                  f"({exc.msg} at offset {exc.pos}) — existing bytes left untouched",
                  file=sys.stderr)
            return EXIT_REFUSED
        return commit(path, payload)

    if args.status is None:
        print(f"REFUSED {path}: give --status {'|'.join(KNOWN_STATUSES)} (or --payload-file) "
              f"— existing bytes left untouched", file=sys.stderr)
        return EXIT_REFUSED
    if args.status == "noop" and (args.last_result is not None or args.last_log is not None):
        print(f"REFUSED {path}: status noop writes no counts, so --last-result/--last-log "
              f"must not be given — existing bytes left untouched", file=sys.stderr)
        return EXIT_REFUSED

    base, found, clobber = read_existing(path)
    if clobber is not None:
        print(f"CLOBBER {path}: found {found} bytes — {clobber}; the block is rebuilt from "
              f"nothing, so no other top-level key survived it")
    payload = dict(base)
    payload[BLOCK_KEY] = build_block(base, args.status, args.last_result, args.last_log)
    return commit(path, payload)


if __name__ == "__main__":
    sys.exit(main())
