#!/usr/bin/env python3
"""
Nightly Deep Extraction - Next-Gen Memory System

Runs at 2 AM PST with the primary model for comprehensive extraction.
"""

import argparse
import fcntl
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Import resolution follows THIS file, never the live checkout (#755). The
# `content_hasher` insert used to be hardcoded to `Path.home() / "lloyd" /
# "scripts" / "memory"`, so a run from a self-mod worktree or any copy of the
# checkout measured a hybrid: a worktree `nightly_extraction` hashing with a
# live-tree `content_hasher`. That is the shape of a paired before/after
# extraction run, and it is silent — both modules import cleanly.
#
# State paths further down (log, pre-clean backups, graph dir, lock) address the
# live data root on purpose (`_STATE_PIPELINE`): a worktree that runs the nightly
# must still write the live `_pipeline`, so only import resolution is
# tree-relative. An explicit `LLOYD_DATA` (the gate, the canary, the suite) is
# the one thing that moves them.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "memory"))  # content_hasher
sys.path.insert(0, str(_REPO_ROOT))                         # app.*
from app.atomic_io import hash_bytes  # noqa: E402
from fact_extractor import CHUNK_BUDGET_CHARS, ExtractionFailed  # noqa: E402

try:
    from content_hasher import ContentHasher
    _HAS_HASHER = True
except ImportError:
    _HAS_HASHER = False

import yaml

sys.path.insert(0, str(_REPO_ROOT))
from app.paths import VAULT_FACTS_ROOT as FACTS_DIR
from app.paths import production_data_root  # noqa: E402
_STATE_PIPELINE = Path(os.environ.get("LLOYD_DATA") or production_data_root()) / "_pipeline"
VAULT = Path.home() / "obsidian"
# No index path belongs here (#1148). This module reaches both indexes through
# `self.rel_generator`: `rebuild()` writes that module's own
# `_pipeline/relations-index-typed.json`, and the derived
# `_pipeline/relations-index.json` is written by scripts/memory/rebuild_index.py
# (the skill's Step 2). A dead `INDEX_FILE` constant naming the derived file sat
# here while two scripts still wrote it, which is exactly the kind of pointer
# that re-arms a clobber.

# Local sibling modules. These resolve off the directory this file runs from:
# as a script that is this directory; loaded by path, the caller injects it
# (`scripts/memory/kg_rebuild.py::_corpus_size` does). The insert of
# `VAULT / "agents" / "memory" / "scripts" / "next-gen-memory"` that used to sit
# here pointed at a directory that does not exist, and while it was inert its
# mere presence put a tree outside the checkout ahead of the caller's entry —
# so which `fact_extractor` these three names bound to depended on which tree
# invoked the script (#755).
from fact_extractor import FactExtractor
from relations_index import RelationsIndexGenerator
from profile_generator import ProfileGenerator


_SELF_WRITTEN_MEMORY_NOTES = frozenset({
    # Derived indexes the pipeline regenerates into `memory/` itself. A name
    # earns its place by having a live writer, not by having had one: two
    # generated reports stayed in this set for nine days after 0b3f00b deleted
    # their generators, so the extractor kept skipping files it had stopped
    # writing, and the relationship counts those dead files asserted in
    # `memory/` were never recomputed by anything (#487). Pinning the rule as
    # behaviour, not just as set membership: tests/test_fact_extractor.py.
    "skills-index.md",
})


CONFIG_PATH = Path(__file__).resolve().parent / "pipeline_config.yaml"


def _load_pipeline_config() -> dict:
    """Read pipeline_config.yaml. It existed since 2026-07 and nothing read it."""
    try:
        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raise RuntimeError(f"pipeline_config.yaml not found at {CONFIG_PATH}")
    except Exception as e:
        raise RuntimeError(f"pipeline_config.yaml is unreadable: {e}") from e
    if not isinstance(data, dict):
        raise RuntimeError("pipeline_config.yaml did not parse to a mapping")
    return data


def _TODAY_STR() -> str:
    return datetime.now().strftime("%Y-%m-%d")


class NightlyExtraction:
    """Nightly deep extraction with primary model."""

    def __init__(self):
        self.extractor = FactExtractor(model_port=8096)  # Uses primary for deep extraction
        self.rel_generator = RelationsIndexGenerator()
        self.profile_generator = ProfileGenerator(model_port=8096)
        self.log_file = _STATE_PIPELINE / "nightly-extraction.log"
        # Thread-safe locks per entity for parallel processing
        self.entity_locks = {}
        self.locks_lock = threading.Lock()
        # What the last extraction pass did, for `run_full_extraction`'s summary.
        # `last_truncated_files` is the vault-relative paths of documents whose
        # tail no pass has read (#1151); it pairs with `last_failed_files` and the
        # two must not be conflated — a truncated document extracted fine, it just
        # is not finished.
        self.last_files_processed = 0
        self.last_failed_files = 0
        self.last_truncated_files: list = []
    
    def _get_entity_lock(self, entity: str) -> threading.Lock:
        """Get or create a lock for an entity."""
        with self.locks_lock:
            if entity not in self.entity_locks:
                self.entity_locks[entity] = threading.Lock()
            return self.entity_locks[entity]
    
    # Files under FACTS_DIR that a clean must NEVER delete. On 2026-08-22 this
    # method wiped the fact tree and took `_relationships.json` (12,131 edges /
    # 7,260 active) and the `memory-graph/` working directory with it. Fact
    # content is derivable from the vault; the EDGE GRAPH, merge history and
    # hand-review state are not — they were built incrementally by the v4
    # classifier and successive resolution sweeps. There was no backup: this
    # tree is gitignored and nothing else copies it. Tasks #48, #67, #69 and
    # #74 have been blocked ever since.
    PROTECTED_NAMES = frozenset({
        "entity-registry.json",
        "_relationships.json",
        "entity-aliases.json",
    })
    PROTECTED_DIRS = frozenset({"templates", "memory-graph"})

    def _is_protected(self, item) -> bool:
        if item.name in self.PROTECTED_NAMES or item.name in self.PROTECTED_DIRS:
            return True
        # Backups and snapshots of the above (…​.bak, .bak.json, .corrupt-*.bak,
        # _relationships.<ts>.bak.json) are the only recovery path there is.
        lowered = item.name.lower()
        return ".bak" in lowered or lowered.startswith("_relationships")

    def backup_graph_state(self):
        """Timestamped copy of the irreplaceable files before a destructive pass."""
        import shutil
        stamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        dest = _STATE_PIPELINE / "backups" / f"pre-clean-{stamp}"
        saved = []
        for name in ("_relationships.json", "entity-aliases.json"):
            src = FACTS_DIR / name
            if src.is_file():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / name)
                saved.append(name)
        graph_dir = _STATE_PIPELINE / "memory-graph"
        if graph_dir.is_dir() and any(graph_dir.iterdir()):
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(graph_dir, dest / "memory-graph", dirs_exist_ok=True)
            saved.append("memory-graph/")
        if saved:
            print(f"  → Backed up {', '.join(saved)} to {dest}")
        return dest if saved else None

    def clean_facts_directory(self):
        """Wipe the entity tree, preserving the graph and its backups."""
        if not FACTS_DIR.exists():
            print("  → Facts directory does not exist, skipping clean")
            return

        self.backup_graph_state()

        removed = kept = 0
        for item in FACTS_DIR.iterdir():
            if self._is_protected(item):
                kept += 1
                continue
            if item.is_dir():
                import shutil
                shutil.rmtree(item)
            else:
                item.unlink()
            removed += 1

        print(f"  → Facts directory cleaned ({removed} removed, "
              f"{kept} protected entries kept)")
    
    def run_full_extraction(self, full_mode=False, workers=1, clean=False, limit=0, force=False):
        """Run complete nightly extraction pipeline."""
        start_time = datetime.now()
        log_lines = [
            f"\n{'='*60}",
            f"Nightly Extraction Started: {start_time.isoformat()}",
            f"Mode: {'Full Vault' if full_mode else '24h Window'}",
            f"Workers: {workers}",
            f"Clean: {'Yes' if clean else 'No'}",
            f"{'='*60}"
        ]
        
        try:
            # Clean facts directory if requested
            if clean:
                log_lines.append("\n[Pre-Flight] Cleaning Facts Directory")
                self.clean_facts_directory()
            
            # Step 1: Full vault fact extraction. Duplicate detection moved to
            # kg_hygiene.near_duplicates, which measures the same thing without
            # a per-run 23k-directory scan whose only output was a log line.
            log_lines.append("\n[Step 1] Full Vault Fact Extraction")
            self.last_files_processed = 0
            self.last_failed_files = 0
            self.last_truncated_files = []
            facts_extracted = self._extract_all_facts(full_mode=full_mode, workers=workers, limit=limit)
            files_processed = getattr(self, "last_files_processed", 0)
            failed_files = getattr(self, "last_failed_files", 0)
            truncated_files = list(getattr(self, "last_truncated_files", []))
            log_lines.append(f"  → Processed {files_processed} files, extracted {facts_extracted} new facts")
            links = self.extractor.link_stats
            log_lines.append(f"  → Linked {links['mentions_linked']} mentions edges; "
                             f"{links['mentions_skipped_typed']} skipped on pairs the "
                             f"classifier had already typed")
            if failed_files:
                log_lines.append(f"  ⚠ {failed_files} file(s) failed extraction and were NOT hashed; "
                                 f"they will be retried next run")
            if truncated_files:
                log_lines.append(f"  ⚠ {len(truncated_files)} file(s) were read only as far as the "
                                 f"{CHUNK_BUDGET_CHARS}-char chunk budget: {truncated_files}. "
                                 f"Each was recorded with its coverage offset and not as "
                                 f"complete, so the next run continues from there")

            # Gate: Steps 2-5 (derives, relation discovery, index rebuild, overview
            # generation) only have new material to chew on when extraction touched
            # at least one file. Running them every poll on an unchanged vault is the
            # expense the outer skill used to guard with a git-commit watermark — now
            # the content-hash result is the authoritative signal. --full/--clean/--force
            # always run the complete pipeline.
            if files_processed == 0 and not full_mode and not clean and not force:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                log_lines.append("  → No new/changed files; skipping relation, index, and overview steps")
                log_lines.append(f"\nNightly Extraction Complete (noop): {end_time.isoformat()} ({duration:.1f}s)")
                self.log_file.write_text("\n".join(log_lines))
                return {
                    "success": True,
                    "noop": True,
                    "files_processed": 0,
                    "facts_extracted": 0,
                    "failed_files": failed_files,
                    # Named even on the noop path: a summary whose shortfall key
                    # appears only when something was truncated is a summary that
                    # reports full coverage by omission.
                    "truncated_files": truncated_files,
                    "total_relationships": None,
                    "overviews_generated": 0,
                    "duration_seconds": duration,
                }

            # Step 2: Index rebuild. `_infer_derives_relationships` and
            # `_discover_relations` used to sit here; both wrote into
            # relations-index.json (document-level co-occurrence), not the
            # entity graph, and `_discover_relations` compared only the first
            # 50 documents of 3,265 — an O(50²) sample presented as vault-wide
            # discovery. Edges now come from the extractor as it writes facts.
            log_lines.append("\n[Step 2] Index Rebuild")
            index = self.rel_generator.rebuild()
            log_lines.append(f"  → Rebuilt index with {index['total_relationships']} relationships")

            # Step 3: Entity overview generation (change-triggered)
            log_lines.append("\n[Step 3] Entity Overview Generation")
            overviews_generated = self._regenerate_entity_overviews()
            log_lines.append(f"  → Generated/updated {overviews_generated} entity overviews")

            # Summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            log_lines.extend([
                f"\n{'='*60}",
                f"Nightly Extraction Complete: {end_time.isoformat()}",
                f"Duration: {duration:.1f} seconds",
                f"{'='*60}\n"
            ])

            # Write log
            self.log_file.write_text("\n".join(log_lines))

            return {
                "success": True,
                "noop": False,
                "files_processed": files_processed,
                "failed_files": failed_files,
                # Paths whose tail no pass has read yet, vault-relative. Without
                # this a run that finished 5% of three feeds reported the same
                # numbers as one that finished three feeds (#1151 clause 4).
                "truncated_files": truncated_files,
                "duration_seconds": duration,
                "facts_extracted": facts_extracted,
                "total_relationships": index['total_relationships'],
                "overviews_generated": overviews_generated,
            }
            
        except Exception as e:
            log_lines.append(f"\nERROR: {str(e)}")
            self.log_file.write_text("\n".join(log_lines))
            raise
    
    def _process_single_file(self, md_file, full_mode, index, total, resume=None):
        """Extract one file. Returns
        `(processed, facts, ok, source_hash, chars_covered, chars_total)`.

        `ok=False` means the extraction failed and the caller must NOT save
        this file's content hash — otherwise a transient vLLM error marks the
        document done and it is never revisited. On a failure `source_hash` is
        the empty string and no chars were read, so both coverage figures are 0.

        `source_hash` is returned as well as used, because the caller is the one
        that records it in the content-hash gate. It has to be the digest of
        THESE bytes, not of the file as it looks when the checkpoint flushes;
        see the comment at its computation.

        `chars_covered` / `chars_total` report how much of the document reached
        the model. One pass sends at most `CHUNK_BUDGET_CHARS` (47,000) chars, so
        a long document is genuinely unfinished work: `ok` stays True — facts were
        extracted — but the caller must not record this digest as done while
        `chars_covered < chars_total` (#1151 clause 3). The tuple deliberately
        says both things at once, which the old 4-tuple could not: it reported
        `ok=True` for a 955,125-char feed whose remaining 95% was then skipped
        forever.

        `resume` is the `(covered_through, stored_hash)` the gate holds for this
        path. The offset is used only while `stored_hash` still equals the digest
        of the bytes just read: an offset taken from a document that has since
        been rewritten points at other people's bytes, and trusting it would both
        skip new content and let the pass claim it had reached the end.
        """
        processed = 0
        facts_count = 0

        try:
            raw = md_file.read_bytes()
        except OSError as e:
            print(f"Cannot read {md_file}: {e}")
            return 0, 0, False, "", 0, 0

        # Hash the bytes we actually extracted from. Re-hashing the file
        # afterwards records whatever it looks like then, so a note appended
        # to mid-run is marked extracted at content nobody read. That is the
        # same hazard on both sides of this line: `source_hash` is the
        # provenance hash on the facts (see `app/atomic_io.py::hash_bytes`) and
        # the gate hash in `_pipeline/content-hashes.json` has to be it too
        # (#482).
        source_hash = hash_bytes(raw)
        try:
            content = raw.decode("utf-8", errors="replace")
            doc_path = str(md_file.relative_to(VAULT))
        except ValueError:
            doc_path = str(md_file)

        # Trust the gate's resume point only for the bytes it was written under.
        resume_from, resume_hash = resume if resume else (0, "")
        if resume_from and resume_hash != source_hash:
            resume_from = 0

        try:
            result = self.extractor.extract_from_document(
                md_file, content, existing_facts="", start_offset=resume_from
            )
        except ExtractionFailed as e:
            print(f"[{index}/{total}] FAILED: {doc_path}: {e}")
            return 0, 0, False, "", 0, 0
        except Exception as e:
            print(f"[{index}/{total}] ERROR: {doc_path}: {e}")
            return 0, 0, False, "", 0, 0

        # A result that names no coverage was produced by something that read the
        # whole document it was handed — the double in
        # `test_a_failed_file_is_not_hashed`, for one. Defaulting to
        # `chars_total` keeps such a caller honest and this one honest too: the
        # alternative default of 0 would make every such file look truncated and
        # it would be re-extracted nightly forever.
        chars_total = len(content)
        chars_covered = int(result.get("chars_covered", chars_total) or 0)

        try:
            if result.get("facts"):
                default_entity = result.get("entity") or "general"
                default_category = result.get("category") or "general"

                # Fan out: file each fact under its OWN entity/category (the
                # extractor tags every fact), so a multi-entity doc populates
                # multiple entity files instead of collapsing onto one primary.
                groups = {}
                for f in result["facts"]:
                    key = (f.get("entity") or default_entity,
                           f.get("category") or default_category)
                    groups.setdefault(key, []).append(f)

                for (entity, category), gfacts in groups.items():
                    self.extractor.write_fact_file(
                        entity, category,
                        {"entity": entity, "category": category, "facts": gfacts},
                        source_doc=doc_path,
                        source_hash=source_hash,
                    )
                    facts_count += len(gfacts)

                processed = 1
                print(f"[{index}/{total}] Processing: {doc_path} "
                      f"→ {len(groups)} entities, {facts_count} facts")
            else:
                # A genuinely factless document IS extracted; hashing it is
                # correct and stops it being re-read every night.
                processed = 1
        except Exception as e:
            print(f"Error writing facts for {md_file}: {e}")
            return 0, 0, False, "", 0, 0

        return processed, facts_count, True, source_hash, chars_covered, chars_total

    def _eligible_files(self, full_mode: bool) -> list:
        """The corpus, from `pipeline_config.yaml` `sources.paths`.

        This used to be `VAULT.rglob("*.md")` minus a list of deny-substrings,
        which is why the config file existed but was never read and why every
        new vault directory was ingested by default. Roughly half the 205k
        facts in the pre-2026-09 tree came from re-extracting the pipeline's
        own output. An allow-list inverts that: a directory is ingested
        because someone named it.

        The explicit excludes stay as belt-and-braces, and daily notes are
        only eligible once they are more than a day old — today's is appended
        to all day, so it never settles.
        """
        cfg = _load_pipeline_config()
        # Resolved like each entry below, or a symlinked `~/obsidian` (the gate's
        # round home) puts every resolved entry "outside the vault".
        vault = VAULT.resolve()
        roots = []
        for raw in (cfg.get("sources", {}).get("paths") or []):
            path = Path(str(raw)).expanduser()
            if not path.is_absolute():
                path = VAULT / path
            try:
                path = path.resolve()
            except OSError:
                continue
            # Never leave the vault, whatever the config says.
            if path == vault or vault in path.parents:
                roots.append(path)
            else:
                print(f"  ⚠ sources.paths entry outside the vault, ignored: {raw}")
        if not roots:
            raise RuntimeError(
                "pipeline_config.yaml lists no usable sources.paths; refusing to "
                "fall back to the whole vault (that fallback is what fed the "
                "extractor its own output)"
            )
        excludes = tuple(cfg.get("sources", {}).get("exclude_patterns") or ())
        today = _TODAY_STR()
        cutoff_ok = lambda p: full_mode or (  # noqa: E731
            (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).total_seconds() < 86400
        )

        eligible: list = []
        seen: set = set()
        for root in roots:
            if not root.exists():
                print(f"  ⚠ sources.paths entry does not exist: {root}")
                continue
            for md_file in root.rglob("*.md"):
                if not md_file.is_file() or md_file in seen:
                    continue
                path_str = str(md_file)
                if any(part in path_str for part in
                       ("node_modules", "/.venv", "/.cache", "/.git/", "__pycache__")):
                    continue
                if any(pat.strip("*/") and pat.strip("*/") in path_str for pat in excludes):
                    continue
                # Belt and braces: these never belong in the corpus even if a
                # sources.paths entry would otherwise reach them.
                if any(skip in path_str for skip in
                       ("/facts/", "/_pipeline/", "/skills/", "/agents/", "/autonomy/",
                        "/memory/vault-maintenance/")):
                    continue
                if md_file.name in _SELF_WRITTEN_MEMORY_NOTES:
                    continue
                # A daily note is appended to all day; it settles tomorrow.
                if md_file.parent.name == "memory" and md_file.stem >= today:
                    continue
                if not cutoff_ok(md_file):
                    continue
                seen.add(md_file)
                eligible.append(md_file)
        return eligible

    def _extract_all_facts(self, full_mode=False, workers=1, limit=0) -> int:
        """Extract facts from all documents."""
        total_facts = 0
        processed = 0
        failed = 0

        eligible_files = self._eligible_files(full_mode)

        print(f"Found {len(eligible_files)} eligible files")

        # Skip files whose recorded extraction is DONE and unchanged. Applied in
        # BOTH window and full mode so --full is a resumable backfill that skips
        # already-done files rather than reprocessing the whole corpus.
        #
        # "Done" is now two things at once (#1151): the digest must still match,
        # AND the pass that recorded it must have reached the end of the document.
        # A digest match alone used to mean "extracted", while one pass over a
        # document reaches at most `CHUNK_BUDGET_CHARS` (47,000) chars — so
        # `knowledge/tools/openclaw/updates.md` (957,489 chars, 4.9% of its chars
        # ever sent to the model) was skipped as complete on a digest written
        # 2026-09-14. An unfinished document goes back into this run carrying the
        # offset where the last pass stopped, so its tail is read next rather than
        # never. The two halves of that cannot be swapped: a withheld digest would
        # restart the same 47,000 chars every night and coverage would never
        # advance, which is the trap the item's own acceptance wording names.
        hasher = ContentHasher() if _HAS_HASHER else None
        resume_map: dict = {}
        if hasher is not None:
            # Judge the recorded entries BEFORE the change filter, over the WHOLE
            # allow-list, and without writing to the index.
            #
            # Whole allow-list, because a record that cannot be trusted is owed
            # work whatever the file's mtime. The nightly window (touched in the
            # last 24 h) is a policy about changed documents; `openclaw/updates.md`
            # is appended to daily and would come back under it on its own, but a
            # feed that stopped being edited has nothing left to re-trigger it and
            # would sit 4.9 % covered forever. The window still decides for every
            # record that IS judgeable — this walk only says which records are.
            #
            # A read, because deleting an entry to make its file count as changed
            # is not a decision anyone can re-read: `--limit` truncates the
            # eligible list below, so a file de-registered and then capped away
            # lost its record and returned next run as a path the index has never
            # seen. Re-admission is membership in `re_admitted` instead.
            #
            # An entry carrying a coverage offset is self-describing and goes into
            # `resume_map` as-is; `has_changed` decides beside it whether the file
            # is due. What is left to judge is the entries that say nothing about
            # coverage — every entry written before #1151. Trusting those keeps the
            # oversized feeds skipped forever, which is the bug; distrusting all of
            # them would put the whole recorded corpus back through the model in
            # one night. Size separates them: a document shorter than
            # `CHUNK_BUDGET_CHARS` could only have been read whole by the pass that
            # recorded it, so its digest still means done. Byte size, because
            # bytes >= chars — the error runs one way, and re-reading a document
            # that did fit one pass costs a call, not a fact.
            re_admitted: set = set()
            untracked_long = []
            sweep_files = (eligible_files if full_mode
                           else self._eligible_files(full_mode=True))
            for md_file in sweep_files:
                covered, digest = hasher.coverage(md_file)
                if not digest:
                    continue
                if covered >= 0:
                    resume_map[md_file] = (covered, digest)
                    continue
                try:
                    too_long_for_one_pass = md_file.stat().st_size > CHUNK_BUDGET_CHARS
                except OSError:
                    too_long_for_one_pass = False
                if too_long_for_one_pass:
                    # Read it from the top this run. Its entry stays in the index
                    # exactly as it was, still silent about coverage, so a run whose
                    # `--limit` cap passes this file by leaves a record the next run
                    # can judge by the same size rule.
                    resume_map[md_file] = (0, digest)
                    re_admitted.add(md_file)
                    untracked_long.append(md_file.name)
            if untracked_long:
                print(f"Re-processing {len(untracked_long)} files whose digest "
                      f"predates coverage tracking and which exceed "
                      f"{CHUNK_BUDGET_CHARS} chars, so no single pass could have "
                      f"finished them: {untracked_long}")
            changed_files = [f for f in eligible_files
                             if f in re_admitted or hasher.has_changed(f)]
            # Re-admitted files the window excluded are owed work and join the
            # run anyway, in allow-list order so a `--limit` cap spends itself
            # deterministically instead of on whatever the mtime sort gave.
            in_window = set(eligible_files)
            owed_past_the_window = [f for f in sweep_files
                                    if f in re_admitted and f not in in_window]
            skipped_unchanged = len(eligible_files) - len(changed_files)
            eligible_files = changed_files + owed_past_the_window
            if skipped_unchanged > 0:
                print(f"Skipped {skipped_unchanged} unchanged files (content hash match)")

        # Optional per-run cap so a long backfill makes durable, bounded progress and
        # exits cleanly before the bash timeout (use with --full).
        if limit and limit > 0 and len(eligible_files) > limit:
            print(f"Limiting this run to {limit} of {len(eligible_files)} eligible files")
            eligible_files = eligible_files[:limit]

        # The `[N/M]` denominator is the queue this run will actually work, set
        # after BOTH the hash skip and the `--limit` cap. It was taken from the
        # pre-skip scan, so a finished run's last line read `[6/31]` (#1011); the
        # scan size stays on the `Found … eligible` / `Skipped …` lines above.
        total_files = len(eligible_files)

        # Incremental checkpoint: persist hashes for already-processed files every
        # CHECKPOINT_EVERY files so a timeout/kill never loses the whole run's work.
        # `pending` carries (path, digest) pairs, not paths: the flush at the end
        # of a run can land 545-1666s after the file was read, so re-hashing at
        # flush time would record whatever the note looks like NOW and mark an
        # appended-to-during-the-run edit as already extracted (#482).
        CHECKPOINT_EVERY = 25
        pending: list = []
        truncated: list = []

        def _record(md_file, source_hash, chars_covered, chars_total):
            """Book one successful extraction, and say how far of it we read.

            The digest always travels with the offset it was computed from;
            `complete` is the flag `ContentHasher.has_changed` consults when the
            digest matches again. Recording a partly-read document as complete is
            what skipped a 957,489-char feed at 4.9% coverage (#1151 clause 3).
            """
            complete = chars_covered >= chars_total
            pending.append((md_file, source_hash, chars_covered, complete))
            if not complete:
                truncated.append(md_file)

        def _flush_checkpoint():
            if hasher is not None and pending:
                try:
                    hasher.update_hashes(pending)
                    hasher.save()
                except Exception as e:
                    print(f"Warning: checkpoint failed: {e}")
                pending.clear()

        if workers == 1:
            # Sequential processing (default)
            for index, md_file in enumerate(eligible_files, 1):
                p, f, ok, source_hash, covered, chars_total = self._process_single_file(
                    md_file, full_mode, index, total_files,
                    resume=resume_map.get(md_file))
                processed += p
                total_facts += f
                if ok:
                    # Only a file we actually extracted gets its hash saved.
                    # Hashing a failed file marks it done forever — and hashing
                    # one that is only partly read marks its unread tail done,
                    # which is the same mistake one layer further in.
                    _record(md_file, source_hash, covered, chars_total)
                else:
                    failed += 1
                if len(pending) >= CHECKPOINT_EVERY:
                    _flush_checkpoint()
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(self._process_single_file, md_file, full_mode,
                                    index, total_files, resume_map.get(md_file)):
                    (index, md_file)
                    for index, md_file in enumerate(eligible_files, 1)
                }

                # Collect results
                for future in as_completed(futures):
                    index, md_file = futures[future]
                    try:
                        p, f, ok, source_hash, covered, chars_total = future.result()
                        processed += p
                        total_facts += f
                        if ok:
                            _record(md_file, source_hash, covered, chars_total)
                        else:
                            failed += 1
                        if len(pending) >= CHECKPOINT_EVERY:
                            _flush_checkpoint()
                    except Exception as e:
                        failed += 1
                        print(f"Error processing {md_file}: {e}")

        vault = VAULT.resolve()
        self.last_truncated_files = [
            str(p.relative_to(vault)) if vault in p.resolve().parents else str(p)
            for p in truncated
        ]
        print(f"Processed {processed} documents, extracted {total_facts} facts, "
              f"{failed} failed, {len(truncated)} truncated (tail not read: "
              f"{self.last_truncated_files})")
        self.last_failed_files = failed

        # Final checkpoint for any remaining processed files
        _flush_checkpoint()
        if hasher is not None:
            print(f"Updated content hashes for {processed} processed files "
                  f"({len(truncated)} of them mid-document: recorded with their "
                  "coverage offset and not as complete, so the next run continues "
                  "where this one stopped)")

        # Expose file-processed count so run_full_extraction can gate the
        # expensive downstream steps on whether any new/changed file was seen.
        self.last_files_processed = processed
        return total_facts
    
    def _regenerate_entity_overviews(self) -> int:
        """Regenerate entity overview files whose source facts have changed."""
        return self.profile_generator.regenerate_all(workers=8)


# ── single-instance lock ──────────────────────────────────────────────────────
# Two extractors racing rewrite `_pipeline/content-hashes.json` and the fact
# tree from under each other. Nothing in this script prevented that; the only
# guard was autonomy task #24's `in_progress` status, and that guard has a hole:
# the task's skill launches this script with `run_in_background`, so the child
# outlives its turn, and stale-task recovery flips the task back to `up_next`
# while the extractor is still working. A backend restart on 2026-09-08 produced
# exactly that state — an orphaned extractor at document 19 of 1395, with the
# task due to be freed for a second run 55 minutes later.
#
# flock, not a pidfile: the kernel releases it when the process dies, so a
# `kill -9` or an OOM cannot leave a stale lock that wedges the pipeline.
#
# `LLOYD_EXTRACTION_LOCK` overrides the path so a test can take a lock of its
# own instead of the live one: `tests/test_extraction_single_instance.py`
# used the live path, and on 2026-09-11 a real extractor (task #24, ~20 min
# a night) held it while the automod gate ran the suite — two tests red for
# every round that gated during that window, none of them about the round.
_LOCK_PATH = Path(os.environ.get("LLOYD_EXTRACTION_LOCK")
                  or (_STATE_PIPELINE / "nightly_extraction.lock"))


def acquire_single_instance_lock():
    """Return the held lock file, or None if another extractor owns it."""
    try:
        _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        fh = open(_LOCK_PATH, "w")
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return None
    except OSError as e:
        # A lock we cannot take is not a reason to refuse to run: this guard
        # protects against a race, and failing closed on e.g. a read-only
        # _pipeline would turn a rare overlap into a total outage.
        print(f"[lock] could not acquire ({e}); continuing without it")
        return False
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    return fh


def main():
    print("Nightly Deep Extraction - Next-Gen Memory")
    print("=" * 50)
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Nightly fact extraction for Next-Gen Memory System"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Process ALL files regardless of modification time (default: 24h window)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe facts directory before extraction"
    )
    parser.add_argument(
        "--rebuild-index-only",
        action="store_true",
        help="Rebuild the document relations index and stop (no extraction)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run the full downstream pipeline (relations/index/overviews) even "
             "when no new files were processed (overrides the noop short-circuit)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N eligible files this run (0 = no cap). "
             "Use with --full for durable, bounded backfill that checkpoints "
             "progress and exits cleanly before the bash timeout."
    )

    args = parser.parse_args()

    _lock = acquire_single_instance_lock()
    if _lock is None:
        # Carries every field the normal line carries: a reader parsing this
        # line out of a run log should not find a key missing because the run
        # never started.
        print("PIPELINE_RESULT files_processed=0 facts=0 failed=0 truncated=0 status=locked")
        print("Another nightly_extraction.py holds the lock; exiting without "
              "starting a second one.")
        return

    extraction = NightlyExtraction()
    
    if args.rebuild_index_only:
        # Index only: no extraction
        print("Quick mode: rebuilding index...")
        index = extraction.rel_generator.rebuild()
        print(f"Total relationships: {index['total_relationships']}")
    else:
        # Full extraction with options
        result = extraction.run_full_extraction(
            full_mode=args.full,
            workers=args.workers,
            clean=args.clean,
            limit=args.limit,
            force=args.force,
        )
        print("\nResult:", json.dumps(result, indent=2))
        # Single-line, grep-friendly summary for the autonomy-data-pipeline skill's
        # gate. status=noop means nothing changed → caller should skip downstream work.
        #
        # `truncated` is how many documents this run read only as far as the
        # chunk budget, and it is why the line no longer stops at `failed`: a run
        # that read 47,000 of 957,489 chars of a feed reported
        # `failed=0 status=ran` — indistinguishable from having read it — and the
        # file's digest was then recorded, so no later run read the rest either
        # (#1151 clause 4). Names are in the JSON result; the count is what a
        # grep of a run log sees.
        truncated = result.get("truncated_files") or []
        print(
            f"PIPELINE_RESULT files_processed={result.get('files_processed', 0)} "
            f"facts={result.get('facts_extracted', 0)} "
            f"failed={result.get('failed_files', 0)} "
            f"truncated={len(truncated)} "
            f"status={'noop' if result.get('noop') else 'ran'}"
        )


if __name__ == "__main__":
    main()
