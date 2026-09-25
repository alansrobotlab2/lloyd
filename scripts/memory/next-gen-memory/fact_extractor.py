#!/usr/bin/env python3
"""
Fact Extraction Pipeline - Next-Gen Memory System

Extracts atomic facts from documents using the local 2B LLM.
Integrates with periodic-memory-capture system.
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Shared entity normalization (case-insensitive alias resolution + self-register).
# Names resolve against the `aliases` table in app.kg_store, never against the
# legacy <VAULT_FACTS_ROOT>/entity-aliases.json export — a frozen migration
# snapshot that reads as ~5x the live table (#474). Without normalization,
# writers accumulate duplicate dirs (gr00t/ + GR00T/).
_LLOYD_ROOT = Path(__file__).resolve().parents[3]
if str(_LLOYD_ROOT) not in sys.path:
    sys.path.insert(0, str(_LLOYD_ROOT))
from app.entity_naming import known_entities_in_text as _known_entities
from app.entity_naming import looks_like_junk_entity as _is_junk_entity
from app.entity_naming import ENTITY_CANDIDATES_PATH, SCHEMA_TYPES
# The declared-identity gate (#537). `normalize_and_register` — mint a canonical
# on every alias miss — is what produced the sibling families the schema now
# declares; it stays available for callers outside this write path.
from app.entity_naming import gate_entity_name as _identity_gate
# #743: a tracker citation is refused as its own verdict rather than folded into
# `junk`, so the name reaches the candidates sidecar and the fact is dropped
# instead of refiled under the document's primary entity.
from app.entity_naming import is_backlog_citation_entity as _is_citation_entity
from app.entity_naming import record_entity_candidate as _record_candidate
from app.atomic_io import atomic_write_text, locked_file
from app.frontmatter import split_frontmatter
from app.fact_ids import assign_ids as _assign_fact_ids
from app.kg_store import StoreUnavailable, text_hash, store as _kg_store

# The seven fallback classes that used to sit here (a hand-rolled YAML parser,
# a pass-through entity normaliser, a never-junk predicate) each turned a
# missing dependency into silently wrong output: unparseable frontmatter read
# as an empty fact list, which `write_fact_file` then wrote back as truth.
# An ImportError is the correct outcome — the extractor cannot do its job
# without these.

import yaml

# Constants
HOME = Path.home()
VAULT = HOME / "obsidian"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from app.paths import VAULT_FACTS_ROOT as FACTS_DIR

# Chunked extraction. The old hard `content[:3000]` cap dropped ~40% of the median
# vault doc (70% of docs exceed 3000 chars) — including the structured Tools /
# GitHub / Papers sections at the bottom of YouTube notes. We now window the doc
# so most files (≤ CHUNK_SIZE) are still a single LLM call.
#
# What the windowing does NOT do is read a long document: one pass sends at most
# MAX_CHUNKS chunks, so it covers at most CHUNK_BUDGET_CHARS of one document, and
# whatever lies past that is unread by that pass. This comment used to promise
# that longer docs were read end to end, and the same promise sat in
# `_chunk_content`'s docstring — which is how a 957,489-char feed could have 4.9%
# of its chars sent to the model, be reported as processed with no failure, get
# its content hash recorded, and then be skipped by every later run (#1151).
# Coverage is a value now, not a warning line: `extract_from_document` returns
# `chars_covered`/`chars_total`, and the caller resumes an unfinished document
# from the offset the last pass stopped at.
CHUNK_SIZE = 8000          # chars per chunk (~2000 tokens); 90% of docs fit in one
CHUNK_OVERLAP = 200        # carry a little context across the cut so facts aren't split
MAX_CHUNKS = 6             # chunks per document PER PASS, not per document
# Chars one pass over one document can reach. Beyond this the pass stops and the
# remainder is the next pass's job: 5 steps of 7,800 plus the last full chunk.
CHUNK_BUDGET_CHARS = (MAX_CHUNKS - 1) * (CHUNK_SIZE - CHUNK_OVERLAP) + CHUNK_SIZE

# One model call's client ceiling. It was a flat 120 s, which an idle engine
# meets (a 6000-token call measured 23.8 s) and a busy one cannot: these calls
# run at priority 2, behind chat and autonomy, and on 2026-09-23 six changed
# documents all timed out queued behind `profile_generator --all` (#1405). The
# waiting is in the engine's queue, not in generation, so the bound has to
# cover queueing. LLOYD_EXTRACTION_CALL_TIMEOUT_S overrides it for a run.
try:
    LLM_CALL_TIMEOUT_S = float(os.environ.get("LLOYD_EXTRACTION_CALL_TIMEOUT_S") or 600)
except ValueError:
    LLM_CALL_TIMEOUT_S = 600.0

# The category vocabulary. 287 distinct category spellings existed on
# 2026-09-03 — `state`, `States`, `current state`, `state/config` and so on —
# because the model's free-text answer was written through verbatim. Each
# spelling makes its own fact file, so `fact_get(entity, category="state")`
# missed most of the entity's state facts. Unknown answers map to the nearest
# by token overlap, else `general`.
CATEGORY_VOCAB = (
    "state", "event", "decision", "preference", "goal", "skill",
    "relationship", "capability", "constraint", "configuration",
    "hardware", "research", "general",
)
_CATEGORY_TOKENS = {c: set(re.findall(r"[a-z]+", c)) for c in CATEGORY_VOCAB}
# Spellings seen in the tree that map cleanly onto the vocabulary.
_CATEGORY_ALIASES = {
    "status": "state", "current state": "state", "states": "state",
    "config": "configuration", "settings": "configuration",
    "preferences": "preference", "prefs": "preference",
    "relationships": "relationship", "relations": "relationship",
    "events": "event", "history": "event", "timeline": "event",
    "goals": "goal", "objectives": "goal", "skills": "skill",
    "capabilities": "capability", "constraints": "constraint",
    "decisions": "decision", "temporary": "state", "fact": "general",
    "facts": "general", "info": "general", "notes": "general",
}


class ExtractionFailed(RuntimeError):
    """The model call failed or returned nothing usable.

    Raised rather than returning an empty fact list. An empty list was
    indistinguishable from `this document genuinely has no facts`, so a
    transient vLLM error marked the document extracted and its content hash
    was saved — the document was then never revisited.
    """


def normalize_category(raw: str | None) -> str:
    """Map a model-supplied category onto CATEGORY_VOCAB."""
    c = (raw or "").strip().lower()
    c = re.sub(r"[^a-z ]+", " ", c).strip()
    if not c:
        return "general"
    if c in _CATEGORY_TOKENS:
        return c
    if c in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[c]
    toks = set(c.split())
    best, best_score = "general", 0.0
    for cat, cat_toks in _CATEGORY_TOKENS.items():
        overlap = len(toks & cat_toks)
        if not overlap:
            continue
        score = overlap / len(toks | cat_toks)
        if score > best_score:
            best, best_score = cat, score
    return best if best_score >= 0.3 else "general"

EXTRACTION_PROMPT = """You are a fact extraction engine. Analyze the following content and extract
atomic facts about entities mentioned.

Rules:
1. Each fact must be a single, unambiguous statement
2. Include temporal context if mentioned ("moved last week" → event_date)
3. A document usually covers SEVERAL distinct entities (people, projects, tools,
   papers, systems). Attribute EACH fact to the specific entity it is about via
   that fact's "entity" field — do NOT collapse everything onto one entity.
4. Categorize each fact: preference, relationship, event, state, skill, goal, temporary
5. Note confidence level (0.0-1.0)
6. If a fact contradicts a known fact, flag it as an update
7. Declare what kind of thing every entity IS, in its "entity_type": exactly one
   of {entity_types}. This is not decoration — an entity with no declared type
   is not created at all, it goes to a review list. Do not invent a type word
   and do not invent a compound one ("data pipeline" is wrong, "pipeline" is right).

CRITICAL GUARDRAILS:
- NEVER extract "session" as an entity. Session metadata (duration, message count,
  triviality, health checks, emptiness) is NOT a valid entity.
- NEVER create entities named "session", "session-distill", "session_<timestamp>",
  "session_pong5", or any session identifier. Sessions are the SOURCE of data,
  not entities to be extracted.
- If content is from a trivial/empty session (<5 messages, health check, routine
  maintenance), return an empty facts list. Do NOT fabricate observations like
  "the session was short" or "had no unresolved threads".
- Only extract facts about actual domain knowledge, tools, decisions, people,
  systems, and concepts discussed — never about the session container itself.

Content:
{content}

Known entities already in the knowledge graph that appear in this content.
When a fact is about one of these, put this EXACT name in its "entity" field —
do not add or drop words like "System", "Pipeline", "Agent", "SDK", "App", and
do not re-spell or re-case it. Only coin a new entity name when none of these
is the thing the fact is about:
{known_entities}

Known facts about relevant entities:
{existing_facts}

Extract facts as structured JSON. Set the top-level "entity" to the single most
central entity of the content; give EVERY fact its own "entity" naming the
specific thing that fact is about (it may differ from the top-level entity), and
give EVERY entity its own "entity_type" (leave it "" only if you genuinely
cannot say what kind of thing it is — that means it will not be created):
{{
  "entity": "primary_entity_name",
  "entity_type": "system",
  "category": "category_name",
  "facts": [
    {{
      "entity": "entity this fact is about",
      "entity_type": "one of {entity_types}",
      "fact": "Fact statement",
      "confidence": 0.95,
      "event_date": null,
      "category": "state"
    }}
  ]
}}
"""

class FactExtractor:
    """Extracts facts from documents using LLM."""
    
    def __init__(self, model_port: int = 8096):
        self.model_port = model_port
        self.facts_dir = FACTS_DIR
        # What `_index_and_link` did this process, for the nightly run summary:
        # `mentions_linked` is edges it asked the store for, and
        # `mentions_skipped_typed` the pairs it left alone because a classifier
        # verdict was already live on them (#1246). Neither is a PIPELINE_RESULT
        # key; the grep-able line keeps its shape.
        self.link_stats = {"mentions_linked": 0, "mentions_skipped_typed": 0}
    
    def _chunk_content(self, content: str, start: int = 0) -> list:
        """Window ONE pass over a document.

        Returns `[(source_offset, chunk_text), ...]`: at most `MAX_CHUNKS`
        chunks of `CHUNK_SIZE` chars, stepping `CHUNK_SIZE - CHUNK_OVERLAP`
        chars, beginning at `start`. A pass therefore reaches at most
        `CHUNK_BUDGET_CHARS` (47,000) chars of a document — everything past that
        is NOT sent to the model by this pass and is the caller's to resume,
        which is why the offsets come back with the text (#1151). When the rest
        of the document fits one chunk the answer is `[(start, content[start:])]`.
        """
        start = max(0, min(int(start), len(content)))
        if len(content) - start <= CHUNK_SIZE:
            return [(start, content[start:])]
        chunks = []
        step = CHUNK_SIZE - CHUNK_OVERLAP
        pos = start
        while pos < len(content) and len(chunks) < MAX_CHUNKS:
            chunks.append((pos, content[pos:pos + CHUNK_SIZE]))
            pos += step
        return chunks

    def _parse_response(self, response: str) -> dict:
        """Parse one LLM response into {entity, category, facts}. Tolerant of code
        fences and list-vs-object shapes."""
        try:
            if response.startswith("```"):
                response = re.sub(r'^```\w*\n?', '', response)
                response = re.sub(r'\n?```$', '', response)
            result = json.loads(response)
        except json.JSONDecodeError:
            print(f"Failed to parse LLM response as JSON: {response[:200]}")
            return {"entity": None, "category": None, "facts": []}

        if isinstance(result, list):
            if not result:
                return {"entity": None, "category": None, "facts": []}
            base = result[0] if isinstance(result[0], dict) else {"entity": None, "category": None, "facts": []}
            for extra in result[1:]:
                if isinstance(extra, dict) and extra.get("facts"):
                    base.setdefault("facts", []).extend(extra["facts"])
            result = base
        elif not isinstance(result, dict):
            return {"entity": None, "category": None, "facts": []}
        return result

    def extract_from_document(self, doc_path: Path, content: str,
                              existing_facts: str = "",
                              start_offset: int = 0) -> dict:
        """Extract facts from ONE pass over a document.

        Docs longer than `CHUNK_SIZE` are windowed (see `_chunk_content`) and
        extracted chunk-by-chunk; facts from the chunks are merged and
        de-duplicated by text. Each fact carries its own sanitized "entity" so a
        multi-entity doc fans out to per-entity fact files instead of collapsing
        onto one primary entity.

        `start_offset` is where this pass begins reading `content`; 0 is the top
        of the document. A caller passes the offset a previous pass reported when
        that pass stopped short of the end, which is how a document too long for
        one pass is finished over several passes instead of having its tail
        dropped (#1151).

        Returns {entity, category, facts, chars_covered, chars_total}:
        `chars_total` is len(content), `chars_covered` is the end offset of the
        last chunk this pass sent to the model — `chars_total` when this pass read
        what it was given, `start_offset + CHUNK_BUDGET_CHARS` when more of the
        document remained than one pass can reach. A document whose tail went
        unread is reported as such rather than printing a warning nobody reads.
        """
        existing = existing_facts[:1000] if existing_facts else "None"
        primary_entity = None
        primary_category = None
        all_facts = []
        seen_fact_text = set()
        gated_out = 0
        # Starts at the top of what this pass was given, not at the end: the
        # figure has to be earned by a chunk that reached the model, or a
        # document that raised before the first call would report itself read.
        chars_covered = max(0, min(int(start_offset), len(content)))

        for offset, chunk in self._chunk_content(content, start_offset):
            chars_covered = max(chars_covered, offset + len(chunk))
            known = _known_entities(chunk, 60)
            known_block = "\n".join(f"- {k}" for k in known) if known else "(none recognised)"
            prompt = EXTRACTION_PROMPT.format(content=chunk, existing_facts=existing,
                                              known_entities=known_block,
                                              entity_types=" | ".join(SCHEMA_TYPES))
            parsed = self._parse_response(self._call_llm(prompt))
            # The document's own declared type is the fallback for a fact that
            # omits one, so a model that answers the type once per document
            # still gets the benefit of the doubt on its facts.
            doc_type = parsed.get("entity_type")

            if primary_entity is None and parsed.get("entity"):
                primary_entity, _v = self._gate_entity(parsed["entity"],
                                                       declared_type=doc_type)
                # A category is a vocabulary term, not an entity. Running it
                # through _sanitize_entity registered every distinct spelling
                # as a canonical entity in the alias table.
                primary_category = normalize_category(parsed.get("category"))

            for f in parsed.get("facts", []):
                if not isinstance(f, dict):
                    continue
                text = (f.get("fact") or "").strip()
                if not text or text in seen_fact_text:
                    continue
                seen_fact_text.add(text)
                # Resolve each fact to its own entity (falls back to the doc primary
                # at write time when absent).
                fact_type = f.get("entity_type") or doc_type
                if f.get("entity"):
                    entity, verdict = self._gate_entity(f["entity"],
                                                        declared_type=fact_type)
                    if verdict == "candidate":
                        # The subject is not an entity this run may create, and
                        # its name is in the sidecar. Filing the fact under the
                        # document's primary entity would attribute it to the
                        # wrong thing, which is the failure this gate exists to
                        # stop making.
                        gated_out += 1
                        f.pop("entity_type", None)
                        continue
                    f["entity"] = entity
                f.pop("entity_type", None)
                f["category"] = normalize_category(f.get("category"))
                all_facts.append(f)

        if gated_out:
            print(f"  ⤫ {gated_out} fact(s) held back: entity not declared and "
                  f"carried no valid type → {ENTITY_CANDIDATES_PATH}")

        return {
            "entity": primary_entity or "general",
            "category": primary_category or "general",
            "facts": all_facts,
            "chars_covered": chars_covered,
            "chars_total": len(content),
        }
    
    def _call_llm(self, prompt: str) -> str:
        """Call the local model. Raises ExtractionFailed rather than faking
        an empty extraction — see the class docstring."""
        import urllib.request

        url = f"http://localhost:{self.model_port}/v1/chat/completions"
        payload = {
            "model": "primary",
            "messages": [
                {"role": "system", "content": "You are a fact extraction engine."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 6000,
            "chat_template_kwargs": {"enable_thinking": False},
            # vLLM --scheduling-policy priority: chat sends 0, autonomy 1. This
            # ran at the default and competed with both.
            "priority": 2,
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=LLM_CALL_TIMEOUT_S) as response:
                data = json.loads(response.read().decode('utf-8'))
                text = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise ExtractionFailed(f"LLM call failed: {e}") from e
        if not text or not text.strip():
            raise ExtractionFailed("LLM returned empty content")
        return text

    def get_existing_facts(self, entity: str, category: str) -> str:
        """Load existing facts for an entity/category."""
        # Sanitize entity and category to prevent nested path creation.
        # `enforce=False`: this is a read. The gate may attach the name to its
        # declared canonical — which is how a variant gets to see the facts it
        # should have seen all along — but withholding them for want of a
        # declared type would degrade the extraction, not protect the graph.
        entity = self._sanitize_entity(entity, enforce=False)
        category = normalize_category(category)
        if not entity:
            return ""
        fact_file = self.facts_dir / entity / f"{entity}-{category}.md"
        if not fact_file.exists():
            return ""
        
        content = fact_file.read_text()
        # Extract YAML frontmatter, bounded by the closing fence LINE: a fact
        # quoting '---' cut the substring split mid-scalar and the prompt was
        # shown a truncated list (#1400).
        if content.startswith("---"):
            split = split_frontmatter(content)
            if split is not None:
                return split[0].strip()
            return content[3:].strip()
        return ""
    
    def _gate_entity(self, entity: str, source_doc: str | None = None,
                     declared_type: str | None = None, *,
                     enforce: bool = True) -> tuple[str, str]:
        """Sanitize, then apply the declared-identity gate. Returns (name, verdict).

        Sanitization strips path characters; the junk predicate runs BEFORE
        registration, so a leaked filename or a pipeline run name never
        enters the alias table. (It used to be registered first and rejected
        at write time, which is why 921 run-named canonicals existed.)
        """
        if not entity:
            return "", "junk"
        # Take last path component if slashes present
        entity = entity.strip().split("/")[-1].split("\\")[-1]
        # Remove any remaining path-unsafe characters
        entity = re.sub(r'[<>:"|?*]', '', entity)
        # Collapse whitespace
        entity = re.sub(r'\s+', ' ', entity).strip()
        if not entity:
            return "", "junk"
        if _is_citation_entity(entity):
            # A tracker citation (#743), checked before the junk predicate and
            # before the gate, and answered `candidate` rather than `junk` for
            # two reasons. `extract_from_document` drops a fact whose verdict is
            # `candidate`; on `junk` it keeps the fact with an empty entity, and
            # `nightly_extraction.py:340` turns an empty entity into the
            # document's primary, which files the fact against the wrong thing.
            # And the sidecar line is the record that this name was seen — a
            # guard that drops a whole class of names in silence stops
            # remembering things and nobody notices.
            _record_candidate(entity, reason="backlog/board-item citation",
                              source_doc=source_doc, declared_type=declared_type)
            return "", "candidate"
        if _is_junk_entity(entity, source_doc):
            return "", "junk"
        return _identity_gate(entity, declared_type=declared_type,
                              source_doc=source_doc, enforce=enforce)

    def _sanitize_entity(self, entity: str, source_doc: str | None = None, *,
                         enforce: bool = True,
                         declared_type: str | None = None) -> str:
        """Name an entity may file under, or "" when it may not.

        `enforce=False` is for READS (`get_existing_facts`): the schema may
        attach a variant to its canonical there, but refusing a name for lack
        of a declared type would silently withhold existing facts from the
        prompt, which degrades extraction instead of protecting the graph.
        """
        name, verdict = self._gate_entity(entity, source_doc, declared_type,
                                          enforce=enforce)
        return name if verdict not in ("junk", "candidate") else ""

    def write_fact_file(self, entity: str, category: str, facts_data: dict,
                        *, source_doc: str | None = None,
                        source_hash: str | None = None) -> Path | None:
        """Write or update a fact file.

        Every fact written here carries `created_at`, `source_doc`,
        `source_hash` and `provenance`, so it can be dated, attributed and
        selectively reverted. 99.7% of the 205k facts in the pre-2026-09 tree
        had none of those, which is why nothing could tell a fact extracted
        from a real vault note from one extracted out of the pipeline's own
        exhaust.

        Returns the written path, `None` when the entity is rejected as junk,
        when the existing file had to be quarantined, or when every fact in the
        batch was a copy the entity already holds and the target file does not
        exist to fold it into (#1144).
        """
        # `enforce=False` because this is not where a name enters: every name
        # arriving from the model has been through `extract_from_document`,
        # which is where the schema gate lives and where a rejected name is
        # recorded. Re-deciding here would reject callers that have no
        # `entity_type` to offer (fact-improvement, the classifiers) without
        # stopping the minting this item is about.
        entity = self._sanitize_entity(entity, source_doc, enforce=False)
        category = normalize_category(category)

        if not entity:
            print(f"  ⤫ skipped junk entity ({len(facts_data.get('facts', []))} facts dropped)")
            return None

        entity_dir = self.facts_dir / entity
        entity_dir.mkdir(parents=True, exist_ok=True)
        fact_file = entity_dir / f"{entity}-{category}.md"

        now_iso = datetime.now(timezone.utc).isoformat()
        new_facts = facts_data.get("facts", [])
        for nf in new_facts:
            if not isinstance(nf, dict):
                continue
            nf.setdefault("provenance", "EXTRACTED")
            nf.setdefault("created_at", now_iso)
            if source_doc:
                nf.setdefault("source_doc", source_doc)
            if source_hash:
                nf.setdefault("source_hash", source_hash)
            # An event the fact itself dates is when it was true, not when we
            # read it. `created_at` stays the extraction time either way.
            if nf.get("event_date") and not nf.get("valid_at"):
                nf["valid_at"] = str(nf["event_date"])
            nf.setdefault("expired_at", None)
            nf.setdefault("invalid_at", None)

        # The lock covers the whole read-modify-write. Four extractor threads
        # and `fact_add` from a chat turn all target the same file; without it
        # the later writer silently drops the earlier one's facts.
        with locked_file(fact_file):
            existing_facts = self._read_existing_facts(fact_file)
            if existing_facts is None:
                return None            # quarantined; do not write over it

            # #1144: the cross-category leak. `_merge_facts` can only see THIS
            # file, so a fact whose text the entity already holds under another
            # category was appended as a second indexed row. It is still
            # happening after #499: measured on the live store at 2026-09-21
            # 07:34Z, 95 same-entity duplicate groups have BOTH copies created
            # on or after 2026-09-14, the day #499's refusal settled. Refuse on
            # the same lookup `_fact_add` uses — `facts_idx.find_duplicate`,
            # keyed `(entity, text_hash)` across every category — and inside the
            # lock for the same reason `_fact_add` does it there: two concurrent
            # writers must not both pass the check and both append.
            incoming = len(new_facts)
            new_facts = self._refuse_held_facts(entity, new_facts)
            refused = incoming - len(new_facts)
            if refused:
                print(f"  ⤫ {fact_file.name}: refused {refused} duplicate fact(s) "
                      f"{entity} already holds")
            if refused and not new_facts and not existing_facts:
                # The whole batch was a copy and this category file holds
                # nothing: writing it would create an empty file behind a write
                # that added no fact, the shape `_duplicate_refusal` (#499)
                # avoids by never opening the file at all.
                return None

            # #1487: the paraphrase gate `_fact_add` carries, on the writer that
            # produces most paraphrases — re-reading a changed document states
            # its claims again in new words. Same module, same mode switch.
            new_facts = self._gate_paraphrases(entity, category, existing_facts,
                                               new_facts, now_iso, source_doc)

            merged_facts = self._merge_facts(existing_facts, new_facts)
            merged_facts = _assign_fact_ids(merged_facts, category)

            frontmatter = {
                "type": "facts",
                "entity": entity,
                "category": category,
                "facts": merged_facts,
                "last_extracted": now_iso,
                "last_updated": now_iso,
            }
            if source_doc:
                frontmatter["source_doc"] = source_doc

            yaml_content = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
            markdown_body = self._generate_markdown_body(entity, category, merged_facts)
            atomic_write_text(fact_file, f"---\n{yaml_content}---\n\n{markdown_body}")

        self._index_and_link(entity, category, fact_file, new_facts, source_doc)
        return fact_file

    def _gate_paraphrases(self, entity: str, category: str, existing: list,
                          new_facts: list, now_iso: str, source_doc) -> list:
        """The incoming facts the #1487 write gate lets through.

        A NOOP drops the fact; an UPDATE stamps `expired_at` on the one existing
        entry it names, in `existing` itself, so the expiry rides the same
        atomic write as the fact replacing it. Each kept fact joins the pool
        the next one is judged against, so a batch cannot restate itself.
        Off (the default) or any failure to load the gate: the list unchanged.
        """
        try:
            from agent_mcp import fact_write_gate
            if fact_write_gate.mode() == "off" or not new_facts:
                return new_facts
        except Exception:  # noqa: BLE001 — the gate never costs the write
            return new_facts
        pool = [f for f in existing if isinstance(f, dict)]
        kept = []
        for nf in new_facts:
            if not isinstance(nf, dict):
                kept.append(nf)
                continue
            took, _ = fact_write_gate.gate_write(
                entity, category, str(nf.get("fact") or ""), pool,
                now_iso=now_iso, source_doc=source_doc)
            if took == "noop":
                continue
            kept.append(nf)
            pool.append(nf)
        if len(kept) < len(new_facts):
            print(f"  ⤫ {entity}-{category}: write gate held back "
                  f"{len(new_facts) - len(kept)} restatement(s)")
        return kept

    def _refuse_held_facts(self, entity: str, new_facts: list) -> list:
        """The incoming facts minus any whose text `entity` already carries.

        One key with `fact_add`, not two (#1144). #499 put the cross-category
        refusal in `_fact_add` alone, so the highest-volume writer of fact files
        kept minting the copies that guard was written to stop: of the duplicate
        groups on the live store 2026-09-21, 95 have both rows created after
        #499 settled. `facts_idx.find_duplicate` is the one lookup that answers
        "does this entity already hold this text, in ANY category" — an
        extractor-local version of it would be a second key to keep in step.

        An expired or invalid copy does not refuse a new one (the store's own
        rule): re-stating a superseded claim is a new claim. And nothing here
        expires anything.

        When the store cannot be read the list comes back whole, which is
        exactly the pre-#1144 behaviour: `_merge_facts` still folds the copies
        inside this one file, so an unreadable index narrows the guard to the
        file rather than dropping facts. A failed lookup likewise keeps the
        fact — this guard refuses writes, so failing toward writing loses
        nothing and failing toward dropping would be a new way to lose a claim.
        """
        if not new_facts:
            return new_facts
        try:
            facts_idx = _kg_store().facts_idx
        except StoreUnavailable:
            return new_facts
        kept: list = []
        for nf in new_facts:
            text = str(nf.get("fact") or "") if isinstance(nf, dict) else ""
            if text.strip():
                try:
                    if facts_idx.find_duplicate(entity, text):
                        continue
                except Exception as e:      # noqa: BLE001 - see the docstring
                    print(f"  ⚠ duplicate lookup failed for {entity!r}: {e}")
            kept.append(nf)
        return kept

    def _read_existing_facts(self, fact_file: Path) -> list | None:
        """Existing facts, or None when the file is corrupt and was quarantined.

        A YAML error used to fall through to `existing_facts = []`, and the
        very next statement wrote the file back with only the new facts in it.
        One unparseable character therefore deleted an entity's whole history.
        The file is renamed instead, and this extraction is skipped.
        """
        if not fact_file.exists():
            return []
        try:
            content = fact_file.read_text(encoding="utf-8")
        except OSError as e:
            print(f"  ⤫ cannot read {fact_file.name}: {e}")
            return None
        if not content.strip():
            return []
        if not content.startswith("---"):
            return self._quarantine(fact_file, "no frontmatter fence")
        # The closing fence is a whole `---` line, never the first `---`
        # substring: a fact quoting '---' cut the YAML mid-scalar here, and the
        # truncated list was re-dumped over the file — the fact left as a
        # provenance-less fragment and every fact after it deleted (#1400).
        split = split_frontmatter(content)
        if split is None:
            return self._quarantine(fact_file, "unterminated frontmatter")
        try:
            frontmatter = yaml.safe_load(split[0])
        except Exception as e:
            return self._quarantine(fact_file, f"YAML error: {e}")
        if not isinstance(frontmatter, dict):
            return self._quarantine(fact_file, "frontmatter is not a mapping")
        facts = frontmatter.get("facts")
        if facts is None:
            return []
        if not isinstance(facts, list):
            return self._quarantine(fact_file, "`facts` is not a list")
        return [f for f in facts if isinstance(f, dict)]

    @staticmethod
    def _quarantine(fact_file: Path, why: str) -> None:
        """Rename a corrupt fact file aside and return None."""
        stamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        dest = fact_file.with_name(f"{fact_file.name}.corrupt-{stamp}")
        try:
            fact_file.rename(dest)
            print(f"  ⚠ quarantined {fact_file.name} ({why}) → {dest.name}")
        except OSError as e:
            print(f"  ⚠ {fact_file.name} is corrupt ({why}) and could not be moved: {e}")
        return None

    def _index_and_link(self, entity: str, category: str, fact_file: Path,
                        new_facts: list, source_doc: str | None) -> None:
        """Index the file and emit the edges its facts imply.

        This is the graph's growth path. Before it, edges only appeared when
        someone ran `seed_relationship_edges.py` by hand, which is why node
        coverage sat at 13.7% and the nightly chain added no edges at all.

        A `mentions` edge is emitted for every OTHER known entity a fact names
        — `relationship`-category facts included, which is where the densest
        signal is. The v4 classifier upgrades them to typed relations.
        """
        try:
            st = _kg_store()
        except StoreUnavailable as e:
            print(f"  ⚠ store unavailable, not indexing {fact_file.name}: {e}")
            return
        try:
            st.entities.register(entity)
            st.facts_idx.update_file(fact_file, root=self.facts_dir)
            with st.transaction():
                for f in new_facts:
                    if not isinstance(f, dict):
                        continue
                    text = (f.get("fact") or "").strip()
                    if not text:
                        continue
                    subject = f.get("entity") or entity
                    for target in _known_entities(text, 12):
                        if target == subject:
                            continue
                        # A pair the v4 classifier has already typed keeps its
                        # verdict. `edges.add` dedupes on the exact
                        # (source, target, type), so a `mentions` row beside an
                        # active `uses` was accepted, and the next nightly apply
                        # re-typed the pair to the type it already carried —
                        # 165 such pairs on 2026-09-20, 3-29% of each apply's
                        # output a no-op retype (#1246).
                        if any(r["type"] != "mentions"
                               for r in st.edges.active(source=subject, target=target)):
                            self.link_stats["mentions_skipped_typed"] += 1
                            continue
                        try:
                            self.link_stats["mentions_linked"] += 1
                            st.edges.add({
                                "source": subject, "target": target, "type": "mentions",
                                "confidence": float(f.get("confidence", 0.8) or 0.8),
                                "provenance": "EXTRACTED",
                                "source_doc": source_doc,
                                "evidence": text[:500],
                            }, origin="extractor")
                        except ValueError:
                            continue   # self-loop or blank endpoint
        except Exception as e:  # the markdown is written; the index is derived
            print(f"  ⚠ index/link failed for {fact_file.name}: {e}")

    def _merge_facts(self, existing: list, new: list) -> list:
        """Merge new facts into existing, keeping one copy of each claim.

        The key is the store's own `text_hash` (strip, casefold, sha256[:16])
        rather than raw string equality, so "a duplicate here" and "a duplicate
        row in `facts_idx`" are the same relation. A casing difference used to
        pass this check and land as two indexed rows.

        #499 clause 4: this used to open with `merged = existing.copy()`, which
        refused a repeat arriving IN while preserving a repeat already THERE —
        so every pass over one of the 708 files holding the same text twice
        re-indexed both copies, forever. Existing copies are folded now too.

        Except where two copies disagree about retirement. One carrying
        `expired_at` or `invalid_at` and one carrying neither are not
        interchangeable: folding them either resurrects a retired claim or
        deletes the record that it was retired — and #499 records what deciding
        that by confidence did to `fact_entity_recall` (0.35 → 0.30). Those
        survive, one per status. This merge still never expires anything.
        """
        merged: list = []
        seen: set = set()

        for fact in list(existing) + list(new):
            if not isinstance(fact, dict):
                merged.append(fact)          # not ours to fold; keep it verbatim
                continue
            key = (text_hash(fact.get("fact", "")),
                   bool(fact.get("expired_at")), bool(fact.get("invalid_at")))
            if key in seen:
                continue
            seen.add(key)
            merged.append(fact)

        return merged
    
    def _assign_ids(self, facts: list, category: str) -> list:
        """Kept as a method for callers; the scheme lives in app.fact_ids."""
        return _assign_fact_ids(facts, category)

    def _generate_markdown_body(self, entity: str, category: str, 
                                facts: list) -> str:
        """Generate human-readable markdown body for fact file."""
        lines = [
            f"# {entity.title()} - {category.title()}",
            "",
            f"**Entity:** {entity}",
            f"**Category:** {category}",
            f"**Fact Count:** {len(facts)}",
            "",
            "## Facts",
            ""
        ]
        
        for fact in facts:
            fact_id = fact.get("id", "unknown")
            fact_text = fact.get("fact", "")
            confidence = fact.get("confidence", 0.0)
            status = fact.get("status", "current")
            
            lines.extend([
                f"### {fact_id}",
                "",
                f"**Fact:** {fact_text}",
                f"**Confidence:** {confidence}",
                f"**Status:** {status}",
                ""
            ])
        
        return "\n".join(lines)


if __name__ == "__main__":
    print("Fact Extraction Pipeline — imported by nightly_extraction.py.")
    print("There is no standalone entry point: run the nightly extraction, "
          "which owns the corpus selection, the content-hash gate and the "
          "failure accounting this module deliberately does not.")
