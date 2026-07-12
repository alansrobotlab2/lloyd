#!/usr/bin/env python3
"""
Groundskeeper Survey Script
Scans the Obsidian vault for issues and produces a queue JSON file.

Scan categories:
1. BROKEN_LINK - Wiki-links that don't resolve to existing files
2. STALE_FACT - Facts with last_updated > 30 days old
3. MEMORY_HYGIENE - Missing file references in MEMORY.md
4. ORPHAN_FILE - Files with zero inbound wiki-links or relations
5. THIN_PROFILE - Entities with < 3 facts (no knowledge/projects references)
6. STALE_RELATION - Broken relation paths in frontmatter
7. ENRICH_THIN_PROFILE - Entities with <3 facts AND 2+ references in knowledge/projects
8. ENRICH_STALE_TOPIC - Knowledge files not updated in >7 days with 2+ inbound links
9. ENRICH_STUB - Files in knowledge/projects with <200 chars of content
10. MISSING_FRONTMATTER - Files missing frontmatter or required fields (segment, type)
11. LARGE_DOC - Documents over 300 lines that may need splitting
"""

import os
import re
import sys
import json
import glob
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.paths import VAULT_FACTS_ROOT

VAULT_ROOT = "/home/alansrobotlab/obsidian"
QUEUE_OUTPUT = os.path.expanduser("~/lloyd/_pipeline/groundskeeper-queue.json")
FACTS_DIR = str(VAULT_FACTS_ROOT)
MEMORY_MD = os.path.join(VAULT_ROOT, "lloyd/MEMORY.md")


def load_existing_queue():
    """Load existing queue to preserve status of pending items."""
    if os.path.exists(QUEUE_OUTPUT):
        try:
            with open(QUEUE_OUTPUT, 'r') as f:
                data = json.load(f)
                return {item['id']: item for item in data.get('items', [])}
        except (json.JSONDecodeError, KeyError):
            pass
    return {}


def get_all_md_files():
    """Get all markdown files in the vault."""
    pattern = os.path.join(VAULT_ROOT, "**", "*.md")
    return [f for f in glob.glob(pattern, recursive=True) if os.path.isfile(f)]


def build_file_index(all_files):
    """Build indices for wiki-link resolution.
    
    Returns:
        basenames: {lowercase_name_without_ext: [full_paths]}
        rel_paths: {lowercase_relative_path_without_ext: full_path}
    """
    basenames = defaultdict(list)
    rel_paths = {}
    
    for filepath in all_files:
        basename = os.path.basename(filepath).lower()
        name_without_ext = basename[:-3] if basename.endswith('.md') else basename
        basenames[name_without_ext].append(filepath)
        
        # Also index by relative path (for path-style wiki-links)
        rel = os.path.relpath(filepath, VAULT_ROOT).lower()
        rel_no_ext = rel[:-3] if rel.endswith('.md') else rel
        rel_paths[rel_no_ext] = filepath
    
    return basenames, rel_paths


def resolve_wiki_link(link, basenames, rel_paths, all_files):
    """Try to resolve a wiki-link to an existing file.
    
    Returns True if the link resolves, False if broken.
    
    Handles:
    - Relative paths ([[projects/alfie/notes]])
    - Case-insensitive matching
    - Heading anchors ([[notes#section]]) - strips anchor
    - Display text ([[notes|alias]]) - strips alias
    - Directory traversal (checks parent directories for matching files)
    """
    link_lower = link.lower()
    
    # Strip heading anchor if present (e.g., "notes#section" -> "notes")
    link_no_anchor = link_lower.split('#')[0]
    
    # Try as relative path first (handles [[projects/lloyd/analysis/foo]])
    if link_no_anchor in rel_paths:
        return True
    
    # Try as basename
    if link_no_anchor in basenames:
        return True
    
    # Try with just the filename (for links like "notes" that might be in a subdirectory)
    filename = os.path.basename(link_no_anchor)
    if filename in basenames:
        return True
    
    # Try case-insensitive path matching by checking all files
    # This handles cases where the link path differs in case from the actual file
    for filepath in all_files:
        rel_path = os.path.relpath(filepath, VAULT_ROOT).lower()
        if rel_path == link_no_anchor:
            return True
        # Also check without .md extension
        if rel_path.endswith('.md') and rel_path[:-3] == link_no_anchor:
            return True
    
    return False


def extract_wiki_links(content):
    """Extract wiki-links [[target]] from content.
    
    Handles [[target]], [[target|display]], [[target#heading]].
    Returns just the file target portion.
    Ignores wiki-links inside code blocks (```) and inline code (`).
    """
    # Remove code blocks first
    content_no_code = re.sub(r'```[\s\S]*?```', '', content)
    content_no_code = re.sub(r'`[^`]+`', '', content_no_code)
    
    # Match [[...]] content
    raw_matches = re.findall(r'\[\[([^\]]+)\]\]', content_no_code)
    targets = []
    for match in raw_matches:
        # Strip display text after |
        target = match.split('|')[0]
        # Strip heading after #
        target = target.split('#')[0]
        target = target.strip()
        if target and not target.startswith('http://') and not target.startswith('https://'):
            targets.append(target)
    return targets


def check_broken_links(existing_queue, basenames, rel_paths, all_files):
    """Find wiki-links that don't resolve to existing files.
    
    Excludes transcript/log directories that contain code syntax (not real wiki-links):
    - agents/main/sessions/ (session transcripts)
    """
    broken = []

    # Directories to exclude from broken-link scanning
    exclude_patterns = [
        'sessions/',
    ]
    
    for filepath in all_files:
        rel_path = os.path.relpath(filepath, VAULT_ROOT)
        
        # Skip transcript/log directories
        if any(pattern in rel_path for pattern in exclude_patterns):
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            continue
        
        links = extract_wiki_links(content)
        for link in links:
            if not resolve_wiki_link(link, basenames, rel_paths, all_files):
                item_id = f"broken-link-{rel_path}-{link}"[:200]
                
                existing = existing_queue.get(item_id)
                if existing and existing.get('status') in ('done', 'skipped'):
                    continue
                
                broken.append({
                    'id': item_id,
                    'type': 'BROKEN_LINK',
                    'priority': 'high',
                    'source_file': rel_path,
                    'target': link,
                    'status': 'pending',
                    'detail': f'Wiki-link [[{link}]] not found in vault'
                })
    
    return broken


def check_stale_facts(existing_queue):
    """Find facts with last_updated > 30 days old."""
    stale = []
    cutoff = datetime.now() - timedelta(days=30)
    
    if not os.path.exists(FACTS_DIR):
        return stale
    
    for entity_dir in os.listdir(FACTS_DIR):
        entity_path = os.path.join(FACTS_DIR, entity_dir)
        if not os.path.isdir(entity_path):
            continue
        
        for fact_file in glob.glob(os.path.join(entity_path, "*.md")):
            try:
                with open(fact_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                match = re.search(r'last_updated:\s*[\'"]([^\'"]+)[\'"]', content)
                if match:
                    date_str = match.group(1)
                    try:
                        last_updated = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        if last_updated.tzinfo:
                            last_updated = last_updated.replace(tzinfo=None)
                        
                        if last_updated < cutoff:
                            item_id = f"stale-fact-{entity_dir}-{os.path.basename(fact_file)}"
                            existing = existing_queue.get(item_id)
                            if existing and existing.get('status') in ('done', 'skipped'):
                                continue
                            
                            stale.append({
                                'id': item_id,
                                'type': 'STALE_FACT',
                                'priority': 'medium',
                                'source_file': os.path.relpath(fact_file, VAULT_ROOT),
                                'target': entity_dir,
                                'status': 'pending',
                                'detail': f'Fact not updated since {date_str} (>30 days)'
                            })
                    except ValueError:
                        pass
            except:
                continue
    
    return stale


def check_memory_hygiene(existing_queue):
    """Find missing file references in MEMORY.md."""
    missing = []
    
    if not os.path.exists(MEMORY_MD):
        return missing
    
    try:
        with open(MEMORY_MD, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return missing
    
    # Find backtick-wrapped paths that look like file paths
    # Must contain / and end with .md, or look like a vault-relative path
    pattern = r'`([^`*]+\.md)`'
    matches = re.findall(pattern, content)
    
    for ref in matches:
        # Skip glob patterns
        if '*' in ref or '?' in ref:
            continue
        
        # Skip paths with ~ prefix — resolve them
        check_ref = ref
        if check_ref.startswith('~/obsidian/'):
            check_ref = check_ref[len('~/obsidian/'):]
        
        full_path = os.path.join(VAULT_ROOT, check_ref)
        if not os.path.exists(full_path):
            item_id = f"memory-hygiene-{check_ref.replace('/', '-').replace('.md', '')}"
            existing = existing_queue.get(item_id)
            if existing and existing.get('status') in ('done', 'skipped'):
                continue
            
            missing.append({
                'id': item_id,
                'type': 'MEMORY_HYGIENE',
                'priority': 'medium',
                'source_file': 'lloyd/MEMORY.md',
                'target': check_ref,
                'status': 'pending',
                'detail': f'Referenced file `{ref}` not found in vault'
            })
    
    return missing


def build_inbound_links(all_files, basenames, rel_paths):
    """Build a map of which files have inbound wiki-links."""
    inbound = defaultdict(set)
    
    for filepath in all_files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            continue
        
        links = extract_wiki_links(content)
        source_rel = os.path.relpath(filepath, VAULT_ROOT)
        
        for link in links:
            link_lower = link.lower()
            # Resolve via path first
            if link_lower in rel_paths:
                target_rel = os.path.relpath(rel_paths[link_lower], VAULT_ROOT)
                inbound[target_rel].add(source_rel)
            # Then basename
            elif link_lower in basenames:
                for target_path in basenames[link_lower]:
                    target_rel = os.path.relpath(target_path, VAULT_ROOT)
                    inbound[target_rel].add(source_rel)
    
    return inbound


def build_relation_inbound(all_files):
    """Build a map of which files have inbound relations from frontmatter."""
    inbound = defaultdict(set)
    
    for filepath in all_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read(2000)
        except:
            continue
        
        # Extract frontmatter
        frontmatter_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
        if not frontmatter_match:
            continue
        
        frontmatter = frontmatter_match.group(1)
        
        # Check for relations block
        relations_match = re.search(r'relations:\s*\n((?:  .*\n)*)', frontmatter)
        if not relations_match:
            continue
        
        relations_block = relations_match.group(1)
        
        # Parse related-to and references arrays
        for section in ['related-to:', 'references:']:
            section_match = re.search(rf'{section}\s*\n((?:  - .*\n)*)', relations_block)
            if section_match:
                items_str = section_match.group(1)
                # Extract paths (lines starting with "  - ")
                paths = re.findall(r'^  - (.+?)$', items_str, re.MULTILINE)
                for path in paths:
                    path = path.strip()
                    # Normalize path (remove .md if present for matching)
                    if path.endswith('.md'):
                        path_no_ext = path[:-3]
                    else:
                        path_no_ext = path
                    
                    # Try to resolve to actual file
                    full_path = os.path.join(VAULT_ROOT, path)
                    if os.path.exists(full_path):
                        rel_target = os.path.relpath(full_path, VAULT_ROOT)
                        source_rel = os.path.relpath(filepath, VAULT_ROOT)
                        inbound[rel_target].add(source_rel)
                    
                    # Also try without .md extension in the path
                    if path.endswith('.md'):
                        full_path_no_ext = os.path.join(VAULT_ROOT, path[:-3])
                        if os.path.exists(full_path_no_ext):
                            rel_target = os.path.relpath(full_path_no_ext, VAULT_ROOT)
                            source_rel = os.path.relpath(filepath, VAULT_ROOT)
                            inbound[rel_target].add(source_rel)
    
    return inbound


def find_hub_pages(rel_path):
    """Find potential hub pages for a given file path.
    
    Hub pages are typically <directory>/<directory-name>.md
    e.g., for "projects/alfie/notes.md", hub pages could be:
    - projects/alfie/alfie.md
    - projects/projects.md
    
    Returns list of potential hub page paths to check.
    """
    parts = rel_path.split('/')
    hub_candidates = []
    
    # Build hub page candidates from directory hierarchy
    for i in range(len(parts) - 1):  # Exclude the file itself
        dir_path = '/'.join(parts[:i+1])
        dir_name = parts[i]
        hub_candidate = f"{dir_path}/{dir_name}.md"
        hub_candidates.append(hub_candidate)
    
    return hub_candidates


def check_orphan_files(existing_queue, all_files, basenames, rel_paths):
    """Find .md files with zero inbound wiki-links or relations.
    
    Before marking a file as orphaned, checks if it's reachable from any hub page.
    Hub pages are typically <directory>/<directory-name>.md (e.g., projects/alfie/alfie.md).
    
    Excludes:
    - sessions/ (auto-generated session transcripts)
    """
    orphans = []
    wiki_inbound = build_inbound_links(all_files, basenames, rel_paths)
    relation_inbound = build_relation_inbound(all_files)

    # Build a set of all file paths for quick lookup
    all_file_paths = set(os.path.relpath(f, VAULT_ROOT) for f in all_files)
    
    # Build hub page link graph - which files are linked from hub pages
    hub_page_links = defaultdict(set)
    hub_pages_to_check = []
    
    # Find all potential hub pages
    for filepath in all_files:
        rel_path = os.path.relpath(filepath, VAULT_ROOT)
        parts = rel_path.split('/')
        
        # A file is a hub page if its name matches its parent directory name
        if len(parts) >= 2:
            dir_name = parts[-2]  # Parent directory name
            file_name = parts[-1].replace('.md', '')  # File name without extension
            
            if dir_name == file_name:
                hub_pages_to_check.append(rel_path)
    
    # For each hub page, find what it links to (both wiki-links and plain text references)
    for hub_path in hub_pages_to_check:
        hub_filepath = os.path.join(VAULT_ROOT, hub_path)
        try:
            with open(hub_filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            continue
        
        # First, check for wiki-links
        links = extract_wiki_links(content)
        for link in links:
            link_lower = link.lower()
            # Resolve the link
            if link_lower in rel_paths:
                target_rel = os.path.relpath(rel_paths[link_lower], VAULT_ROOT)
                hub_page_links[target_rel].add(hub_path)
            elif link_lower in basenames:
                for target_path in basenames[link_lower]:
                    target_rel = os.path.relpath(target_path, VAULT_ROOT)
                    hub_page_links[target_rel].add(hub_path)
        
        # Also check for plain text filename references (e.g., "architecture.md — description")
        # This handles hub pages that use plain text lists instead of wiki-links
        filename_refs = re.findall(r'(\S+\.md)\s*[\—–-]', content)
        for ref in filename_refs:
            ref_lower = ref.lower()
            # Try to resolve
            if ref_lower in rel_paths:
                target_rel = os.path.relpath(rel_paths[ref_lower], VAULT_ROOT)
                hub_page_links[target_rel].add(hub_path)
            elif ref_lower in basenames:
                for target_path in basenames[ref_lower]:
                    target_rel = os.path.relpath(target_path, VAULT_ROOT)
                    hub_page_links[target_rel].add(hub_path)
            # Also try without .md extension
            ref_no_ext = ref_lower[:-3] if ref_lower.endswith('.md') else ref_lower
            if ref_no_ext in rel_paths:
                target_rel = os.path.relpath(rel_paths[ref_no_ext], VAULT_ROOT)
                hub_page_links[target_rel].add(hub_path)
            elif ref_no_ext in basenames:
                for target_path in basenames[ref_no_ext]:
                    target_rel = os.path.relpath(target_path, VAULT_ROOT)
                    hub_page_links[target_rel].add(hub_path)
    
    # Excluded directories from orphan scanning
    excluded_prefixes = (
        'sessions/',
    )
    
    for filepath in all_files:
        rel_path = os.path.relpath(filepath, VAULT_ROOT)
        
        # Skip excluded directories (auto-generated content)
        if any(rel_path.startswith(p) for p in excluded_prefixes):
            continue
        
        if '/SKILL.md' in rel_path:
            continue
        
        # Check frontmatter for type: facts
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                head = f.read(500)
            if re.search(r'type:\s*facts', head):
                continue
        except:
            continue
        
        # Check both wiki-links and relations
        has_wiki_inbound = rel_path in wiki_inbound and len(wiki_inbound[rel_path]) > 0
        has_relation_inbound = rel_path in relation_inbound and len(relation_inbound[rel_path]) > 0
        
        # Check if file is reachable from any hub page
        has_hub_inbound = rel_path in hub_page_links and len(hub_page_links[rel_path]) > 0
        
        if not has_wiki_inbound and not has_relation_inbound and not has_hub_inbound:
            item_id = f"orphan-file-{rel_path.replace('/', '-').replace('.md', '')}"[:200]
            existing = existing_queue.get(item_id)
            if existing and existing.get('status') in ('done', 'skipped'):
                continue
            
            orphans.append({
                'id': item_id,
                'type': 'ORPHAN_FILE',
                'priority': 'low',
                'source_file': rel_path,
                'target': None,
                'status': 'pending',
                'detail': 'File has zero inbound wiki-links, relations, or hub page references'
            })
    
    return orphans


def count_entity_references(entity_name, knowledge_files, projects_files):
    """Count how many documents in knowledge/ and projects/ mention an entity."""
    ref_count = 0
    entity_lower = entity_name.lower()
    
    # Search in knowledge/ and projects/ files
    target_files = knowledge_files + projects_files
    
    for filepath in target_files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()
            # Check if entity name appears (case-insensitive)
            if entity_lower in content:
                ref_count += 1
        except:
            continue
    
    return ref_count


def check_thin_profiles(existing_queue, knowledge_files, projects_files):
    """Find entities with fewer than 3 facts.
    
    Returns two lists:
    - THIN_PROFILE: entities with <3 facts and <2 references in knowledge/projects
    - ENRICH_THIN_PROFILE: entities with <3 facts but 2+ references (enrichment candidates)
    """
    thin = []
    enrich_thin = []
    
    if not os.path.exists(FACTS_DIR):
        return thin, enrich_thin
    
    for entity_dir in os.listdir(FACTS_DIR):
        entity_path = os.path.join(FACTS_DIR, entity_dir)
        if not os.path.isdir(entity_path):
            continue
        
        # Skip special files
        if entity_dir.startswith('.') or entity_dir == 'entity-registry.json':
            continue
        
        fact_count = 0
        for fact_file in glob.glob(os.path.join(entity_path, "*.md")):
            try:
                with open(fact_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                matches = re.findall(r'^\s*- fact:', content, re.MULTILINE)
                fact_count += len(matches)
            except:
                continue
        
        if fact_count < 3:
            # Count references in knowledge/ and projects/
            ref_count = count_entity_references(entity_dir, knowledge_files, projects_files)
            
            item_id = f"thin-profile-{entity_dir}"
            existing = existing_queue.get(item_id)
            if existing and existing.get('status') in ('done', 'skipped'):
                continue
            
            # Split based on reference count
            if ref_count >= 2:
                # Enrichment candidate
                enrich_thin.append({
                    'id': item_id,
                    'type': 'ENRICH_THIN_PROFILE',
                    'priority': 'medium',
                    'source_file': f'memory/facts/{entity_dir}/',
                    'target': entity_dir,
                    'status': 'pending',
                    'detail': f'Entity has only {fact_count} fact(s) but appears in {ref_count} document(s) in knowledge/projects'
                })
            else:
                # Regular thin profile
                thin.append({
                    'id': item_id,
                    'type': 'THIN_PROFILE',
                    'priority': 'low',
                    'source_file': f'memory/facts/{entity_dir}/',
                    'target': entity_dir,
                    'status': 'pending',
                    'detail': f'Entity has only {fact_count} fact(s) (target: 3+)'
                })
    
    return thin, enrich_thin


def check_stale_relations(existing_queue, all_files):
    """Find broken relation paths in frontmatter."""
    stale = []
    
    for filepath in all_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read(2000)
        except:
            continue
        
        # Extract frontmatter
        frontmatter_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
        if not frontmatter_match:
            continue
        
        frontmatter = frontmatter_match.group(1)
        
        # Check for relations block
        relations_match = re.search(r'relations:\s*\n((?:  .*\n)*)', frontmatter)
        if not relations_match:
            continue
        
        relations_block = relations_match.group(1)
        
        # Parse related-to and references arrays
        for section in ['related-to:', 'references:']:
            section_match = re.search(rf'{section}\s*\n((?:  - .*\n)*)', relations_block)
            if section_match:
                items_str = section_match.group(1)
                # Extract paths (lines starting with "  - ")
                paths = re.findall(r'^  - (.+?)$', items_str, re.MULTILINE)
                for path in paths:
                    path = path.strip()
                    
                    # Check if file exists (with and without .md)
                    full_path = os.path.join(VAULT_ROOT, path)
                    full_path_no_ext = os.path.join(VAULT_ROOT, path[:-3]) if path.endswith('.md') else None
                    
                    file_exists = os.path.exists(full_path)
                    if not file_exists and full_path_no_ext:
                        file_exists = os.path.exists(full_path_no_ext)
                    
                    if not file_exists:
                        rel_source = os.path.relpath(filepath, VAULT_ROOT)
                        item_id = f"stale-relation-{rel_source.replace('/', '-')}-{path.replace('/', '-')}"[:200]
                        
                        existing = existing_queue.get(item_id)
                        if existing and existing.get('status') in ('done', 'skipped'):
                            continue
                        
                        stale.append({
                            'id': item_id,
                            'type': 'STALE_RELATION',
                            'priority': 'medium',
                            'source_file': rel_source,
                            'target': path,
                            'status': 'pending',
                            'detail': f'Relation target "{path}" not found in vault'
                        })
    
    return stale


def check_enrich_stubs(existing_queue, knowledge_files, projects_files):
    """Find files in knowledge/ or projects/ with <200 chars of actual content."""
    stubs = []
    
    target_files = knowledge_files + projects_files
    
    for filepath in target_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            continue
        
        # Strip frontmatter
        content_without_fm = re.sub(r'^---\n.*?\n---\n?', '', content, flags=re.DOTALL)
        char_count = len(content_without_fm)
        
        if char_count < 200:
            rel_path = os.path.relpath(filepath, VAULT_ROOT)
            item_id = f"enrich-stub-{rel_path.replace('/', '-').replace('.md', '')}"[:200]
            
            existing = existing_queue.get(item_id)
            if existing and existing.get('status') in ('done', 'skipped'):
                continue
            
            stubs.append({
                'id': item_id,
                'type': 'ENRICH_STUB',
                'priority': 'medium',
                'source_file': rel_path,
                'target': None,
                'status': 'pending',
                'detail': f'File has only {char_count} characters of content (target: 200+)'
            })
    
    return stubs


def check_enrich_stale_topics(existing_queue, knowledge_files):
    """Find knowledge/ files not updated in >7 days with 2+ inbound links/relations.
    
    Uses filesystem mtime for staleness detection (not frontmatter date) to avoid
    false positives when enrichment processes modify files without updating frontmatter.
    """
    stale_topics = []
    cutoff = datetime.now() - timedelta(days=7)
    
    # Build inbound counts using existing functions
    all_files = knowledge_files  # Only check knowledge/ for inbound
    basenames, rel_paths = build_file_index(all_files)
    wiki_inbound = build_inbound_links(all_files, basenames, rel_paths)
    relation_inbound = build_relation_inbound(all_files)
    
    for filepath in knowledge_files:
        # Use filesystem mtime directly — not frontmatter
        # This avoids false positives when enrichment modifies files without updating frontmatter
        try:
            mtime = os.path.getmtime(filepath)
            last_updated = datetime.fromtimestamp(mtime)
        except:
            continue
        
        # Check if stale
        if last_updated and last_updated < cutoff:
            # Count inbound
            rel_path = os.path.relpath(filepath, VAULT_ROOT)
            wiki_count = len(wiki_inbound.get(rel_path, set()))
            relation_count = len(relation_inbound.get(rel_path, set()))
            total_inbound = wiki_count + relation_count
            
            if total_inbound >= 2:
                item_id = f"enrich-stale-topic-{rel_path.replace('/', '-').replace('.md', '')}"[:200]
                
                existing = existing_queue.get(item_id)
                if existing and existing.get('status') in ('done', 'skipped'):
                    continue
                
                stale_topics.append({
                    'id': item_id,
                    'type': 'ENRICH_STALE_TOPIC',
                    'priority': 'low',
                    'source_file': rel_path,
                    'target': None,
                    'status': 'pending',
                    'detail': f'File not updated since {last_updated.date()} ({total_inbound} inbound links/relations)'
                })
    
    return stale_topics


def check_missing_frontmatter(existing_queue, all_files):
    """Find files in knowledge/, projects/, personal/, work/ with missing or incorrect frontmatter.
    
    Checks for:
    - Missing frontmatter entirely (no --- delimiters at start)
    - Missing required segment field
    - Missing required type field
    - segment value doesn't match top-level directory
    
    Excludes: memory/, agents/, skills/
    """
    issues = []
    target_dirs = ('knowledge/', 'projects/', 'personal/', 'work/')
    exclude_dirs = ('memory/', 'agents/', 'skills/')
    
    for filepath in all_files:
        rel_path = os.path.relpath(filepath, VAULT_ROOT)
        
        # Skip excluded directories
        if any(rel_path.startswith(p) for p in exclude_dirs):
            continue
        
        # Only check target directories
        if not any(rel_path.startswith(p) for p in target_dirs):
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            continue
        
        # Check for frontmatter
        frontmatter_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
        
        if not frontmatter_match:
            # No frontmatter at all
            item_id = f"missing-frontmatter-{rel_path.replace('/', '-').replace('.md', '')}"[:200]
            existing = existing_queue.get(item_id)
            if existing and existing.get('status') in ('done', 'skipped'):
                continue
            
            issues.append({
                'id': item_id,
                'type': 'MISSING_FRONTMATTER',
                'priority': 'medium',
                'source_file': rel_path,
                'target': None,
                'status': 'pending',
                'detail': 'Missing frontmatter entirely (no --- delimiters)'
            })
            continue
        
        # Frontmatter exists, check fields
        frontmatter = frontmatter_match.group(1)
        
        # Determine correct segment from directory
        first_component = rel_path.split('/')[0] + '/'
        
        issues_found = []
        
        # Check for segment field (only verify it exists, don't validate value)
        segment_match = re.search(r'^segment:\s*(.+)$', frontmatter, re.MULTILINE)
        if not segment_match:
            issues_found.append('missing segment field')
        # Note: segment value validation removed - segment field value is the agent/segment name,
        # not the directory name, so comparing to first_component causes false positives
        
        # Check for type field
        type_match = re.search(r'^type:\s*(.+)$', frontmatter, re.MULTILINE)
        if not type_match:
            issues_found.append('missing type field')
        
        if issues_found:
            item_id = f"missing-frontmatter-{rel_path.replace('/', '-').replace('.md', '')}"[:200]
            existing = existing_queue.get(item_id)
            if existing and existing.get('status') in ('done', 'skipped'):
                continue
            
            issues.append({
                'id': item_id,
                'type': 'MISSING_FRONTMATTER',
                'priority': 'medium',
                'source_file': rel_path,
                'target': None,
                'status': 'pending',
                'detail': '; '.join(issues_found)
            })
    
    return issues


def check_large_docs(existing_queue, all_files):
    """Find .md files over 300 lines that may need splitting.
    
    Excludes:
    - Files matching *-log.jsonl or *-log.md
    - MEMORY.md files (curated, expected to grow)
    """
    large_docs = []

    for filepath in all_files:
        rel_path = os.path.relpath(filepath, VAULT_ROOT)

        # Skip excluded patterns
        if rel_path.endswith('-log.jsonl') or rel_path.endswith('-log.md'):
            continue
        if rel_path.endswith('MEMORY.md'):
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
        except:
            continue
        
        if line_count > 300:
            item_id = f"large-doc-{rel_path.replace('/', '-').replace('.md', '')}"[:200]
            existing = existing_queue.get(item_id)
            if existing and existing.get('status') in ('done', 'skipped'):
                continue
            
            large_docs.append({
                'id': item_id,
                'type': 'LARGE_DOC',
                'priority': 'low',
                'source_file': rel_path,
                'target': None,
                'status': 'pending',
                'detail': f'File has {line_count} lines (>300)'
            })
    
    return large_docs





def compute_health_score(items, all_files, basenames, rel_paths):
    """Compute health score based on actual vault metrics."""
    type_counts = defaultdict(int)
    for item in items:
        if item['status'] == 'pending':
            type_counts[item['type']] += 1
    
    # Count actual totals
    total_links = 0
    for filepath in all_files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            total_links += len(extract_wiki_links(content))
        except:
            continue
    
    total_facts = 0
    total_entities = 0
    if os.path.exists(FACTS_DIR):
        for entity_dir in os.listdir(FACTS_DIR):
            entity_path = os.path.join(FACTS_DIR, entity_dir)
            if not os.path.isdir(entity_path):
                continue
            total_entities += 1
            for fact_file in glob.glob(os.path.join(entity_path, "*.md")):
                try:
                    with open(fact_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    total_facts += len(re.findall(r'^\s*- fact:', content, re.MULTILINE))
                except:
                    continue
    
    # Count non-excluded files for orphan score
    excluded_prefixes = ('memory/', 'agents/', 'skills/')
    eligible_files = sum(1 for f in all_files 
                        if not any(os.path.relpath(f, VAULT_ROOT).startswith(p) for p in excluded_prefixes))
    
    # Read MEMORY.md ref count
    total_refs = 0
    if os.path.exists(MEMORY_MD):
        try:
            with open(MEMORY_MD, 'r', encoding='utf-8') as f:
                content = f.read()
            refs = re.findall(r'`([^`*]+\.md)`', content)
            total_refs = sum(1 for r in refs if '*' not in r and '?' not in r)
        except:
            pass
    
    # Collect unique tags
    all_tags = set()
    for filepath in all_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                head = f.read(1000)
            match = re.search(r'tags:\s*\[(.*?)\]', head, re.DOTALL)
            if match:
                tags = re.findall(r'["\']?([a-zA-Z0-9_-]+)["\']?', match.group(1))
                all_tags.update(t.lower() for t in tags if t)
        except:
            continue
    
    dimensions = {}
    weights = {
        'broken_links': ('BROKEN_LINK', total_links or 1, 0.15),
        'stale_facts': ('STALE_FACT', total_facts or 1, 0.15),
        'memory_hygiene': ('MEMORY_HYGIENE', total_refs or 1, 0.10),
        'orphan_files': ('ORPHAN_FILE', eligible_files or 1, 0.10),
        'thin_profiles': ('THIN_PROFILE', total_entities or 1, 0.05),
        'stale_relations': ('STALE_RELATION', len(all_files) or 1, 0.10),
        'enrich_thin_profile': ('ENRICH_THIN_PROFILE', total_entities or 1, 0.05),
        'enrich_stale_topic': ('ENRICH_STALE_TOPIC', 100, 0.05),  # Approximate
        'enrich_stub': ('ENRICH_STUB', 100, 0.05),  # Approximate
        'missing_frontmatter': ('MISSING_FRONTMATTER', eligible_files or 1, 0.10),
        'large_doc': ('LARGE_DOC', 100, 0.10),  # Approximate
    }
    
    overall = 0
    for dim_name, (type_key, total, weight) in weights.items():
        count = type_counts.get(type_key, 0)
        score = max(0, (1 - count / total) * 100)
        dimensions[dim_name] = {
            'count': count,
            'total': total,
            'score': round(score, 1)
        }
        overall += score * weight
    
    return {
        'computed_at': datetime.now().isoformat(),
        'overall': round(overall, 1),
        'dimensions': dimensions
    }


def main():
    """Run the survey and produce queue JSON."""
    import time as _t
    print("Groundskeeper Survey Starting...")

    _log = lambda m: print(f"  {_t.strftime('%H:%M:%S')} {m}", flush=True)

    existing_queue = load_existing_queue()
    all_files = get_all_md_files()
    _log(f"Files indexed: {len(all_files)}")
    basenames, rel_paths = build_file_index(all_files)

    # Filter files for enrichment checks
    knowledge_files = [f for f in all_files if f.startswith(os.path.join(VAULT_ROOT, 'knowledge/'))]
    projects_files = [f for f in all_files if f.startswith(os.path.join(VAULT_ROOT, 'projects/'))]

    all_items = []

    t0 = _t.time()
    print("  Scanning broken links...")
    items = check_broken_links(existing_queue, basenames, rel_paths, all_files)
    all_items.extend(items)
    _log(f"broken_links: {len(items)} items in {_t.time()-t0:.1f}s")

    t0 = _t.time()
    print("  Scanning stale facts...")
    items = check_stale_facts(existing_queue)
    all_items.extend(items)
    _log(f"stale_facts: {len(items)} items in {_t.time()-t0:.1f}s")

    t0 = _t.time()
    print("  Scanning memory hygiene...")
    items = check_memory_hygiene(existing_queue)
    all_items.extend(items)
    _log(f"memory_hygiene: {len(items)} items in {_t.time()-t0:.1f}s")

    t0 = _t.time()
    print("  Scanning orphan files...")
    items = check_orphan_files(existing_queue, all_files, basenames, rel_paths)
    all_items.extend(items)
    _log(f"orphan_files: {len(items)} items in {_t.time()-t0:.1f}s")

    t0 = _t.time()
    print("  Scanning thin profiles...")
    thin_profiles, enrich_thin_profiles = check_thin_profiles(existing_queue, knowledge_files, projects_files)
    all_items.extend(thin_profiles)
    all_items.extend(enrich_thin_profiles)
    _log(f"thin_profiles: {len(thin_profiles)+len(enrich_thin_profiles)} items in {_t.time()-t0:.1f}s")
    
    print("  Scanning stale relations...")
    all_items.extend(check_stale_relations(existing_queue, all_files))
    
    print("  Scanning enrichment stubs...")
    all_items.extend(check_enrich_stubs(existing_queue, knowledge_files, projects_files))
    
    print("  Scanning enrichment stale topics...")
    all_items.extend(check_enrich_stale_topics(existing_queue, knowledge_files))
    
    print("  Scanning missing frontmatter...")
    all_items.extend(check_missing_frontmatter(existing_queue, all_files))
    
    print("  Scanning large documents...")
    all_items.extend(check_large_docs(existing_queue, all_files))
    
    health_score = compute_health_score(all_items, all_files, basenames, rel_paths)
    
    output = {
        'generated_at': datetime.now().isoformat(),
        'items': all_items,
        'health_score': health_score
    }
    
    # Queue size verification guards to prevent partial writes
    # Read expected size before writing
    expected_size = len(all_items)
    
    # Write to temp file first, then atomically rename
    temp_output = QUEUE_OUTPUT + '.tmp'
    try:
        with open(temp_output, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Verify written size matches expected
        with open(temp_output, 'r') as f:
            verify_data = json.load(f)
        actual_size = len(verify_data.get('items', []))
        
        if actual_size != expected_size:
            print(f"WARNING: Queue size mismatch! Expected {expected_size}, got {actual_size}")
            print("Rolling back to previous queue if available...")
            if os.path.exists(QUEUE_OUTPUT):
                # Keep old file, delete temp
                os.remove(temp_output)
            raise RuntimeError(f"Queue corruption detected: size mismatch {expected_size} != {actual_size}")
        
        # Atomic rename
        os.rename(temp_output, QUEUE_OUTPUT)
        
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_output):
            os.remove(temp_output)
        raise e
    
    # Print summary
    type_counts = defaultdict(int)
    for item in all_items:
        type_counts[item['type']] += 1
    
    print(f"\nSurvey complete!")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")
    print(f"  Total items: {len(all_items)}")
    print(f"  Health score: {health_score['overall']}")
    print(f"  Output: {QUEUE_OUTPUT}")


if __name__ == '__main__':
    main()
