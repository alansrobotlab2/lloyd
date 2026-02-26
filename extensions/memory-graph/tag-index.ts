/**
 * tag-index.ts — Tag-based document index with IDF scoring and co-occurrence.
 *
 * Replaces the wiki-link graph with a tag-oriented index that supports:
 *  - Search by tags (AND/OR mode) with IDF-weighted ranking
 *  - Tag co-occurrence discovery
 *  - Document similarity via Jaccard on tag sets
 *  - Topic resolution (stem, title, tag, folder matching)
 */

import type { DocMeta, ScanResult } from "./scanner.js";

// ── Types ──────────────────────────────────────────────────────────────

export interface TagIndexStats {
  docCount: number;
  tagCount: number;
  topTags: Array<{ tag: string; docFreq: number; idf: number }>;
  topDocsByImportance: DocMeta[];
  hubPages: DocMeta[];
  typeDistribution: Record<string, number>;
  builtAt: number;
}

// ── Tag Index ──────────────────────────────────────────────────────────

export class TagIndex {
  // Core indexes
  docs = new Map<string, DocMeta>();
  tagToDocs = new Map<string, Set<string>>();
  stemIndex = new Map<string, string[]>();

  // Derived indexes
  tagCooccurrence = new Map<string, Map<string, number>>();
  tagIdf = new Map<string, number>();
  builtAt = 0;

  get docCount(): number { return this.docs.size; }
  get tagCount(): number { return this.tagToDocs.size; }

  /**
   * Build the tag index from a scan result.
   */
  static fromScan(scan: ScanResult): TagIndex {
    const idx = new TagIndex();
    idx.builtAt = Date.now();
    idx.stemIndex = scan.stemIndex;

    // Populate docs
    for (const doc of scan.docs) {
      idx.docs.set(doc.path, doc);
    }

    // Populate tag-to-docs (use Sets for efficient intersection/union)
    for (const [tag, paths] of scan.tagIndex) {
      idx.tagToDocs.set(tag, new Set(paths));
    }

    // Compute IDF for each tag: log(N / df)
    const N = scan.docs.length;
    if (N > 0) {
      for (const [tag, docSet] of idx.tagToDocs) {
        idx.tagIdf.set(tag, Math.log(N / docSet.size));
      }
    }

    // Compute tag co-occurrence
    for (const doc of scan.docs) {
      const tags = doc.tags;
      for (let i = 0; i < tags.length; i++) {
        for (let j = i + 1; j < tags.length; j++) {
          const a = tags[i];
          const b = tags[j];
          // Increment both directions
          if (!idx.tagCooccurrence.has(a)) idx.tagCooccurrence.set(a, new Map());
          if (!idx.tagCooccurrence.has(b)) idx.tagCooccurrence.set(b, new Map());
          const mapA = idx.tagCooccurrence.get(a)!;
          const mapB = idx.tagCooccurrence.get(b)!;
          mapA.set(b, (mapA.get(b) ?? 0) + 1);
          mapB.set(a, (mapB.get(a) ?? 0) + 1);
        }
      }
    }

    // Compute importance scores
    for (const doc of scan.docs) {
      doc.importance = computeImportance(doc, idx.tagIdf);
    }

    return idx;
  }

  // ── Search by Tags ─────────────────────────────────────────────────

  /**
   * Find documents matching the given tags.
   * mode "or": docs with ANY of the tags (union), ranked by IDF-weighted overlap.
   * mode "and": docs with ALL of the tags (intersection), ranked by importance.
   */
  searchByTags(
    tags: string[],
    mode: "and" | "or" = "or",
    typeFilter?: string,
    limit = 10,
  ): DocMeta[] {
    if (tags.length === 0) return [];

    // Normalize tags to lowercase for matching
    const normalizedTags = tags.map((t) => t.toLowerCase().replace(/^#/, ""));

    // Resolve each search tag to actual tag names (case-insensitive)
    const resolvedTags: string[] = [];
    for (const searchTag of normalizedTags) {
      for (const [actualTag] of this.tagToDocs) {
        if (actualTag.toLowerCase() === searchTag) {
          resolvedTags.push(actualTag);
          break;
        }
      }
    }

    if (resolvedTags.length === 0) return [];

    let matchingPaths: Set<string>;

    if (mode === "and") {
      // Intersection: start with first tag's docs, intersect with each subsequent
      const sets = resolvedTags
        .map((t) => this.tagToDocs.get(t))
        .filter((s): s is Set<string> => s != null);
      if (sets.length === 0) return [];

      matchingPaths = new Set(sets[0]);
      for (let i = 1; i < sets.length; i++) {
        for (const path of matchingPaths) {
          if (!sets[i].has(path)) matchingPaths.delete(path);
        }
      }
    } else {
      // Union: all docs that have any of the tags
      matchingPaths = new Set<string>();
      for (const tag of resolvedTags) {
        const docSet = this.tagToDocs.get(tag);
        if (docSet) {
          for (const path of docSet) matchingPaths.add(path);
        }
      }
    }

    // Collect docs and compute relevance scores
    const scored: Array<{ doc: DocMeta; relevance: number }> = [];
    const resolvedTagSet = new Set(resolvedTags);

    for (const path of matchingPaths) {
      const doc = this.docs.get(path);
      if (!doc) continue;
      if (typeFilter && typeFilter !== "any" && doc.type !== typeFilter) continue;

      // Relevance = sum of IDF for each matching tag
      let relevance = 0;
      let matchCount = 0;
      for (const tag of doc.tags) {
        if (resolvedTagSet.has(tag)) {
          relevance += this.tagIdf.get(tag) ?? 0;
          matchCount++;
        }
      }
      // Boost by match count (more matched tags = better)
      relevance += matchCount * 0.5;

      scored.push({ doc, relevance });
    }

    // Sort by relevance descending, then importance as tiebreaker
    scored.sort((a, b) => b.relevance - a.relevance || b.doc.importance - a.doc.importance);

    return scored.slice(0, Math.min(limit, 25)).map((s) => s.doc);
  }

  // ── Related Tags ───────────────────────────────────────────────────

  /**
   * Find tags that co-occur with the given tag, ranked by frequency.
   */
  getRelatedTags(
    tag: string,
    limit = 15,
  ): Array<{ tag: string; count: number; idf: number; docFreq: number }> {
    // Case-insensitive tag lookup
    let actualTag: string | null = null;
    for (const [t] of this.tagToDocs) {
      if (t.toLowerCase() === tag.toLowerCase()) {
        actualTag = t;
        break;
      }
    }
    if (!actualTag) return [];

    const coMap = this.tagCooccurrence.get(actualTag);
    if (!coMap) return [];

    const results: Array<{ tag: string; count: number; idf: number; docFreq: number }> = [];
    for (const [coTag, count] of coMap) {
      results.push({
        tag: coTag,
        count,
        idf: this.tagIdf.get(coTag) ?? 0,
        docFreq: this.tagToDocs.get(coTag)?.size ?? 0,
      });
    }

    // Sort by co-occurrence count descending
    results.sort((a, b) => b.count - a.count);
    return results.slice(0, limit);
  }

  // ── Related Documents ──────────────────────────────────────────────

  /**
   * Find documents similar to the given one, by IDF-weighted Jaccard
   * similarity on tag sets.
   */
  getRelatedDocs(
    path: string,
    limit = 10,
  ): Array<{ doc: DocMeta; similarity: number }> {
    const sourceDoc = this.docs.get(path);
    if (!sourceDoc || sourceDoc.tags.length === 0) return [];

    const sourceTags = new Set(sourceDoc.tags);
    const results: Array<{ doc: DocMeta; similarity: number }> = [];

    for (const [docPath, doc] of this.docs) {
      if (docPath === path || doc.tags.length === 0) continue;

      const docTags = new Set(doc.tags);

      // Compute IDF-weighted Jaccard
      let sharedWeight = 0;
      let unionWeight = 0;

      // Union of all tags from both docs
      const allTags = new Set([...sourceTags, ...docTags]);
      for (const tag of allTags) {
        const idf = this.tagIdf.get(tag) ?? 0;
        if (sourceTags.has(tag) && docTags.has(tag)) {
          sharedWeight += idf;
        }
        unionWeight += idf;
      }

      if (unionWeight === 0) continue;
      const similarity = sharedWeight / unionWeight;
      if (similarity > 0) {
        results.push({ doc, similarity });
      }
    }

    results.sort((a, b) => b.similarity - a.similarity);
    return results.slice(0, limit);
  }

  // ── Topic Resolution ───────────────────────────────────────────────

  /**
   * Map a user query string to document(s).
   * Cascade: exact path → exact stem → case-insensitive stem →
   *          title substring → folder match → tag match.
   */
  resolveTopic(query: string): DocMeta[] {
    const q = query.trim();
    if (!q) return [];

    // 1. Exact path match
    const byPath = this.docs.get(q) ?? this.docs.get(q + ".md");
    if (byPath) return [byPath];

    // 2. Exact stem match
    const stemPaths = this.stemIndex.get(q);
    if (stemPaths) {
      const results = stemPaths
        .map((p) => this.docs.get(p))
        .filter((n): n is DocMeta => n != null);
      if (results.length > 0) return results;
    }

    const lowerQ = q.toLowerCase();

    // 3. Case-insensitive stem match
    for (const [stem, paths] of this.stemIndex) {
      if (stem.toLowerCase() === lowerQ) {
        const results = paths
          .map((p) => this.docs.get(p))
          .filter((n): n is DocMeta => n != null);
        if (results.length > 0) return results;
      }
    }

    // 4. Title substring match
    const titleMatches: DocMeta[] = [];
    for (const doc of this.docs.values()) {
      if (doc.title.toLowerCase().includes(lowerQ)) {
        titleMatches.push(doc);
      }
    }
    if (titleMatches.length > 0) {
      return titleMatches.sort((a, b) => b.importance - a.importance);
    }

    // 5. Folder match
    const folderMatches: DocMeta[] = [];
    for (const doc of this.docs.values()) {
      if (doc.folder.toLowerCase().includes(lowerQ)) {
        folderMatches.push(doc);
      }
    }
    if (folderMatches.length > 0) {
      return folderMatches.sort((a, b) => b.importance - a.importance);
    }

    // 6. Tag match
    const tagMatches: DocMeta[] = [];
    for (const doc of this.docs.values()) {
      if (doc.tags.some((t) => t.toLowerCase() === lowerQ)) {
        tagMatches.push(doc);
      }
    }
    if (tagMatches.length > 0) {
      return tagMatches.sort((a, b) => b.importance - a.importance);
    }

    return [];
  }

  // ── Fuzzy Tag Suggestion ───────────────────────────────────────────

  /**
   * When a searched tag doesn't exist, suggest close matches.
   */
  suggestTags(query: string, limit = 5): string[] {
    const q = query.toLowerCase();
    const scored: Array<{ tag: string; score: number }> = [];

    for (const tag of this.tagToDocs.keys()) {
      const t = tag.toLowerCase();
      // Prefix match
      if (t.startsWith(q) || q.startsWith(t)) {
        scored.push({ tag, score: 3 });
      // Contains match
      } else if (t.includes(q) || q.includes(t)) {
        scored.push({ tag, score: 2 });
      // Hyphen-stripped match
      } else if (t.replace(/-/g, "").includes(q.replace(/-/g, ""))) {
        scored.push({ tag, score: 1 });
      }
    }

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit).map((s) => s.tag);
  }

  // ── Stats ──────────────────────────────────────────────────────────

  getStats(): TagIndexStats {
    // Top tags by document frequency
    const topTags = Array.from(this.tagToDocs.entries())
      .map(([tag, docs]) => ({
        tag,
        docFreq: docs.size,
        idf: this.tagIdf.get(tag) ?? 0,
      }))
      .sort((a, b) => b.docFreq - a.docFreq)
      .slice(0, 20);

    // Top docs by importance
    const topDocsByImportance = Array.from(this.docs.values())
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 10);

    // Hub pages
    const hubPages = Array.from(this.docs.values())
      .filter((d) => d.type === "hub")
      .sort((a, b) => b.importance - a.importance);

    // Type distribution
    const typeDistribution: Record<string, number> = {};
    for (const doc of this.docs.values()) {
      const type = doc.type || "(none)";
      typeDistribution[type] = (typeDistribution[type] ?? 0) + 1;
    }

    return {
      docCount: this.docCount,
      tagCount: this.tagCount,
      topTags,
      topDocsByImportance,
      hubPages,
      typeDistribution,
      builtAt: this.builtAt,
    };
  }
}

// ── Importance Scoring ───────────────────────────────────────────────

function computeImportance(doc: DocMeta, tagIdf: Map<string, number>): number {
  // Type boost: hub pages are index pages, inherently more central
  const typeBoost =
    doc.type === "hub" ? 2.0 :
    doc.type === "project-notes" ? 1.0 :
    doc.type === "talk" ? 0.9 :
    doc.type === "work-notes" ? 0.8 :
    doc.type === "notes" ? 0.7 :
    0.5;

  // Tag connectivity: more tags = more cross-topic connections (diminishing returns)
  const connectivityScore = Math.sqrt(doc.tags.length);

  // Tag specificity: average IDF of the doc's tags
  const avgIdf = doc.tags.length > 0
    ? doc.tags.reduce((sum, t) => sum + (tagIdf.get(t) ?? 0), 0) / doc.tags.length
    : 0;

  // Summary completeness bonus
  const summaryBonus = doc.summary ? 0.3 : 0;

  return typeBoost + connectivityScore * 0.3 + avgIdf * 0.2 + summaryBonus;
}
