# Nightly Reflection - Signal Report
## 2026-06-14 PST

**Signal:** negative
**Category:** scheduling
**Severity:** high
**Source:** daily note 2026-06-14
**Description:** Nightly reflection tasks running at 5pm instead of 2am (local time). 23 failed runs in 10 days. Root cause: schedule misalignment - cron jobs triggering at wrong time.
**Recommendation:** Fix cron schedule config in autonomy task definitions.

---

**Signal:** negative
**Category:** infrastructure
**Severity:** high
**Source:** vault-maintenance 2026-06-12
**Description:** `semantic_relationships.py` extraction timing out. LLM endpoint on port 8096 not running. Ollama is on 11434, Open WebUI on 8091.
**Recommendation:** Fix LLM endpoint config or route through working port 8091.

---

**Signal:** negative
**Category:** data-quality
**Severity:** medium
**Source:** vault-maintenance 2026-06-12, 13, 14
**Description:** 980 docs missing `segment:` field. Same issue recurring across 3 days of maintenance.
**Recommendation:** Batch fix missing segments during next maintenance run.

---

**Signal:** negative
**Category:** data-quality
**Severity:** medium
**Source:** vault-maintenance 2026-06-12, 13, 14
**Description:** Knowledge graph has duplicate facts from multiple extraction runs. Karpathy: 175 facts with 25 contradictions. vLLM: 195 facts. GR00T: 25 contradictions.
**Recommendation:** Implement fact deduplication before insertion.

---

**Signal:** negative
**Category:** data-quality
**Severity:** low
**Source:** vault-maintenance 2026-06-12, 13, 14
**Description:** 1,840 tags used only once (78.6%). 87.4% of 2,340 tags used 1-2 times. Tag fragmentation makes indexing less effective.
**Recommendation:** Consolidate tags, merge near-duplicates.

---

**Signal:** negative
**Category:** data-quality
**Severity:** medium
**Source:** vault-maintenance 2026-06-13, 14
**Description:** 772 stale wiki-links (92% of all wiki links in knowledge/ and memory/ directories).
**Recommendation:** Clean up broken links during maintenance.

---

**Signal:** negative
**Category:** data-quality
**Severity:** low
**Source:** vault-maintenance 2026-06-13
**Description:** GR00T entity has facts about Alfie's sensors mixed in. Entity cross-pollution between unrelated entities.
**Recommendation:** Review entity extraction logic to prevent cross-pollution.

---

**Signal:** positive
**Category:** workflow
**Severity:** medium
**Source:** vault-maintenance 2026-06-13, 14
**Description:** Full pipeline extraction works well when infrastructure is available. 2 files processed, 96 facts, 27,614 relationships, 32,567 index edges.
**Recommendation:** Preserve pipeline configuration.

---

**Signal:** positive
**Category:** output-quality
**Severity:** medium
**Source:** daily note 2026-06-11
**Description:** Deep research outputs are comprehensive and well-structured. AI Agent Arena report: 2,863 tokens with clear sections.
**Recommendation:** Maintain current research pipeline.

---

## Summary
- **Negative signals:** 7 (3 high, 2 medium, 2 low)
- **Positive signals:** 2
- **Recurring issues:** Scheduling (23 failed runs), LLM endpoints, data quality (segments, duplicates, stale links)
- **Key pattern:** Infrastructure issues (scheduling, LLM endpoints) blocking effective nightly operations
- **Action items:** Fix cron schedules, route LLM through working port, batch-fix missing segments
