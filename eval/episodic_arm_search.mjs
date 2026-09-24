// The retrieval half of the episodic arm (#675): qmd's own hybrid search over a
// frozen copy of the exported session transcripts, in a scratch index.
//
// Why a scratch index and not the live daemon: the live `sessions` collection
// holds 6 documents since the 2026-09-22 data-home wipe (656 before it), so the
// daemon cannot answer the question this arm asks. The corpus is rebuilt from
// the last index that held it and re-embedded here with the SAME embed and
// rerank models the daemon names in ~/.config/qmd/index.yml; the request shape
// mirrors the recall doc leg's (lex + vec, lexMode "or", cross-encoder on).
//
// Usage (the Python side, eval/episodic_arm.py, drives this):
//   node eval/episodic_arm_search.mjs <spec.json> <out.json>
// spec: {qmd_dist, db_path, corpus_dir, models: {embed, rerank, generate},
//        limit, candidate_limit, queries: [{id, lex, vec}]}
// out:  {embed: {...}, warmup_ms, rows: [{id, latency_ms, hits: [...]}]}
import { readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const [specPath, outPath] = process.argv.slice(2);
if (!specPath || !outPath) {
  console.error("usage: episodic_arm_search.mjs <spec.json> <out.json>");
  process.exit(2);
}
const spec = JSON.parse(readFileSync(specPath, "utf8"));
const { createStore } = await import(pathToFileURL(spec.qmd_dist).href);

const store = await createStore({
  dbPath: spec.db_path,
  config: {
    collections: { sessions: { path: spec.corpus_dir, pattern: "**/*.md" } },
    models: spec.models,
  },
});
const out = { rows: [] };
try {
  const t0 = performance.now();
  out.update = await store.update({ collections: ["sessions"] });
  out.embed = await store.embed({ collection: "sessions" });
  out.index_ms = Math.round(performance.now() - t0);

  const request = (q) => ({
    queries: [{ type: "lex", query: q.lex }, { type: "vec", query: q.vec }],
    collections: ["sessions"],
    limit: spec.limit,
    candidateLimit: spec.candidate_limit,
    lexMode: "or",
    rerank: true,
  });
  // One throwaway query loads the embed and rerank models, so no scored query
  // pays the model load. Its text is never one of the scored queries: the store
  // caches query embeddings and a repeat would be answered from that cache.
  const w0 = performance.now();
  await store.search(request({ lex: "warm up the models", vec: "warm up the models" }));
  out.warmup_ms = Math.round(performance.now() - w0);

  for (const q of spec.queries) {
    const s0 = performance.now();
    let hits = [];
    let error = null;
    try {
      const res = await store.search(request(q));
      hits = res.map((r) => ({
        path: r.displayPath,
        file: r.file,
        score: r.score,
        best_chunk_pos: r.bestChunkPos,
        best_chunk_len: (r.bestChunk || "").length,
      }));
    } catch (e) {
      error = String(e);
    }
    out.rows.push({ id: q.id, latency_ms: Math.round(performance.now() - s0), hits, error });
  }
} finally {
  await store.close();
}
writeFileSync(outPath, JSON.stringify(out, null, 1));
