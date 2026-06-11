# Lloyd Architecture Review — 2026-06-11

Scope: full-codebase review (backend core, harness, agent_mcp aggregator, frontend, config/ops/testing), calibrated for a **single-user, fully-local system with a nightly self-improvement loop**. Tier 1 + Tier 2 were implemented on `feature/arch-review-tier1-tier2`; Tier 3 remains as a backlog for the nightly loop. Generic enterprise concerns (auth, scaling, multi-tenancy) are deliberately out of scope.

## Overall assessment

The architecture is sound. Strengths worth preserving:

- **Clean layering**: thin 180-line `server.py` app factory → 13 routers → in-process harness (`app/harness/`) → unified MCP aggregator (`agent_mcp/`, 23 modules / 51 tools).
- **Strong error recovery**: context-overflow retry with targeted tool-result truncation (`loop.py`), tolerant qwen3_xml tool-arg parsing with repair path, 3-layer compaction (microcompact → LLM summarize → truncate), autonomy frontmatter self-healing + silent-failure regex detection.
- **Good async patterns**: owner-task AsyncExitStack in `mcp_pool.py`, per-session queues with user-preempts-ambient (`sessions_io.py`), per-session file locks, cancel-event-raced tool dispatch.
- **Standardized tool errors**: `ErrorCode` enum + `Result` envelope in `agent_mcp/_shared.py`.
- **File-based state** (JSON/YAML/markdown + SQLite) is grep-able, diffable, and AI-editable — load-bearing for the nightly loop. Keep it.

The real issues: non-atomic writes of unrecoverable state, event-loop blocking, hidden private-API coupling between the two largest memory modules (facts ↔ vault), silent compaction fallback, and a still-unhardened frontmatter parse path.

## Tier 1 — correctness / data-integrity (implemented)

| # | Item | Files |
|---|------|-------|
| 1.1 | Atomic writes (tmp + `os.replace`) for config.yaml rewrites, session JSON, fact files, and the ~2.3MB gitignored `_relationships.json` | `app/atomic_io.py` (new), `app/routers/tools.py`, `app/sessions_io.py`, `agent_mcp/facts.py` |
| 1.2 | `asyncio.Lock` around Playwright launch — concurrent turns (user + ambient + autonomy) could double-launch Chromium and leak one | `agent_mcp/browser.py` |
| 1.3 | `asyncio.to_thread` for sync Read/Write/Edit/Glob handlers (were blocking the shared event loop serving SSE chat, voice, observer) | `agent_mcp/builtin_fs.py` |
| 1.4 | Event-log entry when compaction's LLM-summarize silently falls back to truncation (silent memory loss when vLLM wedges) | `app/compaction.py`, `app/event_log.py` |
| 1.5 | One shared `parse_frontmatter()` with regex fallback — `agent_mcp/autonomy.py` still hard-dropped tasks on bad YAML (the 34/40 dormant-kill failure mode) | `agent_mcp/_shared.py`, `agent_mcp/autonomy.py`, `agent_mcp/backlog.py` |

## Tier 2 — high-leverage refactors (implemented)

| # | Item | Files |
|---|------|-------|
| 2.1 | Extract shared retrieval core (entity extraction, scoring, graph walk, relationships cache) — vault.py imported 6 private functions from facts.py; coupling most likely to cause a nightly-loop regression | `agent_mcp/retrieval.py` (new), `agent_mcp/facts.py`, `agent_mcp/vault.py` |
| 2.2 | UI-mutable state (server enabled, disabled_tools, tool_search) split to `data/tool_overrides.yaml`; config.yaml becomes read-at-boot/hand-edited (yaml.dump had already destroyed all its comments) | `app/config.py`, `app/routers/tools.py` |
| 2.3 | `get_bound_session()` + `make_http_client()` helpers replace 4× contextvar reads and 3× httpx construction | `agent_mcp/_shared.py` + call sites |
| 2.4 | `services:` endpoint block in config.yaml — port 8096 was hardcoded in 15+ files and backends get swapped regularly | `config.yaml`, `app/config.py`, `app/` + `agent_mcp/` call sites |
| 2.5 | API contract tests pinning response shapes the hand-maintained `web/src/api.ts` (1,419 lines) depends on | `tests/test_api_contracts.py` (new) |

## Tier 3 — backlog for the nightly loop (NOT implemented)

Each fits one nightly-improvement branch:

1. **Disk retention groundskeeping** (S): `_pipeline/` = 920MB (259 task dirs never evicted), `sessions/` = 389MB, `logs/` = 294MB. Extend `scripts/groundskeeper/` to delete `_pipeline/tasks/*` >30d and gzip sessions >90d (don't delete — they feed `scripts/mine-trajectories.py`). Wire as an autonomy task.
2. **Pin dependencies** (S): requirements.txt is all `>=`; `mcp>=1.0.0` is the scariest line. Check in a `requirements.lock` from `pip freeze`; rebuilds install from the lock. No poetry/pip-tools needed.
3. **Warn on unknown model alias** (S): `app/config.py` `_get_model_cfg` returns `{}` for unknown names → call silently proceeds against the wrong endpoint. One `logger.warning`.
4. **Cap exception text returned to the model** (S): `str(exc)[:500]` in `_err()` call sites. Don't sanitize — exception text enables model self-correction; just cap it so a traceback doesn't eat context budget.
5. **Root clutter sweep** (S): delete `2026-06-06` (0-byte stray, now tracked in main), relocate `-.en.srt`, `tgs_rag_*.{pdf,txt}`, and the `yt-dlp` binary out of root.
6. **Tool-name length validation at registration** (S): move the 64-char check from harness translation time into `agent_mcp/main.py` listing so bad names fail at startup, not mid-conversation.
7. **Background bash log eviction** (S): `_pipeline/tasks/*.log` never auto-evicted (overlaps with item 1).
8. **Pydantic response models, opportunistically** (M, ongoing): add when touching a router anyway; revisit OpenAPI→TS codegen only if routers ever get fully modeled.

## What NOT to do (judged and rejected)

- **No Prometheus/Grafana/OTel.** One process, one user. Right-sized observability = event_log counters, the 1.4 fix, and optionally one `/api/health` JSON aggregating supervisord state + vLLM ping + last-autonomy-tick (extend `service_health_check.py`, don't replace).
- **No OpenAPI→TypeScript codegen.** Routers return untyped dict `JSONResponse`s; codegen requires Pydantic-modeling 13 routers first. Contract tests (2.5) buy ~80% of the protection for ~10% of the work.
- **No microservices / splitting the aggregator.** In-process harness + single aggregator is why the system is debuggable and fast. The fix for loop-blocking was `to_thread` (1.3), not process boundaries.
- **No database migration off SQLite/JSON.** File-based state is a feature here. The fix was atomic writes (1.1), not Postgres.
- **No auth/rate-limiting theater.** `bypassPermissions` on a single-user LAN box is a design decision; mTLS material in `agent-services/cert/` covers remote-device exposure.
- **Don't consolidate the 5 stopword sets** in `_shared.py` — verified intentionally distinct vocabularies (entity vs scoring vs query vs fact vs skills); collapsing them changes retrieval behavior for zero structural gain.
- **Don't split ChatPanel.tsx (1,306 lines) as a goal.** Frontend has zero tests; a refactor with no net is worse than a working large component. Split opportunistically.

## Notable findings kept for reference

- Session JSON files have no size cap — compaction manages token count, not file size; long sessions grow unboundedly on disk (mitigated by Tier 3 item 1's gzip).
- `loop.py` spill-aware microcompaction assumes the Read tool is enabled; no fallback if disabled.
- Per-turn MCP pool isn't closed if the loop raises before its finally block (mitigated by the FastAPI shutdown hook).
- Frontend: no tests at all; `ChatPanel.tsx` 1,306 lines, four pages ~1,000 lines each; React context state with sequence-gating is otherwise in good shape.
- Test coverage is strong for harness + inner-voice observer (1,018-line integration test), weak-to-absent for routers and frontend — 2.5 addresses the router/frontend seam.
- Supervisord setup is solid (11 services, sane restart policies); logs rotate 10×10MB with no retention beyond that.
