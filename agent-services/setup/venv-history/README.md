# venv-history

Point-in-time `pip freeze` snapshots of the vLLM venvs, kept as a record of what
worked before each upgrade. **Not** install targets — the current lockfiles are
one level up:

| Live lockfile | Venv |
|---|---|
| `../vllm-qwen3.8.versions.txt` | `.venvs/vllm-qwen3.8` (live primary) |
| `../vllm-experimental.versions.txt` | `.venvs/vllm-experimental` |
| `../qwen3-tts.versions.txt` | `.venvs/qwen3-tts` |

These were rescued from `~/lloyd/.backups/` on 2026-08-21, which also held 4.1 GB
of flashinfer JIT and torch-compile caches. Those caches were regenerable build
artifacts, were tracked in git by mistake, and were deleted. Only the text
freezes were worth keeping.
