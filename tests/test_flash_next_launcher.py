"""The primary's launcher, and the boot-log asserts that guard the stall fix.

The 2026-09-10 cutover to FP8 KV (a 692,263-token pool where BF16 held
398,175, and 3200-token pages) is what fixed the 09-09 stall. 2026-09-06 is
the precedent for how it could be undone without anyone noticing: an automod
rollback reverted the secondary's launcher under the same alias and port. So
the pins live in the supervisord program's own `environment=`, and
`flash-next-bootfacts.sh` exits non-zero when a boot lands on anything else.

The boot lines below are the real ones from
`agent-services/logs/agent-llm-primary.log` (the 06:27 BF16 boot and the
11:29 FP8 boot that morning), trimmed.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "agent-services" / "bin"
BOOTFACTS = BIN / "flash-next-bootfacts.sh"
LAUNCHER = BIN / "start-qwen38-flash-next.sh"
CONF = ROOT / "agent-services" / "supervisor" / "conf.d" / "agent-llm-primary.conf"

BF16 = (
    "A/B config: venv=/home/alansrobotlab/lloyd/.venvs/vllm-qwen38-flash-next "
    "ple=worker kv_dtype=bf16 max_num_seqs=8\n"
    "(APIServer pid=1882) INFO 09-10 06:27:10 [api_utils.py:286] non-default "
    "args: {'port': 8096}\n"
    "(EngineCore pid=4301) INFO 09-10 06:27:20 [core.py:123] Initializing a V1 "
    "LLM engine (v0.28.1rc1.dev157+gc5d840ff6) with config: model='/m', "
    "kv_cache_dtype=auto, max_seq_len=262144\n"
    "(EngineCore pid=4301) INFO 09-10 06:29:53 [kv_cache_utils.py:2020] GPU KV "
    "cache size: 398,175 tokens, Maximum concurrency for 262,144 tokens per "
    "request: 1.52x\n"
)
FP8_BOOTING = (
    "A/B config: venv=/home/alansrobotlab/lloyd/.venvs/vllm-flash-next-main "
    "ple=uva kv_dtype=fp8 max_num_seqs=8\n"
    "(APIServer pid=1768813) INFO 09-10 11:29:56 [api_utils.py:286] non-default "
    "args: {'port': 8096, 'kv_cache_dtype': 'fp8'}\n"
    "(EngineCore pid=1781980) INFO 09-10 11:30:07 [core.py:123] Initializing a "
    "V1 LLM engine (v0.28.1rc1.dev661+g6ee5bb0a0) with config: model='/m', "
    "kv_cache_dtype=fp8, max_seq_len=262144\n"
)
FP8 = FP8_BOOTING + (
    "(EngineCore pid=1781980) INFO 09-10 11:33:51 [kv_cache_utils.py:2315] GPU "
    "KV cache size: 692,263 tokens, Maximum concurrency for 262,144 tokens per "
    "request: 2.64x\n"
)


def _bootfacts(tmp_path, text: str, **env) -> subprocess.CompletedProcess:
    log = tmp_path / "agent-llm-primary.log"
    log.write_text(text)
    return subprocess.run(["bash", str(BOOTFACTS), str(log)], capture_output=True,
                          text=True, env={**os.environ, **env}, timeout=60)


def test_the_fp8_boot_passes(tmp_path):
    r = _bootfacts(tmp_path, BF16 + FP8)
    assert r.returncode == 0, r.stdout
    assert "ok    kv_cache_dtype=fp8" in r.stdout
    assert "ok    GPU KV cache size 692263 tokens" in r.stdout


def test_a_bf16_boot_after_it_is_a_regression_on_both_counts(tmp_path):
    r = _bootfacts(tmp_path, FP8 + BF16)
    assert r.returncode == 1, r.stdout
    assert "FAIL  kv_cache_dtype=auto" in r.stdout
    assert "FAIL  GPU KV cache size 398175 tokens" in r.stdout


def test_a_boot_still_loading_is_incomplete_not_a_pass(tmp_path):
    """The earlier boot's KV line must not answer for this one."""
    r = _bootfacts(tmp_path, BF16 + FP8_BOOTING)
    assert r.returncode == 2, r.stdout
    assert "INCOMPLETE" in r.stdout


def test_a_deliberate_arm_can_waive_both_checks(tmp_path):
    r = _bootfacts(tmp_path, FP8 + BF16, EXPECT_KV_DTYPE="", EXPECT_KV_POOL_MIN="0")
    assert r.returncode == 0, r.stdout


def test_a_log_without_launcher_lines_falls_back_to_the_engines_own(tmp_path):
    text = "".join(line + "\n" for line in (BF16 + FP8).splitlines()
                   if not line.startswith("A/B config:"))
    assert _bootfacts(tmp_path, text).returncode == 0


# ── the launcher ──────────────────────────────────────────────────────


def _conf_env() -> dict[str, str]:
    line = next(line for line in CONF.read_text().splitlines()
                if line.startswith("environment="))
    return dict(re.findall(r'(\w+)="([^"]*)"', line))


def test_the_supervisord_environment_pins_the_fix():
    """In the program's own environment — not only as launcher defaults,
    which fall back to the BF16 worker venv when the variables are absent."""
    env = _conf_env()
    assert env.get("KV_CACHE_DTYPE") == "fp8"
    assert env.get("VLLM_VENV", "").endswith("/.venvs/vllm-flash-next-main")
    # The chunk budget adopted from the Layer 3 arms (the conf's comment and
    # architecture/vllm-throughput-mitigation.md §3.3).
    assert env.get("MAX_NUM_BATCHED_TOKENS") == "4096"


def _dry_run(tmp_path, **extra) -> str:
    # The weights are untracked and live only in the production tree, and the
    # launcher resolves them from its own location — so a worktree has none.
    # A DRY_RUN reads nothing but config.json's presence.
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    env = {**os.environ, **_conf_env(), "DRY_RUN": "1", "MODEL_DIR": str(model),
           # A staged arm is one-shot and consumed on read; a test must never
           # be the reader. See the launcher's ARM_ENV comment.
           "ARM_ENV": str(tmp_path / "no-arm.env"), **extra}
    if not (Path(env["VLLM_VENV"]) / "bin" / "python").exists():
        pytest.skip(f"{env['VLLM_VENV']} is not on this machine")
    r = subprocess.run(["bash", str(LAUNCHER)], capture_output=True, text=True,
                       env=env, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_the_supervisord_environment_launches_fp8_on_the_main_venv(tmp_path):
    out = _dry_run(tmp_path)
    assert "--kv-cache-dtype fp8" in out
    assert "vllm-flash-next-main/bin/python -m vllm.entrypoints.openai.api_server" in out
    assert "--max-num-batched-tokens 4096" in out


def test_the_chunk_budget_is_a_knob(tmp_path):
    out = _dry_run(tmp_path, MAX_NUM_BATCHED_TOKENS="2048")
    assert "--max-num-batched-tokens 2048" in out
    assert "batched=2048" in out


def test_an_empty_chunk_budget_means_vllms_default(tmp_path):
    out = _dry_run(tmp_path, MAX_NUM_BATCHED_TOKENS="")
    assert "--max-num-batched-tokens" not in out
    assert "batched=<default>" in out
