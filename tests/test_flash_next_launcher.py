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

import json
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
# Its `limit_mm_per_prompt` and `lm_only=0` are today's production boot's
# (#1420), grafted on: the tower was not loaded that morning.
FP8_BOOTING = (
    "A/B config: venv=/home/alansrobotlab/lloyd/.venvs/vllm-flash-next-main "
    "ple=uva kv_dtype=fp8 max_num_seqs=8 lm_only=0\n"
    "(APIServer pid=1768813) INFO 09-10 11:29:56 [api_utils.py:286] non-default "
    "args: {'port': 8096, 'kv_cache_dtype': 'fp8', 'limit_mm_per_prompt': "
    "{'image': 20, 'video': 0, 'audio': 0}}\n"
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


def test_a_deliberate_arm_can_waive_every_check(tmp_path):
    r = _bootfacts(tmp_path, FP8 + BF16, EXPECT_KV_DTYPE="", EXPECT_KV_POOL_MIN="0",
                   EXPECT_IMAGE_INPUT="")
    assert r.returncode == 0, r.stdout


# ── image input (#1420): supports_vision is true only while the tower loads ──

# The same FP8 boot with LANGUAGE_MODEL_ONLY=1: the launcher passes
# --language-model-only instead of --limit-mm-per-prompt.
FP8_TEXT_ONLY = FP8.replace("lm_only=0", "lm_only=1").replace(
    ", 'limit_mm_per_prompt': {'image': 20, 'video': 0, 'audio': 0}",
    ", 'language_model_only': True")


def test_a_boot_with_image_input_passes_the_image_check(tmp_path):
    r = _bootfacts(tmp_path, FP8)
    assert r.returncode == 0, r.stdout
    assert "ok    image input: limit_mm_per_prompt image=20" in r.stdout


def test_a_text_only_boot_is_a_regression(tmp_path):
    """The KV numbers are production's; only the tower is gone."""
    r = _bootfacts(tmp_path, FP8 + FP8_TEXT_ONLY)
    assert r.returncode == 1, r.stdout
    assert "ok    kv_cache_dtype=fp8" in r.stdout
    assert "FAIL  no image input this boot" in r.stdout and "lm_only=1" in r.stdout
    zero = FP8.replace("{'image': 20,", "{'image': 0,")
    assert _bootfacts(tmp_path, zero).returncode == 1


def test_a_text_only_arm_can_waive_the_image_check(tmp_path):
    r = _bootfacts(tmp_path, FP8_TEXT_ONLY, EXPECT_IMAGE_INPUT="")
    assert r.returncode == 0, r.stdout
    assert "image input" not in r.stdout


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
    # architecture/vllm.md §3.3).
    assert env.get("MAX_NUM_BATCHED_TOKENS") == "4096"
    # config.yaml's models.primary.supports_vision stands on these two: the
    # launcher's own defaults are text-only (#1420).
    assert env.get("LANGUAGE_MODEL_ONLY") == "0"
    assert int(env.get("MM_IMAGES_PER_PROMPT") or 0) > 0


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
    assert "--limit-mm-per-prompt" in out and "--language-model-only" not in out


def test_the_chunk_budget_is_a_knob(tmp_path):
    out = _dry_run(tmp_path, MAX_NUM_BATCHED_TOKENS="2048")
    assert "--max-num-batched-tokens 2048" in out
    assert "batched=2048" in out


def test_an_empty_chunk_budget_means_vllms_default(tmp_path):
    out = _dry_run(tmp_path, MAX_NUM_BATCHED_TOKENS="")
    assert "--max-num-batched-tokens" not in out
    assert "batched=<default>" in out


# ── FlashInfer's JIT cache is per venv, and its compiles are capped ────
# Two venvs on one flashinfer version shared ~/.cache/flashinfer, whose
# build.ninja embeds the venv's include paths, so each venv switch rebuilt
# fused_moe_120 32 wide. On 2026-09-18 that held 83 GiB in `cicc` on top of
# the PLE table and took MemAvailable to 0.


def _flashinfer_env(out: str) -> dict[str, str]:
    line = next(ln for ln in out.splitlines() if ln.startswith("DRY_RUN: FLASHINFER_"))
    return dict(kv.split("=", 1) for kv in line.removeprefix("DRY_RUN: ").split())


def test_each_venv_gets_its_own_flashinfer_cache(tmp_path):
    env = _flashinfer_env(_dry_run(tmp_path, FLASHINFER_WORKSPACE_BASE=""))
    # The launcher's `:-` default treats empty as unset.
    assert env["FLASHINFER_WORKSPACE_BASE"] == _conf_env()["VLLM_VENV"]


def test_flashinfer_compiles_are_capped(tmp_path):
    out = _dry_run(tmp_path, MAX_JOBS="")
    assert _flashinfer_env(out)["MAX_JOBS"] == "8"


def test_the_flashinfer_knobs_can_still_be_overridden(tmp_path):
    env = _flashinfer_env(_dry_run(tmp_path, FLASHINFER_WORKSPACE_BASE=str(tmp_path),
                                   MAX_JOBS="4"))
    assert env == {"FLASHINFER_WORKSPACE_BASE": str(tmp_path), "MAX_JOBS": "4"}


# ── the arm file is read before the knobs it is meant to move ─────────
# Until 2026-09-17 `source "$ARM_ENV"` sat below VLLM_VENV, MODEL_DIR,
# MAX_MODEL_LEN, MTP_ENABLED and MTP_TOKENS, so an arm could not move any
# of the five. Nothing failed: the boot came up on supervisord's own
# `environment=` and flash-next-run-arm.sh appended the result under the
# arm's label. MTP_TOKENS is documented in the launcher as "an A/B knob"
# and had never once been movable by one.


def _arm(tmp_path, *lines: str) -> dict[str, str]:
    arm = tmp_path / "arm.env"
    arm.write_text("# arm: test\n" + "".join(f"export {ln}\n" for ln in lines))
    return {"ARM_ENV": str(arm)}


def test_an_arm_moves_the_mtp_depth(tmp_path):
    # A model dir carrying the draft head, so the arm is followed all the way
    # to the engine argument rather than only to the echoed config line.
    model = tmp_path / "mtp-model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "nvfp4_experts_mtp.safetensors").write_text("")
    out = _dry_run(tmp_path, MODEL_DIR=str(model),
                   **_arm(tmp_path, "MTP_TOKENS=2"))
    assert "mtp=1/k=2" in out
    # DRY_RUN prints the argv through printf '%q', so the JSON arrives escaped.
    assert '"num_speculative_tokens": 2' in out.replace("\\", "")


def test_an_arm_moves_the_context_length(tmp_path):
    out = _dry_run(tmp_path, **_arm(tmp_path, "MAX_MODEL_LEN=131072"))
    assert "max_model_len=131072" in out
    assert "--max-model-len 131072" in out


def test_an_arm_still_moves_a_knob_that_already_worked(tmp_path):
    out = _dry_run(tmp_path, **_arm(tmp_path, "MAX_NUM_BATCHED_TOKENS=2048"))
    assert "batched=2048" in out


def test_an_arm_beats_the_supervisord_environment(tmp_path):
    """The arm file is the override, not a default under it: supervisord
    passes MAX_NUM_BATCHED_TOKENS=4096 in `environment=` and the arm wins."""
    out = _dry_run(tmp_path, **_arm(tmp_path, "MAX_NUM_BATCHED_TOKENS=8192"))
    assert "batched=8192" in out


def test_the_arm_file_is_consumed_by_the_boot_that_reads_it(tmp_path):
    """One-shot: a config that outlived its experiment is the failure this
    slot can least afford."""
    env = _arm(tmp_path, "MTP_TOKENS=2")
    _dry_run(tmp_path, **env)
    assert not Path(env["ARM_ENV"]).exists()


def test_the_arm_is_sourced_above_every_knob_it_can_move():
    """Structural, because the cost of getting this wrong is silent: an arm
    that fell back reads exactly like an arm that made no difference."""
    text = LAUNCHER.read_text()
    source_at = text.index('source "$ARM_ENV"')
    for knob in ("VLLM_VENV", "MODEL_DIR", "MAX_MODEL_LEN", "MTP_ENABLED",
                 "MTP_TOKENS", "MAX_NUM_SEQS", "GPU_MEMORY_UTILIZATION",
                 "KV_CACHE_MEMORY_BYTES", "KV_CACHE_DTYPE", "MOE_BACKEND",
                 "GDN_PREFILL_BACKEND", "MAX_NUM_BATCHED_TOKENS"):
        at = text.index(f'\n{knob}="${{{knob}:-')
        assert source_at < at, f"{knob} is resolved before the arm file is read"


# ── the host-RAM boot gate: ONE definition, two routes ────────────────
# `agent-llm-primary` holds a 95.37 GiB BF16 n-gram table in HOST ram and
# `supervisorctl stop` returns before the kernel has that memory back, so both
# routes that boot it wait for MemAvailable first. Until #1340 the LANDING
# route's pair stood in `scripts/automod/promote.py` (180 floor / 150 abort)
# and this SWEEP route's pair stood in `flash-next-run-arm.sh` as two bare
# literals (150 wait / 120 abort), with no shared file and nothing relating
# them: a reader who opened the shell saw a 150 with no way to tell which
# route's floor it was, and the restart skill's pointer to the production pair
# was 596 lines off, so neither number could be checked from the page an
# operator follows. These tests are what keeps one number in one place.

RAM_GATE_DEF = BIN / "ram-boot-gate.sh"
ARM = BIN / "flash-next-run-arm.sh"
RESTART_SKILL = Path.home() / "obsidian" / "skills" / "restart-lloyd" / "SKILL.md"

BOOT_GATE_NAMES = ("PRIMARY_RAM_FLOOR_GIB", "PRIMARY_RAM_ABORT_GIB",
                   "SWEEP_RAM_WAIT_GIB", "SWEEP_RAM_ABORT_GIB")

# In these two files any comparison against a three-digit number IS a hardcoded
# boot-gate threshold: their only thresholds are the gate's, and the gate's are
# variables. Same shape as the acceptance falsifier in the item.
BARE_THRESHOLD_CMP = re.compile(r"-(?:ge|gt|le|lt)\s+[0-9]{3}\b")

_WALK_PRUNED = {".git", ".pytest_cache", ".venv", ".venvs", "__pycache__",
                "_pipeline", "dist", "logs", "node_modules", "qmd"}


def _repo_code_files() -> list[Path]:
    """Every .py/.sh in this checkout, minus the trees that are not the repo."""
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in _WALK_PRUNED]
        out += [Path(dirpath) / fn for fn in filenames if fn.endswith((".py", ".sh"))]
    return out


def _files_defining(name: str) -> list[str]:
    """Repo files that assign a boot-gate threshold as a bare literal."""
    pattern = re.compile(rf"^{name}=[0-9]+$", re.MULTILINE)
    hits = []
    for path in _repo_code_files():
        try:
            if pattern.search(path.read_text()):
                hits.append(str(path.relative_to(ROOT)))
        except (OSError, UnicodeDecodeError):
            continue
    return hits


def _gate_values() -> dict[str, int]:
    """The shared definition, read through the reader the landing route uses."""
    from scripts.automod.ram_gate import parse_ram_gate
    return parse_ram_gate(RAM_GATE_DEF.read_text())


def _gate_shell(tmp_path, *, meminfo_gib, extra_assignments=(), tries=3, interval=0,
                meminfo_name="meminfo"):
    """Run the sourced gate in a real bash against a fake /proc/meminfo.

    The cadence is overridable for exactly this: `RAM_GATE_INTERVAL=0` keeps a
    refusal from costing five real minutes of sleep in a test suite.
    """
    mem = tmp_path / meminfo_name
    if meminfo_gib is not None:
        mem.write_text("MemTotal:       264000000 kB\n"
                       f"MemAvailable:   {meminfo_gib * 1048576} kB\n")
    script = " ".join(["source \"$1\";", *extra_assignments,
                       'ram_gate_wait_for_room "${2:-arm}"'])
    return subprocess.run(["bash", "-c", script, "ram-boot-gate", str(RAM_GATE_DEF),
                           "testlabel"],
                          capture_output=True, text=True, timeout=120,
                          env={**os.environ, "MEMINFO": str(mem),
                               "RAM_GATE_TRIES": str(tries),
                               "RAM_GATE_INTERVAL": str(interval)})


def test_the_four_boot_gate_thresholds_are_defined_in_exactly_one_file():
    """#1340 clause 1. Each name may be assigned in exactly one file — the
    definition file — and that file must hold all four, so neither route can
    carry a private copy of the number that decides whether a boot is safe."""
    values = _gate_values()
    expected = "agent-services/bin/ram-boot-gate.sh"
    for name in BOOT_GATE_NAMES:
        assert _files_defining(name) == [expected], \
            f"{name} is defined outside {expected}: {_files_defining(name)}"
    # Each route's abort line has to sit below its own wait line, or the gate is
    # a no-op (always boots) or a permanent refusal.
    assert values["PRIMARY_RAM_FLOOR_GIB"] > values["PRIMARY_RAM_ABORT_GIB"]
    assert values["SWEEP_RAM_WAIT_GIB"] > values["SWEEP_RAM_ABORT_GIB"]


def test_the_definition_names_the_route_each_threshold_gates():
    """A threshold with no route written on it is how one 150 came to mean two
    different things. The definition must carry both consumers' names and the
    reason the sweep pair is the looser of the two."""
    text = RAM_GATE_DEF.read_text()
    assert "_restart_primary" in text, "the landing route is not named"
    assert "flash-next-run-arm.sh" in text, "the sweep route is not named"
    assert re.search(r"sits lower", text, re.IGNORECASE), \
        "nothing says why the sweep pair is lower than the landing pair"
    assert "intentional" in text, "the gap between the pairs is not labelled intentional"


def test_both_routes_read_the_same_four_numbers():
    """The seam this item exists because nothing crossed: promote.py reaches the
    definition through `scripts.automod.ram_gate`, the arm script shells in the
    same file, and a reader of either has to get the same four values."""
    from scripts.automod import promote as P, ram_gate
    values = _gate_values()
    want = tuple(values[n] for n in BOOT_GATE_NAMES)
    assert (ram_gate.PRIMARY_RAM_FLOOR_GIB, ram_gate.PRIMARY_RAM_ABORT_GIB,
            ram_gate.SWEEP_RAM_WAIT_GIB, ram_gate.SWEEP_RAM_ABORT_GIB) == want
    assert (P.PRIMARY_RAM_FLOOR_GIB, P.PRIMARY_RAM_ABORT_GIB) == want[:2]
    assert P.PRIMARY_RAM_WAIT_SECONDS == float(values["PRIMARY_RAM_WAIT_SECONDS"])
    # Now the same file through bash, not through Python: sourcing must yield the
    # same four, or the sweep route is gating on something the reader can't see.
    r = subprocess.run(
        ["bash", "-c", 'source "$1"; for n in "${@:2}"; do printf "%s=%s\\n" "$n" "${!n}"; done',
         "ram-boot-gate", str(RAM_GATE_DEF), *BOOT_GATE_NAMES],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert dict(line.split("=", 1) for line in r.stdout.split()) == \
        {n: str(values[n]) for n in BOOT_GATE_NAMES}


def test_the_landing_route_holds_no_boot_gate_number_of_its_own():
    """#1340 clause 1's other half: `promote.py` binds the names, never the
    values, so the file that boots the engine unattended cannot drift from the
    file the operator is standing in front of."""
    src = (ROOT / "scripts" / "automod" / "promote.py").read_text()
    for name in BOOT_GATE_NAMES + ("PRIMARY_RAM_WAIT_SECONDS",):
        assert not re.search(rf"^{name} *= *[0-9]", src, re.MULTILINE), \
            f"promote.py defines {name} again, next to the copy it imports"


def test_a_definition_the_python_reader_cannot_trust_raises(tmp_path):
    """No fallback and no remembered default: a half-read definition refuses at
    import rather than booting a 95 GiB engine on a number nobody wrote down."""
    from scripts.automod import ram_gate
    text = RAM_GATE_DEF.read_text()
    missing = tmp_path / "missing.sh"
    missing.write_text(text.replace("PRIMARY_RAM_ABORT_GIB=150\n", "", 1))
    with pytest.raises(ValueError, match="no definition for PRIMARY_RAM_ABORT_GIB"):
        ram_gate.load_ram_gate(missing)
    inverted = tmp_path / "inverted.sh"
    inverted.write_text(re.sub(r"^PRIMARY_RAM_ABORT_GIB=[0-9]+$",
                              "PRIMARY_RAM_ABORT_GIB=999", text, flags=re.MULTILINE))
    with pytest.raises(ValueError, match="must sit below"):
        ram_gate.load_ram_gate(inverted)
    duplicated = tmp_path / "duplicated.sh"
    duplicated.write_text(text + "PRIMARY_RAM_FLOOR_GIB=200\n")
    with pytest.raises(ValueError, match="defined twice"):
        ram_gate.load_ram_gate(duplicated)


def test_the_gate_refuses_below_its_abort_line(tmp_path):
    """90 GiB free is under the sweep abort: the gate must come back non-zero
    and say what it measured. This is the status the arm script turns into
    `exit 3` before it starts anything."""
    r = _gate_shell(tmp_path, meminfo_gib=90)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "ABORT testlabel" in r.stdout
    assert "only 90 GiB" in r.stdout


def test_the_gate_passes_once_the_previous_table_is_released(tmp_path):
    """At or over the wait threshold it does not stall: no dots, immediate zero."""
    r = _gate_shell(tmp_path, meminfo_gib=200)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count(".") == 0, "it waited when the room was already there"
    assert "200 GiB available" in r.stdout


def test_the_gate_waits_out_its_budget_between_abort_and_wait(tmp_path):
    """Between the two numbers the wait is the whole point: it spends every
    pass on the previous boot's mapping, then boots, because the abort line —
    not the wait line — is where an arm is refused."""
    r = _gate_shell(tmp_path, meminfo_gib=130, tries=4)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "...." in r.stdout, "it gave up before spending its wait budget"
    assert "130 GiB available" in r.stdout


def test_the_gate_refuses_when_it_cannot_measure_or_the_definition_is_inverted(tmp_path):
    """Two ways the gate gets no answer: no MemAvailable to read, and a pair
    whose abort line sits above its wait line. Both refuse, neither boots."""
    r = _gate_shell(tmp_path, meminfo_gib=None, meminfo_name="absent")
    assert r.returncode == 3, r.stdout + r.stderr
    assert "no MemAvailable reading" in r.stderr
    inverted = _gate_shell(tmp_path, meminfo_gib=200,
                           extra_assignments=("SWEEP_RAM_ABORT_GIB=999",))
    assert inverted.returncode == 4, inverted.stdout + inverted.stderr
    assert "must sit below" in inverted.stderr


def test_the_arm_route_asks_the_shared_gate_before_starting_the_engine():
    """#1340 clause 2, the wiring half: the refusal status is only a refusal if
    the arm script exits on it above the line that starts the engine. The
    behaviour under a low reading is `test_the_gate_refuses_below_its_abort_line`
    — that runs the gate in bash; this one pins that the arm route cannot reach
    `$SUP start` without passing through it, and carries no threshold of its
    own to drift."""
    text = ARM.read_text()
    assert 'RAM_GATE_SCRIPT="$ROOT/agent-services/bin/ram-boot-gate.sh"' in text
    assert 'source "$RAM_GATE_SCRIPT"' in text
    assert 'cannot read the boot-gate definition' in text, \
        "an unreadable definition must refuse the arm, not run without a gate"
    assert 'ram_gate_wait_for_room "$LABEL" || exit 3' in text
    assert not BARE_THRESHOLD_CMP.search(text), \
        f"a bare GiB threshold is left in the arm script: " \
        f"{[ln for ln in text.splitlines() if BARE_THRESHOLD_CMP.search(ln)]}"
    source_at = text.index('source "$RAM_GATE_SCRIPT"')
    ask_at = text.index('ram_gate_wait_for_room "$LABEL"')
    assert source_at < text.index("$SUP stop agent-llm-primary"), \
        "the gate is read only after the engine is already stopped"
    assert text.index("$SUP stop agent-llm-primary") < ask_at \
        < text.index("$SUP start agent-llm-primary"), \
        "the gate must be asked after the stop and before the start"


def test_the_vllm_doc_points_at_the_definition_and_carries_no_line_cite():
    """§8 of `architecture/vllm.md` is the page an operator reads when they are
    NOT following the skill, and the last time it was corrected it gained four
    line-number cites that this change alone invalidated. Pointers into a moving
    file belong on a symbol, so the doc names the definition file and the
    threshold names and no line numbers."""
    doc = (ROOT / "architecture" / "vllm.md").read_text()
    for stale in re.findall(r"(?:promote\.py|flash-next-run-arm\.sh):[0-9]+", doc):
        raise AssertionError(f"architecture/vllm.md cites a moving file by line: {stale}")
    assert "ram-boot-gate.sh" in doc, "the doc does not name the one definition"
    # Positive control: these assertions are only meaningful if the section the
    # item corrected is still the section being read.
    assert "_restart_primary" in doc and "SWEEP_RAM_ABORT_GIB" in doc
    assert "round restart --only agent-llm-primary" in doc, \
        "the doc no longer names the production restart route"


@pytest.mark.skipif(not RAM_GATE_DEF.exists(), reason="no boot-gate definition")
def test_the_arm_scripts_gate_survives_being_sourced_under_set_uo():
    """The arm script runs `set -uo pipefail`; a definition file that tripped on
    an unset variable would abort the arm before it stopped anything."""
    r = subprocess.run(["bash", "-c", 'set -uo pipefail; source "$1"; '
                        'ram_gate_numbers_ok && echo defined-ok', "x", str(RAM_GATE_DEF)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "defined-ok" in r.stdout


@pytest.mark.live_vault
@pytest.mark.skipif(not RESTART_SKILL.exists(), reason=f"vault not present at {RESTART_SKILL.parent}")
def test_the_restart_skill_cites_the_gate_by_symbol_and_quotes_the_shared_numbers():
    """#1340 clause 4. `promote.py:930` was 596 lines from `_restart_primary`,
    which is how a skill stayed wrong for weeks: a line number is a pointer only
    until the file grows. So the skill names the symbol and no line number, and
    the four GiB figures it quotes are read out of its own sentence and compared
    to the one definition both routes enforce."""
    text = RESTART_SKILL.read_text()
    values = _gate_values()
    assert not re.search(r"promote\.py:[0-9]", text), \
        f"the skill cites promote.py by line again: " \
        f"{re.findall(r'.{40}promote\.py:[0-9]+', text)}"
    assert "_restart_primary" in text, "the skill no longer names the function by symbol"
    landing = re.search(r"waits `MemAvailable` back to ([0-9]+) GiB and refuses to "
                        r"start under ([0-9]+)", text)
    sweep = re.search(r"waits to ([0-9]+) and refuses below ([0-9]+)", text)
    assert landing and sweep, "the skill stopped stating the two routes' numbers"
    assert [int(x) for x in landing.groups()] == [values["PRIMARY_RAM_FLOOR_GIB"],
                                                  values["PRIMARY_RAM_ABORT_GIB"]]
    assert [int(x) for x in sweep.groups()] == [values["SWEEP_RAM_WAIT_GIB"],
                                                values["SWEEP_RAM_ABORT_GIB"]]
    assert "ram-boot-gate.sh" in text, "the skill does not point at the one definition"


# ---------------------------------------------------------------------------
# #1625 — the engine-output integrity canary on the sweep route
#
# bootfacts reads the KV dtype and the pool size off the boot log; the boot guard
# counts engine inits. Neither reads a token, and AI21's two vLLM bugs — both in
# the Mamba state cache this hybrid runs on — produce confident WRONG output with
# no crash, no warning and no error line. So an arm could pass everything above
# the canary and still be serving corrupted output under a config that looks
# identical in the log. The canary replays eval/engine_output/corpus.yaml (21
# prompts, temperature 0) and compares it to one named reference record.
# ---------------------------------------------------------------------------

CANARY_START = "# --- engine-output integrity canary"
CANARY_END = "# SKIP_BENCH=1 stops here"


def _canary_statements() -> list[str]:
    """The block's executable statements: comment lines dropped, backslash
    continuations joined into the one statement they are.

    Assertions about a *call* have to see a call as bash sees it. Grepping raw
    lines splits the two probe invocations across their continuation lines, which
    reads as "the flag is missing" and "the status guard is missing" when both are
    present, and a bare `| head` scan matches the prose that warns against it."""
    joined = _canary_block().replace("\\\n", " ")
    return [ln for ln in joined.splitlines(keepends=True)
            if not ln.lstrip().startswith("#")]


def _statement(match: str) -> str:
    """The one executable statement containing `match` — asserted unique, so a
    second call that dodges the checks cannot be silently skipped over."""
    hits = [s for s in _canary_statements() if re.search(match, s)]
    assert len(hits) == 1, f"expected exactly one statement matching {match!r}, got {hits}"
    return hits[0]


def _canary_block() -> str:
    """The canary's own lines, lifted out of the arm script.

    Everything the tests below assert is then asserted about the code that runs,
    not about a copy: the block is executed by real bash in `_canary_run`, so a
    status that escaped into the script's own exit, or a verdict line that lost
    its label, fails the way it would fail an arm.
    """
    text = ARM.read_text()
    start = text.index(CANARY_START)
    end = text.index(CANARY_END, start)
    return text[start:end]


#: What `compare --json-out` writes for a DIVERGING arm: `compare_records` sorts
#: rows worst first (its own docstring), so rows[0] is the exceeder, and its rc is
#: 1. The clean payload has diverged 0 because `main` returns
#: `1 if result["diverged"] else 0` — a payload that contradicted the status the
#: stub exits with would let the block be tested against an impossible probe.
CANARY_PAYLOAD_DIVERGED = {
    "reference": "20260924T211316Z_idle-5", "current": "arm-fp8kv",
    "reference_engine": {"venv": "/x/.venvs/lloyd", "vllm_version": "0.11.2",
                         "kv_cache_dtype": "fp8", "kv_cache_size_tokens": 555776},
    "current_engine": {"venv": "/x/.venvs/lloyd", "vllm_version": "0.11.2",
                       "kv_cache_dtype": "bf16", "kv_cache_size_tokens": 555776},
    "diverged": 3,
    "rows": [
        {"id": "sys_summary", "agreement": 0.412, "first_divergence": 7,
         "median_lp_delta": 0.031, "token_lp_delta": 0.44, "ref_tokens": 64,
         "cur_tokens": 71, "tier": "exact", "exceeds": True,
         "reasons": ["agreement 0.412 < floor 0.950",
                     "token logprob delta 0.44 > floor 0.1"]},
        {"id": "count_up", "agreement": 1.0, "first_divergence": None,
         "median_lp_delta": 0.0, "token_lp_delta": 0.0, "ref_tokens": 40,
         "cur_tokens": 40, "tier": "exact", "exceeds": False, "reasons": []},
    ],
}
CANARY_PAYLOAD_CLEAN = {
    **CANARY_PAYLOAD_DIVERGED,
    "diverged": 0,
    "rows": [{**r, "exceeds": False, "reasons": []} for r in CANARY_PAYLOAD_DIVERGED["rows"]],
}


def _canary_run(tmp_path: Path, *, run_rc: int, cmp_rc: int,
                emit_record: bool = True, skip_bench: bool = False,
                cmp_payload: dict | None = CANARY_PAYLOAD_DIVERGED,
                ) -> "subprocess.CompletedProcess":
    """Run the extracted canary block under `set -euo pipefail` — STRICTER than
    the arm script's own `set -uo pipefail` — against a fake root whose python is
    a stub returning the given statuses.

    The strictness is the assertion: an arm may not be abortable by its own
    canary even under a header it might one day gain, and any status that the
    block fails to swallow kills the harness before the tail echo. The harness
    ends with the arm script's literal tail line so `reaches done` is measured,
    not inferred.

    Two things stand in for the runtime the block really runs in. `LLOYD_DATA` is
    the directory the arm's bench results and engine log already come from
    (`${LLOYD_DATA:-$HOME/lloyd-data}/logs/services/`), so the persisted verdict
    lands where a sweep's throughput already lands. The stub serves `--json-out`
    with `cmp_payload`, which is what the real `compare` writes there; for a
    refused compare (rc>=2) it writes nothing, because `main` raises
    `ProbeRefused` before it reaches the write — the block must therefore tolerate
    an absent payload, which is pinned in tests/test_engine_output_probe.py.

    `skip_bench=True` appends the arm script's OWN `SKIP_BENCH` block, so the run
    really does exit there: a verdict written after it is then not on disk.
    """
    root = tmp_path / "fakeroot"
    (root / ".venvs/lloyd/bin").mkdir(parents=True)
    (root / "eval/engine_output/idle").mkdir(parents=True)
    (root / "eval/engine_output/idle/20260924T211316Z_idle-5.json").write_text("{}\n")
    payload = tmp_path / "cmp-payload.json"
    if cmp_payload is None:
        payload.write_text("not json at all\n", encoding="utf-8")
    else:
        payload.write_text(json.dumps(cmp_payload), encoding="utf-8")
    stub = root / ".venvs/lloyd/bin/python"
    stub.write_text("""#!/usr/bin/env bash
set -uo pipefail
case "$2" in
  run)
    # A real `run` that refuses writes NOTHING: main() raises ProbeRefused before
    # write_record, so it prints no `wrote` line. Emitting one while exiting 2
    # would hand the block a record for an arm that has none.
    if [[ "$STUB_EMIT_RECORD" == "1" && "$STUB_RUN_RC" == "0" ]]; then
      echo "wrote /tmp/arm-record.json  (21 prompts, engine stub)"
    fi
    echo "engine_output_probe: cannot decide — the engine refused the corpus"
    exit "$STUB_RUN_RC"
    ;;
  compare)
    echo "reference $4: stub vllm 0.11.2 kv fp8 pool 555776"
    echo "current   /tmp/arm-record.json: stub vllm 0.11.2 kv fp8 pool 555776"
    echo "3 of 21 prompts past their idle floor"
    # --json-out is served only when the comparison decided: `main` raises
    # ProbeRefused before writing anything, so rc>=2 leaves no payload file.
    prev=""
    for a in "$@"; do
      if [[ "$prev" == "--json-out" && "$STUB_CMP_RC" -lt 2 ]]; then
        cp "$STUB_CMP_PAYLOAD" "$a"
      fi
      prev="$a"
    done
    exit "$STUB_CMP_RC"
    ;;
esac
exit 9
""", encoding="utf-8")
    stub.chmod(0o755)
    body = _canary_block()
    if skip_bench:
        # To the END of the arm script, bench call included: with SKIP_BENCH=1 the
        # run has to stop at the exit, and the bench's own line is how this test
        # knows it did. Slicing before the bench would make the absence of that
        # line unfalsifiable — it was never in the script.
        arm_text = ARM.read_text()
        body += arm_text[arm_text.index(CANARY_END):]
    script = tmp_path / "canary-under-test.sh"
    script.write_text(
        "set -euo pipefail\n"
        f'ROOT="{root}"\n'
        'LABEL=fp8kv\n'
        + body + '\necho "=== arm $LABEL done ==="\n',
        encoding="utf-8")
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, timeout=60,
        env={"PATH": "/usr/bin:/bin", "STUB_RUN_RC": str(run_rc),
             "STUB_CMP_RC": str(cmp_rc), "STUB_CMP_PAYLOAD": str(payload),
             "LLOYD_DATA": str(tmp_path / "lloyd-data"),
             "SKIP_BENCH": "1" if skip_bench else "0",
             "STUB_EMIT_RECORD": "1" if emit_record else "0"})


def _canary_verdicts(tmp_path: Path) -> list[dict]:
    """The persisted canary verdicts, read back from the JSONL the block appends
    to — the file, not the echo. The point of #2163 is that the decision survives
    the restart window, and a verdict that exists only in stdout does not."""
    log = tmp_path / "lloyd-data/logs/services/flash-next-canary.jsonl"
    assert log.is_file(), (
        f"no canary verdict was persisted; expected one JSON line at {log}")
    return [json.loads(ln) for ln in log.read_text().splitlines() if ln.strip()]


def test_the_output_canary_sits_below_the_boot_guard_and_above_the_bench():
    """Clause 1: ordering is the safety property, not the invocation.

    Above the boot guard, a probe would send its corpus to a resurrected engine
    running production defaults — the failure the crash-guard comment above
    records as poisoning a 2026-09-08 arm with a fake 120.7 tok/s. Below the
    SKIP_BENCH exit, a probe-only arm (the admission sweeps, which never
    benchmark) would carry no integrity verdict at all. Both halves, and the
    label: a record nobody can attribute to an arm cannot be compared across a
    sweep."""
    text = ARM.read_text()
    guard = text.index('echo "boot guard: 1 engine init, no startup failures')
    skip = text.index('if [[ "${SKIP_BENCH:-0}" == "1" ]]; then')
    assert CANARY_START in text, "no engine-output canary in the arm route"
    assert guard < text.index(CANARY_START) < skip, (
        "the canary must run after the boot guard proves which engine is "
        "serving and before SKIP_BENCH can exit the arm")

    run_call = _statement(r'"\$ENGINE_OUTPUT_PROBE" run')
    assert '--label "arm-$LABEL"' in run_call, run_call
    assert '"$ROOT/.venvs/lloyd/bin/python"' in run_call, run_call
    assert not any("--base-url" in s for s in _canary_statements()), (
        "the probe defaults to 127.0.0.1:8096, the port this script polls /health on")


def test_the_canary_reports_every_probe_status_and_never_aborts_the_arm(tmp_path):
    """Clauses 2 and 4, in real bash: five statuses, three verdicts, zero aborts.

    exit 1 from `compare` is a finding ABOUT the engine — the bench is about to
    measure the same engine, so stopping here would trade a measurement for an
    alarm. exit 2 means the instrument could not decide, which is even less
    reason to stop. Each row must therefore print a verdict naming the arm label
    and the decision, and still reach the arm script's own tail line, under
    `set -euo pipefail`."""
    cases = [
        dict(run_rc=0, cmp_rc=0, decision="within-floor"),
        dict(run_rc=0, cmp_rc=1, decision="DIVERGED"),
        dict(run_rc=0, cmp_rc=2, decision="could-not-decide"),
        dict(run_rc=2, cmp_rc=0, decision="could-not-decide"),
        dict(run_rc=0, cmp_rc=0, emit_record=False, decision="could-not-decide"),
    ]
    for case in cases:
        want = case.pop("decision")
        r = _canary_run(tmp_path / str(abs(hash(str(case)))), **case)
        assert r.returncode == 0, f"{case}: the canary aborted the arm\n{r.stdout}{r.stderr}"
        assert "=== arm fp8kv done ===" in r.stdout, f"{case}: {r.stdout}"
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("engine-output canary:")]
        assert len(line) == 1, f"{case}: expected one verdict line, got {line}"
        assert "arm fp8kv" in line[0], f"{case}: verdict does not name the arm: {line[0]}"
        assert want in line[0], f"{case}: wanted {want}, got {line[0]}"
    divergence = _canary_run(tmp_path / "diverged", run_rc=0, cmp_rc=1)
    assert "3 of 21 prompts past their idle floor" in divergence.stdout, (
        "a divergence verdict without the comparator's reason is an alarm with "
        "nothing to act on")


def test_the_canary_names_its_reference_rather_than_taking_the_newest(tmp_path):
    """Clause 3: `compare --reference` defaults to "the newest other record in
    the current's directory", and every arm's record lands in one directory — so
    a defaulted sweep grades arm N against arm N-1, and drift across a sweep is
    invisible by construction.

    Pinned three ways: the call passes `--reference "$ENGINE_OUTPUT_REF"`; that
    variable's default is a file that exists in the committed corpus directory
    (a reference that has been moved or renamed turns every arm into exit 2,
    which is exactly the silent-blind state this canary exists to close); and it
    is NOT inside the probe's own output dir, where the newest-other-record
    default lives."""
    text = ARM.read_text()
    compare_call = _statement(r'"\$ENGINE_OUTPUT_PROBE" compare')
    assert '--reference "$ENGINE_OUTPUT_REF"' in compare_call, compare_call
    assert "--current" in compare_call, compare_call
    default = re.search(r'ENGINE_OUTPUT_REF="\$\{ENGINE_OUTPUT_REF:-(.*?)\}"', text)
    assert default, "ENGINE_OUTPUT_REF has no default to name"
    ref = Path(default.group(1).replace("$ROOT", str(ROOT)))
    assert ref.is_file(), f"the named reference record does not exist: {ref}"
    assert "engine_output" in ref.parts, ref
    out_dir = re.search(r'DEFAULT_OUT_DIR\s*=\s*([^#]+)',
                        (ROOT / "eval/engine_output_probe.py").read_text())
    assert out_dir and "EVAL_BASELINES_DIR" in out_dir.group(1), (
        "the probe's output dir changed; re-check that the named reference is "
        "not inside it")
    assert str(ref).startswith(str(ROOT / "eval/engine_output")), (
        f"the reference sits where arm records land: {ref}")


def test_the_canary_status_capture_survives_a_header_that_gains_set_e():
    """The one line that decides clause 2, pinned as a pattern rather than as a
    behaviour: `RC=$?` on the line AFTER a failing assignment aborts under
    `set -e`, and `sed | head -1` under `pipefail` can leave a pipeline at 141.

    Both forms read as "the status was captured" and neither is: the first
    returns the arm script's exit status through the canary, the second can kill
    the arm before the verdict is printed."""
    statements = _canary_statements()
    for var in ("RUN", "CMP"):
        call = _statement(rf'CANARY_{var}_OUT=\$\(')
        assert f"|| CANARY_{var}_RC=$?" in call, (
            f"the {var.lower()} call must carry its own status guard: {call}")
    assert not any(re.fullmatch(r"\s*CANARY_(RUN|CMP)_RC=\$\?\n", s) for s in statements), (
        "reading $? on its own line aborts the arm under set -e")
    assert not any(re.search(r"\|\s*head\b", s) for s in statements), (
        "a pipe into head can return 141 under pipefail")
    assert "awk" in _statement(r"CANARY_RECORD="), (
        "the record path must be pulled out in one process, not a pipeline")


# ---------------------------------------------------------------------------
# #2163 — the verdict has to survive the window it was measured in
#
# Everything above prints a verdict. An A/B arm runs inside a primary restart
# window, and the window is exactly what goes away: `SKIP_BENCH=1` — how an
# admission sweep runs, since bench-flash-next.py measures decode and defeats the
# prefix cache — exits this script before the bench's `--out` write, so a
# canary-only arm persisted NOTHING AT ALL. Three absent bytes proved it: no
# ~/lloyd-data/eval/baselines/engine_output/, no flash-next-arms.jsonl, no
# flash-next-arm.env, and no `engine-output canary:` line anywhere under
# ~/lloyd-data/logs/, since the canary wired in in dfb18646 (2026-09-28).
# ---------------------------------------------------------------------------


def test_the_canary_persists_one_verdict_line_naming_the_worst_prompt(tmp_path):
    """Clause 1: the line is the arm's decision, not a breadcrumb.

    Read back from the file, so every field is the block's, not an echo. The
    reference and the arm's own record are named by path because the next reader
    has to be able to re-run the comparison; the diverged count and the worst row
    because "DIVERGED" without the prompt id and the first_divergence index is an
    alarm with nothing to act on. The row is `rows[0]`: compare_records sorts
    worst first, so the persisted line points at the prompt the comparator itself
    put on top of its own table."""
    root = tmp_path / "fakeroot"
    assert _canary_run(tmp_path, run_rc=0, cmp_rc=1).returncode == 0

    lines = _canary_verdicts(tmp_path)
    assert len(lines) == 1, f"one arm must append exactly one line, got {lines}"
    v = lines[0]
    assert v["arm"] == "fp8kv", v
    assert (v["run_rc"], v["compare_rc"]) == (0, 1), v
    assert v["decision"] == "past-floor", v
    assert v["reference"] == str(root / "eval/engine_output/idle/20260924T211316Z_idle-5.json"), v
    assert v["record"] == "/tmp/arm-record.json", v
    assert v["diverged"] == CANARY_PAYLOAD_DIVERGED["diverged"], v

    worst, row = v["worst_prompt"], CANARY_PAYLOAD_DIVERGED["rows"][0]
    assert worst is not None, v
    assert worst["prompt"] == row["id"], worst
    assert worst["agreement"] == row["agreement"], worst
    assert worst["first_divergence"] == row["first_divergence"], worst
    assert worst["token_lp_delta"] == row["token_lp_delta"], worst
    assert worst["reasons"] == row["reasons"], (
        f"the reasons are the actionable half of a divergence: {worst}")


def test_the_canary_persists_a_line_for_every_outcome_including_its_own_failure(tmp_path):
    """Clause 2: three decisions, and the third one is the one that gets lost.

    An instrument failure that only prints is indistinguishable, a day later,
    from an arm that was never run — which is the state this item was filed from.
    So could-not-decide must carry the status that produced it and the last line
    the instrument said, and it must be written whether the engine refused the
    corpus (run rc 2, no record, compare never reached) or the comparator refused
    to decide (compare rc 2, no payload file at all)."""
    refused_run = "engine_output_probe: cannot decide — the engine refused the corpus"
    cases = [
        (dict(run_rc=0, cmp_rc=0, cmp_payload=CANARY_PAYLOAD_CLEAN),
         dict(decision="within-floor", run_rc=0, compare_rc=0, diverged=0,
              record="/tmp/arm-record.json")),
        (dict(run_rc=0, cmp_rc=1),
         dict(decision="past-floor", run_rc=0, compare_rc=1, diverged=3,
              record="/tmp/arm-record.json")),
        (dict(run_rc=0, cmp_rc=2),
         dict(decision="could-not-decide", run_rc=0, compare_rc=2, diverged=None,
              record="/tmp/arm-record.json",
              last_output="3 of 21 prompts past their idle floor")),
        (dict(run_rc=2, cmp_rc=0),
         dict(decision="could-not-decide", run_rc=2, compare_rc=None, diverged=None,
              record=None, last_output=refused_run)),
        (dict(run_rc=0, cmp_rc=0, emit_record=False),
         dict(decision="could-not-decide", run_rc=0, compare_rc=None, diverged=None,
              record=None, last_output=refused_run)),
    ]
    for i, (run_kwargs, want) in enumerate(cases):
        here = tmp_path / f"case-{i}"
        r = _canary_run(here, **run_kwargs)
        assert r.returncode == 0, f"{run_kwargs}: the canary aborted the arm\n{r.stderr}"
        assert "=== arm fp8kv done ===" in r.stdout, f"{run_kwargs}: {r.stdout}"
        lines = _canary_verdicts(here)
        assert len(lines) == 1, f"{run_kwargs}: expected one persisted line, got {lines}"
        v = lines[0]
        for key, value in want.items():
            assert v[key] == value, f"{run_kwargs}: {key} wanted {value!r}, got {v.get(key)!r}"
        if want["decision"] == "could-not-decide":
            assert v["worst_prompt"] is None, f"{run_kwargs}: {v}"
            assert v["last_output"], f"{run_kwargs}: an instrument failure with no " \
                                     f"last line leaves nothing to read: {v}"


def test_the_canary_verdict_is_on_disk_when_skip_bench_exits_the_arm(tmp_path):
    """Clause 3, pinned as a real exit rather than an index comparison.

    The arm script's own SKIP_BENCH block and everything below it — the bench call
    included — are appended to the lifted canary block, so this run really does
    stop at that `exit 0`: the bench's own line is absent, which is how the test
    knows the exit fired rather than the script running on. A verdict written below
    the exit, where the bench's `--out` write is, is not on disk in that run."""
    r = _canary_run(tmp_path, run_rc=0, cmp_rc=1, skip_bench=True)
    assert r.returncode == 0, r.stderr
    assert "SKIP_BENCH=1 — arm fp8kv is serving; not benchmarking" in r.stdout, r.stdout
    assert "--- benchmarking ---" not in r.stdout, (
        "the run reached the bench, so it never exercised the SKIP_BENCH exit and "
        "proved nothing about ordering: " + r.stdout)

    lines = _canary_verdicts(tmp_path)
    assert len(lines) == 1, f"a canary-only arm must persist exactly one line, got {lines}"
    assert lines[0]["decision"] == "past-floor", lines[0]
    assert lines[0]["diverged"] == 3, lines[0]


def test_the_canary_verdict_not_being_written_never_aborts_the_arm(tmp_path):
    """The block's whole standing principle: non-fatal by design.

    A persistence failure must not cost the sweep its measurement — the engine is
    already booted and the bench has not run yet. Here the runtime directory is
    not a directory at all, so neither the mkdir nor the append can succeed."""
    (tmp_path / "lloyd-data").write_text("not a directory\n", encoding="utf-8")
    r = _canary_run(tmp_path, run_rc=0, cmp_rc=1)
    assert r.returncode == 0, f"a verdict that cannot be written killed the arm\n{r.stderr}"
    assert "=== arm fp8kv done ===" in r.stdout, r.stdout
    assert "verdict NOT persisted" in r.stdout, (
        "an unpersisted verdict that is not reported is the silent-blind state "
        "this item exists to close: " + r.stdout)
    assert not (tmp_path / "lloyd-data/logs").exists()
