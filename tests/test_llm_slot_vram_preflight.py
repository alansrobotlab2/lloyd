#!/usr/bin/env python3
"""The GPU 2 free-VRAM preflight: what each slot start script does with the card.

#1316. GPU 2 is single-tenant — ``config.yaml``'s ``djev:`` comment says the card
carries the djev ranker or the llama.cpp secondary and never both, "an either/or,
not a pair of flags". Until this change only ``start-djev.sh`` checked the card:
``start-secondary.sh`` exec'd ``llama-server`` and let the allocation fail inside
``llama_model_load``, which on 2026-09-20 arrived in ``agent-llm-secondary.err`` as
five ``alloc_tensor_range: failed to allocate CUDA0 buffer of size 16294177280``
lines plus five ``exiting due to model loading error``, with supervisorctl's
``autorestart=true`` queueing each identical retry. A crash loop leaves an .err
file and hopes; a refusal leaves a sentence naming the flag to flip.

``tests/test_llm_slots.py`` owns the *config* half of that either/or —
``server.py``'s ``_sync_llm_slots`` refuses when both flags are enabled, and
``app/routers/services.py`` 409s the Start button when ``secondary_enabled`` is
false. Both are flag-level: neither has ever looked at the card. What a start that
bypasses the reconcile sees — a bare ``supervisorctl start agent-llm-secondary``, a
direct ``start-secondary.sh``, or a start issued mid-migration while the live config
still enabled both slots, which is exactly the 09-20 state — is only the VRAM
preflight, so that is the layer pinned here.

Nothing here touches a real GPU or a real engine. Each boot runs against a
PATH-shimmed ``nvidia-smi`` and a fake engine binary, and the guard's own
path-override defaults (``LLAMA_SERVER``, ``VLLM_VENV``, ``MODEL_DIR``,
``MODEL_FILE``, ``GPU_UTIL``, ``OVERHEAD_MIB``, and for the decider
``DECIDER_VENV`` and ``DECIDER_MODEL_DIR``) exist so the script can be driven to
its guard at all — before #1316 ``LLAMA_SERVER`` was hardcoded, which left no way to
reach the check without building llama.cpp and letting it try.

Three launchers share GPU 2 and are driven here. Two of them are slots the backend
knows: the secondary and djev, whose either/or ``_sync_llm_slots`` enforces at the
flag level. The third is ``start-decider.sh`` (#2716), which no flag can arm — it is
absent from ``app/llm_slots.py::_SLOTS`` and from ``config.yaml``, and its own unit
says so (``agent-services/supervisor/conf.d/agent-decider.conf:5-8``: "started and
stopped by hand", serving ``:8012``). For that launcher the preflight is not one of
two guards, it is the only one: nothing above it can see the card or the other
tenants, so its guard is pinned from the same mocked reading as the other two.

Design points this suite deliberately pins:

1. the refusal prints both figures and the either/or, on BOTH exec branches — a
   guard on the llama.cpp path only would leave the vLLM branch to crash exactly as
   before (``--gpu-memory-utilization 0.90`` against an occupied card is the same
   ``alloc_tensor_range`` failure);
2. the need is *measured* — GGUF bytes + KV + overhead for llama.cpp, weights +
   headroom for the decider, and for vLLM the same ``GPU_UTIL`` variable that feeds
   the flag, never a transcribed comment (the ~21.7 GiB in the script's budget
   comment, and the 19.6 GiB in the decider's, are prose);
3. all three scripts share the reading via ``agent-services/bin/gpu-mem.sh``, so
   they cannot disagree about one card;
4. a sufficient reading still boots — the shipped default must clear its own guard,
   or the guard has silently disabled the slot;
5. an unmeasurable card must NOT refuse for the *optional secondary*: a driver fault
   must not be promoted into an engine outage on the one slot whose off state is the
   designed one. ``start-djev.sh`` is the opposite case, and ``start-decider.sh``
   takes the same posture as djev and for the same reason — a hand start onto a card
   someone believes they just freed. Both still fire on the same mocked reading.

Run: pytest tests/test_llm_slot_vram_preflight.py -q   (no GPU, no vLLM)
"""

import os
import re
import stat as statmod
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BINDIR = ROOT / "agent-services" / "bin"
START_SECONDARY = BINDIR / "start-secondary.sh"
START_DJEV = BINDIR / "start-djev.sh"
START_DECIDER = BINDIR / "start-decider.sh"
GPU_MEM = BINDIR / "gpu-mem.sh"

# The shipped llama.cpp defaults: MAX_LEN 262144 at 20 KiB/token (the KV term the
# script's own budget table computes for its 10 full-attention layers), plus the
# 1024 MiB the script budgets for compute buffers and CUDA graphs.
DEFAULT_MAX_LEN = 262144
DEFAULT_KV_MIB = DEFAULT_MAX_LEN * 20 // 1024          # 5120
DEFAULT_OVERHEAD_MIB = 1024
# Clause 4's shipped-default bound: weights + KV + slack <= 23233 on a 24576 card.
# 16065 + 5120 + 1024 = 22209 — 1024 MiB more slack than the comment's ~21.7 GiB,
# and still 2367 MiB under the whole card.
CLAUSE4_WEIGHTS_MIB = 16065
CLAUSE4_MAX_TOTAL_MIB = CLAUSE4_WEIGHTS_MIB + DEFAULT_KV_MIB + 2048   # 23233
CLAUSE4_CARD_MIB = 24576
# What 0.90 of that card is, rounded up: the vLLM branch's need.
VLLM_NEED_MIB = 22119

GPU2 = "2"
DJEV_HOLDER = "2063085, 24010 MiB, VLLM::EngineCore"
SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# ── The decider (#2716) ────────────────────────────────────────────────
# The fake checkpoint's apparent weight total, in MiB. It stands in for the shipped
# 35b's weights without the 19.6 GiB download: `du -sm --apparent-size` of one
# truncated file is what the script's own need is built from (:140-142), so any
# figure here is one the guard can be tested against — and 19600 is close enough to
# the real budget that the shipped default's fit against a real card is the fit
# being asserted.
DECIDER_WEIGHTS_MIB = 19600
DECIDER_HEADROOM_MIB = 2048         # HEADROOM_MIB's shipped default (:101)
DECIDER_NEED_MIB = DECIDER_WEIGHTS_MIB + DECIDER_HEADROOM_MIB          # 21648
DECIDER_SPARE_ON_CARD_MIB = CLAUSE4_CARD_MIB - DECIDER_NEED_MIB        # 2928
# What the unit's `command=` line and the script's own `${PORT:-8012}` agree on
# (agent-services/supervisor/conf.d/agent-decider.conf:1, start-decider.sh:91).
DECIDER_PORT = "8012"
# The `die`s that fire before the card guard is reached: no venv (:134), the import
# probe (:135-136), the checkpoint's two JSON files (:137-138), the weight sum
# (:141). Any of these exits 2 like a refusal does, so a boot that died on one is
# indistinguishable from a refusal unless the messages are checked for. The fourth is
# listed for detection rather than for use: `set -e -o pipefail` aborts the `du`
# pipeline at :140 before :141 can print, so a no-shard checkpoint currently exits 1
# silently — see the last node in this file.
DECIDER_PREFLIGHT_DIES = ("no venv at", "does not import", "no checkpoint at",
                          "no safetensors under")


def _executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip())
    path.chmod(path.stat().st_mode | statmod.S_IEXEC | statmod.S_IXGRP | statmod.S_IXOTH)
    return path


def _need(weights_mib, max_len=DEFAULT_MAX_LEN, overhead=DEFAULT_OVERHEAD_MIB):
    return weights_mib + (max_len * 20 // 1024) + overhead


SHIM = """
    #!/bin/bash
    [ -n "${SMI_LOG:-}" ] && echo "call $*" >> "$SMI_LOG"
    case "$*" in
      *--query-compute-apps*)
          [ -n "${SMI_APPS:-}" ] && printf '%s\n' "$SMI_APPS"
          exit 0 ;;
      *memory.free*)
          [ -n "${SMI_FAIL:-}" ]  && exit 1
          [ -n "${SMI_EMPTY:-}" ] && { printf '\\n'; exit 0; }
          [ -z "${SMI_FREE:-}" ]  && exit 1
          printf '%s\n' "$SMI_FREE"; exit 0 ;;
      *memory.total*)
          [ -n "${SMI_FAIL:-}" ] && exit 1
          printf '%s\n' "${SMI_TOTAL:-0}"; exit 0 ;;
      *--query-gpu=name*)
          [ -n "${SMI_FAIL:-}" ] && exit 1
          printf '%s\n' "${SMI_NAME:-NVIDIA GeForce RTX 3090}"; exit 0 ;;
    esac
    echo "stub nvidia-smi: unexpected $*" >&2
    exit 1
"""


@pytest.fixture
def shim_dir(tmp_path):
    """The fake `bin` directory the boots put first on PATH."""
    d = tmp_path / "shim-bin"
    d.mkdir(exist_ok=True)
    _executable(d / "nvidia-smi", SHIM)
    return d


@pytest.fixture
def shim(tmp_path, shim_dir):
    """Canned nvidia-smi answers, as env for a boot.

    `free=20` is the live reading taken at triage: GPU 2 at 20 MiB free of 24576,
    `VLLM::EngineCore` the sole compute app. `fail` is nvidia-smi itself failing,
    `empty` is the rarer and nastier one — nvidia-smi answering status 0 with
    nothing numeric in the body, which `N/A` and `ERROR` also do.
    """
    def make(free=None, total=CLAUSE4_CARD_MIB, apps=DJEV_HOLDER,
             fail=None, empty=None, name="NVIDIA GeForce RTX 3090"):
        return {
            "PATH": f"{shim_dir}:{SYSTEM_PATH}",
            "SMI_LOG": str(tmp_path / "smi-calls.log"),
            "SMI_FREE": "" if free is None else str(free),
            "SMI_TOTAL": str(total),
            "SMI_NAME": name,
            "SMI_APPS": apps or "",
            "SMI_FAIL": "" if fail is None else str(fail),
            "SMI_EMPTY": "" if empty is None else str(empty),
        }
    return make


@pytest.fixture
def boot(tmp_path, shim_dir):
    """Runs one slot start script against fakes; reports rc, output, exec markers.

    Every path the script would otherwise resolve inside the repo is pointed at a
    fake under tmp_path, which is the only reason a guard can be tested at all: a
    16 GiB GGUF and a built llama.cpp are not test fixtures.
    """
    def run(script=START_SECONDARY, *, model="qwen36", env_extra=None, overrides=None,
            direct=False):
        home = tmp_path / "home"
        gguf = home / "model.gguf"
        # Clause 4's shipped-default weight size: exactly 16065 MiB, so the need the
        # guard computes for it is 16065 + KV + overhead.
        gguf.parent.mkdir(parents=True, exist_ok=True)
        gguf.write_bytes(b"")
        os.truncate(gguf, CLAUSE4_WEIGHTS_MIB * 1024 * 1024)

        model_dir = home / "model-dir"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text('{"architectures": []}')

        marker = tmp_path / "execed.txt"
        engine = _executable(home / "llama.cpp" / "build" / "bin" / "llama-server", f"""
            #!/bin/bash
            echo "llama-server $*" >> {marker}
            exit 0
        """)
        fake_venv = home / "vllm-venv"
        _executable(fake_venv / "bin" / "python", f"""
            #!/bin/bash
            echo "python $*" >> {marker}
            exit 0
        """)

        env = {
            "HOME": str(home),
            "PATH": f"{shim_dir}:{SYSTEM_PATH}",
            "MODEL": model,
            "GPU": GPU2,
            "PORT": "8091",
            "MODEL_FILE": str(gguf),
            "MODEL_DIR": str(model_dir),
            "LLAMA_SERVER": str(engine),
            "VLLM_VENV": str(fake_venv),
        }
        if env_extra:
            env.update(env_extra)
        if overrides:
            env.update(overrides)
        # `direct=True` execves the script by its shebang instead of handing it to
        # bash, which is the only way a test can see the file's mode bit. Supervisord
        # uses the /bin/bash form, so a lost 100755 changes nothing for it and every
        # other test in this file stays green while `./start-secondary.sh` — a start
        # path named in the script's own Usage header — fails with EACCES.
        argv = [str(script)] if direct else ["/bin/bash", str(script)]
        proc = subprocess.run(
            argv, env=env, cwd=str(tmp_path),
            capture_output=True, text=True, timeout=90,
        )
        return {
            "rc": proc.returncode,
            "out": proc.stdout + proc.stderr,
            "execed": marker.read_text() if marker.exists() else "",
        }
    return run


@pytest.fixture
def decider_fakes(tmp_path):
    """A fake decider install: the four files that stand between `start-decider.sh`
    and its card guard, none of which exists on this box.

    The script dies three times before it reads the card — ``[[ -x "$PY" ]]`` (:134),
    ``$PY -c "import decider.serve_vllm"`` (:135-136) and the checkpoint test
    (:137-138) — and a fourth if the weight sum comes out zero (:141). Two of those
    are not satisfiable by the ``boot`` fixture's fakes: the checkpoint test wants
    ``decider_config.json`` as well as ``config.json``, and the weight sum needs a
    real ``*.safetensors`` for ``du`` to size. So this builds all four:

      * ``$DECIDER_VENV/bin/python`` — executable, and silent. The script runs it
        once, for the import probe; it writes no marker, so ``execed`` below means
        the *serve* step was reached and nothing else.
      * ``$DECIDER_VENV/bin/uvicorn`` — the fake engine. The script ``exec``s this
        path by name (:181), so a refusal that reached it would be visible.
      * ``$DECIDER_MODEL_DIR/config.json`` + ``decider_config.json``, and one
        ``*.safetensors`` truncated to ``weights_mib`` — sparse, so a 19600 MiB
        apparent size costs no disk and no 19.6 GiB download, and ``du -sm
        --apparent-size`` reads back exactly the number the test asked for.

    The two overrides are the whole route in: nothing here consults
    ``.venvs/vllm-decider`` or ``llm/models``, both absent on this box, and
    ``test_the_decider_never_looks_outside_its_fakes`` pins that by checking the
    paths the script actually served and died on are the tmp ones.
    """
    def build(*, weights_mib=DECIDER_WEIGHTS_MIB):
        home = tmp_path / "decider-home"
        venv = home / "decider-venv"
        model = home / "decider-model"
        marker = tmp_path / "decider-exec.txt"
        if marker.exists():
            marker.unlink()
        model.mkdir(parents=True, exist_ok=True)
        (model / "config.json").write_text('{"architectures": ["DeciderForCausalLM"]}')
        (model / "decider_config.json").write_text('{"layout": "plain"}')
        weights = model / "model-00001-of-00001.safetensors"
        weights.write_bytes(b"")
        os.truncate(weights, weights_mib * 1024 * 1024)
        _executable(venv / "bin" / "python", "#!/usr/bin/env bash\nexit 0\n")
        _executable(venv / "bin" / "uvicorn", f"""
            #!/bin/bash
            # The serve step's own argv plus the two env vars the script exports for
            # the app: the checkpoint reaches the engine through DECIDER_MODEL, not
            # through a --model flag, so without this line nothing in the transcript
            # could say which checkpoint this boot would have loaded.
            echo "uvicorn $* | DECIDER_MODEL=${{DECIDER_MODEL:-}} V=${{DECIDER_VARIANT:-}}" >> {marker}
            exit 0
        """)
        return {"home": home, "venv": venv, "model": model, "marker": marker}
    return build


@pytest.fixture
def boot_decider(decider_fakes, shim_dir):
    """Boots ``start-decider.sh`` against those fakes and the shimmed ``nvidia-smi``.

    Reports ``rc``, the merged output, what the fake engine was exec'd with, and
    whether the run ever got past the script's own preflight dies — which matters
    because every one of those dies exits 2 exactly like the refusal does, so a
    fixture that quietly stopped building its fakes would look like a passing refusal
    test. The PATH deliberately carries the shim first: the script prepends its own
    ``$VENV/bin`` and ``/opt/cuda/bin``, neither of which has an ``nvidia-smi`` here,
    and a real one on the path would read this box's actual card and make every
    figure below whatever the tenants happened to be using.
    """
    def run(*, weights_mib=DECIDER_WEIGHTS_MIB, env_extra=None, overrides=None,
            direct=False, broken=None, script=START_DECIDER):
        fakes = decider_fakes(weights_mib=weights_mib)
        env = {
            "HOME": str(fakes["home"]),
            "PATH": f"{shim_dir}:{SYSTEM_PATH}",
            "GPU": GPU2,
            "PORT": DECIDER_PORT,
            "DECIDER_VENV": str(fakes["venv"]),
            "DECIDER_MODEL_DIR": str(fakes["model"]),
        }
        if env_extra:
            env.update(env_extra)
        if overrides:
            env.update(overrides)
        argv = [str(script)] if direct else ["/bin/bash", str(script)]
        proc = subprocess.run(argv, env=env, cwd=str(fakes["home"]),
                              capture_output=True, text=True, timeout=90)
        out = proc.stdout + proc.stderr
        die = next((d for d in DECIDER_PREFLIGHT_DIES if d in out), None)
        if broken is not None:
            assert die == broken, (
                f"expected the run to die on the {broken!r} preflight, got "
                f"die={die!r} rc={proc.returncode}: {out[-500:]}")
        return {
            "rc": proc.returncode,
            "out": out,
            "execed": (fakes["marker"].read_text()
                       if fakes["marker"].exists() else ""),
            "preflight_die": die,
        }
    return run


def _reached_the_guard(r):
    """Fail unless the boot got past the preflight and actually ran the card guard.

    Without this an assert on ``rc == 2`` is satisfied by a missing fake checkpoint
    just as well as by a refusal, and the guard could regress to a no-op with the
    whole decider suite green.
    """
    assert r["preflight_die"] is None, (
        f"the run died in the preflight ({r['preflight_die']!r}) instead of reaching "
        f"the card guard, so it proved nothing about it: {r['out'][-500:]}")
    return r


def _helper_probe(body: str, tmp_path, path_dirs, shim_env=None):
    """Sources gpu-mem.sh in a bare bash and runs `body`, reporting rc and stdout."""
    env = {**os.environ, "PATH": ":".join(path_dirs)}
    if shim_env:
        env.update(shim_env)
    r = subprocess.run(["/bin/bash", "-c", f'source "{GPU_MEM}"; {body}'],
                       env=env, cwd=str(tmp_path), capture_output=True,
                       text=True, timeout=30)
    return {"rc": r.returncode, "out": r.stdout + r.stderr}


# ---------------------------------------------------------------------------
# Clause 1: too little free VRAM -> non-zero, both figures printed, and NO exec —
# on both exec branches (llama.cpp and vLLM).
# ---------------------------------------------------------------------------

def test_llama_branch_refuses_and_never_execs(shim, boot):
    """GPU 2 held by djev: refuse, print both figures, do not exec llama-server."""
    r = boot(env_extra=shim(free=20))
    need = _need(CLAUSE4_WEIGHTS_MIB)
    assert need == 22209, f"the shipped default's need moved: {need}"
    assert r["rc"] == 2, f"guard did not refuse on 20 MiB free (rc={r['rc']}): {r['out'][-600:]}"
    assert not r["execed"], f"refused but still exec'd llama-server: {r['execed']}"
    # Both figures on the refusal line itself, so whoever reads supervisorctl's tail
    # sees the shortfall without hunting the log for the card line.
    assert "has 20 MiB free, this needs 22209" in r["out"], r["out"][-600:]
    # And each figure is also labelled, so neither number is ambiguous about which
    # side of the comparison it sits on.
    assert "20 MiB free" in r["out"] and "needs          22209 MiB" in r["out"]
    assert "24576 MiB total" in r["out"]


def test_vllm_branch_refuses_too(shim, boot):
    """MODEL != qwen36 execs vLLM instead, and must hit the same gate.

    Leaving the second branch unguarded would be the same bug in a different hat:
    ``--gpu-memory-utilization 0.90`` on an occupied card is the identical
    ``alloc_tensor_range`` crash. The need is 0.90 of the card the guard just read,
    so on a 24576 MiB card it is 22119 (0.90 x 24576 = 22118.4, rounded up).
    """
    r = boot(model="qwen35", env_extra=shim(free=20))
    assert r["rc"] == 2, f"vLLM branch did not refuse (rc={r['rc']}): {r['out'][-600:]}"
    assert not r["execed"], f"vLLM branch refused but exec'd anyway: {r['execed']}"
    assert f"has 20 MiB free, this needs {VLLM_NEED_MIB}" in r["out"], r["out"][-600:]


def test_a_start_one_mib_short_still_refuses(shim, boot):
    """The boundary is ``free < need``, pinned from both sides at once.

    One MiB below the need refuses; exactly the need boots. Only the refuse side is
    shown by the card-is-full tests, so without the equal case an accidental ``<=``
    — which would refuse the exact-fit boot this slot is designed to make — passes
    the rest of the suite untouched.
    """
    need = _need(CLAUSE4_WEIGHTS_MIB)
    r = boot(env_extra=shim(free=need - 1))
    assert r["rc"] == 2 and not r["execed"], f"one MiB short must refuse: rc={r['rc']}"
    assert f"has {need - 1} MiB free, this needs {need}" in r["out"]

    exact = boot(env_extra=shim(free=need))
    assert exact["rc"] == 0, f"exactly enough must boot, not refuse: rc={exact['rc']} {exact['out'][-500:]}"
    assert "llama-server" in exact["execed"], f"exactly enough reached no exec: {exact['out'][-500:]}"
    assert "refusing to start" not in exact["out"]


def test_the_vram_need_and_the_gpu_flag_move_on_one_variable(shim, boot):
    """The guard's figure and the flag the engine is handed are one value, not two.

    ``start-djev.sh`` approved nine OOM boots on this card because its preflight
    cost 2.6 GiB less than the ``GPU_UTIL`` it passed vLLM (#1355). Asserting the
    literal ``0.90`` in both places proves nothing: a hardcoded flag and a
    ``GPU_UTIL``-driven need read identically at the default. So the variable is
    moved and both sides must follow — at 0.50 the need is ceil(0.50 x 24576) =
    12288, exactly which also pins that the ceiling is a ceiling and not a
    ceiling-plus-one, and the exec line reads 0.50.
    """
    low = boot(model="qwen35", env_extra=shim(free=20), overrides={"GPU_UTIL": "0.50"})
    assert low["rc"] == 2, f"GPU_UTIL did not move the refusal (rc={low['rc']}): {low['out'][-500:]}"
    assert "has 20 MiB free, this needs 12288" in low["out"], low["out"][-500:]

    exact = boot(model="qwen35", env_extra=shim(free=12288), overrides={"GPU_UTIL": "0.50"})
    assert exact["rc"] == 0, f"a 0.50 start on exactly its need was blocked: {exact['out'][-500:]}"
    assert "--gpu-memory-utilization 0.50" in exact["execed"], \
        f"the exec flag did not follow GPU_UTIL: {exact['execed']}"


def test_util_one_on_an_empty_card_is_not_a_refusal(shim, boot):
    """The vLLM need is a ceiling, not a ceiling-plus-one.

    The first cut here costed the branch ``int(u x total) + 1``, which at u=1.0 asks
    for one MiB more than the card can ever offer: a guard refusing a boot the
    hardware can serve, the opposite failure to #1316 and one no test would have
    caught at the shipped 0.90. 24576 free against ceil(1.00 x 24576) = 24576 gets
    past the strict ``<``.
    """
    r = boot(model="qwen35", env_extra=shim(free=CLAUSE4_CARD_MIB),
             overrides={"GPU_UTIL": "1.0"})
    assert r["rc"] == 0, f"1.0 on an empty card was refused (rc={r['rc']}): {r['out'][-500:]}"
    assert "--gpu-memory-utilization 1.0" in r["execed"], r["execed"]


# ---------------------------------------------------------------------------
# Clause 2: the refusal names the other tenant, the either/or and the flag.
# ---------------------------------------------------------------------------

def test_refusal_names_the_other_tenant_and_the_flag(shim, boot):
    """The message must read as the fix, not the symptom.

    Modelled on the line ``start-djev.sh`` has carried since 2026-09-20: name the
    either/or, name the other slot, name the flag to flip, and name who normally
    arbitrates the pair.
    """
    out = boot(env_extra=shim(free=20))["out"]
    assert "agent-djev" in out, "must name the other GPU 2 tenant"
    assert "either/or" in out, "must state the either/or"
    assert "secondary_enabled: true" in out, "must name the config flag to flip"
    assert "djev.enabled: false" in out, "and the counterpart flag, in that order"
    assert "_sync_llm_slots" in out, "must say who normally starts the slots"
    assert DJEV_HOLDER in out, "must show what is holding the card"


# ---------------------------------------------------------------------------
# Clause 4: a sufficient reading still boots, on both branches.
# ---------------------------------------------------------------------------

def test_sufficient_free_vram_still_execs_the_llama_branch(shim, boot):
    """The shipped default clears its own guard on a whole 24576 MiB card.

    16065 (weights) + 5120 (KV at MAX_LEN 262144) + 1024 (overhead) = 22209, inside
    clause 4's 23233 bound. A guard that refused this boot would not be a guard, it
    would be a silently disabled slot.
    """
    assert _need(CLAUSE4_WEIGHTS_MIB) <= CLAUSE4_MAX_TOTAL_MIB
    r = boot(env_extra=shim(free=CLAUSE4_CARD_MIB))
    assert r["rc"] == 0, f"a legitimate start was blocked: rc={r['rc']} {r['out'][-600:]}"
    assert "llama-server" in r["execed"], f"never reached the exec: {r['out'][-600:]}"
    assert "--ctx-size 262144" in r["execed"]
    assert "refusing to start" not in r["out"]


def test_sufficient_free_vram_still_execs_the_vllm_branch(shim, boot):
    """0.90 of 24576 is 22119 MiB, so a whole card satisfies it and vLLM execs.

    The literal here is the shipped default reaching the engine unchanged; that the
    guard budgets the same number is pinned by
    ``test_the_vram_need_and_the_gpu_flag_move_on_one_variable``, which moves
    ``GPU_UTIL`` and requires both sides to follow — a fixed 0.90 in each place
    proves nothing on its own.
    """
    r = boot(model="qwen35", env_extra=shim(free=CLAUSE4_CARD_MIB))
    assert r["rc"] == 0, f"legitimate vLLM start blocked: rc={r['rc']} {r['out'][-600:]}"
    assert "python -m vllm.entrypoints.openai.api_server" in r["execed"], r["execed"]
    assert "--gpu-memory-utilization 0.90" in r["execed"], \
        f"the shipped default flag changed: {r['execed']}"


def test_need_follows_the_measured_weights_not_a_comment(shim, boot):
    """A different GGUF size moves the need, so the figure cannot be transcribed.

    ``start-djev.sh`` derives its need from ``du`` of the weights rather than "a
    number in a comment that a re-quantized checkpoint would silently invalidate";
    this is the same rule applied to llama.cpp. The 16294177280 B in the .err is one
    CUDA0 tensor range llama.cpp failed on, never the boot's demand, and the ~21.7
    GiB in the budget comment is prose.
    """
    default_out = boot(env_extra=shim(free=20))["out"]
    assert f"needs          {_need(CLAUSE4_WEIGHTS_MIB)} MiB" in default_out
    assert "21.7" not in default_out and "16294177280" not in default_out, \
        "the comment's GiB figure or the .err's buffer size must not become the need"

    half = boot(env_extra=shim(free=20),
                overrides={"OVERHEAD_MIB": "1024", "MAX_LEN": "131072"})
    assert f"needs          {_need(CLAUSE4_WEIGHTS_MIB, max_len=131072)} MiB" in half["out"], \
        "MAX_LEN must move the KV term: the window is what the KV term multiplies"


# ---------------------------------------------------------------------------
# Clause 5: an unmeasurable card warns and BOOTS, it does not refuse.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("broken", ["fail", "nonnumeric", "empty"],
                         ids=["nvidia-smi-exits-nonzero", "nvidia-smi-non-numeric",
                              "nvidia-smi-replies-nothing"])
def test_unreadable_card_warns_and_boots(shim, boot, broken):
    """A driver fault must not be promoted into an engine outage on an optional slot.

    ``start-secondary.sh`` deliberately does the opposite of ``start-djev.sh`` here:
    that one is the live recall ranker and stays down, this one is optional
    (``secondary_enabled: false`` is the designed state and app/secondary_models.py
    routes post-session jobs to the primary around it). Booting blind is what this
    script did before any guard existed, so the fallback is the old behaviour, not a
    new risk.
    """
    kw = {"fail": 1} if broken == "fail" else (
        {"free": "N/A"} if broken == "nonnumeric" else {"empty": 1})
    r = boot(env_extra=shim(**kw))
    assert r["rc"] == 0, f"guard turned a measurement failure into an outage: rc={r['rc']} {r['out'][-600:]}"
    assert "llama-server" in r["execed"], "must still boot the engine"
    out = r["out"].lower()
    assert "warning" in out, "the unread card must be named as a warning"
    assert "preflight" in out and "gpu 2" in out, r["out"][-400:]


def test_unreadable_card_cannot_fabricate_a_measurement(shim, boot):
    """The warning must not print a number it never read.

    The failure this pins is a guard that treats an unread card as an empty one:
    ``free=0`` is below every need, so clause 5 would look satisfied by arithmetic
    alone while the outage shipped looking like a full card.
    """
    out = boot(env_extra=shim(fail=1))["out"]
    assert "MiB free" not in out, f"printed a free figure it cannot have read: {out[-600:]}"
    assert "refusing to start" not in out


# ---------------------------------------------------------------------------
# Clause 3: one shared helper, and the djev refusal still fires on it.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The decider (#2716): the third GPU-2 tenant, the one no flag can arm.
# ---------------------------------------------------------------------------

def test_the_decider_refuses_a_card_held_by_another_tenant(shim, boot_decider):
    """Clause 1 of #2716: a held card means a refusal, both figures, and no exec.

    This is the launcher the other 18 nodes cannot see. It has no flag in
    ``config.yaml``, is absent from ``app/llm_slots.py::_SLOTS``, and its own unit
    says it "is started and stopped by hand"
    (``agent-services/supervisor/conf.d/agent-decider.conf:5-8``), so
    ``server.py``'s ``_sync_llm_slots`` never arbitrates it and the VRAM check is the
    only thing between a hand ``supervisorctl start agent-decider`` and a second
    tenant on a single-tenant card. Nothing above it would notice the regression; a
    vLLM ``alloc_tensor_range`` crash on an occupied card is what "noticing" looks
    like here, which is the failure #1316 was filed against for the secondary.
    """
    r = boot_decider(env_extra=shim(free=20))
    _reached_the_guard(r)
    assert r["rc"] == 2, f"guard did not refuse on 20 MiB free (rc={r['rc']}): {r['out'][-600:]}"
    assert not r["execed"], f"refused but still exec'd the engine: {r['execed']}"
    # Both figures on the refusal line, so the tail of supervisord's log names the
    # shortfall without the reader hunting the banner for the card line.
    assert (f"has 20 MiB free, this needs {DECIDER_NEED_MIB}" in r["out"]), r["out"][-600:]
    assert f"({20} MiB free of {CLAUSE4_CARD_MIB})" in r["out"], r["out"][-600:]
    # The one-tenant rule, and the concrete way out of it: unlike the secondary's
    # refusal this one cannot name a flag of its own (it has none), so the fix it
    # offers is the other tenant's flag or its stop command.
    assert "GPU 2 holds one tenant" in r["out"], r["out"][-600:]
    assert "djev.enabled: false" in r["out"], "must name the counterpart flag to flip"
    assert "supervisorctl stop agent-djev" in r["out"], "or the hand stop that frees the card"
    assert DJEV_HOLDER in r["out"], "must show what is holding the card"


def test_the_deciders_need_is_measured_from_the_checkpoint(shim, boot_decider):
    """Clause 2 of #2716: need = measured weight bytes + HEADROOM_MIB, never a comment.

    ``start-decider.sh``'s header budgets "NVFP4 weights, resident as Marlin 19.6
    GiB", and a guard that costed that sentence would keep working right up to the
    checkpoint swap that made it wrong — the same defect #1316 named in the
    secondary's ~21.7 GiB comment. So the figure is moved twice here: a different
    fake weight total must move the printed need, and so must ``HEADROOM_MIB``.
    """
    default = boot_decider(env_extra=shim(free=20))
    _reached_the_guard(default)
    assert f"weights {DECIDER_WEIGHTS_MIB} + headroom {DECIDER_HEADROOM_MIB}" \
           f" = {DECIDER_NEED_MIB} MiB" in default["out"], default["out"][-600:]

    lighter_mib = 12000
    lighter = boot_decider(weights_mib=lighter_mib, env_extra=shim(free=20))
    _reached_the_guard(lighter)
    assert f"this needs {lighter_mib + DECIDER_HEADROOM_MIB}." in lighter["out"], \
        "the need must follow the checkpoint's measured size: " + lighter["out"][-600:]
    assert "19.6" not in lighter["out"], "the header's GiB prose must not be the need"

    tall_headroom = boot_decider(env_extra=shim(free=20),
                                 overrides={"HEADROOM_MIB": "4096"})
    _reached_the_guard(tall_headroom)
    assert f"this needs {DECIDER_WEIGHTS_MIB + 4096}." in tall_headroom["out"], \
        "HEADROOM_MIB must be the other term: " + tall_headroom["out"][-600:]


def test_a_decider_start_one_mib_short_refuses_and_an_exact_one_boots(shim,
                                                                      boot_decider):
    """Clause 2's boundary from both sides: ``free < need``, not ``<=``.

    One MiB below the need refuses; a reading exactly equal to it boots. The
    refuse-only tests would still pass if the comparison became ``<=``, and the
    failure that ships then is a launcher that refuses the exact-fit start it exists
    to make — the same off-by-one #1316's review caught on the secondary's vLLM
    ceiling, arriving from the other direction.
    """
    short = boot_decider(env_extra=shim(free=DECIDER_NEED_MIB - 1))
    _reached_the_guard(short)
    assert short["rc"] == 2 and not short["execed"], (
        f"one MiB short of the need must refuse: rc={short['rc']} "
        f"execed={short['execed']!r}")
    assert f"has {DECIDER_NEED_MIB - 1} MiB free, this needs {DECIDER_NEED_MIB}" \
        in short["out"], short["out"][-600:]

    exact = boot_decider(env_extra=shim(free=DECIDER_NEED_MIB))
    _reached_the_guard(exact)
    assert exact["rc"] == 0, (
        f"a start with exactly its need was refused: rc={exact['rc']} "
        f"{exact['out'][-500:]}")
    assert "uvicorn" in exact["execed"], f"exact fit never reached the exec: {exact['out'][-500:]}"
    assert "refusing to start" not in exact["out"]


@pytest.mark.parametrize("broken", ["fail", "nonnumeric", "empty"],
                         ids=["nvidia-smi-exits-nonzero", "nvidia-smi-non-numeric",
                              "nvidia-smi-replies-nothing"])
def test_an_unmeasurable_card_kills_the_decider_instead_of_booting_it(
        shim, boot_decider, broken):
    """Clause 3 of #2716: here the decider takes djev's side, not the secondary's.

    Design point 5 (top of this file) lets ``start-secondary.sh`` warn and boot on an
    unreadable card, because that slot is optional and ``secondary_enabled: false``
    is a designed state — a driver fault must not be promoted into an engine outage
    there. The decider's posture is the opposite one, and it is the defensible
    choice rather than an oversight: it has no flag and no designed-off state, so
    the only time it is ever started is by hand, onto a card somebody believes they
    have just freed by stopping djev. A fabricated measurement at that moment is the
    one failure that double-books a single-tenant card, and nothing above this script
    is there to notice. So an unmeasurable card is a death, as it is for
    ``start-djev.sh`` — which is why this asserts the die and not a warning.
    """
    kw = ({"fail": 1} if broken == "fail" else
          {"free": "N/A"} if broken == "nonnumeric" else {"empty": 1})
    r = boot_decider(env_extra=shim(**kw))
    _reached_the_guard(r)
    assert r["rc"] == 2, f"booted on an unread card (rc={r['rc']}): {r['out'][-500:]}"
    assert "no readable gpu 2" in r["out"].lower(), r["out"][-500:]
    assert not r["execed"], f"refused the reading and exec'd anyway: {r['execed']}"
    assert "MiB free" not in r["out"], (
        "printed a free figure it never read: " + r["out"][-500:])
    assert "holds one tenant" not in r["out"], (
        "the refusal advice is for a card it measured; a death names no holders")


def test_the_deciders_shipped_default_clears_its_own_guard(shim, boot_decider):
    """Clause 4 of #2716: a whole card free, the default reaches its serve step.

    A guard that refused every legitimate boot would keep this suite green — every
    refusal assertion above would still hold — so this is the node that tells a
    guard from a disabled launcher. It also carries the arithmetic the launcher
    exists to do: 19600 MiB of weights plus 2048 of headroom is 21648 against a
    24576 MiB card, which fits with 2928 MiB to spare, so a real 35b start on a card
    somebody has just freed is not refused by its own budget.

    The exec is asserted down to ``decider.serve_vllm:app`` and ``--port 8012``,
    because the whole point of reaching the serve step is the engine line: the unit
   's ``command=`` runs this script, so the port the script defaults to is the port
    anything could later read as "the decider is up".
    """
    assert DECIDER_NEED_MIB == 21648 and DECIDER_SPARE_ON_CARD_MIB == 2928
    r = boot_decider(env_extra=shim(free=CLAUSE4_CARD_MIB))
    _reached_the_guard(r)
    assert r["rc"] == 0, f"a legitimate start was blocked: rc={r['rc']} {r['out'][-600:]}"
    assert "refusing to start" not in r["out"]
    assert "uvicorn decider.serve_vllm:app" in r["execed"], r["execed"]
    assert f"--port {DECIDER_PORT}" in r["execed"], \
        f"the shipped default port changed: {r['execed']}"


def test_the_decider_never_looks_outside_its_fakes(shim, boot_decider, tmp_path):
    """Every new node above reached the guard through the two overrides, and nothing else.

    This box has no ``.venvs/vllm-decider`` and no checkpoint under ``llm/models``
    (#2713's hazard (a)), so if a fake ever stopped being wired up the script would
    fall back to those defaults, die with "no checkpoint at ...", exit 2, and every
    refusal test here would read that as its own passing refusal. So this pins the
    two things that make the suite honest: the checkpoint the script served is the
    tmp one — the ``DECIDER_MODEL`` the fake engine reports being handed is the
    witness, since the banner prints the model's *size*, never its path, and the
    exec line carries no ``--model`` flag — and the preflight deaths before the guard
    are still reachable, each naming the tmp path it looked for rather than a repo
    default. A run that dies early has to say so out loud; exiting 2 in character
    would let every refusal node below pass on a broken fixture.
    """
    r = boot_decider(env_extra=shim(free=CLAUSE4_CARD_MIB, apps=""))
    assert r["rc"] == 0, r["out"][-400:]
    fake_model = str(tmp_path / "decider-home" / "decider-model")
    assert f"DECIDER_MODEL={fake_model} " in r["execed"], (
        "the checkpoint the serve step was handed must be the fake one: "
        + r["execed"][-400:])
    for real in ("vllm-decider", "llm/models", "Mapika-decider"):
        assert real not in r["out"] + r["execed"], (
            f"consulted the real {real} path: {(r['out'] + r['execed'])[-500:]}")

    missing_venv = tmp_path / "no-such-venv"
    missing_model = tmp_path / "no-such-model"
    missing_model.mkdir()
    empty_model = tmp_path / "empty-model"
    empty_model.mkdir()
    (empty_model / "config.json").write_text("{}")
    (empty_model / "decider_config.json").write_text("{}")
    silent_venv = tmp_path / "silent-venv"
    _executable(silent_venv / "bin" / "python", "#!/usr/bin/env bash\nexit 1\n")

    for overrides, die in (
            ({"DECIDER_VENV": str(missing_venv)}, "no venv at"),
            ({"DECIDER_VENV": str(silent_venv)}, "does not import"),
            ({"DECIDER_MODEL_DIR": str(missing_model)}, "no checkpoint at")):
        dead = boot_decider(broken=die, overrides=overrides)
        assert dead["rc"] != 0 and not dead["execed"], (
            f"{die} must stop the script before any engine: {dead['out'][-400:]}")
        assert str(tmp_path) in dead["out"], (
            f"{die} must name the path it was pointed at, not a repo default: "
            + dead["out"][-400:])

    # The fourth preflight — a checkpoint dir whose glob finds no weight shard — is
    # asserted fail-closed only, because that is all it is today. `set -e -o
    # pipefail` aborts on the failing `du … | awk` inside the command substitution at
    # :140, so the `die "no safetensors under $MODEL"` at :141 never runs for the case
    # it was written for: the script exits 1 having printed nothing. It still cannot
    # reach an engine, which is the property this suite owns. The missing diagnosis is
    # a launcher defect recorded on #2716, not something to pin as intended.
    no_shards = boot_decider(overrides={"DECIDER_MODEL_DIR": str(empty_model)})
    assert no_shards["rc"] != 0 and not no_shards["execed"], (
        "a checkpoint with no weight shards must not reach the engine: "
        + no_shards["out"][-400:])


def test_all_three_gpu2_launchers_read_the_card_from_one_helper():
    """No launcher may inline the query, and all three must source the helper.

    Counted over the whole file text, not per call site, because the defect is two
    private copies of one measurement: the field list, the ``--id`` form and the
    whitespace-stripping all have to move together or the two guards diverge. The
    third member is ``start-decider.sh``, added by #2716 — before that its card
    reading was the one the sweep could not see, which is precisely how a fourth
    private copy would have arrived.
    """
    assert GPU_MEM.exists(), "the shared reader must live under agent-services/bin/"
    for script in (START_SECONDARY, START_DJEV, START_DECIDER):
        text = script.read_text()
        assert text.count("--query-gpu") == 0, \
            f"{script.name} queries the card itself; it must go through gpu-mem.sh"
        assert re.search(r'source\s+"\$PROJECT_DIR/bin/gpu-mem\.sh"', text), \
            f"{script.name} must source the shared helper"
        assert "--query-compute-apps" not in text, \
            f"{script.name} lists holders inline; gpu_mem_holders is the shared path"


def test_the_launchers_keep_their_exec_bits():
    """The mode bit is part of the start path, and nothing else in the suite can see it.

    Supervisord's ``command=`` runs ``/bin/bash <script>``, which works at 100644, so
    every boot test here passes with the bit gone while ``./start-secondary.sh`` — the
    form the script's own Usage header advertises, and one of the ways a boot bypasses
    ``_sync_llm_slots`` that #1316 names — starts failing with EACCES before the guard
    is reached. The first attempt at this round flipped ``100755 -> 100644`` exactly
    that way and the full suite stayed green, which is why the bit is pinned rather
    than trusted.
    """
    for path in (START_SECONDARY, GPU_MEM):
        mode = path.stat().st_mode
        assert mode & statmod.S_IXUSR, f"{path.name} lost its exec bit (mode {oct(mode)})"


def test_the_launcher_refuses_on_a_direct_execve_too(shim, boot, tmp_path):
    """The guard has to hold on the execve path, not only on the ``bash script`` one.

    Same mocked card as every other refusal here — a full GPU 2 and a fake engine — so
    nothing boots either way. What differs is how the process starts: argv[0] is the
    script itself, which is what ``supervisorctl start`` and a hand-run invocation
    actually do, and it depends on the shebang and the mode rather than on whatever
    interpreter the caller chose.
    """
    assert START_SECONDARY.stat().st_mode & statmod.S_IXUSR, \
        "the mode bit has to be there for execve to reach the guard at all"
    r = boot(env_extra=shim(free=20), direct=True)
    assert r["rc"] == 2, f"direct execve is not the same start path: rc={r['rc']} {r['out'][-400:]}"
    assert "refusing to start" in r["out"], r["out"][-400:]
    assert not r["execed"], f"direct execve refused and exec'd anyway: {r['execed']}"


def test_the_helper_asks_nvidia_smi_exactly_once_per_field(tmp_path, shim_dir, shim):
    """Positive control on the shim: a `free` read is one call with one answer.

    Without this, the `--query-gpu` count in the test above could read 0 simply
    because the shim never matched the helper's invocation form, and every boot test
    here would be measuring a stub that answered nothing.
    """
    r = _helper_probe('gpu_mem_query memory.free 2', tmp_path,
                      [str(shim_dir), SYSTEM_PATH], shim(free=20))
    assert r["rc"] == 0, r["out"]
    assert r["out"].strip() == "20", r["out"]
    calls = (tmp_path / "smi-calls.log").read_text().splitlines()
    assert len(calls) == 1, f"one field must be exactly one nvidia-smi call: {calls}"
    assert "--id=2" in calls[0] and "--query-gpu=memory.free" in calls[0]
    assert "memory.total" not in calls[0], "fields must not be combined into one query"


def test_helper_reports_no_measurement_rather_than_zero(tmp_path, shim_dir, shim):
    """`gpu_mem_read` returns non-zero when the card cannot be read.

    It must not return 0 with an empty variable: that is how an unread card becomes
    `free < need`, and then every caller's policy is enforced by a zero nobody
    measured. Also pinned: nvidia-smi missing from PATH entirely is the same answer,
    since a launcher with a thin PATH is the common way to get there.
    """
    probe = ('if gpu_mem_read F T N 2; then echo "READ F=$F T=$T"; exit 0; fi; '
             'echo "NO-READING"')

    failing = _helper_probe(probe, tmp_path, [str(shim_dir), SYSTEM_PATH], shim(fail=1))
    assert failing["rc"] == 0 and failing["out"].strip() == "NO-READING", failing["out"]

    nonnumeric = _helper_probe(probe, tmp_path, [str(shim_dir), SYSTEM_PATH],
                               shim(free="N/A", total=CLAUSE4_CARD_MIB))
    assert nonnumeric["rc"] == 0 and nonnumeric["out"].strip() == "NO-READING", nonnumeric["out"]

    empty_dir = tmp_path / "nothing-on-path"
    empty_dir.mkdir(exist_ok=True)
    missing = _helper_probe(probe, tmp_path, [str(empty_dir)])
    assert missing["rc"] == 0 and missing["out"].strip() == "NO-READING", missing["out"]


def test_djev_refusal_still_fires_on_the_same_mocked_reading(tmp_path, shim_dir, shim):
    """The refactor must not move start-djev.sh's existing refusal.

    It keeps its own policy — refuse on a full card AND on an unreadable one, because
    it is the live ranker — and the free figure it used to compute with a private
    `nvidia-smi ... | tr -d ' '` it now gets from the same helper the secondary
    reads. Its need is the sum of its own budget terms (weights + KV floor +
    canvas transient + overhead); this asserts the refusal fires and names its
    counterpart, not that arithmetic, which `tests/test_start_djev_flags.py` boots
    against.
    """
    home = tmp_path / "home"
    venv = home / "djev-venv"
    model = home / "djev-model"
    src = home / "djev-spark"
    vllm_log = tmp_path / "vllm-exec.log"
    _executable(venv / "bin" / "python", "#!/usr/bin/env bash\nexit 0\n")
    _executable(venv / "bin" / "vllm", f"""
        #!/bin/bash
        echo "vllm $*" >> {vllm_log}
        sleep 60
    """)
    (venv / ".djev-overlay-test").write_text("applied\n")
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text("{}\n")
    (model / "part-00001.safetensors").write_bytes(b"\x00" * 64)
    (src / "server").mkdir(parents=True, exist_ok=True)
    (src / "server" / "structured_server.py").write_text("# stub\n")

    env = {**os.environ, **shim(free=20),
           "DJEV_VENV": str(venv), "DJEV_MODEL_DIR": str(model), "DJEV_SRC": str(src),
           "PORT": "8110", "STRUCTURED_PORT": "8111", "WAIT_SECS": "5"}
    r = subprocess.run(["/bin/bash", str(START_DJEV)], env=env, cwd=str(tmp_path),
                       capture_output=True, text=True, timeout=90)
    out = r.stdout + r.stderr
    assert r.returncode == 2, f"djev refusal moved (rc={r.returncode}): {out[-600:]}"
    assert "refusing to start: GPU 2 has 20 MiB free, this needs" in out, out[-600:]
    assert "agent-llm-secondary" in out, "the either/or must name its counterpart"
    assert DJEV_HOLDER in out, "and what is holding the card"
    assert not vllm_log.exists(), f"djev refused but still launched its engine: {out[-300:]}"


def test_djev_still_refuses_an_unreadable_card(tmp_path, shim_dir, shim):
    """The two slots' policies differ on an unread card, and that must stay visible.

    #1316 clause 5 loosens the secondary; it must not have loosened djev with it. A
    one-shared-helper refactor is exactly how one caller's policy quietly becomes the
    other's, so this is pinned from the other side of the same mocked failure.
    """
    home = tmp_path / "home"
    venv = home / "djev-venv"
    model = home / "djev-model"
    src = home / "djev-spark"
    _executable(venv / "bin" / "python", "#!/usr/bin/env bash\nexit 0\n")
    _executable(venv / "bin" / "vllm", "#!/bin/bash\nsleep 60\n")
    (venv / ".djev-overlay-test").write_text("applied\n")
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text("{}\n")
    (model / "part-00001.safetensors").write_bytes(b"\x00" * 64)
    (src / "server").mkdir(parents=True, exist_ok=True)
    (src / "server" / "structured_server.py").write_text("# stub\n")

    env = {**os.environ, **shim(fail=1),
           "DJEV_VENV": str(venv), "DJEV_MODEL_DIR": str(model), "DJEV_SRC": str(src),
           "PORT": "8112", "STRUCTURED_PORT": "8113", "WAIT_SECS": "5"}
    r = subprocess.run(["/bin/bash", str(START_DJEV)], env=env, cwd=str(tmp_path),
                       capture_output=True, text=True, timeout=90)
    out = r.stdout + r.stderr
    assert r.returncode == 2, f"djev booted on an unread card (rc={r.returncode}): {out[-400:]}"
    assert "no readable gpu 2" in out.lower(), out[-400:]
    assert "MiB free" not in out, "must not print a figure it could not read"
