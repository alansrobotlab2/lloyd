"""What the vLLM-Omni TTS launcher hands the process it exec's.

`agent-services/bin/start-qwen3-tts-omni.sh` is the second TTS launcher
(architecture/voice.md "The TTS server", 2026-10-09). Like
`tests/test_qwen3_tts_launcher.py` for the first, these tests run the real
script with the venv's `vllm` replaced by a stub that records the argv and
environment the CHILD received — what reaches the exec boundary is the claim,
not the script text. Three things have to cross it:

- the GPU pin, as a UUID: the box's five cards do not keep one numbering
  between `/dev/nvidiaN`, `nvidia-smi` and CUDA, and a bare index once put a
  worker on the engine's card;
- the venv's `bin` on PATH: inductor shells out to `ninja`, and the first boot
  died in the talker's compile because the launcher exec'd `$VENV/bin/vllm`
  without activating the venv;
- a loopback bind by default (the same #1638 rule as the first launcher).
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "agent-services" / "bin" / "start-qwen3-tts-omni.sh"
CONF = ROOT / "agent-services" / "supervisor" / "conf.d" / "agent-tts-omni.conf"
GPU3_UUID = "GPU-0bd41021-2916-fb5d-c100-5bdc5a9c41c0"

_STUB_VLLM = """\
#!/usr/bin/env bash
{
  printf 'argv=%s\\n' "$*"
  printf 'CUDA_VISIBLE_DEVICES=%s\\n' "${CUDA_VISIBLE_DEVICES-<unset>}"
  printf 'CUDA_DEVICE_ORDER=%s\\n' "${CUDA_DEVICE_ORDER-<unset>}"
  printf 'PATH=%s\\n' "$PATH"
  printf 'SPEAKER_SAMPLES_DIR=%s\\n' "${SPEAKER_SAMPLES_DIR-<unset>}"
} > "$KNOB_DUMP"
"""


def _run_launcher(tmp_path: Path, **overrides) -> dict[str, str]:
    sandbox = tmp_path / "sandbox"
    bin_dir = sandbox / "agent-services" / "bin"
    bin_dir.mkdir(parents=True)
    script = bin_dir / LAUNCHER.name
    shutil.copy2(LAUNCHER, script)
    model = sandbox / "agent-services" / "services" / "tts" / "qwen3-tts" / "models" / "Qwen3-TTS-12Hz-1.7B-Base"
    model.mkdir(parents=True)

    venv = sandbox / "home" / "lloyd" / ".venvs" / "vllm-omni"
    (venv / "bin").mkdir(parents=True)
    stub = venv / "bin" / "vllm"
    stub.write_text(_STUB_VLLM)
    stub.chmod(0o755)
    deploy = venv / "lib" / "python3.12" / "site-packages" / "vllm_omni" / "deploy"
    deploy.mkdir(parents=True)
    (deploy / "qwen3_tts.yaml").write_text("stages: []\n")

    dump = tmp_path / "child.txt"
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("TTS_", "CUDA_", "SPEAKER_"))}
    env.update({"HOME": str(sandbox / "home"), "KNOB_DUMP": str(dump),
                # the speakers dir default is production data; point it elsewhere
                "SPEAKER_SAMPLES_DIR": str(tmp_path / "speakers")})
    env.update(overrides)
    r = subprocess.run([str(script)], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, f"launcher exited {r.returncode}\n{r.stdout}{r.stderr}"
    assert dump.exists(), "the launcher never reached the exec'd child"
    out = dict(line.split("=", 1) for line in dump.read_text().splitlines())
    out["_venv"] = str(venv)
    return out


def _flag(argv: str, name: str) -> str:
    parts = argv.split()
    assert name in parts, f"{name} missing from the child's argv: {argv!r}"
    return parts[parts.index(name) + 1]


def test_the_launcher_is_executable_because_supervisor_execs_it_directly():
    assert LAUNCHER.stat().st_mode & stat.S_IXUSR
    conf = CONF.read_text()
    assert f"/agent-services/bin/{LAUNCHER.name}" in conf, "the conf must exec this launcher"
    assert "command=" in conf and "autostart=true" in conf


def test_the_child_gets_the_gpu_pin_as_a_uuid_and_the_venv_bin_on_path(tmp_path):
    child = _run_launcher(tmp_path)
    assert child["CUDA_VISIBLE_DEVICES"] == GPU3_UUID
    assert child["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert child["PATH"].split(os.pathsep)[0] == f"{child['_venv']}/bin", (
        "ninja lives in the venv's bin; inductor must find it first")
    assert child["SPEAKER_SAMPLES_DIR"] == str(tmp_path / "speakers")
    assert (tmp_path / "speakers").is_dir(), "the launcher creates the speakers dir"


def test_the_conf_and_the_launcher_agree_on_the_pin():
    conf = CONF.read_text()
    assert f'CUDA_VISIBLE_DEVICES="{GPU3_UUID}"' in conf
    assert GPU3_UUID in LAUNCHER.read_text()


def test_it_binds_loopback_on_8093_unless_told_otherwise(tmp_path):
    child = _run_launcher(tmp_path)
    assert _flag(child["argv"], "--host") == "127.0.0.1"
    assert _flag(child["argv"], "--port") == "8093"
    assert "--omni" in child["argv"].split()
    assert _flag(child["argv"], "--served-model-name") == "qwen3-tts", (
        "the worker and the guardian send model: qwen3-tts")
    child = _run_launcher(tmp_path / "b", TTS_OMNI_HOST="100.64.0.9", TTS_OMNI_PORT="8099")
    assert _flag(child["argv"], "--host") == "100.64.0.9"
    assert _flag(child["argv"], "--port") == "8099"


def test_an_operator_pin_in_the_environment_is_kept(tmp_path):
    """Supervisor's `environment=` arrives before bash runs; the launcher's
    default must not clobber it (the first launcher's rule, #1446)."""
    child = _run_launcher(tmp_path, CUDA_VISIBLE_DEVICES="GPU-6287dd24-9d9c-4ad6-236b-9a17498d0328")
    assert child["CUDA_VISIBLE_DEVICES"] == "GPU-6287dd24-9d9c-4ad6-236b-9a17498d0328"


def test_a_missing_venv_or_model_fails_before_the_exec(tmp_path):
    sandbox = tmp_path / "sandbox"
    bin_dir = sandbox / "agent-services" / "bin"
    bin_dir.mkdir(parents=True)
    script = bin_dir / LAUNCHER.name
    shutil.copy2(LAUNCHER, script)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TTS_", "CUDA_"))}
    env["HOME"] = str(sandbox / "home")
    r = subprocess.run([str(script)], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 1 and "venv not found" in r.stderr
