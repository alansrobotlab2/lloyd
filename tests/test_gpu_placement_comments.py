"""#1447: the GPU placement comments in the voice stack described a machine
that no longer exists, in the two files a CUDA-pressure debug reaches first.

`start-qwen3-tts.sh` said "(GPU 1)" in its header against its own
`export CUDA_VISIBLE_DEVICES=0`, and `lloyd-agent-worker.conf` justified its
pin with "GPU 2 is now single-tenant" after djev had taken GPU 2 at 0.97
utilisation. Comment-only drift, so the pin here is the same shape as the
drift: the header's number is read off the export it sits above, and the
worker conf's story about GPU 2 is read off `agent-djev.conf`'s own `GPU=`.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "agent-services" / "bin" / "start-qwen3-tts.sh"
CONF_D = ROOT / "agent-services" / "supervisor" / "conf.d"
WORKER_CONF = CONF_D / "lloyd-agent-worker.conf"
DJEV_CONF = CONF_D / "agent-djev.conf"
INFRA_DOC = ROOT / "architecture" / "infrastructure.md"


def _env_pin(text: str, key: str) -> str:
    m = re.search(rf'{re.escape(key)}"?(\d+)"?', text)
    assert m, f"{key} not found"
    return m.group(1)


def test_tts_launcher_header_names_the_gpu_it_exports():
    text = LAUNCHER.read_text()
    header = next(ln for ln in text.splitlines() if ln.startswith("# Starts the Qwen3-TTS"))
    claimed = re.search(r"\(GPU (\d+)", header)
    assert claimed, header
    exported = _env_pin(text, "export CUDA_VISIBLE_DEVICES=")
    assert claimed.group(1) == exported, (header, exported)


def test_worker_conf_names_djev_as_gpu_2s_tenant():
    worker = WORKER_CONF.read_text()
    djev = DJEV_CONF.read_text()
    djev_gpu = _env_pin(djev, "GPU=")
    comments = "\n".join(ln for ln in worker.splitlines() if ln.startswith(";"))
    assert "single-tenant" not in comments
    assert re.search(rf"GPU {djev_gpu}: agent-djev", comments), comments
    # The reason the worker is off GPU 1 is the sentence that still binds.
    assert "2026-09-03" in comments and "gpu-memory-utilization" in comments
    # And the doc's GPU table tells the same story about that card.
    row = next(ln for ln in INFRA_DOC.read_text().splitlines()
               if ln.startswith(f"| {djev_gpu} | RTX 3090"))
    assert "agent-djev" in row and "single-tenant" not in row, row
