"""The djev determinism bisect levers (#1357).

The seam is `env -> bash -> the vLLM argv`, which no Python import can see:
`agent-djev` is launched by supervisord as
`command=/bin/bash .../agent-services/bin/start-djev.sh`
(agent-services/supervisor/conf.d/agent-djev.conf:23), so a variable that is
declared in that script but never reaches argv is a bisect that boots the
incumbent, measures it, and records the numbers under a variant's name. That is
a worse outcome than not shipping the lever, because the variant table in the
script header is the artifact the sweep is supposed to stop repeating.

So these tests do not grep for a variable's spelling. They RUN the script with a
stub `vllm` that records the argv it was handed and the environment it was
exported into, with a stub `nvidia-smi` so the VRAM preflight passes, and they
read the recording back. One of them (`test_the_shipped_defaults_have_a_measured_row`)
is the enforceable half of "the defaults are the measured winner": it refuses a
default whose lever pair has no measured row in the header, so an unbooted kernel
choice cannot become production by editing one line instead of two.
"""

from __future__ import annotations

import os
import re
import socket
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "agent-services" / "bin" / "start-djev.sh"

SCRIPT_TEXT = SCRIPT.read_text()

NVIDIA_SMI = """#!/usr/bin/env bash
case "$*" in
    *memory.free*)        echo 23000 ;;
    *memory.total*)       echo 24576 ;;
    *query-compute-apps*) echo "1 used_memory [stub]" ;;
    *name*)               echo "NVIDIA GeForce RTX 3090 (stub)" ;;
    *) echo "stub nvidia-smi: unexpected $*" >&2; exit 1 ;;
esac
"""

# Records every argument on its own line, plus the one environment variable the
# engine reads, then stays alive: the script's health loop then spends its one
# five-second iteration, which is also what lets this test assert the boot
# actually got as far as launching the engine.
VLLM_STUB = """#!/usr/bin/env bash
{
    for a in "$@"; do printf 'ARG %s\\n' "$a"; done
    printf 'ENV VLLM_BATCH_INVARIANT=%s\\n' "${VLLM_BATCH_INVARIANT-<unset>}"
} > "$ARGV_RECORD"
exec sleep 60
"""


def _executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def boot(tmp_path):
    """Run start-djev.sh against a stubbed-out box; return what the engine saw."""
    venv = tmp_path / "venv"
    model = tmp_path / "model"
    src = tmp_path / "djev-spark"
    binf = tmp_path / "bin"
    record = tmp_path / "argv.txt"

    _executable(venv / "bin" / "python", "#!/usr/bin/env bash\nexit 0\n")
    _executable(venv / "bin" / "vllm", VLLM_STUB)
    (venv / ".djev-overlay-test").write_text("applied\n")
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text("{}\n")
    (model / "part-00001.safetensors").write_bytes(b"\x00" * 64)
    (src / "server").mkdir(parents=True, exist_ok=True)
    (src / "server" / "structured_server.py").write_text("# stub\n")
    _executable(binf / "nvidia-smi", NVIDIA_SMI)

    def run(**overrides) -> dict:
        env = dict(os.environ)
        for drop in ("MOE_BACKEND", "BATCH_INVARIANT", "VLLM_BATCH_INVARIANT"):
            env.pop(drop, None)
        env.update({
            "DJEV_VENV": str(venv),
            "DJEV_MODEL_DIR": str(model),
            "DJEV_SRC": str(src),
            "PORT": str(_free_port()),
            "STRUCTURED_PORT": str(_free_port()),
            # One 5 s health iteration: enough for the stub to write its record,
            # short enough that the boot cannot hang this test.
            "WAIT_SECS": "5",
            "ARGV_RECORD": str(record),
            "PATH": f"{binf}:{env['PATH']}",
        })
        env.update({k: str(v) for k, v in overrides.items()})
        proc = subprocess.run(["bash", str(SCRIPT)], env=env, cwd=str(ROOT),
                              capture_output=True, text=True, timeout=120)
        argv: list[str] = []
        exported: dict[str, str] = {}
        if record.exists():
            for line in record.read_text().splitlines():
                if line.startswith("ARG "):
                    argv.append(line[4:])
                elif line.startswith("ENV "):
                    key, _, value = line[4:].partition("=")
                    exported[key] = value
        return {"rc": proc.returncode, "argv": argv, "env": exported,
                "out": proc.stdout + proc.stderr, "recorded": record.exists()}

    return run


def _flag_pair(argv: list[str], flag: str) -> list[str]:
    """The value(s) following `flag`, or [] when it is absent."""
    if flag not in argv:
        return []
    i = argv.index(flag)
    return argv[i + 1:i + 2]


# ── Clause 1: each lever reaches the engine by env override, no file edit ─────

def test_moe_backend_reaches_the_vllm_argv(boot):
    r = boot(MOE_BACKEND="triton")
    assert r["recorded"], f"the engine was never launched: {r['out'][-800:]}"
    assert _flag_pair(r["argv"], "--moe-backend") == ["triton"], \
        f"--moe-backend did not reach the argv as one flag and one value: {r['argv']}"


def test_the_default_boot_is_the_shipped_measured_row(boot):
    """The default boot is the shipped row of the header: no --moe-backend (vLLM
    resolves auto), batch-invariant ON, 81920 context (#1361, shipped 2026-09-24)."""
    r = boot()
    assert r["recorded"], f"the engine was never launched: {r['out'][-800:]}"
    assert "--moe-backend" not in r["argv"], \
        f"an unset MOE_BACKEND changed the boot: {r['argv']}"
    assert r["env"]["VLLM_BATCH_INVARIANT"] == "1"
    assert _flag_pair(r["argv"], "--max-model-len") == ["81920"], r["argv"]


def test_batch_invariant_is_exported_into_the_engine_environment(boot):
    r = boot(BATCH_INVARIANT="1")
    assert r["recorded"], f"the engine was never launched: {r['out'][-800:]}"
    # VLLM_BATCH_INVARIANT is read by vllm/envs.py in the ENGINE process, so the
    # only thing that matters is what the child was exported.
    assert r["env"]["VLLM_BATCH_INVARIANT"] == "1"


def test_both_levers_together_reach_the_same_boot(boot):
    r = boot(MOE_BACKEND="batched_triton", BATCH_INVARIANT="1")
    assert _flag_pair(r["argv"], "--moe-backend") == ["batched_triton"]
    assert r["env"]["VLLM_BATCH_INVARIANT"] == "1"
    assert "moe batched_triton" in r["out"] and "batch-invariant 1" in r["out"], \
        "the banner does not show the boot's kernel choice, so a person cannot " \
        "tell which variant is running"


def test_batch_invariant_refuses_a_value_vllm_cannot_parse(boot):
    """vllm/envs.py parses it with int(), so an empty or 'true' value would
    raise at import inside the engine, after the preflight has already taken the
    card. Refuse first."""
    r = boot(BATCH_INVARIANT="true")
    assert r["rc"] == 2, f"expected a refusal, got rc={r['rc']}: {r['out'][-800:]}"
    assert "BATCH_INVARIANT" in r["out"]
    assert not r["recorded"], "a rejected lever must not boot the engine"


def test_moe_backend_refuses_a_value_that_is_not_one_kernel_name(boot):
    r = boot(MOE_BACKEND="--moe-backend=triton")
    assert r["rc"] == 2, f"expected a refusal, got rc={r['rc']}: {r['out'][-800:]}"
    assert "MOE_BACKEND" in r["out"]
    assert not r["recorded"], "a rejected lever must not boot the engine"


# ── Clause 4: the shipped defaults are a measured boot, and the header says so ─

DEFAULTS_RE = re.compile(
    r'^# shipped defaults: MOE_BACKEND="(?P<moe>[^"]*)" BATCH_INVARIANT="(?P<inv>[^"]*)"$',
    re.MULTILINE)
# A row is `#   <MOE_BACKEND, or 'auto' when empty> / <BATCH_INVARIANT>  cold  warm  p50`.
# Requiring the `/ 0` or `/ 1` suffix is what keeps a prose line that happens to
# carry three numbers from being read as a measurement.
VARIANT_RE = re.compile(
    r'^#\s{3}(?P<variant>[^#\n]*? / (?P<inv>[01]))\s+(?P<cold>-?\d+\.\d+)\s+'
    r'(?P<warm>-?\d+\.\d+)\s+(?P<p50>\d+(?:\.\d+)?)\s*$',
    re.MULTILINE)


def _script_defaults() -> tuple[str, str]:
    moe = re.search(r'^MOE_BACKEND="\$\{MOE_BACKEND:-(.*?)\}"$', SCRIPT_TEXT, re.MULTILINE)
    inv = re.search(r'^BATCH_INVARIANT="\$\{BATCH_INVARIANT:-(.*?)\}"$', SCRIPT_TEXT, re.MULTILINE)
    assert moe and inv, "start-djev.sh no longer declares its defaults in one line"
    return moe.group(1), inv.group(1)


def _header_rows() -> list[dict[str, str]]:
    return [m.groupdict() for m in VARIANT_RE.finditer(SCRIPT_TEXT)]


def _header_defaults_line() -> tuple[str, str]:
    m = DEFAULTS_RE.search(SCRIPT_TEXT)
    assert m, ('the script must carry a line exactly of the form '
               '# shipped defaults: MOE_BACKEND="x" BATCH_INVARIANT="y"')
    return m.group("moe"), m.group("inv")


def test_the_header_names_every_booted_variant_with_both_measurements():
    """A row without numbers is how a sweep gets repeated: it records that a
    variant was thought about, not what it measured."""
    rows = _header_rows()
    assert rows, "the header has no measured variant row at all"
    for row in rows:
        assert float(row["cold"]) >= 0.0 and float(row["warm"]) >= 0.0
        assert float(row["p50"]) > 0.0, f"{row['variant']}: no measured recall p50"
    # The 0.55 s ceiling is on the SHIPPED row (below), not on every booted one:
    # a variant that boots and measures slower still has to be recorded, or the
    # next sweep boots it again (#1361's BATCH_INVARIANT=1 at 65536 read 552.9).


def test_the_defaults_line_and_the_script_defaults_cannot_drift_apart():
    """One line is prose, one line is behaviour. If they can disagree, the prose
    is what the next reader trusts."""
    assert _header_defaults_line() == _script_defaults(), (
        "the `# shipped defaults:` header line disagrees with the script's own "
        "MOE_BACKEND / BATCH_INVARIANT defaults")


def test_the_shipped_defaults_have_a_measured_row():
    """No unmeasured boot becomes production. An empty MOE_BACKEND is written
    `auto` in the table because that is what vLLM resolves it to."""
    moe, inv = _script_defaults()
    want = (moe or "auto").split()[0]
    ctx = re.search(r'^MAX_MODEL_LEN="\$\{MAX_MODEL_LEN:-(\d+)\}"$', SCRIPT_TEXT, re.MULTILINE)
    assert ctx, "start-djev.sh no longer declares its MAX_MODEL_LEN default in one line"
    # A row that names a MAX_MODEL_LEN measured THAT context; only the shipped
    # one is the shipped boot (#1361 booted BATCH_INVARIANT=1 at 81920 and 65536).
    rows = [r for r in _header_rows()
            if r["variant"].split()[0].rstrip(",") == want
            and r["variant"].rstrip().endswith(f"/ {inv}")
            and ("MAX_MODEL_LEN" not in r["variant"]
                 or f"MAX_MODEL_LEN {ctx.group(1)} " in r["variant"])]
    assert rows, (
        f"shipped defaults MOE_BACKEND={moe!r} BATCH_INVARIANT={inv!r} have no row in "
        f"the header's variant table, so this boot has never been measured by "
        f"scripts/djev_determinism_probe.py")
    for row in rows:
        assert float(row["p50"]) <= 550.0, (
            f"{row['variant']}: recall p50 {row['p50']} ms breaches the 0.55 s "
            f"ceiling #1357 sets, so it cannot be the shipped boot")


def test_the_measured_incumbent_row_records_the_nondeterminism_being_bisected():
    """The row that documents the defect must not read like a passing boot: the
    incumbent is shipped precisely because it has not been fixed."""
    row = _header_rows()[0]
    assert float(row["warm"]) > 0.0 and float(row["cold"]) > 0.0, (
        f"the incumbent row claims {row['cold']}/{row['warm']} nats, which would "
        f"mean the defect is already fixed and #1357 needs no sweep")
