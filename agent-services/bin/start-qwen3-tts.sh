#!/usr/bin/env bash
set -euo pipefail

# Starts the Qwen3-TTS API server on port 8090 (GPU 0, the desktop 3090 —
# the CUDA_VISIBLE_DEVICES export below is the pin; agent-tts.conf repeats it)
#
# OpenAI-compatible TTS with streaming PCM output
# Model: Qwen/Qwen3-TTS-12Hz-1.7B-Base (local copy)
# VRAM: ~4-6GB estimated
#
# API: POST /v1/audio/speech

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$HOME/lloyd/.venvs/qwen3-tts"
QWEN3_TTS_DIR="$PROJECT_DIR/services/tts/qwen3-tts"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Qwen3-TTS venv not found at $VENV"
  echo "Create it with: uv venv $VENV --python 3.12"
  exit 1
fi

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="/usr/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
export TTS_BACKEND=optimized
export TTS_CONFIG="$QWEN3_TTS_DIR/config.yaml"
export PORT=8090

# Eager loading: load and compile the model at BOOT, not on the first thing
# anyone says. The backend's own default is lazy — api/main.py resolves
# `TTS_LAZY_LOAD = _env_bool("TTS_LAZY_LOAD", True)` — and this model's
# `compile_mode: max-autotune` makes the first synthesis request pay the
# inductor autotune. Measured 2026-09-19: the stack restarted at 19:54, the
# first voice turn arrived at 20:04:42 and audio came out at 20:08:48 — 4 min
# 6 s, of which "Warmup 1/3 streaming" alone was 2 min 59 s.
#
# The cost of that first request is not only latency. TTSStreamer._http in
# agent-services/livekit_worker.py is one serial client with `read=120.0`, so an
# utterance that waits longer than 120 s is DISCARDED, not merely delayed: that
# incident lost "One moment." and "Honestly?" exactly 120 s apart, and the answer
# to a 20:04:39 question was first heard at 20:08:56, starting mid-sentence.
#
# Eager loading moves the wait to where nobody is talking: the warmup runs
# inside uvicorn's lifespan, so :8090 does not answer /health for ~4 min after a
# restart. TTS_WARMUP_ON_START (set in conf.d/agent-tts.conf) rides along but
# does nothing for this Base model — eager init is the fix, not the warmup.
#
# `${VAR:-default}`, never a bare export: supervisor passes its `environment=`
# line into this script's environment before bash runs it, so a hard `export
# TTS_LAZY_LOAD=false` would silently clobber an operator who set it back to lazy
# there. The default lives here rather than only in the conf so that a person
# running this script gets the engine production boots — until #1446 the knob
# existed only in conf.d/agent-tts.conf, and a hand-run of this file reproduced
# the 2026-09-19 loss exactly.
export TTS_LAZY_LOAD="${TTS_LAZY_LOAD:-false}"

cd "$QWEN3_TTS_DIR"

# Wait for port to be free (don't kill — supervisor handles process lifecycle)
for i in $(seq 1 30); do
  ss -tlnp 2>/dev/null | grep -q ":${PORT} " || break
  echo "Waiting for port $PORT to be free... (${i}/30)"
  [[ $i -eq 30 ]] && { echo "ERROR: port $PORT still in use after 30s"; exit 1; }
  sleep 1
done

# Bind interface: loopback, by default. This listener answers /v1/voices and
# POST /v1/audio/speech with no credential of any kind, so the interface it
# binds is the only access control it has. The exec below used to name the wide
# value explicitly, which no commit ever asked for: `git log -S'--host'` on this
# file returns only a3833043, "move agent-services into lloyd". The result was
# :8090 answering unauthenticated on 192.168.50.0/24 — the range server.py:77-78
# declines to trust for /api/*, and the 1b050d82 gate there is middleware in
# another process; this is a second uvicorn with none. start-cosyvoice-tts.sh and
# start-orpheus-tts.sh already pass 127.0.0.1 for this same port, and every
# client in the repo points at loopback (config.yaml tts.api_url,
# agent-services/guardian/speak.py, agent-services/livekit_worker.py): 638 of the
# 642 access lines in agent-tts.log on 2026-09-27 came from 127.0.0.1, the other
# 4 from this box's own tailnet address, none from a third host.
#
# The check that proves it, once agent-tts has been restarted (192.168.50.108
# was this box's LAN address as measured 2026-09-27; take `hostname -I`'s first
# if the lease has moved):
#   ss -ltn | grep :8090                                   -> 127.0.0.1:8090
#   curl --max-time 5 http://192.168.50.108:8090/v1/voices  -> no connection
#   curl --max-time 5 http://127.0.0.1:8090/v1/voices       -> 200
#
# `${VAR:-default}`, so loopback is a default and not a hard-coded flag — same
# shape as TTS_LAZY_LOAD above, and here the narrow side is the one that can
# strand somebody. To serve a tailnet device or Voice Studio directly, put
# HOST="0.0.0.0", or the single address to serve, on conf.d/agent-tts.conf's
# `environment=` line and supervisor hands it to this script before bash runs.
#
# Do not "simplify" that into dropping the flag: `--host "$HOST"` is the only
# thing carrying it. uvicorn's CLI does not read HOST — its auto_envvar_prefix is
# UVICORN (main.py:62), so the env name it would honour is UVICORN_HOST — and the
# CLI's own --host default is already 127.0.0.1 (main.py:64-67). Measured
# 2026-09-27 in the venv that runs this service: HOST=0.0.0.0 with --host omitted
# bound 127.0.0.1:8099. Dropping the flag would therefore strand the override
# rather than widen anything, which is also why the wide value below is not
# uvicorn's: 0.0.0.0 is what the *app module* resolves for its own entry point,
# `HOST = os.getenv("HOST", "0.0.0.0")` at api/main.py:72 feeding
# uvicorn.run(host=HOST) at api/main.py:304-306 — the `python api/main.py` path,
# not this one.
#
# A HOST that turns out not to be a local address is no silent wide bind either:
# uvicorn exits on the bind OSError (config.py:536-539, and server.py:172-182 on
# the create_server route), and main.py:614 exits 3 if the server never started,
# so a mistyped or inherited value fails the boot loudly — startretries=3 on this
# conf makes it FATAL.
export HOST="${HOST:-127.0.0.1}"

exec "$VENV/bin/python" -m uvicorn api.main:app \
    --host "$HOST" \
    --port 8090
