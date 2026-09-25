#!/usr/bin/env bash
# Download the voice worker's untracked models into agent-services/models/.
#
#   bash agent-services/setup/fetch-voice-models.sh          # fetch what is missing
#   bash agent-services/setup/fetch-voice-models.sh --check  # report, change nothing
#
# Tracked in the repo (force-added past the unanchored `models/` ignore rule,
# because a missing one leaves the worker deaf): the wake-word models and
# Silero VAD. Downloaded here, because they are large and re-downloadable:
#
#   smart-turn/            pipecat Smart Turn v3.2, CPU int8 ONNX       ~9 MB   BSD-2-Clause
#   parakeet-tdt-v3/       NeMo Parakeet TDT 0.6B v3, int8, sherpa-onnx ~670 MB CC-BY-4.0
#   nemo-streaming-480ms/  NeMo streaming FastConformer CTC, 480 ms     ~460 MB (livekit.stt.streaming, off)
#   campplus/              3D-Speaker CAM++ VoxCeleb speaker embedding  ~28 MB  Apache-2.0
#                          (livekit.voiceprint.backend: campplus; speaker_id.py patches it as it loads)
#
# Sizes are checked, not just presence: a truncated encoder still loads and
# still transcribes — badly — and nothing else would say why.
set -euo pipefail

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
M="$REPO/agent-services/models"
HF=https://huggingface.co

# dir|file|expected bytes|url
FILES=(
  "smart-turn|smart-turn-v3.2-cpu.onnx|8679182|$HF/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx"
  "parakeet-tdt-v3|encoder.int8.onnx|652184281|$HF/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main/encoder.int8.onnx"
  "parakeet-tdt-v3|decoder.int8.onnx|11845275|$HF/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main/decoder.int8.onnx"
  "parakeet-tdt-v3|joiner.int8.onnx|6355277|$HF/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main/joiner.int8.onnx"
  "parakeet-tdt-v3|tokens.txt|93939|$HF/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main/tokens.txt"
  "nemo-streaming-480ms|model.onnx|458883812|$HF/csukuangfj/sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-480ms/resolve/main/model.onnx"
  "nemo-streaming-480ms|tokens.txt|11896|$HF/csukuangfj/sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-480ms/resolve/main/tokens.txt"
  "campplus|3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx|29596978|https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
)

missing=0
for row in "${FILES[@]}"; do
  IFS='|' read -r dir file size url <<<"$row"
  path="$M/$dir/$file"
  have=$(stat -c %s "$path" 2>/dev/null || echo 0)
  if [[ "$have" == "$size" ]]; then
    echo "  ✓ $dir/$file"
    continue
  fi
  missing=1
  if (( CHECK_ONLY )); then
    echo "  → $dir/$file missing or wrong size ($have != $size)"
    continue
  fi
  mkdir -p "$M/$dir"
  echo "  ↓ $dir/$file"
  curl -fL --retry 3 -o "$path.part" "$url"
  got=$(stat -c %s "$path.part")
  if [[ "$got" != "$size" ]]; then
    echo "  ✗ $dir/$file: downloaded $got bytes, expected $size" >&2
    exit 1
  fi
  mv "$path.part" "$path"
done
(( CHECK_ONLY && missing )) && exit 1
exit 0
