#!/usr/bin/env bash
# Clamp every NVIDIA GPU to a fixed board power limit.
#
# Why this exists: the RTX PRO 6000 Blackwell in this box has a documented
# fall-off-the-bus defect (Xid 79 -> Xid 154, full power-cycle to recover; see
# gpu-xid79-falloff-report.md, RMA open). It is stable only when clamped below
# rated spec. `nvidia-smi -pl` is RUNTIME-ONLY state and resets every boot, so
# the clamp has to be reapplied at startup or the card comes back at 600 W.
#
# Run as root, via nvidia-power-limit.service. Override with
# GPU_POWER_LIMIT_W=<watts>. No dependency on bc — integer math only.
set -uo pipefail

WATTS="${GPU_POWER_LIMIT_W:-300}"
SMI=/usr/bin/nvidia-smi

if [[ ! -x "$SMI" ]]; then
  echo "[gpu-power-limit] ERROR: $SMI not found" >&2
  exit 1
fi

mapfile -t IDS < <("$SMI" --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ')
if (( ${#IDS[@]} == 0 )); then
  echo "[gpu-power-limit] ERROR: no GPUs enumerated" >&2
  exit 1
fi

# Query one field at a time: GPU names contain spaces, so a combined query
# cannot be split safely with `read`.
q() { "$SMI" -i "$1" --query-gpu="$2" --format=csv,noheader 2>/dev/null; }

rc=0
for i in "${IDS[@]}"; do
  name=$(q "$i" name)
  min=$(q "$i" power.min_limit | tr -dc '0-9.'); min=${min%%.*}
  max=$(q "$i" power.max_limit | tr -dc '0-9.'); max=${max%%.*}

  # Clamp into this card's accepted range rather than letting nvidia-smi reject it.
  target=${WATTS%%.*}
  [[ -n "$min" ]] && (( target < min )) && target=$min
  [[ -n "$max" ]] && (( target > max )) && target=$max

  if "$SMI" -i "$i" -pl "$target" >/dev/null 2>&1; then
    echo "[gpu-power-limit] GPU $i ($name): set ${target} W  (range ${min}-${max} W)"
  else
    echo "[gpu-power-limit] GPU $i ($name): FAILED to set ${target} W (need root?)" >&2
    rc=1
  fi
done

exit "$rc"
