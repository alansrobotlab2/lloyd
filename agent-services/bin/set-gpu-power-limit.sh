#!/usr/bin/env bash
# Clamp each NVIDIA GPU to a board power limit, per card.
#
# Why this exists: the RTX PRO 6000 Blackwell in this box has a documented
# fall-off-the-bus defect (Xid 79 -> Xid 154, full power-cycle to recover; see
# gpu-xid79-falloff-report.md, RMA open). It is stable only when clamped below
# rated spec. `nvidia-smi -pl` is RUNTIME-ONLY state and resets every boot, so
# the clamp has to be reapplied at startup or the card comes back at its default
# limit (600 W on the PRO 6000, 350 W on each 3090).
#
# Limits are keyed by GPU index so the two 3090s and the PRO 6000 can differ:
#   GPU_POWER_LIMIT_W        default for any index without its own override
#   GPU_POWER_LIMIT_W_<n>    override for GPU index <n>, e.g. ..._1=400
# The values in force live in nvidia-power-limit.service, not here.
#
# Index caveat: <n> is nvidia-smi enumeration order, not a stable hardware id.
# If a card drops off the bus (the known defect) the remaining cards shift down,
# so a 3090 could inherit an override meant for the PRO 6000 — the range clamp
# below bounds the damage at that card's own max_limit.
#
# Values outside a card's own [power.min_limit, power.max_limit] are clamped to
# it rather than left for nvidia-smi to reject. No dependency on bc — integer
# math only.
#
# Run as root, via nvidia-power-limit.service.
set -uo pipefail

DEFAULT_W="${GPU_POWER_LIMIT_W:-300}"
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

  # Per-index override, else the default. Indirect expansion, so an unset
  # GPU_POWER_LIMIT_W_<n> falls through without tripping `set -u`.
  ov="GPU_POWER_LIMIT_W_${i}"
  src="GPU_POWER_LIMIT_W"
  if [[ -n "${!ov:-}" ]]; then raw="${!ov}"; src="$ov"; else raw="$DEFAULT_W"; fi

  # Clamp into this card's accepted range rather than letting nvidia-smi reject it.
  target=${raw%%.*}
  [[ -n "$min" ]] && (( target < min )) && target=$min
  [[ -n "$max" ]] && (( target > max )) && target=$max

  if "$SMI" -i "$i" -pl "$target" >/dev/null 2>&1; then
    echo "[gpu-power-limit] GPU $i ($name): set ${target} W  (from $src, range ${min}-${max} W)"
  else
    echo "[gpu-power-limit] GPU $i ($name): FAILED to set ${target} W (need root?)" >&2
    rc=1
  fi
done

exit "$rc"
