#!/usr/bin/env bash
set -euo pipefail

# Sweep --speculative-config num_speculative_tokens for sakamakismile
# Qwen3.6-27B-Text-NVFP4-MTP. Run while the existing primary slot (port 8096)
# is FREE — this script launches and tears down its own vLLM server for each n.
#
# Methodology per n:
#   1. sed the start script's num_speculative_tokens
#   2. Launch in background, wait for /health
#   3. Capture /metrics baseline
#   4. Run a fixed prompt set (bench-single-user.py-style, 4 prompts, 1500 max_tokens)
#   5. Capture /metrics again, compute delta
#   6. Record decode tok/s, TTFT, accept rate, per-position acceptance
#   7. SIGTERM the server, wait for /health to fail
#
# Final restore: original n value in start script.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
START_SCRIPT="$PROJECT_DIR/bin/start-27b-nvfp4-sakamakismile-mtp.sh"
VLLM_PYTHON="$HOME/lloyd/.venvs/vllm-experimental/bin/python"
BASE_URL="http://127.0.0.1:8096"
MODEL="primary"
LOG_FILE="/home/alansrobotlab/lloyd/logs/sakamakismile-mtp-sweep.log"
RESULTS_DIR="/tmp/sakamakismile-mtp-sweep"

# Sweep these values by default; override with: N_VALUES="1 2 3" ./bench-mtp-sakamakismile.sh
N_VALUES=${N_VALUES:-"1 2 3 4"}

mkdir -p "$RESULTS_DIR"
: > "$LOG_FILE"

cp "$START_SCRIPT" "$START_SCRIPT.bak"
trap 'cp "$START_SCRIPT.bak" "$START_SCRIPT"; rm -f "$START_SCRIPT.bak"; pkill -TERM -f "sakamakismile-Qwen3.6-27B-Text-NVFP4-MTP" 2>/dev/null || true; echo "[restore] original script + killed server"' EXIT

set_n() {
  local n=$1
  sed -i "s/\"num_speculative_tokens\": [0-9]*/\"num_speculative_tokens\": $n/" "$START_SCRIPT"
}

wait_for_health() {
  echo "  waiting for /health..."
  for i in $(seq 1 180); do
    if curl -sf "$BASE_URL/health" > /dev/null 2>&1; then
      echo "  healthy at ~$((i*2))s"
      sleep 5  # extra settle for cuda-graph capture
      return 0
    fi
    sleep 2
  done
  echo "  ERROR: /health never came up"
  return 1
}

wait_for_dead() {
  for i in $(seq 1 30); do
    if ! curl -sf "$BASE_URL/health" > /dev/null 2>&1; then
      echo "  server down"
      sleep 2
      return 0
    fi
    sleep 2
  done
  echo "  WARNING: server still responding after 60s"
}

run_one() {
  local n=$1
  echo "============================================"
  echo "=== n=$n"
  echo "============================================"
  set_n "$n"

  nohup "$START_SCRIPT" >> "$LOG_FILE" 2>&1 &
  local server_pid=$!
  echo "  launched PID=$server_pid"

  if ! wait_for_health; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait_for_dead
    return 1
  fi

  # Baseline metrics snapshot
  curl -s "$BASE_URL/metrics" > "$RESULTS_DIR/n${n}-metrics-before.txt"

  # Run benchmark
  "$VLLM_PYTHON" "$PROJECT_DIR/bin/bench-single-user.py" "n${n}" \
    --base-url "$BASE_URL" --model "$MODEL" --runs 4 --max-tokens 1500 \
    | tee "$RESULTS_DIR/n${n}-bench.txt"

  # Post metrics snapshot
  curl -s "$BASE_URL/metrics" > "$RESULTS_DIR/n${n}-metrics-after.txt"

  # SIGTERM and wait for the server to release the port
  kill -TERM "$server_pid" 2>/dev/null || true
  pkill -TERM -f "sakamakismile-Qwen3.6-27B-Text-NVFP4-MTP" 2>/dev/null || true
  wait_for_dead
  sleep 3
}

for n in $N_VALUES; do
  run_one "$n" || echo "  n=$n failed, continuing"
done

echo ""
echo "============================================"
echo "=== SUMMARY"
echo "============================================"

"$VLLM_PYTHON" - "$RESULTS_DIR" "$N_VALUES" << 'PY'
import json, os, re, sys

results_dir = sys.argv[1]
n_values = [int(x) for x in sys.argv[2].split()]

def parse_metrics(path):
    out = {}
    if not os.path.exists(path):
        return out
    line_re = re.compile(r'^(vllm:spec_decode_\S+?)\{([^}]*)\}\s+([0-9.eE+\-]+)')
    pos_re = re.compile(r'position="(\d+)"')
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            m = line_re.match(line)
            if not m:
                continue
            name, labels, val = m.group(1), m.group(2), float(m.group(3))
            pm = pos_re.search(labels)
            if pm:
                out.setdefault(name, {})[int(pm.group(1))] = val
            else:
                out[name] = val
    return out

def parse_bench(path):
    """Extract per-run rows and summary line from bench-single-user.py output."""
    if not os.path.exists(path):
        return None
    runs = []
    summary = None
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            # data rows like:  0   0   0.124    1500     12.34    121.5
            m = re.match(r'\s*\d+\s+\d+\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$', line)
            if m:
                ttft, tokens, decode, tps = m.groups()
                runs.append({'ttft': float(ttft), 'tokens': int(tokens), 'decode': float(decode), 'tps': float(tps)})
            if line.startswith('SUMMARY'):
                summary = line
    return {'runs': runs, 'summary': summary}

rows = []
for n in n_values:
    before = parse_metrics(f"{results_dir}/n{n}-metrics-before.txt")
    after  = parse_metrics(f"{results_dir}/n{n}-metrics-after.txt")
    bench  = parse_bench(f"{results_dir}/n{n}-bench.txt")
    if bench is None:
        print(f"n={n}: missing bench output")
        continue

    drafts = after.get('vllm:spec_decode_num_drafts_total',0) - before.get('vllm:spec_decode_num_drafts_total',0)
    draft_tok = after.get('vllm:spec_decode_num_draft_tokens_total',0) - before.get('vllm:spec_decode_num_draft_tokens_total',0)
    accepted = after.get('vllm:spec_decode_num_accepted_tokens_total',0) - before.get('vllm:spec_decode_num_accepted_tokens_total',0)
    per_pos_after  = after.get('vllm:spec_decode_num_accepted_tokens_per_pos_total',{})
    per_pos_before = before.get('vllm:spec_decode_num_accepted_tokens_per_pos_total',{})
    per_pos = {p: per_pos_after.get(p,0) - per_pos_before.get(p,0) for p in sorted(set(per_pos_after) | set(per_pos_before))}

    overall_accept = accepted / draft_tok if draft_tok else 0.0
    mean_accepted_len = (accepted/drafts + 1) if drafts else 0.0  # +1 for the always-accepted target token

    tpss = [r['tps'] for r in bench['runs']]
    ttfts = [r['ttft'] for r in bench['runs']]
    avg_tps = sum(tpss)/len(tpss) if tpss else 0
    min_tps = min(tpss) if tpss else 0
    max_tps = max(tpss) if tpss else 0
    avg_ttft = sum(ttfts)/len(ttfts) if ttfts else 0

    rows.append({
        'n': n,
        'avg_tps': avg_tps, 'min_tps': min_tps, 'max_tps': max_tps,
        'avg_ttft': avg_ttft,
        'drafts': int(drafts), 'draft_tok': int(draft_tok), 'accepted': int(accepted),
        'overall_accept': overall_accept, 'mean_accepted_len': mean_accepted_len,
        'per_pos': per_pos,
    })

# Print
hdr = ['n', 'avg_tok/s', 'min/max', 'avg_ttft', 'accept', 'mean_len', 'per-pos accept %']
print(f"{hdr[0]:>3}  {hdr[1]:>9}  {hdr[2]:>13}  {hdr[3]:>9}  {hdr[4]:>7}  {hdr[5]:>8}  {hdr[6]}")
print('-' * 90)
for r in rows:
    pp = ' '.join(
        f"{(r['per_pos'][p]/r['drafts']*100 if r['drafts'] else 0):5.1f}"
        for p in sorted(r['per_pos'])
    )
    print(f"{r['n']:>3}  {r['avg_tps']:>9.1f}  {r['min_tps']:>5.1f}/{r['max_tps']:<6.1f}  {r['avg_ttft']:>9.3f}  {r['overall_accept']*100:>6.1f}%  {r['mean_accepted_len']:>8.2f}  {pp}")

# Save JSON
with open(f"{results_dir}/summary.json", 'w') as f:
    json.dump(rows, f, indent=2)
print(f"\nFull JSON: {results_dir}/summary.json")
PY
