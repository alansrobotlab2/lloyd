#!/usr/bin/env bash
set -euo pipefail

# Benchmark MTP 1 vs 2 vs 3 for Qwen3.5-122B-A10B
# Uses real conversation prompts from session history

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$PROJECT_DIR/bin/start-llm-122b.sh"
VLLM_PYTHON="$HOME/.venvs/vllm/bin/python"
PROMPTS_FILE="/tmp/mtp-bench-prompts.jsonl"
RESULTS_DIR="/tmp/mtp-bench-results"
BASE_URL="http://127.0.0.1:8096"
MODEL="Qwen3.5-122B-A10B"

mkdir -p "$RESULTS_DIR"

# Backup original script
cp "$SCRIPT" "$SCRIPT.bak"
trap 'cp "$SCRIPT.bak" "$SCRIPT"; rm -f "$SCRIPT.bak"; echo "Restored original config."' EXIT

# Extract prompts if not already done
if [[ ! -f "$PROMPTS_FILE" ]]; then
    echo "=== Extracting prompts from session history ==="
    python3 "$PROJECT_DIR/bin/bench-mtp-extract.py"
fi

NUM_PROMPTS=$(wc -l < "$PROMPTS_FILE")
echo "Using $NUM_PROMPTS prompts from real sessions"
echo ""

# Benchmark function using streaming API
run_benchmark() {
    local mtp_depth=$1
    local results_file="$RESULTS_DIR/mtp-${mtp_depth}.json"

    $VLLM_PYTHON - "$PROMPTS_FILE" "$BASE_URL" "$MODEL" "$results_file" "$mtp_depth" << 'PYEOF'
import json, sys, time, subprocess

prompts_file = sys.argv[1]
base_url = sys.argv[2]
model = sys.argv[3]
results_file = sys.argv[4]
mtp_depth = int(sys.argv[5])

# Load prompts
prompts = []
with open(prompts_file) as f:
    for line in f:
        prompts.append(json.loads(line))

results = []
total_tokens = 0
total_gen_time = 0.0

for i, prompt in enumerate(prompts):
    session = prompt["session"]
    messages = prompt["messages"]
    num_chars = sum(len(m["content"]) for m in messages)

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    })

    # Use curl -N for proper streaming — reads lines as they arrive
    cmd = [
        "curl", "-sN", "--max-time", "180",
        f"{base_url}/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-d", payload,
    ]

    chunk_times = []
    ttft = None
    start = time.perf_counter()
    stream_token_count = 0
    usage_tokens = None
    error = None

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                continue
            try:
                data = json.loads(line[6:])
                # Capture usage from final chunk
                usage = data.get("usage")
                if usage and usage.get("completion_tokens"):
                    usage_tokens = usage["completion_tokens"]
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                # Count any chunk that has reasoning or content text
                # Use 'in' check so empty string "" is still detected as a key present
                has_reasoning = "reasoning" in delta and delta["reasoning"] is not None and len(delta["reasoning"]) > 0
                has_content = "content" in delta and delta["content"] is not None and len(delta["content"]) > 0
                if has_reasoning or has_content:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - start
                    chunk_times.append(now)
                    stream_token_count += 1
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode()
            if stderr.strip():
                error = stderr.strip()[:200]
    except Exception as e:
        error = str(e)

    if error and stream_token_count == 0:
        print(f"  [{i+1}/{len(prompts)}] ERROR: {error}")
        continue

    elapsed = time.perf_counter() - start

    # Use server-reported token count (includes reasoning + content)
    output_tokens = usage_tokens if usage_tokens else stream_token_count

    # Calculate ITL from chunk timestamps
    itls = []
    for j in range(1, len(chunk_times)):
        itls.append(chunk_times[j] - chunk_times[j-1])

    avg_itl = sum(itls) / len(itls) if itls else 0
    tps = output_tokens / elapsed if elapsed > 0 else 0

    total_tokens += output_tokens
    total_gen_time += elapsed

    result = {
        "session": session,
        "input_chars": num_chars,
        "input_msgs": len(messages),
        "output_tokens": output_tokens,
        "stream_chunks": stream_token_count,
        "elapsed_s": round(elapsed, 3),
        "ttft_s": round(ttft, 3) if ttft else None,
        "avg_itl_ms": round(avg_itl * 1000, 1) if itls else None,
        "tokens_per_sec": round(tps, 1),
    }
    results.append(result)
    ttft_str = f"{ttft:.3f}s" if ttft is not None else "N/A"
    itl_str = f"{avg_itl*1000:.1f}ms" if itls else "N/A"
    print(f"  [{i+1}/{len(prompts)}] {output_tokens} tok ({stream_token_count} chunks), {tps:.1f} tok/s, TTFT={ttft_str}, ITL={itl_str}")

# Fetch speculative decoding metrics
accept_rate = None
try:
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
    conn.request("GET", "/metrics")
    resp = conn.getresponse()
    metrics_text = resp.read().decode()
    conn.close()
    for line in metrics_text.split("\n"):
        if "spec_decode" in line.lower() and "accept" in line.lower() and not line.startswith("#"):
            print(f"  [metrics] {line.strip()}")
        if "draft_acceptance_rate" in line and not line.startswith("#"):
            try:
                accept_rate = float(line.split()[-1])
            except ValueError:
                pass
except Exception:
    pass

# Summary
overall_tps = total_tokens / total_gen_time if total_gen_time > 0 else 0
ttfts = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
itls_all = [r["avg_itl_ms"] for r in results if r["avg_itl_ms"] is not None]
tps_all = [r["tokens_per_sec"] for r in results]

summary = {
    "mtp_depth": mtp_depth,
    "num_prompts": len(results),
    "total_output_tokens": total_tokens,
    "total_time_s": round(total_gen_time, 2),
    "overall_tokens_per_sec": round(overall_tps, 1),
    "avg_ttft_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else None,
    "avg_itl_ms": round(sum(itls_all) / len(itls_all), 1) if itls_all else None,
    "avg_tps": round(sum(tps_all) / len(tps_all), 1) if tps_all else None,
    "accept_rate": accept_rate,
    "per_prompt": results,
}

with open(results_file, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  MTP {mtp_depth} summary: {overall_tps:.1f} tok/s overall, TTFT={summary['avg_ttft_s']}s, ITL={summary['avg_itl_ms']}ms")
if accept_rate is not None:
    print(f"  Acceptance rate: {accept_rate:.3f}")
PYEOF
}

wait_for_health() {
    echo "  Waiting for server to be healthy..."
    for i in $(seq 1 120); do
        if curl -sf "$BASE_URL/health" > /dev/null 2>&1; then
            echo "  Server healthy after ~${i}s"
            # Extra settle time
            sleep 3
            return 0
        fi
        sleep 3
    done
    echo "  ERROR: Server did not become healthy within 6 minutes"
    return 1
}

set_mtp_depth() {
    local depth=$1
    sed -i "s/\"num_speculative_tokens\": [0-9]*/\"num_speculative_tokens\": $depth/" "$SCRIPT"
    echo "  Set num_speculative_tokens=$depth"
}

# ============================================================
# Run benchmarks
# ============================================================

for MTP in 1 2; do
    echo "============================================"
    echo "=== Benchmarking MTP depth = $MTP"
    echo "============================================"

    if [[ $MTP -eq 1 ]]; then
        # MTP 1 is already loaded, just verify health
        if curl -sf "$BASE_URL/health" > /dev/null 2>&1; then
            echo "  Server already running with MTP 1"
        else
            echo "  Server not healthy, restarting..."
            set_mtp_depth "$MTP"
            systemctl --user restart agent-llm
            wait_for_health
        fi
    else
        set_mtp_depth "$MTP"
        systemctl --user restart agent-llm
        wait_for_health
    fi

    run_benchmark "$MTP"
    echo ""
done

# ============================================================
# Comparison table
# ============================================================
echo ""
echo "============================================"
echo "=== COMPARISON TABLE"
echo "============================================"

$VLLM_PYTHON - "$RESULTS_DIR" << 'PYEOF'
import json, sys, os

results_dir = sys.argv[1]
summaries = []
for mtp in [1, 2]:
    path = os.path.join(results_dir, f"mtp-{mtp}.json")
    if os.path.exists(path):
        with open(path) as f:
            summaries.append(json.load(f))

if not summaries:
    print("No results found!")
    sys.exit(1)

# Header
print(f"{'Metric':<28} ", end="")
for s in summaries:
    print(f"{'MTP ' + str(s['mtp_depth']):>12}", end="")
print()
print("-" * (28 + 12 * len(summaries) + 1))

# Rows
rows = [
    ("Throughput (tok/s overall)", "overall_tokens_per_sec", ""),
    ("Avg tok/s per request", "avg_tps", ""),
    ("Avg TTFT (s)", "avg_ttft_s", ""),
    ("Avg ITL (ms)", "avg_itl_ms", ""),
    ("Total output tokens", "total_output_tokens", ""),
    ("Total time (s)", "total_time_s", ""),
    ("Acceptance rate", "accept_rate", ""),
]

for label, key, unit in rows:
    print(f"{label:<28} ", end="")
    for s in summaries:
        val = s.get(key)
        if val is None:
            print(f"{'N/A':>12}", end="")
        elif isinstance(val, float):
            print(f"{val:>12.2f}", end="")
        else:
            print(f"{val:>12}", end="")
    print()

# Speedup relative to MTP 1
if len(summaries) >= 2:
    base = summaries[0]["overall_tokens_per_sec"]
    print()
    print(f"{'Speedup vs MTP 1':<28} ", end="")
    for s in summaries:
        speedup = s["overall_tokens_per_sec"] / base if base > 0 else 0
        print(f"{speedup:>11.2f}x", end="")
    print()
PYEOF

# Restore MTP 1 config
set_mtp_depth 1
echo ""
echo "Restored config to MTP 1. Restarting server..."
systemctl --user restart agent-llm
echo "Done! Server restarting with MTP 1 in background."
