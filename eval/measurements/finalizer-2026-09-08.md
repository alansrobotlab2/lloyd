# Finalizer live verification — 2026-09-08

Primary: Qwen3.8-Flash-Next-nvfp4, vLLM 0.28.1 on :8096, MTP + `--async-scheduling`
+ `--enable-prefix-caching`. Secondary: llama.cpp (Qwen3.6-35B-A3B) on :8091.

## 1. The identical-tools rule — why the finalizer must not drop `tools`

Qwen's chat template renders the tools array **inside the system message**, so
dropping it changes the rendered prompt near the top, not the bottom. Measured
deterministically through the model's own template (no engine, no cache):

| | tokens |
|---|---|
| with tools | 1536 |
| tools omitted | 888 |
| **shared prefix** | **41 tokens — 2.7%** |

`'Bash'` first appears 5.7% into the rendered string. Everything after the
divergence is a cache miss.

### Confirmed end to end, in the production shape

Two independent fresh conversations (~180k tokens each), each warmed with four
tools-bearing iterations the way the agent loop warms one, then hit **once**
with a finalizer:

```
finalizer KEEPS tools
   loop iteration 3: prompt=180049 cached=177600   0.73s
   loop iteration 4: prompt=180049 cached=177600   0.74s
   >>> FINALIZER:    prompt=180068 cached=177600   1.19s   reuse 98.6%

finalizer DROPS tools
   loop iteration 3: prompt=189051 cached=187200   0.66s
   loop iteration 4: prompt=189051 cached=187200   0.66s
   >>> FINALIZER:    prompt=188004 cached=0       25.00s   reuse 0%
```

**21x.** One object costs 25 seconds and a full 188k re-prefill if `tools` is
dropped, and 1.2 seconds if it is not.

### Two traps in measuring this, both of which produce a wrong answer

1. **`usage.prompt_tokens_details.cached_tokens` is 0 for the first two
   requests of any prefix**, on this engine, whatever you send.
   `start-qwen38-flash-next.sh` documents it: cross-request reuse needs two
   warm-up passes, so a 2-pass A/B reads 0% on both arms and proves nothing.
   Every arm above is warmed past it.
2. **Alternating the two shapes caches both.** An A/B that sends
   with-tools/without-tools repeatedly ends with two independently cached
   prefixes and shows a ~10% difference — which reads as "the rule does not
   matter". It is not the production pattern: production is N tools-bearing
   iterations and then exactly **one** finalizer, and only the identical-tools
   one can hit what the turn just built. That is why the experiment above uses
   two separate fresh conversations.

## 2. Streaming: five runs, all inside the 1024-token budget

```
run 1:  1.40s finish=stop completion_tokens=153
run 2:  2.91s finish=stop completion_tokens=355
run 3:  3.23s finish=stop completion_tokens=347
run 4:  4.64s finish=stop completion_tokens=527
run 5:  6.42s finish=stop completion_tokens=734
```

All five: `finish_reason: stop`, reasoning deltas first, `content` pure JSON
that parses and whose `verdict` is in the enum. The grammar applies after
`</think>` — `enable_in_reasoning` is False — so thinking can stay on.

## 3. `enable_thinking: false`

Accepted; content still parses; 42 completion tokens against 153-734 with
thinking on, 0.40s against 1.4-6.4s. Left **off by default** and not exposed:
the verdict is a transcription of reasoning the turn already did, but this was
measured on one prompt shape and a quality claim needs more than a latency
number. Worth a follow-up A/B on real triage verdicts.

## 4. Both schema spellings on the secondary (:8091, llama.cpp)

| spelling | result |
|---|---|
| `response_format.json_schema` | HTTP 200, valid, 0.67s |
| top-level `json_schema` | HTTP 200, valid, 0.27s |

This build accepts **both**, so the fallback chain never fires here. It stays
in: the two-spelling problem is the same one the dual `reasoning` /
`reasoning_content` keys exist for, and a build that accepts only one would
otherwise return unconstrained prose that happens to parse — or not.

## 5. A large transcript end to end

210,420 chars / 42,514 prompt tokens: 9.0s cold, 6.4s warm, `finish_reason:
stop`, valid object. With the cache warm (measurement 1) the same shape is
~1.2s. Seconds, not a re-prefill.
