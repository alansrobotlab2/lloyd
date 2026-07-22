# RTX PRO 6000 Blackwell — recurring Xid 79 "GPU has fallen off the bus" under sustained compute

**Prepared:** 2026-06-21 · **Last updated:** 2026-07-21 · **Host:** goliath · **Reporter contact:** gestalt73@gmail.com

## Summary

A single **NVIDIA RTX PRO 6000 Blackwell Workstation Edition** repeatedly drops off the
PCIe bus (Xid 79 → Xid 154 "OS Reboot required") under sustained local LLM-inference
load. The fault has occurred **14 times in the 47 days from 2026-06-04 to 2026-07-21**
(≈ one every ~1.9 days under unclamped load), on driver/GSP **610.43.02** and, for the
latest two events, **610.43.03** — a driver update did not resolve it. Event #14 also
establishes the fault is **workload-independent within vLLM**: it fired on a completely
different model (Laguna S 2.1 118B MoE), engine build (vLLM 0.25.1), and venv than
events #1–13 (Qwen dense/MoE on the 0.19–0.23 stack). As of 2026-07-18 the
fault is **reproducible on demand in ~11 minutes** with a documented saturation workload
(see *Controlled reproduction*), eliminating the intermittency problem for any bench
validation.
Each event is unrecoverable without a full reboot/power-cycle. The failure is **not** tied
to a specific process, time of day, the display path, PCIe gen, core temperature, the PSU,
or case airflow, and persists with the board power limit capped to 500 W. The card is
stable **only** when both power (400 W) and graphics clock (≤ 2400 MHz) are clamped
*below rated spec*; at its rated full-power/full-clock envelope it falls off the bus within
hours to ~2 days. The signature matches several other unresolved Blackwell reports (see
*Related reports*), none of which have an NVIDIA response or fix as of this date.

## System configuration

| Component | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB) |
| GPU serial | 1794625038151 |
| GPU UUID | GPU-71ed578a-0901-df7b-cbe7-3aadde3b3a1d |
| VBIOS | 98.02.81.00.07 |
| GSP firmware | 610.43.02 (events #1–12); 610.43.03 (events #13–14) |
| Driver | nvidia-open (open kernel modules) 610.43.02 → 610.43.03 (fault persists on both) |
| CUDA | 13.3 |
| Kernel | 7.0.12-arch1-1 (Arch Linux) |
| CPU | AMD Ryzen Threadripper PRO 5955WX (16-core) |
| Motherboard | ASUS Pro WS WRX80E-SAGE SE WIFI II |
| System BIOS | 1801 (2025-10-16) |
| RAM | 256 GB |
| PCIe link | Gen4 x16 (current == max; WRX80 platform is PCIe 4.0) |
| GPU power limit | 500 W (capped from 600 W default via systemd unit) |
| PSU (during all 9 faults) | older **ATX 2.x** 1600 W (predates ATX 3.0 excursion spec; GPU fed via 8-pin→12V-2x6 adapter) — **replaced 2026-06-25** with a 3000 W / 240 V unit; see Mitigations |
| Display stack | Hyprland / Wayland |
| Workload | local vLLM inference server, sustained high GPU-mem-util |

## Failure signature

```
NVRM: Xid (PCI:0000:01:00): 79, pid=<varies>, name=<chrome|Xwayland|none>, GPU has fallen off the bus.
NVRM: GPU 0000:01:00.0: GPU has fallen off the bus.
NVRM: Xid (PCI:0000:01:00): 154, GPU recovery action changed from 0x0 (None) to 0x2 (OS Reboot)
NVRM: _threadNodeCheckTimeout: API_GPU_ATTACHED_SANITY_CHECK failed!   (xN)
NVRM: GPU0 _issueRpcAndWait: rpcSendMessage failed with status 0x0000000f for fn 10 ... (x187)   [GSP RPC timeout during teardown]
NVRM: GPU0 _intrServiceStallCommonCheckBegin: Failed GPU reg read : 0xffffffff. Check whether GPU is present on the bus
```

- The GPU stops acknowledging PCIe transactions; all subsequent register reads return
  `0xffffffff` and GSP RPCs time out with `0x0000000f` (NV_ERR_TIMEOUT). This is
  teardown noise *after* the card is already gone — not the trigger.
- The drop is **atomic with no precursor** in 7 of 9 events. In 2 boots an earlier
  `Xid 31` GRAPHICS MMU fault (`FAULT_PDE`, raised by the vLLM engine process) appeared
  hours before the eventual bus drop.
- The `pid`/`name` on the Xid 79 line is always an incidental graphics client
  (`chrome`, `Xwayland`) or none — **never** the inference engine at the moment of the
  drop. It reflects whatever touched the dead GPU first, not the cause.
- Recovery: full reboot / power-cycle only (Xid 154 = `NV_ERR_GPU_IS_LOST`).

## Event log (current journal retention)

| # | Timestamp (local) | Xid | pid/name on drop |
|---|---|---|---|
| 1 | 2026-06-04 02:02:37 | 79 → 154 | chrome |
| 2 | 2026-06-04 22:26:24 | 79 → 154 | chrome |
| 3 | 2026-06-13 04:45:25 | 79 → 154 | (none) |
| 4 | 2026-06-13 23:53:04 | 79 → 154 | Xwayland |
| 5 | 2026-06-15 04:43:32 | 79 → 154 | (none) |
| 6 | 2026-06-16 08:37:23 | 79 → 154 | (none) |
| 7 | 2026-06-18 15:02:34 | 79 → 154 | chrome |
| 8 | 2026-06-19 04:48:25 | 79 → 154 | chrome |
| 9 | 2026-06-21 08:18:52 | 79 → 154 | chrome |
| 10 | 2026-06-25 17:25:41 | 79 → 154 | (none) — GSP RPC timeout `0x0000000f`; deliberate full-speed/unclamped load test |
| 11 | 2026-07-01 12:18:06 | 79 → 154 | (none) — 500 W / clocks unlocked, revamped airflow; ~5 h 20 m after boot |
| 12 | 2026-07-03 ~07:33:40 | (inferred) | hard lock; journal cut off mid-line with no shutdown sequence and **no Xid flushed to disk** — kernel died before writing the fault. Collateral: an in-flight config-file save was left 0 bytes on btrfs (rename committed, data extents lost) |
| 13 | 2026-07-18 21:08:48 | 79 → 154 | (none) — **deliberate controlled reproduction** under saturation load; driver **610.43.03**; Xid 154 issued for **all three GPUs** (01:00, 41:00, 61:00). See *Controlled reproduction* below |
| 14 | 2026-07-21 16:07:09 | 79 → 154 | (none) — first event on the **Laguna S 2.1** workload (118B-A8B MoE NVFP4 + DFlash, vLLM 0.25.1, new `.venvs/vllm-laguna` stack): fired ~2 min after a ~50-min DFlash bench sweep (three cold boots, concurrent `ignore_eos` batches up to 8×32K ctx) while the follow-on supervised restart was loading weights. One 96K bench stream returned zero tokens just before sweep end — possibly the first symptom. Xid 154 again issued for all three GPUs; clamp state at fault time unverified |

Precursor (same uptime as #9): `2026-06-19 15:25:23  Xid 31 MMU Fault, name=VLLM::EngineCor, GPCCLIENT_T1_2 @ 0x23_17000000, FAULT_PDE`.

Events #1–12 occurred under driver/GSP **610.43.02**; events #13–14 under **610.43.03**.
VBIOS **98.02.81.00.07** throughout. Events #1–9 were logged at PCI **01:00** (card's
prior slot); #10–14 at PCI **41:00** after the card was moved (same physical unit — see
*Topology change*).

## Controlled reproduction (event #13, 2026-07-18)

The fault was reproduced **on demand, from idle, in ~11 minutes** — the first
non-spontaneous occurrence. Configuration: driver **610.43.03** (open modules), power
limit at the **default 600 W**, clocks **unlocked**, vLLM 0.23.1rc1.dev1218 serving
Qwen3.6-27B-NVFP4 (MTP speculative decode).

**Discriminating observation — duty cycle matters, not just load.** In the ~40 minutes
immediately prior, the card ran 13 back-to-back agentic-inference tasks (bursty
generation with tool-execution gaps): peaks of 100 % util / **595 W** / **89 °C**, zero
faults, no throttle flags. The card then survived only ~11 minutes of *gap-free*
saturation: **32 concurrent chat completions with `ignore_eos`, 3 000 tokens each,
re-issued continuously** (steady state ≈ 92 % util, 470–487 W, throughput steady at
8 completions/min, 24 000 tokens/min, zero request errors until the drop).

Thermal/throttle timeline (15 s samples of `utilization.gpu, power.draw,
temperature.gpu, clocks_event_reasons.active`):

```
20:56–21:04   steady climb 80 → 87 °C @ ~475 W, throttle mask 0x0 throughout
21:05:11      100 %, 487 W, **92 °C**, mask 0x0000000000000020  ← sw_thermal_slowdown asserts (first time ever observed on this card)
21:05:26      throttle recovery: 81 °C, mask back to 0x0; temp re-climbs to 86 °C
21:08:42      last good sample: 93 %, 472 W, 86 °C, mask 0x0
21:08:48      Xid 79 — GPU has fallen off the bus
21:09:18+     nvidia-smi: "Unable to determine the device handle for GPU1: 0000:41:00.0: Unknown Error"
```

```
Jul 18 21:08:48 kernel: NVRM: Xid (PCI:0000:41:00): 79, GPU has fallen off the bus.
Jul 18 21:08:48 kernel: NVRM: Xid (PCI:0000:61:00): 154, GPU recovery action changed from 0x0 (None) to 0x2 (OS Reboot)
Jul 18 21:08:48 kernel: NVRM: Xid (PCI:0000:41:00): 154, GPU recovery action changed from 0x0 (None) to 0x2 (OS Reboot)
Jul 18 21:08:48 kernel: NVRM: Xid (PCI:0000:01:00): 154, GPU recovery action changed from 0x0 (None) to 0x2 (OS Reboot)
Jul 18 21:08:48 kernel: NVRM: krcRcAndNotifyAllChannels_IMPL: RC all channels for critical error 79.
Jul 18 21:08:48+ kernel: NVRM: _issueRpcAndWait: rpcSendMessage failed with status 0x0000000f  (storm, ongoing)
```

Application-side signature: the vLLM EngineCore died with `torch.AcceleratorError: CUDA
error: unspecified launch failure` (cudaErrorLaunchFailure), first surfacing at the
speculative-decode sync point (`gpu_model_runner._prepare_inputs →
num_accepted_tokens_event.synchronize()`); the API server then hung in shutdown.

Interpretation (two readings, not mutually exclusive, both consistent with the data):

1. **Board-level thermal:** the drop followed the card's first-ever excursion into
   sw_thermal_slowdown (92 °C core) by ~3.5 min. Core temp at the moment of the drop was
   only 86 °C — but this card exposes no memory/VRM junction temperature, so a board
   hotspot that continued heat-soaking during sustained saturation (while the gappy
   agent workload always got cooling breathers) fits: **core temperature is not the
   trigger variable; sustained duty cycle is.**
2. **Transient/clock envelope:** the throttle event itself introduces large clock/voltage
   swings (2 797 MHz ↔ throttled), i.e. exactly the transient envelope already implicated
   by the clamp results (stable at `-lgc 0,2400`, crashes unlocked).

Either way, the practical repro recipe for a bench or RMA validation is simply:
*sustained saturated inference with no idle gaps, stock settings — fault in ≈ 11 min.*
The exact load generator used is preserved at `~/lloyd/scripts/xid79_repro_heat_soak.py`
(32 async workers, `ignore_eos`, 3 000 tokens/request, against any OpenAI-compatible
endpoint).

## What has been ruled out

- **Workload / software stack** — pid at drop is always a graphics client, never the
  engine; happens across day and night and varying load. A peer report (#1111)
  reproduces the same hang on **llama.cpp** (not vLLM), and #1151 reproduces it on
  **Windows**, so it is not specific to the OSS driver or any one server.
- **Display / DPMS / modeset** — DPMS-off and system suspend are disabled; events #7
  and #9 fired with the display fully active.
- **PCIe Gen5 signal integrity** — link runs at platform-max **Gen4 x16**, not a
  downshift.
- **Steady-state core thermal** — GPU core reads **52–59 °C** around events #1–11; no
  throttle flags asserted in any spontaneous event. *(Revised by event #13: under a
  deliberate gap-free saturation soak the card did reach 92 °C / sw_thermal_slowdown and
  dropped off the bus 3.5 min later at 86 °C core — so while instantaneous core temp is
  clearly not the trigger, **sustained duty cycle** is now an established aggravator.
  This card exposes **no** memory/VRM-junction temperature via NVML, so board-level/VRM
  thermal remains unobservable in software and is *not* ruled out — see #369440.)*
- **PCIe link fault as primary cause** — no DPC containment at the drop (platform
  firmware reports `_OSC: platform does not support [AER LTR DPC]`). Corrected
  Data-Link-Layer AER errors do appear on the GPU's upstream root port `0000:00:01.1`,
  but they are **corrected-only and not time-correlated** with the fall-off events.
- **PSU *gross/steady-state* failure** — the rest of the system (CPU, RAM, NVMe, board)
  stays fully alive and logging through every GPU loss; every reboot was
  clean/software-initiated; no MCE, brownout, or under-voltage events. *However, PSU
  transient handling is NOT ruled out and is a leading suspect* — see below.

### Open suspects (not ruled out)

- **PSU transient response (leading local suspect).** The unit is an older **ATX 2.x**
  1600 W supply, predating the ATX 3.0 power-excursion spec written specifically for
  modern NVIDIA GPUs. Blackwell draws sub-millisecond transients well above its rated
  power; an aged ATX 2.x unit can sag the 12V rail or trip OCP/OPP on such spikes
  *regardless of total wattage headroom*, dropping only the GPU (its own dedicated 12V
  cable) while other rails ride through — consistent with "GPU gone, system still
  logging." The GPU is fed via an 8-pin→12V-2x6 adapter, itself a common Xid-79 /
  contact-resistance cause. A `-pl 500` average cap does not constrain these transients,
  consistent with it not having fixed the fault. **Decisive test: swap to a modern ATX
  3.1 PSU with a native 12V-2x6 cable.**
- **Board-level (chipset/VRM) thermal** — unobservable here (no NVML memory/VRM sensor),
  and now **effectively ruled out**: case airflow was revamped and the card re-tested at
  500 W with clocks unlocked (core 81–82 °C). It fell off the bus again after ~5 h 20 m
  (event #11, 2026-07-01) → improved airflow did not resolve the fault, unlike the
  cramped consumer AM5 boards where the #369440 airflow fix applied. This rig is already a
  workstation WRX80 board with heatsinked/actively cooled chipset, generous slot spacing,
  and a large case with forced airflow.

## Mitigations applied / planned (frequency-reducers; none eliminate the fault)

- Board power limit **600 W → 500 W** (persistent systemd unit) — still crashes at 500 W.
- Autonomy worker concurrency **4 → 2** slots (2026-06-21), then **restored to 4**
  (2026-06-25) after the PSU swap — full autonomy load is now part of the validation run.
- **Transient-clamp test (applied 2026-06-21):** power limit **→ 400 W** *and* max
  graphics clock locked **≤ 2400 MHz** (`nvidia-smi -lgc 0,2400`), both persisted in the
  `nvidia-pl.service` unit. Purpose: flatten the sub-millisecond transient envelope an
  aged ATX 2.x PSU may be failing on. **Diagnostic value:** if the crash rate drops
  sharply, it implicates PSU/transient delivery over pure GSP firmware.
- **PSU REPLACED 2026-06-25** — new 3000 W / 240 V unit. The transient clamp was
  reverted (power → 600 W max, clocks unlocked) to validate the new PSU at full load.
  **Test:** if the fall-off-bus fault does not recur while running unclamped at 600 W,
  the old ATX 2.x PSU's transient handling was the dominant local cause. If it recurs at
  full power on the new PSU, weight returns to the GSP-firmware bug / a marginal GPU unit
  (RMA). Baseline crash rate before this change: ~1 per 1.9 days.
- **RESULT — the clamp is an effective mitigation; full-clock running triggers the fault.**
  The clamp (`-pl 400` + clock lock ≤2400 MHz) kept the card **stable** from 2026-06-21
  through the old→new PSU swap (no crashes while clamped). Event #10 (2026-06-25
  17:25:41, Xid 79 at PCI 41:00, GSP signature `rpcSendMessage status 0x0000000f` → Xid
  154 reboot of all cards) occurred during a **deliberate full-speed/unclamped load
  test** — the mitigation was intentionally removed, not a spontaneous failure.
  - Inference: the GSP fault is triggered by the **high-clock/voltage transient
    envelope**. All 9 original crashes were at 500 W but with **unlocked clocks**, so the
    **clock lock is the likely effective lever** (a power cap alone never prevented it).
    Proven-stable config = **400 W *and* clock-lock 2400 together**.
  - PSU is **not the differentiator** (clamp held on both PSUs; full speed crashed on
    both) — but it is also not the cure; the lever is the clock/transient envelope.
  - **Current state is risky:** card is at 500 W with clocks **unlocked** — the same
    envelope that crashed 9×. The proven clamp (esp. the clock lock) should be restored,
    pinned to the correct GPU.
- **AIRFLOW TEST RESULT (event #11, 2026-07-01):** with case airflow revamped and the card
  run at 500 W / clocks unlocked (core 81–82 °C), it fell off the bus again after
  ~5 h 20 m — confirming the fault is **independent of cooling** and that the effective
  lever remains the clock/transient clamp, not temperature. Signature: Xid 79 → 154 at PCI
  41:00; the whole host (all 3 GPUs) was forced to reboot. A boot-time corrected AER
  event was also logged on the *01:00* RTX 3090's link (not on the PRO 6000, not
  time-correlated with the drop).

> **TOPOLOGY CHANGE (2026-06-25):** host is now **multi-GPU** — GPU0 = RTX 3090 @ PCI
> 01:00, **GPU1 = RTX PRO 6000 @ PCI 41:00** (UUID GPU-71ed578a-…-3a1d), GPU2 = RTX 3090
> @ PCI 61:00. All 9 prior events were logged at PCI 01:00 *when the PRO 6000 lived
> there*; event #10 is at PCI **41:00** = same physical card, new slot. **Config bug:**
> `nvidia-pl.service` targets `-i 0`, which is now a 3090 — power/clock settings must be
> re-pinned to the PRO 6000 by **PCI bus id 41:00.0 or UUID**, never by index.

## Related reports (all open, no NVIDIA response/fix as of 2026-06-21)

- NVIDIA/open-gpu-kernel-modules **#1111** — RTX PRO 6000 Blackwell, *llama.cpp*,
  silent hard-hang after ~45 min sustained inference (driver 580.126.20).
- NVIDIA/open-gpu-kernel-modules **#1151** — RTX 5080 (GB203), atomic Xid 79, **also
  reproduces on Windows 11** (595.71.05).
- Dev forums **370415** — RTX PRO 6000 + vLLM falls off bus on **595.71.05** (a
  *different* unit, serial 1795…); reporter mitigated with `nvidia-smi -pl 500`.
- Dev forums **364958** — RTX PRO 6000 Blackwell GSP PMU halt (Xid 62/120/154) under
  vLLM and gpu_burn; persists across 580.126.09 / 595.58.03.
- Dev forums **369440** — RTX 5090 Xid 79 under sustained CUDA load; community fix was
  **aggressive airflow into the card↔motherboard thermal dead zone** (GPU core looked
  fine at ~70 °C).

## Request

**For NVIDIA / upstream:** another RTX PRO 6000 Blackwell data point for the open Xid 79
/ GSP-fall-off-bus cluster, now confirmed on **two consecutive drivers (610.43.02 and
610.43.03)** and VBIOS 98.02.81.00.07, and now **reproducible on demand in ~11 minutes**
(see *Controlled reproduction*). Requesting acknowledgment, a bug ID, and guidance on
whether a VBIOS/GSP firmware update is planned.

**For RMA / vendor:** a single unit falling off the PCIe bus **13 times in 44 days**,
with every documented mitigation (power cap, reduced load, PSU replacement, revamped
airflow, driver update) ineffective and the card **only stable when clamped below its
rated spec**. The failure is no longer intermittent-only: a documented stock-settings
workload (sustained saturated inference, no idle gaps) reproduces it from idle in
≈ 11 minutes, so it **will fail a bench test**. Full kernel logs (`journalctl`) for all
events are available on request.
