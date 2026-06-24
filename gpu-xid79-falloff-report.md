# RTX PRO 6000 Blackwell — recurring Xid 79 "GPU has fallen off the bus" under sustained compute

**Prepared:** 2026-06-21 · **Host:** goliath · **Reporter contact:** gestalt73@gmail.com

## Summary

A single **NVIDIA RTX PRO 6000 Blackwell Workstation Edition** repeatedly drops off the
PCIe bus (Xid 79 → Xid 154 "OS Reboot required") under sustained local LLM-inference
load. The fault has occurred **9 times in the 17 days from 2026-06-04 to 2026-06-21**
(≈ one every ~1.9 days), every time on driver/GSP **610.43.02**. Each event is
unrecoverable without a full reboot/power-cycle. The failure is **not** tied to a
specific process, time of day, the display path, PCIe gen, or core temperature, and
persists with the board power limit already capped to 500 W. The signature matches
several other unresolved Blackwell reports (see *Related reports*), none of which have
an NVIDIA response or fix as of this date.

## System configuration

| Component | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB) |
| GPU serial | 1794625038151 |
| GPU UUID | GPU-71ed578a-0901-df7b-cbe7-3aadde3b3a1d |
| VBIOS | 98.02.81.00.07 |
| GSP firmware | 610.43.02 |
| Driver | nvidia-open (open kernel modules) 610.43.02 |
| CUDA | 13.3 |
| Kernel | 7.0.12-arch1-1 (Arch Linux) |
| CPU | AMD Ryzen Threadripper PRO 5955WX (16-core) |
| Motherboard | ASUS Pro WS WRX80E-SAGE SE WIFI II |
| System BIOS | 1801 (2025-10-16) |
| RAM | 256 GB |
| PCIe link | Gen4 x16 (current == max; WRX80 platform is PCIe 4.0) |
| GPU power limit | 500 W (capped from 600 W default via systemd unit) |
| PSU | older **ATX 2.x** unit, 1600 W (predates ATX 3.0 transient/excursion spec; GPU fed via 8-pin→12V-2x6 adapter) |
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

Precursor (same uptime as #9): `2026-06-19 15:25:23  Xid 31 MMU Fault, name=VLLM::EngineCor, GPCCLIENT_T1_2 @ 0x23_17000000, FAULT_PDE`.

All 9 events occurred under driver/GSP **610.43.02**, VBIOS **98.02.81.00.07**.

## What has been ruled out

- **Workload / software stack** — pid at drop is always a graphics client, never the
  engine; happens across day and night and varying load. A peer report (#1111)
  reproduces the same hang on **llama.cpp** (not vLLM), and #1151 reproduces it on
  **Windows**, so it is not specific to the OSS driver or any one server.
- **Display / DPMS / modeset** — DPMS-off and system suspend are disabled; events #7
  and #9 fired with the display fully active.
- **PCIe Gen5 signal integrity** — link runs at platform-max **Gen4 x16**, not a
  downshift.
- **Steady-state core thermal** — GPU core reads **52–59 °C** around the faults; no
  `hw_thermal_slowdown` / `sw_thermal_slowdown` flags ever asserted. (Note: this card
  exposes **no** memory/VRM-junction temperature via NVML, so board-level/VRM thermal
  cannot be observed in software and is *not* ruled out — see #369440.)
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
- **Board-level (chipset/VRM) thermal** — unobservable here (no NVML memory/VRM sensor)
  but considered *unlikely* on this rig: workstation WRX80 board with heatsinked/actively
  cooled chipset, generous slot spacing, large case with forced airflow — unlike the
  cramped consumer AM5 boards where the #369440 airflow fix applied.

## Mitigations applied / planned (frequency-reducers; none eliminate the fault)

- Board power limit **600 W → 500 W** (persistent systemd unit) — still crashes at 500 W.
- Autonomy worker concurrency **4 → 2** slots (applied 2026-06-21) — under observation.
- **Transient-clamp test (applied 2026-06-21):** power limit **→ 400 W** *and* max
  graphics clock locked **≤ 2400 MHz** (`nvidia-smi -lgc 0,2400`), both persisted in the
  `nvidia-pl.service` unit. Purpose: flatten the sub-millisecond transient envelope an
  aged ATX 2.x PSU may be failing on. **Diagnostic value:** if the crash rate drops
  sharply, it implicates PSU/transient delivery over pure GSP firmware.
- Pending hardware fix: replace PSU with a modern ATX 3.1 unit + native 12V-2x6 cable.

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
/ GSP-fall-off-bus cluster, on a *newer* VBIOS (98.02.81.00.07) and the latest open
driver (610.43.02), confirming the fault is **not** fixed by current firmware/driver.
Requesting acknowledgment, a bug ID, and guidance on whether a VBIOS/GSP firmware
update is planned.

**For RMA / vendor:** a single unit falling off the PCIe bus 9 times in 17 days under
nominal thermal and power conditions, with the documented mitigations (power cap,
reduced load) ineffective, is requested for inspection/replacement. Full kernel logs
(`journalctl`) for all 9 events are available on request.
