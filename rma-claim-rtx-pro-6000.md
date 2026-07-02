# Warranty Claim — NVIDIA RTX PRO 6000 Blackwell Workstation Edition

**Claim type:** Hardware defect, within 3-year manufacturer warranty
**Date prepared:** 2026-06-25

## Product / purchase

| Field | Value |
|---|---|
| Product | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB |
| Serial number | 1794625038151 |
| GPU UUID | GPU-71ed578a-0901-df7b-cbe7-3aadde3b3a1d |
| VBIOS | 98.02.81.00.07 |
| GSP firmware | 610.43.02 |
| Retailer | Micro Center |
| Purchase date | **<FILL IN from receipt>** |
| Order / receipt # | **<FILL IN>** |
| Card brand (on sticker) | PNY |

## Defect statement

The card repeatedly **falls off the PCIe bus (NVIDIA Xid 79, "GPU has fallen off the
bus")** under sustained compute load, immediately followed by **Xid 154 (recovery action
→ "OS Reboot")**. The GPU stops responding on the bus entirely (register reads return
`0xffffffff`; GSP RPCs time out with `status 0x0000000f`), and **only a full power-cycle
recovers it.** When it fails it also forces a reboot of the entire host.

This is **not** acceptable behavior for a Workstation-class card rated for sustained,
full-power, 24/7 professional compute — and the card **only remains stable when
underclocked below its rated specification** (see "What keeps it alive").

## Frequency

**11 fall-off-bus events in 27 days** (2026-06-04 → 2026-07-01), ≈ 1 per 1.9 days under
normal (unclamped) sustained load. The card is stable *only* while clamped below its rated
spec (see "What keeps it alive"); each recurrence below either predates the clamp or
occurred when the clamp was deliberately removed to validate the card at its rated
envelope:

| # | Timestamp (local) | Signature |
|---|---|---|
| 1 | 2026-06-04 02:02:37 | Xid 79 → 154 |
| 2 | 2026-06-04 22:26:24 | Xid 79 → 154 |
| 3 | 2026-06-13 04:45:25 | Xid 79 → 154 |
| 4 | 2026-06-13 23:53:04 | Xid 79 → 154 |
| 5 | 2026-06-15 04:43:32 | Xid 79 → 154 |
| 6 | 2026-06-16 08:37:23 | Xid 79 → 154 |
| 7 | 2026-06-18 15:02:34 | Xid 79 → 154 |
| 8 | 2026-06-19 04:48:25 | Xid 79 → 154 |
| 9 | 2026-06-21 08:18:52 | Xid 79 → 154 |
| 10 | 2026-06-25 17:25:41 | Xid 79 → 154 (GSP RPC timeout `0x0000000f`) |
| 11 | 2026-07-01 12:18:06 | Xid 79 → 154 (500 W / clocks unlocked, revamped airflow; ~5 h 20 m after boot) |

## Troubleshooting already performed (defect persists through all of it)

- **Power supply replaced** — swapped from a 1600 W unit to a new **3000 W / 240 V**
  supply. Fault persisted → not a power-delivery problem.
- **Driver** — on this unit the fault is present on the current driver 610.43.02; the
  card ran stably for ~2.5 weeks on the prior 595.71.05 driver (and for months before
  that), with the failures beginning within days of the 610.43.02 update. The identical
  Xid 79 / GSP fall-off is independently documented on other Blackwell GPUs across
  580.x / 595.x / 610.x (see corroborating reports) — i.e. a firmware/silicon failure
  mode, not a local misconfiguration. No driver that resolves it is currently available.
- **Display/desktop** — DPMS-off and system suspend disabled; failures occur with the
  display active and at idle, ruling out a display/modeset cause.
- **PCIe** — link runs at platform-max Gen4 x16 (no downshift); no PCIe AER/DPC
  containment event coincident with the drops.
- **PCIe slot / lane** — the card has been installed in multiple physical slots on
  different CPU PCIe root complexes over its life (PCI buses 01, 41, 61). It falls off the
  bus in every slot used during the failure period (9 events in bus 01, 1 in bus 41) →
  not a slot, lane-group, or motherboard-port problem; the fault follows the card.
- **Thermal** — GPU core temperature is normal (50–69 °C) at the time of failures; no
  thermal-slowdown throttle flags are ever asserted. Note: a healthy GPU responds to
  thermal stress by *throttling* (asserting slowdown, reducing clocks), not by dropping
  off the PCIe bus. This unit falls off the bus at core temperatures well within its
  rated operating range, with no throttle event — i.e. not a normal thermal-protection
  response, indicating a fault (power-stage/VRM under load, or GSP firmware) rather than
  expected hot-but-safe operation. The card also remains stable when run *hotter*
  (~82 °C core) under a clock clamp than at the cooler temps (~60 °C) where it failed
  unclamped — further evidence core temperature is not the trigger.
- **Cooling / airflow** — case airflow was revamped and the card re-tested at 500 W with
  clocks unlocked (core 81–82 °C, well under limit). It fell off the bus again after
  ~5 h 20 m (event #11, 2026-07-01) → improved cooling does not resolve the fault; core
  temperature is not the trigger.
- **Workload-independent** — occurs under multiple compute workloads and is not tied to
  any single process.

## What keeps it alive (evidence of out-of-spec behavior)

The card is **only stable when power and clocks are clamped below its rated
specification**: power limit reduced to **400 W** (rated 600 W) **and** the graphics
clock locked to **≤ 2400 MHz** (rated boost up to 3090 MHz). At its rated full-power /
full-clock envelope it falls off the bus within hours to ~2 days. A professional GPU that
cannot sustain its own rated operating envelope is defective.

## Known failure mode (corroborating reports)

The identical Xid 79 / GSP-firmware fall-off-bus signature is documented by multiple
other RTX PRO 6000 Blackwell and Blackwell GPU owners under sustained compute, e.g.
NVIDIA open-gpu-kernel-modules issues #1111, #1151, #1080 and NVIDIA Developer Forum
threads on RTX PRO 6000 GSP PMU halts — indicating a reproducible firmware/silicon
failure mode, not a configuration error on our end.

## Requested resolution

Warranty **repair or replacement** of the unit. Given this is a production workstation,
we request an **advance / cross-ship replacement** if available to minimize downtime.

## Attachments to include when submitting

1. `nvidia-bug-report.log.gz` — generated via `sudo nvidia-bug-report.sh`
2. Micro Center purchase receipt / proof of purchase
3. This document and the full technical report (`gpu-xid79-falloff-report.md`)

## How to submit (PNY)

Warranty is handled by **PNY**, not Micro Center (past the 30-day in-store window).

1. **Call PNY support: 1-800-234-4597.** State it is a **professional RTX PRO 6000
   Blackwell Workstation Edition** — pro products may have a separate/expedited path and
   advance-replacement option not offered on the consumer web form. Ask specifically for
   **cross-ship / advance replacement** to avoid downtime.
2. Online alternative: **Retail RMA form** — https://www.pny.com/company/support/retail-rma-form
   (and register the product: https://www.pny.com/company/support/product-registration).
   Main support hub: https://www.pny.com/support
3. Have ready: **Micro Center proof of purchase** (receipt/invoice with date), the card
   **serial number** (1794625038151), **original packaging**, and the attachments above.
4. Note: standard PNY RMA is return-first and the customer pays outbound shipping unless
   advance replacement is granted — worth pushing for given it's a production card.
