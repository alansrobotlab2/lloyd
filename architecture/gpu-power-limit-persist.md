---
segment: architecture
relations:
  related-to: []
tags: [architecture,gpu,ops,nvidia]
type: reference

---

# Persisting NVIDIA GPU Power Limit Across Reboots

`nvidia-smi -i 0 -pl 500` only changes **runtime** state — it resets every boot.
To make it stick,run it at startup via a systemd service.

## The problem

- `-pl <watts>` sets the power **limit**. This is the value to persist.
- `-pm 1` sets persistence **mode** (keeps the driver resident so settings don't drop
  when the GPU goes idle). It does **not** set or remember a power limit,and it also
  resets on reboot. NVIDIA has **deprecated** the `-pm` flag in favor of the
  `nvidia-persistenced` daemon.

Neither flag survives a reboot on its own.

## Fix — systemd service for the power limit

```bash
sudo tee /etc/systemd/system/nvidia-pl.service >/dev/null <<'EOF'
[Unit]
Description=Set NVIDIA GPU 0 power limit to 500W
After=multi-user.target
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -i 0 -pl 500

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-pl.service
```

`enable` → runs every boot. `--now` → applies immediately.

## Complement — persistence mode (reboot-safe daemon)

```bash
sudo systemctl enable --now nvidia-persistenced
```

Replaces hand-running `-pm 1`; persists across reboots. The `Wants=` line above orders
the power-limit service after it.

## Verify

```bash
nvidia-smi -q -d POWER | grep "Power Limit"      # shows Current + Min/Max allowed
systemctl status nvidia-pl.service               # check it didn't fail silently at boot
```

If 500 is above the enforced Max,the service fails silently at boot — only
`systemctl status` reveals it.

## Notes

- Single Blackwell card on this host; both vLLM scripts target GPU 0 (`-i 0`).
- For multiple GPUs,add another `ExecStart=` line per card (`Type=oneshot` runs them
  in order) or drop `-i 0` to apply to all.
