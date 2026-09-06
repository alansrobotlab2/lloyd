"""Alert fan-out. Stdlib only, and no channel may raise.

Five channels, ordered so the most reliable goes first. They fail differently
on purpose: the ledger works when the network is down, the journal works when
the filesystem state dir is unreadable, the desktop notification works when
nobody is looking at a browser, and the vault note works tomorrow morning.

Deliberately **not** `app/discord_notify.py`: it is async, it imports
`app.config`, and with `config.yaml`'s token empty — which is the current
production state — it degrades to `logger.warning`, which lands in
`logs/server.err` where nobody reads it. The gap between "alerting configured"
and "alerting works" is exactly what bites during an incident, so this module
reports an unconfigured channel as a visible fact instead of swallowing it.

The channel that actually closes the loop is `backlog_task`: after a rollback,
Lloyd wakes up on known-good code with a work item naming what was reverted,
why, and the `git cherry-pick` that restores his work. That converts a
rollback from punishment into a task, which is the point of the whole design.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


def _run(cmd: list[str], timeout: float = 5.0) -> bool:
    try:
        subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
        return True
    except Exception:
        return False


class Notifier:
    def __init__(self, *, ledger: Path, state_dir: Path, vault_root: str,
                 backend_url: str = "http://127.0.0.1:8080"):
        self.ledger = ledger
        self.state_dir = Path(state_dir)
        self.vault_root = Path(vault_root)
        self.backend_url = backend_url.rstrip("/")

    def alert(self, level: str, title: str, body: str, *, evidence: str = "",
              commit: str = "", trigger: str = "", tag: str = "") -> dict:
        """Fan out one alert. Returns per-channel success for the heartbeat."""
        results: dict[str, bool] = {}
        text = f"{title}\n\n{body}".strip()
        if evidence:
            text += f"\n\n--- evidence ---\n{evidence[:4000]}"

        results["ledger"] = self._ledger(level, title, body, evidence, commit, trigger, tag)
        results["journal"] = self._journal(level, f"{title} :: {body}")
        results["desktop"] = self._desktop(level, title, body)
        results["alert_file"] = self._alert_file(level, title, text)
        results["vault"] = self._vault_note(title, text)
        if level == "critical" or trigger:
            results["backlog"] = self._backlog_task(title, text, commit, tag)
        return results

    # ── channels ───────────────────────────────────────────────────────
    def _ledger(self, level, title, body, evidence, commit, trigger, tag) -> bool:
        try:
            import gstate
            gstate.append_event(self.ledger, {
                "event": "alert", "level": level, "title": title,
                "body": body[:2000], "evidence": evidence[:4000],
                "commit": commit, "trigger": trigger, "tag": tag,
            })
            return True
        except Exception:
            return False

    def _journal(self, level: str, message: str) -> bool:
        prio = {"critical": "2", "error": "3", "warn": "4"}.get(level, "5")
        return _run(["systemd-cat", "-t", "lloyd-guardian", "-p", prio,
                     "--", "echo", message[:4000]])

    def _desktop(self, level: str, title: str, body: str) -> bool:
        urgency = "critical" if level == "critical" else "normal"
        env_ok = bool(os.environ.get("DBUS_SESSION_BUS_ADDRESS"))
        if not env_ok:
            return False
        return _run(["notify-send", "-u", urgency, f"Lloyd guardian: {title}", body[:400]])

    def _alert_file(self, level: str, title: str, text: str) -> bool:
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            (self.state_dir / "ALERT.md").write_text(
                f"# {title}\n\nlevel: {level}\nwritten: {datetime.now().isoformat()}\n\n{text}\n",
                encoding="utf-8",
            )
            return True
        except Exception:
            return False

    def _vault_note(self, title: str, text: str) -> bool:
        try:
            memory = self.vault_root / "memory"
            if not memory.is_dir():
                return False
            note = memory / f"{datetime.now().strftime('%Y-%m-%d')}.md"
            with open(note, "a", encoding="utf-8") as f:
                f.write(f"\n\n## Self-mod guardian: {title}\n\n{text}\n")
            return True
        except Exception:
            return False

    def _backlog_task(self, title: str, text: str, commit: str, tag: str) -> bool:
        """File a backlog item so the revert becomes work, not a mystery."""
        body = text
        if commit:
            body += (f"\n\nYour work is preserved. To re-apply and investigate:\n"
                     f"    git cherry-pick {commit}\n")
        if tag:
            body += f"The pre-rollback tree is tagged `{tag}`.\n"
        # Field names match app/routers/backlog.py::backlog_task_create, which
        # takes `name`/`description`/`status`, NOT title/body. Sending the wrong
        # keys does not error — the endpoint defaults `name` to "New Task" and
        # returns 2xx, so the alert reads as delivered while filing a task that
        # says nothing. `status` must be one of _VALID_STATUSES.
        payload = json.dumps({
            "name": f"[guardian] {title}"[:120],
            "description": body[:6000],
            "status": "up_next",
            "priority": "high",
        }).encode()
        req = urllib.request.Request(
            f"{self.backend_url}/api/backlog/task-create",
            data=payload, headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if not (200 <= resp.status < 300):
                    return False
                created = json.loads(resp.read().decode("utf-8", "replace") or "{}")
                # A 2xx with a defaulted name means the payload contract drifted.
                name = str(created.get("name") or created.get("task", {}).get("name") or "")
                return "guardian" in name.lower() if name else True
        except (urllib.error.URLError, OSError, ValueError):
            return False
