"""`/api/workers/enable` — the switch that used to rewrite config.yaml.

The endpoint did `config.yaml.write_text(yaml.dump(CONFIG))`, and each of the
three consequences was serious on its own:

  * `CONFIG` is the *loaded* config, with `${VAR}` already expanded — so the
    dump would have written `livekit.api_secret` in clear into a tracked file.
  * It flattened every comment out of a 600-line file that is mostly comments.
  * It left the live tree dirty, which `scripts/automod/gate.py` and
    `promote.py` both refuse — one click silently stopping the
    self-modification loop until a human committed the damage.

That is the identical defect the Tools page was moved off `config.yaml` to
avoid (see `test_tool_overrides.py`); this endpoint kept it because nothing in
the frontend calls it yet. `workers.enabled` now travels the same route.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import yaml

import app.config as appconfig
import app.routers.workers as router

ROOT = Path(__file__).resolve().parent.parent


def test_the_endpoint_does_not_write_config_yaml():
    src = inspect.getsource(router.workers_enable)
    code = "\n".join(line for line in src.splitlines()
                     if not line.strip().startswith("#")).split('"""')[2]
    assert "config.yaml" not in code, "the workers switch writes the tracked config again"
    assert "yaml.dump" not in code
    assert "save_tool_overrides()" in code


def test_the_switch_is_persisted_to_the_untracked_override_file(monkeypatch, tmp_path):
    overrides = tmp_path / "tool_overrides.yaml"
    monkeypatch.setattr(appconfig, "TOOL_OVERRIDES_PATH", overrides)
    monkeypatch.setitem(appconfig.CONFIG, "workers", {"enabled": False, "slots": 2})

    appconfig.save_tool_overrides()
    written = yaml.safe_load(overrides.read_text())
    assert written["workers"] == {"enabled": False}
    assert "slots" not in written["workers"], "only the UI-mutable key belongs here"


def test_the_override_decides_what_the_pool_does_on_the_next_boot(monkeypatch, tmp_path):
    overrides = tmp_path / "tool_overrides.yaml"
    overrides.write_text(yaml.dump({"workers": {"enabled": False}}))
    monkeypatch.setattr(appconfig, "TOOL_OVERRIDES_PATH", overrides)

    merged = appconfig._merge_tool_overrides({"workers": {"enabled": True, "slots": 2}})
    assert merged["workers"]["enabled"] is False
    assert merged["workers"]["slots"] == 2, "the override must not replace the block"


def test_a_disagreement_with_the_tracked_config_is_logged(monkeypatch, tmp_path, caplog):
    """Agreement stays silent; a silent win is how the tracked file starts
    describing a state nobody is serving."""
    overrides = tmp_path / "tool_overrides.yaml"
    overrides.write_text(yaml.dump({"workers": {"enabled": False}}))
    monkeypatch.setattr(appconfig, "TOOL_OVERRIDES_PATH", overrides)

    with caplog.at_level("WARNING"):
        appconfig._merge_tool_overrides({"workers": {"enabled": True}})
    assert "workers.enabled" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        appconfig._merge_tool_overrides({"workers": {"enabled": False}})
    assert "workers.enabled" not in caplog.text, "agreement should not warn every boot"


def test_no_secret_can_reach_the_override_file(monkeypatch, tmp_path):
    """The dump wrote the *expanded* config. Only three keys are emitted here,
    so an expanded secret elsewhere in CONFIG cannot ride along."""
    overrides = tmp_path / "tool_overrides.yaml"
    monkeypatch.setattr(appconfig, "TOOL_OVERRIDES_PATH", overrides)
    monkeypatch.setitem(appconfig.CONFIG, "livekit",
                        {"api_secret": "super-secret-value"})
    monkeypatch.setitem(appconfig.CONFIG, "workers", {"enabled": True})

    appconfig.save_tool_overrides()
    text = overrides.read_text()
    assert "super-secret-value" not in text
    assert set(yaml.safe_load(text)) <= {"mcp_servers", "harness", "workers"}


def test_the_override_file_stays_untracked():
    """It now carries `workers.enabled` too, so re-tracking it would arm the
    same trap for a second endpoint."""
    import subprocess
    r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
                        "data/tool_overrides.yaml"], capture_output=True, text=True)
    assert r.returncode != 0, "data/tool_overrides.yaml is tracked again"


def test_config_yaml_holds_no_expanded_secret():
    """A regression canary for the whole class: if some future endpoint dumps
    CONFIG over the tracked file again, the placeholders disappear and this
    fails."""
    raw = (ROOT / "config.yaml").read_text()
    assert "${LIVEKIT_API_SECRET}" in raw, \
        "config.yaml no longer carries its placeholder — a secret may be in the tree"
