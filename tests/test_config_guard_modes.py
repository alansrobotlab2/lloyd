"""#1948 (a) — a guard mode this build cannot honour is a boot failure.

Two guards read a `mode` key out of the config on every turn:
`app/harness/action_review.py` (`harness.action_review.mode`) and
`agent_mcp/_injection_probe.py` (`harness.injection_probe.mode`). Both used to
resolve it with `mode if mode in MODES else DEFAULT_MODE`, so a typo was
indistinguishable from the setting it replaced: the process came up, the guard
ran at `shadow`, and the only surface that could show the miss was a corpus with
no rows in it — which reads exactly like a window in which nothing was flagged.

`app/config.py::validate_guard_modes` is the refusal. Pinned here in the two
shapes that matter: a **process** that does not come up over a typo (the
subprocess nodes below — an in-process call to the validator would only prove a
function raises, not that boot runs it), and a config that *does* boot with the
key absent, with each value the readers implement, and with `warn` on either key
(a value #1944 removed from `action_review`'s behaviour while a deployment may
still name it).

The last node pins the drift that would make this check a landmine: a reader's
own `MODES` has to stay inside the set the boot check accepts, or boot would
refuse a value the guard actually implements.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
VALID = ("off", "shadow", "warn")
KEYS = ("harness.action_review.mode", "harness.injection_probe.mode")


def _overlay(tmp_path: Path, key: str, value) -> Path:
    """A `$LLOYD_CONFIG_OVERLAY` file setting one guard-mode key. The overlay is
    deep-merged over `config.yaml` (`app/config.py::_apply_config_overlay`), so
    this is the same route a canary takes to change config without editing the
    tracked file — which is exactly the route a typo reaches production by."""
    server, block, leaf = key.split(".")
    path = tmp_path / f"overlay-{block}.yaml"
    path.write_text(yaml.dump({server: {block: {leaf: value}}}))
    return path


def _boot(overlay: Path | None = None, *, timeout: int = 180) -> subprocess.CompletedProcess:
    """Run the boot: import `app.config` in a fresh interpreter. Every service on
    this box imports it — the two guards themselves do — so a refusal here is the
    process not coming up, not a library complaining."""
    env = dict(os.environ)
    if overlay is not None:
        env["LLOYD_CONFIG_OVERLAY"] = str(overlay)
    else:
        env.pop("LLOYD_CONFIG_OVERLAY", None)
    return subprocess.run([sys.executable, "-c", "import app.config"],
                          cwd=str(ROOT), env=env, capture_output=True, text=True,
                          timeout=timeout)


# ---------------------------------------------------------------- refuse boot --

def test_a_typo_in_the_reviewer_mode_refuses_to_boot(tmp_path):
    out = _boot(_overlay(tmp_path, "harness.action_review.mode", "shodow"))
    assert out.returncode != 0, (
        "boot came up on `mode: shodow`; the recorder would run at the default "
        "and an empty corpus would read as a clean window")
    err = out.stderr
    assert "harness.action_review.mode" in err
    assert "shodow" in err
    for mode in VALID:
        assert mode in err, f"the refusal did not name the valid value {mode!r}"


def test_a_typo_in_the_probe_mode_refuses_to_boot(tmp_path):
    out = _boot(_overlay(tmp_path, "harness.injection_probe.mode", "shawdow"))
    assert out.returncode != 0, (
        "boot came up on `mode: shawdow`; the probe would run in shadow, and on "
        "the one guard that can append a <warning> to a tool result that is the "
        "more consequential of the two silent fallbacks")
    err = out.stderr
    assert "harness.injection_probe.mode" in err
    assert "shawdow" in err
    for mode in VALID:
        assert mode in err, f"the refusal did not name the valid value {mode!r}"


@pytest.mark.parametrize("key", KEYS)
def test_the_refusal_names_the_key_the_value_and_every_valid_value(key):
    """The same refusal, read as a string rather than as an exit code, so the
    wording an operator has to act on is pinned, not just the fact of failing."""
    from app.config import validate_guard_modes
    cfg = {"harness": {key.split(".")[1]: {"mode": "Shadoww"}}}
    with pytest.raises(RuntimeError) as exc:
        validate_guard_modes(cfg)
    message = str(exc.value)
    assert key in message
    assert "Shadoww" in message
    assert all(mode in message for mode in VALID), message


# -------------------------------------------------------------------- boot ok --

@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("mode", VALID)
def test_every_valid_value_boots(key, mode):
    from app.config import validate_guard_modes
    validate_guard_modes({"harness": {key.split(".")[1]: {"mode": mode}}})


def test_a_missing_mode_key_boots_and_the_reader_defaults_to_shadow():
    """Absent is the documented state, not a misconfiguration: both guards still
    default to `shadow`, so the refusal fires only on a value that is present and
    cannot be honoured."""
    from app.config import validate_guard_modes
    import app.harness.action_review as AR
    from agent_mcp import _injection_probe as P

    for cfg in ({}, {"harness": {}}, {"harness": {"action_review": {}}},
                {"harness": {"action_review": {"mode": None}}},
                {"harness": {"injection_probe": {"mode": None}}}):
        validate_guard_modes(cfg)
    assert AR.DEFAULT_MODE == P.DEFAULT_MODE == "shadow"


def test_the_shipped_config_boots_under_the_check():
    """The check must not be a new way to break a working deployment: the
    tracked `config.yaml`, unmodified, comes up."""
    out = _boot()
    assert out.returncode == 0, out.stderr[-2000:]


def test_a_canary_overlay_naming_warn_on_either_key_boots(tmp_path):
    """Across the process boundary, and for the one value with a history.

    #1944 took `warn` out of `action_review`'s `MODES` but a config written
    before that ruling may still name it, and #1948 requires boot to accept each
    documented mode. Accepting it is not re-opening the switch: the module maps
    `warn` onto the mode it always ran as, which
    `tests/test_action_review.py::test_a_config_still_naming_warn_gets_shadow_not_a_gate`
    pins, and this node pins that the deployment is not bricked over it.
    """
    first = _overlay(tmp_path, "harness.action_review.mode", "warn")
    assert _boot(first).returncode == 0, "boot refused `warn`, which #1944 left valid"
    second = _overlay(tmp_path, "harness.injection_probe.mode", "off")
    assert _boot(second).returncode == 0, "boot refused `off`"


# ----------------------------------------------------------------------- drift --

def test_each_readers_own_modes_stays_inside_what_boot_accepts():
    """The one way this check could be worse than the fail-open it replaced.

    If a reader grew a mode that `GUARD_MODE_KEYS` did not list, boot would
    refuse a value the guard implements — a refusal that is loud, but wrong, and
    payable only by someone who never touched either file. Each reader names its
    own key (`CONFIG_KEY`) and its own behavioural set (`MODES`); this pins both
    against the boot inventory, and pins the deprecated map on top of it: every
    value `action_review` maps must itself be boot-valid and its target a mode
    the module implements.
    """
    from app.config import GUARD_MODE_KEYS
    import app.harness.action_review as AR
    from agent_mcp import _injection_probe as P

    for module in (AR, P):
        valid = GUARD_MODE_KEYS[module.CONFIG_KEY]
        assert set(module.MODES) <= set(valid), module.CONFIG_KEY
        assert module.DEFAULT_MODE in valid, module.CONFIG_KEY
    assert set(AR.DEPRECATED_MODES) <= set(GUARD_MODE_KEYS[AR.CONFIG_KEY])
    assert set(AR.DEPRECATED_MODES.values()) <= set(AR.MODES)
    assert AR.DEPRECATED_MODES.get("warn") == "shadow"


def test_the_read_path_and_the_boot_check_agree_on_one_value():
    """One dotted path, resolved once (`read_guard_mode`), so the guard can never
    see a different config from the one that was validated."""
    from app.config import read_guard_mode, validate_guard_modes
    cfg = {"harness": {"action_review": {"mode": "  Warn "}}}
    validate_guard_modes(cfg)
    assert read_guard_mode(cfg, "harness.action_review.mode") == "warn"
    assert read_guard_mode({}, "harness.action_review.mode") == ""


def test_the_shipped_config_resolves_both_guard_modes_to_shadow():
    """#1962 clause 5: the comment rewrite above `harness.action_review` moved no
    value. Read from the tracked file itself, not from `CONFIG`, so an overlay
    in the environment cannot make this green."""
    from app.config import validate_guard_modes

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    validate_guard_modes(cfg)
    assert cfg["harness"]["action_review"]["mode"] == "shadow"
    assert cfg["harness"]["injection_probe"]["mode"] == "shadow"
