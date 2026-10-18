#!/usr/bin/env bash
set -euo pipefail

# Installs the systemd user units by symlinking them from systemd/ to
# ~/.config/systemd/user/, reloads the systemd daemon, and then ARMS every installed unit
# that declares an `[Install]` section (#2774). Arming is the half this script never did:
# it linked and daemon-reloaded and then *printed* the `systemctl --user enable` lines as
# prose, which is why #1891, #2537, #2702 and #2771 each shipped a unit that was installed
# and never ran. The three rules the arming block obeys, and the failure each was learned
# from, are written out beside that block.
#
# There are TWO units, and the second one is a deliberate exception.
#
#   agent-supervisord.service — runs supervisord, which manages every Lloyd
#     service (see supervisor/conf.d/). Individual per-service systemd units
#     were retired; do not expect lloyd-llm.service etc.
#
#   lloyd-guardian.service — the self-modification rollback watchdog. It
#     cannot live under supervisord: agent-supervisord.service sets
#     KillMode=control-group, so every supervisord child dies when that unit
#     restarts or supervisord crashes, which is precisely the situation the
#     guardian exists to survive. It also needs to be able to restart
#     supervisord itself, and a child cannot restart its own supervisor and
#     live. Plus supervisord parks a program in FATAL after startretries and
#     never un-parks it, while systemd Restart=always never gives up.
#
#   thunderbird.service and voxtype.service — two more units that are neither
#     supervisord programs nor watchdogs, but load-bearing Lloyd infra, and so
#     are tracked here (#1109). Both are WantedBy=graphical-session.target:
#     Thunderbird hosts the thunderbird-mcp extension its 40 tools bridge over
#     (:8765) and voxtype is push-to-talk STT, so each needs the desktop's
#     display session and must come up with it, which a headless supervisord
#     child cannot. Before #1109 they existed only as untracked files under
#     ~/.config/systemd/user/, so a rebuild silently lost 40 tools and voice-in.
#     Linking them here (this loop) is what makes a rebuild reproduce them.
#     Since #2774 the arming block below also ENABLES them (bare `enable`, never
#     `--now`: the flag on a service is a start, and a start belongs to the
#     desktop session, which owns graphical-session.target), so the line this
#     script used to echo for them — `systemctl --user enable thunderbird voxtype`
#     — is now performed rather than printed. SETUP.md still carries it, because
#     SETUP.md documents the route for a box that has never run this script.
#
# Anything that is not a watchdog or a graphical-session app still belongs in
# supervisor/conf.d/.
#
# SYSTEM_SCOPE_UNITS are tracked here but are NOT user units, so the loop skips
# them (#1108). nvidia-power-limit.service runs `nvidia-smi -pl`, which needs
# root; it is installed by hand at /etc/systemd/system/ (with its script at
# /usr/local/sbin/set-gpu-power-limit.sh, see its own header). A user-scope link
# can never clamp anything and only makes `systemctl --user restart` look like
# the right command during an Xid-79 investigation.
#
# Lingering must also be enabled or supervisord dies at logout and never starts
# at boot:
#   sudo loginctl enable-linger "$USER"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SYSTEMD_SRC="$PROJECT_DIR/systemd"
SYSTEMD_DST="$HOME/.config/systemd/user"
SYSTEM_SCOPE_UNITS=(nvidia-power-limit.service)
# The repo's TWO deliberate gaps, declared side by side because `scripts/service_health_check.py`
# reads both out of THIS file instead of carrying a second copy of either (#2756, #2949).
#
#   SYSTEM_SCOPE_UNITS is a tracked unit this script refuses to place into the user manager,
#     because clamping GPU power needs root and a user-scope link can only pretend to (#1108).
#   HOST_LOCAL_UNITS is its mirror image: a unit that is LIVE in ~/.config/systemd/user and
#     that the repo deliberately refuses to own. Recording it here is what tells the deploy
#     check the gap is a decision rather than drift. An unlisted live unit reads `untracked:`
#     and turns the `Deploy:` heading degraded; a listed one reads ok and the row quotes the
#     why written on its line. The array is NOT a silencer — it excuses only the names it
#     spells, so a live unit that appears later with nothing behind it still reads untracked.
#
# The shape the reader accepts is strict, and two of its rules fail SILENTLY, so change this
# array only in this shape: one bare deployed unit name per line, the one-line why as that
# line's `#` comment, and the why contains no closing parenthesis and no second hash. The
# reader's capture group is `[^)]*`, so a closing paren anywhere in a why ends the array at
# that character and every entry below it silently goes back to reading untracked; a second
# hash just becomes part of the why. Nothing in bash or systemd notices either mistake.
#
# Pinned by the five HOST_LOCAL_UNITS nodes in tests/test_service_health_check_unit_links.py,
# which read this file through the checker's own parser rather than restating it.
HOST_LOCAL_UNITS=(
    freeclip-a2dp-fix.service  # host-local fix for the FreeClip earbuds A2DP profile, whose ExecStart script hardcodes this box speaker MAC, so the repo could not own a unit that means something on exactly one machine
)
# The guard that asserts every repo timer is enabled, and the read this script's own
# verdict has to agree with (#2774). Absolute, because the installer is run from wherever
# a rebuild happens to be standing.
ENABLEDNESS_GUARD="$(cd "$PROJECT_DIR/.." && pwd)/scripts/maintenance/check-unit-enabledness.sh"

echo "=== Installing Systemd Service Files ==="

mkdir -p "$SYSTEMD_DST"

# Every unit this run placed, as a source path, in link order. The arming block works off
# THIS list and not off a fresh glob of $SYSTEMD_DST, because the deploy directory also
# holds units this repo does not own — `freeclip-a2dp-fix.service` is live on this box with
# no file under agent-services/systemd/, and that disposition is RECORDED, in the
# HOST_LOCAL_UNITS array above, which is where the deploy check reads it from — and arming
# one of those would be this script taking responsibility for a unit it never installed.
LINKED=()

for f in "$SYSTEMD_SRC"/*.service "$SYSTEMD_SRC"/*.timer; do
    [ -e "$f" ] || continue
    name="$(basename "$f")"
    if [[ " ${SYSTEM_SCOPE_UNITS[*]} " == *" $name "* ]]; then
        echo "  $name — system scope, not linked (install to /etc/systemd/system by hand)"
        continue
    fi
    src="$(realpath "$f")"
    dst="$SYSTEMD_DST/$name"

    if [ -L "$dst" ] && [ "$(readlink "$dst")" = "$src" ]; then
        echo "  $name — already linked"
    else
        if [ -e "$dst" ]; then
            echo "  $name — replacing existing file"
            rm "$dst"
        fi
        ln -sf "$src" "$dst"
        echo "  $name — linked"
    fi
    LINKED+=("$src")
done

echo ""
echo "Reloading systemd daemon..."
systemctl --user daemon-reload

# ─── ARMING: a linked unit is not an armed one (#2774) ───────────────────────────────
#
# Everything above placed files; nothing above activated anything. Until now the
# `systemctl --user enable ...` commands lived only in `echo` lines at the foot of this
# file, so a rebuilt box — or this one, which is the measured case — came up with every
# repo timer `disabled` and `scripts/maintenance/check-unit-enabledness.sh` printing
# `FAIL lloyd-egress-window.timer … never-armed`. Three rules, each one paid for:
#
# 1. A `.timer` is armed with `enable --now`; bare `enable` is NOT enough. Plain `enable`
#    links the unit into `timers.target.wants/` and leaves the loaded timer
#    `ActiveState=inactive` until the user manager next restarts. #2537 measured that on
#    `lloyd-persistence-window.timer`, which was armed by the bare form, read `enabled`,
#    and never fired; `scripts/maintenance/check-unit-enabledness.sh:46-58` says the same
#    thing about its own FAIL line. Hence rule 4 below printing both halves.
#
# 2. A `.service` is armed with bare `enable`; `--now` never goes on a service. Since
#    #2541 (2026-10-11) `--now` is classified as the operation systemd performs for it —
#    `enable --now <service>` IS `systemctl start` to the service-control guard, refused
#    to an unattended turn — while `SYSTEMCTL_NOW_UNIT_EXEMPT = ".timer"`
#    (app/harness/service_control.py) exempts the timer form precisely because arming a
#    schedule starts no unit. So the executed `--now` in this script is only ever on a
#    `.timer`, which keeps the whole installer runnable by the turn that reads it.
#    STARTING anything is still a human act, and the two lines at the foot of this file
#    are the start a fresh box needs.
#
# 3. A unit with no `[Install]` section is not armable, and neither is a `.service` that
#    its own `.timer` already starts. Seven co-located services read `static` or `linked`
#    (`lloyd-cert-renew`, `lloyd-data-snapshot`, `lloyd-graph-backup`,
#    `lloyd-guardian-nag`, `lloyd-persistence-window`, `lloyd-qmd-cleanup`,
#    `lloyd-vault-backup`); `enable` on
#    one ERRORS, and with `set -euo pipefail` an unguarded `enable` would kill this script
#    before the status print — so the filter is load-bearing AND every arming call is
#    guarded, counted, and never fatal. The second half is subtler and is why the rule is
#    mechanical rather than a hand-list: `lloyd-egress-window.service` declares
#    `WantedBy=default.target` beside the `lloyd-egress-window.timer` that fires it weekly
#    (Mondays 06:23, `Persistent=true`). Enabling that service puts it into
#    `default.target.wants/`, and the drain-an-hour/timeout-five-hours canary window it
#    runs would then start at EVERY login as well as on its calendar — defeating the
#    weekly cadence #2754 and #2770 exist to hold, and dropping uncontrolled ON-arm rows
#    into `eval/measurements/injection-canary/rows.jsonl`, the corpus #2771 publishes
#    from. A timer-triggered oneshot is `static` by this directory's own convention: its
#    sibling `lloyd-persistence-window.service` carries no `[Install]` at all. So the
#    installer declines to arm any service whose stem has a `.timer` beside it, says so on
#    stdout, and moves on.
#
# 4. `is-enabled` alone is not a verdict. It answers the unit FILE; `ActiveState` answers
#    the loaded unit. #2537's whole lesson is that printing only the first shipped a
#    precedent unarmed while looking green, so each armed unit prints both.
#
# Pinned by tests/test_install_services_arms_units.py, which runs this file against a
# throwaway HOME and a stub `systemctl` that records argv — the same harness
# tests/test_persistence_window_unit.py::_run_installer and
# tests/test_egress_window_unit.py::test_the_installer_links_both_units_by_its_existing_glob
# already use, and the reason a script whose only `enable` sat inside an `echo` could not
# pass either of them.

arm_ok=0
arm_failed=""
arm_unarmed=""

for src in "${LINKED[@]}"; do
    name="$(basename "$src")"
    if ! grep -qs '^\[Install\]' "$src"; then
        # `static`: no [Install] section, so `enable` would error and `set -e` would end
        # this run here. Silent on purpose — the status print below lists every service.
        continue
    fi

    stem="${name%.service}"
    if [[ "$name" == *.service && -f "$SYSTEMD_SRC/$stem.timer" ]]; then
        wanted="$(sed -n 's/^WantedBy=//p' "$src" | head -n 1)"
        echo "  $name — NOT armed: $stem.timer starts it, and enabling it would ALSO pull it into ${wanted:-its target} at every login"
        arm_unarmed="$arm_unarmed $name"
        continue
    fi

    if [[ "$name" == *.timer ]]; then
        # rule 1: `--now` is what arms a timer; bare `enable` leaves it inactive.
        ok=1
        systemctl --user enable --now "$name" || ok=0
    else
        # rule 2: bare `enable`. `--now` here would be a service start (#2541).
        ok=1
        systemctl --user enable "$name" || ok=0
    fi

    # rule 4: both halves, read back from systemd rather than asserted from the call.
    state="$(systemctl --user is-enabled "$name" 2>&1 | head -n 1 || true)"
    state="${state//[[:space:]]/}"
    if [[ -z "$state" ]]; then
        state="no-answer"
    fi
    active="$(systemctl --user show -p ActiveState "$name" 2>/dev/null | sed -n 's/^ActiveState=//p' | head -n 1 || true)"
    if [[ -z "$active" || "$active" == "n/a" ]]; then
        active="no-answer"
    fi

    if [[ "$ok" == 1 ]]; then
        echo "  $name — armed: is-enabled=$state ActiveState=$active"
        arm_ok=$(( arm_ok + 1 ))
    else
        echo "  $name — ENABLE FAILED: is-enabled=$state ActiveState=$active"
        arm_failed="$arm_failed $name"
    fi
done

echo ""
echo "Armed $arm_ok unit(s) of ${#LINKED[@]} installed."
if [[ -n "$arm_unarmed" ]]; then
    echo "  deliberately not armed (started by their own timer):$arm_unarmed"
fi
if [[ -n "$arm_failed" ]]; then
    echo "  ENABLE FAILED:$arm_failed — the units are installed; re-run the enable by hand"
fi

echo ""
echo "Done. Services installed:"
for f in "$SYSTEMD_SRC"/*.service; do
    name="$(basename "$f" .service)"
    status=$(systemctl --user is-active "$name" 2>/dev/null || echo "inactive")
    echo "  $name — $status"
done

LINGER=$(loginctl show-user "$USER" -p Linger --value 2>/dev/null || echo "no")
if [ "$LINGER" != "yes" ]; then
    echo ""
    echo "WARNING: lingering is not enabled for $USER."
    echo "         supervisord will stop at logout and will not start at boot."
    echo "         Enable it with:  sudo loginctl enable-linger $USER"
fi

echo ""
echo "Still a human act, because both of these START something (arm-only was done above):"
echo "  systemctl --user enable --now agent-supervisord.service   # or: systemctl --user start agent-supervisord.service"
echo "  (thunderbird and voxtype are enabled above and started by graphical-session.target, not by this script)"
echo ""
echo "Then check the services supervisord manages:"
echo "  $HOME/.local/share/uv/tools/supervisor/bin/supervisorctl \\"
echo "    -c $PROJECT_DIR/supervisor/supervisord.conf status"

# The enabledness guard's own verdict, printed last so the installer's arithmetic and the
# fleet check's arithmetic cannot diverge (#2774: "end with the check's own exit status
# rather than a hand-rolled count"). Its exit status is REPORTED and never propagated:
# this script's job is to install and arm, and both existing stub-harness tests
# (tests/test_persistence_window_unit.py::_run_installer,
# tests/test_egress_window_unit.py::test_the_installer_links_both_units_by_its_existing_glob)
# assert that a run against a stub `systemctl` exits 0 — under a stub every timer reads
# un-enabled, so propagating the guard's rc would make the installer fail on a box that
# did nothing wrong. Run the guard yourself for a verdict with an exit code:
#   bash scripts/maintenance/check-unit-enabledness.sh; echo rc=$?
echo ""
echo "Enabledness guard (scripts/maintenance/check-unit-enabledness.sh):"
if [ -r "$ENABLEDNESS_GUARD" ]; then
    guard_out="$(bash "$ENABLEDNESS_GUARD" 2>&1)" && guard_rc=0 || guard_rc=$?
    printf '%s\n' "$guard_out"
    echo "guard exit status: $guard_rc (reported only — deliberately NOT this installer's exit code)"
else
    echo "  not found at $ENABLEDNESS_GUARD, so the guard's verdict could not be printed"
fi
