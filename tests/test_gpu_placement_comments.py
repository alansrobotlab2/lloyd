"""#1447: the GPU placement comments in the voice stack described a machine
that no longer exists, in the two files a CUDA-pressure debug reaches first.

`start-qwen3-tts.sh` said "(GPU 1)" in its header against its own
`export CUDA_VISIBLE_DEVICES=0`, and `lloyd-agent-worker.conf` justified its
pin with "GPU 2 is now single-tenant" after djev had taken GPU 2 at 0.97
utilisation. Comment-only drift, so the pin there is the same shape as the
drift: the header's number is read off the export it sits above, and the
worker conf's story about GPU 2 is read off `agent-djev.conf`'s own `GPU=`.

#2698 is the half of that property #1447 could not state. It pinned GPU 2's
*tenant name*, so when `agent-tts-omni` moved onto GPU 3 on 2026-10-09 the map
below it stayed green: the worker conf's tenant map listed cards 0, 1 and 2, and a
node that only ever asks about card 2 cannot see a card that is simply absent. The
nodes below therefore stop checking one row's wording and check the map's
COVERAGE, derived from `conf.d/*.conf` — the one thing in the tree that decides
which program sits on which card.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from tests.test_qwen3_tts_omni_launcher import GPU3_UUID

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "agent-services" / "bin" / "start-qwen3-tts.sh"
CONF_D = ROOT / "agent-services" / "supervisor" / "conf.d"
WORKER_CONF = CONF_D / "lloyd-agent-worker.conf"
DJEV_CONF = CONF_D / "agent-djev.conf"
OMNI_CONF = CONF_D / "agent-tts-omni.conf"
PRIMARY_CONF = CONF_D / "agent-llm-primary.conf"
INFRA_DOC = ROOT / "architecture" / "infrastructure.md"
INDEX_DOC = ROOT / "architecture" / "index.md"
PRIMARY_LAUNCHER = ROOT / "agent-services" / "bin" / "start-qwen38-flash-next.sh"

#: The card index each non-numeric pin stands for. Imported rather than retyped:
#: `tests/test_qwen3_tts_omni_launcher.py` already owns the GPU-3 UUID, and a second
#: literal here would make this file yet another copy of the map it is checking.
UUID_TO_CARD = {GPU3_UUID: "3"}

_MAP_LINE = re.compile(r"^\s*;\s*GPU (\d+):\s*(.*)$")
_GLANCE_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*([^|]*)\|\s*(.*)\|\s*$")
_PIN = re.compile(r'(?:\bGPU|CUDA_VISIBLE_DEVICES)="([^"]+)"')


def _env_pin(text: str, key: str) -> str:
    m = re.search(rf'{re.escape(key)}"?(\d+)"?', text)
    assert m, f"{key} not found"
    return m.group(1)


def test_tts_launcher_header_names_the_gpu_it_exports():
    text = LAUNCHER.read_text()
    header = next(ln for ln in text.splitlines() if ln.startswith("# Starts the Qwen3-TTS"))
    claimed = re.search(r"\(GPU (\d+)", header)
    assert claimed, header
    exported = _env_pin(text, "export CUDA_VISIBLE_DEVICES=")
    assert claimed.group(1) == exported, (header, exported)


def test_worker_conf_names_djev_as_gpu_2s_tenant():
    worker = WORKER_CONF.read_text()
    djev = DJEV_CONF.read_text()
    djev_gpu = _env_pin(djev, "GPU=")
    comments = "\n".join(ln for ln in worker.splitlines() if ln.startswith(";"))
    assert "single-tenant" not in comments
    assert re.search(rf"GPU {djev_gpu}: agent-djev", comments), comments
    # The reason the worker is off GPU 1 is the sentence that still binds.
    assert "2026-09-03" in comments and "gpu-memory-utilization" in comments
    # And the doc's GPU table tells the same story about that card.
    row = next(ln for ln in INFRA_DOC.read_text().splitlines()
               if ln.startswith(f"| {djev_gpu} | RTX 3090"))
    assert "agent-djev" in row and "single-tenant" not in row, row


# --- #2698: the hand-maintained tenant map, tied to the confs it describes ----

def _env_line(text: str) -> str:
    """The conf's single `environment=` line, or "" when it sets no environment.

    The scope is the whole point, and it is measured rather than assumed:
    `lloyd-agent-worker.conf`'s own comment quotes `GPU="2"` while explaining why
    this worker must stay OFF that card, so a scan over the whole file files the
    worker as a GPU 2 tenant — a claim the same file's map line contradicts three
    lines later. `test_the_derivation_reads_the_environment_line_not_its_prose`
    pins that difference, and `_naive_pin` exists so the node can demonstrate the
    trap instead of merely describing it."""
    for ln in text.splitlines():
        if ln.strip().startswith("environment="):
            return ln.strip()
    return ""


def _pin(text: str) -> str | None:
    """The raw card pin a conf's `environment=` line sets: a card index, a UUID, or
    None when the conf pins no card there at all. `GPU=` is the supervisord-local
    spelling (`agent-djev`, `agent-llm-secondary`; `agent-decider` pinned the same way
    until #4029 deleted its conf on 2026-11-11), and `CUDA_VISIBLE_DEVICES=` the
    runtime one."""
    m = _PIN.search(_env_line(text))
    return m.group(1) if m else None


def _naive_pin(text: str) -> str | None:
    """What the same search does over a WHOLE conf file. Test-only, and it exists
    only so the node beside it can demonstrate the trap on this tree rather than
    assert it hypothetically."""
    m = _PIN.search(text)
    return m.group(1) if m else None


def _derived_cards() -> dict[str, set[str]]:
    """`{card index: conf stems that pin it}`, read off every conf in `conf.d`.

    A pin that is neither an index nor a UUID listed in `UUID_TO_CARD` raises rather
    than being skipped: silently dropping an unrecognised card is exactly how a
    tenant vanishes from a map while the map still looks complete."""
    cards: dict[str, set[str]] = {}
    for conf in sorted(CONF_D.glob("*.conf")):
        raw = _pin(conf.read_text(encoding="utf-8"))
        if raw is None:
            continue
        idx = UUID_TO_CARD[raw] if raw in UUID_TO_CARD else raw
        assert idx.isdigit(), (
            f"{conf.name} pins {raw!r}, which is neither a card index nor a UUID in "
            "UUID_TO_CARD — resolve it there rather than dropping the tenant")
        cards.setdefault(idx, set()).add(conf.stem)
    assert cards, "no conf in conf.d pins a GPU at all: the extraction is broken, not the map"
    return cards


def _map_rows_of(text: str) -> dict[str, str]:
    """The worker conf's tenant map as `{card index: the rest of that line}`."""
    rows: dict[str, str] = {}
    for ln in text.splitlines():
        m = _MAP_LINE.match(ln)
        if m:
            rows[m.group(1)] = m.group(2).strip()
    assert len(rows) >= 3, f"the tenant map only has lines for {sorted(rows)}: the harvest failed"
    return rows


def _map_rows() -> dict[str, str]:
    return _map_rows_of(WORKER_CONF.read_text(encoding="utf-8"))


def _cards_without_a_line(rows: dict[str, str]) -> dict[str, list[str]]:
    """Every card some conf pins that the given map has no line for — the rot, named."""
    derived = _derived_cards()
    return {c: sorted(derived[c]) for c in derived if c not in rows}


def _glance_rows() -> dict[str, str]:
    """`architecture/index.md` §GPU allocation as `{card: hardware + serves text}`.
    Both cells come back because a row reads as one sentence split across them."""
    rows: dict[str, str] = {}
    in_section = False
    for ln in INDEX_DOC.read_text(encoding="utf-8").splitlines():
        if ln.startswith("## "):
            in_section = ln.startswith("## GPU allocation")
            continue
        if in_section:
            m = _GLANCE_ROW.match(ln)
            if m:
                rows[m.group(1)] = f"{m.group(2)} {m.group(3)}"
    assert len(rows) >= 5, f"index.md's glance map only has rows for {sorted(rows)}"
    return rows


def test_the_map_has_a_line_for_every_card_the_confs_pin():
    """The property #1447 could not state: a card that appears in `conf.d` cannot be
    absent from the worker conf's tenant map. GPU 3 is the case in point —
    `agent-tts-omni.conf` has pinned it since 2026-10-09 while the map had no line for
    it, and #2698 was filed because a reader of that map could not find the card the
    voice path was actually running on."""
    missing = _cards_without_a_line(_map_rows())
    assert not missing, (
        f"these cards have a conf pinning them and no line in the tenant map: {missing}. "
        "This map is what a CUDA-pressure debug reads first, so an omitted card reads "
        "as a spare card.")


def test_the_map_harvester_notices_a_deleted_card_line():
    """The node above must be able to fail, and it must fail the way the real rot
    looked: the GPU 3 line simply gone while its conf stays alive. Cut in memory, so
    the shipped conf is never edited to prove a test works."""
    text = WORKER_CONF.read_text(encoding="utf-8")
    trimmed = "".join(ln for ln in text.splitlines(keepends=True)
                      if not re.match(r"^\s*;\s*GPU 3:", ln))
    assert trimmed != text, "the map has no GPU 3 line to delete, so this control is toothless"
    assert _cards_without_a_line(_map_rows_of(trimmed)) == {"3": ["agent-tts-omni"]}


def test_the_derivation_reads_the_environment_line_not_its_prose():
    """Measured trap: this conf's comment quotes `GPU="2"` to explain why the worker
    is NOT on GPU 2. A whole-file scan therefore files it as a GPU 2 tenant, and a
    derivation built that way disagrees with the map it exists to check — so the parse
    is scoped to `environment=`, and this node keeps that difference honest rather than
    invisible. Dropping the scope makes this node and
    `test_the_gpu_2_row_names_every_conf_that_pins_gpu_two` fail together."""
    text = WORKER_CONF.read_text(encoding="utf-8")
    assert _naive_pin(text) == "2", "the quoted pin this trap depends on has moved out of the comment"
    assert _pin(text) == "0", "the worker's real card is the one its environment= line sets"
    assert "lloyd-agent-worker" not in _derived_cards().get("2", set())
    assert "lloyd-agent-worker" in _derived_cards()["0"]


def test_the_map_covers_the_cards_this_box_has_tenants_on():
    """#1447's second half, still unverified when #2698 was filed: the map read
    `GPU 0 / GPU 1 / GPU 2` and nothing else. The map's set is the cards with a pinned
    tenant, and it grows only when a conf names a new one — card 4 joined on 2026-11-10,
    when qmd moved there."""
    assert sorted(_map_rows()) == ["0", "1", "2", "3", "4"], sorted(_map_rows())


def test_the_gpu_3_row_names_the_voice_tts_and_when_it_arrived():
    """The row #1447 never got: which program, which port, and since when — the three
    things the rows for cards 0, 1 and 2 already answer."""
    row = _map_rows()["3"]
    assert "agent-tts-omni" in row and ":8093" in row, row
    assert "2026-10-09" in row, row


def test_the_gpu_2_row_names_every_conf_that_pins_gpu_two():
    """GPU 2 is the one card with a queue for it, and until #4029 removed
    `agent-decider.conf` on 2026-11-11 the map named two claimants where three confs
    pinned it: the hand-started third appeared nowhere, and an operator could stop
    djev for it without knowing the option existed. Generic on purpose — a fourth
    claimant must be named here too, or the map goes red instead of quietly going
    stale."""
    pinned = sorted(_derived_cards()["2"])
    row = _map_rows()["2"]
    unnamed = [c for c in pinned if c not in row]
    assert not unnamed, (
        f"GPU 2 is pinned by {pinned} and the map's line omits {unnamed}: {row!r}")
    assert pinned == ["agent-djev", "agent-llm-secondary"], (
        f"GPU 2 is pinned by {pinned} since #4029 took agent-decider.conf out of "
        "conf.d on 2026-11-11; this set is the card's whole queue")


def test_the_uuid_pinned_conf_resolves_to_the_card_the_map_names():
    """`agent-tts-omni.conf` pins by UUID, so a derivation that understood only
    integers would find no tenant on GPU 3 — the same hole as a missing row, reached
    from the other side. The UUID→index pair is imported, not retyped, and nothing here
    asks the live box: `nvidia-smi` would make these assertions about today's hardware
    instead of about the tree, which is the rule
    `tests/test_service_health_check_gpu_power_drift.py` keeps by monkeypatching its
    reader rather than calling it."""
    assert _pin(OMNI_CONF.read_text(encoding="utf-8")) == GPU3_UUID
    assert UUID_TO_CARD[GPU3_UUID] == "3"
    assert _derived_cards()["3"] == {"agent-tts-omni"}
    assert "agent-tts-omni" in _map_rows()["3"]
    # Read off the syntax tree, not off the file's text: the sentence naming this rule
    # contains the words it forbids, so a substring search over the source is true of
    # the file precisely because the check exists (measured — it failed its own node).
    imported = {a.name for n in ast.walk(ast.parse(Path(__file__).read_text(encoding="utf-8")))
                if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    assert "subprocess" not in imported, \
        "these assertions are about the tree; a shell-out would make them about the box"


def test_the_conf_that_pins_no_card_is_excluded_from_the_derivation():
    """`agent-llm-primary.conf` sets no card in its `environment=` line at all: the pin
    is launcher-side, an `export CUDA_VISIBLE_DEVICES=1` in its start script, which
    `architecture/infrastructure.md` records in prose. A derivation that demanded a
    card for every conf would be red on the tree as it stands, so the exclusion is
    asserted rather than left to fall out of the loop — this is a property of the
    parse, not a card to be invented."""
    assert _pin(PRIMARY_CONF.read_text(encoding="utf-8")) is None
    assert "agent-llm-primary" not in {c for v in _derived_cards().values() for c in v}
    assert re.search(r"^export CUDA_VISIBLE_DEVICES=1$",
                     PRIMARY_LAUNCHER.read_text(encoding="utf-8"), re.M), \
        "the primary's card moved into the launcher; this node's premise went with it"
    assert "agent-llm-primary" in _map_rows()["1"]


def test_the_index_glance_map_names_every_gpu_2_candidate():
    """`architecture/index.md` is the copy an operator reads, and for a while its GPU 2
    row offered two candidates for a card three confs pinned — the same two-candidate
    sentence the worker conf carried, which is how the decider came to be named in no
    operator-facing map at all. #4029 has since taken the third conf out, so the row
    and the confs are back to agreeing at two. Only GPU 2 is pinned this way: it is the
    one card where the question is *which of these*, while other rows name a role."""
    row = _glance_rows()["2"]
    pinned = sorted(_derived_cards()["2"])
    unnamed = [c for c in pinned if c not in row]
    assert not unnamed, f"index.md's GPU 2 row omits {unnamed} of {pinned}: {row!r}"


# --- #4473: qmd's card, derived across BOTH surfaces a runner can set it ------

SYSTEMD = ROOT / "agent-services" / "systemd"
QMD_DAEMON_CONF = CONF_D / "agent-qmd-daemon.conf"
CLEANUP_UNIT = SYSTEMD / "lloyd-qmd-cleanup.service"

#: Where a qmd runner lives, and the glob that names it. Globs rather than a list,
#: because #4473 is exactly the drift a list permits. `a372c876` (2026-11-10, "move
#: qmd's whole stack onto the 3090 at index 4") rewrote `agent-qmd-daemon.conf` and
#: `agent-qmd-watcher.conf` and never touched the nightly cleanup unit, and the only
#: guard in the tree — `_derived_cards` above — iterates `conf.d/*.conf`, so the file
#: that went stale was never opened by anything that could have noticed it. A fourth
#: qmd runner has to be swept the day it appears, not the day someone remembers this.
QMD_RUNNER_SURFACES = {CONF_D: "agent-qmd-*.conf", SYSTEMD: "lloyd-qmd-*"}

#: supervisord spells the key `environment=` and systemd spells it `Environment=`.
#: Both spellings in one reader, because a reader that recognises only one of them is
#: the same blind spot redrawn — and every line, because systemd applies all its
#: `Environment=` keys where an INI reader keeps the last one.
_ENV_KEY = re.compile(r"^\s*(?:environment|Environment)\s*=")

#: The assignment qmd's runners use on both surfaces. `GPU="2"` — supervisord's local
#: spelling, what `_PIN` above exists to read — names no qmd runner today, and one
#: that started using it would be reported by `_runner_pin` as a runner with no pin.
#: The value ends at a comma as well as at whitespace because supervisord joins its
#: pairs with commas: `CUDA_VISIBLE_DEVICES="4",LD_LIBRARY_PATH="…"` is one field, not
#: a card list, and a value that ran to the end of the line read every later variable
#: as a card.
_CUDA_ASSIGN = re.compile(r"CUDA_VISIBLE_DEVICES=([^\s,]+)")

#: The lines that make a file actually run something, on either surface.
_EXEC_KEY = re.compile(r"^\s*(?:ExecStart|command)\s*=")


def _env_declarations(text: str) -> list[str]:
    """Every environment-setting line in a runner file, on either surface.

    Both surfaces and every line are needed at once: `lloyd-qmd-cleanup.service`
    splits its environment over three `Environment=` lines with the card on neither
    the first nor the last, so a reader scoped to one spelling or one line reads a
    runner while missing the part of it that matters. `_env_line` above is the
    conf-surface reader #1447 and #2698 needed and is left alone; this is the one
    #4473 needed."""
    return [ln.strip() for ln in text.splitlines() if _ENV_KEY.match(ln)]


def _cuda_cards_in(text: str) -> set[str]:
    """Every card INDEX a `CUDA_VISIBLE_DEVICES=` assignment names in `text`.

    Quotes stripped, so supervisord's `CUDA_VISIBLE_DEVICES="4"` and systemd's bare
    `CUDA_VISIBLE_DEVICES=4` parse alike; comma lists split, because
    `CUDA_VISIBLE_DEVICES=0,1` names two cards and a guard that read it as one would
    miss which one it is talking about; and only digit tokens count, because a UUID
    pin (`agent-tts-omni.conf`, read by `UUID_TO_CARD` above) is not an index and
    guessing at one here would be a second, lazier resolution of the same question.
    Whole-file, because the clause this serves is about the retired index surviving
    anywhere under `agent-services/systemd/`."""
    return {card for m in _CUDA_ASSIGN.finditer(text.replace('"', ""))
            for card in m.group(1).split(",") if card.isdigit()}


def _runner_pin(text: str) -> str | None:
    """The one card a runner's own environment declarations pin, or None when they
    pin none. Scoped to the declarations and not to the file:
    `test_the_derivation_reads_the_environment_line_not_its_prose` above is the
    measured reason a whole-file scan is the wrong instrument for a pin.

    More than one card raises rather than resolves: every node in this section asks
    which card qmd is on, and a runner that spreads across two has made that question
    ambiguous — that is a rewrite of these nodes on purpose, not a guess to be filed
    as a pass."""
    cards = _cuda_cards_in("\n".join(_env_declarations(text)))
    assert len(cards) <= 1, (
        f"a runner pins {sorted(cards)}: this section asks which card qmd runs on, so a "
        "multi-card runner needs the question restated rather than answered arbitrarily")
    return next(iter(cards)) if cards else None


def _qmd_runners() -> dict[str, str]:
    """`{file name: card index}` for every qmd runner, gathered from both surfaces.

    A file that runs nothing is excluded by its shape rather than by its name:
    `lloyd-qmd-cleanup.timer` only schedules the unit and systemd gives a timer no
    process of its own, so there is no environment for a timer to pin — asserted
    rather than assumed, so the exclusion cannot quietly become an omission. Anything
    that does run something must pin a card, because a runner with no pin inherits
    whatever card its caller happened to have."""
    runners: dict[str, str] = {}
    for directory, pattern in QMD_RUNNER_SURFACES.items():
        for path in sorted(directory.glob(pattern)):
            text = path.read_text(encoding="utf-8")
            if not any(_EXEC_KEY.match(ln) for ln in text.splitlines()):
                assert not _env_declarations(text), (
                    f"{path.name} runs nothing yet sets an environment — this exclusion "
                    "is for timers, not for a runner that lost its pin")
                continue
            pin = _runner_pin(text)
            assert pin is not None, (
                f"{path.name} runs qmd with no card pinned in its environment")
            runners[path.name] = pin
    assert runners, "no qmd runner pins a card on either surface: the glob or the parse is broken"
    return runners


def _off_card(runners: dict[str, str]) -> dict[str, str]:
    """`{name: card}` for every qmd runner disagreeing with the daemon's card. The
    daemon is the reference because it is the long-lived process holding qmd's
    embedder and reranker — the one the 2026-11-10 move was making room for."""
    home = runners["agent-qmd-daemon.conf"]
    return {name: card for name, card in runners.items() if card != home}


def test_qmd_runners_are_swept_from_both_surfaces_and_agree_on_one_card():
    """Clause 2. qmd is one stack, so the predicate is one card, and the denominator
    is a glob over both surfaces because the drift this node exists for was a file on
    the surface no guard read. Today's three runners are a floor, not a ceiling: a
    fourth runner is swept, not rejected."""
    runners = _qmd_runners()
    confs = {n for n in runners if n.endswith(".conf")}
    units = {n for n in runners if n.endswith(".service")}
    assert confs and units, f"the sweep reached only one surface: {sorted(runners)}"
    assert {"agent-qmd-daemon.conf", "agent-qmd-watcher.conf",
            "lloyd-qmd-cleanup.service"} <= set(runners), sorted(runners)
    off = _off_card(runners)
    assert not off, (
        f"these qmd runners are on another card than agent-qmd-daemon.conf: {off}. The "
        "exposure is latent rather than current — cleanup binds no model today — but the "
        "day it loads the embedder or reranker, an off-card runner puts 6+ GiB on the "
        "desktop card a372c876 moved qmd off precisely to stop doing.")


def test_the_sweep_goes_red_when_the_unit_slips_back_off_the_daemons_card():
    """The node above has to be able to fail, and fail the way the real drift looked:
    the unit naming card 0 while both confs stay on the daemon's card. The drift is
    cut into the unit's text in memory, so the shipped unit is never edited to prove a
    test works, and the cut is asserted to bite — the same control
    `test_the_map_harvester_notices_a_deleted_card_line` runs."""
    runners = _qmd_runners()
    card = runners["agent-qmd-daemon.conf"]
    text = CLEANUP_UNIT.read_text(encoding="utf-8")
    drifted = re.sub(rf"CUDA_VISIBLE_DEVICES={card}", "CUDA_VISIBLE_DEVICES=0", text)
    assert drifted != text, (
        f"the unit no longer sets CUDA_VISIBLE_DEVICES={card} where this cut expects it, "
        "so the control below proves nothing")
    assert _runner_pin(drifted) == "0", "the drifted text does not read as card 0"
    off = _off_card({**runners, CLEANUP_UNIT.name: _runner_pin(drifted)})
    assert off == {CLEANUP_UNIT.name: "0"}, off


def test_the_conf_only_guard_could_not_have_seen_this_drift():
    """Not a hypothesis: the guard that went green on the 2026-11-10 move is
    `_derived_cards` above, and these are its blind spots measured on the file that
    rotted. Its directory is `conf.d`, so a unit is never opened; and handed the text
    anyway, `_PIN` demands a quoted value while systemd writes the pin bare. Both
    assertions fail the moment someone widens the old reader, which is the point — the
    hole gets closed, not duplicated around."""
    text = CLEANUP_UNIT.read_text(encoding="utf-8")
    assert _env_line(text) == "", "the conf-surface reader now sees this unit: the trap moved"
    assert _PIN.search(text) is None, "the quoted-only pin regex now matches a systemd line"
    assert "lloyd-qmd-cleanup" not in {c for v in _derived_cards().values() for c in v}
    assert _runner_pin(text) == _qmd_runners()["agent-qmd-daemon.conf"]


def test_the_cleanup_unit_pins_the_card_the_qmd_daemon_conf_pins():
    """Clause 1. One card for the unit and the daemon conf, DERIVED rather than
    compared to a literal: the daemon conf is where qmd's card lives, so a hard-coded
    4 in this node would only re-create the drift the next time qmd moves."""
    unit_card = _runner_pin(CLEANUP_UNIT.read_text(encoding="utf-8"))
    daemon_card = _runner_pin(QMD_DAEMON_CONF.read_text(encoding="utf-8"))
    assert unit_card == daemon_card, (
        f"lloyd-qmd-cleanup.service runs qmd on card {unit_card} while "
        f"agent-qmd-daemon.conf runs it on card {daemon_card}")


def test_no_cuda_visible_devices_zero_survives_under_agent_services_systemd():
    """Clause 1's second half, taken literally. `0` is not "none" — it is a specific
    card, the 3090 the desktop sits on with 3653 MiB in use as this item was filed,
    which is the card qmd was moved off. Whole files, so a comment that spells the
    retired assignment trips this too; `GPU 0` is the spelling for prose."""
    offenders = sorted(
        str(p.relative_to(SYSTEMD)) for p in SYSTEMD.rglob("*")
        if p.is_file() and "0" in _cuda_cards_in(p.read_text(encoding="utf-8",
                                                              errors="replace")))
    assert not offenders, (
        f"these files under agent-services/systemd/ still assign CUDA_VISIBLE_DEVICES to "
        f"card 0: {offenders}. qmd's card is derived from agent-qmd-daemon.conf; a unit "
        "naming a card of its own is how #4473 happened.")


def test_the_cleanup_unit_keeps_every_device_free_cuda_variable():
    """Clause 3. The card index was the only wrong thing about this unit's
    environment; the rest is load-bearing while it binds no model. The fork's CLI
    imports its llama.cpp bindings at module load and disposes them on the shared exit
    path (`qmd/src/cli/qmd.ts`, in the gitignored fork clone, not in this repo's
    tree), so the library paths are what let that import resolve on a job that
    currently succeeds every night. These five strings are asserted because the fix
    shape this item rejected was the one that deleted them."""
    env = "\n".join(_env_declarations(CLEANUP_UNIT.read_text(encoding="utf-8")))
    for decl in ("CUDA_DEVICE_ORDER=PCI_BUS_ID", "QMD_LLAMA_GPU=cuda",
                 "LD_LIBRARY_PATH=/usr/lib:/opt/cuda/lib64",
                 "CUDA_PATH=/opt/cuda", "CUDA_HOME=/opt/cuda"):
        assert decl in env, f"{decl!r} is no longer in the unit's environment: {env!r}"


def test_the_cleanup_unit_names_its_card_and_says_cleanup_loads_no_model():
    """Clause 4. The comment is the half a future card move has to rewrite, so it is
    checked against the derived card instead of a literal: move qmd's card and this
    unit's own sentence goes red with the confs, which is the sweep the item asked for
    turned into a node."""
    unit = CLEANUP_UNIT.read_text(encoding="utf-8")
    card = _qmd_runners()[CLEANUP_UNIT.name]
    comments = "\n".join(ln for ln in unit.splitlines() if ln.lstrip().startswith("#"))
    assert re.search(rf"GPU {card}\b", comments), (
        f"nothing in the unit's comments names GPU {card}, the card its own "
        "Environment= line sets")
    assert "qmd cleanup" in comments and "binds no model" in comments and "today" in comments, (
        f"the comment must say that qmd cleanup binds no model today: {comments!r}")
    assert "conf.d" in comments, (
        "the comment must name the other surface a card move sweeps: the unit is one of "
        "the files that set qmd's card, not the only one")
    assert "tests/test_gpu_placement_comments.py" in comments, (
        "the comment must point at the node that derives the card across both surfaces")


