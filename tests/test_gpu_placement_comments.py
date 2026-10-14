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
    spelling (`agent-djev`, `agent-decider`, `agent-llm-secondary`), and
    `CUDA_VISIBLE_DEVICES=` the runtime one."""
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
    `GPU 0 / GPU 1 / GPU 2` and nothing else. Card 4 is absent on purpose — no conf
    pins it, and `architecture/index.md` says so in words — so the map's set is the
    cards with a pinned tenant, and it grows only when a conf names a new one."""
    assert sorted(_map_rows()) == ["0", "1", "2", "3"], sorted(_map_rows())


def test_the_gpu_3_row_names_the_voice_tts_and_when_it_arrived():
    """The row #1447 never got: which program, which port, and since when — the three
    things the rows for cards 0, 1 and 2 already answer."""
    row = _map_rows()["3"]
    assert "agent-tts-omni" in row and ":8093" in row, row
    assert "2026-10-09" in row, row


def test_the_gpu_2_row_names_every_conf_that_pins_gpu_two():
    """GPU 2 is the one card with a queue for it, and the map named two claimants
    where three confs pin it: `agent-decider.conf` (`autostart=false`, started by hand
    while it is measured against djev) appeared nowhere, so the comment described a
    two-candidate card and an operator could stop djev for the decider without knowing
    the third option exists. Generic on purpose — a fourth claimant must be named here
    too, or the map goes red instead of quietly going stale."""
    pinned = sorted(_derived_cards()["2"])
    row = _map_rows()["2"]
    unnamed = [c for c in pinned if c not in row]
    assert not unnamed, (
        f"GPU 2 is pinned by {pinned} and the map's line omits {unnamed}: {row!r}")
    assert pinned == ["agent-decider", "agent-djev", "agent-llm-secondary"], pinned


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


def test_the_index_glance_map_names_all_three_gpu_2_candidates():
    """`architecture/index.md` is the copy an operator reads, and its GPU 2 row offered
    two candidates for a card three confs pin — the same two-candidate sentence the
    worker conf carried, which is how the decider came to be named in no operator-facing
    map at all. Only GPU 2 is pinned this way: it is the one card where the question is
    *which of these three*, while the other rows describe their tenants by role."""
    row = _glance_rows()["2"]
    pinned = sorted(_derived_cards()["2"])
    unnamed = [c for c in pinned if c not in row]
    assert not unnamed, f"index.md's GPU 2 row omits {unnamed} of {pinned}: {row!r}"
