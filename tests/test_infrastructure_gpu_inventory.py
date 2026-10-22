"""#2697: `architecture/infrastructure.md` §GPUs is the card inventory's one
owner, and until this round it carried three rows against a five-card box.

`2e08e6d3` ("moar gpus", 2026-10-07) added two RTX 3090s and updated
`architecture/vllm.md` §2's power-clamp table to five rows with their PCI bus
ids, but never touched §GPUs. `agent-tts-omni` then landed on GPU 3 on
2026-10-09 and still appeared nowhere in this doc: not in §GPUs, not in
§Process management's program table, not in §Ports — so a reader of the port
table alone concluded the voice TTS was `:8090`. `architecture/index.md` §GPU
allocation was corrected on 2026-10-14 and now names this table as its owner
("The owner of the card-by-card map ... is [[infrastructure]] § GPUs"), which is
what made the gap load-bearing: the pointer was right and the thing it pointed
at was three rows. `tests/test_architecture_index_parity.py` deliberately
exempts a GPU table from its no-owned-count rule, so nothing went red — which is
exactly how the §GPUs table rotted in the first place.

Every assertion reads the doc against a source **in the tree**, never against
the live box — the rule the other GPU tests keep
(`tests/test_service_health_check_gpu_power_drift.py` monkeypatches the
nvidia-smi reader rather than calling it). Three sources:

  * `architecture/index.md` §GPU allocation and `architecture/vllm.md` §2 — the
    two other copies of this roster, both already at five rows. Agreement
    between the copies is the check; the doc-claim files are the precedent for
    pinning a doc to the tree rather than to a sentence.
  * `agent-services/supervisor/conf.d/*.conf` — what actually decides which
    program is on which card. A tenant the confs do not pin is a tenant the doc
    invented; a card no conf pins must be said so out loud.
  * the program's own launcher, for the port it serves and the date it moved.

Each harvester asserts it extracted something, and
`test_the_row_harvester_can_report_a_missing_card` feeds this file a table with
GPU 3 and 4 deleted: a green from a regex that matched nothing is the failure
mode the doc's own review log records.

#2712 extends the same discipline to the two tables that answer "what is
running", which until now were pinned by nothing but one hardcoded row about the
omni TTS. `6ed8753c` (2026-10-14, "arch-review: doc:infrastructure — stale")
added the `agent-djev` and `agent-decider` program rows and their `8010 / 8011`
and `8012` port rows — 100 insertions in one file with **no test file**, which is
the unpinned hand-edit this item is about: the symptom fixed, the mechanism
untouched. So the expected set is now derived from `conf.d/` itself (the 14
`[program:…]` files, not `lloyd-mc.conf`, which is a `[group:…]`), the serving
ports from those confs' `environment=…PORT=…` (8010, 8011, 8012, 8091 — not
`VLLM_PORT=`, which is 5300/5310, an internal vLLM port no §Ports row carries),
and the port rows are read as cells, because `| 7880 / 7881 |`,
`| 8010 / 8011 |` and `| 18080 / 18500 |` spell two ports where the
`^\\|\\s*(\\d+)\\s*\\|` shape this file used for them sees nothing at all.

The direction is one-way on purpose: §Ports legitimately lists ports no conf
declares (`8765` is the Thunderbird bridge, `18080 / 18500` the automod canary),
so requiring a conf for every row is unsatisfiable, while
requiring a row for every program and every serving port is the check that was
missing. Its one live gap was `8501`, listed in §Process management for
`lloyd-agent-worker` and absent from §Ports — a reader of the port table alone
could not find the thing at all.

#3137 is the same failure in the table this file chose not to read. `_clamp_cards`
scopes itself to `architecture/vllm.md` §2 by the clamp table's header, and its
docstring says the reason is that §2 carries a *second* index-keyed table whose
columns would come back in the wrong slots. That exclusion was structural, and
nothing took it up, so `architecture/vllm.md` §2.4 — the `nvidia-smi` index / PCI bus
/ `/dev/nvidiaN` / tenant map, the one table on the box that answers "which card is
this process actually on" — stayed at three rows against a five-card box from
`2e08e6d3` (2026-10-07) until `55f6eda1` (2026-10-21) rewrote it by hand: it gave
index 2 the bus id `61:00.0`, which is index 3's card, and gave index 2 the tenant
"the llama.cpp secondary", switched off since 2026-09-20. The nodes at the bottom of
this file harvest that table by its own header and pin it four ways: row count against
`CARD_COUNT`, its PCI column against §2.3's copy of the same bus ids, its tenants
against the pin each named program actually sets (conf `environment=`, the launcher
its `command=` runs, or a UUID attributed by that launcher's own `# GPU <n>` line),
and its "nothing pins it" cells against the set of cards conf.d leaves unowned. The
`/dev/nvidiaN` column is checked for shape only: outside §2.4's own rows,
`git grep '/dev/nvidia[0-9]'` returns nothing, so no source in the tree holds that
mapping and this file's rule forbids reading the live box.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from app import llm_slots

ROOT = Path(__file__).resolve().parent.parent
ARCH = ROOT / "architecture"
INFRA = ARCH / "infrastructure.md"
INDEX = ARCH / "index.md"
VLLM = ARCH / "vllm.md"
CONF_D = ROOT / "agent-services" / "supervisor" / "conf.d"
LAUNCHER_DIR = ROOT / "agent-services" / "bin"

#: The box's cards, `nvidia-smi` index order. Cross-checked against the other
#: two in-tree copies of the roster by
#: `test_gpus_table_agrees_with_the_other_two_copies_of_the_roster`; the literal
#: here is what makes a deleted row fail rather than merely disagree.
CARD_COUNT = 5

_GPU_ROW = re.compile(r"^\|\s*(\d+)\s*\|(.*)$")
_NUM = re.compile(r"(\d+)\s*GB")


def _section(text: str, heading: str) -> str:
    """The body of `heading`, up to the next `##` heading."""
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == heading)
    except StopIteration:
        raise AssertionError(f"{heading} is not a heading any more") from None
    rest = lines[start + 1:]
    end = next((i for i, ln in enumerate(rest) if ln.startswith("## ")), len(rest))
    body = "\n".join(rest[:end])
    assert body.strip(), f"{heading} has no body"
    return body


def _gpus_rows_of(text: str) -> dict[int, list[str]]:
    """§GPUs' table as `{index: [hardware, vram, serves, ...]}`.

    The same shape `tests/test_gpu_placement_comments.py` reads, so a change to
    the table's spelling breaks both files rather than quietly dropping one.
    """
    rows: dict[int, list[str]] = {}
    for ln in _section(text, "## GPUs").splitlines():
        m = _GPU_ROW.match(ln)
        if not m:
            continue
        idx = int(m.group(1))
        cells = [c.strip() for c in m.group(2).split("|")]
        assert idx not in rows, f"§GPUs has two rows for GPU {idx}"
        rows[idx] = cells
    assert rows, "§GPUs has no data rows at all — the table stopped parsing"
    return rows


def _gpus_rows() -> dict[int, list[str]]:
    return _gpus_rows_of(INFRA.read_text(encoding="utf-8"))


def _clamp_cards() -> dict[int, str]:
    """`vllm.md` §2's power-clamp table: `{index: card family}`, bus id stripped.

    Scoped by that table's own header on purpose: §2 carries a *second*
    index-keyed table (`nvidia-smi` index to PCI bus to `/dev/nvidiaN`), and
    scanning the section whole returns `01:00.0` where the card name belongs —
    a parser that reads the wrong table is exactly as blind as a doc with the
    wrong row count. This is the table `2e08e6d3` brought to five rows while
    leaving §GPUs at three, so it is the copy worth being tied to.
    """
    lines = _section(VLLM.read_text(encoding="utf-8"),
                     "## 2. The card underneath it").splitlines()
    try:
        start = next(i for i, ln in enumerate(lines)
                     if ln.startswith("| GPU | Card | Clamp |"))
    except StopIteration:
        raise AssertionError("vllm §2's clamp-table header is gone — this check "
                             "needs re-pointing, not a relaxed assertion") from None
    cards: dict[int, str] = {}
    for ln in lines[start + 2:]:
        m = _GPU_ROW.match(ln)
        if not m:
            break
        cards[int(m.group(1))] = re.sub(
            r"\s*\([0-9a-f:.]+\)\s*", "", m.group(2).split("|")[0]).strip()
    assert len(cards) >= CARD_COUNT, f"the clamp table only knows {sorted(cards)}"
    return cards


def _index_rows() -> dict[int, str]:
    """`index.md` §GPU allocation, the glance map that cites §GPUs as its owner."""
    rows: dict[int, str] = {}
    for ln in _section(INDEX.read_text(encoding="utf-8"), "## GPU allocation").splitlines():
        m = _GPU_ROW.match(ln)
        if m:
            rows[int(m.group(1))] = m.group(2).strip()
    assert len(rows) >= CARD_COUNT, f"index.md's GPU table only has {sorted(rows)}"
    return rows


def _cuda_pins() -> dict[str, str]:
    """`{conf name: CUDA_VISIBLE_DEVICES}` for every conf.d program that sets it."""
    pins: dict[str, str] = {}
    for conf in sorted(CONF_D.glob("*.conf")):
        m = re.search(r'CUDA_VISIBLE_DEVICES="?([^",\s]+)"?', conf.read_text(encoding="utf-8"))
        if m:
            pins[conf.stem] = m.group(1)
    assert pins, "no conf.d program pins a GPU — the extraction is broken, not the box"
    return pins


def _confs_setting(key: str) -> set[str]:
    out = {p.stem for p in CONF_D.glob("*.conf")
           if key in p.read_text(encoding="utf-8")}
    assert out, f"no conf.d program sets {key} — the extraction is broken"
    return out


# --- clause 1: one row per card -------------------------------------------

def test_gpus_table_has_exactly_one_row_per_card():
    rows = _gpus_rows()
    assert sorted(rows) == list(range(CARD_COUNT)), (
        f"§GPUs covers {sorted(rows)} but the box has indices "
        f"0-{CARD_COUNT - 1}: a card with no row is indistinguishable from a "
        "spare card, which is the bug this item is about")
    for idx, cells in sorted(rows.items()):
        hardware, vram = cells[0], cells[1]
        assert re.search(r"RTX", hardware), f"GPU {idx} names no card model: {hardware!r}"
        assert _NUM.fullmatch(vram), f"GPU {idx} names no VRAM: {vram!r}"


def test_gpus_table_agrees_with_the_other_two_copies_of_the_roster():
    """Three docs carry this roster. The owner must not be the one behind."""
    rows, clamp, glance = _gpus_rows(), _clamp_cards(), _index_rows()
    assert sorted(rows) == sorted(clamp) == sorted(glance), (
        f"the three GPU tables disagree on which cards exist: §GPUs "
        f"{sorted(rows)}, vllm §2 {sorted(clamp)}, index {sorted(glance)}")
    for idx in sorted(rows):
        assert rows[idx][0].startswith(clamp[idx]), (
            f"GPU {idx}: §GPUs says {rows[idx][0]!r}, vllm §2's clamp table says "
            f"the card is {clamp[idx]!r}")
        # index.md folds model and VRAM into one cell: "RTX 3090 24 GB".
        glance_vram = _NUM.search(glance[idx])
        assert glance_vram, f"index.md names no VRAM for GPU {idx}: {glance[idx]!r}"
        infra_vram = _NUM.search(rows[idx][1])
        assert infra_vram, f"§GPUs names no VRAM for GPU {idx}: {rows[idx][1]!r}"
        assert infra_vram.group(0) == glance_vram.group(0), (
            f"GPU {idx}: §GPUs says {rows[idx][1]!r}, index.md says {glance[idx]!r}")


def test_the_row_harvester_can_report_a_missing_card():
    """The row-count assertion above must be able to fail, and it must fail the
    way the real rot looked: rows 3 and 4 simply not there."""
    text = INFRA.read_text(encoding="utf-8")
    body = _section(text, "## GPUs")
    trimmed_body = "\n".join(ln for ln in body.splitlines()
                             if not re.match(r"\|\s*[34]\s*\|", ln))
    idx = sorted(_gpus_rows_of(text.replace(body, trimmed_body, 1)))
    assert idx == [0, 1, 2], idx
    assert sorted(set(range(CARD_COUNT)) - set(idx)) == [3, 4]


# --- clause 2: GPU 3 is the voice TTS, GPU 0 keeps the fallback ------------

def test_gpu_3_row_names_the_omni_tts_the_voice_path_uses():
    conf = (CONF_D / "agent-tts-omni.conf").read_text(encoding="utf-8")
    assert "autostart=true" in conf, "the omni TTS conf is not autostart any more"
    launcher = (LAUNCHER_DIR / "start-qwen3-tts-omni.sh").read_text(encoding="utf-8")
    port = re.search(r'--port "\$\{TTS_OMNI_PORT:-(\d+)\}"', launcher)
    assert port, "the omni launcher no longer defaults its port in place"
    since = re.search(r"since (\d{4}-\d{2}-\d{2})", launcher)
    assert since, "the omni launcher no longer dates the move"
    row = " | ".join(_gpus_rows()[3])
    assert "agent-tts-omni" in row, row
    assert port.group(1) in row, f"the GPU 3 row does not name port {port.group(1)}: {row!r}"
    assert "voice" in row.lower(), f"the GPU 3 row does not say this is the voice TTS: {row!r}"
    assert since.group(1) in row, f"the GPU 3 row is undated: {row!r}"


def test_gpu_0_row_keeps_agent_tts_as_the_fallback_engine():
    """The GPU 0 row was never wrong — `agent-tts` is pinned there and still
    runs as the fallback. Adding GPU 3 must not delete it."""
    pins = _cuda_pins()
    assert pins["agent-tts"] == "0", f"agent-tts moved off GPU 0: {pins['agent-tts']!r}"
    row = " | ".join(_gpus_rows()[0])
    assert "agent-tts" in row, f"the GPU 0 row lost the fallback engine: {row!r}"
    assert "fallback" in row.lower(), (
        f"the GPU 0 row does not say the engine it names is the fallback: {row!r}")


# --- clause 3: GPU 4 says so in words -------------------------------------

def test_gpu_4_row_says_in_words_that_nothing_is_pinned_to_it():
    row = " | ".join(_gpus_rows()[4])
    assert re.search(r"\bnothing\b", row, re.I), (
        f"the GPU 4 row does not say in words that nothing runs here: {row!r}")
    assert re.search(r"pin", row, re.I), (
        f"the GPU 4 row does not say anything is *unpinned*: {row!r}")
    # Derived from the tree: which indices do the confs actually pin? A UUID pin
    # names a card, not an index, so it is attributed through the launcher that
    # carries it — and only agent-tts-omni pins by UUID on this box.
    pins = _cuda_pins()
    uuid_pinned = sorted(n for n, v in pins.items() if v.startswith("GPU-"))
    assert uuid_pinned == ["agent-tts-omni"], uuid_pinned
    launcher = (LAUNCHER_DIR / "start-qwen3-tts-omni.sh").read_text(encoding="utf-8")
    assert re.search(r"GPU 3\b", launcher), (
        "the omni launcher no longer names the card it UUID-pins, so the UUID "
        "pin can no longer be attributed and this check needs a new source")
    assert "4" not in [v for v in pins.values() if v.isdigit()], (
        f"a conf pins index 4 ({pins}), so the GPU 4 row's wording is now wrong")


# --- clause 4: the CUDA_DEVICE_ORDER prose --------------------------------

def test_gpus_prose_names_every_conf_that_sets_cuda_device_order():
    confs = _confs_setting("CUDA_DEVICE_ORDER")
    assert len(confs) >= 5, sorted(confs)
    prose = _section(INFRA.read_text(encoding="utf-8"), "## GPUs")
    sentence = [s for s in re.split(r"(?<=[.!])\s+", prose) if "programs set it" in s]
    assert len(sentence) == 1, f"the prose has {len(sentence)} 'programs set it' sentences"
    listed = set(re.findall(r"`([a-z0-9-]+)`", sentence[0]))
    assert listed == confs, (
        f"§GPUs names {sorted(listed)} as the confs that set CUDA_DEVICE_ORDER; "
        f"the tree has {sorted(confs)} — new: {sorted(confs - listed)}")
    assert re.search(r"\b(Five|five)\b", sentence[0]), (
        f"the sentence still counts itself wrong: {sentence[0]!r}")
    assert "Four programs set it" not in prose


# --- clause 5: the omni TTS in the program and port tables ----------------

def test_process_and_ports_tables_record_the_omni_tts():
    launcher = "bin/start-qwen3-tts-omni.sh"
    assert (ROOT / "agent-services" / launcher).is_file(), launcher + " is gone"
    launcher_src = (LAUNCHER_DIR / "start-qwen3-tts-omni.sh").read_text(encoding="utf-8")
    port_m = re.search(r'--port "\$\{TTS_OMNI_PORT:-(\d+)\}"', launcher_src)
    assert port_m, "the omni launcher no longer defaults its port in place"
    port = port_m.group(1)

    text = INFRA.read_text(encoding="utf-8")
    program_rows = {}
    for ln in _section(text, "## Process management").splitlines():
        m = re.match(r"^\|\s*`([a-z0-9:_-]+)`\s*\|(.*)$", ln)
        if m:
            program_rows[m.group(1)] = [c.strip() for c in m.group(2).split("|")]
    assert program_rows, "the program table stopped parsing"
    assert "agent-tts-omni" in program_rows, sorted(program_rows)
    cells = program_rows["agent-tts-omni"]
    assert cells[0] == port, f"the program table puts agent-tts-omni on {cells[0]!r}, not {port}"
    assert launcher in " ".join(cells), f"the program table omits the start script: {cells!r}"

    port_rows = {}
    for ln in _section(text, "## Ports").splitlines():
        m = re.match(r"^\|\s*(\d+)\s*\|(.*)$", ln)
        if m:
            port_rows[m.group(1)] = m.group(2).strip()
    assert port_rows, "the port table stopped parsing"
    assert port in port_rows, f"§Ports has no {port} row; a reader concludes voice is 8090"
    assert "omni" in port_rows[port].lower(), f"§Ports' {port} row names no engine: {port_rows[port]!r}"


# --- clause 5-8: conf.d owns the program and port tables (#2712) ------------

CONF_DIR = ROOT / "agent-services" / "supervisor" / "conf.d"

#: A supervisor program section. A `[group:…]` header is deliberately absent: a
#: group is a label on programs that each have their own conf, not a process, and
#: `lloyd-mc.conf` is only `[group:lloyd-mc]` with
#: `programs=lloyd-backend,lloyd-frontend,lloyd-mcp`.
_PROGRAM_HEADER = re.compile(r"^\[program:([a-z0-9-]+)\]", re.M)
_GROUP_HEADER = re.compile(r"^\[group:([a-z0-9-]+)\]", re.M)
_GROUP_MEMBERS = re.compile(r"^programs=(.+)$", re.M)
#: Any `…PORT="…"` a conf sets in `environment=`, so `PORT` and `STRUCTURED_PORT`
#: are harvested as served ports and `VLLM_PORT` is filtered out by name below.
_ENV_PORT = re.compile(r'\b([A-Z_]*PORT)="(\d+)"')
#: Ports a conf sets for a process it *starts* rather than serves. `VLLM_PORT` is
#: an env vLLM reads for its own socket — 5300 under djev, 5310 under the decider —
#: and neither appears in either table: `grep -n "5300\|5310"
#: architecture/infrastructure.md` lands on one Review-log line and no row, because
#: nothing connects to them. This set is a name list, so the node below derives the
#: exclusion's safety instead of trusting it: an excluded port must not be one the
#: program table itself credits to a program.
_INTERNAL_PORT_KEYS = frozenset({"VLLM_PORT"})
#: A program table row, keyed by backticked name. The key may carry the group
#: prefix `supervisorctl status` prints (`lloyd-mc:lloyd-backend`), which is
#: stripped — see `_program_rows`.
_PROGRAM_ROW = re.compile(r"^\|\s*`([a-z0-9:_-]+)`\s*\|(.*)$")
#: A §Ports row's first cell, which may spell several ports: `| 7880 / 7881 |`.
_PORT_ROW = re.compile(r"^\|\s*([^|]+?)\s*\|(.*)$")
#: The shape this file used before #2712 for its one §Ports lookup. Kept as a
#: named thing so the blind spot can be *asserted* rather than asserted about.
_ONE_PORT_ROW = re.compile(r"^\|\s*(\d+)\s*\|(.*)$")
_PORT_NUM = re.compile(r"\b\d{4,5}\b")


def _conf_programs() -> dict[str, str]:
    """{program name: conf filename} for every `[program:…]` header in conf.d/.

    Read from section headers, not filenames, because the two are not the same
    set today: 15 `*.conf`, 14 programs, and the file that is not a program is
    `lloyd-mc.conf`.
    """
    found: dict[str, str] = {}
    for path in sorted(CONF_DIR.glob("*.conf")):
        for name in _PROGRAM_HEADER.findall(path.read_text(encoding="utf-8")):
            found[name] = path.name
    assert found, f"no [program:…] header anywhere under {CONF_DIR}"
    return found


def _conf_groups() -> dict[str, list[str]]:
    """{group name: [member programs]} from every `[group:…]` conf."""
    out: dict[str, list[str]] = {}
    for path in sorted(CONF_DIR.glob("*.conf")):
        text = path.read_text(encoding="utf-8")
        members = _GROUP_MEMBERS.search(text)
        for group in _GROUP_HEADER.findall(text):
            out[group] = (sorted(m.strip() for m in members.group(1).split(","))
                          if members else [])
    return out


def _program_rows(text: str) -> dict[str, list[str]]:
    """§Process management's rows keyed by bare program name, cells as a list.

    A row for a grouped program is spelled the way `supervisorctl status` prints
    it — ``lloyd-mc:lloyd-backend`` — so the prefix is stripped rather than
    hardcoded, and `_conf_programs()`'s exclusion of the group conf costs nothing.
    """
    rows: dict[str, list[str]] = {}
    for ln in _section(text, "## Process management").splitlines():
        m = _PROGRAM_ROW.match(ln)
        if not m:
            continue
        name = m.group(1).rsplit(":", 1)[-1]
        assert name not in rows, (
            f"§Process management lists {name} twice, once at least as "
            f"{m.group(1)!r}; the key would silently collapse and a missing row "
            "would read as present")
        rows[name] = [c.strip() for c in m.group(2).split("|")]
    assert rows, "§Process management's table stopped parsing"
    return rows


def _programs_without_rows(text: str) -> list[str]:
    """Program names conf.d declares that `text`'s program table has no row for.

    Split out and taking any text so the mutation node below can hand it a table
    with a row deleted: a green here has to be attributable to rows existing, not
    to a regex that matched nothing.
    """
    return sorted(set(_conf_programs()) - set(_program_rows(text)))


def _conf_ports() -> dict[str, tuple[str, str]]:
    """{port: (conf filename, environment key)} for every `…PORT="…"` in conf.d.

    `PORT=` is the port a launcher binds: `PORT="${PORT:-8010}"` and
    `STRUCTURED_PORT="${STRUCTURED_PORT:-8011}"` in `start-djev.sh`,
    `PORT="${PORT:-8012}"` in `start-decider.sh`, and `PORT="8091"` handed to
    `start-secondary.sh`. Today: 8010, 8011, 8012, 8091 served, plus 5300 and 5310
    under `VLLM_PORT=`.
    """
    seen: dict[str, list[tuple[str, str]]] = {}
    for path in sorted(CONF_DIR.glob("*.conf")):
        for key, port in _ENV_PORT.findall(path.read_text(encoding="utf-8")):
            seen.setdefault(port, []).append((path.name, key))
    for port, who in seen.items():
        assert len(who) == 1, f"port {port} is declared by more than one conf: {who}"
    return {port: who[0] for port, who in seen.items()}


def _serving_and_internal_ports() -> tuple[dict[str, str], dict[str, str]]:
    """(serving, internal) port maps, split by environment key name.

    Both are port -> "<conf> <KEY>" so a failure names where the port came from,
    and the filter on the *key* lives in these two lines and nowhere else.
    """
    declared = _conf_ports()
    serving = {p: f"{c} {k}" for p, (c, k) in declared.items()
               if k not in _INTERNAL_PORT_KEYS}
    internal = {p: f"{c} {k}" for p, (c, k) in declared.items()
                if k in _INTERNAL_PORT_KEYS}
    assert serving, f"no conf under {CONF_DIR} declares a served PORT="
    return serving, internal


def _documented_ports(text: str) -> dict[str, str]:
    """§Ports rows as {port: what the row says}, counting ports a cell shares.

    The first cell spells two ports in `| 7880 / 7881 |`, `| 8010 / 8011 |` and
    `| 18080 / 18500 |`, and the one-port shape `_ONE_PORT_ROW` matches skips all
    three cells whole. On today's doc the cell count is 17 ports and the one-port
    shape sees 11 of them; the six it misses include djev's `8010` and `8011`, so a
    check keyed on that shape would report the engine
    `scripts/service_health_check.py` probes as undocumented. The assertion below
    is what keeps that from coming back as a "simplification".
    """
    out: dict[str, str] = {}
    one_port_only: set[str] = set()
    for ln in _section(text, "## Ports").splitlines():
        m = _ONE_PORT_ROW.match(ln)
        if m:
            one_port_only.add(m.group(1))
        m = _PORT_ROW.match(ln)
        if not m:
            continue
        for port in _PORT_NUM.findall(m.group(1)):
            assert port not in out, f"§Ports has two rows for {port}"
            out[port] = m.group(2).strip()
    assert out, "§Ports' table stopped parsing"
    assert set(out) - one_port_only, (
        f"every §Ports port is one the single-port shape sees too, so shared-cell "
        f"rows like '| 8010 / 8011 |' are being skipped, not counted ({len(out)} "
        f"ports, all also matched by {_ONE_PORT_ROW.pattern})")
    return out


def _ports_the_program_table_credits(text: str) -> dict[str, set[str]]:
    """{program: ports} from the Ports column of §Process management's rows."""
    ports = {name: set(_PORT_NUM.findall(cells[0]))
             for name, cells in _program_rows(text).items() if cells}
    assert any(ports.values()), "no program row names a port at all"
    return ports


def test_every_program_conf_d_declares_has_a_row_in_the_process_management_table():
    """§Process management is written against the confs, not the other way round.

    The expected set is the 14 `[program:…]` headers in `conf.d/`, so adding a
    15th conf and forgetting the doc is the failure this catches — the exact
    shape of `6ed8753c`, which added `agent-djev` and `agent-decider` to the
    table 100 insertions after the confs existed, with no test in the commit.
    `lloyd-mc.conf` contributes nothing because it declares no program: it is
    `[group:lloyd-mc]`, and all three of its members have confs of their own,
    which the loop below re-checks rather than assumes. Row count and program
    count are then tied exactly, so the green is "14 names, 14 rows", not "some
    rows parsed".
    """
    programs, groups = _conf_programs(), _conf_groups()
    assert len(programs) == 14, (
        f"conf.d declares {len(programs)} programs, not the 14 this node's "
        "docstring and the module docstring name. Count them again with "
        "`grep -l '^\\[program:' agent-services/supervisor/conf.d/*.conf | wc -l`, "
        "write the row the new program is missing, then update the number in both "
        "docstrings")
    for group, members in groups.items():
        assert members, f"{group}.conf has a [group:] header with no programs= line"
        assert group not in programs, (
            f"{group} is both a group and a program name; the group conf's "
            "exclusion would then hide a real program")
        for member in members:
            assert member in programs, (
                f"[group:{group}] names {member}, which has no [program:] conf of "
                "its own — ignoring group files would drop it from the roster")
    rows = _program_rows(INFRA.read_text(encoding="utf-8"))
    assert len(rows) == len(programs), (
        f"§Process management has {len(rows)} program rows for {len(programs)} "
        f"confs; a name appearing twice or a row keyed differently hides a gap")
    missing = _programs_without_rows(INFRA.read_text(encoding="utf-8"))
    assert not missing, (
        f"programs with a conf and no §Process management row: {missing} — a reader "
        "of this doc cannot find, restart or disarm them")


def test_the_program_row_harvester_can_report_a_missing_program():
    """Negative control: delete one row from the real table, the gap must be named.

    Same discipline as `test_the_row_harvester_can_report_a_missing_card` for the
    GPU table. The deleted row is `agent-djev`, the engine `djev.enabled: true`
    switches on and `scripts/service_health_check.py:116` probes — the program
    whose row was in fact missing from this table until `6ed8753c` added it by
    hand, so the mutation is not hypothetical, it is last week.
    """
    text = INFRA.read_text(encoding="utf-8")
    gone = "agent-djev"
    lines = text.splitlines(keepends=True)
    dropped = [i for i, ln in enumerate(lines)
               if (m := _PROGRAM_ROW.match(ln)) and m.group(1).rsplit(":", 1)[-1] == gone]
    assert len(dropped) == 1, f"expected one §Process management row naming {gone}, got {dropped}"
    del lines[dropped[0]]
    mutated = "".join(lines)
    assert len(_program_rows(mutated)) == len(_program_rows(text)) - 1, (
        "deleting the row did not reach the table")
    reported = _programs_without_rows(mutated)
    assert gone in reported, (
        f"{gone} has a conf and no row after the mutation, and the harvester said "
        f"{reported!r}")
    assert gone not in _programs_without_rows(text), (
        f"{gone} is already missing a row on the real table, so this node is "
        "reporting a live gap rather than proving it can see one")


def test_every_serving_port_a_conf_declares_has_a_row_in_the_ports_table():
    """§Ports must carry the ports conf.d binds, and never gain a port from them.

    One direction, deliberately: `8765` is the Thunderbird bridge and
    `18080 / 18500` the automod canary, rows for things that are not supervisor
    programs at all, so "every §Ports row has a conf" is unsatisfiable and asking
    it would train the next reader to delete the check.

    The harvest is filtered to the serving ports — 8010, 8011, 8012, 8091 — and
    `VLLM_PORT=` is left out by key name. That exclusion is derived, not trusted:
    5300 and 5310 are checked *not* to be ports the program table credits to a
    program, which is what makes them internal; the moment a conf names a real
    served port `VLLM_PORT=`, this node goes red instead of silently dropping it.
    """
    serving, internal = _serving_and_internal_ports()
    text = INFRA.read_text(encoding="utf-8")
    documented = _documented_ports(text)
    missing = sorted(set(serving) - set(documented))
    assert not missing, (
        f"serving ports with no §Ports row: {missing} — "
        + "; ".join(f"{p} from {serving[p]}" for p in missing)
        + "; a reader choosing an engine by port cannot find them")
    credited = {p for ports in _ports_the_program_table_credits(text).values()
                for p in ports}
    overlap = sorted(set(internal) & credited)
    assert not overlap, (
        f"{overlap} is set as VLLM_PORT= and is also a port §Process management "
        "credits to a program, so it is served after all and belongs in §Ports")


def test_the_ports_table_covers_every_port_the_program_table_lists():
    """Every port §Process management names has a §Ports row, 8501 included.

    The two tables are read by different questions — "what restarts it" and "what
    answers on this port" — and today they disagreed once: `lloyd-agent-worker`'s
    8501, the loopback `WakeMissCapture` aiohttp rig in
    `agent-services/livekit_worker.py` that the backend proxies `/api/voice/ww_miss`
    to, sat in the program table (`architecture/infrastructure.md:281`) and in no
    §Ports row, so `grep -n "8501" architecture/infrastructure.md` printed one
    line and the port table alone could not find the thing. The row is checked
    through `_documented_ports`, the same cell-counting parse the conf-driven
    check uses, and then mutated out: a port documented in a single row has to be
    the row that fails if it goes.
    """
    text = INFRA.read_text(encoding="utf-8")
    credited = _ports_the_program_table_credits(text)
    documented = _documented_ports(text)
    missing = sorted({p for ports in credited.values() for p in ports} - set(documented))
    assert not missing, (
        f"§Process management names ports {missing} that §Ports has no row for — "
        "the two tables answer different questions and have to agree anyway")
    assert credited.get("lloyd-agent-worker") == {"8501"}, credited.get("lloyd-agent-worker")
    assert "8501" in documented, "the 8501 row this item added is gone from §Ports"
    assert "lloyd-agent-worker" in documented["8501"], (
        f"§Ports' 8501 row does not name the program that owns it: {documented['8501']!r}")
    assert not [ln for ln in text.splitlines() if ln.startswith("| 8501 ")][1:], (
        "more than one §Ports row starts at 8501")

    lines = text.splitlines(keepends=True)
    dropped = [i for i, ln in enumerate(lines) if ln.startswith("| 8501 ")]
    assert len(dropped) == 1, f"expected exactly one 8501 row, got {dropped}"
    del lines[dropped[0]]
    mutated = _documented_ports("".join(lines))
    assert "8501" not in mutated, "the deleted 8501 row is still being documented"
    assert len(mutated) == len(documented) - 1, (
        f"deleting one row cost {len(documented) - len(mutated)} ports: the parse "
        "is reading cells it should not")


# --- #3137: vllm §2.4, the index map `_clamp_cards` scoped itself out of -------

#: vllm §2 carries TWO index-keyed tables and `_clamp_cards` reads only the clamp
#: one — its own docstring names the other as what it deliberately does not return.
#: The exclusion was by construction, so nothing replaced it, and that is how
#: `2e08e6d3` ("moar gpus", 2026-10-07) could take §2.3 to five rows while leaving
#: §2.4 at three: still index 2's bus id as `61:00.0`, which is index 3's card today,
#: and still index 2's tenant "the llama.cpp secondary", off since 2026-09-20
#: (`secondary_enabled: false`, read by `app/llm_slots.py`) and djev's card since.
#: `55f6eda1` corrected those rows by hand on 2026-10-21 — the symptom fixed and the
#: mechanism untouched, which is what #3137 is about. These nodes harvest the table
#: by this header rather than by §2.4's heading, because the heading is prose that
#: may be reworded and the header is what every row is keyed on.
_INDEX_MAP_HEADER = re.compile(r"^\|\s*`nvidia-smi`\s*\|\s*PCI\s*\|")
#: §2.4's PCI cell, and §2.3's same fact in parentheses after the card family.
_BUS_ID = re.compile(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]")
_CLAMP_BUS_ID = re.compile(r"\(([0-9a-f]{2}:[0-9a-f]{2}\.[0-9])\)")
#: A backticked span in a Serves cell. It spans punctuation a program name does not
#: contain (`agent-djev.conf`, `GPU=2`, `/dev/nvidia3`), because the cell writes the
#: conf's file name, not the bare program name, and a class restricted to name
#: characters would match nothing at all and read as a row with no tenant. Only a
#: span conf.d declares counts as a tenant, and the direction is one-way — see
#: `_named_programs`.
_PROGRAM_TOKEN = re.compile(r"`([^`]+)`")
_CARD_PIN_IN_ENV = re.compile(r'(?:\bGPU|CUDA_VISIBLE_DEVICES)="([^"]+)"')
_SERVED_PORT = re.compile(r'(?<![A-Z_])PORT="(\d+)"')
_URL_PORT = re.compile(r":(\d+)")
_LAUNCHER_SCRIPT = re.compile(r"^command=.*?agent-services/bin/([\w.+-]+)", re.M)
_LAUNCHER_EXPORT = re.compile(r'^export CUDA_VISIBLE_DEVICES="?(\d+)"?', re.M)
_UUID_PIN = re.compile(r"GPU-[0-9a-f-]{36}")
#: A launcher line opening `# GPU <n>`: the only place in the tree that says which
#: card the box's one UUID pin is on. Anchored deliberately — the same launcher also
#: mentions "the vendored server on GPU 0" mid-sentence, and an unanchored
#: `GPU (\d+)` search reads THAT first and files the omni TTS on card 0.
_GPU_COMMENT_CARD = re.compile(r"^#\s*GPU (\d+)\b", re.M)
_NOTHING_PINNED = re.compile(r"\bnothing\b[^.]*\bpin", re.I)
_INDEX_MAP_COLS = ("pci", "node", "card", "serves")

#: §2.4 verbatim as it shipped until the correction, from
#: `git show 55f6eda1^:architecture/vllm.md`. A table with an index 2 nobody can
#: invent (`61:00.0`, a card added after the three-row version was written) and a
#: tenant no switch still allows, so the nodes that splice it in are grading the
#: checks against the table that actually rotted, not a hypothetical one.
_PREFIX_INDEX_MAP = [
    "| `nvidia-smi` | PCI | `/dev/nvidiaN` | Card | Serves |",
    "|---|---|---|---|---|",
    "| 0 | 01:00.0 | `/dev/nvidia2` | RTX 3090 | Qwen3-TTS, the voice worker, qmd |",
    "| 1 | 41:00.0 | `/dev/nvidia1` | RTX PRO 6000 | **this engine** |",
    "| 2 | 61:00.0 | `/dev/nvidia0` | RTX 3090 | the llama.cpp secondary |",
]


def _index_map_table_of(text: str) -> list[str]:
    """vllm §2's `nvidia-smi` index / PCI bus / `/dev/nvidiaN` / tenant map, header
    line first, then the separator, then each data row.

    Scoped inside §2 by its header, the same way `_clamp_cards` is scoped to
    `| GPU | Card | Clamp |`: the section has two index-keyed tables and a harvester
    that scans it whole reads one and believes it read the other.
    """
    lines = _section(text, "## 2. The card underneath it").splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if _INDEX_MAP_HEADER.match(ln))
    except StopIteration:
        raise AssertionError(
            "vllm §2 no longer has a `| `nvidia-smi` | PCI |` table — the index map "
            "was renamed or moved, so this harvester needs re-pointing, not a "
            "relaxed assertion") from None
    table = list(lines[start:start + 2])
    for ln in lines[start + 2:]:
        if not _GPU_ROW.match(ln):
            break
        table.append(ln)
    assert len(table) >= 3, f"the index map's header has no data row under it: {table!r}"
    return table


def _index_map_of(table: list[str]) -> dict[int, dict[str, str]]:
    """The table as `{index: {pci, node, card, serves}}`, columns named.

    Deliberately does NOT assert the row count: `_index_map_gaps` has to be able to
    see which indices are absent, and a harvester that raised on a short table could
    only ever report "the table is short", never "index 3 and index 4 are". Naming
    the missing card is the whole difference between this and a green regex.
    """
    rows: dict[int, dict[str, str]] = {}
    for ln in table[2:]:
        m = _GPU_ROW.match(ln)
        assert m, f"an index map row is not keyed by a card index: {ln!r}"
        cells = [c.strip() for c in m.group(2).split("|")]
        while cells and cells[-1] == "":
            cells.pop()
        assert len(cells) == len(_INDEX_MAP_COLS), (
            f"index map row for {m.group(1)} has {len(cells)} cells after the index, "
            f"expected the {len(_INDEX_MAP_COLS)} of them ({_INDEX_MAP_COLS}): "
            f"{ln!r}. Adding or dropping a column means this harvester and the row "
            "checks below move together")
        idx = int(m.group(1))
        assert idx not in rows, f"the index map has two rows for index {idx}"
        rows[idx] = dict(zip(_INDEX_MAP_COLS, cells))
    assert rows, "the index map has no data rows at all — the table stopped parsing"
    return rows


def _index_map_rows_of(text: str) -> dict[int, dict[str, str]]:
    return _index_map_of(_index_map_table_of(text))


def _index_map_rows() -> dict[int, dict[str, str]]:
    return _index_map_rows_of(VLLM.read_text(encoding="utf-8"))


def _index_map_gaps(rows: dict[int, dict[str, str]]) -> list[str]:
    """What the index map is missing, one line per defect, the index named in each."""
    out = [f"index {i}: no row, and the box has indices 0-{CARD_COUNT - 1}"
           for i in sorted(set(range(CARD_COUNT)) - set(rows))]
    out += [f"index {i}: a row, but the box has no such card"
            for i in sorted(set(rows) - set(range(CARD_COUNT)))]
    for i in sorted(rows):
        for col in _INDEX_MAP_COLS:
            if not rows[i][col]:
                out.append(f"index {i}: the {col} cell is empty")
    return out


def _clamp_bus_ids() -> dict[int, str]:
    """§2.3's clamp table as `{index: PCI bus id}` — the column `_clamp_cards` throws away.

    `_clamp_cards` strips the `(41:00.0)` parenthetical on purpose, because its own
    comparison is card family against card family; nobody harvested what it dropped,
    so §2.4's PCI column had no in-tree counterparty at all. Reading it here is the
    only way to check that column without the live box: `git grep '/dev/nvidia[0-9]'`
    lands on §2.4's own rows and nowhere else in the tree, which is also why the
    `/dev/nvidiaN` column below is checked for shape only. Two copies of one fact in
    two tables of one page is the disagreement that rotted, and doc-against-doc is
    what the module docstring's rule allows.
    """
    lines = _section(VLLM.read_text(encoding="utf-8"),
                     "## 2. The card underneath it").splitlines()
    try:
        start = next(i for i, ln in enumerate(lines)
                     if ln.startswith("| GPU | Card | Clamp |"))
    except StopIteration:
        raise AssertionError("vllm §2's clamp-table header is gone — the bus-id check "
                             "needs re-pointing, not a relaxed assertion") from None
    buses: dict[int, str] = {}
    for ln in lines[start + 2:]:
        m = _GPU_ROW.match(ln)
        if not m:
            break
        found = _CLAMP_BUS_ID.search(m.group(2).split("|")[0])
        assert found, (f"§2.3's clamp row for GPU {m.group(1)} carries no PCI bus id "
                       f"in its Card cell, so it can no longer be §2.4's counterparty: "
                       f"{ln!r}")
        buses[int(m.group(1))] = found.group(1)
    assert len(buses) >= CARD_COUNT, f"§2.3's clamp table only bus-tags {sorted(buses)}"
    return buses


def _bus_mismatches(rows: dict[int, dict[str, str]]) -> list[str]:
    """§2.4's PCI column against §2.3's same-bus-id column, per index, disagreements named."""
    buses = _clamp_bus_ids()
    out = []
    for i in sorted(set(rows) & set(buses)):
        claimed = rows[i]["pci"]
        if not _BUS_ID.fullmatch(claimed):
            out.append(f"index {i}: the PCI cell {claimed!r} is not a PCI bus id")
        elif claimed != buses[i]:
            out.append(f"index {i}: §2.4 puts it on {claimed} but §2.3's clamp table "
                       f"bus-tags that card {buses[i]}")
    return out


def _conf_text(program: str) -> str:
    path = CONF_D / f"{program}.conf"
    assert path.is_file(), f"{program} is named as a tenant but has no conf at {path}"
    return path.read_text(encoding="utf-8")


def _conf_env_pin(program: str) -> str | None:
    """The raw card pin on the program's own `environment=` line: an index, a UUID, or None.

    Scoped to that line, not the file. `lloyd-agent-worker.conf` quotes `GPU="2"` in
    the comment explaining why the worker must stay OFF card 2 — the trap
    `tests/test_gpu_placement_comments.py` proves — and a whole-file scan files it as
    a card 2 tenant, which every check below would then inherit.
    """
    for ln in _conf_text(program).splitlines():
        if ln.strip().startswith("environment="):
            m = _CARD_PIN_IN_ENV.search(ln)
            return m.group(1) if m else None
    return None


def _launcher_of(program: str) -> Path | None:
    """The script under agent-services/bin that the program's `command=` runs, or None.

    This leg is load-bearing, not decorative: `agent-llm-primary.conf` — the tenant of
    the box's only 96 GB card — sets no GPU pin in its conf at all. Its pin is
    `export CUDA_VISIBLE_DEVICES=1` inside `start-qwen38-flash-next.sh`, which its
    `command=` runs. Reusing `_cuda_pins`, which reads conf `CUDA_VISIBLE_DEVICES=`
    only, therefore reports card 1 as unowned and fails a correct doc.
    """
    m = _LAUNCHER_SCRIPT.search(_conf_text(program))
    if not m:
        return None
    path = LAUNCHER_DIR / m.group(1)
    assert path.is_file(), f"{program}.conf runs {path}, which is not in the tree"
    return path


def _card_pin(program: str) -> str | None:
    """Which `nvidia-smi` index a program's pin resolves to, or None if it pins no card.

    Three legs, in the order the tree states them: the conf's `environment=`
    (`CUDA_VISIBLE_DEVICES=`, or supervisord's local `GPU=`); the launcher's own
    `export CUDA_VISIBLE_DEVICES=<index>` when the conf pins nothing; and a UUID,
    which is only an index because the launcher carrying it says so on a line of its
    own (`# GPU 3 …` in `start-qwen3-tts-omni.sh`). Anything unreadable is an
    assertion rather than a skip — a pin this cannot resolve is precisely how a
    tenant disappears from a card map while the map still looks complete.
    """
    raw = _conf_env_pin(program)
    launcher = _launcher_of(program)
    if raw is None:
        if launcher is None:
            return None
        m = _LAUNCHER_EXPORT.search(launcher.read_text(encoding="utf-8"))
        return m.group(1) if m else None
    if raw.isdigit():
        return raw
    if _UUID_PIN.fullmatch(raw):
        assert launcher, f"{program} pins the UUID {raw!r} but has no launcher to attribute it"
        src = launcher.read_text(encoding="utf-8")
        assert _UUID_PIN.search(src), (
            f"{program}.conf pins {raw!r} but {launcher.name} no longer carries that "
            "UUID, so nothing in the tree says which card it is")
        cards = _GPU_COMMENT_CARD.findall(src)
        assert len(cards) == 1, (
            f"{launcher.name} no longer attributes its UUID pin on exactly one "
            f"`# GPU <n>` line (found {cards}), so this check cannot say which card "
            f"{program} runs on")
        return cards[0]
    raise AssertionError(
        f"{program} pins {raw!r}, which is neither a card index nor a GPU- UUID: "
        "teach `_card_pin` the new spelling rather than dropping the tenant")


def _card_ownership() -> dict[str, set[str]]:
    """`{card index: programs whose pin resolves there}`, over every conf.d program.

    Derived rather than listed so a card that gains a tenant stops being allowed to
    say nothing pins it, and a card nobody pins any more has to start saying so.
    """
    owned: dict[str, set[str]] = {}
    for program in sorted(_conf_programs()):
        idx = _card_pin(program)
        if idx is None:
            continue
        assert idx.isdigit(), f"{program} resolved to {idx!r}, not a card index"
        owned.setdefault(idx, set()).add(program)
    assert owned, "no conf.d program pins a card — the extraction is broken, not the box"
    return owned


def _named_programs(serves: str, declared: set[str]) -> set[str]:
    """The conf.d programs a Serves cell names, by their own backticked tokens.

    One-directional on purpose: card 2 has three programs pinned to it and its row
    names one, so a cell may name a subset and must never name a name the tree does
    not have. A token that is spelled like a program (`agent-…`, `lloyd-…`) and is
    not declared therefore raises — that is how a row goes on advertising a program
    that has been deleted.
    """
    out: set[str] = set()
    for token in _PROGRAM_TOKEN.findall(serves):
        stem = token[:-len(".conf")] if token.endswith(".conf") else token
        if stem in declared:
            out.add(stem)
        elif stem.startswith(("agent-", "lloyd-")):
            raise AssertionError(
                f"the index map names `{token}`, which conf.d does not declare; a "
                "row has to name a program that exists and pins the card it is on")
    return out


def _index_map_problems(rows: dict[int, dict[str, str]]) -> list[str]:
    """Every way §2.4's rows can disagree with the tree, as strings, index named.

    Takes rows and not a path so `test_the_index_map_checks_go_red_on_the_shipped_
    pre_fix_table` can hand it the table as it stood before the correction.
    """
    declared = set(_conf_programs())
    owned = _card_ownership()
    out = _index_map_gaps(rows) + _bus_mismatches(rows)
    for idx, row in sorted(rows.items()):
        node = re.fullmatch(r"/dev/nvidia(\d)", row["node"].strip("`"))
        if not node:
            out.append(f"index {idx}: the node cell {row['node']!r} is not a "
                       "`/dev/nvidiaN` device node")
        elif int(node.group(1)) >= CARD_COUNT:
            out.append(f"index {idx}: device minor {node.group(1)} is past the box's "
                       f"five cards (0-{CARD_COUNT - 1})")
        if not re.search("RTX", row["card"]):
            out.append(f"index {idx}: the Card cell {row['card']!r} names no card model")
        named = _named_programs(row["serves"], declared)
        owners = owned.get(str(idx), set())
        if owners and not named:
            out.append(
                f"index {idx}: {sorted(owners)} pin this card and the Serves cell "
                f"({row['serves']!r}) names no program a check can resolve — prose "
                "like 'the omni TTS (pinned by GPU UUID)' or 'Qwen3-TTS, the voice "
                "worker, qmd' is unreadable by anything but a human, and this table "
                "rotted on exactly that")
        for program in sorted(named):
            pin = _card_pin(program)
            if pin != str(idx):
                out.append(
                    f"index {idx}: the row names {program}, whose pin resolves to "
                    f"card {pin} (conf {str(_conf_env_pin(program))!r}, launcher "
                    f"{_launcher_of(program).name if _launcher_of(program) else None})")
    minors = [re.fullmatch(r"/dev/nvidia(\d)", rows[i]["node"].strip("`"))
              for i in sorted(rows)]
    if all(minors):
        numbers = [m.group(1) for m in minors]
        if len(set(numbers)) != len(numbers):
            out.append(
                f"the node column gives two cards the same device minor {numbers}: "
                "that column is a live reading nothing else in the tree holds, so a "
                "duplicated minor is the only way it can be called out from here")
    out += _ownership_wording_problems(rows, owned)
    out += _disarmed_tenant_hits(rows, _disarmed_slots())
    return out


def _ownership_wording_problems(rows: dict[int, dict[str, str]],
                                owned: dict[str, set[str]]) -> list[str]:
    """Rows whose Serves cell disagrees about whether anything pins its card.

    `owned` is a parameter rather than a call to `_card_ownership()` so the node
    beside it can hand the check a box on which card 4 has acquired a tenant: today
    exactly one card is unowned and a check that only ever asks about today's answer
    cannot see the day that changes.
    """
    out = []
    for idx, row in sorted(rows.items()):
        unowned = str(idx) not in owned
        says_so = bool(_NOTHING_PINNED.search(row["serves"]))
        if unowned and not says_so:
            out.append(
                f"index {idx}: no conf.d program pins this card and the row does not "
                f"say so ({row['serves']!r}) — an unowned card left blank reads as a "
                "card the doc forgot, which is how index 4 vanished from the "
                "three-row version of this table")
        elif says_so and not unowned:
            out.append(
                f"index {idx}: {sorted(owned[str(idx)])} pin this card, so the row is "
                f"wrong to say nothing pins it ({row['serves']!r})")
    return out


def _model_alias(program: str, cfg: dict) -> list[str]:
    """What the tree calls this program, which is what a doc row would call it.

    The program name, its name without the `agent-` prefix, and the `models:` alias
    whose `base_url` is the port the program's own conf serves. That last leg is why
    "the llama.cpp secondary" is catchable at all: `agent-llm-secondary.conf` sets
    `PORT="8091"` and `models.secondary.base_url` is `http://127.0.0.1:8091`, so the
    word the doc used for the tenant is derivable rather than hardcoded.
    """
    names = [program, program.removeprefix("agent-")]
    port = None
    for ln in _conf_text(program).splitlines():
        if ln.strip().startswith("environment="):
            m = _SERVED_PORT.search(ln)
            port = m.group(1) if m else None
    if port:
        for alias, spec in (cfg.get("models") or {}).items():
            url = str((spec or {}).get("base_url") or "")
            found = _URL_PORT.search(url)
            if found and found.group(1) == port:
                names.append(alias)
    return names


def _disarmed_slots() -> dict[str, list[str]]:
    """`{program: the names a row might use for it}` for every optional LLM slot the
    tree has switched off.

    Armedness comes from `app.llm_slots` — the module whose entire job is "is this
    program supposed to be running" — against config.yaml read off disk, and never
    from `autostart=`. `agent-djev.conf` ships `autostart=false` and is armed by
    `djev.enabled: true`, so a reader who took armedness from the conf would call
    djev disarmed and fail a correct doc. Nor does it come from pin existence:
    `agent-llm-secondary.conf` still pins `GPU="2"`, which is exactly why the wrong
    row survived — a card's tenant can be pinned and switched off at the same time,
    and only `secondary_enabled` says so.
    """
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for flag, program, enabled in llm_slots.slots(cfg):
        assert flag == llm_slots.slot_flag(program), (program, flag)
        if not enabled:
            out[program] = _model_alias(program, cfg)
    return out


def _disarmed_tenant_hits(rows: dict[int, dict[str, str]],
                          disarmed: dict[str, list[str]]) -> list[str]:
    """§2.4 rows crediting a card to a slot the tree has switched off."""
    out = []
    for idx, row in sorted(rows.items()):
        for program, names in sorted(disarmed.items()):
            hit = sorted({n for n in names
                          if re.search(rf"\b{re.escape(n)}\b", row["serves"], re.I)})
            if hit:
                out.append(
                    f"index {idx}: the Serves cell ({row['serves']!r}) names this card's "
                    f"tenant as {hit}, but {program} is off — "
                    f"`{llm_slots.slot_flag(program)}` is false in config.yaml, so "
                    "nothing runs here whatever its conf still pins")
    return out


# --- clause 1: one row per card, and the harvester names an absent one ---------

def test_the_index_map_has_exactly_one_row_per_card():
    """§2.4 covers indices 0-4, and every column of every row is filled in.

    `_clamp_cards` asserts `len(cards) >= CARD_COUNT` for §2.3; this is that
    assertion for the table it deliberately does not read.
    """
    rows = _index_map_rows()
    assert not _index_map_gaps(rows), "; ".join(_index_map_gaps(rows))
    assert sorted(rows) == list(range(CARD_COUNT)), sorted(rows)


def test_the_index_map_harvester_names_a_missing_card_rather_than_returning_short():
    """Rows 3 and 4 deleted from a copy of the table: the gaps say "index 3" and
    "index 4", which is the difference between a harvester and a green regex.

    This is `test_the_row_harvester_can_report_a_missing_card` for §2.4, and the
    deleted rows are the real rot — `2e08e6d3` added two cards and this table never
    gained their rows.
    """
    text = VLLM.read_text(encoding="utf-8")
    table = _index_map_table_of(text)
    trimmed = [ln for ln in table if not re.match(r"\|\s*[34]\s*\|", ln)]
    assert len(trimmed) == len(table) - 2, f"{len(table)} → {len(trimmed)}"
    rows = _index_map_of(trimmed)
    assert sorted(rows) == [0, 1, 2], sorted(rows)
    assert _index_map_gaps(rows) == [
        f"index 3: no row, and the box has indices 0-{CARD_COUNT - 1}",
        f"index 4: no row, and the box has indices 0-{CARD_COUNT - 1}",
    ]


# --- clause 2: the PCI column, held by §2.3's copy of the same fact -----------

def test_the_index_map_bus_ids_agree_with_the_clamp_table():
    """§2.4's PCI cell equals §2.3's bus tag for the same index. Doc against doc.

    Not a hardware check: §2.3's bus-id column is itself unpinned (that is the owed
    question on #3137), so what this holds is the agreement between two copies of one
    fact on one page, which is exactly what `2e08e6d3` broke and `55f6eda1` restored
    by hand. Neither side is the live box, per this file's own rule.
    """
    problems = _bus_mismatches(_index_map_rows())
    assert not problems, "; ".join(problems)
    buses = _clamp_bus_ids()
    assert buses == {0: "01:00.0", 1: "41:00.0", 2: "42:00.0",
                     3: "61:00.0", 4: "62:00.0"}, buses


def test_the_index_map_checks_go_red_on_the_table_that_shipped_before_55f6eda1():
    """The real pre-fix table, spliced into the real page, fails every §2.4 check —
    and says index 2 while doing it.

    Not a mutation invented for the occasion: `_PREFIX_INDEX_MAP` is `55f6eda1^`'s
    §2.4. Three rows, index 2 on `61:00.0`, card 2 let to the llama.cpp secondary.
    """
    text = VLLM.read_text(encoding="utf-8")
    current = "\n".join(_index_map_table_of(text))
    mutated = text.replace(current, "\n".join(_PREFIX_INDEX_MAP), 1)
    assert mutated != text, "the splice did not reach the doc, so nothing here is proven"
    rows = _index_map_rows_of(mutated)
    assert sorted(rows) == [0, 1, 2] and rows[2]["pci"] == "61:00.0", rows
    assert _index_map_gaps(rows) == [
        "index 3: no row, and the box has indices 0-4",
        "index 4: no row, and the box has indices 0-4",
    ]
    mismatches = _bus_mismatches(rows)
    assert len(mismatches) == 1 and mismatches[0].startswith("index 2:"), mismatches
    assert "61:00.0" in mismatches[0] and "42:00.0" in mismatches[0], mismatches
    hits = _disarmed_tenant_hits(rows, _disarmed_slots())
    assert len(hits) == 1 and hits[0].startswith("index 2:"), hits
    assert "secondary_enabled" in hits[0] and "llama.cpp secondary" in hits[0], hits
    declared = set(_conf_programs())
    named = {i: _named_programs(r["serves"], declared) for i, r in rows.items()}
    assert named == {0: set(), 1: set(), 2: set()}, (
        f"the pre-fix rows named a resolvable program ({named}), so the node that "
        "requires a pinned program in every tenanted row would have passed the "
        "document that was actually wrong")
    problems = _index_map_problems(rows)
    assert sum(1 for p in problems if p.startswith("index 2:")) >= 3, problems
    assert not _index_map_problems(_index_map_rows()), "the real doc is not clean"


# --- clause 3: a row's tenant is a program, and pinned to that card ------------

def test_every_index_map_row_with_a_tenant_names_a_program_pinned_to_that_card():
    """The card map's tenants are conf.d programs whose pin resolves to that index.

    Three pin spellings resolve, and all three are load-bearing on today's tree: the
    conf's own `CUDA_VISIBLE_DEVICES=`/`GPU=`; `agent-llm-primary`'s launcher export,
    which is the only pin that program has; and `agent-tts-omni`'s UUID, attributed by
    that launcher's `# GPU 3` line. Rows 0, 1 and 3 used to carry prose instead of a
    program token ("the omni TTS (pinned by GPU UUID, not index)"), which nothing but
    a human could check; they name the program now, the way `architecture/
    infrastructure.md` §GPUs rows already do.
    """
    rows = _index_map_rows()
    declared = set(_conf_programs())
    owned = _card_ownership()
    for idx, row in sorted(rows.items()):
        named = _named_programs(row["serves"], declared)
        if owned.get(str(idx)):
            assert named, f"index {idx}'s row ({row['serves']!r}) names no program"
        for program in sorted(named):
            assert _card_pin(program) == str(idx), (
                f"index {idx}: {program} pins card {_card_pin(program)}")
    assert _card_pin("agent-llm-primary") == "1"
    assert _card_pin("agent-tts-omni") == "3"
    assert _card_pin("agent-djev") == "2"
    assert "agent-qmd-daemon" in _named_programs(rows[0]["serves"], declared)


def test_the_pin_derivation_reads_the_launcher_and_the_uuid_not_just_the_conf():
    """The two pins a conf-only reading loses, asserted rather than assumed.

    `agent-llm-primary.conf` sets no GPU key and is the box's only 96 GB card, and
    `agent-tts-omni.conf` pins a UUID that is an index only because its launcher says
    so. Both cases are what #3137's row-3 wording was hiding.
    """
    owned = _card_ownership()
    assert _conf_env_pin("agent-llm-primary") is None, (
        "agent-llm-primary.conf gained a conf-side pin, so the launcher leg below is "
        "no longer the only thing that knows where the primary runs")
    assert _launcher_of("agent-llm-primary").name == "start-qwen38-flash-next.sh"
    assert owned["1"] == {"agent-llm-primary"}, owned.get("1")
    omni = _conf_env_pin("agent-tts-omni")
    assert _UUID_PIN.fullmatch(omni), f"the omni TTS no longer pins a UUID: {omni!r}"
    assert owned["3"] == {"agent-tts-omni"}, owned.get("3")
    assert _card_pin("lloyd-agent-worker") == "0", (
        "the worker's environment= pins 0 and its comment quotes GPU=2: if this "
        "returns 2, the whole-file scan crept back in")


# --- clause 4: an unowned card says so, over the set the confs give ------------

def test_a_row_says_nothing_pins_its_card_exactly_when_no_conf_pins_that_index():
    """The Serves cells and conf.d agree about which cards have no tenant.

    One direction only, on purpose: `agent-decider` and `agent-llm-secondary` both pin
    card 2 and neither is named in its row — the row names the tenant that answers,
    not every process that allocates. Demanding all of them would fail a correct doc,
    which is the mistake the item's original wording made.
    """
    rows, owned = _index_map_rows(), _card_ownership()
    problems = _ownership_wording_problems(rows, owned)
    assert not problems, "; ".join(problems)
    named_on_two = _named_programs(rows[2]["serves"], set(_conf_programs()))
    assert named_on_two == {"agent-djev"}, named_on_two
    assert {"agent-decider", "agent-llm-secondary"} <= owned["2"], owned["2"]
    assert not _ownership_wording_problems(rows, owned), (
        "card 2 names one of its three tenants and the check demanded all three")


def test_the_unowned_card_check_follows_the_confs_and_not_a_hardcoded_four():
    """The unowned set is derived, and proved from both directions.

    Today exactly one card of the five has no pin, and its row says so in words; the
    day that changes, the row has to change with it. Feeding the check a tenant on
    card 4, and then removing the primary's from card 1, shows it moves with the map
    instead of with the number 4.
    """
    rows, owned = _index_map_rows(), _card_ownership()
    says = [i for i, r in sorted(rows.items()) if _NOTHING_PINNED.search(r["serves"])]
    unowned = [i for i in sorted(rows) if str(i) not in owned]
    assert says == unowned, (says, unowned, sorted(owned))
    gained = _ownership_wording_problems(rows, {**owned, "4": {"agent-tts"}})
    assert len(gained) == 1 and gained[0].startswith("index 4:"), gained
    lost = _ownership_wording_problems(
        rows, {k: v for k, v in owned.items() if k != "1"})
    assert len(lost) == 1 and lost[0].startswith("index 1:"), lost


# --- clause 5: a row may not name a slot the tree has switched off -------------

def test_no_index_map_row_names_a_tenant_whose_switch_is_off():
    """While `secondary_enabled: false`, no row may credit a card to the secondary.

    Pin existence is not the test — `agent-llm-secondary.conf` still pins `GPU="2"`,
    so "a card the doc gives a tenant must have a conf pinning it" passes the very row
    that was wrong. Armedness is the discriminator, read through `app.llm_slots`
    against config.yaml on disk.
    """
    disarmed = _disarmed_slots()
    assert "agent-llm-secondary" in disarmed, sorted(disarmed)
    problems = _disarmed_tenant_hits(_index_map_rows(), disarmed)
    assert not problems, "; ".join(problems)


def test_armedness_comes_from_config_yaml_and_never_from_autostart():
    """Both wrong ways of asking this question, pinned so neither comes back.

    `agent-djev.conf` is `autostart=false` while `djev.enabled: true` arms it — the
    conf's own comment explains that supervisord autostart is not the switch — so a
    check reading `autostart=` fails a correct doc. And `agent-llm-secondary.conf`
    still pins `GPU="2"` while switched off, so a check reading pins passes an
    incorrect one.
    """
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    disarmed = _disarmed_slots()
    assert "agent-djev" not in disarmed, sorted(disarmed)
    assert llm_slots.is_enabled("agent-djev", cfg) is True
    assert cfg["secondary_enabled"] is False
    assert "autostart=false" in _conf_text("agent-djev"), (
        "agent-djev.conf is no longer autostart=false, and it was the counterexample "
        "that made this node read config.yaml instead")
    assert _card_pin("agent-llm-secondary") == "2", (
        "the switched-off secondary no longer pins card 2, so pin existence would no "
        "longer be the trap this item records — re-derive the check's basis")
    assert "agent-llm-secondary" in _card_ownership()["2"]
