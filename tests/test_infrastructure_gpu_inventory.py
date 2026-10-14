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
"""

from __future__ import annotations

import re
from pathlib import Path

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
