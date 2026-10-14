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
