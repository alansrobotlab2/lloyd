"""#3754: the GPU pins in `conf.d` are graded against the cards the driver answers for.

On 2026-10-31 16:40 `agent-djev` began refusing to start on GPU 2 while `agent-tts-omni`
held that same silicon. The two unit files did not disagree in any way a reader could see:
`agent-djev.conf:26` pins `GPU="2"` — a launcher variable, not CUDA's — and
`agent-tts-omni.conf:14` pins `CUDA_VISIBLE_DEVICES="GPU-0bd41021-2916-fb5d-c100-5bdc5a9c41c0"`
under `CUDA_DEVICE_ORDER="PCI_BUS_ID"`. One card, two vocabularies, no shared string, and
`agent-services/bin/set-gpu-power-limit.sh:15-19` had already documented why the index
spelling is the fragile half: "If a card drops off the bus (the known defect) the remaining
cards shift down, so a 3090 could inherit an override meant for the PRO 6000." The outage
lasted 17 hours, to the 2026-11-01 09:45 power cycle, and no check on the box named either
fact — `git grep -n 'CUDA_VISIBLE_DEVICES' -- scripts/` returned nothing.

Each test names the acceptance clause it pins and feeds the three readings (pins, cards,
compute apps) as fakes, for the reason `_live_power_limits` is injectable: the fleet's GPU
truth lives only in `nvidia-smi`, and a test that called it would be an assertion about
whatever hardware the check happens to run on.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "service_health_check.py"


def _load_module():
    """Load the CLI script by path: it has no package and nothing imports it."""
    spec = importlib.util.spec_from_file_location("shc_gpu_presence", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shc = _load_module()

# Two cards, named the way `nvidia-smi` names them. The uuid spellings are shaped like the
# real ones so a uuid-typed pin reads as one; no claim here is about this box's hardware.
UUID0 = "GPU-0bd41021-2916-fb5d-c100-5bdc5a9c41c0"
UUID1 = "GPU-6287dd24-9d9c-4ad6-236b-9a17498d0328"
# nvidia-smi's index, as the reader returns it: an INT key, so a fixture keyed by the
# string "1" would make every pin look unanswerable and every card look unclaimed.
TWO_CARDS = {0: {"uuid": UUID0, "name": "NVIDIA GeForce RTX 3090"},
             1: {"uuid": UUID1, "name": "NVIDIA GeForce RTX 3090"}}


def _run(pins, cards=TWO_CARDS, apps=()):
    return shc.check_gpu_presence(pins=lambda: pins,
                                  enumeration=lambda: dict(cards),
                                  apps=lambda: list(apps))


def _graded(rows):
    """The rows that can turn this heading unhealthy: graded by `_scored_rows` — the
    script's ONE definition of what the verdict is arithmetic over, carrying #1726's
    advisory exclusion — and not healthy. `exit_code` is deliberately not the test:
    a card is not a supervisor program, and every non-service row in this file
    (gpu-power's drift and unknown arms alike) carries 0 for it."""
    return [r for r in shc._scored_rows(rows) if not r["healthy"]]


def _named(rows, suffix):
    return [r for r in rows if r["name"] == f"{shc.GPU_PRESENCE_CATEGORY}:{suffix}"]


def test_a_pin_naming_a_card_nvidia_smi_does_not_answer_for_is_red():
    """Clause 4, first half: `GPU="4"` on a box that enumerates two cards is the
    index-shift signature — the pin survives, the card it meant does not — and it
    has to be a red row that names the program AND the card, because the remedy is
    re-pointing the pin and nothing else."""
    rows = _run({"agent-djev": {("index", 4)}})
    bad = _named(rows, "agent-djev")
    assert len(bad) == 1, rows
    row = bad[0]
    assert row["verdict"] == "missing-card", row
    assert row["healthy"] is False and not row.get("advisory"), row
    assert "agent-djev" in row["status"], row["status"]
    assert 'GPU="4"' in row["status"], row["status"]
    text = row["status"] + row["output"]
    assert "GPU 0" in text and "GPU 1" in text, (
        f"a reader has to see which cards DO answer: {text}")
    assert _graded(rows) == [row], rows


def test_two_programs_resolving_to_one_card_by_different_spellings_is_red():
    """Clause 4, second half — #3675's exact state. One launcher index, one CUDA
    uuid, the same silicon: nothing in the directory shares a string between them,
    so only resolving both against the driver can say it. Red, naming BOTH programs
    and the card they landed on."""
    rows = _run({"agent-djev": {("index", 1)},
                 "agent-tts-omni": {("uuid", UUID1)}})
    clash = _named(rows, "collision")
    assert len(clash) == 1, rows
    row = clash[0]
    assert row["verdict"] == "collision", row
    assert row["healthy"] is False and not row.get("advisory"), row
    assert "agent-djev" in row["status"] and "agent-tts-omni" in row["status"], row["status"]
    assert "GPU 1" in row["status"], row["status"]
    assert sorted(row["programs"]) == ["agent-djev", "agent-tts-omni"], row
    assert "index" in row["spellings"] and "uuid" in row["spellings"], row
    assert _graded(rows) == [row], rows


def test_programs_sharing_one_card_under_the_same_spelling_stay_advisory():
    """The negative control that keeps clause 4 honest. Three programs write
    `GPU="2"` on this fleet today as a declared either/or between the djev, decider
    and llama.cpp slots, and a check that called that an outage would report red on
    a healthy box every day — and then stop being read."""
    rows = _run({"agent-djev": {("index", 0)},
                 "agent-decider": {("index", 0)},
                 "agent-llm-secondary": {("index", 0)}})
    shared = _named(rows, "shared-pin")
    assert len(shared) == 1, rows
    row = shared[0]
    assert row["verdict"] == "shared-pin", row
    assert row["advisory"] is True, row
    assert row not in shc._scored_rows(rows), "an advisory row is not arithmetic"
    assert "agent-djev" in row["status"] and "agent-decider" in row["status"], row["status"]
    assert _graded(rows) == [], f"a declared either/or must never go red: {rows}"


def test_a_box_answering_no_gpu_rows_cannot_go_red():
    """Clause 4's carve-out: zero cards is what a driver-less image legitimately
    reports, and a box whose cards have all left the bus looks identical from here.
    Advisory and exit 0 — the same treatment `check_gpu_power_limit` gives an empty
    live list — so the check cannot become the alarm it is not qualified to raise."""
    rows = _run({"agent-djev": {("index", 1)}}, cards={})
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["verdict"] == "unknown", row
    assert row["advisory"] is True, row
    assert row["exit_code"] == 0, row
    assert row not in shc._scored_rows(rows), rows
    assert _graded(rows) == [], rows


def test_a_box_with_no_driver_at_all_cannot_go_red_either():
    """The rarer half of the same rule: `nvidia-smi` missing from PATH raises
    rather than answering empty, and an unread driver is not evidence about the
    fleet's hardware either."""
    def boom():
        raise FileNotFoundError("nvidia-smi")
    rows = shc.check_gpu_presence(pins=lambda: {"agent-djev": {("index", 1)}},
                                  enumeration=boom, apps=lambda: [])
    assert len(rows) == 1, rows
    assert rows[0]["verdict"] == "unknown" and rows[0]["advisory"] is True, rows
    assert _graded(rows) == [], rows


def test_a_compute_app_on_a_card_no_pin_names_is_advisory_and_never_red():
    """Clause 5: the hand-launched engine. Since 2026-11-01 10:15 a `vllm serve`
    no unit owns has held 21,576 MiB of a card no `conf.d` file names, and nothing
    on the box said so. It must be VISIBLE — the card is unclaimed, so a launcher
    told to find a free one cannot know it is busy, which is how GPU 2 was
    double-booked — and it must never be red, because going red over somebody's
    running experiment calls a healthy box broken."""
    rows = _run({"agent-djev": {("index", 0)}},
                apps=[{"uuid": UUID1, "pid": "139348", "mib": 21576,
                       "name": "VLLM::EngineCore"}])
    seen = _named(rows, "unpinned-compute")
    assert len(seen) == 1, rows
    row = seen[0]
    assert row["verdict"] == "unpinned-compute", row
    assert row["advisory"] is True, row
    assert row not in shc._scored_rows(rows), "an unclaimed card is never an outage"
    assert row["gpu_index"] == 1 and row["gpu_uuid"] == UUID1, row
    assert row["pids"] == ["139348"], row
    text = row["status"] + row["output"]
    assert "139348" in text and "21576 MiB" in text, text
    assert "VLLM::EngineCore" in text, text
    assert _graded(rows) == [], f"an unclaimed card is never an outage: {rows}"


def test_a_pinned_card_carrying_compute_is_not_reported_as_unclaimed():
    """The other side of clause 5's word "no pin names": `agent-qmd-daemon` and
    `agent-tts` hold card 0 through a CUDA device list (kind `cuda-index`), and an app answering on that
    card is exactly where those pins say it should be. Counting only exclusive
    claims would stamp the busiest card in the fleet as unclaimed every day."""
    pinned_card = {"agent-qmd-daemon": {("cuda-index", 0)},
                   "agent-tts": {("cuda-index", 0)}}
    rows = _run(pinned_card, apps=[{"uuid": UUID0, "pid": "7001", "mib": 1200,
                                   "name": "python3"}])
    assert _named(rows, "unpinned-compute") == [], (
        f"a card a mask pins is a named card: {[r['status'] for r in rows]}")
    assert _graded(rows) == [], rows


def test_the_row_shape_and_the_wiring_are_the_ones_the_formatters_read():
    """Every row carries the six keys `format_json` and the category summary
    require, the heading is a real one, and `main` calls the check only for a
    whole-fleet run or its own category — the rule `check_gpu_power_limit`,
    `check_ca_trust` and `check_path_sandbox` each follow, so a `--services` ask
    never pays for two `nvidia-smi` calls."""
    rows = _run({"agent-djev": {("index", 0)}})
    for row in rows:
        assert {"name", "status", "healthy", "exit_code", "output", "category",
                "verdict", "advisory"} <= set(row), sorted(row)
        assert row["category"] == shc.GPU_PRESENCE_CATEGORY, row
        assert row["exit_code"] == 0, (
            "a card row is not a supervisor program and must not claim one's code")
    assert shc.CATEGORIES[shc.GPU_PRESENCE_CATEGORY] == [], (
        "the heading is answered by main, not by a supervisor program")
    src = SCRIPT.read_text(encoding="utf-8")
    assert ("if not args.services and args.category in (None, GPU_PRESENCE_CATEGORY):"
            "\n        results.extend(check_gpu_presence())") in src, (
        "main must gate the check on the same rule as the other non-service checks")


def test_a_real_conf_d_directory_is_parsed_without_reading_the_driver():
    """The reader half of clause 4, offline: the pins come out typed, from the
    tracked `conf.d` files this repo owns, with no `nvidia-smi` call anywhere. The
    two spellings have to land as two different claim kinds or the collision rule
    below them can never fire, and a fleet that gained a third spelling would
    silently drop out of the check — which is what the third assertion is for."""
    pins = shc._pinned_gpu_cards()
    assert pins, "the tracked conf.d directory must yield pins from the tree"
    kinds = {kind for claims in pins.values() for kind, _ in claims}
    assert "index" in kinds, f"the GPU= spelling vanished from the roster: {pins}"
    assert "uuid" in kinds or "cuda-index" in kinds, (
        f"CUDA_VISIBLE_DEVICES stopped being readable: {pins}")
    for program, claims in pins.items():
        for kind, value in claims:
            assert kind in ("index", "uuid", "cuda-index"), (program, kind, value)
            if kind == "uuid":
                assert value.startswith("GPU-"), (program, kind, value)
            else:
                # Both index spellings come out as the driver's int, which is what
                # makes a pin comparable with `cards` at all.
                assert isinstance(value, int) and not isinstance(value, bool), \
                    (program, kind, value)
