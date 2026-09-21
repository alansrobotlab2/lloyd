"""The three still-live incidents, each pinned at the consumer that produced the green
verdict (#644).

Clause 5 of item #644: each incident is a matrix row that fails against unmodified code
and passes after its fix, *and has its own test*. The matrix rows
(`tests/degradation/matrix.yaml`) prove the behaviour end to end through the runner; the
nodes here pin the same three claims directly on the consumers, so the fix cannot be
reverted by a change that leaves the runner's adapters intact.

These tests inject — they bind ports, write a throwaway git repo and stand in a fake
`/proc` — so the module carries the marker the gate's tests rung selects against.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.degradation import runner as R  # noqa: E402

pytestmark = pytest.mark.fault_injection

VAULT_HEALTH = Path.home() / "obsidian" / "skills" / "system-health-check" / "system_health_check.py"

def _load(path: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

SHC = _load(VAULT_HEALTH, "shc_for_consumer_pins")

# ---------------------------------------------------------------- calibration 1: the 404
def test_404_from_an_ignore_404_endpoint_is_not_reported_healthy():
    """`system_health_check.py` used to write `{'healthy': True, 'note': '404'}` for
    exactly this: an endpoint flagged `ignore_404` whose route had gone. A server that
    answers 404 has told us the endpoint does not exist, which is a named state, not a
    pass — and `:8096/health` answers 200 today, so this row can only fire when the
    route really disappears, never as background noise.
    """
    listener = R.FixtureListener(code=404, body=b'{"error": "not found"}')
    try:
        row = SHC.check_endpoints([{"name": "matrix-404", "port": listener.port,
                                   "schemes": ["http"], "ignore_404": True,
                                   "timeout": 2}])[0]
    finally:
        listener.close()
    assert row["state"] == SHC.HTTP_ROUTE_ABSENT, row
    assert row["healthy"] is False, row
    assert "404" in (row.get("error") or ""), row

# ---------------------------------------------------------------- calibration 2: the count
def test_a_git_only_change_does_not_move_the_data_damage_count(tmp_path):
    """`git gc` repacks by the hundred and leaves `git status --porcelain` empty. The
    guardian compares a file count for `data_damage`, and until `add6692` both sides of
    that comparison walked `.git/**` — so 2026-09-09 (#537) and 2026-09-17 (#1206, "vault
    files dropped 6.7% (6075 -> 5667)" while the note count sat at 5622) spent two items'
    single unattended attempts on a repack. A guard that alarms when nothing was lost is
    worse than no guard, because the response to its trip is a rollback.
    """
    vault = R._fixture_vault(tmp_path, notes=30)
    guarded_before, unguarded_before = R.guarded_count(vault), R.unfiltered_count(vault)
    loose_before = R.loose_object_count(vault)
    assert loose_before > 20, f"only {loose_before} loose objects; nothing to repack"
    assert R.git_porcelain(vault) == "", "the fixture repo started dirty"

    R.remove_loose_objects(vault)

    assert R.git_porcelain(vault) == "", "the repack dirtied the tree: not a gc"
    assert R.guarded_count(vault) == guarded_before, (
        f"the data_damage count moved {guarded_before} -> {R.guarded_count(vault)} with "
        f"zero notes touched — the exact reading that filed a rollback twice")
    # The control: if `.git` churn could not move an unfiltered count, the fixture
    # stopped being a repo and the assertion above would be holding nothing up.
    assert R.unfiltered_count(vault) < unguarded_before, (
        f"an unfiltered walk did not move ({unguarded_before} -> "
        f"{R.unfiltered_count(vault)}), so this fixture is not a repo any more")

def test_a_real_note_loss_still_moves_the_count_both_ways(tmp_path):
    """The other side of the same guard: excluding `.git` must not have blinded it. The
    fault the trip exists for — notes actually deleted — still moves both counters.
    """
    vault = R._fixture_vault(tmp_path, notes=30)
    before = (R.guarded_count(vault), R.guardian_count(vault))
    R.delete_notes(vault, notes=30)
    after = (R.guarded_count(vault), R.guardian_count(vault))
    assert before[0] - after[0] == 30, f"{before} -> {after}"
    assert before[1] - after[1] == 30, f"{before} -> {after}"

# ---------------------------------------------------------------- calibration 3: media
def _with_proc_root(root, pid="900001"):
    os.environ[SHC.VOICE_MEDIA_PROC_ROOT] = str(root)
    os.environ[SHC.VOICE_MEDIA_PID_SELECTION] = pid

@pytest.fixture
def clean_proc_env():
    saved = {k: os.environ.get(k) for k in
             (SHC.VOICE_MEDIA_PROC_ROOT, SHC.VOICE_MEDIA_PID_SELECTION)}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

def test_a_bound_media_port_is_the_only_healthy_voice_state(clean_proc_env, tmp_path):
    """The positive control, and it is read from the same parser: one fixture row naming
    UDP 50387 must flip the verdict to green. Reading the socket table's state column
    instead of its local-address column parses no port at all and reports this as nothing
    bound forever — which is what the first version of this component did, and the
    degradation matrix caught it against code landed minutes earlier.
    """
    _with_proc_root(R.proc_tables(tmp_path / "p2", media_ports=(50387,)))
    row = SHC.check_voice_media()
    assert row["state"] == SHC.VOICE_MEDIA_GREEN, row
    assert row["healthy"] is True, row
    assert 50387 in row["media_ports"], row

def test_an_unreadable_socket_table_is_no_probe_and_never_healthy(clean_proc_env, tmp_path):
    """Clause 5's third incident and clause 6's whole point: when the input cannot be
    read, the consumer names that instead of concluding. Before this component existed
    there was no detector at all for the media-socket case.
    """
    _with_proc_root(tmp_path / "absent")
    row = SHC.check_voice_media()
    assert row["state"] == SHC.VOICE_MEDIA_NO_PROBE, row
    assert row["healthy"] is False, row
    assert row["error"], row

# ---------------------------------------------------------------- clause 3: the guard
def test_the_injector_refuses_a_port_a_live_instance_already_owns():
    """The suite must not inject against a dependency that is serving Alan — the report
    would describe a fault this run manufactured, which is the false-DOWN defect this
    item catalogues. Liveness is checked before the bind, because a listening socket is
    by then answering on its own port and a post-bind check would refuse every fixture.
    """
    held = R.FixtureListener()
    try:
        assert R.port_in_use(held.port), "the fixture is not answering; the probe is broken"
        with pytest.raises(R.InjectRefused) as caught:
            R.bind_fixture_socket(preferred=held.port)
        assert "live" in str(caught.value).lower(), str(caught.value)
    finally:
        held.close()


def test_the_injector_admits_a_state_path_inside_the_run_own_root(tmp_path):
    """The positive half of the same guard, as its own node: the same call must ADMIT a path
    under the root this run owns. Folded into the refusal test it read as trailing detail — and
    a guard that always raises would have passed the refusal half while leaving every state row
    in the matrix unfunded, because a row whose fixture is refused is a row that never ran.
    """
    admitted = R.assert_fixture_path(tmp_path / "lloyd" / ".consolidate-lock", tmp_path)
    assert str(admitted).startswith(str(tmp_path.resolve())), admitted


def test_the_injector_refuses_a_state_path_a_live_job_owns_and_leaves_it_alone(tmp_path):
    """The state-file half of clause 3, which on this box is the half that matters: most of
    the gate bugs this matrix catalogues are file-based, so the tempting fixture is always the
    real file. `~/obsidian/lloyd/.consolidate-lock` is the one whose mtime the dream gate
    reads on a 24 h timer — a row that aged it to prove the staleness fault would consolidate
    a vault that was not due, and report green while doing it. Refusal is not enough either:
    the assertion is that the file is still there with its mtime intact afterwards.
    """
    live = Path.home() / "obsidian" / "lloyd" / ".consolidate-lock"
    assert live.exists(), f"{live} is absent, so this fixture would hold nothing"
    mtime_before = live.stat().st_mtime
    with pytest.raises(R.InjectRefused) as caught:
        R.assert_fixture_path(live, tmp_path)
    assert "outside" in str(caught.value).lower(), str(caught.value)
    assert live.exists() and live.stat().st_mtime == mtime_before, (
        "the guard refused and had already touched the file")


# ---------------------------------------------------------------- clause 2: one command
def test_one_command_runs_the_matrix_and_prints_exactly_one_evidence_line_per_row():
    """Clause 2, pinned at the CLI rather than at the library: `python -m
    tests.degradation.runner --check` is that one command, headless, CPU-only, no cloud call
    and no interactive session, and its output contract is one line per row shaped
    `row_id | injected fault | what the consumer reported | verdict | evidence`.

    Run as a subprocess on purpose: the clause is about the command a person or a gate rung
    types, so asserting on `run_row`'s return dict would pin the wrong surface. The bound the
    clause states is under ten minutes; the runner holds itself to `SUITE_TIMEOUT_S`, and the
    measured wall time of a full pass is a few seconds.
    """
    rows = R.load_matrix()
    assert len(rows) >= 15, f"the matrix shrank to {len(rows)} rows"
    started = time.monotonic()
    proc = subprocess.run([sys.executable, "-m", "tests.degradation.runner", "--check"],
                          cwd=ROOT, capture_output=True, text=True, timeout=540)
    elapsed = time.monotonic() - started
    lines = [ln for ln in proc.stdout.splitlines() if ln.count(" | ") >= 4]
    assert len(lines) == len(rows), (
        f"{len(lines)} evidence lines for {len(rows)} rows:\n" + "\n".join(lines))
    assert {ln.split(" | ", 1)[0] for ln in lines} == {r["id"] for r in rows}, (
        "the printed row ids are not the matrix's row ids")
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    assert elapsed < 600, f"{elapsed:.1f}s, over the clause's ten minutes"


# ------------------------------------------------- clause 6: one node per enumerated consumer
_BLINDED = [row for row in R.load_matrix() if row["expect"].get("non_healthy")]
_BY_CONSUMER: dict[str, list] = {}
for _row in _BLINDED:
    _BY_CONSUMER.setdefault(_row["consumer"], []).append(_row)


@pytest.mark.parametrize("consumer", sorted(_BY_CONSUMER))
def test_a_consumer_blinded_by_a_fault_names_a_non_healthy_verdict(consumer):
    """Clause 6, executed: for every row whose fault makes the backing probe unreadable or
    absent, the consumer emits a named non-healthy verdict, and nothing it can emit is a
    silent `healthy`. One node per enumerated consumer, so a regression reads as
    "check_voice_media stopped naming its blind spot", not as a count in a log line.
    """
    assert len(_BY_CONSUMER) >= 4, (
        f"only {len(_BY_CONSUMER)} blinded consumers enumerated: the set is derived from the "
        f"matrix and has emptied, which is the vacuous-green shape this item catalogues")
    for row in _BY_CONSUMER[consumer]:
        record = R.run_row(row)
        reported = str(record["reported"]).strip()
        assert reported, f"{row['id']}: {consumer} emitted no verdict at all"
        assert reported not in R.HEALTHY_TOKENS, (
            f"{row['id']}: {consumer} reported {reported!r} — reads as healthy — on a fault "
            f"that removed the probe behind it")
        assert not R.HEALTHY_EVIDENCE_RE.search(record["evidence"] or ""), (
            f"{row['id']}: {consumer} wrote healthy=True into its evidence: {record['evidence']}")
        assert record["passed"], f"{row['id']}: {record['evidence']}"
        # And the ban has to be the matcher's, not this test's: both arms are asserted through
        # `verdict_problems` against the row and its real evidence, so deleting the enforcement
        # from the runner reddens this node too. A check re-implemented here would keep passing
        # over a runner that had stopped enforcing it, which is the difference between a guard
        # and a rumour (verified by mutation: with the arm disabled, only these two lines fail).
        smuggled = R.verdict_problems(row, row["expect"]["reported"],
                                      f"{record['evidence']} healthy=True")
        assert any("healthy=True" in p for p in smuggled), (
            f"{row['id']}: {consumer} could smuggle healthy=True into its own evidence and the "
            f"row would still pass — {smuggled}")
        silence = R.verdict_problems(row, "", record["evidence"])
        assert any("empty verdict" in p for p in silence), (
            f"{row['id']}: {consumer} could say nothing and the row would still pass — {silence}")

