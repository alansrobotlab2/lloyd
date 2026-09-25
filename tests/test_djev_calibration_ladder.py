"""#1479: the calibration ladder a djev yes/no schema climbs before it may gate."""
from __future__ import annotations

import json
import random

import pytest

from eval.djev import calibration_ladder as L


def _rows(n, sep, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        y = rng.random() < 0.3
        z = rng.gauss(sep if y else -sep, 1.0)
        out.append((L.sigmoid(z), int(y)))
    return out


def test_refuses_under_the_label_floor():
    with pytest.raises(ValueError, match="at least 200"):
        L.ladder(_rows(199, 2.0))


def test_a_separable_schema_gates_within_alpha_and_abstains_on_the_middle():
    rep = L.ladder(_rows(900, 2.5, seed=1), alpha=0.05)
    assert rep["error_among_accepted"] is not None and rep["error_among_accepted"] <= 0.05
    assert 0.0 < rep["abstain_rate"] < 1.0
    assert rep["gate_ready_candidate"] is True


def test_a_coin_flip_schema_never_becomes_a_gate():
    rep = L.ladder(_rows(900, 0.0, seed=2), alpha=0.05)
    # Nothing to separate. Marginal coverage still holds, but the rows the
    # conformal set does accept are accepted on the class prior, so their
    # error is far above alpha — which is exactly what must keep it a non-gate.
    assert rep["coverage"] >= 0.9
    assert rep["error_among_accepted"] > 0.05
    assert rep["gate_ready_candidate"] is False


def test_batch_calibration_removes_a_constant_lean():
    ps = [L.sigmoid(z + 3.0) for z in (-1.0, 0.0, 1.0)]
    cal = L.batch_calibrate(ps)
    assert abs(sum(L.logit(p) for p in cal)) < 1e-9


def test_the_cli_refuses_an_unlabelled_file(tmp_path, capsys):
    f = tmp_path / "labels.jsonl"
    f.write_text("\n".join(json.dumps({"djev_p_same": 0.4, "label": None}) for _ in range(300)))
    assert L.main(["--labels", str(f)]) == 2
    assert "0 labelled rows" in capsys.readouterr().err
