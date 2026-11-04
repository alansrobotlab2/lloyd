"""The Stage-1 complement warp's guarantees (#3642 clauses 3 and 4).

Clause 3: the warp is a pure function of ``(seed, strong, weak, knobs)`` in
which the injected residual is orthogonal to the unit-normalised seed and no
longer than ``cap * ||seed||`` for every input, and an empty weak set returns
the seed unchanged.

Clause 4: the control mode is reachable through the SAME arm switch, injects a
random unit vector at the same cap with the neighbour projection skipped, is
reproducible for a given seed key, and differs from warp-on only in how the
residual was produced.

Every cap and orthogonality assertion is checked over a fixed pseudo-random
battery of shapes and scales rather than one happy vector, because both
guarantees are stated as "for every input".
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "scripts" / "retrieval" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


W = _load("complement_warp")


def _battery(rng: np.random.Generator, n: int = 24):
    """Deterministic (seed, strong, weak) triples across shapes, scales and
    degenerate cases: axis-parallel weak sets, a duplicate substitute, a seed
    of zeros, ragged and zero neighbours, one neighbour, many."""
    cases = []
    for i in range(n):
        dim = int(rng.integers(4, 40))
        scale = float(10.0 ** rng.uniform(-2, 2))
        seed = rng.standard_normal(dim) * scale
        n_strong = int(rng.integers(0, 5))
        n_weak = int(rng.integers(0, 6))
        strong = [seed * 1.7 + rng.standard_normal(dim) * 1e-6 * scale
                  if j == 0 else rng.standard_normal(dim) * scale
                  for j in range(n_strong)]
        weak = list(rng.standard_normal((n_weak, dim)) * scale)
        if i % 5 == 0:                       # weak set exactly on the seed axis
            weak = [seed * f for f in (1.0, -1.0, 2.0)][:max(1, n_weak)]
        if i % 7 == 0:                       # degenerate neighbours among the weak
            weak = weak + [np.zeros(dim), seed * 0.0]
        if i % 11 == 0:                      # ragged neighbour, wrong dimension
            weak = weak + [rng.standard_normal(dim + 3)]
        cases.append((seed, strong, weak))
    cases.append((np.zeros(8), [np.ones(8)], [np.ones(8)]))       # zero seed
    cases.append((np.ones(8), [], [np.ones(8) * 3]))              # no substitutes
    return cases


CASES = _battery(np.random.default_rng(20261104))


# ── clause 3: the warp's guarantees ─────────────────────────────────────

def _usable(vecs, dim):
    return [np.asarray(v, dtype=np.float64) for v in vecs
            if np.asarray(v).size == dim and np.linalg.norm(v) > 1e-12]


@pytest.mark.parametrize("idx", range(len(CASES)))
def test_the_injected_residual_is_orthogonal_to_the_unit_seed(idx):
    """Clause 3, first half: the residual is off-axis for EVERY input.

    Gram-Schmidt makes this exact for whatever amplitude survives, so the test
    only has to look at a residual the module judged big enough to inject; a
    residual of pure conditioning slop (see the parallel test below) carries no
    direction to speak of, which is why the assertion is gated on a real
    residual rather than waived.
    """
    seed0, strong, weak = CASES[idx]
    seed = np.asarray(seed0, dtype=np.float64).reshape(-1)
    out = W.warp_seed(seed, strong, weak)
    if float(np.linalg.norm(seed)) <= 1e-9 or not _usable(weak, seed.size):
        assert np.array_equal(out, seed), "a zero seed or unusable weak set must be untouched"
        return
    residual = out - seed
    nr = float(np.linalg.norm(residual))
    if nr <= 1e-9 * float(np.linalg.norm(seed)):
        return
    cos = residual @ seed / (nr * float(np.linalg.norm(seed)))
    assert abs(cos) < 1e-9, f"residual leaked along the seed axis: cos={cos}"


@pytest.mark.parametrize("idx", range(len(CASES)))
def test_the_injected_residual_is_no_longer_than_cap_times_the_seed_norm(idx):
    seed, strong, weak = CASES[idx]
    knobs = W.WARP_KNOBS
    out = W.warp_seed(seed, strong, weak, knobs)
    residual = out - np.asarray(seed, dtype=np.float64).reshape(-1)
    bound = knobs.cap * float(np.linalg.norm(seed))
    assert np.linalg.norm(residual) <= bound + 1e-9, (
        f"residual {np.linalg.norm(residual)} exceeds cap {knobs.cap} x seed norm "
        f"{np.linalg.norm(seed)}")


def _off_axis_pair(rng, dim):
    """A seed and a unit direction orthogonal to it, so the evidence is 1.0."""
    seed = rng.standard_normal(dim)
    off = rng.standard_normal(dim)
    off -= (off @ seed) * seed / float(seed @ seed)
    return seed, off / float(np.linalg.norm(off))


def test_the_cap_is_a_bound_the_code_reaches_and_a_tighter_one_binds_exactly():
    """Clause 3, second half, read as a live ceiling rather than a dead one.

    The weak evidence here is a unit neighbour sitting exactly off the seed's
    axis, so the surviving evidence is 1.0 and the requested amplitude is
    `min(1.0, cap)` = the cap itself. With no substitutes in the neighbourhood
    there is no local gap to preserve, so nothing else bites and the residual
    lands ON 0.35 and 0.07 of the seed's own norm, to the float.
    """
    seed, off = _off_axis_pair(np.random.default_rng(3), 32)
    n = float(np.linalg.norm(seed))
    for cap in (0.35, 0.07):
        out = W.warp_seed(seed, [], [off, off * 3.0, off * 0.2], W.WarpKnobs(cap=cap))
        assert abs(float(np.linalg.norm(out - seed)) - cap * n) < 1e-9
    wide = W.warp_seed(seed, [], [off], W.WarpKnobs(cap=0.35))
    tight = W.warp_seed(seed, [], [off], W.WarpKnobs(cap=0.07))
    assert abs(float(np.linalg.norm(wide - seed)) / float(np.linalg.norm(tight - seed))
               - 0.35 / 0.07) < 1e-9


def test_an_empty_weak_set_returns_the_seed_unchanged():
    rng = np.random.default_rng(11)
    seed = rng.standard_normal(16)
    strong = list(rng.standard_normal((3, 16)))
    for empty in ([], None, ()):
        assert np.array_equal(W.warp_seed(seed, strong, empty), seed)
    # and every battery case with an empty weak set, likewise
    for seed, strong, weak in CASES:
        if not weak:
            assert np.array_equal(W.warp_seed(seed, strong, weak),
                                  np.asarray(seed, dtype=np.float64).reshape(-1))


def test_a_weak_set_parallel_to_the_seed_returns_the_seed_unchanged():
    """The cosine punchline: information along the seed's own direction moves nothing.

    A weak set of pure parallels leaves no off-axis remainder to inject, so the
    warp must decline rather than hand back a longer vector that cosine would
    read as identical — a longer-but-parallel result is the bug this pins.
    """
    seed = np.array([0.4, -1.2, 3.0, 0.7])
    assert np.array_equal(W.warp_seed(seed, [], [seed * 2.0, seed * -0.5]), seed)
    assert np.array_equal(W.warp_seed(seed, [], [seed * 3.0]), seed)


def test_a_weak_set_with_no_off_axis_evidence_does_not_buy_a_cap_sized_step():
    """The reinjection is proportional to the surviving evidence, capped.

    A near-duplicate substitute — cosine 0.99996 to the seed, the ordinary
    shape of a `supersedes` pair in this store — makes the SVD stack
    ill-conditioned, and a soft conceptor on that stack leaks conditioning slop
    of order 1e-3. If the residual were renormalised before injection, that
    slop would be scaled to the FULL 0.35 cap and the seed would be shoved
    sideways on no evidence at all — exactly the failure mode the control arm
    exists to detect, baked into the warp itself. Under the cap-as-ceiling
    reading the same neighbourhood moves by under 1% of the seed, not by 35%.
    """
    rng = np.random.default_rng(91)
    dim = 40
    seed = rng.standard_normal(dim)
    seed_u = seed / np.linalg.norm(seed)
    tilt = rng.standard_normal(dim)
    tilt -= (tilt @ seed_u) * seed_u
    tilt /= np.linalg.norm(tilt)
    near = 0.99996 * seed_u + 0.0089 * tilt          # cosine ~0.99996: a duplicate
    weak = [seed * 1.5]                              # a pure parallel neighbour
    out = W.warp_seed(seed, [near], weak, W.WarpKnobs(cap=0.35))
    rel = float(np.linalg.norm(out - seed)) / float(np.linalg.norm(seed))
    assert rel < 1e-2, f"a parallel weak set moved the seed by {rel:.2e} of its norm"
    assert rel < 0.35 / 20, "the near-duplicate case must be far inside the cap"
    # and the SAME seed with real off-axis complement evidence reaches the cap
    seed2, off = _off_axis_pair(rng, dim)
    out2 = W.warp_seed(seed2, [near], [off], W.WarpKnobs(cap=0.35))
    rel2 = float(np.linalg.norm(out2 - seed2)) / float(np.linalg.norm(seed2))
    assert rel2 > 0.3, f"real evidence should reach the cap, got {rel2:.4f}"


def test_the_warp_is_a_pure_function_and_does_not_touch_its_inputs():
    seed, strong, weak = CASES[1]
    seed_c, strong_c, weak_c = seed.copy(), [v.copy() for v in strong], [v.copy() for v in weak]
    first = W.warp_seed(seed, strong, weak)
    second = W.warp_seed(seed, strong, weak)
    assert np.array_equal(first, second)
    assert np.array_equal(seed, seed_c)
    assert all(np.array_equal(a, b) for a, b in zip(strong, strong_c))
    assert all(np.array_equal(a, b) for a, b in zip(weak, weak_c))
    # the knobs are part of the input: a different aperture is a different answer
    other = W.warp_seed(seed, strong, weak, W.WarpKnobs(aperture=1.0))
    assert not np.array_equal(first, other)


def test_the_conceptor_is_soft_so_a_different_aperture_is_a_different_operator():
    """A diagonal shrinkage per direction, not one rank cut (module header).

    The aperture runs from `P = 0` (weak neighbours averaged unprojected) to
    every weight at 1 (the hard full-rank cut), so three apertures must give
    three different warps of the same neighbourhood — and every one of them
    still has to respect the cap, which is what makes the knob safe to move.
    """
    rng = np.random.default_rng(5)
    seed = rng.standard_normal(24)
    strong = list(rng.standard_normal((8, 24)))
    weak = list(rng.standard_normal((6, 24)))
    outs = {a: W.warp_seed(seed, strong, weak, W.WarpKnobs(aperture=a))
            for a in (0.5, 12.0, 1e6)}
    assert not np.allclose(outs[0.5], outs[12.0])
    assert not np.allclose(outs[12.0], outs[1e6])
    for out in outs.values():
        assert float(np.linalg.norm(out - seed)) <= 0.35 * float(np.linalg.norm(seed)) + 1e-9
    # aperture 0 is the conceptor switched off: P = 0, so the raw weak mean
    # survives unprojected and the seed still moves
    moved = W.warp_seed(seed, strong, weak, W.WarpKnobs(aperture=0.0))
    assert not np.allclose(moved, seed)


def test_the_gap_constraint_bounds_the_amplitude_from_above_and_is_monotone():
    """A hand-built neighbourhood whose substitute gap the warp must not eat.

    Two unit substitutes at 0.1 and 0.2 radians from the seed, so their cosine
    gap is 0.0152. A residual pointing toward the wider-angle substitute raises
    that neighbour's cosine faster and NARROWS the gap, so the feasible
    amplitudes stop well below the cap; the same residual reversed widens the
    gap and the cap stands unconstrained.
    """
    seed_u = np.array([1.0, 0.0])
    strong_u = np.array([[math.cos(0.1), math.sin(0.1)],
                         [math.cos(0.2), math.sin(0.2)]])
    narrows = np.array([0.0, 1.0])
    widens = np.array([0.0, -1.0])
    blocked = W._gap_preserving_amplitude(seed_u, strong_u, narrows, 0.35, 0.4)
    allowed = W._gap_preserving_amplitude(seed_u, strong_u, widens, 0.35, 0.4)
    assert allowed == 0.35, "a residual that widens the gap must not be cut"
    assert 0.05 < blocked < 0.12, blocked
    assert W._gap_preserving_amplitude(seed_u, strong_u, narrows, 0.35, 0.2) >= blocked
    assert W._gap_preserving_amplitude(seed_u, strong_u, narrows, 0.35, 0.9) <= blocked


def test_the_warp_never_reorders_a_tight_substitute_neighbourhood():
    """What "the neighbourhood is barely disturbed" is allowed to mean exactly.

    Substitutes two and three sit 0.001 and 0.002 of cosine from the seed — an
    ordinary `supersedes` pair in this store — so a naive "keep the distance to
    the seed" rule would forbid the warp outright. Stated on the GAP between
    neighbours, the warp goes ahead and the pair keeps at least
    ``gap_keep_ratio`` of the gap it had, in the same order, for any weak set.
    """
    rng = np.random.default_rng(21)
    dim = 40
    base = rng.standard_normal(dim)
    base /= np.linalg.norm(base)
    off = rng.standard_normal(dim)
    off -= (off @ base) * base
    off /= np.linalg.norm(off)
    third = rng.standard_normal(dim)
    third -= (third @ base) * base
    third /= np.linalg.norm(third)
    seed = base * 5.0
    strong = [0.999 * base + 0.0447 * off, 0.998 * base + 0.0632 * off, third]
    keep = 0.4
    before = [float(v @ (seed / np.linalg.norm(seed)) / np.linalg.norm(v)) for v in strong]
    for trial in range(6):
        weak = list(rng.standard_normal((4, dim)))
        out = W.warp_seed(seed, strong, weak, W.WarpKnobs(cap=0.35,
                                                          gap_keep_ratio=keep))
        after = [float(v @ (out / np.linalg.norm(out)) / np.linalg.norm(v)) for v in strong]
        assert np.linalg.norm(out - seed) <= 0.35 * np.linalg.norm(seed) + 1e-9
        for i in range(len(strong)):
            for j in range(len(strong)):
                g = before[i] - before[j]
                if g > 1e-9:
                    assert after[i] - after[j] >= keep * g - 1e-12, (
                        f"trial {trial}: neighbour gap {g:.6f} fell to "
                        f"{after[i] - after[j]:.6f}, below {keep} of it")


def test_no_strong_neighbours_leaves_the_cap_untouched():
    """Fewer than two substitutes means there is no gap to preserve (clause: cap)."""
    rng = np.random.default_rng(31)
    dim = 20
    seed_u = rng.standard_normal(dim)
    seed_u /= np.linalg.norm(seed_u)
    for rows in ([], [rng.standard_normal(dim)]):
        assert W._gap_preserving_amplitude(seed_u, W._unit_rows(rows, dim),
                                           rng.standard_normal(dim), 0.35, 0.4) == 0.35


# ── clause 4: the control arm ───────────────────────────────────────────

def test_the_control_arm_is_reachable_through_the_same_switch_as_the_warp():
    rng = np.random.default_rng(41)
    seed = rng.standard_normal(12)
    strong = list(rng.standard_normal((2, 12)))
    weak = list(rng.standard_normal((3, 12)))
    assert W.ARMS == ("off", "warp", "control")
    assert W.DEFAULT_ARM == "off"
    assert np.array_equal(W.apply_arm("off", seed, strong, weak), seed)
    warp_out = W.apply_arm("warp", seed, strong, weak, key="entity-a")
    ctrl_out = W.apply_arm("control", seed, strong, weak, key="entity-a")
    assert np.array_equal(warp_out, W.warp_seed(seed, strong, weak))
    assert np.array_equal(ctrl_out, W.control_seed(seed, strong, weak, key="entity-a"))
    assert not np.array_equal(warp_out, ctrl_out)
    with pytest.raises(ValueError):
        W.apply_arm("warpd", seed, strong, weak)


@pytest.mark.parametrize("idx", range(len(CASES)))
def test_the_control_residual_is_a_unit_direction_at_the_same_cap(idx):
    seed, strong, weak = CASES[idx]
    knobs = W.WARP_KNOBS
    out = W.apply_arm("control", seed, strong, weak, knobs, key=f"key-{idx}")
    residual = out - np.asarray(seed, dtype=np.float64).reshape(-1)
    if np.linalg.norm(seed) <= 1e-9 or not weak:
        assert np.array_equal(out, np.asarray(seed, dtype=np.float64).reshape(-1))
        return
    n = float(np.linalg.norm(seed))
    cos = residual @ seed / (np.linalg.norm(residual) * n)
    assert abs(cos) < 1e-9
    # the control direction is unconstrained by neighbour geometry, so it lands
    # on the cap wherever the gap constraint does not bite: unit vector, cap length
    assert np.linalg.norm(residual) <= knobs.cap * n + 1e-9


def test_the_control_is_reproducible_for_a_seed_key_and_changes_with_another():
    rng = np.random.default_rng(51)
    seed = rng.standard_normal(24)
    weak = list(rng.standard_normal((4, 24)))
    a1 = W.control_seed(seed, [], weak, key="qmd:knowledge/foo.md#3")
    a2 = W.control_seed(seed, [], weak, key="qmd:knowledge/foo.md#3")
    b = W.control_seed(seed, [], weak, key="qmd:knowledge/foo.md#4")
    assert np.array_equal(a1, a2)
    assert not np.allclose(a1, b)
    assert not np.allclose(a1, seed)


def test_the_control_skips_the_neighbour_projection():
    """Only the WEAK VECTORS' own geometry may not reach the control answer.

    The warp's residual is an average over the weak set, so replacing those
    vectors changes it; the control's comes from the seed key, so the same
    substitution must leave the control exactly where it was.
    """
    rng = np.random.default_rng(61)
    seed = rng.standard_normal(30)
    strong = list(rng.standard_normal((3, 30)))
    weak_a = list(rng.standard_normal((4, 30)))
    weak_b = list(rng.standard_normal((5, 30)))
    assert not np.allclose(W.warp_seed(seed, strong, weak_a),
                           W.warp_seed(seed, strong, weak_b))
    assert np.array_equal(W.control_seed(seed, strong, weak_a, key="k"),
                          W.control_seed(seed, strong, weak_b, key="k"))
    # but an EMPTY weak set still declines to move the seed, so the two arms
    # touch the same set of vectors and differ only in the direction's source
    assert np.array_equal(W.control_seed(seed, strong, [], key="k"), seed)


def test_both_arms_move_the_ranking_whereas_a_parallel_perturbation_would_not():
    """The reason the residual is orthonormal: cosine ignores length.

    A perturbation along the seed leaves cosine at 1.0 and the ranking
    untouched; either arm of this switch must not do that.
    """
    rng = np.random.default_rng(71)
    seed = rng.standard_normal(16)
    weak = list(rng.standard_normal((4, 16)))
    for arm in ("warp", "control"):
        out = W.apply_arm(arm, seed, [], weak, key="k")
        cos = out @ seed / (np.linalg.norm(out) * np.linalg.norm(seed))
        assert cos < 1.0 - 1e-6, f"{arm} changed only the length"
    along = seed * 1.35
    cos = along @ seed / (np.linalg.norm(along) * np.linalg.norm(seed))
    assert cos > 1.0 - 1e-12, "the premise of the test itself is wrong"


def test_the_weak_set_is_edge_type_plus_a_measured_band_and_never_a_percentile():
    """Clause 3's second half, and the reason a percentile splitter is not in this file.

    MAGE ships percentile tails (strong 95, weak 45 above a 0.20 floor), and the
    previous revision of `complement_warp.py` carried that splitter; #3642's
    acceptance replaced it with a weak set defined by EDGE TYPE plus a MEASURED COSINE
    BAND. The distinction is not cosmetic: a percentile of a distribution whose mean
    pairwise cosine sits near 0.48 — the figure the probes print from the live index —
    is a cut that moves whenever the corpus moves, so it redefines the weak set
    implicitly on every run. Edge type is declared intent (`COMPLEMENT_TYPES`: uses,
    depends_on, part_of) and the band's floor is a constant the warp and the harvest
    share, so exactly one definition exists.

    Hence this node checks the definition from both ends: no percentile survives
    anywhere under `scripts/retrieval`, the knobs carry no percentile field for a
    caller to set, and the floor the warp applies is the same constant the harvest
    filters with.
    """
    sources = {q.name: q.read_text(encoding="utf-8")
               for q in (ROOT / "scripts" / "retrieval").glob("*.py")}
    assert sources, "no scripts/retrieval modules to read — this node proved nothing"
    for name, src in sources.items():
        # The CODE form is what is banned; the word itself appears in the comments
        # that record why the rule is gone, and a guard that reddens on prose teaches
        # the next reader to delete the explanation.
        assert "np.percentile(" not in src and "_pct" not in src, (
            f"{name} reintroduces a percentile rule: the weak set is edge type plus a "
            "measured band, and a second definition of it would be a secret one")
    assert not any("pct" in f for f in W.WARP_KNOBS.__dict__), W.WARP_KNOBS.__dict__
    H = _load("complement_harvest_probe")
    assert W.WARP_KNOBS.min_weak_cos == H.MIN_COMPLEMENT_COS, (
        "the warp's weak floor and the harvest's band floor have parted, so a pair the "
        "harvest calls usable can be a neighbour the warp silently drops")
    assert set(H.COMPLEMENT_TYPES) == {"uses", "depends_on", "part_of"}, H.COMPLEMENT_TYPES

def test_a_stacked_array_of_neighbours_is_the_same_input_as_the_same_list(tmp_path):
    """The probe hands `apply_arm` a stacked ``(k, dim)`` array; the purity must hold there.

    Clause 3 says the warp is a pure function of ``(seed, strong, weak, knobs)``. The
    caller in ``scripts/retrieval/complement_probe.py`` builds its strong set with
    ``np.asarray(rows)``, so if ``_unit_rows`` iterated an ndarray's rows as scalars
    the arm the probe reports would quietly not be the arm this file proves. Same
    numbers either spelling is the clause; a bare ``(dim,)`` vector is ONE neighbour,
    not ``dim`` of them, which is the other way that seam goes wrong.
    """
    rng = np.random.default_rng(11)
    seed = np.array([1.0, 0.0, 0.0, 0.0])
    weak_rows = [rng.normal(size=4) for _ in range(3)]
    strong_rows = [[1.0, 0.0, 0.0, 0.0], [0.7, 0.7, 0.0, 0.0]]

    list_out = W.warp_seed(seed, strong_rows, weak_rows)
    array_out = W.warp_seed(seed, np.asarray(strong_rows, dtype=float),
                               np.asarray(weak_rows, dtype=float))
    assert np.array_equal(list_out, array_out), \
        "the shape the probe passes is not the shape the proof covers"

    # A single vector, passed as (dim,) and as (1, dim), is one neighbour.
    one = rng.normal(size=4)
    assert np.array_equal(W.warp_seed(seed, [one], [one]),
                          W.warp_seed(seed, one, one)), \
        "a (dim,) vector was read as dim neighbours"

    # The invariants clause 3 names still hold on the array path.
    nrm = float(np.linalg.norm(seed))
    u_hat = seed / nrm
    residual = array_out - seed
    assert abs(float(residual @ u_hat)) < 1e-9, float(residual @ u_hat)
    assert float(np.linalg.norm(residual)) <= W.WARP_KNOBS.cap * nrm + 1e-9
    assert np.array_equal(W.warp_seed(seed, np.zeros((0, 4)), np.zeros((0, 4))), seed)
