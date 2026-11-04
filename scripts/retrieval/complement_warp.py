#!/usr/bin/env python3
"""MAGE Stage-1 complement-aware warp, as a pure function (#3642).

MAGE's diagnosis (Discover AI 2026-10-31, "Your Embeddings Are Haunted";
github.com/Alex-bupt/mage): a space built by encoding items and comparing them
with cosine encodes SUBSTITUTES exhaustively and COMPLEMENTS not at all. Every
phone lands near every phone and the charger sits on a different island, so a
vector index over that space answers "more like this" and structurally cannot
answer "what goes with this".

Stage 1 fixes the geometry offline, per item, with no training:

1. split an item's neighbours by cosine into a strong set (upper tail:
   substitutes) and a weak set (lower tail, above a hard floor: complements);
2. SVD the stack of [seed; strong set] — its right singular vectors span the
   substitute subspace;
3. apply a SOFT conceptor ``P`` over those vectors (a shrinkage of each
   direction by its own support, not a hard rank cut) and project the
   subspace out of every weak neighbour with ``I - P``;
4. average the remainders and Gram-Schmidt the result against the
   unit-normalised seed, so the residual is strictly OFF-AXIS;
5. reinject it at a capped amplitude under a local gap-preserving constraint,
   where the cap is a ceiling on an amount proportional to the surviving
   complement evidence rather than a fixed step every seed takes.

Step 4 is not decoration. Under cosine, adding information *along* the seed's
own direction changes only its length, which cosine ignores, so nothing in the
ranking moves; an orthonormal residual is the only perturbation that registers.

Everything here is a pure function of ``(seed, strong set, weak set, knobs)``
plus, for the control arm, a seed key. It reads no database, no config and no
clock, which is what makes the three arms testable side by side and what keeps
the warp DOCUMENT-SIDE: it is applied to stored vectors offline and never to a
query at request time. That last property is the whole reason #577's arm cost
3.5x latency and this one must not — see the item's step 2.

Arms (``ARM_*``, same switch for the neighbour residual and the control):

``off``
    the stored vector, untouched. The default, and nothing in the tree turns
    it on: the paired nightly run decides keep-versus-archive (#3642).
``warp``
    the neighbour-evidence residual — steps 1-5 above.
``control``
    a random unit direction at the same cap, with the neighbour projection
    skipped. The paper concedes its residuals "are not guaranteed to contain
    only complementary semantics" and ships no ablation separating "the
    residual helped" from "any orthogonal perturbation at that cap helped", so
    this arm, not the warp-off baseline, is the acceptance comparison. The two
    arms differ ONLY in how the residual was produced: same seeds (a seed with
    no weak neighbour is left alone in both), same orthogonalisation, same
    gap-preserving amplitude limit, same cap.

Numeric choices that are ours, not the repo's, and why:

* ``aperture`` is applied as a ratio against the top singular value rather than
  as an absolute radius. The repo ships 12, calibrated for raw embeddings whose
  singular values are O(10^2) (measured on the live qmd store: vector norms
  76-120); this module unit-normalises every row before the SVD, where a bare
  12 would sit above every singular value and project nothing out at all. The
  ratio reading keeps the shipped number meaningful on either scale.
* the local gap-preserving constraint is stated on the neighbourhood's own
  cosine *gaps* (what the ranking is made of), not on each neighbour's distance
  to the seed. Stated the other way a near-duplicate substitute — cos 0.999 to
  its seed is ordinary in this store, see ``supersedes`` — would leave a gap of
  0.001 and forbid the warp outright.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

#: The three arms, and the switch that selects between them.
ARM_OFF = "off"
ARM_WARP = "warp"
ARM_CONTROL = "control"
ARMS = (ARM_OFF, ARM_WARP, ARM_CONTROL)
#: Default-off: the warp is measured before it is shipped (#3642 acceptance).
DEFAULT_ARM = ARM_OFF

#: Below this a candidate residual is treated as no direction at all: a zero or
#: ragged neighbour carries nothing, and a zero-length seed cannot be warped.
_EPS = 1e-9
#: A Gram-Schmidt'd remainder no longer than this is not a direction at all
#: (weak rows are unit-normalised first, so the number is scale-free). This is a
#: zero guard, not a noise threshold: the thing that keeps a neighbourhood with
#: no off-axis evidence from being shoved sideways is the evidence-proportional
#: amplitude in :func:`_inject`, not a tuned cutoff — a near-duplicate
#: substitute makes the SVD stack ill-conditioned and leaves conditioning slop
#: along the seed axis of order 1e-6, and that slop now buys an injection 1e-6
#: of the seed instead of being renormalised up to the full cap.
_MIN_OFF_AXIS = 1e-12


@dataclass(frozen=True)
class WarpKnobs:
    """The repo-shipped Stage-1 values that reach the warp itself.

    Field values are the ones MAGE ships (item: "aperture 12, injection cap
    0.35, gap-keep-ratio 0.2-0.4, minimum weak similarity 0.20 (hard floor)"),
    with ``gap_keep_ratio`` at the loose end of the shipped range because a
    loose value lets MORE of the neighbourhood gap be consumed: keeping 0.4 of
    every gap permits dropping 0.6 of it.
    """

    #: Hard floor: a neighbour below this is noise, not a complement.
    min_weak_cos: float = 0.20
    #: Softness of the conceptor cut, as a ratio against the top singular value.
    aperture: float = 12.0
    #: Injected residual length, as a fraction of the seed's own norm.
    cap: float = 0.35
    #: Minimum share of every local cosine gap the warp is allowed to keep.
    gap_keep_ratio: float = 0.4


#: The knobs the probes use unless a flag overrides them.
WARP_KNOBS = WarpKnobs()


def _as_vector(vec) -> np.ndarray:
    return np.asarray(vec, dtype=np.float64).reshape(-1)


def _norm(v: np.ndarray) -> float:
    n = float(np.linalg.norm(v))
    return n if math.isfinite(n) else 0.0


def _unit_rows(vecs, dim: int) -> np.ndarray:
    """Unit-normalise each neighbour, dropping ragged/zero/non-finite rows.

    Row count is capped at 256 per set: the conceptor's rank is what it is
    bounded by, and a god-node seed with thousands of neighbours would
    otherwise pay for a stack the residual cannot use.

    Accepts EITHER shape a caller naturally has: a list of vectors (possibly with
    ragged or None entries, which are dropped), or one stacked ``(k, dim)`` array —
    which is what ``scripts/retrieval/complement_probe.py`` builds for a seed's
    strong set, so the pure function must not require the list form. A bare
    ``(dim,)`` array is one neighbour, not ``dim`` of them.
    """
    if vecs is None:
        iterable: list = []
    elif isinstance(vecs, np.ndarray):
        iterable = [] if vecs.size == 0 else list(np.atleast_2d(vecs))
    else:
        iterable = list(vecs)
    rows = []
    for v in iterable:
        a = _as_vector(v)
        if a.size != dim:
            continue
        n = _norm(a)
        if n <= _EPS:
            continue
        rows.append(a / n)
        if len(rows) >= 256:
            break
    return np.asarray(rows, dtype=np.float64).reshape(-1, dim)


# No percentile splitter lives here, and that is a decision rather than an omission.
# MAGE ships one — neighbours ranked by cosine, strong the 95th percentile and weak the
# 45th above a 0.20 floor — and an earlier revision of this file carried it. #3642's
# acceptance replaced it with a WEAK SET DEFINED BY EDGE TYPE PLUS A MEASURED COSINE
# BAND, and the two definitions are not interchangeable on this corpus: a percentile of
# a vector distribution whose mean pairwise cosine is ~0.48 (the figure the probes
# print from the live index) has no fixed meaning, so it silently redefines the weak
# set every time the corpus moves, while edge type is declared intent and the band is
# measured against that printed distribution. The rule that selects pairs now lives in
# the one place that builds them — complement_harvest_probe.COMPLEMENT_TYPES with
# MIN/MAX_COMPLEMENT_COS — and `warp_seed` takes the sets it is handed. A percentile
# route re-added here would be a second, secret definition of the weak set; the node
# `test_the_weak_set_is_edge_type_plus_a_measured_band_not_a_percentile` keeps it out.


def _conceptor_removal(seed_u: np.ndarray, strong_u: np.ndarray,
                       weak_u: np.ndarray, aperture: float) -> np.ndarray:
    """Mean of ``(I - P) w`` over the weak rows, orthogonalised against the seed.

    ``P`` is the soft conceptor over the right singular vectors of the stack
    ``[seed; strong set]``: direction ``i`` is removed in proportion
    ``s_i^2 / (s_i^2 + (s_max/aperture)^2)``, so a well-supported substitute
    direction is projected out almost completely and a thinly-supported one
    barely at all. That is the "diagonal operator over those vectors, not a
    hard rank cut" the method asks for, and it is applied as two
    matrix-products rather than a dense d x d matrix.

    The knob runs from no cut to a hard one: ``aperture -> 0`` leaves ``P`` at
    zero, so the weak neighbours are averaged unprojected (the conceptor off,
    which still moves the seed but on unfiltered evidence); a large aperture
    drives every weight to 1, which is the hard full-rank cut. The shipped 12
    sits near the hard end on a small neighbour stack — where the stack has few
    directions to argue about — and does the soft work on a real one.

    Returns an UNNORMALISED off-axis direction (possibly the zero vector).
    """
    stack = np.vstack((seed_u.reshape(1, -1), strong_u))
    _, sing, vt = np.linalg.svd(stack, full_matrices=False)
    if sing.size == 0:
        return np.zeros_like(seed_u)
    s2 = sing ** 2
    if aperture > 0:
        thr = (float(sing[0]) / aperture) ** 2
        weights = s2 / (s2 + thr) if thr > 0 else np.ones_like(s2)
    else:
        weights = np.zeros_like(s2)
    basis = vt.T                                  # (dim, k), the right vectors
    coords = weak_u @ basis                       # (n, k)
    remainders = weak_u - (coords * weights) @ basis.T   # (I - P) w, row-wise
    r = remainders.mean(axis=0)
    return r - float(r @ seed_u) * seed_u         # Gram-Schmidt: off-axis


def _gap_preserving_amplitude(seed_u: np.ndarray, strong_u: np.ndarray,
                              res_u: np.ndarray, cap: float,
                              keep: float) -> float:
    """Largest amplitude ``a`` in ``[0, cap]`` that keeps >= ``keep`` of every
    cosine gap inside the seed's own neighbourhood.

    With the warped unit vector ``(seed_u + a*res_u) / sqrt(1+a^2)`` (res_u is
    unit and orthogonal to seed_u), the gap between neighbours i and j becomes
    ``(g + a*dk) / sqrt(1+a^2)`` where ``g`` and ``dk`` are their original
    cosine gap and residual-gap. Requiring that to stay >= ``keep*g`` for every
    pair is what "the item's existing neighbourhood is barely disturbed" means
    operationally: a reordering would need a negative gap, so the ranking
    cannot flip while the constraint holds.

    Each pair's feasible amplitudes form one interval starting at 0 — the
    constraint function rises then falls monotonically, so it can only cross
    zero once above its peak — hence an intersection that is one interval and a
    plain bisection for its top end. Fewer than two strong neighbours means no
    gap exists to preserve, so ``cap`` applies unchanged.
    """
    cap = max(0.0, float(cap))
    if cap == 0.0 or strong_u.shape[0] < 2:
        return cap
    keep = min(1.0, max(0.0, float(keep)))
    c = strong_u @ seed_u
    k = strong_u @ res_u
    pairs = [(float(c[i] - c[j]), float(k[i] - k[j]))
             for i in range(c.size) for j in range(c.size)
             if c[i] - c[j] > _EPS]
    if not pairs:
        return cap

    def holds(a: float) -> bool:
        den = math.sqrt(1.0 + a * a)
        return all((g + a * dk) / den >= keep * g for g, dk in pairs)

    if holds(cap):
        return cap
    lo, hi = 0.0, cap
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if holds(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _random_unit(key: str, dim: int) -> np.ndarray:
    """The control arm's direction: deterministic in ``key``, not in anything else."""
    digest = hashlib.sha256(f"complement-warp-control:{key}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return rng.standard_normal(dim)


def _inject(seed: np.ndarray, direction: np.ndarray, strong_u: np.ndarray,
            knobs: WarpKnobs) -> np.ndarray:
    """Gram-Schmidt ``direction`` off the seed, then inject it at an
    evidence-proportional, cap-bounded, gap-preserving amplitude.

    The amplitude asked for is ``min(||direction_perp||, knobs.cap)``. The
    numerator is the surviving complement evidence: the mean projected weak
    remainder, whose weak rows are unit vectors, so it reads as "how much
    off-axis signal per neighbour this seed actually has". MAGE's "reinject a
    capped amount" is therefore a CEILING on an evidence-proportional amount,
    not a fixed one — normalising the direction and injecting at the cap every
    time would make a seed with a thin, nearly-explained weak set move exactly
    as far as one with a rich complement neighbourhood, which is the opposite
    of what the evidence says. It also makes the control arm's
    "random unit vector at the same cap" literal: a unit direction has evidence
    1.0, so the control asks for the cap outright.

    Returns the seed unchanged when there is no usable direction.
    """
    ns = _norm(seed)
    if ns <= _EPS:
        return seed.copy()
    seed_u = seed / ns
    r = direction - float(direction @ seed_u) * seed_u
    evidence = _norm(r)
    if evidence <= _MIN_OFF_AXIS:
        return seed.copy()
    res_u = r / evidence
    amp = _gap_preserving_amplitude(seed_u, strong_u, res_u,
                                    min(evidence, knobs.cap),
                                    knobs.gap_keep_ratio)
    if amp <= 0.0:
        return seed.copy()
    return seed + (amp * ns) * res_u


def warp_seed(seed, strong, weak, knobs: WarpKnobs = WARP_KNOBS) -> np.ndarray:
    """The Stage-1 warped seed.

    Guarantees, for every input (each pinned by ``tests/test_complement_warp.py``):

    * the injected residual ``warp_seed(...) - seed`` is orthogonal to the seed,
      because it is Gram-Schmidt'd against the unit seed before it is added;
    * ``||warp_seed(...) - seed|| <= knobs.cap * ||seed||``;
    * an empty weak set returns the seed unchanged — there is no complement
      evidence to inject, and a warp that moved the seed on no evidence would
      be the noise arm, not this one.
    """
    s = _as_vector(seed)
    dim = s.size
    knobs = knobs or WARP_KNOBS
    weak_u = _unit_rows(weak, dim)
    if weak_u.shape[0] == 0:
        return s.copy()
    ns = _norm(s)
    if ns <= _EPS:
        return s.copy()
    strong_u = _unit_rows(strong, dim)
    direction = _conceptor_removal(s / ns, strong_u, weak_u, knobs.aperture)
    return _inject(s, direction, strong_u, knobs)


def control_seed(seed, strong, weak, knobs: WarpKnobs = WARP_KNOBS,
                 key: str = "") -> np.ndarray:
    """The control arm: a random unit direction at the same cap.

    The neighbour projection is skipped — no SVD, no conceptor, no weak-set
    average — and nothing else differs from :func:`warp_seed`: the same seeds
    are touched (an empty weak set still returns the seed unchanged, so the two
    arms perturb the same set of vectors and differ only in where the direction
    came from), the same Gram-Schmidt, the same gap-preserving limit, the same
    cap. Deterministic for a given ``key``.
    """
    s = _as_vector(seed)
    dim = s.size
    knobs = knobs or WARP_KNOBS
    weak_u = _unit_rows(weak, dim)
    if weak_u.shape[0] == 0:
        return s.copy()
    ns = _norm(s)
    if ns <= _EPS:
        return s.copy()
    strong_u = _unit_rows(strong, dim)
    return _inject(s, _random_unit(str(key), dim), strong_u, knobs)


def apply_arm(arm: str, seed, strong, weak,
              knobs: WarpKnobs = WARP_KNOBS, key: str = "") -> np.ndarray:
    """One vector, under one arm of the switch.

    ``off`` is the stored vector; ``warp`` the neighbour residual; ``control``
    the random direction. An unrecognised arm is a caller bug and raises — a
    typo that silently measured the unwarped space would read as a null result.
    """
    if arm == ARM_OFF:
        return _as_vector(seed)
    if arm == ARM_WARP:
        return warp_seed(seed, strong, weak, knobs)
    if arm == ARM_CONTROL:
        return control_seed(seed, strong, weak, knobs, key)
    raise ValueError(f"unknown arm {arm!r}; the arms are {ARMS}")
