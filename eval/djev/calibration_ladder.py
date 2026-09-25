#!/usr/bin/env python3
"""The calibration ladder a djev yes/no schema climbs before it may gate (#1479).

`architecture/djev.md` §9.4 lists what a schema needs before `gate_ready` may
flip; this is the measurement half, over a file of HUMAN labels:

  1. batch calibration  (Zhou et al., ICLR 2024, arXiv 2309.17249): subtract
     the batch-mean log-odds, which removes the template's yes/no lean;
  2. Platt scaling on a fitting split (two parameters; isotonic is marginal
     at this n);
  3. split-conformal acceptance (Conformal Cascade, arXiv 2607.25018): on a
     held-out calibration split, find the nonconformity quantile for error
     rate `alpha`; a row is ACCEPTED only when its prediction set is a single
     label, otherwise it is abstained on (routed to the primary or a human).

It reports, on a final test split, the error among accepted rows (what a gate
would get wrong), the abstain rate, and coverage. It refuses with fewer than
`MIN_LABELS` labelled rows: the method's error bound resolves at 1/(n+1), and
n >= 200 is the published floor per tier.

Input: JSONL rows carrying `djev_p_same` (a probability) and `label`
(`same` / `different`; `unsure` or null rows are skipped). The dedupe seam's
file is `~/lloyd-data/eval/djev/dedupe-labels-1479.jsonl`; it has no labels
until a person writes them.

    .venvs/lloyd/bin/python eval/djev/calibration_ladder.py --labels FILE [--alpha 0.05]

Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

MIN_LABELS = 200
EPS = 1e-6


def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def batch_calibrate(ps: list[float]) -> list[float]:
    mu = sum(logit(p) for p in ps) / len(ps)
    return [sigmoid(logit(p) - mu) for p in ps]


def fit_platt(xs: list[float], ys: list[int], iters: int = 200) -> tuple[float, float]:
    """a, b for p = sigmoid(a * logit(x) + b), by Newton on the log-loss."""
    a, b = 1.0, 0.0
    zs = [logit(x) for x in xs]
    for _ in range(iters):
        ga = gb = haa = hab = hbb = 0.0
        for z, y in zip(zs, ys):
            p = sigmoid(a * z + b)
            g = p - y
            w = p * (1 - p)
            ga += g * z; gb += g
            haa += w * z * z; hab += w * z; hbb += w
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a -= da; b -= db
        if abs(da) + abs(db) < 1e-9:
            break
    return a, b


def conformal_quantile(p_true: list[float], alpha: float) -> float:
    """Nonconformity 1 - p(true label); the finite-sample (1 - alpha) quantile."""
    s = sorted(1.0 - p for p in p_true)
    n = len(s)
    k = min(n - 1, max(0, math.ceil((n + 1) * (1 - alpha)) - 1))
    return s[k]


def prediction_set(p_same: float, q: float) -> set[str]:
    out = set()
    if 1.0 - p_same <= q:
        out.add("same")
    if p_same <= q:            # 1 - p(different) = p_same
        out.add("different")
    return out


def load(path: Path) -> list[tuple[float, int]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        lab, p = r.get("label"), r.get("djev_p_same")
        if lab not in ("same", "different") or p is None:
            continue
        rows.append((float(p), 1 if lab == "same" else 0))
    return rows


def ladder(rows: list[tuple[float, int]], *, alpha: float = 0.05, seed: int = 1479) -> dict:
    if len(rows) < MIN_LABELS:
        raise ValueError(f"{len(rows)} labelled rows; the ladder needs at least {MIN_LABELS}")
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    ps = batch_calibrate([p for p, _ in rows])
    ys = [y for _, y in rows]
    n = len(rows)
    i1, i2 = n // 3, 2 * n // 3
    a, b = fit_platt(ps[:i1], ys[:i1])
    cal = [sigmoid(a * logit(p) + b) for p in ps]
    q = conformal_quantile([c if y else 1 - c for c, y in zip(cal[i1:i2], ys[i1:i2])], alpha)
    test = list(zip(cal[i2:], ys[i2:]))
    accepted = wrong = covered = 0
    for c, y in test:
        s = prediction_set(c, q)
        truth = "same" if y else "different"
        covered += truth in s
        if len(s) == 1:
            accepted += 1
            wrong += truth not in s
    return {"n": n, "alpha": alpha, "platt": {"a": round(a, 4), "b": round(b, 4)},
            "quantile": round(q, 4), "test_n": len(test),
            "accepted": accepted, "abstained": len(test) - accepted,
            "abstain_rate": round(1 - accepted / len(test), 3) if test else None,
            "error_among_accepted": round(wrong / accepted, 3) if accepted else None,
            "coverage": round(covered / len(test), 3) if test else None,
            "gate_ready_candidate": bool(accepted and wrong / accepted <= alpha)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1479)
    args = ap.parse_args(argv)
    rows = load(args.labels)
    try:
        rep = ladder(rows, alpha=args.alpha, seed=args.seed)
    except ValueError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
