#!/usr/bin/env python3
"""#2254 — send the four arms to one engine, and refuse to report a partial run.

The ~200-call pass (50 samples × 4 arms, temperature 0) is a deliberate act: this
file only refuses to run, it does not run. Nothing in ``workers/``, ``app/`` or
``scripts/`` imports it, and the endpoint is resolved through
``app.secondary_models._endpoint(JOB)`` rather than a hard-coded URL — because the
hard-coded one in ``eval/durable_write_judge/judge.py:33`` is port 8091, the
secondary slot that has been switched off since 2026-09-20 (``config.yaml:37``, GPU 2
reassigned to the ``djev:`` process), which is exactly why the published 32.4 % is
not this probe's comparator. Routing through the job map also means that if a
cross-family engine is ever registered for a job, this probe follows it and logs the
alias rewrite (#1445) instead of silently staying in-family.

The one rule that keeps the numbers honest: **a run with any unjudged row writes no
report.** A sample whose reply could not be parsed, or whose call errored, would
otherwise drop out of every denominator quietly — the same silent-denominator class
#580's `unparsed` rule exists to stop. The raw rows are still written first, because
the evidence of a failed run is what you need to fix it; only the percentages are
withheld, since those are the part a partial run falsifies.

Calls are sequential on purpose. The `repeat` arm is this engine's noise floor, and
that floor is only expected to be zero at temperature 0 on one concurrent request;
batching it would fold contention into the number that every other rate is measured
against. If the floor is non-sequential-signal (any non-zero repeat flip rate), read
every `self`/`human` delta below it as an upper bound, not a bias measurement.

Usage (post-landing, one decision away, on the engine that actually answers):

    cd ~/lloyd && .venvs/lloyd/bin/python eval/judge_self_preference/run_arms.py \
        --raw eval/judge_self_preference/raw.jsonl \
        --report eval/judge_self_preference/report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE.parents[1])) if str(HERE.parents[1]) not in sys.path else None

from eval.judge_self_preference import arms as A
from eval.judge_self_preference import report as R
from eval.durable_write_judge.judge import call_engine, parse_verdict

#: The job name the endpoint is resolved under. Named after the probe so the job map
#: shows what asked for an engine, and so a future cross-family assignment is a
#: config line against this name rather than an edit to this file.
JOB = "judge_self_preference"


@dataclass(frozen=True)
class EngineChoice:
    """What the resolver was *asked* for, and what it actually handed back."""

    url: str
    model: str
    source: str
    asked_engine: str
    served_engine: str
    rewritten: bool

    def describe(self) -> str:
        # Request and answer on one line, never folded into one word like `engine`:
        # #1445's failure was a run that read as "routed" while a policy override
        # decided the model somewhere else.
        return (f"{self.source} | asked_engine={self.asked_engine} "
                f"served_engine={self.served_engine} rewritten={self.rewritten} "
                f"url={self.url} model={self.model}")


def resolve_endpoint(url: str | None, model: str | None) -> EngineChoice:
    """Where the calls go, with the engine's identity attached rather than assumed.

    Not ``judge.py``'s ``ENGINE_URL``, and not a hand-picked
    ``config.models.engines.secondary``: :8091 has answered nothing since 2026-09-20
    (``config.yaml:37``, GPU 2 went to the ``djev:`` slot), so either one would send
    ~200 calls into a dead port. ``_endpoint()`` is the only place allowed to answer
    "which engine runs job X".

    What it actually does with a job it has never heard of — which is what this probe
    is, since ``JOBS_ON_PRIMARY`` may not be filled in by hand
    (``app/secondary_models.py:64-70``) and this job appears in no ``decisions.yaml`` —
    is default it to ``secondary`` at ``app/secondary_models.py:82``, after which
    ``config.resolve_model_alias`` rewrites ``secondary`` to ``primary`` because
    ``secondary_enabled`` is false (``app/config.py:192-199``, the #1445 rule). So the
    answer is primary today *through that override*, not through any default naming
    primary, and ``tests/test_judge_self_preference.py`` pins precisely that: register
    this job as ``keep_secondary``, or re-enable the secondary beside ``djev``, and the
    same call starts returning :8091 — a run silently measuring a different judge than
    its own ``none`` arm, which is the confound this probe exists to avoid.
    """
    if url:
        chosen = model or "primary"
        return EngineChoice(url, chosen, "explicit --url", "explicit",
                            chosen.rsplit("/", 1)[-1], False)
    from app import secondary_models as SM      # lazy: no app import at module load
    asked = SM._engine_for(JOB)
    resolved_url, resolved_model = SM._endpoint(JOB)
    served = (model or resolved_model).rsplit("/", 1)[-1]
    return EngineChoice(resolved_url, model or resolved_model,
                        f"job {JOB!r} via _endpoint", asked, served, asked != served)


def engine_call(prompt_text: str, url: str, model: str,
                temperature: int = A.TEMPERATURE) -> tuple[str, str]:
    """One judging call: #580's own caller, signature and all.

    ``judge.call_engine`` is imported, not copied, because it is the thing that sends
    ``chat_template_kwargs: {"enable_thinking": False}`` — measured, not cosmetic
    (``eval/durable_write_judge/judge.py:179``): with thinking on, replies carry no
    JSON object the parser can find and every row comes back `unparsed`. A second copy
    of that request body would be a second thing to keep correct, and getting it wrong
    here reads as a whole run of unjudged rows rather than as a bug.
    """
    return call_engine(prompt_text, url=url, model=model, temperature=temperature)


def judge_one(row: dict, call, url: str, model: str) -> dict:
    """One arm of one sample: the prompt's bytes in, a verdict row out."""
    try:
        content, finish = call(row["text"], url, model)
    except Exception as exc:                       # noqa: BLE001 — a failed call is data
        return {**_identity(row), "verdict": "error", "defect_class": None,
                "reason": f"{type(exc).__name__}: {str(exc)[:180]}",
                "model": model, "url": url}
    parsed = parse_verdict(content)
    return {**_identity(row), "verdict": parsed["verdict"],
            "defect_class": parsed["defect_class"],
            "reason": parsed["reason"], "model": model, "url": url,
            "finish_reason": finish}


def _identity(row: dict) -> dict:
    return {"sample_id": row["sample_id"], "arm": row["arm"],
            "attribution": row["attribution"], "prompt_sha256": row["prompt_sha256"],
            "prompt_chars": row["prompt_chars"]}


def unjudged_rows(rows: list[dict]) -> list[dict]:
    """Rows whose verdict cannot be counted — the run must stop at these."""
    return [r for r in rows if r["verdict"] not in R.JUDGED]


def main(argv: list[str] | None = None, call=engine_call) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default=str(A.CORPUS))
    ap.add_argument("--raw", default=str(HERE / "raw.jsonl"))
    ap.add_argument("--report", default=str(HERE / "report.md"))
    ap.add_argument("--url", default="", help="engine to ask, default: the job map")
    ap.add_argument("--model", default="")
    ap.add_argument("--limit", type=int, default=0,
                    help="judge fewer samples (changes the example draw; never for "
                         "a published number)")
    args = ap.parse_args(argv)

    offenders = A.audit_attribution_text()
    if offenders:
        print(f"[refuse] un-audited attribution phrasing: {'; '.join(offenders)}")
        return 1

    samples = A.load_corpus(args.corpus)
    judged_samples = samples[:args.limit] if args.limit else samples
    if args.limit:
        print(f"[warn] --limit {args.limit} judges {len(judged_samples)} of "
              f"{len(samples)} samples and draws examples from the full corpus; "
              "no figure from this run may be published", file=sys.stderr)

    rows_in = A.prompt_rows(A.generate(samples))
    if args.limit:
        keep = {s["id"] for s in judged_samples}
        rows_in = [r for r in rows_in if r["sample_id"] in keep]

    choice = resolve_endpoint(args.url or None, args.model or None)
    # Printed before the first call, not only into the report: this is ~200 temp-0
    # calls, and an engine identity discoverable only afterwards is an identity that
    # cannot stop a wasted run or a mis-attributed delta.
    print(f"engine {choice.describe()} — {len(rows_in)} calls, "
          f"temperature {A.TEMPERATURE}, sequential")
    url, model = choice.url, choice.model

    rows = [judge_one(r, call, url, model) for r in rows_in]
    Path(args.raw).parent.mkdir(parents=True, exist_ok=True)
    Path(args.raw).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    bad = unjudged_rows(rows)
    if bad:
        for r in bad:
            print(f"[refuse] sample={r['sample_id']} arm={r['arm']} "
                  f"verdict={r['verdict']} reason={r['reason']}", file=sys.stderr)
        print(f"[refuse] {len(bad)} of {len(rows)} rows are not "
              f"{'/'.join(R.JUDGED)} — every rate would be computed over a smaller "
              f"denominator than it claims. Raw rows are in {args.raw}; "
              f"no report was written (would have been {args.report}).",
              file=sys.stderr)
        return 1

    text = R.render(judged_samples, rows, url=url, model=model,
                    temperature=A.TEMPERATURE,
                    engine_asked=(None if choice.source.startswith("explicit")
                                  else choice.asked_engine),
                    engine_rewritten=choice.rewritten)
    Path(args.report).write_text(text)
    print(f"wrote {len(rows)} rows -> {args.raw}")
    print(f"wrote report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
