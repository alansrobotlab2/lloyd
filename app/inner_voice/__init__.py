"""Inner Voice (#345) — parallel cognitive process for Lloyd.

Stage 0 ships scaffolding + event-log infrastructure only. the critic
ensembles, heuristics, and intervention dispatch land in later stages.

Module layout (planned):
    heuristics.py  — PreToolUse regex deny rules + post-loop completion
                     heuristic. Stage 1.
    critic.py      — single-persona the critic call wrapper. Stage 2.
    ensemble.py    — concurrent persona runner + aggregation. Stage 3.

This package is intentionally empty in Stage 0 — see the router at
`app/routers/inner_voice.py` and the event log at `app/event_log.py`.
"""
