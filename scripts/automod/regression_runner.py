"""`python -m scripts.automod.regression_runner run|pending|latest`.

The detached regression runner's entry point, and nothing else. The runner is
`workers.sources.automod_regression.main`; it is started from HERE because
`workers/sources/__init__.py` imports every source to build its registry, so
`python -m workers.sources.automod_regression` executes that module twice —
once as the registry's copy and once as `__main__` — and runpy says so in the
log of a process nobody is watching ("may result in unpredictable behaviour").
"""

from __future__ import annotations

from workers.sources import automod_regression

if __name__ == "__main__":
    raise SystemExit(automod_regression.main())
