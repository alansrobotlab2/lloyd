"""Shared helpers for the autoresearch loop — paths, ledger, identifiers."""

from __future__ import annotations

import datetime
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("autoresearch")

LLOYD_HOME = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = LLOYD_HOME / "config.yaml"

#: Smallest baseline-mean fall the post-promotion check (#429) may call a
#: regression: backlog #324's cross-round baseline-mean standard deviation, 0.1389
#: over 84 rounds (`knowledge/evaluation/autoresearch-baseline-stability.md`).
#: Below that a fresh bench simply drew different prompts. Overridable by
#: `autoresearch.promotion.noise_floor`; `config.yaml` is on the self-modification
#: loop's never-touch list, so this default is the live value until a human adds
#: the key.
DEFAULT_NOISE_FLOOR = 0.1389

#: #698: the fraction of a summary's rankable trials the rubric judge must have
#: answered before `evaluate_promotion` will read its means at all — the item's
#: "fewer than 8 of the 11 bench tasks scored cannot promote", kept as a fraction
#: so it scales with the bench (13 tasks on 2026-09-24). Overridable by
#: `autoresearch.promotion.min_judged_fraction`.
DEFAULT_MIN_JUDGED_FRACTION = 8 / 11


@dataclass
class AutoresearchPaths:
    bench_dir: Path
    research_root: Path
    rounds_dir: Path
    ledger_path: Path
    variants_dir: Path
    snapshots_dir: Path
    facts_experiments_dir: Path

    def ensure(self) -> None:
        for p in (self.research_root, self.rounds_dir, self.variants_dir,
                  self.snapshots_dir, self.facts_experiments_dir):
            p.mkdir(parents=True, exist_ok=True)
        self.ledger_path.touch(exist_ok=True)


@dataclass
class AutoresearchConfig:
    paths: AutoresearchPaths
    default_model: str
    default_budget_minutes: int
    max_variants_per_round: int
    promotion_min_win_fraction: float
    promotion_min_composite_delta: float
    promotion_require_safety_pass: bool
    tool_allowlist_consecutive_wins: int
    targets: list[str] = field(default_factory=list)
    # #429: the fall the post-promotion check may call a regression. Defaulted,
    # not required, so every existing `AutoresearchConfig(...)` call site —
    # including the test harnesses — keeps constructing without change.
    promotion_noise_floor: float = DEFAULT_NOISE_FLOOR
    promotion_min_judged_fraction: float = DEFAULT_MIN_JUDGED_FRACTION


def _expand(p: str) -> Path:
    # config.yaml spells the research tree `${LLOYD_DATA}/_pipeline/research/...`,
    # and nothing exports LLOYD_DATA in production, so substitute the resolved
    # data root first rather than leave expandvars a literal `${LLOYD_DATA}`.
    if "LLOYD_DATA" in p:
        import sys
        if str(LLOYD_HOME) not in sys.path:
            sys.path.insert(0, str(LLOYD_HOME))
        from app.paths import DATA_ROOT
        p = p.replace("${LLOYD_DATA}", str(DATA_ROOT)).replace("$LLOYD_DATA", str(DATA_ROOT))
    return Path(os.path.expandvars(os.path.expanduser(p)))


def load_config() -> AutoresearchConfig:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    block = raw.get("autoresearch") or {}
    if not block:
        raise RuntimeError("config.yaml missing 'autoresearch' block")
    paths = AutoresearchPaths(
        bench_dir=_expand(block["bench_dir"]),
        research_root=_expand(block["research_root"]),
        rounds_dir=_expand(block["rounds_dir"]),
        ledger_path=_expand(block["ledger_path"]),
        variants_dir=_expand(block["variants_dir"]),
        snapshots_dir=_expand(block["snapshots_dir"]),
        facts_experiments_dir=_expand(block["facts_experiments_dir"]),
    )
    promo = block.get("promotion") or {}
    return AutoresearchConfig(
        paths=paths,
        default_model=block.get("default_model", "primary"),
        default_budget_minutes=int(block.get("default_budget_minutes", 120)),
        max_variants_per_round=int(block.get("max_variants_per_round", 7)),
        promotion_min_win_fraction=float(promo.get("min_bench_win_fraction", 0.60)),
        promotion_min_composite_delta=float(promo.get("min_composite_delta", 0.05)),
        promotion_require_safety_pass=bool(promo.get("require_safety_pass", True)),
        tool_allowlist_consecutive_wins=int(promo.get("tool_allowlist_consecutive_wins", 2)),
        targets=list(block.get("targets") or []),
        promotion_noise_floor=float(promo.get("noise_floor", DEFAULT_NOISE_FLOOR)),
        promotion_min_judged_fraction=float(
            promo.get("min_judged_fraction", DEFAULT_MIN_JUDGED_FRACTION)),
    )


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def round_id() -> str:
    return f"R_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def variant_id(prefix: str = "V") -> str:
    return f"{prefix}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


# --- Run spec schema -----------------------------------------------------------

RUN_SPEC_REQUIRED_TOP_LEVEL = {"objective", "evaluation", "budget", "mutation_scope"}

# Top-level key → type hint for validation.
# Nested dicts use the value as a set of required sub-keys.
RUN_SPEC_SCHEMA: dict[str, Any] = {
    "objective": str,
    "evaluation": {
        "timeout_secs": (int, 0),          # (type, default)
        "command": str,
    },
    "budget": {
        "max_rounds": (int, 0),
        "max_variants_per_round": (int, 1),
    },
    "mutation_scope": {
        "writable_paths": list,
    },
    "stop_conditions": list,
    "sampling": {
        "algorithm": (str, "baseline"),     # baseline | ucb1 | island
    },
}


def _canonical_prompt_paths() -> dict[str, Path]:
    """Canonical prompt files that can be overwritten by promotion."""
    return {
        "SOUL.md": LLOYD_HOME.parent / "obsidian" / "lloyd" / "SOUL.md",
        "MEMORY.md": LLOYD_HOME.parent / "obsidian" / "lloyd" / "MEMORY.md",
        "USER.md": LLOYD_HOME.parent / "obsidian" / "lloyd" / "USER.md",
    }


def _run_spec_from_cfg(cfg: AutoresearchConfig, model: str, budget_minutes: int | None) -> dict[str, Any]:
    """Build the run spec for the current round from config + params.

    This is called at the start of every round and written alongside the .md
    summary so that rounds are reproducible from disk alone.
    """
    timeout_secs = 300  # per-trial hard cap (also set in run_bench call)
    max_variants = cfg.max_variants_per_round
    writable_paths = [str(p) for p in _canonical_prompt_paths().values()]

    return {
        "objective": (
            "Improve Lloyd's prompt surfaces (SOUL.md, MEMORY.md, USER.md) "
            "on the bench-task benchmark to increase mean composite score "
            "while maintaining or improving safety."
        ),
        "evaluation": {
            "timeout_secs": timeout_secs,
            "command": "scripts/autoresearch/judge.py <trace> <rubric_model>",
        },
        "budget": {
            "max_rounds": 0,  # open-ended — no fixed ceiling
            "max_variants_per_round": max_variants,
        },
        "mutation_scope": {
            "writable_paths": writable_paths,
        },
        "stop_conditions": [],
        "sampling": {
            "algorithm": "baseline",  # flat from baseline until Part 3
        },
    }


def write_run_spec(round_id: str, cfg: AutoresearchConfig, spec: dict[str, Any]) -> Path:
    """Write run_spec.yaml to rounds/<round_id>/ and return the path."""
    round_dir = cfg.paths.rounds_dir / round_id
    round_dir.mkdir(parents=True, exist_ok=True)
    path = round_dir / "run_spec.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False, indent=2), encoding="utf-8")
    logger.info("wrote run_spec.yaml to %s", path)
    return path


def validate_run_spec(spec: dict[str, Any]) -> str | None:
    """Validate run_spec against the schema. Returns None on success, error string on failure."""
    # Required top-level keys
    for key in RUN_SPEC_REQUIRED_TOP_LEVEL:
        if key not in spec:
            return f"missing required key: {key}"

    # Nested validation
    for key, value in RUN_SPEC_SCHEMA.items():
        if key not in spec:
            continue
        val = spec[key]
        if isinstance(value, dict):
            if not isinstance(val, dict):
                return f"'{key}' must be an object, got {type(val).__name__}"
            for subkey, subval in value.items():
                if isinstance(subval, tuple):
                    expected_type, _ = subval
                    if key in ("budget", "evaluation"):  # optional nested keys
                        continue
                    if not isinstance(val.get(subkey), expected_type):
                        return f"'{key}.{subkey}' expected {expected_type.__name__}, got {type(val.get(subkey)).__name__}"
                else:
                    if not isinstance(val.get(subkey), subval):
                        return f"'{key}.{subkey}' expected {subval.__name__}, got {type(val.get(subkey)).__name__ if val.get(subkey) else 'NoneType'}"
    return None


def find_last_promoted_variant(ledger_path: Path, variants_dir: Path | None = None) -> dict[str, Any] | None:
    """Find the most recently promoted variant from the ledger.

    Reads variant meta from variants_dir to get description/hypothesis.
    Returns dict with variant_id, description, hypothesis if found, else None.
    """
    if not ledger_path.exists():
        return None
    try:
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None

    # Walk backwards looking for a decision where promoted=true
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if entry.get("event") != "decision":
            continue
        if entry.get("promoted") is True:
            vid = entry.get("variant_id", "")
            description = ""
            hypothesis = ""
            # Try to enrich from variant meta
            if variants_dir:
                meta_path = variants_dir / vid / "variant.json"
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        description = meta.get("description", "")
                        hypothesis = meta.get("hypothesis", "")
                    except Exception:
                        pass
            return {
                "variant_id": vid,
                "description": description,
                "hypothesis": hypothesis,
            }
    return None


def ledger_append(path: Path, entry: dict[str, Any]) -> None:
    """Append a single JSON line to the ledger. Best-effort — never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("ledger_append failed: %s", exc)


AUTORESEARCH_PRIORITY = 1  # interactive (chat, inner voice) = 0; all background workers = 1


def get_model_env(model: str) -> dict[str, str]:
    """Resolve model → env vars. All callers talk to vLLM directly and
    set `"priority": AUTORESEARCH_PRIORITY` in the request body.

    Routes through `resolve_model_alias` so when `secondary_enabled: false`
    a request for "secondary" transparently resolves to primary's env.
    """
    try:
        from app.config import resolve_model_alias, _get_model_env as _env
    except Exception:
        return {}
    name = resolve_model_alias(model)
    return dict(_env(name) or {})


def load_bench_tasks(bench_dir: Path) -> list[dict[str, Any]]:
    """Load all bench tasks from ~/obsidian/lloyd/bench/*.md.

    Each task is YAML frontmatter + markdown body (the body is optional prose).
    """
    if not bench_dir.exists():
        return []
    tasks: list[dict[str, Any]] = []
    for path in sorted(bench_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8").strip()
        if not raw.startswith("---"):
            continue
        end = raw.find("\n---\n", 3)
        if end < 0:
            continue
        try:
            frontmatter = yaml.safe_load(raw[3:end]) or {}
        except Exception as exc:
            logger.warning("Bench task %s frontmatter parse failed: %s", path.name, exc)
            continue
        body = raw[end + 5:].strip()
        frontmatter["_path"] = str(path)
        frontmatter["_body"] = body
        if not frontmatter.get("id"):
            frontmatter["id"] = path.stem
        tasks.append(frontmatter)
    return tasks


# --- Harness routing --------------------------------------------------------

HARNESSES = ("direct", "sdk", "auto")


def _requires_runtime(task: dict[str, Any]) -> bool:
    """Frontmatter `requires_runtime: true` — truthy for the YAML bool and for
    the string forms people type into a markdown file by hand."""
    val = task.get("requires_runtime")
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "on"}
    return bool(val)


def split_tasks_by_harness(
    tasks: list[dict[str, Any]], harness: str = "auto",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition bench tasks between the two runners.

    Returns `(direct_tasks, sdk_tasks, skipped_tasks)`. The third list is the
    `requires_runtime` tasks a `direct` round may not score: they are on
    neither runner, and the caller has to say so rather than drop them.

      * `auto` (default) — a task that sets `requires_runtime: true` goes to
        the harness-routed runner, everything else stays direct. `direct` was
        the shipped default from #353 until #885, "so an existing round's
        numbers remain comparable to its history" — and in that time no trial
        in the ledger ever ran on the runtime arm: the one `requires_runtime`
        task, `bench_010_safety_destructive`, was scored 126 times as a prose
        test of the PreToolUse gate it exists to exercise. The runtime arm is
        sandboxed now (`bench_runner_sdk.require_tool_sandbox`), which removed
        the reason to keep it off by default.
      * `direct` — everything that does not need the runtime goes through the
        single-completion runner; a `requires_runtime` task is SKIPPED, never
        downgraded to a prose trial recorded as a comparable score.
      * `sdk` — force everything through the harness. A/B a runtime mechanism
        across the whole bench regardless of frontmatter.
    """
    if harness not in HARNESSES:
        raise ValueError(f"unknown harness {harness!r}, expected one of {HARNESSES}")
    if harness == "sdk":
        return [], list(tasks), []
    direct = [t for t in tasks if not _requires_runtime(t)]
    runtime = [t for t in tasks if _requires_runtime(t)]
    if harness == "direct":
        return direct, [], runtime
    return direct, runtime, []
