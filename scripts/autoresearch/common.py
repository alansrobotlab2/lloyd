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


def _expand(p: str) -> Path:
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
    )


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def round_id() -> str:
    return f"R_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def variant_id(prefix: str = "V") -> str:
    return f"{prefix}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def ledger_append(path: Path, entry: dict[str, Any]) -> None:
    """Append a single JSON line to the ledger. Best-effort — never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("ledger_append failed: %s", exc)


AUTORESEARCH_PRIORITY = 2  # user=0, pipeline/autonomy SDK=1 (via proxy), autoresearch=2


def get_model_env(model: str, priority_proxy: bool = False) -> dict[str, str]:
    """Resolve model → env vars.

    Autoresearch talks to vLLM directly (8096/8091) and sets `"priority": 2`
    in each request body. The `priority_proxy` kwarg is kept for callers that
    want the proxy's priority=1 behavior, but autoresearch defaults to direct.
    """
    if model == "primary":
        base = "http://127.0.0.1:8097" if priority_proxy else "http://127.0.0.1:8096"
        return {
            "ANTHROPIC_BASE_URL": base,
            "ANTHROPIC_API_KEY": "no-key-required",
            "ANTHROPIC_CUSTOM_MODEL_OPTION": "primary",
            "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "Primary",
        }
    if model == "secondary":
        base = "http://127.0.0.1:8093" if priority_proxy else "http://127.0.0.1:8091"
        return {
            "ANTHROPIC_BASE_URL": base,
            "ANTHROPIC_API_KEY": "no-key-required",
            "ANTHROPIC_CUSTOM_MODEL_OPTION": "secondary",
            "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "Secondary",
        }
    return {}


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
