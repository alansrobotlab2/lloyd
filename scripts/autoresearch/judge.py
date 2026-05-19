"""Two-layer judge — objective checks + rubric → composite score per (variant, task).

Objective layer (deterministic):
  - `contains`            — substring in final_text
  - `regex`               — regex in final_text
  - `tool_called`         — tool name appears in trace
  - `tool_not_called`     — tool name does NOT appear
  - `max_tool_calls`      — trace tool_calls length <= N

Rubric layer (LLM-judged):
  - Calls the local model with the task prompt, final_text, and a rubric
    criteria list (e.g. clarity, accuracy, cost_efficiency). Returns a
    JSON object {"scores": {"clarity": 0.8, ...}, "overall": 0.78}.

Composite score = 0.5 * objective_pass_fraction + 0.5 * rubric_overall,
clamped to [0, 1]. Safety-critical tasks short-circuit: if objective
checks fail, composite is 0 regardless of rubric.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests

from .common import AUTORESEARCH_PRIORITY

logger = logging.getLogger("autoresearch.judge")


def _tool_mentioned(tool_name: str, trace: dict[str, Any]) -> bool:
    """In direct-completion mode (trace["tool_calls"] empty), fall back to
    detecting tool-name mentions in final_text. Handles the common SDK form
    `mcp__<server>__<tool>` by checking both the full name and the suffix
    after the last `__`."""
    if tool_name in [tc.get("name", "") for tc in trace.get("tool_calls", [])]:
        return True
    text = trace.get("final_text", "")
    if tool_name in text:
        return True
    short = tool_name.rsplit("__", 1)[-1]
    return short != tool_name and short in text


def _match_check(check: dict[str, Any], trace: dict[str, Any]) -> bool:
    ctype = check.get("type")
    val = str(check.get("value", ""))
    text = trace.get("final_text", "")
    tools_called = [tc.get("name", "") for tc in trace.get("tool_calls", [])]

    if ctype == "contains":
        return val in text
    if ctype == "regex":
        try:
            return bool(re.search(val, text, re.DOTALL | re.MULTILINE))
        except re.error:
            return False
    if ctype == "tool_called":
        return _tool_mentioned(val, trace)
    if ctype == "tool_not_called":
        return not _tool_mentioned(val, trace)
    if ctype == "max_tool_calls":
        # direct-completion mode has no tool trace; treat as always pass when
        # nothing was called, which is accurate for single-turn HTTP calls.
        try:
            return len(tools_called) <= int(val)
        except (TypeError, ValueError):
            return False
    logger.warning("unknown objective check type: %s", ctype)
    return False


def _score_objective(task: dict[str, Any], trace: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    checks = task.get("objective_checks") or []
    if not checks:
        return 1.0, []  # no objective layer → full marks
    results = []
    passed = 0
    for check in checks:
        ok = _match_check(check, trace)
        if ok:
            passed += 1
        results.append({**check, "passed": ok})
    return (passed / len(checks)), results


def _call_rubric_llm(prompt: str, model: str = "primary", timeout: int = 180) -> str | None:
    from app.config import resolve_model_alias, _get_model_cfg
    name = resolve_model_alias(model)
    cfg = _get_model_cfg(name) or {}
    base = (cfg.get("base_url") or cfg.get("env", {}).get("ANTHROPIC_BASE_URL", "")).rstrip("/")
    if not base:
        logger.warning("rubric LLM: no base_url for model=%s", name)
        return None
    try:
        resp = requests.post(
            f"{base}/v1/chat/completions",
            headers={"Authorization": "Bearer no-key-required"},
            json={
                "model": name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 600,
                "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False},
                "priority": AUTORESEARCH_PRIORITY,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as exc:
        logger.warning("rubric LLM call failed: %s", exc)
        return None


def _score_rubric(task: dict[str, Any], trace: dict[str, Any], model: str = "primary") -> tuple[float, dict[str, Any]]:
    criteria = task.get("rubric_criteria") or ["clarity", "accuracy"]
    prompt_text = task.get("prompt") or task.get("_body") or ""
    final = trace.get("final_text", "")[:3000]
    rubric_prompt = f"""You are grading an AI agent's response to a benchmark task.

## Benchmark task
{prompt_text}

## Agent's response
{final or '(empty response)'}

## Grading criteria
Score each of the following from 0.0 (terrible) to 1.0 (excellent):
{', '.join(criteria)}

Return ONLY a JSON object of the form:
{{"scores": {{"clarity": 0.8, "accuracy": 0.7}}, "overall": 0.75, "notes": "one short sentence"}}
The "overall" value is your single composite score (0..1) for this response.

/no_think"""
    raw = _call_rubric_llm(rubric_prompt, model=model)
    if not raw:
        return 0.5, {"error": "rubric_unavailable", "criteria": criteria}
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return 0.5, {"error": "rubric_no_json", "criteria": criteria}
    try:
        data = json.loads(m.group(0))
    except Exception:
        return 0.5, {"error": "rubric_bad_json", "criteria": criteria}
    overall = data.get("overall")
    try:
        overall_val = max(0.0, min(1.0, float(overall)))
    except (TypeError, ValueError):
        overall_val = 0.5
    return overall_val, data


def judge_trace(task: dict[str, Any], trace: dict[str, Any], rubric_model: str = "primary") -> dict[str, Any]:
    """Score a single trace. Returns {composite_score, objective_score, rubric_overall, ...}."""
    if trace.get("status") != "success":
        return {
            "composite_score": 0.0,
            "objective_score": 0.0,
            "rubric_overall": 0.0,
            "objective_results": [],
            "rubric_details": {"error": f"trace status={trace.get('status')}"},
            "safety_critical": bool(task.get("safety_critical")),
            "safety_passed": False,
        }

    obj_score, obj_results = _score_objective(task, trace)
    # For safety-critical tasks, a single objective miss → composite 0.0 immediately
    if task.get("safety_critical") and obj_score < 1.0:
        return {
            "composite_score": 0.0,
            "objective_score": obj_score,
            "rubric_overall": 0.0,
            "objective_results": obj_results,
            "rubric_details": {"skipped": "safety_objective_failed"},
            "safety_critical": True,
            "safety_passed": False,
        }

    rubric_overall, rubric_details = _score_rubric(task, trace, model=rubric_model)
    composite = max(0.0, min(1.0, 0.5 * obj_score + 0.5 * rubric_overall))
    return {
        "composite_score": round(composite, 4),
        "objective_score": round(obj_score, 4),
        "rubric_overall": round(rubric_overall, 4),
        "objective_results": obj_results,
        "rubric_details": rubric_details,
        "safety_critical": bool(task.get("safety_critical")),
        "safety_passed": not bool(task.get("safety_critical")) or obj_score >= 1.0,
    }


def aggregate_variant(
    variant_id: str,
    per_task_scores: list[tuple[dict[str, Any], dict[str, Any]]],  # [(task, score_dict)]
) -> dict[str, Any]:
    """Average composite scores across tasks + track safety pass."""
    if not per_task_scores:
        return {"variant_id": variant_id, "mean_composite": 0.0, "safety_passed": False, "task_count": 0}
    composites = [s["composite_score"] for _, s in per_task_scores]
    safety_tasks = [s for _, s in per_task_scores if s.get("safety_critical")]
    safety_passed = all(s.get("safety_passed", False) for s in safety_tasks) if safety_tasks else True
    return {
        "variant_id": variant_id,
        "mean_composite": round(sum(composites) / len(composites), 4),
        "median_composite": round(sorted(composites)[len(composites) // 2], 4),
        "safety_passed": safety_passed,
        "task_count": len(per_task_scores),
        "per_task": [
            {
                "task_id": t.get("id", t.get("_path", "?")),
                "category": t.get("category", "unknown"),
                "composite_score": s["composite_score"],
                "objective_score": s["objective_score"],
                "rubric_overall": s["rubric_overall"],
                "safety_critical": s.get("safety_critical", False),
                "safety_passed": s.get("safety_passed", True),
            }
            for t, s in per_task_scores
        ],
    }
