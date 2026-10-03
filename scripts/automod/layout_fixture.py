"""A frozen `/api/dashboard` snapshot, so the served build renders offline.

Why this exists: `frontend_probe.serve_build` is a plain static file server with
no backend behind it, so the real Mission Control build boots into its error
state — `DashboardPage.tsx` polls `/api/dashboard` every 5 s and renders
`Dashboard unavailable: ...` when the poll fails (the `sectionOk` guard at
`web/src/api.ts:101` drops a section whose sub-object is missing, and the whole
page is replaced when the poll itself fails). The gate's own probe record for
round SM_20261002_180956 shows the shape: 3 console checks, a 183-character
body, `Failed to load sessions: SyntaxError: Unexpected token \'<\'`. A layout
fingerprint collected there is a fingerprint of an error message, so the layout
leg serves this snapshot instead and grades the dashboard while it is actually
laid out.

How it was made: one live `GET :8080/api/dashboard` on 2026-10-03, with every
list truncated to its first two entries and every string to 40 characters, then
the `timestamp` pinned to the capture's own epoch. The truncation is the point:
the fingerprint must never be able to move because a backlog item was filed, a
worker id changed or a queue drained, so the data behind the panels is fixed at
capture time and only the CSS decides the geometry.

Every top-level key is kept on purpose. Dropping one is not a smaller dashboard,
it is a different page: `host` is read unguarded (`DashboardPage.tsx:1215` reads
`host.memory.used_bytes`), so a snapshot missing a key renders a React error
boundary where the dashboard should be — which is exactly the state this fixture
exists to avoid.
"""

from typing import Any, Mapping

DASHBOARD_SNAPSHOT: Mapping[str, Any] = {
 "network": {
  "window_days": 7.0,
  "since": "2026-09-26T15:22:42+00:00",
  "total": 472,
  "distinct_hosts": 92,
  "by_decision": {
   "allow": 414,
   "deny": 0,
   "grant-required": 58
  },
  "per_destination": [
   {
    "destination": "https://html.duckduckgo.com",
    "host": "html.duckduckgo.com",
    "count": 194,
    "denied": 0,
    "grant_required": 21,
    "scopes": 3,
    "last_at": "2026-10-03T03:13:13+00:00"
   },
   {
    "destination": "https://arxiv.org",
    "host": "arxiv.org",
    "count": 97,
    "denied": 0,
    "grant_required": 9,
    "scopes": 3,
    "last_at": "2026-10-03T13:45:54+00:00"
   }
  ],
  "per_scope": [
   {
    "scope": "worker:deep-research",
    "total": 401,
    "denied": 0,
    "grant_required": 0,
    "distinct_hosts": 73
   },
   {
    "scope": "interactive",
    "total": 58,
    "denied": 0,
    "grant_required": 58,
    "distinct_hosts": 20
   }
  ],
  "database": "/home/alansrobotlab/lloyd-data/workers.d",
  "database_present": True,
  "policy": {
   "telemetry": True,
   "enforce": False,
   "allow_entries": 0,
   "permanent_allow_entries": 0
  }
 },
 "host": {
  "cpu": {
   "percent": 4.6,
   "count": 32,
   "physical_count": 16,
   "load_average": [
    2.2001953125,
    2.27197265625
   ]
  },
  "memory": {
   "used_bytes": 202505191424,
   "total_bytes": 270076805120,
   "percent": 75.0
  },
  "swap": {
   "used_bytes": 13423210496,
   "total_bytes": 270076473344,
   "percent": 5.0
  },
  "disks": [
   {
    "path": "/",
    "used_bytes": 922988040192,
    "total_bytes": 3996467068928,
    "percent": 23.1
   }
  ],
  "gpus": [
   {
    "index": 0,
    "name": "NVIDIA GeForce RTX 3090",
    "gpu_util": 7.0,
    "mem_util": 3.0,
    "memory_used_mb": 14900.0,
    "memory_total_mb": 24576.0,
    "memory_pct": 60.628255208333336,
    "temperature_c": 48.0,
    "power_draw_w": 123.36,
    "power_limit_w": 275.0
   },
   {
    "index": 1,
    "name": "NVIDIA RTX PRO 6000 Blackwell Workstatio",
    "gpu_util": 98.0,
    "mem_util": 52.0,
    "memory_used_mb": 95844.0,
    "memory_total_mb": 97887.0,
    "memory_pct": 97.91289956786908,
    "temperature_c": 89.0,
    "power_draw_w": 453.38,
    "power_limit_w": 450.0
   }
  ],
  "uptime_seconds": 57084.88897371292,
  "boot_time": 1790983878.0
 },
 "vllm": [
  {
   "alias": "primary",
   "reachable": True,
   "engine": "vllm",
   "model_name": "Qwen3.8-Flash-Next-nvfp4",
   "awake": True,
   "requests_running": 2,
   "requests_waiting": 0,
   "requests_waiting_by_reason": {},
   "kv_cache_usage": 0.2416918429003021,
   "prompt_tokens_per_s": 5053.79993708705,
   "generation_tokens_per_s": 281.70397221395694,
   "preemptions_per_s": 0.0,
   "ttft_s": 1.6021554470062256,
   "itl_s": 0.026033414466957835,
   "prompt_tokens_total": 710892864.0,
   "generation_tokens_total": 8810041.0,
   "preemptions_total": 0.0,
   "finished_by_reason": {
    "stop": 16964,
    "length": 85,
    "abort": 0,
    "error": 0,
    "repetition": 0
   },
   "prefix_cache_hit_rate": 0.874703023199076,
   "prefix_cache_hit_rate_recent": None,
   "spec_decode_hit_rate": 0.5783281675400693,
   "spec_decode_hit_rate_recent": 0.5361635220125787,
   "base_url": "http://127.0.0.1:8096",
   "pressure": {
    "alias": "primary",
    "base_url": "http://127.0.0.1:8096",
    "sampling": True,
    "window_s": 300.0,
    "samples": 58,
    "kv_now": 0.30513595166163143,
    "kv_p50": 0.2809667673716012,
    "kv_p90": 0.31117824773413894,
    "kv_max": 0.3595166163141994,
    "warn_line": 0.65,
    "stale": False,
    "error": None
   }
  },
  {
   "alias": "djev",
   "reachable": True,
   "engine": "vllm",
   "model_name": "djev",
   "awake": True,
   "requests_running": 0,
   "requests_waiting": 0,
   "requests_waiting_by_reason": {},
   "kv_cache_usage": 0.0,
   "prompt_tokens_per_s": 0.0,
   "generation_tokens_per_s": 0.0,
   "preemptions_per_s": 0.0,
   "ttft_s": None,
   "itl_s": None,
   "prompt_tokens_total": 47817735.0,
   "generation_tokens_total": 867607.0,
   "preemptions_total": 0.0,
   "finished_by_reason": {
    "stop": 0,
    "length": 27542,
    "abort": 0,
    "error": 0,
    "repetition": 0
   },
   "prefix_cache_hit_rate": 0.6809597317815241,
   "prefix_cache_hit_rate_recent": None,
   "spec_decode_hit_rate": None,
   "spec_decode_hit_rate_recent": None,
   "base_url": "http://127.0.0.1:8010"
  }
 ],
 "primary": {
  "model": "primary",
  "base_url": "http://127.0.0.1:8096",
  "context_length": 262144,
  "max_turns": 60,
  "permission_mode": "bypassPermissions",
  "preserve_thinking_iterations": 6,
  "sessions": [
   {
    "session_id": "20261003_074608_autocode_e526",
    "running": True,
    "turn_id": "faf520ddcd2b",
    "source": "user",
    "started_at": "2026-10-03T08:16:00.823598",
    "enqueued_at": "2026-10-03T08:16:00.822840",
    "preempted": False,
    "activity": {
     "kind": "thinking",
     "label": "",
     "detail": "",
     "at": "2026-10-03T08:22:40.945092"
    },
    "pending_user": 0,
    "pending_ambient": 0,
    "title": "autocode #2128: The shipped injection ca"
   },
   {
    "session_id": "20261003_081809_autocode_adfc",
    "running": True,
    "turn_id": "ac62972c3cec",
    "source": "user",
    "started_at": "2026-10-03T08:18:12.454168",
    "enqueued_at": "2026-10-03T08:18:12.449572",
    "preempted": False,
    "activity": {
     "kind": "tool",
     "label": "Bash",
     "detail": "Probing candidate backend ports for /api",
     "at": "2026-10-03T08:22:42.853616"
    },
    "pending_user": 0,
    "pending_ambient": 0,
    "title": "autocode #2130: Diff a frontend layout f"
   }
  ],
  "running_count": 2,
  "queued_count": 0,
  "busy": True
 },
 "recent": {
  "sessions": [
   {
    "session_id": "20261002_181818_iv10c0",
    "title": "Today's Autonomous Autocode Run Summary",
    "preview": "what are the things that you've done tod",
    "last_active": "2026-10-03T01:19:11.015973Z",
    "message_count": 1,
    "platform": "mission-control",
    "inner_voice": True,
    "model": "primary",
    "goal": "",
    "goal_achieved": False,
    "todo_counts": {
     "pending": 0,
     "in_progress": 0,
     "completed": 0
    },
    "captions": {
     "total": 5,
     "captioned": 5
    }
   },
   {
    "session_id": "20261001_214646_iv87df",
    "title": "JitMem Concept and Asset Hierarchy Searc",
    "preview": "Well you can yeah you can open up it say",
    "last_active": "2026-10-02T16:24:26.880004Z",
    "message_count": 3,
    "platform": "mission-control",
    "inner_voice": True,
    "model": "",
    "goal": "",
    "goal_achieved": False,
    "todo_counts": {
     "pending": 0,
     "in_progress": 0,
     "completed": 0
    },
    "captions": {
     "total": 4,
     "captioned": 3
    }
   }
  ]
 },
 "agents": {
  "subagents": {
   "active": [],
   "active_count": 0,
   "recent": []
  },
  "background_tasks": {
   "active": [],
   "active_count": 0,
   "recent": [
    {
     "task_id": "bg-20261003-082138-ef51cb",
     "session_id": "20261003_081809_autocode_adfc",
     "description": "Building the frontend to /tmp in the bac",
     "command": "W=/home/alansrobotlab/lloyd-work/SM_2026",
     "status": "completed",
     "started_at": 1791040898.7882454,
     "finished_at": 1791040923.3905356,
     "exit_code": 0,
     "elapsed_s": 24.6,
     "output_path": "/home/alansrobotlab/lloyd-data/_pipeline"
    }
   ]
  },
  "tsc": {
   "last_run": {
    "/home/alansrobotlab/lloyd": {
     "root": "/home/alansrobotlab/lloyd",
     "status": "ok",
     "seconds": 5.4,
     "at": 1791038713.161403,
     "sessions": [],
     "errors_total": 2,
     "seeded": True
    }
   },
   "pending": {},
   "baseline_roots": [
    "/home/alansrobotlab/lloyd"
   ]
  },
  "qmd": {
   "responses": 1,
   "without_meta": 0,
   "reranked": 1,
   "rerank_fallbacks": 0,
   "last_fallback_at": None,
   "last_fallback_reason": None,
   "last_ms": 704,
   "last_phases": {
    "fts": 0,
    "embed": 23,
    "vec": 38,
    "chunk": 19,
    "rerank": 619
   },
   "djev_ranked": 0,
   "djev_fallbacks": 0,
   "last_djev_fallback_at": None,
   "last_djev_fallback_reason": None
  },
  "changes": {
   "turns_in_memory": 23,
   "enabled": True,
   "root": "/home/alansrobotlab/lloyd-data/sessions"
  },
  "tools": 147,
  "tool_sandbox": {
   "enforced": True,
   "bwrap": True,
   "error": None,
   "prefixes": [
    "bench_",
    "pt-eval-"
   ],
   "background_slugs": [
    "bench"
   ]
  },
  "protected_path_sandbox": {
   "enforcing": True,
   "fallback": False,
   "bwrap": True,
   "entries": [
    "/home/alansrobotlab/lloyd/agent-services",
    "/home/alansrobotlab/lloyd/.venvs"
   ],
   "error": None
  }
 },
 "services": {
  "services": [
   {
    "id": "agent-llm-primary",
    "name": "LLM Primary",
    "group": "infra",
    "port": 8096,
    "state": "active",
    "sub_state": "running",
    "port_healthy": True,
    "health": "healthy",
    "uptime": "pid 2520, uptime 15:50:57"
   },
   {
    "id": "agent-djev",
    "name": "djev (DiffusionGemma)",
    "group": "infra",
    "port": 8011,
    "state": "active",
    "sub_state": "running",
    "port_healthy": True,
    "health": "healthy",
    "uptime": "pid 3582, uptime 15:50:55"
   }
  ],
  "unhealthy": [],
  "total": 10
 },
 "workers": {
  "enabled": True,
  "duplicate_effects_suppressed": 0,
  "pool": {
   "running": True,
   "paused": False,
   "paused_by": [],
   "paused_since": None,
   "slots": 6,
   "in_flight": {
    "4424": {
     "source": "autocode",
     "kind": "round",
     "started_at": "2026-10-03T15:15:56.378805+00:00"
    },
    "4425": {
     "source": "autocode",
     "kind": "round",
     "started_at": "2026-10-03T15:18:06.830332+00:00"
    }
   },
   "in_flight_count": 2,
   "kv_gate": {
    "enabled": True,
    "max_kv_usage": 0.6,
    "window_seconds": 60.0,
    "engaged": False,
    "engaged_since": None,
    "engagements": 0,
    "kv_usage": 0.2990936555891238,
    "held_sources": []
   },
   "round_hold": {
    "enabled": True,
    "exempt": [
     "scheduled-task",
     "autotriage"
    ],
    "exempt_bound": None,
    "engaged": True,
    "engaged_since": "2026-10-03T15:15:58.551298+00:00",
    "engagements": 3,
    "held_sources": [
     "arch-review",
     "backlog-cluster"
    ],
    "exempt_refused": {},
    "bound_held": []
   },
   "primary_hold": {
    "enabled": True,
    "sources": [
     "autocode"
    ],
    "probe_seconds": 10.0,
    "probe_timeout_s": 2.0,
    "engaged": False,
    "engaged_since": None,
    "engagements": 0,
    "held_sources": [],
    "answering": True,
    "last_detail": "HTTP 200",
    "last_probe": 1791040958.9365108,
    "probing": False
   }
  },
  "depth_by_source": {
   "arch-review": {
    "completed": 23
   },
   "autocode": {
    "completed": 695,
    "quarantined": 1,
    "running": 2
   },
   "automod-regression": {
    "completed": 2
   },
   "autoresearch": {
    "completed": 50,
    "quarantined": 8
   },
   "autotriage": {
    "completed": 1846
   },
   "backlog-cluster": {
    "completed": 77
   },
   "bench-mine": {
    "completed": 227,
    "quarantined": 1
   },
   "board-steward": {
    "completed": 375,
    "quarantined": 5
   },
   "deep-research": {
    "completed": 34
   },
   "frontend-probe-canary": {
    "completed": 2
   },
   "owed-check": {
    "completed": 592,
    "quarantined": 33,
    "queued": 3
   },
   "scheduled-task": {
    "completed": 316,
    "quarantined": 14
   },
   "session-distill": {
    "completed": 44,
    "quarantined": 1
   },
   "youtube-digest": {
    "completed": 75
   }
  },
  "by_state": {
   "completed": 4358,
   "quarantined": 63,
   "running": 2,
   "queued": 3
  },
  "open_total": 5,
  "poisoned_total": 0,
  "quarantined_total": 63,
  "run_outcomes": {
   "window_hours": 48,
   "window_start": "2026-10-01T15:22:42.902366+00:00",
   "total": 1119,
   "ok": 739,
   "failed": 38,
   "skipped": 342,
   "fail_rate": 0.03395889186773905,
   "sources_failing": 6,
   "by_source": {
    "autocode": {
     "total": 104,
     "ok": 100,
     "failed": 3,
     "skipped": 1,
     "unfinished_matrix": 0,
     "deadline_cut": 0,
     "matrix_shrunk": 0,
     "fail_rate": 0.028846153846153848,
     "gpu_hours": 48.16,
     "last_completed": "2026-10-03T15:02:54.007609+00:00"
    },
    "autoresearch": {
     "total": 12,
     "ok": 12,
     "failed": 0,
     "skipped": 0,
     "unfinished_matrix": 12,
     "deadline_cut": 0,
     "matrix_shrunk": 12,
     "fail_rate": 0.0,
     "gpu_hours": 0.61,
     "last_completed": "2026-10-03T12:38:15.160222+00:00"
    },
    "autotriage": {
     "total": 479,
     "ok": 155,
     "failed": 0,
     "skipped": 324,
     "unfinished_matrix": 0,
     "deadline_cut": 0,
     "matrix_shrunk": 0,
     "fail_rate": 0.0,
     "gpu_hours": 8.02,
     "last_completed": "2026-10-03T15:17:48.240450+00:00"
    },
    "backlog-cluster": {
     "total": 17,
     "ok": 17,
     "failed": 0,
     "skipped": 0,
     "unfinished_matrix": 0,
     "deadline_cut": 0,
     "matrix_shrunk": 0,
     "fail_rate": 0.0,
     "gpu_hours": 0.03,
     "last_completed": "2026-10-03T13:44:43.599483+00:00"
    },
    "bench-mine": {
     "total": 36,
     "ok": 28,
     "failed": 1,
     "skipped": 7,
     "unfinished_matrix": 0,
     "deadline_cut": 0,
     "matrix_shrunk": 0,
     "fail_rate": 0.027777777777777776,
     "gpu_hours": 2.26,
     "last_completed": "2026-10-03T14:14:28.842576+00:00"
    },
    "board-steward": {
     "total": 85,
     "ok": 64,
     "failed": 11,
     "skipped": 10,
     "unfinished_matrix": 0,
     "deadline_cut": 0,
     "matrix_shrunk": 0,
     "fail_rate": 0.12941176470588237,
     "gpu_hours": 1.46,
     "last_completed": "2026-10-03T15:14:53.739744+00:00"
    },
    "deep-research": {
     "total": 6,
     "ok": 6,
     "failed": 0,
     "skipped": 0,
     "unfinished_matrix": 0,
     "deadline_cut": 0,
     "matrix_shrunk": 0,
     "fail_rate": 0.0,
     "gpu_hours": 0.54,
     "last_completed": "2026-10-03T03:17:00.545141+00:00"
    },
    "frontend-probe-canary": {
     "total": 2,
     "ok": 2,
     "failed": 0,
     "skipped": 0,
     "unfinished_matrix": 0,
     "deadline_cut": 0,
     "matrix_shrunk": 0,
     "fail_rate": 0.0,
     "gpu_hours": 0.04,
     "last_completed": "2026-10-02T16:00:04.378690+00:00"
    },
    "owed-check": {
     "total": 292,
     "ok": 273,
     "failed": 19,
     "skipped": 0,
     "unfinished_matrix": 0,
     "deadline_cut": 0,
     "matrix_shrunk": 0,
     "fail_rate": 0.06506849315068493,
     "gpu_hours": 13.1,
     "last_completed": "2026-10-03T15:17:29.641268+00:00"
    },
    "scheduled-task": {
     "total": 65,
     "ok": 64,
     "failed": 1,
     "skipped": 0,
     "unfinished_matrix": 0,
     "deadline_cut": 0,
     "matrix_shrunk": 0,
     "fail_rate": 0.015384615384615385,
     "gpu_hours": 7.97,
     "last_completed": "2026-10-03T13:26:26.445858+00:00"
    },
    "session-distill": {
     "total": 4,
     "ok": 4,
     "failed": 0,
     "skipped": 0,
     "unfinished_matrix": 0,
     "deadline_cut": 0,
     "matrix_shrunk": 0,
     "fail_rate": 0.0,
     "gpu_hours": 0.21,
     "last_completed": "2026-10-03T03:14:46.810705+00:00"
    },
    "youtube-digest": {
     "total": 17,
     "ok": 14,
     "failed": 3,
     "skipped": 0,
     "unfinished_matrix": 0,
     "deadline_cut": 0,
     "matrix_shrunk": 0,
     "fail_rate": 0.17647058823529413,
     "gpu_hours": 0.72,
     "last_completed": "2026-10-03T15:08:53.717758+00:00"
    }
   }
  },
  "maintenance": {
   "at": "2026-10-03T15:15:53.678467+00:00",
   "scanned": 0,
   "revived": 0,
   "quarantined": 0,
   "escalations": [],
   "report_path": None
  },
  "sources": [
   {
    "name": "autocode",
    "enabled": True,
    "open": 2,
    "running": 2,
    "completed": 695,
    "queue_failed": 0,
    "poisoned": 0,
    "quarantined": 1,
    "run_total": 104,
    "run_failed": 3,
    "run_fail_rate": 0.028846153846153848
   },
   {
    "name": "owed-check",
    "enabled": True,
    "open": 3,
    "running": 0,
    "completed": 592,
    "queue_failed": 0,
    "poisoned": 0,
    "quarantined": 33,
    "run_total": 292,
    "run_failed": 19,
    "run_fail_rate": 0.06506849315068493
   }
  ],
  "recent_runs": [
   {
    "run_id": "run_autotriage_20261003_151346_a9ad5c",
    "source": "autotriage",
    "status": "success",
    "started_at": "2026-10-03T15:13:46.073174+00:00",
    "duration_seconds": 242.16726950099837,
    "summary": "#2130 \u2192 confirmed"
   },
   {
    "run_id": "run_owed-check_20261003_151252_03508c",
    "source": "owed-check",
    "status": "success",
    "started_at": "2026-10-03T15:12:52.698426+00:00",
    "duration_seconds": 276.94282662600017,
    "summary": "#1416: applied 1 ruling, 1 settled \u2014 Ent"
   }
  ]
 },
 "autonomy": {
  "total": 35,
  "by_status": {
   "up_next": 33,
   "draft": 2
  },
  "overdue": [],
  "overdue_count": 0,
  "held": [
   {
    "name": "Email & Calendar Triage",
    "status": "draft",
    "frequency": "every-15min",
    "next_run": "2026-09-16T14:29:05.933955+00:00",
    "last_run": "2026-09-16T14:14:05.933955+00:00",
    "blocked": "draft"
   }
  ],
  "held_count": 1,
  "upcoming": [
   {
    "name": "Data Pipeline",
    "status": "up_next",
    "frequency": "6x-daily",
    "next_run": "2026-10-03T15:39:40.991106+00:00",
    "last_run": "2026-10-03T11:39:40.991106+00:00",
    "blocked": "already ran this period"
   },
   {
    "name": "Fact Improvement Pass",
    "status": "up_next",
    "frequency": "daily",
    "next_run": "2026-10-03T21:00:00+00:00",
    "last_run": "2026-10-02T21:02:25.939266+00:00",
    "blocked": "outside hours 14"
   }
  ],
  "failing": [],
  "classifier": "autonomy",
  "running": [],
  "running_count": 0
 },
 "backlog": {
  "total": 2067,
  "by_status": {
   "done": 2053,
   "draft": 12,
   "in_progress": 2
  },
  "by_board": [
   {
    "board": "lloyd",
    "open": 14,
    "total": 2058
   },
   {
    "board": "alfie",
    "open": 0,
    "total": 7
   }
  ],
  "open_total": 14,
  "recent_open": [
   {
    "name": "2130-diff-a-frontend-layout-fingerprint-",
    "status": "in_progress",
    "board": "lloyd",
    "mtime": 1791040688.5729458
   },
   {
    "name": "2129-pin-the-four-5173-scheme-certificat",
    "status": "in_progress",
    "board": "lloyd",
    "mtime": 1791040558.0667436
   }
  ],
  "umbrellas": 0,
  "grouped": 0,
  "health": {
   "open": {
    "draft": 12,
    "in_progress": 2
   },
   "draft": {
    "pool": 0,
    "quarantined": 1,
    "grouped": 0,
    "needs_human": 0,
    "held": 0,
    "parked": 2,
    "triaged": 9,
    "total": 12
   },
   "closed_needs_human": 0,
   "owed": {
    "items": 434,
    "entries": 886,
    "due": 776,
    "outside": [
     {
      "item_id": 947,
      "name": "_pipeline (3.1 GB) is absent from the co",
      "what": "Choose and mount an off-box destination ",
      "needs": "Alan must supply the target \u2014 a secret o",
      "since": "2026-09-27T18:16:25.221172"
     },
     {
      "item_id": 947,
      "name": "_pipeline (3.1 GB) is absent from the co",
      "what": "1. Choosing and mounting the actual off-",
      "needs": "One decision, then one mount. Decide whe",
      "since": "2026-10-03T03:14:56.746523"
     }
    ]
   },
   "up_next": {
    "total": 0,
    "umbrellas": 0,
    "singles": 0,
    "never_attempted": 0,
    "ready": 0,
    "unready": 0
   },
   "flow": {
    "24h": {
     "created": 65,
     "closed": 59,
     "net": 6
    },
    "7d": {
     "created": 582,
     "closed": 576,
     "net": 6
    }
   },
   "self_spawned_open": 4,
   "live_blockers": {
    "open": 0,
    "untriaged": 0
   },
   "landed_items_7d": 394,
   "implement_pool": {
    "ready": 0,
    "bound": 394,
    "floor": 20
   },
   "sweep": {
    "unswept": 0,
    "swept": 12,
    "parked": 2,
    "worth": {
     "high": 1,
     "medium": 9,
     "low": 2
    }
   },
   "decisions": {
    "window": {
     "days": 7,
     "since": "2026-09-26T15:21:58Z",
     "until": "2026-10-03T15:21:58Z"
    },
    "promotions": {
     "count": 524,
     "by_decider": {
      "triage": 486,
      "gate": 34,
      "reopen": 4
     },
     "outcome": {
      "verdict": "measured",
      "promotions": 524,
      "terminal": {
       "landed": 371,
       "closed": 25,
       "returned": 50,
       "needs_human": 1,
       "closed_off_ledger": 68,
       "retriage": 2,
       "open": 7
      },
      "done_rate": 0.885,
      "landed_rate": 0.708,
      "dwell_h_median": 1.1
     }
    },
    "per_day": [
     {
      "day": "2026-09-26",
      "promotions": 18,
      "by_decider": {
       "triage": 14,
       "gate": 4
      }
     },
     {
      "day": "2026-09-27",
      "promotions": 112,
      "by_decider": {
       "triage": 107,
       "gate": 5
      }
     }
    ],
    "zero_streak": {
     "longest": 0,
     "current": 0
    },
    "retire_reopen": {
     "retired": 531,
     "reopened": 41,
     "ids": [
      1574,
      1596
     ],
     "by_kind": {
      "landed": {
       "retired": 368,
       "reopened": 0,
       "after_retriage": 0,
       "by_reopen_item": 0
      },
      "closed": {
       "retired": 102,
       "reopened": 0,
       "after_retriage": 0,
       "by_reopen_item": 0
      },
      "parked": {
       "retired": 60,
       "reopened": 41,
       "after_retriage": 0,
       "by_reopen_item": 0
      },
      "needs_human": {
       "retired": 1,
       "reopened": 0,
       "after_retriage": 0,
       "by_reopen_item": 0
      }
     }
    }
   }
  }
 },
 "automod": {
  "computed_at": "2026-10-03T15:21:46+00:00",
  "since_days": 7.0,
  "events": 11362,
  "grouping": {
   "cluster_runs": 60,
   "clusters_formed": 0,
   "items_clustered": 0,
   "group_triages": 1,
   "duplicates_closed": 0,
   "retired_in_group": 1,
   "folded": 0,
   "kept": 1,
   "umbrellas_formed": 0,
   "umbrellas_landed": 0,
   "members_closed": 0,
   "members_per_landing": None
  },
  "acceptance": {
   "landed": 336,
   "with_outcome": 324,
   "met": 313,
   "hit_rate": 0.966
  },
  "audit": {
   "rounds_compared": 391,
   "author_met": 1623,
   "grader_met": 1634,
   "delta": 1.007
  },
  "review": {
   "rounds_graded": 381,
   "refusals": 139,
   "rounds_refused": 107,
   "fixed_in_turn": 70,
   "premise_unsound": 0,
   "escalated": 8,
   "refusal_rate": 0.281,
   "grader_unavailable": 42,
   "review_reused": 0,
   "post_landing_clauses": 4,
   "grader_retries": 0,
   "reused_rungs": 73,
   "seam_only_refusals": 0
  },
  "spawn": {
   "triage_filed": 33,
   "triage_closed": 96,
   "triage_ratio": 0.344,
   "implement_filed": 20,
   "implement_closed": 374,
   "implement_ratio": 0.053,
   "triage_merged": 16,
   "implement_merged": 1,
   "findings_appended": 1741,
   "triage_findings_appended": 3821,
   "expired": 0,
   "self_spawned_open": {
    "count": 4,
    "oldest_days": 5.8,
    "bound_days": 30,
    "over_bound": 0
   }
  },
  "human_touch": {
   "landed": 338,
   "touched_within_7d": 40,
   "rate": 0.118,
   "rounds": [
    "SM_20260926_182902",
    "SM_20260926_233722"
   ]
  },
  "test_honesty": {
   "grader_findings": 1305,
   "landed_with_or_true": 3,
   "per_gated_round": 3.425
  },
  "bookkeeping": {
   "nameless_deferrals": 0,
   "stranded_landings": 0,
   "bare_aborts": 0
  },
  "verdict_plumbing": {
   "verdicts_with_source": 544,
   "regex": 5,
   "regex_rate": 0.009,
   "truncated": 7,
   "finalizer_tokens_median": 1147.5
  },
  "throughput": {
   "items_closed": 470,
   "landings_rescued": 11,
   "item_verdicts_refused": 14,
   "gates_rescued": 14,
   "reviews_unavailable": 42,
   "landed_without_restart": 170,
   "items_closed_per_day": 67.14,
   "rounds_finished": 453,
   "rounds_landed": 336,
   "rounds_rejected": 1,
   "median_turns_landed": 62.0,
   "median_gate_seconds": 483.6,
   "median_full_gate_s": 414.4,
   "tests_pre_existing_passes": 68,
   "red_tree_filed": 44,
   "red_tree_closed": 19
  },
  "rollbacks": {
   "regression_coverage": {
    "promotions": 338,
    "measured": 338,
    "could_not_evaluate": 0,
    "compared_nothing": 0
   },
   "count": 0,
   "triggers": [],
   "true_positives": None
  },
  "arch_review": {
   "reviewed": 13,
   "by_kind": {
    "doc": 5,
    "group": 8
   },
   "updated": 8,
   "committed": 6,
   "rejected": 5,
   "filed": 51,
   "merged": 0,
   "appended_to": 4,
   "stray_writes": 21,
   "by_status": {
    "current": 1,
    "stale": 12
   }
  },
  "flow": {
   "24h": {
    "created": 65,
    "closed": 59,
    "net": 6
   },
   "7d": {
    "created": 582,
    "closed": 576,
    "net": 6
   }
  },
  "duty_cycle": {
   "rate": 0.6816557244380946,
   "busy_hours": 114.5,
   "window_hours": 168.0,
   "turns": 176,
   "gaps": 157,
   "idle_minutes": {
    "abort": 143.7,
    "other": 211.4,
    "promotion_gap": 2617.3,
    "restart": 62.3
   },
   "gap_counts": {
    "abort": 11,
    "other": 17,
    "promotion_gap": 128,
    "restart": 1
   },
   "largest_gap_minutes": 346.9,
   "waits": {
    "settle": {
     "n": 1,
     "waited": 1,
     "minutes": 5.2,
     "median_s": 310.4,
     "max_s": 310.4
    },
    "rounds": {
     "n": 343,
     "waited": 52,
     "minutes": 212.7,
     "median_s": 2.8,
     "max_s": 2284.8
    },
    "flush": {
     "n": 202,
     "waited": 96,
     "minutes": 197.3,
     "median_s": 5.0,
     "max_s": 1303.4
    }
   }
  },
  "overrides": {
   "count": 21,
   "found": 21,
   "cap": 120,
   "by_event": {
    "item_landed": 20,
    "restart": 1
   },
   "fields": {
    "reason": 21,
    "note": 0
   },
   "rows": [
    {
     "created_at": "2026-10-03T04:38:47Z",
     "event": "item_landed",
     "round_id": "SM_20261003_040232",
     "item_id": 2092,
     "person": "Alan",
     "field": "reason",
     "text": "the round reported every acceptance clau"
    },
    {
     "created_at": "2026-10-03T03:30:53Z",
     "event": "item_landed",
     "round_id": "SM_20261003_021139",
     "item_id": 2090,
     "person": "Alan",
     "field": "reason",
     "text": "the round reported every acceptance clau"
    }
   ]
  },
  "guard_denials": {
   "recorded": True,
   "count": 37,
   "by_guard": {
    "safety": 30,
    "protected_write": 1,
    "grant": 2,
    "tool_sandbox": 4
   },
   "by_class": {
    "background": 33,
    "bench": 4
   },
   "by_where": {
    "hook": 32,
    "dispatch": 5
   },
   "guard_by_class": {
    "safety": {
     "background": 30
    },
    "protected_write": {
     "background": 1
    },
    "grant": {
     "background": 2
    },
    "tool_sandbox": {
     "bench": 4
    }
   },
   "top_labels": {
    "safety:sudo": 15,
    "safety:rm -rf on root/home/system path": 7,
    "safety:Obsidian Sync registration: `cp` on the Obsidian Sync registration (/home/alansrobotlab/.config/obsidian-headless)": 2,
    "safety:install provenance: \\": 1,
    "protected_write:the supervisor, guardian and service units": 1,
    "safety:destructive operation on find -delete over a top-level folder of the vault (/home/alansrobotlab/obsidian/_pipeline)": 1,
    "safety:destructive operation on rm on a glob over the lloyd tree (/home/alansrobotlab/lloyd)": 1,
    "safety:destructive operation on find -delete over the lloyd tree (/home/alansrobotlab/lloyd)": 1
   }
  },
  "install_provenance": {
   "recorded": True,
   "decisions": 20,
   "rows": 36,
   "names": 1100,
   "by_outcome": {
    "denied": 4,
    "overridden": 0,
    "unvetted": 4,
    "cleared": 6,
    "declared": 10
   }
  },
  "enabled": True,
  "current": {
   "round_id": None,
   "state": None
  },
  "halted": False,
  "broken": False,
  "pending_restart": {
   "draining": False,
   "drain_remaining_s": 0.0,
   "entries": [],
   "restart_needed": 0,
   "oldest_age_s": None,
   "flushing": False,
   "flush_due": False,
   "flush_why": "nothing pending",
   "stage": None
  }
 },
 "usage": {
  "last_hour": {
   "requests": 25,
   "input_tokens": 2131738,
   "output_tokens": 442047,
   "cache_create": 636800,
   "cache_read": 1868800,
   "cost_usd": 0.0,
   "duration_ms": 6294535,
   "duration_api_ms": 0
  },
  "last_24h": {
   "requests": 595,
   "input_tokens": 49904341,
   "output_tokens": 11404644,
   "cache_create": 19875200,
   "cache_read": 43606400,
   "cost_usd": 0.0,
   "duration_ms": 173628191,
   "duration_api_ms": 0
  },
  "last_7d": {
   "requests": 4662,
   "input_tokens": 361095489,
   "output_tokens": 79648594,
   "cache_create": 158764800,
   "cache_read": 313756800,
   "cost_usd": 0.0,
   "duration_ms": 1270758078,
   "duration_api_ms": 0
  },
  "daily": [
   {
    "bucket": "2026-09-26",
    "requests": 233,
    "input_tokens": 15846515,
    "output_tokens": 3016420,
    "cache_create": 6105600,
    "cache_read": 13184000,
    "cost_usd": 0.0
   },
   {
    "bucket": "2026-09-27",
    "requests": 627,
    "input_tokens": 45674074,
    "output_tokens": 10085848,
    "cache_create": 24003200,
    "cache_read": 38892800,
    "cost_usd": 0.0
   }
  ],
  "by_model_24h": [
   {
    "model": "primary",
    "requests": 595,
    "input_tokens": 49904341,
    "output_tokens": 11404644,
    "cache_create": 19875200,
    "cache_read": 43606400,
    "cost_usd": 0.0
   }
  ],
  "by_skill_24h": [
   {
    "skill": "retention-sweep",
    "route": "prefetch",
    "requests": 36,
    "input_tokens": 3215712,
    "output_tokens": 820467,
    "cache_create": 1379200,
    "cache_read": 2972800
   },
   {
    "skill": "service-health-check",
    "route": "prefetch_excerpt",
    "requests": 30,
    "input_tokens": 2880328,
    "output_tokens": 718310,
    "cache_create": 1401600,
    "cache_read": 2585600
   }
  ],
  "prefix_misses_1h": {
   "turns": 25,
   "turns_measured": 25,
   "turns_with_misses": 2,
   "prefix_misses": 2,
   "reprefill_tokens": 197851,
   "worst_turn_reprefill": 145074
  },
  "prefix_misses_24h": {
   "turns": 595,
   "turns_measured": 595,
   "turns_with_misses": 40,
   "prefix_misses": 112,
   "reprefill_tokens": 13387958,
   "worst_turn_reprefill": 1783193
  },
  "stop_reasons_24h": [
   {
    "stop_reason": "stop",
    "turns": 591,
    "wrapped_up": 0
   },
   {
    "stop_reason": "max_turns",
    "turns": 2,
    "wrapped_up": 1
   }
  ],
  "tool_errors_24h": {
   "turns_measured": 595,
   "tool_calls": 13276,
   "tool_errors": 473,
   "tool_ms_total": 60688830,
   "by_class": {
    "tool_error": 457,
    "denied": 15,
    "parse_error": 1
   }
  },
  "ttft_24h": {
   "turns_measured": 595,
   "first_p50_ms": 4062,
   "first_p90_ms": 7536,
   "max_ms": 38841
  },
  "reasoning_tokens_24h": {
   "turns_measured": 595,
   "reasoning_tokens": 6796672,
   "output_tokens": 11404644
  }
 },
 "timestamp": 1791040962.9285274
}


#: What the layout leg's server answers for each `/api/...` path. `*` is the
#: default for every path the dashboard polls but does not gate on —
#: `/api/mc/state` (the tab/focus mirror), `/api/sessions`,
#: `/api/workers/status` — so a server that never answers is not itself the
#: thing under test.
API_STUB: Mapping[str, Any] = {
    "/api/dashboard": DASHBOARD_SNAPSHOT,
    "/api/mc/state": DASHBOARD_SNAPSHOT,
    "*": {},
}
