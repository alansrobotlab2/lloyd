"""Bash calls from turn 20260905_011748_iv84e4, verbatim.

The turn that motivated the repetition guard: a code-review request that ran
10m14s over 33 iterations, produced zero characters of user-visible text, and
was cancelled by Inner Voice. Message indices are positions in that session's
`messages` array, kept so a finding can be traced back to the transcript.

Shape of the turn:

  4-51   healthy exploration — each call opens new ground
  53-66  the loop — six reformulations of "who consumes iv_inject_queue /
         iv_cancel_requested outside the IV package". The answer (nobody) was
         correct and stable from the first attempt; the primary read the empty
         result as a broken query and kept rewriting the filter
  72-79   chasing symbols that do not exist (`lloyd_mcp`, the package is
         `agent_mcp`; `build_subliminal_context`, never defined anywhere),
         ending in two `find /` scans and a `grep -rn` that hit the 120s wall

A guard that fires anywhere in 4-51 is too eager to ship; one that never fires
in 53-66 would not have helped.
"""

REAL_BASH_CALLS = [
    (4, 'ls -la ~/lloyd/ | head -50; echo "--- inner voice files ---"; find ~/lloyd -iname \'*inner*voice*\' -o -iname \'*inner_voice*\' -o -iname \'*iv_*\' 2>/dev/null | grep -v \'.venv\' | grep -v node_modules | grep -v \'\\.git/\' | head -60'),
    (6, 'cd ~/lloyd && grep -rl "inner_voice\\|inner voice\\|InnerVoice" --include="*.py" --include="*.js" --include="*.ts" --include="*.html" --include="*.yaml" --include="*.md" . 2>/dev/null | grep -v \'\\.venv\\|_pipeline\\|node_modules\\|/sessions/\\|/event_logs/\\|\\.git/\' | sort | head -60'),
    (8, 'cd ~/lloyd && ls -la app/ architecture/ workers/ 2>/dev/null'),
    (10, 'cd ~/lloyd && ls -la app/inner_voice/ app/routers/ app/harness/ && wc -l app/inner_voice/*.py app/routers/inner_voice.py app/routers/_messages_inner_voice.py app/routers/messages.py app/event_log.py tests/integration/test_observer.py tests/integration/test_iv_guards.py scripts/iv_grade.py scripts/iv_v4b_replay.py 2>/dev/null'),
    (12, 'cd ~/lloyd && grep -n "inner_voice" -A 60 config.yaml | head -100'),
    (18, 'cd ~/lloyd && grep -n "^\\(class\\|def\\|async def\\|@\\|[A-Z_]* =\\|    def \\|    async def \\)" app/inner_voice/observer.py | head -150'),
    (20, 'cd ~/lloyd && head -120 app/inner_voice/observer.py'),
    (30, 'cd ~/lloyd && grep -n "def build_subliminal_context" -A 60 app/routers/_messages_subliminal.py | head -80; echo "=== inner_voice router ==="; cat app/routers/inner_voice.py'),
    (34, 'cd ~/lloyd && sqlite3 usage.db "select count(*), min(created_at), max(created_at) from inner_voice_observations" 2>&1; echo "=== by action ==="; sqlite3 usage.db "select action, count(*) c from inner_voice_observations group by 1 order by c desc limit 25" 2>&1; echo "=== by trigger ==="; sqlite3 usage.db "select trigger, count(*), sum(input_tokens+output_tokens) tok, round(avg(latency_ms)) avglat, sum(case when error is not null then 1 else 0 end) errs from inner_voice_observations group by 1 order by 2 desc" 2>&1'),
    (38, 'cd ~/lloyd && echo "=== sessions count/size ==="; ls sessions/*.json | wc -l; du -sh sessions; ls -S sessions/*.json | head -3 | xargs ls -la; echo "=== reuse of fast-path predicates ==="; grep -rn "_bash_command_is_safely_readonly\\|_SAFE_BASH_FIRST_WORDS\\|injects_primary_has_seen" --include="*.py" . | grep -v \'\\.venv\'; echo "=== safety.py ==="; cat app/harness/safety.py'),
    (42, 'cd ~/lloyd && echo "=== TRIGGER strings in prompt module ==="; grep -n "TRIGGER:" app/inner_voice/observer_prompt.py; echo "=== deny_tool remaining ==="; grep -rn "deny_tool" --include="*.py" --include="*.md" --include="*.js" . | grep -v \'\\.venv\\|_pipeline\\|\\.git/\' | head -20; echo "=== _session_protected_paths callers ==="; grep -rn "_session_protected_paths\\|_load_todos_from_session\\|_session_inner_voice_enabled(" --include="*.py" . | grep -v \'\\.venv\''),
    (44, 'cd ~/lloyd && grep -n "TRIGGER_\\|== \\"pretool\\"\\|startswith(\\"PRETOOL\\|user_prompt.startswith" app/inner_voice/observer_prompt.py | head -30'),
    (46, 'cd ~/lloyd && grep -n "trigger=\\"pretool\\"\\|trigger=\'pretool\'\\|trigger=f\\"pretool\\|_call_observer(\\|user_prompt=summary\\|^def build_\\|^    lines.append(\\"PRETOOL" app/inner_voice/observer.py app/inner_voice/observer_prompt.py | head -40; echo "=== first line of each prompt builder ==="; for f in build_user_prompt_for_event build_pretool_event_summary; do echo "--- $f"; grep -n "def $f" -A 30 app/inner_voice/observer_prompt.py | grep -n "append\\|return\\|f\\"" | head -12; done'),
    (48, 'cd ~/lloyd && echo "=== is_pretool call sites ==="; grep -rn "is_pretool=True\\|OBSERVER_SYSTEM_PROMPT_PRETOOL" --include="*.py" . | grep -v \'\\.venv\'; echo; echo "=== system prompt sizes ==="; python3 - <<\'PY\'\nimport re,io,sys\nsrc=open(\'app/inner_voice/observer_prompt.py\').read()\ntry:\n    sys.path.insert(0,\'.\')\n    import importlib\n    m=importlib.import_module(\'app.inner_voice.observer_prompt\')\n    for name in dir(m):\n        if name.startswith(\'OBSERVER_SYSTEM_PROMPT\'):\n            v=getattr(m,name)\n            if isinstance(v,str):\n                print(f"{name}: {len(v)} chars ~{len(v)//4} tok")\nexcept Exception as e:\n    print("import failed:",e)\nPY\necho "=== config knobs (non-default set) ==="; python3 -c "\nimport yaml;c=yaml.safe_load(open(\'config.yaml\'))\niv=c[\'autonomy\'][\'inner_voice\']\nfor k,v in sorted(iv.items()):\n    print(f\'  {k}: {v}\')\n"'),
    (51, 'cd ~/lloyd && echo "=== where is the autonomy.inner_voice config? ==="; grep -ln "inner_voice:" *.yaml *.yml 2>/dev/null; for f in config.yaml lloyd-config.yaml config.example.yaml; do echo "--- $f"; python3 -c "\nimport yaml,sys\nd=yaml.safe_load(open(\'$f\'))\ndef walk(o,p=\'\'):\n    if isinstance(o,dict):\n        for k,v in o.items():\n            if k==\'inner_voice\' and isinstance(v,dict):\n                print(\'$f\', p+\'.\'+k if p else k, \'->\', len(v), \'keys\')\n                for kk,vv in sorted(v.items()): print(\'   \',kk,\'=\',vv)\n            walk(v,p+\'.\'+str(k))\n    elif isinstance(o,list):\n        for i,v in enumerate(o): walk(v,p+f\'[{i}]\')\nwalk(d)\n" 2>/dev/null; done'),
    (53, 'cd ~/lloyd && echo "=== sinks wired where ==="; grep -rn "set_inject_sink\\|set_clarify_sink\\|register_pending_clarify\\|resolve_clarify\\|iv_inject_queue\\|set_last_turn_usage" --include="*.py" . | grep -v \'\\.venv\' | grep -v tests/ ; echo; echo "=== primary honors cancel? ==="; grep -rn "iv_cancel_requested\\|iv_goal_clarify" --include="*.py" app/ | grep -v inner_voice/ ; echo; echo "=== inject consumed? ==="; grep -rn "iv_inject_queue.qsize\\|iv_inject_queue.get\\|iv_inject" --include="*.py" app/ | grep -v inner_voice/ | head'),
    (55, 'cd ~/lloyd && grep -rn "iv_inject\\|iv_cancel\\|iv_goal_clarify\\|set_last_turn_usage\\|iv_prior\\|iv_last_" app/ --include="*.py" | grep -v "app/inner_voice/" ; echo "=== EXIT $? ==="; echo "--- tools registry: how does primary consume?"; grep -rn "iv_set_inject_sink\\|iv_set_clarify_sink" app/ | head'),
    (58, 'cd ~/lloyd && for pat in iv_prior iv_inject_queue iv_cancel_requested iv_goal_clarify set_last_turn_usage iv_last_inject iv_goal iv_goal_card iv_meta iv_events iv_dedup_key iv_seen_tools; do n_all=$(grep -rl "$pat" app/ --include=\'*.py\' 2>/dev/null | grep -vc \'inner_voice\' ); echo "--- $pat : files_outside_inner_voice=$n_all"; grep -rl "$pat" app/ --include=\'*.py\' 2>/dev/null | grep -v \'inner_voice\' | sed \'s/^/      /\'; done'),
    (60, 'cd ~/lloyd && grep -rn "goal_card" app/ --include=\'*.py\' | grep -v "app/inner_voice/" | head -20; echo "=== end ==="; grep -rn "async_drain_timeout_seconds\\|async_nonterminal" app/ --include=\'*.py\' | grep -v inner_voice/ | head'),
    (62, 'cd ~/lloyd && echo "=== consumers outside the IV package (correct filter) ==="; for pat in iv_events iv_cancel_requested iv_inject_queue iv_goal_clarify iv_goal iv_last_inject iv_prior iv_goal_meta set_last_turn_usage; do files=$(grep -rl "$pat" app/ --include=\'*.py\' 2>/dev/null | grep -v \'^app/inner_voice/\'); echo "--- $pat"; echo "$files" | sed \'s/^/     /\'; done'),
    (64, 'cd ~/lloyd && echo "=== repo-wide, excluding IV package ==="; grep -rn "iv_inject_queue\\|iv_cancel_requested\\|iv_goal_clarify\\|iv_last_inject\\|set_last_turn_usage\\|\\[iv_prior\\]" --include="*.py" . 2>/dev/null | grep -v "/\\.venv/\\|^\\./app/inner_voice/\\|/_pipeline/" | head -40'),
    (66, 'cd ~/lloyd && echo "=== ALL files repo-wide (any type) ==="; grep -rIl "iv_inject_queue\\|iv_cancel_requested" . 2>/dev/null | grep -v "/\\.venv/\\|/_pipeline/\\|\\.git/" | head -20; echo "=== subliminal state keys ==="; grep -n "state\\[\\|state.get(" app/routers/_messages_subliminal.py | head -30; echo "=== SDK/agent loop entry ==="; grep -rn "ClaudeSDKClient\\|query(" app/*.py | head -10'),
    (68, 'cd ~/lloyd && ls app/ && echo "=== messages.py: IV touchpoints ===" && grep -n "inner_voice\\|_iv\\b\\|iv_" app/routers/messages.py | head -60'),
    (70, 'cd ~/lloyd && echo "--- _messages_subliminal.py iv refs ---"; grep -rn "iv_\\|inner_voice" app/routers/_messages_subliminal.py | head -40; echo; echo "--- files in routers ---"; ls -la app/routers/ | head -40'),
    (72, 'ls ~/ | head -30; echo "=== lloyd-mcp location ==="; python3 -c "import lloyd_mcp,os;print(os.path.dirname(lloyd_mcp.__file__))" 2>&1; ls ~/lloyd-mcp 2>/dev/null | head -20'),
    (74, 'find / -name "subliminal*.py" -path "*lloyd*" 2>/dev/null | head; echo "=== harness ==="; find / -path "*lloyd_mcp*" -name "*.py" 2>/dev/null | grep -v test | head -30'),
    (76, 'cd ~/lloyd && head -20 app/routers/_messages_subliminal.py; echo "==="; grep -rn "build_subliminal_context(" --include="*.py" . | grep -v \'\\.venv\' | head'),
    (79, 'cd ~/lloyd && cat app/routers/_messages_subliminal.py | sed -n \'20,120p\'; echo "=== grep anywhere ==="; grep -rn "def build_subliminal_context" . 2>/dev/null | grep -v \'\\.venv\''),
]
