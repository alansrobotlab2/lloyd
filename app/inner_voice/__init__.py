"""Inner Voice — thin observer that watches the primary agent.

One LLM, one prompt, four levers (inject / cancel / ambient / deny_tool).
All judgment lives in `observer_prompt.SYSTEM_PROMPT`. The Python here is
plumbing.

Public surface:
    observer.install_observer(...)  — wire onto a HookRegistry per turn
    observer.close_observer(state)  — best-effort cleanup
"""
