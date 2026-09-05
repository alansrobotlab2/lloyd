"""Inner Voice — thin observer that watches the primary agent.

One LLM, one prompt, five soft levers (noop / inject / cancel / ambient /
clarify). There is no `deny_tool`: since v4 Inner Voice cannot block a tool
dispatch, and the only hard gate is `app.harness.safety`, which runs on every
primary turn whether or not the session opted into Inner Voice.

All judgment lives in the vault prompt behind
`observer_prompt.get_system_prompt()`. The Python here is plumbing.

Public surface:
    observer.install_observer(...)  — wire onto a HookRegistry per turn
    observer.close_observer(state)  — best-effort cleanup
"""
