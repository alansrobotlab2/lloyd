"""The text of an exception, with an anyio task group's wrapper removed.

Stdlib-only and imported from both sides of the MCP seam — the discovery path
in `app/mcp_discovery.py`, which renders a down server's failure in the Tools
page, and tool dispatch in `app/harness/mcp_pool.py`, which puts that text in
the tool_result the model reads — so it sits below both rather than inside
`app.harness`, which `agent_mcp` code must not import.
"""


def root_cause(exc: BaseException) -> str:
    """Innermost message from a (possibly nested) ExceptionGroup.

    anyio task groups repackage a failure as an ExceptionGroup whose str()
    is "unhandled errors in a TaskGroup (1 sub-exception)" — true, and
    useless in a UI. Unwrap to the part a human can act on.

    On the dispatch path this is what made a defect unreadable (#936): 52 rows
    of `_pipeline/trajectories/2026-09-*.jsonl` through 2026-09-17 carried the
    group summary as the entire error, so a read timeout and a dead server
    reached the model indistinguishable.

    Only a group is rewritten. A plain exception's `str()` passes through
    untouched — messages that were never opaque must not be reformatted on the
    way past, or the fix turns into a change to every transport error message.
    """
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return str(exc) or exc.__class__.__name__
