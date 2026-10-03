"""Shared pieces for #2134: skill markup that is quoted, not injected.

Three files need the same two things — text whose only `<skill>` tags are
quotations *about* the injection mechanism, and a way to pin the set of names the
skill library is answering with — and a gate that has to be invisible to three
callers cannot be tested through three private copies of its input.

`QUOTED_MARKUP` carries the three shapes a live re-measure of `usage.db` actually
found on production turns: a doc placeholder (`X`), a format string quoted out of
a writer's own example (`{rule.skill}`), and a name an old synthetic fixture
invented (`alpha`). None of the three is a directory under any root in
`agent_mcp.skills.SKILLS_DIRS`, which is the whole question #2134 asks.
"""

from __future__ import annotations

#: Tag markup quoted as prose rather than rendered by `prefetch._format_context`.
QUOTED_MARKUP = (
    "Notes about the injector, quoting it rather than running it:\n"
    '1. The doc placeholder is `<skill name="X" score="9.9">body</skill>`.\n'
    '2. The writer loop is shown as `<skill name="{rule.skill}" score="2.0">`.\n'
    "3. An old unit-test fixture carried\n"
    '<skill name="alpha" score="3.0">\nnot a protocol, a quotation\n</skill>\n'
)

#: The names `QUOTED_MARKUP` quotes, in the order they appear in it.
PHANTOM_NAMES = ("X", "{rule.skill}", "alpha")


def install_skill_names(monkeypatch, *names: str) -> None:
    """Pin the name-resolution source to exactly `names`.

    `_walk_deliveries` asks `agent_mcp.skills._iter_skills` — the walk the
    turn-start injector itself uses — so replacing that one attribute is what "a
    controlled skill library" means here. `_NAMES_CACHE` keys on the callable it
    was built from, which is why swapping it is enough on its own: no caller
    clears a cache and no test reaches into `_NAMES_CACHE`.
    """
    from agent_mcp import skills as mcp_skills

    monkeypatch.setattr(mcp_skills, "_iter_skills",
                        lambda: [{"name": n} for n in names])
