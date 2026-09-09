"""One definition of "a backlog task's `tags` is a list of strings".

Four programs read that field — `app/routers/backlog.py` for the Mission
Control board, `agent_mcp/backlog.py` for the `backlog_*` tools,
`scripts/autoimplement/backlog.py` for the triage/implement loop, and the React
`TaskCard` — and before this each coerced it privately, or not at all.

It lives in `app/` and imports only the standard library for the same reason
`lint_findings` does: `scripts/autoimplement/backlog.py` is the light module the
autoimplement CLI loads, and it must not pull `mcp` and `httpx` in behind a
fifteen-line helper.
"""

from __future__ import annotations

from typing import Any


def normalize_tags(value: Any) -> list[str]:
    """Coerce a frontmatter `tags` field to a list of strings.

    Frontmatter is written by models and by hand, so `tags` arrives in
    every shape YAML can express it in: a block list, an inline list, a
    bare word, and — when a model answers the schema's array with prose —
    a *string* that merely looks like a list:

        tags: '[youtube-eval, ai-engineer, eval, retrieval, memory]'

    Four of those are on the board today, all filed by `youtube_digest`.
    A string where a list belongs is not a cosmetic problem, because
    every reader degrades differently and none of them says so:
    `BacklogPage`'s `task.tags.map` throws and blanks the *whole* lloyd
    board for one bad row; `backlog._item_from_fm` iterates the string
    and gets 47 single-character tags, so `is_quarantined` — the gate
    that keeps the triage queue from doubling — silently stops matching;
    `_handle_tasks` wrapped the string in a list, which reads as a fix
    and means `tag="youtube-eval"` matches none of the four items that
    carry it. Hence one definition, imported by all of them.

    Order is preserved and duplicates are dropped: `tags` is rendered
    with the tag as the React key, so a repeat is a key collision.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw = [v if isinstance(v, str) else str(v) for v in value]
    elif isinstance(value, str):
        raw = _split_tag_string(value)
    else:
        raw = [str(value)]
    out: list[str] = []
    for item in raw:
        tag = item.strip().strip("'\"").strip()
        if tag and tag not in out:
            out.append(tag)
    return out


def _split_tag_string(value: str) -> list[str]:
    """Split a `tags` scalar that was meant to be a list.

    Handles `[a, b, c]`, `a, b, c` and the plain single tag `a`. A tag
    never legitimately contains a comma or a bracket, so splitting on
    them cannot damage a well-formed value.
    """
    text = value.strip()
    # The quote comes off first: YAML hands back `[a, b]` for the scalar
    # `'[a, b]'`, but a caller reading the raw frontmatter line sees the
    # quotes, and checking for the bracket before stripping them leaves
    # `'[a` as the first tag.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1].strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if "," not in text:
        return [text]
    return text.split(",")
