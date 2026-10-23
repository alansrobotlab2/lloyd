"""The one route a research card may be written to, and who enforces it.

Written for #3190, whose card carried ``type: knowledge``, ``pipeline: research``
and ``source: research-topic-1937`` at
``projects/ecep/cpp-adaptive-constraints-predictive-archery-blocks.md`` — a topic
id above the registry's maximum (129 on 2026-10-23, so topic 1937 has never
existed), readable for under two hours, then gone: never committed, no registry
row, and no tripwire that could see a one-file loss
(``agent-services/guardian/vaultwatch.py`` needs ``DROP_MIN = 200`` files in 900
seconds or a ``TOPDIR_MIN_FILES = 20`` folder emptied).

**Why a route and not a label.** ``workers/sources/deep_research.py::note_path_for``
stamps the artifact path into the job's payload at enqueue, and
``ResearchSource.execute`` requires the worker to hand that path back to
``research_complete``, where ``app/research_store.py::_require_real_note`` refuses
any ``artifact_path`` that is not a real note of at least ``MIN_NOTE_BYTES``. A
card that reaches the vault anywhere else has no row, so the on-disk check never
runs on it, and its disappearance is unreportable — which is exactly the incident.
So ``pipeline: research`` in a card's front matter is a claim about where it must
live, and the one directory that satisfies it is ``knowledge/research/``.

**Why this module and not ``okf_taxonomy``.** #2418's standing ruling froze
``scripts/vault/okf_taxonomy.py``'s code — its comment text is free, its code is
not — and ``tests/test_okf_domain_taxonomy.py::test_the_ruling_rewrote_comment_text_and_nothing_else``
holds it against an AST snapshot taken at ``7ec50f98`` until a human rules on its
retirement. This rule is also not a type rule: the lost card's ``type: knowledge``
is a legal non-research value that ``normalize_type`` folds to ``reference``, so a
guard keyed on ``type`` would have passed that exact write. It reads the
``pipeline:`` axis instead, and borrows the two primitives it needs from
``okf_taxonomy`` rather than restating them, so casing and separators normalise
one way on the box.

**Who asks it.** Both surfaces that can put a vault note on disk:
``agent_mcp/vault.py::_guard_research_route`` (``vault_write``) and
``agent_mcp/builtin_fs.py::_research_route_refusal`` (``Write``/``Edit`` — the lane
the #3190 card actually used). One refusal shape on both, and both ask *before*
creating any directory, so a refusal creates nothing. The exemption below is what
keeps this from being a wall: the worker's own sanctioned writer lands on the
route, and an ordinary turn may not.
"""

from __future__ import annotations

from scripts.vault.okf_taxonomy import _FRONTMATTER_RE, _lookup_form

#: The directory the registry hands the worker, and the only one a card declaring
#: ``pipeline: research`` may land in. One literal; ``note_path_for`` builds it.
RESEARCH_NOTE_DIR = "knowledge/research"

#: Front-matter ``pipeline:`` values that name a ROUTE rather than a job. Every
#: other value on the box (``nightly-reflection`` and friends — 67 files carry a
#: ``pipeline:`` key as of 2026-10-23) names the autonomy task that wrote the file
#: and says nothing about where it belongs, so it is nobody's business here. A
#: second routed pipeline joins this set and both lanes follow with no new code.
ROUTED_PIPELINES = frozenset({"research"})


def research_pipeline_declared(text: str) -> str | None:
    """The routed pipeline this note's front matter declares, or None.

    Reads the front-matter block only, which is what keeps the one vault file
    containing the literal ``pipeline: research`` today — backlog item #3190's own
    body, quoting the lost card's front matter in prose — writable by the lane that
    maintains the board. A *top-level* key only: an indented ``pipeline:`` belongs to
    a nested mapping, and a value is matched after ``_lookup_form``, so casing
    ("Research") is caught and a prefix is not (``research-deep`` names a ``type``
    spelling, not this route).
    """
    match = _FRONTMATTER_RE.match(text or "")
    if match is None:
        return None
    for line in match.group(1).splitlines():
        if line[:1].isspace() or not line.lstrip().startswith("pipeline:"):
            continue
        value = _lookup_form(line.lstrip().split(":", 1)[1])
        return value if value in ROUTED_PIPELINES else None
    return None


def research_route_refusal(path: str, text: str) -> str | None:
    """The routed pipeline ``path`` may NOT hold — refuse, or None to write it.

    ``path`` is vault-relative with forward slashes on every lane: ``vault.py``
    matches the string ``_normalize_vault_path`` hands it, and
    ``builtin_fs._research_route_refusal`` realpaths the target against the vault
    root first and skips it when it is not under there at all. Each lane owns only
    that path-shape translation — the rule and its message are this one function's,
    which is what keeps the two from disagreeing about one card.

    The comparison is against the directory plus a separator, not the bare prefix:
    ``knowledge/research-notes/x.md`` is a different directory and is off the route.
    """
    declared = research_pipeline_declared(text)
    if declared is None:
        return None
    normalised = (path or "").strip().lstrip("./").rstrip("/")
    if normalised.startswith(RESEARCH_NOTE_DIR + "/"):
        return None
    return declared
