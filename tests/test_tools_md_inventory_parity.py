"""`architecture/tools.md` §6 is that page's only complete inventory of the tool
surface — 147 tools across 29 modules as measured on 2026-10-21, from the page's own
rows and from a token-bearing `tools/list` against the running aggregator, which agree
name for name — and §11 of the same page instructs "**Regenerate this inventory; do not
edit it by hand.**" Nothing enforced either half of that sentence until #3126.

What existed before was two guards that could not see the hole.
`tests/test_code_graph_doc_claims.py::test_every_graph_name_in_claude_md_and_tools_md_exists`
walks doc → catalog over the `graph_*` namespace only, so a row that goes missing from
§6 is invisible to it, and `tests/test_memory_improvement.py::_tools_md_row` pins one
row. Nothing asked that every advertised tool have a row, or that the three places §6
states a module's tool count — the `#### `module` (n)` heading, the rows under it, and
that module's row in the §6 Modules table — be the same number. So §6 drifted and stayed
green: at `b316b5b9` the `#### `autonomy` (7)` heading sat over 8 rows while the Modules
table said 8 and the live module had 8; `facts` read 9 in the table and 7 in its own
heading and rows; `desktop` and `recall_observation` were in `MODULES` and in the live
surface with no row anywhere; `Scratchpad` had a table row and a §5 discussion but no row
in "Every tool". `a5349cc2` ("arch-review: doc:tools — stale") corrected all of it. This
file is the reason the correction has to stick.

Design decisions worth their keep:

- **The parser takes the page text as an argument** (the shape
  `tests/test_architecture_index_parity.py` uses), so the planted-defect nodes can feed
  a perturbed copy without touching the tracked page. Green-and-failing-on-a-planted-defect
  is the only pair that separates this guard from a vacuous one: the page carried three
  different module counts — 26, 27 and 29 — across its prose for a month.
- **Zero-export modules drop out of BOTH sides.** The Thunderbird bridge
  (`agent-services/services/thunderbird-mcp/mcp-bridge.cjs`) is gitignored, so
  `thunderbird` exports 0 tools in any automod worktree while §6 still carries its 40
  rows. Exempting the served side alone would leave those 40 rows reported as unserved,
  and the guard would be red in every round — the `test_tool_overrides.py` failure mode
  [[testing]] calls a test about untracked state. Degradation is detected exactly the way
  `tests/test_mcp_layer.py::_unverifiable_names` detects it: `M._discovery_status`,
  degraded when `not ok or not tools`. A module that DOES export here is compared in
  full, so a stale §6 row for a loaded module still fails.
- **The parser refuses rather than returning empty.** A section header renamed, and every
  set difference is clean because every set is empty; each extract raises naming what it
  could not find, and the parity nodes additionally assert they parsed 29 modules and a
  non-zero number of rows.

The last node runs §11's regeneration snippet in a child interpreter and reads its
totals line, which is how the four prose copies of the count — the front-matter
`summary:`, the intro's "mounts 29 Python modules", the §intro count line and the §1
diagram — get refreshed from the same measurement that regenerates §6 instead of being
retyped, which is what let them read 26 / 27 / 29 at once.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_mcp import main as M  # noqa: E402
from agent_mcp import thunderbird as TB  # noqa: E402

ARCH_DOC = ROOT / "architecture" / "tools.md"

#: §6 in full, up to the `## 7.` heading that closes it.
_SECTION6 = re.compile(r"^## 6\. Inventory\n(.*?)^## 7\. ", re.S | re.M)
#: `### Every tool` splits the inventory in two: the hand-typed Modules table above it,
#: the per-module tool tables below it. Parsing each half only inside its own region is
#: what keeps `| `grant_list` | RO† FX | — |` from being read as a Modules row.
_EVERY_TOOL = "### Every tool"
#: `| `facts` | 7 | the knowledge-graph fact layer …`
_MODULES_ROW = re.compile(r"^\| `([a-z_]+)` \| (\d+) \|", re.M)
#: `#### `facts` (7)` — anchored, so a heading carrying stray text is not silently skipped.
_HEADING = re.compile(r"^#### `([a-z_]+)` \((\d+)\)$", re.M)
#: The first cell of a tool row. Separator rows (`|---|---|`) never match.
_TOOL_ROW = re.compile(r"^\| `([^`]+)` \|", re.M)


def _section6(page_text: str) -> str:
    m = _SECTION6.search(page_text)
    if not m:
        raise LookupError(
            "architecture/tools.md has no `## 6. Inventory` section running to a `## 7.` "
            "heading, so a parse of it would compare two empty sets and read as parity")
    return m.group(1)


def _modules_table(sec6: str) -> dict[str, int]:
    """The `### Modules` table: module -> the `n` the page's hand-typed row declares."""
    head, _sep, _rest = sec6.partition(_EVERY_TOOL)
    table = dict(_MODULES_ROW.findall(head))
    if not table:
        raise LookupError(
            "no `| `module` | n | …` row parsed out of the §6 Modules table, so every "
            "count comparison below would be over nothing")
    return {name: int(n) for name, n in table.items()}


def _tool_sections(sec6: str) -> dict[str, dict]:
    """The `### Every tool` part: module -> declared heading `n`, and its row names."""
    _head, sep, every = sec6.partition(_EVERY_TOOL)
    if not sep:
        raise LookupError(f"architecture/tools.md §6 has no {_EVERY_TOOL!r} subsection, "
                          "so there are no per-module tool tables to parse")
    heads = list(_HEADING.finditer(every))
    if not heads:
        raise LookupError(
            "no `#### `module` (n)` heading parsed out of §6 'Every tool', so no declared "
            "count could be checked")
    out: dict[str, dict] = {}
    for i, hm in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(every)
        name = hm.group(1)
        rows = _TOOL_ROW.findall(every[hm.end():end])
        if name in out:
            raise LookupError(f"§6 carries two `#### `{name}` sections")
        out[name] = {"declared": int(hm.group(2)), "rows": rows, "table": None}
    return out


def _inventory(page_text: str) -> dict[str, dict]:
    """Parse §6 into module -> heading-declared `n`, row names, and Modules-table `n`."""
    sec6 = _section6(page_text)
    inv = _tool_sections(sec6)
    table = _modules_table(sec6)
    for name, entry in inv.items():
        entry["table"] = table.get(name)
    return inv


def _served(tools) -> dict[str, list[str]]:
    """The surface as the aggregator actually builds it: module -> advertised names.

    Module attribution is `M._dispatch[name].__name__`, the same expression §11's
    regeneration snippet prints, so the test and the page read the one mapping.
    """
    out: dict[str, list[str]] = {}
    for t in tools:
        out.setdefault(M._dispatch[t.name].__name__.rsplit(".", 1)[-1], []).append(t.name)
    return {mod: sorted(names) for mod, names in out.items()}


def _zero_export_modules() -> set[str]:
    """Modules that export nothing in THIS checkout, and so cannot be compared at all.

    The Thunderbird bridge is a gitignored build artifact, so `thunderbird` exports zero
    tools in any worktree — and the self-modification gate runs this suite inside one —
    while `architecture/tools.md` §6 keeps its 40 rows, because in the live checkout
    with Thunderbird running it is a real 40-tool module. Detection is
    `tests/test_mcp_layer.py::_unverifiable_names`'s, down to the predicate: a module is
    degraded when its `/health` discovery entry is not `ok` or carries no tools.
    """
    return {name for name, st in M._discovery_status.items()
            if not st.get("ok") or not st.get("tools")}


def _module_names() -> set[str]:
    return {mod.__name__.rsplit(".", 1)[-1] for mod in M.MODULES}


def _unrowed(served: dict[str, list[str]], inv: dict[str, dict]) -> list[str]:
    """Served tools with no §6 row, and §6 sections for modules that serve nothing."""
    out = []
    for mod in sorted(set(served) - set(inv)):
        out.append(f"{mod}: the module serves {len(served[mod])} tools "
                   f"({', '.join(served[mod][:3])}…) and §6 has no `#### `{mod}`` section")
    for mod in sorted(set(inv) - set(served)):
        out.append(f"{mod}: §6 has a section with {len(inv[mod]['rows'])} rows and no such "
                   f"module is served — the modules `MODULES` carries are "
                   f"{', '.join(sorted(_module_names()))[:200]}")
    for mod in sorted(set(inv) & set(served)):
        missing = sorted(set(served[mod]) - set(inv[mod]["rows"]))
        unserved = sorted(set(inv[mod]["rows"]) - set(served[mod]))
        if missing:
            out.append(f"{mod}: §6 has no row for {missing}, all served by that module")
        if unserved:
            out.append(f"{mod}: §6 has rows for {unserved}, which that module does not serve")
        dupes = sorted(n for n, c in Counter(inv[mod]["rows"]).items() if c > 1)
        if dupes:
            out.append(f"{mod}: §6 rows {dupes} twice")
    return out


def _count_defects(inv: dict[str, dict]) -> list[str]:
    """Every place §6 states a per-module count, against the rows it counts."""
    out = []
    for mod, entry in sorted(inv.items()):
        rows = len(entry["rows"])
        if entry["declared"] != rows:
            out.append(f"{mod}: its `#### `{mod}` ({entry['declared']})` heading does not "
                       f"match the {rows} rows under it")
        if entry["table"] != rows:
            out.append(f"{mod}: the §6 Modules table says {entry['table']} and its own "
                       f"section has {rows} rows")
    return out


def _inventory_defects(page_text: str, served: dict[str, list[str]],
                       exempt: set[str] | frozenset[str] = frozenset()) -> list[str]:
    """All the ways one page's inventory can disagree with one dispatch table.

    `exempt` drops a module from BOTH sides: comparing it is not possible in this
    checkout, and half-comparing it is how a guard ends up red in every worktree.
    """
    inv = {m: e for m, e in _inventory(page_text).items() if m not in exempt}
    served = {m: names for m, names in served.items() if m not in exempt}
    return _unrowed(served, inv) + _count_defects(inv)


@pytest.fixture(scope="module")
def tools():
    return asyncio.run(M.list_tools())


@pytest.fixture(scope="module")
def served(tools) -> dict[str, list[str]]:
    # `list_tools()` above is what populates `M._discovery_status`; anything that wants
    # the degradation state has to depend on this fixture, not just call the helper.
    return _served(tools)


@pytest.fixture(scope="module")
def page_text() -> str:
    return ARCH_DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def exempt(served) -> set[str]:
    # `served` is a dependency and not decoration: `_discovery_status` is only filled by
    # a discovery pass, so an `exempt` built before one reads `{}` — which is how this
    # fixture first shipped red, with 147 doc rows compared against 107 served ones.
    _ = served
    return _zero_export_modules()


# ── The parse itself ─────────────────────────────────────────────────────────

def test_the_page_parses_into_two_inventories_over_the_same_modules(page_text):
    """Anti-vacuity for the extracts, before anything is compared.

    Both halves have to come back populated and cover the same modules, or the parity
    assertions below would be comparing empty sets and reporting them as equal. The
    Modules table's module *names* are checked against `M.MODULES` — the list that owns
    them — rather than against the served surface, because that list is static and so a
    30th module gets reported here even in a worktree where it exports nothing.
    """
    inv = _inventory(page_text)
    assert inv, "§6 parsed to no modules at all"
    assert all(e["rows"] for e in inv.values()), (
        f"sections with zero parsed rows: "
        f"{[m for m, e in inv.items() if not e['rows']]}")
    assert set(inv) == _module_names(), (
        f"§6's modules and `agent_mcp.main.MODULES` disagree: "
        f"doc-only {sorted(set(inv) - _module_names())}, "
        f"MODULES-only {sorted(_module_names() - set(inv))}")
    assert all(e["table"] is not None for e in inv.values()), (
        f"sections with no Modules-table row: "
        f"{[m for m, e in inv.items() if e['table'] is None]}")
def test_the_row_total_covers_every_served_tool(page_text, served, exempt):
    """The count-level version of parity, so a whole missing section cannot hide.

    Per-module diffs are what name the culprit, and they are also what a
    dropped-heading parse cannot produce: read §6 without the last section's rows and
    each remaining module still agrees with itself. The totals are the second lock.
    """
    inv = _inventory(page_text)
    doc_rows = sum(len(e["rows"]) for m, e in inv.items() if m not in exempt)
    served_rows = sum(len(names) for m, names in served.items() if m not in exempt)
    assert doc_rows == served_rows, (
        f"§6 has {doc_rows} tool rows outside the exempt modules and the aggregator "
        f"serves {served_rows} there")


# ── Clause 1: the tool-name mapping, both directions ────────────────────────

def test_every_served_tool_has_a_row_and_every_row_names_a_served_tool(page_text, served,
                                                                       exempt):
    """§6's rows and the advertised surface are the same 147 names, per module.

    Both directions are asserted, because they fail for opposite reasons: a tool added
    to a module with no §6 row is a missing row, and a §6 row whose tool was retired is
    a promise the page still makes. `_unrowed` reports each with its module named.
    """
    assert _unrowed({m: n for m, n in served.items() if m not in exempt},
                    {m: e for m, e in _inventory(page_text).items() if m not in exempt}) == []


# ── Clause 2: the declared counts ───────────────────────────────────────────

def test_every_declared_count_matches_its_rows_and_the_modules_table(page_text, exempt):
    """Each `#### `module` (n)` equals its own rows AND its Modules-table `n`.

    §6 states a module's tool count in three places 30 lines apart, and #3126 caught it
    disagreeing with itself: `#### `autonomy` (7)` over 8 rows with the table saying 8,
    and `facts` typed 9 in the table beside its own 7. Every defect string names the
    module, because "a count is wrong somewhere in a 344-line section" is not a fixable
    failure.
    """
    inv = {m: e for m, e in _inventory(page_text).items() if m not in exempt}
    assert _count_defects(inv) == []


# ── Clause 3: the zero-export exemption ─────────────────────────────────────

def test_a_module_that_exports_nothing_here_is_dropped_from_both_sides(page_text, served,
                                                                      exempt):
    """The worktree case, pinned rather than assumed.

    `thunderbird` carries 40 rows in §6 and exports 0 tools in this checkout, so the
    only green reading available here is one that drops the module from BOTH sides. This
    node is that reading, and it is not a tautology: it asserts the exemption is
    populated (a `frozenset()` of exemptions would make every other node's green look
    like this one's), that the page really does carry rows for it, and that the same
    page then compares clean.
    """
    inv = _inventory(page_text)
    assert exempt == _zero_export_modules(), "the fixture and the helper must agree"
    unloaded = [m for m in exempt if inv.get(m) and inv[m]["rows"]]
    assert unloaded, (
        f"no exempt module has §6 rows ({sorted(exempt)}), so this node stopped "
        "exercising the both-sides drop — in the live checkout thunderbird exports 40 "
        "tools and is not exempt at all, which is the other way this can quietly pass")
    assert _inventory_defects(page_text, served, exempt) == []


def test_a_module_that_exports_something_is_compared_in_full(page_text, served, exempt):
    """The exemption is scoped to what cannot be checked, not to a name.

    `tests/test_mcp_layer.py::_unverifiable_names` carries the same discipline: "a stale
    entry for a module that IS loaded still fails". Hence the control — put one bogus
    tool on thunderbird's served side and the 40 rows §6 promises for it are compared
    again, and disagree.
    """
    assert exempt, "no module is degraded in this checkout, so the control below is vacuous"
    bogus = dict(served)
    for mod in sorted(exempt):
        bogus[mod] = ["_NotATool"]
    defects = _inventory_defects(page_text, bogus, exempt=frozenset())
    assert defects, (
        "a bogus tool on a degraded module's served side found no disagreement, so the "
        "doc side is being skipped wholesale rather than per-module")
    assert any(mod in "".join(defects) for mod in exempt), (
        f"the disagreement does not name the module whose rows it rejected: {defects[:3]}")


# ── Clause 4: planted defects, so the green means something ─────────────────

def _planted(page_text: str, old: str, new: str) -> str:
    """One exact substitution into the page text, refusing if it would not land."""
    assert page_text.count(old) == 1, f"a planted defect must match exactly once: {old!r}"
    return page_text.replace(old, new)


def test_the_parity_check_can_see_a_planted_heading_count(page_text, served, exempt):
    """`#### `facts` (7)` → `(8)`, over rows that stay 7."""
    perturbed = _planted(page_text, "#### `facts` (7)", "#### `facts` (8)")
    defects = _inventory_defects(perturbed, served, exempt)
    assert defects, "a heading count off by one was not reported at all"
    facts = [d for d in defects if d.startswith("facts:")]
    assert facts, f"the defect did not name `facts`: {defects[:3]}"
    assert any("heading" in d for d in facts), facts


def test_the_parity_check_can_see_a_deleted_tool_row(page_text, served, exempt):
    """One row deleted from a section, which two separate assertions must catch."""
    row = next(ln for ln in page_text.splitlines()
               if ln.startswith("| `Grep` |"))
    perturbed = page_text.replace(row + "\n", "", 1)
    assert perturbed != page_text, "the planted row was not found"
    defects = _inventory_defects(perturbed, served, exempt)
    fs = [d for d in defects if d.startswith("builtin_fs:")]
    assert fs, f"deleting `Grep` reported nothing about its module: {defects[:3]}"
    assert any("Grep" in d for d in fs), f"the missing-row report did not name `Grep`: {fs}"
    assert any("heading" in d for d in fs), (
        f"5 rows became 4 and the `(5)` heading said nothing: {fs}")


def test_the_parity_check_can_see_a_modules_table_count(page_text, served, exempt):
    """The Modules table's `n` for `vault` changed while heading and rows stay 5."""
    perturbed = _planted(page_text, "| `vault` | 5 |", "| `vault` | 6 |")
    defects = _inventory_defects(perturbed, served, exempt)
    vault = [d for d in defects if d.startswith("vault:")]
    assert vault, f"a changed Modules-table count was not reported about `vault`: {defects[:3]}"
    assert any("Modules table" in d for d in vault), vault


def test_the_parity_check_can_see_a_row_for_a_tool_that_is_not_served(page_text, served,
                                                                     exempt):
    """The other direction of clause 1, planted: a row the page invents.

    Without this node a parser that quietly failed to read §6's rows would still pass
    the missing-row half of the parity check, because an empty doc set has no
    differences to report.
    """
    perturbed = page_text.replace("#### `facts` (7)", "#### `facts` (8)\n\n"
                                 "| `fact_invented` | RO | — | Made up by a test |", 1)
    defects = _inventory_defects(perturbed, served, exempt)
    facts = [d for d in defects if d.startswith("facts:")]
    assert facts, "an invented row was reported by no module"
    assert any("fact_invented" in d and "does not serve" in d for d in facts), facts


# ── Clause 5: §11's regeneration snippet prints the totals ─────────────────

_FENCE = re.compile(r"^```bash\n(.*?)^```$\n?", re.S | re.M)
_TOTALS = re.compile(r"^TOTALS\ttools=(\d+)\tmodules=(\d+)$", re.M)


def _regeneration_block(page_text: str) -> str:
    """§11's offline dump: the one fenced bash block in that section calling
    `list_tools()`."""
    i11 = page_text.index("## 11. Regenerating the inventory")
    tail = page_text[i11:page_text.index("## Review log", i11)]
    blocks = [b for b in _FENCE.findall(tail) if "list_tools()" in b]
    if len(blocks) != 1:
        raise LookupError(
            f"§11 has {len(blocks)} fenced blocks calling `list_tools()`, expected 1, so "
            "the totals node below would be grading the wrong block or none")
    return blocks[0]


def test_the_regen_snippet_prints_the_tool_and_module_totals(page_text, tools, served):
    """Run the page's own regeneration command and read its totals line.

    §11 tells a reader to regenerate §6 from this snippet, and until #3126 the snippet
    printed one line per tool and no total — so the four places the page states "147
    tools across 29 modules" (the front-matter `summary:`, the intro's module count, the
    §intro count line and the §1 diagram) were retyped from a different measurement than
    the table they sit beside, which is how they came to read 26, 27 and 29 at once. The
    snippet now prints `TOTALS\\ttools=N\\tmodules=K`, and this runs it in a child
    interpreter — `.venvs/lloyd/bin/python` replaced by `sys.executable`, because
    `.venvs/` is gitignored and absent from a worktree, the same substitution
    `tests/test_architecture_coverage_doc_claims.py` makes for the page it runs — and
    holds the printed figures against the same dispatch table the rest of this file
    compares. In a worktree that is 107 across 28, because Thunderbird is not there;
    what is pinned is that the snippet's total and the aggregator's agree, not a number.
    """
    block = _regeneration_block(page_text)
    runnable = block.replace(".venvs/lloyd/bin/python", sys.executable)
    proc = subprocess.run(runnable, shell=True, cwd=ROOT, capture_output=True,
                          text=True, timeout=600,
                          env=dict(os.environ, PYTHONPATH=str(ROOT)))
    out = proc.stdout
    assert proc.returncode == 0, (
        f"§11's regeneration snippet exited {proc.returncode}: "
        f"{(proc.stdout + proc.stderr)[-1500:]}")
    assert "Traceback" not in out and "No module named" not in out, out[-1500:]
    printed = _TOTALS.findall(out)
    assert len(printed) == 1, (
        f"the snippet printed {len(printed)} totals lines; §11 promises exactly one, and "
        f"a page whose regeneration prints the number twice has two of them to keep in "
        f"step. Tail: {out[-300:]!r}")
    tools_out, modules_out = (int(a) for a in printed[0])
    assert tools_out == len(tools), (
        f"the snippet printed tools={tools_out} for a surface listing {len(tools)}")
    assert modules_out == len(served), (
        f"the snippet printed modules={modules_out} for a dispatch table attributing "
        f"tools to {len(served)} modules")
    assert out.count("\n") > tools_out, (
        "the snippet printed its totals but stopped printing the per-tool rows §6 is "
        "made of, which leaves the regeneration with nothing to write")


# ── The Thunderbird bridge's own 49 rows (#4433) ─────────────────────────────
#
# Every node above drops the bridge on BOTH sides when it exports nothing here
# (`_zero_export_modules`), which is the case in every automod worktree: the bridge
# is gitignored, so `thunderbird` serves zero tools and the wire says nothing about
# its names. That exemption is what let §6 carry nine rows under
# `tb_getMessageHeaders` … `tb_dryRunCompose` while every parity node read green —
# a regeneration that trusted §6 would have re-shipped the mismatch. These nodes
# read the page against the map in git instead of against the wire, so they are
# non-vacuous exactly where the wire nodes are blind.

#: The five reads among the nine tools that arrived after the map's first 40, and
#: the one delete. These are the classification facts §6 has to carry, not a second
#: copy of the tables: `agent_mcp/annotations.py` holds the truth and
#: `tests/test_mcp_layer.py` pins it — a §6 that disagrees with it misleads the next
#: regeneration, which copies its marks out of this page.
_BRIDGE_READS = {
    "email_message_headers", "email_message_headers_batch",
    "email_attachment_search", "email_virtual_folders", "email_compose_dry_run",
}
_BRIDGE_DELETE = "email_virtual_folder_delete"
_BRIDGE_WRITERS = {
    "email_virtual_folder_create", "email_folder_refresh", "calendar_refresh",
    _BRIDGE_DELETE,
}

_TB_ROW = re.compile(r"^\| `([^`]+)` \| ([^|]+)\|", re.M)
_SECTION11 = re.compile(r"^## 11\. Regenerating the inventory\n(.*?)^## Review log",
                        re.S | re.M)
#: Each §11 sentence that states how many bridge rows a dump prints, serves or
#: holds — what a regeneration pass copies §6's counts from — plus the `bridge=`
#: total it copies into the totals line.
_BRIDGE_ROW_COUNT = re.compile(r"\b(?:prints|serves|holds)\s+all\s+(\d+)\b")
_BRIDGE_TOTAL = re.compile(r"bridge=`?\s*(\d+)")


def _bridge_marks(page_text: str) -> dict[str, str]:
    """Every tool row on the page as name -> its marks cell."""
    marks = {hit.group(1): hit.group(2).strip()
             for hit in (_TB_ROW.match(line) for line in page_text.splitlines())
             if hit}
    if not marks:
        raise LookupError("no `| `tool` | marks | …` row parsed out of the page, so "
                          "every check below would be over nothing")
    return marks


def _section11(page_text: str) -> str:
    hit = _SECTION11.search(page_text)
    if not hit:
        raise LookupError("architecture/tools.md has no `## 11. Regenerating the "
                          "inventory` section running to the Review log")
    return hit.group(1)


def _bridge_inventory_defects(page_text: str) -> list[str]:
    """Where §6's bridge rows and §11's counts have drifted from the map in git."""
    declared = _tool_sections(_section6(page_text))["thunderbird"]["declared"]
    marks = _bridge_marks(page_text)
    mapped = set(TB.TOOL_NAME_MAP.values())
    defects: list[str] = []
    strays = sorted(n for n in marks if n.startswith("tb_"))
    if strays:
        defects.append(
            f"§6 inventories {len(strays)} tool(s) under the bridge's own names instead "
            f"of the mapped ones §11's regeneration block prints: {strays}")
    missing = sorted(mapped - set(marks))
    if missing:
        defects.append(f"{len(missing)} mapped bridge tool(s) have no §6 row: {missing}")
    if declared != len(mapped):
        defects.append(
            f"the `#### `thunderbird` ({declared})` heading declares {declared} while "
            f"the map carries {len(mapped)} names — a regeneration copies its row count "
            "from that heading")
    for name in sorted(_BRIDGE_READS):
        cell = marks.get(name, "")
        if "RO" not in cell:
            defects.append(f"{name} is a pure read and §6 marks it {cell!r}, no `RO`")
    for name in sorted(_BRIDGE_WRITERS):
        cell = marks.get(name, "")
        if "RO" in cell:
            defects.append(f"{name} changes state and §6 marks it {cell!r} as a read")
    delete_cell = marks.get(_BRIDGE_DELETE, "")
    if not {"DX", "ID"} <= set(delete_cell.split()):
        defects.append(
            f"{_BRIDGE_DELETE} destroys a saved-search view and §6 marks it "
            f"{delete_cell!r} — `email_delete_folder` beside it carries `DX ID`")
    no_ow = sorted(n for n in mapped if n in marks and "OW" not in marks[n])
    if no_ow:
        defects.append(
            f"{len(no_ow)} bridge tool(s) §6 marks closed-world, which none of them is "
            f"— every one dials out through the bridge: {no_ow}")
    sec11 = _section11(page_text)
    for pattern in (_BRIDGE_ROW_COUNT, _BRIDGE_TOTAL):
        for stated in pattern.findall(sec11):
            if int(stated) != len(mapped):
                defects.append(
                    f"§11 states {stated} as the bridge's row count while the map "
                    "carries "
                    f"{len(mapped)} names; a regeneration that trusts it silently "
                    "freezes or drops the difference")
    return defects


def test_the_bridges_six_section_rows_are_the_mapped_names_with_their_marks(page_text):
    """Clause 5: §6 carries all 49 bridge tools under their Lloyd names, each with
    the marks the annotation tables give it."""
    assert _bridge_inventory_defects(page_text) == []


def test_a_row_left_on_the_bridges_own_name_is_reported(page_text):
    """The node above can fail: the §6 state #4433 was filed against, planted back.

    `tb_deleteVirtualFolder` in place of `email_virtual_folder_delete` is exactly
    what shipped on 2026-11-20 — an unbadged delete, under a name the regeneration
    block will never print.
    """
    planted = _planted(page_text,
                       "| `email_virtual_folder_delete` | DX ID OW | `folderPath` |",
                       "| `tb_deleteVirtualFolder` | FX | `folderPath` |")
    defects = _bridge_inventory_defects(planted)
    assert any("tb_deleteVirtualFolder" in d for d in defects), defects
    assert any(_BRIDGE_DELETE in d and "DX ID" in d for d in defects), defects


def test_section11_states_the_row_count_the_map_carries(page_text):
    """§11's row-count sentences and its `bridge=` total are what a regeneration
    copies §6's counts from, so each one must state the map's 49."""
    stated = (_BRIDGE_ROW_COUNT.findall(_section11(page_text))
              + _BRIDGE_TOTAL.findall(_section11(page_text)))
    assert len(stated) >= 4, (
        f"§11 names the bridge's row count {len(stated)} time(s); it documents two "
        "dumps and a total, and a section that stopped stating the number would leave "
        "this node with nothing to check")
    assert sorted(set(stated)) == [str(len(set(TB.TOOL_NAME_MAP.values())))], stated


def test_a_stale_row_count_in_section11_is_reported(page_text):
    """The revival this guards: the page saying `all 40 rows` while the map and the
    server both carry 49."""
    planted = _planted(page_text, "The dump below prints all 49",
                       "The dump below prints all 40")
    assert any("40" in d for d in _bridge_inventory_defects(planted)), (
        "§11 could go back to stating 40 while the map carries 49 and the "
        "regeneration would freeze nine rows without a sound")
