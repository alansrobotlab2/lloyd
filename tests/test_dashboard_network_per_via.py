"""#4292: the dashboard's egress panel says which arm its headline counts.

`agent_mcp.egress.network_report` has carried `per_via` since #4044, and the typed
reader in `web/src/api.ts` declared neither that key nor `policy.bash_record` — so the
Network panel's headline (`494 calls · 89 hosts · 464 allow · …`) was a
`via='tool'`-only denominator wearing no label. It is tool-only by construction, not by
accident of the window: `total`, `hosts` and `by_decision` are computed with
`via = VIA_TOOL` bound (`egress.py:1311-1312`), and `per_destination` / `per_scope`
likewise, as that module's own comment at `:1263` states. The reader who has to make
the `harness.egress_policy.enforce` call — #4294's clause (4) waits on this panel —
was choosing on a number whose edge it could not see.

These nodes pin the seam at both ends, because the change crosses a language
boundary and a passing type-check proves nothing about the payload and a passing
render proves nothing about the type:

* `test_policy_block_matches_the_emitters_five_keys` and
  `test_per_via_row_type_is_exactly_the_six_emitted_keys` call the sanctioned
  accessor over a seeded store to get the key sets the Python writer actually emits,
  and compare them to the key sets the TypeScript reader declares, parsed out of
  `web/src/api.ts`. Drift on either side reddens here — which is where a
  `NetworkState` that lies about its payload used to be invisible: TypeScript's
  structural typing accepts a missing property silently on a read of a JSON body.
* `test_panel_renders_a_seeded_bash_proxy_arm` and
  `test_panel_reads_the_bash_arm_as_off_and_zero_without_rows` run the vitest spec
  that renders `NetworkPanel` and assert those nodes passed, then grep the spec for
  the literals it must still hold. The live payload cannot show a non-zero Bash
  count — `config.yaml:838` is `bash_record: false` and #4293 owns arming it — so the
  arm is seeded in the spec, and the live figure is owed after #4293 plus a week of
  Bash traffic.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_mcp import egress  # noqa: E402
from app.lint_findings import node_env  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
API_TS = WEB / "src" / "api.ts"
PANEL_TSX = WEB / "src" / "components" / "pages" / "DashboardPage.tsx"
PANEL_SPEC = WEB / "src" / "components" / "pages" / "DashboardNetworkPanel.test.tsx"

#: The five keys `/api/dashboard` returns under `network.policy`, read live on
#: 2026-11-22 and emitted at `egress.py:1358`.
POLICY_KEYS = {"telemetry", "enforce", "bash_record", "allow_entries",
               "permanent_allow_entries"}

#: The six keys one `per_via` row carries, built at `egress.py:1327-1337`.
VIA_ROW_KEYS = {"via", "count", "distinct_hosts", "denies", "last_at", "scopes"}


def _state_block() -> str:
    """`awk '/export interface NetworkState/,/^}/' web/src/api.ts`, in Python.

    Deliberately the same shape as the item's check: it ends at the first
    column-zero `}`, so a declaration that grows a nested object still yields the
    interface and nothing else. It has to be long — a truncated read is what makes a
    grep-based claim about an interface meaningless.
    """
    src = API_TS.read_text(encoding="utf-8")
    match = re.search(r"export interface NetworkState \{.*?^\}", src, re.S | re.M)
    assert match, "NetworkState not found in web/src/api.ts"
    block = match.group(0)
    assert len(block) > 400, f"interface body looks truncated: {len(block)} chars"
    return block


def _declared_policy_keys(block: str) -> set[str]:
    body = re.search(r"policy:\s*\{(.*?)\}", block, re.S)
    assert body, "NetworkState declares no `policy` object"
    return set(re.findall(r"([A-Za-z_]\w*)\s*:", body.group(1)))


def _declared_via_keys(block: str) -> tuple[set[str], set[str]]:
    """The `per_via` row's keys, and the keys of its nested `scopes` element.

    The nested object is replaced before the key scan: left in, `scope` and `count`
    would be counted as row keys and the set would match a row that does not have
    them.
    """
    head = re.search(r"per_via\??\s*:\s*\{(.*)", block, re.S)
    assert head, "NetworkState declares no `per_via` row"
    body = head.group(1)
    nested = re.search(r"scopes:\s*\{([^}]*)\}", body)
    assert nested, "`per_via` has no nested `scopes` object"
    scopes_keys = set(re.findall(r"([A-Za-z_]\w*)\s*:", nested.group(1)))
    flat = (body[:nested.start()] + "scopes: NESTED" + body[nested.end():])
    row_keys = set(re.findall(r"([A-Za-z_]\w*)\s*:", flat.split("}", 1)[0]))
    return row_keys, scopes_keys


@pytest.fixture
def emitted(tmp_path, monkeypatch):
    """`network_report()` over a store holding one tool row and one Bash-arm row.

    Both go through the sanctioned writers (`_record` is what `guard` calls;
    `record_bash_destination` is the only route by which a Bash destination becomes a
    row), so the key sets compared below are the emitted ones and not a hand-typed
    copy that could agree with the interface by sharing its author's mistake.
    """
    monkeypatch.setenv("LLOYD_EGRESS_DB", str(tmp_path / "workers.db"))
    monkeypatch.setenv("LLOYD_GRANT_DB", str(tmp_path / "workers.db"))
    monkeypatch.setattr(egress, "config",
                        lambda: {"telemetry": True, "enforce": False,
                                 "allow": [], "retention_days": 90})
    egress._record(tool="http_fetch", destination="https://docs.example.org",
                   host="docs.example.org", decision=egress.DECISION_ALLOW,
                   scope="interactive", session_id="s1", worker_source="",
                   task_id="", reason="allowlist", at=egress._now())
    egress.record_bash_destination(destination="https://api.github.com",
                                   host="api.github.com", scope="autonomy-task:54",
                                   session_id="s1")
    return egress.network_report(days=7, limit=50)


def test_policy_block_matches_the_emitters_five_keys(emitted):
    """Clause 1: `policy.bash_record` exists, and the block is the five real keys.

    The item's check is that the `NetworkState` block greps `bash_record`; that is
    necessary but not sufficient — a sixth key invented here would satisfy the grep
    and still mis-describe the payload. So the assertion is set equality against what
    the accessor returned a moment ago.
    """
    block = _state_block()
    assert "bash_record" in block, "NetworkState.policy declares no bash_record"
    assert "bash_record: boolean" in block, (
        "bash_record must be typed `boolean`; egress.py:1358 emits `bash_record_on()`")
    assert set(emitted["policy"]) == POLICY_KEYS, (
        f"the writer emits {sorted(emitted['policy'])}")
    assert _declared_policy_keys(block) == set(emitted["policy"]), (
        "NetworkState.policy's key set has drifted from what network_report() emits")
    assert block.count("per_via") >= 1, "the awk block must also carry per_via"


def test_per_via_row_type_is_exactly_the_six_emitted_keys(emitted):
    """Clause 2: `per_via` is optional and its row has exactly the emitted six keys.

    Optional is load-bearing, not tidiness: a backend older than the #4044 split sends
    no such key, and a required field there is a panel that falls to the section
    boundary over a version skew. `?? []` on the read side is pinned in the panel
    spec; this is the type side of the same fact.
    """
    block = _state_block()
    assert re.search(r"per_via\?:", block), (
        "`per_via` must be optional — an older backend sends no such key")
    row_keys, scope_keys = _declared_via_keys(block)
    assert row_keys == VIA_ROW_KEYS, (
        f"row declares {sorted(row_keys)}, the emit site writes {sorted(VIA_ROW_KEYS)}")
    assert scope_keys == {"scope", "count"}, (
        f"`scopes` element declares {sorted(scope_keys)}")

    rows = emitted["per_via"]
    assert {str(r["via"]) for r in rows} == {egress.VIA_TOOL, egress.VIA_BASH_PROXY}, (
        "the seeded store should have produced one row per arm")
    for row in rows:
        assert set(row) == VIA_ROW_KEYS, f"row emitted {sorted(row)}"
    bash = next(r for r in rows if r["via"] == egress.VIA_BASH_PROXY)
    assert bash["count"] == 1 and bash["distinct_hosts"] == 1 and bash["denies"] == 0, (
        "an arm with one destination must not report a denial against it")
    assert bash["scopes"] == [{"scope": "autonomy-task:54", "count": 1}], (
        "`scopes` is the arm's own attribution (egress.py:1326), not the tool-only "
        "`per_scope`")
    assert bash["last_at"], "an untimestamped arm row cannot be aged"


_PANEL_SPEC_RESULTS: dict[str, str] | None = None


def _panel_spec_results() -> dict[str, str]:
    """Run the panel's vitest spec once and return {test title: status}.

    Same loud rule as `test_dashboard_doc_claims.test_the_section_error_helpers_have_a
    spec_the_suite_actually_runs`, for the same reason: vitest is advisory at the gate,
    so a missing runner here would otherwise reduce the panel's only pin to a grep.
    Module-scoped and cached because one run covers both clause nodes.
    """
    global _PANEL_SPEC_RESULTS
    if _PANEL_SPEC_RESULTS is not None:
        return _PANEL_SPEC_RESULTS
    vitest = WEB / "node_modules" / ".bin" / "vitest"
    if not vitest.exists():
        pytest.fail(
            f"web/ has no vitest binary, so {PANEL_SPEC.name} never executed. "
            "vitest is ADVISORY at the gate (gate.py reports it, does not fail on "
            "it), and this spec is the only thing that renders NetworkPanel and "
            "asserts on its text — a machine without web/node_modules would pass "
            "the whole #4292 clause set on greps alone. Run `npm install` in the "
            "repo's web/ directory."
        )
    proc = subprocess.run(
        [str(vitest), "run", "--reporter=json", str(PANEL_SPEC.relative_to(WEB))],
        cwd=str(WEB), capture_output=True, text=True, env=node_env(), timeout=600)
    data = None
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            data = json.loads(line)
            break
    results: dict[str, str] = {}
    for suite in (data or {}).get("testResults", []):
        for assertion in suite.get("assertionResults", []):
            title = re.sub(r"^#\d+\s+", "", assertion.get("title") or "")
            results[title] = str(assertion.get("status"))
    assert results, (
        f"{PANEL_SPEC.name} produced no test results — the spec was not executed "
        f"(exit {proc.returncode}). An empty run is not a green one.")
    _PANEL_SPEC_RESULTS = results
    return results


def test_panel_renders_a_seeded_bash_proxy_arm():
    """Clause 3: with a `bash_proxy` row on screen, its count and a recording-on word
    sit beside the existing policy word.

    The pytest node is the citable one because the Python suite is what the gate's
    `tests` rung runs and reports nodes from; the vitest node it drives is the render
    itself, and `NetworkPanel` is exported from DashboardPage.tsx for exactly that.
    """
    titles = _panel_spec_results()
    assert titles.get("renders the bash_proxy count and a recording-on word beside "
                      "the policy word") == "passed", (
        f"panel spec node did not pass: {titles}")
    spec = PANEL_SPEC.read_text(encoding="utf-8")
    # `enforcing` is the pre-existing policy word in this fixture (`PANEL_BASE` sets
    # `enforce: true`), so its presence in the spec is what pins the clause's "beside":
    # the spec's own element search requires the policy word, the recording word and
    # the arm's count to sit in ONE element.
    for literal in ("31 bash calls", "bash recording", "enforcing"):
        assert literal in spec, (
            f"the spec no longer pins the on-screen literal {literal!r}; the clause-3 "
            "assertion has been weakened")
    assert "export function NetworkPanel" in PANEL_TSX.read_text(encoding="utf-8"), (
        "NetworkPanel is not exported, so no render test can reach it")


def test_panel_reads_the_bash_arm_as_off_and_zero_without_rows():
    """Clause 4: today's live shape (tool row only), and no `per_via` at all, both
    render — with the arm read as off and zero rather than a crashed or missing panel.

    `bash_record` is false on this box (`config.yaml:838`), so this is not a hypothetical
    degraded path: it is the only path the live panel takes, which is why the
    zero-and-word reading is pinned as hard as the populated one.
    """
    titles = _panel_spec_results()
    for node in ("reads the Bash arm as off and zero when per_via holds only a tool row",
                 "reads the Bash arm as off and zero when the backend sends no per_via"):
        assert titles.get(node) == "passed", f"panel spec node did not pass: {titles}"
    spec = PANEL_SPEC.read_text(encoding="utf-8")
    assert "bash off · 0 bash calls" in spec, (
        "the degraded headline is no longer pinned literal-for-literal")
    assert "494 tool calls" in spec, (
        "the figures line no longer names the arm it counts, which is the premise "
        "this item exists to fix")
