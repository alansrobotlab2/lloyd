"""The guarded vault mirror for the Inner Voice metrics series (backlog #2121).

The nightly job (#86) has always mirrored `iv-metrics.jsonl` into the vault with a bare
`cp`. On 2026-09-23 the source path moved under the data-root cutover, so the `cp` read a
1-row file at the new location and wrote it over the 11-row mirror: 10 rows of the
observer's measurement history vanished from the only committed copy of that history, and
were restored by hand as vault commit `e508d468`. Nothing refused, and nothing in this
repository could have refused — `git grep -n "mirror-shrunk"` returned 0 hits before this
file existed, and `grep -c -i mirror tests/test_iv_metrics_series.py` returned 0.

The 09-23 shrink is still invisible in `git log --numstat`: every commit touching the
mirror since it landed (`f704f76e`, +11/-0) is `+N/-0`, *including* the recovery commit,
because the rows were deleted from the working file and restored before anything was
committed. A future -10 has no reason to be caught by anything but this script.

So every runtime node here drives the real `scripts/iv_metrics_mirror.py` across a real
process boundary — the same `python3 scripts/iv_metrics_mirror.py` the skill and the task
front-matter tell the nightly run to type, with `--source`/`--mirror` aimed at a fixture —
because the thing under test is a copy step an agent shells out to, not a function. The two
vault-reading nodes at the end pin the two prompt-facing surfaces, which are the only thing
the runner actually receives: `_build_task_prompt` (`app/autonomy.py`) renders the skill
body and the front-matter `description` and nothing else, so a guard that exists but is not
named by those two strings is a guard no night ever runs. That is the same live-vault
coupling `tests/test_iv_metrics_series.py::_task_file` already uses.

Why the vault-reading nodes are NOT marked `live_vault`
-------------------------------------------------------
`pytest.ini:14-19` defines that marker for exactly these reads and the automod gate runs
`-m "not live_vault"` — so a marked node is deselected on the one rung that grades clauses
4, 5 and 6, and certifies nothing there. That trade is struck deliberately, for the reason
`tests/test_archived_skill_artifacts.py:30-40` records for the same choice: a marked
assertion is only ever reported, never enforced, and these three have to be enforced. The
price is the coupling — a nightly rewrite of the skill or the task file reddens the next
round's `tests` rung — so each vault assertion below is structural (does this surface name
the copy step, does it carry the token requirement) rather than a quote of prose a person
may legitimately re-wrap.

Run:
  .venvs/lloyd/bin/python -m pytest tests/test_iv_metrics_mirror.py -q
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MIRROR_SCRIPT = ROOT / "scripts" / "iv_metrics_mirror.py"

#: The token acceptance clauses 1 and 5 name. Clause 5 requires it VERBATIM in the run's
#: final report, so it is a literal here rather than a fragment a later edit can reword:
#: a token matched only by a pattern the test itself invented proves nothing.
REFUSAL_TOKEN = "mirror-shrunk"

#: The copy step every surface must agree on — the script, the skill's step 3, the task's
#: `description:`, and the string these tests look for.
SCRIPT_CWD_REL = "scripts/iv_metrics_mirror.py"

VAULT = Path.home() / "obsidian"
SKILL = VAULT / "skills" / "iv-metrics-series" / "SKILL.md"
AUTONOMY_DIR = VAULT / "autonomy"
#: Acceptance clause 6's witness: the committed extract the nightly report quotes its row
#: count from, so the figure survives whatever happens to the gitignored data root.
WITNESS = VAULT / "backlog" / "data" / "iv-metrics.jsonl"
MIRROR = VAULT / "knowledge" / "inner-voice" / "iv-metrics.jsonl"
#: Globbed, not hardcoded, so renaming the task file turns these red instead of leaving
#: them green against nothing (the convention `tests/test_iv_metrics_series.py:58` sets).
TASK_GLOB = "86-*iv*metrics*.md"


def _rows(n: int) -> str:
    """`n` JSONL rows in the shape the real series uses: one JSON object per line.

    `since`/`llm_calls`/`dropped_rate` are the three keys a nightly report quotes, so a
    fixture row stays readable by the `tail -1 | python3 -c` the task body documents.
    """
    return "".join(
        json.dumps({"since": f"2026-10-{i + 1:02d}T08:00:00",
                    "llm_calls": 100 + i, "dropped_rate": round(0.01 + i / 1000, 4)})
        + "\n"
        for i in range(n))


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    """A source and a mirror, under one fixture root, at their production-relative names.

    TWO paths, always, and never the same one: `--source` and `--mirror` resolving to the
    same file makes the script copy a file onto itself, which exits 0 and leaves
    byte-identical bytes by construction — every refusal and every byte-identity assertion
    here would then pass while comparing nothing.
    """
    source = tmp_path / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    mirror = tmp_path / "knowledge" / "inner-voice" / "iv-metrics.jsonl"
    source.parent.mkdir(parents=True)
    mirror.parent.mkdir(parents=True)
    return source, mirror


def _run(*args: str) -> subprocess.CompletedProcess:
    """One real invocation of the mirror step, from the repo root, as the nightly runs it.

    `--source`/`--mirror` are how a test reaches a fixture without touching the live pair;
    with neither flag the script uses the production paths, which is what step 3 of the
    skill documents.
    """
    assert MIRROR_SCRIPT.exists(), f"{MIRROR_SCRIPT} is the guarded copy step; it is missing"
    return subprocess.run(
        [sys.executable, str(MIRROR_SCRIPT), *args],
        capture_output=True, text=True, cwd=str(ROOT))


def _skill_step_3() -> str:
    """Step 3 of the skill a dispatched #86 run is handed, whitespace collapsed.

    The file is wrapped at ~80 columns, so a matched phrase can straddle a line break; one
    markdown line break is not a different instruction, and a test that depended on one
    would fail on a re-wrap rather than on a changed step.
    """
    assert SKILL.exists(), f"{SKILL} is the skill #86 dispatches; it is missing"
    text = SKILL.read_text(encoding="utf-8")
    step = re.search(r"^3\.\s.*?(?=^4\.\s)", text, re.S | re.M)
    assert step, "SKILL.md has no numbered step 3 ending at step 4 to carry the copy step"
    return re.sub(r"\s+", " ", step.group(0))


def _task_file_text() -> str:
    """Whole text of the #86 task file, body included — clause 4 says "anywhere in".

    The body is documentation, not an instruction channel, and it still carried the `cp` at
    lines 141 and 162: the next reader treats it as the procedure, so it moves too.
    """
    matches = sorted(AUTONOMY_DIR.glob(TASK_GLOB))
    assert matches, f"no IV-metrics task file matching {TASK_GLOB} under ~/obsidian/autonomy/"
    return matches[0].read_text(encoding="utf-8")


def _task_description() -> str:
    """The front-matter `description:` — the half of #86 the runner actually receives."""
    parts = _task_file_text().split("\n---\n", 1)
    assert len(parts) == 2, "the task file has no front matter to read a description from"
    front = yaml.safe_load(parts[0])
    assert isinstance(front, dict) and front.get("description"), "no `description:` to pin"
    return str(front["description"])


# ── clause 1: a shrinking source is refused, and the mirror bytes do not move ──


def test_a_source_shorter_than_the_mirror_is_refused_and_the_mirror_is_untouched(tmp_path):
    """Clause 1: the 2026-09-23 incident, replayed — 1 row in, 11 rows already mirrored.

    Non-zero exit (a refusal an autonomy run can read: `scripts/backup/backup-vault.sh:64`
    refuses with `exit 0`, which is precisely what is NOT copied here, because the caller of
    this script reads its own exit code), the literal token printed, and the mirror left
    byte-for-byte as it was.
    """
    source, mirror = _pair(tmp_path)
    mirror.write_text(_rows(11), encoding="utf-8")
    before = mirror.read_bytes()
    source.write_text(_rows(1), encoding="utf-8")

    result = _run("--source", str(source), "--mirror", str(mirror))

    assert result.returncode != 0, (
        f"a 1-row source over an 11-row mirror must not exit 0: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}")
    out = result.stdout + result.stderr
    assert REFUSAL_TOKEN in out, f"no {REFUSAL_TOKEN!r} token in the refusal: {out!r}"
    assert mirror.read_bytes() == before, "the mirror changed on a refused copy"
    assert list(tmp_path.rglob("*.part")) == [], "a partial write was left beside the mirror"


def test_the_shrink_refusal_names_the_two_counts_it_compared(tmp_path):
    """The refusal line has to be reportable, not just loud: 1 source row, 11 mirror rows.

    A run that must put the token in its final report can only do that usefully if the same
    line carries the two numbers the guard compared — that pair is the whole diff a person
    reads, and it is what distinguishes a moved data root from a compacted series.
    """
    source, mirror = _pair(tmp_path)
    mirror.write_text(_rows(11), encoding="utf-8")
    source.write_text(_rows(1), encoding="utf-8")

    result = _run("--source", str(source), "--mirror", str(mirror))

    out = result.stdout + result.stderr
    assert "source=1 mirror=11" in out, f"the refusal never names its own counts: {out!r}"


# ── clause 2: at-or-above copies whole, exits 0, and `cmp` is clean ────────────


def test_a_longer_source_copies_whole_exits_zero_and_is_byte_identical(tmp_path):
    """Clause 2, the healthy night: 22 rows over a 21-row mirror → copy, exit 0, `cmp` clean.

    Byte-identical is what `cmp` tests, so the fixture compares bytes rather than counting
    lines — a mirror that had reached 22 rows by rewriting a row would still count right.
    """
    source, mirror = _pair(tmp_path)
    mirror.write_text(_rows(21), encoding="utf-8")
    source.write_text(_rows(22), encoding="utf-8")

    result = _run("--source", str(source), "--mirror", str(mirror))

    assert result.returncode == 0, result.stdout + result.stderr
    assert mirror.read_bytes() == source.read_bytes(), "the mirror is not byte-identical"
    assert REFUSAL_TOKEN not in result.stdout + result.stderr, (
        "a healthy copy printed the refusal token, so the token would mean nothing")


def test_an_equal_source_copies_whole_and_exits_zero(tmp_path):
    """Clause 2 at its boundary: equal is allowed, because "shorter" is the refusal.

    The re-run case — the same source copied twice, or a night whose copy step already ran.
    Refusing at equal would strand the mirror on every idempotent re-run.

    The newest SOURCE row here has no trailing newline, so `wc -l` reads 20 against the
    mirror's 21, and the guard must compare ROWS: on a newline count this copy looks like a
    21-to-20 shrink and is refused every single night, and a guard that refuses nightly is
    indistinguishable from a mirror that is simply current. Byte-identity rather than a line
    count is also what stops the copy from "repairing" that newline and leaving two files
    that disagree for anything that compares them with `cmp`.
    """
    source, mirror = _pair(tmp_path)
    source.write_text(_rows(21)[:-1], encoding="utf-8")
    mirror.write_text(_rows(20)
                      + '{"since": "2026-09-30T08:00:00", "llm_calls": 7, "dropped_rate": 0.5}\n',
                      encoding="utf-8")

    result = _run("--source", str(source), "--mirror", str(mirror))

    assert result.returncode == 0, result.stdout + result.stderr
    assert mirror.read_bytes() == source.read_bytes(), (
        "an equal-length copy must still land the source's own bytes, not keep the old ones")


def test_no_mirror_yet_copies_and_exits_zero(tmp_path):
    """Clause 2's other edge: no destination at all — the first night, or a wiped mirror.

    The guard compares against zero rows, so the whole file lands, parent directories
    created, exit 0.
    """
    source, mirror = _pair(tmp_path)
    source.write_text(_rows(3), encoding="utf-8")
    shutil.rmtree(mirror.parent)   # no destination directory at all, not merely no file
    assert not mirror.parent.exists()

    result = _run("--source", str(source), "--mirror", str(mirror))

    assert result.returncode == 0, result.stdout + result.stderr
    assert mirror.read_bytes() == source.read_bytes(), (
        "a missing mirror must be created and filled, not refused")


# ── clause 3: a missing source cannot overwrite the mirror ────────────────────


def test_a_missing_source_is_refused_and_the_mirror_left_byte_identical(tmp_path):
    """Clause 3: a night where the grader emitted nothing must not empty the mirror.

    The recorder exits 3 and appends nothing on exactly this night, so the run should never
    reach the copy step at all — but a guard cannot depend on that being remembered, and the
    09-23 incident was a source file that was not where the command thought it was.
    """
    source, mirror = _pair(tmp_path)
    mirror.write_text(_rows(21), encoding="utf-8")
    before = mirror.read_bytes()

    result = _run("--source", str(source), "--mirror", str(mirror))

    assert result.returncode != 0, (
        f"a source that does not exist must not exit 0: {result.stdout + result.stderr!r}")
    assert mirror.read_bytes() == before, "the mirror changed on a night with no source"
    assert "source-missing" in result.stdout + result.stderr, (
        "the refusal must name which guard fired, not only that one did")


def test_an_empty_source_is_refused_and_the_mirror_left_byte_identical(tmp_path):
    """The same clause one failure later: a source that exists but holds zero rows.

    0 rows is fewer than any mirror, so the count guard alone would catch it; this pins that
    it is refused as an empty source (`source-empty`) rather than as a shrink, because those
    two mean different things to whoever reads the nightly report — one is a dead grader, the
    other is a damaged history.
    """
    source, mirror = _pair(tmp_path)
    source.write_text("", encoding="utf-8")
    mirror.write_text(_rows(21), encoding="utf-8")
    before = mirror.read_bytes()

    result = _run("--source", str(source), "--mirror", str(mirror))

    assert result.returncode != 0, "an empty source must not exit 0"
    assert mirror.read_bytes() == before, "an empty source overwrote the mirror"
    assert "source-empty" in result.stdout + result.stderr, (
        "an empty source must be named as such rather than as a shrink")


# ── clause 4: both prompt-facing surfaces name the script, and neither a `cp` ─


def test_step_3_and_the_task_description_name_the_script_and_no_cp_survives():
    """Clause 4: the guarded script is THE copy step in both surfaces the run receives.

    `_build_task_prompt` renders the skill body and the front-matter `description` and
    nothing else, and the dispatched prompt for run `run_86_20261003_080206` contained
    `cp ~/lloyd-data` twice for exactly that reason — once from each surface. A guard named
    by only one of them is a guard the other half still overrides.
    """
    step = _skill_step_3()
    desc = _task_description()
    assert SCRIPT_CWD_REL in step, f"step 3 does not name {SCRIPT_CWD_REL}: {step!r}"
    assert SCRIPT_CWD_REL in desc, f"the task description does not name {SCRIPT_CWD_REL}"

    for name, text in (("SKILL.md", SKILL.read_text(encoding="utf-8")),
                       ("autonomy/86", _task_file_text())):
        assert "cp ~/lloyd-data" not in text, f"{name} still carries the bare cp"
        stale = re.findall(r"\bcp\b[^\n]*iv-metrics\.jsonl", text)
        assert not stale, f"{name} still names a cp of iv-metrics.jsonl: {stale}"


# ── clause 5: a refused copy must reach a person as a token ───────────────────


def test_step_3_requires_the_refusal_token_verbatim_in_the_final_report():
    """Clause 5: a refused mirror must not be able to read as a green night.

    Step 4 already does this for `exit code 2`: `_detect_silent_failures`
    (`app/autonomy.py:39`, over `_SILENT_FAILURE_PATTERNS` at `:30-36`, whose
    `exit code` regex is `:33`) greps the run's FINAL PROSE, because a
    script's own exit code dies inside the run's Bash tool result. Step 3 has to say the
    same thing about `mirror-shrunk`, in the same words, or a 09-23-sized shrink reports as
    a normal night on which the mirror was deliberately not updated.
    """
    step = _skill_step_3().lower()
    assert REFUSAL_TOKEN in step, f"step 3 never names {REFUSAL_TOKEN!r}"
    assert "verbatim" in step, "step 3 does not require the token verbatim"
    assert "final report" in step, "step 3 does not say the token belongs in the final report"
    assert re.search(r"exit[s]? (?:with )?(?:non-zero|code 1|1\b)", step), (
        f"step 3 does not tie the token to a non-zero exit: {step!r}")


def test_the_flagless_command_the_skill_documents_resolves_both_default_paths(tmp_path):
    """The seam every node above skips: the copy step as the run is told to type it.

    Step 3 and #86's description both say `cd ~/lloyd && python3 scripts/iv_metrics_mirror.py`
    with no flags, so the paths that matter in production are the ones the script computes for
    itself — `DEFAULT_SOURCE` from `app.paths.PIPELINE_DIR` and `DEFAULT_MIRROR` from
    `app.data_root.vault_root()`. A node that only ever passes `--source`/`--mirror` would keep
    passing after a layout change moved one of those two constants, which is exactly the 2026-09-23
    root cause: the data root moved, the path the copy named did not, and the copy read a short
    file at the new location.

    Both roots are therefore overridden through the two env seams that produce them
    (`LLOYD_DATA`, `LLOYD_VAULT_ROOT`) rather than through flags, so the defaults themselves are
    under test: no source must refuse, and a longer source must land at the vault path the
    skill names.
    """
    data_root, vault = tmp_path / "lloyd-data", tmp_path / "vault"
    source = data_root / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    source.parent.mkdir(parents=True)
    mirror = vault / "knowledge" / "inner-voice" / "iv-metrics.jsonl"
    mirror.parent.mkdir(parents=True)
    mirror.write_text(_rows(21), encoding="utf-8")
    before = mirror.read_bytes()
    env = {**os.environ, "LLOYD_DATA": str(data_root), "LLOYD_VAULT_ROOT": str(vault)}

    def flagless() -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(MIRROR_SCRIPT)],
                              capture_output=True, text=True, cwd=str(ROOT), env=env)

    refused = flagless()
    assert refused.returncode != 0, (
        f"the documented command must refuse when its default source is not there: "
        f"{refused.stdout + refused.stderr!r}")
    assert "source-missing" in refused.stdout + refused.stderr
    assert mirror.read_bytes() == before, "the flagless run moved the default mirror anyway"

    source.write_text(_rows(22), encoding="utf-8")
    copied = flagless()
    out = copied.stdout + copied.stderr
    assert copied.returncode == 0, f"the documented command failed on a good night: {out!r}"
    assert mirror.read_bytes() == source.read_bytes(), (
        "the flagless copy did not land where the skill says it lands")


def test_the_witness_copy_is_a_real_prefix_of_the_mirror_it_quotes():
    """Clause 6: the row count a nightly report quotes is re-derivable from committed bytes.

    `_pipeline/` is gitignored, so a figure quoted from the live source is unfalsifiable once
    that disk is reimaged — which is what `backlog/data/iv-metrics.jsonl` exists to prevent. The
    two assertions that matter are that it is REAL data from the same history and not a stub:
    every row parses as one of the series' own objects, and its bytes are a byte-prefix of the
    vault mirror, which only holds if the witness came out of that append-only series rather than
    being typed alongside it.

    The count is a floor, not an equality: `wc -l` printed 21 on the day the witness was cut, and
    the series only grows — this round's own guard is what keeps it append-only — so pinning 21
    exactly would make the next healthy nightly row a red test.
    """
    assert WITNESS.exists(), f"{WITNESS} is the committed witness clause 6 asks for"
    witness_bytes, mirror_bytes = WITNESS.read_bytes(), MIRROR.read_bytes()
    assert mirror_bytes.startswith(witness_bytes), (
        "the witness is no longer a prefix of the vault mirror: one of the two has been "
        "rewritten, which the append-only invariant forbids")

    rows = [json.loads(line) for line in witness_bytes.decode().splitlines() if line.strip()]
    assert len(rows) >= 21, (
        f"the witness holds {len(rows)} rows but the report quotes 21, so the figure is not "
        f"the one in the committed bytes")
    assert all("since" in r and "llm_calls" in r for r in rows), (
        "the witness rows are not rows of the series")


def test_step_3_still_forbids_copying_on_the_recorder_s_exit_3():
    """The pre-existing half of step 3 survives the rewrite.

    "On exit 3 nothing was appended, so do not copy" is what stopped a grader failure from
    becoming a datapoint before this round; a rewrite that silently dropped it would trade
    one unguarded path for another.
    """
    step = _skill_step_3()
    assert re.search(r"exit 3", step), "step 3 no longer mentions the recorder's exit 3"
    assert re.search(r"(do not|don't) copy", step, re.I), "step 3 no longer forbids copying on exit 3"
