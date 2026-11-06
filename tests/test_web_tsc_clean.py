"""`web/` must type-check with ZERO diagnostics, and this node is what keeps it there.

Why a suite node when the gate already runs tsc. `rung_frontend`
(`scripts/automod/gate.py`) invokes `tsc --noEmit` twice and subtracts — `head - base` —
so a diagnostic that is already in the tree prints as tolerated "pre-existing" and no
round is ever refused for it. That is how two type errors survived four items that each
disclaimed them: `npm run build` is `tsc && vite build`, so the human-facing build died
at the `&&` while every gate stayed green, and the only runner of the project
type-check was the one check that cannot see them (`#3816`). The gate's delta rule stays
as written — an absolute bar there would have been switched off within the hour, which
is what its own docstring says — so the absolute bar belongs here, where every suite run
pays it, on every tree, whether or not the diff touched `web/`.

Both fixes are in Lloyd's own source and neither hides anything: `particleArrowFn` now
returns an empty `THREE.Object3D()` for a link outside `highlightLinks` (which
`linkParticlesFn` already gives 0 particles, so nothing is drawn), `handleDrop`'s third
parameter is `_insertIndex`, and `lights` — a real prop the package's `.d.ts` omits —
is declared in `web/src/three-force-graph-lights.d.ts`. The third was masked until the
first was fixed: TypeScript reports one failed JSX attributes check per element, so
`EntityGraph.tsx` carried two errors and only ever printed one.

Cost, from the runs taken while writing this: the triage reading of the same command on
an idle box was 6.8 s (`28a31b1f`, 2026-11-05), and 11.8 s twice in a row on a box with a
gate running at `38eec6e5` the next day — so quote it as "6-12 s", not one number, and
the budget below is 25-50x that. The negative control costs one more compile of a
one-file project. `web/dist` is gitignored and holds no tracked file, which
`test_the_build_output_is_not_tracked` re-derives; this node never emits, so nothing here
dirties the tree it checks.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
TSC = WEB / "node_modules" / ".bin" / "tsc"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontend_deps  # noqa: E402  (sibling module, same import the conftest makes)

from app.lint_findings import node_env  # noqa: E402  (the gate's own node environment)

#: The gate's own budget for this command. The slowest measurement in the paragraph
#: above (11.8 s) is 25x inside it, so the timeout only ever fires on a hang, which is
#: what a fixed bound is for — never on a slow-but-working compile.
TSC_TIMEOUT_SECONDS = 300

# `_provision_frontend_deps` in tests/conftest.py already tried this at import, and
# `rung_frontend` does it for a round's worktree. Asking again is idempotent
# (`PRESENT` when a `web/node_modules` is already there) and makes this file answerable
# on its own, which is how it gets run when someone edits `web/src` and wants to know
# whether the tree is clean.
_LINK_OUTCOME = frontend_deps.ensure_node_modules_link(ROOT)
_TSC_ABSENT = not TSC.exists()

#: The two regions this item edited, as a start marker and the next declaration, so
#: clause 5's "no NEW `as any`" is checkable without knowing which commit is the base.
#: A whole-file count cannot express that: the number it wants is a measurement of one
#: head, and `preflight` rebases a round onto whatever landed since. That is how this node
#: first went red on its own diff — 21 `as any` lines in `EntityGraph.tsx` against the 17
#: counted at `4852af78`, the extra four shipped by another round in between. A stale
#: expected value reads a landed change as a regression: the instrument was red, the tree
#: was not.
_EDITED_SITES = (
    ("src/components/EntityGraph.tsx",
     "const particleArrowFn = useCallback(", "const nodeTooltip = useCallback"),
    ("src/components/pages/AutonomyPage.tsx",
     "  const handleDrop = async (", "  const handleSave = async (id: number"),
)

_SUPPRESSIONS = ("@ts-ignore", "@ts-expect-error")


def _skip_if_no_tsc() -> None:
    if _TSC_ABSENT:
        pytest.skip(
            f"no {TSC} in this tree: `npm install` has never been run here and "
            f"frontend_deps.ensure_node_modules_link reported `{_LINK_OUTCOME}` "
            "(a tree with no install to borrow keeps the skip it always had — #2233)"
        )


def _run_tsc(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(TSC), *args],
        cwd=cwd,
        env=node_env(),
        capture_output=True,
        text=True,
        timeout=TSC_TIMEOUT_SECONDS,
    )


def test_web_project_type_checks_with_zero_diagnostics() -> None:
    """The acceptance check, made permanent: `tsc --noEmit -p tsconfig.json`, exit 0,
    and not one byte of output.

    This is `web/package.json`'s `build` script's first half — the half whose failure
    `&&` hides from anyone who only ever runs the gate. Neither of #3816's two errors
    may come back (`TS2322` at `EntityGraph.tsx`'s particle-object prop, `TS6133` for
    `handleDrop`'s third parameter), and no file under `web/src` may gain one.
    """
    _skip_if_no_tsc()
    r = _run_tsc(WEB, "--noEmit", "-p", "tsconfig.json")
    output = (r.stdout + r.stderr).strip()
    assert r.returncode == 0, f"tsc exited {r.returncode}:\n{output}"
    assert output == "", f"tsc exited 0 but reported diagnostics:\n{output}"


def test_the_type_check_is_given_the_whole_web_tree() -> None:
    """A green run over zero files is not a check, so pin what the run covers.

    `web/tsconfig.json` is a denied build input, which makes its `include` the stable
    definition of "the web tree" — but a denied file is still a file someone can edit
    by hand, so the covered set is asserted from disk rather than trusted from config.
    """
    cfg = json.loads((WEB / "tsconfig.json").read_text())
    assert cfg["include"] == ["src"], (
        f"web/tsconfig.json no longer includes exactly `src` ({cfg['include']}), so"
        " this node's green run covers a different set than the build's does — update"
        " both together"
    )
    sources = sorted(
        p for pattern in ("**/*.ts", "**/*.tsx")
        for p in (WEB / "src").glob(pattern)
        if p.is_file()
    )
    # 120 `.ts`/`.tsx` under `web/src` at `38eec6e5` counting this round's two new files,
    # 16 of them vitest files. Well above 100 and far above what a gutted `src` would
    # leave, so "0 diagnostics" cannot mean "0 files".
    assert len(sources) >= 100, f"only {len(sources)} files under web/src"
    for required in (
        WEB / "src" / "components" / "EntityGraph.tsx",
        WEB / "src" / "components" / "pages" / "AutonomyPage.tsx",
    ):
        assert required.exists(), f"{required} is gone from the tree it is checked in"


def test_the_runner_reports_a_type_error_it_is_shown(tmp_path: Path) -> None:
    """The negative control, because the check above passes on silence.

    An empty output and a swallowed output are the same bytes to `assert output == ""`:
    a wrong `cwd`, a mis-parsed flag, a `tsc` that cannot find its project — any of them
    turns clause 1 into a node that can never fail. So the same binary, the same
    environment and the SAME compilerOptions as the project are handed two deliberate
    errors, and both diagnostics must come back. They are the two this item is about:
    an unread parameter under `noUnusedParameters`, and a bad assignment under `strict`.
    """
    _skip_if_no_tsc()
    cfg = json.loads((WEB / "tsconfig.json").read_text())
    options = {k: v for k, v in cfg["compilerOptions"].items()
               if k not in ("baseUrl", "paths", "incremental")}
    assert options["strict"] is True and options["noUnusedParameters"] is True, (
        "web/tsconfig.json lost the flags that make these two errors errors"
    )
    # The same compilerOptions, verbatim, over a one-file project — so what is being
    # proven is that THESE flags turn THESE errors into diagnostics and that this
    # invocation reports them, which is exactly the half a green main branch cannot show.
    (tmp_path / "tsconfig.json").write_text(json.dumps(
        {"compilerOptions": options, "include": ["probe.ts"]}))
    (tmp_path / "probe.ts").write_text(
        "export function probe(used: number, unusedParam: string): void {\n"
        "  const target: number = used === 1 ? 1 : \"two\";\n"
        "  void target;\n"
        "}\n"
    )
    r = _run_tsc(tmp_path, "--noEmit", "-p", "tsconfig.json")
    reported = r.stdout + r.stderr
    assert r.returncode != 0, (
        "tsc reported exit 0 over a file with an unread parameter and a bad"
        " assignment, so the green run above is measuring nothing"
    )
    assert "probe.ts" in reported, f"nothing names the probe:\n{reported}"
    for code, why in (("TS6133", "unread parameter"), ("TS2322", "bad assignment")):
        assert code in reported, f"{why} ({code}) was not reported:\n{reported}"


def test_the_two_fixed_sites_gained_no_suppression() -> None:
    """Clause 5: the build went green by fixing types, not by hiding them.

    `@ts-ignore` and `@ts-expect-error` would silence `tsc` and leave the defect
    invisible, which is the outcome #3816 exists to prevent, so neither may appear in
    either edited file at all — not merely not-newly. A cast is what a round would reach
    for to satisfy `LinkAccessor<…, Object3D>` quickly, and "no NEW cast" is judged at the
    two edited sites rather than over a whole file, because a file-wide count needs a base
    this node does not have — which is what `_EDITED_SITES` exists to avoid.
    """
    for relpath, start, end in _EDITED_SITES:
        text = (WEB / relpath).read_text()
        for needle in _SUPPRESSIONS:
            assert needle not in text, (
                f"{relpath} carries {needle} — #3816 clause 5: the build goes green by"
                " fixing types, not by hiding them"
            )
        a = text.find(start)
        assert a > -1, (
            f"{relpath} no longer contains `{start}`: the site this node pins has moved,"
            " so re-point the marker rather than deleting the check"
        )
        b = text.find(end, a + 1)
        assert b > a, f"{relpath} has no `{end}` after `{start}` to bound the site"
        site = text[a:b]
        # `not in` against an empty string is vacuously true, which is what a mis-pointed
        # marker would leave behind: so the region has to be real code first.
        assert len(site) > 200, f"{relpath}: the `{start}` region reads as gutted"
        assert "as any" not in site, (
            f"{relpath}: `{start}` carries an `as any` — a cast is how the build goes"
            " green and the defect stays invisible (#3816 clause 5)"
        )


def test_tsconfig_still_forbids_unused_things() -> None:
    """Clause 5's other half: `web/tsconfig.json` was not loosened to reach green.

    It is a denied path for the automod loop (`scripts/automod/spec.py`), so a round
    cannot edit it — but the flags are the actual precondition for `TS6133` existing at
    all, and this item's own fix (`_insertIndex`) is only meaningful while
    `noUnusedParameters` is on. Read from the file, so a loosened flag is caught even in
    a tree that edited it by hand and committed it; `test_tsconfig_json_is_byte_identical_to_what_git_holds`
    is the other arm, which asks the same question of git.
    """
    options = json.loads((WEB / "tsconfig.json").read_text())["compilerOptions"]
    for flag in ("strict", "noUnusedLocals", "noUnusedParameters"):
        assert options.get(flag) is True, f"web/tsconfig.json: {flag} is not true"


def test_tsconfig_json_is_byte_identical_to_what_git_holds() -> None:
    """The other half of clause 5: `web/tsconfig.json` was not touched to reach green.

    Flags being `true` is necessary but not sufficient — `"exclude": ["src"]` would clear
    every diagnostic in `web/src` while leaving all three flags pristine, and so would
    `"skipLibCheck"` plus a narrowed `include`. So the comparison here is byte-for-byte
    against the blob `HEAD` holds for the file, which is what "no build-input edit" means
    without needing a base commit: `preflight` rebases a round onto whatever has landed,
    and `HEAD` is the tree this round's diff sits on, so a round that edits the file fails
    and a round that merely sits behind one that did is measuring what is actually shipped.

    A `sed` of a denied path is not hypothetical either: the deny list is the gate
    refusing to STAGE the path, and a human or a mis-scoped `git add` can still land it.
    """
    r = subprocess.run(["git", "show", "HEAD:web/tsconfig.json"], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (
        f"`git show HEAD:web/tsconfig.json` exited {r.returncode}: {r.stderr.strip()[:200]}\n"
        "This node compares the build input against its committed blob, so it needs that\n"
        "blob to exist. `web/tsconfig.json` is tracked at #3816's head (`git ls-files`\n"
        f"names it); a round that un-tracks it is a deliberate change and should say so\n"
        "in this file rather than leave the node failing for the wrong reason."
    )
    on_disk = (WEB / "tsconfig.json").read_text()
    assert on_disk == r.stdout, (
        "web/tsconfig.json differs, byte for byte, from HEAD:web/tsconfig.json. It is a\n"
        "denied build input and every flag in it is a way to make a red build green by\n"
        "hiding an error rather than fixing a type — revert it (`git checkout HEAD --\n"
        "web/tsconfig.json`) and fix the source instead. #3816 clause 5."
    )


def test_the_build_output_is_not_tracked() -> None:
    """`npm run build` may be run in a worktree because `web/dist` is not in git.

    Part of #3816's acceptance is that the build now COMPLETES, so the gate and any human
    will run it — and if its output were tracked, every build would show up as a source
    change under whoever ran it. Re-derived rather than quoted from the item, and the
    derivation is fussy in a way worth writing down: the pattern is `/web/dist/` at
    `.gitignore:16`, which is a directory pattern with a trailing slash, so it matches a
    path UNDER the directory (`git check-ignore -v web/dist/index.html` prints that line)
    and NOT the bare directory itself (`git check-ignore web/dist` exits 1). Asking about
    the bare path is how this node first went red on a tree that was fine.
    """
    ignored = subprocess.run(["git", "check-ignore", "-q", "web/dist/index.html"],
                             cwd=str(ROOT), timeout=120)
    assert ignored.returncode == 0, (
        "web/dist is no longer gitignored (`.gitignore:16` is the pattern that used to\n"
        "cover it), so running the build dirties the tree — which is exactly what makes it\n"
        "safe to run here, and what #3816 relied on to say `npm run build` \"runs vite to\n"
        "completion\" without leaving a diff behind."
    )
    listed = subprocess.run(["git", "ls-files", "web/dist"], cwd=str(ROOT),
                            capture_output=True, text=True, timeout=120)
    assert listed.returncode == 0, listed.stderr.strip()[:200]
    assert not listed.stdout.strip(), (
        f"tracked files under web/dist: {listed.stdout.splitlines()[:5]} — compiled output\n"
        "in git makes every build look like a source edit by whoever ran it."
    )


def test_the_lights_prop_is_declared_in_our_own_source() -> None:
    """The third error #3816 had to clear, and the file that clears it, stay in place.

    `lights` is a real `<ForceGraph3D>` prop whose package typings omit it, and TypeScript
    reports one failed JSX attributes check per element — so while the particle accessor
    was red, this one never printed. Clearing the first error is what exposed it, which is
    why the item's two-error premise undercounts the work: three sites went green, not
    two, and the third is the only one whose fix is a NEW file.

    That makes this node the only thing standing between a green `tsc` and the file being
    deleted as "unused plumbing" by a later tidy: deleting `web/src/three-force-graph-lights.d.ts`
    puts `npm run build` back to red at `EntityGraph.tsx`, and this node goes red first and
    by name instead of the build dying on a human. It is read as text, not compiled, for
    the reason the runner has no DOM: what is pinned is that the declaration exists, that
    it types the prop as `Light[]` rather than `any`, and that it hides nothing.
    """
    decl = WEB / "src" / "three-force-graph-lights.d.ts"
    assert decl.exists(), (
        f"{decl.relative_to(ROOT)} is gone. It is what makes `<ForceGraph3D lights={{…}}>`\n"
        f"type-check: the package declares `lights` on the instance only, and deleting this\n"
        f"file returns `npm run build` to red — not to two errors, to three."
    )
    text = decl.read_text()
    assert 'declare module "react-force-graph-3d"' in text, (
        "the file no longer augments the package it is augmenting, so it declares nothing")
    assert "lights?: Light[];" in text, (
        "`lights` is no longer declared as Light[] — the prop's value has to stay checked,\n"
        "which is the difference between declaring a missing prop and casting one away.")
    for needle in _SUPPRESSIONS + ("as any",):
        assert needle not in text, (
            f"the augmentation carries {needle}: clause 5 — the build goes green by fixing\n"
            "types, not by hiding them."
        )


# The two clause pins below restate, in the suite the gate runs on every round, what
# `web/src/components/EntityGraph.test.ts` and `.../pages/AutonomyPage.test.ts` assert
# under vitest. Not duplication for its own sake: `scripts/automod/testpaths.py`
# `is_test_file` accepts only `.py`, so a `.test.ts` node is invisible to the review's
# own evidence bookkeeping — which is the one thing that made #3816's first round fail
# review while its own reasoning called both pins two-directional. The clause needs a
# node in a file the review can classify as a test file, and vitest cannot be that file
# without editing the helper, which is not this item.


def test_the_arrow_cone_stays_confined_to_highlighted_links() -> None:
    """Clause 3 from the Python side: the cone is confined to the highlighted set.

    `particleArrowFn` has to keep three properties and change one: the `highlightLinks`
    membership test, the `THREE.Mesh` cone for a link inside that set, and the same key
    expression its sibling `linkParticlesFn` uses — and the value it returns for a link
    outside the set becomes an empty `Object3D` instead of `undefined`.

    Why not simply return `undefined` and cast: the prop's type admits no `undefined`
    (`LinkAccessor<…, Object3D>`), which is the error this item exists to clear, and the
    object is never drawn either way because the library digests
    `visibleLinks.filter(particlesAccessor)` before asking for a particle object
    (`three-forcegraph.mjs:1394`) while `linkParticlesFn` gives the excluded links 0
    particles. But an `Object3D` is truthy, so the membership test is what keeps a
    photonless `Mesh(undefined, undefined)` off the screen: that line is asserted below in
    both directions, and it is the line a future "simplify this" commit would drop.
    """
    text = (WEB / "src" / "components" / "EntityGraph.tsx").read_text()
    start = text.index("const particleArrowFn = useCallback(")
    end = text.index("const nodeTooltip = useCallback", start)
    block = text[start:end]
    assert len(block) > 200, "particleArrowFn reads as gutted"
    assert "highlightLinks.has(" in block, (
        "particleArrowFn lost its membership test. With an Object3D on the other branch\n"
        "there is no longer an `undefined` to stop a cone being built for a dimmed link,\n"
        "so this test is the only thing confining arrow cones to highlighted links."
    )
    assert "return new THREE.Object3D();" in block, (
        "the excluded branch is back to returning undefined, which is the TS2322 #3816\n"
        "cleared: `LinkAccessor<…, Object3D>` admits no undefined."
    )
    assert "return undefined" not in block, (
        "an explicit `return undefined` reappeared in particleArrowFn — same error, one\n"
        "token away from the fix."
    )
    assert "new THREE.Mesh(" in block and "new THREE.ConeGeometry(" in block, (
        "the cone for a highlighted link is gone, which would 'fix' the type error by\n"
        "deleting the feature clause 3 asks to keep."
    )
    # The two accessors must agree on which link is which, or the 0-particle guarantee
    # lands on different links than the cone does.
    assert block.count("${src}\\x00${tgt}") == 1, (
        "particleArrowFn's key expression no longer matches linkParticlesFn's; the\n"
        "exclusion sets have to be the same set or a dimmed link gets a bare Mesh.")
    particles = text[text.index("const linkParticlesFn = useCallback("):start]
    assert "${src}\\x00${tgt}" in particles and "? 5 : 0" in particles, (
        "linkParticlesFn no longer hands 0 particles to the complement of the same key —\n"
        "the guarantee the empty object depends on.")


def test_the_drop_handler_still_writes_only_status() -> None:
    """Clause 4 from the Python side: one write, same arguments, no unread parameter.

    `handleDrop`'s third parameter is what made `tsc` exit 2 (`TS6133` under
    `noUnusedParameters`). The fix is the underscore: the caller still passes an index and
    this handler still does not read it, because the one write it makes changes a task's
    status and nothing else. Deleting the parameter instead would have meant editing
    `KanbanColumn`'s computation and the prop type, and the drag-to-reorder decision
    #3816 deliberately leaves to a human would have been made by removing its UI.
    """
    text = (WEB / "src" / "components" / "pages" / "AutonomyPage.tsx").read_text()
    head = text[text.index("  const handleDrop = async ("):]
    signature = head.split("\n", 1)[0]
    assert "_insertIndex: number" in signature, (
        "the third parameter is neither `_insertIndex` nor gone, so `noUnusedParameters`\n"
        "is back to reporting TS6133 and `npm run build` dies before vite."
    )
    assert "(taskId: number, newStatus: string," in signature, (
        f"handleDrop's first two parameters changed: {signature!r} — this item clears a\n"
        "type error and is not allowed to move the drag-to-change-status contract.")
    body = head[:head.index("  const handleSave = async (id: number")]
    assert "api.autonomyWriteTask({ id: taskId, status: newStatus })" in body, (
        "the drop no longer sends exactly that call: clause 4 requires the same write, on\n"
        "the same arguments, and nothing else added to the drag path."
    )
    assert body.count("await api.autonomyWriteTask(") == 1, (
        "handleDrop now issues more than one write per drop — a status change plus\n"
        "whatever the index became is a behaviour change #3816 does not authorise.")
    # The caller side is untouched, which is what makes the underscore honest rather than
    # a rename that hides a broken seam: the index is still computed and still delivered.
    assert "onDrop(draggedId, status, insertIndex);" in text, (
        "KanbanColumn stopped passing the computed index, so `_insertIndex` is a lie\n"
        "about the seam — delete the computation and the prop too, or leave both."
    )
    assert "insertIndex: number) => void;" in text, (
        "the prop type's third parameter is gone while the call site still passes three\n"
        "arguments, which is a new type error rather than a fixed one.")
