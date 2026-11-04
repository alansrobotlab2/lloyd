"""The behaviour-spec registry: six fields, and pointers that resolve (#3749 clauses 1-2).

Two things are pinned here, and the second is the one that keeps the instrument honest over
time rather than only on the day it was written:

1. Every declared behaviour carries all six fields and names a known detector and
   population, or the loader raises. A spec missing `trigger` is a question nobody will ask,
   and one missing `observable` is an assertion nobody can falsify.
2. Every `source` pointer resolves to a real line in the live tree — a repo file the gate
   runs in, or a vault prompt-surface file the prompt builder loads. Nightly jobs rewrite
   SOUL.md and automod rounds move harness code, and a registry whose pointer rotted would
   keep reporting an obeyance rate for a rule that is no longer anywhere. That is the drift
   the method's own author warns about ("if you put everything, you'll be duplicating a lot
   of state"), and a pointer-not-copy registry is worth nothing unless a test checks the
   pointer.

These nodes read the registry THAT SHIPPED (`eval/behaviour_specs.yaml`), not a copy, so
editing the shipped file to satisfy a docstring is not available as a shortcut.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.behaviour_exhibit as BE  # noqa: E402

REGISTRY = ROOT / "eval" / "behaviour_specs.yaml"
SPECS = BE.load_registry(REGISTRY)


def write_registry(tmp_path, behaviours, schema=1):
    """A registry file with exactly these behaviours, for the negative cases."""
    p = tmp_path / "behaviour_specs.yaml"
    p.write_text(yaml.safe_dump({"schema": schema, "behaviours": behaviours},
                                sort_keys=False), encoding="utf-8")
    return p


def spec_dict(**over):
    """One complete spec with its pointers aimed at files that exist, then overridden."""
    base = {"id": "probe-behaviour",
            "source": "app/harness/loop.py:1",
            "trigger": "one situation per turn",
            "observable": "the turn did the thing",
            "detector": "deterministic",
            "population": "both"}
    base.update(over)
    return base


# ── clause 1: the six-field contract ──────────────────────────────────────


def test_the_shipped_registry_declares_at_least_three_behaviours():
    assert len(SPECS) >= 3, (
        f"{REGISTRY.relative_to(ROOT)} carries {len(SPECS)} behaviours; the item asks for "
        "at least three real standing instruction rules")


def test_every_shipped_behaviour_carries_all_six_fields():
    for spec in SPECS:
        missing = [f for f in BE.REQUIRED_FIELDS
                   if f not in spec or not str(spec[f] or "").strip()]
        assert not missing, f"{spec.get('id')} is missing {missing}"
        assert spec["detector"] in BE.DETECTORS, spec["id"]
        assert spec["population"] in BE.POPULATIONS, spec["id"]


def test_the_loader_raises_on_a_missing_field_naming_the_behaviour_and_the_field():
    """The clause's own wording: 'the loader raises on any missing field'.

    Seeded one field at a time, so the guard cannot be a check that only the first field
    happens to satisfy, and asserted the message names BOTH the behaviour and the field — a
    failure that says `invalid registry` sends a reader to a bisect instead of to a fix.
    """
    for field in BE.REQUIRED_FIELDS:
        broken = spec_dict()
        broken.pop(field)
        with pytest.raises(ValueError) as ei:
            BE.load_registry(write_registry(_tmp(field), [broken]))
        msg = str(ei.value)
        assert (broken.get("id") or broken["source"]) in str(ei.value), (
            f"{field}: message named neither the behaviour nor its source: {msg}")
        assert field in msg, f"message did not name the missing field {field}: {msg}"


def _tmp(tag: str):
    """A scratch dir for one seeded registry (per-tag so parallel nodes cannot collide)."""
    import tempfile
    d = Path(tempfile.mkdtemp(prefix=f"beh-spec-{tag}-"))
    return d


def test_the_loader_raises_on_an_empty_value_for_a_required_field():
    """`trigger: ""` is a missing trigger wearing a key."""
    with pytest.raises(ValueError) as ei:
        BE.load_registry(write_registry(_tmp("blank"), [spec_dict(trigger="   ")]))
    assert "trigger" in str(ei.value)


def test_the_loader_raises_on_a_detector_outside_the_two_allowed():
    """`deterministic` | `djev`, and nothing else, however it is spelled."""
    for bad in ("llm", "regex", "Deterministic", "none", ""):
        with pytest.raises(ValueError) as ei:
            BE.load_registry(write_registry(_tmp(f"det-{bad or 'empty'}"),
                                            [spec_dict(detector=bad)]))
        assert "detector" in str(ei.value)


def test_the_loader_raises_on_an_unknown_population():
    for bad in ("users", "all", "sessions"):
        with pytest.raises(ValueError) as ei:
            BE.load_registry(write_registry(_tmp(f"pop-{bad}"),
                                            [spec_dict(population=bad)]))
        assert "population" in str(ei.value)


def test_the_loader_raises_on_a_duplicate_id():
    """Two specs sharing an id would have their tallies merged into one row."""
    with pytest.raises(ValueError) as ei:
        BE.load_registry(write_registry(_tmp("dup"),
                                        [spec_dict(), spec_dict(trigger="other")]))
    assert "duplicate" in str(ei.value)


def test_the_loader_raises_on_an_empty_registry():
    """An empty registry would sweep nothing and print a clean zero-behaviour artifact."""
    with pytest.raises(ValueError):
        BE.load_registry(write_registry(_tmp("empty"), []))


def test_the_loader_refuses_to_run_a_djev_detector_it_does_not_implement():
    """`djev` is a legal token in the registry and NOT a runnable detector in v1.

    Accepted by `load_registry` so the contract is expressive, refused by `sweep` so nobody
    can obtain a rate from an unimplemented judge by writing the word. Without this, a
    djev row would either crash deep in a loop or, worse, silently borrow certainty from a
    model with no agreement ceiling (#654's precedent).
    """
    reg = write_registry(_tmp("djev-run"), [spec_dict(detector="djev")])
    BE.load_registry(reg)                                   # the token is legal…
    with pytest.raises(ValueError) as ei:                    # …but not runnable
        BE.sweep(sessions_dir=_tmp("nosessions"), registry=reg, write=False, days=7)
    assert "djev" in str(ei.value) and "agreement" in str(ei.value)


# ── clause 2: source pointers resolve, in both trees ──────────────────────


def test_every_shipped_source_pointer_resolves_to_a_real_line():
    """The rot guard, run over what ships.

    Both surfaces are live here: `app/harness/loop.py:2341` is repo code the gate checks
    out, and `lloyd/SOUL.md:71` is the prompt surface a nightly job rewrites. A pointer into
    either that stops resolving is this node failing, on the day the line moves, rather than
    a rate quietly measured against a rule that no longer exists.
    """
    for spec in SPECS:
        resolved = BE.resolve_source(spec["source"])
        assert resolved.is_file(), spec["source"]


def test_a_repo_file_line_pointer_resolves_inside_the_checkout():
    p = BE.resolve_source("app/harness/loop.py:2341")
    assert p == ROOT / "app" / "harness" / "loop.py"
    assert "_CAPTION_NUDGE" in p.read_text(encoding="utf-8").splitlines()[2340], (
        "the pointer resolved but not to the anchor the spec is about: the code moved and "
        "the line number came with it")


def test_a_vault_prompt_surface_pointer_resolves_against_the_vault_root():
    p = BE.resolve_source("lloyd/SOUL.md:71")
    assert p.name == "SOUL.md" and p.is_file(), p
    line = p.read_text(encoding="utf-8").splitlines()[70]
    assert "reading it back" in line, (
        f"SOUL.md:71 no longer carries the verify-before-claiming rule: {line!r}")


def test_a_seeded_unresolvable_pointer_raises_naming_what_it_could_not_find():
    """The clause's second half, asserted directly rather than by absence."""
    with pytest.raises(ValueError) as ei:
        BE.resolve_source("app/harness/no_such_module.py:12")
    assert "no_such_module" in str(ei.value)
    with pytest.raises(ValueError) as ei:
        BE.resolve_source("lloyd/NoSoulHere.md:3")
    assert "NoSoulHere" in str(ei.value)


def test_a_pointer_past_the_end_of_a_real_file_raises_as_drift():
    """The file exists, the line does not — the shape a rewrite that shortened a file makes.

    The two-line guard would pass this, and it is the commonest way a pointer rots in
    practice: the rule is deleted, the file survives, and the number now points at nothing.
    """
    d = _tmp("short")
    (d / "app").mkdir()
    (d / "app" / "tiny.py").write_text("one\n", encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        BE.resolve_source("app/tiny.py:400", repo_root=d, vault_root=d)
    assert "past the end" in str(ei.value)


@pytest.mark.parametrize("bad", ["app/harness/loop.py", "loop.py", ":12", "app/harness/loop.py:x"])
def test_a_pointer_that_is_not_a_line_pointer_raises(bad):
    """A quotation or a bare filename is the copy-this-file-must-not-become, so it is refused.

    `source: SOUL.md` would resolve to a 71-line file and prove nothing about where the rule
    is; the requirement is a location, and the error says why a location is required.
    """
    with pytest.raises(ValueError) as ei:
        BE.resolve_source(bad)
    assert "path:line" in str(ei.value)


def test_the_registry_is_yaml_and_tracked_by_git():
    """The load-bearing trackedness assertion — for the file the instrument reads.

    `.gitignore:42` is a bare `*.json`, which is why the registry is YAML at all, and #1549
    recorded an eval artifact that `git add -A` silently skipped and a later run read as an
    unexplained regression. Checked through git rather than by reading `.gitignore`, so this
    is about trackedness and not about the ignore rules that happen to produce it.

    The version this node replaces asserted `out.stdout.strip() or REGISTRY.exists()`, and the
    reviewer caught the second half: `git ls-files` prints nothing for an untracked file, so
    the `or` fired on a plain on-disk check and the arm could not fail. `require_in_git_index`
    is now the thing under test, and its failure branch has its own node below.
    """
    assert REGISTRY.suffix == ".yaml"
    BE.require_in_git_index(REGISTRY)


def test_a_registry_the_index_does_not_know_fails_the_check(tmp_path):
    """The arm has to be capable of failing: same bytes, same checkout, untracked.

    Copied to a NEW name inside the checkout and deleted again, so the only difference from
    the node above is whether git's index knows the path — which is precisely the condition
    `require_in_git_index` claims to detect. The message must name the remedy: an untracked
    registry and a malformed one look identical from a traceback, and only one of them is
    fixed with `git add`.
    """
    import shutil
    probe = REGISTRY.with_name("behaviour_specs_trackedness_probe.yaml")
    shutil.copyfile(REGISTRY, probe)
    try:
        with pytest.raises(ValueError) as ei:
            BE.require_in_git_index(probe)
        msg = str(ei.value)
        assert probe.name in msg and "index" in msg.lower(), msg
        assert "git add" in msg, f"the failure does not name its own remedy: {msg}"
    finally:
        probe.unlink(missing_ok=True)
    assert not probe.exists(), "the probe outlived the node and would be the next untracked file"


def test_a_copy_outside_the_checkout_is_not_scored_by_the_trackedness_check(tmp_path):
    """A fixture copy in `/tmp` says nothing about git, so it must assert nothing.

    Without that boundary every fixture-driven node in the sweep test file would fail for the
    wrong reason, and the honest reading of a path outside the repository is unknown, not
    untracked.
    """
    import shutil
    probe = tmp_path / "behaviour_specs.yaml"
    shutil.copyfile(REGISTRY, probe)
    BE.require_in_git_index(probe)  # returns quietly: no claim made either way
