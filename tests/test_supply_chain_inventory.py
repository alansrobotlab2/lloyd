"""The observed component inventory (#2593 clauses 2, 3, 4).

`app.harness.supply_chain` answers "does the tree *declare* a vulnerable
package?". These nodes pin the other question, the one the board keeps citing
and nothing could answer: "is the thing that just got a disclosure in it on this
box, and where". Three separate claims, and each has to be shown to be able to
fail, because the failure mode of an inventory is that it reads nothing and
reports a clean set:

  * an inventory that reads `*/dist-info/METADATA` from roots the box actually
    runs, and keeps which interpreter installed what;
  * a scan target synthesised from that inventory, pushed through the same
    offline argv the declared scan uses, with the parse count pinned against the
    inventory so a target the parser choked on cannot arrive as zero;
  * the component kinds no advisory feed can see, which must be queryable and
    must be structurally unable to move a verdict.

The roots are all planted. Nothing here reads `~/.venvs` or the real vault: a
test that passes by measuring the box it runs on is a snapshot, and the machine
moves (the `~/lloyd-data` root incident is the standing reminder). The one place
a real box is consulted — `test_the_inventory_finds_the_component_declared_in_the
provenance_of_the_one_real_install_event_on_this_box` — is a fixture that is
allowed to skip, never a claim about the corpus.

The fake scanner is the seam for clause 3. It is a real subprocess that writes
real osv-scanner-shaped JSON and a real count line, because what clause 3 forbids
is a synthesised target the *parser* silently rejects — a monkeypatched
`subprocess.run` could not express that, only confirm the code path ran.
"""
import datetime as dt
import hashlib
import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.harness import supply_chain as sc          # noqa: E402
from app.harness import supply_chain_inventory as inv  # noqa: E402

NOW = dt.datetime(2026, 10, 11, 12, 0, 0, tzinfo=dt.timezone.utc)


def write_dist(root: Path, name: str, version: str) -> Path:
    """A real `.dist-info/METADATA`, as pip lays one down."""
    info = root / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="utf-8")
    return info


def fake_scanner(path: Path, *, fail_for: str = "", counts: bool = True) -> Path:
    """A scanner that reports what it is told to, in osv-scanner's own shapes.

    Advisory ids are derived from the package name so a test can assert which
    rows were counted without hard-coding digests. `fail_for` plants one package
    whose scan exits non-zero, which is how a site goes to `failed`. `counts`
    toggles the `found N packages` line: off, the scanner returns a valid report
    and declines to say how much it parsed — the exact condition clause 3 exists
    to refuse.
    """
    path.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env python3
        import json, sys
        args = sys.argv[1:]
        lockfiles = [args[i + 1] for i, a in enumerate(args) if a == "--lockfile"]
        lockfiles = [p.split(":", 1)[1] if ":" in p and not p.startswith("/") else p
                     for p in lockfiles]
        rows = []
        for lf in lockfiles:
            for line in open(lf):
                line = line.strip()
                if not line or line.startswith("#") or "==" not in line:
                    continue
                name, version = line.split("==", 1)
                rows.append((name.strip(), version.strip()))
        if any(n == "{fail_for}" for n, v in rows) and "{fail_for}":
            sys.stderr.write("the planted package broke the parser\\n")
            sys.exit(2)
        out = {{"results": [{{"packages": [{{"package": {{"name": n, "version": v}},
                 "vulnerabilities": [{{"id": "GHSA-" + n.upper()[:6],
                                      "summary": n + " planted advisory"}}]}}
                for n, v in rows]}}]}}
        print(json.dumps(out))
        if {counts!r}:
            sys.stderr.write("Scanned a file and found %d packages\\n" % len(rows))
        """), encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def box(tmp_path):
    """A machine with two venvs, one launcher that names them, and a user-site.

    `home` is the account home and `repo` the checkout, both under tmp. The
    launcher text is the only thing connecting a venv to the inventory: the scope
    is "what a service execs", so a venv that exists and is not named is not in
    scope, and a test that wanted it in scope would edit the launcher rather than
    the module.
    """
    home = tmp_path / "home"
    repo = tmp_path / "checkout"
    (repo / "agent-services" / "bin").mkdir(parents=True)
    (repo / "agent-services" / "supervisor" / "conf.d").mkdir(parents=True)
    (repo / "requirements.txt").write_text("requests==2.32.0\n", encoding="utf-8")
    # `requests` is installed at exactly this pin (the covered case) and `vllm` is
    # installed at a version this pin does NOT carry (the drift case): the delta
    # rule asks which NAMES the declared set omits, not which versions differ.
    (repo / "requirements.lock").write_text(
        "# generated\nrequests==2.32.0\nvllm==0.29.0\n", encoding="utf-8")
    (repo / "agent-services" / "bin" / "start-djev.sh").write_text(
        'export VIRTUAL_ENV="$HOME/lloyd/.venvs/vllm-djev"\n', encoding="utf-8")
    (repo / "agent-services" / "supervisor" / "conf.d"
     / "agent-llm-primary.conf").write_text(
        'command="/home/alansrobotlab/lloyd/.venvs/qwen3-tts/bin/python" serve\n',
        encoding="utf-8")
    tts = home / "lloyd/.venvs/qwen3-tts/lib/python3.12/site-packages"
    djev = home / "lloyd/.venvs/vllm-djev/lib/python3.14/site-packages"
    user = home / ".local/lib/python3.14/site-packages"
    for site in (tts, djev, user):
        site.mkdir(parents=True)
    write_dist(tts, "gradio", "6.9.0")
    write_dist(tts, "pillow", "12.1.1")
    write_dist(tts, "requests", "2.32.0")          # declared, so not "missed"
    # declared by NAME in requirements.lock, at a different version: drift, and
    # deliberately not part of the missed count
    write_dist(djev, "vllm", "0.2.1.dev19")
    write_dist(djev, "requests", "2.32.0")         # declared, advised, not missed
    write_dist(djev, "pip", "24.0")                # declared nowhere, not advised
    write_dist(djev, "gradio", "6.9.0")            # same pair as tts: one package,
    write_dist(djev, "pillow", "12.1.1")           # two installed copies
    write_dist(user, "youtube-transcript-api", "1.2.4")
    return {"home": home, "repo": repo, "tts": tts, "djev": djev, "user": user}


def roots(box):
    return inv.site_roots(box["repo"], box["home"])


def run_observed(box, tmp_path, **kw):
    """`scanner` is on PATH by `with_path`; the shipped resolver finds it there.

    No stub of `resolve_scanner`: the seam under test is "the argv the declared
    scan builds, handed to the binary the box has", and stubbing the resolver
    would test a path the scan never takes.
    """
    return inv.observed_scan(box["repo"], out_dir=tmp_path / "observed",
                             scanner_name="osv-scanner", account_home=box["home"],
                             roots=roots(box), now=NOW, **kw)


def with_path(box, tmp_path, extra_env=""):
    """A scanner on PATH, resolved by the shipped `resolve_scanner`, no stubbing."""
    os.environ["PATH"] = f"{tmp_path}:{os.environ['PATH']}" + extra_env
    os.environ.pop(sc.OSV_API_URL_ENV, None)


# ── clause 2 — the inventory reads what the box installed ───────────────────

def test_the_scope_is_the_venvs_a_launcher_names_plus_the_account_user_site(box):
    """The scope decision, written into the artifact and pinned here.

    #688's triage left "the scan target must be decided, not assumed" open. This
    is the decision: five venvs named by a launcher or a supervisor drop-in, plus
    every `~/.local/lib/python3.*/site-packages`. Two venvs are planted here and
    both appear; the third venv directory that exists but is named by nothing does
    not, which is the half that makes this a scope rather than a disk census.
    """
    stray = box["home"] / "lloyd/.venvs/leftover/lib/python3.12/site-packages"
    stray.mkdir(parents=True)
    write_dist(stray, "something-nobody-runs", "0.1")
    found = {r.label for r in roots(box)}
    assert found == {"qwen3-tts", "vllm-djev", "user-site"}, (
        "the roots of the inventory are not the venvs a launcher execs plus the "
        f"user-site: {sorted(found)}")
    assert "leftover" not in found, (
        "a venv no service names entered the inventory, so the scope has become a "
        "disk census — which is the answer the item refused to give")
    user = [r for r in roots(box) if r.label == "user-site"][0]
    assert user.interpreter == "python3.14", (
        "the user-site row lost its interpreter tag, and the interpreter is the "
        "half of that row that matters: it is a different runtime from the venvs")


def test_a_planted_dist_info_reaches_the_inventory_and_the_scan_target(box, tmp_path):
    """Clause 2's planted-fail proof, and the reason the reader is worth trusting.

    A new distribution planted in a temp root has to show up twice: in the
    component rows, and inside the synthesised file the scanner is handed. Either
    alone is weak — a glob that finds the metadata but writes no target is a
    catalog, and a target built from a stale list is a fiction.
    """
    fake_scanner(tmp_path / "osv-scanner")
    with_path(box, tmp_path)
    os.environ[sc.OSV_DB_ENV] = str(tmp_path / "db")
    (tmp_path / "db").mkdir()
    write_dist(box["tts"], "brand-new-dep", "9.9.9")
    report = run_observed(box, tmp_path)
    rows = {r["component"] for r in report["components"]}
    assert "brand-new-dep==9.9.9@qwen3-tts" in rows, sorted(rows)
    target = next(Path(s["lockfile"]) for s in report["sites"] if s["site"] == "qwen3-tts")
    assert "brand-new-dep==9.9.9" in target.read_text(encoding="utf-8")
    # 9 installed rows before the plant, 10 after: three in the TTS venv, five in
    # the vLLM one, one in the account user-site.
    assert report["component_count"] == 10, report["component_count"]
    assert "brand-new-dep==9.9.9@qwen3-tts" in rows


def test_each_row_keeps_its_site_and_interpreter_so_the_answer_is_where(box):
    """The query the item's whole framing is for, answered on one row.

    "69 assets, 3 models, 4 agents ... llama-3-8b: one critical" — the demo's
    value was that a component name resolved to a place. `youtube-transcript-api`
    is pinned here to the 3.14 user-site because that install is #688's exposure
    baseline: the one real `pip install --user --break-system-packages` event this
    box recorded, into an interpreter no declared lockfile describes. Tagged to
    the wrong interpreter, that row would say the box's one real install is
    somewhere it is not.
    """
    rows = inv.inventory(roots(box))
    yt = [r for r in rows if r.name == "youtube-transcript-api"]
    assert len(yt) == 1, yt
    assert yt[0].version == "1.2.4"
    assert yt[0].interpreter == "python3.14"
    assert yt[0].site == "user-site"
    assert str(box["user"]) == yt[0].root
    same_name = sorted(r.site for r in rows if r.name == "requests")
    assert same_name == ["qwen3-tts", "vllm-djev"], (
        "the same name+version installed in two venvs collapsed into one row or "
        "went missing: an inventory keyed on the name alone cannot say which "
        "engine to stop using, which is the query the artifact is for")
    assert {r.interpreter for r in rows} == {"python3.12", "python3.14"}


def test_an_absent_root_is_a_scope_gap_and_never_a_zero_component_set(box):
    """The #688 honesty rule applied to the inventory's own inputs.

    A launcher that execs a venv with no site-packages is a broken launcher, or
    this module is running on the wrong machine. Either way `[]` is a report of
    nothing, so the gap is named where a reader of the artifact will see it.
    """
    (box["repo"] / "agent-services" / "bin" / "start-omni.sh").write_text(
        'VIRTUAL_ENV="$HOME/lloyd/.venvs/vllm-omni"\\n', encoding="utf-8")
    gaps = inv.scope_gaps(box["repo"], box["home"])
    assert any("vllm-omni" in g for g in gaps), gaps
    assert inv.installed_distributions(box["home"] / "lloyd/.venvs/nope") == []


def test_a_launcher_the_scope_claim_does_not_admit_is_reported_not_absorbed(box):
    """The open-set rule: a hand-maintained allowlist cannot close a property
    over a corpus that grows.

    `SCOPE_FILES` is a claim, and `launcher_venv_names` reads every `.sh`/`.conf`
    under `agent-services`. A new engine's launcher naming a new venv therefore
    has to surface as a scope gap, because the alternative is that the inventory
    quietly stops covering the newest engine — which is the shape of every stale
    allowlist this repo has had to widen after the fact.
    """
    (box["repo"] / "agent-services" / "bin" / "start-newengine.sh").write_text(
        'VIRTUAL_ENV="$HOME/lloyd/.venvs/newengine/lib"\n', encoding="utf-8")
    origin = inv.launcher_venv_names(box["repo"])
    assert origin["newengine"] == "agent-services/bin/start-newengine.sh"
    gaps = inv.scope_gaps(box["repo"], box["home"])
    assert any("start-newengine.sh" in g and "SCOPE_FILES" in g for g in gaps), gaps
    assert "newengine" not in inv.service_venv_names(
        box["repo"], scope_files=inv.SCOPE_FILES)


# ── clause 3 — the observed set is scanned, and the delta is a number ───────


def test_the_observed_delta_is_counted_by_name_not_version(box, tmp_path):
    """Both advisory numbers the artifact carries, and the rule that splits them.

    The delta answers "what does this box run that no declared file even NAMES", so
    it is keyed on name: `vllm` is installed at a version `requirements.lock` does
    not carry and is still not counted, because a version drift of a named package
    is a pinning question with a different remedy, and folding drift into the delta
    would bury the real blind spots under a few hundred rows of it. `pip` is
    undeclared and unadvised, so it is a component row and never a delta row.

    Per installed copy, `gradio` and `pillow` in two venvs are two advisories each —
    two environments to patch — while the same pair deduplicated into one target is
    one row apiece for whoever works the package list. That is why the three sites
    sum to 6 missed advisories while the merged target reports 4: the pair twice,
    plus `pip` and `youtube-transcript-api` once each. Both numbers travel because
    the reader who needs one does not want the other, and an artifact that printed
    only one would silently answer a question nobody asked.
    """
    fake_scanner(tmp_path / "osv-scanner")
    with_path(box, tmp_path)
    (tmp_path / "db").mkdir()
    os.environ[sc.OSV_DB_ENV] = str(tmp_path / "db")
    report = run_observed(box, tmp_path)
    assert report["scan_status"] == "completed", report["reason"]

    tts = next(s for s in report["sites"] if s["site"] == "qwen3-tts")
    assert tts["advisories_total"] == 3, tts
    assert tts["missed_advisories"] == 2, tts
    assert tts["missed_packages"] == ["gradio", "pillow"], tts

    djev = next(s for s in report["sites"] if s["site"] == "vllm-djev")
    assert djev["advisories_total"] == 5, djev
    assert djev["missed_advisories"] == 3, djev
    assert djev["missed_packages"] == ["gradio", "pillow", "pip"], djev
    assert "vllm" not in djev["missed_packages"], (
        "a package the declared set names was counted as a blind spot because its "
        "version differs — that is drift, not a missing component")

    user = next(s for s in report["sites"] if s["site"] == "user-site")
    assert user["missed_packages"] == ["youtube-transcript-api"], user

    assert report["advisories_in_observed_set_total"] == 9, report["sites"]
    assert report["advisories_in_observed_set_that_declared_lockfiles_missed"] == 6

    merged = report["merged_scan"]
    assert merged["scan_status"] == "completed", merged
    assert merged["packages_scanned"] == 6, merged
    assert merged["advisories_in_observed_set"] == 6, merged
    assert merged["advisories_in_observed_set_that_declared_lockfiles_missed"] == 4
    assert merged["distinct_missed_package_names"] == [
        "gradio", "pillow", "pip", "youtube-transcript-api"], merged
    assert 4 < 6, (
        "the deduplicated count is no longer below the per-copy sum, which can only "
        "mean the shared pair stopped being installed in both venvs — see `box`")


def test_the_parsed_package_count_is_pinned_to_the_inventory_size(box, tmp_path):
    """Clause 3's last words: a target the parser choked on cannot read as zero.

    `found N packages` is what the scanner says it read. Here it lies low —
    reports a valid empty report and counts one package against a three-package
    target — and the site has to be marked as a mismatch rather than contributing
    its zero to the fleet total. Without this the failure is invisible in exactly
    the way #688's `scanner_absent` would have been, had nothing read it.
    """
    fake_scanner(tmp_path / "osv-scanner", counts=False)
    with_path(box, tmp_path)
    (tmp_path / "db").mkdir()
    os.environ[sc.OSV_DB_ENV] = str(tmp_path / "db")
    report = run_observed(box, tmp_path)
    assert report["target_parse_mismatches"], (
        "a scanner that never reported how much it parsed was accepted as a clean "
        "scan: that is the hole clause 3 exists to close")
    tts = next(s for s in report["sites"] if s["site"] == "qwen3-tts")
    assert tts["packages_scanned"] is None
    assert tts["components_declared_in_target"] == 3
    assert not tts["parse_count_matches_inventory"]


def test_a_site_whose_scan_fails_contributes_no_count_rather_than_a_zero(box, tmp_path):
    """A failed site makes the fleet number null, not smaller.

    The count that ships must not silently lose a site: an inventory of five venvs
    where the TTS scan crashed and the number still says "2" is a false all-clear
    about the engine with the 105 advisories on it.
    """
    fake_scanner(tmp_path / "osv-scanner", fail_for="gradio")
    with_path(box, tmp_path)
    (tmp_path / "db").mkdir()
    os.environ[sc.OSV_DB_ENV] = str(tmp_path / "db")
    report = run_observed(box, tmp_path)
    assert report["scan_status"] == "coverage-gap"
    assert report[
        "advisories_in_observed_set_that_declared_lockfiles_missed"] is None
    assert "qwen3-tts" in report["sites_not_scanned"]
    assert "No observed advisory count is reported" in report["reason"], report


def test_the_scanner_argv_carries_the_lockfile_format_prefix(box, tmp_path):
    """The synthesised targets go through the shipped argv builder, not a parallel
    one. `OsvScannerSpec.argv` prefixes the format only where the file name needs
    it: `requirements.txt:<path>` for a `.lock` (osv-scanner picks its parser by
    name and exits 127 without the hint), bare for a file already named
    `requirements.txt`. Asserting both spellings is what stops a tidy change from
    prefixing everything — a prefix on the bare name names a format its parser does
    not have, and the scan would parse nothing while exiting 0.
    """
    fake_scanner(tmp_path / "osv-scanner")
    with_path(box, tmp_path)
    (tmp_path / "db").mkdir()
    os.environ[sc.OSV_DB_ENV] = str(tmp_path / "db")
    spec, why = sc.resolve_scanner("osv-scanner")
    assert spec is not None and spec.name == "osv-scanner", why
    argv = spec.argv([Path("/x/y/requirements.txt")])
    assert "/x/y/requirements.txt" in argv, argv
    lock = spec.argv([Path("/x/y/requirements.lock")])
    assert "requirements.txt:/x/y/requirements.lock" in lock, lock




# ── clause 4 — the kinds no feed can see ────────────────────────────────────

def test_vendored_weights_are_rows_with_a_digest_and_no_advisory_source(box):
    """The weight tree, which is on no dependency list by design.

    `requirements.txt:37-42` records that the Silero VAD model is deliberately
    *not* a package dependency and is vendored instead. So it exists on no
    inventory the declared scan reads, and the row's `version` is null because an
    ONNX file carries no upstream release string — the `SOURCE` note beside it is
    the honest form of provenance, and inventing a version would be a fabrication
    in a field meant to be exact.
    """
    models = box["repo"] / "agent-services/models/silero-vad"
    models.mkdir(parents=True)
    payload = b"fake-onnx-bytes" * 40
    (models / "silero_vad.onnx").write_bytes(payload)
    (models / "SOURCE").write_text("Silero VAD v6, vendored 2026-08", encoding="utf-8")
    rows = inv.weight_rows(box["repo"])
    assert len(rows) == 1
    row = rows[0]
    assert row["advisory_source"] == "none"
    assert row["component_kind"] == "vendored-weight"
    assert row["version"] is None
    assert row["sha256"] and len(row["sha256"]) == 16
    assert row["bytes_hashed"] == "whole-file"
    assert row["size_bytes"] == len(payload)
    assert row["sha256"] == hashlib.sha256(payload).hexdigest()[:16], (
        "the digest is not sha256 of the file's own bytes, so a disclosure naming "
        "a weight by hash cannot be matched against this row")
    assert "Silero VAD v6" in row["provenance"]
    assert row["path"] == "agent-services/models/silero-vad/silero_vad.onnx"


def test_a_weight_too_big_to_hash_is_hashed_in_part_and_says_so(box):
    """The cap is disclosed, not assumed.

    A 30 GiB GGUF is not read to build an inventory. A truncated digest that does
    not say it is truncated is worse than none, because it reads as a fingerprint
    to whoever resolves a disclosure against it.
    """
    big = box["repo"] / "agent-services/models/big"
    big.mkdir(parents=True)
    (big / "model.gguf").write_bytes(b"a" * (inv.DIGEST_CAP + 1024))
    row = inv.weight_rows(box["repo"])[0]
    assert row["bytes_hashed"] == "first-4MiB"
    assert row["size_bytes"] == inv.DIGEST_CAP + 1024
    small = box["repo"] / "agent-services/models/small"
    small.mkdir(parents=True)
    (small / "w.onnx").write_bytes(b"b" * 10)
    assert {r["bytes_hashed"] for r in inv.weight_rows(box["repo"])} == {
        "first-4MiB", "whole-file"}


def test_mcp_servers_and_skills_are_rows_and_skills_keep_their_version(tmp_path):
    """The two remaining component kinds, read from where they actually live.

    The MCP section reads only real `url:`-bearing entries: `config.yaml` carries
    commented-out third-party servers, and a commented server is not a component
    the box runs. Skills live in the vault, not the checkout, so the root is a
    parameter — and most skills carry no version, which is the gap this section
    exists to make visible rather than fill in.
    """
    cfg = box_cfg(tmp_path)
    servers = inv.mcp_server_rows(cfg)
    assert [s["name"] for s in servers] == ["lloyd-mcp"]
    assert servers[0]["url"] == "http://127.0.0.1:8500/mcp"
    assert servers[0]["advisory_source"] == "none"
    assert servers[0]["version"] is None

    vault = tmp_path / "vault"
    (vault / "skills" / "deep-research").mkdir(parents=True)
    (vault / "skills" / "deep-research" / "SKILL.md").write_text(
        "---\nname: deep-research\nversion: 1.4.2\n---\n# body\n", encoding="utf-8")
    (vault / "skills" / "no-version").mkdir(parents=True)
    (vault / "skills" / "no-version" / "SKILL.md").write_text(
        "---\nname: no-version\n---\n# body\n", encoding="utf-8")
    (vault / "skills" / "no-skill-md").mkdir()
    skills = inv.skill_rows(vault)
    assert [s["name"] for s in skills] == ["deep-research", "no-version"]
    assert skills[0]["version"] == "1.4.2"
    assert skills[1]["version"] is None
    assert all(s["advisory_source"] == "none" for s in skills)
    assert all("/SKILL.md" in s["path"] for s in skills)


def box_cfg(tmp_path) -> Path:
    repo = tmp_path / "cfg-checkout"
    (repo / "agent-services").mkdir(parents=True)
    (repo / "config.yaml").write_text(textwrap.dedent("""\
        tools:
          enabled: true
        mcp_servers:
          lloyd-mcp:
            type: http
            name: lloyd-mcp
            url: http://127.0.0.1:8500/mcp
          # github:
          #   type: stdio
          #   command: npx
        other: 1
        """), encoding="utf-8")
    return repo


def test_a_non_advisory_row_cannot_move_the_verdict_by_construction(box):
    """Clause 4's mechanism, not its intent.

    `advisory_count_from` is the only thing that turns rows into a count, so the
    rule is asserted where the rows are built: double the section, add an
    arbitrarily alarming component, and the count is still zero. The reason is in
    the predicate — a row whose `advisory_source` is `none` contributes nothing,
    so there is no arithmetic path from "inventoried" to "clean".
    """
    weights = box["repo"] / "agent-services/models/wakeword"
    weights.mkdir(parents=True)
    (weights / "Lloyd.onnx").write_bytes(b"wakeword" * 11)
    rows, gaps = inv.non_advisory_rows(box["repo"], box["home"], box["repo"])
    assert rows, (
        "a planted weight tree yielded no rows, so every count below measures "
        "nothing: the weights root is what this row set is made of")
    assert inv.advisory_count_from(rows) == 0
    assert gaps, "a planted box with no HF cache must say so rather than look scanned"
    assert len(rows) == (len(inv.weight_rows(box["repo"]))
                         + len(inv.mcp_server_rows(box["repo"]))
                         + len(inv.skill_rows(box["repo"])))
    assert inv.advisory_count_from(rows + rows) == 0
    assert inv.advisory_count_from(rows + [{
        "component": "model:evil", "component_kind": "model-artifact",
        "advisory_source": "none", "advisory_count": 99,
        "name": "evil", "version": None}]) == 0, (
        "an advisory_source of `none` with an advisory_count beside it was counted: "
        "the exclusion is the clause, not the label")
    assert inv.advisory_count_from([
        {"advisory_source": "osv", "advisory_count": 3}]) == 3, (
        "the counter is zero for every row, which means it is not a rule about "
        "advisory sources at all but a constant wearing one")



def test_the_observed_half_cannot_leak_into_the_declared_verdict(box, tmp_path):
    """`attach_to_baseline` copies the declared keys and adds one of its own.

    The reason the two advisory numbers are never reconciled into one figure is
    that they answer different questions, and a merged total is how a coverage gap
    gets laundered into a clean tree. So the declared half travels unchanged — an
    observed scan reporting 9 advisories across installed copies leaves
    `advisory_count: 0` exactly as the declared scan reported it — and the observed
    half rides on one key under its own schema name.
    """
    fake_scanner(tmp_path / "osv-scanner")
    with_path(box, tmp_path)
    (tmp_path / "db").mkdir()
    os.environ[sc.OSV_DB_ENV] = str(tmp_path / "db")
    report = run_observed(box, tmp_path)
    declared = {"schema": "lloyd-supply-chain-baseline/1",
                "verdict": "no_advisories", "advisory_count": 0,
                "generated_at": "2026-10-11T12:00:00Z"}
    merged = inv.attach_to_baseline(declared, report)
    assert merged["advisory_count"] == 0
    assert merged["verdict"] == "no_advisories"
    assert merged["schema"] == "lloyd-supply-chain-baseline/1"
    assert merged["observed"]["advisories_in_observed_set_that_declared_lockfiles_missed"
                        ] == 6
    assert merged["observed"]["non_advisory_advisory_count"] == 0
    assert set(declared) <= set(merged), "attach dropped a declared key"
    assert "components" in merged["observed"] and merged["observed"]["components"]

# ── the shape of the report ─────────────────────────────────────────────────

def test_the_report_names_its_scope_so_a_reader_can_tell_which_box_it_describes(
        box, tmp_path):
    """The scope is written into the artifact, per the item's decision.

    "Decide deliberately whether the inventory's scope is everything installed or
    everything importable by the processes that run, and write the choice down."
    Both halves: the sentence, and the per-root list with the file that named each
    venv, so a reader can check the scope against the launchers without reading
    this module.
    """
    fake_scanner(tmp_path / "osv-scanner")
    with_path(box, tmp_path)
    (tmp_path / "db").mkdir()
    os.environ[sc.OSV_DB_ENV] = str(tmp_path / "db")
    report = run_observed(box, tmp_path)
    assert report["schema"] == inv.OBSERVED_SCHEMA
    assert "importable set of the processes that run" in report["scope"]["decision"]
    assert "not a disk census" in report["scope"]["decision"]
    assert report["scope"]["derived_from"] == list(inv.SCOPE_FILES)
    origins = {r["label"]: r["origin"] for r in report["scope"]["roots"]}
    assert origins["vllm-djev"] == "agent-services/bin/start-djev.sh"
    assert origins["qwen3-tts"] == (
        "agent-services/supervisor/conf.d/agent-llm-primary.conf")
    assert origins["user-site"] == "user-site"
    assert json.loads(json.dumps(report)) == report, "the report must be JSON-safe"


def test_no_scanner_yields_a_coverage_gap_and_no_number(box):
    """The #688 rule, at the inventory's own seam.

    conftest strips `PATH`, so a test run never has the scanner and this is the
    every-night state of a task run with a minimal PATH. Null, never zero — the
    bug #688 was filed about was a zero that meant "instrument missing", and this
    is the same instrument with a second place to fail.
    """
    os.environ.pop(sc.OSV_API_URL_ENV, None)
    os.environ["PATH"] = "/nonexistent-dir-for-this-test"
    report = inv.observed_scan(box["repo"], out_dir=box["home"] / "observed-here",
                               scanner_name="osv-scanner", account_home=box["home"],
                               roots=roots(box), now=NOW)
    assert report["scan_status"] == "coverage-gap"
    assert report[
        "advisories_in_observed_set_that_declared_lockfiles_missed"] is None
    assert report["advisories_in_observed_set_total"] is None
    assert "no offline scanner" in report["reason"]
    assert all(s["scan_status"] == "scanner_absent" for s in report["sites"])
    assert report["components"], (
        "the inventory is still enumerated when no scanner can run: the answer to "
        "'what is on this box' does not depend on the advisory feed")
    assert report["non_advisory_components"]
