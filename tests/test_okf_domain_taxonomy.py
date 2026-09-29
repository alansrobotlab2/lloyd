"""The knowledge/ ``domain`` vocabulary (#949).

``type`` was closed by #370/#872; ``domain`` — the axis naming the
``knowledge/<domain>/`` directory — was free text: 138 top-level directories on
2026-09-18, 90 holding two notes or fewer, ``ai`` split nine ways, 146 distinct
``domain:`` values. These tests pin the code half: a literal vocabulary that
the filesystem cannot widen, an alias map for the named near-duplicates, a
validator that warns (never fails) on an out-of-set value, and the
``vault_write`` guard that refuses an invented domain.

The second section is #1642's tranche: the off-set spellings whose subject one of
those 47 members already names became aliases, the members whose subject has no
home stayed warnings, and the set itself was not widened by one value.

Everything runs over scratch trees in ``tmp_path`` except the ``live_vault`` test
at the bottom, which pins #949's clause 5: the schema doc and the four research
skills' domain tables name only canonical or aliased domains and no longer tell a
writer to "create if needed".
"""
from __future__ import annotations

import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.vault import okf_taxonomy as T  # noqa: E402

AI_FAMILY = ("ai-agentic", "ai-agents", "ai-coding", "ai-eigenvectors",
             "ai-engineering", "ai-inference", "ai-llms", "ai-research")


# ── clause 1: a literal the filesystem cannot widen ─────────────────────────────

def test_canonical_domains_is_a_literal_that_a_new_directory_cannot_widen(tmp_path):
    """Import the module in a child whose HOME and vault point at a scratch tree
    carrying an extra populated directory: the set must come back identical.
    A set derived from 'directories that hold content' would grow here."""
    vault = tmp_path / "obsidian"
    for d, n in (("ai", 5), ("invented-domain", 9)):
        (vault / "knowledge" / d).mkdir(parents=True)
        for i in range(n):
            (vault / "knowledge" / d / f"n{i}.md").write_text("---\ntype: research\n---\n")
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from scripts.vault import okf_taxonomy as T\n"
            "print(repr(sorted(T.CANONICAL_DOMAINS)))" % str(ROOT))
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin",
           "LLOYD_DATA": str(tmp_path / "data")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, cwd=str(tmp_path), check=True).stdout
    assert out.strip().splitlines()[-1] == repr(sorted(T.CANONICAL_DOMAINS))
    assert "invented-domain" not in T.CANONICAL_DOMAINS


def test_the_vocabulary_is_written_out_in_the_source():
    src = (ROOT / "scripts/vault/okf_taxonomy.py").read_text()
    block = src.split("CANONICAL_DOMAINS = frozenset({", 1)[1].split("})", 1)[0]
    for d in T.CANONICAL_DOMAINS:
        assert f'"{d}"' in block, f"{d} is not a literal in CANONICAL_DOMAINS"
    for banned in ("iterdir", "glob(", "listdir", "scandir", "os.walk"):
        assert banned not in src, f"okf_taxonomy reads the filesystem ({banned})"
    assert 40 <= len(T.CANONICAL_DOMAINS) <= 60, len(T.CANONICAL_DOMAINS)


# ── clause 2: aliases fold onto the canonical set ───────────────────────────────

def test_every_alias_resolves_into_the_canonical_set():
    for alias, target in T.DOMAIN_ALIASES.items():
        assert target in T.CANONICAL_DOMAINS, f"{alias} -> {target} is not canonical"
        assert alias not in T.CANONICAL_DOMAINS, f"{alias} is both alias and canonical"
    for a in AI_FAMILY:
        assert T.DOMAIN_ALIASES[a] == "ai"
    assert T.DOMAIN_ALIASES["Robotics"] == "robotics"
    assert T.DOMAIN_ALIASES["robots"] == "robotics"


def test_normalize_domain_folds_case_and_separators():
    assert T.normalize_domain("Robotics") == "robotics"
    assert T.normalize_domain("ROBOTS") == "robotics"
    assert T.normalize_domain("AI_Research") == "ai"
    assert T.normalize_domain("robotics") == "robotics"
    assert T.normalize_domain("made-up") == "made-up"
    assert T.is_known_domain("ai-llms") and not T.is_known_domain("made-up")


# ── clause 3: the validator warns, names value and file, never fails ────────────

def _vault(tmp_path: Path) -> Path:
    root = tmp_path / "v"
    notes = {
        "knowledge/ai/good.md": "---\ntype: research\ndomain: ai\n---\n",
        "knowledge/ai/alias.md": "---\ntype: research\ndomain: ai-agents\n---\n",
        "knowledge/ai/nodomain.md": "---\ntype: research\n---\n",
        "knowledge/x/bad.md": "---\ntype: research\ndomain: tooling-infra\n---\n",
        # Outside knowledge/ the domain axis is not governed.
        "projects/p.md": "---\ntype: reference\ndomain: whatever\n---\n",
    }
    for rel, text in notes.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def _validate(root: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/vault/validate_okf.py"), "--root", str(root), *extra],
        capture_output=True, text=True, cwd=str(ROOT))


def test_the_validator_warns_on_an_unknown_domain_naming_value_and_file(tmp_path):
    root = _vault(tmp_path)
    strict = _validate(root, "--strict")
    assert strict.returncode == 2, strict.stdout + strict.stderr
    assert "knowledge/x/bad.md: unknown domain 'tooling-infra'" in strict.stdout
    assert "alias.md" not in strict.stdout and "good.md" not in strict.stdout
    assert "whatever" not in strict.stdout
    plain = _validate(root)
    assert plain.returncode == 0, "an out-of-set domain must not fail the vault-wide gate"
    assert "VIOLATIONS : 0" in plain.stdout


def test_the_validator_imports_its_domain_set_rather_than_copying_it():
    src = (ROOT / "scripts/vault/validate_okf.py").read_text()
    assert "from scripts.vault.okf_taxonomy import is_known_domain" in src
    for d in ("stack-updates", "opentelemetry", "computer-vision"):
        assert f'"{d}"' not in src, f"validate_okf.py restates the domain literal {d}"


# ── clause 4: the write path refuses an invention, rewrites an alias ────────────

@pytest.fixture
def scratch(tmp_path, monkeypatch):
    import agent_mcp.vault as vault_mod
    monkeypatch.setattr(vault_mod, "VAULT", tmp_path)
    monkeypatch.setattr(vault_mod, "AUDIT_LOG_DIR", tmp_path / "audit")
    monkeypatch.setattr(vault_mod, "AUDIT_LOG_FILE", tmp_path / "audit" / "writes.jsonl")
    return types.SimpleNamespace(root=tmp_path, mod=vault_mod)


def _write(scratch, rel: str, domain: str | None) -> dict:
    fm = "type: research\n" + (f"domain: {domain}\n" if domain is not None else "")
    return scratch.mod._vault_write({"path": rel, "content": f"---\n{fm}---\n# Note\n"})


def test_an_invented_domain_is_refused_by_name_and_creates_nothing(scratch):
    result = _write(scratch, "knowledge/tooling-infra/n.md", "tooling-infra")
    assert "error" in result, f"an invented domain was accepted: {result}"
    assert "tooling-infra" in result["error"]
    assert result.get("invalid_domain") == "tooling-infra"
    assert not (scratch.root / "knowledge/tooling-infra").exists()


def test_an_aliased_domain_lands_canonical_and_is_reported(scratch):
    result = _write(scratch, "knowledge/ai/n.md", "ai-agents")
    assert result.get("success") is True, result
    landed = (scratch.root / "knowledge/ai/n.md").read_text()
    assert "domain: ai\n" in landed and "ai-agents" not in landed
    assert result["domain_normalized"] == {"from": "ai-agents", "to": "ai"}


def test_a_canonical_domain_is_written_verbatim(scratch):
    result = _write(scratch, "knowledge/robotics/n.md", "robotics")
    assert result.get("success") is True, result
    assert "domain_normalized" not in result


def test_a_note_with_no_domain_is_writable_and_lands_in_its_directory(scratch):
    result = _write(scratch, "knowledge/hardware/n.md", None)
    assert result.get("success") is True, result
    assert (scratch.root / "knowledge/hardware/n.md").is_file()


def test_the_domain_guard_governs_only_knowledge(scratch):
    result = _write(scratch, "projects/n.md", "anything-goes")
    assert result.get("success") is True, result


# --- clause 5: the guidance a writer reads names only the closed set -------

import pwd  # noqa: E402

# The vault is read off the passwd home, never Path.home(): under a gate's
# isolated HOME that name is the round's symlink farm (CLAUDE.md §4.3a).
LIVE_VAULT = Path(pwd.getpwuid(__import__("os").getuid()).pw_dir) / "obsidian"
GUIDANCE = (
    "knowledge/KNOWLEDGE_SCHEMA.md",
    "skills/research-agent/SKILL.md",
    "skills/deep-research/SKILL.md",
    "skills/medium-research/SKILL.md",
    "skills/web-search-and-fetch/SKILL.md",
)
_KNOWLEDGE_DIR_RE = re.compile(r"knowledge/([A-Za-z0-9_.-]+)/")
_BARE_DIR_RE = re.compile(r"`([a-z0-9-]+)/`")


def _named_domains(text: str) -> set[str]:
    named = set(_KNOWLEDGE_DIR_RE.findall(text))
    # web-search-and-fetch and the schema list folders bare: `ai/`, `papers/`.
    for line in text.splitlines():
        if "domain" in line.lower() or "knowledge/" in line:
            named.update(_BARE_DIR_RE.findall(line))
    named.discard("knowledge")  # the segment root, `knowledge/`, is no domain
    return named


# ── #1642: the alias tranche, and the closed set it may not widen ───────────────
#
# #949 shipped the 47-member set and 10 aliases; the rest of the census was left
# unstarted. Measured on the live vault before this tranche: 2759 `.md` under
# `knowledge/`, 2553 carrying a non-empty `domain:`, 145 distinct values — 46
# canonical (2366 files), 8 aliased (77 files) and **91 values across 110 files
# off-set**. This section pins the tranche that closed 16 of those 91 values / 24
# of those 110 files (off-set is now 75 values / 86 files, and `DOMAIN_ALIASES`
# holds 26 keys) without touching the set, and pins that the rest still warn.

SHIPPED_47 = frozenset({
    "agent-lloyd", "agents", "ai", "application", "asimov", "aveva", "books",
    "browser", "computer-vision", "distributed-systems", "evaluation", "foundational",
    "general", "hardware", "inference", "infrastructure", "inner-voice", "llm",
    "llm-inference", "llm-serving", "lloyd", "machine-learning", "misc",
    "mission-control", "ml", "ml-inference", "neuroscience", "observability",
    "openclaw", "opentelemetry", "ops", "papers", "patterns", "reading", "research",
    "robotics", "security", "software", "stack-updates", "synthesis", "system",
    "systems", "thinking", "tools", "video-summaries", "vllm", "youtube",
})

# The tranche this round adds: every value the item names plus the census spellings
# whose subject one member already names (see the rule in `DOMAIN_ALIASES`).
TRANCHE_1642 = {
    "local-llm-serving": "llm-serving",
    "local-llm": "llm-serving",
    "model-serving": "llm-serving",
    "serving": "llm-serving",
    "llm-evaluation": "evaluation",
    "model-architecture": "ml",
    "model-optimization": "ml",
    "3d-printing-robotics": "robotics",
    "humanoid-robotics": "robotics",
    "quadruped-robotics": "robotics",
    "robotic-perception": "robotics",
    "robotics-perception": "robotics",
    "cv": "computer-vision",
    "miscellaneous": "misc",
    "infra": "infrastructure",
    "llm-agents": "agents",
}

# The 10 keys `767f72d5` (#949) shipped. Held apart from `TRANCHE_1642` so "every
# newly added key" stays a derived question — a live-tree node that iterated this
# literal would silently stop covering a key someone adds next month.
LEGACY_949 = ("ai-agentic", "ai-agents", "ai-coding", "ai-eigenvectors",
              "ai-engineering", "ai-inference", "ai-llms", "ai-research",
              "Robotics", "robots")

# Values the census found whose SUBJECT has no member of its own. Naming them here
# is what makes clause 4 checkable: these are the warnings this round must not
# silence, and the 2026-09-29 ruling on #1642's owed entry 1 is what settles their
# fate — none is promoted and none is folded, so each keeps warning. The bar that
# replaced the open question is the block above `CANONICAL_DOMAINS`, pinned below
# by `test_the_block_above_the_domain_set_records_the_made_ruling`.
RESIDUAL_NO_HOME = (
    "tts-voice", "voice-tts", "embodied-ai", "gpu", "rag", "databases", "science",
    "skills", "autonomy", "nightly", "vlm", "voting", "web", "entrepreneurship",
    "reverse-proxy",
)


def test_the_1642_tranche_all_reads_as_a_shipped_canonical_member():
    """Clause 1: each new key is known AND folds onto a member that was already in
    the set — `is_known_domain` alone would also be true of a value minted as a
    48th canonical, which is exactly what the item forbids.

    The literal is pinned against the map first: a key added to `DOMAIN_ALIASES`
    without being added here fails this node, rather than quietly escaping every
    assertion that iterates the tranche."""
    assert set(T.DOMAIN_ALIASES) - set(LEGACY_949) == set(TRANCHE_1642), (
        f"unpinned new keys: {sorted(set(T.DOMAIN_ALIASES) - set(LEGACY_949) - set(TRANCHE_1642))}; "
        f"keys not in the map: {sorted(set(TRANCHE_1642) - set(T.DOMAIN_ALIASES))}")
    for alias, target in TRANCHE_1642.items():
        assert T.DOMAIN_ALIASES[alias] == target, alias
        assert T.is_known_domain(alias) is True, alias
        folded = T.normalize_domain(alias)
        assert folded == target, f"{alias} folded to {folded!r}, not {target!r}"
        assert folded in SHIPPED_47, f"{alias} -> {folded} is not a shipped member"
    # Separators and case fold the same way for the new keys as for the old ones.
    assert T.normalize_domain("Local_LLM_Serving") == "llm-serving"
    assert T.normalize_domain("ROBOTICS-PERCEPTION") == "robotics"


def test_the_closed_set_is_the_same_47_values_that_shipped():
    """Clause 2, pinned by duplication rather than by a count.

    `SHIPPED_47` above is the set as `767f72d5` wrote it, transcribed into the test
    on purpose: a `len() == 47` assertion would still pass if this round promoted an
    off-set value and dropped a shipped one, and promotion-by-accident is the failure
    the whole clause exists to catch. Two places to edit is the cost of making a
    silent widening loud.
    """
    assert T.CANONICAL_DOMAINS == SHIPPED_47, (
        f"added: {sorted(T.CANONICAL_DOMAINS - SHIPPED_47)}; "
        f"removed: {sorted(SHIPPED_47 - T.CANONICAL_DOMAINS)}")
    assert len(T.CANONICAL_DOMAINS) == 47, len(T.CANONICAL_DOMAINS)
    # And the tranche specifically promoted nothing: every key it aliased is still
    # outside the set, so its subject is reachable only through the alias map.
    for alias in TRANCHE_1642:
        assert alias not in T.CANONICAL_DOMAINS, alias
    for residual in RESIDUAL_NO_HOME:
        assert residual not in T.CANONICAL_DOMAINS, residual


def test_no_alias_key_shadows_another_or_diverges_from_a_canonical_member():
    """Two hazards in how the lookup map is built (`_DOMAIN_ALIAS_LOOKUP` keys
    `_lookup_form(k)`, and `normalize_domain` returns at the canonical check before it
    reads the map):

    * two keys whose folded forms collide overwrite each other, so one of the two
      spellings silently stops folding — 26 keys must produce 26 distinct forms;
    * a key whose folded form IS a canonical member is unreachable through the alias
      map. That is only harmless when the alias target is that same member, as it is
      for `Robotics` -> `robotics`; a key shadowing a *different* member would mean
      the two routes give different answers for one spelling.
    """
    forms = {}
    for key, target in T.DOMAIN_ALIASES.items():
        forms.setdefault(T._lookup_form(key), []).append(key)
    assert len(T._DOMAIN_ALIAS_LOOKUP) == len(T.DOMAIN_ALIASES) == 26, forms
    assert [f for f, ks in forms.items() if len(ks) > 1] == [], forms
    divergent = [(k, T._lookup_form(k), t) for k, t in T.DOMAIN_ALIASES.items()
                 if T._lookup_form(k) in T.CANONICAL_DOMAINS
                 and T._lookup_form(k) != t]
    assert divergent == [], divergent
    # `Robotics` is the one identity-shadowed key, and it still folds correctly:
    # `normalize_domain` answers `robotics` from the canonical check, which is what
    # the alias said anyway.
    assert T.normalize_domain("Robotics") == T.DOMAIN_ALIASES["Robotics"] == "robotics"


def _alias_tree(tmp_path: Path, values) -> Path:
    """A scratch vault with one knowledge note per value, and nothing else that can
    warn — so the warning list out of the validator is exactly the set under test."""
    root = tmp_path / "v"
    for i, value in enumerate(values):
        p = root / "knowledge" / "domain-aliases" / f"n{i:02d}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"---\ntype: research\ndomain: {value}\n---\n")
    return root


def test_every_alias_key_clears_the_validator_warning(tmp_path):
    """Clause 3: all 26 keys of `DOMAIN_ALIASES` — the 10 from #949 and the 16 from
    #1642 — run through the real `validate_okf.py` over a fixture tree, and not one
    of them may draw an `unknown domain` warning.

    Driven as a subprocess through the shipped script rather than through
    `is_known_domain` because the clause is about the file a weekly job reads: the
    validator imports the predicate (`validate_okf.py` must not restate the set), and
    that import is the seam here.

    `--strict` is what makes this measurable rather than vacuous. Non-strict only
    counts warnings on the summary line and names nothing, so asserting the absence of
    the words `unknown domain` against it would pass whatever the vocabulary did;
    `--strict` prints each warning and exits 2. The assertion is therefore `warnings:
    0` plus exit 0 under `--strict` — a key that stopped folding cannot hide.
    """
    keys = sorted(T.DOMAIN_ALIASES)
    assert len(keys) == 26, keys
    root = _alias_tree(tmp_path, keys)
    # The denominator beside the verdict: `warnings : 0` over a tree that silently
    # held fewer notes than keys would be a measurement of nothing.
    written = list((root / "knowledge" / "domain-aliases").glob("*.md"))
    assert len(written) == len(keys), (len(written), len(keys))
    assert "scanned 26 concept files" in _validate(root).stdout, _validate(root).stdout
    strict = _validate(root, "--strict")
    assert strict.returncode == 0, (
        "an aliased domain still warns, so --strict refuses the tree:\n" + strict.stdout)
    assert "VIOLATIONS : 0" in strict.stdout, strict.stdout
    assert "warnings   : 0" in strict.stdout, strict.stdout
    assert "unknown domain" not in strict.stdout, strict.stdout
    # And the same tree non-strict: warnings counted, exit still 0.
    out = _validate(root)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "warnings   : 0" in out.stdout, out.stdout


def test_a_value_with_no_canonical_home_still_warns(tmp_path):
    """Clause 4: aliasing 15 spellings did not make the vocabulary accept anything.

    The residuals are pinned by name, and `embodied-ai` is the one the clause names —
    2 files on the live vault, off-set before this tranche and off-set after it. The
    assertion is on the EXACT set of warned values, read out of `--strict` output
    because non-strict names nothing, so a surface widened to swallow the tranche fails
    here as loudly as one narrowed past the residuals. Two near-misses of my own
    selection are in the pinned set for the same reason: `tooling-infra` is the value
    the write guard and the older tests use as their invention, and `local-inference`
    is the spelling deliberately left off-set because three shipped members already
    name inference — both must still warn.
    """
    values = sorted(set(RESIDUAL_NO_HOME) | {"tooling-infra", "local-inference"})
    root = _alias_tree(tmp_path, values)
    strict = _validate(root, "--strict")
    warned = set(re.findall(r"unknown domain '([^']+)'", strict.stdout))
    assert warned == set(values), (sorted(warned), sorted(values))
    assert "embodied-ai" in warned, strict.stdout
    assert strict.returncode == 2, strict.stdout
    # And it stays a warning, never a violation: the gate must still exit 0.
    out = _validate(root)
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"warnings   : {len(values)}" in out.stdout, out.stdout
    assert "VIOLATIONS : 0" in out.stdout, out.stdout


def test_the_tranche_is_measured_on_the_live_vault_not_only_on_fixtures(tmp_path):
    """The census this tranche was selected by, re-run against the real vault.

    The fixtures prove what the alias map does; only the live tree proves the map was
    written against the values that are actually on disk. Measured before the tranche:
    91 off-set values across 110 files, of which these 16 keys cover 24 files, leaving
    75 values / 86 files. What this node asserts is the part of that which stays true
    as the vault grows: no key of the alias map is still off-set, every value the
    2026-09-29 ruling left off-set is still off-set, and the set is still 47.

    Skips, never fails, when `~/obsidian` is not readable — a vault it cannot see would
    make the census a measurement of nothing, which reads exactly like a pass.
    """
    if not (LIVE_VAULT / "knowledge").is_dir():
        pytest.skip("live ~/obsidian not readable from this tree")
    fm_re = re.compile(r"^---\n(.*?)\n---", re.S)
    dom_re = re.compile(r"^domain:[ \t]*(.*?)[ \t]*$", re.M)
    values: set[str] = set()
    for path in (LIVE_VAULT / "knowledge").rglob("*.md"):
        m = fm_re.match(path.read_text(encoding="utf-8", errors="replace"))
        if not m:
            continue
        d = dom_re.search(m.group(1))
        if d and d.group(1).strip().strip("\"'"):
            values.add(d.group(1).strip().strip("\"'"))
    off_set = {v for v in values if not T.is_known_domain(v)}
    assert len(T.CANONICAL_DOMAINS) == 47
    assert off_set.isdisjoint(T.DOMAIN_ALIASES), (
        f"aliased values still counted off-set: {sorted(off_set & set(T.DOMAIN_ALIASES))}")
    # Derived, not the literal: a key added to the map next month is covered by this
    # node only if the node asks the map which keys are new.
    for alias in sorted(set(T.DOMAIN_ALIASES) - set(LEGACY_949)):
        assert alias not in off_set, alias
    for residual in RESIDUAL_NO_HOME:
        assert residual in off_set, residual


@pytest.mark.live_vault
def test_live_domain_guidance_names_only_known_domains():
    if not (LIVE_VAULT / GUIDANCE[0]).is_file():
        pytest.skip("live ~/obsidian not readable from this tree")
    offenders: dict[str, list[str]] = {}
    for rel in GUIDANCE:
        text = (LIVE_VAULT / rel).read_text(encoding="utf-8")
        named = _named_domains(text)
        bad = sorted(d for d in named if not T.is_known_domain(d))
        if bad:
            offenders[rel] = bad
        assert "create if needed" not in text, rel
    assert offenders == {}, offenders
    schema = (LIVE_VAULT / GUIDANCE[0]).read_text(encoding="utf-8")
    assert "general/`,etc." not in schema
    assert "CANONICAL_DOMAINS" in schema


# ── #1854: the ruling is on the record, not still pending ────────────────────────
#
# #1642's owed entry 1 was ruled on 2026-09-29 — no residual off-set value
# promoted, none given a guessed fold, and a promotion bar written down in place
# of the open question — and owed entry 3 closed directory consolidation as a
# final disposition. What that left behind were three comments in
# `okf_taxonomy.py` and two in this file pointing at a ruling they called
# unmade, which every later round and triage pass re-derives as an open
# question. These nodes keep the record straight in both directions: the stale
# pointers must stay gone, and the disposition, the bar and the two declined
# subjects must stay legible in the block above `CANONICAL_DOMAINS`.

import ast  # noqa: E402

SRC_PATH = "scripts/vault/okf_taxonomy.py"
TEST_PATH = "tests/test_okf_domain_taxonomy.py"
# The commit the ruling postdates. `okf_taxonomy.py` was byte-identical to its
# copy of that commit at this round's base, which makes it both the positive
# control for the absence check and the reference for "comment text only".
PRE_RULING = "7ec50f98"

# The three phrasings that read as an open question. Written as adjacent
# fragments on purpose: a scan of THIS file for the pattern must not match the
# line that declares it, so no line below holds any of the three contiguously.
STALE_RULING = re.compile(
    r"has not been " r"made"
    r"|owed " r"1"
    r"|has not " r"ruled")


def _git_blob(rev: str, rel: str) -> str:
    """`rev`'s copy of `rel`, read out of the repo this checkout belongs to."""
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:{rel}"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"git show {rev}:{rel} failed: {r.stderr[:200]}"
    return r.stdout


def _block_above(name: str) -> str:
    """The run of `#` lines directly above the assignment to `name`, folded into one
    whitespace-normalised line with every line's leading `#` marker removed.

    Both folds matter: the prose gets re-wrapped whenever the file is edited, and a
    sentence that straddles a line break only matches once the `# ` that split it is
    gone — otherwise a multi-line phrase here would pin one particular wrap rather
    than the claim the comment makes.
    """
    lines = (ROOT / SRC_PATH).read_text().splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith(f"{name} ="))
    start = at
    while start > 0 and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    body = [re.sub(r"^\s*#+\s*", "", line) for line in lines[start:at]]
    return " ".join(" ".join(body).split())


def _unsaid(block: str, *phrases: str) -> list[str]:
    """The phrases `block` does not say, compared whitespace-flexibly, case-blind."""
    return [p for p in phrases
            if re.search(r"\s+".join(re.escape(w) for w in p.split()),
                         block, re.I) is None]


def test_no_comment_in_the_taxonomy_or_here_still_claims_the_ruling_is_open():
    """Clause 1: `STALE_RULING` matches no line of `okf_taxonomy.py` — the three
    sites this round rewrote are the `CANONICAL_DOMAINS` preamble, the no-home
    residual sentence in the `DOMAIN_ALIASES` block, and the quantization
    exclusion — and none of this file either (the `RESIDUAL_NO_HOME` comment and
    the live-vault census docstring).

    A zero-match absence check proves nothing unless the pattern can match, so the
    same alternation is run against the pre-ruling blobs of both files and EVERY
    alternative must find something there: the script blob carries the
    not-yet-made sentence on one line and the owed-entry pointer on two, this
    file's blob carries the owed-entry pointer and the has-not-ruled sentence on
    one line each. Mistype an alternative and the control goes red; bring the
    stale prose back and the first assert goes red.
    """
    offenders = {}
    for rel in (SRC_PATH, TEST_PATH):
        hits = [n for n, line in enumerate((ROOT / rel).read_text().splitlines(), 1)
                if STALE_RULING.search(line)]
        if hits:
            offenders[rel] = hits
    assert offenders == {}, f"stale-ruling pointers still on lines {offenders}"
    before = _git_blob(PRE_RULING, SRC_PATH) + _git_blob(PRE_RULING, TEST_PATH)
    for alt in STALE_RULING.pattern.split("|"):
        assert alt in before, (
            f"positive control broken at {PRE_RULING}: {alt!r} matches nothing "
            f"there, so the zero-match assert above would pass on a vacuous pattern")


def test_the_block_above_the_domain_set_records_the_made_ruling():
    """Clause 2: the comment block above `CANONICAL_DOMAINS` says the ruling is
    MADE and that no residual off-set value was promoted to canonical or given a
    guessed fold — which is why the closed set is still the 47 that shipped.

    Asserted as separate signals rather than one blob hash: the paragraph may be
    re-wrapped for width without failing here, but it cannot quietly lose a claim.
    """
    block = _block_above("CANONICAL_DOMAINS")
    missing = _unsaid(
        block,
        "were a human ruling",          # past tense: the question is closed
        "Both are MADE",
        "2026-09-29",
        "the 47 members below are kept exactly as shipped",
        "no residual off-set value is promoted to canonical",
        "none is given a guessed fold",
        "aliasing a value that has no member of its own",
    )
    assert missing == [], (
        f"the block above CANONICAL_DOMAINS no longer says: {missing}")


def test_the_same_block_records_the_promotion_bar_and_its_three_conditions():
    """Clause 3: the bar is recorded with its threshold, so the next reader does not
    re-open the disposition to ask whether a value deserves promotion. Three
    conditions, and the third is about the shape of the change, not its size: (a)
    the subject counts every spelling that would fold to it and must hold at least
    3 notes on disk, (b) no current member may name that subject, (c) the edit is
    one hand-written value added to the literal with the count in the commit
    message — never a derived or bulk write."""
    block = _block_above("CANONICAL_DOMAINS")
    missing = _unsaid(
        block,
        "THE PROMOTION BAR",
        "A residual value joins the set below",
        "the 48th member",
        "only when all three hold",
        "(a) its SUBJECT, counting every spelling that would fold to it",
        "holds >= 3 notes on disk",
        "(b) no current member names that subject",
        "(c) the change is one hand-written value added to the literal below",
        "with its file count in the commit message",
    )
    assert missing == [], f"the promotion bar is incomplete: {missing}"


def test_the_same_block_declines_the_two_subjects_that_clear_the_bar():
    """Clause 4: voice/TTS at 5 files and retrieval/RAG at 3 are named as the only
    two subjects clearing the bar today, both DECLINED as they stand at 8 files of
    the ~2553 tagged notes, with the ~10 revisit trigger; and directory
    consolidation is RULED OUT — the sentence must keep the words "not deferred",
    because "deferred" is the word that would send the next round looking for the
    migration that owes it.
    """
    block = _block_above("CANONICAL_DOMAINS")
    missing = _unsaid(
        block,
        "exactly two subjects clear",
        "voice/TTS",
        "voice-mode`` 1 = 5 files",
        "retrieval/RAG",
        "retrieval`` 1 = 3 files",
        "Both are DECLINED as they stand",
        "8 files of the ~2553 tagged notes",
        "either subject reaches ~10 notes",
        "Directory consolidation is RULED OUT",
        "not deferred",
        "final disposition",
    )
    assert missing == [], f"the declined-by-ruling record is incomplete: {missing}"


def test_the_ruling_rewrote_comment_text_and_nothing_else():
    """Clause 5, the half the counts cannot pin: the AST of `okf_taxonomy.py` in
    this tree is byte-for-byte the AST of the pre-ruling blob, so the only thing
    that changed is comment text — comments are not in the AST, so adding a member
    to `CANONICAL_DOMAINS` or a key to `DOMAIN_ALIASES` makes the two dumps
    differ. `len() == 47` and `len() == 26` would also pass on a promotion paired
    with a retirement, which is why this node asks for structural identity instead.

    The values themselves stay pinned by the pre-existing nodes, unmodified:
    `test_the_closed_set_is_the_same_47_values_that_shipped` (set equality against
    `SHIPPED_47`, and `len(T.CANONICAL_DOMAINS) == 47`) and
    `test_no_alias_key_shadows_another_or_diverges_from_a_canonical_member`
    (`len(T._DOMAIN_ALIAS_LOOKUP) == len(T.DOMAIN_ALIASES) == 26`).
    """
    now = ast.dump(ast.parse((ROOT / SRC_PATH).read_text(), filename=SRC_PATH))
    was = ast.dump(ast.parse(_git_blob(PRE_RULING, SRC_PATH), filename=SRC_PATH))
    assert now == was, (
        "okf_taxonomy.py changed beyond comment text since "
        f"{PRE_RULING}, and this item is a comments-only ruling record")
