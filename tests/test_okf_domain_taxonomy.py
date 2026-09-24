"""The knowledge/ ``domain`` vocabulary (#949).

``type`` was closed by #370/#872; ``domain`` — the axis naming the
``knowledge/<domain>/`` directory — was free text: 138 top-level directories on
2026-09-18, 90 holding two notes or fewer, ``ai`` split nine ways, 146 distinct
``domain:`` values. These tests pin the code half: a literal vocabulary that
the filesystem cannot widen, an alias map for the named near-duplicates, a
validator that warns (never fails) on an out-of-set value, and the
``vault_write`` guard that refuses an invented domain.

Everything runs over scratch trees in ``tmp_path``; nothing reads ~/obsidian.
Clause 5 (the schema doc and four research skills' domain tables) is vault
text, landed by hand, and is not pinned here.
"""
from __future__ import annotations

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
