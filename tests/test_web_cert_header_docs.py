"""web/vite.config.ts says what its client-cert plugin really does (#2206).

Until 2026-10-05 the `clientCertHeaders()` docstring called the injected
fingerprint header trusted, verified by the TLS layer. That stopped being true
when 5e1351f3 dropped mTLS: the HTTPS config requests no client cert, so the
plugin never fires, and the one server-side reader can only refuse. A comment
that claims an authentication property the code does not have is how the next
change comes to rely on it.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VITE = "web/vite.config.ts"


def _tracked_source() -> str:
    """The whole tracked text, with positive controls: a 0-hit below must not be
    able to come from an unread, truncated or emptied file."""
    tracked = subprocess.run(["git", "ls-files", "--", VITE], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    assert tracked == [VITE], f"{VITE} is not a tracked file"
    text = (ROOT / VITE).read_text(encoding="utf-8")
    for control in ("function clientCertHeaders()", "export default defineConfig",
                    "const httpsConfig"):
        assert control in text, f"positive control {control!r} missing from {VITE}"
    return text


def _docstring(text: str) -> str:
    found = re.search(r"/\*\*((?:(?!\*/).)*?)\*/\s*function clientCertHeaders\(\)", text, re.S)
    assert found, "clientCertHeaders() has no docstring directly above it"
    return " ".join(re.sub(r"^\s*\*", "", line).strip() for line in found.group(1).splitlines()).strip()


def _https_config(text: str) -> str:
    found = re.search(r"const httpsConfig\b.*?;\n", text, re.S)
    assert found, "httpsConfig not found"
    return found.group(0)


def test_the_trusted_input_claim_is_gone_and_the_plugin_is_still_wired():
    text = _tracked_source()
    assert not re.search(r"trusted\W{0,3}input", text, re.I)
    plugins = re.search(r"plugins:\s*\[(.*?)\]", text, re.S)
    assert plugins and "clientCertHeaders()" in plugins.group(1)


def test_the_docstring_says_the_plugin_is_inert_and_why():
    text = _tracked_source()
    doc = _docstring(text)
    assert "Inert" in doc and "5e1351f3" in doc
    assert "requestCert" in doc and "x-client-cn" in doc and "x-client-fingerprint" in doc
    # ...and the reason it gives is true of the code beside it.
    config = _https_config(text)
    for key in ("requestCert", "rejectUnauthorized"):
        assert key not in config, f"{key} is back in httpsConfig: the docstring is now false"
    assert not re.search(r"\bca\s*:", config), "a `ca:` is back in httpsConfig"
    assert "key:" in config and "cert:" in config


def test_the_docstring_names_the_one_reader_and_that_it_can_only_refuse():
    doc = _docstring(_tracked_source())
    assert "server._cert_fingerprint" in doc and "ApiPeerGate" in doc
    assert "deny-only" in doc and "403" in doc and "grants access" in doc
    # The role it describes is the code's: the reader exists, and it is called
    # after the peer-address refusal.
    server = (ROOT / "server.py").read_text(encoding="utf-8")
    assert "def _cert_fingerprint(scope)" in server
    refusal = server.index("if not _is_trusted_peer(client_host):", server.index("PRE_AUTH_PATHS or"))
    assert refusal < server.index("fp = _cert_fingerprint(scope)")


def test_the_docstring_says_cn_is_unread_and_an_empty_allowlist_refuses():
    doc = _docstring(_tracked_source())
    assert "x-client-cn has no server-side reader" in doc
    assert "agent-services/cert/clients.json" in doc and "is refused" in doc
    readers = subprocess.run(["git", "grep", "-il", "x-client-cn", "--", "*.py", ":!tests/"],
                             cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert readers == [], f"x-client-cn gained a server-side reader: {readers}"
    allowlist = ROOT / "agent-services" / "cert" / "clients.json"
    if allowlist.exists():      # untracked runtime state: absent on a fresh clone
        assert json.loads(allowlist.read_text(encoding="utf-8") or "{}") == {}, \
            "the allowlist is no longer empty: reword the docstring's last claim"


def test_the_docstring_cites_symbols_not_line_numbers():
    """The anchors #2206 itself carried drifted ~55 lines in a day."""
    assert not re.search(r":\d{2,}", _docstring(_tracked_source()))
