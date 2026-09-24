"""`app/harness/client.py`'s docstring names real files (#1048).

For four months it justified httpx by citing `inner_voice/critic.py`, a module
that had been renamed to `app/inner_voice/observer.py`, and it was the only
reference to the old name left in the tree. The docstring is also the answer to
"is `stream_chat` the only place a request leaves the process?", which it now
answers by naming the other send sites — so each path it names must exist, or
the map goes stale the same way again.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIENT = ROOT / "app" / "harness" / "client.py"


def _docstring() -> str:
    text = CLIENT.read_text()
    return text.split('"""', 2)[1]


def test_every_source_path_the_docstring_names_exists():
    named = set(re.findall(r"`((?:app|workers)/[\w/.-]+\.py)`", _docstring()))
    assert {"app/inner_voice/observer.py", "app/harness/finalizer.py"} <= named
    missing = sorted(p for p in named if not (ROOT / p).is_file())
    assert missing == [], f"client.py's docstring names files that do not exist: {missing}"


def test_the_deleted_critic_module_is_cited_nowhere():
    me = Path(__file__).resolve()
    hits = [p for p in ROOT.rglob("*.py")
            if ".venvs" not in p.parts and "node_modules" not in p.parts
            and p.resolve() != me
            and "inner_voice/critic" in p.read_text(errors="replace")]
    assert hits == [], [str(p.relative_to(ROOT)) for p in hits]
