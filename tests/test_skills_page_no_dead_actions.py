"""#1293: the Skills page posted to three routes the backend never registered
(`/skill-toggle`, `/skill-content` POST, `/skills/refresh`). `fetch` resolves
on 404/405 and none of the mutators checked `ok`, so the toggle flipped and
stayed flipped, and Save closed the editor as if SKILL.md had been written.
The page is read-only now: skills are retired and edited in the vault.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_SRC = ROOT / "web" / "src"
API_TS = WEB_SRC / "api.ts"
PAGE = WEB_SRC / "components" / "pages" / "SkillsPage.tsx"


def _sources():
    for p in WEB_SRC.rglob("*"):
        if p.suffix in (".ts", ".tsx") and p.is_file():
            yield p, p.read_text()


def test_no_file_requests_a_dead_skill_route():
    offenders = []
    for p, text in _sources():
        if "/skill-toggle" in text or "/skills/refresh" in text:
            offenders.append(str(p.relative_to(ROOT)))
        # `/skill-content` stays as a GET; any POST to it is the dead save.
        for m in re.finditer(r"/skill-content", text):
            window = text[m.end(): m.end() + 200]
            if re.search(r"method:\s*['\"]POST", window.split("fetch(")[0]):
                offenders.append(f"{p.relative_to(ROOT)} POST /skill-content")
    assert offenders == []
    api = API_TS.read_text()
    for name in ("skillToggle", "skillContentSave", "skillsRefresh"):
        assert not re.search(rf"\b{name}\s*\(", api), name
    # Positive control: the GET loaders are still there, so a wrong path
    # would not make the absence above pass.
    assert "skillContent(" in api and "/skill-content?name=" in api


def test_detail_pane_has_no_toggle_and_no_save_path():
    page = PAGE.read_text()
    assert "ToggleSwitch" not in page
    assert "handleToggle" not in page and "handleSave" not in page
    # No handler adopts edited text as saved and closes an editor.
    assert "setEditContent" not in page and "setIsEditing" not in page
    assert "<textarea" not in page
    assert "api.skill" not in page.replace("api.skills(", "").replace("api.skillContent(", "")


def test_refresh_still_refetches_through_the_get_loader():
    page = PAGE.read_text()
    m = re.search(r"const handleRefresh = \(\) => \{(.*?)\n  \};", page, re.DOTALL)
    assert m, "handleRefresh not found"
    body = m.group(1)
    assert "loadSkills()" in body and "POST" not in body and "fetch(" not in body
    loader = re.search(r"const loadSkills = useCallback\(\(\) => \{(.*?)\n  \}, \[\]\);",
                       page, re.DOTALL)
    assert loader and "api.skills()" in loader.group(1)
    assert "onClick={handleRefresh}" in page
    # api.skills() is a GET: no method option on its fetch.
    api = API_TS.read_text()
    skills_fn = re.search(r"skills\(\): Promise<SkillsData> \{(.*?)\n  \},", api, re.DOTALL)
    assert skills_fn and "method:" not in skills_fn.group(1)
