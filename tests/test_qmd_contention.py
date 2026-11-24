"""#1495: the log join that prices contention on qmd's own card (eval/qmd_contention.py).

The card number is not hardcoded in here. qmd's own supervisord program is what
decides it, so `test_the_docstring_names_the_card_the_daemon_conf_pins` reads that
conf and turns this suite red the day the module's prose and the pin disagree (#4474).
"""
import configparser
import datetime as dt
import re
from pathlib import Path

import pytest

from eval import qmd_contention
from eval.qmd_contention import analyse, parse_daemon, parse_pin_windows
from scripts.automod.evalpin import (QMD_PROGRAM_CONF, QMD_PROGRAM_SECTION,
                                     _program_environment)

# A card mention in prose: "GPU" then a space or a hyphen then a card index. Spelled
# as a character class and with no worked example, so this pattern's own bytes never
# match the files it scans — the suite stays free of the card labels it is checking for.
_CARD_MENTION = re.compile(r"GPU[- ](\d+)")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
# Wording that would place a service co-resident with qmd on qmd's own card.
_CO_RESIDENT = re.compile(r"(same|this|that)\s+card|shares? (?:the |qmd)?\s*card|on the card",
                          re.I)
# The card qmd's daemon left on 2026-11-10, when a372c876 moved the stack. Held as a
# bare digit so this file stays free of the label it forbids.
LEFT_CARD = "0"


def pinned_card(conf: Path = QMD_PROGRAM_CONF):
    """CUDA_VISIBLE_DEVICES as qmd's daemon program sets it.

    ``None`` means there is no pin to agree with in this checkout -- the conf is
    absent or carries no ``[program:agent-qmd-daemon]`` -- which is a skip. A conf
    that has the section but no ``CUDA_VISIBLE_DEVICES`` comes back as ``""``, so it
    fails rather than quietly skipping: that daemon is on no card the prose can name.
    """
    if not conf.exists():
        return None
    cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=(" ;",),
                                   strict=False)
    cp.read(conf, encoding="utf-8")
    if not cp.has_section(QMD_PROGRAM_SECTION):
        return None
    env = _program_environment(cp.get(QMD_PROGRAM_SECTION, "environment", fallback=""))
    return env.get("CUDA_VISIBLE_DEVICES", "")


def _sentences(text: str) -> list[str]:
    return [s.replace("\n", " ").strip() for s in _SENTENCE_BREAK.split(text)]

DAEMON = """QMD MCP server listening on http://localhost:8181/mcp
23:59:58.000 POST /query 2 queries (50ms) [fts=2 embed=7 vec=30 chunk=2]
00:00:30.000 POST /query 2 queries (300ms) [fts=2 embed=160 vec=30 chunk=2]
00:00:31.000 POST /query 1 queries (5ms) [fts=2 vec=0 chunk=2]
00:05:00.000 POST /query 2 queries (900ms) [fts=2 embed=9 vec=40 chunk=2 rerank=800]
"""
REG = """2026-09-25 17:00:10,000 [INFO] lloyd-workers.automod-regression: measuring abcdef12 against 12345678 (1 pending)
2026-09-25 17:01:00,000 [INFO] lloyd-workers.automod-regression: abcdef12: success — no regression
"""


def test_the_clock_wrap_is_the_next_day_and_phases_parse():
    rows = parse_daemon(DAEMON, dt.date(2026, 9, 25))
    assert [r["ts"].day for r in rows] == [25, 26, 26, 26]
    assert rows[1]["embed"] == 160 and "rerank" in rows[3]


def test_pin_windows_are_local_time_and_classify_requests():
    rows = parse_daemon(DAEMON, dt.date(2026, 9, 25))
    wins = parse_pin_windows(REG)
    assert len(wins) == 1 and wins[0][0].utcoffset() == dt.timedelta(hours=-7)
    out = analyse(rows, wins)
    assert out["recall_doc_leg__pin"]["total_ms"]["n"] == 1      # 00:00:30Z = 17:00:30 PDT
    assert out["recall_doc_leg__no_pin"]["total_ms"]["n"] == 1
    assert out["rerank__no_pin"]["rerank_ms"]["p50"] == 800


def test_the_docstring_names_the_card_the_daemon_conf_pins():
    """#4474: the module's prose and the daemon's own pin cannot disagree.

    This is the rail a card move spends. Nothing here knows which card qmd runs on:
    the number comes from the conf, so moving the daemon off it is what turns this red.
    """
    card = pinned_card()
    if card is None:
        pytest.skip(f"{QMD_PROGRAM_CONF} is absent here, so nothing pins a card to agree with")
    assert card.isdigit(), (f"{QMD_PROGRAM_CONF.name} sets CUDA_VISIBLE_DEVICES={card!r}, "
                            "which is not a card index this docstring could name")
    named = set(_CARD_MENTION.findall(qmd_contention.__doc__ or ""))
    assert named == {card}, (
        f"eval/qmd_contention.py names card(s) {sorted(named)}, but "
        f"{QMD_PROGRAM_CONF.name} pins {card}")


def test_no_file_of_the_join_teaches_the_card_qmd_left():
    """#4474: the retired pairing must not survive in either file of the join.

    This file's own bytes are in scope: its docstring taught the retired pairing once.
    Runs even with the conf absent -- what is asserted is about the prose, not the card.
    """
    if pinned_card() == LEFT_CARD:
        pytest.skip("qmd is back on the card this test retires")
    for f in (Path(qmd_contention.__file__), Path(__file__)):
        named = set(_CARD_MENTION.findall(f.read_text()))
        assert LEFT_CARD not in named, (
            f"{f.name} still names card {LEFT_CARD}, which qmd's daemon left on 2026-11-10")


def test_the_docstring_names_only_the_eval_pin_as_qmd_s_neighbour():
    """#4474: no sentence may put TTS on qmd's card; the pin is the only other tenant.

    The join only prices what leaves a `/query` log, so the caveat that says so has to
    name the right neighbours: since a372c876 the TTS engine pins a different card and
    the vendored TTS server that lived on qmd's is retired.
    """
    doc = qmd_contention.__doc__ or ""
    for s in _sentences(doc):
        if "tts" in s.lower():
            assert not _CO_RESIDENT.search(s), f"TTS is still placed on qmd's card: {s!r}"
    tenants = [s for s in _sentences(doc) if re.search(r"only\b.*\btenant", s, re.I)]
    assert tenants, "the docstring must say which service shares qmd's card"
    assert all("pin" in s.lower() for s in tenants), (
        f"a tenant sentence names no eval pin: {tenants}")
