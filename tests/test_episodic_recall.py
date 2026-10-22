"""#1485: chat transcripts and autonomy runs as recall floors; session_recall on qmd.

Both ship OFF (`RECALL_EPISODIC_FLOORS`, `SESSION_RECALL_BACKEND`); these tests
pin that the off state is today's request byte for byte, what the on state asks
for, and that #1511's echo rule applies on both paths. Hermetic: `_qmd_post` and
`_qmd_daemon_search` are stubbed.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_mcp import session as session_mod  # noqa: E402
from agent_mcp import transcript_self_hit as tsh  # noqa: E402
from agent_mcp import vault as vault_mod  # noqa: E402


@pytest.fixture(autouse=True)
def _no_note_reads(monkeypatch):
    monkeypatch.setattr(tsh, "_sessions_root", lambda: None)


def test_off_is_todays_shape(monkeypatch):
    monkeypatch.setattr(vault_mod, "RECALL_EPISODIC_FLOORS", False)
    for ranker in ("djev", "qmd"):
        shape = vault_mod.recall_doc_leg_shape(ranker)
        assert "extra" not in shape
        assert "sessions" not in shape["floor"] and "autonomy-runs" not in shape["floor"]
    assert vault_mod.recall_doc_leg_shape("djev")["candidateLimit"] == vault_mod.RECALL_DJEV_HEAD


def test_on_names_both_collections_and_still_fits_one_canvas(monkeypatch):
    monkeypatch.setattr(vault_mod, "RECALL_EPISODIC_FLOORS", True)
    shape = vault_mod.recall_doc_leg_shape("djev")
    assert shape["extra"] == ["sessions"]
    assert shape["floor"]["sessions"] == 1
    # head + every floor x two search legs must fit the rows djev ranks in one
    # read. The ceiling is the derived one (`recall_djev_pool()` reads the canvas
    # width `app/djev.py` declares — 128 here, nothing patched), so this checks
    # the arithmetic against the value the recall actually uses rather than a
    # literal that could drift from it (#2112).
    assert shape["candidateLimit"] + 2 * sum(shape["floor"].values()) <= vault_mod.recall_djev_pool()
    assert vault_mod.recall_doc_leg_shape("qmd")["extra"] == ["sessions"]


def _capture_post(monkeypatch, rows):
    sent = {}

    def _post(payload):
        sent.update(payload)
        return rows
    monkeypatch.setattr(vault_mod, "_qmd_post", _post)
    return sent


def test_extra_collections_join_the_request_and_survive_the_fold(monkeypatch):
    rows = [{"file": "qmd://sessions/2026-09-20/20260920_101010_ivaaaa.md", "score": 0.9},
            {"file": "qmd://autonomy-runs/39/run_1.md", "score": 0.8},
            {"file": "qmd://knowledge/x.md", "score": 0.7},
            {"file": "qmd://agents/y.md", "score": 0.6}]
    sent = _capture_post(monkeypatch, rows)
    out = vault_mod._qmd_daemon_search("q words here", 32, vault_mod.VAULT_SEGMENTS,
                                       extra_collections=["sessions", "autonomy-runs"],
                                       floor={"sessions": 1, "autonomy-runs": 1})
    assert sent["collections"][-2:] == ["sessions", "autonomy-runs"]
    assert sent["collectionFloor"] == {"sessions": 1, "autonomy-runs": 1}
    files = [r["file"] for r in out]
    assert "qmd://sessions/2026-09-20/20260920_101010_ivaaaa.md" in files
    assert "qmd://autonomy-runs/39/run_1.md" in files
    assert "qmd://agents/y.md" not in files


def test_without_extras_the_fold_still_drops_them(monkeypatch):
    rows = [{"file": "qmd://sessions/2026-09-20/a.md", "score": 0.9},
            {"file": "qmd://knowledge/x.md", "score": 0.7}]
    sent = _capture_post(monkeypatch, rows)
    out = vault_mod._qmd_daemon_search("q words here", 32, vault_mod.VAULT_SEGMENTS)
    assert "sessions" not in sent["collections"]
    assert [r["file"] for r in out] == ["qmd://knowledge/x.md"]


def test_extras_are_ignored_on_a_restricted_search(monkeypatch):
    sent = _capture_post(monkeypatch, [])
    vault_mod._qmd_daemon_search("q words here", 5, ["knowledge"], extra_collections=["sessions"])
    assert sent["collections"] == ["knowledge"]


PROMPT = "please walk me through how the qmd daemon picks its embedding model again"


def test_recall_drops_an_echoing_transcript_and_the_callers_own(monkeypatch):
    from agent_mcp import _task_registry
    docs = [{"path": "sessions/2026-09-20/20260920_101010_ivaaaa.md", "snippet": f"user: {PROMPT}"},
            {"path": "sessions/2026-09-21/20260921_101010_ivbbbb.md", "snippet": "unrelated chat"},
            {"path": "sessions/2026-09-22/20260922_101010_ivcccc.md", "snippet": "user: other"},
            {"path": "knowledge/qmd.md", "snippet": f"user: {PROMPT}"}]
    tok = _task_registry.current_session_id.set("20260922_101010_ivcccc")
    try:
        kept = vault_mod._drop_recall_self_hits(PROMPT, docs)
    finally:
        _task_registry.current_session_id.reset(tok)
    assert [d["path"] for d in kept] == ["sessions/2026-09-21/20260921_101010_ivbbbb.md",
                                         "knowledge/qmd.md"]


def test_recall_keeps_only_chat_transcripts(monkeypatch):
    docs = [{"path": "sessions/2026-09-20/20260920_101010_ivaaaa.md", "snippet": "a chat"},
            {"path": "sessions/2026-09-20/20260920_101010_youtubed_1a2b.md", "snippet": "a digest"},
            {"path": "sessions/2026-09-07/e2e_selfmod294_1788815463.md", "snippet": "fixture"},
            {"path": "knowledge/qmd.md", "snippet": "doc"}]
    kept = [d["path"] for d in vault_mod._drop_recall_self_hits("unrelated words here", docs)]
    assert kept == ["sessions/2026-09-20/20260920_101010_ivaaaa.md", "knowledge/qmd.md"]
    monkeypatch.setattr(vault_mod, "RECALL_EPISODIC_CHAT_ONLY", False)
    assert len(vault_mod._drop_recall_self_hits("unrelated words here", docs)) == 4


def test_session_recall_qmd_skips_background_exports(monkeypatch):
    recent = _day(1)
    rows = [{"file": f"qmd://sessions/x/{recent}_101010_autocode_9f2a.md", "score": 0.9, "snippet": "bg"},
            {"file": f"qmd://sessions/x/{recent}_101010_ivaaaa.md", "score": 0.8, "snippet": "chat"}]
    monkeypatch.setattr(vault_mod, "_qmd_daemon_search", lambda *a, **k: rows)
    monkeypatch.setattr(session_mod, "SESSION_RECALL_BACKEND", "qmd")
    out = session_mod._session_recall({"query": "what did we say", "days": 7, "limit": 5})
    assert [s["session_id"] for s in out["sessions"]] == [f"{recent}_101010_ivaaaa"]


def test_recall_filter_is_a_noop_without_transcripts():
    docs = [{"path": "knowledge/qmd.md", "snippet": "x"}]
    assert vault_mod._drop_recall_self_hits(PROMPT, docs) is docs


# ── session_recall on qmd ───────────────────────────────────────────────────

def _day(offset: int) -> str:
    return (datetime.datetime.now() - datetime.timedelta(days=offset)).strftime("%Y%m%d")


def test_session_recall_qmd_keeps_the_shape_window_and_echo_rule(monkeypatch):
    recent, old = _day(1), _day(40)
    rows = [
        {"file": f"qmd://sessions/x/{recent}_101010_ivaaaa.md", "score": 0.9,
         "snippet": "we changed the embedding model to qwen3 in index.yml"},
        {"file": f"qmd://sessions/x/{old}_101010_ivbbbb.md", "score": 0.8, "snippet": "old"},
        {"file": f"qmd://sessions/x/{recent}_111111_ivcccc.md", "score": 0.7,
         "snippet": f"user: {PROMPT}"},
        {"file": "qmd://knowledge/k.md", "score": 0.6, "snippet": "not a session"},
    ]
    seen = {}

    def _search(query, limit, collections, **kw):
        seen.update(collections=collections, limit=limit, **kw)
        return rows
    monkeypatch.setattr(vault_mod, "_qmd_daemon_search", _search)
    monkeypatch.setattr(session_mod, "SESSION_RECALL_BACKEND", "qmd")
    out = session_mod._session_recall({"query": PROMPT, "days": 7, "limit": 5})
    assert seen["collections"] == ["sessions"] and seen["limit"] == 5 * session_mod._SESSION_QMD_OVERFETCH
    assert out["backend"] == "qmd"
    assert [s["session_id"] for s in out["sessions"]] == [f"{recent}_101010_ivaaaa"]
    s = out["sessions"][0]
    assert set(s) >= {"session_id", "created_at", "preview", "snippets", "match_score"}
    assert s["created_at"].startswith(f"{recent[:4]}-{recent[4:6]}-{recent[6:]}T10:10:10")


def test_session_recall_qmd_falls_back_to_the_scorer_on_an_outage(monkeypatch):
    def _down(*a, **k):
        raise vault_mod.QmdUnavailable("down")
    monkeypatch.setattr(vault_mod, "_qmd_daemon_search", _down)
    monkeypatch.setattr(session_mod, "SESSION_RECALL_BACKEND", "qmd")
    monkeypatch.setattr(session_mod, "_load_session_index", lambda max_days=14: {})
    out = session_mod._session_recall({"query": "anything at all", "days": 7})
    assert "backend" not in out and out["sessions"] == []


def _config(monkeypatch, retrieval: dict):
    from app import config as cfg
    monkeypatch.setattr(cfg, "CONFIG", {**(cfg.CONFIG or {}), "retrieval": retrieval})


def test_session_recall_backend_reads_config(monkeypatch):
    assert session_mod.SESSION_RECALL_BACKEND is None      # no override ships
    _config(monkeypatch, {})
    assert session_mod.session_recall_backend() == "tokens"
    _config(monkeypatch, {"session_recall": {"backend": "qmd"}})
    assert session_mod.session_recall_backend() == "qmd"
    _config(monkeypatch, {"session_recall": {"backend": "bogus"}})
    assert session_mod.session_recall_backend() == "tokens"
    monkeypatch.setattr(session_mod, "SESSION_RECALL_BACKEND", "tokens")
    _config(monkeypatch, {"session_recall": {"backend": "qmd"}})
    assert session_mod.session_recall_backend() == "tokens"


def test_episodic_floors_read_config_and_the_kill_env(monkeypatch):
    assert vault_mod.RECALL_EPISODIC_FLOORS is None        # no override ships
    monkeypatch.delenv("LLOYD_RECALL_EPISODIC", raising=False)
    _config(monkeypatch, {})
    assert vault_mod.recall_episodic_floors() is False
    assert "extra" not in vault_mod.recall_doc_leg_shape("djev")
    _config(monkeypatch, {"recall": {"episodic_floors": True}})
    assert vault_mod.recall_episodic_floors() is True
    assert vault_mod.recall_doc_leg_shape("djev")["extra"] == ["sessions"]
    monkeypatch.setenv("LLOYD_RECALL_EPISODIC", "0")
    assert vault_mod.recall_episodic_floors() is False


def test_the_tracked_config_matches_the_verdict():
    """config.yaml carries both keys, so a rebuild boots into what is served."""
    import yaml
    raw = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
    r = raw["retrieval"]
    assert isinstance(r["recall"]["episodic_floors"], bool)
    assert r["session_recall"]["backend"] in ("tokens", "qmd")


# ── the regression pin sends what production sends ──────────────────────────

def test_production_payload_mirrors_the_floor(monkeypatch):
    from scripts.automod import evalpin
    monkeypatch.setattr(vault_mod, "RECALL_EPISODIC_FLOORS", False)
    off = evalpin.production_payload("what did we decide")
    assert "sessions" not in off["collections"]
    assert "sessions" not in off.get("collectionFloor", {})
    monkeypatch.setattr(vault_mod, "RECALL_EPISODIC_FLOORS", True)
    on = evalpin.production_payload("what did we decide")
    assert on["collections"] == list(vault_mod.VAULT_SEGMENTS) + ["sessions"]
    assert on["collectionFloor"]["sessions"] == 1
    monkeypatch.setattr(vault_mod, "recall_reranker", lambda: "djev")
    djev = evalpin.production_payload("what did we decide")
    assert djev["candidateLimit"] == vault_mod.RECALL_EPISODIC_DJEV_HEAD
    assert djev["collectionFloor"]["sessions"] == 1 and djev["rerank"] is False


def test_production_payload_is_the_doc_legs_request(monkeypatch):
    """Byte-level: the pin's payload equals what `_qmd_daemon_search` posts for
    the recall doc leg, flag on and off (lex/vec query text aside)."""
    from scripts.automod import evalpin
    for on in (False, True):
        monkeypatch.setattr(vault_mod, "RECALL_EPISODIC_FLOORS", on)
        sent = _capture_post(monkeypatch, [])
        shape = vault_mod.recall_doc_leg_shape()
        vault_mod._qmd_daemon_search("what did we decide", shape["limit"], vault_mod.VAULT_SEGMENTS,
                                     skip_rerank=not shape["rerank"], exact_pool=shape["limit"],
                                     candidate_limit=shape["candidateLimit"], floor=shape["floor"],
                                     lex_mode=shape.get("lexMode"),
                                     extra_collections=shape.get("extra"))
        pin = evalpin.production_payload("what did we decide")
        for k in ("limit", "candidateLimit", "collections", "rerank", "fusion",
                  "collectionFloor", "lexMode"):
            assert pin.get(k) == sent.get(k), (on, k)


def test_pin_refuses_a_snapshot_missing_a_requested_collection():
    from scripts.automod import evalpin
    prov = {"collections": {"knowledge": 10, "sessions": 0}}
    assert evalpin.missing_collections(prov, ["knowledge", "sessions"]) == ["sessions"]
    assert evalpin.missing_collections({"collections": None}, ["sessions"]) == []


def test_snapshot_collections_counts_active_documents(tmp_path):
    import sqlite3
    from scripts.automod import evalpin
    con = sqlite3.connect(tmp_path / "i.sqlite")
    con.execute("create table documents (collection text, path text, active int)")
    con.executemany("insert into documents values (?,?,?)",
                    [("sessions", "a", 1), ("sessions", "b", 0), ("knowledge", "c", 1)])
    assert evalpin.snapshot_collections(con) == {"sessions": 1, "knowledge": 1}


def test_pin_config_report(tmp_path):
    from scripts.automod import evalpin
    (tmp_path / "index.yml").write_text(
        "collections:\n  sessions: {path: /x, pattern: '**/*.md'}\nmodels:\n  embed: E\n")
    (tmp_path / "evalpin.yml").write_text("collections: {}\nmodels:\n  embed: E\n")
    r = evalpin.pin_config_report("evalpin", config_dir=tmp_path)
    assert r["embed_match"] and not r["collections_match"]
    assert r["missing_collections"] == ["sessions"]



# --- #3158: the pin's qmd config is generated at pin time, and drift is named --
#
# Before this, `~/.config/qmd/evalpin.yml` was a hand-kept copy of
# `~/.config/qmd/index.yml` whose own header asked for `cp index.yml evalpin.yml`
# after any edit to the live file. Nothing noticed the copy had not been re-synced
# since 2026-09-25 while the live file's `subliminal` pattern was narrowed on
# 2026-10-15: `pin_config_report` returned one boolean, `collections_match: false`,
# and the only refusal in the path — `missing_collections` — is computed from the
# snapshot's document counts, never from this compare. The drift is not cosmetic:
# qmd's `syncConfigToDb` (qmd/src/store.ts:1463-1470) rebuilds the PIN's
# `store_collections` from the pin's YAML on every `serve --index evalpin` whose
# parsed config hash differs, so the drifted copy re-widened `subliminal` to the
# whole vault inside the frozen snapshot — and `eval/embed_side_index.py:66` hands
# that same text to every eval arm as its base config.

#: The live pair of 2026-10-22, transcribed: the live leg of #2180's narrowing
#: (`subliminal` one level, its `ignore:` list kept), `memory` unchanged, one
#: embed model shared by both files. The two `pattern` strings below are the two
#: values the drift report has to name, so they are written out, not generated.
LIVE_YML_2026_10_22 = """\
serve:
  port: 8139
embed:
  max_context: 8192
collections:
  subliminal:
    path: /home/me/obsidian
    pattern: "*.md"
    ignore:
      - agents/**
      - templates/**
  memory:
    path: /home/me/obsidian/memory
    pattern: "**/*.md"
models:
  embed: E
  rerank: R
  generate: G
"""

#: The same file as the live one, with that narrowing undone in the PIN's copy —
#: exactly the state `~/.config/qmd/` sat in on 2026-10-22.
DRIFTED_PIN_YML_2026_10_22 = LIVE_YML_2026_10_22.replace(
    '    pattern: "*.md"\n    ignore:', '    pattern: "**/*.md"\n    ignore:')


def test_config_drift_fixture_differsexactly_where_the_live_pair_differed():
    """Guard on the two fixtures above. Every assertion below it is about the
    2026-10-22 state, so if a fixture were transcribed wrong the tests would be
    pinning a state that never existed. One collection, one field, differing in
    the direction the live file moved.
    """
    import yaml
    live = yaml.safe_load(LIVE_YML_2026_10_22)["collections"]
    pin = yaml.safe_load(DRIFTED_PIN_YML_2026_10_22)["collections"]
    assert live["subliminal"]["pattern"] == "*.md"
    assert pin["subliminal"]["pattern"] == "**/*.md"
    assert [n for n in live if live[n] != pin[n]] == ["subliminal"]


def test_sync_pin_config_regenerates_a_drifted_copy_from_index_yml(tmp_path):
    """Clause 1: the hand copy stops being a thing a person has to remember.

    #1485's convention was `cp index.yml evalpin.yml`, re-run by hand after any
    edit to `index.yml`, with the header re-added by hand — and the copy that
    convention produced is the fixture above. A byte-copy onto the file would fix
    that file; generating it from `index.yml` at pin time removes the dependency.
    The live file is the source and never the thing that changes, and the header
    survives because it is the file's only explanation of why an empty
    `collections:` block once emptied `store_collections`.
    """
    from scripts.automod import evalpin
    (tmp_path / "index.yml").write_text(LIVE_YML_2026_10_22)
    (tmp_path / "evalpin.yml").write_text(DRIFTED_PIN_YML_2026_10_22)

    report = evalpin.sync_pin_config("evalpin", config_dir=tmp_path)

    import yaml
    assert (yaml.safe_load((tmp_path / "evalpin.yml").read_text())
            == yaml.safe_load(LIVE_YML_2026_10_22)), "the two must PARSE equal"
    assert report["collections_match"] is True, report
    assert report["collections_drift"] == [], report
    assert report["missing_collections"] == [], report
    assert report["embed_match"] is True, report
    assert report["synced_from"] == str(tmp_path / "index.yml"), report
    assert "GENERATED" in (tmp_path / "evalpin.yml").read_text(), (
        "the generated file lost the comment saying which job owns it")
    assert (tmp_path / "index.yml").read_text() == LIVE_YML_2026_10_22, (
        "the live file the pin is generated FROM was edited")


def test_sync_pin_config_overwrites_a_hand_edit_to_the_pin(tmp_path):
    """The class this retires is not "someone forgot to re-sync" but "the pin's
    config is a file someone can edit". A one-field experiment aimed at the pin —
    widening a pattern to see what the corpus would return — is exactly what
    `evalpin.yml:4-6` says the byte-copy exists to make impossible, and it now
    survives one pin run at most.
    """
    from scripts.automod import evalpin
    (tmp_path / "index.yml").write_text(LIVE_YML_2026_10_22)
    (tmp_path / "evalpin.yml").write_text(
        LIVE_YML_2026_10_22.replace("    path: /home/me/obsidian/memory\n",
                                    "    path: /tmp/scratch\n"))
    assert (tmp_path / "evalpin.yml").read_text() != LIVE_YML_2026_10_22

    evalpin.sync_pin_config("evalpin", config_dir=tmp_path)

    import yaml
    got = yaml.safe_load((tmp_path / "evalpin.yml").read_text())["collections"]
    assert got["memory"]["path"] == "/home/me/obsidian/memory", got


def test_sync_pin_config_refuses_without_a_live_config_rather_than_pinning_nothing(tmp_path):
    """With no readable `index.yml` there is no idea what production indexes, and
    generating an empty collection block is #1485's shape: qmd answers a query
    scoped to a collection missing from `store_collections` with ZERO ROWS, which
    reads as "the corpus does not contain that" about a corpus that was never
    generated. So the step refuses to WRITE, leaves the file it found exactly as it
    found it, and names the reason — see
    `test_a_pin_run_records_the_config_refusal_and_still_serves` for why that is a
    recorded field and not a raised `PinError`.
    """
    from scripts.automod import evalpin
    (tmp_path / "evalpin.yml").write_text(DRIFTED_PIN_YML_2026_10_22)

    r = evalpin.sync_pin_config("evalpin", config_dir=tmp_path)

    assert "index.yml" in r["sync_refused"], r
    assert r["synced_from"] == str(tmp_path / "index.yml"), r
    assert (tmp_path / "evalpin.yml").read_text() == DRIFTED_PIN_YML_2026_10_22, (
        "a refused sync still wrote the file")


def test_sync_pin_config_refuses_a_live_config_declaring_no_collections(tmp_path):
    """The other half of the same refusal: a live file that parses but declares
    nothing — a truncation, a half-finished edit — would generate a pin that
    indexes nothing and still report `collections_match: true`, because both sides
    would be empty and therefore agree. Agreement between two empties is not the
    property anyone is asking about, so the write is refused for the same reason
    and for the same reason it is recorded rather than raised.
    """
    from scripts.automod import evalpin
    (tmp_path / "index.yml").write_text("models:\n  embed: E\n")
    (tmp_path / "evalpin.yml").write_text(DRIFTED_PIN_YML_2026_10_22)

    r = evalpin.sync_pin_config("evalpin", config_dir=tmp_path)

    assert "declares no collections" in r["sync_refused"], r
    assert (tmp_path / "evalpin.yml").read_text() == DRIFTED_PIN_YML_2026_10_22
    assert r["collections_match"] is False, (
        "the untouched drifted copy must not read as a match")


def test_a_pin_run_records_the_config_refusal_and_still_serves(tmp_path, monkeypatch):
    """The refusal is a field, not an exception, and this is why rather than a
    preference. `PinnedCorpus.__enter__` has one job — hand back a daemon serving a
    frozen snapshot — and three of this module's existing tests drive it to a PORT
    CONFLICT or a DEAD DAEMON with no qmd config dir in reach. A `PinError` raised
    here would arrive at them as a different failure than the one they pin, and a
    guard that rewrites what unrelated failures look like is worse than the drift it
    guards. So the pin proceeds, and the provenance says in as many words that its
    config could not be generated.
    """
    import sys
    from scripts.automod import evalpin

    cli = tmp_path / "qmd.js"
    cli.write_text("// not qmd")   # no index.yml anywhere beside it
    entered: list = []

    monkeypatch.setattr(evalpin, "production_daemon", lambda conf=None: ([str(cli)], {}, "test"))
    monkeypatch.setattr(evalpin, "reap_stale", lambda port, name: [])
    monkeypatch.setattr(evalpin, "port_free", lambda port: True)
    monkeypatch.setattr(evalpin, "_probe", lambda port, timeout=5.0: True)
    monkeypatch.setattr(evalpin, "pin_command",
                        lambda argv, port, name: entered.append(name) or [
                            sys.executable, "-c", "import time; time.sleep(30)", str(cli)])
    monkeypatch.setattr(evalpin, "snapshot",
                        lambda name: {"index": str(tmp_path / "pin.sqlite"),
                                      "collections": {"subliminal": 5}})
    monkeypatch.setattr(evalpin, "production_payload", lambda q: {"collections": []})
    monkeypatch.setenv("QMD_CONFIG_DIR", str(tmp_path))

    with evalpin.PinnedCorpus(workdir=tmp_path / "work") as pin:
        pass

    assert entered == ["evalpin"], "the daemon was never spawned"
    assert "sync_refused" in pin.provenance["pin_config"], pin.provenance["pin_config"]
    assert not (tmp_path / "evalpin.yml").exists(), "a refused sync created a config"


def test_taking_a_pin_generates_the_config_before_the_daemon_is_spawned(tmp_path, monkeypatch):
    """Clause 1 across the process boundary, which is where the ordering matters.

    `PinnedCorpus.__enter__` snapshots, then spawns `qmd serve --index evalpin`
    against that snapshot — and the spawned daemon is what reads the pin's YAML and
    rewrites `store_collections` from it. Generate after the spawn and the daemon
    has already served the drifted corpus this round measures against, so the
    assertion is taken inside `pin_command`, at the last point before the `Popen`
    whose child reads the file.
    """
    import sys
    import yaml
    from scripts.automod import evalpin

    (tmp_path / "index.yml").write_text(LIVE_YML_2026_10_22)
    (tmp_path / "evalpin.yml").write_text(DRIFTED_PIN_YML_2026_10_22)
    cli = tmp_path / "qmd.js"
    cli.write_text("// not qmd")
    seen: list = []

    def fake_pin_command(argv, port, name):
        seen.append(yaml.safe_load((tmp_path / f"{name}.yml").read_text())["collections"])
        return [sys.executable, "-c", "import time; time.sleep(30)", str(cli)]

    monkeypatch.setattr(evalpin, "production_daemon", lambda conf=None: ([str(cli)], {}, "test"))
    monkeypatch.setattr(evalpin, "reap_stale", lambda port, name: [])
    monkeypatch.setattr(evalpin, "port_free", lambda port: True)
    monkeypatch.setattr(evalpin, "_probe", lambda port, timeout=5.0: True)
    monkeypatch.setattr(evalpin, "pin_command", fake_pin_command)
    monkeypatch.setattr(evalpin, "snapshot",
                        lambda name: {"index": str(tmp_path / "pin.sqlite"),
                                      "collections": {"subliminal": 5, "memory": 2}})
    # Production's recall names the fixture's two collections, so the #1485
    # refusal below is not what stops the run before the spawn is reached.
    monkeypatch.setattr(evalpin, "production_payload",
                        lambda q: {"collections": ["subliminal", "memory"]})
    monkeypatch.setenv("QMD_CONFIG_DIR", str(tmp_path))

    with evalpin.PinnedCorpus(workdir=tmp_path / "work") as pin:
        pass

    assert len(seen) == 1, "the daemon was never reached"
    assert seen[0]["subliminal"]["pattern"] == "*.md", (
        "the spawned daemon read the drifted copy: it rewrote the snapshot's "
        "store_collections to the vault-wide glob before this run's first query")
    assert pin.provenance["pin_config"]["collections_match"] is True, (
        "provenance must record the config the daemon was actually served")
    assert pin.provenance["pin_config"]["synced_from"] == str(tmp_path / "index.yml")


def test_pin_config_report_names_the_drifted_field_and_both_values(tmp_path):
    """Clause 2: a bare boolean recorded into provenance nobody reads.

    `collections_match: false` could not tell a collection the pin never declared
    — the #1485 shape, which empties `store_collections` — from one collection
    whose `pattern` was narrowed on the live side and not in the copy. The second
    is a corpus difference the pin measures silently, so it is named
    collection-and-field with both values, the shape
    `scripts/maintenance/qmd_index_maintenance.py` has used for a pattern drift
    since #2543.
    """
    from scripts.automod import evalpin
    (tmp_path / "index.yml").write_text(LIVE_YML_2026_10_22)
    (tmp_path / "evalpin.yml").write_text(DRIFTED_PIN_YML_2026_10_22)

    r = evalpin.pin_config_report("evalpin", config_dir=tmp_path)

    assert r["collections_match"] is False
    assert r["missing_collections"] == [], "nothing was dropped here"
    assert r["collections_drift"] == [{
        "collection": "subliminal", "field": "pattern",
        "pin": "**/*.md", "live": "*.md",
    }], r["collections_drift"]
    assert r["embed_match"] is True
    for row in r["collections_drift"]:
        assert set(row) == {"collection", "field", "pin", "live"}, row


def test_pin_config_report_separates_a_dropped_collection_from_a_changed_one(tmp_path):
    """Both kinds at once, because the distinction IS the severity: the dropped
    collection is the one `PinnedCorpus.__enter__` refuses the run over, the
    changed one is the one nothing refused for 27 days. One list would let a reader
    apply the wrong remedy to either, so `missing_collections` keeps its meaning
    unchanged and the changed field goes in its own rows.
    """
    from scripts.automod import evalpin
    (tmp_path / "index.yml").write_text(LIVE_YML_2026_10_22)
    (tmp_path / "evalpin.yml").write_text(
        LIVE_YML_2026_10_22
        .replace('    pattern: "*.md"\n    ignore:',
                 '    pattern: "**/*.md"\n    ignore:')
        .replace("  memory:\n    path: /home/me/obsidian/memory\n"
                 '    pattern: "**/*.md"\n', ""))

    r = evalpin.pin_config_report("evalpin", config_dir=tmp_path)

    assert r["missing_collections"] == ["memory"], r
    assert [(d["collection"], d["field"]) for d in r["collections_drift"]] == [
        ("subliminal", "pattern")], r
    assert r["collections_drift"][0]["pin"] == "**/*.md", r


def test_pin_config_report_names_a_path_change_and_an_ignore_change(tmp_path):
    """`pattern` is the field that drifted on this box, but it is not the only one
    a copy can fall behind on: an `ignore:` list that differs changes which
    documents are in the corpus as surely as a `pattern` does, and a `path` that
    moved changes which tree is read at all. One row per differing field, so a
    collection that moved AND narrowed yields two rows whose values are each true.
    """
    from scripts.automod import evalpin
    (tmp_path / "index.yml").write_text(LIVE_YML_2026_10_22)
    (tmp_path / "evalpin.yml").write_text(
        LIVE_YML_2026_10_22
        .replace("    path: /home/me/obsidian\n", "    path: /home/me/vault\n")
        .replace("      - agents/**\n", ""))

    got = evalpin.pin_config_report("evalpin", config_dir=tmp_path)["collections_drift"]

    assert {(d["collection"], d["field"]) for d in got} == {
        ("subliminal", "path"), ("subliminal", "ignore")}, got
    ignore = next(d for d in got if d["field"] == "ignore")
    assert ignore["pin"] == ["templates/**"], ignore
    assert ignore["live"] == ["agents/**", "templates/**"], ignore


def test_pin_config_report_of_a_pin_declaring_nothing_names_every_live_collection(tmp_path):
    """The #1485 shape — `collections: {}` — is named collection by collection now
    instead of as one boolean, and `missing_collections` still lists exactly the
    set `PinnedCorpus.__enter__` refuses on. No field rows: nothing is declared on
    both sides, so there is no field to compare.
    """
    from scripts.automod import evalpin
    (tmp_path / "index.yml").write_text(LIVE_YML_2026_10_22)
    (tmp_path / "evalpin.yml").write_text("collections: {}\nmodels:\n  embed: E\n")

    r = evalpin.pin_config_report("evalpin", config_dir=tmp_path)

    assert r["missing_collections"] == ["memory", "subliminal"], r
    assert r["collections_drift"] == [], r


def test_pin_config_report_survives_an_unreadable_file_the_way_it_always_did(tmp_path):
    """The report has never raised — a pin with no config file gets an `error` key
    and keeps its provenance — and the new compare must not be the thing that
    changes that. `collections_drift` is absent when nothing could be read, so a
    reader never sees an empty list and concludes the two configs agree.
    """
    from scripts.automod import evalpin
    r = evalpin.pin_config_report("nope", config_dir=tmp_path)
    assert "error" in r, r
    assert "collections_drift" not in r and "missing_collections" not in r, r
