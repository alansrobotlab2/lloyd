"""The report-only prefetch conflict counter (#3801 step 4): per-turn verdict and
the distribution, both built from injected turn fixtures.

Nothing here touches the engine, the KG store or the assembler: `resolve` is
always injected (a dict's `.get`), the session tree is a `tmp_path` fixture, and
one test asserts outright that importing the counter neither reaches
`app.prefetch` nor opens a transport.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "prefetch_conflict_count", ROOT / "eval" / "prefetch_conflict_count.py")
P = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = P
_spec.loader.exec_module(P)


def block(*lines: str) -> str:
    return "<context>\n" + "\n".join(lines) + "\n</context>"


def facts(*lines: str) -> str:
    return "<facts>\n" + "\n".join(lines) + "\n</facts>"


def vault(*lines: str) -> str:
    return "<vault-context>\n" + "\n".join(lines) + "\n</vault-context>"


# ── the per-turn verdict, on fixtures ───────────────────────────────────────


def test_the_fixture_turn_is_a_conflict_because_one_entity_arrives_twice():
    """The clause's own shape: two items resolved to one entity carrying
    different values. The fixture's `djev` pair is the control inside it — two
    items, one value, no conflict."""
    rep = P.fixture_report()
    assert rep["has_prefetch"] is True
    assert rep["n_items"] == 7 and rep["n_entities"] == 2
    assert rep["conflict"] is True
    assert rep["conflicting_entities"] == ["Voice Pipeline"]
    assert rep["conflict_details"]["Voice Pipeline"] == {
        "items": 2, "distinct_values": 2,
        "sources": ["Voice Pipeline", "Voice Pipeline"]}
    assert rep["max_distinct_values_one_entity"] == 2
    # djev is the control inside the fixture: two items, byte-identical value
    # after folding (only the confidence suffix differs), so no conflict.
    assert rep["max_items_for_one_entity"] == 2
    assert set(rep["conflict_details"]) == {"Voice Pipeline"}


def test_the_kg_resolving_two_notes_to_one_entity_is_itself_a_conflict():
    """The class the per-file cap cannot bound: `app/prefetch.py` dedupes vault
    hits by file, so two different paths can and do land together. When the KG
    says those two paths are one entity saying two things, that IS the
    look-alike stimulus — and it is invisible to `dup_detect`'s predicate."""
    bare = P.fixture_report()
    assert bare["n_unresolved"] == 3 and bare["n_resolved"] == 4
    kg = P.fixture_report(resolve=P.FIXTURE_KG.get)
    assert kg["n_unresolved"] == 1 and kg["n_resolved"] == 6
    assert kg["conflicting_entities"] == ["Voice Pipeline", "Wake-Word & Audio"]
    assert kg["conflict_details"]["Wake-Word & Audio"]["sources"] == [
        "knowledge/ops/wake-word-audio.md", "memory/2026/wake-word-retrain.md"]


def test_one_entity_stated_twice_is_a_conflict_but_the_same_statement_twice_is_not():
    entity = "Relay Inventory"
    same = block(facts(f"- [{entity}] the relay listens on port 7254 (confidence: 0.9)",
                       f"- [{entity}] The relay listens on   PORT 7254 (confidence: 0.7)"))
    rep = P.turn_report(same)
    assert rep["conflict"] is False and rep["max_items_for_one_entity"] == 2
    assert rep["max_distinct_values_one_entity"] == 1

    diff = block(facts(f"- [{entity}] the relay listens on port 7254 (confidence: 0.9)",
                       f"- [{entity}] the relay listens on port 7425 since the migration "
                       f"(confidence: 0.6)"))
    rep2 = P.turn_report(diff)
    assert rep2["conflict"] is True and rep2["max_distinct_values_one_entity"] == 2


def test_two_entities_never_make_a_conflict_however_similar_their_text():
    """The predicate is same-entity, not similar-text: two sibling entities
    describing the same shape in different words are standing problem #1, but
    counting them here would answer a question this counter does not ask."""
    rep = P.turn_report(block(facts(
        "- [billing-west relay] listens on port 7245 (confidence: 0.9)",
        "- [search-east relay] listens on port 7524 (confidence: 0.9)")))
    assert rep["n_entities"] == 2 and rep["conflict"] is False
    assert rep["max_distinct_values_one_entity"] == 1


def test_an_unresolved_vault_hit_is_counted_and_never_guessed_into_a_conflict():
    """A file stem is not an entity assertion, so with no KG answer the item is
    `entity=None`: reported in `n_unresolved`, and it cannot push a turn over the
    line either way."""
    rep = P.turn_report(block(vault(
        "- **A note** (score: 1.00, file: knowledge/a.md, line: 3): the port is 7254",
        "- **Another note** (score: 0.90, file: knowledge/b.md, line: 8): the port is 7425")))
    assert rep["n_items"] == 2 and rep["n_resolved"] == 0 and rep["n_unresolved"] == 2
    assert rep["conflict"] is False and rep["n_entities"] == 0


# ── the boundary with the renderer ─────────────────────────────────────────


def test_the_two_rendered_line_formats_both_parse_including_their_variants():
    """The counter reads what `app/prefetch.py` writes, across a module boundary
    with no shared type — so the exact rendered shapes are pinned here: the
    `- [Entity] text (confidence: x)` fact line (app/prefetch.py:858) and the
    `- **Title** (meta): excerpt` vault line (app/prefetch.py:1353), including the
    carried-semantic-hit variant of the meta and a vault hit with no `file:` key."""
    items = P.parse_items(block(
        facts("- [Voice Pipeline] Barge-in is ON since 2026-09-24 (confidence: 0.9)",
              "- [djev] No confidence suffix at all"),
        vault("- **X** (score: 1.00, file: backlog/12-x.md, line: 52): excerpt one",
              "- **Y** (score: 0.50, semantic hit from the previous turn's query): excerpt two",
              "- **Z** (score: 0.40): excerpt three")))
    assert [i.section for i in items] == ["facts", "facts", "vault-context",
                                          "vault-context", "vault-context"]
    assert items[0].entity == "Voice Pipeline"
    assert items[0].value == "barge-in is on since 2026-09-24"     # confidence stripped
    assert items[1].value == "no confidence suffix at all"
    assert items[2].source == "backlog/12-x.md"
    assert items[3].source == "Y"              # no file: key -> the title is the source
    assert items[4].value == "excerpt three"
    assert P.parse_items(block(facts("- not a fact line"), vault("plain prose"))) == []
    assert P.turn_report(block("<facts>\n</facts>"))["has_prefetch"] is False


# ── the distribution over injected turn fixtures ────────────────────────────


def _session(tmp_path: Path, name: str, blocks: list[str]) -> Path:
    f = tmp_path / name
    f.write_text(json.dumps({"id": name, "messages": [
        {"role": "user", "content": b} for b in blocks] +
        [{"role": "assistant", "content": "answer"}]}), encoding="utf-8")
    return f


def test_the_distribution_is_built_from_turn_fixtures_not_from_the_engine(tmp_path):
    """Two conflicting turns and one clean turn in a scratch session tree, plus
    two turns that carry no retrieved context at all — which must be excluded
    from the denominator rather than counted as evidence that conflicts are rare."""
    _session(tmp_path, "a.json", [P.FIXTURE,
                                 block(facts("- [x] one value (confidence: 0.9)")),
                                 "<context>no prefetch here</context>"])
    _session(tmp_path, "b.json", [P.FIXTURE])
    (tmp_path / "unreadable.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "note.md").write_text("ignore me", encoding="utf-8")

    rep = P.collect(tmp_path, days=1, resolve=P.FIXTURE_KG.get)
    assert rep["sessions_read"] == 3 and rep["n_turns"] == 3      # the 4th turn has no prefetch
    assert rep["n_turns_with_conflict"] == 2
    assert rep["share_with_conflict"] == 0.6667
    assert rep["items_by_section"] == {"facts": 9, "vault-context": 6}
    assert rep["values_histogram"] == {"1": 1, "2": 2}
    assert rep["max_values_seen"] == 2
    assert rep["items_histogram"] == {"1": 1, "2": 2}
    assert len(rep["conflicting_turns"]) == 2
    assert {t["max_distinct_values_one_entity"] for t in rep["conflicting_turns"]} == {2}

    no_kg = P.collect(tmp_path, days=1, resolve=lambda n: None)
    assert no_kg["n_turns_with_conflict"] == 2          # the facts conflict either way
    assert all(t["entities"] == ["Voice Pipeline"] for t in no_kg["conflicting_turns"])
    assert no_kg["max_values_seen"] == 2

    assert P.collect(tmp_path, days=0, resolve=lambda n: None)["n_turns"] == 0


def test_loading_the_counter_reaches_neither_the_engine_nor_the_assembler():
    """`report-only` is a claim about imports, so it is pinned as one: a fresh
    load of the module pulls in no assembler, no transport and no KG store. The
    counter only takes the store by request, inside `kg_resolver()`."""
    spec = importlib.util.spec_from_file_location("pcc_isolated",
                                                  ROOT / "eval" / "prefetch_conflict_count.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pcc_isolated"] = mod
    before = set(sys.modules)
    spec.loader.exec_module(mod)
    fresh = set(sys.modules) - before
    assert not [m for m in fresh if m.startswith(("app.prefetch", "app.harness",
                                                  "app.kg_store"))]
    src = (ROOT / "eval" / "prefetch_conflict_count.py").read_text()
    assert "app/prefetch.py" in src                   # the assembler is named in prose only…
    assert "from app.prefetch" not in src and "import app.prefetch" not in src
    assert "write_text" not in src and "open(" not in src   # …and nothing is ever written


def test_a_missing_or_broken_kg_store_leaves_items_unresolved_rather_than_raising(monkeypatch):
    """`kg_store.store()` raises when nothing is configured, and a report-only
    counter must never be the thing that opens a database. A fact line still
    asserts its own entity, so facts keep resolving; vault hits fall to
    unresolved and the report says so instead of guessing."""
    from app import kg_store

    def boom():
        raise RuntimeError("kg.sqlite is not configured")

    monkeypatch.setattr(kg_store, "store", boom)
    resolve = P.kg_resolver()
    assert resolve("wake-word-retrain") is None
    rep = P.turn_report(P.FIXTURE, resolve=resolve)
    assert rep["n_resolved"] == 4 and rep["n_unresolved"] == 3
    assert rep["conflicting_entities"] == ["Voice Pipeline"]


def test_the_store_is_asked_first_for_an_alias_then_for_an_entity_name(monkeypatch):
    """The lookup order is the resolver's whole behaviour: an alias surface is
    canonicalised, an unknown surface is not invented."""
    class Aliases:
        def resolve(self, name):
            return "Voice Pipeline" if name == "voice pipeline" else None

    class Entities:
        def lookup(self, name):
            return "djev" if name == "djev" else None

    class Fake:
        aliases = Aliases()
        entities = Entities()

    from app import kg_store
    monkeypatch.setattr(kg_store, "store", lambda: Fake())
    resolve = P.kg_resolver()
    assert resolve("voice pipeline") == "Voice Pipeline"
    assert resolve("djev") == "djev"
    assert resolve("no such thing") is None


def test_the_cli_reports_the_fixture_without_touching_the_session_tree(tmp_path, monkeypatch,
                                                                      capsys):
    monkeypatch.chdir(tmp_path)
    assert P.main(["--fixture-check"]) == 0
    assert json.loads(capsys.readouterr().out)["conflict"] is True
    assert list(tmp_path.iterdir()) == []

    _session(tmp_path, "a.json", [P.FIXTURE])
    assert P.main(["--sessions-dir", str(tmp_path), "--no-kg", "--days", "1"]) == 0
    out = capsys.readouterr().out
    assert "turns with injected facts/vault context: 1" in out
    assert ">=2 items on one entity with different values: 1" in out
    assert "report-only: nothing here changes prefetch or assembly" in out
    assert list(tmp_path.iterdir()) == [tmp_path / "a.json"]


# ── what a saved turn actually looks like, and what the report owes a reader ──


def test_a_block_carried_on_a_list_content_message_is_still_counted(tmp_path):
    """Measured on the 200 newest transcripts before this was written: the
    injected block rides on list-content messages (154 `subliminal`, 7 `user`) far
    more often than on a plain string, so reading only `str` content would report
    an empty distribution — which reads as "conflicts are rare" instead of "the
    instrument saw nothing"."""
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"messages": [
        {"role": "subliminal", "content": [{"type": "text", "text": P.FIXTURE}]},
        {"role": "user", "content": [{"type": "text", "text": "question"},
                                     {"type": "text", "text": P.FIXTURE}]},
        {"role": "assistant", "content": "answer"}]}), encoding="utf-8")
    refs = [ref for ref, _ in P._iter_turn_blocks(f)]
    assert refs == ["s.json#msg0:subliminal", "s.json#msg1:user"]
    rep = P.collect(tmp_path, days=1, resolve=P.FIXTURE_KG.get)
    assert rep["n_turns"] == 2 and rep["n_turns_with_conflict"] == 2


def test_the_report_states_its_own_resolution_coverage(tmp_path, capsys, monkeypatch):
    """An unresolved item can never be counted either way, so a conflict number
    without its coverage is a number that can be misread as a rate. `collect`
    carries the coverage and the CLI prints it."""
    _session(tmp_path, "a.json", [P.FIXTURE])
    rep = P.collect(tmp_path, days=1, resolve=P.FIXTURE_KG.get)
    assert rep["n_items"] == 7 and rep["n_resolved"] == 6 and rep["n_unresolved"] == 1
    assert rep["resolution_coverage"] == 0.8571
    bare = P.collect(tmp_path, days=1, resolve=lambda n: None)
    assert bare["n_resolved"] == 4 and bare["resolution_coverage"] == 0.5714

    monkeypatch.chdir(tmp_path)
    assert P.main(["--sessions-dir", str(tmp_path), "--no-kg", "--days", "1"]) == 0
    out = capsys.readouterr().out
    assert "resolved to an entity: 4 (coverage 0.5714), unresolved: 3" in out
    assert "is a floor on that sample, not a rate" in out
