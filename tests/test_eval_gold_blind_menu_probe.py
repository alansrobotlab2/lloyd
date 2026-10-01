"""#1937: the gold-blind menu probe — three candidate builders, measured, nothing deployed.

The probe exists to answer one question from printed numbers: does any gold-blind
builder offer >= 0.80 of entity gold at a cap <= 60, where the shipped token-overlap
order offers 0.468 at 40. These tests run it on a synthetic pool; the decision is made
by an off-peak run over the live pools, which no test node may do (embedding 12,693
names is minutes of CPU).
"""
from __future__ import annotations

import copy
import io
import json
import socket
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import eval.gold_blind_menu_probe as P
import eval.label_agreement_ceiling as lac

NAMES = ["Robot", "Raspberry Pi 5", "Vision System", "GPU", "Backlog", "Qwen", "Djev",
         "Vault", "Guardian"]
PATHS = ["memory/entities/robot.md", "memory/entities/pi.md", "notes/gpu.md",
         "notes/other.md", "notes/backlog.md"]
ALIASES = {"the board": "Raspberry Pi 5", "ghost": "Not In The Pool"}
CORPUS = [
    # The gold shares no token with the query: the shipped order cannot reach it.
    {"id": "q1", "query": "which board does it use",
     "expect_entities": ["Raspberry Pi 5"], "expect_docs": ["memory/entities/pi.md"]},
    {"id": "q2", "query": "what is the robot",
     "expect_entities": ["Robot"], "expect_docs": ["memory/entities/robot.md"]},
]
EXPANSIONS = {"q1": ["Raspberry Pi 5", "pi"], "q2": []}


def _vec(text: str) -> list[float]:
    """A deterministic toy embedder: one axis per marker word."""
    t = text.lower()
    return [1.0 if ("board" in t or "raspberry" in t or "/pi." in t) else 0.0,
            1.0 if "robot" in t else 0.0, 0.1]


def _run(**kw):
    args = dict(entity_names=NAMES, vault_paths=PATHS, corpus=CORPUS, caps=[1, 2],
                aliases=ALIASES, expansions=EXPANSIONS, embed_fn=_vec)
    args.update(kw)
    return P.probe(**args)


def test_the_probe_takes_its_namespace_and_corpus_as_arguments_and_prints_every_cell(
        monkeypatch):
    """Clause 3. Nothing is opened: the store accessors and every HTTP route are made
    to raise, and the probe still runs on the values it was handed. For each of the
    three builders it prints the entity and doc offered fraction at each cap."""
    def _boom(*a, **k):
        raise AssertionError("the probe opened the live store or the network")

    monkeypatch.setattr(lac, "entity_name_table", _boom)
    monkeypatch.setattr(lac, "vault_markdown_paths", _boom)
    monkeypatch.setattr(lac, "http_post_json", _boom)
    monkeypatch.setattr(lac, "http_get_json", _boom)
    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)

    res = _run()
    assert res["labeler_calls"] == 0 and res["gold_aware"] is False
    assert res["namespace"] == {"entity": len(NAMES), "doc": len(PATHS)}
    assert list(res["builders"]) == [P.BASELINE, *P.BUILDERS] and res["skipped"] == {}
    for name in res["builders"]:
        for leg in ("entity", "doc"):
            assert list(res["builders"][name][leg]) == ["1", "2"], (name, leg)
            for cell in res["builders"][name][leg].values():
                assert cell["labels"] == 2 and 0.0 <= cell["fraction"] <= 1.0

    # The shipped order misses the zero-overlap gold at cap 1; each probe builder,
    # given a side input that names it, reaches it — which is what the probe measures.
    ent = {b: res["builders"][b]["entity"]["1"]["fraction"] for b in res["builders"]}
    assert ent[P.BASELINE] == 0.5, ent
    assert ent["alias"] == ent["embed"] == ent["expansion"] == 1.0, ent

    out = io.StringIO()
    with redirect_stdout(out):
        P.print_report(res)
    text = out.getvalue()
    for name in (P.BASELINE, *P.BUILDERS):
        assert sum(1 for l in text.splitlines() if l.strip().startswith(name + " ")) == 2, (
            f"{name} must print one entity row and one doc row\n{text}")
    assert "cap 1" in text and "cap 2" in text and "labeler calls: 0" in text
    assert "verdict:" in text


def test_a_builder_without_its_side_input_is_skipped_by_name_not_scored_as_zero():
    res = _run(aliases=None, expansions=None, embed_fn=None)
    assert list(res["builders"]) == [P.BASELINE]
    assert set(res["skipped"]) == set(P.BUILDERS)
    assert "nothing to rule on" in P.verdict(res)


@pytest.mark.parametrize("builder", P.BUILDERS)
def test_each_builder_is_a_deterministic_gold_blind_narrowing(builder):
    """Clause 4. Identical inputs give a byte-identical menu; every candidate is a
    member of the pool it was handed; and editing that query's gold in a corpus copy
    leaves the menu unchanged — the gold is read only afterwards, to count."""
    vecs = P.embed_all(set(NAMES) | {q["query"] for q in CORPUS}, _vec)

    def menus(corpus, pool):
        out = {}
        for q in corpus:
            if builder == "alias":
                order = P.rank_alias(q["query"], pool, ALIASES)
            elif builder == "expansion":
                order = P.rank_expansion(q["query"], pool, EXPANSIONS[q["id"]])
            else:
                order = P.rank_embed(vecs[q["query"]], pool, vecs)
            assert sorted(order) == sorted(pool), "a total order over the pool, no more"
            out[q["id"]] = P.menu(order, 3)
        return out

    first = menus(CORPUS, NAMES)
    assert json.dumps(first, sort_keys=True) == json.dumps(menus(CORPUS, NAMES), sort_keys=True)
    assert json.dumps(first) == json.dumps(menus(CORPUS, list(reversed(NAMES)))), (
        "the order must not depend on the order the pool arrived in")
    for menu in first.values():
        assert len(menu) == 3 and set(menu) <= set(NAMES)
    assert "Not In The Pool" not in sum(first.values(), []), "an alias is never a candidate"

    regolded = copy.deepcopy(CORPUS)
    for q in regolded:
        q["expect_entities"] = ["Guardian", "Something Else Entirely"]
        q["expect_docs"] = ["notes/other.md"]
    assert menus(regolded, NAMES) == first

    # And the measurement moves when the gold does, so it is the gold being counted.
    side = dict(aliases=ALIASES, expansions=EXPANSIONS, embed_fn=_vec, builders=[builder])
    before = P.probe(entity_names=NAMES, vault_paths=PATHS, corpus=CORPUS, caps=[1], **side)
    after = P.probe(entity_names=NAMES, vault_paths=PATHS, corpus=regolded, caps=[1], **side)
    assert before["builders"][builder]["entity"]["1"]["offered"] == 2
    assert after["builders"][builder]["entity"]["1"]["offered"] < 2


def test_no_builder_takes_the_gold_or_the_labeler():
    """Gold-blindness by signature, and no HTTP labeler call by construction."""
    import inspect

    for fn in (P.rank_alias, P.rank_expansion, P.rank_embed, P.rank_token_overlap, P.menu):
        params = set(inspect.signature(fn).parameters)
        assert not {p for p in params if "gold" in p or "expect" in p or "label" in p}, fn
    src = Path(P.__file__).read_text(encoding="utf-8")
    for banned in ("http_post_json(", "http_get_json(", "urlopen(", "requests.",
                   "label_corpus(", "labeler("):
        assert banned not in src, f"the probe names {banned}"
    assert "lac._ranked" in src, "positive control: the control row is the shipped order"


def test_the_verdict_applies_the_items_bar_to_the_printed_numbers():
    res = _run(caps=[40, 60, 320])
    assert "deploy candidate" in P.verdict(res)
    low = copy.deepcopy(res)
    for name in P.BUILDERS:
        for cell in low["builders"][name]["entity"].values():
            cell["fraction"] = 0.6
    low["builders"]["alias"]["entity"]["320"]["fraction"] = 0.99   # past the cap bar
    assert "no builder reaches the bar" in P.verdict(low) and "0.6" in P.verdict(low)


def test_the_cli_reads_files_and_never_the_store_unless_asked(tmp_path, monkeypatch, capsys):
    def _boom(*a, **k):
        raise AssertionError("opened the live store without --from-live")

    monkeypatch.setattr(lac, "entity_name_table", _boom)
    monkeypatch.setattr(lac, "vault_markdown_paths", _boom)
    files = {}
    for name, value in (("names", NAMES), ("paths", PATHS), ("aliases", ALIASES),
                        ("exp", EXPANSIONS)):
        files[name] = tmp_path / f"{name}.json"
        files[name].write_text(json.dumps(value), encoding="utf-8")
    corpus = tmp_path / "corpus.yaml"
    import yaml
    corpus.write_text(yaml.safe_dump({"queries": CORPUS}), encoding="utf-8")
    rc = P.main(["--corpus", str(corpus), "--entity-names", str(files["names"]),
                 "--doc-paths", str(files["paths"]), "--aliases", str(files["aliases"]),
                 "--expansions", str(files["exp"]), "--caps", "1,2"])
    out = capsys.readouterr().out
    assert rc == 0 and "skipped embed: no embedder supplied" in out, out
    assert "alias" in out and "expansion" in out
    with pytest.raises(SystemExit):
        P.main(["--corpus", str(corpus)])


def test_the_embedding_cache_round_trips_and_spares_the_embedder(tmp_path):
    """An interrupted live run must not pay for 19,000 strings twice: a cached text
    is never re-embedded, and the saved file gives back the same order."""
    calls: list[str] = []

    def counting(text):
        calls.append(text)
        return _vec(text)

    cache: dict = {}
    first = P.embed_all(NAMES, counting, cache)
    assert sorted(calls) == sorted(NAMES)
    path = tmp_path / "vecs.npz"
    P.save_embed_cache(str(path), cache)
    loaded = P.load_embed_cache(str(path))
    calls.clear()
    second = P.embed_all(NAMES, counting, loaded)
    assert calls == [], "a cached string was embedded again"
    q = P.embed_all(["which board does it use"], _vec)["which board does it use"]
    assert P.rank_embed(q, NAMES, first) == P.rank_embed(q, NAMES, second)
    assert P.rank_embed(q, NAMES, first)[0] == "Raspberry Pi 5"
    assert P.load_embed_cache(str(tmp_path / "absent.npz")) == {}
