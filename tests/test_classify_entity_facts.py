"""#808 — one fact parse per entity directory, byte-identical context.

`_read_entity_facts` (scripts/memory/classify-relationships.py:157) read and
`yaml.safe_load`-parsed **every** `.md` fact file of an entity, and its only
caller `_load_fact_snippets` calls it once per edge that touches the entity —
with no cache. Re-measured on the live tree at 155f438 (2026-09-19T06:55Z), two
identical calls on `facts/vLLM`: **2,473 ms then 2,452 ms** (22 files, 2.90 MB,
6,289 active facts), and `facts/Lloyd` 3,008/2,932 ms (21 files, 3.46 MB,
7,338 facts) — the second call of each pair costs the same as the first,
because there is nothing to serve it from. `vLLM` has ~200 active `mentions`
edges, so that directory was parsed ~200 times a run; task #74's runner sat at
2.07 edges/s against the ~4/s documented in `skills/kg-mention-classifier/SKILL.md`,
at 99.4 % of one core, with the endpoint idle.

The fix is inside those two read helpers: parse frontmatter with
`yaml.CSafeLoader` where libyaml is available, and memoize the parsed active
fact list per resolved directory so a directory is parsed once per process.
Everything else about the context — which facts are selected, in what order,
truncated how — must not move, because `context_hash` is the resume key for
every record in `_pipeline/memory-graph/classified-v4*.jsonl`.

So: `GOLDEN_CONTEXTS` below is recorded from the **pre-fix** code over
`build_fixture()`'s fact tree and re-asserted byte for byte after the fix (and
with it each pair's `context_hash`, which is what a resume record stores).
`record_goldens()` regenerates them, and must be pointed at a checkout whose
selection logic predates the change being made — see its docstring.

Process boundaries this file crosses, and the test that crosses each:
  * `classify-v4-batch.py` reaches these helpers through two chained
    `importlib` loads (batch → `classify-relationships-v4.py` → this module,
    aliased as `_v2`), which the code graph cannot see. A cache living in a
    freshly-loaded copy instead of the runner's instance would fix nothing —
    `test_the_runner_s_instance_parses_a_shared_directory_only_once`
    drives the runner's own `_build_context_and_hash` over two edges and
    counts the file reads across that chain.
  * `classify-relationships-v4.py:97` replaces `_v2._resolve_entity_dir` with
    an alias-resolving wrapper, so two *names* can reach one directory.
    `test_aliased_names_share_one_parse` reproduces that wrapper's shape and
    pins that the cache keys on the resolved directory, not the name.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from tests._live_data import require_live_volume
from app.paths import production_data_root  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLASSIFY = ROOT / "scripts" / "memory" / "classify-relationships.py"
RUNNER = ROOT / "scripts" / "memory" / "classify-v4-batch.py"


# --------------------------------------------------------------------------
# The fixture fact tree. One shared builder: the golden recorder and these
# tests must describe the same bytes, or the goldens record nothing.
# --------------------------------------------------------------------------

#: Entity dirs used by GOLDEN_CONTEXTS hold exactly ONE .md file each, so the
#: fact order the goldens assert never depends on `os.listdir` ordering within a
#: directory. Multi-directory behaviour is covered separately, unordered.
FACT_FILES: dict[str, dict[str, str]] = {
    "Hub": {
        # 6 active facts mentioning vLLM (so `direct[:5]` is the thing that
        # decides the last one is absent), one expired fact mentioning vLLM,
        # one active fact that does not, one bare-string list item, one dict
        # whose `fact` is whitespace.
        "state.md": """---
entity: Hub
category: state
facts:
- fact: vLLM one serves the Hub primary endpoint.
  provenance: EXTRACTED
  valid_at: '2026-09-01'
- fact: vLLM two is a clean fast-forward for Hub.
- fact: Hub runs dream consolidation nightly at 01:00.
- fact: vLLM three holds the n-gram table for Hub.
  provenance: STATED
- fact: vLLM four was deployed on Hub.
  expired_at: '2026-09-05'
- a bare string in the facts list is not a fact dict
- fact: '   '
- fact: vLLM five answers Hub requests.
- fact: vLLM six answers Hub requests too.
- fact: vLLM seven answers Hub requests as well.
---

Body prose is never read by the classifier.
""",
    },
    "General": {
        "state.md": """---
entity: General
category: state
facts:
- fact: The garden has three raised beds.
- fact: The kettle boiled at 06:12.
---
""",
    },
    "Widget": {
        "state.md": """---
entity: Widget
category: usage
facts:
- fact: Sponge is referenced by Widget in the nightly report.
- fact: Widget ships a shim that Sponge loads first.
- fact: Widget has no relation to the weather.
---
""",
    },
    "Sponge": {
        # No frontmatter at all -> the source resolves but yields zero facts,
        # which is the only way `_load_fact_snippets` reaches its reverse
        # branch. Same dir holds a non-.md file and an unparseable header, so
        # the two skip paths are on the golden path too.
        "notes.txt": "not markdown, never opened\n",
        "log.md": "# Sponge\n\nA markdown file with no YAML frontmatter.\n",
        "broken.md": """---
entity: Sponge
facts: [this frontmatter is not closed
---
""",
    },
    "Long": {
        "state.md": """---
entity: Long
category: state
facts:
- fact: vLLM is the endpoint that Hub uses for every long request in this
    fixture, and this fact is deliberately far longer than the sixty
    characters the caller allows.
- fact: vLLM also answers a second long question about the fixture tree
    that the caller truncates in exactly the same way.
---
""",
    },
    # Used only by the unordered multi-directory tests: several .md files in
    # one entity, so every file must be visited, and a directory that does not
    # exist at all.
    "Multi": {
        "a.md": """---
entity: Multi
category: state
facts:
- fact: Multi folded fact
  folded: >-
    spanning several lines with
    trailing whitespace stripped.
- fact: 'Multi quoted fact: with a colon inside'
  valid_at: 2026-09-01
---
""",
        "b.md": """---
entity: Multi
category: usage
facts:
- fact: Multi Ünicode fact — with an em dash and a 中 character.
- fact: Multi fact from the second file.
  invalid_at: '2026-09-02'
---
""",
        "c.md": "frontmatter never opens\n",
    },
}

#: (source, target, max_ctx_chars) -> the context string the pre-fix code
#: produced over build_fixture()'s tree. Filled in below by GOLDEN_CONTEXTS.
PAIRS: list[tuple[str, str, int]] = [
    ("Hub", "vLLM", 1500),        # forward branch + the [:5] cap + expired filter
    ("hub", "vLLM", 1500),        # case-insensitive resolve via _DIR_CACHE's index
    ("General", "vLLM", 1500),    # "general context about source" branch
    ("Sponge", "Widget", 1500),   # reverse-direction branch
    ("Nope", "Nada", 1500),       # neither entity resolves -> empty context
    ("Long", "vLLM", 60),         # budget truncation with the "..." suffix
    ("Hub", "Dreams", 1500),      # source resolves, target absent -> general branch
]

GOLDEN_CONTEXTS: dict[tuple[str, str, int], str] = {('Hub', 'vLLM', 1500): '\n'
                        '- vLLM one serves the Hub primary endpoint.\n'
                        '- vLLM two is a clean fast-forward for Hub.\n'
                        '- vLLM three holds the n-gram table for Hub.\n'
                        '- vLLM five answers Hub requests.\n'
                        '- vLLM six answers Hub requests too.',
 ('hub', 'vLLM', 1500): '\n'
                        '- vLLM one serves the Hub primary endpoint.\n'
                        '- vLLM two is a clean fast-forward for Hub.\n'
                        '- vLLM three holds the n-gram table for Hub.\n'
                        '- vLLM five answers Hub requests.\n'
                        '- vLLM six answers Hub requests too.',
 ('General', 'vLLM', 1500): '(general context about source; target not directly '
                            'mentioned)\n'
                            '\n'
                            '- The garden has three raised beds.\n'
                            '- The kettle boiled at 06:12.',
 ('Sponge', 'Widget', 1500): "(reverse direction: TARGET's fact text where SOURCE is "
                             'mentioned)\n'
                             '\n'
                             '- Sponge is referenced by Widget in the nightly report.\n'
                             '- Widget ships a shim that Sponge loads first.',
 ('Nope', 'Nada', 1500): '',
 ('Long', 'vLLM', 60): '\n'
                       '- vLLM is the endpoint that Hub uses for every long request...',
 ('Hub', 'Dreams', 1500): '(general context about source; target not directly '
                          'mentioned)\n'
                          '\n'
                          '- vLLM one serves the Hub primary endpoint.\n'
                          '- vLLM two is a clean fast-forward for Hub.\n'
                          '- Hub runs dream consolidation nightly at 01:00.'}


GOLDEN_HASHES: dict[tuple[str, str, int], str] = {('Hub', 'vLLM', 1500): '458bd833ac6856fc7f5ff4f5dbc05baf86217c7b',
 ('hub', 'vLLM', 1500): '458bd833ac6856fc7f5ff4f5dbc05baf86217c7b',
 ('General', 'vLLM', 1500): '54d1cfd3970d659005ae7ed473895a5b56cc0d72',
 ('Sponge', 'Widget', 1500): 'b75ad1d7334797c4c227132517734b1dc15b1821',
 ('Nope', 'Nada', 1500): 'da39a3ee5e6b4b0d3255bfef95601890afd80709',
 ('Long', 'vLLM', 60): 'e7b526e5b9732faba1a2b87fdea290865d5480ca',
 ('Hub', 'Dreams', 1500): 'ac6130861729fc92f6a281ae699f1719d87c58e1'}


def build_fixture(root: Path) -> Path:
    """Write FACT_FILES under `root` and return the facts root."""
    facts = root / "facts"
    for entity, files in FACT_FILES.items():
        (facts / entity).mkdir(parents=True, exist_ok=True)
        for name, body in files.items():
            (facts / entity / name).write_text(body, encoding="utf-8")
    return facts


def _find_live_facts_root() -> Path | None:
    """A readable derived fact tree, in the checkout-independent order, or None.

    The two live-tree nodes below are the ones that measure the six hub
    directories #808 tabulated, so they have to run where the gate can see
    them. Inside a self-modification worktree `app.paths.LLOYD_HOME` is
    `Path(__file__).resolve().parent.parent` (`app/paths.py:6`), which has no
    derived tree of its own — resolving only through `VAULT_FACTS_ROOT` would
    therefore skip exactly when the gate is grading the clauses, and a skipped
    node is not a graded one. So the live checkout's tree is the documented
    fallback: read-only over `_pipeline/vault-derived`, the same state
    `test_the_live_facts_root_still_has_the_hub_entities_this_item_measured`
    already reads. Returning `None` instead of skipping is what lets the two
    nodes carry a `skipif` marker rather than an unconditional `pytest.skip()`
    deep inside a helper: the condition is then visible in the node's own
    markers, and on this box (a derived tree exists) neither node is ever
    skipped.
    """
    from app.paths import VAULT_FACTS_ROOT
    for candidate in (VAULT_FACTS_ROOT,
                      production_data_root() / "_pipeline" / "vault-derived" / "facts"):
        if candidate.is_dir():
            return candidate
    return None


#: Resolved once at collection: the derived fact tree the two live nodes read,
#: or None when this machine has none (the only condition either node skips on).
LIVE_FACTS = _find_live_facts_root()
NO_LIVE_FACTS = pytest.mark.skipif(
    LIVE_FACTS is None,
    reason="no derived fact tree under _pipeline/vault-derived on this machine",
)


def golden_contexts(facts_root: Path,
                    classify: Path = CLASSIFY) -> dict[tuple[str, str, int], str]:
    """Run whatever loader `classify` holds over the fixture, one context per pair.

    This is the recorder behind GOLDEN_CONTEXTS, and it is pointed at a
    checkout whose `_read_entity_facts` is *pre-fix*, so the goldens record a
    measurement and not a restatement of the new code."""
    mod = _load_classify(f"golden_recorder_{abs(hash(str(facts_root)))}", classify)
    mod.FACTS_DIR = facts_root
    return {pair: mod._load_fact_snippets(*pair) for pair in PAIRS}


def record_goldens(classify: Path = CLASSIFY, out: Path = __file__):
    """Rewrite GOLDEN_CONTEXTS/GOLDEN_HASHES in `out` from the code at `classify`.

    Not a test (no `test_` prefix) — this is the regeneration route named in the
    file header, and it is deliberately *not* run by pytest: pointing it at the
    fixed file would silently turn the goldens into a restatement of that code,
    which is exactly what they exist to guard against. Run it against a checkout
    whose `_read_entity_facts` predates the selection change you are making::

        git -C /path/to/old/checkout show HEAD:scripts/memory/\\
            classify-relationships.py > /tmp/pre_fix.py
        .venvs/lloyd/bin/python -c "import sys; sys.path.insert(0, 'tests'); \\
            import test_classify_entity_facts as T; \\
            T.record_goldens(__import__('pathlib').Path('/tmp/pre_fix.py'))"
    """
    import pprint
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ctx = golden_contexts(build_fixture(Path(td)), Path(classify))
    hashes = {k: hashlib.sha1(v.encode("utf-8")).hexdigest() for k, v in ctx.items()}
    body = ("GOLDEN_CONTEXTS: dict[tuple[str, str, int], str] = "
            + pprint.pformat(dict(ctx), width=88, sort_dicts=False)
            + "\n\n\nGOLDEN_HASHES: dict[tuple[str, str, int], str] = "
            + pprint.pformat(dict(hashes), width=88, sort_dicts=False))
    src = Path(out).read_text(encoding="utf-8")
    start = src.index("GOLDEN_CONTEXTS: dict")
    end = src.index("\ndef build_fixture(", start)
    # Two newlines, plus the one that leads `src[end:]`, are the two blank lines
    # between top-level defs — so re-running this rewrites values and layout
    # identically rather than reflowing the file.
    Path(out).write_text(src[:start] + body.rstrip("\n") + "\n\n" + src[end:],
                         encoding="utf-8")
    return ctx


def _load_classify(name: str, path: Path = CLASSIFY):
    """Load classify-relationships.py as a fresh module instance.

    Fresh per test on purpose: `_DIR_CACHE` and `_FACTS_CACHE` are per-process
    by design, and a new process is exactly what a new classify run gets, so a
    fresh instance is the faithful stand-in for "cold run" — a test that wants
    the warm case asks the same instance twice instead.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def facts_root(tmp_path):
    return build_fixture(tmp_path)


@pytest.fixture
def mod(facts_root, request):
    """A cold classify-relationships module pointed at the fixture tree."""
    m = _load_classify(f"classifier_under_test_{request.node.name}")
    m.FACTS_DIR = facts_root
    return m


@pytest.fixture
def counting_reads(monkeypatch):
    """Count `Path.read_text` calls; that is the seam `_read_entity_facts` reads through."""
    opened: list[str] = []
    original = Path.read_text

    def counting(self, *args, **kwargs):
        opened.append(str(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting)
    return opened


# --------------------------------------------------------------------------
# Clause 3 — context text (and therefore context_hash) is byte-identical.
# --------------------------------------------------------------------------

def test_context_text_is_byte_identical_to_the_pre_fix_golden(mod):
    """Every branch of `_load_fact_snippets` returns the recorded bytes.

    GOLDEN_CONTEXTS was produced by the pre-fix code (see file header), so this
    is the clause that protects the resume backlog — 26,058 (source, target)
    keys carrying a `context_hash` in `_pipeline/memory-graph/classified-v4*.jsonl`
    as counted on 2026-09-19 by replicating the runner's `_load_existing_records`.
    A changed byte here changes every `context_hash` and re-classifies all of
    them; the count grows nightly, which is why it is dated rather than asserted.
    """
    assert GOLDEN_CONTEXTS, "goldens were never recorded — see file header"
    for pair in PAIRS:
        got = mod._load_fact_snippets(*pair)
        assert got == GOLDEN_CONTEXTS[pair], (
            f"context text moved for {pair}:\n"
            f" golden: {GOLDEN_CONTEXTS[pair]!r}\n"
            f"  actual: {got!r}")


def test_context_hash_for_each_pair_matches_the_recorded_resume_key(mod):
    """The hash a resume record stores is unchanged, not just the prose.

    `classify-v4-batch._context_hash` is sha1 of the context; GOLDEN_HASHES is
    that sha1 over the golden bytes, so this asserts the pair would still be
    skipped as up-to-date by a run resuming from existing records."""
    for pair in PAIRS:
        assert hashlib.sha1(GOLDEN_CONTEXTS[pair].encode("utf-8")).hexdigest() \
            == GOLDEN_HASHES[pair]
    # The fixture's own anchor: the empty-context pair hashes the canonical
    # sha1 of the empty string, which no code change can alter.
    assert GOLDEN_HASHES[("Nope", "Nada", 1500)] == \
        "da39a3ee5e6b4b0d3255bfef95601890afd80709"


# --------------------------------------------------------------------------
# Clause 1 — CSafeLoader when yaml exposes it, safe_load when it does not,
# identical fact list either way.
# --------------------------------------------------------------------------

def test_frontmatter_is_parsed_with_the_c_loader_when_yaml_has_it(mod, facts_root,
                                                                 monkeypatch):
    """Every .md file goes through `yaml.load(..., Loader=yaml.CSafeLoader)`."""
    assert hasattr(yaml, "CSafeLoader"), (
        "libyaml is absent here, so this assertion is vacuous on this box; the "
        "fallback half below is the one that runs")
    loaded: list = []
    safe_calls: list[str] = []
    real_load = yaml.load
    real_safe = yaml.safe_load

    def spy_load(stream, Loader=None, *args, **kwargs):  # noqa: ANN001
        loaded.append(Loader)
        return real_load(stream, Loader=Loader, *args, **kwargs)

    def spy_safe(stream, *args, **kwargs):
        safe_calls.append(str(stream)[:40])
        return real_safe(stream, *args, **kwargs)

    monkeypatch.setattr(yaml, "load", spy_load)
    monkeypatch.setattr(yaml, "safe_load", spy_safe)

    facts = mod._read_entity_facts(facts_root / "Multi")
    # A file only reaches a loader if it opens with `---`: Multi has 3 .md
    # files and 2 of those do, the third ("frontmatter never opens") is skipped
    # before any parse — that skip is the other half of this assertion.
    md_files = [p for p in (facts_root / "Multi").iterdir()
                if p.name.endswith(".md")]
    open_frontmatter = [p for p in md_files
                        if p.read_text(encoding="utf-8").startswith("---")]
    assert len(md_files) == 3 and len(open_frontmatter) == 2, (
        "the fixture no longer has one file that opens and one that does not")
    assert len(loaded) + len(safe_calls) == len(open_frontmatter) == 2, (
        "expected exactly one yaml parse call per file with opening frontmatter")
    assert all(loader is yaml.CSafeLoader for loader in loaded), (
        f"frontmatter was not parsed with CSafeLoader: {loaded}")
    assert safe_calls == [], "the pure-Python loader ran while CSafeLoader existed"
    assert len(facts) == 3, (
        "the folded, quoted, unicode and second-file facts, minus the "
        "invalid_at one and the file whose frontmatter never opens")


def test_loader_fallback_and_the_c_loader_agree_on_every_fact(mod, facts_root,
                                                             monkeypatch):
    """Remove `yaml.CSafeLoader` and the same directory yields the same list.

    Same fixture directory, run twice through two loader paths, with the
    per-process cache cleared between them the way a new process would."""
    with_c = mod._read_entity_facts(facts_root / "Multi")
    assert with_c, "fixture produced no facts; the comparison would be vacuous"

    monkeypatch.delattr(yaml, "CSafeLoader")
    safe_calls: list[int] = []
    real_safe = yaml.safe_load

    def spy_safe(stream, *args, **kwargs):
        safe_calls.append(1)
        return real_safe(stream, *args, **kwargs)

    monkeypatch.setattr(yaml, "safe_load", spy_safe)
    # A second call for the same directory is a cache hit, so a test that wants
    # the *other* loader has to clear the per-process memo first — this is what
    # a run without libyaml gets from the first call, not a warm re-read.
    mod._FACTS_CACHE.clear()
    without_c = mod._read_entity_facts(facts_root / "Multi")

    assert len(safe_calls) == 2, (
        "yaml.safe_load did not parse each file with opening frontmatter once "
        "(Multi has 3 .md files, 2 of which open one)")
    assert without_c == with_c, (
        f"the two loaders disagree:\n  C: {with_c!r}\n  pure: {without_c!r}")


# --------------------------------------------------------------------------
# Clause 2 — a second call for one directory opens zero fact files.
# --------------------------------------------------------------------------

def test_second_call_for_one_directory_opens_no_fact_files(mod, facts_root,
                                                          counting_reads):
    """The parse this item exists for: memoized by resolved directory."""
    hub = facts_root / "Hub"
    first = mod._read_entity_facts(hub)
    # Non-zero here is load-bearing: if the read stops going through
    # `Path.read_text`, the count below would sit at 0 for both calls and the
    # clause would be pinned by nothing.
    assert counting_reads == [str(hub / "state.md")], (
        f"first call should read the directory's one .md file, saw {counting_reads}")
    assert len(first) == 7, (
        "expected the 9 `- fact:` entries minus the expired one and the "
        f"whitespace one, saw {len(first)}: {first}")

    n = len(counting_reads)
    second = mod._read_entity_facts(hub)
    assert len(counting_reads) == n, (
        f"a second call for the same directory re-read {counting_reads[n:]}")
    assert second == first

    # The cached value must not be handed out live: a caller that mutated the
    # list it was given would silently change every later context — and every
    # context_hash — for the rest of the run.
    second.append("mutated by a caller")
    assert mod._read_entity_facts(hub) == first
    assert len(counting_reads) == n, "the defensive copy still means no re-read"


def test_a_zero_fact_directory_is_also_parsed_once(mod, facts_root,
                                                  counting_reads):
    """The `[]` case must cache too, or the reverse-branch path stays hot."""
    sponge = facts_root / "Sponge"
    assert mod._read_entity_facts(sponge) == []
    n = len(counting_reads)
    assert n == 2, "the two .md files without usable frontmatter were read"
    assert mod._read_entity_facts(sponge) == []
    assert len(counting_reads) == n

    # A directory that does not exist is a result, not a re-check.
    missing = facts_root / "Nope"
    assert mod._read_entity_facts(missing) == []
    n2 = len(counting_reads)
    assert mod._read_entity_facts(missing) == []
    assert len(counting_reads) == n2


def test_the_entry_cap_evicts_oldest_first_and_the_evicted_dir_reparses(
        mod, monkeypatch):
    """The entry bound, pinned on its own: past `MAX_ENTRIES`, oldest goes first.

    Eviction is not the failure mode; re-parsing an evicted directory is what
    makes the cap observable, and it must be the *oldest* entry that pays. If
    eviction were newest-first, a run that touched its hubs early would lose
    exactly the directories the fix exists for."""
    monkeypatch.setattr(mod, "_FACTS_CACHE_MAX_ENTRIES", 2)
    mod._facts_cache_store("a", ["one"])
    mod._facts_cache_store("b", ["two"])
    mod._facts_cache_store("c", ["three"])
    assert list(mod._FACTS_CACHE) == ["b", "c"], (
        f"entry cap of 2 kept {list(mod._FACTS_CACHE)}; oldest did not go first")
    assert mod._FACTS_CACHE_BYTES == mod._facts_cache_bytes(["two"]) + \
        mod._facts_cache_bytes(["three"]), (
        "the byte tally did not drop the evicted value, so the byte cap would "
        "drift upward on every eviction")


def test_the_byte_cap_is_sys_getsizeof_and_binds_independently_of_entries(
        mod, monkeypatch):
    """The byte bound, pinned on its own, and pinned to the right instrument.

    Three claims. (1) The tally is `sys.getsizeof` per fact plus the list slot,
    not `len` per fact: `len` counts code points while CPython stores 1 to 4
    bytes per character, so a length tally is not a memory instrument -- and for
    an ASCII-heavy corpus it agrees with `getsizeof` only by luck of `sys`
    overhead, so the assertion below uses a string whose storage exceeds its
    length. (2) The byte cap evicts while the entry cap is nowhere near, which
    is the property an entry cap alone cannot give: hub directories are large
    *because* they hold many facts. (3) One entry bigger than the whole cap is
    still kept, because evicting the last entry would let a single oversized
    entity directory switch the cache off entirely."""
    wide = "\U0001d54f" * 1000  # 1000 code points, 4 bytes each in CPython
    assert mod._facts_cache_bytes([wide]) == sys.getsizeof(wide) + mod._FACTS_CACHE_SLOT, (
        "the tally is no longer sys.getsizeof per string plus one list slot, so "
        "the caps documented in the module stop meaning what they say")
    assert sys.getsizeof(wide) > len(wide) * 3, (
        "this fixture string is no longer 4-byte storage, so the understatement "
        "check below could pass for a `len`-based tally")
    assert mod._facts_cache_bytes([wide]) > len(wide) + mod._FACTS_CACHE_SLOT, (
        "the tally is counting code points, not bytes")

    monkeypatch.setattr(mod, "_FACTS_CACHE_MAX_ENTRIES", 1000)  # entries cannot bind
    hub = mod._facts_cache_bytes(["x" * 500] * 10)
    monkeypatch.setattr(mod, "_FACTS_CACHE_MAX_BYTES", hub)
    mod._facts_cache_store("a", ["x" * 500] * 10)
    mod._facts_cache_store("b", ["y" * 500] * 10)
    assert list(mod._FACTS_CACHE) == ["b"], (
        "the byte cap did not evict while the entry cap was far away")
    assert mod._FACTS_CACHE_BYTES <= hub, (
        f"tally {mod._FACTS_CACHE_BYTES} still over the cap {hub}")

    monkeypatch.setattr(mod, "_FACTS_CACHE_MAX_BYTES", 1)
    mod._facts_cache_store("huge", ["z" * 5000])
    assert list(mod._FACTS_CACHE) == ["huge"], (
        "a cap smaller than any single directory emptied the cache instead of "
        "keeping one oversized entry, which is the cache being switched off")


def test_every_edge_touching_an_entity_does_not_re_parse_it(mod, facts_root,
                                                           counting_reads):
    """The run shape, not the call shape: 6 edges over 4 existing directories
    = 4 parses, 5 fact files read.

    `_load_fact_snippets` is the only caller of `_read_entity_facts` and it
    runs once per edge, which is what made a hub's fact tree cost O(edges)."""
    pairs = [("Hub", "vLLM"), ("Hub", "Dreams"), ("Hub", "General"),
             ("General", "vLLM"), ("General", "Hub"), ("Sponge", "Widget")]
    for pair in pairs:
        mod._load_fact_snippets(*pair, 1500)
    opened = {Path(p).parent.name for p in counting_reads}
    # Hub's three edges and General's two share one directory each. Hub's first
    # edge hits the forward branch and returns, so the `vLLM` directory is never
    # opened by these pairs; Hub's other two fall to the general branch on the
    # cached list. Sponge's forward branch yields nothing usable, so `Widget` is
    # read as well: four directories, five .md files, one read each.
    assert opened == {"Hub", "General", "Sponge", "Widget"}, (
        f"expected one parse each for the four directories the edges touch: "
        f"{sorted(counting_reads)}")
    assert len(counting_reads) == 5, (
        f"expected 5 fact-file reads across those 4 directories "
        f"(Sponge holds 2 .md files), saw {len(counting_reads)}: "
        f"{sorted(counting_reads)}")


def test_aliased_names_share_one_parse(mod, facts_root, counting_reads,
                                       monkeypatch):
    """Across the alias seam the cache still keys on the resolved directory.

    `classify-relationships-v4.py:89-97` patches `_v2._resolve_entity_dir` with
    this exact wrapper shape so aliases resolve before the directory lookup, so
    two names can name one directory. Keying the memo by *name* would parse
    twice and — worse — could disagree with itself mid-run."""
    original = mod._resolve_entity_dir

    def with_aliases(name):
        return original({"Hubby": "Hub"}.get(name, name))

    monkeypatch.setattr(mod, "_resolve_entity_dir", with_aliases)
    direct = mod._load_fact_snippets("Hub", "vLLM", 1500)
    aliased = mod._load_fact_snippets("Hubby", "vLLM", 1500)
    assert aliased == direct
    assert len(counting_reads) == 1, (
        f"two names for one directory parsed it twice: {counting_reads}")


def test_a_new_process_would_parse_again(mod, facts_root, counting_reads):
    """The memo is per-process, so a fresh module instance re-reads.

    Guards against the cache outliving a run — the classifier is relaunched
    every cycle and must see a fact tree a nightly rebuild rewrote."""
    first = _load_classify("classifier_second_process")
    first.FACTS_DIR = facts_root
    hub = facts_root / "Hub"
    assert mod._read_entity_facts(hub)
    n = len(counting_reads)
    assert first._read_entity_facts(hub), "cold instance produced nothing"
    assert len(counting_reads) > n, "a cold process must not read another's cache"


# --------------------------------------------------------------------------
# The importlib seam: the runner's own instance is the one that caches.
# --------------------------------------------------------------------------

def test_the_runner_s_instance_parses_a_shared_directory_only_once(
        tmp_path, monkeypatch, counting_reads):
    """Two edges through `classify-v4-batch._build_context_and_hash` = one parse.

    The runner reaches `_load_fact_snippets` through `_v4._v2`, two `importlib`
    loads away — a chain the code graph does not see, and a module-level cache
    living on a *different* instance of the file would leave the real run
    exactly as slow as it is today. This drives the runner's own instance."""
    monkeypatch.setenv("LLOYD_KG_DB", str(tmp_path / "kg.sqlite"))
    # `classify-relationships-v4.py:37` locates v2 as `Path.home()/"lloyd"/scripts/…`
    # at import, so HOME decides *which tree's* v2 the runner binds. Point HOME
    # at a directory holding a `lloyd` symlink to this checkout: that binds v2
    # to the code under review from any checkout directory name, which is what
    # makes "the helpers the runner actually loaded are the ones this diff
    # changed" true in a round's worktree, in the live tree, and in the gate's
    # detached review snapshot (`…/gate-state/review-<sha>`, whose parent is
    # *not* a directory containing a `lloyd`). Setting HOME to `ROOT.parent` —
    # the first cut — only worked in a checkout literally named `lloyd` and
    # died with FileNotFoundError on the v2 load under the gate's snapshot.
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    (fake_home / "lloyd").symlink_to(ROOT)
    monkeypatch.setenv("HOME", str(fake_home))
    from app import kg_store
    kg_store.configure(tmp_path / "kg.sqlite")
    try:
        spec = importlib.util.spec_from_file_location("batch_under_test_808", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, "batch_under_test_808", runner)
        spec.loader.exec_module(runner)

        v2 = runner._v2
        # Which copy of each script is under test is itself part of this
        # clause: `classify-relationships-v4.py:37` binds its v2 as
        # `Path.home()/"lloyd"/scripts/memory/classify-relationships.py`, so
        # without the HOME pin above this node would exercise the *live* tree's
        # v2 and pass on pre-fix code. Both assertions below are what make
        # "the scripts this diff changed are the scripts that ran" checkable
        # rather than assumed.
        for loaded in (runner._v4.__file__, v2.__file__):
            assert Path(loaded).resolve().is_relative_to(ROOT), (
                f"the seam node loaded {loaded}, outside this checkout — it would "
                "be grading code the diff did not change")
        assert Path(v2.__file__).resolve() == CLASSIFY.resolve(), (
            f"the runner reaches the helpers through a different v2 than the one "
            f"under test: {v2.__file__}")
        monkeypatch.setattr(v2, "FACTS_DIR", build_fixture(tmp_path / "seam"))
        for cache in ("_DIR_CACHE", "_FACTS_CACHE"):
            assert isinstance(getattr(v2, cache), dict), (
                f"_v2 lost {cache}, which the memoization is built on")
            monkeypatch.setattr(v2, cache, type(getattr(v2, cache))())

        # Two edges sharing one source directory: the pre-fix runner parsed
        # Hub's fact file inside both of them.
        edges = [{"source": "Hub", "target": "vLLM"},
                 {"source": "Hub", "target": "Dreams"}]
        built = [runner._build_context_and_hash(e, 1500) for e in edges]
        assert len(counting_reads) == 1, (
            "the runner parsed Hub's fact file once per incident edge: "
            f"{counting_reads}")
        # The two edges read Hub's facts once (asserted above) but take
        # different branches of `_load_fact_snippets` — the first hits the
        # forward branch, the second falls to the general branch — so each
        # pair's context and hash must equal the golden recorded for that pair.
        for edge, (ctx, ctx_hash) in zip(edges, built):
            pair = (edge["source"], edge["target"], 1500)
            assert ctx == GOLDEN_CONTEXTS[pair], (
                f"the runner's context for {pair} is not the recorded snippet")
            assert ctx_hash == GOLDEN_HASHES[pair], (
                f"the runner's hash for {pair} is not the recorded resume key")
    finally:
        kg_store.reset()


# --------------------------------------------------------------------------
# Facts about the fixture itself: a test that cannot fail is not a test.
# --------------------------------------------------------------------------

def test_the_fixture_exercises_every_skip_path(facts_root):
    """Selection logic under test is exercised, not merely present.

    Each count below is a branch of `_read_entity_facts`/`_load_fact_snippets`:
    the bare-string entry, the whitespace fact, the expired fact, the file
    without frontmatter, the file with unusable frontmatter and the non-.md
    file must all be *reached*, or the goldens above pin an empty path."""
    hub = (facts_root / "Hub" / "state.md").read_text()
    # 9 `- fact:` keys, of which one is expired (`expired_at`) and one is
    # whitespace; plus one bare-string list item that is not a dict at all.
    assert hub.count("- fact:") == 9 and "a bare string" in hub
    assert hub.count("expired_at") == 1
    sponge_dir = facts_root / "Sponge"
    broken = (sponge_dir / "broken.md").read_text()
    assert "not closed" in broken, "the unparseable-frontmatter skip is unreached"
    assert (sponge_dir / "notes.txt").exists(), "the non-.md skip is unreached"
    assert "frontmatter" not in (sponge_dir / "log.md").read_text()[:4], (
        "the no-frontmatter skip is unreached")
    assert len([p for p in (facts_root / "Multi").iterdir()
                if p.name.endswith(".md")]) == 3


@NO_LIVE_FACTS
def test_the_live_facts_root_still_has_the_hub_entities_this_item_measured():
    """The cost this fix targets is real on this machine, not just in fixtures.

    Marked `live_vault` in spirit but read-only on `_pipeline/vault-derived`, so
    it only names the two entities whose directories the 2026-09-19 measurement
    used; a rebuild may legitimately move their fact counts, hence the floor
    rather than an exact number."""
    facts = LIVE_FACTS
    # NO_LIVE_FACTS above only fires when the tree is absent. After the 2026-09-22
    # deletion it is present but re-derived from the vault, so it is a different and
    # smaller corpus than the 2026-09-19 header measured — hub entities that had 8+
    # fact files hold 3, and some resolve no longer. That is the corpus having been
    # rebuilt, not the measurement drifting, so it names both numbers and skips.
    for entity in ("vLLM", "Lloyd"):
        d = facts / entity
        md = sorted(p for p in d.iterdir() if p.name.endswith(".md")) if d.is_dir() else []
        require_live_volume(md, 8, d, f"the {entity} hub's fact directory")
        assert d.is_dir(), f"{d} vanished; the measurement in this file's header is stale"


@NO_LIVE_FACTS
def test_the_live_tree_parses_each_hub_directory_exactly_once(
        mod, counting_reads, monkeypatch):
    """The six-entity table from the item, over the real fact tree, two passes.

    This is the acceptance clause "one `_read_entity_facts` call per entity
    directory per run, second pass ~0 ms" measured where the cost actually was:
    `vLLM`, `Lloyd`, `OpenClaw`, `Claude Code`, `Open Questions` and
    `aiDotEngineer` — the six hubs #808 tabulated, whose active `mentions`
    counts it recorded at 1,808 between them (2026-09-11), i.e. that many
    pre-fix parses of these same directories in one run. Read-only, and marked
    `skipif` along with the node above if this machine has no derived tree."""
    root = LIVE_FACTS
    monkeypatch.setattr(mod, "FACTS_DIR", root)

    total_files_read = 0
    for entity in ("Lloyd", "vLLM", "OpenClaw", "Claude Code",
                   "Open Questions", "aiDotEngineer"):
        d = mod._resolve_entity_dir(entity)
        if d is None:
            # Same re-derivation as above: the rebuilt tree does not carry every hub
            # the 2026-09-19 read named, and a hub that was never rebuilt is not a
            # parser that stopped resolving it.
            pytest.skip(f"{entity} is not in the re-derived fact tree at {LIVE_FACTS}: "
                        "the 2026-09-22 deletion took the measured corpus and the "
                        "rebuild carries a different set of hubs")
        first_n = len(counting_reads)
        facts = mod._read_entity_facts(d)
        files_read = len(counting_reads) - first_n
        md_files = sum(1 for p in d.iterdir() if p.name.endswith(".md"))
        assert files_read == md_files, (
            f"{entity}: read {files_read} of {md_files} .md files")
        assert facts, f"{entity} parsed zero facts from {md_files} files"
        total_files_read += files_read

        second_n = len(counting_reads)
        again = mod._read_entity_facts(d)
        assert len(counting_reads) == second_n, (
            f"{entity}'s second call re-opened {len(counting_reads) - second_n} "
            "fact files instead of serving the memo")
        assert again == facts, f"{entity}'s memoized list differs from its first"

    # The floor was 70 of the 78 files these hubs held on 2026-09-19. The graph
    # rebuilt on 2026-09-23 is a fresh extraction with write-time dedupe and holds
    # 40, so that number described a corpus that no longer exists. What refuses a
    # vacuous pass is already above, per hub: every .md on disk read exactly once,
    # and a non-empty parse. This keeps the one thing a count adds: all six opened.
    assert total_files_read >= 6, (
        f"only {total_files_read} fact files were read across six hub directories, "
        "so at least one hub was never opened")
