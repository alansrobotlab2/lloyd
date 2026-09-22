"""Which DOCUMENT corpus a retrieval run scored, recorded beside the fact corpus.

Backlog #1374. ``eval/run_eval.py`` has always recorded the *fact* half of what
it scored — ``facts_root``, ``kg_db``, ``entities``, ``edges_active``, ``facts``
— and has never recorded the *document* half, which is the half that produced
every ``doc_hit`` / ``ndcg10`` / ``mrr_doc`` in the artifact. It does not need
to: the recall reaches qmd over HTTP at whatever the daemon holds at that
minute (``agent_mcp/vault.py:70 QMD_DAEMON_URL``), and the daemon re-embeds
continuously. Measured on 2026-09-22 inside one daemon process that never
restarted: ``vecIndex.vectors`` 38 768 at 14:11Z and 39 210 at 14:41Z — 442
vectors, and no artifact anywhere that says so.

So ``scripts/eval_trend_stats.py``'s corpus diff — which diffs
``facts``/``edges_active``/``entities`` — prints ``corpus identical`` across a
re-embed that moved every document-side number it is comparing. That is the
confound ``architecture/retrieval.md`` §2 names for the pinned regression arms
("two arms on different snapshots are only compared through a control arm"),
and the nightly series breaks it by construction because it has no snapshot to
name: ``PinnedCorpus`` is constructed only in ``workers/sources/automod_regression.py``,
never by ``run_eval.py``.

There is a second drifting half, and the pin already knows it (see
``evalpin.py:564 env_for``): the keyword leg greps the repository it ships in,
so any commit that adds prose to ``app/`` or ``scripts/`` moves ``doc_hit``
between two nightlies with no retrieval change at all.

Both halves are therefore recorded here, at the moment the run scores them:

* the daemon's own ``vecIndex.vectors`` — the count that decides what the vector
  leg could return, and the term that turns doc drift non-zero downstream;
* the identity of what was reached — the health URL, plus the index file's
  path/mtime/size and its ``content_vectors`` row count when the file is
  readable, and whether this run was served a pinned snapshot or the live
  daemon (``corpus_mode``, derived from ``LLOYD_CONFIG_OVERLAY`` — the one thing
  ``PinnedCorpus.env_for`` sets to redirect ``services.qmd``). Recording that is
  what keeps the live-vs-pinned choice *visible* while it stays a person's call;
* the ``code_root`` the keyword leg grepped.

The rule the whole module obeys, and the one ``eval_trend_stats`` imports rather
than restating: **the ``doc`` block exists exactly when the vector count is
known.** A probe that cannot answer writes ``null``, and ``null`` downstream
means *unknown* — never 0, never "identical". That is the same rule
``eval_trend_stats.corpus_diff`` already applies to a baseline with no ``corpus``
block at all, and it is why every artifact written before this key existed —
which is every artifact on disk the day this landed — reads as doc drift unknown
rather than as doc drift zero. No count is quoted here on purpose: the directory
grows nightly, and a number in this sentence is stale before the round that
wrote it lands.

Third contract, and the one no clause asks for: **the block holds identity, not
liveness.** ``eval/ci_backtest.py:125`` canonicalises this entire ``corpus`` dict
into a fingerprint and runs a PAIRED test only when two artifacts' fingerprints
match. An elapsed-time counter (``uptime``) or a behaviour counter (the rerank
leg's ``ranked``/``fallbacks``) ticks between two runs on a corpus that did not
move, so recording one here would cost those two runs their paired test and
silently weaken the instrument on the same-day pairs this item is about. What may
be recorded is a value that can only change if the scored corpus changed.

Stdlib only, like the rest of ``app/``'s leaf modules: this runs inside the eval
process and inside ``scripts/eval_trend_stats.py``, neither of which should
inherit a config loader to find out what a daemon counted.
"""
from __future__ import annotations

import json
import os
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

#: Key for the document half inside an artifact's ``corpus`` block. Owned here so
#: the writer (`eval/run_eval.py`) and the reader (`scripts/eval_trend_stats.py`)
#: cannot come to disagree about which key means "this run recorded its corpus".
DOC_KEY = "doc"
#: Key for the document term inside the *drift* dict `corpus_diff` returns. Named
#: for what it counts, because it sits beside `facts`/`edges_active`/`entities`
#: in the same printed line and a bare `doc` reads like a category.
DOC_DRIFT_KEY = "doc_vectors"

#: Escape hatch for the probe, same shape as `LLOYD_FACTS_ROOT` /
#: `LLOYD_KG_DB` / `LLOYD_CODE_ROOT`: a test that must record "the daemon said
#: zero" cannot wait for the live index to empty itself. The URL actually probed
#: is recorded in the artifact beside the number, so an override can never pass
#: for a measurement of the live daemon.
HEALTH_URL_ENV = "LLOYD_QMD_HEALTH_URL"
#: Which sqlite file the daemon being scored serves. Unset, this is qmd's own
#: default; a pinned run is served a different file, so
#: `PinnedCorpus.env_for` sets this to the snapshot it started the daemon on.
INDEX_PATH_ENV = "LLOYD_QMD_INDEX"
#: The env var that says "this process was pointed at a pinned corpus", and the
#: only signal that distinguishes a pinned nightly from a live one without
#: deciding which one it should have been.
CONFIG_OVERLAY_ENV = "LLOYD_CONFIG_OVERLAY"

PROBE_TIMEOUT_SECONDS = 5.0


#: Why the default index path is refused under an overlay, recorded verbatim so a
#: reader of an artifact does not need this module to decode a null.
PIN_UNNAMED_REASON = (
    "LLOYD_CONFIG_OVERLAY is set, so this run scores a pinned qmd index, but "
    "LLOYD_QMD_INDEX was not set; the default ~/.cache/qmd/index.sqlite is a "
    "different daemon's file. Naming the live index for a pinned run would be worse "
    "than silence: an identity that reads as precise and is wrong, on an artifact "
    "whose every doc number came from a snapshot."
)

#: The file qmd serves when launched with no `--index`. A default for a LIVE run
#: only — see `index_path_for` for why a pinned run may not inherit it.
LIVE_INDEX = Path.home() / ".cache" / "qmd" / "index.sqlite"


def index_path_for(index_path: Path | str | None = None) -> Path | None:
    """The index file this process may describe, or None when it cannot name it.

    The default is correct for a live run and wrong for a pinned one. qmd is served
    its index by filename (`scripts/automod/evalpin.py:183-196` passes
    `--index <name>`, and `pin_index_path` at `:133` puts the pin at
    `~/.cache/qmd/<name>.sqlite`), so under an overlay this process cannot know
    which file the daemon it queried is serving. `PinnedCorpus.env_for` names it
    through `INDEX_PATH_ENV`; when that did not happen, the honest answer is None
    plus `PIN_UNNAMED_REASON`, never the live default — a wrong identity would read
    as a precise one, and would make a pinned run's artifact claim the live index's
    mtime as its own.
    """
    explicit = index_path or os.environ.get(INDEX_PATH_ENV)
    if explicit:
        return Path(str(explicit)).expanduser()
    return None if os.environ.get(CONFIG_OVERLAY_ENV) else LIVE_INDEX


def health_url_for(query_url: str | None = None) -> str:
    """The ``/health`` of whatever daemon ``query_url`` points at.

    Derived from the recall's own URL rather than from a constant, so the probe
    cannot answer about a daemon on :8181 while the recall was served by the pin
    on :8182 — the two halves of one measurement must describe one corpus.
    """
    override = os.environ.get(HEALTH_URL_ENV)
    if override:
        return override
    from urllib.parse import urlsplit

    parts = urlsplit(query_url or "")
    if not parts.scheme or not parts.netloc:
        # Empty, not a raise and not a hard-coded fallback port: an eval that cannot
        # name its daemon records `doc: null` and the audit prints `unknown`. Killing
        # a run of 87 queries over a provenance field, or inventing the :8181 this
        # module exists to avoid naming, are both worse than an honest blank.
        return ""
    return f"{parts.scheme}://{parts.netloc}/health"


def fetch_json(url: str, timeout: float = PROBE_TIMEOUT_SECONDS) -> dict:
    """GET ``url`` and parse it as JSON. Raises on any failure — the caller
    decides what "could not be answered" means, this function does not guess."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _int(value) -> int | None:
    """An int, or None. Zero is an answer; a string, a float, an absent key and
    None are not. Truthiness cannot tell those apart, which is the whole reason
    this helper exists instead of ``int(x or 0)``."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def count_content_vectors(index_path: Path) -> int | None:
    """``select count(*) from content_vectors`` on ``index_path``, read-only.

    The daemon's own count is what the vector leg searched; this is what the
    index file on disk holds. Recording both is the only way a later reader can
    tell "the corpus grew and the daemon re-embedded it" from "the file grew and
    the daemon has not read it yet". A missing table, an unreadable file or a
    locked database all answer None, which is unknown, never 0.
    """
    try:
        con = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True, timeout=10)
    except sqlite3.Error:
        return None
    try:
        return _int(con.execute("select count(*) from content_vectors").fetchone()[0])
    except sqlite3.Error:
        return None
    finally:
        con.close()


#: The five identity keys, in one place so `collect` cannot quietly start
#: shipping six and stop shipping four. A pinned identity (index `evalpin.sqlite`
#: instead of `index.sqlite`) is worth roughly one false `paired` verdict per
#: pinned run, which is why the null-plus-reason case is spelled out rather than
#: left to the fallback.
IDENTITY_KEYS = ("index_path", "index_mtime", "index_size_bytes",
                 "content_vectors", "index_reason")


def _no_identity(reason: str) -> dict:
    return {k: None for k in IDENTITY_KEYS} | {"index_reason": reason}


def index_identity(index_path: Path | str | None = None) -> dict:
    """Path / mtime / size / row count of the index file this run believes served it.

    Always returns a dict with every `IDENTITY_KEYS` entry: the four identity
    fields, plus `index_reason`, which is None when they are filled and a sentence
    when they are not. A dict-with-a-reason beats a bare None because "could not
    name it" has two very different causes — there is no file, and a pin is serving
    a file this process cannot name — and an artifact reader deserves to tell them
    apart without reading this module.

    ``mtime`` is the cheap proof of a re-embed that needs no daemon at all: the
    watcher writes the same file the vector leg reads.
    """
    path = index_path_for(index_path)
    if path is None:
        return _no_identity(PIN_UNNAMED_REASON)
    try:
        st = path.stat()
    except OSError:
        return _no_identity(f"no qmd index file at {path}")
    return {
        "index_path": str(path),
        "index_mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        "index_size_bytes": st.st_size,
        "content_vectors": count_content_vectors(path),
        "index_reason": None,
    }


def collect(query_url: str | None = None, *, code_root: Path | str | None = None,
            index_path: Path | str | None = None,
            fetch: Callable[[str], dict] | None = None) -> dict | None:
    """The document corpus this run scored, or None when it cannot be answered.

    ``None`` is the only failure mode, and it is load-bearing: the artifact
    writes ``corpus["doc"] = null`` so that every downstream reader is forced to
    say *unknown*. A block with ``vectors: 0`` is a different and much rarer
    thing — a daemon that answered and counted nothing — and it is recorded as
    the measurement it is, which is what lets ``corpus_ok`` refuse it.

    ``fetch`` is injectable for the same reason ``qmd_health.note_response``
    takes an ``announce``/``now``: the probe's failure paths (unreachable,
    not-JSON, no ``vecIndex``, a timeout) must be testable without stopping a
    supervised daemon.
    """
    url = health_url_for(query_url)
    try:
        health = (fetch or fetch_json)(url)
    except Exception:
        # Any failure at all — URLError, JSONDecodeError, a body that parsed to
        # a list, a raise from inside a test's fake. The caller's question is
        # "do you know the count", and "no" is one bit, not a stack trace.
        return None
    if not isinstance(health, dict):
        return None
    vec = health.get("vecIndex")
    vectors = _int(vec.get("vectors")) if isinstance(vec, dict) else None
    if vectors is None:
        return None

    block = {
        "vectors": vectors,
        "health_url": url,
        # Was this corpus frozen for the run (a pin) or whatever the live daemon
        # happened to hold? Recorded, not decided: which one the nightly *should*
        # run under is a person's call — a second daemon holding an embedding
        # model on GPU 0 beside the primary engine is not a change an
        # unattended round makes. See skills/retrieval-eval/SKILL.md.
        "corpus_mode": "pinned" if os.environ.get(CONFIG_OVERLAY_ENV) else "live",
        "code_root": str(code_root) if code_root else None,
    }
    # Deliberately absent — `uptime`, the rerank leg's `ranked`/`fallbacks`, and
    # `vecIndex.fullBuilds` / `incrementalRefreshes`, which the triage asked for and
    # which are NOT in the block. All four are counters of the daemon PROCESS, not of
    # the corpus: a daemon restarted over an unchanged index answers `fullBuilds 0 ->
    # 1`, `incrementalRefreshes 250 -> 0`, and `vectors` identical. That matters
    # because a second consumer canonicalises this whole block to decide whether two
    # runs scored the SAME corpus — `eval/ci_backtest.py:125` builds a fingerprint out
    # of `corpus` and only runs a PAIRED test when it matches — so shipping a
    # process-lifetime counter would make the fingerprint move while the corpus stood
    # still, and cost the instrument its power on the same-day pairs this item is
    # about. `vectors` is the corpus event; `index_mtime` is the same event from the
    # file side, and neither depends on how long the daemon has been up.
    identity = index_identity(index_path)
    block.update({k: identity.get(k) for k in IDENTITY_KEYS})
    return block


def vectors_of(corpus: dict | None) -> int | None:
    """The recorded vector count of one artifact's ``corpus`` block.

    None for every artifact written before this key existed, for one that wrote
    ``doc: null``, and for a block whose count was not an int. All three are
    *unknown*, and unknown must never be arithmetic'd into a 0.
    """
    block = (corpus or {}).get(DOC_KEY)
    return _int(block.get("vectors")) if isinstance(block, dict) else None


def doc_ok(corpus: dict | None) -> bool:
    """Whether the recorded document corpus is scoreable.

    Unknown is not a defect of the corpus, so an unanswerable probe does not
    fail this — the artifact already says ``doc: null``, and the trend audit
    prints that pair as unknown drift. A daemon that answered zero vectors is a
    document corpus that cannot be scored: every ``doc_hit`` in the run would be
    a zero produced by an empty index, indistinguishable from a retrieval
    failure, which is the same blind spot ``corpus_ok`` was built for on the
    fact half.
    """
    vectors = vectors_of(corpus)
    return True if vectors is None else vectors > 0


def doc_drift(prev_corpus: dict | None, cur_corpus: dict | None) -> int | None:
    """Later minus earlier vector count, or None when either side is unknown.

    The one definition of "doc drift unknown", imported rather than restated by
    ``scripts/eval_trend_stats.py`` — the writer and the reader disagreeing about
    what an absent key means is the defect class this item is filed under.
    """
    prev, cur = vectors_of(prev_corpus), vectors_of(cur_corpus)
    if prev is None or cur is None:
        return None
    return cur - prev
