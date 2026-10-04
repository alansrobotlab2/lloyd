"""A skill that reads an uptake table's verdict must state it for every key (#1850).

The uptake instrument decides one thing: whether a table may be **cited**. Three
keys carry that verdict, and all three are emitted by the same probe run:

* `engine_rerouted` + `measured` — the scorer that answered is not the engine the
  instrument NAMES, so `measured` is forced false on `metrics`, `holdout` and
  `zero_shot` and `passed` is false (`scripts/uptake_probe.stamp_engine`);
* `audit_only` — the run did not clear its floors and the operator asked for the
  table anyway (`--ignore-precision-floor`), so every dispute count is an
  unvalidated classifier's guess (`uptake.GLOSSARY`).

Only one skill consumes that verdict — `nightly-reflection-knowledge-write` §2a
item 1, which is the job that rewrites loaded memory on the strength of these
numbers. A sentence dropped from that item does not fail a build: it silently
re-arms the failure #1850 exists to close, where a nightly archives a `USER.md`
entry on a table that was never a measurement.

So the invariant is enforced at the writer, which is the architecture
`pytest.ini` states for prose a round does not control ("Enforcement lives at
the writers … this mark is the reporting copy"), and the same shape as
`scripts/reflection_archive.py` and `scripts/skill_timezone.py`: one definition
here, called by `scripts/automod/vault_round.py` on every `skills/**/SKILL.md` a
vault round touches, and scanned by the `live_vault` reporting node in
`tests/test_uptake.py::test_the_knowledge_write_item_states_every_citability_key`.

The obligation is *discovered from the candidate's own bytes*, not from a list of
names: a body that mentions `engine_rerouted` at all is reading the instrument's
verdict, and must state a citability verdict beside each key it names. Measured
on 2026-10-04 over the live vault: exactly 1 of the 202 `SKILL.md` files under
`~/obsidian/skills` names `engine_rerouted` (`nightly-reflection-knowledge-write`),
and it satisfies the rule — so turning this on refuses nothing that lands today.
"""
from __future__ import annotations

from pathlib import Path

#: The keys whose presence decides whether a table may be cited. The seam test
#: (`tests/test_uptake.py::test_the_citability_rule_asks_about_the_keys_uptake_emits`)
#: asserts each of these is a key in the bytes `uptake.write_table` writes, so
#: adding a fourth citability key to the instrument makes this tuple's test red
#: until the consumer-side prose is written too.
CITABILITY_KEYS: tuple[str, ...] = ("engine_rerouted", "measured", "audit_only")

#: A body is an uptake consumer when it names the reroute key at all: the other
#: two keys are generic English (`measured`, `audit`) and cannot identify a
#: reader on their own.
READER_MARKER = "engine_rerouted"

#: Words that, beside a key, state a citability verdict. A body that names a key
#: and none of these is telling the reader the key exists but not what it permits.
VERDICT_PHRASES: tuple[str, ...] = (
    "not citable", "cite nothing", "not a measurement", "not a comparable",
    "not comparables", "do not cite", "never archive", "is citable",
)

#: How far after a key mention a verdict phrase may sit and still be about it.
#: 400 characters is the longest of the three verdict sentences actually shipped
#: in `nightly-reflection-knowledge-write` §2a item 1, plus margin.
VERDICT_WINDOW = 400

#: The re-base #1850 recorded, in the artifact's own reader-facing prose: which
#: engine scores, when that was decided, and when the slot it replaced went away.
#: Dates, not prose, because a paraphrase survives a rewrite and a date does not
#: silently change.
RE_BASE_FACTS: tuple[str, ...] = ("2026-10-04", "2026-09-20", "primary")


def is_uptake_consumer(body: str) -> bool:
    """Does this skill body read the uptake instrument's verdict at all?"""
    return READER_MARKER in body


def detail_text(skill_dir: Path) -> str:
    """The skill's sibling `*.md` files, concatenated (never raises).

    #624's spill route moves a rule out of the body into a sibling the body
    points at, so a requirement may legitimately live there. Discovery is the
    body alone (`skill_rule_violations`' `body`): a skill only owes what its own
    body sends a reader to read.
    """
    try:
        return "\n".join(
            p.read_text(encoding="utf-8", errors="replace")
            for p in sorted(Path(skill_dir).glob("*.md"))
            if p.name != "SKILL.md")
    except OSError:
        return ""


def verdict_stated(corpus: str, key: str) -> bool:
    """Is some citability verdict stated within `VERDICT_WINDOW` of a mention of `key`?"""
    low = corpus.lower()
    needle, phrases = key.lower(), [p.lower() for p in VERDICT_PHRASES]
    start = 0
    while (i := low.find(needle, start)) >= 0:
        window = low[i:i + len(needle) + VERDICT_WINDOW]
        if any(p in window for p in phrases):
            return True
        start = i + len(needle)
    return False


def skill_rule_violations(name: str, body: str, *, detail: str = "") -> list[str]:
    """What this skill fails to tell a reader about the instrument. `[]` = fine.

    `body` is the skill's `SKILL.md`, and it alone decides whether the skill is
    asked at all (`is_uptake_consumer` — a body that never names the reroute key
    sends no reader to the instrument, so nothing is owed). `detail` is the folder's
    sibling `*.md` files: #624's spill route moves a rule out of a body into a
    sibling the body points at, so a requirement may be met there.

    The two are checked **per key, in either file, not as one concatenation**. A
    concatenated corpus would let a verdict pass merely because the sibling happened
    to be appended within `VERDICT_WINDOW` of the body's mention — an artefact of
    read order, and one that flips when `glob` does. So a key is satisfied when its
    verdict sits within `VERDICT_WINDOW` of one of its mentions *inside a single
    file*, and the key must be named in at least one of them too.
    """
    if not is_uptake_consumer(body):
        return []
    errs: list[str] = []
    for key in CITABILITY_KEYS:
        if key not in body and key not in detail:
            errs.append(f"names `{READER_MARKER}` (so it reads the uptake verdict) "
                        f"but never states the rule for `{key}`, one of the three "
                        f"keys that decide whether a table is citable")
        elif not (verdict_stated(body, key) or verdict_stated(detail, key)):
            errs.append(f"names `{key}` without a citability verdict within "
                        f"{VERDICT_WINDOW} characters of a mention of it, in the body "
                        f"or in a sibling file — a reader is told the key exists, not "
                        f"whether they may cite the number")
    corpus = body + "\n" + detail
    missing = [f for f in RE_BASE_FACTS if f not in corpus]
    if missing:
        errs.append("does not state the scoring-engine re-base (#1850): "
                    + ", ".join(repr(f) for f in missing)
                    + " must appear — which engine scores, the day that was "
                      "decided, and the day the slot it replaced was retired")
    return errs
