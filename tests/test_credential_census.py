"""#3846 clause 1: the census exists as a committed dated artifact, and the generator
that produced it can be re-run and re-checked.

Two separate obligations, and this file keeps them in separate nodes because they fail
for opposite reasons:

- **The committed artifact** —
  `eval/measurements/credentials-reachable-from-a-bash-child-2026-11-06.md` —
  is a record of a measurement of THIS box, taken before the change. Its nodes read the
  file's CONTENT and nothing else, so they pass in a promotion gate whose `$HOME` is an
  empty round home and would still pass if this box's vault were edited tomorrow. A node
  that mixed the file with a live `measure()` would go red in exactly one of the two
  places and read as a broken census in the other.
- **The generator** — `credential_store.measure/render_artifact/write_artifact` — is
  what makes the census re-derivable rather than a hand-written summary. Its nodes run
  against the `credential_vault` fixture's own six names under a monkeypatched `$HOME`,
  so the population they assert on belongs to the test.

The split matters because #3846's first attempt at this file did mix them: one node
asserted `env_file_name_count == 6` against this box's `~/lloyd/.env` and counted 0 in
the gate's round home, and the run died on the instrument.

Why the census keys on NAME SHAPE and not on the env file is itself pinned here (see
`test_a_secret_only_in_the_environment_is_still_vaulted`): what a Bash child inherits is
the process environment, and a guard defined over one file fails open in every process
where a secret arrived another way.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.harness import credential_store as CS

#: The committed census. Pinned by name: it is the file `write_artifact()` names for
#: 2026-11-06, and the item's acceptance says a dated artifact must exist.
# Stem taken from the GENERATOR's own constant, not retyped: a filename invented in
# prose cannot drift from the code that writes it if the test compares the two — the
# module docstring used to cite `credential-census-<date>.md`, a file no code wrote.
ARTIFACT = (ROOT / "eval" / "measurements"
            / f"{CS.ARTIFACT_STEM}-2026-11-06.md")

#: The three secrets the box holds, named here as literals rather than derived, because
#: this section is a RECORD of a measurement and a record that re-derives its own
#: population cannot show that the measurement changed.
BOX_VAULTED = ("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "ANTHROPIC_API_KEY")

#: The three names in the same file that are configuration. Named as literals for the
#: same reason as above: the census's judgement is that these are NOT credentials, which
#: is a claim about these names specifically. `*_CUSTOM_MODEL_OPTION*` is the model
#: selector and `ANTHROPIC_BASE_URL` is endpoint selection — placeholder-ing either is
#: an outage rather than a control.
BOX_NOT_VAULTED = ("ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_MODEL_OPTION",
                   "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME")


def _rows(body: str) -> "dict[str, list[str]]":
    """The census's table as name -> cells, backtick-tolerant.

    Read as a table and not as prose because a paragraph saying "three names, all
    consumed by the aggregator" would satisfy a substring check and still be the
    silence the clause forbids.
    """
    out: dict[str, list[str]] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        name = cells[0].strip("` ").strip()
        if name and name not in ("Env name", "") and not set(name) <= {"-"}:
            out[name] = cells
    return out


# ---------------------------------------------------------------------------
# The committed artifact: content only, so a clean-root gate run cannot move it
# ---------------------------------------------------------------------------

def test_the_census_artifact_exists_and_is_dated_in_its_name():
    """Step 1's deliverable is the file itself.

    #3846's acceptance is explicit — "a named residual list, not silence" — and a dated
    name is what makes it a measurement rather than a standing claim: the same path next
    month would be a different fact.
    """
    assert ARTIFACT.is_file(), (
        "step 1's deliverable is the dated artifact itself; #3846's acceptance says a "
        "named residual list, not silence")
    assert re.search(r"-\d{4}-\d{2}-\d{2}\.md$", ARTIFACT.name), ARTIFACT.name


def test_the_committed_census_has_a_row_per_secret_with_a_consumer_and_a_lane():
    """Each of the three box secrets appears as a table row, with the columns filled in.

    Content-only by construction: no `CS.measure()` here, so the node says the same
    thing in the gate's round home as on this box. The names are literals for the same
    reason — the row for `LIVEKIT_API_KEY` is a fact about this box on 2026-11-06.
    """
    rows = _rows(ARTIFACT.read_text(encoding="utf-8"))
    for name in BOX_VAULTED:
        cells = rows.get(name)
        assert cells, f"{name} has no census row — the list is prose again"
        assert len(cells) >= 5, f"{name} row is too thin to be a census: {cells}"
        consumer, lane = cells[3], cells[4]
        assert consumer and lane, f"{name} names no consumer / no lane judgement"
        # A census cell that says "TBD" is the silence clause forbids, wearing a table.
        assert not re.search(r"\b(TBD|TODO|unknown|n/?a)\b", consumer, re.I), (
            f"{name}'s consumer is an unfilled cell: {consumer!r}")


def test_the_committed_census_names_the_names_it_skipped_and_the_lane_it_refuted():
    """The judgement half: three names in the file are not credentials, and the config
    coordinates the item cited are refuted in print.

    The skipped three have to be named, because an artifact that vaulted all six would
    placeholder a base URL and a model id, and a reader needs to be able to contest the
    call rather than rediscover it.

    The config half is a refutation with its numbers in it. #3846's clause 1 says "the
    config.yaml literals at :339/:1040-1041"; at HEAD `21411475` the scan finds ZERO
    literal credentials — `:339` is an empty `token:` and :1040/:1041 are `${VAR}`
    indirections. An artifact that repeated the item would be a claim about a file that
    does not exist, so the node pins the words that carry the correction.
    """
    body = ARTIFACT.read_text(encoding="utf-8")
    for name in BOX_NOT_VAULTED:
        assert name in body, f"{name} was skipped without being named"
    for needle in (":339", ":1040", ":1041"):
        assert needle in body, (
            f"{needle} is a clause-1 coordinate and the artifact does not name it")
    # The measured count is 0, so the file has to REFUTE the coordinates rather than
    # list them: naming `:339` while agreeing two literals live there would be the
    # item's error copied into the record. Either phrasing is accepted — the generator's
    # ("there are **no literal credentials**") or this artifact's ("**It does not.**") —
    # because what is pinned is that the file takes a position, not which words it uses.
    assert re.search(r"\*\*it does not\*\*|no literal credentials"
                     r"|literal credentials in the file: \*\*0\*\*", body.lower()), (
        "the artifact names the coordinates but never states that no literal "
        "credentials are there")


def test_the_committed_census_carries_no_secret_value_and_no_unnameable_placeholder():
    """The invariant that lets a model-readable file describe secrets at all.

    Value-free by construction — names, lengths, destinations — and the check is
    structural rather than eyeballed: any placeholder-shaped token in the file must be
    one the vault scheme can account for, and no `lloyd/.env` value may be reproduced.
    Values are not available in a clean-root run, so what this node can and does pin
    without them is the second half, plus the first half for anything the file itself
    could not have invented.
    """
    body = ARTIFACT.read_text(encoding="utf-8")
    for token in CS.placeholder_tokens(body):
        assert CS.name_of_placeholder(token), (
            "a placeholder-shaped token the vault scheme cannot name is in the "
            "artifact, which means it is not the scheme's placeholder")
    # No field that could hold a value: the census reports LENGTH, never content.
    assert not re.search(r"(?im)^\s*(value|secret|token)\s*[:=]\s*\S", body), (
        "the artifact has a value-bearing assignment; it may name a length only")


# ---------------------------------------------------------------------------
# The generator: measured against a vault the test owns
# ---------------------------------------------------------------------------

def test_the_generator_counts_the_vault_it_is_given(credential_vault):
    """Six names in the vault, three vaulted, three skipped — and the counts come from
    the fixture's file, not from whatever the machine happens to hold.

    `env_file_name_count` is the figure that went red in the gate: measured off
    `Path.home()/"lloyd"/".env"`, which is this box's file locally (6) and does not
    exist in a round home (0). Under `credential_vault` it is 6 in both places, which is
    what makes the same assertion meaningful in each.
    """
    m = CS.measure()
    assert m["env_file_name_count"] == 6, (
        f"expected the fixture's six names, counted {m['env_file_name_count']} — the "
        "generator is reading a different file than the fixture wrote")
    assert sorted(m["env_file_secret_names"]) == [
        "LLOYD_TEST_ALPHA_SECRET", "LLOYD_TEST_BETA_KEY", "LLOYD_TEST_GAMMA_TOKEN"], (
        "the name-shape rule did not select exactly the fixture's three secrets")
    assert sorted(m["env_file_other_names"]) == [
        "LLOYD_TEST_DELTA_BASE_URL", "LLOYD_TEST_EPSILON_MODEL_OPTION",
        "LLOYD_TEST_ZETA_MODE"], "the skipped set is the fixture's configuration names"


def test_the_generated_artifact_names_every_entry_and_leaks_no_value(credential_vault,
                                                                    tmp_path):
    """Re-deriving the census reproduces the properties the committed one is graded on.

    This is the node that makes the artifact re-generatable rather than authored: it
    runs the generator into a scratch path under the fixture vault and grades the output
    with the same three tests the committed file is graded on — a row per name with a
    consumer and a lane, the skipped names present, and no value written out. The value
    check IS a hash comparison here, because in this node's world the fixture owns the
    values.
    """
    out = CS.write_artifact(tmp_path / "census-2099-01-01.md", today="2099-01-01")
    assert out.is_file() and out.read_text(encoding="utf-8").strip()
    body = out.read_text(encoding="utf-8")
    rows, m = _rows(body), CS.measure()
    # One row per VAULTED name; the skipped names are listed under their own heading as
    # prose, which is why only the vaulted population is counted against the table.
    assert len(rows) >= len(m["vaulted_names"]), (
        f"the generated table has {len(rows)} rows for {len(m['vaulted_names'])} "
        "vaulted names — every entry needs a row, not a mention")
    for name in m["vaulted_names"]:
        cells = rows.get(name)
        assert cells, f"the generator emitted no row for {name}"
        assert cells[3].strip() and cells[4].strip(), (
            f"{name} generated with an empty consumer or lane cell")
    for name in m["env_file_other_names"]:
        assert name in body, f"{name} skipped without being named"
    for name in m["vaulted_names"]:
        value = CS.real_value(name)
        assert value and value not in body, f"{name} was written out in full"
        assert hashlib.sha256(value.encode()).hexdigest()[:16] not in body, name


def test_an_unresolved_consumer_is_printed_as_one_instead_of_a_guess(credential_vault):
    """A name the census cannot place is labelled, not left blank or invented.

    The alternative to "we do not know" in a census is a plausible wrong consumer, and a
    reader cannot tell them apart. So the generator has one shared marker string, and
    every row it fills from the table either uses it or names a real lane — which means
    adding a credential here without adding a row cannot produce a clean-looking
    artifact.
    """
    rows = _rows(CS.render_artifact(today="2099-01-01"))
    marker = CS.UNRESOLVED_CONSUMER.split(" —")[0]
    for name, cells in rows.items():
        consumer = cells[3] if len(cells) > 3 else ""
        lane = cells[4] if len(cells) > 4 else ""
        if marker in consumer:
            assert marker in lane, (
                f"{name} was generated with an unresolved consumer and a RESOLVED lane "
                f"({lane!r}) — a half-answer is the failure this marker exists to avoid")
    # The marker is one string shared by generator and test, so rewording the prose
    # cannot silently turn an unresolved row into a resolved-looking one.
    assert marker == "UNRESOLVED", marker


def test_a_secret_only_in_the_environment_is_still_vaulted(monkeypatch):
    """The vault reads the ENVIRONMENT, not only the file — or it fails open.

    The census's headline is `~/lloyd/.env`, and a guard defined over that file alone
    would placeholder nothing in a process where the secret arrived another way (a
    supervisor `environment=` line, a launcher export, or a root change that made the
    file unreadable after `app/config.py` had already loaded it). What a child inherits
    is the environment, so that is what the vault reads — the clause's "suffix-shaped
    only, never substring" rule is what keeps that from swallowing an unrelated name.
    """
    monkeypatch.setenv("LLOYD_TEST_PHANTOM_TOKEN", "a-value-no-file-holds")
    assert "LLOYD_TEST_PHANTOM_TOKEN" in CS.vaulted_names(), (
        "a secret-shaped name in the process environment was not vaulted, so no "
        "placeholder would ever be applied to it")
    out = CS.vaulted_env({"LLOYD_TEST_PHANTOM_TOKEN": "a-value-no-file-holds"})
    assert CS.is_placeholder(out["LLOYD_TEST_PHANTOM_TOKEN"])
    # The shape rule is suffix-shaped, so a name that merely contains a secret word
    # stays put: a substring rule would placeholder `TOKEN_COUNT` and every *_STATS.
    monkeypatch.setenv("LLOYD_TEST_TOKEN_COUNT", "17")
    monkeypatch.setenv("LLOYD_TEST_DELTA_BASE_URL", "https://example/v1")
    assert "LLOYD_TEST_TOKEN_COUNT" not in CS.vaulted_names()
    assert "LLOYD_TEST_DELTA_BASE_URL" not in CS.vaulted_names()


def test_the_registry_holds_no_value_and_is_itself_read_protected(credential_vault):
    """Two claims the design rests on, checked on a registry this node seeded.

    The registry is safe to describe in a model-readable census because it holds no
    value, and it is read-denied because it is a protected read root. Both are pinned
    together: if a value ever lands in a registry entry, the read-deny is the only thing
    standing between it and a transcript.

    Seeded here rather than read from disk because in a clean root the file does not
    exist and a loop over `{}` passes for the wrong reason.
    """
    CS.store_path().parent.mkdir(parents=True, exist_ok=True)
    CS.store_path().write_text(
        '{"_comment": "fixture", "credentials": {"LLOYD_TEST_ALPHA_SECRET": '
        '{"destinations": ["https://api.example.com"], "lanes": ["bash", "http"], '
        '"note": "fixture"}}}', encoding="utf-8")
    registry = CS.registry()
    assert list(registry) == ["LLOYD_TEST_ALPHA_SECRET"], registry
    for name, entry in registry.items():
        assert "value" not in entry and "secret" not in entry, name
    m = CS.measure()
    assert m["registry_read_denied"], (
        "the registry is readable through the agent's own tools, so its contents — "
        "which secrets exist and where each may be spent — are exfiltrable")
