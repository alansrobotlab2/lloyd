#!/usr/bin/env python3
"""Step 3 of #2717 — the one part of ReHarness worth copying verbatim.

The paper's backward-revision loop has a guard that is not an opinion: a factual edit to a
past reflection must be supported by *that reflection's own source artifact*, so a fact
learned on day 10 cannot migrate into the day-1 record. Its other half — a model judging
whether a line helped or hurt a run — is a textual gradient, "English imitating
back-propagation", and its precision is capped by whatever backbone does it. This script
implements the deterministic half and refuses to implement the other one.

Given a proposed narrowing of a memory line and the artifact that line was written from,
it extracts every literal the proposal introduces and refuses the proposal if any of them
is absent from that source. It never rewrites the line, never touches a memory file, and
never asks the model anything.

    .venvs/lloyd/bin/python scripts/memory_source_grounding.py --check --proposals p.json

Scope, and the precedent for it. Backlog **#653** ("gate cited literals in tool arguments
against a session evidence index") is closed `rejected` on measured incidence: a wide,
always-on literal gate over every tool argument cost a lot and nothing had been observed
fabricating literals. That ruling is respected here rather than re-litigated — this check
runs over *proposed memory rewrites only*, which is exactly the surface #653's own closing
note says belongs here (its ruling was made against a 100% true positive class, which is a
statement about tool arguments, not about rewriting a durable memory line). Two
consequences are visible in the code and tested:

* A backticked identifier or an ``#NNNN`` backlog id inside a proposal is treated as a
  citation, not a copied fact. The check's job is to refuse an invented *claim*, and the
  thing that makes a narrowing dangerous is a path or a number asserted as fact.
* Words are not literals. This is not an entailment model: a limiting condition that says
  something the source does not say, in words, passes. The refusals are therefore a floor
  on fabrication, never a ceiling on it, and the batch report says so in its own text.

What is checked: paths, dates, ISO-ish timestamps, hex ids of 7+ characters, dotted
versions, bare numbers, code identifiers (something with an underscore or a hyphen that
reads like a name), and capitalised entity names.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

KIND_PATH = "path"
KIND_DATE = "date"
KIND_SHA = "hex-id"
KIND_VERSION = "version"
KIND_NUMBER = "number"
KIND_TIME = "time"
KIND_SYMBOL = "symbol"
KIND_ENTITY = "entity"
KINDS = (KIND_PATH, KIND_DATE, KIND_TIME, KIND_SHA, KIND_VERSION, KIND_NUMBER,
         KIND_SYMBOL, KIND_ENTITY)

#: The two classes an ungrounded literal belongs to, reported per refusal.
#: `introduced` is a literal `proposed_line` asserts that `current_line` does not contain:
#: the invented fact. `carried_over` is one the original line already asserted and the cited
#: artifact never supported. Both are refused; the class only says which.
CLASS_INTRODUCED = "introduced"
CLASS_CARRIED_OVER = "carried_over"

#: Reason names, because clause 5 asks for one per refusal and a reason that is free
#: prose cannot be counted.
REASON_UNGROUNDED = "ungrounded_literal"
REASON_NO_SOURCE = "no_cited_source"
REASON_SOURCE_UNREADABLE = "cited_source_unreadable"

#: A proposal with nothing in it that the check can read as a literal is not a proposal
#: that was verified: it is a proposal the check had no purchase on. Reported separately
#: so a 100% acceptance rate cannot be read as a clean batch.
NOTE_NO_LITERALS = "no_literals_to_check"

ARTIFACT_DIR_NAME = "memory-revision"
ARTIFACT_PREFIX = "source-grounding-"


def default_out_dir() -> Path:
    """The data root, never the code checkout — #2717's artifacts are run output.

    Lazy import, for the reason `scripts/memory_revision_candidates.py::default_out_dir`
    gives in full: a module whose job is to grade a batch of proposals must stay importable
    on a tree whose data root cannot be resolved.
    """
    from app.paths import DATA_ROOT
    return Path(DATA_ROOT) / "eval" / ARTIFACT_DIR_NAME

_NAME_CHARS = r"A-Za-z0-9_./~\-*"
_PATH_RE = re.compile(rf"(?:~//?|/|\./)[{_NAME_CHARS}]+(?:/[{_NAME_CHARS}]+)+")
#: A bare file reference — `install-services.sh`, `nvidia-power-limit.service` — counts as
#: a path literal even though it is relative, because that is how a memory line names a
#: unit or a script, and it is the shape a fabricated claim takes.
_FILE_RE = re.compile(r"\b[\w._-]+\.(?:sh|py|md|mdx|service|timer|json|ya?ml|db|log|"
                      r"conf|csv|txt|ts|tsx|js|sql|toml)\b")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_SHA_RE = re.compile(r"\b[0-9a-f]{7,40}\b")
_VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")
#: A number with no separators (thousands or decimal). A single digit is excluded on
#: purpose: it is almost always prose ("top 2", "GPU 1"), and refusing on it would spend
#: the check's whole credibility on noise.
_NUMBER_RE = re.compile(r"\b\d[\d,]*\.?\d*\b")
#: Something that reads like a code identifier rather than prose: an underscore, a digit,
#: or three or more hyphen-joined segments.
_SYMBOL_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[_-][A-Za-z0-9_]+)+\b")
#: A capitalised multi-word name (Mission Control, Inner Voice), or a single capitalised
#: word that is not sentence-initial and not a stopword — the shape an entity name takes
#: inside a memory bullet.
#: A multi-word capitalised name (Mission Control, Inner Voice). Position-independent: a
#: two-word capitalised sequence is a name even straight after a full stop, which is where
#: most memory bullets put it.
_ENTITY_PHRASE_RE = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b")
#: A single capitalised word mid-sentence ("... said Alan, ..."). Sentence-initial
#: capitalisation is indistinguishable from a name at a full stop, so this class is only
#: read where a sentence cannot be starting, and `code_shaped` marks it as the soft reading.
_ENTITY_WORD_RE = re.compile(r"(?<=[a-z,;:)][ ])([A-Z][a-z]{2,})\b")
_STOPWORDS = frozenset("""
    The This That These Those A An And But Or If When While Because Also It Its Itself
    They Them Their There Here What Which Who Whom How Why All Any Each Every Both Few
    More Most Other Some Such No Nor Not Only Own Same So Than Too Very Can Will Just
    Should Now Before After During Without Within Upon Into Over Under Again Further Then
    Once Here Above Below To Of In On At By For From With As Is Are Was Were Be Been Being
    Have Has Had Do Does Did May Might Must Shall Would Could Monday Tuesday Wednesday
    Thursday Friday Saturday Sunday January February March April May June July August
    September October November December Yes No OK
""".split())


class Literal(dict):
    """One extracted literal: its surface text, its kind, and how it is matched."""

    @property
    def text(self) -> str:
        return self["text"]

    @property
    def kind(self) -> str:
        return self["kind"]

    @property
    def norm(self) -> str:
        return self["norm"]


def _norm(kind: str, text: str) -> str:
    if kind == KIND_PATH:
        # A path matcher's character class includes the dot, so the sentence's own final
        # stop lands inside the match. `/etc/systemd/system.` must not be a different
        # literal from the `/etc/systemd/system` the source names.
        return text.rstrip("/").rstrip(".,;:")
    if kind == KIND_DATE:
        return text.replace("-", "")
    if kind == KIND_SHA:
        return text.lower()
    if kind == KIND_NUMBER:
        stripped = text.replace(",", "")
        return stripped[:-2] if stripped.endswith(".0") else stripped
    return text


def _protected_spans(text: str) -> list[tuple[int, int]]:
    """Ranges the check must not read as asserted facts: backticked code and `#NNNN` ids.

    #653's rejection was a cost ruling over every tool argument; the narrowing proposal is
    a different surface, but the same over-refusal risk applies — a proposal is *allowed*
    to cite the file it is about, and a check that refuses every citation would be switched
    off within a night.
    """
    spans = [(m.start(), m.end()) for m in re.finditer(r"`[^`\n]+`", text)]
    spans += [(m.start(), m.end()) for m in re.finditer(r"#\d{2,6}\b", text)]
    return spans


def _inside(pos: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


def extract_literals(text: str) -> list[Literal]:
    """Every literal a proposal asserts, in the order it appears, deduplicated by norm.

    Scans overlap by design: `2026-09-22` is a date and also a number and also a hyphen
    candidate, and the check only needs one of those readings to be grounded. What matters
    is that no scan is *narrower* than the class the item names — path, symbol, number,
    entity name, date.
    """
    spans = _protected_spans(text)
    out: list[Literal] = []
    seen: set[tuple[str, str, bool]] = set()
    for kind, pattern in ((KIND_PATH, _PATH_RE), (KIND_PATH, _FILE_RE),
                          (KIND_DATE, _DATE_RE), (KIND_TIME, _TIME_RE),
                          (KIND_SHA, _SHA_RE), (KIND_VERSION, _VERSION_RE),
                          (KIND_NUMBER, _NUMBER_RE), (KIND_SYMBOL, _SYMBOL_RE),
                          (KIND_ENTITY, _ENTITY_PHRASE_RE)):
        for match in pattern.finditer(text):
            if _inside(match.start(), spans):
                continue
            raw = match.group(0)
            if kind == KIND_PATH:
                # The path class allows dots, so a sentence's own final stop arrives inside
                # the match. It is trimmed here as well as in `_norm`, because the refusal
                # prints this text and must name the directory it is complaining about.
                raw = re.sub(r"[.,;:]+$", "", raw)
            if kind == KIND_ENTITY and raw in _STOPWORDS:
                # No sentence-initial test here: a TWO-word capitalised sequence is a name
                # wherever it sits, and "driver. Mission Control" is the position a memory
                # bullet actually puts it in. The position-sensitive single-word reading is
                # the separate scan below.
                continue
            if kind == KIND_SYMBOL and len(raw) < 4:
                continue
            #: A two-segment hyphenated word ("re-armed", "hand-placed") is prose, and
            #: refusing on it would refuse half of every honest narrowing. It is still
            #: reported, so the soft class cannot quietly become the whole check.
            code_shaped = not (kind == KIND_SYMBOL
                               and "_" not in raw and "_" not in raw
                               and not re.search(r"\d", raw) and raw.count("-") < 2)
            literal = Literal(text=raw, kind=kind, norm=_norm(kind, raw),
                              code_shaped=code_shaped)
            key = (kind, literal["norm"], code_shaped)
            if key in seen:
                continue
            seen.add(key)
            out.append(literal)

    for match in _ENTITY_WORD_RE.finditer(text):
        if _inside(match.start() + 1, spans):
            continue
        raw = match.group(1)
        if raw in _STOPWORDS:
            continue
        literal = Literal(text=raw, kind=KIND_ENTITY, norm=_norm(KIND_ENTITY, raw),
                          code_shaped=True)
        key = (literal["kind"], literal["norm"], True)
        if key not in seen:
            seen.add(key)
            out.append(literal)
    return out


def _grounded(literal: Literal, source_literals: Iterable[Literal], source_text: str) -> bool:
    """Is this literal in the cited source artifact?

    Paths and entity names may also match as plain substrings of the source: a note that
    says `~/lloyd/scripts/service_health_check.py` grounds a proposal that says
    `scripts/service_health_check.py`, and a note that mentions Mission Control grounds the
    bare name. Every other kind is matched on its normalised form, so a date does not
    ground a different date that happens to share digits.
    """
    if any(other.kind == literal.kind and other.norm == literal.norm
           for other in source_literals):
        return True
    if literal.kind in (KIND_PATH, KIND_ENTITY):
        needle = literal.norm.lower() if literal.kind == KIND_ENTITY else literal.norm
        return needle in source_text or needle.lower() in source_text.lower()
    return False


def check_proposal(proposal: Mapping[str, Any], source_text: Optional[str]) -> dict:
    """One verdict: accepted, or refused with a reason and the literals behind it.

    TWO classes of offender are refused, and neither one is a licence for the other:

    * **introduced** — a literal that `proposed_line` asserts and `current_line` does not
      contain, absent from the cited artifact. This is the invented fact the gate exists
      for, and the comparison with `current_line` only ever *labels* it: it is never a way
      past the grounding check. A proposal with no `current_line` at all therefore has every
      one of its literals in this class and is refused on the same footing.
    * **carried over** — a literal the original line already asserted and the artifact does
      not support. Grading only the introduced class would let a re-narrowing launder a
      drift that was already in the line, which is the paper's rule that a day-10 fact may
      not live in the day-1 record.

    So the rule is: every hard literal in the NEW text must be in the source artifact, and
    `current_line` decides only which of the two classes an offender is reported under.
    """
    proposed = proposal.get("proposed_line") or ""
    current = proposal.get("current_line") or ""
    cited = proposal.get("cited_source")
    result: dict[str, Any] = {
        "id": proposal.get("id") or proposal.get("entry") or "",
        "cited_source": cited,
        "accepted": False,
        "reason": None,
        "literals": [],
        "ungrounded": [],
        "added_tokens": max(0, len(proposed.split()) - len(current.split())),
    }
    if not cited:
        result["reason"] = REASON_NO_SOURCE
        result["detail"] = ("the proposal names no source artifact, so there is nothing a "
                            "factual edit could be grounded in")
        return result
    if source_text is None:
        result["reason"] = REASON_SOURCE_UNREADABLE
        result["detail"] = f"the cited source artifact could not be read: {cited}"
        return result

    source_literals = extract_literals(source_text)
    current_norms = {(lit.kind, lit.norm) for lit in extract_literals(current)}
    in_proposal = extract_literals(proposed)
    added = [lit for lit in in_proposal if (lit.kind, lit.norm) not in current_norms]
    # EVERY hard literal in the new text is graded — the comparison with `current_line`
    # CLASSIFIES an offender, it never excuses one. Grading only the literals the proposal
    # adds would let a re-narrowing launder a fact that never came from this artifact, so the
    # two classes below are both refused and the `if` at the bottom keys on their union.
    ungrounded_all = [lit for lit in in_proposal
                      if not _grounded(lit, source_literals, source_text)]
    # A two-segment hyphenated word is prose, and refusing on it would refuse half of every
    # honest narrowing; it is reported so the soft class cannot quietly become the check.
    ungrounded = [lit for lit in ungrounded_all if lit["code_shaped"]]
    soft = [lit for lit in ungrounded_all if not lit["code_shaped"]]
    # The two named classes. `introduced` is the proposed-only literal the gate was asked to
    # catch — asserted by `proposed_line`, absent from `current_line`, absent from the source.
    # `carried_over` is the already-drifted one. Named on the literal itself so a reader of
    # the artifact can tell an invented fact from an inherited one, and so neither class can
    # be mistaken for the whole check.
    for lit in ungrounded:
        lit["class"] = (CLASS_CARRIED_OVER if (lit.kind, lit.norm) in current_norms
                        else CLASS_INTRODUCED)
    introduced = [lit for lit in ungrounded if lit["class"] == CLASS_INTRODUCED]
    carried_over = [lit for lit in ungrounded if lit["class"] == CLASS_CARRIED_OVER]
    result["literals"] = [{"text": lit.text, "kind": lit.kind} for lit in added]
    result["ungrounded"] = [{"text": lit.text, "kind": lit.kind,
                             "class": lit["class"],
                             "added_by_this_proposal": lit["class"] == CLASS_INTRODUCED}
                            for lit in ungrounded]
    result["introduced"] = [{"text": lit.text, "kind": lit.kind} for lit in introduced]
    result["carried_over"] = [{"text": lit.text, "kind": lit.kind} for lit in carried_over]
    result["soft_ungrounded"] = [{"text": lit.text, "kind": lit.kind} for lit in soft]
    if introduced or carried_over:
        kinds = sorted({lit.kind for lit in ungrounded})
        result["reason"] = f"{REASON_UNGROUNDED}:{','.join(kinds)}"
        result["detail"] = (
            "asserts, absent from the cited source artifact: "
            + ", ".join(f"{lit.text!r} ({lit.kind}, {lit['class']})"
                        for lit in ungrounded))
        return result
    result["accepted"] = True
    if not added:
        result["note"] = NOTE_NO_LITERALS
    return result


def run_batch(proposals: Sequence[Mapping[str, Any]],
              source_reader) -> dict:
    """Grade a batch and report it. `source_reader(cited_source) -> str | None`.

    The report distinguishes the two ways a batch comes out with nothing accepted —
    `nothing_proposed` is true exactly when the input was empty. Without that flag, "the
    check refused everything" and "nobody proposed anything" are the same number, and the
    second is the one a nightly job will hit most often.
    """
    verdicts = [check_proposal(p, source_reader(p.get("cited_source")))
                for p in proposals]
    accepted = [v for v in verdicts if v["accepted"]]
    refused = [v for v in verdicts if not v["accepted"]]
    reasons: dict[str, int] = {}
    for verdict in refused:
        reasons[verdict["reason"]] = reasons.get(verdict["reason"], 0) + 1
    return {
        "schema": 1,
        "item": "2717",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checked": len(verdicts),
        "accepted": len(accepted),
        "refused": len(refused),
        "nothing_proposed": len(verdicts) == 0,
        "denominators": {"proposals": len(verdicts),
                         "distinct_refusal_reasons": len(reasons)},
        "refusal_reasons": reasons,
        "accepted_ids": [v["id"] for v in accepted],
        "no_literals_to_check": [v["id"] for v in accepted
                                 if v.get("note") == NOTE_NO_LITERALS],
        "verdicts": verdicts,
        "meaning": ("a refusal here means the proposal asserts something its cited source "
                    "artifact does not. An acceptance means no invented literal was found; "
                    "it does NOT mean the narrowing is correct — wording-level invention "
                    "is out of scope, and hand review (#2717 step 4) is the control for "
                    "that."),
    }


def file_source_reader(source_root: Path | str):
    root = Path(source_root).resolve()

    def read(cited: Optional[str]) -> Optional[str]:
        if not cited:
            return None
        candidate = Path(cited)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve()
        except OSError:
            return None
        if root not in resolved.parents and resolved != root:
            return None
        try:
            return resolved.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    return read


def write_report(report: Mapping[str, Any], out_dir: Path, as_of: str) -> Path:
    path = Path(out_dir) / f"{ARTIFACT_PREFIX}{as_of}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def render(report: Mapping[str, Any]) -> str:
    lines = [f"source-grounding: {report['accepted']} accepted, {report['refused']} refused "
             f"of {report['checked']} proposed"]
    if report["nothing_proposed"]:
        lines.append("  nothing was proposed: the batch was empty (this is not 0 accepted "
                     "from a non-empty batch)")
    for reason, count in sorted(report["refusal_reasons"].items()):
        lines.append(f"  refused x{count}: {reason}")
    for verdict in report["verdicts"]:
        if not verdict["accepted"]:
            lines.append(f"  [{verdict['id']}] {verdict['reason']}: {verdict['detail']}")
    if report["no_literals_to_check"]:
        lines.append(f"  accepted with no literal to check: "
                     f"{', '.join(report['no_literals_to_check'])}")
    return "\n".join(lines)


def load_proposals(path: Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("proposals", [])
    return list(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic source-grounding gate over proposed memory narrowings.")
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--proposals", required=True,
                        help="JSON array of {id, entry, current_line, proposed_line, "
                             "cited_source, triggering_runs}")
    parser.add_argument("--source-root", default=str(REPO),
                        help="root the cited source artifacts are resolved under "
                             "(the vault for real proposals)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    proposals = load_proposals(Path(args.proposals))
    report = run_batch(proposals, file_source_reader(args.source_root))
    print(render(report))
    if args.dry_run:
        print("  (dry run: nothing written)")
        return 0
    as_of = args.as_of or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"  wrote {write_report(report, Path(args.out_dir or default_out_dir()), as_of)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
