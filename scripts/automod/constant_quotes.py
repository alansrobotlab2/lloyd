"""#2317: does an autonomy task description still quote a constant's OLD value?

`autonomy/79-retention-sweep.md`'s description states what `scripts/groundskeeper/
retention-sweep.py` bounds, store by store, and it states it as numbers:
`LEDGER_ARCHIVE_AGE_DAYS (14)`. Those numbers are copied by hand out of the script,
and nothing in the tree compared them. So when `01dea8bc` moved that constant from
30 to 14 the prose kept saying 30 until vault commit `dfe2207d` fixed it under
#2098 — and the same description-vs-script drift on that one file had already been
fixed twice before that (#1573, #1734). Three repairs of one instance, no guard.

`vault_guards.agreement()` could not have caught it. It is a pytest-delta probe: it
refuses only a node that passes against the pre-land vault and fails against the
proposed one, so it can see a prose↔code disagreement only if some node already
reads BOTH sides. Nothing did — the one place that parses that file at all
(`tests/test_dashboard_cold_render.py`) reads only `frequency` — which is exactly
the hole this module fills: it reads both sides itself.

**What it resolves.** A description↔constant pair is resolved when a name that
`tree_constants()` scraped (`^NAME = <int>`) appears in the description with a
number bound to it. Three spellings are in the corpus that motivated this, all three
are read, and `mismatches()` says which one it used nowhere because the answer is
the same tuple either way:

* `NAME (N)` — `WORKTREE_DIR_MAX_AGE_DAYS (7)`, including the code-span form
  `` `vault_writer.RELEVANCE_FLOOR` (4) ``, which names the module inside the span.
* `NAME (N, prose…)` — `VOICE_TURNS_MAX_AGE_DAYS (90, deliberately not the 30 the
  file stores use, because …)`: only the FIRST number in the brackets is the
  constant's value, the rest of that parenthesis is the argument for 90.
* `Nd … (NAME` — `deleted >30d on mtime (GROUNDSKEEPER_QUEUE_MAX_AGE_DAYS)`, and
  `… 30d for a background session (platform autonomy or worker,
  BACKGROUND_SESSION_ARCHIVE_AGE_DAYS, the window that covers nearly every file …)`:
  the number comes first and the name is named inside the following brackets.

**What it does not resolve, and says so.** A constant quoted as bare prose with no
number adjacent — #79 states `older than PROVENANCE_ARCHIVE_AGE_DAYS days` — binds
no pair, because there is no number to disagree with. So does a corpus that names no
tree constant at all. That is why a zero pair count is never agreement: a check that
resolved nothing and a check that resolved everything and found it correct both
print "0 mismatches", and only the denominator tells them apart. #1691 and the
zero-denominator rule are the same lesson, and `Report.instrument_failed` is this
module's half of it. `mismatches()` therefore returns a count beside the tuples, and
every caller here carries it.

**What it is not.** Not a claim registry: it verifies nothing about a description
that does not name a scraped constant, and it never edits prose. It is not
surface-scoped either — the two write edges that call it (`vault_guards
.description_constant_errors` refuses a vault land, `review` names an advisory
finding when code moves the constant) share this one resolver, so the two directions
of the same drift cannot drift apart from each other.
"""
from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

#: A constant this module can resolve: a module-level ALL-CAPS name bound to a bare
#: integer. This is the only spelling whose value a description can quote, because
#: it is the only one with the value sitting on the same line as the name.
TREE_ASSIGN_RX = re.compile(r"^(?P<name>[A-Z][A-Z0-9_]*) = (?P<value>-?\d+)$", re.M)

#: The same name's spelling inside prose. `[A-Z][A-Z0-9_]*` cannot start mid-word
#: (`\b` would also stop at `_`, which is a word character, so a shorter name never
#: matches inside a longer one: `SESSION_ARCHIVE_AGE_DAYS` does not match inside
#: `BACKGROUND_SESSION_ARCHIVE_AGE_DAYS`).
_NAME_TOKEN_RX = re.compile(r"[A-Z][A-Z0-9_]*")

#: Spellings 1 and 2: the number sits in the brackets straight after the name.
#: Anchored at the name's end, so `NAME days` and `NAME, which is 30` bind nothing.
#: The optional backtick or quote is the name's own code span closing before the
#: brackets open — `autonomy/30-intelligence-pipeline-scan-score.md` writes
#: `` `vault_writer.RELEVANCE_FLOOR` (4) `` and naming the module in the span is how
#: that file says which module's constant it means. A grammar that stopped at the
#: name's last letter would resolve 7 of the 8 pairs in this vault and never say so.
_AFTER_NAME_RX = re.compile(r"[`\"]?\s*\(\s*(?P<n>\d+)")

#: Spelling 3's number: `30d`, `>30d`. `(?<![\\w.])` keeps `1.30d` and `X30d` from
#: reading as a day count and `\\b` keeps `30do` from it.
_DAYS_BEFORE_RX = re.compile(r"(?<![\w.])(?P<n>\d+)d\b")

#: How far back spelling 3's number may sit from the brackets that name the
#: constant. 80 is the real case: `30d for a background session (platform autonomy
#: or worker, BACKGROUND_SESSION_ARCHIVE_AGE_DAYS` puts the name 56 characters on.
#: The window exists so a stray `90d` three sentences back cannot be read as the
#: value of the next constant the prose happens to name.
_MAX_LOOKBEHIND = 80

#: Between spelling 3's number and the name, none of these may appear: a closing
#: bracket (the number belongs to a group that already closed), a semicolon (the
#: next store's entry), or another resolvable constant's name (the number is
#: that one's, not this one's).
_S3_BREAKS = (")", ";")

#: Directories a non-git tree scrape never walks into. `.venvs` alone is tens of
#: thousands of third-party files, and a vendored `MAX_RETRIES = 3` would join the
#: map on its way past.
_SKIP_DIRS = frozenset({".git", ".venv", ".venvs", "node_modules", "__pycache__",
                        ".mypy_cache", ".pytest_cache", ".ruff_cache",
                        "dist", "build"})

#: What a result reports when it resolved nothing at all. Spelled once so a
#: refusal, a finding and a witness line cannot each invent their own wording for
#: "this check did not measure anything".
INSTRUMENT_FAILURE = "instrument failure: 0 description/constant pairs resolved"


@dataclass(frozen=True)
class Report(Sequence):
    """The mismatches, and the denominator that makes them mean something.

    A sequence of `(name, quoted, actual)` so `list(rep)`, `len(rep)` and `rep[0]`
    all read as the tuples themselves — the clause this exists for is about the
    tuples — while `.resolved` answers the separate question of how many pairs the
    resolver was able to compare at all.
    """

    found: tuple[tuple[str, int, int], ...] = ()
    resolved: int = 0
    names: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.found)

    def __getitem__(self, i):
        return self.found[i]

    @property
    def instrument_failed(self) -> bool:
        """True when nothing at all was resolved: no pair, so no agreement either.

        A description that names no tree constant yields zero mismatches exactly
        like a description whose every quote is correct does. Only a caller that
        reads this (or `.resolved`) can tell the two apart.
        """
        return self.resolved == 0

    def summary(self) -> str:
        """One line with BOTH numbers in it, for a witness print or a refusal."""
        if self.instrument_failed:
            return INSTRUMENT_FAILURE
        return (f"{len(self.found)} mismatch(es) over {self.resolved} resolved "
                f"pair(s){': ' + ', '.join(_text(m) for m in self.found) if self.found else ''}")


def _text(m: tuple[str, int, int]) -> str:
    """A mismatch as one readable clause: the name and both numbers."""
    name, quoted, actual = m
    return f"{name} quoted {quoted}, tree says {actual}"


def _paren_spans(text: str) -> list[tuple[int, int]]:
    """`(open_index, close_index)` for every bracket pair, nested ones included.

    An unbalanced open bracket runs to the end of the text: prose that opens a
    parenthesis and never closes it still brackets what it names.
    """
    spans: list[tuple[int, int]] = []
    open_at: list[int] = []
    for i, ch in enumerate(text):
        if ch == "(":
            open_at.append(i)
        elif ch == ")" and open_at:
            spans.append((open_at.pop(), i))
    while open_at:
        spans.append((open_at.pop(), len(text)))
    return spans


def _inside_a_span(spans: list[tuple[int, int]], at: int) -> bool:
    return any(start < at < end for start, end in spans)


def _bind(text: str, start: int, end: int, spans: list[tuple[int, int]],
          tree: Mapping[str, int]) -> int | None:
    """The number this occurrence of `text[start:end]` (a known constant's name)
    states, or None when the prose binds no number to it here.

    `start`/`end` are the name's own offsets. Spellings are tried in the order the
    corpus states them; the first that matches wins, so a name written
    `NAME (30)` never also claims the `Nd` sitting before it.
    """
    after = _AFTER_NAME_RX.match(text, end)
    if after:
        return int(after.group("n"))
    if not _inside_a_span(spans, start):
        return None
    window = text[max(0, start - _MAX_LOOKBEHIND):start]
    for m in reversed(list(_DAYS_BEFORE_RX.finditer(window))):
        between = window[m.end():]
        if any(b in between for b in _S3_BREAKS):
            continue
        if any(tree.get(t) is not None for t in _NAME_TOKEN_RX.findall(between)):
            continue  # the number belongs to the nearer name
        return int(m.group("n"))
    return None


def mismatches(description_text: str,
             tree_consts: Mapping[str, int]) -> Report:
    """Compare every constant quote in `description_text` against `tree_consts`.

    Pure: it reads nothing from disk, so a caller decides what counts as the
    description (both edges pass the YAML-parsed front-matter `description` and
    nothing else) and a test can hand it a string. `tree_consts` is name ->
    value, the
    shape `tree_constants()` returns or a test's own one-entry dict.

    One entry per distinct `(name, quoted)` pair whose number differs from
    `tree_consts[name]`, in the order the description states them; the same correct quote
    twice is one resolved pair, not two. `.resolved` counts every pair that was
    compared, right or wrong, and a caller that gets 0 has not been told the prose
    agrees — it has been told nothing was checked.
    """
    text = description_text or ""
    if not text or not tree_consts:
        return Report()
    spans = _paren_spans(text)
    pairs: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for m in _NAME_TOKEN_RX.finditer(text):
        name = m.group()
        if tree_consts.get(name) is None:
            continue
        quoted = _bind(text, m.start(), m.end(), spans, tree_consts)
        if quoted is None:
            continue
        key = (name, quoted)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    found = tuple((name, quoted, int(tree_consts[name]))
                  for name, quoted in pairs
                  if quoted != int(tree_consts[name]))
    return Report(found=found, resolved=len(pairs),
                  names=tuple(dict.fromkeys(name for name, _ in pairs)))


def front_matter_description(text: str) -> str:
    """The YAML-parsed front-matter `description` of a task file's text, or "".

    The scheduler's own reader: `parse_frontmatter_text` with `AUTONOMY_TASK_FIELDS`
    is exactly what `app.autonomy._parse_task_file` (`app/autonomy.py:129`) calls, and
    that is the dict `_build_task_prompt` takes `description` out of — so this is the
    text the task actually runs with, graduated recovery included, and nothing else.
    Body prose and the Activity Log are not inputs, which is what lets a land that
    appends an activity note pass a check that a stale number in the description must
    not; `status:` and the rest of the run-state fields are not inputs either, so no
    engine state can trigger the rail.

    Returns "" for a file with no front matter, no `description`, or one that is only
    whitespace. A file whose YAML cannot be parsed gets whatever the loader's regex
    fallback extracted, because that is the text the scheduler would run with — and
    `vault_round.frontmatter_error` is the rail that refuses such a file, not this
    check: two rails refusing one unreadable file is how a check becomes the thing
    authors route around. "" resolves zero pairs, which `Report.instrument_failed`
    reports as nothing-was-checked rather than as agreement.
    """
    if not text.startswith("---"):
        return ""
    lines = text.splitlines()
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return ""
    try:
        from agent_mcp._shared import AUTONOMY_TASK_FIELDS, parse_frontmatter_text
        fm = parse_frontmatter_text("\n".join(lines[1:end]),
                                    fallback_fields=AUTONOMY_TASK_FIELDS,
                                    log_label="constant_quotes:#2317")
    except Exception:  # noqa: BLE001 - an unreadable file is not a disagreement
        return ""
    if not isinstance(fm, dict):
        return ""
    return str(fm.get("description") or "").strip()


def task_descriptions(vault_dir: Path) -> dict[str, str]:
    """`{vault-relative path: description}` for every `autonomy/*.md` task file.

    A directory of task files is the corpus both write edges police, and the labels
    are the paths a refusal or a finding can name. A file that cannot be read is
    skipped rather than reported as an empty description: an unreadable task file is
    the landing route's `frontmatter_error` to refuse, and calling it "a description
    that quotes nothing" here would be a false statement about a file this module
    never read.
    """
    out: dict[str, str] = {}
    for f in sorted(Path(vault_dir).glob("*.md")):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        out[f"autonomy/{f.name}"] = front_matter_description(text)
    return out


def tree_constants(root: Path) -> dict[str, int]:
    """Every `^NAME = <int>` in `root`'s Python files, as name -> value.

    A git tree is scraped with `git grep`, which sees tracked files only; anything
    else (a test's scratch directory) is walked, skipping `_SKIP_DIRS`. Both
    branches end in the same regex over the same lines, so the two cannot answer
    differently about one file.

    A name bound to MORE THAN ONE value in the tree is dropped rather than guessed:
    with `DEFAULT_MAX_TURNS = 7` in one module and `= 20` in another, which one a
    description means is a question about the task's script, and refusing on a guess
    would refuse a correct description. Dropping it costs a resolved pair, which the
    denominator reports; it never manufactures a mismatch.
    """
    root = Path(root)
    rows: dict[str, set[int]] = {}
    if (root / ".git").exists():
        r = subprocess.run(["git", "-C", str(root), "grep", "-h", "-E",
                            r"^[A-Z][A-Z0-9_]* = -?[0-9]+$", "--", "*.py"],
                           capture_output=True, text=True, timeout=120)
        lines = r.stdout
    else:
        chunks = []
        for f in sorted(root.rglob("*.py")):
            if _SKIP_DIRS & set(f.relative_to(root).parts):
                continue
            try:
                chunks.append(f.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
        lines = "\n".join(chunks)
    for m in TREE_ASSIGN_RX.finditer(lines):
        rows.setdefault(m.group("name"), set()).add(int(m.group("value")))
    return {name: next(iter(values))
            for name, values in rows.items() if len(values) == 1}


def scan(descriptions: Mapping[str, str],
         tree_consts: Mapping[str, int]) -> dict:
    """`mismatches()` over a whole corpus, with one denominator for all of it.

    `descriptions` maps a label (a vault-relative path is what both callers pass) to
    the parsed front-matter description. Returns `{"mismatched": [(label, name,
    quoted, actual)], "pairs": int, "silent": [label for a file that resolved
    nothing], "report": str}`.

    `pairs` is the corpus total, which is the number a caller must read before
    believing `mismatched == []`: a vault in which no task names a constant would
    otherwise report a clean sweep forever. The per-file `silent` list is the same
    fact one level down — those files contributed nothing to the denominator — and
    is NOT a failure of its own, since most autonomy tasks legitimately state no
    constant and a land that touches one of them must not be refused for it.
    """
    mismatched: list[tuple[str, str, int, int]] = []
    silent: list[str] = []
    pairs = 0
    for label in sorted(descriptions):
        rep = mismatches(descriptions[label], tree_consts)
        pairs += rep.resolved
        if rep.instrument_failed:
            silent.append(label)
        mismatched.extend((label,) + m for m in rep.found)
    return {"mismatched": mismatched, "pairs": pairs, "silent": silent,
            "report": (INSTRUMENT_FAILURE if pairs == 0 else
                       f"{len(mismatched)} mismatch(es) over {pairs} resolved pair(s) "
                       f"across {len(descriptions)} description(s)")}
