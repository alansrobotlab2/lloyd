"""A reference the Bash tool's own environment does not define expands to nothing —
and in a measurement position that empty string is a different measurement, not a
failed one. (#3630 braced, #3707 the name-lengthening unbraced arm)

Two grammars, because there are two distinct mechanisms and only one of them is
this tool's blanking pass. `${NAME}` is rewritten before bash ever sees the text;
`$NAME` is expanded by bash itself, so it is refused only when bash's own greedy
`[A-Za-z0-9_]` name scan makes it a name that does not exist — a trailing `_`,
`$d_` for `d` — and only when the command never binds that name.

Something upstream of every Bash spawn rewrites `${NAME}` for a name that is not
in the tool's environment to the empty string, and it does so inside single
quotes, inside double quotes and inside a loop body alike — all measured
2026-10-31: `echo 'single-quoted-[${zzznotset4}]'` printed `single-quoted-[]`,
and `for x in one two; do echo "[${x}]"; done` printed `[]` twice and named `x`
in the one diagnostic line it emits, ahead of the output, naming bare names with
no position. The emitter is not in this repo (0 hits repo-wide, 0 in
`strings /usr/bin/bash` for its own message, none in `~/.openclaw`), so the only
lever available is to refuse the command upstream of the rewrite. That position
is not a weakness: the pass sits at or below `agent_mcp/builtin_bash._bash`
(`scripts/bash_tool_prepass_probe.py`), while the gate this lane joins runs at
dispatch on the raw `arguments["command"]` (`agent_mcp/main.py`), so the
reference is still spelled out here and no spelling can hide it from this check.

Why warn-and-continue is not enough. The diagnostic exists and fires, but it is
advisory and the command runs anyway, so the number it produced gets reported.
Three lanes did that inside 24 h: task #38's three per-day run censuses built
from `run_*_${d}_*` printed `n=0 / n=0 / n=0` beside literal-date arms printing
`40 / 40 / 40`; task #39's `${n}`/`${c}` loop echoed an 84-line ledger and burned
~15 KB of context; task #40's `printf '%-62s ' "${f%%.md}"` printed seven empty
labels. The shape that reaches a report as a measurement is the glob: measured
2026-10-31, `echo "n=$(ls -1 /tmp/*_${zzznotset}_* 2>/dev/null | wc -l)"`
returned **`n=10`** — the pass had rewritten the pattern to `/tmp/*__*`, which
matches ten unrelated files. The same collapse in `find` position: `find /tmp
-maxdepth 1 -name "*_${zzznotset5}*"` returned 202 matches. A count of something
else is a plausible non-zero answer to a question nobody asked, and unlike an
`n` that is zero three times running, nothing downstream can see it.

This lane therefore refuses only where an empty expansion changes what is
measured, and leaves the rest exactly as it is:

* **A `find` predicate argument** — `-name`, `-iname`, `-path`, `-ipath`,
  `-lname`, `-wholename` and the case-insensitive variants — where the tool does
  the matching, quoted or not.
* **A word carrying `*` or `?`**, which is pathname-expanded or matched, so the
  pattern the command finally uses is the one with the hole in it. `[` is
  deliberately not in that set: `echo "[${zzznotset}]"` is the item's own
  leave-alone case, and bracket classes sit inside far more non-glob text than
  `*` and `?` do.
* **A path-shaped operand** — a word beginning `/`, `~`, `./`, `../` or `${` that
  contains a `/`. A bare `${UNSET}` with no path shape is left alone, which is
  what "keep today's behaviour for text output" requires: `echo "${UNSET}"` is a
  report of nothing, not a measurement of the wrong thing.

Two forms are outside the population and stay runnable. `${NAME:-fallback}` and
its `:=`/`:+`/`:?` siblings are how you say what you mean, and this lane exists
to push commands toward them. A name the tool's own environment defines —
`${HOME}/lloyd` — is resolved silently by the pass and behaves normally, so
judging it would refuse the common case. And a name the shell gives a value to
even when its parent has none is judged by that shell, not by this process:
:data:`_SHELL_OWN_NAMES` below is the measured list. A loop-control variable is
in the population on purpose — `for d in a b; do ls -d /tmp/*_${d}_*; done` is
rewritten before bash parses the loop, so the command that reaches the shell is
`ls -d /tmp/*__*` twice, and the refused command could not have done the thing
its text describes.

Returns a reason naming the reference and the word it sat in, or None. It never
reads the filesystem and never runs anything.
"""

from __future__ import annotations

import os
import re

#: `${NAME}` and nothing else: no `:` operator, no `#`/`%` trim, no `!` indirect,
#: no `${#NAME}`, no positional `${1}`. Those spellings either carry their own
#: fallback or are not a plain name, and this lane is about the plain name that
#: silently becomes nothing.
_BRACED_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: A reference in the shell's own form: `$NAME`, no braces. Bash reads a name as the
#: LONGEST run of `[A-Za-z0-9_]` after the `$`, so `$d_` is the name `d_` and not `d`
#: followed by a literal underscore — which is the difference between #3707's
#: `find . -name "run_*$d_*.md"` returning the day's 38 files and returning the whole
#: corpus (measured 2026-11-02: 1089 of 1089, then 1129 of 1089 for the loop-variable
#: spelling). A braced reference has an explicit end, so `_BRACED_REF` cannot express
#: the mistake: this is a second grammar, not a wider version of the first, and a patch
#: that widened only `_BRACED_REF` would scan tokens that never match.
_BARE_REF = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")

#: A bare reference whose scanned name ends in `_`, i.e. the only bare spelling this lane
#: refuses. It exists so the cheap exit in `check_empty_expansion_measurement` can stay
#: cheap: without it, widening that test from `${` to `$` would put the shared shell parse
#: on every command that mentions any variable, for an arm that is rare.
_LENGTHENING_REF = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*_)(?![A-Za-z0-9_])")

#: Names a command binds for itself. The braced arm does not consult these — the
#: upstream pass consumes `${NAME}` whatever the command says, which is #3630's whole
#: point — but a BARE reference is expanded by bash at execution time, so a name the
#: command assigns really does have a value: `d_=20261031; find . -name "run_*$d_*.md"`
#: prints the day's files (measured), and refusing it would be the false block this lane
#: exists not to be. A `for`/`select` control variable and a `read` target bind the same
#: way; `for x in …; do … "$x_*"` still binds `x`, not `x_`, and stays refused.
_NAME_ASSIGNMENT = re.compile(
    r"(?:^|[;&|\n(]|\s)(?:export\s+|local\s+|declare\s+(?:-\S+\s+)*|readonly\s+)?"
    r"([A-Za-z_][A-Za-z0-9_]*)=")
_NAME_BINDER = re.compile(
    r"(?:^|[;&|\n(]|\bdo\s+)(?:for|select)\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\b"
    r"|(?:^|[;&|\n(]|\bdo\s+)read\s+(?:-\S+\s+)*([A-Za-z_][A-Za-z0-9_]*)\b")

#: Names bash itself gives a value, measured 2026-10-31 by running each through
#: `bash -c 'printf "[%s]" "${NAME}"'` and comparing against this process's own
#: `os.environ`. On this box the pass blanks them too (`echo "[${RANDOM}]"`
#: printed `[]`), so leaving them out of the population costs no protection
#: today; it keeps the refusal honest about what this process can actually see,
#: and correct if a caller ever reaches the gate without the pass in front of it.
#: `IFS`, `COLUMNS`, `LINES`, `FUNCNAME`, `PIPESTATUS` and `BASH_SOURCE` are NOT
#: here: measured quoted, they print `[]` from a fresh `bash -c` as well, so they
#: really are an empty string at the moment the command is judged.
_SHELL_OWN_NAMES = frozenset({
    "RANDOM", "SECONDS", "LINENO", "OPTIND", "OPTERR", "BASHPID", "PPID",
    "UID", "EUID", "HOSTNAME", "BASH_VERSION", "OSTYPE", "MACHTYPE",
    "DIRSTACK", "GROUPS", "SHELLOPTS", "BASHOPTS",
    # Bash's last-argument parameter, and the ONLY name in this set that matters to the
    # bare grammar: its name is a single `_`, so it ENDS in `_` and a name-lengthening
    # rule that exempted nothing would refuse `$_` in measurement position — clause 4 of
    # #3707. Not a guess about it having a value: `bash -c 'printf "[%s]" "$_"'` prints
    # `[/usr/bin/bash]` on this box (the shell's own argv0 is the previous "argument"),
    # and after one command it is that command's last argument. It was inert under the
    # braced grammar because `${_}` is a spelling nobody writes; under the bare one,
    # `$_` is ordinary.
    "_",
})

#: The `find` tests whose argument is a pattern the tool matches, not the shell.
_FIND_PREDICATES = frozenset({"-name", "-iname", "-path", "-ipath", "-lname",
                              "-ilname", "-wholename", "-iwholename"})

#: A word carrying one of these is expanded against the filesystem (`ls`, `cp`)
#: or matched inside the program (`find`, `rsync --include`), so the empty string
#: changes the set the command acts on.
_GLOB_METACHARS = "*?"

#: What a path argument looks like from outside: a leading path sigil and a
#: separator somewhere in the word. `"s/a/${X}/"` has a separator but no sigil,
#: and `echo "n/b ${X}"` is prose.
_PATH_SIGILS = ("/", "~", "./", "../", "${")


def _bound_names(command: str) -> frozenset[str]:
    """Names this command binds for itself: `NAME=`, `for NAME in`, `read NAME`."""
    out = set(_NAME_ASSIGNMENT.findall(command))
    for pair in _NAME_BINDER.findall(command):
        out.update(n for n in pair if n)
    return frozenset(out)


def _undefined_refs(word: str, bound: frozenset[str]) -> list[tuple[str, bool]]:
    """References in `word` that expand to the empty string, as `(name, braced)`.

    Two grammars, two populations, deliberately unequal:

    * **braced** — every `${NAME}` this process's environment does not define and no
      shell built-in owns. `bound` is not consulted: the upstream pass consumes the
      reference before the shell parses, so `d=20261031; echo "[${d}]"` prints `[]`
      (measured) and the empty expansion is real however the command reads.
    * **bare** — only `$NAME` references whose SCANNED name ends in `_`. Refusing every
      bare name undefined in `os.environ` would deny commands that measure the right
      thing: bash resolves `find . -name "run_*$d*.md"` against a `d` the same call
      assigns (measured 38 = the real day), so that arm stays warn-and-run. The
      name-lengthening arm is the one bash cannot rescue, because `$d_` is a different
      name from the `d` the command assigned — and it is exempted when the command
      assigns the lengthened name itself, which bash does resolve (`d_=x; …"$d_*"` runs).

    `_` is in `_SHELL_OWN_NAMES` and so never reaches here: its name ends in `_`, so a
    trailing-underscore rule that exempted nothing would start refusing bash's own
    last-argument parameter in measurement position.
    """
    out: list[tuple[str, bool]] = [
        (n, True) for n in _BRACED_REF.findall(word)
        if n not in os.environ and n not in _SHELL_OWN_NAMES]
    seen = {n for n, _ in out}
    for n in _BARE_REF.findall(word):
        if (n.endswith("_") and n not in seen and n not in bound
                and n not in os.environ and n not in _SHELL_OWN_NAMES):
            out.append((n, False))
    return out


def _measurement_position(argv: list[str], i: int) -> str | None:
    """What makes `argv[i]` a measurement, or None if it is just text."""
    word = argv[i]
    if i and argv[i - 1] in _FIND_PREDICATES:
        return f"the `find {argv[i - 1]}` pattern"
    if any(c in word for c in _GLOB_METACHARS):
        return "the glob word"
    if "/" in word and word.startswith(_PATH_SIGILS):
        return "the path argument"
    return None


def check_empty_expansion_measurement(command: str) -> str | None:
    """Refuse a command whose empty `${NAME}` lands in a measurement position.

    Segmented with the shared shell parse (`protected_paths._tokens`/`_segments`),
    so a command substitution, a pipeline or `2>/dev/null` cannot smuggle a word
    past it, and read after quoting is stripped — which is right, because the pass
    rewrites references inside quotes too and `find -name "*pat*"` matches exactly
    as wrongly quoted as unquoted.
    """
    # The cheap exit has to cover BOTH grammars. It used to read `"${" not in command`,
    # which is the braced-only assumption sitting in front of the scanner: #3707's
    # reproduction contains no `${`, so it returned ALLOW here and never reached a token.
    # Scanning every command that merely contains a `$` would put a line parse on nearly
    # every Bash call for an arm that is rare, so the second test is the one thing the
    # bare grammar can be refused for — a name whose scanned form ends in `_`.
    if "${" not in command and not _LENGTHENING_REF.search(command):
        return None
    from app.harness.protected_paths import _segments, _tokens

    bound = _bound_names(command)
    for argv in _segments(_tokens(command)):
        for i, word in enumerate(argv):
            refs = _undefined_refs(word, bound)
            if not refs:
                continue
            position = _measurement_position(argv, i)
            if position is None:
                continue
            name, braced = refs[0]
            # The reference as the caller wrote it. `d` and `d_` are different names to
            # bash and the two spellings have different fixes, so the message that names
            # `${d}` for a command that wrote `$d_` sends the operator to the wrong one.
            spelling = "${%s}" % name if braced else "$%s" % name
            # Quote what the caller typed, not the token this check matched: the
            # shared parse rewrites `$(` to a command boundary, so inside a
            # `$( … )` the tokenized word reads `n= ; ls -1 /tmp/*_…`, which is not
            # anything the reader can find in their own command. Whitespace-bounded
            # against the original string is the literal that contains the
            # reference — `/tmp/*_${RUNDATE}_*`, `"*_${RUNDATE}*"`.
            around = re.search(r"\S*" + re.escape(spelling) + r"\S*", command)
            # Operators AND the quoting/punctuation that sits against a word: the old set
            # left `"` and `,` inside the quoted literal, so the refusal triage recorded
            # for #3630 reads ``inside the glob word `"run_*_${d}_*.md"', `` — a quote
            # and a comma from the CALLER'S SYNTAX, presented as part of the pattern.
            literal = (around.group(0).strip(";&|()<>\"'`, ")
                       if around else word.strip(";&|()<>\"'`, "))
            return (
                f"`{spelling}` expands to nothing inside {position} "
                f"`{literal}`, so the command measures what the empty string "
                f"matches rather than what you wrote — `{name}` is not in this "
                f"tool's environment"
                + (" — and bash reads the trailing `_` as part of the name, so the "
                   "variable you meant to write is not this one"
                   if not braced else "")
                + f". Spell the value, or write "
                f"`${{{name}:-something}}` to say what the fallback is.")
    return None
