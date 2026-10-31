"""A braced reference the Bash tool's own environment does not define expands to
nothing **before the shell parses the command** — and in a measurement position
that empty string is a different measurement, not a failed one. (#3630)

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


def _undefined_names(word: str) -> list[str]:
    """Braced references in `word` that this process's environment does not
    define and no shell built-in owns."""
    return [n for n in _BRACED_REF.findall(word)
            if n not in os.environ and n not in _SHELL_OWN_NAMES]


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
    if "${" not in command:
        return None
    from app.harness.protected_paths import _segments, _tokens

    for argv in _segments(_tokens(command)):
        for i, word in enumerate(argv):
            names = _undefined_names(word)
            if not names:
                continue
            position = _measurement_position(argv, i)
            if position is None:
                continue
            name = names[0]
            # Quote what the caller typed, not the token this check matched: the
            # shared parse rewrites `$(` to a command boundary, so inside a
            # `$( … )` the tokenized word reads `n= ; ls -1 /tmp/*_…`, which is not
            # anything the reader can find in their own command. Whitespace-bounded
            # against the original string is the literal that contains the
            # reference — `/tmp/*_${RUNDATE}_*`, `"*_${RUNDATE}*"`.
            around = re.search(r"\S*" + re.escape("${" + name + "}") + r"\S*",
                               command)
            literal = around.group(0).strip(";&|()<>") if around else word
            return (
                f"`${{{name}}}` expands to nothing inside {position} "
                f"`{literal}`, so the command measures what the empty string "
                f"matches rather than what you wrote — `{name}` is not in this "
                f"tool's environment. Spell the value, or write "
                f"`${{{name}:-something}}` to say what the fallback is.")
    return None
