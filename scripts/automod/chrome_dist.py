"""The shipped panel half of `chrome-extension/dist/`, and how to prove its bytes.

One caller: `scripts/automod/gate.py`'s `chrome_bundle_provenance`, which #4171 widened
from the worker bundle to the panel half of the same build. It lives here rather than
inline because `gate.py` carries six architecture-doc anchors that resolve by LINE
NUMBER — `Gate.run` at :2618 (cited by `architecture/automod.md:4534` and
`architecture/threat-model.md:106`, and hardcoded as `GATE_LADDER_SPAN` in the doc-claim
rail), :3511 (`measurement.md:297`), :3566 and :3595 (`testing.md:302` and `:300`), and
`rung_review` at :4320 (`threat-model.md:154` and `:155`). Every one of them is below the
chrome section at :613-806, so a round that grew that section in place would silently
rot all six, and the rail that re-checks them by number and by content
(`tests/test_architecture_coverage_doc_claims.py`) is not a thing a round may edit to
accommodate itself. Living here is the other answer the item offers: the gate's edit
stays line-count-neutral and the six numbers keep landing.

**Why the compare is by content and not by name.** Every chunk under `dist/assets/` is
named `[name]-[hash]` by `vite.chrome.config.ts`, so a chunk's name is a hash of the
bytes under it and a name compare proves nothing: a tracked tree holding a chunk the
build no longer emits looks exactly like a match. The bytes ARE reproducible — commit
`41bc89de` witnesses that a second identical build leaves `git status --porcelain --
chrome-extension/dist/` empty, hashed names included — which is what makes a byte-multiset
compare a check instead of a permanent red. That is also the refutation of the reason this
compare did not exist until now — the hashed half was said to re-name itself on every build. The
names move when the source moves, which is the entire point of comparing bytes.
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path

#: The build command's own directory and file, as `chrome-extension/README.md` spells them.
WEB_DIR = "web"
VITE_CONFIG = "vite.chrome.config.ts"
#: The README is where the API base the shipped bytes were baked with is HARVESTED from;
#: an invented base would build a different bundle and call the result provenance, so both
#: empties are a stated skip upstream, never a default.
README = "chrome-extension/README.md"
_API_BASE_RE = re.compile(r"""VITE_API_BASE=['"]?([^'"\s]*)['"]?""")

#: The panel's own html entry: one fixed name, so it is compared by name AND by byte.
PANEL_HTML = "chrome-extension/dist/sidepanel.html"
#: The panel's chunks. These names are content hashes, so this is a PREFIX and can never
#: be a set member — spelled the same way as the grant's `chrome-extension/dist/assets/**`
#: at `spec.py`, whose comment records that a verbatim list of hashes is a grant that
#: expires the first time it is used.
PANEL_ASSETS = "chrome-extension/dist/assets/"
#: What the config's copy plugin writes to the ABSOLUTE `dist/` whatever `--outDir` says.
#: Neither compare below can see them (they are not in the scratch dir), so they stay
#: owned by the porcelain leg alone — #4171 clause 2's exclusion, not an oversight.
COPY_PLUGIN_OUTPUTS = ("chrome-extension/dist/manifest.json",
                       "chrome-extension/dist/icons/")


def panel_path(p: str) -> bool:
    """Is this repo path part of the panel half — the html by name, chunks by prefix?"""
    return p == PANEL_HTML or p.startswith(PANEL_ASSETS)


def rebuild_command(api_base: str, *paths: str) -> str:
    """The command that regenerates the shipped bytes, spelled as the README spells it."""
    return (f"cd {WEB_DIR} && VITE_API_BASE='{api_base}' npx vite build -c {VITE_CONFIG} "
            f"&& git add {' '.join(paths)} && git commit -m \"rebuild chrome bundle\"")


def api_base_from_readme(root: Path) -> str:
    """The `VITE_API_BASE` `chrome-extension/README.md` tells a human to build with."""
    try:
        text = (root / README).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    m = _API_BASE_RE.search(text)
    return m.group(1) if m else ""


def porcelain_paths(root: Path, scope: str) -> list[str]:
    """Every path `git status --porcelain -- <scope>` reports, git's order preserved.

    Splits rename lines on `" -> "` and keeps the target, because a rename inside the
    extension is exactly as dirty as an edit and reporting only the left-hand side would
    name a path that no longer exists.
    """
    r = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", scope],
                       capture_output=True, text=True, check=False)
    out: list[str] = []
    for line in r.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:] if len(line) > 3 else ""
        out.append(path.split(" -> ", 1)[1] if " -> " in path else path)
    return out


def _files_under(root: Path, prefix: str) -> dict[str, bytes]:
    """Every regular file under `root`, keyed by `prefix` + its path relative to `root`;
    `{}` when `root` is absent. An absent half is a real state (a worker-only tree), not
    an error — the caller decides what its absence means.
    """
    if not root.is_dir():
        return {}
    return {prefix + p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _namer(files: dict[str, bytes], wanted: Counter, cap: int = 3) -> str:
    """The NAMES holding the blobs in `wanted`, capped: a reader has to `git add`
    something, and a detail that lists forty chunks would not fit the rung's own cap.
    """
    found: list[str] = []
    for blob, copies in wanted.items():
        found.extend([name for name, body in sorted(files.items())
                      if body == blob][:copies])
    shown = found[:cap]
    return ", ".join(shown) + (f" (+{len(found) - len(shown)} more)"
                               if len(found) > len(shown) else "")


def html_target(html: bytes) -> str:
    """What the shipped panel page points at: its first `src=`/`href=` target.

    A detail-string aid only. Whether that target's bytes exist is the multiset's business;
    reading names to DECIDE would be the exact mistake this compare was written to stop, and
    the docstring says so because the next reader will be tempted. #3769's history is that
    one attribute pointing at a chunk nothing emits, so naming it in the refusal is what a
    reader needs in order to `git add` the right path.
    """
    text = html.decode("utf-8", "replace")
    for attr in ("src=", "href="):
        i = text.find(attr)
        if i == -1:
            continue
        rest = text[i + len(attr) + 1:]
        ends = [x for x in (rest.find('"'), rest.find("'"), rest.find(">"),
                            rest.find(" ")) if x != -1]
        return rest[:min(ends)] if ends else rest
    return "(no src= or href=)"


def compare_panel_half(worktree: Path, built: Path, *, api_base: str,
                       changed: list[str] | None = None) -> tuple[bool, str, dict]:
    """Do the tracked panel-half bytes equal what the build put in `built`?

    `worktree` is the candidate checkout; `built` is the scratch dir the SAME
    `_chrome_vite_build` invocation just emitted into. `changed` is the diff's path list:
    a disagreement BLOCKS only when that diff names a panel path, and is RECORDED as
    `chrome_panel_would_red` otherwise, because the wider-graph leg (a `web/` round that
    ships no dist bytes at all) has never been measured — the round that edited a `.tsx`
    is not the round that left a chunk stale.

    `sidepanel.html` is compared by name and byte. The chunks under `assets/` are compared
    as a BYTE MULTISET IN BOTH DIRECTIONS — every emitted blob has a tracked counterpart
    and every tracked blob has an emitted one, multiplicities included — because their
    names are hashes: what has to match is the SET OF BYTES, and a direction left out is a
    whole failure class (a stale chunk left in, a new one never added). Excluded by
    construction: `dist/manifest.json` and `icons/`, which the copy plugin writes to the
    absolute `dist` and which the porcelain leg beside this one already owns.
    """
    blocking = changed is None or any(panel_path(p) for p in changed)
    tracked_html = worktree / PANEL_HTML
    emitted_html = built / Path(PANEL_HTML).name
    tracked = _files_under(worktree / PANEL_ASSETS, PANEL_ASSETS)
    emitted = _files_under(built / "assets", PANEL_ASSETS)
    t_html = tracked_html.read_bytes() if tracked_html.is_file() else None
    e_html = emitted_html.read_bytes() if emitted_html.is_file() else None
    # ONE shape on every path: the same keys are present whether the half agrees, disagrees
    # loudly, or is absent on both sides. A reader of a ledger row should never have to infer
    # that an absent key meant "verified" rather than "the leg did not run".
    #
    # `chrome_panel_bytes_compared` is the intersection — what the compare actually proved
    # equal — while the two totals beside it are each side's own sum. On a disagreement the
    # intersection is NOT either total: at #3769's state the tree held 3 files and the build 2,
    # and only the two totals say which side grew. The spent branch summed both sides' whole
    # bytes into `bytes_compared` and a node read that as the tracked total; one of the two
    # names had to be wrong, and a counter that double-counts on the red path cannot be
    # compared across rounds, which is what the promotion bar is measured on.
    # Pairs chunks by CONTENT, which is also how the disagreement below is computed, so the
    # count and the verdict cannot disagree about what matched.
    t_blobs, e_blobs = Counter(tracked.values()), Counter(emitted.values())
    data = {
        # Matched BYTES, not "bytes on one side": the compare pairs chunk blobs by content, so
        # a clean rename (#3963's own case, where every name moved and no byte of state
        # changed) has an empty name-intersection yet proved its whole half. Pairing the blobs
        # is the only reading that is right on both that and a missing chunk, and it is what
        # makes `chrome_panel_bytes_compared > 0` mean "a compare happened" rather than "the
        # names happened to line up".
        "chrome_panel_bytes_compared": (len(t_html)
                                        if t_html is not None and t_html == e_html else 0)
        + sum(min(t_blobs[b], e_blobs[b]) * len(b)
              for b in set(tracked.values()) & set(emitted.values())),
        "chrome_panel_tracked_files": len(tracked) + (1 if t_html is not None else 0),
        "chrome_panel_emitted_files": len(emitted) + (1 if e_html is not None else 0),
        "chrome_panel_tracked_bytes": (len(t_html) if t_html is not None else 0)
        + sum(len(b) for b in tracked.values()),
        "chrome_panel_emitted_bytes": (len(e_html) if e_html is not None else 0)
        + sum(len(b) for b in emitted.values()),
        # Files whose BYTES matched, the same pairing as `chrome_panel_bytes_compared` above,
        # so the two figures describe the same set. On a green half this is the whole panel
        # (`sidepanel.html` plus its chunks); on a disagreement it is what the compare still
        # proved, which is never the tracked file count — a half where two of three chunks
        # matched must not read as a full compare. Emitted on every exit, so a refusal, a
        # recorded would-red and a green all carry the same names.
        "chrome_panel_files_compared": (1 if t_html is not None and t_html == e_html else 0)
        + sum(min(t_blobs[b], e_blobs[b])
              for b in set(tracked.values()) & set(emitted.values())),
    }
    if t_html is None and e_html is None and not tracked and not emitted:
        # No panel half on either side: nothing to compare and nothing to claim about it.
        # A base without `chrome-extension/` reaches here, as does this repo's own test
        # fixtures — an empty answer must not read as a passing compare.
        return True, "", data

    problems: list[str] = []
    if t_html != e_html:
        if t_html is None:
            problems.append(f"the build emits a `sidepanel.html` and `{PANEL_HTML}` is "
                            "not tracked at all: a round deleted the panel's entry")
        elif e_html is None:
            problems.append(f"`{PANEL_HTML}` is tracked but the build emitted no "
                            "`sidepanel.html` for it to be")
        else:
            problems.append(f"`{PANEL_HTML}` holds {len(t_html)} tracked bytes and the "
                            f"build emits {len(e_html)} — it points at "
                            f"`{html_target(t_html)}`, a path no build output is "
                            "named by unless the bytes under it are the build's")

    extra, missing = e_blobs - t_blobs, t_blobs - e_blobs
    if extra or missing:
        data["chrome_panel_extra_bytes"] = sum(len(b) * n for b, n in extra.items())
        data["chrome_panel_missing_bytes"] = sum(len(b) * n for b, n in missing.items())
    if extra:
        problems.append(f"the build emits {sum(extra.values())} chunk(s) the tracked tree "
                        f"does not hold — emitted but not tracked "
                        f"({data['chrome_panel_extra_bytes']} bytes): "
                        f"{_namer(emitted, extra)}")
    if missing:
        problems.append(f"the tracked tree holds {sum(missing.values())} chunk(s) this "
                        f"build does not emit — tracked but not emitted "
                        f"({data['chrome_panel_missing_bytes']} "
                        f"bytes): {_namer(tracked, missing)}")

    if not problems:
        return True, (f"the panel half (`{PANEL_HTML}` plus {len(tracked)} hashed "
                      f"chunk(s)) is byte-identical to what the build emits, by content "
                      f"and not by name ({data['chrome_panel_bytes_compared']} bytes)"), data

    sentence = ("the panel half of `chrome-extension/dist/` is not what this build "
                "emits: " + "; ".join(problems)
                + ". Rebuild and commit it (`"
                + rebuild_command(api_base, PANEL_HTML, PANEL_ASSETS + "*") + "`). The "
                "compare is by CONTENT precisely because a chunk's name is a hash of its "
                "bytes, so a matching name says nothing about the bytes under it; the "
                "bytes themselves are reproducible — `41bc89de`'s second identical build "
                "left `chrome-extension/dist/` porcelain-clean, hashed names included.")
    if blocking:
        return False, sentence, data
    data["chrome_panel_would_red"] = True
    return True, ("chrome panel half RECORDED, not blocking: this diff names no path "
                  "under `chrome-extension/dist/`, so the round that could have rebuilt "
                  "it is not this one. " + sentence), data
