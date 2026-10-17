"""Arm 1 of the cross-engine disagreement instrument — backlog #2824.

A 120-row labelled gold sample of the djev shadow store
(`~/.local/state/lloyd-djev/shadow.jsonl`), read-only, plus the written rubric that
labels it. The runner surface that scores it lives in
`eval/cross_engine_disagreement.py` as `SURFACE_ARM1`.

The two shipped surfaces take gold from artifacts that already carry a label. This
arm has to author one, so the whole discipline of the package is in two rules:

1. **A row is only admissible if its canvas re-derives byte-exactly.** Every shadow
   row carries `inputs_digest` = the sha1 `app/djev_shadow._digest` took over
   `[state, questions]`, so rebuilding the state from the artifacts the seam read and
   re-digesting it proves the text this arm will put to the two engines is the text the
   seam put to djev. A row whose reconstruction does not digest-match is dropped, not
   labelled: an engine shown a different page than the one the label was read from is
   not being re-asked anything.
2. **A label is only admissible if a rubric line yields it from an artifact that is
   neither scored engine's opinion.** `rubric.py` holds the lines; every gold row
   carries the line it was labelled under plus the evidence read, and
   `rubric.parse_warrant` refuses a warrant that names no artifact.

Read `README.md` beside this file: it states what a Lloyd-authored label is and is not
evidence of, which is the part a reader is most likely to over-read.
"""
