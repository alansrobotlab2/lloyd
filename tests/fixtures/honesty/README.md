# Test-file bodies measured by the honesty prechecks

Each `.txt` here is a whole Python test file as a round would commit it. The
honesty nodes in `tests/test_automod_review.py` write one into a scratch repo,
commit it as `tests/test_a.py`, and measure the delta; the file is never
imported and never collected.

They live here rather than inline in that test because `honesty_prechecks` runs
on the round's *own* changed test files, and the review rung computes it with
the LIVE checkout's `review.py` (#1755): a round that changes that checker is
graded by the version it is replacing, and the version that is running counts
the five dishonesty patterns in the file's raw text, prose and string literals
included. A fixture spelled inline would therefore be counted as newly
dishonest code and refuse the very round that added it — which is exactly how
round `SM_20260928_210355` spent its second review attempt. A `.txt` is not a
test file (`testpaths.is_test_file` asks for `.py`), so the shapes that are
being pinned stay readable, in the code they are, in one place.

`five_literal_patterns.txt` is the only body that holds all five spellings as
real code; the `patterns_named_in_prose*` pair holds them as prose beside a
clean assertion; `real_assert_true_added.txt` is the same file with one of them
written as code instead, which is the pair the blanking pass has to tell apart.
`quote_heavy_docstring_skip.txt` and `multiline_docstring_then_real_assert.txt`
are the two docstring shapes, and `conditional_skip.txt` is the skip a round is
allowed to add.
