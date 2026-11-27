---
type: note
timestamp: '2026-11-27T11:50:00Z'
title: "qmd contention on GPU 4: the eval pin costs production qmd nothing measurable (2026-11-10 to 2026-11-27)"
tags: [eval, measurements, qmd, gpu, automod, regression]
---

# qmd contention on GPU 4, measured from the logs (#1495, #4631)

**Window:** 2026-11-10T00:02Z → 2026-11-27T10:42Z. As the join measures it, the first
`/query` row inside it is 2026-11-10T00:02:31.644Z and the last is
2026-11-27T10:41:25.857Z: **418.6 h, 352,281 rows**, from **339** pinned-corpus
regression windows. (The item this card answers quotes 418.7 h and 352,289 rows from
the same window read an hour earlier off the then-current logs; both are in §5's table,
and the difference is the row count growing, not a different join.)
**Inputs:** the production daemon's own log — `agent-qmd-daemon.err` plus its rotated
generations, UTC stamps, no dates on any line — joined against
`~/.local/state/lloyd-automod/regression.log`, which records each pinned check from its
`measuring` line to its verdict. **Method:** `eval/qmd_contention.py`, read-only; no
synthetic load, no second daemon, nothing restarted.

> **Numbers on this page are from the real logs** (production `/query` timings × the
> regression runner's own window records), not synthetic load or a second daemon. The
> daemon writes a line only when a `/query` returns, so idle minutes are absent and the
> inside/outside comparison is only meaningful on the busy hours: 8,460 minutes in this
> window carried at least one request and **4,141 carried ten or more**, and that is
> what the tables below pool.

**Verdict: on GPU 4, as on the card qmd held before it, the eval pin costs production qmd
nothing measurable.**
The recall doc leg is 198 ms p50 on both sides of a pin, its GPU leg 8 ms both sides, and
the two means that differ at all differ by 1.1 ms and 0.1 ms in opposite directions. This
is the second card the join has produced and the second "nothing measurable", now
recorded on the card qmd actually runs on and reproducible by one command (§6).

## 1. Pin duty cycle

**0.0481 of wall time — 20.13 h of the 418.6 h window — across 339 regression checks.**
A check's own duration is p50 243 s, p95 368 s, mean 214 s, max 446 s: a pinned check is a
few-minute blip about every 74 minutes of wall time. The pin is therefore not a resident
load on this card, it is a short burst that overlaps a working daemon, and only the
inside/outside comparison can see such an event at all. §7 of the 2026-09-25 card made
the same measurement on the previous card at 0.064; the duty cycle is the figure most
sensitive to how a window is cut, so a later re-run that finds a different one is not
contradicting this card.

## 2. Inside vs outside a pin, by request class

`embed` is the query's own embedding — the request's only GPU leg on the recall path, and
the only leg a second tenant on GPU 4 can pre-empt. `vec` is the in-memory VecIndex on
the **CPU**, quoted because it is the largest single leg and shows that most of the doc
leg was never on the card. `total` is the daemon's own end-to-end `(...ms)`.

| class (n outside / n inside) | leg | outside a pin | inside a pin |
|---|---|---|---|
| **recall, doc leg** (320,392 / 21,604) | total | **p50 198 · p95 322** · mean 213.2 | **p50 198 · p95 312** · mean 212.1 |
| | embed (GPU) | p50 8 · p95 13 · mean 13.2 | p50 8 · p95 12 · mean 13.0 |
| | vec (CPU) | p50 140 · p95 205 | p50 141 · p95 194 |
| **lex-only** (8,597 / 1,042) | total | **p50 14 · p95 40** · mean 16.7 | **p50 16 · p95 41** · mean 19.2 |
| **rerank** (624 / 22) | see §3 | see §3 | see §3 |

The recall doc leg — the request shape every MCP search and `vault_search` pays — is
**198 ms p50 on both sides**, and the inside mean is 1.1 ms *faster* than the outside. Its
GPU leg is 8 ms p50 either side with p95 13 ms outside against 12 ms inside. `vec` is 140
against 141 ms: the biggest leg in the request is the one the pin cannot touch, which is
why "the pin shares qmd's card" was never on its own an argument that this path pays
anything.

Lex-only goes 14 → 16 ms p50 and 16.7 → 19.2 ms mean inside a pin, on 1,042 rows against
8,597. It is the only class whose p50 moves, and it is the one class with **no GPU leg at
all** — those lines carry `fts` and `chunk` and no `embed`, no `rerank`. A 2 ms shift on
the leg with nothing for a second GPU tenant to pre-empt is not contention; it is what a
different minute's mix of collection sets looks like. It is quoted here rather than
dropped because it is the only number a critic can point at, and pointed at it argues
against the premise.

## 3. The rerank arm: 22 rows, not priced

Production reaches `rerank` only on the recall's fallback path, on `vault_search`, and on
backlog dedupe — since djev ranks the recall (#1336) it is not the common `vault_search`
leg. In this window there were **624 rerank requests and 22 of them landed inside a pin**
(the figure the item quotes from its own earlier read is 19; §5's table shows why the two
reads of the same window differ, and it is not a disagreement about the arm).
Twenty-two numbers cannot carry a percentile: outside total p50 1033 ms / p95 2656 ms,
inside total p50 1115 ms / p95 1369 ms, and the *leg* itself 807 ms outside against 815 ms
inside. The inside total's p95 sat at 5306 ms on the earlier read and 1369 ms on this one
— which is the whole problem in one number: **one slow request out of twenty-two moves it
by 4 seconds**, so the p95 of that arm is an artefact of sample size and not a
measurement. **This card does not price rerank contention on GPU 4, and says so in those
words.** Pricing it needs either a window holding hundreds of in-pin reranks — at this
traffic that is months, and the log rotates every ~3 days — or the concurrent-daemon
experiment the 2026-09-25 card's §4 describes and nobody has run. #1495 stays open on
that one arm.

## 4. What the two join defects did, measured

`regression.log` stamps are `logging`'s asctime: the machine's **local** wall clock, with
no UTC marker. The daemon log is **UTC** (`10:41:25` on the window's last line, which is
02:41 here). Two defects made the figure irreproducible; both are now fixed, and what
each one cost is measurable on the same bytes:

**(a) one fixed offset applied to stamps that cross a DST change.** The script placed
every regression stamp at `-7` hours (Pacific Daylight) while parsing the daemon as UTC.
This window lies entirely after the 2026-11-01 change, so every stamp in it is −8 and the
constant put **all 339 pin windows an hour early** against the rows they classify — and a
window long enough to matter can spring forward, so no single number is right for one
file.

**(b) one `--first-day` for a log that is rotated, and a day-advance rule blind to
silence.** The old rule advanced the day only when the clock ran *backwards* by more than
an hour; a 30 h silence advances the stamp by 6 h, so the silence is invisible by
construction — a 90 h generation with one 30 h gap parsed to 51 h across 3 calendar days.
And `main()` took one `--first-day` for what is four files, so at most one generation
could be right in a concatenated read. Measured on this box: every generation read under
that scheme landed 86,400 s off its own mtime, because seeding a *local* birth date is
already a UTC day early and rotations land whenever a 10 MB file fills (`.err.1` was
created 08:34Z, which is 00:34 the previous day here).

The same window, three ways:

| | shipped code, −7 | shipped code, −8 | this card (both defects fixed) |
|---|---|---|---|
| pin windows in window | 338 | 339 | **339** |
| recall doc leg total | p50 198 · mean 213.1 | p50 198 · mean 213.2 | **p50 198 · mean 213.2** |
| lex-only inside a pin | n 564 · p50 14 | n 1,026 · p50 16 | **n 1,042 · p50 16** |
| rerank leg inside a pin | p50 678 ms | p50 815 ms | **p50 815 ms** |
| rerank rows inside a pin | — | 19 | **22** |

The percentiles survive both defects — which is why the 09-25 card's verdict still
stands — but the **sample sizes do not**, and the sample size is what decides the one arm
that matters. A one-hour shift moves 462 lex-only rows across the inside/outside boundary
and changes which rerank requests exist at all. That is why the fix is a zone and not a
number, and why `test_a_pin_window_is_placed_by_its_own_stamp_date` pins both sides of
the boundary (2026-10-15 01:00 → 08:00Z, 2026-11-15 01:00 → 09:00Z) while
`test_a_quiet_stretch_longer_than_a_day_gains_its_day_back` pins the silence.

## 5. How an undated, rotated log was dated

The daemon log has no date on any line, and supervisord rotates it at 10 MB, so this
window is spread over **four generations** on this box right now (`.err.3` holds its
2026-11-10 start; `.err` alone covers about 3 hours). What the join does per file now:

- the generation's **creation time in UTC** — `stat -c %W`, the field `os.stat()` does not
  expose on this interpreter even where the volume records one — is the date of its first
  line. A *local* date will not do, and that is the 86,400 s above.
- the file's **creation-to-mtime duration** is its observed span; whole days that its
  stamps fall short of it are assigned to the longest stretches of unchanged clock, which
  is the only place a silence longer than a day can be. Nothing is subtracted when the
  stamps claim more time than the file ran.
- the set is read together (`--daemon-log-glob '…agent-qmd-daemon.err*'`, or repeated
  `--daemon-log`) and the rows sorted, because a rotation can land mid-day inside another
  file's range. No shared `--first-day` is involved: that flag now exists only for a
  generation whose filesystem records no creation time at all (`stat`'s `%W` prints 0
  there, as it does on tmpfs), and such a generation is refused rather than guessed at.

This is what made the 2026-09-25 figure unauditable: its window came from a `--first-day`
someone chose, and nobody could re-derive it. Now the date is in the filesystem.

## 6. Re-running it, and what will have moved

```bash
.venvs/lloyd/bin/python eval/qmd_contention.py \
      --daemon-log-glob ~/lloyd-data/logs/services/agent-qmd-daemon.err* \
      --json /tmp/qmd_gpu4_contention.json
```

Read-only, a few seconds per 10 MB generation. Two things to expect from a later run:

**A shorter window, legitimately.** Each further rotation deletes days from the window's
2026-11-10 end, so re-run a month out and it starts later and holds fewer rows. That is
not a contradiction of this card; the row count, duty cycle and rerank-arm n are the
figures that move, and the recall percentiles are the ones that do not.

**Nothing in `/tmp` to check it against.** `/tmp` on this box is **tmpfs**, so it holds no
artifact across a reboot. The machine-readable rows for this card are written by the
script's own `--json` to `$LLOYD_DATA/eval/qmd-contention-2026-11-27.json` (that is
`~/lloyd-data/eval/`), and that tree is the running system's — outside git, not in this
repo, deliberately. Do not go looking for a JSON under `/tmp`: the copy this card was
drafted from was `/tmp/qmd_gpu4_contention.json` and it is gone, which is exactly why the
`--json` route to `$LLOYD_DATA` exists and why the figures are on this page rather than
in a file that cannot survive.

## 7. What is being decided with this

#1495 asks whether the pinned-corpus check has to move to a dedicated eval daemon — the
2026-09-25 card's option (b), which that card's §1 argued for on architecture rather than
on contention. This card is the contention half of the answer, on the card qmd now runs
on: **the pin costs production qmd nothing measurable on the recall path, on either card,
and the single arm that might care is 22 rows.** No measured contention justifies a
second daemon. What this card does *not* settle is the reason to move anyway — one
codebase instead of two, one warm corpus, one embedding version to keep in step — which is
#1495's judgement and belongs on #1495.

## 8. Caveats

**Both sides of this join read the same production logs.** The inside/outside split is
real — the daemon has no idea whether a request lands during a check — but the halves are
not randomly assigned: pin windows are chosen by the regression runner, not by the
weather, so they land in the hours the box is doing automod work and the traffic mix in
those hours can differ from the rest of the day. That is the shape of the confounder, and
it is also why the 2 ms lex-only shift (§2) must not be read as contention: a 14 ms
leg with no GPU work in it has nothing for a second tenant on this card to pre-empt.

**The eval pin is not the card's only other tenant.** `agent-qmd-watcher` sets the same
`CUDA_VISIBLE_DEVICES` in its own supervisord conf as the daemon's conf does, and it is
the supervisor that moves the pair — `supervisorctl status` shows it running beside the
daemon. Its work is inotify-triggered re-embedding of the vault, so it is load on the card
that no `/query` line shows and no window in this join can classify: it sits in **both**
arms, which is why it does not bias the inside/outside comparison, and it is also why the
card is never as idle as the duty cycle in §1 implies. The docstring and the architecture
page both used to name the eval pin as the card's only other tenant, and the test that
enforced that wording enforced the mistake; `test_the_docstring_names_the_watcher_as_a_co_resident_tenant`
is what stops it surviving again (#4631).

**TTS is not a co-tenant.** The vendored TTS server that used to share qmd's port was
retired in the move that brought qmd to GPU 4 (`a372c876`, 2026-11-10), and TTS pins
`CUDA_VISIBLE_DEVICES=2`. The 2026-09-25 card's §7 still quotes its
`--listen 127.0.0.1:8181`, which is why "the same Python" was once the sharpest objection
to this join; it is not any more, and that card's §8 records the pairing as history.

**The daemon log records only requests.** A line appears when a `/query` returns, so an
idle stretch is indistinguishable from a daemon that was down, and this join says nothing
about behaviour under no load. The busy-minute restriction at the top is what keeps that
from mattering for the claim being made.

**One card's number is a confounder nobody has separated.** On GPU 4 the eval pin and the
watcher's re-embeds share the card with production qmd, so a contention effect belonging
to the watcher is inside both arms here and cannot be attributed. That cuts against
reading the lex-only 2 ms as anything, and it cuts against reading the *absence* of a
recall-path effect as proof that a pin never costs anything — the watcher's own embeds are
already occupying the same resource the pin would have to contend for.
