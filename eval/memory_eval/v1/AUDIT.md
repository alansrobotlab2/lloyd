# LloydMemEval v1 — gold-label spot audit (2026-09-25)

Audited by hand (Claude Code, for #1480): 35 dev questions, 7 per category,
drawn with `random.Random(25).sample` per category (the script is the one in
the #1480 write-up). For each: the grounding fact, the question, the gold and
anti values, and the user turns of the evidence sessions that carry them.
Holdout questions were not opened.

Verdicts: **clean** (question answerable from the sessions, gold is the right
value and is checkable), **brittle** (label right, but the gold string is long
or generic enough that the rules judge will miss a correct paraphrase or match
a wrong answer), **defective** (the label is wrong or the item cannot measure
what its category claims).

| category | clean | brittle | defective |
|---|---|---|---|
| single_session | 4 (si-032, si-061, si-069, si-070) | 3 (si-001 gold "language space" under-specifies the asked difference; si-018 two-term phrase; si-025 either dataset alone passes) | 0 |
| multi_session | 5 (mu-023, mu-067, mu-026, mu-055, mu-003*) | 1 (mu-047 long phrases) | 1 (mu-060: both facts say the same thing; gold is the verbs "discussed"/"presented") |
| knowledge_update | 5 (kn-009, kn-054, kn-011, kn-060, kn-038) | 2 (kn-052 old value is a date while gold is an arXiv id — not a competing value; kn-064 gold "5" is a bare digit) | 0 |
| temporal | 7 (te-070, te-018, te-045, te-067, te-053, te-029, te-017; day counts re-computed from the session dates) | 0 | 0 |
| preference | 3 (pr-045, pr-011, pr-055) | 2 (pr-049, pr-030 long gold phrases) | 2 (pr-044: the anti "OAuth 2.0" is part of the right answer; pr-042: gold "any" is a common English word and the question states the preference) |
| **total** | **24 / 35** | **8 / 35** | **3 / 35** |

\* mu-003's gold strings are exact config text (`startretries=3 and exitcodes=0`); counted clean, but an answer that lists the two keys on separate lines misses the first alias.

Reading: 3 of 35 defective (8.6%, Wilson 95% [3.0%, 22.4%]) — preference is
the weak category (2 of 7). 32 of 35 labels are right; 8 of those are brittle
for the deterministic judge, which biases `correct_strict` DOWN (a correct
paraphrase is missed) more often than up (only pr-042 and kn-064 can match a
wrong answer). Two single-session prompts say "you noted" where Alan said it —
wording, not label, errors.

What this audit does not do: it is 35 of 333, not the human review #1471's
closure asks for before a synthetic set is called gold. v2 should drop the
three defective items, give the rules judge shorter gold values for the
brittle shape (the build validator now caps gold at 6 words; many of these
sit at 5-6), and re-state preference probes so the anti value is the generic
alternative, never a component of the right answer.
