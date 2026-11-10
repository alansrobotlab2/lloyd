# Voice prefetch searches without the `[Speaker]: ` tag (2026-11-10)

**Adopted.** `livekit.voice_turn.prefetch_strip_speaker: true`.

Before this change, voice turns reached `prefetch_context` as `[Alan]: <text>`. The speaker
name therefore became a keyword in the lexical leg's query and an entity candidate for the
facts leg, competing for one of its two slots. `app/routers/voice.py::strip_speaker_tags`
now strips the tag from the text that is searched (`query_text`), while the model still
reads the original message, so it still knows who spoke.

## Measurement

`eval/run_prefetch_eval.py` over the 81-query gold set ran twice with one variable changed.
The `--prefix "[Alan]: "` arm prepends the tag to every query, which is how a voice turn
arrived before this change. Artifacts: `speaker-tag-{off,on}-20261110-1055*.json`.

| leg | without tag | with `[Alan]: ` |
|---|---|---|
| lexical doc_hit / MRR | 0.099 / 0.069 | 0.074 / 0.037 |
| hybrid doc_hit / MRR | 0.160 / 0.123 | 0.086 / 0.049 |
| merged doc_hit / MRR | **0.210 / 0.113** | 0.148 / 0.064 |

The legs ran at the same latency in both arms (hybrid p50 116 ms against 117 ms).

Adding the tag cost 30% of merged doc hits and 43% of merged MRR. The off arm is what
typed chat already gets, so stripping the tag only gives voice the same query that chat
already sends.
