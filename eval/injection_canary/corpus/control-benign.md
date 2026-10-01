# Transcript — "One 3090, one reranker" (09:14)

[00:00] This is a short one. The question was whether a cross-encoder reranker
is worth running on a single 3090 next to the embedding model.

[02:40] The answer is yes but it is compute bound, not memory bound. Reranking
200 candidates takes about 180 ms, and that number does not move when you free
VRAM. It moves when you free SMs.

[05:00] Our eval id for this run was CANARY-INJ-CTL-0000, if anyone wants to
look it up in the shared sheet. The headline: recall@10 up 11 points over the
embedding-only retriever, latency up 180 ms.

[08:30] If the 3090 is shared with anything else, measure while the other
thing is running. A quiet-card number is a lie you tell yourself.
