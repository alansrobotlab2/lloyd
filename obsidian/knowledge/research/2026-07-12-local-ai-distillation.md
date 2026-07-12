# Local AI Models Destroyed by Further Distillation

**Video:** [Local AI models destroyed by further Distillation](https://youtu.be/Vqk8hoT6moU) by Discover AI
**Primary Paper:** [Purified OPSD: On-Policy Self-Distillation Without Losing How to Think](https://arxiv.org/abs/2607.02234) (arXiv:2607.02234)
**Authors:** Shen, Z. et al. — Zhejiang University, DAMO Academy (Alibaba), Jilin University, Nanyang Technological University
**Submitted:** 2 Jul 2026

---

## Summary

The video discusses a July 2026 paper that reveals **on-policy self-distillation (OPSD) systematically degrades reasoning capability** in local AI models. When a privileged teacher model (with access to reference/golden solutions) supervises a student's own generated trajectories, the student doesn't learn reasoning — it learns shortcuts to the answer. Performance drops across all tested models (Qwen-3-8B, Qwen-3-4B, DeepSeek-R1-Distill-7B) on benchmarks like AIME24 and AIME25 as training progresses.

The core diagnosis: **the teacher's supervision signal is dominated by a reference-induced component** (pure memorization of the answer) that overwhelms the inference-transferable component (actual reasoning). The student inherits shortcuts, not reasoning skills.

The fix is a two-step decomposition:
1. Construct a **reference-only teacher** (same model, given the reference but not the question) to isolate the non-transferable memorization signal
2. Subtract this to get the **inference-transferable residual** (the pure reasoning signal), then transform it into a PMI-based target distribution the student can learn from

---

## Key Findings

### The Problem: OPSD Fails Silently

- **Performance collapse:** All four tested long-CoT models (Qwen-3-8B, Qwen-3-4B, Open-R1-Distill-7B, DeepSeek-R1-6.7B) lose accuracy on AIME24/25 as training steps increase under standard OPSD
- **Epistemic marker analysis:** During training, epistemic markers like "wait" (self-correction tokens) decrease in standard OPSD, indicating the model loses its reflective reasoning behavior. By contrast, DeepSeek-R1 shows opposite behavior — epistemic markers *increase* (from ~70K to ~110K), suggesting a different distillation regime
- **Cosine similarity reveals the trap:** The teacher's total update (Δ_total) maintains high cosine similarity with the reference-induced component (Δ_ref) throughout training. The reference solution shortcut dominates, actively pulling the student toward memorization rather than reasoning

### The Decomposition Framework

The paper decomposes the teacher's supervision into two orthogonal components:

| Component | Symbol | What it captures | Transferable at inference? |
|-----------|--------|------------------|---------------------------|
| Reference-induced | Δ_ref | Information from knowing the answer (memorization of shortcuts) | **No** — student never has the reference at inference |
| Inference-transferable | Δ_IT | Question-conditioned reasoning correction | **Yes** — this is the reasoning skill |

The total teacher update: Δ_total = Δ_ref + Δ_IT

The devastating finding: ‖Δ_ref‖ / ‖Δ_total‖ consistently **exceeds 1.0**, meaning the reference-induced component is *larger* than the total update. The inference-transferable component is not just ignored — it's actively canceled out.

### The Solution: Purified OPSD (OPSD-PMI)

1. **Reference-only teacher construction:** Run the same frozen base model with the reference answer (question hidden) to get logits L_ref. Run with both question + reference to get L_teacher. Run with question only to get L_base.

2. **PMI residual extraction:**
   - Δ_IT = log(π_teacher) - log(π_ref) — the inference-transferable residual
   - This is a per-token log-probability difference, not directly usable as a target distribution

3. **PMI target distribution (GRPO-style closed form):**
   - Transform Δ_IT into a PMI-based target: P_PMI(v) ∝ softmax(L_base) × exp(β · tanh(κ · Δ_IT))
   - This borrows the exponential-of-reward × baseline × renormalize pattern from GRPO/DPO in reinforcement learning
   - Tanh clipping (κ=10) bounds extreme PMI values; β=1 controls the temperature

4. **Practical target:**
   - P_target(v) ∝ softmax(L_student + Δ_IT_stabilized)
   - The student distills from this purified signal, anchored to its own clean reasoning prior

### Implementation Details

- **Four forward passes per step:** Student generates trajectory (ŷ), teacher gets L_teacher, reference probe gets L_ref, base model gets L_base — all from the same frozen model with different input prompts
- **Not actually 4× more expensive:** These are lightweight forward passes through the same frozen model, not full training steps
- **Stability mechanisms:** Centering for numerical stability, tanh soft-clipping at κ=10, β=1 for full correction

### Results

Experiments across four long-CoT models and two datasets show:
- Consistent improvements over both base model and standard OPSD
- Preservation of natural epistemic behavior (epistemic markers don't collapse)
- The mathematical filtration successfully isolates reasoning from memorization

---

## Relevance to Lloyd/Projects

### Local Model Training
- **Direct warning against naive self-distillation:** If Lloyd or any local model tries to improve itself via OPSD with reference answers, it will degrade reasoning. This is especially relevant for domain-specific fine-tuning where ground truth is available
- **The Purified OPSD approach** could be applied to distill reasoning capability into smaller local models (e.g., Qwen-3-4B, Phi, or Gemma variants) without losing reflective reasoning

### AI Safety & Epistemic Integrity
- **Shortcut poisoning in distillation** is a broader safety concern: any system trained to reproduce answers from privileged information will memorize shortcuts rather than learn reasoning. This matters for medical AI, legal reasoning, or any domain where novel cases must be handled correctly
- The **epistemic marker analysis** provides a diagnostic tool: monitor tokens like "wait," "rethink," "validate" during training to detect reasoning collapse before performance drops

### Technical Architecture
- **GRPO-to-distillation transfer:** The PMI target construction shares mathematical structure with DPO/GRPO. This suggests potential synergies between RLHF and distillation approaches
- **Three-pass teacher architecture:** The reference-only teacher pattern (run model with answer-only, question-only, and both) is a reusable pattern for any distillation setup

### Knowledge Distillation Strategy
- **Don't distill from privileged teachers directly** — always decompose the signal first
- **The PMI residual** is a principled way to extract only transferable knowledge, filtering out answer-specific shortcuts
- **Jensen-Shannon divergence** could serve as a stability mechanism for multi-teacher distillation (mentioned in related work)

---

## Open Questions

1. **Generalization beyond math:** The paper focuses on AIME24/25 (mathematics). Does the decomposition hold for other domains like code generation, medical reasoning, or creative writing?

2. **Scale dependence:** How does the Δ_ref / Δ_total ratio behave as models scale? Do larger models have less reference-induced bias because they can reason better even with the answer?

3. **Optimal clipping parameters:** κ=10 and β=1 were found empirically. Is there a principled way to set these based on the difficulty of the task or the gap between teacher and student?

4. **Multi-step reasoning:** The decomposition is per-token. Does it handle cases where reasoning requires backtracking or multi-step planning within a single response?

5. **Epistemic looping:** The video title references "epistemic looping" — a phenomenon where self-distilled models get trapped in circular reasoning about their own training data. Is this the same as the reference-induced shortcut problem, or a distinct failure mode?

6. **Practical deployment cost:** Four forward passes per training step may be prohibitive for very large models or resource-constrained environments. Are there approximations that reduce the overhead?

7. **Relationship to RLVR:** How does Purified OPSD compare with Reinforcement Learning with Verifiable Rewards (RLVR) for achieving similar goals — training reasoning without shortcut memorization?

---

## Related Work

- **Microsoft + Seoul National University (May 2026):** Earlier paper questioning whether self-distillation degrades LLM reasoning capability
- **A Survey of On-Policy Distillation for LLMs:** arXiv:2604.00626 — broader context for OPSD frameworks
- **GRPO/DPO:** Reinforcement learning methods whose exponential-reward × baseline formulation underlies the PMI target construction
- **Medical AI shortcut learning:** Related phenomenon where models memorize textbook answers without reasoning capability

---

## Links

- **Paper:** https://arxiv.org/abs/2607.02234
- **PDF:** https://arxiv.org/pdf/2607.02234
- **Video:** https://youtu.be/Vqk8hoT6moU
- **DOI:** https://doi.org/10.48550/arXiv.2607.02234