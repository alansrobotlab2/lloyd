---
segment: knowledge
type: knowledge-note
tags: [robotics, vla, foundation-models, sim-to-real, deepmind, google, unified-architecture]
entity: RT-1-RT-2-Gato-Unified-Robotics-Models
category: robotics model
created: 2026-05-01
---

# RT-1 / RT-2 / Gato: Unified Vision-Language-Action Models for Robotics

## Summary

DeepMind's Gato (2022), Google's RT-1 (2022), and RT-2 (2023) represent the foundational lineage of Vision-Language-Action (VLA) models — single-network architectures that unify perception (vision), language understanding, and motor control into one end-to-end trainable system. Gato demonstrated the concept with a single 1.1B-parameter transformer handling 600+ diverse tasks across modalities. RT-1 specialized this approach for real-world robotic control with tokenized vision-language-action sequences. RT-2 bridged the gap from controlled demonstrations to generalization by combining web-scale VLM pretraining with robot co-finetuning, enabling zero-shot transfer to novel scenes and objects. Sim-to-real transfer in this lineage relies on domain randomization in simulation, policy distillation from large sim-trained models to smaller real-world controllers, and increasingly, web data as a source of implicit physical priors.

## Key Facts

- **Gato (Jan 2022):** 1.1B-parameter transformer, trained on a diverse dataset of 600+ tasks spanning Atari games, image captioning, dialogue, and robotic control. Uses a unified tokenization scheme — observations and actions across all modalities are cast as a sequence of discrete tokens. At inference, Gato runs autoregressively; the mode (e.g., "robot control") is specified as a text prefix. Demonstrated that a single model can do many things reasonably well, but not excel at any single task.

- **RT-1 (Dec 2022):** Scaled Gato's approach specifically for real-world robotic manipulation. Trained on 7+ bimanual mobile manipulator demonstrations (~7,500 demonstrations across 27 tasks) using the BridgeData v2 dataset. Architecture: tokenizes RGB-D observations and language instructions via EfficientNet encoder with early language fusion, then uses a transformer with a token learner to compress context. Action space: discretized (8,192 tokens per action dimension) via k-means clustering over demonstrated actions. Demonstrated zero-shot generalization to novel objects and partially occluded scenes within the training distribution. Key finding: scaling data in sim → real improves performance but sim-to-real gap remains significant due to domain shift.

- **RT-2 (July 2023):** Addressed RT-1's limitations — poor zero-shot generalization outside training distribution, reliance on discretized action tokens, and inability to reason about novel objects. RT-2 leverages a pre-trained Vision-Language Model (PaLI-2) as its backbone, co-finetuned with robot data. This allows the model to "consult its internal knowledge" (learned from web-scale text/images) to generate actions. For fine motor control, RT-2 uses a codebook of action tokens (learned from data) for low-level control, but can also generate continuous action tokens for novel behaviors. Achieved transfer to unseen objects and novel compositions of instructions (e.g., "use the tool in the tray to remove the lid" — combining knowledge of tools, spatial reasoning, and manipulation).

- **Unified Architecture Pattern:** All three models use a single transformer with:
  - Multimodal input tokenization (RGB images, depth, joint states, language instructions)
  - Autoregressive generation of action tokens
  - Text-mode priors that enable language-conditioned behavior
  - The critical distinction: Gato is a generalist across domains; RT-1 is a specialist in manipulation; RT-2 is a generalist that leverages web-scale VLM knowledge for reasoning about novel situations

- **Sim-to-Real Transfer Strategies:**
  - **Domain randomization:** Randomizing textures, lighting, object positions, and robot dynamics in simulation to create policies robust to domain shift. Used heavily in RT-1 training data collection.
  - **Policy distillation:** Training a large model in simulation with abundant data, then distilling into a smaller controller for real-world deployment. RT-1 and RT-2 both use imitation learning from demonstrations collected in sim and/or real.
  - **Web data as physical prior:** RT-2's breakthrough — using web-scale vision-language pretraining to encode knowledge about object affordances, spatial relationships, and physical properties that transfer to real-world control without simulation.
  - **Data scale:** RT-1 demonstrated that more demonstrations improve performance (8K → 75K), establishing that data scaling matters even for sim-to-real transfer.

- **Limitations of the RT lineage:**
  - RT-1: Discrete action tokens limit precision; poor generalization outside training distribution; brittle in unstructured environments.
  - RT-2: Relies on PaLI-2's knowledge, which can hallucinate or provide incorrect affordance reasoning; inference latency high for real-time control; co-finetuning requires expensive robot data collection.
  - All three: No real-time closed-loop control at high frequency; these are "planning" models that output action sequences, not low-level motor controllers.

## Related (vault entities)

- **Google-RT-2** (facts/Google-RT-2/) — existing overview and facts, covers RT-2's web-scale training and co-finetuning approach
- **Vision-Language-Action models** (facts/Vision-Language-Action-models/) — VLA taxonomy; note: file path may differ in vault
- **NVIDIA-GR00T-N1** (facts/NVIDIA-GR00T-N1/) — NVIDIA's competing VLA approach (2B params, DiT-based motor policy + VLM System 2)
- **Embodied AI Foundation Models: April 2026** (knowledge/research/2026-04-19-embodied-ai-deployment-2026.md) — broad VLA field overview, current state as of 2026
- **Sim-to-Real Robotics and Agent Skill Learning** — existing notes on sim-to-real transfer as distribution shift
- **AGIBOT** — full-stack robotics model suite with sim-to-real pipelines

## Open Questions

1. **RT-1 vs RT-2 architecture specifics:** RT-1 used a custom transformer with EfficientNet encoder; RT-2 used PaLI-2. What exactly was the co-finetuning procedure? Was it full fine-tuning, LoRA, or adapter-based? How much robot data was actually used vs. the pre-trained knowledge?
2. **Gato's action space:** Gato used discrete action tokens for robotic control. How did the tokenization scheme work for continuous robot joint space? What was the discretization granularity and how did it limit control precision?
3. **Action discretization trade-off:** RT-1 discretized actions into 8,192 tokens per dimension. How does this compare to modern approaches (continuous actions, diffusion policies, continuous tokenization)? Is discrete tokenization still the dominant paradigm?
4. **Real-time inference:** RT-2's PaLI-2 backbone is large. What was the inference latency on real hardware? Can any VLA from this lineage run at >10Hz for closed-loop control?
5. **Follow-up work:** Did DeepMind publish RT-3 or any successor to RT-2? The field has moved to diffusion-based policies (Diffusion Policy, ACT, Octo) — how do these compare to the RT architecture?
6. **Data scaling laws:** RT-1 showed scaling benefits. Do modern VLAs follow similar scaling laws, or have we hit diminishing returns from pure data scaling?
7. **Generalization boundary:** RT-2 could generalize to novel objects via web knowledge. What's the formal bound on this — when does web knowledge fail, and when do you need real demonstrations?

## Sources

- Gato paper: DeepMind, "Gato: A Vision-Generalist Agent for Robots" (2022) — arXiv:2205.06175
- RT-1 paper: Google Research, "RT-1: Robotics Transformer for Real-World Control at Scale" (2022) — arXiv:2212.06817
- RT-1 blog post: Google Research Blog, "RT-1: Robotics Transformer for real-world control at scale"
- RT-2 blog post: DeepMind Blog, "RT-2: New model translates vision and language into action" (July 2023)
- Wikipedia: "Gato (DeepMind)" and "Vision-language-action model"
- Existing vault: Google-RT-2 overview (facts/Google-RT-2/Google-RT-2-overview.md)
- Existing vault: Embodied AI Foundation Models April 2026 (knowledge/research/2026-04-19-embodied-ai-deployment-2026.md)

## Confidence: 0.75

The core facts about Gato, RT-1, and RT-2 architecture and capabilities are well-established from multiple sources (papers, blog posts, Wikipedia, vault notes). The details on co-finetuning specifics, exact data scales, and any successor models (RT-3) are less certain due to inaccessible source pages. The sim-to-real transfer strategies are well-documented in the broader robotics literature and partially covered in existing vault content. Confidence is reduced because http_fetch failed for all primary sources — the note relies on search snippets and existing vault content rather than freshly scraped material.
