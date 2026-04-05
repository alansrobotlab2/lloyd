---
type: research-note
tags:
  - ai/agents
  - ai/learning-agents
  - ai/memory-systems
  - ai/agent-amnesia
  - mlops
date: 2026-04-04
slug: agent-amnesia-self-learning
domain: ai
summary: "Research note on building self-learning agents that overcome 'amnesia' through persistent memory systems, user feedback loops, and ambient context absorption. Based on presentation by Erin Ahmed (Head of Product at Clerk) at MLOps.community."
---

# How to Fix Your Agent's Amnesia: Lessons from Building a Self-learning Agent

## Summary

This note synthesizes key insights from a presentation by Erin Ahmed, Head of Product at Clerk, on building effective self-learning agents. The core problem addressed is "agent amnesia" — the stateless nature of most AI agents that start each session with no memory of past interactions, making them ineffective as long-term collaborators.

The presenter argues that **learning capability is the next frontier of agent differentiation**. As foundation model capabilities commoditize, agents that accumulate knowledge about their environment, team workflows, and past outcomes will become the primary source of competitive advantage.

## Key Concepts

### Stateless vs. Learning Agents

**Stateless Agent (Current State)**
- Starts each session fresh with no memory or context
- Equivalent to a co-worker who remembers nothing from yesterday
- Cannot build cumulative knowledge or improve over time

**Learning/Stateful Agent**
- Learns from experience and adapts to improve performance
- Accumulates knowledge across three dimensions:
  1. **Environment** — infrastructure, tools, systems
  2. **Team & Work Patterns** — preferences, procedures, communication styles
  3. **Past Outcomes & Corrections** — what worked, what didn't, why

### Three Pillars of Effective Learning Agents

#### 1. Make It Easy to Teach via Correction

Wrong answers are inevitable; the winning agents are those that make it easy to correct mistakes and visibly not repeat them.

**Implementation patterns:**
- Propose "memories" that persist across sessions
- Self-harvest skills to encode learned procedures
- Capture user ratings on agent messages as explicit quality signals
- **Most critical:** Visibly never repeat the mistakes

This is especially important for team-based agents where corrections from one user session apply to the next user session.

#### 2. Reward Corrections with Better Performance

Users lose trust when they expend effort to teach the agent something and the lesson is not retained.

**A correction should do three things:**

1. **Persist** — Apply knowledge in the same context
   - Example: If a user teaches the agent how to perform a customer impact assessment after an outage, the agent should recall that procedure the next time it's asked to do the same thing.

2. **Compound** — Generalize knowledge into different contexts
   - Example: If a procedure includes guidance about distinguishing between internal instances and production customer instances, the agent should apply that logic in different investigations (e.g., automatically excluding internal instances from error rate analysis).

3. **Be Visible** — Show your work
   - Display the use of knowledge in actions and reasoning
   - Gives users visibility into when knowledge needs correction
   - Shows impact of contributions, encouraging more input

#### 3. Absorb Context Continuously

Don't rely solely on explicit user direction — create opportunities for ambient learning.

**Implementation approaches:**
- Put the agent in the path of real work automatically
- Absorb every source of environment-specific context available
- Richer model of the user's world = higher utility

The agent should learn from:
- Production incidents and alerts
- Channel history and communications
- Infrastructure and observability data
- User interactions and corrections

### The Learning Loop

The three pillars form a complete learning loop:

```
┌─────────────────────────────────────┐
│         Agent Takes Action          │
│  (fixes bug, investigates, etc.)    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│         User Provides Feedback      │
│      (corrections, ratings)         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│       Memory System Processes       │
│  (persist, compound, make visible)  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    Agent Applies Learnings Next     │
│       Time (Visible Improvement)    │
└─────────────────────────────────────┘
```

**Critical insight:** You cannot have one lesson without the others:
- Easy correction + No visible improvement → Lose user trust
- Visible improvement + No ambient learning → Limited scope (only learns where directed)
- Ambient learning + No user correction → Risk of compounding errors

### Practical Example: Clerk (AI SRE)

Clerk demonstrates these principles in practice:

1. **Environment Learning**
   - Connects to infrastructure and observability tools
   - Builds model of the specific deployment environment

2. **Team Learning**
   - Joins alert channels and learns from history
   - Learns user preferences and procedures over time

3. **Outcome Learning**
   - Learns from every production incident
   - Captures what worked, what didn't, and why

## Papers & Links

### From the Video

- **Speaker:** Erin Ahmed, Head of Product at Clerk
- **Event:** MLOps.community
- **Video ID:** `tKRel4kEpSI`
- **YouTube URL:** https://www.youtube.com/watch?v=tKRel4kEpSI

### Related Concepts (for further research)

- **Memory Systems for LLM Agents:** Research on vector databases, retrieval-augmented generation (RAG), and persistent state management
- **Human-in-the-loop Learning:** Techniques for incorporating human feedback into agent behavior
- **Fine-tuning vs. Memory-based Learning:** The presenter noted they are NOT doing fine-tuning — learning happens through memory systems and context engineering

## Key Quotes

> "Agents that earn their place will be the ones that accumulate knowledge about three things: your environment, your team and how you work, and past outcomes and corrections."

> "The agents that win will be the ones that make it easy to correct mistakes and not visibly repeat those mistakes later."

> "You lose your users' trust when they expend the effort to teach the agent something and that lesson is not retained."

> "The agents that we'll be talking about at this conference next year will be the ones that are investing in this loop today."

## Implementation Considerations

### What NOT to Do

- **Don't rely solely on fine-tuning:** The presenter explicitly states they're not doing any fine-tuning. Memory-based learning through context engineering is sufficient.
- **Don't make corrections opaque:** Users must see the agent apply their lessons.
- **Don't rely only on explicit user direction:** Ambient learning is essential.

### What TO Do

- **Build memory systems that persist across sessions**
- **Capture and act on explicit feedback (ratings, corrections)**
- **Show your work — make learning visible to users**
- **Design for ambient context absorption**
- **Close the learning loop — corrections must lead to visible improvement**

## Relevance to Lloyd

For the Lloyd agent built on Claude Code SDK, this research suggests:

1. **Memory integration is critical** — The current system should be evaluated for persistent memory capabilities across sessions
2. **User feedback mechanisms** — Consider implementing explicit rating/correction systems
3. **Visible learning** — Show users how their input affects agent behavior
4. **Ambient context** — Design for passive learning from environment signals, not just explicit prompts

---

*Generated from MLOps.community video presentation. Note: Web search verification was attempted but unavailable; content synthesized from transcript.*
