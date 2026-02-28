# Blef AI

Artificial intelligence to beat humans at the game of **Blef**.

---

## Agents

The repository includes several agent implementations with different learning and reasoning approaches:

- **[cfr_ai/](cfr_ai)** – Counterfactual Regret Minimization (CFR) agent designed to approximate a Nash equilibrium through iterative self-play.
- **[nfsp_ai/](nfsp_ai)** – Neural Fictitious Self-Play agent (NFSP), combining reinforcement and supervised learning for Nash-style convergence.
- **[conservative_ai/](conservative_ai)** – Rule-based baseline using set probability computations for cautious play.
- **[conservative_crawling_ai/](conservative_crawling_ai)** – A variant of the conservative agent.

---

## Shared Modules

Common logic is centralized in **[shared/](shared)**, including:
- [**ai/**](shared/ai)
- [**api/**](shared/api)
- **[probabilities/](shared/probabilities)** – utilities for set existence, card likelihoods, and combinatorial reasoning  
- **[utils/](shared/utils)**

---

## Tools

Utility scripts for development and analysis:
- **Card and history embedding pretraining**
- **Pretty-printing saved game JSONs** for readability and debugging

All under **[tools/](tools)**.

---

## Documentation and Tests

- **[docs/](docs)** – Detailed guides for the control plane, embedding systems
- **[tests/](tests)**
