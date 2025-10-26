# NFSP Blef AI

This project implements a **Neural Fictitious Self-Play (NFSP)** agent for *Blef*, an imperfect-information card game.
NFSP combines **reinforcement learning (RL)** and **supervised learning (SL)** to approximate a Nash equilibrium through continual self-play.


---

### Summary

NFSP enables a form of self-play that’s both **strategically exploratory** and **self-stabilizing**.  
By separating short-term adaptation (RL) from long-term averaging (SL), the agent learns to play optimally against its own evolving strategy without collapsing into exploitation or overfitting.  
The round-aware reward propagation ensures that even short games yield meaningful learning signals for every decision along the way.

---

## Quick Start (5 minutes)

1. **Create a virtual environment** and install dependencies:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **(Optional) Add pretrained embeddings** to `artifacts/` if you have them:
   - `card_embedding_pretrain.pt`
   - `history_embedding_pretrain.pt`

3. **Launch a self-play training run:**
   ```bash
   PYTHONPATH=. python3 nfsp_ai/nfsp_run_local.py        --deck-size 24        --use-card-embeddings auto        --use-history-embeddings auto        --total-steps 100000000        --control-plane control_plane_run.json
   ```

   This will:
   - Print live training metrics  
   - Save checkpoints under `nfsp_blef_<timestamp>.pt`  
   - Log progress in `logs/`  
   - Store sample evaluation games in `games_<timestamp>_eval/`

   To **warm-start** from a saved model but reset the training schedules, use:
   ```bash
   --initialise-with path/to/checkpoint.pt
   ```
   To **resume** from an exact previous point (keeping schedule continuity), use:
   ```bash
   --resume path/to/checkpoint.pt
   ```

Stop here if you just want a working self-play job.  
The defaults (two agents, 24-card deck, no jokers) are ideal for smoke testing and short runs.

## About this NFSP setup

### NFSP Overview

NFSP alternates between two complementary modes that balance **adaptation** and **stability**:

- **Reinforcement Learning (RL)** learns a *best response* to the opponents’ current average strategy.  
  It’s trained via Double DQN using only data from **best-response (BR)** episodes, where the agent acts according to the Q-network (with ε-greedy exploration). 

- **Supervised Learning (SL)** learns the agent’s *average behavior* by imitating all actions the agent actually takes during self-play (except illegal or forced ones).  
  This forms the **average policy** — the stable, long-term strategy the agent uses most of the time, and which the RL component learns to exploit.

Together, these two systems allow the agent to both **adapt** to its opponents and **stabilize** its own policy over time.

---

### Round structure and reward propagation

Each round proceeds through a sequence of player actions and ends when any player performs the **CHECK** action.  
The CHECK action terminates the round and determines the loser:

- The losing player receives `–1`.  
- All other players receive `+1`.  
- Intermediate actions yield `0` reward.

To make early decisions meaningful for learning, the agent uses a **round-aware N-step credit assignment** system:

- When a round ends, the terminal reward is **propagated backward** through all actions of that round.  
  - If the actor eventually loses → `R = –γ^distance`  
  - Otherwise → `R = +γ^distance`  
  - `distance = 0` for the penultimate step (full credit), increasing for earlier steps.  
- The terminal step itself keeps the environment’s ±1 reward.

All transitions are stored as terminal (`done=True`), so Q-targets reduce to `target = reward`.  
This keeps training fully **terminal-based** while giving earlier actions a graded sense of responsibility for the outcome.

---

### Agent actions

During self-play, the agent alternates between two behavior modes:

- **Best-response mode (RL):**  
  The agent acts using its Q-network with ε-greedy exploration over legal actions.  
  Transitions from these steps populate the **RL replay buffer**.

- **Average-policy mode (SL):**  
  The agent acts by sampling from its average-policy network.  
  All actions are stored for **supervised learning**, which continually updates the average policy.  

The switch between these modes is controlled by the **anticipatory parameter (η)**:  
with probability η the agent acts as best-response, and with 1–η it acts using its average policy.

---

### Evaluation

The trained NFSP agent is evaluated against a **fixed, rule-based opponent** that plays a simple but consistent strategy.  
This opponent is **never observed** during the agent’s self-play training, so evaluation measures genuine generalization — how well the agent’s policy performs against unfamiliar, deterministic play styles.

Evaluation reports:

- **win rate:** the fraction of rounds in which the NFSP agent avoids being the loser, and  
- **average reward:** the mean terminal reward (±1 per round), which aligns closely with win rate.  

You can easily replace the baseline opponent with another trained agent, a heuristic bot, or even a human player to test robustness against different strategies.

---

### Training dynamics

Training proceeds through curriculum phases:

1. **Early phase (0–5M steps):** High exploration (`ε`) and anticipatory rate (`η`); frequent forced-checks help the agent learn round boundaries and terminal conditions.  
2. **Mid phase (5–20M):** Forced-checks are phased out; exploration and η gradually decrease; both RL and SL stabilize.  
3. **Late phase (20–100M+):** Learning rates decay; exploration becomes minimal; the agent refines and stabilizes its equilibrium policy.

This schedule allows the agent to first understand the structure of the game, then optimize decisions, and finally converge toward stable mixed strategies.

## Advanced Components
The [/tools](../tools) directory contains some useful utilities.

### 🔧 Control Plane
NFSP Blef includes a **JSON-based control plane** that lets you safely tweak hyperparameters and runtime schedules while training is in progress — without restarting the run.  
It supports live overrides, cooldown intervals, and automatic validation of parameter ranges to prevent invalid edits.  
You can use it to fine-tune behavior such as exploration rates, learning frequencies, or curriculum pacing mid-run.

See the [**Control Plane Operator Guide**](../docs/control_plane_guide.md) for:
- how to launch with a control plane file,
- best practices for atomic edits,
- cooldown semantics, and
- monitoring overrides in logs or TensorBoard.

---

### 🧠 Embedding Systems
The agent can leverage **card embeddings** (and optionally **history embeddings**) to represent input state features in a more structured, learnable space.  
This significantly improves generalization and sample efficiency, especially in large or variant decks.

Pretrained embeddings can be plugged directly into the NFSP agent, or fine-tuned during self-play with controlled learning rates.

See the [**Card Embedding Usage Guide**](../docs/card_embedding_usage.md) for:
- pretraining instructions and artifact contents,
- integration into the observation pipeline,
- fine-tuning and stability tips, and
- regeneration workflows for deck-rule changes.
