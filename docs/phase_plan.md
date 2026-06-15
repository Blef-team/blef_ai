# NFSP Blef Bot — Deep Audit & Path Forward

## Context

This Blef AI project uses NFSP (Heinrich & Silver 2016) as the active learning approach. This document is a no-flattery deep dive: what's actually wrong, which choices were made by intuition rather than evidence, and what concretely makes this bot strong.

Blef is a Polish set-claiming card game (rules in `Blef_game_readme_rules.md`). Each round, players take turns making a **bet** (a claim that a particular **set** — pair / two pair / straight / full house / flush / etc. — exists in the *union* of all players' cards), strictly more **senior** than the previous bet. The only round-ending action is **check**: the checker calls out the previous bettor's claim. If the pool satisfies the last bet, the checker loses; otherwise the bettor loses. Loser gains a card next round and starts it. A player at the card cap who loses is eliminated. Game ends with one survivor. 24-card deck has 88 sets (89 actions including CHECK); 32-card deck has 160 sets (161 actions). Variants: jokers, blanks, common cards.

This document is a strategy spec, not an implementation plan. It is meant to be executed in order — measurement first, then bug fixes, then architecture.

**A note on language.** Blef ≠ poker. There is no "call", no "raise", no "fold". Bets advance through a strict ordering of sets; the only response to a bet that ends the round is a check. Where this plan uses "bet" / "check" / "set" / "more senior", those are Blef's own terms, not poker imports.

---

## Status quo (verified by reading code)

- **Single agent, shared weights, plays all seats** during `train_from_selfplay` (`nfsp_ai/agent.py:1296+`). Anticipatory mixing η decides per-step whether the agent acts from BR (Q ε-greedy) or π (avg policy).
- **Q-network is NOT learning Q-values.** The "RL update" is `F.smooth_l1_loss(q(s)[a], r)` (`nfsp_ai/agent.py:521-524`) with no bootstrap and `target = rew`. The comment is honest: *"Terminal-only target = reward (no bootstrap)"*. Combined with manual `γ^distance` reward shaping in `_flush_nstep` (`agent.py:1230-1242`), this is **Monte-Carlo regression with hand-crafted credit assignment**, not Q-learning. The γ shaping itself is fine — it's the credit-assignment mechanism. What's dead is the TD-related machinery sitting alongside it: `q_tgt`, Polyak updates, hard-target interval, `target_tau`, `use_double_dqn` flag, and the `gamma_power` *field on the buffer* (different from γ — this is the unused per-transition `γ^k` slot). All overhead with no consumer.
- **NFSP buffer assignment is polluted, not inverted.** RL buffer accepts only BR transitions (gated on `is_br` at `agent.py:1141, 1218`). SL reservoir accepts BR **and** π transitions (`agent.py:1488` — no `is_br` gate). Canonical NFSP wants only BR (s, a) in M_SL so π converges to the average BR; here ~75–90% of SL labels are π imitating itself.
- **Public-prior hard-masks the legal action set** (`nfsp_run_local.py:431`: `mask *= (pri > 1e-9)`). Bugs in `shared/probabilities/dynamic_probabilities.py` retroactively change which actions the agent can ever take. PR #40 was exactly such a bug — every pre-#40 checkpoint was trained on a slightly different effective game.
- **Round-end is treated as a hard episode terminal** (`nfsp_run_local.py:1023`: `done = True` whenever a check resolves the round). This is the right design choice. Blef has dense per-round ±1 rewards (~5 actions per round), so each round being an episode gives the cleanest possible learning signal. The structural argument for cross-round value flow was turn-order asymmetry (loser of T starts T+1, starter ≠ follower), but existing CFR analysis shows positional advantage matters only in early rounds and converges to 50/50 in later rounds — i.e. it lives in the regime where NFSP is already strong, not where NFSP needs help. Round-end done stays.
- **Eval is win-rate vs `ConservativeAgent` only** (`agent.py:93`). No exploitability, no league, no head-to-head against the project's own CFR agent.
- **Latent bug:** `ReplayBuffer.gpow` is overwritten with a scalar tensor every `add()` (`agent.py:257`: `self.gpow = gamma_power` instead of `self.gpow[self.ptr] = gamma_power`). Harmless today because `gpow` is never read. Becomes a silent corruptor of n-step targets the moment bootstrapping is reinstated.
- **Reward perspective IS coherent.** Both terminal (`nfsp_run_local.py:1009`: `elif loser == actor_nick: reward = -1.0`) and non-terminal shaped rewards (`agent.py:1241`) are actor-relative. `_ref_nick` is a kill-switch when no reference is set, not a perspective bug.
- **Curriculum is enormous and unjustified.** 100M-step multi-phase schedule (η, ε, lr_q, lr_pi, train_rl_every, batch_rl, train_sl_every, n_step, check_prob, max_cards) tuned across `GPT_improvement_advice*.txt` iterations with no outcome metric attached.

---

## What was done in an immature / arbitrary way

Severity-ranked. "Immature" = decision was clearly intuition or LLM-iterated rather than evidence- or experience-driven.

### Critical (correctness)

1. **Q-loss has no bootstrap, but the rest of the stack carries TD machinery anyway.** `q_tgt`, `target_tau=0.005`, `hard_target_interval`, `use_double_dqn` flag, the `gamma_power` buffer field, `n_step` infra — all dead because `target = rew`. The γ used inside `_flush_nstep` for within-round credit shaping is *not* the dead piece; it's doing real work and stays. Recommendation: keep round-end done, commit to MC, delete the TD scaffolding. See "On the bootstrap question" below.
2. **SL reservoir is polluted by π's own actions.** π is partly trained to imitate itself. Convergence target shifts from "average BR over training" toward "whatever I've been doing." One-line fix: gate `sl_buf.add` on `is_br`.
3. **Public-prior hard-mask binds policy to a probability calculator.** `_legal_action_mask` zeros out actions whose generic public probability is < 1e-9. This is a heuristic, not a Blef rule. Probability-calculator bugs change the action set the agent ever sees. Should be a feature, not a mask.
4. **`ReplayBuffer.gpow` storage is a one-line landmine** (`agent.py:257`). Won't fire under MC; only matters if anyone reintroduces bootstrapping. Easiest resolution: delete the `gpow` field along with the rest of the dead TD machinery in fix 1.1.

### High (algorithmic / methodological)

6. **No exploitability metric.** Win-rate vs `ConservativeAgent` (which itself falls back to `random.choice(legal)` when its action is publicly-mask-zeroed) is the sole tracked outcome. Improvements against this baseline are uninformative about Nash convergence and could even reflect overfitting to ConservativeAgent's quirks.
7. **Multi-phase 100M-step schedule with no validation loop.** Numbers came from successive LLM advice iterations. Annealing η, ε, lr, cadence, n_step, max_cards, check_prob simultaneously over 100M steps without an outcome metric is bayesian-optimization-on-noise.
8. **`gamma=0.666` was set without justification.** Innocuous today (γ doesn't enter the loss); will matter the moment bootstrapping is real.
9. **Forced-CHECK exploration probability decaying 0.25 → 0** is a bandaid. The agent never learns to check at the right moment because the flat 89/161-action softmax has no inductive bias separating CHECK from BET, and CHECK has unique reward semantics (it's the only round-ending action). Curriculum hides the symptom; factorized action head fixes the cause.
10. **Single static training opponent (the agent itself).** Symmetric self-play with shared weights is canonical NFSP, but for n>2 player Blef the convergence guarantees don't hold even in principle. There's no opponent diversity, no snapshot league, no exploiter.

### Medium (architecture / scale)

11. **Hidden width = 128** chosen by recent commit "Set hidden layer width to 128" with no ablation. Obs dim is 392–716. Probably not the bottleneck, but unjustified.
12. **Flat softmax over 89 (24-card) / 161 (32-card) actions** with no factorization. CHECK is just another logit, indistinguishable from a "Pair of 9s" bet to the policy. Requires curriculum bandaids; doesn't transfer between deck sizes.
13. **Probability priors fed into observation as 2 × hist_dim features.** Helpful (the agent gets ground-truth-ish bet feasibility) but creates a brittle policy that depends on a separate computational module's correctness. PR #40 changed those features for every state in retrospect.
14. **`MAX_PLAYERS=8`, `HISTORY_LEN=8`, `HISTORY_SLOTS=8`** hardcoded with custom seat-rotation tables. Limits scaling; makes 9+ player a refactor not a config change.
15. **Card and history embeddings exist as pretrained artifacts but are NOT integrated into the live NFSP networks.** `docs/card_embedding_plan.md` and `docs/embedding_pretraining_plan.md` are marked "in progress" — encoders ship as files, but the policy/Q nets still consume legacy multi-hots by default.

### Low (code hygiene — defer)

16. **`agent.py` is 1862 lines** with `#DEBUG` print blocks (`agent.py:1124-1133, 1211-1215, 1244-1249`), unreachable code (`agent.py:163-168` after the return on 162; `agent.py:1239` after `raise`), dead variables (`gamma_power` "not really used"), and one giant `train_from_selfplay` method.
17. **`ConservativeAgent` falls back to `random.choice(legal)` on illegal action**, which silently weakens the only baseline.

---

## On the bootstrap question (basics + Blef-specific reasoning)

The Q-network learns `Q(s, a)` = expected total future reward from taking action `a` in state `s`. The choice is in the *target* used to train it.

- **Monte Carlo target.** Roll out the episode, observe `R = r_t + γ·r_{t+1} + … + γ^{T-t}·r_T`, regress `Q(s_t, a_t)` against `R`. No model contribution to the target — the target is what actually happened.
- **TD / Q-learning target with bootstrap.** After observing `r_t` and `s_{t+1}`, target = `r_t + γ · max_{a'} Q_target(s_{t+1}, a')` if not terminal, else `r_t`. The bootstrap is the `Q_target(...)` term: future value is filled in by the model's own current estimate.

Trade-offs in general:
- MC: zero bias, high variance, learns only when episode ends, off-policy data goes stale.
- TD: lower variance, propagates one step per update, biased while Q is wrong, off-policy data stays usable because the bootstrap is recomputed against current Q.

**The Blef-specific subtlety.** People sometimes argue TD beats MC in self-play because old transitions go stale (MC returns were generated against an older opponent; TD recomputes the bootstrap with current Q). That argument *only matters if there's a cross-episode bootstrap term that survives the `done` flag*. In the current env, every round-terminal sets `done=True` (`nfsp_run_local.py:1023`), so `(1 - done) · Q_target(s', a') = 0` at round-end. Within a single round, MC and TD produce identical targets at the terminal step. The "TD is more robust to opponent drift" argument has no purchase here.

**Round-end done is the right design choice for Blef.** The trade-off is signal-to-noise per training target:

- *Round-end done*: target for every action ≈ ±1 from this round. Each of ~5 actions per round gets a clean ±1 signal. Low variance.
- *Game-end done with MC*: target = sum of ~20 ±1 round outcomes (Blef games run 17-23 rounds). Variance ≈ 20× higher per target.
- *Game-end done with TD*: bootstrap shrinks variance back down, at the cost of bootstrap bias and additional target-net machinery.

Throughput is identical across all three — same env steps/sec, same Q updates/sec. What differs is signal quality per update. Blef has the rare luxury of dense per-round terminal rewards; treating each round as an episode is exactly what that luxury affords.

The only structural argument for cross-round value flow was turn-order asymmetry (loser of T starts T+1, starter and follower face different decision problems). Existing CFR analysis already shows this asymmetry is meaningful only in early rounds and decays to 50/50 in later rounds — i.e. it exists in the regime where NFSP already converges quickly, and is gone in the regime (high cardinality, late rounds) where NFSP needs help. Card-count and elimination effects are even weaker.

So: round-end done is correct. The TD-with-game-level-done branch is closed.

**The current code is genuinely worst-of-both.** It MC-shapes rewards backward at flush time (targets are baked in like MC) but maintains the TD machinery (target net, Polyak, `gamma_power`) with no payoff. Pure cost, no benefit.

**Recommendation.** Keep round-end done. Commit to MC. Delete `q_tgt`, Polyak, hard-target interval, `target_tau`, `gamma_power` plumbing, and the "n-step Q-learning" framing in `_flush_nstep`. Regress Q(s, a) against the round's actual return (±1 with the existing γ^d within-round shaping kept or dropped — worth ablating). This is honest about what the code is doing and removes the dead overhead.

---

## Path forward

Strict ordering. **Do not train new checkpoints until Phase 0 is done.** You have ~250 saved checkpoints at the repo root, each from a 1–8 hour run, and none has a comparable exploitability number. Fix the metric before fixing anything else.

CFR is **not** the path forward as a teacher / initializer. Behavioral cloning π from CFR was in an earlier draft of this plan; it was wrong. Reasoning: NFSP and CFR both self-play, and both get strong at small game-state cardinalities. The hard regime is later rounds with many cards, where state space explodes — CFR struggles there too. Initializing NFSP from CFR helps NFSP exactly where NFSP doesn't need help. CFR's role in this plan is a **measurement opponent**, not a teacher: at small cardinalities CFR's strategy is near-Nash, so a head-to-head margin against CFR is a strong "is the agent strong" signal, much better than win-rate vs ConservativeAgent.

### Phase 0 — Measure before you fix (1–2 days)

The only commit that should land before any algorithmic change.

- **0.1 Build the metric ladder.** Head-to-head eval harness with four opponents, in order of strength: `random_legal` (floor), `ConservativeAgent` (current baseline), `cfr_ai/agent.py` loaded from CSV strategies (strong baseline, near-Nash at low cardinality), and a rolling pool of past π snapshots (self-improvement signal). Each is a head-to-head winrate / mean-reward number per checkpoint. Cheap rollouts only — no exploitability calculator, no abstraction marginalization.
- **0.2 Backfill on past checkpoints.** Run 0.1 on the most recent ~10 checkpoints. Plot all four curves vs training step. Reveals whether the existing pipeline is improving at all and gives a baseline every subsequent fix must beat.
- **0.3 Wire into training loop.** Run 0.1 at the existing eval cadence; write to CSV + TensorBoard alongside the current metrics.
- **0.4 Define metric-based hyperparameter triggers.** Replace step-based schedule with rules like "drop η when CFR-margin plateaus for K evaluations," "raise batch_rl when ρ falls below threshold," "freeze schedule when snapshot-league self-improvement reverses." Concrete thresholds get tuned later; the deliverable here is the *trigger framework*, expressed as control-plane rules. The 100M-step schedule does not get deleted yet — it stays as the default until metric-driven overrides demonstrate they're better.

- **0.5 Diagnose the late-game generalization failure.** Empirical evidence: at 100M+ training steps, models are "stupid" in late game — no generalization. That rules out compute and most capacity hypotheses (more steps with the same setup wouldn't fix it). The remaining candidates, in order of likely culprit given the evidence:

  1. *Buffer-distribution starvation:* late-game states (high card count) are rare in the natural data distribution. Within max_cards=11 self-play games, only the round-loser accumulates cards, so most rounds happen at low/mid cardinality. Replay buffer + SL reservoir end up with ~5–10% late-game data, 90+% early/mid. Network has effectively seen 5–10× less late-game variety than early-game and can't extrapolate.
  2. *Multi-hot encoding fails to scale:* hand encoding goes from sparse (1 card → one 1-bit) to dense (10 cards → ten 1-bits). The first linear layer's activation patterns change qualitatively, not just quantitatively. A network trained on sparse inputs can't generalize to dense ones. *The pretrained card embeddings sitting unused in `nfsp_ai/embedding/` would normalize this*; they encode "set of cards" as a smooth learned distribution.
  3. *Overfitting + schedule lock:* network with sufficient capacity memorizes the limited late-game samples it does see; schedule anneals lr/η/ε down, freezing the overfit policy.
  4. *Input dimensionality:* high nominal D dominated by correlated priors. Lower probability now given the buffer-distribution story explains the symptom more directly.

  Cheapest-first resolution:

  - **0.5a Free buffer-distribution diagnostic.** Histogram replay buffer + SL reservoir transitions by total card count. Histogram visited-states by round number and max_cards. Confirms whether late-game is starved. Runs on existing checkpoints, no retrain.
  - **0.5b Free instrumentation.** Add Q-loss (in-distribution) and head-to-head-vs-CFR (held-out) to same plot; add policy entropy. Train↓ eval↔ ⇒ overfitting confirmed.
  - **0.5c Curriculum redesign (1 retrain).** Reweight buffer sampling toward high-card-count transitions, OR sample max_cards uniformly during training instead of growing. If late-game metric improves materially → buffer-distribution was a primary cause.
  - **0.5d Schedule-freeze ablation (1 retrain).** Hold η/ε/lr constant at initial values. Wins ⇒ schedule actively harming.
  - **0.5e Width sweep (3 retrains, defer unless 0.5a–d don't explain the gap).** `hidden ∈ {128, 256, 512}`. Demoted because the empirical 100M-step evidence suggests capacity isn't the bottleneck; bigger nets on starved data will likely overfit faster, not generalize better.

  Net effect: targeted experiments based on the strongest current evidence, not hopeful changes.

- **0.6 Run management + live dashboard.** Both needed for me (Claude) to operate experiments and for you to monitor them.
  - *Logging:* every metric (head-to-head ladder, train/eval gap, entropy, ρ, buffer fills, override-source trace) writes to a per-run CSV. Single canonical metric store; CSV is what I poll programmatically and what the dashboard reads. No TensorBoard — dropped because you don't want to learn it.
  - *Naming convention:* each run lives at `runs/<YYYYMMDD-HHMMSS>__<experiment-name>/` with `metrics.csv`, checkpoints, control-plane JSON, and saved games inside. Timestamp + name guarantee runs are clearly identifiable and chronologically sortable.
  - *Dashboard `tools/dashboard.py` (Streamlit, ~80 lines):* sidebar auto-discovers run dirs, multi-select checkboxes for overlaying runs, main panel of line charts (one per metric) with per-series toggle. Auto-refresh every few seconds so in-progress runs update live. Launched once with `streamlit run tools/dashboard.py`; new runs appear as they start. No account, no config.
  - *CLI `tools/run_manager.py`:* subcommands `start` / `status` / `override` / `stop` / `list` / `compare`. `start` launches `nfsp_run_local.py` in background with per-experiment paths. `status` reads the per-experiment CSV and returns the latest snapshot. `override` writes to the experiment's control-plane JSON (existing cooldown logic applies). `stop` clean-shutdowns. `list` and `compare` are summary helpers across runs.
  - *I/O loop:* I (Claude) start a run, poll `status` periodically, watch the four-metric ladder, issue `override` if a tripwire fires, kill+launch a new variant when the current one plateaus or degenerates. You see the same data live in the dashboard.

### Phase 1 — Correctness fixes (3–5 days, pause to measure between each)

After Phase 0 you will know which of these actually moves exploitability. Apply in this order, retraining from scratch for 5M steps after each, comparing against the Phase 0 baseline.

- **1.1 Commit to MC, round-end done.** Delete `q_tgt` (`agent.py:535-540`), Polyak, hard-target interval (`agent.py:533-534`), `target_tau` from config, `gamma_power` plumbing, the `use_double_dqn` flag, and the "n-step Q-learning" framing in `_flush_nstep`. Keep the γ^distance within-round reward propagation but rename it MC return shaping (it's an off-policy value-regression target, not a Bellman target). Optionally ablate γ^d shaping vs no shaping. No TD branch — existing CFR data already establishes that cross-round value flow isn't material in the regime where NFSP needs help.
- **1.2 Gate `sl_buf.add` on `is_br`** at `agent.py:1488`. One-line fix; removes π self-imitation pollution.
- **1.3 Drop public-prior from `_legal_action_mask`** at `nfsp_run_local.py:431`, or move to a diagnostic-only flag. Keep priors as observation features. Decouples policy from `dynamic_probabilities.py` correctness.
- ~~**1.4 Fix `ReplayBuffer.gpow`**~~ *Removed.* Since 1.1 commits to MC and `gamma_power` is being deleted from the buffer entirely, the bug becomes moot. Just delete the field.

### Phase 2 — Architecture (only after Phase 0–1 land)

- **2.1 Factorize the action head** into (set_type, parametric_detail) sub-heads, with CHECK as its own logit. Drop the forced-CHECK curriculum once factorization is in. Bonus: action head transfers cleanly between 24-card and 32-card decks because the parametric_detail head learns "rank-as-an-index" rather than tying each rank to a fixed action ID.
- **2.2 Wire the existing card and history embeddings** (`nfsp_ai/embedding/encoder.py`, `nfsp_ai/embedding/history.py`) into the live policy/Q nets. Pretraining is already done; integration was on the roadmap as "in progress" and never finished. This is where representation-learning gains for high-cardinality late game are most likely to come from.
- **2.3 Hidden-width sweep** with proper compute control: {128, 256, 512, 1024} × fixed total compute budget. Run after 2.1 and 2.2 because they change the upstream representation.
- **2.4 Snapshot-league self-play training** (not just eval). Maintain a rolling pool of past π checkpoints; sample one per episode as the opponent for each non-learner seat. Trades pure-symmetric-self-play correlation for diversity. This is the correct way to break the n>2-player NFSP-isn't-guaranteed-to-converge problem in practice.

### Phase 3 — Process / scaling (defer)

- **3.1 Refactor `agent.py` (1862 lines) into focused modules** ≤300 lines each: `nfsp_agent.py`, `replay.py`, `reservoir.py`, `train_loop.py`, `eval.py`. Defer because algorithm correctness must come first; refactoring before fixing buries bugs in diff noise.
- **3.2 Delete the 100M-step multi-phase schedule.** Only after the Phase-0 metric-driven trigger framework has demonstrated it can do the schedule's job better. Keep the simplest working schedule.
- **3.3 Switch algorithm only if Phase 0–2 plateau high.** NeuRD is a near-drop-in replacement for the policy network with theoretical multi-player benefits; Deep CFR is a larger rewrite with multi-player extensions. Don't rewrite preemptively — the bug list above explains the symptoms without needing a different algorithm. R-NaD / Player of Games is overkill until cheaper fixes are exhausted.

---

## Critical files

- `nfsp_ai/agent.py` — 1862 lines; SL/RL gating, Q-loss, `ReplayBuffer.gpow`, `_flush_nstep`, schedules.
- `nfsp_ai/nfsp_run_local.py` — env wrapper, observation encoding, `_legal_action_mask` (line 431, public-prior hard-mask), terminal reward (line 1009), round-end-as-done (line 1023).
- `cfr_ai/agent.py` + `cfr_ai/outputs/*.csv` — *NOTE: `cfr_ai/outputs/` does not exist locally.* CFR strategies are referenced by the agent at `cfr_ai/agent.py:28` (`'cfr_ai/outputs/<hand_sizes>/<key>/<key>.csv'`), but the data lives elsewhere — likely S3 or production storage given the project has Lambda dispatch infrastructure. **Phase 0 precondition:** locate or regenerate CFR strategies, at minimum for 1v1 24-card-deck no-jokers. Without this, the "head-to-head vs CFR" metric is unreachable and the ladder degrades to {random_legal, ConservativeAgent, snapshot-pool}.
- `shared/probabilities/dynamic_probabilities.py` — calculator whose bugs currently propagate into the action mask via `_legal_action_mask`.
- `nfsp_ai/embedding/{encoder.py,history.py}` — pretrained artifacts ready for Phase 2.2 integration.

---

## Verification

Each phase has one outcome that must hold before moving on. If it doesn't, debug rather than skip ahead.

- **Phase 0 done when:** head-to-head eval harness runs all four opponents (random_legal / ConservativeAgent / CFR-from-CSV / snapshot pool) on a checkpoint, writes to per-run CSV. Backfilled across ~10 historical checkpoints; all four curves plotted in the Streamlit dashboard. Metric-based hyperparameter trigger framework expressed in control-plane rule form, even if no triggers are firing yet. Late-game decision tree (0.5a–e) executed at least through the free diagnostics (0.5a + 0.5b); downstream retrains run only if those rule out cheaper explanations. Result: a documented hypothesis (buffer-distribution / multi-hot scaling / overfitting / capacity / none-of-these) with the diagnostic data backing it. `tools/run_manager.py` and `tools/dashboard.py` both functional.
- **Phase 1.1 (RL formulation) done when:** `q_tgt`, Polyak, `target_tau`, hard-target interval, `gamma_power` field+plumbing, and `use_double_dqn` are deleted from config and code paths. 5M-step retrain matches or improves Phase-0 baseline exploitability.
- **Phase 1.2 (SL gating) done when:** unit test simulates 1000 steps with η=0.5 and asserts `sl_buf.size` ≈ `0.5 × steps_with_legal_BR`. Retrained 5M-step checkpoint exploitability changes (up or down — either is informative).
- **Phase 1.3 (mask) done when:** `_legal_action_mask` returns identical results regardless of `pub_prior` input. Action-set diff vs current implementation reported. 5M retrain exploitability does not get worse.
- **Phase 2 fixes done when:** each architecture change ablated against fixed compute, exploitability comparison plotted. No change ships without the comparison.

The discipline this enforces: every change is a hypothesis tested against a single outcome metric, instead of being a step on a schedule chosen by intuition.

---

## Critical re-evaluation of each step

A pass through the plan to surface load-bearing assumptions, missing preconditions, and statistical-significance concerns. Fix the highest-impact ones before executing.

### Highest-impact issues (fix these)

- **CFR strategy is NOT on disk.** `cfr_ai/agent.py:28` reads from `cfr_ai/outputs/<hand_sizes>/<key>/<key>.csv`; that directory does not exist locally. CFR's training outputs likely live on S3 (the project has `cfr_ai/dispatcher` + Lambda deployment). The "head-to-head vs CFR-from-CSV" pillar of Phase 0.1 has an undocumented precondition: either regenerate CFR for the configs we want to evaluate, or fetch the production artifacts. Before Phase 0 starts, locate or rebuild CFR data for at least the 1v1 24-card-deck no-jokers config — that's the regime where CFR is near-Nash and where the comparison is most informative. If unavailable, fall back to a snapshot-pool league as the strong baseline (and accept that "is the agent strong vs Nash?" can't be answered).
- **CFR is only valid for a narrow regime.** Even with CSVs, CFR was solved at small abstractions and likely only for low-cardinality 2-player setups. Head-to-head vs CFR is a "is the agent at CFR's level in CFR's regime?" check, not a universal strength signal. The metric ladder must label which opponents are valid in which configs; conflating regimes will mislead.
- **Phase 0 wall-clock cost is underestimated.** The plan says "1–2 days." Realistic breakdown: head-to-head harness with 4 opponents at e.g. 1000 games each = ~5–10 minutes per checkpoint eval. Backfill on 10 checkpoints = ~1 hour. `tools/dashboard.py` (Streamlit, ~80 lines) ≈ half a day if done well. `tools/run_manager.py` (start/status/override/stop/list/compare with proper background-process management and atomic CSV reads) ≈ 1–2 days. Late-game decision tree retrains (0.5c, 0.5d, possibly 0.5e) = ~1–3 hours each at ~5M steps. Phase 0 realistically takes **3–5 working days**, not 1–2.
- **Eval-during-training overhead.** If training runs at ~4.4k steps/sec, 50k steps takes ~11 sec; running a 5–10-minute head-to-head eval every 50k steps is a ~30× slowdown. Either run eval less frequently (per 1M steps), use a much smaller game count per eval and larger windows for averaging, or keep the snapshot-pool eval offline-only. Specify the cadence and game-count up front; don't discover it during the run.
- **Statistical significance is unaddressed.** A 1000-game head-to-head winrate of 51% has roughly ±3% confidence interval. Many comparisons (e.g. "Phase 1.1 retrain matches or improves Phase-0 baseline") will be within noise. Need: (a) explicit N to target a chosen detectable effect size, (b) multiple seeds for any retraining-based claim, (c) a "minimum detectable difference" threshold below which we declare "no signal."
- **Backfill across heterogeneous checkpoints is risky.** The ~250 saved checkpoints span PR #40 (changing the action mask), schedule revisions, and possible config drift. A "trend over training step" plot mixes all of those. Frame backfill as "snapshot of current state of past checkpoints," not as a learning-progress curve.
- **`_nstep_weight_scale=0.5` and `_nstep_terminal_boost=0.5` are active.** The earlier exploration agent reported these as "optional, default 0.0" — but `_apply_phase_schedules` (`agent.py:912-915`) sets both to 0.5 by default. So the within-round MC shaping has *additional* weighting beyond γ^d, multiplying earlier-step rewards by `1 + scale·k + boost·k`. This compounds with the schedule and was never separately validated. Phase 1.1's "rename to MC return shaping" must explicitly decide to keep, ablate, or delete this weighting.

### Per-step critique

**0.1 Build the metric ladder.**
- "Snapshot pool" never specified: pool size, snapshot cadence, retirement policy. Without these, "self-improvement signal" is impossible to compute reproducibly. Recommend: 8 snapshots, every 1M steps, FIFO retire when full.
- Multi-player evaluation undefined. In a 3-player game, what does "head-to-head vs CFR" mean? All-other-seats CFR? One CFR + one ConservativeAgent? Mixed pool? Pick a convention and stick with it.
- Target N for statistical power not specified. 1000 games detects ~3% margins; 10000 games detects ~1%. Pick based on the smallest difference you'd want to act on.

**0.2 Backfill on past checkpoints.**
- See "heterogeneous checkpoints" above. Frame as snapshot, not trend.

**0.3 Wire into training loop.**
- See "eval-during-training overhead" above. Specify cadence (every 1M env steps?) and game-count budget per eval.
- Atomic CSV writes: pandas reads can race with appends. Use append-only line writes from training, and read-with-retry from the dashboard.

**0.4 Metric-based hyperparameter triggers.**
- This is the vaguest item in the plan. "Trigger framework expressed as control-plane rules" gives no concrete rules. Risk: this turns into endless tuning of meta-rules that themselves need an outcome metric.
- Honest recommendation: **demote to manual operator overrides** for now. I (Claude) read the dashboard, decide an override is warranted, write to control-plane JSON. No automated trigger framework yet. Build automation only after we know which manual interventions matter.

**0.5 Late-game diagnosis decision tree.**
- 0.5a (buffer-distribution diagnostic): requires the buffer to expose card-count per stored transition. The buffer stores `obs`, which encodes seat counts (including agent's own card count). Extract from the appropriate slice of the obs vector — possible, but not zero-effort. Plus the SL reservoir is a *Vitter sample*, not the full visited distribution; histogram is biased toward whatever the reservoir kept.
- 0.5b (overfitting test via train↓ eval↔): "eval" here is head-to-head winrate, not held-out test loss. The mapping from declining winrate to "overfit" is loose. Could also be: opponent improving (snapshot pool), CFR exploiting the agent more effectively as agent specializes, distribution shift from curriculum advancement. Be explicit about confounders.
- 0.5c (curriculum redesign): "sample max_cards uniformly" mid-train will dump max_cards=11 onto an agent only trained at low cardinalities → catastrophic divergence. Safer formulation: keep the env curriculum but reweight buffer sampling toward high-card-count transitions. Or: train two seeds from scratch, one with current curriculum, one with uniform max_cards from step 0.
- 0.5d (schedule-freeze): "freeze at initial values" is an unfair test — initial values are exploration-heavy and meant to anneal. Fairer: freeze at *current/mid* values, where the agent is supposed to be operating. Better still: ablate one schedule axis at a time (η, ε, lr) to identify which axis is locking in the failure.
- 0.5e (width sweep): plan says {128, 256, 512} but if the hypothesis is overfitting, also include narrower (e.g. 64) — smaller nets may generalize better with limited data.

**0.6 Run management + dashboard.**
- Process state across Claude sessions: if I launch a run via `Bash run_in_background`, the process persists but my handle on it doesn't survive a session ending. Need a PID file at `runs/<run-id>/pid` so any future Claude session can rediscover and manage existing runs.
- "Stop" semantics: must trigger checkpoint flush + control-plane finalize, not just SIGKILL. `nfsp_run_local.py` should install a SIGTERM handler that flushes state.
- Streamlit auto-refresh polls files. Atomic writes from the trainer side are not optional.
- I haven't specified my polling cadence or the tool I'll use. Recommendation: `ScheduleWakeup` with delay tuned to checkpoint cadence (e.g. every 10–30 minutes for a long run); wake up, read latest CSV, decide action, sleep again. Don't busy-loop.

**1.1 Commit to MC, delete TD machinery.**
- "5M-step retrain matches baseline" needs a baseline definition. The current latest checkpoint is at 100M+ steps; comparing a 5M MC-retrain to a 100M scheduled run isn't apples-to-apples. Two options: (a) train an *unmodified-code* control from scratch for 5M as the baseline (fair comparison, doubles cost), or (b) accept the comparison is noisy and only require "non-catastrophic regression."
- Also need to decide: keep `_nstep_weight_scale` / `_nstep_terminal_boost` weighting (active by default at 0.5)? Ablate? Delete? Should be an explicit sub-bullet.
- "burst RL updates on reward" logic (`burst_rl_updates_on_reward=2`, threshold 0.5) is a related orthogonal mechanism that interacts with the RL update path. Verify it still makes sense under MC, or delete it as part of 1.1.

**1.2 SL gating.**
- Steady-state η ≈ 0.10 means the SL reservoir fills ~10× slower after the gate. Early SL training is now data-starved. May want to keep the cooldown / warmup logic or extend warmup_steps when 1.2 lands.
- Verification should also assert: at η=0.5, fraction of SL samples that came from BR is ≈ 1.0 (all of them), not just count-match.

**1.3 Drop public-prior mask.**
- Action set widens. ConservativeAgent's `random.choice(legal)` fallback pool grows (currently it falls back when its action is mask-zeroed by the prior). Eval comparison vs old ConservativeAgent is now slightly different game-theoretically. Recompute baseline for the new action set.
- Possible regression: agent now has to learn that low-prior bets are bad rather than being unable to take them. Initial training will likely degrade before recovering. Plan must say "regression in the first ~1M steps is acceptable; reject only if 5M-step exploitability is materially worse than baseline."

**2.1 Factorize action head.**
- 13 set-types but with incompatible parametric details (pair has 6 ranks; flush has 4 suits; full house has 30 (rank, rank) combos; two-pair has 15; straight has 1–3 categories per deck). Either one shared "detail" output of size max(detail_count) with masking, or a separate head per set-type. Shared+masked is simpler; per-head is cleaner. Pick one explicitly.
- Cross-deck transfer claim is overstated: 24-card has 1 straight category, 32-card has 3 (small/big/great). Heads aren't trivially shared.
- Implicit assumption: factorization fixes CHECK underexploration. If softmax over set_types still rarely outputs CHECK at init, the curriculum bandaid still earns its keep. Verify with a small ablation before committing.

**2.2 Wire embeddings.**
- Card embedding pretraining task is "set-existence classifier" (per `docs/embedding_pretraining_plan.md`). That objective may not produce representations optimal for policy/Q. May need to fine-tune during NFSP, or pretrain may be partially wasted.
- obs_dim changes when embeddings replace multi-hots. Hidden width and downstream dim plumbing need coordinated updates. Compounds with 2.3.

**2.3 Hidden-width sweep.**
- "Fixed total compute budget" needs a unit: wall clock? FLOPs? env steps? Pick one. If wall clock, larger nets get fewer env steps — explicitly account for that.
- If 0.5e was already done, 2.3 is redundant. Reconcile.

**2.4 Snapshot-league self-play training.**
- Memory and disk: each checkpoint is a few MB but a pool of 10+ adds up; loading multiple opponents into memory simultaneously affects throughput. Estimate before committing.
- Pool sampling distribution affects training: uniform over snapshots biases toward old, weak versions; recency-weighted may not help diversity. Pick a policy.
- May make the agent better at average-population play but worse at any specific opponent (wash-out effect).

**Phase 3 — Defer.**
- 3.1 (refactor): trigger condition unclear. Recommend: refactor only after Phase 2 lands and metrics stabilize for at least 1 retrain cycle, so diff signal can be separated from refactor-introduced bugs.
- 3.2 (delete schedule): the only path to deleting it is for metric-based triggers (0.4) to demonstrate they're better. If 0.4 stays manual (per my recommendation above), 3.2 stays indefinitely.
- 3.3 (algorithm switch): low likelihood given the bug list; correct to defer.

### Cross-cutting concerns

- **Multiple seeds.** Any "X improves on Y" claim from a single retrain is unsupported. Budget 2–3 seeds per claim or accept that single-seed results are exploratory.
- **Confounders during retraining.** Phase 1 retrains run with the existing schedule, conflating "fix X helps" with "fix X + schedule helps." For clean attribution, also run with schedule frozen / null; otherwise note that signals are mixed.
- **NFSP appropriateness.** The plan acknowledges in 3.3 that NFSP convergence isn't guaranteed in multi-player. This is a foundational risk, not a deferred item. If your goal is multi-player play and Phase 0–2 plateau low, the answer is "try a different algorithm," and that's months of work. Worth surfacing now so it's not a surprise.
- **Risk register absent.** Possible bad outcomes the plan doesn't address: (a) metric ladder reveals all checkpoints are weak → demoralizing but actionable; (b) Phase 1.1 retrain fails to converge → need to revert vs debug; (c) buffer-distribution diagnostic shows late-game is fine in the buffer → primary hypothesis dies, decision tree must continue; (d) CFR strategy unrecoverable → strong baseline gone, ladder degraded. Plan should pre-commit to responses for each.
- **`agent.py:1239`** (unreachable code after `raise`) is harmless but indicates the file has gone through fast iteration without code review. Real risk: more bugs of similar shape lurk. The 1862-line refactor (3.1) is partly justified by needing to find these.

### Adjusted recommendations

- **Phase 0 split into 0a (infrastructure) and 0b (diagnostics).** 0a = run_manager + dashboard + metric harness. Land first. 0b = backfill + decision tree + trigger framework. Land after 0a is functional.
- **CFR availability is a precondition.** Resolve before any Phase 0 work starts: locate or regenerate CFR data, or downgrade the metric ladder.
- **Drop 0.4 (auto-trigger framework) for now.** Manual overrides via control-plane JSON. Revisit only after the metric ladder identifies which interventions matter.
- **Run two seeds for each Phase 1 retrain.** Cost doubles; signal becomes interpretable.
- **Add explicit risk responses** at the bottom of the plan or per-phase, so failures don't trigger ad-hoc rework.
