# Dead ends — experiments that failed, and why

Each entry: what was tried → what happened → root cause → the rule it left behind.
These cost real time. Don't pay for them twice.

---

## A. Reward / credit-assignment traps

### A1. Actor-relative reward → `avgR ≈ 1` that means nothing (historical, now fixed)
Early `step()` paid reward only at CHECK with `done=True`, `+1 if loser != actor`.
Because the *actor* is whoever just pressed CHECK, reward was systematically credited
to the acting side → `avgR` pinned near **1** regardless of skill, and the agent drifted
to **spam-CHECK / panic-CHECK**.
- **Fix adopted:** fixed-perspective reward from a `ref_nick` chosen at `reset()`
  (`+1 if loser != ref_nick else -1`, zero-sum); per-round episodes; `done` only at
  round end (not on every CHECK).
- **Rule:** In balanced self-play, **`avgR ≈ 0` is correct** (symmetry) — don't "fix" it.
  **`avgR ≈ 1` means the actor-credit bug is back**, not that you're winning. The graph
  is not your KPI; deterministic eval vs baselines is.

### A2. Don't backfill/n-step on top of broken reward
Sparse terminal reward is propagated by n-step (n≈5–10) or **round-level MC backfill**
(overwrite the last K=5–10 transitions with the final fixed-perspective reward). If you
do this *before* fixing actor-relative reward, you just amplify the bug. Precondition:
reward must already be fixed-perspective.

### A3. Reward shaping outside [0.7, 1.4] → policy collapse (the canonical disaster)
**kupala, 4 documented attempts** (`5ab1c27` = "v4"). v1 `win=1.8, loss=0.5,
forbid_check_unless_only` collapsed to **2.65% vs baseline** (53/1947), mean game length
2.0 actions. Even gentle `win=1.4, loss=0.7` collapsed (0.026 vs baseline).
- **Root cause:** the dampened-loss MC return makes max-bet bluffs EV-positive even when
  they lose 95% of the time → escalation death-spiral; the Nash attractor punishes
  asymmetric shaping. `hand_type_bias` is **additive per matching bet → it compounds**,
  so "small" biases aren't small.
- **Rule:** keep **effective reward scale in [0.7, 1.4]**. The **aggression direction
  (`win>1, loss<1`) is not a tunable knob in this game** — bake aggression nowhere; keep
  "snowball/aggressor/reveler" archetypes as *sculpted overlays only*.

### A4. `win_multiplier > loss_multiplier` biases toward CHECK, not aggression (counter-intuitive)
**czernobog** was specced Terror-aggressive (`win=1.5, loss=0.6, +0.4 bias on SF/4oak,
bluff_caught=-0.2`) and converged to the **most passive** trained bog (CHECK +25pp,
bluffs −30pp). In Blef, CHECK is often the highest-EV action, so asymmetric reward
amplifies every winning path *including* catching bluffs.
- **Rule:** `loss_multiplier < 1` → **more** checking; `> 1` → **less** checking.
  **Never reward high-tier bets** — big claims are *exposure, not power*; the strike is
  the CHECK (`bluff_call_bonus`).

---

## B. Personality-mechanism dead ends

### B1. `forbid_hand_types` on low tiers → opener traps
The action mask is HARD at train **and** inference. Forbidding the LOW tiers removes the
only legal opening bets in 1–3-card rounds → forced bluffs → near-unwinnable.
- **czernobog** forbids bottom-3 (HC/Pair/2P) — **still broken** (game-wr 0.04–0.16 at
  1v1, opener trap since Jun 4).
- **triglav (broken)** forbade {HC, Pair} → **0.000 game-wr vs dazhbog**.
- **mokosh (broken)** forbade top-4 → round-wr ~0.5 but game-wr **0.010 vs dazhbog**,
  collapsing 0.58→0.01 from 3→11 cards.
- **Rule:** never forbid the bottom two tiers. Fix recipe that worked (triglav):
  **forbid {High card} only + C-blind `private_priors`** → row-mean 0.667.

### B2. Mechanism E aggression/passivity bias → trait inversion (failed twice)
`opponent_aggression_bias` / `opponent_passivity_bias` resample the opponent to a single
deterministic action. The learner overfits the degenerate distribution and the **trait
inverts**: mokosh `aggression=0.8` → 0.054 vs baseline, bluff share 0.877 (a *honest*
read of an all-in opponent); triglav `passivity=0.8` → CHECK 0.525, bluff 0.179.
- The "smarter" **quartile resampler** (sample from upper/lower quartile of legal bets)
  was tried next — **also inverted**; it's **uncommitted in the working tree**, bound to
  nothing. mokosh.json/triglav.json were reverted.
- **Rule:** only **honesty** bias works in mechanism E (truthful spans a diverse action
  set; single-action resamplers don't). Don't retry aggression/passivity via resampling.

### B3. Mechanism D (`trust_history` inference override) — empirically inert
A/B at n=2000 across 4 slices: all deltas within 95% CI (`.trust_eval/results.csv`:
domovoi 0.362 OFF vs 0.343 ON; kupala 0.303 vs 0.316). A Nash policy is robust to
single-channel input distortion because the other truth signals are correlated.
- **Rule:** kept as a tested primitive but **bound to no personality**. Don't build on it.

### B4. mavka `loss_mult 2.5` — out of band (latent)
`win=0.8, loss=2.5` → ~34.6% (playable but weak) and far outside [0.7, 1.4]. Listed as a
latent breakage to fix with the triglav recipe.

---

## C. Architecture / training dead ends

### C1. Factorized action head + snapshot-league ("treatment", May-8 bake-off)
Regressed *with more training* (cfr_mean 0.481@5M → 0.453@9M) and lost to plain "control"
on every metric (by 0.014/0.022/0.031). The league pool fills with weak historical
policies and drags opponent quality down.
- **Rule:** **retired.** The control recipe (W256 + embeddings + η/ε pin, MC Q-regression)
  is enough. Don't reintroduce a self-play league expecting gains.

### C2. Vestigial TD / target-net machinery
`fbbd720` committed Q-loss to **Monte-Carlo regression with NO bootstrap/target net**.
The leftover `tau`/EMA/`hard_target_interval`/n-step-TD knobs are **vestigial for Q**.
- **Rule:** don't reintroduce bootstrapping assuming the old advice files (which
  recommend Double-Q + Polyak) are current. They were superseded.

### C3. Triangular RIH (randomize-initial-hands)
Implemented but **costs early-game**; not in the SOTA recipe (used selectively, e.g. 1v1
mokosh distinctness retrain, not as a default).

---

## D. Eval-interpretation traps (these make a broken model look fine)

- **Round winrate ≠ game winrate.** A B-mask bog can sit ~0.50 round-level and lose ~99%
  of *games* (it accumulates cards toward elimination). Always check game-level + per-N
  depth columns.
- **In-training `EVALUATION` on 3M fine-tunes only exercises `max_cards ≈ 2`** — it
  **cannot** detect deep-round or game-level breakage. The trainer's eval opponent can be
  a frozen self-snapshot → misleading ~100% when both play the same degenerate strategy.
- **"vs CFR" winrates are contaminated** by CFR's hardcoded fallback (action 87/88) on
  info-set misses: **44–47% miss at `mc=1` (the noisiest, not mc=11)**, ~27% at mc=11.
  CFR is **24-deck only** → any 32-deck "vs CFR" number ≈ random opponent; strip it.
- **morana CFR pantheon cache never auto-invalidates** (`bog_version("morana")` is the
  constant `delegated:cfr`). After any CFR swap, only re-run cells purge; 24-multi/
  24-team/32-deck morana cells silently hold stale results. Purge keys manually.

---

## E. Operational burns (process, not modeling)

- **Prod swap on a 2p-only slice (2026-05-08).** Declared `control-15M` "SOTA" and tried
  to copy it over prod — it was trained only on `(2p, deck=24, j=0, b=0, cc=0)`; prod
  serves 3p+, jokers, blanks, common cards. Would have regressed coverage. **Never swap
  prod without full deployment-surface coverage; never say "SOTA" without naming the slice.**
- **Redundant retrain (2026-05-09).** Launched a 12M run when an existing final ckpt
  already covered the slot. **Inventory `runs/` and `df -h` first** — you can often export
  from an existing ckpt in seconds instead of retraining for hours.
- **Shared control-plane file (2026-05-09).** Two trainings shared one `--control-plane`;
  tuning one perturbed the other. **One control file per training.**
- **Parallel dependent FS ops (2026-05-31).** Batched reorg + `mv` + upload in one block;
  they raced and corrupted the `games/` tree. **Sequence one step per turn.**

See [`04_deployment_and_ops.md`](04_deployment_and_ops.md) for the positive rules.
