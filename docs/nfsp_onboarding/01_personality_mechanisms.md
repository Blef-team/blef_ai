# Personalities ("bogs") — training mechanisms A–E

16 NFSP personalities + morana (CFR), in `nfsp_ai/personalities.py`. Toolkit:
`nfsp_ai/personality_train.py`; specs: `nfsp_ai/personality_specs/<name>.json`.

## Two layers — never both at once
1. **Sculpt overlay** — inference-only logit shaping (`risk/guard/susp/tempo/chaos` +
   mood ramps, `BETA=0.40`). Used **only when no trained ckpt exists** for the routed config.
2. **Trained spec** — baked into the weights; mask/blind-spots reapplied symmetrically at
   serve, sculpting **skipped**. Export stamps `cfg["personality"]` (the stamped spec
   wins over the on-disk spec at serve time).

## The hard truth about the trait space
Balanced specs converge near the **Nash equilibrium attractor** (passive, honest,
simple-hand). The trained-personality *action-distribution* space is **narrower than the
spec space** — distinct traits exist but are subtle. The only lever that moves a policy
**meaningfully** off the attractor is mechanism **C**.

## Preference order (softest first)
**C → A-toward-caution → B → (A+B, risky) → never A-toward-aggression.**

---

## A — terminal reward scaling
Fields: `win_multiplier`, `loss_multiplier`, `bluff_caught_penalty`, `bluff_call_bonus`,
`hand_type_bias` (**additive per matching bet → compounds**). Fights the Nash attractor.
- **HAZARD:** keep effective scale in **[0.7, 1.4]** (kupala-v1 collapse). The aggression
  direction (`win>1, loss<1`) is structurally unstable — collapses even when gentle.
- Caution direction (`loss>1, win<1`) is safe → under-betting, defensible.
- **`win > loss` biases toward CHECK, not aggression** (see dead-end A4).
- **Never reward high-tier bets.** Positive `hand_type_bias` on SF/4oak is a red flag —
  *claims are exposure, not power*. Power = `bluff_call_bonus` + credible escalation.

## B — `forbid_hand_types` / `forbid_check_unless_only`
HARD action mask at train AND inference.
- **HAZARD:** opener traps (dead-end B1) — **never forbid the bottom two tiers**.
  `forbid_check_unless_only` removes the only bluff-punisher → collapse.
- Reliable when the forbidden tiers don't trap the opener (trait verdicts flip to OK in
  `runs/.phase3_redos_masked`).

## C — `obs_blind_spots`
Zeroed obs blocks, train + serve. A perceptual deficit the agent learns to play optimally
*within*. **Hardest to collapse; the only real off-attractor lever.** Only blinds that
strip non-trivially-recoverable features actually move the policy (e.g. leshy's
`private_priors + last_bet_prob`).

## D — `trust_history`
Inference-time prior override. **Empirically inert** (dead-end B3). Bound to nothing.

## E — `opponent_{honesty,aggression,passivity}_bias`
Training-time opponent resampling (≤1 per spec). Changes the **world model**, not the
reward; the learner's own seat is never resampled, so the trait reads as a coherent
worldview.
- **honesty works** (kupala full 6-variant grid; rusalka). **aggression/passivity fail**
  → trait inversion (dead-end B2).

---

## Roster status

| bog | status |
|---|---|
| kupala | trained, pure-honesty E, all 6 variants validated ✅ |
| triglav | FIXED (forbid {High card} + C-blind `private_priors`), all 12 slots promoted Jun 11 ✅ |
| mokosh | FIXED v3 (forbid {FH, 4oak, SF} — Flush left open as a terminal counter-raise outlet — + loss 0.8), all 12 slots promoted ✅ |
| rusalka, veles, baba_yaga, poludnica, leshy, zorya | trained, healthy |
| domovoi | trained ckpts **RETIRED 2026-06-13**; **shipped sculpt-only** (caretaker was too strong for the laughing-house-spirit art) → `runs/.bog_program/retired_domovoi/` |
| **czernobog** | **BROKEN** — bottom-3 opener-trap mask since Jun 4; triglav recipe likely fixes it |
| **mavka** | latent — `loss_mult 2.5` out of band |
| perun, svetovid | sculpted-only |
| dazhbog, porevit | **delegated heuristics — out of scope** (Conservative / ConservativeCrawling) |
| morana | CFR (V3.2x4); pantheon cache stale outside 24_1v1-plain |

> Validation of E-honesty (kupala): the trust signature (wr vs honest − wr vs bluffer) was
> consistently positive across all 6 variants (+0.011 … +0.087) — lower bluff rate, longer
> games, High-card/Full-house bet distribution. "The drunk who believes everyone is honest."

**The recipe that fixed triglav AND mokosh:** 6M fine-tune from `nfsp_inference_24_1v1.pt`,
`max_cards` **pinned 11** via per-run control plane (engine clamps per game; surfaces
deep-round breakage the 1→11 curriculum hides), staging export → **game-level pantheon
gate** → promote only on pass. Keep run dirs.
