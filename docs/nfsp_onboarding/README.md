# NFSP onboarding — pick up Blef without re-running dead ends

> Reference for picking up NFSP training, personalities, and serving. Specifics
> (numbers, `file:line`, roster status) drift over time — verify against current code
> before relying on them.

**Audience:** a new collaborator taking over NFSP training, personalities, and serving.
**Goal of this folder:** save you the weeks that were already spent discovering what
*doesn't* work. Read [`00_dead_ends.md`](00_dead_ends.md) first.

## The one rule, if you read nothing else
The personality trait card and the in-training `EVALUATION` line both **lie**. A policy
that loses ~99% of *games* can show a fine action-distribution card and ~0.5 *round*
winrate. **Always gate on game-level winrate vs the baseline snapshot, with
`max_cards` pinned at 11** so deep-round collapse can't hide. See
[`02_eval_gotchas.md`](02_eval_gotchas.md).

## What Blef is (30 seconds)
Polish bluffing card game. Players hold private cards pooled face-down; they make
**escalating set-bets** ("there exist two pairs among all cards") that must strictly
increase in seniority. **CHECK** challenges the last bet → showdown over the union of
all hands → the loser gains a card. Exceed `max_cards` → eliminated. Tier order to
memorize: `High card < Pair < Two pair < 3oak < Full house < Flush < 4oak < Straight flush`.
RL view: each **round** is an episode; reward ±1 at CHECK with `γ^d` shaping (γ=0.666).

## Engines
NFSP (`nfsp_ai/`, the main one), CFR (`cfr_ai/`, **READ-ONLY**, external maintainer),
plus rule bots `conservative_ai/`, `conservative_crawling_ai/`, and `gpt_ai/`.

## Read in this order
1. [`00_dead_ends.md`](00_dead_ends.md) — failed experiments; do not repeat them.
2. [`01_personality_mechanisms.md`](01_personality_mechanisms.md) — bog training A–E, hazards, roster.
3. [`02_eval_gotchas.md`](02_eval_gotchas.md) — how the metrics fool you; the evidence map.
4. [`03_training_recipe.md`](03_training_recipe.md) — the load-bearing recipe, schedules, what's NOT built.
5. [`04_deployment_and_ops.md`](04_deployment_and_ops.md) — serving, prod-safety, training hygiene, scope.

## Fastest path to ground truth (the real evidence)
- `runs/.claude_overnight/FINAL_REPORT.md` + `SOTA_C-5M_pivot.md` — the May-8 SOTA bake-off and the CFR-contamination caveats.
- `docs/personality_deployment_surface.md` — current per-config standings + the deploy gate.
- `docs/nfsp_training_playbook.yaml` — canonical hyperparameters.
- **Hidden `runs/.*` dirs are the real eval evidence. Top-level `eval_results/` is stale — ignore it.**
