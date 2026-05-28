# Bot Personalities Plan — Slavic-pantheon play-style bots

## Validation results (160 games/god vs the standard NFSP baseline, deck 24)

Measured with `tools/personality_winrate.py` (full games via the training manager
`simpleschema_local_manager`) and `tools/personality_probe.py` (move signatures).
Calibration: baseline-vs-baseline 46.9% (≈50%, slight seat bias). Heuristic floor:
`conservative_ai` 20.0%, `conservative_crawling` 22.5%.

| tier | gods (win% vs baseline) |
|---|---|
| wall / strong (≈baseline) | perun 51, svetovid 52, dazhbog 52, mokosh 50, triglav 49, porevit 48, zorya 46, kupala 44 |
| mid | mavka 32, rusalka 30 |
| loud / easy | czernobog 28, veles 28, poludnica 26, leshy 24, domovoi 22 (crawling), baba_yaga 22 |

Every god lands in 20–52% — weaker-but-competitive, none a free win, none constantly
losing (tempo/chaos bounded). Move signatures confirm intent: disciplined gods (mokosh,
svetovid, veles) low-bluff + concealed; loud gods (czernobog, baba_yaga, leshy) bluffy +
chaotic but coherent; mood cohorts fire correctly (poludnica spikes late −0.27 pPriv; perun
bolder when ahead; zorya swings by round parity). Chaos is in-distribution top-k sampling,
not temperature (temperature collapsed the peaked NFSP policy toward random).

## Context

We have strong, near-equilibrium Blef AIs: 6 NFSP specialists (deck 24/32 × 1v1/multi/team,
served by one ECR container `blef-aiagent-nfsp` with internal variant routing) and a separate
CFR bot (`blef-aiagent-cfr` → 66 hand-size workers, 1v1/24 only). We have art + animations for
~17 named deities and want each to feel like a distinct *character*: different difficulty AND
recognizable quirks ("don't trust Morana when she goes quiet"; "Perun always overcommits late").

The constraint is cost: we must not upload N Docker images to ECR. The decision (made with the
user) is **Path B** — keep **one** NFSP image and **one** function `blef-aiagent-nfsp`; the
personality is selected per move by a field in the payload. On-demand only (no provisioned
concurrency), so a shared warm pool that loads the model once is the efficient choice.

Two anchors are fixed:
- **Morana** = the CFR bot (Nash-like, minimal-tell, patient) — *already a perfect Ice Queen*. No
  work; she stays on `blef-aiagent-cfr` and can only be seated in 1v1/24 games.
- **Perun** = the strong NFSP baseline (the difficulty wall), with a light Warlord tint.

Everything else is **inference-time sculpting** over the *shared* NFSP weights — no retraining.
A personality is a small config (a knob vector + optional mood rule), not a new artifact.

## Design: two orthogonal axes

- **Difficulty** (how strong) — emerges from how far a bot is pushed off equilibrium plus chaos.
  Biasing a near-Nash policy *is* deviating from it, so quirkier ⇒ more exploitable ⇒ easier. The
  two unbiased bots (Perun=NFSP baseline, Morana=CFR) are the hardest *by construction*.
- **Personality** (how flavored) — primarily the *choice of NFSP policy* (below), then a bounded
  output overlay + a stateless "mood" that shifts the knobs over a game.

### Policy source (NFSP-native — the PRIMARY style axis)

The probability overlay in the next section, used alone, is just a parametrised heuristic
(essentially the existing `conservative_ai`): crank it and the NFSP signal washes out. To get
genuinely *NFSP-based* playstyles the model itself is the primary axis; the overlay is a bounded
perturbation on top. NFSP-native levers (no retraining — both heads already live in every inference
artifact: `export_inference` saves `q` and `pi`, and `select_action(use_average_policy=False)` runs `q`):

- **Head** — `π` (average policy: balanced Nash approximation, prod default) vs `Q` (best-response:
  sharper, greedier, more exploitative *and* more exploitable). A real, free style switch.
- **Model temperature** `τ_model` — softmax temperature on the chosen head's *own* logits. Low ⇒
  decisive; high ⇒ mixes across what the *model* deems reasonable (coherent, in-distribution
  "chaos", unlike uniform noise).
- **Checkpoint** — an earlier training step is a genuinely weaker, less-refined NFSP policy: a free
  skill tier / literal "young" god (requires exporting that checkpoint, ~1.5 MB each).
- **(Tier 3, deferred)** reward-shaped retraining for one flagship.

So **personality = source (`π` / `Q` / weak checkpoint, or `cfr`, or the pure `conservative`
heuristic) + `τ_model` + a bounded overlay + mood.** Strong gods (Perun, Svetovid, Mokosh, Veles)
are NFSP-dominated with a light overlay; easy/funny gods (Leshy, Porevit, Czernobog) lean on the `Q`
head, high `τ_model`, a weak checkpoint, or larger overlays. A deliberately-simple god (Domovoi) may
bypass NFSP and route to `conservative_ai`.

### Output overlay (bounded, secondary — applied to the chosen head's logits before selection)

The 88 sets are ordered only by **seniority** — the legality ladder. On your turn you may bet a
**strictly more senior** set than the last bet, or **check** (assert the last bet is false). There
is no showdown/comparison; on a check, everyone reveals and the only question is whether the claimed
set *exists among all pooled cards*. **Seniority index is NOT probability** — P(a set exists) is
highly non-linear in the index, so knobs operate in **probability space**, never on the index.

We use **two** probability vectors, both already in the repo:
- **private** `p_priv(a) = get_bet_probabilities(game_state, for_betting=True, last_bet=...)` —
  `P(set a exists | my hand, others' card counts)`; `specific_action_id=last` gives `p_priv(last)`
  for the check decision.
- **public** `pub_prior(a) = get_generic_bet_probabilities(game_state, ...)` — plausibility given
  only public info (card counts, common cards, deck), already returned by `vectorize_obs`.

The gap between them is strategically central: betting strictly along `p_priv` **leaks your hand**
(the rules README names this tension — *"Alice doesn't want other players to know her cards"*).
Betting along `pub_prior` is *concealing*: claims look generic and reveal little. The NFSP baseline
already balances this near-equilibrium; the knobs deliberately distort it for character.

| Knob | Meaning | Effect on logits |
|---|---|---|
| `risk` | honesty vs bluff (private) | bias bets by `−risk·z(p_priv(a))`: `risk>0` ⇒ bluffy (claims sets unlikely true → loses if checked), `risk<0` ⇒ honest (claims likely-true sets). Drives bluff *frequency* and *depth*. |
| `guard` | concealment vs readability | bias bets by `+guard·z(pub_prior(a))`: `guard>0` ⇒ unreadable (bets to public expectation, hides the hand), `guard<0` ⇒ naive/leaky (bets its private truth openly → exploitable). |
| `susp` | check threshold (private) | check when `p_priv(last) < θ(susp)`: `susp>0` ⇒ paranoid (challenges plausible bets), `susp<0` ⇒ trusting. Check-logit bias ∝ `(θ − p_priv(last))`. |
| `tempo` | betting cadence | the **only** index-based knob, purely cadence: `tempo<0` ⇒ minimal legal raise (patient), `tempo>0` ⇒ big senior leaps. Not a truth proxy. |
| `chaos` | randomness | raises `τ_model` (temperature on the head's *own* logits) + samples instead of argmax — coherent, in-distribution mixing, not uniform noise. |

Total bet bias `= −risk·z(p_priv(a)) + guard·z(pub_prior(a))`. `z(·)` normalizes over the legal bets
per state. Biases hit **legal** entries only, then the mask is re-applied (never lift an illegal
action). Selection: `argmax` when `chaos == 0`, else sample at temperature τ.

### Mood (stateless, derived from the per-move `game_state`)

- `tilt` = my `n_cards` deficit vs table (behind ⇒ some bots tilt up, others tighten).
- `phase` = total cards remaining / round number ⇒ early / mid / late knob sets.
- `momentum` = recent round outcomes parsed from `history` ⇒ ramp after wins (Kupala).

**Honest limits:** true cross-*game* opponent modelling ("this human bluffed 3 games ago") is NOT
feasible — fixed policies don't learn you online. "Reader"/"manipulator" gods (Svetovid, Veles,
Baba Yaga) are realized as *pseudo-adaptivity*: nudging knobs off `pub_prior` and the current
round's `history`, not learning. Deeper authenticity for one flagship (Veles) is a deferred
Tier-3 retrain, out of scope for v1.

## The roster (v1 knob vectors, scale −2..+2; chaos 0..2)

| Deity | Engine | Archetype | risk | guard | tempo | susp | chaos | Mood | Difficulty |
|---|---|---|---|---|---|---|---|---|---|
| **Morana** | CFR | Ice Queen | — | — | — | — | — | — (equilibrium) | Hard (wall) |
| **Perun** | NFSP | Warlord | +1 | −0.5 | +1.5 | 0 | 0.2 | double-down after surviving a check | Hard |
| Czernobog | NFSP | Terror | +2 | −0.5 | +1.5 | +0.5 | +1 | — | Med |
| Svetovid | NFSP | Oracle | −0.5 | +1 | 0 | +1.5 | 0 | sharp check threshold on `p_priv(last)` | Hard |
| Mokosh | NFSP | Weaver | −0.5 | +1 | −2 | +0.5 | 0 | tighten late | Med-Hard |
| Triglav | NFSP | Strategist | phase | phase | phase | 0 | 0 | safe → base → aggressive by phase | Med-Hard |
| Veles | NFSP | Serpent | phase | +2 | 0 | 0 | 0 | honest early → bluff late | Med-Hard |
| Dazhbog | NFSP | King | +0.5 | 0 | +0.5 | −0.5 | 0 | suppress check when behind (ego) | Med |
| Rusalka | NFSP | Siren | +1 | +1 | 0 | −0.5 | +0.3 | weak early → strike late | Med |
| Mavka | NFSP | Illusion | +0.5 | +1.5 | 0 | −0.5 | +1 | — | Med |
| Zorya | NFSP | Twins | ±1 by round | 0 | 0 | ±1 by round | +0.3 | dawn/dusk alternation by round parity | Med |
| Kupala | NFSP | Reveler | +0.5 | −0.5 | 0 | 0 | +0.5 | ramp after wins, drop after losses | Med |
| Baba Yaga | NFSP | Mind-gamer | +1 | +1 | spiky | +0.5 | +1.5 | structured "irrational" spikes | Med |
| Poludnica | NFSP | Fever | phase | 0 | 0 | 0 | phase | passive → explosive spike late | Med |
| Porevit | NFSP | Youth | +1 | −1.5 | −0.5 | 0 | +1.5 | chaos decays late ("learns") | Easy |
| Leshy | NFSP | Trickster | +1 | 0 | 0 | 0 | +2 | per-move random chaos scaling | Easy |
| Domovoi | NFSP | Caretaker | −1.5 | −1 | −1.5 | +0.5 | 0 | +susp vs over-claimers (trap) | Easy-Med |

**Source overrides** (default head `π`): `Q` head — Perun, Czernobog, Dazhbog; weak checkpoint —
Porevit; high `τ_model` — Leshy, Baba Yaga; pure `conservative_ai` (no NFSP) — Domovoi. Morana = `cfr`.

(Values are the starting point; final tuning happens against the behavioral harness below.)

## Implementation

### A. AI repo (`blef_ai`) — sculpting mechanism (local, reversible)

1. **`nfsp_ai/personalities.py`** (new):
   - `@dataclass PersonalityConfig` — `source` (`pi`/`q`/`cfr`/`conservative` + optional checkpoint
     id), `tau_model`, the overlay knobs (`risk`/`guard`/`susp`/`tempo`/`chaos`), + optional
     mood/phase rules.
   - `PERSONALITIES: dict[str, PersonalityConfig]` — the 16 NFSP entries (15 sculpted + Perun).
     Missing/unknown id ⇒ `None` ⇒ baseline behavior (backward compatible).
   - `derive_mood(game_state) -> dict` — stateless tilt/phase/momentum.
   - `sculpt_logits(head_logits, mask, cfg, *, p_priv, pub_prior, p_last, deck_size, mood) ->
     Tensor` — applies the bounded `risk`/`guard`/`susp`/`tempo` overlay (+
     `GameRules(deck_size).check_action_id` for the check entry), then re-applies the mask.
2. **`nfsp_ai/agent.py`** — add `policy_logits(obs, mask, *, head="pi", tau=1.0) -> Tensor`:
   masked logits from the chosen head (`pi`→`self.pi`, `q`→`self.q`) at temperature `tau`. Refactor
   `select_action` to reuse it. No behavior change to existing callers (defaults reproduce current).
3. **`nfsp_ai/production_agent.py:410-479`** — in `determine_action`: read the current player's
   `personality` from `game_state` (`players[cp].personality`); if absent/unknown ⇒ current greedy
   path. Else fetch cfg, select head/`τ_model`/checkpoint, compute `policy_logits(..., head, tau)`,
   then `p_priv = get_bet_probabilities(game_state, for_betting=True, last_bet=last)`, `p_last =
   get_bet_probabilities(..., specific_action_id=last)`, reuse `pub_prior` from `vectorize_obs`
   (line 460); pass to `sculpt_logits`, then argmax (`chaos==0`) or sample. `cfr`/`conservative`
   sources delegate to those existing agents instead.

### B. Game engine repo (`blef_game_engine`) — carry the personality (small change)

- `api/invite-aiagent.py`: let an `AGENT_MAPPING` value be `{"agent_type": "...", "personality": "..."}`;
  store `ai_agent = agent_type` (so the orchestrator still routes to `blef-aiagent-nfsp`/`-cfr`)
  and add `"personality": personality` to the player object (lines ~64-71).
- Set the `agent_mapping` env var: deity name → `{nfsp, <deity>}` for the 15 + Perun; Morana → `{cfr}`.
- `api/aiagent-orchestrator.py`: **no change** — it already forwards the full game state, so the
  new `personality` field reaches the lambda for free.

### C. Deploy (no new ECR images)

- AI: rebuild the single image via `nfsp_ai/scripts/deploy_lambda.sh`, push to the existing
  `blef-nfsp-lambda` repo, update `blef-aiagent-nfsp`. CFR/Morana untouched.
- Engine: redeploy `blef-invite-aiagent` and set the `agent_mapping` env var.
- **Hold for explicit go-ahead** — commit/push and AWS deploy are high-blast-radius.

## Verification

1. **Behavioral harness** (local, reuse the sim in `nfsp_ai/nfsp_run_local.py`): run each
   personality over a batch of states / self-play vs baseline and measure observable signatures —
   challenge rate, bluff rate (mean `p_priv` of chosen bets), readability (mean `p_priv − pub_prior`
   divergence of chosen bets), tempo (mean seniority jump), and win rate vs Perun. Assert each knob
   moves its metric in the designed direction and that difficulty ordering holds (Perun/Morana
   strongest; Leshy/Porevit weakest).
2. **Unit**: `sculpt_logits` never assigns probability to a masked action; unknown personality ⇒
   byte-identical to current baseline output.
3. **Smoke**: `determine_action` on crafted game states with a `personality` field for several
   deities → legal, distinct actions.
4. **End-to-end** (post-deploy, test game): invite a deity, confirm orchestrator → `blef-aiagent-nfsp`
   → legal move played.
