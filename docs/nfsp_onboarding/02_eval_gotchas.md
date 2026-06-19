# Eval gotchas + the evidence map

The metrics are adversarial — most of them can make a broken model look healthy. This is
where the hours hide.

## The traps

- **Round winrate ≠ game winrate.** The single biggest one. A B-mask bog can sit ~0.50
  round-level and lose ~99% of *games* (mokosh-vs-dazhbog: **0.58 game-wr @ 3 cards →
  0.01 @ 11**). It accumulates cards toward elimination round by round. **Always check
  game-level winrate AND the per-N depth columns.**
- **Pin `max_cards = 11` for any personality/spec gate.** The 1→11 curriculum and the
  in-training `EVALUATION` line on 3M fine-tunes only exercise `max_cards ≈ 2` — they
  **cannot** see deep-round collapse. Use the per-run control plane to clamp.
- **`tools/pantheon_matrix.py --n-games` is a misnomer** — it counts **ROUNDS/episodes**,
  not whole games. Whole games ≈ n/18 (n=5000 → ~280 settled games/cell).
- **Learner / seat-0 is the forced opener** (structural disadvantage). **Rank only on
  balanced matrices.** `tools/seat_split_matrices.py` is 1v1-only.
- **Gate vs the BASELINE snapshot, not the trainer's `EVAL` print.** Use
  `python -m tools.eval_ladder --opponents snapshot --snapshot-paths <baseline>` (~2k
  games). Heuristic: ≥~25% vs baseline = playable; ~25–45% = the trait is paying for
  itself; **<~10% = collapse, redesign the spec.**
- **The trait card (`tools/personality_trait_card.py`) shows only action distribution +
  bluff rate** — a policy that loses every game can have a clean card. Treat
  winrate-vs-baseline as a separate, required check.
- **"vs CFR" numbers are contaminated.** CFR falls back to action 87/88 on info-set
  misses: 44–47% miss at `mc=1` (the noisiest depth, not mc=11), ~27% at mc=11. CFR is
  **24-deck only** → any 32-deck "vs CFR" ≈ random opponent. Strip these.
- **morana CFR cache never auto-invalidates** (`bog_version` constant `delegated:cfr`).
  After a CFR swap, only re-run cells update; purge non-24-1v1 morana cache keys manually
  (`runs/.pantheon_matrix_cache/`).
- **`tools/behavioral_uniqueness.py` is mask-direction-blind** (5-dim behavior z-distance).
  `tools/audit_phase3.py` is heavy (~45 min).

## Evidence map — which dir proves what
**Top-level `eval_results/` is STALE (≤ Jun 1). Ignore it.** The real evidence is in
hidden `runs/.*` dirs:

| Dir | What it is | Headline result |
|---|---|---|
| `runs/.claude_overnight/` | May-8 SOTA bake-off (richest narrative) — `FINAL_REPORT.md`, `SUMMARY.md`, `SOTA_C-5M_pivot.md` | control-15M = SOTA: 0.499 vs CFR (Nash parity), 0.564 vs Conservative, NashConv 0.0147. Treatment (factorize+league) lost. CFR fallback contamination quantified. |
| `runs/.pantheon_matrix/` | 17×17 round+game matrices, per-N columns, heatmaps | `matrix_deck24_1v1_*_n5000.csv`. morana V3.2x4 rerun → game row-mean 0.30→0.83 (24_1v1). Pre-rerun backup in `pre_cfr_v32x4_backup/`. |
| `runs/.pantheon_matrix_cache/` | per-cell JSON cache | morana key constant `delegated:cfr` → stale-CFR cache hits off 24_1v1. Purge before trusting. |
| `runs/.behavioral_uniqueness/` | behavior z-distance, dendrograms | morana most-unique (~7.3); lowest pair domovoi↔rusalka 0.65. Mask-blind. |
| `runs/.phase3_audit/`, `.phase3_redos_*` | trait verdicts — does behavior match spec? | `trait_verdicts.csv`. Many misses flip to OK once the B-mask is applied (`.phase3_redos_masked`). |
| `runs/.B_C_pivot/` | Jun-9 B/C mechanism pivot (the masks+blind-spots retrain) | fixed triglav/mokosh/domovoi/rusalka. |
| `runs/.distinct_pivot/` | Jun 10–11 distinctness retrains (6M, max_cards 11) | exports → `artifacts/staging/`; promote only after game-level gate. |
| `runs/.bog_program/` | Jun-12 A/B spec search | `baseline_reference.txt`, `panel_results.csv`, `specs/`, `retired_domovoi/`. |
| `runs/.E_quartile_poc/`, `.kupala_e_poc/`, `.kupala_pureE/` | mechanism-E proofs | E alone is weak (~0.30–0.35 vs conservative); doesn't deliver the trait by itself. |
| `runs/.trust_eval/` | mechanism-D proof | inert (deltas within CI). |
