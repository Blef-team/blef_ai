#!/usr/bin/env bash
# Build a team-aware eval CURVE by running tools.team_eval against every
# named-step checkpoint plus the latest .pt for one or more run dirs.
#
# Usage:
#   tools/team_eval_curve.sh DECK 24|32 RUN_DIR [RUN_DIR2 ...]
#
# Output: one line per (run, step) point — easy to eyeball as a text curve.
set -euo pipefail

DECK="${1:?deck size required}"
shift
RUNS=("$@")
PY="${PY:-/Users/adriangolian/work/blef_ai/venv_nfsp/bin/python}"
ROOT="${ROOT:-/tmp/blef_train2}"

CARD="/Users/adriangolian/work/blef_ai/artifacts/card_embedding_pretrain_${DECK}.pt"
HIST="/Users/adriangolian/work/blef_ai/artifacts/history_embedding_pretrain_${DECK}.pt"

GAMES=400
SEEDS=3

cd "$ROOT"
echo "step_label    team_wr team_rew indiv_wr  ckpt"
for run in "${RUNS[@]}"; do
  base=$(basename "$run")
  ckdir="$run/checkpoints"
  if [ ! -d "$ckdir" ]; then continue; fi
  for ck in $(ls "$ckdir"/nfsp_blef_*M.pt 2>/dev/null | sort -V) "$ckdir"/nfsp_blef.pt; do
    [ -f "$ck" ] || continue
    label=$(basename "$ck" | sed -e 's/nfsp_blef_//' -e 's/\.pt//' -e 's/^nfsp_blef$/latest/')
    out=$($PY -m tools.team_eval \
      --checkpoint "$ck" --deck-size "$DECK" \
      --n-players 4 --n-teams 2 --max-cards 11 \
      --games "$GAMES" --seeds "$SEEDS" \
      --card-embedding "$CARD" \
      --history-embedding "$HIST" 2>&1 | grep summary || true)
    twr=$(echo "$out" | sed -nE 's/.*team_wr=([0-9.+-]+).*/\1/p')
    trw=$(echo "$out" | sed -nE 's/.*team_rew=([0-9.+-]+).*/\1/p')
    iwr=$(echo "$out" | sed -nE 's/.*indiv_wr=([0-9.+-]+).*/\1/p')
    printf "%-13s %-7s %-8s %-9s %s\n" "$base@$label" "$twr" "$trw" "$iwr" "$ck"
  done
done
