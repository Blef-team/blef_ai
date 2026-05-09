#!/usr/bin/env bash
# Watch /Users disk free space; emit one line on threshold crossings.
# Auto-clears intermediate checkpoints (nfsp_blef_*M.pt, NOT the latest
# nfsp_blef.pt) from the OLDEST training run dir if free < hard limit.
#
# Usage: disk_watcher.sh [WARN_GB] [HARD_GB] [POLL_SEC]
set -euo pipefail
WARN_GB="${1:-50}"
HARD_GB="${2:-20}"
POLL="${3:-300}"

free_gb() {
  # macOS-portable: -g reports in 1G blocks.
  df -g /Users/adriangolian | awk 'NR==2 {print $4}'
}

last_state=""
while true; do
  fg=$(free_gb)
  state="ok"
  if [ "$fg" -lt "$HARD_GB" ]; then state="hard"; fi
  if [ "$state" = "ok" ] && [ "$fg" -lt "$WARN_GB" ]; then state="warn"; fi
  if [ "$state" != "$last_state" ] && [ "$state" != "ok" ]; then
    echo "[disk-watcher] state=$state free=${fg}GB (warn<${WARN_GB} hard<${HARD_GB})"
  fi
  if [ "$state" = "hard" ]; then
    # Find the run dir with the most intermediate ckpts and free one.
    target=$(find /Users/adriangolian/work/blef_ai/runs -maxdepth 3 -name 'nfsp_blef_*M.pt' -type f 2>/dev/null | head -1)
    if [ -n "$target" ]; then
      sz=$(du -h "$target" | cut -f1)
      rm -f "$target"
      echo "[disk-watcher] AUTO-CLEAR removed $target ($sz)"
    else
      echo "[disk-watcher] AUTO-CLEAR no intermediate ckpts to delete; manual intervention needed"
    fi
  fi
  last_state="$state"
  sleep "$POLL"
done
