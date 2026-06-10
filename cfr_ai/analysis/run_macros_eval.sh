#!/bin/bash
# Box-side orchestrator for the 3-macro campaign: wait for m3_* trainings, then
# run the H2H battery. Final marker "MACROS-EVAL-DONE" (only here, so the poller
# doesn't trip on the per-invocation "setups-complete" lines).
cd /root/blef_ai
PY=/root/blef_ai/.venv/bin/python
LOG=cfr_ai/experiments/macros3/eval.log
: > "$LOG"
while true; do
  n=$(ls cfr_ai/experiments/macros3/outputs/*/metadata.csv 2>/dev/null | wc -l)
  act=$(systemctl list-units 'm3_*' --state=active --no-legend 2>/dev/null | grep -c 'm3_[0-9]')
  echo "$(date -u +%H:%M:%S) waiting done=$n active=$act" >> "$LOG"
  [ "$n" -ge 6 ] && break
  [ "$act" -eq 0 ] && break
  sleep 60
done
echo "=== TRAININGS DONE $(date -u +%H:%M:%S) ===" >> "$LOG"
echo "=== A: all-3 ON vs OFF (clean macro isolation) ===" >> "$LOG"
$PY -m cfr_ai.analysis.selftest_macros --macros-folder cfr_ai/experiments/macros3 --num-deals 10000 >> "$LOG" 2>&1
echo "=== B: all-3 vs V2.1 ===" >> "$LOG"
$PY -m cfr_ai.analysis.selftest_macros --macros-folder cfr_ai/experiments/macros3 --baseline-dir cfr_ai/scratch_v21 --num-deals 10000 >> "$LOG" 2>&1
echo "=== B2: difftruthy-only vs V2.1 (same 3-macro model) ===" >> "$LOG"
$PY -m cfr_ai.analysis.selftest_macros --macros-folder cfr_ai/experiments/macros3 --baseline-dir cfr_ai/scratch_v21 --enable difftruthy --num-deals 10000 >> "$LOG" 2>&1
echo "=== MACROS-EVAL-DONE $(date -u +%H:%M:%S) ===" >> "$LOG"
