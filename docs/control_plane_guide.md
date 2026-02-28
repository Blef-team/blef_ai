# Control Plane Operator Guide

This guide explains how to operate the JSON control plane that now ships with the NFSP Blef runner.

## Launching with a Control Plane
- Run `python nfsp_run_local.py --control-plane control_plane_<postfix>.json [...other flags...]`.
- The runner bootstraps the JSON file if it does not exist, populating `defaults` with the live schedule, plus an `overrides` section (all `null` to start) and `meta.cooldown_steps`.
- The cooldown can also be set via CLI: `--control-plane-cooldown 25000` overrides the default 50k steps.

## Editing Overrides Safely
1. Always write edits atomically to avoid partial reads:
   ```bash
   cat > control_plane_tmp.json <<'EOF'
   {
     "meta": { "cooldown_steps": 5000 },
     "overrides": {
       "eta": 0.18,
       "epsilon": 0.035,
       "lr_q": 7e-5,
       "train_rl_every": 48,
       "n_step": 7,
       "max_cards": 4,
       "burst_rl_updates_on_reward": 4
     }
   }
   EOF
   mv control_plane_tmp.json control_plane_<postfix>.json
   ```
2. Values must respect the bounds in `docs/control_plane_plan.md` (e.g., `eta ∈ [0,1]`, `n_step ∈ [1,32]`). Invalid edits are rejected with a `[control] invalid control-plane file` message.
3. Use `null` for any key you want to unpin; the internal schedules reclaim the parameter on the next schedule tick.

## Monitoring Runtime Behaviour
- Every time overrides apply, the console prints `ctrl=1` along with the updated pin sets (`pins=… env_pins=…`). The control-plane module also logs individual change messages prefixed with `[control]`.
- CSV rows now include `control_override` (0/1) and `control_changes` columns, plus `pinned_overrides` / `pinned_env`. TensorBoard logs under `Control/*` mirror these signals for dashboards.
- The NFSP checkpoint inspector reports pinned overrides, env pins, and the current CHECK exploration bias:
  ```bash
  python shared/utils/inspect_checkpoint.py nfsp_blef_<timestamp>.pt
  ```

## Cooldown Semantics
- The cooldown prevents rapid-fire edits; overrides queue until `total_env_steps - last_applied_step ≥ cooldown_steps`.
- Adjust `meta.cooldown_steps` in the JSON or pass `--control-plane-cooldown` for shorter iteration loops during debugging. The watcher prints `[control] cooldown active (…)` while waiting.

## Recommended Workflow
1. Start runs with the control plane enabled (and include the JSON path in run notes).
2. When behaviour drifts (e.g., CHECK mastery regresses), atomically pin the relevant parameters.
3. Verify the effect within the next logging window (look for `control_override=1`) and track the change in dashboards.
4. Once the schedule can resume, set the overrides back to `null` and confirm the pins clear (CSV `pinned_overrides` becomes empty).
5. On resume, supply both `--resume` and `--control-plane` so pinned overrides stored inside the checkpoint and the watcher stay aligned.

For a quick smoke test, follow `docs/control_plane_testing.md` to apply, verify, and inspect overrides on a short 20k-step run.
