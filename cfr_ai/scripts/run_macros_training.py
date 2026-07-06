"""Run multi-macro training through the production training pipeline.
Swaps TrainerMacros into training.main(); saves one macro-aware strategy.npz
(concrete probs + macro masses + kinds), no separate macro file.
Kinds come from the BLUFF_MACRO_KINDS env var (default value,difftruthy,bluff).
"""
import os
import cfr_ai.training as T
from cfr_ai.abstraction.trainer_macros import TrainerMacros

T.Trainer = TrainerMacros
_orig_save = T.save_strategies


def _save_unified(trainer, out_root, hand_sizes, *a, **k):
    """Extract the macro strategy (concrete `c/T` probs + `masses` + `kinds`)
    BEFORE the normal save frees the trainer arrays, let the normal save write
    diagnostic + metadata + summary (+ a concrete strategy.npz), then overwrite
    strategy.npz with the SINGLE macro-aware file."""
    from cfr_ai.strategy_io import save_macro_strategy
    setup_dir = os.path.join(out_root, "outputs", "_".join(str(x) for x in hand_sizes))
    d = trainer.get_macros_strategy()
    out = _orig_save(trainer, out_root, hand_sizes, *a, **k)
    try:
        save_macro_strategy(setup_dir, keys=d["keys"], lower=d["lower"], upper=d["upper"],
                            probs=d["probs"], masses=d["masses"], kinds=d["kinds"],
                            min_bet=d["min_bet"],
                            history_depth=getattr(trainer, "history_depth", 3))
        nm = int((d["masses"].sum(axis=1) > 0).sum()) if len(d["keys"]) else 0
        print(f"  [macros] wrote {len(d['keys']):,} infosets -> {setup_dir}/strategy.npz "
              f"(kinds={d['kinds']}, {nm:,} with macro mass)", flush=True)
    except Exception as e:
        print(f"  [macros] unified strategy write FAILED: {e}", flush=True)
    return out


T.save_strategies = _save_unified

if __name__ == "__main__":
    T.main()
