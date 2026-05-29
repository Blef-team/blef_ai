#!/usr/bin/env bash
#
# One-shot setup for a fresh Hetzner CCX43 (Ubuntu) box, intended to be run
# AFTER the local `cfr_ai/` tree has been rsync'd into ~/blef_ai/ on the box.
#
# Installs system deps + Python deps, then runs a 50-iter warmup so the
# orchestrator's first batch doesn't all eat ~10s of numba JIT compilation
# at the same time.
#
# Usage (on the Hetzner box):
#     bash cfr_ai/scripts/hetzner_setup.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
ROOT="$(pwd)"
echo "[setup] root: $ROOT"

# ----- system deps --------------------------------------------------------
echo "[setup] installing apt packages..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv tmux rsync htop curl ca-certificates

# ----- python venv --------------------------------------------------------
if [[ ! -d "$ROOT/.venv" ]]; then
    python3 -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1090
source "$ROOT/.venv/bin/activate"
pip install --upgrade pip
# Hetzner CCX (Ubuntu 26.04) ships Python 3.14, for which numpy 1.26.4 +
# numba 0.60.0 have no wheels (predate cp314). Install unpinned so pip
# resolves to numpy>=2.1 + a matching numba — both have cp314 wheels.
# This is fine: the saved strategy.npz files use plain numpy arrays (no
# pickled objects, no version-specific binary structures), so models
# trained here load equally well in the Lambda's numpy 1.26 / numba 0.60
# (Python 3.12) environment.
pip install --upgrade numpy numba llvmlite tqdm psutil matplotlib

# ----- sanity ------------------------------------------------------------
echo "[setup] python: $(python --version) at $(which python)"
python -c "import numpy, numba, llvmlite; print('numpy', numpy.__version__, 'numba', numba.__version__, 'llvmlite', llvmlite.__version__)"

# ----- JIT warmup --------------------------------------------------------
echo "[setup] warming numba JIT (compiles _traverse_jit + LBR JIT once)..."
python -c "
from cfr_ai.trainer import Trainer, _seed_numba
from cfr_ai import lbr
import numpy as np, time
_seed_numba(0)
t = Trainer([1,1], 0, [-20,-22], 0.0, 0, initial_capacity=2000, numba_seed=0, regret_dtype=np.float32)
t0 = time.time(); t.train(50); print(f'  trainer JIT compile + 50 iter: {time.time()-t0:.1f}s')
# Now LBR — exercise the recursion paths.
from cfr_ai.lbr import lbr_exploitability, load_flat_strategy
fs, _ = t.get_final_flat_strategy(drop_check_only=True, clear_lows_threshold=0.01)
t0 = time.time(); lbr_exploitability([1,1], 0, fs, depth=1, n_belief_samples=50, n_lbr_hand_samples=50, show_progress=False); print(f'  lbr JIT compile + tiny eval: {time.time()-t0:.1f}s')
"

echo "[setup] done. Next: tmux new -s retrain; python -m cfr_ai.scripts.hetzner_run --workers 16"
