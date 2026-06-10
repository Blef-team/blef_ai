import numpy as np


def clear_lows(probabilities: np.ndarray) -> np.ndarray:
    """Return a copy with entries below an absolute 0.01 zeroed and the rest
    renormalised. The 0.01 cutoff is hardcoded (absolute, not a fraction of
    the row max). Used by the trainer's `get_final_flat_strategy` to drop
    noise below the deployment significance threshold before writing
    `strategy.npz`."""
    cleaned = np.where(probabilities < 0.01, 0.0, probabilities)
    s = cleaned.sum()
    return cleaned / s if s > 0 else cleaned
