import numpy as np


def clear_lows(probabilities: np.ndarray) -> np.ndarray:
    """Return a copy with sub-1% entries zeroed and the rest renormalised.
    Used by the trainer's `get_final_flat_strategy` to drop noise below the
    deployment significance threshold before writing `strategy.npz`."""
    cleaned = np.where(probabilities < 0.01, 0.0, probabilities)
    s = cleaned.sum()
    return cleaned / s if s > 0 else cleaned
