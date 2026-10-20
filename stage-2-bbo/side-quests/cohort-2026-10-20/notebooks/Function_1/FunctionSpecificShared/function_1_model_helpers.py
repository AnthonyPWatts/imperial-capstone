"""Function 1's target scale, shared by exploration and model notebooks.

The default scale matches the Week 4 model study. Choosing a height scale does
not round, clip or replace the measured outputs.
"""

import numpy as np


def signed_log(values, scale=1e-200):
    """Preserve sign and ordering on the fixed height scale used in exploration."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values) / scale) / np.log(10.0)
