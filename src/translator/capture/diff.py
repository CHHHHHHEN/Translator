from __future__ import annotations

import numpy as np


def detect_change(before: np.ndarray | None, after: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Return true when the captured image deviates enough from the previous capture.
    Uses simple Mean Squared Error (MSE) for performance.
    """
    if before is None:
        return True
        
    if before.shape != after.shape:
        return True
        
    # Calculate MSE
    err = np.sum((before.astype("float") - after.astype("float")) ** 2)
    err /= float(before.shape[0] * before.shape[1] * before.shape[2])
    
    # Normalize error to 0-1 range roughly (255^2 = 65025)
    normalized_err = err / 65025.0
    
    return normalized_err > threshold
