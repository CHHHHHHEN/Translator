from __future__ import annotations

import cv2
import numpy as np
from translator.utils.logger import get_logger

logger = get_logger(__name__)


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index Measure (SSIM) between two images.
    Images must be grayscale and same size.
    """
    C1 = 6.5025
    C2 = 58.5225
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    kernel = (11, 11)
    sigma = 1.5
    
    mu1 = cv2.GaussianBlur(img1, kernel, sigma)
    mu2 = cv2.GaussianBlur(img2, kernel, sigma)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, kernel, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, kernel, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, kernel, sigma) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def detect_change(before: np.ndarray | None, after: np.ndarray, threshold: float = 0.98) -> bool:
    """
    Return True if the similarity is BELOW the threshold (i.e., content changed).
    Uses SSIM for robust change detection.
    """
    if before is None:
        return True
        
    if before.shape != after.shape:
        return True
        
    try:
        # Convert to grayscale
        gray_before = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
        gray_after = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
        
        score = ssim(gray_before, gray_after)
        
        # If score is 1.0, they are identical.
        # If score < threshold, they are different enough.
        return score < threshold
    except Exception as e:
        logger.error(f"SSIM calculation failed: {e}")
        # Fallback to simple diff if SSIM fails
        return True
