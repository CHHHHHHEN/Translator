from __future__ import annotations

from typing import Mapping, Tuple, Union

import mss
import numpy as np


def capture_screen(region: Union[Mapping[str, int], Tuple[int, int, int, int], None] = None) -> np.ndarray:
    """
    Grab the selected screen region or the primary monitor.
    Returns a numpy array (RGB).
    """
    with mss.mss() as sct:
        if region is None:
            monitor = sct.monitors[1] # Primary monitor
        elif isinstance(region, tuple):
            # Convert (x, y, w, h) to mss format
            monitor = {"left": region[0], "top": region[1], "width": region[2], "height": region[3]}
        else:
            monitor = region

        sct_img = sct.grab(monitor)
        
        # Convert to numpy array
        # mss returns BGRA, we want RGB usually for OCR (or BGR for OpenCV)
        # PaddleOCR uses OpenCV which is BGR usually, but let's check.
        # Actually standard is often RGB for PIL.
        # Let's return RGB for consistency.
        img = np.array(sct_img)
        
        # BGRA to RGB
        # img is [height, width, 4]
        rgb = img[:, :, :3]
        # Swap B and R (mss is BGRA)
        rgb = rgb[:, :, ::-1]
        
        return rgb
