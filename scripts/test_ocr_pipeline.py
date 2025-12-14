import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from translator.ocr.paddle import PaddleOcrEngine
from translator.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    print("Initializing PaddleOcrEngine...")
    try:
        engine = PaddleOcrEngine()
        print("Engine initialized.")
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return

    if engine._rec_compiled_model:
        print("Rec model loaded successfully.")
    else:
        print("Rec model failed to load.")

    if engine._det_compiled_model:
        print("Det model loaded successfully.")
    else:
        print("Det model failed to load.")

    # Create a dummy image
    # White background, black text (simulated with a rectangle)
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    # Draw some "text" boxes
    cv2.rectangle(img, (50, 50), (200, 100), (0, 0, 0), -1)
    cv2.rectangle(img, (50, 150), (300, 200), (0, 0, 0), -1)

    print("Running extract_text on dummy image...")
    try:
        result = engine.extract_text(img)
        print("Extraction finished.")
        print(f"Result: '{result}'") # Likely empty or garbage since it's just rectangles, but shouldn't crash
    except Exception as e:
        print(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
