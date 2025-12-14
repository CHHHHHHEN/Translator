from __future__ import annotations

import time
import traceback
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from ..core.state import state
from ..capture.screen import capture_screen
from ..capture.diff import detect_change
from .ocr_service import OcrService
from .translate_service import TranslateService
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MonitorService(QObject):
    """
    Background service that monitors the screen, detects changes,
    runs OCR, and triggers translation.
    """
    
    # Signals to update UI
    text_detected = pyqtSignal(str)
    translation_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._ocr_service: OcrService | None = None
        self._translate_service: TranslateService | None = None
        self._last_image: np.ndarray | None = None
        self._running = False
        self._target_lang = "zh"
        self._interval = 0.2

    def set_target_language(self, lang: str) -> None:
        self._target_lang = lang

    def set_interval(self, interval: float) -> None:
        self._interval = interval

    def start_monitoring(self) -> None:
        # Initialize services here to avoid blocking main thread
        if self._ocr_service is None:
            logger.info("Initializing services in background thread...")
            try:
                self._ocr_service = OcrService()
                self._translate_service = TranslateService()
            except Exception as e:
                logger.error(f"Failed to initialize services: {e}")
                self.error_occurred.emit(f"Init Failed: {e}")
                return

        self._running = True
        self._loop()

    def stop_monitoring(self) -> None:
        self._running = False

    def _loop(self) -> None:
        logger.info("Monitor service started.")
        while self._running:
            if not state.is_monitoring or not state.monitoring_region:
                time.sleep(0.5)
                continue

            try:
                # 1. Capture
                current_image = capture_screen(state.monitoring_region)
                
                # 2. Detect Change
                if detect_change(self._last_image, current_image):
                    logger.debug("Change detected, running OCR...")
                    self._last_image = current_image
                    
                    # 3. OCR
                    text = self._ocr_service.detect_text(current_image)
                    
                    if text and text.strip():
                        logger.info(f"OCR Text: {text}")
                        state.last_ocr_text = text
                        self.text_detected.emit(text)
                        
                        # 4. Translate
                        translated = self._translate_service.translate(text, self._target_lang)
                        state.last_translated_text = translated
                        self.translation_finished.emit(translated)
                        
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                logger.debug(traceback.format_exc())
                self.error_occurred.emit(str(e))
                time.sleep(1) # Backoff

            time.sleep(self._interval) # Poll interval
        
        logger.info("Monitor service stopped.")
