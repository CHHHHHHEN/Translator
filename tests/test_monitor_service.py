import time
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from translator.services.monitor_service import MonitorService
from translator.core.state import state

@pytest.fixture
def monitor_service():
    return MonitorService()

def test_monitor_service_init(monitor_service):
    assert monitor_service._running is False
    assert monitor_service._ocr_service is None

def test_monitor_service_start_stop(monitor_service):
    # Mock the loop to exit immediately or run once
    with patch.object(monitor_service, "_loop") as mock_loop:
        monitor_service.start_monitoring()
        assert monitor_service._running is True
        mock_loop.assert_called_once()
        
        monitor_service.stop_monitoring()
        assert monitor_service._running is False

def test_monitor_loop_logic(monitor_service):
    # We want to test one iteration of the loop
    # We can mock capture_screen, detect_change, ocr_service
    
    state.is_monitoring = True
    state.monitoring_region = (0, 0, 100, 100)
    
    mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    with patch("translator.services.monitor_service.capture_screen", return_value=mock_img) as mock_capture:
        with patch("translator.services.monitor_service.detect_change", return_value=True) as mock_diff:
            with patch("translator.services.monitor_service.OcrService") as MockOcrService:
                # Setup OCR mock
                mock_ocr_instance = MockOcrService.return_value
                mock_ocr_instance.detect_text.return_value = "Detected Text"
                
                # Inject the mock OCR service directly since start_monitoring initializes it
                monitor_service._ocr_service = mock_ocr_instance
                
                # We need to break the loop after one iteration
                # We can do this by side_effect on time.sleep to raise an exception or change _running
                def stop_loop(*args):
                    monitor_service._running = False
                
                with patch("time.sleep", side_effect=stop_loop):
                    monitor_service._running = True
                    monitor_service._loop()
                
                # Verify calls
                mock_capture.assert_called()
                mock_diff.assert_called()
                mock_ocr_instance.detect_text.assert_called_with(mock_img)
                
                # Verify state update
                assert state.last_ocr_text == "Detected Text"
                # Verify translation triggered (we didn't mock translate service in this test block fully, 
                # but it should be initialized and called)
                # Since we didn't mock TranslateService class, it used the real one or failed?
                # Wait, we didn't mock TranslateService in the test function, so it might try to use real one.
                # But real one uses LLMTranslator which uses OpenAI.
                # It might fail or return error string.
                # Let's check if translation_finished signal was emitted?
                # We can't easily check signal emission without connecting a slot or spy.
                
                # Ideally we should mock TranslateService too.

def test_monitor_loop_with_translation(monitor_service):
    state.is_monitoring = True
    state.monitoring_region = (0, 0, 100, 100)
    mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    with patch("translator.services.monitor_service.capture_screen", return_value=mock_img):
        with patch("translator.services.monitor_service.detect_change", return_value=True):
            with patch("translator.services.monitor_service.OcrService") as MockOcr:
                MockOcr.return_value.detect_text.return_value = "Hello"
                
                with patch("translator.services.monitor_service.TranslateService") as MockTrans:
                    MockTrans.return_value.translate.return_value = "Bonjour"
                    
                    # Inject mocks
                    monitor_service._ocr_service = MockOcr.return_value
                    monitor_service._translate_service = MockTrans.return_value
                    
                    def stop_loop(*args):
                        monitor_service._running = False
                    
                    with patch("time.sleep", side_effect=stop_loop):
                        monitor_service._running = True
                        monitor_service._loop()
                    
                    MockTrans.return_value.translate.assert_called()
                    assert state.last_translated_text == "Bonjour"

def test_monitor_loop_no_change(monitor_service):
    state.is_monitoring = True
    state.monitoring_region = (0, 0, 100, 100)
    
    with patch("translator.services.monitor_service.capture_screen") as mock_capture:
        with patch("translator.services.monitor_service.detect_change", return_value=False) as mock_diff:
            monitor_service._ocr_service = MagicMock()
            
            def stop_loop(*args):
                monitor_service._running = False
            
            with patch("time.sleep", side_effect=stop_loop):
                monitor_service._running = True
                monitor_service._loop()
            
            mock_capture.assert_called()
            mock_diff.assert_called()
            # OCR should NOT be called
            monitor_service._ocr_service.detect_text.assert_not_called()
