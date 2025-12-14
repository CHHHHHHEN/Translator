import numpy as np
from unittest.mock import patch, MagicMock
from translator.capture.screen import capture_screen

def test_capture_screen_full() -> None:
    with patch("mss.mss") as mock_mss:
        mock_sct = MagicMock()
        mock_mss.return_value.__enter__.return_value = mock_sct
        
        # Mock monitors
        mock_sct.monitors = [{}, {"top": 0, "left": 0, "width": 1920, "height": 1080}]
        
        # Mock grab result
        # mss returns a ScreenShot object, but we can mock it to return something that can be converted to array
        # Actually capture_screen converts sct_img to np.array(sct_img)
        # So we need mock_sct.grab to return something array-like
        
        # Let's mock the return value of grab to be a numpy array directly if we can, 
        # but the code does `np.array(sct_img)`.
        # If we make grab return a numpy array, np.array() on it works fine.
        # BGRA image
        mock_img = np.zeros((100, 100, 4), dtype=np.uint8)
        mock_sct.grab.return_value = mock_img
        
        result = capture_screen()
        
        # Should call grab with monitor 1
        mock_sct.grab.assert_called_with(mock_sct.monitors[1])
        
        # Result should be RGB (3 channels)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100, 3)

def test_capture_screen_region() -> None:
    with patch("mss.mss") as mock_mss:
        mock_sct = MagicMock()
        mock_mss.return_value.__enter__.return_value = mock_sct
        
        mock_img = np.zeros((50, 50, 4), dtype=np.uint8)
        mock_sct.grab.return_value = mock_img
        
        region = (10, 10, 50, 50)
        result = capture_screen(region)
        
        # Should call grab with dict
        expected_monitor = {"left": 10, "top": 10, "width": 50, "height": 50}
        mock_sct.grab.assert_called_with(expected_monitor)
        
        assert result.shape == (50, 50, 3)
