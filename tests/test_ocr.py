import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import onnxruntime as ort

from translator.ocr.paddle import PaddleOcrEngine
from translator.ocr.preprocess import preprocess, preprocess_rec


def test_preprocess_returns_same_bytes() -> None:
    payload = b"abc"
    assert preprocess(payload) is payload


def test_preprocess_rec_shape() -> None:
    # Create a dummy image (H, W, C)
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    processed = preprocess_rec(img)
    
    # Check shape: [1, 3, 48, W]
    assert processed.shape[0] == 1
    assert processed.shape[1] == 3
    assert processed.shape[2] == 48
    # Width should be scaled. 100->48 (scale 0.48). 200*0.48 = 96.
    assert processed.shape[3] == 96


def test_extract_text_returns_empty_if_no_model() -> None:
    with patch("translator.ocr.paddle.ort.InferenceSession") as mock_session:
        # Mock InferenceSession to raise exception to simulate failure
        mock_session.side_effect = Exception("Model not found")
        
        engine = PaddleOcrEngine()
        result = engine.extract_text(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result == ""


@pytest.mark.onnx
def test_onnx_model_loading() -> None:
    # This test tries to actually load the model if it exists
    # If not, it should handle it gracefully
    engine = PaddleOcrEngine()
    if engine._rec_session:
        assert isinstance(engine._rec_session, ort.InferenceSession)

def test_load_dict_yaml() -> None:
    with patch("translator.ocr.paddle.ort.InferenceSession"):
        # Mock settings to point to a .yml file
        with patch("translator.ocr.paddle.settings.get", return_value="dummy.yml"):
            with patch("pathlib.Path.exists", return_value=True):
                # Mock open
                with patch("builtins.open", MagicMock()):
                    # Mock yaml.safe_load
                    with patch("yaml.safe_load", return_value={"PostProcess": {"character_dict": ["x", "y", "z"]}}):
                        engine = PaddleOcrEngine()
                        assert engine._character_dict == ["x", "y", "z"]

def test_ctc_decode() -> None:
    with patch("translator.ocr.paddle.ov.Core"):
        engine = PaddleOcrEngine()
        # Mock dictionary
        engine._character_dict = ["a", "b", "c"]
        
        # Preds: [1, 5, 4] (Batch=1, Time=5, Classes=4)
        # Classes: 0='a', 1='b', 2='c', 3='blank'
        # Note: In my new implementation, if num_classes == dict_size + 1, then space is NOT supported (unless in dict).
        # Here dict_size=3, num_classes=4. So space_index=-1.
        
        preds = np.zeros((1, 5, 4))
        
        # Sequence: a, blank, b, b, blank -> "ab" (duplicates merged? No, duplicates merged if adjacent)
        # Sequence: a, blank, b, b, blank -> "abb" ?
        # CTC rule: collapse repeats, then remove blanks.
        # a, -, b, b, - -> a, b, b -> abb?
        # Wait, standard CTC:
        # a, a, -, b, b -> a, b
        # a, -, a -> a, a
        
        # Let's construct specific indices
        # 0 ('a'), 3 ('-'), 1 ('b'), 1 ('b'), 3 ('-')
        # indices: [0, 3, 1, 1, 3]
        
        # Set probabilities (argmax will pick these)
        preds[0, 0, 0] = 1.0 # 'a'
        preds[0, 1, 3] = 1.0 # '-'
        preds[0, 2, 1] = 1.0 # 'b'
        preds[0, 3, 1] = 1.0 # 'b'
        preds[0, 4, 3] = 1.0 # '-'
        
        decoded = engine.decode(preds)
        # My implementation:
        # prev=-1. 
        # idx=0 ('a'). != prev. != blank. append 'a'. prev=0.
        # idx=3 ('-'). != prev. == blank. skip. prev=3.
        # idx=1 ('b'). != prev. != blank. append 'b'. prev=1.
        # idx=1 ('b'). == prev. skip. prev=1.
        # idx=3 ('-'). != prev. == blank. skip. prev=3.
        # Result: "ab"
        
        assert decoded == "ab"

def test_ctc_decode_with_space() -> None:
    with patch("translator.ocr.paddle.ov.Core"):
        engine = PaddleOcrEngine()
        # Mock dictionary
        engine._character_dict = ["a"] # size 1
        
        # We want to simulate: [char, space, blank] -> size 3
        # dict_size = 1. num_classes = 3.
        # 3 == 1 + 2. Condition met.
        
        # Classes: 0='a', 1='space', 2='blank'
        preds = np.zeros((1, 3, 3))
        
        # Sequence: a, space, a
        preds[0, 0, 0] = 1.0 # 'a'
        preds[0, 1, 1] = 1.0 # ' '
        preds[0, 2, 0] = 1.0 # 'a'
        
        decoded = engine.decode(preds)
        assert decoded == "a a"

def test_ctc_decode_with_space() -> None:
    with patch("translator.ocr.paddle.ov.Core"):
        engine = PaddleOcrEngine()
        # Mock dictionary
        engine._character_dict = ["a"] # size 1
        
        # We want to simulate: [char, space, blank] -> size 3
        # dict_size = 1. num_classes = 3.
        # 3 == 1 + 2. Condition met.
        
        # Classes: 0='a', 1='space', 2='blank'
        preds = np.zeros((1, 3, 3))
        
        # Sequence: a, space, a
        preds[0, 0, 0] = 1.0 # 'a'
        preds[0, 1, 1] = 1.0 # ' '
        preds[0, 2, 0] = 1.0 # 'a'
        
        decoded = engine.decode(preds)
        assert decoded == "a a"

