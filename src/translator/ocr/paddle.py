from __future__ import annotations

from pathlib import Path
import time
import os
import math
from typing import Any, Union, List, Tuple, Optional, cast

import cv2
import yaml
import numpy as np
import onnxruntime as ort
import pyclipper
from shapely.geometry import Polygon

from .engine import OcrEngine
from .preprocess import preprocess_det, preprocess_rec, get_rotate_crop_image
from translator.utils.logger import get_logger
from translator.core.config import settings

logger = get_logger(__name__)


class PaddleOcrEngine(OcrEngine):
    """PaddleOCR implementation using ONNX Runtime (DirectML)."""

    def __init__(self, languages=None) -> None:
        super().__init__(languages)
        
        # Recognition Model
        self._rec_session: Optional[ort.InferenceSession] = None
        self._rec_input_name: Optional[str] = None
        self._rec_output_name: Optional[str] = None
        
        # Detection Model
        self._det_session: Optional[ort.InferenceSession] = None
        self._det_input_name: Optional[str] = None
        self._det_output_name: Optional[str] = None
        
        self._character_dict: List[str] = []
        
        # Load character dictionary
        self._load_dict()
        
        # Load models
        self.load_models()

    def _load_dict(self) -> None:
        dict_path_str = settings.get("ocr.rec_char_dict_path", "assets/ppocr_keys_v1.txt")
        dict_path = Path(dict_path_str)
        
        if not dict_path.exists():
            root = Path.cwd()
            dict_path = root / dict_path_str
            
        if dict_path.exists():
            try:
                if dict_path.suffix.lower() in (".yml", ".yaml"):
                    with open(dict_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                        if "PostProcess" in data and "character_dict" in data["PostProcess"]:
                            self._character_dict = data["PostProcess"]["character_dict"]
                        else:
                            logger.warning("Could not find PostProcess.character_dict in YAML.")
                            self._character_dict = []
                else:
                    with open(dict_path, "r", encoding="utf-8") as f:
                        self._character_dict = [line.strip("\n").strip("\r") for line in f.readlines()]
                
                logger.info(f"Loaded {len(self._character_dict)} characters from dictionary.")
            except Exception as e:
                logger.error(f"Failed to load dictionary: {e}")
        else:
            logger.warning(f"Dictionary not found at {dict_path}. Using minimal default.")
            self._character_dict = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def __del__(self) -> None:
        self._rec_session = None
        self._det_session = None

    def load_models(self) -> None:
        self._load_rec_model()
        self._load_det_model()

    def _get_providers(self) -> List[str]:
        # Prefer DirectML if available, otherwise fall back to CPU
        available = ort.get_available_providers()
        providers: List[str] = []
        if 'DmlExecutionProvider' in available:
            providers.append('DmlExecutionProvider')
            logger.info("DirectML Execution Provider is available.")
        if 'CUDAExecutionProvider' in available and 'DmlExecutionProvider' not in providers:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        return providers

    def _load_rec_model(self) -> None:
        # Check for ONNX model
        onnx_path = Path("models/PP-OCRv5_mobile_rec_infer_ONNX/inference.onnx")
        
        if not onnx_path.exists():
            logger.error(f"Rec Model not found at {onnx_path}")
            return

        try:
            logger.info(f"Loading Rec model from {onnx_path}")
            so = ort.SessionOptions()
            try:
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            except Exception:
                pass
            providers = self._get_providers()
            self._rec_session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
            self._rec_input_name = self._rec_session.get_inputs()[0].name
            self._rec_output_name = self._rec_session.get_outputs()[0].name
            logger.info(f"Rec Model loaded successfully using {self._rec_session.get_providers()}.")
            # Warm-up the session with a small dummy input to reduce first-inference overhead
            try:
                in0 = self._rec_session.get_inputs()[0]
                shape = []
                for d in in0.shape:
                    if isinstance(d, str) or d is None:
                        # choose reasonable defaults: batch=1, C=3, H=48, W=32
                        if len(shape) == 0:
                            shape.append(1)
                        elif len(shape) == 1:
                            shape.append(3)
                        elif len(shape) == 2:
                            shape.append(48)
                        else:
                            shape.append(32)
                    else:
                        shape.append(int(d))
                if len(shape) < 4:
                    shape = [1, 3, 48, 32]
                dummy = np.zeros(tuple(shape), dtype=np.float32)
                try:
                    self._rec_session.run([self._rec_output_name], {self._rec_input_name: dummy})
                    logger.info("Rec session warm-up completed")
                except Exception:
                    logger.debug("Rec session warm-up failed, continuing")
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Failed to load Rec model: {e}")

    def _load_det_model(self) -> None:
        # Check for ONNX model
        onnx_path = Path("models/PP-OCRv5_mobile_dec_infer_ONNX/inference.onnx")
        
        if not onnx_path.exists():
            logger.error(f"Det Model not found at {onnx_path}")
            return

        try:
            logger.info(f"Loading Det model from {onnx_path}")
            so = ort.SessionOptions()
            try:
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            except Exception:
                pass
            providers = self._get_providers()
            self._det_session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
            self._det_input_name = self._det_session.get_inputs()[0].name
            self._det_output_name = self._det_session.get_outputs()[0].name
            logger.info(f"Det Model loaded successfully using {self._det_session.get_providers()}.")
            try:
                in0 = self._det_session.get_inputs()[0]
                shape = []
                for d in in0.shape:
                    if isinstance(d, str) or d is None:
                        if len(shape) == 0:
                            shape.append(1)
                        elif len(shape) == 1:
                            shape.append(3)
                        elif len(shape) == 2:
                            shape.append(320)
                        else:
                            shape.append(256)
                    else:
                        shape.append(int(d))
                if len(shape) < 4:
                    shape = [1, 3, 320, 256]
                dummy = np.zeros(tuple(shape), dtype=np.float32)
                try:
                    self._det_session.run([self._det_output_name], {self._det_input_name: dummy})
                    logger.info("Det session warm-up completed")
                except Exception:
                    logger.debug("Det session warm-up failed, continuing")
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Failed to load Det model: {e}")

    def postprocess_det(self, preds: np.ndarray, shape_list: Tuple[int, int, float, float]) -> List[np.ndarray]:
        # preds: [1, 1, H, W]
        pred = preds[0, 0, :, :]
        segmentation = pred > 0.3
        
        boxes_batch = []
        src_h, src_w, ratio_h, ratio_w = shape_list
        
        # Find contours
        mask = (segmentation * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            points, sside = self._get_mini_boxes(contour)
            if sside < 3:
                continue
                
            points = np.array(points)
            score = self._box_score_fast(pred, points)
            if score < 0.6: # box_thresh
                continue
                
            box = self._unclip(points)
            box, sside = self._get_mini_boxes(box)
            if sside < 3:
                continue
                
            box = np.array(box)
            
            # Clip to image size
            box[:, 0] = np.clip(np.round(box[:, 0] / ratio_w), 0, src_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / ratio_h), 0, src_h)
            
            boxes_batch.append(box.astype(np.int32))
            
        return boxes_batch

    def _get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def _box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(int), 1)
        mean_val = cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)
        return mean_val[0] # type: ignore

    def _unclip(self, box):
        unclip_ratio = 1.5
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset() # type: ignore
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON) # type: ignore
        expanded = np.array(offset.Execute(distance))
        return expanded

    def decode(self, preds: np.ndarray) -> str:
        # Accept preds in shapes: (1, T, C), (T, C) or (N, T, C) - operate on single-line preds
        if preds is None:
            return ""
        arr = np.array(preds)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        # expect shape (1, T, C)
        if arr.ndim == 3 and arr.shape[0] > 1:
            arr = arr[0:1]

        indices = np.argmax(arr, axis=2)
        line_indices = indices[0]
        num_classes = arr.shape[2]
        dict_size = len(self._character_dict)
        
        # For PP-OCRv5 ONNX, the CTC blank token is usually at index 0.
        # The character indices are shifted by +1 (index 1 -> dict[0]).
        blank_index = 0
        
        sb = []
        prev_index = -1
        for index in line_indices:
            if index != prev_index:
                if index == blank_index:
                    pass
                elif index > 0 and index - 1 < dict_size:
                    sb.append(self._character_dict[index - 1])
                # Handle implicit space if supported by model topology (e.g. last class)
                elif num_classes == dict_size + 2 and index == num_classes - 1:
                     sb.append(" ")
            prev_index = index
        return "".join(sb)

    def extract_text(self, image: Union[bytes, np.ndarray, Any]) -> str:
        if self._rec_session is None:
            self._load_rec_model()
        if self._det_session is None:
            self._load_det_model()
            
        if isinstance(image, bytes):
            nparr = np.frombuffer(image, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        if image is None:
            return ""
            
        if not isinstance(image, np.ndarray):
             logger.warning(f"Invalid image type: {type(image)}")
             return ""

        try:
            # 1. Detection
            boxes = []
            if self._det_session and self._det_input_name and self._det_output_name:
                det_img, ratio_h, ratio_w = preprocess_det(image)
                
                t0 = time.perf_counter()
                det_preds = self._det_session.run(
                    [self._det_output_name],
                    {self._det_input_name: det_img}
                )[0]
                t1 = time.perf_counter()
                try:
                    providers = self._det_session.get_providers()
                except Exception:
                    providers = None
                logger.info(f"Det inference: boxes_input_shape={det_img.shape if hasattr(det_img,'shape') else 'N/A'} elapsed={t1-t0:.4f}s providers={providers}")
                
                boxes = self.postprocess_det(det_preds, (image.shape[0], image.shape[1], ratio_h, ratio_w))
                
                # Sort boxes (top to bottom, left to right)
                boxes = sorted(boxes, key=lambda x: (x[0][1], x[0][0]))
            else:
                # Fallback: treat whole image as one box
                h, w = image.shape[:2]
                boxes = [np.array([[0, 0], [w, 0], [w, h], [0, h]])]

            # 2. Recognition
            if not self._rec_session or not self._rec_input_name or not self._rec_output_name:
                 logger.warning("Recognition model not loaded. Skipping recognition.")
                 return ""

            full_text = []

            # Batch recognition with width bucketing to reduce padding overhead
            rec_items: list[tuple[int, np.ndarray]] = []
            for idx, box in enumerate(boxes):
                crop = get_rotate_crop_image(image, box)
                rec_in = preprocess_rec(crop)
                if rec_in is None:
                    continue
                arr = np.asarray(rec_in, dtype=np.float32)
                if arr.ndim == 4 and arr.shape[0] == 1:
                    arr = arr[0]
                # arr shape (C, H, W)
                rec_items.append((idx, arr))

            if len(rec_items) == 0:
                return ""

            # Optimization: Process all items in a single batch to minimize session.run overhead.
            # DmlExecutionProvider can have high latency per call, so fewer calls is better.
            # We pad all items to the maximum width in the current set.
            
            # 1. Determine max width for the batch
            max_w = 0
            for _, arr in rec_items:
                if arr.shape[2] > max_w:
                    max_w = arr.shape[2]
            
            # Align to 32 pixels
            max_w = ((max_w + 31) // 32) * 32
            
            # 2. Prepare the batch
            padded_list = []
            idxs = []
            for idx, arr_in in rec_items:
                c, h, w = arr_in.shape
                if w < max_w:
                    pad_width = ((0, 0), (0, 0), (0, max_w - w))
                    arr_in = np.pad(arr_in, pad_width, mode='constant', constant_values=0)
                padded_list.append(arr_in)
                idxs.append(idx)

            batch = np.stack(padded_list, axis=0) # (N, C, H, W)
            
            # Prepare output container
            outputs: dict[int, np.ndarray] = {}

            try:
                t0 = time.perf_counter()
                rec_preds = self._rec_session.run(
                    [self._rec_output_name],
                    {self._rec_input_name: batch}
                )[0]
                t1 = time.perf_counter()
                try:
                    providers = self._rec_session.get_providers()
                except Exception:
                    providers = None
                logger.info(f"Rec inference batch_size={batch.shape[0]} input_shape={batch.shape} elapsed={t1-t0:.4f}s providers={providers}")
                
                arr = np.array(rec_preds)
                # Normalize output to (N, T, C)
                if arr.ndim == 4 and arr.shape[1] == 1:
                    arr = np.squeeze(arr, axis=1)
                if arr.ndim == 3 and arr.shape[2] < arr.shape[1]:
                    arr = arr.transpose(0, 2, 1)
                
                # Assign outputs
                for i_out, idx in enumerate(idxs):
                    if i_out < len(arr):
                        outputs[idx] = arr[i_out]
                    else:
                        outputs[idx] = np.array([])

            except Exception as e:
                logger.error(f"Recognition run failed: {e}")
                # Fallback or empty
                pass

            # decode in original order
            for i in range(len(boxes)):
                arr_out = outputs.get(i)
                if arr_out is None or arr_out.size == 0:
                    continue
                try:
                    text = self.decode(arr_out)
                except Exception:
                    text = ""
                if text and text.strip():
                    full_text.append(text)

            return "\n".join(full_text)
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return ""
