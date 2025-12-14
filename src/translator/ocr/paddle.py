from __future__ import annotations

from pathlib import Path
from typing import Any, Union, List, Tuple

import cv2
import yaml
import math
import numpy as np
import openvino as ov
import pyclipper
from shapely.geometry import Polygon

from .engine import OcrEngine
from translator.utils.logger import get_logger
from translator.core.config import settings

logger = get_logger(__name__)


class PaddleOcrEngine(OcrEngine):
    """PaddleOCR implementation using OpenVINO runtime directly."""

    def __init__(self, languages=None) -> None:
        super().__init__(languages)
        self._core = ov.Core()
        
        # Recognition Model
        self._rec_compiled_model: ov.CompiledModel | None = None
        self._rec_input_layer = None
        self._rec_output_layer = None
        
        # Detection Model
        self._det_compiled_model: ov.CompiledModel | None = None
        self._det_input_layer = None
        self._det_output_layer = None
        
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
        try:
            self._rec_compiled_model = None
            self._det_compiled_model = None
            self._core = None
        except:
            pass

    def load_models(self) -> None:
        self._load_rec_model()
        self._load_det_model()

    def _load_rec_model(self) -> None:
        model_path = Path("models/PP-OCRv5_mobile_rec_infer_IR/inference.xml")
        if not model_path.exists():
            logger.error(f"Rec Model not found at {model_path.absolute()}")
            return

        try:
            logger.info(f"Loading Rec model from {model_path}")
            model = self._core.read_model(model=model_path)
            
            for input_layer in model.inputs:
                input_shape = input_layer.partial_shape
                input_shape[3] = -1
                model.reshape({input_layer: input_shape})
            
            device = self._select_device()
            logger.info(f"Compiling Rec model on {device}...")
            self._rec_compiled_model = self._core.compile_model(model, device_name=device)
            self._rec_input_layer = self._rec_compiled_model.input(0)
            self._rec_output_layer = self._rec_compiled_model.output(0)
            logger.info(f"Rec Model compiled successfully on {device}.")
        except Exception as e:
            logger.error(f"Failed to load Rec model: {e}")

    def _load_det_model(self) -> None:
        # Try to find the model file. It could be inference.pdmodel or inference.json
        base_path = Path("models/PP-OCRv5_mobile_det_infer")
        model_path = base_path / "inference.pdmodel"
        if not model_path.exists():
            model_path = base_path / "inference.json"
            
        if not model_path.exists():
            logger.error(f"Det Model not found at {base_path}")
            return

        try:
            logger.info(f"Loading Det model from {model_path}")
            model = self._core.read_model(model=model_path)
            
            # Dynamic shape for detection is usually handled automatically or we can set it
            # Input shape: [1, 3, H, W]
            # We can set dynamic H, W
            for input_layer in model.inputs:
                input_shape = input_layer.partial_shape
                input_shape[2] = -1
                input_shape[3] = -1
                model.reshape({input_layer: input_shape})

            device = self._select_device()
            logger.info(f"Compiling Det model on {device}...")
            self._det_compiled_model = self._core.compile_model(model, device_name=device)
            self._det_input_layer = self._det_compiled_model.input(0)
            self._det_output_layer = self._det_compiled_model.output(0)
            logger.info(f"Det Model compiled successfully on {device}.")
        except Exception as e:
            logger.error(f"Failed to load Det model: {e}")

    def _select_device(self) -> str:
        available_devices = self._core.available_devices
        if "GPU" in available_devices:
            return "GPU"
        elif "NPU" in available_devices:
            return "NPU"
        return "CPU"

    def preprocess_det(self, image: np.ndarray, limit_side_len: int = 960) -> Tuple[np.ndarray, float, float]:
        h, w = image.shape[:2]
        ratio = 1.0
        if max(h, w) > limit_side_len:
            if h > w:
                ratio = float(limit_side_len) / h
            else:
                ratio = float(limit_side_len) / w
        
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)
        
        # Ensure multiple of 32
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)
        
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        
        img = cv2.resize(image, (resize_w, resize_h))
        
        # Normalize: (img / 255.0 - mean) / std
        # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
        img = img.astype('float32') / 255.0
        img -= np.array([0.485, 0.456, 0.406])
        img /= np.array([0.229, 0.224, 0.225])
        
        img = img.transpose((2, 0, 1))
        img = img[np.newaxis, :]
        return img, ratio_h, ratio_w

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
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def _unclip(self, box):
        unclip_ratio = 1.5
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_rotate_crop_image(self, img, points):
        img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points.astype(np.float32), pts_std)
        dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
        
        # Check if vertical
        if dst_img.shape[0] / dst_img.shape[1] > 1.5:
            dst_img = np.rot90(dst_img)
            
        return dst_img

    def preprocess_rec(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for PP-OCRv5 Rec model.
        """
        # Convert RGB to BGR (OpenCV default)
        # Note: If input is already BGR (from cv2.imread or converted), this might be redundant or wrong.
        # But capture_screen returns RGB.
        # However, get_rotate_crop_image operates on the original image.
        # If original image was RGB, crop is RGB.
        # So we convert to BGR here.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w = image.shape[:2]
        imgH = 48
        
        ratio = w / float(h)
        resized_w = int(math.ceil(imgH * ratio))
        
        resized_image = cv2.resize(image, (resized_w, imgH))
        
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0
        resized_image -= 0.5
        resized_image /= 0.5
        
        resized_image = resized_image[np.newaxis, :]
        return resized_image

    def decode(self, preds: np.ndarray) -> str:
        indices = np.argmax(preds, axis=2)
        line_indices = indices[0]
        num_classes = preds.shape[2]
        dict_size = len(self._character_dict)
        blank_index = num_classes - 1
        space_index = -1
        if num_classes == dict_size + 2:
            space_index = num_classes - 2
        
        sb = []
        prev_index = -1
        for index in line_indices:
            if index != prev_index:
                if index == blank_index:
                    pass
                elif index == space_index:
                    sb.append(" ")
                elif index < dict_size:
                    sb.append(self._character_dict[index])
            prev_index = index
        return "".join(sb)

    def extract_text(self, image: Union[bytes, np.ndarray, Any]) -> str:
        if self._rec_compiled_model is None:
            self._load_rec_model()
        if self._det_compiled_model is None:
            self._load_det_model()
            
        if isinstance(image, bytes):
            nparr = np.frombuffer(image, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # cv2.imdecode returns BGR. capture_screen returns RGB.
            # We should standardize. Let's assume input is RGB if numpy.
            # If bytes, it's BGR from cv2.
            # Let's convert BGR to RGB for consistency if bytes.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        if image is None:
            return ""

        try:
            # 1. Detection
            if self._det_compiled_model:
                # Preprocess Det (expects RGB? No, usually BGR for Paddle. But let's check normalization)
                # Paddle Det normalization: Mean [0.485, 0.456, 0.406] is for RGB usually (ImageNet).
                # But PaddleOCR usually uses BGR.
                # Let's assume RGB input to preprocess_det.
                det_img, ratio_h, ratio_w = self.preprocess_det(image)
                
                request = self._det_compiled_model.create_infer_request()
                results = request.infer({self._det_input_layer: det_img})
                det_preds = results[self._det_output_layer]
                
                boxes = self.postprocess_det(det_preds, (image.shape[0], image.shape[1], ratio_h, ratio_w))
                
                # Sort boxes (top to bottom, left to right)
                boxes = sorted(boxes, key=lambda x: (x[0][1], x[0][0]))
            else:
                # Fallback: treat whole image as one box
                h, w = image.shape[:2]
                boxes = [np.array([[0, 0], [w, 0], [w, h], [0, h]])]

            # 2. Recognition
            full_text = []
            for box in boxes:
                crop = self.get_rotate_crop_image(image, box)
                
                rec_input = self.preprocess_rec(crop)
                
                request = self._rec_compiled_model.create_infer_request()
                results = request.infer({self._rec_input_layer: rec_input})
                rec_preds = results[self._rec_output_layer]
                
                text = self.decode(rec_preds)
                if text.strip():
                    full_text.append(text)
            
            return "\n".join(full_text)
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return ""
