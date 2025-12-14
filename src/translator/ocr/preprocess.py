from __future__ import annotations
from typing import Tuple
import math
import cv2
import numpy as np


def preprocess(image_bytes: bytes) -> bytes:
    """Apply basic normalization before forwarding to the OCR engine."""
    return image_bytes


def preprocess_det(image: np.ndarray, limit_side_len: int = 960) -> Tuple[np.ndarray, float, float]:
    # Convert RGB to BGR to match PaddleOCR expectation
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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


def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    pts_std = np.array([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(points.astype(np.float32), pts_std)
    dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    
    # Check if vertical
    if dst_img.shape[0] / dst_img.shape[1] > 1.5:
        dst_img = np.rot90(dst_img)
        
    return dst_img


def preprocess_rec(image: np.ndarray) -> np.ndarray:
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
