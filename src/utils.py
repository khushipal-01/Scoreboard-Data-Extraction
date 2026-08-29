"""
Utility functions for the Scoreboard Data Extraction pipeline.

Contains common helpers used across multiple modules.
"""
import os
import logging
from typing import Optional

import cv2
import numpy as np


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the application logger."""
    logger = logging.getLogger("scoreboard_extractor")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def ensure_directory(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def validate_video_path(path: str) -> bool:
    """Check if a video file exists and is readable."""
    if not os.path.isfile(path):
        return False
    cap = cv2.VideoCapture(path)
    is_valid = cap.isOpened()
    cap.release()
    return is_valid


def crop_roi(frame: np.ndarray, roi: tuple) -> Optional[np.ndarray]:
    """Extract a Region of Interest from a frame.
    
    Args:
        frame: Source image (BGR or grayscale)
        roi: (x1, y1, x2, y2) coordinates
    
    Returns:
        Cropped image, or None if ROI is invalid
    """
    x1, y1, x2, y2 = roi
    h, w = frame.shape[:2]
    
    # Clamp to frame boundaries
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    return frame[y1:y2, x1:x2].copy()


def draw_roi_box(frame: np.ndarray, roi: tuple, 
                 color: tuple = (0, 255, 0), 
                 thickness: int = 2,
                 label: str = "") -> np.ndarray:
    """Draw a bounding box on a frame for visualization."""
    x1, y1, x2, y2 = roi
    annotated = frame.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
    
    if label:
        font_scale = 0.5
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                       font_scale, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), 
                     color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
    
    return annotated
