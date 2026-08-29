"""
Image Preprocessing Module.

Prepares cropped cell images for OCR by resizing and padding.
"""
import logging
from typing import Optional

import cv2
import numpy as np

from .config import PreprocessingConfig

logger = logging.getLogger("scoreboard_extractor")


class ImagePreprocessor:
    """Preprocesses cell images for optimal OCR accuracy.
    
    Provides high-contrast padding and scaling to maximize EasyOCR / Tesseract accuracy.
    """
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
    
    def preprocess(self, cell_image: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess a score or mark cell image.
        
        Args:
            cell_image: BGR cropped cell image.
            
        Returns:
            Processed RGB/BGR image ready for OCR.
        """
        if cell_image is None or cell_image.size == 0:
            return None
        
        try:
            h, w = cell_image.shape[:2]
            scale = self.config.resize_scale
            resized = cv2.resize(
                cell_image, 
                (int(w * scale), int(h * scale)), 
                interpolation=cv2.INTER_CUBIC
            )
            
            # Add white border padding so characters aren't touching image edge
            p = self.config.padding
            padded = cv2.copyMakeBorder(
                resized, p, p, p, p,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255)
            )
            return padded
            
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
            return None
    
    def preprocess_for_color_text(self, cell_image: np.ndarray) -> Optional[np.ndarray]:
        """Preprocessing for header/player names."""
        return self.preprocess(cell_image)
