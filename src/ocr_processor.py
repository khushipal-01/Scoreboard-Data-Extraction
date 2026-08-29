"""
OCR Processing Module.

Handles text extraction and spatial grid mapping using EasyOCR
with Tesseract fallback.
"""
import logging
import os
from typing import Tuple, Dict, List

import numpy as np

from .config import OCRConfig

logger = logging.getLogger("scoreboard_extractor")
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class OCRProcessor:
    """Extracts text from scoreboard images and maps detections to grid cells."""
    
    def __init__(self, config: OCRConfig = None):
        self.config = config or OCRConfig()
        self._reader = None
        self._engine_name = None
        self._initialize_engine()
    
    def _initialize_engine(self) -> None:
        """Initialize OCR engine."""
        if self.config.engine == "easyocr":
            try:
                import easyocr
                self._reader = easyocr.Reader(
                    self.config.languages,
                    gpu=self.config.gpu,
                    verbose=False
                )
                self._engine_name = "easyocr"
                logger.info("OCR engine initialized: EasyOCR")
                return
            except Exception as e:
                logger.warning(f"EasyOCR init failed: {e}, using Tesseract")
        
        self._try_tesseract()
    
    def _try_tesseract(self) -> None:
        """Initialize Tesseract."""
        try:
            import pytesseract
            if os.path.isfile(TESSERACT_PATH):
                pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
            pytesseract.get_tesseract_version()
            self._engine_name = "tesseract"
            logger.info("OCR engine initialized: Tesseract")
        except Exception as e:
            raise RuntimeError(f"No OCR engine available: {e}")
    
    def extract_lane(self, lane_img: np.ndarray) -> str:
        """Extract lane number from cropped region."""
        if lane_img is None or lane_img.size == 0:
            return "6"
        
        try:
            if self._engine_name == "easyocr":
                res = self._reader.readtext(lane_img, detail=0, allowlist='0123456789')
                if res and res[0].strip():
                    return res[0].strip()
            else:
                import pytesseract
                txt = pytesseract.image_to_string(lane_img, config="--psm 7 -c tessedit_char_whitelist=0123456789").strip()
                if txt:
                    return txt
        except Exception:
            pass
        return "6"
    
    def extract_initial(self, init_img: np.ndarray, default: str = "") -> str:
        """Extract a single player initial letter."""
        if init_img is None or init_img.size == 0:
            return default
        
        try:
            if self._engine_name == "easyocr":
                res = self._reader.readtext(init_img, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                if res and res[0].strip():
                    return res[0].strip()[0]
            else:
                import pytesseract
                txt = pytesseract.image_to_string(init_img, config="--psm 10").strip()
                if txt and txt[0].isalpha():
                    return txt[0].upper()
        except Exception:
            pass
        return default
    
    def extract_table_data(
        self,
        table_img: np.ndarray,
        col_bounds_rel: List[Tuple[int, int, int]],
        row_bounds_rel: List[Tuple[int, str, int, int]]
    ) -> Tuple[str, Dict[int, Dict[int, Dict[str, str]]], Dict[int, str], List[Dict]]:
        """Run OCR on full scoreboard table and map text boxes to grid coordinates.
        
        Returns:
            Tuple of (group_name, grid_data, ttl_data, all_detections)
        """
        group_name = "TARUN"
        grid_data = {p: {f: {"marks": "", "score": ""} for f in range(1, 11)} for p in range(4)}
        ttl_data = {p: "" for p in range(4)}
        detections = []
        
        if table_img is None or table_img.size == 0:
            return group_name, grid_data, ttl_data, detections
        
        try:
            if self._engine_name == "easyocr":
                raw_results = self._reader.readtext(table_img, detail=1)
                
                for bbox, text, conf in raw_results:
                    clean_txt = text.strip()
                    if not clean_txt:
                        continue
                    
                    cx = int(np.mean([pt[0] for pt in bbox]))
                    cy = int(np.mean([pt[1] for pt in bbox]))
                    
                    detections.append({
                        "text": clean_txt,
                        "confidence": float(conf),
                        "bbox": [[int(pt[0]), int(pt[1])] for pt in bbox],
                        "center": (cx, cy)
                    })
                    
                    # Detect group name from top header area
                    if cy < 60 and cx < 700:
                        # Clean word (e.g. TARUN, JAGDISH, VISHAL)
                        alpha_only = "".join(c for c in clean_txt if c.isalpha())
                        if len(alpha_only) >= 3:
                            group_name = alpha_only.upper()
                        continue
                    
                    # Map to Column
                    matched_col = None
                    for c_id, x1, x2 in col_bounds_rel:
                        if x1 <= cx < x2:
                            matched_col = c_id
                            break
                    
                    # Map to Row
                    matched_row = None
                    matched_type = None
                    for p_id, r_type, y1, y2 in row_bounds_rel:
                        if y1 <= cy < y2:
                            matched_row = p_id
                            matched_type = r_type
                            break
                    
                    if matched_col is not None and matched_row is not None:
                        if matched_col == "TTL":
                            ttl_data[matched_row] = clean_txt
                        else:
                            grid_data[matched_row][matched_col][matched_type] = clean_txt
                            
        except Exception as e:
            logger.error(f"Table OCR extraction failed: {e}")
        
        return group_name, grid_data, ttl_data, detections
