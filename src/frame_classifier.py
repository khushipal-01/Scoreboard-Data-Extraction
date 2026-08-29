"""
Frame Classification Module.

Classifies video frames as scoreboard, pin diagram, animation, or other.
Uses color histogram analysis and edge density to distinguish frame types
without any ML model.
"""
import logging
from enum import Enum
from typing import Tuple

import cv2
import numpy as np

from .config import FrameClassifierConfig

logger = logging.getLogger("scoreboard_extractor")


class FrameType(Enum):
    """Classification of video frame content."""
    SCOREBOARD = "scoreboard"       # Full scoring table visible
    PIN_DIAGRAM = "pin_diagram"     # Bowling pin layout display
    ANIMATION = "animation"         # Strike/spare animation
    OTHER = "other"                 # Logo, transition, or unrecognized


class FrameClassifier:
    """Classifies frames based on visual characteristics.
    
    The Brunswick bowling system alternates between:
    1. Scoreboard display (blue gradient background, white text grid)
    2. Pin diagrams (dark blue background, white pin shapes)
    3. Strike/spare animations (colorful, high contrast)
    4. Logo/idle screens
    
    Detection uses simple heuristics - no ML required because
    the visual differences between frame types are very distinct.
    """
    
    def __init__(self, config: FrameClassifierConfig = None):
        self.config = config or FrameClassifierConfig()
    
    def classify(self, frame: np.ndarray) -> Tuple[FrameType, float]:
        """Classify a frame into one of the known types.
        
        Args:
            frame: BGR frame image.
            
        Returns:
            Tuple of (FrameType, confidence_score)
        """
        if frame is None or frame.size == 0:
            return FrameType.OTHER, 0.0
        
        h, w = frame.shape[:2]
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Feature 1: Check for the scoreboard grid structure
        # Scoreboard frames have distinct horizontal bright bands (row dividers)
        # and vertical dividers creating a grid pattern
        has_grid = self._detect_grid_structure(gray, h, w)
        
        # Feature 2: Color distribution
        # Scoreboard: dominated by blue with cyan/white text
        # Pin diagram: mostly dark blue with white pin shapes  
        # Animation: contains large red/green areas
        blue_ratio, red_ratio, brightness = self._analyze_colors(frame, hsv)
        
        # Feature 3: Edge density in the scoreboard region
        edge_density = self._compute_edge_density(gray)
        
        # Decision logic
        if has_grid and blue_ratio > 0.3 and edge_density > self.config.min_edge_density:
            return FrameType.SCOREBOARD, 0.9
        
        if red_ratio > 0.15 and brightness > 100:
            return FrameType.ANIMATION, 0.8
        
        if blue_ratio > 0.5 and brightness < 100 and not has_grid:
            return FrameType.PIN_DIAGRAM, 0.7
        
        if blue_ratio > 0.4 and edge_density < 0.02:
            return FrameType.PIN_DIAGRAM, 0.6
        
        return FrameType.OTHER, 0.5
    
    def _detect_grid_structure(self, gray: np.ndarray, h: int, w: int) -> bool:
        """Check if the frame contains a grid-like structure (scoreboard).
        
        Looks for consistent horizontal and vertical bright lines
        characteristic of the bowling scoreboard table.
        """
        # Focus on the expected scoreboard region (top 60% of frame)
        roi = gray[:int(h * 0.6), :]
        
        # Detect horizontal lines using morphological operations
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 4, 1))
        horizontal_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Count significant horizontal line segments
        _, binary = cv2.threshold(horizontal_lines, 150, 255, cv2.THRESH_BINARY)
        row_sums = np.sum(binary > 0, axis=1)
        line_rows = np.sum(row_sums > w * 0.3)
        
        # Scoreboard typically has 5+ horizontal grid lines
        return line_rows >= self.config.min_horizontal_lines
    
    def _analyze_colors(self, frame: np.ndarray, hsv: np.ndarray) -> Tuple[float, float, float]:
        """Analyze color distribution of the frame.
        
        Returns:
            Tuple of (blue_ratio, red_ratio, mean_brightness)
        """
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        
        total_pixels = frame.shape[0] * frame.shape[1]
        
        # Blue hue range in HSV: approximately 100-130
        blue_mask = (h_channel > 95) & (h_channel < 135) & (s_channel > 50)
        blue_ratio = np.sum(blue_mask) / total_pixels
        
        # Red hue range: 0-10 and 170-180
        red_mask = ((h_channel < 10) | (h_channel > 170)) & (s_channel > 80) & (v_channel > 80)
        red_ratio = np.sum(red_mask) / total_pixels
        
        brightness = np.mean(v_channel)
        
        return blue_ratio, red_ratio, brightness
    
    def _compute_edge_density(self, gray: np.ndarray) -> float:
        """Compute the ratio of edge pixels in the frame."""
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges > 0) / edges.size
