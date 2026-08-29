"""
Scoreboard Detection and ROI Extraction Module.

Extracts table, header, and player initial regions of interest (ROIs)
from bowling scoreboard video frames.
"""
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np

from .config import ScoreboardLayout
from .utils import crop_roi

logger = logging.getLogger("scoreboard_extractor")


@dataclass
class CellInfo:
    """Metadata about a single scoreboard cell."""
    player_index: int      # 0-3 (which player row)
    frame_index: int       # 0-9 for frames 1-10, 10 for TTL
    row_type: str          # "marks" or "score"
    roi: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


class ScoreboardDetector:
    """Extracts scoreboard regions and maps bounding boxes to grid cells.
    
    Coordinates are calibrated for Brunswick 10-frame bowling scoreboard displays.
    """
    
    def __init__(self, layout: ScoreboardLayout = None):
        self.layout = layout or ScoreboardLayout()
        self.table_roi = (70, 25, 1900, 780)  # (x1, y1, x2, y2)
        self._cell_map = self._build_cell_map()
    
    def _build_cell_map(self) -> Dict[str, CellInfo]:
        """Pre-compute ROI coordinates for all scoreboard cells."""
        cells = {}
        col_bounds = self.layout.frame_col_boundaries
        row_y_offsets = self.layout.player_row_y_offsets
        
        m_dy1, m_dy2 = self.layout.marks_y_offset
        s_dy1, s_dy2 = self.layout.score_y_offset
        
        for player_idx in range(4):
            base_y = row_y_offsets[player_idx]
            
            for frame_idx in range(11):  # 0-9 = frames 1-10, 10 = TTL
                col_left = col_bounds[frame_idx]
                col_right = col_bounds[frame_idx + 1]
                
                # Marks row
                marks_key = f"p{player_idx}_f{frame_idx}_marks"
                cells[marks_key] = CellInfo(
                    player_index=player_idx,
                    frame_index=frame_idx,
                    row_type="marks",
                    roi=(col_left, base_y + m_dy1, col_right, base_y + m_dy2)
                )
                
                # Score row
                score_key = f"p{player_idx}_f{frame_idx}_score"
                cells[score_key] = CellInfo(
                    player_index=player_idx,
                    frame_index=frame_idx,
                    row_type="score",
                    roi=(col_left, base_y + s_dy1, col_right, base_y + s_dy2)
                )
        
        return cells
    
    def extract_table(self, frame: np.ndarray) -> np.ndarray:
        """Extract the full scoreboard table area."""
        return crop_roi(frame, self.table_roi)
    
    def extract_lane(self, frame: np.ndarray) -> np.ndarray:
        """Extract lane number crop."""
        return crop_roi(frame, self.layout.lane_roi)
    
    def extract_player_initials(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        """Extract player initial crops from the left column."""
        initials = {}
        x1, x2 = self.layout.player_initial_x
        i_dy1, i_dy2 = self.layout.initial_y_offset
        
        for player_idx in range(4):
            base_y = self.layout.player_row_y_offsets[player_idx]
            roi = (x1, base_y + i_dy1, x2, base_y + i_dy2)
            initial_img = crop_roi(frame, roi)
            if initial_img is not None:
                initials[player_idx] = initial_img
        
        return initials
    
    def get_annotation_rois(self) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """Get all cell ROIs for drawing annotation boxes."""
        return [(key, info.roi) for key, info in self._cell_map.items()]
