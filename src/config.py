"""
Configuration for Bowling Scoreboard Data Extraction.

All tunable parameters are defined here to avoid hard-coded values
throughout the codebase. ROI coordinates were calibrated from the
actual input video (Brunswick bowling scoring system, 1920x1080).
"""
import os
from dataclasses import dataclass, field
from typing import Tuple, Dict, List


@dataclass
class VideoConfig:
    """Video processing parameters."""
    input_path: str = os.path.join("input", "bowling_scoreboard.mp4")
    output_dir: str = "output"
    frames_dir: str = os.path.join("output", "frames")
    annotated_video_path: str = os.path.join("output", "annotated_video.mp4")
    output_json_path: str = os.path.join("output", "extracted_data.json")
    
    # Frame sampling: process every Nth frame (30 = ~1 per second at 30fps)
    sample_interval: int = 30
    
    # Video codec for output
    output_codec: str = "mp4v"
    output_fps: float = 2.0  # Annotated video FPS (slow for readability)


@dataclass
class ScoreboardLayout:
    """ROI coordinates for the bowling scoreboard grid.
    
    Calibrated from the actual Brunswick scoring system video (1920x1080).
    
    Layout:
    - Lane number: top-left gold number
    - Group name: top header (e.g. TARUN, JAGDISH, VISHAL)
    - 4 player rows, each with 2 sub-rows (marks + cumulative score)
    - 10 frame columns + TTL column on the right
    """
    frame_width: int = 1920
    frame_height: int = 1080
    
    # Lane number ROI (x1, y1, x2, y2)
    lane_roi: Tuple[int, int, int, int] = (95, 25, 205, 125)
    
    # Group/team name ROI
    group_name_roi: Tuple[int, int, int, int] = (260, 25, 650, 85)
    
    # Player initials column X bounds
    player_initial_x: Tuple[int, int] = (75, 140)
    
    # Frame column X boundaries: [F1_left, F2_left, ..., F10_left, TTL_left, right_edge]
    frame_col_boundaries: List[int] = field(default_factory=lambda: [
        260,   # Frame 1 left
        410,   # Frame 2 left  
        550,   # Frame 3 left
        690,   # Frame 4 left
        830,   # Frame 5 left
        970,   # Frame 6 left
        1110,  # Frame 7 left
        1250,  # Frame 8 left
        1390,  # Frame 9 left
        1530,  # Frame 10 left
        1730,  # TTL left
        1900,  # TTL right
    ])
    
    # Player row Y base offsets: [Player 1, Player 2, Player 3, Player 4]
    player_row_y_offsets: List[int] = field(default_factory=lambda: [
        135,   # Player 1 top
        295,   # Player 2 top
        455,   # Player 3 top
        615,   # Player 4 top
    ])
    
    # Relative offsets within each player row (height = 160)
    marks_y_offset: Tuple[int, int] = (10, 70)     # Per-ball marks row
    score_y_offset: Tuple[int, int] = (75, 150)    # Cumulative score row
    initial_y_offset: Tuple[int, int] = (10, 70)   # Player initial letter


@dataclass
class FrameClassifierConfig:
    """Parameters for classifying frames as scoreboard vs non-scoreboard."""
    min_blue_mean: float = 80.0
    min_edge_density: float = 0.01
    max_edge_density: float = 0.20
    min_horizontal_lines: int = 3


@dataclass
class PreprocessingConfig:
    """Image preprocessing parameters for OCR."""
    resize_scale: float = 2.5
    padding: int = 15
    denoise: bool = True


@dataclass
class OCRConfig:
    """OCR engine configuration."""
    engine: str = "easyocr"  # "easyocr" or "tesseract"
    languages: List[str] = field(default_factory=lambda: ["en"])
    confidence_threshold: float = 0.3
    gpu: bool = False
    tesseract_config: str = "--psm 7 --oem 3"


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    video: VideoConfig = field(default_factory=VideoConfig)
    layout: ScoreboardLayout = field(default_factory=ScoreboardLayout)
    classifier: FrameClassifierConfig = field(default_factory=FrameClassifierConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    
    log_level: str = "INFO"
    save_debug_frames: bool = False
