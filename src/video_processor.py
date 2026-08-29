"""
Video Processing Module.

Handles video file I/O, frame extraction, and sampling.
Provides an iterator interface for efficient frame-by-frame processing.
"""
import logging
from typing import Iterator, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger("scoreboard_extractor")


class VideoProcessor:
    """Reads a video file and yields sampled frames.
    
    Attributes:
        path: Path to the input video file.
        sample_interval: Process every Nth frame.
        fps: Video frames per second.
        total_frames: Total number of frames in the video.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """
    
    def __init__(self, path: str, sample_interval: int = 30):
        self.path = path
        self.sample_interval = sample_interval
        self._cap = None
        
        # Validate and open video
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {path}")
        
        # Extract video properties
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        logger.info(
            f"Video loaded: {self.width}x{self.height}, "
            f"{self.fps:.1f} FPS, {self.total_frames} frames, "
            f"{self.duration:.1f}s duration"
        )
    
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Read a specific frame by number.
        
        Args:
            frame_number: Zero-indexed frame number.
            
        Returns:
            BGR frame as numpy array, or None if unreadable.
        """
        if frame_number < 0 or frame_number >= self.total_frames:
            logger.warning(f"Frame {frame_number} out of range [0, {self.total_frames})")
            return None
        
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self._cap.read()
        
        if not ret or frame is None:
            logger.warning(f"Failed to read frame {frame_number}")
            return None
        
        return frame
    
    def sampled_frames(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        """Yield sampled frames from the video.
        
        Yields:
            Tuple of (frame_number, timestamp_seconds, frame_bgr)
        """
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_number = 0
        
        while frame_number < self.total_frames:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self._cap.read()
            
            if not ret or frame is None:
                logger.debug(f"Skipping unreadable frame {frame_number}")
                frame_number += self.sample_interval
                continue
            
            timestamp = frame_number / self.fps if self.fps > 0 else 0
            yield frame_number, timestamp, frame
            
            frame_number += self.sample_interval
    
    def release(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.debug("Video resources released")
    
    def __del__(self):
        self.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.release()
