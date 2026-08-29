"""
Output Generation Module.

Exports extracted scoreboard data to:
1. JSON (hierarchical state timeline)
2. CSV (flattened tabular scoreboard records)
3. Annotated frames (with bounding boxes & detected labels)
4. Annotated MP4 Video
"""
import csv
import json
import logging
import os
from typing import List, Dict, Tuple

import cv2
import numpy as np

from .config import VideoConfig
from .utils import ensure_directory, draw_roi_box

logger = logging.getLogger("scoreboard_extractor")


class OutputGenerator:
    """Generates deliverables in JSON, CSV, image, and video formats."""
    
    def __init__(self, config: VideoConfig = None):
        self.config = config or VideoConfig()
        ensure_directory(self.config.output_dir)
        ensure_directory(self.config.frames_dir)
        self.csv_path = os.path.join(self.config.output_dir, "extracted_data.csv")
    
    def save_json(self, scoreboard_states: List[Dict], output_path: str = None) -> str:
        """Save extracted scoreboard states to JSON."""
        path = output_path or self.config.output_json_path
        
        data = {
            "source_video": self.config.input_path,
            "total_states_extracted": len(scoreboard_states),
            "scoreboard_states": scoreboard_states
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved extracted data JSON to: {path}")
        return path
    
    def save_csv(self, scoreboard_states: List[Dict], output_path: str = None) -> str:
        """Save extracted scoreboard states to CSV format."""
        path = output_path or self.csv_path
        
        headers = [
            "timestamp", "frame_number", "lane", "group_name", "player_initial",
            "F1_marks", "F1_score", "F2_marks", "F2_score", "F3_marks", "F3_score",
            "F4_marks", "F4_score", "F5_marks", "F5_score", "F6_marks", "F6_score",
            "F7_marks", "F7_score", "F8_marks", "F8_score", "F9_marks", "F9_score",
            "F10_marks", "F10_score", "total_score"
        ]
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for state in scoreboard_states:
                t = state.get("timestamp", 0)
                fn = state.get("frame_number", 0)
                lane = state.get("lane", "")
                grp = state.get("group_name", "")
                
                for p in state.get("players", []):
                    row = [t, fn, lane, grp, p.get("initial", "")]
                    for f in p.get("frames", []):
                        row.extend([f.get("marks", ""), f.get("score", "")])
                    row.append(p.get("total", ""))
                    writer.writerow(row)
                    
        logger.info(f"Saved extracted data CSV to: {path}")
        return path
    
    def annotate_frame(
        self,
        frame: np.ndarray,
        state_dict: Dict,
        detections: List[Dict] = None
    ) -> np.ndarray:
        """Draw bounding boxes, recognized labels, and a live scoreboard overlay."""
        annotated = frame.copy()
        
        # 1. Draw detected OCR bounding boxes
        if detections:
            for det in detections:
                bbox = det.get("bbox", [])
                text = det.get("text", "")
                conf = det.get("confidence", 0.0)
                
                if len(bbox) == 4:
                    # Offset by table ROI start (x=70, y=25)
                    pts = np.array([[pt[0] + 70, pt[1] + 25] for pt in bbox], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    
                    color = (0, 255, 0) if conf > 0.6 else (0, 255, 255)
                    cv2.polylines(annotated, [pts], True, color, 2)
                    
                    # Text label
                    x, y = pts[0][0]
                    cv2.putText(
                        annotated, f"{text} ({int(conf*100)}%)",
                        (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA
                    )
        
        # 2. Draw Bottom HUD overlay showing active state summary
        h, w = annotated.shape[:2]
        hud_h = 75
        hud_overlay = annotated.copy()
        cv2.rectangle(hud_overlay, (0, h - hud_h), (w, h), (20, 20, 20), -1)
        annotated = cv2.addWeighted(annotated, 0.25, hud_overlay, 0.75, 0)
        
        # HUD text info
        t = state_dict.get("timestamp", 0)
        grp = state_dict.get("group_name", "TARUN")
        lane = state_dict.get("lane", "6")
        
        info_str = f"Time: {t:.1f}s | Lane: {lane} | Group: {grp}"
        cv2.putText(
            annotated, info_str, (30, h - 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
        )
        
        # Player totals summary on HUD
        players_summary = "  |  ".join([
            f"{p.get('initial', '?')}: {p.get('total', '0')} pts"
            for p in state_dict.get("players", [])
        ])
        cv2.putText(
            annotated, f"Scores:  {players_summary}", (30, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
        
        return annotated
    
    def save_annotated_frame(
        self,
        frame: np.ndarray,
        state_dict: Dict,
        frame_number: int,
        detections: List[Dict] = None
    ) -> str:
        """Save a single annotated frame image."""
        annotated = self.annotate_frame(frame, state_dict, detections)
        filename = f"annotated_frame_{frame_number:06d}.png"
        filepath = os.path.join(self.config.frames_dir, filename)
        cv2.imwrite(filepath, annotated)
        return filepath
    
    def create_annotated_video(
        self,
        annotated_frames: List[np.ndarray],
        output_path: str = None
    ) -> str:
        """Render annotated frames to video file."""
        path = output_path or self.config.annotated_video_path
        
        if not annotated_frames:
            logger.warning("No frames to write to annotated video")
            return path
            
        h, w = annotated_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.config.output_codec)
        writer = cv2.VideoWriter(path, fourcc, self.config.output_fps, (w, h))
        
        if not writer.isOpened():
            # Fallback to alternative codec
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(path, fourcc, self.config.output_fps, (w, h))
            
        try:
            for frame in annotated_frames:
                writer.write(frame)
            logger.info(f"Saved annotated video ({len(annotated_frames)} frames) to: {path}")
        finally:
            writer.release()
            
        return path
    
    def print_summary(self, scoreboard_states: List[Dict]) -> None:
        """Print structured extraction summary to console."""
        print("\n" + "=" * 75)
        print("BOWLING SCOREBOARD EXTRACTION SUMMARY")
        print("=" * 75)
        print(f"Total Scoreboard State Transitions Detected: {len(scoreboard_states)}")
        
        for i, state in enumerate(scoreboard_states):
            print(f"\n--- State {i+1} [t={state['timestamp']:.1f}s | Frame {state['frame_number']}] ---")
            print(f"  Lane: {state.get('lane', '6')}  |  Group: {state.get('group_name', 'N/A')}")
            
            for p in state.get('players', []):
                init = p.get('initial', '?')
                ttl = p.get('total', '0')
                frames_str = " ".join([
                    f"[{f.get('marks',' ')}/{f.get('score',' ')}]"
                    for f in p.get('frames', [])
                ])
                print(f"  Player {init}: {frames_str}  |  TTL: {ttl}")
        print("\n" + "=" * 75)
