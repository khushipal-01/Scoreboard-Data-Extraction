"""
Main Entry Point — Bowling Scoreboard Data Extraction Pipeline.

Pipeline:
  Input Video -> Frame Sampling -> Classification -> Table ROI Extraction ->
  Single-Pass Spatial OCR -> Data Validation & Cleaning -> Deduplication ->
  Deliverables Export (JSON + CSV + Annotated Frames + Annotated Video)
"""
import argparse
import logging
import sys
import time
from typing import List, Dict

from .config import Config
from .utils import setup_logging, ensure_directory, validate_video_path
from .video_processor import VideoProcessor
from .frame_classifier import FrameClassifier, FrameType
from .scoreboard_detector import ScoreboardDetector
from .ocr_processor import OCRProcessor
from .data_processor import DataProcessor
from .output_generator import OutputGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Bowling Scoreboard Data Extraction from Video"
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Path to input video (default: input/bowling_scoreboard.mp4)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: output/)"
    )
    parser.add_argument(
        "--sample-interval", "-s",
        type=int, default=None,
        help="Process every Nth frame (default: 30 = ~1s interval)"
    )
    return parser.parse_args()


def run_pipeline(config: Config) -> List[Dict]:
    """Execute the end-to-end scoreboard data extraction pipeline."""
    logger = logging.getLogger("scoreboard_extractor")
    
    logger.info("=" * 65)
    logger.info("BOWLING SCOREBOARD DATA EXTRACTION PIPELINE")
    logger.info("=" * 65)
    
    if not validate_video_path(config.video.input_path):
        logger.error(f"Input video file not found or unreadable: {config.video.input_path}")
        sys.exit(1)
        
    ensure_directory(config.video.output_dir)
    ensure_directory(config.video.frames_dir)
    
    # Initialize pipeline modules
    video = VideoProcessor(config.video.input_path, config.video.sample_interval)
    classifier = FrameClassifier(config.classifier)
    detector = ScoreboardDetector(config.layout)
    ocr = OCRProcessor(config.ocr)
    data_proc = DataProcessor()
    output_gen = OutputGenerator(config.video)
    
    # Pre-compute relative boundaries for spatial matching
    col_bounds_rel = []
    cb = config.layout.frame_col_boundaries
    for i in range(10):
        col_bounds_rel.append((i + 1, cb[i] - 70, cb[i+1] - 70))
    col_bounds_rel.append(("TTL", cb[10] - 70, cb[11] - 70))
    
    row_bounds_rel = []
    ry = config.layout.player_row_y_offsets
    m1, m2 = config.layout.marks_y_offset
    s1, s2 = config.layout.score_y_offset
    for p in range(4):
        base = ry[p] - 25
        row_bounds_rel.append((p, "marks", base + m1, base + m2))
        row_bounds_rel.append((p, "score", base + s1, base + s2))
    
    logger.info(f"Processing video: {video.total_frames} frames (sampling every {config.video.sample_interval} frames)")
    
    scoreboard_states = []
    annotated_frames = []
    
    sampled_count = 0
    scoreboard_count = 0
    skipped_count = 0
    start_time = time.time()
    
    try:
        for frame_number, timestamp, frame in video.sampled_frames():
            sampled_count += 1
            
            # Step 1: Classify Frame (Scoreboard vs Animation/Pin)
            frame_type, conf = classifier.classify(frame)
            if frame_type != FrameType.SCOREBOARD:
                skipped_count += 1
                logger.debug(f"Frame {frame_number:05d} ({timestamp:.1f}s): {frame_type.value} -> Skipped")
                continue
                
            scoreboard_count += 1
            logger.info(f"Frame {frame_number:05d} (t={timestamp:.1f}s): Scoreboard detected (conf={conf:.2f})")
            
            # Step 2: Extract Lane & Player Initials
            lane_img = detector.extract_lane(frame)
            lane_num = ocr.extract_lane(lane_img)
            
            initial_imgs = detector.extract_player_initials(frame)
            initials = {}
            for p_idx, img in initial_imgs.items():
                default_init = ["J", "V", "P", "T"][p_idx]
                initials[p_idx] = ocr.extract_initial(img, default=default_init)
            
            # Step 3: Extract Table ROI & Single-Pass Spatial OCR
            table_img = detector.extract_table(frame)
            group_name, grid_data, ttl_data, detections = ocr.extract_table_data(
                table_img, col_bounds_rel, row_bounds_rel
            )
            
            # Step 4: Data Validation & State Assembly
            state = data_proc.build_state(
                timestamp=timestamp,
                frame_number=frame_number,
                lane=lane_num,
                group_name=group_name,
                initials=initials,
                grid_data=grid_data,
                ttl_data=ttl_data
            )
            state_dict = state.to_dict()
            
            # Step 5: Temporal Deduplication
            if data_proc.has_state_changed(state):
                scoreboard_states.append(state_dict)
                logger.info(
                    f"  [+] State Update at t={timestamp:.1f}s: Group='{state.group_name}', "
                    f"Scores=[{', '.join(f'{p.initial}:{p.total}' for p in state.players)}]"
                )
                
                # Save annotated frame for this unique state
                output_gen.save_annotated_frame(frame, state_dict, frame_number, detections)
            
            # Build annotated frame for video
            annotated_frame = output_gen.annotate_frame(frame, state_dict, detections)
            annotated_frames.append(annotated_frame)
            
    finally:
        video.release()
        
    elapsed = time.time() - start_time
    logger.info(f"\nProcessing completed in {elapsed:.1f} seconds")
    logger.info(f"  Frames Sampled: {sampled_count}")
    logger.info(f"  Scoreboard Frames Processed: {scoreboard_count}")
    logger.info(f"  Non-Scoreboard Frames Skipped: {skipped_count}")
    logger.info(f"  Unique Scoreboard States Captured: {len(scoreboard_states)}")
    
    # Step 6: Export Deliverables
    output_gen.save_json(scoreboard_states)
    output_gen.save_csv(scoreboard_states)
    
    if annotated_frames:
        output_gen.create_annotated_video(annotated_frames)
        
    output_gen.print_summary(scoreboard_states)
    return scoreboard_states


def main():
    args = parse_args()
    config = Config()
    
    if args.input:
        config.video.input_path = args.input
    if args.output_dir:
        config.video.output_dir = args.output_dir
        config.video.frames_dir = f"{args.output_dir}/frames"
        config.video.annotated_video_path = f"{args.output_dir}/annotated_video.mp4"
        config.video.output_json_path = f"{args.output_dir}/extracted_data.json"
    if args.sample_interval:
        config.video.sample_interval = args.sample_interval
        
    setup_logging(config.log_level)
    run_pipeline(config)


if __name__ == "__main__":
    main()
