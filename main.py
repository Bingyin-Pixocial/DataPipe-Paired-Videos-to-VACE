#!/usr/bin/env python3
"""
Paired Video to VACE Data Processing Pipeline

This pipeline processes paired videos through 7 stages:
0. Video trimming (trim videos longer than max_length)
1. Mirrored video correction
2. FPS normalization
3. Clip extraction
4. Quality check and screening (motion similarity + DTW)
5. Reference image extraction
6. Metadata construction

Usage:
    python main.py --root_folder /path/to/videos --output_folder /path/to/output
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.scanner import find_paired_video_dirs
from pipeline.video_trimming import trim_long_videos
from pipeline.mirror_correction import correct_mirrored_videos
from pipeline.fps_normalization import normalize_fps
from pipeline.clip_extraction import extract_clips
from pipeline.quality_check import screen_clips
from pipeline.reference_extraction import extract_reference_images
from pipeline.metadata_builder import build_metadata


def setup_logging(log_level: str = "INFO", log_file: Path = None):
    """Setup logging configuration with both console and file handlers.
    
    Args:
        log_level: Logging level (e.g., "INFO", "DEBUG")
        log_file: Optional path to log file. If None, no file logging.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Paired video to VACE data processing pipeline'
    )
    parser.add_argument(
        '--root_folder',
        type=str,
        required=True,
        help='Root folder containing paired videos in subfolders'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help='Output folder for processed clips and metadata'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=81,
        help='Number of frames per clip (default: 81)'
    )
    parser.add_argument(
        '--num_stride',
        type=int,
        default=30,
        help='Stride between clip starting frames (default: 30)'
    )
    parser.add_argument(
        '--motion_similarity_threshold',
        type=float,
        default=0.8,
        help='Minimum motion similarity score threshold (default: 0.8)'
    )
    parser.add_argument(
        '--dtw_similarity_threshold',
        type=float,
        default=0.9,
        help='Minimum DTW-based similarity score threshold (default: 0.9)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=16.0,
        help='Target FPS for processing (default: 16.0)'
    )
    parser.add_argument(
        '--max_video_length',
        type=float,
        default=15.0,
        help='Maximum video length in seconds. Videos longer than this will be trimmed from the end (default: 15.0)'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--skip_stages',
        nargs='+',
        type=float,
        default=[],
        help='Skip specific stages (0, 1, 2, 3-6). Use 0 to skip video trimming, 2 to skip FPS normalization'
    )
    
    return parser.parse_args()


def run_pipeline(config: PipelineConfig, skip_stages: list = None):
    """
    Run the complete pipeline.
    
    Args:
        config: Pipeline configuration
        skip_stages: List of stage numbers to skip (0, 1, 2, 3-6)
    """
    if skip_stages is None:
        skip_stages = []
    
    logger = logging.getLogger(__name__)
    root_path = Path(config.root_folder)
    output_path = Path(config.output_folder)
    
    # Log file location (already set up in main, but log it here too)
    log_file = output_path / "pipeline.log"
    logger.info(f"Log file: {log_file}")
    
    logger.info("=" * 70)
    logger.info("Paired Video to VACE Pipeline")
    logger.info("=" * 70)
    logger.info(f"Root folder: {root_path}")
    logger.info(f"Output folder: {output_path}")
    logger.info(f"Skip stages: {skip_stages}")
    logger.info("=" * 70)
    
    # Scan for paired videos (pre-stage)
    logger.info("\n[Scanning] Finding paired videos...")
    paired_dirs = find_paired_video_dirs(root_path)
    if not paired_dirs:
        logger.error("No paired videos found!")
        return False
    
    logger.info(f"Found {len(paired_dirs)} directories with paired videos")
    
    # Stage 0: Video trimming
    if 0 not in skip_stages:
        logger.info(f"\n[Stage 0] Trimming videos longer than {config.max_video_length}s...")
        # Save trimmed videos to output folder to keep original dataset clean
        trim_output_folder = output_path / "trimmed"
        paired_dirs = trim_long_videos(paired_dirs, max_length=config.max_video_length, output_folder=trim_output_folder)
    else:
        logger.info("\n[Stage 0] Skipped (video trimming)")
        # When skipping, use original videos directly (no folder creation)
        # paired_dirs already contains the correct paths from scanning
    
    # Stage 1: Mirrored video correction
    if 1 not in skip_stages:
        logger.info("\n[Stage 1] Correcting mirrored videos...")
        # Save corrected videos to output folder to keep original dataset clean
        # Use trimmed folder as input if Stage 0 ran, otherwise use original videos
        mirror_output_folder = output_path / "mirror_corrected"
        paired_dirs = correct_mirrored_videos(paired_dirs, output_folder=mirror_output_folder)
    else:
        logger.info("\n[Stage 1] Skipped (mirror correction)")
        # When skipping, use videos from previous stage directly (no folder creation)
        # paired_dirs already contains the correct paths from previous stage
    
    # Stage 2: FPS normalization
    if 2 not in skip_stages:
        logger.info("\n[Stage 2] Normalizing video FPS to {} fps...".format(config.fps))
        # Normalize FPS and save to output folder to keep original dataset clean
        # Use mirror_corrected folder as input if Stage 1 ran, otherwise use original videos
        fps_output_folder = output_path / "fps_normalized"
        paired_dirs = normalize_fps(paired_dirs, target_fps=config.fps, output_folder=fps_output_folder)
    else:
        logger.info("\n[Stage 2] Skipped (FPS normalization)")
        # When skipping, use videos from previous stage directly (no folder creation)
        # paired_dirs already contains the correct paths from previous stage
    
    # Stage 3: Clip extraction
    if 3 not in skip_stages:
        logger.info("\n[Stage 3] Extracting clips...")
        clip_pairs = extract_clips(
            paired_dirs,
            config.num_frames,
            config.num_stride,
            output_path / "clips"
        )
    else:
        logger.info("\n[Stage 3] Skipped (clip extraction)")
        clip_pairs = []
    
    if not clip_pairs:
        logger.error("No clips extracted!")
        return False
    
    # Stage 4: Quality check and screening
    if 4 not in skip_stages:
        logger.info(f"\n[Stage 4] Screening clips for quality (using motion similarity + DTW)...")
        logger.info(f"  Motion similarity threshold: {config.motion_similarity_threshold:.2f}")
        logger.info(f"  DTW similarity threshold: {config.dtw_similarity_threshold:.2f}")
        logger.info("  Multi-person detection and segmentation enabled for quality check")
        
        # Create detector instances for multi-person handling (will be reused across clips)
        from pipeline.yolo_person_detector import YOLOPersonDetector
        from pipeline.multi_person import MultiPersonHandler
        
        yolo_detector = YOLOPersonDetector()
        multi_person_handler = MultiPersonHandler(yolo_detector=yolo_detector)
        
        # Path for saving quality scores CSV
        scores_csv_path = output_path / "quality_scores.csv"
        
        qualified_clips = screen_clips(
            clip_pairs,
            config.motion_similarity_threshold,
            config.dtw_similarity_threshold,
            output_path / "clips",
            yolo_detector=yolo_detector,
            multi_person_handler=multi_person_handler,
            scores_csv_path=scores_csv_path
        )
        
        # Cleanup detectors
        try:
            yolo_detector.cleanup()
            multi_person_handler.cleanup()
        except Exception as e:
            logger.debug(f"Error during detector cleanup: {e}")
    else:
        logger.info("\n[Stage 4] Skipped (quality check)")
        qualified_clips = clip_pairs
    
    if not qualified_clips:
        logger.error("No qualified clips found!")
        return False
    
    logger.info(f"Qualified clips: {len(qualified_clips)}")
    
    # Stage 5: Reference image extraction
    if 5 not in skip_stages:
        logger.info("\n[Stage 5] Extracting reference images...")
        reference_results = extract_reference_images(
            qualified_clips,
            output_path / "clips"
        )
    else:
        logger.info("\n[Stage 5] Skipped (reference extraction)")
        reference_results = []
    
    if not reference_results:
        logger.error("No reference images extracted!")
        return False
    
    # Stage 6: Metadata construction
    if 6 not in skip_stages:
        logger.info("\n[Stage 6] Building metadata...")
        metadata_path = build_metadata(
            reference_results,
            output_path,  # Root output folder
            output_path / "clips"  # Clips folder
        )
        logger.info(f"Global metadata saved to: {metadata_path}")
        logger.info(f"Metadata contains {len(reference_results) * 2} training samples")
    else:
        logger.info("\n[Stage 6] Skipped (metadata construction)")
    
    logger.info("\n" + "=" * 70)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 70)
    
    return True


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output folder first (needed for log file)
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging with file output
    log_file = output_path / "pipeline.log"
    setup_logging(args.log_level, log_file=log_file)
    logger = logging.getLogger(__name__)
    logger.info(f"Pipeline started. Log file: {log_file}")
    
    config = PipelineConfig(
        root_folder=args.root_folder,
        output_folder=args.output_folder,
        num_frames=args.num_frames,
        num_stride=args.num_stride,
        motion_similarity_threshold=args.motion_similarity_threshold,
        dtw_similarity_threshold=args.dtw_similarity_threshold,
        fps=args.fps,
        max_video_length=args.max_video_length
    )
    
    success = run_pipeline(config, skip_stages=args.skip_stages)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
