"""Shared utilities for the paired video to VACE pipeline."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Generator

# Supported video extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}


def is_video_file(path: Path) -> bool:
    """Check if a file is a video based on extension."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.
    
    Returns:
        Dictionary with keys: fps, width, height, frame_count, duration
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info['duration'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0
    cap.release()
    return info


def read_video_frames(
    video_path: str,
    start_frame: int = 0,
    num_frames: Optional[int] = None,
    target_fps: Optional[float] = None
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Generator that yields video frames.
    
    Args:
        video_path: Path to video file
        start_frame: Frame index to start from
        num_frames: Number of frames to read (None = all remaining)
        target_fps: If set, subsample to this fps
        
    Yields:
        (frame_index, frame) tuples
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame skip for fps conversion
    frame_skip = 1
    if target_fps is not None and target_fps < original_fps:
        frame_skip = int(round(original_fps / target_fps))
    
    # Seek to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames_read = 0
    current_frame = start_frame
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for fps conversion
        if current_frame % frame_skip == 0:
            yield current_frame, frame
            frames_read += 1
            if num_frames is not None and frames_read >= num_frames:
                break
        
        current_frame += 1
    
    cap.release()


def run_ffmpeg_command(cmd: List[str]) -> bool:
    """
    Run an ffmpeg command using subprocess.
    
    Args:
        cmd: List of command arguments
        
    Returns:
        True if successful, False otherwise
    """
    import subprocess
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        result = subprocess.run(
            ['ffmpeg'] + cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # 5 minute timeout
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg command failed: {' '.join(['ffmpeg'] + cmd)}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        if e.stdout:
            logger.error(f"FFmpeg stdout: {e.stdout}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg command timed out: {' '.join(['ffmpeg'] + cmd)}")
        return False
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install ffmpeg.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running ffmpeg: {e}")
        return False
