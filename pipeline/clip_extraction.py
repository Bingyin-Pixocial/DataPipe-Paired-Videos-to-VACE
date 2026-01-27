"""Stage 3: Cut videos into clips."""

import logging
from pathlib import Path
from typing import List, Tuple

from .utils import get_video_info, run_ffmpeg_command

logger = logging.getLogger(__name__)


def extract_clips(
    paired_dirs: List[Tuple[Path, List[Path]]],
    num_frames: int,
    num_stride: int,
    output_folder: Path
) -> List[Tuple[Path, Path]]:
    """
    Extract clips from paired video directories using sliding window.
    
    Args:
        paired_dirs: List of (dir_path, [video1_path, video2_path]) tuples
        num_frames: Number of frames per clip
        num_stride: Stride between clip starts
        output_folder: Output directory for clips
        
    Returns:
        List of (clip1_path, clip2_path) tuples for all extracted clip pairs
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    all_clips = []
    
    # Track clip index across all video pairs to ensure unique names
    global_clip_idx = 1
    
    for dir_path, video_paths in paired_dirs:
        if len(video_paths) != 2:
            logger.warning(f"Skipping directory with {len(video_paths)} videos (expected 2): {dir_path}")
            continue
        
        video1_path = video_paths[0]
        video2_path = video_paths[1]
        
        # Note: video_paths come from previous stage (fps_normalized/ if Stage 2 ran)
        logger.info(f"Extracting clips from: {dir_path}")
        logger.debug(f"  Using video1: {video1_path}")
        logger.debug(f"  Using video2: {video2_path}")
        
        # Check if video files exist and are readable
        if not video1_path.exists():
            logger.error(f"Video file does not exist: {video1_path}")
            continue
        if not video2_path.exists():
            logger.error(f"Video file does not exist: {video2_path}")
            continue
        
        # Check file sizes (corrupted files might be very small)
        size1 = video1_path.stat().st_size
        size2 = video2_path.stat().st_size
        if size1 < 1000:  # Less than 1KB is suspicious
            logger.error(f"Video file is suspiciously small (possibly corrupted): {video1_path} ({size1} bytes)")
            continue
        if size2 < 1000:
            logger.error(f"Video file is suspiciously small (possibly corrupted): {video2_path} ({size2} bytes)")
            continue
        
        # Get video lengths
        try:
            info1 = get_video_info(str(video1_path))
            info2 = get_video_info(str(video2_path))
            length1 = info1['frame_count']
            length2 = info2['frame_count']
            
            # Validate video info
            if length1 <= 0 or length2 <= 0:
                logger.error(f"Invalid video frame count for {dir_path}: video1={length1}, video2={length2}")
                continue
            
            logger.info(f"  Video lengths: {length1} frames (video1), {length2} frames (video2)")
        except Exception as e:
            logger.error(f"Failed to get video info for {dir_path}: {e}")
            logger.error(f"  Video1: {video1_path} (exists: {video1_path.exists()}, size: {video1_path.stat().st_size if video1_path.exists() else 'N/A'} bytes)")
            logger.error(f"  Video2: {video2_path} (exists: {video2_path.exists()}, size: {video2_path.stat().st_size if video2_path.exists() else 'N/A'} bytes)")
            logger.error(f"  This video pair will be skipped. The video file may be corrupted or incomplete.")
            continue
        
        # Use minimum length to ensure both videos are processed consistently
        video_length = min(length1, length2)
        
        # Handle corner cases
        if num_stride > video_length:
            # Stride exceeds video length: extract single clip from start
            logger.info(f"  Stride ({num_stride}) > video length ({video_length}): extracting single clip")
            clip_pairs = extract_single_clip_pair(
                video1_path, video2_path, video_length, num_frames, output_folder, clip_idx=global_clip_idx
            )
            all_clips.extend(clip_pairs)
            global_clip_idx += 1
            continue
        
        if video_length < num_stride:
            # Video shorter than stride: extract single clip covering entire video
            logger.info(f"  Video length ({video_length}) < stride ({num_stride}): extracting single clip")
            clip_pairs = extract_single_clip_pair(
                video1_path, video2_path, video_length, num_frames, output_folder, clip_idx=global_clip_idx
            )
            all_clips.extend(clip_pairs)
            global_clip_idx += 1
            continue
        
        # Normal sliding window extraction
        start_frame = 0
        
        while start_frame < video_length:
            # Calculate actual clip length (may be shorter at the end)
            remaining_frames = video_length - start_frame
            actual_clip_frames = min(num_frames, remaining_frames)
            
            if actual_clip_frames < num_frames:
                # Last clip may be shorter
                logger.info(f"  Extracting final clip: start={start_frame}, frames={actual_clip_frames}")
            
            clip_pairs = extract_clip_pair(
                video1_path, video2_path, start_frame, actual_clip_frames,
                output_folder, global_clip_idx
            )
            all_clips.extend(clip_pairs)
            
            global_clip_idx += 1
            start_frame += num_stride
            
            # Stop if next clip would be too short
            if start_frame + num_frames > video_length:
                # Check if we should extract a final shorter clip
                if start_frame < video_length:
                    remaining = video_length - start_frame
                    if remaining >= num_frames // 2:  # Extract if at least half the desired length
                        clip_pairs = extract_clip_pair(
                            video1_path, video2_path, start_frame, remaining,
                            output_folder, global_clip_idx
                        )
                        all_clips.extend(clip_pairs)
                        global_clip_idx += 1
                break
    
    logger.info(f"Extracted {len(all_clips)} clip pairs")
    return all_clips


def extract_clip_pair(
    video1_path: Path,
    video2_path: Path,
    start_frame: int,
    num_frames: int,
    output_folder: Path,
    clip_idx: int
) -> List[Tuple[Path, Path]]:
    """
    Extract a single clip pair from two videos.
    
    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        start_frame: Starting frame index
        num_frames: Number of frames to extract
        output_folder: Output directory
        clip_idx: Clip index for naming
        
    Returns:
        List with single (clip1_path, clip2_path) tuple
    """
    info1 = get_video_info(str(video1_path))
    info2 = get_video_info(str(video2_path))
    
    fps1 = info1['fps']
    fps2 = info2['fps']
    
    start_time1 = start_frame / fps1
    duration1 = num_frames / fps1
    start_time2 = start_frame / fps2
    duration2 = num_frames / fps2
    
    # Create output paths (ensure absolute paths)
    clip1_name = f"clip{clip_idx}_a{video1_path.suffix}"
    clip2_name = f"clip{clip_idx}_b{video2_path.suffix}"
    
    clip1_path = (output_folder / clip1_name).resolve()
    clip2_path = (output_folder / clip2_name).resolve()
    
    # Extract clip from video 1
    # Use frame-accurate extraction with re-encoding (not -c copy) to ensure accuracy
    # -c copy can fail if trimming doesn't align with keyframes
    cmd1 = [
        '-i', str(video1_path),
        '-ss', str(start_time1),
        '-t', str(duration1),
        '-avoid_negative_ts', 'make_zero',  # Handle timestamp issues
        '-y',
        str(clip1_path)
    ]
    
    logger.debug(f"Extracting clip1: start={start_time1:.3f}s, duration={duration1:.3f}s, frames={num_frames}")
    success1 = run_ffmpeg_command(cmd1)
    if not success1:
        logger.error(f"Failed to extract clip from video 1: {video1_path}")
        return []
    
    # Verify clip1 was created and is valid
    if not clip1_path.exists():
        logger.error(f"Clip file was not created: {clip1_path}")
        return []
    
    # Check if file is readable
    try:
        info = get_video_info(str(clip1_path))
        if info['frame_count'] == 0:
            logger.error(f"Clip file is empty or corrupted: {clip1_path}")
            clip1_path.unlink()
            return []
        logger.debug(f"Clip1 created successfully: {info['frame_count']} frames")
    except Exception as e:
        logger.error(f"Clip file cannot be opened: {clip1_path}, error: {e}")
        if clip1_path.exists():
            clip1_path.unlink()
        return []
    
    # Extract clip from video 2
    cmd2 = [
        '-i', str(video2_path),
        '-ss', str(start_time2),
        '-t', str(duration2),
        '-avoid_negative_ts', 'make_zero',  # Handle timestamp issues
        '-y',
        str(clip2_path)
    ]
    
    logger.debug(f"Extracting clip2: start={start_time2:.3f}s, duration={duration2:.3f}s, frames={num_frames}")
    success2 = run_ffmpeg_command(cmd2)
    if not success2:
        logger.error(f"Failed to extract clip from video 2: {video2_path}")
        if clip1_path.exists():
            clip1_path.unlink()  # Clean up partial output
        return []
    
    # Verify clip2 was created and is valid
    if not clip2_path.exists():
        logger.error(f"Clip file was not created: {clip2_path}")
        if clip1_path.exists():
            clip1_path.unlink()
        return []
    
    # Check if file is readable
    try:
        info = get_video_info(str(clip2_path))
        if info['frame_count'] == 0:
            logger.error(f"Clip file is empty or corrupted: {clip2_path}")
            clip2_path.unlink()
            if clip1_path.exists():
                clip1_path.unlink()
            return []
        logger.debug(f"Clip2 created successfully: {info['frame_count']} frames")
    except Exception as e:
        logger.error(f"Clip file cannot be opened: {clip2_path}, error: {e}")
        if clip2_path.exists():
            clip2_path.unlink()
        if clip1_path.exists():
            clip1_path.unlink()
        return []
    
    logger.debug(f"Successfully extracted clip pair: {clip1_path.name}, {clip2_path.name}")
    return [(clip1_path, clip2_path)]


def extract_single_clip_pair(
    video1_path: Path,
    video2_path: Path,
    video_length: int,
    num_frames: int,
    output_folder: Path,
    clip_idx: int
) -> List[Tuple[Path, Path]]:
    """
    Extract a single clip pair covering the entire video (or first num_frames).
    
    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        video_length: Length of video in frames
        num_frames: Desired number of frames (may be more than video_length)
        output_folder: Output directory
        clip_idx: Clip index for naming
        
    Returns:
        List with single (clip1_path, clip2_path) tuple
    """
    actual_frames = min(num_frames, video_length)
    return extract_clip_pair(
        video1_path, video2_path, 0, actual_frames, output_folder, clip_idx
    )
