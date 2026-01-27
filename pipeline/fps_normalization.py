"""Stage 2: FPS normalization - Set all videos to target FPS."""

import logging
import shutil
from pathlib import Path
from typing import List, Tuple

from .utils import get_video_info, run_ffmpeg_command

logger = logging.getLogger(__name__)


def normalize_fps(
    paired_dirs: List[Tuple[Path, List[Path]]],
    target_fps: float = 16.0,
    output_folder: Path = None
) -> List[Tuple[Path, List[Path]]]:
    """
    Normalize FPS of all videos to target FPS while preserving all frames.
    
    This keeps all original frames and extends the duration accordingly.
    Example: 30 fps video with 150 frames (5s) -> 16 fps video with 150 frames (9.375s)
    
    Args:
        paired_dirs: List of (directory_path, [video1_path, video2_path]) tuples
        target_fps: Target FPS (default: 16.0)
        output_folder: Output folder for normalized videos (if None, modifies in place)
        
    Returns:
        Updated list with normalized video paths
    """
    normalized_dirs = []
    
    # Create output folder structure if provided
    if output_folder:
        output_folder.mkdir(parents=True, exist_ok=True)
    
    for dir_path, video_paths in paired_dirs:
        normalized_videos = []
        
        # Determine output directory for this pair
        if output_folder:
            # Preserve relative structure in output folder
            rel_path = dir_path.relative_to(dir_path.parent.parent if dir_path.parent.parent != dir_path.parent else dir_path.parent)
            pair_output_dir = output_folder / rel_path
            pair_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            pair_output_dir = dir_path
        
        for video_path in video_paths:
            # Get current video info
            # Note: video_path comes from previous stage (mirror_corrected/ if Stage 1 ran)
            logger.debug(f"Processing video for FPS normalization: {video_path}")
            try:
                info = get_video_info(str(video_path))
                current_fps = info['fps']
            except Exception as e:
                logger.error(f"Failed to get video info for {video_path.name}: {e}")
                normalized_videos.append(video_path)
                continue
            
            # Check if FPS normalization is needed
            if abs(current_fps - target_fps) < 0.01:  # Already at target FPS (within 0.01 tolerance)
                logger.debug(f"Video {video_path.name} already at {current_fps:.2f} fps, skipping")
                # If using output folder, copy to output; otherwise use original
                if output_folder:
                    output_video = pair_output_dir / video_path.name
                    if not output_video.exists():
                        shutil.copy2(video_path, output_video)
                    normalized_videos.append(output_video)
                else:
                    normalized_videos.append(video_path)
                continue
            
            logger.info(f"Normalizing FPS for {video_path.name}: {current_fps:.2f} -> {target_fps:.2f} fps")
            logger.info(f"  Original: {info['frame_count']} frames, {info['duration']:.2f}s at {current_fps:.2f} fps")
            
            # Calculate expected duration after FPS change (keeping all frames)
            original_frames = info['frame_count']
            expected_duration = original_frames / target_fps
            logger.info(f"  Expected: {original_frames} frames, {expected_duration:.2f}s at {target_fps:.2f} fps")
            
            # Create output path (in output folder if provided, otherwise in same directory)
            if output_folder:
                normalized_output = pair_output_dir / video_path.name
            else:
                normalized_output = video_path.parent / video_path.name
            
            # Create temporary output path for processing
            temp_output = normalized_output.parent / f"{normalized_output.stem}_temp_fps{target_fps}{normalized_output.suffix}"
            
            # Execute ffmpeg fps conversion
            # To keep all original frames and extend time:
            # - Use setpts filter to adjust timestamps: PTS * (original_fps / target_fps)
            # - Set output frame rate to target_fps
            # This preserves all frames but changes playback rate (slower playback)
            pts_scale = current_fps / target_fps
            cmd = [
                '-i', str(video_path),
                '-filter:v', f'setpts=PTS*{pts_scale:.6f}',  # Adjust timestamps to slow down playback
                '-r', str(target_fps),  # Set output frame rate
                '-c:v', 'libx264',  # Re-encode video
                '-preset', 'fast',  # Fast encoding preset
                '-crf', '23',  # Good quality
                '-y',  # Overwrite output file
                str(temp_output)
            ]
            
            success = run_ffmpeg_command(cmd)
            
            if success:
                # Verify output video
                try:
                    output_info = get_video_info(str(temp_output))
                    output_fps = output_info['fps']
                    output_frames = output_info['frame_count']
                    output_duration = output_info['duration']
                    
                    # Verify frame count is preserved (within small tolerance for rounding)
                    frame_diff = abs(output_frames - original_frames)
                    frame_preserved = frame_diff <= 1  # Allow 1 frame difference for rounding
                    
                    # Verify FPS is correct (within 0.1 tolerance)
                    fps_correct = abs(output_fps - target_fps) < 0.1
                    
                    # Verify duration matches expected (within 0.1s tolerance)
                    duration_diff = abs(output_duration - expected_duration)
                    duration_correct = duration_diff < 0.1
                    
                    if frame_preserved and fps_correct:
                        # Move normalized video to final location
                        if normalized_output.exists():
                            normalized_output.unlink()  # Remove existing file if any
                        temp_output.rename(normalized_output)
                        
                        logger.info(f"Successfully normalized: {video_path.name}")
                        logger.info(f"  Output: {normalized_output}")
                        logger.info(f"  FPS: {current_fps:.2f} -> {output_fps:.2f}")
                        logger.info(f"  Frames: {original_frames} -> {output_frames} (preserved: {frame_preserved})")
                        logger.info(f"  Duration: {info['duration']:.2f}s -> {output_duration:.2f}s (expected: {expected_duration:.2f}s)")
                        if not duration_correct:
                            logger.warning(f"  Duration mismatch: expected {expected_duration:.2f}s, got {output_duration:.2f}s")
                        normalized_videos.append(normalized_output)
                    else:
                        if not frame_preserved:
                            logger.warning(f"Frame count mismatch: expected {original_frames}, got {output_frames} (difference: {frame_diff})")
                        if not fps_correct:
                            logger.warning(f"FPS conversion failed: expected {target_fps}, got {output_fps}")
                        temp_output.unlink()  # Remove failed conversion
                        # On failure, use original (or copy to output if using output folder)
                        if output_folder:
                            output_video = pair_output_dir / video_path.name
                            if not output_video.exists():
                                shutil.copy2(video_path, output_video)
                            normalized_videos.append(output_video)
                        else:
                            normalized_videos.append(video_path)
                except Exception as e:
                    logger.error(f"Failed to verify normalized video: {e}")
                    if temp_output.exists():
                        temp_output.unlink()
                    # On failure, use original (or copy to output if using output folder)
                    if output_folder:
                        output_video = pair_output_dir / video_path.name
                        if not output_video.exists():
                            shutil.copy2(video_path, output_video)
                        normalized_videos.append(output_video)
                    else:
                        normalized_videos.append(video_path)
            else:
                logger.error(f"Failed to normalize FPS for {video_path.name}")
                # On failure, use original (or copy to output if using output folder)
                if output_folder:
                    output_video = pair_output_dir / video_path.name
                    if not output_video.exists():
                        shutil.copy2(video_path, output_video)
                    normalized_videos.append(output_video)
                else:
                    normalized_videos.append(video_path)
        
        normalized_dirs.append((dir_path, normalized_videos))
    
    logger.info(f"FPS normalization completed for {len(normalized_dirs)} video pairs")
    return normalized_dirs
