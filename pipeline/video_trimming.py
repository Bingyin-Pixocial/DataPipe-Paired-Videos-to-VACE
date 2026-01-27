"""Stage 0: Video trimming - Trim videos longer than max_length to max_length."""

import logging
import shutil
from pathlib import Path
from typing import List, Tuple

from .utils import get_video_info, run_ffmpeg_command

logger = logging.getLogger(__name__)


def trim_long_videos(
    paired_dirs: List[Tuple[Path, List[Path]]],
    max_length: float = 15.0,
    output_folder: Path = None
) -> List[Tuple[Path, List[Path]]]:
    """
    Trim videos longer than max_length to max_length by trimming from the end.
    Videos shorter than max_length are kept as-is.
    
    Args:
        paired_dirs: List of (directory_path, [video1_path, video2_path]) tuples
        max_length: Maximum video length in seconds (default: 15.0)
        output_folder: Output folder for trimmed videos (if None, modifies in place)
        
    Returns:
        Updated list with trimmed video paths
    """
    trimmed_dirs = []
    
    # Create output folder structure if provided
    if output_folder:
        output_folder.mkdir(parents=True, exist_ok=True)
    
    for dir_path, video_paths in paired_dirs:
        trimmed_videos = []
        
        # Determine output directory for this pair
        if output_folder:
            # Preserve relative structure in output folder
            rel_path = dir_path.relative_to(dir_path.parent.parent if dir_path.parent.parent != dir_path.parent else dir_path.parent)
            pair_output_dir = output_folder / rel_path
            pair_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            pair_output_dir = dir_path
        
        for video_path in video_paths:
            # Get video info
            try:
                info = get_video_info(str(video_path))
                duration = info['duration']
            except Exception as e:
                logger.error(f"Failed to get video info for {video_path.name}: {e}")
                # On error, copy original to output if using output folder
                if output_folder:
                    output_video = pair_output_dir / video_path.name
                    if not output_video.exists():
                        shutil.copy2(video_path, output_video)
                    trimmed_videos.append(output_video)
                else:
                    trimmed_videos.append(video_path)
                continue
            
            # Check if trimming is needed
            if duration <= max_length:
                logger.debug(f"Video {video_path.name} ({duration:.2f}s) is within limit ({max_length}s), keeping as-is")
                # If using output folder, copy to output; otherwise use original
                if output_folder:
                    output_video = pair_output_dir / video_path.name
                    if not output_video.exists():
                        shutil.copy2(video_path, output_video)
                    trimmed_videos.append(output_video)
                else:
                    trimmed_videos.append(video_path)
                continue
            
            # Trim video from the end
            trim_amount = duration - max_length
            logger.info(f"Trimming {video_path.name}: {duration:.2f}s -> {max_length:.2f}s (trimming last {trim_amount:.2f}s)")
            
            # Create output path
            if output_folder:
                trimmed_output = pair_output_dir / video_path.name
            else:
                trimmed_output = video_path.parent / video_path.name
            
            # Create temporary output path for processing
            temp_output = trimmed_output.parent / f"{trimmed_output.stem}_temp_trimmed{trimmed_output.suffix}"
            
            # Execute ffmpeg trim command: trim from start to max_length
            cmd = [
                '-i', str(video_path),
                '-t', str(max_length),  # Duration from start
                '-c', 'copy',  # Use stream copy for speed (no re-encoding)
                '-y',  # Overwrite output file
                str(temp_output)
            ]
            
            success = run_ffmpeg_command(cmd)
            
            if success:
                # Verify output video
                try:
                    output_info = get_video_info(str(temp_output))
                    output_duration = output_info['duration']
                    
                    # Verify duration is correct (within 0.1s tolerance)
                    duration_diff = abs(output_duration - max_length)
                    if duration_diff < 0.1:
                        # Move trimmed video to final location
                        if trimmed_output.exists():
                            trimmed_output.unlink()  # Remove existing file if any
                        temp_output.rename(trimmed_output)
                        logger.info(f"Successfully trimmed: {video_path.name} -> {trimmed_output} ({output_duration:.2f}s)")
                        trimmed_videos.append(trimmed_output)
                    else:
                        logger.warning(f"Trimmed video duration mismatch: expected {max_length:.2f}s, got {output_duration:.2f}s")
                        # Still use it, but log warning
                        if trimmed_output.exists():
                            trimmed_output.unlink()
                        temp_output.rename(trimmed_output)
                        trimmed_videos.append(trimmed_output)
                except Exception as e:
                    logger.error(f"Failed to verify trimmed video: {e}")
                    # On verification failure, try to use the temp file
                    if temp_output.exists():
                        if trimmed_output.exists():
                            trimmed_output.unlink()
                        temp_output.rename(trimmed_output)
                        trimmed_videos.append(trimmed_output)
                    else:
                        # Fallback: copy original
                        if output_folder:
                            output_video = pair_output_dir / video_path.name
                            if not output_video.exists():
                                shutil.copy2(video_path, output_video)
                            trimmed_videos.append(output_video)
                        else:
                            trimmed_videos.append(video_path)
            else:
                logger.error(f"Failed to trim video: {video_path.name}")
                if temp_output.exists():
                    temp_output.unlink()
                # On failure, copy original to output if using output folder
                if output_folder:
                    output_video = pair_output_dir / video_path.name
                    if not output_video.exists():
                        shutil.copy2(video_path, output_video)
                    trimmed_videos.append(output_video)
                else:
                    trimmed_videos.append(video_path)
        
        trimmed_dirs.append((dir_path, trimmed_videos))
    
    logger.info(f"Video trimming completed for {len(trimmed_dirs)} video pairs")
    return trimmed_dirs
