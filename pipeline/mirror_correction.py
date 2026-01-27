"""Stage 1: Mirrored video correction."""

import logging
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

from .utils import run_ffmpeg_command

logger = logging.getLogger(__name__)


def correct_mirrored_videos(
    paired_dirs: List[Tuple[Path, List[Path]]],
    output_folder: Path = None
) -> List[Tuple[Path, List[Path]]]:
    """
    Correct mirrored videos by flipping them horizontally.
    
    Args:
        paired_dirs: List of (directory_path, [video1_path, video2_path]) tuples
        output_folder: Output folder for corrected videos (if None, modifies in place)
        
    Returns:
        Updated list with corrected video paths
    """
    corrected_dirs = []
    
    # Create output folder structure if provided
    if output_folder:
        output_folder.mkdir(parents=True, exist_ok=True)
    
    for dir_path, video_paths in paired_dirs:
        corrected_videos = []
        
        # Determine output directory for this pair
        if output_folder:
            # Preserve relative structure in output folder
            rel_path = dir_path.relative_to(dir_path.parent.parent if dir_path.parent.parent != dir_path.parent else dir_path.parent)
            pair_output_dir = output_folder / rel_path
            pair_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            pair_output_dir = dir_path
        
        for video_path in video_paths:
            # Check if filename contains "mirrored" (case-insensitive)
            if "mirrored" in video_path.stem.lower():
                logger.info(f"Found mirrored video: {video_path.name}")
                
                # Determine output path
                if output_folder:
                    corrected_output = pair_output_dir / video_path.name
                else:
                    corrected_output = video_path.parent / video_path.name
                
                # Create temporary output path for processing
                temp_output = corrected_output.parent / f"{corrected_output.stem}_temp_corrected{corrected_output.suffix}"
                
                # Execute ffmpeg flip command
                cmd = [
                    '-i', str(video_path),
                    '-vf', 'hflip',
                    '-y',  # Overwrite output file
                    str(temp_output)
                ]
                
                logger.info(f"Flipping video: {video_path.name}")
                success = run_ffmpeg_command(cmd)
                
                if success:
                    # Move corrected video to final location
                    if corrected_output.exists():
                        corrected_output.unlink()  # Remove existing file if any
                    temp_output.rename(corrected_output)
                    logger.info(f"Successfully corrected: {video_path.name} -> {corrected_output}")
                    corrected_videos.append(corrected_output)
                else:
                    logger.error(f"Failed to correct mirrored video: {video_path.name}")
                    if temp_output.exists():
                        temp_output.unlink()
                    # On failure, copy original to output if using output folder
                    if output_folder:
                        output_video = pair_output_dir / video_path.name
                        if not output_video.exists():
                            shutil.copy2(video_path, output_video)
                        corrected_videos.append(output_video)
                    else:
                        corrected_videos.append(video_path)  # Keep original on failure
            else:
                # Not mirrored - copy to output folder if using output folder
                if output_folder:
                    output_video = pair_output_dir / video_path.name
                    if not output_video.exists():
                        shutil.copy2(video_path, output_video)
                    corrected_videos.append(output_video)
                else:
                    corrected_videos.append(video_path)
        
        corrected_dirs.append((dir_path, corrected_videos))
    
    logger.info(f"Mirror correction completed for {len(corrected_dirs)} video pairs")
    return corrected_dirs
