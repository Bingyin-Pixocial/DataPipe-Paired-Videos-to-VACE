"""Scanner to find paired videos in folder structure."""

import logging
from pathlib import Path
from typing import List, Tuple

from .utils import is_video_file

logger = logging.getLogger(__name__)


def find_paired_video_dirs(root_path: Path) -> List[Tuple[Path, List[Path]]]:
    """
    Find directories containing exactly 2 video files (paired videos).
    
    Recursively traverses the folder structure to find lowest-level directories
    that contain exactly 2 video files.
    
    Args:
        root_path: Root directory to search
        
    Returns:
        List of tuples: (directory_path, [video1_path, video2_path])
    """
    paired_dirs = []
    
    def is_leaf_dir(path: Path) -> bool:
        """Check if directory is a leaf (no subdirectories with videos)."""
        for item in path.iterdir():
            if item.is_dir():
                # Check if subdirectory contains videos
                sub_videos = [f for f in item.iterdir() if f.is_file() and is_video_file(f)]
                if sub_videos:
                    return False
        return True
    
    # Traverse all directories
    for dir_path in root_path.rglob('*'):
        if not dir_path.is_dir():
            continue
        
        # Get all video files in this directory
        video_files = [f for f in dir_path.iterdir() if f.is_file() and is_video_file(f)]
        
        # We want directories with exactly 2 videos
        if len(video_files) == 2:
            # Check if this is a leaf directory (no subdirectories with videos)
            if is_leaf_dir(dir_path):
                paired_dirs.append((dir_path, sorted(video_files)))
                logger.debug(f"Found paired videos in: {dir_path}")
                logger.debug(f"  Videos: {[v.name for v in video_files]}")
    
    logger.info(f"Found {len(paired_dirs)} directories with paired videos")
    return paired_dirs
