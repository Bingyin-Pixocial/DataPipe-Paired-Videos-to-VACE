"""Stage 6: Prompt and metadata construction."""

import csv
import json
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

PROMPT_TEXT = "The character is mimicking the dancing movements."


def format_reference_images(ref_images: Tuple[Path, Path], folder: Path) -> str:
    """
    Format reference images as JSON array string for CSV.
    
    Args:
        ref_images: Tuple of (first_frame_path, face_frame_path)
        folder: Base folder for relative paths (should be absolute/resolved)
        
    Returns:
        JSON array string with relative paths
    """
    first_frame, face_frame = ref_images
    
    # Resolve paths to absolute before computing relative paths
    first_frame = Path(first_frame).resolve()
    face_frame = Path(face_frame).resolve()
    folder = Path(folder).resolve()
    
    # Use relative paths from folder
    relative_paths = [
        str(first_frame.relative_to(folder)),
        str(face_frame.relative_to(folder))
    ]
    
    # Format as JSON array string
    json_str = json.dumps(relative_paths)
    return json_str


def build_metadata(
    reference_results: List[Tuple[Path, Path, Tuple[Path, Path], Tuple[Path, Path]]],
    output_folder: Path,
    clips_folder: Path
) -> Path:
    """
    Build metadata_vace.csv file from reference image results.
    
    Creates a global metadata file in the root output folder with all paths
    relative to the output folder root.
    
    Args:
        reference_results: List of (clip1_path, clip2_path, ref1_images, ref2_images) tuples
        output_folder: Root output folder (where metadata will be saved)
        clips_folder: Folder containing clips and reference images (usually output_folder/clips)
        
    Returns:
        Path to created metadata file
    """
    # Resolve all paths to absolute paths to avoid relative/absolute path issues
    output_folder = Path(output_folder).resolve()
    clips_folder = Path(clips_folder).resolve()
    metadata_path = output_folder / "metadata_vace.csv"
    
    rows = []
    
    for clip1_path, clip2_path, ref1_images, ref2_images in reference_results:
        # Resolve clip paths to absolute paths
        clip1_path = Path(clip1_path).resolve()
        clip2_path = Path(clip2_path).resolve()
        
        # Create two training samples:
        # 1. video: clip1_a, vace_video: clip1_b, vace_reference_image: refs from clip1_a
        # 2. video: clip1_b, vace_video: clip1_a, vace_reference_image: refs from clip1_b
        
        # All paths should be relative to output_folder root
        # Sample 1: clip1 -> clip2
        ref1_str = format_reference_images(ref1_images, output_folder)
        rows.append({
            'video': str(clip1_path.relative_to(output_folder)),
            'vace_video': str(clip2_path.relative_to(output_folder)),
            'vace_reference_image': ref1_str,
            'prompt': PROMPT_TEXT
        })
        
        # Sample 2: clip2 -> clip1
        ref2_str = format_reference_images(ref2_images, output_folder)
        rows.append({
            'video': str(clip2_path.relative_to(output_folder)),
            'vace_video': str(clip1_path.relative_to(output_folder)),
            'vace_reference_image': ref2_str,
            'prompt': PROMPT_TEXT
        })
    
    # Write CSV file
    with metadata_path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=['video', 'vace_video', 'vace_reference_image', 'prompt']
        )
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"Created global metadata file: {metadata_path} with {len(rows)} rows")
    logger.info(f"All paths are relative to: {output_folder}")
    return metadata_path
