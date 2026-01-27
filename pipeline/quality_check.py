"""Stage 4: Video clip quality check and screening.

Uses optical flow-based motion features and DTW similarity (from motion similarity notebook).
For multi-person videos, segments the center person before computing motion features.
"""

import cv2
import logging
import shutil
import numpy as np
import tempfile
import csv
from pathlib import Path
from typing import List, Tuple, Optional

from .utils import get_video_info
from .yolo_person_detector import YOLOPersonDetector
from .multi_person import MultiPersonHandler

logger = logging.getLogger(__name__)


def extract_motion_features_fast(video_path: str, step: int = 10, resize_width: int = 320) -> np.ndarray:
    """
    Extract motion features from video using optical flow (from motion similarity notebook).
    
    Args:
        video_path: Path to video file
        step: Frame step (process every Nth frame)
        resize_width: Width to resize frames for faster processing
        
    Returns:
        Array of shape (N, 2) with [magnitude, angle] motion vectors
    """
    cap = cv2.VideoCapture(video_path)
    motion_vectors = []
    ret, prev_frame = cap.read()

    if not ret:
        cap.release()
        raise ValueError("Cannot read video file.")

    prev_frame = cv2.resize(prev_frame, (resize_width, int(prev_frame.shape[0] * resize_width / prev_frame.shape[1])))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % step != 0:
            continue

        frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * resize_width / frame.shape[1])))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using a lower resolution and fewer pyramid levels
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 2, 9, 2, 5, 1.1, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_vectors.append([np.mean(magnitude), np.mean(angle)])
        prev_gray = gray

    cap.release()
    return np.array(motion_vectors)


def get_magnitude_dtw_path(mag1: np.ndarray, mag2: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Compute DTW path for magnitude sequences to find optimal pattern alignment (from notebook).
    
    Args:
        mag1: Magnitude sequence 1
        mag2: Magnitude sequence 2
        
    Returns:
        Tuple of (path, dist_matrix)
    """
    if len(mag1) == 0 or len(mag2) == 0:
        return [], np.array([])
    
    # Normalize magnitudes to [0, 1] to focus on pattern shape, not absolute values
    mag1_norm = (mag1 - mag1.min()) / (mag1.max() - mag1.min() + 1e-8)
    mag2_norm = (mag2 - mag2.min()) / (mag2.max() - mag2.min() + 1e-8)
    
    # Build distance matrix using absolute difference of normalized magnitudes
    dist_matrix = np.zeros((len(mag1_norm), len(mag2_norm)))
    for i in range(len(mag1_norm)):
        for j in range(len(mag2_norm)):
            dist_matrix[i, j] = abs(mag1_norm[i] - mag2_norm[j])
    
    # DTW algorithm
    dtw = np.full_like(dist_matrix, np.inf)
    dtw[0, 0] = dist_matrix[0, 0]

    for i in range(len(mag1_norm)):
        for j in range(len(mag2_norm)):
            cost = dist_matrix[i, j]
            if i == 0 and j == 0:
                continue
            elif i == 0:
                dtw[i, j] = cost + dtw[i, j - 1]
            elif j == 0:
                dtw[i, j] = cost + dtw[i - 1, j]
            else:
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    # Backtrack DTW path
    i, j = len(mag1_norm) - 1, len(mag2_norm) - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = np.argmin([dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1]])
            if step == 0:
                i -= 1
            elif step == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))
    path.reverse()
    
    return path, dist_matrix


def get_dtw_path(m1: np.ndarray, m2: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """
    Compute DTW path between two motion sequences (from notebook).
    
    Args:
        m1: Motion sequence 1, shape (N, 2) with [magnitude, angle]
        m2: Motion sequence 2, shape (M, 2) with [magnitude, angle]
        
    Returns:
        Tuple of (path, dist_matrix, dtw_matrix)
    """
    if len(m1) == 0 or len(m2) == 0:
        return [], np.array([]), np.array([])
    
    # Custom distance function that handles angles properly
    def custom_distance(v1, v2):
        mag_diff = abs(v1[0] - v2[0])
        avg_mag = (v1[0] + v2[0]) / 2.0
        if avg_mag > 1e-8:
            mag_dist = mag_diff / avg_mag
        else:
            mag_dist = 0.0
        angle_diff = abs(v1[1] - v2[1])
        angle_dist = min(angle_diff, 2 * np.pi - angle_diff) / np.pi
        return 0.7 * mag_dist + 0.3 * angle_dist
    
    # Build custom distance matrix
    dist_matrix = np.zeros((len(m1), len(m2)))
    for i in range(len(m1)):
        for j in range(len(m2)):
            dist_matrix[i, j] = custom_distance(m1[i], m2[j])
    
    # DTW algorithm
    dtw = np.full_like(dist_matrix, np.inf)
    dtw[0, 0] = dist_matrix[0, 0]

    for i in range(len(m1)):
        for j in range(len(m2)):
            cost = dist_matrix[i, j]
            if i == 0 and j == 0:
                continue
            elif i == 0:
                dtw[i, j] = cost + dtw[i, j - 1]
            elif j == 0:
                dtw[i, j] = cost + dtw[i - 1, j]
            else:
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    # Backtrack DTW path
    i, j = len(m1) - 1, len(m2) - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = np.argmin([dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1]])
            if step == 0:
                i -= 1
            elif step == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))
    path.reverse()
    
    return path, dist_matrix, dtw


def compute_path_straightness(path: List[Tuple[int, int]]) -> float:
    """
    Measure how straight the DTW path is compared to ideal diagonal (from notebook).
    
    Returns a score between 0 (curved) and 1 (perfectly straight).
    
    Args:
        path: List of (i, j) tuples representing DTW path
        
    Returns:
        Straightness score in [0, 1]
    """
    if len(path) < 2:
        return 0.0
    
    path = np.array(path)
    x_coords = path[:, 0].astype(float)
    y_coords = path[:, 1].astype(float)
    
    # Normalize coordinates to [0, 1] range for fair comparison
    if x_coords.max() > x_coords.min():
        x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())
    else:
        x_norm = np.zeros_like(x_coords)
    
    if y_coords.max() > y_coords.min():
        y_norm = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())
    else:
        y_norm = np.zeros_like(y_coords)
    
    # Ideal diagonal: y = x (after normalization)
    # Measure deviation from diagonal
    ideal_y = x_norm
    deviation = np.abs(y_norm - ideal_y)
    mean_deviation = np.mean(deviation)
    
    # Also measure the path length ratio: straight path should have length ≈ diagonal length
    # Calculate actual path length (sum of Euclidean distances between consecutive points)
    path_length = 0.0
    for i in range(len(path) - 1):
        dx = x_coords[i+1] - x_coords[i]
        dy = y_coords[i+1] - y_coords[i]
        path_length += np.sqrt(dx*dx + dy*dy)
    
    # Diagonal length (from start to end)
    diagonal_length = np.sqrt((x_coords[-1] - x_coords[0])**2 + (y_coords[-1] - y_coords[0])**2)
    
    # Path length ratio: 1.0 means perfectly straight, >1.0 means curved
    if diagonal_length > 1e-8:
        length_ratio = path_length / diagonal_length
        # Convert to similarity: ratio of 1.0 -> 1.0, ratio of 2.0 -> ~0.5, ratio of 3.0 -> ~0.33
        length_similarity = 1.0 / length_ratio
    else:
        length_similarity = 1.0
    
    # Convert deviation to similarity: 0 deviation = 1.0, max deviation = 0.0
    # Use exponential decay with higher sensitivity for curves
    # k=5 gives better sensitivity: deviation of 0.1 -> similarity ~0.61, deviation of 0.2 -> similarity ~0.37
    deviation_similarity = np.exp(-5.0 * mean_deviation)
    
    # Also consider correlation: high correlation means linear relationship
    if len(x_coords) > 1 and np.std(x_coords) > 1e-8 and np.std(y_coords) > 1e-8:
        correlation = np.corrcoef(x_coords, y_coords)[0, 1]
        correlation_score = max(0, correlation)
    else:
        correlation_score = 1.0 if np.allclose(x_coords, y_coords) else 0.0
    
    # Combine: deviation similarity (50%), length similarity (30%), and correlation (20%)
    final_score = 0.5 * deviation_similarity + 0.3 * length_similarity + 0.2 * correlation_score
    
    return float(final_score)


def compute_motion_similarity(m1: np.ndarray, m2: np.ndarray) -> float:
    """
    Compute similarity based on pattern alignment of motion magnitudes (from notebook).
    
    Uses DTW path straightness to measure how well the patterns align visually.
    
    Args:
        m1: Motion features array 1, shape (N, 2) with [magnitude, angle]
        m2: Motion features array 2, shape (M, 2) with [magnitude, angle]
        
    Returns:
        Similarity score between 0 and 1 (higher is more similar)
    """
    min_len = min(len(m1), len(m2))
    if min_len == 0:
        return 0.0
    
    # Extract magnitude and angle components
    mag1 = m1[:, 0]
    mag2 = m2[:, 0]
    angle1 = m1[:, 1]
    angle2 = m2[:, 1]
    
    # Normalize magnitudes to [0, 1] range for pattern comparison
    mag1_norm = (mag1 - mag1.min()) / (mag1.max() - mag1.min() + 1e-8)
    mag2_norm = (mag2 - mag2.min()) / (mag2.max() - mag2.min() + 1e-8)
    
    # Get DTW path for magnitude patterns
    path, dist_matrix = get_magnitude_dtw_path(mag1, mag2)
    
    if len(path) == 0:
        return 0.0
    
    # Measure path straightness - straighter path means better pattern alignment
    path_straightness = compute_path_straightness(path)
    
    # Also measure correlation of aligned patterns
    # Extract aligned values along the DTW path
    aligned_mag1 = [mag1_norm[p[0]] for p in path]
    aligned_mag2 = [mag2_norm[p[1]] for p in path]
    
    # Compute correlation of aligned patterns
    if len(aligned_mag1) > 1 and np.std(aligned_mag1) > 1e-8 and np.std(aligned_mag2) > 1e-8:
        pattern_correlation = np.corrcoef(aligned_mag1, aligned_mag2)[0, 1]
        pattern_correlation = max(0, pattern_correlation)  # Ensure non-negative
    else:
        pattern_correlation = 1.0 if np.allclose(aligned_mag1, aligned_mag2) else 0.0
    
    # Measure average distance along the path (lower is better)
    path_distances = [dist_matrix[p[0], p[1]] for p in path]
    avg_path_distance = np.mean(path_distances)
    distance_similarity = np.exp(-3.0 * avg_path_distance)  # Convert distance to similarity
    
    # Handle circular angle distance
    min_angle_len = min(len(angle1), len(angle2))
    angle_diff = np.abs(angle1[:min_angle_len] - angle2[:min_angle_len])
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    angle_similarity = np.mean(np.cos(angle_diff))
    
    # Combine metrics:
    # - Path straightness (40%): measures temporal alignment of patterns
    # - Pattern correlation (30%): measures shape similarity of aligned patterns
    # - Distance similarity (20%): measures how close aligned values are
    # - Angle similarity (10%): secondary factor
    similarity_score = (0.4 * path_straightness + 
                        0.3 * pattern_correlation + 
                        0.2 * distance_similarity + 
                        0.1 * angle_similarity)
    
    return float(similarity_score)


def compute_dtw_similarity(m1: np.ndarray, m2: np.ndarray) -> float:
    """
    Compute similarity based on DTW path straightness (from notebook).
    
    A straighter path indicates higher temporal alignment and similarity.
    Path straightness is the primary and dominant metric.
    
    Args:
        m1: Motion features array 1, shape (N, 2) with [magnitude, angle]
        m2: Motion features array 2, shape (M, 2) with [magnitude, angle]
        
    Returns:
        DTW similarity score between 0 and 1 (higher is more similar)
    """
    if len(m1) == 0 or len(m2) == 0:
        return 0.0
    
    # Get DTW path
    path, dist_matrix, dtw = get_dtw_path(m1, m2)
    
    if len(path) == 0:
        return 0.0
    
    # Measure path straightness as the PRIMARY similarity metric
    # A straight path means sequences are temporally aligned (similar)
    # A curved path means sequences need warping (less similar)
    path_straightness = compute_path_straightness(path)
    
    # Path straightness is the dominant factor (95% weight)
    # We only use a small component (5%) of distance similarity as a tie-breaker
    # This ensures that curved paths get penalized regardless of local match quality
    path_distances = [dist_matrix[p[0], p[1]] for p in path]
    avg_path_distance = np.mean(path_distances)
    distance_similarity = np.exp(-2.0 * avg_path_distance)
    
    # Path straightness dominates: 95% weight
    # Distance similarity is only a minor tie-breaker: 5% weight
    final_similarity = 0.95 * path_straightness + 0.05 * distance_similarity
    
    return float(final_similarity)


def check_clip_quality(
    clip_pairs: List[Tuple[Path, Path]],
    motion_similarity_threshold: float = 0.8,
    dtw_similarity_threshold: float = 0.9,
    yolo_detector: Optional[YOLOPersonDetector] = None,
    multi_person_handler: Optional[MultiPersonHandler] = None,
    temp_segmented_dir: Optional[Path] = None,
    scores_csv_path: Optional[Path] = None
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    Check quality of clip pairs using motion similarity and DTW similarity (from notebook).
    
    For multi-person videos, segments the center person before computing motion features.
    Both scores must pass their thresholds for a clip pair to be qualified.
    
    Args:
        clip_pairs: List of (clip1_path, clip2_path) tuples
        motion_similarity_threshold: Minimum motion similarity score (default: 0.8)
        dtw_similarity_threshold: Minimum DTW-based similarity score (default: 0.9)
        yolo_detector: YOLO person detector instance (will create if None)
        multi_person_handler: Multi-person handler instance (will create if None)
        temp_segmented_dir: Directory for temporary segmented videos (will use tempfile if None)
        scores_csv_path: Path to CSV file for saving all scores (None = don't save)
        
    Returns:
        (qualified_clips, unqualified_clips) tuples
    """
    qualified_clips = []
    unqualified_clips = []
    
    # Prepare CSV file for score recording
    csv_writer = None
    csv_file = None
    if scores_csv_path:
        scores_csv_path = Path(scores_csv_path)
        scores_csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(scores_csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow([
            'clip_a_name',
            'clip_b_name',
            'motion_similarity_score',
            'dtw_score',
            'average_score',
            'sum_score'
        ])
        csv_file.flush()  # Flush header immediately
        logger.info(f"Saving quality scores to: {scores_csv_path}")
    
    # Initialize detectors if not provided
    if yolo_detector is None:
        yolo_detector = YOLOPersonDetector()
    if multi_person_handler is None:
        multi_person_handler = MultiPersonHandler(yolo_detector=yolo_detector)
    
    # Create temporary directory for segmented videos if needed
    if temp_segmented_dir is None:
        temp_segmented_dir = Path(tempfile.mkdtemp(prefix="segmented_clips_"))
        cleanup_temp_dir = True
    else:
        temp_segmented_dir = Path(temp_segmented_dir)
        temp_segmented_dir.mkdir(parents=True, exist_ok=True)
        cleanup_temp_dir = False
    
    try:
        for clip1_path, clip2_path in clip_pairs:
            # Ensure paths are absolute
            clip1_path = Path(clip1_path).resolve()
            clip2_path = Path(clip2_path).resolve()
            
            logger.info(f"Checking quality: {clip1_path.name} <-> {clip2_path.name}")
            logger.debug(f"  Clip1 path: {clip1_path}")
            logger.debug(f"  Clip2 path: {clip2_path}")
            
            # Check if files exist
            if not clip1_path.exists():
                logger.error(f"Clip file does not exist: {clip1_path}")
                unqualified_clips.append((clip1_path, clip2_path))
                continue
            
            if not clip2_path.exists():
                logger.error(f"Clip file does not exist: {clip2_path}")
                unqualified_clips.append((clip1_path, clip2_path))
                continue
            
            # Check file sizes (corrupted files might be very small)
            size1 = clip1_path.stat().st_size
            size2 = clip2_path.stat().st_size
            if size1 < 1000 or size2 < 1000:  # Less than 1KB is suspicious
                logger.error(f"Clip file is suspiciously small: {clip1_path if size1 < 1000 else clip2_path}")
                unqualified_clips.append((clip1_path, clip2_path))
                continue
            
            logger.debug(f"  Clip1 size: {size1} bytes, Clip2 size: {size2} bytes")
            
            # Verify files are valid videos before processing
            try:
                info1 = get_video_info(str(clip1_path))
                info2 = get_video_info(str(clip2_path))
                logger.debug(f"  Clip1 info: {info1['frame_count']} frames, {info1['fps']:.2f} fps")
                logger.debug(f"  Clip2 info: {info2['frame_count']} frames, {info2['fps']:.2f} fps")
            except Exception as e:
                logger.error(f"Cannot read video info: {e}")
                unqualified_clips.append((clip1_path, clip2_path))
                continue
            
            clip1_segmented = None
            clip2_segmented = None
            
            try:
                # Check if clips have multiple people
                has_multi1, conf1 = yolo_detector.detect_multiple_people(str(clip1_path))
                has_multi2, conf2 = yolo_detector.detect_multiple_people(str(clip2_path))
                
                # Determine which clips need segmentation
                if has_multi1:
                    logger.info(f"  Clip1 has multiple people (confidence: {conf1:.2f}), segmenting center person...")
                    segmented_path = temp_segmented_dir / f"{clip1_path.stem}_segmented{clip1_path.suffix}"
                    clip1_segmented = multi_person_handler.process_video(
                        str(clip1_path),
                        str(segmented_path)
                    )
                    logger.info(f"  Created segmented version: {clip1_segmented}")
                
                if has_multi2:
                    logger.info(f"  Clip2 has multiple people (confidence: {conf2:.2f}), segmenting center person...")
                    segmented_path = temp_segmented_dir / f"{clip2_path.stem}_segmented{clip2_path.suffix}"
                    clip2_segmented = multi_person_handler.process_video(
                        str(clip2_path),
                        str(segmented_path)
                    )
                    logger.info(f"  Created segmented version: {clip2_segmented}")
                
                # Use segmented videos for motion feature extraction if available, otherwise use originals
                video1_for_features = clip1_segmented if clip1_segmented else str(clip1_path)
                video2_for_features = clip2_segmented if clip2_segmented else str(clip2_path)
                
                # Extract motion features using optical flow (from notebook)
                # Note: We use segmented videos for feature extraction, but original clips are preserved
                motion1 = extract_motion_features_fast(video1_for_features)
                motion2 = extract_motion_features_fast(video2_for_features)
                
                # Compute both similarity scores (from notebook)
                motion_similarity = compute_motion_similarity(motion1, motion2)
                dtw_similarity = compute_dtw_similarity(motion1, motion2)
                
                # Calculate average and sum scores
                average_score = (motion_similarity + dtw_similarity) / 2.0
                sum_score = motion_similarity + dtw_similarity
                
                # Record scores to CSV (for all clips, qualified or not)
                if csv_writer:
                    csv_writer.writerow([
                        clip1_path.name,
                        clip2_path.name,
                        f"{motion_similarity:.6f}",
                        f"{dtw_similarity:.6f}",
                        f"{average_score:.6f}",
                        f"{sum_score:.6f}"
                    ])
                    # Flush immediately so CSV is updated in real-time
                    csv_file.flush()
                
                # Log detailed statistics
                logger.info(f"  Statistics for {clip1_path.name} <-> {clip2_path.name}:")
                logger.info(f"    Motion similarity score: {motion_similarity:.3f} (threshold: {motion_similarity_threshold:.2f})")
                logger.info(f"    DTW similarity score: {dtw_similarity:.3f} (threshold: {dtw_similarity_threshold:.2f})")
                
                # Both scores must pass their thresholds
                motion_pass = motion_similarity >= motion_similarity_threshold
                dtw_pass = dtw_similarity >= dtw_similarity_threshold
                
                if motion_pass and dtw_pass:
                    logger.info(f"  ✓ QUALIFIED (motion {motion_similarity:.3f} >= {motion_similarity_threshold:.2f} AND dtw {dtw_similarity:.3f} >= {dtw_similarity_threshold:.2f})")
                    qualified_clips.append((clip1_path, clip2_path))
                else:
                    fail_reasons = []
                    if not motion_pass:
                        fail_reasons.append(f"motion {motion_similarity:.3f} < {motion_similarity_threshold:.2f}")
                    if not dtw_pass:
                        fail_reasons.append(f"dtw {dtw_similarity:.3f} < {dtw_similarity_threshold:.2f}")
                    logger.info(f"  ✗ UNQUALIFIED ({' AND '.join(fail_reasons)})")
                    unqualified_clips.append((clip1_path, clip2_path))
            
            except Exception as e:
                logger.error(f"Error checking quality of {clip1_path.name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Record error case in CSV with NaN values
                if csv_writer:
                    csv_writer.writerow([
                        clip1_path.name,
                        clip2_path.name,
                        'NaN',
                        'NaN',
                        'NaN',
                        'NaN'
                    ])
                    csv_file.flush()  # Flush immediately
                unqualified_clips.append((clip1_path, clip2_path))
            finally:
                # Clean up temporary segmented videos after processing each pair
                if clip1_segmented and Path(clip1_segmented).exists():
                    try:
                        Path(clip1_segmented).unlink()
                    except Exception as e:
                        logger.debug(f"Failed to delete temporary segmented video {clip1_segmented}: {e}")
                if clip2_segmented and Path(clip2_segmented).exists():
                    try:
                        Path(clip2_segmented).unlink()
                    except Exception as e:
                        logger.debug(f"Failed to delete temporary segmented video {clip2_segmented}: {e}")
    
    finally:
        # Clean up temporary directory if we created it
        if cleanup_temp_dir and temp_segmented_dir.exists():
            try:
                shutil.rmtree(temp_segmented_dir)
            except Exception as e:
                logger.debug(f"Failed to delete temporary directory {temp_segmented_dir}: {e}")
        
        # Close CSV file if opened
        if csv_file:
            csv_file.close()
            logger.info(f"Quality scores saved to: {scores_csv_path}")
    
    # Log summary statistics
    logger.info("=" * 70)
    logger.info("Quality Check Summary")
    logger.info("=" * 70)
    logger.info(f"Total clip pairs processed: {len(clip_pairs)}")
    logger.info(f"Qualified: {len(qualified_clips)} ({len(qualified_clips)/len(clip_pairs)*100:.1f}%)")
    logger.info(f"Unqualified: {len(unqualified_clips)} ({len(unqualified_clips)/len(clip_pairs)*100:.1f}%)")
    logger.info(f"Pass condition: motion_similarity >= {motion_similarity_threshold:.2f} AND dtw_similarity >= {dtw_similarity_threshold:.2f}")
    logger.info("=" * 70)
    
    return qualified_clips, unqualified_clips


def move_unqualified_clips(
    unqualified_clips: List[Tuple[Path, Path]],
    output_folder: Path
):
    """
    Move unqualified clips to an 'unqualified' subdirectory.
    
    Args:
        unqualified_clips: List of (clip1_path, clip2_path) tuples
        output_folder: Output directory where clips are located
    """
    unqualified_dir = Path(output_folder).resolve() / "unqualified"
    unqualified_dir.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    for clip1_path, clip2_path in unqualified_clips:
        # Ensure paths are absolute
        clip1_path = Path(clip1_path).resolve()
        clip2_path = Path(clip2_path).resolve()
        
        try:
            # Only move if files exist
            if not clip1_path.exists() or not clip2_path.exists():
                logger.warning(f"Clip file does not exist, skipping move: {clip1_path if not clip1_path.exists() else clip2_path}")
                continue
            
            # Move both clips
            dest1 = unqualified_dir / clip1_path.name
            dest2 = unqualified_dir / clip2_path.name
            
            shutil.move(str(clip1_path), str(dest1))
            shutil.move(str(clip2_path), str(dest2))
            
            moved_count += 1
            logger.debug(f"Moved unqualified clips to unqualified/: {clip1_path.name}, {clip2_path.name}")
        except Exception as e:
            logger.error(f"Failed to move unqualified clips {clip1_path.name}: {e}")
    
    logger.info(f"Successfully moved {moved_count} clip pairs to unqualified/ directory")


def screen_clips(
    clip_pairs: List[Tuple[Path, Path]],
    motion_similarity_threshold: float = 0.8,
    dtw_similarity_threshold: float = 0.9,
    output_folder: Path = None,
    yolo_detector: Optional[YOLOPersonDetector] = None,
    multi_person_handler: Optional[MultiPersonHandler] = None,
    temp_segmented_dir: Optional[Path] = None,
    scores_csv_path: Optional[Path] = None
) -> List[Tuple[Path, Path]]:
    """
    Screen clip pairs for quality using motion similarity and DTW similarity (from notebook).
    
    For multi-person videos, segments the center person before computing motion features.
    Both scores must pass their thresholds for a clip pair to be qualified.
    
    Args:
        clip_pairs: List of (clip1_path, clip2_path) tuples
        motion_similarity_threshold: Minimum motion similarity score (default: 0.8)
        dtw_similarity_threshold: Minimum DTW-based similarity score (default: 0.9)
        output_folder: Output directory (for moving unqualified clips)
        yolo_detector: YOLO person detector instance (will create if None)
        multi_person_handler: Multi-person handler instance (will create if None)
        temp_segmented_dir: Directory for temporary segmented videos (will use tempfile if None)
        scores_csv_path: Path to CSV file for saving all scores (None = don't save)
        
    Returns:
        List of qualified clip pairs
    """
    qualified_clips, unqualified_clips = check_clip_quality(
        clip_pairs, motion_similarity_threshold, dtw_similarity_threshold,
        yolo_detector=yolo_detector,
        multi_person_handler=multi_person_handler,
        temp_segmented_dir=temp_segmented_dir,
        scores_csv_path=scores_csv_path
    )
    
    if unqualified_clips and output_folder:
        move_unqualified_clips(unqualified_clips, output_folder)
    
    return qualified_clips
