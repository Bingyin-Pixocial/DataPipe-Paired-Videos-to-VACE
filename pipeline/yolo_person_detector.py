"""
YOLO + ByteTrack-based multi-person detection.

Uses YOLOv8 with ByteTrack tracking to detect and count people in videos.
Fast and accurate multi-person detection.
"""

import os
import subprocess
import tempfile
import numpy as np
from typing import Tuple, List
from pathlib import Path
from tqdm import tqdm
import logging

from .utils import get_video_info

logger = logging.getLogger(__name__)


class YOLOPersonDetector:
    """
    Detects multiple people in videos using YOLOv8 + ByteTrack.
    
    Process:
    1. Extract frames at specified fps using ffmpeg
    2. Run YOLO tracking with ByteTrack on frames
    3. Parse tracking results to count unique person IDs
    4. Determine if video has multiple people based on track counts
    """
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        frame_rate: int = 10,
        confidence_threshold: float = 0.25,
        tracker: str = "bytetrack.yaml"
    ):
        """
        Args:
            model_name: YOLOv8 model name (e.g., "yolov8n.pt", "yolov8s.pt", "yolov8m.pt")
            frame_rate: FPS for frame extraction (default: 10)
            confidence_threshold: Detection confidence threshold (default: 0.25)
            tracker: Tracker config file (default: "bytetrack.yaml")
        """
        self.model_name = model_name
        self.frame_rate = frame_rate
        self.confidence_threshold = confidence_threshold
        self.tracker = tracker
        
        # Ensure ultralytics is available
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package not found. Install it with: pip install ultralytics"
            )
        
        # Load YOLO model (will auto-download if not present)
        self._model = None
    
    def _load_model(self):
        """Lazy load YOLO model."""
        if self._model is not None:
            return
        
        logger.info(f"Loading YOLO model: {self.model_name}")
        
        # Check if model_name is a path to an existing file
        model_path = Path(self.model_name)
        if model_path.exists() and model_path.is_file():
            logger.info(f"Found YOLO model at local path: {model_path}")
            model_to_load = str(model_path.resolve())
        else:
            # Check common locations for YOLO models
            # 1. Current directory
            current_dir_model = Path.cwd() / self.model_name
            # 2. Project root (assuming we're in pipeline/ subdirectory)
            project_root_model = Path(__file__).parent.parent / self.model_name
            # 3. Check if it's already in ultralytics cache (ultralytics will handle this)
            
            if current_dir_model.exists() and current_dir_model.is_file():
                logger.info(f"Found YOLO model in current directory: {current_dir_model}")
                model_to_load = str(current_dir_model.resolve())
            elif project_root_model.exists() and project_root_model.is_file():
                logger.info(f"Found YOLO model in project root: {project_root_model}")
                model_to_load = str(project_root_model.resolve())
            else:
                # Let ultralytics handle download if not found locally
                logger.info(f"YOLO model not found locally, will download if needed: {self.model_name}")
                model_to_load = self.model_name
        
        try:
            self._model = self.YOLO(model_to_load)
            logger.info(f"YOLO model loaded successfully from: {model_to_load}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Failed to load YOLO model {self.model_name}: {e}")
    
    def detect_multiple_people(self, video_path: str) -> Tuple[bool, float]:
        """
        Detect if a video contains multiple people using YOLO + ByteTrack.
        
        Args:
            video_path: Path to video file
            
        Returns:
            (has_multiple_people, confidence) tuple
            - has_multiple_people: True if video contains 2+ people
            - confidence: Average number of unique tracks per frame (normalized)
        """
        # Use batch processing for efficiency
        results = self.detect_multiple_people_batch([video_path])
        return results[0] if results else (False, 0.0)
    
    def detect_multiple_people_batch(
        self, 
        video_paths: List[str]
    ) -> List[Tuple[bool, float]]:
        """
        Detect multiple people in multiple videos using YOLO + ByteTrack.
        
        Args:
            video_paths: List of video file paths
            
        Returns:
            List of (has_multiple_people, confidence) tuples, one per video
        """
        self._load_model()
        
        results = []
        
        for video_path in tqdm(video_paths, desc="YOLO multi-person detection"):
            try:
                has_multi, confidence = self._detect_single_video(video_path)
                results.append((has_multi, confidence))
            except Exception as e:
                logger.warning(f"Error detecting people in {Path(video_path).name}: {e}")
                results.append((False, 0.0))
        
        return results
    
    def _detect_single_video(self, video_path: str) -> Tuple[bool, float]:
        """
        Detect multiple people in a single video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            (has_multiple_people, confidence) tuple
        """
        video_path = Path(video_path)
        logger.debug(f"Detecting people in: {video_path.name}")
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory(prefix="yolo_frames_") as temp_dir:
            frames_dir = Path(temp_dir) / "frames"
            frames_dir.mkdir(exist_ok=True)
            
            # Step 1: Extract frames at specified FPS
            logger.debug(f"Extracting frames at {self.frame_rate} fps...")
            frame_pattern = str(frames_dir / "%04d.png")
            
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-i", str(video_path),
                        "-vf", f"fps={self.frame_rate}",
                        "-y",  # Overwrite output files
                        frame_pattern
                    ],
                    check=True,
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg frame extraction failed: {e.stderr}")
                raise RuntimeError(f"Failed to extract frames from {video_path}")
            
            # Count extracted frames
            frame_files = sorted(frames_dir.glob("*.png"))
            if len(frame_files) == 0:
                logger.warning(f"No frames extracted from {video_path.name}")
                return (False, 0.0)
            
            logger.debug(f"Extracted {len(frame_files)} frames")
            
            # Step 2: Run YOLO tracking with ByteTrack
            logger.debug("Running YOLO tracking with ByteTrack...")
            
            try:
                # Run tracking on frames directory with streaming to avoid OOM
                # Class 0 = person class in COCO dataset
                # Use stream=True to process frames incrementally and avoid memory accumulation
                logger.debug("Running YOLO tracking with streaming mode...")
                tracking_results = self._model.track(
                    source=str(frames_dir),
                    tracker=self.tracker,
                    classes=[0],  # Only detect people (class 0)
                    conf=self.confidence_threshold,
                    save=True,
                    save_txt=True,  # Save tracking results as text files
                    stream=True,  # Use streaming mode to avoid OOM for large sources
                    verbose=False
                )
            except Exception as e:
                logger.error(f"YOLO tracking failed: {e}")
                raise RuntimeError(f"YOLO tracking failed for {video_path}: {e}")
            
            # Step 3: Parse tracking results incrementally (streaming mode)
            track_ids_per_frame = []
            
            logger.debug("Parsing tracking results incrementally from YOLO output...")
            try:
                import torch
                for result in tracking_results:
                    if result.boxes is not None and result.boxes.id is not None:
                        # Get unique track IDs for this frame
                        track_ids_array = result.boxes.id.cpu().numpy().astype(int)
                        track_ids = set(track_ids_array)
                        track_ids_per_frame.append(len(track_ids))
                        # Explicitly delete to free GPU/CPU memory immediately
                        del track_ids_array, track_ids
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        # No detections in this frame
                        track_ids_per_frame.append(0)
            except Exception as e:
                logger.warning(f"Error parsing streaming results: {e}")
                # Fallback to text file parsing
                track_ids_per_frame = []
            
            # Fallback: Try parsing from saved text files if direct parsing failed
            if not track_ids_per_frame:
                logger.debug("Trying to parse from saved tracking files...")
                # Find the output directory (usually runs/track/exp*/labels)
                output_base = Path("runs/track")
                if output_base.exists():
                    # Find the most recent experiment directory
                    exp_dirs = sorted(output_base.glob("exp*"), key=lambda x: x.stat().st_mtime, reverse=True)
                    if exp_dirs:
                        labels_dir = exp_dirs[0] / "labels"
                        if labels_dir.exists():
                            # Parse each frame's tracking file incrementally
                            frame_files = sorted(frames_dir.glob("*.png"))
                            for frame_file in frame_files:
                                # Find corresponding label file
                                label_file = labels_dir / f"{frame_file.stem}.txt"
                                if label_file.exists():
                                    track_ids = set()
                                    try:
                                        with open(label_file, 'r') as f:
                                            for line in f:
                                                parts = line.strip().split()
                                                if len(parts) >= 2:
                                                    # Format: class_id track_id x_center y_center width height confidence
                                                    # parts[1] is track_id
                                                    try:
                                                        track_id = int(float(parts[1]))
                                                        track_ids.add(track_id)
                                                    except (ValueError, IndexError):
                                                        pass
                                        track_ids_per_frame.append(len(track_ids))
                                    except Exception as e:
                                        logger.debug(f"Error reading label file {label_file}: {e}")
                                        track_ids_per_frame.append(0)
                                else:
                                    # No detections in this frame
                                    track_ids_per_frame.append(0)
            
            if not track_ids_per_frame:
                logger.warning(f"No tracking results found for {video_path.name}")
                return (False, 0.0)
            
            # Step 4: Analyze track counts
            track_counts = np.array(track_ids_per_frame, dtype=np.float32)
            
            # Calculate statistics efficiently
            avg_tracks = float(np.mean(track_counts))
            max_tracks = int(np.max(track_counts))
            frames_with_multiple = int(np.sum(track_counts >= 2))
            total_frames = len(track_counts)
            
            # Determine if video has multiple people
            # Consider it multi-person if:
            # - Average tracks >= 1.5 (accounting for tracking noise)
            # - OR more than 30% of frames have 2+ tracks
            has_multiple = avg_tracks >= 1.5 or (frames_with_multiple / total_frames) > 0.3
            
            # Confidence is based on how consistently we see multiple tracks
            if has_multiple:
                confidence = min(1.0, avg_tracks / 5.0)  # Normalize to [0, 1], assuming max ~5 people
            else:
                confidence = min(1.0, avg_tracks)  # For single person, confidence is just avg tracks
            
            logger.info(
                f"[{video_path.name}] YOLO detection: "
                f"avg_tracks={avg_tracks:.2f}, max_tracks={max_tracks}, "
                f"multi_person={has_multiple}, confidence={confidence:.2f}"
            )
            
            # Clean up intermediate data structures after all calculations
            del track_ids_per_frame, track_counts
            
            return (has_multiple, confidence)
    
    def cleanup(self):
        """Release GPU memory and clean up YOLO model resources."""
        import torch
        import gc
        try:
            if self._model is not None:
                del self._model
                self._model = None
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                torch.cuda.synchronize()
            gc.collect()
        except Exception as e:
            pass
